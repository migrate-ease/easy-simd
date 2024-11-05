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
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 *   2020      Hidayat Khan <huk2209@gmail.com>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN div

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/div.h>

static int
test_easysimd_mm_mask_div_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   139.25), EASYSIMD_FLOAT32_C(    46.67), EASYSIMD_FLOAT32_C(   568.53), EASYSIMD_FLOAT32_C(  -734.32) },
      UINT8_C(215),
      { EASYSIMD_FLOAT32_C(   215.81), EASYSIMD_FLOAT32_C(  -677.17), EASYSIMD_FLOAT32_C(  -253.56), EASYSIMD_FLOAT32_C(  -997.32) },
      { EASYSIMD_FLOAT32_C(   -87.77), EASYSIMD_FLOAT32_C(   746.64), EASYSIMD_FLOAT32_C(  -337.69), EASYSIMD_FLOAT32_C(   775.78) },
      { EASYSIMD_FLOAT32_C(    -2.46), EASYSIMD_FLOAT32_C(    -0.91), EASYSIMD_FLOAT32_C(     0.75), EASYSIMD_FLOAT32_C(  -734.32) } },
    { { EASYSIMD_FLOAT32_C(   836.58), EASYSIMD_FLOAT32_C(  -441.96), EASYSIMD_FLOAT32_C(   -56.85), EASYSIMD_FLOAT32_C(  -260.13) },
      UINT8_C(151),
      { EASYSIMD_FLOAT32_C(  -356.43), EASYSIMD_FLOAT32_C(    15.17), EASYSIMD_FLOAT32_C(   866.08), EASYSIMD_FLOAT32_C(   727.42) },
      { EASYSIMD_FLOAT32_C(   -55.42), EASYSIMD_FLOAT32_C(   -87.34), EASYSIMD_FLOAT32_C(  -262.26), EASYSIMD_FLOAT32_C(  -710.35) },
      { EASYSIMD_FLOAT32_C(     6.43), EASYSIMD_FLOAT32_C(    -0.17), EASYSIMD_FLOAT32_C(    -3.30), EASYSIMD_FLOAT32_C(  -260.13) } },
    { { EASYSIMD_FLOAT32_C(   671.47), EASYSIMD_FLOAT32_C(  -543.17), EASYSIMD_FLOAT32_C(  -931.41), EASYSIMD_FLOAT32_C(  -858.06) },
      UINT8_C(223),
      { EASYSIMD_FLOAT32_C(   207.84), EASYSIMD_FLOAT32_C(   188.61), EASYSIMD_FLOAT32_C(   660.78), EASYSIMD_FLOAT32_C(   473.52) },
      { EASYSIMD_FLOAT32_C(   373.96), EASYSIMD_FLOAT32_C(  -123.41), EASYSIMD_FLOAT32_C(   796.36), EASYSIMD_FLOAT32_C(  -879.60) },
      { EASYSIMD_FLOAT32_C(     0.56), EASYSIMD_FLOAT32_C(    -1.53), EASYSIMD_FLOAT32_C(     0.83), EASYSIMD_FLOAT32_C(    -0.54) } },
    { { EASYSIMD_FLOAT32_C(  -120.73), EASYSIMD_FLOAT32_C(  -291.41), EASYSIMD_FLOAT32_C(   867.04), EASYSIMD_FLOAT32_C(   541.58) },
      UINT8_C( 76),
      { EASYSIMD_FLOAT32_C(   703.62), EASYSIMD_FLOAT32_C(  -900.37), EASYSIMD_FLOAT32_C(   427.52), EASYSIMD_FLOAT32_C(  -556.51) },
      { EASYSIMD_FLOAT32_C(   802.96), EASYSIMD_FLOAT32_C(  -928.91), EASYSIMD_FLOAT32_C(   458.67), EASYSIMD_FLOAT32_C(   669.03) },
      { EASYSIMD_FLOAT32_C(  -120.73), EASYSIMD_FLOAT32_C(  -291.41), EASYSIMD_FLOAT32_C(     0.93), EASYSIMD_FLOAT32_C(    -0.83) } },
    { { EASYSIMD_FLOAT32_C(   798.50), EASYSIMD_FLOAT32_C(  -596.76), EASYSIMD_FLOAT32_C(  -418.31), EASYSIMD_FLOAT32_C(  -463.75) },
      UINT8_C( 30),
      { EASYSIMD_FLOAT32_C(  -746.84), EASYSIMD_FLOAT32_C(    -6.92), EASYSIMD_FLOAT32_C(  -238.51), EASYSIMD_FLOAT32_C(  -604.90) },
      { EASYSIMD_FLOAT32_C(    85.33), EASYSIMD_FLOAT32_C(   969.33), EASYSIMD_FLOAT32_C(   583.71), EASYSIMD_FLOAT32_C(  -253.89) },
      { EASYSIMD_FLOAT32_C(   798.50), EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(    -0.41), EASYSIMD_FLOAT32_C(     2.38) } },
    { { EASYSIMD_FLOAT32_C(   442.85), EASYSIMD_FLOAT32_C(   -42.33), EASYSIMD_FLOAT32_C(   622.71), EASYSIMD_FLOAT32_C(   239.21) },
      UINT8_C( 36),
      { EASYSIMD_FLOAT32_C(  -498.02), EASYSIMD_FLOAT32_C(   947.80), EASYSIMD_FLOAT32_C(   -54.89), EASYSIMD_FLOAT32_C(  -956.44) },
      { EASYSIMD_FLOAT32_C(  -567.83), EASYSIMD_FLOAT32_C(  -351.26), EASYSIMD_FLOAT32_C(  -856.81), EASYSIMD_FLOAT32_C(   859.69) },
      { EASYSIMD_FLOAT32_C(   442.85), EASYSIMD_FLOAT32_C(   -42.33), EASYSIMD_FLOAT32_C(     0.06), EASYSIMD_FLOAT32_C(   239.21) } },
    { { EASYSIMD_FLOAT32_C(    92.23), EASYSIMD_FLOAT32_C(   946.14), EASYSIMD_FLOAT32_C(   930.77), EASYSIMD_FLOAT32_C(  -449.11) },
      UINT8_C( 14),
      { EASYSIMD_FLOAT32_C(   729.28), EASYSIMD_FLOAT32_C(   -45.86), EASYSIMD_FLOAT32_C(  -803.13), EASYSIMD_FLOAT32_C(  -734.48) },
      { EASYSIMD_FLOAT32_C(   647.03), EASYSIMD_FLOAT32_C(  -549.97), EASYSIMD_FLOAT32_C(   258.60), EASYSIMD_FLOAT32_C(  -591.48) },
      { EASYSIMD_FLOAT32_C(    92.23), EASYSIMD_FLOAT32_C(     0.08), EASYSIMD_FLOAT32_C(    -3.11), EASYSIMD_FLOAT32_C(     1.24) } },
    { { EASYSIMD_FLOAT32_C(  -154.87), EASYSIMD_FLOAT32_C(  -656.07), EASYSIMD_FLOAT32_C(  -622.15), EASYSIMD_FLOAT32_C(  -571.16) },
      UINT8_C(228),
      { EASYSIMD_FLOAT32_C(   820.70), EASYSIMD_FLOAT32_C(   386.51), EASYSIMD_FLOAT32_C(  -287.25), EASYSIMD_FLOAT32_C(    59.91) },
      { EASYSIMD_FLOAT32_C(  -535.41), EASYSIMD_FLOAT32_C(   214.73), EASYSIMD_FLOAT32_C(     7.71), EASYSIMD_FLOAT32_C(   409.70) },
      { EASYSIMD_FLOAT32_C(  -154.87), EASYSIMD_FLOAT32_C(  -656.07), EASYSIMD_FLOAT32_C(   -37.26), EASYSIMD_FLOAT32_C(  -571.16) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_div_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_div_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_div_ps(src, k, a, b);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_div_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(126),
      { EASYSIMD_FLOAT32_C(    -1.50), EASYSIMD_FLOAT32_C(   574.61), EASYSIMD_FLOAT32_C(   533.54), EASYSIMD_FLOAT32_C(  -494.04) },
      { EASYSIMD_FLOAT32_C(  -290.82), EASYSIMD_FLOAT32_C(  -231.49), EASYSIMD_FLOAT32_C(  -260.09), EASYSIMD_FLOAT32_C(   715.70) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.48), EASYSIMD_FLOAT32_C(    -2.05), EASYSIMD_FLOAT32_C(    -0.69) } },
    { UINT8_C(203),
      { EASYSIMD_FLOAT32_C(   717.44), EASYSIMD_FLOAT32_C(   922.77), EASYSIMD_FLOAT32_C(   572.61), EASYSIMD_FLOAT32_C(  -633.20) },
      { EASYSIMD_FLOAT32_C(  -640.70), EASYSIMD_FLOAT32_C(  -263.54), EASYSIMD_FLOAT32_C(   853.83), EASYSIMD_FLOAT32_C(   804.19) },
      { EASYSIMD_FLOAT32_C(    -1.12), EASYSIMD_FLOAT32_C(    -3.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.79) } },
    { UINT8_C(115),
      { EASYSIMD_FLOAT32_C(  -820.47), EASYSIMD_FLOAT32_C(   -86.40), EASYSIMD_FLOAT32_C(  -954.56), EASYSIMD_FLOAT32_C(  -177.34) },
      { EASYSIMD_FLOAT32_C(  -459.47), EASYSIMD_FLOAT32_C(  -719.18), EASYSIMD_FLOAT32_C(   534.09), EASYSIMD_FLOAT32_C(   196.10) },
      { EASYSIMD_FLOAT32_C(     1.79), EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(235),
      { EASYSIMD_FLOAT32_C(  -861.07), EASYSIMD_FLOAT32_C(  -296.94), EASYSIMD_FLOAT32_C(  -986.70), EASYSIMD_FLOAT32_C(  -241.35) },
      { EASYSIMD_FLOAT32_C(   701.56), EASYSIMD_FLOAT32_C(   587.92), EASYSIMD_FLOAT32_C(  -707.82), EASYSIMD_FLOAT32_C(  -792.47) },
      { EASYSIMD_FLOAT32_C(    -1.23), EASYSIMD_FLOAT32_C(    -0.51), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.30) } },
    { UINT8_C(250),
      { EASYSIMD_FLOAT32_C(    60.69), EASYSIMD_FLOAT32_C(   -52.56), EASYSIMD_FLOAT32_C(  -987.19), EASYSIMD_FLOAT32_C(  -162.15) },
      { EASYSIMD_FLOAT32_C(  -335.12), EASYSIMD_FLOAT32_C(   935.58), EASYSIMD_FLOAT32_C(  -589.53), EASYSIMD_FLOAT32_C(    31.68) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -5.12) } },
    { UINT8_C( 30),
      { EASYSIMD_FLOAT32_C(   146.93), EASYSIMD_FLOAT32_C(  -114.49), EASYSIMD_FLOAT32_C(  -900.93), EASYSIMD_FLOAT32_C(  -595.73) },
      { EASYSIMD_FLOAT32_C(    65.04), EASYSIMD_FLOAT32_C(    12.67), EASYSIMD_FLOAT32_C(  -550.29), EASYSIMD_FLOAT32_C(   887.70) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -9.04), EASYSIMD_FLOAT32_C(     1.64), EASYSIMD_FLOAT32_C(    -0.67) } },
    { UINT8_C( 51),
      { EASYSIMD_FLOAT32_C(  -269.47), EASYSIMD_FLOAT32_C(   421.79), EASYSIMD_FLOAT32_C(  -250.70), EASYSIMD_FLOAT32_C(  -571.03) },
      { EASYSIMD_FLOAT32_C(   560.72), EASYSIMD_FLOAT32_C(   452.36), EASYSIMD_FLOAT32_C(  -557.72), EASYSIMD_FLOAT32_C(  -680.63) },
      { EASYSIMD_FLOAT32_C(    -0.48), EASYSIMD_FLOAT32_C(     0.93), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(130),
      { EASYSIMD_FLOAT32_C(  -969.81), EASYSIMD_FLOAT32_C(  -388.45), EASYSIMD_FLOAT32_C(   361.45), EASYSIMD_FLOAT32_C(  -672.71) },
      { EASYSIMD_FLOAT32_C(   672.24), EASYSIMD_FLOAT32_C(  -691.10), EASYSIMD_FLOAT32_C(  -659.90), EASYSIMD_FLOAT32_C(  -489.91) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_div_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_div_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_div_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_div_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   874.76), EASYSIMD_FLOAT64_C(   900.61) },
      UINT8_C(219),
      { EASYSIMD_FLOAT64_C(  -333.57), EASYSIMD_FLOAT64_C(   120.12) },
      { EASYSIMD_FLOAT64_C(   640.13), EASYSIMD_FLOAT64_C(   202.33) },
      { EASYSIMD_FLOAT64_C(    -0.52), EASYSIMD_FLOAT64_C(     0.59) } },
    { { EASYSIMD_FLOAT64_C(   998.46), EASYSIMD_FLOAT64_C(  -468.32) },
      UINT8_C(123),
      { EASYSIMD_FLOAT64_C(  -826.14), EASYSIMD_FLOAT64_C(  -518.12) },
      { EASYSIMD_FLOAT64_C(  -262.26), EASYSIMD_FLOAT64_C(   718.64) },
      { EASYSIMD_FLOAT64_C(     3.15), EASYSIMD_FLOAT64_C(    -0.72) } },
    { { EASYSIMD_FLOAT64_C(  -452.78), EASYSIMD_FLOAT64_C(   908.68) },
      UINT8_C(108),
      { EASYSIMD_FLOAT64_C(  -654.17), EASYSIMD_FLOAT64_C(   936.07) },
      { EASYSIMD_FLOAT64_C(  -344.43), EASYSIMD_FLOAT64_C(    56.02) },
      { EASYSIMD_FLOAT64_C(  -452.78), EASYSIMD_FLOAT64_C(   908.68) } },
    { { EASYSIMD_FLOAT64_C(  -430.11), EASYSIMD_FLOAT64_C(  -730.84) },
      UINT8_C(178),
      { EASYSIMD_FLOAT64_C(  -483.99), EASYSIMD_FLOAT64_C(   402.56) },
      { EASYSIMD_FLOAT64_C(  -161.95), EASYSIMD_FLOAT64_C(  -423.47) },
      { EASYSIMD_FLOAT64_C(  -430.11), EASYSIMD_FLOAT64_C(    -0.95) } },
    { { EASYSIMD_FLOAT64_C(  -823.54), EASYSIMD_FLOAT64_C(   734.07) },
      UINT8_C(243),
      { EASYSIMD_FLOAT64_C(  -948.78), EASYSIMD_FLOAT64_C(   634.68) },
      { EASYSIMD_FLOAT64_C(  -203.65), EASYSIMD_FLOAT64_C(  -282.35) },
      { EASYSIMD_FLOAT64_C(     4.66), EASYSIMD_FLOAT64_C(    -2.25) } },
    { { EASYSIMD_FLOAT64_C(  -245.20), EASYSIMD_FLOAT64_C(  -563.52) },
      UINT8_C( 24),
      { EASYSIMD_FLOAT64_C(  -246.74), EASYSIMD_FLOAT64_C(   -31.83) },
      { EASYSIMD_FLOAT64_C(   412.33), EASYSIMD_FLOAT64_C(   -72.88) },
      { EASYSIMD_FLOAT64_C(  -245.20), EASYSIMD_FLOAT64_C(  -563.52) } },
    { { EASYSIMD_FLOAT64_C(   450.05), EASYSIMD_FLOAT64_C(  -849.92) },
      UINT8_C(120),
      { EASYSIMD_FLOAT64_C(   997.27), EASYSIMD_FLOAT64_C(  -941.24) },
      { EASYSIMD_FLOAT64_C(   609.39), EASYSIMD_FLOAT64_C(  -656.89) },
      { EASYSIMD_FLOAT64_C(   450.05), EASYSIMD_FLOAT64_C(  -849.92) } },
    { { EASYSIMD_FLOAT64_C(   994.83), EASYSIMD_FLOAT64_C(  -735.04) },
      UINT8_C(118),
      { EASYSIMD_FLOAT64_C(  -435.28), EASYSIMD_FLOAT64_C(  -465.88) },
      { EASYSIMD_FLOAT64_C(  -419.14), EASYSIMD_FLOAT64_C(    80.73) },
      { EASYSIMD_FLOAT64_C(   994.83), EASYSIMD_FLOAT64_C(    -5.77) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_div_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_div_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_div_pd(src, k, a, b);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_div_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(140),
      { EASYSIMD_FLOAT64_C(  -333.89), EASYSIMD_FLOAT64_C(  -119.46) },
      { EASYSIMD_FLOAT64_C(   512.51), EASYSIMD_FLOAT64_C(  -654.08) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 87),
      { EASYSIMD_FLOAT64_C(   291.83), EASYSIMD_FLOAT64_C(   912.75) },
      { EASYSIMD_FLOAT64_C(  -934.32), EASYSIMD_FLOAT64_C(   589.65) },
      { EASYSIMD_FLOAT64_C(    -0.31), EASYSIMD_FLOAT64_C(     1.55) } },
    { UINT8_C(210),
      { EASYSIMD_FLOAT64_C(   -14.65), EASYSIMD_FLOAT64_C(  -608.58) },
      { EASYSIMD_FLOAT64_C(  -338.25), EASYSIMD_FLOAT64_C(  -930.31) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.65) } },
    { UINT8_C(199),
      { EASYSIMD_FLOAT64_C(  -850.92), EASYSIMD_FLOAT64_C(  -839.25) },
      { EASYSIMD_FLOAT64_C(   247.94), EASYSIMD_FLOAT64_C(  -149.94) },
      { EASYSIMD_FLOAT64_C(    -3.43), EASYSIMD_FLOAT64_C(     5.60) } },
    { UINT8_C(161),
      { EASYSIMD_FLOAT64_C(   722.16), EASYSIMD_FLOAT64_C(   781.45) },
      { EASYSIMD_FLOAT64_C(  -425.53), EASYSIMD_FLOAT64_C(   -91.81) },
      { EASYSIMD_FLOAT64_C(    -1.70), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 55),
      { EASYSIMD_FLOAT64_C(   181.47), EASYSIMD_FLOAT64_C(  -836.98) },
      { EASYSIMD_FLOAT64_C(  -214.24), EASYSIMD_FLOAT64_C(   703.37) },
      { EASYSIMD_FLOAT64_C(    -0.85), EASYSIMD_FLOAT64_C(    -1.19) } },
    { UINT8_C( 16),
      { EASYSIMD_FLOAT64_C(   714.52), EASYSIMD_FLOAT64_C(  -630.52) },
      { EASYSIMD_FLOAT64_C(   651.53), EASYSIMD_FLOAT64_C(   227.03) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(139),
      { EASYSIMD_FLOAT64_C(  -747.53), EASYSIMD_FLOAT64_C(  -481.14) },
      { EASYSIMD_FLOAT64_C(  -371.85), EASYSIMD_FLOAT64_C(  -681.85) },
      { EASYSIMD_FLOAT64_C(     2.01), EASYSIMD_FLOAT64_C(     0.71) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_div_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_div_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_div_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_div_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   870.43), EASYSIMD_FLOAT32_C(   254.54), EASYSIMD_FLOAT32_C(   658.81), EASYSIMD_FLOAT32_C(   885.10),
        EASYSIMD_FLOAT32_C(   273.30), EASYSIMD_FLOAT32_C(  -744.31), EASYSIMD_FLOAT32_C(   642.92), EASYSIMD_FLOAT32_C(   742.96) },
      UINT8_C(158),
      { EASYSIMD_FLOAT32_C(  -352.80), EASYSIMD_FLOAT32_C(   -97.46), EASYSIMD_FLOAT32_C(   333.63), EASYSIMD_FLOAT32_C(  -179.26),
        EASYSIMD_FLOAT32_C(   968.52), EASYSIMD_FLOAT32_C(  -562.64), EASYSIMD_FLOAT32_C(   847.67), EASYSIMD_FLOAT32_C(   627.14) },
      { EASYSIMD_FLOAT32_C(   460.83), EASYSIMD_FLOAT32_C(   672.97), EASYSIMD_FLOAT32_C(  -204.41), EASYSIMD_FLOAT32_C(  -738.76),
        EASYSIMD_FLOAT32_C(   598.43), EASYSIMD_FLOAT32_C(  -165.95), EASYSIMD_FLOAT32_C(   865.74), EASYSIMD_FLOAT32_C(  -154.29) },
      { EASYSIMD_FLOAT32_C(   870.43), EASYSIMD_FLOAT32_C(    -0.14), EASYSIMD_FLOAT32_C(    -1.63), EASYSIMD_FLOAT32_C(     0.24),
        EASYSIMD_FLOAT32_C(     1.62), EASYSIMD_FLOAT32_C(  -744.31), EASYSIMD_FLOAT32_C(   642.92), EASYSIMD_FLOAT32_C(    -4.06) } },
    { { EASYSIMD_FLOAT32_C(    37.08), EASYSIMD_FLOAT32_C(  -893.42), EASYSIMD_FLOAT32_C(   562.12), EASYSIMD_FLOAT32_C(  -554.45),
        EASYSIMD_FLOAT32_C(   159.46), EASYSIMD_FLOAT32_C(  -678.32), EASYSIMD_FLOAT32_C(  -684.02), EASYSIMD_FLOAT32_C(  -585.99) },
      UINT8_C(124),
      { EASYSIMD_FLOAT32_C(  -798.92), EASYSIMD_FLOAT32_C(   687.31), EASYSIMD_FLOAT32_C(  -763.82), EASYSIMD_FLOAT32_C(   844.00),
        EASYSIMD_FLOAT32_C(   430.27), EASYSIMD_FLOAT32_C(   259.49), EASYSIMD_FLOAT32_C(  -508.80), EASYSIMD_FLOAT32_C(  -667.19) },
      { EASYSIMD_FLOAT32_C(  -406.88), EASYSIMD_FLOAT32_C(   311.93), EASYSIMD_FLOAT32_C(  -698.66), EASYSIMD_FLOAT32_C(    30.49),
        EASYSIMD_FLOAT32_C(   159.60), EASYSIMD_FLOAT32_C(   928.48), EASYSIMD_FLOAT32_C(  -508.68), EASYSIMD_FLOAT32_C(  -167.43) },
      { EASYSIMD_FLOAT32_C(    37.08), EASYSIMD_FLOAT32_C(  -893.42), EASYSIMD_FLOAT32_C(     1.09), EASYSIMD_FLOAT32_C(    27.68),
        EASYSIMD_FLOAT32_C(     2.70), EASYSIMD_FLOAT32_C(     0.28), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(  -585.99) } },
    { { EASYSIMD_FLOAT32_C(  -275.94), EASYSIMD_FLOAT32_C(  -247.44), EASYSIMD_FLOAT32_C(  -569.00), EASYSIMD_FLOAT32_C(   558.11),
        EASYSIMD_FLOAT32_C(  -381.70), EASYSIMD_FLOAT32_C(   276.71), EASYSIMD_FLOAT32_C(  -404.81), EASYSIMD_FLOAT32_C(  -275.12) },
      UINT8_C(219),
      { EASYSIMD_FLOAT32_C(    40.74), EASYSIMD_FLOAT32_C(   884.35), EASYSIMD_FLOAT32_C(   160.50), EASYSIMD_FLOAT32_C(   356.73),
        EASYSIMD_FLOAT32_C(  -701.65), EASYSIMD_FLOAT32_C(   140.99), EASYSIMD_FLOAT32_C(   557.81), EASYSIMD_FLOAT32_C(   985.66) },
      { EASYSIMD_FLOAT32_C(   377.17), EASYSIMD_FLOAT32_C(   401.81), EASYSIMD_FLOAT32_C(   415.93), EASYSIMD_FLOAT32_C(  -363.34),
        EASYSIMD_FLOAT32_C(   893.01), EASYSIMD_FLOAT32_C(   748.75), EASYSIMD_FLOAT32_C(   229.78), EASYSIMD_FLOAT32_C(   204.94) },
      { EASYSIMD_FLOAT32_C(     0.11), EASYSIMD_FLOAT32_C(     2.20), EASYSIMD_FLOAT32_C(  -569.00), EASYSIMD_FLOAT32_C(    -0.98),
        EASYSIMD_FLOAT32_C(    -0.79), EASYSIMD_FLOAT32_C(   276.71), EASYSIMD_FLOAT32_C(     2.43), EASYSIMD_FLOAT32_C(     4.81) } },
    { { EASYSIMD_FLOAT32_C(  -949.92), EASYSIMD_FLOAT32_C(  -739.74), EASYSIMD_FLOAT32_C(  -635.46), EASYSIMD_FLOAT32_C(   978.56),
        EASYSIMD_FLOAT32_C(  -248.42), EASYSIMD_FLOAT32_C(   197.10), EASYSIMD_FLOAT32_C(  -297.38), EASYSIMD_FLOAT32_C(   504.14) },
      UINT8_C(251),
      { EASYSIMD_FLOAT32_C(  -739.26), EASYSIMD_FLOAT32_C(  -877.56), EASYSIMD_FLOAT32_C(   -95.19), EASYSIMD_FLOAT32_C(  -144.07),
        EASYSIMD_FLOAT32_C(  -152.67), EASYSIMD_FLOAT32_C(   743.63), EASYSIMD_FLOAT32_C(   896.67), EASYSIMD_FLOAT32_C(  -268.33) },
      { EASYSIMD_FLOAT32_C(   -95.86), EASYSIMD_FLOAT32_C(   253.40), EASYSIMD_FLOAT32_C(    30.02), EASYSIMD_FLOAT32_C(  -954.88),
        EASYSIMD_FLOAT32_C(  -188.79), EASYSIMD_FLOAT32_C(    15.68), EASYSIMD_FLOAT32_C(   422.29), EASYSIMD_FLOAT32_C(  -786.98) },
      { EASYSIMD_FLOAT32_C(     7.71), EASYSIMD_FLOAT32_C(    -3.46), EASYSIMD_FLOAT32_C(  -635.46), EASYSIMD_FLOAT32_C(     0.15),
        EASYSIMD_FLOAT32_C(     0.81), EASYSIMD_FLOAT32_C(    47.43), EASYSIMD_FLOAT32_C(     2.12), EASYSIMD_FLOAT32_C(     0.34) } },
    { { EASYSIMD_FLOAT32_C(  -568.38), EASYSIMD_FLOAT32_C(  -941.06), EASYSIMD_FLOAT32_C(  -893.98), EASYSIMD_FLOAT32_C(  -819.64),
        EASYSIMD_FLOAT32_C(   288.72), EASYSIMD_FLOAT32_C(   310.96), EASYSIMD_FLOAT32_C(  -769.55), EASYSIMD_FLOAT32_C(   548.98) },
      UINT8_C(175),
      { EASYSIMD_FLOAT32_C(  -790.99), EASYSIMD_FLOAT32_C(  -699.43), EASYSIMD_FLOAT32_C(  -127.40), EASYSIMD_FLOAT32_C(   -88.37),
        EASYSIMD_FLOAT32_C(   804.71), EASYSIMD_FLOAT32_C(  -499.29), EASYSIMD_FLOAT32_C(   172.37), EASYSIMD_FLOAT32_C(   927.15) },
      { EASYSIMD_FLOAT32_C(   405.52), EASYSIMD_FLOAT32_C(  -971.70), EASYSIMD_FLOAT32_C(  -225.52), EASYSIMD_FLOAT32_C(   149.15),
        EASYSIMD_FLOAT32_C(   924.97), EASYSIMD_FLOAT32_C(   506.15), EASYSIMD_FLOAT32_C(  -946.71), EASYSIMD_FLOAT32_C(   178.36) },
      { EASYSIMD_FLOAT32_C(    -1.95), EASYSIMD_FLOAT32_C(     0.72), EASYSIMD_FLOAT32_C(     0.56), EASYSIMD_FLOAT32_C(    -0.59),
        EASYSIMD_FLOAT32_C(   288.72), EASYSIMD_FLOAT32_C(    -0.99), EASYSIMD_FLOAT32_C(  -769.55), EASYSIMD_FLOAT32_C(     5.20) } },
    { { EASYSIMD_FLOAT32_C(  -463.83), EASYSIMD_FLOAT32_C(  -901.59), EASYSIMD_FLOAT32_C(   989.57), EASYSIMD_FLOAT32_C(   551.86),
        EASYSIMD_FLOAT32_C(   520.70), EASYSIMD_FLOAT32_C(  -797.42), EASYSIMD_FLOAT32_C(   983.47), EASYSIMD_FLOAT32_C(   579.65) },
      UINT8_C(194),
      { EASYSIMD_FLOAT32_C(  -836.16), EASYSIMD_FLOAT32_C(  -131.63), EASYSIMD_FLOAT32_C(   619.57), EASYSIMD_FLOAT32_C(  -605.71),
        EASYSIMD_FLOAT32_C(  -582.65), EASYSIMD_FLOAT32_C(   295.07), EASYSIMD_FLOAT32_C(  -396.71), EASYSIMD_FLOAT32_C(  -282.08) },
      { EASYSIMD_FLOAT32_C(  -832.33), EASYSIMD_FLOAT32_C(   514.92), EASYSIMD_FLOAT32_C(  -477.37), EASYSIMD_FLOAT32_C(  -331.62),
        EASYSIMD_FLOAT32_C(  -312.71), EASYSIMD_FLOAT32_C(  -550.22), EASYSIMD_FLOAT32_C(  -926.10), EASYSIMD_FLOAT32_C(  -284.42) },
      { EASYSIMD_FLOAT32_C(  -463.83), EASYSIMD_FLOAT32_C(    -0.26), EASYSIMD_FLOAT32_C(   989.57), EASYSIMD_FLOAT32_C(   551.86),
        EASYSIMD_FLOAT32_C(   520.70), EASYSIMD_FLOAT32_C(  -797.42), EASYSIMD_FLOAT32_C(     0.43), EASYSIMD_FLOAT32_C(     0.99) } },
    { { EASYSIMD_FLOAT32_C(   224.26), EASYSIMD_FLOAT32_C(   223.05), EASYSIMD_FLOAT32_C(  -359.45), EASYSIMD_FLOAT32_C(  -269.59),
        EASYSIMD_FLOAT32_C(   276.34), EASYSIMD_FLOAT32_C(   818.91), EASYSIMD_FLOAT32_C(   266.58), EASYSIMD_FLOAT32_C(   374.76) },
      UINT8_C(137),
      { EASYSIMD_FLOAT32_C(  -181.56), EASYSIMD_FLOAT32_C(  -104.54), EASYSIMD_FLOAT32_C(  -988.93), EASYSIMD_FLOAT32_C(  -198.09),
        EASYSIMD_FLOAT32_C(  -524.90), EASYSIMD_FLOAT32_C(  -680.32), EASYSIMD_FLOAT32_C(   -34.25), EASYSIMD_FLOAT32_C(   343.47) },
      { EASYSIMD_FLOAT32_C(   939.24), EASYSIMD_FLOAT32_C(   360.03), EASYSIMD_FLOAT32_C(   760.83), EASYSIMD_FLOAT32_C(   234.31),
        EASYSIMD_FLOAT32_C(   963.33), EASYSIMD_FLOAT32_C(  -521.26), EASYSIMD_FLOAT32_C(   401.99), EASYSIMD_FLOAT32_C(   478.25) },
      { EASYSIMD_FLOAT32_C(    -0.19), EASYSIMD_FLOAT32_C(   223.05), EASYSIMD_FLOAT32_C(  -359.45), EASYSIMD_FLOAT32_C(    -0.85),
        EASYSIMD_FLOAT32_C(   276.34), EASYSIMD_FLOAT32_C(   818.91), EASYSIMD_FLOAT32_C(   266.58), EASYSIMD_FLOAT32_C(     0.72) } },
    { { EASYSIMD_FLOAT32_C(     1.37), EASYSIMD_FLOAT32_C(  -929.63), EASYSIMD_FLOAT32_C(  -834.46), EASYSIMD_FLOAT32_C(   451.15),
        EASYSIMD_FLOAT32_C(  -855.73), EASYSIMD_FLOAT32_C(  -118.88), EASYSIMD_FLOAT32_C(  -324.59), EASYSIMD_FLOAT32_C(   367.32) },
      UINT8_C(193),
      { EASYSIMD_FLOAT32_C(   405.82), EASYSIMD_FLOAT32_C(  -356.34), EASYSIMD_FLOAT32_C(   340.59), EASYSIMD_FLOAT32_C(  -327.60),
        EASYSIMD_FLOAT32_C(  -981.58), EASYSIMD_FLOAT32_C(   149.07), EASYSIMD_FLOAT32_C(   490.84), EASYSIMD_FLOAT32_C(   -86.13) },
      { EASYSIMD_FLOAT32_C(   160.14), EASYSIMD_FLOAT32_C(  -707.25), EASYSIMD_FLOAT32_C(   388.98), EASYSIMD_FLOAT32_C(   479.81),
        EASYSIMD_FLOAT32_C(   258.50), EASYSIMD_FLOAT32_C(  -267.55), EASYSIMD_FLOAT32_C(   419.06), EASYSIMD_FLOAT32_C(  -381.47) },
      { EASYSIMD_FLOAT32_C(     2.53), EASYSIMD_FLOAT32_C(  -929.63), EASYSIMD_FLOAT32_C(  -834.46), EASYSIMD_FLOAT32_C(   451.15),
        EASYSIMD_FLOAT32_C(  -855.73), EASYSIMD_FLOAT32_C(  -118.88), EASYSIMD_FLOAT32_C(     1.17), EASYSIMD_FLOAT32_C(     0.23) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_div_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_div_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_div_ps(src, k, a, b);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_div_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(134),
      { EASYSIMD_FLOAT32_C(   -35.50), EASYSIMD_FLOAT32_C(  -833.36), EASYSIMD_FLOAT32_C(   594.08), EASYSIMD_FLOAT32_C(   921.51),
        EASYSIMD_FLOAT32_C(   244.38), EASYSIMD_FLOAT32_C(   586.17), EASYSIMD_FLOAT32_C(  -899.52), EASYSIMD_FLOAT32_C(   466.42) },
      { EASYSIMD_FLOAT32_C(   444.89), EASYSIMD_FLOAT32_C(  -431.56), EASYSIMD_FLOAT32_C(   410.26), EASYSIMD_FLOAT32_C(  -960.26),
        EASYSIMD_FLOAT32_C(   906.80), EASYSIMD_FLOAT32_C(  -929.34), EASYSIMD_FLOAT32_C(   173.44), EASYSIMD_FLOAT32_C(   954.93) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.93), EASYSIMD_FLOAT32_C(     1.45), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.49) } },
    { UINT8_C(110),
      { EASYSIMD_FLOAT32_C(  -334.39), EASYSIMD_FLOAT32_C(   653.43), EASYSIMD_FLOAT32_C(   126.40), EASYSIMD_FLOAT32_C(  -219.78),
        EASYSIMD_FLOAT32_C(  -858.33), EASYSIMD_FLOAT32_C(   915.61), EASYSIMD_FLOAT32_C(  -943.95), EASYSIMD_FLOAT32_C(   541.64) },
      { EASYSIMD_FLOAT32_C(  -206.14), EASYSIMD_FLOAT32_C(  -819.46), EASYSIMD_FLOAT32_C(   292.24), EASYSIMD_FLOAT32_C(   716.10),
        EASYSIMD_FLOAT32_C(  -243.85), EASYSIMD_FLOAT32_C(   623.25), EASYSIMD_FLOAT32_C(  -319.40), EASYSIMD_FLOAT32_C(   -77.21) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.80), EASYSIMD_FLOAT32_C(     0.43), EASYSIMD_FLOAT32_C(    -0.31),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.47), EASYSIMD_FLOAT32_C(     2.96), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(146),
      { EASYSIMD_FLOAT32_C(  -397.89), EASYSIMD_FLOAT32_C(  -832.83), EASYSIMD_FLOAT32_C(  -196.50), EASYSIMD_FLOAT32_C(  -297.41),
        EASYSIMD_FLOAT32_C(   633.58), EASYSIMD_FLOAT32_C(  -751.61), EASYSIMD_FLOAT32_C(   271.02), EASYSIMD_FLOAT32_C(    43.85) },
      { EASYSIMD_FLOAT32_C(  -711.87), EASYSIMD_FLOAT32_C(   177.82), EASYSIMD_FLOAT32_C(   114.50), EASYSIMD_FLOAT32_C(   461.57),
        EASYSIMD_FLOAT32_C(   132.75), EASYSIMD_FLOAT32_C(   -83.43), EASYSIMD_FLOAT32_C(  -872.82), EASYSIMD_FLOAT32_C(  -213.82) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -4.68), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     4.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.21) } },
    { UINT8_C(117),
      { EASYSIMD_FLOAT32_C(   -92.61), EASYSIMD_FLOAT32_C(   -72.16), EASYSIMD_FLOAT32_C(   958.58), EASYSIMD_FLOAT32_C(   -36.56),
        EASYSIMD_FLOAT32_C(  -530.52), EASYSIMD_FLOAT32_C(  -247.56), EASYSIMD_FLOAT32_C(   143.98), EASYSIMD_FLOAT32_C(   761.73) },
      { EASYSIMD_FLOAT32_C(  -531.46), EASYSIMD_FLOAT32_C(   900.12), EASYSIMD_FLOAT32_C(   384.98), EASYSIMD_FLOAT32_C(   149.15),
        EASYSIMD_FLOAT32_C(  -177.09), EASYSIMD_FLOAT32_C(  -397.70), EASYSIMD_FLOAT32_C(   751.25), EASYSIMD_FLOAT32_C(    -9.92) },
      { EASYSIMD_FLOAT32_C(     0.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.49), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(     0.62), EASYSIMD_FLOAT32_C(     0.19), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 34),
      { EASYSIMD_FLOAT32_C(  -546.16), EASYSIMD_FLOAT32_C(  -376.34), EASYSIMD_FLOAT32_C(   654.19), EASYSIMD_FLOAT32_C(   724.86),
        EASYSIMD_FLOAT32_C(   667.51), EASYSIMD_FLOAT32_C(   942.32), EASYSIMD_FLOAT32_C(   -97.31), EASYSIMD_FLOAT32_C(  -217.99) },
      { EASYSIMD_FLOAT32_C(   403.89), EASYSIMD_FLOAT32_C(  -964.56), EASYSIMD_FLOAT32_C(   698.58), EASYSIMD_FLOAT32_C(   531.07),
        EASYSIMD_FLOAT32_C(  -178.39), EASYSIMD_FLOAT32_C(   741.55), EASYSIMD_FLOAT32_C(  -561.54), EASYSIMD_FLOAT32_C(   749.46) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 70),
      { EASYSIMD_FLOAT32_C(   401.90), EASYSIMD_FLOAT32_C(  -781.06), EASYSIMD_FLOAT32_C(  -547.42), EASYSIMD_FLOAT32_C(  -454.13),
        EASYSIMD_FLOAT32_C(   980.67), EASYSIMD_FLOAT32_C(   -78.88), EASYSIMD_FLOAT32_C(  -554.00), EASYSIMD_FLOAT32_C(   365.64) },
      { EASYSIMD_FLOAT32_C(  -929.74), EASYSIMD_FLOAT32_C(   268.90), EASYSIMD_FLOAT32_C(   967.95), EASYSIMD_FLOAT32_C(   821.52),
        EASYSIMD_FLOAT32_C(  -741.02), EASYSIMD_FLOAT32_C(   373.75), EASYSIMD_FLOAT32_C(  -724.64), EASYSIMD_FLOAT32_C(  -117.35) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.90), EASYSIMD_FLOAT32_C(    -0.57), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.76), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 29),
      { EASYSIMD_FLOAT32_C(  -999.78), EASYSIMD_FLOAT32_C(  -449.85), EASYSIMD_FLOAT32_C(   -29.74), EASYSIMD_FLOAT32_C(   -97.09),
        EASYSIMD_FLOAT32_C(   332.17), EASYSIMD_FLOAT32_C(  -625.85), EASYSIMD_FLOAT32_C(   -61.66), EASYSIMD_FLOAT32_C(    30.75) },
      { EASYSIMD_FLOAT32_C(   905.22), EASYSIMD_FLOAT32_C(   759.95), EASYSIMD_FLOAT32_C(  -227.70), EASYSIMD_FLOAT32_C(  -656.32),
        EASYSIMD_FLOAT32_C(   509.41), EASYSIMD_FLOAT32_C(  -527.57), EASYSIMD_FLOAT32_C(   745.58), EASYSIMD_FLOAT32_C(   728.35) },
      { EASYSIMD_FLOAT32_C(    -1.10), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.15),
        EASYSIMD_FLOAT32_C(     0.65), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(223),
      { EASYSIMD_FLOAT32_C(  -708.55), EASYSIMD_FLOAT32_C(   709.02), EASYSIMD_FLOAT32_C(   846.13), EASYSIMD_FLOAT32_C(  -262.55),
        EASYSIMD_FLOAT32_C(    74.66), EASYSIMD_FLOAT32_C(   916.39), EASYSIMD_FLOAT32_C(  -993.65), EASYSIMD_FLOAT32_C(    42.60) },
      { EASYSIMD_FLOAT32_C(   737.91), EASYSIMD_FLOAT32_C(  -734.67), EASYSIMD_FLOAT32_C(  -583.65), EASYSIMD_FLOAT32_C(  -986.74),
        EASYSIMD_FLOAT32_C(   147.98), EASYSIMD_FLOAT32_C(   444.29), EASYSIMD_FLOAT32_C(  -986.51), EASYSIMD_FLOAT32_C(   698.13) },
      { EASYSIMD_FLOAT32_C(    -0.96), EASYSIMD_FLOAT32_C(    -0.97), EASYSIMD_FLOAT32_C(    -1.45), EASYSIMD_FLOAT32_C(     0.27),
        EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.01), EASYSIMD_FLOAT32_C(     0.06) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_div_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_div_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_div_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_div_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   161.51), EASYSIMD_FLOAT64_C(  -557.74), EASYSIMD_FLOAT64_C(   247.70), EASYSIMD_FLOAT64_C(   187.40) },
      UINT8_C(227),
      { EASYSIMD_FLOAT64_C(  -311.86), EASYSIMD_FLOAT64_C(  -727.54), EASYSIMD_FLOAT64_C(  -342.19), EASYSIMD_FLOAT64_C(  -474.76) },
      { EASYSIMD_FLOAT64_C(   662.32), EASYSIMD_FLOAT64_C(  -935.07), EASYSIMD_FLOAT64_C(  -729.31), EASYSIMD_FLOAT64_C(  -540.37) },
      { EASYSIMD_FLOAT64_C(    -0.47), EASYSIMD_FLOAT64_C(     0.78), EASYSIMD_FLOAT64_C(   247.70), EASYSIMD_FLOAT64_C(   187.40) } },
    { { EASYSIMD_FLOAT64_C(   820.43), EASYSIMD_FLOAT64_C(   546.47), EASYSIMD_FLOAT64_C(  -588.65), EASYSIMD_FLOAT64_C(    85.11) },
      UINT8_C(190),
      { EASYSIMD_FLOAT64_C(   423.08), EASYSIMD_FLOAT64_C(   -15.78), EASYSIMD_FLOAT64_C(   826.60), EASYSIMD_FLOAT64_C(    48.54) },
      { EASYSIMD_FLOAT64_C(   956.91), EASYSIMD_FLOAT64_C(   494.38), EASYSIMD_FLOAT64_C(   574.90), EASYSIMD_FLOAT64_C(  -499.69) },
      { EASYSIMD_FLOAT64_C(   820.43), EASYSIMD_FLOAT64_C(    -0.03), EASYSIMD_FLOAT64_C(     1.44), EASYSIMD_FLOAT64_C(    -0.10) } },
    { { EASYSIMD_FLOAT64_C(  -241.57), EASYSIMD_FLOAT64_C(  -586.69), EASYSIMD_FLOAT64_C(  -927.84), EASYSIMD_FLOAT64_C(    22.06) },
      UINT8_C( 19),
      { EASYSIMD_FLOAT64_C(   233.67), EASYSIMD_FLOAT64_C(   464.32), EASYSIMD_FLOAT64_C(  -274.79), EASYSIMD_FLOAT64_C(  -578.92) },
      { EASYSIMD_FLOAT64_C(  -939.02), EASYSIMD_FLOAT64_C(   413.35), EASYSIMD_FLOAT64_C(  -306.46), EASYSIMD_FLOAT64_C(  -281.21) },
      { EASYSIMD_FLOAT64_C(    -0.25), EASYSIMD_FLOAT64_C(     1.12), EASYSIMD_FLOAT64_C(  -927.84), EASYSIMD_FLOAT64_C(    22.06) } },
    { { EASYSIMD_FLOAT64_C(   938.59), EASYSIMD_FLOAT64_C(  -644.14), EASYSIMD_FLOAT64_C(  -216.28), EASYSIMD_FLOAT64_C(  -790.71) },
      UINT8_C( 73),
      { EASYSIMD_FLOAT64_C(  -395.84), EASYSIMD_FLOAT64_C(   755.76), EASYSIMD_FLOAT64_C(   226.84), EASYSIMD_FLOAT64_C(   689.26) },
      { EASYSIMD_FLOAT64_C(  -632.97), EASYSIMD_FLOAT64_C(  -350.08), EASYSIMD_FLOAT64_C(  -326.52), EASYSIMD_FLOAT64_C(  -806.38) },
      { EASYSIMD_FLOAT64_C(     0.63), EASYSIMD_FLOAT64_C(  -644.14), EASYSIMD_FLOAT64_C(  -216.28), EASYSIMD_FLOAT64_C(    -0.85) } },
    { { EASYSIMD_FLOAT64_C(   698.46), EASYSIMD_FLOAT64_C(  -369.61), EASYSIMD_FLOAT64_C(   688.01), EASYSIMD_FLOAT64_C(   273.36) },
      UINT8_C( 85),
      { EASYSIMD_FLOAT64_C(  -553.57), EASYSIMD_FLOAT64_C(   686.67), EASYSIMD_FLOAT64_C(   202.86), EASYSIMD_FLOAT64_C(   468.49) },
      { EASYSIMD_FLOAT64_C(   164.18), EASYSIMD_FLOAT64_C(  -563.46), EASYSIMD_FLOAT64_C(   -67.19), EASYSIMD_FLOAT64_C(   889.39) },
      { EASYSIMD_FLOAT64_C(    -3.37), EASYSIMD_FLOAT64_C(  -369.61), EASYSIMD_FLOAT64_C(    -3.02), EASYSIMD_FLOAT64_C(   273.36) } },
    { { EASYSIMD_FLOAT64_C(  -142.38), EASYSIMD_FLOAT64_C(    -6.21), EASYSIMD_FLOAT64_C(   302.74), EASYSIMD_FLOAT64_C(   551.16) },
      UINT8_C(185),
      { EASYSIMD_FLOAT64_C(   241.33), EASYSIMD_FLOAT64_C(   907.01), EASYSIMD_FLOAT64_C(  -503.70), EASYSIMD_FLOAT64_C(   450.62) },
      { EASYSIMD_FLOAT64_C(  -277.50), EASYSIMD_FLOAT64_C(   100.45), EASYSIMD_FLOAT64_C(   206.37), EASYSIMD_FLOAT64_C(   949.34) },
      { EASYSIMD_FLOAT64_C(    -0.87), EASYSIMD_FLOAT64_C(    -6.21), EASYSIMD_FLOAT64_C(   302.74), EASYSIMD_FLOAT64_C(     0.47) } },
    { { EASYSIMD_FLOAT64_C(  -210.29), EASYSIMD_FLOAT64_C(   573.40), EASYSIMD_FLOAT64_C(  -400.73), EASYSIMD_FLOAT64_C(   463.20) },
      UINT8_C(204),
      { EASYSIMD_FLOAT64_C(  -702.28), EASYSIMD_FLOAT64_C(  -906.41), EASYSIMD_FLOAT64_C(   455.03), EASYSIMD_FLOAT64_C(   571.08) },
      { EASYSIMD_FLOAT64_C(   224.29), EASYSIMD_FLOAT64_C(   901.46), EASYSIMD_FLOAT64_C(   257.75), EASYSIMD_FLOAT64_C(  -572.84) },
      { EASYSIMD_FLOAT64_C(  -210.29), EASYSIMD_FLOAT64_C(   573.40), EASYSIMD_FLOAT64_C(     1.77), EASYSIMD_FLOAT64_C(    -1.00) } },
    { { EASYSIMD_FLOAT64_C(   369.95), EASYSIMD_FLOAT64_C(  -578.07), EASYSIMD_FLOAT64_C(  -136.31), EASYSIMD_FLOAT64_C(  -697.23) },
      UINT8_C(191),
      { EASYSIMD_FLOAT64_C(   721.31), EASYSIMD_FLOAT64_C(   296.56), EASYSIMD_FLOAT64_C(   614.05), EASYSIMD_FLOAT64_C(   272.47) },
      { EASYSIMD_FLOAT64_C(     9.13), EASYSIMD_FLOAT64_C(  -144.62), EASYSIMD_FLOAT64_C(   179.48), EASYSIMD_FLOAT64_C(   505.43) },
      { EASYSIMD_FLOAT64_C(    79.00), EASYSIMD_FLOAT64_C(    -2.05), EASYSIMD_FLOAT64_C(     3.42), EASYSIMD_FLOAT64_C(     0.54) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_div_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_div_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_div_pd(src, k, a, b);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_div_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(217),
      { EASYSIMD_FLOAT64_C(   -59.98), EASYSIMD_FLOAT64_C(   658.95), EASYSIMD_FLOAT64_C(  -956.60), EASYSIMD_FLOAT64_C(   382.77) },
      { EASYSIMD_FLOAT64_C(  -744.84), EASYSIMD_FLOAT64_C(  -946.46), EASYSIMD_FLOAT64_C(   937.05), EASYSIMD_FLOAT64_C(  -855.41) },
      { EASYSIMD_FLOAT64_C(     0.08), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.45) } },
    { UINT8_C(153),
      { EASYSIMD_FLOAT64_C(  -798.46), EASYSIMD_FLOAT64_C(  -949.85), EASYSIMD_FLOAT64_C(  -388.70), EASYSIMD_FLOAT64_C(   216.54) },
      { EASYSIMD_FLOAT64_C(   982.22), EASYSIMD_FLOAT64_C(  -982.43), EASYSIMD_FLOAT64_C(   677.55), EASYSIMD_FLOAT64_C(   573.86) },
      { EASYSIMD_FLOAT64_C(    -0.81), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.38) } },
    { UINT8_C(  9),
      { EASYSIMD_FLOAT64_C(  -152.54), EASYSIMD_FLOAT64_C(   481.62), EASYSIMD_FLOAT64_C(   358.15), EASYSIMD_FLOAT64_C(   545.97) },
      { EASYSIMD_FLOAT64_C(   860.09), EASYSIMD_FLOAT64_C(   236.34), EASYSIMD_FLOAT64_C(   713.51), EASYSIMD_FLOAT64_C(   733.84) },
      { EASYSIMD_FLOAT64_C(    -0.18), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.74) } },
    { UINT8_C(125),
      { EASYSIMD_FLOAT64_C(  -949.34), EASYSIMD_FLOAT64_C(  -564.57), EASYSIMD_FLOAT64_C(  -743.96), EASYSIMD_FLOAT64_C(   -54.94) },
      { EASYSIMD_FLOAT64_C(   375.45), EASYSIMD_FLOAT64_C(   914.99), EASYSIMD_FLOAT64_C(   -11.54), EASYSIMD_FLOAT64_C(  -241.77) },
      { EASYSIMD_FLOAT64_C(    -2.53), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    64.47), EASYSIMD_FLOAT64_C(     0.23) } },
    { UINT8_C(234),
      { EASYSIMD_FLOAT64_C(    42.00), EASYSIMD_FLOAT64_C(  -304.72), EASYSIMD_FLOAT64_C(  -685.26), EASYSIMD_FLOAT64_C(  -764.50) },
      { EASYSIMD_FLOAT64_C(  -103.19), EASYSIMD_FLOAT64_C(  -635.11), EASYSIMD_FLOAT64_C(  -153.20), EASYSIMD_FLOAT64_C(  -886.65) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.48), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.86) } },
    { UINT8_C(159),
      { EASYSIMD_FLOAT64_C(  -135.63), EASYSIMD_FLOAT64_C(   790.90), EASYSIMD_FLOAT64_C(   920.98), EASYSIMD_FLOAT64_C(   417.71) },
      { EASYSIMD_FLOAT64_C(  -361.64), EASYSIMD_FLOAT64_C(   402.59), EASYSIMD_FLOAT64_C(  -224.13), EASYSIMD_FLOAT64_C(  -815.67) },
      { EASYSIMD_FLOAT64_C(     0.38), EASYSIMD_FLOAT64_C(     1.96), EASYSIMD_FLOAT64_C(    -4.11), EASYSIMD_FLOAT64_C(    -0.51) } },
    { UINT8_C( 68),
      { EASYSIMD_FLOAT64_C(  -987.79), EASYSIMD_FLOAT64_C(   897.84), EASYSIMD_FLOAT64_C(    -3.48), EASYSIMD_FLOAT64_C(  -566.26) },
      { EASYSIMD_FLOAT64_C(   948.50), EASYSIMD_FLOAT64_C(   431.96), EASYSIMD_FLOAT64_C(  -310.22), EASYSIMD_FLOAT64_C(  -106.44) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.01), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 74),
      { EASYSIMD_FLOAT64_C(  -395.23), EASYSIMD_FLOAT64_C(   882.03), EASYSIMD_FLOAT64_C(   565.63), EASYSIMD_FLOAT64_C(  -225.08) },
      { EASYSIMD_FLOAT64_C(   -75.97), EASYSIMD_FLOAT64_C(  -739.09), EASYSIMD_FLOAT64_C(    89.66), EASYSIMD_FLOAT64_C(   159.53) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.19), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.41) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_div_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_div_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_div_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_div_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   653.62), EASYSIMD_FLOAT32_C(   981.74), EASYSIMD_FLOAT32_C(   780.10), EASYSIMD_FLOAT32_C(    59.38),
                         EASYSIMD_FLOAT32_C(  -795.11), EASYSIMD_FLOAT32_C(   923.87), EASYSIMD_FLOAT32_C(  -270.01), EASYSIMD_FLOAT32_C(  -411.99),
                         EASYSIMD_FLOAT32_C(   -97.83), EASYSIMD_FLOAT32_C(  -393.82), EASYSIMD_FLOAT32_C(   934.81), EASYSIMD_FLOAT32_C(    74.53),
                         EASYSIMD_FLOAT32_C(   843.79), EASYSIMD_FLOAT32_C(   465.05), EASYSIMD_FLOAT32_C(   -42.07), EASYSIMD_FLOAT32_C(  -685.83)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   596.54), EASYSIMD_FLOAT32_C(  -116.40), EASYSIMD_FLOAT32_C(  -989.77), EASYSIMD_FLOAT32_C(  -794.40),
                         EASYSIMD_FLOAT32_C(   183.38), EASYSIMD_FLOAT32_C(  -185.75), EASYSIMD_FLOAT32_C(   429.70), EASYSIMD_FLOAT32_C(   664.04),
                         EASYSIMD_FLOAT32_C(   296.78), EASYSIMD_FLOAT32_C(  -698.78), EASYSIMD_FLOAT32_C(   908.33), EASYSIMD_FLOAT32_C(   181.85),
                         EASYSIMD_FLOAT32_C(  -397.89), EASYSIMD_FLOAT32_C(  -586.75), EASYSIMD_FLOAT32_C(   904.99), EASYSIMD_FLOAT32_C(  -321.15)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     1.10), EASYSIMD_FLOAT32_C(    -8.43), EASYSIMD_FLOAT32_C(    -0.79), EASYSIMD_FLOAT32_C(    -0.07),
                         EASYSIMD_FLOAT32_C(    -4.34), EASYSIMD_FLOAT32_C(    -4.97), EASYSIMD_FLOAT32_C(    -0.63), EASYSIMD_FLOAT32_C(    -0.62),
                         EASYSIMD_FLOAT32_C(    -0.33), EASYSIMD_FLOAT32_C(     0.56), EASYSIMD_FLOAT32_C(     1.03), EASYSIMD_FLOAT32_C(     0.41),
                         EASYSIMD_FLOAT32_C(    -2.12), EASYSIMD_FLOAT32_C(    -0.79), EASYSIMD_FLOAT32_C(    -0.05), EASYSIMD_FLOAT32_C(     2.14)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   729.63), EASYSIMD_FLOAT32_C(  -908.06), EASYSIMD_FLOAT32_C(  -769.77), EASYSIMD_FLOAT32_C(   -70.66),
                         EASYSIMD_FLOAT32_C(   482.71), EASYSIMD_FLOAT32_C(   244.66), EASYSIMD_FLOAT32_C(  -615.83), EASYSIMD_FLOAT32_C(   841.42),
                         EASYSIMD_FLOAT32_C(  -571.10), EASYSIMD_FLOAT32_C(   971.96), EASYSIMD_FLOAT32_C(   149.38), EASYSIMD_FLOAT32_C(   497.71),
                         EASYSIMD_FLOAT32_C(   988.69), EASYSIMD_FLOAT32_C(   479.68), EASYSIMD_FLOAT32_C(  -128.24), EASYSIMD_FLOAT32_C(   585.28)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   359.65), EASYSIMD_FLOAT32_C(  -730.08), EASYSIMD_FLOAT32_C(   977.98), EASYSIMD_FLOAT32_C(  -215.53),
                         EASYSIMD_FLOAT32_C(  -315.50), EASYSIMD_FLOAT32_C(    80.64), EASYSIMD_FLOAT32_C(  -996.10), EASYSIMD_FLOAT32_C(  -556.83),
                         EASYSIMD_FLOAT32_C(  -628.68), EASYSIMD_FLOAT32_C(   938.60), EASYSIMD_FLOAT32_C(  -147.98), EASYSIMD_FLOAT32_C(   378.31),
                         EASYSIMD_FLOAT32_C(   246.47), EASYSIMD_FLOAT32_C(   109.18), EASYSIMD_FLOAT32_C(  -575.64), EASYSIMD_FLOAT32_C(  -426.86)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     2.03), EASYSIMD_FLOAT32_C(     1.24), EASYSIMD_FLOAT32_C(    -0.79), EASYSIMD_FLOAT32_C(     0.33),
                         EASYSIMD_FLOAT32_C(    -1.53), EASYSIMD_FLOAT32_C(     3.03), EASYSIMD_FLOAT32_C(     0.62), EASYSIMD_FLOAT32_C(    -1.51),
                         EASYSIMD_FLOAT32_C(     0.91), EASYSIMD_FLOAT32_C(     1.04), EASYSIMD_FLOAT32_C(    -1.01), EASYSIMD_FLOAT32_C(     1.32),
                         EASYSIMD_FLOAT32_C(     4.01), EASYSIMD_FLOAT32_C(     4.39), EASYSIMD_FLOAT32_C(     0.22), EASYSIMD_FLOAT32_C(    -1.37)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -148.70), EASYSIMD_FLOAT32_C(  -327.17), EASYSIMD_FLOAT32_C(  -310.14), EASYSIMD_FLOAT32_C(  -718.80),
                         EASYSIMD_FLOAT32_C(   382.69), EASYSIMD_FLOAT32_C(  -181.61), EASYSIMD_FLOAT32_C(  -214.09), EASYSIMD_FLOAT32_C(    55.72),
                         EASYSIMD_FLOAT32_C(   438.03), EASYSIMD_FLOAT32_C(  -458.01), EASYSIMD_FLOAT32_C(   144.59), EASYSIMD_FLOAT32_C(   165.00),
                         EASYSIMD_FLOAT32_C(  -331.04), EASYSIMD_FLOAT32_C(   406.96), EASYSIMD_FLOAT32_C(  -326.43), EASYSIMD_FLOAT32_C(   373.82)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   791.83), EASYSIMD_FLOAT32_C(   191.69), EASYSIMD_FLOAT32_C(  -460.58), EASYSIMD_FLOAT32_C(  -915.08),
                         EASYSIMD_FLOAT32_C(  -877.38), EASYSIMD_FLOAT32_C(  -915.27), EASYSIMD_FLOAT32_C(   207.85), EASYSIMD_FLOAT32_C(   567.35),
                         EASYSIMD_FLOAT32_C(   304.30), EASYSIMD_FLOAT32_C(  -777.07), EASYSIMD_FLOAT32_C(  -683.73), EASYSIMD_FLOAT32_C(  -113.32),
                         EASYSIMD_FLOAT32_C(  -701.16), EASYSIMD_FLOAT32_C(  -942.92), EASYSIMD_FLOAT32_C(  -489.97), EASYSIMD_FLOAT32_C(   911.34)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    -0.19), EASYSIMD_FLOAT32_C(    -1.71), EASYSIMD_FLOAT32_C(     0.67), EASYSIMD_FLOAT32_C(     0.79),
                         EASYSIMD_FLOAT32_C(    -0.44), EASYSIMD_FLOAT32_C(     0.20), EASYSIMD_FLOAT32_C(    -1.03), EASYSIMD_FLOAT32_C(     0.10),
                         EASYSIMD_FLOAT32_C(     1.44), EASYSIMD_FLOAT32_C(     0.59), EASYSIMD_FLOAT32_C(    -0.21), EASYSIMD_FLOAT32_C(    -1.46),
                         EASYSIMD_FLOAT32_C(     0.47), EASYSIMD_FLOAT32_C(    -0.43), EASYSIMD_FLOAT32_C(     0.67), EASYSIMD_FLOAT32_C(     0.41)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -869.58), EASYSIMD_FLOAT32_C(   763.75), EASYSIMD_FLOAT32_C(  -558.93), EASYSIMD_FLOAT32_C(   756.19),
                         EASYSIMD_FLOAT32_C(   509.82), EASYSIMD_FLOAT32_C(  -855.71), EASYSIMD_FLOAT32_C(  -965.40), EASYSIMD_FLOAT32_C(  -279.29),
                         EASYSIMD_FLOAT32_C(  -798.08), EASYSIMD_FLOAT32_C(   256.40), EASYSIMD_FLOAT32_C(   739.89), EASYSIMD_FLOAT32_C(  -903.46),
                         EASYSIMD_FLOAT32_C(  -771.75), EASYSIMD_FLOAT32_C(   -54.77), EASYSIMD_FLOAT32_C(   397.04), EASYSIMD_FLOAT32_C(   925.94)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -355.51), EASYSIMD_FLOAT32_C(   136.73), EASYSIMD_FLOAT32_C(   586.70), EASYSIMD_FLOAT32_C(   712.56),
                         EASYSIMD_FLOAT32_C(   135.88), EASYSIMD_FLOAT32_C(  -693.91), EASYSIMD_FLOAT32_C(  -131.33), EASYSIMD_FLOAT32_C(  -933.79),
                         EASYSIMD_FLOAT32_C(   864.29), EASYSIMD_FLOAT32_C(  -834.00), EASYSIMD_FLOAT32_C(   475.52), EASYSIMD_FLOAT32_C(   502.31),
                         EASYSIMD_FLOAT32_C(  -746.87), EASYSIMD_FLOAT32_C(  -364.10), EASYSIMD_FLOAT32_C(  -995.18), EASYSIMD_FLOAT32_C(   683.54)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     2.45), EASYSIMD_FLOAT32_C(     5.59), EASYSIMD_FLOAT32_C(    -0.95), EASYSIMD_FLOAT32_C(     1.06),
                         EASYSIMD_FLOAT32_C(     3.75), EASYSIMD_FLOAT32_C(     1.23), EASYSIMD_FLOAT32_C(     7.35), EASYSIMD_FLOAT32_C(     0.30),
                         EASYSIMD_FLOAT32_C(    -0.92), EASYSIMD_FLOAT32_C(    -0.31), EASYSIMD_FLOAT32_C(     1.56), EASYSIMD_FLOAT32_C(    -1.80),
                         EASYSIMD_FLOAT32_C(     1.03), EASYSIMD_FLOAT32_C(     0.15), EASYSIMD_FLOAT32_C(    -0.40), EASYSIMD_FLOAT32_C(     1.35)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   119.21), EASYSIMD_FLOAT32_C(   360.54), EASYSIMD_FLOAT32_C(   885.26), EASYSIMD_FLOAT32_C(  -618.98),
                         EASYSIMD_FLOAT32_C(    -8.97), EASYSIMD_FLOAT32_C(  -881.58), EASYSIMD_FLOAT32_C(   -89.25), EASYSIMD_FLOAT32_C(  -937.64),
                         EASYSIMD_FLOAT32_C(  -660.18), EASYSIMD_FLOAT32_C(  -649.17), EASYSIMD_FLOAT32_C(  -279.52), EASYSIMD_FLOAT32_C(   812.95),
                         EASYSIMD_FLOAT32_C(  -471.80), EASYSIMD_FLOAT32_C(   805.98), EASYSIMD_FLOAT32_C(   532.44), EASYSIMD_FLOAT32_C(   126.30)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   944.81), EASYSIMD_FLOAT32_C(   946.29), EASYSIMD_FLOAT32_C(   161.37), EASYSIMD_FLOAT32_C(  -637.11),
                         EASYSIMD_FLOAT32_C(    16.54), EASYSIMD_FLOAT32_C(   417.79), EASYSIMD_FLOAT32_C(   257.34), EASYSIMD_FLOAT32_C(  -857.05),
                         EASYSIMD_FLOAT32_C(   770.17), EASYSIMD_FLOAT32_C(  -559.67), EASYSIMD_FLOAT32_C(  -862.75), EASYSIMD_FLOAT32_C(  -541.96),
                         EASYSIMD_FLOAT32_C(   412.30), EASYSIMD_FLOAT32_C(  -147.64), EASYSIMD_FLOAT32_C(   553.94), EASYSIMD_FLOAT32_C(  -736.63)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(     0.38), EASYSIMD_FLOAT32_C(     5.49), EASYSIMD_FLOAT32_C(     0.97),
                         EASYSIMD_FLOAT32_C(    -0.54), EASYSIMD_FLOAT32_C(    -2.11), EASYSIMD_FLOAT32_C(    -0.35), EASYSIMD_FLOAT32_C(     1.09),
                         EASYSIMD_FLOAT32_C(    -0.86), EASYSIMD_FLOAT32_C(     1.16), EASYSIMD_FLOAT32_C(     0.32), EASYSIMD_FLOAT32_C(    -1.50),
                         EASYSIMD_FLOAT32_C(    -1.14), EASYSIMD_FLOAT32_C(    -5.46), EASYSIMD_FLOAT32_C(     0.96), EASYSIMD_FLOAT32_C(    -0.17)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -81.24), EASYSIMD_FLOAT32_C(  -934.88), EASYSIMD_FLOAT32_C(   -84.21), EASYSIMD_FLOAT32_C(  -265.16),
                         EASYSIMD_FLOAT32_C(  -978.34), EASYSIMD_FLOAT32_C(  -425.47), EASYSIMD_FLOAT32_C(   792.31), EASYSIMD_FLOAT32_C(  -306.03),
                         EASYSIMD_FLOAT32_C(   911.07), EASYSIMD_FLOAT32_C(   992.01), EASYSIMD_FLOAT32_C(   172.45), EASYSIMD_FLOAT32_C(  -135.31),
                         EASYSIMD_FLOAT32_C(   652.11), EASYSIMD_FLOAT32_C(  -529.15), EASYSIMD_FLOAT32_C(    -0.58), EASYSIMD_FLOAT32_C(   883.05)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -110.89), EASYSIMD_FLOAT32_C(  -325.07), EASYSIMD_FLOAT32_C(   834.96), EASYSIMD_FLOAT32_C(  -681.06),
                         EASYSIMD_FLOAT32_C(  -877.63), EASYSIMD_FLOAT32_C(  -653.45), EASYSIMD_FLOAT32_C(    40.48), EASYSIMD_FLOAT32_C(  -644.02),
                         EASYSIMD_FLOAT32_C(  -687.76), EASYSIMD_FLOAT32_C(  -660.68), EASYSIMD_FLOAT32_C(   802.46), EASYSIMD_FLOAT32_C(  -477.95),
                         EASYSIMD_FLOAT32_C(  -125.80), EASYSIMD_FLOAT32_C(  -475.50), EASYSIMD_FLOAT32_C(  -806.50), EASYSIMD_FLOAT32_C(  -778.62)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.73), EASYSIMD_FLOAT32_C(     2.88), EASYSIMD_FLOAT32_C(    -0.10), EASYSIMD_FLOAT32_C(     0.39),
                         EASYSIMD_FLOAT32_C(     1.11), EASYSIMD_FLOAT32_C(     0.65), EASYSIMD_FLOAT32_C(    19.57), EASYSIMD_FLOAT32_C(     0.48),
                         EASYSIMD_FLOAT32_C(    -1.32), EASYSIMD_FLOAT32_C(    -1.50), EASYSIMD_FLOAT32_C(     0.21), EASYSIMD_FLOAT32_C(     0.28),
                         EASYSIMD_FLOAT32_C(    -5.18), EASYSIMD_FLOAT32_C(     1.11), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.13)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -387.95), EASYSIMD_FLOAT32_C(   255.55), EASYSIMD_FLOAT32_C(   948.01), EASYSIMD_FLOAT32_C(   520.84),
                         EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -412.39), EASYSIMD_FLOAT32_C(   412.17), EASYSIMD_FLOAT32_C(  -913.22),
                         EASYSIMD_FLOAT32_C(   810.06), EASYSIMD_FLOAT32_C(  -696.65), EASYSIMD_FLOAT32_C(   807.84), EASYSIMD_FLOAT32_C(    63.85),
                         EASYSIMD_FLOAT32_C(    -2.75), EASYSIMD_FLOAT32_C(  -763.61), EASYSIMD_FLOAT32_C(  -850.85), EASYSIMD_FLOAT32_C(   913.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -915.78), EASYSIMD_FLOAT32_C(   471.39), EASYSIMD_FLOAT32_C(  -324.79), EASYSIMD_FLOAT32_C(  -855.69),
                         EASYSIMD_FLOAT32_C(   966.81), EASYSIMD_FLOAT32_C(   668.44), EASYSIMD_FLOAT32_C(   925.33), EASYSIMD_FLOAT32_C(   564.88),
                         EASYSIMD_FLOAT32_C(  -130.24), EASYSIMD_FLOAT32_C(   360.71), EASYSIMD_FLOAT32_C(   966.21), EASYSIMD_FLOAT32_C(  -919.67),
                         EASYSIMD_FLOAT32_C(   198.47), EASYSIMD_FLOAT32_C(  -796.49), EASYSIMD_FLOAT32_C(   428.08), EASYSIMD_FLOAT32_C(   264.02)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.42), EASYSIMD_FLOAT32_C(     0.54), EASYSIMD_FLOAT32_C(    -2.92), EASYSIMD_FLOAT32_C(    -0.61),
                         EASYSIMD_FLOAT32_C(     0.32), EASYSIMD_FLOAT32_C(    -0.62), EASYSIMD_FLOAT32_C(     0.45), EASYSIMD_FLOAT32_C(    -1.62),
                         EASYSIMD_FLOAT32_C(    -6.22), EASYSIMD_FLOAT32_C(    -1.93), EASYSIMD_FLOAT32_C(     0.84), EASYSIMD_FLOAT32_C(    -0.07),
                         EASYSIMD_FLOAT32_C(    -0.01), EASYSIMD_FLOAT32_C(     0.96), EASYSIMD_FLOAT32_C(    -1.99), EASYSIMD_FLOAT32_C(     3.46)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   534.55), EASYSIMD_FLOAT32_C(  -263.46), EASYSIMD_FLOAT32_C(  -958.21), EASYSIMD_FLOAT32_C(   927.39),
                         EASYSIMD_FLOAT32_C(   830.49), EASYSIMD_FLOAT32_C(  -394.19), EASYSIMD_FLOAT32_C(  -755.65), EASYSIMD_FLOAT32_C(  -594.24),
                         EASYSIMD_FLOAT32_C(  -371.00), EASYSIMD_FLOAT32_C(   623.04), EASYSIMD_FLOAT32_C(   879.76), EASYSIMD_FLOAT32_C(   838.28),
                         EASYSIMD_FLOAT32_C(  -100.77), EASYSIMD_FLOAT32_C(  -708.14), EASYSIMD_FLOAT32_C(  -206.06), EASYSIMD_FLOAT32_C(  -203.03)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    65.94), EASYSIMD_FLOAT32_C(   158.39), EASYSIMD_FLOAT32_C(   532.17), EASYSIMD_FLOAT32_C(    -1.61),
                         EASYSIMD_FLOAT32_C(  -802.21), EASYSIMD_FLOAT32_C(  -782.13), EASYSIMD_FLOAT32_C(   831.96), EASYSIMD_FLOAT32_C(  -692.14),
                         EASYSIMD_FLOAT32_C(   581.38), EASYSIMD_FLOAT32_C(   943.65), EASYSIMD_FLOAT32_C(   585.87), EASYSIMD_FLOAT32_C(   329.94),
                         EASYSIMD_FLOAT32_C(  -747.39), EASYSIMD_FLOAT32_C(   976.32), EASYSIMD_FLOAT32_C(   362.23), EASYSIMD_FLOAT32_C(  -137.03)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     8.11), EASYSIMD_FLOAT32_C(    -1.66), EASYSIMD_FLOAT32_C(    -1.80), EASYSIMD_FLOAT32_C(  -576.02),
                         EASYSIMD_FLOAT32_C(    -1.04), EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(    -0.91), EASYSIMD_FLOAT32_C(     0.86),
                         EASYSIMD_FLOAT32_C(    -0.64), EASYSIMD_FLOAT32_C(     0.66), EASYSIMD_FLOAT32_C(     1.50), EASYSIMD_FLOAT32_C(     2.54),
                         EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(    -0.73), EASYSIMD_FLOAT32_C(    -0.57), EASYSIMD_FLOAT32_C(     1.48)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_div_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_div_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_div_round_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    easysimd_float32 a[16];
    easysimd_float32 b[16];
    easysimd_float32 nearest_inf[16];
    easysimd_float32 neg_inf[16];
    easysimd_float32 pos_inf[16];
    easysimd_float32 zero[16];
    easysimd_float32 direction[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -903.55), EASYSIMD_FLOAT32_C(  -411.28), EASYSIMD_FLOAT32_C(   791.75), EASYSIMD_FLOAT32_C(  -571.69),
        EASYSIMD_FLOAT32_C(    87.97), EASYSIMD_FLOAT32_C(  -149.56), EASYSIMD_FLOAT32_C(   533.34), EASYSIMD_FLOAT32_C(   909.95),
        EASYSIMD_FLOAT32_C(   175.32), EASYSIMD_FLOAT32_C(   117.20), EASYSIMD_FLOAT32_C(  -637.14), EASYSIMD_FLOAT32_C(  -423.11),
        EASYSIMD_FLOAT32_C(   525.13), EASYSIMD_FLOAT32_C(    13.68), EASYSIMD_FLOAT32_C(  -571.22), EASYSIMD_FLOAT32_C(    92.07) },
      { EASYSIMD_FLOAT32_C(  -181.85), EASYSIMD_FLOAT32_C(  -154.18), EASYSIMD_FLOAT32_C(   703.97), EASYSIMD_FLOAT32_C(   264.49),
        EASYSIMD_FLOAT32_C(   395.18), EASYSIMD_FLOAT32_C(   369.84), EASYSIMD_FLOAT32_C(   724.23), EASYSIMD_FLOAT32_C(   171.79),
        EASYSIMD_FLOAT32_C(    77.88), EASYSIMD_FLOAT32_C(   -84.55), EASYSIMD_FLOAT32_C(  -459.40), EASYSIMD_FLOAT32_C(  -184.55),
        EASYSIMD_FLOAT32_C(   387.25), EASYSIMD_FLOAT32_C(   931.58), EASYSIMD_FLOAT32_C(  -362.11), EASYSIMD_FLOAT32_C(   483.69) },
      { EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     5.00),
        EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     4.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -3.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     5.00),
        EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     6.00),
        EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     1.00) },
      { EASYSIMD_FLOAT32_C(     4.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     5.00),
        EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     5.00),
        EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -479.70), EASYSIMD_FLOAT32_C(  -570.36), EASYSIMD_FLOAT32_C(   912.00), EASYSIMD_FLOAT32_C(   608.27),
        EASYSIMD_FLOAT32_C(   280.08), EASYSIMD_FLOAT32_C(   445.34), EASYSIMD_FLOAT32_C(   518.22), EASYSIMD_FLOAT32_C(  -544.61),
        EASYSIMD_FLOAT32_C(  -437.45), EASYSIMD_FLOAT32_C(   881.08), EASYSIMD_FLOAT32_C(    32.29), EASYSIMD_FLOAT32_C(  -912.32),
        EASYSIMD_FLOAT32_C(  -105.24), EASYSIMD_FLOAT32_C(   461.06), EASYSIMD_FLOAT32_C(   179.75), EASYSIMD_FLOAT32_C(   712.92) },
      { EASYSIMD_FLOAT32_C(  -693.12), EASYSIMD_FLOAT32_C(  -116.28), EASYSIMD_FLOAT32_C(   -22.59), EASYSIMD_FLOAT32_C(   702.06),
        EASYSIMD_FLOAT32_C(  -746.44), EASYSIMD_FLOAT32_C(  -298.36), EASYSIMD_FLOAT32_C(  -126.14), EASYSIMD_FLOAT32_C(   331.44),
        EASYSIMD_FLOAT32_C(   617.10), EASYSIMD_FLOAT32_C(   414.46), EASYSIMD_FLOAT32_C(  -853.11), EASYSIMD_FLOAT32_C(     4.34),
        EASYSIMD_FLOAT32_C(   346.04), EASYSIMD_FLOAT32_C(  -215.21), EASYSIMD_FLOAT32_C(  -511.96), EASYSIMD_FLOAT32_C(   866.33) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(   -40.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -210.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     4.00), EASYSIMD_FLOAT32_C(   -41.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(  -211.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(   -40.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -210.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     4.00), EASYSIMD_FLOAT32_C(   -40.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -210.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(   -40.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -210.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00) } },
    { { EASYSIMD_FLOAT32_C(   214.43), EASYSIMD_FLOAT32_C(  -599.96), EASYSIMD_FLOAT32_C(   474.60), EASYSIMD_FLOAT32_C(  -505.50),
        EASYSIMD_FLOAT32_C(   845.38), EASYSIMD_FLOAT32_C(    -7.18), EASYSIMD_FLOAT32_C(   -50.10), EASYSIMD_FLOAT32_C(  -592.08),
        EASYSIMD_FLOAT32_C(  -126.10), EASYSIMD_FLOAT32_C(   982.19), EASYSIMD_FLOAT32_C(  -504.40), EASYSIMD_FLOAT32_C(   768.66),
        EASYSIMD_FLOAT32_C(   443.25), EASYSIMD_FLOAT32_C(   675.35), EASYSIMD_FLOAT32_C(   481.58), EASYSIMD_FLOAT32_C(   750.13) },
      { EASYSIMD_FLOAT32_C(  -440.93), EASYSIMD_FLOAT32_C(  -541.01), EASYSIMD_FLOAT32_C(   452.19), EASYSIMD_FLOAT32_C(  -187.36),
        EASYSIMD_FLOAT32_C(   160.63), EASYSIMD_FLOAT32_C(  -673.95), EASYSIMD_FLOAT32_C(  -855.92), EASYSIMD_FLOAT32_C(  -222.27),
        EASYSIMD_FLOAT32_C(   740.51), EASYSIMD_FLOAT32_C(  -709.03), EASYSIMD_FLOAT32_C(   782.07), EASYSIMD_FLOAT32_C(    86.55),
        EASYSIMD_FLOAT32_C(    75.76), EASYSIMD_FLOAT32_C(  -729.89), EASYSIMD_FLOAT32_C(   -47.12), EASYSIMD_FLOAT32_C(  -709.81) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     9.00),
        EASYSIMD_FLOAT32_C(     6.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     8.00),
        EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(   -11.00), EASYSIMD_FLOAT32_C(    -2.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     6.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     9.00),
        EASYSIMD_FLOAT32_C(     6.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     8.00),
        EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     5.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     9.00),
        EASYSIMD_FLOAT32_C(     6.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(    -1.00) } },
    { { EASYSIMD_FLOAT32_C(  -329.85), EASYSIMD_FLOAT32_C(  -572.52), EASYSIMD_FLOAT32_C(  -215.31), EASYSIMD_FLOAT32_C(  -484.48),
        EASYSIMD_FLOAT32_C(   420.30), EASYSIMD_FLOAT32_C(   734.59), EASYSIMD_FLOAT32_C(   -76.55), EASYSIMD_FLOAT32_C(  -705.80),
        EASYSIMD_FLOAT32_C(   716.78), EASYSIMD_FLOAT32_C(   419.05), EASYSIMD_FLOAT32_C(  -937.14), EASYSIMD_FLOAT32_C(   160.02),
        EASYSIMD_FLOAT32_C(    94.40), EASYSIMD_FLOAT32_C(   544.44), EASYSIMD_FLOAT32_C(   -89.85), EASYSIMD_FLOAT32_C(   653.47) },
      { EASYSIMD_FLOAT32_C(  -996.57), EASYSIMD_FLOAT32_C(  -637.65), EASYSIMD_FLOAT32_C(  -533.89), EASYSIMD_FLOAT32_C(   164.06),
        EASYSIMD_FLOAT32_C(  -311.60), EASYSIMD_FLOAT32_C(  -389.81), EASYSIMD_FLOAT32_C(   941.79), EASYSIMD_FLOAT32_C(  -571.09),
        EASYSIMD_FLOAT32_C(   -98.83), EASYSIMD_FLOAT32_C(   723.86), EASYSIMD_FLOAT32_C(   515.46), EASYSIMD_FLOAT32_C(   976.93),
        EASYSIMD_FLOAT32_C(   993.97), EASYSIMD_FLOAT32_C(  -531.67), EASYSIMD_FLOAT32_C(  -732.88), EASYSIMD_FLOAT32_C(  -335.89) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -3.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -7.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -3.00),
        EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(    -7.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -7.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -3.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -7.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00) } },
    { { EASYSIMD_FLOAT32_C(  -104.19), EASYSIMD_FLOAT32_C(    51.81), EASYSIMD_FLOAT32_C(   179.64), EASYSIMD_FLOAT32_C(  -683.89),
        EASYSIMD_FLOAT32_C(  -213.60), EASYSIMD_FLOAT32_C(  -896.92), EASYSIMD_FLOAT32_C(  -389.69), EASYSIMD_FLOAT32_C(  -496.83),
        EASYSIMD_FLOAT32_C(   522.13), EASYSIMD_FLOAT32_C(  -326.83), EASYSIMD_FLOAT32_C(   663.20), EASYSIMD_FLOAT32_C(  -383.47),
        EASYSIMD_FLOAT32_C(  -782.39), EASYSIMD_FLOAT32_C(  -426.65), EASYSIMD_FLOAT32_C(  -729.99), EASYSIMD_FLOAT32_C(  -778.97) },
      { EASYSIMD_FLOAT32_C(   -64.30), EASYSIMD_FLOAT32_C(  -263.88), EASYSIMD_FLOAT32_C(   385.09), EASYSIMD_FLOAT32_C(   624.10),
        EASYSIMD_FLOAT32_C(   346.31), EASYSIMD_FLOAT32_C(   326.88), EASYSIMD_FLOAT32_C(  -946.99), EASYSIMD_FLOAT32_C(  -752.53),
        EASYSIMD_FLOAT32_C(    50.74), EASYSIMD_FLOAT32_C(   568.46), EASYSIMD_FLOAT32_C(  -775.60), EASYSIMD_FLOAT32_C(    44.71),
        EASYSIMD_FLOAT32_C(  -963.20), EASYSIMD_FLOAT32_C(  -508.48), EASYSIMD_FLOAT32_C(   708.82), EASYSIMD_FLOAT32_C(   -67.39) },
      { EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    10.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -9.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    12.00) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    10.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -9.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    11.00) },
      { EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    11.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -8.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    12.00) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    10.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -8.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    11.00) },
      { EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    10.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -9.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    12.00) } },
    { { EASYSIMD_FLOAT32_C(   543.33), EASYSIMD_FLOAT32_C(  -111.54), EASYSIMD_FLOAT32_C(   248.72), EASYSIMD_FLOAT32_C(  -670.28),
        EASYSIMD_FLOAT32_C(    -8.46), EASYSIMD_FLOAT32_C(   859.02), EASYSIMD_FLOAT32_C(  -167.10), EASYSIMD_FLOAT32_C(  -486.32),
        EASYSIMD_FLOAT32_C(  -467.81), EASYSIMD_FLOAT32_C(  -503.91), EASYSIMD_FLOAT32_C(   130.21), EASYSIMD_FLOAT32_C(  -250.20),
        EASYSIMD_FLOAT32_C(    69.45), EASYSIMD_FLOAT32_C(   400.22), EASYSIMD_FLOAT32_C(   -29.17), EASYSIMD_FLOAT32_C(  -994.86) },
      { EASYSIMD_FLOAT32_C(  -863.67), EASYSIMD_FLOAT32_C(  -644.08), EASYSIMD_FLOAT32_C(   629.24), EASYSIMD_FLOAT32_C(   482.64),
        EASYSIMD_FLOAT32_C(   682.81), EASYSIMD_FLOAT32_C(   682.25), EASYSIMD_FLOAT32_C(   730.12), EASYSIMD_FLOAT32_C(  -266.45),
        EASYSIMD_FLOAT32_C(   250.71), EASYSIMD_FLOAT32_C(   954.52), EASYSIMD_FLOAT32_C(   778.26), EASYSIMD_FLOAT32_C(   287.50),
        EASYSIMD_FLOAT32_C(  -553.96), EASYSIMD_FLOAT32_C(   487.07), EASYSIMD_FLOAT32_C(  -779.89), EASYSIMD_FLOAT32_C(   989.37) },
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00) } },
    { { EASYSIMD_FLOAT32_C(  -624.47), EASYSIMD_FLOAT32_C(   468.83), EASYSIMD_FLOAT32_C(  -680.91), EASYSIMD_FLOAT32_C(   367.07),
        EASYSIMD_FLOAT32_C(   327.85), EASYSIMD_FLOAT32_C(   151.99), EASYSIMD_FLOAT32_C(   880.75), EASYSIMD_FLOAT32_C(   860.04),
        EASYSIMD_FLOAT32_C(   648.08), EASYSIMD_FLOAT32_C(    10.96), EASYSIMD_FLOAT32_C(  -390.16), EASYSIMD_FLOAT32_C(  -282.47),
        EASYSIMD_FLOAT32_C(  -588.83), EASYSIMD_FLOAT32_C(   580.67), EASYSIMD_FLOAT32_C(  -277.33), EASYSIMD_FLOAT32_C(  -452.49) },
      { EASYSIMD_FLOAT32_C(   936.60), EASYSIMD_FLOAT32_C(  -648.09), EASYSIMD_FLOAT32_C(  -969.85), EASYSIMD_FLOAT32_C(   619.40),
        EASYSIMD_FLOAT32_C(  -965.84), EASYSIMD_FLOAT32_C(   760.27), EASYSIMD_FLOAT32_C(  -647.05), EASYSIMD_FLOAT32_C(   284.87),
        EASYSIMD_FLOAT32_C(   714.79), EASYSIMD_FLOAT32_C(  -868.80), EASYSIMD_FLOAT32_C(  -427.63), EASYSIMD_FLOAT32_C(  -839.17),
        EASYSIMD_FLOAT32_C(   618.28), EASYSIMD_FLOAT32_C(  -207.52), EASYSIMD_FLOAT32_C(  -849.80), EASYSIMD_FLOAT32_C(   993.81) },
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00) },
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     4.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00) },
      { EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(  -738.69), EASYSIMD_FLOAT32_C(  -530.71), EASYSIMD_FLOAT32_C(   360.88), EASYSIMD_FLOAT32_C(   589.16),
        EASYSIMD_FLOAT32_C(   621.27), EASYSIMD_FLOAT32_C(   241.63), EASYSIMD_FLOAT32_C(   449.20), EASYSIMD_FLOAT32_C(   269.36),
        EASYSIMD_FLOAT32_C(  -747.41), EASYSIMD_FLOAT32_C(  -940.96), EASYSIMD_FLOAT32_C(   986.88), EASYSIMD_FLOAT32_C(  -336.24),
        EASYSIMD_FLOAT32_C(   639.71), EASYSIMD_FLOAT32_C(  -290.45), EASYSIMD_FLOAT32_C(   211.27), EASYSIMD_FLOAT32_C(   576.31) },
      { EASYSIMD_FLOAT32_C(    61.46), EASYSIMD_FLOAT32_C(   241.42), EASYSIMD_FLOAT32_C(   195.71), EASYSIMD_FLOAT32_C(    95.62),
        EASYSIMD_FLOAT32_C(     1.69), EASYSIMD_FLOAT32_C(   548.66), EASYSIMD_FLOAT32_C(  -619.51), EASYSIMD_FLOAT32_C(  -283.52),
        EASYSIMD_FLOAT32_C(   679.87), EASYSIMD_FLOAT32_C(   -47.14), EASYSIMD_FLOAT32_C(  -122.69), EASYSIMD_FLOAT32_C(   298.15),
        EASYSIMD_FLOAT32_C(   745.34), EASYSIMD_FLOAT32_C(    27.51), EASYSIMD_FLOAT32_C(   291.96), EASYSIMD_FLOAT32_C(  -993.36) },
      { EASYSIMD_FLOAT32_C(   -12.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     6.00),
        EASYSIMD_FLOAT32_C(   368.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    20.00), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(   -11.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(   -13.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     6.00),
        EASYSIMD_FLOAT32_C(   367.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    19.00), EASYSIMD_FLOAT32_C(    -9.00), EASYSIMD_FLOAT32_C(    -2.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -11.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00) },
      { EASYSIMD_FLOAT32_C(   -12.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     7.00),
        EASYSIMD_FLOAT32_C(   368.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    20.00), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00) },
      { EASYSIMD_FLOAT32_C(   -12.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     6.00),
        EASYSIMD_FLOAT32_C(   367.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    19.00), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00) },
      { EASYSIMD_FLOAT32_C(   -12.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     6.00),
        EASYSIMD_FLOAT32_C(   368.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    20.00), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(   -11.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -1.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 r;
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);

    easysimd__m512 nearest_inf = easysimd_mm512_loadu_ps(test_vec[i].nearest_inf);
    easysimd__m512 neg_inf = easysimd_mm512_loadu_ps(test_vec[i].neg_inf);
    easysimd__m512 pos_inf = easysimd_mm512_loadu_ps(test_vec[i].pos_inf);
    easysimd__m512 zero = easysimd_mm512_loadu_ps(test_vec[i].zero);
    easysimd__m512 direction = easysimd_mm512_loadu_ps(test_vec[i].direction);

    r = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512_close(r, nearest_inf, 1);

    r = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512_close(r, neg_inf, 1);

    r = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512_close(r, pos_inf, 1);

    r = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512_close(r, zero, 1);
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_div_round_ps");
    easysimd_assert_m512_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512 nearest_inf = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512 neg_inf = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512 pos_inf = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512 zero = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512 direction = easysimd_mm512_div_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, nearest_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, neg_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, pos_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, zero, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, direction, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_mask_div_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -745.89), EASYSIMD_FLOAT32_C(   663.97), EASYSIMD_FLOAT32_C(   886.69), EASYSIMD_FLOAT32_C(  -271.39),
                         EASYSIMD_FLOAT32_C(   845.36), EASYSIMD_FLOAT32_C(  -391.34), EASYSIMD_FLOAT32_C(  -606.86), EASYSIMD_FLOAT32_C(   818.59),
                         EASYSIMD_FLOAT32_C(   953.36), EASYSIMD_FLOAT32_C(   863.40), EASYSIMD_FLOAT32_C(   241.85), EASYSIMD_FLOAT32_C(  -815.86),
                         EASYSIMD_FLOAT32_C(   460.12), EASYSIMD_FLOAT32_C(  -674.64), EASYSIMD_FLOAT32_C(   868.62), EASYSIMD_FLOAT32_C(  -710.40)),
      UINT16_C( 9207),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -956.83), EASYSIMD_FLOAT32_C(  -855.01), EASYSIMD_FLOAT32_C(  -219.41), EASYSIMD_FLOAT32_C(    94.89),
                         EASYSIMD_FLOAT32_C(  -270.85), EASYSIMD_FLOAT32_C(   356.85), EASYSIMD_FLOAT32_C(   872.24), EASYSIMD_FLOAT32_C(   100.53),
                         EASYSIMD_FLOAT32_C(   234.39), EASYSIMD_FLOAT32_C(  -639.13), EASYSIMD_FLOAT32_C(   981.49), EASYSIMD_FLOAT32_C(   706.62),
                         EASYSIMD_FLOAT32_C(  -983.90), EASYSIMD_FLOAT32_C(   124.15), EASYSIMD_FLOAT32_C(  -820.87), EASYSIMD_FLOAT32_C(  -946.81)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -17.46), EASYSIMD_FLOAT32_C(   371.67), EASYSIMD_FLOAT32_C(   390.77), EASYSIMD_FLOAT32_C(  -132.40),
                         EASYSIMD_FLOAT32_C(   276.69), EASYSIMD_FLOAT32_C(  -338.80), EASYSIMD_FLOAT32_C(   359.09), EASYSIMD_FLOAT32_C(  -631.66),
                         EASYSIMD_FLOAT32_C(  -455.96), EASYSIMD_FLOAT32_C(    16.63), EASYSIMD_FLOAT32_C(   194.96), EASYSIMD_FLOAT32_C(  -407.18),
                         EASYSIMD_FLOAT32_C(  -447.59), EASYSIMD_FLOAT32_C(  -276.48), EASYSIMD_FLOAT32_C(   631.98), EASYSIMD_FLOAT32_C(   430.67)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -745.89), EASYSIMD_FLOAT32_C(   663.97), EASYSIMD_FLOAT32_C(    -0.56), EASYSIMD_FLOAT32_C(  -271.39),
                         EASYSIMD_FLOAT32_C(   845.36), EASYSIMD_FLOAT32_C(  -391.34), EASYSIMD_FLOAT32_C(     2.43), EASYSIMD_FLOAT32_C(    -0.16),
                         EASYSIMD_FLOAT32_C(    -0.51), EASYSIMD_FLOAT32_C(   -38.43), EASYSIMD_FLOAT32_C(     5.03), EASYSIMD_FLOAT32_C(    -1.74),
                         EASYSIMD_FLOAT32_C(   460.12), EASYSIMD_FLOAT32_C(    -0.45), EASYSIMD_FLOAT32_C(    -1.30), EASYSIMD_FLOAT32_C(    -2.20)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   769.85), EASYSIMD_FLOAT32_C(   -75.51), EASYSIMD_FLOAT32_C(   417.80), EASYSIMD_FLOAT32_C(   327.87),
                         EASYSIMD_FLOAT32_C(   287.24), EASYSIMD_FLOAT32_C(  -627.46), EASYSIMD_FLOAT32_C(   540.48), EASYSIMD_FLOAT32_C(  -625.88),
                         EASYSIMD_FLOAT32_C(  -108.88), EASYSIMD_FLOAT32_C(   663.67), EASYSIMD_FLOAT32_C(  -412.74), EASYSIMD_FLOAT32_C(  -226.36),
                         EASYSIMD_FLOAT32_C(   757.77), EASYSIMD_FLOAT32_C(  -897.40), EASYSIMD_FLOAT32_C(    27.15), EASYSIMD_FLOAT32_C(  -443.34)),
      UINT16_C(26651),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -896.67), EASYSIMD_FLOAT32_C(  -181.49), EASYSIMD_FLOAT32_C(  -338.89), EASYSIMD_FLOAT32_C(   -19.28),
                         EASYSIMD_FLOAT32_C(   886.35), EASYSIMD_FLOAT32_C(  -662.07), EASYSIMD_FLOAT32_C(   925.60), EASYSIMD_FLOAT32_C(   651.41),
                         EASYSIMD_FLOAT32_C(   597.16), EASYSIMD_FLOAT32_C(    67.32), EASYSIMD_FLOAT32_C(  -911.68), EASYSIMD_FLOAT32_C(   202.35),
                         EASYSIMD_FLOAT32_C(   208.06), EASYSIMD_FLOAT32_C(   747.61), EASYSIMD_FLOAT32_C(    81.71), EASYSIMD_FLOAT32_C(    40.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   429.04), EASYSIMD_FLOAT32_C(    13.69), EASYSIMD_FLOAT32_C(   491.03), EASYSIMD_FLOAT32_C(   366.42),
                         EASYSIMD_FLOAT32_C(  -264.60), EASYSIMD_FLOAT32_C(   201.75), EASYSIMD_FLOAT32_C(  -598.58), EASYSIMD_FLOAT32_C(  -939.94),
                         EASYSIMD_FLOAT32_C(   118.06), EASYSIMD_FLOAT32_C(   355.92), EASYSIMD_FLOAT32_C(   978.59), EASYSIMD_FLOAT32_C(  -224.11),
                         EASYSIMD_FLOAT32_C(   -71.37), EASYSIMD_FLOAT32_C(   333.99), EASYSIMD_FLOAT32_C(  -515.40), EASYSIMD_FLOAT32_C(   -38.06)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   769.85), EASYSIMD_FLOAT32_C(   -13.26), EASYSIMD_FLOAT32_C(    -0.69), EASYSIMD_FLOAT32_C(   327.87),
                         EASYSIMD_FLOAT32_C(    -3.35), EASYSIMD_FLOAT32_C(  -627.46), EASYSIMD_FLOAT32_C(   540.48), EASYSIMD_FLOAT32_C(  -625.88),
                         EASYSIMD_FLOAT32_C(  -108.88), EASYSIMD_FLOAT32_C(   663.67), EASYSIMD_FLOAT32_C(  -412.74), EASYSIMD_FLOAT32_C(    -0.90),
                         EASYSIMD_FLOAT32_C(    -2.92), EASYSIMD_FLOAT32_C(  -897.40), EASYSIMD_FLOAT32_C(    -0.16), EASYSIMD_FLOAT32_C(    -1.07)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -301.18), EASYSIMD_FLOAT32_C(  -952.56), EASYSIMD_FLOAT32_C(   361.18), EASYSIMD_FLOAT32_C(    53.08),
                         EASYSIMD_FLOAT32_C(   179.94), EASYSIMD_FLOAT32_C(  -914.68), EASYSIMD_FLOAT32_C(  -695.32), EASYSIMD_FLOAT32_C(  -492.39),
                         EASYSIMD_FLOAT32_C(   -86.02), EASYSIMD_FLOAT32_C(   123.88), EASYSIMD_FLOAT32_C(   274.86), EASYSIMD_FLOAT32_C(   554.74),
                         EASYSIMD_FLOAT32_C(  -845.80), EASYSIMD_FLOAT32_C(  -156.28), EASYSIMD_FLOAT32_C(  -737.31), EASYSIMD_FLOAT32_C(   590.88)),
      UINT16_C(31164),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   694.79), EASYSIMD_FLOAT32_C(   687.99), EASYSIMD_FLOAT32_C(  -648.58), EASYSIMD_FLOAT32_C(  -272.14),
                         EASYSIMD_FLOAT32_C(   -52.85), EASYSIMD_FLOAT32_C(  -298.63), EASYSIMD_FLOAT32_C(   917.21), EASYSIMD_FLOAT32_C(  -876.76),
                         EASYSIMD_FLOAT32_C(   677.44), EASYSIMD_FLOAT32_C(  -857.42), EASYSIMD_FLOAT32_C(   -56.60), EASYSIMD_FLOAT32_C(   488.58),
                         EASYSIMD_FLOAT32_C(   876.79), EASYSIMD_FLOAT32_C(  -578.18), EASYSIMD_FLOAT32_C(  -335.03), EASYSIMD_FLOAT32_C(   980.62)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   247.15), EASYSIMD_FLOAT32_C(   126.77), EASYSIMD_FLOAT32_C(   867.78), EASYSIMD_FLOAT32_C(  -450.16),
                         EASYSIMD_FLOAT32_C(    94.73), EASYSIMD_FLOAT32_C(  -587.88), EASYSIMD_FLOAT32_C(   776.58), EASYSIMD_FLOAT32_C(  -595.96),
                         EASYSIMD_FLOAT32_C(   345.82), EASYSIMD_FLOAT32_C(  -768.91), EASYSIMD_FLOAT32_C(   -31.17), EASYSIMD_FLOAT32_C(    -4.10),
                         EASYSIMD_FLOAT32_C(  -234.58), EASYSIMD_FLOAT32_C(   278.53), EASYSIMD_FLOAT32_C(  -336.24), EASYSIMD_FLOAT32_C(  -974.01)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -301.18), EASYSIMD_FLOAT32_C(     5.43), EASYSIMD_FLOAT32_C(    -0.75), EASYSIMD_FLOAT32_C(     0.60),
                         EASYSIMD_FLOAT32_C(    -0.56), EASYSIMD_FLOAT32_C(  -914.68), EASYSIMD_FLOAT32_C(  -695.32), EASYSIMD_FLOAT32_C(     1.47),
                         EASYSIMD_FLOAT32_C(     1.96), EASYSIMD_FLOAT32_C(   123.88), EASYSIMD_FLOAT32_C(     1.82), EASYSIMD_FLOAT32_C(  -119.17),
                         EASYSIMD_FLOAT32_C(    -3.74), EASYSIMD_FLOAT32_C(    -2.08), EASYSIMD_FLOAT32_C(  -737.31), EASYSIMD_FLOAT32_C(   590.88)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -412.81), EASYSIMD_FLOAT32_C(  -265.44), EASYSIMD_FLOAT32_C(  -550.71), EASYSIMD_FLOAT32_C(  -725.27),
                         EASYSIMD_FLOAT32_C(  -302.34), EASYSIMD_FLOAT32_C(  -375.85), EASYSIMD_FLOAT32_C(   423.25), EASYSIMD_FLOAT32_C(   778.83),
                         EASYSIMD_FLOAT32_C(   480.23), EASYSIMD_FLOAT32_C(  -401.59), EASYSIMD_FLOAT32_C(   489.09), EASYSIMD_FLOAT32_C(   775.60),
                         EASYSIMD_FLOAT32_C(  -569.06), EASYSIMD_FLOAT32_C(  -632.55), EASYSIMD_FLOAT32_C(  -156.10), EASYSIMD_FLOAT32_C(   658.93)),
      UINT16_C( 3671),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -474.43), EASYSIMD_FLOAT32_C(  -465.91), EASYSIMD_FLOAT32_C(   545.15), EASYSIMD_FLOAT32_C(   262.71),
                         EASYSIMD_FLOAT32_C(   599.41), EASYSIMD_FLOAT32_C(  -408.02), EASYSIMD_FLOAT32_C(  -664.44), EASYSIMD_FLOAT32_C(   446.87),
                         EASYSIMD_FLOAT32_C(   816.32), EASYSIMD_FLOAT32_C(   622.16), EASYSIMD_FLOAT32_C(    40.97), EASYSIMD_FLOAT32_C(  -230.30),
                         EASYSIMD_FLOAT32_C(   122.84), EASYSIMD_FLOAT32_C(   457.98), EASYSIMD_FLOAT32_C(  -118.87), EASYSIMD_FLOAT32_C(  -211.46)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   271.75), EASYSIMD_FLOAT32_C(   598.59), EASYSIMD_FLOAT32_C(  -130.09), EASYSIMD_FLOAT32_C(   474.47),
                         EASYSIMD_FLOAT32_C(   -94.60), EASYSIMD_FLOAT32_C(   846.28), EASYSIMD_FLOAT32_C(   108.99), EASYSIMD_FLOAT32_C(  -793.22),
                         EASYSIMD_FLOAT32_C(   -12.05), EASYSIMD_FLOAT32_C(  -325.70), EASYSIMD_FLOAT32_C(  -510.95), EASYSIMD_FLOAT32_C(   213.60),
                         EASYSIMD_FLOAT32_C(  -818.29), EASYSIMD_FLOAT32_C(  -431.12), EASYSIMD_FLOAT32_C(  -186.49), EASYSIMD_FLOAT32_C(    53.27)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -412.81), EASYSIMD_FLOAT32_C(  -265.44), EASYSIMD_FLOAT32_C(  -550.71), EASYSIMD_FLOAT32_C(  -725.27),
                         EASYSIMD_FLOAT32_C(    -6.34), EASYSIMD_FLOAT32_C(    -0.48), EASYSIMD_FLOAT32_C(    -6.10), EASYSIMD_FLOAT32_C(   778.83),
                         EASYSIMD_FLOAT32_C(   480.23), EASYSIMD_FLOAT32_C(    -1.91), EASYSIMD_FLOAT32_C(   489.09), EASYSIMD_FLOAT32_C(    -1.08),
                         EASYSIMD_FLOAT32_C(  -569.06), EASYSIMD_FLOAT32_C(    -1.06), EASYSIMD_FLOAT32_C(     0.64), EASYSIMD_FLOAT32_C(    -3.97)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -254.94), EASYSIMD_FLOAT32_C(   234.45), EASYSIMD_FLOAT32_C(   235.56), EASYSIMD_FLOAT32_C(   930.35),
                         EASYSIMD_FLOAT32_C(   137.93), EASYSIMD_FLOAT32_C(   979.46), EASYSIMD_FLOAT32_C(   688.15), EASYSIMD_FLOAT32_C(   707.95),
                         EASYSIMD_FLOAT32_C(    35.42), EASYSIMD_FLOAT32_C(   748.55), EASYSIMD_FLOAT32_C(   649.98), EASYSIMD_FLOAT32_C(   702.04),
                         EASYSIMD_FLOAT32_C(   443.56), EASYSIMD_FLOAT32_C(  -944.39), EASYSIMD_FLOAT32_C(   717.51), EASYSIMD_FLOAT32_C(   716.62)),
      UINT16_C(24144),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -982.71), EASYSIMD_FLOAT32_C(   639.75), EASYSIMD_FLOAT32_C(   842.03), EASYSIMD_FLOAT32_C(   717.68),
                         EASYSIMD_FLOAT32_C(   294.25), EASYSIMD_FLOAT32_C(  -411.52), EASYSIMD_FLOAT32_C(   632.28), EASYSIMD_FLOAT32_C(   531.91),
                         EASYSIMD_FLOAT32_C(  -198.66), EASYSIMD_FLOAT32_C(   722.92), EASYSIMD_FLOAT32_C(  -890.25), EASYSIMD_FLOAT32_C(   -36.77),
                         EASYSIMD_FLOAT32_C(  -651.17), EASYSIMD_FLOAT32_C(   559.24), EASYSIMD_FLOAT32_C(   496.39), EASYSIMD_FLOAT32_C(  -143.68)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -779.19), EASYSIMD_FLOAT32_C(   587.24), EASYSIMD_FLOAT32_C(   850.25), EASYSIMD_FLOAT32_C(   172.75),
                         EASYSIMD_FLOAT32_C(   237.73), EASYSIMD_FLOAT32_C(   792.79), EASYSIMD_FLOAT32_C(  -225.26), EASYSIMD_FLOAT32_C(   810.16),
                         EASYSIMD_FLOAT32_C(   235.61), EASYSIMD_FLOAT32_C(   123.68), EASYSIMD_FLOAT32_C(  -869.51), EASYSIMD_FLOAT32_C(   811.23),
                         EASYSIMD_FLOAT32_C(   292.28), EASYSIMD_FLOAT32_C(   158.60), EASYSIMD_FLOAT32_C(  -861.10), EASYSIMD_FLOAT32_C(   297.31)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -254.94), EASYSIMD_FLOAT32_C(     1.09), EASYSIMD_FLOAT32_C(   235.56), EASYSIMD_FLOAT32_C(     4.15),
                         EASYSIMD_FLOAT32_C(     1.24), EASYSIMD_FLOAT32_C(    -0.52), EASYSIMD_FLOAT32_C(    -2.81), EASYSIMD_FLOAT32_C(   707.95),
                         EASYSIMD_FLOAT32_C(    35.42), EASYSIMD_FLOAT32_C(     5.85), EASYSIMD_FLOAT32_C(   649.98), EASYSIMD_FLOAT32_C(    -0.05),
                         EASYSIMD_FLOAT32_C(   443.56), EASYSIMD_FLOAT32_C(  -944.39), EASYSIMD_FLOAT32_C(   717.51), EASYSIMD_FLOAT32_C(   716.62)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   711.46), EASYSIMD_FLOAT32_C(  -417.42), EASYSIMD_FLOAT32_C(  -736.14), EASYSIMD_FLOAT32_C(  -654.73),
                         EASYSIMD_FLOAT32_C(  -297.59), EASYSIMD_FLOAT32_C(   899.88), EASYSIMD_FLOAT32_C(   819.21), EASYSIMD_FLOAT32_C(  -451.55),
                         EASYSIMD_FLOAT32_C(   831.09), EASYSIMD_FLOAT32_C(   694.55), EASYSIMD_FLOAT32_C(  -231.88), EASYSIMD_FLOAT32_C(  -711.25),
                         EASYSIMD_FLOAT32_C(  -213.96), EASYSIMD_FLOAT32_C(  -411.84), EASYSIMD_FLOAT32_C(  -325.79), EASYSIMD_FLOAT32_C(  -424.22)),
      UINT16_C( 4465),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   985.56), EASYSIMD_FLOAT32_C(  -969.44), EASYSIMD_FLOAT32_C(   -91.63), EASYSIMD_FLOAT32_C(  -416.19),
                         EASYSIMD_FLOAT32_C(   716.00), EASYSIMD_FLOAT32_C(   579.33), EASYSIMD_FLOAT32_C(   678.78), EASYSIMD_FLOAT32_C(   650.46),
                         EASYSIMD_FLOAT32_C(  -988.30), EASYSIMD_FLOAT32_C(   206.47), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -226.18),
                         EASYSIMD_FLOAT32_C(  -410.63), EASYSIMD_FLOAT32_C(  -238.02), EASYSIMD_FLOAT32_C(   520.82), EASYSIMD_FLOAT32_C(  -882.63)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   373.48), EASYSIMD_FLOAT32_C(  -376.14), EASYSIMD_FLOAT32_C(   103.99), EASYSIMD_FLOAT32_C(   900.82),
                         EASYSIMD_FLOAT32_C(   827.14), EASYSIMD_FLOAT32_C(   -50.15), EASYSIMD_FLOAT32_C(   675.06), EASYSIMD_FLOAT32_C(   239.90),
                         EASYSIMD_FLOAT32_C(   531.97), EASYSIMD_FLOAT32_C(    52.69), EASYSIMD_FLOAT32_C(  -376.06), EASYSIMD_FLOAT32_C(  -290.42),
                         EASYSIMD_FLOAT32_C(  -325.12), EASYSIMD_FLOAT32_C(  -471.17), EASYSIMD_FLOAT32_C(  -511.21), EASYSIMD_FLOAT32_C(   -90.11)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   711.46), EASYSIMD_FLOAT32_C(  -417.42), EASYSIMD_FLOAT32_C(  -736.14), EASYSIMD_FLOAT32_C(    -0.46),
                         EASYSIMD_FLOAT32_C(  -297.59), EASYSIMD_FLOAT32_C(   899.88), EASYSIMD_FLOAT32_C(   819.21), EASYSIMD_FLOAT32_C(     2.71),
                         EASYSIMD_FLOAT32_C(   831.09), EASYSIMD_FLOAT32_C(     3.92), EASYSIMD_FLOAT32_C(    -0.57), EASYSIMD_FLOAT32_C(     0.78),
                         EASYSIMD_FLOAT32_C(  -213.96), EASYSIMD_FLOAT32_C(  -411.84), EASYSIMD_FLOAT32_C(  -325.79), EASYSIMD_FLOAT32_C(     9.80)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -788.99), EASYSIMD_FLOAT32_C(   888.94), EASYSIMD_FLOAT32_C(   861.99), EASYSIMD_FLOAT32_C(  -655.94),
                         EASYSIMD_FLOAT32_C(  -815.78), EASYSIMD_FLOAT32_C(   460.30), EASYSIMD_FLOAT32_C(  -596.09), EASYSIMD_FLOAT32_C(   480.08),
                         EASYSIMD_FLOAT32_C(  -800.23), EASYSIMD_FLOAT32_C(  -511.53), EASYSIMD_FLOAT32_C(   235.71), EASYSIMD_FLOAT32_C(   833.52),
                         EASYSIMD_FLOAT32_C(   343.49), EASYSIMD_FLOAT32_C(   413.97), EASYSIMD_FLOAT32_C(   264.73), EASYSIMD_FLOAT32_C(   769.22)),
      UINT16_C(57880),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -782.73), EASYSIMD_FLOAT32_C(   -41.33), EASYSIMD_FLOAT32_C(   183.64), EASYSIMD_FLOAT32_C(     0.86),
                         EASYSIMD_FLOAT32_C(  -449.70), EASYSIMD_FLOAT32_C(   153.64), EASYSIMD_FLOAT32_C(   543.55), EASYSIMD_FLOAT32_C(  -321.17),
                         EASYSIMD_FLOAT32_C(   944.46), EASYSIMD_FLOAT32_C(  -863.15), EASYSIMD_FLOAT32_C(   155.57), EASYSIMD_FLOAT32_C(   671.09),
                         EASYSIMD_FLOAT32_C(   138.46), EASYSIMD_FLOAT32_C(   937.90), EASYSIMD_FLOAT32_C(   367.36), EASYSIMD_FLOAT32_C(  -187.79)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -177.92), EASYSIMD_FLOAT32_C(   382.46), EASYSIMD_FLOAT32_C(  -344.53), EASYSIMD_FLOAT32_C(   306.51),
                         EASYSIMD_FLOAT32_C(   804.79), EASYSIMD_FLOAT32_C(    74.50), EASYSIMD_FLOAT32_C(  -171.92), EASYSIMD_FLOAT32_C(  -865.07),
                         EASYSIMD_FLOAT32_C(   788.06), EASYSIMD_FLOAT32_C(  -723.82), EASYSIMD_FLOAT32_C(    43.98), EASYSIMD_FLOAT32_C(  -303.25),
                         EASYSIMD_FLOAT32_C(  -511.21), EASYSIMD_FLOAT32_C(   460.56), EASYSIMD_FLOAT32_C(   217.57), EASYSIMD_FLOAT32_C(  -900.02)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     4.40), EASYSIMD_FLOAT32_C(    -0.11), EASYSIMD_FLOAT32_C(    -0.53), EASYSIMD_FLOAT32_C(  -655.94),
                         EASYSIMD_FLOAT32_C(  -815.78), EASYSIMD_FLOAT32_C(   460.30), EASYSIMD_FLOAT32_C(    -3.16), EASYSIMD_FLOAT32_C(   480.08),
                         EASYSIMD_FLOAT32_C(  -800.23), EASYSIMD_FLOAT32_C(  -511.53), EASYSIMD_FLOAT32_C(   235.71), EASYSIMD_FLOAT32_C(    -2.21),
                         EASYSIMD_FLOAT32_C(    -0.27), EASYSIMD_FLOAT32_C(   413.97), EASYSIMD_FLOAT32_C(   264.73), EASYSIMD_FLOAT32_C(   769.22)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    54.65), EASYSIMD_FLOAT32_C(   276.51), EASYSIMD_FLOAT32_C(   227.98), EASYSIMD_FLOAT32_C(  -128.31),
                         EASYSIMD_FLOAT32_C(  -191.48), EASYSIMD_FLOAT32_C(   348.63), EASYSIMD_FLOAT32_C(   444.48), EASYSIMD_FLOAT32_C(   206.11),
                         EASYSIMD_FLOAT32_C(  -692.44), EASYSIMD_FLOAT32_C(  -865.72), EASYSIMD_FLOAT32_C(   763.64), EASYSIMD_FLOAT32_C(  -849.66),
                         EASYSIMD_FLOAT32_C(   804.26), EASYSIMD_FLOAT32_C(   570.08), EASYSIMD_FLOAT32_C(   125.91), EASYSIMD_FLOAT32_C(   149.60)),
      UINT16_C(24771),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   747.34), EASYSIMD_FLOAT32_C(   607.83), EASYSIMD_FLOAT32_C(    25.24), EASYSIMD_FLOAT32_C(  -542.52),
                         EASYSIMD_FLOAT32_C(   568.70), EASYSIMD_FLOAT32_C(   899.42), EASYSIMD_FLOAT32_C(   120.86), EASYSIMD_FLOAT32_C(  -424.59),
                         EASYSIMD_FLOAT32_C(   377.13), EASYSIMD_FLOAT32_C(   761.91), EASYSIMD_FLOAT32_C(  -902.23), EASYSIMD_FLOAT32_C(  -759.84),
                         EASYSIMD_FLOAT32_C(   430.99), EASYSIMD_FLOAT32_C(   555.32), EASYSIMD_FLOAT32_C(  -397.14), EASYSIMD_FLOAT32_C(   608.52)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -615.94), EASYSIMD_FLOAT32_C(   113.16), EASYSIMD_FLOAT32_C(    26.04), EASYSIMD_FLOAT32_C(  -142.02),
                         EASYSIMD_FLOAT32_C(   273.49), EASYSIMD_FLOAT32_C(   374.88), EASYSIMD_FLOAT32_C(   453.99), EASYSIMD_FLOAT32_C(  -241.36),
                         EASYSIMD_FLOAT32_C(   181.97), EASYSIMD_FLOAT32_C(   143.35), EASYSIMD_FLOAT32_C(   400.04), EASYSIMD_FLOAT32_C(   610.27),
                         EASYSIMD_FLOAT32_C(  -726.06), EASYSIMD_FLOAT32_C(  -819.96), EASYSIMD_FLOAT32_C(   674.91), EASYSIMD_FLOAT32_C(   406.86)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    54.65), EASYSIMD_FLOAT32_C(     5.37), EASYSIMD_FLOAT32_C(     0.97), EASYSIMD_FLOAT32_C(  -128.31),
                         EASYSIMD_FLOAT32_C(  -191.48), EASYSIMD_FLOAT32_C(   348.63), EASYSIMD_FLOAT32_C(   444.48), EASYSIMD_FLOAT32_C(   206.11),
                         EASYSIMD_FLOAT32_C(     2.07), EASYSIMD_FLOAT32_C(     5.32), EASYSIMD_FLOAT32_C(   763.64), EASYSIMD_FLOAT32_C(  -849.66),
                         EASYSIMD_FLOAT32_C(   804.26), EASYSIMD_FLOAT32_C(   570.08), EASYSIMD_FLOAT32_C(    -0.59), EASYSIMD_FLOAT32_C(     1.50)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_div_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_div_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_div_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { UINT16_C(32824),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   745.69), EASYSIMD_FLOAT32_C(  -258.59), EASYSIMD_FLOAT32_C(  -549.06), EASYSIMD_FLOAT32_C(   646.98),
                         EASYSIMD_FLOAT32_C(   925.86), EASYSIMD_FLOAT32_C(   378.90), EASYSIMD_FLOAT32_C(  -524.10), EASYSIMD_FLOAT32_C(  -563.31),
                         EASYSIMD_FLOAT32_C(   112.08), EASYSIMD_FLOAT32_C(   712.48), EASYSIMD_FLOAT32_C(  -754.71), EASYSIMD_FLOAT32_C(   256.61),
                         EASYSIMD_FLOAT32_C(   768.73), EASYSIMD_FLOAT32_C(   227.99), EASYSIMD_FLOAT32_C(   174.97), EASYSIMD_FLOAT32_C(   338.39)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   841.82), EASYSIMD_FLOAT32_C(  -330.14), EASYSIMD_FLOAT32_C(  -498.05), EASYSIMD_FLOAT32_C(  -706.46),
                         EASYSIMD_FLOAT32_C(  -284.71), EASYSIMD_FLOAT32_C(  -940.98), EASYSIMD_FLOAT32_C(  -491.84), EASYSIMD_FLOAT32_C(    52.49),
                         EASYSIMD_FLOAT32_C(   759.92), EASYSIMD_FLOAT32_C(   629.58), EASYSIMD_FLOAT32_C(    23.76), EASYSIMD_FLOAT32_C(   980.95),
                         EASYSIMD_FLOAT32_C(   224.97), EASYSIMD_FLOAT32_C(   818.07), EASYSIMD_FLOAT32_C(  -531.75), EASYSIMD_FLOAT32_C(  -531.67)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -31.76), EASYSIMD_FLOAT32_C(     0.26),
                         EASYSIMD_FLOAT32_C(     3.42), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C( 4283),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   838.22), EASYSIMD_FLOAT32_C(   464.78), EASYSIMD_FLOAT32_C(  -248.37), EASYSIMD_FLOAT32_C(    28.49),
                         EASYSIMD_FLOAT32_C(  -176.67), EASYSIMD_FLOAT32_C(  -468.39), EASYSIMD_FLOAT32_C(  -893.30), EASYSIMD_FLOAT32_C(   771.96),
                         EASYSIMD_FLOAT32_C(  -167.30), EASYSIMD_FLOAT32_C(  -738.71), EASYSIMD_FLOAT32_C(  -816.67), EASYSIMD_FLOAT32_C(    43.31),
                         EASYSIMD_FLOAT32_C(   -98.40), EASYSIMD_FLOAT32_C(   217.89), EASYSIMD_FLOAT32_C(   626.98), EASYSIMD_FLOAT32_C(  -409.09)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -156.59), EASYSIMD_FLOAT32_C(   153.69), EASYSIMD_FLOAT32_C(   895.38), EASYSIMD_FLOAT32_C(  -242.63),
                         EASYSIMD_FLOAT32_C(   994.17), EASYSIMD_FLOAT32_C(  -265.23), EASYSIMD_FLOAT32_C(   -57.91), EASYSIMD_FLOAT32_C(  -586.11),
                         EASYSIMD_FLOAT32_C(  -443.71), EASYSIMD_FLOAT32_C(  -786.78), EASYSIMD_FLOAT32_C(   -92.41), EASYSIMD_FLOAT32_C(  -378.62),
                         EASYSIMD_FLOAT32_C(   632.49), EASYSIMD_FLOAT32_C(  -867.20), EASYSIMD_FLOAT32_C(   977.79), EASYSIMD_FLOAT32_C(  -788.71)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.12),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     8.84), EASYSIMD_FLOAT32_C(    -0.11),
                         EASYSIMD_FLOAT32_C(    -0.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.64), EASYSIMD_FLOAT32_C(     0.52)) },
    { UINT16_C(27708),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -309.30), EASYSIMD_FLOAT32_C(  -478.69), EASYSIMD_FLOAT32_C(  -499.66), EASYSIMD_FLOAT32_C(  -834.97),
                         EASYSIMD_FLOAT32_C(  -926.76), EASYSIMD_FLOAT32_C(   306.74), EASYSIMD_FLOAT32_C(   350.68), EASYSIMD_FLOAT32_C(   698.74),
                         EASYSIMD_FLOAT32_C(  -748.23), EASYSIMD_FLOAT32_C(   960.31), EASYSIMD_FLOAT32_C(   -52.56), EASYSIMD_FLOAT32_C(   -18.49),
                         EASYSIMD_FLOAT32_C(  -174.79), EASYSIMD_FLOAT32_C(  -875.70), EASYSIMD_FLOAT32_C(   270.45), EASYSIMD_FLOAT32_C(   571.57)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -891.46), EASYSIMD_FLOAT32_C(   298.88), EASYSIMD_FLOAT32_C(   907.92), EASYSIMD_FLOAT32_C(   585.94),
                         EASYSIMD_FLOAT32_C(   976.00), EASYSIMD_FLOAT32_C(   860.60), EASYSIMD_FLOAT32_C(  -807.57), EASYSIMD_FLOAT32_C(  -501.53),
                         EASYSIMD_FLOAT32_C(   887.26), EASYSIMD_FLOAT32_C(  -380.63), EASYSIMD_FLOAT32_C(   603.15), EASYSIMD_FLOAT32_C(   906.17),
                         EASYSIMD_FLOAT32_C(  -446.90), EASYSIMD_FLOAT32_C(   518.96), EASYSIMD_FLOAT32_C(   325.09), EASYSIMD_FLOAT32_C(   394.29)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.60), EASYSIMD_FLOAT32_C(    -0.55), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    -0.95), EASYSIMD_FLOAT32_C(     0.36), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.09), EASYSIMD_FLOAT32_C(    -0.02),
                         EASYSIMD_FLOAT32_C(     0.39), EASYSIMD_FLOAT32_C(    -1.69), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(21979),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -722.04), EASYSIMD_FLOAT32_C(  -251.19), EASYSIMD_FLOAT32_C(   885.20), EASYSIMD_FLOAT32_C(  -718.95),
                         EASYSIMD_FLOAT32_C(  -995.18), EASYSIMD_FLOAT32_C(   316.41), EASYSIMD_FLOAT32_C(   425.49), EASYSIMD_FLOAT32_C(  -889.60),
                         EASYSIMD_FLOAT32_C(  -764.37), EASYSIMD_FLOAT32_C(  -698.84), EASYSIMD_FLOAT32_C(   111.54), EASYSIMD_FLOAT32_C(   627.05),
                         EASYSIMD_FLOAT32_C(   619.20), EASYSIMD_FLOAT32_C(   107.79), EASYSIMD_FLOAT32_C(   830.07), EASYSIMD_FLOAT32_C(  -991.50)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    -9.32), EASYSIMD_FLOAT32_C(   588.15), EASYSIMD_FLOAT32_C(   740.36), EASYSIMD_FLOAT32_C(   589.64),
                         EASYSIMD_FLOAT32_C(  -146.10), EASYSIMD_FLOAT32_C(   771.62), EASYSIMD_FLOAT32_C(  -975.31), EASYSIMD_FLOAT32_C(   550.04),
                         EASYSIMD_FLOAT32_C(   902.97), EASYSIMD_FLOAT32_C(  -970.67), EASYSIMD_FLOAT32_C(  -396.71), EASYSIMD_FLOAT32_C(   740.42),
                         EASYSIMD_FLOAT32_C(  -740.07), EASYSIMD_FLOAT32_C(   691.95), EASYSIMD_FLOAT32_C(  -434.89), EASYSIMD_FLOAT32_C(   270.74)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.43), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.22),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.41), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.62),
                         EASYSIMD_FLOAT32_C(    -0.85), EASYSIMD_FLOAT32_C(     0.72), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.85),
                         EASYSIMD_FLOAT32_C(    -0.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.91), EASYSIMD_FLOAT32_C(    -3.66)) },
    { UINT16_C( 1193),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   347.59), EASYSIMD_FLOAT32_C(   162.65), EASYSIMD_FLOAT32_C(   724.10), EASYSIMD_FLOAT32_C(   124.00),
                         EASYSIMD_FLOAT32_C(  -823.97), EASYSIMD_FLOAT32_C(  -185.15), EASYSIMD_FLOAT32_C(    33.85), EASYSIMD_FLOAT32_C(  -430.54),
                         EASYSIMD_FLOAT32_C(  -534.02), EASYSIMD_FLOAT32_C(   815.29), EASYSIMD_FLOAT32_C(   942.25), EASYSIMD_FLOAT32_C(  -825.08),
                         EASYSIMD_FLOAT32_C(   638.03), EASYSIMD_FLOAT32_C(   599.07), EASYSIMD_FLOAT32_C(   164.45), EASYSIMD_FLOAT32_C(   429.94)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -817.35), EASYSIMD_FLOAT32_C(  -889.89), EASYSIMD_FLOAT32_C(   528.79), EASYSIMD_FLOAT32_C(  -600.85),
                         EASYSIMD_FLOAT32_C(  -168.12), EASYSIMD_FLOAT32_C(  -798.12), EASYSIMD_FLOAT32_C(  -637.75), EASYSIMD_FLOAT32_C(  -580.73),
                         EASYSIMD_FLOAT32_C(   697.23), EASYSIMD_FLOAT32_C(   654.25), EASYSIMD_FLOAT32_C(  -236.09), EASYSIMD_FLOAT32_C(   234.13),
                         EASYSIMD_FLOAT32_C(  -696.60), EASYSIMD_FLOAT32_C(  -486.03), EASYSIMD_FLOAT32_C(    69.79), EASYSIMD_FLOAT32_C(   435.18)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.23), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    -0.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -3.99), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    -0.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.99)) },
    { UINT16_C(47777),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    96.65), EASYSIMD_FLOAT32_C(   -38.76), EASYSIMD_FLOAT32_C(   585.22), EASYSIMD_FLOAT32_C(  -683.52),
                         EASYSIMD_FLOAT32_C(   268.64), EASYSIMD_FLOAT32_C(  -393.28), EASYSIMD_FLOAT32_C(   102.94), EASYSIMD_FLOAT32_C(   786.82),
                         EASYSIMD_FLOAT32_C(   138.90), EASYSIMD_FLOAT32_C(   225.78), EASYSIMD_FLOAT32_C(   449.88), EASYSIMD_FLOAT32_C(   347.32),
                         EASYSIMD_FLOAT32_C(    33.80), EASYSIMD_FLOAT32_C(  -559.14), EASYSIMD_FLOAT32_C(  -159.05), EASYSIMD_FLOAT32_C(  -491.42)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -55.95), EASYSIMD_FLOAT32_C(  -837.50), EASYSIMD_FLOAT32_C(  -575.23), EASYSIMD_FLOAT32_C(   248.03),
                         EASYSIMD_FLOAT32_C(   907.04), EASYSIMD_FLOAT32_C(   -74.96), EASYSIMD_FLOAT32_C(  -821.80), EASYSIMD_FLOAT32_C(  -847.93),
                         EASYSIMD_FLOAT32_C(  -925.94), EASYSIMD_FLOAT32_C(   664.01), EASYSIMD_FLOAT32_C(  -745.59), EASYSIMD_FLOAT32_C(  -301.31),
                         EASYSIMD_FLOAT32_C(   146.53), EASYSIMD_FLOAT32_C(  -440.81), EASYSIMD_FLOAT32_C(   427.27), EASYSIMD_FLOAT32_C(  -219.59)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    -1.73), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.02), EASYSIMD_FLOAT32_C(    -2.76),
                         EASYSIMD_FLOAT32_C(     0.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.13), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    -0.15), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.60), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.24)) },
    { UINT16_C(50336),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -724.66), EASYSIMD_FLOAT32_C(  -778.29), EASYSIMD_FLOAT32_C(  -888.11), EASYSIMD_FLOAT32_C(  -623.31),
                         EASYSIMD_FLOAT32_C(   617.08), EASYSIMD_FLOAT32_C(    42.91), EASYSIMD_FLOAT32_C(   907.40), EASYSIMD_FLOAT32_C(  -402.88),
                         EASYSIMD_FLOAT32_C(  -278.23), EASYSIMD_FLOAT32_C(  -640.08), EASYSIMD_FLOAT32_C(   108.85), EASYSIMD_FLOAT32_C(  -527.72),
                         EASYSIMD_FLOAT32_C(  -791.82), EASYSIMD_FLOAT32_C(  -207.31), EASYSIMD_FLOAT32_C(  -642.88), EASYSIMD_FLOAT32_C(   536.44)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   766.31), EASYSIMD_FLOAT32_C(   592.01), EASYSIMD_FLOAT32_C(   324.90), EASYSIMD_FLOAT32_C(    55.55),
                         EASYSIMD_FLOAT32_C(   -34.13), EASYSIMD_FLOAT32_C(  -588.88), EASYSIMD_FLOAT32_C(   991.78), EASYSIMD_FLOAT32_C(  -468.91),
                         EASYSIMD_FLOAT32_C(    78.86), EASYSIMD_FLOAT32_C(    18.25), EASYSIMD_FLOAT32_C(   295.51), EASYSIMD_FLOAT32_C(  -293.26),
                         EASYSIMD_FLOAT32_C(  -877.24), EASYSIMD_FLOAT32_C(   952.33), EASYSIMD_FLOAT32_C(  -274.18), EASYSIMD_FLOAT32_C(   654.17)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    -0.95), EASYSIMD_FLOAT32_C(    -1.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.07), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    -3.53), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.37), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(  740),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -222.38), EASYSIMD_FLOAT32_C(  -847.72), EASYSIMD_FLOAT32_C(  -497.04), EASYSIMD_FLOAT32_C(   862.38),
                         EASYSIMD_FLOAT32_C(  -840.40), EASYSIMD_FLOAT32_C(   998.10), EASYSIMD_FLOAT32_C(  -257.93), EASYSIMD_FLOAT32_C(  -204.46),
                         EASYSIMD_FLOAT32_C(  -373.11), EASYSIMD_FLOAT32_C(  -912.42), EASYSIMD_FLOAT32_C(   207.13), EASYSIMD_FLOAT32_C(   784.69),
                         EASYSIMD_FLOAT32_C(    82.66), EASYSIMD_FLOAT32_C(   123.09), EASYSIMD_FLOAT32_C(  -384.17), EASYSIMD_FLOAT32_C(  -845.08)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   839.49), EASYSIMD_FLOAT32_C(  -285.39), EASYSIMD_FLOAT32_C(  -548.76), EASYSIMD_FLOAT32_C(   -35.10),
                         EASYSIMD_FLOAT32_C(  -295.04), EASYSIMD_FLOAT32_C(   738.77), EASYSIMD_FLOAT32_C(   340.04), EASYSIMD_FLOAT32_C(  -585.87),
                         EASYSIMD_FLOAT32_C(  -711.46), EASYSIMD_FLOAT32_C(   926.37), EASYSIMD_FLOAT32_C(   696.23), EASYSIMD_FLOAT32_C(   766.17),
                         EASYSIMD_FLOAT32_C(  -330.24), EASYSIMD_FLOAT32_C(   369.18), EASYSIMD_FLOAT32_C(  -498.71), EASYSIMD_FLOAT32_C(  -288.61)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.76), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.52), EASYSIMD_FLOAT32_C(    -0.98), EASYSIMD_FLOAT32_C(     0.30), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_div_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_div_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_div_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -97.83), EASYSIMD_FLOAT64_C( -393.82),
                         EASYSIMD_FLOAT64_C(  934.81), EASYSIMD_FLOAT64_C(   74.53),
                         EASYSIMD_FLOAT64_C(  843.79), EASYSIMD_FLOAT64_C(  465.05),
                         EASYSIMD_FLOAT64_C(  -42.07), EASYSIMD_FLOAT64_C( -685.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  296.78), EASYSIMD_FLOAT64_C( -698.78),
                         EASYSIMD_FLOAT64_C(  908.33), EASYSIMD_FLOAT64_C(  181.85),
                         EASYSIMD_FLOAT64_C( -397.89), EASYSIMD_FLOAT64_C( -586.75),
                         EASYSIMD_FLOAT64_C(  904.99), EASYSIMD_FLOAT64_C( -321.15)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -0.33), EASYSIMD_FLOAT64_C(    0.56),
                         EASYSIMD_FLOAT64_C(    1.03), EASYSIMD_FLOAT64_C(    0.41),
                         EASYSIMD_FLOAT64_C(   -2.12), EASYSIMD_FLOAT64_C(   -0.79),
                         EASYSIMD_FLOAT64_C(   -0.05), EASYSIMD_FLOAT64_C(    2.14)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  653.62), EASYSIMD_FLOAT64_C(  981.74),
                         EASYSIMD_FLOAT64_C(  780.10), EASYSIMD_FLOAT64_C(   59.38),
                         EASYSIMD_FLOAT64_C( -795.11), EASYSIMD_FLOAT64_C(  923.87),
                         EASYSIMD_FLOAT64_C( -270.01), EASYSIMD_FLOAT64_C( -411.99)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  596.54), EASYSIMD_FLOAT64_C( -116.40),
                         EASYSIMD_FLOAT64_C( -989.77), EASYSIMD_FLOAT64_C( -794.40),
                         EASYSIMD_FLOAT64_C(  183.38), EASYSIMD_FLOAT64_C( -185.75),
                         EASYSIMD_FLOAT64_C(  429.70), EASYSIMD_FLOAT64_C(  664.04)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    1.10), EASYSIMD_FLOAT64_C(   -8.43),
                         EASYSIMD_FLOAT64_C(   -0.79), EASYSIMD_FLOAT64_C(   -0.07),
                         EASYSIMD_FLOAT64_C(   -4.34), EASYSIMD_FLOAT64_C(   -4.97),
                         EASYSIMD_FLOAT64_C(   -0.63), EASYSIMD_FLOAT64_C(   -0.62)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -571.10), EASYSIMD_FLOAT64_C(  971.96),
                         EASYSIMD_FLOAT64_C(  149.38), EASYSIMD_FLOAT64_C(  497.71),
                         EASYSIMD_FLOAT64_C(  988.69), EASYSIMD_FLOAT64_C(  479.68),
                         EASYSIMD_FLOAT64_C( -128.24), EASYSIMD_FLOAT64_C(  585.28)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -628.68), EASYSIMD_FLOAT64_C(  938.60),
                         EASYSIMD_FLOAT64_C( -147.98), EASYSIMD_FLOAT64_C(  378.31),
                         EASYSIMD_FLOAT64_C(  246.47), EASYSIMD_FLOAT64_C(  109.18),
                         EASYSIMD_FLOAT64_C( -575.64), EASYSIMD_FLOAT64_C( -426.86)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.91), EASYSIMD_FLOAT64_C(    1.04),
                         EASYSIMD_FLOAT64_C(   -1.01), EASYSIMD_FLOAT64_C(    1.32),
                         EASYSIMD_FLOAT64_C(    4.01), EASYSIMD_FLOAT64_C(    4.39),
                         EASYSIMD_FLOAT64_C(    0.22), EASYSIMD_FLOAT64_C(   -1.37)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  729.63), EASYSIMD_FLOAT64_C( -908.06),
                         EASYSIMD_FLOAT64_C( -769.77), EASYSIMD_FLOAT64_C(  -70.66),
                         EASYSIMD_FLOAT64_C(  482.71), EASYSIMD_FLOAT64_C(  244.66),
                         EASYSIMD_FLOAT64_C( -615.83), EASYSIMD_FLOAT64_C(  841.42)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  359.65), EASYSIMD_FLOAT64_C( -730.08),
                         EASYSIMD_FLOAT64_C(  977.98), EASYSIMD_FLOAT64_C( -215.53),
                         EASYSIMD_FLOAT64_C( -315.50), EASYSIMD_FLOAT64_C(   80.64),
                         EASYSIMD_FLOAT64_C( -996.10), EASYSIMD_FLOAT64_C( -556.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.03), EASYSIMD_FLOAT64_C(    1.24),
                         EASYSIMD_FLOAT64_C(   -0.79), EASYSIMD_FLOAT64_C(    0.33),
                         EASYSIMD_FLOAT64_C(   -1.53), EASYSIMD_FLOAT64_C(    3.03),
                         EASYSIMD_FLOAT64_C(    0.62), EASYSIMD_FLOAT64_C(   -1.51)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  438.03), EASYSIMD_FLOAT64_C( -458.01),
                         EASYSIMD_FLOAT64_C(  144.59), EASYSIMD_FLOAT64_C(  165.00),
                         EASYSIMD_FLOAT64_C( -331.04), EASYSIMD_FLOAT64_C(  406.96),
                         EASYSIMD_FLOAT64_C( -326.43), EASYSIMD_FLOAT64_C(  373.82)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  304.30), EASYSIMD_FLOAT64_C( -777.07),
                         EASYSIMD_FLOAT64_C( -683.73), EASYSIMD_FLOAT64_C( -113.32),
                         EASYSIMD_FLOAT64_C( -701.16), EASYSIMD_FLOAT64_C( -942.92),
                         EASYSIMD_FLOAT64_C( -489.97), EASYSIMD_FLOAT64_C(  911.34)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    1.44), EASYSIMD_FLOAT64_C(    0.59),
                         EASYSIMD_FLOAT64_C(   -0.21), EASYSIMD_FLOAT64_C(   -1.46),
                         EASYSIMD_FLOAT64_C(    0.47), EASYSIMD_FLOAT64_C(   -0.43),
                         EASYSIMD_FLOAT64_C(    0.67), EASYSIMD_FLOAT64_C(    0.41)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -148.70), EASYSIMD_FLOAT64_C( -327.17),
                         EASYSIMD_FLOAT64_C( -310.14), EASYSIMD_FLOAT64_C( -718.80),
                         EASYSIMD_FLOAT64_C(  382.69), EASYSIMD_FLOAT64_C( -181.61),
                         EASYSIMD_FLOAT64_C( -214.09), EASYSIMD_FLOAT64_C(   55.72)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  791.83), EASYSIMD_FLOAT64_C(  191.69),
                         EASYSIMD_FLOAT64_C( -460.58), EASYSIMD_FLOAT64_C( -915.08),
                         EASYSIMD_FLOAT64_C( -877.38), EASYSIMD_FLOAT64_C( -915.27),
                         EASYSIMD_FLOAT64_C(  207.85), EASYSIMD_FLOAT64_C(  567.35)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -0.19), EASYSIMD_FLOAT64_C(   -1.71),
                         EASYSIMD_FLOAT64_C(    0.67), EASYSIMD_FLOAT64_C(    0.79),
                         EASYSIMD_FLOAT64_C(   -0.44), EASYSIMD_FLOAT64_C(    0.20),
                         EASYSIMD_FLOAT64_C(   -1.03), EASYSIMD_FLOAT64_C(    0.10)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -798.08), EASYSIMD_FLOAT64_C(  256.40),
                         EASYSIMD_FLOAT64_C(  739.89), EASYSIMD_FLOAT64_C( -903.46),
                         EASYSIMD_FLOAT64_C( -771.75), EASYSIMD_FLOAT64_C(  -54.77),
                         EASYSIMD_FLOAT64_C(  397.04), EASYSIMD_FLOAT64_C(  925.94)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  864.29), EASYSIMD_FLOAT64_C( -834.00),
                         EASYSIMD_FLOAT64_C(  475.52), EASYSIMD_FLOAT64_C(  502.31),
                         EASYSIMD_FLOAT64_C( -746.87), EASYSIMD_FLOAT64_C( -364.10),
                         EASYSIMD_FLOAT64_C( -995.18), EASYSIMD_FLOAT64_C(  683.54)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -0.92), EASYSIMD_FLOAT64_C(   -0.31),
                         EASYSIMD_FLOAT64_C(    1.56), EASYSIMD_FLOAT64_C(   -1.80),
                         EASYSIMD_FLOAT64_C(    1.03), EASYSIMD_FLOAT64_C(    0.15),
                         EASYSIMD_FLOAT64_C(   -0.40), EASYSIMD_FLOAT64_C(    1.35)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -869.58), EASYSIMD_FLOAT64_C(  763.75),
                         EASYSIMD_FLOAT64_C( -558.93), EASYSIMD_FLOAT64_C(  756.19),
                         EASYSIMD_FLOAT64_C(  509.82), EASYSIMD_FLOAT64_C( -855.71),
                         EASYSIMD_FLOAT64_C( -965.40), EASYSIMD_FLOAT64_C( -279.29)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -355.51), EASYSIMD_FLOAT64_C(  136.73),
                         EASYSIMD_FLOAT64_C(  586.70), EASYSIMD_FLOAT64_C(  712.56),
                         EASYSIMD_FLOAT64_C(  135.88), EASYSIMD_FLOAT64_C( -693.91),
                         EASYSIMD_FLOAT64_C( -131.33), EASYSIMD_FLOAT64_C( -933.79)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.45), EASYSIMD_FLOAT64_C(    5.59),
                         EASYSIMD_FLOAT64_C(   -0.95), EASYSIMD_FLOAT64_C(    1.06),
                         EASYSIMD_FLOAT64_C(    3.75), EASYSIMD_FLOAT64_C(    1.23),
                         EASYSIMD_FLOAT64_C(    7.35), EASYSIMD_FLOAT64_C(    0.30)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_div_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_div_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_div_round_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  static const struct {
    easysimd_float64 a[8];
    easysimd_float64 b[8];
    easysimd_float64 nearest_inf[8];
    easysimd_float64 neg_inf[8];
    easysimd_float64 pos_inf[8];
    easysimd_float64 zero[8];
    easysimd_float64 direction[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   496.80), EASYSIMD_FLOAT64_C(  -347.16), EASYSIMD_FLOAT64_C(   595.80), EASYSIMD_FLOAT64_C(   118.07),
        EASYSIMD_FLOAT64_C(   894.47), EASYSIMD_FLOAT64_C(    45.00), EASYSIMD_FLOAT64_C(  -612.57), EASYSIMD_FLOAT64_C(  -852.94) },
      { EASYSIMD_FLOAT64_C(   104.05), EASYSIMD_FLOAT64_C(  -625.69), EASYSIMD_FLOAT64_C(  -189.18), EASYSIMD_FLOAT64_C(  -256.24),
        EASYSIMD_FLOAT64_C(    83.86), EASYSIMD_FLOAT64_C(  -977.91), EASYSIMD_FLOAT64_C(  -679.93), EASYSIMD_FLOAT64_C(  -854.67) },
      { EASYSIMD_FLOAT64_C(     5.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    11.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(     4.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -4.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    10.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     5.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    11.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(     4.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    10.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     5.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    11.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(   263.51), EASYSIMD_FLOAT64_C(   515.78), EASYSIMD_FLOAT64_C(   240.95), EASYSIMD_FLOAT64_C(  -734.80),
        EASYSIMD_FLOAT64_C(    64.45), EASYSIMD_FLOAT64_C(   621.44), EASYSIMD_FLOAT64_C(   -18.32), EASYSIMD_FLOAT64_C(  -255.69) },
      { EASYSIMD_FLOAT64_C(  -425.70), EASYSIMD_FLOAT64_C(   859.00), EASYSIMD_FLOAT64_C(  -957.54), EASYSIMD_FLOAT64_C(  -680.36),
        EASYSIMD_FLOAT64_C(  -113.49), EASYSIMD_FLOAT64_C(   334.41), EASYSIMD_FLOAT64_C(  -673.72), EASYSIMD_FLOAT64_C(  -616.69) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     2.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   987.25), EASYSIMD_FLOAT64_C(   922.08), EASYSIMD_FLOAT64_C(   501.38), EASYSIMD_FLOAT64_C(   881.73),
        EASYSIMD_FLOAT64_C(   -32.92), EASYSIMD_FLOAT64_C(   888.81), EASYSIMD_FLOAT64_C(  -971.22), EASYSIMD_FLOAT64_C(  -928.87) },
      { EASYSIMD_FLOAT64_C(  -736.88), EASYSIMD_FLOAT64_C(  -160.40), EASYSIMD_FLOAT64_C(  -185.11), EASYSIMD_FLOAT64_C(   346.99),
        EASYSIMD_FLOAT64_C(  -138.31), EASYSIMD_FLOAT64_C(   134.96), EASYSIMD_FLOAT64_C(   492.32), EASYSIMD_FLOAT64_C(  -874.80) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -6.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(     3.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     7.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(    -6.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(     2.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     6.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -5.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     3.00),
        EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     7.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     2.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -5.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     2.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     6.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -6.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(     3.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     7.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(  -349.26), EASYSIMD_FLOAT64_C(  -266.73), EASYSIMD_FLOAT64_C(  -609.60), EASYSIMD_FLOAT64_C(   715.19),
        EASYSIMD_FLOAT64_C(  -645.29), EASYSIMD_FLOAT64_C(   372.08), EASYSIMD_FLOAT64_C(  -540.50), EASYSIMD_FLOAT64_C(   -70.99) },
      { EASYSIMD_FLOAT64_C(   231.08), EASYSIMD_FLOAT64_C(  -498.04), EASYSIMD_FLOAT64_C(   248.64), EASYSIMD_FLOAT64_C(  -882.41),
        EASYSIMD_FLOAT64_C(   836.37), EASYSIMD_FLOAT64_C(   574.92), EASYSIMD_FLOAT64_C(  -499.10), EASYSIMD_FLOAT64_C(   823.63) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(    -0.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(   497.00), EASYSIMD_FLOAT64_C(  -997.72), EASYSIMD_FLOAT64_C(   705.35), EASYSIMD_FLOAT64_C(  -535.91),
        EASYSIMD_FLOAT64_C(   891.09), EASYSIMD_FLOAT64_C(   734.14), EASYSIMD_FLOAT64_C(  -464.78), EASYSIMD_FLOAT64_C(  -845.79) },
      { EASYSIMD_FLOAT64_C(  -426.26), EASYSIMD_FLOAT64_C(   350.11), EASYSIMD_FLOAT64_C(   501.20), EASYSIMD_FLOAT64_C(   435.43),
        EASYSIMD_FLOAT64_C(  -514.93), EASYSIMD_FLOAT64_C(    -6.48), EASYSIMD_FLOAT64_C(   560.63), EASYSIMD_FLOAT64_C(   135.81) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(  -113.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -6.00) },
      { EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -2.00),
        EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(  -114.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -7.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(  -113.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -6.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(  -113.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -6.00) },
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -3.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(  -113.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -6.00) } },
    { { EASYSIMD_FLOAT64_C(   726.78), EASYSIMD_FLOAT64_C(   951.03), EASYSIMD_FLOAT64_C(  -149.00), EASYSIMD_FLOAT64_C(  -918.51),
        EASYSIMD_FLOAT64_C(   323.12), EASYSIMD_FLOAT64_C(   310.50), EASYSIMD_FLOAT64_C(    10.50), EASYSIMD_FLOAT64_C(  -445.80) },
      { EASYSIMD_FLOAT64_C(   812.46), EASYSIMD_FLOAT64_C(  -740.85), EASYSIMD_FLOAT64_C(  -328.21), EASYSIMD_FLOAT64_C(   648.83),
        EASYSIMD_FLOAT64_C(   834.07), EASYSIMD_FLOAT64_C(   172.68), EASYSIMD_FLOAT64_C(   472.46), EASYSIMD_FLOAT64_C(   331.07) },
      { EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -2.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -2.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -2.00) },
      { EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00) },
      { EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00) } },
    { { EASYSIMD_FLOAT64_C(   174.96), EASYSIMD_FLOAT64_C(   177.81), EASYSIMD_FLOAT64_C(   795.16), EASYSIMD_FLOAT64_C(    66.05),
        EASYSIMD_FLOAT64_C(   -88.05), EASYSIMD_FLOAT64_C(  -669.62), EASYSIMD_FLOAT64_C(   220.27), EASYSIMD_FLOAT64_C(   485.69) },
      { EASYSIMD_FLOAT64_C(   680.49), EASYSIMD_FLOAT64_C(  -278.54), EASYSIMD_FLOAT64_C(   -78.88), EASYSIMD_FLOAT64_C(  -834.44),
        EASYSIMD_FLOAT64_C(   714.98), EASYSIMD_FLOAT64_C(  -518.24), EASYSIMD_FLOAT64_C(   301.37), EASYSIMD_FLOAT64_C(   441.76) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(   -10.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(   -11.00), EASYSIMD_FLOAT64_C(    -1.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(   -10.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     2.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(   -10.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(   -10.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(  -567.21), EASYSIMD_FLOAT64_C(  -847.63), EASYSIMD_FLOAT64_C(   523.26), EASYSIMD_FLOAT64_C(   755.90),
        EASYSIMD_FLOAT64_C(   462.87), EASYSIMD_FLOAT64_C(  -466.24), EASYSIMD_FLOAT64_C(  -689.90), EASYSIMD_FLOAT64_C(   275.33) },
      { EASYSIMD_FLOAT64_C(  -207.09), EASYSIMD_FLOAT64_C(   -18.11), EASYSIMD_FLOAT64_C(   -75.84), EASYSIMD_FLOAT64_C(  -373.02),
        EASYSIMD_FLOAT64_C(  -845.43), EASYSIMD_FLOAT64_C(  -603.38), EASYSIMD_FLOAT64_C(   958.05), EASYSIMD_FLOAT64_C(   329.53) },
      { EASYSIMD_FLOAT64_C(     3.00), EASYSIMD_FLOAT64_C(    47.00), EASYSIMD_FLOAT64_C(    -7.00), EASYSIMD_FLOAT64_C(    -2.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(    46.00), EASYSIMD_FLOAT64_C(    -7.00), EASYSIMD_FLOAT64_C(    -3.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     3.00), EASYSIMD_FLOAT64_C(    47.00), EASYSIMD_FLOAT64_C(    -6.00), EASYSIMD_FLOAT64_C(    -2.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00) },
      { EASYSIMD_FLOAT64_C(     2.00), EASYSIMD_FLOAT64_C(    46.00), EASYSIMD_FLOAT64_C(    -6.00), EASYSIMD_FLOAT64_C(    -2.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     3.00), EASYSIMD_FLOAT64_C(    47.00), EASYSIMD_FLOAT64_C(    -7.00), EASYSIMD_FLOAT64_C(    -2.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);

    easysimd__m512d nearest_inf = easysimd_mm512_loadu_pd(test_vec[i].nearest_inf);
    easysimd__m512d neg_inf = easysimd_mm512_loadu_pd(test_vec[i].neg_inf);
    easysimd__m512d pos_inf = easysimd_mm512_loadu_pd(test_vec[i].pos_inf);
    easysimd__m512d zero = easysimd_mm512_loadu_pd(test_vec[i].zero);
    easysimd__m512d direction = easysimd_mm512_loadu_pd(test_vec[i].direction);

    r = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512d_close(r, nearest_inf, 1);

    r = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512d_close(r, neg_inf, 1);

    r = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512d_close(r, pos_inf, 1);

    r = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512d_close(r, zero, 1);
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_div_round_pd");
    easysimd_assert_m512d_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512d nearest_inf = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512d neg_inf = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512d pos_inf = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512d zero = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512d direction = easysimd_mm512_div_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, nearest_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, neg_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, pos_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, zero, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, direction, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_mask_div_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -962.94), EASYSIMD_FLOAT64_C(  989.45),
                         EASYSIMD_FLOAT64_C( -190.71), EASYSIMD_FLOAT64_C(  -80.90),
                         EASYSIMD_FLOAT64_C( -820.03), EASYSIMD_FLOAT64_C(  710.84),
                         EASYSIMD_FLOAT64_C(  742.77), EASYSIMD_FLOAT64_C( -124.19)),
      UINT8_C( 62),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  764.73), EASYSIMD_FLOAT64_C( -738.72),
                         EASYSIMD_FLOAT64_C(  462.89), EASYSIMD_FLOAT64_C( -909.36),
                         EASYSIMD_FLOAT64_C(  920.77), EASYSIMD_FLOAT64_C(  830.94),
                         EASYSIMD_FLOAT64_C( -436.90), EASYSIMD_FLOAT64_C( -984.49)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  318.55), EASYSIMD_FLOAT64_C( -958.54),
                         EASYSIMD_FLOAT64_C( -878.41), EASYSIMD_FLOAT64_C(  198.47),
                         EASYSIMD_FLOAT64_C(  585.51), EASYSIMD_FLOAT64_C(  -97.52),
                         EASYSIMD_FLOAT64_C( -112.08), EASYSIMD_FLOAT64_C( -145.20)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -962.94), EASYSIMD_FLOAT64_C(  989.45),
                         EASYSIMD_FLOAT64_C(   -0.53), EASYSIMD_FLOAT64_C(   -4.58),
                         EASYSIMD_FLOAT64_C(    1.57), EASYSIMD_FLOAT64_C(   -8.52),
                         EASYSIMD_FLOAT64_C(    3.90), EASYSIMD_FLOAT64_C( -124.19)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  267.17), EASYSIMD_FLOAT64_C( -878.34),
                         EASYSIMD_FLOAT64_C(  132.07), EASYSIMD_FLOAT64_C(  827.87),
                         EASYSIMD_FLOAT64_C(  178.51), EASYSIMD_FLOAT64_C(  362.39),
                         EASYSIMD_FLOAT64_C(  200.13), EASYSIMD_FLOAT64_C( -407.98)),
      UINT8_C( 51),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -126.54), EASYSIMD_FLOAT64_C( -164.31),
                         EASYSIMD_FLOAT64_C( -971.32), EASYSIMD_FLOAT64_C(  611.23),
                         EASYSIMD_FLOAT64_C(  591.83), EASYSIMD_FLOAT64_C(  793.58),
                         EASYSIMD_FLOAT64_C(  171.77), EASYSIMD_FLOAT64_C(  109.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  225.35), EASYSIMD_FLOAT64_C( -734.84),
                         EASYSIMD_FLOAT64_C(  728.29), EASYSIMD_FLOAT64_C( -721.11),
                         EASYSIMD_FLOAT64_C( -448.10), EASYSIMD_FLOAT64_C(  310.61),
                         EASYSIMD_FLOAT64_C( -362.27), EASYSIMD_FLOAT64_C( -413.07)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  267.17), EASYSIMD_FLOAT64_C( -878.34),
                         EASYSIMD_FLOAT64_C(   -1.33), EASYSIMD_FLOAT64_C(   -0.85),
                         EASYSIMD_FLOAT64_C(  178.51), EASYSIMD_FLOAT64_C(  362.39),
                         EASYSIMD_FLOAT64_C(   -0.47), EASYSIMD_FLOAT64_C(   -0.27)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  439.30), EASYSIMD_FLOAT64_C(  935.39),
                         EASYSIMD_FLOAT64_C(   20.00), EASYSIMD_FLOAT64_C( -941.65),
                         EASYSIMD_FLOAT64_C(  988.79), EASYSIMD_FLOAT64_C(  773.96),
                         EASYSIMD_FLOAT64_C( -788.78), EASYSIMD_FLOAT64_C( -311.91)),
      UINT8_C(178),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -374.30), EASYSIMD_FLOAT64_C(  599.21),
                         EASYSIMD_FLOAT64_C(  966.83), EASYSIMD_FLOAT64_C(  775.18),
                         EASYSIMD_FLOAT64_C(  846.32), EASYSIMD_FLOAT64_C(  124.04),
                         EASYSIMD_FLOAT64_C( -883.36), EASYSIMD_FLOAT64_C( -405.09)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  467.70), EASYSIMD_FLOAT64_C( -626.02),
                         EASYSIMD_FLOAT64_C(  355.93), EASYSIMD_FLOAT64_C(  294.34),
                         EASYSIMD_FLOAT64_C( -575.79), EASYSIMD_FLOAT64_C( -504.82),
                         EASYSIMD_FLOAT64_C(  854.52), EASYSIMD_FLOAT64_C( -173.82)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -0.80), EASYSIMD_FLOAT64_C(  935.39),
                         EASYSIMD_FLOAT64_C(    2.72), EASYSIMD_FLOAT64_C(    2.63),
                         EASYSIMD_FLOAT64_C(  988.79), EASYSIMD_FLOAT64_C(  773.96),
                         EASYSIMD_FLOAT64_C(   -1.03), EASYSIMD_FLOAT64_C( -311.91)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -274.81), EASYSIMD_FLOAT64_C(  196.78),
                         EASYSIMD_FLOAT64_C( -805.22), EASYSIMD_FLOAT64_C(  855.89),
                         EASYSIMD_FLOAT64_C( -996.67), EASYSIMD_FLOAT64_C(  424.78),
                         EASYSIMD_FLOAT64_C(  489.73), EASYSIMD_FLOAT64_C(  635.35)),
      UINT8_C( 38),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   79.19), EASYSIMD_FLOAT64_C( -114.25),
                         EASYSIMD_FLOAT64_C(  983.59), EASYSIMD_FLOAT64_C(  645.66),
                         EASYSIMD_FLOAT64_C(  982.80), EASYSIMD_FLOAT64_C( -683.73),
                         EASYSIMD_FLOAT64_C(  259.13), EASYSIMD_FLOAT64_C(  186.09)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  665.49), EASYSIMD_FLOAT64_C( -110.92),
                         EASYSIMD_FLOAT64_C(  978.65), EASYSIMD_FLOAT64_C(  104.45),
                         EASYSIMD_FLOAT64_C(  903.68), EASYSIMD_FLOAT64_C( -580.74),
                         EASYSIMD_FLOAT64_C(  776.44), EASYSIMD_FLOAT64_C(  571.14)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -274.81), EASYSIMD_FLOAT64_C(  196.78),
                         EASYSIMD_FLOAT64_C(    1.01), EASYSIMD_FLOAT64_C(  855.89),
                         EASYSIMD_FLOAT64_C( -996.67), EASYSIMD_FLOAT64_C(    1.18),
                         EASYSIMD_FLOAT64_C(    0.33), EASYSIMD_FLOAT64_C(  635.35)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  105.93), EASYSIMD_FLOAT64_C( -486.73),
                         EASYSIMD_FLOAT64_C(  293.04), EASYSIMD_FLOAT64_C(  328.58),
                         EASYSIMD_FLOAT64_C( -725.03), EASYSIMD_FLOAT64_C(    3.53),
                         EASYSIMD_FLOAT64_C(  663.75), EASYSIMD_FLOAT64_C(  -59.32)),
      UINT8_C( 67),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  917.98), EASYSIMD_FLOAT64_C( -430.92),
                         EASYSIMD_FLOAT64_C(  839.77), EASYSIMD_FLOAT64_C( -412.68),
                         EASYSIMD_FLOAT64_C( -397.37), EASYSIMD_FLOAT64_C( -947.31),
                         EASYSIMD_FLOAT64_C(  584.59), EASYSIMD_FLOAT64_C( -352.12)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  781.61), EASYSIMD_FLOAT64_C(  978.32),
                         EASYSIMD_FLOAT64_C(  374.68), EASYSIMD_FLOAT64_C( -857.00),
                         EASYSIMD_FLOAT64_C(  821.72), EASYSIMD_FLOAT64_C(  -88.08),
                         EASYSIMD_FLOAT64_C(  243.00), EASYSIMD_FLOAT64_C( -640.77)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  105.93), EASYSIMD_FLOAT64_C(   -0.44),
                         EASYSIMD_FLOAT64_C(  293.04), EASYSIMD_FLOAT64_C(  328.58),
                         EASYSIMD_FLOAT64_C( -725.03), EASYSIMD_FLOAT64_C(    3.53),
                         EASYSIMD_FLOAT64_C(    2.41), EASYSIMD_FLOAT64_C(    0.55)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -591.91), EASYSIMD_FLOAT64_C(  615.29),
                         EASYSIMD_FLOAT64_C( -726.07), EASYSIMD_FLOAT64_C(  857.36),
                         EASYSIMD_FLOAT64_C(  636.31), EASYSIMD_FLOAT64_C(  104.40),
                         EASYSIMD_FLOAT64_C( -167.77), EASYSIMD_FLOAT64_C( -372.65)),
      UINT8_C( 15),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  411.16), EASYSIMD_FLOAT64_C(  928.95),
                         EASYSIMD_FLOAT64_C(  110.13), EASYSIMD_FLOAT64_C(  933.76),
                         EASYSIMD_FLOAT64_C(  836.76), EASYSIMD_FLOAT64_C(  628.60),
                         EASYSIMD_FLOAT64_C( -586.52), EASYSIMD_FLOAT64_C(  293.24)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -457.28), EASYSIMD_FLOAT64_C(  705.56),
                         EASYSIMD_FLOAT64_C( -798.08), EASYSIMD_FLOAT64_C(  773.61),
                         EASYSIMD_FLOAT64_C( -590.48), EASYSIMD_FLOAT64_C( -291.69),
                         EASYSIMD_FLOAT64_C(  654.27), EASYSIMD_FLOAT64_C( -537.59)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -591.91), EASYSIMD_FLOAT64_C(  615.29),
                         EASYSIMD_FLOAT64_C( -726.07), EASYSIMD_FLOAT64_C(  857.36),
                         EASYSIMD_FLOAT64_C(   -1.42), EASYSIMD_FLOAT64_C(   -2.16),
                         EASYSIMD_FLOAT64_C(   -0.90), EASYSIMD_FLOAT64_C(   -0.55)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  270.92), EASYSIMD_FLOAT64_C( -517.94),
                         EASYSIMD_FLOAT64_C(   36.22), EASYSIMD_FLOAT64_C(  204.54),
                         EASYSIMD_FLOAT64_C(  579.30), EASYSIMD_FLOAT64_C(  257.34),
                         EASYSIMD_FLOAT64_C( -998.24), EASYSIMD_FLOAT64_C( -146.41)),
      UINT8_C(152),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  268.93), EASYSIMD_FLOAT64_C( -893.46),
                         EASYSIMD_FLOAT64_C( -476.89), EASYSIMD_FLOAT64_C( -696.00),
                         EASYSIMD_FLOAT64_C( -817.69), EASYSIMD_FLOAT64_C(  127.75),
                         EASYSIMD_FLOAT64_C( -366.34), EASYSIMD_FLOAT64_C( -437.04)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -372.16), EASYSIMD_FLOAT64_C(  900.88),
                         EASYSIMD_FLOAT64_C( -550.65), EASYSIMD_FLOAT64_C(  567.85),
                         EASYSIMD_FLOAT64_C(  968.56), EASYSIMD_FLOAT64_C( -695.12),
                         EASYSIMD_FLOAT64_C(  555.56), EASYSIMD_FLOAT64_C(  952.92)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -0.72), EASYSIMD_FLOAT64_C( -517.94),
                         EASYSIMD_FLOAT64_C(   36.22), EASYSIMD_FLOAT64_C(   -1.23),
                         EASYSIMD_FLOAT64_C(   -0.84), EASYSIMD_FLOAT64_C(  257.34),
                         EASYSIMD_FLOAT64_C( -998.24), EASYSIMD_FLOAT64_C( -146.41)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -468.36), EASYSIMD_FLOAT64_C(  186.16),
                         EASYSIMD_FLOAT64_C( -910.43), EASYSIMD_FLOAT64_C( -280.07),
                         EASYSIMD_FLOAT64_C(  -96.94), EASYSIMD_FLOAT64_C(  387.95),
                         EASYSIMD_FLOAT64_C(  198.14), EASYSIMD_FLOAT64_C( -504.51)),
      UINT8_C( 21),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  573.90), EASYSIMD_FLOAT64_C(  496.67),
                         EASYSIMD_FLOAT64_C( -823.61), EASYSIMD_FLOAT64_C(  204.56),
                         EASYSIMD_FLOAT64_C( -856.87), EASYSIMD_FLOAT64_C( -449.28),
                         EASYSIMD_FLOAT64_C(    9.73), EASYSIMD_FLOAT64_C( -739.12)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -471.24), EASYSIMD_FLOAT64_C( -711.43),
                         EASYSIMD_FLOAT64_C( -281.50), EASYSIMD_FLOAT64_C(  493.76),
                         EASYSIMD_FLOAT64_C(  103.01), EASYSIMD_FLOAT64_C( -996.35),
                         EASYSIMD_FLOAT64_C(  670.04), EASYSIMD_FLOAT64_C( -895.53)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -468.36), EASYSIMD_FLOAT64_C(  186.16),
                         EASYSIMD_FLOAT64_C( -910.43), EASYSIMD_FLOAT64_C(    0.41),
                         EASYSIMD_FLOAT64_C(  -96.94), EASYSIMD_FLOAT64_C(    0.45),
                         EASYSIMD_FLOAT64_C(  198.14), EASYSIMD_FLOAT64_C(    0.83)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_div_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_div_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_div_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { UINT8_C(113),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  112.08), EASYSIMD_FLOAT64_C(  712.48),
                         EASYSIMD_FLOAT64_C( -754.71), EASYSIMD_FLOAT64_C(  256.61),
                         EASYSIMD_FLOAT64_C(  768.73), EASYSIMD_FLOAT64_C(  227.99),
                         EASYSIMD_FLOAT64_C(  174.97), EASYSIMD_FLOAT64_C(  338.39)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  759.92), EASYSIMD_FLOAT64_C(  629.58),
                         EASYSIMD_FLOAT64_C(   23.76), EASYSIMD_FLOAT64_C(  980.95),
                         EASYSIMD_FLOAT64_C(  224.97), EASYSIMD_FLOAT64_C(  818.07),
                         EASYSIMD_FLOAT64_C( -531.75), EASYSIMD_FLOAT64_C( -531.67)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    1.13),
                         EASYSIMD_FLOAT64_C(  -31.76), EASYSIMD_FLOAT64_C(    0.26),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   -0.64)) },
    { UINT8_C( 88),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  841.82), EASYSIMD_FLOAT64_C( -330.14),
                         EASYSIMD_FLOAT64_C( -498.05), EASYSIMD_FLOAT64_C( -706.46),
                         EASYSIMD_FLOAT64_C( -284.71), EASYSIMD_FLOAT64_C( -940.98),
                         EASYSIMD_FLOAT64_C( -491.84), EASYSIMD_FLOAT64_C(   52.49)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  259.38), EASYSIMD_FLOAT64_C(  745.69),
                         EASYSIMD_FLOAT64_C( -258.59), EASYSIMD_FLOAT64_C( -549.06),
                         EASYSIMD_FLOAT64_C(  646.98), EASYSIMD_FLOAT64_C(  925.86),
                         EASYSIMD_FLOAT64_C(  378.90), EASYSIMD_FLOAT64_C( -524.10)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   -0.44),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    1.29),
                         EASYSIMD_FLOAT64_C(   -0.44), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(184),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -167.30), EASYSIMD_FLOAT64_C( -738.71),
                         EASYSIMD_FLOAT64_C( -816.67), EASYSIMD_FLOAT64_C(   43.31),
                         EASYSIMD_FLOAT64_C(  -98.40), EASYSIMD_FLOAT64_C(  217.89),
                         EASYSIMD_FLOAT64_C(  626.98), EASYSIMD_FLOAT64_C( -409.09)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -443.71), EASYSIMD_FLOAT64_C( -786.78),
                         EASYSIMD_FLOAT64_C(  -92.41), EASYSIMD_FLOAT64_C( -378.62),
                         EASYSIMD_FLOAT64_C(  632.49), EASYSIMD_FLOAT64_C( -867.20),
                         EASYSIMD_FLOAT64_C(  977.79), EASYSIMD_FLOAT64_C( -788.71)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.38), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    8.84), EASYSIMD_FLOAT64_C(   -0.11),
                         EASYSIMD_FLOAT64_C(   -0.16), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(119),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -156.59), EASYSIMD_FLOAT64_C(  153.69),
                         EASYSIMD_FLOAT64_C(  895.38), EASYSIMD_FLOAT64_C( -242.63),
                         EASYSIMD_FLOAT64_C(  994.17), EASYSIMD_FLOAT64_C( -265.23),
                         EASYSIMD_FLOAT64_C(  -57.91), EASYSIMD_FLOAT64_C( -586.11)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -851.62), EASYSIMD_FLOAT64_C(  838.22),
                         EASYSIMD_FLOAT64_C(  464.78), EASYSIMD_FLOAT64_C( -248.37),
                         EASYSIMD_FLOAT64_C(   28.49), EASYSIMD_FLOAT64_C( -176.67),
                         EASYSIMD_FLOAT64_C( -468.39), EASYSIMD_FLOAT64_C( -893.30)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.18),
                         EASYSIMD_FLOAT64_C(    1.93), EASYSIMD_FLOAT64_C(    0.98),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    1.50),
                         EASYSIMD_FLOAT64_C(    0.12), EASYSIMD_FLOAT64_C(    0.66)) },
    { UINT8_C(181),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -748.23), EASYSIMD_FLOAT64_C(  960.31),
                         EASYSIMD_FLOAT64_C(  -52.56), EASYSIMD_FLOAT64_C(  -18.49),
                         EASYSIMD_FLOAT64_C( -174.79), EASYSIMD_FLOAT64_C( -875.70),
                         EASYSIMD_FLOAT64_C(  270.45), EASYSIMD_FLOAT64_C(  571.57)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  887.26), EASYSIMD_FLOAT64_C( -380.63),
                         EASYSIMD_FLOAT64_C(  603.15), EASYSIMD_FLOAT64_C(  906.17),
                         EASYSIMD_FLOAT64_C( -446.90), EASYSIMD_FLOAT64_C(  518.96),
                         EASYSIMD_FLOAT64_C(  325.09), EASYSIMD_FLOAT64_C(  394.29)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -0.84), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(   -0.09), EASYSIMD_FLOAT64_C(   -0.02),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   -1.69),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    1.45)) },
    { UINT8_C(108),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -891.46), EASYSIMD_FLOAT64_C(  298.88),
                         EASYSIMD_FLOAT64_C(  907.92), EASYSIMD_FLOAT64_C(  585.94),
                         EASYSIMD_FLOAT64_C(  976.00), EASYSIMD_FLOAT64_C(  860.60),
                         EASYSIMD_FLOAT64_C( -807.57), EASYSIMD_FLOAT64_C( -501.53)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -634.78), EASYSIMD_FLOAT64_C( -309.30),
                         EASYSIMD_FLOAT64_C( -478.69), EASYSIMD_FLOAT64_C( -499.66),
                         EASYSIMD_FLOAT64_C( -834.97), EASYSIMD_FLOAT64_C( -926.76),
                         EASYSIMD_FLOAT64_C(  306.74), EASYSIMD_FLOAT64_C(  350.68)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   -0.97),
                         EASYSIMD_FLOAT64_C(   -1.90), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(   -1.17), EASYSIMD_FLOAT64_C(   -0.93),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(  5),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -764.37), EASYSIMD_FLOAT64_C( -698.84),
                         EASYSIMD_FLOAT64_C(  111.54), EASYSIMD_FLOAT64_C(  627.05),
                         EASYSIMD_FLOAT64_C(  619.20), EASYSIMD_FLOAT64_C(  107.79),
                         EASYSIMD_FLOAT64_C(  830.07), EASYSIMD_FLOAT64_C( -991.50)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  902.97), EASYSIMD_FLOAT64_C( -970.67),
                         EASYSIMD_FLOAT64_C( -396.71), EASYSIMD_FLOAT64_C(  740.42),
                         EASYSIMD_FLOAT64_C( -740.07), EASYSIMD_FLOAT64_C(  691.95),
                         EASYSIMD_FLOAT64_C( -434.89), EASYSIMD_FLOAT64_C(  270.74)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.16),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   -3.66)) },
    { UINT8_C( 94),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -9.32), EASYSIMD_FLOAT64_C(  588.15),
                         EASYSIMD_FLOAT64_C(  740.36), EASYSIMD_FLOAT64_C(  589.64),
                         EASYSIMD_FLOAT64_C( -146.10), EASYSIMD_FLOAT64_C(  771.62),
                         EASYSIMD_FLOAT64_C( -975.31), EASYSIMD_FLOAT64_C(  550.04)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  380.47), EASYSIMD_FLOAT64_C( -722.04),
                         EASYSIMD_FLOAT64_C( -251.19), EASYSIMD_FLOAT64_C(  885.20),
                         EASYSIMD_FLOAT64_C( -718.95), EASYSIMD_FLOAT64_C( -995.18),
                         EASYSIMD_FLOAT64_C(  316.41), EASYSIMD_FLOAT64_C(  425.49)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   -0.81),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.67),
                         EASYSIMD_FLOAT64_C(    0.20), EASYSIMD_FLOAT64_C(   -0.78),
                         EASYSIMD_FLOAT64_C(   -3.08), EASYSIMD_FLOAT64_C(    0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r = easysimd_mm512_maskz_div_pd(test_vec[i].k, test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_div_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_div_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_div_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_div_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_div_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_div_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_div_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_div_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_div_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_div_round_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_div_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_div_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_div_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_div_round_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_div_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_div_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
