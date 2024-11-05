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

#define EASYSIMD_TEST_X86_AVX512_INSN extract

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/extract.h>

static int
test_easysimd_mm256_mask_extractf32x4_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 r0[4];
    const easysimd_float32 r1[4];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT32_C(   725.90), EASYSIMD_FLOAT32_C(  -271.20), EASYSIMD_FLOAT32_C(  -115.69), EASYSIMD_FLOAT32_C(  -413.08) },
      UINT8_C(184),
      { EASYSIMD_FLOAT32_C(  -814.58), EASYSIMD_FLOAT32_C(  -809.90), EASYSIMD_FLOAT32_C(   314.62), EASYSIMD_FLOAT32_C(   132.83),
        EASYSIMD_FLOAT32_C(   950.57), EASYSIMD_FLOAT32_C(   213.33), EASYSIMD_FLOAT32_C(  -199.23), EASYSIMD_FLOAT32_C(   258.14) },
      { EASYSIMD_FLOAT32_C(   725.90), EASYSIMD_FLOAT32_C(  -271.20), EASYSIMD_FLOAT32_C(  -115.69), EASYSIMD_FLOAT32_C(   132.83) },
      { EASYSIMD_FLOAT32_C(   725.90), EASYSIMD_FLOAT32_C(  -271.20), EASYSIMD_FLOAT32_C(  -115.69), EASYSIMD_FLOAT32_C(   258.14) } },
    { { EASYSIMD_FLOAT32_C(   334.68), EASYSIMD_FLOAT32_C(  -399.85), EASYSIMD_FLOAT32_C(   968.21), EASYSIMD_FLOAT32_C(  -669.38) },
      UINT8_C(177),
      { EASYSIMD_FLOAT32_C(  -676.42), EASYSIMD_FLOAT32_C(   482.45), EASYSIMD_FLOAT32_C(  -163.33), EASYSIMD_FLOAT32_C(  -960.93),
        EASYSIMD_FLOAT32_C(   756.06), EASYSIMD_FLOAT32_C(   -88.15), EASYSIMD_FLOAT32_C(   145.93), EASYSIMD_FLOAT32_C(  -342.83) },
      { EASYSIMD_FLOAT32_C(  -676.42), EASYSIMD_FLOAT32_C(  -399.85), EASYSIMD_FLOAT32_C(   968.21), EASYSIMD_FLOAT32_C(  -669.38) },
      { EASYSIMD_FLOAT32_C(   756.06), EASYSIMD_FLOAT32_C(  -399.85), EASYSIMD_FLOAT32_C(   968.21), EASYSIMD_FLOAT32_C(  -669.38) } },
    { { EASYSIMD_FLOAT32_C(   176.00), EASYSIMD_FLOAT32_C(  -420.28), EASYSIMD_FLOAT32_C(   347.80), EASYSIMD_FLOAT32_C(  -424.60) },
      UINT8_C( 79),
      { EASYSIMD_FLOAT32_C(    73.70), EASYSIMD_FLOAT32_C(   304.19), EASYSIMD_FLOAT32_C(  -302.00), EASYSIMD_FLOAT32_C(   660.62),
        EASYSIMD_FLOAT32_C(   393.52), EASYSIMD_FLOAT32_C(  -116.59), EASYSIMD_FLOAT32_C(   850.72), EASYSIMD_FLOAT32_C(  -291.87) },
      { EASYSIMD_FLOAT32_C(    73.70), EASYSIMD_FLOAT32_C(   304.19), EASYSIMD_FLOAT32_C(  -302.00), EASYSIMD_FLOAT32_C(   660.62) },
      { EASYSIMD_FLOAT32_C(   393.52), EASYSIMD_FLOAT32_C(  -116.59), EASYSIMD_FLOAT32_C(   850.72), EASYSIMD_FLOAT32_C(  -291.87) } },
    { { EASYSIMD_FLOAT32_C(  -983.75), EASYSIMD_FLOAT32_C(   801.29), EASYSIMD_FLOAT32_C(   921.47), EASYSIMD_FLOAT32_C(  -182.98) },
      UINT8_C( 77),
      { EASYSIMD_FLOAT32_C(   256.15), EASYSIMD_FLOAT32_C(   417.17), EASYSIMD_FLOAT32_C(    27.64), EASYSIMD_FLOAT32_C(   586.77),
        EASYSIMD_FLOAT32_C(  -966.05), EASYSIMD_FLOAT32_C(   351.22), EASYSIMD_FLOAT32_C(    69.22), EASYSIMD_FLOAT32_C(  -129.38) },
      { EASYSIMD_FLOAT32_C(   256.15), EASYSIMD_FLOAT32_C(   801.29), EASYSIMD_FLOAT32_C(    27.64), EASYSIMD_FLOAT32_C(   586.77) },
      { EASYSIMD_FLOAT32_C(  -966.05), EASYSIMD_FLOAT32_C(   801.29), EASYSIMD_FLOAT32_C(    69.22), EASYSIMD_FLOAT32_C(  -129.38) } },
    { { EASYSIMD_FLOAT32_C(   390.29), EASYSIMD_FLOAT32_C(  -174.72), EASYSIMD_FLOAT32_C(   782.47), EASYSIMD_FLOAT32_C(  -463.78) },
      UINT8_C(244),
      { EASYSIMD_FLOAT32_C(   -41.53), EASYSIMD_FLOAT32_C(   115.94), EASYSIMD_FLOAT32_C(  -169.75), EASYSIMD_FLOAT32_C(   533.87),
        EASYSIMD_FLOAT32_C(   -70.37), EASYSIMD_FLOAT32_C(   903.95), EASYSIMD_FLOAT32_C(  -161.94), EASYSIMD_FLOAT32_C(   627.62) },
      { EASYSIMD_FLOAT32_C(   390.29), EASYSIMD_FLOAT32_C(  -174.72), EASYSIMD_FLOAT32_C(  -169.75), EASYSIMD_FLOAT32_C(  -463.78) },
      { EASYSIMD_FLOAT32_C(   390.29), EASYSIMD_FLOAT32_C(  -174.72), EASYSIMD_FLOAT32_C(  -161.94), EASYSIMD_FLOAT32_C(  -463.78) } },
    { { EASYSIMD_FLOAT32_C(   564.58), EASYSIMD_FLOAT32_C(  -768.42), EASYSIMD_FLOAT32_C(  -488.96), EASYSIMD_FLOAT32_C(   415.30) },
      UINT8_C(212),
      { EASYSIMD_FLOAT32_C(  -472.71), EASYSIMD_FLOAT32_C(   216.59), EASYSIMD_FLOAT32_C(  -138.82), EASYSIMD_FLOAT32_C(   344.30),
        EASYSIMD_FLOAT32_C(  -723.98), EASYSIMD_FLOAT32_C(  -882.67), EASYSIMD_FLOAT32_C(  -238.52), EASYSIMD_FLOAT32_C(   303.66) },
      { EASYSIMD_FLOAT32_C(   564.58), EASYSIMD_FLOAT32_C(  -768.42), EASYSIMD_FLOAT32_C(  -138.82), EASYSIMD_FLOAT32_C(   415.30) },
      { EASYSIMD_FLOAT32_C(   564.58), EASYSIMD_FLOAT32_C(  -768.42), EASYSIMD_FLOAT32_C(  -238.52), EASYSIMD_FLOAT32_C(   415.30) } },
    { { EASYSIMD_FLOAT32_C(   704.10), EASYSIMD_FLOAT32_C(  -204.57), EASYSIMD_FLOAT32_C(  -345.13), EASYSIMD_FLOAT32_C(  -226.68) },
      UINT8_C(237),
      { EASYSIMD_FLOAT32_C(  -954.84), EASYSIMD_FLOAT32_C(   598.60), EASYSIMD_FLOAT32_C(   448.51), EASYSIMD_FLOAT32_C(  -418.61),
        EASYSIMD_FLOAT32_C(    81.05), EASYSIMD_FLOAT32_C(  -593.02), EASYSIMD_FLOAT32_C(   697.33), EASYSIMD_FLOAT32_C(   911.31) },
      { EASYSIMD_FLOAT32_C(  -954.84), EASYSIMD_FLOAT32_C(  -204.57), EASYSIMD_FLOAT32_C(   448.51), EASYSIMD_FLOAT32_C(  -418.61) },
      { EASYSIMD_FLOAT32_C(    81.05), EASYSIMD_FLOAT32_C(  -204.57), EASYSIMD_FLOAT32_C(   697.33), EASYSIMD_FLOAT32_C(   911.31) } },
    { { EASYSIMD_FLOAT32_C(   940.85), EASYSIMD_FLOAT32_C(  -373.05), EASYSIMD_FLOAT32_C(   815.26), EASYSIMD_FLOAT32_C(  -221.10) },
      UINT8_C( 65),
      { EASYSIMD_FLOAT32_C(   379.84), EASYSIMD_FLOAT32_C(    10.48), EASYSIMD_FLOAT32_C(  -234.38), EASYSIMD_FLOAT32_C(  -204.86),
        EASYSIMD_FLOAT32_C(   950.19), EASYSIMD_FLOAT32_C(   292.90), EASYSIMD_FLOAT32_C(  -988.28), EASYSIMD_FLOAT32_C(  -188.63) },
      { EASYSIMD_FLOAT32_C(   379.84), EASYSIMD_FLOAT32_C(  -373.05), EASYSIMD_FLOAT32_C(   815.26), EASYSIMD_FLOAT32_C(  -221.10) },
      { EASYSIMD_FLOAT32_C(   950.19), EASYSIMD_FLOAT32_C(  -373.05), EASYSIMD_FLOAT32_C(   815.26), EASYSIMD_FLOAT32_C(  -221.10) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m128 r;
    r = easysimd_mm256_mask_extractf32x4_ps(src, k, a, 0);
    easysimd_assert_m128_close(r, easysimd_mm_loadu_ps(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_extractf32x4_ps(src, k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_extractf32x4_ps");
    easysimd_assert_m128_close(r, easysimd_mm_loadu_ps(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r0 = easysimd_mm256_mask_extractf32x4_ps(src, k, a, 0);
    easysimd__m128 r1 = easysimd_mm256_mask_extractf32x4_ps(src, k, a, 1);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_extractf32x4_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 r0[4];
    const easysimd_float32 r1[4];
  } test_vec[8] = {
    { UINT8_C(248),
      { EASYSIMD_FLOAT32_C(  -712.26), EASYSIMD_FLOAT32_C(   -71.31), EASYSIMD_FLOAT32_C(   398.68), EASYSIMD_FLOAT32_C(   591.40),
        EASYSIMD_FLOAT32_C(  -367.21), EASYSIMD_FLOAT32_C(  -805.89), EASYSIMD_FLOAT32_C(  -753.73), EASYSIMD_FLOAT32_C(   406.11) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   591.40) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   406.11) } },
    { UINT8_C(231),
      { EASYSIMD_FLOAT32_C(  -708.57), EASYSIMD_FLOAT32_C(     4.72), EASYSIMD_FLOAT32_C(   308.66), EASYSIMD_FLOAT32_C(  -127.18),
        EASYSIMD_FLOAT32_C(  -914.23), EASYSIMD_FLOAT32_C(   715.64), EASYSIMD_FLOAT32_C(  -429.86), EASYSIMD_FLOAT32_C(   997.08) },
      { EASYSIMD_FLOAT32_C(  -708.57), EASYSIMD_FLOAT32_C(     4.72), EASYSIMD_FLOAT32_C(   308.66), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -914.23), EASYSIMD_FLOAT32_C(   715.64), EASYSIMD_FLOAT32_C(  -429.86), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(117),
      { EASYSIMD_FLOAT32_C(   197.10), EASYSIMD_FLOAT32_C(   812.34), EASYSIMD_FLOAT32_C(  -564.60), EASYSIMD_FLOAT32_C(   451.68),
        EASYSIMD_FLOAT32_C(   192.18), EASYSIMD_FLOAT32_C(   445.88), EASYSIMD_FLOAT32_C(  -782.70), EASYSIMD_FLOAT32_C(   987.31) },
      { EASYSIMD_FLOAT32_C(   197.10), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -564.60), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(   192.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -782.70), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(119),
      { EASYSIMD_FLOAT32_C(   510.20), EASYSIMD_FLOAT32_C(   999.04), EASYSIMD_FLOAT32_C(  -792.56), EASYSIMD_FLOAT32_C(  -852.60),
        EASYSIMD_FLOAT32_C(  -713.22), EASYSIMD_FLOAT32_C(   136.13), EASYSIMD_FLOAT32_C(   546.09), EASYSIMD_FLOAT32_C(   878.18) },
      { EASYSIMD_FLOAT32_C(   510.20), EASYSIMD_FLOAT32_C(   999.04), EASYSIMD_FLOAT32_C(  -792.56), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -713.22), EASYSIMD_FLOAT32_C(   136.13), EASYSIMD_FLOAT32_C(   546.09), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(118),
      { EASYSIMD_FLOAT32_C(   740.20), EASYSIMD_FLOAT32_C(  -875.55), EASYSIMD_FLOAT32_C(   175.04), EASYSIMD_FLOAT32_C(   600.35),
        EASYSIMD_FLOAT32_C(  -584.12), EASYSIMD_FLOAT32_C(  -820.24), EASYSIMD_FLOAT32_C(   -90.99), EASYSIMD_FLOAT32_C(   288.69) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -875.55), EASYSIMD_FLOAT32_C(   175.04), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -820.24), EASYSIMD_FLOAT32_C(   -90.99), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(147),
      { EASYSIMD_FLOAT32_C(  -375.35), EASYSIMD_FLOAT32_C(   858.84), EASYSIMD_FLOAT32_C(  -737.39), EASYSIMD_FLOAT32_C(  -718.86),
        EASYSIMD_FLOAT32_C(    55.94), EASYSIMD_FLOAT32_C(  -925.06), EASYSIMD_FLOAT32_C(  -283.46), EASYSIMD_FLOAT32_C(  -492.38) },
      { EASYSIMD_FLOAT32_C(  -375.35), EASYSIMD_FLOAT32_C(   858.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(    55.94), EASYSIMD_FLOAT32_C(  -925.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(246),
      { EASYSIMD_FLOAT32_C(  -837.58), EASYSIMD_FLOAT32_C(  -275.09), EASYSIMD_FLOAT32_C(   254.43), EASYSIMD_FLOAT32_C(   558.49),
        EASYSIMD_FLOAT32_C(  -764.89), EASYSIMD_FLOAT32_C(   253.47), EASYSIMD_FLOAT32_C(   765.93), EASYSIMD_FLOAT32_C(  -617.48) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -275.09), EASYSIMD_FLOAT32_C(   254.43), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   253.47), EASYSIMD_FLOAT32_C(   765.93), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(220),
      { EASYSIMD_FLOAT32_C(   -97.94), EASYSIMD_FLOAT32_C(   928.60), EASYSIMD_FLOAT32_C(   418.42), EASYSIMD_FLOAT32_C(  -329.02),
        EASYSIMD_FLOAT32_C(   668.80), EASYSIMD_FLOAT32_C(   542.87), EASYSIMD_FLOAT32_C(   846.02), EASYSIMD_FLOAT32_C(   269.15) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   418.42), EASYSIMD_FLOAT32_C(  -329.02) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   846.02), EASYSIMD_FLOAT32_C(   269.15) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m128 r;
    r = easysimd_mm256_maskz_extractf32x4_ps(k, a, 0);
    easysimd_assert_m128_close(r, easysimd_mm_loadu_ps(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_extractf32x4_ps(k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_extractf32x4_ps");
    easysimd_assert_m128_close(r, easysimd_mm_loadu_ps(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r0 = easysimd_mm256_maskz_extractf32x4_ps(k, a, 0);
    easysimd__m128 r1 = easysimd_mm256_maskz_extractf32x4_ps(k, a, 1);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_extractf64x2_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float64 src[2];
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 r0[2];
    easysimd_float64 r1[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -575.30), EASYSIMD_FLOAT64_C(   -66.81) },
      UINT8_C(129),
      { EASYSIMD_FLOAT64_C(   395.10), EASYSIMD_FLOAT64_C(  -913.72), EASYSIMD_FLOAT64_C(   185.27), EASYSIMD_FLOAT64_C(   217.66) },
      { EASYSIMD_FLOAT64_C(   395.10), EASYSIMD_FLOAT64_C(   -66.81) },
      { EASYSIMD_FLOAT64_C(   185.27), EASYSIMD_FLOAT64_C(   -66.81) } },
    { { EASYSIMD_FLOAT64_C(  -110.96), EASYSIMD_FLOAT64_C(  -621.83) },
      UINT8_C(105),
      { EASYSIMD_FLOAT64_C(   950.49), EASYSIMD_FLOAT64_C(  -263.08), EASYSIMD_FLOAT64_C(  -456.19), EASYSIMD_FLOAT64_C(  -827.81) },
      { EASYSIMD_FLOAT64_C(   950.49), EASYSIMD_FLOAT64_C(  -621.83) },
      { EASYSIMD_FLOAT64_C(  -456.19), EASYSIMD_FLOAT64_C(  -621.83) } },
    { { EASYSIMD_FLOAT64_C(   707.91), EASYSIMD_FLOAT64_C(   -62.53) },
      UINT8_C(217),
      { EASYSIMD_FLOAT64_C(   127.90), EASYSIMD_FLOAT64_C(   276.42), EASYSIMD_FLOAT64_C(   723.38), EASYSIMD_FLOAT64_C(  -515.19) },
      { EASYSIMD_FLOAT64_C(   127.90), EASYSIMD_FLOAT64_C(   -62.53) },
      { EASYSIMD_FLOAT64_C(   723.38), EASYSIMD_FLOAT64_C(   -62.53) } },
    { { EASYSIMD_FLOAT64_C(  -553.15), EASYSIMD_FLOAT64_C(   279.93) },
      UINT8_C(244),
      { EASYSIMD_FLOAT64_C(  -480.63), EASYSIMD_FLOAT64_C(  -116.84), EASYSIMD_FLOAT64_C(   269.22), EASYSIMD_FLOAT64_C(   874.86) },
      { EASYSIMD_FLOAT64_C(  -553.15), EASYSIMD_FLOAT64_C(   279.93) },
      { EASYSIMD_FLOAT64_C(  -553.15), EASYSIMD_FLOAT64_C(   279.93) } },
    { { EASYSIMD_FLOAT64_C(   178.23), EASYSIMD_FLOAT64_C(  -656.26) },
      UINT8_C( 27),
      { EASYSIMD_FLOAT64_C(   602.93), EASYSIMD_FLOAT64_C(   276.94), EASYSIMD_FLOAT64_C(   416.34), EASYSIMD_FLOAT64_C(    -1.97) },
      { EASYSIMD_FLOAT64_C(   602.93), EASYSIMD_FLOAT64_C(   276.94) },
      { EASYSIMD_FLOAT64_C(   416.34), EASYSIMD_FLOAT64_C(    -1.97) } },
    { { EASYSIMD_FLOAT64_C(   363.21), EASYSIMD_FLOAT64_C(  -398.39) },
      UINT8_C( 35),
      { EASYSIMD_FLOAT64_C(  -747.75), EASYSIMD_FLOAT64_C(   -20.22), EASYSIMD_FLOAT64_C(  -715.03), EASYSIMD_FLOAT64_C(  -797.26) },
      { EASYSIMD_FLOAT64_C(  -747.75), EASYSIMD_FLOAT64_C(   -20.22) },
      { EASYSIMD_FLOAT64_C(  -715.03), EASYSIMD_FLOAT64_C(  -797.26) } },
    { { EASYSIMD_FLOAT64_C(   716.70), EASYSIMD_FLOAT64_C(  -171.22) },
      UINT8_C( 71),
      { EASYSIMD_FLOAT64_C(   424.61), EASYSIMD_FLOAT64_C(   766.25), EASYSIMD_FLOAT64_C(   820.26), EASYSIMD_FLOAT64_C(  -447.49) },
      { EASYSIMD_FLOAT64_C(   424.61), EASYSIMD_FLOAT64_C(   766.25) },
      { EASYSIMD_FLOAT64_C(   820.26), EASYSIMD_FLOAT64_C(  -447.49) } },
    { { EASYSIMD_FLOAT64_C(    42.67), EASYSIMD_FLOAT64_C(   543.64) },
      UINT8_C( 48),
      { EASYSIMD_FLOAT64_C(   489.52), EASYSIMD_FLOAT64_C(  -176.43), EASYSIMD_FLOAT64_C(  -461.17), EASYSIMD_FLOAT64_C(  -991.11) },
      { EASYSIMD_FLOAT64_C(    42.67), EASYSIMD_FLOAT64_C(   543.64) },
      { EASYSIMD_FLOAT64_C(    42.67), EASYSIMD_FLOAT64_C(   543.64) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    r = easysimd_mm256_mask_extractf64x2_pd(src, k, a, 0);
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_extractf64x2_pd(src, k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_extractf64x2_pd");
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128d r0 = easysimd_mm256_mask_extractf64x2_pd(src, k, a, 0);
    easysimd__m128d r1 = easysimd_mm256_mask_extractf64x2_pd(src, k, a, 1);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_extractf64x2_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 r0[2];
    easysimd_float64 r1[2];
  } test_vec[] = {
    { UINT8_C(231),
      { EASYSIMD_FLOAT64_C(   808.05), EASYSIMD_FLOAT64_C(   883.75), EASYSIMD_FLOAT64_C(  -115.04), EASYSIMD_FLOAT64_C(  -848.21) },
      { EASYSIMD_FLOAT64_C(   808.05), EASYSIMD_FLOAT64_C(   883.75) },
      { EASYSIMD_FLOAT64_C(  -115.04), EASYSIMD_FLOAT64_C(  -848.21) } },
    { UINT8_C( 19),
      { EASYSIMD_FLOAT64_C(  -512.11), EASYSIMD_FLOAT64_C(   428.73), EASYSIMD_FLOAT64_C(  -230.00), EASYSIMD_FLOAT64_C(   485.91) },
      { EASYSIMD_FLOAT64_C(  -512.11), EASYSIMD_FLOAT64_C(   428.73) },
      { EASYSIMD_FLOAT64_C(  -230.00), EASYSIMD_FLOAT64_C(   485.91) } },
    { UINT8_C( 43),
      { EASYSIMD_FLOAT64_C(   371.61), EASYSIMD_FLOAT64_C(   701.60), EASYSIMD_FLOAT64_C(    44.20), EASYSIMD_FLOAT64_C(  -648.61) },
      { EASYSIMD_FLOAT64_C(   371.61), EASYSIMD_FLOAT64_C(   701.60) },
      { EASYSIMD_FLOAT64_C(    44.20), EASYSIMD_FLOAT64_C(  -648.61) } },
    { UINT8_C( 53),
      { EASYSIMD_FLOAT64_C(   246.94), EASYSIMD_FLOAT64_C(  -931.91), EASYSIMD_FLOAT64_C(  -184.65), EASYSIMD_FLOAT64_C(   621.86) },
      { EASYSIMD_FLOAT64_C(   246.94), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(  -184.65), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(225),
      { EASYSIMD_FLOAT64_C(  -418.40), EASYSIMD_FLOAT64_C(   442.13), EASYSIMD_FLOAT64_C(  -954.79), EASYSIMD_FLOAT64_C(   624.27) },
      { EASYSIMD_FLOAT64_C(  -418.40), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(  -954.79), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 58),
      { EASYSIMD_FLOAT64_C(    82.53), EASYSIMD_FLOAT64_C(   113.79), EASYSIMD_FLOAT64_C(   809.34), EASYSIMD_FLOAT64_C(   621.36) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   113.79) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   621.36) } },
    { UINT8_C(244),
      { EASYSIMD_FLOAT64_C(   516.07), EASYSIMD_FLOAT64_C(   429.41), EASYSIMD_FLOAT64_C(     6.43), EASYSIMD_FLOAT64_C(  -598.97) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 91),
      { EASYSIMD_FLOAT64_C(  -639.92), EASYSIMD_FLOAT64_C(  -111.08), EASYSIMD_FLOAT64_C(     9.93), EASYSIMD_FLOAT64_C(   130.08) },
      { EASYSIMD_FLOAT64_C(  -639.92), EASYSIMD_FLOAT64_C(  -111.08) },
      { EASYSIMD_FLOAT64_C(     9.93), EASYSIMD_FLOAT64_C(   130.08) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    r = easysimd_mm256_maskz_extractf64x2_pd(k, a, 0);
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_extractf64x2_pd(k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_extractf64x2_pd");
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128d r0 = easysimd_mm256_maskz_extractf64x2_pd(k, a, 0);
    easysimd__m128d r1 = easysimd_mm256_maskz_extractf64x2_pd(k, a, 1);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_extractf32x4_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 a;
    easysimd__m128 r0;
    easysimd__m128 r1;
    easysimd__m128 r2;
    easysimd__m128 r3;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -563.83), EASYSIMD_FLOAT32_C(   799.30), EASYSIMD_FLOAT32_C(   938.85), EASYSIMD_FLOAT32_C(  -576.01),
                         EASYSIMD_FLOAT32_C(  -465.05), EASYSIMD_FLOAT32_C(   439.15), EASYSIMD_FLOAT32_C(  -104.57), EASYSIMD_FLOAT32_C(   -28.15),
                         EASYSIMD_FLOAT32_C(  -431.26), EASYSIMD_FLOAT32_C(   481.25), EASYSIMD_FLOAT32_C(   -57.75), EASYSIMD_FLOAT32_C(  -784.26),
                         EASYSIMD_FLOAT32_C(   438.04), EASYSIMD_FLOAT32_C(   549.03), EASYSIMD_FLOAT32_C(   729.46), EASYSIMD_FLOAT32_C(   582.53)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   438.04), EASYSIMD_FLOAT32_C(   549.03), EASYSIMD_FLOAT32_C(   729.46), EASYSIMD_FLOAT32_C(   582.53)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -431.26), EASYSIMD_FLOAT32_C(   481.25), EASYSIMD_FLOAT32_C(   -57.75), EASYSIMD_FLOAT32_C(  -784.26)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -465.05), EASYSIMD_FLOAT32_C(   439.15), EASYSIMD_FLOAT32_C(  -104.57), EASYSIMD_FLOAT32_C(   -28.15)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -563.83), EASYSIMD_FLOAT32_C(   799.30), EASYSIMD_FLOAT32_C(   938.85), EASYSIMD_FLOAT32_C(  -576.01)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   120.10), EASYSIMD_FLOAT32_C(   -64.06), EASYSIMD_FLOAT32_C(  -620.03), EASYSIMD_FLOAT32_C(   559.81),
                         EASYSIMD_FLOAT32_C(   185.23), EASYSIMD_FLOAT32_C(  -423.61), EASYSIMD_FLOAT32_C(   -11.91), EASYSIMD_FLOAT32_C(   407.56),
                         EASYSIMD_FLOAT32_C(   355.11), EASYSIMD_FLOAT32_C(  -787.72), EASYSIMD_FLOAT32_C(   472.82), EASYSIMD_FLOAT32_C(  -703.51),
                         EASYSIMD_FLOAT32_C(  -202.49), EASYSIMD_FLOAT32_C(  -470.36), EASYSIMD_FLOAT32_C(   966.37), EASYSIMD_FLOAT32_C(   135.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -202.49), EASYSIMD_FLOAT32_C(  -470.36), EASYSIMD_FLOAT32_C(   966.37), EASYSIMD_FLOAT32_C(   135.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   355.11), EASYSIMD_FLOAT32_C(  -787.72), EASYSIMD_FLOAT32_C(   472.82), EASYSIMD_FLOAT32_C(  -703.51)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   185.23), EASYSIMD_FLOAT32_C(  -423.61), EASYSIMD_FLOAT32_C(   -11.91), EASYSIMD_FLOAT32_C(   407.56)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   120.10), EASYSIMD_FLOAT32_C(   -64.06), EASYSIMD_FLOAT32_C(  -620.03), EASYSIMD_FLOAT32_C(   559.81)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   193.01), EASYSIMD_FLOAT32_C(  -435.27), EASYSIMD_FLOAT32_C(   -84.06), EASYSIMD_FLOAT32_C(   298.40),
                         EASYSIMD_FLOAT32_C(   208.07), EASYSIMD_FLOAT32_C(   -94.60), EASYSIMD_FLOAT32_C(   834.28), EASYSIMD_FLOAT32_C(   260.50),
                         EASYSIMD_FLOAT32_C(  -859.51), EASYSIMD_FLOAT32_C(   -69.45), EASYSIMD_FLOAT32_C(    40.36), EASYSIMD_FLOAT32_C(    95.61),
                         EASYSIMD_FLOAT32_C(  -743.10), EASYSIMD_FLOAT32_C(  -688.01), EASYSIMD_FLOAT32_C(   442.76), EASYSIMD_FLOAT32_C(   931.17)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -743.10), EASYSIMD_FLOAT32_C(  -688.01), EASYSIMD_FLOAT32_C(   442.76), EASYSIMD_FLOAT32_C(   931.17)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -859.51), EASYSIMD_FLOAT32_C(   -69.45), EASYSIMD_FLOAT32_C(    40.36), EASYSIMD_FLOAT32_C(    95.61)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   208.07), EASYSIMD_FLOAT32_C(   -94.60), EASYSIMD_FLOAT32_C(   834.28), EASYSIMD_FLOAT32_C(   260.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   193.01), EASYSIMD_FLOAT32_C(  -435.27), EASYSIMD_FLOAT32_C(   -84.06), EASYSIMD_FLOAT32_C(   298.40)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   664.52), EASYSIMD_FLOAT32_C(  -224.13), EASYSIMD_FLOAT32_C(   633.65), EASYSIMD_FLOAT32_C(  -834.15),
                         EASYSIMD_FLOAT32_C(  -157.33), EASYSIMD_FLOAT32_C(  -819.46), EASYSIMD_FLOAT32_C(   541.44), EASYSIMD_FLOAT32_C(   112.81),
                         EASYSIMD_FLOAT32_C(   -98.08), EASYSIMD_FLOAT32_C(   464.19), EASYSIMD_FLOAT32_C(   711.12), EASYSIMD_FLOAT32_C(   282.83),
                         EASYSIMD_FLOAT32_C(  -774.08), EASYSIMD_FLOAT32_C(   841.24), EASYSIMD_FLOAT32_C(  -414.07), EASYSIMD_FLOAT32_C(    79.76)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -774.08), EASYSIMD_FLOAT32_C(   841.24), EASYSIMD_FLOAT32_C(  -414.07), EASYSIMD_FLOAT32_C(    79.76)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -98.08), EASYSIMD_FLOAT32_C(   464.19), EASYSIMD_FLOAT32_C(   711.12), EASYSIMD_FLOAT32_C(   282.83)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -157.33), EASYSIMD_FLOAT32_C(  -819.46), EASYSIMD_FLOAT32_C(   541.44), EASYSIMD_FLOAT32_C(   112.81)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   664.52), EASYSIMD_FLOAT32_C(  -224.13), EASYSIMD_FLOAT32_C(   633.65), EASYSIMD_FLOAT32_C(  -834.15)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   141.08), EASYSIMD_FLOAT32_C(  -832.50), EASYSIMD_FLOAT32_C(  -990.15), EASYSIMD_FLOAT32_C(   438.46),
                         EASYSIMD_FLOAT32_C(  -887.47), EASYSIMD_FLOAT32_C(   336.35), EASYSIMD_FLOAT32_C(  -396.24), EASYSIMD_FLOAT32_C(    99.21),
                         EASYSIMD_FLOAT32_C(    -2.60), EASYSIMD_FLOAT32_C(   -38.88), EASYSIMD_FLOAT32_C(   165.88), EASYSIMD_FLOAT32_C(   218.73),
                         EASYSIMD_FLOAT32_C(   375.27), EASYSIMD_FLOAT32_C(  -966.90), EASYSIMD_FLOAT32_C(  -512.98), EASYSIMD_FLOAT32_C(  -737.78)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   375.27), EASYSIMD_FLOAT32_C(  -966.90), EASYSIMD_FLOAT32_C(  -512.98), EASYSIMD_FLOAT32_C(  -737.78)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    -2.60), EASYSIMD_FLOAT32_C(   -38.88), EASYSIMD_FLOAT32_C(   165.88), EASYSIMD_FLOAT32_C(   218.73)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -887.47), EASYSIMD_FLOAT32_C(   336.35), EASYSIMD_FLOAT32_C(  -396.24), EASYSIMD_FLOAT32_C(    99.21)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   141.08), EASYSIMD_FLOAT32_C(  -832.50), EASYSIMD_FLOAT32_C(  -990.15), EASYSIMD_FLOAT32_C(   438.46)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -179.98), EASYSIMD_FLOAT32_C(   258.23), EASYSIMD_FLOAT32_C(   246.22), EASYSIMD_FLOAT32_C(    97.85),
                         EASYSIMD_FLOAT32_C(   666.32), EASYSIMD_FLOAT32_C(   364.80), EASYSIMD_FLOAT32_C(   759.27), EASYSIMD_FLOAT32_C(  -524.19),
                         EASYSIMD_FLOAT32_C(  -726.51), EASYSIMD_FLOAT32_C(   381.71), EASYSIMD_FLOAT32_C(   819.12), EASYSIMD_FLOAT32_C(   145.28),
                         EASYSIMD_FLOAT32_C(   -99.37), EASYSIMD_FLOAT32_C(  -151.02), EASYSIMD_FLOAT32_C(   551.65), EASYSIMD_FLOAT32_C(   155.58)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -99.37), EASYSIMD_FLOAT32_C(  -151.02), EASYSIMD_FLOAT32_C(   551.65), EASYSIMD_FLOAT32_C(   155.58)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -726.51), EASYSIMD_FLOAT32_C(   381.71), EASYSIMD_FLOAT32_C(   819.12), EASYSIMD_FLOAT32_C(   145.28)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   666.32), EASYSIMD_FLOAT32_C(   364.80), EASYSIMD_FLOAT32_C(   759.27), EASYSIMD_FLOAT32_C(  -524.19)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -179.98), EASYSIMD_FLOAT32_C(   258.23), EASYSIMD_FLOAT32_C(   246.22), EASYSIMD_FLOAT32_C(    97.85)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   254.48), EASYSIMD_FLOAT32_C(  -211.73), EASYSIMD_FLOAT32_C(   755.70), EASYSIMD_FLOAT32_C(   908.31),
                         EASYSIMD_FLOAT32_C(  -363.93), EASYSIMD_FLOAT32_C(  -144.11), EASYSIMD_FLOAT32_C(   789.10), EASYSIMD_FLOAT32_C(  -343.92),
                         EASYSIMD_FLOAT32_C(   344.74), EASYSIMD_FLOAT32_C(   961.65), EASYSIMD_FLOAT32_C(   652.93), EASYSIMD_FLOAT32_C(   754.42),
                         EASYSIMD_FLOAT32_C(   184.91), EASYSIMD_FLOAT32_C(  -432.97), EASYSIMD_FLOAT32_C(  -455.33), EASYSIMD_FLOAT32_C(   164.52)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   184.91), EASYSIMD_FLOAT32_C(  -432.97), EASYSIMD_FLOAT32_C(  -455.33), EASYSIMD_FLOAT32_C(   164.52)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   344.74), EASYSIMD_FLOAT32_C(   961.65), EASYSIMD_FLOAT32_C(   652.93), EASYSIMD_FLOAT32_C(   754.42)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -363.93), EASYSIMD_FLOAT32_C(  -144.11), EASYSIMD_FLOAT32_C(   789.10), EASYSIMD_FLOAT32_C(  -343.92)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   254.48), EASYSIMD_FLOAT32_C(  -211.73), EASYSIMD_FLOAT32_C(   755.70), EASYSIMD_FLOAT32_C(   908.31)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -985.32), EASYSIMD_FLOAT32_C(   485.76), EASYSIMD_FLOAT32_C(   234.60), EASYSIMD_FLOAT32_C(   786.03),
                         EASYSIMD_FLOAT32_C(   859.59), EASYSIMD_FLOAT32_C(   489.95), EASYSIMD_FLOAT32_C(  -409.35), EASYSIMD_FLOAT32_C(   796.52),
                         EASYSIMD_FLOAT32_C(  -846.10), EASYSIMD_FLOAT32_C(  -248.07), EASYSIMD_FLOAT32_C(  -411.92), EASYSIMD_FLOAT32_C(   -88.91),
                         EASYSIMD_FLOAT32_C(   481.68), EASYSIMD_FLOAT32_C(   170.00), EASYSIMD_FLOAT32_C(  -341.91), EASYSIMD_FLOAT32_C(   366.57)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   481.68), EASYSIMD_FLOAT32_C(   170.00), EASYSIMD_FLOAT32_C(  -341.91), EASYSIMD_FLOAT32_C(   366.57)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -846.10), EASYSIMD_FLOAT32_C(  -248.07), EASYSIMD_FLOAT32_C(  -411.92), EASYSIMD_FLOAT32_C(   -88.91)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   859.59), EASYSIMD_FLOAT32_C(   489.95), EASYSIMD_FLOAT32_C(  -409.35), EASYSIMD_FLOAT32_C(   796.52)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -985.32), EASYSIMD_FLOAT32_C(   485.76), EASYSIMD_FLOAT32_C(   234.60), EASYSIMD_FLOAT32_C(   786.03)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r;
    r = easysimd_mm512_extractf32x4_ps(test_vec[i].a, 0);
    easysimd_assert_m128_close(r, test_vec[i].r0, 1);
    r = easysimd_mm512_extractf32x4_ps(test_vec[i].a, 1);
    easysimd_assert_m128_close(r, test_vec[i].r1, 1);
    r = easysimd_mm512_extractf32x4_ps(test_vec[i].a, 2);
    easysimd_assert_m128_close(r, test_vec[i].r2, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extractf32x4_ps(test_vec[i].a, 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extractf32x4_ps");
    easysimd_assert_m128_close(r, test_vec[i].r3, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_extractf32x4_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 src;
    easysimd__mmask8 k;
    easysimd__m512 a;
    easysimd__m128 r0;
    easysimd__m128 r1;
    easysimd__m128 r2;
    easysimd__m128 r3;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -172.36), EASYSIMD_FLOAT32_C(   393.53), EASYSIMD_FLOAT32_C(    36.69), EASYSIMD_FLOAT32_C(  -135.52)),
      UINT8_C( 25),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   903.50), EASYSIMD_FLOAT32_C(   -43.35), EASYSIMD_FLOAT32_C(   309.91), EASYSIMD_FLOAT32_C(   846.15),
                         EASYSIMD_FLOAT32_C(  -514.56), EASYSIMD_FLOAT32_C(  -860.98), EASYSIMD_FLOAT32_C(  -280.30), EASYSIMD_FLOAT32_C(   128.51),
                         EASYSIMD_FLOAT32_C(   522.06), EASYSIMD_FLOAT32_C(  -932.28), EASYSIMD_FLOAT32_C(   600.12), EASYSIMD_FLOAT32_C(  -491.12),
                         EASYSIMD_FLOAT32_C(  -139.11), EASYSIMD_FLOAT32_C(  -268.86), EASYSIMD_FLOAT32_C(   -71.72), EASYSIMD_FLOAT32_C(    98.47)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -139.11), EASYSIMD_FLOAT32_C(   393.53), EASYSIMD_FLOAT32_C(    36.69), EASYSIMD_FLOAT32_C(    98.47)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   522.06), EASYSIMD_FLOAT32_C(   393.53), EASYSIMD_FLOAT32_C(    36.69), EASYSIMD_FLOAT32_C(  -491.12)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -514.56), EASYSIMD_FLOAT32_C(   393.53), EASYSIMD_FLOAT32_C(    36.69), EASYSIMD_FLOAT32_C(   128.51)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   903.50), EASYSIMD_FLOAT32_C(   393.53), EASYSIMD_FLOAT32_C(    36.69), EASYSIMD_FLOAT32_C(   846.15)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -895.71), EASYSIMD_FLOAT32_C(  -736.92), EASYSIMD_FLOAT32_C(   283.06), EASYSIMD_FLOAT32_C(  -333.94)),
      UINT8_C( 61),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   337.35), EASYSIMD_FLOAT32_C(  -278.32), EASYSIMD_FLOAT32_C(  -744.41), EASYSIMD_FLOAT32_C(    39.32),
                         EASYSIMD_FLOAT32_C(    29.68), EASYSIMD_FLOAT32_C(  -490.28), EASYSIMD_FLOAT32_C(   841.53), EASYSIMD_FLOAT32_C(   526.21),
                         EASYSIMD_FLOAT32_C(  -203.04), EASYSIMD_FLOAT32_C(   -80.71), EASYSIMD_FLOAT32_C(   632.01), EASYSIMD_FLOAT32_C(   456.89),
                         EASYSIMD_FLOAT32_C(    51.33), EASYSIMD_FLOAT32_C(  -868.59), EASYSIMD_FLOAT32_C(  -921.00), EASYSIMD_FLOAT32_C(  -471.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    51.33), EASYSIMD_FLOAT32_C(  -868.59), EASYSIMD_FLOAT32_C(   283.06), EASYSIMD_FLOAT32_C(  -471.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -203.04), EASYSIMD_FLOAT32_C(   -80.71), EASYSIMD_FLOAT32_C(   283.06), EASYSIMD_FLOAT32_C(   456.89)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    29.68), EASYSIMD_FLOAT32_C(  -490.28), EASYSIMD_FLOAT32_C(   283.06), EASYSIMD_FLOAT32_C(   526.21)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   337.35), EASYSIMD_FLOAT32_C(  -278.32), EASYSIMD_FLOAT32_C(   283.06), EASYSIMD_FLOAT32_C(    39.32)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   957.37), EASYSIMD_FLOAT32_C(  -934.92), EASYSIMD_FLOAT32_C(  -657.02), EASYSIMD_FLOAT32_C(  -629.37)),
      UINT8_C(214),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -207.87), EASYSIMD_FLOAT32_C(  -765.42), EASYSIMD_FLOAT32_C(   138.83), EASYSIMD_FLOAT32_C(   699.07),
                         EASYSIMD_FLOAT32_C(  -143.73), EASYSIMD_FLOAT32_C(   709.96), EASYSIMD_FLOAT32_C(  -767.34), EASYSIMD_FLOAT32_C(  -588.28),
                         EASYSIMD_FLOAT32_C(   586.29), EASYSIMD_FLOAT32_C(  -760.88), EASYSIMD_FLOAT32_C(  -617.12), EASYSIMD_FLOAT32_C(  -751.58),
                         EASYSIMD_FLOAT32_C(   907.23), EASYSIMD_FLOAT32_C(  -359.60), EASYSIMD_FLOAT32_C(  -213.75), EASYSIMD_FLOAT32_C(   403.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   957.37), EASYSIMD_FLOAT32_C(  -359.60), EASYSIMD_FLOAT32_C(  -213.75), EASYSIMD_FLOAT32_C(  -629.37)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   957.37), EASYSIMD_FLOAT32_C(  -760.88), EASYSIMD_FLOAT32_C(  -617.12), EASYSIMD_FLOAT32_C(  -629.37)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   957.37), EASYSIMD_FLOAT32_C(   709.96), EASYSIMD_FLOAT32_C(  -767.34), EASYSIMD_FLOAT32_C(  -629.37)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   957.37), EASYSIMD_FLOAT32_C(  -765.42), EASYSIMD_FLOAT32_C(   138.83), EASYSIMD_FLOAT32_C(  -629.37)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   799.57), EASYSIMD_FLOAT32_C(  -820.22), EASYSIMD_FLOAT32_C(  -959.11), EASYSIMD_FLOAT32_C(   268.99)),
      UINT8_C(196),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -659.70), EASYSIMD_FLOAT32_C(   493.30), EASYSIMD_FLOAT32_C(   831.29), EASYSIMD_FLOAT32_C(  -619.50),
                         EASYSIMD_FLOAT32_C(   952.47), EASYSIMD_FLOAT32_C(  -492.61), EASYSIMD_FLOAT32_C(   -68.16), EASYSIMD_FLOAT32_C(   717.69),
                         EASYSIMD_FLOAT32_C(  -663.74), EASYSIMD_FLOAT32_C(   179.29), EASYSIMD_FLOAT32_C(   989.70), EASYSIMD_FLOAT32_C(  -695.21),
                         EASYSIMD_FLOAT32_C(  -786.23), EASYSIMD_FLOAT32_C(   873.30), EASYSIMD_FLOAT32_C(   241.45), EASYSIMD_FLOAT32_C(  -432.13)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   799.57), EASYSIMD_FLOAT32_C(   873.30), EASYSIMD_FLOAT32_C(  -959.11), EASYSIMD_FLOAT32_C(   268.99)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   799.57), EASYSIMD_FLOAT32_C(   179.29), EASYSIMD_FLOAT32_C(  -959.11), EASYSIMD_FLOAT32_C(   268.99)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   799.57), EASYSIMD_FLOAT32_C(  -492.61), EASYSIMD_FLOAT32_C(  -959.11), EASYSIMD_FLOAT32_C(   268.99)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   799.57), EASYSIMD_FLOAT32_C(   493.30), EASYSIMD_FLOAT32_C(  -959.11), EASYSIMD_FLOAT32_C(   268.99)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -789.54), EASYSIMD_FLOAT32_C(  -790.16), EASYSIMD_FLOAT32_C(  -415.61), EASYSIMD_FLOAT32_C(   994.61)),
      UINT8_C(  8),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -388.47), EASYSIMD_FLOAT32_C(  -643.43), EASYSIMD_FLOAT32_C(  -331.34), EASYSIMD_FLOAT32_C(    72.67),
                         EASYSIMD_FLOAT32_C(  -870.79), EASYSIMD_FLOAT32_C(  -722.44), EASYSIMD_FLOAT32_C(   529.44), EASYSIMD_FLOAT32_C(  -949.73),
                         EASYSIMD_FLOAT32_C(   280.87), EASYSIMD_FLOAT32_C(   380.83), EASYSIMD_FLOAT32_C(  -236.67), EASYSIMD_FLOAT32_C(  -211.91),
                         EASYSIMD_FLOAT32_C(  -925.76), EASYSIMD_FLOAT32_C(  -915.62), EASYSIMD_FLOAT32_C(   -30.05), EASYSIMD_FLOAT32_C(   -70.79)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -925.76), EASYSIMD_FLOAT32_C(  -790.16), EASYSIMD_FLOAT32_C(  -415.61), EASYSIMD_FLOAT32_C(   994.61)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   280.87), EASYSIMD_FLOAT32_C(  -790.16), EASYSIMD_FLOAT32_C(  -415.61), EASYSIMD_FLOAT32_C(   994.61)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -870.79), EASYSIMD_FLOAT32_C(  -790.16), EASYSIMD_FLOAT32_C(  -415.61), EASYSIMD_FLOAT32_C(   994.61)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -388.47), EASYSIMD_FLOAT32_C(  -790.16), EASYSIMD_FLOAT32_C(  -415.61), EASYSIMD_FLOAT32_C(   994.61)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -352.24), EASYSIMD_FLOAT32_C(  -479.79), EASYSIMD_FLOAT32_C(   602.83), EASYSIMD_FLOAT32_C(     2.55)),
      UINT8_C( 60),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -607.82), EASYSIMD_FLOAT32_C(   296.47), EASYSIMD_FLOAT32_C(  -327.04), EASYSIMD_FLOAT32_C(   -23.06),
                         EASYSIMD_FLOAT32_C(   -95.21), EASYSIMD_FLOAT32_C(    10.75), EASYSIMD_FLOAT32_C(  -668.43), EASYSIMD_FLOAT32_C(  -210.00),
                         EASYSIMD_FLOAT32_C(   915.68), EASYSIMD_FLOAT32_C(   -53.79), EASYSIMD_FLOAT32_C(   703.31), EASYSIMD_FLOAT32_C(   930.79),
                         EASYSIMD_FLOAT32_C(   111.33), EASYSIMD_FLOAT32_C(  -176.75), EASYSIMD_FLOAT32_C(  -316.94), EASYSIMD_FLOAT32_C(   639.68)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   111.33), EASYSIMD_FLOAT32_C(  -176.75), EASYSIMD_FLOAT32_C(   602.83), EASYSIMD_FLOAT32_C(     2.55)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   915.68), EASYSIMD_FLOAT32_C(   -53.79), EASYSIMD_FLOAT32_C(   602.83), EASYSIMD_FLOAT32_C(     2.55)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -95.21), EASYSIMD_FLOAT32_C(    10.75), EASYSIMD_FLOAT32_C(   602.83), EASYSIMD_FLOAT32_C(     2.55)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -607.82), EASYSIMD_FLOAT32_C(   296.47), EASYSIMD_FLOAT32_C(   602.83), EASYSIMD_FLOAT32_C(     2.55)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -334.42), EASYSIMD_FLOAT32_C(   660.53), EASYSIMD_FLOAT32_C(   748.73), EASYSIMD_FLOAT32_C(   996.15)),
      UINT8_C( 47),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   383.31), EASYSIMD_FLOAT32_C(   641.22), EASYSIMD_FLOAT32_C(  -747.07), EASYSIMD_FLOAT32_C(  -762.67),
                         EASYSIMD_FLOAT32_C(   744.11), EASYSIMD_FLOAT32_C(   350.11), EASYSIMD_FLOAT32_C(   409.27), EASYSIMD_FLOAT32_C(   481.83),
                         EASYSIMD_FLOAT32_C(   601.37), EASYSIMD_FLOAT32_C(  -660.24), EASYSIMD_FLOAT32_C(  -675.56), EASYSIMD_FLOAT32_C(  -194.09),
                         EASYSIMD_FLOAT32_C(   149.22), EASYSIMD_FLOAT32_C(   161.52), EASYSIMD_FLOAT32_C(   632.78), EASYSIMD_FLOAT32_C(   346.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   149.22), EASYSIMD_FLOAT32_C(   161.52), EASYSIMD_FLOAT32_C(   632.78), EASYSIMD_FLOAT32_C(   346.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   601.37), EASYSIMD_FLOAT32_C(  -660.24), EASYSIMD_FLOAT32_C(  -675.56), EASYSIMD_FLOAT32_C(  -194.09)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   744.11), EASYSIMD_FLOAT32_C(   350.11), EASYSIMD_FLOAT32_C(   409.27), EASYSIMD_FLOAT32_C(   481.83)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   383.31), EASYSIMD_FLOAT32_C(   641.22), EASYSIMD_FLOAT32_C(  -747.07), EASYSIMD_FLOAT32_C(  -762.67)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   122.69), EASYSIMD_FLOAT32_C(    65.13), EASYSIMD_FLOAT32_C(  -972.27), EASYSIMD_FLOAT32_C(   628.22)),
      UINT8_C(171),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -653.71), EASYSIMD_FLOAT32_C(   371.73), EASYSIMD_FLOAT32_C(   757.18), EASYSIMD_FLOAT32_C(   214.84),
                         EASYSIMD_FLOAT32_C(   830.24), EASYSIMD_FLOAT32_C(   903.53), EASYSIMD_FLOAT32_C(  -831.08), EASYSIMD_FLOAT32_C(   815.07),
                         EASYSIMD_FLOAT32_C(   196.06), EASYSIMD_FLOAT32_C(   -83.06), EASYSIMD_FLOAT32_C(   687.82), EASYSIMD_FLOAT32_C(  -517.82),
                         EASYSIMD_FLOAT32_C(  -294.36), EASYSIMD_FLOAT32_C(   702.71), EASYSIMD_FLOAT32_C(  -920.22), EASYSIMD_FLOAT32_C(  -923.04)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -294.36), EASYSIMD_FLOAT32_C(    65.13), EASYSIMD_FLOAT32_C(  -920.22), EASYSIMD_FLOAT32_C(  -923.04)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   196.06), EASYSIMD_FLOAT32_C(    65.13), EASYSIMD_FLOAT32_C(   687.82), EASYSIMD_FLOAT32_C(  -517.82)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   830.24), EASYSIMD_FLOAT32_C(    65.13), EASYSIMD_FLOAT32_C(  -831.08), EASYSIMD_FLOAT32_C(   815.07)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -653.71), EASYSIMD_FLOAT32_C(    65.13), EASYSIMD_FLOAT32_C(   757.18), EASYSIMD_FLOAT32_C(   214.84)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r;
    r = easysimd_mm512_mask_extractf32x4_ps(test_vec[i].src, test_vec[i].k, test_vec[i].a, 0);
    easysimd_assert_m128_close(r, test_vec[i].r0, 1);
    r = easysimd_mm512_mask_extractf32x4_ps(test_vec[i].src, test_vec[i].k, test_vec[i].a, 1);
    easysimd_assert_m128_close(r, test_vec[i].r1, 1);
    r = easysimd_mm512_mask_extractf32x4_ps(test_vec[i].src, test_vec[i].k, test_vec[i].a, 2);
    easysimd_assert_m128_close(r, test_vec[i].r2, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extractf32x4_ps(test_vec[i].src, test_vec[i].k, test_vec[i].a, 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extractf32x4_ps");
    easysimd_assert_m128_close(r, test_vec[i].r3, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_extractf32x4_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512 a;
    easysimd__m128 r0;
    easysimd__m128 r1;
    easysimd__m128 r2;
    easysimd__m128 r3;
  } test_vec[8] = {
    { UINT8_C( 63),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   522.06), EASYSIMD_FLOAT32_C(   160.98), EASYSIMD_FLOAT32_C(  -932.28), EASYSIMD_FLOAT32_C(   391.82),
                         EASYSIMD_FLOAT32_C(   600.12), EASYSIMD_FLOAT32_C(  -569.99), EASYSIMD_FLOAT32_C(  -491.12), EASYSIMD_FLOAT32_C(  -327.63),
                         EASYSIMD_FLOAT32_C(  -139.11), EASYSIMD_FLOAT32_C(  -172.36), EASYSIMD_FLOAT32_C(  -268.86), EASYSIMD_FLOAT32_C(   393.53),
                         EASYSIMD_FLOAT32_C(   -71.72), EASYSIMD_FLOAT32_C(    36.69), EASYSIMD_FLOAT32_C(    98.47), EASYSIMD_FLOAT32_C(  -135.52)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -71.72), EASYSIMD_FLOAT32_C(    36.69), EASYSIMD_FLOAT32_C(    98.47), EASYSIMD_FLOAT32_C(  -135.52)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -139.11), EASYSIMD_FLOAT32_C(  -172.36), EASYSIMD_FLOAT32_C(  -268.86), EASYSIMD_FLOAT32_C(   393.53)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   600.12), EASYSIMD_FLOAT32_C(  -569.99), EASYSIMD_FLOAT32_C(  -491.12), EASYSIMD_FLOAT32_C(  -327.63)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   522.06), EASYSIMD_FLOAT32_C(   160.98), EASYSIMD_FLOAT32_C(  -932.28), EASYSIMD_FLOAT32_C(   391.82)) },
    { UINT8_C(157),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   483.08), EASYSIMD_FLOAT32_C(   903.50), EASYSIMD_FLOAT32_C(   232.04), EASYSIMD_FLOAT32_C(   -43.35),
                         EASYSIMD_FLOAT32_C(   774.81), EASYSIMD_FLOAT32_C(   309.91), EASYSIMD_FLOAT32_C(  -599.01), EASYSIMD_FLOAT32_C(   846.15),
                         EASYSIMD_FLOAT32_C(    69.04), EASYSIMD_FLOAT32_C(  -514.56), EASYSIMD_FLOAT32_C(  -149.02), EASYSIMD_FLOAT32_C(  -860.98),
                         EASYSIMD_FLOAT32_C(   240.79), EASYSIMD_FLOAT32_C(  -280.30), EASYSIMD_FLOAT32_C(  -839.80), EASYSIMD_FLOAT32_C(   128.51)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   240.79), EASYSIMD_FLOAT32_C(  -280.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   128.51)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    69.04), EASYSIMD_FLOAT32_C(  -514.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -860.98)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   774.81), EASYSIMD_FLOAT32_C(   309.91), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   846.15)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   483.08), EASYSIMD_FLOAT32_C(   903.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -43.35)) },
    { UINT8_C( 33),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -835.53), EASYSIMD_FLOAT32_C(  -203.04), EASYSIMD_FLOAT32_C(   571.79), EASYSIMD_FLOAT32_C(   -80.71),
                         EASYSIMD_FLOAT32_C(   675.92), EASYSIMD_FLOAT32_C(   632.01), EASYSIMD_FLOAT32_C(   490.41), EASYSIMD_FLOAT32_C(   456.89),
                         EASYSIMD_FLOAT32_C(    47.59), EASYSIMD_FLOAT32_C(    51.33), EASYSIMD_FLOAT32_C(  -895.71), EASYSIMD_FLOAT32_C(  -868.59),
                         EASYSIMD_FLOAT32_C(  -736.92), EASYSIMD_FLOAT32_C(  -921.00), EASYSIMD_FLOAT32_C(   283.06), EASYSIMD_FLOAT32_C(  -471.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -471.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -868.59)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   456.89)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -80.71)) },
    { UINT8_C(176),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -629.37), EASYSIMD_FLOAT32_C(  -198.67), EASYSIMD_FLOAT32_C(   337.35), EASYSIMD_FLOAT32_C(   447.98),
                         EASYSIMD_FLOAT32_C(  -278.32), EASYSIMD_FLOAT32_C(  -925.69), EASYSIMD_FLOAT32_C(  -744.41), EASYSIMD_FLOAT32_C(   717.83),
                         EASYSIMD_FLOAT32_C(    39.32), EASYSIMD_FLOAT32_C(  -489.88), EASYSIMD_FLOAT32_C(    29.68), EASYSIMD_FLOAT32_C(   -37.49),
                         EASYSIMD_FLOAT32_C(  -490.28), EASYSIMD_FLOAT32_C(  -373.66), EASYSIMD_FLOAT32_C(   841.53), EASYSIMD_FLOAT32_C(  -292.35)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C(169),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -588.28), EASYSIMD_FLOAT32_C(   846.67), EASYSIMD_FLOAT32_C(   586.29), EASYSIMD_FLOAT32_C(   670.52),
                         EASYSIMD_FLOAT32_C(  -760.88), EASYSIMD_FLOAT32_C(   149.72), EASYSIMD_FLOAT32_C(  -617.12), EASYSIMD_FLOAT32_C(   213.24),
                         EASYSIMD_FLOAT32_C(  -751.58), EASYSIMD_FLOAT32_C(  -577.36), EASYSIMD_FLOAT32_C(   907.23), EASYSIMD_FLOAT32_C(   957.37),
                         EASYSIMD_FLOAT32_C(  -359.60), EASYSIMD_FLOAT32_C(  -934.92), EASYSIMD_FLOAT32_C(  -213.75), EASYSIMD_FLOAT32_C(  -657.02)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -359.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -657.02)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -751.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   957.37)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -760.88), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   213.24)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -588.28), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   670.52)) },
    { UINT8_C( 52),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -432.13), EASYSIMD_FLOAT32_C(   268.99), EASYSIMD_FLOAT32_C(  -842.15), EASYSIMD_FLOAT32_C(  -207.87),
                         EASYSIMD_FLOAT32_C(   908.84), EASYSIMD_FLOAT32_C(  -765.42), EASYSIMD_FLOAT32_C(  -315.78), EASYSIMD_FLOAT32_C(   138.83),
                         EASYSIMD_FLOAT32_C(   -86.06), EASYSIMD_FLOAT32_C(   699.07), EASYSIMD_FLOAT32_C(  -413.85), EASYSIMD_FLOAT32_C(  -143.73),
                         EASYSIMD_FLOAT32_C(   752.26), EASYSIMD_FLOAT32_C(   709.96), EASYSIMD_FLOAT32_C(   609.29), EASYSIMD_FLOAT32_C(  -767.34)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   709.96), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   699.07), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -765.42), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   268.99), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C(217),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   180.78), EASYSIMD_FLOAT32_C(   717.69), EASYSIMD_FLOAT32_C(  -289.23), EASYSIMD_FLOAT32_C(  -663.74),
                         EASYSIMD_FLOAT32_C(   918.52), EASYSIMD_FLOAT32_C(   179.29), EASYSIMD_FLOAT32_C(  -422.76), EASYSIMD_FLOAT32_C(   989.70),
                         EASYSIMD_FLOAT32_C(  -433.33), EASYSIMD_FLOAT32_C(  -695.21), EASYSIMD_FLOAT32_C(    48.49), EASYSIMD_FLOAT32_C(  -786.23),
                         EASYSIMD_FLOAT32_C(   799.57), EASYSIMD_FLOAT32_C(   873.30), EASYSIMD_FLOAT32_C(  -820.22), EASYSIMD_FLOAT32_C(   241.45)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   799.57), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   241.45)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -433.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -786.23)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   918.52), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   989.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   180.78), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -663.74)) },
    { UINT8_C(237),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -415.61), EASYSIMD_FLOAT32_C(   -70.79), EASYSIMD_FLOAT32_C(   994.61), EASYSIMD_FLOAT32_C(   493.65),
                         EASYSIMD_FLOAT32_C(  -659.70), EASYSIMD_FLOAT32_C(    52.79), EASYSIMD_FLOAT32_C(   493.30), EASYSIMD_FLOAT32_C(   835.54),
                         EASYSIMD_FLOAT32_C(   831.29), EASYSIMD_FLOAT32_C(  -712.24), EASYSIMD_FLOAT32_C(  -619.50), EASYSIMD_FLOAT32_C(   518.12),
                         EASYSIMD_FLOAT32_C(   952.47), EASYSIMD_FLOAT32_C(  -173.80), EASYSIMD_FLOAT32_C(  -492.61), EASYSIMD_FLOAT32_C(   487.08)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   952.47), EASYSIMD_FLOAT32_C(  -173.80), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   487.08)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   831.29), EASYSIMD_FLOAT32_C(  -712.24), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   518.12)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -659.70), EASYSIMD_FLOAT32_C(    52.79), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   835.54)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -415.61), EASYSIMD_FLOAT32_C(   -70.79), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   493.65)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r;
    r = easysimd_mm512_maskz_extractf32x4_ps(test_vec[i].k, test_vec[i].a, 0);
    easysimd_assert_m128_close(r, test_vec[i].r0, 1);
    r = easysimd_mm512_maskz_extractf32x4_ps(test_vec[i].k, test_vec[i].a, 1);
    easysimd_assert_m128_close(r, test_vec[i].r1, 1);
    r = easysimd_mm512_maskz_extractf32x4_ps(test_vec[i].k, test_vec[i].a, 2);
    easysimd_assert_m128_close(r, test_vec[i].r2, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extractf32x4_ps(test_vec[i].k, test_vec[i].a, 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extractf32x4_ps");
    easysimd_assert_m128_close(r, test_vec[i].r3, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_extractf32x8_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 a;
    easysimd__m256 r0;
    easysimd__m256 r1;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -884.15), EASYSIMD_FLOAT32_C(  -590.68), EASYSIMD_FLOAT32_C(   609.01), EASYSIMD_FLOAT32_C(   270.76),
                         EASYSIMD_FLOAT32_C(  -283.08), EASYSIMD_FLOAT32_C(   459.83), EASYSIMD_FLOAT32_C(   895.85), EASYSIMD_FLOAT32_C(   841.15),
                         EASYSIMD_FLOAT32_C(   601.63), EASYSIMD_FLOAT32_C(  -274.13), EASYSIMD_FLOAT32_C(   609.93), EASYSIMD_FLOAT32_C(   258.59),
                         EASYSIMD_FLOAT32_C(   -89.35), EASYSIMD_FLOAT32_C(   965.70), EASYSIMD_FLOAT32_C(  -530.41), EASYSIMD_FLOAT32_C(   486.93)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   601.63), EASYSIMD_FLOAT32_C(  -274.13), EASYSIMD_FLOAT32_C(   609.93), EASYSIMD_FLOAT32_C(   258.59),
                         EASYSIMD_FLOAT32_C(   -89.35), EASYSIMD_FLOAT32_C(   965.70), EASYSIMD_FLOAT32_C(  -530.41), EASYSIMD_FLOAT32_C(   486.93)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -884.15), EASYSIMD_FLOAT32_C(  -590.68), EASYSIMD_FLOAT32_C(   609.01), EASYSIMD_FLOAT32_C(   270.76),
                         EASYSIMD_FLOAT32_C(  -283.08), EASYSIMD_FLOAT32_C(   459.83), EASYSIMD_FLOAT32_C(   895.85), EASYSIMD_FLOAT32_C(   841.15)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   992.87), EASYSIMD_FLOAT32_C(   428.37), EASYSIMD_FLOAT32_C(    84.23), EASYSIMD_FLOAT32_C(  -526.39),
                         EASYSIMD_FLOAT32_C(   512.49), EASYSIMD_FLOAT32_C(  -378.17), EASYSIMD_FLOAT32_C(   265.19), EASYSIMD_FLOAT32_C(  -361.71),
                         EASYSIMD_FLOAT32_C(   630.78), EASYSIMD_FLOAT32_C(   764.67), EASYSIMD_FLOAT32_C(  -523.60), EASYSIMD_FLOAT32_C(   302.14),
                         EASYSIMD_FLOAT32_C(  -536.13), EASYSIMD_FLOAT32_C(   212.47), EASYSIMD_FLOAT32_C(   795.66), EASYSIMD_FLOAT32_C(  -420.28)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   630.78), EASYSIMD_FLOAT32_C(   764.67), EASYSIMD_FLOAT32_C(  -523.60), EASYSIMD_FLOAT32_C(   302.14),
                         EASYSIMD_FLOAT32_C(  -536.13), EASYSIMD_FLOAT32_C(   212.47), EASYSIMD_FLOAT32_C(   795.66), EASYSIMD_FLOAT32_C(  -420.28)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   992.87), EASYSIMD_FLOAT32_C(   428.37), EASYSIMD_FLOAT32_C(    84.23), EASYSIMD_FLOAT32_C(  -526.39),
                         EASYSIMD_FLOAT32_C(   512.49), EASYSIMD_FLOAT32_C(  -378.17), EASYSIMD_FLOAT32_C(   265.19), EASYSIMD_FLOAT32_C(  -361.71)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   621.80), EASYSIMD_FLOAT32_C(   404.67), EASYSIMD_FLOAT32_C(   850.48), EASYSIMD_FLOAT32_C(  -661.28),
                         EASYSIMD_FLOAT32_C(  -135.50), EASYSIMD_FLOAT32_C(   746.33), EASYSIMD_FLOAT32_C(  -820.13), EASYSIMD_FLOAT32_C(  -533.87),
                         EASYSIMD_FLOAT32_C(  -527.80), EASYSIMD_FLOAT32_C(   789.80), EASYSIMD_FLOAT32_C(   724.72), EASYSIMD_FLOAT32_C(   382.85),
                         EASYSIMD_FLOAT32_C(   755.50), EASYSIMD_FLOAT32_C(  -805.70), EASYSIMD_FLOAT32_C(  -130.22), EASYSIMD_FLOAT32_C(   748.36)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -527.80), EASYSIMD_FLOAT32_C(   789.80), EASYSIMD_FLOAT32_C(   724.72), EASYSIMD_FLOAT32_C(   382.85),
                         EASYSIMD_FLOAT32_C(   755.50), EASYSIMD_FLOAT32_C(  -805.70), EASYSIMD_FLOAT32_C(  -130.22), EASYSIMD_FLOAT32_C(   748.36)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   621.80), EASYSIMD_FLOAT32_C(   404.67), EASYSIMD_FLOAT32_C(   850.48), EASYSIMD_FLOAT32_C(  -661.28),
                         EASYSIMD_FLOAT32_C(  -135.50), EASYSIMD_FLOAT32_C(   746.33), EASYSIMD_FLOAT32_C(  -820.13), EASYSIMD_FLOAT32_C(  -533.87)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   622.68), EASYSIMD_FLOAT32_C(   954.01), EASYSIMD_FLOAT32_C(  -778.03), EASYSIMD_FLOAT32_C(   135.16),
                         EASYSIMD_FLOAT32_C(  -424.16), EASYSIMD_FLOAT32_C(   487.16), EASYSIMD_FLOAT32_C(   773.45), EASYSIMD_FLOAT32_C(  -793.38),
                         EASYSIMD_FLOAT32_C(   251.83), EASYSIMD_FLOAT32_C(  -750.14), EASYSIMD_FLOAT32_C(   508.76), EASYSIMD_FLOAT32_C(   715.70),
                         EASYSIMD_FLOAT32_C(   462.33), EASYSIMD_FLOAT32_C(   304.42), EASYSIMD_FLOAT32_C(  -704.58), EASYSIMD_FLOAT32_C(    84.12)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   251.83), EASYSIMD_FLOAT32_C(  -750.14), EASYSIMD_FLOAT32_C(   508.76), EASYSIMD_FLOAT32_C(   715.70),
                         EASYSIMD_FLOAT32_C(   462.33), EASYSIMD_FLOAT32_C(   304.42), EASYSIMD_FLOAT32_C(  -704.58), EASYSIMD_FLOAT32_C(    84.12)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   622.68), EASYSIMD_FLOAT32_C(   954.01), EASYSIMD_FLOAT32_C(  -778.03), EASYSIMD_FLOAT32_C(   135.16),
                         EASYSIMD_FLOAT32_C(  -424.16), EASYSIMD_FLOAT32_C(   487.16), EASYSIMD_FLOAT32_C(   773.45), EASYSIMD_FLOAT32_C(  -793.38)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -290.91), EASYSIMD_FLOAT32_C(  -854.10), EASYSIMD_FLOAT32_C(   422.85), EASYSIMD_FLOAT32_C(   573.59),
                         EASYSIMD_FLOAT32_C(   892.23), EASYSIMD_FLOAT32_C(   602.71), EASYSIMD_FLOAT32_C(  -960.28), EASYSIMD_FLOAT32_C(  -635.56),
                         EASYSIMD_FLOAT32_C(   392.51), EASYSIMD_FLOAT32_C(   764.43), EASYSIMD_FLOAT32_C(   747.29), EASYSIMD_FLOAT32_C(   148.00),
                         EASYSIMD_FLOAT32_C(   958.74), EASYSIMD_FLOAT32_C(  -382.93), EASYSIMD_FLOAT32_C(  -103.63), EASYSIMD_FLOAT32_C(   581.41)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   392.51), EASYSIMD_FLOAT32_C(   764.43), EASYSIMD_FLOAT32_C(   747.29), EASYSIMD_FLOAT32_C(   148.00),
                         EASYSIMD_FLOAT32_C(   958.74), EASYSIMD_FLOAT32_C(  -382.93), EASYSIMD_FLOAT32_C(  -103.63), EASYSIMD_FLOAT32_C(   581.41)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -290.91), EASYSIMD_FLOAT32_C(  -854.10), EASYSIMD_FLOAT32_C(   422.85), EASYSIMD_FLOAT32_C(   573.59),
                         EASYSIMD_FLOAT32_C(   892.23), EASYSIMD_FLOAT32_C(   602.71), EASYSIMD_FLOAT32_C(  -960.28), EASYSIMD_FLOAT32_C(  -635.56)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -428.92), EASYSIMD_FLOAT32_C(   118.34), EASYSIMD_FLOAT32_C(  -283.42), EASYSIMD_FLOAT32_C(   146.93),
                         EASYSIMD_FLOAT32_C(  -394.50), EASYSIMD_FLOAT32_C(  -509.97), EASYSIMD_FLOAT32_C(   353.55), EASYSIMD_FLOAT32_C(   857.33),
                         EASYSIMD_FLOAT32_C(  -260.12), EASYSIMD_FLOAT32_C(  -137.69), EASYSIMD_FLOAT32_C(   573.03), EASYSIMD_FLOAT32_C(  -797.79),
                         EASYSIMD_FLOAT32_C(  -833.27), EASYSIMD_FLOAT32_C(   868.45), EASYSIMD_FLOAT32_C(   286.33), EASYSIMD_FLOAT32_C(  -124.18)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -260.12), EASYSIMD_FLOAT32_C(  -137.69), EASYSIMD_FLOAT32_C(   573.03), EASYSIMD_FLOAT32_C(  -797.79),
                         EASYSIMD_FLOAT32_C(  -833.27), EASYSIMD_FLOAT32_C(   868.45), EASYSIMD_FLOAT32_C(   286.33), EASYSIMD_FLOAT32_C(  -124.18)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -428.92), EASYSIMD_FLOAT32_C(   118.34), EASYSIMD_FLOAT32_C(  -283.42), EASYSIMD_FLOAT32_C(   146.93),
                         EASYSIMD_FLOAT32_C(  -394.50), EASYSIMD_FLOAT32_C(  -509.97), EASYSIMD_FLOAT32_C(   353.55), EASYSIMD_FLOAT32_C(   857.33)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -985.65), EASYSIMD_FLOAT32_C(  -290.82), EASYSIMD_FLOAT32_C(  -550.59), EASYSIMD_FLOAT32_C(   906.58),
                         EASYSIMD_FLOAT32_C(  -688.11), EASYSIMD_FLOAT32_C(  -510.87), EASYSIMD_FLOAT32_C(  -728.98), EASYSIMD_FLOAT32_C(   704.40),
                         EASYSIMD_FLOAT32_C(  -746.44), EASYSIMD_FLOAT32_C(  -981.69), EASYSIMD_FLOAT32_C(  -147.59), EASYSIMD_FLOAT32_C(  -787.70),
                         EASYSIMD_FLOAT32_C(  -364.62), EASYSIMD_FLOAT32_C(   748.77), EASYSIMD_FLOAT32_C(   793.71), EASYSIMD_FLOAT32_C(   206.47)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -746.44), EASYSIMD_FLOAT32_C(  -981.69), EASYSIMD_FLOAT32_C(  -147.59), EASYSIMD_FLOAT32_C(  -787.70),
                         EASYSIMD_FLOAT32_C(  -364.62), EASYSIMD_FLOAT32_C(   748.77), EASYSIMD_FLOAT32_C(   793.71), EASYSIMD_FLOAT32_C(   206.47)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -985.65), EASYSIMD_FLOAT32_C(  -290.82), EASYSIMD_FLOAT32_C(  -550.59), EASYSIMD_FLOAT32_C(   906.58),
                         EASYSIMD_FLOAT32_C(  -688.11), EASYSIMD_FLOAT32_C(  -510.87), EASYSIMD_FLOAT32_C(  -728.98), EASYSIMD_FLOAT32_C(   704.40)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -132.89), EASYSIMD_FLOAT32_C(  -489.71), EASYSIMD_FLOAT32_C(  -646.61), EASYSIMD_FLOAT32_C(   472.61),
                         EASYSIMD_FLOAT32_C(     0.31), EASYSIMD_FLOAT32_C(   706.95), EASYSIMD_FLOAT32_C(   329.93), EASYSIMD_FLOAT32_C(   740.19),
                         EASYSIMD_FLOAT32_C(  -430.74), EASYSIMD_FLOAT32_C(   -97.04), EASYSIMD_FLOAT32_C(   942.40), EASYSIMD_FLOAT32_C(  -264.01),
                         EASYSIMD_FLOAT32_C(  -228.59), EASYSIMD_FLOAT32_C(   228.74), EASYSIMD_FLOAT32_C(   611.81), EASYSIMD_FLOAT32_C(  -214.24)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -430.74), EASYSIMD_FLOAT32_C(   -97.04), EASYSIMD_FLOAT32_C(   942.40), EASYSIMD_FLOAT32_C(  -264.01),
                         EASYSIMD_FLOAT32_C(  -228.59), EASYSIMD_FLOAT32_C(   228.74), EASYSIMD_FLOAT32_C(   611.81), EASYSIMD_FLOAT32_C(  -214.24)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -132.89), EASYSIMD_FLOAT32_C(  -489.71), EASYSIMD_FLOAT32_C(  -646.61), EASYSIMD_FLOAT32_C(   472.61),
                         EASYSIMD_FLOAT32_C(     0.31), EASYSIMD_FLOAT32_C(   706.95), EASYSIMD_FLOAT32_C(   329.93), EASYSIMD_FLOAT32_C(   740.19)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 r;
    r = easysimd_mm512_extractf32x8_ps(test_vec[i].a, 0);
    easysimd_assert_m256_close(r, test_vec[i].r0, 1);

    easysimd__m512 a = test_vec[i].a;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extractf32x8_ps(a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extractf32x8_ps");
    easysimd_assert_m256_close(r, test_vec[i].r1, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_extractf32x8_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[16];
    const easysimd_float32 r0[8];
    const easysimd_float32 r1[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   702.03), EASYSIMD_FLOAT32_C(   220.80), EASYSIMD_FLOAT32_C(  -883.92), EASYSIMD_FLOAT32_C(  -123.23),
        EASYSIMD_FLOAT32_C(  -988.30), EASYSIMD_FLOAT32_C(  -586.38), EASYSIMD_FLOAT32_C(   100.81), EASYSIMD_FLOAT32_C(  -827.38) },
      UINT8_C(219),
      { EASYSIMD_FLOAT32_C(   780.62), EASYSIMD_FLOAT32_C(  -415.56), EASYSIMD_FLOAT32_C(   952.38), EASYSIMD_FLOAT32_C(   567.08),
        EASYSIMD_FLOAT32_C(    36.48), EASYSIMD_FLOAT32_C(   111.26), EASYSIMD_FLOAT32_C(  -888.51), EASYSIMD_FLOAT32_C(   397.36),
        EASYSIMD_FLOAT32_C(  -830.90), EASYSIMD_FLOAT32_C(  -176.13), EASYSIMD_FLOAT32_C(  -396.13), EASYSIMD_FLOAT32_C(  -775.22),
        EASYSIMD_FLOAT32_C(   873.57), EASYSIMD_FLOAT32_C(   230.60), EASYSIMD_FLOAT32_C(  -933.77), EASYSIMD_FLOAT32_C(   574.71) },
      { EASYSIMD_FLOAT32_C(   780.62), EASYSIMD_FLOAT32_C(  -415.56), EASYSIMD_FLOAT32_C(  -883.92), EASYSIMD_FLOAT32_C(   567.08),
        EASYSIMD_FLOAT32_C(    36.48), EASYSIMD_FLOAT32_C(  -586.38), EASYSIMD_FLOAT32_C(  -888.51), EASYSIMD_FLOAT32_C(   397.36) },
      { EASYSIMD_FLOAT32_C(  -830.90), EASYSIMD_FLOAT32_C(  -176.13), EASYSIMD_FLOAT32_C(  -883.92), EASYSIMD_FLOAT32_C(  -775.22),
        EASYSIMD_FLOAT32_C(   873.57), EASYSIMD_FLOAT32_C(  -586.38), EASYSIMD_FLOAT32_C(  -933.77), EASYSIMD_FLOAT32_C(   574.71) } },
    { { EASYSIMD_FLOAT32_C(   754.66), EASYSIMD_FLOAT32_C(   660.63), EASYSIMD_FLOAT32_C(   744.06), EASYSIMD_FLOAT32_C(  -534.38),
        EASYSIMD_FLOAT32_C(  -733.97), EASYSIMD_FLOAT32_C(  -552.71), EASYSIMD_FLOAT32_C(  -832.35), EASYSIMD_FLOAT32_C(   486.84) },
      UINT8_C(198),
      { EASYSIMD_FLOAT32_C(    44.42), EASYSIMD_FLOAT32_C(   498.53), EASYSIMD_FLOAT32_C(   -23.00), EASYSIMD_FLOAT32_C(  -854.76),
        EASYSIMD_FLOAT32_C(   671.16), EASYSIMD_FLOAT32_C(   285.08), EASYSIMD_FLOAT32_C(   925.86), EASYSIMD_FLOAT32_C(  -744.41),
        EASYSIMD_FLOAT32_C(   237.46), EASYSIMD_FLOAT32_C(   492.94), EASYSIMD_FLOAT32_C(   292.08), EASYSIMD_FLOAT32_C(  -651.28),
        EASYSIMD_FLOAT32_C(   604.43), EASYSIMD_FLOAT32_C(  -310.56), EASYSIMD_FLOAT32_C(  -482.18), EASYSIMD_FLOAT32_C(  -571.70) },
      { EASYSIMD_FLOAT32_C(   754.66), EASYSIMD_FLOAT32_C(   498.53), EASYSIMD_FLOAT32_C(   -23.00), EASYSIMD_FLOAT32_C(  -534.38),
        EASYSIMD_FLOAT32_C(  -733.97), EASYSIMD_FLOAT32_C(  -552.71), EASYSIMD_FLOAT32_C(   925.86), EASYSIMD_FLOAT32_C(  -744.41) },
      { EASYSIMD_FLOAT32_C(   754.66), EASYSIMD_FLOAT32_C(   492.94), EASYSIMD_FLOAT32_C(   292.08), EASYSIMD_FLOAT32_C(  -534.38),
        EASYSIMD_FLOAT32_C(  -733.97), EASYSIMD_FLOAT32_C(  -552.71), EASYSIMD_FLOAT32_C(  -482.18), EASYSIMD_FLOAT32_C(  -571.70) } },
    { { EASYSIMD_FLOAT32_C(   293.31), EASYSIMD_FLOAT32_C(  -257.40), EASYSIMD_FLOAT32_C(  -698.13), EASYSIMD_FLOAT32_C(  -476.09),
        EASYSIMD_FLOAT32_C(  -191.17), EASYSIMD_FLOAT32_C(   876.58), EASYSIMD_FLOAT32_C(  -721.43), EASYSIMD_FLOAT32_C(  -530.53) },
      UINT8_C( 19),
      { EASYSIMD_FLOAT32_C(  -255.81), EASYSIMD_FLOAT32_C(  -264.50), EASYSIMD_FLOAT32_C(  -932.07), EASYSIMD_FLOAT32_C(   -88.16),
        EASYSIMD_FLOAT32_C(  -777.66), EASYSIMD_FLOAT32_C(  -368.69), EASYSIMD_FLOAT32_C(   956.26), EASYSIMD_FLOAT32_C(   720.87),
        EASYSIMD_FLOAT32_C(   608.31), EASYSIMD_FLOAT32_C(  -898.50), EASYSIMD_FLOAT32_C(   392.03), EASYSIMD_FLOAT32_C(  -106.62),
        EASYSIMD_FLOAT32_C(  -972.64), EASYSIMD_FLOAT32_C(   647.62), EASYSIMD_FLOAT32_C(  -869.15), EASYSIMD_FLOAT32_C(   520.31) },
      { EASYSIMD_FLOAT32_C(  -255.81), EASYSIMD_FLOAT32_C(  -264.50), EASYSIMD_FLOAT32_C(  -698.13), EASYSIMD_FLOAT32_C(  -476.09),
        EASYSIMD_FLOAT32_C(  -777.66), EASYSIMD_FLOAT32_C(   876.58), EASYSIMD_FLOAT32_C(  -721.43), EASYSIMD_FLOAT32_C(  -530.53) },
      { EASYSIMD_FLOAT32_C(   608.31), EASYSIMD_FLOAT32_C(  -898.50), EASYSIMD_FLOAT32_C(  -698.13), EASYSIMD_FLOAT32_C(  -476.09),
        EASYSIMD_FLOAT32_C(  -972.64), EASYSIMD_FLOAT32_C(   876.58), EASYSIMD_FLOAT32_C(  -721.43), EASYSIMD_FLOAT32_C(  -530.53) } },
    { { EASYSIMD_FLOAT32_C(   -60.30), EASYSIMD_FLOAT32_C(  -520.43), EASYSIMD_FLOAT32_C(   124.74), EASYSIMD_FLOAT32_C(   629.14),
        EASYSIMD_FLOAT32_C(    -2.61), EASYSIMD_FLOAT32_C(   553.04), EASYSIMD_FLOAT32_C(   -77.55), EASYSIMD_FLOAT32_C(   740.00) },
      UINT8_C(226),
      { EASYSIMD_FLOAT32_C(   446.36), EASYSIMD_FLOAT32_C(  -451.17), EASYSIMD_FLOAT32_C(   731.49), EASYSIMD_FLOAT32_C(   724.93),
        EASYSIMD_FLOAT32_C(    18.30), EASYSIMD_FLOAT32_C(   352.13), EASYSIMD_FLOAT32_C(  -530.88), EASYSIMD_FLOAT32_C(   753.80),
        EASYSIMD_FLOAT32_C(   420.06), EASYSIMD_FLOAT32_C(   380.96), EASYSIMD_FLOAT32_C(   976.14), EASYSIMD_FLOAT32_C(  -948.63),
        EASYSIMD_FLOAT32_C(   337.22), EASYSIMD_FLOAT32_C(   697.01), EASYSIMD_FLOAT32_C(   659.68), EASYSIMD_FLOAT32_C(   438.72) },
      { EASYSIMD_FLOAT32_C(   -60.30), EASYSIMD_FLOAT32_C(  -451.17), EASYSIMD_FLOAT32_C(   124.74), EASYSIMD_FLOAT32_C(   629.14),
        EASYSIMD_FLOAT32_C(    -2.61), EASYSIMD_FLOAT32_C(   352.13), EASYSIMD_FLOAT32_C(  -530.88), EASYSIMD_FLOAT32_C(   753.80) },
      { EASYSIMD_FLOAT32_C(   -60.30), EASYSIMD_FLOAT32_C(   380.96), EASYSIMD_FLOAT32_C(   124.74), EASYSIMD_FLOAT32_C(   629.14),
        EASYSIMD_FLOAT32_C(    -2.61), EASYSIMD_FLOAT32_C(   697.01), EASYSIMD_FLOAT32_C(   659.68), EASYSIMD_FLOAT32_C(   438.72) } },
    { { EASYSIMD_FLOAT32_C(    89.04), EASYSIMD_FLOAT32_C(  -446.94), EASYSIMD_FLOAT32_C(   466.08), EASYSIMD_FLOAT32_C(  -263.33),
        EASYSIMD_FLOAT32_C(  -316.09), EASYSIMD_FLOAT32_C(   -13.61), EASYSIMD_FLOAT32_C(   676.37), EASYSIMD_FLOAT32_C(   163.48) },
      UINT8_C(217),
      { EASYSIMD_FLOAT32_C(   305.50), EASYSIMD_FLOAT32_C(  -839.13), EASYSIMD_FLOAT32_C(   664.16), EASYSIMD_FLOAT32_C(  -772.05),
        EASYSIMD_FLOAT32_C(   900.87), EASYSIMD_FLOAT32_C(   519.07), EASYSIMD_FLOAT32_C(   674.31), EASYSIMD_FLOAT32_C(  -550.31),
        EASYSIMD_FLOAT32_C(   250.56), EASYSIMD_FLOAT32_C(   399.23), EASYSIMD_FLOAT32_C(   467.99), EASYSIMD_FLOAT32_C(  -397.31),
        EASYSIMD_FLOAT32_C(   868.35), EASYSIMD_FLOAT32_C(   221.79), EASYSIMD_FLOAT32_C(  -977.24), EASYSIMD_FLOAT32_C(   249.31) },
      { EASYSIMD_FLOAT32_C(   305.50), EASYSIMD_FLOAT32_C(  -446.94), EASYSIMD_FLOAT32_C(   466.08), EASYSIMD_FLOAT32_C(  -772.05),
        EASYSIMD_FLOAT32_C(   900.87), EASYSIMD_FLOAT32_C(   -13.61), EASYSIMD_FLOAT32_C(   674.31), EASYSIMD_FLOAT32_C(  -550.31) },
      { EASYSIMD_FLOAT32_C(   250.56), EASYSIMD_FLOAT32_C(  -446.94), EASYSIMD_FLOAT32_C(   466.08), EASYSIMD_FLOAT32_C(  -397.31),
        EASYSIMD_FLOAT32_C(   868.35), EASYSIMD_FLOAT32_C(   -13.61), EASYSIMD_FLOAT32_C(  -977.24), EASYSIMD_FLOAT32_C(   249.31) } },
    { { EASYSIMD_FLOAT32_C(   197.93), EASYSIMD_FLOAT32_C(  -925.87), EASYSIMD_FLOAT32_C(  -413.47), EASYSIMD_FLOAT32_C(  -105.06),
        EASYSIMD_FLOAT32_C(   733.81), EASYSIMD_FLOAT32_C(  -974.75), EASYSIMD_FLOAT32_C(   983.99), EASYSIMD_FLOAT32_C(  -713.13) },
      UINT8_C(235),
      { EASYSIMD_FLOAT32_C(  -279.35), EASYSIMD_FLOAT32_C(   -29.22), EASYSIMD_FLOAT32_C(  -522.28), EASYSIMD_FLOAT32_C(  -602.98),
        EASYSIMD_FLOAT32_C(  -865.74), EASYSIMD_FLOAT32_C(  -411.16), EASYSIMD_FLOAT32_C(   702.53), EASYSIMD_FLOAT32_C(  -704.87),
        EASYSIMD_FLOAT32_C(  -747.00), EASYSIMD_FLOAT32_C(   930.48), EASYSIMD_FLOAT32_C(  -804.00), EASYSIMD_FLOAT32_C(   772.08),
        EASYSIMD_FLOAT32_C(   604.78), EASYSIMD_FLOAT32_C(  -354.31), EASYSIMD_FLOAT32_C(    22.64), EASYSIMD_FLOAT32_C(     4.02) },
      { EASYSIMD_FLOAT32_C(  -279.35), EASYSIMD_FLOAT32_C(   -29.22), EASYSIMD_FLOAT32_C(  -413.47), EASYSIMD_FLOAT32_C(  -602.98),
        EASYSIMD_FLOAT32_C(   733.81), EASYSIMD_FLOAT32_C(  -411.16), EASYSIMD_FLOAT32_C(   702.53), EASYSIMD_FLOAT32_C(  -704.87) },
      { EASYSIMD_FLOAT32_C(  -747.00), EASYSIMD_FLOAT32_C(   930.48), EASYSIMD_FLOAT32_C(  -413.47), EASYSIMD_FLOAT32_C(   772.08),
        EASYSIMD_FLOAT32_C(   733.81), EASYSIMD_FLOAT32_C(  -354.31), EASYSIMD_FLOAT32_C(    22.64), EASYSIMD_FLOAT32_C(     4.02) } },
    { { EASYSIMD_FLOAT32_C(  -886.32), EASYSIMD_FLOAT32_C(   625.33), EASYSIMD_FLOAT32_C(  -127.63), EASYSIMD_FLOAT32_C(   335.47),
        EASYSIMD_FLOAT32_C(   648.09), EASYSIMD_FLOAT32_C(  -878.32), EASYSIMD_FLOAT32_C(  -466.60), EASYSIMD_FLOAT32_C(   722.21) },
      UINT8_C(133),
      { EASYSIMD_FLOAT32_C(   428.34), EASYSIMD_FLOAT32_C(   456.02), EASYSIMD_FLOAT32_C(  -266.55), EASYSIMD_FLOAT32_C(   412.33),
        EASYSIMD_FLOAT32_C(   742.89), EASYSIMD_FLOAT32_C(  -775.22), EASYSIMD_FLOAT32_C(  -867.02), EASYSIMD_FLOAT32_C(  -286.32),
        EASYSIMD_FLOAT32_C(  -297.50), EASYSIMD_FLOAT32_C(  -470.00), EASYSIMD_FLOAT32_C(  -152.06), EASYSIMD_FLOAT32_C(   291.34),
        EASYSIMD_FLOAT32_C(  -767.47), EASYSIMD_FLOAT32_C(   143.07), EASYSIMD_FLOAT32_C(   544.34), EASYSIMD_FLOAT32_C(  -836.99) },
      { EASYSIMD_FLOAT32_C(   428.34), EASYSIMD_FLOAT32_C(   625.33), EASYSIMD_FLOAT32_C(  -266.55), EASYSIMD_FLOAT32_C(   335.47),
        EASYSIMD_FLOAT32_C(   648.09), EASYSIMD_FLOAT32_C(  -878.32), EASYSIMD_FLOAT32_C(  -466.60), EASYSIMD_FLOAT32_C(  -286.32) },
      { EASYSIMD_FLOAT32_C(  -297.50), EASYSIMD_FLOAT32_C(   625.33), EASYSIMD_FLOAT32_C(  -152.06), EASYSIMD_FLOAT32_C(   335.47),
        EASYSIMD_FLOAT32_C(   648.09), EASYSIMD_FLOAT32_C(  -878.32), EASYSIMD_FLOAT32_C(  -466.60), EASYSIMD_FLOAT32_C(  -836.99) } },
    { { EASYSIMD_FLOAT32_C(   339.06), EASYSIMD_FLOAT32_C(   316.42), EASYSIMD_FLOAT32_C(   767.79), EASYSIMD_FLOAT32_C(   984.75),
        EASYSIMD_FLOAT32_C(  -660.94), EASYSIMD_FLOAT32_C(  -228.19), EASYSIMD_FLOAT32_C(  -901.56), EASYSIMD_FLOAT32_C(   964.39) },
      UINT8_C(240),
      { EASYSIMD_FLOAT32_C(   433.91), EASYSIMD_FLOAT32_C(   612.47), EASYSIMD_FLOAT32_C(   765.86), EASYSIMD_FLOAT32_C(   967.31),
        EASYSIMD_FLOAT32_C(   334.69), EASYSIMD_FLOAT32_C(  -525.93), EASYSIMD_FLOAT32_C(   395.65), EASYSIMD_FLOAT32_C(  -209.29),
        EASYSIMD_FLOAT32_C(   207.52), EASYSIMD_FLOAT32_C(  -192.02), EASYSIMD_FLOAT32_C(  -466.39), EASYSIMD_FLOAT32_C(   432.30),
        EASYSIMD_FLOAT32_C(   -59.04), EASYSIMD_FLOAT32_C(   247.28), EASYSIMD_FLOAT32_C(  -865.20), EASYSIMD_FLOAT32_C(   470.96) },
      { EASYSIMD_FLOAT32_C(   339.06), EASYSIMD_FLOAT32_C(   316.42), EASYSIMD_FLOAT32_C(   767.79), EASYSIMD_FLOAT32_C(   984.75),
        EASYSIMD_FLOAT32_C(   334.69), EASYSIMD_FLOAT32_C(  -525.93), EASYSIMD_FLOAT32_C(   395.65), EASYSIMD_FLOAT32_C(  -209.29) },
      { EASYSIMD_FLOAT32_C(   339.06), EASYSIMD_FLOAT32_C(   316.42), EASYSIMD_FLOAT32_C(   767.79), EASYSIMD_FLOAT32_C(   984.75),
        EASYSIMD_FLOAT32_C(   -59.04), EASYSIMD_FLOAT32_C(   247.28), EASYSIMD_FLOAT32_C(  -865.20), EASYSIMD_FLOAT32_C(   470.96) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m256 r;

    r = easysimd_mm512_mask_extractf32x8_ps(src, k, a, 0);
    easysimd_assert_m256_close(r, easysimd_mm256_loadu_ps(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extractf32x8_ps(src, k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extractf32x8_ps");

    easysimd_assert_m256_close(r, easysimd_mm256_loadu_ps(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r0 = easysimd_mm512_mask_extractf32x8_ps(src, k, a, 0);
    easysimd__m256 r1 = easysimd_mm512_mask_extractf32x8_ps(src, k, a, 1);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_extractf32x8_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const easysimd_float32 a[16];
    const easysimd_float32 r0[8];
    const easysimd_float32 r1[8];
  } test_vec[] = {
    { UINT8_C( 60),
      { EASYSIMD_FLOAT32_C(   473.92), EASYSIMD_FLOAT32_C(  -477.07), EASYSIMD_FLOAT32_C(   660.56), EASYSIMD_FLOAT32_C(  -607.79),
        EASYSIMD_FLOAT32_C(    81.21), EASYSIMD_FLOAT32_C(   871.80), EASYSIMD_FLOAT32_C(  -138.91), EASYSIMD_FLOAT32_C(   618.68),
        EASYSIMD_FLOAT32_C(   705.21), EASYSIMD_FLOAT32_C(  -761.67), EASYSIMD_FLOAT32_C(   644.15), EASYSIMD_FLOAT32_C(  -136.34),
        EASYSIMD_FLOAT32_C(   799.97), EASYSIMD_FLOAT32_C(    36.72), EASYSIMD_FLOAT32_C(   782.45), EASYSIMD_FLOAT32_C(  -958.17) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   660.56), EASYSIMD_FLOAT32_C(  -607.79),
        EASYSIMD_FLOAT32_C(    81.21), EASYSIMD_FLOAT32_C(   871.80), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   644.15), EASYSIMD_FLOAT32_C(  -136.34),
        EASYSIMD_FLOAT32_C(   799.97), EASYSIMD_FLOAT32_C(    36.72), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 92),
      { EASYSIMD_FLOAT32_C(  -247.09), EASYSIMD_FLOAT32_C(  -644.41), EASYSIMD_FLOAT32_C(  -511.75), EASYSIMD_FLOAT32_C(  -278.34),
        EASYSIMD_FLOAT32_C(  -507.18), EASYSIMD_FLOAT32_C(   928.65), EASYSIMD_FLOAT32_C(   713.17), EASYSIMD_FLOAT32_C(  -818.35),
        EASYSIMD_FLOAT32_C(   738.95), EASYSIMD_FLOAT32_C(   531.22), EASYSIMD_FLOAT32_C(   901.34), EASYSIMD_FLOAT32_C(  -589.97),
        EASYSIMD_FLOAT32_C(  -635.52), EASYSIMD_FLOAT32_C(   243.20), EASYSIMD_FLOAT32_C(   883.95), EASYSIMD_FLOAT32_C(  -112.59) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -511.75), EASYSIMD_FLOAT32_C(  -278.34),
        EASYSIMD_FLOAT32_C(  -507.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   713.17), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   901.34), EASYSIMD_FLOAT32_C(  -589.97),
        EASYSIMD_FLOAT32_C(  -635.52), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   883.95), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 24),
      { EASYSIMD_FLOAT32_C(  -723.84), EASYSIMD_FLOAT32_C(   968.61), EASYSIMD_FLOAT32_C(  -224.44), EASYSIMD_FLOAT32_C(   137.25),
        EASYSIMD_FLOAT32_C(   587.30), EASYSIMD_FLOAT32_C(  -519.23), EASYSIMD_FLOAT32_C(   375.58), EASYSIMD_FLOAT32_C(   231.45),
        EASYSIMD_FLOAT32_C(   344.44), EASYSIMD_FLOAT32_C(   175.55), EASYSIMD_FLOAT32_C(  -731.83), EASYSIMD_FLOAT32_C(   126.88),
        EASYSIMD_FLOAT32_C(   217.38), EASYSIMD_FLOAT32_C(    55.92), EASYSIMD_FLOAT32_C(   879.79), EASYSIMD_FLOAT32_C(   572.97) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   137.25),
        EASYSIMD_FLOAT32_C(   587.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   126.88),
        EASYSIMD_FLOAT32_C(   217.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 63),
      { EASYSIMD_FLOAT32_C(  -398.55), EASYSIMD_FLOAT32_C(  -934.21), EASYSIMD_FLOAT32_C(   472.82), EASYSIMD_FLOAT32_C(  -685.39),
        EASYSIMD_FLOAT32_C(  -752.56), EASYSIMD_FLOAT32_C(   211.77), EASYSIMD_FLOAT32_C(   845.84), EASYSIMD_FLOAT32_C(  -851.23),
        EASYSIMD_FLOAT32_C(   621.80), EASYSIMD_FLOAT32_C(  -789.69), EASYSIMD_FLOAT32_C(   391.97), EASYSIMD_FLOAT32_C(   505.75),
        EASYSIMD_FLOAT32_C(    97.72), EASYSIMD_FLOAT32_C(  -704.27), EASYSIMD_FLOAT32_C(   781.91), EASYSIMD_FLOAT32_C(    66.33) },
      { EASYSIMD_FLOAT32_C(  -398.55), EASYSIMD_FLOAT32_C(  -934.21), EASYSIMD_FLOAT32_C(   472.82), EASYSIMD_FLOAT32_C(  -685.39),
        EASYSIMD_FLOAT32_C(  -752.56), EASYSIMD_FLOAT32_C(   211.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(   621.80), EASYSIMD_FLOAT32_C(  -789.69), EASYSIMD_FLOAT32_C(   391.97), EASYSIMD_FLOAT32_C(   505.75),
        EASYSIMD_FLOAT32_C(    97.72), EASYSIMD_FLOAT32_C(  -704.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 46),
      { EASYSIMD_FLOAT32_C(   -80.85), EASYSIMD_FLOAT32_C(  -346.37), EASYSIMD_FLOAT32_C(   552.06), EASYSIMD_FLOAT32_C(  -705.27),
        EASYSIMD_FLOAT32_C(   885.07), EASYSIMD_FLOAT32_C(  -103.50), EASYSIMD_FLOAT32_C(   470.28), EASYSIMD_FLOAT32_C(  -846.76),
        EASYSIMD_FLOAT32_C(  -976.62), EASYSIMD_FLOAT32_C(  -312.34), EASYSIMD_FLOAT32_C(   209.16), EASYSIMD_FLOAT32_C(   903.17),
        EASYSIMD_FLOAT32_C(  -739.37), EASYSIMD_FLOAT32_C(  -246.66), EASYSIMD_FLOAT32_C(  -495.39), EASYSIMD_FLOAT32_C(  -673.59) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -346.37), EASYSIMD_FLOAT32_C(   552.06), EASYSIMD_FLOAT32_C(  -705.27),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -103.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -312.34), EASYSIMD_FLOAT32_C(   209.16), EASYSIMD_FLOAT32_C(   903.17),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -246.66), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 63),
      { EASYSIMD_FLOAT32_C(  -180.77), EASYSIMD_FLOAT32_C(  -426.15), EASYSIMD_FLOAT32_C(   437.93), EASYSIMD_FLOAT32_C(  -334.93),
        EASYSIMD_FLOAT32_C(  -277.38), EASYSIMD_FLOAT32_C(    59.73), EASYSIMD_FLOAT32_C(  -124.62), EASYSIMD_FLOAT32_C(  -885.41),
        EASYSIMD_FLOAT32_C(  -434.52), EASYSIMD_FLOAT32_C(   973.10), EASYSIMD_FLOAT32_C(  -589.68), EASYSIMD_FLOAT32_C(  -652.62),
        EASYSIMD_FLOAT32_C(    39.43), EASYSIMD_FLOAT32_C(   481.61), EASYSIMD_FLOAT32_C(   266.54), EASYSIMD_FLOAT32_C(   693.06) },
      { EASYSIMD_FLOAT32_C(  -180.77), EASYSIMD_FLOAT32_C(  -426.15), EASYSIMD_FLOAT32_C(   437.93), EASYSIMD_FLOAT32_C(  -334.93),
        EASYSIMD_FLOAT32_C(  -277.38), EASYSIMD_FLOAT32_C(    59.73), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -434.52), EASYSIMD_FLOAT32_C(   973.10), EASYSIMD_FLOAT32_C(  -589.68), EASYSIMD_FLOAT32_C(  -652.62),
        EASYSIMD_FLOAT32_C(    39.43), EASYSIMD_FLOAT32_C(   481.61), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(153),
      { EASYSIMD_FLOAT32_C(   561.27), EASYSIMD_FLOAT32_C(   578.14), EASYSIMD_FLOAT32_C(   930.17), EASYSIMD_FLOAT32_C(    31.55),
        EASYSIMD_FLOAT32_C(   731.38), EASYSIMD_FLOAT32_C(   953.55), EASYSIMD_FLOAT32_C(   719.21), EASYSIMD_FLOAT32_C(   -59.46),
        EASYSIMD_FLOAT32_C(   856.72), EASYSIMD_FLOAT32_C(   979.84), EASYSIMD_FLOAT32_C(   693.88), EASYSIMD_FLOAT32_C(  -638.67),
        EASYSIMD_FLOAT32_C(  -693.75), EASYSIMD_FLOAT32_C(   920.03), EASYSIMD_FLOAT32_C(   180.56), EASYSIMD_FLOAT32_C(  -119.91) },
      { EASYSIMD_FLOAT32_C(   561.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    31.55),
        EASYSIMD_FLOAT32_C(   731.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -59.46) },
      { EASYSIMD_FLOAT32_C(   856.72), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -638.67),
        EASYSIMD_FLOAT32_C(  -693.75), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -119.91) } },
    { UINT8_C( 28),
      { EASYSIMD_FLOAT32_C(   845.63), EASYSIMD_FLOAT32_C(   602.71), EASYSIMD_FLOAT32_C(  -582.32), EASYSIMD_FLOAT32_C(  -278.99),
        EASYSIMD_FLOAT32_C(   717.30), EASYSIMD_FLOAT32_C(   -16.84), EASYSIMD_FLOAT32_C(  -305.89), EASYSIMD_FLOAT32_C(  -872.38),
        EASYSIMD_FLOAT32_C(   330.55), EASYSIMD_FLOAT32_C(   733.55), EASYSIMD_FLOAT32_C(   609.23), EASYSIMD_FLOAT32_C(  -402.92),
        EASYSIMD_FLOAT32_C(   426.61), EASYSIMD_FLOAT32_C(  -357.10), EASYSIMD_FLOAT32_C(  -841.65), EASYSIMD_FLOAT32_C(     4.74) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -582.32), EASYSIMD_FLOAT32_C(  -278.99),
        EASYSIMD_FLOAT32_C(   717.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   609.23), EASYSIMD_FLOAT32_C(  -402.92),
        EASYSIMD_FLOAT32_C(   426.61), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m256 r;

    r = easysimd_mm512_maskz_extractf32x8_ps(k, a, 0);
    easysimd_assert_m256_close(r, easysimd_mm256_loadu_ps(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extractf32x8_ps(k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extractf32x8_ps");

    easysimd_assert_m256_close(r, easysimd_mm256_loadu_ps(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r0 = easysimd_mm512_maskz_extractf32x8_ps(k, a, 0);
    easysimd__m256 r1 = easysimd_mm512_maskz_extractf32x8_ps(k, a, 1);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_extractf64x2_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float64 a[8];
    easysimd_float64 r0[2];
    easysimd_float64 r1[2];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT64_C(   146.38), EASYSIMD_FLOAT64_C(  -167.62), EASYSIMD_FLOAT64_C(   956.22), EASYSIMD_FLOAT64_C(  -578.59),
        EASYSIMD_FLOAT64_C(   185.48), EASYSIMD_FLOAT64_C(   825.76), EASYSIMD_FLOAT64_C(  -954.76), EASYSIMD_FLOAT64_C(   509.28) },
      { EASYSIMD_FLOAT64_C(   146.38), EASYSIMD_FLOAT64_C(  -167.62) },
      { EASYSIMD_FLOAT64_C(   956.22), EASYSIMD_FLOAT64_C(  -578.59) } },
    { { EASYSIMD_FLOAT64_C(   863.19), EASYSIMD_FLOAT64_C(   454.21), EASYSIMD_FLOAT64_C(   489.78), EASYSIMD_FLOAT64_C(   385.30),
        EASYSIMD_FLOAT64_C(   804.65), EASYSIMD_FLOAT64_C(   147.69), EASYSIMD_FLOAT64_C(   566.11), EASYSIMD_FLOAT64_C(  -605.54) },
      { EASYSIMD_FLOAT64_C(   863.19), EASYSIMD_FLOAT64_C(   454.21) },
      { EASYSIMD_FLOAT64_C(   489.78), EASYSIMD_FLOAT64_C(   385.30) } },
    { { EASYSIMD_FLOAT64_C(   407.74), EASYSIMD_FLOAT64_C(  -507.67), EASYSIMD_FLOAT64_C(  -955.21), EASYSIMD_FLOAT64_C(  -169.04),
        EASYSIMD_FLOAT64_C(    19.51), EASYSIMD_FLOAT64_C(  -626.80), EASYSIMD_FLOAT64_C(   167.26), EASYSIMD_FLOAT64_C(  -272.71) },
      { EASYSIMD_FLOAT64_C(   407.74), EASYSIMD_FLOAT64_C(  -507.67) },
      { EASYSIMD_FLOAT64_C(  -955.21), EASYSIMD_FLOAT64_C(  -169.04) } },
    { { EASYSIMD_FLOAT64_C(  -261.60), EASYSIMD_FLOAT64_C(  -478.68), EASYSIMD_FLOAT64_C(   248.90), EASYSIMD_FLOAT64_C(   783.13),
        EASYSIMD_FLOAT64_C(  -144.56), EASYSIMD_FLOAT64_C(  -980.83), EASYSIMD_FLOAT64_C(  -696.09), EASYSIMD_FLOAT64_C(  -998.18) },
      { EASYSIMD_FLOAT64_C(  -261.60), EASYSIMD_FLOAT64_C(  -478.68) },
      { EASYSIMD_FLOAT64_C(   248.90), EASYSIMD_FLOAT64_C(   783.13) } },
    { { EASYSIMD_FLOAT64_C(  -148.45), EASYSIMD_FLOAT64_C(  -739.86), EASYSIMD_FLOAT64_C(  -576.77), EASYSIMD_FLOAT64_C(  -962.97),
        EASYSIMD_FLOAT64_C(  -914.11), EASYSIMD_FLOAT64_C(  -531.53), EASYSIMD_FLOAT64_C(   546.31), EASYSIMD_FLOAT64_C(   949.08) },
      { EASYSIMD_FLOAT64_C(  -148.45), EASYSIMD_FLOAT64_C(  -739.86) },
      { EASYSIMD_FLOAT64_C(  -576.77), EASYSIMD_FLOAT64_C(  -962.97) } },
    { { EASYSIMD_FLOAT64_C(   922.68), EASYSIMD_FLOAT64_C(    36.09), EASYSIMD_FLOAT64_C(   334.38), EASYSIMD_FLOAT64_C(   727.33),
        EASYSIMD_FLOAT64_C(  -816.22), EASYSIMD_FLOAT64_C(   -99.51), EASYSIMD_FLOAT64_C(  -878.20), EASYSIMD_FLOAT64_C(   591.52) },
      { EASYSIMD_FLOAT64_C(   922.68), EASYSIMD_FLOAT64_C(    36.09) },
      { EASYSIMD_FLOAT64_C(   334.38), EASYSIMD_FLOAT64_C(   727.33) } },
    { { EASYSIMD_FLOAT64_C(   392.82), EASYSIMD_FLOAT64_C(  -833.42), EASYSIMD_FLOAT64_C(  -577.52), EASYSIMD_FLOAT64_C(  -587.68),
        EASYSIMD_FLOAT64_C(  -460.22), EASYSIMD_FLOAT64_C(   589.74), EASYSIMD_FLOAT64_C(   139.62), EASYSIMD_FLOAT64_C(   278.18) },
      { EASYSIMD_FLOAT64_C(   392.82), EASYSIMD_FLOAT64_C(  -833.42) },
      { EASYSIMD_FLOAT64_C(  -577.52), EASYSIMD_FLOAT64_C(  -587.68) } },
    { { EASYSIMD_FLOAT64_C(  -888.94), EASYSIMD_FLOAT64_C(  -611.48), EASYSIMD_FLOAT64_C(    61.30), EASYSIMD_FLOAT64_C(   -33.50),
        EASYSIMD_FLOAT64_C(  -592.31), EASYSIMD_FLOAT64_C(   365.22), EASYSIMD_FLOAT64_C(   -31.68), EASYSIMD_FLOAT64_C(   259.23) },
      { EASYSIMD_FLOAT64_C(  -888.94), EASYSIMD_FLOAT64_C(  -611.48) },
      { EASYSIMD_FLOAT64_C(    61.30), EASYSIMD_FLOAT64_C(   -33.50) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    r = easysimd_mm512_extractf64x2_pd(a, 0);
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extractf64x2_pd(a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extractf64x2_pd");
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128d r0 = easysimd_mm512_extractf64x2_pd(a, 0);
    easysimd__m128d r1 = easysimd_mm512_extractf64x2_pd(a, 1);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_extractf64x2_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float64 src[2];
    uint8_t k;
    easysimd_float64 a[8];
    easysimd_float64 r0[2];
    easysimd_float64 r1[2];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT64_C(  -745.23), EASYSIMD_FLOAT64_C(  -601.99) },
      UINT8_C(131),
      { EASYSIMD_FLOAT64_C(   815.66), EASYSIMD_FLOAT64_C(  -898.04), EASYSIMD_FLOAT64_C(  -774.93), EASYSIMD_FLOAT64_C(   377.29),
        EASYSIMD_FLOAT64_C(   905.69), EASYSIMD_FLOAT64_C(   650.33), EASYSIMD_FLOAT64_C(  -548.55), EASYSIMD_FLOAT64_C(  -847.84) },
      { EASYSIMD_FLOAT64_C(   815.66), EASYSIMD_FLOAT64_C(  -898.04) },
      { EASYSIMD_FLOAT64_C(  -774.93), EASYSIMD_FLOAT64_C(   377.29) } },
    { { EASYSIMD_FLOAT64_C(   789.25), EASYSIMD_FLOAT64_C(    78.02) },
      UINT8_C(223),
      { EASYSIMD_FLOAT64_C(  -121.17), EASYSIMD_FLOAT64_C(  -777.94), EASYSIMD_FLOAT64_C(   929.64), EASYSIMD_FLOAT64_C(   644.28),
        EASYSIMD_FLOAT64_C(   444.57), EASYSIMD_FLOAT64_C(    64.53), EASYSIMD_FLOAT64_C(  -794.34), EASYSIMD_FLOAT64_C(   914.59) },
      { EASYSIMD_FLOAT64_C(  -121.17), EASYSIMD_FLOAT64_C(  -777.94) },
      { EASYSIMD_FLOAT64_C(   929.64), EASYSIMD_FLOAT64_C(   644.28) } },
    { { EASYSIMD_FLOAT64_C(  -367.06), EASYSIMD_FLOAT64_C(   729.68) },
      UINT8_C( 11),
      { EASYSIMD_FLOAT64_C(  -408.67), EASYSIMD_FLOAT64_C(   367.40), EASYSIMD_FLOAT64_C(  -701.43), EASYSIMD_FLOAT64_C(  -246.89),
        EASYSIMD_FLOAT64_C(   598.67), EASYSIMD_FLOAT64_C(  -128.79), EASYSIMD_FLOAT64_C(     7.88), EASYSIMD_FLOAT64_C(   996.68) },
      { EASYSIMD_FLOAT64_C(  -408.67), EASYSIMD_FLOAT64_C(   367.40) },
      { EASYSIMD_FLOAT64_C(  -701.43), EASYSIMD_FLOAT64_C(  -246.89) } },
    { { EASYSIMD_FLOAT64_C(  -413.79), EASYSIMD_FLOAT64_C(  -176.46) },
      UINT8_C( 40),
      { EASYSIMD_FLOAT64_C(  -188.72), EASYSIMD_FLOAT64_C(  -799.17), EASYSIMD_FLOAT64_C(  -995.67), EASYSIMD_FLOAT64_C(  -538.40),
        EASYSIMD_FLOAT64_C(  -347.72), EASYSIMD_FLOAT64_C(  -843.51), EASYSIMD_FLOAT64_C(  -749.15), EASYSIMD_FLOAT64_C(   730.30) },
      { EASYSIMD_FLOAT64_C(  -413.79), EASYSIMD_FLOAT64_C(  -176.46) },
      { EASYSIMD_FLOAT64_C(  -413.79), EASYSIMD_FLOAT64_C(  -176.46) } },
    { { EASYSIMD_FLOAT64_C(   771.72), EASYSIMD_FLOAT64_C(   129.68) },
      UINT8_C( 27),
      { EASYSIMD_FLOAT64_C(   701.36), EASYSIMD_FLOAT64_C(  -226.04), EASYSIMD_FLOAT64_C(   396.94), EASYSIMD_FLOAT64_C(  -234.11),
        EASYSIMD_FLOAT64_C(   -20.38), EASYSIMD_FLOAT64_C(   311.53), EASYSIMD_FLOAT64_C(   398.83), EASYSIMD_FLOAT64_C(  -290.70) },
      { EASYSIMD_FLOAT64_C(   701.36), EASYSIMD_FLOAT64_C(  -226.04) },
      { EASYSIMD_FLOAT64_C(   396.94), EASYSIMD_FLOAT64_C(  -234.11) } },
    { { EASYSIMD_FLOAT64_C(  -673.59), EASYSIMD_FLOAT64_C(   990.16) },
      UINT8_C(105),
      { EASYSIMD_FLOAT64_C(  -375.02), EASYSIMD_FLOAT64_C(  -256.74), EASYSIMD_FLOAT64_C(   675.37), EASYSIMD_FLOAT64_C(   496.20),
        EASYSIMD_FLOAT64_C(   751.14), EASYSIMD_FLOAT64_C(   672.05), EASYSIMD_FLOAT64_C(  -917.59), EASYSIMD_FLOAT64_C(  -425.32) },
      { EASYSIMD_FLOAT64_C(  -375.02), EASYSIMD_FLOAT64_C(   990.16) },
      { EASYSIMD_FLOAT64_C(   675.37), EASYSIMD_FLOAT64_C(   990.16) } },
    { { EASYSIMD_FLOAT64_C(   770.69), EASYSIMD_FLOAT64_C(  -106.32) },
      UINT8_C(188),
      { EASYSIMD_FLOAT64_C(   775.02), EASYSIMD_FLOAT64_C(   355.29), EASYSIMD_FLOAT64_C(   427.80), EASYSIMD_FLOAT64_C(   931.51),
        EASYSIMD_FLOAT64_C(   606.14), EASYSIMD_FLOAT64_C(   158.10), EASYSIMD_FLOAT64_C(   703.23), EASYSIMD_FLOAT64_C(  -264.18) },
      { EASYSIMD_FLOAT64_C(   770.69), EASYSIMD_FLOAT64_C(  -106.32) },
      { EASYSIMD_FLOAT64_C(   770.69), EASYSIMD_FLOAT64_C(  -106.32) } },
    { { EASYSIMD_FLOAT64_C(   110.47), EASYSIMD_FLOAT64_C(   404.60) },
      UINT8_C(253),
      { EASYSIMD_FLOAT64_C(  -492.59), EASYSIMD_FLOAT64_C(  -829.51), EASYSIMD_FLOAT64_C(  -510.59), EASYSIMD_FLOAT64_C(   818.93),
        EASYSIMD_FLOAT64_C(   569.32), EASYSIMD_FLOAT64_C(   198.71), EASYSIMD_FLOAT64_C(  -854.65), EASYSIMD_FLOAT64_C(   559.48) },
      { EASYSIMD_FLOAT64_C(  -492.59), EASYSIMD_FLOAT64_C(   404.60) },
      { EASYSIMD_FLOAT64_C(  -510.59), EASYSIMD_FLOAT64_C(   404.60) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    r = easysimd_mm512_mask_extractf64x2_pd(src, k, a, 0);
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extractf64x2_pd(src, k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extractf64x2_pd");
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128d r0 = easysimd_mm512_mask_extractf64x2_pd(src, k, a, 0);
    easysimd__m128d r1 = easysimd_mm512_mask_extractf64x2_pd(src, k, a, 1);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_extractf64x2_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    easysimd_float64 a[8];
    easysimd_float64 r0[2];
    easysimd_float64 r1[2];
  } test_vec[8] = {
    { UINT8_C(178),
      { EASYSIMD_FLOAT64_C(  -229.67), EASYSIMD_FLOAT64_C(  -697.26), EASYSIMD_FLOAT64_C(   -49.22), EASYSIMD_FLOAT64_C(  -733.47),
        EASYSIMD_FLOAT64_C(  -946.11), EASYSIMD_FLOAT64_C(  -377.17), EASYSIMD_FLOAT64_C(  -651.07), EASYSIMD_FLOAT64_C(  -371.43) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -697.26) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -733.47) } },
    { UINT8_C(109),
      { EASYSIMD_FLOAT64_C(   242.62), EASYSIMD_FLOAT64_C(   404.09), EASYSIMD_FLOAT64_C(  -831.46), EASYSIMD_FLOAT64_C(  -402.10),
        EASYSIMD_FLOAT64_C(  -168.12), EASYSIMD_FLOAT64_C(  -899.95), EASYSIMD_FLOAT64_C(  -795.96), EASYSIMD_FLOAT64_C(   989.99) },
      { EASYSIMD_FLOAT64_C(   242.62), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(  -831.46), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(136),
      { EASYSIMD_FLOAT64_C(   -60.13), EASYSIMD_FLOAT64_C(   100.46), EASYSIMD_FLOAT64_C(   207.88), EASYSIMD_FLOAT64_C(  -550.35),
        EASYSIMD_FLOAT64_C(   607.86), EASYSIMD_FLOAT64_C(   378.37), EASYSIMD_FLOAT64_C(   -60.94), EASYSIMD_FLOAT64_C(   426.80) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(136),
      { EASYSIMD_FLOAT64_C(  -862.23), EASYSIMD_FLOAT64_C(   572.14), EASYSIMD_FLOAT64_C(  -492.84), EASYSIMD_FLOAT64_C(   413.18),
        EASYSIMD_FLOAT64_C(  -657.53), EASYSIMD_FLOAT64_C(  -190.10), EASYSIMD_FLOAT64_C(  -636.03), EASYSIMD_FLOAT64_C(  -391.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(  7),
      { EASYSIMD_FLOAT64_C(   -13.20), EASYSIMD_FLOAT64_C(   -42.07), EASYSIMD_FLOAT64_C(   492.36), EASYSIMD_FLOAT64_C(   380.32),
        EASYSIMD_FLOAT64_C(  -799.45), EASYSIMD_FLOAT64_C(  -103.56), EASYSIMD_FLOAT64_C(   548.86), EASYSIMD_FLOAT64_C(  -201.55) },
      { EASYSIMD_FLOAT64_C(   -13.20), EASYSIMD_FLOAT64_C(   -42.07) },
      { EASYSIMD_FLOAT64_C(   492.36), EASYSIMD_FLOAT64_C(   380.32) } },
    { UINT8_C(164),
      { EASYSIMD_FLOAT64_C(   648.91), EASYSIMD_FLOAT64_C(     2.50), EASYSIMD_FLOAT64_C(   718.32), EASYSIMD_FLOAT64_C(   452.19),
        EASYSIMD_FLOAT64_C(   942.36), EASYSIMD_FLOAT64_C(  -181.23), EASYSIMD_FLOAT64_C(  -339.93), EASYSIMD_FLOAT64_C(  -607.99) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(206),
      { EASYSIMD_FLOAT64_C(  -961.56), EASYSIMD_FLOAT64_C(   331.07), EASYSIMD_FLOAT64_C(   853.43), EASYSIMD_FLOAT64_C(   -13.87),
        EASYSIMD_FLOAT64_C(   468.85), EASYSIMD_FLOAT64_C(   425.57), EASYSIMD_FLOAT64_C(   493.29), EASYSIMD_FLOAT64_C(  -117.97) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   331.07) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   -13.87) } },
    { UINT8_C( 22),
      { EASYSIMD_FLOAT64_C(  -696.81), EASYSIMD_FLOAT64_C(   245.99), EASYSIMD_FLOAT64_C(  -622.96), EASYSIMD_FLOAT64_C(   166.98),
        EASYSIMD_FLOAT64_C(  -767.21), EASYSIMD_FLOAT64_C(   334.98), EASYSIMD_FLOAT64_C(  -340.66), EASYSIMD_FLOAT64_C(   613.11) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   245.99) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   166.98) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    r = easysimd_mm512_maskz_extractf64x2_pd(k, a, 0);
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r0), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extractf64x2_pd(k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extractf64x2_pd");
    easysimd_assert_m128d_close(r, easysimd_mm_loadu_pd(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128d r0 = easysimd_mm512_maskz_extractf64x2_pd(k, a, 0);
    easysimd__m128d r1 = easysimd_mm512_maskz_extractf64x2_pd(k, a, 1);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_extractf64x4_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d a;
    easysimd__m256d r0;
    easysimd__m256d r1;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -431.26), EASYSIMD_FLOAT64_C(  481.25),
                         EASYSIMD_FLOAT64_C(  -57.75), EASYSIMD_FLOAT64_C( -784.26),
                         EASYSIMD_FLOAT64_C(  438.04), EASYSIMD_FLOAT64_C(  549.03),
                         EASYSIMD_FLOAT64_C(  729.46), EASYSIMD_FLOAT64_C(  582.53)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  438.04), EASYSIMD_FLOAT64_C(  549.03),
                         EASYSIMD_FLOAT64_C(  729.46), EASYSIMD_FLOAT64_C(  582.53)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -431.26), EASYSIMD_FLOAT64_C(  481.25),
                         EASYSIMD_FLOAT64_C(  -57.75), EASYSIMD_FLOAT64_C( -784.26)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -563.83), EASYSIMD_FLOAT64_C(  799.30),
                         EASYSIMD_FLOAT64_C(  938.85), EASYSIMD_FLOAT64_C( -576.01),
                         EASYSIMD_FLOAT64_C( -465.05), EASYSIMD_FLOAT64_C(  439.15),
                         EASYSIMD_FLOAT64_C( -104.57), EASYSIMD_FLOAT64_C(  -28.15)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -465.05), EASYSIMD_FLOAT64_C(  439.15),
                         EASYSIMD_FLOAT64_C( -104.57), EASYSIMD_FLOAT64_C(  -28.15)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -563.83), EASYSIMD_FLOAT64_C(  799.30),
                         EASYSIMD_FLOAT64_C(  938.85), EASYSIMD_FLOAT64_C( -576.01)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  355.11), EASYSIMD_FLOAT64_C( -787.72),
                         EASYSIMD_FLOAT64_C(  472.82), EASYSIMD_FLOAT64_C( -703.51),
                         EASYSIMD_FLOAT64_C( -202.49), EASYSIMD_FLOAT64_C( -470.36),
                         EASYSIMD_FLOAT64_C(  966.37), EASYSIMD_FLOAT64_C(  135.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -202.49), EASYSIMD_FLOAT64_C( -470.36),
                         EASYSIMD_FLOAT64_C(  966.37), EASYSIMD_FLOAT64_C(  135.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  355.11), EASYSIMD_FLOAT64_C( -787.72),
                         EASYSIMD_FLOAT64_C(  472.82), EASYSIMD_FLOAT64_C( -703.51)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  120.10), EASYSIMD_FLOAT64_C(  -64.06),
                         EASYSIMD_FLOAT64_C( -620.03), EASYSIMD_FLOAT64_C(  559.81),
                         EASYSIMD_FLOAT64_C(  185.23), EASYSIMD_FLOAT64_C( -423.61),
                         EASYSIMD_FLOAT64_C(  -11.91), EASYSIMD_FLOAT64_C(  407.56)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  185.23), EASYSIMD_FLOAT64_C( -423.61),
                         EASYSIMD_FLOAT64_C(  -11.91), EASYSIMD_FLOAT64_C(  407.56)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  120.10), EASYSIMD_FLOAT64_C(  -64.06),
                         EASYSIMD_FLOAT64_C( -620.03), EASYSIMD_FLOAT64_C(  559.81)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -859.51), EASYSIMD_FLOAT64_C(  -69.45),
                         EASYSIMD_FLOAT64_C(   40.36), EASYSIMD_FLOAT64_C(   95.61),
                         EASYSIMD_FLOAT64_C( -743.10), EASYSIMD_FLOAT64_C( -688.01),
                         EASYSIMD_FLOAT64_C(  442.76), EASYSIMD_FLOAT64_C(  931.17)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -743.10), EASYSIMD_FLOAT64_C( -688.01),
                         EASYSIMD_FLOAT64_C(  442.76), EASYSIMD_FLOAT64_C(  931.17)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -859.51), EASYSIMD_FLOAT64_C(  -69.45),
                         EASYSIMD_FLOAT64_C(   40.36), EASYSIMD_FLOAT64_C(   95.61)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  193.01), EASYSIMD_FLOAT64_C( -435.27),
                         EASYSIMD_FLOAT64_C(  -84.06), EASYSIMD_FLOAT64_C(  298.40),
                         EASYSIMD_FLOAT64_C(  208.07), EASYSIMD_FLOAT64_C(  -94.60),
                         EASYSIMD_FLOAT64_C(  834.28), EASYSIMD_FLOAT64_C(  260.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  208.07), EASYSIMD_FLOAT64_C(  -94.60),
                         EASYSIMD_FLOAT64_C(  834.28), EASYSIMD_FLOAT64_C(  260.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  193.01), EASYSIMD_FLOAT64_C( -435.27),
                         EASYSIMD_FLOAT64_C(  -84.06), EASYSIMD_FLOAT64_C(  298.40)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -98.08), EASYSIMD_FLOAT64_C(  464.19),
                         EASYSIMD_FLOAT64_C(  711.12), EASYSIMD_FLOAT64_C(  282.83),
                         EASYSIMD_FLOAT64_C( -774.08), EASYSIMD_FLOAT64_C(  841.24),
                         EASYSIMD_FLOAT64_C( -414.07), EASYSIMD_FLOAT64_C(   79.76)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -774.08), EASYSIMD_FLOAT64_C(  841.24),
                         EASYSIMD_FLOAT64_C( -414.07), EASYSIMD_FLOAT64_C(   79.76)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -98.08), EASYSIMD_FLOAT64_C(  464.19),
                         EASYSIMD_FLOAT64_C(  711.12), EASYSIMD_FLOAT64_C(  282.83)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  664.52), EASYSIMD_FLOAT64_C( -224.13),
                         EASYSIMD_FLOAT64_C(  633.65), EASYSIMD_FLOAT64_C( -834.15),
                         EASYSIMD_FLOAT64_C( -157.33), EASYSIMD_FLOAT64_C( -819.46),
                         EASYSIMD_FLOAT64_C(  541.44), EASYSIMD_FLOAT64_C(  112.81)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -157.33), EASYSIMD_FLOAT64_C( -819.46),
                         EASYSIMD_FLOAT64_C(  541.44), EASYSIMD_FLOAT64_C(  112.81)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  664.52), EASYSIMD_FLOAT64_C( -224.13),
                         EASYSIMD_FLOAT64_C(  633.65), EASYSIMD_FLOAT64_C( -834.15)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d r;
    r = easysimd_mm512_extractf64x4_pd(test_vec[i].a, 0);
    easysimd_assert_m256d_close(r, test_vec[i].r0, 1);

    easysimd__m512d a = test_vec[i].a;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extractf64x4_pd(a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extractf64x4_pd");
    easysimd_assert_m256d_close(r, test_vec[i].r1, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_extractf64x4_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d src;
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m256d r0;
    easysimd__m256d r1;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -172.36), EASYSIMD_FLOAT64_C(  393.53),
                         EASYSIMD_FLOAT64_C(   36.69), EASYSIMD_FLOAT64_C( -135.52)),
      UINT8_C( 63),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  522.06), EASYSIMD_FLOAT64_C( -932.28),
                         EASYSIMD_FLOAT64_C(  600.12), EASYSIMD_FLOAT64_C( -491.12),
                         EASYSIMD_FLOAT64_C( -139.11), EASYSIMD_FLOAT64_C( -268.86),
                         EASYSIMD_FLOAT64_C(  -71.72), EASYSIMD_FLOAT64_C(   98.47)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -139.11), EASYSIMD_FLOAT64_C( -268.86),
                         EASYSIMD_FLOAT64_C(  -71.72), EASYSIMD_FLOAT64_C(   98.47)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  522.06), EASYSIMD_FLOAT64_C( -932.28),
                         EASYSIMD_FLOAT64_C(  600.12), EASYSIMD_FLOAT64_C( -491.12)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -514.56), EASYSIMD_FLOAT64_C( -860.98),
                         EASYSIMD_FLOAT64_C( -280.30), EASYSIMD_FLOAT64_C(  128.51)),
      UINT8_C(157),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  483.08), EASYSIMD_FLOAT64_C(  232.04),
                         EASYSIMD_FLOAT64_C(  774.81), EASYSIMD_FLOAT64_C( -599.01),
                         EASYSIMD_FLOAT64_C(   69.04), EASYSIMD_FLOAT64_C( -149.02),
                         EASYSIMD_FLOAT64_C(  240.79), EASYSIMD_FLOAT64_C( -839.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   69.04), EASYSIMD_FLOAT64_C( -149.02),
                         EASYSIMD_FLOAT64_C( -280.30), EASYSIMD_FLOAT64_C( -839.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  483.08), EASYSIMD_FLOAT64_C(  232.04),
                         EASYSIMD_FLOAT64_C( -280.30), EASYSIMD_FLOAT64_C( -599.01)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   51.33), EASYSIMD_FLOAT64_C( -868.59),
                         EASYSIMD_FLOAT64_C( -921.00), EASYSIMD_FLOAT64_C( -471.60)),
      UINT8_C( 33),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -835.53), EASYSIMD_FLOAT64_C(  571.79),
                         EASYSIMD_FLOAT64_C(  675.92), EASYSIMD_FLOAT64_C(  490.41),
                         EASYSIMD_FLOAT64_C(   47.59), EASYSIMD_FLOAT64_C( -895.71),
                         EASYSIMD_FLOAT64_C( -736.92), EASYSIMD_FLOAT64_C(  283.06)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   51.33), EASYSIMD_FLOAT64_C( -868.59),
                         EASYSIMD_FLOAT64_C( -921.00), EASYSIMD_FLOAT64_C(  283.06)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   51.33), EASYSIMD_FLOAT64_C( -868.59),
                         EASYSIMD_FLOAT64_C( -921.00), EASYSIMD_FLOAT64_C(  490.41)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -489.88), EASYSIMD_FLOAT64_C(  -37.49),
                         EASYSIMD_FLOAT64_C( -373.66), EASYSIMD_FLOAT64_C( -292.35)),
      UINT8_C(176),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -629.37), EASYSIMD_FLOAT64_C(  337.35),
                         EASYSIMD_FLOAT64_C( -278.32), EASYSIMD_FLOAT64_C( -744.41),
                         EASYSIMD_FLOAT64_C(   39.32), EASYSIMD_FLOAT64_C(   29.68),
                         EASYSIMD_FLOAT64_C( -490.28), EASYSIMD_FLOAT64_C(  841.53)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -489.88), EASYSIMD_FLOAT64_C(  -37.49),
                         EASYSIMD_FLOAT64_C( -373.66), EASYSIMD_FLOAT64_C( -292.35)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -489.88), EASYSIMD_FLOAT64_C(  -37.49),
                         EASYSIMD_FLOAT64_C( -373.66), EASYSIMD_FLOAT64_C( -292.35)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -577.36), EASYSIMD_FLOAT64_C(  957.37),
                         EASYSIMD_FLOAT64_C( -934.92), EASYSIMD_FLOAT64_C( -657.02)),
      UINT8_C(169),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -588.28), EASYSIMD_FLOAT64_C(  586.29),
                         EASYSIMD_FLOAT64_C( -760.88), EASYSIMD_FLOAT64_C( -617.12),
                         EASYSIMD_FLOAT64_C( -751.58), EASYSIMD_FLOAT64_C(  907.23),
                         EASYSIMD_FLOAT64_C( -359.60), EASYSIMD_FLOAT64_C( -213.75)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -751.58), EASYSIMD_FLOAT64_C(  957.37),
                         EASYSIMD_FLOAT64_C( -934.92), EASYSIMD_FLOAT64_C( -213.75)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -588.28), EASYSIMD_FLOAT64_C(  957.37),
                         EASYSIMD_FLOAT64_C( -934.92), EASYSIMD_FLOAT64_C( -617.12)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  699.07), EASYSIMD_FLOAT64_C( -143.73),
                         EASYSIMD_FLOAT64_C(  709.96), EASYSIMD_FLOAT64_C( -767.34)),
      UINT8_C( 52),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -432.13), EASYSIMD_FLOAT64_C( -842.15),
                         EASYSIMD_FLOAT64_C(  908.84), EASYSIMD_FLOAT64_C( -315.78),
                         EASYSIMD_FLOAT64_C(  -86.06), EASYSIMD_FLOAT64_C( -413.85),
                         EASYSIMD_FLOAT64_C(  752.26), EASYSIMD_FLOAT64_C(  609.29)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  699.07), EASYSIMD_FLOAT64_C( -413.85),
                         EASYSIMD_FLOAT64_C(  709.96), EASYSIMD_FLOAT64_C( -767.34)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  699.07), EASYSIMD_FLOAT64_C( -842.15),
                         EASYSIMD_FLOAT64_C(  709.96), EASYSIMD_FLOAT64_C( -767.34)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -695.21), EASYSIMD_FLOAT64_C( -786.23),
                         EASYSIMD_FLOAT64_C(  873.30), EASYSIMD_FLOAT64_C(  241.45)),
      UINT8_C(217),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  180.78), EASYSIMD_FLOAT64_C( -289.23),
                         EASYSIMD_FLOAT64_C(  918.52), EASYSIMD_FLOAT64_C( -422.76),
                         EASYSIMD_FLOAT64_C( -433.33), EASYSIMD_FLOAT64_C(   48.49),
                         EASYSIMD_FLOAT64_C(  799.57), EASYSIMD_FLOAT64_C( -820.22)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -433.33), EASYSIMD_FLOAT64_C( -786.23),
                         EASYSIMD_FLOAT64_C(  873.30), EASYSIMD_FLOAT64_C( -820.22)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  180.78), EASYSIMD_FLOAT64_C( -786.23),
                         EASYSIMD_FLOAT64_C(  873.30), EASYSIMD_FLOAT64_C( -422.76)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -712.24), EASYSIMD_FLOAT64_C(  518.12),
                         EASYSIMD_FLOAT64_C( -173.80), EASYSIMD_FLOAT64_C(  487.08)),
      UINT8_C(237),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -415.61), EASYSIMD_FLOAT64_C(  994.61),
                         EASYSIMD_FLOAT64_C( -659.70), EASYSIMD_FLOAT64_C(  493.30),
                         EASYSIMD_FLOAT64_C(  831.29), EASYSIMD_FLOAT64_C( -619.50),
                         EASYSIMD_FLOAT64_C(  952.47), EASYSIMD_FLOAT64_C( -492.61)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  831.29), EASYSIMD_FLOAT64_C( -619.50),
                         EASYSIMD_FLOAT64_C( -173.80), EASYSIMD_FLOAT64_C( -492.61)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -415.61), EASYSIMD_FLOAT64_C(  994.61),
                         EASYSIMD_FLOAT64_C( -173.80), EASYSIMD_FLOAT64_C(  493.30)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m256d r;
    r = easysimd_mm512_mask_extractf64x4_pd(src, k, a, 0);
    easysimd_assert_m256d_close(r, test_vec[i].r0, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extractf64x4_pd(src, k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extractf64x4_pd");
    easysimd_assert_m256d_close(r, test_vec[i].r1, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_extractf64x4_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m256d r0;
    easysimd__m256d r1;
  } test_vec[8] = {
    { UINT8_C( 21),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -139.11), EASYSIMD_FLOAT64_C( -172.36),
                         EASYSIMD_FLOAT64_C( -268.86), EASYSIMD_FLOAT64_C(  393.53),
                         EASYSIMD_FLOAT64_C(  -71.72), EASYSIMD_FLOAT64_C(   36.69),
                         EASYSIMD_FLOAT64_C(   98.47), EASYSIMD_FLOAT64_C( -135.52)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   36.69),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -135.52)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -172.36),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  393.53)) },
    { UINT8_C(150),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -556.90), EASYSIMD_FLOAT64_C(  522.06),
                         EASYSIMD_FLOAT64_C(  160.98), EASYSIMD_FLOAT64_C( -932.28),
                         EASYSIMD_FLOAT64_C(  391.82), EASYSIMD_FLOAT64_C(  600.12),
                         EASYSIMD_FLOAT64_C( -569.99), EASYSIMD_FLOAT64_C( -491.12)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  600.12),
                         EASYSIMD_FLOAT64_C( -569.99), EASYSIMD_FLOAT64_C(    0.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  522.06),
                         EASYSIMD_FLOAT64_C(  160.98), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(132),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  846.15), EASYSIMD_FLOAT64_C(   69.04),
                         EASYSIMD_FLOAT64_C( -514.56), EASYSIMD_FLOAT64_C( -149.02),
                         EASYSIMD_FLOAT64_C( -860.98), EASYSIMD_FLOAT64_C(  240.79),
                         EASYSIMD_FLOAT64_C( -280.30), EASYSIMD_FLOAT64_C( -839.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  240.79),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   69.04),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(158),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -471.60), EASYSIMD_FLOAT64_C( -333.94),
                         EASYSIMD_FLOAT64_C(  483.08), EASYSIMD_FLOAT64_C(  903.50),
                         EASYSIMD_FLOAT64_C(  232.04), EASYSIMD_FLOAT64_C(  -43.35),
                         EASYSIMD_FLOAT64_C(  774.81), EASYSIMD_FLOAT64_C(  309.91)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  232.04), EASYSIMD_FLOAT64_C(  -43.35),
                         EASYSIMD_FLOAT64_C(  774.81), EASYSIMD_FLOAT64_C(    0.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -471.60), EASYSIMD_FLOAT64_C( -333.94),
                         EASYSIMD_FLOAT64_C(  483.08), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(192),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  490.41), EASYSIMD_FLOAT64_C(  456.89),
                         EASYSIMD_FLOAT64_C(   47.59), EASYSIMD_FLOAT64_C(   51.33),
                         EASYSIMD_FLOAT64_C( -895.71), EASYSIMD_FLOAT64_C( -868.59),
                         EASYSIMD_FLOAT64_C( -736.92), EASYSIMD_FLOAT64_C( -921.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(209),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  841.53), EASYSIMD_FLOAT64_C( -292.35),
                         EASYSIMD_FLOAT64_C(  526.21), EASYSIMD_FLOAT64_C( -835.53),
                         EASYSIMD_FLOAT64_C( -203.04), EASYSIMD_FLOAT64_C(  571.79),
                         EASYSIMD_FLOAT64_C(  -80.71), EASYSIMD_FLOAT64_C(  675.92)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  675.92)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -835.53)) },
    { UINT8_C( 43),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -925.69), EASYSIMD_FLOAT64_C( -744.41),
                         EASYSIMD_FLOAT64_C(  717.83), EASYSIMD_FLOAT64_C(   39.32),
                         EASYSIMD_FLOAT64_C( -489.88), EASYSIMD_FLOAT64_C(   29.68),
                         EASYSIMD_FLOAT64_C(  -37.49), EASYSIMD_FLOAT64_C( -490.28)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -489.88), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  -37.49), EASYSIMD_FLOAT64_C( -490.28)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -925.69), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  717.83), EASYSIMD_FLOAT64_C(   39.32)) },
    { UINT8_C(120),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -934.92), EASYSIMD_FLOAT64_C( -213.75),
                         EASYSIMD_FLOAT64_C( -657.02), EASYSIMD_FLOAT64_C(  403.00),
                         EASYSIMD_FLOAT64_C( -629.37), EASYSIMD_FLOAT64_C( -198.67),
                         EASYSIMD_FLOAT64_C(  337.35), EASYSIMD_FLOAT64_C(  447.98)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -629.37), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -934.92), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) { 
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m256d r;
    r = easysimd_mm512_maskz_extractf64x4_pd(k, a, 0);
    easysimd_assert_m256d_close(r, test_vec[i].r0, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extractf64x4_pd(k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extractf64x4_pd");

    r = easysimd_mm512_maskz_extractf64x4_pd(test_vec[i].k, test_vec[i].a, 1);
    easysimd_assert_m256d_close(r, test_vec[i].r1, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_extracti32x4_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m128i r0;
    easysimd__m128i r1;
    easysimd__m128i r2;
    easysimd__m128i r3;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C(  936676195), INT32_C( -430989686), INT32_C( -131327474), INT32_C(  910508384),
                            INT32_C( 1148801293), INT32_C(-1204409147), INT32_C( 1922921929), INT32_C( 2087027240),
                            INT32_C( 1221368626), INT32_C(-1114006136), INT32_C( 2023469730), INT32_C(  463308257),
                            INT32_C(-1206798920), INT32_C( -968449396), INT32_C( -580990777), INT32_C( -896508445)),
      easysimd_mm_set_epi32(INT32_C(-1206798920), INT32_C( -968449396), INT32_C( -580990777), INT32_C( -896508445)),
      easysimd_mm_set_epi32(INT32_C( 1221368626), INT32_C(-1114006136), INT32_C( 2023469730), INT32_C(  463308257)),
      easysimd_mm_set_epi32(INT32_C( 1148801293), INT32_C(-1204409147), INT32_C( 1922921929), INT32_C( 2087027240)),
      easysimd_mm_set_epi32(INT32_C(  936676195), INT32_C( -430989686), INT32_C( -131327474), INT32_C(  910508384)) },
    { easysimd_mm512_set_epi32(INT32_C(-1889562474), INT32_C( 2009910179), INT32_C(  815981096), INT32_C( -945310665),
                            INT32_C(-1749696319), INT32_C( 1237778758), INT32_C( 2121903540), INT32_C(-1272250958),
                            INT32_C(-1384883580), INT32_C(  455864550), INT32_C(-1132110758), INT32_C(  636699315),
                            INT32_C( 1712647786), INT32_C( 1137382661), INT32_C(  -72212648), INT32_C(-1857153584)),
      easysimd_mm_set_epi32(INT32_C( 1712647786), INT32_C( 1137382661), INT32_C(  -72212648), INT32_C(-1857153584)),
      easysimd_mm_set_epi32(INT32_C(-1384883580), INT32_C(  455864550), INT32_C(-1132110758), INT32_C(  636699315)),
      easysimd_mm_set_epi32(INT32_C(-1749696319), INT32_C( 1237778758), INT32_C( 2121903540), INT32_C(-1272250958)),
      easysimd_mm_set_epi32(INT32_C(-1889562474), INT32_C( 2009910179), INT32_C(  815981096), INT32_C( -945310665)) },
    { easysimd_mm512_set_epi32(INT32_C(-1732993162), INT32_C( 1212743926), INT32_C( 1966971402), INT32_C(-1506668774),
                            INT32_C(-1700657265), INT32_C( 1944327234), INT32_C( -355879099), INT32_C(-1588067414),
                            INT32_C(  301696052), INT32_C( 1998339065), INT32_C(-2060809025), INT32_C(-1942156019),
                            INT32_C(  551689125), INT32_C(  669995747), INT32_C(-1196653219), INT32_C( -147816939)),
      easysimd_mm_set_epi32(INT32_C(  551689125), INT32_C(  669995747), INT32_C(-1196653219), INT32_C( -147816939)),
      easysimd_mm_set_epi32(INT32_C(  301696052), INT32_C( 1998339065), INT32_C(-2060809025), INT32_C(-1942156019)),
      easysimd_mm_set_epi32(INT32_C(-1700657265), INT32_C( 1944327234), INT32_C( -355879099), INT32_C(-1588067414)),
      easysimd_mm_set_epi32(INT32_C(-1732993162), INT32_C( 1212743926), INT32_C( 1966971402), INT32_C(-1506668774)) },
    { easysimd_mm512_set_epi32(INT32_C( -720429002), INT32_C( 1666176035), INT32_C( -786738545), INT32_C(  356149527),
                            INT32_C( 1809623523), INT32_C(  387697241), INT32_C( -984752565), INT32_C(-1905225073),
                            INT32_C( 1936855390), INT32_C(-1150638889), INT32_C( -620356961), INT32_C(-1540113901),
                            INT32_C(  485150966), INT32_C( -340934070), INT32_C( 1258270405), INT32_C(-1976197296)),
      easysimd_mm_set_epi32(INT32_C(  485150966), INT32_C( -340934070), INT32_C( 1258270405), INT32_C(-1976197296)),
      easysimd_mm_set_epi32(INT32_C( 1936855390), INT32_C(-1150638889), INT32_C( -620356961), INT32_C(-1540113901)),
      easysimd_mm_set_epi32(INT32_C( 1809623523), INT32_C(  387697241), INT32_C( -984752565), INT32_C(-1905225073)),
      easysimd_mm_set_epi32(INT32_C( -720429002), INT32_C( 1666176035), INT32_C( -786738545), INT32_C(  356149527)) },
    { easysimd_mm512_set_epi32(INT32_C(-1844524534), INT32_C(  359706932), INT32_C(   21147132), INT32_C(-1205907433),
                            INT32_C(  241660444), INT32_C(-1425169590), INT32_C( 1296561443), INT32_C(-1934442075),
                            INT32_C( 2141890625), INT32_C( 2063982974), INT32_C(-1791266937), INT32_C(-1677757015),
                            INT32_C(-1341587157), INT32_C(   71085124), INT32_C( 1045857655), INT32_C(  563120574)),
      easysimd_mm_set_epi32(INT32_C(-1341587157), INT32_C(   71085124), INT32_C( 1045857655), INT32_C(  563120574)),
      easysimd_mm_set_epi32(INT32_C( 2141890625), INT32_C( 2063982974), INT32_C(-1791266937), INT32_C(-1677757015)),
      easysimd_mm_set_epi32(INT32_C(  241660444), INT32_C(-1425169590), INT32_C( 1296561443), INT32_C(-1934442075)),
      easysimd_mm_set_epi32(INT32_C(-1844524534), INT32_C(  359706932), INT32_C(   21147132), INT32_C(-1205907433)) },
    { easysimd_mm512_set_epi32(INT32_C( 1760980702), INT32_C(-1592941833), INT32_C(-1618734568), INT32_C(-1937346052),
                            INT32_C( -716563340), INT32_C(-1364071584), INT32_C( -516953475), INT32_C( 1021791773),
                            INT32_C(  587319712), INT32_C(-1327772936), INT32_C( -388433125), INT32_C(-1835488163),
                            INT32_C( 1934085090), INT32_C( 1823172786), INT32_C( -962834173), INT32_C(-1813383694)),
      easysimd_mm_set_epi32(INT32_C( 1934085090), INT32_C( 1823172786), INT32_C( -962834173), INT32_C(-1813383694)),
      easysimd_mm_set_epi32(INT32_C(  587319712), INT32_C(-1327772936), INT32_C( -388433125), INT32_C(-1835488163)),
      easysimd_mm_set_epi32(INT32_C( -716563340), INT32_C(-1364071584), INT32_C( -516953475), INT32_C( 1021791773)),
      easysimd_mm_set_epi32(INT32_C( 1760980702), INT32_C(-1592941833), INT32_C(-1618734568), INT32_C(-1937346052)) },
    { easysimd_mm512_set_epi32(INT32_C(-1600993635), INT32_C( 1692797667), INT32_C( -524624106), INT32_C( -196896874),
                            INT32_C( 1365949044), INT32_C( 1838002887), INT32_C( -452898509), INT32_C( 1408911553),
                            INT32_C(-1407150071), INT32_C(  -82352116), INT32_C( -745337283), INT32_C( -527368953),
                            INT32_C(-1750389986), INT32_C( 1217697098), INT32_C( 1169663592), INT32_C(-1794175196)),
      easysimd_mm_set_epi32(INT32_C(-1750389986), INT32_C( 1217697098), INT32_C( 1169663592), INT32_C(-1794175196)),
      easysimd_mm_set_epi32(INT32_C(-1407150071), INT32_C(  -82352116), INT32_C( -745337283), INT32_C( -527368953)),
      easysimd_mm_set_epi32(INT32_C( 1365949044), INT32_C( 1838002887), INT32_C( -452898509), INT32_C( 1408911553)),
      easysimd_mm_set_epi32(INT32_C(-1600993635), INT32_C( 1692797667), INT32_C( -524624106), INT32_C( -196896874)) },
    { easysimd_mm512_set_epi32(INT32_C(   31532768), INT32_C(-1104316005), INT32_C(-1643683522), INT32_C( -459507150),
                            INT32_C( -301521916), INT32_C(-1095317885), INT32_C( 1268414902), INT32_C( -436965349),
                            INT32_C(  330503221), INT32_C( 1614750696), INT32_C( 1262893786), INT32_C( 1956553172),
                            INT32_C(-1113093793), INT32_C(-1782413198), INT32_C( 1413241306), INT32_C(-1360271723)),
      easysimd_mm_set_epi32(INT32_C(-1113093793), INT32_C(-1782413198), INT32_C( 1413241306), INT32_C(-1360271723)),
      easysimd_mm_set_epi32(INT32_C(  330503221), INT32_C( 1614750696), INT32_C( 1262893786), INT32_C( 1956553172)),
      easysimd_mm_set_epi32(INT32_C( -301521916), INT32_C(-1095317885), INT32_C( 1268414902), INT32_C( -436965349)),
      easysimd_mm_set_epi32(INT32_C(   31532768), INT32_C(-1104316005), INT32_C(-1643683522), INT32_C( -459507150)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    r = easysimd_mm512_extracti32x4_epi32(test_vec[i].a, 0);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r0);
    r = easysimd_mm512_extracti32x4_epi32(test_vec[i].a, 1);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r1);
    r = easysimd_mm512_extracti32x4_epi32(test_vec[i].a, 2);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r2);

    easysimd__m512i a = test_vec[i].a;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extracti32x4_epi32(a, 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extracti32x4_epi32");
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r3);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_extracti32x4_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m128i r0;
    easysimd__m128i r1;
    easysimd__m128i r2;
    easysimd__m128i r3;
  } test_vec[8] = {
    {  easysimd_mm_set_epi32(INT32_C( 1993455974), INT32_C(-2068684593), INT32_C(-1936012201), INT32_C( 1856459607)),
      UINT8_C(  6),
      easysimd_mm512_set_epi32(INT32_C(-1630396605), INT32_C( 1545554432), INT32_C(  344023940), INT32_C(-1871515754),
                            INT32_C(  951544639), INT32_C(-1026363374), INT32_C(-1801776439), INT32_C(  145438126),
                            INT32_C(-1306064352), INT32_C( -858736392), INT32_C(  923442479), INT32_C( 1092805562),
                            INT32_C( 1443901717), INT32_C( 1848749100), INT32_C( 1777333881), INT32_C( 1570116932)),
      easysimd_mm_set_epi32(INT32_C( 1993455974), INT32_C( 1848749100), INT32_C( 1777333881), INT32_C( 1856459607)),
      easysimd_mm_set_epi32(INT32_C( 1993455974), INT32_C( -858736392), INT32_C(  923442479), INT32_C( 1856459607)),
      easysimd_mm_set_epi32(INT32_C( 1993455974), INT32_C(-1026363374), INT32_C(-1801776439), INT32_C( 1856459607)),
      easysimd_mm_set_epi32(INT32_C( 1993455974), INT32_C( 1545554432), INT32_C(  344023940), INT32_C( 1856459607)) },
    {  easysimd_mm_set_epi32(INT32_C(-1999224530), INT32_C( 1042470181), INT32_C( 1827473477), INT32_C(  298546792)),
      UINT8_C(109),
      easysimd_mm512_set_epi32(INT32_C(-2045280751), INT32_C(-2037261521), INT32_C(  223952317), INT32_C(  282198336),
                            INT32_C(  564965997), INT32_C(  169645898), INT32_C(-1539616610), INT32_C( 1134735685),
                            INT32_C( 1430356381), INT32_C(-1110068455), INT32_C( -207240031), INT32_C(-1649179267),
                            INT32_C( 2054398444), INT32_C( -483586503), INT32_C(-1481960002), INT32_C(  861125508)),
      easysimd_mm_set_epi32(INT32_C( 2054398444), INT32_C( -483586503), INT32_C( 1827473477), INT32_C(  861125508)),
      easysimd_mm_set_epi32(INT32_C( 1430356381), INT32_C(-1110068455), INT32_C( 1827473477), INT32_C(-1649179267)),
      easysimd_mm_set_epi32(INT32_C(  564965997), INT32_C(  169645898), INT32_C( 1827473477), INT32_C( 1134735685)),
      easysimd_mm_set_epi32(INT32_C(-2045280751), INT32_C(-2037261521), INT32_C( 1827473477), INT32_C(  282198336)) },
    {  easysimd_mm_set_epi32(INT32_C( -695949043), INT32_C( -790242624), INT32_C(-1094331335), INT32_C(-1166320093)),
      UINT8_C(181),
      easysimd_mm512_set_epi32(INT32_C( 1549802795), INT32_C(  159583350), INT32_C(  548883180), INT32_C( -605945909),
                            INT32_C(-2063050181), INT32_C( 1095467003), INT32_C(-2083755741), INT32_C( 2066979701),
                            INT32_C( 1094609712), INT32_C( 1345059025), INT32_C( -340318359), INT32_C( 1519671047),
                            INT32_C(-1017461983), INT32_C(  353198331), INT32_C( 1711460779), INT32_C( -919570191)),
      easysimd_mm_set_epi32(INT32_C( -695949043), INT32_C(  353198331), INT32_C(-1094331335), INT32_C( -919570191)),
      easysimd_mm_set_epi32(INT32_C( -695949043), INT32_C( 1345059025), INT32_C(-1094331335), INT32_C( 1519671047)),
      easysimd_mm_set_epi32(INT32_C( -695949043), INT32_C( 1095467003), INT32_C(-1094331335), INT32_C( 2066979701)),
      easysimd_mm_set_epi32(INT32_C( -695949043), INT32_C(  159583350), INT32_C(-1094331335), INT32_C( -605945909)) },
    {  easysimd_mm_set_epi32(INT32_C(  795925067), INT32_C( 1720852541), INT32_C(-1423023772), INT32_C(-1185448755)),
      UINT8_C(176),
      easysimd_mm512_set_epi32(INT32_C(  884163960), INT32_C( -329275629), INT32_C( -888441293), INT32_C( -707551350),
                            INT32_C(  513515868), INT32_C(-1825967755), INT32_C(  822222164), INT32_C(-1689559027),
                            INT32_C(  533478787), INT32_C(  907615417), INT32_C( -199229058), INT32_C(  -91537812),
                            INT32_C( 1375258232), INT32_C(  139748399), INT32_C( 1688468565), INT32_C(  736544549)),
      easysimd_mm_set_epi32(INT32_C(  795925067), INT32_C( 1720852541), INT32_C(-1423023772), INT32_C(-1185448755)),
      easysimd_mm_set_epi32(INT32_C(  795925067), INT32_C( 1720852541), INT32_C(-1423023772), INT32_C(-1185448755)),
      easysimd_mm_set_epi32(INT32_C(  795925067), INT32_C( 1720852541), INT32_C(-1423023772), INT32_C(-1185448755)),
      easysimd_mm_set_epi32(INT32_C(  795925067), INT32_C( 1720852541), INT32_C(-1423023772), INT32_C(-1185448755)) },
    {  easysimd_mm_set_epi32(INT32_C( -622852205), INT32_C( -839037220), INT32_C(  499633910), INT32_C( -260167255)),
      UINT8_C( 21),
      easysimd_mm512_set_epi32(INT32_C( -272088075), INT32_C(  386072301), INT32_C(-1628984154), INT32_C(   87817524),
                            INT32_C( 1219490517), INT32_C(-1569831145), INT32_C(  338985942), INT32_C( 1701079465),
                            INT32_C( -195770682), INT32_C(  503748315), INT32_C( 1469355417), INT32_C(-1849349632),
                            INT32_C( 1962664621), INT32_C( -646247370), INT32_C( 1258747662), INT32_C( 1838830023)),
      easysimd_mm_set_epi32(INT32_C( -622852205), INT32_C( -646247370), INT32_C(  499633910), INT32_C( 1838830023)),
      easysimd_mm_set_epi32(INT32_C( -622852205), INT32_C(  503748315), INT32_C(  499633910), INT32_C(-1849349632)),
      easysimd_mm_set_epi32(INT32_C( -622852205), INT32_C(-1569831145), INT32_C(  499633910), INT32_C( 1701079465)),
      easysimd_mm_set_epi32(INT32_C( -622852205), INT32_C(  386072301), INT32_C(  499633910), INT32_C(   87817524)) },
    {  easysimd_mm_set_epi32(INT32_C(  654527510), INT32_C(-2043358500), INT32_C(  459072440), INT32_C( -430427651)),
      UINT8_C(229),
      easysimd_mm512_set_epi32(INT32_C(  617951303), INT32_C(  817116152), INT32_C(-1034835761), INT32_C( -102069057),
                            INT32_C( 1774242298), INT32_C( 1089620040), INT32_C(-1101477862), INT32_C( 2001101785),
                            INT32_C(-1759250988), INT32_C( -606254738), INT32_C( 1526367108), INT32_C(  722122834),
                            INT32_C( -174985661), INT32_C(-1762469023), INT32_C( 1239606494), INT32_C(  -22119232)),
      easysimd_mm_set_epi32(INT32_C(  654527510), INT32_C(-1762469023), INT32_C(  459072440), INT32_C(  -22119232)),
      easysimd_mm_set_epi32(INT32_C(  654527510), INT32_C( -606254738), INT32_C(  459072440), INT32_C(  722122834)),
      easysimd_mm_set_epi32(INT32_C(  654527510), INT32_C( 1089620040), INT32_C(  459072440), INT32_C( 2001101785)),
      easysimd_mm_set_epi32(INT32_C(  654527510), INT32_C(  817116152), INT32_C(  459072440), INT32_C( -102069057)) },
    {  easysimd_mm_set_epi32(INT32_C(-2034110695), INT32_C(-1088138491), INT32_C( -353174912), INT32_C( -362301616)),
      UINT8_C( 42),
      easysimd_mm512_set_epi32(INT32_C(  204417556), INT32_C(-1329665093), INT32_C(-2039025377), INT32_C( 1639231015),
                            INT32_C( 1541217841), INT32_C( 1692413538), INT32_C(  738521275), INT32_C(  159429100),
                            INT32_C(  451955897), INT32_C(  181201098), INT32_C(  450627934), INT32_C( 2082954477),
                            INT32_C( 1254960767), INT32_C( 1995459397), INT32_C(  -11572946), INT32_C(-1087388220)),
      easysimd_mm_set_epi32(INT32_C( 1254960767), INT32_C(-1088138491), INT32_C(  -11572946), INT32_C( -362301616)),
      easysimd_mm_set_epi32(INT32_C(  451955897), INT32_C(-1088138491), INT32_C(  450627934), INT32_C( -362301616)),
      easysimd_mm_set_epi32(INT32_C( 1541217841), INT32_C(-1088138491), INT32_C(  738521275), INT32_C( -362301616)),
      easysimd_mm_set_epi32(INT32_C(  204417556), INT32_C(-1088138491), INT32_C(-2039025377), INT32_C( -362301616)) },
    {  easysimd_mm_set_epi32(INT32_C(-1687118128), INT32_C(  107945377), INT32_C( 1174128677), INT32_C(-1544325740)),
      UINT8_C(132),
      easysimd_mm512_set_epi32(INT32_C( -852914371), INT32_C( -773785464), INT32_C(-2142007253), INT32_C(  466013192),
                            INT32_C( 1313258175), INT32_C( 1928049651), INT32_C(  765730488), INT32_C(  -85899231),
                            INT32_C( 1435935141), INT32_C(-2098236580), INT32_C(-1991433794), INT32_C( 1298943776),
                            INT32_C(  277470244), INT32_C(-1834748849), INT32_C(  596054477), INT32_C( 1827419510)),
      easysimd_mm_set_epi32(INT32_C(-1687118128), INT32_C(-1834748849), INT32_C( 1174128677), INT32_C(-1544325740)),
      easysimd_mm_set_epi32(INT32_C(-1687118128), INT32_C(-2098236580), INT32_C( 1174128677), INT32_C(-1544325740)),
      easysimd_mm_set_epi32(INT32_C(-1687118128), INT32_C( 1928049651), INT32_C( 1174128677), INT32_C(-1544325740)),
      easysimd_mm_set_epi32(INT32_C(-1687118128), INT32_C( -773785464), INT32_C( 1174128677), INT32_C(-1544325740)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    r = easysimd_mm512_mask_extracti32x4_epi32(test_vec[i].src, test_vec[i].k, test_vec[i].a, 0);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r0);
    r = easysimd_mm512_mask_extracti32x4_epi32(test_vec[i].src, test_vec[i].k, test_vec[i].a, 1);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r1);
    r = easysimd_mm512_mask_extracti32x4_epi32(test_vec[i].src, test_vec[i].k, test_vec[i].a, 2);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r2);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extracti32x4_epi32(test_vec[i].src, test_vec[i].k, test_vec[i].a, 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extracti32x4_epi32");
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r3);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_extracti32x4_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m128i r0;
    easysimd__m128i r1;
    easysimd__m128i r2;
    easysimd__m128i r3;
  } test_vec[8] = {
    { UINT8_C( 87),
      easysimd_mm512_set_epi32(INT32_C(  951544639), INT32_C(-1026363374), INT32_C(-1801776439), INT32_C(  145438126),
                            INT32_C(-1306064352), INT32_C( -858736392), INT32_C(  923442479), INT32_C( 1092805562),
                            INT32_C( 1443901717), INT32_C( 1848749100), INT32_C( 1777333881), INT32_C( 1570116932),
                            INT32_C(-1302383354), INT32_C( 1993455974), INT32_C(-2068684593), INT32_C(-1936012201)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1993455974), INT32_C(-2068684593), INT32_C(-1936012201)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1848749100), INT32_C( 1777333881), INT32_C( 1570116932)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( -858736392), INT32_C(  923442479), INT32_C( 1092805562)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(-1026363374), INT32_C(-1801776439), INT32_C(  145438126)) },
    { UINT8_C(150),
      easysimd_mm512_set_epi32(INT32_C( 1430356381), INT32_C(-1110068455), INT32_C( -207240031), INT32_C(-1649179267),
                            INT32_C( 2054398444), INT32_C( -483586503), INT32_C(-1481960002), INT32_C(  861125508),
                            INT32_C( -330381203), INT32_C(-1999224530), INT32_C( 1042470181), INT32_C( 1827473477),
                            INT32_C(  298546792), INT32_C(-1630396605), INT32_C( 1545554432), INT32_C(  344023940)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(-1630396605), INT32_C( 1545554432), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(-1999224530), INT32_C( 1042470181), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( -483586503), INT32_C(-1481960002), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(-1110068455), INT32_C( -207240031), INT32_C(          0)) },
    { UINT8_C( 69),
      easysimd_mm512_set_epi32(INT32_C(-1017461983), INT32_C(  353198331), INT32_C( 1711460779), INT32_C( -919570191),
                            INT32_C( 1974152373), INT32_C( -695949043), INT32_C( -790242624), INT32_C(-1094331335),
                            INT32_C(-1166320093), INT32_C(-2045280751), INT32_C(-2037261521), INT32_C(  223952317),
                            INT32_C(  282198336), INT32_C(  564965997), INT32_C(  169645898), INT32_C(-1539616610)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(  564965997), INT32_C(          0), INT32_C(-1539616610)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(-2045280751), INT32_C(          0), INT32_C(  223952317)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( -695949043), INT32_C(          0), INT32_C(-1094331335)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(  353198331), INT32_C(          0), INT32_C( -919570191)) },
    { UINT8_C(  7),
      easysimd_mm512_set_epi32(INT32_C(-1282057552), INT32_C(  795925067), INT32_C( 1720852541), INT32_C(-1423023772),
                            INT32_C(-1185448755), INT32_C( 1549802795), INT32_C(  159583350), INT32_C(  548883180),
                            INT32_C( -605945909), INT32_C(-2063050181), INT32_C( 1095467003), INT32_C(-2083755741),
                            INT32_C( 2066979701), INT32_C( 1094609712), INT32_C( 1345059025), INT32_C( -340318359)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1094609712), INT32_C( 1345059025), INT32_C( -340318359)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(-2063050181), INT32_C( 1095467003), INT32_C(-2083755741)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1549802795), INT32_C(  159583350), INT32_C(  548883180)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(  795925067), INT32_C( 1720852541), INT32_C(-1423023772)) },
    { UINT8_C( 37),
      easysimd_mm512_set_epi32(INT32_C( -260167255), INT32_C(  884163960), INT32_C( -329275629), INT32_C( -888441293),
                            INT32_C( -707551350), INT32_C(  513515868), INT32_C(-1825967755), INT32_C(  822222164),
                            INT32_C(-1689559027), INT32_C(  533478787), INT32_C(  907615417), INT32_C( -199229058),
                            INT32_C(  -91537812), INT32_C( 1375258232), INT32_C(  139748399), INT32_C( 1688468565)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1375258232), INT32_C(          0), INT32_C( 1688468565)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(  533478787), INT32_C(          0), INT32_C( -199229058)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(  513515868), INT32_C(          0), INT32_C(  822222164)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(  884163960), INT32_C(          0), INT32_C( -888441293)) },
    { UINT8_C(246),
      easysimd_mm512_set_epi32(INT32_C(   87817524), INT32_C( 1219490517), INT32_C(-1569831145), INT32_C(  338985942),
                            INT32_C( 1701079465), INT32_C( -195770682), INT32_C(  503748315), INT32_C( 1469355417),
                            INT32_C(-1849349632), INT32_C( 1962664621), INT32_C( -646247370), INT32_C( 1258747662),
                            INT32_C( 1838830023), INT32_C( -532007659), INT32_C( -622852205), INT32_C( -839037220)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( -532007659), INT32_C( -622852205), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1962664621), INT32_C( -646247370), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( -195770682), INT32_C(  503748315), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1219490517), INT32_C(-1569831145), INT32_C(          0)) },
    { UINT8_C(166),
      easysimd_mm512_set_epi32(INT32_C( 2001101785), INT32_C(-1759250988), INT32_C( -606254738), INT32_C( 1526367108),
                            INT32_C(  722122834), INT32_C( -174985661), INT32_C(-1762469023), INT32_C( 1239606494),
                            INT32_C(  -22119232), INT32_C( 1216907749), INT32_C(  654527510), INT32_C(-2043358500),
                            INT32_C(  459072440), INT32_C( -430427651), INT32_C( -272088075), INT32_C(  386072301)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( -430427651), INT32_C( -272088075), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( 1216907749), INT32_C(  654527510), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C( -174985661), INT32_C(-1762469023), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(-1759250988), INT32_C( -606254738), INT32_C(          0)) },
    { UINT8_C( 26),
      easysimd_mm512_set_epi32(INT32_C( 2082954477), INT32_C( 1254960767), INT32_C( 1995459397), INT32_C(  -11572946),
                            INT32_C(-1087388220), INT32_C(  730787370), INT32_C(-2034110695), INT32_C(-1088138491),
                            INT32_C( -353174912), INT32_C( -362301616), INT32_C(  617951303), INT32_C(  817116152),
                            INT32_C(-1034835761), INT32_C( -102069057), INT32_C( 1774242298), INT32_C( 1089620040)),
      easysimd_mm_set_epi32(INT32_C(-1034835761), INT32_C(          0), INT32_C( 1774242298), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C( -353174912), INT32_C(          0), INT32_C(  617951303), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(-1087388220), INT32_C(          0), INT32_C(-2034110695), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C( 2082954477), INT32_C(          0), INT32_C( 1995459397), INT32_C(          0)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    r = easysimd_mm512_maskz_extracti32x4_epi32(test_vec[i].k, test_vec[i].a, 0);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r0);
    r = easysimd_mm512_maskz_extracti32x4_epi32(test_vec[i].k, test_vec[i].a, 1);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r1);
    r = easysimd_mm512_maskz_extracti32x4_epi32(test_vec[i].k, test_vec[i].a, 2);
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r2);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extracti32x4_epi32(test_vec[i].k, test_vec[i].a, 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extracti32x4_epi32");
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r3);
  }

  return 0;
}

static int
test_easysimd_mm512_extracti32x8_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t a[16];
    const int32_t r0[8];
    const int32_t r1[8];
  } test_vec[8] = {
    { {  INT32_C(  1533917321), -INT32_C(   120154707),  INT32_C(   747524052),  INT32_C(  1563597506),  INT32_C(   769357336),  INT32_C(  1684241017), -INT32_C(   837940437), -INT32_C(   245863064),
        -INT32_C(   766720475),  INT32_C(   818553435),  INT32_C(   878467186),  INT32_C(   177311474),  INT32_C(  2100849668),  INT32_C(   299998182),  INT32_C(   249622438), -INT32_C(  2097203108) },
      {  INT32_C(  1533917321), -INT32_C(   120154707),  INT32_C(   747524052),  INT32_C(  1563597506),  INT32_C(   769357336),  INT32_C(  1684241017), -INT32_C(   837940437), -INT32_C(   245863064) },
      { -INT32_C(   766720475),  INT32_C(   818553435),  INT32_C(   878467186),  INT32_C(   177311474),  INT32_C(  2100849668),  INT32_C(   299998182),  INT32_C(   249622438), -INT32_C(  2097203108) } },
    { {  INT32_C(  1515474174), -INT32_C(   527818898),  INT32_C(  1779754615),  INT32_C(  2020910452), -INT32_C(   118117358), -INT32_C(   301279673),  INT32_C(   587000517),  INT32_C(   564460578),
        -INT32_C(  1233389496), -INT32_C(  1885928169),  INT32_C(  1610197995),  INT32_C(  1675062609),  INT32_C(  1633405978),  INT32_C(  1733256610),  INT32_C(  1921600591), -INT32_C(  1869402808) },
      {  INT32_C(  1515474174), -INT32_C(   527818898),  INT32_C(  1779754615),  INT32_C(  2020910452), -INT32_C(   118117358), -INT32_C(   301279673),  INT32_C(   587000517),  INT32_C(   564460578) },
      { -INT32_C(  1233389496), -INT32_C(  1885928169),  INT32_C(  1610197995),  INT32_C(  1675062609),  INT32_C(  1633405978),  INT32_C(  1733256610),  INT32_C(  1921600591), -INT32_C(  1869402808) } },
    { {  INT32_C(  1028001318), -INT32_C(    20128493), -INT32_C(   631388791),  INT32_C(  1295856691), -INT32_C(  1565616128),  INT32_C(  1292500734), -INT32_C(  1832938678), -INT32_C(   433958208),
         INT32_C(  1931765856), -INT32_C(   814616506), -INT32_C(   374747466),  INT32_C(    37152258),  INT32_C(  2091181182),  INT32_C(   751480546),  INT32_C(    29264193),  INT32_C(  1021894876) },
      {  INT32_C(  1028001318), -INT32_C(    20128493), -INT32_C(   631388791),  INT32_C(  1295856691), -INT32_C(  1565616128),  INT32_C(  1292500734), -INT32_C(  1832938678), -INT32_C(   433958208) },
      {  INT32_C(  1931765856), -INT32_C(   814616506), -INT32_C(   374747466),  INT32_C(    37152258),  INT32_C(  2091181182),  INT32_C(   751480546),  INT32_C(    29264193),  INT32_C(  1021894876) } },
    { { -INT32_C(  1884287927), -INT32_C(  1302453764), -INT32_C(   224720912),  INT32_C(  1811272173), -INT32_C(  1729586762), -INT32_C(  2000309689),  INT32_C(   394953531), -INT32_C(  1386974620),
         INT32_C(  2050753662),  INT32_C(   355310117), -INT32_C(  1912026975),  INT32_C(  1358560666), -INT32_C(   555163242), -INT32_C(   815354477), -INT32_C(  1780027343), -INT32_C(   532530590) },
      { -INT32_C(  1884287927), -INT32_C(  1302453764), -INT32_C(   224720912),  INT32_C(  1811272173), -INT32_C(  1729586762), -INT32_C(  2000309689),  INT32_C(   394953531), -INT32_C(  1386974620) },
      {  INT32_C(  2050753662),  INT32_C(   355310117), -INT32_C(  1912026975),  INT32_C(  1358560666), -INT32_C(   555163242), -INT32_C(   815354477), -INT32_C(  1780027343), -INT32_C(   532530590) } },
    { {  INT32_C(  1683717694), -INT32_C(  1183217640), -INT32_C(   364412592),  INT32_C(   356139134), -INT32_C(  1242356958),  INT32_C(    25450960), -INT32_C(  1399428278), -INT32_C(   460466011),
         INT32_C(  1850271830), -INT32_C(  1071136400), -INT32_C(  1045729725), -INT32_C(   774445649), -INT32_C(   645478136),  INT32_C(  1843006243),  INT32_C(   454652278), -INT32_C(  1593858487) },
      {  INT32_C(  1683717694), -INT32_C(  1183217640), -INT32_C(   364412592),  INT32_C(   356139134), -INT32_C(  1242356958),  INT32_C(    25450960), -INT32_C(  1399428278), -INT32_C(   460466011) },
      {  INT32_C(  1850271830), -INT32_C(  1071136400), -INT32_C(  1045729725), -INT32_C(   774445649), -INT32_C(   645478136),  INT32_C(  1843006243),  INT32_C(   454652278), -INT32_C(  1593858487) } },
    { { -INT32_C(    32618610),  INT32_C(  1287599625),  INT32_C(  1393388196),  INT32_C(  1478812751), -INT32_C(   802051155),  INT32_C(   742198198), -INT32_C(   968403076), -INT32_C(  1939453955),
        -INT32_C(  1752533874),  INT32_C(  1340295594),  INT32_C(    61010355), -INT32_C(  2107914283),  INT32_C(   676564082),  INT32_C(   341086359), -INT32_C(   438658073),  INT32_C(  1903247586) },
      { -INT32_C(    32618610),  INT32_C(  1287599625),  INT32_C(  1393388196),  INT32_C(  1478812751), -INT32_C(   802051155),  INT32_C(   742198198), -INT32_C(   968403076), -INT32_C(  1939453955) },
      { -INT32_C(  1752533874),  INT32_C(  1340295594),  INT32_C(    61010355), -INT32_C(  2107914283),  INT32_C(   676564082),  INT32_C(   341086359), -INT32_C(   438658073),  INT32_C(  1903247586) } },
    { {  INT32_C(  1594424244), -INT32_C(   122753979), -INT32_C(  1292152611), -INT32_C(  1993058793),  INT32_C(  2058455010), -INT32_C(     7469800), -INT32_C(  2082183007),  INT32_C(  1559516584),
        -INT32_C(  1766064815), -INT32_C(   963745303), -INT32_C(   780629318), -INT32_C(  1017402144),  INT32_C(  1279069236), -INT32_C(  1286878446), -INT32_C(   617205709), -INT32_C(   701027451) },
      {  INT32_C(  1594424244), -INT32_C(   122753979), -INT32_C(  1292152611), -INT32_C(  1993058793),  INT32_C(  2058455010), -INT32_C(     7469800), -INT32_C(  2082183007),  INT32_C(  1559516584) },
      { -INT32_C(  1766064815), -INT32_C(   963745303), -INT32_C(   780629318), -INT32_C(  1017402144),  INT32_C(  1279069236), -INT32_C(  1286878446), -INT32_C(   617205709), -INT32_C(   701027451) } },
    { {  INT32_C(   292352808),  INT32_C(   383253340),  INT32_C(  1709723525),  INT32_C(   791167995),  INT32_C(  1635476815),  INT32_C(  1662306096),  INT32_C(  2084457463), -INT32_C(  1638697354),
        -INT32_C(   978337943),  INT32_C(  1071417018), -INT32_C(   794442539),  INT32_C(  1442893063),  INT32_C(  1673034547),  INT32_C(   969395266), -INT32_C(  1917450985), -INT32_C(   466941573) },
      {  INT32_C(   292352808),  INT32_C(   383253340),  INT32_C(  1709723525),  INT32_C(   791167995),  INT32_C(  1635476815),  INT32_C(  1662306096),  INT32_C(  2084457463), -INT32_C(  1638697354) },
      { -INT32_C(   978337943),  INT32_C(  1071417018), -INT32_C(   794442539),  INT32_C(  1442893063),  INT32_C(  1673034547),  INT32_C(   969395266), -INT32_C(  1917450985), -INT32_C(   466941573) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    r = easysimd_mm512_extracti32x8_epi32(a, 0);
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r0));

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extracti32x8_epi32(a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extracti32x8_epi32");
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r1));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m256i r0 = easysimd_mm512_extracti32x8_epi32(a, 0);
    easysimd__m256i r1 = easysimd_mm512_extracti32x8_epi32(a, 1);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_extracti32x8_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[16];
    const int32_t r0[8];
    const int32_t r1[8];
  } test_vec[8] = {
    { { -INT32_C(  2085954872),  INT32_C(   901940832),  INT32_C(  1359374154),  INT32_C(  1755776565), -INT32_C(   993304703),  INT32_C(  1140691500),  INT32_C(   332510104), -INT32_C(  2047279940) },
      UINT8_C(215),
      {  INT32_C(   674695330),  INT32_C(   846360010),  INT32_C(  2036843379), -INT32_C(   889532566),  INT32_C(   771145370),  INT32_C(  1875196348),  INT32_C(   120379402),  INT32_C(  1927197136),
        -INT32_C(  2087053895), -INT32_C(   172684158),  INT32_C(   980294863), -INT32_C(  2046531093), -INT32_C(   474744281),  INT32_C(  1045657651),  INT32_C(   558202704), -INT32_C(   376233168) },
      {  INT32_C(   674695330),  INT32_C(   846360010),  INT32_C(  2036843379),  INT32_C(  1755776565),  INT32_C(   771145370),  INT32_C(  1140691500),  INT32_C(   120379402),  INT32_C(  1927197136) },
      { -INT32_C(  2087053895), -INT32_C(   172684158),  INT32_C(   980294863),  INT32_C(  1755776565), -INT32_C(   474744281),  INT32_C(  1140691500),  INT32_C(   558202704), -INT32_C(   376233168) } },
    { { -INT32_C(  1150538184),  INT32_C(   179315258),  INT32_C(   709107518), -INT32_C(  1347401592),  INT32_C(  1972593474),  INT32_C(   733210331), -INT32_C(  1790117787),  INT32_C(  1417601051) },
      UINT8_C( 14),
      {  INT32_C(   206049258), -INT32_C(   565488961), -INT32_C(   563710570), -INT32_C(  2011163099), -INT32_C(  1889298775),  INT32_C(  1123323465), -INT32_C(  1151497765), -INT32_C(   238440185),
        -INT32_C(  2130832960), -INT32_C(    94484124), -INT32_C(   472267330), -INT32_C(  2073298469), -INT32_C(   653013361),  INT32_C(   924518236), -INT32_C(  1745717104), -INT32_C(   360137942) },
      { -INT32_C(  1150538184), -INT32_C(   565488961), -INT32_C(   563710570), -INT32_C(  2011163099),  INT32_C(  1972593474),  INT32_C(   733210331), -INT32_C(  1790117787),  INT32_C(  1417601051) },
      { -INT32_C(  1150538184), -INT32_C(    94484124), -INT32_C(   472267330), -INT32_C(  2073298469),  INT32_C(  1972593474),  INT32_C(   733210331), -INT32_C(  1790117787),  INT32_C(  1417601051) } },
    { {  INT32_C(   829064909), -INT32_C(  1926510129),  INT32_C(  1768948878), -INT32_C(  1913791490),  INT32_C(   107348137), -INT32_C(  1757576953),  INT32_C(   607006970), -INT32_C(  1206995221) },
      UINT8_C( 61),
      {  INT32_C(  1108142457),  INT32_C(   433101333), -INT32_C(   434685686),  INT32_C(   646948134), -INT32_C(  1926392565),  INT32_C(    59229395), -INT32_C(  1460753422),  INT32_C(   853911481),
        -INT32_C(  1519062384), -INT32_C(  1765849972), -INT32_C(  1552099715), -INT32_C(  2016867205),  INT32_C(  1947530913), -INT32_C(  1401447494), -INT32_C(    11180474), -INT32_C(  1657652723) },
      {  INT32_C(  1108142457), -INT32_C(  1926510129), -INT32_C(   434685686),  INT32_C(   646948134), -INT32_C(  1926392565),  INT32_C(    59229395),  INT32_C(   607006970), -INT32_C(  1206995221) },
      { -INT32_C(  1519062384), -INT32_C(  1926510129), -INT32_C(  1552099715), -INT32_C(  2016867205),  INT32_C(  1947530913), -INT32_C(  1401447494),  INT32_C(   607006970), -INT32_C(  1206995221) } },
    { { -INT32_C(  1203526100),  INT32_C(  1750008555),  INT32_C(  1410124760),  INT32_C(  2044450263), -INT32_C(  2031226932), -INT32_C(   801938038), -INT32_C(   657487669), -INT32_C(   277544510) },
      UINT8_C(168),
      { -INT32_C(  1164728392), -INT32_C(  1030489098), -INT32_C(   577116409), -INT32_C(  1314319678),  INT32_C(  1698377472), -INT32_C(   365950110), -INT32_C(   592639782),  INT32_C(   914660477),
         INT32_C(   988813123),  INT32_C(   452756242),  INT32_C(   754423146), -INT32_C(  1461870424),  INT32_C(   839719119), -INT32_C(    31703773), -INT32_C(  1025849019), -INT32_C(  1443340699) },
      { -INT32_C(  1203526100),  INT32_C(  1750008555),  INT32_C(  1410124760), -INT32_C(  1314319678), -INT32_C(  2031226932), -INT32_C(   365950110), -INT32_C(   657487669),  INT32_C(   914660477) },
      { -INT32_C(  1203526100),  INT32_C(  1750008555),  INT32_C(  1410124760), -INT32_C(  1461870424), -INT32_C(  2031226932), -INT32_C(    31703773), -INT32_C(   657487669), -INT32_C(  1443340699) } },
    { { -INT32_C(  2015106699), -INT32_C(   677257364),  INT32_C(   469997684),  INT32_C(   147120440),  INT32_C(   490394105),  INT32_C(  1394300430), -INT32_C(  2062093024), -INT32_C(   936505773) },
      UINT8_C(247),
      { -INT32_C(   261861615), -INT32_C(  1989854223),  INT32_C(   532840766),  INT32_C(   387566150),  INT32_C(  1512453636),  INT32_C(  1182431569), -INT32_C(  1634140017),  INT32_C(  1083531566),
        -INT32_C(  1590625872),  INT32_C(  1932236084),  INT32_C(  1569910039), -INT32_C(  1149981769),  INT32_C(   840276705), -INT32_C(  1552379884), -INT32_C(  1103031920),  INT32_C(   603903858) },
      { -INT32_C(   261861615), -INT32_C(  1989854223),  INT32_C(   532840766),  INT32_C(   147120440),  INT32_C(  1512453636),  INT32_C(  1182431569), -INT32_C(  1634140017),  INT32_C(  1083531566) },
      { -INT32_C(  1590625872),  INT32_C(  1932236084),  INT32_C(  1569910039),  INT32_C(   147120440),  INT32_C(   840276705), -INT32_C(  1552379884), -INT32_C(  1103031920),  INT32_C(   603903858) } },
    { {  INT32_C(    96743376), -INT32_C(   612831292), -INT32_C(  1825043748), -INT32_C(  1756451402),  INT32_C(  1539990599), -INT32_C(  2063645964), -INT32_C(   968736684), -INT32_C(   387366633) },
      UINT8_C(112),
      { -INT32_C(  1657475666),  INT32_C(  1870270565), -INT32_C(   182121144), -INT32_C(  1069695652), -INT32_C(   910911353), -INT32_C(   669173609), -INT32_C(  1141906310),  INT32_C(  2066470861),
         INT32_C(   689529028), -INT32_C(  1197894800), -INT32_C(    55656800),  INT32_C(    45935483),  INT32_C(   466382979),  INT32_C(   603187624), -INT32_C(  1679891763),  INT32_C(  2115373754) },
      {  INT32_C(    96743376), -INT32_C(   612831292), -INT32_C(  1825043748), -INT32_C(  1756451402), -INT32_C(   910911353), -INT32_C(   669173609), -INT32_C(  1141906310), -INT32_C(   387366633) },
      {  INT32_C(    96743376), -INT32_C(   612831292), -INT32_C(  1825043748), -INT32_C(  1756451402),  INT32_C(   466382979),  INT32_C(   603187624), -INT32_C(  1679891763), -INT32_C(   387366633) } },
    { { -INT32_C(   626512022),  INT32_C(  1670529474),  INT32_C(  2069840127), -INT32_C(  1350755285),  INT32_C(   902449548),  INT32_C(     5815603),  INT32_C(  1503344287), -INT32_C(  1428639168) },
      UINT8_C(225),
      { -INT32_C(  1046182784),  INT32_C(  1472202519), -INT32_C(  2105328794),  INT32_C(    34550457), -INT32_C(  1187691268), -INT32_C(   765970788), -INT32_C(  2095861039),  INT32_C(   174439818),
         INT32_C(  1506478402),  INT32_C(  1991281424), -INT32_C(  2131151929),  INT32_C(  1635911781), -INT32_C(   400902068), -INT32_C(  1078299666), -INT32_C(  1354511067), -INT32_C(   843470709) },
      { -INT32_C(  1046182784),  INT32_C(  1670529474),  INT32_C(  2069840127), -INT32_C(  1350755285),  INT32_C(   902449548), -INT32_C(   765970788), -INT32_C(  2095861039),  INT32_C(   174439818) },
      {  INT32_C(  1506478402),  INT32_C(  1670529474),  INT32_C(  2069840127), -INT32_C(  1350755285),  INT32_C(   902449548), -INT32_C(  1078299666), -INT32_C(  1354511067), -INT32_C(   843470709) } },
    { { -INT32_C(  1054440271), -INT32_C(   700983793),  INT32_C(  1867919370), -INT32_C(  2066622152),  INT32_C(  2137844625), -INT32_C(  2093078690),  INT32_C(  2134016500), -INT32_C(   632493271) },
      UINT8_C(111),
      {  INT32_C(  1233099634),  INT32_C(    55793107), -INT32_C(  2059615572),  INT32_C(  2132197523),  INT32_C(  1423807789),  INT32_C(  1430806995),  INT32_C(  2139015315), -INT32_C(  2014422764),
        -INT32_C(   942641676),  INT32_C(  1875583939),  INT32_C(  2046035941), -INT32_C(   185070905),  INT32_C(  1917375903), -INT32_C(   892890826),  INT32_C(  1833518681), -INT32_C(  1812711521) },
      {  INT32_C(  1233099634),  INT32_C(    55793107), -INT32_C(  2059615572),  INT32_C(  2132197523),  INT32_C(  2137844625),  INT32_C(  1430806995),  INT32_C(  2139015315), -INT32_C(   632493271) },
      { -INT32_C(   942641676),  INT32_C(  1875583939),  INT32_C(  2046035941), -INT32_C(   185070905),  INT32_C(  2137844625), -INT32_C(   892890826),  INT32_C(  1833518681), -INT32_C(   632493271) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    r = easysimd_mm512_mask_extracti32x8_epi32(src, k, a, 0);
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r0));

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extracti32x8_epi32(src, k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extracti32x8_epi32");
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r1));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m256i r0 = easysimd_mm512_mask_extracti32x8_epi32(src, k, a, 0);
    easysimd__m256i r1 = easysimd_mm512_mask_extracti32x8_epi32(src, k, a, 1);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_extracti32x8_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const int32_t a[16];
    const int32_t r0[8];
    const int32_t r1[8];
  } test_vec[8] = {
    { UINT8_C(165),
      { -INT32_C(   412591164),  INT32_C(   768464678), -INT32_C(   705411381),  INT32_C(   326428990), -INT32_C(  1035278799), -INT32_C(   199551826), -INT32_C(  1802270371),  INT32_C(  1111041661),
        -INT32_C(  1490443903),  INT32_C(  1138030200),  INT32_C(  2048444732), -INT32_C(   477197134),  INT32_C(   564582515),  INT32_C(  1226162668), -INT32_C(   924997558),  INT32_C(  1342838734) },
      { -INT32_C(   412591164),  INT32_C(           0), -INT32_C(   705411381),  INT32_C(           0),  INT32_C(           0), -INT32_C(   199551826),  INT32_C(           0),  INT32_C(  1111041661) },
      { -INT32_C(  1490443903),  INT32_C(           0),  INT32_C(  2048444732),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1226162668),  INT32_C(           0),  INT32_C(  1342838734) } },
    { UINT8_C(184),
      {  INT32_C(   707917619), -INT32_C(  1788447540),  INT32_C(   424141197),  INT32_C(  1200368239), -INT32_C(  1842106928),  INT32_C(  1792834754),  INT32_C(  1882825817), -INT32_C(   500594258),
         INT32_C(  1275878016),  INT32_C(  1541501646), -INT32_C(  1032509357),  INT32_C(   587792723),  INT32_C(  1907702959),  INT32_C(   316445368), -INT32_C(   444459722),  INT32_C(   516402078) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1200368239), -INT32_C(  1842106928),  INT32_C(  1792834754),  INT32_C(           0), -INT32_C(   500594258) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   587792723),  INT32_C(  1907702959),  INT32_C(   316445368),  INT32_C(           0),  INT32_C(   516402078) } },
    { UINT8_C(  5),
      {  INT32_C(  1171549139),  INT32_C(  1972973388), -INT32_C(  1496818780), -INT32_C(  1588204699),  INT32_C(   861587105), -INT32_C(  1217827678), -INT32_C(  1705685266), -INT32_C(   392203243),
         INT32_C(   724464607),  INT32_C(  1201719203), -INT32_C(  2014484446), -INT32_C(   181845420), -INT32_C(  1423408375), -INT32_C(   580677137), -INT32_C(   176703264),  INT32_C(   199104300) },
      {  INT32_C(  1171549139),  INT32_C(           0), -INT32_C(  1496818780),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(   724464607),  INT32_C(           0), -INT32_C(  2014484446),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(138),
      { -INT32_C(   752011508),  INT32_C(  1089828311), -INT32_C(  1516995230),  INT32_C(   699304358),  INT32_C(  1108892081),  INT32_C(  1965225404), -INT32_C(  2069817235),  INT32_C(    34516470),
        -INT32_C(  1143653148),  INT32_C(   335268529), -INT32_C(   306671801), -INT32_C(   921278952),  INT32_C(  2081173184), -INT32_C(  1846464988),  INT32_C(  1008046918),  INT32_C(   608052032) },
      {  INT32_C(           0),  INT32_C(  1089828311),  INT32_C(           0),  INT32_C(   699304358),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(    34516470) },
      {  INT32_C(           0),  INT32_C(   335268529),  INT32_C(           0), -INT32_C(   921278952),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   608052032) } },
    { UINT8_C( 95),
      { -INT32_C(   569319661),  INT32_C(  1780819163),  INT32_C(  1132663772),  INT32_C(  1476611113), -INT32_C(  2021884072),  INT32_C(    80547185),  INT32_C(  1162086946),  INT32_C(  1554344008),
         INT32_C(   574272839), -INT32_C(  1232248871), -INT32_C(  1661398926), -INT32_C(  1242235812), -INT32_C(   331583365), -INT32_C(  1611658883),  INT32_C(  1558459411), -INT32_C(   457668196) },
      { -INT32_C(   569319661),  INT32_C(  1780819163),  INT32_C(  1132663772),  INT32_C(  1476611113), -INT32_C(  2021884072),  INT32_C(           0),  INT32_C(  1162086946),  INT32_C(           0) },
      {  INT32_C(   574272839), -INT32_C(  1232248871), -INT32_C(  1661398926), -INT32_C(  1242235812), -INT32_C(   331583365),  INT32_C(           0),  INT32_C(  1558459411),  INT32_C(           0) } },
    { UINT8_C( 63),
      {  INT32_C(  1360529138), -INT32_C(  1547383149), -INT32_C(  1023385401), -INT32_C(  1002457772), -INT32_C(    96392207),  INT32_C(  1343152156),  INT32_C(  1324182212),  INT32_C(   344838434),
         INT32_C(  1801823703),  INT32_C(   990783860), -INT32_C(   570487159), -INT32_C(  1264501053), -INT32_C(  2052070807), -INT32_C(  2015969854),  INT32_C(  1238745895),  INT32_C(  1801282196) },
      {  INT32_C(  1360529138), -INT32_C(  1547383149), -INT32_C(  1023385401), -INT32_C(  1002457772), -INT32_C(    96392207),  INT32_C(  1343152156),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(  1801823703),  INT32_C(   990783860), -INT32_C(   570487159), -INT32_C(  1264501053), -INT32_C(  2052070807), -INT32_C(  2015969854),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(  7),
      { -INT32_C(   327428414), -INT32_C(   210389019), -INT32_C(   189312074), -INT32_C(   681743372), -INT32_C(   677780710),  INT32_C(  2097029305),  INT32_C(  1460684789),  INT32_C(  1734310820),
         INT32_C(   928242258),  INT32_C(  1193986193),  INT32_C(   272359707),  INT32_C(  1743231309),  INT32_C(   893354108), -INT32_C(  1766769247),  INT32_C(   703512965), -INT32_C(  1903145668) },
      { -INT32_C(   327428414), -INT32_C(   210389019), -INT32_C(   189312074),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(   928242258),  INT32_C(  1193986193),  INT32_C(   272359707),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 39),
      { -INT32_C(  1397111325), -INT32_C(   775487248), -INT32_C(   719399108),  INT32_C(  1062307518),  INT32_C(    48268997), -INT32_C(   125340105), -INT32_C(  1321946780),  INT32_C(   635028289),
         INT32_C(  2027000456), -INT32_C(   833972078),  INT32_C(   782461040), -INT32_C(  1284639506), -INT32_C(  1313452678),  INT32_C(   682180036),  INT32_C(   819650286),  INT32_C(   693482401) },
      { -INT32_C(  1397111325), -INT32_C(   775487248), -INT32_C(   719399108),  INT32_C(           0),  INT32_C(           0), -INT32_C(   125340105),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(  2027000456), -INT32_C(   833972078),  INT32_C(   782461040),  INT32_C(           0),  INT32_C(           0),  INT32_C(   682180036),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    r = easysimd_mm512_maskz_extracti32x8_epi32(k, a, 0);
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r0));

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extracti32x8_epi32(k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extracti32x8_epi32");
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r1));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m256i r0 = easysimd_mm512_maskz_extracti32x8_epi32(k, a, 0);
    easysimd__m256i r1 = easysimd_mm512_maskz_extracti32x8_epi32(k, a, 1);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_extracti64x2_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t a[8];
    const int64_t r0[2];
    const int64_t r1[2];
  } test_vec[] = {
    { { -INT64_C( 1926534861649420577), -INT64_C( 8024347015817134063), -INT64_C( 2625657214745679846),  INT64_C( 8519367488962424222),
        -INT64_C( 2854193889930496579),  INT64_C(  672974713772778974), -INT64_C( 2102823388506064310),  INT64_C(  882730054631321395) },
      { -INT64_C( 1926534861649420577), -INT64_C( 8024347015817134063) },
      { -INT64_C( 2625657214745679846),  INT64_C( 8519367488962424222) } },
    { { -INT64_C( 7571696973635724725),  INT64_C( 6344359169496918802),  INT64_C( 4202175143617675563), -INT64_C( 8076849394155770481),
         INT64_C( 8317460637703893791), -INT64_C( 6548655543346149191),  INT64_C( 2054317137020153153), -INT64_C( 1556024681575119212) },
      { -INT64_C( 7571696973635724725),  INT64_C( 6344359169496918802) },
      {  INT64_C( 4202175143617675563), -INT64_C( 8076849394155770481) } },
    { {  INT64_C( 5848819774523509334),  INT64_C( 6995838584114094326),  INT64_C( 2687572079731682211),  INT64_C( 9180699064998380979),
         INT64_C( 5573311565282027887),  INT64_C( 5225253798380728065),  INT64_C(  916135463171403263), -INT64_C( 8668881621970406646) },
      {  INT64_C( 5848819774523509334),  INT64_C( 6995838584114094326) },
      {  INT64_C( 2687572079731682211),  INT64_C( 9180699064998380979) } },
    { {  INT64_C( 5458615951502010395),  INT64_C( 5856068555173035606), -INT64_C( 8393980173164102628),  INT64_C( 2011824801231736139),
        -INT64_C( 5266030261873837737),  INT64_C( 7236991337020610763),  INT64_C( 3656462095548567730),  INT64_C( 4721041153184922473) },
      {  INT64_C( 5458615951502010395),  INT64_C( 5856068555173035606) },
      { -INT64_C( 8393980173164102628),  INT64_C( 2011824801231736139) } },
    { {  INT64_C( 3031148358650696186),  INT64_C( 7625076543043715621),  INT64_C( 8151614315655113478),  INT64_C( 3603546245401589462),
        -INT64_C( 2884100699353979793), -INT64_C(  108062240082214561),  INT64_C( 5564211048211029184), -INT64_C( 2293441132332510525) },
      {  INT64_C( 3031148358650696186),  INT64_C( 7625076543043715621) },
      {  INT64_C( 8151614315655113478),  INT64_C( 3603546245401589462) } },
    { { -INT64_C( 5369232665143756301), -INT64_C( 7647591735256956696), -INT64_C( 7409382536828368280),  INT64_C( 4393472856677431719),
         INT64_C( 3928367368050419321),  INT64_C(  254440854588379165),  INT64_C( 2318806773510890686), -INT64_C(  230474664290762168) },
      { -INT64_C( 5369232665143756301), -INT64_C( 7647591735256956696) },
      { -INT64_C( 7409382536828368280),  INT64_C( 4393472856677431719) } },
    { {  INT64_C( 3052121185577643991),  INT64_C( 2684044661938221728),  INT64_C( 8656455251667853960),  INT64_C( 7001645559178681003),
        -INT64_C( 1748273758297721210),  INT64_C(  423166281641089364),  INT64_C( 3255583022971787653), -INT64_C( 7751393088251414655) },
      {  INT64_C( 3052121185577643991),  INT64_C( 2684044661938221728) },
      {  INT64_C( 8656455251667853960),  INT64_C( 7001645559178681003) } },
    { {  INT64_C( 3074520207170466940), -INT64_C( 5314594794546868548), -INT64_C( 1365547894116236195),  INT64_C( 7598812742643443665),
         INT64_C( 4277248108597648219),  INT64_C( 7028908472489230244),  INT64_C( 7012471377328404471),  INT64_C(  867705652741491380) },
      {  INT64_C( 3074520207170466940), -INT64_C( 5314594794546868548) },
      { -INT64_C( 1365547894116236195),  INT64_C( 7598812742643443665) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i r;
    r = easysimd_mm512_extracti64x2_epi64(a, 0);
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r0));

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extracti64x2_epi64(a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extracti64x2_epi64");
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r1));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m128i r0 = easysimd_mm512_extracti64x2_epi64(a, 0);
    easysimd__m128i r1 = easysimd_mm512_extracti64x2_epi64(a, 1);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_extracti64x2_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[8];
    const int64_t r0[2];
    const int64_t r1[2];
  } test_vec[] = {
    { {  INT64_C( 7562677652854690283), -INT64_C(   83604072676222378) },
      UINT8_C(166),
      { -INT64_C( 6282576186308280788),  INT64_C( 5452802495545783474), -INT64_C( 1959922037647714962), -INT64_C( 4471673566530715824),
         INT64_C( 3931727707926017706), -INT64_C( 7257934331564790015),  INT64_C( 1740604732702053369), -INT64_C( 5984346016216541795) },
      {  INT64_C( 7562677652854690283),  INT64_C( 5452802495545783474) },
      {  INT64_C( 7562677652854690283), -INT64_C( 4471673566530715824) } },
    { { -INT64_C( 4628251545965400053),  INT64_C( 6475051877183175956) },
      UINT8_C(212),
      {  INT64_C( 6442879499269610734), -INT64_C(  110329882569106664), -INT64_C( 8515044876113672293), -INT64_C( 2302953551864130251),
         INT64_C( 8357859966818494176), -INT64_C(  343302305486972752), -INT64_C( 6217904886185723809), -INT64_C( 4433817049191440475) },
      { -INT64_C( 4628251545965400053),  INT64_C( 6475051877183175956) },
      { -INT64_C( 4628251545965400053),  INT64_C( 6475051877183175956) } },
    { { -INT64_C( 1702813413469183088), -INT64_C(  786050113584349508) },
      UINT8_C( 91),
      {  INT64_C( 8433123164427330002),  INT64_C( 4238478886491889495),  INT64_C( 6530601114313932160), -INT64_C( 6466915262838751870),
        -INT64_C( 2354800379934732537),  INT64_C( 4900746271500839600),  INT64_C( 5919641758352373168),  INT64_C(  360702479013713799) },
      {  INT64_C( 8433123164427330002),  INT64_C( 4238478886491889495) },
      {  INT64_C( 6530601114313932160), -INT64_C( 6466915262838751870) } },
    { { -INT64_C( 3679945455168189817), -INT64_C( 5627083900127605025) },
      UINT8_C( 10),
      { -INT64_C( 4944339259107869423), -INT64_C( 3934810447441969819),  INT64_C( 8121872912289657761),  INT64_C( 6044994270349318724),
         INT64_C( 5908037630577000351),  INT64_C( 1645101894805141069), -INT64_C( 5919800937539585786),  INT64_C(  762070870344026622) },
      { -INT64_C( 3679945455168189817), -INT64_C( 3934810447441969819) },
      { -INT64_C( 3679945455168189817),  INT64_C( 6044994270349318724) } },
    { { -INT64_C( 7032289106012094166),  INT64_C( 3498309727092839936) },
      UINT8_C( 55),
      { -INT64_C( 5108403108881328007), -INT64_C( 5806777168727614507),  INT64_C( 9206939735786815610),  INT64_C( 4683137535783221832),
        -INT64_C( 4568486293002150544),  INT64_C( 7382080300035128788),  INT64_C( 6436469677302117355),  INT64_C( 4913458890053516074) },
      { -INT64_C( 5108403108881328007), -INT64_C( 5806777168727614507) },
      {  INT64_C( 9206939735786815610),  INT64_C( 4683137535783221832) } },
    { { -INT64_C( 2637211591734161734), -INT64_C( 4842194433269143941) },
      UINT8_C(132),
      { -INT64_C( 2516966958896410633), -INT64_C( 4546959752442150642),  INT64_C( 6292270784686033433),  INT64_C( 9139743759350151489),
         INT64_C( 7580251253169810291), -INT64_C( 1123232664882191497),  INT64_C( 1462292001658974211), -INT64_C( 3830245137202121773) },
      { -INT64_C( 2637211591734161734), -INT64_C( 4842194433269143941) },
      { -INT64_C( 2637211591734161734), -INT64_C( 4842194433269143941) } },
    { {  INT64_C( 3056627700889304487),  INT64_C( 8934364821610364725) },
      UINT8_C(110),
      {  INT64_C( 3292788538659139525),  INT64_C( 2422461494395669691),  INT64_C(  790380685588471919), -INT64_C( 4143595326608898861),
        -INT64_C( 3506799891233521811), -INT64_C( 7464042204407342492),  INT64_C( 1232667150525218565),  INT64_C(  656576239009635924) },
      {  INT64_C( 3056627700889304487),  INT64_C( 2422461494395669691) },
      {  INT64_C( 3056627700889304487), -INT64_C( 4143595326608898861) } },
    { {  INT64_C( 9098264219906851423),  INT64_C( 8368306512258861665) },
      UINT8_C(113),
      {  INT64_C( 4548847424800523643),  INT64_C( 6188478685173741281), -INT64_C( 6701657955748588266),  INT64_C( 2691048845933119669),
        -INT64_C( 6274901007001385205), -INT64_C( 4508090397474765176), -INT64_C( 9035526187683750048),  INT64_C( 4518167274064555236) },
      {  INT64_C( 4548847424800523643),  INT64_C( 8368306512258861665) },
      { -INT64_C( 6701657955748588266),  INT64_C( 8368306512258861665) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i r;
    r = easysimd_mm512_mask_extracti64x2_epi64(src, k, a, 0);
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r0));

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extracti64x2_epi64(src, k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extracti64x2_epi64");
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r1));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m128i r0 = easysimd_mm512_mask_extracti64x2_epi64(src, k, a, 0);
    easysimd__m128i r1 = easysimd_mm512_mask_extracti64x2_epi64(src, k, a, 1);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_extracti64x2_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const int64_t a[8];
    const int64_t r0[2];
    const int64_t r1[2];
  } test_vec[] = {
    { UINT8_C(200),
      { -INT64_C( 7187965084917831003), -INT64_C( 6067464928186901663),  INT64_C(  911691268075603302), -INT64_C( 8220056783863991197),
         INT64_C( 1502658227236797287), -INT64_C( 8046440313583033890),  INT64_C( 3062159431670418706), -INT64_C( 6727469397204615890) },
      {  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(181),
      { -INT64_C( 2645447269819097113),  INT64_C( 1538987835372467196),  INT64_C(  653028868967157224), -INT64_C( 5745496045756668332),
         INT64_C( 2475118487192382149), -INT64_C( 2543470610602041770), -INT64_C( 1592076800169701485), -INT64_C( 3067591190108492812) },
      { -INT64_C( 2645447269819097113),  INT64_C(                   0) },
      {  INT64_C(  653028868967157224),  INT64_C(                   0) } },
    { UINT8_C(101),
      {  INT64_C(  883459439410762888),  INT64_C( 2554064995633651736),  INT64_C( 7088281222408392603),  INT64_C( 1241102929129249562),
        -INT64_C( 8550859661199489892), -INT64_C( 3614463938648386273),  INT64_C( 5530864761454366504),  INT64_C( 1678253296434820016) },
      {  INT64_C(  883459439410762888),  INT64_C(                   0) },
      {  INT64_C( 7088281222408392603),  INT64_C(                   0) } },
    { UINT8_C( 30),
      {  INT64_C(  910881296351256902), -INT64_C( 4791536239146805375), -INT64_C( 3060693330469855167),  INT64_C( 5752337014964928347),
        -INT64_C(  104857685902203703),  INT64_C( 1637860923309345231),  INT64_C( 4553265548391918010), -INT64_C( 5581708259595288271) },
      {  INT64_C(                   0), -INT64_C( 4791536239146805375) },
      {  INT64_C(                   0),  INT64_C( 5752337014964928347) } },
    { UINT8_C( 45),
      { -INT64_C( 8744114993677848226),  INT64_C(  474128508690080513), -INT64_C( 6845426212699180963),  INT64_C( 2383596914886720030),
         INT64_C( 7616042615111729006), -INT64_C( 2691788643223008191), -INT64_C( 2987388795459367224), -INT64_C( 4627217368974813981) },
      { -INT64_C( 8744114993677848226),  INT64_C(                   0) },
      { -INT64_C( 6845426212699180963),  INT64_C(                   0) } },
    { UINT8_C(145),
      { -INT64_C( 6296205334587774272),  INT64_C( 8017815680070270373), -INT64_C( 3731516273118990679),  INT64_C( 7358279974066707898),
         INT64_C( 2164653039120881350), -INT64_C( 2741600153950802014), -INT64_C( 3345222243940820465),  INT64_C( 6074365846801527104) },
      { -INT64_C( 6296205334587774272),  INT64_C(                   0) },
      { -INT64_C( 3731516273118990679),  INT64_C(                   0) } },
    { UINT8_C(146),
      { -INT64_C( 1218178892124473839), -INT64_C( 6290129487990383676), -INT64_C( 4627852075500340444),  INT64_C(  791685727112768751),
        -INT64_C(  934892544607647913),  INT64_C( 4713945125675869492), -INT64_C( 6154856378318443833), -INT64_C( 2720717869672963525) },
      {  INT64_C(                   0), -INT64_C( 6290129487990383676) },
      {  INT64_C(                   0),  INT64_C(  791685727112768751) } },
    { UINT8_C( 24),
      {  INT64_C( 2128861186665017791), -INT64_C( 2024230897951486252),  INT64_C( 3895131352103632452),  INT64_C( 4578072300590023262),
        -INT64_C( 2262176950781489019), -INT64_C( 7562132759712311696),  INT64_C( 6066481695931673766),  INT64_C(  250596830013090234) },
      {  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i r;
    r = easysimd_mm512_maskz_extracti64x2_epi64(k, a, 0);
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r0));

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extracti64x2_epi64(k, a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extracti64x2_epi64");
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r1));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m128i r0 = easysimd_mm512_maskz_extracti64x2_epi64(k, a, 0);
    easysimd__m128i r1 = easysimd_mm512_maskz_extracti64x2_epi64(k, a, 1);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_extracti64x4_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m256i r0;
    easysimd__m256i r1;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 4022993628330696330), INT64_C( -564047204985781920),
                            INT64_C( 4934063986128071877), INT64_C( 8258886799903261224),
                            INT64_C( 5245738308211416456), INT64_C( 8690736315259258337),
                            INT64_C(-5183161890921602420), INT64_C(-2495336383094170141)),
      easysimd_mm256_set_epi64x(INT64_C( 5245738308211416456), INT64_C( 8690736315259258337),
                             INT64_C(-5183161890921602420), INT64_C(-2495336383094170141)),
      easysimd_mm256_set_epi64x(INT64_C( 4022993628330696330), INT64_C( -564047204985781920),
                             INT64_C( 4934063986128071877), INT64_C( 8258886799903261224)) },
    { easysimd_mm512_set_epi64(INT64_C(-8115609027568940125), INT64_C( 3504612124823893047),
                            INT64_C(-7514888466798804666), INT64_C( 9113506312589344178),
                            INT64_C(-5948029684411535130), INT64_C(-4862378680423071053),
                            INT64_C( 7355766231574189317), INT64_C( -310150959079746096)),
      easysimd_mm256_set_epi64x(INT64_C(-5948029684411535130), INT64_C(-4862378680423071053),
                             INT64_C( 7355766231574189317), INT64_C( -310150959079746096)),
      easysimd_mm256_set_epi64x(INT64_C(-8115609027568940125), INT64_C( 3504612124823893047),
                             INT64_C(-7514888466798804666), INT64_C( 9113506312589344178)) },
    { easysimd_mm512_set_epi64(INT64_C(-7443148953768886026), INT64_C( 8448077846545567514),
                            INT64_C(-7304267332935478206), INT64_C(-1528489088828046422),
                            INT64_C( 1295774678670654457), INT64_C(-8851107363323835123),
                            INT64_C( 2369486750103851747), INT64_C(-5139586436110975467)),
      easysimd_mm256_set_epi64x(INT64_C( 1295774678670654457), INT64_C(-8851107363323835123),
                             INT64_C( 2369486750103851747), INT64_C(-5139586436110975467)),
      easysimd_mm256_set_epi64x(INT64_C(-7443148953768886026), INT64_C( 8448077846545567514),
                             INT64_C(-7304267332935478206), INT64_C(-1528489088828046422)) },
    { easysimd_mm512_set_epi64(INT64_C(-3094219001013742557), INT64_C(-3379016320921474793),
                            INT64_C( 7772273849745001049), INT64_C(-4229480058937372017),
                            INT64_C( 8318730560275653847), INT64_C(-2664412856586094061),
                            INT64_C( 2083707536546841162), INT64_C( 5404230241318444880)),
      easysimd_mm256_set_epi64x(INT64_C( 8318730560275653847), INT64_C(-2664412856586094061),
                             INT64_C( 2083707536546841162), INT64_C( 5404230241318444880)),
      easysimd_mm256_set_epi64x(INT64_C(-3094219001013742557), INT64_C(-3379016320921474793),
                             INT64_C( 7772273849745001049), INT64_C(-4229480058937372017)) },
    { easysimd_mm512_set_epi64(INT64_C(-7922172549839933132), INT64_C(   90826243433254935),
                            INT64_C( 1037923706586637130), INT64_C( 5568688997300093349),
                            INT64_C( 9199350188047982974), INT64_C(-7693432910203882071),
                            INT64_C(-5762072963977532348), INT64_C( 4491924425059371454)),
      easysimd_mm256_set_epi64x(INT64_C( 9199350188047982974), INT64_C(-7693432910203882071),
                             INT64_C(-5762072963977532348), INT64_C( 4491924425059371454)),
      easysimd_mm256_set_epi64x(INT64_C(-7922172549839933132), INT64_C(   90826243433254935),
                             INT64_C( 1037923706586637130), INT64_C( 5568688997300093349)) },
    { easysimd_mm512_set_epi64(INT64_C( 7563354526679147255), INT64_C(-6952412028107066884),
                            INT64_C(-3077616107881632928), INT64_C(-2220298267656761827),
                            INT64_C( 2522518958303333112), INT64_C(-1668307566098600867),
                            INT64_C( 8306832211054389426), INT64_C(-4135341282024622606)),
      easysimd_mm256_set_epi64x(INT64_C( 2522518958303333112), INT64_C(-1668307566098600867),
                             INT64_C( 8306832211054389426), INT64_C(-4135341282024622606)),
      easysimd_mm256_set_epi64x(INT64_C( 7563354526679147255), INT64_C(-6952412028107066884),
                             INT64_C(-3077616107881632928), INT64_C(-2220298267656761827)) },
    { easysimd_mm512_set_epi64(INT64_C(-6876215301736363293), INT64_C(-2253243373865166954),
                            INT64_C( 5866706473820467911), INT64_C(-1945184283153250111),
                            INT64_C(-6043663531296462836), INT64_C(-3201199251206898425),
                            INT64_C(-7517867743898200758), INT64_C( 5023666877462679332)),
      easysimd_mm256_set_epi64x(INT64_C(-6043663531296462836), INT64_C(-3201199251206898425),
                             INT64_C(-7517867743898200758), INT64_C( 5023666877462679332)),
      easysimd_mm256_set_epi64x(INT64_C(-6876215301736363293), INT64_C(-2253243373865166954),
                             INT64_C( 5866706473820467911), INT64_C(-1945184283153250111)) },
    { easysimd_mm512_set_epi64(INT64_C(  135432210503006619), INT64_C(-7059566968128636366),
                            INT64_C(-1295026765047609725), INT64_C( 5447800525707046939),
                            INT64_C( 1419500527032411112), INT64_C( 5424087511148175828),
                            INT64_C(-4780701435803039630), INT64_C( 6069825193561024149)),
      easysimd_mm256_set_epi64x(INT64_C( 1419500527032411112), INT64_C( 5424087511148175828),
                             INT64_C(-4780701435803039630), INT64_C( 6069825193561024149)),
      easysimd_mm256_set_epi64x(INT64_C(  135432210503006619), INT64_C(-7059566968128636366),
                             INT64_C(-1295026765047609725), INT64_C( 5447800525707046939)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i r;
    r = easysimd_mm512_extracti64x4_epi64(test_vec[i].a, 0);
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r0);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_extracti64x4_epi64(test_vec[i].a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_extracti64x4_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_extracti64x4_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m256i r0;
    easysimd__m256i r1;
  } test_vec[8] = {
    { easysimd_mm256_set_epi64x(INT64_C( 7940316924786767481), INT64_C( 6743600876828439814),
                             INT64_C( 8561828216572109007), INT64_C(-8315109086095518889)),
      UINT8_C( 21),
      easysimd_mm512_set_epi64(INT64_C(-8586603972668500699), INT64_C( 7848938818320954984),
                            INT64_C(-7002500096438875648), INT64_C( 1477571573764517782),
                            INT64_C( 4086853108457730066), INT64_C(-7738570880062900818),
                            INT64_C(-5609503674875201288), INT64_C( 3966155248134972346)),
      easysimd_mm256_set_epi64x(INT64_C( 7940316924786767481), INT64_C(-7738570880062900818),
                             INT64_C( 8561828216572109007), INT64_C( 3966155248134972346)),
      easysimd_mm256_set_epi64x(INT64_C( 7940316924786767481), INT64_C( 7848938818320954984),
                             INT64_C( 8561828216572109007), INT64_C( 1477571573764517782)) },
    { easysimd_mm256_set_epi64x(INT64_C(-4767707706458520415), INT64_C(-7083171014951853588),
                             INT64_C(-2076988212358998594), INT64_C( 3698505898575972461)),
      UINT8_C(157),
      easysimd_mm512_set_epi64(INT64_C( 1516975282358243755), INT64_C(-3949523894747321163),
                            INT64_C(-2989078375862773056), INT64_C(-4700117291684372957),
                            INT64_C(-8784413934425613521), INT64_C(  961867877660623168),
                            INT64_C( 2426510480636680010), INT64_C(-6612602987193650875)),
      easysimd_mm256_set_epi64x(INT64_C(-8784413934425613521), INT64_C(  961867877660623168),
                             INT64_C(-2076988212358998594), INT64_C(-6612602987193650875)),
      easysimd_mm256_set_epi64x(INT64_C( 1516975282358243755), INT64_C(-3949523894747321163),
                             INT64_C(-2076988212358998594), INT64_C(-4700117291684372957)) },
    { easysimd_mm256_set_epi64x(INT64_C( 4704994953943345443), INT64_C( 8877610218385468208),
                             INT64_C( 5776984527519295337), INT64_C( 6526937450820584225)),
      UINT8_C( 59),
      easysimd_mm512_set_epi64(INT64_C( 3898178537456140670), INT64_C( -393151907512138120),
                            INT64_C(  600214805061827669), INT64_C( 3163434753014979248),
                            INT64_C( 3418472134552461373), INT64_C(-6111840559061041971),
                            INT64_C( 6656352319933975670), INT64_C( 2357435311113502667)),
      easysimd_mm256_set_epi64x(INT64_C( 3418472134552461373), INT64_C( 8877610218385468208),
                             INT64_C( 6656352319933975670), INT64_C( 2357435311113502667)),
      easysimd_mm256_set_epi64x(INT64_C( 3898178537456140670), INT64_C( 8877610218385468208),
                             INT64_C(  600214805061827669), INT64_C( 3163434753014979248)) },
    { easysimd_mm256_set_epi64x(INT64_C(-1414228054518303181), INT64_C(-3038909907977133732),
                             INT64_C(-7842471790453318316), INT64_C(-7256600765093102205)),
      UINT8_C(120),
      easysimd_mm512_set_epi64(INT64_C(-6742373427678247978), INT64_C( 7306080674171373254),
                            INT64_C( 2163582539809461657), INT64_C(-7942896186346970451),
                            INT64_C(-2775611318017263858), INT64_C( 7897714815450887445),
                            INT64_C(-2675129847260557604), INT64_C( 2145911307457407401)),
      easysimd_mm256_set_epi64x(INT64_C(-2775611318017263858), INT64_C(-3038909907977133732),
                             INT64_C(-7842471790453318316), INT64_C(-7256600765093102205)),
      easysimd_mm256_set_epi64x(INT64_C(-6742373427678247978), INT64_C(-3038909907977133732),
                             INT64_C(-7842471790453318316), INT64_C(-7256600765093102205)) },
    { easysimd_mm256_set_epi64x(INT64_C(-8776157931044543560), INT64_C(-1848672680316222475),
                             INT64_C( 1658167909352451238), INT64_C(  377173394815185621)),
      UINT8_C( 22),
      easysimd_mm512_set_epi64(INT64_C( 3509487153133496527), INT64_C( -438383259974317574),
                            INT64_C( 4679882440059701274), INT64_C( 8594666725077939668),
                            INT64_C(-2603844271228681340), INT64_C( 3101493959844818499),
                            INT64_C(-7569746812758465314), INT64_C(  -95001376835728923)),
      easysimd_mm256_set_epi64x(INT64_C(-8776157931044543560), INT64_C( 3101493959844818499),
                             INT64_C(-7569746812758465314), INT64_C(  377173394815185621)),
      easysimd_mm256_set_epi64x(INT64_C(-8776157931044543560), INT64_C( -438383259974317574),
                             INT64_C( 4679882440059701274), INT64_C(  377173394815185621)) },
    { easysimd_mm256_set_epi64x(INT64_C(  -49705421380794940), INT64_C( 3138707856740708121),
                             INT64_C(-4673519228421997952), INT64_C(-1556073591389999033)),
      UINT8_C( 69),
      easysimd_mm512_set_epi64(INT64_C(  463621865143519269), INT64_C(-6632828547466581484),
                            INT64_C(-5710868086811856609), INT64_C( 7040443601555103281),
                            INT64_C( 7268860797756174523), INT64_C(  684742770982669497),
                            INT64_C(  778252790359918942), INT64_C( 8946221359026744959)),
      easysimd_mm256_set_epi64x(INT64_C(  -49705421380794940), INT64_C(  684742770982669497),
                             INT64_C(-4673519228421997952), INT64_C( 8946221359026744959)),
      easysimd_mm256_set_epi64x(INT64_C(  -49705421380794940), INT64_C(-6632828547466581484),
                             INT64_C(-4673519228421997952), INT64_C( 7040443601555103281)) },
    { easysimd_mm256_set_epi64x(INT64_C(-8553143016080257248), INT64_C( 1191725626053358671),
                             INT64_C( 2560034487176803702), INT64_C(-4340183042637127984)),
      UINT8_C( 92),
      easysimd_mm512_set_epi64(INT64_C( -638332694652688568), INT64_C(-8196543121330681227),
                            INT64_C( 7593109912492073141), INT64_C( 6300090425305304893),
                            INT64_C(-3323383259847225301), INT64_C( 2001511420457827007),
                            INT64_C( 8280910196874944184), INT64_C( -368934386460614235)),
      easysimd_mm256_set_epi64x(INT64_C(-3323383259847225301), INT64_C( 2001511420457827007),
                             INT64_C( 2560034487176803702), INT64_C(-4340183042637127984)),
      easysimd_mm256_set_epi64x(INT64_C( -638332694652688568), INT64_C(-8196543121330681227),
                             INT64_C( 2560034487176803702), INT64_C(-4340183042637127984)) },
    { easysimd_mm256_set_epi64x(INT64_C( 7286481320132913626), INT64_C( -777692308098335861),
                             INT64_C( 8727238559278288416), INT64_C(-2736507802934917164)),
      UINT8_C(160),
      easysimd_mm512_set_epi64(INT64_C(-6023807055599376167), INT64_C( 2056379472574346663),
                            INT64_C(-3486865648830471282), INT64_C( 8151787653682140580),
                            INT64_C( -831601358278995789), INT64_C(-2800664419916301039),
                            INT64_C( 3280702275774868225), INT64_C(-4735905134864699368)),
      easysimd_mm256_set_epi64x(INT64_C( 7286481320132913626), INT64_C( -777692308098335861),
                             INT64_C( 8727238559278288416), INT64_C(-2736507802934917164)),
      easysimd_mm256_set_epi64x(INT64_C( 7286481320132913626), INT64_C( -777692308098335861),
                             INT64_C( 8727238559278288416), INT64_C(-2736507802934917164)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i r;
    r = easysimd_mm512_mask_extracti64x4_epi64(test_vec[i].src, test_vec[i].k, test_vec[i].a, 0);
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r0);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_extracti64x4_epi64(test_vec[i].src, test_vec[i].k, test_vec[i].a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_extracti64x4_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_extracti64x4_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m256i r0;
    easysimd__m256i r1;
  } test_vec[8] = {
    { UINT8_C( 87),
      easysimd_mm512_set_epi64(INT64_C( 4086853108457730066), INT64_C(-7738570880062900818),
                            INT64_C(-5609503674875201288), INT64_C( 3966155248134972346),
                            INT64_C( 6201510655001996332), INT64_C( 7633590894537872708),
                            INT64_C(-5593693910291334810), INT64_C(-8884932670315115433)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 7633590894537872708),
                             INT64_C(-5593693910291334810), INT64_C(-8884932670315115433)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-7738570880062900818),
                             INT64_C(-5609503674875201288), INT64_C( 3966155248134972346)) },
    { UINT8_C(150),
      easysimd_mm512_set_epi64(INT64_C( 6143333881204814617), INT64_C( -890089152921238147),
                            INT64_C( 8823574133744668217), INT64_C(-6364969741708969084),
                            INT64_C(-1418976459802394322), INT64_C( 4477375336277674053),
                            INT64_C( 1282248710630285123), INT64_C( 6638105739971879812)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 4477375336277674053),
                             INT64_C( 1282248710630285123), INT64_C(                   0)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( -890089152921238147),
                             INT64_C( 8823574133744668217), INT64_C(                   0)) },
    { UINT8_C( 69),
      easysimd_mm512_set_epi64(INT64_C(-4369965941555109637), INT64_C( 7350668077567080689),
                            INT64_C( 8478919882954811661), INT64_C(-3394066222784588743),
                            INT64_C(-5009306653852991983), INT64_C(-8749971605870264899),
                            INT64_C( 1212032624670585453), INT64_C(  728623586565902494)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-8749971605870264899),
                             INT64_C(                   0), INT64_C(  728623586565902494)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 7350668077567080689),
                             INT64_C(                   0), INT64_C(-3394066222784588743)) },
    { UINT8_C(  7),
      easysimd_mm512_set_epi64(INT64_C(-5506395256633894325), INT64_C( 7391005387705442660),
                            INT64_C(-5091463632259113685), INT64_C(  685405269785004780),
                            INT64_C(-2602517860068074949), INT64_C( 4704994953943345443),
                            INT64_C( 8877610218385468208), INT64_C( 5776984527519295337)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 4704994953943345443),
                             INT64_C( 8877610218385468208), INT64_C( 5776984527519295337)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 7391005387705442660),
                             INT64_C(-5091463632259113685), INT64_C(  685405269785004780)) },
    { UINT8_C( 37),
      easysimd_mm512_set_epi64(INT64_C(-1117409850830928520), INT64_C(-1414228054518303181),
                            INT64_C(-3038909907977133732), INT64_C(-7842471790453318316),
                            INT64_C(-7256600765093102205), INT64_C( 3898178537456140670),
                            INT64_C( -393151907512138120), INT64_C(  600214805061827669)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 3898178537456140670),
                             INT64_C(                   0), INT64_C(  600214805061827669)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-1414228054518303181),
                             INT64_C(                   0), INT64_C(-7842471790453318316)) },
    { UINT8_C(246),
      easysimd_mm512_set_epi64(INT64_C(  377173394815185621), INT64_C(-6742373427678247978),
                            INT64_C( 7306080674171373254), INT64_C( 2163582539809461657),
                            INT64_C(-7942896186346970451), INT64_C(-2775611318017263858),
                            INT64_C( 7897714815450887445), INT64_C(-2675129847260557604)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-2775611318017263858),
                             INT64_C( 7897714815450887445), INT64_C(                   0)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-6742373427678247978),
                             INT64_C( 7306080674171373254), INT64_C(                   0)) },
    { UINT8_C(166),
      easysimd_mm512_set_epi64(INT64_C( 8594666725077939668), INT64_C(-2603844271228681340),
                            INT64_C( 3101493959844818499), INT64_C(-7569746812758465314),
                            INT64_C(  -95001376835728923), INT64_C( 2811174252033921756),
                            INT64_C( 1971701120159461885), INT64_C(-1168609383370522899)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 2811174252033921756),
                             INT64_C( 1971701120159461885), INT64_C(                   0)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-2603844271228681340),
                             INT64_C( 3101493959844818499), INT64_C(                   0)) },
    { UINT8_C( 26),
      easysimd_mm512_set_epi64(INT64_C( 8946221359026744959), INT64_C( 8570432854894274862),
                            INT64_C(-4670296842224865750), INT64_C(-8736438908262001915),
                            INT64_C(-1516874692875012272), INT64_C( 2654080637722702840),
                            INT64_C(-4444585746033374017), INT64_C( 7620312646179506248)),
      easysimd_mm256_set_epi64x(INT64_C(-1516874692875012272), INT64_C(                   0),
                             INT64_C(-4444585746033374017), INT64_C(                   0)),
      easysimd_mm256_set_epi64x(INT64_C( 8946221359026744959), INT64_C(                   0),
                             INT64_C(-4670296842224865750), INT64_C(                   0)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i r;
    r = easysimd_mm512_maskz_extracti64x4_epi64(test_vec[i].k, test_vec[i].a, 0);
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r0);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_extracti64x4_epi64(test_vec[i].k, test_vec[i].a, 1);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_extracti64x4_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_extractf32x4_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_extractf32x4_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_extractf64x2_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_extractf64x2_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extractf32x4_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extractf32x4_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extractf32x4_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extractf32x8_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extractf32x8_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extractf32x8_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extractf64x2_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extractf64x2_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extractf64x2_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extractf64x4_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extractf64x4_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extractf64x4_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extracti32x4_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extracti32x4_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extracti32x4_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extracti32x8_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extracti32x8_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extracti32x8_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extracti64x2_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extracti64x2_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extracti64x2_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_extracti64x4_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_extracti64x4_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_extracti64x4_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
