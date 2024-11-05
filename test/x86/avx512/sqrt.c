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
 */

#define EASYSIMD_TEST_X86_AVX512_INSN sqrt

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/sqrt.h>

static int
test_easysimd_mm_mask_sqrt_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -283.07), EASYSIMD_FLOAT32_C(   491.86), EASYSIMD_FLOAT32_C(  -684.73), EASYSIMD_FLOAT32_C(  -121.83) },
      UINT8_C(249),
      { EASYSIMD_FLOAT32_C(    66.71), EASYSIMD_FLOAT32_C(   409.24), EASYSIMD_FLOAT32_C(  -354.26), EASYSIMD_FLOAT32_C(   181.42) },
      { EASYSIMD_FLOAT32_C(     8.17), EASYSIMD_FLOAT32_C(   491.86), EASYSIMD_FLOAT32_C(  -684.73), EASYSIMD_FLOAT32_C(    13.47) } },
    { { EASYSIMD_FLOAT32_C(  -495.18), EASYSIMD_FLOAT32_C(    65.19), EASYSIMD_FLOAT32_C(  -670.28), EASYSIMD_FLOAT32_C(  -924.25) },
      UINT8_C(119),
      { EASYSIMD_FLOAT32_C(  -270.07), EASYSIMD_FLOAT32_C(  -891.38), EASYSIMD_FLOAT32_C(  -753.88), EASYSIMD_FLOAT32_C(   380.51) },
      {            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -924.25) } },
    { { EASYSIMD_FLOAT32_C(   996.09), EASYSIMD_FLOAT32_C(    96.88), EASYSIMD_FLOAT32_C(   680.96), EASYSIMD_FLOAT32_C(  -806.24) },
      UINT8_C(235),
      { EASYSIMD_FLOAT32_C(   596.63), EASYSIMD_FLOAT32_C(  -284.09), EASYSIMD_FLOAT32_C(   951.40), EASYSIMD_FLOAT32_C(   -33.69) },
      { EASYSIMD_FLOAT32_C(    24.43),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   680.96),            EASYSIMD_MATH_NANF } },
    { { EASYSIMD_FLOAT32_C(    21.83), EASYSIMD_FLOAT32_C(  -882.75), EASYSIMD_FLOAT32_C(   459.40), EASYSIMD_FLOAT32_C(   901.95) },
      UINT8_C( 29),
      { EASYSIMD_FLOAT32_C(   -48.74), EASYSIMD_FLOAT32_C(  -782.79), EASYSIMD_FLOAT32_C(   712.35), EASYSIMD_FLOAT32_C(    35.11) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -882.75), EASYSIMD_FLOAT32_C(    26.69), EASYSIMD_FLOAT32_C(     5.93) } },
    { { EASYSIMD_FLOAT32_C(   283.92), EASYSIMD_FLOAT32_C(   121.58), EASYSIMD_FLOAT32_C(   680.84), EASYSIMD_FLOAT32_C(  -534.66) },
      UINT8_C(136),
      { EASYSIMD_FLOAT32_C(  -253.97), EASYSIMD_FLOAT32_C(  -204.94), EASYSIMD_FLOAT32_C(   702.16), EASYSIMD_FLOAT32_C(   139.15) },
      { EASYSIMD_FLOAT32_C(   283.92), EASYSIMD_FLOAT32_C(   121.58), EASYSIMD_FLOAT32_C(   680.84), EASYSIMD_FLOAT32_C(    11.80) } },
    { { EASYSIMD_FLOAT32_C(   525.00), EASYSIMD_FLOAT32_C(   810.77), EASYSIMD_FLOAT32_C(   385.27), EASYSIMD_FLOAT32_C(   -94.49) },
      UINT8_C( 93),
      { EASYSIMD_FLOAT32_C(  -517.84), EASYSIMD_FLOAT32_C(  -413.53), EASYSIMD_FLOAT32_C(  -999.38), EASYSIMD_FLOAT32_C(   685.15) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   810.77),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    26.18) } },
    { { EASYSIMD_FLOAT32_C(  -816.90), EASYSIMD_FLOAT32_C(  -283.47), EASYSIMD_FLOAT32_C(   636.55), EASYSIMD_FLOAT32_C(   149.41) },
      UINT8_C( 53),
      { EASYSIMD_FLOAT32_C(   753.80), EASYSIMD_FLOAT32_C(  -391.19), EASYSIMD_FLOAT32_C(   640.31), EASYSIMD_FLOAT32_C(  -412.03) },
      { EASYSIMD_FLOAT32_C(    27.46), EASYSIMD_FLOAT32_C(  -283.47), EASYSIMD_FLOAT32_C(    25.30), EASYSIMD_FLOAT32_C(   149.41) } },
    { { EASYSIMD_FLOAT32_C(   560.08), EASYSIMD_FLOAT32_C(   857.53), EASYSIMD_FLOAT32_C(  -699.68), EASYSIMD_FLOAT32_C(  -404.82) },
      UINT8_C( 70),
      { EASYSIMD_FLOAT32_C(   421.90), EASYSIMD_FLOAT32_C(  -723.98), EASYSIMD_FLOAT32_C(   606.79), EASYSIMD_FLOAT32_C(    48.31) },
      { EASYSIMD_FLOAT32_C(   560.08),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    24.63), EASYSIMD_FLOAT32_C(  -404.82) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sqrt_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sqrt_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_sqrt_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_sqrt_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(207),
      { EASYSIMD_FLOAT32_C(  -598.14), EASYSIMD_FLOAT32_C(  -249.54), EASYSIMD_FLOAT32_C(  -838.79), EASYSIMD_FLOAT32_C(   926.86) },
      {            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    30.44) } },
    { UINT8_C( 56),
      { EASYSIMD_FLOAT32_C(   546.48), EASYSIMD_FLOAT32_C(  -167.63), EASYSIMD_FLOAT32_C(  -631.91), EASYSIMD_FLOAT32_C(  -971.36) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF } },
    { UINT8_C(  4),
      { EASYSIMD_FLOAT32_C(  -631.29), EASYSIMD_FLOAT32_C(   713.79), EASYSIMD_FLOAT32_C(   601.95), EASYSIMD_FLOAT32_C(    85.25) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    24.53), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(220),
      { EASYSIMD_FLOAT32_C(  -248.64), EASYSIMD_FLOAT32_C(  -176.39), EASYSIMD_FLOAT32_C(   104.13), EASYSIMD_FLOAT32_C(   360.17) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    10.20), EASYSIMD_FLOAT32_C(    18.98) } },
    { UINT8_C( 48),
      { EASYSIMD_FLOAT32_C(   692.10), EASYSIMD_FLOAT32_C(   -79.75), EASYSIMD_FLOAT32_C(  -678.55), EASYSIMD_FLOAT32_C(   992.42) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 63),
      { EASYSIMD_FLOAT32_C(   462.90), EASYSIMD_FLOAT32_C(   414.33), EASYSIMD_FLOAT32_C(   791.45), EASYSIMD_FLOAT32_C(    69.69) },
      { EASYSIMD_FLOAT32_C(    21.52), EASYSIMD_FLOAT32_C(    20.36), EASYSIMD_FLOAT32_C(    28.13), EASYSIMD_FLOAT32_C(     8.35) } },
    { UINT8_C(206),
      { EASYSIMD_FLOAT32_C(  -186.49), EASYSIMD_FLOAT32_C(   471.55), EASYSIMD_FLOAT32_C(   213.10), EASYSIMD_FLOAT32_C(   -25.28) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    21.72), EASYSIMD_FLOAT32_C(    14.60),            EASYSIMD_MATH_NANF } },
    { UINT8_C(190),
      { EASYSIMD_FLOAT32_C(   774.33), EASYSIMD_FLOAT32_C(  -478.80), EASYSIMD_FLOAT32_C(  -769.22), EASYSIMD_FLOAT32_C(  -857.58) },
      { EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sqrt_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sqrt_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_sqrt_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_sqrt_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -450.16), EASYSIMD_FLOAT64_C(   649.62) },
      UINT8_C(138),
      { EASYSIMD_FLOAT64_C(  -736.37), EASYSIMD_FLOAT64_C(   251.57) },
      { EASYSIMD_FLOAT64_C(  -450.16), EASYSIMD_FLOAT64_C(    15.86) } },
    { { EASYSIMD_FLOAT64_C(   596.39), EASYSIMD_FLOAT64_C(   613.96) },
      UINT8_C(113),
      { EASYSIMD_FLOAT64_C(  -580.00), EASYSIMD_FLOAT64_C(  -281.90) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   613.96) } },
    { { EASYSIMD_FLOAT64_C(   363.09), EASYSIMD_FLOAT64_C(  -116.08) },
      UINT8_C(218),
      { EASYSIMD_FLOAT64_C(  -716.66), EASYSIMD_FLOAT64_C(   205.38) },
      { EASYSIMD_FLOAT64_C(   363.09), EASYSIMD_FLOAT64_C(    14.33) } },
    { { EASYSIMD_FLOAT64_C(  -597.38), EASYSIMD_FLOAT64_C(   798.77) },
      UINT8_C( 39),
      { EASYSIMD_FLOAT64_C(   816.95), EASYSIMD_FLOAT64_C(   590.22) },
      { EASYSIMD_FLOAT64_C(    28.58), EASYSIMD_FLOAT64_C(    24.29) } },
    { { EASYSIMD_FLOAT64_C(   737.97), EASYSIMD_FLOAT64_C(  -720.42) },
      UINT8_C( 43),
      { EASYSIMD_FLOAT64_C(   209.52), EASYSIMD_FLOAT64_C(   492.68) },
      { EASYSIMD_FLOAT64_C(    14.47), EASYSIMD_FLOAT64_C(    22.20) } },
    { { EASYSIMD_FLOAT64_C(   378.45), EASYSIMD_FLOAT64_C(  -392.07) },
      UINT8_C( 70),
      { EASYSIMD_FLOAT64_C(   899.65), EASYSIMD_FLOAT64_C(  -161.29) },
      { EASYSIMD_FLOAT64_C(   378.45),             EASYSIMD_MATH_NAN } },
    { { EASYSIMD_FLOAT64_C(   409.43), EASYSIMD_FLOAT64_C(  -550.52) },
      UINT8_C(178),
      { EASYSIMD_FLOAT64_C(   920.57), EASYSIMD_FLOAT64_C(  -286.89) },
      { EASYSIMD_FLOAT64_C(   409.43),             EASYSIMD_MATH_NAN } },
    { { EASYSIMD_FLOAT64_C(   739.90), EASYSIMD_FLOAT64_C(   516.96) },
      UINT8_C(153),
      { EASYSIMD_FLOAT64_C(   742.82), EASYSIMD_FLOAT64_C(   936.95) },
      { EASYSIMD_FLOAT64_C(    27.25), EASYSIMD_FLOAT64_C(   516.96) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sqrt_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sqrt_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_sqrt_pd(src, k, a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_sqrt_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(133),
      { EASYSIMD_FLOAT64_C(   105.91), EASYSIMD_FLOAT64_C(  -179.12) },
      { EASYSIMD_FLOAT64_C(    10.29), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 96),
      { EASYSIMD_FLOAT64_C(   389.25), EASYSIMD_FLOAT64_C(  -973.75) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(   188.02), EASYSIMD_FLOAT64_C(  -305.47) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(129),
      { EASYSIMD_FLOAT64_C(  -221.75), EASYSIMD_FLOAT64_C(  -567.50) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(121),
      { EASYSIMD_FLOAT64_C(   181.98), EASYSIMD_FLOAT64_C(   642.02) },
      { EASYSIMD_FLOAT64_C(    13.49), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(252),
      { EASYSIMD_FLOAT64_C(  -439.57), EASYSIMD_FLOAT64_C(  -750.05) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 66),
      { EASYSIMD_FLOAT64_C(  -539.93), EASYSIMD_FLOAT64_C(    88.66) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     9.42) } },
    { UINT8_C(225),
      { EASYSIMD_FLOAT64_C(   -90.44), EASYSIMD_FLOAT64_C(   576.99) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sqrt_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sqrt_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_sqrt_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sqrt_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -955.78), EASYSIMD_FLOAT32_C(   622.67), EASYSIMD_FLOAT32_C(   316.89), EASYSIMD_FLOAT32_C(   561.18),
        EASYSIMD_FLOAT32_C(   949.74), EASYSIMD_FLOAT32_C(    59.71), EASYSIMD_FLOAT32_C(   498.13), EASYSIMD_FLOAT32_C(    -5.09) },
      UINT8_C( 12),
      { EASYSIMD_FLOAT32_C(  -680.99), EASYSIMD_FLOAT32_C(  -549.72), EASYSIMD_FLOAT32_C(   554.87), EASYSIMD_FLOAT32_C(  -654.74),
        EASYSIMD_FLOAT32_C(  -691.72), EASYSIMD_FLOAT32_C(  -257.11), EASYSIMD_FLOAT32_C(    39.79), EASYSIMD_FLOAT32_C(   983.22) },
      { EASYSIMD_FLOAT32_C(  -955.78), EASYSIMD_FLOAT32_C(   622.67), EASYSIMD_FLOAT32_C(    23.56),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(   949.74), EASYSIMD_FLOAT32_C(    59.71), EASYSIMD_FLOAT32_C(   498.13), EASYSIMD_FLOAT32_C(    -5.09) } },
    { { EASYSIMD_FLOAT32_C(   521.14), EASYSIMD_FLOAT32_C(   472.29), EASYSIMD_FLOAT32_C(   937.75), EASYSIMD_FLOAT32_C(  -296.88),
        EASYSIMD_FLOAT32_C(   114.31), EASYSIMD_FLOAT32_C(   384.96), EASYSIMD_FLOAT32_C(   263.54), EASYSIMD_FLOAT32_C(   364.27) },
      UINT8_C(142),
      { EASYSIMD_FLOAT32_C(   723.62), EASYSIMD_FLOAT32_C(  -547.07), EASYSIMD_FLOAT32_C(  -777.17), EASYSIMD_FLOAT32_C(  -366.82),
        EASYSIMD_FLOAT32_C(  -970.08), EASYSIMD_FLOAT32_C(  -732.95), EASYSIMD_FLOAT32_C(  -744.15), EASYSIMD_FLOAT32_C(   346.81) },
      { EASYSIMD_FLOAT32_C(   521.14),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(   114.31), EASYSIMD_FLOAT32_C(   384.96), EASYSIMD_FLOAT32_C(   263.54), EASYSIMD_FLOAT32_C(    18.62) } },
    { { EASYSIMD_FLOAT32_C(   828.23), EASYSIMD_FLOAT32_C(  -794.41), EASYSIMD_FLOAT32_C(  -593.48), EASYSIMD_FLOAT32_C(   326.36),
        EASYSIMD_FLOAT32_C(   200.50), EASYSIMD_FLOAT32_C(  -427.86), EASYSIMD_FLOAT32_C(   645.37), EASYSIMD_FLOAT32_C(   650.79) },
      UINT8_C(158),
      { EASYSIMD_FLOAT32_C(   990.63), EASYSIMD_FLOAT32_C(   959.07), EASYSIMD_FLOAT32_C(  -130.10), EASYSIMD_FLOAT32_C(    30.43),
        EASYSIMD_FLOAT32_C(   942.29), EASYSIMD_FLOAT32_C(  -608.97), EASYSIMD_FLOAT32_C(  -497.28), EASYSIMD_FLOAT32_C(   880.04) },
      { EASYSIMD_FLOAT32_C(   828.23), EASYSIMD_FLOAT32_C(    30.97),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     5.52),
        EASYSIMD_FLOAT32_C(    30.70), EASYSIMD_FLOAT32_C(  -427.86), EASYSIMD_FLOAT32_C(   645.37), EASYSIMD_FLOAT32_C(    29.67) } },
    { { EASYSIMD_FLOAT32_C(    94.15), EASYSIMD_FLOAT32_C(   617.03), EASYSIMD_FLOAT32_C(   264.99), EASYSIMD_FLOAT32_C(  -642.31),
        EASYSIMD_FLOAT32_C(   -18.70), EASYSIMD_FLOAT32_C(   364.17), EASYSIMD_FLOAT32_C(  -918.69), EASYSIMD_FLOAT32_C(   434.23) },
      UINT8_C(198),
      { EASYSIMD_FLOAT32_C(  -285.52), EASYSIMD_FLOAT32_C(   464.15), EASYSIMD_FLOAT32_C(   854.05), EASYSIMD_FLOAT32_C(   -29.67),
        EASYSIMD_FLOAT32_C(  -189.04), EASYSIMD_FLOAT32_C(   682.27), EASYSIMD_FLOAT32_C(   175.92), EASYSIMD_FLOAT32_C(   217.48) },
      { EASYSIMD_FLOAT32_C(    94.15), EASYSIMD_FLOAT32_C(    21.54), EASYSIMD_FLOAT32_C(    29.22), EASYSIMD_FLOAT32_C(  -642.31),
        EASYSIMD_FLOAT32_C(   -18.70), EASYSIMD_FLOAT32_C(   364.17), EASYSIMD_FLOAT32_C(    13.26), EASYSIMD_FLOAT32_C(    14.75) } },
    { { EASYSIMD_FLOAT32_C(     8.63), EASYSIMD_FLOAT32_C(  -623.58), EASYSIMD_FLOAT32_C(   789.62), EASYSIMD_FLOAT32_C(  -346.00),
        EASYSIMD_FLOAT32_C(  -972.79), EASYSIMD_FLOAT32_C(   916.62), EASYSIMD_FLOAT32_C(  -355.37), EASYSIMD_FLOAT32_C(   986.28) },
      UINT8_C( 37),
      { EASYSIMD_FLOAT32_C(   675.06), EASYSIMD_FLOAT32_C(   928.56), EASYSIMD_FLOAT32_C(   177.55), EASYSIMD_FLOAT32_C(  -822.22),
        EASYSIMD_FLOAT32_C(   808.60), EASYSIMD_FLOAT32_C(  -728.30), EASYSIMD_FLOAT32_C(   794.81), EASYSIMD_FLOAT32_C(    73.60) },
      { EASYSIMD_FLOAT32_C(    25.98), EASYSIMD_FLOAT32_C(  -623.58), EASYSIMD_FLOAT32_C(    13.32), EASYSIMD_FLOAT32_C(  -346.00),
        EASYSIMD_FLOAT32_C(  -972.79),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -355.37), EASYSIMD_FLOAT32_C(   986.28) } },
    { { EASYSIMD_FLOAT32_C(  -370.61), EASYSIMD_FLOAT32_C(  -223.88), EASYSIMD_FLOAT32_C(  -562.23), EASYSIMD_FLOAT32_C(  -289.31),
        EASYSIMD_FLOAT32_C(  -789.65), EASYSIMD_FLOAT32_C(  -975.24), EASYSIMD_FLOAT32_C(   425.18), EASYSIMD_FLOAT32_C(   674.50) },
      UINT8_C( 97),
      { EASYSIMD_FLOAT32_C(  -604.49), EASYSIMD_FLOAT32_C(  -514.53), EASYSIMD_FLOAT32_C(   561.08), EASYSIMD_FLOAT32_C(   571.43),
        EASYSIMD_FLOAT32_C(   702.95), EASYSIMD_FLOAT32_C(  -430.28), EASYSIMD_FLOAT32_C(   947.85), EASYSIMD_FLOAT32_C(   492.57) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -223.88), EASYSIMD_FLOAT32_C(  -562.23), EASYSIMD_FLOAT32_C(  -289.31),
        EASYSIMD_FLOAT32_C(  -789.65),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    30.79), EASYSIMD_FLOAT32_C(   674.50) } },
    { { EASYSIMD_FLOAT32_C(   223.72), EASYSIMD_FLOAT32_C(   975.07), EASYSIMD_FLOAT32_C(   409.19), EASYSIMD_FLOAT32_C(   868.35),
        EASYSIMD_FLOAT32_C(   961.34), EASYSIMD_FLOAT32_C(  -804.29), EASYSIMD_FLOAT32_C(   543.41), EASYSIMD_FLOAT32_C(   889.91) },
      UINT8_C(226),
      { EASYSIMD_FLOAT32_C(   721.19), EASYSIMD_FLOAT32_C(   698.51), EASYSIMD_FLOAT32_C(   644.95), EASYSIMD_FLOAT32_C(   516.01),
        EASYSIMD_FLOAT32_C(  -227.89), EASYSIMD_FLOAT32_C(  -725.66), EASYSIMD_FLOAT32_C(  -707.88), EASYSIMD_FLOAT32_C(   209.87) },
      { EASYSIMD_FLOAT32_C(   223.72), EASYSIMD_FLOAT32_C(    26.43), EASYSIMD_FLOAT32_C(   409.19), EASYSIMD_FLOAT32_C(   868.35),
        EASYSIMD_FLOAT32_C(   961.34),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    14.49) } },
    { { EASYSIMD_FLOAT32_C(   -14.97), EASYSIMD_FLOAT32_C(  -497.53), EASYSIMD_FLOAT32_C(   234.64), EASYSIMD_FLOAT32_C(  -589.79),
        EASYSIMD_FLOAT32_C(  -823.03), EASYSIMD_FLOAT32_C(   113.45), EASYSIMD_FLOAT32_C(  -194.28), EASYSIMD_FLOAT32_C(  -337.56) },
      UINT8_C(230),
      { EASYSIMD_FLOAT32_C(  -622.86), EASYSIMD_FLOAT32_C(  -634.62), EASYSIMD_FLOAT32_C(   244.25), EASYSIMD_FLOAT32_C(  -675.00),
        EASYSIMD_FLOAT32_C(   857.95), EASYSIMD_FLOAT32_C(  -532.03), EASYSIMD_FLOAT32_C(  -699.94), EASYSIMD_FLOAT32_C(   267.14) },
      { EASYSIMD_FLOAT32_C(   -14.97),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    15.63), EASYSIMD_FLOAT32_C(  -589.79),
        EASYSIMD_FLOAT32_C(  -823.03),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    16.34) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sqrt_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sqrt_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_sqrt_ps(src, k, a);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_sqrt_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(224),
      { EASYSIMD_FLOAT32_C(  -738.59), EASYSIMD_FLOAT32_C(   462.84), EASYSIMD_FLOAT32_C(   879.73), EASYSIMD_FLOAT32_C(  -848.68),
        EASYSIMD_FLOAT32_C(  -163.90), EASYSIMD_FLOAT32_C(   600.92), EASYSIMD_FLOAT32_C(   849.82), EASYSIMD_FLOAT32_C(  -518.95) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    24.51), EASYSIMD_FLOAT32_C(    29.15),            EASYSIMD_MATH_NANF } },
    { UINT8_C(132),
      { EASYSIMD_FLOAT32_C(  -378.07), EASYSIMD_FLOAT32_C(  -244.62), EASYSIMD_FLOAT32_C(   409.05), EASYSIMD_FLOAT32_C(   831.80),
        EASYSIMD_FLOAT32_C(   740.42), EASYSIMD_FLOAT32_C(   911.52), EASYSIMD_FLOAT32_C(    66.44), EASYSIMD_FLOAT32_C(  -849.38) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    20.22), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF } },
    { UINT8_C(136),
      { EASYSIMD_FLOAT32_C(  -820.11), EASYSIMD_FLOAT32_C(   -43.66), EASYSIMD_FLOAT32_C(  -249.07), EASYSIMD_FLOAT32_C(  -145.57),
        EASYSIMD_FLOAT32_C(   333.49), EASYSIMD_FLOAT32_C(   116.31), EASYSIMD_FLOAT32_C(  -901.32), EASYSIMD_FLOAT32_C(   658.48) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    25.66) } },
    { UINT8_C(186),
      { EASYSIMD_FLOAT32_C(  -433.35), EASYSIMD_FLOAT32_C(   958.55), EASYSIMD_FLOAT32_C(  -758.60), EASYSIMD_FLOAT32_C(   -97.04),
        EASYSIMD_FLOAT32_C(  -780.05), EASYSIMD_FLOAT32_C(   704.25), EASYSIMD_FLOAT32_C(  -217.30), EASYSIMD_FLOAT32_C(  -628.73) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    30.96), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF,
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    26.54), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF } },
    { UINT8_C(177),
      { EASYSIMD_FLOAT32_C(  -616.38), EASYSIMD_FLOAT32_C(  -778.91), EASYSIMD_FLOAT32_C(    21.39), EASYSIMD_FLOAT32_C(   500.55),
        EASYSIMD_FLOAT32_C(  -156.97), EASYSIMD_FLOAT32_C(   776.78), EASYSIMD_FLOAT32_C(   -90.40), EASYSIMD_FLOAT32_C(  -325.17) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    27.87), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF } },
    { UINT8_C( 26),
      { EASYSIMD_FLOAT32_C(  -178.88), EASYSIMD_FLOAT32_C(   741.27), EASYSIMD_FLOAT32_C(   667.82), EASYSIMD_FLOAT32_C(   -90.39),
        EASYSIMD_FLOAT32_C(   921.16), EASYSIMD_FLOAT32_C(  -375.84), EASYSIMD_FLOAT32_C(   660.53), EASYSIMD_FLOAT32_C(  -224.41) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    27.23), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(    30.35), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(100),
      { EASYSIMD_FLOAT32_C(  -223.16), EASYSIMD_FLOAT32_C(  -125.73), EASYSIMD_FLOAT32_C(   616.13), EASYSIMD_FLOAT32_C(   751.11),
        EASYSIMD_FLOAT32_C(   440.91), EASYSIMD_FLOAT32_C(   574.67), EASYSIMD_FLOAT32_C(   992.51), EASYSIMD_FLOAT32_C(  -656.12) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    24.82), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    23.97), EASYSIMD_FLOAT32_C(    31.50), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(177),
      { EASYSIMD_FLOAT32_C(   696.75), EASYSIMD_FLOAT32_C(   126.57), EASYSIMD_FLOAT32_C(  -834.10), EASYSIMD_FLOAT32_C(  -762.90),
        EASYSIMD_FLOAT32_C(   510.19), EASYSIMD_FLOAT32_C(  -613.01), EASYSIMD_FLOAT32_C(   258.49), EASYSIMD_FLOAT32_C(    10.74) },
      { EASYSIMD_FLOAT32_C(    26.40), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    22.59),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     3.28) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sqrt_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sqrt_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_sqrt_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sqrt_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   230.02), EASYSIMD_FLOAT64_C(    35.27), EASYSIMD_FLOAT64_C(   920.34), EASYSIMD_FLOAT64_C(   904.85) },
      UINT8_C(210),
      { EASYSIMD_FLOAT64_C(  -258.54), EASYSIMD_FLOAT64_C(   646.12), EASYSIMD_FLOAT64_C(  -779.72), EASYSIMD_FLOAT64_C(   651.07) },
      { EASYSIMD_FLOAT64_C(   230.02), EASYSIMD_FLOAT64_C(    25.42), EASYSIMD_FLOAT64_C(   920.34), EASYSIMD_FLOAT64_C(   904.85) } },
    { { EASYSIMD_FLOAT64_C(   567.28), EASYSIMD_FLOAT64_C(  -155.57), EASYSIMD_FLOAT64_C(   311.60), EASYSIMD_FLOAT64_C(  -657.13) },
      UINT8_C( 10),
      { EASYSIMD_FLOAT64_C(  -911.55), EASYSIMD_FLOAT64_C(   217.14), EASYSIMD_FLOAT64_C(  -581.80), EASYSIMD_FLOAT64_C(   839.55) },
      { EASYSIMD_FLOAT64_C(   567.28), EASYSIMD_FLOAT64_C(    14.74), EASYSIMD_FLOAT64_C(   311.60), EASYSIMD_FLOAT64_C(    28.97) } },
    { { EASYSIMD_FLOAT64_C(  -341.95), EASYSIMD_FLOAT64_C(   992.88), EASYSIMD_FLOAT64_C(   832.06), EASYSIMD_FLOAT64_C(     1.93) },
      UINT8_C(119),
      { EASYSIMD_FLOAT64_C(   528.82), EASYSIMD_FLOAT64_C(  -871.49), EASYSIMD_FLOAT64_C(   953.40), EASYSIMD_FLOAT64_C(   765.91) },
      { EASYSIMD_FLOAT64_C(    23.00),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    30.88), EASYSIMD_FLOAT64_C(     1.93) } },
    { { EASYSIMD_FLOAT64_C(   638.70), EASYSIMD_FLOAT64_C(  -659.61), EASYSIMD_FLOAT64_C(    24.40), EASYSIMD_FLOAT64_C(  -350.56) },
      UINT8_C( 69),
      { EASYSIMD_FLOAT64_C(  -940.33), EASYSIMD_FLOAT64_C(  -430.22), EASYSIMD_FLOAT64_C(   475.26), EASYSIMD_FLOAT64_C(  -387.87) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -659.61), EASYSIMD_FLOAT64_C(    21.80), EASYSIMD_FLOAT64_C(  -350.56) } },
    { { EASYSIMD_FLOAT64_C(   311.24), EASYSIMD_FLOAT64_C(   121.38), EASYSIMD_FLOAT64_C(  -167.60), EASYSIMD_FLOAT64_C(   -37.69) },
      UINT8_C(123),
      { EASYSIMD_FLOAT64_C(   676.84), EASYSIMD_FLOAT64_C(  -726.09), EASYSIMD_FLOAT64_C(    31.53), EASYSIMD_FLOAT64_C(  -521.09) },
      { EASYSIMD_FLOAT64_C(    26.02),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -167.60),             EASYSIMD_MATH_NAN } },
    { { EASYSIMD_FLOAT64_C(  -637.64), EASYSIMD_FLOAT64_C(  -751.33), EASYSIMD_FLOAT64_C(  -102.89), EASYSIMD_FLOAT64_C(  -798.08) },
      UINT8_C( 36),
      { EASYSIMD_FLOAT64_C(  -110.01), EASYSIMD_FLOAT64_C(  -966.02), EASYSIMD_FLOAT64_C(   908.66), EASYSIMD_FLOAT64_C(  -322.51) },
      { EASYSIMD_FLOAT64_C(  -637.64), EASYSIMD_FLOAT64_C(  -751.33), EASYSIMD_FLOAT64_C(    30.14), EASYSIMD_FLOAT64_C(  -798.08) } },
    { { EASYSIMD_FLOAT64_C(   562.79), EASYSIMD_FLOAT64_C(  -962.83), EASYSIMD_FLOAT64_C(  -369.11), EASYSIMD_FLOAT64_C(   328.70) },
      UINT8_C(173),
      { EASYSIMD_FLOAT64_C(   -28.71), EASYSIMD_FLOAT64_C(  -646.89), EASYSIMD_FLOAT64_C(  -674.69), EASYSIMD_FLOAT64_C(  -458.30) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -962.83),             EASYSIMD_MATH_NAN,             EASYSIMD_MATH_NAN } },
    { { EASYSIMD_FLOAT64_C(  -587.22), EASYSIMD_FLOAT64_C(  -104.91), EASYSIMD_FLOAT64_C(  -983.04), EASYSIMD_FLOAT64_C(    24.90) },
      UINT8_C( 63),
      { EASYSIMD_FLOAT64_C(   138.34), EASYSIMD_FLOAT64_C(   857.31), EASYSIMD_FLOAT64_C(   168.65), EASYSIMD_FLOAT64_C(   827.00) },
      { EASYSIMD_FLOAT64_C(    11.76), EASYSIMD_FLOAT64_C(    29.28), EASYSIMD_FLOAT64_C(    12.99), EASYSIMD_FLOAT64_C(    28.76) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sqrt_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sqrt_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_sqrt_pd(src, k, a);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_sqrt_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C( 83),
      { EASYSIMD_FLOAT64_C(   442.56), EASYSIMD_FLOAT64_C(  -141.47), EASYSIMD_FLOAT64_C(  -986.95), EASYSIMD_FLOAT64_C(   804.92) },
      { EASYSIMD_FLOAT64_C(    21.04),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(218),
      { EASYSIMD_FLOAT64_C(   -89.84), EASYSIMD_FLOAT64_C(  -993.16), EASYSIMD_FLOAT64_C(  -986.06), EASYSIMD_FLOAT64_C(   800.15) },
      { EASYSIMD_FLOAT64_C(     0.00),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    28.29) } },
    { UINT8_C( 13),
      { EASYSIMD_FLOAT64_C(   922.60), EASYSIMD_FLOAT64_C(  -522.36), EASYSIMD_FLOAT64_C(   603.61), EASYSIMD_FLOAT64_C(   959.76) },
      { EASYSIMD_FLOAT64_C(    30.37), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    24.57), EASYSIMD_FLOAT64_C(    30.98) } },
    { UINT8_C(158),
      { EASYSIMD_FLOAT64_C(   -67.68), EASYSIMD_FLOAT64_C(   635.63), EASYSIMD_FLOAT64_C(  -920.18), EASYSIMD_FLOAT64_C(   285.42) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    25.21),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    16.89) } },
    { UINT8_C(112),
      { EASYSIMD_FLOAT64_C(  -378.48), EASYSIMD_FLOAT64_C(   698.20), EASYSIMD_FLOAT64_C(  -143.97), EASYSIMD_FLOAT64_C(  -361.52) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(155),
      { EASYSIMD_FLOAT64_C(    62.36), EASYSIMD_FLOAT64_C(   776.82), EASYSIMD_FLOAT64_C(  -419.59), EASYSIMD_FLOAT64_C(  -768.99) },
      { EASYSIMD_FLOAT64_C(     7.90), EASYSIMD_FLOAT64_C(    27.87), EASYSIMD_FLOAT64_C(     0.00),             EASYSIMD_MATH_NAN } },
    { UINT8_C(132),
      { EASYSIMD_FLOAT64_C(  -885.45), EASYSIMD_FLOAT64_C(   673.57), EASYSIMD_FLOAT64_C(  -537.65), EASYSIMD_FLOAT64_C(  -872.40) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 75),
      { EASYSIMD_FLOAT64_C(   569.56), EASYSIMD_FLOAT64_C(    37.76), EASYSIMD_FLOAT64_C(   485.34), EASYSIMD_FLOAT64_C(   583.49) },
      { EASYSIMD_FLOAT64_C(    23.87), EASYSIMD_FLOAT64_C(     6.14), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    24.16) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++){
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sqrt_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sqrt_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_sqrt_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_sqrt_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
     { { EASYSIMD_FLOAT32_C(   217.83), EASYSIMD_FLOAT32_C(   108.94), EASYSIMD_FLOAT32_C(    38.33), EASYSIMD_FLOAT32_C(   277.15),
        EASYSIMD_FLOAT32_C(    59.82), EASYSIMD_FLOAT32_C(   344.99), EASYSIMD_FLOAT32_C(   240.56), EASYSIMD_FLOAT32_C(   162.50),
        EASYSIMD_FLOAT32_C(    91.12), EASYSIMD_FLOAT32_C(   517.25), EASYSIMD_FLOAT32_C(   419.27), EASYSIMD_FLOAT32_C(   630.58),
        EASYSIMD_FLOAT32_C(   139.48), EASYSIMD_FLOAT32_C(   227.23), EASYSIMD_FLOAT32_C(   130.66), EASYSIMD_FLOAT32_C(   968.78) },
      { EASYSIMD_FLOAT32_C(    14.76), EASYSIMD_FLOAT32_C(    10.44), EASYSIMD_FLOAT32_C(     6.19), EASYSIMD_FLOAT32_C(    16.65),
        EASYSIMD_FLOAT32_C(     7.73), EASYSIMD_FLOAT32_C(    18.57), EASYSIMD_FLOAT32_C(    15.51), EASYSIMD_FLOAT32_C(    12.75),
        EASYSIMD_FLOAT32_C(     9.55), EASYSIMD_FLOAT32_C(    22.74), EASYSIMD_FLOAT32_C(    20.48), EASYSIMD_FLOAT32_C(    25.11),
        EASYSIMD_FLOAT32_C(    11.81), EASYSIMD_FLOAT32_C(    15.07), EASYSIMD_FLOAT32_C(    11.43), EASYSIMD_FLOAT32_C(    31.13) } },
    { { EASYSIMD_FLOAT32_C(   223.24), EASYSIMD_FLOAT32_C(    61.22), EASYSIMD_FLOAT32_C(     5.71), EASYSIMD_FLOAT32_C(   939.37),
        EASYSIMD_FLOAT32_C(   950.58), EASYSIMD_FLOAT32_C(   463.21), EASYSIMD_FLOAT32_C(    93.23), EASYSIMD_FLOAT32_C(   926.17),
        EASYSIMD_FLOAT32_C(   149.54), EASYSIMD_FLOAT32_C(   345.84), EASYSIMD_FLOAT32_C(   517.84), EASYSIMD_FLOAT32_C(   367.13),
        EASYSIMD_FLOAT32_C(   366.95), EASYSIMD_FLOAT32_C(   396.36), EASYSIMD_FLOAT32_C(   650.42), EASYSIMD_FLOAT32_C(   583.12) },
      { EASYSIMD_FLOAT32_C(    14.94), EASYSIMD_FLOAT32_C(     7.82), EASYSIMD_FLOAT32_C(     2.39), EASYSIMD_FLOAT32_C(    30.65),
        EASYSIMD_FLOAT32_C(    30.83), EASYSIMD_FLOAT32_C(    21.52), EASYSIMD_FLOAT32_C(     9.66), EASYSIMD_FLOAT32_C(    30.43),
        EASYSIMD_FLOAT32_C(    12.23), EASYSIMD_FLOAT32_C(    18.60), EASYSIMD_FLOAT32_C(    22.76), EASYSIMD_FLOAT32_C(    19.16),
        EASYSIMD_FLOAT32_C(    19.16), EASYSIMD_FLOAT32_C(    19.91), EASYSIMD_FLOAT32_C(    25.50), EASYSIMD_FLOAT32_C(    24.15) } },
    { { EASYSIMD_FLOAT32_C(   710.05), EASYSIMD_FLOAT32_C(   748.28), EASYSIMD_FLOAT32_C(   893.06), EASYSIMD_FLOAT32_C(   -62.84),
        EASYSIMD_FLOAT32_C(   792.96), EASYSIMD_FLOAT32_C(   635.10), EASYSIMD_FLOAT32_C(   563.04), EASYSIMD_FLOAT32_C(   594.48),
        EASYSIMD_FLOAT32_C(   976.52), EASYSIMD_FLOAT32_C(   154.93), EASYSIMD_FLOAT32_C(    90.22), EASYSIMD_FLOAT32_C(   370.25),
        EASYSIMD_FLOAT32_C(   935.93), EASYSIMD_FLOAT32_C(   -51.25), EASYSIMD_FLOAT32_C(   771.97), EASYSIMD_FLOAT32_C(   851.63) },
      { EASYSIMD_FLOAT32_C(    26.65), EASYSIMD_FLOAT32_C(    27.35), EASYSIMD_FLOAT32_C(    29.88),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(    28.16), EASYSIMD_FLOAT32_C(    25.20), EASYSIMD_FLOAT32_C(    23.73), EASYSIMD_FLOAT32_C(    24.38),
        EASYSIMD_FLOAT32_C(    31.25), EASYSIMD_FLOAT32_C(    12.45), EASYSIMD_FLOAT32_C(     9.50), EASYSIMD_FLOAT32_C(    19.24),
        EASYSIMD_FLOAT32_C(    30.59),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    27.78), EASYSIMD_FLOAT32_C(    29.18) } },
    { { EASYSIMD_FLOAT32_C(   -30.75), EASYSIMD_FLOAT32_C(   -68.78), EASYSIMD_FLOAT32_C(   -79.81), EASYSIMD_FLOAT32_C(   475.72),
        EASYSIMD_FLOAT32_C(   407.95), EASYSIMD_FLOAT32_C(   958.53), EASYSIMD_FLOAT32_C(   380.76), EASYSIMD_FLOAT32_C(   553.07),
        EASYSIMD_FLOAT32_C(   201.21), EASYSIMD_FLOAT32_C(   214.86), EASYSIMD_FLOAT32_C(   771.54), EASYSIMD_FLOAT32_C(   348.19),
        EASYSIMD_FLOAT32_C(   997.59), EASYSIMD_FLOAT32_C(   154.92), EASYSIMD_FLOAT32_C(   997.20), EASYSIMD_FLOAT32_C(   140.62) },
      {            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    21.81),
        EASYSIMD_FLOAT32_C(    20.20), EASYSIMD_FLOAT32_C(    30.96), EASYSIMD_FLOAT32_C(    19.51), EASYSIMD_FLOAT32_C(    23.52),
        EASYSIMD_FLOAT32_C(    14.19), EASYSIMD_FLOAT32_C(    14.66), EASYSIMD_FLOAT32_C(    27.78), EASYSIMD_FLOAT32_C(    18.66),
        EASYSIMD_FLOAT32_C(    31.58), EASYSIMD_FLOAT32_C(    12.45), EASYSIMD_FLOAT32_C(    31.58), EASYSIMD_FLOAT32_C(    11.86) } },
    { { EASYSIMD_FLOAT32_C(   466.31), EASYSIMD_FLOAT32_C(   614.68), EASYSIMD_FLOAT32_C(   580.31), EASYSIMD_FLOAT32_C(   539.32),
        EASYSIMD_FLOAT32_C(   203.17), EASYSIMD_FLOAT32_C(   122.82), EASYSIMD_FLOAT32_C(   465.01), EASYSIMD_FLOAT32_C(   751.36),
        EASYSIMD_FLOAT32_C(   957.86), EASYSIMD_FLOAT32_C(    40.61), EASYSIMD_FLOAT32_C(   299.33), EASYSIMD_FLOAT32_C(   397.65),
        EASYSIMD_FLOAT32_C(   571.56), EASYSIMD_FLOAT32_C(   866.02), EASYSIMD_FLOAT32_C(   947.17), EASYSIMD_FLOAT32_C(   787.06) },
      { EASYSIMD_FLOAT32_C(    21.59), EASYSIMD_FLOAT32_C(    24.79), EASYSIMD_FLOAT32_C(    24.09), EASYSIMD_FLOAT32_C(    23.22),
        EASYSIMD_FLOAT32_C(    14.25), EASYSIMD_FLOAT32_C(    11.08), EASYSIMD_FLOAT32_C(    21.56), EASYSIMD_FLOAT32_C(    27.41),
        EASYSIMD_FLOAT32_C(    30.95), EASYSIMD_FLOAT32_C(     6.37), EASYSIMD_FLOAT32_C(    17.30), EASYSIMD_FLOAT32_C(    19.94),
        EASYSIMD_FLOAT32_C(    23.91), EASYSIMD_FLOAT32_C(    29.43), EASYSIMD_FLOAT32_C(    30.78), EASYSIMD_FLOAT32_C(    28.05) } },
    { { EASYSIMD_FLOAT32_C(   379.06), EASYSIMD_FLOAT32_C(   518.14), EASYSIMD_FLOAT32_C(   498.86), EASYSIMD_FLOAT32_C(    -3.46),
        EASYSIMD_FLOAT32_C(   -23.53), EASYSIMD_FLOAT32_C(   266.36), EASYSIMD_FLOAT32_C(   681.68), EASYSIMD_FLOAT32_C(   242.19),
        EASYSIMD_FLOAT32_C(   263.88), EASYSIMD_FLOAT32_C(   654.06), EASYSIMD_FLOAT32_C(   331.27), EASYSIMD_FLOAT32_C(   317.61),
        EASYSIMD_FLOAT32_C(   624.18), EASYSIMD_FLOAT32_C(   874.14), EASYSIMD_FLOAT32_C(   894.91), EASYSIMD_FLOAT32_C(   175.60) },
      { EASYSIMD_FLOAT32_C(    19.47), EASYSIMD_FLOAT32_C(    22.76), EASYSIMD_FLOAT32_C(    22.34),            EASYSIMD_MATH_NANF,
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    16.32), EASYSIMD_FLOAT32_C(    26.11), EASYSIMD_FLOAT32_C(    15.56),
        EASYSIMD_FLOAT32_C(    16.24), EASYSIMD_FLOAT32_C(    25.57), EASYSIMD_FLOAT32_C(    18.20), EASYSIMD_FLOAT32_C(    17.82),
        EASYSIMD_FLOAT32_C(    24.98), EASYSIMD_FLOAT32_C(    29.57), EASYSIMD_FLOAT32_C(    29.91), EASYSIMD_FLOAT32_C(    13.25) } },
    { { EASYSIMD_FLOAT32_C(   910.44), EASYSIMD_FLOAT32_C(   492.48), EASYSIMD_FLOAT32_C(   518.91), EASYSIMD_FLOAT32_C(   259.60),
        EASYSIMD_FLOAT32_C(   324.91), EASYSIMD_FLOAT32_C(   233.97), EASYSIMD_FLOAT32_C(   654.12), EASYSIMD_FLOAT32_C(   260.58),
        EASYSIMD_FLOAT32_C(   230.74), EASYSIMD_FLOAT32_C(   276.07), EASYSIMD_FLOAT32_C(   -86.08), EASYSIMD_FLOAT32_C(   582.99),
        EASYSIMD_FLOAT32_C(   393.66), EASYSIMD_FLOAT32_C(   633.68), EASYSIMD_FLOAT32_C(   958.09), EASYSIMD_FLOAT32_C(   559.47) },
      { EASYSIMD_FLOAT32_C(    30.17), EASYSIMD_FLOAT32_C(    22.19), EASYSIMD_FLOAT32_C(    22.78), EASYSIMD_FLOAT32_C(    16.11),
        EASYSIMD_FLOAT32_C(    18.03), EASYSIMD_FLOAT32_C(    15.30), EASYSIMD_FLOAT32_C(    25.58), EASYSIMD_FLOAT32_C(    16.14),
        EASYSIMD_FLOAT32_C(    15.19), EASYSIMD_FLOAT32_C(    16.62),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    24.15),
        EASYSIMD_FLOAT32_C(    19.84), EASYSIMD_FLOAT32_C(    25.17), EASYSIMD_FLOAT32_C(    30.95), EASYSIMD_FLOAT32_C(    23.65) } },
    { { EASYSIMD_FLOAT32_C(   421.74), EASYSIMD_FLOAT32_C(   223.58), EASYSIMD_FLOAT32_C(   636.89), EASYSIMD_FLOAT32_C(   288.28),
        EASYSIMD_FLOAT32_C(    38.61), EASYSIMD_FLOAT32_C(   936.22), EASYSIMD_FLOAT32_C(    31.25), EASYSIMD_FLOAT32_C(   215.71),
        EASYSIMD_FLOAT32_C(   498.70), EASYSIMD_FLOAT32_C(   630.00), EASYSIMD_FLOAT32_C(   370.58), EASYSIMD_FLOAT32_C(   365.46),
        EASYSIMD_FLOAT32_C(   300.68), EASYSIMD_FLOAT32_C(   498.16), EASYSIMD_FLOAT32_C(   559.20), EASYSIMD_FLOAT32_C(   547.97) },
      { EASYSIMD_FLOAT32_C(    20.54), EASYSIMD_FLOAT32_C(    14.95), EASYSIMD_FLOAT32_C(    25.24), EASYSIMD_FLOAT32_C(    16.98),
        EASYSIMD_FLOAT32_C(     6.21), EASYSIMD_FLOAT32_C(    30.60), EASYSIMD_FLOAT32_C(     5.59), EASYSIMD_FLOAT32_C(    14.69),
        EASYSIMD_FLOAT32_C(    22.33), EASYSIMD_FLOAT32_C(    25.10), EASYSIMD_FLOAT32_C(    19.25), EASYSIMD_FLOAT32_C(    19.12),
        EASYSIMD_FLOAT32_C(    17.34), EASYSIMD_FLOAT32_C(    22.32), EASYSIMD_FLOAT32_C(    23.65), EASYSIMD_FLOAT32_C(    23.41) } },
    { { EASYSIMD_FLOAT32_C(   482.41), EASYSIMD_FLOAT32_C(   904.16), EASYSIMD_FLOAT32_C(   301.69), EASYSIMD_FLOAT32_C(   497.46),
        EASYSIMD_FLOAT32_C(   869.63), EASYSIMD_FLOAT32_C(   866.07), EASYSIMD_FLOAT32_C(    86.91), EASYSIMD_FLOAT32_C(   705.04),
        EASYSIMD_FLOAT32_C(   534.39), EASYSIMD_FLOAT32_C(   480.29), EASYSIMD_FLOAT32_C(   152.20), EASYSIMD_FLOAT32_C(     7.09),
        EASYSIMD_FLOAT32_C(    89.72), EASYSIMD_FLOAT32_C(   938.68), EASYSIMD_FLOAT32_C(   472.63), EASYSIMD_FLOAT32_C(   431.56) },
      { EASYSIMD_FLOAT32_C(    21.96), EASYSIMD_FLOAT32_C(    30.07), EASYSIMD_FLOAT32_C(    17.37), EASYSIMD_FLOAT32_C(    22.30),
        EASYSIMD_FLOAT32_C(    29.49), EASYSIMD_FLOAT32_C(    29.43), EASYSIMD_FLOAT32_C(     9.32), EASYSIMD_FLOAT32_C(    26.55),
        EASYSIMD_FLOAT32_C(    23.12), EASYSIMD_FLOAT32_C(    21.92), EASYSIMD_FLOAT32_C(    12.34), EASYSIMD_FLOAT32_C(     2.66),
        EASYSIMD_FLOAT32_C(     9.47), EASYSIMD_FLOAT32_C(    30.64), EASYSIMD_FLOAT32_C(    21.74), EASYSIMD_FLOAT32_C(    20.77) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sqrt_ps(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_sqrt_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sqrt_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 src[16];
    const easysimd__mmask8 k;
    const easysimd_float32 a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   437.33), EASYSIMD_FLOAT32_C(   830.55), EASYSIMD_FLOAT32_C(   885.40), EASYSIMD_FLOAT32_C(   946.45),
        EASYSIMD_FLOAT32_C(   740.66), EASYSIMD_FLOAT32_C(   515.39), EASYSIMD_FLOAT32_C(   501.14), EASYSIMD_FLOAT32_C(   807.71),
        EASYSIMD_FLOAT32_C(   988.01), EASYSIMD_FLOAT32_C(   854.01), EASYSIMD_FLOAT32_C(   302.94), EASYSIMD_FLOAT32_C(   510.25),
        EASYSIMD_FLOAT32_C(    -3.35), EASYSIMD_FLOAT32_C(   705.28), EASYSIMD_FLOAT32_C(   895.93), EASYSIMD_FLOAT32_C(   944.13) },
      UINT8_C( 44),
      { EASYSIMD_FLOAT32_C(   -17.46), EASYSIMD_FLOAT32_C(   104.01), EASYSIMD_FLOAT32_C(   -12.13), EASYSIMD_FLOAT32_C(   572.59),
        EASYSIMD_FLOAT32_C(   553.23), EASYSIMD_FLOAT32_C(   667.21), EASYSIMD_FLOAT32_C(   175.86), EASYSIMD_FLOAT32_C(   857.51),
        EASYSIMD_FLOAT32_C(   875.76), EASYSIMD_FLOAT32_C(   661.26), EASYSIMD_FLOAT32_C(   359.55), EASYSIMD_FLOAT32_C(   492.88),
        EASYSIMD_FLOAT32_C(   614.94), EASYSIMD_FLOAT32_C(   592.23), EASYSIMD_FLOAT32_C(   639.48), EASYSIMD_FLOAT32_C(   586.75) },
      { EASYSIMD_FLOAT32_C(   437.33), EASYSIMD_FLOAT32_C(   830.55),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    23.93),
        EASYSIMD_FLOAT32_C(   740.66), EASYSIMD_FLOAT32_C(    25.83), EASYSIMD_FLOAT32_C(   501.14), EASYSIMD_FLOAT32_C(   807.71),
        EASYSIMD_FLOAT32_C(   988.01), EASYSIMD_FLOAT32_C(   854.01), EASYSIMD_FLOAT32_C(   302.94), EASYSIMD_FLOAT32_C(   510.25),
        EASYSIMD_FLOAT32_C(    -3.35), EASYSIMD_FLOAT32_C(   705.28), EASYSIMD_FLOAT32_C(   895.93), EASYSIMD_FLOAT32_C(   944.13) } },
    { { EASYSIMD_FLOAT32_C(   830.85), EASYSIMD_FLOAT32_C(   416.09), EASYSIMD_FLOAT32_C(   252.98), EASYSIMD_FLOAT32_C(   170.02),
        EASYSIMD_FLOAT32_C(   649.47), EASYSIMD_FLOAT32_C(    61.92), EASYSIMD_FLOAT32_C(   -30.00), EASYSIMD_FLOAT32_C(   565.15),
        EASYSIMD_FLOAT32_C(   804.54), EASYSIMD_FLOAT32_C(   537.62), EASYSIMD_FLOAT32_C(   139.69), EASYSIMD_FLOAT32_C(   223.23),
        EASYSIMD_FLOAT32_C(   700.71), EASYSIMD_FLOAT32_C(    84.06), EASYSIMD_FLOAT32_C(   154.25), EASYSIMD_FLOAT32_C(   749.16) },
      UINT8_C(186),
      { EASYSIMD_FLOAT32_C(   637.38), EASYSIMD_FLOAT32_C(   483.43), EASYSIMD_FLOAT32_C(   245.18), EASYSIMD_FLOAT32_C(   987.92),
        EASYSIMD_FLOAT32_C(   407.77), EASYSIMD_FLOAT32_C(   184.67), EASYSIMD_FLOAT32_C(   504.64), EASYSIMD_FLOAT32_C(   244.98),
        EASYSIMD_FLOAT32_C(   -92.45), EASYSIMD_FLOAT32_C(   233.10), EASYSIMD_FLOAT32_C(   347.51), EASYSIMD_FLOAT32_C(   453.74),
        EASYSIMD_FLOAT32_C(   654.02), EASYSIMD_FLOAT32_C(   778.35), EASYSIMD_FLOAT32_C(   364.48), EASYSIMD_FLOAT32_C(   774.62) },
      { EASYSIMD_FLOAT32_C(   830.85), EASYSIMD_FLOAT32_C(    21.99), EASYSIMD_FLOAT32_C(   252.98), EASYSIMD_FLOAT32_C(    31.43),
        EASYSIMD_FLOAT32_C(    20.19), EASYSIMD_FLOAT32_C(    13.59), EASYSIMD_FLOAT32_C(   -30.00), EASYSIMD_FLOAT32_C(    15.65),
        EASYSIMD_FLOAT32_C(   804.54), EASYSIMD_FLOAT32_C(   537.62), EASYSIMD_FLOAT32_C(   139.69), EASYSIMD_FLOAT32_C(   223.23),
        EASYSIMD_FLOAT32_C(   700.71), EASYSIMD_FLOAT32_C(    84.06), EASYSIMD_FLOAT32_C(   154.25), EASYSIMD_FLOAT32_C(   749.16) } },
    { { EASYSIMD_FLOAT32_C(   341.01), EASYSIMD_FLOAT32_C(   234.85), EASYSIMD_FLOAT32_C(    83.58), EASYSIMD_FLOAT32_C(   -91.38),
        EASYSIMD_FLOAT32_C(   735.59), EASYSIMD_FLOAT32_C(   -51.68), EASYSIMD_FLOAT32_C(   211.29), EASYSIMD_FLOAT32_C(   125.75),
        EASYSIMD_FLOAT32_C(   171.18), EASYSIMD_FLOAT32_C(   387.03), EASYSIMD_FLOAT32_C(   278.80), EASYSIMD_FLOAT32_C(   688.49),
        EASYSIMD_FLOAT32_C(   284.47), EASYSIMD_FLOAT32_C(   309.43), EASYSIMD_FLOAT32_C(   761.03), EASYSIMD_FLOAT32_C(   804.65) },
      UINT8_C( 32),
      { EASYSIMD_FLOAT32_C(   348.92), EASYSIMD_FLOAT32_C(   -22.74), EASYSIMD_FLOAT32_C(   451.50), EASYSIMD_FLOAT32_C(   370.23),
        EASYSIMD_FLOAT32_C(   582.81), EASYSIMD_FLOAT32_C(   734.74), EASYSIMD_FLOAT32_C(    42.12), EASYSIMD_FLOAT32_C(   353.92),
        EASYSIMD_FLOAT32_C(   504.09), EASYSIMD_FLOAT32_C(   977.38), EASYSIMD_FLOAT32_C(   328.27), EASYSIMD_FLOAT32_C(   482.22),
        EASYSIMD_FLOAT32_C(   737.20), EASYSIMD_FLOAT32_C(   630.17), EASYSIMD_FLOAT32_C(   265.58), EASYSIMD_FLOAT32_C(   661.60) },
      { EASYSIMD_FLOAT32_C(   341.01), EASYSIMD_FLOAT32_C(   234.85), EASYSIMD_FLOAT32_C(    83.58), EASYSIMD_FLOAT32_C(   -91.38),
        EASYSIMD_FLOAT32_C(   735.59), EASYSIMD_FLOAT32_C(    27.11), EASYSIMD_FLOAT32_C(   211.29), EASYSIMD_FLOAT32_C(   125.75),
        EASYSIMD_FLOAT32_C(   171.18), EASYSIMD_FLOAT32_C(   387.03), EASYSIMD_FLOAT32_C(   278.80), EASYSIMD_FLOAT32_C(   688.49),
        EASYSIMD_FLOAT32_C(   284.47), EASYSIMD_FLOAT32_C(   309.43), EASYSIMD_FLOAT32_C(   761.03), EASYSIMD_FLOAT32_C(   804.65) } },
    { { EASYSIMD_FLOAT32_C(   525.05), EASYSIMD_FLOAT32_C(   166.18), EASYSIMD_FLOAT32_C(   952.07), EASYSIMD_FLOAT32_C(   664.08),
        EASYSIMD_FLOAT32_C(   409.88), EASYSIMD_FLOAT32_C(   422.77), EASYSIMD_FLOAT32_C(   381.48), EASYSIMD_FLOAT32_C(   505.76),
        EASYSIMD_FLOAT32_C(   441.87), EASYSIMD_FLOAT32_C(   222.70), EASYSIMD_FLOAT32_C(   519.86), EASYSIMD_FLOAT32_C(   854.25),
        EASYSIMD_FLOAT32_C(   -46.91), EASYSIMD_FLOAT32_C(    81.38), EASYSIMD_FLOAT32_C(   328.69), EASYSIMD_FLOAT32_C(   977.87) },
      UINT8_C(124),
      { EASYSIMD_FLOAT32_C(   332.53), EASYSIMD_FLOAT32_C(   706.88), EASYSIMD_FLOAT32_C(   312.95), EASYSIMD_FLOAT32_C(   533.68),
        EASYSIMD_FLOAT32_C(   -71.13), EASYSIMD_FLOAT32_C(   -10.56), EASYSIMD_FLOAT32_C(   585.48), EASYSIMD_FLOAT32_C(   449.30),
        EASYSIMD_FLOAT32_C(   860.34), EASYSIMD_FLOAT32_C(    80.38), EASYSIMD_FLOAT32_C(   990.66), EASYSIMD_FLOAT32_C(   203.10),
        EASYSIMD_FLOAT32_C(   -25.23), EASYSIMD_FLOAT32_C(   283.85), EASYSIMD_FLOAT32_C(   906.28), EASYSIMD_FLOAT32_C(   992.76) },
      { EASYSIMD_FLOAT32_C(   525.05), EASYSIMD_FLOAT32_C(   166.18), EASYSIMD_FLOAT32_C(    17.69), EASYSIMD_FLOAT32_C(    23.10),
                   EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    24.20), EASYSIMD_FLOAT32_C(   505.76),
        EASYSIMD_FLOAT32_C(   441.87), EASYSIMD_FLOAT32_C(   222.70), EASYSIMD_FLOAT32_C(   519.86), EASYSIMD_FLOAT32_C(   854.25),
        EASYSIMD_FLOAT32_C(   -46.91), EASYSIMD_FLOAT32_C(    81.38), EASYSIMD_FLOAT32_C(   328.69), EASYSIMD_FLOAT32_C(   977.87) } },
    { { EASYSIMD_FLOAT32_C(   261.52), EASYSIMD_FLOAT32_C(   593.91), EASYSIMD_FLOAT32_C(   282.09), EASYSIMD_FLOAT32_C(   905.01),
        EASYSIMD_FLOAT32_C(   558.85), EASYSIMD_FLOAT32_C(   546.97), EASYSIMD_FLOAT32_C(    39.41), EASYSIMD_FLOAT32_C(    37.09),
        EASYSIMD_FLOAT32_C(   653.22), EASYSIMD_FLOAT32_C(   550.08), EASYSIMD_FLOAT32_C(   671.18), EASYSIMD_FLOAT32_C(   893.07),
        EASYSIMD_FLOAT32_C(    49.27), EASYSIMD_FLOAT32_C(   666.55), EASYSIMD_FLOAT32_C(    76.85), EASYSIMD_FLOAT32_C(    59.26) },
      UINT8_C( 52),
      { EASYSIMD_FLOAT32_C(   191.70), EASYSIMD_FLOAT32_C(   831.43), EASYSIMD_FLOAT32_C(   284.20), EASYSIMD_FLOAT32_C(   147.82),
        EASYSIMD_FLOAT32_C(   463.91), EASYSIMD_FLOAT32_C(   -90.80), EASYSIMD_FLOAT32_C(   595.96), EASYSIMD_FLOAT32_C(   665.44),
        EASYSIMD_FLOAT32_C(   187.07), EASYSIMD_FLOAT32_C(   126.37), EASYSIMD_FLOAT32_C(   751.70), EASYSIMD_FLOAT32_C(   153.73),
        EASYSIMD_FLOAT32_C(   678.31), EASYSIMD_FLOAT32_C(   781.00), EASYSIMD_FLOAT32_C(   842.34), EASYSIMD_FLOAT32_C(     5.66) },
      { EASYSIMD_FLOAT32_C(   261.52), EASYSIMD_FLOAT32_C(   593.91), EASYSIMD_FLOAT32_C(    16.86), EASYSIMD_FLOAT32_C(   905.01),
        EASYSIMD_FLOAT32_C(    21.54),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    39.41), EASYSIMD_FLOAT32_C(    37.09),
        EASYSIMD_FLOAT32_C(   653.22), EASYSIMD_FLOAT32_C(   550.08), EASYSIMD_FLOAT32_C(   671.18), EASYSIMD_FLOAT32_C(   893.07),
        EASYSIMD_FLOAT32_C(    49.27), EASYSIMD_FLOAT32_C(   666.55), EASYSIMD_FLOAT32_C(    76.85), EASYSIMD_FLOAT32_C(    59.26) } },
    { { EASYSIMD_FLOAT32_C(   370.90), EASYSIMD_FLOAT32_C(   934.09), EASYSIMD_FLOAT32_C(   929.98), EASYSIMD_FLOAT32_C(   111.97),
        EASYSIMD_FLOAT32_C(   630.79), EASYSIMD_FLOAT32_C(   778.41), EASYSIMD_FLOAT32_C(   263.20), EASYSIMD_FLOAT32_C(   298.61),
        EASYSIMD_FLOAT32_C(   360.62), EASYSIMD_FLOAT32_C(   832.32), EASYSIMD_FLOAT32_C(   957.47), EASYSIMD_FLOAT32_C(   168.49),
        EASYSIMD_FLOAT32_C(   294.36), EASYSIMD_FLOAT32_C(   406.95), EASYSIMD_FLOAT32_C(   757.71), EASYSIMD_FLOAT32_C(   992.73) },
      UINT8_C( 43),
      { EASYSIMD_FLOAT32_C(   358.33), EASYSIMD_FLOAT32_C(   783.52), EASYSIMD_FLOAT32_C(   332.05), EASYSIMD_FLOAT32_C(   318.37),
        EASYSIMD_FLOAT32_C(   298.14), EASYSIMD_FLOAT32_C(    66.82), EASYSIMD_FLOAT32_C(   869.43), EASYSIMD_FLOAT32_C(   946.18),
        EASYSIMD_FLOAT32_C(   680.16), EASYSIMD_FLOAT32_C(   120.71), EASYSIMD_FLOAT32_C(   248.65), EASYSIMD_FLOAT32_C(   -79.28),
        EASYSIMD_FLOAT32_C(   590.86), EASYSIMD_FLOAT32_C(   707.03), EASYSIMD_FLOAT32_C(   570.73), EASYSIMD_FLOAT32_C(    84.44) },
      { EASYSIMD_FLOAT32_C(    18.93), EASYSIMD_FLOAT32_C(    27.99), EASYSIMD_FLOAT32_C(   929.98), EASYSIMD_FLOAT32_C(    17.84),
        EASYSIMD_FLOAT32_C(   630.79), EASYSIMD_FLOAT32_C(     8.17), EASYSIMD_FLOAT32_C(   263.20), EASYSIMD_FLOAT32_C(   298.61),
        EASYSIMD_FLOAT32_C(   360.62), EASYSIMD_FLOAT32_C(   832.32), EASYSIMD_FLOAT32_C(   957.47), EASYSIMD_FLOAT32_C(   168.49),
        EASYSIMD_FLOAT32_C(   294.36), EASYSIMD_FLOAT32_C(   406.95), EASYSIMD_FLOAT32_C(   757.71), EASYSIMD_FLOAT32_C(   992.73) } },
    { { EASYSIMD_FLOAT32_C(   -62.52), EASYSIMD_FLOAT32_C(   613.71), EASYSIMD_FLOAT32_C(   789.90), EASYSIMD_FLOAT32_C(   932.36),
        EASYSIMD_FLOAT32_C(   552.83), EASYSIMD_FLOAT32_C(   530.45), EASYSIMD_FLOAT32_C(   607.43), EASYSIMD_FLOAT32_C(   797.28),
        EASYSIMD_FLOAT32_C(   661.83), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   318.54), EASYSIMD_FLOAT32_C(   945.75),
        EASYSIMD_FLOAT32_C(   -58.88), EASYSIMD_FLOAT32_C(   130.26), EASYSIMD_FLOAT32_C(    25.25), EASYSIMD_FLOAT32_C(   627.24) },
      UINT8_C( 40),
      { EASYSIMD_FLOAT32_C(   242.87), EASYSIMD_FLOAT32_C(   758.11), EASYSIMD_FLOAT32_C(    97.46), EASYSIMD_FLOAT32_C(   -58.70),
        EASYSIMD_FLOAT32_C(   972.97), EASYSIMD_FLOAT32_C(   -55.48), EASYSIMD_FLOAT32_C(   199.83), EASYSIMD_FLOAT32_C(    10.44),
        EASYSIMD_FLOAT32_C(   304.24), EASYSIMD_FLOAT32_C(   341.28), EASYSIMD_FLOAT32_C(   281.04), EASYSIMD_FLOAT32_C(   900.87),
        EASYSIMD_FLOAT32_C(   363.46), EASYSIMD_FLOAT32_C(   218.41), EASYSIMD_FLOAT32_C(   940.41), EASYSIMD_FLOAT32_C(   457.09) },
      { EASYSIMD_FLOAT32_C(   -62.52), EASYSIMD_FLOAT32_C(   613.71), EASYSIMD_FLOAT32_C(   789.90),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(   552.83),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   607.43), EASYSIMD_FLOAT32_C(   797.28),
        EASYSIMD_FLOAT32_C(   661.83), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   318.54), EASYSIMD_FLOAT32_C(   945.75),
        EASYSIMD_FLOAT32_C(   -58.88), EASYSIMD_FLOAT32_C(   130.26), EASYSIMD_FLOAT32_C(    25.25), EASYSIMD_FLOAT32_C(   627.24) } },
    { { EASYSIMD_FLOAT32_C(   750.99), EASYSIMD_FLOAT32_C(   296.52), EASYSIMD_FLOAT32_C(   456.51), EASYSIMD_FLOAT32_C(   964.82),
        EASYSIMD_FLOAT32_C(   376.02), EASYSIMD_FLOAT32_C(    -0.77), EASYSIMD_FLOAT32_C(   -19.77), EASYSIMD_FLOAT32_C(   808.40),
        EASYSIMD_FLOAT32_C(   462.68), EASYSIMD_FLOAT32_C(   106.03), EASYSIMD_FLOAT32_C(   864.38), EASYSIMD_FLOAT32_C(   846.10),
        EASYSIMD_FLOAT32_C(   539.67), EASYSIMD_FLOAT32_C(   599.36), EASYSIMD_FLOAT32_C(   551.35), EASYSIMD_FLOAT32_C(   -77.63) },
      UINT8_C( 43),
      { EASYSIMD_FLOAT32_C(   762.24), EASYSIMD_FLOAT32_C(   130.02), EASYSIMD_FLOAT32_C(   518.26), EASYSIMD_FLOAT32_C(   332.17),
        EASYSIMD_FLOAT32_C(   129.59), EASYSIMD_FLOAT32_C(   952.63), EASYSIMD_FLOAT32_C(    71.40), EASYSIMD_FLOAT32_C(   788.60),
        EASYSIMD_FLOAT32_C(   964.30), EASYSIMD_FLOAT32_C(   468.08), EASYSIMD_FLOAT32_C(   636.78), EASYSIMD_FLOAT32_C(   267.82),
        EASYSIMD_FLOAT32_C(   875.24), EASYSIMD_FLOAT32_C(   684.32), EASYSIMD_FLOAT32_C(   694.20), EASYSIMD_FLOAT32_C(   586.14) },
      { EASYSIMD_FLOAT32_C(    27.61), EASYSIMD_FLOAT32_C(    11.40), EASYSIMD_FLOAT32_C(   456.51), EASYSIMD_FLOAT32_C(    18.23),
        EASYSIMD_FLOAT32_C(   376.02), EASYSIMD_FLOAT32_C(    30.86), EASYSIMD_FLOAT32_C(   -19.77), EASYSIMD_FLOAT32_C(   808.40),
        EASYSIMD_FLOAT32_C(   462.68), EASYSIMD_FLOAT32_C(   106.03), EASYSIMD_FLOAT32_C(   864.38), EASYSIMD_FLOAT32_C(   846.10),
        EASYSIMD_FLOAT32_C(   539.67), EASYSIMD_FLOAT32_C(   599.36), EASYSIMD_FLOAT32_C(   551.35), EASYSIMD_FLOAT32_C(   -77.63) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sqrt_ps(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sqrt_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_sqrt_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   475.48), EASYSIMD_FLOAT64_C(   699.80), EASYSIMD_FLOAT64_C(   552.88), EASYSIMD_FLOAT64_C(   673.91),
        EASYSIMD_FLOAT64_C(   591.26), EASYSIMD_FLOAT64_C(   249.70), EASYSIMD_FLOAT64_C(   639.44), EASYSIMD_FLOAT64_C(   997.04) },
      { EASYSIMD_FLOAT64_C(    21.81), EASYSIMD_FLOAT64_C(    26.45), EASYSIMD_FLOAT64_C(    23.51), EASYSIMD_FLOAT64_C(    25.96),
        EASYSIMD_FLOAT64_C(    24.32), EASYSIMD_FLOAT64_C(    15.80), EASYSIMD_FLOAT64_C(    25.29), EASYSIMD_FLOAT64_C(    31.58) } },
    { { EASYSIMD_FLOAT64_C(   727.89), EASYSIMD_FLOAT64_C(   978.25), EASYSIMD_FLOAT64_C(     4.89), EASYSIMD_FLOAT64_C(   693.62),
        EASYSIMD_FLOAT64_C(   611.57), EASYSIMD_FLOAT64_C(   256.31), EASYSIMD_FLOAT64_C(   600.18), EASYSIMD_FLOAT64_C(   836.50) },
      { EASYSIMD_FLOAT64_C(    26.98), EASYSIMD_FLOAT64_C(    31.28), EASYSIMD_FLOAT64_C(     2.21), EASYSIMD_FLOAT64_C(    26.34),
        EASYSIMD_FLOAT64_C(    24.73), EASYSIMD_FLOAT64_C(    16.01), EASYSIMD_FLOAT64_C(    24.50), EASYSIMD_FLOAT64_C(    28.92) } },
    { { EASYSIMD_FLOAT64_C(   214.90), EASYSIMD_FLOAT64_C(   393.95), EASYSIMD_FLOAT64_C(   919.26), EASYSIMD_FLOAT64_C(   432.55),
        EASYSIMD_FLOAT64_C(   371.71), EASYSIMD_FLOAT64_C(   880.26), EASYSIMD_FLOAT64_C(   482.63), EASYSIMD_FLOAT64_C(   601.65) },
      { EASYSIMD_FLOAT64_C(    14.66), EASYSIMD_FLOAT64_C(    19.85), EASYSIMD_FLOAT64_C(    30.32), EASYSIMD_FLOAT64_C(    20.80),
        EASYSIMD_FLOAT64_C(    19.28), EASYSIMD_FLOAT64_C(    29.67), EASYSIMD_FLOAT64_C(    21.97), EASYSIMD_FLOAT64_C(    24.53) } },
    { { EASYSIMD_FLOAT64_C(   234.49), EASYSIMD_FLOAT64_C(   534.45), EASYSIMD_FLOAT64_C(    -6.91), EASYSIMD_FLOAT64_C(   455.17),
        EASYSIMD_FLOAT64_C(   785.50), EASYSIMD_FLOAT64_C(   558.55), EASYSIMD_FLOAT64_C(    29.83), EASYSIMD_FLOAT64_C(   758.42) },
      { EASYSIMD_FLOAT64_C(    15.31), EASYSIMD_FLOAT64_C(    23.12),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    21.33),
        EASYSIMD_FLOAT64_C(    28.03), EASYSIMD_FLOAT64_C(    23.63), EASYSIMD_FLOAT64_C(     5.46), EASYSIMD_FLOAT64_C(    27.54) } },
    { { EASYSIMD_FLOAT64_C(   958.27), EASYSIMD_FLOAT64_C(   519.34), EASYSIMD_FLOAT64_C(   591.49), EASYSIMD_FLOAT64_C(    62.08),
        EASYSIMD_FLOAT64_C(   499.87), EASYSIMD_FLOAT64_C(   535.48), EASYSIMD_FLOAT64_C(    73.76), EASYSIMD_FLOAT64_C(   819.20) },
      { EASYSIMD_FLOAT64_C(    30.96), EASYSIMD_FLOAT64_C(    22.79), EASYSIMD_FLOAT64_C(    24.32), EASYSIMD_FLOAT64_C(     7.88),
        EASYSIMD_FLOAT64_C(    22.36), EASYSIMD_FLOAT64_C(    23.14), EASYSIMD_FLOAT64_C(     8.59), EASYSIMD_FLOAT64_C(    28.62) } },
    { { EASYSIMD_FLOAT64_C(   299.98), EASYSIMD_FLOAT64_C(   211.87), EASYSIMD_FLOAT64_C(    78.11), EASYSIMD_FLOAT64_C(   113.36),
        EASYSIMD_FLOAT64_C(   727.13), EASYSIMD_FLOAT64_C(   252.70), EASYSIMD_FLOAT64_C(   421.79), EASYSIMD_FLOAT64_C(   169.91) },
      { EASYSIMD_FLOAT64_C(    17.32), EASYSIMD_FLOAT64_C(    14.56), EASYSIMD_FLOAT64_C(     8.84), EASYSIMD_FLOAT64_C(    10.65),
        EASYSIMD_FLOAT64_C(    26.97), EASYSIMD_FLOAT64_C(    15.90), EASYSIMD_FLOAT64_C(    20.54), EASYSIMD_FLOAT64_C(    13.04) } },
    { { EASYSIMD_FLOAT64_C(   878.93), EASYSIMD_FLOAT64_C(   333.65), EASYSIMD_FLOAT64_C(   469.80), EASYSIMD_FLOAT64_C(   224.14),
        EASYSIMD_FLOAT64_C(   245.21), EASYSIMD_FLOAT64_C(   905.97), EASYSIMD_FLOAT64_C(   267.17), EASYSIMD_FLOAT64_C(   243.63) },
      { EASYSIMD_FLOAT64_C(    29.65), EASYSIMD_FLOAT64_C(    18.27), EASYSIMD_FLOAT64_C(    21.67), EASYSIMD_FLOAT64_C(    14.97),
        EASYSIMD_FLOAT64_C(    15.66), EASYSIMD_FLOAT64_C(    30.10), EASYSIMD_FLOAT64_C(    16.35), EASYSIMD_FLOAT64_C(    15.61) } },
    { { EASYSIMD_FLOAT64_C(   486.76), EASYSIMD_FLOAT64_C(   343.81), EASYSIMD_FLOAT64_C(   521.65), EASYSIMD_FLOAT64_C(   919.38),
        EASYSIMD_FLOAT64_C(   462.37), EASYSIMD_FLOAT64_C(   489.02), EASYSIMD_FLOAT64_C(   941.81), EASYSIMD_FLOAT64_C(   719.89) },
      { EASYSIMD_FLOAT64_C(    22.06), EASYSIMD_FLOAT64_C(    18.54), EASYSIMD_FLOAT64_C(    22.84), EASYSIMD_FLOAT64_C(    30.32),
        EASYSIMD_FLOAT64_C(    21.50), EASYSIMD_FLOAT64_C(    22.11), EASYSIMD_FLOAT64_C(    30.69), EASYSIMD_FLOAT64_C(    26.83) } },
    { { EASYSIMD_FLOAT64_C(   260.10), EASYSIMD_FLOAT64_C(  2158.90), EASYSIMD_FLOAT64_C(  9449.89), EASYSIMD_FLOAT64_C(  5861.96),
        EASYSIMD_FLOAT64_C(   394.19), EASYSIMD_FLOAT64_C(  3597.89), EASYSIMD_FLOAT64_C(  5033.33), EASYSIMD_FLOAT64_C(  1539.23) },
      { EASYSIMD_FLOAT64_C(    16.13), EASYSIMD_FLOAT64_C(    46.46), EASYSIMD_FLOAT64_C(    97.21), EASYSIMD_FLOAT64_C(    76.56),
        EASYSIMD_FLOAT64_C(    19.85), EASYSIMD_FLOAT64_C(    59.98), EASYSIMD_FLOAT64_C(    70.95), EASYSIMD_FLOAT64_C(    39.23) } }

    };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sqrt_pd(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_sqrt_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sqrt_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   436.97), EASYSIMD_FLOAT64_C(   398.19), EASYSIMD_FLOAT64_C(   907.60), EASYSIMD_FLOAT64_C(    61.33),
        EASYSIMD_FLOAT64_C(   912.86), EASYSIMD_FLOAT64_C(   540.33), EASYSIMD_FLOAT64_C(   579.42), EASYSIMD_FLOAT64_C(   990.91) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT64_C(   499.36), EASYSIMD_FLOAT64_C(   607.00), EASYSIMD_FLOAT64_C(   184.57), EASYSIMD_FLOAT64_C(   -79.89),
        EASYSIMD_FLOAT64_C(   246.08), EASYSIMD_FLOAT64_C(   684.81), EASYSIMD_FLOAT64_C(   154.65), EASYSIMD_FLOAT64_C(   825.63) },
      { EASYSIMD_FLOAT64_C(    22.35), EASYSIMD_FLOAT64_C(   398.19), EASYSIMD_FLOAT64_C(   907.60), EASYSIMD_FLOAT64_C(    61.33),
        EASYSIMD_FLOAT64_C(    15.69), EASYSIMD_FLOAT64_C(   540.33), EASYSIMD_FLOAT64_C(   579.42), EASYSIMD_FLOAT64_C(   990.91) } },
    { { EASYSIMD_FLOAT64_C(   735.36), EASYSIMD_FLOAT64_C(   411.96), EASYSIMD_FLOAT64_C(   273.29), EASYSIMD_FLOAT64_C(   443.97),
        EASYSIMD_FLOAT64_C(   379.78), EASYSIMD_FLOAT64_C(   504.36), EASYSIMD_FLOAT64_C(    13.17), EASYSIMD_FLOAT64_C(    95.38) },
      UINT8_C(184),
      { EASYSIMD_FLOAT64_C(   913.65), EASYSIMD_FLOAT64_C(   567.81), EASYSIMD_FLOAT64_C(   431.31), EASYSIMD_FLOAT64_C(   891.24),
        EASYSIMD_FLOAT64_C(   236.76), EASYSIMD_FLOAT64_C(   364.35), EASYSIMD_FLOAT64_C(   850.12), EASYSIMD_FLOAT64_C(   890.20) },
      { EASYSIMD_FLOAT64_C(   735.36), EASYSIMD_FLOAT64_C(   411.96), EASYSIMD_FLOAT64_C(   273.29), EASYSIMD_FLOAT64_C(    29.85),
        EASYSIMD_FLOAT64_C(    15.39), EASYSIMD_FLOAT64_C(    19.09), EASYSIMD_FLOAT64_C(    13.17), EASYSIMD_FLOAT64_C(    29.84) } },
    { { EASYSIMD_FLOAT64_C(   218.79), EASYSIMD_FLOAT64_C(   849.62), EASYSIMD_FLOAT64_C(   238.02), EASYSIMD_FLOAT64_C(   635.35),
        EASYSIMD_FLOAT64_C(   466.14), EASYSIMD_FLOAT64_C(    -6.77), EASYSIMD_FLOAT64_C(   423.69), EASYSIMD_FLOAT64_C(   491.52) },
      UINT8_C( 45),
      { EASYSIMD_FLOAT64_C(   263.35), EASYSIMD_FLOAT64_C(   539.75), EASYSIMD_FLOAT64_C(   722.58), EASYSIMD_FLOAT64_C(   197.33),
        EASYSIMD_FLOAT64_C(   953.96), EASYSIMD_FLOAT64_C(   549.94), EASYSIMD_FLOAT64_C(   504.50), EASYSIMD_FLOAT64_C(   168.47) },
      { EASYSIMD_FLOAT64_C(    16.23), EASYSIMD_FLOAT64_C(   849.62), EASYSIMD_FLOAT64_C(    26.88), EASYSIMD_FLOAT64_C(    14.05),
        EASYSIMD_FLOAT64_C(   466.14), EASYSIMD_FLOAT64_C(    23.45), EASYSIMD_FLOAT64_C(   423.69), EASYSIMD_FLOAT64_C(   491.52) } },
    { { EASYSIMD_FLOAT64_C(   937.73), EASYSIMD_FLOAT64_C(   521.55), EASYSIMD_FLOAT64_C(   689.62), EASYSIMD_FLOAT64_C(   704.92),
        EASYSIMD_FLOAT64_C(   375.20), EASYSIMD_FLOAT64_C(   562.70), EASYSIMD_FLOAT64_C(   460.68), EASYSIMD_FLOAT64_C(   702.26) },
      UINT8_C( 99),
      { EASYSIMD_FLOAT64_C(   247.65), EASYSIMD_FLOAT64_C(   799.76), EASYSIMD_FLOAT64_C(   469.26), EASYSIMD_FLOAT64_C(   689.97),
        EASYSIMD_FLOAT64_C(   966.92), EASYSIMD_FLOAT64_C(   496.27), EASYSIMD_FLOAT64_C(   437.90), EASYSIMD_FLOAT64_C(   542.79) },
      { EASYSIMD_FLOAT64_C(    15.74), EASYSIMD_FLOAT64_C(    28.28), EASYSIMD_FLOAT64_C(   689.62), EASYSIMD_FLOAT64_C(   704.92),
        EASYSIMD_FLOAT64_C(   375.20), EASYSIMD_FLOAT64_C(    22.28), EASYSIMD_FLOAT64_C(    20.93), EASYSIMD_FLOAT64_C(   702.26) } },
    { { EASYSIMD_FLOAT64_C(   239.42), EASYSIMD_FLOAT64_C(   982.54), EASYSIMD_FLOAT64_C(   153.70), EASYSIMD_FLOAT64_C(   223.51),
        EASYSIMD_FLOAT64_C(   914.90), EASYSIMD_FLOAT64_C(   712.58), EASYSIMD_FLOAT64_C(   479.58), EASYSIMD_FLOAT64_C(   340.52) },
      UINT8_C(110),
      { EASYSIMD_FLOAT64_C(   629.70), EASYSIMD_FLOAT64_C(    40.18), EASYSIMD_FLOAT64_C(   773.21), EASYSIMD_FLOAT64_C(   826.47),
        EASYSIMD_FLOAT64_C(   650.68), EASYSIMD_FLOAT64_C(   597.70), EASYSIMD_FLOAT64_C(    99.58), EASYSIMD_FLOAT64_C(   419.32) },
      { EASYSIMD_FLOAT64_C(   239.42), EASYSIMD_FLOAT64_C(     6.34), EASYSIMD_FLOAT64_C(    27.81), EASYSIMD_FLOAT64_C(    28.75),
        EASYSIMD_FLOAT64_C(   914.90), EASYSIMD_FLOAT64_C(    24.45), EASYSIMD_FLOAT64_C(     9.98), EASYSIMD_FLOAT64_C(   340.52) } },
    { { EASYSIMD_FLOAT64_C(   659.09), EASYSIMD_FLOAT64_C(   166.50), EASYSIMD_FLOAT64_C(   866.73), EASYSIMD_FLOAT64_C(   759.19),
        EASYSIMD_FLOAT64_C(   667.51), EASYSIMD_FLOAT64_C(   836.84), EASYSIMD_FLOAT64_C(   988.98), EASYSIMD_FLOAT64_C(   546.59) },
      UINT8_C(105),
      { EASYSIMD_FLOAT64_C(   223.00), EASYSIMD_FLOAT64_C(   -16.57), EASYSIMD_FLOAT64_C(   745.33), EASYSIMD_FLOAT64_C(   813.45),
        EASYSIMD_FLOAT64_C(   615.39), EASYSIMD_FLOAT64_C(   490.76), EASYSIMD_FLOAT64_C(   133.77), EASYSIMD_FLOAT64_C(   749.00) },
      { EASYSIMD_FLOAT64_C(    14.93), EASYSIMD_FLOAT64_C(   166.50), EASYSIMD_FLOAT64_C(   866.73), EASYSIMD_FLOAT64_C(    28.52),
        EASYSIMD_FLOAT64_C(   667.51), EASYSIMD_FLOAT64_C(    22.15), EASYSIMD_FLOAT64_C(    11.57), EASYSIMD_FLOAT64_C(   546.59) } },
    { { EASYSIMD_FLOAT64_C(   910.80), EASYSIMD_FLOAT64_C(    67.30), EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(   999.85),
        EASYSIMD_FLOAT64_C(   617.26), EASYSIMD_FLOAT64_C(    51.15), EASYSIMD_FLOAT64_C(   319.01), EASYSIMD_FLOAT64_C(    38.40) },
      UINT8_C(223),
      { EASYSIMD_FLOAT64_C(   305.09), EASYSIMD_FLOAT64_C(   369.65), EASYSIMD_FLOAT64_C(   856.43), EASYSIMD_FLOAT64_C(   297.17),
        EASYSIMD_FLOAT64_C(   331.69), EASYSIMD_FLOAT64_C(   827.02), EASYSIMD_FLOAT64_C(   -88.21), EASYSIMD_FLOAT64_C(   -20.46) },
      { EASYSIMD_FLOAT64_C(    17.47), EASYSIMD_FLOAT64_C(    19.23), EASYSIMD_FLOAT64_C(    29.26), EASYSIMD_FLOAT64_C(    17.24),
        EASYSIMD_FLOAT64_C(    18.21), EASYSIMD_FLOAT64_C(    51.15),             EASYSIMD_MATH_NAN,             EASYSIMD_MATH_NAN } },
    { { EASYSIMD_FLOAT64_C(   151.41), EASYSIMD_FLOAT64_C(   225.20), EASYSIMD_FLOAT64_C(   805.52), EASYSIMD_FLOAT64_C(   450.20),
        EASYSIMD_FLOAT64_C(   464.68), EASYSIMD_FLOAT64_C(   748.22), EASYSIMD_FLOAT64_C(   -32.49), EASYSIMD_FLOAT64_C(   491.49) },
      UINT8_C(  9),
      { EASYSIMD_FLOAT64_C(   616.36), EASYSIMD_FLOAT64_C(   262.82), EASYSIMD_FLOAT64_C(   503.00), EASYSIMD_FLOAT64_C(   184.91),
        EASYSIMD_FLOAT64_C(    81.40), EASYSIMD_FLOAT64_C(   665.20), EASYSIMD_FLOAT64_C(   481.84), EASYSIMD_FLOAT64_C(   -81.41) },
      { EASYSIMD_FLOAT64_C(    24.83), EASYSIMD_FLOAT64_C(   225.20), EASYSIMD_FLOAT64_C(   805.52), EASYSIMD_FLOAT64_C(    13.60),
        EASYSIMD_FLOAT64_C(   464.68), EASYSIMD_FLOAT64_C(   748.22), EASYSIMD_FLOAT64_C(   -32.49), EASYSIMD_FLOAT64_C(   491.49) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_loadu_pd(test_vec[i].src);
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sqrt_pd(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sqrt_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sqrt_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sqrt_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sqrt_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sqrt_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sqrt_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sqrt_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sqrt_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sqrt_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sqrt_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sqrt_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sqrt_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sqrt_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
