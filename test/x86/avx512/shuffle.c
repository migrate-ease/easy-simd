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

#define EASYSIMD_TEST_X86_AVX512_INSN shuffle

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/setzero.h>
#include <easysimd/x86/avx512/shuffle.h>

static int
test_easysimd_mm_mask_shuffle_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   238.62), EASYSIMD_FLOAT32_C(   559.91), EASYSIMD_FLOAT32_C(   220.89), EASYSIMD_FLOAT32_C(   269.10) },
      UINT8_C(238),
      { EASYSIMD_FLOAT32_C(   740.96), EASYSIMD_FLOAT32_C(  -960.90), EASYSIMD_FLOAT32_C(  -130.07), EASYSIMD_FLOAT32_C(  -891.01) },
      { EASYSIMD_FLOAT32_C(  -975.82), EASYSIMD_FLOAT32_C(  -158.14), EASYSIMD_FLOAT32_C(   592.48), EASYSIMD_FLOAT32_C(   166.87) },
      { EASYSIMD_FLOAT32_C(   238.62), EASYSIMD_FLOAT32_C(  -891.01), EASYSIMD_FLOAT32_C(  -158.14), EASYSIMD_FLOAT32_C(  -975.82) } },
    { { EASYSIMD_FLOAT32_C(   300.16), EASYSIMD_FLOAT32_C(  -685.37), EASYSIMD_FLOAT32_C(  -119.89), EASYSIMD_FLOAT32_C(   515.96) },
      UINT8_C( 65),
      { EASYSIMD_FLOAT32_C(   746.17), EASYSIMD_FLOAT32_C(  -362.19), EASYSIMD_FLOAT32_C(  -403.06), EASYSIMD_FLOAT32_C(   213.09) },
      { EASYSIMD_FLOAT32_C(   500.67), EASYSIMD_FLOAT32_C(  -933.06), EASYSIMD_FLOAT32_C(   830.93), EASYSIMD_FLOAT32_C(   -83.65) },
      { EASYSIMD_FLOAT32_C(   213.09), EASYSIMD_FLOAT32_C(  -685.37), EASYSIMD_FLOAT32_C(  -119.89), EASYSIMD_FLOAT32_C(   515.96) } },
    { { EASYSIMD_FLOAT32_C(   634.56), EASYSIMD_FLOAT32_C(  -588.22), EASYSIMD_FLOAT32_C(   668.85), EASYSIMD_FLOAT32_C(   -65.28) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(  -451.03), EASYSIMD_FLOAT32_C(  -549.32), EASYSIMD_FLOAT32_C(   358.63), EASYSIMD_FLOAT32_C(  -704.87) },
      { EASYSIMD_FLOAT32_C(    88.49), EASYSIMD_FLOAT32_C(   955.58), EASYSIMD_FLOAT32_C(   508.22), EASYSIMD_FLOAT32_C(  -410.85) },
      { EASYSIMD_FLOAT32_C(  -704.87), EASYSIMD_FLOAT32_C(  -588.22), EASYSIMD_FLOAT32_C(   668.85), EASYSIMD_FLOAT32_C(   -65.28) } },
    { { EASYSIMD_FLOAT32_C(   339.15), EASYSIMD_FLOAT32_C(   505.50), EASYSIMD_FLOAT32_C(   576.19), EASYSIMD_FLOAT32_C(  -986.94) },
      UINT8_C( 95),
      { EASYSIMD_FLOAT32_C(  -452.57), EASYSIMD_FLOAT32_C(   925.59), EASYSIMD_FLOAT32_C(  -648.40), EASYSIMD_FLOAT32_C(  -260.44) },
      { EASYSIMD_FLOAT32_C(   107.22), EASYSIMD_FLOAT32_C(   864.84), EASYSIMD_FLOAT32_C(  -327.35), EASYSIMD_FLOAT32_C(  -672.05) },
      { EASYSIMD_FLOAT32_C(   925.59), EASYSIMD_FLOAT32_C(   925.59), EASYSIMD_FLOAT32_C(  -672.05), EASYSIMD_FLOAT32_C(   107.22) } },
    { { EASYSIMD_FLOAT32_C(   583.60), EASYSIMD_FLOAT32_C(  -792.84), EASYSIMD_FLOAT32_C(  -469.14), EASYSIMD_FLOAT32_C(  -701.68) },
      UINT8_C(190),
      { EASYSIMD_FLOAT32_C(  -996.11), EASYSIMD_FLOAT32_C(  -352.40), EASYSIMD_FLOAT32_C(   764.67), EASYSIMD_FLOAT32_C(   145.77) },
      { EASYSIMD_FLOAT32_C(   931.60), EASYSIMD_FLOAT32_C(  -524.28), EASYSIMD_FLOAT32_C(  -163.38), EASYSIMD_FLOAT32_C(   666.28) },
      { EASYSIMD_FLOAT32_C(   583.60), EASYSIMD_FLOAT32_C(   145.77), EASYSIMD_FLOAT32_C(  -524.28), EASYSIMD_FLOAT32_C(   931.60) } }
  };

  easysimd__m128 src;
  easysimd__mmask8 k;
  easysimd__m128 a;
  easysimd__m128 b;
  easysimd__m128 r;

  src = easysimd_mm_loadu_ps(test_vec[0].src);
  k = test_vec[0].k;
  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_mask_shuffle_ps(src, k, a, b, 30);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[1].src);
  k = test_vec[1].k;
  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_mask_shuffle_ps(src, k, a, b, 15);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[2].src);
  k = test_vec[2].k;
  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_mask_shuffle_ps(src, k, a, b, 67);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[3].src);
  k = test_vec[3].k;
  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_mask_shuffle_ps(src, k, a, b, 53);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[4].src);
  k = test_vec[4].k;
  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r = easysimd_mm_mask_shuffle_ps(src, k, a, b, 30);
  } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_shuffle_ps");
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_random_i8();
    easysimd__m128 r = easysimd_mm_mask_shuffle_ps(src, k, a, b, imm8);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i8(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_shuffle_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(101),
      { EASYSIMD_FLOAT32_C(   630.05), EASYSIMD_FLOAT32_C(   288.07), EASYSIMD_FLOAT32_C(   737.44), EASYSIMD_FLOAT32_C(   218.79) },
      { EASYSIMD_FLOAT32_C(  -137.97), EASYSIMD_FLOAT32_C(   985.88), EASYSIMD_FLOAT32_C(   548.37), EASYSIMD_FLOAT32_C(   393.80) },
      { EASYSIMD_FLOAT32_C(   630.05), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   393.80), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(197),
      { EASYSIMD_FLOAT32_C(   710.22), EASYSIMD_FLOAT32_C(  -236.20), EASYSIMD_FLOAT32_C(   687.54), EASYSIMD_FLOAT32_C(   590.44) },
      { EASYSIMD_FLOAT32_C(  -978.87), EASYSIMD_FLOAT32_C(   625.35), EASYSIMD_FLOAT32_C(  -609.78), EASYSIMD_FLOAT32_C(    37.99) },
      { EASYSIMD_FLOAT32_C(  -236.20), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -978.87), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(176),
      { EASYSIMD_FLOAT32_C(   172.26), EASYSIMD_FLOAT32_C(  -169.35), EASYSIMD_FLOAT32_C(  -657.57), EASYSIMD_FLOAT32_C(   187.13) },
      { EASYSIMD_FLOAT32_C(   904.93), EASYSIMD_FLOAT32_C(   421.20), EASYSIMD_FLOAT32_C(   442.55), EASYSIMD_FLOAT32_C(   534.53) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 52),
      { EASYSIMD_FLOAT32_C(    30.06), EASYSIMD_FLOAT32_C(   645.25), EASYSIMD_FLOAT32_C(   399.49), EASYSIMD_FLOAT32_C(  -232.50) },
      { EASYSIMD_FLOAT32_C(  -135.96), EASYSIMD_FLOAT32_C(  -738.48), EASYSIMD_FLOAT32_C(  -246.63), EASYSIMD_FLOAT32_C(  -587.59) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -135.96), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(203),
      { EASYSIMD_FLOAT32_C(    76.11), EASYSIMD_FLOAT32_C(   365.54), EASYSIMD_FLOAT32_C(  -234.69), EASYSIMD_FLOAT32_C(  -236.35) },
      { EASYSIMD_FLOAT32_C(   -44.03), EASYSIMD_FLOAT32_C(  -213.56), EASYSIMD_FLOAT32_C(  -611.00), EASYSIMD_FLOAT32_C(   346.19) },
      { EASYSIMD_FLOAT32_C(  -236.35), EASYSIMD_FLOAT32_C(  -234.69), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -213.56) } },
    { UINT8_C( 91),
      { EASYSIMD_FLOAT32_C(   737.79), EASYSIMD_FLOAT32_C(  -795.44), EASYSIMD_FLOAT32_C(  -146.93), EASYSIMD_FLOAT32_C(   137.28) },
      { EASYSIMD_FLOAT32_C(   -27.94), EASYSIMD_FLOAT32_C(   717.11), EASYSIMD_FLOAT32_C(   398.79), EASYSIMD_FLOAT32_C(   725.43) },
      { EASYSIMD_FLOAT32_C(  -795.44), EASYSIMD_FLOAT32_C(  -795.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   717.11) } }
  };

  easysimd__mmask8 k;
  easysimd__m128 a;
  easysimd__m128 b;
  easysimd__m128 r;

  k = test_vec[0].k;
  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_maskz_shuffle_ps(k, a, b, 52);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  k = test_vec[1].k;
  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_maskz_shuffle_ps(k, a, b, 77);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  k = test_vec[2].k;
  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_maskz_shuffle_ps(k, a, b, 18);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  k = test_vec[3].k;
  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_maskz_shuffle_ps(k, a, b, 10);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  k = test_vec[4].k;
  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_mm_maskz_shuffle_ps(k, a, b, 123);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  k = test_vec[5].k;
  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r = easysimd_mm_maskz_shuffle_ps(k, a, b, 85);
  } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_shuffle_ps");
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_random_i8();
    easysimd__m128 r = easysimd_mm_maskz_shuffle_ps(k, a, b, imm8);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i8(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shuffle_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C(  56), INT8_C( -94), INT8_C( -41), INT8_C( -59),
                           INT8_C(  40), INT8_C(  78), INT8_C(  93), INT8_C( 107),
                           INT8_C( -47), INT8_C(  66), INT8_C(  -8), INT8_C( -52),
                           INT8_C(  50), INT8_C(  78), INT8_C(  13), INT8_C(  26),
                           INT8_C( -33), INT8_C( 120), INT8_C(  67), INT8_C( -12),
                           INT8_C(  80), INT8_C(  88), INT8_C( -65), INT8_C(  49),
                           INT8_C(  28), INT8_C(  96), INT8_C( -99), INT8_C(-119),
                           INT8_C( 126), INT8_C(  24), INT8_C(  35), INT8_C(  11),
                           INT8_C( -43), INT8_C( -24), INT8_C(   1), INT8_C( -61),
                           INT8_C( 101), INT8_C(   5), INT8_C(  67), INT8_C( -42),
                           INT8_C( -46), INT8_C(-115), INT8_C(-105), INT8_C( -92),
                           INT8_C(  -3), INT8_C( -44), INT8_C(  13), INT8_C( -36),
                           INT8_C( 110), INT8_C( -38), INT8_C( -48), INT8_C(  36),
                           INT8_C( 117), INT8_C( -59), INT8_C( 109), INT8_C( -27),
                           INT8_C( -62), INT8_C(  98), INT8_C(-121), INT8_C(-108),
                           INT8_C(  -7), INT8_C(-112), INT8_C( -52), INT8_C( -84)),
      easysimd_mm512_set_epi8(INT8_C(  34), INT8_C(  34), INT8_C( -71), INT8_C( -63),
                           INT8_C( -34), INT8_C(-128), INT8_C( 113), INT8_C( -72),
                           INT8_C(  86), INT8_C( -73), INT8_C( -43), INT8_C(  95),
                           INT8_C(   3), INT8_C(  13), INT8_C(  19), INT8_C(  -1),
                           INT8_C( -54), INT8_C( 114), INT8_C(  27), INT8_C( -30),
                           INT8_C(  52), INT8_C( -54), INT8_C(  78), INT8_C(  23),
                           INT8_C(  71), INT8_C(  87), INT8_C( 107), INT8_C( -27),
                           INT8_C( -35), INT8_C(-122), INT8_C(  40), INT8_C(  55),
                           INT8_C(  -2), INT8_C(  40), INT8_C( -63), INT8_C(  85),
                           INT8_C( -25), INT8_C( -93), INT8_C(  25), INT8_C(  70),
                           INT8_C( 117), INT8_C(  66), INT8_C( -79), INT8_C(  57),
                           INT8_C(-115), INT8_C( -27), INT8_C(  59), INT8_C(  25),
                           INT8_C(  -6), INT8_C(  42), INT8_C( -27), INT8_C( -34),
                           INT8_C( -16), INT8_C(   5), INT8_C( -55), INT8_C(  74),
                           INT8_C( -29), INT8_C(  77), INT8_C( -17), INT8_C(  16),
                           INT8_C( -98), INT8_C( -38), INT8_C( 116), INT8_C( -56)),
      easysimd_mm512_set_epi8(INT8_C(  78), INT8_C(  78), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  13), INT8_C(   0),
                           INT8_C(  66), INT8_C(   0), INT8_C(   0), INT8_C(  56),
                           INT8_C(  50), INT8_C( -41), INT8_C(  50), INT8_C(   0),
                           INT8_C(   0), INT8_C(  24), INT8_C(  80), INT8_C(   0),
                           INT8_C(-119), INT8_C(   0), INT8_C( 120), INT8_C(  28),
                           INT8_C(  28), INT8_C(  28), INT8_C(  80), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  49), INT8_C(  28),
                           INT8_C(   0), INT8_C( -42), INT8_C(   0), INT8_C(-105),
                           INT8_C(   0), INT8_C(   0), INT8_C(  67), INT8_C(-115),
                           INT8_C(-105), INT8_C( -44), INT8_C(   0), INT8_C(  67),
                           INT8_C(   0), INT8_C(   0), INT8_C( 101), INT8_C(  67),
                           INT8_C(   0), INT8_C( -59), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-121), INT8_C(   0), INT8_C( -59),
                           INT8_C(   0), INT8_C( -48), INT8_C(   0), INT8_C( -84),
                           INT8_C(   0), INT8_C(   0), INT8_C(-108), INT8_C(   0)) },
    { easysimd_mm512_set_epi8(INT8_C( -75), INT8_C(-123), INT8_C( -44), INT8_C( -97),
                           INT8_C(  61), INT8_C(-112), INT8_C( 109), INT8_C(  27),
                           INT8_C(  -9), INT8_C( -31), INT8_C(-111), INT8_C( -59),
                           INT8_C(  94), INT8_C( 110), INT8_C(  31), INT8_C(  50),
                           INT8_C( -86), INT8_C(  24), INT8_C(  91), INT8_C(-105),
                           INT8_C(  39), INT8_C(  25), INT8_C( -65), INT8_C(  69),
                           INT8_C( -59), INT8_C( -57), INT8_C( -98), INT8_C(  26),
                           INT8_C( -35), INT8_C(  -2), INT8_C( 114), INT8_C( -51),
                           INT8_C( -16), INT8_C( 101), INT8_C( -92), INT8_C(  65),
                           INT8_C( -88), INT8_C(  19), INT8_C(-102), INT8_C( -51),
                           INT8_C( -94), INT8_C( -30), INT8_C(  60), INT8_C(  -9),
                           INT8_C(   4), INT8_C( -93), INT8_C( 110), INT8_C(-102),
                           INT8_C( 109), INT8_C(   1), INT8_C( 113), INT8_C( 114),
                           INT8_C( 100), INT8_C(  80), INT8_C(  18), INT8_C(  -7),
                           INT8_C( -87), INT8_C(   0), INT8_C( -55), INT8_C(  37),
                           INT8_C(  13), INT8_C(  86), INT8_C(-114), INT8_C(-125)),
      easysimd_mm512_set_epi8(INT8_C(  18), INT8_C(  11), INT8_C(  73), INT8_C(  57),
                           INT8_C( -54), INT8_C(   9), INT8_C(  69), INT8_C( -45),
                           INT8_C(  71), INT8_C(  -6), INT8_C( -68), INT8_C(  -8),
                           INT8_C(  14), INT8_C( -27), INT8_C(  84), INT8_C( -90),
                           INT8_C( -10), INT8_C( -81), INT8_C(  77), INT8_C(-113),
                           INT8_C(  32), INT8_C(  17), INT8_C(  62), INT8_C( -18),
                           INT8_C( 124), INT8_C( -42), INT8_C( -52), INT8_C(-109),
                           INT8_C( -19), INT8_C(  30), INT8_C( -51), INT8_C(-111),
                           INT8_C( 100), INT8_C(  63), INT8_C( -80), INT8_C(-126),
                           INT8_C(   3), INT8_C( 119), INT8_C(  38), INT8_C( -44),
                           INT8_C( -48), INT8_C( -43), INT8_C(   6), INT8_C(-113),
                           INT8_C(  -5), INT8_C( -18), INT8_C(  58), INT8_C(  77),
                           INT8_C( -49), INT8_C(-107), INT8_C( -77), INT8_C( 119),
                           INT8_C(  -6), INT8_C(  92), INT8_C(-122), INT8_C(  43),
                           INT8_C(  85), INT8_C(-111), INT8_C( -15), INT8_C(  90),
                           INT8_C(-104), INT8_C(  31), INT8_C(  -4), INT8_C(  57)),
      easysimd_mm512_set_epi8(INT8_C( 110), INT8_C(  61), INT8_C( 109), INT8_C( 109),
                           INT8_C(   0), INT8_C( 109), INT8_C(-111), INT8_C(   0),
                           INT8_C(  -9), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-123), INT8_C(   0), INT8_C( -59), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  91), INT8_C(   0),
                           INT8_C( -51), INT8_C( 114), INT8_C(  24), INT8_C(   0),
                           INT8_C(-105), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  24), INT8_C(   0), INT8_C(   0),
                           INT8_C(  -9), INT8_C( -16), INT8_C(   0), INT8_C(   0),
                           INT8_C(   4), INT8_C( -94), INT8_C( -30), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -30), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  19), INT8_C( -92),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -87),
                           INT8_C(   0), INT8_C( 114), INT8_C(   0), INT8_C( 100),
                           INT8_C( -55), INT8_C(   0), INT8_C(   0), INT8_C(  80),
                           INT8_C(   0), INT8_C( 109), INT8_C(   0), INT8_C(  18)) },
    { easysimd_mm512_set_epi8(INT8_C( -54), INT8_C(  53), INT8_C(  35), INT8_C(  51),
                           INT8_C(  24), INT8_C( 100), INT8_C(  80), INT8_C(   7),
                           INT8_C(  54), INT8_C( 114), INT8_C( -42), INT8_C(  -2),
                           INT8_C(  18), INT8_C( 100), INT8_C(  26), INT8_C( -68),
                           INT8_C(  82), INT8_C( -30), INT8_C(  -4), INT8_C( 110),
                           INT8_C(  24), INT8_C(  88), INT8_C(-100), INT8_C(-120),
                           INT8_C( -17), INT8_C(-109), INT8_C(  92), INT8_C(-105),
                           INT8_C( -20), INT8_C(  34), INT8_C(  98), INT8_C( -17),
                           INT8_C(   2), INT8_C(   6), INT8_C(  37), INT8_C( -85),
                           INT8_C( 118), INT8_C(  28), INT8_C(  34), INT8_C( 117),
                           INT8_C(  77), INT8_C( -32), INT8_C(  25), INT8_C( -74),
                           INT8_C(  93), INT8_C(-108), INT8_C(   3), INT8_C( -90),
                           INT8_C(  13), INT8_C( -29), INT8_C( -88), INT8_C( 116),
                           INT8_C(  14), INT8_C( -29), INT8_C(  30), INT8_C( -26),
                           INT8_C(-115), INT8_C( -36), INT8_C(  94), INT8_C(-125),
                           INT8_C( 118), INT8_C( 125), INT8_C( -91), INT8_C(  31)),
      easysimd_mm512_set_epi8(INT8_C( -74), INT8_C( -13), INT8_C( -87), INT8_C( -10),
                           INT8_C(  21), INT8_C(  49), INT8_C( -64), INT8_C(  52),
                           INT8_C( -97), INT8_C(-103), INT8_C(-127), INT8_C(  53),
                           INT8_C(  99), INT8_C(  80), INT8_C(  55), INT8_C( 111),
                           INT8_C(  61), INT8_C( -93), INT8_C( -87), INT8_C(  81),
                           INT8_C( 108), INT8_C(  89), INT8_C(  63), INT8_C(-109),
                           INT8_C( -47), INT8_C( -32), INT8_C( 105), INT8_C(  91),
                           INT8_C(  88), INT8_C(  85), INT8_C( -96), INT8_C(  88),
                           INT8_C( -85), INT8_C(   3), INT8_C( 124), INT8_C( -33),
                           INT8_C( -21), INT8_C(  -7), INT8_C( -31), INT8_C(-126),
                           INT8_C(   3), INT8_C( -17), INT8_C(  40), INT8_C(   5),
                           INT8_C( 126), INT8_C( -60), INT8_C( -91), INT8_C(-112),
                           INT8_C(  88), INT8_C(  11), INT8_C( 100), INT8_C( 114),
                           INT8_C( 112), INT8_C( -53), INT8_C(  89), INT8_C(  78),
                           INT8_C( 115), INT8_C( 117), INT8_C( -33), INT8_C(  14),
                           INT8_C( -42), INT8_C( -15), INT8_C(  80), INT8_C( -46)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -42), INT8_C(  26), INT8_C(   0), INT8_C(  -2),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -42),
                           INT8_C(  18), INT8_C( -68), INT8_C(  54), INT8_C( -54),
                           INT8_C(  -4), INT8_C(   0), INT8_C(   0), INT8_C(  98),
                           INT8_C( 110), INT8_C(-100), INT8_C(  82), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(-100), INT8_C(  24),
                           INT8_C(-120), INT8_C(  92), INT8_C(   0), INT8_C(-120),
                           INT8_C(   0), INT8_C(  93), INT8_C( -85), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  93), INT8_C(   0), INT8_C( 117), INT8_C(  25),
                           INT8_C(   6), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -26), INT8_C(  14), INT8_C(-125), INT8_C( 125),
                           INT8_C(  31), INT8_C(   0), INT8_C(  30), INT8_C( -29),
                           INT8_C( 118), INT8_C(  94), INT8_C(   0), INT8_C( -29),
                           INT8_C(   0), INT8_C(   0), INT8_C(  31), INT8_C(   0)) },
    { easysimd_mm512_set_epi8(INT8_C(-109), INT8_C(  14), INT8_C( -91), INT8_C( -44),
                           INT8_C(  79), INT8_C(  -4), INT8_C(  88), INT8_C(  49),
                           INT8_C( -95), INT8_C( -92), INT8_C(-116), INT8_C(   8),
                           INT8_C(  25), INT8_C(  30), INT8_C( -55), INT8_C(-120),
                           INT8_C( -58), INT8_C( -84), INT8_C(  99), INT8_C( -92),
                           INT8_C( -29), INT8_C( -47), INT8_C(  42), INT8_C( -56),
                           INT8_C( 109), INT8_C( -71), INT8_C( -73), INT8_C( -30),
                           INT8_C( -15), INT8_C(-106), INT8_C( -91), INT8_C( -57),
                           INT8_C(  51), INT8_C(  -8), INT8_C(  55), INT8_C(  36),
                           INT8_C( -55), INT8_C(-103), INT8_C(  51), INT8_C( -28),
                           INT8_C(  70), INT8_C( 114), INT8_C(  -3), INT8_C(  49),
                           INT8_C(  23), INT8_C( -28), INT8_C( -87), INT8_C(  57),
                           INT8_C(  28), INT8_C( -12), INT8_C( -51), INT8_C(  37),
                           INT8_C(  29), INT8_C( -58), INT8_C( -69), INT8_C(  30),
                           INT8_C(  43), INT8_C( -65), INT8_C(  -2), INT8_C( -83),
                           INT8_C( -27), INT8_C(-122), INT8_C(  77), INT8_C(  47)),
      easysimd_mm512_set_epi8(INT8_C(-116), INT8_C(-127), INT8_C( -95), INT8_C( -97),
                           INT8_C( -78), INT8_C( -53), INT8_C( -36), INT8_C(  62),
                           INT8_C(  63), INT8_C(  28), INT8_C( -37), INT8_C(  40),
                           INT8_C( -14), INT8_C(   2), INT8_C( -65), INT8_C( -67),
                           INT8_C(  79), INT8_C(  51), INT8_C( -51), INT8_C( -64),
                           INT8_C( -40), INT8_C(-102), INT8_C(-107), INT8_C(  82),
                           INT8_C(  20), INT8_C(  54), INT8_C(  66), INT8_C(   7),
                           INT8_C(  33), INT8_C( 108), INT8_C( -88), INT8_C(  69),
                           INT8_C(  98), INT8_C(  43), INT8_C(  25), INT8_C(  58),
                           INT8_C( 117), INT8_C( -34), INT8_C( -70), INT8_C(   3),
                           INT8_C(   1), INT8_C( 100), INT8_C(   3), INT8_C(  14),
                           INT8_C( 117), INT8_C( -91), INT8_C( -89), INT8_C(  66),
                           INT8_C(  32), INT8_C( -35), INT8_C( 104), INT8_C(  65),
                           INT8_C(-100), INT8_C(  27), INT8_C(  42), INT8_C(  91),
                           INT8_C( -47), INT8_C(  39), INT8_C( -65), INT8_C(-124),
                           INT8_C( -84), INT8_C( -95), INT8_C( -46), INT8_C(-113)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  14),
                           INT8_C(-109), INT8_C( -44), INT8_C(   0), INT8_C(  49),
                           INT8_C(   0), INT8_C(  30), INT8_C(   0), INT8_C(   0),
                           INT8_C( -58), INT8_C( -15), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(-106),
                           INT8_C( -30), INT8_C( -71), INT8_C(-106), INT8_C( 109),
                           INT8_C( -91), INT8_C( -92), INT8_C(   0), INT8_C( -73),
                           INT8_C( -28), INT8_C( -55), INT8_C(  51), INT8_C(-103),
                           INT8_C(  -3), INT8_C(   0), INT8_C(   0), INT8_C(  23),
                           INT8_C( -87), INT8_C(  49), INT8_C(  23), INT8_C(  -8),
                           INT8_C(  -3), INT8_C(   0), INT8_C(   0), INT8_C( -28),
                           INT8_C(  47), INT8_C(   0), INT8_C(  30), INT8_C(  77),
                           INT8_C(   0), INT8_C(  29), INT8_C( -58), INT8_C(  29),
                           INT8_C(   0), INT8_C(  43), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { easysimd_mm512_set_epi8(INT8_C(-119), INT8_C(  55), INT8_C( -13), INT8_C(-118),
                           INT8_C(   5), INT8_C(  14), INT8_C(-103), INT8_C(  58),
                           INT8_C( 109), INT8_C(-113), INT8_C( -44), INT8_C( 126),
                           INT8_C(  70), INT8_C(-126), INT8_C(  24), INT8_C(-117),
                           INT8_C(  80), INT8_C(  25), INT8_C(  64), INT8_C( -89),
                           INT8_C(  60), INT8_C(   5), INT8_C(-119), INT8_C( -29),
                           INT8_C( -95), INT8_C( -26), INT8_C(  25), INT8_C( -65),
                           INT8_C( -81), INT8_C( -84), INT8_C(  46), INT8_C(  62),
                           INT8_C( -25), INT8_C(-116), INT8_C(-126), INT8_C(-103),
                           INT8_C( 126), INT8_C( -39), INT8_C( 111), INT8_C( -10),
                           INT8_C( -48), INT8_C( 122), INT8_C( -86), INT8_C( -25),
                           INT8_C(  35), INT8_C(  -4), INT8_C(  -7), INT8_C(  76),
                           INT8_C(-116), INT8_C(   1), INT8_C( 126), INT8_C( -30),
                           INT8_C( -11), INT8_C(  82), INT8_C(  81), INT8_C(  27),
                           INT8_C( -15), INT8_C( -14), INT8_C(  56), INT8_C(  60),
                           INT8_C( -29), INT8_C( -65), INT8_C( -35), INT8_C(  21)),
      easysimd_mm512_set_epi8(INT8_C( 125), INT8_C( -43), INT8_C( -42), INT8_C( -51),
                           INT8_C( -36), INT8_C(  38), INT8_C(  42), INT8_C(  18),
                           INT8_C(  24), INT8_C( -25), INT8_C(  19), INT8_C(  15),
                           INT8_C(-119), INT8_C(-124), INT8_C( 104), INT8_C( -93),
                           INT8_C(  22), INT8_C( -16), INT8_C(  31), INT8_C( 116),
                           INT8_C( -99), INT8_C(  45), INT8_C( -54), INT8_C(  50),
                           INT8_C( -72), INT8_C(-107), INT8_C(  48), INT8_C( 118),
                           INT8_C( -68), INT8_C( -15), INT8_C(-108), INT8_C( -29),
                           INT8_C( 118), INT8_C(  96), INT8_C( -51), INT8_C( 107),
                           INT8_C( -23), INT8_C( -97), INT8_C(  -6), INT8_C(  91),
                           INT8_C( -57), INT8_C( -67), INT8_C(-105), INT8_C(  68),
                           INT8_C( -50), INT8_C(  16), INT8_C( -86), INT8_C(   6),
                           INT8_C(  55), INT8_C(-118), INT8_C( 112), INT8_C( 125),
                           INT8_C(  92), INT8_C(  91), INT8_C( -27), INT8_C(  53),
                           INT8_C(  28), INT8_C( -66), INT8_C( -69), INT8_C(-117),
                           INT8_C( -30), INT8_C(-106), INT8_C(   1), INT8_C(  95)),
      easysimd_mm512_set_epi8(INT8_C( -13), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-113), INT8_C(  14), INT8_C(-126),
                           INT8_C(  58), INT8_C(   0), INT8_C(  70), INT8_C(-119),
                           INT8_C(   0), INT8_C(   0), INT8_C(  58), INT8_C(   0),
                           INT8_C( -26), INT8_C(   0), INT8_C(  80), INT8_C( -65),
                           INT8_C(   0), INT8_C(  64), INT8_C(   0), INT8_C( -84),
                           INT8_C(   0), INT8_C(   0), INT8_C(  62), INT8_C( -26),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 122), INT8_C(  76), INT8_C(   0), INT8_C( 126),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( 126),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -25),
                           INT8_C(   0), INT8_C(  76), INT8_C(   0), INT8_C( 122),
                           INT8_C( -15), INT8_C(   0), INT8_C(  21), INT8_C( 126),
                           INT8_C( -30), INT8_C( -11), INT8_C(   0), INT8_C(  56),
                           INT8_C( -30), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -35), INT8_C(-116)) },
    { easysimd_mm512_set_epi8(INT8_C(  18), INT8_C(-124), INT8_C(  -6), INT8_C( -41),
                           INT8_C(-109), INT8_C(  19), INT8_C(  47), INT8_C(  73),
                           INT8_C( -58), INT8_C( -50), INT8_C(  73), INT8_C( 110),
                           INT8_C(  56), INT8_C( -30), INT8_C( -79), INT8_C( 123),
                           INT8_C(  77), INT8_C(  62), INT8_C( -28), INT8_C(  45),
                           INT8_C( -95), INT8_C(  12), INT8_C(  53), INT8_C(  75),
                           INT8_C(  66), INT8_C( -89), INT8_C(  46), INT8_C( -47),
                           INT8_C(   6), INT8_C(   4), INT8_C( 112), INT8_C( -34),
                           INT8_C( 121), INT8_C(  -3), INT8_C(-102), INT8_C(-115),
                           INT8_C( -12), INT8_C(  80), INT8_C(   3), INT8_C(  56),
                           INT8_C(  85), INT8_C( -43), INT8_C( -93), INT8_C(  34),
                           INT8_C( -58), INT8_C(-101), INT8_C(  51), INT8_C(   1),
                           INT8_C( -18), INT8_C( -90), INT8_C( -71), INT8_C(  74),
                           INT8_C(  -5), INT8_C(-113), INT8_C(  71), INT8_C(  61),
                           INT8_C(  92), INT8_C(  73), INT8_C( 116), INT8_C( 110),
                           INT8_C(-117), INT8_C(  96), INT8_C( -77), INT8_C(  45)),
      easysimd_mm512_set_epi8(INT8_C(  52), INT8_C( -84), INT8_C( 103), INT8_C(-105),
                           INT8_C( -57), INT8_C(  17), INT8_C(-110), INT8_C(-113),
                           INT8_C(  56), INT8_C( 125), INT8_C(  56), INT8_C( -88),
                           INT8_C( -31), INT8_C( 119), INT8_C( -86), INT8_C( 114),
                           INT8_C( -85), INT8_C(  14), INT8_C(  60), INT8_C(  84),
                           INT8_C( -61), INT8_C(-111), INT8_C(  74), INT8_C( 113),
                           INT8_C(  13), INT8_C( -89), INT8_C(-114), INT8_C( -79),
                           INT8_C( -34), INT8_C( -73), INT8_C(  -9), INT8_C(-108),
                           INT8_C(  93), INT8_C( -78), INT8_C(  -1), INT8_C(-102),
                           INT8_C(  12), INT8_C( 117), INT8_C(  80), INT8_C(  44),
                           INT8_C(  29), INT8_C(  18), INT8_C(  91), INT8_C( -23),
                           INT8_C( 110), INT8_C(-126), INT8_C( -71), INT8_C(  65),
                           INT8_C( -76), INT8_C(  85), INT8_C(  93), INT8_C( -86),
                           INT8_C(-123), INT8_C(-115), INT8_C(-105), INT8_C(  75),
                           INT8_C(  45), INT8_C(  22), INT8_C(  73), INT8_C(  74),
                           INT8_C( 107), INT8_C( -60), INT8_C( -28), INT8_C( -57)),
      easysimd_mm512_set_epi8(INT8_C( 110), INT8_C(   0), INT8_C( -58), INT8_C(   0),
                           INT8_C(   0), INT8_C( -79), INT8_C(   0), INT8_C(   0),
                           INT8_C(  73), INT8_C(  -6), INT8_C(  73), INT8_C(   0),
                           INT8_C(   0), INT8_C( -58), INT8_C(   0), INT8_C( -30),
                           INT8_C(   0), INT8_C(  62), INT8_C(  45), INT8_C( -47),
                           INT8_C(   0), INT8_C(   0), INT8_C(  12), INT8_C( 112),
                           INT8_C( -28), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-102), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-115), INT8_C( -93), INT8_C(   1), INT8_C(-115),
                           INT8_C(-102), INT8_C(-101), INT8_C( -12), INT8_C(   0),
                           INT8_C(  -3), INT8_C(   0), INT8_C(   0), INT8_C(  51),
                           INT8_C(   0), INT8_C( 116), INT8_C( -71), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  -5),
                           INT8_C( -71), INT8_C(  73), INT8_C(  71), INT8_C(-113),
                           INT8_C(  -5), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { easysimd_mm512_set_epi8(INT8_C(   4), INT8_C( -30), INT8_C( -68), INT8_C(  57),
                           INT8_C(  73), INT8_C(  25), INT8_C( -45), INT8_C( 112),
                           INT8_C(  41), INT8_C(  41), INT8_C( -56), INT8_C( -93),
                           INT8_C(  29), INT8_C(  99), INT8_C(  15), INT8_C( 122),
                           INT8_C(  74), INT8_C(   0), INT8_C(  89), INT8_C(  61),
                           INT8_C( 124), INT8_C(  36), INT8_C( -87), INT8_C( -25),
                           INT8_C(  42), INT8_C(-125), INT8_C( -87), INT8_C(  69),
                           INT8_C(  81), INT8_C(   4), INT8_C(  55), INT8_C( 107),
                           INT8_C(   0), INT8_C(  -7), INT8_C(  37), INT8_C( -15),
                           INT8_C( 117), INT8_C(   9), INT8_C( -27), INT8_C(  11),
                           INT8_C(  16), INT8_C(  87), INT8_C( 119), INT8_C(-107),
                           INT8_C( -22), INT8_C(  27), INT8_C( -86), INT8_C( -63),
                           INT8_C(  90), INT8_C( -44), INT8_C(-113), INT8_C(-126),
                           INT8_C( -75), INT8_C(  67), INT8_C(  88), INT8_C( 105),
                           INT8_C(  80), INT8_C(   7), INT8_C(  66), INT8_C( 126),
                           INT8_C(  17), INT8_C( -20), INT8_C( -91), INT8_C( -99)),
      easysimd_mm512_set_epi8(INT8_C( 123), INT8_C( 117), INT8_C(  18), INT8_C(   2),
                           INT8_C(-105), INT8_C( -51), INT8_C(  -9), INT8_C(  72),
                           INT8_C(  69), INT8_C(-116), INT8_C(  95), INT8_C(-110),
                           INT8_C( -32), INT8_C( 110), INT8_C(  -8), INT8_C(  85),
                           INT8_C( -41), INT8_C(  80), INT8_C( -44), INT8_C( 110),
                           INT8_C( -28), INT8_C( 116), INT8_C(  41), INT8_C( -31),
                           INT8_C(  12), INT8_C(  40), INT8_C(   5), INT8_C(   1),
                           INT8_C( -14), INT8_C(  66), INT8_C( -95), INT8_C( 106),
                           INT8_C(  66), INT8_C( -91), INT8_C( -20), INT8_C(-109),
                           INT8_C( -46), INT8_C(  41), INT8_C(  82), INT8_C( -61),
                           INT8_C(   2), INT8_C(   3), INT8_C(  62), INT8_C( -90),
                           INT8_C(  99), INT8_C( -54), INT8_C( -28), INT8_C( -21),
                           INT8_C( 126), INT8_C( 115), INT8_C( -46), INT8_C(  76),
                           INT8_C( -43), INT8_C(  46), INT8_C( 107), INT8_C(  75),
                           INT8_C( -93), INT8_C(  67), INT8_C( -37), INT8_C(   4),
                           INT8_C(-127), INT8_C( -50), INT8_C( -26), INT8_C(  99)),
      easysimd_mm512_set_epi8(INT8_C(  73), INT8_C( -56), INT8_C(  99), INT8_C(  99),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( 112),
                           INT8_C( -56), INT8_C(   0), INT8_C(   4), INT8_C(   0),
                           INT8_C(   0), INT8_C( -30), INT8_C(   0), INT8_C( -56),
                           INT8_C(   0), INT8_C( 107), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  69), INT8_C( -87), INT8_C(   0),
                           INT8_C(  61), INT8_C( -25), INT8_C( -87), INT8_C(  55),
                           INT8_C(   0), INT8_C(   4), INT8_C(   0), INT8_C(  36),
                           INT8_C(  27), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -27), INT8_C(  27), INT8_C(   0),
                           INT8_C(  27), INT8_C( -22), INT8_C(  -7), INT8_C(   0),
                           INT8_C( -22), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -44), INT8_C(  17), INT8_C(   0), INT8_C(-126),
                           INT8_C(   0), INT8_C( -44), INT8_C( -75), INT8_C( -75),
                           INT8_C(   0), INT8_C(  17), INT8_C(   0), INT8_C( 126),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  17)) },
    { easysimd_mm512_set_epi8(INT8_C(  56), INT8_C(  24), INT8_C( -33), INT8_C(  82),
                           INT8_C(  72), INT8_C(  63), INT8_C(-114), INT8_C(  14),
                           INT8_C(   2), INT8_C(  71), INT8_C( -31), INT8_C( -46),
                           INT8_C(  64), INT8_C( -11), INT8_C(  35), INT8_C(  -7),
                           INT8_C( -78), INT8_C( 125), INT8_C( -28), INT8_C( -59),
                           INT8_C(   6), INT8_C(-127), INT8_C(  34), INT8_C(  64),
                           INT8_C(  88), INT8_C( -88), INT8_C(  70), INT8_C( -90),
                           INT8_C(  47), INT8_C( -21), INT8_C(-104), INT8_C( -27),
                           INT8_C( -16), INT8_C( -29), INT8_C(  51), INT8_C(-116),
                           INT8_C( -87), INT8_C(  91), INT8_C(-106), INT8_C(  -3),
                           INT8_C( -55), INT8_C( -15), INT8_C(-109), INT8_C(-108),
                           INT8_C( -41), INT8_C(  79), INT8_C(  54), INT8_C(  68),
                           INT8_C( -51), INT8_C(-108), INT8_C( -83), INT8_C( 104),
                           INT8_C(  18), INT8_C(-128), INT8_C( -75), INT8_C( -12),
                           INT8_C(  56), INT8_C(-104), INT8_C(  56), INT8_C(  47),
                           INT8_C( -30), INT8_C( -15), INT8_C(-115), INT8_C(  88)),
      easysimd_mm512_set_epi8(INT8_C( 107), INT8_C( 125), INT8_C(-111), INT8_C( -89),
                           INT8_C(  -9), INT8_C( 122), INT8_C(  -2), INT8_C( -26),
                           INT8_C(-100), INT8_C(  80), INT8_C(  89), INT8_C(  38),
                           INT8_C(  95), INT8_C( -82), INT8_C( -66), INT8_C( 108),
                           INT8_C( 104), INT8_C( -20), INT8_C(  72), INT8_C( -82),
                           INT8_C(  23), INT8_C( -45), INT8_C( -20), INT8_C(   8),
                           INT8_C(  -2), INT8_C( -77), INT8_C(-112), INT8_C( 102),
                           INT8_C(  83), INT8_C(  80), INT8_C(  56), INT8_C( -53),
                           INT8_C( -83), INT8_C( -86), INT8_C(  15), INT8_C(  30),
                           INT8_C( -92), INT8_C(  -2), INT8_C(  42), INT8_C( -79),
                           INT8_C( 108), INT8_C(  44), INT8_C(  17), INT8_C(  83),
                           INT8_C( -49), INT8_C( -44), INT8_C( -42), INT8_C( -54),
                           INT8_C(-127), INT8_C(  12), INT8_C(  88), INT8_C(-118),
                           INT8_C( -23), INT8_C( -69), INT8_C( -23), INT8_C( -36),
                           INT8_C( -23), INT8_C( -80), INT8_C(  33), INT8_C(  82),
                           INT8_C( -43), INT8_C( -91), INT8_C( -36), INT8_C( -56)),
      easysimd_mm512_set_epi8(INT8_C(  72), INT8_C( -33), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  63), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  -7), INT8_C(-114), INT8_C(  71),
                           INT8_C(  56), INT8_C(   0), INT8_C(   0), INT8_C(  82),
                           INT8_C(  64), INT8_C(   0), INT8_C(  64), INT8_C(   0),
                           INT8_C(  88), INT8_C(   0), INT8_C(   0), INT8_C(  64),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -88),
                           INT8_C(  47), INT8_C( -27), INT8_C(  64), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -16), INT8_C( -29),
                           INT8_C(   0), INT8_C(   0), INT8_C(  91), INT8_C(   0),
                           INT8_C(-116), INT8_C(-116), INT8_C(  54), INT8_C( -41),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( 104), INT8_C( -12), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(-115), INT8_C( -15),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_shuffle_epi8(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_shuffle_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

#define easysimd_mm512_set_x_epi32(e0, e1, e2, e3, e4, e5,  e6,  e7, e8,  e9,  e10,  e11,  e12,  e13,  e14,  e15) \
        easysimd_mm512_set_epi32((e15), (e14), (e13), (e12), (e11), (e10),  (e9),  (e8), (e7),  (e6),  (e5),  (e4),  (e3),  (e2),  (e1),  (e0))

static int
test_easysimd_mm512_shuffle_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    const int imm8;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_x_epi32( -INT32_C(   879891209),  INT32_C(  1183824997), -INT32_C(    11456019), -INT32_C(  1350161757), -INT32_C(  2087333922), -INT32_C(   594127307),  INT32_C(   632404716), -INT32_C(  1494711121),
      -INT32_C(   713984400),  INT32_C(   538705970), -INT32_C(   736137935),  INT32_C(  1971561879), -INT32_C(  1443358604),  INT32_C(  1535479406), -INT32_C(   125815224),  INT32_C(   715024826) ),
      45,
      easysimd_mm512_set_x_epi32(  INT32_C(  1183824997), -INT32_C(  1350161757), -INT32_C(    11456019), -INT32_C(   879891209), -INT32_C(   594127307), -INT32_C(  1494711121),  INT32_C(   632404716), -INT32_C(  2087333922),
         INT32_C(   538705970),  INT32_C(  1971561879), -INT32_C(   736137935), -INT32_C(   713984400),  INT32_C(  1535479406),  INT32_C(   715024826), -INT32_C(   125815224), -INT32_C(  1443358604) ) },
    { easysimd_mm512_set_x_epi32(  INT32_C(   285216991),  INT32_C(  1110514704),  INT32_C(   538333321),  INT32_C(  1771412213),  INT32_C(   554864050),  INT32_C(  1685886747), -INT32_C(  2007171890),  INT32_C(  1152645733),
      INT32_C(   458666762),  INT32_C(  1482524623), -INT32_C(   847744040), -INT32_C(  1070134003), -INT32_C(  1243526758), -INT32_C(  1357292063), -INT32_C(  1086884519),  INT32_C(  2047076976) ),
      45,
      easysimd_mm512_set_x_epi32(  INT32_C(  1110514704),  INT32_C(  1771412213),  INT32_C(   538333321),  INT32_C(   285216991),  INT32_C(  1685886747),  INT32_C(  1152645733), -INT32_C(  2007171890),  INT32_C(   554864050),
      INT32_C(  1482524623), -INT32_C(  1070134003), -INT32_C(   847744040),  INT32_C(   458666762), -INT32_C(  1357292063),  INT32_C(  2047076976), -INT32_C(  1086884519), -INT32_C(  1243526758) ) },
    { easysimd_mm512_set_x_epi32(  INT32_C(  1821727133), -INT32_C(  1178275103),  INT32_C(  1938177126), -INT32_C(   483148471), -INT32_C(   392621049), -INT32_C(   879250831), -INT32_C(  1735733720),  INT32_C(  1444056505),
        -INT32_C(   926701337),  INT32_C(     8488858),  INT32_C(   225707972), -INT32_C(   856578107),  INT32_C(   766806716),  INT32_C(  1693993788), -INT32_C(   755203559), -INT32_C(   148304112) ),
      45,
      easysimd_mm512_set_x_epi32( -INT32_C(  1178275103), -INT32_C(   483148471),  INT32_C(  1938177126),  INT32_C(  1821727133), -INT32_C(   879250831),  INT32_C(  1444056505), -INT32_C(  1735733720), -INT32_C(   392621049),
         INT32_C(     8488858), -INT32_C(   856578107),  INT32_C(   225707972), -INT32_C(   926701337),  INT32_C(  1693993788), -INT32_C(   148304112), -INT32_C(   755203559),  INT32_C(   766806716) ) },
    { easysimd_mm512_set_x_epi32(  INT32_C(  1371532471),  INT32_C(   928137331),  INT32_C(   205899335),  INT32_C(   702035565), -INT32_C(    61371200), -INT32_C(   245280809), -INT32_C(   490512942),  INT32_C(   601484396),
         INT32_C(  1282775256),  INT32_C(   528730072), -INT32_C(    97728371), -INT32_C(  1104935682),  INT32_C(  1757117329), -INT32_C(  1671881526), -INT32_C(   427942535), -INT32_C(   502704375) ),
      45,
      easysimd_mm512_set_x_epi32(  INT32_C(   928137331),  INT32_C(   702035565),  INT32_C(   205899335),  INT32_C(  1371532471), -INT32_C(   245280809),  INT32_C(   601484396), -INT32_C(   490512942), -INT32_C(    61371200),
         INT32_C(   528730072), -INT32_C(  1104935682), -INT32_C(    97728371),  INT32_C(  1282775256), -INT32_C(  1671881526), -INT32_C(   502704375), -INT32_C(   427942535),  INT32_C(  1757117329) ) },
    { easysimd_mm512_set_x_epi32( -INT32_C(   953254161), -INT32_C(   756567739),  INT32_C(  2026705786), -INT32_C(  1472728809),  INT32_C(   907145836), -INT32_C(  2016187890), -INT32_C(  1855106681), -INT32_C(  1737263191),
         INT32_C(   996188661), -INT32_C(   871545006),  INT32_C(  1900403290),  INT32_C(   924482763),  INT32_C(  2087594862),  INT32_C(   486752661),  INT32_C(  1001288082), -INT32_C(   573300248) ),
      45,
      easysimd_mm512_set_x_epi32( -INT32_C(   756567739), -INT32_C(  1472728809),  INT32_C(  2026705786), -INT32_C(   953254161), -INT32_C(  2016187890), -INT32_C(  1737263191), -INT32_C(  1855106681),  INT32_C(   907145836),
        -INT32_C(   871545006),  INT32_C(   924482763),  INT32_C(  1900403290),  INT32_C(   996188661),  INT32_C(   486752661), -INT32_C(   573300248),  INT32_C(  1001288082),  INT32_C(  2087594862) ) },
    { easysimd_mm512_set_x_epi32(  INT32_C(   337130690), -INT32_C(   706664837), -INT32_C(   867817984),  INT32_C(   268656802),  INT32_C(   562852235),  INT32_C(  1161727922), -INT32_C(   394204160), -INT32_C(   809085939),
         INT32_C(    65265288),  INT32_C(    98092036), -INT32_C(  1932451862),  INT32_C(   194827391), -INT32_C(   131323834), -INT32_C(  1203934536),  INT32_C(  1671544406), -INT32_C(  1691195630) ),
      45,
      easysimd_mm512_set_x_epi32( -INT32_C(   706664837),  INT32_C(   268656802), -INT32_C(   867817984),  INT32_C(   337130690),  INT32_C(  1161727922), -INT32_C(   809085939), -INT32_C(   394204160),  INT32_C(   562852235),
         INT32_C(    98092036),  INT32_C(   194827391), -INT32_C(  1932451862),  INT32_C(    65265288), -INT32_C(  1203934536), -INT32_C(  1691195630),  INT32_C(  1671544406), -INT32_C(   131323834) ) },
    { easysimd_mm512_set_x_epi32(  INT32_C(  1251874117), -INT32_C(  1001424934),  INT32_C(   357638294),  INT32_C(   975236596), -INT32_C(   835498986),  INT32_C(   210137270),  INT32_C(  1097803566), -INT32_C(   723738226),
        -INT32_C(  1860273481), -INT32_C(  2024444431), -INT32_C(  2120374643), -INT32_C(  1430471276), -INT32_C(  1065816310), -INT32_C(  1899102625), -INT32_C(  1261487066), -INT32_C(  1786205218) ),
      45,
      easysimd_mm512_set_x_epi32( -INT32_C(  1001424934),  INT32_C(   975236596),  INT32_C(   357638294),  INT32_C(  1251874117),  INT32_C(   210137270), -INT32_C(   723738226),  INT32_C(  1097803566), -INT32_C(   835498986),
        -INT32_C(  2024444431), -INT32_C(  1430471276), -INT32_C(  2120374643), -INT32_C(  1860273481), -INT32_C(  1899102625), -INT32_C(  1786205218), -INT32_C(  1261487066), -INT32_C(  1065816310) ) },
    { easysimd_mm512_set_x_epi32(  INT32_C(   388408869), -INT32_C(  1600226541), -INT32_C(  1239270622),  INT32_C(    56680185),  INT32_C(   751032525), -INT32_C(    54882090), -INT32_C(  1414428211),  INT32_C(  1514158388),
        -INT32_C(   210671904),  INT32_C(    60035041),  INT32_C(  1153021515),  INT32_C(  1632049556), -INT32_C(   930280719),  INT32_C(  1757694107),  INT32_C(   101938641), -INT32_C(  1889512529) ),
      45,
      easysimd_mm512_set_x_epi32( -INT32_C(  1600226541),  INT32_C(    56680185), -INT32_C(  1239270622),  INT32_C(   388408869), -INT32_C(    54882090),  INT32_C(  1514158388), -INT32_C(  1414428211),  INT32_C(   751032525),
         INT32_C(    60035041),  INT32_C(  1632049556),  INT32_C(  1153021515), -INT32_C(   210671904),  INT32_C(  1757694107), -INT32_C(  1889512529),  INT32_C(   101938641), -INT32_C(   930280719) ) },

 };
 
  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_shuffle_epi32(a, 45);
    } EASYSIMD_TEST_PERF_END("_mm512_shuffle_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}



static int
test_easysimd_mm512_mask_shuffle_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C(  92), INT8_C( 116), INT8_C( -78), INT8_C( -19),
                           INT8_C( -73), INT8_C(  22), INT8_C( -66), INT8_C( -29),
                           INT8_C(  55), INT8_C(  78), INT8_C( -45), INT8_C(-119),
                           INT8_C(  30), INT8_C( -29), INT8_C( -23), INT8_C(  89),
                           INT8_C(  15), INT8_C( 119), INT8_C( -88), INT8_C( 109),
                           INT8_C( -16), INT8_C( -10), INT8_C(   0), INT8_C( -18),
                           INT8_C(   1), INT8_C( 116), INT8_C(  62), INT8_C( -73),
                           INT8_C(-122), INT8_C(   0), INT8_C( 111), INT8_C(  95),
                           INT8_C( -20), INT8_C(   3), INT8_C( 122), INT8_C(-121),
                           INT8_C(  13), INT8_C(  25), INT8_C( -35), INT8_C(-127),
                           INT8_C(  82), INT8_C(  90), INT8_C(  53), INT8_C( 123),
                           INT8_C(  73), INT8_C( 108), INT8_C( -18), INT8_C(  15),
                           INT8_C(  81), INT8_C( -27), INT8_C( 102), INT8_C( 118),
                           INT8_C(  39), INT8_C( -77), INT8_C(  45), INT8_C(  81),
                           INT8_C( -17), INT8_C( -28), INT8_C(  67), INT8_C(-118),
                           INT8_C(  79), INT8_C(-113), INT8_C(-122), INT8_C( 124)),
      UINT64_C(10224647434006242820),
      easysimd_mm512_set_epi8(INT8_C(  68), INT8_C(-108), INT8_C(-121), INT8_C(-115),
                           INT8_C( -59), INT8_C( -78), INT8_C(-111), INT8_C(  56),
                           INT8_C(   9), INT8_C(-121), INT8_C( 120), INT8_C( -36),
                           INT8_C(  36), INT8_C(-103), INT8_C(-104), INT8_C(  25),
                           INT8_C(-111), INT8_C( -63), INT8_C( -35), INT8_C(-120),
                           INT8_C(  28), INT8_C( -44), INT8_C( -26), INT8_C( -86),
                           INT8_C( -13), INT8_C(  53), INT8_C( -88), INT8_C(-107),
                           INT8_C(  68), INT8_C(  42), INT8_C(-118), INT8_C( 111),
                           INT8_C(  54), INT8_C( -58), INT8_C( -13), INT8_C(  27),
                           INT8_C(  23), INT8_C(  41), INT8_C(-119), INT8_C(  44),
                           INT8_C(   7), INT8_C(-120), INT8_C(  32), INT8_C( -43),
                           INT8_C( 114), INT8_C( -72), INT8_C(  73), INT8_C( -96),
                           INT8_C(  96), INT8_C( 110), INT8_C( -81), INT8_C( -76),
                           INT8_C( 103), INT8_C(-100), INT8_C( -22), INT8_C(  18),
                           INT8_C( 115), INT8_C(  54), INT8_C( -40), INT8_C( 125),
                           INT8_C( 110), INT8_C(  31), INT8_C(  51), INT8_C(-104)),
      easysimd_mm512_set_epi8(INT8_C( -14), INT8_C(  85), INT8_C( -80), INT8_C(  80),
                           INT8_C(  48), INT8_C(  93), INT8_C(  79), INT8_C( 127),
                           INT8_C(  16), INT8_C(  41), INT8_C(  54), INT8_C(-116),
                           INT8_C(  17), INT8_C(  42), INT8_C( -86), INT8_C(  38),
                           INT8_C(-118), INT8_C( -56), INT8_C(  60), INT8_C(  19),
                           INT8_C(-100), INT8_C(-107), INT8_C( 105), INT8_C( -76),
                           INT8_C(  10), INT8_C(  -9), INT8_C( -12), INT8_C( -56),
                           INT8_C( -71), INT8_C(  96), INT8_C(   6), INT8_C(  24),
                           INT8_C(  69), INT8_C(  73), INT8_C( -27), INT8_C(   3),
                           INT8_C( -95), INT8_C(  50), INT8_C( -67), INT8_C( -39),
                           INT8_C(  97), INT8_C(  62), INT8_C( 101), INT8_C(  56),
                           INT8_C( -79), INT8_C(  83), INT8_C( -99), INT8_C( -56),
                           INT8_C( -17), INT8_C( -56), INT8_C(  -8), INT8_C( -16),
                           INT8_C(  84), INT8_C( -95), INT8_C( -73), INT8_C(  54),
                           INT8_C( 125), INT8_C(  85), INT8_C(  78), INT8_C( -65),
                           INT8_C(   4), INT8_C( 113), INT8_C( -16), INT8_C( -53)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C( 116), INT8_C( -78), INT8_C( -19),
                           INT8_C(  25), INT8_C(-121), INT8_C( -66), INT8_C(  68),
                           INT8_C(  25), INT8_C(-111), INT8_C(-121), INT8_C(-119),
                           INT8_C(  30), INT8_C( -78), INT8_C( -23), INT8_C(-121),
                           INT8_C(  15), INT8_C( 119), INT8_C(-120), INT8_C(  68),
                           INT8_C(   0), INT8_C(   0), INT8_C( -26), INT8_C( -18),
                           INT8_C( -44), INT8_C( 116), INT8_C(   0), INT8_C( -73),
                           INT8_C(   0), INT8_C(   0), INT8_C(  53), INT8_C( -86),
                           INT8_C( -20), INT8_C(   3), INT8_C(   0), INT8_C( 114),
                           INT8_C(  13), INT8_C(  25), INT8_C( -35), INT8_C(   0),
                           INT8_C(  73), INT8_C(  90), INT8_C(  53), INT8_C( 123),
                           INT8_C(   0), INT8_C( 114), INT8_C( -18), INT8_C(   0),
                           INT8_C(  81), INT8_C( -27), INT8_C(   0), INT8_C( 118),
                           INT8_C(  39), INT8_C(   0), INT8_C(   0), INT8_C(  81),
                           INT8_C( -17), INT8_C( -28), INT8_C(  67), INT8_C(-118),
                           INT8_C(  79), INT8_C(  51), INT8_C(-122), INT8_C( 124)) },
    { easysimd_mm512_set_epi8(INT8_C(-123), INT8_C(-100), INT8_C( -39), INT8_C( -44),
                           INT8_C(  22), INT8_C( 112), INT8_C(  16), INT8_C(  15),
                           INT8_C(-104), INT8_C(-102), INT8_C( 102), INT8_C( -95),
                           INT8_C(  69), INT8_C(  13), INT8_C( -21), INT8_C(  45),
                           INT8_C(  62), INT8_C(-120), INT8_C(-111), INT8_C(  32),
                           INT8_C(-107), INT8_C( -30), INT8_C(  99), INT8_C( -64),
                           INT8_C(   8), INT8_C( -42), INT8_C(  81), INT8_C( -34),
                           INT8_C( -46), INT8_C(  26), INT8_C(  31), INT8_C(  -2),
                           INT8_C(  68), INT8_C(  -7), INT8_C( -71), INT8_C(  46),
                           INT8_C( -21), INT8_C( -73), INT8_C(  21), INT8_C(  83),
                           INT8_C(-108), INT8_C( -97), INT8_C( -69), INT8_C(  73),
                           INT8_C(  57), INT8_C( -37), INT8_C(  21), INT8_C(  82),
                           INT8_C(-119), INT8_C(-126), INT8_C( 126), INT8_C(  91),
                           INT8_C( 115), INT8_C(  31), INT8_C( -79), INT8_C(  28),
                           INT8_C(-106), INT8_C( -18), INT8_C(  65), INT8_C(-104),
                           INT8_C(  81), INT8_C(  38), INT8_C( -84), INT8_C(  -2)),
      UINT64_C( 4597426592773770833),
      easysimd_mm512_set_epi8(INT8_C(  74), INT8_C(  72), INT8_C(  -4), INT8_C(  28),
                           INT8_C(-115), INT8_C(  93), INT8_C( 102), INT8_C(  44),
                           INT8_C(-103), INT8_C( -29), INT8_C( -50), INT8_C(  48),
                           INT8_C( -96), INT8_C( -50), INT8_C(  46), INT8_C( -65),
                           INT8_C(   4), INT8_C(  43), INT8_C( -75), INT8_C(  97),
                           INT8_C( -26), INT8_C(-103), INT8_C(  71), INT8_C(-107),
                           INT8_C(  91), INT8_C(  45), INT8_C( -11), INT8_C(  47),
                           INT8_C(  29), INT8_C(  25), INT8_C(  26), INT8_C(  -9),
                           INT8_C(  10), INT8_C(  36), INT8_C( -79), INT8_C( -53),
                           INT8_C(  41), INT8_C(   1), INT8_C( -23), INT8_C( -63),
                           INT8_C(-127), INT8_C(  68), INT8_C(  48), INT8_C(  36),
                           INT8_C(  89), INT8_C(-112), INT8_C( -31), INT8_C( 120),
                           INT8_C(  35), INT8_C(  62), INT8_C( -21), INT8_C(-114),
                           INT8_C(-104), INT8_C(  57), INT8_C(  42), INT8_C(-111),
                           INT8_C(  94), INT8_C( -63), INT8_C(  -9), INT8_C(  64),
                           INT8_C( -65), INT8_C(  -2), INT8_C( 110), INT8_C(  -8)),
      easysimd_mm512_set_epi8(INT8_C(  44), INT8_C( -92), INT8_C( -31), INT8_C(  26),
                           INT8_C( -99), INT8_C( -53), INT8_C( 117), INT8_C(  18),
                           INT8_C( -63), INT8_C(  45), INT8_C(  12), INT8_C(  24),
                           INT8_C(-108), INT8_C(  18), INT8_C( -60), INT8_C(  28),
                           INT8_C(  50), INT8_C( -11), INT8_C( -68), INT8_C( -31),
                           INT8_C( 105), INT8_C(-106), INT8_C(  98), INT8_C(  51),
                           INT8_C(  58), INT8_C( 103), INT8_C( 111), INT8_C(-127),
                           INT8_C(  68), INT8_C( -56), INT8_C( 124), INT8_C(-119),
                           INT8_C(  74), INT8_C( -62), INT8_C(-116), INT8_C(  37),
                           INT8_C( -12), INT8_C( 114), INT8_C(   0), INT8_C(  61),
                           INT8_C( 103), INT8_C(  -4), INT8_C(-105), INT8_C( -68),
                           INT8_C(  39), INT8_C(-100), INT8_C( -93), INT8_C(  11),
                           INT8_C( -80), INT8_C( -19), INT8_C( -22), INT8_C( -39),
                           INT8_C( 127), INT8_C( -38), INT8_C(-125), INT8_C(-111),
                           INT8_C(  84), INT8_C( -96), INT8_C(  87), INT8_C( -22),
                           INT8_C(  -5), INT8_C(  -3), INT8_C(-127), INT8_C(  41)),
      easysimd_mm512_set_epi8(INT8_C(-123), INT8_C(-100), INT8_C(   0), INT8_C(  93),
                           INT8_C(   0), INT8_C(   0), INT8_C( -50), INT8_C( -50),
                           INT8_C(   0), INT8_C(  -4), INT8_C( 102), INT8_C( -95),
                           INT8_C(   0), INT8_C( -50), INT8_C( -21), INT8_C(  28),
                           INT8_C(  62), INT8_C(   0), INT8_C(-111), INT8_C(   0),
                           INT8_C(-107), INT8_C(   0), INT8_C(  25), INT8_C(  29),
                           INT8_C(   8), INT8_C( -42), INT8_C(   4), INT8_C( -34),
                           INT8_C( -46), INT8_C(  26), INT8_C(  31), INT8_C(  -2),
                           INT8_C(   1), INT8_C(  -7), INT8_C(   0), INT8_C(  48),
                           INT8_C(   0), INT8_C(-112), INT8_C( 120), INT8_C( -79),
                           INT8_C(-108), INT8_C( -97), INT8_C(   0), INT8_C(   0),
                           INT8_C(  57), INT8_C(   0), INT8_C(   0), INT8_C(  41),
                           INT8_C(-119), INT8_C(-126), INT8_C( 126), INT8_C(  91),
                           INT8_C(  35), INT8_C(   0), INT8_C(   0), INT8_C(  28),
                           INT8_C(-106), INT8_C(   0), INT8_C(  65), INT8_C(   0),
                           INT8_C(  81), INT8_C(  38), INT8_C( -84), INT8_C(  42)) },
    { easysimd_mm512_set_epi8(INT8_C( -30), INT8_C( -37), INT8_C(  51), INT8_C( -36),
                           INT8_C(   8), INT8_C(  52), INT8_C(  97), INT8_C( 123),
                           INT8_C( -49), INT8_C(-124), INT8_C(  95), INT8_C( -83),
                           INT8_C(  70), INT8_C( -50), INT8_C( -61), INT8_C(  25),
                           INT8_C( -97), INT8_C(  28), INT8_C( -58), INT8_C(  11),
                           INT8_C( -14), INT8_C( 126), INT8_C(  81), INT8_C(  45),
                           INT8_C( -23), INT8_C( 120), INT8_C( -83), INT8_C( -16),
                           INT8_C(   7), INT8_C(  51), INT8_C( -57), INT8_C( -50),
                           INT8_C( -21), INT8_C(  98), INT8_C(  88), INT8_C(   0),
                           INT8_C( -66), INT8_C(   3), INT8_C( 124), INT8_C(-113),
                           INT8_C(  50), INT8_C(  88), INT8_C( -85), INT8_C( -93),
                           INT8_C( -44), INT8_C( -13), INT8_C( -94), INT8_C(  17),
                           INT8_C(  -2), INT8_C(  79), INT8_C(-116), INT8_C(  43),
                           INT8_C( -77), INT8_C(-125), INT8_C( -23), INT8_C(-120),
                           INT8_C(  96), INT8_C( -64), INT8_C( -23), INT8_C( -46),
                           INT8_C( -29), INT8_C( -71), INT8_C(  71), INT8_C(  90)),
      UINT64_C(12627002542648829104),
      easysimd_mm512_set_epi8(INT8_C(  56), INT8_C( -45), INT8_C(-108), INT8_C( -19),
                           INT8_C(-124), INT8_C( -27), INT8_C(  22), INT8_C( 126),
                           INT8_C(-106), INT8_C( -68), INT8_C( -60), INT8_C(   8),
                           INT8_C(  60), INT8_C(  93), INT8_C( -33), INT8_C( -27),
                           INT8_C(  -7), INT8_C(  27), INT8_C(-122), INT8_C( -38),
                           INT8_C(  23), INT8_C(   6), INT8_C(  45), INT8_C( -21),
                           INT8_C( -23), INT8_C(-101), INT8_C( 116), INT8_C( 127),
                           INT8_C(  96), INT8_C(  40), INT8_C( -97), INT8_C(  40),
                           INT8_C(  86), INT8_C( -44), INT8_C(  70), INT8_C( -71),
                           INT8_C(  62), INT8_C( -21), INT8_C(  66), INT8_C(  68),
                           INT8_C( -87), INT8_C( -61), INT8_C(  48), INT8_C( -70),
                           INT8_C(  18), INT8_C( -78), INT8_C( -98), INT8_C( 117),
                           INT8_C(  74), INT8_C(  32), INT8_C(  93), INT8_C( 125),
                           INT8_C( -47), INT8_C( -60), INT8_C( -86), INT8_C( 117),
                           INT8_C( 122), INT8_C( -54), INT8_C(  50), INT8_C( 123),
                           INT8_C( -31), INT8_C( -74), INT8_C( -64), INT8_C(  54)),
      easysimd_mm512_set_epi8(INT8_C( 115), INT8_C(  51), INT8_C( -91), INT8_C(  56),
                           INT8_C(  64), INT8_C( -39), INT8_C(-119), INT8_C( -28),
                           INT8_C( -54), INT8_C(  28), INT8_C(  54), INT8_C(  -8),
                           INT8_C( -54), INT8_C(-128), INT8_C( -28), INT8_C( -71),
                           INT8_C( 107), INT8_C( -66), INT8_C(-114), INT8_C( -88),
                           INT8_C(  34), INT8_C( -83), INT8_C( -21), INT8_C( -64),
                           INT8_C( 121), INT8_C( -20), INT8_C( -89), INT8_C( -94),
                           INT8_C( 112), INT8_C( -27), INT8_C(  81), INT8_C( -54),
                           INT8_C( -64), INT8_C(-114), INT8_C(  48), INT8_C( -89),
                           INT8_C( -61), INT8_C(  26), INT8_C(  43), INT8_C(  29),
                           INT8_C(   0), INT8_C(  64), INT8_C( 123), INT8_C( -67),
                           INT8_C(  15), INT8_C( 120), INT8_C(  36), INT8_C(  40),
                           INT8_C( 106), INT8_C(-118), INT8_C(-108), INT8_C( -58),
                           INT8_C(  26), INT8_C(-111), INT8_C(  63), INT8_C( -98),
                           INT8_C( -13), INT8_C( -12), INT8_C(-124), INT8_C(  96),
                           INT8_C( -13), INT8_C( -98), INT8_C(  99), INT8_C( -13)),
      easysimd_mm512_set_epi8(INT8_C(  60), INT8_C( -37), INT8_C(   0), INT8_C( -36),
                           INT8_C( -27), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -49), INT8_C(-124), INT8_C( -68), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -61), INT8_C(  25),
                           INT8_C( -97), INT8_C(  28), INT8_C( -58), INT8_C(   0),
                           INT8_C(  40), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  45), INT8_C(   0), INT8_C(   0), INT8_C( -16),
                           INT8_C(  40), INT8_C(  51), INT8_C( -57), INT8_C(   0),
                           INT8_C( -21), INT8_C(   0), INT8_C( 117), INT8_C(   0),
                           INT8_C(   0), INT8_C( -21), INT8_C( 124), INT8_C(-113),
                           INT8_C( 117), INT8_C(  88), INT8_C( -85), INT8_C( -93),
                           INT8_C(  86), INT8_C( -13), INT8_C( -94), INT8_C(  68),
                           INT8_C( -60), INT8_C(  79), INT8_C(   0), INT8_C(  43),
                           INT8_C( -77), INT8_C(   0), INT8_C( -23), INT8_C(-120),
                           INT8_C(   0), INT8_C( -64), INT8_C(   0), INT8_C(  54),
                           INT8_C( -29), INT8_C( -71), INT8_C(  71), INT8_C(  90)) },
    { easysimd_mm512_set_epi8(INT8_C(  72), INT8_C(  45), INT8_C( 120), INT8_C(  -5),
                           INT8_C(-109), INT8_C(  62), INT8_C(  17), INT8_C(  31),
                           INT8_C( -30), INT8_C( -58), INT8_C(  56), INT8_C(  21),
                           INT8_C(  72), INT8_C( -74), INT8_C( -40), INT8_C( 120),
                           INT8_C(  95), INT8_C( 108), INT8_C(  32), INT8_C(  64),
                           INT8_C(-128), INT8_C( 102), INT8_C( -21), INT8_C(  28),
                           INT8_C( 105), INT8_C(  52), INT8_C(  85), INT8_C(-104),
                           INT8_C(  57), INT8_C( -31), INT8_C( -38), INT8_C(-124),
                           INT8_C(-107), INT8_C(  -2), INT8_C(  55), INT8_C(  46),
                           INT8_C( -71), INT8_C(  77), INT8_C(  18), INT8_C(  70),
                           INT8_C(  89), INT8_C( 125), INT8_C( -42), INT8_C(-125),
                           INT8_C( 121), INT8_C( -11), INT8_C( -69), INT8_C( -59),
                           INT8_C( -53), INT8_C(  34), INT8_C(   9), INT8_C(  64),
                           INT8_C( -61), INT8_C( -25), INT8_C(-115), INT8_C( 100),
                           INT8_C(  65), INT8_C(   8), INT8_C(  69), INT8_C(  -8),
                           INT8_C( -15), INT8_C( -51), INT8_C(   1), INT8_C(  90)),
      UINT64_C(14515151237088493607),
      easysimd_mm512_set_epi8(INT8_C( -26), INT8_C(  -9), INT8_C(  66), INT8_C(   1),
                           INT8_C( -13), INT8_C(  60), INT8_C(-119), INT8_C( -83),
                           INT8_C(-122), INT8_C( -64), INT8_C( -83), INT8_C( -74),
                           INT8_C( 119), INT8_C(  -8), INT8_C(  12), INT8_C( 113),
                           INT8_C( -12), INT8_C( -84), INT8_C(   6), INT8_C(  69),
                           INT8_C(   2), INT8_C( -75), INT8_C( -34), INT8_C(-126),
                           INT8_C(   3), INT8_C(-128), INT8_C(  -9), INT8_C(  24),
                           INT8_C(  11), INT8_C( -94), INT8_C( -32), INT8_C( 110),
                           INT8_C(  33), INT8_C( -24), INT8_C( 125), INT8_C(  35),
                           INT8_C(-103), INT8_C( -48), INT8_C( -22), INT8_C(  38),
                           INT8_C( -81), INT8_C(   9), INT8_C( -11), INT8_C(-124),
                           INT8_C(  71), INT8_C(  31), INT8_C( -42), INT8_C(  93),
                           INT8_C(  67), INT8_C(  45), INT8_C(  51), INT8_C( -92),
                           INT8_C( 126), INT8_C( 108), INT8_C(-123), INT8_C( -71),
                           INT8_C( 113), INT8_C(  32), INT8_C(  71), INT8_C(  55),
                           INT8_C( -26), INT8_C(  82), INT8_C( -81), INT8_C( -20)),
      easysimd_mm512_set_epi8(INT8_C(-125), INT8_C( 121), INT8_C(-128), INT8_C( 103),
                           INT8_C(   0), INT8_C( 101), INT8_C( -41), INT8_C(  89),
                           INT8_C( -83), INT8_C( -65), INT8_C(   9), INT8_C(  -7),
                           INT8_C( -63), INT8_C(  13), INT8_C( 105), INT8_C(  92),
                           INT8_C( -18), INT8_C( -21), INT8_C(-102), INT8_C(-114),
                           INT8_C(  74), INT8_C( 121), INT8_C( -45), INT8_C(  52),
                           INT8_C( -63), INT8_C( -93), INT8_C(  98), INT8_C( 106),
                           INT8_C(-109), INT8_C( -47), INT8_C(  37), INT8_C(  70),
                           INT8_C( 100), INT8_C( 121), INT8_C(  18), INT8_C(  28),
                           INT8_C(-117), INT8_C( 107), INT8_C(   3), INT8_C( -62),
                           INT8_C(  42), INT8_C(  72), INT8_C(  91), INT8_C(  86),
                           INT8_C( -72), INT8_C(   9), INT8_C( -80), INT8_C( 118),
                           INT8_C( 122), INT8_C(-108), INT8_C( -70), INT8_C( -63),
                           INT8_C(  56), INT8_C(  71), INT8_C( -14), INT8_C(  49),
                           INT8_C( -73), INT8_C(  53), INT8_C( -29), INT8_C(   3),
                           INT8_C( -73), INT8_C(  43), INT8_C( -22), INT8_C(  85)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(-119), INT8_C( 120), INT8_C(  -5),
                           INT8_C( 113), INT8_C(  62), INT8_C(  17), INT8_C(-119),
                           INT8_C( -30), INT8_C(   0), INT8_C(-119), INT8_C(   0),
                           INT8_C(  72), INT8_C( -74), INT8_C( -40), INT8_C( 120),
                           INT8_C(  95), INT8_C( 108), INT8_C(   0), INT8_C(  64),
                           INT8_C( -75), INT8_C( -34), INT8_C( -21), INT8_C(  24),
                           INT8_C( 105), INT8_C(  52), INT8_C( -94), INT8_C(-104),
                           INT8_C(  57), INT8_C(   0), INT8_C( -38), INT8_C(-128),
                           INT8_C(-107), INT8_C( -22), INT8_C(  55), INT8_C(  46),
                           INT8_C( -71), INT8_C(  77), INT8_C(  71), INT8_C(   0),
                           INT8_C( -48), INT8_C( 125), INT8_C(-103), INT8_C(   9),
                           INT8_C(   0), INT8_C( -11), INT8_C(   0), INT8_C(   9),
                           INT8_C( -53), INT8_C(   0), INT8_C(   9), INT8_C(  64),
                           INT8_C( -61), INT8_C( -25), INT8_C(-115), INT8_C( 100),
                           INT8_C(  65), INT8_C(   8), INT8_C(   0), INT8_C(  -8),
                           INT8_C( -15), INT8_C( 126), INT8_C(   0), INT8_C(  71)) },
    { easysimd_mm512_set_epi8(INT8_C( -47), INT8_C(  84), INT8_C(-126), INT8_C( -64),
                           INT8_C(  14), INT8_C(  11), INT8_C(  37), INT8_C( -23),
                           INT8_C(  67), INT8_C( 124), INT8_C(  58), INT8_C( -94),
                           INT8_C(  30), INT8_C( -33), INT8_C(  70), INT8_C( -24),
                           INT8_C(  38), INT8_C( -97), INT8_C( -56), INT8_C( -60),
                           INT8_C( -59), INT8_C(  65), INT8_C( -74), INT8_C(  45),
                           INT8_C( -11), INT8_C(  55), INT8_C( -82), INT8_C(  12),
                           INT8_C( 106), INT8_C(  22), INT8_C(-124), INT8_C(  -4),
                           INT8_C(   2), INT8_C( -81), INT8_C(  14), INT8_C(  90),
                           INT8_C(-100), INT8_C(-122), INT8_C( -35), INT8_C(  81),
                           INT8_C( -14), INT8_C( -42), INT8_C( 125), INT8_C(-125),
                           INT8_C( -57), INT8_C(  90), INT8_C(  -9), INT8_C(  63),
                           INT8_C(  53), INT8_C(  77), INT8_C(  63), INT8_C( -84),
                           INT8_C(  27), INT8_C(  22), INT8_C(   3), INT8_C( -37),
                           INT8_C(  65), INT8_C( 118), INT8_C(-126), INT8_C(  97),
                           INT8_C( 109), INT8_C(   7), INT8_C(-114), INT8_C( -75)),
      UINT64_C( 8707623543556880126),
      easysimd_mm512_set_epi8(INT8_C(  84), INT8_C( -71), INT8_C(   8), INT8_C(  12),
                           INT8_C( -11), INT8_C( -76), INT8_C(  62), INT8_C(  93),
                           INT8_C( -75), INT8_C( -77), INT8_C( -84), INT8_C(-108),
                           INT8_C( -35), INT8_C(  14), INT8_C( -60), INT8_C(  18),
                           INT8_C(  23), INT8_C( -60), INT8_C( -63), INT8_C(-114),
                           INT8_C( -55), INT8_C(  75), INT8_C( -99), INT8_C( -55),
                           INT8_C(  58), INT8_C(  76), INT8_C(-102), INT8_C(-118),
                           INT8_C( -30), INT8_C(  39), INT8_C( 119), INT8_C(  85),
                           INT8_C(  -8), INT8_C( -72), INT8_C( -60), INT8_C( -94),
                           INT8_C(-112), INT8_C( 119), INT8_C( 124), INT8_C(  76),
                           INT8_C( -42), INT8_C(-124), INT8_C(  54), INT8_C(  74),
                           INT8_C( -92), INT8_C(  99), INT8_C(  79), INT8_C(  -3),
                           INT8_C(  61), INT8_C( -89), INT8_C(  84), INT8_C( -94),
                           INT8_C(  31), INT8_C(-116), INT8_C( -67), INT8_C(-102),
                           INT8_C( -72), INT8_C( -91), INT8_C(-105), INT8_C(-108),
                           INT8_C( -44), INT8_C(  74), INT8_C( -28), INT8_C( 124)),
      easysimd_mm512_set_epi8(INT8_C(   1), INT8_C(  75), INT8_C(  21), INT8_C( -36),
                           INT8_C(-126), INT8_C( 122), INT8_C(  71), INT8_C(  76),
                           INT8_C(  28), INT8_C( -56), INT8_C(  32), INT8_C( 101),
                           INT8_C(-107), INT8_C(-111), INT8_C( -88), INT8_C( -19),
                           INT8_C( -77), INT8_C(  19), INT8_C( -21), INT8_C(-111),
                           INT8_C( -68), INT8_C(  82), INT8_C(-118), INT8_C( -76),
                           INT8_C(  47), INT8_C( 127), INT8_C(  62), INT8_C( -16),
                           INT8_C(  10), INT8_C( -14), INT8_C(-100), INT8_C(  86),
                           INT8_C(  29), INT8_C( 107), INT8_C(  56), INT8_C(  21),
                           INT8_C(  24), INT8_C(  68), INT8_C( -96), INT8_C(  64),
                           INT8_C(  48), INT8_C(  13), INT8_C( -83), INT8_C(   4),
                           INT8_C(  -3), INT8_C( -64), INT8_C(  17), INT8_C(-115),
                           INT8_C(  21), INT8_C( 108), INT8_C( 125), INT8_C( -60),
                           INT8_C( -72), INT8_C(  74), INT8_C(  -5), INT8_C( -58),
                           INT8_C( -41), INT8_C(  22), INT8_C(-115), INT8_C( 102),
                           INT8_C(  59), INT8_C( -80), INT8_C( -15), INT8_C( -63)),
      easysimd_mm512_set_epi8(INT8_C( -47), INT8_C( -11), INT8_C( -84), INT8_C(   0),
                           INT8_C(   0), INT8_C(  11), INT8_C(  37), INT8_C( -23),
                           INT8_C(  12), INT8_C(   0), INT8_C(  58), INT8_C( -84),
                           INT8_C(  30), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -97), INT8_C(   0), INT8_C(   0),
                           INT8_C( -59), INT8_C(  65), INT8_C( -74), INT8_C(   0),
                           INT8_C( -11), INT8_C(  23), INT8_C( -60), INT8_C(   0),
                           INT8_C(  75), INT8_C(  22), INT8_C(   0), INT8_C(  -4),
                           INT8_C(   2), INT8_C(-112), INT8_C(  14), INT8_C(  54),
                           INT8_C(-100), INT8_C(  74), INT8_C(   0), INT8_C(  -3),
                           INT8_C(  -3), INT8_C( -42), INT8_C( 125), INT8_C(-125),
                           INT8_C(   0), INT8_C(  90), INT8_C(  -9), INT8_C(   0),
                           INT8_C(  53), INT8_C(  77), INT8_C(  84), INT8_C(   0),
                           INT8_C(  27), INT8_C(-116), INT8_C(   0), INT8_C( -37),
                           INT8_C(   0), INT8_C( -91), INT8_C(   0), INT8_C( -91),
                           INT8_C(  31), INT8_C(   0), INT8_C(   0), INT8_C( -75)) },
    { easysimd_mm512_set_epi8(INT8_C( -64), INT8_C( -53), INT8_C( -42), INT8_C( 126),
                           INT8_C(  67), INT8_C(  50), INT8_C( -18), INT8_C(  76),
                           INT8_C( -19), INT8_C( 123), INT8_C( -87), INT8_C( 106),
                           INT8_C( -74), INT8_C(  44), INT8_C( 117), INT8_C( 103),
                           INT8_C(  81), INT8_C( 122), INT8_C(  56), INT8_C( -10),
                           INT8_C(  67), INT8_C(  79), INT8_C(  83), INT8_C( -38),
                           INT8_C( -13), INT8_C(  43), INT8_C(  27), INT8_C( -97),
                           INT8_C( 102), INT8_C( 126), INT8_C(  38), INT8_C( -62),
                           INT8_C( -24), INT8_C( 117), INT8_C( -38), INT8_C( -93),
                           INT8_C( -58), INT8_C(-124), INT8_C( -75), INT8_C(  10),
                           INT8_C(  18), INT8_C( -74), INT8_C(  14), INT8_C(  36),
                           INT8_C(  -7), INT8_C( 113), INT8_C(  40), INT8_C(  48),
                           INT8_C(-107), INT8_C( -34), INT8_C( -75), INT8_C(  85),
                           INT8_C( -35), INT8_C(-116), INT8_C(  65), INT8_C( -21),
                           INT8_C(  15), INT8_C(   3), INT8_C(  45), INT8_C(  21),
                           INT8_C(  72), INT8_C(  93), INT8_C( 108), INT8_C( 125)),
      UINT64_C(12576710173448868104),
      easysimd_mm512_set_epi8(INT8_C(  90), INT8_C( -38), INT8_C( -98), INT8_C( -70),
                           INT8_C(-108), INT8_C(  20), INT8_C(  43), INT8_C(-128),
                           INT8_C(  77), INT8_C( 108), INT8_C(  53), INT8_C(  82),
                           INT8_C( -50), INT8_C(  52), INT8_C(  56), INT8_C(  58),
                           INT8_C(-120), INT8_C( -43), INT8_C( 114), INT8_C(  93),
                           INT8_C( -44), INT8_C( -15), INT8_C(  38), INT8_C( -17),
                           INT8_C(-110), INT8_C(-123), INT8_C( -39), INT8_C( 114),
                           INT8_C(  51), INT8_C(-115), INT8_C( -74), INT8_C(  43),
                           INT8_C(  41), INT8_C( -36), INT8_C(  19), INT8_C(  69),
                           INT8_C(  60), INT8_C( -53), INT8_C( 112), INT8_C( 108),
                           INT8_C(   8), INT8_C(  46), INT8_C( -35), INT8_C(  26),
                           INT8_C(  11), INT8_C(  42), INT8_C(  47), INT8_C(  59),
                           INT8_C( -57), INT8_C(  94), INT8_C(-125), INT8_C(-124),
                           INT8_C(  36), INT8_C(  57), INT8_C(  68), INT8_C( -52),
                           INT8_C(  39), INT8_C(  50), INT8_C( -48), INT8_C(  94),
                           INT8_C(  53), INT8_C(  11), INT8_C(  29), INT8_C(  65)),
      easysimd_mm512_set_epi8(INT8_C(-107), INT8_C( 120), INT8_C( -58), INT8_C( 107),
                           INT8_C( -32), INT8_C( -32), INT8_C(  88), INT8_C( -43),
                           INT8_C(  31), INT8_C( -32), INT8_C( -64), INT8_C(  27),
                           INT8_C(  82), INT8_C( -90), INT8_C( -54), INT8_C( -84),
                           INT8_C( -30), INT8_C(  63), INT8_C(  24), INT8_C(  81),
                           INT8_C(  -8), INT8_C(   9), INT8_C( -35), INT8_C(-101),
                           INT8_C(  83), INT8_C( 107), INT8_C( -47), INT8_C( -56),
                           INT8_C(  57), INT8_C( -88), INT8_C(-115), INT8_C( -20),
                           INT8_C(  58), INT8_C(  75), INT8_C(  56), INT8_C(  93),
                           INT8_C(  49), INT8_C(  43), INT8_C( 108), INT8_C( 118),
                           INT8_C( -79), INT8_C( 112), INT8_C(  44), INT8_C(-112),
                           INT8_C( -52), INT8_C(  10), INT8_C(  28), INT8_C( -86),
                           INT8_C(  65), INT8_C(  62), INT8_C(  86), INT8_C(-107),
                           INT8_C(  24), INT8_C( -55), INT8_C(  54), INT8_C(-110),
                           INT8_C( -33), INT8_C( 110), INT8_C(-116), INT8_C( -39),
                           INT8_C(  39), INT8_C(-112), INT8_C(  64), INT8_C(  43)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C( -53), INT8_C(   0), INT8_C( 126),
                           INT8_C(   0), INT8_C(   0), INT8_C(-128), INT8_C(  76),
                           INT8_C(  90), INT8_C( 123), INT8_C( -87), INT8_C( 106),
                           INT8_C(  52), INT8_C(  44), INT8_C( 117), INT8_C(   0),
                           INT8_C(  81), INT8_C(-120), INT8_C( -17), INT8_C( -74),
                           INT8_C(  67), INT8_C(  79), INT8_C(   0), INT8_C(   0),
                           INT8_C( -13), INT8_C( -44), INT8_C(  27), INT8_C( -97),
                           INT8_C( 102), INT8_C(   0), INT8_C(  38), INT8_C( -62),
                           INT8_C( -53), INT8_C( 117), INT8_C( 108), INT8_C(  19),
                           INT8_C(  47), INT8_C(-124), INT8_C(  69), INT8_C(  46),
                           INT8_C(  18), INT8_C(  59), INT8_C(  69), INT8_C(   0),
                           INT8_C(  -7), INT8_C( -53), INT8_C(  69), INT8_C(  48),
                           INT8_C(-107), INT8_C( -34), INT8_C( -75), INT8_C(   0),
                           INT8_C( -35), INT8_C(-116), INT8_C(  65), INT8_C(   0),
                           INT8_C(  15), INT8_C(   3), INT8_C(  45), INT8_C(  21),
                           INT8_C(  39), INT8_C(  93), INT8_C( 108), INT8_C( 125)) },
    { easysimd_mm512_set_epi8(INT8_C(-103), INT8_C(-124), INT8_C(  13), INT8_C( -80),
                           INT8_C( -32), INT8_C( 123), INT8_C( 115), INT8_C( -49),
                           INT8_C( 113), INT8_C( -51), INT8_C( 119), INT8_C( -67),
                           INT8_C( -21), INT8_C(  19), INT8_C( -12), INT8_C(  29),
                           INT8_C(  91), INT8_C( -64), INT8_C(  88), INT8_C(-105),
                           INT8_C(  24), INT8_C(  31), INT8_C(  77), INT8_C(  92),
                           INT8_C( -85), INT8_C(  48), INT8_C(-113), INT8_C(-114),
                           INT8_C(  16), INT8_C(  18), INT8_C( -44), INT8_C( -66),
                           INT8_C(  25), INT8_C(-105), INT8_C(-100), INT8_C( -91),
                           INT8_C(  26), INT8_C(-123), INT8_C( -33), INT8_C( -37),
                           INT8_C(  63), INT8_C(  13), INT8_C(-124), INT8_C(  41),
                           INT8_C(-104), INT8_C( -59), INT8_C(  -2), INT8_C( -54),
                           INT8_C(  73), INT8_C(   3), INT8_C( -18), INT8_C(  10),
                           INT8_C(  19), INT8_C(  58), INT8_C( -12), INT8_C( -75),
                           INT8_C( -88), INT8_C(  59), INT8_C( 104), INT8_C( -40),
                           INT8_C( -82), INT8_C(  42), INT8_C( -73), INT8_C( -94)),
      UINT64_C( 2633789449456316803),
      easysimd_mm512_set_epi8(INT8_C( 118), INT8_C(  44), INT8_C( -14), INT8_C(  84),
                           INT8_C( -69), INT8_C( -48), INT8_C(  40), INT8_C(  86),
                           INT8_C(  -1), INT8_C( 121), INT8_C( -40), INT8_C(  44),
                           INT8_C(  -1), INT8_C(  38), INT8_C( -44), INT8_C(  38),
                           INT8_C(  93), INT8_C(-107), INT8_C(  -4), INT8_C( -61),
                           INT8_C( -52), INT8_C(  22), INT8_C( -54), INT8_C(-125),
                           INT8_C( -29), INT8_C(  25), INT8_C( -77), INT8_C( -68),
                           INT8_C(   9), INT8_C( -35), INT8_C( -11), INT8_C(  53),
                           INT8_C(  28), INT8_C(  61), INT8_C( -35), INT8_C(-106),
                           INT8_C( -46), INT8_C( 121), INT8_C(-102), INT8_C( 121),
                           INT8_C( -54), INT8_C( -60), INT8_C(   7), INT8_C(   2),
                           INT8_C(-119), INT8_C( 111), INT8_C( -20), INT8_C( -58),
                           INT8_C( -98), INT8_C(  83), INT8_C(  32), INT8_C( -49),
                           INT8_C( -11), INT8_C(  48), INT8_C(  92), INT8_C(  47),
                           INT8_C( -38), INT8_C(  11), INT8_C(  26), INT8_C(  90),
                           INT8_C(  -5), INT8_C(  73), INT8_C(  45), INT8_C(  -2)),
      easysimd_mm512_set_epi8(INT8_C( -31), INT8_C( 126), INT8_C(-118), INT8_C( -57),
                           INT8_C(-126), INT8_C(  29), INT8_C( 101), INT8_C( -15),
                           INT8_C( -61), INT8_C( 118), INT8_C( 102), INT8_C(  12),
                           INT8_C( -59), INT8_C( -41), INT8_C( -60), INT8_C(  46),
                           INT8_C(  63), INT8_C(  78), INT8_C(  87), INT8_C(  18),
                           INT8_C(  18), INT8_C(  32), INT8_C( -26), INT8_C(  -6),
                           INT8_C( -20), INT8_C(-123), INT8_C(  99), INT8_C(  65),
                           INT8_C(  13), INT8_C(  25), INT8_C( 108), INT8_C(-121),
                           INT8_C(  42), INT8_C(   0), INT8_C( 104), INT8_C( -17),
                           INT8_C( 111), INT8_C(-110), INT8_C( -39), INT8_C( 125),
                           INT8_C( -50), INT8_C( -47), INT8_C(  30), INT8_C(  92),
                           INT8_C(  19), INT8_C(  -8), INT8_C(-120), INT8_C( 127),
                           INT8_C( -49), INT8_C(  23), INT8_C(  16), INT8_C( -64),
                           INT8_C( -79), INT8_C( 116), INT8_C(  -5), INT8_C( -50),
                           INT8_C(  32), INT8_C(  22), INT8_C( -42), INT8_C(  -3),
                           INT8_C(  30), INT8_C(  64), INT8_C(  96), INT8_C( -66)),
      easysimd_mm512_set_epi8(INT8_C(-103), INT8_C(-124), INT8_C(   0), INT8_C( -80),
                           INT8_C( -32), INT8_C( -14), INT8_C( 115), INT8_C( -49),
                           INT8_C(   0), INT8_C( -51), INT8_C( 119), INT8_C( -67),
                           INT8_C(   0), INT8_C(   0), INT8_C( -12), INT8_C(  44),
                           INT8_C(  91), INT8_C( -64), INT8_C(  88), INT8_C( -35),
                           INT8_C( -35), INT8_C(  31), INT8_C(  77), INT8_C(   0),
                           INT8_C(   0), INT8_C(  48), INT8_C(-113), INT8_C(-114),
                           INT8_C(  -4), INT8_C( -54), INT8_C( -44), INT8_C( -66),
                           INT8_C( 121), INT8_C( -58), INT8_C(-100), INT8_C( -91),
                           INT8_C(  26), INT8_C(   0), INT8_C( -33), INT8_C( -37),
                           INT8_C(  63), INT8_C(  13), INT8_C(-124), INT8_C(  41),
                           INT8_C(-104), INT8_C( -59), INT8_C(   0), INT8_C(  28),
                           INT8_C(   0), INT8_C( -38), INT8_C(  -2), INT8_C(   0),
                           INT8_C(  19), INT8_C(  58), INT8_C( -12), INT8_C(   0),
                           INT8_C(  -2), INT8_C(  59), INT8_C( 104), INT8_C( -40),
                           INT8_C( -82), INT8_C(  42), INT8_C(  -2), INT8_C(   0)) },
    { easysimd_mm512_set_epi8(INT8_C( 117), INT8_C(  69), INT8_C( 121), INT8_C( -45),
                           INT8_C(  30), INT8_C( -73), INT8_C(   2), INT8_C( -40),
                           INT8_C(  95), INT8_C(  -3), INT8_C(  16), INT8_C( -78),
                           INT8_C(-128), INT8_C( -41), INT8_C( -66), INT8_C(  66),
                           INT8_C(  93), INT8_C( -32), INT8_C( -50), INT8_C(  -2),
                           INT8_C( -61), INT8_C(  29), INT8_C( -88), INT8_C(-118),
                           INT8_C( -27), INT8_C(  42), INT8_C(  78), INT8_C( -46),
                           INT8_C( -79), INT8_C(  38), INT8_C( -75), INT8_C(  14),
                           INT8_C(-118), INT8_C(-114), INT8_C(-120), INT8_C(  42),
                           INT8_C(   4), INT8_C(  79), INT8_C( -84), INT8_C(-110),
                           INT8_C(   0), INT8_C(  85), INT8_C( -20), INT8_C(  61),
                           INT8_C(  40), INT8_C( -75), INT8_C(  69), INT8_C( -99),
                           INT8_C(  73), INT8_C( -34), INT8_C( -14), INT8_C(  72),
                           INT8_C( 101), INT8_C(  96), INT8_C(-114), INT8_C(-124),
                           INT8_C( -87), INT8_C(  43), INT8_C(-118), INT8_C( 114),
                           INT8_C( -88), INT8_C(   8), INT8_C( -52), INT8_C(  75)),
      UINT64_C(14890918166471265655),
      easysimd_mm512_set_epi8(INT8_C( -56), INT8_C( -70), INT8_C(  66), INT8_C(-111),
                           INT8_C( -69), INT8_C(  15), INT8_C( -82), INT8_C( -12),
                           INT8_C( -89), INT8_C(  37), INT8_C(  80), INT8_C( 120),
                           INT8_C( -83), INT8_C(-120), INT8_C(  95), INT8_C(  21),
                           INT8_C(  91), INT8_C( -97), INT8_C( -72), INT8_C(  42),
                           INT8_C(  22), INT8_C( -70), INT8_C(  71), INT8_C( -78),
                           INT8_C(  -5), INT8_C(  52), INT8_C( -22), INT8_C( -34),
                           INT8_C(  16), INT8_C(  92), INT8_C(  91), INT8_C( -72),
                           INT8_C(   3), INT8_C( -31), INT8_C( -95), INT8_C( -56),
                           INT8_C( -50), INT8_C(  68), INT8_C( -24), INT8_C( -50),
                           INT8_C(  94), INT8_C(  67), INT8_C( 108), INT8_C(-118),
                           INT8_C( -65), INT8_C(  31), INT8_C(  70), INT8_C( 108),
                           INT8_C( -66), INT8_C( 107), INT8_C( -85), INT8_C(  38),
                           INT8_C(   2), INT8_C(  32), INT8_C(  56), INT8_C(  66),
                           INT8_C(  -2), INT8_C( -74), INT8_C( 112), INT8_C(  10),
                           INT8_C(  64), INT8_C( 100), INT8_C( -55), INT8_C(  83)),
      easysimd_mm512_set_epi8(INT8_C( -69), INT8_C( -45), INT8_C(  23), INT8_C(  70),
                           INT8_C(  22), INT8_C( 113), INT8_C(  75), INT8_C(-117),
                           INT8_C(  69), INT8_C( -74), INT8_C(  40), INT8_C( -39),
                           INT8_C( -51), INT8_C( -91), INT8_C(  25), INT8_C(  -4),
                           INT8_C( -12), INT8_C( -13), INT8_C( -77), INT8_C(  75),
                           INT8_C(-116), INT8_C( -36), INT8_C(  38), INT8_C( -52),
                           INT8_C(  13), INT8_C( -52), INT8_C(-100), INT8_C( -46),
                           INT8_C(  13), INT8_C(  83), INT8_C( -94), INT8_C( 102),
                           INT8_C( 116), INT8_C( -60), INT8_C(  44), INT8_C(  -7),
                           INT8_C( 104), INT8_C( -50), INT8_C( -30), INT8_C(  17),
                           INT8_C(  13), INT8_C(-107), INT8_C( 111), INT8_C( -13),
                           INT8_C(  17), INT8_C( -45), INT8_C( -31), INT8_C(  76),
                           INT8_C(-102), INT8_C( -71), INT8_C( 116), INT8_C(  72),
                           INT8_C( -11), INT8_C( -97), INT8_C(  93), INT8_C(  48),
                           INT8_C( -75), INT8_C( -90), INT8_C( -10), INT8_C( -78),
                           INT8_C(  -8), INT8_C( -10), INT8_C(  48), INT8_C( -11)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C( 121), INT8_C( -45),
                           INT8_C(  37), INT8_C(  95), INT8_C( -69), INT8_C( -40),
                           INT8_C(  80), INT8_C(  -3), INT8_C( -12), INT8_C( -78),
                           INT8_C(-128), INT8_C(   0), INT8_C( -82), INT8_C(   0),
                           INT8_C(  93), INT8_C( -32), INT8_C(   0), INT8_C(  -2),
                           INT8_C(   0), INT8_C(  29), INT8_C(  52), INT8_C(   0),
                           INT8_C( -27), INT8_C(  42), INT8_C(   0), INT8_C( -46),
                           INT8_C( -72), INT8_C(  16), INT8_C( -75), INT8_C(  52),
                           INT8_C(-118), INT8_C(   0), INT8_C(-120), INT8_C(  42),
                           INT8_C( -50), INT8_C(  79), INT8_C(   0), INT8_C(  70),
                           INT8_C( -95), INT8_C(   0), INT8_C(   3), INT8_C(  61),
                           INT8_C(  70), INT8_C(   0), INT8_C(   0), INT8_C( -99),
                           INT8_C(  73), INT8_C(   0), INT8_C( -14), INT8_C(  72),
                           INT8_C(   0), INT8_C(   0), INT8_C(-114), INT8_C(  83),
                           INT8_C( -87), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -88), INT8_C(   0), INT8_C(  83), INT8_C(   0)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_mask_shuffle_epi8(test_vec[i].src, test_vec[i].k, test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_shuffle_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT64_C(17286015531074160252),
      easysimd_mm512_set_epi8(INT8_C(-115), INT8_C( -27), INT8_C(  62), INT8_C( -85),
                           INT8_C(  49), INT8_C(-115), INT8_C(  38), INT8_C(   4),
                           INT8_C(  92), INT8_C( 116), INT8_C( -78), INT8_C( -19),
                           INT8_C( -73), INT8_C(  22), INT8_C( -66), INT8_C( -29),
                           INT8_C(  55), INT8_C(  78), INT8_C( -45), INT8_C(-119),
                           INT8_C(  30), INT8_C( -29), INT8_C( -23), INT8_C(  89),
                           INT8_C(  15), INT8_C( 119), INT8_C( -88), INT8_C( 109),
                           INT8_C( -16), INT8_C( -10), INT8_C(   0), INT8_C( -18),
                           INT8_C(   1), INT8_C( 116), INT8_C(  62), INT8_C( -73),
                           INT8_C(-122), INT8_C(   0), INT8_C( 111), INT8_C(  95),
                           INT8_C( -20), INT8_C(   3), INT8_C( 122), INT8_C(-121),
                           INT8_C(  13), INT8_C(  25), INT8_C( -35), INT8_C(-127),
                           INT8_C(  82), INT8_C(  90), INT8_C(  53), INT8_C( 123),
                           INT8_C(  73), INT8_C( 108), INT8_C( -18), INT8_C(  15),
                           INT8_C(  81), INT8_C( -27), INT8_C( 102), INT8_C( 118),
                           INT8_C(  39), INT8_C( -77), INT8_C(  45), INT8_C(  81)),
      easysimd_mm512_set_epi8(INT8_C(  68), INT8_C(-108), INT8_C(-121), INT8_C(-115),
                           INT8_C( -59), INT8_C( -78), INT8_C(-111), INT8_C(  56),
                           INT8_C(   9), INT8_C(-121), INT8_C( 120), INT8_C( -36),
                           INT8_C(  36), INT8_C(-103), INT8_C(-104), INT8_C(  25),
                           INT8_C(-111), INT8_C( -63), INT8_C( -35), INT8_C(-120),
                           INT8_C(  28), INT8_C( -44), INT8_C( -26), INT8_C( -86),
                           INT8_C( -13), INT8_C(  53), INT8_C( -88), INT8_C(-107),
                           INT8_C(  68), INT8_C(  42), INT8_C(-118), INT8_C( 111),
                           INT8_C(  54), INT8_C( -58), INT8_C( -13), INT8_C(  27),
                           INT8_C(  23), INT8_C(  41), INT8_C(-119), INT8_C(  44),
                           INT8_C(   7), INT8_C(-120), INT8_C(  32), INT8_C( -43),
                           INT8_C( 114), INT8_C( -72), INT8_C(  73), INT8_C( -96),
                           INT8_C(  96), INT8_C( 110), INT8_C( -81), INT8_C( -76),
                           INT8_C( 103), INT8_C(-100), INT8_C( -22), INT8_C(  18),
                           INT8_C( 115), INT8_C(  54), INT8_C( -40), INT8_C( 125),
                           INT8_C( 110), INT8_C(  31), INT8_C(  51), INT8_C(-104)),
      easysimd_mm512_set_epi8(INT8_C( -19), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   4),
                           INT8_C(  38), INT8_C(   0), INT8_C(   4), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 109), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -20), INT8_C( 111), INT8_C(   0), INT8_C( -73),
                           INT8_C( -20), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  25), INT8_C(   0), INT8_C( 111), INT8_C(   0),
                           INT8_C(  81), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -27), INT8_C(   0), INT8_C(  53),
                           INT8_C(  90), INT8_C(  82), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C( 9031211210081693899),
      easysimd_mm512_set_epi8(INT8_C(-106), INT8_C( -18), INT8_C(  65), INT8_C(-104),
                           INT8_C(  81), INT8_C(  38), INT8_C( -84), INT8_C(  -2),
                           INT8_C( -14), INT8_C(  85), INT8_C( -80), INT8_C(  80),
                           INT8_C(  48), INT8_C(  93), INT8_C(  79), INT8_C( 127),
                           INT8_C(  16), INT8_C(  41), INT8_C(  54), INT8_C(-116),
                           INT8_C(  17), INT8_C(  42), INT8_C( -86), INT8_C(  38),
                           INT8_C(-118), INT8_C( -56), INT8_C(  60), INT8_C(  19),
                           INT8_C(-100), INT8_C(-107), INT8_C( 105), INT8_C( -76),
                           INT8_C(  10), INT8_C(  -9), INT8_C( -12), INT8_C( -56),
                           INT8_C( -71), INT8_C(  96), INT8_C(   6), INT8_C(  24),
                           INT8_C(  69), INT8_C(  73), INT8_C( -27), INT8_C(   3),
                           INT8_C( -95), INT8_C(  50), INT8_C( -67), INT8_C( -39),
                           INT8_C(  97), INT8_C(  62), INT8_C( 101), INT8_C(  56),
                           INT8_C( -79), INT8_C(  83), INT8_C( -99), INT8_C( -56),
                           INT8_C( -17), INT8_C( -56), INT8_C(  -8), INT8_C( -16),
                           INT8_C(  84), INT8_C( -95), INT8_C( -73), INT8_C(  54)),
      easysimd_mm512_set_epi8(INT8_C(  63), INT8_C( -51), INT8_C(  87), INT8_C(  32),
                           INT8_C( -65), INT8_C(  55), INT8_C(  14), INT8_C(  81),
                           INT8_C(-123), INT8_C(-100), INT8_C( -39), INT8_C( -44),
                           INT8_C(  22), INT8_C( 112), INT8_C(  16), INT8_C(  15),
                           INT8_C(-104), INT8_C(-102), INT8_C( 102), INT8_C( -95),
                           INT8_C(  69), INT8_C(  13), INT8_C( -21), INT8_C(  45),
                           INT8_C(  62), INT8_C(-120), INT8_C(-111), INT8_C(  32),
                           INT8_C(-107), INT8_C( -30), INT8_C(  99), INT8_C( -64),
                           INT8_C(   8), INT8_C( -42), INT8_C(  81), INT8_C( -34),
                           INT8_C( -46), INT8_C(  26), INT8_C(  31), INT8_C(  -2),
                           INT8_C(  68), INT8_C(  -7), INT8_C( -71), INT8_C(  46),
                           INT8_C( -21), INT8_C( -73), INT8_C(  21), INT8_C(  83),
                           INT8_C(-108), INT8_C( -97), INT8_C( -69), INT8_C(  73),
                           INT8_C(  57), INT8_C( -37), INT8_C(  21), INT8_C(  82),
                           INT8_C(-119), INT8_C(-126), INT8_C( 126), INT8_C(  91),
                           INT8_C( 115), INT8_C(  31), INT8_C( -79), INT8_C(  28)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C( -14), INT8_C( 127),
                           INT8_C(   0), INT8_C( -14), INT8_C(   0), INT8_C(  79),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( 127), INT8_C(   0), INT8_C(-106),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  60), INT8_C(  54), INT8_C(   0), INT8_C(   0),
                           INT8_C(  41), INT8_C(   0), INT8_C(   0), INT8_C( -76),
                           INT8_C(   0), INT8_C(   0), INT8_C(-100), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  96), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  -9),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -95),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -99),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  84), INT8_C(   0), INT8_C(   0), INT8_C(  56)) },
    { UINT64_C( 6828010367541473016),
      easysimd_mm512_set_epi8(INT8_C(  84), INT8_C( -96), INT8_C(  87), INT8_C( -22),
                           INT8_C(  -5), INT8_C(  -3), INT8_C(-127), INT8_C(  41),
                           INT8_C(  74), INT8_C(  72), INT8_C(  -4), INT8_C(  28),
                           INT8_C(-115), INT8_C(  93), INT8_C( 102), INT8_C(  44),
                           INT8_C(-103), INT8_C( -29), INT8_C( -50), INT8_C(  48),
                           INT8_C( -96), INT8_C( -50), INT8_C(  46), INT8_C( -65),
                           INT8_C(   4), INT8_C(  43), INT8_C( -75), INT8_C(  97),
                           INT8_C( -26), INT8_C(-103), INT8_C(  71), INT8_C(-107),
                           INT8_C(  91), INT8_C(  45), INT8_C( -11), INT8_C(  47),
                           INT8_C(  29), INT8_C(  25), INT8_C(  26), INT8_C(  -9),
                           INT8_C(  10), INT8_C(  36), INT8_C( -79), INT8_C( -53),
                           INT8_C(  41), INT8_C(   1), INT8_C( -23), INT8_C( -63),
                           INT8_C(-127), INT8_C(  68), INT8_C(  48), INT8_C(  36),
                           INT8_C(  89), INT8_C(-112), INT8_C( -31), INT8_C( 120),
                           INT8_C(  35), INT8_C(  62), INT8_C( -21), INT8_C(-114),
                           INT8_C(-104), INT8_C(  57), INT8_C(  42), INT8_C(-111)),
      easysimd_mm512_set_epi8(INT8_C(  96), INT8_C( -64), INT8_C( -23), INT8_C( -46),
                           INT8_C( -29), INT8_C( -71), INT8_C(  71), INT8_C(  90),
                           INT8_C(  44), INT8_C( -92), INT8_C( -31), INT8_C(  26),
                           INT8_C( -99), INT8_C( -53), INT8_C( 117), INT8_C(  18),
                           INT8_C( -63), INT8_C(  45), INT8_C(  12), INT8_C(  24),
                           INT8_C(-108), INT8_C(  18), INT8_C( -60), INT8_C(  28),
                           INT8_C(  50), INT8_C( -11), INT8_C( -68), INT8_C( -31),
                           INT8_C( 105), INT8_C(-106), INT8_C(  98), INT8_C(  51),
                           INT8_C(  58), INT8_C( 103), INT8_C( 111), INT8_C(-127),
                           INT8_C(  68), INT8_C( -56), INT8_C( 124), INT8_C(-119),
                           INT8_C(  74), INT8_C( -62), INT8_C(-116), INT8_C(  37),
                           INT8_C( -12), INT8_C( 114), INT8_C(   0), INT8_C(  61),
                           INT8_C( 103), INT8_C(  -4), INT8_C(-105), INT8_C( -68),
                           INT8_C(  39), INT8_C(-100), INT8_C( -93), INT8_C(  11),
                           INT8_C( -80), INT8_C( -19), INT8_C( -22), INT8_C( -39),
                           INT8_C( 127), INT8_C( -38), INT8_C(-125), INT8_C(-111)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  74), INT8_C(   0),
                           INT8_C( -22), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  93),
                           INT8_C(   0), INT8_C( -50), INT8_C(  48), INT8_C( -65),
                           INT8_C(   0), INT8_C(-103), INT8_C(   0), INT8_C(  48),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  25), INT8_C(   0), INT8_C(  91), INT8_C(   0),
                           INT8_C( -53), INT8_C(   0), INT8_C(  47), INT8_C(   0),
                           INT8_C(  25), INT8_C(   0), INT8_C(   0), INT8_C( -79),
                           INT8_C(   0), INT8_C(   1), INT8_C( -63), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  35), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-127), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(18325019528117086600),
      easysimd_mm512_set_epi8(INT8_C( 122), INT8_C( -54), INT8_C(  50), INT8_C( 123),
                           INT8_C( -31), INT8_C( -74), INT8_C( -64), INT8_C(  54),
                           INT8_C( -81), INT8_C(  60), INT8_C(  31), INT8_C( -23),
                           INT8_C( 108), INT8_C(-119), INT8_C( -92), INT8_C( -80),
                           INT8_C( -30), INT8_C( -37), INT8_C(  51), INT8_C( -36),
                           INT8_C(   8), INT8_C(  52), INT8_C(  97), INT8_C( 123),
                           INT8_C( -49), INT8_C(-124), INT8_C(  95), INT8_C( -83),
                           INT8_C(  70), INT8_C( -50), INT8_C( -61), INT8_C(  25),
                           INT8_C( -97), INT8_C(  28), INT8_C( -58), INT8_C(  11),
                           INT8_C( -14), INT8_C( 126), INT8_C(  81), INT8_C(  45),
                           INT8_C( -23), INT8_C( 120), INT8_C( -83), INT8_C( -16),
                           INT8_C(   7), INT8_C(  51), INT8_C( -57), INT8_C( -50),
                           INT8_C( -21), INT8_C(  98), INT8_C(  88), INT8_C(   0),
                           INT8_C( -66), INT8_C(   3), INT8_C( 124), INT8_C(-113),
                           INT8_C(  50), INT8_C(  88), INT8_C( -85), INT8_C( -93),
                           INT8_C( -44), INT8_C( -13), INT8_C( -94), INT8_C(  17)),
      easysimd_mm512_set_epi8(INT8_C( -13), INT8_C( -12), INT8_C(-124), INT8_C(  96),
                           INT8_C( -13), INT8_C( -98), INT8_C(  99), INT8_C( -13),
                           INT8_C(  56), INT8_C( -45), INT8_C(-108), INT8_C( -19),
                           INT8_C(-124), INT8_C( -27), INT8_C(  22), INT8_C( 126),
                           INT8_C(-106), INT8_C( -68), INT8_C( -60), INT8_C(   8),
                           INT8_C(  60), INT8_C(  93), INT8_C( -33), INT8_C( -27),
                           INT8_C(  -7), INT8_C(  27), INT8_C(-122), INT8_C( -38),
                           INT8_C(  23), INT8_C(   6), INT8_C(  45), INT8_C( -21),
                           INT8_C( -23), INT8_C(-101), INT8_C( 116), INT8_C( 127),
                           INT8_C(  96), INT8_C(  40), INT8_C( -97), INT8_C(  40),
                           INT8_C(  86), INT8_C( -44), INT8_C(  70), INT8_C( -71),
                           INT8_C(  62), INT8_C( -21), INT8_C(  66), INT8_C(  68),
                           INT8_C( -87), INT8_C( -61), INT8_C(  48), INT8_C( -70),
                           INT8_C(  18), INT8_C( -78), INT8_C( -98), INT8_C( 117),
                           INT8_C(  74), INT8_C(  32), INT8_C(  93), INT8_C( 125),
                           INT8_C( -47), INT8_C( -60), INT8_C( -86), INT8_C( 117)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -80),
                           INT8_C(   0), INT8_C(   0), INT8_C( 108), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  60), INT8_C( -54),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -36), INT8_C(  51), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -49), INT8_C(   0), INT8_C(  51), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -16), INT8_C( -97),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  45),
                           INT8_C( 120), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  51), INT8_C( -16),
                           INT8_C(   0), INT8_C(   0), INT8_C(  17), INT8_C(   0),
                           INT8_C( -13), INT8_C(   0), INT8_C(   0), INT8_C( -85),
                           INT8_C(   3), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C( 7677112093376593822),
      easysimd_mm512_set_epi8(INT8_C( -53), INT8_C(  34), INT8_C(   9), INT8_C(  64),
                           INT8_C( -61), INT8_C( -25), INT8_C(-115), INT8_C( 100),
                           INT8_C(  65), INT8_C(   8), INT8_C(  69), INT8_C(  -8),
                           INT8_C( -15), INT8_C( -51), INT8_C(   1), INT8_C(  90),
                           INT8_C( 115), INT8_C(  51), INT8_C( -91), INT8_C(  56),
                           INT8_C(  64), INT8_C( -39), INT8_C(-119), INT8_C( -28),
                           INT8_C( -54), INT8_C(  28), INT8_C(  54), INT8_C(  -8),
                           INT8_C( -54), INT8_C(-128), INT8_C( -28), INT8_C( -71),
                           INT8_C( 107), INT8_C( -66), INT8_C(-114), INT8_C( -88),
                           INT8_C(  34), INT8_C( -83), INT8_C( -21), INT8_C( -64),
                           INT8_C( 121), INT8_C( -20), INT8_C( -89), INT8_C( -94),
                           INT8_C( 112), INT8_C( -27), INT8_C(  81), INT8_C( -54),
                           INT8_C( -64), INT8_C(-114), INT8_C(  48), INT8_C( -89),
                           INT8_C( -61), INT8_C(  26), INT8_C(  43), INT8_C(  29),
                           INT8_C(   0), INT8_C(  64), INT8_C( 123), INT8_C( -67),
                           INT8_C(  15), INT8_C( 120), INT8_C(  36), INT8_C(  40)),
      easysimd_mm512_set_epi8(INT8_C( 113), INT8_C(  32), INT8_C(  71), INT8_C(  55),
                           INT8_C( -26), INT8_C(  82), INT8_C( -81), INT8_C( -20),
                           INT8_C( -55), INT8_C( 112), INT8_C(  45), INT8_C(  37),
                           INT8_C(  67), INT8_C( -69), INT8_C(  64), INT8_C(  39),
                           INT8_C(  72), INT8_C(  45), INT8_C( 120), INT8_C(  -5),
                           INT8_C(-109), INT8_C(  62), INT8_C(  17), INT8_C(  31),
                           INT8_C( -30), INT8_C( -58), INT8_C(  56), INT8_C(  21),
                           INT8_C(  72), INT8_C( -74), INT8_C( -40), INT8_C( 120),
                           INT8_C(  95), INT8_C( 108), INT8_C(  32), INT8_C(  64),
                           INT8_C(-128), INT8_C( 102), INT8_C( -21), INT8_C(  28),
                           INT8_C( 105), INT8_C(  52), INT8_C(  85), INT8_C(-104),
                           INT8_C(  57), INT8_C( -31), INT8_C( -38), INT8_C(-124),
                           INT8_C(-107), INT8_C(  -2), INT8_C(  55), INT8_C(  46),
                           INT8_C( -71), INT8_C(  77), INT8_C(  18), INT8_C(  70),
                           INT8_C(  89), INT8_C( 125), INT8_C( -42), INT8_C(-125),
                           INT8_C( 121), INT8_C( -11), INT8_C( -69), INT8_C( -59)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(  90), INT8_C(  65), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -15), INT8_C(   0), INT8_C(  90), INT8_C(   0),
                           INT8_C( -28), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  51), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -54),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -21), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(-114),
                           INT8_C(   0), INT8_C(  48), INT8_C( 120), INT8_C(  64),
                           INT8_C(  43), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  43), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C( 4840581956081845689),
      easysimd_mm512_set_epi8(INT8_C( 122), INT8_C(-108), INT8_C( -70), INT8_C( -63),
                           INT8_C(  56), INT8_C(  71), INT8_C( -14), INT8_C(  49),
                           INT8_C( -73), INT8_C(  53), INT8_C( -29), INT8_C(   3),
                           INT8_C( -73), INT8_C(  43), INT8_C( -22), INT8_C(  85),
                           INT8_C( -26), INT8_C(  -9), INT8_C(  66), INT8_C(   1),
                           INT8_C( -13), INT8_C(  60), INT8_C(-119), INT8_C( -83),
                           INT8_C(-122), INT8_C( -64), INT8_C( -83), INT8_C( -74),
                           INT8_C( 119), INT8_C(  -8), INT8_C(  12), INT8_C( 113),
                           INT8_C( -12), INT8_C( -84), INT8_C(   6), INT8_C(  69),
                           INT8_C(   2), INT8_C( -75), INT8_C( -34), INT8_C(-126),
                           INT8_C(   3), INT8_C(-128), INT8_C(  -9), INT8_C(  24),
                           INT8_C(  11), INT8_C( -94), INT8_C( -32), INT8_C( 110),
                           INT8_C(  33), INT8_C( -24), INT8_C( 125), INT8_C(  35),
                           INT8_C(-103), INT8_C( -48), INT8_C( -22), INT8_C(  38),
                           INT8_C( -81), INT8_C(   9), INT8_C( -11), INT8_C(-124),
                           INT8_C(  71), INT8_C(  31), INT8_C( -42), INT8_C(  93)),
      easysimd_mm512_set_epi8(INT8_C(  53), INT8_C(  77), INT8_C(  63), INT8_C( -84),
                           INT8_C(  27), INT8_C(  22), INT8_C(   3), INT8_C( -37),
                           INT8_C(  65), INT8_C( 118), INT8_C(-126), INT8_C(  97),
                           INT8_C( 109), INT8_C(   7), INT8_C(-114), INT8_C( -75),
                           INT8_C(-125), INT8_C( 121), INT8_C(-128), INT8_C( 103),
                           INT8_C(   0), INT8_C( 101), INT8_C( -41), INT8_C(  89),
                           INT8_C( -83), INT8_C( -65), INT8_C(   9), INT8_C(  -7),
                           INT8_C( -63), INT8_C(  13), INT8_C( 105), INT8_C(  92),
                           INT8_C( -18), INT8_C( -21), INT8_C(-102), INT8_C(-114),
                           INT8_C(  74), INT8_C( 121), INT8_C( -45), INT8_C(  52),
                           INT8_C( -63), INT8_C( -93), INT8_C(  98), INT8_C( 106),
                           INT8_C(-109), INT8_C( -47), INT8_C(  37), INT8_C(  70),
                           INT8_C( 100), INT8_C( 121), INT8_C(  18), INT8_C(  28),
                           INT8_C(-117), INT8_C( 107), INT8_C(   3), INT8_C( -62),
                           INT8_C(  42), INT8_C(  72), INT8_C(  91), INT8_C(  86),
                           INT8_C( -72), INT8_C(   9), INT8_C( -80), INT8_C( 118)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C( -70), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -73), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -70), INT8_C( -73), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(-122),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(-119),
                           INT8_C(   0), INT8_C(   0), INT8_C(-119), INT8_C(   0),
                           INT8_C(   0), INT8_C(  66), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -75), INT8_C( -34), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -94), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-124), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-103), INT8_C(   0), INT8_C(   0),
                           INT8_C( -48), INT8_C(   0), INT8_C(-103), INT8_C(   9),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   9)) },
    { UINT64_C(17498311407133456191),
      easysimd_mm512_set_epi8(INT8_C(  61), INT8_C( -89), INT8_C(  84), INT8_C( -94),
                           INT8_C(  31), INT8_C(-116), INT8_C( -67), INT8_C(-102),
                           INT8_C( -72), INT8_C( -91), INT8_C(-105), INT8_C(-108),
                           INT8_C( -44), INT8_C(  74), INT8_C( -28), INT8_C( 124),
                           INT8_C( 120), INT8_C( -41), INT8_C( -79), INT8_C( 122),
                           INT8_C(  87), INT8_C(-119), INT8_C(  54), INT8_C(  -2),
                           INT8_C( -47), INT8_C(  84), INT8_C(-126), INT8_C( -64),
                           INT8_C(  14), INT8_C(  11), INT8_C(  37), INT8_C( -23),
                           INT8_C(  67), INT8_C( 124), INT8_C(  58), INT8_C( -94),
                           INT8_C(  30), INT8_C( -33), INT8_C(  70), INT8_C( -24),
                           INT8_C(  38), INT8_C( -97), INT8_C( -56), INT8_C( -60),
                           INT8_C( -59), INT8_C(  65), INT8_C( -74), INT8_C(  45),
                           INT8_C( -11), INT8_C(  55), INT8_C( -82), INT8_C(  12),
                           INT8_C( 106), INT8_C(  22), INT8_C(-124), INT8_C(  -4),
                           INT8_C(   2), INT8_C( -81), INT8_C(  14), INT8_C(  90),
                           INT8_C(-100), INT8_C(-122), INT8_C( -35), INT8_C(  81)),
      easysimd_mm512_set_epi8(INT8_C(  21), INT8_C( 108), INT8_C( 125), INT8_C( -60),
                           INT8_C( -72), INT8_C(  74), INT8_C(  -5), INT8_C( -58),
                           INT8_C( -41), INT8_C(  22), INT8_C(-115), INT8_C( 102),
                           INT8_C(  59), INT8_C( -80), INT8_C( -15), INT8_C( -63),
                           INT8_C(  84), INT8_C( -71), INT8_C(   8), INT8_C(  12),
                           INT8_C( -11), INT8_C( -76), INT8_C(  62), INT8_C(  93),
                           INT8_C( -75), INT8_C( -77), INT8_C( -84), INT8_C(-108),
                           INT8_C( -35), INT8_C(  14), INT8_C( -60), INT8_C(  18),
                           INT8_C(  23), INT8_C( -60), INT8_C( -63), INT8_C(-114),
                           INT8_C( -55), INT8_C(  75), INT8_C( -99), INT8_C( -55),
                           INT8_C(  58), INT8_C(  76), INT8_C(-102), INT8_C(-118),
                           INT8_C( -30), INT8_C(  39), INT8_C( 119), INT8_C(  85),
                           INT8_C(  -8), INT8_C( -72), INT8_C( -60), INT8_C( -94),
                           INT8_C(-112), INT8_C( 119), INT8_C( 124), INT8_C(  76),
                           INT8_C( -42), INT8_C(-124), INT8_C(  54), INT8_C(  74),
                           INT8_C( -92), INT8_C(  99), INT8_C(  79), INT8_C(  -3)),
      easysimd_mm512_set_epi8(INT8_C(-105), INT8_C( -94), INT8_C(  84), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -91), INT8_C(   0), INT8_C( -91),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  -2), INT8_C( 122),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -79),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  11),
                           INT8_C(  38), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  30), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -94), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  38), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   2), INT8_C(  12), INT8_C(  12),
                           INT8_C(   0), INT8_C(   0), INT8_C( -81), INT8_C(  22),
                           INT8_C(   0), INT8_C(-100), INT8_C( -11), INT8_C(   0)) },
    { UINT64_C( 3462613925466476941),
      easysimd_mm512_set_epi8(INT8_C(  18), INT8_C( -74), INT8_C(  14), INT8_C(  36),
                           INT8_C(  -7), INT8_C( 113), INT8_C(  40), INT8_C(  48),
                           INT8_C(-107), INT8_C( -34), INT8_C( -75), INT8_C(  85),
                           INT8_C( -35), INT8_C(-116), INT8_C(  65), INT8_C( -21),
                           INT8_C(  15), INT8_C(   3), INT8_C(  45), INT8_C(  21),
                           INT8_C(  72), INT8_C(  93), INT8_C( 108), INT8_C( 125),
                           INT8_C(   1), INT8_C(  75), INT8_C(  21), INT8_C( -36),
                           INT8_C(-126), INT8_C( 122), INT8_C(  71), INT8_C(  76),
                           INT8_C(  28), INT8_C( -56), INT8_C(  32), INT8_C( 101),
                           INT8_C(-107), INT8_C(-111), INT8_C( -88), INT8_C( -19),
                           INT8_C( -77), INT8_C(  19), INT8_C( -21), INT8_C(-111),
                           INT8_C( -68), INT8_C(  82), INT8_C(-118), INT8_C( -76),
                           INT8_C(  47), INT8_C( 127), INT8_C(  62), INT8_C( -16),
                           INT8_C(  10), INT8_C( -14), INT8_C(-100), INT8_C(  86),
                           INT8_C(  29), INT8_C( 107), INT8_C(  56), INT8_C(  21),
                           INT8_C(  24), INT8_C(  68), INT8_C( -96), INT8_C(  64)),
      easysimd_mm512_set_epi8(INT8_C( -57), INT8_C(  94), INT8_C(-125), INT8_C(-124),
                           INT8_C(  36), INT8_C(  57), INT8_C(  68), INT8_C( -52),
                           INT8_C(  39), INT8_C(  50), INT8_C( -48), INT8_C(  94),
                           INT8_C(  53), INT8_C(  11), INT8_C(  29), INT8_C(  65),
                           INT8_C( -82), INT8_C(-119), INT8_C( 115), INT8_C(  68),
                           INT8_C( -69), INT8_C( 118), INT8_C(  17), INT8_C(   8),
                           INT8_C( -64), INT8_C( -53), INT8_C( -42), INT8_C( 126),
                           INT8_C(  67), INT8_C(  50), INT8_C( -18), INT8_C(  76),
                           INT8_C( -19), INT8_C( 123), INT8_C( -87), INT8_C( 106),
                           INT8_C( -74), INT8_C(  44), INT8_C( 117), INT8_C( 103),
                           INT8_C(  81), INT8_C( 122), INT8_C(  56), INT8_C( -10),
                           INT8_C(  67), INT8_C(  79), INT8_C(  83), INT8_C( -38),
                           INT8_C( -13), INT8_C(  43), INT8_C(  27), INT8_C( -97),
                           INT8_C( 102), INT8_C( 126), INT8_C(  38), INT8_C( -62),
                           INT8_C( -24), INT8_C( 117), INT8_C( -38), INT8_C( -93),
                           INT8_C( -58), INT8_C(-124), INT8_C( -75), INT8_C(  10)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -75), INT8_C(  -7), INT8_C(   0), INT8_C(  65),
                           INT8_C(   0), INT8_C(   0), INT8_C(-126), INT8_C(   0),
                           INT8_C(   0), INT8_C(  75), INT8_C(   0), INT8_C( 125),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( 122), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-107), INT8_C(   0), INT8_C(-111),
                           INT8_C(   0), INT8_C( 101), INT8_C(   0), INT8_C( -77),
                           INT8_C(-118), INT8_C(-111), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -14)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_shuffle_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_shuffle_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_shuffle_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r0[16];
    const easysimd_float32 r1[16];
    const easysimd_float32 r2[16];
    const easysimd_float32 r3[16];
  } test_vec[4] = {
   { { EASYSIMD_FLOAT32_C(  -180.16), EASYSIMD_FLOAT32_C(  -789.86), EASYSIMD_FLOAT32_C(   107.29), EASYSIMD_FLOAT32_C(   489.11),
        EASYSIMD_FLOAT32_C(   914.56), EASYSIMD_FLOAT32_C(   797.57), EASYSIMD_FLOAT32_C(   982.30), EASYSIMD_FLOAT32_C(   320.59) },
      { EASYSIMD_FLOAT32_C(  -128.27), EASYSIMD_FLOAT32_C(  -299.60), EASYSIMD_FLOAT32_C(   846.68), EASYSIMD_FLOAT32_C(   701.28),
        EASYSIMD_FLOAT32_C(  -495.11), EASYSIMD_FLOAT32_C(  -859.93), EASYSIMD_FLOAT32_C(   964.29), EASYSIMD_FLOAT32_C(   599.86) },
      { EASYSIMD_FLOAT32_C(  -180.16), EASYSIMD_FLOAT32_C(  -789.86), EASYSIMD_FLOAT32_C(   107.29), EASYSIMD_FLOAT32_C(   489.11),
        EASYSIMD_FLOAT32_C(  -128.27), EASYSIMD_FLOAT32_C(  -299.60), EASYSIMD_FLOAT32_C(   846.68), EASYSIMD_FLOAT32_C(   701.28) },
      { EASYSIMD_FLOAT32_C(   914.56), EASYSIMD_FLOAT32_C(   797.57), EASYSIMD_FLOAT32_C(   982.30), EASYSIMD_FLOAT32_C(   320.59),
        EASYSIMD_FLOAT32_C(  -128.27), EASYSIMD_FLOAT32_C(  -299.60), EASYSIMD_FLOAT32_C(   846.68), EASYSIMD_FLOAT32_C(   701.28) },
      { EASYSIMD_FLOAT32_C(  -180.16), EASYSIMD_FLOAT32_C(  -789.86), EASYSIMD_FLOAT32_C(   107.29), EASYSIMD_FLOAT32_C(   489.11),
        EASYSIMD_FLOAT32_C(  -495.11), EASYSIMD_FLOAT32_C(  -859.93), EASYSIMD_FLOAT32_C(   964.29), EASYSIMD_FLOAT32_C(   599.86) },
      { EASYSIMD_FLOAT32_C(   914.56), EASYSIMD_FLOAT32_C(   797.57), EASYSIMD_FLOAT32_C(   982.30), EASYSIMD_FLOAT32_C(   320.59),
        EASYSIMD_FLOAT32_C(  -495.11), EASYSIMD_FLOAT32_C(  -859.93), EASYSIMD_FLOAT32_C(   964.29), EASYSIMD_FLOAT32_C(   599.86) } },
    { { EASYSIMD_FLOAT32_C(   619.10), EASYSIMD_FLOAT32_C(  -510.63), EASYSIMD_FLOAT32_C(   541.95), EASYSIMD_FLOAT32_C(   148.21),
        EASYSIMD_FLOAT32_C(  -639.14), EASYSIMD_FLOAT32_C(  -554.80), EASYSIMD_FLOAT32_C(   761.16), EASYSIMD_FLOAT32_C(   -78.91) },
      { EASYSIMD_FLOAT32_C(   645.85), EASYSIMD_FLOAT32_C(   405.43), EASYSIMD_FLOAT32_C(   111.06), EASYSIMD_FLOAT32_C(  -405.34),
        EASYSIMD_FLOAT32_C(   -37.23), EASYSIMD_FLOAT32_C(   -49.39), EASYSIMD_FLOAT32_C(   669.49), EASYSIMD_FLOAT32_C(   782.61) },
      { EASYSIMD_FLOAT32_C(   619.10), EASYSIMD_FLOAT32_C(  -510.63), EASYSIMD_FLOAT32_C(   541.95), EASYSIMD_FLOAT32_C(   148.21),
        EASYSIMD_FLOAT32_C(   645.85), EASYSIMD_FLOAT32_C(   405.43), EASYSIMD_FLOAT32_C(   111.06), EASYSIMD_FLOAT32_C(  -405.34) },
      { EASYSIMD_FLOAT32_C(  -639.14), EASYSIMD_FLOAT32_C(  -554.80), EASYSIMD_FLOAT32_C(   761.16), EASYSIMD_FLOAT32_C(   -78.91),
        EASYSIMD_FLOAT32_C(   645.85), EASYSIMD_FLOAT32_C(   405.43), EASYSIMD_FLOAT32_C(   111.06), EASYSIMD_FLOAT32_C(  -405.34) },
      { EASYSIMD_FLOAT32_C(   619.10), EASYSIMD_FLOAT32_C(  -510.63), EASYSIMD_FLOAT32_C(   541.95), EASYSIMD_FLOAT32_C(   148.21),
        EASYSIMD_FLOAT32_C(   -37.23), EASYSIMD_FLOAT32_C(   -49.39), EASYSIMD_FLOAT32_C(   669.49), EASYSIMD_FLOAT32_C(   782.61) },
      { EASYSIMD_FLOAT32_C(  -639.14), EASYSIMD_FLOAT32_C(  -554.80), EASYSIMD_FLOAT32_C(   761.16), EASYSIMD_FLOAT32_C(   -78.91),
        EASYSIMD_FLOAT32_C(   -37.23), EASYSIMD_FLOAT32_C(   -49.39), EASYSIMD_FLOAT32_C(   669.49), EASYSIMD_FLOAT32_C(   782.61) } },
    { { EASYSIMD_FLOAT32_C(   160.75), EASYSIMD_FLOAT32_C(  -223.22), EASYSIMD_FLOAT32_C(   271.72), EASYSIMD_FLOAT32_C(    75.31),
        EASYSIMD_FLOAT32_C(  -425.65), EASYSIMD_FLOAT32_C(   254.02), EASYSIMD_FLOAT32_C(  -604.10), EASYSIMD_FLOAT32_C(   446.08) },
      { EASYSIMD_FLOAT32_C(   954.41), EASYSIMD_FLOAT32_C(  -757.42), EASYSIMD_FLOAT32_C(   147.36), EASYSIMD_FLOAT32_C(  -540.70),
        EASYSIMD_FLOAT32_C(  -617.36), EASYSIMD_FLOAT32_C(   111.65), EASYSIMD_FLOAT32_C(  -940.84), EASYSIMD_FLOAT32_C(  -998.26) },
      { EASYSIMD_FLOAT32_C(   160.75), EASYSIMD_FLOAT32_C(  -223.22), EASYSIMD_FLOAT32_C(   271.72), EASYSIMD_FLOAT32_C(    75.31),
        EASYSIMD_FLOAT32_C(   954.41), EASYSIMD_FLOAT32_C(  -757.42), EASYSIMD_FLOAT32_C(   147.36), EASYSIMD_FLOAT32_C(  -540.70) },
      { EASYSIMD_FLOAT32_C(  -425.65), EASYSIMD_FLOAT32_C(   254.02), EASYSIMD_FLOAT32_C(  -604.10), EASYSIMD_FLOAT32_C(   446.08),
        EASYSIMD_FLOAT32_C(   954.41), EASYSIMD_FLOAT32_C(  -757.42), EASYSIMD_FLOAT32_C(   147.36), EASYSIMD_FLOAT32_C(  -540.70) },
      { EASYSIMD_FLOAT32_C(   160.75), EASYSIMD_FLOAT32_C(  -223.22), EASYSIMD_FLOAT32_C(   271.72), EASYSIMD_FLOAT32_C(    75.31),
        EASYSIMD_FLOAT32_C(  -617.36), EASYSIMD_FLOAT32_C(   111.65), EASYSIMD_FLOAT32_C(  -940.84), EASYSIMD_FLOAT32_C(  -998.26) },
      { EASYSIMD_FLOAT32_C(  -425.65), EASYSIMD_FLOAT32_C(   254.02), EASYSIMD_FLOAT32_C(  -604.10), EASYSIMD_FLOAT32_C(   446.08),
        EASYSIMD_FLOAT32_C(  -617.36), EASYSIMD_FLOAT32_C(   111.65), EASYSIMD_FLOAT32_C(  -940.84), EASYSIMD_FLOAT32_C(  -998.26) } },
    { { EASYSIMD_FLOAT32_C(   601.02), EASYSIMD_FLOAT32_C(   601.12), EASYSIMD_FLOAT32_C(   149.95), EASYSIMD_FLOAT32_C(   961.88),
        EASYSIMD_FLOAT32_C(  -953.68), EASYSIMD_FLOAT32_C(   -88.89), EASYSIMD_FLOAT32_C(  -117.03), EASYSIMD_FLOAT32_C(   692.17) },
      { EASYSIMD_FLOAT32_C(  -683.46), EASYSIMD_FLOAT32_C(   994.03), EASYSIMD_FLOAT32_C(  -713.17), EASYSIMD_FLOAT32_C(   279.31),
        EASYSIMD_FLOAT32_C(   -55.35), EASYSIMD_FLOAT32_C(   956.32), EASYSIMD_FLOAT32_C(    61.92), EASYSIMD_FLOAT32_C(  -894.60) },
      { EASYSIMD_FLOAT32_C(   601.02), EASYSIMD_FLOAT32_C(   601.12), EASYSIMD_FLOAT32_C(   149.95), EASYSIMD_FLOAT32_C(   961.88),
        EASYSIMD_FLOAT32_C(  -683.46), EASYSIMD_FLOAT32_C(   994.03), EASYSIMD_FLOAT32_C(  -713.17), EASYSIMD_FLOAT32_C(   279.31) },
      { EASYSIMD_FLOAT32_C(  -953.68), EASYSIMD_FLOAT32_C(   -88.89), EASYSIMD_FLOAT32_C(  -117.03), EASYSIMD_FLOAT32_C(   692.17),
        EASYSIMD_FLOAT32_C(  -683.46), EASYSIMD_FLOAT32_C(   994.03), EASYSIMD_FLOAT32_C(  -713.17), EASYSIMD_FLOAT32_C(   279.31) },
      { EASYSIMD_FLOAT32_C(   601.02), EASYSIMD_FLOAT32_C(   601.12), EASYSIMD_FLOAT32_C(   149.95), EASYSIMD_FLOAT32_C(   961.88),
        EASYSIMD_FLOAT32_C(   -55.35), EASYSIMD_FLOAT32_C(   956.32), EASYSIMD_FLOAT32_C(    61.92), EASYSIMD_FLOAT32_C(  -894.60) },
      { EASYSIMD_FLOAT32_C(  -953.68), EASYSIMD_FLOAT32_C(   -88.89), EASYSIMD_FLOAT32_C(  -117.03), EASYSIMD_FLOAT32_C(   692.17),
        EASYSIMD_FLOAT32_C(   -55.35), EASYSIMD_FLOAT32_C(   956.32), EASYSIMD_FLOAT32_C(    61.92), EASYSIMD_FLOAT32_C(  -894.60) } },
   };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;

    r = easysimd_mm256_shuffle_f32x4(a, b, 0);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r0), 1);
    r = easysimd_mm256_shuffle_f32x4(a, b, 1);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r1), 1);
    r = easysimd_mm256_shuffle_f32x4(a, b, 2);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r2), 1);
    r = easysimd_mm256_shuffle_f32x4(a, b, 3);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r;

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f32x4(a, b, 0);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f32x4(a, b, 1);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f32x4(a, b, 2);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f32x4(a, b, 3);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_shuffle_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[8];
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r0[8];
    const easysimd_float32 r1[8];
    const easysimd_float32 r2[8];
    const easysimd_float32 r3[8];
  } test_vec[4] = {
    { { EASYSIMD_FLOAT32_C(  -410.56), EASYSIMD_FLOAT32_C(   408.21), EASYSIMD_FLOAT32_C(  -376.06), EASYSIMD_FLOAT32_C(  -599.18),
        EASYSIMD_FLOAT32_C(   884.20), EASYSIMD_FLOAT32_C(  -695.71), EASYSIMD_FLOAT32_C(   448.37), EASYSIMD_FLOAT32_C(   207.77) },
      UINT8_C(145),
      { EASYSIMD_FLOAT32_C(   700.72), EASYSIMD_FLOAT32_C(   963.37), EASYSIMD_FLOAT32_C(   188.56), EASYSIMD_FLOAT32_C(  -599.52),
        EASYSIMD_FLOAT32_C(   860.41), EASYSIMD_FLOAT32_C(   527.11), EASYSIMD_FLOAT32_C(  -761.05), EASYSIMD_FLOAT32_C(   372.33) },
      { EASYSIMD_FLOAT32_C(  -711.48), EASYSIMD_FLOAT32_C(  -848.21), EASYSIMD_FLOAT32_C(   434.42), EASYSIMD_FLOAT32_C(   109.85),
        EASYSIMD_FLOAT32_C(   382.55), EASYSIMD_FLOAT32_C(   606.63), EASYSIMD_FLOAT32_C(  -450.37), EASYSIMD_FLOAT32_C(  -569.99) },
      { EASYSIMD_FLOAT32_C(   700.72), EASYSIMD_FLOAT32_C(   408.21), EASYSIMD_FLOAT32_C(  -376.06), EASYSIMD_FLOAT32_C(  -599.18),
        EASYSIMD_FLOAT32_C(  -711.48), EASYSIMD_FLOAT32_C(  -695.71), EASYSIMD_FLOAT32_C(   448.37), EASYSIMD_FLOAT32_C(   109.85) },
      { EASYSIMD_FLOAT32_C(   860.41), EASYSIMD_FLOAT32_C(   408.21), EASYSIMD_FLOAT32_C(  -376.06), EASYSIMD_FLOAT32_C(  -599.18),
        EASYSIMD_FLOAT32_C(  -711.48), EASYSIMD_FLOAT32_C(  -695.71), EASYSIMD_FLOAT32_C(   448.37), EASYSIMD_FLOAT32_C(   109.85) },
      { EASYSIMD_FLOAT32_C(   700.72), EASYSIMD_FLOAT32_C(   408.21), EASYSIMD_FLOAT32_C(  -376.06), EASYSIMD_FLOAT32_C(  -599.18),
        EASYSIMD_FLOAT32_C(   382.55), EASYSIMD_FLOAT32_C(  -695.71), EASYSIMD_FLOAT32_C(   448.37), EASYSIMD_FLOAT32_C(  -569.99) },
      { EASYSIMD_FLOAT32_C(   860.41), EASYSIMD_FLOAT32_C(   408.21), EASYSIMD_FLOAT32_C(  -376.06), EASYSIMD_FLOAT32_C(  -599.18),
        EASYSIMD_FLOAT32_C(   382.55), EASYSIMD_FLOAT32_C(  -695.71), EASYSIMD_FLOAT32_C(   448.37), EASYSIMD_FLOAT32_C(  -569.99) } },
    { { EASYSIMD_FLOAT32_C(   673.95), EASYSIMD_FLOAT32_C(   135.70), EASYSIMD_FLOAT32_C(  -124.67), EASYSIMD_FLOAT32_C(  -944.44),
        EASYSIMD_FLOAT32_C(   326.62), EASYSIMD_FLOAT32_C(  -646.51), EASYSIMD_FLOAT32_C(  -355.00), EASYSIMD_FLOAT32_C(  -265.17) },
      UINT8_C( 56),
      { EASYSIMD_FLOAT32_C(    45.82), EASYSIMD_FLOAT32_C(  -380.97), EASYSIMD_FLOAT32_C(   281.73), EASYSIMD_FLOAT32_C(  -505.81),
        EASYSIMD_FLOAT32_C(   826.80), EASYSIMD_FLOAT32_C(   650.60), EASYSIMD_FLOAT32_C(  -805.09), EASYSIMD_FLOAT32_C(   790.17) },
      { EASYSIMD_FLOAT32_C(  -160.84), EASYSIMD_FLOAT32_C(  -404.61), EASYSIMD_FLOAT32_C(   650.58), EASYSIMD_FLOAT32_C(  -633.73),
        EASYSIMD_FLOAT32_C(  -165.66), EASYSIMD_FLOAT32_C(    22.91), EASYSIMD_FLOAT32_C(  -345.22), EASYSIMD_FLOAT32_C(   -13.87) },
      { EASYSIMD_FLOAT32_C(   673.95), EASYSIMD_FLOAT32_C(   135.70), EASYSIMD_FLOAT32_C(  -124.67), EASYSIMD_FLOAT32_C(  -505.81),
        EASYSIMD_FLOAT32_C(  -160.84), EASYSIMD_FLOAT32_C(  -404.61), EASYSIMD_FLOAT32_C(  -355.00), EASYSIMD_FLOAT32_C(  -265.17) },
      { EASYSIMD_FLOAT32_C(   673.95), EASYSIMD_FLOAT32_C(   135.70), EASYSIMD_FLOAT32_C(  -124.67), EASYSIMD_FLOAT32_C(   790.17),
        EASYSIMD_FLOAT32_C(  -160.84), EASYSIMD_FLOAT32_C(  -404.61), EASYSIMD_FLOAT32_C(  -355.00), EASYSIMD_FLOAT32_C(  -265.17) },
      { EASYSIMD_FLOAT32_C(   673.95), EASYSIMD_FLOAT32_C(   135.70), EASYSIMD_FLOAT32_C(  -124.67), EASYSIMD_FLOAT32_C(  -505.81),
        EASYSIMD_FLOAT32_C(  -165.66), EASYSIMD_FLOAT32_C(    22.91), EASYSIMD_FLOAT32_C(  -355.00), EASYSIMD_FLOAT32_C(  -265.17) },
      { EASYSIMD_FLOAT32_C(   673.95), EASYSIMD_FLOAT32_C(   135.70), EASYSIMD_FLOAT32_C(  -124.67), EASYSIMD_FLOAT32_C(   790.17),
        EASYSIMD_FLOAT32_C(  -165.66), EASYSIMD_FLOAT32_C(    22.91), EASYSIMD_FLOAT32_C(  -355.00), EASYSIMD_FLOAT32_C(  -265.17) } },
    { { EASYSIMD_FLOAT32_C(  -542.67), EASYSIMD_FLOAT32_C(   764.64), EASYSIMD_FLOAT32_C(  -631.31), EASYSIMD_FLOAT32_C(  -936.05),
        EASYSIMD_FLOAT32_C(  -685.73), EASYSIMD_FLOAT32_C(  -201.30), EASYSIMD_FLOAT32_C(   737.90), EASYSIMD_FLOAT32_C(   449.97) },
      UINT8_C(251),
      { EASYSIMD_FLOAT32_C(   793.46), EASYSIMD_FLOAT32_C(  -223.41), EASYSIMD_FLOAT32_C(  -972.48), EASYSIMD_FLOAT32_C(  -561.54),
        EASYSIMD_FLOAT32_C(   511.42), EASYSIMD_FLOAT32_C(     4.96), EASYSIMD_FLOAT32_C(   484.27), EASYSIMD_FLOAT32_C(  -869.55) },
      { EASYSIMD_FLOAT32_C(  -713.31), EASYSIMD_FLOAT32_C(   978.46), EASYSIMD_FLOAT32_C(   957.25), EASYSIMD_FLOAT32_C(   937.29),
        EASYSIMD_FLOAT32_C(  -826.63), EASYSIMD_FLOAT32_C(   747.42), EASYSIMD_FLOAT32_C(  -223.55), EASYSIMD_FLOAT32_C(  -231.24) },
      { EASYSIMD_FLOAT32_C(   793.46), EASYSIMD_FLOAT32_C(  -223.41), EASYSIMD_FLOAT32_C(  -631.31), EASYSIMD_FLOAT32_C(  -561.54),
        EASYSIMD_FLOAT32_C(  -713.31), EASYSIMD_FLOAT32_C(   978.46), EASYSIMD_FLOAT32_C(   957.25), EASYSIMD_FLOAT32_C(   937.29) },
      { EASYSIMD_FLOAT32_C(   511.42), EASYSIMD_FLOAT32_C(     4.96), EASYSIMD_FLOAT32_C(  -631.31), EASYSIMD_FLOAT32_C(  -869.55),
        EASYSIMD_FLOAT32_C(  -713.31), EASYSIMD_FLOAT32_C(   978.46), EASYSIMD_FLOAT32_C(   957.25), EASYSIMD_FLOAT32_C(   937.29) },
      { EASYSIMD_FLOAT32_C(   793.46), EASYSIMD_FLOAT32_C(  -223.41), EASYSIMD_FLOAT32_C(  -631.31), EASYSIMD_FLOAT32_C(  -561.54),
        EASYSIMD_FLOAT32_C(  -826.63), EASYSIMD_FLOAT32_C(   747.42), EASYSIMD_FLOAT32_C(  -223.55), EASYSIMD_FLOAT32_C(  -231.24) },
      { EASYSIMD_FLOAT32_C(   511.42), EASYSIMD_FLOAT32_C(     4.96), EASYSIMD_FLOAT32_C(  -631.31), EASYSIMD_FLOAT32_C(  -869.55),
        EASYSIMD_FLOAT32_C(  -826.63), EASYSIMD_FLOAT32_C(   747.42), EASYSIMD_FLOAT32_C(  -223.55), EASYSIMD_FLOAT32_C(  -231.24) } },
    { { EASYSIMD_FLOAT32_C(   398.00), EASYSIMD_FLOAT32_C(   142.72), EASYSIMD_FLOAT32_C(   603.11), EASYSIMD_FLOAT32_C(  -579.09),
        EASYSIMD_FLOAT32_C(   797.50), EASYSIMD_FLOAT32_C(  -410.76), EASYSIMD_FLOAT32_C(  -121.76), EASYSIMD_FLOAT32_C(   562.14) },
      UINT8_C( 18),
      { EASYSIMD_FLOAT32_C(   -57.81), EASYSIMD_FLOAT32_C(   876.41), EASYSIMD_FLOAT32_C(   756.63), EASYSIMD_FLOAT32_C(  -319.91),
        EASYSIMD_FLOAT32_C(   326.38), EASYSIMD_FLOAT32_C(   430.66), EASYSIMD_FLOAT32_C(  -526.45), EASYSIMD_FLOAT32_C(  -897.02) },
      { EASYSIMD_FLOAT32_C(   458.18), EASYSIMD_FLOAT32_C(   -88.00), EASYSIMD_FLOAT32_C(   614.40), EASYSIMD_FLOAT32_C(  -536.86),
        EASYSIMD_FLOAT32_C(  -603.72), EASYSIMD_FLOAT32_C(   744.85), EASYSIMD_FLOAT32_C(  -250.16), EASYSIMD_FLOAT32_C(  -625.26) },
      { EASYSIMD_FLOAT32_C(   398.00), EASYSIMD_FLOAT32_C(   876.41), EASYSIMD_FLOAT32_C(   603.11), EASYSIMD_FLOAT32_C(  -579.09),
        EASYSIMD_FLOAT32_C(   458.18), EASYSIMD_FLOAT32_C(  -410.76), EASYSIMD_FLOAT32_C(  -121.76), EASYSIMD_FLOAT32_C(   562.14) },
      { EASYSIMD_FLOAT32_C(   398.00), EASYSIMD_FLOAT32_C(   430.66), EASYSIMD_FLOAT32_C(   603.11), EASYSIMD_FLOAT32_C(  -579.09),
        EASYSIMD_FLOAT32_C(   458.18), EASYSIMD_FLOAT32_C(  -410.76), EASYSIMD_FLOAT32_C(  -121.76), EASYSIMD_FLOAT32_C(   562.14) },
      { EASYSIMD_FLOAT32_C(   398.00), EASYSIMD_FLOAT32_C(   876.41), EASYSIMD_FLOAT32_C(   603.11), EASYSIMD_FLOAT32_C(  -579.09),
        EASYSIMD_FLOAT32_C(  -603.72), EASYSIMD_FLOAT32_C(  -410.76), EASYSIMD_FLOAT32_C(  -121.76), EASYSIMD_FLOAT32_C(   562.14) },
      { EASYSIMD_FLOAT32_C(   398.00), EASYSIMD_FLOAT32_C(   430.66), EASYSIMD_FLOAT32_C(   603.11), EASYSIMD_FLOAT32_C(  -579.09),
        EASYSIMD_FLOAT32_C(  -603.72), EASYSIMD_FLOAT32_C(  -410.76), EASYSIMD_FLOAT32_C(  -121.76), EASYSIMD_FLOAT32_C(   562.14) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;

    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 0);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r0), 1);
    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 1);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r1), 1);
    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 2);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r2), 1);
    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 3);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r;

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 0);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 1);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 2);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f32x4(src, k, a, b, 3);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_shuffle_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r0[8];
    const easysimd_float32 r1[8];
    const easysimd_float32 r2[8];
    const easysimd_float32 r3[8];
  } test_vec[4] = {
    { UINT8_C( 37),
      { EASYSIMD_FLOAT32_C(  -503.55), EASYSIMD_FLOAT32_C(  -649.65), EASYSIMD_FLOAT32_C(   910.61), EASYSIMD_FLOAT32_C(   771.78),
        EASYSIMD_FLOAT32_C(  -605.93), EASYSIMD_FLOAT32_C(  -622.75), EASYSIMD_FLOAT32_C(   832.26), EASYSIMD_FLOAT32_C(   766.22) },
      { EASYSIMD_FLOAT32_C(  -841.56), EASYSIMD_FLOAT32_C(   991.01), EASYSIMD_FLOAT32_C(  -280.43), EASYSIMD_FLOAT32_C(  -392.32),
        EASYSIMD_FLOAT32_C(  -589.75), EASYSIMD_FLOAT32_C(  -912.19), EASYSIMD_FLOAT32_C(  -591.95), EASYSIMD_FLOAT32_C(   -83.34) },
      { EASYSIMD_FLOAT32_C(  -503.55), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   910.61), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   991.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -605.93), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   832.26), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   991.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -503.55), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   910.61), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -912.19), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -605.93), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   832.26), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -912.19), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(137),
      { EASYSIMD_FLOAT32_C(   551.56), EASYSIMD_FLOAT32_C(  -923.11), EASYSIMD_FLOAT32_C(  -542.77), EASYSIMD_FLOAT32_C(  -505.26),
        EASYSIMD_FLOAT32_C(   797.95), EASYSIMD_FLOAT32_C(  -348.51), EASYSIMD_FLOAT32_C(   505.07), EASYSIMD_FLOAT32_C(   188.52) },
      { EASYSIMD_FLOAT32_C(    -7.34), EASYSIMD_FLOAT32_C(  -957.28), EASYSIMD_FLOAT32_C(   510.02), EASYSIMD_FLOAT32_C(   615.87),
        EASYSIMD_FLOAT32_C(   148.64), EASYSIMD_FLOAT32_C(  -767.81), EASYSIMD_FLOAT32_C(  -887.67), EASYSIMD_FLOAT32_C(   499.00) },
      { EASYSIMD_FLOAT32_C(   551.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -505.26),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   615.87) },
      { EASYSIMD_FLOAT32_C(   797.95), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   188.52),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   615.87) },
      { EASYSIMD_FLOAT32_C(   551.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -505.26),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   499.00) },
      { EASYSIMD_FLOAT32_C(   797.95), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   188.52),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   499.00) } },
    { UINT8_C(123),
      { EASYSIMD_FLOAT32_C(   884.11), EASYSIMD_FLOAT32_C(   893.06), EASYSIMD_FLOAT32_C(  -479.94), EASYSIMD_FLOAT32_C(   716.37),
        EASYSIMD_FLOAT32_C(   659.28), EASYSIMD_FLOAT32_C(  -321.51), EASYSIMD_FLOAT32_C(   707.38), EASYSIMD_FLOAT32_C(  -621.15) },
      { EASYSIMD_FLOAT32_C(   286.17), EASYSIMD_FLOAT32_C(  -882.37), EASYSIMD_FLOAT32_C(  -533.34), EASYSIMD_FLOAT32_C(   694.23),
        EASYSIMD_FLOAT32_C(    34.29), EASYSIMD_FLOAT32_C(  -894.33), EASYSIMD_FLOAT32_C(   245.79), EASYSIMD_FLOAT32_C(   111.18) },
      { EASYSIMD_FLOAT32_C(   884.11), EASYSIMD_FLOAT32_C(   893.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   716.37),
        EASYSIMD_FLOAT32_C(   286.17), EASYSIMD_FLOAT32_C(  -882.37), EASYSIMD_FLOAT32_C(  -533.34), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(   659.28), EASYSIMD_FLOAT32_C(  -321.51), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -621.15),
        EASYSIMD_FLOAT32_C(   286.17), EASYSIMD_FLOAT32_C(  -882.37), EASYSIMD_FLOAT32_C(  -533.34), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(   884.11), EASYSIMD_FLOAT32_C(   893.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   716.37),
        EASYSIMD_FLOAT32_C(    34.29), EASYSIMD_FLOAT32_C(  -894.33), EASYSIMD_FLOAT32_C(   245.79), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(   659.28), EASYSIMD_FLOAT32_C(  -321.51), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -621.15),
        EASYSIMD_FLOAT32_C(    34.29), EASYSIMD_FLOAT32_C(  -894.33), EASYSIMD_FLOAT32_C(   245.79), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(192),
      { EASYSIMD_FLOAT32_C(   740.53), EASYSIMD_FLOAT32_C(   -90.87), EASYSIMD_FLOAT32_C(   214.39), EASYSIMD_FLOAT32_C(   245.59),
        EASYSIMD_FLOAT32_C(  -902.35), EASYSIMD_FLOAT32_C(  -792.95), EASYSIMD_FLOAT32_C(   288.32), EASYSIMD_FLOAT32_C(   607.67) },
      { EASYSIMD_FLOAT32_C(   822.92), EASYSIMD_FLOAT32_C(  -563.04), EASYSIMD_FLOAT32_C(   839.85), EASYSIMD_FLOAT32_C(   935.25),
        EASYSIMD_FLOAT32_C(   935.95), EASYSIMD_FLOAT32_C(   982.66), EASYSIMD_FLOAT32_C(   819.36), EASYSIMD_FLOAT32_C(   829.02) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   839.85), EASYSIMD_FLOAT32_C(   935.25) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   839.85), EASYSIMD_FLOAT32_C(   935.25) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   819.36), EASYSIMD_FLOAT32_C(   829.02) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   819.36), EASYSIMD_FLOAT32_C(   829.02) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;

    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 0);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r0), 1);
    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 1);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r1), 1);
    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 2);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r2), 1);
    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 3);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r;

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 0);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 1);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 2);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f32x4(k, a, b, 3);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_shuffle_f64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r0[4];
    const easysimd_float64 r1[4];
    const easysimd_float64 r2[4];
    const easysimd_float64 r3[4];
  } test_vec[4] = {
    { { EASYSIMD_FLOAT64_C(   830.90), EASYSIMD_FLOAT64_C(   926.38), EASYSIMD_FLOAT64_C(   812.31), EASYSIMD_FLOAT64_C(  -579.52) },
      { EASYSIMD_FLOAT64_C(   629.36), EASYSIMD_FLOAT64_C(  -269.59), EASYSIMD_FLOAT64_C(   113.88), EASYSIMD_FLOAT64_C(   874.17) },
      { EASYSIMD_FLOAT64_C(   830.90), EASYSIMD_FLOAT64_C(   926.38), EASYSIMD_FLOAT64_C(   629.36), EASYSIMD_FLOAT64_C(  -269.59) },
      { EASYSIMD_FLOAT64_C(   812.31), EASYSIMD_FLOAT64_C(  -579.52), EASYSIMD_FLOAT64_C(   629.36), EASYSIMD_FLOAT64_C(  -269.59) },
      { EASYSIMD_FLOAT64_C(   830.90), EASYSIMD_FLOAT64_C(   926.38), EASYSIMD_FLOAT64_C(   113.88), EASYSIMD_FLOAT64_C(   874.17) },
      { EASYSIMD_FLOAT64_C(   812.31), EASYSIMD_FLOAT64_C(  -579.52), EASYSIMD_FLOAT64_C(   113.88), EASYSIMD_FLOAT64_C(   874.17) } },
    { { EASYSIMD_FLOAT64_C(   865.84), EASYSIMD_FLOAT64_C(  -710.16), EASYSIMD_FLOAT64_C(   619.64), EASYSIMD_FLOAT64_C(   -47.96) },
      { EASYSIMD_FLOAT64_C(   272.44), EASYSIMD_FLOAT64_C(   425.20), EASYSIMD_FLOAT64_C(  -475.06), EASYSIMD_FLOAT64_C(   355.52) },
      { EASYSIMD_FLOAT64_C(   865.84), EASYSIMD_FLOAT64_C(  -710.16), EASYSIMD_FLOAT64_C(   272.44), EASYSIMD_FLOAT64_C(   425.20) },
      { EASYSIMD_FLOAT64_C(   619.64), EASYSIMD_FLOAT64_C(   -47.96), EASYSIMD_FLOAT64_C(   272.44), EASYSIMD_FLOAT64_C(   425.20) },
      { EASYSIMD_FLOAT64_C(   865.84), EASYSIMD_FLOAT64_C(  -710.16), EASYSIMD_FLOAT64_C(  -475.06), EASYSIMD_FLOAT64_C(   355.52) },
      { EASYSIMD_FLOAT64_C(   619.64), EASYSIMD_FLOAT64_C(   -47.96), EASYSIMD_FLOAT64_C(  -475.06), EASYSIMD_FLOAT64_C(   355.52) } },
    { { EASYSIMD_FLOAT64_C(   626.39), EASYSIMD_FLOAT64_C(   445.81), EASYSIMD_FLOAT64_C(  -682.23), EASYSIMD_FLOAT64_C(   942.94) },
      { EASYSIMD_FLOAT64_C(  -162.65), EASYSIMD_FLOAT64_C(   -82.05), EASYSIMD_FLOAT64_C(    14.57), EASYSIMD_FLOAT64_C(   671.89) },
      { EASYSIMD_FLOAT64_C(   626.39), EASYSIMD_FLOAT64_C(   445.81), EASYSIMD_FLOAT64_C(  -162.65), EASYSIMD_FLOAT64_C(   -82.05) },
      { EASYSIMD_FLOAT64_C(  -682.23), EASYSIMD_FLOAT64_C(   942.94), EASYSIMD_FLOAT64_C(  -162.65), EASYSIMD_FLOAT64_C(   -82.05) },
      { EASYSIMD_FLOAT64_C(   626.39), EASYSIMD_FLOAT64_C(   445.81), EASYSIMD_FLOAT64_C(    14.57), EASYSIMD_FLOAT64_C(   671.89) },
      { EASYSIMD_FLOAT64_C(  -682.23), EASYSIMD_FLOAT64_C(   942.94), EASYSIMD_FLOAT64_C(    14.57), EASYSIMD_FLOAT64_C(   671.89) } },
    { { EASYSIMD_FLOAT64_C(   458.27), EASYSIMD_FLOAT64_C(   217.29), EASYSIMD_FLOAT64_C(  -704.67), EASYSIMD_FLOAT64_C(  -316.48) },
      { EASYSIMD_FLOAT64_C(   549.40), EASYSIMD_FLOAT64_C(   639.72), EASYSIMD_FLOAT64_C(   943.11), EASYSIMD_FLOAT64_C(   380.30) },
      { EASYSIMD_FLOAT64_C(   458.27), EASYSIMD_FLOAT64_C(   217.29), EASYSIMD_FLOAT64_C(   549.40), EASYSIMD_FLOAT64_C(   639.72) },
      { EASYSIMD_FLOAT64_C(  -704.67), EASYSIMD_FLOAT64_C(  -316.48), EASYSIMD_FLOAT64_C(   549.40), EASYSIMD_FLOAT64_C(   639.72) },
      { EASYSIMD_FLOAT64_C(   458.27), EASYSIMD_FLOAT64_C(   217.29), EASYSIMD_FLOAT64_C(   943.11), EASYSIMD_FLOAT64_C(   380.30) },
      { EASYSIMD_FLOAT64_C(  -704.67), EASYSIMD_FLOAT64_C(  -316.48), EASYSIMD_FLOAT64_C(   943.11), EASYSIMD_FLOAT64_C(   380.30) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;

    r = easysimd_mm256_shuffle_f64x2(a, b, 0);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r0), 1);
    r = easysimd_mm256_shuffle_f64x2(a, b, 1);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r1), 1);
    r = easysimd_mm256_shuffle_f64x2(a, b, 2);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r2), 1);
    r = easysimd_mm256_shuffle_f64x2(a, b, 3);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r;

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f64x2(a, b, 0);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f64x2(a, b, 1);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f64x2(a, b, 2);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_f64x2(a, b, 3);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_shuffle_f64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float64 src[4];
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r0[4];
    const easysimd_float64 r1[4];
    const easysimd_float64 r2[4];
    const easysimd_float64 r3[4];
  } test_vec[4] = {
    { { EASYSIMD_FLOAT64_C(   211.42), EASYSIMD_FLOAT64_C(  -547.45), EASYSIMD_FLOAT64_C(  -286.14), EASYSIMD_FLOAT64_C(  -940.55) },
      UINT8_C( 76),
      { EASYSIMD_FLOAT64_C(  -611.26), EASYSIMD_FLOAT64_C(  -905.87), EASYSIMD_FLOAT64_C(   815.09), EASYSIMD_FLOAT64_C(  -241.57) },
      { EASYSIMD_FLOAT64_C(  -880.14), EASYSIMD_FLOAT64_C(  -483.26), EASYSIMD_FLOAT64_C(  -269.54), EASYSIMD_FLOAT64_C(   979.22) },
      { EASYSIMD_FLOAT64_C(   211.42), EASYSIMD_FLOAT64_C(  -547.45), EASYSIMD_FLOAT64_C(  -880.14), EASYSIMD_FLOAT64_C(  -483.26) },
      { EASYSIMD_FLOAT64_C(   211.42), EASYSIMD_FLOAT64_C(  -547.45), EASYSIMD_FLOAT64_C(  -880.14), EASYSIMD_FLOAT64_C(  -483.26) },
      { EASYSIMD_FLOAT64_C(   211.42), EASYSIMD_FLOAT64_C(  -547.45), EASYSIMD_FLOAT64_C(  -269.54), EASYSIMD_FLOAT64_C(   979.22) },
      { EASYSIMD_FLOAT64_C(   211.42), EASYSIMD_FLOAT64_C(  -547.45), EASYSIMD_FLOAT64_C(  -269.54), EASYSIMD_FLOAT64_C(   979.22) } },
    { { EASYSIMD_FLOAT64_C(  -556.92), EASYSIMD_FLOAT64_C(  -759.71), EASYSIMD_FLOAT64_C(   -65.30), EASYSIMD_FLOAT64_C(   731.13) },
      UINT8_C( 50),
      { EASYSIMD_FLOAT64_C(  -613.42), EASYSIMD_FLOAT64_C(   156.49), EASYSIMD_FLOAT64_C(  -801.68), EASYSIMD_FLOAT64_C(   886.07) },
      { EASYSIMD_FLOAT64_C(   749.49), EASYSIMD_FLOAT64_C(  -810.15), EASYSIMD_FLOAT64_C(  -492.14), EASYSIMD_FLOAT64_C(    99.86) },
      { EASYSIMD_FLOAT64_C(  -556.92), EASYSIMD_FLOAT64_C(   156.49), EASYSIMD_FLOAT64_C(   -65.30), EASYSIMD_FLOAT64_C(   731.13) },
      { EASYSIMD_FLOAT64_C(  -556.92), EASYSIMD_FLOAT64_C(   886.07), EASYSIMD_FLOAT64_C(   -65.30), EASYSIMD_FLOAT64_C(   731.13) },
      { EASYSIMD_FLOAT64_C(  -556.92), EASYSIMD_FLOAT64_C(   156.49), EASYSIMD_FLOAT64_C(   -65.30), EASYSIMD_FLOAT64_C(   731.13) },
      { EASYSIMD_FLOAT64_C(  -556.92), EASYSIMD_FLOAT64_C(   886.07), EASYSIMD_FLOAT64_C(   -65.30), EASYSIMD_FLOAT64_C(   731.13) } },
    { { EASYSIMD_FLOAT64_C(    33.49), EASYSIMD_FLOAT64_C(   874.95), EASYSIMD_FLOAT64_C(  -437.24), EASYSIMD_FLOAT64_C(  -510.52) },
      UINT8_C( 10),
      { EASYSIMD_FLOAT64_C(   774.18), EASYSIMD_FLOAT64_C(   -57.97), EASYSIMD_FLOAT64_C(   947.46), EASYSIMD_FLOAT64_C(   833.63) },
      { EASYSIMD_FLOAT64_C(  -822.49), EASYSIMD_FLOAT64_C(  -663.80), EASYSIMD_FLOAT64_C(   927.75), EASYSIMD_FLOAT64_C(   992.60) },
      { EASYSIMD_FLOAT64_C(    33.49), EASYSIMD_FLOAT64_C(   -57.97), EASYSIMD_FLOAT64_C(  -437.24), EASYSIMD_FLOAT64_C(  -663.80) },
      { EASYSIMD_FLOAT64_C(    33.49), EASYSIMD_FLOAT64_C(   833.63), EASYSIMD_FLOAT64_C(  -437.24), EASYSIMD_FLOAT64_C(  -663.80) },
      { EASYSIMD_FLOAT64_C(    33.49), EASYSIMD_FLOAT64_C(   -57.97), EASYSIMD_FLOAT64_C(  -437.24), EASYSIMD_FLOAT64_C(   992.60) },
      { EASYSIMD_FLOAT64_C(    33.49), EASYSIMD_FLOAT64_C(   833.63), EASYSIMD_FLOAT64_C(  -437.24), EASYSIMD_FLOAT64_C(   992.60) } },
    { { EASYSIMD_FLOAT64_C(    94.63), EASYSIMD_FLOAT64_C(  -952.39), EASYSIMD_FLOAT64_C(  -490.66), EASYSIMD_FLOAT64_C(   825.09) },
      UINT8_C(233),
      { EASYSIMD_FLOAT64_C(   -47.58), EASYSIMD_FLOAT64_C(  -934.62), EASYSIMD_FLOAT64_C(   -38.48), EASYSIMD_FLOAT64_C(  -316.46) },
      { EASYSIMD_FLOAT64_C(  -835.78), EASYSIMD_FLOAT64_C(   348.10), EASYSIMD_FLOAT64_C(   840.03), EASYSIMD_FLOAT64_C(  -637.46) },
      { EASYSIMD_FLOAT64_C(   -47.58), EASYSIMD_FLOAT64_C(  -952.39), EASYSIMD_FLOAT64_C(  -490.66), EASYSIMD_FLOAT64_C(   348.10) },
      { EASYSIMD_FLOAT64_C(   -38.48), EASYSIMD_FLOAT64_C(  -952.39), EASYSIMD_FLOAT64_C(  -490.66), EASYSIMD_FLOAT64_C(   348.10) },
      { EASYSIMD_FLOAT64_C(   -47.58), EASYSIMD_FLOAT64_C(  -952.39), EASYSIMD_FLOAT64_C(  -490.66), EASYSIMD_FLOAT64_C(  -637.46) },
      { EASYSIMD_FLOAT64_C(   -38.48), EASYSIMD_FLOAT64_C(  -952.39), EASYSIMD_FLOAT64_C(  -490.66), EASYSIMD_FLOAT64_C(  -637.46) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;

    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 0);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r0), 1);
    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 1);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r1), 1);
    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 2);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r2), 1);
    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 3);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r;

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 0);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 1);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 2);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_f64x2(src, k, a, b, 3);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_shuffle_f64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r0[4];
    const easysimd_float64 r1[4];
    const easysimd_float64 r2[4];
    const easysimd_float64 r3[4];
  } test_vec[4] = {
    { UINT8_C(146),
      { EASYSIMD_FLOAT64_C(   519.29), EASYSIMD_FLOAT64_C(    30.37), EASYSIMD_FLOAT64_C(   576.58), EASYSIMD_FLOAT64_C(   498.93) },
      { EASYSIMD_FLOAT64_C(  -198.77), EASYSIMD_FLOAT64_C(   114.03), EASYSIMD_FLOAT64_C(  -687.70), EASYSIMD_FLOAT64_C(   903.89) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    30.37), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   498.93), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    30.37), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   498.93), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 37),
      { EASYSIMD_FLOAT64_C(    68.15), EASYSIMD_FLOAT64_C(  -745.26), EASYSIMD_FLOAT64_C(   -93.84), EASYSIMD_FLOAT64_C(  -979.52) },
      { EASYSIMD_FLOAT64_C(  -682.93), EASYSIMD_FLOAT64_C(   146.97), EASYSIMD_FLOAT64_C(  -788.41), EASYSIMD_FLOAT64_C(   393.72) },
      { EASYSIMD_FLOAT64_C(    68.15), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -682.93), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(   -93.84), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -682.93), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(    68.15), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -788.41), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(   -93.84), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -788.41), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(187),
      { EASYSIMD_FLOAT64_C(   -30.90), EASYSIMD_FLOAT64_C(  -576.88), EASYSIMD_FLOAT64_C(   637.60), EASYSIMD_FLOAT64_C(  -671.29) },
      { EASYSIMD_FLOAT64_C(   124.20), EASYSIMD_FLOAT64_C(   203.69), EASYSIMD_FLOAT64_C(   -87.92), EASYSIMD_FLOAT64_C(   509.77) },
      { EASYSIMD_FLOAT64_C(   -30.90), EASYSIMD_FLOAT64_C(  -576.88), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   203.69) },
      { EASYSIMD_FLOAT64_C(   637.60), EASYSIMD_FLOAT64_C(  -671.29), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   203.69) },
      { EASYSIMD_FLOAT64_C(   -30.90), EASYSIMD_FLOAT64_C(  -576.88), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   509.77) },
      { EASYSIMD_FLOAT64_C(   637.60), EASYSIMD_FLOAT64_C(  -671.29), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   509.77) } },
    { UINT8_C(177),
      { EASYSIMD_FLOAT64_C(  -933.79), EASYSIMD_FLOAT64_C(  -943.60), EASYSIMD_FLOAT64_C(   943.63), EASYSIMD_FLOAT64_C(   102.23) },
      { EASYSIMD_FLOAT64_C(   575.68), EASYSIMD_FLOAT64_C(   -26.00), EASYSIMD_FLOAT64_C(  -321.19), EASYSIMD_FLOAT64_C(    74.62) },
      { EASYSIMD_FLOAT64_C(  -933.79), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(   943.63), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(  -933.79), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(   943.63), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;

    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 0);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r0), 1);
    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 1);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r1), 1);
    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 2);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r2), 1);
    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 3);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r;

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 0);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 1);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 2);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_f64x2(k, a, b, 3);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_shuffle_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r0[8];
    const int32_t r1[8];
    const int32_t r2[8];
    const int32_t r3[8];
  } test_vec[4] = {
    { {  INT32_C(   472292888), -INT32_C(  1050141756),  INT32_C(   896688052), -INT32_C(  1361646918), -INT32_C(  1915024474), -INT32_C(    10389770),  INT32_C(   590002938),  INT32_C(  1234553392) },
      {  INT32_C(   744864872), -INT32_C(  1947349545), -INT32_C(   423600340), -INT32_C(   275474871), -INT32_C(  1736675422), -INT32_C(   526918170), -INT32_C(   939277672), -INT32_C(   183396212) },
      {  INT32_C(   472292888), -INT32_C(  1050141756),  INT32_C(   896688052), -INT32_C(  1361646918),  INT32_C(   744864872), -INT32_C(  1947349545), -INT32_C(   423600340), -INT32_C(   275474871) },
      { -INT32_C(  1915024474), -INT32_C(    10389770),  INT32_C(   590002938),  INT32_C(  1234553392),  INT32_C(   744864872), -INT32_C(  1947349545), -INT32_C(   423600340), -INT32_C(   275474871) },
      {  INT32_C(   472292888), -INT32_C(  1050141756),  INT32_C(   896688052), -INT32_C(  1361646918), -INT32_C(  1736675422), -INT32_C(   526918170), -INT32_C(   939277672), -INT32_C(   183396212) },
      { -INT32_C(  1915024474), -INT32_C(    10389770),  INT32_C(   590002938),  INT32_C(  1234553392), -INT32_C(  1736675422), -INT32_C(   526918170), -INT32_C(   939277672), -INT32_C(   183396212) } },
    { {  INT32_C(   740390484),  INT32_C(  1891045187), -INT32_C(  1202292626), -INT32_C(  1331172594),  INT32_C(  1078469722), -INT32_C(  1725898751),  INT32_C(   778183586),  INT32_C(   270758844) },
      {  INT32_C(   758924778), -INT32_C(  1029835692),  INT32_C(  2054878316),  INT32_C(   959128287),  INT32_C(  1199207238), -INT32_C(   169764269),  INT32_C(  2049196990), -INT32_C(  1601550410) },
      {  INT32_C(   740390484),  INT32_C(  1891045187), -INT32_C(  1202292626), -INT32_C(  1331172594),  INT32_C(   758924778), -INT32_C(  1029835692),  INT32_C(  2054878316),  INT32_C(   959128287) },
      {  INT32_C(  1078469722), -INT32_C(  1725898751),  INT32_C(   778183586),  INT32_C(   270758844),  INT32_C(   758924778), -INT32_C(  1029835692),  INT32_C(  2054878316),  INT32_C(   959128287) },
      {  INT32_C(   740390484),  INT32_C(  1891045187), -INT32_C(  1202292626), -INT32_C(  1331172594),  INT32_C(  1199207238), -INT32_C(   169764269),  INT32_C(  2049196990), -INT32_C(  1601550410) },
      {  INT32_C(  1078469722), -INT32_C(  1725898751),  INT32_C(   778183586),  INT32_C(   270758844),  INT32_C(  1199207238), -INT32_C(   169764269),  INT32_C(  2049196990), -INT32_C(  1601550410) } },
    { { -INT32_C(   523319412),  INT32_C(   665021371),  INT32_C(  1050746207), -INT32_C(  2055680961), -INT32_C(  1815219648),  INT32_C(  1250537100), -INT32_C(  1480282639), -INT32_C(  2125967372) },
      { -INT32_C(   782166506), -INT32_C(   503839615),  INT32_C(  1629460770), -INT32_C(  1494771866),  INT32_C(   372880521),  INT32_C(  1398850146),  INT32_C(  1694115183), -INT32_C(  1964686732) },
      { -INT32_C(   523319412),  INT32_C(   665021371),  INT32_C(  1050746207), -INT32_C(  2055680961), -INT32_C(   782166506), -INT32_C(   503839615),  INT32_C(  1629460770), -INT32_C(  1494771866) },
      { -INT32_C(  1815219648),  INT32_C(  1250537100), -INT32_C(  1480282639), -INT32_C(  2125967372), -INT32_C(   782166506), -INT32_C(   503839615),  INT32_C(  1629460770), -INT32_C(  1494771866) },
      { -INT32_C(   523319412),  INT32_C(   665021371),  INT32_C(  1050746207), -INT32_C(  2055680961),  INT32_C(   372880521),  INT32_C(  1398850146),  INT32_C(  1694115183), -INT32_C(  1964686732) },
      { -INT32_C(  1815219648),  INT32_C(  1250537100), -INT32_C(  1480282639), -INT32_C(  2125967372),  INT32_C(   372880521),  INT32_C(  1398850146),  INT32_C(  1694115183), -INT32_C(  1964686732) } },
    { { -INT32_C(   631552424),  INT32_C(  1840993099),  INT32_C(  1389288172), -INT32_C(    67586702), -INT32_C(   888065431),  INT32_C(  1679717108),  INT32_C(   197663127), -INT32_C(  1265259173) },
      {  INT32_C(  1049555187),  INT32_C(   799754563), -INT32_C(  1786611165), -INT32_C(  1718519249), -INT32_C(  1587240276), -INT32_C(  1425702124), -INT32_C(   139014756),  INT32_C(  1839942522) },
      { -INT32_C(   631552424),  INT32_C(  1840993099),  INT32_C(  1389288172), -INT32_C(    67586702),  INT32_C(  1049555187),  INT32_C(   799754563), -INT32_C(  1786611165), -INT32_C(  1718519249) },
      { -INT32_C(   888065431),  INT32_C(  1679717108),  INT32_C(   197663127), -INT32_C(  1265259173),  INT32_C(  1049555187),  INT32_C(   799754563), -INT32_C(  1786611165), -INT32_C(  1718519249) },
      { -INT32_C(   631552424),  INT32_C(  1840993099),  INT32_C(  1389288172), -INT32_C(    67586702), -INT32_C(  1587240276), -INT32_C(  1425702124), -INT32_C(   139014756),  INT32_C(  1839942522) },
      { -INT32_C(   888065431),  INT32_C(  1679717108),  INT32_C(   197663127), -INT32_C(  1265259173), -INT32_C(  1587240276), -INT32_C(  1425702124), -INT32_C(   139014756),  INT32_C(  1839942522) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;

    r = easysimd_mm256_shuffle_i32x4(a, b, 0);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r0));
    r = easysimd_mm256_shuffle_i32x4(a, b, 1);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r1));
    r = easysimd_mm256_shuffle_i32x4(a, b, 2);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r2));
    r = easysimd_mm256_shuffle_i32x4(a, b, 3);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r3));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r;

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i32x4(a, b, 0);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i32x4(a, b, 1);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i32x4(a, b, 2);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i32x4(a, b, 3);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_shuffle_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r0[8];
    const int32_t r1[8];
    const int32_t r2[8];
    const int32_t r3[8];
  } test_vec[4] = {
    { { -INT32_C(   426491337), -INT32_C(  1735303558),  INT32_C(  1998478902), -INT32_C(   236589506), -INT32_C(  1344094934), -INT32_C(  1393533496), -INT32_C(  1532062390), -INT32_C(   952087920) },
      UINT8_C(140),
      {  INT32_C(   839298516), -INT32_C(  1654088129), -INT32_C(  1411587908), -INT32_C(  2083140155),  INT32_C(   223052976),  INT32_C(   173602677), -INT32_C(   275055451),  INT32_C(   276587068) },
      {  INT32_C(  1313047055), -INT32_C(   571692255),  INT32_C(  1367918731),  INT32_C(  1171545749),  INT32_C(  1465065442), -INT32_C(  1151161578), -INT32_C(   475267929),  INT32_C(  1861429086) },
      { -INT32_C(   426491337), -INT32_C(  1735303558), -INT32_C(  1411587908), -INT32_C(  2083140155), -INT32_C(  1344094934), -INT32_C(  1393533496), -INT32_C(  1532062390),  INT32_C(  1171545749) },
      { -INT32_C(   426491337), -INT32_C(  1735303558), -INT32_C(   275055451),  INT32_C(   276587068), -INT32_C(  1344094934), -INT32_C(  1393533496), -INT32_C(  1532062390),  INT32_C(  1171545749) },
      { -INT32_C(   426491337), -INT32_C(  1735303558), -INT32_C(  1411587908), -INT32_C(  2083140155), -INT32_C(  1344094934), -INT32_C(  1393533496), -INT32_C(  1532062390),  INT32_C(  1861429086) },
      { -INT32_C(   426491337), -INT32_C(  1735303558), -INT32_C(   275055451),  INT32_C(   276587068), -INT32_C(  1344094934), -INT32_C(  1393533496), -INT32_C(  1532062390),  INT32_C(  1861429086) } },
    { { -INT32_C(   893634903),  INT32_C(  1839769826),  INT32_C(   113127536),  INT32_C(  1900778126), -INT32_C(   943153487), -INT32_C(   259839415), -INT32_C(  2049757657), -INT32_C(    17578155) },
      UINT8_C(253),
      {  INT32_C(  1491061168), -INT32_C(  1580642959), -INT32_C(  1640968437), -INT32_C(  1185963750), -INT32_C(  1811802263), -INT32_C(   927206502), -INT32_C(  1927462714), -INT32_C(   460645580) },
      {  INT32_C(  1430022884), -INT32_C(  1007221321), -INT32_C(   278845484),  INT32_C(   833138888),  INT32_C(  1640344263),  INT32_C(  1680441501), -INT32_C(   168737087),  INT32_C(  1188658274) },
      {  INT32_C(  1491061168),  INT32_C(  1839769826), -INT32_C(  1640968437), -INT32_C(  1185963750),  INT32_C(  1430022884), -INT32_C(  1007221321), -INT32_C(   278845484),  INT32_C(   833138888) },
      { -INT32_C(  1811802263),  INT32_C(  1839769826), -INT32_C(  1927462714), -INT32_C(   460645580),  INT32_C(  1430022884), -INT32_C(  1007221321), -INT32_C(   278845484),  INT32_C(   833138888) },
      {  INT32_C(  1491061168),  INT32_C(  1839769826), -INT32_C(  1640968437), -INT32_C(  1185963750),  INT32_C(  1640344263),  INT32_C(  1680441501), -INT32_C(   168737087),  INT32_C(  1188658274) },
      { -INT32_C(  1811802263),  INT32_C(  1839769826), -INT32_C(  1927462714), -INT32_C(   460645580),  INT32_C(  1640344263),  INT32_C(  1680441501), -INT32_C(   168737087),  INT32_C(  1188658274) } },
    { { -INT32_C(  1633937945), -INT32_C(   278818021), -INT32_C(  2099330374),  INT32_C(   984843891), -INT32_C(   828606159), -INT32_C(  1171077639),  INT32_C(  1856971788), -INT32_C(  2018211680) },
      UINT8_C(158),
      { -INT32_C(   474405296),  INT32_C(  1251846279),  INT32_C(   230498183),  INT32_C(  1279195091),  INT32_C(  1497697683),  INT32_C(  1667563583),  INT32_C(   939840431), -INT32_C(   657028217) },
      {  INT32_C(   968593329), -INT32_C(  1098688201),  INT32_C(  1271677048), -INT32_C(   879293897),  INT32_C(  1462033687), -INT32_C(  1933932067), -INT32_C(   473645476), -INT32_C(    71591350) },
      { -INT32_C(  1633937945),  INT32_C(  1251846279),  INT32_C(   230498183),  INT32_C(  1279195091),  INT32_C(   968593329), -INT32_C(  1171077639),  INT32_C(  1856971788), -INT32_C(   879293897) },
      { -INT32_C(  1633937945),  INT32_C(  1667563583),  INT32_C(   939840431), -INT32_C(   657028217),  INT32_C(   968593329), -INT32_C(  1171077639),  INT32_C(  1856971788), -INT32_C(   879293897) },
      { -INT32_C(  1633937945),  INT32_C(  1251846279),  INT32_C(   230498183),  INT32_C(  1279195091),  INT32_C(  1462033687), -INT32_C(  1171077639),  INT32_C(  1856971788), -INT32_C(    71591350) },
      { -INT32_C(  1633937945),  INT32_C(  1667563583),  INT32_C(   939840431), -INT32_C(   657028217),  INT32_C(  1462033687), -INT32_C(  1171077639),  INT32_C(  1856971788), -INT32_C(    71591350) } },
    { {  INT32_C(  1630828329),  INT32_C(  1210038224),  INT32_C(   798288887),  INT32_C(   234499062), -INT32_C(   446423544),  INT32_C(    57810855),  INT32_C(   669398749), -INT32_C(    98327855) },
      UINT8_C( 25),
      {  INT32_C(   266951511),  INT32_C(  1711681915), -INT32_C(   245615163),  INT32_C(  1308191279), -INT32_C(   302719026), -INT32_C(  2016676015), -INT32_C(  2141654306),  INT32_C(  1838764821) },
      {  INT32_C(   696025774),  INT32_C(  2039513779), -INT32_C(   412422984),  INT32_C(   624255830), -INT32_C(  1827526334), -INT32_C(    14951135), -INT32_C(   444632112),  INT32_C(  1968314566) },
      {  INT32_C(   266951511),  INT32_C(  1210038224),  INT32_C(   798288887),  INT32_C(  1308191279),  INT32_C(   696025774),  INT32_C(    57810855),  INT32_C(   669398749), -INT32_C(    98327855) },
      { -INT32_C(   302719026),  INT32_C(  1210038224),  INT32_C(   798288887),  INT32_C(  1838764821),  INT32_C(   696025774),  INT32_C(    57810855),  INT32_C(   669398749), -INT32_C(    98327855) },
      {  INT32_C(   266951511),  INT32_C(  1210038224),  INT32_C(   798288887),  INT32_C(  1308191279), -INT32_C(  1827526334),  INT32_C(    57810855),  INT32_C(   669398749), -INT32_C(    98327855) },
      { -INT32_C(   302719026),  INT32_C(  1210038224),  INT32_C(   798288887),  INT32_C(  1838764821), -INT32_C(  1827526334),  INT32_C(    57810855),  INT32_C(   669398749), -INT32_C(    98327855) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;

    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 0);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r0));
    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 1);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r1));
    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 2);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r2));
    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 3);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r3));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r;

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 0);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 1);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 2);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i32x4(src, k, a, b, 3);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_shuffle_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r0[8];
    const int32_t r1[8];
    const int32_t r2[8];
    const int32_t r3[8];
  } test_vec[4] = {
    { UINT8_C(133),
      { -INT32_C(   509342325), -INT32_C(  1485532748),  INT32_C(  1451290319),  INT32_C(   264777338), -INT32_C(  1029505922), -INT32_C(   893145572), -INT32_C(  2043526745),  INT32_C(  2013988077) },
      { -INT32_C(   983912431),  INT32_C(   292343361),  INT32_C(   912780476), -INT32_C(  1723453669),  INT32_C(  1213982763),  INT32_C(  1024597654),  INT32_C(  1287865695),  INT32_C(  1522913097) },
      { -INT32_C(   509342325),  INT32_C(           0),  INT32_C(  1451290319),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1723453669) },
      { -INT32_C(  1029505922),  INT32_C(           0), -INT32_C(  2043526745),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1723453669) },
      { -INT32_C(   509342325),  INT32_C(           0),  INT32_C(  1451290319),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1522913097) },
      { -INT32_C(  1029505922),  INT32_C(           0), -INT32_C(  2043526745),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1522913097) } },
    { UINT8_C(127),
      { -INT32_C(   306176225),  INT32_C(  2007617931),  INT32_C(  1754455864),  INT32_C(   227748645), -INT32_C(  1515988090),  INT32_C(   855957998),  INT32_C(  1937527204),  INT32_C(   905106966) },
      { -INT32_C(  2145209355), -INT32_C(  1107833980), -INT32_C(   802846038),  INT32_C(  1021163701), -INT32_C(  2099150444),  INT32_C(   112584034),  INT32_C(  1283076406), -INT32_C(    58627065) },
      { -INT32_C(   306176225),  INT32_C(  2007617931),  INT32_C(  1754455864),  INT32_C(   227748645), -INT32_C(  2145209355), -INT32_C(  1107833980), -INT32_C(   802846038),  INT32_C(           0) },
      { -INT32_C(  1515988090),  INT32_C(   855957998),  INT32_C(  1937527204),  INT32_C(   905106966), -INT32_C(  2145209355), -INT32_C(  1107833980), -INT32_C(   802846038),  INT32_C(           0) },
      { -INT32_C(   306176225),  INT32_C(  2007617931),  INT32_C(  1754455864),  INT32_C(   227748645), -INT32_C(  2099150444),  INT32_C(   112584034),  INT32_C(  1283076406),  INT32_C(           0) },
      { -INT32_C(  1515988090),  INT32_C(   855957998),  INT32_C(  1937527204),  INT32_C(   905106966), -INT32_C(  2099150444),  INT32_C(   112584034),  INT32_C(  1283076406),  INT32_C(           0) } },
    { UINT8_C( 31),
      {  INT32_C(  1856273571), -INT32_C(    48668301),  INT32_C(  1051978118),  INT32_C(  1205006278), -INT32_C(  1247193904),  INT32_C(   988590089), -INT32_C(  1774110678),  INT32_C(  1572224442) },
      {  INT32_C(   768367289),  INT32_C(  1093330363), -INT32_C(  1803559474), -INT32_C(  1663282484), -INT32_C(  1336769114),  INT32_C(  1609186869),  INT32_C(   821439606),  INT32_C(   596552809) },
      {  INT32_C(  1856273571), -INT32_C(    48668301),  INT32_C(  1051978118),  INT32_C(  1205006278),  INT32_C(   768367289),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      { -INT32_C(  1247193904),  INT32_C(   988590089), -INT32_C(  1774110678),  INT32_C(  1572224442),  INT32_C(   768367289),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(  1856273571), -INT32_C(    48668301),  INT32_C(  1051978118),  INT32_C(  1205006278), -INT32_C(  1336769114),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      { -INT32_C(  1247193904),  INT32_C(   988590089), -INT32_C(  1774110678),  INT32_C(  1572224442), -INT32_C(  1336769114),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(  6),
      {  INT32_C(  1069633626),  INT32_C(  1477247610), -INT32_C(   752574079),  INT32_C(    58376573),  INT32_C(  1362635283),  INT32_C(  1086822420),  INT32_C(   984283278), -INT32_C(   532623994) },
      { -INT32_C(  1759575779), -INT32_C(  2064700413),  INT32_C(  1264063693), -INT32_C(   397487403),  INT32_C(   272205564), -INT32_C(  1403977698),  INT32_C(  2129067000), -INT32_C(   446814520) },
      {  INT32_C(           0),  INT32_C(  1477247610), -INT32_C(   752574079),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(  1086822420),  INT32_C(   984283278),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(  1477247610), -INT32_C(   752574079),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(  1086822420),  INT32_C(   984283278),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;

    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 0);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r0));
    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 1);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r1));
    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 2);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r2));
    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 3);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r3));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r;

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 0);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 1);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 2);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i32x4(k, a, b, 3);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_shuffle_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r0[4];
    const int64_t r1[4];
    const int64_t r2[4];
    const int64_t r3[4];
  } test_vec[4] = {
    { {  INT64_C( 3551315409324880873),  INT64_C( 8008329497760772479),  INT64_C( 2445691485855378668),  INT64_C( 3145002663167902100) },
      {  INT64_C( 2394514919912297342), -INT64_C( 4258764945951323337), -INT64_C( 1453948719952461761),  INT64_C( 7113696077798676811) },
      {  INT64_C( 3551315409324880873),  INT64_C( 8008329497760772479),  INT64_C( 2394514919912297342), -INT64_C( 4258764945951323337) },
      {  INT64_C( 2445691485855378668),  INT64_C( 3145002663167902100),  INT64_C( 2394514919912297342), -INT64_C( 4258764945951323337) },
      {  INT64_C( 3551315409324880873),  INT64_C( 8008329497760772479), -INT64_C( 1453948719952461761),  INT64_C( 7113696077798676811) },
      {  INT64_C( 2445691485855378668),  INT64_C( 3145002663167902100), -INT64_C( 1453948719952461761),  INT64_C( 7113696077798676811) } },
    { { -INT64_C( 5511377366319073601),  INT64_C( 4958197129583079986),  INT64_C( 4454562452028680334), -INT64_C( 4275743416332445598) },
      { -INT64_C( 1213253092931987975), -INT64_C( 1214342002771340581),  INT64_C( 4999461655375321906), -INT64_C( 6374786247804834881) },
      { -INT64_C( 5511377366319073601),  INT64_C( 4958197129583079986), -INT64_C( 1213253092931987975), -INT64_C( 1214342002771340581) },
      {  INT64_C( 4454562452028680334), -INT64_C( 4275743416332445598), -INT64_C( 1213253092931987975), -INT64_C( 1214342002771340581) },
      { -INT64_C( 5511377366319073601),  INT64_C( 4958197129583079986),  INT64_C( 4999461655375321906), -INT64_C( 6374786247804834881) },
      {  INT64_C( 4454562452028680334), -INT64_C( 4275743416332445598),  INT64_C( 4999461655375321906), -INT64_C( 6374786247804834881) } },
    { {  INT64_C( 3600422662894824778), -INT64_C( 1231674317120343656),  INT64_C( 2628541310555045457), -INT64_C( 8722223178297375073) },
      { -INT64_C( 1250447814667988706),  INT64_C( 4476526271792658038),  INT64_C( 4978710313897767566), -INT64_C( 8163188358641417996) },
      {  INT64_C( 3600422662894824778), -INT64_C( 1231674317120343656), -INT64_C( 1250447814667988706),  INT64_C( 4476526271792658038) },
      {  INT64_C( 2628541310555045457), -INT64_C( 8722223178297375073), -INT64_C( 1250447814667988706),  INT64_C( 4476526271792658038) },
      {  INT64_C( 3600422662894824778), -INT64_C( 1231674317120343656),  INT64_C( 4978710313897767566), -INT64_C( 8163188358641417996) },
      {  INT64_C( 2628541310555045457), -INT64_C( 8722223178297375073),  INT64_C( 4978710313897767566), -INT64_C( 8163188358641417996) } },
    { {  INT64_C( 4529961281649787800),  INT64_C( 2560735734515352926), -INT64_C( 2329436418187264575),  INT64_C( 6858073793719385901) },
      {  INT64_C( 3751266315769097993), -INT64_C( 3555300477136568591),  INT64_C( 1932293747428933382), -INT64_C( 5888205111561756639) },
      {  INT64_C( 4529961281649787800),  INT64_C( 2560735734515352926),  INT64_C( 3751266315769097993), -INT64_C( 3555300477136568591) },
      { -INT64_C( 2329436418187264575),  INT64_C( 6858073793719385901),  INT64_C( 3751266315769097993), -INT64_C( 3555300477136568591) },
      {  INT64_C( 4529961281649787800),  INT64_C( 2560735734515352926),  INT64_C( 1932293747428933382), -INT64_C( 5888205111561756639) },
      { -INT64_C( 2329436418187264575),  INT64_C( 6858073793719385901),  INT64_C( 1932293747428933382), -INT64_C( 5888205111561756639) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;

    r = easysimd_mm256_shuffle_i64x2(a, b, 0);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r0));
    r = easysimd_mm256_shuffle_i64x2(a, b, 1);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r1));
    r = easysimd_mm256_shuffle_i64x2(a, b, 2);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r2));
    r = easysimd_mm256_shuffle_i64x2(a, b, 3);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r3));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r;

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i64x2(a, b, 0);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i64x2(a, b, 1);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i64x2(a, b, 2);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_shuffle_i64x2(a, b, 3);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_shuffle_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r0[4];
    const int64_t r1[4];
    const int64_t r2[4];
    const int64_t r3[4];
  } test_vec[4] = {
    { { -INT64_C( 4750049666754593255),  INT64_C( 7348475432003887810),  INT64_C(  835478380301128017),  INT64_C( 1476489067024726412) },
      UINT8_C(200),
      { -INT64_C( 4733137129855072481),  INT64_C( 1307340219673347531), -INT64_C( 9170376919911550983),  INT64_C( 3123966575694810632) },
      {  INT64_C( 8138014032272040023), -INT64_C( 5034911404046277106), -INT64_C( 6134281326629856362),  INT64_C( 4642251244478159157) },
      { -INT64_C( 4750049666754593255),  INT64_C( 7348475432003887810),  INT64_C(  835478380301128017), -INT64_C( 5034911404046277106) },
      { -INT64_C( 4750049666754593255),  INT64_C( 7348475432003887810),  INT64_C(  835478380301128017), -INT64_C( 5034911404046277106) },
      { -INT64_C( 4750049666754593255),  INT64_C( 7348475432003887810),  INT64_C(  835478380301128017),  INT64_C( 4642251244478159157) },
      { -INT64_C( 4750049666754593255),  INT64_C( 7348475432003887810),  INT64_C(  835478380301128017),  INT64_C( 4642251244478159157) } },
    { {  INT64_C( 4308646514968896950),  INT64_C(  722100362974727817), -INT64_C( 3741850724285747002), -INT64_C( 7293444474572747105) },
      UINT8_C(159),
      { -INT64_C( 3519834956080744901),  INT64_C( 7530949282372022866),  INT64_C( 5448453940943252406),  INT64_C( 2950924202014614745) },
      {  INT64_C( 1968334486882402495), -INT64_C( 7403816753243319200), -INT64_C( 1977588166704087226),  INT64_C( 5074197265111366998) },
      { -INT64_C( 3519834956080744901),  INT64_C( 7530949282372022866),  INT64_C( 1968334486882402495), -INT64_C( 7403816753243319200) },
      {  INT64_C( 5448453940943252406),  INT64_C( 2950924202014614745),  INT64_C( 1968334486882402495), -INT64_C( 7403816753243319200) },
      { -INT64_C( 3519834956080744901),  INT64_C( 7530949282372022866), -INT64_C( 1977588166704087226),  INT64_C( 5074197265111366998) },
      {  INT64_C( 5448453940943252406),  INT64_C( 2950924202014614745), -INT64_C( 1977588166704087226),  INT64_C( 5074197265111366998) } },
    { {  INT64_C( 9209887762445447660),  INT64_C( 5416026072730282412),  INT64_C( 5171291196819947478),  INT64_C( 1488710330141637850) },
      UINT8_C( 40),
      {  INT64_C( 4770656267994122607), -INT64_C( 6085890401103264995),  INT64_C( 3105482834620213792), -INT64_C( 7958591573362247313) },
      { -INT64_C( 5256742437348912080),  INT64_C(  479903728446165776), -INT64_C( 7513592798553881504), -INT64_C( 6447032102297989419) },
      {  INT64_C( 9209887762445447660),  INT64_C( 5416026072730282412),  INT64_C( 5171291196819947478),  INT64_C(  479903728446165776) },
      {  INT64_C( 9209887762445447660),  INT64_C( 5416026072730282412),  INT64_C( 5171291196819947478),  INT64_C(  479903728446165776) },
      {  INT64_C( 9209887762445447660),  INT64_C( 5416026072730282412),  INT64_C( 5171291196819947478), -INT64_C( 6447032102297989419) },
      {  INT64_C( 9209887762445447660),  INT64_C( 5416026072730282412),  INT64_C( 5171291196819947478), -INT64_C( 6447032102297989419) } },
    { { -INT64_C(  238271948034891681),  INT64_C( 8267378642703162576),  INT64_C( 3433356286396334053), -INT64_C( 8294032728888270135) },
      UINT8_C( 77),
      {  INT64_C( 6026157385464841557),  INT64_C( 1276444020211799855),  INT64_C(  891039179036476632), -INT64_C( 2973284494263870271) },
      {  INT64_C( 6689051806349653578), -INT64_C( 1673963500626691680), -INT64_C( 8278503648347344729),  INT64_C( 7335158844038600052) },
      {  INT64_C( 6026157385464841557),  INT64_C( 8267378642703162576),  INT64_C( 6689051806349653578), -INT64_C( 1673963500626691680) },
      {  INT64_C(  891039179036476632),  INT64_C( 8267378642703162576),  INT64_C( 6689051806349653578), -INT64_C( 1673963500626691680) },
      {  INT64_C( 6026157385464841557),  INT64_C( 8267378642703162576), -INT64_C( 8278503648347344729),  INT64_C( 7335158844038600052) },
      {  INT64_C(  891039179036476632),  INT64_C( 8267378642703162576), -INT64_C( 8278503648347344729),  INT64_C( 7335158844038600052) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;

    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 0);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r0));
    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 1);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r1));
    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 2);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r2));
    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 3);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r3));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r;

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 0);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 1);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 2);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_mask_shuffle_i64x2(src, k, a, b, 3);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_shuffle_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r0[4];
    const int64_t r1[4];
    const int64_t r2[4];
    const int64_t r3[4];
  } test_vec[4] = {
    { UINT8_C(130),
      {  INT64_C( 5897795453911463084),  INT64_C( 1234160324116783402), -INT64_C( 3441831997656375695),  INT64_C(  364741372620777884) },
      { -INT64_C( 5453987194644374202),  INT64_C( 4216626858216074028),  INT64_C( 7398880696424309461), -INT64_C( 4998711411531865789) },
      {  INT64_C(                   0),  INT64_C( 1234160324116783402),  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C(                   0),  INT64_C(  364741372620777884),  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C(                   0),  INT64_C( 1234160324116783402),  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C(                   0),  INT64_C(  364741372620777884),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 95),
      {  INT64_C( 3605868208288544850), -INT64_C( 6755059690134340363),  INT64_C( 7958093419730771730), -INT64_C(   60343750250853177) },
      { -INT64_C( 6517016866100276651), -INT64_C( 3456983586772514461),  INT64_C( 1057621640285614962), -INT64_C( 9068842080715435911) },
      {  INT64_C( 3605868208288544850), -INT64_C( 6755059690134340363), -INT64_C( 6517016866100276651), -INT64_C( 3456983586772514461) },
      {  INT64_C( 7958093419730771730), -INT64_C(   60343750250853177), -INT64_C( 6517016866100276651), -INT64_C( 3456983586772514461) },
      {  INT64_C( 3605868208288544850), -INT64_C( 6755059690134340363),  INT64_C( 1057621640285614962), -INT64_C( 9068842080715435911) },
      {  INT64_C( 7958093419730771730), -INT64_C(   60343750250853177),  INT64_C( 1057621640285614962), -INT64_C( 9068842080715435911) } },
    { UINT8_C( 19),
      { -INT64_C(  770844044711174654),  INT64_C( 3116354292617032058), -INT64_C( 4871065016503992704), -INT64_C( 5446140910634431401) },
      {  INT64_C( 1345146475543146057),  INT64_C( 7025087517084835228), -INT64_C( 3997589598203088739),  INT64_C( 2497399761580092993) },
      { -INT64_C(  770844044711174654),  INT64_C( 3116354292617032058),  INT64_C(                   0),  INT64_C(                   0) },
      { -INT64_C( 4871065016503992704), -INT64_C( 5446140910634431401),  INT64_C(                   0),  INT64_C(                   0) },
      { -INT64_C(  770844044711174654),  INT64_C( 3116354292617032058),  INT64_C(                   0),  INT64_C(                   0) },
      { -INT64_C( 4871065016503992704), -INT64_C( 5446140910634431401),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(185),
      { -INT64_C( 2392146420534487226), -INT64_C( 8070696138312835006), -INT64_C( 4871267184011065234),  INT64_C( 5056618793708968168) },
      {  INT64_C( 2589645514894441872),  INT64_C( 4437034983047343973),  INT64_C( 4852283745272697676),  INT64_C( 3847731754999147550) },
      { -INT64_C( 2392146420534487226),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4437034983047343973) },
      { -INT64_C( 4871267184011065234),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4437034983047343973) },
      { -INT64_C( 2392146420534487226),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 3847731754999147550) },
      { -INT64_C( 4871267184011065234),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 3847731754999147550) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;

    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 0);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r0));
    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 1);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r1));
    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 2);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r2));
    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 3);
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r3));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r;

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 0);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 1);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 2);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm256_maskz_shuffle_i64x2(k, a, b, 3);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shuffle_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r1[16];
    const easysimd_float32 r2[16];
    const easysimd_float32 r3[16];
    const easysimd_float32 r4[16];
  } test_vec[4] = {
    { { EASYSIMD_FLOAT32_C(   612.00), EASYSIMD_FLOAT32_C(    41.95), EASYSIMD_FLOAT32_C(   976.38), EASYSIMD_FLOAT32_C(   600.17),
        EASYSIMD_FLOAT32_C(   703.58), EASYSIMD_FLOAT32_C(   547.81), EASYSIMD_FLOAT32_C(   -87.34), EASYSIMD_FLOAT32_C(  -795.34),
        EASYSIMD_FLOAT32_C(   -60.06), EASYSIMD_FLOAT32_C(  -487.81), EASYSIMD_FLOAT32_C(   639.28), EASYSIMD_FLOAT32_C(  -834.87),
        EASYSIMD_FLOAT32_C(   869.20), EASYSIMD_FLOAT32_C(  -130.23), EASYSIMD_FLOAT32_C(  -140.41), EASYSIMD_FLOAT32_C(  -683.29) },
      { EASYSIMD_FLOAT32_C(  -862.91), EASYSIMD_FLOAT32_C(  -708.52), EASYSIMD_FLOAT32_C(  -653.90), EASYSIMD_FLOAT32_C(  -928.55),
        EASYSIMD_FLOAT32_C(   531.84), EASYSIMD_FLOAT32_C(   569.89), EASYSIMD_FLOAT32_C(  -311.59), EASYSIMD_FLOAT32_C(  -160.89),
        EASYSIMD_FLOAT32_C(   412.77), EASYSIMD_FLOAT32_C(   762.26), EASYSIMD_FLOAT32_C(  -945.34), EASYSIMD_FLOAT32_C(   174.68),
        EASYSIMD_FLOAT32_C(   709.50), EASYSIMD_FLOAT32_C(  -817.46), EASYSIMD_FLOAT32_C(   656.33), EASYSIMD_FLOAT32_C(   321.50) },
      { EASYSIMD_FLOAT32_C(   -60.06), EASYSIMD_FLOAT32_C(  -487.81), EASYSIMD_FLOAT32_C(   639.28), EASYSIMD_FLOAT32_C(  -834.87),
        EASYSIMD_FLOAT32_C(   869.20), EASYSIMD_FLOAT32_C(  -130.23), EASYSIMD_FLOAT32_C(  -140.41), EASYSIMD_FLOAT32_C(  -683.29),
        EASYSIMD_FLOAT32_C(  -862.91), EASYSIMD_FLOAT32_C(  -708.52), EASYSIMD_FLOAT32_C(  -653.90), EASYSIMD_FLOAT32_C(  -928.55),
        EASYSIMD_FLOAT32_C(   531.84), EASYSIMD_FLOAT32_C(   569.89), EASYSIMD_FLOAT32_C(  -311.59), EASYSIMD_FLOAT32_C(  -160.89) },
      { EASYSIMD_FLOAT32_C(   703.58), EASYSIMD_FLOAT32_C(   547.81), EASYSIMD_FLOAT32_C(   -87.34), EASYSIMD_FLOAT32_C(  -795.34),
        EASYSIMD_FLOAT32_C(   -60.06), EASYSIMD_FLOAT32_C(  -487.81), EASYSIMD_FLOAT32_C(   639.28), EASYSIMD_FLOAT32_C(  -834.87),
        EASYSIMD_FLOAT32_C(   709.50), EASYSIMD_FLOAT32_C(  -817.46), EASYSIMD_FLOAT32_C(   656.33), EASYSIMD_FLOAT32_C(   321.50),
        EASYSIMD_FLOAT32_C(  -862.91), EASYSIMD_FLOAT32_C(  -708.52), EASYSIMD_FLOAT32_C(  -653.90), EASYSIMD_FLOAT32_C(  -928.55) },
      { EASYSIMD_FLOAT32_C(   -60.06), EASYSIMD_FLOAT32_C(  -487.81), EASYSIMD_FLOAT32_C(   639.28), EASYSIMD_FLOAT32_C(  -834.87),
        EASYSIMD_FLOAT32_C(   -60.06), EASYSIMD_FLOAT32_C(  -487.81), EASYSIMD_FLOAT32_C(   639.28), EASYSIMD_FLOAT32_C(  -834.87),
        EASYSIMD_FLOAT32_C(   531.84), EASYSIMD_FLOAT32_C(   569.89), EASYSIMD_FLOAT32_C(  -311.59), EASYSIMD_FLOAT32_C(  -160.89),
        EASYSIMD_FLOAT32_C(   709.50), EASYSIMD_FLOAT32_C(  -817.46), EASYSIMD_FLOAT32_C(   656.33), EASYSIMD_FLOAT32_C(   321.50) },
      { EASYSIMD_FLOAT32_C(   869.20), EASYSIMD_FLOAT32_C(  -130.23), EASYSIMD_FLOAT32_C(  -140.41), EASYSIMD_FLOAT32_C(  -683.29),
        EASYSIMD_FLOAT32_C(   703.58), EASYSIMD_FLOAT32_C(   547.81), EASYSIMD_FLOAT32_C(   -87.34), EASYSIMD_FLOAT32_C(  -795.34),
        EASYSIMD_FLOAT32_C(   412.77), EASYSIMD_FLOAT32_C(   762.26), EASYSIMD_FLOAT32_C(  -945.34), EASYSIMD_FLOAT32_C(   174.68),
        EASYSIMD_FLOAT32_C(   412.77), EASYSIMD_FLOAT32_C(   762.26), EASYSIMD_FLOAT32_C(  -945.34), EASYSIMD_FLOAT32_C(   174.68) } },
    { { EASYSIMD_FLOAT32_C(   224.49), EASYSIMD_FLOAT32_C(   632.71), EASYSIMD_FLOAT32_C(   -78.33), EASYSIMD_FLOAT32_C(   -71.93),
        EASYSIMD_FLOAT32_C(   180.52), EASYSIMD_FLOAT32_C(   834.33), EASYSIMD_FLOAT32_C(   132.73), EASYSIMD_FLOAT32_C(  -879.54),
        EASYSIMD_FLOAT32_C(  -653.48), EASYSIMD_FLOAT32_C(  -227.99), EASYSIMD_FLOAT32_C(  -714.41), EASYSIMD_FLOAT32_C(  -784.28),
        EASYSIMD_FLOAT32_C(   641.77), EASYSIMD_FLOAT32_C(   145.18), EASYSIMD_FLOAT32_C(  -467.57), EASYSIMD_FLOAT32_C(   778.87) },
      { EASYSIMD_FLOAT32_C(   436.67), EASYSIMD_FLOAT32_C(  -121.47), EASYSIMD_FLOAT32_C(   850.32), EASYSIMD_FLOAT32_C(   -31.50),
        EASYSIMD_FLOAT32_C(  -551.58), EASYSIMD_FLOAT32_C(  -461.27), EASYSIMD_FLOAT32_C(   807.61), EASYSIMD_FLOAT32_C(   861.18),
        EASYSIMD_FLOAT32_C(  -699.01), EASYSIMD_FLOAT32_C(   862.27), EASYSIMD_FLOAT32_C(    35.86), EASYSIMD_FLOAT32_C(  -989.51),
        EASYSIMD_FLOAT32_C(  -955.19), EASYSIMD_FLOAT32_C(  -307.81), EASYSIMD_FLOAT32_C(   331.99), EASYSIMD_FLOAT32_C(   269.30) },
      { EASYSIMD_FLOAT32_C(  -653.48), EASYSIMD_FLOAT32_C(  -227.99), EASYSIMD_FLOAT32_C(  -714.41), EASYSIMD_FLOAT32_C(  -784.28),
        EASYSIMD_FLOAT32_C(   641.77), EASYSIMD_FLOAT32_C(   145.18), EASYSIMD_FLOAT32_C(  -467.57), EASYSIMD_FLOAT32_C(   778.87),
        EASYSIMD_FLOAT32_C(   436.67), EASYSIMD_FLOAT32_C(  -121.47), EASYSIMD_FLOAT32_C(   850.32), EASYSIMD_FLOAT32_C(   -31.50),
        EASYSIMD_FLOAT32_C(  -551.58), EASYSIMD_FLOAT32_C(  -461.27), EASYSIMD_FLOAT32_C(   807.61), EASYSIMD_FLOAT32_C(   861.18) },
      { EASYSIMD_FLOAT32_C(   180.52), EASYSIMD_FLOAT32_C(   834.33), EASYSIMD_FLOAT32_C(   132.73), EASYSIMD_FLOAT32_C(  -879.54),
        EASYSIMD_FLOAT32_C(  -653.48), EASYSIMD_FLOAT32_C(  -227.99), EASYSIMD_FLOAT32_C(  -714.41), EASYSIMD_FLOAT32_C(  -784.28),
        EASYSIMD_FLOAT32_C(  -955.19), EASYSIMD_FLOAT32_C(  -307.81), EASYSIMD_FLOAT32_C(   331.99), EASYSIMD_FLOAT32_C(   269.30),
        EASYSIMD_FLOAT32_C(   436.67), EASYSIMD_FLOAT32_C(  -121.47), EASYSIMD_FLOAT32_C(   850.32), EASYSIMD_FLOAT32_C(   -31.50) },
      { EASYSIMD_FLOAT32_C(  -653.48), EASYSIMD_FLOAT32_C(  -227.99), EASYSIMD_FLOAT32_C(  -714.41), EASYSIMD_FLOAT32_C(  -784.28),
        EASYSIMD_FLOAT32_C(  -653.48), EASYSIMD_FLOAT32_C(  -227.99), EASYSIMD_FLOAT32_C(  -714.41), EASYSIMD_FLOAT32_C(  -784.28),
        EASYSIMD_FLOAT32_C(  -551.58), EASYSIMD_FLOAT32_C(  -461.27), EASYSIMD_FLOAT32_C(   807.61), EASYSIMD_FLOAT32_C(   861.18),
        EASYSIMD_FLOAT32_C(  -955.19), EASYSIMD_FLOAT32_C(  -307.81), EASYSIMD_FLOAT32_C(   331.99), EASYSIMD_FLOAT32_C(   269.30) },
      { EASYSIMD_FLOAT32_C(   641.77), EASYSIMD_FLOAT32_C(   145.18), EASYSIMD_FLOAT32_C(  -467.57), EASYSIMD_FLOAT32_C(   778.87),
        EASYSIMD_FLOAT32_C(   180.52), EASYSIMD_FLOAT32_C(   834.33), EASYSIMD_FLOAT32_C(   132.73), EASYSIMD_FLOAT32_C(  -879.54),
        EASYSIMD_FLOAT32_C(  -699.01), EASYSIMD_FLOAT32_C(   862.27), EASYSIMD_FLOAT32_C(    35.86), EASYSIMD_FLOAT32_C(  -989.51),
        EASYSIMD_FLOAT32_C(  -699.01), EASYSIMD_FLOAT32_C(   862.27), EASYSIMD_FLOAT32_C(    35.86), EASYSIMD_FLOAT32_C(  -989.51) } },
    { { EASYSIMD_FLOAT32_C(  -675.10), EASYSIMD_FLOAT32_C(  -746.34), EASYSIMD_FLOAT32_C(  -802.63), EASYSIMD_FLOAT32_C(   505.42),
        EASYSIMD_FLOAT32_C(  -912.01), EASYSIMD_FLOAT32_C(   330.10), EASYSIMD_FLOAT32_C(   625.89), EASYSIMD_FLOAT32_C(  -565.49),
        EASYSIMD_FLOAT32_C(  -897.90), EASYSIMD_FLOAT32_C(   911.48), EASYSIMD_FLOAT32_C(  -349.78), EASYSIMD_FLOAT32_C(   743.88),
        EASYSIMD_FLOAT32_C(    56.66), EASYSIMD_FLOAT32_C(   182.65), EASYSIMD_FLOAT32_C(   522.74), EASYSIMD_FLOAT32_C(  -506.67) },
      { EASYSIMD_FLOAT32_C(  -938.82), EASYSIMD_FLOAT32_C(   373.07), EASYSIMD_FLOAT32_C(   461.83), EASYSIMD_FLOAT32_C(  -490.40),
        EASYSIMD_FLOAT32_C(   911.80), EASYSIMD_FLOAT32_C(   269.44), EASYSIMD_FLOAT32_C(  -629.22), EASYSIMD_FLOAT32_C(  -787.21),
        EASYSIMD_FLOAT32_C(   131.71), EASYSIMD_FLOAT32_C(   406.64), EASYSIMD_FLOAT32_C(  -776.72), EASYSIMD_FLOAT32_C(   176.53),
        EASYSIMD_FLOAT32_C(  -901.17), EASYSIMD_FLOAT32_C(   555.27), EASYSIMD_FLOAT32_C(  -554.17), EASYSIMD_FLOAT32_C(  -576.26) },
      { EASYSIMD_FLOAT32_C(  -897.90), EASYSIMD_FLOAT32_C(   911.48), EASYSIMD_FLOAT32_C(  -349.78), EASYSIMD_FLOAT32_C(   743.88),
        EASYSIMD_FLOAT32_C(    56.66), EASYSIMD_FLOAT32_C(   182.65), EASYSIMD_FLOAT32_C(   522.74), EASYSIMD_FLOAT32_C(  -506.67),
        EASYSIMD_FLOAT32_C(  -938.82), EASYSIMD_FLOAT32_C(   373.07), EASYSIMD_FLOAT32_C(   461.83), EASYSIMD_FLOAT32_C(  -490.40),
        EASYSIMD_FLOAT32_C(   911.80), EASYSIMD_FLOAT32_C(   269.44), EASYSIMD_FLOAT32_C(  -629.22), EASYSIMD_FLOAT32_C(  -787.21) },
      { EASYSIMD_FLOAT32_C(  -912.01), EASYSIMD_FLOAT32_C(   330.10), EASYSIMD_FLOAT32_C(   625.89), EASYSIMD_FLOAT32_C(  -565.49),
        EASYSIMD_FLOAT32_C(  -897.90), EASYSIMD_FLOAT32_C(   911.48), EASYSIMD_FLOAT32_C(  -349.78), EASYSIMD_FLOAT32_C(   743.88),
        EASYSIMD_FLOAT32_C(  -901.17), EASYSIMD_FLOAT32_C(   555.27), EASYSIMD_FLOAT32_C(  -554.17), EASYSIMD_FLOAT32_C(  -576.26),
        EASYSIMD_FLOAT32_C(  -938.82), EASYSIMD_FLOAT32_C(   373.07), EASYSIMD_FLOAT32_C(   461.83), EASYSIMD_FLOAT32_C(  -490.40) },
      { EASYSIMD_FLOAT32_C(  -897.90), EASYSIMD_FLOAT32_C(   911.48), EASYSIMD_FLOAT32_C(  -349.78), EASYSIMD_FLOAT32_C(   743.88),
        EASYSIMD_FLOAT32_C(  -897.90), EASYSIMD_FLOAT32_C(   911.48), EASYSIMD_FLOAT32_C(  -349.78), EASYSIMD_FLOAT32_C(   743.88),
        EASYSIMD_FLOAT32_C(   911.80), EASYSIMD_FLOAT32_C(   269.44), EASYSIMD_FLOAT32_C(  -629.22), EASYSIMD_FLOAT32_C(  -787.21),
        EASYSIMD_FLOAT32_C(  -901.17), EASYSIMD_FLOAT32_C(   555.27), EASYSIMD_FLOAT32_C(  -554.17), EASYSIMD_FLOAT32_C(  -576.26) },
      { EASYSIMD_FLOAT32_C(    56.66), EASYSIMD_FLOAT32_C(   182.65), EASYSIMD_FLOAT32_C(   522.74), EASYSIMD_FLOAT32_C(  -506.67),
        EASYSIMD_FLOAT32_C(  -912.01), EASYSIMD_FLOAT32_C(   330.10), EASYSIMD_FLOAT32_C(   625.89), EASYSIMD_FLOAT32_C(  -565.49),
        EASYSIMD_FLOAT32_C(   131.71), EASYSIMD_FLOAT32_C(   406.64), EASYSIMD_FLOAT32_C(  -776.72), EASYSIMD_FLOAT32_C(   176.53),
        EASYSIMD_FLOAT32_C(   131.71), EASYSIMD_FLOAT32_C(   406.64), EASYSIMD_FLOAT32_C(  -776.72), EASYSIMD_FLOAT32_C(   176.53) } },
    { { EASYSIMD_FLOAT32_C(   808.93), EASYSIMD_FLOAT32_C(  -356.80), EASYSIMD_FLOAT32_C(   929.16), EASYSIMD_FLOAT32_C(   896.92),
        EASYSIMD_FLOAT32_C(   973.30), EASYSIMD_FLOAT32_C(   555.05), EASYSIMD_FLOAT32_C(  -668.58), EASYSIMD_FLOAT32_C(  -924.60),
        EASYSIMD_FLOAT32_C(   466.53), EASYSIMD_FLOAT32_C(   -18.35), EASYSIMD_FLOAT32_C(   819.28), EASYSIMD_FLOAT32_C(  -476.81),
        EASYSIMD_FLOAT32_C(  -835.70), EASYSIMD_FLOAT32_C(   342.03), EASYSIMD_FLOAT32_C(    16.52), EASYSIMD_FLOAT32_C(  -774.52) },
      { EASYSIMD_FLOAT32_C(  -284.91), EASYSIMD_FLOAT32_C(  -521.66), EASYSIMD_FLOAT32_C(  -264.92), EASYSIMD_FLOAT32_C(  -373.11),
        EASYSIMD_FLOAT32_C(   747.78), EASYSIMD_FLOAT32_C(   105.87), EASYSIMD_FLOAT32_C(  -160.32), EASYSIMD_FLOAT32_C(  -120.51),
        EASYSIMD_FLOAT32_C(  -487.49), EASYSIMD_FLOAT32_C(    62.96), EASYSIMD_FLOAT32_C(  -943.98), EASYSIMD_FLOAT32_C(  -388.66),
        EASYSIMD_FLOAT32_C(  -381.77), EASYSIMD_FLOAT32_C(  -498.15), EASYSIMD_FLOAT32_C(    35.08), EASYSIMD_FLOAT32_C(  -572.84) },
      { EASYSIMD_FLOAT32_C(   466.53), EASYSIMD_FLOAT32_C(   -18.35), EASYSIMD_FLOAT32_C(   819.28), EASYSIMD_FLOAT32_C(  -476.81),
        EASYSIMD_FLOAT32_C(  -835.70), EASYSIMD_FLOAT32_C(   342.03), EASYSIMD_FLOAT32_C(    16.52), EASYSIMD_FLOAT32_C(  -774.52),
        EASYSIMD_FLOAT32_C(  -284.91), EASYSIMD_FLOAT32_C(  -521.66), EASYSIMD_FLOAT32_C(  -264.92), EASYSIMD_FLOAT32_C(  -373.11),
        EASYSIMD_FLOAT32_C(   747.78), EASYSIMD_FLOAT32_C(   105.87), EASYSIMD_FLOAT32_C(  -160.32), EASYSIMD_FLOAT32_C(  -120.51) },
      { EASYSIMD_FLOAT32_C(   973.30), EASYSIMD_FLOAT32_C(   555.05), EASYSIMD_FLOAT32_C(  -668.58), EASYSIMD_FLOAT32_C(  -924.60),
        EASYSIMD_FLOAT32_C(   466.53), EASYSIMD_FLOAT32_C(   -18.35), EASYSIMD_FLOAT32_C(   819.28), EASYSIMD_FLOAT32_C(  -476.81),
        EASYSIMD_FLOAT32_C(  -381.77), EASYSIMD_FLOAT32_C(  -498.15), EASYSIMD_FLOAT32_C(    35.08), EASYSIMD_FLOAT32_C(  -572.84),
        EASYSIMD_FLOAT32_C(  -284.91), EASYSIMD_FLOAT32_C(  -521.66), EASYSIMD_FLOAT32_C(  -264.92), EASYSIMD_FLOAT32_C(  -373.11) },
      { EASYSIMD_FLOAT32_C(   466.53), EASYSIMD_FLOAT32_C(   -18.35), EASYSIMD_FLOAT32_C(   819.28), EASYSIMD_FLOAT32_C(  -476.81),
        EASYSIMD_FLOAT32_C(   466.53), EASYSIMD_FLOAT32_C(   -18.35), EASYSIMD_FLOAT32_C(   819.28), EASYSIMD_FLOAT32_C(  -476.81),
        EASYSIMD_FLOAT32_C(   747.78), EASYSIMD_FLOAT32_C(   105.87), EASYSIMD_FLOAT32_C(  -160.32), EASYSIMD_FLOAT32_C(  -120.51),
        EASYSIMD_FLOAT32_C(  -381.77), EASYSIMD_FLOAT32_C(  -498.15), EASYSIMD_FLOAT32_C(    35.08), EASYSIMD_FLOAT32_C(  -572.84) },
      { EASYSIMD_FLOAT32_C(  -835.70), EASYSIMD_FLOAT32_C(   342.03), EASYSIMD_FLOAT32_C(    16.52), EASYSIMD_FLOAT32_C(  -774.52),
        EASYSIMD_FLOAT32_C(   973.30), EASYSIMD_FLOAT32_C(   555.05), EASYSIMD_FLOAT32_C(  -668.58), EASYSIMD_FLOAT32_C(  -924.60),
        EASYSIMD_FLOAT32_C(  -487.49), EASYSIMD_FLOAT32_C(    62.96), EASYSIMD_FLOAT32_C(  -943.98), EASYSIMD_FLOAT32_C(  -388.66),
        EASYSIMD_FLOAT32_C(  -487.49), EASYSIMD_FLOAT32_C(    62.96), EASYSIMD_FLOAT32_C(  -943.98), EASYSIMD_FLOAT32_C(  -388.66) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;

    r = easysimd_mm512_shuffle_f32x4(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r1), 1);
    r = easysimd_mm512_shuffle_f32x4(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r2), 1);
    r = easysimd_mm512_shuffle_f32x4(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r3), 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_shuffle_f32x4(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_shuffle_f32x4");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r4), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r;

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f32x4(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f32x4(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f32x4(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f32x4(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_shuffle_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[16];
    const easysimd__mmask16 k;
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r1[16];
    const easysimd_float32 r2[16];
    const easysimd_float32 r3[16];
    const easysimd_float32 r4[16];
  } test_vec[4] = {
    { { EASYSIMD_FLOAT32_C(  -725.57), EASYSIMD_FLOAT32_C(  -383.90), EASYSIMD_FLOAT32_C(   720.60), EASYSIMD_FLOAT32_C(   554.17),
        EASYSIMD_FLOAT32_C(   448.60), EASYSIMD_FLOAT32_C(   822.68), EASYSIMD_FLOAT32_C(  -709.42), EASYSIMD_FLOAT32_C(  -354.77),
        EASYSIMD_FLOAT32_C(  -252.87), EASYSIMD_FLOAT32_C(   698.39), EASYSIMD_FLOAT32_C(   892.46), EASYSIMD_FLOAT32_C(   204.87),
        EASYSIMD_FLOAT32_C(   942.56), EASYSIMD_FLOAT32_C(   -22.09), EASYSIMD_FLOAT32_C(   -20.72), EASYSIMD_FLOAT32_C(  -714.33) },
      UINT16_C(57460),
      { EASYSIMD_FLOAT32_C(   182.18), EASYSIMD_FLOAT32_C(   667.71), EASYSIMD_FLOAT32_C(   770.89), EASYSIMD_FLOAT32_C(  -144.25),
        EASYSIMD_FLOAT32_C(  -458.14), EASYSIMD_FLOAT32_C(  -547.82), EASYSIMD_FLOAT32_C(   115.83), EASYSIMD_FLOAT32_C(  -131.04),
        EASYSIMD_FLOAT32_C(   173.86), EASYSIMD_FLOAT32_C(   778.80), EASYSIMD_FLOAT32_C(  -446.61), EASYSIMD_FLOAT32_C(  -472.37),
        EASYSIMD_FLOAT32_C(    42.51), EASYSIMD_FLOAT32_C(  -172.17), EASYSIMD_FLOAT32_C(   143.73), EASYSIMD_FLOAT32_C(  -236.89) },
      { EASYSIMD_FLOAT32_C(  -618.00), EASYSIMD_FLOAT32_C(  -407.67), EASYSIMD_FLOAT32_C(  -414.21), EASYSIMD_FLOAT32_C(  -327.43),
        EASYSIMD_FLOAT32_C(   237.55), EASYSIMD_FLOAT32_C(   332.92), EASYSIMD_FLOAT32_C(  -629.03), EASYSIMD_FLOAT32_C(   130.01),
        EASYSIMD_FLOAT32_C(  -462.21), EASYSIMD_FLOAT32_C(  -686.47), EASYSIMD_FLOAT32_C(  -892.08), EASYSIMD_FLOAT32_C(   517.07),
        EASYSIMD_FLOAT32_C(  -400.81), EASYSIMD_FLOAT32_C(  -907.38), EASYSIMD_FLOAT32_C(   859.44), EASYSIMD_FLOAT32_C(   781.38) },
      { EASYSIMD_FLOAT32_C(  -725.57), EASYSIMD_FLOAT32_C(  -383.90), EASYSIMD_FLOAT32_C(  -446.61), EASYSIMD_FLOAT32_C(   554.17),
        EASYSIMD_FLOAT32_C(    42.51), EASYSIMD_FLOAT32_C(  -172.17), EASYSIMD_FLOAT32_C(   143.73), EASYSIMD_FLOAT32_C(  -354.77),
        EASYSIMD_FLOAT32_C(  -252.87), EASYSIMD_FLOAT32_C(   698.39), EASYSIMD_FLOAT32_C(   892.46), EASYSIMD_FLOAT32_C(   204.87),
        EASYSIMD_FLOAT32_C(   942.56), EASYSIMD_FLOAT32_C(   332.92), EASYSIMD_FLOAT32_C(  -629.03), EASYSIMD_FLOAT32_C(   130.01) },
      { EASYSIMD_FLOAT32_C(  -725.57), EASYSIMD_FLOAT32_C(  -383.90), EASYSIMD_FLOAT32_C(   115.83), EASYSIMD_FLOAT32_C(   554.17),
        EASYSIMD_FLOAT32_C(   173.86), EASYSIMD_FLOAT32_C(   778.80), EASYSIMD_FLOAT32_C(  -446.61), EASYSIMD_FLOAT32_C(  -354.77),
        EASYSIMD_FLOAT32_C(  -252.87), EASYSIMD_FLOAT32_C(   698.39), EASYSIMD_FLOAT32_C(   892.46), EASYSIMD_FLOAT32_C(   204.87),
        EASYSIMD_FLOAT32_C(   942.56), EASYSIMD_FLOAT32_C(  -407.67), EASYSIMD_FLOAT32_C(  -414.21), EASYSIMD_FLOAT32_C(  -327.43) },
      { EASYSIMD_FLOAT32_C(  -725.57), EASYSIMD_FLOAT32_C(  -383.90), EASYSIMD_FLOAT32_C(  -446.61), EASYSIMD_FLOAT32_C(   554.17),
        EASYSIMD_FLOAT32_C(   173.86), EASYSIMD_FLOAT32_C(   778.80), EASYSIMD_FLOAT32_C(  -446.61), EASYSIMD_FLOAT32_C(  -354.77),
        EASYSIMD_FLOAT32_C(  -252.87), EASYSIMD_FLOAT32_C(   698.39), EASYSIMD_FLOAT32_C(   892.46), EASYSIMD_FLOAT32_C(   204.87),
        EASYSIMD_FLOAT32_C(   942.56), EASYSIMD_FLOAT32_C(  -907.38), EASYSIMD_FLOAT32_C(   859.44), EASYSIMD_FLOAT32_C(   781.38) },
      { EASYSIMD_FLOAT32_C(  -725.57), EASYSIMD_FLOAT32_C(  -383.90), EASYSIMD_FLOAT32_C(   143.73), EASYSIMD_FLOAT32_C(   554.17),
        EASYSIMD_FLOAT32_C(  -458.14), EASYSIMD_FLOAT32_C(  -547.82), EASYSIMD_FLOAT32_C(   115.83), EASYSIMD_FLOAT32_C(  -354.77),
        EASYSIMD_FLOAT32_C(  -252.87), EASYSIMD_FLOAT32_C(   698.39), EASYSIMD_FLOAT32_C(   892.46), EASYSIMD_FLOAT32_C(   204.87),
        EASYSIMD_FLOAT32_C(   942.56), EASYSIMD_FLOAT32_C(  -686.47), EASYSIMD_FLOAT32_C(  -892.08), EASYSIMD_FLOAT32_C(   517.07) } },
    { { EASYSIMD_FLOAT32_C(   760.33), EASYSIMD_FLOAT32_C(   630.33), EASYSIMD_FLOAT32_C(  -362.87), EASYSIMD_FLOAT32_C(  -697.81),
        EASYSIMD_FLOAT32_C(  -917.49), EASYSIMD_FLOAT32_C(   752.96), EASYSIMD_FLOAT32_C(   171.15), EASYSIMD_FLOAT32_C(   256.37),
        EASYSIMD_FLOAT32_C(   531.76), EASYSIMD_FLOAT32_C(   724.54), EASYSIMD_FLOAT32_C(   784.00), EASYSIMD_FLOAT32_C(  -425.73),
        EASYSIMD_FLOAT32_C(  -447.63), EASYSIMD_FLOAT32_C(   -72.27), EASYSIMD_FLOAT32_C(   337.38), EASYSIMD_FLOAT32_C(   -65.64) },
      UINT16_C(36452),
      { EASYSIMD_FLOAT32_C(   606.94), EASYSIMD_FLOAT32_C(  -242.39), EASYSIMD_FLOAT32_C(   256.10), EASYSIMD_FLOAT32_C(   977.90),
        EASYSIMD_FLOAT32_C(   887.62), EASYSIMD_FLOAT32_C(   793.89), EASYSIMD_FLOAT32_C(  -708.57), EASYSIMD_FLOAT32_C(   995.54),
        EASYSIMD_FLOAT32_C(   310.96), EASYSIMD_FLOAT32_C(  -109.37), EASYSIMD_FLOAT32_C(  -911.84), EASYSIMD_FLOAT32_C(   170.40),
        EASYSIMD_FLOAT32_C(  -328.00), EASYSIMD_FLOAT32_C(   848.49), EASYSIMD_FLOAT32_C(  -199.28), EASYSIMD_FLOAT32_C(   309.13) },
      { EASYSIMD_FLOAT32_C(  -849.32), EASYSIMD_FLOAT32_C(  -116.77), EASYSIMD_FLOAT32_C(    62.09), EASYSIMD_FLOAT32_C(   321.83),
        EASYSIMD_FLOAT32_C(  -860.40), EASYSIMD_FLOAT32_C(  -406.15), EASYSIMD_FLOAT32_C(    46.37), EASYSIMD_FLOAT32_C(   923.60),
        EASYSIMD_FLOAT32_C(   168.12), EASYSIMD_FLOAT32_C(   598.74), EASYSIMD_FLOAT32_C(  -148.68), EASYSIMD_FLOAT32_C(  -494.50),
        EASYSIMD_FLOAT32_C(  -466.90), EASYSIMD_FLOAT32_C(  -628.62), EASYSIMD_FLOAT32_C(  -571.32), EASYSIMD_FLOAT32_C(  -859.96) },
      { EASYSIMD_FLOAT32_C(   760.33), EASYSIMD_FLOAT32_C(   630.33), EASYSIMD_FLOAT32_C(  -911.84), EASYSIMD_FLOAT32_C(  -697.81),
        EASYSIMD_FLOAT32_C(  -917.49), EASYSIMD_FLOAT32_C(   848.49), EASYSIMD_FLOAT32_C(  -199.28), EASYSIMD_FLOAT32_C(   256.37),
        EASYSIMD_FLOAT32_C(   531.76), EASYSIMD_FLOAT32_C(  -116.77), EASYSIMD_FLOAT32_C(    62.09), EASYSIMD_FLOAT32_C(   321.83),
        EASYSIMD_FLOAT32_C(  -447.63), EASYSIMD_FLOAT32_C(   -72.27), EASYSIMD_FLOAT32_C(   337.38), EASYSIMD_FLOAT32_C(   923.60) },
      { EASYSIMD_FLOAT32_C(   760.33), EASYSIMD_FLOAT32_C(   630.33), EASYSIMD_FLOAT32_C(  -708.57), EASYSIMD_FLOAT32_C(  -697.81),
        EASYSIMD_FLOAT32_C(  -917.49), EASYSIMD_FLOAT32_C(  -109.37), EASYSIMD_FLOAT32_C(  -911.84), EASYSIMD_FLOAT32_C(   256.37),
        EASYSIMD_FLOAT32_C(   531.76), EASYSIMD_FLOAT32_C(  -628.62), EASYSIMD_FLOAT32_C(  -571.32), EASYSIMD_FLOAT32_C(  -859.96),
        EASYSIMD_FLOAT32_C(  -447.63), EASYSIMD_FLOAT32_C(   -72.27), EASYSIMD_FLOAT32_C(   337.38), EASYSIMD_FLOAT32_C(   321.83) },
      { EASYSIMD_FLOAT32_C(   760.33), EASYSIMD_FLOAT32_C(   630.33), EASYSIMD_FLOAT32_C(  -911.84), EASYSIMD_FLOAT32_C(  -697.81),
        EASYSIMD_FLOAT32_C(  -917.49), EASYSIMD_FLOAT32_C(  -109.37), EASYSIMD_FLOAT32_C(  -911.84), EASYSIMD_FLOAT32_C(   256.37),
        EASYSIMD_FLOAT32_C(   531.76), EASYSIMD_FLOAT32_C(  -406.15), EASYSIMD_FLOAT32_C(    46.37), EASYSIMD_FLOAT32_C(   923.60),
        EASYSIMD_FLOAT32_C(  -447.63), EASYSIMD_FLOAT32_C(   -72.27), EASYSIMD_FLOAT32_C(   337.38), EASYSIMD_FLOAT32_C(  -859.96) },
      { EASYSIMD_FLOAT32_C(   760.33), EASYSIMD_FLOAT32_C(   630.33), EASYSIMD_FLOAT32_C(  -199.28), EASYSIMD_FLOAT32_C(  -697.81),
        EASYSIMD_FLOAT32_C(  -917.49), EASYSIMD_FLOAT32_C(   793.89), EASYSIMD_FLOAT32_C(  -708.57), EASYSIMD_FLOAT32_C(   256.37),
        EASYSIMD_FLOAT32_C(   531.76), EASYSIMD_FLOAT32_C(   598.74), EASYSIMD_FLOAT32_C(  -148.68), EASYSIMD_FLOAT32_C(  -494.50),
        EASYSIMD_FLOAT32_C(  -447.63), EASYSIMD_FLOAT32_C(   -72.27), EASYSIMD_FLOAT32_C(   337.38), EASYSIMD_FLOAT32_C(  -494.50) } },
    { { EASYSIMD_FLOAT32_C(   128.98), EASYSIMD_FLOAT32_C(   684.78), EASYSIMD_FLOAT32_C(  -882.06), EASYSIMD_FLOAT32_C(    16.60),
        EASYSIMD_FLOAT32_C(   478.66), EASYSIMD_FLOAT32_C(  -590.62), EASYSIMD_FLOAT32_C(    12.14), EASYSIMD_FLOAT32_C(  -210.38),
        EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(   100.31), EASYSIMD_FLOAT32_C(   960.02), EASYSIMD_FLOAT32_C(   972.00),
        EASYSIMD_FLOAT32_C(   -51.20), EASYSIMD_FLOAT32_C(  -239.26), EASYSIMD_FLOAT32_C(   281.13), EASYSIMD_FLOAT32_C(    99.48) },
      UINT16_C(60205),
      { EASYSIMD_FLOAT32_C(  -578.69), EASYSIMD_FLOAT32_C(   783.57), EASYSIMD_FLOAT32_C(   -62.93), EASYSIMD_FLOAT32_C(   467.68),
        EASYSIMD_FLOAT32_C(   707.16), EASYSIMD_FLOAT32_C(  -894.81), EASYSIMD_FLOAT32_C(    66.42), EASYSIMD_FLOAT32_C(  -441.51),
        EASYSIMD_FLOAT32_C(  -389.31), EASYSIMD_FLOAT32_C(   599.52), EASYSIMD_FLOAT32_C(   -70.14), EASYSIMD_FLOAT32_C(    39.37),
        EASYSIMD_FLOAT32_C(   739.56), EASYSIMD_FLOAT32_C(  -941.15), EASYSIMD_FLOAT32_C(  -275.85), EASYSIMD_FLOAT32_C(   857.50) },
      { EASYSIMD_FLOAT32_C(    75.45), EASYSIMD_FLOAT32_C(  -797.19), EASYSIMD_FLOAT32_C(  -733.12), EASYSIMD_FLOAT32_C(  -912.41),
        EASYSIMD_FLOAT32_C(    -7.57), EASYSIMD_FLOAT32_C(   566.88), EASYSIMD_FLOAT32_C(   187.90), EASYSIMD_FLOAT32_C(   -47.55),
        EASYSIMD_FLOAT32_C(   538.88), EASYSIMD_FLOAT32_C(  -863.30), EASYSIMD_FLOAT32_C(   713.19), EASYSIMD_FLOAT32_C(  -179.98),
        EASYSIMD_FLOAT32_C(   236.18), EASYSIMD_FLOAT32_C(   357.17), EASYSIMD_FLOAT32_C(   163.24), EASYSIMD_FLOAT32_C(   657.49) },
      { EASYSIMD_FLOAT32_C(  -389.31), EASYSIMD_FLOAT32_C(   684.78), EASYSIMD_FLOAT32_C(   -70.14), EASYSIMD_FLOAT32_C(    39.37),
        EASYSIMD_FLOAT32_C(   478.66), EASYSIMD_FLOAT32_C(  -941.15), EASYSIMD_FLOAT32_C(    12.14), EASYSIMD_FLOAT32_C(  -210.38),
        EASYSIMD_FLOAT32_C(    75.45), EASYSIMD_FLOAT32_C(  -797.19), EASYSIMD_FLOAT32_C(   960.02), EASYSIMD_FLOAT32_C(  -912.41),
        EASYSIMD_FLOAT32_C(   -51.20), EASYSIMD_FLOAT32_C(   566.88), EASYSIMD_FLOAT32_C(   187.90), EASYSIMD_FLOAT32_C(   -47.55) },
      { EASYSIMD_FLOAT32_C(   707.16), EASYSIMD_FLOAT32_C(   684.78), EASYSIMD_FLOAT32_C(    66.42), EASYSIMD_FLOAT32_C(  -441.51),
        EASYSIMD_FLOAT32_C(   478.66), EASYSIMD_FLOAT32_C(   599.52), EASYSIMD_FLOAT32_C(    12.14), EASYSIMD_FLOAT32_C(  -210.38),
        EASYSIMD_FLOAT32_C(   236.18), EASYSIMD_FLOAT32_C(   357.17), EASYSIMD_FLOAT32_C(   960.02), EASYSIMD_FLOAT32_C(   657.49),
        EASYSIMD_FLOAT32_C(   -51.20), EASYSIMD_FLOAT32_C(  -797.19), EASYSIMD_FLOAT32_C(  -733.12), EASYSIMD_FLOAT32_C(  -912.41) },
      { EASYSIMD_FLOAT32_C(  -389.31), EASYSIMD_FLOAT32_C(   684.78), EASYSIMD_FLOAT32_C(   -70.14), EASYSIMD_FLOAT32_C(    39.37),
        EASYSIMD_FLOAT32_C(   478.66), EASYSIMD_FLOAT32_C(   599.52), EASYSIMD_FLOAT32_C(    12.14), EASYSIMD_FLOAT32_C(  -210.38),
        EASYSIMD_FLOAT32_C(    -7.57), EASYSIMD_FLOAT32_C(   566.88), EASYSIMD_FLOAT32_C(   960.02), EASYSIMD_FLOAT32_C(   -47.55),
        EASYSIMD_FLOAT32_C(   -51.20), EASYSIMD_FLOAT32_C(   357.17), EASYSIMD_FLOAT32_C(   163.24), EASYSIMD_FLOAT32_C(   657.49) },
      { EASYSIMD_FLOAT32_C(   739.56), EASYSIMD_FLOAT32_C(   684.78), EASYSIMD_FLOAT32_C(  -275.85), EASYSIMD_FLOAT32_C(   857.50),
        EASYSIMD_FLOAT32_C(   478.66), EASYSIMD_FLOAT32_C(  -894.81), EASYSIMD_FLOAT32_C(    12.14), EASYSIMD_FLOAT32_C(  -210.38),
        EASYSIMD_FLOAT32_C(   538.88), EASYSIMD_FLOAT32_C(  -863.30), EASYSIMD_FLOAT32_C(   960.02), EASYSIMD_FLOAT32_C(  -179.98),
        EASYSIMD_FLOAT32_C(   -51.20), EASYSIMD_FLOAT32_C(  -863.30), EASYSIMD_FLOAT32_C(   713.19), EASYSIMD_FLOAT32_C(  -179.98) } },
    { { EASYSIMD_FLOAT32_C(   140.73), EASYSIMD_FLOAT32_C(  -899.69), EASYSIMD_FLOAT32_C(   125.17), EASYSIMD_FLOAT32_C(  -152.10),
        EASYSIMD_FLOAT32_C(  -794.50), EASYSIMD_FLOAT32_C(  -808.41), EASYSIMD_FLOAT32_C(   406.38), EASYSIMD_FLOAT32_C(  -183.81),
        EASYSIMD_FLOAT32_C(   791.11), EASYSIMD_FLOAT32_C(  -663.76), EASYSIMD_FLOAT32_C(   855.56), EASYSIMD_FLOAT32_C(   530.67),
        EASYSIMD_FLOAT32_C(  -604.91), EASYSIMD_FLOAT32_C(  -420.29), EASYSIMD_FLOAT32_C(   388.17), EASYSIMD_FLOAT32_C(   470.54) },
      UINT16_C( 6216),
      { EASYSIMD_FLOAT32_C(   558.14), EASYSIMD_FLOAT32_C(   774.95), EASYSIMD_FLOAT32_C(   221.93), EASYSIMD_FLOAT32_C(  -253.96),
        EASYSIMD_FLOAT32_C(  -272.60), EASYSIMD_FLOAT32_C(  -239.19), EASYSIMD_FLOAT32_C(  -117.26), EASYSIMD_FLOAT32_C(  -559.40),
        EASYSIMD_FLOAT32_C(   580.83), EASYSIMD_FLOAT32_C(  -881.08), EASYSIMD_FLOAT32_C(   797.76), EASYSIMD_FLOAT32_C(  -255.93),
        EASYSIMD_FLOAT32_C(   776.41), EASYSIMD_FLOAT32_C(   -61.50), EASYSIMD_FLOAT32_C(  -155.62), EASYSIMD_FLOAT32_C(   -98.42) },
      { EASYSIMD_FLOAT32_C(   786.39), EASYSIMD_FLOAT32_C(    49.88), EASYSIMD_FLOAT32_C(    93.16), EASYSIMD_FLOAT32_C(   192.77),
        EASYSIMD_FLOAT32_C(   866.08), EASYSIMD_FLOAT32_C(  -115.73), EASYSIMD_FLOAT32_C(   529.02), EASYSIMD_FLOAT32_C(   721.64),
        EASYSIMD_FLOAT32_C(  -585.06), EASYSIMD_FLOAT32_C(   924.11), EASYSIMD_FLOAT32_C(  -698.65), EASYSIMD_FLOAT32_C(   803.11),
        EASYSIMD_FLOAT32_C(   394.65), EASYSIMD_FLOAT32_C(    83.87), EASYSIMD_FLOAT32_C(   458.16), EASYSIMD_FLOAT32_C(   -47.21) },
      { EASYSIMD_FLOAT32_C(   140.73), EASYSIMD_FLOAT32_C(  -899.69), EASYSIMD_FLOAT32_C(   125.17), EASYSIMD_FLOAT32_C(  -255.93),
        EASYSIMD_FLOAT32_C(  -794.50), EASYSIMD_FLOAT32_C(  -808.41), EASYSIMD_FLOAT32_C(  -155.62), EASYSIMD_FLOAT32_C(  -183.81),
        EASYSIMD_FLOAT32_C(   791.11), EASYSIMD_FLOAT32_C(  -663.76), EASYSIMD_FLOAT32_C(   855.56), EASYSIMD_FLOAT32_C(   192.77),
        EASYSIMD_FLOAT32_C(   866.08), EASYSIMD_FLOAT32_C(  -420.29), EASYSIMD_FLOAT32_C(   388.17), EASYSIMD_FLOAT32_C(   470.54) },
      { EASYSIMD_FLOAT32_C(   140.73), EASYSIMD_FLOAT32_C(  -899.69), EASYSIMD_FLOAT32_C(   125.17), EASYSIMD_FLOAT32_C(  -559.40),
        EASYSIMD_FLOAT32_C(  -794.50), EASYSIMD_FLOAT32_C(  -808.41), EASYSIMD_FLOAT32_C(   797.76), EASYSIMD_FLOAT32_C(  -183.81),
        EASYSIMD_FLOAT32_C(   791.11), EASYSIMD_FLOAT32_C(  -663.76), EASYSIMD_FLOAT32_C(   855.56), EASYSIMD_FLOAT32_C(   -47.21),
        EASYSIMD_FLOAT32_C(   786.39), EASYSIMD_FLOAT32_C(  -420.29), EASYSIMD_FLOAT32_C(   388.17), EASYSIMD_FLOAT32_C(   470.54) },
      { EASYSIMD_FLOAT32_C(   140.73), EASYSIMD_FLOAT32_C(  -899.69), EASYSIMD_FLOAT32_C(   125.17), EASYSIMD_FLOAT32_C(  -255.93),
        EASYSIMD_FLOAT32_C(  -794.50), EASYSIMD_FLOAT32_C(  -808.41), EASYSIMD_FLOAT32_C(   797.76), EASYSIMD_FLOAT32_C(  -183.81),
        EASYSIMD_FLOAT32_C(   791.11), EASYSIMD_FLOAT32_C(  -663.76), EASYSIMD_FLOAT32_C(   855.56), EASYSIMD_FLOAT32_C(   721.64),
        EASYSIMD_FLOAT32_C(   394.65), EASYSIMD_FLOAT32_C(  -420.29), EASYSIMD_FLOAT32_C(   388.17), EASYSIMD_FLOAT32_C(   470.54) },
      { EASYSIMD_FLOAT32_C(   140.73), EASYSIMD_FLOAT32_C(  -899.69), EASYSIMD_FLOAT32_C(   125.17), EASYSIMD_FLOAT32_C(   -98.42),
        EASYSIMD_FLOAT32_C(  -794.50), EASYSIMD_FLOAT32_C(  -808.41), EASYSIMD_FLOAT32_C(  -117.26), EASYSIMD_FLOAT32_C(  -183.81),
        EASYSIMD_FLOAT32_C(   791.11), EASYSIMD_FLOAT32_C(  -663.76), EASYSIMD_FLOAT32_C(   855.56), EASYSIMD_FLOAT32_C(   803.11),
        EASYSIMD_FLOAT32_C(  -585.06), EASYSIMD_FLOAT32_C(  -420.29), EASYSIMD_FLOAT32_C(   388.17), EASYSIMD_FLOAT32_C(   470.54) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;

    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r1), 1);
    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r2), 1);
    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r3), 1);
    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r4), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512 src = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r;

    easysimd_test_x86_write_f32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f32x4(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_shuffle_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask16 k;
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r1[16];
    const easysimd_float32 r2[16];
    const easysimd_float32 r3[16];
    const easysimd_float32 r4[16];
  } test_vec[4] = {
    { UINT16_C(43927),
      { EASYSIMD_FLOAT32_C(   369.14), EASYSIMD_FLOAT32_C(   354.32), EASYSIMD_FLOAT32_C(   623.24), EASYSIMD_FLOAT32_C(   353.44),
        EASYSIMD_FLOAT32_C(   642.86), EASYSIMD_FLOAT32_C(    91.50), EASYSIMD_FLOAT32_C(   851.20), EASYSIMD_FLOAT32_C(  -709.38),
        EASYSIMD_FLOAT32_C(   791.58), EASYSIMD_FLOAT32_C(   170.73), EASYSIMD_FLOAT32_C(   771.96), EASYSIMD_FLOAT32_C(   365.77),
        EASYSIMD_FLOAT32_C(  -663.84), EASYSIMD_FLOAT32_C(  -903.26), EASYSIMD_FLOAT32_C(  -343.54), EASYSIMD_FLOAT32_C(   983.63) },
      { EASYSIMD_FLOAT32_C(   796.16), EASYSIMD_FLOAT32_C(  -952.47), EASYSIMD_FLOAT32_C(   568.60), EASYSIMD_FLOAT32_C(   931.49),
        EASYSIMD_FLOAT32_C(  -388.48), EASYSIMD_FLOAT32_C(   788.36), EASYSIMD_FLOAT32_C(   380.68), EASYSIMD_FLOAT32_C(   675.57),
        EASYSIMD_FLOAT32_C(    26.10), EASYSIMD_FLOAT32_C(   385.39), EASYSIMD_FLOAT32_C(   261.01), EASYSIMD_FLOAT32_C(   184.97),
        EASYSIMD_FLOAT32_C(     9.80), EASYSIMD_FLOAT32_C(  -711.89), EASYSIMD_FLOAT32_C(  -425.15), EASYSIMD_FLOAT32_C(  -621.06) },
      { EASYSIMD_FLOAT32_C(   791.58), EASYSIMD_FLOAT32_C(   170.73), EASYSIMD_FLOAT32_C(   771.96), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -663.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   983.63),
        EASYSIMD_FLOAT32_C(   796.16), EASYSIMD_FLOAT32_C(  -952.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   931.49),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   788.36), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   675.57) },
      { EASYSIMD_FLOAT32_C(   642.86), EASYSIMD_FLOAT32_C(    91.50), EASYSIMD_FLOAT32_C(   851.20), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   791.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   365.77),
        EASYSIMD_FLOAT32_C(     9.80), EASYSIMD_FLOAT32_C(  -711.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -621.06),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -952.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   931.49) },
      { EASYSIMD_FLOAT32_C(   791.58), EASYSIMD_FLOAT32_C(   170.73), EASYSIMD_FLOAT32_C(   771.96), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   791.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   365.77),
        EASYSIMD_FLOAT32_C(  -388.48), EASYSIMD_FLOAT32_C(   788.36), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   675.57),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -711.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -621.06) },
      { EASYSIMD_FLOAT32_C(  -663.84), EASYSIMD_FLOAT32_C(  -903.26), EASYSIMD_FLOAT32_C(  -343.54), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   642.86), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -709.38),
        EASYSIMD_FLOAT32_C(    26.10), EASYSIMD_FLOAT32_C(   385.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   184.97),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   385.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   184.97) } },
    { UINT16_C(50370),
      { EASYSIMD_FLOAT32_C(   732.38), EASYSIMD_FLOAT32_C(   285.29), EASYSIMD_FLOAT32_C(   289.59), EASYSIMD_FLOAT32_C(   583.58),
        EASYSIMD_FLOAT32_C(   575.91), EASYSIMD_FLOAT32_C(    81.17), EASYSIMD_FLOAT32_C(  -245.69), EASYSIMD_FLOAT32_C(   347.87),
        EASYSIMD_FLOAT32_C(  -553.05), EASYSIMD_FLOAT32_C(    90.47), EASYSIMD_FLOAT32_C(   444.61), EASYSIMD_FLOAT32_C(   103.40),
        EASYSIMD_FLOAT32_C(    74.10), EASYSIMD_FLOAT32_C(   240.78), EASYSIMD_FLOAT32_C(   150.93), EASYSIMD_FLOAT32_C(  -357.31) },
      { EASYSIMD_FLOAT32_C(   172.27), EASYSIMD_FLOAT32_C(   762.45), EASYSIMD_FLOAT32_C(  -568.95), EASYSIMD_FLOAT32_C(  -447.05),
        EASYSIMD_FLOAT32_C(   438.01), EASYSIMD_FLOAT32_C(   457.15), EASYSIMD_FLOAT32_C(   938.34), EASYSIMD_FLOAT32_C(  -300.97),
        EASYSIMD_FLOAT32_C(  -357.88), EASYSIMD_FLOAT32_C(   -51.86), EASYSIMD_FLOAT32_C(   -12.86), EASYSIMD_FLOAT32_C(   216.97),
        EASYSIMD_FLOAT32_C(   327.09), EASYSIMD_FLOAT32_C(  -370.42), EASYSIMD_FLOAT32_C(   415.06), EASYSIMD_FLOAT32_C(    59.47) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   150.93), EASYSIMD_FLOAT32_C(  -357.31),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -568.95), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   938.34), EASYSIMD_FLOAT32_C(  -300.97) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    81.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   444.61), EASYSIMD_FLOAT32_C(   103.40),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   415.06), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -568.95), EASYSIMD_FLOAT32_C(  -447.05) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   444.61), EASYSIMD_FLOAT32_C(   103.40),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   938.34), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   415.06), EASYSIMD_FLOAT32_C(    59.47) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   240.78), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -245.69), EASYSIMD_FLOAT32_C(   347.87),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -12.86), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -12.86), EASYSIMD_FLOAT32_C(   216.97) } },
    { UINT16_C(27593),
      { EASYSIMD_FLOAT32_C(  -356.95), EASYSIMD_FLOAT32_C(   490.78), EASYSIMD_FLOAT32_C(   785.82), EASYSIMD_FLOAT32_C(   397.35),
        EASYSIMD_FLOAT32_C(  -161.34), EASYSIMD_FLOAT32_C(  -767.23), EASYSIMD_FLOAT32_C(  -512.18), EASYSIMD_FLOAT32_C(  -716.73),
        EASYSIMD_FLOAT32_C(   336.17), EASYSIMD_FLOAT32_C(   561.92), EASYSIMD_FLOAT32_C(   524.05), EASYSIMD_FLOAT32_C(  -512.90),
        EASYSIMD_FLOAT32_C(  -795.39), EASYSIMD_FLOAT32_C(  -303.68), EASYSIMD_FLOAT32_C(  -750.45), EASYSIMD_FLOAT32_C(  -364.34) },
      { EASYSIMD_FLOAT32_C(   249.27), EASYSIMD_FLOAT32_C(   687.56), EASYSIMD_FLOAT32_C(  -907.19), EASYSIMD_FLOAT32_C(   187.61),
        EASYSIMD_FLOAT32_C(  -613.41), EASYSIMD_FLOAT32_C(  -265.06), EASYSIMD_FLOAT32_C(  -864.25), EASYSIMD_FLOAT32_C(   373.73),
        EASYSIMD_FLOAT32_C(   951.91), EASYSIMD_FLOAT32_C(   462.84), EASYSIMD_FLOAT32_C(  -996.69), EASYSIMD_FLOAT32_C(   366.97),
        EASYSIMD_FLOAT32_C(  -477.69), EASYSIMD_FLOAT32_C(   918.18), EASYSIMD_FLOAT32_C(  -928.38), EASYSIMD_FLOAT32_C(   165.36) },
      { EASYSIMD_FLOAT32_C(   336.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -512.90),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -750.45), EASYSIMD_FLOAT32_C(  -364.34),
        EASYSIMD_FLOAT32_C(   249.27), EASYSIMD_FLOAT32_C(   687.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   187.61),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -265.06), EASYSIMD_FLOAT32_C(  -864.25), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -161.34), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -716.73),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   524.05), EASYSIMD_FLOAT32_C(  -512.90),
        EASYSIMD_FLOAT32_C(  -477.69), EASYSIMD_FLOAT32_C(   918.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   165.36),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   687.56), EASYSIMD_FLOAT32_C(  -907.19), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(   336.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -512.90),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   524.05), EASYSIMD_FLOAT32_C(  -512.90),
        EASYSIMD_FLOAT32_C(  -613.41), EASYSIMD_FLOAT32_C(  -265.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   373.73),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   918.18), EASYSIMD_FLOAT32_C(  -928.38), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(  -795.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -364.34),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -512.18), EASYSIMD_FLOAT32_C(  -716.73),
        EASYSIMD_FLOAT32_C(   951.91), EASYSIMD_FLOAT32_C(   462.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   366.97),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   462.84), EASYSIMD_FLOAT32_C(  -996.69), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(32572),
      { EASYSIMD_FLOAT32_C(  -437.29), EASYSIMD_FLOAT32_C(  -752.38), EASYSIMD_FLOAT32_C(  -909.78), EASYSIMD_FLOAT32_C(    50.53),
        EASYSIMD_FLOAT32_C(  -469.11), EASYSIMD_FLOAT32_C(   426.39), EASYSIMD_FLOAT32_C(  -387.55), EASYSIMD_FLOAT32_C(  -945.06),
        EASYSIMD_FLOAT32_C(   913.49), EASYSIMD_FLOAT32_C(  -182.94), EASYSIMD_FLOAT32_C(  -248.74), EASYSIMD_FLOAT32_C(  -836.96),
        EASYSIMD_FLOAT32_C(   452.73), EASYSIMD_FLOAT32_C(  -999.47), EASYSIMD_FLOAT32_C(   850.60), EASYSIMD_FLOAT32_C(   545.54) },
      { EASYSIMD_FLOAT32_C(   188.14), EASYSIMD_FLOAT32_C(  -762.80), EASYSIMD_FLOAT32_C(  -719.53), EASYSIMD_FLOAT32_C(   323.90),
        EASYSIMD_FLOAT32_C(   610.93), EASYSIMD_FLOAT32_C(  -767.62), EASYSIMD_FLOAT32_C(  -213.26), EASYSIMD_FLOAT32_C(   614.24),
        EASYSIMD_FLOAT32_C(   599.35), EASYSIMD_FLOAT32_C(   309.05), EASYSIMD_FLOAT32_C(   532.42), EASYSIMD_FLOAT32_C(   670.97),
        EASYSIMD_FLOAT32_C(  -525.60), EASYSIMD_FLOAT32_C(   -58.61), EASYSIMD_FLOAT32_C(   528.42), EASYSIMD_FLOAT32_C(    37.11) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -248.74), EASYSIMD_FLOAT32_C(  -836.96),
        EASYSIMD_FLOAT32_C(   452.73), EASYSIMD_FLOAT32_C(  -999.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   188.14), EASYSIMD_FLOAT32_C(  -762.80), EASYSIMD_FLOAT32_C(  -719.53), EASYSIMD_FLOAT32_C(   323.90),
        EASYSIMD_FLOAT32_C(   610.93), EASYSIMD_FLOAT32_C(  -767.62), EASYSIMD_FLOAT32_C(  -213.26), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -387.55), EASYSIMD_FLOAT32_C(  -945.06),
        EASYSIMD_FLOAT32_C(   913.49), EASYSIMD_FLOAT32_C(  -182.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -525.60), EASYSIMD_FLOAT32_C(   -58.61), EASYSIMD_FLOAT32_C(   528.42), EASYSIMD_FLOAT32_C(    37.11),
        EASYSIMD_FLOAT32_C(   188.14), EASYSIMD_FLOAT32_C(  -762.80), EASYSIMD_FLOAT32_C(  -719.53), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -248.74), EASYSIMD_FLOAT32_C(  -836.96),
        EASYSIMD_FLOAT32_C(   913.49), EASYSIMD_FLOAT32_C(  -182.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   610.93), EASYSIMD_FLOAT32_C(  -767.62), EASYSIMD_FLOAT32_C(  -213.26), EASYSIMD_FLOAT32_C(   614.24),
        EASYSIMD_FLOAT32_C(  -525.60), EASYSIMD_FLOAT32_C(   -58.61), EASYSIMD_FLOAT32_C(   528.42), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   850.60), EASYSIMD_FLOAT32_C(   545.54),
        EASYSIMD_FLOAT32_C(  -469.11), EASYSIMD_FLOAT32_C(   426.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   599.35), EASYSIMD_FLOAT32_C(   309.05), EASYSIMD_FLOAT32_C(   532.42), EASYSIMD_FLOAT32_C(   670.97),
        EASYSIMD_FLOAT32_C(   599.35), EASYSIMD_FLOAT32_C(   309.05), EASYSIMD_FLOAT32_C(   532.42), EASYSIMD_FLOAT32_C(     0.00) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;

    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r1), 1);
    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r2), 1);
    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r3), 1);
    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r4), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r;

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f32x4(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shuffle_f64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r1[8];
    const easysimd_float64 r2[8];
    const easysimd_float64 r3[8];
    const easysimd_float64 r4[8];
  } test_vec[4] = {
    { { EASYSIMD_FLOAT64_C(  -520.93), EASYSIMD_FLOAT64_C(   576.45), EASYSIMD_FLOAT64_C(   563.43), EASYSIMD_FLOAT64_C(   352.92),
        EASYSIMD_FLOAT64_C(  -311.69), EASYSIMD_FLOAT64_C(  -719.51), EASYSIMD_FLOAT64_C(  -514.13), EASYSIMD_FLOAT64_C(   476.28) },
      { EASYSIMD_FLOAT64_C(  -203.97), EASYSIMD_FLOAT64_C(  -513.95), EASYSIMD_FLOAT64_C(  -501.55), EASYSIMD_FLOAT64_C(   134.12),
        EASYSIMD_FLOAT64_C(  -664.49), EASYSIMD_FLOAT64_C(    81.08), EASYSIMD_FLOAT64_C(   457.05), EASYSIMD_FLOAT64_C(  -756.64) },
      { EASYSIMD_FLOAT64_C(  -311.69), EASYSIMD_FLOAT64_C(  -719.51), EASYSIMD_FLOAT64_C(  -514.13), EASYSIMD_FLOAT64_C(   476.28),
        EASYSIMD_FLOAT64_C(  -203.97), EASYSIMD_FLOAT64_C(  -513.95), EASYSIMD_FLOAT64_C(  -501.55), EASYSIMD_FLOAT64_C(   134.12) },
      { EASYSIMD_FLOAT64_C(   563.43), EASYSIMD_FLOAT64_C(   352.92), EASYSIMD_FLOAT64_C(  -311.69), EASYSIMD_FLOAT64_C(  -719.51),
        EASYSIMD_FLOAT64_C(   457.05), EASYSIMD_FLOAT64_C(  -756.64), EASYSIMD_FLOAT64_C(  -203.97), EASYSIMD_FLOAT64_C(  -513.95) },
      { EASYSIMD_FLOAT64_C(  -311.69), EASYSIMD_FLOAT64_C(  -719.51), EASYSIMD_FLOAT64_C(  -311.69), EASYSIMD_FLOAT64_C(  -719.51),
        EASYSIMD_FLOAT64_C(  -501.55), EASYSIMD_FLOAT64_C(   134.12), EASYSIMD_FLOAT64_C(   457.05), EASYSIMD_FLOAT64_C(  -756.64) },
      { EASYSIMD_FLOAT64_C(  -514.13), EASYSIMD_FLOAT64_C(   476.28), EASYSIMD_FLOAT64_C(   563.43), EASYSIMD_FLOAT64_C(   352.92),
        EASYSIMD_FLOAT64_C(  -664.49), EASYSIMD_FLOAT64_C(    81.08), EASYSIMD_FLOAT64_C(  -664.49), EASYSIMD_FLOAT64_C(    81.08) } },
    { { EASYSIMD_FLOAT64_C(  -556.38), EASYSIMD_FLOAT64_C(  -801.12), EASYSIMD_FLOAT64_C(   -37.21), EASYSIMD_FLOAT64_C(   708.98),
        EASYSIMD_FLOAT64_C(   510.14), EASYSIMD_FLOAT64_C(  -557.21), EASYSIMD_FLOAT64_C(   880.68), EASYSIMD_FLOAT64_C(  -422.13) },
      { EASYSIMD_FLOAT64_C(  -257.69), EASYSIMD_FLOAT64_C(   779.79), EASYSIMD_FLOAT64_C(  -543.55), EASYSIMD_FLOAT64_C(  -721.39),
        EASYSIMD_FLOAT64_C(  -242.21), EASYSIMD_FLOAT64_C(   -59.62), EASYSIMD_FLOAT64_C(  -781.42), EASYSIMD_FLOAT64_C(   236.86) },
      { EASYSIMD_FLOAT64_C(   510.14), EASYSIMD_FLOAT64_C(  -557.21), EASYSIMD_FLOAT64_C(   880.68), EASYSIMD_FLOAT64_C(  -422.13),
        EASYSIMD_FLOAT64_C(  -257.69), EASYSIMD_FLOAT64_C(   779.79), EASYSIMD_FLOAT64_C(  -543.55), EASYSIMD_FLOAT64_C(  -721.39) },
      { EASYSIMD_FLOAT64_C(   -37.21), EASYSIMD_FLOAT64_C(   708.98), EASYSIMD_FLOAT64_C(   510.14), EASYSIMD_FLOAT64_C(  -557.21),
        EASYSIMD_FLOAT64_C(  -781.42), EASYSIMD_FLOAT64_C(   236.86), EASYSIMD_FLOAT64_C(  -257.69), EASYSIMD_FLOAT64_C(   779.79) },
      { EASYSIMD_FLOAT64_C(   510.14), EASYSIMD_FLOAT64_C(  -557.21), EASYSIMD_FLOAT64_C(   510.14), EASYSIMD_FLOAT64_C(  -557.21),
        EASYSIMD_FLOAT64_C(  -543.55), EASYSIMD_FLOAT64_C(  -721.39), EASYSIMD_FLOAT64_C(  -781.42), EASYSIMD_FLOAT64_C(   236.86) },
      { EASYSIMD_FLOAT64_C(   880.68), EASYSIMD_FLOAT64_C(  -422.13), EASYSIMD_FLOAT64_C(   -37.21), EASYSIMD_FLOAT64_C(   708.98),
        EASYSIMD_FLOAT64_C(  -242.21), EASYSIMD_FLOAT64_C(   -59.62), EASYSIMD_FLOAT64_C(  -242.21), EASYSIMD_FLOAT64_C(   -59.62) } },
    { { EASYSIMD_FLOAT64_C(  -483.17), EASYSIMD_FLOAT64_C(   782.01), EASYSIMD_FLOAT64_C(  -410.22), EASYSIMD_FLOAT64_C(   205.14),
        EASYSIMD_FLOAT64_C(  -937.50), EASYSIMD_FLOAT64_C(    75.64), EASYSIMD_FLOAT64_C(  -318.58), EASYSIMD_FLOAT64_C(  -141.47) },
      { EASYSIMD_FLOAT64_C(   561.70), EASYSIMD_FLOAT64_C(   179.87), EASYSIMD_FLOAT64_C(   992.65), EASYSIMD_FLOAT64_C(   897.21),
        EASYSIMD_FLOAT64_C(  -739.05), EASYSIMD_FLOAT64_C(   449.70), EASYSIMD_FLOAT64_C(  -859.43), EASYSIMD_FLOAT64_C(  -295.43) },
      { EASYSIMD_FLOAT64_C(  -937.50), EASYSIMD_FLOAT64_C(    75.64), EASYSIMD_FLOAT64_C(  -318.58), EASYSIMD_FLOAT64_C(  -141.47),
        EASYSIMD_FLOAT64_C(   561.70), EASYSIMD_FLOAT64_C(   179.87), EASYSIMD_FLOAT64_C(   992.65), EASYSIMD_FLOAT64_C(   897.21) },
      { EASYSIMD_FLOAT64_C(  -410.22), EASYSIMD_FLOAT64_C(   205.14), EASYSIMD_FLOAT64_C(  -937.50), EASYSIMD_FLOAT64_C(    75.64),
        EASYSIMD_FLOAT64_C(  -859.43), EASYSIMD_FLOAT64_C(  -295.43), EASYSIMD_FLOAT64_C(   561.70), EASYSIMD_FLOAT64_C(   179.87) },
      { EASYSIMD_FLOAT64_C(  -937.50), EASYSIMD_FLOAT64_C(    75.64), EASYSIMD_FLOAT64_C(  -937.50), EASYSIMD_FLOAT64_C(    75.64),
        EASYSIMD_FLOAT64_C(   992.65), EASYSIMD_FLOAT64_C(   897.21), EASYSIMD_FLOAT64_C(  -859.43), EASYSIMD_FLOAT64_C(  -295.43) },
      { EASYSIMD_FLOAT64_C(  -318.58), EASYSIMD_FLOAT64_C(  -141.47), EASYSIMD_FLOAT64_C(  -410.22), EASYSIMD_FLOAT64_C(   205.14),
        EASYSIMD_FLOAT64_C(  -739.05), EASYSIMD_FLOAT64_C(   449.70), EASYSIMD_FLOAT64_C(  -739.05), EASYSIMD_FLOAT64_C(   449.70) } },
    { { EASYSIMD_FLOAT64_C(   648.58), EASYSIMD_FLOAT64_C(   103.36), EASYSIMD_FLOAT64_C(  -586.45), EASYSIMD_FLOAT64_C(   158.71),
        EASYSIMD_FLOAT64_C(   546.15), EASYSIMD_FLOAT64_C(  -705.77), EASYSIMD_FLOAT64_C(   736.58), EASYSIMD_FLOAT64_C(  -711.55) },
      { EASYSIMD_FLOAT64_C(  -925.98), EASYSIMD_FLOAT64_C(  -806.97), EASYSIMD_FLOAT64_C(  -432.94), EASYSIMD_FLOAT64_C(  -168.19),
        EASYSIMD_FLOAT64_C(   133.41), EASYSIMD_FLOAT64_C(  -214.36), EASYSIMD_FLOAT64_C(  -931.33), EASYSIMD_FLOAT64_C(   650.24) },
      { EASYSIMD_FLOAT64_C(   546.15), EASYSIMD_FLOAT64_C(  -705.77), EASYSIMD_FLOAT64_C(   736.58), EASYSIMD_FLOAT64_C(  -711.55),
        EASYSIMD_FLOAT64_C(  -925.98), EASYSIMD_FLOAT64_C(  -806.97), EASYSIMD_FLOAT64_C(  -432.94), EASYSIMD_FLOAT64_C(  -168.19) },
      { EASYSIMD_FLOAT64_C(  -586.45), EASYSIMD_FLOAT64_C(   158.71), EASYSIMD_FLOAT64_C(   546.15), EASYSIMD_FLOAT64_C(  -705.77),
        EASYSIMD_FLOAT64_C(  -931.33), EASYSIMD_FLOAT64_C(   650.24), EASYSIMD_FLOAT64_C(  -925.98), EASYSIMD_FLOAT64_C(  -806.97) },
      { EASYSIMD_FLOAT64_C(   546.15), EASYSIMD_FLOAT64_C(  -705.77), EASYSIMD_FLOAT64_C(   546.15), EASYSIMD_FLOAT64_C(  -705.77),
        EASYSIMD_FLOAT64_C(  -432.94), EASYSIMD_FLOAT64_C(  -168.19), EASYSIMD_FLOAT64_C(  -931.33), EASYSIMD_FLOAT64_C(   650.24) },
      { EASYSIMD_FLOAT64_C(   736.58), EASYSIMD_FLOAT64_C(  -711.55), EASYSIMD_FLOAT64_C(  -586.45), EASYSIMD_FLOAT64_C(   158.71),
        EASYSIMD_FLOAT64_C(   133.41), EASYSIMD_FLOAT64_C(  -214.36), EASYSIMD_FLOAT64_C(   133.41), EASYSIMD_FLOAT64_C(  -214.36) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r;

    r = easysimd_mm512_shuffle_f64x2(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r1), 1);
    r = easysimd_mm512_shuffle_f64x2(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r2), 1);
    r = easysimd_mm512_shuffle_f64x2(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r3), 1);
    r = easysimd_mm512_shuffle_f64x2(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r4), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r;

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f64x2(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f64x2(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f64x2(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_f64x2(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_shuffle_f64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float64 src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r1[8];
    const easysimd_float64 r2[8];
    const easysimd_float64 r3[8];
    const easysimd_float64 r4[8];
  } test_vec[4] = {
    { { EASYSIMD_FLOAT64_C(  -452.32), EASYSIMD_FLOAT64_C(  -238.42), EASYSIMD_FLOAT64_C(   874.88), EASYSIMD_FLOAT64_C(   875.56),
        EASYSIMD_FLOAT64_C(    19.67), EASYSIMD_FLOAT64_C(   -26.49), EASYSIMD_FLOAT64_C(   -41.91), EASYSIMD_FLOAT64_C(   532.26) },
      UINT8_C(187),
      { EASYSIMD_FLOAT64_C(  -670.56), EASYSIMD_FLOAT64_C(    78.82), EASYSIMD_FLOAT64_C(  -438.00), EASYSIMD_FLOAT64_C(  -112.77),
        EASYSIMD_FLOAT64_C(   -95.70), EASYSIMD_FLOAT64_C(   217.05), EASYSIMD_FLOAT64_C(  -809.66), EASYSIMD_FLOAT64_C(  -527.36) },
      { EASYSIMD_FLOAT64_C(   -21.99), EASYSIMD_FLOAT64_C(  -162.55), EASYSIMD_FLOAT64_C(   960.43), EASYSIMD_FLOAT64_C(   988.41),
        EASYSIMD_FLOAT64_C(   662.95), EASYSIMD_FLOAT64_C(   852.97), EASYSIMD_FLOAT64_C(  -758.86), EASYSIMD_FLOAT64_C(   340.18) },
      { EASYSIMD_FLOAT64_C(   -95.70), EASYSIMD_FLOAT64_C(   217.05), EASYSIMD_FLOAT64_C(   874.88), EASYSIMD_FLOAT64_C(  -527.36),
        EASYSIMD_FLOAT64_C(   -21.99), EASYSIMD_FLOAT64_C(  -162.55), EASYSIMD_FLOAT64_C(   -41.91), EASYSIMD_FLOAT64_C(   988.41) },
      { EASYSIMD_FLOAT64_C(  -438.00), EASYSIMD_FLOAT64_C(  -112.77), EASYSIMD_FLOAT64_C(   874.88), EASYSIMD_FLOAT64_C(   217.05),
        EASYSIMD_FLOAT64_C(  -758.86), EASYSIMD_FLOAT64_C(   340.18), EASYSIMD_FLOAT64_C(   -41.91), EASYSIMD_FLOAT64_C(  -162.55) },
      { EASYSIMD_FLOAT64_C(   -95.70), EASYSIMD_FLOAT64_C(   217.05), EASYSIMD_FLOAT64_C(   874.88), EASYSIMD_FLOAT64_C(   217.05),
        EASYSIMD_FLOAT64_C(   960.43), EASYSIMD_FLOAT64_C(   988.41), EASYSIMD_FLOAT64_C(   -41.91), EASYSIMD_FLOAT64_C(   340.18) },
      { EASYSIMD_FLOAT64_C(  -809.66), EASYSIMD_FLOAT64_C(  -527.36), EASYSIMD_FLOAT64_C(   874.88), EASYSIMD_FLOAT64_C(  -112.77),
        EASYSIMD_FLOAT64_C(   662.95), EASYSIMD_FLOAT64_C(   852.97), EASYSIMD_FLOAT64_C(   -41.91), EASYSIMD_FLOAT64_C(   852.97) } },
    { { EASYSIMD_FLOAT64_C(   401.34), EASYSIMD_FLOAT64_C(  -871.43), EASYSIMD_FLOAT64_C(    12.13), EASYSIMD_FLOAT64_C(   459.55),
        EASYSIMD_FLOAT64_C(  -800.76), EASYSIMD_FLOAT64_C(   805.68), EASYSIMD_FLOAT64_C(  -992.78), EASYSIMD_FLOAT64_C(   -39.18) },
      UINT8_C(162),
      { EASYSIMD_FLOAT64_C(   882.78), EASYSIMD_FLOAT64_C(   980.49), EASYSIMD_FLOAT64_C(  -345.93), EASYSIMD_FLOAT64_C(  -159.12),
        EASYSIMD_FLOAT64_C(   512.75), EASYSIMD_FLOAT64_C(  -484.64), EASYSIMD_FLOAT64_C(   170.32), EASYSIMD_FLOAT64_C(  -408.43) },
      { EASYSIMD_FLOAT64_C(    77.36), EASYSIMD_FLOAT64_C(  -942.46), EASYSIMD_FLOAT64_C(   495.87), EASYSIMD_FLOAT64_C(  -705.58),
        EASYSIMD_FLOAT64_C(  -752.11), EASYSIMD_FLOAT64_C(   968.51), EASYSIMD_FLOAT64_C(   272.43), EASYSIMD_FLOAT64_C(    85.33) },
      { EASYSIMD_FLOAT64_C(   401.34), EASYSIMD_FLOAT64_C(  -484.64), EASYSIMD_FLOAT64_C(    12.13), EASYSIMD_FLOAT64_C(   459.55),
        EASYSIMD_FLOAT64_C(  -800.76), EASYSIMD_FLOAT64_C(  -942.46), EASYSIMD_FLOAT64_C(  -992.78), EASYSIMD_FLOAT64_C(  -705.58) },
      { EASYSIMD_FLOAT64_C(   401.34), EASYSIMD_FLOAT64_C(  -159.12), EASYSIMD_FLOAT64_C(    12.13), EASYSIMD_FLOAT64_C(   459.55),
        EASYSIMD_FLOAT64_C(  -800.76), EASYSIMD_FLOAT64_C(    85.33), EASYSIMD_FLOAT64_C(  -992.78), EASYSIMD_FLOAT64_C(  -942.46) },
      { EASYSIMD_FLOAT64_C(   401.34), EASYSIMD_FLOAT64_C(  -484.64), EASYSIMD_FLOAT64_C(    12.13), EASYSIMD_FLOAT64_C(   459.55),
        EASYSIMD_FLOAT64_C(  -800.76), EASYSIMD_FLOAT64_C(  -705.58), EASYSIMD_FLOAT64_C(  -992.78), EASYSIMD_FLOAT64_C(    85.33) },
      { EASYSIMD_FLOAT64_C(   401.34), EASYSIMD_FLOAT64_C(  -408.43), EASYSIMD_FLOAT64_C(    12.13), EASYSIMD_FLOAT64_C(   459.55),
        EASYSIMD_FLOAT64_C(  -800.76), EASYSIMD_FLOAT64_C(   968.51), EASYSIMD_FLOAT64_C(  -992.78), EASYSIMD_FLOAT64_C(   968.51) } },
    { { EASYSIMD_FLOAT64_C(   928.94), EASYSIMD_FLOAT64_C(   260.83), EASYSIMD_FLOAT64_C(  -251.71), EASYSIMD_FLOAT64_C(   781.91),
        EASYSIMD_FLOAT64_C(   501.97), EASYSIMD_FLOAT64_C(  -911.54), EASYSIMD_FLOAT64_C(   183.25), EASYSIMD_FLOAT64_C(   630.54) },
      UINT8_C( 36),
      { EASYSIMD_FLOAT64_C(  -357.21), EASYSIMD_FLOAT64_C(   829.78), EASYSIMD_FLOAT64_C(   -93.73), EASYSIMD_FLOAT64_C(  -349.98),
        EASYSIMD_FLOAT64_C(  -209.40), EASYSIMD_FLOAT64_C(  -413.17), EASYSIMD_FLOAT64_C(  -467.20), EASYSIMD_FLOAT64_C(  -228.91) },
      { EASYSIMD_FLOAT64_C(   240.90), EASYSIMD_FLOAT64_C(   373.68), EASYSIMD_FLOAT64_C(  -716.16), EASYSIMD_FLOAT64_C(   756.26),
        EASYSIMD_FLOAT64_C(  -456.01), EASYSIMD_FLOAT64_C(  -124.59), EASYSIMD_FLOAT64_C(  -166.37), EASYSIMD_FLOAT64_C(  -398.47) },
      { EASYSIMD_FLOAT64_C(   928.94), EASYSIMD_FLOAT64_C(   260.83), EASYSIMD_FLOAT64_C(  -467.20), EASYSIMD_FLOAT64_C(   781.91),
        EASYSIMD_FLOAT64_C(   501.97), EASYSIMD_FLOAT64_C(   373.68), EASYSIMD_FLOAT64_C(   183.25), EASYSIMD_FLOAT64_C(   630.54) },
      { EASYSIMD_FLOAT64_C(   928.94), EASYSIMD_FLOAT64_C(   260.83), EASYSIMD_FLOAT64_C(  -209.40), EASYSIMD_FLOAT64_C(   781.91),
        EASYSIMD_FLOAT64_C(   501.97), EASYSIMD_FLOAT64_C(  -398.47), EASYSIMD_FLOAT64_C(   183.25), EASYSIMD_FLOAT64_C(   630.54) },
      { EASYSIMD_FLOAT64_C(   928.94), EASYSIMD_FLOAT64_C(   260.83), EASYSIMD_FLOAT64_C(  -209.40), EASYSIMD_FLOAT64_C(   781.91),
        EASYSIMD_FLOAT64_C(   501.97), EASYSIMD_FLOAT64_C(   756.26), EASYSIMD_FLOAT64_C(   183.25), EASYSIMD_FLOAT64_C(   630.54) },
      { EASYSIMD_FLOAT64_C(   928.94), EASYSIMD_FLOAT64_C(   260.83), EASYSIMD_FLOAT64_C(   -93.73), EASYSIMD_FLOAT64_C(   781.91),
        EASYSIMD_FLOAT64_C(   501.97), EASYSIMD_FLOAT64_C(  -124.59), EASYSIMD_FLOAT64_C(   183.25), EASYSIMD_FLOAT64_C(   630.54) } },
    { { EASYSIMD_FLOAT64_C(  -628.72), EASYSIMD_FLOAT64_C(   128.04), EASYSIMD_FLOAT64_C(  -150.58), EASYSIMD_FLOAT64_C(  -660.21),
        EASYSIMD_FLOAT64_C(  -599.53), EASYSIMD_FLOAT64_C(   934.76), EASYSIMD_FLOAT64_C(  -731.27), EASYSIMD_FLOAT64_C(   661.30) },
      UINT8_C(192),
      { EASYSIMD_FLOAT64_C(  -949.36), EASYSIMD_FLOAT64_C(   163.28), EASYSIMD_FLOAT64_C(  -228.49), EASYSIMD_FLOAT64_C(   233.89),
        EASYSIMD_FLOAT64_C(  -206.18), EASYSIMD_FLOAT64_C(   872.11), EASYSIMD_FLOAT64_C(   876.68), EASYSIMD_FLOAT64_C(  -376.40) },
      { EASYSIMD_FLOAT64_C(  -221.62), EASYSIMD_FLOAT64_C(  -473.31), EASYSIMD_FLOAT64_C(   414.20), EASYSIMD_FLOAT64_C(   365.21),
        EASYSIMD_FLOAT64_C(    59.49), EASYSIMD_FLOAT64_C(  -814.71), EASYSIMD_FLOAT64_C(  -393.89), EASYSIMD_FLOAT64_C(  -566.83) },
      { EASYSIMD_FLOAT64_C(  -628.72), EASYSIMD_FLOAT64_C(   128.04), EASYSIMD_FLOAT64_C(  -150.58), EASYSIMD_FLOAT64_C(  -660.21),
        EASYSIMD_FLOAT64_C(  -599.53), EASYSIMD_FLOAT64_C(   934.76), EASYSIMD_FLOAT64_C(   414.20), EASYSIMD_FLOAT64_C(   365.21) },
      { EASYSIMD_FLOAT64_C(  -628.72), EASYSIMD_FLOAT64_C(   128.04), EASYSIMD_FLOAT64_C(  -150.58), EASYSIMD_FLOAT64_C(  -660.21),
        EASYSIMD_FLOAT64_C(  -599.53), EASYSIMD_FLOAT64_C(   934.76), EASYSIMD_FLOAT64_C(  -221.62), EASYSIMD_FLOAT64_C(  -473.31) },
      { EASYSIMD_FLOAT64_C(  -628.72), EASYSIMD_FLOAT64_C(   128.04), EASYSIMD_FLOAT64_C(  -150.58), EASYSIMD_FLOAT64_C(  -660.21),
        EASYSIMD_FLOAT64_C(  -599.53), EASYSIMD_FLOAT64_C(   934.76), EASYSIMD_FLOAT64_C(  -393.89), EASYSIMD_FLOAT64_C(  -566.83) },
      { EASYSIMD_FLOAT64_C(  -628.72), EASYSIMD_FLOAT64_C(   128.04), EASYSIMD_FLOAT64_C(  -150.58), EASYSIMD_FLOAT64_C(  -660.21),
        EASYSIMD_FLOAT64_C(  -599.53), EASYSIMD_FLOAT64_C(   934.76), EASYSIMD_FLOAT64_C(    59.49), EASYSIMD_FLOAT64_C(  -814.71) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d src = easysimd_mm512_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r;

    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r1), 1);
    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r2), 1);
    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r3), 1);
    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r4), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512d src = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r;

    easysimd_test_x86_write_f64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_f64x2(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_shuffle_f64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r1[8];
    const easysimd_float64 r2[8];
    const easysimd_float64 r3[8];
    const easysimd_float64 r4[8];
  } test_vec[4] = {
    { UINT8_C( 76),
      { EASYSIMD_FLOAT64_C(  -379.23), EASYSIMD_FLOAT64_C(   115.72), EASYSIMD_FLOAT64_C(    73.21), EASYSIMD_FLOAT64_C(   368.89),
        EASYSIMD_FLOAT64_C(  -957.25), EASYSIMD_FLOAT64_C(   678.78), EASYSIMD_FLOAT64_C(  -404.68), EASYSIMD_FLOAT64_C(  -949.07) },
      { EASYSIMD_FLOAT64_C(    93.24), EASYSIMD_FLOAT64_C(   662.55), EASYSIMD_FLOAT64_C(   397.06), EASYSIMD_FLOAT64_C(  -397.64),
        EASYSIMD_FLOAT64_C(    60.46), EASYSIMD_FLOAT64_C(   418.60), EASYSIMD_FLOAT64_C(  -942.59), EASYSIMD_FLOAT64_C(  -459.80) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -404.68), EASYSIMD_FLOAT64_C(  -949.07),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   397.06), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -957.25), EASYSIMD_FLOAT64_C(   678.78),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    93.24), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -957.25), EASYSIMD_FLOAT64_C(   678.78),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -942.59), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    73.21), EASYSIMD_FLOAT64_C(   368.89),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    60.46), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(214),
      { EASYSIMD_FLOAT64_C(  -399.74), EASYSIMD_FLOAT64_C(   463.35), EASYSIMD_FLOAT64_C(  -417.58), EASYSIMD_FLOAT64_C(   305.42),
        EASYSIMD_FLOAT64_C(   556.46), EASYSIMD_FLOAT64_C(   407.97), EASYSIMD_FLOAT64_C(   544.91), EASYSIMD_FLOAT64_C(   643.69) },
      { EASYSIMD_FLOAT64_C(   775.59), EASYSIMD_FLOAT64_C(  -852.73), EASYSIMD_FLOAT64_C(   451.69), EASYSIMD_FLOAT64_C(   -86.20),
        EASYSIMD_FLOAT64_C(   719.16), EASYSIMD_FLOAT64_C(  -340.00), EASYSIMD_FLOAT64_C(   534.57), EASYSIMD_FLOAT64_C(  -165.13) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   407.97), EASYSIMD_FLOAT64_C(   544.91), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   775.59), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   451.69), EASYSIMD_FLOAT64_C(   -86.20) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   305.42), EASYSIMD_FLOAT64_C(   556.46), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   534.57), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   775.59), EASYSIMD_FLOAT64_C(  -852.73) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   407.97), EASYSIMD_FLOAT64_C(   556.46), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   451.69), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   534.57), EASYSIMD_FLOAT64_C(  -165.13) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   643.69), EASYSIMD_FLOAT64_C(  -417.58), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   719.16), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   719.16), EASYSIMD_FLOAT64_C(  -340.00) } },
    { UINT8_C(161),
      { EASYSIMD_FLOAT64_C(   -96.54), EASYSIMD_FLOAT64_C(  -122.37), EASYSIMD_FLOAT64_C(   411.99), EASYSIMD_FLOAT64_C(   498.78),
        EASYSIMD_FLOAT64_C(   -71.44), EASYSIMD_FLOAT64_C(  -494.78), EASYSIMD_FLOAT64_C(   161.33), EASYSIMD_FLOAT64_C(  -674.38) },
      { EASYSIMD_FLOAT64_C(   107.58), EASYSIMD_FLOAT64_C(  -778.21), EASYSIMD_FLOAT64_C(   744.22), EASYSIMD_FLOAT64_C(   164.99),
        EASYSIMD_FLOAT64_C(  -238.01), EASYSIMD_FLOAT64_C(  -673.11), EASYSIMD_FLOAT64_C(   765.25), EASYSIMD_FLOAT64_C(  -774.66) },
      { EASYSIMD_FLOAT64_C(   -71.44), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -778.21), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   164.99) },
      { EASYSIMD_FLOAT64_C(   411.99), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -774.66), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -778.21) },
      { EASYSIMD_FLOAT64_C(   -71.44), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   164.99), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -774.66) },
      { EASYSIMD_FLOAT64_C(   161.33), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -673.11), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -673.11) } },
    { UINT8_C( 92),
      { EASYSIMD_FLOAT64_C(    70.66), EASYSIMD_FLOAT64_C(   781.80), EASYSIMD_FLOAT64_C(  -682.72), EASYSIMD_FLOAT64_C(  -384.43),
        EASYSIMD_FLOAT64_C(   425.49), EASYSIMD_FLOAT64_C(  -907.13), EASYSIMD_FLOAT64_C(  -237.16), EASYSIMD_FLOAT64_C(  -122.82) },
      { EASYSIMD_FLOAT64_C(     6.67), EASYSIMD_FLOAT64_C(  -518.01), EASYSIMD_FLOAT64_C(   537.18), EASYSIMD_FLOAT64_C(  -458.76),
        EASYSIMD_FLOAT64_C(   316.87), EASYSIMD_FLOAT64_C(   270.38), EASYSIMD_FLOAT64_C(   444.70), EASYSIMD_FLOAT64_C(  -805.51) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -237.16), EASYSIMD_FLOAT64_C(  -122.82),
        EASYSIMD_FLOAT64_C(     6.67), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   537.18), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   425.49), EASYSIMD_FLOAT64_C(  -907.13),
        EASYSIMD_FLOAT64_C(   444.70), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     6.67), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   425.49), EASYSIMD_FLOAT64_C(  -907.13),
        EASYSIMD_FLOAT64_C(   537.18), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   444.70), EASYSIMD_FLOAT64_C(     0.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -682.72), EASYSIMD_FLOAT64_C(  -384.43),
        EASYSIMD_FLOAT64_C(   316.87), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   316.87), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r;

    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r1), 1);
    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r2), 1);
    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r3), 1);
    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r4), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r;

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_f64x2(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shuffle_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r1[16];
    const int32_t r2[16];
    const int32_t r3[16];
    const int32_t r4[16];
  } test_vec[4] = {
    { {  INT32_C(  1991146578),  INT32_C(  1899045466), -INT32_C(  1068581367), -INT32_C(   826279770),  INT32_C(   117359005),  INT32_C(  1832948451),  INT32_C(   516563938), -INT32_C(  2119307474),
        -INT32_C(   973644949), -INT32_C(  2026428290), -INT32_C(  1924626969),  INT32_C(   458950525), -INT32_C(  1407100472), -INT32_C(   853974293), -INT32_C(  1259543675),  INT32_C(  1043700178) },
      {  INT32_C(  1929587957),  INT32_C(  1023031893),  INT32_C(  1036600000),  INT32_C(   307766346),  INT32_C(  1757313661),  INT32_C(  1630984156), -INT32_C(  1927994694), -INT32_C(  1328854341),
        -INT32_C(   870068617), -INT32_C(   922214903), -INT32_C(  1442393760),  INT32_C(  1925013493), -INT32_C(  1243972647),  INT32_C(   219549779), -INT32_C(   291886030), -INT32_C(   308386442) },
      { -INT32_C(   973644949), -INT32_C(  2026428290), -INT32_C(  1924626969),  INT32_C(   458950525), -INT32_C(  1407100472), -INT32_C(   853974293), -INT32_C(  1259543675),  INT32_C(  1043700178),
         INT32_C(  1929587957),  INT32_C(  1023031893),  INT32_C(  1036600000),  INT32_C(   307766346),  INT32_C(  1757313661),  INT32_C(  1630984156), -INT32_C(  1927994694), -INT32_C(  1328854341) },
      {  INT32_C(   117359005),  INT32_C(  1832948451),  INT32_C(   516563938), -INT32_C(  2119307474), -INT32_C(   973644949), -INT32_C(  2026428290), -INT32_C(  1924626969),  INT32_C(   458950525),
        -INT32_C(  1243972647),  INT32_C(   219549779), -INT32_C(   291886030), -INT32_C(   308386442),  INT32_C(  1929587957),  INT32_C(  1023031893),  INT32_C(  1036600000),  INT32_C(   307766346) },
      { -INT32_C(   973644949), -INT32_C(  2026428290), -INT32_C(  1924626969),  INT32_C(   458950525), -INT32_C(   973644949), -INT32_C(  2026428290), -INT32_C(  1924626969),  INT32_C(   458950525),
         INT32_C(  1757313661),  INT32_C(  1630984156), -INT32_C(  1927994694), -INT32_C(  1328854341), -INT32_C(  1243972647),  INT32_C(   219549779), -INT32_C(   291886030), -INT32_C(   308386442) },
      { -INT32_C(  1407100472), -INT32_C(   853974293), -INT32_C(  1259543675),  INT32_C(  1043700178),  INT32_C(   117359005),  INT32_C(  1832948451),  INT32_C(   516563938), -INT32_C(  2119307474),
        -INT32_C(   870068617), -INT32_C(   922214903), -INT32_C(  1442393760),  INT32_C(  1925013493), -INT32_C(   870068617), -INT32_C(   922214903), -INT32_C(  1442393760),  INT32_C(  1925013493) } },
    { {  INT32_C(  1035584052),  INT32_C(  1074184672), -INT32_C(  2014638958),  INT32_C(  1157212267),  INT32_C(  1996084259),  INT32_C(   394531044), -INT32_C(  1308287428), -INT32_C(  1197431932),
         INT32_C(  1173707109), -INT32_C(  1383662821),  INT32_C(  1932882183),  INT32_C(  1018637849), -INT32_C(   407654142), -INT32_C(    33671231), -INT32_C(   642841771),  INT32_C(   210849702) },
      { -INT32_C(  1001290071), -INT32_C(  1989027967),  INT32_C(  1643947592), -INT32_C(   677465131),  INT32_C(   633229668), -INT32_C(   584926072),  INT32_C(  1723323071), -INT32_C(   898480095),
         INT32_C(  1351533519), -INT32_C(   472317797),  INT32_C(  2068174246), -INT32_C(   313269367), -INT32_C(  1139601100), -INT32_C(  1919339058),  INT32_C(   703811591),  INT32_C(  1760781721) },
      {  INT32_C(  1173707109), -INT32_C(  1383662821),  INT32_C(  1932882183),  INT32_C(  1018637849), -INT32_C(   407654142), -INT32_C(    33671231), -INT32_C(   642841771),  INT32_C(   210849702),
        -INT32_C(  1001290071), -INT32_C(  1989027967),  INT32_C(  1643947592), -INT32_C(   677465131),  INT32_C(   633229668), -INT32_C(   584926072),  INT32_C(  1723323071), -INT32_C(   898480095) },
      {  INT32_C(  1996084259),  INT32_C(   394531044), -INT32_C(  1308287428), -INT32_C(  1197431932),  INT32_C(  1173707109), -INT32_C(  1383662821),  INT32_C(  1932882183),  INT32_C(  1018637849),
        -INT32_C(  1139601100), -INT32_C(  1919339058),  INT32_C(   703811591),  INT32_C(  1760781721), -INT32_C(  1001290071), -INT32_C(  1989027967),  INT32_C(  1643947592), -INT32_C(   677465131) },
      {  INT32_C(  1173707109), -INT32_C(  1383662821),  INT32_C(  1932882183),  INT32_C(  1018637849),  INT32_C(  1173707109), -INT32_C(  1383662821),  INT32_C(  1932882183),  INT32_C(  1018637849),
         INT32_C(   633229668), -INT32_C(   584926072),  INT32_C(  1723323071), -INT32_C(   898480095), -INT32_C(  1139601100), -INT32_C(  1919339058),  INT32_C(   703811591),  INT32_C(  1760781721) },
      { -INT32_C(   407654142), -INT32_C(    33671231), -INT32_C(   642841771),  INT32_C(   210849702),  INT32_C(  1996084259),  INT32_C(   394531044), -INT32_C(  1308287428), -INT32_C(  1197431932),
         INT32_C(  1351533519), -INT32_C(   472317797),  INT32_C(  2068174246), -INT32_C(   313269367),  INT32_C(  1351533519), -INT32_C(   472317797),  INT32_C(  2068174246), -INT32_C(   313269367) } },
    { { -INT32_C(   994541015),  INT32_C(   682070658), -INT32_C(   257627033),  INT32_C(    64944079), -INT32_C(   692063992),  INT32_C(   778328358),  INT32_C(  1113020329), -INT32_C(   425047363),
         INT32_C(  1319789516),  INT32_C(  1551323637),  INT32_C(   223157054),  INT32_C(   437332754),  INT32_C(  1123143708), -INT32_C(   747612887),  INT32_C(  1763035052), -INT32_C(   565198830) },
      {  INT32_C(   405666083), -INT32_C(  1988844469), -INT32_C(   778649153),  INT32_C(   149661676), -INT32_C(  1588863880), -INT32_C(   562775247), -INT32_C(  1790473597),  INT32_C(  1836291914),
        -INT32_C(   611999600),  INT32_C(    56949060), -INT32_C(  1479214149),  INT32_C(   447791266), -INT32_C(   826475620),  INT32_C(   967585974),  INT32_C(    80671674),  INT32_C(   460407434) },
      {  INT32_C(  1319789516),  INT32_C(  1551323637),  INT32_C(   223157054),  INT32_C(   437332754),  INT32_C(  1123143708), -INT32_C(   747612887),  INT32_C(  1763035052), -INT32_C(   565198830),
         INT32_C(   405666083), -INT32_C(  1988844469), -INT32_C(   778649153),  INT32_C(   149661676), -INT32_C(  1588863880), -INT32_C(   562775247), -INT32_C(  1790473597),  INT32_C(  1836291914) },
      { -INT32_C(   692063992),  INT32_C(   778328358),  INT32_C(  1113020329), -INT32_C(   425047363),  INT32_C(  1319789516),  INT32_C(  1551323637),  INT32_C(   223157054),  INT32_C(   437332754),
        -INT32_C(   826475620),  INT32_C(   967585974),  INT32_C(    80671674),  INT32_C(   460407434),  INT32_C(   405666083), -INT32_C(  1988844469), -INT32_C(   778649153),  INT32_C(   149661676) },
      {  INT32_C(  1319789516),  INT32_C(  1551323637),  INT32_C(   223157054),  INT32_C(   437332754),  INT32_C(  1319789516),  INT32_C(  1551323637),  INT32_C(   223157054),  INT32_C(   437332754),
        -INT32_C(  1588863880), -INT32_C(   562775247), -INT32_C(  1790473597),  INT32_C(  1836291914), -INT32_C(   826475620),  INT32_C(   967585974),  INT32_C(    80671674),  INT32_C(   460407434) },
      {  INT32_C(  1123143708), -INT32_C(   747612887),  INT32_C(  1763035052), -INT32_C(   565198830), -INT32_C(   692063992),  INT32_C(   778328358),  INT32_C(  1113020329), -INT32_C(   425047363),
        -INT32_C(   611999600),  INT32_C(    56949060), -INT32_C(  1479214149),  INT32_C(   447791266), -INT32_C(   611999600),  INT32_C(    56949060), -INT32_C(  1479214149),  INT32_C(   447791266) } },
    { {  INT32_C(   670496482), -INT32_C(  1423287312), -INT32_C(   128778410),  INT32_C(  1527972543), -INT32_C(  1272328195), -INT32_C(  1158818305),  INT32_C(  1405009097), -INT32_C(   529649410),
         INT32_C(   369583398),  INT32_C(   381760192), -INT32_C(   267512783),  INT32_C(   340467990), -INT32_C(   255298064),  INT32_C(   329954634),  INT32_C(  1869047921), -INT32_C(  1085221480) },
      { -INT32_C(    86681798), -INT32_C(  1173317751), -INT32_C(  1045815381),  INT32_C(   836105792), -INT32_C(  1256088213), -INT32_C(   993408173), -INT32_C(   868995021),  INT32_C(  1082885125),
         INT32_C(  1698324699), -INT32_C(  1575007241), -INT32_C(  1436235158),  INT32_C(   735787456),  INT32_C(   719387863), -INT32_C(    68245049), -INT32_C(   540597543), -INT32_C(  2111876442) },
      {  INT32_C(   369583398),  INT32_C(   381760192), -INT32_C(   267512783),  INT32_C(   340467990), -INT32_C(   255298064),  INT32_C(   329954634),  INT32_C(  1869047921), -INT32_C(  1085221480),
        -INT32_C(    86681798), -INT32_C(  1173317751), -INT32_C(  1045815381),  INT32_C(   836105792), -INT32_C(  1256088213), -INT32_C(   993408173), -INT32_C(   868995021),  INT32_C(  1082885125) },
      { -INT32_C(  1272328195), -INT32_C(  1158818305),  INT32_C(  1405009097), -INT32_C(   529649410),  INT32_C(   369583398),  INT32_C(   381760192), -INT32_C(   267512783),  INT32_C(   340467990),
         INT32_C(   719387863), -INT32_C(    68245049), -INT32_C(   540597543), -INT32_C(  2111876442), -INT32_C(    86681798), -INT32_C(  1173317751), -INT32_C(  1045815381),  INT32_C(   836105792) },
      {  INT32_C(   369583398),  INT32_C(   381760192), -INT32_C(   267512783),  INT32_C(   340467990),  INT32_C(   369583398),  INT32_C(   381760192), -INT32_C(   267512783),  INT32_C(   340467990),
        -INT32_C(  1256088213), -INT32_C(   993408173), -INT32_C(   868995021),  INT32_C(  1082885125),  INT32_C(   719387863), -INT32_C(    68245049), -INT32_C(   540597543), -INT32_C(  2111876442) },
      { -INT32_C(   255298064),  INT32_C(   329954634),  INT32_C(  1869047921), -INT32_C(  1085221480), -INT32_C(  1272328195), -INT32_C(  1158818305),  INT32_C(  1405009097), -INT32_C(   529649410),
         INT32_C(  1698324699), -INT32_C(  1575007241), -INT32_C(  1436235158),  INT32_C(   735787456),  INT32_C(  1698324699), -INT32_C(  1575007241), -INT32_C(  1436235158),  INT32_C(   735787456) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;

    r = easysimd_mm512_shuffle_i32x4(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r1));
    r = easysimd_mm512_shuffle_i32x4(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r2));
    r = easysimd_mm512_shuffle_i32x4(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r3));

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_shuffle_i32x4(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_shuffle_i32x4");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r4));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r;

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i32x4(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i32x4(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i32x4(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i32x4(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_shuffle_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r1[16];
    const int32_t r2[16];
    const int32_t r3[16];
    const int32_t r4[16];
  } test_vec[4] = {
    { { -INT32_C(  1024729526), -INT32_C(   210388682), -INT32_C(    73862869), -INT32_C(   260015396), -INT32_C(  1919425577), -INT32_C(   420651029), -INT32_C(  1844649463), -INT32_C(  1214752147),
         INT32_C(  1853457208),  INT32_C(  1751314236), -INT32_C(  1117521184),  INT32_C(  1286464373), -INT32_C(  1311160890), -INT32_C(  1382561884),  INT32_C(   373269673),  INT32_C(   919459838) },
      UINT16_C(18266),
      {  INT32_C(   104241060),  INT32_C(  1644238591), -INT32_C(  2126154029), -INT32_C(  1681519422),  INT32_C(  1432513213),  INT32_C(  1459161879), -INT32_C(   282134751), -INT32_C(   784955347),
         INT32_C(   517499935),  INT32_C(  1451284866),  INT32_C(   299353679), -INT32_C(  1884512814),  INT32_C(   518262535),  INT32_C(  1014291738),  INT32_C(    19636948),  INT32_C(  1255366955) },
      {  INT32_C(  1332325325), -INT32_C(   744101500), -INT32_C(  2115732305),  INT32_C(   538021913), -INT32_C(  1170279009), -INT32_C(  1477004333), -INT32_C(  2136399531),  INT32_C(  1338735746),
        -INT32_C(  1415695321), -INT32_C(   847363043), -INT32_C(   632397120), -INT32_C(  1829085198),  INT32_C(   676084053),  INT32_C(  1104102124), -INT32_C(   440240029),  INT32_C(   456429044) },
      { -INT32_C(  1024729526),  INT32_C(  1451284866), -INT32_C(    73862869), -INT32_C(  1884512814),  INT32_C(   518262535), -INT32_C(   420651029),  INT32_C(    19636948), -INT32_C(  1214752147),
         INT32_C(  1332325325), -INT32_C(   744101500), -INT32_C(  2115732305),  INT32_C(  1286464373), -INT32_C(  1311160890), -INT32_C(  1382561884), -INT32_C(  2136399531),  INT32_C(   919459838) },
      { -INT32_C(  1024729526),  INT32_C(  1459161879), -INT32_C(    73862869), -INT32_C(   784955347),  INT32_C(   517499935), -INT32_C(   420651029),  INT32_C(   299353679), -INT32_C(  1214752147),
         INT32_C(   676084053),  INT32_C(  1104102124), -INT32_C(   440240029),  INT32_C(  1286464373), -INT32_C(  1311160890), -INT32_C(  1382561884), -INT32_C(  2115732305),  INT32_C(   919459838) },
      { -INT32_C(  1024729526),  INT32_C(  1451284866), -INT32_C(    73862869), -INT32_C(  1884512814),  INT32_C(   517499935), -INT32_C(   420651029),  INT32_C(   299353679), -INT32_C(  1214752147),
        -INT32_C(  1170279009), -INT32_C(  1477004333), -INT32_C(  2136399531),  INT32_C(  1286464373), -INT32_C(  1311160890), -INT32_C(  1382561884), -INT32_C(   440240029),  INT32_C(   919459838) },
      { -INT32_C(  1024729526),  INT32_C(  1014291738), -INT32_C(    73862869),  INT32_C(  1255366955),  INT32_C(  1432513213), -INT32_C(   420651029), -INT32_C(   282134751), -INT32_C(  1214752147),
        -INT32_C(  1415695321), -INT32_C(   847363043), -INT32_C(   632397120),  INT32_C(  1286464373), -INT32_C(  1311160890), -INT32_C(  1382561884), -INT32_C(   632397120),  INT32_C(   919459838) } },
    { { -INT32_C(   557395263), -INT32_C(   676641770), -INT32_C(  1716389210), -INT32_C(  1372869799), -INT32_C(   791250972),  INT32_C(   470984377),  INT32_C(   318886942),  INT32_C(   573453665),
         INT32_C(   503379207), -INT32_C(   520770503), -INT32_C(     8804698),  INT32_C(   917414993), -INT32_C(   737770469),  INT32_C(  1240471594),  INT32_C(  1297936876),  INT32_C(   762284582) },
      UINT16_C(28799),
      {  INT32_C(  1075624267),  INT32_C(   317112985),  INT32_C(  1874213057),  INT32_C(  1962201454), -INT32_C(  1785913691), -INT32_C(  1014597273),  INT32_C(   911125703),  INT32_C(   631688665),
         INT32_C(   526762630),  INT32_C(  1177635972), -INT32_C(   222959740),  INT32_C(  1567009208),  INT32_C(   804451272),  INT32_C(   888371309), -INT32_C(    26590940), -INT32_C(  1792863986) },
      {  INT32_C(  1488226515),  INT32_C(  1486808788), -INT32_C(  2041949235), -INT32_C(   974933507),  INT32_C(   301258404),  INT32_C(  1933961038),  INT32_C(   930197544), -INT32_C(  1798531903),
        -INT32_C(   236158948),  INT32_C(   877234790), -INT32_C(   608529186), -INT32_C(   375349947), -INT32_C(  1023699853), -INT32_C(  1540013956), -INT32_C(  1311004944),  INT32_C(  1447470906) },
      {  INT32_C(   526762630),  INT32_C(  1177635972), -INT32_C(   222959740),  INT32_C(  1567009208),  INT32_C(   804451272),  INT32_C(   888371309), -INT32_C(    26590940),  INT32_C(   573453665),
         INT32_C(   503379207), -INT32_C(   520770503), -INT32_C(     8804698),  INT32_C(   917414993),  INT32_C(   301258404),  INT32_C(  1933961038),  INT32_C(   930197544),  INT32_C(   762284582) },
      { -INT32_C(  1785913691), -INT32_C(  1014597273),  INT32_C(   911125703),  INT32_C(   631688665),  INT32_C(   526762630),  INT32_C(  1177635972), -INT32_C(   222959740),  INT32_C(   573453665),
         INT32_C(   503379207), -INT32_C(   520770503), -INT32_C(     8804698),  INT32_C(   917414993),  INT32_C(  1488226515),  INT32_C(  1486808788), -INT32_C(  2041949235),  INT32_C(   762284582) },
      {  INT32_C(   526762630),  INT32_C(  1177635972), -INT32_C(   222959740),  INT32_C(  1567009208),  INT32_C(   526762630),  INT32_C(  1177635972), -INT32_C(   222959740),  INT32_C(   573453665),
         INT32_C(   503379207), -INT32_C(   520770503), -INT32_C(     8804698),  INT32_C(   917414993), -INT32_C(  1023699853), -INT32_C(  1540013956), -INT32_C(  1311004944),  INT32_C(   762284582) },
      {  INT32_C(   804451272),  INT32_C(   888371309), -INT32_C(    26590940), -INT32_C(  1792863986), -INT32_C(  1785913691), -INT32_C(  1014597273),  INT32_C(   911125703),  INT32_C(   573453665),
         INT32_C(   503379207), -INT32_C(   520770503), -INT32_C(     8804698),  INT32_C(   917414993), -INT32_C(   236158948),  INT32_C(   877234790), -INT32_C(   608529186),  INT32_C(   762284582) } },
    { { -INT32_C(  1907936728), -INT32_C(  1681747523),  INT32_C(  1786149925), -INT32_C(  1923934694),  INT32_C(   642731690),  INT32_C(  2144044175),  INT32_C(  1680975402),  INT32_C(  1991997262),
         INT32_C(  1711538857), -INT32_C(  1207842925),  INT32_C(  1562539843),  INT32_C(   954955405),  INT32_C(  1398684356), -INT32_C(   372037185),  INT32_C(   508429520),  INT32_C(   613681531) },
      UINT16_C(38923),
      { -INT32_C(  1939890293),  INT32_C(  2047124311), -INT32_C(   336555776),  INT32_C(   673625289), -INT32_C(   615389944),  INT32_C(   484385230), -INT32_C(   752526785),  INT32_C(   158085502),
         INT32_C(   664194000),  INT32_C(  1872861806), -INT32_C(   178613973),  INT32_C(  1310556230),  INT32_C(   875130469), -INT32_C(   816838512), -INT32_C(   526158238),  INT32_C(  2011828135) },
      {  INT32_C(  1235124442),  INT32_C(  1169702938),  INT32_C(   389681873), -INT32_C(   127576173),  INT32_C(  1445760710), -INT32_C(   131760746), -INT32_C(  1697003277), -INT32_C(  1307458601),
         INT32_C(  1576775491), -INT32_C(  1046301713),  INT32_C(  1490607557), -INT32_C(    95338956),  INT32_C(  1666219468), -INT32_C(   312772870),  INT32_C(   377959486),  INT32_C(   986224887) },
      {  INT32_C(   664194000),  INT32_C(  1872861806),  INT32_C(  1786149925),  INT32_C(  1310556230),  INT32_C(   642731690),  INT32_C(  2144044175),  INT32_C(  1680975402),  INT32_C(  1991997262),
         INT32_C(  1711538857), -INT32_C(  1207842925),  INT32_C(  1562539843), -INT32_C(   127576173),  INT32_C(  1445760710), -INT32_C(   372037185),  INT32_C(   508429520), -INT32_C(  1307458601) },
      { -INT32_C(   615389944),  INT32_C(   484385230),  INT32_C(  1786149925),  INT32_C(   158085502),  INT32_C(   642731690),  INT32_C(  2144044175),  INT32_C(  1680975402),  INT32_C(  1991997262),
         INT32_C(  1711538857), -INT32_C(  1207842925),  INT32_C(  1562539843),  INT32_C(   986224887),  INT32_C(  1235124442), -INT32_C(   372037185),  INT32_C(   508429520), -INT32_C(   127576173) },
      {  INT32_C(   664194000),  INT32_C(  1872861806),  INT32_C(  1786149925),  INT32_C(  1310556230),  INT32_C(   642731690),  INT32_C(  2144044175),  INT32_C(  1680975402),  INT32_C(  1991997262),
         INT32_C(  1711538857), -INT32_C(  1207842925),  INT32_C(  1562539843), -INT32_C(  1307458601),  INT32_C(  1666219468), -INT32_C(   372037185),  INT32_C(   508429520),  INT32_C(   986224887) },
      {  INT32_C(   875130469), -INT32_C(   816838512),  INT32_C(  1786149925),  INT32_C(  2011828135),  INT32_C(   642731690),  INT32_C(  2144044175),  INT32_C(  1680975402),  INT32_C(  1991997262),
         INT32_C(  1711538857), -INT32_C(  1207842925),  INT32_C(  1562539843), -INT32_C(    95338956),  INT32_C(  1576775491), -INT32_C(   372037185),  INT32_C(   508429520), -INT32_C(    95338956) } },
    { {  INT32_C(   932692808),  INT32_C(  1006123638),  INT32_C(  1268044055), -INT32_C(   616110833),  INT32_C(  1564382818),  INT32_C(  1263180300), -INT32_C(   966667570), -INT32_C(  1308612246),
         INT32_C(  1659541740), -INT32_C(   375528750), -INT32_C(  1036766797),  INT32_C(  2040429078),  INT32_C(   500620305),  INT32_C(  1164451958),  INT32_C(  1561053682), -INT32_C(   569439246) },
      UINT16_C(63907),
      { -INT32_C(   555977408), -INT32_C(  1827696802), -INT32_C(   267573678),  INT32_C(  1976311455), -INT32_C(  1533721796), -INT32_C(  1821472632), -INT32_C(   190947099),  INT32_C(  2146320959),
         INT32_C(   375245495), -INT32_C(  1414959783),  INT32_C(   849065619),  INT32_C(   296184021),  INT32_C(   867515563), -INT32_C(  1446632508), -INT32_C(  1013029756),  INT32_C(  1581419686) },
      { -INT32_C(  1351311274), -INT32_C(  1604641523), -INT32_C(  1462569261),  INT32_C(   163150174),  INT32_C(  2050781109),  INT32_C(   371393170),  INT32_C(   232374631), -INT32_C(  1553261491),
        -INT32_C(   917250116), -INT32_C(   798380292),  INT32_C(    58211236),  INT32_C(  1762407092),  INT32_C(   870533537), -INT32_C(  1303771317),  INT32_C(   381690824), -INT32_C(    71750849) },
      {  INT32_C(   375245495), -INT32_C(  1414959783),  INT32_C(  1268044055), -INT32_C(   616110833),  INT32_C(  1564382818), -INT32_C(  1446632508), -INT32_C(   966667570),  INT32_C(  1581419686),
        -INT32_C(  1351311274), -INT32_C(   375528750), -INT32_C(  1036766797),  INT32_C(   163150174),  INT32_C(  2050781109),  INT32_C(   371393170),  INT32_C(   232374631), -INT32_C(  1553261491) },
      { -INT32_C(  1533721796), -INT32_C(  1821472632),  INT32_C(  1268044055), -INT32_C(   616110833),  INT32_C(  1564382818), -INT32_C(  1414959783), -INT32_C(   966667570),  INT32_C(   296184021),
         INT32_C(   870533537), -INT32_C(   375528750), -INT32_C(  1036766797), -INT32_C(    71750849), -INT32_C(  1351311274), -INT32_C(  1604641523), -INT32_C(  1462569261),  INT32_C(   163150174) },
      {  INT32_C(   375245495), -INT32_C(  1414959783),  INT32_C(  1268044055), -INT32_C(   616110833),  INT32_C(  1564382818), -INT32_C(  1414959783), -INT32_C(   966667570),  INT32_C(   296184021),
         INT32_C(  2050781109), -INT32_C(   375528750), -INT32_C(  1036766797), -INT32_C(  1553261491),  INT32_C(   870533537), -INT32_C(  1303771317),  INT32_C(   381690824), -INT32_C(    71750849) },
      {  INT32_C(   867515563), -INT32_C(  1446632508),  INT32_C(  1268044055), -INT32_C(   616110833),  INT32_C(  1564382818), -INT32_C(  1821472632), -INT32_C(   966667570),  INT32_C(  2146320959),
        -INT32_C(   917250116), -INT32_C(   375528750), -INT32_C(  1036766797),  INT32_C(  1762407092), -INT32_C(   917250116), -INT32_C(   798380292),  INT32_C(    58211236),  INT32_C(  1762407092) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;

    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r1));
    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r2));
    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r3));
    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r4));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r;

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i32x4(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_shuffle_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r1[16];
    const int32_t r2[16];
    const int32_t r3[16];
    const int32_t r4[16];
  } test_vec[4] = {
    { UINT16_C(59334),
      { -INT32_C(  1109512267), -INT32_C(    80650447),  INT32_C(  1002028998), -INT32_C(  1570143339),  INT32_C(  1819608205), -INT32_C(  1683399021), -INT32_C(      525025),  INT32_C(  2011610562),
         INT32_C(   506840556), -INT32_C(   367434204), -INT32_C(  1171926491), -INT32_C(   480473514),  INT32_C(   642765458),  INT32_C(  1338112048), -INT32_C(  1202800394),  INT32_C(  1630549365) },
      {  INT32_C(   511665658), -INT32_C(   251094581), -INT32_C(  1045746069),  INT32_C(  1336150204),  INT32_C(   175502298), -INT32_C(   497404180),  INT32_C(  1671080174), -INT32_C(   674968867),
        -INT32_C(    67812305),  INT32_C(  1223491293), -INT32_C(   401959124),  INT32_C(  2033692319), -INT32_C(  1903907678), -INT32_C(   797909278),  INT32_C(  1681066630),  INT32_C(    71039189) },
      {  INT32_C(           0), -INT32_C(   367434204), -INT32_C(  1171926491),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1202800394),  INT32_C(  1630549365),
         INT32_C(   511665658), -INT32_C(   251094581), -INT32_C(  1045746069),  INT32_C(           0),  INT32_C(           0), -INT32_C(   497404180),  INT32_C(  1671080174), -INT32_C(   674968867) },
      {  INT32_C(           0), -INT32_C(  1683399021), -INT32_C(      525025),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1171926491), -INT32_C(   480473514),
        -INT32_C(  1903907678), -INT32_C(   797909278),  INT32_C(  1681066630),  INT32_C(           0),  INT32_C(           0), -INT32_C(   251094581), -INT32_C(  1045746069),  INT32_C(  1336150204) },
      {  INT32_C(           0), -INT32_C(   367434204), -INT32_C(  1171926491),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1171926491), -INT32_C(   480473514),
         INT32_C(   175502298), -INT32_C(   497404180),  INT32_C(  1671080174),  INT32_C(           0),  INT32_C(           0), -INT32_C(   797909278),  INT32_C(  1681066630),  INT32_C(    71039189) },
      {  INT32_C(           0),  INT32_C(  1338112048), -INT32_C(  1202800394),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(      525025),  INT32_C(  2011610562),
        -INT32_C(    67812305),  INT32_C(  1223491293), -INT32_C(   401959124),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1223491293), -INT32_C(   401959124),  INT32_C(  2033692319) } },
    { UINT16_C(12604),
      { -INT32_C(   349234689),  INT32_C(  1803770721),  INT32_C(  2065310275),  INT32_C(   539475100), -INT32_C(  1157756342),  INT32_C(   247760090),  INT32_C(   604412392), -INT32_C(  1655356770),
        -INT32_C(  1131838373),  INT32_C(   573050079), -INT32_C(   895663570),  INT32_C(  1223345662), -INT32_C(  1459427121),  INT32_C(  1438107245), -INT32_C(    25576096),  INT32_C(  1520160511) },
      {  INT32_C(   823600210),  INT32_C(  1599356720),  INT32_C(  2133455233), -INT32_C(  2050550602),  INT32_C(  1764739580), -INT32_C(   272636273), -INT32_C(  1544734556),  INT32_C(  1509853447),
        -INT32_C(   561310291), -INT32_C(   717365420), -INT32_C(  2024511792),  INT32_C(  1997282170),  INT32_C(  1944075236), -INT32_C(   966549726), -INT32_C(   546746152), -INT32_C(  2026346535) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   895663570),  INT32_C(  1223345662), -INT32_C(  1459427121),  INT32_C(  1438107245),  INT32_C(           0),  INT32_C(           0),
         INT32_C(   823600210),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1764739580), -INT32_C(   272636273),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   604412392), -INT32_C(  1655356770), -INT32_C(  1131838373),  INT32_C(   573050079),  INT32_C(           0),  INT32_C(           0),
         INT32_C(  1944075236),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   823600210),  INT32_C(  1599356720),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   895663570),  INT32_C(  1223345662), -INT32_C(  1131838373),  INT32_C(   573050079),  INT32_C(           0),  INT32_C(           0),
         INT32_C(  1764739580),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1944075236), -INT32_C(   966549726),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(    25576096),  INT32_C(  1520160511), -INT32_C(  1157756342),  INT32_C(   247760090),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(   561310291),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   561310291), -INT32_C(   717365420),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(50044),
      { -INT32_C(  1566388123), -INT32_C(   116886619),  INT32_C(   102007802), -INT32_C(   633145094), -INT32_C(   814062484), -INT32_C(  1826663894),  INT32_C(  1778121009), -INT32_C(   450005120),
        -INT32_C(   309866680),  INT32_C(  1021808450),  INT32_C(   205781778),  INT32_C(  1625785844),  INT32_C(   321872361), -INT32_C(   458862925), -INT32_C(   951213753),  INT32_C(  1638693401) },
      { -INT32_C(  1941032118), -INT32_C(   708233789),  INT32_C(   635571249),  INT32_C(  2055588241), -INT32_C(   577915862),  INT32_C(  1254175490), -INT32_C(   300871979), -INT32_C(   749748599),
        -INT32_C(  1268802063),  INT32_C(    76163283), -INT32_C(   987141068),  INT32_C(  1598008885),  INT32_C(  1681706082), -INT32_C(   709952256), -INT32_C(  1799110645),  INT32_C(  1869025918) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   205781778),  INT32_C(  1625785844),  INT32_C(   321872361), -INT32_C(   458862925), -INT32_C(   951213753),  INT32_C(           0),
        -INT32_C(  1941032118), -INT32_C(   708233789),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   300871979), -INT32_C(   749748599) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1778121009), -INT32_C(   450005120), -INT32_C(   309866680),  INT32_C(  1021808450),  INT32_C(   205781778),  INT32_C(           0),
         INT32_C(  1681706082), -INT32_C(   709952256),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   635571249),  INT32_C(  2055588241) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   205781778),  INT32_C(  1625785844), -INT32_C(   309866680),  INT32_C(  1021808450),  INT32_C(   205781778),  INT32_C(           0),
        -INT32_C(   577915862),  INT32_C(  1254175490),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1799110645),  INT32_C(  1869025918) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   951213753),  INT32_C(  1638693401), -INT32_C(   814062484), -INT32_C(  1826663894),  INT32_C(  1778121009),  INT32_C(           0),
        -INT32_C(  1268802063),  INT32_C(    76163283),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   987141068),  INT32_C(  1598008885) } },
    { UINT16_C(51119),
      { -INT32_C(  1360035292), -INT32_C(  1357241210),  INT32_C(   693981161), -INT32_C(   352993362), -INT32_C(   756550364), -INT32_C(  1919749430), -INT32_C(   291565433), -INT32_C(  1531621504),
         INT32_C(  1481811409), -INT32_C(  1308136247),  INT32_C(  1775986107),  INT32_C(  1213452580), -INT32_C(  1860486458), -INT32_C(  1289835220),  INT32_C(  1050853054), -INT32_C(   555591923) },
      { -INT32_C(   986303236),  INT32_C(  1534607008), -INT32_C(   943434845), -INT32_C(   351267036),  INT32_C(  2105289553), -INT32_C(  1775133992),  INT32_C(  1708446552),  INT32_C(   658814506),
        -INT32_C(  1964213526),  INT32_C(  1541760184), -INT32_C(   601642568),  INT32_C(   298267584),  INT32_C(   915358558),  INT32_C(   919453918), -INT32_C(  1113808493),  INT32_C(  1122295896) },
      {  INT32_C(  1481811409), -INT32_C(  1308136247),  INT32_C(  1775986107),  INT32_C(  1213452580),  INT32_C(           0), -INT32_C(  1289835220),  INT32_C(           0), -INT32_C(   555591923),
        -INT32_C(   986303236),  INT32_C(  1534607008), -INT32_C(   943434845),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1708446552),  INT32_C(   658814506) },
      { -INT32_C(   756550364), -INT32_C(  1919749430), -INT32_C(   291565433), -INT32_C(  1531621504),  INT32_C(           0), -INT32_C(  1308136247),  INT32_C(           0),  INT32_C(  1213452580),
         INT32_C(   915358558),  INT32_C(   919453918), -INT32_C(  1113808493),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   943434845), -INT32_C(   351267036) },
      {  INT32_C(  1481811409), -INT32_C(  1308136247),  INT32_C(  1775986107),  INT32_C(  1213452580),  INT32_C(           0), -INT32_C(  1308136247),  INT32_C(           0),  INT32_C(  1213452580),
         INT32_C(  2105289553), -INT32_C(  1775133992),  INT32_C(  1708446552),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1113808493),  INT32_C(  1122295896) },
      { -INT32_C(  1860486458), -INT32_C(  1289835220),  INT32_C(  1050853054), -INT32_C(   555591923),  INT32_C(           0), -INT32_C(  1919749430),  INT32_C(           0), -INT32_C(  1531621504),
        -INT32_C(  1964213526),  INT32_C(  1541760184), -INT32_C(   601642568),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   601642568),  INT32_C(   298267584) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;

    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r1));
    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r2));
    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r3));
    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r4));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r;

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i32x4(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shuffle_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r1[8];
    const int64_t r2[8];
    const int64_t r3[8];
    const int64_t r4[8];
  } test_vec[4] = {
    { {  INT64_C( 9064358367993939014),  INT64_C( 1655253076713723789), -INT64_C( 5202867091852910807), -INT64_C( 6235410276320205197),
         INT64_C( 3749974182315333027), -INT64_C( 4869958472303186068),  INT64_C( 1925200501811770125), -INT64_C( 2890610351864884010) },
      { -INT64_C( 5769834314734457581), -INT64_C( 2329367258326736035),  INT64_C( 1594719845308338272), -INT64_C( 5155441086053800087),
        -INT64_C( 6570647101520095010), -INT64_C( 6957146357537385790),  INT64_C( 5775523367379947984), -INT64_C( 8841584596049103377) },
      {  INT64_C( 3749974182315333027), -INT64_C( 4869958472303186068),  INT64_C( 1925200501811770125), -INT64_C( 2890610351864884010),
        -INT64_C( 5769834314734457581), -INT64_C( 2329367258326736035),  INT64_C( 1594719845308338272), -INT64_C( 5155441086053800087) },
      { -INT64_C( 5202867091852910807), -INT64_C( 6235410276320205197),  INT64_C( 3749974182315333027), -INT64_C( 4869958472303186068),
         INT64_C( 5775523367379947984), -INT64_C( 8841584596049103377), -INT64_C( 5769834314734457581), -INT64_C( 2329367258326736035) },
      {  INT64_C( 3749974182315333027), -INT64_C( 4869958472303186068),  INT64_C( 3749974182315333027), -INT64_C( 4869958472303186068),
         INT64_C( 1594719845308338272), -INT64_C( 5155441086053800087),  INT64_C( 5775523367379947984), -INT64_C( 8841584596049103377) },
      {  INT64_C( 1925200501811770125), -INT64_C( 2890610351864884010), -INT64_C( 5202867091852910807), -INT64_C( 6235410276320205197),
        -INT64_C( 6570647101520095010), -INT64_C( 6957146357537385790), -INT64_C( 6570647101520095010), -INT64_C( 6957146357537385790) } },
    { { -INT64_C( 2367637639300465351), -INT64_C( 4023993786916685751), -INT64_C( 8975674520688276169), -INT64_C(  603091206924706699),
         INT64_C(  512455698990712865), -INT64_C( 5974950190718677165),  INT64_C( 7033280028771872131),  INT64_C( 3984593951100089239) },
      { -INT64_C( 5535608529132075929), -INT64_C(  701696829275849695), -INT64_C( 2283982201537294337),  INT64_C( 5884941281556548445),
        -INT64_C( 3289699544413626298),  INT64_C( 7622813414819714912), -INT64_C( 2313264154265921349), -INT64_C( 2042103578795828371) },
      {  INT64_C(  512455698990712865), -INT64_C( 5974950190718677165),  INT64_C( 7033280028771872131),  INT64_C( 3984593951100089239),
        -INT64_C( 5535608529132075929), -INT64_C(  701696829275849695), -INT64_C( 2283982201537294337),  INT64_C( 5884941281556548445) },
      { -INT64_C( 8975674520688276169), -INT64_C(  603091206924706699),  INT64_C(  512455698990712865), -INT64_C( 5974950190718677165),
        -INT64_C( 2313264154265921349), -INT64_C( 2042103578795828371), -INT64_C( 5535608529132075929), -INT64_C(  701696829275849695) },
      {  INT64_C(  512455698990712865), -INT64_C( 5974950190718677165),  INT64_C(  512455698990712865), -INT64_C( 5974950190718677165),
        -INT64_C( 2283982201537294337),  INT64_C( 5884941281556548445), -INT64_C( 2313264154265921349), -INT64_C( 2042103578795828371) },
      {  INT64_C( 7033280028771872131),  INT64_C( 3984593951100089239), -INT64_C( 8975674520688276169), -INT64_C(  603091206924706699),
        -INT64_C( 3289699544413626298),  INT64_C( 7622813414819714912), -INT64_C( 3289699544413626298),  INT64_C( 7622813414819714912) } },
    { {  INT64_C( 2495523895361675037),  INT64_C( 8217591265508002360),  INT64_C( 8068300996032814999),  INT64_C( 7347745147448674936),
        -INT64_C( 7389326850122284914), -INT64_C( 4680852822059947552),  INT64_C( 6203375191466738085), -INT64_C( 2609726800046334694) },
      { -INT64_C( 5485488996149822236),  INT64_C(  355877138387835401), -INT64_C( 5617182149092587306), -INT64_C( 4699262278827673953),
        -INT64_C( 7106712558798375719), -INT64_C( 3143302540216587780),  INT64_C( 7913268533751450758), -INT64_C( 7160122232820537846) },
      { -INT64_C( 7389326850122284914), -INT64_C( 4680852822059947552),  INT64_C( 6203375191466738085), -INT64_C( 2609726800046334694),
        -INT64_C( 5485488996149822236),  INT64_C(  355877138387835401), -INT64_C( 5617182149092587306), -INT64_C( 4699262278827673953) },
      {  INT64_C( 8068300996032814999),  INT64_C( 7347745147448674936), -INT64_C( 7389326850122284914), -INT64_C( 4680852822059947552),
         INT64_C( 7913268533751450758), -INT64_C( 7160122232820537846), -INT64_C( 5485488996149822236),  INT64_C(  355877138387835401) },
      { -INT64_C( 7389326850122284914), -INT64_C( 4680852822059947552), -INT64_C( 7389326850122284914), -INT64_C( 4680852822059947552),
        -INT64_C( 5617182149092587306), -INT64_C( 4699262278827673953),  INT64_C( 7913268533751450758), -INT64_C( 7160122232820537846) },
      {  INT64_C( 6203375191466738085), -INT64_C( 2609726800046334694),  INT64_C( 8068300996032814999),  INT64_C( 7347745147448674936),
        -INT64_C( 7106712558798375719), -INT64_C( 3143302540216587780), -INT64_C( 7106712558798375719), -INT64_C( 3143302540216587780) } },
    { { -INT64_C( 8914691256948841194),  INT64_C( 6639222660098168657),  INT64_C( 5298567696017435469),  INT64_C( 1986596530830565937),
        -INT64_C( 3278323102294550156), -INT64_C( 1200415308882288347),  INT64_C( 5654149610722742767), -INT64_C( 1687208892942801803) },
      { -INT64_C( 8142396623853167699),  INT64_C( 6150194591829729735),  INT64_C( 7790941442859557810), -INT64_C( 2090942134086114913),
         INT64_C(  922693962766082006), -INT64_C( 1374215564511848397),  INT64_C(  834530889533452594), -INT64_C( 3795942993997956550) },
      { -INT64_C( 3278323102294550156), -INT64_C( 1200415308882288347),  INT64_C( 5654149610722742767), -INT64_C( 1687208892942801803),
        -INT64_C( 8142396623853167699),  INT64_C( 6150194591829729735),  INT64_C( 7790941442859557810), -INT64_C( 2090942134086114913) },
      {  INT64_C( 5298567696017435469),  INT64_C( 1986596530830565937), -INT64_C( 3278323102294550156), -INT64_C( 1200415308882288347),
         INT64_C(  834530889533452594), -INT64_C( 3795942993997956550), -INT64_C( 8142396623853167699),  INT64_C( 6150194591829729735) },
      { -INT64_C( 3278323102294550156), -INT64_C( 1200415308882288347), -INT64_C( 3278323102294550156), -INT64_C( 1200415308882288347),
         INT64_C( 7790941442859557810), -INT64_C( 2090942134086114913),  INT64_C(  834530889533452594), -INT64_C( 3795942993997956550) },
      {  INT64_C( 5654149610722742767), -INT64_C( 1687208892942801803),  INT64_C( 5298567696017435469),  INT64_C( 1986596530830565937),
         INT64_C(  922693962766082006), -INT64_C( 1374215564511848397),  INT64_C(  922693962766082006), -INT64_C( 1374215564511848397) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;

    r = easysimd_mm512_shuffle_i64x2(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r1));
    r = easysimd_mm512_shuffle_i64x2(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r2));
    r = easysimd_mm512_shuffle_i64x2(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r3));
    r = easysimd_mm512_shuffle_i64x2(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r4));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r;

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i64x2(a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i64x2(a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i64x2(a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_shuffle_i64x2(a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_shuffle_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r1[8];
    const int64_t r2[8];
    const int64_t r3[8];
    const int64_t r4[8];
  } test_vec[4] = {
    { { -INT64_C( 5450500260720638046), -INT64_C(  303069419900149776), -INT64_C( 8586981197250616758),  INT64_C( 6464661867379747878),
         INT64_C( 2825660459400055198),  INT64_C( 6419273303731259426),  INT64_C( 1268602132586889465),  INT64_C( 9148604011860445414) },
      UINT8_C(168),
      { -INT64_C(   52909362248744610),  INT64_C( 8350632976837467411),  INT64_C( 6372153185034761614),  INT64_C( 6997949517676416022),
         INT64_C( 1652475833497091255), -INT64_C(  861906724410295004), -INT64_C( 1314511409601812519),  INT64_C( 7447405327590554102) },
      { -INT64_C( 4065267619625247621),  INT64_C( 6024804178023257666), -INT64_C(  463170361341765426), -INT64_C( 1257264663508093834),
         INT64_C( 7607146190033773352),  INT64_C( 2695032016521523353),  INT64_C( 1711751179010712517),  INT64_C( 2860571913174758224) },
      { -INT64_C( 5450500260720638046), -INT64_C(  303069419900149776), -INT64_C( 8586981197250616758),  INT64_C( 7447405327590554102),
         INT64_C( 2825660459400055198),  INT64_C( 6024804178023257666),  INT64_C( 1268602132586889465), -INT64_C( 1257264663508093834) },
      { -INT64_C( 5450500260720638046), -INT64_C(  303069419900149776), -INT64_C( 8586981197250616758), -INT64_C(  861906724410295004),
         INT64_C( 2825660459400055198),  INT64_C( 2860571913174758224),  INT64_C( 1268602132586889465),  INT64_C( 6024804178023257666) },
      { -INT64_C( 5450500260720638046), -INT64_C(  303069419900149776), -INT64_C( 8586981197250616758), -INT64_C(  861906724410295004),
         INT64_C( 2825660459400055198), -INT64_C( 1257264663508093834),  INT64_C( 1268602132586889465),  INT64_C( 2860571913174758224) },
      { -INT64_C( 5450500260720638046), -INT64_C(  303069419900149776), -INT64_C( 8586981197250616758),  INT64_C( 6997949517676416022),
         INT64_C( 2825660459400055198),  INT64_C( 2695032016521523353),  INT64_C( 1268602132586889465),  INT64_C( 2695032016521523353) } },
    { { -INT64_C( 4954376155412881491), -INT64_C( 4399394521808482443),  INT64_C( 7570352073549200472), -INT64_C( 4384853372345430786),
         INT64_C( 1431254751958800638),  INT64_C( 4099913796378667813), -INT64_C( 5111514402357481586),  INT64_C( 1974507502212431244) },
      UINT8_C(244),
      { -INT64_C( 4324964065583481602),  INT64_C( 8206643744777378948), -INT64_C( 2821252711386936820), -INT64_C( 2275425146850935908),
         INT64_C( 2749640684518899276), -INT64_C( 5530878118052705333), -INT64_C( 1414737971373970710), -INT64_C( 5902581516866399661) },
      {  INT64_C( 5014823577851251407),  INT64_C( 6728466301046625286), -INT64_C( 8433764122048176198), -INT64_C(  386231857752089966),
        -INT64_C(  550279245437618529),  INT64_C( 8784278492937547543), -INT64_C( 7122873769374004379),  INT64_C( 6482710365369198034) },
      { -INT64_C( 4954376155412881491), -INT64_C( 4399394521808482443), -INT64_C( 1414737971373970710), -INT64_C( 4384853372345430786),
         INT64_C( 5014823577851251407),  INT64_C( 6728466301046625286), -INT64_C( 8433764122048176198), -INT64_C(  386231857752089966) },
      { -INT64_C( 4954376155412881491), -INT64_C( 4399394521808482443),  INT64_C( 2749640684518899276), -INT64_C( 4384853372345430786),
        -INT64_C( 7122873769374004379),  INT64_C( 6482710365369198034),  INT64_C( 5014823577851251407),  INT64_C( 6728466301046625286) },
      { -INT64_C( 4954376155412881491), -INT64_C( 4399394521808482443),  INT64_C( 2749640684518899276), -INT64_C( 4384853372345430786),
        -INT64_C( 8433764122048176198), -INT64_C(  386231857752089966), -INT64_C( 7122873769374004379),  INT64_C( 6482710365369198034) },
      { -INT64_C( 4954376155412881491), -INT64_C( 4399394521808482443), -INT64_C( 2821252711386936820), -INT64_C( 4384853372345430786),
        -INT64_C(  550279245437618529),  INT64_C( 8784278492937547543), -INT64_C(  550279245437618529),  INT64_C( 8784278492937547543) } },
    { {  INT64_C( 9139720534520587244), -INT64_C( 1767678788104520898),  INT64_C( 5000590375510080188),  INT64_C( 1919160357390382991),
         INT64_C( 2414719824660272285),  INT64_C( 4157551968255970121),  INT64_C( 8019334095005824401),  INT64_C( 7324079806595239261) },
      UINT8_C(111),
      {  INT64_C( 8553589047221709212),  INT64_C( 7402557902687469459), -INT64_C( 7498857272533663700), -INT64_C( 5028335926252766867),
        -INT64_C( 8030523979005130452), -INT64_C( 2886959456527417257),  INT64_C( 6329342989913634101),  INT64_C( 2713515502149318096) },
      { -INT64_C(  257045314966598431), -INT64_C( 3184761139703240630),  INT64_C( 6839535778382624425),  INT64_C( 4844734526285154846),
        -INT64_C( 8330938845354690368),  INT64_C(  299294054410217399),  INT64_C( 4715410797492648580),  INT64_C( 8084405194229005452) },
      { -INT64_C( 8030523979005130452), -INT64_C( 2886959456527417257),  INT64_C( 6329342989913634101),  INT64_C( 2713515502149318096),
         INT64_C( 2414719824660272285), -INT64_C( 3184761139703240630),  INT64_C( 6839535778382624425),  INT64_C( 7324079806595239261) },
      { -INT64_C( 7498857272533663700), -INT64_C( 5028335926252766867), -INT64_C( 8030523979005130452), -INT64_C( 2886959456527417257),
         INT64_C( 2414719824660272285),  INT64_C( 8084405194229005452), -INT64_C(  257045314966598431),  INT64_C( 7324079806595239261) },
      { -INT64_C( 8030523979005130452), -INT64_C( 2886959456527417257), -INT64_C( 8030523979005130452), -INT64_C( 2886959456527417257),
         INT64_C( 2414719824660272285),  INT64_C( 4844734526285154846),  INT64_C( 4715410797492648580),  INT64_C( 7324079806595239261) },
      {  INT64_C( 6329342989913634101),  INT64_C( 2713515502149318096), -INT64_C( 7498857272533663700), -INT64_C( 5028335926252766867),
         INT64_C( 2414719824660272285),  INT64_C(  299294054410217399), -INT64_C( 8330938845354690368),  INT64_C( 7324079806595239261) } },
    { { -INT64_C( 7432847169673604597),  INT64_C( 3278080296916638239),  INT64_C( 2500253803823961422), -INT64_C( 6373647233911355029),
        -INT64_C( 5112485413701524845),  INT64_C( 6628930374438484264),  INT64_C( 6449012749347077830),  INT64_C( 7262899722324840327) },
      UINT8_C( 98),
      { -INT64_C( 2398811228970624576), -INT64_C( 4368695573899147544), -INT64_C( 3884344527171830766), -INT64_C( 4043950958213120898),
         INT64_C( 8907287267678477785), -INT64_C( 2172576912451991407),  INT64_C( 1240331113154644272), -INT64_C( 6154889269332684346) },
      { -INT64_C( 8837180719984154034),  INT64_C( 1603128525128954254),  INT64_C(  117859106248897022), -INT64_C( 1151404580747202167),
         INT64_C(  272236279839682125), -INT64_C( 3054672174236277857),  INT64_C( 6884887643602427216),  INT64_C(  922322850172001082) },
      { -INT64_C( 7432847169673604597), -INT64_C( 2172576912451991407),  INT64_C( 2500253803823961422), -INT64_C( 6373647233911355029),
        -INT64_C( 5112485413701524845),  INT64_C( 1603128525128954254),  INT64_C(  117859106248897022),  INT64_C( 7262899722324840327) },
      { -INT64_C( 7432847169673604597), -INT64_C( 4043950958213120898),  INT64_C( 2500253803823961422), -INT64_C( 6373647233911355029),
        -INT64_C( 5112485413701524845),  INT64_C(  922322850172001082), -INT64_C( 8837180719984154034),  INT64_C( 7262899722324840327) },
      { -INT64_C( 7432847169673604597), -INT64_C( 2172576912451991407),  INT64_C( 2500253803823961422), -INT64_C( 6373647233911355029),
        -INT64_C( 5112485413701524845), -INT64_C( 1151404580747202167),  INT64_C( 6884887643602427216),  INT64_C( 7262899722324840327) },
      { -INT64_C( 7432847169673604597), -INT64_C( 6154889269332684346),  INT64_C( 2500253803823961422), -INT64_C( 6373647233911355029),
        -INT64_C( 5112485413701524845), -INT64_C( 3054672174236277857),  INT64_C(  272236279839682125),  INT64_C( 7262899722324840327) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;

    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r1));
    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r2));
    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r3));
    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r4));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r;

    easysimd_test_x86_write_i64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k,EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_mask_shuffle_i64x2(src, k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_shuffle_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r1[8];
    const int64_t r2[8];
    const int64_t r3[8];
    const int64_t r4[8];
  } test_vec[4] = {
    { UINT8_C( 21),
      {  INT64_C( 5490098567529945882),  INT64_C( 5911748109366692064), -INT64_C( 8897163291908773346),  INT64_C( 3301347518828238380),
        -INT64_C( 1119829433731952487), -INT64_C( 5331585107673360358),  INT64_C( 4464554940584633169),  INT64_C( 3197776265679000607) },
      {  INT64_C( 7789479484726821144),  INT64_C( 3381955780673304226),  INT64_C( 1985688297139057217),  INT64_C( 7432819889251714664),
        -INT64_C( 1788306984465892497), -INT64_C( 6257452164178687458), -INT64_C( 4261943065901872957),  INT64_C( 5027677267904374799) },
      { -INT64_C( 1119829433731952487),  INT64_C(                   0),  INT64_C( 4464554940584633169),  INT64_C(                   0),
         INT64_C( 7789479484726821144),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) },
      { -INT64_C( 8897163291908773346),  INT64_C(                   0), -INT64_C( 1119829433731952487),  INT64_C(                   0),
        -INT64_C( 4261943065901872957),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) },
      { -INT64_C( 1119829433731952487),  INT64_C(                   0), -INT64_C( 1119829433731952487),  INT64_C(                   0),
         INT64_C( 1985688297139057217),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C( 4464554940584633169),  INT64_C(                   0), -INT64_C( 8897163291908773346),  INT64_C(                   0),
        -INT64_C( 1788306984465892497),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 97),
      {  INT64_C( 9159070629727962964),  INT64_C(  685268701085636527),  INT64_C( 4909814318836826506), -INT64_C( 7336539410741362199),
         INT64_C( 1376891355797116519),  INT64_C( 2095112337392534218), -INT64_C( 8289064626617812586), -INT64_C( 2687089379983617324) },
      { -INT64_C( 3902488924882842528), -INT64_C( 7690767184866006636), -INT64_C( 7165439453353773984), -INT64_C( 3126223317569639856),
         INT64_C( 4215773483752756152), -INT64_C( 7085106431956789992), -INT64_C( 9043738006621392103),  INT64_C( 6440959600597742874) },
      {  INT64_C( 1376891355797116519),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 7690767184866006636), -INT64_C( 7165439453353773984),  INT64_C(                   0) },
      {  INT64_C( 4909814318836826506),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C( 6440959600597742874), -INT64_C( 3902488924882842528),  INT64_C(                   0) },
      {  INT64_C( 1376891355797116519),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 3126223317569639856), -INT64_C( 9043738006621392103),  INT64_C(                   0) },
      { -INT64_C( 8289064626617812586),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 7085106431956789992),  INT64_C( 4215773483752756152),  INT64_C(                   0) } },
    { UINT8_C(246),
      { -INT64_C( 1692555680829206022),  INT64_C( 4104401340214909238), -INT64_C( 8525334622065994140), -INT64_C( 7346632544040108750),
        -INT64_C( 5615559392548837739),  INT64_C( 6745411449051297311),  INT64_C( 2753847769439536985),  INT64_C( 8870198896857146078) },
      { -INT64_C( 4459771292523684371), -INT64_C( 8772235626085438997), -INT64_C( 5412082286445972279),  INT64_C( 5994092533518573517),
         INT64_C( 8344562663899845480),  INT64_C( 9165840875523732353), -INT64_C( 6892440561819752497),  INT64_C( 4609860655822179136) },
      {  INT64_C(                   0),  INT64_C( 6745411449051297311),  INT64_C( 2753847769439536985),  INT64_C(                   0),
        -INT64_C( 4459771292523684371), -INT64_C( 8772235626085438997), -INT64_C( 5412082286445972279),  INT64_C( 5994092533518573517) },
      {  INT64_C(                   0), -INT64_C( 7346632544040108750), -INT64_C( 5615559392548837739),  INT64_C(                   0),
        -INT64_C( 6892440561819752497),  INT64_C( 4609860655822179136), -INT64_C( 4459771292523684371), -INT64_C( 8772235626085438997) },
      {  INT64_C(                   0),  INT64_C( 6745411449051297311), -INT64_C( 5615559392548837739),  INT64_C(                   0),
        -INT64_C( 5412082286445972279),  INT64_C( 5994092533518573517), -INT64_C( 6892440561819752497),  INT64_C( 4609860655822179136) },
      {  INT64_C(                   0),  INT64_C( 8870198896857146078), -INT64_C( 8525334622065994140),  INT64_C(                   0),
         INT64_C( 8344562663899845480),  INT64_C( 9165840875523732353),  INT64_C( 8344562663899845480),  INT64_C( 9165840875523732353) } },
    { UINT8_C( 86),
      {  INT64_C(  868720944030305205),  INT64_C( 6202655082719394730), -INT64_C( 6998371639605216898),  INT64_C( 3888775267815360030),
        -INT64_C( 3448402552616200705), -INT64_C( 6426308844223863156), -INT64_C( 2047970398376166429), -INT64_C( 4800419675374351956) },
      { -INT64_C( 9203112485147302832),  INT64_C( 7686744639541042399), -INT64_C( 4363719656928644408), -INT64_C( 4650158549917240067),
         INT64_C( 2660075167475387509),  INT64_C( 8847840104046912985),  INT64_C( 1919790311379773897),  INT64_C( 6528333684952896619) },
      {  INT64_C(                   0), -INT64_C( 6426308844223863156), -INT64_C( 2047970398376166429),  INT64_C(                   0),
        -INT64_C( 9203112485147302832),  INT64_C(                   0), -INT64_C( 4363719656928644408),  INT64_C(                   0) },
      {  INT64_C(                   0),  INT64_C( 3888775267815360030), -INT64_C( 3448402552616200705),  INT64_C(                   0),
         INT64_C( 1919790311379773897),  INT64_C(                   0), -INT64_C( 9203112485147302832),  INT64_C(                   0) },
      {  INT64_C(                   0), -INT64_C( 6426308844223863156), -INT64_C( 3448402552616200705),  INT64_C(                   0),
        -INT64_C( 4363719656928644408),  INT64_C(                   0),  INT64_C( 1919790311379773897),  INT64_C(                   0) },
      {  INT64_C(                   0), -INT64_C( 4800419675374351956), -INT64_C( 6998371639605216898),  INT64_C(                   0),
         INT64_C( 2660075167475387509),  INT64_C(                   0),  INT64_C( 2660075167475387509),  INT64_C(                   0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;

    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r1));
    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r2));
    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r3));
    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r4));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 4 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r;

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (1 << 6) + (0 << 4) + (3 << 2) + 2);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (0 << 6) + (3 << 4) + (2 << 2) + 1);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (3 << 6) + (1 << 4) + (2 << 2) + 2);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    r = easysimd_mm512_maskz_shuffle_i64x2(k, a, b, (2 << 6) + (2 << 4) + (1 << 2) + 3);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shuffle_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const int imm8;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   514.49), EASYSIMD_FLOAT32_C(  -368.57), EASYSIMD_FLOAT32_C(   282.53), EASYSIMD_FLOAT32_C(   595.38),
        EASYSIMD_FLOAT32_C(   753.27), EASYSIMD_FLOAT32_C(    38.43), EASYSIMD_FLOAT32_C(  -933.79), EASYSIMD_FLOAT32_C(  -225.40),
        EASYSIMD_FLOAT32_C(   332.60), EASYSIMD_FLOAT32_C(   373.66), EASYSIMD_FLOAT32_C(  -355.71), EASYSIMD_FLOAT32_C(   243.38),
        EASYSIMD_FLOAT32_C(  -712.53), EASYSIMD_FLOAT32_C(  -321.02), EASYSIMD_FLOAT32_C(   309.70), EASYSIMD_FLOAT32_C(  -515.61) },
      { EASYSIMD_FLOAT32_C(   335.57), EASYSIMD_FLOAT32_C(   278.90), EASYSIMD_FLOAT32_C(   976.62), EASYSIMD_FLOAT32_C(   291.46),
        EASYSIMD_FLOAT32_C(   300.14), EASYSIMD_FLOAT32_C(   685.03), EASYSIMD_FLOAT32_C(   706.83), EASYSIMD_FLOAT32_C(   754.20),
        EASYSIMD_FLOAT32_C(    88.74), EASYSIMD_FLOAT32_C(  -596.79), EASYSIMD_FLOAT32_C(  -624.05), EASYSIMD_FLOAT32_C(   211.73),
        EASYSIMD_FLOAT32_C(  -308.89), EASYSIMD_FLOAT32_C(   977.33), EASYSIMD_FLOAT32_C(  -993.40), EASYSIMD_FLOAT32_C(  -794.40) },
       INT32_C(         117),
      { EASYSIMD_FLOAT32_C(  -368.57), EASYSIMD_FLOAT32_C(  -368.57), EASYSIMD_FLOAT32_C(   291.46), EASYSIMD_FLOAT32_C(   278.90),
        EASYSIMD_FLOAT32_C(    38.43), EASYSIMD_FLOAT32_C(    38.43), EASYSIMD_FLOAT32_C(   754.20), EASYSIMD_FLOAT32_C(   685.03),
        EASYSIMD_FLOAT32_C(   373.66), EASYSIMD_FLOAT32_C(   373.66), EASYSIMD_FLOAT32_C(   211.73), EASYSIMD_FLOAT32_C(  -596.79),
        EASYSIMD_FLOAT32_C(  -321.02), EASYSIMD_FLOAT32_C(  -321.02), EASYSIMD_FLOAT32_C(  -794.40), EASYSIMD_FLOAT32_C(   977.33) } },
    { { EASYSIMD_FLOAT32_C(  -672.44), EASYSIMD_FLOAT32_C(   867.19), EASYSIMD_FLOAT32_C(   136.64), EASYSIMD_FLOAT32_C(   660.16),
        EASYSIMD_FLOAT32_C(   240.85), EASYSIMD_FLOAT32_C(   780.93), EASYSIMD_FLOAT32_C(   -96.46), EASYSIMD_FLOAT32_C(   528.32),
        EASYSIMD_FLOAT32_C(  -540.09), EASYSIMD_FLOAT32_C(  -786.77), EASYSIMD_FLOAT32_C(  -987.30), EASYSIMD_FLOAT32_C(   795.49),
        EASYSIMD_FLOAT32_C(   492.13), EASYSIMD_FLOAT32_C(   989.33), EASYSIMD_FLOAT32_C(    86.95), EASYSIMD_FLOAT32_C(  -207.73) },
      { EASYSIMD_FLOAT32_C(   674.36), EASYSIMD_FLOAT32_C(  -206.22), EASYSIMD_FLOAT32_C(  -453.53), EASYSIMD_FLOAT32_C(  -236.90),
        EASYSIMD_FLOAT32_C(   196.99), EASYSIMD_FLOAT32_C(   -77.58), EASYSIMD_FLOAT32_C(   974.83), EASYSIMD_FLOAT32_C(   888.09),
        EASYSIMD_FLOAT32_C(  -100.25), EASYSIMD_FLOAT32_C(   981.43), EASYSIMD_FLOAT32_C(  -906.31), EASYSIMD_FLOAT32_C(   508.51),
        EASYSIMD_FLOAT32_C(   270.56), EASYSIMD_FLOAT32_C(   894.67), EASYSIMD_FLOAT32_C(   870.55), EASYSIMD_FLOAT32_C(   598.13) },
       INT32_C(          72),
      { EASYSIMD_FLOAT32_C(  -672.44), EASYSIMD_FLOAT32_C(   136.64), EASYSIMD_FLOAT32_C(   674.36), EASYSIMD_FLOAT32_C(  -206.22),
        EASYSIMD_FLOAT32_C(   240.85), EASYSIMD_FLOAT32_C(   -96.46), EASYSIMD_FLOAT32_C(   196.99), EASYSIMD_FLOAT32_C(   -77.58),
        EASYSIMD_FLOAT32_C(  -540.09), EASYSIMD_FLOAT32_C(  -987.30), EASYSIMD_FLOAT32_C(  -100.25), EASYSIMD_FLOAT32_C(   981.43),
        EASYSIMD_FLOAT32_C(   492.13), EASYSIMD_FLOAT32_C(    86.95), EASYSIMD_FLOAT32_C(   270.56), EASYSIMD_FLOAT32_C(   894.67) } },
    { { EASYSIMD_FLOAT32_C(  -211.88), EASYSIMD_FLOAT32_C(  -838.18), EASYSIMD_FLOAT32_C(  -468.98), EASYSIMD_FLOAT32_C(   248.04),
        EASYSIMD_FLOAT32_C(  -624.95), EASYSIMD_FLOAT32_C(  -456.28), EASYSIMD_FLOAT32_C(    43.52), EASYSIMD_FLOAT32_C(   867.18),
        EASYSIMD_FLOAT32_C(  -466.95), EASYSIMD_FLOAT32_C(  -869.53), EASYSIMD_FLOAT32_C(  -340.55), EASYSIMD_FLOAT32_C(  -792.59),
        EASYSIMD_FLOAT32_C(   -75.75), EASYSIMD_FLOAT32_C(   205.91), EASYSIMD_FLOAT32_C(   -29.49), EASYSIMD_FLOAT32_C(  -878.76) },
      { EASYSIMD_FLOAT32_C(  -871.67), EASYSIMD_FLOAT32_C(   -54.66), EASYSIMD_FLOAT32_C(  -990.66), EASYSIMD_FLOAT32_C(    28.08),
        EASYSIMD_FLOAT32_C(   -73.23), EASYSIMD_FLOAT32_C(  -896.97), EASYSIMD_FLOAT32_C(  -463.40), EASYSIMD_FLOAT32_C(  -802.67),
        EASYSIMD_FLOAT32_C(   997.70), EASYSIMD_FLOAT32_C(  -592.86), EASYSIMD_FLOAT32_C(   795.46), EASYSIMD_FLOAT32_C(   759.56),
        EASYSIMD_FLOAT32_C(   414.34), EASYSIMD_FLOAT32_C(    53.74), EASYSIMD_FLOAT32_C(  -237.74), EASYSIMD_FLOAT32_C(  -797.54) },
       INT32_C(         230),
      { EASYSIMD_FLOAT32_C(  -468.98), EASYSIMD_FLOAT32_C(  -838.18), EASYSIMD_FLOAT32_C(  -990.66), EASYSIMD_FLOAT32_C(    28.08),
        EASYSIMD_FLOAT32_C(    43.52), EASYSIMD_FLOAT32_C(  -456.28), EASYSIMD_FLOAT32_C(  -463.40), EASYSIMD_FLOAT32_C(  -802.67),
        EASYSIMD_FLOAT32_C(  -340.55), EASYSIMD_FLOAT32_C(  -869.53), EASYSIMD_FLOAT32_C(   795.46), EASYSIMD_FLOAT32_C(   759.56),
        EASYSIMD_FLOAT32_C(   -29.49), EASYSIMD_FLOAT32_C(   205.91), EASYSIMD_FLOAT32_C(  -237.74), EASYSIMD_FLOAT32_C(  -797.54) } },
    { { EASYSIMD_FLOAT32_C(   837.00), EASYSIMD_FLOAT32_C(  -505.98), EASYSIMD_FLOAT32_C(   457.79), EASYSIMD_FLOAT32_C(  -629.94),
        EASYSIMD_FLOAT32_C(  -375.50), EASYSIMD_FLOAT32_C(  -882.76), EASYSIMD_FLOAT32_C(  -422.53), EASYSIMD_FLOAT32_C(   548.75),
        EASYSIMD_FLOAT32_C(   323.15), EASYSIMD_FLOAT32_C(   547.98), EASYSIMD_FLOAT32_C(   669.99), EASYSIMD_FLOAT32_C(   451.49),
        EASYSIMD_FLOAT32_C(  -506.68), EASYSIMD_FLOAT32_C(   679.33), EASYSIMD_FLOAT32_C(  -520.43), EASYSIMD_FLOAT32_C(   420.08) },
      { EASYSIMD_FLOAT32_C(   782.36), EASYSIMD_FLOAT32_C(    16.17), EASYSIMD_FLOAT32_C(   617.42), EASYSIMD_FLOAT32_C(   780.06),
        EASYSIMD_FLOAT32_C(   423.31), EASYSIMD_FLOAT32_C(   412.87), EASYSIMD_FLOAT32_C(   539.61), EASYSIMD_FLOAT32_C(  -162.35),
        EASYSIMD_FLOAT32_C(  -533.39), EASYSIMD_FLOAT32_C(  -698.13), EASYSIMD_FLOAT32_C(    40.11), EASYSIMD_FLOAT32_C(   682.17),
        EASYSIMD_FLOAT32_C(   595.15), EASYSIMD_FLOAT32_C(  -509.40), EASYSIMD_FLOAT32_C(   272.78), EASYSIMD_FLOAT32_C(   432.16) },
       INT32_C(         111),
      { EASYSIMD_FLOAT32_C(  -629.94), EASYSIMD_FLOAT32_C(  -629.94), EASYSIMD_FLOAT32_C(   617.42), EASYSIMD_FLOAT32_C(    16.17),
        EASYSIMD_FLOAT32_C(   548.75), EASYSIMD_FLOAT32_C(   548.75), EASYSIMD_FLOAT32_C(   539.61), EASYSIMD_FLOAT32_C(   412.87),
        EASYSIMD_FLOAT32_C(   451.49), EASYSIMD_FLOAT32_C(   451.49), EASYSIMD_FLOAT32_C(    40.11), EASYSIMD_FLOAT32_C(  -698.13),
        EASYSIMD_FLOAT32_C(   420.08), EASYSIMD_FLOAT32_C(   420.08), EASYSIMD_FLOAT32_C(   272.78), EASYSIMD_FLOAT32_C(  -509.40) } },
    { { EASYSIMD_FLOAT32_C(  -152.18), EASYSIMD_FLOAT32_C(  -620.32), EASYSIMD_FLOAT32_C(   157.87), EASYSIMD_FLOAT32_C(  -829.03),
        EASYSIMD_FLOAT32_C(   927.66), EASYSIMD_FLOAT32_C(  -172.14), EASYSIMD_FLOAT32_C(   622.46), EASYSIMD_FLOAT32_C(  -579.02),
        EASYSIMD_FLOAT32_C(  -492.81), EASYSIMD_FLOAT32_C(  -897.97), EASYSIMD_FLOAT32_C(   841.06), EASYSIMD_FLOAT32_C(  -710.45),
        EASYSIMD_FLOAT32_C(   118.20), EASYSIMD_FLOAT32_C(   458.48), EASYSIMD_FLOAT32_C(  -930.40), EASYSIMD_FLOAT32_C(  -458.49) },
      { EASYSIMD_FLOAT32_C(  -128.65), EASYSIMD_FLOAT32_C(   609.22), EASYSIMD_FLOAT32_C(   379.15), EASYSIMD_FLOAT32_C(   337.97),
        EASYSIMD_FLOAT32_C(   911.09), EASYSIMD_FLOAT32_C(  -580.74), EASYSIMD_FLOAT32_C(    20.14), EASYSIMD_FLOAT32_C(   506.24),
        EASYSIMD_FLOAT32_C(   -90.14), EASYSIMD_FLOAT32_C(  -707.08), EASYSIMD_FLOAT32_C(   -61.60), EASYSIMD_FLOAT32_C(   894.49),
        EASYSIMD_FLOAT32_C(    23.50), EASYSIMD_FLOAT32_C(  -259.38), EASYSIMD_FLOAT32_C(   503.61), EASYSIMD_FLOAT32_C(   871.32) },
       INT32_C(         186),
      { EASYSIMD_FLOAT32_C(   157.87), EASYSIMD_FLOAT32_C(   157.87), EASYSIMD_FLOAT32_C(   337.97), EASYSIMD_FLOAT32_C(   379.15),
        EASYSIMD_FLOAT32_C(   622.46), EASYSIMD_FLOAT32_C(   622.46), EASYSIMD_FLOAT32_C(   506.24), EASYSIMD_FLOAT32_C(    20.14),
        EASYSIMD_FLOAT32_C(   841.06), EASYSIMD_FLOAT32_C(   841.06), EASYSIMD_FLOAT32_C(   894.49), EASYSIMD_FLOAT32_C(   -61.60),
        EASYSIMD_FLOAT32_C(  -930.40), EASYSIMD_FLOAT32_C(  -930.40), EASYSIMD_FLOAT32_C(   871.32), EASYSIMD_FLOAT32_C(   503.61) } },
    { { EASYSIMD_FLOAT32_C(   489.34), EASYSIMD_FLOAT32_C(   664.76), EASYSIMD_FLOAT32_C(   468.94), EASYSIMD_FLOAT32_C(   996.53),
        EASYSIMD_FLOAT32_C(   766.79), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -713.92), EASYSIMD_FLOAT32_C(  -115.01),
        EASYSIMD_FLOAT32_C(  -231.52), EASYSIMD_FLOAT32_C(  -644.32), EASYSIMD_FLOAT32_C(   426.50), EASYSIMD_FLOAT32_C(   639.84),
        EASYSIMD_FLOAT32_C(   964.90), EASYSIMD_FLOAT32_C(  -194.35), EASYSIMD_FLOAT32_C(   -22.20), EASYSIMD_FLOAT32_C(   875.99) },
      { EASYSIMD_FLOAT32_C(   224.92), EASYSIMD_FLOAT32_C(   997.94), EASYSIMD_FLOAT32_C(   382.23), EASYSIMD_FLOAT32_C(  -865.22),
        EASYSIMD_FLOAT32_C(  -709.13), EASYSIMD_FLOAT32_C(  -679.37), EASYSIMD_FLOAT32_C(  -970.73), EASYSIMD_FLOAT32_C(   314.37),
        EASYSIMD_FLOAT32_C(    61.24), EASYSIMD_FLOAT32_C(   532.88), EASYSIMD_FLOAT32_C(   185.69), EASYSIMD_FLOAT32_C(  -818.46),
        EASYSIMD_FLOAT32_C(  -805.64), EASYSIMD_FLOAT32_C(   227.99), EASYSIMD_FLOAT32_C(   229.50), EASYSIMD_FLOAT32_C(   683.70) },
       INT32_C(         150),
      { EASYSIMD_FLOAT32_C(   468.94), EASYSIMD_FLOAT32_C(   664.76), EASYSIMD_FLOAT32_C(   997.94), EASYSIMD_FLOAT32_C(   382.23),
        EASYSIMD_FLOAT32_C(  -713.92), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -679.37), EASYSIMD_FLOAT32_C(  -970.73),
        EASYSIMD_FLOAT32_C(   426.50), EASYSIMD_FLOAT32_C(  -644.32), EASYSIMD_FLOAT32_C(   532.88), EASYSIMD_FLOAT32_C(   185.69),
        EASYSIMD_FLOAT32_C(   -22.20), EASYSIMD_FLOAT32_C(  -194.35), EASYSIMD_FLOAT32_C(   227.99), EASYSIMD_FLOAT32_C(   229.50) } },
    { { EASYSIMD_FLOAT32_C(  -991.55), EASYSIMD_FLOAT32_C(   966.31), EASYSIMD_FLOAT32_C(   544.53), EASYSIMD_FLOAT32_C(  -223.07),
        EASYSIMD_FLOAT32_C(  -678.00), EASYSIMD_FLOAT32_C(   -28.97), EASYSIMD_FLOAT32_C(  -583.23), EASYSIMD_FLOAT32_C(  -713.11),
        EASYSIMD_FLOAT32_C(   776.69), EASYSIMD_FLOAT32_C(   394.57), EASYSIMD_FLOAT32_C(  -837.12), EASYSIMD_FLOAT32_C(     1.60),
        EASYSIMD_FLOAT32_C(   392.51), EASYSIMD_FLOAT32_C(   545.11), EASYSIMD_FLOAT32_C(   136.38), EASYSIMD_FLOAT32_C(   683.38) },
      { EASYSIMD_FLOAT32_C(   865.74), EASYSIMD_FLOAT32_C(   165.65), EASYSIMD_FLOAT32_C(    -2.26), EASYSIMD_FLOAT32_C(   -73.02),
        EASYSIMD_FLOAT32_C(  -301.47), EASYSIMD_FLOAT32_C(  -816.57), EASYSIMD_FLOAT32_C(   108.52), EASYSIMD_FLOAT32_C(  -107.11),
        EASYSIMD_FLOAT32_C(   411.42), EASYSIMD_FLOAT32_C(  -661.97), EASYSIMD_FLOAT32_C(  -423.40), EASYSIMD_FLOAT32_C(  -695.83),
        EASYSIMD_FLOAT32_C(    36.47), EASYSIMD_FLOAT32_C(  -743.17), EASYSIMD_FLOAT32_C(   -36.29), EASYSIMD_FLOAT32_C(    44.91) },
       INT32_C(         221),
      { EASYSIMD_FLOAT32_C(   966.31), EASYSIMD_FLOAT32_C(  -223.07), EASYSIMD_FLOAT32_C(   165.65), EASYSIMD_FLOAT32_C(   -73.02),
        EASYSIMD_FLOAT32_C(   -28.97), EASYSIMD_FLOAT32_C(  -713.11), EASYSIMD_FLOAT32_C(  -816.57), EASYSIMD_FLOAT32_C(  -107.11),
        EASYSIMD_FLOAT32_C(   394.57), EASYSIMD_FLOAT32_C(     1.60), EASYSIMD_FLOAT32_C(  -661.97), EASYSIMD_FLOAT32_C(  -695.83),
        EASYSIMD_FLOAT32_C(   545.11), EASYSIMD_FLOAT32_C(   683.38), EASYSIMD_FLOAT32_C(  -743.17), EASYSIMD_FLOAT32_C(    44.91) } },
    { { EASYSIMD_FLOAT32_C(   479.28), EASYSIMD_FLOAT32_C(  -761.39), EASYSIMD_FLOAT32_C(  -167.97), EASYSIMD_FLOAT32_C(   255.97),
        EASYSIMD_FLOAT32_C(   633.18), EASYSIMD_FLOAT32_C(    -5.09), EASYSIMD_FLOAT32_C(  -742.43), EASYSIMD_FLOAT32_C(    25.69),
        EASYSIMD_FLOAT32_C(  -459.98), EASYSIMD_FLOAT32_C(   393.95), EASYSIMD_FLOAT32_C(  -290.94), EASYSIMD_FLOAT32_C(  -594.25),
        EASYSIMD_FLOAT32_C(  -440.40), EASYSIMD_FLOAT32_C(   706.81), EASYSIMD_FLOAT32_C(   332.73), EASYSIMD_FLOAT32_C(   258.14) },
      { EASYSIMD_FLOAT32_C(   890.24), EASYSIMD_FLOAT32_C(  -558.74), EASYSIMD_FLOAT32_C(  -848.97), EASYSIMD_FLOAT32_C(   301.66),
        EASYSIMD_FLOAT32_C(  -220.72), EASYSIMD_FLOAT32_C(  -272.38), EASYSIMD_FLOAT32_C(   605.83), EASYSIMD_FLOAT32_C(   815.75),
        EASYSIMD_FLOAT32_C(   -15.54), EASYSIMD_FLOAT32_C(  -430.46), EASYSIMD_FLOAT32_C(  -139.33), EASYSIMD_FLOAT32_C(   207.60),
        EASYSIMD_FLOAT32_C(    77.79), EASYSIMD_FLOAT32_C(  -317.49), EASYSIMD_FLOAT32_C(   752.74), EASYSIMD_FLOAT32_C(  -442.93) },
       INT32_C(         166),
      { EASYSIMD_FLOAT32_C(  -167.97), EASYSIMD_FLOAT32_C(  -761.39), EASYSIMD_FLOAT32_C(  -848.97), EASYSIMD_FLOAT32_C(  -848.97),
        EASYSIMD_FLOAT32_C(  -742.43), EASYSIMD_FLOAT32_C(    -5.09), EASYSIMD_FLOAT32_C(   605.83), EASYSIMD_FLOAT32_C(   605.83),
        EASYSIMD_FLOAT32_C(  -290.94), EASYSIMD_FLOAT32_C(   393.95), EASYSIMD_FLOAT32_C(  -139.33), EASYSIMD_FLOAT32_C(  -139.33),
        EASYSIMD_FLOAT32_C(   332.73), EASYSIMD_FLOAT32_C(   706.81), EASYSIMD_FLOAT32_C(   752.74), EASYSIMD_FLOAT32_C(   752.74) } },
  };

  easysimd__m512 a, b, r;

  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  b = easysimd_mm512_loadu_ps(test_vec[0].b);
  r = easysimd_mm512_shuffle_ps(a, b, INT32_C(         117));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  b = easysimd_mm512_loadu_ps(test_vec[1].b);
  r = easysimd_mm512_shuffle_ps(a, b, INT32_C(          72));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  b = easysimd_mm512_loadu_ps(test_vec[2].b);
  r = easysimd_mm512_shuffle_ps(a, b, INT32_C(         230));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  b = easysimd_mm512_loadu_ps(test_vec[3].b);
  r = easysimd_mm512_shuffle_ps(a, b, INT32_C(         111));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  b = easysimd_mm512_loadu_ps(test_vec[4].b);
  r = easysimd_mm512_shuffle_ps(a, b, INT32_C(         186));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  b = easysimd_mm512_loadu_ps(test_vec[5].b);
  r = easysimd_mm512_shuffle_ps(a, b, INT32_C(         150));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  b = easysimd_mm512_loadu_ps(test_vec[6].b);
  r = easysimd_mm512_shuffle_ps(a, b, INT32_C(         221));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  b = easysimd_mm512_loadu_ps(test_vec[7].b);

  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r = easysimd_mm512_shuffle_ps(a, b, INT32_C(         166));
  } EASYSIMD_TEST_PERF_END("easysimd_mm512_shuffle_ps");
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_shuffle_ps, r, easysimd_mm512_setzero_ps(), imm8, a, b);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_shuffle_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_shuffle_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shuffle_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_shuffle_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shuffle_epi8)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_shuffle_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_shuffle_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_shuffle_f32x4)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_shuffle_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_shuffle_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_shuffle_f64x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_shuffle_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_shuffle_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_shuffle_i32x4)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_shuffle_i64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_shuffle_i64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_shuffle_i64x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shuffle_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_shuffle_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shuffle_f32x4)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shuffle_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_shuffle_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shuffle_f64x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shuffle_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_shuffle_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shuffle_i32x4)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shuffle_i64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_shuffle_i64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shuffle_i64x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shuffle_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shuffle_epi32)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
