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

#define EASYSIMD_TEST_X86_AVX512_INSN cvt

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/cvt.h>

static int
test_easysimd_mm_mask_cvtepi32_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const int32_t a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -622.84), EASYSIMD_FLOAT32_C(   175.90), EASYSIMD_FLOAT32_C(   890.47), EASYSIMD_FLOAT32_C(    -5.09) },
      UINT8_C(216),
      {  INT32_C(   540325598),  INT32_C(  2117636426), -INT32_C(   736953562),  INT32_C(   257179605) },
      { EASYSIMD_FLOAT32_C(  -622.84), EASYSIMD_FLOAT32_C(   175.90), EASYSIMD_FLOAT32_C(   890.47), EASYSIMD_FLOAT32_C(257179600.00) } },
    { { EASYSIMD_FLOAT32_C(  -859.62), EASYSIMD_FLOAT32_C(   474.59), EASYSIMD_FLOAT32_C(   262.67), EASYSIMD_FLOAT32_C(  -464.70) },
      UINT8_C(200),
      {  INT32_C(     6241146),  INT32_C(   829716514), -INT32_C(  1072673782),  INT32_C(   285945984) },
      { EASYSIMD_FLOAT32_C(  -859.62), EASYSIMD_FLOAT32_C(   474.59), EASYSIMD_FLOAT32_C(   262.67), EASYSIMD_FLOAT32_C(285945984.00) } },
    { { EASYSIMD_FLOAT32_C(  -806.63), EASYSIMD_FLOAT32_C(  -370.85), EASYSIMD_FLOAT32_C(    76.86), EASYSIMD_FLOAT32_C(  -998.90) },
      UINT8_C(155),
      {  INT32_C(  1624913932),  INT32_C(  1060229447),  INT32_C(  1958345017), -INT32_C(   778651048) },
      { EASYSIMD_FLOAT32_C(1624913920.00), EASYSIMD_FLOAT32_C(1060229440.00), EASYSIMD_FLOAT32_C(    76.86), EASYSIMD_FLOAT32_C(-778651072.00) } },
    { { EASYSIMD_FLOAT32_C(  -110.77), EASYSIMD_FLOAT32_C(  -233.71), EASYSIMD_FLOAT32_C(  -409.17), EASYSIMD_FLOAT32_C(   134.98) },
      UINT8_C(216),
      { -INT32_C(  1509360997),  INT32_C(  1127182348),  INT32_C(   189778643),  INT32_C(  1985130661) },
      { EASYSIMD_FLOAT32_C(  -110.77), EASYSIMD_FLOAT32_C(  -233.71), EASYSIMD_FLOAT32_C(  -409.17), EASYSIMD_FLOAT32_C(1985130624.00) } },
    { { EASYSIMD_FLOAT32_C(  -392.23), EASYSIMD_FLOAT32_C(  -379.05), EASYSIMD_FLOAT32_C(  -379.45), EASYSIMD_FLOAT32_C(   698.32) },
      UINT8_C( 75),
      { -INT32_C(  1190841821), -INT32_C(   561958141), -INT32_C(  1468376659), -INT32_C(   776724383) },
      { EASYSIMD_FLOAT32_C(-1190841856.00), EASYSIMD_FLOAT32_C(-561958144.00), EASYSIMD_FLOAT32_C(  -379.45), EASYSIMD_FLOAT32_C(-776724352.00) } },
    { { EASYSIMD_FLOAT32_C(  -259.52), EASYSIMD_FLOAT32_C(   546.09), EASYSIMD_FLOAT32_C(  -662.08), EASYSIMD_FLOAT32_C(  -856.87) },
      UINT8_C( 71),
      {  INT32_C(    49790895), -INT32_C(   460007371), -INT32_C(   469246030),  INT32_C(   417907173) },
      { EASYSIMD_FLOAT32_C(49790896.00), EASYSIMD_FLOAT32_C(-460007360.00), EASYSIMD_FLOAT32_C(-469246016.00), EASYSIMD_FLOAT32_C(  -856.87) } },
    { { EASYSIMD_FLOAT32_C(   575.69), EASYSIMD_FLOAT32_C(  -261.83), EASYSIMD_FLOAT32_C(   -32.31), EASYSIMD_FLOAT32_C(   674.39) },
      UINT8_C( 64),
      {  INT32_C(   576847214),  INT32_C(  1931063759), -INT32_C(  1960615733),  INT32_C(   801121623) },
      { EASYSIMD_FLOAT32_C(   575.69), EASYSIMD_FLOAT32_C(  -261.83), EASYSIMD_FLOAT32_C(   -32.31), EASYSIMD_FLOAT32_C(   674.39) } },
    { { EASYSIMD_FLOAT32_C(  -222.95), EASYSIMD_FLOAT32_C(  -329.16), EASYSIMD_FLOAT32_C(   332.77), EASYSIMD_FLOAT32_C(  -656.05) },
      UINT8_C(172),
      { -INT32_C(  1368556090),  INT32_C(  1551151254),  INT32_C(  1238021452), -INT32_C(   954668010) },
      { EASYSIMD_FLOAT32_C(  -222.95), EASYSIMD_FLOAT32_C(  -329.16), EASYSIMD_FLOAT32_C(1238021504.00), EASYSIMD_FLOAT32_C(-954668032.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtepi32_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtepi32_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128 r = easysimd_mm_mask_cvtepi32_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtepi32_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(216),
      { -INT32_C(  2010755554),  INT32_C(    66228366), -INT32_C(   694179544),  INT32_C(  1880811831) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(1880811776.00) } },
    { UINT8_C( 28),
      {  INT32_C(  1624243879),  INT32_C(  1822969663), -INT32_C(   185193687),  INT32_C(  1678954262) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-185193680.00), EASYSIMD_FLOAT32_C(1678954240.00) } },
    { UINT8_C(245),
      { -INT32_C(  1920601190), -INT32_C(  1791578634), -INT32_C(  1652658813), -INT32_C(   616195363) },
      { EASYSIMD_FLOAT32_C(-1920601216.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-1652658816.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(105),
      {  INT32_C(  1303976869),  INT32_C(  2121917063), -INT32_C(   397570090), -INT32_C(   427605261) },
      { EASYSIMD_FLOAT32_C(1303976832.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-427605248.00) } },
    { UINT8_C(199),
      {  INT32_C(  1182063888), -INT32_C(   239863950), -INT32_C(   645164908), -INT32_C(  2038434453) },
      { EASYSIMD_FLOAT32_C(1182063872.00), EASYSIMD_FLOAT32_C(-239863952.00), EASYSIMD_FLOAT32_C(-645164928.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(174),
      {  INT32_C(  1167068876), -INT32_C(   640391540), -INT32_C(   786708402), -INT32_C(  1948130642) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-640391552.00), EASYSIMD_FLOAT32_C(-786708416.00), EASYSIMD_FLOAT32_C(-1948130688.00) } },
    { UINT8_C( 86),
      { -INT32_C(   632422873),  INT32_C(  2053825007),  INT32_C(   980407995),  INT32_C(  1778785628) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(2053825024.00), EASYSIMD_FLOAT32_C(980408000.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(173),
      {  INT32_C(   538179148), -INT32_C(   353869105),  INT32_C(   365729587),  INT32_C(   540811810) },
      { EASYSIMD_FLOAT32_C(538179136.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(365729600.00), EASYSIMD_FLOAT32_C(540811840.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtepi32_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtepi32_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128 r = easysimd_mm_maskz_cvtepi32_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi32_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const int32_t a[4];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -637.77), EASYSIMD_FLOAT64_C(  -762.06) },
      UINT8_C( 42),
      {  INT32_C(   739896809), -INT32_C(  1004122556),  INT32_C(  1498750877), -INT32_C(   461669930) },
      { EASYSIMD_FLOAT64_C(  -637.77), EASYSIMD_FLOAT64_C(-1004122556.00) } },
    { { EASYSIMD_FLOAT64_C(  -323.14), EASYSIMD_FLOAT64_C(  -770.25) },
      UINT8_C( 28),
      {  INT32_C(   879199555),  INT32_C(   905852137), -INT32_C(  1126240878),  INT32_C(    47566038) },
      { EASYSIMD_FLOAT64_C(  -323.14), EASYSIMD_FLOAT64_C(  -770.25) } },
    { { EASYSIMD_FLOAT64_C(  -158.78), EASYSIMD_FLOAT64_C(  -586.91) },
      UINT8_C( 40),
      {  INT32_C(   692568532), -INT32_C(  1767694822),  INT32_C(   515261954), -INT32_C(  1098564727) },
      { EASYSIMD_FLOAT64_C(  -158.78), EASYSIMD_FLOAT64_C(  -586.91) } },
    { { EASYSIMD_FLOAT64_C(   454.07), EASYSIMD_FLOAT64_C(   249.82) },
      UINT8_C(188),
      {  INT32_C(   950619225), -INT32_C(   586907896), -INT32_C(  1459547007),  INT32_C(   317768424) },
      { EASYSIMD_FLOAT64_C(   454.07), EASYSIMD_FLOAT64_C(   249.82) } },
    { { EASYSIMD_FLOAT64_C(   777.20), EASYSIMD_FLOAT64_C(  -710.83) },
      UINT8_C(181),
      {  INT32_C(   687542642), -INT32_C(  1251769297), -INT32_C(    26245054),  INT32_C(   430423009) },
      { EASYSIMD_FLOAT64_C(687542642.00), EASYSIMD_FLOAT64_C(  -710.83) } },
    { { EASYSIMD_FLOAT64_C(   514.27), EASYSIMD_FLOAT64_C(   650.40) },
      UINT8_C( 29),
      { -INT32_C(  1590581344),  INT32_C(  1097013840), -INT32_C(   514899924), -INT32_C(   606314318) },
      { EASYSIMD_FLOAT64_C(-1590581344.00), EASYSIMD_FLOAT64_C(   650.40) } },
    { { EASYSIMD_FLOAT64_C(   -32.03), EASYSIMD_FLOAT64_C(   402.33) },
      UINT8_C( 62),
      { -INT32_C(  1262050491),  INT32_C(  1232054433), -INT32_C(   546360639),  INT32_C(  1947211730) },
      { EASYSIMD_FLOAT64_C(   -32.03), EASYSIMD_FLOAT64_C(1232054433.00) } },
    { { EASYSIMD_FLOAT64_C(   543.52), EASYSIMD_FLOAT64_C(   -16.53) },
      UINT8_C(215),
      { -INT32_C(   132688216),  INT32_C(   324586039), -INT32_C(   478877531), -INT32_C(   508946643) },
      { EASYSIMD_FLOAT64_C(-132688216.00), EASYSIMD_FLOAT64_C(324586039.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtepi32_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtepi32_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128d r = easysimd_mm_mask_cvtepi32_pd(src, k, a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtepi32_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(135),
      {  INT32_C(   452902768),  INT32_C(  1795586996),  INT32_C(   931930437), -INT32_C(  1855723566) },
      { EASYSIMD_FLOAT64_C(452902768.00), EASYSIMD_FLOAT64_C(1795586996.00) } },
    { UINT8_C(176),
      {  INT32_C(   508430471), -INT32_C(  1073330377),  INT32_C(   282900827), -INT32_C(  1367317522) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 97),
      { -INT32_C(  1596169574),  INT32_C(  1522606541), -INT32_C(  1153330344),  INT32_C(   759427365) },
      { EASYSIMD_FLOAT64_C(-1596169574.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 63),
      {  INT32_C(  1753638241),  INT32_C(    18997541),  INT32_C(  1902383345),  INT32_C(   537708222) },
      { EASYSIMD_FLOAT64_C(1753638241.00), EASYSIMD_FLOAT64_C(18997541.00) } },
    { UINT8_C(162),
      {  INT32_C(  1858399916), -INT32_C(  1996415928),  INT32_C(   460990168), -INT32_C(  1183008428) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-1996415928.00) } },
    { UINT8_C( 63),
      {  INT32_C(   102883044),  INT32_C(  1142297312), -INT32_C(  1878338428), -INT32_C(   499339788) },
      { EASYSIMD_FLOAT64_C(102883044.00), EASYSIMD_FLOAT64_C(1142297312.00) } },
    { UINT8_C(113),
      { -INT32_C(  1400034646),  INT32_C(   768763315), -INT32_C(    35248512), -INT32_C(  1092540705) },
      { EASYSIMD_FLOAT64_C(-1400034646.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 71),
      { -INT32_C(    44458265), -INT32_C(   305013278), -INT32_C(  1432697235), -INT32_C(   749466712) },
      { EASYSIMD_FLOAT64_C(-44458265.00), EASYSIMD_FLOAT64_C(-305013278.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtepi32_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtepi32_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128d r = easysimd_mm_maskz_cvtepi32_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepu32_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const uint32_t a[4];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   439.93), EASYSIMD_FLOAT64_C(   262.91) },
      UINT8_C( 11),
      { UINT32_C(3798881670), UINT32_C(    994584), UINT32_C(2746279068), UINT32_C( 391445202) },
      { EASYSIMD_FLOAT64_C(3798881670.00), EASYSIMD_FLOAT64_C(994584.00) } },
    { { EASYSIMD_FLOAT64_C(  -591.51), EASYSIMD_FLOAT64_C(   -45.38) },
      UINT8_C(104),
      { UINT32_C(3684264098), UINT32_C(1205720945), UINT32_C(1042229555), UINT32_C(2443998383) },
      { EASYSIMD_FLOAT64_C(  -591.51), EASYSIMD_FLOAT64_C(   -45.38) } },
    { { EASYSIMD_FLOAT64_C(  -559.10), EASYSIMD_FLOAT64_C(  -480.15) },
      UINT8_C(160),
      { UINT32_C(1032091020), UINT32_C(1832343320), UINT32_C(3616222318), UINT32_C(2859529167) },
      { EASYSIMD_FLOAT64_C(  -559.10), EASYSIMD_FLOAT64_C(  -480.15) } },
    { { EASYSIMD_FLOAT64_C(   931.69), EASYSIMD_FLOAT64_C(     8.36) },
      UINT8_C(135),
      { UINT32_C(3417406380), UINT32_C(1698652345), UINT32_C(2453589234), UINT32_C(2501358680) },
      { EASYSIMD_FLOAT64_C(3417406380.00), EASYSIMD_FLOAT64_C(1698652345.00) } },
    { { EASYSIMD_FLOAT64_C(   314.38), EASYSIMD_FLOAT64_C(  -518.91) },
      UINT8_C(205),
      { UINT32_C(3321486393), UINT32_C( 616089779), UINT32_C(4201389427), UINT32_C(2494293961) },
      { EASYSIMD_FLOAT64_C(3321486393.00), EASYSIMD_FLOAT64_C(  -518.91) } },
    { { EASYSIMD_FLOAT64_C(   483.10), EASYSIMD_FLOAT64_C(   296.02) },
      UINT8_C(211),
      { UINT32_C(1134558725), UINT32_C(2818046096), UINT32_C(1511376013), UINT32_C(3260347133) },
      { EASYSIMD_FLOAT64_C(1134558725.00), EASYSIMD_FLOAT64_C(2818046096.00) } },
    { { EASYSIMD_FLOAT64_C(   995.63), EASYSIMD_FLOAT64_C(  -944.20) },
      UINT8_C(123),
      { UINT32_C( 899190729), UINT32_C( 891052426), UINT32_C(3359751413), UINT32_C(  90718145) },
      { EASYSIMD_FLOAT64_C(899190729.00), EASYSIMD_FLOAT64_C(891052426.00) } },
    { { EASYSIMD_FLOAT64_C(  -198.79), EASYSIMD_FLOAT64_C(   583.50) },
      UINT8_C(252),
      { UINT32_C(2378231159), UINT32_C(2625617480), UINT32_C(4223149184), UINT32_C( 596855022) },
      { EASYSIMD_FLOAT64_C(  -198.79), EASYSIMD_FLOAT64_C(   583.50) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtepu32_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtepu32_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128d r = easysimd_mm_mask_cvtepu32_pd(src, k, a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtepu32_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint32_t a[4];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(152),
      { UINT32_C(3556673024), UINT32_C(2600042164), UINT32_C(1572919675), UINT32_C( 893650839) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 97),
      { UINT32_C( 322836290), UINT32_C(2665065476), UINT32_C(2863235670), UINT32_C(3685433941) },
      { EASYSIMD_FLOAT64_C(322836290.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 64),
      { UINT32_C(2008977278), UINT32_C(3931126314), UINT32_C(3690129303), UINT32_C( 874337052) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(145),
      { UINT32_C( 171784241), UINT32_C(2156696535), UINT32_C(3921862974), UINT32_C(1281819325) },
      { EASYSIMD_FLOAT64_C(171784241.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(193),
      { UINT32_C( 805009118), UINT32_C(1393988193), UINT32_C(2342925165), UINT32_C(2679912295) },
      { EASYSIMD_FLOAT64_C(805009118.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(117),
      { UINT32_C(1393063622), UINT32_C(3107210999), UINT32_C(2528964911), UINT32_C(3027598654) },
      { EASYSIMD_FLOAT64_C(1393063622.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(120),
      { UINT32_C(3121223076), UINT32_C( 250443624), UINT32_C(3192280066), UINT32_C(1787149300) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(195),
      { UINT32_C( 202007000), UINT32_C(3623762203), UINT32_C(1163148496), UINT32_C( 115985648) },
      { EASYSIMD_FLOAT64_C(202007000.00), EASYSIMD_FLOAT64_C(3623762203.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtepu32_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtepu32_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128d r = easysimd_mm_maskz_cvtepu32_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cvtepi64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { -INT64_C(  479843983702231798),  INT64_C( 1105642441621960780) },
      { EASYSIMD_FLOAT32_C(-479843985511153664.00), EASYSIMD_FLOAT32_C(1105642435893002240.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { -INT64_C( 3579979013088449657), -INT64_C( 6774133373070644962) },
      { EASYSIMD_FLOAT32_C(-3579979073713078272.00), EASYSIMD_FLOAT32_C(-6774133469925605376.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { {  INT64_C( 6687672891335298918), -INT64_C( 3000513015525062565) },
      { EASYSIMD_FLOAT32_C(6687672823319625728.00), EASYSIMD_FLOAT32_C(-3000513081253036032.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { {  INT64_C( 6363943866058727028),  INT64_C( 3091127147073721006) },
      { EASYSIMD_FLOAT32_C(6363943614753538048.00), EASYSIMD_FLOAT32_C(3091127133032939520.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { -INT64_C( 5684158087942835534),  INT64_C( 5421589407633165848) },
      { EASYSIMD_FLOAT32_C(-5684157906497306624.00), EASYSIMD_FLOAT32_C(5421589581982072832.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { {  INT64_C( 2651247399247163704),  INT64_C( 5070858103900713329) },
      { EASYSIMD_FLOAT32_C(2651247440253943808.00), EASYSIMD_FLOAT32_C(5070858017105248256.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { {  INT64_C( 5136114786350096192), -INT64_C( 2947908735006259725) },
      { EASYSIMD_FLOAT32_C(5136114581969567744.00), EASYSIMD_FLOAT32_C(-2947908871567441920.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { {  INT64_C( 8482554102780978205), -INT64_C( 5719279673604955705) },
      { EASYSIMD_FLOAT32_C(8482554087408140288.00), EASYSIMD_FLOAT32_C(-5719279606423355392.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cvtepi64_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128 r = easysimd_mm_cvtepi64_ps(a);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const int64_t a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   349.15), EASYSIMD_FLOAT32_C(  -982.36), EASYSIMD_FLOAT32_C(  -860.75), EASYSIMD_FLOAT32_C(   413.74) },
      UINT8_C(115),
      {  INT64_C( 7411147672697789106),  INT64_C( 3707307549011064398) },
      { EASYSIMD_FLOAT32_C(7411147626105536512.00), EASYSIMD_FLOAT32_C(3707307468011864064.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(    27.04), EASYSIMD_FLOAT32_C(   158.78), EASYSIMD_FLOAT32_C(   324.18), EASYSIMD_FLOAT32_C(  -490.14) },
      UINT8_C(250),
      { -INT64_C( 7531297091986421513), -INT64_C( 4970528077218386143) },
      { EASYSIMD_FLOAT32_C(    27.04), EASYSIMD_FLOAT32_C(-4970528130826502144.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -633.56), EASYSIMD_FLOAT32_C(  -231.84), EASYSIMD_FLOAT32_C(   790.76), EASYSIMD_FLOAT32_C(  -107.34) },
      UINT8_C( 89),
      { -INT64_C( 1289822275736353308), -INT64_C( 4594961024785509504) },
      { EASYSIMD_FLOAT32_C(-1289822247137050624.00), EASYSIMD_FLOAT32_C(  -231.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   323.44), EASYSIMD_FLOAT32_C(   920.10), EASYSIMD_FLOAT32_C(   -32.93), EASYSIMD_FLOAT32_C(  -271.85) },
      UINT8_C( 28),
      { -INT64_C( 6427414446531878308),  INT64_C( 2427125189969585903) },
      { EASYSIMD_FLOAT32_C(   323.44), EASYSIMD_FLOAT32_C(   920.10), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   593.71), EASYSIMD_FLOAT32_C(  -279.84), EASYSIMD_FLOAT32_C(   377.30), EASYSIMD_FLOAT32_C(  -196.83) },
      UINT8_C(129),
      { -INT64_C(  452660034142381911),  INT64_C( 5776350391177761706) },
      { EASYSIMD_FLOAT32_C(-452660037983141888.00), EASYSIMD_FLOAT32_C(  -279.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   -91.52), EASYSIMD_FLOAT32_C(  -144.77), EASYSIMD_FLOAT32_C(   317.16), EASYSIMD_FLOAT32_C(  -422.36) },
      UINT8_C( 89),
      { -INT64_C( 1243498941216314252),  INT64_C(  122296675201400897) },
      { EASYSIMD_FLOAT32_C(-1243498997625126912.00), EASYSIMD_FLOAT32_C(  -144.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   299.71), EASYSIMD_FLOAT32_C(   993.19), EASYSIMD_FLOAT32_C(   702.82), EASYSIMD_FLOAT32_C(   148.93) },
      UINT8_C(  2),
      { -INT64_C( 6255711631272437193), -INT64_C(  692357240352503573) },
      { EASYSIMD_FLOAT32_C(   299.71), EASYSIMD_FLOAT32_C(-692357249330315264.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   387.58), EASYSIMD_FLOAT32_C(   755.74), EASYSIMD_FLOAT32_C(   162.29), EASYSIMD_FLOAT32_C(   -47.76) },
      UINT8_C(234),
      {  INT64_C( 1743061751444171985),  INT64_C( 3108521599071892127) },
      { EASYSIMD_FLOAT32_C(   387.58), EASYSIMD_FLOAT32_C(3108521681862262784.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtepi64_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128 r = easysimd_mm_mask_cvtepi64_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtepi64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C( 45),
      {  INT64_C( 2950030178485350700),  INT64_C( 6099143655089055809) },
      { EASYSIMD_FLOAT32_C(2950030104375328768.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(197),
      { -INT64_C( 2767368983688347996), -INT64_C( 7984885029793742328) },
      { EASYSIMD_FLOAT32_C(-2767369062286622720.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 74),
      {  INT64_C( 3785601540872888944),  INT64_C( 3630881740831481627) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(3630881789154689024.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 77),
      {  INT64_C( 7893590364402074800), -INT64_C( 2428649129981056340) },
      { EASYSIMD_FLOAT32_C(7893590238629462016.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(155),
      {  INT64_C( 6022683368494426540),  INT64_C( 4589479706097502365) },
      { EASYSIMD_FLOAT32_C(6022683243038375936.00), EASYSIMD_FLOAT32_C(4589479731837009920.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(189),
      { -INT64_C( 5517169812030628615),  INT64_C( 3595565511010216687) },
      { EASYSIMD_FLOAT32_C(-5517169578028826624.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(198),
      { -INT64_C( 1155101691467215333), -INT64_C( 8147635479936011454) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-8147635698761990144.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 47),
      { -INT64_C(  879554825388107663), -INT64_C( 7823107947009502356) },
      { EASYSIMD_FLOAT32_C(-879554807675748352.00), EASYSIMD_FLOAT32_C(-7823107694998323200.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtepi64_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128 r = easysimd_mm_maskz_cvtepi64_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cvtepu64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { UINT64_C( 9936247095012171360), UINT64_C( 3938057873639856303) },
      { EASYSIMD_FLOAT32_C(9936246997793112064.00), EASYSIMD_FLOAT32_C(3938057850164609024.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT64_C( 9665254877367364002), UINT64_C( 3805057712020592501) },
      { EASYSIMD_FLOAT32_C(9665254764941672448.00), EASYSIMD_FLOAT32_C(3805057625133940736.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT64_C(15493995871293834085), UINT64_C(12434145045996519571) },
      { EASYSIMD_FLOAT32_C(15493995701712453632.00), EASYSIMD_FLOAT32_C(12434145001565323264.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT64_C(18266595175052256734), UINT64_C(  295836384843225106) },
      { EASYSIMD_FLOAT32_C(18266595690568220672.00), EASYSIMD_FLOAT32_C(295836385275805696.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT64_C( 4577773423475814792), UINT64_C(12140621347992098245) },
      { EASYSIMD_FLOAT32_C(4577773506413985792.00), EASYSIMD_FLOAT32_C(12140621576437497856.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT64_C( 7156255341559240395), UINT64_C( 1520920955823228792) },
      { EASYSIMD_FLOAT32_C(7156255542019620864.00), EASYSIMD_FLOAT32_C(1520920899597893632.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT64_C(16447792485201451606), UINT64_C(14669683042246973986) },
      { EASYSIMD_FLOAT32_C(16447792351994183680.00), EASYSIMD_FLOAT32_C(14669683038275764224.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT64_C(14860311239700919941), UINT64_C( 4582413623789716726) },
      { EASYSIMD_FLOAT32_C(14860310866741428224.00), EASYSIMD_FLOAT32_C(4582413720361107456.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cvtepu64_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128 r = easysimd_mm_cvtepu64_ps(a);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepu64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const uint64_t a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   650.12), EASYSIMD_FLOAT32_C(   616.27), EASYSIMD_FLOAT32_C(   231.83), EASYSIMD_FLOAT32_C(   918.44) },
      UINT8_C(235),
      { UINT64_C( 6189399573697079342), UINT64_C(14862629967973830505) },
      { EASYSIMD_FLOAT32_C(6189399442378981376.00), EASYSIMD_FLOAT32_C(14862629736764407808.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(    -9.38), EASYSIMD_FLOAT32_C(   490.88), EASYSIMD_FLOAT32_C(  -293.26), EASYSIMD_FLOAT32_C(   558.68) },
      UINT8_C(169),
      { UINT64_C( 4258212977189648679), UINT64_C( 6477017591417799580) },
      { EASYSIMD_FLOAT32_C(4258213096184610816.00), EASYSIMD_FLOAT32_C(   490.88), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   750.16), EASYSIMD_FLOAT32_C(   -12.99), EASYSIMD_FLOAT32_C(    45.66), EASYSIMD_FLOAT32_C(  -668.84) },
      UINT8_C( 95),
      { UINT64_C( 2609244132636026776), UINT64_C( 7858665041699790578) },
      { EASYSIMD_FLOAT32_C(2609244171927552000.00), EASYSIMD_FLOAT32_C(7858664801528971264.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -340.56), EASYSIMD_FLOAT32_C(    32.08), EASYSIMD_FLOAT32_C(   671.14), EASYSIMD_FLOAT32_C(   -44.50) },
      UINT8_C(181),
      { UINT64_C(18152202748608831694), UINT64_C( 4027465266023843910) },
      { EASYSIMD_FLOAT32_C(18152202500814405632.00), EASYSIMD_FLOAT32_C(    32.08), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   226.29), EASYSIMD_FLOAT32_C(   339.39), EASYSIMD_FLOAT32_C(   715.99), EASYSIMD_FLOAT32_C(  -913.03) },
      UINT8_C( 83),
      { UINT64_C( 3489894050837477396), UINT64_C( 7666164320798909258) },
      { EASYSIMD_FLOAT32_C(3489894161904041984.00), EASYSIMD_FLOAT32_C(7666164504761204736.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -771.95), EASYSIMD_FLOAT32_C(   184.38), EASYSIMD_FLOAT32_C(   653.51), EASYSIMD_FLOAT32_C(  -464.62) },
      UINT8_C(242),
      { UINT64_C(10577270427600564958), UINT64_C(11557303205505670788) },
      { EASYSIMD_FLOAT32_C(  -771.95), EASYSIMD_FLOAT32_C(11557303170111635456.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   927.02), EASYSIMD_FLOAT32_C(  -133.95), EASYSIMD_FLOAT32_C(  -818.40), EASYSIMD_FLOAT32_C(  -769.54) },
      UINT8_C(145),
      { UINT64_C( 5536852053642522900), UINT64_C( 3122944151644719290) },
      { EASYSIMD_FLOAT32_C(5536851935677644800.00), EASYSIMD_FLOAT32_C(  -133.95), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   860.73), EASYSIMD_FLOAT32_C(  -413.57), EASYSIMD_FLOAT32_C(   904.45), EASYSIMD_FLOAT32_C(  -281.99) },
      UINT8_C(143),
      { UINT64_C( 7557763096364309996), UINT64_C(10284801757808522477) },
      { EASYSIMD_FLOAT32_C(7557763103622955008.00), EASYSIMD_FLOAT32_C(10284802079402754048.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtepu64_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128 r = easysimd_mm_mask_cvtepu64_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtepu64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(124),
      { UINT64_C(17156235411192052746), UINT64_C( 7238213685335398743) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 41),
      { UINT64_C( 3202956202250151255), UINT64_C(15455571457345377073) },
      { EASYSIMD_FLOAT32_C(3202956261670780928.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 76),
      { UINT64_C(14025569305643256820), UINT64_C( 3378041705303286806) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(120),
      { UINT64_C(14506896001696522251), UINT64_C(10213405853571263564) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(204),
      { UINT64_C( 6273715806910056136), UINT64_C( 2822920543518267712) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(175),
      { UINT64_C(   47803462704444587), UINT64_C( 4549227952688999325) },
      { EASYSIMD_FLOAT32_C(47803462745849856.00), EASYSIMD_FLOAT32_C(4549227985533665280.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(187),
      { UINT64_C( 7978323413409368206), UINT64_C( 6605695598609235924) },
      { EASYSIMD_FLOAT32_C(7978323552468205568.00), EASYSIMD_FLOAT32_C(6605695435375902720.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(173),
      { UINT64_C(12069398895378127274), UINT64_C(18314025833058154360) },
      { EASYSIMD_FLOAT32_C(12069398511725051904.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtepu64_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128 r = easysimd_mm_maskz_cvtepu64_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cvtepi64_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { -INT64_C( 8614850050345457108),  INT64_C( 7285479202649351501) },
      { EASYSIMD_FLOAT64_C(-8614850050345456640.00), EASYSIMD_FLOAT64_C(7285479202649351168.00) } },
    { { -INT64_C( 1053323752731789121),  INT64_C( 4215849800157546739) },
      { EASYSIMD_FLOAT64_C(-1053323752731789184.00), EASYSIMD_FLOAT64_C(4215849800157546496.00) } },
    { { -INT64_C( 7066862152626902817),  INT64_C( 5673917116813760156) },
      { EASYSIMD_FLOAT64_C(-7066862152626903040.00), EASYSIMD_FLOAT64_C(5673917116813760512.00) } },
    { {  INT64_C( 8653485836458438277),  INT64_C( 8460994408982831395) },
      { EASYSIMD_FLOAT64_C(8653485836458438656.00), EASYSIMD_FLOAT64_C(8460994408982831104.00) } },
    { { -INT64_C( 8947589338295078682), -INT64_C( 2884106609811028963) },
      { EASYSIMD_FLOAT64_C(-8947589338295078912.00), EASYSIMD_FLOAT64_C(-2884106609811028992.00) } },
    { {  INT64_C( 8305785945735427142), -INT64_C( 6629280240916741899) },
      { EASYSIMD_FLOAT64_C(8305785945735427072.00), EASYSIMD_FLOAT64_C(-6629280240916742144.00) } },
    { { -INT64_C( 4761802751542109824), -INT64_C( 6603857745514281679) },
      { EASYSIMD_FLOAT64_C(-4761802751542110208.00), EASYSIMD_FLOAT64_C(-6603857745514281984.00) } },
    { {  INT64_C( 6924482801512240982),  INT64_C( 5572900857620184564) },
      { EASYSIMD_FLOAT64_C(6924482801512241152.00), EASYSIMD_FLOAT64_C(5572900857620184064.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cvtepi64_pd(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cvtepi64_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128d r = easysimd_mm_cvtepi64_pd(a);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi64_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -473.83), EASYSIMD_FLOAT64_C(  -406.96) },
      UINT8_C(196),
      { -INT64_C( 1992989217109780021), -INT64_C( 6433624543855457372) },
      { EASYSIMD_FLOAT64_C(  -473.83), EASYSIMD_FLOAT64_C(  -406.96) } },
    { { EASYSIMD_FLOAT64_C(  -513.89), EASYSIMD_FLOAT64_C(  -815.78) },
      UINT8_C( 96),
      {  INT64_C( 2086762660172277667), -INT64_C( 9170721129076888069) },
      { EASYSIMD_FLOAT64_C(  -513.89), EASYSIMD_FLOAT64_C(  -815.78) } },
    { { EASYSIMD_FLOAT64_C(  -542.00), EASYSIMD_FLOAT64_C(  -263.18) },
      UINT8_C(215),
      { -INT64_C(  126888838459893298), -INT64_C(  984278822482543143) },
      { EASYSIMD_FLOAT64_C(-126888838459893296.00), EASYSIMD_FLOAT64_C(-984278822482543104.00) } },
    { { EASYSIMD_FLOAT64_C(   864.63), EASYSIMD_FLOAT64_C(   848.40) },
      UINT8_C(231),
      {  INT64_C( 8652977002587160277), -INT64_C( 2340033633719550196) },
      { EASYSIMD_FLOAT64_C(8652977002587160576.00), EASYSIMD_FLOAT64_C(-2340033633719549952.00) } },
    { { EASYSIMD_FLOAT64_C(   533.58), EASYSIMD_FLOAT64_C(   829.96) },
      UINT8_C( 28),
      {  INT64_C( 2588366695741100747), -INT64_C( 9049943346622585684) },
      { EASYSIMD_FLOAT64_C(   533.58), EASYSIMD_FLOAT64_C(   829.96) } },
    { { EASYSIMD_FLOAT64_C(   318.23), EASYSIMD_FLOAT64_C(    13.31) },
      UINT8_C(151),
      {  INT64_C( 7677904969774072153), -INT64_C( 1262779044074398908) },
      { EASYSIMD_FLOAT64_C(7677904969774071808.00), EASYSIMD_FLOAT64_C(-1262779044074398976.00) } },
    { { EASYSIMD_FLOAT64_C(   -96.33), EASYSIMD_FLOAT64_C(  -858.32) },
      UINT8_C(218),
      { -INT64_C( 2670924357076853084),  INT64_C( 3362164892836482938) },
      { EASYSIMD_FLOAT64_C(   -96.33), EASYSIMD_FLOAT64_C(3362164892836483072.00) } },
    { { EASYSIMD_FLOAT64_C(   450.36), EASYSIMD_FLOAT64_C(  -203.41) },
      UINT8_C(188),
      {  INT64_C( 3096011100367835422), -INT64_C( 1836277756508407764) },
      { EASYSIMD_FLOAT64_C(   450.36), EASYSIMD_FLOAT64_C(  -203.41) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtepi64_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtepi64_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128d r = easysimd_mm_mask_cvtepi64_pd(src, k, a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtepi64_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(113),
      { -INT64_C( 6355182479456762882),  INT64_C( 8509915296800176537) },
      { EASYSIMD_FLOAT64_C(-6355182479456762880.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 34),
      { -INT64_C( 6715931139535670978),  INT64_C( 1737937097877344696) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(1737937097877344768.00) } },
    { UINT8_C(149),
      {  INT64_C( 7109507003511295203), -INT64_C( 1053896188309501688) },
      { EASYSIMD_FLOAT64_C(7109507003511294976.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(204),
      { -INT64_C( 4825399424035051413),  INT64_C(  206804629430836700) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 40),
      {  INT64_C(  414958266262681231), -INT64_C( 2583200638144560549) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(164),
      { -INT64_C( 8934229738644199843), -INT64_C( 5573657829329618632) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(149),
      { -INT64_C( 2061475383289640225), -INT64_C( 7320274690666981146) },
      { EASYSIMD_FLOAT64_C(-2061475383289640192.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(137),
      { -INT64_C( 4788171370732782129),  INT64_C(  404117779594102806) },
      { EASYSIMD_FLOAT64_C(-4788171370732782592.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtepi64_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtepi64_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128d r = easysimd_mm_maskz_cvtepi64_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cvtepu64_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { UINT64_C(12129835160146142399), UINT64_C(  415701853865018694) },
      { EASYSIMD_FLOAT64_C(12129835160146143232.00), EASYSIMD_FLOAT64_C(415701853865018688.00) } },
    { { UINT64_C(11701328859157263148), UINT64_C( 5439218652176141863) },
      { EASYSIMD_FLOAT64_C(11701328859157262336.00), EASYSIMD_FLOAT64_C(5439218652176142336.00) } },
    { { UINT64_C( 2475115971179383009), UINT64_C( 8974858565589561829) },
      { EASYSIMD_FLOAT64_C(2475115971179382784.00), EASYSIMD_FLOAT64_C(8974858565589561344.00) } },
    { { UINT64_C( 5028763889406223757), UINT64_C( 5397144593534250256) },
      { EASYSIMD_FLOAT64_C(5028763889406223360.00), EASYSIMD_FLOAT64_C(5397144593534249984.00) } },
    { { UINT64_C( 4001540981810524554), UINT64_C( 9624558507067956933) },
      { EASYSIMD_FLOAT64_C(4001540981810524672.00), EASYSIMD_FLOAT64_C(9624558507067957248.00) } },
    { { UINT64_C(14002666173547475185), UINT64_C( 7392453026316638947) },
      { EASYSIMD_FLOAT64_C(14002666173547474944.00), EASYSIMD_FLOAT64_C(7392453026316639232.00) } },
    { { UINT64_C(12375703650134954549), UINT64_C(13524400709247794738) },
      { EASYSIMD_FLOAT64_C(12375703650134953984.00), EASYSIMD_FLOAT64_C(13524400709247795200.00) } },
    { { UINT64_C(13093685758513470530), UINT64_C(10212643709661615223) },
      { EASYSIMD_FLOAT64_C(13093685758513471488.00), EASYSIMD_FLOAT64_C(10212643709661616128.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cvtepu64_pd(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cvtepu64_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128d r = easysimd_mm_cvtepu64_pd(a);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepu64_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const uint64_t a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -70.88), EASYSIMD_FLOAT64_C(  -983.27) },
      UINT8_C(141),
      { UINT64_C( 1342140673989284870), UINT64_C( 6114805528453016844) },
      { EASYSIMD_FLOAT64_C(1342140673989284864.00), EASYSIMD_FLOAT64_C(  -983.27) } },
    { { EASYSIMD_FLOAT64_C(   757.46), EASYSIMD_FLOAT64_C(  -237.70) },
      UINT8_C(  3),
      { UINT64_C(12371934396181641962), UINT64_C( 4195297921930704366) },
      { EASYSIMD_FLOAT64_C(12371934396181641216.00), EASYSIMD_FLOAT64_C(4195297921930704384.00) } },
    { { EASYSIMD_FLOAT64_C(  -479.64), EASYSIMD_FLOAT64_C(  -398.25) },
      UINT8_C(218),
      { UINT64_C( 6901909930786652567), UINT64_C(15181698248034235606) },
      { EASYSIMD_FLOAT64_C(  -479.64), EASYSIMD_FLOAT64_C(15181698248034236416.00) } },
    { { EASYSIMD_FLOAT64_C(  -228.43), EASYSIMD_FLOAT64_C(   -78.14) },
      UINT8_C(132),
      { UINT64_C( 4898777349031038893), UINT64_C(13281178072225710126) },
      { EASYSIMD_FLOAT64_C(  -228.43), EASYSIMD_FLOAT64_C(   -78.14) } },
    { { EASYSIMD_FLOAT64_C(   675.75), EASYSIMD_FLOAT64_C(  -757.80) },
      UINT8_C(128),
      { UINT64_C( 3717391103573666075), UINT64_C(10440017138710517925) },
      { EASYSIMD_FLOAT64_C(   675.75), EASYSIMD_FLOAT64_C(  -757.80) } },
    { { EASYSIMD_FLOAT64_C(   369.02), EASYSIMD_FLOAT64_C(   747.41) },
      UINT8_C(140),
      { UINT64_C( 6819333905096377778), UINT64_C(15425070042323181532) },
      { EASYSIMD_FLOAT64_C(   369.02), EASYSIMD_FLOAT64_C(   747.41) } },
    { { EASYSIMD_FLOAT64_C(   797.63), EASYSIMD_FLOAT64_C(  -932.55) },
      UINT8_C(108),
      { UINT64_C(10793521587023022481), UINT64_C( 6157389271900305883) },
      { EASYSIMD_FLOAT64_C(   797.63), EASYSIMD_FLOAT64_C(  -932.55) } },
    { { EASYSIMD_FLOAT64_C(   356.72), EASYSIMD_FLOAT64_C(   695.51) },
      UINT8_C(248),
      { UINT64_C(16206715421298488285), UINT64_C( 4546468319053694887) },
      { EASYSIMD_FLOAT64_C(   356.72), EASYSIMD_FLOAT64_C(   695.51) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtepu64_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtepu64_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128d r = easysimd_mm_mask_cvtepu64_pd(src, k, a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtepu64_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(214),
      { UINT64_C(14115777107301592739), UINT64_C(11919692601554295054) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(11919692601554294784.00) } },
    { UINT8_C( 75),
      { UINT64_C(16983731253784461432), UINT64_C( 1668176466891219610) },
      { EASYSIMD_FLOAT64_C(16983731253784461312.00), EASYSIMD_FLOAT64_C(1668176466891219712.00) } },
    { UINT8_C(219),
      { UINT64_C(15381544625792324791), UINT64_C( 2355490770512165581) },
      { EASYSIMD_FLOAT64_C(15381544625792325632.00), EASYSIMD_FLOAT64_C(2355490770512165376.00) } },
    { UINT8_C(231),
      { UINT64_C( 6592098478782724134), UINT64_C(14859560151553680217) },
      { EASYSIMD_FLOAT64_C(6592098478782724096.00), EASYSIMD_FLOAT64_C(14859560151553679360.00) } },
    { UINT8_C(238),
      { UINT64_C( 1226656928863102420), UINT64_C( 4396738667229701421) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(4396738667229701632.00) } },
    { UINT8_C(150),
      { UINT64_C( 2628911670473056733), UINT64_C(12272263374110766660) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(12272263374110767104.00) } },
    { UINT8_C(254),
      { UINT64_C( 2883466725380301978), UINT64_C(12168753436191525375) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(12168753436191524864.00) } },
    { UINT8_C(132),
      { UINT64_C(17323815084669658168), UINT64_C( 4464128913786269193) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtepu64_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtepu64_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128d r = easysimd_mm_maskz_cvtepu64_pd(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtpd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   533.13), EASYSIMD_FLOAT32_C(   402.26), EASYSIMD_FLOAT32_C(  -882.17), EASYSIMD_FLOAT32_C(  -349.83) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT64_C(  -389.40), EASYSIMD_FLOAT64_C(  -176.66), EASYSIMD_FLOAT64_C(   802.72), EASYSIMD_FLOAT64_C(   691.69) },
      { EASYSIMD_FLOAT32_C(   533.13), EASYSIMD_FLOAT32_C(  -176.66), EASYSIMD_FLOAT32_C(  -882.17), EASYSIMD_FLOAT32_C(   691.69) } },
    { { EASYSIMD_FLOAT32_C(  -708.06), EASYSIMD_FLOAT32_C(  -802.37), EASYSIMD_FLOAT32_C(  -868.11), EASYSIMD_FLOAT32_C(  -866.66) },
      UINT8_C(120),
      { EASYSIMD_FLOAT64_C(   -69.79), EASYSIMD_FLOAT64_C(   367.06), EASYSIMD_FLOAT64_C(  -211.35), EASYSIMD_FLOAT64_C(  -842.15) },
      { EASYSIMD_FLOAT32_C(  -708.06), EASYSIMD_FLOAT32_C(  -802.37), EASYSIMD_FLOAT32_C(  -868.11), EASYSIMD_FLOAT32_C(  -842.15) } },
    { { EASYSIMD_FLOAT32_C(  -101.17), EASYSIMD_FLOAT32_C(  -291.82), EASYSIMD_FLOAT32_C(  -349.07), EASYSIMD_FLOAT32_C(  -616.02) },
      UINT8_C( 82),
      { EASYSIMD_FLOAT64_C(    -5.20), EASYSIMD_FLOAT64_C(  -937.08), EASYSIMD_FLOAT64_C(   810.05), EASYSIMD_FLOAT64_C(   336.47) },
      { EASYSIMD_FLOAT32_C(  -101.17), EASYSIMD_FLOAT32_C(  -937.08), EASYSIMD_FLOAT32_C(  -349.07), EASYSIMD_FLOAT32_C(  -616.02) } },
    { { EASYSIMD_FLOAT32_C(   911.71), EASYSIMD_FLOAT32_C(  -669.98), EASYSIMD_FLOAT32_C(   708.93), EASYSIMD_FLOAT32_C(  -429.31) },
      UINT8_C( 35),
      { EASYSIMD_FLOAT64_C(   111.19), EASYSIMD_FLOAT64_C(  -311.49), EASYSIMD_FLOAT64_C(  -486.68), EASYSIMD_FLOAT64_C(  -360.00) },
      { EASYSIMD_FLOAT32_C(   111.19), EASYSIMD_FLOAT32_C(  -311.49), EASYSIMD_FLOAT32_C(   708.93), EASYSIMD_FLOAT32_C(  -429.31) } },
    { { EASYSIMD_FLOAT32_C(   299.11), EASYSIMD_FLOAT32_C(   336.66), EASYSIMD_FLOAT32_C(  -557.28), EASYSIMD_FLOAT32_C(    -9.19) },
      UINT8_C(146),
      { EASYSIMD_FLOAT64_C(  -359.65), EASYSIMD_FLOAT64_C(   122.70), EASYSIMD_FLOAT64_C(   761.94), EASYSIMD_FLOAT64_C(   -34.61) },
      { EASYSIMD_FLOAT32_C(   299.11), EASYSIMD_FLOAT32_C(   122.70), EASYSIMD_FLOAT32_C(  -557.28), EASYSIMD_FLOAT32_C(    -9.19) } },
    { { EASYSIMD_FLOAT32_C(  -947.10), EASYSIMD_FLOAT32_C(   129.00), EASYSIMD_FLOAT32_C(   754.04), EASYSIMD_FLOAT32_C(  -789.25) },
      UINT8_C( 74),
      { EASYSIMD_FLOAT64_C(  -537.78), EASYSIMD_FLOAT64_C(  -138.32), EASYSIMD_FLOAT64_C(  -588.19), EASYSIMD_FLOAT64_C(   585.99) },
      { EASYSIMD_FLOAT32_C(  -947.10), EASYSIMD_FLOAT32_C(  -138.32), EASYSIMD_FLOAT32_C(   754.04), EASYSIMD_FLOAT32_C(   585.99) } },
    { { EASYSIMD_FLOAT32_C(   856.48), EASYSIMD_FLOAT32_C(  -525.27), EASYSIMD_FLOAT32_C(   396.04), EASYSIMD_FLOAT32_C(   192.94) },
      UINT8_C( 30),
      { EASYSIMD_FLOAT64_C(   726.06), EASYSIMD_FLOAT64_C(   -98.13), EASYSIMD_FLOAT64_C(   -42.87), EASYSIMD_FLOAT64_C(   589.21) },
      { EASYSIMD_FLOAT32_C(   856.48), EASYSIMD_FLOAT32_C(   -98.13), EASYSIMD_FLOAT32_C(   -42.87), EASYSIMD_FLOAT32_C(   589.21) } },
    { { EASYSIMD_FLOAT32_C(  -986.94), EASYSIMD_FLOAT32_C(   645.65), EASYSIMD_FLOAT32_C(  -897.47), EASYSIMD_FLOAT32_C(  -346.94) },
      UINT8_C( 53),
      { EASYSIMD_FLOAT64_C(   439.19), EASYSIMD_FLOAT64_C(    95.78), EASYSIMD_FLOAT64_C(   935.56), EASYSIMD_FLOAT64_C(    67.79) },
      { EASYSIMD_FLOAT32_C(   439.19), EASYSIMD_FLOAT32_C(   645.65), EASYSIMD_FLOAT32_C(   935.56), EASYSIMD_FLOAT32_C(  -346.94) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cvtpd_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_cvtpd_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm256_mask_cvtpd_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_cvtpd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(178),
      { EASYSIMD_FLOAT64_C(    58.26), EASYSIMD_FLOAT64_C(  -170.27), EASYSIMD_FLOAT64_C(  -298.48), EASYSIMD_FLOAT64_C(   111.16) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -170.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 68),
      { EASYSIMD_FLOAT64_C(  -544.44), EASYSIMD_FLOAT64_C(   321.91), EASYSIMD_FLOAT64_C(   986.56), EASYSIMD_FLOAT64_C(   -82.22) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   986.56), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(124),
      { EASYSIMD_FLOAT64_C(  -601.63), EASYSIMD_FLOAT64_C(  -496.23), EASYSIMD_FLOAT64_C(  -959.93), EASYSIMD_FLOAT64_C(  -126.90) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -959.93), EASYSIMD_FLOAT32_C(  -126.90) } },
    { UINT8_C(161),
      { EASYSIMD_FLOAT64_C(   233.01), EASYSIMD_FLOAT64_C(   259.55), EASYSIMD_FLOAT64_C(   625.86), EASYSIMD_FLOAT64_C(  -865.11) },
      { EASYSIMD_FLOAT32_C(   233.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(  8),
      { EASYSIMD_FLOAT64_C(   215.07), EASYSIMD_FLOAT64_C(  -852.05), EASYSIMD_FLOAT64_C(   862.33), EASYSIMD_FLOAT64_C(   317.60) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   317.60) } },
    { UINT8_C(147),
      { EASYSIMD_FLOAT64_C(  -192.92), EASYSIMD_FLOAT64_C(  -243.21), EASYSIMD_FLOAT64_C(   896.79), EASYSIMD_FLOAT64_C(  -257.35) },
      { EASYSIMD_FLOAT32_C(  -192.92), EASYSIMD_FLOAT32_C(  -243.21), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 63),
      { EASYSIMD_FLOAT64_C(   632.92), EASYSIMD_FLOAT64_C(   800.91), EASYSIMD_FLOAT64_C(  -345.69), EASYSIMD_FLOAT64_C(  -665.56) },
      { EASYSIMD_FLOAT32_C(   632.92), EASYSIMD_FLOAT32_C(   800.91), EASYSIMD_FLOAT32_C(  -345.69), EASYSIMD_FLOAT32_C(  -665.56) } },
    {    UINT8_MAX,
      { EASYSIMD_FLOAT64_C(  -386.95), EASYSIMD_FLOAT64_C(  -209.99), EASYSIMD_FLOAT64_C(  -766.01), EASYSIMD_FLOAT64_C(  -400.39) },
      { EASYSIMD_FLOAT32_C(  -386.95), EASYSIMD_FLOAT32_C(  -209.99), EASYSIMD_FLOAT32_C(  -766.01), EASYSIMD_FLOAT32_C(  -400.39) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_cvtpd_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_cvtpd_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm256_maskz_cvtpd_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepi32_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int32_t a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { -INT32_C(    54513482),  INT32_C(  1138961205),  INT32_C(   343962185), -INT32_C(   369293577),  INT32_C(  1254917685),  INT32_C(  2080833130),  INT32_C(  1142021979),  INT32_C(   394507617),
         INT32_C(   437470181), -INT32_C(  1201801362),  INT32_C(  1624039017),  INT32_C(   424266212), -INT32_C(  1167911344),  INT32_C(  1865837076), -INT32_C(  1431025847), -INT32_C(   507365380) },
      { EASYSIMD_FLOAT32_C(-54513480.00), EASYSIMD_FLOAT32_C(1138961152.00), EASYSIMD_FLOAT32_C(343962176.00), EASYSIMD_FLOAT32_C(-369293568.00),
        EASYSIMD_FLOAT32_C(1254917632.00), EASYSIMD_FLOAT32_C(2080833152.00), EASYSIMD_FLOAT32_C(1142022016.00), EASYSIMD_FLOAT32_C(394507616.00),
        EASYSIMD_FLOAT32_C(437470176.00), EASYSIMD_FLOAT32_C(-1201801344.00), EASYSIMD_FLOAT32_C(1624039040.00), EASYSIMD_FLOAT32_C(424266208.00),
        EASYSIMD_FLOAT32_C(-1167911296.00), EASYSIMD_FLOAT32_C(1865837056.00), EASYSIMD_FLOAT32_C(-1431025792.00), EASYSIMD_FLOAT32_C(-507365376.00) } },
    { { -INT32_C(   369371782),  INT32_C(   899766732),  INT32_C(   462777655), -INT32_C(  2043289802),  INT32_C(   155228405),  INT32_C(  1282963202), -INT32_C(  1141494594), -INT32_C(   560154525),
         INT32_C(  1523030158),  INT32_C(   680552689),  INT32_C(   188949973), -INT32_C(   107841532),  INT32_C(   318951953),  INT32_C(   140475209),  INT32_C(   197351079), -INT32_C(  1662427378) },
      { EASYSIMD_FLOAT32_C(-369371776.00), EASYSIMD_FLOAT32_C(899766720.00), EASYSIMD_FLOAT32_C(462777664.00), EASYSIMD_FLOAT32_C(-2043289856.00),
        EASYSIMD_FLOAT32_C(155228400.00), EASYSIMD_FLOAT32_C(1282963200.00), EASYSIMD_FLOAT32_C(-1141494656.00), EASYSIMD_FLOAT32_C(-560154496.00),
        EASYSIMD_FLOAT32_C(1523030144.00), EASYSIMD_FLOAT32_C(680552704.00), EASYSIMD_FLOAT32_C(188949968.00), EASYSIMD_FLOAT32_C(-107841536.00),
        EASYSIMD_FLOAT32_C(318951968.00), EASYSIMD_FLOAT32_C(140475216.00), EASYSIMD_FLOAT32_C(197351072.00), EASYSIMD_FLOAT32_C(-1662427392.00) } },
    { { -INT32_C(   386420489), -INT32_C(   317683944), -INT32_C(  1309125460), -INT32_C(   576025908), -INT32_C(  1494110883), -INT32_C(   810659800), -INT32_C(  1243975258), -INT32_C(   934165551),
        -INT32_C(  1951315853),  INT32_C(  2088288719), -INT32_C(   500338411),  INT32_C(  1488967675), -INT32_C(  1392529276), -INT32_C(  1485001471), -INT32_C(   262384097), -INT32_C(  1917276646) },
      { EASYSIMD_FLOAT32_C(-386420480.00), EASYSIMD_FLOAT32_C(-317683936.00), EASYSIMD_FLOAT32_C(-1309125504.00), EASYSIMD_FLOAT32_C(-576025920.00),
        EASYSIMD_FLOAT32_C(-1494110848.00), EASYSIMD_FLOAT32_C(-810659776.00), EASYSIMD_FLOAT32_C(-1243975296.00), EASYSIMD_FLOAT32_C(-934165568.00),
        EASYSIMD_FLOAT32_C(-1951315840.00), EASYSIMD_FLOAT32_C(2088288768.00), EASYSIMD_FLOAT32_C(-500338400.00), EASYSIMD_FLOAT32_C(1488967680.00),
        EASYSIMD_FLOAT32_C(-1392529280.00), EASYSIMD_FLOAT32_C(-1485001472.00), EASYSIMD_FLOAT32_C(-262384096.00), EASYSIMD_FLOAT32_C(-1917276672.00) } },
    { { -INT32_C(   971413002),  INT32_C(  1078104363), -INT32_C(    31297790), -INT32_C(   883498426), -INT32_C(  1820895854),  INT32_C(   574354179),  INT32_C(  1678939978),  INT32_C(  1022478917),
         INT32_C(  1593969204), -INT32_C(  1633729380), -INT32_C(   107167053),  INT32_C(   918877092),  INT32_C(  1271544904),  INT32_C(  2037187887), -INT32_C(   488800356),  INT32_C(  2115948362) },
      { EASYSIMD_FLOAT32_C(-971412992.00), EASYSIMD_FLOAT32_C(1078104320.00), EASYSIMD_FLOAT32_C(-31297790.00), EASYSIMD_FLOAT32_C(-883498432.00),
        EASYSIMD_FLOAT32_C(-1820895872.00), EASYSIMD_FLOAT32_C(574354176.00), EASYSIMD_FLOAT32_C(1678940032.00), EASYSIMD_FLOAT32_C(1022478912.00),
        EASYSIMD_FLOAT32_C(1593969152.00), EASYSIMD_FLOAT32_C(-1633729408.00), EASYSIMD_FLOAT32_C(-107167056.00), EASYSIMD_FLOAT32_C(918877120.00),
        EASYSIMD_FLOAT32_C(1271544960.00), EASYSIMD_FLOAT32_C(2037187840.00), EASYSIMD_FLOAT32_C(-488800352.00), EASYSIMD_FLOAT32_C(2115948416.00) } },
    { {  INT32_C(  1977426137),  INT32_C(   387218532), -INT32_C(   502222786), -INT32_C(   333851229),  INT32_C(  1077404433), -INT32_C(  2068142616),  INT32_C(  1852217124),  INT32_C(  1089242214),
         INT32_C(   146131364), -INT32_C(  2078291642),  INT32_C(   493301882),  INT32_C(   369721349),  INT32_C(  1263944035),  INT32_C(   181342438),  INT32_C(   242824872),  INT32_C(  1598973370) },
      { EASYSIMD_FLOAT32_C(1977426176.00), EASYSIMD_FLOAT32_C(387218528.00), EASYSIMD_FLOAT32_C(-502222784.00), EASYSIMD_FLOAT32_C(-333851232.00),
        EASYSIMD_FLOAT32_C(1077404416.00), EASYSIMD_FLOAT32_C(-2068142592.00), EASYSIMD_FLOAT32_C(1852217088.00), EASYSIMD_FLOAT32_C(1089242240.00),
        EASYSIMD_FLOAT32_C(146131360.00), EASYSIMD_FLOAT32_C(-2078291584.00), EASYSIMD_FLOAT32_C(493301888.00), EASYSIMD_FLOAT32_C(369721344.00),
        EASYSIMD_FLOAT32_C(1263944064.00), EASYSIMD_FLOAT32_C(181342432.00), EASYSIMD_FLOAT32_C(242824864.00), EASYSIMD_FLOAT32_C(1598973312.00) } },
    { {  INT32_C(  1969685551),  INT32_C(  1207535565), -INT32_C(  1134206793),  INT32_C(  1137864416), -INT32_C(  1785845585), -INT32_C(   509583815),  INT32_C(  1324292500), -INT32_C(  1381155202),
         INT32_C(   253891906),  INT32_C(  1398217884),  INT32_C(  1561312380), -INT32_C(   626990806),  INT32_C(  1114582793),  INT32_C(   555945869), -INT32_C(  1485892824), -INT32_C(  1823204015) },
      { EASYSIMD_FLOAT32_C(1969685504.00), EASYSIMD_FLOAT32_C(1207535616.00), EASYSIMD_FLOAT32_C(-1134206848.00), EASYSIMD_FLOAT32_C(1137864448.00),
        EASYSIMD_FLOAT32_C(-1785845632.00), EASYSIMD_FLOAT32_C(-509583808.00), EASYSIMD_FLOAT32_C(1324292480.00), EASYSIMD_FLOAT32_C(-1381155200.00),
        EASYSIMD_FLOAT32_C(253891904.00), EASYSIMD_FLOAT32_C(1398217856.00), EASYSIMD_FLOAT32_C(1561312384.00), EASYSIMD_FLOAT32_C(-626990784.00),
        EASYSIMD_FLOAT32_C(1114582784.00), EASYSIMD_FLOAT32_C(555945856.00), EASYSIMD_FLOAT32_C(-1485892864.00), EASYSIMD_FLOAT32_C(-1823203968.00) } },
    { { -INT32_C(   828213454),  INT32_C(   253884819), -INT32_C(   529780555),  INT32_C(   448400657), -INT32_C(   916641476),  INT32_C(  1642758201), -INT32_C(   469214829), -INT32_C(  1468572298),
         INT32_C(  1735793364), -INT32_C(   914974957), -INT32_C(   659954745),  INT32_C(   754148336), -INT32_C(   973778804),  INT32_C(  1680334800), -INT32_C(  1354223816),  INT32_C(  1616363660) },
      { EASYSIMD_FLOAT32_C(-828213440.00), EASYSIMD_FLOAT32_C(253884816.00), EASYSIMD_FLOAT32_C(-529780544.00), EASYSIMD_FLOAT32_C(448400672.00),
        EASYSIMD_FLOAT32_C(-916641472.00), EASYSIMD_FLOAT32_C(1642758144.00), EASYSIMD_FLOAT32_C(-469214816.00), EASYSIMD_FLOAT32_C(-1468572288.00),
        EASYSIMD_FLOAT32_C(1735793408.00), EASYSIMD_FLOAT32_C(-914974976.00), EASYSIMD_FLOAT32_C(-659954752.00), EASYSIMD_FLOAT32_C(754148352.00),
        EASYSIMD_FLOAT32_C(-973778816.00), EASYSIMD_FLOAT32_C(1680334848.00), EASYSIMD_FLOAT32_C(-1354223872.00), EASYSIMD_FLOAT32_C(1616363648.00) } },
    { { -INT32_C(   305672486),  INT32_C(   766918245),  INT32_C(   285564705),  INT32_C(  1329461442),  INT32_C(   420753992),  INT32_C(  1232943889), -INT32_C(   134691477),  INT32_C(  1599623301),
        -INT32_C(  2092097762),  INT32_C(  2125464413),  INT32_C(   630175331), -INT32_C(   143340370),  INT32_C(   252742142),  INT32_C(   794398148), -INT32_C(   668511918), -INT32_C(  1086881887) },
      { EASYSIMD_FLOAT32_C(-305672480.00), EASYSIMD_FLOAT32_C(766918272.00), EASYSIMD_FLOAT32_C(285564704.00), EASYSIMD_FLOAT32_C(1329461504.00),
        EASYSIMD_FLOAT32_C(420753984.00), EASYSIMD_FLOAT32_C(1232943872.00), EASYSIMD_FLOAT32_C(-134691472.00), EASYSIMD_FLOAT32_C(1599623296.00),
        EASYSIMD_FLOAT32_C(-2092097792.00), EASYSIMD_FLOAT32_C(2125464448.00), EASYSIMD_FLOAT32_C(630175360.00), EASYSIMD_FLOAT32_C(-143340368.00),
        EASYSIMD_FLOAT32_C(252742144.00), EASYSIMD_FLOAT32_C(794398144.00), EASYSIMD_FLOAT32_C(-668511936.00), EASYSIMD_FLOAT32_C(-1086881920.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvtepi32_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cvtepi32_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
  #else
    fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512 r = easysimd_mm512_cvtepi32_ps(a);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm_mask_cvtph_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const uint16_t a[8];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -145.65), EASYSIMD_FLOAT32_C(   166.66), EASYSIMD_FLOAT32_C(   778.97), EASYSIMD_FLOAT32_C(  -311.85) },
      UINT8_C(151),
      { UINT16_C(15984), UINT16_C( 4910), UINT16_C(22952), UINT16_C(21581), UINT16_C(41052), UINT16_C( 1121), UINT16_C(57821), UINT16_C(65192) },
      { EASYSIMD_FLOAT32_C(     1.61), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   181.00), EASYSIMD_FLOAT32_C(  -311.85) } },
    { { EASYSIMD_FLOAT32_C(   389.30), EASYSIMD_FLOAT32_C(  -792.97), EASYSIMD_FLOAT32_C(  -885.11), EASYSIMD_FLOAT32_C(   495.57) },
      UINT8_C( 12),
      { UINT16_C(35156), UINT16_C(11938), UINT16_C( 3813), UINT16_C(52198), UINT16_C(32048), UINT16_C(28475), UINT16_C(20139), UINT16_C( 1047) },
      { EASYSIMD_FLOAT32_C(   389.30), EASYSIMD_FLOAT32_C(  -792.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -15.80) } },
    { { EASYSIMD_FLOAT32_C(   454.78), EASYSIMD_FLOAT32_C(    67.12), EASYSIMD_FLOAT32_C(  -118.51), EASYSIMD_FLOAT32_C(   236.20) },
      UINT8_C(205),
      { UINT16_C( 6501), UINT16_C( 3502), UINT16_C(20247), UINT16_C(37851), UINT16_C(59563), UINT16_C(13799), UINT16_C( 5514), UINT16_C(38938) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    67.12), EASYSIMD_FLOAT32_C(    28.36), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(  -507.60), EASYSIMD_FLOAT32_C(  -566.24), EASYSIMD_FLOAT32_C(    24.14), EASYSIMD_FLOAT32_C(   413.01) },
      UINT8_C( 33),
      { UINT16_C( 9271), UINT16_C(20335), UINT16_C( 2857), UINT16_C(35259), UINT16_C(34887), UINT16_C(24814), UINT16_C(64310), UINT16_C(34167) },
      { EASYSIMD_FLOAT32_C(     0.02), EASYSIMD_FLOAT32_C(  -566.24), EASYSIMD_FLOAT32_C(    24.14), EASYSIMD_FLOAT32_C(   413.01) } },
    { { EASYSIMD_FLOAT32_C(   543.72), EASYSIMD_FLOAT32_C(   141.98), EASYSIMD_FLOAT32_C(  -526.45), EASYSIMD_FLOAT32_C(   805.99) },
      UINT8_C(242),
      { UINT16_C(18790), UINT16_C(32776), UINT16_C(  993), UINT16_C(43365), UINT16_C(34428), UINT16_C(41441), UINT16_C(12534), UINT16_C(  458) },
      { EASYSIMD_FLOAT32_C(   543.72), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -526.45), EASYSIMD_FLOAT32_C(   805.99) } },
    { { EASYSIMD_FLOAT32_C(  -593.27), EASYSIMD_FLOAT32_C(   -53.95), EASYSIMD_FLOAT32_C(   610.26), EASYSIMD_FLOAT32_C(   118.49) },
      UINT8_C( 66),
      { UINT16_C(43433), UINT16_C( 8509), UINT16_C( 5167), UINT16_C(24620), UINT16_C( 7891), UINT16_C( 7366), UINT16_C(17958), UINT16_C(11005) },
      { EASYSIMD_FLOAT32_C(  -593.27), EASYSIMD_FLOAT32_C(     0.01), EASYSIMD_FLOAT32_C(   610.26), EASYSIMD_FLOAT32_C(   118.49) } },
    { { EASYSIMD_FLOAT32_C(   301.13), EASYSIMD_FLOAT32_C(  -243.64), EASYSIMD_FLOAT32_C(  -467.95), EASYSIMD_FLOAT32_C(  -957.46) },
      UINT8_C(136),
      { UINT16_C(10311), UINT16_C( 4536), UINT16_C(41769), UINT16_C(29285), UINT16_C(42774), UINT16_C(48924), UINT16_C(15844), UINT16_C(63982) },
      { EASYSIMD_FLOAT32_C(   301.13), EASYSIMD_FLOAT32_C(  -243.64), EASYSIMD_FLOAT32_C(  -467.95), EASYSIMD_FLOAT32_C( 13096.00) } },
    { { EASYSIMD_FLOAT32_C(  -541.63), EASYSIMD_FLOAT32_C(   525.17), EASYSIMD_FLOAT32_C(  -148.64), EASYSIMD_FLOAT32_C(   762.81) },
      UINT8_C( 20),
      { UINT16_C(44777), UINT16_C(58970), UINT16_C( 1752), UINT16_C(32397), UINT16_C( 5432), UINT16_C(24774), UINT16_C(55245), UINT16_C(28809) },
      { EASYSIMD_FLOAT32_C(  -541.63), EASYSIMD_FLOAT32_C(   525.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   762.81) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtph_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtph_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_u16x8();
    easysimd__m128 r = easysimd_mm_mask_cvtph_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtph_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint16_t a[8];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(158),
      { UINT16_C(64718), UINT16_C(20006), UINT16_C(63065), UINT16_C(  342), UINT16_C(48269), UINT16_C( 7911), UINT16_C( 7545), UINT16_C(44825) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    24.59), EASYSIMD_FLOAT32_C(-26000.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(104),
      { UINT16_C(61876), UINT16_C(59509), UINT16_C(34841), UINT16_C(55784), UINT16_C( 6555), UINT16_C(61623), UINT16_C(22078), UINT16_C(15039) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -189.00) } },
    { UINT8_C(124),
      { UINT16_C(37901), UINT16_C(25714), UINT16_C(  149), UINT16_C(31776), UINT16_C(39198), UINT16_C(14490), UINT16_C(  585), UINT16_C(15084) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF } },
    { UINT8_C(119),
      { UINT16_C(21716), UINT16_C(48383), UINT16_C(39725), UINT16_C(58837), UINT16_C( 5003), UINT16_C(19003), UINT16_C(46926), UINT16_C(57944) },
      { EASYSIMD_FLOAT32_C(    77.25), EASYSIMD_FLOAT32_C(    -1.25), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 41),
      { UINT16_C(30652), UINT16_C(56361), UINT16_C(18675), UINT16_C(36214), UINT16_C(49024), UINT16_C(27792), UINT16_C( 2041), UINT16_C(19776) },
      { EASYSIMD_FLOAT32_C( 31680.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00) } },
    { UINT8_C(  7),
      { UINT16_C(31740), UINT16_C(53666), UINT16_C(11616), UINT16_C(39909), UINT16_C(13176), UINT16_C(53330), UINT16_C(31509), UINT16_C(35980) },
      { EASYSIMD_FLOAT32_C( 65408.00), EASYSIMD_FLOAT32_C(   -45.06), EASYSIMD_FLOAT32_C(     0.08), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(165),
      { UINT16_C(32616), UINT16_C(57069), UINT16_C(27917), UINT16_C(40349), UINT16_C(38873), UINT16_C( 6564), UINT16_C(44004), UINT16_C(24341) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  5172.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 77),
      { UINT16_C(49126), UINT16_C(52091), UINT16_C(62298), UINT16_C(44286), UINT16_C( 5059), UINT16_C(20264), UINT16_C(52639), UINT16_C( 8119) },
      { EASYSIMD_FLOAT32_C(    -1.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-15056.00), EASYSIMD_FLOAT32_C(    -0.08) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtph_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtph_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_u16x8();
    easysimd__m128 r = easysimd_mm_maskz_cvtph_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtpd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    87.02), EASYSIMD_FLOAT32_C(   675.56), EASYSIMD_FLOAT32_C(  -952.23), EASYSIMD_FLOAT32_C(   543.14) },
      UINT8_C( 49),
      { EASYSIMD_FLOAT64_C(  -100.60), EASYSIMD_FLOAT64_C(  -697.81) },
      { EASYSIMD_FLOAT32_C(  -100.60), EASYSIMD_FLOAT32_C(   675.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -505.91), EASYSIMD_FLOAT32_C(   760.44), EASYSIMD_FLOAT32_C(  -409.54), EASYSIMD_FLOAT32_C(  -645.18) },
      UINT8_C(115),
      { EASYSIMD_FLOAT64_C(  -505.09), EASYSIMD_FLOAT64_C(    72.83) },
      { EASYSIMD_FLOAT32_C(  -505.09), EASYSIMD_FLOAT32_C(    72.83), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -570.00), EASYSIMD_FLOAT32_C(   493.81), EASYSIMD_FLOAT32_C(   664.77), EASYSIMD_FLOAT32_C(   708.97) },
      UINT8_C(209),
      { EASYSIMD_FLOAT64_C(   806.72), EASYSIMD_FLOAT64_C(   -85.04) },
      { EASYSIMD_FLOAT32_C(   806.72), EASYSIMD_FLOAT32_C(   493.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -353.02), EASYSIMD_FLOAT32_C(  -869.72), EASYSIMD_FLOAT32_C(  -648.59), EASYSIMD_FLOAT32_C(   835.53) },
      UINT8_C(236),
      { EASYSIMD_FLOAT64_C(  -838.72), EASYSIMD_FLOAT64_C(   -20.01) },
      { EASYSIMD_FLOAT32_C(  -353.02), EASYSIMD_FLOAT32_C(  -869.72), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -824.84), EASYSIMD_FLOAT32_C(  -922.76), EASYSIMD_FLOAT32_C(  -691.58), EASYSIMD_FLOAT32_C(   262.19) },
      UINT8_C(186),
      { EASYSIMD_FLOAT64_C(  -643.81), EASYSIMD_FLOAT64_C(  -194.67) },
      { EASYSIMD_FLOAT32_C(  -824.84), EASYSIMD_FLOAT32_C(  -194.67), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   133.59), EASYSIMD_FLOAT32_C(   255.59), EASYSIMD_FLOAT32_C(   107.52), EASYSIMD_FLOAT32_C(   627.68) },
      UINT8_C(115),
      { EASYSIMD_FLOAT64_C(   697.98), EASYSIMD_FLOAT64_C(   982.51) },
      { EASYSIMD_FLOAT32_C(   697.98), EASYSIMD_FLOAT32_C(   982.51), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   362.89), EASYSIMD_FLOAT32_C(  -807.11), EASYSIMD_FLOAT32_C(    55.34), EASYSIMD_FLOAT32_C(   792.89) },
      UINT8_C(244),
      { EASYSIMD_FLOAT64_C(  -279.89), EASYSIMD_FLOAT64_C(   501.86) },
      { EASYSIMD_FLOAT32_C(   362.89), EASYSIMD_FLOAT32_C(  -807.11), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -675.93), EASYSIMD_FLOAT32_C(  -473.17), EASYSIMD_FLOAT32_C(  -583.18), EASYSIMD_FLOAT32_C(   -28.96) },
      UINT8_C(136),
      { EASYSIMD_FLOAT64_C(  -231.78), EASYSIMD_FLOAT64_C(  -193.43) },
      { EASYSIMD_FLOAT32_C(  -675.93), EASYSIMD_FLOAT32_C(  -473.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvtpd_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_cvtpd_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_cvtpd_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvtpd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(   943.19), EASYSIMD_FLOAT64_C(  -952.74) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(241),
      { EASYSIMD_FLOAT64_C(  -118.01), EASYSIMD_FLOAT64_C(  -291.66) },
      { EASYSIMD_FLOAT32_C(  -118.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(120),
      { EASYSIMD_FLOAT64_C(    82.50), EASYSIMD_FLOAT64_C(   561.51) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(161),
      { EASYSIMD_FLOAT64_C(   988.17), EASYSIMD_FLOAT64_C(   906.70) },
      { EASYSIMD_FLOAT32_C(   988.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 64),
      { EASYSIMD_FLOAT64_C(  -289.44), EASYSIMD_FLOAT64_C(   566.49) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 85),
      { EASYSIMD_FLOAT64_C(   669.04), EASYSIMD_FLOAT64_C(   550.19) },
      { EASYSIMD_FLOAT32_C(   669.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(141),
      { EASYSIMD_FLOAT64_C(  -779.75), EASYSIMD_FLOAT64_C(  -695.67) },
      { EASYSIMD_FLOAT32_C(  -779.75), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(146),
      { EASYSIMD_FLOAT64_C(   -15.20), EASYSIMD_FLOAT64_C(   -67.47) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -67.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvtpd_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_cvtpd_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_cvtpd_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepi64_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const int64_t a[4];
    const easysimd_float32 r[4];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT32_C(   838.77), EASYSIMD_FLOAT32_C(  -785.06), EASYSIMD_FLOAT32_C(   233.52), EASYSIMD_FLOAT32_C(   545.75) },
      UINT8_C( 67),
      { -INT64_C(  996056663603267864), -INT64_C( 1854852747849430717), -INT64_C( 5578955172846991381),  INT64_C( 1783311311460048830) },
      { EASYSIMD_FLOAT32_C(-996056654573207552.00), EASYSIMD_FLOAT32_C(-1854852751436021760.00), EASYSIMD_FLOAT32_C(   233.52), EASYSIMD_FLOAT32_C(   545.75) } },
    { { EASYSIMD_FLOAT32_C(  -404.99), EASYSIMD_FLOAT32_C(   733.98), EASYSIMD_FLOAT32_C(  -999.65), EASYSIMD_FLOAT32_C(   972.33) },
      UINT8_C(207),
      { -INT64_C( 9118409065400992470), -INT64_C( 8393128630490932694),  INT64_C( 6016745035470092080), -INT64_C( 4550979142348072828) },
      { EASYSIMD_FLOAT32_C(-9118408808576581632.00), EASYSIMD_FLOAT32_C(-8393128607697862656.00), EASYSIMD_FLOAT32_C(6016744780736757760.00), EASYSIMD_FLOAT32_C(-4550979232678805504.00) } },
    { { EASYSIMD_FLOAT32_C(   383.37), EASYSIMD_FLOAT32_C(   743.16), EASYSIMD_FLOAT32_C(   -13.19), EASYSIMD_FLOAT32_C(  -832.90) },
      UINT8_C(209),
      { -INT64_C(  451296348235632792),  INT64_C( 6279277450197878833),  INT64_C( 4910863787627411449),  INT64_C( 4788974411374782139) },
      { EASYSIMD_FLOAT32_C(-451296334327054336.00), EASYSIMD_FLOAT32_C(   743.16), EASYSIMD_FLOAT32_C(   -13.19), EASYSIMD_FLOAT32_C(  -832.90) } },
    { { EASYSIMD_FLOAT32_C(   108.69), EASYSIMD_FLOAT32_C(  -595.73), EASYSIMD_FLOAT32_C(    36.46), EASYSIMD_FLOAT32_C(  -488.20) },
      UINT8_C( 28),
      { -INT64_C( 4423740684644769855), -INT64_C( 5554542741373062437),  INT64_C( 8445395659656081011),  INT64_C( 4329195625418552967) },
      { EASYSIMD_FLOAT32_C(   108.69), EASYSIMD_FLOAT32_C(  -595.73), EASYSIMD_FLOAT32_C(8445395542191636480.00), EASYSIMD_FLOAT32_C(4329195642728480768.00) } },
    { { EASYSIMD_FLOAT32_C(   313.64), EASYSIMD_FLOAT32_C(    58.86), EASYSIMD_FLOAT32_C(   557.10), EASYSIMD_FLOAT32_C(  -301.06) },
      UINT8_C( 25),
      { -INT64_C( 7520652615551250464), -INT64_C( 3172394780348230864),  INT64_C( 1335689672010657687), -INT64_C( 1521264450649799321) },
      { EASYSIMD_FLOAT32_C(-7520652387162259456.00), EASYSIMD_FLOAT32_C(    58.86), EASYSIMD_FLOAT32_C(   557.10), EASYSIMD_FLOAT32_C(-1521264496981573632.00) } },
    { { EASYSIMD_FLOAT32_C(   -22.20), EASYSIMD_FLOAT32_C(   950.20), EASYSIMD_FLOAT32_C(  -126.50), EASYSIMD_FLOAT32_C(   224.00) },
      UINT8_C( 30),
      {  INT64_C( 4643965333446513501), -INT64_C(  953665125131729498), -INT64_C( 3585390405210220945),  INT64_C( 4826729485560719365) },
      { EASYSIMD_FLOAT32_C(   -22.20), EASYSIMD_FLOAT32_C(-953665121203257344.00), EASYSIMD_FLOAT32_C(-3585390320189177856.00), EASYSIMD_FLOAT32_C(4826729602099445760.00) } },
    { { EASYSIMD_FLOAT32_C(   669.04), EASYSIMD_FLOAT32_C(   199.01), EASYSIMD_FLOAT32_C(  -639.28), EASYSIMD_FLOAT32_C(   118.02) },
      UINT8_C(246),
      { -INT64_C( 6176208856216114075), -INT64_C( 5208327669726758026),  INT64_C( 4798753931905500811),  INT64_C( 3662761088850713236) },
      { EASYSIMD_FLOAT32_C(   669.04), EASYSIMD_FLOAT32_C(-5208327756902825984.00), EASYSIMD_FLOAT32_C(4798754177998127104.00), EASYSIMD_FLOAT32_C(   118.02) } },
    { { EASYSIMD_FLOAT32_C(  -865.05), EASYSIMD_FLOAT32_C(  -890.00), EASYSIMD_FLOAT32_C(   954.80), EASYSIMD_FLOAT32_C(   948.78) },
      UINT8_C( 36),
      { -INT64_C( 9154157563500422875),  INT64_C(  174443468881112222),  INT64_C( 9006765768818797911),  INT64_C( 3593754909576933449) },
      { EASYSIMD_FLOAT32_C(  -865.05), EASYSIMD_FLOAT32_C(  -890.00), EASYSIMD_FLOAT32_C(9006765497403834368.00), EASYSIMD_FLOAT32_C(   948.78) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cvtepi64_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m128 r = easysimd_mm256_mask_cvtepi64_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_cvtepi64_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const int64_t a[4];
    const easysimd_float32 r[4];
  } test_vec[8] = {
    { UINT8_C(174),
      {  INT64_C( 3818314197920420473),  INT64_C( 4216108725059789164), -INT64_C( 2164694639909948617),  INT64_C(  639263272476717212) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(4216108672788463616.00), EASYSIMD_FLOAT32_C(-2164694578387484672.00), EASYSIMD_FLOAT32_C(639263275934023680.00) } },
    { UINT8_C(177),
      { -INT64_C( 8751499699082926145), -INT64_C( 1377454725454973173),  INT64_C( 3771622317199560261),  INT64_C( 2754738867155470443) },
      { EASYSIMD_FLOAT32_C(-8751499579364474880.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 51),
      { -INT64_C( 8508391704932891021),  INT64_C(  559201140571543344),  INT64_C( 6644145171401964875),  INT64_C( 7818788780286131842) },
      { EASYSIMD_FLOAT32_C(-8508391511149248512.00), EASYSIMD_FLOAT32_C(559201134166671360.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 48),
      { -INT64_C( 3429336938878644967),  INT64_C( 2686767335142450511),  INT64_C(  367982660086731482),  INT64_C(  302983528750868761) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(182),
      { -INT64_C( 8641634024882750665),  INT64_C( 7724611628090527752),  INT64_C( 5818289974137769038),  INT64_C( 3230557530203275701) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(7724611794114707456.00), EASYSIMD_FLOAT32_C(5818290078748770304.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(122),
      {  INT64_C( 1336625719158894788), -INT64_C( 1983317662072314410), -INT64_C( 7064428332673458551), -INT64_C( 5904571051495189419) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-1983317628441067520.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-5904570955203608576.00) } },
    { UINT8_C(175),
      { -INT64_C( 6654133600575671765),  INT64_C( 7678812513562776112), -INT64_C( 2532179919391764118), -INT64_C( 6276483524043903114) },
      { EASYSIMD_FLOAT32_C(-6654133870381760512.00), EASYSIMD_FLOAT32_C(7678812736771325952.00), EASYSIMD_FLOAT32_C(-2532179951692546048.00), EASYSIMD_FLOAT32_C(-6276483512077910016.00) } },
    { UINT8_C(165),
      { -INT64_C(  825276791850239337), -INT64_C(  734744095242546237), -INT64_C( 7184762932922920208), -INT64_C(  182463517329620898) },
      { EASYSIMD_FLOAT32_C(-825276797695295488.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-7184763129748783104.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_cvtepi64_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m128 r = easysimd_mm256_maskz_cvtepi64_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepu64_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const uint64_t a[4];
    const easysimd_float32 r[4];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT32_C(  -197.41), EASYSIMD_FLOAT32_C(   588.52), EASYSIMD_FLOAT32_C(  -556.69), EASYSIMD_FLOAT32_C(  -641.25) },
      UINT8_C(226),
      { UINT64_C( 4295036090515280357), UINT64_C(16525395114540634325), UINT64_C(14784777767069413705), UINT64_C( 7586707766132681612) },
      { EASYSIMD_FLOAT32_C(  -197.41), EASYSIMD_FLOAT32_C(16525394783170985984.00), EASYSIMD_FLOAT32_C(  -556.69), EASYSIMD_FLOAT32_C(  -641.25) } },
    { { EASYSIMD_FLOAT32_C(   293.36), EASYSIMD_FLOAT32_C(   265.11), EASYSIMD_FLOAT32_C(  -621.50), EASYSIMD_FLOAT32_C(  -943.88) },
      UINT8_C(247),
      { UINT64_C(12635679642012648829), UINT64_C(10972388836532305992), UINT64_C( 9340618575702135536), UINT64_C( 7546770649222023637) },
      { EASYSIMD_FLOAT32_C(12635679985378525184.00), EASYSIMD_FLOAT32_C(10972389372413870080.00), EASYSIMD_FLOAT32_C(9340618459282669568.00), EASYSIMD_FLOAT32_C(  -943.88) } },
    { { EASYSIMD_FLOAT32_C(  -576.50), EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(  -525.54), EASYSIMD_FLOAT32_C(   912.76) },
      UINT8_C(122),
      { UINT64_C( 2491613607905945808), UINT64_C(15883964233790139458), UINT64_C(12520698740911239373), UINT64_C( 8012305744818154032) },
      { EASYSIMD_FLOAT32_C(  -576.50), EASYSIMD_FLOAT32_C(15883963888782278656.00), EASYSIMD_FLOAT32_C(  -525.54), EASYSIMD_FLOAT32_C(8012305608592064512.00) } },
    { { EASYSIMD_FLOAT32_C(   820.63), EASYSIMD_FLOAT32_C(    68.31), EASYSIMD_FLOAT32_C(   279.05), EASYSIMD_FLOAT32_C(   193.19) },
      UINT8_C(241),
      { UINT64_C(10921096511793354771), UINT64_C(10453202304942566390), UINT64_C(16197047020861636350), UINT64_C(14524369067360794538) },
      { EASYSIMD_FLOAT32_C(10921096055466491904.00), EASYSIMD_FLOAT32_C(    68.31), EASYSIMD_FLOAT32_C(   279.05), EASYSIMD_FLOAT32_C(   193.19) } },
    { { EASYSIMD_FLOAT32_C(  -537.55), EASYSIMD_FLOAT32_C(  -346.68), EASYSIMD_FLOAT32_C(   397.46), EASYSIMD_FLOAT32_C(   746.63) },
      UINT8_C( 75),
      { UINT64_C(16582298538642326936), UINT64_C(16963384638664126777), UINT64_C(17132031636588022740), UINT64_C( 3157832113461496772) },
      { EASYSIMD_FLOAT32_C(16582298907954905088.00), EASYSIMD_FLOAT32_C(16963384140583927808.00), EASYSIMD_FLOAT32_C(   397.46), EASYSIMD_FLOAT32_C(3157832029588946944.00) } },
    { { EASYSIMD_FLOAT32_C(  -700.33), EASYSIMD_FLOAT32_C(   484.77), EASYSIMD_FLOAT32_C(   121.61), EASYSIMD_FLOAT32_C(  -862.70) },
      UINT8_C( 60),
      { UINT64_C(11994050740817933647), UINT64_C( 2892584144292437976), UINT64_C( 2406457406254747975), UINT64_C( 1123389448433013062) },
      { EASYSIMD_FLOAT32_C(  -700.33), EASYSIMD_FLOAT32_C(   484.77), EASYSIMD_FLOAT32_C(2406457394126127104.00), EASYSIMD_FLOAT32_C(1123389446918504448.00) } },
    { { EASYSIMD_FLOAT32_C(   -22.23), EASYSIMD_FLOAT32_C(  -909.36), EASYSIMD_FLOAT32_C(  -752.19), EASYSIMD_FLOAT32_C(  -552.44) },
      UINT8_C(122),
      { UINT64_C(  780940803283487363), UINT64_C( 6911939023188656025), UINT64_C( 7483294087758876041), UINT64_C( 8871974686082328983) },
      { EASYSIMD_FLOAT32_C(   -22.23), EASYSIMD_FLOAT32_C(6911939110525468672.00), EASYSIMD_FLOAT32_C(  -752.19), EASYSIMD_FLOAT32_C(8871974717687332864.00) } },
    { { EASYSIMD_FLOAT32_C(  -751.24), EASYSIMD_FLOAT32_C(  -807.31), EASYSIMD_FLOAT32_C(   590.44), EASYSIMD_FLOAT32_C(   892.64) },
      UINT8_C(231),
      { UINT64_C(11825937505436315444), UINT64_C( 2627928603138817559), UINT64_C(11962510661537029921), UINT64_C( 1213838696261254783) },
      { EASYSIMD_FLOAT32_C(11825938050033123328.00), EASYSIMD_FLOAT32_C(2627928722774163456.00), EASYSIMD_FLOAT32_C(11962510588342435840.00), EASYSIMD_FLOAT32_C(   892.64) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cvtepu64_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m128 r = easysimd_mm256_mask_cvtepu64_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_cvtepu64_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const uint64_t a[4];
    const easysimd_float32 r[4];
  } test_vec[8] = {
    { UINT8_C(155),
      { UINT64_C(16112683616892109656), UINT64_C( 1681857207718887662), UINT64_C(16412470971592106468), UINT64_C(16509298540707558677) },
      { EASYSIMD_FLOAT32_C(16112683198080614400.00), EASYSIMD_FLOAT32_C(1681857241189187584.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(16509299032451973120.00) } },
    { UINT8_C( 18),
      { UINT64_C(14188118956890614072), UINT64_C(15303815870530985046), UINT64_C( 2314881454543841084), UINT64_C(17707189178223927771) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(15303815374479294464.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(251),
      { UINT64_C(17355117956547210799), UINT64_C( 6993871121391245247), UINT64_C(15882203822153276017), UINT64_C(18241706931344970393) },
      { EASYSIMD_FLOAT32_C(17355118143327961088.00), EASYSIMD_FLOAT32_C(6993870868736638976.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(18241707145361883136.00) } },
    { UINT8_C(125),
      { UINT64_C(12729778775690149926), UINT64_C(11911681866017176741), UINT64_C( 6132983790733155223), UINT64_C(10763921348844800085) },
      { EASYSIMD_FLOAT32_C(12729778389506850816.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(6132984050513608704.00), EASYSIMD_FLOAT32_C(10763920868275912704.00) } },
    { UINT8_C(161),
      { UINT64_C(11933037259868766131), UINT64_C(11707343381002811413), UINT64_C( 6973888999176141750), UINT64_C(13916232862011468894) },
      { EASYSIMD_FLOAT32_C(11933037079648272384.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(228),
      { UINT64_C(14473076412481648154), UINT64_C( 3346228617084386019), UINT64_C( 6318733531934024470), UINT64_C( 7691345867305201525) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(6318733345886830592.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(220),
      { UINT64_C(14150579450301722570), UINT64_C( 7983796421892905604), UINT64_C(18268131489538973034), UINT64_C(16706585627827462102) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(18268131708312223744.00), EASYSIMD_FLOAT32_C(16706585503335448576.00) } },
    { UINT8_C(231),
      { UINT64_C(16105071025610318854), UINT64_C( 2250348466111395243), UINT64_C( 3402445124707044081), UINT64_C( 1429103492173870134) },
      { EASYSIMD_FLOAT32_C(16105071279081521152.00), EASYSIMD_FLOAT32_C(2250348458336583680.00), EASYSIMD_FLOAT32_C(3402445054344691712.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_cvtepu64_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m128 r = easysimd_mm256_maskz_cvtepu64_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepi32_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int32_t a[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { -INT32_C(  1205067714),  INT32_C(    56205114), -INT32_C(  1994837822),  INT32_C(   304728932), -INT32_C(   349792391),  INT32_C(  1819351522),  INT32_C(  1567100440), -INT32_C(   453584986) },
      { EASYSIMD_FLOAT64_C(-1205067714.00), EASYSIMD_FLOAT64_C(56205114.00), EASYSIMD_FLOAT64_C(-1994837822.00), EASYSIMD_FLOAT64_C(304728932.00),
        EASYSIMD_FLOAT64_C(-349792391.00), EASYSIMD_FLOAT64_C(1819351522.00), EASYSIMD_FLOAT64_C(1567100440.00), EASYSIMD_FLOAT64_C(-453584986.00) } },
    { {  INT32_C(   832316151), -INT32_C(  2093681215), -INT32_C(  1995682267), -INT32_C(  1835321831), -INT32_C(  1350647347), -INT32_C(   233050150), -INT32_C(  1538292482),  INT32_C(  1401439579) },
      { EASYSIMD_FLOAT64_C(832316151.00), EASYSIMD_FLOAT64_C(-2093681215.00), EASYSIMD_FLOAT64_C(-1995682267.00), EASYSIMD_FLOAT64_C(-1835321831.00),
        EASYSIMD_FLOAT64_C(-1350647347.00), EASYSIMD_FLOAT64_C(-233050150.00), EASYSIMD_FLOAT64_C(-1538292482.00), EASYSIMD_FLOAT64_C(1401439579.00) } },
    { {  INT32_C(   696525927),  INT32_C(  1051506969),  INT32_C(   550025479), -INT32_C(  1129094161), -INT32_C(     9686747),  INT32_C(   519210784),  INT32_C(  1740783883), -INT32_C(   289781113) },
      { EASYSIMD_FLOAT64_C(696525927.00), EASYSIMD_FLOAT64_C(1051506969.00), EASYSIMD_FLOAT64_C(550025479.00), EASYSIMD_FLOAT64_C(-1129094161.00),
        EASYSIMD_FLOAT64_C(-9686747.00), EASYSIMD_FLOAT64_C(519210784.00), EASYSIMD_FLOAT64_C(1740783883.00), EASYSIMD_FLOAT64_C(-289781113.00) } },
    { { -INT32_C(  2028519826), -INT32_C(    20527881),  INT32_C(  1814007421),  INT32_C(   371774193),  INT32_C(   588682243),  INT32_C(   658638876), -INT32_C(   795999159), -INT32_C(  1111537585) },
      { EASYSIMD_FLOAT64_C(-2028519826.00), EASYSIMD_FLOAT64_C(-20527881.00), EASYSIMD_FLOAT64_C(1814007421.00), EASYSIMD_FLOAT64_C(371774193.00),
        EASYSIMD_FLOAT64_C(588682243.00), EASYSIMD_FLOAT64_C(658638876.00), EASYSIMD_FLOAT64_C(-795999159.00), EASYSIMD_FLOAT64_C(-1111537585.00) } },
    { {  INT32_C(  2118506119),  INT32_C(   394070938), -INT32_C(  1971086183),  INT32_C(  1906420846),  INT32_C(  1553250112),  INT32_C(   142923455),  INT32_C(   718869211), -INT32_C(   488138661) },
      { EASYSIMD_FLOAT64_C(2118506119.00), EASYSIMD_FLOAT64_C(394070938.00), EASYSIMD_FLOAT64_C(-1971086183.00), EASYSIMD_FLOAT64_C(1906420846.00),
        EASYSIMD_FLOAT64_C(1553250112.00), EASYSIMD_FLOAT64_C(142923455.00), EASYSIMD_FLOAT64_C(718869211.00), EASYSIMD_FLOAT64_C(-488138661.00) } },
    { {  INT32_C(   157297774), -INT32_C(   803152585), -INT32_C(   413424519), -INT32_C(  1873216432),  INT32_C(  1928195507), -INT32_C(  1636142653), -INT32_C(   557296765),  INT32_C(  1522577643) },
      { EASYSIMD_FLOAT64_C(157297774.00), EASYSIMD_FLOAT64_C(-803152585.00), EASYSIMD_FLOAT64_C(-413424519.00), EASYSIMD_FLOAT64_C(-1873216432.00),
        EASYSIMD_FLOAT64_C(1928195507.00), EASYSIMD_FLOAT64_C(-1636142653.00), EASYSIMD_FLOAT64_C(-557296765.00), EASYSIMD_FLOAT64_C(1522577643.00) } },
    { {  INT32_C(   342041052),  INT32_C(  2028241918),  INT32_C(  2002730791), -INT32_C(   301418437),  INT32_C(  1751184805), -INT32_C(   385361050),  INT32_C(   449367854),  INT32_C(  1551140991) },
      { EASYSIMD_FLOAT64_C(342041052.00), EASYSIMD_FLOAT64_C(2028241918.00), EASYSIMD_FLOAT64_C(2002730791.00), EASYSIMD_FLOAT64_C(-301418437.00),
        EASYSIMD_FLOAT64_C(1751184805.00), EASYSIMD_FLOAT64_C(-385361050.00), EASYSIMD_FLOAT64_C(449367854.00), EASYSIMD_FLOAT64_C(1551140991.00) } },
    { { -INT32_C(  1468999767), -INT32_C(  2111810470), -INT32_C(   805732460), -INT32_C(   591527625),  INT32_C(  1548033782),  INT32_C(   675695865), -INT32_C(  1690169829),  INT32_C(  1089975958) },
      { EASYSIMD_FLOAT64_C(-1468999767.00), EASYSIMD_FLOAT64_C(-2111810470.00), EASYSIMD_FLOAT64_C(-805732460.00), EASYSIMD_FLOAT64_C(-591527625.00),
        EASYSIMD_FLOAT64_C(1548033782.00), EASYSIMD_FLOAT64_C(675695865.00), EASYSIMD_FLOAT64_C(-1690169829.00), EASYSIMD_FLOAT64_C(1089975958.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvtepi32_pd(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cvtepi32_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
  #else
    fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m512d r = easysimd_mm512_cvtepi32_pd(a);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_cvtepi16_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi16(INT16_C( 14423), INT16_C(  3775), INT16_C( 16156), INT16_C( 17811),
                            INT16_C(-14881), INT16_C(-30283), INT16_C( 27295), INT16_C(-12290),
                            INT16_C( 12394), INT16_C( 32764), INT16_C(  8681), INT16_C( 21255),
                            INT16_C(-21785), INT16_C(-24065), INT16_C(-28005), INT16_C( 15206),
                            INT16_C(  6131), INT16_C(-29323), INT16_C( -9530), INT16_C( -6655),
                            INT16_C( 14785), INT16_C( -9158), INT16_C(  7009), INT16_C(  4834),
                            INT16_C(-15579), INT16_C(  5296), INT16_C( 20054), INT16_C( 12832),
                            INT16_C( 15724), INT16_C(  5918), INT16_C( 25398), INT16_C( 14084)),
      easysimd_mm256_set_epi8(INT8_C(  87), INT8_C( -65), INT8_C(  28), INT8_C(-109),
                           INT8_C( -33), INT8_C( -75), INT8_C( -97), INT8_C(  -2),
                           INT8_C( 106), INT8_C(  -4), INT8_C( -23), INT8_C(   7),
                           INT8_C( -25), INT8_C(  -1), INT8_C(-101), INT8_C( 102),
                           INT8_C( -13), INT8_C( 117), INT8_C( -58), INT8_C(   1),
                           INT8_C( -63), INT8_C(  58), INT8_C(  97), INT8_C( -30),
                           INT8_C(  37), INT8_C( -80), INT8_C(  86), INT8_C(  32),
                           INT8_C( 108), INT8_C(  30), INT8_C(  54), INT8_C(   4)) },
    { easysimd_mm512_set_epi16(INT16_C(  8455), INT16_C(  1140), INT16_C(-23383), INT16_C( 22825),
                            INT16_C(-21438), INT16_C(  8713), INT16_C(-25940), INT16_C(-31180),
                            INT16_C(-13214), INT16_C( 10200), INT16_C(-21253), INT16_C(  2612),
                            INT16_C(-27891), INT16_C( 14031), INT16_C( -9014), INT16_C( 10287),
                            INT16_C(-11660), INT16_C( 26858), INT16_C(-19518), INT16_C(  2472),
                            INT16_C( 27637), INT16_C( 14857), INT16_C( 30034), INT16_C(-24153),
                            INT16_C( 31935), INT16_C( -6397), INT16_C( -2502), INT16_C( 31062),
                            INT16_C( 30236), INT16_C(  5156), INT16_C( 18439), INT16_C(-13074)),
      easysimd_mm256_set_epi8(INT8_C(   7), INT8_C( 116), INT8_C( -87), INT8_C(  41),
                           INT8_C(  66), INT8_C(   9), INT8_C( -84), INT8_C(  52),
                           INT8_C(  98), INT8_C( -40), INT8_C(  -5), INT8_C(  52),
                           INT8_C(  13), INT8_C( -49), INT8_C( -54), INT8_C(  47),
                           INT8_C( 116), INT8_C( -22), INT8_C( -62), INT8_C( -88),
                           INT8_C( -11), INT8_C(   9), INT8_C(  82), INT8_C( -89),
                           INT8_C( -65), INT8_C(   3), INT8_C(  58), INT8_C(  86),
                           INT8_C(  28), INT8_C(  36), INT8_C(   7), INT8_C( -18)) },
    { easysimd_mm512_set_epi16(INT16_C( 18175), INT16_C( -3760), INT16_C( 10318), INT16_C(-31849),
                            INT16_C(-32429), INT16_C(-26500), INT16_C( 24084), INT16_C(-23946),
                            INT16_C(  2525), INT16_C(  2478), INT16_C(-15141), INT16_C(-27410),
                            INT16_C( 30961), INT16_C(-31554), INT16_C( -9533), INT16_C(-20012),
                            INT16_C(-21820), INT16_C( 11767), INT16_C(-17849), INT16_C( 24518),
                            INT16_C(-22206), INT16_C(-24996), INT16_C(-19566), INT16_C( 17826),
                            INT16_C( 25765), INT16_C( 29123), INT16_C( 28065), INT16_C(  1432),
                            INT16_C(-24949), INT16_C( 30580), INT16_C( 20499), INT16_C(-29164)),
      easysimd_mm256_set_epi8(INT8_C(  -1), INT8_C(  80), INT8_C(  78), INT8_C(-105),
                           INT8_C(  83), INT8_C( 124), INT8_C(  20), INT8_C( 118),
                           INT8_C( -35), INT8_C( -82), INT8_C( -37), INT8_C( -18),
                           INT8_C( -15), INT8_C( -66), INT8_C( -61), INT8_C( -44),
                           INT8_C( -60), INT8_C(  -9), INT8_C(  71), INT8_C( -58),
                           INT8_C(  66), INT8_C(  92), INT8_C(-110), INT8_C( -94),
                           INT8_C( -91), INT8_C( -61), INT8_C( -95), INT8_C(-104),
                           INT8_C(-117), INT8_C( 116), INT8_C(  19), INT8_C(  20)) },
    { easysimd_mm512_set_epi16(INT16_C( 10816), INT16_C( 16713), INT16_C( 29707), INT16_C( 15186),
                            INT16_C( 31860), INT16_C(-28520), INT16_C( 18947), INT16_C(-27460),
                            INT16_C( 10883), INT16_C(   310), INT16_C(  8277), INT16_C(-28768),
                            INT16_C( -4553), INT16_C( 23273), INT16_C(-27696), INT16_C(-20678),
                            INT16_C( 13089), INT16_C( -6620), INT16_C( 31575), INT16_C(-20169),
                            INT16_C( 14440), INT16_C( -9264), INT16_C(-26919), INT16_C(-25720),
                            INT16_C(-18371), INT16_C( 25765), INT16_C(-13162), INT16_C(-16808),
                            INT16_C(  5695), INT16_C(-25080), INT16_C( 19142), INT16_C(  3825)),
      easysimd_mm256_set_epi8(INT8_C(  64), INT8_C(  73), INT8_C(  11), INT8_C(  82),
                           INT8_C( 116), INT8_C(-104), INT8_C(   3), INT8_C( -68),
                           INT8_C(-125), INT8_C(  54), INT8_C(  85), INT8_C( -96),
                           INT8_C(  55), INT8_C( -23), INT8_C( -48), INT8_C(  58),
                           INT8_C(  33), INT8_C(  36), INT8_C(  87), INT8_C(  55),
                           INT8_C( 104), INT8_C( -48), INT8_C( -39), INT8_C(-120),
                           INT8_C(  61), INT8_C( -91), INT8_C(-106), INT8_C(  88),
                           INT8_C(  63), INT8_C(   8), INT8_C( -58), INT8_C( -15)) },
    { easysimd_mm512_set_epi16(INT16_C(  5079), INT16_C(-24746), INT16_C( 23487), INT16_C(-22087),
                            INT16_C( -8346), INT16_C( 29848), INT16_C( 14241), INT16_C( 18254),
                            INT16_C( -3124), INT16_C(-16186), INT16_C(-13364), INT16_C( 10652),
                            INT16_C( 31028), INT16_C( 21346), INT16_C(  1443), INT16_C(-20222),
                            INT16_C(-17028), INT16_C(-21899), INT16_C( 18933), INT16_C(  6935),
                            INT16_C( 24619), INT16_C(  1737), INT16_C( 12596), INT16_C( 31606),
                            INT16_C(-32691), INT16_C( 11392), INT16_C( 32126), INT16_C(-32712),
                            INT16_C( 20927), INT16_C(-27859), INT16_C( 22640), INT16_C(  8969)),
      easysimd_mm256_set_epi8(INT8_C( -41), INT8_C(  86), INT8_C( -65), INT8_C( -71),
                           INT8_C( 102), INT8_C(-104), INT8_C( -95), INT8_C(  78),
                           INT8_C( -52), INT8_C( -58), INT8_C( -52), INT8_C(-100),
                           INT8_C(  52), INT8_C(  98), INT8_C( -93), INT8_C(   2),
                           INT8_C( 124), INT8_C( 117), INT8_C( -11), INT8_C(  23),
                           INT8_C(  43), INT8_C( -55), INT8_C(  52), INT8_C( 118),
                           INT8_C(  77), INT8_C(-128), INT8_C( 126), INT8_C(  56),
                           INT8_C( -65), INT8_C(  45), INT8_C( 112), INT8_C(   9)) },
    { easysimd_mm512_set_epi16(INT16_C(  6901), INT16_C(-23435), INT16_C(-26040), INT16_C(-11295),
                            INT16_C(   623), INT16_C(-23058), INT16_C( 17549), INT16_C(-23291),
                            INT16_C( 17215), INT16_C( -4892), INT16_C(  -849), INT16_C( 21086),
                            INT16_C(-13056), INT16_C( 19549), INT16_C( 16492), INT16_C(-22767),
                            INT16_C(-24079), INT16_C(  6429), INT16_C( 15302), INT16_C( -9175),
                            INT16_C( 17671), INT16_C(-29856), INT16_C(-12718), INT16_C(-22914),
                            INT16_C(-19613), INT16_C( 14088), INT16_C(-10443), INT16_C( 31757),
                            INT16_C( 24994), INT16_C( 24174), INT16_C( -9596), INT16_C(-22481)),
      easysimd_mm256_set_epi8(INT8_C( -11), INT8_C( 117), INT8_C(  72), INT8_C( -31),
                           INT8_C( 111), INT8_C( -18), INT8_C(-115), INT8_C(   5),
                           INT8_C(  63), INT8_C( -28), INT8_C( -81), INT8_C(  94),
                           INT8_C(   0), INT8_C(  93), INT8_C( 108), INT8_C(  17),
                           INT8_C( -15), INT8_C(  29), INT8_C( -58), INT8_C(  41),
                           INT8_C(   7), INT8_C(  96), INT8_C(  82), INT8_C( 126),
                           INT8_C(  99), INT8_C(   8), INT8_C(  53), INT8_C(  13),
                           INT8_C( -94), INT8_C( 110), INT8_C(-124), INT8_C(  47)) },
    { easysimd_mm512_set_epi16(INT16_C( 15520), INT16_C( 15679), INT16_C(  8541), INT16_C(-20376),
                            INT16_C(  8861), INT16_C( 12926), INT16_C( 25712), INT16_C( -8433),
                            INT16_C( -7066), INT16_C(-23691), INT16_C(-20251), INT16_C( 18056),
                            INT16_C(  5498), INT16_C(-18751), INT16_C(-26321), INT16_C(  7918),
                            INT16_C(  1647), INT16_C( 21774), INT16_C(  5430), INT16_C(-19512),
                            INT16_C(-14894), INT16_C( 12466), INT16_C( -9612), INT16_C(-23130),
                            INT16_C( 18357), INT16_C( 32349), INT16_C(-25760), INT16_C( -6559),
                            INT16_C(-24198), INT16_C( 13614), INT16_C( 13473), INT16_C(-25578)),
      easysimd_mm256_set_epi8(INT8_C( -96), INT8_C(  63), INT8_C(  93), INT8_C( 104),
                           INT8_C( -99), INT8_C( 126), INT8_C( 112), INT8_C(  15),
                           INT8_C( 102), INT8_C( 117), INT8_C( -27), INT8_C(-120),
                           INT8_C( 122), INT8_C( -63), INT8_C(  47), INT8_C( -18),
                           INT8_C( 111), INT8_C(  14), INT8_C(  54), INT8_C( -56),
                           INT8_C( -46), INT8_C( -78), INT8_C( 116), INT8_C( -90),
                           INT8_C( -75), INT8_C(  93), INT8_C(  96), INT8_C(  97),
                           INT8_C( 122), INT8_C(  46), INT8_C( -95), INT8_C(  22)) },
    { easysimd_mm512_set_epi16(INT16_C(-13944), INT16_C( 30422), INT16_C( 10523), INT16_C( 28986),
                            INT16_C(-23789), INT16_C(-20754), INT16_C( 29282), INT16_C(-10845),
                            INT16_C( 10721), INT16_C(  2777), INT16_C(-18838), INT16_C(  8324),
                            INT16_C( 19192), INT16_C(   114), INT16_C( -9073), INT16_C(  2615),
                            INT16_C( 21008), INT16_C( 12652), INT16_C(-14859), INT16_C(  5734),
                            INT16_C( -5598), INT16_C(-10707), INT16_C(  2170), INT16_C( 23903),
                            INT16_C( 29988), INT16_C( 24405), INT16_C(  5383), INT16_C(-29994),
                            INT16_C(  7143), INT16_C( 22270), INT16_C( -1480), INT16_C( 15491)),
      easysimd_mm256_set_epi8(INT8_C(-120), INT8_C( -42), INT8_C(  27), INT8_C(  58),
                           INT8_C(  19), INT8_C( -18), INT8_C(  98), INT8_C( -93),
                           INT8_C( -31), INT8_C( -39), INT8_C( 106), INT8_C(-124),
                           INT8_C(  -8), INT8_C( 114), INT8_C(-113), INT8_C(  55),
                           INT8_C(  16), INT8_C( 108), INT8_C( -11), INT8_C( 102),
                           INT8_C(  34), INT8_C(  45), INT8_C( 122), INT8_C(  95),
                           INT8_C(  36), INT8_C(  85), INT8_C(   7), INT8_C( -42),
                           INT8_C( -25), INT8_C(  -2), INT8_C(  56), INT8_C(-125)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i r = easysimd_mm512_cvtepi16_epi8(test_vec[i].a);
    easysimd_assert_m256i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_cvtepi16_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i src;
    easysimd__mmask32 k;
    easysimd__m512i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi8(INT8_C(-112), INT8_C(  50), INT8_C( -90), INT8_C( -47),
                           INT8_C(  24), INT8_C( -14), INT8_C( -76), INT8_C(  -4),
                           INT8_C(-104), INT8_C( 115), INT8_C( -75), INT8_C(  98),
                           INT8_C( -58), INT8_C( -14), INT8_C(  98), INT8_C(  29),
                           INT8_C( -34), INT8_C(  91), INT8_C(  -9), INT8_C( -32),
                           INT8_C( 105), INT8_C( -54), INT8_C(  11), INT8_C(  76),
                           INT8_C(  83), INT8_C(   3), INT8_C(  48), INT8_C(   2),
                           INT8_C(  92), INT8_C( -54), INT8_C(  99), INT8_C(  95)),
      UINT32_C(     36055),
      easysimd_mm512_set_epi16(INT16_C( 29253), INT16_C(-14914), INT16_C(  8284), INT16_C( 18521),
                            INT16_C( 32034), INT16_C( 27278), INT16_C( -3730), INT16_C( -7695),
                            INT16_C(  8989), INT16_C(-29300), INT16_C(-14890), INT16_C( 11419),
                            INT16_C( -1355), INT16_C( 25284), INT16_C(-28026), INT16_C(  1548),
                            INT16_C( 26140), INT16_C( -8634), INT16_C( 26242), INT16_C(  1035),
                            INT16_C(-29578), INT16_C( -2997), INT16_C( 22546), INT16_C(-28782),
                            INT16_C(-11973), INT16_C( 12912), INT16_C(-22923), INT16_C(-12898),
                            INT16_C(  4984), INT16_C(   989), INT16_C(  2511), INT16_C( 26483)),
      easysimd_mm256_set_epi8(INT8_C(-112), INT8_C(  50), INT8_C( -90), INT8_C( -47),
                           INT8_C(  24), INT8_C( -14), INT8_C( -76), INT8_C(  -4),
                           INT8_C(-104), INT8_C( 115), INT8_C( -75), INT8_C(  98),
                           INT8_C( -58), INT8_C( -14), INT8_C(  98), INT8_C(  29),
                           INT8_C(  28), INT8_C(  91), INT8_C(  -9), INT8_C( -32),
                           INT8_C( 118), INT8_C(  75), INT8_C(  11), INT8_C(  76),
                           INT8_C(  59), INT8_C( 112), INT8_C(  48), INT8_C( -98),
                           INT8_C(  92), INT8_C( -35), INT8_C( -49), INT8_C( 115)) },
    { easysimd_mm256_set_epi8(INT8_C( -93), INT8_C( -75), INT8_C( 109), INT8_C(  43),
                           INT8_C( -79), INT8_C( -91), INT8_C( -13), INT8_C( 103),
                           INT8_C(  -6), INT8_C( -39), INT8_C(   3), INT8_C(-115),
                           INT8_C(  30), INT8_C( -62), INT8_C(  30), INT8_C( 115),
                           INT8_C( -28), INT8_C( -92), INT8_C( 110), INT8_C( -10),
                           INT8_C(  20), INT8_C( -82), INT8_C(  59), INT8_C(  62),
                           INT8_C(  57), INT8_C(  97), INT8_C(  29), INT8_C(  -4),
                           INT8_C( -48), INT8_C(   1), INT8_C(  47), INT8_C(  43)),
      UINT32_C(     13848),
      easysimd_mm512_set_epi16(INT16_C( 19920), INT16_C( 28417), INT16_C(-26944), INT16_C( -1327),
                            INT16_C(-18966), INT16_C(-19374), INT16_C(  9639), INT16_C(-25572),
                            INT16_C(-16315), INT16_C( 16363), INT16_C( -4686), INT16_C(-14474),
                            INT16_C( 26743), INT16_C( 20737), INT16_C(-16355), INT16_C( 24251),
                            INT16_C(-20830), INT16_C( 19809), INT16_C(-32085), INT16_C(-29115),
                            INT16_C(-21999), INT16_C( 14843), INT16_C( 13075), INT16_C(-28846),
                            INT16_C(-12894), INT16_C( 31357), INT16_C( 16553), INT16_C(-16546),
                            INT16_C(-16544), INT16_C( 30528), INT16_C( -9494), INT16_C(  8241)),
      easysimd_mm256_set_epi8(INT8_C( -93), INT8_C( -75), INT8_C( 109), INT8_C(  43),
                           INT8_C( -79), INT8_C( -91), INT8_C( -13), INT8_C( 103),
                           INT8_C(  -6), INT8_C( -39), INT8_C(   3), INT8_C(-115),
                           INT8_C(  30), INT8_C( -62), INT8_C(  30), INT8_C( 115),
                           INT8_C( -28), INT8_C( -92), INT8_C( -85), INT8_C(  69),
                           INT8_C(  20), INT8_C(  -5), INT8_C(  19), INT8_C(  62),
                           INT8_C(  57), INT8_C(  97), INT8_C(  29), INT8_C(  94),
                           INT8_C(  96), INT8_C(   1), INT8_C(  47), INT8_C(  43)) },
    { easysimd_mm256_set_epi8(INT8_C(  57), INT8_C( 119), INT8_C(   6), INT8_C( -62),
                           INT8_C( -27), INT8_C( -22), INT8_C( -69), INT8_C( -61),
                           INT8_C(   8), INT8_C(-101), INT8_C( -24), INT8_C(  69),
                           INT8_C(-111), INT8_C(  66), INT8_C( -48), INT8_C(-122),
                           INT8_C( -19), INT8_C( -25), INT8_C( -88), INT8_C(  96),
                           INT8_C( -81), INT8_C(  28), INT8_C( -73), INT8_C(-105),
                           INT8_C( 109), INT8_C( -84), INT8_C(  26), INT8_C( 108),
                           INT8_C(  16), INT8_C(  69), INT8_C( -67), INT8_C(-122)),
      UINT32_C(     52950),
      easysimd_mm512_set_epi16(INT16_C(-28100), INT16_C(  2824), INT16_C(-32113), INT16_C(-30059),
                            INT16_C(-19864), INT16_C(-29923), INT16_C( 19573), INT16_C(-11183),
                            INT16_C(-18980), INT16_C( 26281), INT16_C( -7946), INT16_C( 14491),
                            INT16_C( 28715), INT16_C( 26138), INT16_C( 16023), INT16_C( 24398),
                            INT16_C( 20578), INT16_C( -1642), INT16_C( 24774), INT16_C( 26937),
                            INT16_C(-19881), INT16_C(-20408), INT16_C( 26365), INT16_C( -2980),
                            INT16_C( -4479), INT16_C(-10298), INT16_C( 13784), INT16_C(-25535),
                            INT16_C(-26583), INT16_C(-31618), INT16_C(  -202), INT16_C( 28295)),
      easysimd_mm256_set_epi8(INT8_C(  57), INT8_C( 119), INT8_C(   6), INT8_C( -62),
                           INT8_C( -27), INT8_C( -22), INT8_C( -69), INT8_C( -61),
                           INT8_C(   8), INT8_C(-101), INT8_C( -24), INT8_C(  69),
                           INT8_C(-111), INT8_C(  66), INT8_C( -48), INT8_C(-122),
                           INT8_C(  98), INT8_C(-106), INT8_C( -88), INT8_C(  96),
                           INT8_C(  87), INT8_C(  72), INT8_C(  -3), INT8_C(-105),
                           INT8_C(-127), INT8_C( -58), INT8_C(  26), INT8_C(  65),
                           INT8_C(  16), INT8_C( 126), INT8_C(  54), INT8_C(-122)) },
    { easysimd_mm256_set_epi8(INT8_C(  89), INT8_C(  16), INT8_C(  86), INT8_C( 124),
                           INT8_C(-106), INT8_C(  54), INT8_C(  30), INT8_C( -60),
                           INT8_C(  41), INT8_C(  45), INT8_C(-103), INT8_C( -75),
                           INT8_C( -46), INT8_C(  -2), INT8_C( 119), INT8_C(  28),
                           INT8_C(  69), INT8_C( -84), INT8_C(  78), INT8_C( -36),
                           INT8_C(  42), INT8_C( -59), INT8_C(  42), INT8_C(   5),
                           INT8_C( -74), INT8_C( -70), INT8_C( 107), INT8_C(  22),
                           INT8_C(  91), INT8_C(  10), INT8_C( -44), INT8_C(  28)),
      UINT32_C(      4183),
      easysimd_mm512_set_epi16(INT16_C(  8531), INT16_C(  2537), INT16_C(  7090), INT16_C( 32184),
                            INT16_C(   918), INT16_C( -4406), INT16_C( -1230), INT16_C(-20248),
                            INT16_C( 28454), INT16_C( -8033), INT16_C( 29491), INT16_C(  9038),
                            INT16_C( 31537), INT16_C(-32476), INT16_C( 15213), INT16_C(  2771),
                            INT16_C(  9158), INT16_C( 15700), INT16_C( 24392), INT16_C(-14500),
                            INT16_C( 20701), INT16_C( -9424), INT16_C( -5862), INT16_C(  8150),
                            INT16_C(-14293), INT16_C( 29409), INT16_C(-21051), INT16_C(-16951),
                            INT16_C(-32102), INT16_C(-16442), INT16_C(  4517), INT16_C(-32738)),
      easysimd_mm256_set_epi8(INT8_C(  89), INT8_C(  16), INT8_C(  86), INT8_C( 124),
                           INT8_C(-106), INT8_C(  54), INT8_C(  30), INT8_C( -60),
                           INT8_C(  41), INT8_C(  45), INT8_C(-103), INT8_C( -75),
                           INT8_C( -46), INT8_C(  -2), INT8_C( 119), INT8_C(  28),
                           INT8_C(  69), INT8_C( -84), INT8_C(  78), INT8_C(  92),
                           INT8_C(  42), INT8_C( -59), INT8_C(  42), INT8_C(   5),
                           INT8_C( -74), INT8_C( -31), INT8_C( 107), INT8_C( -55),
                           INT8_C(  91), INT8_C( -58), INT8_C( -91), INT8_C(  30)) },
    { easysimd_mm256_set_epi8(INT8_C(  66), INT8_C( -53), INT8_C( -22), INT8_C(-109),
                           INT8_C(-122), INT8_C( -34), INT8_C(  49), INT8_C( -51),
                           INT8_C(  45), INT8_C(  96), INT8_C(  21), INT8_C(   9),
                           INT8_C(-107), INT8_C(  88), INT8_C(  41), INT8_C(  63),
                           INT8_C( -15), INT8_C(  66), INT8_C( -60), INT8_C(  80),
                           INT8_C( -27), INT8_C(   9), INT8_C(  30), INT8_C( -73),
                           INT8_C( -55), INT8_C( -22), INT8_C(-122), INT8_C(  86),
                           INT8_C( -35), INT8_C( -54), INT8_C(  95), INT8_C( -17)),
      UINT32_C(     34749),
      easysimd_mm512_set_epi16(INT16_C(  6349), INT16_C( -1940), INT16_C( 12009), INT16_C( 26974),
                            INT16_C( 15374), INT16_C(  6913), INT16_C(-19915), INT16_C(-14530),
                            INT16_C(-31337), INT16_C( 22983), INT16_C(  6281), INT16_C(  -506),
                            INT16_C(-24168), INT16_C(-22228), INT16_C(-32449), INT16_C(-30658),
                            INT16_C(-16400), INT16_C( -7823), INT16_C( -6600), INT16_C( -5428),
                            INT16_C( 10840), INT16_C(-16201), INT16_C(-15359), INT16_C(-30650),
                            INT16_C(  6966), INT16_C(-30042), INT16_C( 32539), INT16_C(-32588),
                            INT16_C(-23367), INT16_C(-13235), INT16_C(-19835), INT16_C( 15017)),
      easysimd_mm256_set_epi8(INT8_C(  66), INT8_C( -53), INT8_C( -22), INT8_C(-109),
                           INT8_C(-122), INT8_C( -34), INT8_C(  49), INT8_C( -51),
                           INT8_C(  45), INT8_C(  96), INT8_C(  21), INT8_C(   9),
                           INT8_C(-107), INT8_C(  88), INT8_C(  41), INT8_C(  63),
                           INT8_C( -16), INT8_C(  66), INT8_C( -60), INT8_C(  80),
                           INT8_C( -27), INT8_C( -73), INT8_C(   1), INT8_C(  70),
                           INT8_C(  54), INT8_C( -22), INT8_C(  27), INT8_C( -76),
                           INT8_C( -71), INT8_C(  77), INT8_C(  95), INT8_C( -87)) },
    { easysimd_mm256_set_epi8(INT8_C(-124), INT8_C(  59), INT8_C( -81), INT8_C(  66),
                           INT8_C( -65), INT8_C( -38), INT8_C( -36), INT8_C(   5),
                           INT8_C(  15), INT8_C(  28), INT8_C( -18), INT8_C( -54),
                           INT8_C(  82), INT8_C(  30), INT8_C(-110), INT8_C(-114),
                           INT8_C(   3), INT8_C(  71), INT8_C(  64), INT8_C(  21),
                           INT8_C( 115), INT8_C( 123), INT8_C( -22), INT8_C(-111),
                           INT8_C( -10), INT8_C(  18), INT8_C(   3), INT8_C(  -8),
                           INT8_C( -97), INT8_C(  26), INT8_C(  72), INT8_C( -94)),
      UINT32_C(     31044),
      easysimd_mm512_set_epi16(INT16_C(-26750), INT16_C(-23902), INT16_C( 29963), INT16_C(  2819),
                            INT16_C(  9258), INT16_C( 16800), INT16_C(-21230), INT16_C( -2332),
                            INT16_C(-12889), INT16_C( 23107), INT16_C( 17922), INT16_C(  3552),
                            INT16_C( 16956), INT16_C(-21244), INT16_C( -9865), INT16_C( 24672),
                            INT16_C(-32513), INT16_C( -3970), INT16_C( 14993), INT16_C(-21626),
                            INT16_C(-29335), INT16_C( -2219), INT16_C(  4209), INT16_C( 11969),
                            INT16_C( -6560), INT16_C(-26729), INT16_C(  7233), INT16_C( 27170),
                            INT16_C(  5881), INT16_C( -9473), INT16_C(-30967), INT16_C(  3275)),
      easysimd_mm256_set_epi8(INT8_C(-124), INT8_C(  59), INT8_C( -81), INT8_C(  66),
                           INT8_C( -65), INT8_C( -38), INT8_C( -36), INT8_C(   5),
                           INT8_C(  15), INT8_C(  28), INT8_C( -18), INT8_C( -54),
                           INT8_C(  82), INT8_C(  30), INT8_C(-110), INT8_C(-114),
                           INT8_C(   3), INT8_C( 126), INT8_C(-111), INT8_C(-122),
                           INT8_C( 105), INT8_C( 123), INT8_C( -22), INT8_C( -63),
                           INT8_C( -10), INT8_C(-105), INT8_C(   3), INT8_C(  -8),
                           INT8_C( -97), INT8_C(  -1), INT8_C(  72), INT8_C( -94)) },
    { easysimd_mm256_set_epi8(INT8_C(  76), INT8_C( -68), INT8_C(   3), INT8_C( 100),
                           INT8_C(  64), INT8_C( -71), INT8_C( -39), INT8_C(  30),
                           INT8_C( 110), INT8_C(  44), INT8_C(  96), INT8_C(  10),
                           INT8_C(  66), INT8_C(  40), INT8_C(  31), INT8_C( -85),
                           INT8_C( 120), INT8_C(  70), INT8_C( -37), INT8_C( -25),
                           INT8_C(  51), INT8_C( -19), INT8_C( 124), INT8_C( -52),
                           INT8_C(  69), INT8_C( 107), INT8_C(  96), INT8_C( 106),
                           INT8_C(-126), INT8_C(  61), INT8_C( -71), INT8_C(   9)),
      UINT32_C(     63997),
      easysimd_mm512_set_epi16(INT16_C( 25271), INT16_C( 20153), INT16_C(-23804), INT16_C(-24091),
                            INT16_C(  6064), INT16_C(  3189), INT16_C( -2682), INT16_C(  5283),
                            INT16_C( 14900), INT16_C(   731), INT16_C(-14623), INT16_C( 14729),
                            INT16_C( -3836), INT16_C( 26379), INT16_C( 13131), INT16_C( 14975),
                            INT16_C( 19045), INT16_C( 14845), INT16_C(-21672), INT16_C(  4155),
                            INT16_C(  9032), INT16_C(-30375), INT16_C( 14167), INT16_C( 25860),
                            INT16_C( -6683), INT16_C(-21473), INT16_C( -6588), INT16_C( 22432),
                            INT16_C( -4408), INT16_C( -2180), INT16_C( 26333), INT16_C( 18369)),
      easysimd_mm256_set_epi8(INT8_C(  76), INT8_C( -68), INT8_C(   3), INT8_C( 100),
                           INT8_C(  64), INT8_C( -71), INT8_C( -39), INT8_C(  30),
                           INT8_C( 110), INT8_C(  44), INT8_C(  96), INT8_C(  10),
                           INT8_C(  66), INT8_C(  40), INT8_C(  31), INT8_C( -85),
                           INT8_C( 101), INT8_C(  -3), INT8_C(  88), INT8_C(  59),
                           INT8_C(  72), INT8_C( -19), INT8_C( 124), INT8_C(   4),
                           INT8_C( -27), INT8_C(  31), INT8_C(  68), INT8_C( -96),
                           INT8_C( -56), INT8_C( 124), INT8_C( -71), INT8_C( -63)) },
    { easysimd_mm256_set_epi8(INT8_C(  40), INT8_C( -41), INT8_C(-126), INT8_C(   8),
                           INT8_C(-115), INT8_C( 108), INT8_C(  31), INT8_C(  41),
                           INT8_C( -21), INT8_C( -60), INT8_C(  76), INT8_C(  74),
                           INT8_C(  86), INT8_C(  39), INT8_C(  41), INT8_C( -61),
                           INT8_C( 120), INT8_C(  -6), INT8_C(-117), INT8_C(  43),
                           INT8_C(  64), INT8_C( -40), INT8_C( -63), INT8_C(  39),
                           INT8_C(  82), INT8_C(  -3), INT8_C(  -8), INT8_C(-102),
                           INT8_C(  21), INT8_C(-109), INT8_C(  -6), INT8_C( 102)),
      UINT32_C(     16734),
      easysimd_mm512_set_epi16(INT16_C(-25905), INT16_C( 19727), INT16_C( 28735), INT16_C(  3852),
                            INT16_C(-23084), INT16_C( -6530), INT16_C( -1505), INT16_C(  9601),
                            INT16_C( -7362), INT16_C(  8505), INT16_C(-26382), INT16_C( 25139),
                            INT16_C(  4198), INT16_C( -1011), INT16_C( -5955), INT16_C( 29084),
                            INT16_C( 25996), INT16_C( 30463), INT16_C( -4775), INT16_C( 11032),
                            INT16_C(-28689), INT16_C(-14740), INT16_C( -1416), INT16_C(  8406),
                            INT16_C(-23209), INT16_C( 25079), INT16_C( 23521), INT16_C( 23507),
                            INT16_C( 15383), INT16_C(-27993), INT16_C(  2371), INT16_C(-19992)),
      easysimd_mm256_set_epi8(INT8_C(  40), INT8_C( -41), INT8_C(-126), INT8_C(   8),
                           INT8_C(-115), INT8_C( 108), INT8_C(  31), INT8_C(  41),
                           INT8_C( -21), INT8_C( -60), INT8_C(  76), INT8_C(  74),
                           INT8_C(  86), INT8_C(  39), INT8_C(  41), INT8_C( -61),
                           INT8_C( 120), INT8_C(  -1), INT8_C(-117), INT8_C(  43),
                           INT8_C(  64), INT8_C( -40), INT8_C( -63), INT8_C( -42),
                           INT8_C(  82), INT8_C(  -9), INT8_C(  -8), INT8_C( -45),
                           INT8_C(  23), INT8_C( -89), INT8_C(  67), INT8_C( 102)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i r = easysimd_mm512_mask_cvtepi16_epi8(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    easysimd_assert_m256i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_cvtepi16_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask32 k;
    easysimd__m512i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { UINT32_C(     25439),
      easysimd_mm512_set_epi16(INT16_C( 26140), INT16_C( -8634), INT16_C( 26242), INT16_C(  1035),
                            INT16_C(-29578), INT16_C( -2997), INT16_C( 22546), INT16_C(-28782),
                            INT16_C(-11973), INT16_C( 12912), INT16_C(-22923), INT16_C(-12898),
                            INT16_C(  4984), INT16_C(   989), INT16_C(  2511), INT16_C( 26483),
                            INT16_C(-18247), INT16_C( 15612), INT16_C( -5009), INT16_C(-29481),
                            INT16_C(-28622), INT16_C(-22831), INT16_C(  6386), INT16_C(-19204),
                            INT16_C(-26509), INT16_C(-19102), INT16_C(-14606), INT16_C( 25117),
                            INT16_C( -8613), INT16_C( -2080), INT16_C( 27082), INT16_C(  2892)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  -4), INT8_C( 111), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -14), INT8_C(  -4),
                           INT8_C(   0), INT8_C(  98), INT8_C(   0), INT8_C(  29),
                           INT8_C(  91), INT8_C( -32), INT8_C( -54), INT8_C(  76)) },
    { UINT32_C(      1548),
      easysimd_mm512_set_epi16(INT16_C( -5255), INT16_C( 10791), INT16_C(-28009), INT16_C( 13848),
                            INT16_C(-23627), INT16_C( 27947), INT16_C(-20059), INT16_C( -3225),
                            INT16_C( -1319), INT16_C(   909), INT16_C(  7874), INT16_C(  7795),
                            INT16_C( -7004), INT16_C( 28406), INT16_C(  5294), INT16_C( 15166),
                            INT16_C( 14689), INT16_C(  7676), INT16_C(-12287), INT16_C( 12075),
                            INT16_C( 29253), INT16_C(-14914), INT16_C(  8284), INT16_C( 18521),
                            INT16_C( 32034), INT16_C( 27278), INT16_C( -3730), INT16_C( -7695),
                            INT16_C(  8989), INT16_C(-29300), INT16_C(-14890), INT16_C( 11419)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -66), INT8_C(  92), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  29), INT8_C(-116), INT8_C(   0), INT8_C(   0)) },
    { UINT32_C(      8241),
      easysimd_mm512_set_epi16(INT16_C( 28076), INT16_C(  6764), INT16_C(  4165), INT16_C(-17018),
                            INT16_C( 19920), INT16_C( 28417), INT16_C(-26944), INT16_C( -1327),
                            INT16_C(-18966), INT16_C(-19374), INT16_C(  9639), INT16_C(-25572),
                            INT16_C(-16315), INT16_C( 16363), INT16_C( -4686), INT16_C(-14474),
                            INT16_C( 26743), INT16_C( 20737), INT16_C(-16355), INT16_C( 24251),
                            INT16_C(-20830), INT16_C( 19809), INT16_C(-32085), INT16_C(-29115),
                            INT16_C(-21999), INT16_C( 14843), INT16_C( 13075), INT16_C(-28846),
                            INT16_C(-12894), INT16_C( 31357), INT16_C( 16553), INT16_C(-16546)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  29), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  19), INT8_C(  82),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  94)) },
    { UINT32_C(     46999),
      easysimd_mm512_set_epi16(INT16_C( 28715), INT16_C( 26138), INT16_C( 16023), INT16_C( 24398),
                            INT16_C( 20578), INT16_C( -1642), INT16_C( 24774), INT16_C( 26937),
                            INT16_C(-19881), INT16_C(-20408), INT16_C( 26365), INT16_C( -2980),
                            INT16_C( -4479), INT16_C(-10298), INT16_C( 13784), INT16_C(-25535),
                            INT16_C(-26583), INT16_C(-31618), INT16_C(  -202), INT16_C( 28295),
                            INT16_C(-12554), INT16_C( -5929), INT16_C(-27764), INT16_C(-12586),
                            INT16_C( 14711), INT16_C(  1730), INT16_C( -6678), INT16_C(-17469),
                            INT16_C(  2203), INT16_C( -6075), INT16_C(-28350), INT16_C(-12154)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  41), INT8_C(   0), INT8_C(  54), INT8_C(-121),
                           INT8_C(   0), INT8_C( -41), INT8_C(-116), INT8_C( -42),
                           INT8_C( 119), INT8_C(   0), INT8_C(   0), INT8_C( -61),
                           INT8_C(   0), INT8_C(  69), INT8_C(  66), INT8_C(-122)) },
    { UINT32_C(     14491),
      easysimd_mm512_set_epi16(INT16_C(-32102), INT16_C(-16442), INT16_C(  4517), INT16_C(-32738),
                            INT16_C(  -320), INT16_C(  2839), INT16_C( 18963), INT16_C(  4183),
                            INT16_C( 22800), INT16_C( 22140), INT16_C(-27082), INT16_C(  7876),
                            INT16_C( 10541), INT16_C(-26187), INT16_C(-11522), INT16_C( 30492),
                            INT16_C( 17836), INT16_C( 20188), INT16_C( 10949), INT16_C( 10757),
                            INT16_C(-18758), INT16_C( 27414), INT16_C( 23306), INT16_C(-11236),
                            INT16_C(-28100), INT16_C(  2824), INT16_C(-32113), INT16_C(-30059),
                            INT16_C(-19864), INT16_C(-29923), INT16_C( 19573), INT16_C(-11183)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -59), INT8_C(   5),
                           INT8_C( -70), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  60), INT8_C(   0), INT8_C(   0), INT8_C(-107),
                           INT8_C( 104), INT8_C(   0), INT8_C( 117), INT8_C(  81)) },
    { UINT32_C(     48585),
      easysimd_mm512_set_epi16(INT16_C( -3774), INT16_C(-15280), INT16_C( -6903), INT16_C(  7863),
                            INT16_C(-13846), INT16_C(-31146), INT16_C( -8758), INT16_C( 24559),
                            INT16_C(  8531), INT16_C(  2537), INT16_C(  7090), INT16_C( 32184),
                            INT16_C(   918), INT16_C( -4406), INT16_C( -1230), INT16_C(-20248),
                            INT16_C( 28454), INT16_C( -8033), INT16_C( 29491), INT16_C(  9038),
                            INT16_C( 31537), INT16_C(-32476), INT16_C( 15213), INT16_C(  2771),
                            INT16_C(  9158), INT16_C( 15700), INT16_C( 24392), INT16_C(-14500),
                            INT16_C( 20701), INT16_C( -9424), INT16_C( -5862), INT16_C(  8150)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  38), INT8_C(   0), INT8_C(  51), INT8_C(  78),
                           INT8_C(  49), INT8_C(  36), INT8_C(   0), INT8_C( -45),
                           INT8_C( -58), INT8_C(  84), INT8_C(   0), INT8_C(   0),
                           INT8_C( -35), INT8_C(   0), INT8_C(   0), INT8_C( -42)) },
    { UINT32_C(     10559),
      easysimd_mm512_set_epi16(INT16_C(-31337), INT16_C( 22983), INT16_C(  6281), INT16_C(  -506),
                            INT16_C(-24168), INT16_C(-22228), INT16_C(-32449), INT16_C(-30658),
                            INT16_C(-16400), INT16_C( -7823), INT16_C( -6600), INT16_C( -5428),
                            INT16_C( 10840), INT16_C(-16201), INT16_C(-15359), INT16_C(-30650),
                            INT16_C(  6966), INT16_C(-30042), INT16_C( 32539), INT16_C(-32588),
                            INT16_C(-23367), INT16_C(-13235), INT16_C(-19835), INT16_C( 15017),
                            INT16_C( -4677), INT16_C(-14834), INT16_C(  9957), INT16_C(-30787),
                            INT16_C( 17099), INT16_C( -5485), INT16_C(-31010), INT16_C( 12749)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  27), INT8_C(   0),
                           INT8_C( -71), INT8_C(   0), INT8_C(   0), INT8_C( -87),
                           INT8_C(   0), INT8_C(   0), INT8_C( -27), INT8_C( -67),
                           INT8_C( -53), INT8_C(-109), INT8_C( -34), INT8_C( -51)) },
    { UINT32_C(     51006),
      easysimd_mm512_set_epi16(INT16_C( -6560), INT16_C(-26729), INT16_C(  7233), INT16_C( 27170),
                            INT16_C(  5881), INT16_C( -9473), INT16_C(-30967), INT16_C(  3275),
                            INT16_C( -2646), INT16_C( 14621), INT16_C( 19871), INT16_C( 31044),
                            INT16_C(-31685), INT16_C(-20670), INT16_C(-16422), INT16_C( -9211),
                            INT16_C(  3868), INT16_C( -4406), INT16_C( 21022), INT16_C(-28018),
                            INT16_C(   839), INT16_C( 16405), INT16_C( 29563), INT16_C( -5487),
                            INT16_C( -2542), INT16_C(  1016), INT16_C(-24806), INT16_C( 18594),
                            INT16_C(  6349), INT16_C( -1940), INT16_C( 12009), INT16_C( 26974)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  28), INT8_C( -54), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  21), INT8_C( 123), INT8_C(-111),
                           INT8_C(   0), INT8_C(   0), INT8_C(  26), INT8_C( -94),
                           INT8_C( -51), INT8_C( 108), INT8_C( -23), INT8_C(   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i r = easysimd_mm512_maskz_cvtepi16_epi8(test_vec[i].k, test_vec[i].a);
    easysimd_assert_m256i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cvtepi8_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi8(INT8_C(   7), INT8_C(  68), INT8_C( -86), INT8_C( -36),
                           INT8_C( -19), INT8_C(  73), INT8_C(  92), INT8_C( -27),
                           INT8_C(  55), INT8_C( -65), INT8_C( -50), INT8_C(  19),
                           INT8_C(-111), INT8_C( -79), INT8_C( -16), INT8_C(  70),
                           INT8_C(  27), INT8_C( -28), INT8_C( 116), INT8_C(  42),
                           INT8_C(  -4), INT8_C(  78), INT8_C(  31), INT8_C(  51),
                           INT8_C(  92), INT8_C(  39), INT8_C(-125), INT8_C(  94),
                           INT8_C( -78), INT8_C(  67), INT8_C( -43), INT8_C( -71)),
      easysimd_mm512_set_epi16(INT16_C(     7), INT16_C(    68), INT16_C(   -86), INT16_C(   -36),
                            INT16_C(   -19), INT16_C(    73), INT16_C(    92), INT16_C(   -27),
                            INT16_C(    55), INT16_C(   -65), INT16_C(   -50), INT16_C(    19),
                            INT16_C(  -111), INT16_C(   -79), INT16_C(   -16), INT16_C(    70),
                            INT16_C(    27), INT16_C(   -28), INT16_C(   116), INT16_C(    42),
                            INT16_C(    -4), INT16_C(    78), INT16_C(    31), INT16_C(    51),
                            INT16_C(    92), INT16_C(    39), INT16_C(  -125), INT16_C(    94),
                            INT16_C(   -78), INT16_C(    67), INT16_C(   -43), INT16_C(   -71)) },
    { easysimd_mm256_set_epi8(INT8_C(  29), INT8_C( -37), INT8_C(  27), INT8_C(  10),
                           INT8_C( -22), INT8_C(  -9), INT8_C(-125), INT8_C(  -3),
                           INT8_C( -53), INT8_C(  92), INT8_C( 103), INT8_C(  92),
                           INT8_C( 123), INT8_C(  74), INT8_C(  36), INT8_C(  59),
                           INT8_C(  46), INT8_C( -29), INT8_C(-103), INT8_C(  -4),
                           INT8_C( 109), INT8_C( -54), INT8_C(  41), INT8_C(  79),
                           INT8_C(  15), INT8_C( -92), INT8_C( 102), INT8_C( 116),
                           INT8_C( -42), INT8_C(  52), INT8_C( -61), INT8_C( -99)),
      easysimd_mm512_set_epi16(INT16_C(    29), INT16_C(   -37), INT16_C(    27), INT16_C(    10),
                            INT16_C(   -22), INT16_C(    -9), INT16_C(  -125), INT16_C(    -3),
                            INT16_C(   -53), INT16_C(    92), INT16_C(   103), INT16_C(    92),
                            INT16_C(   123), INT16_C(    74), INT16_C(    36), INT16_C(    59),
                            INT16_C(    46), INT16_C(   -29), INT16_C(  -103), INT16_C(    -4),
                            INT16_C(   109), INT16_C(   -54), INT16_C(    41), INT16_C(    79),
                            INT16_C(    15), INT16_C(   -92), INT16_C(   102), INT16_C(   116),
                            INT16_C(   -42), INT16_C(    52), INT16_C(   -61), INT16_C(   -99)) },
    { easysimd_mm256_set_epi8(INT8_C(  -9), INT8_C( -47), INT8_C( 107), INT8_C( -74),
                           INT8_C(-126), INT8_C(  34), INT8_C(  64), INT8_C( 115),
                           INT8_C( -65), INT8_C(-124), INT8_C(  54), INT8_C(  27),
                           INT8_C(  41), INT8_C( 112), INT8_C(  61), INT8_C(   6),
                           INT8_C(   7), INT8_C(  39), INT8_C(-109), INT8_C( -99),
                           INT8_C(  63), INT8_C( -35), INT8_C(-111), INT8_C( -72),
                           INT8_C( 109), INT8_C( -39), INT8_C( -99), INT8_C(  26),
                           INT8_C(  66), INT8_C( -78), INT8_C(  30), INT8_C(  38)),
      easysimd_mm512_set_epi16(INT16_C(    -9), INT16_C(   -47), INT16_C(   107), INT16_C(   -74),
                            INT16_C(  -126), INT16_C(    34), INT16_C(    64), INT16_C(   115),
                            INT16_C(   -65), INT16_C(  -124), INT16_C(    54), INT16_C(    27),
                            INT16_C(    41), INT16_C(   112), INT16_C(    61), INT16_C(     6),
                            INT16_C(     7), INT16_C(    39), INT16_C(  -109), INT16_C(   -99),
                            INT16_C(    63), INT16_C(   -35), INT16_C(  -111), INT16_C(   -72),
                            INT16_C(   109), INT16_C(   -39), INT16_C(   -99), INT16_C(    26),
                            INT16_C(    66), INT16_C(   -78), INT16_C(    30), INT16_C(    38)) },
    { easysimd_mm256_set_epi8(INT8_C( -72), INT8_C( -80), INT8_C( 101), INT8_C(  81),
                           INT8_C(  23), INT8_C( -68), INT8_C( -57), INT8_C(-111),
                           INT8_C(  -3), INT8_C(  21), INT8_C( 121), INT8_C( -22),
                           INT8_C(-104), INT8_C( -10), INT8_C( -37), INT8_C(  66),
                           INT8_C( -93), INT8_C( -80), INT8_C(  34), INT8_C( 104),
                           INT8_C( -39), INT8_C( -99), INT8_C(  18), INT8_C( 110),
                           INT8_C(-118), INT8_C(  38), INT8_C( 112), INT8_C( -67),
                           INT8_C(  60), INT8_C(  47), INT8_C(  32), INT8_C(  33)),
      easysimd_mm512_set_epi16(INT16_C(   -72), INT16_C(   -80), INT16_C(   101), INT16_C(    81),
                            INT16_C(    23), INT16_C(   -68), INT16_C(   -57), INT16_C(  -111),
                            INT16_C(    -3), INT16_C(    21), INT16_C(   121), INT16_C(   -22),
                            INT16_C(  -104), INT16_C(   -10), INT16_C(   -37), INT16_C(    66),
                            INT16_C(   -93), INT16_C(   -80), INT16_C(    34), INT16_C(   104),
                            INT16_C(   -39), INT16_C(   -99), INT16_C(    18), INT16_C(   110),
                            INT16_C(  -118), INT16_C(    38), INT16_C(   112), INT16_C(   -67),
                            INT16_C(    60), INT16_C(    47), INT16_C(    32), INT16_C(    33)) },
    { easysimd_mm256_set_epi8(INT8_C( 120), INT8_C( -90), INT8_C(-101), INT8_C(-106),
                           INT8_C(  70), INT8_C( -49), INT8_C(  29), INT8_C( -43),
                           INT8_C( -42), INT8_C(  38), INT8_C(  16), INT8_C( -43),
                           INT8_C( -40), INT8_C( -76), INT8_C( -67), INT8_C(  53),
                           INT8_C( -73), INT8_C( -17), INT8_C(  66), INT8_C(  57),
                           INT8_C( -65), INT8_C( -63), INT8_C(  17), INT8_C(  -9),
                           INT8_C(  95), INT8_C( -50), INT8_C(-118), INT8_C( 114),
                           INT8_C(  58), INT8_C( -28), INT8_C( -81), INT8_C( -37)),
      easysimd_mm512_set_epi16(INT16_C(   120), INT16_C(   -90), INT16_C(  -101), INT16_C(  -106),
                            INT16_C(    70), INT16_C(   -49), INT16_C(    29), INT16_C(   -43),
                            INT16_C(   -42), INT16_C(    38), INT16_C(    16), INT16_C(   -43),
                            INT16_C(   -40), INT16_C(   -76), INT16_C(   -67), INT16_C(    53),
                            INT16_C(   -73), INT16_C(   -17), INT16_C(    66), INT16_C(    57),
                            INT16_C(   -65), INT16_C(   -63), INT16_C(    17), INT16_C(    -9),
                            INT16_C(    95), INT16_C(   -50), INT16_C(  -118), INT16_C(   114),
                            INT16_C(    58), INT16_C(   -28), INT16_C(   -81), INT16_C(   -37)) },
    { easysimd_mm256_set_epi8(INT8_C( -97), INT8_C(  10), INT8_C( -75), INT8_C(-120),
                           INT8_C( -32), INT8_C(-105), INT8_C( -75), INT8_C(-101),
                           INT8_C(  71), INT8_C(-122), INT8_C(-112), INT8_C(  -2),
                           INT8_C(  60), INT8_C( -71), INT8_C( 101), INT8_C(  -1),
                           INT8_C(  95), INT8_C( -58), INT8_C( -70), INT8_C( 102),
                           INT8_C( 115), INT8_C( -68), INT8_C(-110), INT8_C( -36),
                           INT8_C(   6), INT8_C(  58), INT8_C(  73), INT8_C(  97),
                           INT8_C( -51), INT8_C(  -4), INT8_C(  58), INT8_C(  31)),
      easysimd_mm512_set_epi16(INT16_C(   -97), INT16_C(    10), INT16_C(   -75), INT16_C(  -120),
                            INT16_C(   -32), INT16_C(  -105), INT16_C(   -75), INT16_C(  -101),
                            INT16_C(    71), INT16_C(  -122), INT16_C(  -112), INT16_C(    -2),
                            INT16_C(    60), INT16_C(   -71), INT16_C(   101), INT16_C(    -1),
                            INT16_C(    95), INT16_C(   -58), INT16_C(   -70), INT16_C(   102),
                            INT16_C(   115), INT16_C(   -68), INT16_C(  -110), INT16_C(   -36),
                            INT16_C(     6), INT16_C(    58), INT16_C(    73), INT16_C(    97),
                            INT16_C(   -51), INT16_C(    -4), INT16_C(    58), INT16_C(    31)) },
    { easysimd_mm256_set_epi8(INT8_C( -73), INT8_C(-123), INT8_C( -11), INT8_C(  62),
                           INT8_C( -96), INT8_C(-103), INT8_C(  85), INT8_C(  88),
                           INT8_C( -19), INT8_C(  28), INT8_C(-107), INT8_C( -81),
                           INT8_C(-125), INT8_C(  88), INT8_C(  84), INT8_C( 115),
                           INT8_C( 105), INT8_C( -47), INT8_C(  68), INT8_C(-124),
                           INT8_C(  32), INT8_C(-100), INT8_C(  10), INT8_C( -69),
                           INT8_C( 124), INT8_C( -51), INT8_C( -89), INT8_C( -72),
                           INT8_C( -92), INT8_C(  -5), INT8_C( -46), INT8_C( 115)),
      easysimd_mm512_set_epi16(INT16_C(   -73), INT16_C(  -123), INT16_C(   -11), INT16_C(    62),
                            INT16_C(   -96), INT16_C(  -103), INT16_C(    85), INT16_C(    88),
                            INT16_C(   -19), INT16_C(    28), INT16_C(  -107), INT16_C(   -81),
                            INT16_C(  -125), INT16_C(    88), INT16_C(    84), INT16_C(   115),
                            INT16_C(   105), INT16_C(   -47), INT16_C(    68), INT16_C(  -124),
                            INT16_C(    32), INT16_C(  -100), INT16_C(    10), INT16_C(   -69),
                            INT16_C(   124), INT16_C(   -51), INT16_C(   -89), INT16_C(   -72),
                            INT16_C(   -92), INT16_C(    -5), INT16_C(   -46), INT16_C(   115)) },
    { easysimd_mm256_set_epi8(INT8_C( 104), INT8_C(  66), INT8_C(  51), INT8_C(  81),
                           INT8_C( -69), INT8_C( 104), INT8_C( 126), INT8_C( -43),
                           INT8_C( -40), INT8_C(  23), INT8_C(-124), INT8_C(  98),
                           INT8_C(-125), INT8_C(  95), INT8_C( -36), INT8_C(  46),
                           INT8_C(-115), INT8_C( -93), INT8_C(   2), INT8_C( -77),
                           INT8_C(  80), INT8_C(-116), INT8_C(  61), INT8_C( -89),
                           INT8_C( -37), INT8_C(   9), INT8_C(  84), INT8_C( -64),
                           INT8_C(  94), INT8_C(  67), INT8_C( -53), INT8_C( 111)),
      easysimd_mm512_set_epi16(INT16_C(   104), INT16_C(    66), INT16_C(    51), INT16_C(    81),
                            INT16_C(   -69), INT16_C(   104), INT16_C(   126), INT16_C(   -43),
                            INT16_C(   -40), INT16_C(    23), INT16_C(  -124), INT16_C(    98),
                            INT16_C(  -125), INT16_C(    95), INT16_C(   -36), INT16_C(    46),
                            INT16_C(  -115), INT16_C(   -93), INT16_C(     2), INT16_C(   -77),
                            INT16_C(    80), INT16_C(  -116), INT16_C(    61), INT16_C(   -89),
                            INT16_C(   -37), INT16_C(     9), INT16_C(    84), INT16_C(   -64),
                            INT16_C(    94), INT16_C(    67), INT16_C(   -53), INT16_C(   111)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_cvtepi8_epi16(test_vec[i].a);
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cvtepi64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { -INT64_C( 6990754942988407763),  INT64_C( 2832793691089995699),  INT64_C( 2544370250689205702), -INT64_C( 5460348983824763727),
         INT64_C( 2496875934845736605), -INT64_C( 1605566099757135434),  INT64_C( 4187611573269962423),  INT64_C( 2401253696001249814) },
      { EASYSIMD_FLOAT32_C(-6990754852783521792.00), EASYSIMD_FLOAT32_C(2832793577552084992.00), EASYSIMD_FLOAT32_C(2544370237109698560.00), EASYSIMD_FLOAT32_C(-5460349016128618496.00),
        EASYSIMD_FLOAT32_C(2496876007714193408.00), EASYSIMD_FLOAT32_C(-1605566115067461632.00), EASYSIMD_FLOAT32_C(4187611530419765248.00), EASYSIMD_FLOAT32_C(2401253680469770240.00) } },
    { {  INT64_C( 7646777593553290637),  INT64_C( 5325139953791512793),  INT64_C( 8183991988534966480), -INT64_C( 2983528884486647052),
         INT64_C( 7675806745445450086),  INT64_C(  251159457720336545),  INT64_C( 1590034699954995269), -INT64_C( 1917928162045649086) },
      { EASYSIMD_FLOAT32_C(7646777365984444416.00), EASYSIMD_FLOAT32_C(5325139872237748224.00), EASYSIMD_FLOAT32_C(8183992150246031360.00), EASYSIMD_FLOAT32_C(-2983528925138780160.00),
        EASYSIMD_FLOAT32_C(7675806671980986368.00), EASYSIMD_FLOAT32_C(251159465447915520.00), EASYSIMD_FLOAT32_C(1590034688691404800.00), EASYSIMD_FLOAT32_C(-1917928160108740608.00) } },
    { { -INT64_C( 4043523115156752921), -INT64_C( 2721732641530689275), -INT64_C( 3931771635744729289),  INT64_C(  724719046915128489),
        -INT64_C( 7432516552659693843), -INT64_C( 9142858879433911122), -INT64_C( 3678797534545143647), -INT64_C( 1409059337901095098) },
      { EASYSIMD_FLOAT32_C(-4043523005500882944.00), EASYSIMD_FLOAT32_C(-2721732732664152064.00), EASYSIMD_FLOAT32_C(-3931771667310706688.00), EASYSIMD_FLOAT32_C(724719037631692800.00),
        EASYSIMD_FLOAT32_C(-7432516634591363072.00), EASYSIMD_FLOAT32_C(-9142858648643436544.00), EASYSIMD_FLOAT32_C(-3678797406381539328.00), EASYSIMD_FLOAT32_C(-1409059335367032832.00) } },
    { {  INT64_C(  594151184214783250), -INT64_C( 5009857554507166035), -INT64_C( 2021882461295586303), -INT64_C(  429883860314168036),
        -INT64_C( 8208409687635261517), -INT64_C( 3693513776737918484), -INT64_C( 7938191970995756716), -INT64_C( 7806789981750615611) },
      { EASYSIMD_FLOAT32_C(594151207199571968.00), EASYSIMD_FLOAT32_C(-5009857661752049664.00), EASYSIMD_FLOAT32_C(-2021882449029496832.00), EASYSIMD_FLOAT32_C(-429883860772192256.00),
        EASYSIMD_FLOAT32_C(-8208409554719866880.00), EASYSIMD_FLOAT32_C(-3693513819763507200.00), EASYSIMD_FLOAT32_C(-7938191927810195456.00), EASYSIMD_FLOAT32_C(-7806789842930499584.00) } },
    { {  INT64_C( 1309621868627511484),  INT64_C(  481275699455767063),  INT64_C( 4791211519305620854), -INT64_C( 7250627313066300690),
        -INT64_C( 3596012465697521184),  INT64_C( 8777617944611859991), -INT64_C( 8954595777474868998),  INT64_C( 2487870745142733324) },
      { EASYSIMD_FLOAT32_C(1309621840213180416.00), EASYSIMD_FLOAT32_C(481275687089471488.00), EASYSIMD_FLOAT32_C(4791211528231583744.00), EASYSIMD_FLOAT32_C(-7250627174787448832.00),
        EASYSIMD_FLOAT32_C(-3596012427147214848.00), EASYSIMD_FLOAT32_C(8777617928326479872.00), EASYSIMD_FLOAT32_C(-8954595869689118720.00), EASYSIMD_FLOAT32_C(2487870732604801024.00) } },
    { {  INT64_C( 4339007291122040754),  INT64_C( 4534234514314422127),  INT64_C( 4591544284845855135),  INT64_C( 7890500861643327771),
         INT64_C( 6399631466880739696),  INT64_C( 5510074623737976565), -INT64_C( 5009629378947043832), -INT64_C( 4293732179435281343) },
      { EASYSIMD_FLOAT32_C(4339007409616846848.00), EASYSIMD_FLOAT32_C(4534234495221497856.00), EASYSIMD_FLOAT32_C(4591544339796066304.00), EASYSIMD_FLOAT32_C(7890500610955411456.00),
        EASYSIMD_FLOAT32_C(6399631563167891456.00), EASYSIMD_FLOAT32_C(5510074429494788096.00), EASYSIMD_FLOAT32_C(-5009629513089286144.00), EASYSIMD_FLOAT32_C(-4293732269564100608.00) } },
    { {  INT64_C( 4793001314845540162),  INT64_C( 1079160795737837312), -INT64_C(  902193602836727519),  INT64_C( 7962461090345882471),
         INT64_C(  155406085423487384), -INT64_C( 5223990203627101332), -INT64_C(  880468515453679326), -INT64_C( 6404348400645458043) },
      { EASYSIMD_FLOAT32_C(4793001533161603072.00), EASYSIMD_FLOAT32_C(1079160767057494016.00), EASYSIMD_FLOAT32_C(-902193614652702720.00), EASYSIMD_FLOAT32_C(7962460898214281216.00),
        EASYSIMD_FLOAT32_C(155406090161356800.00), EASYSIMD_FLOAT32_C(-5223990300040495104.00), EASYSIMD_FLOAT32_C(-880468501838430208.00), EASYSIMD_FLOAT32_C(-6404348468051050496.00) } },
    { {  INT64_C( 4759660312105170952), -INT64_C( 8672185320525677480), -INT64_C( 7674377785563209985),  INT64_C( 7318460848497110745),
         INT64_C( 3113813437842398493),  INT64_C( 2244411235638111033),  INT64_C( 8972061761929445598),  INT64_C( 5483965077237647704) },
      { EASYSIMD_FLOAT32_C(4759660492316737536.00), EASYSIMD_FLOAT32_C(-8672185208583225344.00), EASYSIMD_FLOAT32_C(-7674377856620691456.00), EASYSIMD_FLOAT32_C(7318460994907275264.00),
        EASYSIMD_FLOAT32_C(3113813356448841728.00), EASYSIMD_FLOAT32_C(2244411232985546752.00), EASYSIMD_FLOAT32_C(8972061611896340480.00), EASYSIMD_FLOAT32_C(5483964876625805312.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvtepi64_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m256 r = easysimd_mm512_cvtepi64_ps(a);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepi64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const int64_t a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   217.76), EASYSIMD_FLOAT32_C(   166.48), EASYSIMD_FLOAT32_C(  -155.25), EASYSIMD_FLOAT32_C(   624.51),
        EASYSIMD_FLOAT32_C(  -371.54), EASYSIMD_FLOAT32_C(  -538.23), EASYSIMD_FLOAT32_C(   808.09), EASYSIMD_FLOAT32_C(   811.60) },
      UINT8_C(121),
      {  INT64_C( 6738468720793335188),  INT64_C( 4435010719342786039), -INT64_C( 7730517527152118834),  INT64_C( 2129100398026116311),
         INT64_C( 6107198150888227787),  INT64_C( 6490607959171217941), -INT64_C( 4264502108403122696),  INT64_C( 3414683324324990653) },
      { EASYSIMD_FLOAT32_C(6738468611255435264.00), EASYSIMD_FLOAT32_C(   166.48), EASYSIMD_FLOAT32_C(  -155.25), EASYSIMD_FLOAT32_C(2129100363339399168.00),
        EASYSIMD_FLOAT32_C(6107198303819005952.00), EASYSIMD_FLOAT32_C(6490607904029147136.00), EASYSIMD_FLOAT32_C(-4264502027817582592.00), EASYSIMD_FLOAT32_C(   811.60) } },
    { { EASYSIMD_FLOAT32_C(  -868.33), EASYSIMD_FLOAT32_C(   479.96), EASYSIMD_FLOAT32_C(  -615.79), EASYSIMD_FLOAT32_C(   926.08),
        EASYSIMD_FLOAT32_C(   599.06), EASYSIMD_FLOAT32_C(   893.90), EASYSIMD_FLOAT32_C(   280.11), EASYSIMD_FLOAT32_C(   276.77) },
      UINT8_C(142),
      {  INT64_C( 6669677247891901265), -INT64_C( 1487017331328955346),  INT64_C(  445994928557761508),  INT64_C( 6769854773662741521),
        -INT64_C( 5773187555479253657),  INT64_C( 2078600584236051024), -INT64_C( 5162369105732032433),  INT64_C( 4859966584814141704) },
      { EASYSIMD_FLOAT32_C(  -868.33), EASYSIMD_FLOAT32_C(-1487017321116467200.00), EASYSIMD_FLOAT32_C(445994935934517248.00), EASYSIMD_FLOAT32_C(6769854720426115072.00),
        EASYSIMD_FLOAT32_C(   599.06), EASYSIMD_FLOAT32_C(   893.90), EASYSIMD_FLOAT32_C(   280.11), EASYSIMD_FLOAT32_C(4859966739095486464.00) } },
    { { EASYSIMD_FLOAT32_C(   -39.73), EASYSIMD_FLOAT32_C(   141.16), EASYSIMD_FLOAT32_C(   -52.22), EASYSIMD_FLOAT32_C(  -222.80),
        EASYSIMD_FLOAT32_C(  -260.44), EASYSIMD_FLOAT32_C(  -379.60), EASYSIMD_FLOAT32_C(   961.75), EASYSIMD_FLOAT32_C(  -851.44) },
      UINT8_C(213),
      {  INT64_C( 2925413414661269406), -INT64_C( 3879140480368316787),  INT64_C( 1716973927641071268),  INT64_C( 7004008655284695987),
        -INT64_C(  753432198003277981), -INT64_C( 8914804129464427980), -INT64_C( 8266904839540576385), -INT64_C( 1754962926366930816) },
      { EASYSIMD_FLOAT32_C(2925413413418958848.00), EASYSIMD_FLOAT32_C(   141.16), EASYSIMD_FLOAT32_C(1716973993312911360.00), EASYSIMD_FLOAT32_C(  -222.80),
        EASYSIMD_FLOAT32_C(-753432165315772416.00), EASYSIMD_FLOAT32_C(  -379.60), EASYSIMD_FLOAT32_C(-8266904672829177856.00), EASYSIMD_FLOAT32_C(-1754962944686292992.00) } },
    { { EASYSIMD_FLOAT32_C(   179.53), EASYSIMD_FLOAT32_C(  -878.51), EASYSIMD_FLOAT32_C(   -24.06), EASYSIMD_FLOAT32_C(  -244.64),
        EASYSIMD_FLOAT32_C(  -550.98), EASYSIMD_FLOAT32_C(   183.16), EASYSIMD_FLOAT32_C(  -650.49), EASYSIMD_FLOAT32_C(  -634.86) },
      UINT8_C( 48),
      {  INT64_C( 9192011285457357009),  INT64_C( 5646639850445127887), -INT64_C( 7616508060268060464), -INT64_C( 7130051650811850491),
        -INT64_C( 9129289357412393665),  INT64_C( 6653147660081276671), -INT64_C( 1930469874257418662), -INT64_C( 7561036519578705496) },
      { EASYSIMD_FLOAT32_C(   179.53), EASYSIMD_FLOAT32_C(  -878.51), EASYSIMD_FLOAT32_C(   -24.06), EASYSIMD_FLOAT32_C(  -244.64),
        EASYSIMD_FLOAT32_C(-9129289575645052928.00), EASYSIMD_FLOAT32_C(6653147608451645440.00), EASYSIMD_FLOAT32_C(  -650.49), EASYSIMD_FLOAT32_C(  -634.86) } },
    { { EASYSIMD_FLOAT32_C(   203.35), EASYSIMD_FLOAT32_C(  -527.65), EASYSIMD_FLOAT32_C(  -201.25), EASYSIMD_FLOAT32_C(   643.78),
        EASYSIMD_FLOAT32_C(   970.15), EASYSIMD_FLOAT32_C(  -750.43), EASYSIMD_FLOAT32_C(   915.16), EASYSIMD_FLOAT32_C(  -837.17) },
      UINT8_C( 91),
      {  INT64_C( 9119581812294091643), -INT64_C( 4193339010415972722),  INT64_C( 1494268552287175173), -INT64_C(  996683690537691109),
        -INT64_C( 1951545910997085676), -INT64_C( 7994215291015313179), -INT64_C(  251603580394198246), -INT64_C( 4980708891528751752) },
      { EASYSIMD_FLOAT32_C(9119581987483418624.00), EASYSIMD_FLOAT32_C(-4193338886488850432.00), EASYSIMD_FLOAT32_C(  -201.25), EASYSIMD_FLOAT32_C(-996683719798423552.00),
        EASYSIMD_FLOAT32_C(-1951545865566945280.00), EASYSIMD_FLOAT32_C(  -750.43), EASYSIMD_FLOAT32_C(-251603582246191104.00), EASYSIMD_FLOAT32_C(  -837.17) } },
    { { EASYSIMD_FLOAT32_C(    40.44), EASYSIMD_FLOAT32_C(   917.06), EASYSIMD_FLOAT32_C(   709.97), EASYSIMD_FLOAT32_C(   183.66),
        EASYSIMD_FLOAT32_C(  -958.93), EASYSIMD_FLOAT32_C(    56.71), EASYSIMD_FLOAT32_C(  -448.08), EASYSIMD_FLOAT32_C(  -382.09) },
      UINT8_C( 89),
      {  INT64_C( 2827548061581307212), -INT64_C( 4435479187003298854),  INT64_C( 5318508450356420815), -INT64_C( 4849342775201092023),
        -INT64_C( 1532317820436572006), -INT64_C( 7513879086450293559), -INT64_C( 8196501547583200842), -INT64_C( 8104056049462917029) },
      { EASYSIMD_FLOAT32_C(2827548082453872640.00), EASYSIMD_FLOAT32_C(   917.06), EASYSIMD_FLOAT32_C(   709.97), EASYSIMD_FLOAT32_C(-4849342707992100864.00),
        EASYSIMD_FLOAT32_C(-1532317887375605760.00), EASYSIMD_FLOAT32_C(    56.71), EASYSIMD_FLOAT32_C(-8196501294035238912.00), EASYSIMD_FLOAT32_C(  -382.09) } },
    { { EASYSIMD_FLOAT32_C(   346.80), EASYSIMD_FLOAT32_C(  -246.60), EASYSIMD_FLOAT32_C(  -546.63), EASYSIMD_FLOAT32_C(   319.73),
        EASYSIMD_FLOAT32_C(    59.89), EASYSIMD_FLOAT32_C(  -880.33), EASYSIMD_FLOAT32_C(   257.41), EASYSIMD_FLOAT32_C(  -633.72) },
      UINT8_C(165),
      {  INT64_C( 2241952580692100369), -INT64_C( 6995794096868289183), -INT64_C( 6581973212335983398),  INT64_C( 3011914295935691590),
        -INT64_C( 7071867213987169859),  INT64_C(  843884181269987267), -INT64_C( 5572597148798662199),  INT64_C(  744965634476594196) },
      { EASYSIMD_FLOAT32_C(2241952587546886144.00), EASYSIMD_FLOAT32_C(  -246.60), EASYSIMD_FLOAT32_C(-6581973472006635520.00), EASYSIMD_FLOAT32_C(   319.73),
        EASYSIMD_FLOAT32_C(    59.89), EASYSIMD_FLOAT32_C(843884176569532416.00), EASYSIMD_FLOAT32_C(   257.41), EASYSIMD_FLOAT32_C(744965650903990272.00) } },
    { { EASYSIMD_FLOAT32_C(   342.40), EASYSIMD_FLOAT32_C(    29.01), EASYSIMD_FLOAT32_C(  -133.25), EASYSIMD_FLOAT32_C(   637.18),
        EASYSIMD_FLOAT32_C(  -759.72), EASYSIMD_FLOAT32_C(  -998.49), EASYSIMD_FLOAT32_C(   -57.10), EASYSIMD_FLOAT32_C(  -750.55) },
      UINT8_C(223),
      {  INT64_C( 8125086531931107556),  INT64_C( 5032928103360087963),  INT64_C( 4750343617557693543), -INT64_C( 6493858884987058335),
        -INT64_C( 6254549845254671715),  INT64_C(  278736765834825511),  INT64_C( 7991070245061161033), -INT64_C( 8905496732273427964) },
      { EASYSIMD_FLOAT32_C(8125086364543746048.00), EASYSIMD_FLOAT32_C(5032928164481859584.00), EASYSIMD_FLOAT32_C(4750343780538777600.00), EASYSIMD_FLOAT32_C(-6493858610156666880.00),
        EASYSIMD_FLOAT32_C(-6254549904371220480.00), EASYSIMD_FLOAT32_C(  -998.49), EASYSIMD_FLOAT32_C(7991070190769012736.00), EASYSIMD_FLOAT32_C(-8905496628194967552.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cvtepi64_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m256 r = easysimd_mm512_mask_cvtepi64_ps(src, k, a);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_cvtepi64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(219),
      {  INT64_C( 4910535922165103097),  INT64_C( 6783850546275701186),  INT64_C( 2583555753969603982),  INT64_C( 6632360113550332517),
        -INT64_C(  521024241453401063), -INT64_C( 9168129332556734908), -INT64_C( 7998261903629645161), -INT64_C( 7113494414168644471) },
      { EASYSIMD_FLOAT32_C(4910536027635974144.00), EASYSIMD_FLOAT32_C(6783850403936075776.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(6632360241616912384.00),
        EASYSIMD_FLOAT32_C(-521024235024416768.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-7998262096326295552.00), EASYSIMD_FLOAT32_C(-7113494435303784448.00) } },
    { UINT8_C( 70),
      { -INT64_C( 4590165410338562795),  INT64_C(   30174407975678609), -INT64_C( 1310338269628839728),  INT64_C( 1769367110866287486),
         INT64_C( 1210461688480926251), -INT64_C( 9036947973834768609), -INT64_C(   85962564050500968),  INT64_C( 8678388435608026964) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(30174408702164992.00), EASYSIMD_FLOAT32_C(-1310338309477629952.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-85962567838597120.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(222),
      { -INT64_C( 3451044490555587441),  INT64_C(  664285646424289361),  INT64_C( 2790693446548527565),  INT64_C( 1261548312665985742),
        -INT64_C( 7164901226343178640), -INT64_C( 2006224891824761896),  INT64_C(  620656968491250063),  INT64_C( 7829838890565046294) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(664285617961566208.00), EASYSIMD_FLOAT32_C(2790693552202448896.00), EASYSIMD_FLOAT32_C(1261548305628790784.00),
        EASYSIMD_FLOAT32_C(-7164901001948823552.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(620656996571414528.00), EASYSIMD_FLOAT32_C(7829838905183567872.00) } },
    { UINT8_C( 11),
      { -INT64_C( 3335210472921748042), -INT64_C( 5821803028500916944),  INT64_C( 6355124741676247830),  INT64_C( 2941906580822870427),
         INT64_C( 5511486875309058541),  INT64_C( 5061525021226329234), -INT64_C( 5516185965667735599),  INT64_C( 8427728622868743771) },
      { EASYSIMD_FLOAT32_C(-3335210468062003200.00), EASYSIMD_FLOAT32_C(-5821803018399514624.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(2941906637591412736.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(119),
      {  INT64_C(   92852231037264123),  INT64_C( 4524560820007459282),  INT64_C( 2073354945223386329), -INT64_C( 8986624605512559740),
        -INT64_C( 8010350407069278920),  INT64_C( 6391545202561582941), -INT64_C( 7590103509181986504), -INT64_C( 7050471637222142436) },
      { EASYSIMD_FLOAT32_C(92852227957325824.00), EASYSIMD_FLOAT32_C(4524560717042417664.00), EASYSIMD_FLOAT32_C(2073354986372202496.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-8010350676917878784.00), EASYSIMD_FLOAT32_C(6391545204901412864.00), EASYSIMD_FLOAT32_C(-7590103588886544384.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(143),
      {  INT64_C( 5711370582613652065),  INT64_C( 8355063973561677671),  INT64_C( 8758968198701089257), -INT64_C( 4989006140103773565),
         INT64_C( 6163144127859921136),  INT64_C( 7238977277911913881), -INT64_C( 7410281663259280508),  INT64_C( 8550525376274499734) },
      { EASYSIMD_FLOAT32_C(5711370819284762624.00), EASYSIMD_FLOAT32_C(8355064064900071424.00), EASYSIMD_FLOAT32_C(8758968012096143360.00), EASYSIMD_FLOAT32_C(-4989005973487091712.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(8550525346481438720.00) } },
    { UINT8_C(162),
      { -INT64_C( 8447093509519790519),  INT64_C( 4286416668814854645),  INT64_C( 6717659067949591359),  INT64_C( 2012081888967207097),
         INT64_C( 6252698744413191175),  INT64_C( 8309363637543160789),  INT64_C( 5019067773684720118), -INT64_C( 1557263730523207049) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(4286416668948692992.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(8309363413847375872.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-1557263744625541120.00) } },
    { UINT8_C(249),
      { -INT64_C( 5220602237782710614), -INT64_C( 7800130162021129729),  INT64_C( 4339893009377069684), -INT64_C( 7968544605821959179),
         INT64_C( 8490926276239476017), -INT64_C( 7783628495615180682),  INT64_C( 8428646680430582282),  INT64_C( 6092332511405611028) },
      { EASYSIMD_FLOAT32_C(-5220602154959503360.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-7968544496050765824.00),
        EASYSIMD_FLOAT32_C(8490926318697840640.00), EASYSIMD_FLOAT32_C(-7783628630491398144.00), EASYSIMD_FLOAT32_C(8428646681565724672.00), EASYSIMD_FLOAT32_C(6092332356855660544.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_cvtepi64_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_cvtepi64_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m256 r = easysimd_mm512_maskz_cvtepi64_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { UINT64_C(18064432866853605913), UINT64_C( 1274148936533025876), UINT64_C(12450120256744939630), UINT64_C( 2825506353499720202),
        UINT64_C(18338511080483580297), UINT64_C(11197545359226076830), UINT64_C( 5918104055528133133), UINT64_C(15516438169313613343) },
      { EASYSIMD_FLOAT32_C(18064432885615558656.00), EASYSIMD_FLOAT32_C(1274148983761010688.00), EASYSIMD_FLOAT32_C(12450119806005280768.00), EASYSIMD_FLOAT32_C(2825506289361092608.00),
        EASYSIMD_FLOAT32_C(18338511447606165504.00), EASYSIMD_FLOAT32_C(11197545164526583808.00), EASYSIMD_FLOAT32_C(5918104294074089472.00), EASYSIMD_FLOAT32_C(15516437833546989568.00) } },
    { { UINT64_C( 3893885898405453408), UINT64_C(13916949357175375706), UINT64_C( 2457935170409770881), UINT64_C( 3716496446310079597),
        UINT64_C( 2707022690249469891), UINT64_C(10215234926491212878), UINT64_C(13664891384465127805), UINT64_C( 6617209077191275471) },
      { EASYSIMD_FLOAT32_C(3893885795152429056.00), EASYSIMD_FLOAT32_C(13916949681318920192.00), EASYSIMD_FLOAT32_C(2457935154149064704.00), EASYSIMD_FLOAT32_C(3716496361563095040.00),
        EASYSIMD_FLOAT32_C(2707022641474043904.00), EASYSIMD_FLOAT32_C(10215234879201738752.00), EASYSIMD_FLOAT32_C(13664891038697783296.00), EASYSIMD_FLOAT32_C(6617208971386159104.00) } },
    { { UINT64_C(12100125849277567739), UINT64_C(16572239920327399780), UINT64_C(10696622809714938725), UINT64_C(17577843978591988733),
        UINT64_C( 5989408101885454110), UINT64_C(10992371019083401023), UINT64_C(17900893707277562386), UINT64_C(14741829343295480747) },
      { EASYSIMD_FLOAT32_C(12100125463674880000.00), EASYSIMD_FLOAT32_C(16572239476072382464.00), EASYSIMD_FLOAT32_C(10696623060074627072.00), EASYSIMD_FLOAT32_C(17577844014743289856.00),
        EASYSIMD_FLOAT32_C(5989408172891176960.00), EASYSIMD_FLOAT32_C(10992370797225443328.00), EASYSIMD_FLOAT32_C(17900893725123411968.00), EASYSIMD_FLOAT32_C(14741829692755542016.00) } },
    { { UINT64_C( 3848715066128158502), UINT64_C(18371142669337111592), UINT64_C(12343877297405050970), UINT64_C( 9395040280141225190),
        UINT64_C( 3341109099518010418), UINT64_C(18237464830515231278), UINT64_C( 1759115468290125758), UINT64_C(14725659836044588670) },
      { EASYSIMD_FLOAT32_C(3848715108704321536.00), EASYSIMD_FLOAT32_C(18371142753695301632.00), EASYSIMD_FLOAT32_C(12343877295948169216.00), EASYSIMD_FLOAT32_C(9395039886811070464.00),
        EASYSIMD_FLOAT32_C(3341109073067114496.00), EASYSIMD_FLOAT32_C(18237465229501923328.00), EASYSIMD_FLOAT32_C(1759115525226496000.00), EASYSIMD_FLOAT32_C(14725660274757468160.00) } },
    { { UINT64_C( 2586846251294454962), UINT64_C(12406668129487122573), UINT64_C(11641755711056806213), UINT64_C(18153965332081621897),
        UINT64_C(14115575592543052747), UINT64_C(13665082873293775370), UINT64_C( 7667030790071652001), UINT64_C( 9530316896680665598) },
      { EASYSIMD_FLOAT32_C(2586846295436034048.00), EASYSIMD_FLOAT32_C(12406668205987201024.00), EASYSIMD_FLOAT32_C(11641755558729482240.00), EASYSIMD_FLOAT32_C(18153965017953730560.00),
        EASYSIMD_FLOAT32_C(14115575357365026816.00), EASYSIMD_FLOAT32_C(13665082353721016320.00), EASYSIMD_FLOAT32_C(7667030919923892224.00), EASYSIMD_FLOAT32_C(9530317200422862848.00) } },
    { { UINT64_C(11049909956920079712), UINT64_C( 1045175329797262412), UINT64_C( 8542827631833429315), UINT64_C(17894772300767435802),
        UINT64_C( 1377793380147174902), UINT64_C( 2859320552401121075), UINT64_C(15275041951985664013), UINT64_C(12720133175253580277) },
      { EASYSIMD_FLOAT32_C(11049910439730216960.00), EASYSIMD_FLOAT32_C(1045175343679275008.00), EASYSIMD_FLOAT32_C(8542827665575378944.00), EASYSIMD_FLOAT32_C(17894772743891582976.00),
        EASYSIMD_FLOAT32_C(1377793347841687552.00), EASYSIMD_FLOAT32_C(2859320669961715712.00), EASYSIMD_FLOAT32_C(15275042254692024320.00), EASYSIMD_FLOAT32_C(12720133473507999744.00) } },
    { { UINT64_C( 3491507954546709914), UINT64_C(12685934967789734658), UINT64_C( 2849143803862145278), UINT64_C(14435227777147154666),
        UINT64_C(16032144917533460816), UINT64_C(14501258522366594538), UINT64_C(10340840494933832489), UINT64_C(14065454611220870972) },
      { EASYSIMD_FLOAT32_C(3491507970095710208.00), EASYSIMD_FLOAT32_C(12685935363349282816.00), EASYSIMD_FLOAT32_C(2849143865212928000.00), EASYSIMD_FLOAT32_C(14435227576815321088.00),
        EASYSIMD_FLOAT32_C(16032145070857650176.00), EASYSIMD_FLOAT32_C(14501258747621408768.00), EASYSIMD_FLOAT32_C(10340840888535613440.00), EASYSIMD_FLOAT32_C(14065454119812857856.00) } },
    { { UINT64_C(16558766043104828433), UINT64_C(10119219350477115866), UINT64_C( 3987772525188214564), UINT64_C( 2654259634275191712),
        UINT64_C( 5481126427951818032), UINT64_C( 6295575104809173616), UINT64_C( 5656934197405631823), UINT64_C( 2754293077662310914) },
      { EASYSIMD_FLOAT32_C(16558766060585615360.00), EASYSIMD_FLOAT32_C(10119218926794571776.00), EASYSIMD_FLOAT32_C(3987772543292407808.00), EASYSIMD_FLOAT32_C(2654259552358236160.00),
        EASYSIMD_FLOAT32_C(5481126487358701568.00), EASYSIMD_FLOAT32_C(6295574882226798592.00), EASYSIMD_FLOAT32_C(5656933998593572864.00), EASYSIMD_FLOAT32_C(2754293120253296640.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvtepu64_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m256i r = easysimd_mm512_cvtepu64_ps(a);

    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepu64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const uint64_t a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -99.57), EASYSIMD_FLOAT32_C(   857.00), EASYSIMD_FLOAT32_C(  -833.24), EASYSIMD_FLOAT32_C(   906.62),
        EASYSIMD_FLOAT32_C(   975.60), EASYSIMD_FLOAT32_C(  -758.37), EASYSIMD_FLOAT32_C(  -408.14), EASYSIMD_FLOAT32_C(  -661.07) },
      UINT8_C(225),
      { UINT64_C( 5429771612975717258), UINT64_C(18197588703828665137), UINT64_C( 2279026362495784878), UINT64_C(  806752918650611276),
        UINT64_C(13462367085121576196), UINT64_C(12813340190174556499), UINT64_C( 2845487556845576696), UINT64_C( 6388262976322349594) },
      { EASYSIMD_FLOAT32_C(5429771597760167936.00), EASYSIMD_FLOAT32_C(   857.00), EASYSIMD_FLOAT32_C(  -833.24), EASYSIMD_FLOAT32_C(   906.62),
        EASYSIMD_FLOAT32_C(   975.60), EASYSIMD_FLOAT32_C(12813340173706199040.00), EASYSIMD_FLOAT32_C(2845487439294758912.00), EASYSIMD_FLOAT32_C(6388263162692501504.00) } },
    { { EASYSIMD_FLOAT32_C(  -553.42), EASYSIMD_FLOAT32_C(     2.07), EASYSIMD_FLOAT32_C(  -809.07), EASYSIMD_FLOAT32_C(  -752.92),
        EASYSIMD_FLOAT32_C(   379.26), EASYSIMD_FLOAT32_C(   526.19), EASYSIMD_FLOAT32_C(   150.56), EASYSIMD_FLOAT32_C(   918.07) },
      UINT8_C(228),
      { UINT64_C( 5169455239587648675), UINT64_C(14157428926202387500), UINT64_C(17885888399728448963), UINT64_C( 4173046390150433105),
        UINT64_C(10765498019441725688), UINT64_C(10751621535550965975), UINT64_C( 1024353124242181507), UINT64_C(17839599247609257156) },
      { EASYSIMD_FLOAT32_C(  -553.42), EASYSIMD_FLOAT32_C(     2.07), EASYSIMD_FLOAT32_C(17885888689939152896.00), EASYSIMD_FLOAT32_C(  -752.92),
        EASYSIMD_FLOAT32_C(   379.26), EASYSIMD_FLOAT32_C(10751621731207610368.00), EASYSIMD_FLOAT32_C(1024353136069836800.00), EASYSIMD_FLOAT32_C(17839599250409783296.00) } },
    { { EASYSIMD_FLOAT32_C(   242.69), EASYSIMD_FLOAT32_C(  -841.78), EASYSIMD_FLOAT32_C(   333.62), EASYSIMD_FLOAT32_C(   472.30),
        EASYSIMD_FLOAT32_C(  -623.76), EASYSIMD_FLOAT32_C(   370.42), EASYSIMD_FLOAT32_C(  -458.61), EASYSIMD_FLOAT32_C(    96.41) },
      UINT8_C(  4),
      { UINT64_C( 1634926276048819391), UINT64_C( 9398979603065007984), UINT64_C(17059205874326882805), UINT64_C( 9011803726158558095),
        UINT64_C(14181934542233462768), UINT64_C( 9003504373555242057), UINT64_C( 9074520071139587240), UINT64_C( 8312123754390497055) },
      { EASYSIMD_FLOAT32_C(   242.69), EASYSIMD_FLOAT32_C(  -841.78), EASYSIMD_FLOAT32_C(17059205479432978432.00), EASYSIMD_FLOAT32_C(   472.30),
        EASYSIMD_FLOAT32_C(  -623.76), EASYSIMD_FLOAT32_C(   370.42), EASYSIMD_FLOAT32_C(  -458.61), EASYSIMD_FLOAT32_C(    96.41) } },
    { { EASYSIMD_FLOAT32_C(  -934.74), EASYSIMD_FLOAT32_C(  -709.77), EASYSIMD_FLOAT32_C(    54.59), EASYSIMD_FLOAT32_C(   210.84),
        EASYSIMD_FLOAT32_C(  -813.79), EASYSIMD_FLOAT32_C(   640.07), EASYSIMD_FLOAT32_C(   154.48), EASYSIMD_FLOAT32_C(  -232.85) },
      UINT8_C(160),
      { UINT64_C( 8975008891018876422), UINT64_C(12131544559208010000), UINT64_C(10846812577584593007), UINT64_C(17899446706751429322),
        UINT64_C( 4771118716578270897), UINT64_C(17808598002462269692), UINT64_C(13274789731868036881), UINT64_C( 8776080574084230625) },
      { EASYSIMD_FLOAT32_C(  -934.74), EASYSIMD_FLOAT32_C(  -709.77), EASYSIMD_FLOAT32_C(    54.59), EASYSIMD_FLOAT32_C(   210.84),
        EASYSIMD_FLOAT32_C(  -813.79), EASYSIMD_FLOAT32_C(17808598520064638976.00), EASYSIMD_FLOAT32_C(   154.48), EASYSIMD_FLOAT32_C(8776080811070849024.00) } },
    { { EASYSIMD_FLOAT32_C(  -189.37), EASYSIMD_FLOAT32_C(  -670.42), EASYSIMD_FLOAT32_C(  -466.70), EASYSIMD_FLOAT32_C(   517.40),
        EASYSIMD_FLOAT32_C(  -127.29), EASYSIMD_FLOAT32_C(  -107.85), EASYSIMD_FLOAT32_C(  -944.07), EASYSIMD_FLOAT32_C(  -710.11) },
      UINT8_C( 10),
      { UINT64_C(18397355867882294118), UINT64_C( 6480579997457052552), UINT64_C( 3782913237782217252), UINT64_C( 8449691944950900430),
        UINT64_C(16614066404092859874), UINT64_C( 5151805863370194680), UINT64_C(  677740321120100518), UINT64_C(16680071901523128055) },
      { EASYSIMD_FLOAT32_C(  -189.37), EASYSIMD_FLOAT32_C(6480579808228016128.00), EASYSIMD_FLOAT32_C(  -466.70), EASYSIMD_FLOAT32_C(8449691883877171200.00),
        EASYSIMD_FLOAT32_C(  -127.29), EASYSIMD_FLOAT32_C(  -107.85), EASYSIMD_FLOAT32_C(  -944.07), EASYSIMD_FLOAT32_C(  -710.11) } },
    { { EASYSIMD_FLOAT32_C(    67.14), EASYSIMD_FLOAT32_C(  -978.47), EASYSIMD_FLOAT32_C(    77.98), EASYSIMD_FLOAT32_C(  -147.60),
        EASYSIMD_FLOAT32_C(   504.76), EASYSIMD_FLOAT32_C(  -792.55), EASYSIMD_FLOAT32_C(   884.16), EASYSIMD_FLOAT32_C(  -356.47) },
      UINT8_C(137),
      { UINT64_C( 8078601007777206930), UINT64_C( 7210181322129231686), UINT64_C(13823324687753090551), UINT64_C(13717156516475473609),
        UINT64_C( 4878614444545214835), UINT64_C(  845385990642818976), UINT64_C( 4525657432906998911), UINT64_C( 6257311753413247723) },
      { EASYSIMD_FLOAT32_C(8078601211944632320.00), EASYSIMD_FLOAT32_C(  -978.47), EASYSIMD_FLOAT32_C(    77.98), EASYSIMD_FLOAT32_C(13717156323924115456.00),
        EASYSIMD_FLOAT32_C(   504.76), EASYSIMD_FLOAT32_C(  -792.55), EASYSIMD_FLOAT32_C(   884.16), EASYSIMD_FLOAT32_C(6257311877580193792.00) } },
    { { EASYSIMD_FLOAT32_C(  -845.54), EASYSIMD_FLOAT32_C(   105.65), EASYSIMD_FLOAT32_C(  -728.24), EASYSIMD_FLOAT32_C(  -519.64),
        EASYSIMD_FLOAT32_C(  -666.21), EASYSIMD_FLOAT32_C(   452.88), EASYSIMD_FLOAT32_C(   -81.50), EASYSIMD_FLOAT32_C(   734.37) },
      UINT8_C(173),
      { UINT64_C( 3746657402175509269), UINT64_C(11429826735582481749), UINT64_C(10872209068714472953), UINT64_C( 2841190903277901938),
        UINT64_C( 5946340290932911939), UINT64_C(  876691282314145201), UINT64_C(17760603209240783960), UINT64_C(11594274651260743086) },
      { EASYSIMD_FLOAT32_C(3746657339902525440.00), EASYSIMD_FLOAT32_C(   105.65), EASYSIMD_FLOAT32_C(10872209569472315392.00), EASYSIMD_FLOAT32_C(2841190822731317248.00),
        EASYSIMD_FLOAT32_C(  -666.21), EASYSIMD_FLOAT32_C(876691267080159232.00), EASYSIMD_FLOAT32_C(   -81.50), EASYSIMD_FLOAT32_C(11594274248595603456.00) } },
    { { EASYSIMD_FLOAT32_C(  -517.71), EASYSIMD_FLOAT32_C(  -774.53), EASYSIMD_FLOAT32_C(  -261.45), EASYSIMD_FLOAT32_C(  -241.51),
        EASYSIMD_FLOAT32_C(  -382.22), EASYSIMD_FLOAT32_C(  -172.49), EASYSIMD_FLOAT32_C(  -642.59), EASYSIMD_FLOAT32_C(  -511.23) },
      UINT8_C(164),
      { UINT64_C( 5017507622513307558), UINT64_C( 7727825110686073913), UINT64_C( 8425225442046836597), UINT64_C( 9545248425849994331),
        UINT64_C( 3302949373403279351), UINT64_C(15302664597882642419), UINT64_C(16885500848097797627), UINT64_C( 8860890006038641557) },
      { EASYSIMD_FLOAT32_C(  -517.71), EASYSIMD_FLOAT32_C(  -774.53), EASYSIMD_FLOAT32_C(8425225551135899648.00), EASYSIMD_FLOAT32_C(  -241.51),
        EASYSIMD_FLOAT32_C(  -382.22), EASYSIMD_FLOAT32_C(15302664185805012992.00), EASYSIMD_FLOAT32_C(  -642.59), EASYSIMD_FLOAT32_C(8860889991211909120.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cvtepu64_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m256 r = easysimd_mm512_mask_cvtepu64_ps(src, k, a);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_cvtepu64_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C( 21),
      { UINT64_C(  512847711042236884), UINT64_C(11117621115607221824), UINT64_C( 1636651639376512272), UINT64_C( 1285932990022136179),
        UINT64_C(11767017847295468543), UINT64_C( 4357562685704073822), UINT64_C(10612333537436537650), UINT64_C(  906681535574736966) },
      { EASYSIMD_FLOAT32_C(512847716804460544.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(1636651645246898176.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(11767017420923863040.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(145),
      { UINT64_C( 1262864550583423567), UINT64_C( 8276061266753971579), UINT64_C(15002376912726264408), UINT64_C(11258130787836278652),
        UINT64_C( 5038485417763287356), UINT64_C(18014899485472008624), UINT64_C( 3959257508675556561), UINT64_C(17851176812032413906) },
      { EASYSIMD_FLOAT32_C(1262864558486192128.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(5038485646004453376.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(17851177107850264576.00) } },
    { UINT8_C( 79),
      { UINT64_C(16680312392929366656), UINT64_C(11665875601487170380), UINT64_C(14580845773787432812), UINT64_C(13105612584676325747),
        UINT64_C(11958742944489536079), UINT64_C( 8121853817902432008), UINT64_C( 1615473555052892582), UINT64_C(13669699675469902025) },
      { EASYSIMD_FLOAT32_C(16680312672989741056.00), EASYSIMD_FLOAT32_C(11665875545308004352.00), EASYSIMD_FLOAT32_C(14580845797286346752.00), EASYSIMD_FLOAT32_C(13105612354601615360.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(1615473539467444224.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(129),
      { UINT64_C( 4628153130051881781), UINT64_C( 1765327520443682873), UINT64_C( 8918765031936171947), UINT64_C(12430398381332415352),
        UINT64_C( 1545044576389978992), UINT64_C( 1479270202719712617), UINT64_C(10057052621334594188), UINT64_C(10429224164627544728) },
      { EASYSIMD_FLOAT32_C(4628153404076589056.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(10429224031222759424.00) } },
    { UINT8_C(103),
      { UINT64_C(16653282496888422810), UINT64_C( 6214514571576642629), UINT64_C( 6980849791830414451), UINT64_C( 5112447231466663808),
        UINT64_C(15341818404805359312), UINT64_C( 1228291505022974370), UINT64_C(16887998784382530840), UINT64_C(13698377686793510440) },
      { EASYSIMD_FLOAT32_C(16653282279132495872.00), EASYSIMD_FLOAT32_C(6214514486980640768.00), EASYSIMD_FLOAT32_C(6980849902284701696.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(1228291514862403584.00), EASYSIMD_FLOAT32_C(16887998324848721920.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(160),
      { UINT64_C(16414216196740530195), UINT64_C(  100647697962336648), UINT64_C(11550701436621513707), UINT64_C( 5680529218376376523),
        UINT64_C(10275317720932614924), UINT64_C(10903848756652481220), UINT64_C(15998202130707758499), UINT64_C(16911554117595534120) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(10903849116073197568.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(16911554261962194944.00) } },
    { UINT8_C(236),
      { UINT64_C( 6066208211498868776), UINT64_C(10322136901267587163), UINT64_C( 4432810501937996884), UINT64_C(13936677773878486932),
        UINT64_C( 3978187945280297894), UINT64_C( 2467147750262847196), UINT64_C(17698305827401556150), UINT64_C( 7297198907542649066) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(4432810594873114624.00), EASYSIMD_FLOAT32_C(13936678218456104960.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(2467147687200292864.00), EASYSIMD_FLOAT32_C(17698305409170800640.00), EASYSIMD_FLOAT32_C(7297198639049342976.00) } },
    { UINT8_C( 54),
      { UINT64_C(12729505138521294192), UINT64_C( 8422753385903511067), UINT64_C(10956636532484948768), UINT64_C( 4994418130904649789),
        UINT64_C(14207936165835792641), UINT64_C( 7861267679031765776), UINT64_C(17695097828685180184), UINT64_C(11909589599083423790) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(8422753299240845312.00), EASYSIMD_FLOAT32_C(10956636669322723328.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(14207936533121466368.00), EASYSIMD_FLOAT32_C(7861267895307730944.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_cvtepu64_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_cvtepu64_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m256 r = easysimd_mm512_maskz_cvtepu64_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepi64_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT64_C( 5499118837676661407),  INT64_C(  802518787675509433), -INT64_C( 1895252414684915744), -INT64_C(  815293002885254977),
        -INT64_C( 7667159720415325990), -INT64_C( 8054238576114298471), -INT64_C( 5796790294263940432),  INT64_C( 1126953469552277081) },
      { -INT32_C(  2124391777), -INT32_C(   520498503),  INT32_C(  1728124896),  INT32_C(  1596567743),  INT32_C(  1282805978),  INT32_C(   779456921), -INT32_C(  1460192592), -INT32_C(  1357990311) } },
    { {  INT64_C( 8736688473384884686), -INT64_C(  232532088305765499), -INT64_C( 4631053477468652517),  INT64_C( 6137300763706051047),
         INT64_C(   37222295617373995),  INT64_C( 2898622769958038769), -INT64_C(  730446195659055690), -INT64_C( 8604904046600303944) },
      { -INT32_C(  1453647410),  INT32_C(   900182917),  INT32_C(   195345435),  INT32_C(   477022695),  INT32_C(   201295659),  INT32_C(  1026895089),  INT32_C(   473226678),  INT32_C(  1074875064) } },
    { { -INT64_C( 4377471546498509627),  INT64_C( 2700252601848460612),  INT64_C( 6131966023613721878),  INT64_C( 3264311934393330280),
         INT64_C( 4347290823422763199), -INT64_C( 6650119495152217054),  INT64_C( 9198285675416778932),  INT64_C( 2638987771557198602) },
      {  INT32_C(  1083413701),  INT32_C(  1358984516),  INT32_C(   608349462), -INT32_C(   980080024), -INT32_C(  1855070017), -INT32_C(  1853008862),  INT32_C(  1372059828),  INT32_C(  1917074186) } },
    { { -INT64_C( 2619395441305514895), -INT64_C( 1185252386685569955),  INT64_C( 4589702607123010587), -INT64_C(  600296487079774943),
        -INT64_C( 3967099070008172195), -INT64_C( 7104920433032274116), -INT64_C( 5834271970793904236),  INT64_C( 7562635636565784949) },
      {  INT32_C(  1790250097), -INT32_C(   362007459),  INT32_C(   843076635), -INT32_C(  2018380511),  INT32_C(   392257885),  INT32_C(  2008178492), -INT32_C(   909138028), -INT32_C(    80299659) } },
    { { -INT64_C( 7460932564807625662), -INT64_C(  220546195610720304), -INT64_C( 5379042602290659555), -INT64_C( 5478166664036905082),
        -INT64_C( 4977086815038048008), -INT64_C( 8043151361953694211),  INT64_C( 8768953720801732028), -INT64_C(   51788068926365149) },
      { -INT32_C(  1384164286),  INT32_C(  1376724944), -INT32_C(  1429881059), -INT32_C(  1850634362),  INT32_C(  1432385784),  INT32_C(  1678573053), -INT32_C(    63298116), -INT32_C(  1811258845) } },
    { {  INT64_C( 8871600327071410299), -INT64_C( 2071303814381622466),  INT64_C( 3733925065463397477),  INT64_C( 6483231688452791283),
        -INT64_C( 3624873721694696008),  INT64_C( 3896388632289644807),  INT64_C( 8538648431752573244),  INT64_C( 1184843999768233786) },
      {  INT32_C(  1683269755), -INT32_C(  1310774466),  INT32_C(  1491041381), -INT32_C(    87499789),  INT32_C(   918375864),  INT32_C(   780046599),  INT32_C(  1284436284),  INT32_C(   410011450) } },
    { {  INT64_C( 1302094494560563126), -INT64_C( 8318058006688264055), -INT64_C( 5099104476076695740), -INT64_C( 3817289872535606113),
        -INT64_C( 3562592197892092558),  INT64_C( 7389765019772702641),  INT64_C( 6633637038055515607), -INT64_C( 7640821710201888747) },
      {  INT32_C(  1145450422),  INT32_C(  1514180745), -INT32_C(   958914748), -INT32_C(   154030945),  INT32_C(  2098220402),  INT32_C(    19451825),  INT32_C(  1445750231),  INT32_C(   726851605) } },
    { { -INT64_C( 2757996099673127258),  INT64_C( 8212710404306755953), -INT64_C( 6564042024150424114),  INT64_C( 4066686223065758134),
        -INT64_C( 7724275738892598580), -INT64_C( 2321570349060740931),  INT64_C( 1979421534026633030), -INT64_C( 2786209687447189622) },
      { -INT32_C(   351140186), -INT32_C(  1814306447), -INT32_C(  1949882930), -INT32_C(   623953482), -INT32_C(   198999348),  INT32_C(  1462216893), -INT32_C(   731214010),  INT32_C(   502613898) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m256i r = easysimd_mm512_cvtepi64_epi32(a);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m256i r = easysimd_mm512_cvtepi64_epi32(a);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu32_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { UINT32_C(1870160146), UINT32_C( 989482389), UINT32_C( 886655870), UINT32_C(1378107877), UINT32_C(3861032350), UINT32_C(3088204513), UINT32_C(3493347028), UINT32_C(4188785639),
        UINT32_C(3261670189), UINT32_C(4026360688), UINT32_C(2485376687), UINT32_C(3068544791), UINT32_C(3600550132), UINT32_C( 512667466), UINT32_C(3572418541), UINT32_C(3217987986) },
      { EASYSIMD_FLOAT32_C(1870160128.00), EASYSIMD_FLOAT32_C(989482368.00), EASYSIMD_FLOAT32_C(886655872.00), EASYSIMD_FLOAT32_C(1378107904.00),
        EASYSIMD_FLOAT32_C(3861032448.00), EASYSIMD_FLOAT32_C(3088204544.00), EASYSIMD_FLOAT32_C(3493347072.00), EASYSIMD_FLOAT32_C(4188785664.00),
        EASYSIMD_FLOAT32_C(3261670144.00), EASYSIMD_FLOAT32_C(4026360576.00), EASYSIMD_FLOAT32_C(2485376768.00), EASYSIMD_FLOAT32_C(3068544768.00),
        EASYSIMD_FLOAT32_C(3600550144.00), EASYSIMD_FLOAT32_C(512667456.00), EASYSIMD_FLOAT32_C(3572418560.00), EASYSIMD_FLOAT32_C(3217988096.00) } },
    { { UINT32_C( 746731452), UINT32_C(1226538906), UINT32_C(1843215957), UINT32_C(2049164166), UINT32_C( 357613515), UINT32_C(1530126190), UINT32_C( 942678438), UINT32_C(1996029626),
        UINT32_C(3483597365), UINT32_C(1310310137), UINT32_C(2210133757), UINT32_C(2248007354), UINT32_C( 211504798), UINT32_C(3546861101), UINT32_C(2852886768), UINT32_C(3407938454) },
      { EASYSIMD_FLOAT32_C(746731456.00), EASYSIMD_FLOAT32_C(1226538880.00), EASYSIMD_FLOAT32_C(1843216000.00), EASYSIMD_FLOAT32_C(2049164160.00),
        EASYSIMD_FLOAT32_C(357613504.00), EASYSIMD_FLOAT32_C(1530126208.00), EASYSIMD_FLOAT32_C(942678464.00), EASYSIMD_FLOAT32_C(1996029568.00),
        EASYSIMD_FLOAT32_C(3483597312.00), EASYSIMD_FLOAT32_C(1310310144.00), EASYSIMD_FLOAT32_C(2210133760.00), EASYSIMD_FLOAT32_C(2248007424.00),
        EASYSIMD_FLOAT32_C(211504800.00), EASYSIMD_FLOAT32_C(3546861056.00), EASYSIMD_FLOAT32_C(2852886784.00), EASYSIMD_FLOAT32_C(3407938560.00) } },
    { { UINT32_C(1989854333), UINT32_C(2143662978), UINT32_C(1677885610), UINT32_C(4259905631), UINT32_C(2064221262), UINT32_C(1129214291), UINT32_C(2683132169), UINT32_C(3664383581),
        UINT32_C(1431373266), UINT32_C(1658066616), UINT32_C(4123449238), UINT32_C( 636661975), UINT32_C(2275474484), UINT32_C(2009788013), UINT32_C(2752952391), UINT32_C(2575270342) },
      { EASYSIMD_FLOAT32_C(1989854336.00), EASYSIMD_FLOAT32_C(2143662976.00), EASYSIMD_FLOAT32_C(1677885568.00), EASYSIMD_FLOAT32_C(4259905536.00),
        EASYSIMD_FLOAT32_C(2064221312.00), EASYSIMD_FLOAT32_C(1129214336.00), EASYSIMD_FLOAT32_C(2683132160.00), EASYSIMD_FLOAT32_C(3664383488.00),
        EASYSIMD_FLOAT32_C(1431373312.00), EASYSIMD_FLOAT32_C(1658066560.00), EASYSIMD_FLOAT32_C(4123449344.00), EASYSIMD_FLOAT32_C(636661952.00),
        EASYSIMD_FLOAT32_C(2275474432.00), EASYSIMD_FLOAT32_C(2009788032.00), EASYSIMD_FLOAT32_C(2752952320.00), EASYSIMD_FLOAT32_C(2575270400.00) } },
    { { UINT32_C(1055838342), UINT32_C(2090975974), UINT32_C(1886545817), UINT32_C(1284858903), UINT32_C(3469948256), UINT32_C(1799724579), UINT32_C( 470768470), UINT32_C(1656065756),
        UINT32_C(1151443806), UINT32_C(4290855526), UINT32_C(3245355945), UINT32_C(4161602967), UINT32_C(1590091835), UINT32_C(3569945470), UINT32_C(1139923302), UINT32_C(3332744807) },
      { EASYSIMD_FLOAT32_C(1055838336.00), EASYSIMD_FLOAT32_C(2090976000.00), EASYSIMD_FLOAT32_C(1886545792.00), EASYSIMD_FLOAT32_C(1284858880.00),
        EASYSIMD_FLOAT32_C(3469948160.00), EASYSIMD_FLOAT32_C(1799724544.00), EASYSIMD_FLOAT32_C(470768480.00), EASYSIMD_FLOAT32_C(1656065792.00),
        EASYSIMD_FLOAT32_C(1151443840.00), EASYSIMD_FLOAT32_C(4290855424.00), EASYSIMD_FLOAT32_C(3245356032.00), EASYSIMD_FLOAT32_C(4161603072.00),
        EASYSIMD_FLOAT32_C(1590091776.00), EASYSIMD_FLOAT32_C(3569945344.00), EASYSIMD_FLOAT32_C(1139923328.00), EASYSIMD_FLOAT32_C(3332744704.00) } },
    { { UINT32_C(2953463370), UINT32_C( 850381704), UINT32_C(2532515838), UINT32_C(1619918885), UINT32_C(1606309088), UINT32_C(3308488799), UINT32_C(3355976801), UINT32_C( 361672395),
        UINT32_C(2110102004), UINT32_C(1672442980), UINT32_C(3120145044), UINT32_C(2182711202), UINT32_C( 987879387), UINT32_C(3237942623), UINT32_C(  76089401), UINT32_C(2853771190) },
      { EASYSIMD_FLOAT32_C(2953463296.00), EASYSIMD_FLOAT32_C(850381696.00), EASYSIMD_FLOAT32_C(2532515840.00), EASYSIMD_FLOAT32_C(1619918848.00),
        EASYSIMD_FLOAT32_C(1606309120.00), EASYSIMD_FLOAT32_C(3308488704.00), EASYSIMD_FLOAT32_C(3355976704.00), EASYSIMD_FLOAT32_C(361672384.00),
        EASYSIMD_FLOAT32_C(2110102016.00), EASYSIMD_FLOAT32_C(1672443008.00), EASYSIMD_FLOAT32_C(3120145152.00), EASYSIMD_FLOAT32_C(2182711296.00),
        EASYSIMD_FLOAT32_C(987879360.00), EASYSIMD_FLOAT32_C(3237942528.00), EASYSIMD_FLOAT32_C(76089400.00), EASYSIMD_FLOAT32_C(2853771264.00) } },
    { { UINT32_C( 354934448), UINT32_C(3883456083), UINT32_C( 446722424), UINT32_C(3550329336), UINT32_C(4027416208), UINT32_C(3450866835), UINT32_C(3402709268), UINT32_C(  24505169),
        UINT32_C( 471243977), UINT32_C(3942878835), UINT32_C(4144407551), UINT32_C(3989480284), UINT32_C(3051214625), UINT32_C(4169305572), UINT32_C( 415454151), UINT32_C( 135870526) },
      { EASYSIMD_FLOAT32_C(354934464.00), EASYSIMD_FLOAT32_C(3883456000.00), EASYSIMD_FLOAT32_C(446722432.00), EASYSIMD_FLOAT32_C(3550329344.00),
        EASYSIMD_FLOAT32_C(4027416320.00), EASYSIMD_FLOAT32_C(3450866944.00), EASYSIMD_FLOAT32_C(3402709248.00), EASYSIMD_FLOAT32_C(24505168.00),
        EASYSIMD_FLOAT32_C(471243968.00), EASYSIMD_FLOAT32_C(3942878720.00), EASYSIMD_FLOAT32_C(4144407552.00), EASYSIMD_FLOAT32_C(3989480192.00),
        EASYSIMD_FLOAT32_C(3051214592.00), EASYSIMD_FLOAT32_C(4169305600.00), EASYSIMD_FLOAT32_C(415454144.00), EASYSIMD_FLOAT32_C(135870528.00) } },
    { { UINT32_C(1193554132), UINT32_C(3191023806), UINT32_C( 682965451), UINT32_C(4246044892), UINT32_C(1001583191), UINT32_C(1177826431), UINT32_C(3328112520), UINT32_C(  63862831),
        UINT32_C(1716253608), UINT32_C(3861151259), UINT32_C(2467224247), UINT32_C(2979013466), UINT32_C(2515354389), UINT32_C(4292551031), UINT32_C(1204173336), UINT32_C(1514837170) },
      { EASYSIMD_FLOAT32_C(1193554176.00), EASYSIMD_FLOAT32_C(3191023872.00), EASYSIMD_FLOAT32_C(682965440.00), EASYSIMD_FLOAT32_C(4246044928.00),
        EASYSIMD_FLOAT32_C(1001583168.00), EASYSIMD_FLOAT32_C(1177826432.00), EASYSIMD_FLOAT32_C(3328112640.00), EASYSIMD_FLOAT32_C(63862832.00),
        EASYSIMD_FLOAT32_C(1716253568.00), EASYSIMD_FLOAT32_C(3861151232.00), EASYSIMD_FLOAT32_C(2467224320.00), EASYSIMD_FLOAT32_C(2979013376.00),
        EASYSIMD_FLOAT32_C(2515354368.00), EASYSIMD_FLOAT32_C(4292550912.00), EASYSIMD_FLOAT32_C(1204173312.00), EASYSIMD_FLOAT32_C(1514837120.00) } },
    { { UINT32_C(2730530183), UINT32_C(3398034707), UINT32_C( 425564095), UINT32_C(3502960315), UINT32_C(2825238321), UINT32_C(4037558744), UINT32_C( 758607483), UINT32_C(2324136450),
        UINT32_C( 724322071), UINT32_C(3958748460), UINT32_C( 134501197), UINT32_C(1926811457), UINT32_C(1595555462), UINT32_C(4199531135), UINT32_C( 858228528), UINT32_C( 549301769) },
      { EASYSIMD_FLOAT32_C(2730530304.00), EASYSIMD_FLOAT32_C(3398034688.00), EASYSIMD_FLOAT32_C(425564096.00), EASYSIMD_FLOAT32_C(3502960384.00),
        EASYSIMD_FLOAT32_C(2825238272.00), EASYSIMD_FLOAT32_C(4037558784.00), EASYSIMD_FLOAT32_C(758607488.00), EASYSIMD_FLOAT32_C(2324136448.00),
        EASYSIMD_FLOAT32_C(724322048.00), EASYSIMD_FLOAT32_C(3958748416.00), EASYSIMD_FLOAT32_C(134501200.00), EASYSIMD_FLOAT32_C(1926811520.00),
        EASYSIMD_FLOAT32_C(1595555456.00), EASYSIMD_FLOAT32_C(4199531008.00), EASYSIMD_FLOAT32_C(858228544.00), EASYSIMD_FLOAT32_C(549301760.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512 r = easysimd_mm512_cvtepu32_ps(a);
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u32x16();
    easysimd__m512 r = easysimd_mm512_cvtepu32_ps(a);

    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtpd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -943.21), EASYSIMD_FLOAT64_C(  -413.66), EASYSIMD_FLOAT64_C(   700.40), EASYSIMD_FLOAT64_C(   638.73),
        EASYSIMD_FLOAT64_C(  -180.93), EASYSIMD_FLOAT64_C(  -257.44), EASYSIMD_FLOAT64_C(  -992.83), EASYSIMD_FLOAT64_C(  -917.04) },
      { EASYSIMD_FLOAT32_C(  -943.21), EASYSIMD_FLOAT32_C(  -413.66), EASYSIMD_FLOAT32_C(   700.40), EASYSIMD_FLOAT32_C(   638.73),
        EASYSIMD_FLOAT32_C(  -180.93), EASYSIMD_FLOAT32_C(  -257.44), EASYSIMD_FLOAT32_C(  -992.83), EASYSIMD_FLOAT32_C(  -917.04) } },
    { { EASYSIMD_FLOAT64_C(  -431.08), EASYSIMD_FLOAT64_C(  -164.05), EASYSIMD_FLOAT64_C(  -690.90), EASYSIMD_FLOAT64_C(  -120.44),
        EASYSIMD_FLOAT64_C(  -782.94), EASYSIMD_FLOAT64_C(   -19.74), EASYSIMD_FLOAT64_C(   665.29), EASYSIMD_FLOAT64_C(   654.50) },
      { EASYSIMD_FLOAT32_C(  -431.08), EASYSIMD_FLOAT32_C(  -164.05), EASYSIMD_FLOAT32_C(  -690.90), EASYSIMD_FLOAT32_C(  -120.44),
        EASYSIMD_FLOAT32_C(  -782.94), EASYSIMD_FLOAT32_C(   -19.74), EASYSIMD_FLOAT32_C(   665.29), EASYSIMD_FLOAT32_C(   654.50) } },
    { { EASYSIMD_FLOAT64_C(  -982.12), EASYSIMD_FLOAT64_C(   579.14), EASYSIMD_FLOAT64_C(  -143.14), EASYSIMD_FLOAT64_C(  -543.64),
        EASYSIMD_FLOAT64_C(  -707.23), EASYSIMD_FLOAT64_C(  -287.19), EASYSIMD_FLOAT64_C(   951.12), EASYSIMD_FLOAT64_C(   117.91) },
      { EASYSIMD_FLOAT32_C(  -982.12), EASYSIMD_FLOAT32_C(   579.14), EASYSIMD_FLOAT32_C(  -143.14), EASYSIMD_FLOAT32_C(  -543.64),
        EASYSIMD_FLOAT32_C(  -707.23), EASYSIMD_FLOAT32_C(  -287.19), EASYSIMD_FLOAT32_C(   951.12), EASYSIMD_FLOAT32_C(   117.91) } },
    { { EASYSIMD_FLOAT64_C(   -62.08), EASYSIMD_FLOAT64_C(  -610.48), EASYSIMD_FLOAT64_C(  -189.96), EASYSIMD_FLOAT64_C(   -54.92),
        EASYSIMD_FLOAT64_C(  -629.01), EASYSIMD_FLOAT64_C(   429.00), EASYSIMD_FLOAT64_C(  -620.19), EASYSIMD_FLOAT64_C(  -572.23) },
      { EASYSIMD_FLOAT32_C(   -62.08), EASYSIMD_FLOAT32_C(  -610.48), EASYSIMD_FLOAT32_C(  -189.96), EASYSIMD_FLOAT32_C(   -54.92),
        EASYSIMD_FLOAT32_C(  -629.01), EASYSIMD_FLOAT32_C(   429.00), EASYSIMD_FLOAT32_C(  -620.19), EASYSIMD_FLOAT32_C(  -572.23) } },
    { { EASYSIMD_FLOAT64_C(  -984.67), EASYSIMD_FLOAT64_C(  -919.79), EASYSIMD_FLOAT64_C(  -933.50), EASYSIMD_FLOAT64_C(  -165.60),
        EASYSIMD_FLOAT64_C(  -177.23), EASYSIMD_FLOAT64_C(  -926.32), EASYSIMD_FLOAT64_C(   -82.64), EASYSIMD_FLOAT64_C(   391.69) },
      { EASYSIMD_FLOAT32_C(  -984.67), EASYSIMD_FLOAT32_C(  -919.79), EASYSIMD_FLOAT32_C(  -933.50), EASYSIMD_FLOAT32_C(  -165.60),
        EASYSIMD_FLOAT32_C(  -177.23), EASYSIMD_FLOAT32_C(  -926.32), EASYSIMD_FLOAT32_C(   -82.64), EASYSIMD_FLOAT32_C(   391.69) } },
    { { EASYSIMD_FLOAT64_C(   -90.37), EASYSIMD_FLOAT64_C(   226.46), EASYSIMD_FLOAT64_C(  -728.75), EASYSIMD_FLOAT64_C(   126.69),
        EASYSIMD_FLOAT64_C(  -793.28), EASYSIMD_FLOAT64_C(   936.54), EASYSIMD_FLOAT64_C(  -218.81), EASYSIMD_FLOAT64_C(  -775.40) },
      { EASYSIMD_FLOAT32_C(   -90.37), EASYSIMD_FLOAT32_C(   226.46), EASYSIMD_FLOAT32_C(  -728.75), EASYSIMD_FLOAT32_C(   126.69),
        EASYSIMD_FLOAT32_C(  -793.28), EASYSIMD_FLOAT32_C(   936.54), EASYSIMD_FLOAT32_C(  -218.81), EASYSIMD_FLOAT32_C(  -775.40) } },
    { { EASYSIMD_FLOAT64_C(   515.67), EASYSIMD_FLOAT64_C(   638.04), EASYSIMD_FLOAT64_C(  -319.04), EASYSIMD_FLOAT64_C(   808.44),
        EASYSIMD_FLOAT64_C(  -649.14), EASYSIMD_FLOAT64_C(  -367.92), EASYSIMD_FLOAT64_C(   -73.65), EASYSIMD_FLOAT64_C(   288.78) },
      { EASYSIMD_FLOAT32_C(   515.67), EASYSIMD_FLOAT32_C(   638.04), EASYSIMD_FLOAT32_C(  -319.04), EASYSIMD_FLOAT32_C(   808.44),
        EASYSIMD_FLOAT32_C(  -649.14), EASYSIMD_FLOAT32_C(  -367.92), EASYSIMD_FLOAT32_C(   -73.65), EASYSIMD_FLOAT32_C(   288.78) } },
    { { EASYSIMD_FLOAT64_C(    21.60), EASYSIMD_FLOAT64_C(   736.39), EASYSIMD_FLOAT64_C(  -766.15), EASYSIMD_FLOAT64_C(   392.59),
        EASYSIMD_FLOAT64_C(   165.39), EASYSIMD_FLOAT64_C(  -386.34), EASYSIMD_FLOAT64_C(   820.36), EASYSIMD_FLOAT64_C(   180.72) },
      { EASYSIMD_FLOAT32_C(    21.60), EASYSIMD_FLOAT32_C(   736.39), EASYSIMD_FLOAT32_C(  -766.15), EASYSIMD_FLOAT32_C(   392.59),
        EASYSIMD_FLOAT32_C(   165.39), EASYSIMD_FLOAT32_C(  -386.34), EASYSIMD_FLOAT32_C(   820.36), EASYSIMD_FLOAT32_C(   180.72) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvtpd_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cvtpd_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256 r = easysimd_mm512_cvtpd_ps(a);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtpd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float64 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -318.46), EASYSIMD_FLOAT32_C(   453.99), EASYSIMD_FLOAT32_C(   621.81), EASYSIMD_FLOAT32_C(   664.22),
        EASYSIMD_FLOAT32_C(  -381.34), EASYSIMD_FLOAT32_C(  -898.50), EASYSIMD_FLOAT32_C(   737.78), EASYSIMD_FLOAT32_C(   783.82) },
      UINT8_C(222),
      { EASYSIMD_FLOAT64_C(    58.76), EASYSIMD_FLOAT64_C(   -21.76), EASYSIMD_FLOAT64_C(   746.05), EASYSIMD_FLOAT64_C(   864.28),
        EASYSIMD_FLOAT64_C(  -425.41), EASYSIMD_FLOAT64_C(  -174.33), EASYSIMD_FLOAT64_C(  -682.12), EASYSIMD_FLOAT64_C(   -53.37) },
      { EASYSIMD_FLOAT32_C(  -318.46), EASYSIMD_FLOAT32_C(   -21.76), EASYSIMD_FLOAT32_C(   746.05), EASYSIMD_FLOAT32_C(   864.28),
        EASYSIMD_FLOAT32_C(  -425.41), EASYSIMD_FLOAT32_C(  -898.50), EASYSIMD_FLOAT32_C(  -682.12), EASYSIMD_FLOAT32_C(   -53.37) } },
    { { EASYSIMD_FLOAT32_C(   997.65), EASYSIMD_FLOAT32_C(   882.11), EASYSIMD_FLOAT32_C(  -146.68), EASYSIMD_FLOAT32_C(  -540.34),
        EASYSIMD_FLOAT32_C(   880.29), EASYSIMD_FLOAT32_C(   866.02), EASYSIMD_FLOAT32_C(   372.38), EASYSIMD_FLOAT32_C(  -699.92) },
      UINT8_C( 33),
      { EASYSIMD_FLOAT64_C(  -571.94), EASYSIMD_FLOAT64_C(   189.70), EASYSIMD_FLOAT64_C(   888.04), EASYSIMD_FLOAT64_C(   301.61),
        EASYSIMD_FLOAT64_C(  -348.39), EASYSIMD_FLOAT64_C(  -430.42), EASYSIMD_FLOAT64_C(  -244.41), EASYSIMD_FLOAT64_C(  -726.57) },
      { EASYSIMD_FLOAT32_C(  -571.94), EASYSIMD_FLOAT32_C(   882.11), EASYSIMD_FLOAT32_C(  -146.68), EASYSIMD_FLOAT32_C(  -540.34),
        EASYSIMD_FLOAT32_C(   880.29), EASYSIMD_FLOAT32_C(  -430.42), EASYSIMD_FLOAT32_C(   372.38), EASYSIMD_FLOAT32_C(  -699.92) } },
    { { EASYSIMD_FLOAT32_C(  -766.20), EASYSIMD_FLOAT32_C(   374.25), EASYSIMD_FLOAT32_C(  -625.07), EASYSIMD_FLOAT32_C(   971.58),
        EASYSIMD_FLOAT32_C(   158.08), EASYSIMD_FLOAT32_C(  -962.02), EASYSIMD_FLOAT32_C(    30.34), EASYSIMD_FLOAT32_C(  -863.68) },
      UINT8_C(187),
      { EASYSIMD_FLOAT64_C(  -105.38), EASYSIMD_FLOAT64_C(  -289.09), EASYSIMD_FLOAT64_C(  -390.31), EASYSIMD_FLOAT64_C(   212.50),
        EASYSIMD_FLOAT64_C(   657.54), EASYSIMD_FLOAT64_C(  -392.66), EASYSIMD_FLOAT64_C(    94.61), EASYSIMD_FLOAT64_C(  -489.14) },
      { EASYSIMD_FLOAT32_C(  -105.38), EASYSIMD_FLOAT32_C(  -289.09), EASYSIMD_FLOAT32_C(  -625.07), EASYSIMD_FLOAT32_C(   212.50),
        EASYSIMD_FLOAT32_C(   657.54), EASYSIMD_FLOAT32_C(  -392.66), EASYSIMD_FLOAT32_C(    30.34), EASYSIMD_FLOAT32_C(  -489.14) } },
    { { EASYSIMD_FLOAT32_C(    67.00), EASYSIMD_FLOAT32_C(   -25.10), EASYSIMD_FLOAT32_C(  -623.12), EASYSIMD_FLOAT32_C(  -560.62),
        EASYSIMD_FLOAT32_C(   274.98), EASYSIMD_FLOAT32_C(   970.80), EASYSIMD_FLOAT32_C(  -132.56), EASYSIMD_FLOAT32_C(  -535.32) },
      UINT8_C(223),
      { EASYSIMD_FLOAT64_C(  -830.95), EASYSIMD_FLOAT64_C(   116.29), EASYSIMD_FLOAT64_C(  -571.57), EASYSIMD_FLOAT64_C(   -75.36),
        EASYSIMD_FLOAT64_C(   389.72), EASYSIMD_FLOAT64_C(  -337.77), EASYSIMD_FLOAT64_C(  -701.10), EASYSIMD_FLOAT64_C(   764.65) },
      { EASYSIMD_FLOAT32_C(  -830.95), EASYSIMD_FLOAT32_C(   116.29), EASYSIMD_FLOAT32_C(  -571.57), EASYSIMD_FLOAT32_C(   -75.36),
        EASYSIMD_FLOAT32_C(   389.72), EASYSIMD_FLOAT32_C(   970.80), EASYSIMD_FLOAT32_C(  -701.10), EASYSIMD_FLOAT32_C(   764.65) } },
    { { EASYSIMD_FLOAT32_C(  -366.19), EASYSIMD_FLOAT32_C(   456.97), EASYSIMD_FLOAT32_C(   802.62), EASYSIMD_FLOAT32_C(   664.15),
        EASYSIMD_FLOAT32_C(   593.29), EASYSIMD_FLOAT32_C(   586.64), EASYSIMD_FLOAT32_C(  -441.23), EASYSIMD_FLOAT32_C(  -695.80) },
      UINT8_C(111),
      { EASYSIMD_FLOAT64_C(   771.27), EASYSIMD_FLOAT64_C(   961.74), EASYSIMD_FLOAT64_C(  -196.32), EASYSIMD_FLOAT64_C(  -134.12),
        EASYSIMD_FLOAT64_C(  -527.40), EASYSIMD_FLOAT64_C(   870.68), EASYSIMD_FLOAT64_C(   840.78), EASYSIMD_FLOAT64_C(  -150.52) },
      { EASYSIMD_FLOAT32_C(   771.27), EASYSIMD_FLOAT32_C(   961.74), EASYSIMD_FLOAT32_C(  -196.32), EASYSIMD_FLOAT32_C(  -134.12),
        EASYSIMD_FLOAT32_C(   593.29), EASYSIMD_FLOAT32_C(   870.68), EASYSIMD_FLOAT32_C(   840.78), EASYSIMD_FLOAT32_C(  -695.80) } },
    { { EASYSIMD_FLOAT32_C(  -689.94), EASYSIMD_FLOAT32_C(   115.76), EASYSIMD_FLOAT32_C(  -179.71), EASYSIMD_FLOAT32_C(   177.50),
        EASYSIMD_FLOAT32_C(   580.44), EASYSIMD_FLOAT32_C(  -320.87), EASYSIMD_FLOAT32_C(   346.55), EASYSIMD_FLOAT32_C(  -303.27) },
      UINT8_C( 58),
      { EASYSIMD_FLOAT64_C(  -728.81), EASYSIMD_FLOAT64_C(  -913.55), EASYSIMD_FLOAT64_C(   769.79), EASYSIMD_FLOAT64_C(  -429.92),
        EASYSIMD_FLOAT64_C(   851.10), EASYSIMD_FLOAT64_C(  -596.40), EASYSIMD_FLOAT64_C(  -972.94), EASYSIMD_FLOAT64_C(   653.72) },
      { EASYSIMD_FLOAT32_C(  -689.94), EASYSIMD_FLOAT32_C(  -913.55), EASYSIMD_FLOAT32_C(  -179.71), EASYSIMD_FLOAT32_C(  -429.92),
        EASYSIMD_FLOAT32_C(   851.10), EASYSIMD_FLOAT32_C(  -596.40), EASYSIMD_FLOAT32_C(   346.55), EASYSIMD_FLOAT32_C(  -303.27) } },
    { { EASYSIMD_FLOAT32_C(  -932.25), EASYSIMD_FLOAT32_C(   620.35), EASYSIMD_FLOAT32_C(   240.36), EASYSIMD_FLOAT32_C(  -373.48),
        EASYSIMD_FLOAT32_C(   924.55), EASYSIMD_FLOAT32_C(   436.70), EASYSIMD_FLOAT32_C(  -602.20), EASYSIMD_FLOAT32_C(   886.29) },
      UINT8_C( 28),
      { EASYSIMD_FLOAT64_C(   263.68), EASYSIMD_FLOAT64_C(  -641.11), EASYSIMD_FLOAT64_C(  -888.95), EASYSIMD_FLOAT64_C(   104.46),
        EASYSIMD_FLOAT64_C(   208.38), EASYSIMD_FLOAT64_C(  -578.89), EASYSIMD_FLOAT64_C(  -779.78), EASYSIMD_FLOAT64_C(  -971.34) },
      { EASYSIMD_FLOAT32_C(  -932.25), EASYSIMD_FLOAT32_C(   620.35), EASYSIMD_FLOAT32_C(  -888.95), EASYSIMD_FLOAT32_C(   104.46),
        EASYSIMD_FLOAT32_C(   208.38), EASYSIMD_FLOAT32_C(   436.70), EASYSIMD_FLOAT32_C(  -602.20), EASYSIMD_FLOAT32_C(   886.29) } },
    { { EASYSIMD_FLOAT32_C(   598.61), EASYSIMD_FLOAT32_C(   800.66), EASYSIMD_FLOAT32_C(  -292.20), EASYSIMD_FLOAT32_C(   -54.84),
        EASYSIMD_FLOAT32_C(  -502.61), EASYSIMD_FLOAT32_C(   815.36), EASYSIMD_FLOAT32_C(   216.35), EASYSIMD_FLOAT32_C(  -416.16) },
      UINT8_C(129),
      { EASYSIMD_FLOAT64_C(   786.43), EASYSIMD_FLOAT64_C(  -565.07), EASYSIMD_FLOAT64_C(   988.75), EASYSIMD_FLOAT64_C(   813.49),
        EASYSIMD_FLOAT64_C(  -911.35), EASYSIMD_FLOAT64_C(  -943.50), EASYSIMD_FLOAT64_C(   433.83), EASYSIMD_FLOAT64_C(   329.01) },
      { EASYSIMD_FLOAT32_C(   786.43), EASYSIMD_FLOAT32_C(   800.66), EASYSIMD_FLOAT32_C(  -292.20), EASYSIMD_FLOAT32_C(   -54.84),
        EASYSIMD_FLOAT32_C(  -502.61), EASYSIMD_FLOAT32_C(   815.36), EASYSIMD_FLOAT32_C(   216.35), EASYSIMD_FLOAT32_C(   329.01) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cvtpd_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_cvtpd_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm512_mask_cvtpd_ps(src, k, a);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_cvtpd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(137),
      { EASYSIMD_FLOAT64_C(  -596.22), EASYSIMD_FLOAT64_C(  -839.99), EASYSIMD_FLOAT64_C(   447.96), EASYSIMD_FLOAT64_C(   257.74),
        EASYSIMD_FLOAT64_C(   898.78), EASYSIMD_FLOAT64_C(   237.41), EASYSIMD_FLOAT64_C(  -825.88), EASYSIMD_FLOAT64_C(    50.49) },
      { EASYSIMD_FLOAT32_C(  -596.22), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   257.74),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    50.49) } },
    { UINT8_C( 36),
      { EASYSIMD_FLOAT64_C(   888.46), EASYSIMD_FLOAT64_C(   678.91), EASYSIMD_FLOAT64_C(  -300.79), EASYSIMD_FLOAT64_C(    26.35),
        EASYSIMD_FLOAT64_C(   629.06), EASYSIMD_FLOAT64_C(   599.80), EASYSIMD_FLOAT64_C(   305.58), EASYSIMD_FLOAT64_C(  -749.67) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -300.79), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   599.80), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(110),
      { EASYSIMD_FLOAT64_C(  -517.22), EASYSIMD_FLOAT64_C(   142.71), EASYSIMD_FLOAT64_C(  -824.37), EASYSIMD_FLOAT64_C(   994.49),
        EASYSIMD_FLOAT64_C(   106.82), EASYSIMD_FLOAT64_C(    82.61), EASYSIMD_FLOAT64_C(   946.00), EASYSIMD_FLOAT64_C(  -527.34) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   142.71), EASYSIMD_FLOAT32_C(  -824.37), EASYSIMD_FLOAT32_C(   994.49),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    82.61), EASYSIMD_FLOAT32_C(   946.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 22),
      { EASYSIMD_FLOAT64_C(   102.80), EASYSIMD_FLOAT64_C(    83.38), EASYSIMD_FLOAT64_C(  -336.92), EASYSIMD_FLOAT64_C(   911.91),
        EASYSIMD_FLOAT64_C(   487.16), EASYSIMD_FLOAT64_C(  -176.91), EASYSIMD_FLOAT64_C(   359.88), EASYSIMD_FLOAT64_C(  -255.10) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    83.38), EASYSIMD_FLOAT32_C(  -336.92), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   487.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(174),
      { EASYSIMD_FLOAT64_C(  -402.72), EASYSIMD_FLOAT64_C(   -80.98), EASYSIMD_FLOAT64_C(   772.37), EASYSIMD_FLOAT64_C(   -47.08),
        EASYSIMD_FLOAT64_C(  -192.52), EASYSIMD_FLOAT64_C(   451.27), EASYSIMD_FLOAT64_C(   652.13), EASYSIMD_FLOAT64_C(   833.83) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -80.98), EASYSIMD_FLOAT32_C(   772.37), EASYSIMD_FLOAT32_C(   -47.08),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   451.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   833.83) } },
    { UINT8_C( 87),
      { EASYSIMD_FLOAT64_C(   251.93), EASYSIMD_FLOAT64_C(   139.41), EASYSIMD_FLOAT64_C(   330.67), EASYSIMD_FLOAT64_C(  -597.02),
        EASYSIMD_FLOAT64_C(   622.18), EASYSIMD_FLOAT64_C(  -526.62), EASYSIMD_FLOAT64_C(  -421.39), EASYSIMD_FLOAT64_C(   616.67) },
      { EASYSIMD_FLOAT32_C(   251.93), EASYSIMD_FLOAT32_C(   139.41), EASYSIMD_FLOAT32_C(   330.67), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   622.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -421.39), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(121),
      { EASYSIMD_FLOAT64_C(   661.21), EASYSIMD_FLOAT64_C(   562.67), EASYSIMD_FLOAT64_C(  -947.14), EASYSIMD_FLOAT64_C(   798.08),
        EASYSIMD_FLOAT64_C(  -334.52), EASYSIMD_FLOAT64_C(   136.23), EASYSIMD_FLOAT64_C(  -538.83), EASYSIMD_FLOAT64_C(  -422.61) },
      { EASYSIMD_FLOAT32_C(   661.21), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   798.08),
        EASYSIMD_FLOAT32_C(  -334.52), EASYSIMD_FLOAT32_C(   136.23), EASYSIMD_FLOAT32_C(  -538.83), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(185),
      { EASYSIMD_FLOAT64_C(   284.26), EASYSIMD_FLOAT64_C(   937.27), EASYSIMD_FLOAT64_C(   368.29), EASYSIMD_FLOAT64_C(  -993.87),
        EASYSIMD_FLOAT64_C(  -465.45), EASYSIMD_FLOAT64_C(  -712.69), EASYSIMD_FLOAT64_C(   778.50), EASYSIMD_FLOAT64_C(   487.47) },
      { EASYSIMD_FLOAT32_C(   284.26), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -993.87),
        EASYSIMD_FLOAT32_C(  -465.45), EASYSIMD_FLOAT32_C(  -712.69), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   487.47) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_cvtpd_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_cvtpd_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm512_maskz_cvtpd_ps(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtph_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { UINT16_C(51482), UINT16_C(24801), UINT16_C(59802), UINT16_C(13138), UINT16_C(35245), UINT16_C( 1652), UINT16_C(19260), UINT16_C(62499),
        UINT16_C(37027), UINT16_C(13652), UINT16_C(57450), UINT16_C( 4263), UINT16_C(17060), UINT16_C(26418), UINT16_C(50171), UINT16_C( 5502) },
      { EASYSIMD_FLOAT32_C(   -10.20), EASYSIMD_FLOAT32_C(   624.50), EASYSIMD_FLOAT32_C( -2868.00), EASYSIMD_FLOAT32_C(     0.23),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    14.47), EASYSIMD_FLOAT32_C(-16944.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.33), EASYSIMD_FLOAT32_C(  -565.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     3.32), EASYSIMD_FLOAT32_C(  1842.00), EASYSIMD_FLOAT32_C(    -3.99), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT16_C(24460), UINT16_C( 9845), UINT16_C(51272), UINT16_C(63066), UINT16_C(52817), UINT16_C(36604), UINT16_C( 7961), UINT16_C(48258),
        UINT16_C(54960), UINT16_C( 6897), UINT16_C(39094), UINT16_C(23338), UINT16_C(23770), UINT16_C(54978), UINT16_C(16415), UINT16_C(44011) },
      { EASYSIMD_FLOAT32_C(   483.00), EASYSIMD_FLOAT32_C(     0.03), EASYSIMD_FLOAT32_C(    -8.56), EASYSIMD_FLOAT32_C(-26016.00),
        EASYSIMD_FLOAT32_C(   -25.27), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.01), EASYSIMD_FLOAT32_C(    -1.13),
        EASYSIMD_FLOAT32_C(  -107.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(   229.25),
        EASYSIMD_FLOAT32_C(   310.50), EASYSIMD_FLOAT32_C(  -108.12), EASYSIMD_FLOAT32_C(     2.06), EASYSIMD_FLOAT32_C(    -0.06) } },
    { { UINT16_C(24991), UINT16_C(59346), UINT16_C(11305), UINT16_C(31453), UINT16_C(55802), UINT16_C( 4872), UINT16_C(35577), UINT16_C(43471),
        UINT16_C(49249), UINT16_C( 6083), UINT16_C(60760), UINT16_C(12914), UINT16_C(13386), UINT16_C(26888), UINT16_C(62580), UINT16_C( 4885) },
      { EASYSIMD_FLOAT32_C(   719.50), EASYSIMD_FLOAT32_C( -2002.00), EASYSIMD_FLOAT32_C(     0.07), EASYSIMD_FLOAT32_C( 56224.00),
        EASYSIMD_FLOAT32_C(  -191.25), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.05),
        EASYSIMD_FLOAT32_C(    -2.19), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -5472.00), EASYSIMD_FLOAT32_C(     0.20),
        EASYSIMD_FLOAT32_C(     0.27), EASYSIMD_FLOAT32_C(  2576.00), EASYSIMD_FLOAT32_C(-18240.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { UINT16_C(59221), UINT16_C(32507), UINT16_C(55315), UINT16_C( 3576), UINT16_C(  434), UINT16_C(43808), UINT16_C(61323), UINT16_C(60500),
        UINT16_C( 6063), UINT16_C( 1796), UINT16_C(30212), UINT16_C(20025), UINT16_C(17067), UINT16_C( 8120), UINT16_C(52534), UINT16_C(35635) },
      { EASYSIMD_FLOAT32_C( -1877.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -130.38), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.06), EASYSIMD_FLOAT32_C( -7724.00), EASYSIMD_FLOAT32_C( -4432.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 24640.00), EASYSIMD_FLOAT32_C(    24.89),
        EASYSIMD_FLOAT32_C(     3.33), EASYSIMD_FLOAT32_C(     0.01), EASYSIMD_FLOAT32_C(   -20.84), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { UINT16_C(11956), UINT16_C(50953), UINT16_C(  262), UINT16_C(47316), UINT16_C(62466), UINT16_C(36451), UINT16_C(47075), UINT16_C(37498),
        UINT16_C(32462), UINT16_C(54169), UINT16_C(54005), UINT16_C(40993), UINT16_C(55572), UINT16_C(19135), UINT16_C(62118), UINT16_C(23253) },
      { EASYSIMD_FLOAT32_C(     0.10), EASYSIMD_FLOAT32_C(    -7.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.60),
        EASYSIMD_FLOAT32_C(-16416.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.49), EASYSIMD_FLOAT32_C(    -0.00),
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   -60.78), EASYSIMD_FLOAT32_C(   -55.66), EASYSIMD_FLOAT32_C(    -0.01),
        EASYSIMD_FLOAT32_C(  -162.50), EASYSIMD_FLOAT32_C(    13.49), EASYSIMD_FLOAT32_C(-13616.00), EASYSIMD_FLOAT32_C(   218.62) } },
    { { UINT16_C(56864), UINT16_C(10017), UINT16_C(62944), UINT16_C(58079), UINT16_C(17385), UINT16_C(52336), UINT16_C(60410), UINT16_C(51550),
        UINT16_C(63337), UINT16_C(24220), UINT16_C(48586), UINT16_C(57086), UINT16_C(48791), UINT16_C(15657), UINT16_C(65200), UINT16_C(53656) },
      { EASYSIMD_FLOAT32_C(  -392.00), EASYSIMD_FLOAT32_C(     0.03), EASYSIMD_FLOAT32_C(-24064.00), EASYSIMD_FLOAT32_C(  -879.50),
        EASYSIMD_FLOAT32_C(     3.96), EASYSIMD_FLOAT32_C(   -17.75), EASYSIMD_FLOAT32_C( -4084.00), EASYSIMD_FLOAT32_C(   -10.73),
        EASYSIMD_FLOAT32_C(-30352.00), EASYSIMD_FLOAT32_C(   423.00), EASYSIMD_FLOAT32_C(    -1.45), EASYSIMD_FLOAT32_C(  -447.50),
        EASYSIMD_FLOAT32_C(    -1.65), EASYSIMD_FLOAT32_C(     1.29),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   -44.75) } },
    { { UINT16_C(47581), UINT16_C(48632), UINT16_C(55215), UINT16_C(39071), UINT16_C( 4122), UINT16_C( 5477), UINT16_C(50171), UINT16_C(25822),
        UINT16_C(31419), UINT16_C(34243), UINT16_C(49463), UINT16_C(52835), UINT16_C(35967), UINT16_C(12300), UINT16_C(42123), UINT16_C(26625) },
      { EASYSIMD_FLOAT32_C(    -0.73), EASYSIMD_FLOAT32_C(    -1.49), EASYSIMD_FLOAT32_C(  -122.94), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -3.99), EASYSIMD_FLOAT32_C(  1246.00),
        EASYSIMD_FLOAT32_C( 55136.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -2.61), EASYSIMD_FLOAT32_C(   -25.55),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.13), EASYSIMD_FLOAT32_C(    -0.02), EASYSIMD_FLOAT32_C(  2050.00) } },
    { { UINT16_C(63837), UINT16_C( 3109), UINT16_C(50384), UINT16_C(60325), UINT16_C( 2772), UINT16_C(52992), UINT16_C(57037), UINT16_C(34868),
        UINT16_C(63320), UINT16_C(36621), UINT16_C(29112), UINT16_C(14430), UINT16_C(27389), UINT16_C(34920), UINT16_C(26894), UINT16_C(27632) },
      { EASYSIMD_FLOAT32_C(-43936.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -4.81), EASYSIMD_FLOAT32_C( -3914.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -28.00), EASYSIMD_FLOAT32_C(  -435.25), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(-30080.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C( 11712.00), EASYSIMD_FLOAT32_C(     0.55),
        EASYSIMD_FLOAT32_C(  3578.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  2588.00), EASYSIMD_FLOAT32_C(  4064.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvtph_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cvtph_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m512 r = easysimd_mm512_cvtph_ps(a);

    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepi16_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[16];
    int32_t r[16];
  } test_vec[8] = {
    { {  INT16_C( 32025),  INT16_C( 31049),  INT16_C( 19483),  INT16_C( 13330), -INT16_C( 17150), -INT16_C( 28546), -INT16_C( 20418), -INT16_C( 23882),
         INT16_C( 29310), -INT16_C( 13967), -INT16_C( 19196), -INT16_C(  8444), -INT16_C(  8705), -INT16_C( 14413),  INT16_C(  2399),  INT16_C( 30784) },
      {  INT32_C(       32025),  INT32_C(       31049),  INT32_C(       19483),  INT32_C(       13330), -INT32_C(       17150), -INT32_C(       28546), -INT32_C(       20418), -INT32_C(       23882),
         INT32_C(       29310), -INT32_C(       13967), -INT32_C(       19196), -INT32_C(        8444), -INT32_C(        8705), -INT32_C(       14413),  INT32_C(        2399),  INT32_C(       30784) } },
    { { -INT16_C( 30074), -INT16_C( 24079),  INT16_C(   982), -INT16_C( 10026),  INT16_C( 21697), -INT16_C(   151),  INT16_C(  7941), -INT16_C( 31838),
         INT16_C(  5010), -INT16_C( 27060),  INT16_C( 20936), -INT16_C( 14475),  INT16_C( 10286), -INT16_C( 29298), -INT16_C( 12494), -INT16_C( 18426) },
      { -INT32_C(       30074), -INT32_C(       24079),  INT32_C(         982), -INT32_C(       10026),  INT32_C(       21697), -INT32_C(         151),  INT32_C(        7941), -INT32_C(       31838),
         INT32_C(        5010), -INT32_C(       27060),  INT32_C(       20936), -INT32_C(       14475),  INT32_C(       10286), -INT32_C(       29298), -INT32_C(       12494), -INT32_C(       18426) } },
    { { -INT16_C(  2215),  INT16_C( 12122),  INT16_C( 12539), -INT16_C( 17400),  INT16_C( 29060), -INT16_C( 30277),  INT16_C( 23952),  INT16_C(  8716),
         INT16_C( 22896),  INT16_C( 14520),  INT16_C( 11946), -INT16_C(  9985), -INT16_C( 29098), -INT16_C( 30619),  INT16_C( 27485), -INT16_C( 18879) },
      { -INT32_C(        2215),  INT32_C(       12122),  INT32_C(       12539), -INT32_C(       17400),  INT32_C(       29060), -INT32_C(       30277),  INT32_C(       23952),  INT32_C(        8716),
         INT32_C(       22896),  INT32_C(       14520),  INT32_C(       11946), -INT32_C(        9985), -INT32_C(       29098), -INT32_C(       30619),  INT32_C(       27485), -INT32_C(       18879) } },
    { { -INT16_C( 25757),  INT16_C( 24293), -INT16_C(  4661),  INT16_C( 20250), -INT16_C( 10914), -INT16_C(  4135), -INT16_C(  6861), -INT16_C( 23791),
        -INT16_C( 13762), -INT16_C(  5924), -INT16_C(  9224),  INT16_C( 20160),  INT16_C(  9833), -INT16_C( 14633),  INT16_C(  6289), -INT16_C(  2948) },
      { -INT32_C(       25757),  INT32_C(       24293), -INT32_C(        4661),  INT32_C(       20250), -INT32_C(       10914), -INT32_C(        4135), -INT32_C(        6861), -INT32_C(       23791),
        -INT32_C(       13762), -INT32_C(        5924), -INT32_C(        9224),  INT32_C(       20160),  INT32_C(        9833), -INT32_C(       14633),  INT32_C(        6289), -INT32_C(        2948) } },
    { {  INT16_C( 25267),  INT16_C( 32338),  INT16_C( 27727), -INT16_C( 20787), -INT16_C( 22974),  INT16_C( 30109), -INT16_C( 20852), -INT16_C( 13800),
        -INT16_C(  2952),  INT16_C( 28851),  INT16_C( 29648),  INT16_C( 14783), -INT16_C( 26983),  INT16_C( 11008),  INT16_C( 31918),  INT16_C( 24863) },
      {  INT32_C(       25267),  INT32_C(       32338),  INT32_C(       27727), -INT32_C(       20787), -INT32_C(       22974),  INT32_C(       30109), -INT32_C(       20852), -INT32_C(       13800),
        -INT32_C(        2952),  INT32_C(       28851),  INT32_C(       29648),  INT32_C(       14783), -INT32_C(       26983),  INT32_C(       11008),  INT32_C(       31918),  INT32_C(       24863) } },
    { {  INT16_C( 29406),  INT16_C( 11999), -INT16_C( 21282),  INT16_C(  8412),  INT16_C( 31059), -INT16_C(  8299), -INT16_C( 20953), -INT16_C( 24407),
         INT16_C( 23714),  INT16_C( 29200), -INT16_C( 12336),  INT16_C( 27052), -INT16_C( 21403),  INT16_C(  5012), -INT16_C( 19416),  INT16_C(  1908) },
      {  INT32_C(       29406),  INT32_C(       11999), -INT32_C(       21282),  INT32_C(        8412),  INT32_C(       31059), -INT32_C(        8299), -INT32_C(       20953), -INT32_C(       24407),
         INT32_C(       23714),  INT32_C(       29200), -INT32_C(       12336),  INT32_C(       27052), -INT32_C(       21403),  INT32_C(        5012), -INT32_C(       19416),  INT32_C(        1908) } },
    { {  INT16_C( 21286),  INT16_C(  1077),  INT16_C(  4352),  INT16_C( 21285), -INT16_C( 17782), -INT16_C( 20174), -INT16_C(  9368),  INT16_C(  2897),
         INT16_C( 25144),  INT16_C(  2173),  INT16_C( 10545), -INT16_C( 26767),  INT16_C(  1749), -INT16_C(   342),  INT16_C(  8122), -INT16_C(  8187) },
      {  INT32_C(       21286),  INT32_C(        1077),  INT32_C(        4352),  INT32_C(       21285), -INT32_C(       17782), -INT32_C(       20174), -INT32_C(        9368),  INT32_C(        2897),
         INT32_C(       25144),  INT32_C(        2173),  INT32_C(       10545), -INT32_C(       26767),  INT32_C(        1749), -INT32_C(         342),  INT32_C(        8122), -INT32_C(        8187) } },
    { {  INT16_C( 14962),  INT16_C( 29412),  INT16_C(  2379), -INT16_C( 10811), -INT16_C(  2108),  INT16_C( 11398), -INT16_C( 10029),  INT16_C(  2871),
        -INT16_C( 19142),  INT16_C( 27411), -INT16_C( 31522), -INT16_C( 19454), -INT16_C( 21110),  INT16_C( 17586), -INT16_C( 18484),  INT16_C( 15908) },
      {  INT32_C(       14962),  INT32_C(       29412),  INT32_C(        2379), -INT32_C(       10811), -INT32_C(        2108),  INT32_C(       11398), -INT32_C(       10029),  INT32_C(        2871),
        -INT32_C(       19142),  INT32_C(       27411), -INT32_C(       31522), -INT32_C(       19454), -INT32_C(       21110),  INT32_C(       17586), -INT32_C(       18484),  INT32_C(       15908) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_cvtepi16_epi32(a);
    } EASYSIMD_TEST_PERF_END("_mm512_cvtepi16_epi32");
    easysimd_assert_m512i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m512i r = easysimd_mm512_cvtepi16_epi32(a);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu16_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[16];
    int32_t r[16];
  } test_vec[8] = {
    { { UINT16_C(54075), UINT16_C( 7951), UINT16_C(54220), UINT16_C(36748), UINT16_C(17522), UINT16_C(17165), UINT16_C(44972), UINT16_C(64255),
        UINT16_C(24616), UINT16_C( 2868), UINT16_C(39483), UINT16_C(40162), UINT16_C(25697), UINT16_C( 8380), UINT16_C(62399), UINT16_C(64109) },
      {  INT32_C(       54075),  INT32_C(        7951),  INT32_C(       54220),  INT32_C(       36748),  INT32_C(       17522),  INT32_C(       17165),  INT32_C(       44972),  INT32_C(       64255),
         INT32_C(       24616),  INT32_C(        2868),  INT32_C(       39483),  INT32_C(       40162),  INT32_C(       25697),  INT32_C(        8380),  INT32_C(       62399),  INT32_C(       64109) } },
    { { UINT16_C(31942), UINT16_C(37402), UINT16_C(42575), UINT16_C(49697), UINT16_C(12010), UINT16_C(38661), UINT16_C( 1245), UINT16_C( 1425),
        UINT16_C(50533), UINT16_C(40976), UINT16_C(62303), UINT16_C(49469), UINT16_C(63831), UINT16_C( 5857), UINT16_C(20204), UINT16_C(45585) },
      {  INT32_C(       31942),  INT32_C(       37402),  INT32_C(       42575),  INT32_C(       49697),  INT32_C(       12010),  INT32_C(       38661),  INT32_C(        1245),  INT32_C(        1425),
         INT32_C(       50533),  INT32_C(       40976),  INT32_C(       62303),  INT32_C(       49469),  INT32_C(       63831),  INT32_C(        5857),  INT32_C(       20204),  INT32_C(       45585) } },
    { { UINT16_C(11210), UINT16_C( 6468), UINT16_C(26065), UINT16_C(48347), UINT16_C(57748), UINT16_C(29011), UINT16_C(58597), UINT16_C(19063),
        UINT16_C(34730), UINT16_C( 2539), UINT16_C(10362), UINT16_C(53706), UINT16_C(43809), UINT16_C( 3816), UINT16_C(63993), UINT16_C(50112) },
      {  INT32_C(       11210),  INT32_C(        6468),  INT32_C(       26065),  INT32_C(       48347),  INT32_C(       57748),  INT32_C(       29011),  INT32_C(       58597),  INT32_C(       19063),
         INT32_C(       34730),  INT32_C(        2539),  INT32_C(       10362),  INT32_C(       53706),  INT32_C(       43809),  INT32_C(        3816),  INT32_C(       63993),  INT32_C(       50112) } },
    { { UINT16_C( 1316), UINT16_C(62941), UINT16_C(47210), UINT16_C(65201), UINT16_C( 1177), UINT16_C(32624), UINT16_C(59369), UINT16_C(37833),
        UINT16_C(46190), UINT16_C(59804), UINT16_C(26588), UINT16_C(65210), UINT16_C(41490), UINT16_C( 3084), UINT16_C(52379), UINT16_C(49103) },
      {  INT32_C(        1316),  INT32_C(       62941),  INT32_C(       47210),  INT32_C(       65201),  INT32_C(        1177),  INT32_C(       32624),  INT32_C(       59369),  INT32_C(       37833),
         INT32_C(       46190),  INT32_C(       59804),  INT32_C(       26588),  INT32_C(       65210),  INT32_C(       41490),  INT32_C(        3084),  INT32_C(       52379),  INT32_C(       49103) } },
    { { UINT16_C(44241), UINT16_C(15541), UINT16_C(26213), UINT16_C(65082), UINT16_C(43627), UINT16_C(21629), UINT16_C(18321), UINT16_C(  231),
        UINT16_C(33787), UINT16_C(55529), UINT16_C(41962), UINT16_C(64982), UINT16_C(57926), UINT16_C(57609), UINT16_C(55470), UINT16_C(32929) },
      {  INT32_C(       44241),  INT32_C(       15541),  INT32_C(       26213),  INT32_C(       65082),  INT32_C(       43627),  INT32_C(       21629),  INT32_C(       18321),  INT32_C(         231),
         INT32_C(       33787),  INT32_C(       55529),  INT32_C(       41962),  INT32_C(       64982),  INT32_C(       57926),  INT32_C(       57609),  INT32_C(       55470),  INT32_C(       32929) } },
    { { UINT16_C(22149), UINT16_C(60092), UINT16_C(63164), UINT16_C(10216), UINT16_C(26273), UINT16_C(12923), UINT16_C(25261), UINT16_C(43058),
        UINT16_C( 7142), UINT16_C(53376), UINT16_C(22207), UINT16_C( 1485), UINT16_C(54840), UINT16_C(59366), UINT16_C(34735), UINT16_C(13415) },
      {  INT32_C(       22149),  INT32_C(       60092),  INT32_C(       63164),  INT32_C(       10216),  INT32_C(       26273),  INT32_C(       12923),  INT32_C(       25261),  INT32_C(       43058),
         INT32_C(        7142),  INT32_C(       53376),  INT32_C(       22207),  INT32_C(        1485),  INT32_C(       54840),  INT32_C(       59366),  INT32_C(       34735),  INT32_C(       13415) } },
    { { UINT16_C( 9181), UINT16_C(39454), UINT16_C( 1561), UINT16_C(47809), UINT16_C(15724), UINT16_C( 6637), UINT16_C( 8095), UINT16_C(34242),
        UINT16_C(16955), UINT16_C(64086), UINT16_C( 9113), UINT16_C(53759), UINT16_C(58874), UINT16_C(43448), UINT16_C( 8045), UINT16_C(19165) },
      {  INT32_C(        9181),  INT32_C(       39454),  INT32_C(        1561),  INT32_C(       47809),  INT32_C(       15724),  INT32_C(        6637),  INT32_C(        8095),  INT32_C(       34242),
         INT32_C(       16955),  INT32_C(       64086),  INT32_C(        9113),  INT32_C(       53759),  INT32_C(       58874),  INT32_C(       43448),  INT32_C(        8045),  INT32_C(       19165) } },
    { { UINT16_C(64322), UINT16_C(23780), UINT16_C(42497), UINT16_C(28182), UINT16_C(  995), UINT16_C(33415), UINT16_C(18723), UINT16_C(24072),
        UINT16_C(24204), UINT16_C( 9560), UINT16_C(22401), UINT16_C(31734), UINT16_C(44860), UINT16_C(43300), UINT16_C(  462), UINT16_C( 4596) },
      {  INT32_C(       64322),  INT32_C(       23780),  INT32_C(       42497),  INT32_C(       28182),  INT32_C(         995),  INT32_C(       33415),  INT32_C(       18723),  INT32_C(       24072),
         INT32_C(       24204),  INT32_C(        9560),  INT32_C(       22401),  INT32_C(       31734),  INT32_C(       44860),  INT32_C(       43300),  INT32_C(         462),  INT32_C(        4596) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_cvtepu16_epi32(a);
    } EASYSIMD_TEST_PERF_END("_mm512_cvtepu16_epi32");
    easysimd_assert_m512i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m512i r = easysimd_mm512_cvtepu16_epi32(a);

    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu32_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t a[8];
    int64_t r[8];
  } test_vec[8] = {
    { { UINT32_C( 335012691), UINT32_C( 223973060), UINT32_C(3265431271), UINT32_C(1572645041), UINT32_C(2025214931), UINT32_C(3097790117), UINT32_C( 103967032), UINT32_C( 215032505) },
      {  INT64_C(           335012691),  INT64_C(           223973060),  INT64_C(          3265431271),  INT64_C(          1572645041),
         INT64_C(          2025214931),  INT64_C(          3097790117),  INT64_C(           103967032),  INT64_C(           215032505) } },
    { { UINT32_C(3391145989), UINT32_C(1054308695), UINT32_C(2952821503), UINT32_C(4111318050), UINT32_C(3043869455), UINT32_C(2188186186), UINT32_C( 881369211), UINT32_C(3342949058) },
      {  INT64_C(          3391145989),  INT64_C(          1054308695),  INT64_C(          2952821503),  INT64_C(          4111318050),
         INT64_C(          3043869455),  INT64_C(          2188186186),  INT64_C(           881369211),  INT64_C(          3342949058) } },
    { { UINT32_C(2039570722), UINT32_C(3652676058), UINT32_C(  92911587), UINT32_C(2214238068), UINT32_C(2755159898), UINT32_C(4096239225), UINT32_C( 136949574), UINT32_C( 751790601) },
      {  INT64_C(          2039570722),  INT64_C(          3652676058),  INT64_C(            92911587),  INT64_C(          2214238068),
         INT64_C(          2755159898),  INT64_C(          4096239225),  INT64_C(           136949574),  INT64_C(           751790601) } },
    { { UINT32_C(2779079115), UINT32_C(2910739914), UINT32_C(2293368596), UINT32_C(4178357406), UINT32_C(2375894035), UINT32_C( 813810922), UINT32_C(2100865652), UINT32_C(3752396820) },
      {  INT64_C(          2779079115),  INT64_C(          2910739914),  INT64_C(          2293368596),  INT64_C(          4178357406),
         INT64_C(          2375894035),  INT64_C(           813810922),  INT64_C(          2100865652),  INT64_C(          3752396820) } },
    { { UINT32_C( 864309097), UINT32_C(3235906220), UINT32_C(2823393802), UINT32_C(1369527614), UINT32_C(2229157785), UINT32_C(2008309763), UINT32_C( 536210698), UINT32_C(1593745141) },
      {  INT64_C(           864309097),  INT64_C(          3235906220),  INT64_C(          2823393802),  INT64_C(          1369527614),
         INT64_C(          2229157785),  INT64_C(          2008309763),  INT64_C(           536210698),  INT64_C(          1593745141) } },
    { { UINT32_C(2576450541), UINT32_C(2405069189), UINT32_C(1094230787), UINT32_C(2442320376), UINT32_C( 471167256), UINT32_C(3683896017), UINT32_C(2902100151), UINT32_C( 336263463) },
      {  INT64_C(          2576450541),  INT64_C(          2405069189),  INT64_C(          1094230787),  INT64_C(          2442320376),
         INT64_C(           471167256),  INT64_C(          3683896017),  INT64_C(          2902100151),  INT64_C(           336263463) } },
    { { UINT32_C(  28220284), UINT32_C( 261163020), UINT32_C(2739980715), UINT32_C(3140805282), UINT32_C( 618089043), UINT32_C(3405802004), UINT32_C( 444070643), UINT32_C(1865318899) },
      {  INT64_C(            28220284),  INT64_C(           261163020),  INT64_C(          2739980715),  INT64_C(          3140805282),
         INT64_C(           618089043),  INT64_C(          3405802004),  INT64_C(           444070643),  INT64_C(          1865318899) } },
    { { UINT32_C( 678550556), UINT32_C(2402747108), UINT32_C(1832028107), UINT32_C(3173541737), UINT32_C(3319922609), UINT32_C(1569775978), UINT32_C(3480684508), UINT32_C(2755634568) },
      {  INT64_C(           678550556),  INT64_C(          2402747108),  INT64_C(          1832028107),  INT64_C(          3173541737),
         INT64_C(          3319922609),  INT64_C(          1569775978),  INT64_C(          3480684508),  INT64_C(          2755634568) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_cvtepu32_epi64(a);
    } EASYSIMD_TEST_PERF_END("_mm512_cvtepu32_epi64");
    easysimd_assert_m512i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m512i r = easysimd_mm512_cvtepu32_epi64(a);

    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu8_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[32];
    int16_t r[32];
  } test_vec[8] = {
    { { UINT8_C(107), UINT8_C(143), UINT8_C(231), UINT8_C(  7), UINT8_C(109), UINT8_C(175), UINT8_C( 97), UINT8_C( 73),
        UINT8_C(197), UINT8_C(145), UINT8_C( 29), UINT8_C( 66), UINT8_C(183), UINT8_C( 16), UINT8_C( 43), UINT8_C(117),
        UINT8_C(176), UINT8_C(249), UINT8_C(111), UINT8_C(170), UINT8_C(168), UINT8_C( 26), UINT8_C( 76), UINT8_C(210),
        UINT8_C(  1), UINT8_C( 98), UINT8_C(160), UINT8_C(249), UINT8_C(106), UINT8_C( 23), UINT8_C(202), UINT8_C(214) },
      {  INT16_C(   107),  INT16_C(   143),  INT16_C(   231),  INT16_C(     7),  INT16_C(   109),  INT16_C(   175),  INT16_C(    97),  INT16_C(    73),
         INT16_C(   197),  INT16_C(   145),  INT16_C(    29),  INT16_C(    66),  INT16_C(   183),  INT16_C(    16),  INT16_C(    43),  INT16_C(   117),
         INT16_C(   176),  INT16_C(   249),  INT16_C(   111),  INT16_C(   170),  INT16_C(   168),  INT16_C(    26),  INT16_C(    76),  INT16_C(   210),
         INT16_C(     1),  INT16_C(    98),  INT16_C(   160),  INT16_C(   249),  INT16_C(   106),  INT16_C(    23),  INT16_C(   202),  INT16_C(   214) } },
    { { UINT8_C(166), UINT8_C(177), UINT8_C(221), UINT8_C( 20), UINT8_C( 96), UINT8_C( 62), UINT8_C( 93), UINT8_C( 37),
        UINT8_C(207), UINT8_C(122), UINT8_C(103), UINT8_C(135), UINT8_C(138), UINT8_C(147), UINT8_C(252), UINT8_C( 58),
        UINT8_C(140), UINT8_C(108), UINT8_C(228), UINT8_C( 53), UINT8_C(134), UINT8_C( 49), UINT8_C(  7), UINT8_C(135),
        UINT8_C(147), UINT8_C(167), UINT8_C(128), UINT8_C(253), UINT8_C(190), UINT8_C( 74), UINT8_C(211), UINT8_C(101) },
      {  INT16_C(   166),  INT16_C(   177),  INT16_C(   221),  INT16_C(    20),  INT16_C(    96),  INT16_C(    62),  INT16_C(    93),  INT16_C(    37),
         INT16_C(   207),  INT16_C(   122),  INT16_C(   103),  INT16_C(   135),  INT16_C(   138),  INT16_C(   147),  INT16_C(   252),  INT16_C(    58),
         INT16_C(   140),  INT16_C(   108),  INT16_C(   228),  INT16_C(    53),  INT16_C(   134),  INT16_C(    49),  INT16_C(     7),  INT16_C(   135),
         INT16_C(   147),  INT16_C(   167),  INT16_C(   128),  INT16_C(   253),  INT16_C(   190),  INT16_C(    74),  INT16_C(   211),  INT16_C(   101) } },
    { { UINT8_C(252), UINT8_C(176), UINT8_C(121), UINT8_C( 92), UINT8_C(238), UINT8_C(214), UINT8_C(130), UINT8_C(190),
        UINT8_C( 80), UINT8_C(233), UINT8_C( 69), UINT8_C(218), UINT8_C(124), UINT8_C( 65), UINT8_C( 20), UINT8_C(  9),
        UINT8_C(173), UINT8_C(249), UINT8_C( 62), UINT8_C( 51), UINT8_C( 42), UINT8_C( 69), UINT8_C(186), UINT8_C(189),
        UINT8_C(236), UINT8_C( 58), UINT8_C(186), UINT8_C(171), UINT8_C(133), UINT8_C(142), UINT8_C( 16), UINT8_C(129) },
      {  INT16_C(   252),  INT16_C(   176),  INT16_C(   121),  INT16_C(    92),  INT16_C(   238),  INT16_C(   214),  INT16_C(   130),  INT16_C(   190),
         INT16_C(    80),  INT16_C(   233),  INT16_C(    69),  INT16_C(   218),  INT16_C(   124),  INT16_C(    65),  INT16_C(    20),  INT16_C(     9),
         INT16_C(   173),  INT16_C(   249),  INT16_C(    62),  INT16_C(    51),  INT16_C(    42),  INT16_C(    69),  INT16_C(   186),  INT16_C(   189),
         INT16_C(   236),  INT16_C(    58),  INT16_C(   186),  INT16_C(   171),  INT16_C(   133),  INT16_C(   142),  INT16_C(    16),  INT16_C(   129) } },
    { { UINT8_C( 62), UINT8_C(137), UINT8_C(221), UINT8_C( 45), UINT8_C( 95), UINT8_C( 95), UINT8_C(235), UINT8_C(175),
        UINT8_C( 73), UINT8_C( 48), UINT8_C(137), UINT8_C(197), UINT8_C(113), UINT8_C(157), UINT8_C(206), UINT8_C( 31),
        UINT8_C(150), UINT8_C( 12), UINT8_C( 82), UINT8_C(192), UINT8_C( 81), UINT8_C( 13), UINT8_C(125), UINT8_C( 62),
        UINT8_C( 71), UINT8_C( 56), UINT8_C(233), UINT8_C(204), UINT8_C(198), UINT8_C(249), UINT8_C( 77), UINT8_C(  4) },
      {  INT16_C(    62),  INT16_C(   137),  INT16_C(   221),  INT16_C(    45),  INT16_C(    95),  INT16_C(    95),  INT16_C(   235),  INT16_C(   175),
         INT16_C(    73),  INT16_C(    48),  INT16_C(   137),  INT16_C(   197),  INT16_C(   113),  INT16_C(   157),  INT16_C(   206),  INT16_C(    31),
         INT16_C(   150),  INT16_C(    12),  INT16_C(    82),  INT16_C(   192),  INT16_C(    81),  INT16_C(    13),  INT16_C(   125),  INT16_C(    62),
         INT16_C(    71),  INT16_C(    56),  INT16_C(   233),  INT16_C(   204),  INT16_C(   198),  INT16_C(   249),  INT16_C(    77),  INT16_C(     4) } },
    { { UINT8_C(130), UINT8_C( 43), UINT8_C( 49), UINT8_C(225), UINT8_C(138), UINT8_C( 28), UINT8_C(144), UINT8_C(211),
        UINT8_C( 76), UINT8_C( 25), UINT8_C(153), UINT8_C(190), UINT8_C(182), UINT8_C(103), UINT8_C(221), UINT8_C( 77),
        UINT8_C(116), UINT8_C( 47), UINT8_C( 13), UINT8_C(197), UINT8_C( 60), UINT8_C(139), UINT8_C(  3), UINT8_C(132),
        UINT8_C(195), UINT8_C(236), UINT8_C( 80), UINT8_C(137), UINT8_C(229), UINT8_C(158), UINT8_C(141), UINT8_C(103) },
      {  INT16_C(   130),  INT16_C(    43),  INT16_C(    49),  INT16_C(   225),  INT16_C(   138),  INT16_C(    28),  INT16_C(   144),  INT16_C(   211),
         INT16_C(    76),  INT16_C(    25),  INT16_C(   153),  INT16_C(   190),  INT16_C(   182),  INT16_C(   103),  INT16_C(   221),  INT16_C(    77),
         INT16_C(   116),  INT16_C(    47),  INT16_C(    13),  INT16_C(   197),  INT16_C(    60),  INT16_C(   139),  INT16_C(     3),  INT16_C(   132),
         INT16_C(   195),  INT16_C(   236),  INT16_C(    80),  INT16_C(   137),  INT16_C(   229),  INT16_C(   158),  INT16_C(   141),  INT16_C(   103) } },
    { { UINT8_C(201), UINT8_C(191), UINT8_C( 72), UINT8_C( 83), UINT8_C(219), UINT8_C(216), UINT8_C( 39), UINT8_C( 40),
        UINT8_C(241), UINT8_C(192), UINT8_C(230), UINT8_C(168), UINT8_C( 39), UINT8_C(195), UINT8_C(245), UINT8_C(155),
        UINT8_C(242), UINT8_C(  2), UINT8_C( 97), UINT8_C( 47), UINT8_C(141), UINT8_C(100), UINT8_C(179), UINT8_C( 80),
        UINT8_C( 81), UINT8_C(  3), UINT8_C(217), UINT8_C( 54), UINT8_C(161), UINT8_C(103), UINT8_C(158), UINT8_C(106) },
      {  INT16_C(   201),  INT16_C(   191),  INT16_C(    72),  INT16_C(    83),  INT16_C(   219),  INT16_C(   216),  INT16_C(    39),  INT16_C(    40),
         INT16_C(   241),  INT16_C(   192),  INT16_C(   230),  INT16_C(   168),  INT16_C(    39),  INT16_C(   195),  INT16_C(   245),  INT16_C(   155),
         INT16_C(   242),  INT16_C(     2),  INT16_C(    97),  INT16_C(    47),  INT16_C(   141),  INT16_C(   100),  INT16_C(   179),  INT16_C(    80),
         INT16_C(    81),  INT16_C(     3),  INT16_C(   217),  INT16_C(    54),  INT16_C(   161),  INT16_C(   103),  INT16_C(   158),  INT16_C(   106) } },
    { { UINT8_C( 38), UINT8_C(230), UINT8_C(190), UINT8_C(  1), UINT8_C(191), UINT8_C(229), UINT8_C( 41), UINT8_C(176),
        UINT8_C(165), UINT8_C( 15), UINT8_C( 88), UINT8_C(204), UINT8_C(210), UINT8_C( 77), UINT8_C(104), UINT8_C(197),
        UINT8_C( 80), UINT8_C(201), UINT8_C(244), UINT8_C(221), UINT8_C( 45), UINT8_C(167), UINT8_C( 46), UINT8_C(126),
        UINT8_C(170), UINT8_C(  7), UINT8_C(181), UINT8_C( 76), UINT8_C(110), UINT8_C( 83), UINT8_C(182), UINT8_C(148) },
      {  INT16_C(    38),  INT16_C(   230),  INT16_C(   190),  INT16_C(     1),  INT16_C(   191),  INT16_C(   229),  INT16_C(    41),  INT16_C(   176),
         INT16_C(   165),  INT16_C(    15),  INT16_C(    88),  INT16_C(   204),  INT16_C(   210),  INT16_C(    77),  INT16_C(   104),  INT16_C(   197),
         INT16_C(    80),  INT16_C(   201),  INT16_C(   244),  INT16_C(   221),  INT16_C(    45),  INT16_C(   167),  INT16_C(    46),  INT16_C(   126),
         INT16_C(   170),  INT16_C(     7),  INT16_C(   181),  INT16_C(    76),  INT16_C(   110),  INT16_C(    83),  INT16_C(   182),  INT16_C(   148) } },
    { { UINT8_C( 57), UINT8_C(116), UINT8_C(150), UINT8_C(248), UINT8_C( 89), UINT8_C(191), UINT8_C(169), UINT8_C(254),
        UINT8_C(207), UINT8_C(  1), UINT8_C(203), UINT8_C(161), UINT8_C( 79), UINT8_C( 51), UINT8_C(102), UINT8_C(159),
        UINT8_C(252), UINT8_C( 90), UINT8_C(124), UINT8_C( 41), UINT8_C(  1), UINT8_C(170), UINT8_C(168), UINT8_C(172),
        UINT8_C(178), UINT8_C( 93), UINT8_C(248), UINT8_C( 32), UINT8_C(176), UINT8_C(174), UINT8_C(181), UINT8_C(233) },
      {  INT16_C(    57),  INT16_C(   116),  INT16_C(   150),  INT16_C(   248),  INT16_C(    89),  INT16_C(   191),  INT16_C(   169),  INT16_C(   254),
         INT16_C(   207),  INT16_C(     1),  INT16_C(   203),  INT16_C(   161),  INT16_C(    79),  INT16_C(    51),  INT16_C(   102),  INT16_C(   159),
         INT16_C(   252),  INT16_C(    90),  INT16_C(   124),  INT16_C(    41),  INT16_C(     1),  INT16_C(   170),  INT16_C(   168),  INT16_C(   172),
         INT16_C(   178),  INT16_C(    93),  INT16_C(   248),  INT16_C(    32),  INT16_C(   176),  INT16_C(   174),  INT16_C(   181),  INT16_C(   233) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_cvtepu8_epi16(a);
    } EASYSIMD_TEST_PERF_END("_mm512_cvtepu8_epi16");
    easysimd_assert_m512i_i16(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m512i r = easysimd_mm512_cvtepu8_epi16(a);

    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu16_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[8];
    int64_t r[8];
  } test_vec[8] = {
    { { UINT16_C( 5372), UINT16_C(45634), UINT16_C(17667), UINT16_C(57887), UINT16_C(37285), UINT16_C(25143), UINT16_C(47233), UINT16_C(23802) },
      {  INT64_C(                5372),  INT64_C(               45634),  INT64_C(               17667),  INT64_C(               57887),
         INT64_C(               37285),  INT64_C(               25143),  INT64_C(               47233),  INT64_C(               23802) } },
    { { UINT16_C( 4741), UINT16_C(17198), UINT16_C(41224), UINT16_C(32100), UINT16_C(63425), UINT16_C(36970), UINT16_C(43129), UINT16_C(30011) },
      {  INT64_C(                4741),  INT64_C(               17198),  INT64_C(               41224),  INT64_C(               32100),
         INT64_C(               63425),  INT64_C(               36970),  INT64_C(               43129),  INT64_C(               30011) } },
    { { UINT16_C(32189), UINT16_C(49191), UINT16_C(18114), UINT16_C(26530), UINT16_C(56024), UINT16_C(22985), UINT16_C(50322), UINT16_C( 6325) },
      {  INT64_C(               32189),  INT64_C(               49191),  INT64_C(               18114),  INT64_C(               26530),
         INT64_C(               56024),  INT64_C(               22985),  INT64_C(               50322),  INT64_C(                6325) } },
    { { UINT16_C(58326), UINT16_C(56923), UINT16_C(49028), UINT16_C(18012), UINT16_C(50870), UINT16_C(12246), UINT16_C( 4462), UINT16_C(11172) },
      {  INT64_C(               58326),  INT64_C(               56923),  INT64_C(               49028),  INT64_C(               18012),
         INT64_C(               50870),  INT64_C(               12246),  INT64_C(                4462),  INT64_C(               11172) } },
    { { UINT16_C(52110), UINT16_C(20716), UINT16_C(36370), UINT16_C(60087), UINT16_C(33128), UINT16_C(64323), UINT16_C(63557), UINT16_C( 6931) },
      {  INT64_C(               52110),  INT64_C(               20716),  INT64_C(               36370),  INT64_C(               60087),
         INT64_C(               33128),  INT64_C(               64323),  INT64_C(               63557),  INT64_C(                6931) } },
    { { UINT16_C(28380), UINT16_C(24825), UINT16_C(21805), UINT16_C(58534), UINT16_C(31771), UINT16_C(35347), UINT16_C(47245), UINT16_C( 7093) },
      {  INT64_C(               28380),  INT64_C(               24825),  INT64_C(               21805),  INT64_C(               58534),
         INT64_C(               31771),  INT64_C(               35347),  INT64_C(               47245),  INT64_C(                7093) } },
    { { UINT16_C(41347), UINT16_C(38251), UINT16_C( 9008), UINT16_C(39039), UINT16_C(49828), UINT16_C(59795), UINT16_C(42683), UINT16_C(38660) },
      {  INT64_C(               41347),  INT64_C(               38251),  INT64_C(                9008),  INT64_C(               39039),
         INT64_C(               49828),  INT64_C(               59795),  INT64_C(               42683),  INT64_C(               38660) } },
    { { UINT16_C(64788), UINT16_C(17143), UINT16_C(40531), UINT16_C(28198), UINT16_C(14618), UINT16_C(43256), UINT16_C(44785), UINT16_C(30147) },
      {  INT64_C(               64788),  INT64_C(               17143),  INT64_C(               40531),  INT64_C(               28198),
         INT64_C(               14618),  INT64_C(               43256),  INT64_C(               44785),  INT64_C(               30147) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_cvtepu16_epi64(a);
    } EASYSIMD_TEST_PERF_END("_mm512_cvtepu16_epi64");
    easysimd_assert_m512i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m512i r = easysimd_mm512_cvtepu16_epi64(a);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu8_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[16];
    int32_t r[16];
  } test_vec[8] = {
    { { UINT8_C(241), UINT8_C( 90), UINT8_C(173), UINT8_C( 20), UINT8_C( 92), UINT8_C(  3), UINT8_C( 96), UINT8_C( 82),
        UINT8_C(190), UINT8_C(101), UINT8_C( 19), UINT8_C(230), UINT8_C(245), UINT8_C(141), UINT8_C( 23), UINT8_C( 41) },
      {  INT32_C(         241),  INT32_C(          90),  INT32_C(         173),  INT32_C(          20),  INT32_C(          92),  INT32_C(           3),  INT32_C(          96),  INT32_C(          82),
         INT32_C(         190),  INT32_C(         101),  INT32_C(          19),  INT32_C(         230),  INT32_C(         245),  INT32_C(         141),  INT32_C(          23),  INT32_C(          41) } },
    { { UINT8_C( 58), UINT8_C( 84), UINT8_C(  1), UINT8_C( 78), UINT8_C( 99), UINT8_C( 10), UINT8_C( 17), UINT8_C( 25),
        UINT8_C(216), UINT8_C( 31), UINT8_C(238), UINT8_C( 47), UINT8_C(194), UINT8_C( 61), UINT8_C(130), UINT8_C(180) },
      {  INT32_C(          58),  INT32_C(          84),  INT32_C(           1),  INT32_C(          78),  INT32_C(          99),  INT32_C(          10),  INT32_C(          17),  INT32_C(          25),
         INT32_C(         216),  INT32_C(          31),  INT32_C(         238),  INT32_C(          47),  INT32_C(         194),  INT32_C(          61),  INT32_C(         130),  INT32_C(         180) } },
    { { UINT8_C(151), UINT8_C( 47), UINT8_C(200), UINT8_C(244), UINT8_C( 50), UINT8_C( 40), UINT8_C( 70), UINT8_C(240),
        UINT8_C(141), UINT8_C( 89), UINT8_C(214), UINT8_C(130), UINT8_C(230), UINT8_C(237), UINT8_C(172), UINT8_C( 33) },
      {  INT32_C(         151),  INT32_C(          47),  INT32_C(         200),  INT32_C(         244),  INT32_C(          50),  INT32_C(          40),  INT32_C(          70),  INT32_C(         240),
         INT32_C(         141),  INT32_C(          89),  INT32_C(         214),  INT32_C(         130),  INT32_C(         230),  INT32_C(         237),  INT32_C(         172),  INT32_C(          33) } },
    { { UINT8_C( 65), UINT8_C(173), UINT8_C(111), UINT8_C(165), UINT8_C(184), UINT8_C(128), UINT8_C(190), UINT8_C(144),
        UINT8_C(159), UINT8_C(172), UINT8_C(192), UINT8_C( 98), UINT8_C(234), UINT8_C( 66), UINT8_C( 22), UINT8_C(129) },
      {  INT32_C(          65),  INT32_C(         173),  INT32_C(         111),  INT32_C(         165),  INT32_C(         184),  INT32_C(         128),  INT32_C(         190),  INT32_C(         144),
         INT32_C(         159),  INT32_C(         172),  INT32_C(         192),  INT32_C(          98),  INT32_C(         234),  INT32_C(          66),  INT32_C(          22),  INT32_C(         129) } },
    { { UINT8_C(113), UINT8_C(222), UINT8_C(117), UINT8_C(163), UINT8_C(  6), UINT8_C(187), UINT8_C(148), UINT8_C(147),
        UINT8_C( 21), UINT8_C(106), UINT8_C( 22), UINT8_C(251), UINT8_C( 88), UINT8_C(194), UINT8_C( 28), UINT8_C(153) },
      {  INT32_C(         113),  INT32_C(         222),  INT32_C(         117),  INT32_C(         163),  INT32_C(           6),  INT32_C(         187),  INT32_C(         148),  INT32_C(         147),
         INT32_C(          21),  INT32_C(         106),  INT32_C(          22),  INT32_C(         251),  INT32_C(          88),  INT32_C(         194),  INT32_C(          28),  INT32_C(         153) } },
    { { UINT8_C(111), UINT8_C(139), UINT8_C( 62), UINT8_C( 39), UINT8_C( 12), UINT8_C(252), UINT8_C(184), UINT8_C(171),
        UINT8_C(169), UINT8_C(120), UINT8_C( 13), UINT8_C(147), UINT8_C(186), UINT8_C( 35), UINT8_C( 20), UINT8_C( 43) },
      {  INT32_C(         111),  INT32_C(         139),  INT32_C(          62),  INT32_C(          39),  INT32_C(          12),  INT32_C(         252),  INT32_C(         184),  INT32_C(         171),
         INT32_C(         169),  INT32_C(         120),  INT32_C(          13),  INT32_C(         147),  INT32_C(         186),  INT32_C(          35),  INT32_C(          20),  INT32_C(          43) } },
    { { UINT8_C(  1), UINT8_C(138), UINT8_C(206), UINT8_C(  7), UINT8_C( 69), UINT8_C( 98), UINT8_C(155), UINT8_C( 90),
        UINT8_C(205), UINT8_C(177), UINT8_C( 86), UINT8_C( 37), UINT8_C(115), UINT8_C(114), UINT8_C(190), UINT8_C(226) },
      {  INT32_C(           1),  INT32_C(         138),  INT32_C(         206),  INT32_C(           7),  INT32_C(          69),  INT32_C(          98),  INT32_C(         155),  INT32_C(          90),
         INT32_C(         205),  INT32_C(         177),  INT32_C(          86),  INT32_C(          37),  INT32_C(         115),  INT32_C(         114),  INT32_C(         190),  INT32_C(         226) } },
    { { UINT8_C(254), UINT8_C(253), UINT8_C( 10), UINT8_C( 10), UINT8_C(249), UINT8_C(194), UINT8_C(181), UINT8_C(162),
        UINT8_C( 58), UINT8_C(195), UINT8_C( 53), UINT8_C(244), UINT8_C(230), UINT8_C( 74), UINT8_C( 31), UINT8_C(232) },
      {  INT32_C(         254),  INT32_C(         253),  INT32_C(          10),  INT32_C(          10),  INT32_C(         249),  INT32_C(         194),  INT32_C(         181),  INT32_C(         162),
         INT32_C(          58),  INT32_C(         195),  INT32_C(          53),  INT32_C(         244),  INT32_C(         230),  INT32_C(          74),  INT32_C(          31),  INT32_C(         232) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_cvtepu8_epi32(a);
    } EASYSIMD_TEST_PERF_END("_mm512_cvtepu8_epi32");
    easysimd_assert_m512i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m512i r = easysimd_mm512_cvtepu8_epi32(a);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvtepu8_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[16];
    int64_t r[8];
  } test_vec[8] = {
    { { UINT8_C(212), UINT8_C( 53), UINT8_C( 34), UINT8_C( 35), UINT8_C( 18), UINT8_C(154), UINT8_C(101), UINT8_C(126),
        UINT8_C(197), UINT8_C(207), UINT8_C(  0), UINT8_C(218), UINT8_C(215), UINT8_C(212), UINT8_C(130), UINT8_C(103) },
      {  INT64_C(                 212),  INT64_C(                  53),  INT64_C(                  34),  INT64_C(                  35),
         INT64_C(                  18),  INT64_C(                 154),  INT64_C(                 101),  INT64_C(                 126) } },
    { { UINT8_C( 75), UINT8_C(238), UINT8_C(108), UINT8_C(126), UINT8_C(108), UINT8_C(189), UINT8_C( 43), UINT8_C( 25),
        UINT8_C(202), UINT8_C(186), UINT8_C( 11), UINT8_C(180), UINT8_C(  9), UINT8_C( 51), UINT8_C( 91), UINT8_C(222) },
      {  INT64_C(                  75),  INT64_C(                 238),  INT64_C(                 108),  INT64_C(                 126),
         INT64_C(                 108),  INT64_C(                 189),  INT64_C(                  43),  INT64_C(                  25) } },
    { { UINT8_C(105), UINT8_C(125), UINT8_C(  1), UINT8_C(123), UINT8_C( 24), UINT8_C(102), UINT8_C(250), UINT8_C(221),
        UINT8_C( 54), UINT8_C(250), UINT8_C(183), UINT8_C( 13), UINT8_C(207), UINT8_C( 57), UINT8_C(116), UINT8_C( 26) },
      {  INT64_C(                 105),  INT64_C(                 125),  INT64_C(                   1),  INT64_C(                 123),
         INT64_C(                  24),  INT64_C(                 102),  INT64_C(                 250),  INT64_C(                 221) } },
    { { UINT8_C( 39), UINT8_C(224), UINT8_C(152), UINT8_C(147), UINT8_C(158), UINT8_C(195), UINT8_C(172), UINT8_C(104),
        UINT8_C(125), UINT8_C(184), UINT8_C( 28), UINT8_C(135), UINT8_C(235), UINT8_C(120), UINT8_C(101), UINT8_C( 84) },
      {  INT64_C(                  39),  INT64_C(                 224),  INT64_C(                 152),  INT64_C(                 147),
         INT64_C(                 158),  INT64_C(                 195),  INT64_C(                 172),  INT64_C(                 104) } },
    { { UINT8_C(245), UINT8_C(102), UINT8_C(208), UINT8_C( 13), UINT8_C(205), UINT8_C(202), UINT8_C(235), UINT8_C(  3),
        UINT8_C(196), UINT8_C(162), UINT8_C( 16), UINT8_C(147), UINT8_C(220), UINT8_C(132), UINT8_C(173), UINT8_C(  3) },
      {  INT64_C(                 245),  INT64_C(                 102),  INT64_C(                 208),  INT64_C(                  13),
         INT64_C(                 205),  INT64_C(                 202),  INT64_C(                 235),  INT64_C(                   3) } },
    { { UINT8_C(101), UINT8_C( 69), UINT8_C(151), UINT8_C(  3), UINT8_C(  9), UINT8_C( 67), UINT8_C(107), UINT8_C(134),
        UINT8_C(251), UINT8_C(136), UINT8_C( 13), UINT8_C(231), UINT8_C(  0), UINT8_C(114), UINT8_C( 59), UINT8_C(245) },
      {  INT64_C(                 101),  INT64_C(                  69),  INT64_C(                 151),  INT64_C(                   3),
         INT64_C(                   9),  INT64_C(                  67),  INT64_C(                 107),  INT64_C(                 134) } },
    { { UINT8_C(217), UINT8_C( 11), UINT8_C(  3), UINT8_C(166), UINT8_C(213), UINT8_C(238), UINT8_C(169), UINT8_C(154),
        UINT8_C(144), UINT8_C(185), UINT8_C( 45), UINT8_C(108), UINT8_C( 61), UINT8_C(219), UINT8_C(112), UINT8_C(162) },
      {  INT64_C(                 217),  INT64_C(                  11),  INT64_C(                   3),  INT64_C(                 166),
         INT64_C(                 213),  INT64_C(                 238),  INT64_C(                 169),  INT64_C(                 154) } },
    { { UINT8_C( 32), UINT8_C(  7), UINT8_C(165), UINT8_C( 41), UINT8_C( 74), UINT8_C( 17), UINT8_C(176), UINT8_C( 70),
        UINT8_C(153), UINT8_C(189), UINT8_C( 45), UINT8_C(153), UINT8_C( 48), UINT8_C(104), UINT8_C(142), UINT8_C(  9) },
      {  INT64_C(                  32),  INT64_C(                   7),  INT64_C(                 165),  INT64_C(                  41),
         INT64_C(                  74),  INT64_C(                  17),  INT64_C(                 176),  INT64_C(                  70) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_cvtepu8_epi64(a);
    } EASYSIMD_TEST_PERF_END("_mm512_cvtepu8_epi64");
    easysimd_assert_m512i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m512i r = easysimd_mm512_cvtepu8_epi64(a);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepi16_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[64];
    easysimd__mmask32 k;
    int16_t a[32];
  } test_vec[8] = {
    { {  INT8_C(  23),  INT8_C(  80),  INT8_C(   0),  INT8_C(  83),  INT8_C(   0),  INT8_C(   0),  INT8_C(  84), -INT8_C( 112),
             INT8_MIN,  INT8_C(   0), -INT8_C( 127),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  77),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 113), -INT8_C(  71),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  50),  INT8_C( 107), -INT8_C(  89),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C(1891665355),
      { -INT16_C( 24297),  INT16_C( 24912),  INT16_C( 13109),  INT16_C( 16723),  INT16_C( 10926),  INT16_C( 21022),  INT16_C( 22612), -INT16_C( 21616),
         INT16_C( 16768),  INT16_C(   638),  INT16_C( 31105), -INT16_C(  8477),  INT16_C( 15735),  INT16_C( 17216),  INT16_C(   194), -INT16_C(  9549),
         INT16_C(   929), -INT16_C( 10693), -INT16_C( 28873), -INT16_C(  6888),  INT16_C( 14009),  INT16_C(  3639), -INT16_C( 14449),  INT16_C(  4025),
         INT16_C( 14344), -INT16_C( 30447), -INT16_C(  2639),  INT16_C( 10343), -INT16_C( 22734), -INT16_C(  2709),  INT16_C(  8103),  INT16_C( 18895) } },
    { {  INT8_C(   0),  INT8_C(  62),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  12),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  44),  INT8_C(   0),  INT8_C(  58),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  37),  INT8_C(   0), -INT8_C(  26), -INT8_C( 112),  INT8_C( 123),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 110),  INT8_C(   0),  INT8_C(   0), -INT8_C(  46),  INT8_C(  84),  INT8_C(   0), -INT8_C( 114),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C(1495206434),
      {  INT16_C( 14233),  INT16_C( 21310),  INT16_C( 30062), -INT16_C(   671),  INT16_C(  6717),  INT16_C( 17676),  INT16_C(  7506),  INT16_C(   975),
         INT16_C( 13842),  INT16_C( 17708), -INT16_C( 26658), -INT16_C( 31430),  INT16_C(  2486), -INT16_C(  9778), -INT16_C(  4589), -INT16_C( 21198),
         INT16_C( 28965), -INT16_C( 27904),  INT16_C( 25062),  INT16_C(  9104), -INT16_C( 25477), -INT16_C( 12695),  INT16_C( 14522), -INT16_C( 13103),
        -INT16_C(   658),  INT16_C( 19473),  INT16_C( 19349),  INT16_C( 19410), -INT16_C( 24492),  INT16_C( 26660),  INT16_C( 22414), -INT16_C( 19435) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  65),  INT8_C(   0),  INT8_C(   0), -INT8_C( 114),  INT8_C(  46),
         INT8_C(  88),  INT8_C(   0), -INT8_C(  96),  INT8_C(   0), -INT8_C(  22),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   2), -INT8_C(  90),  INT8_C( 110),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  83),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  79), -INT8_C(  85), -INT8_C(  64),  INT8_C(   0),  INT8_C(  72),  INT8_C(   0), -INT8_C(  30),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C(2923894216),
      { -INT16_C( 10122), -INT16_C(  3630),  INT16_C( 15220),  INT16_C( 11967), -INT16_C( 28301), -INT16_C(  7685),  INT16_C(  3214),  INT16_C(  9006),
         INT16_C(    88), -INT16_C( 21393), -INT16_C( 27744),  INT16_C( 12052),  INT16_C( 10730), -INT16_C( 19741),  INT16_C( 10814), -INT16_C( 19359),
         INT16_C( 13058),  INT16_C( 30630),  INT16_C( 25966), -INT16_C(  7771), -INT16_C( 24330), -INT16_C( 31294), -INT16_C(  3923),  INT16_C(  1448),
         INT16_C(  6128), -INT16_C( 28239), -INT16_C( 14677), -INT16_C( 27200), -INT16_C( 23569),  INT16_C( 11848), -INT16_C( 22067), -INT16_C( 12062) } },
    { {  INT8_C(   0),  INT8_C(   0), -INT8_C( 115),  INT8_C( 105), -INT8_C(  34),  INT8_C(   0),  INT8_C(  41),  INT8_C(  95),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  44),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 127),
        -INT8_C( 109),  INT8_C(  17), -INT8_C( 103),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 118),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  26),  INT8_C(   0),  INT8_C(  23),  INT8_C(   0),  INT8_C(   0),  INT8_C(  30),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C(1246202076),
      { -INT16_C(  4882), -INT16_C(  7125), -INT16_C(  4723),  INT16_C( 14953),  INT16_C(  4830), -INT16_C( 12737), -INT16_C(  4055), -INT16_C( 11169),
         INT16_C(  8118), -INT16_C( 22934), -INT16_C( 19774), -INT16_C( 28460), -INT16_C( 18853),  INT16_C( 14176), -INT16_C( 22721),  INT16_C( 11649),
        -INT16_C( 21357),  INT16_C(  8209),  INT16_C( 31641),  INT16_C( 30554), -INT16_C( 26227), -INT16_C( 18874), -INT16_C( 23158),  INT16_C( 16523),
        -INT16_C(  2619), -INT16_C( 30746), -INT16_C( 17753),  INT16_C(   535),  INT16_C( 30577), -INT16_C( 20423), -INT16_C( 17890), -INT16_C( 19747) } },
    { {  INT8_C(   0),  INT8_C( 119), -INT8_C(  58),  INT8_C(   0),  INT8_C(   0), -INT8_C( 111),  INT8_C(  45),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  42),  INT8_C(  62),  INT8_C(  83),  INT8_C(   0),  INT8_C(  14),  INT8_C(  30),  INT8_C(  46),
         INT8_C(   0),  INT8_C( 126),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0),  INT8_C(  45),  INT8_C( 100),
         INT8_C(  98),  INT8_C(   2),  INT8_C(  73), -INT8_C(   4), -INT8_C( 123),  INT8_C(  65), -INT8_C(  20),  INT8_C(  44),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C(4292013670),
      {  INT16_C( 11625), -INT16_C(  2441), -INT16_C( 16954),  INT16_C( 20653),  INT16_C( 14434),  INT16_C( 10129),  INT16_C( 30509), -INT16_C( 11089),
        -INT16_C( 14798), -INT16_C( 23594),  INT16_C(  3902),  INT16_C( 23635),  INT16_C( 12489),  INT16_C( 12046), -INT16_C(  7906), -INT16_C( 30674),
        -INT16_C( 23282), -INT16_C( 11138),  INT16_C( 11106), -INT16_C( 15067), -INT16_C( 18845), -INT16_C( 28436), -INT16_C( 25811),  INT16_C( 24420),
         INT16_C( 14946), -INT16_C( 24574),  INT16_C( 21833),  INT16_C(  4860),  INT16_C(  2949), -INT16_C( 23487),  INT16_C( 28908), -INT16_C(  1492) } },
    { { -INT8_C(  42),  INT8_C(   0), -INT8_C(  87),  INT8_C(   0), -INT8_C(  59),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  59),  INT8_C(   0), -INT8_C(  72),  INT8_C(   0), -INT8_C(  76),  INT8_C(   0),  INT8_C(   4),
         INT8_C(   0), -INT8_C(  98),  INT8_C( 106), -INT8_C(   9),  INT8_C(   0),  INT8_C(   0),  INT8_C( 102), -INT8_C(  78),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 106), -INT8_C(  91),      INT8_MIN, -INT8_C(  95),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C(2026809877),
      { -INT16_C(  3114),  INT16_C( 14653),  INT16_C( 10665), -INT16_C( 10294),  INT16_C( 11973),  INT16_C( 10038),  INT16_C( 14697), -INT16_C( 19769),
        -INT16_C( 15474),  INT16_C(  5317),  INT16_C(  1742), -INT16_C( 17736), -INT16_C(  7050), -INT16_C( 29516), -INT16_C( 31858),  INT16_C( 25604),
         INT16_C( 16758),  INT16_C(  8350),  INT16_C( 26730),  INT16_C( 12279),  INT16_C( 11670), -INT16_C(   170),  INT16_C(  7526), -INT16_C(  2638),
         INT16_C( 30689), -INT16_C( 20727), -INT16_C( 16003), -INT16_C(  2966),  INT16_C(  7845),  INT16_C( 13184), -INT16_C( 31583),  INT16_C(  6296) } },
    { { -INT8_C(  98),  INT8_C(   0),  INT8_C(  92),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  93),  INT8_C(  99),
         INT8_C(   0), -INT8_C(  50), -INT8_C(  20),  INT8_C(   0), -INT8_C(  46), -INT8_C(  91),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  44),  INT8_C(  21),  INT8_C(  98),  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 109),  INT8_C(  78),  INT8_C( 105),  INT8_C(  13),  INT8_C(   0), -INT8_C(  45),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C( 792213189),
      {  INT16_C( 12190),  INT16_C( 13407), -INT16_C( 19108), -INT16_C( 15564), -INT16_C(  6445), -INT16_C( 19272), -INT16_C( 16035), -INT16_C(  9629),
        -INT16_C( 12926),  INT16_C( 10190),  INT16_C( 20204), -INT16_C( 29350), -INT16_C(  3374), -INT16_C( 26715), -INT16_C(  8920), -INT16_C( 14649),
         INT16_C(  9740),  INT16_C( 27131),  INT16_C( 12251), -INT16_C( 20948), -INT16_C(  7147),  INT16_C( 29282), -INT16_C( 14683),  INT16_C( 10060),
         INT16_C(  7059),  INT16_C( 32590), -INT16_C( 22423),  INT16_C( 15373), -INT16_C( 19813), -INT16_C( 15405), -INT16_C( 25968), -INT16_C( 25462) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  94), -INT8_C( 122),
        -INT8_C(  40),  INT8_C(   0),  INT8_C(  69),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  18),
        -INT8_C(  87),  INT8_C(   0),  INT8_C(  10),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 105), -INT8_C(  73), -INT8_C(  52),  INT8_C(   0),  INT8_C(   0),  INT8_C(  11),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT32_C(2617607616),
      {  INT16_C( 12724), -INT16_C( 14006), -INT16_C( 21227), -INT16_C( 17861), -INT16_C( 30861),  INT16_C(  1761),  INT16_C( 12194),  INT16_C(  3206),
        -INT16_C( 27688),  INT16_C( 29512),  INT16_C(  6981), -INT16_C( 10954), -INT16_C( 16202),  INT16_C( 30322),  INT16_C( 30533), -INT16_C(  1774),
         INT16_C( 23977), -INT16_C( 16702), -INT16_C(   758),  INT16_C( 32121),  INT16_C( 23173),  INT16_C( 10115),  INT16_C(  2442),  INT16_C( 25139),
         INT16_C( 31644), -INT16_C(  7467),  INT16_C(  2967),  INT16_C( 19895),  INT16_C( 10700),  INT16_C(  4547), -INT16_C( 10591),  INT16_C( 18955) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    int8_t arr_r[64] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm512_mask_cvtepi16_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cvtepi16_storeu_epi8");
    easysimd_assert_equal_vi8(32, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m512i r = easysimd_mm512_cvtepu8_epi64(a);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

    int8_t arr_r[64] = {0};
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i  a = easysimd_test_x86_random_i16x32();
    easysimd_mm512_mask_cvtepi16_storeu_epi8((void *)arr_r, k, a);
    easysimd__m512i  r = easysimd_mm512_loadu_si512((void *)arr_r);

    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepi32_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[64];
    easysimd__mmask16 k;
    int32_t a[16];
  } test_vec[8] =  {
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(  43),  INT8_C(   0),  INT8_C(  81),  INT8_C( 113),  INT8_C(   0),  INT8_C( 125),
        -INT8_C(   9), -INT8_C( 104),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 104),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C( 9140),
      { -INT32_C(    22211589), -INT32_C(  1724187218), -INT32_C(   472270293), -INT32_C(   233371402),  INT32_C(  1900108369),  INT32_C(  1912669041), -INT32_C(   902823812),  INT32_C(  2045633661),
        -INT32_C(  1518888201), -INT32_C(  1019235944),  INT32_C(  1655052396),  INT32_C(  1918156320), -INT32_C(  1981573864),  INT32_C(   352052376),  INT32_C(  1658792933),  INT32_C(   115067919) } },
    { { -INT8_C(  84), -INT8_C(  62), -INT8_C(  46),  INT8_C(   0),  INT8_C(   0),  INT8_C( 104), -INT8_C(  91),  INT8_C(   0),
         INT8_C( 122),  INT8_C(  72),  INT8_C(   0),  INT8_C(   0),  INT8_C(  81),  INT8_C(   0), -INT8_C( 124),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(21351),
      { -INT32_C(   351993940),  INT32_C(  1761833154),  INT32_C(   640099538),  INT32_C(  2042314646), -INT32_C(  1017293624),  INT32_C(  1206862440), -INT32_C(  2146107739), -INT32_C(  1311540475),
         INT32_C(  1016911994),  INT32_C(   463839048),  INT32_C(  1497484483), -INT32_C(   741147381), -INT32_C(  1181274031),  INT32_C(   385975922), -INT32_C(  1986521724),  INT32_C(   171600784) } },
    { {  INT8_C(  70), -INT8_C(  89),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  62),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 125),  INT8_C(  79),  INT8_C(   0),  INT8_C(  25),  INT8_C(   0),  INT8_C(  27),  INT8_C( 121),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(54851),
      { -INT32_C(   327775162), -INT32_C(   390579801),  INT32_C(  1709556883),  INT32_C(   781530775), -INT32_C(   239859472), -INT32_C(  1224328929), -INT32_C(    47999294), -INT32_C(   422353248),
        -INT32_C(  1714271758),  INT32_C(   377654147), -INT32_C(   428120241), -INT32_C(  1911221858),  INT32_C(   947898649), -INT32_C(   974093054), -INT32_C(  1144909029),  INT32_C(  1822528889) } },
    { {  INT8_C(   5),  INT8_C( 120),  INT8_C(  52),  INT8_C(  27),  INT8_C(  86),  INT8_C(   0), -INT8_C(  11),  INT8_C(  62),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  71),  INT8_C(   0), -INT8_C(   2),  INT8_C(  85), -INT8_C(  56),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(29919),
      { -INT32_C(  2013306363), -INT32_C(   202420616),  INT32_C(  1225035060), -INT32_C(  1710023397),  INT32_C(  1176442198), -INT32_C(  1671873830), -INT32_C(  1775054091),  INT32_C(  1141510462),
        -INT32_C(   339015309), -INT32_C(  1931560361),  INT32_C(  1658184519),  INT32_C(  1459415040), -INT32_C(   660857858),  INT32_C(  1249244501),  INT32_C(   115451848),  INT32_C(   726330296) } },
    { {  INT8_C(  22),  INT8_C(   0),  INT8_C( 122),  INT8_C(   0),  INT8_C( 109),  INT8_C(  50), -INT8_C( 113),  INT8_C( 108),
         INT8_C(   0), -INT8_C(  40),  INT8_C(  38),  INT8_C(   0),  INT8_C(  68),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C( 5877),
      { -INT32_C(   187675626), -INT32_C(  1361635112),  INT32_C(  2006374522), -INT32_C(   879586258), -INT32_C(   490674067),  INT32_C(   327780402), -INT32_C(   637582961), -INT32_C(  2098203540),
         INT32_C(   427212864),  INT32_C(  1388793560),  INT32_C(  1422482726),  INT32_C(  1595890930),  INT32_C(  1984028740), -INT32_C(   141899160),  INT32_C(  2010220811), -INT32_C(  1107705731) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  40),
         INT8_C(   2), -INT8_C(  30),  INT8_C(  52),  INT8_C(   6),  INT8_C(   0), -INT8_C(  58),  INT8_C(  95),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(28544),
      { -INT32_C(  1648535338),  INT32_C(  1962664875),  INT32_C(  1473310008), -INT32_C(  1925770164),  INT32_C(   358064267),  INT32_C(  1755210391), -INT32_C(   769057831), -INT32_C(  1354651176),
        -INT32_C(  1387462910),  INT32_C(   438388706), -INT32_C(  2139950540), -INT32_C(  1861375738), -INT32_C(   509189047), -INT32_C(  1622588218),  INT32_C(   946958943),  INT32_C(   518501148) } },
    { {  INT8_C(   0), -INT8_C(  81),  INT8_C(   0),  INT8_C(   0), -INT8_C(  11),  INT8_C(   8),  INT8_C(   0), -INT8_C(  21),
         INT8_C( 115),  INT8_C(  62),  INT8_C(   0),  INT8_C(   0), -INT8_C(  74),  INT8_C(  28),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(13234),
      { -INT32_C(   327445301),  INT32_C(   568242095),  INT32_C(  1036707120),  INT32_C(   497161334),  INT32_C(  1046571253),  INT32_C(  2041692424), -INT32_C(   533934599), -INT32_C(  1240211733),
         INT32_C(   581078643),  INT32_C(  1849917758), -INT32_C(   592769690),  INT32_C(   385436961), -INT32_C(  1101702218),  INT32_C(   355927580), -INT32_C(  1091214125), -INT32_C(  1250686910) } },
    { { -INT8_C(  41),  INT8_C(  67), -INT8_C(  38),  INT8_C(   0), -INT8_C(  80),  INT8_C(   0),  INT8_C(   0),  INT8_C(  67),
         INT8_C(   0),  INT8_C( 107), -INT8_C(  89),  INT8_C(   0), -INT8_C( 110),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C( 5783),
      {  INT32_C(   446158295), -INT32_C(   299303613), -INT32_C(   751023654), -INT32_C(  1255214496), -INT32_C(   407156048),  INT32_C(  1414237791), -INT32_C(  1067610548),  INT32_C(   467072067),
         INT32_C(   204828361),  INT32_C(  1174035819),  INT32_C(   119092903), -INT32_C(   658749656), -INT32_C(   239044718),  INT32_C(  1363544837), -INT32_C(   586046567),  INT32_C(  1626924951) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    int8_t arr_r[64] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm512_mask_cvtepi32_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cvtepi32_storeu_epi8");
    easysimd_assert_equal_vi8(16, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[64] = {0};
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i  a = easysimd_test_x86_random_i32x16();
    easysimd_mm512_mask_cvtepi32_storeu_epi8((void *)arr_r, k, a);
    easysimd__m512i  r = easysimd_mm512_loadu_si512((void *)arr_r);

    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepi64_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[64];
    easysimd__mmask8 k;
    int64_t a[8];
  } test_vec[8] =  {
    { {  INT8_C(   0),  INT8_C(   0), -INT8_C(  41),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  41),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 68),
      {  INT64_C( 5389225249329858859), -INT64_C( 8360502807713615340),  INT64_C( 4518538441789727959), -INT64_C( 5790016668183610195),
        -INT64_C(  568558683260746749),  INT64_C(  976704103218065156),  INT64_C( 2295579560442552535), -INT64_C( 1371021828138455099) } },
    { {  INT8_C(   0),      INT8_MIN,  INT8_C(   0),  INT8_C(  89),  INT8_C(   0),  INT8_C( 120),  INT8_C(   0),  INT8_C( 108),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(170),
      { -INT64_C( 8478159496943845567),  INT64_C( 5137426193926085248), -INT64_C(  869237866033977184),  INT64_C( 8298383907625908569),
        -INT64_C( 4689058388007012473),  INT64_C( 1992699324671372664), -INT64_C( 7918116249232187290),  INT64_C( 5265335517621385580) } },
    { {  INT8_C(   0),  INT8_C(  31),  INT8_C(   0),  INT8_C(   0), -INT8_C(   7),      INT8_MIN, -INT8_C(  24),  INT8_C(  14),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(242),
      {  INT64_C( 5147594144640642265), -INT64_C( 3025332126946832865),  INT64_C( 4882360811656260897),  INT64_C( 5466892153655681202),
         INT64_C( 2007112278012464889), -INT64_C( 7364586637441328256),  INT64_C( 3275716148140563176), -INT64_C( 4553991587430237938) } },
    { {  INT8_C(   0), -INT8_C(  43), -INT8_C(  57),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(  6),
      { -INT64_C( 3945121191244442865), -INT64_C( 2560056687910111531), -INT64_C( 7456688712842761529), -INT64_C( 2189276280000674401),
        -INT64_C( 7896249692552322957),  INT64_C( 5545508412619664152),  INT64_C( 8213237314382613828), -INT64_C( 1225077525906433204) } },
    { {  INT8_C(   0), -INT8_C(  73),  INT8_C(  56),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  95),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 70),
      { -INT64_C( 6585505131221774431), -INT64_C(  606448100933258057),  INT64_C( 5750035646035654712), -INT64_C( 8417305859447920062),
        -INT64_C( 9158446114661321912), -INT64_C( 4684192741359284387),  INT64_C( 4945525242102400863),  INT64_C( 7883933940399828212) } },
    { {  INT8_C( 119), -INT8_C(  87),  INT8_C(  37), -INT8_C(  67),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  45),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(143),
      { -INT64_C( 6593876148691095945),  INT64_C( 5436099300334077353), -INT64_C( 3304920407998676187), -INT64_C( 2258163335562854211),
        -INT64_C( 8985949158212894445), -INT64_C( 9092482476754635517),  INT64_C( 7854227920690093767),  INT64_C( 7934550187400689619) } },
    { {  INT8_C(  29),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  17), -INT8_C(  32),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 49),
      { -INT64_C( 2981790452920216035),  INT64_C( 1507497469547848993),  INT64_C( 1083730441397278820), -INT64_C( 9145164445248764235),
        -INT64_C( 3591695824756924177),  INT64_C( 8353014089205897952), -INT64_C( 7917692233327137033),  INT64_C( 5827530792665254138) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(  77), -INT8_C(  21),  INT8_C(   0), -INT8_C(  18),  INT8_C(   0),  INT8_C(  92),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(172),
      {  INT64_C( 2492224286314045947),  INT64_C(  500079161171275035), -INT64_C( 6414123001836208307),  INT64_C( 3756943517990476523),
        -INT64_C( 4547648649791046151),  INT64_C(  309810178523993070), -INT64_C( 8039766853049503831), -INT64_C( 4194773420245289892) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    int8_t arr_r[64] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm512_mask_cvtepi64_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cvtepi64_storeu_epi8");
    easysimd_assert_equal_vi8(8, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[64] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i  a = easysimd_test_x86_random_i64x8();
    easysimd_mm512_mask_cvtepi64_storeu_epi8((void *)arr_r, k, a);
    easysimd__m512i  r = easysimd_mm512_loadu_si512((void *)arr_r);

    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepi32_storeu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t  r[32];
    easysimd__mmask16 k;
    int32_t a[16];
  } test_vec[8] =  {
    { {  INT16_C(     0),  INT16_C(  6805),  INT16_C(     0), -INT16_C( 10544),  INT16_C(     0),  INT16_C(  1554),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 15341),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2347),  INT16_C(     0),  INT16_C(     0), -INT16_C( 14097),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C(37418),
      { -INT32_C(  1439161341), -INT32_C(  2067522923),  INT32_C(   175688987), -INT32_C(    40184112),  INT32_C(  1042295168), -INT32_C(  1590229486),  INT32_C(   144617514), -INT32_C(   845494070),
        -INT32_C(  2122853652),  INT32_C(   134560749), -INT32_C(   653099512), -INT32_C(   707351468), -INT32_C(   418122027),  INT32_C(   646531836), -INT32_C(   986830854), -INT32_C(   611137297) } },
    { { -INT16_C( 30627),  INT16_C( 20112),  INT16_C(     0),  INT16_C(  9482), -INT16_C(  4084),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22393),
         INT16_C(     0),  INT16_C( 10071),  INT16_C(     0),  INT16_C( 28632),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C( 2715),
      {  INT32_C(  1648724061), -INT32_C(  1545580912), -INT32_C(    45140697),  INT32_C(   502473994), -INT32_C(  1788284916),  INT32_C(  1169973782), -INT32_C(  1173509337), -INT32_C(   456873849),
        -INT32_C(  1052374736),  INT32_C(  2120492887),  INT32_C(  1719382876), -INT32_C(   444370984),  INT32_C(  1970989919),  INT32_C(  1975138126),  INT32_C(  1764739298), -INT32_C(  1588726928) } },
    { {  INT16_C( 21602),  INT16_C(     0),  INT16_C( 21117),  INT16_C(  7223),  INT16_C( 14994), -INT16_C( 13393), -INT16_C( 31436), -INT16_C( 12506),
         INT16_C(     0),  INT16_C(     0),  INT16_C(   793),  INT16_C(     0), -INT16_C(  5493),  INT16_C(     0),  INT16_C(     0), -INT16_C( 32068),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C(38141),
      { -INT32_C(   960801694),  INT32_C(  1316558803),  INT32_C(    12407421), -INT32_C(  1309926345),  INT32_C(  1290287762), -INT32_C(   552285265), -INT32_C(  2100132556), -INT32_C(  2011771098),
        -INT32_C(   145829596),  INT32_C(  1699071976),  INT32_C(  1348797209), -INT32_C(  1325248225),  INT32_C(  1006561931), -INT32_C(   367390027), -INT32_C(  1116935017), -INT32_C(   532315460) } },
    { {  INT16_C( 15319),  INT16_C( 29856),  INT16_C(     0),  INT16_C(     0),  INT16_C( 26397),  INT16_C(     0), -INT16_C(  8106),  INT16_C(     0),
        -INT16_C( 30003), -INT16_C( 11266),  INT16_C(     0),  INT16_C(     0), -INT16_C(  3421),  INT16_C(     0),  INT16_C(     0), -INT16_C( 21630),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C(37715),
      {  INT32_C(   475741143),  INT32_C(    85947552), -INT32_C(   950583355), -INT32_C(   290331920),  INT32_C(   922838813), -INT32_C(  1121675183), -INT32_C(  1690312618), -INT32_C(  1758489920),
         INT32_C(  1840483021), -INT32_C(  1015884802),  INT32_C(    42649874), -INT32_C(   906937172), -INT32_C(   201198941), -INT32_C(   525261430), -INT32_C(   981667835),  INT32_C(  1331473282) } },
    { {  INT16_C( 13244),  INT16_C(     0), -INT16_C( 23305),  INT16_C(     0),  INT16_C( 25684), -INT16_C( 26555),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 21548), -INT16_C( 22112),  INT16_C(  2637),  INT16_C( 22891),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C( 3893),
      {  INT32_C(   786576316), -INT32_C(  2114390793), -INT32_C(   390224649),  INT32_C(  1859805293),  INT32_C(    93545556), -INT32_C(  1040869307), -INT32_C(  1184073891), -INT32_C(  2100715066),
        -INT32_C(   877548588), -INT32_C(  1756517984), -INT32_C(  1166079411), -INT32_C(  1087809173),  INT32_C(    46513341), -INT32_C(  1295795627), -INT32_C(   127127758), -INT32_C(  1518651952) } },
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 23351),  INT16_C( 29596),  INT16_C( 19224),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 26945), -INT16_C(  8682),  INT16_C(     0), -INT16_C(  9471),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C(11488),
      { -INT32_C(  1110081424), -INT32_C(  1765268969),  INT32_C(    99562460), -INT32_C(  1211978254),  INT32_C(  1936857007),  INT32_C(   899851465),  INT32_C(   392852380), -INT32_C(  2008855784),
        -INT32_C(   481945397),  INT32_C(   393809466),  INT32_C(   857499969), -INT32_C(   974397930), -INT32_C(  1086824971), -INT32_C(  1644897535),  INT32_C(  1723162446),  INT32_C(  1978595498) } },
    { {  INT16_C( 19288),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    18),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C(  6597),  INT16_C(     0), -INT16_C( 13537), -INT16_C( 18621),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C(13329),
      { -INT32_C(   767407272),  INT32_C(  2134606690), -INT32_C(  1570942537),  INT32_C(  1342133014),  INT32_C(   103415826),  INT32_C(  1382381725), -INT32_C(   867495970), -INT32_C(   587113339),
         INT32_C(   179258279),  INT32_C(  2122967751),  INT32_C(  1377887803),  INT32_C(  1268850489), -INT32_C(  1135424737),  INT32_C(   554678083),  INT32_C(  1290623687),  INT32_C(  1563029174) } },
    { {  INT16_C(     0), -INT16_C(   137),  INT16_C(     0),  INT16_C(     0),  INT16_C(   145),  INT16_C( 11297),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 15912),  INT16_C(     0),  INT16_C(     0),  INT16_C( 15809), -INT16_C( 17091),  INT16_C(     0),  INT16_C( 27294),  INT16_C( 32665),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT16_C(55602),
      { -INT32_C(   255592089), -INT32_C(  1747452041), -INT32_C(   222949295), -INT32_C(  1363290788), -INT32_C(  1603993455),  INT32_C(   251276321), -INT32_C(  1560432520),  INT32_C(  1971007246),
        -INT32_C(  1620689368), -INT32_C(  1909048003), -INT32_C(  1451168691),  INT32_C(  1381449153),  INT32_C(  1593031997),  INT32_C(  1651371497), -INT32_C(  1408996706), -INT32_C(  1054769255) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    int16_t arr_r[16] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm512_mask_cvtepi32_storeu_epi16(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cvtepi32_storeu_epi16");
    easysimd_assert_equal_vi16(16, (const int16_t *)arr_r, (const int16_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t arr_r[32] = {0};
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i  a = easysimd_test_x86_random_i32x16();
    easysimd_mm512_mask_cvtepi32_storeu_epi16((void *)arr_r, k, a);
    easysimd__m512i  r = easysimd_mm512_loadu_si512((void *)arr_r);

    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepi64_storeu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t  r[32];
    easysimd__mmask8 k;
    int64_t a[8];
  } test_vec[8] =  {
    { {  INT16_C( 32324),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 29260),  INT16_C( 14130),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 49),
      {  INT64_C( 3034805780397915716),  INT64_C( 3213746195626389845), -INT64_C(  950604551995267623), -INT64_C( 1323435239118859767),
        -INT64_C( 5934266185223533132), -INT64_C( 7673699748562258126), -INT64_C( 4353660624117304209), -INT64_C( 7757214098786047294) } },
    { {  INT16_C(  3963),  INT16_C( 17242),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   445), -INT16_C( 21677),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 99),
      { -INT64_C( 1446102205357944965), -INT64_C( 5164434956811287718),  INT64_C( 3859925230213592606), -INT64_C( 3412415334515803030),
         INT64_C( 2547483540498177978),  INT64_C( 6717501742722318781),  INT64_C( 5409875255683165011), -INT64_C( 8152025687670445639) } },
    { { -INT16_C(   917),  INT16_C(  9346),  INT16_C(  4127), -INT16_C( 23871),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31211),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(143),
      {  INT64_C( 6387083210175020139),  INT64_C(  662859683980518530),  INT64_C( 5590151465603305503), -INT64_C( 1405211586101927231),
        -INT64_C( 3145073498476390485),  INT64_C(  216425553684852889), -INT64_C( 2205229860834579467), -INT64_C( 2197104377597884907) } },
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4632),  INT16_C(     0),  INT16_C( 21867),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 40),
      {  INT64_C( 3269980831711159124),  INT64_C( 4961581790823274497), -INT64_C( 2343201831332285453), -INT64_C( 1688015793031933416),
         INT64_C( 6576120059028114649),  INT64_C( 6631581691892815211), -INT64_C( 3870774943742268401),  INT64_C( 3228454571507735888) } },
    { {  INT16_C(     0), -INT16_C( 16440),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22078),  INT16_C(     0),  INT16_C( 20011),  INT16_C(  9425),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(210),
      { -INT64_C( 2667389346567667074),  INT64_C( 3815570089701588936),  INT64_C(  140088384074177635),  INT64_C( 2341168817450046949),
        -INT64_C(  566701279734027838),  INT64_C( 7595266922130466412), -INT64_C( 1958788761797374421),  INT64_C( 7391546788874757329) } },
    { { -INT16_C( 20923),  INT16_C(     0), -INT16_C( 31824),  INT16_C(     0),  INT16_C( 24113),  INT16_C( 29865),  INT16_C(     0), -INT16_C( 24860),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(181),
      {  INT64_C( 3917531140363497029),  INT64_C( 5094880664830644665),  INT64_C( 8682141100284412848), -INT64_C( 3928964593654828872),
        -INT64_C(  849183363024855503), -INT64_C( 3825753892986194775),  INT64_C( 1189499331732188852),  INT64_C( 5994939959383203556) } },
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C( 18267), -INT16_C( 25693),  INT16_C(     0), -INT16_C( 10817),  INT16_C(     0), -INT16_C( 31695),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(172),
      {  INT64_C(   49500614877009133),  INT64_C( 7439916045220598105),  INT64_C( 7364176799925749595), -INT64_C( 4999316858767631453),
        -INT64_C( 4847159203598465489),  INT64_C( 1540378110653289919), -INT64_C( 7855237698588303411), -INT64_C( 8162737628433382351) } },
    { {  INT16_C( 19063), -INT16_C( 25076), -INT16_C(  3059),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8492),  INT16_C(     0), -INT16_C(  6621),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(167),
      { -INT64_C( 2619469879609505161),  INT64_C( 2188655958578011660),  INT64_C( 8418529459338867725),  INT64_C(  210370153602601506),
        -INT64_C( 3995662932101237332), -INT64_C( 2789494211922484948),  INT64_C( 1734538921858848429), -INT64_C(  815073155065518557) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    int16_t arr_r[32] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm512_mask_cvtepi64_storeu_epi16(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cvtepi64_storeu_epi16");
    easysimd_assert_equal_vi16(8, (const int16_t *)arr_r, (const int16_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t arr_r[32] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i  a = easysimd_test_x86_random_i64x8();
    easysimd_mm512_mask_cvtepi64_storeu_epi16((void *)arr_r, k, a);
    easysimd__m512i  r = easysimd_mm512_loadu_si512((void *)arr_r);

    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvtepi64_storeu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t  r[16];
    easysimd__mmask8 k;
    int64_t a[8];
  } test_vec[8] =  {
    { { -INT32_C(   261453860),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1446883291),  INT32_C(    24894001),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(   261453860),  INT32_C(  1452613840), -INT32_C(  1873281861),  INT32_C(   927862198),  INT32_C(  1489170680),  INT32_C(  1270468644),  INT32_C(  1446883291), -INT32_C(  1955550545) },
      UINT8_C( 25),
      {  INT64_C( 6238928940550490076),  INT64_C( 3985137798026362043),  INT64_C( 5456621278062637304), -INT64_C( 8399025635003093029),
        -INT64_C( 5379812563918005711), -INT64_C( 5405844864253972980),  INT64_C( 7490830387657948292),  INT64_C(  986058362166587765) } },
    { {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1598033294),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(   973213226),  INT32_C(   340244326), -INT32_C(  1865413892), -INT32_C(  1072331516),  INT32_C(  1481490066), -INT32_C(   473058126), -INT32_C(  1598033294), -INT32_C(   861352799) },
      UINT8_C(  8),
      {  INT64_C( 1461338253792775722), -INT64_C( 4605628789260547332), -INT64_C( 2031769178795557230), -INT64_C( 3699482099326127502),
         INT64_C( 6797424292666976986),  INT64_C( 4331340971845231959), -INT64_C( 8045854330641421887), -INT64_C( 4845352155274537388) } },
    { {  INT32_C(   367394248),  INT32_C(   891471921),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1405368244),
         INT32_C(   367394248),  INT32_C(  2003649362),  INT32_C(   891471921),  INT32_C(  1307991780),  INT32_C(  1433103091), -INT32_C(   676725823),  INT32_C(    79273515),  INT32_C(   680031584) },
      UINT8_C(131),
      {  INT64_C( 8605608482808659400),  INT64_C( 5617781919428298801), -INT64_C( 2906515276710581517),  INT64_C( 2920713413606350379),
        -INT64_C( 2216709240389800590), -INT64_C( 5790206643348808077),  INT64_C( 3895522466394804343), -INT64_C( 5704720158321555380) } },
    { {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2110163917), -INT32_C(  1492560005),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(  1126140911),  INT32_C(   238010800), -INT32_C(   439646504),  INT32_C(   995916331),  INT32_C(   742823039),  INT32_C(  1803058595),  INT32_C(  2074617016),  INT32_C(   179001592) },
      UINT8_C( 48),
      {  INT64_C( 1022248605263623185),  INT64_C( 4277428075052631768),  INT64_C( 7744077699039532159),  INT64_C(  768805985646552248),
         INT64_C( 2633749590981381069),  INT64_C( 5828332818895820667),  INT64_C( 6775654260341025273), -INT64_C( 3090450074072141798) } },
    { { -INT32_C(   681880606), -INT32_C(   284689783), -INT32_C(   894728747),  INT32_C(  1715861474),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(   681880606),  INT32_C(   911442143), -INT32_C(   284689783),  INT32_C(    99113180), -INT32_C(   894728747),  INT32_C(  1021577612),  INT32_C(  1715861474), -INT32_C(   160097773) },
      UINT8_C( 15),
      {  INT64_C( 3914614199994242018),  INT64_C(  425687870712838793),  INT64_C( 4387642437266015701), -INT64_C(  687614697481570334),
        -INT64_C( 2701561292915879571), -INT64_C( 5189075982748120550), -INT64_C( 7782107242420918217), -INT64_C( 3646103293090183843) } },
    { {  INT32_C(  1418860852),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(    52600551),  INT32_C(           0),
         INT32_C(  1418860852),  INT32_C(   644836508), -INT32_C(   402102730),  INT32_C(   169852770), -INT32_C(  1497570752),  INT32_C(   654528738), -INT32_C(  1165527990),  INT32_C(   251352282) },
      UINT8_C( 65),
      {  INT64_C( 2769551714545703220),  INT64_C(  729512096177874486),  INT64_C( 2811179526799548992),  INT64_C( 1079549834094408778),
         INT64_C( 3472225809067117933),  INT64_C( 4009672575530055480), -INT64_C(  350188191436938983),  INT64_C( 7799867309932587605) } },
    { {  INT32_C(  1933145505), -INT32_C(  1194113663), -INT32_C(   922956313),  INT32_C(  1603947950),  INT32_C(   547346299),  INT32_C(  2054803599),  INT32_C(           0),  INT32_C(           0),
         INT32_C(  1933145505), -INT32_C(   592746075), -INT32_C(  1194113663),  INT32_C(  1271991018), -INT32_C(   922956313), -INT32_C(  1692534543),  INT32_C(  1603947950),  INT32_C(   748553867) },
      UINT8_C( 63),
      { -INT64_C( 2545825005024217695),  INT64_C( 5463159826216600961), -INT64_C( 7269380506163294745),  INT64_C( 3215014379663281582),
        -INT64_C( 4468333615995955333), -INT64_C( 4484094452247703409),  INT64_C( 7540617568567607832), -INT64_C( 4563441348727324684) } },
    { { -INT32_C(  1803624118),  INT32_C(           0),  INT32_C(  2078222035),  INT32_C(  1968172815), -INT32_C(   830918416), -INT32_C(  1624439749),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(  1803624118), -INT32_C(  1373356066),  INT32_C(    76062393),  INT32_C(   622610787),  INT32_C(  2078222035),  INT32_C(   175065035),  INT32_C(  1968172815), -INT32_C(   441315430) },
      UINT8_C( 61),
      { -INT64_C( 5898519386741874358),  INT64_C( 2674092968377884345),  INT64_C(  751898602076317395), -INT64_C( 1895435337102004465),
         INT64_C( 2989437360493113584),  INT64_C( 2433150300713714747), -INT64_C(  413191061371050897), -INT64_C( 7819899716993313797) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    int32_t arr_r[16] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm512_mask_cvtepi64_storeu_epi32(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cvtepi64_storeu_epi32");
    easysimd_assert_equal_vi32(8, (const int32_t *)arr_r, (const int32_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t arr_r[8] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i  a = easysimd_test_x86_random_i64x8();
    easysimd_mm512_mask_cvtepi64_storeu_epi32((void *)arr_r, k, a);
    easysimd__m512i  r = easysimd_mm512_loadu_si512((void *)arr_r);

    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi16_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[16];
    easysimd__mmask8 k;
    int16_t a[8];
  } test_vec[8] = {
    { {  INT8_C(  33), -INT8_C( 115), -INT8_C(  42),  INT8_C(  93),  INT8_C(   0),  INT8_C( 114),  INT8_C(  25),  INT8_C(  12),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(239),
      {  INT16_C(  4897),  INT16_C( 30861), -INT16_C( 16426),  INT16_C( 30301),  INT16_C( 28517),  INT16_C( 28786),  INT16_C( 21529),  INT16_C(  4620) } },
    { {  INT8_C(   0),  INT8_C(  84),  INT8_C(  85),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  43), -INT8_C(  76),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(198),
      { -INT16_C( 20461),  INT16_C( 26708),  INT16_C( 13141),  INT16_C( 32691), -INT16_C(    47), -INT16_C( 27789),  INT16_C( 25301), -INT16_C(  5708) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 124), -INT8_C(   4),  INT8_C(  36), -INT8_C(  93),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(240),
      { -INT16_C( 16595), -INT16_C( 30033),  INT16_C(  5173), -INT16_C( 22279),  INT16_C(  4740), -INT16_C( 28420), -INT16_C( 15836), -INT16_C( 10845) } },
    { {  INT8_C(   0),  INT8_C(  73), -INT8_C(  86),  INT8_C(   0), -INT8_C(  82),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 22),
      {  INT16_C( 10764), -INT16_C( 16567),  INT16_C(  6826),  INT16_C(  7614), -INT16_C( 27474),  INT16_C( 25215),  INT16_C( 28541),  INT16_C( 15503) } },
    { {  INT8_C(  26),  INT8_C(  51),  INT8_C(  25),  INT8_C(  38),  INT8_C(  72),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 31),
      {  INT16_C( 28954),  INT16_C(  4915), -INT16_C( 18407),  INT16_C(  5414),  INT16_C( 19016), -INT16_C(  4905), -INT16_C(  4833),  INT16_C( 19192) } },
    { { -INT8_C(  73),  INT8_C(  81),  INT8_C(  17),  INT8_C(   0),  INT8_C(  98),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 55),
      { -INT16_C(  2889),  INT16_C( 30033), -INT16_C(   239), -INT16_C( 28663), -INT16_C( 31134), -INT16_C(  3840),  INT16_C(  8130),  INT16_C( 13323) } },
    { {  INT8_C(   0),  INT8_C(  10),  INT8_C(   0),  INT8_C(   0),  INT8_C(  63),  INT8_C(   0), -INT8_C(   7),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 82),
      {  INT16_C( 19743),  INT16_C( 17674),  INT16_C( 21347),  INT16_C( 14991), -INT16_C( 20673),  INT16_C( 14120),  INT16_C( 24569), -INT16_C(  4626) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   3), -INT8_C(  83),  INT8_C(   0),  INT8_C(  46),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(176),
      { -INT16_C(   413),  INT16_C( 28080),  INT16_C(  4750), -INT16_C( 28941), -INT16_C( 18941),  INT16_C(  4013),  INT16_C(   234),  INT16_C( 14126) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    int8_t arr_r[16] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm_mask_cvtepi16_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cvtepi16_storeu_epi8");
    easysimd_assert_equal_vi8(8, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[16] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i  a = easysimd_test_x86_random_i16x8();
    easysimd_mm_mask_cvtepi16_storeu_epi8((void *)arr_r, k, a);
    easysimd__m128i  r = easysimd_mm_loadu_si128((void *)arr_r);

    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi32_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[16];
    easysimd__mmask8 k;
    int32_t a[4];
  } test_vec[8] =  {
    { { -INT8_C( 108),  INT8_C(  54),  INT8_C(  25),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(135),
      { -INT32_C(  2091752044),  INT32_C(   838899510),  INT32_C(   833938201), -INT32_C(  1311342357) } },
    { {  INT8_C(  75),  INT8_C(   0),  INT8_C(  53),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(117),
      {  INT32_C(  1883770699), -INT32_C(  1453829330), -INT32_C(  1628325067),  INT32_C(   288520364) } },
    { {  INT8_C(   0),  INT8_C( 122),  INT8_C(   0),  INT8_C(  53),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(202),
      { -INT32_C(  1235073098),  INT32_C(   781810554), -INT32_C(  2102491988),  INT32_C(  1154294581) } },
    { {  INT8_C(   0),  INT8_C(   0), -INT8_C( 109),  INT8_C(  14),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(108),
      { -INT32_C(  1765838275),  INT32_C(   206697755), -INT32_C(   981140077),  INT32_C(  1434144526) } },
    { {  INT8_C(  49),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(177),
      { -INT32_C(   903033039), -INT32_C(  1404052995),  INT32_C(   701465691),  INT32_C(   996555721) } },
    { { -INT8_C(   4),  INT8_C(  99),  INT8_C(   0), -INT8_C(  82),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(251),
      {  INT32_C(  1307596796), -INT32_C(   397769885), -INT32_C(  1003005880),  INT32_C(  2113267886) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C( 114),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 20),
      {  INT32_C(   267221952), -INT32_C(   158119641), -INT32_C(   667788174), -INT32_C(   271241832) } },
    { {  INT8_C(  34),  INT8_C(   0), -INT8_C(  84), -INT8_C(  15),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 29),
      {  INT32_C(  1839223330),  INT32_C(  1908795450), -INT32_C(  1571195988),  INT32_C(  1818390257) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    int8_t arr_r[16] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm_mask_cvtepi32_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cvtepi32_storeu_epi8");
    easysimd_assert_equal_vi8(4, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[16] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i  a = easysimd_test_x86_random_i32x4();
    easysimd_mm_mask_cvtepi32_storeu_epi8((void *)arr_r, k, a);
    easysimd__m128i  r = easysimd_mm_loadu_si128((void *)arr_r);

    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi64_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[16];
    easysimd__mmask8 k;
    int64_t a[2];
  } test_vec[8] =  {
    { { -INT8_C( 113),  INT8_C(  64),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(179),
      { -INT64_C( 5954024784681618801),  INT64_C( 4985789678149637184) } },
    { {  INT8_C(  82),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(197),
      { -INT64_C( 2242228996191766958),  INT64_C( 3344723154267558206) } },
    { {  INT8_C(   0),  INT8_C(  54),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(  6),
      { -INT64_C( 3968985334824030089),  INT64_C( 3655413802278467382) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(140),
      { -INT64_C( 3276708829310693806), -INT64_C( 2195873493794782556) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(196),
      {  INT64_C( 7241593064477212253), -INT64_C(  233461303517578059) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 32),
      {  INT64_C( 2882178956158216950),  INT64_C( 5728749773689172125) } },
    { { -INT8_C(  56),  INT8_C(  38),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(179),
      {  INT64_C( 4775141866352006600),  INT64_C( 1864198256286134822) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(224),
      { -INT64_C( 7779392498435189157), -INT64_C( 4649998592664167634) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    int8_t arr_r[16] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm_mask_cvtepi64_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cvtepi64_storeu_epi8");
    easysimd_assert_equal_vi8(2, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[16] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i  a = easysimd_test_x86_random_i64x2();
    easysimd_mm_mask_cvtepi64_storeu_epi8((void *)arr_r, k, a);
    easysimd__m128i  r = easysimd_mm_loadu_si128((void *)arr_r);

    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi32_storeu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t  r[8];
    easysimd__mmask8 k;
    int32_t a[4];
  } test_vec[8] =  {
    { { -INT16_C(  7082),  INT16_C(  5166),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(115),
      {  INT32_C(  2085545046), -INT32_C(  1029893074), -INT32_C(   685868735),  INT32_C(   844803091) } },
    { {  INT16_C( 22005),  INT16_C(     0), -INT16_C( 11674),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(117),
      { -INT32_C(   773761547),  INT32_C(  1998150232), -INT32_C(  2080976282), -INT32_C(  1596364100) } },
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 11854),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(184),
      { -INT32_C(   204681386), -INT32_C(  1351807599),  INT32_C(  1063026916),  INT32_C(   137679282) } },
    { {  INT16_C(     0),  INT16_C( 29399),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(178),
      {  INT32_C(   520904710), -INT32_C(   839814441), -INT32_C(   818434314),  INT32_C(   489025358) } },
    { {  INT16_C(     0),  INT16_C(     0), -INT16_C( 18195),  INT16_C( 16064),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(188),
      { -INT32_C(  2033602791), -INT32_C(  1173967266),  INT32_C(   562870509),  INT32_C(   539442880) } },
    { { -INT16_C(  2233), -INT16_C( 19515),  INT16_C(     0), -INT16_C( 22446),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 75),
      {  INT32_C(   951973703), -INT32_C(    51989563), -INT32_C(  1460914813),  INT32_C(    29468754) } },
    { {  INT16_C( 24647),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(113),
      {  INT32_C(  1293901895), -INT32_C(  1509553126),  INT32_C(  1441121838), -INT32_C(   560189209) } },
    { { -INT16_C( 23595),  INT16_C(     0),  INT16_C( 17356), -INT16_C( 23228),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(237),
      { -INT32_C(  1147034667), -INT32_C(  1930419040), -INT32_C(  1909177396), -INT32_C(  1529502396) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    int16_t arr_r[8] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm_mask_cvtepi32_storeu_epi16(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cvtepi32_storeu_epi16");
    easysimd_assert_equal_vi16(4, (const int16_t *)arr_r, (const int16_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t arr_r[8] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i  a = easysimd_test_x86_random_i32x4();
    easysimd_mm_mask_cvtepi32_storeu_epi16((void *)arr_r, k, a);
    easysimd__m128i  r = easysimd_mm_loadu_si128((void *)arr_r);

    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi64_storeu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t  r[8];
    easysimd__mmask8 k;
    int64_t a[2];
  } test_vec[8] =  {
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(212),
      {  INT64_C( 6292198106829895995),  INT64_C( 1511193202813055111) } },
    { {  INT16_C( 21005),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(101),
      {  INT64_C( 3660556235373826573),  INT64_C( 4532809277650380917) } },
    { {  INT16_C(     0), -INT16_C( 10815),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(194),
      {  INT64_C( 1558150283574514972),  INT64_C( 4307501632891704769) } },
    { { -INT16_C( 17616),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 29),
      { -INT64_C( 2295292168697955536),  INT64_C( 9163533766770502439) } },
    { { -INT16_C(  4966), -INT16_C(  2129),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(139),
      { -INT64_C( 1364884251443139430), -INT64_C( 1250210470451742801) } },
    { {  INT16_C(     0),  INT16_C( 23827),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 98),
      {  INT64_C( 7274444511292284067), -INT64_C( 3974047493191279341) } },
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(140),
      { -INT64_C(  713485502862210541),  INT64_C( 1725077919292935334) } },
    { {  INT16_C(     0), -INT16_C(  6167),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(146),
      {  INT64_C(  363571355340690617),  INT64_C( 8851237588051027945) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    int16_t arr_r[8] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm_mask_cvtepi64_storeu_epi16(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cvtepi64_storeu_epi16");
    easysimd_assert_equal_vi16(2, (const int16_t *)arr_r, (const int16_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t arr_r[8] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i  a = easysimd_test_x86_random_i64x2();
    easysimd_mm_mask_cvtepi64_storeu_epi16((void *)arr_r, k, a);
    easysimd__m128i  r = easysimd_mm_loadu_si128((void *)arr_r);

    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvtepi64_storeu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t  r[4];
    easysimd__mmask8 k;
    int64_t a[2];
  } test_vec[8] =  {
    { {  INT32_C(  1618488776),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 61),
      {  INT64_C( 8126091268250414536), -INT64_C( 6489599845550582432) } },
    { {  INT32_C(           0), -INT32_C(  1569978559),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(110),
      {  INT64_C( 2845734159938495413), -INT64_C( 7463685681979848895) } },
    { { -INT32_C(  1865577269),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 33),
      { -INT64_C( 8888646622954812213), -INT64_C( 4558976955607418859) } },
    { {  INT32_C(           0), -INT32_C(   115160178),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(150),
      {  INT64_C( 8746531489118348763), -INT64_C( 1097395738022589554) } },
    { { -INT32_C(    79696555),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 17),
      {  INT64_C( 5040466355992653141),  INT64_C( 3886920183193243738) } },
    { {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(200),
      { -INT64_C( 2488446895291874742), -INT64_C( 5235940574076216772) } },
    { {  INT32_C(           0),  INT32_C(   672277815),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(114),
      { -INT64_C(  732922982358898351), -INT64_C( 7173149700716878537) } },
    { {  INT32_C(           0),  INT32_C(  1686714894),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(142),
      {  INT64_C( 6228492808451348254),  INT64_C( 1997061602784393742) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    int32_t arr_r[4] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm_mask_cvtepi64_storeu_epi32(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cvtepi64_storeu_epi32");
    easysimd_assert_equal_vi32(2, (const int32_t *)arr_r, (const int32_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t arr_r[4] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i  a = easysimd_test_x86_random_i64x2();
    easysimd_mm_mask_cvtepi64_storeu_epi32((void *)arr_r, k, a);
    easysimd__m128i  r = easysimd_mm_loadu_si128((void *)arr_r);

    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepi16_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[32];
    easysimd__mmask16 k;
    int16_t a[16];
  } test_vec[8] = {
    { {  INT8_C(   0),  INT8_C(   5),  INT8_C( 126),  INT8_C(   0),  INT8_C(  16),  INT8_C(   0), -INT8_C(  25),  INT8_C(  99),
         INT8_C(  87),  INT8_C(   0),  INT8_C(  16),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 107),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(17878),
      {  INT16_C( 14471),  INT16_C( 12805),  INT16_C( 19582),  INT16_C(  7568), -INT16_C( 26864),  INT16_C( 28606),  INT16_C( 26599), -INT16_C( 18845),
         INT16_C( 11607),  INT16_C( 23156),  INT16_C(  5392),  INT16_C( 17418),  INT16_C( 10252),  INT16_C(  2980),  INT16_C( 31339), -INT16_C(  3504) } },
    { {  INT8_C(   0), -INT8_C(  95),  INT8_C(   0),  INT8_C(   0),  INT8_C(  33),  INT8_C( 116),  INT8_C(   0), -INT8_C(  79),
         INT8_C(  37),  INT8_C(   0),  INT8_C(   6),  INT8_C(   0), -INT8_C( 117),  INT8_C(   0), -INT8_C(  74),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(21938),
      {  INT16_C( 12581), -INT16_C( 19039), -INT16_C( 20146),  INT16_C(  3148),  INT16_C( 13345), -INT16_C( 31628), -INT16_C( 13334),  INT16_C( 24241),
        -INT16_C( 16091),  INT16_C( 12148), -INT16_C( 32762), -INT16_C( 21928), -INT16_C( 15477), -INT16_C(  9436), -INT16_C( 10570), -INT16_C(  9424) } },
    { { -INT8_C( 112), -INT8_C( 125),  INT8_C(  98),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  10),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  75),  INT8_C(   0), -INT8_C(  93), -INT8_C(  73),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(53511),
      {  INT16_C( 21904), -INT16_C(  8829), -INT16_C( 23454), -INT16_C( 10735), -INT16_C(  1240), -INT16_C(  9823), -INT16_C( 14503), -INT16_C( 12902),
        -INT16_C( 24330),  INT16_C( 20046), -INT16_C(  9910),  INT16_C( 28178), -INT16_C( 14155), -INT16_C(  6843),  INT16_C( 19619),  INT16_C( 13239) } },
    { {  INT8_C(   0), -INT8_C(  34),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  66),  INT8_C(   0),  INT8_C(  26),
         INT8_C(   0),  INT8_C( 107),  INT8_C(   0),  INT8_C(  97),  INT8_C(   5),  INT8_C( 100),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(15010),
      {  INT16_C(  1040),  INT16_C(  8670),  INT16_C(  1754),  INT16_C( 31516),  INT16_C( 30431),  INT16_C( 31042),  INT16_C( 14659), -INT16_C( 28390),
         INT16_C( 25735), -INT16_C( 26261),  INT16_C(  8403),  INT16_C(  6241),  INT16_C(  1029), -INT16_C( 17308),  INT16_C(  1592),  INT16_C( 18678) } },
    { {  INT8_C(   0), -INT8_C(  38),  INT8_C(   0), -INT8_C(   4),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 100),  INT8_C(   0),  INT8_C(  30),  INT8_C(   0), -INT8_C(  29), -INT8_C(  23),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(54282),
      { -INT16_C(  7062), -INT16_C( 31014), -INT16_C( 18080), -INT16_C( 23812),  INT16_C( 16435),  INT16_C( 19931),  INT16_C( 25553),  INT16_C( 15537),
        -INT16_C( 31492),  INT16_C( 24156),  INT16_C( 25244),  INT16_C(   354), -INT16_C( 26082),  INT16_C(  5383),  INT16_C(  4835),  INT16_C( 19945) } },
    { {  INT8_C(   0),  INT8_C( 125), -INT8_C(   7),  INT8_C(   0), -INT8_C(   3),  INT8_C(  55),  INT8_C(  30),  INT8_C(  51),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  47),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  65), -INT8_C(  67),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(50422),
      {  INT16_C( 22227), -INT16_C( 12163), -INT16_C( 20231), -INT16_C( 11248), -INT16_C(  7683), -INT16_C( 20681),  INT16_C( 13342),  INT16_C( 31283),
        -INT16_C( 12142), -INT16_C(  2852), -INT16_C(  1071), -INT16_C( 10097),  INT16_C( 29200), -INT16_C(  1558), -INT16_C(  7745), -INT16_C( 27971) } },
    { {  INT8_C(  98), -INT8_C(  21),  INT8_C(   5),  INT8_C(   0), -INT8_C( 104),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(   9), -INT8_C(  60),  INT8_C(   0), -INT8_C(   8), -INT8_C(  51),  INT8_C(  16),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(15159),
      {  INT16_C( 12386),  INT16_C( 29419), -INT16_C(  5883),  INT16_C( 15444),  INT16_C( 29336), -INT16_C( 13456),  INT16_C(   748), -INT16_C( 13925),
         INT16_C( 27895), -INT16_C( 31036), -INT16_C( 11195),  INT16_C( 12280), -INT16_C( 18483), -INT16_C( 29936),  INT16_C( 18505), -INT16_C( 21306) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  28), -INT8_C(   2),  INT8_C(  45), -INT8_C( 102),  INT8_C(   0),
        -INT8_C(  86),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 118), -INT8_C(  45),  INT8_C(   0),  INT8_C(   2),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT16_C(45432),
      {  INT16_C( 32030),  INT16_C( 29338),  INT16_C( 12986),  INT16_C( 10980), -INT16_C( 11778), -INT16_C( 26323),  INT16_C(  9370),  INT16_C( 24070),
         INT16_C( 19370), -INT16_C( 24014), -INT16_C(   134), -INT16_C( 29863), -INT16_C( 23926),  INT16_C( 20691),  INT16_C( 19278),  INT16_C( 27906) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    int8_t arr_r[32] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm256_mask_cvtepi16_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cvtepi16_storeu_epi8");
    easysimd_assert_equal_vi8(16, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[32] = {0};
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i  a = easysimd_test_x86_random_i16x16();
    easysimd_mm256_mask_cvtepi16_storeu_epi8((void *)arr_r, k, a);
    easysimd__m256i  r = easysimd_mm256_loadu_si256((void *)arr_r);

    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepi32_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[32];
    easysimd__mmask8 k;
    int32_t a[8];
  } test_vec[8] =  {
    { {  INT8_C(  14),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  45),  INT8_C(   0),  INT8_C(   5),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 81),
      { -INT32_C(  2037767154), -INT32_C(  1737218565), -INT32_C(   647272152),  INT32_C(  1618142158), -INT32_C(  2003779629), -INT32_C(   970763106), -INT32_C(   417987067), -INT32_C(    63358227) } },
    { { -INT8_C(  61), -INT8_C(   9),  INT8_C(  93),  INT8_C(  92),  INT8_C(  31),  INT8_C(   0), -INT8_C( 106),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 95),
      { -INT32_C(   396721469),  INT32_C(  1611723511),  INT32_C(  1093593437),  INT32_C(   454331996),  INT32_C(  1941543967), -INT32_C(   965181505), -INT32_C(   793550954),  INT32_C(  1529851800) } },
    { {  INT8_C(   0),  INT8_C(  84),  INT8_C(   0),  INT8_C(   0),  INT8_C( 109),  INT8_C(  80),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 50),
      {  INT32_C(  2083079050),  INT32_C(  1037732180),  INT32_C(  1184439223), -INT32_C(   882527185), -INT32_C(   309667731), -INT32_C(  1350348720), -INT32_C(  1287105788),  INT32_C(   233153411) } },
    { {  INT8_C(  14),  INT8_C(  99),  INT8_C(  18),  INT8_C(   0),  INT8_C(   0), -INT8_C(  92),  INT8_C(  25), -INT8_C(  16),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(231),
      { -INT32_C(  1757705970),  INT32_C(  2119137379), -INT32_C(   961702638), -INT32_C(   768313094),  INT32_C(  1394745603), -INT32_C(   145239644), -INT32_C(  1116075239), -INT32_C(     5994512) } },
    { { -INT8_C(  33),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 104),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 17),
      {  INT32_C(  1467258591),  INT32_C(  2053764069),  INT32_C(   427045024), -INT32_C(  2061744540),  INT32_C(   975794024), -INT32_C(   766304057),  INT32_C(   583143579), -INT32_C(  1825324620) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 112), -INT8_C( 106),  INT8_C(   0),  INT8_C(  59),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 88),
      { -INT32_C(  1690440792), -INT32_C(  2076460972),  INT32_C(  1944605740), -INT32_C(   522490512), -INT32_C(  1213786730),  INT32_C(  2035448169), -INT32_C(    47287237),  INT32_C(  1347797416) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C( 110), -INT8_C(  59),  INT8_C(   0),  INT8_C( 123),  INT8_C(   0),  INT8_C( 121),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(172),
      {  INT32_C(  1241639826),  INT32_C(  2071430438), -INT32_C(   605296018), -INT32_C(   613233723), -INT32_C(   347854478), -INT32_C(   282608261), -INT32_C(  1382603541),  INT32_C(   190441337) } },
    { {  INT8_C(   0), -INT8_C(  51),  INT8_C(   0),  INT8_C(   0), -INT8_C( 101),  INT8_C(   0), -INT8_C(  58),  INT8_C(  75),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(210),
      { -INT32_C(   520530342), -INT32_C(  1219595059),  INT32_C(   729557343), -INT32_C(   996321381),  INT32_C(  1497336219), -INT32_C(   733728848), -INT32_C(  1370623546), -INT32_C(  1518315445) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    int8_t arr_r[32] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm256_mask_cvtepi32_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cvtepi32_storeu_epi8");
    easysimd_assert_equal_vi8(8, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[32] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i  a = easysimd_test_x86_random_i32x8();
    easysimd_mm256_mask_cvtepi32_storeu_epi8((void *)arr_r, k, a);
    easysimd__m256i  r = easysimd_mm256_loadu_si256((void *)arr_r);

    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepi64_storeu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t  r[32];
    easysimd__mmask8 k;
    int64_t a[4];
  } test_vec[8] =  {
    { { -INT8_C(   7),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(113),
      { -INT64_C( 6471757381553601799),  INT64_C( 6997630916436991997),  INT64_C( 8908469480030415497),  INT64_C( 7240148189179295959) } },
    { {  INT8_C(   0), -INT8_C( 116), -INT8_C(  72),  INT8_C(  62),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(110),
      {  INT64_C( 8729673561266039670), -INT64_C( 4564088619073570420), -INT64_C( 8877432362870527048), -INT64_C( 4118450254373726658) } },
    { {  INT8_C(   0),  INT8_C(   9),  INT8_C( 110),  INT8_C(   2),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(254),
      { -INT64_C( 8156773098845573319),  INT64_C( 4745355391345618185),  INT64_C( 8336746327690891374),  INT64_C( 3338444906292255746) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(120),
      {  INT64_C( 2992671899022880556),  INT64_C( 6165711036172184127), -INT64_C( 3124541872799611505),  INT64_C( 8731062740678474024) } },
    { {  INT8_C(   0), -INT8_C( 118), -INT8_C(  10),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(102),
      {  INT64_C( 3031709253340690615), -INT64_C( 3811728888117119862),  INT64_C( 6378037564355624182),  INT64_C( 1620485934202303776) } },
    { {  INT8_C(  31),  INT8_C( 113),  INT8_C(  67),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 23),
      {  INT64_C( 3491910373812463903), -INT64_C( 2923750833894653839), -INT64_C( 9170578661320505533),  INT64_C( 8100556605132624692) } },
    { {  INT8_C(   1),  INT8_C(   0), -INT8_C(   8), -INT8_C( 121),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C(223),
      { -INT64_C( 3444115927571211775), -INT64_C( 2520676492733863680),  INT64_C( 3295694134096470008),  INT64_C( 8888960499387862919) } },
    { {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  86),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) },
      UINT8_C( 72),
      {  INT64_C( 2035882608970710582), -INT64_C( 6589595262632525357), -INT64_C( 2277364756664937162),  INT64_C( 8349909269182341802) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    int8_t arr_r[32] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm256_mask_cvtepi64_storeu_epi8(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cvtepi64_storeu_epi8");
    easysimd_assert_equal_vi8(4, (const int8_t *)arr_r, (const int8_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t arr_r[32] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i  a = easysimd_test_x86_random_i64x4();
    easysimd_mm256_mask_cvtepi64_storeu_epi8((void *)arr_r, k, a);
    easysimd__m256i  r = easysimd_mm256_loadu_si256((void *)arr_r);

    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_LAST);

  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepi32_storeu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t  r[16];
    easysimd__mmask8 k;
    int32_t a[8];
  } test_vec[8] =  {
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   640),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8512),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 72),
      {  INT32_C(  1678596354), -INT32_C(  1842001220),  INT32_C(  1900753294), -INT32_C(   366411136), -INT32_C(    83642377), -INT32_C(   203961422), -INT32_C(  1413537472),  INT32_C(  1626596446) } },
    { { -INT16_C( 15360),  INT16_C(     0),  INT16_C( 19415), -INT16_C( 27276), -INT16_C( 13160), -INT16_C( 11869),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 61),
      {  INT32_C(  1274659840),  INT32_C(   735677689), -INT32_C(   643085353),  INT32_C(   751932788),  INT32_C(  1675545752), -INT32_C(   979054173),  INT32_C(  1948471184),  INT32_C(  1135772482) } },
    { { -INT16_C( 29013),  INT16_C( 27495), -INT16_C(  6121),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1808),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 71),
      {  INT32_C(   926977707), -INT32_C(  1307677849), -INT32_C(  1406670825), -INT32_C(  2059054151),  INT32_C(    36218929), -INT32_C(  1684804276), -INT32_C(  1814165744),  INT32_C(  1708794297) } },
    { { -INT16_C( 25573), -INT16_C( 13909),  INT16_C( 19184), -INT16_C( 11889),  INT16_C(     0),  INT16_C(  7945),  INT16_C(     0),  INT16_C( 12133),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(175),
      { -INT32_C(  2045338597), -INT32_C(  1818375765),  INT32_C(  1129073392),  INT32_C(   930402703), -INT32_C(   410749191),  INT32_C(   301408009),  INT32_C(   516590589), -INT32_C(  2134036635) } },
    { {  INT16_C(     0),  INT16_C(     0), -INT16_C(  8106), -INT16_C( 19116),  INT16_C(     0),  INT16_C(     0),  INT16_C( 29871),  INT16_C( 28738),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(204),
      { -INT32_C(  1384708124), -INT32_C(   291698012),  INT32_C(   662560854), -INT32_C(   903826092),  INT32_C(  1490290489), -INT32_C(  1974016513), -INT32_C(   537955153),  INT32_C(   648769602) } },
    { { -INT16_C( 11486),  INT16_C(  2672), -INT16_C( 22137),  INT16_C(     0), -INT16_C( 12734),  INT16_C( 23076), -INT16_C( 18871),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(119),
      {  INT32_C(   740021026),  INT32_C(  1350699632),  INT32_C(  1017424263), -INT32_C(   780833079),  INT32_C(   667995714), -INT32_C(  1730717148), -INT32_C(  1176848823), -INT32_C(  2094006175) } },
    { { -INT16_C( 20660), -INT16_C( 27855),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4164),  INT16_C(     0), -INT16_C( 24076), -INT16_C( 32302),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(211),
      {  INT32_C(  1447276364), -INT32_C(   623013071), -INT32_C(  1499260361),  INT32_C(  1575515279), -INT32_C(  1635708860), -INT32_C(  1645733145), -INT32_C(   167861772),  INT32_C(   516456914) } },
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14042), -INT16_C(  1762),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 48),
      { -INT32_C(  1621003252),  INT32_C(  1809267537),  INT32_C(  1408990686),  INT32_C(  1972852837), -INT32_C(   195283238), -INT32_C(  1075250914),  INT32_C(  2022825463), -INT32_C(  1297567834) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    int16_t arr_r[16] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm256_mask_cvtepi32_storeu_epi16(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cvtepi32_storeu_epi16");
    easysimd_assert_equal_vi16(8, (const int16_t *)arr_r, (const int16_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t arr_r[16] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i  a = easysimd_test_x86_random_i32x8();
    easysimd_mm256_mask_cvtepi32_storeu_epi16((void *)arr_r, k, a);
    easysimd__m256i  r = easysimd_mm256_loadu_si256((void *)arr_r);

    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_LAST);

  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepi64_storeu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t  r[16];
    easysimd__mmask8 k;
    int64_t a[4];
  } test_vec[8] =  {
    { {  INT16_C( 20972), -INT16_C(  3461), -INT16_C( 28152), -INT16_C( 29076),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(223),
      {  INT64_C( 4512281247056810476), -INT64_C( 6612641042176216453), -INT64_C( 1069350984087006712),  INT64_C( 2255296481543425644) } },
    { {  INT16_C( 14534),  INT16_C(     0),  INT16_C(  5557),  INT16_C( 30019),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(189),
      { -INT64_C( 4028223753822324538), -INT64_C( 3818931288760713783), -INT64_C( 3816609104567396939), -INT64_C( 8616197790059563709) } },
    { {  INT16_C(     0), -INT16_C( 20801),  INT16_C( 32510),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 86),
      {  INT64_C( 4374953738652886542), -INT64_C( 4331828774745821505), -INT64_C(  292510196507312386),  INT64_C( 2034578157184327881) } },
    { {  INT16_C(     0),  INT16_C(     0),  INT16_C( 18143),  INT16_C( 23017),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(108),
      {  INT64_C( 7934751832392993129), -INT64_C( 6153360558157706661), -INT64_C( 2994000713976101153), -INT64_C(   28428327599187479) } },
    { {  INT16_C( 23878),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4341),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(  9),
      { -INT64_C( 4285114047566095034),  INT64_C(  293186691956483155), -INT64_C( 8583560521377399861), -INT64_C(  936997568785805067) } },
    { {  INT16_C(     0), -INT16_C( 31064),  INT16_C(     0), -INT16_C( 15321),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(122),
      { -INT64_C( 3867544631861043512),  INT64_C( 5985158110611277480), -INT64_C(  518184540733110584), -INT64_C( 8305034897410178009) } },
    { {  INT16_C(     0),  INT16_C(     0), -INT16_C(  9007),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C(180),
      {  INT64_C( 7283206663880543410),  INT64_C( 7783152976754726574),  INT64_C( 8071096683692547281), -INT64_C( 5171611191651121169) } },
    { {  INT16_C(  8853), -INT16_C( 23667),  INT16_C(     0),  INT16_C( 29138),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) },
      UINT8_C( 75),
      { -INT64_C( 8329248950390676843), -INT64_C( 8957838162547072115), -INT64_C( 3175989308839523663),  INT64_C( 4674052154352824786) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    int16_t arr_r[16] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm256_mask_cvtepi64_storeu_epi16(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cvtepi64_storeu_epi16");
    easysimd_assert_equal_vi16(4, (const int16_t *)arr_r, (const int16_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t arr_r[16] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i  a = easysimd_test_x86_random_i64x4();
    easysimd_mm256_mask_cvtepi64_storeu_epi16((void *)arr_r, k, a);
    easysimd__m256i  r = easysimd_mm256_loadu_si256((void *)arr_r);

    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvtepi64_storeu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t  r[8];
    easysimd__mmask8 k;
    int64_t a[4];
  } test_vec[8] =  {
    { {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1169045322), -INT32_C(  2077831167),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 60),
      {  INT64_C( 9176829045622427837),  INT64_C( 8471165830317933752), -INT64_C( 7139361555617822538),  INT64_C( 2288220459967631361) } },
    { {  INT32_C(           0),  INT32_C(           0), -INT32_C(    41373415),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 52),
      { -INT64_C( 2094709434019642224),  INT64_C(  790038971522724768),  INT64_C( 7925812650905743641), -INT64_C( 3106657714662571190) } },
    { { -INT32_C(  1806628484),  INT32_C(  1324184966),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(115),
      { -INT64_C( 6182259250034570884),  INT64_C( 2047158070897636742), -INT64_C( 1734406592805640833), -INT64_C(  335518722461819659) } },
    { {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   205817100),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(  8),
      {  INT64_C( 4135055170360872872),  INT64_C( 9083409813981147543),  INT64_C( 4365617105536725626),  INT64_C( 8862159179611863796) } },
    { {  INT32_C(           0), -INT32_C(  1467822617), -INT32_C(  1389509917),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 70),
      {  INT64_C( 7583573282877232892), -INT64_C( 8277051122216940057), -INT64_C( 3269214925399145757), -INT64_C( 8277020652844378843) } },
    { {  INT32_C(  1865167660),  INT32_C(   421199640),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C(115),
      {  INT64_C( 4491941992777659180), -INT64_C( 1946497178333937896), -INT64_C( 2277679795196352043), -INT64_C( 8195708671520325455) } },
    { {  INT32_C(    77856110),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 49),
      {  INT64_C( 5988909355467275630), -INT64_C( 2632257011979110917),  INT64_C( 6652949265287395220), -INT64_C( 3123724193372429449) } },
    { {  INT32_C(           0), -INT32_C(  1101346458),  INT32_C(           0),  INT32_C(  2077379886),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      UINT8_C( 74),
      {  INT64_C( 2965323435676194889), -INT64_C(  553039772753671834),  INT64_C( 7843142211529405787),  INT64_C( 4451428850681924910) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    int32_t arr_r[8] = {0};

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      easysimd_mm256_mask_cvtepi64_storeu_epi32(arr_r, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cvtepi64_storeu_epi32");
    easysimd_assert_equal_vi32(4, (const int32_t *)arr_r, (const int32_t *)test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t arr_r[8] = {0};
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i  a = easysimd_test_x86_random_i64x4();
    easysimd_mm256_mask_cvtepi64_storeu_epi32((void *)arr_r, k, a);
    easysimd__m256i  r = easysimd_mm256_loadu_si256((void *)arr_r);

    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi32_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtepi32_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi32_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtepi32_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepu32_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtepu32_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cvtepi64_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi64_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtepi64_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cvtepu64_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepu64_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtepu64_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtph_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtph_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtpd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvtpd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtpd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_cvtpd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepi32_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepi32_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepi8_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepi16_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi16_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_cvtepi16_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_cvtepi64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_cvtepu64_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepi64_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu32_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtpd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtpd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_cvtpd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtph_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepi16_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu16_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu32_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu8_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu16_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu8_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvtepu8_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi16_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi32_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi64_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi32_storeu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi64_storeu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvtepi64_storeu_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepi16_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepi32_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepi64_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepi32_storeu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepi64_storeu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvtepi64_storeu_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi16_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi32_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi64_storeu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi32_storeu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi64_storeu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvtepi64_storeu_epi32)

EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
