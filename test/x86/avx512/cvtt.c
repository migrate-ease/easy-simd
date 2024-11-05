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

#define EASYSIMD_TEST_X86_AVX512_INSN cvtt

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/cvtt.h>

static int
test_easysimd_mm_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const int64_t r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   548.43), EASYSIMD_FLOAT64_C(  -160.32) },
      {  INT64_C(                 548), -INT64_C(                 160) } },
    { { EASYSIMD_FLOAT64_C(  -665.23), EASYSIMD_FLOAT64_C(   909.86) },
      { -INT64_C(                 665),  INT64_C(                 909) } },
    { { EASYSIMD_FLOAT64_C(   869.84), EASYSIMD_FLOAT64_C(  -522.84) },
      {  INT64_C(                 869), -INT64_C(                 522) } },
    { { EASYSIMD_FLOAT64_C(  -396.75), EASYSIMD_FLOAT64_C(  -885.22) },
      { -INT64_C(                 396), -INT64_C(                 885) } },
    { { EASYSIMD_FLOAT64_C(   670.62), EASYSIMD_FLOAT64_C(  -665.50) },
      {  INT64_C(                 670), -INT64_C(                 665) } },
    { { EASYSIMD_FLOAT64_C(    66.13), EASYSIMD_FLOAT64_C(   606.16) },
      {  INT64_C(                  66),  INT64_C(                 606) } },
    { { EASYSIMD_FLOAT64_C(   -31.40), EASYSIMD_FLOAT64_C(  -401.49) },
      { -INT64_C(                  31), -INT64_C(                 401) } },
    { { EASYSIMD_FLOAT64_C(    27.98), EASYSIMD_FLOAT64_C(   872.01) },
      {  INT64_C(                  27),  INT64_C(                 872) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cvttpd_epi64(a);
    }
    EASYSIMD_TEST_PERF_END("_mm_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128i r = easysimd_mm_cvttpd_epi64(a);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C(  361962842543030048), -INT64_C( 3474737081876049297) },
      UINT8_C(108),
      { EASYSIMD_FLOAT64_C(   957.75), EASYSIMD_FLOAT64_C(   751.50) },
      { -INT64_C(  361962842543030048), -INT64_C( 3474737081876049297) } },
    { {  INT64_C(  688050963011175950), -INT64_C( 7209247915320387714) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT64_C(   -23.92), EASYSIMD_FLOAT64_C(  -116.15) },
      { -INT64_C(                  23), -INT64_C( 7209247915320387714) } },
    { { -INT64_C( 1696959715454980277), -INT64_C( 8986700034522357505) },
      UINT8_C(136),
      { EASYSIMD_FLOAT64_C(    40.92), EASYSIMD_FLOAT64_C(  -333.21) },
      { -INT64_C( 1696959715454980277), -INT64_C( 8986700034522357505) } },
    { { -INT64_C( 3906979509015529327), -INT64_C( 7568519742650574791) },
      UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(   414.99), EASYSIMD_FLOAT64_C(  -679.12) },
      { -INT64_C( 3906979509015529327), -INT64_C( 7568519742650574791) } },
    { {  INT64_C( 1324577026109679169), -INT64_C( 6354291571612128933) },
      UINT8_C(219),
      { EASYSIMD_FLOAT64_C(   727.53), EASYSIMD_FLOAT64_C(   -94.39) },
      {  INT64_C(                 727), -INT64_C(                  94) } },
    { {  INT64_C( 7972755038995392676), -INT64_C( 3786475889184342912) },
      UINT8_C(133),
      { EASYSIMD_FLOAT64_C(   -53.71), EASYSIMD_FLOAT64_C(   557.50) },
      { -INT64_C(                  53), -INT64_C( 3786475889184342912) } },
    { { -INT64_C( 2204071043036436841),  INT64_C( 6065779050755933240) },
      UINT8_C(112),
      { EASYSIMD_FLOAT64_C(   377.02), EASYSIMD_FLOAT64_C(  -369.85) },
      { -INT64_C( 2204071043036436841),  INT64_C( 6065779050755933240) } },
    { { -INT64_C( 7696596639603648289), -INT64_C( 5674106389503363330) },
      UINT8_C(151),
      { EASYSIMD_FLOAT64_C(  -383.32), EASYSIMD_FLOAT64_C(   295.30) },
      { -INT64_C(                 383),  INT64_C(                 295) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi64(test_vec[i].src);
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvttpd_epi64(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128i r = easysimd_mm_mask_cvttpd_epi64(src, k, a);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(120),
      { EASYSIMD_FLOAT64_C(  -637.12), EASYSIMD_FLOAT64_C(   286.78) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(201),
      { EASYSIMD_FLOAT64_C(    14.68), EASYSIMD_FLOAT64_C(  -536.46) },
      {  INT64_C(                  14),  INT64_C(                   0) } },
    { UINT8_C(194),
      { EASYSIMD_FLOAT64_C(     4.38), EASYSIMD_FLOAT64_C(  -326.23) },
      {  INT64_C(                   0), -INT64_C(                 326) } },
    { UINT8_C( 50),
      { EASYSIMD_FLOAT64_C(  -618.60), EASYSIMD_FLOAT64_C(   303.92) },
      {  INT64_C(                   0),  INT64_C(                 303) } },
    { UINT8_C( 17),
      { EASYSIMD_FLOAT64_C(  -934.04), EASYSIMD_FLOAT64_C(  -855.69) },
      { -INT64_C(                 934),  INT64_C(                   0) } },
    { UINT8_C(134),
      { EASYSIMD_FLOAT64_C(  -621.04), EASYSIMD_FLOAT64_C(   304.82) },
      {  INT64_C(                   0),  INT64_C(                 304) } },
    { UINT8_C(182),
      { EASYSIMD_FLOAT64_C(  -152.60), EASYSIMD_FLOAT64_C(   172.70) },
      {  INT64_C(                   0),  INT64_C(                 172) } },
    { UINT8_C(108),
      { EASYSIMD_FLOAT64_C(  -737.88), EASYSIMD_FLOAT64_C(  -401.93) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvttpd_epi64(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128i r = easysimd_mm_maskz_cvttpd_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const int64_t r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   964.24), EASYSIMD_FLOAT64_C(   196.46), EASYSIMD_FLOAT64_C(  -802.10), EASYSIMD_FLOAT64_C(   792.02) },
      {  INT64_C(                 964),  INT64_C(                 196), -INT64_C(                 802),  INT64_C(                 792) } },
    { { EASYSIMD_FLOAT64_C(  -782.21), EASYSIMD_FLOAT64_C(  -964.01), EASYSIMD_FLOAT64_C(   725.47), EASYSIMD_FLOAT64_C(    42.47) },
      { -INT64_C(                 782), -INT64_C(                 964),  INT64_C(                 725),  INT64_C(                  42) } },
    { { EASYSIMD_FLOAT64_C(   245.73), EASYSIMD_FLOAT64_C(   551.11), EASYSIMD_FLOAT64_C(   248.64), EASYSIMD_FLOAT64_C(   446.39) },
      {  INT64_C(                 245),  INT64_C(                 551),  INT64_C(                 248),  INT64_C(                 446) } },
    { { EASYSIMD_FLOAT64_C(  -413.24), EASYSIMD_FLOAT64_C(   232.92), EASYSIMD_FLOAT64_C(  -605.20), EASYSIMD_FLOAT64_C(  -401.61) },
      { -INT64_C(                 413),  INT64_C(                 232), -INT64_C(                 605), -INT64_C(                 401) } },
    { { EASYSIMD_FLOAT64_C(  -197.92), EASYSIMD_FLOAT64_C(   442.33), EASYSIMD_FLOAT64_C(    52.50), EASYSIMD_FLOAT64_C(  -365.91) },
      { -INT64_C(                 197),  INT64_C(                 442),  INT64_C(                  52), -INT64_C(                 365) } },
    { { EASYSIMD_FLOAT64_C(   134.25), EASYSIMD_FLOAT64_C(  -257.54), EASYSIMD_FLOAT64_C(  -209.31), EASYSIMD_FLOAT64_C(   368.37) },
      {  INT64_C(                 134), -INT64_C(                 257), -INT64_C(                 209),  INT64_C(                 368) } },
    { { EASYSIMD_FLOAT64_C(  -145.34), EASYSIMD_FLOAT64_C(  -657.50), EASYSIMD_FLOAT64_C(  -512.09), EASYSIMD_FLOAT64_C(  -475.04) },
      { -INT64_C(                 145), -INT64_C(                 657), -INT64_C(                 512), -INT64_C(                 475) } },
    { { EASYSIMD_FLOAT64_C(   129.13), EASYSIMD_FLOAT64_C(    21.86), EASYSIMD_FLOAT64_C(  -398.80), EASYSIMD_FLOAT64_C(    93.37) },
      {  INT64_C(                 129),  INT64_C(                  21), -INT64_C(                 398),  INT64_C(                  93) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cvttpd_epi64(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_x_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256i r = easysimd_mm256_cvttpd_epi64(a);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 4998618497395861255),  INT64_C( 6083523638670096080),  INT64_C( 5361254335933275451),  INT64_C( 7506626946315849653) },
      UINT8_C(189),
      { EASYSIMD_FLOAT64_C(  -457.66), EASYSIMD_FLOAT64_C(   779.06), EASYSIMD_FLOAT64_C(   248.03), EASYSIMD_FLOAT64_C(   419.24) },
      { -INT64_C(                 457),  INT64_C( 6083523638670096080),  INT64_C(                 248),  INT64_C(                 419) } },
    { {  INT64_C( 1216764696101154024),  INT64_C(  721734871190026046), -INT64_C( 4898698309276840314), -INT64_C( 7117890056492585601) },
      UINT8_C(152),
      { EASYSIMD_FLOAT64_C(  -735.77), EASYSIMD_FLOAT64_C(   594.57), EASYSIMD_FLOAT64_C(  -763.54), EASYSIMD_FLOAT64_C(   146.56) },
      {  INT64_C( 1216764696101154024),  INT64_C(  721734871190026046), -INT64_C( 4898698309276840314),  INT64_C(                 146) } },
    { {  INT64_C( 2170519512181421278),  INT64_C( 4351948285911771456), -INT64_C( 4080423321534258883), -INT64_C( 3056532411830502236) },
      UINT8_C(183),
      { EASYSIMD_FLOAT64_C(  -366.29), EASYSIMD_FLOAT64_C(  -272.27), EASYSIMD_FLOAT64_C(   540.01), EASYSIMD_FLOAT64_C(   633.75) },
      { -INT64_C(                 366), -INT64_C(                 272),  INT64_C(                 540), -INT64_C( 3056532411830502236) } },
    { { -INT64_C( 5612822720813083383),  INT64_C( 9106729326957516709), -INT64_C( 8364215261705656071), -INT64_C( 5290890995007503636) },
      UINT8_C(207),
      { EASYSIMD_FLOAT64_C(   635.80), EASYSIMD_FLOAT64_C(    49.91), EASYSIMD_FLOAT64_C(   594.20), EASYSIMD_FLOAT64_C(   548.36) },
      {  INT64_C(                 635),  INT64_C(                  49),  INT64_C(                 594),  INT64_C(                 548) } },
    { { -INT64_C( 8976255796272104957),  INT64_C( 7205883710757727056), -INT64_C( 3905620542911116194),  INT64_C( 7983721515828412932) },
      UINT8_C(218),
      { EASYSIMD_FLOAT64_C(   602.50), EASYSIMD_FLOAT64_C(  -312.23), EASYSIMD_FLOAT64_C(   142.93), EASYSIMD_FLOAT64_C(   960.76) },
      { -INT64_C( 8976255796272104957), -INT64_C(                 312), -INT64_C( 3905620542911116194),  INT64_C(                 960) } },
    { {  INT64_C( 3832469562967803702),  INT64_C( 6198558254894912682),  INT64_C( 5074892013637979775), -INT64_C( 2719359830323045522) },
      UINT8_C( 43),
      { EASYSIMD_FLOAT64_C(  -442.57), EASYSIMD_FLOAT64_C(   -11.97), EASYSIMD_FLOAT64_C(  -999.94), EASYSIMD_FLOAT64_C(   159.14) },
      { -INT64_C(                 442), -INT64_C(                  11),  INT64_C( 5074892013637979775),  INT64_C(                 159) } },
    { {  INT64_C( 7612490430529676075), -INT64_C(  560157924770842108), -INT64_C( 8874123574862512040), -INT64_C( 2694104004787457526) },
      UINT8_C(193),
      { EASYSIMD_FLOAT64_C(  -469.12), EASYSIMD_FLOAT64_C(   -24.20), EASYSIMD_FLOAT64_C(   737.05), EASYSIMD_FLOAT64_C(   284.98) },
      { -INT64_C(                 469), -INT64_C(  560157924770842108), -INT64_C( 8874123574862512040), -INT64_C( 2694104004787457526) } },
    { { -INT64_C( 5364565841412562627), -INT64_C( 5434364889390021004),  INT64_C( 7456338970469603609),  INT64_C( 3143689480681575640) },
      UINT8_C(130),
      { EASYSIMD_FLOAT64_C(  -199.47), EASYSIMD_FLOAT64_C(   -88.38), EASYSIMD_FLOAT64_C(  -176.48), EASYSIMD_FLOAT64_C(  -118.36) },
      { -INT64_C( 5364565841412562627), -INT64_C(                  88),  INT64_C( 7456338970469603609),  INT64_C( 3143689480681575640) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi64(test_vec[i].src);
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cvttpd_epi64(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_x_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256i r = easysimd_mm256_mask_cvttpd_epi64(src, k, a);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(153),
      { EASYSIMD_FLOAT64_C(   959.73), EASYSIMD_FLOAT64_C(  -336.46), EASYSIMD_FLOAT64_C(   525.44), EASYSIMD_FLOAT64_C(  -349.74) },
      {  INT64_C(                 959),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                 349) } },
    { UINT8_C( 66),
      { EASYSIMD_FLOAT64_C(    20.84), EASYSIMD_FLOAT64_C(   -40.03), EASYSIMD_FLOAT64_C(   563.05), EASYSIMD_FLOAT64_C(  -425.37) },
      {  INT64_C(                   0), -INT64_C(                  40),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 43),
      { EASYSIMD_FLOAT64_C(   499.28), EASYSIMD_FLOAT64_C(  -787.44), EASYSIMD_FLOAT64_C(   649.19), EASYSIMD_FLOAT64_C(  -884.92) },
      {  INT64_C(                 499), -INT64_C(                 787),  INT64_C(                   0), -INT64_C(                 884) } },
    { UINT8_C( 27),
      { EASYSIMD_FLOAT64_C(   535.07), EASYSIMD_FLOAT64_C(  -643.93), EASYSIMD_FLOAT64_C(  -339.93), EASYSIMD_FLOAT64_C(  -109.23) },
      {  INT64_C(                 535), -INT64_C(                 643),  INT64_C(                   0), -INT64_C(                 109) } },
    { UINT8_C( 18),
      { EASYSIMD_FLOAT64_C(   855.99), EASYSIMD_FLOAT64_C(   521.28), EASYSIMD_FLOAT64_C(  -357.55), EASYSIMD_FLOAT64_C(  -687.61) },
      {  INT64_C(                   0),  INT64_C(                 521),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 49),
      { EASYSIMD_FLOAT64_C(   271.31), EASYSIMD_FLOAT64_C(   530.67), EASYSIMD_FLOAT64_C(   574.40), EASYSIMD_FLOAT64_C(  -658.05) },
      {  INT64_C(                 271),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(241),
      { EASYSIMD_FLOAT64_C(  -379.34), EASYSIMD_FLOAT64_C(  -698.32), EASYSIMD_FLOAT64_C(   -20.38), EASYSIMD_FLOAT64_C(  -853.90) },
      { -INT64_C(                 379),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 85),
      { EASYSIMD_FLOAT64_C(  -394.93), EASYSIMD_FLOAT64_C(   166.94), EASYSIMD_FLOAT64_C(   911.91), EASYSIMD_FLOAT64_C(  -831.89) },
      { -INT64_C(                 394),  INT64_C(                   0),  INT64_C(                 911),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_cvttpd_epi64(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_x_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256i r = easysimd_mm256_maskz_cvttpd_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[8];
    const int64_t r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -102.63), EASYSIMD_FLOAT64_C(  -450.87), EASYSIMD_FLOAT64_C(  -985.58), EASYSIMD_FLOAT64_C(  -692.01),
        EASYSIMD_FLOAT64_C(  -691.07), EASYSIMD_FLOAT64_C(   737.41), EASYSIMD_FLOAT64_C(  -932.55), EASYSIMD_FLOAT64_C(   193.74) },
      { -INT64_C(                 102), -INT64_C(                 450), -INT64_C(                 985), -INT64_C(                 692),
        -INT64_C(                 691),  INT64_C(                 737), -INT64_C(                 932),  INT64_C(                 193) } },
    { { EASYSIMD_FLOAT64_C(  -506.85), EASYSIMD_FLOAT64_C(  -404.92), EASYSIMD_FLOAT64_C(   593.80), EASYSIMD_FLOAT64_C(  -385.89),
        EASYSIMD_FLOAT64_C(   488.60), EASYSIMD_FLOAT64_C(  -746.23), EASYSIMD_FLOAT64_C(  -199.55), EASYSIMD_FLOAT64_C(   284.38) },
      { -INT64_C(                 506), -INT64_C(                 404),  INT64_C(                 593), -INT64_C(                 385),
         INT64_C(                 488), -INT64_C(                 746), -INT64_C(                 199),  INT64_C(                 284) } },
    { { EASYSIMD_FLOAT64_C(  -194.82), EASYSIMD_FLOAT64_C(   945.34), EASYSIMD_FLOAT64_C(   882.03), EASYSIMD_FLOAT64_C(  -125.46),
        EASYSIMD_FLOAT64_C(  -481.47), EASYSIMD_FLOAT64_C(  -265.93), EASYSIMD_FLOAT64_C(   872.36), EASYSIMD_FLOAT64_C(   663.87) },
      { -INT64_C(                 194),  INT64_C(                 945),  INT64_C(                 882), -INT64_C(                 125),
        -INT64_C(                 481), -INT64_C(                 265),  INT64_C(                 872),  INT64_C(                 663) } },
    { { EASYSIMD_FLOAT64_C(    94.47), EASYSIMD_FLOAT64_C(  -362.86), EASYSIMD_FLOAT64_C(  -972.84), EASYSIMD_FLOAT64_C(  -521.03),
        EASYSIMD_FLOAT64_C(  -693.17), EASYSIMD_FLOAT64_C(  -657.90), EASYSIMD_FLOAT64_C(   434.78), EASYSIMD_FLOAT64_C(   204.20) },
      {  INT64_C(                  94), -INT64_C(                 362), -INT64_C(                 972), -INT64_C(                 521),
        -INT64_C(                 693), -INT64_C(                 657),  INT64_C(                 434),  INT64_C(                 204) } },
    { { EASYSIMD_FLOAT64_C(  -108.77), EASYSIMD_FLOAT64_C(   449.20), EASYSIMD_FLOAT64_C(   512.19), EASYSIMD_FLOAT64_C(   200.16),
        EASYSIMD_FLOAT64_C(   186.61), EASYSIMD_FLOAT64_C(   579.64), EASYSIMD_FLOAT64_C(  -606.10), EASYSIMD_FLOAT64_C(   679.76) },
      { -INT64_C(                 108),  INT64_C(                 449),  INT64_C(                 512),  INT64_C(                 200),
         INT64_C(                 186),  INT64_C(                 579), -INT64_C(                 606),  INT64_C(                 679) } },
    { { EASYSIMD_FLOAT64_C(  -825.28), EASYSIMD_FLOAT64_C(   987.69), EASYSIMD_FLOAT64_C(  -706.13), EASYSIMD_FLOAT64_C(   663.32),
        EASYSIMD_FLOAT64_C(  -758.53), EASYSIMD_FLOAT64_C(    94.32), EASYSIMD_FLOAT64_C(   -52.30), EASYSIMD_FLOAT64_C(    46.64) },
      { -INT64_C(                 825),  INT64_C(                 987), -INT64_C(                 706),  INT64_C(                 663),
        -INT64_C(                 758),  INT64_C(                  94), -INT64_C(                  52),  INT64_C(                  46) } },
    { { EASYSIMD_FLOAT64_C(    39.66), EASYSIMD_FLOAT64_C(  -170.27), EASYSIMD_FLOAT64_C(   921.18), EASYSIMD_FLOAT64_C(   558.19),
        EASYSIMD_FLOAT64_C(   563.80), EASYSIMD_FLOAT64_C(   793.54), EASYSIMD_FLOAT64_C(   222.06), EASYSIMD_FLOAT64_C(  -341.72) },
      {  INT64_C(                  39), -INT64_C(                 170),  INT64_C(                 921),  INT64_C(                 558),
         INT64_C(                 563),  INT64_C(                 793),  INT64_C(                 222), -INT64_C(                 341) } },
    { { EASYSIMD_FLOAT64_C(  -569.31), EASYSIMD_FLOAT64_C(   249.22), EASYSIMD_FLOAT64_C(   137.24), EASYSIMD_FLOAT64_C(  -262.49),
        EASYSIMD_FLOAT64_C(   591.31), EASYSIMD_FLOAT64_C(  -427.98), EASYSIMD_FLOAT64_C(   941.72), EASYSIMD_FLOAT64_C(  -517.46) },
      { -INT64_C(                 569),  INT64_C(                 249),  INT64_C(                 137), -INT64_C(                 262),
         INT64_C(                 591), -INT64_C(                 427),  INT64_C(                 941), -INT64_C(                 517) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvttpd_epi64(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512i r = easysimd_mm512_cvttpd_epi64(a);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 7338432075860755761),  INT64_C( 1949455613941636163), -INT64_C( 4767102332944251491), -INT64_C(  989665187246774309),
        -INT64_C( 1962887047264516283), -INT64_C( 7065778380137683896), -INT64_C( 7871754800921977515), -INT64_C( 7169085921176430897) },
      UINT8_C(145),
      { EASYSIMD_FLOAT64_C(   234.34), EASYSIMD_FLOAT64_C(   166.18), EASYSIMD_FLOAT64_C(  -923.95), EASYSIMD_FLOAT64_C(   486.93),
        EASYSIMD_FLOAT64_C(  -514.13), EASYSIMD_FLOAT64_C(  -197.21), EASYSIMD_FLOAT64_C(  -188.32), EASYSIMD_FLOAT64_C(  -630.58) },
      {  INT64_C(                 234),  INT64_C( 1949455613941636163), -INT64_C( 4767102332944251491), -INT64_C(  989665187246774309),
        -INT64_C(                 514), -INT64_C( 7065778380137683896), -INT64_C( 7871754800921977515), -INT64_C(                 630) } },
    { {  INT64_C( 8214139744022405721),  INT64_C( 1820785498011468429),  INT64_C( 7970257904709129408),  INT64_C( 4584592839680380420),
        -INT64_C( 8690285940416414111),  INT64_C(  631700528605990376),  INT64_C( 4889131557401604884), -INT64_C( 1539488312140248195) },
      UINT8_C( 33),
      { EASYSIMD_FLOAT64_C(    48.41), EASYSIMD_FLOAT64_C(   251.12), EASYSIMD_FLOAT64_C(  -713.33), EASYSIMD_FLOAT64_C(  -427.99),
        EASYSIMD_FLOAT64_C(  -811.37), EASYSIMD_FLOAT64_C(  -639.37), EASYSIMD_FLOAT64_C(   605.69), EASYSIMD_FLOAT64_C(   484.19) },
      {  INT64_C(                  48),  INT64_C( 1820785498011468429),  INT64_C( 7970257904709129408),  INT64_C( 4584592839680380420),
        -INT64_C( 8690285940416414111), -INT64_C(                 639),  INT64_C( 4889131557401604884), -INT64_C( 1539488312140248195) } },
    { { -INT64_C(  363345953079819360),  INT64_C(  315952611464929333), -INT64_C( 1061309752802425472), -INT64_C( 6598414449590050473),
        -INT64_C( 2342581358393194564), -INT64_C( 5469861217306363942),  INT64_C( 8305514145287546091),  INT64_C( 1812554116095779966) },
      UINT8_C(  4),
      { EASYSIMD_FLOAT64_C(   701.11), EASYSIMD_FLOAT64_C(   652.88), EASYSIMD_FLOAT64_C(  -990.38), EASYSIMD_FLOAT64_C(   609.90),
        EASYSIMD_FLOAT64_C(   -14.12), EASYSIMD_FLOAT64_C(  -551.25), EASYSIMD_FLOAT64_C(   -34.75), EASYSIMD_FLOAT64_C(  -487.94) },
      { -INT64_C(  363345953079819360),  INT64_C(  315952611464929333), -INT64_C(                 990), -INT64_C( 6598414449590050473),
        -INT64_C( 2342581358393194564), -INT64_C( 5469861217306363942),  INT64_C( 8305514145287546091),  INT64_C( 1812554116095779966) } },
    { { -INT64_C( 3575091562142231727), -INT64_C( 7265327680970836847), -INT64_C( 9084303787891839379),  INT64_C( 8905770586033593517),
        -INT64_C( 1823019596753232710),  INT64_C( 7682227252978032137),  INT64_C( 6108361751476344174), -INT64_C( 3494034561797755426) },
      UINT8_C( 78),
      { EASYSIMD_FLOAT64_C(  -433.75), EASYSIMD_FLOAT64_C(   655.03), EASYSIMD_FLOAT64_C(  -919.47), EASYSIMD_FLOAT64_C(   534.25),
        EASYSIMD_FLOAT64_C(  -747.65), EASYSIMD_FLOAT64_C(  -501.04), EASYSIMD_FLOAT64_C(   204.14), EASYSIMD_FLOAT64_C(  -152.42) },
      { -INT64_C( 3575091562142231727),  INT64_C(                 655), -INT64_C(                 919),  INT64_C(                 534),
        -INT64_C( 1823019596753232710),  INT64_C( 7682227252978032137),  INT64_C(                 204), -INT64_C( 3494034561797755426) } },
    { {  INT64_C(  453855799240938525), -INT64_C( 4404801545444749121), -INT64_C( 2441893924569618917),  INT64_C( 7477236838036979801),
        -INT64_C( 1168807039204014768), -INT64_C( 8126517428471434464),  INT64_C( 1152211998010576822),  INT64_C( 8136477444657882426) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT64_C(   673.06), EASYSIMD_FLOAT64_C(   790.77), EASYSIMD_FLOAT64_C(   915.75), EASYSIMD_FLOAT64_C(  -210.81),
        EASYSIMD_FLOAT64_C(   391.17), EASYSIMD_FLOAT64_C(  -200.11), EASYSIMD_FLOAT64_C(  -884.85), EASYSIMD_FLOAT64_C(   798.07) },
      {  INT64_C(                 673), -INT64_C( 4404801545444749121), -INT64_C( 2441893924569618917), -INT64_C(                 210),
         INT64_C(                 391), -INT64_C( 8126517428471434464), -INT64_C(                 884),  INT64_C( 8136477444657882426) } },
    { { -INT64_C( 6994249039636048279), -INT64_C( 5728189450457772339),  INT64_C( 2170244568510891057), -INT64_C( 8442071029536413838),
        -INT64_C(  918538444974344429),  INT64_C( 2845074258260562880), -INT64_C(   14343624449045568),  INT64_C( 8889051682985126578) },
      UINT8_C(248),
      { EASYSIMD_FLOAT64_C(  -609.13), EASYSIMD_FLOAT64_C(   -14.95), EASYSIMD_FLOAT64_C(   186.62), EASYSIMD_FLOAT64_C(   236.79),
        EASYSIMD_FLOAT64_C(   132.13), EASYSIMD_FLOAT64_C(   -32.20), EASYSIMD_FLOAT64_C(   326.82), EASYSIMD_FLOAT64_C(  -358.91) },
      { -INT64_C( 6994249039636048279), -INT64_C( 5728189450457772339),  INT64_C( 2170244568510891057),  INT64_C(                 236),
         INT64_C(                 132), -INT64_C(                  32),  INT64_C(                 326), -INT64_C(                 358) } },
    { {  INT64_C( 2578327103399746636), -INT64_C( 1245335164768258053),  INT64_C(  455376924124285464), -INT64_C( 3585312244967706385),
         INT64_C( 3368866465801708997),  INT64_C( 3911634791949430549),  INT64_C( 6945361448064294517),  INT64_C( 7539847810740205905) },
      UINT8_C( 16),
      { EASYSIMD_FLOAT64_C(  -396.13), EASYSIMD_FLOAT64_C(  -266.27), EASYSIMD_FLOAT64_C(  -920.33), EASYSIMD_FLOAT64_C(   115.98),
        EASYSIMD_FLOAT64_C(  -762.99), EASYSIMD_FLOAT64_C(  -823.70), EASYSIMD_FLOAT64_C(   -12.08), EASYSIMD_FLOAT64_C(  -981.10) },
      {  INT64_C( 2578327103399746636), -INT64_C( 1245335164768258053),  INT64_C(  455376924124285464), -INT64_C( 3585312244967706385),
        -INT64_C(                 762),  INT64_C( 3911634791949430549),  INT64_C( 6945361448064294517),  INT64_C( 7539847810740205905) } },
    { {  INT64_C( 7784342001227564710), -INT64_C(  122655169319181543),  INT64_C( 4243808951984005098),  INT64_C( 4613794447188109326),
        -INT64_C(  270539355094623741),  INT64_C( 4827650271321922364), -INT64_C( 5233749388566224798),  INT64_C( 3458643759733583396) },
      UINT8_C(236),
      { EASYSIMD_FLOAT64_C(   282.80), EASYSIMD_FLOAT64_C(   185.96), EASYSIMD_FLOAT64_C(   762.60), EASYSIMD_FLOAT64_C(  -784.87),
        EASYSIMD_FLOAT64_C(  -769.36), EASYSIMD_FLOAT64_C(  -639.78), EASYSIMD_FLOAT64_C(   450.74), EASYSIMD_FLOAT64_C(  -481.64) },
      {  INT64_C( 7784342001227564710), -INT64_C(  122655169319181543),  INT64_C(                 762), -INT64_C(                 784),
        -INT64_C(  270539355094623741), -INT64_C(                 639),  INT64_C(                 450), -INT64_C(                 481) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cvttpd_epi64(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512i r = easysimd_mm512_mask_cvttpd_epi64(src, k, a);

    easysimd_test_x86_write_i64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_cvttpd_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(193),
      { EASYSIMD_FLOAT64_C(  -780.21), EASYSIMD_FLOAT64_C(  -667.39), EASYSIMD_FLOAT64_C(   667.29), EASYSIMD_FLOAT64_C(  -640.55),
        EASYSIMD_FLOAT64_C(   602.24), EASYSIMD_FLOAT64_C(   253.79), EASYSIMD_FLOAT64_C(   640.74), EASYSIMD_FLOAT64_C(  -457.44) },
      { -INT64_C(                 780),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                 640), -INT64_C(                 457) } },
    { UINT8_C(137),
      { EASYSIMD_FLOAT64_C(  -103.85), EASYSIMD_FLOAT64_C(    13.47), EASYSIMD_FLOAT64_C(    21.68), EASYSIMD_FLOAT64_C(  -140.59),
        EASYSIMD_FLOAT64_C(    94.65), EASYSIMD_FLOAT64_C(   -77.07), EASYSIMD_FLOAT64_C(   767.53), EASYSIMD_FLOAT64_C(   -37.60) },
      { -INT64_C(                 103),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                 140),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                  37) } },
    { UINT8_C(193),
      { EASYSIMD_FLOAT64_C(   639.41), EASYSIMD_FLOAT64_C(   756.26), EASYSIMD_FLOAT64_C(   809.62), EASYSIMD_FLOAT64_C(   -63.95),
        EASYSIMD_FLOAT64_C(    39.05), EASYSIMD_FLOAT64_C(    -4.43), EASYSIMD_FLOAT64_C(  -301.34), EASYSIMD_FLOAT64_C(   254.18) },
      {  INT64_C(                 639),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C(                 301),  INT64_C(                 254) } },
    { UINT8_C( 49),
      { EASYSIMD_FLOAT64_C(    58.87), EASYSIMD_FLOAT64_C(  -295.08), EASYSIMD_FLOAT64_C(   744.58), EASYSIMD_FLOAT64_C(  -527.50),
        EASYSIMD_FLOAT64_C(   -75.29), EASYSIMD_FLOAT64_C(  -922.82), EASYSIMD_FLOAT64_C(  -860.21), EASYSIMD_FLOAT64_C(   284.16) },
      {  INT64_C(                  58),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                  75), -INT64_C(                 922),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 49),
      { EASYSIMD_FLOAT64_C(   393.58), EASYSIMD_FLOAT64_C(   -75.10), EASYSIMD_FLOAT64_C(  -778.02), EASYSIMD_FLOAT64_C(  -102.45),
        EASYSIMD_FLOAT64_C(   821.05), EASYSIMD_FLOAT64_C(   235.45), EASYSIMD_FLOAT64_C(   919.23), EASYSIMD_FLOAT64_C(  -319.54) },
      {  INT64_C(                 393),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                 821),  INT64_C(                 235),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(110),
      { EASYSIMD_FLOAT64_C(  -157.84), EASYSIMD_FLOAT64_C(  -552.01), EASYSIMD_FLOAT64_C(   292.50), EASYSIMD_FLOAT64_C(    37.77),
        EASYSIMD_FLOAT64_C(  -912.60), EASYSIMD_FLOAT64_C(    48.75), EASYSIMD_FLOAT64_C(  -152.61), EASYSIMD_FLOAT64_C(    23.45) },
      {  INT64_C(                   0), -INT64_C(                 552),  INT64_C(                 292),  INT64_C(                  37),
         INT64_C(                   0),  INT64_C(                  48), -INT64_C(                 152),  INT64_C(                   0) } },
    { UINT8_C(254),
      { EASYSIMD_FLOAT64_C(   842.96), EASYSIMD_FLOAT64_C(   722.11), EASYSIMD_FLOAT64_C(   341.98), EASYSIMD_FLOAT64_C(    69.18),
        EASYSIMD_FLOAT64_C(  -219.02), EASYSIMD_FLOAT64_C(  -953.09), EASYSIMD_FLOAT64_C(  -186.25), EASYSIMD_FLOAT64_C(   253.48) },
      {  INT64_C(                   0),  INT64_C(                 722),  INT64_C(                 341),  INT64_C(                  69),
        -INT64_C(                 219), -INT64_C(                 953), -INT64_C(                 186),  INT64_C(                 253) } },
    { UINT8_C(229),
      { EASYSIMD_FLOAT64_C(  -109.06), EASYSIMD_FLOAT64_C(   393.27), EASYSIMD_FLOAT64_C(  -744.22), EASYSIMD_FLOAT64_C(  -429.64),
        EASYSIMD_FLOAT64_C(  -213.14), EASYSIMD_FLOAT64_C(   180.68), EASYSIMD_FLOAT64_C(  -207.66), EASYSIMD_FLOAT64_C(   684.40) },
      { -INT64_C(                 109),  INT64_C(                   0), -INT64_C(                 744),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                 180), -INT64_C(                 207),  INT64_C(                 684) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_cvttpd_epi64(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_cvttpd_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(-EASYSIMD_FLOAT64_C(1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512i r = easysimd_mm512_maskz_cvttpd_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cvttps_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(  1146110046), -INT32_C(  1263803033), -INT32_C(  1115973896), -INT32_C(  1683927264) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT32_C(   885.34), EASYSIMD_FLOAT32_C(  -424.41), EASYSIMD_FLOAT32_C(  -503.24), EASYSIMD_FLOAT32_C(  -644.46) },
      {  INT32_C(         885), -INT32_C(  1263803033), -INT32_C(  1115973896), -INT32_C(         644) } },
    { { -INT32_C(  2059349295),  INT32_C(   434711025),  INT32_C(   247209551), -INT32_C(  1284081715) },
      UINT8_C( 34),
      { EASYSIMD_FLOAT32_C(   595.96), EASYSIMD_FLOAT32_C(   772.68), EASYSIMD_FLOAT32_C(   229.59), EASYSIMD_FLOAT32_C(   757.98) },
      { -INT32_C(  2059349295),  INT32_C(         772),  INT32_C(   247209551), -INT32_C(  1284081715) } },
    { {  INT32_C(   183818601), -INT32_C(  1946792588), -INT32_C(  1570970672),  INT32_C(  1284825379) },
      UINT8_C(203),
      { EASYSIMD_FLOAT32_C(   274.42), EASYSIMD_FLOAT32_C(   368.59), EASYSIMD_FLOAT32_C(  -196.38), EASYSIMD_FLOAT32_C(  -164.49) },
      {  INT32_C(         274),  INT32_C(         368), -INT32_C(  1570970672), -INT32_C(         164) } },
    { {  INT32_C(   551663530),  INT32_C(   357172074), -INT32_C(  1786843204),  INT32_C(   822708451) },
      UINT8_C(126),
      { EASYSIMD_FLOAT32_C(  -499.89), EASYSIMD_FLOAT32_C(   690.19), EASYSIMD_FLOAT32_C(     4.78), EASYSIMD_FLOAT32_C(   -90.42) },
      {  INT32_C(   551663530),  INT32_C(         690),  INT32_C(           4), -INT32_C(          90) } },
    { {  INT32_C(   953386148),  INT32_C(  1826987729),  INT32_C(  1041649543),  INT32_C(   866661936) },
      UINT8_C(128),
      { EASYSIMD_FLOAT32_C(  -992.15), EASYSIMD_FLOAT32_C(  -378.11), EASYSIMD_FLOAT32_C(  -373.75), EASYSIMD_FLOAT32_C(  -343.01) },
      {  INT32_C(   953386148),  INT32_C(  1826987729),  INT32_C(  1041649543),  INT32_C(   866661936) } },
    { { -INT32_C(  1882893435), -INT32_C(  2061220988),  INT32_C(   690558373), -INT32_C(  2030411288) },
      UINT8_C( 71),
      { EASYSIMD_FLOAT32_C(   898.22), EASYSIMD_FLOAT32_C(  -967.02), EASYSIMD_FLOAT32_C(   992.61), EASYSIMD_FLOAT32_C(   371.97) },
      {  INT32_C(         898), -INT32_C(         967),  INT32_C(         992), -INT32_C(  2030411288) } },
    { { -INT32_C(   172702132), -INT32_C(   374131463), -INT32_C(   143659100), -INT32_C(   126091596) },
      UINT8_C( 34),
      { EASYSIMD_FLOAT32_C(  -939.79), EASYSIMD_FLOAT32_C(  -571.98), EASYSIMD_FLOAT32_C(   694.61), EASYSIMD_FLOAT32_C(   924.55) },
      { -INT32_C(   172702132), -INT32_C(         571), -INT32_C(   143659100), -INT32_C(   126091596) } },
    { { -INT32_C(  1047846714), -INT32_C(  1272393050),  INT32_C(   771859816), -INT32_C(  1909918119) },
      UINT8_C(169),
      { EASYSIMD_FLOAT32_C(  -786.01), EASYSIMD_FLOAT32_C(   344.97), EASYSIMD_FLOAT32_C(   379.70), EASYSIMD_FLOAT32_C(  -209.63) },
      { -INT32_C(         786), -INT32_C(  1272393050),  INT32_C(   771859816), -INT32_C(         209) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cvttps_epi32(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cvttps_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(-EASYSIMD_FLOAT32_C(1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128i r = easysimd_mm_mask_cvttps_epi32(src, k, a);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_cvttps_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 41),
      { EASYSIMD_FLOAT32_C(  -337.12), EASYSIMD_FLOAT32_C(   102.88), EASYSIMD_FLOAT32_C(   793.37), EASYSIMD_FLOAT32_C(  -983.91) },
      { -INT32_C(         337),  INT32_C(           0),  INT32_C(           0), -INT32_C(         983) } },
    { UINT8_C(160),
      { EASYSIMD_FLOAT32_C(   853.58), EASYSIMD_FLOAT32_C(  -555.89), EASYSIMD_FLOAT32_C(   346.26), EASYSIMD_FLOAT32_C(   778.13) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(168),
      { EASYSIMD_FLOAT32_C(   197.17), EASYSIMD_FLOAT32_C(   148.32), EASYSIMD_FLOAT32_C(   131.13), EASYSIMD_FLOAT32_C(  -259.08) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(         259) } },
    { UINT8_C( 44),
      { EASYSIMD_FLOAT32_C(  -338.12), EASYSIMD_FLOAT32_C(  -682.36), EASYSIMD_FLOAT32_C(  -515.95), EASYSIMD_FLOAT32_C(   564.61) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(         515),  INT32_C(         564) } },
    { UINT8_C( 84),
      { EASYSIMD_FLOAT32_C(  -719.72), EASYSIMD_FLOAT32_C(  -952.74), EASYSIMD_FLOAT32_C(   553.01), EASYSIMD_FLOAT32_C(   974.45) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(         553),  INT32_C(           0) } },
    { UINT8_C( 30),
      { EASYSIMD_FLOAT32_C(    32.14), EASYSIMD_FLOAT32_C(  -811.56), EASYSIMD_FLOAT32_C(   992.92), EASYSIMD_FLOAT32_C(  -588.16) },
      {  INT32_C(           0), -INT32_C(         811),  INT32_C(         992), -INT32_C(         588) } },
    { UINT8_C(125),
      { EASYSIMD_FLOAT32_C(  -153.18), EASYSIMD_FLOAT32_C(    74.72), EASYSIMD_FLOAT32_C(  -918.31), EASYSIMD_FLOAT32_C(  -359.81) },
      { -INT32_C(         153),  INT32_C(           0), -INT32_C(         918), -INT32_C(         359) } },
    { UINT8_C( 29),
      { EASYSIMD_FLOAT32_C(   733.35), EASYSIMD_FLOAT32_C(  -506.23), EASYSIMD_FLOAT32_C(   534.93), EASYSIMD_FLOAT32_C(    79.61) },
      {  INT32_C(         733),  INT32_C(           0),  INT32_C(         534),  INT32_C(          79) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_cvttps_epi32(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_cvttps_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(-EASYSIMD_FLOAT32_C(1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128i r = easysimd_mm_maskz_cvttps_epi32(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cvttps_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   479555711), -INT32_C(  1929140051),  INT32_C(   443682856),  INT32_C(    82608531),  INT32_C(  1263353145), -INT32_C(  1257217526),  INT32_C(  1496390636), -INT32_C(  1115414725) },
      UINT8_C(175),
      { EASYSIMD_FLOAT32_C(   489.40), EASYSIMD_FLOAT32_C(   209.54), EASYSIMD_FLOAT32_C(  -734.21), EASYSIMD_FLOAT32_C(    97.55),
        EASYSIMD_FLOAT32_C(  -933.20), EASYSIMD_FLOAT32_C(   169.77), EASYSIMD_FLOAT32_C(   770.20), EASYSIMD_FLOAT32_C(   133.33) },
      {  INT32_C(         489),  INT32_C(         209), -INT32_C(         734),  INT32_C(          97),  INT32_C(  1263353145),  INT32_C(         169),  INT32_C(  1496390636),  INT32_C(         133) } },
    { { -INT32_C(   582559396),  INT32_C(    35015621), -INT32_C(    99786344), -INT32_C(  1914256782),  INT32_C(   315113715), -INT32_C(  1295940156), -INT32_C(   934797787),  INT32_C(  1669072391) },
      UINT8_C(223),
      { EASYSIMD_FLOAT32_C(   104.53), EASYSIMD_FLOAT32_C(  -652.36), EASYSIMD_FLOAT32_C(   333.41), EASYSIMD_FLOAT32_C(  -808.59),
        EASYSIMD_FLOAT32_C(   -11.83), EASYSIMD_FLOAT32_C(  -721.84), EASYSIMD_FLOAT32_C(   875.69), EASYSIMD_FLOAT32_C(    -0.55) },
      {  INT32_C(         104), -INT32_C(         652),  INT32_C(         333), -INT32_C(         808), -INT32_C(          11), -INT32_C(  1295940156),  INT32_C(         875),  INT32_C(           0) } },
    { {  INT32_C(  1982504884), -INT32_C(   966150266),  INT32_C(    92961664),  INT32_C(  1529560125), -INT32_C(  1973226620),  INT32_C(   845858415),  INT32_C(  1564413446),  INT32_C(  1763108277) },
      UINT8_C(132),
      { EASYSIMD_FLOAT32_C(  -821.48), EASYSIMD_FLOAT32_C(  -782.23), EASYSIMD_FLOAT32_C(  -143.59), EASYSIMD_FLOAT32_C(   788.83),
        EASYSIMD_FLOAT32_C(   722.54), EASYSIMD_FLOAT32_C(   355.18), EASYSIMD_FLOAT32_C(    24.63), EASYSIMD_FLOAT32_C(  -933.32) },
      {  INT32_C(  1982504884), -INT32_C(   966150266), -INT32_C(         143),  INT32_C(  1529560125), -INT32_C(  1973226620),  INT32_C(   845858415),  INT32_C(  1564413446), -INT32_C(         933) } },
    { { -INT32_C(  1778287270), -INT32_C(  1675993944), -INT32_C(  2079611458),  INT32_C(   495664399),  INT32_C(  1658054781),  INT32_C(  1072053502),  INT32_C(  1681387804),  INT32_C(   472428481) },
      UINT8_C( 44),
      { EASYSIMD_FLOAT32_C(  -823.32), EASYSIMD_FLOAT32_C(  -610.58), EASYSIMD_FLOAT32_C(  -907.48), EASYSIMD_FLOAT32_C(  -664.74),
        EASYSIMD_FLOAT32_C(   702.96), EASYSIMD_FLOAT32_C(  -531.21), EASYSIMD_FLOAT32_C(   677.15), EASYSIMD_FLOAT32_C(   212.97) },
      { -INT32_C(  1778287270), -INT32_C(  1675993944), -INT32_C(         907), -INT32_C(         664),  INT32_C(  1658054781), -INT32_C(         531),  INT32_C(  1681387804),  INT32_C(   472428481) } },
    { { -INT32_C(  1165899652),  INT32_C(  1010277971), -INT32_C(  1371891343),  INT32_C(  1892317567),  INT32_C(  1597124272), -INT32_C(  2138288553), -INT32_C(   855219968),  INT32_C(  1312770770) },
      UINT8_C( 18),
      { EASYSIMD_FLOAT32_C(    24.08), EASYSIMD_FLOAT32_C(   290.93), EASYSIMD_FLOAT32_C(  -321.56), EASYSIMD_FLOAT32_C(  -951.66),
        EASYSIMD_FLOAT32_C(   -23.11), EASYSIMD_FLOAT32_C(  -375.84), EASYSIMD_FLOAT32_C(   -14.56), EASYSIMD_FLOAT32_C(     0.13) },
      { -INT32_C(  1165899652),  INT32_C(         290), -INT32_C(  1371891343),  INT32_C(  1892317567), -INT32_C(          23), -INT32_C(  2138288553), -INT32_C(   855219968),  INT32_C(  1312770770) } },
    { {  INT32_C(  1448640220),  INT32_C(  1996933192),  INT32_C(  1221486330),  INT32_C(  1397313266), -INT32_C(  1641736620),  INT32_C(   380728150), -INT32_C(  1149954180), -INT32_C(  1785444679) },
      UINT8_C(196),
      { EASYSIMD_FLOAT32_C(   899.52), EASYSIMD_FLOAT32_C(  -752.28), EASYSIMD_FLOAT32_C(   368.90), EASYSIMD_FLOAT32_C(   455.27),
        EASYSIMD_FLOAT32_C(  -955.74), EASYSIMD_FLOAT32_C(   819.03), EASYSIMD_FLOAT32_C(   754.99), EASYSIMD_FLOAT32_C(   598.27) },
      {  INT32_C(  1448640220),  INT32_C(  1996933192),  INT32_C(         368),  INT32_C(  1397313266), -INT32_C(  1641736620),  INT32_C(   380728150),  INT32_C(         754),  INT32_C(         598) } },
    { { -INT32_C(  1605699758),  INT32_C(  1475714368),  INT32_C(   900568002),  INT32_C(  1538376516),  INT32_C(  2132045113), -INT32_C(   314332671), -INT32_C(  2019471211),  INT32_C(   652169428) },
      UINT8_C( 72),
      { EASYSIMD_FLOAT32_C(   164.40), EASYSIMD_FLOAT32_C(  -102.70), EASYSIMD_FLOAT32_C(   695.46), EASYSIMD_FLOAT32_C(   472.59),
        EASYSIMD_FLOAT32_C(  -344.85), EASYSIMD_FLOAT32_C(  -720.37), EASYSIMD_FLOAT32_C(   498.01), EASYSIMD_FLOAT32_C(   556.28) },
      { -INT32_C(  1605699758),  INT32_C(  1475714368),  INT32_C(   900568002),  INT32_C(         472),  INT32_C(  2132045113), -INT32_C(   314332671),  INT32_C(         498),  INT32_C(   652169428) } },
    { {  INT32_C(  1351925388), -INT32_C(   594940049), -INT32_C(  1361246204), -INT32_C(  1673278901), -INT32_C(  1133458580), -INT32_C(   737896534),  INT32_C(   446336350), -INT32_C(   110550164) },
      UINT8_C(225),
      { EASYSIMD_FLOAT32_C(   941.72), EASYSIMD_FLOAT32_C(   360.78), EASYSIMD_FLOAT32_C(   261.56), EASYSIMD_FLOAT32_C(     0.42),
        EASYSIMD_FLOAT32_C(   269.87), EASYSIMD_FLOAT32_C(  -970.79), EASYSIMD_FLOAT32_C(   963.02), EASYSIMD_FLOAT32_C(   755.08) },
      {  INT32_C(         941), -INT32_C(   594940049), -INT32_C(  1361246204), -INT32_C(  1673278901), -INT32_C(  1133458580), -INT32_C(         970),  INT32_C(         963),  INT32_C(         755) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cvttps_epi32(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cvttps_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(-EASYSIMD_FLOAT32_C(1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256i r = easysimd_mm256_mask_cvttps_epi32(src, k, a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_cvttps_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 10),
      { EASYSIMD_FLOAT32_C(   877.69), EASYSIMD_FLOAT32_C(   -30.87), EASYSIMD_FLOAT32_C(  -292.23), EASYSIMD_FLOAT32_C(    -7.83),
        EASYSIMD_FLOAT32_C(  -719.08), EASYSIMD_FLOAT32_C(   973.47), EASYSIMD_FLOAT32_C(   350.72), EASYSIMD_FLOAT32_C(   509.47) },
      {  INT32_C(           0), -INT32_C(          30),  INT32_C(           0), -INT32_C(           7),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(253),
      { EASYSIMD_FLOAT32_C(  -448.40), EASYSIMD_FLOAT32_C(  -122.20), EASYSIMD_FLOAT32_C(   -24.22), EASYSIMD_FLOAT32_C(   -83.12),
        EASYSIMD_FLOAT32_C(   143.42), EASYSIMD_FLOAT32_C(  -116.50), EASYSIMD_FLOAT32_C(  -245.24), EASYSIMD_FLOAT32_C(  -935.80) },
      { -INT32_C(         448),  INT32_C(           0), -INT32_C(          24), -INT32_C(          83),  INT32_C(         143), -INT32_C(         116), -INT32_C(         245), -INT32_C(         935) } },
    { UINT8_C(251),
      { EASYSIMD_FLOAT32_C(  -909.36), EASYSIMD_FLOAT32_C(  -458.74), EASYSIMD_FLOAT32_C(   581.37), EASYSIMD_FLOAT32_C(   340.57),
        EASYSIMD_FLOAT32_C(  -517.02), EASYSIMD_FLOAT32_C(   -57.85), EASYSIMD_FLOAT32_C(  -397.86), EASYSIMD_FLOAT32_C(   483.40) },
      { -INT32_C(         909), -INT32_C(         458),  INT32_C(           0),  INT32_C(         340), -INT32_C(         517), -INT32_C(          57), -INT32_C(         397),  INT32_C(         483) } },
    { UINT8_C( 15),
      { EASYSIMD_FLOAT32_C(  -368.65), EASYSIMD_FLOAT32_C(   446.41), EASYSIMD_FLOAT32_C(   967.11), EASYSIMD_FLOAT32_C(   842.79),
        EASYSIMD_FLOAT32_C(   324.11), EASYSIMD_FLOAT32_C(   -63.76), EASYSIMD_FLOAT32_C(  -449.44), EASYSIMD_FLOAT32_C(  -683.72) },
      { -INT32_C(         368),  INT32_C(         446),  INT32_C(         967),  INT32_C(         842),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(208),
      { EASYSIMD_FLOAT32_C(  -475.97), EASYSIMD_FLOAT32_C(   667.00), EASYSIMD_FLOAT32_C(  -273.37), EASYSIMD_FLOAT32_C(   287.55),
        EASYSIMD_FLOAT32_C(  -781.40), EASYSIMD_FLOAT32_C(   604.43), EASYSIMD_FLOAT32_C(  -736.67), EASYSIMD_FLOAT32_C(   135.48) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(         781),  INT32_C(           0), -INT32_C(         736),  INT32_C(         135) } },
    { UINT8_C(242),
      { EASYSIMD_FLOAT32_C(   146.83), EASYSIMD_FLOAT32_C(   890.24), EASYSIMD_FLOAT32_C(  -187.95), EASYSIMD_FLOAT32_C(  -782.34),
        EASYSIMD_FLOAT32_C(   980.88), EASYSIMD_FLOAT32_C(   353.32), EASYSIMD_FLOAT32_C(   799.04), EASYSIMD_FLOAT32_C(   321.45) },
      {  INT32_C(           0),  INT32_C(         890),  INT32_C(           0),  INT32_C(           0),  INT32_C(         980),  INT32_C(         353),  INT32_C(         799),  INT32_C(         321) } },
    { UINT8_C(134),
      { EASYSIMD_FLOAT32_C(  -258.81), EASYSIMD_FLOAT32_C(   923.59), EASYSIMD_FLOAT32_C(   319.69), EASYSIMD_FLOAT32_C(   -46.79),
        EASYSIMD_FLOAT32_C(  -445.06), EASYSIMD_FLOAT32_C(  -233.89), EASYSIMD_FLOAT32_C(   -79.68), EASYSIMD_FLOAT32_C(  -602.27) },
      {  INT32_C(           0),  INT32_C(         923),  INT32_C(         319),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(         602) } },
    { UINT8_C(198),
      { EASYSIMD_FLOAT32_C(   856.56), EASYSIMD_FLOAT32_C(   -51.71), EASYSIMD_FLOAT32_C(  -593.51), EASYSIMD_FLOAT32_C(    73.72),
        EASYSIMD_FLOAT32_C(   472.32), EASYSIMD_FLOAT32_C(  -926.51), EASYSIMD_FLOAT32_C(   800.35), EASYSIMD_FLOAT32_C(  -240.13) },
      {  INT32_C(           0), -INT32_C(          51), -INT32_C(         593),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(         800), -INT32_C(         240) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_cvttps_epi32(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_cvttps_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(-EASYSIMD_FLOAT32_C(1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256i r = easysimd_mm256_maskz_cvttps_epi32(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cvttps_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const int32_t r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -708.91), EASYSIMD_FLOAT32_C(  -279.71), EASYSIMD_FLOAT32_C(   688.80), EASYSIMD_FLOAT32_C(   393.78),
        EASYSIMD_FLOAT32_C(   -25.20), EASYSIMD_FLOAT32_C(   995.98), EASYSIMD_FLOAT32_C(   981.61), EASYSIMD_FLOAT32_C(   -55.32),
        EASYSIMD_FLOAT32_C(  -354.21), EASYSIMD_FLOAT32_C(   456.73), EASYSIMD_FLOAT32_C(  -649.21), EASYSIMD_FLOAT32_C(  -657.30),
        EASYSIMD_FLOAT32_C(  -132.20), EASYSIMD_FLOAT32_C(  -124.45), EASYSIMD_FLOAT32_C(   465.69), EASYSIMD_FLOAT32_C(   471.73) },
      { -INT32_C(         708), -INT32_C(         279),  INT32_C(         688),  INT32_C(         393), -INT32_C(          25),  INT32_C(         995),  INT32_C(         981), -INT32_C(          55),
        -INT32_C(         354),  INT32_C(         456), -INT32_C(         649), -INT32_C(         657), -INT32_C(         132), -INT32_C(         124),  INT32_C(         465),  INT32_C(         471) } },
    { { EASYSIMD_FLOAT32_C(  -118.90), EASYSIMD_FLOAT32_C(  -889.17), EASYSIMD_FLOAT32_C(   669.50), EASYSIMD_FLOAT32_C(   -62.39),
        EASYSIMD_FLOAT32_C(   610.64), EASYSIMD_FLOAT32_C(  -873.27), EASYSIMD_FLOAT32_C(  -513.94), EASYSIMD_FLOAT32_C(   -14.44),
        EASYSIMD_FLOAT32_C(  -565.09), EASYSIMD_FLOAT32_C(  -379.43), EASYSIMD_FLOAT32_C(   355.45), EASYSIMD_FLOAT32_C(   323.44),
        EASYSIMD_FLOAT32_C(  -634.01), EASYSIMD_FLOAT32_C(   906.82), EASYSIMD_FLOAT32_C(    67.22), EASYSIMD_FLOAT32_C(  -342.93) },
      { -INT32_C(         118), -INT32_C(         889),  INT32_C(         669), -INT32_C(          62),  INT32_C(         610), -INT32_C(         873), -INT32_C(         513), -INT32_C(          14),
        -INT32_C(         565), -INT32_C(         379),  INT32_C(         355),  INT32_C(         323), -INT32_C(         634),  INT32_C(         906),  INT32_C(          67), -INT32_C(         342) } },
    { { EASYSIMD_FLOAT32_C(  -372.89), EASYSIMD_FLOAT32_C(  -243.98), EASYSIMD_FLOAT32_C(  -949.14), EASYSIMD_FLOAT32_C(   601.91),
        EASYSIMD_FLOAT32_C(  -248.00), EASYSIMD_FLOAT32_C(  -967.53), EASYSIMD_FLOAT32_C(  -453.41), EASYSIMD_FLOAT32_C(   397.80),
        EASYSIMD_FLOAT32_C(   489.20), EASYSIMD_FLOAT32_C(  -102.62), EASYSIMD_FLOAT32_C(   740.50), EASYSIMD_FLOAT32_C(  -643.00),
        EASYSIMD_FLOAT32_C(   772.93), EASYSIMD_FLOAT32_C(   206.18), EASYSIMD_FLOAT32_C(   828.73), EASYSIMD_FLOAT32_C(  -345.97) },
      { -INT32_C(         372), -INT32_C(         243), -INT32_C(         949),  INT32_C(         601), -INT32_C(         248), -INT32_C(         967), -INT32_C(         453),  INT32_C(         397),
         INT32_C(         489), -INT32_C(         102),  INT32_C(         740), -INT32_C(         643),  INT32_C(         772),  INT32_C(         206),  INT32_C(         828), -INT32_C(         345) } },
    { { EASYSIMD_FLOAT32_C(   317.01), EASYSIMD_FLOAT32_C(   498.24), EASYSIMD_FLOAT32_C(   591.64), EASYSIMD_FLOAT32_C(   -72.35),
        EASYSIMD_FLOAT32_C(   624.96), EASYSIMD_FLOAT32_C(  -922.30), EASYSIMD_FLOAT32_C(   913.22), EASYSIMD_FLOAT32_C(  -940.12),
        EASYSIMD_FLOAT32_C(  -301.73), EASYSIMD_FLOAT32_C(   268.66), EASYSIMD_FLOAT32_C(   383.31), EASYSIMD_FLOAT32_C(    64.26),
        EASYSIMD_FLOAT32_C(   175.49), EASYSIMD_FLOAT32_C(  -549.47), EASYSIMD_FLOAT32_C(   721.33), EASYSIMD_FLOAT32_C(   802.59) },
      {  INT32_C(         317),  INT32_C(         498),  INT32_C(         591), -INT32_C(          72),  INT32_C(         624), -INT32_C(         922),  INT32_C(         913), -INT32_C(         940),
        -INT32_C(         301),  INT32_C(         268),  INT32_C(         383),  INT32_C(          64),  INT32_C(         175), -INT32_C(         549),  INT32_C(         721),  INT32_C(         802) } },
    { { EASYSIMD_FLOAT32_C(   206.55), EASYSIMD_FLOAT32_C(   772.19), EASYSIMD_FLOAT32_C(   404.50), EASYSIMD_FLOAT32_C(   958.56),
        EASYSIMD_FLOAT32_C(   804.66), EASYSIMD_FLOAT32_C(   951.09), EASYSIMD_FLOAT32_C(   356.35), EASYSIMD_FLOAT32_C(   293.86),
        EASYSIMD_FLOAT32_C(  -151.52), EASYSIMD_FLOAT32_C(    96.85), EASYSIMD_FLOAT32_C(   650.86), EASYSIMD_FLOAT32_C(  -378.59),
        EASYSIMD_FLOAT32_C(  -696.96), EASYSIMD_FLOAT32_C(   479.60), EASYSIMD_FLOAT32_C(   275.44), EASYSIMD_FLOAT32_C(   620.05) },
      {  INT32_C(         206),  INT32_C(         772),  INT32_C(         404),  INT32_C(         958),  INT32_C(         804),  INT32_C(         951),  INT32_C(         356),  INT32_C(         293),
        -INT32_C(         151),  INT32_C(          96),  INT32_C(         650), -INT32_C(         378), -INT32_C(         696),  INT32_C(         479),  INT32_C(         275),  INT32_C(         620) } },
    { { EASYSIMD_FLOAT32_C(   -22.16), EASYSIMD_FLOAT32_C(  -132.92), EASYSIMD_FLOAT32_C(  -452.29), EASYSIMD_FLOAT32_C(  -397.20),
        EASYSIMD_FLOAT32_C(   -55.22), EASYSIMD_FLOAT32_C(  -539.08), EASYSIMD_FLOAT32_C(  -337.32), EASYSIMD_FLOAT32_C(   643.05),
        EASYSIMD_FLOAT32_C(   729.58), EASYSIMD_FLOAT32_C(  -954.01), EASYSIMD_FLOAT32_C(  -292.70), EASYSIMD_FLOAT32_C(   -94.93),
        EASYSIMD_FLOAT32_C(  -503.48), EASYSIMD_FLOAT32_C(  -571.37), EASYSIMD_FLOAT32_C(  -292.34), EASYSIMD_FLOAT32_C(   703.08) },
      { -INT32_C(          22), -INT32_C(         132), -INT32_C(         452), -INT32_C(         397), -INT32_C(          55), -INT32_C(         539), -INT32_C(         337),  INT32_C(         643),
         INT32_C(         729), -INT32_C(         954), -INT32_C(         292), -INT32_C(          94), -INT32_C(         503), -INT32_C(         571), -INT32_C(         292),  INT32_C(         703) } },
    { { EASYSIMD_FLOAT32_C(  -799.18), EASYSIMD_FLOAT32_C(  -887.84), EASYSIMD_FLOAT32_C(   661.63), EASYSIMD_FLOAT32_C(  -994.52),
        EASYSIMD_FLOAT32_C(  -936.74), EASYSIMD_FLOAT32_C(    17.99), EASYSIMD_FLOAT32_C(   299.35), EASYSIMD_FLOAT32_C(   -88.27),
        EASYSIMD_FLOAT32_C(  -885.16), EASYSIMD_FLOAT32_C(   -49.79), EASYSIMD_FLOAT32_C(   533.14), EASYSIMD_FLOAT32_C(  -582.12),
        EASYSIMD_FLOAT32_C(  -570.19), EASYSIMD_FLOAT32_C(  -191.42), EASYSIMD_FLOAT32_C(  -962.07), EASYSIMD_FLOAT32_C(   407.65) },
      { -INT32_C(         799), -INT32_C(         887),  INT32_C(         661), -INT32_C(         994), -INT32_C(         936),  INT32_C(          17),  INT32_C(         299), -INT32_C(          88),
        -INT32_C(         885), -INT32_C(          49),  INT32_C(         533), -INT32_C(         582), -INT32_C(         570), -INT32_C(         191), -INT32_C(         962),  INT32_C(         407) } },
    { { EASYSIMD_FLOAT32_C(   675.66), EASYSIMD_FLOAT32_C(  -414.37), EASYSIMD_FLOAT32_C(  -989.55), EASYSIMD_FLOAT32_C(  -379.56),
        EASYSIMD_FLOAT32_C(    46.56), EASYSIMD_FLOAT32_C(  -326.88), EASYSIMD_FLOAT32_C(  -736.52), EASYSIMD_FLOAT32_C(  -223.86),
        EASYSIMD_FLOAT32_C(  -280.89), EASYSIMD_FLOAT32_C(   -29.21), EASYSIMD_FLOAT32_C(   681.21), EASYSIMD_FLOAT32_C(   215.64),
        EASYSIMD_FLOAT32_C(   399.42), EASYSIMD_FLOAT32_C(  -611.13), EASYSIMD_FLOAT32_C(   -81.29), EASYSIMD_FLOAT32_C(   600.25) },
      {  INT32_C(         675), -INT32_C(         414), -INT32_C(         989), -INT32_C(         379),  INT32_C(          46), -INT32_C(         326), -INT32_C(         736), -INT32_C(         223),
        -INT32_C(         280), -INT32_C(          29),  INT32_C(         681),  INT32_C(         215),  INT32_C(         399), -INT32_C(         611), -INT32_C(          81),  INT32_C(         600) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cvttps_epi32(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cvttps_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(-EASYSIMD_FLOAT32_C(1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512i r = easysimd_mm512_cvttps_epi32(a);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}


static int
test_easysimd_mm512_mask_cvttps_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const easysimd_float32 a[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(   755895319),  INT32_C(   871357699), -INT32_C(  1607482520),  INT32_C(  1565485603), -INT32_C(   683729380), -INT32_C(  1002744486), -INT32_C(  1866309377), -INT32_C(   286489340),
         INT32_C(  2093014649),  INT32_C(   615559099), -INT32_C(  1765482638),  INT32_C(  1710429001), -INT32_C(  1958989263), -INT32_C(  2041612153), -INT32_C(   602467624),  INT32_C(   281674647) },
      UINT16_C(35553),
      { EASYSIMD_FLOAT32_C(  -373.84), EASYSIMD_FLOAT32_C(   255.82), EASYSIMD_FLOAT32_C(   510.57), EASYSIMD_FLOAT32_C(   747.80),
        EASYSIMD_FLOAT32_C(   969.71), EASYSIMD_FLOAT32_C(   708.49), EASYSIMD_FLOAT32_C(     0.48), EASYSIMD_FLOAT32_C(   -31.29),
        EASYSIMD_FLOAT32_C(   724.15), EASYSIMD_FLOAT32_C(  -577.85), EASYSIMD_FLOAT32_C(   660.47), EASYSIMD_FLOAT32_C(   233.22),
        EASYSIMD_FLOAT32_C(  -401.57), EASYSIMD_FLOAT32_C(  -944.58), EASYSIMD_FLOAT32_C(  -143.86), EASYSIMD_FLOAT32_C(  -424.12) },
      { -INT32_C(         373),  INT32_C(   871357699), -INT32_C(  1607482520),  INT32_C(  1565485603), -INT32_C(   683729380),  INT32_C(         708),  INT32_C(           0), -INT32_C(          31),
         INT32_C(  2093014649), -INT32_C(         577), -INT32_C(  1765482638),  INT32_C(         233), -INT32_C(  1958989263), -INT32_C(  2041612153), -INT32_C(   602467624), -INT32_C(         424) } },
    { { -INT32_C(  1551962284), -INT32_C(  1917495690), -INT32_C(    40874701), -INT32_C(   360156579), -INT32_C(   802701041), -INT32_C(  1353366675),  INT32_C(  1927671208),  INT32_C(   175721910),
        -INT32_C(  1297221828), -INT32_C(  2126552242),  INT32_C(   226414768),  INT32_C(  1375209026),  INT32_C(   891363272),  INT32_C(   199587427),  INT32_C(   427739747),  INT32_C(  1394865943) },
      UINT16_C(53486),
      { EASYSIMD_FLOAT32_C(  -491.54), EASYSIMD_FLOAT32_C(  -126.25), EASYSIMD_FLOAT32_C(  -342.89), EASYSIMD_FLOAT32_C(   238.29),
        EASYSIMD_FLOAT32_C(   820.19), EASYSIMD_FLOAT32_C(  -621.17), EASYSIMD_FLOAT32_C(  -287.54), EASYSIMD_FLOAT32_C(  -219.70),
        EASYSIMD_FLOAT32_C(   597.00), EASYSIMD_FLOAT32_C(   328.20), EASYSIMD_FLOAT32_C(  -924.11), EASYSIMD_FLOAT32_C(   -62.22),
        EASYSIMD_FLOAT32_C(  -658.68), EASYSIMD_FLOAT32_C(   217.53), EASYSIMD_FLOAT32_C(  -851.04), EASYSIMD_FLOAT32_C(   721.52) },
      { -INT32_C(  1551962284), -INT32_C(         126), -INT32_C(         342),  INT32_C(         238), -INT32_C(   802701041), -INT32_C(         621), -INT32_C(         287), -INT32_C(         219),
        -INT32_C(  1297221828), -INT32_C(  2126552242),  INT32_C(   226414768),  INT32_C(  1375209026), -INT32_C(         658),  INT32_C(   199587427), -INT32_C(         851),  INT32_C(         721) } },
    { {  INT32_C(   624913216), -INT32_C(   168844681), -INT32_C(   554957125),  INT32_C(  1605360217), -INT32_C(   727326185), -INT32_C(  1223640378), -INT32_C(  1163898094), -INT32_C(  1652250531),
        -INT32_C(  1950170348),  INT32_C(   562082150),  INT32_C(   285240503),  INT32_C(  1567665990),  INT32_C(  1462900113), -INT32_C(   502381872), -INT32_C(   207835499),  INT32_C(  1804607575) },
      UINT16_C(21476),
      { EASYSIMD_FLOAT32_C(  -496.73), EASYSIMD_FLOAT32_C(   105.73), EASYSIMD_FLOAT32_C(  -190.93), EASYSIMD_FLOAT32_C(   703.07),
        EASYSIMD_FLOAT32_C(  -233.35), EASYSIMD_FLOAT32_C(   923.11), EASYSIMD_FLOAT32_C(   509.46), EASYSIMD_FLOAT32_C(   983.72),
        EASYSIMD_FLOAT32_C(   346.64), EASYSIMD_FLOAT32_C(   254.34), EASYSIMD_FLOAT32_C(   971.09), EASYSIMD_FLOAT32_C(   588.12),
        EASYSIMD_FLOAT32_C(   811.43), EASYSIMD_FLOAT32_C(   385.86), EASYSIMD_FLOAT32_C(   885.85), EASYSIMD_FLOAT32_C(     0.94) },
      {  INT32_C(   624913216), -INT32_C(   168844681), -INT32_C(         190),  INT32_C(  1605360217), -INT32_C(   727326185),  INT32_C(         923),  INT32_C(         509),  INT32_C(         983),
         INT32_C(         346),  INT32_C(         254),  INT32_C(   285240503),  INT32_C(  1567665990),  INT32_C(         811), -INT32_C(   502381872),  INT32_C(         885),  INT32_C(  1804607575) } },
    { {  INT32_C(   318382851), -INT32_C(  1581215739),  INT32_C(   331487107),  INT32_C(  2019993218),  INT32_C(  1542351856), -INT32_C(   188231641),  INT32_C(  2133975545), -INT32_C(  1858632563),
        -INT32_C(  1398591065),  INT32_C(  1145988033), -INT32_C(    44560261), -INT32_C(  1502167370),  INT32_C(  1342268457),  INT32_C(   776259637),  INT32_C(   934114986), -INT32_C(  1563892486) },
      UINT16_C(27413),
      { EASYSIMD_FLOAT32_C(   936.30), EASYSIMD_FLOAT32_C(   609.60), EASYSIMD_FLOAT32_C(  -823.87), EASYSIMD_FLOAT32_C(  -892.75),
        EASYSIMD_FLOAT32_C(     7.92), EASYSIMD_FLOAT32_C(   -88.00), EASYSIMD_FLOAT32_C(   756.71), EASYSIMD_FLOAT32_C(  -685.93),
        EASYSIMD_FLOAT32_C(   475.51), EASYSIMD_FLOAT32_C(   243.29), EASYSIMD_FLOAT32_C(   453.28), EASYSIMD_FLOAT32_C(  -630.07),
        EASYSIMD_FLOAT32_C(   672.72), EASYSIMD_FLOAT32_C(   234.66), EASYSIMD_FLOAT32_C(   298.32), EASYSIMD_FLOAT32_C(   811.63) },
      {  INT32_C(         936), -INT32_C(  1581215739), -INT32_C(         823),  INT32_C(  2019993218),  INT32_C(           7), -INT32_C(   188231641),  INT32_C(  2133975545), -INT32_C(  1858632563),
         INT32_C(         475),  INT32_C(         243), -INT32_C(    44560261), -INT32_C(         630),  INT32_C(  1342268457),  INT32_C(         234),  INT32_C(         298), -INT32_C(  1563892486) } },
    { { -INT32_C(   254650452),  INT32_C(   845577349),  INT32_C(  2081841588),  INT32_C(  1374169859),  INT32_C(   518960898),  INT32_C(  1217501697),  INT32_C(    84329469), -INT32_C(   888199137),
         INT32_C(    79422079),  INT32_C(   305537630), -INT32_C(  2037429117),  INT32_C(  2060941175),  INT32_C(   798541102),  INT32_C(  1568090464),  INT32_C(   207781357),  INT32_C(   618099365) },
      UINT16_C(37460),
      { EASYSIMD_FLOAT32_C(     8.55), EASYSIMD_FLOAT32_C(  -929.76), EASYSIMD_FLOAT32_C(   380.88), EASYSIMD_FLOAT32_C(   950.96),
        EASYSIMD_FLOAT32_C(   294.15), EASYSIMD_FLOAT32_C(   872.31), EASYSIMD_FLOAT32_C(   300.83), EASYSIMD_FLOAT32_C(  -591.52),
        EASYSIMD_FLOAT32_C(  -410.97), EASYSIMD_FLOAT32_C(   760.41), EASYSIMD_FLOAT32_C(  -495.06), EASYSIMD_FLOAT32_C(   467.32),
        EASYSIMD_FLOAT32_C(   -60.68), EASYSIMD_FLOAT32_C(   759.66), EASYSIMD_FLOAT32_C(  -385.57), EASYSIMD_FLOAT32_C(   845.47) },
      { -INT32_C(   254650452),  INT32_C(   845577349),  INT32_C(         380),  INT32_C(  1374169859),  INT32_C(         294),  INT32_C(  1217501697),  INT32_C(         300), -INT32_C(   888199137),
         INT32_C(    79422079),  INT32_C(         760), -INT32_C(  2037429117),  INT32_C(  2060941175), -INT32_C(          60),  INT32_C(  1568090464),  INT32_C(   207781357),  INT32_C(         845) } },
    { { -INT32_C(  1621247448),  INT32_C(  2048739863),  INT32_C(   770490966),  INT32_C(   264257767), -INT32_C(  1217497870),  INT32_C(  1762334892), -INT32_C(   687942086), -INT32_C(   150316593),
         INT32_C(   731342867),  INT32_C(   161854642),  INT32_C(  1563857270), -INT32_C(  1016203567),  INT32_C(   393993067),  INT32_C(   780175091),  INT32_C(   738492253), -INT32_C(   349958184) },
      UINT16_C(47991),
      { EASYSIMD_FLOAT32_C(  -622.08), EASYSIMD_FLOAT32_C(  -166.96), EASYSIMD_FLOAT32_C(  -110.93), EASYSIMD_FLOAT32_C(   449.40),
        EASYSIMD_FLOAT32_C(   799.85), EASYSIMD_FLOAT32_C(   731.80), EASYSIMD_FLOAT32_C(  -731.36), EASYSIMD_FLOAT32_C(  -345.20),
        EASYSIMD_FLOAT32_C(  -698.41), EASYSIMD_FLOAT32_C(   368.76), EASYSIMD_FLOAT32_C(    30.07), EASYSIMD_FLOAT32_C(   607.01),
        EASYSIMD_FLOAT32_C(  -179.99), EASYSIMD_FLOAT32_C(  -693.46), EASYSIMD_FLOAT32_C(   599.02), EASYSIMD_FLOAT32_C(   916.53) },
      { -INT32_C(         622), -INT32_C(         166), -INT32_C(         110),  INT32_C(   264257767),  INT32_C(         799),  INT32_C(         731), -INT32_C(         731), -INT32_C(   150316593),
        -INT32_C(         698),  INT32_C(         368),  INT32_C(  1563857270),  INT32_C(         607), -INT32_C(         179), -INT32_C(         693),  INT32_C(   738492253),  INT32_C(         916) } },
    { {  INT32_C(  1659010785), -INT32_C(  1310638164), -INT32_C(  1866417812), -INT32_C(  1152698460), -INT32_C(  1820935584), -INT32_C(   486751329), -INT32_C(  1047372832),  INT32_C(   119348518),
         INT32_C(  1214840988), -INT32_C(  1409725889), -INT32_C(  1472481021),  INT32_C(  1365477104), -INT32_C(   521873088),  INT32_C(  2109923484),  INT32_C(  1631475003),  INT32_C(   241785970) },
      UINT16_C(53852),
      { EASYSIMD_FLOAT32_C(   683.89), EASYSIMD_FLOAT32_C(  -492.48), EASYSIMD_FLOAT32_C(    65.43), EASYSIMD_FLOAT32_C(   438.73),
        EASYSIMD_FLOAT32_C(   970.58), EASYSIMD_FLOAT32_C(  -535.68), EASYSIMD_FLOAT32_C(   860.59), EASYSIMD_FLOAT32_C(   667.06),
        EASYSIMD_FLOAT32_C(  -520.71), EASYSIMD_FLOAT32_C(   222.00), EASYSIMD_FLOAT32_C(  -614.27), EASYSIMD_FLOAT32_C(   488.26),
        EASYSIMD_FLOAT32_C(  -644.01), EASYSIMD_FLOAT32_C(    15.66), EASYSIMD_FLOAT32_C(   -59.36), EASYSIMD_FLOAT32_C(    62.43) },
      {  INT32_C(  1659010785), -INT32_C(  1310638164),  INT32_C(          65),  INT32_C(         438),  INT32_C(         970), -INT32_C(   486751329),  INT32_C(         860),  INT32_C(   119348518),
         INT32_C(  1214840988),  INT32_C(         222), -INT32_C(  1472481021),  INT32_C(  1365477104), -INT32_C(         644),  INT32_C(  2109923484), -INT32_C(          59),  INT32_C(          62) } },
    { { -INT32_C(   334520023),  INT32_C(  1564560158),  INT32_C(   364491692),  INT32_C(   434640322), -INT32_C(   144112208), -INT32_C(   293965019), -INT32_C(  1189510290), -INT32_C(   152494388),
        -INT32_C(   555550528), -INT32_C(   264559805), -INT32_C(  1744440106), -INT32_C(  1162744567),  INT32_C(   414260210), -INT32_C(    66704498),  INT32_C(  2075533230), -INT32_C(    26108098) },
      UINT16_C(21399),
      { EASYSIMD_FLOAT32_C(  -960.09), EASYSIMD_FLOAT32_C(  -671.92), EASYSIMD_FLOAT32_C(   599.16), EASYSIMD_FLOAT32_C(  -588.72),
        EASYSIMD_FLOAT32_C(   921.43), EASYSIMD_FLOAT32_C(   719.08), EASYSIMD_FLOAT32_C(   242.42), EASYSIMD_FLOAT32_C(   131.90),
        EASYSIMD_FLOAT32_C(   683.66), EASYSIMD_FLOAT32_C(  -619.09), EASYSIMD_FLOAT32_C(   324.08), EASYSIMD_FLOAT32_C(   176.79),
        EASYSIMD_FLOAT32_C(  -665.73), EASYSIMD_FLOAT32_C(   563.40), EASYSIMD_FLOAT32_C(   637.18), EASYSIMD_FLOAT32_C(    73.08) },
      { -INT32_C(         960), -INT32_C(         671),  INT32_C(         599),  INT32_C(   434640322),  INT32_C(         921), -INT32_C(   293965019), -INT32_C(  1189510290),  INT32_C(         131),
         INT32_C(         683), -INT32_C(         619), -INT32_C(  1744440106), -INT32_C(  1162744567), -INT32_C(         665), -INT32_C(    66704498),  INT32_C(         637), -INT32_C(    26108098) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cvttps_epi32(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cvttps_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(-EASYSIMD_FLOAT32_C(1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512i r = easysimd_mm512_mask_cvttps_epi32(src, k, a);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_cvttps_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const easysimd_float32 a[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(16584),
      { EASYSIMD_FLOAT32_C(  -735.60), EASYSIMD_FLOAT32_C(  -944.28), EASYSIMD_FLOAT32_C(  -736.44), EASYSIMD_FLOAT32_C(  -706.05),
        EASYSIMD_FLOAT32_C(  -723.73), EASYSIMD_FLOAT32_C(  -992.47), EASYSIMD_FLOAT32_C(  -822.60), EASYSIMD_FLOAT32_C(   916.46),
        EASYSIMD_FLOAT32_C(   219.78), EASYSIMD_FLOAT32_C(   203.38), EASYSIMD_FLOAT32_C(  -918.64), EASYSIMD_FLOAT32_C(   896.41),
        EASYSIMD_FLOAT32_C(  -121.83), EASYSIMD_FLOAT32_C(  -878.73), EASYSIMD_FLOAT32_C(  -775.52), EASYSIMD_FLOAT32_C(  -522.68) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(         706),  INT32_C(           0),  INT32_C(           0), -INT32_C(         822),  INT32_C(         916),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(         775),  INT32_C(           0) } },
    { UINT16_C(53022),
      { EASYSIMD_FLOAT32_C(  -803.59), EASYSIMD_FLOAT32_C(   774.98), EASYSIMD_FLOAT32_C(   277.81), EASYSIMD_FLOAT32_C(   880.07),
        EASYSIMD_FLOAT32_C(  -844.11), EASYSIMD_FLOAT32_C(  -398.11), EASYSIMD_FLOAT32_C(    56.86), EASYSIMD_FLOAT32_C(  -509.84),
        EASYSIMD_FLOAT32_C(  -834.71), EASYSIMD_FLOAT32_C(  -305.96), EASYSIMD_FLOAT32_C(   563.24), EASYSIMD_FLOAT32_C(  -520.53),
        EASYSIMD_FLOAT32_C(   619.49), EASYSIMD_FLOAT32_C(   827.64), EASYSIMD_FLOAT32_C(  -464.81), EASYSIMD_FLOAT32_C(   883.05) },
      {  INT32_C(           0),  INT32_C(         774),  INT32_C(         277),  INT32_C(         880), -INT32_C(         844),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(         834), -INT32_C(         305),  INT32_C(         563), -INT32_C(         520),  INT32_C(           0),  INT32_C(           0), -INT32_C(         464),  INT32_C(         883) } },
    { UINT16_C(37787),
      { EASYSIMD_FLOAT32_C(   890.58), EASYSIMD_FLOAT32_C(  -701.01), EASYSIMD_FLOAT32_C(  -272.08), EASYSIMD_FLOAT32_C(   110.37),
        EASYSIMD_FLOAT32_C(   502.38), EASYSIMD_FLOAT32_C(  -190.72), EASYSIMD_FLOAT32_C(     6.77), EASYSIMD_FLOAT32_C(  -619.46),
        EASYSIMD_FLOAT32_C(   -69.45), EASYSIMD_FLOAT32_C(   231.26), EASYSIMD_FLOAT32_C(  -142.13), EASYSIMD_FLOAT32_C(   463.11),
        EASYSIMD_FLOAT32_C(   377.17), EASYSIMD_FLOAT32_C(    54.28), EASYSIMD_FLOAT32_C(   238.08), EASYSIMD_FLOAT32_C(  -345.01) },
      {  INT32_C(         890), -INT32_C(         701),  INT32_C(           0),  INT32_C(         110),  INT32_C(         502),  INT32_C(           0),  INT32_C(           0), -INT32_C(         619),
        -INT32_C(          69),  INT32_C(         231),  INT32_C(           0),  INT32_C(           0),  INT32_C(         377),  INT32_C(           0),  INT32_C(           0), -INT32_C(         345) } },
    { UINT16_C(31235),
      { EASYSIMD_FLOAT32_C(   256.88), EASYSIMD_FLOAT32_C(   991.20), EASYSIMD_FLOAT32_C(   884.13), EASYSIMD_FLOAT32_C(   422.16),
        EASYSIMD_FLOAT32_C(  -314.76), EASYSIMD_FLOAT32_C(   447.38), EASYSIMD_FLOAT32_C(   901.64), EASYSIMD_FLOAT32_C(  -695.27),
        EASYSIMD_FLOAT32_C(   275.02), EASYSIMD_FLOAT32_C(  -563.17), EASYSIMD_FLOAT32_C(  -812.22), EASYSIMD_FLOAT32_C(   396.61),
        EASYSIMD_FLOAT32_C(   248.29), EASYSIMD_FLOAT32_C(  -921.64), EASYSIMD_FLOAT32_C(   695.61), EASYSIMD_FLOAT32_C(   976.21) },
      {  INT32_C(         256),  INT32_C(         991),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(         563),  INT32_C(           0),  INT32_C(         396),  INT32_C(         248), -INT32_C(         921),  INT32_C(         695),  INT32_C(           0) } },
    { UINT16_C( 8889),
      { EASYSIMD_FLOAT32_C(  -214.51), EASYSIMD_FLOAT32_C(  -804.50), EASYSIMD_FLOAT32_C(   578.53), EASYSIMD_FLOAT32_C(   716.05),
        EASYSIMD_FLOAT32_C(   426.76), EASYSIMD_FLOAT32_C(  -563.60), EASYSIMD_FLOAT32_C(   179.15), EASYSIMD_FLOAT32_C(  -196.07),
        EASYSIMD_FLOAT32_C(   490.67), EASYSIMD_FLOAT32_C(  -582.76), EASYSIMD_FLOAT32_C(   458.92), EASYSIMD_FLOAT32_C(  -574.98),
        EASYSIMD_FLOAT32_C(   811.21), EASYSIMD_FLOAT32_C(  -284.21), EASYSIMD_FLOAT32_C(  -583.78), EASYSIMD_FLOAT32_C(   695.34) },
      { -INT32_C(         214),  INT32_C(           0),  INT32_C(           0),  INT32_C(         716),  INT32_C(         426), -INT32_C(         563),  INT32_C(           0), -INT32_C(         196),
         INT32_C(           0), -INT32_C(         582),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(         284),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(57318),
      { EASYSIMD_FLOAT32_C(   142.72), EASYSIMD_FLOAT32_C(  -960.40), EASYSIMD_FLOAT32_C(   406.19), EASYSIMD_FLOAT32_C(  -582.26),
        EASYSIMD_FLOAT32_C(  -523.57), EASYSIMD_FLOAT32_C(   593.97), EASYSIMD_FLOAT32_C(   814.35), EASYSIMD_FLOAT32_C(   724.72),
        EASYSIMD_FLOAT32_C(   672.34), EASYSIMD_FLOAT32_C(   509.96), EASYSIMD_FLOAT32_C(   700.93), EASYSIMD_FLOAT32_C(  -138.93),
        EASYSIMD_FLOAT32_C(  -292.06), EASYSIMD_FLOAT32_C(  -513.58), EASYSIMD_FLOAT32_C(    56.57), EASYSIMD_FLOAT32_C(  -713.53) },
      {  INT32_C(           0), -INT32_C(         960),  INT32_C(         406),  INT32_C(           0),  INT32_C(           0),  INT32_C(         593),  INT32_C(         814),  INT32_C(         724),
         INT32_C(         672),  INT32_C(         509),  INT32_C(         700), -INT32_C(         138), -INT32_C(         292),  INT32_C(           0),  INT32_C(          56), -INT32_C(         713) } },
    { UINT16_C(48857),
      { EASYSIMD_FLOAT32_C(  -277.14), EASYSIMD_FLOAT32_C(   381.62), EASYSIMD_FLOAT32_C(   287.26), EASYSIMD_FLOAT32_C(  -786.47),
        EASYSIMD_FLOAT32_C(   798.85), EASYSIMD_FLOAT32_C(  -253.82), EASYSIMD_FLOAT32_C(  -361.45), EASYSIMD_FLOAT32_C(   610.07),
        EASYSIMD_FLOAT32_C(   461.98), EASYSIMD_FLOAT32_C(    54.77), EASYSIMD_FLOAT32_C(   305.41), EASYSIMD_FLOAT32_C(   599.93),
        EASYSIMD_FLOAT32_C(  -843.76), EASYSIMD_FLOAT32_C(  -551.87), EASYSIMD_FLOAT32_C(   639.53), EASYSIMD_FLOAT32_C(   562.43) },
      { -INT32_C(         277),  INT32_C(           0),  INT32_C(           0), -INT32_C(         786),  INT32_C(         798),  INT32_C(           0), -INT32_C(         361),  INT32_C(         610),
         INT32_C(           0),  INT32_C(          54),  INT32_C(         305),  INT32_C(         599), -INT32_C(         843), -INT32_C(         551),  INT32_C(           0),  INT32_C(         562) } },
    { UINT16_C( 7864),
      { EASYSIMD_FLOAT32_C(   156.40), EASYSIMD_FLOAT32_C(  -319.78), EASYSIMD_FLOAT32_C(   840.67), EASYSIMD_FLOAT32_C(  -171.26),
        EASYSIMD_FLOAT32_C(  -809.83), EASYSIMD_FLOAT32_C(   541.60), EASYSIMD_FLOAT32_C(   689.81), EASYSIMD_FLOAT32_C(  -101.89),
        EASYSIMD_FLOAT32_C(  -971.98), EASYSIMD_FLOAT32_C(  -253.62), EASYSIMD_FLOAT32_C(   184.58), EASYSIMD_FLOAT32_C(  -769.52),
        EASYSIMD_FLOAT32_C(   229.71), EASYSIMD_FLOAT32_C(   907.44), EASYSIMD_FLOAT32_C(   612.10), EASYSIMD_FLOAT32_C(  -483.03) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(         171), -INT32_C(         809),  INT32_C(         541),  INT32_C(           0), -INT32_C(         101),
         INT32_C(           0), -INT32_C(         253),  INT32_C(         184), -INT32_C(         769),  INT32_C(         229),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_cvttps_epi32(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_cvttps_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(-EASYSIMD_FLOAT32_C(1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512i r = easysimd_mm512_maskz_cvttps_epi32(k, a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvttpd_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_cvttpd_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cvttps_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_cvttps_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cvttps_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_cvttps_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cvttps_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cvttps_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_cvttps_epi32)

EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
