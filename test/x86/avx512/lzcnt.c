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

#define EASYSIMD_TEST_X86_AVX512_INSN lzcnt

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/lzcnt.h>

static int
test_easysimd_mm_lzcnt_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi32(           INT32_MAX,            INT32_MIN,          ~INT32_C(0), INT32_C(          0)),
      easysimd_mm_set_epi32(INT32_C(          1), INT32_C(          0), INT32_C(          0), INT32_C(         32)) },
    { easysimd_mm_set_epi32(INT32_C(        179), INT32_C(     -17551), INT32_C(   -2202065), INT32_C(    -743837)),
      easysimd_mm_set_epi32(INT32_C(         24), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { easysimd_mm_set_epi32(INT32_C(      -2559), INT32_C(  388806146), INT32_C(    1927808), INT32_C(       -112)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(          3), INT32_C(         11), INT32_C(          0)) },
    { easysimd_mm_set_epi32(INT32_C(         22), INT32_C(    -189678), INT32_C(        252), INT32_C(      27703)),
      easysimd_mm_set_epi32(INT32_C(         27), INT32_C(          0), INT32_C(         24), INT32_C(         17)) },
    { easysimd_mm_set_epi32(INT32_C(   -9106380), INT32_C(    8952567), INT32_C(         -4), INT32_C(     685169)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(          8), INT32_C(          0), INT32_C(         12)) },
    { easysimd_mm_set_epi32(INT32_C(     267298), INT32_C(      -3422), INT32_C(          4), INT32_C(      31229)),
      easysimd_mm_set_epi32(INT32_C(         13), INT32_C(          0), INT32_C(         29), INT32_C(         17)) },
    { easysimd_mm_set_epi32(INT32_C(     167383), INT32_C(        214), INT32_C(          0), INT32_C(     -20257)),
      easysimd_mm_set_epi32(INT32_C(         14), INT32_C(         24), INT32_C(         32), INT32_C(          0)) },
    { easysimd_mm_set_epi32(INT32_C(       -147), INT32_C(   -1774263), INT32_C(     143922), INT32_C(    -914728)),
      easysimd_mm_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(         14), INT32_C(          0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_lzcnt_epi32(a);
    } EASYSIMD_TEST_PERF_END("_mm_lzcnt_epi32");
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_mask_lzcnt_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(  1967906480),  INT32_C(   444241631),  INT32_C(  1751328815),  INT32_C(  1797987849) },
      UINT8_C( 10),
      {  INT32_C(     5428843),  INT32_C(   517032875),  INT32_C(   698580121),  INT32_C(   725459874) },
      { -INT32_C(  1967906480),  INT32_C(           3),  INT32_C(  1751328815),  INT32_C(           2) } },
    { { -INT32_C(    69694592),  INT32_C(   153290917), -INT32_C(  1675154320), -INT32_C(   246713960) },
      UINT8_C( 13),
      {  INT32_C(       18677),  INT32_C(       21570),  INT32_C(       30363),  INT32_C(        9014) },
      {  INT32_C(          17),  INT32_C(   153290917),  INT32_C(          17),  INT32_C(          18) } },
    { { -INT32_C(  1957041304),  INT32_C(   481872372), -INT32_C(  1332916123), -INT32_C(   503559615) },
      UINT8_C(  6),
      {  INT32_C(           0),  INT32_C(           6),  INT32_C(           6),  INT32_C(           0) },
      { -INT32_C(  1957041304),  INT32_C(          29),  INT32_C(          29), -INT32_C(   503559615) } },
    { { -INT32_C(  2084109621), -INT32_C(  1412223970), -INT32_C(   638184227), -INT32_C(   244896523) },
      UINT8_C(  2),
      {  INT32_C(      315764),  INT32_C(      402356),  INT32_C(      357196),  INT32_C(      345035) },
      { -INT32_C(  2084109621),  INT32_C(          13), -INT32_C(   638184227), -INT32_C(   244896523) } },
    { { -INT32_C(   816134404),  INT32_C(  1743009649),  INT32_C(  2022955280),  INT32_C(  1643201995) },
      UINT8_C(  4),
      {  INT32_C(        4198),  INT32_C(      523483),  INT32_C(      334642),  INT32_C(      359676) },
      { -INT32_C(   816134404),  INT32_C(  1743009649),  INT32_C(          13),  INT32_C(  1643201995) } },
    { {  INT32_C(  1394876527),  INT32_C(   455882120), -INT32_C(  1637746771),  INT32_C(   707450200) },
      UINT8_C(  1),
      {  INT32_C(      820684),  INT32_C(      155800),  INT32_C(      822191),  INT32_C(      791418) },
      {  INT32_C(          12),  INT32_C(   455882120), -INT32_C(  1637746771),  INT32_C(   707450200) } },
    { { -INT32_C(  1241429622), -INT32_C(   961630619),  INT32_C(  2006648396), -INT32_C(   224101327) },
      UINT8_C(  8),
      {  INT32_C(    15383396),  INT32_C(    19320589),  INT32_C(    17462288),  INT32_C(     2498061) },
      { -INT32_C(  1241429622), -INT32_C(   961630619),  INT32_C(  2006648396),  INT32_C(          10) } },
    { {  INT32_C(  1941272773), -INT32_C(   646315458),  INT32_C(   492153721),  INT32_C(  1334390173) },
      UINT8_C( 10),
      {  INT32_C(          50),  INT32_C(         123),  INT32_C(         121),  INT32_C(          96) },
      {  INT32_C(  1941272773),  INT32_C(          25),  INT32_C(   492153721),  INT32_C(          25) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_lzcnt_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_lzcnt_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_lzcnt_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT8_C( 11),
      {  INT32_C(           6),  INT32_C(           2),  INT32_C(           1),  INT32_C(           7) },
      {  INT32_C(          29),  INT32_C(          30),  INT32_C(           0),  INT32_C(          29) } },
    { UINT8_C(  1),
      {  INT32_C(    57768613),  INT32_C(    44212542),  INT32_C(   220122657),  INT32_C(   188272304) },
      {  INT32_C(           6),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(  1),
      {  INT32_C(       15428),  INT32_C(        3147),  INT32_C(         201),  INT32_C(       13035) },
      {  INT32_C(          18),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 14),
      {  INT32_C(        7895),  INT32_C(        6520),  INT32_C(        2703),  INT32_C(        1256) },
      {  INT32_C(           0),  INT32_C(          19),  INT32_C(          20),  INT32_C(          21) } },
    { UINT8_C( 18),
      {  INT32_C(     3584232),  INT32_C(     3831674),  INT32_C(      372002),  INT32_C(     3456164) },
      {  INT32_C(           0),  INT32_C(          10),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 16),
      {  INT32_C(        1915),  INT32_C(       47708),  INT32_C(       61410),  INT32_C(       63376) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(  4),
      {  INT32_C(   576116464),  INT32_C(   682438940),  INT32_C(  1066509946),  INT32_C(  1013501310) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0) } },
    { UINT8_C( 22),
      {  INT32_C(      121955),  INT32_C(      108474),  INT32_C(      112020),  INT32_C(      114447) },
      {  INT32_C(           0),  INT32_C(          15),  INT32_C(          15),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_lzcnt_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("_mm_maskz_lzcnt_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_lzcnt_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[2];
    int64_t r[2];
  } test_vec[8] = {
    { { -INT64_C( 3472028010233954141), -INT64_C( 2566785297777978417) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C( 2702463667477614644), -INT64_C( 3037937221312375759) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C( 6075140962117264532),  INT64_C( 5479107784770555599) },
      {  INT64_C(                   1),  INT64_C(                   1) } },
    { { -INT64_C( 4820360628396514909), -INT64_C( 9169620293895923784) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C( 3851227931363781274),  INT64_C( 2065300073128052445) },
      {  INT64_C(                   2),  INT64_C(                   3) } },
    { {  INT64_C( 3051037507138145805),  INT64_C(  412873673818731853) },
      {  INT64_C(                   2),  INT64_C(                   5) } },
    { { -INT64_C( 1830548281463883012),  INT64_C( 8355000185151290204) },
      {  INT64_C(                   0),  INT64_C(                   1) } },
    { {  INT64_C( 1321354334395256506),  INT64_C( 4056060170819300057) },
      {  INT64_C(                   3),  INT64_C(                   2) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_lzcnt_epi64(a);
    } EASYSIMD_TEST_PERF_END("_mm_lzcnt_epi64");
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_si128(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_lzcnt_epi64(a);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_lzcnt_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 6817106086103308370),  INT64_C( 4272882130533940875) },
      UINT8_C(159),
      { -INT64_C( 7717995957178901082),  INT64_C(  492701464720217827) },
      {  INT64_C(                   0),  INT64_C(                   5) } },
    { {  INT64_C( 6225886013574148885),  INT64_C( 1032231857162165088) },
      UINT8_C(151),
      {  INT64_C( 1933377419216462022), -INT64_C( 4164563210667950926) },
      {  INT64_C(                   3),  INT64_C(                   0) } },
    { { -INT64_C( 8011212739364122712), -INT64_C( 6970720266259757454) },
      UINT8_C(166),
      {  INT64_C( 7051023363563229291),  INT64_C( 4532303676520594110) },
      { -INT64_C( 8011212739364122712),  INT64_C(                   2) } },
    { { -INT64_C( 5692756454433138920),  INT64_C( 6478473067756104508) },
      UINT8_C(184),
      {  INT64_C( 6202720422331187474), -INT64_C( 8087125486477225295) },
      { -INT64_C( 5692756454433138920),  INT64_C( 6478473067756104508) } },
    { { -INT64_C(  290929330467810954),  INT64_C( 3662062152663513330) },
      UINT8_C(196),
      { -INT64_C( 1216685905824151314), -INT64_C( 2363345754775959046) },
      { -INT64_C(  290929330467810954),  INT64_C( 3662062152663513330) } },
    { {  INT64_C( 2002178667399081326), -INT64_C( 6622141151485107437) },
      UINT8_C(245),
      { -INT64_C(  600525567081443271), -INT64_C( 8661437369183541336) },
      {  INT64_C(                   0), -INT64_C( 6622141151485107437) } },
    { {  INT64_C( 5823301024565863315),  INT64_C( 8656668622074553792) },
      UINT8_C(186),
      {  INT64_C( 4170015299437275317), -INT64_C( 6099463751561167773) },
      {  INT64_C( 5823301024565863315),  INT64_C(                   0) } },
    { {  INT64_C(  939398439368136644),  INT64_C(  996355825239602799) },
      UINT8_C(236),
      {  INT64_C( 6844617261676643514),  INT64_C( 3956432193314014410) },
      {  INT64_C(  939398439368136644),  INT64_C(  996355825239602799) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_si128(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_lzcnt_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("_mm_mask_lzcnt_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_si128(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_mask_lzcnt_epi64(src, k, a);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm_maskz_lzcnt_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(123),
      {  INT64_C( 8021210244332127869),  INT64_C(  763507043675292583) },
      {  INT64_C(                   1),  INT64_C(                   4) } },
    { UINT8_C( 88),
      { -INT64_C( 1463463564767798335), -INT64_C(  883834805962073860) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 33),
      {  INT64_C( 7089185449195941069), -INT64_C( 2045972298444833215) },
      {  INT64_C(                   1),  INT64_C(                   0) } },
    { UINT8_C(137),
      { -INT64_C(  818458367255869221), -INT64_C( 5690840715723959938) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 67),
      { -INT64_C( 7672904860976853980), -INT64_C(  527661742665921481) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(162),
      {  INT64_C( 5868015908171180857), -INT64_C(  953172515811494758) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(193),
      {  INT64_C(  818622926166073419),  INT64_C( 3390660509478538279) },
      {  INT64_C(                   4),  INT64_C(                   0) } },
    { UINT8_C(237),
      {  INT64_C( 4850633948508622828), -INT64_C( 8603093648899705206) },
      {  INT64_C(                   1),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_lzcnt_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("_mm_maskz_lzcnt_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_si128(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_lzcnt_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm256_lzcnt_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[8];
    int32_t r[8];
  } test_vec[8] = {
    { {  INT32_C(   556729146),  INT32_C(  1566382062),  INT32_C(  1013655925), -INT32_C(  1859253614), -INT32_C(   466415578), -INT32_C(  1535086461), -INT32_C(  1876747932), -INT32_C(  1175510401) },
      {  INT32_C(           2),  INT32_C(           1),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { {  INT32_C(   333061669), -INT32_C(  1435486411), -INT32_C(   219685792),  INT32_C(   209917414), -INT32_C(  1443776987), -INT32_C(  1957859034),  INT32_C(   152793226), -INT32_C(  1195242862) },
      {  INT32_C(           3),  INT32_C(           0),  INT32_C(           0),  INT32_C(           4),  INT32_C(           0),  INT32_C(           0),  INT32_C(           4),  INT32_C(           0) } },
    { {  INT32_C(  1590402088),  INT32_C(   856177875), -INT32_C(    31002856),  INT32_C(   705407236), -INT32_C(  2032927648), -INT32_C(   149872531),  INT32_C(   570502288),  INT32_C(  1608172342) },
      {  INT32_C(           1),  INT32_C(           2),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           1) } },
    { {  INT32_C(   868066912), -INT32_C(    93862430), -INT32_C(  1174893131), -INT32_C(  1763507402),  INT32_C(  1813886719),  INT32_C(  1717841622), -INT32_C(  1853266597), -INT32_C(  1997511896) },
      {  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   339956471),  INT32_C(   686105459), -INT32_C(   421405008), -INT32_C(   511851039),  INT32_C(  1380817531),  INT32_C(   599306696),  INT32_C(  1068843286), -INT32_C(  1362647643) },
      {  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           2),  INT32_C(           2),  INT32_C(           0) } },
    { { -INT32_C(   962952365),  INT32_C(  1458470822),  INT32_C(  1061015389),  INT32_C(   270580372),  INT32_C(   476212564),  INT32_C(   893393439),  INT32_C(    24442204), -INT32_C(   307282790) },
      {  INT32_C(           0),  INT32_C(           1),  INT32_C(           2),  INT32_C(           3),  INT32_C(           3),  INT32_C(           2),  INT32_C(           7),  INT32_C(           0) } },
    { {  INT32_C(  1723025855),  INT32_C(   649896393),  INT32_C(    90569073),  INT32_C(   118850995),  INT32_C(   304379891), -INT32_C(   297311086), -INT32_C(   202392487), -INT32_C(  1209950472) },
      {  INT32_C(           1),  INT32_C(           2),  INT32_C(           5),  INT32_C(           5),  INT32_C(           3),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(  1323461400), -INT32_C(  1479026122), -INT32_C(  2018755117), -INT32_C(  1248935230), -INT32_C(   876105159),  INT32_C(  1874398998), -INT32_C(  1016878901),  INT32_C(   796542023) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           2) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_lzcnt_epi32(a);
    } EASYSIMD_TEST_PERF_END("_mm256_lzcnt_epi32");
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_si256(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_lzcnt_epi32(a);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_lzcnt_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(   132595042), -INT32_C(   717345097),  INT32_C(  1025517876),  INT32_C(    18744806), -INT32_C(    13071532), -INT32_C(  1396765700),  INT32_C(   244291415), -INT32_C(   392951418) },
      UINT8_C( 70),
      { -INT32_C(  1442975621),  INT32_C(  1490932270), -INT32_C(   130147342), -INT32_C(  1001636039), -INT32_C(  2084549512), -INT32_C(  1562743797),  INT32_C(    86567164), -INT32_C(   129298307) },
      {  INT32_C(   132595042),  INT32_C(           1),  INT32_C(           0),  INT32_C(    18744806), -INT32_C(    13071532), -INT32_C(  1396765700),  INT32_C(           5), -INT32_C(   392951418) } },
    { {  INT32_C(   799164416),  INT32_C(   226983706), -INT32_C(   754596454),  INT32_C(  2090291460), -INT32_C(  1476372579), -INT32_C(  1068836156),  INT32_C(  1086681795), -INT32_C(  2093477757) },
      UINT8_C( 88),
      {  INT32_C(  1517466330), -INT32_C(    17531079), -INT32_C(   704395132), -INT32_C(  1217167521),  INT32_C(  1518017663), -INT32_C(   652395674),  INT32_C(   274488576),  INT32_C(  1885921173) },
      {  INT32_C(   799164416),  INT32_C(   226983706), -INT32_C(   754596454),  INT32_C(           0),  INT32_C(           1), -INT32_C(  1068836156),  INT32_C(           3), -INT32_C(  2093477757) } },
    { { -INT32_C(   875898222), -INT32_C(   557138342), -INT32_C(   424358522), -INT32_C(   862115764), -INT32_C(  1440343996),  INT32_C(  1401111379),  INT32_C(   895737760),  INT32_C(  1369820095) },
      UINT8_C(165),
      {  INT32_C(   788470895), -INT32_C(  1279992090), -INT32_C(  1174365550),  INT32_C(  1342098487),  INT32_C(   899852786),  INT32_C(   215348524),  INT32_C(   600509016),  INT32_C(   533273776) },
      {  INT32_C(           2), -INT32_C(   557138342),  INT32_C(           0), -INT32_C(   862115764), -INT32_C(  1440343996),  INT32_C(           4),  INT32_C(   895737760),  INT32_C(           3) } },
    { {  INT32_C(   525191224),  INT32_C(   970064550), -INT32_C(   722218340), -INT32_C(  1876692322), -INT32_C(   943339877),  INT32_C(   349412027),  INT32_C(  1429708453), -INT32_C(   210501446) },
      UINT8_C(201),
      { -INT32_C(   999353662), -INT32_C(  1218402076), -INT32_C(  1907018596),  INT32_C(   506062424),  INT32_C(  1188688043),  INT32_C(  1659628996),  INT32_C(   622673957),  INT32_C(  1995313332) },
      {  INT32_C(           0),  INT32_C(   970064550), -INT32_C(   722218340),  INT32_C(           3), -INT32_C(   943339877),  INT32_C(   349412027),  INT32_C(           2),  INT32_C(           1) } },
    { {  INT32_C(   104488482), -INT32_C(  1564632314),  INT32_C(   657462223), -INT32_C(  1538958855),  INT32_C(   250224458),  INT32_C(   829478156), -INT32_C(   900231915), -INT32_C(  1086306915) },
      UINT8_C(163),
      {  INT32_C(   380290683), -INT32_C(  1763357565), -INT32_C(   695267971),  INT32_C(  1897935954), -INT32_C(   193057250),  INT32_C(   738832287),  INT32_C(  1288360710), -INT32_C(  1880127212) },
      {  INT32_C(           3),  INT32_C(           0),  INT32_C(   657462223), -INT32_C(  1538958855),  INT32_C(   250224458),  INT32_C(           2), -INT32_C(   900231915),  INT32_C(           0) } },
    { { -INT32_C(   744122033),  INT32_C(  1667861222), -INT32_C(   365299305),  INT32_C(  1264278061),  INT32_C(   658495880), -INT32_C(  1890301559),  INT32_C(   819666460), -INT32_C(   138425433) },
      UINT8_C(100),
      { -INT32_C(   280311196),  INT32_C(   747023667),  INT32_C(  1096380647), -INT32_C(  1513511476),  INT32_C(   758051300),  INT32_C(  1665842757),  INT32_C(  1678408345), -INT32_C(  1630994118) },
      { -INT32_C(   744122033),  INT32_C(  1667861222),  INT32_C(           1),  INT32_C(  1264278061),  INT32_C(   658495880),  INT32_C(           1),  INT32_C(           1), -INT32_C(   138425433) } },
    { { -INT32_C(     7531573), -INT32_C(  1473571647),  INT32_C(  1357481348),  INT32_C(   251048490), -INT32_C(   398711645),  INT32_C(  2085324514),  INT32_C(   987780608),  INT32_C(   601467223) },
      UINT8_C(189),
      {  INT32_C(  2055086694), -INT32_C(   755030451), -INT32_C(  1040429297),  INT32_C(  1785006917), -INT32_C(   850637497), -INT32_C(   271726439),  INT32_C(  1380321449),  INT32_C(  1192192481) },
      {  INT32_C(           1), -INT32_C(  1473571647),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(   987780608),  INT32_C(           1) } },
    { { -INT32_C(   641561205), -INT32_C(  1028931149),  INT32_C(  1451534352), -INT32_C(    88020557), -INT32_C(   809038794),  INT32_C(  2126419157),  INT32_C(  2110784924), -INT32_C(    87695250) },
      UINT8_C(109),
      {  INT32_C(  1210176391),  INT32_C(   643359614),  INT32_C(  1356443239),  INT32_C(  2072499054),  INT32_C(   793794202),  INT32_C(   449564181),  INT32_C(  2122860958), -INT32_C(  1779662322) },
      {  INT32_C(           1), -INT32_C(  1028931149),  INT32_C(           1),  INT32_C(           1), -INT32_C(   809038794),  INT32_C(           3),  INT32_C(           1), -INT32_C(    87695250) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_si256(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_lzcnt_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_lzcnt_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_si256(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_lzcnt_epi32(src, k, a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm256_maskz_lzcnt_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(245),
      {  INT32_C(   674084170),  INT32_C(  1081764929),  INT32_C(   231425485),  INT32_C(  1971423991),  INT32_C(  2114823537), -INT32_C(   671193743), -INT32_C(   392840476),  INT32_C(  1742548765) },
      {  INT32_C(           2),  INT32_C(           0),  INT32_C(           4),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1) } },
    { UINT8_C(228),
      {  INT32_C(  1998950155),  INT32_C(  1313170953), -INT32_C(  1153019343),  INT32_C(  2016197587),  INT32_C(   787131080),  INT32_C(  1662173608), -INT32_C(  2038367657), -INT32_C(   479533096) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(118),
      { -INT32_C(   176203121), -INT32_C(   232337761), -INT32_C(   809115372), -INT32_C(  1198047730), -INT32_C(   379468504), -INT32_C(   767507240),  INT32_C(   732612164), -INT32_C(  1063154384) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0) } },
    { UINT8_C(232),
      { -INT32_C(   293096160), -INT32_C(  1123911204),  INT32_C(  2110509375),  INT32_C(   782599273), -INT32_C(  1459187995), -INT32_C(  1779574321),  INT32_C(   298195075), -INT32_C(   637958471) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           3),  INT32_C(           0) } },
    { UINT8_C( 59),
      { -INT32_C(    99104640), -INT32_C(  1673931318),  INT32_C(   604354208), -INT32_C(   385272997),  INT32_C(   330871354), -INT32_C(  1214886241), -INT32_C(  1703893228),  INT32_C(   567626400) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           3),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 18),
      { -INT32_C(  1025762323),  INT32_C(   174225492), -INT32_C(  1318746243),  INT32_C(  1139494544),  INT32_C(  1440939527),  INT32_C(  1013553812),  INT32_C(  1423704842), -INT32_C(   983106088) },
      {  INT32_C(           0),  INT32_C(           4),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 24),
      { -INT32_C(  1150515389),  INT32_C(  1899591402),  INT32_C(   704768731), -INT32_C(   751745835), -INT32_C(  1050180057), -INT32_C(   221535249),  INT32_C(  2110398592), -INT32_C(   912879482) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 23),
      {  INT32_C(  2030142722), -INT32_C(  1470860610), -INT32_C(  1199735180),  INT32_C(   903827631),  INT32_C(  1529127095), -INT32_C(  1931798932),  INT32_C(  1897027809), -INT32_C(   242688786) },
      {  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_lzcnt_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_maskz_lzcnt_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_si256(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_lzcnt_epi32(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm256_lzcnt_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[4];
    int64_t r[4];
  } test_vec[8] = {
    { {  INT64_C( 4732290617405453566), -INT64_C( 5517151123110942000), -INT64_C( 6625121355923521773),  INT64_C(  339587490994482879) },
      {  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   5) } },
    { {  INT64_C( 6676896013391500074), -INT64_C( 5458739191330735037),  INT64_C( 2848295217845408168),  INT64_C(  767291855686691739) },
      {  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   2),  INT64_C(                   4) } },
    { {  INT64_C( 7994482162557653514),  INT64_C( 9145747431717678231), -INT64_C( 3289342729068148023), -INT64_C( 7188279839517599149) },
      {  INT64_C(                   1),  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C( 7188912464739938347), -INT64_C( 4709208348344730286),  INT64_C( 4727734998771173010),  INT64_C( 5986043841079480291) },
      {  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   1) } },
    { { -INT64_C( 4858419270169049757),  INT64_C( 5251067105866165803), -INT64_C(  306054453955276912),  INT64_C( 5420508662822155709) },
      {  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   1) } },
    { {  INT64_C( 8930372323706135240), -INT64_C( 1424691792483462340), -INT64_C( 5306036435891303351),  INT64_C( 1843051217611968096) },
      {  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   3) } },
    { {  INT64_C( 6449781508403826359), -INT64_C( 2137372983964262899),  INT64_C( 5513145832737849810), -INT64_C( 6136109464935981458) },
      {  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0) } },
    { {  INT64_C( 7830691431826203702),  INT64_C( 5750301218561259346),  INT64_C( 2244936326461460718),  INT64_C( 1898469431456021435) },
      {  INT64_C(                   1),  INT64_C(                   1),  INT64_C(                   3),  INT64_C(                   3) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_lzcnt_epi64(a);
    } EASYSIMD_TEST_PERF_END("_mm256_lzcnt_epi64");
    easysimd_assert_m256i_i64(r, ==, easysimd_mm256_loadu_si256(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_lzcnt_epi64(a);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_lzcnt_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 3101730104897101724),  INT64_C( 8243738292444942508),  INT64_C( 5572907477220373555), -INT64_C( 2101232389794061595) },
      UINT8_C(203),
      {  INT64_C(  114859047355983452),  INT64_C( 8241235191625766534),  INT64_C( 5146649713814294421), -INT64_C( 3021510208828628135) },
      {  INT64_C(                   7),  INT64_C(                   1),  INT64_C( 5572907477220373555),  INT64_C(                   0) } },
    { { -INT64_C( 8374344530517625578),  INT64_C( 7629402146382157233),  INT64_C( 1247599037997131107),  INT64_C( 7228109513693781759) },
      UINT8_C( 92),
      {  INT64_C(  584602581874060049), -INT64_C( 7919222136527615070),  INT64_C( 5877859916093923894),  INT64_C( 7743098819828452273) },
      { -INT64_C( 8374344530517625578),  INT64_C( 7629402146382157233),  INT64_C(                   1),  INT64_C(                   1) } },
    { {  INT64_C( 7217567892954863150), -INT64_C( 9188131267900996226), -INT64_C( 3307149411824187818), -INT64_C( 1552229235317622352) },
      UINT8_C( 53),
      {  INT64_C( 4160581910713011019), -INT64_C( 1441428795930990475), -INT64_C( 7247172634947041257),  INT64_C( 3917641204108850889) },
      {  INT64_C(                   2), -INT64_C( 9188131267900996226),  INT64_C(                   0), -INT64_C( 1552229235317622352) } },
    { {  INT64_C( 2790317182078506316),  INT64_C( 6335102604415024346), -INT64_C( 7056155636340099178), -INT64_C(  642003001736205579) },
      UINT8_C(122),
      {  INT64_C( 7897152315896592268),  INT64_C( 8684634987077408545),  INT64_C( 9047146791066074904),  INT64_C( 1248377280941026995) },
      {  INT64_C( 2790317182078506316),  INT64_C(                   1), -INT64_C( 7056155636340099178),  INT64_C(                   3) } },
    { { -INT64_C(  993773996345426281),  INT64_C(   69921137716552446), -INT64_C( 1803932227003201945), -INT64_C( 2942801918814183533) },
      UINT8_C(144),
      { -INT64_C( 5305993713885863944),  INT64_C( 1946855579240553780), -INT64_C( 7642833463488446922),  INT64_C( 2995926404137486322) },
      { -INT64_C(  993773996345426281),  INT64_C(   69921137716552446), -INT64_C( 1803932227003201945), -INT64_C( 2942801918814183533) } },
    { {  INT64_C( 8923291425538962506),  INT64_C( 5166504059415851971),  INT64_C( 2784791597401899419),  INT64_C( 8139056921591839385) },
      UINT8_C(177),
      { -INT64_C( 3015596029761778052),  INT64_C( 8305251732802748299),  INT64_C( 8285443505680569297),  INT64_C( 3848054000336815608) },
      {  INT64_C(                   0),  INT64_C( 5166504059415851971),  INT64_C( 2784791597401899419),  INT64_C( 8139056921591839385) } },
    { {  INT64_C( 6760113953551441814), -INT64_C( 5231977809902716802), -INT64_C( 3241160328308192102),  INT64_C( 4380296706546834907) },
      UINT8_C( 80),
      { -INT64_C( 4710905921392200094), -INT64_C(   42963469256519803), -INT64_C( 7104518574062241162),  INT64_C( 2771542001613732662) },
      {  INT64_C( 6760113953551441814), -INT64_C( 5231977809902716802), -INT64_C( 3241160328308192102),  INT64_C( 4380296706546834907) } },
    { { -INT64_C( 7027332745344345676), -INT64_C( 1051078710034555274),  INT64_C( 2767713693876437963), -INT64_C( 2464943262414361889) },
      UINT8_C( 92),
      {  INT64_C( 3489747733457508370), -INT64_C( 8927650407473489795), -INT64_C(  513803647684647654),  INT64_C( 6847158417990189798) },
      { -INT64_C( 7027332745344345676), -INT64_C( 1051078710034555274),  INT64_C(                   0),  INT64_C(                   1) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_si256(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_lzcnt_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_lzcnt_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_si256(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_mask_lzcnt_epi64(src, k, a);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm256_maskz_lzcnt_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(164),
      {  INT64_C( 8828771759409074217), -INT64_C(  892643663280268005), -INT64_C( 2919355773766638254),  INT64_C( 2558992341048549975) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(109),
      {  INT64_C( 3480167689602764056), -INT64_C( 6828735505517928393), -INT64_C( 4841126768531907044),  INT64_C( 5389358714446065408) },
      {  INT64_C(                   2),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1) } },
    { UINT8_C(102),
      {  INT64_C( 7537806178990521434),  INT64_C( 2142664207762752299), -INT64_C( 5465483163545946806), -INT64_C( 6365502390389996694) },
      {  INT64_C(                   0),  INT64_C(                   3),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(173),
      {  INT64_C( 3096469551793441570), -INT64_C( 8497442205916948349), -INT64_C( 2081995220642153987),  INT64_C( 2957413631462039083) },
      {  INT64_C(                   2),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   2) } },
    { UINT8_C(228),
      { -INT64_C( 8727214191992441216), -INT64_C(  603650860377199971), -INT64_C( 3934392992246837002),  INT64_C( 5632027093556577131) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(240),
      {  INT64_C( 4108519327154482610), -INT64_C( 7593932884437129664), -INT64_C(    5687513821555131),  INT64_C(  686266056226713500) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(200),
      {  INT64_C( 1947965670089519204),  INT64_C( 2430526848088780423),  INT64_C( 6080243388733274873), -INT64_C( 2229927790345717974) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 37),
      {  INT64_C( 6952630605454092644), -INT64_C( 1871583555790049003),  INT64_C( 2571660032060672809),  INT64_C( 7413911260370105610) },
      {  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   2),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_lzcnt_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("_mm256_maskz_lzcnt_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_si256(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_lzcnt_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_lzcnt_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[16];
    int32_t r[16];
  } test_vec[8] = {
    { {  INT32_C(    28206417),  INT32_C(   365787333), -INT32_C(   400979648),  INT32_C(  1474399048), -INT32_C(   834368611), -INT32_C(  1713628531),  INT32_C(   279699062),  INT32_C(  1968972580),
         INT32_C(  1316358793), -INT32_C(  1016904829),  INT32_C(   363560397), -INT32_C(  1519612664), -INT32_C(  1519144680),  INT32_C(  1094602699),  INT32_C(  1364322861), -INT32_C(  1765364211) },
      {  INT32_C(           7),  INT32_C(           3),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           3),  INT32_C(           1),
         INT32_C(           1),  INT32_C(           0),  INT32_C(           3),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1),  INT32_C(           0) } },
    { {  INT32_C(  1004879032),  INT32_C(  1291798655), -INT32_C(   849171771),  INT32_C(  1332923959),  INT32_C(  1257563519),  INT32_C(  1636512564),  INT32_C(   716365085),  INT32_C(  1136752779),
         INT32_C(   880780980), -INT32_C(  1266647314),  INT32_C(  1619124776),  INT32_C(   816837809),  INT32_C(   242984153), -INT32_C(   194050345),  INT32_C(  1847468771),  INT32_C(  1337122714) },
      {  INT32_C(           2),  INT32_C(           1),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1),  INT32_C(           1),  INT32_C(           2),  INT32_C(           1),
         INT32_C(           2),  INT32_C(           0),  INT32_C(           1),  INT32_C(           2),  INT32_C(           4),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1) } },
    { {  INT32_C(  1954754949), -INT32_C(   685243473), -INT32_C(  1757959706),  INT32_C(  2009589661),  INT32_C(  1652900491),  INT32_C(   743896137), -INT32_C(  1315212266), -INT32_C(   654291628),
         INT32_C(   760054654),  INT32_C(  1812231558), -INT32_C(  1140638689), -INT32_C(  1372337373),  INT32_C(  1443936269), -INT32_C(  1014798675),  INT32_C(   779362010), -INT32_C(   385321877) },
      {  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           2),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0) } },
    { {  INT32_C(  2115393015), -INT32_C(   353756469),  INT32_C(  2040983126), -INT32_C(   970466631),  INT32_C(  1058879378),  INT32_C(  2013503645),  INT32_C(   698775486), -INT32_C(   485314836),
        -INT32_C(   815716348), -INT32_C(  1715909566), -INT32_C(   216899782), -INT32_C(   860276167),  INT32_C(   252434033),  INT32_C(   881266294),  INT32_C(  1918709126), -INT32_C(   531271716) },
      {  INT32_C(           1),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           2),  INT32_C(           1),  INT32_C(           2),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           4),  INT32_C(           2),  INT32_C(           1),  INT32_C(           0) } },
    { { -INT32_C(   626018409),  INT32_C(  1030973443),  INT32_C(    19957191),  INT32_C(   835578303),  INT32_C(   910219456),  INT32_C(  1835780071), -INT32_C(   790640396), -INT32_C(   810535880),
        -INT32_C(   290889749), -INT32_C(  1892999993),  INT32_C(  1636850594),  INT32_C(    93478213),  INT32_C(   473682485), -INT32_C(  1903581543), -INT32_C(  1486985105), -INT32_C(  2005528675) },
      {  INT32_C(           0),  INT32_C(           2),  INT32_C(           7),  INT32_C(           2),  INT32_C(           2),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           5),  INT32_C(           3),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { {  INT32_C(   913776750), -INT32_C(   557473220),  INT32_C(  1128289790), -INT32_C(   414657870),  INT32_C(  1040483237), -INT32_C(  1714647766), -INT32_C(  1824510986), -INT32_C(  1474578886),
         INT32_C(   333353686),  INT32_C(   871473973), -INT32_C(  1435094536), -INT32_C(  1450000892),  INT32_C(  1810339393),  INT32_C(   419738659),  INT32_C(   430720223), -INT32_C(   775829253) },
      {  INT32_C(           2),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           3),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           3),  INT32_C(           3),  INT32_C(           0) } },
    { { -INT32_C(  1880842150),  INT32_C(  1019401795),  INT32_C(   199636999),  INT32_C(   951351542),  INT32_C(   849583118),  INT32_C(   793487440), -INT32_C(   414648084),  INT32_C(   448334272),
        -INT32_C(   307585623),  INT32_C(  2066312307), -INT32_C(  1685712987), -INT32_C(  1764541560),  INT32_C(   667449303),  INT32_C(   190190623), -INT32_C(   856449524),  INT32_C(  1374071975) },
      {  INT32_C(           0),  INT32_C(           2),  INT32_C(           4),  INT32_C(           2),  INT32_C(           2),  INT32_C(           2),  INT32_C(           0),  INT32_C(           3),
         INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           4),  INT32_C(           0),  INT32_C(           1) } },
    { { -INT32_C(  1119973303), -INT32_C(  1573361667), -INT32_C(    29507978), -INT32_C(   795536903), -INT32_C(  1476960888),  INT32_C(  2108837233), -INT32_C(  1823889941), -INT32_C(  1679544239),
        -INT32_C(  1118297408), -INT32_C(    10514295),  INT32_C(  1224645966),  INT32_C(   907580334),  INT32_C(  1658654960),  INT32_C(  1239387997), -INT32_C(  2032391883),  INT32_C(   421642329) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           2),  INT32_C(           1),  INT32_C(           1),  INT32_C(           0),  INT32_C(           3) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_lzcnt_epi32(a);
    } EASYSIMD_TEST_PERF_END("_mm512_lzcnt_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_si512(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_lzcnt_epi32(a);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_lzcnt_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(  1559140281),  INT32_C(   391407479),  INT32_C(  1030999718), -INT32_C(  2108832516), -INT32_C(  1108660764),  INT32_C(   397218066),  INT32_C(  1210364788),  INT32_C(    65029194),
         INT32_C(  1382010587), -INT32_C(   597052362),  INT32_C(  2132401282), -INT32_C(  2063505759), -INT32_C(  1505563501),  INT32_C(  1908273149), -INT32_C(   323362398),  INT32_C(    15702309) },
      UINT16_C(20328),
      { -INT32_C(  1140613549), -INT32_C(  1818655366),  INT32_C(   100284932),  INT32_C(    32607679), -INT32_C(   235868621),  INT32_C(   433230431),  INT32_C(  1840445566),  INT32_C(  1270618872),
         INT32_C(   839434168),  INT32_C(  1237754181), -INT32_C(  1706049317), -INT32_C(  2120531891), -INT32_C(  1905095889), -INT32_C(  1683536867),  INT32_C(   889739836),  INT32_C(   763413876) },
      {  INT32_C(  1559140281),  INT32_C(   391407479),  INT32_C(  1030999718),  INT32_C(           7), -INT32_C(  1108660764),  INT32_C(           3),  INT32_C(           1),  INT32_C(    65029194),
         INT32_C(  1382010587), -INT32_C(   597052362),  INT32_C(  2132401282), -INT32_C(  2063505759), -INT32_C(  1505563501),  INT32_C(  1908273149), -INT32_C(   323362398),  INT32_C(    15702309) } },
    { { -INT32_C(   916486012),  INT32_C(    68363561),  INT32_C(   866018021), -INT32_C(   776717918), -INT32_C(   513792316), -INT32_C(  1485043862), -INT32_C(   690191007), -INT32_C(   838640566),
         INT32_C(   244867813),  INT32_C(  1830005640), -INT32_C(  1348423411), -INT32_C(  1350478613), -INT32_C(   443424390),  INT32_C(  1250692584), -INT32_C(   601855854), -INT32_C(  1448401980) },
      UINT16_C(17285),
      { -INT32_C(   873591368),  INT32_C(   461175675),  INT32_C(   728786858), -INT32_C(  1475548649),  INT32_C(  1538651599),  INT32_C(  1606633535), -INT32_C(   813529052), -INT32_C(   384694479),
        -INT32_C(  1867251691), -INT32_C(  1498729989), -INT32_C(  1345250408), -INT32_C(   715661818),  INT32_C(   305204691),  INT32_C(  2037511253), -INT32_C(  1370885252),  INT32_C(   261577722) },
      {  INT32_C(           0),  INT32_C(    68363561),  INT32_C(           2), -INT32_C(   776717918), -INT32_C(   513792316), -INT32_C(  1485043862), -INT32_C(   690191007),  INT32_C(           0),
         INT32_C(   244867813),  INT32_C(  1830005640), -INT32_C(  1348423411), -INT32_C(  1350478613), -INT32_C(   443424390),  INT32_C(  1250692584), -INT32_C(   601855854), -INT32_C(  1448401980) } },
    { {  INT32_C(  1470057564),  INT32_C(   368921469),  INT32_C(  1824903014), -INT32_C(  2143150931),  INT32_C(  2123526953), -INT32_C(   453508249), -INT32_C(   258850314), -INT32_C(   134207076),
        -INT32_C(   229597323),  INT32_C(  1359498730), -INT32_C(   910308068),  INT32_C(   323616745), -INT32_C(   627975310), -INT32_C(   725710370),  INT32_C(  1740984522), -INT32_C(   278936199) },
      UINT16_C(44900),
      { -INT32_C(   369274911),  INT32_C(  1572215200),  INT32_C(   744333539),  INT32_C(  1141428147),  INT32_C(  1741612713), -INT32_C(  2135451461),  INT32_C(  1598370303),  INT32_C(    17738016),
        -INT32_C(  1729426440),  INT32_C(   150380837), -INT32_C(   197897407), -INT32_C(   868664286), -INT32_C(   567081181), -INT32_C(  1621169249),  INT32_C(  1023320860),  INT32_C(  1161628748) },
      {  INT32_C(  1470057564),  INT32_C(   368921469),  INT32_C(           2), -INT32_C(  2143150931),  INT32_C(  2123526953),  INT32_C(           0),  INT32_C(           1), -INT32_C(   134207076),
        -INT32_C(   229597323),  INT32_C(  1359498730), -INT32_C(   910308068),  INT32_C(   323616745), -INT32_C(   627975310), -INT32_C(   725710370),  INT32_C(  1740984522), -INT32_C(   278936199) } },
    { {  INT32_C(  1037903896),  INT32_C(   189125578),  INT32_C(  1224767782), -INT32_C(   652920394), -INT32_C(   541636544),  INT32_C(  1333663027),  INT32_C(    76250296), -INT32_C(  1588999799),
        -INT32_C(  1143068687),  INT32_C(   566699002),  INT32_C(  1382729628),  INT32_C(  1076592384), -INT32_C(    81796409), -INT32_C(  1354064137), -INT32_C(  1548429798), -INT32_C(  1874526817) },
      UINT16_C( 8740),
      {  INT32_C(   323297100), -INT32_C(  1428495808),  INT32_C(  1613355572),  INT32_C(   960688154),  INT32_C(   920074987), -INT32_C(  1660095767), -INT32_C(   627332203), -INT32_C(  2013479109),
         INT32_C(   513426142),  INT32_C(  1489531940),  INT32_C(  1756950862), -INT32_C(   844956702),  INT32_C(   503544117),  INT32_C(    29101931), -INT32_C(   153397318), -INT32_C(   176302314) },
      {  INT32_C(  1037903896),  INT32_C(   189125578),  INT32_C(           1), -INT32_C(   652920394), -INT32_C(   541636544),  INT32_C(           0),  INT32_C(    76250296), -INT32_C(  1588999799),
        -INT32_C(  1143068687),  INT32_C(   566699002),  INT32_C(  1382729628),  INT32_C(  1076592384), -INT32_C(    81796409), -INT32_C(  1354064137), -INT32_C(  1548429798), -INT32_C(  1874526817) } },
    { {  INT32_C(  1024661529), -INT32_C(   610870132), -INT32_C(  1354543411),  INT32_C(  2122179913), -INT32_C(   895647649),  INT32_C(  1254840720), -INT32_C(   968841552), -INT32_C(  1749303682),
         INT32_C(  1658179542),  INT32_C(  2017291179),  INT32_C(    52986297), -INT32_C(   981359258), -INT32_C(  1248846299),  INT32_C(   654334839), -INT32_C(  2131869694), -INT32_C(   736646658) },
      UINT16_C(60536),
      {  INT32_C(  1951867703), -INT32_C(   990572132), -INT32_C(  1788257260), -INT32_C(  1313567199),  INT32_C(  1141713732),  INT32_C(  1099173715), -INT32_C(  1494580337), -INT32_C(  1902943401),
         INT32_C(   570681990),  INT32_C(   266795259),  INT32_C(  1990545236),  INT32_C(   573004254), -INT32_C(   681167740), -INT32_C(   770119101), -INT32_C(   981925011), -INT32_C(   330101658) },
      {  INT32_C(  1024661529), -INT32_C(   610870132), -INT32_C(  1354543411),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1),  INT32_C(           0), -INT32_C(  1749303682),
         INT32_C(  1658179542),  INT32_C(  2017291179),  INT32_C(    52986297), -INT32_C(   981359258), -INT32_C(  1248846299),  INT32_C(   654334839), -INT32_C(  2131869694), -INT32_C(   736646658) } },
    { { -INT32_C(   233875722), -INT32_C(  1560152753),  INT32_C(   588883525), -INT32_C(  2075835905), -INT32_C(  1185174667),  INT32_C(    59470997), -INT32_C(   574094217),  INT32_C(   130620176),
        -INT32_C(  1040590734),  INT32_C(   325384910), -INT32_C(  1607041375),  INT32_C(   874806207), -INT32_C(  1142063066),  INT32_C(  1807645172), -INT32_C(  1907849603),  INT32_C(   345313698) },
      UINT16_C(36586),
      {  INT32_C(   965261525),  INT32_C(    28781003), -INT32_C(   293833014), -INT32_C(  1737579861),  INT32_C(   470901341),  INT32_C(   362975181), -INT32_C(  1306049507),  INT32_C(   759173464),
        -INT32_C(  1805137719), -INT32_C(  1131077902),  INT32_C(  1084887445),  INT32_C(   282597811),  INT32_C(  1227680379), -INT32_C(  1772171399),  INT32_C(  1816691987),  INT32_C(  1603897494) },
      { -INT32_C(   233875722),  INT32_C(           7),  INT32_C(   588883525),  INT32_C(           0), -INT32_C(  1185174667),  INT32_C(           3),  INT32_C(           0),  INT32_C(           2),
        -INT32_C(  1040590734),  INT32_C(   325384910), -INT32_C(  1607041375),  INT32_C(   874806207), -INT32_C(  1142063066),  INT32_C(  1807645172), -INT32_C(  1907849603),  INT32_C(   345313698) } },
    { {  INT32_C(  1123221584), -INT32_C(  1258387425),  INT32_C(  1291102617),  INT32_C(  1029557442),  INT32_C(   797346230),  INT32_C(  1824908632),  INT32_C(    30936426), -INT32_C(   429887083),
        -INT32_C(  1859627918),  INT32_C(  1984243676), -INT32_C(  1832764976), -INT32_C(  1127276795),  INT32_C(    32200361), -INT32_C(  1519537861),  INT32_C(  1420182974),  INT32_C(   691668919) },
      UINT16_C(25179),
      { -INT32_C(     7784518),  INT32_C(  1882741165), -INT32_C(  1148240405), -INT32_C(   451856135), -INT32_C(  1483322310), -INT32_C(  1729276687), -INT32_C(   492854104), -INT32_C(  2025456947),
        -INT32_C(   544813519),  INT32_C(   323993128), -INT32_C(   170991877),  INT32_C(  1373298455),  INT32_C(   502820907),  INT32_C(  1840637380),  INT32_C(  1448039561),  INT32_C(  2145227854) },
      {  INT32_C(           0),  INT32_C(           1),  INT32_C(  1291102617),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1824908632),  INT32_C(           0), -INT32_C(   429887083),
        -INT32_C(  1859627918),  INT32_C(  1984243676), -INT32_C(  1832764976), -INT32_C(  1127276795),  INT32_C(    32200361), -INT32_C(  1519537861),  INT32_C(  1420182974),  INT32_C(   691668919) } },
    { { -INT32_C(  1956748445),  INT32_C(   496938273), -INT32_C(  1559073396),  INT32_C(  2029317196),  INT32_C(   546696284),  INT32_C(  1535986386), -INT32_C(   323822178), -INT32_C(   731148431),
         INT32_C(   341822195),  INT32_C(    53608055), -INT32_C(  1213840533), -INT32_C(  1959814609),  INT32_C(  1504494727), -INT32_C(  1397475058), -INT32_C(  2003278314), -INT32_C(   379845386) },
      UINT16_C(48334),
      {  INT32_C(   800736765), -INT32_C(   277732023),  INT32_C(   210412252), -INT32_C(   640675539),  INT32_C(   504618858),  INT32_C(   612706699),  INT32_C(   237534129),  INT32_C(  1640691300),
        -INT32_C(  2070903749), -INT32_C(  2055994455), -INT32_C(   762184027),  INT32_C(  2041340431), -INT32_C(   862470335), -INT32_C(  1695539992), -INT32_C(    72869737),  INT32_C(  1247638030) },
      { -INT32_C(  1956748445),  INT32_C(           0),  INT32_C(           4),  INT32_C(           0),  INT32_C(   546696284),  INT32_C(  1535986386),  INT32_C(           4),  INT32_C(           1),
         INT32_C(   341822195),  INT32_C(    53608055), -INT32_C(  1213840533), -INT32_C(  1959814609),  INT32_C(  1504494727), -INT32_C(  1397475058), -INT32_C(  2003278314), -INT32_C(   379845386) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_si512(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_lzcnt_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_lzcnt_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_si512(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_mask_lzcnt_epi32(src, k, a);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_maskz_lzcnt_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(29846),
      { -INT32_C(  2136332798),  INT32_C(  2010314952),  INT32_C(  1062701142),  INT32_C(   636715923), -INT32_C(  1705378966),  INT32_C(  1930196792), -INT32_C(  2109534942),  INT32_C(  1073142077),
        -INT32_C(   943742465), -INT32_C(   281111911), -INT32_C(  1372678886), -INT32_C(  2083315432),  INT32_C(  1427975452), -INT32_C(   959960412),  INT32_C(  1950878519), -INT32_C(   474726684) },
      {  INT32_C(           0),  INT32_C(           1),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(29663),
      { -INT32_C(   402294358), -INT32_C(  1753407384), -INT32_C(  1581738290),  INT32_C(   919524633), -INT32_C(   228560342), -INT32_C(  2130798536), -INT32_C(  1061166580),  INT32_C(  1882496710),
        -INT32_C(  2141636329),  INT32_C(   655873625), -INT32_C(  2050371732), -INT32_C(   826501212),  INT32_C(  1119951882), -INT32_C(  1044136012),  INT32_C(  1736540833),  INT32_C(   953726240) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(12783),
      { -INT32_C(   821606216),  INT32_C(   966685808), -INT32_C(  1244642311), -INT32_C(   757933551), -INT32_C(   527333859), -INT32_C(   916311225),  INT32_C(  1920893594),  INT32_C(  1923313082),
         INT32_C(   641838006),  INT32_C(   408936479),  INT32_C(   885927715),  INT32_C(   654811146),  INT32_C(  1845991462),  INT32_C(  1714907595), -INT32_C(  1495747093), -INT32_C(   652706781) },
      {  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(23079),
      {  INT32_C(  1580877567),  INT32_C(   747461982), -INT32_C(  1731422319), -INT32_C(   986582082), -INT32_C(  1741751199),  INT32_C(   994908770), -INT32_C(   659066688),  INT32_C(  1211293257),
        -INT32_C(  2103022556),  INT32_C(  1538208969), -INT32_C(  1963754549), -INT32_C(   816896914), -INT32_C(  2090369503),  INT32_C(  1488893336),  INT32_C(  1865446693),  INT32_C(  2008507219) },
      {  INT32_C(           1),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(24271),
      { -INT32_C(  1450010118), -INT32_C(   417047052),  INT32_C(   923571175), -INT32_C(   894096030),  INT32_C(  1870613936), -INT32_C(   706435931), -INT32_C(   885508077), -INT32_C(  1456928593),
        -INT32_C(  1789740127),  INT32_C(     8156952),  INT32_C(  1815578634),  INT32_C(  1714875573), -INT32_C(   556419783),  INT32_C(  1840494938), -INT32_C(  1590104847), -INT32_C(  1790287372) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(40220),
      { -INT32_C(  1508625110),  INT32_C(  1815027253), -INT32_C(  1051138934),  INT32_C(   527864138),  INT32_C(   584700015),  INT32_C(  1980680766), -INT32_C(  1227357589), -INT32_C(  1051462505),
         INT32_C(  1583900457),  INT32_C(   264935301), -INT32_C(   959438213),  INT32_C(   585451443),  INT32_C(  1430634007), -INT32_C(   187935863), -INT32_C(   307583914), -INT32_C(  1028653672) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           3),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C( 5988),
      { -INT32_C(   340858592), -INT32_C(   921884167), -INT32_C(   720322065),  INT32_C(   680732899),  INT32_C(  1232870525), -INT32_C(  1142042095),  INT32_C(  1857652415),  INT32_C(  1753554248),
         INT32_C(     5452551),  INT32_C(  1288266077),  INT32_C(    86104610),  INT32_C(  2133767426), -INT32_C(   490100015),  INT32_C(  1000257148), -INT32_C(  2069211332),  INT32_C(  2079141748) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(16483),
      {  INT32_C(  1168228475),  INT32_C(   773833484), -INT32_C(   136109623),  INT32_C(  1772205216),  INT32_C(   891297175),  INT32_C(    42753112), -INT32_C(   835583775), -INT32_C(   150039172),
         INT32_C(  1631367253),  INT32_C(  1016028019),  INT32_C(   489910908), -INT32_C(  1098460121),  INT32_C(  1257481969), -INT32_C(   464748797), -INT32_C(    72188289),  INT32_C(  1760804883) },
      {  INT32_C(           1),  INT32_C(           2),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           6),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_lzcnt_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_maskz_lzcnt_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_si512(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_maskz_lzcnt_epi32(k, a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_lzcnt_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[8];
    int64_t r[8];
  } test_vec[8] = {
    { {  INT64_C( 1591691654996826440), -INT64_C( 7687713213491748987),  INT64_C( 8394705991448453511),  INT64_C( 6676338044889584758),
         INT64_C(  679474565120045367),  INT64_C( 8394604074629082202),  INT64_C( 1339040906451066753), -INT64_C(  324548601851042290) },
      {  INT64_C(                   3),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   1),
         INT64_C(                   4),  INT64_C(                   1),  INT64_C(                   3),  INT64_C(                   0) } },
    { { -INT64_C( 5244512834119638359), -INT64_C( 6319359346603882517),  INT64_C( 5795672310061467839),  INT64_C( 6972962795209015044),
        -INT64_C( 4510417876400859565), -INT64_C( 1793303291147712435),  INT64_C( 5081099702490097136),  INT64_C( 4145514967469127280) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   2) } },
    { {  INT64_C( 2990760536593471209), -INT64_C(  835213206671713448), -INT64_C( 3735548574995862625), -INT64_C(  810513747277564000),
         INT64_C( 6023059801824799376), -INT64_C( 8194515102182722480), -INT64_C( 5111981292762665242),  INT64_C(  922676609406730704) },
      {  INT64_C(                   2),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   4) } },
    { {  INT64_C(  502168635586478763), -INT64_C( 1243097919866090429), -INT64_C( 7430306306290864368), -INT64_C( 3474174011503402176),
        -INT64_C( 7746071162379812157), -INT64_C( 8417643889094698439), -INT64_C( 2444161990802513996), -INT64_C( 8381932517575653446) },
      {  INT64_C(                   5),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C( 4068012088834645594), -INT64_C( 8474306831713808290),  INT64_C( 7715259559518280341), -INT64_C( 5190972440816538206),
        -INT64_C( 2563329396778796777),  INT64_C( 4749358808704644489), -INT64_C( 6029608470192984266), -INT64_C( 6270123862649300172) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C( 7809167964230298117),  INT64_C( 7502913634784974385),  INT64_C( 5718341292137616548), -INT64_C( 9046165406509403844),
        -INT64_C( 5653877355781438406), -INT64_C( 5974959449040792198),  INT64_C( 4660300943795708738),  INT64_C(  731698396616540200) },
      {  INT64_C(                   1),  INT64_C(                   1),  INT64_C(                   1),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   4) } },
    { {  INT64_C(  826604671780465735), -INT64_C(  595375972694882555), -INT64_C(  543527290092120016),  INT64_C( 5483912121821266496),
        -INT64_C(  608393926456163855),  INT64_C( 4774030088314701459), -INT64_C( 2980278400818113061), -INT64_C(  594474027865194386) },
      {  INT64_C(                   4),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C(  924355713739140030), -INT64_C( 3340969480720469427),  INT64_C( 8393899389633698066), -INT64_C( 6387045018063313825),
         INT64_C( 3084975493735044476), -INT64_C( 7277485455605887942), -INT64_C(  306032997134271162), -INT64_C( 3660937967987283932) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),
         INT64_C(                   2),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_lzcnt_epi64(a);
    } EASYSIMD_TEST_PERF_END("_mm512_lzcnt_epi64");
    easysimd_assert_m512i_i64(r, ==, easysimd_mm512_loadu_si512(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_lzcnt_epi64(a);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_lzcnt_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 7028608760938581938),  INT64_C( 6335552780561246560),  INT64_C(  300073565166146868), -INT64_C( 5775872033034151472),
        -INT64_C( 3812291128666109879), -INT64_C( 3713916533108887442), -INT64_C( 8111183321681188070), -INT64_C( 7652099229368626910) },
      UINT8_C( 16),
      { -INT64_C( 7728945088365441975), -INT64_C(  253054454911411235), -INT64_C( 6304395888748206314), -INT64_C( 5182929111266057798),
        -INT64_C( 5706406171141239339),  INT64_C(  643327495861631648), -INT64_C( 6091704596711299211), -INT64_C( 4056968164487042677) },
      {  INT64_C( 7028608760938581938),  INT64_C( 6335552780561246560),  INT64_C(  300073565166146868), -INT64_C( 5775872033034151472),
         INT64_C(                   0), -INT64_C( 3713916533108887442), -INT64_C( 8111183321681188070), -INT64_C( 7652099229368626910) } },
    { {  INT64_C( 2292559255724669517), -INT64_C( 1890388757965643064), -INT64_C( 6899190180478833755), -INT64_C(  195540787824758896),
        -INT64_C( 2259138948488148730),  INT64_C( 6351586973435278799), -INT64_C( 8197061501133072503),  INT64_C( 3155995453507557407) },
      UINT8_C(162),
      {  INT64_C(  839810648521748714), -INT64_C( 2153540807817238986), -INT64_C( 3363503955096061180), -INT64_C( 2847927604160945794),
         INT64_C(  374175691755697361),  INT64_C(  660821904149049149), -INT64_C(  559695972746327183),  INT64_C( 2483535433952145841) },
      {  INT64_C( 2292559255724669517),  INT64_C(                   0), -INT64_C( 6899190180478833755), -INT64_C(  195540787824758896),
        -INT64_C( 2259138948488148730),  INT64_C(                   4), -INT64_C( 8197061501133072503),  INT64_C(                   2) } },
    { { -INT64_C( 5020534480985053570),  INT64_C( 1547318126943858026), -INT64_C(  672237811323128263),  INT64_C( 4926952033624620012),
        -INT64_C( 8256535025381365453), -INT64_C( 3203336304150614830), -INT64_C( 5663221807335470548),  INT64_C( 4446828588371541421) },
      UINT8_C(248),
      { -INT64_C( 4395327044412773048),  INT64_C(  885010108107290278),  INT64_C( 2275869368368871862),  INT64_C(  484531404389549938),
         INT64_C( 2552755377631712339), -INT64_C( 5568269949253727317),  INT64_C( 9187676449347897921),  INT64_C( 7204625880121911862) },
      { -INT64_C( 5020534480985053570),  INT64_C( 1547318126943858026), -INT64_C(  672237811323128263),  INT64_C(                   5),
         INT64_C(                   2),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   1) } },
    { {  INT64_C( 1184300897707831246),  INT64_C( 8643382085257375302), -INT64_C(  896959887515742976), -INT64_C( 7675854745165640187),
         INT64_C( 5225952389124687634), -INT64_C( 4995761127233600514), -INT64_C( 1893106269443029750), -INT64_C( 7601946957246127837) },
      UINT8_C(252),
      { -INT64_C( 2976801753531544479),  INT64_C( 1537779781390623951),  INT64_C( 8758334523343969810),  INT64_C( 6361215684730255062),
         INT64_C( 6487920745711748512), -INT64_C(  773494459283826274), -INT64_C( 1698446316342003684), -INT64_C( 7013095286969767074) },
      {  INT64_C( 1184300897707831246),  INT64_C( 8643382085257375302),  INT64_C(                   1),  INT64_C(                   1),
         INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C(  983433364693431999),  INT64_C( 7371687022848330868), -INT64_C( 1102881000471745866),  INT64_C( 2292316126505723186),
        -INT64_C( 1142772479313731161), -INT64_C( 7707181929308069798),  INT64_C( 6828318145222960192), -INT64_C( 6812996282516305676) },
      UINT8_C(  0),
      {  INT64_C( 4415054710101031132),  INT64_C( 6452445454416997045),  INT64_C( 3031735234854215182), -INT64_C( 5272933574861625741),
         INT64_C( 8041430822745624190),  INT64_C( 8295144962291053776), -INT64_C(  436387889625848924),  INT64_C( 6528422702595380490) },
      {  INT64_C(  983433364693431999),  INT64_C( 7371687022848330868), -INT64_C( 1102881000471745866),  INT64_C( 2292316126505723186),
        -INT64_C( 1142772479313731161), -INT64_C( 7707181929308069798),  INT64_C( 6828318145222960192), -INT64_C( 6812996282516305676) } },
    { { -INT64_C( 1074867029620213013), -INT64_C( 4982718320031117310),  INT64_C( 1799031984882805367),  INT64_C( 5529740158707680135),
        -INT64_C( 4403948960357387569), -INT64_C( 1970322165820974889),  INT64_C( 2491575380199221613), -INT64_C(  916128326848508259) },
      UINT8_C(181),
      {  INT64_C( 6780794608851215163), -INT64_C( 2751506200200017823),  INT64_C( 7856976906971007890), -INT64_C( 8425246096672946660),
         INT64_C( 2489058759965313383), -INT64_C( 7531612844639037967), -INT64_C( 6850805694937698655),  INT64_C( 7561000621566819602) },
      {  INT64_C(                   1), -INT64_C( 4982718320031117310),  INT64_C(                   1),  INT64_C( 5529740158707680135),
         INT64_C(                   2),  INT64_C(                   0),  INT64_C( 2491575380199221613),  INT64_C(                   1) } },
    { { -INT64_C( 7602604325871502181), -INT64_C(  355589368423447436),  INT64_C( 3658163280297005983),  INT64_C( 8164888508965897190),
         INT64_C( 8609558687961244479),  INT64_C(  262899160124011088),  INT64_C( 8099421058197037332), -INT64_C( 4830209017293309265) },
      UINT8_C(137),
      { -INT64_C( 3575291739385323323), -INT64_C( 3167941820161061645), -INT64_C( 7063668814824719011),  INT64_C( 4203220153650544165),
        -INT64_C( 2927148564867194761),  INT64_C( 2684379335196644026), -INT64_C( 7975199373050077679), -INT64_C( 5002459858258334492) },
      {  INT64_C(                   0), -INT64_C(  355589368423447436),  INT64_C( 3658163280297005983),  INT64_C(                   2),
         INT64_C( 8609558687961244479),  INT64_C(  262899160124011088),  INT64_C( 8099421058197037332),  INT64_C(                   0) } },
    { {  INT64_C( 5065614499659964305), -INT64_C(  203519688071506113), -INT64_C( 1905018235763123566),  INT64_C( 3417619267003083888),
        -INT64_C( 6447764440264426831), -INT64_C( 4759783448847173881),  INT64_C( 1683022068376642421),  INT64_C( 2162915765434161720) },
      UINT8_C(242),
      {  INT64_C( 1774419222107610280),  INT64_C( 2881459905808798638),  INT64_C( 7584034074378793916), -INT64_C( 4405376843829078607),
        -INT64_C( 2068284497461627567),  INT64_C( 5420761164704157326), -INT64_C( 6983514093377156899),  INT64_C( 3639583469141093543) },
      {  INT64_C( 5065614499659964305),  INT64_C(                   2), -INT64_C( 1905018235763123566),  INT64_C( 3417619267003083888),
         INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   2) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_si512(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_lzcnt_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_lzcnt_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_si512(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_mask_lzcnt_epi64(src, k, a);

    easysimd_test_x86_write_i64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_maskz_lzcnt_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(109),
      {  INT64_C( 8113783188123067438), -INT64_C( 6331726917790944160), -INT64_C( 6728485738128072804), -INT64_C( 4935747773095087686),
         INT64_C( 8113653707765390679), -INT64_C( 5853312744853675215), -INT64_C( 5513790255457280773),  INT64_C( 4777835331808868358) },
      {  INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(100),
      { -INT64_C( 1738138747407144143),  INT64_C( 1978673151654624158),  INT64_C(   87759484194475549),  INT64_C( 5348920766181001871),
         INT64_C( 3641116952308002715),  INT64_C( 5021268567701356756),  INT64_C( 2033848624000517231), -INT64_C( 1019169941318588807) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   7),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   3),  INT64_C(                   0) } },
    { UINT8_C(222),
      { -INT64_C( 7457781660852842284), -INT64_C( 8656192522302778289),  INT64_C( 8621067612355499901), -INT64_C( 1932116733522528734),
        -INT64_C( 7782763021061676977),  INT64_C( 4787576843052302118), -INT64_C( 2971510767843703557), -INT64_C( 7043816393385995960) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(246),
      { -INT64_C( 4464529564491971119), -INT64_C( 1679621226629696558),  INT64_C( 5877604059377826871), -INT64_C(  367818345231095144),
        -INT64_C( 4680672756887576764), -INT64_C( 7915316746243436069), -INT64_C( 3660463097721928926),  INT64_C( 6231469035750150998) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1) } },
    { UINT8_C(192),
      {  INT64_C( 4115284728916844120),  INT64_C( 5055614400909827042),  INT64_C( 1352284121568250625),  INT64_C( 1685326744277763075),
         INT64_C( 6988024760065528089), -INT64_C( 5217259960351092785),  INT64_C( 6768321291928998556), -INT64_C( 6556567551097171668) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   1),  INT64_C(                   0) } },
    { UINT8_C(117),
      { -INT64_C( 1108919728707770791), -INT64_C( 7428585853483094342), -INT64_C( 4497319558772505727), -INT64_C( 2582265164328959664),
         INT64_C( 3002569922764676186),  INT64_C( 3337833274557157427), -INT64_C( 1976748644260953056),  INT64_C( 3246745494195068846) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   2),  INT64_C(                   2),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(221),
      { -INT64_C( 2076030149747480393), -INT64_C( 9194170948145583382), -INT64_C( 6326720982130870442),  INT64_C( 2967494409135547280),
        -INT64_C( 2736122659222422896), -INT64_C( 5981473725535316160), -INT64_C( 7335694145137208895), -INT64_C( 1774670770569695547) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   2),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(149),
      { -INT64_C( 6261024851633042811), -INT64_C( 6355516313191868031),  INT64_C( 3891475701042337498),  INT64_C( 2082761545956013880),
         INT64_C( 5564502782942669950), -INT64_C( 4761671622491359058),  INT64_C( 2394220555625143674), -INT64_C( 5051531098980104654) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   2),  INT64_C(                   0),
         INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_lzcnt_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_maskz_lzcnt_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_si512(test_vec[i].r));
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_maskz_lzcnt_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_lzcnt_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_lzcnt_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_lzcnt_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_lzcnt_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_lzcnt_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_lzcnt_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_lzcnt_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_lzcnt_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_lzcnt_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_lzcnt_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
