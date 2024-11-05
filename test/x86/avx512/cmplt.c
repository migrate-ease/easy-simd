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

#define EASYSIMD_TEST_X86_AVX512_INSN cmplt

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/cmplt.h>

static int
test_easysimd_mm_cmplt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int8_t a[16];
    const int8_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { { -INT8_C(  42), -INT8_C( 115), -INT8_C(  30), -INT8_C(  67),  INT8_C(  70), -INT8_C(  47),  INT8_C(  35), -INT8_C(  60),
         INT8_C(  43),  INT8_C(  75), -INT8_C(  76), -INT8_C(   8), -INT8_C(  76), -INT8_C(  53), -INT8_C(  61),  INT8_C(  51) },
      {  INT8_C(   4),  INT8_C(  71), -INT8_C(  23), -INT8_C(  79),  INT8_C(  46),  INT8_C( 105),  INT8_C( 120), -INT8_C(   1),
         INT8_C(  55),  INT8_C( 118), -INT8_C(  92),  INT8_C(   0),  INT8_C(  82),  INT8_C(  43), -INT8_C( 125),  INT8_C(  41) },
      UINT16_C(15335) },
    { { -INT8_C(  72),  INT8_C( 102), -INT8_C(  26), -INT8_C(   1),  INT8_C(  55),  INT8_C(   9), -INT8_C(  61),  INT8_C(  99),
         INT8_C(  84),  INT8_C( 119),  INT8_C(  91),  INT8_C(   9),  INT8_C(  66),  INT8_C(  30),  INT8_C(  60),  INT8_C(  70) },
      {  INT8_C( 102),  INT8_C(  37), -INT8_C(   8), -INT8_C( 108), -INT8_C( 113),  INT8_C( 112), -INT8_C( 108), -INT8_C(  58),
        -INT8_C(  26),  INT8_C(  56), -INT8_C(  58),  INT8_C(  56),  INT8_C(  99),  INT8_C(  74),  INT8_C(  97),  INT8_C(  28) },
      UINT16_C(30757) },
    { { -INT8_C(  80),  INT8_C(  71),  INT8_C(  27), -INT8_C(  25),  INT8_C(  80), -INT8_C(  34),  INT8_C(  74), -INT8_C(  91),
         INT8_C(  85), -INT8_C(  91), -INT8_C(  82), -INT8_C( 104), -INT8_C(  60), -INT8_C(  22), -INT8_C(  34),  INT8_C(  42) },
      {  INT8_C(  16), -INT8_C(  42), -INT8_C(  66), -INT8_C(  97),  INT8_C(  70),  INT8_C(  82),  INT8_C( 101),  INT8_C(  44),
        -INT8_C( 118),  INT8_C(  43),  INT8_C( 101), -INT8_C(  18),  INT8_C( 117), -INT8_C(  58),  INT8_C(  10),  INT8_C(  37) },
      UINT16_C(24289) },
    { {  INT8_C(  14),  INT8_C(  37),  INT8_C(  13),  INT8_C(  94),  INT8_C(   3),  INT8_C(  87),  INT8_C(   3),  INT8_C(  88),
        -INT8_C(   3), -INT8_C(  79), -INT8_C(  16), -INT8_C(  63), -INT8_C( 100), -INT8_C(  49), -INT8_C(  21), -INT8_C(  84) },
      { -INT8_C(  91), -INT8_C(  87),  INT8_C(  75), -INT8_C(  20), -INT8_C(   4), -INT8_C(  80),  INT8_C(  24), -INT8_C( 122),
        -INT8_C(  37),  INT8_C( 125),  INT8_C( 116),  INT8_C(  81),  INT8_C(  68),  INT8_C( 126),  INT8_C( 118),  INT8_C(  82) },
      UINT16_C(65092) },
    { { -INT8_C(  93), -INT8_C( 125), -INT8_C(  80), -INT8_C(  90), -INT8_C(  37), -INT8_C(  76), -INT8_C(   1), -INT8_C(  40),
         INT8_C( 101), -INT8_C(  17), -INT8_C( 103),  INT8_C(   1), -INT8_C(  66), -INT8_C( 124), -INT8_C(  83),  INT8_C( 100) },
      {  INT8_C(  45), -INT8_C(   8),  INT8_C(  80),  INT8_C(  41), -INT8_C(  88),  INT8_C( 104), -INT8_C(  80), -INT8_C( 124),
        -INT8_C(  26),  INT8_C(  36), -INT8_C(  43),  INT8_C(  42), -INT8_C(  93),  INT8_C(  75),  INT8_C( 124),  INT8_C(  70) },
      UINT16_C(28207) },
    { { -INT8_C(  49),  INT8_C(  44), -INT8_C(  19), -INT8_C(  86), -INT8_C(  32), -INT8_C(  20), -INT8_C( 126),  INT8_C(  70),
        -INT8_C(  37),  INT8_C(  27),  INT8_C(  71), -INT8_C( 102), -INT8_C(  97), -INT8_C(  11), -INT8_C(   2), -INT8_C(  52) },
      { -INT8_C(  19),  INT8_C(  78), -INT8_C(  10), -INT8_C( 106), -INT8_C(  74), -INT8_C(  90),  INT8_C(  26), -INT8_C( 100),
        -INT8_C(  54), -INT8_C(  17), -INT8_C(  58),  INT8_C( 109),  INT8_C(  58),  INT8_C(  66), -INT8_C(  76),  INT8_C(   9) },
      UINT16_C(47175) },
    { {  INT8_C( 111), -INT8_C(  95), -INT8_C(  77),  INT8_C(  79), -INT8_C( 115),  INT8_C(  53), -INT8_C( 107),  INT8_C( 104),
         INT8_C(  80), -INT8_C(  35),  INT8_C(   2), -INT8_C(  17), -INT8_C(  46),  INT8_C(   0), -INT8_C(  68), -INT8_C(  65) },
      {  INT8_C(  78), -INT8_C(  78),  INT8_C(  85),  INT8_C(   5),  INT8_C(  88),  INT8_C( 111), -INT8_C(  95),  INT8_C(  34),
         INT8_C(  94),  INT8_C( 104), -INT8_C( 112), -INT8_C( 103), -INT8_C(  86),  INT8_C(  68), -INT8_C(  94),  INT8_C(  25) },
      UINT16_C(41846) },
    { { -INT8_C(  27),  INT8_C(  86),  INT8_C( 105),  INT8_C( 114), -INT8_C( 117), -INT8_C(   2), -INT8_C(  38), -INT8_C(  36),
        -INT8_C(  37), -INT8_C(  35), -INT8_C(  53), -INT8_C(  83), -INT8_C(  35), -INT8_C( 121),  INT8_C( 109),  INT8_C(  44) },
      {  INT8_C(  57), -INT8_C(  62),  INT8_C(  49), -INT8_C( 111),  INT8_C(  50), -INT8_C(  46), -INT8_C(  76), -INT8_C( 112),
         INT8_C(  58),  INT8_C(  68),  INT8_C(  41), -INT8_C(  27), -INT8_C( 120), -INT8_C(  52), -INT8_C(   2),  INT8_C( 109) },
      UINT16_C(44817) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cmplt_epi8_mask");

    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_cmplt_epi8_mask(a, b);

    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmplt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[8];
    int16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { {  INT16_C(  8688), -INT16_C(  2038), -INT16_C( 19603), -INT16_C(  7743),  INT16_C(  6364), -INT16_C(  8781), -INT16_C( 28786),  INT16_C( 14409) },
      { -INT16_C( 30446),  INT16_C(  8048), -INT16_C( 31123),  INT16_C(   374),  INT16_C( 10458),  INT16_C(  2276), -INT16_C( 19564), -INT16_C( 31571) },
      UINT8_C(122) },
    { { -INT16_C( 18476),  INT16_C( 16764),  INT16_C( 15978),  INT16_C( 18211), -INT16_C( 10666), -INT16_C(  7132),  INT16_C( 28006),  INT16_C( 30748) },
      { -INT16_C( 29194),  INT16_C( 25496),  INT16_C(  3603), -INT16_C(  4764),  INT16_C( 18742), -INT16_C( 13578), -INT16_C( 23556), -INT16_C( 12210) },
      UINT8_C( 18) },
    { { -INT16_C( 13477), -INT16_C( 15087),  INT16_C( 13321),  INT16_C( 24332),  INT16_C( 12299),  INT16_C( 28995),  INT16_C( 24733), -INT16_C( 27415) },
      { -INT16_C( 32275),  INT16_C(   247),  INT16_C( 23695), -INT16_C( 14611), -INT16_C(  7259), -INT16_C( 24176), -INT16_C(  8313), -INT16_C(  7567) },
      UINT8_C(134) },
    { { -INT16_C( 32086), -INT16_C( 19545), -INT16_C( 19273), -INT16_C( 15854),  INT16_C( 21988), -INT16_C( 32205),  INT16_C(  7349), -INT16_C( 24042) },
      {  INT16_C(  3486),  INT16_C( 11682), -INT16_C( 28567),  INT16_C(  3827), -INT16_C( 31629), -INT16_C(  1361),  INT16_C(  8291),  INT16_C(  3548) },
      UINT8_C(235) },
    { { -INT16_C( 31581),  INT16_C( 23232), -INT16_C( 11720),  INT16_C(  7196),  INT16_C( 20263), -INT16_C(  8802), -INT16_C( 19349),  INT16_C(  2431) },
      {  INT16_C(  8898),  INT16_C( 11063),  INT16_C( 10930),  INT16_C(  9530), -INT16_C(  5714),  INT16_C(  4384), -INT16_C(  1014), -INT16_C( 21218) },
      UINT8_C(109) },
    { { -INT16_C(  8576), -INT16_C( 18425),  INT16_C(  9136), -INT16_C( 10027),  INT16_C( 29554), -INT16_C(  8779),  INT16_C( 13352), -INT16_C(  5401) },
      {  INT16_C(  7766),  INT16_C(  2069),  INT16_C( 20296), -INT16_C(  2258),  INT16_C( 20025),  INT16_C( 17160),  INT16_C( 10058), -INT16_C( 13328) },
      UINT8_C( 47) },
    { { -INT16_C(  2299), -INT16_C( 18813),  INT16_C( 22554), -INT16_C( 29554),  INT16_C( 17356), -INT16_C(  2967),  INT16_C( 20599), -INT16_C( 12578) },
      { -INT16_C(  3218), -INT16_C( 18474),  INT16_C(  1091),  INT16_C( 31918), -INT16_C( 18862), -INT16_C( 25153), -INT16_C( 20515), -INT16_C(  7320) },
      UINT8_C(138) },
    { { -INT16_C(  5210), -INT16_C( 16231),  INT16_C( 10052),  INT16_C(  4172), -INT16_C( 19094), -INT16_C(  7932), -INT16_C(  7674),  INT16_C( 29871) },
      { -INT16_C( 31019),  INT16_C(  6187), -INT16_C(  9846), -INT16_C(  8812),  INT16_C( 21392),  INT16_C( 28026), -INT16_C(  7678), -INT16_C( 22448) },
      UINT8_C( 50) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epi16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmplt_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_cmplt_epi16_mask(a, b);

    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmplt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[4];
    int32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { {  INT32_C(   342509471),  INT32_C(  1807707120), -INT32_C(  1883067433), -INT32_C(  1776246145) },
      { -INT32_C(  1986807670), -INT32_C(    46382240),  INT32_C(  1166105412), -INT32_C(   664632007) },
      UINT8_C( 12) },
    { { -INT32_C(  1108554548),  INT32_C(   237546551), -INT32_C(   694293929),  INT32_C(   376225420) },
      { -INT32_C(   677314698), -INT32_C(  2032870334),  INT32_C(  1909216567), -INT32_C(  1505153318) },
      UINT8_C(  5) },
    { {  INT32_C(   828585722),  INT32_C(   960466146),  INT32_C(    51371126),  INT32_C(   286882714) },
      { -INT32_C(  1125598854), -INT32_C(   851264363), -INT32_C(   348254703),  INT32_C(   915507004) },
      UINT8_C(  8) },
    { { -INT32_C(  1620576835), -INT32_C(   136796543),  INT32_C(   502982531), -INT32_C(   600959902) },
      {  INT32_C(  1670911693), -INT32_C(   483337262),  INT32_C(   651063017), -INT32_C(  1285791499) },
      UINT8_C(  5) },
    { { -INT32_C(   699218859), -INT32_C(   305321110),  INT32_C(  1963706386), -INT32_C(  1454294564) },
      {  INT32_C(   571271760), -INT32_C(  1375322939), -INT32_C(  1596664662), -INT32_C(  1991036620) },
      UINT8_C(  1) },
    { {  INT32_C(  1600103925), -INT32_C(   481481264), -INT32_C(   782739211), -INT32_C(   512054895) },
      {  INT32_C(  1476691859),  INT32_C(  1845955267),  INT32_C(   319740894),  INT32_C(    27025676) },
      UINT8_C( 14) },
    { { -INT32_C(   681444346),  INT32_C(   515550761), -INT32_C(  1745939962),  INT32_C(  1333356987) },
      { -INT32_C(  1264091664),  INT32_C(  1696771719), -INT32_C(  1770508150), -INT32_C(  1751640687) },
      UINT8_C(  2) },
    { {  INT32_C(   980351249), -INT32_C(  1386731353), -INT32_C(   163297478), -INT32_C(  1589264976) },
      { -INT32_C(  1051333574),  INT32_C(   623343515),  INT32_C(   951820199), -INT32_C(   976268364) },
      UINT8_C( 14) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epi32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmplt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_cmplt_epi32_mask(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmplt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[2];
    int64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { -INT64_C( 5647861073004593668),  INT64_C( 7513159270578101773) },
      { -INT64_C( 4158000517519879220), -INT64_C( 3722011992148576727) },
      UINT8_C(  1) },
    { { -INT64_C(  596861189713615426),  INT64_C( 6563735385959986211) },
      {  INT64_C( 7474804671959035494),  INT64_C( 8713037812717713485) },
      UINT8_C(  3) },
    { {  INT64_C( 8787427045161226515),  INT64_C( 4221070030989279658) },
      {  INT64_C( 4923818643163127982), -INT64_C( 1579272704892999197) },
      UINT8_C(  0) },
    { {  INT64_C( 8425349565949318172),  INT64_C(  335871482417452443) },
      { -INT64_C( 6441197761245161375),  INT64_C( 7417244181946695214) },
      UINT8_C(  2) },
    { {  INT64_C( 5416639235809138156),  INT64_C( 8877549354110081241) },
      { -INT64_C( 7591538000592100548),  INT64_C( 3251187995882809454) },
      UINT8_C(  0) },
    { { -INT64_C( 7756870112092293537), -INT64_C( 5368864185469692697) },
      {  INT64_C( 3465619700429639962), -INT64_C( 7155931471639170238) },
      UINT8_C(  1) },
    { { -INT64_C( 8409340199193307652),  INT64_C( 2325838755904950042) },
      { -INT64_C( 7472392802704098213),  INT64_C( 3983096572301554797) },
      UINT8_C(  3) },
    { {  INT64_C( 1265857802915476066), -INT64_C( 8975415498965246390) },
      {  INT64_C( 4167778867129357031),  INT64_C( 2419041851708535329) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epi64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmplt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_cmplt_epi64_mask(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int8_t a[16];
    int8_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(17708),
      {  INT8_C( 102),  INT8_C(  87), -INT8_C(  59),  INT8_C(  93),  INT8_C( 103), -INT8_C(  18),  INT8_C(  55), -INT8_C(  33),
         INT8_C(  23), -INT8_C(  73),  INT8_C(  52),  INT8_C( 114), -INT8_C(  57),  INT8_C( 119), -INT8_C( 101),  INT8_C(  44) },
      {  INT8_C(  21),  INT8_C( 125),  INT8_C( 108),  INT8_C(  36),  INT8_C(  29),  INT8_C(  60), -INT8_C(  59), -INT8_C( 100),
        -INT8_C(   7), -INT8_C(  21), -INT8_C(  91),  INT8_C(  62),  INT8_C(  42), -INT8_C(  46), -INT8_C( 125), -INT8_C( 112) },
      UINT16_C(   36) },
    { UINT16_C(18473),
      { -INT8_C(  18), -INT8_C( 112),  INT8_C(  55),  INT8_C(  37),  INT8_C( 112),  INT8_C(  78), -INT8_C(  35), -INT8_C(  92),
        -INT8_C(  64), -INT8_C(  92),  INT8_C(  27),  INT8_C(  91), -INT8_C(  48),  INT8_C(  48), -INT8_C(  39),  INT8_C(  60) },
      {  INT8_C(  84), -INT8_C(  10),  INT8_C( 121),  INT8_C(  26), -INT8_C( 110),  INT8_C( 114),  INT8_C(   5),  INT8_C(  55),
        -INT8_C(  80),  INT8_C(  48),  INT8_C(   9),  INT8_C(  52), -INT8_C(  64),  INT8_C(  51),  INT8_C( 124), -INT8_C(  82) },
      UINT16_C(16417) },
    { UINT16_C(46019),
      { -INT8_C(  44),  INT8_C(  51),  INT8_C(   1), -INT8_C(  79), -INT8_C(  41), -INT8_C(  62),  INT8_C(  85), -INT8_C(  13),
         INT8_C(  29),  INT8_C(  38),  INT8_C(  35), -INT8_C(  10),  INT8_C(  98),  INT8_C( 120), -INT8_C(  20), -INT8_C(  37) },
      { -INT8_C( 110),  INT8_C( 126),  INT8_C(  77), -INT8_C( 105), -INT8_C(  74), -INT8_C(   2), -INT8_C(  57), -INT8_C(  65),
         INT8_C(  50), -INT8_C( 120), -INT8_C(  14), -INT8_C(  82),  INT8_C(  54), -INT8_C(  74),  INT8_C(  98),  INT8_C(  10) },
      UINT16_C(33026) },
    { UINT16_C(25577),
      { -INT8_C(  69), -INT8_C(  63),  INT8_C(  37),  INT8_C(  17), -INT8_C(  76),  INT8_C(  67),  INT8_C(  55), -INT8_C(  41),
         INT8_C(  57), -INT8_C( 103),  INT8_C(  79),  INT8_C(  38),  INT8_C( 117), -INT8_C(  31), -INT8_C(  92), -INT8_C(  62) },
      {  INT8_C( 121),  INT8_C(  90), -INT8_C(  64),  INT8_C(  64),  INT8_C(  26), -INT8_C(  14), -INT8_C(  56),  INT8_C(  12),
        -INT8_C(  95), -INT8_C(   1), -INT8_C(  62),  INT8_C(   3),  INT8_C(   9), -INT8_C(  84),  INT8_C( 102), -INT8_C(  59) },
      UINT16_C(17033) },
    { UINT16_C(35949),
      { -INT8_C(  42),  INT8_C(  33), -INT8_C(  49),  INT8_C(  13), -INT8_C(   8),  INT8_C(   8), -INT8_C(  90),  INT8_C(  72),
         INT8_C(  46),  INT8_C(  27),  INT8_C(  41), -INT8_C(  45), -INT8_C(  34), -INT8_C(  94),  INT8_C(  45), -INT8_C(  98) },
      { -INT8_C(  29),  INT8_C(  71), -INT8_C( 111), -INT8_C(  85),  INT8_C(  84),  INT8_C(  50), -INT8_C(  86),  INT8_C(  22),
         INT8_C(  53), -INT8_C(  76), -INT8_C(  62), -INT8_C( 101),  INT8_C( 121),  INT8_C(  47),  INT8_C(  39),  INT8_C(  79) },
      UINT16_C(32865) },
    { UINT16_C(63056),
      {  INT8_C(  92),  INT8_C(  73), -INT8_C(   1),  INT8_C(   2), -INT8_C( 111),  INT8_C(  45),  INT8_C(  30), -INT8_C(  70),
         INT8_C(   0), -INT8_C(   4),  INT8_C(  93),  INT8_C(  46), -INT8_C( 102),  INT8_C(  64),  INT8_C( 117),  INT8_C(  43) },
      { -INT8_C(  21), -INT8_C(  55),  INT8_C(  93), -INT8_C( 106), -INT8_C(  32), -INT8_C( 110),  INT8_C(  74), -INT8_C(  94),
         INT8_C(  46), -INT8_C(  61), -INT8_C(  46),  INT8_C(  85),  INT8_C(  18),  INT8_C(  34),  INT8_C(  76),  INT8_C( 110) },
      UINT16_C(36944) },
    { UINT16_C(19307),
      {  INT8_C( 112), -INT8_C(   4),  INT8_C( 120), -INT8_C( 114), -INT8_C(  73),  INT8_C( 121), -INT8_C( 118),  INT8_C(  20),
        -INT8_C(  89),  INT8_C(  37),  INT8_C(  84),  INT8_C(  28),  INT8_C(  80),  INT8_C(  63), -INT8_C(  26), -INT8_C(  82) },
      { -INT8_C(  43), -INT8_C(  58),  INT8_C(  64),  INT8_C(  31),  INT8_C( 104),  INT8_C( 110), -INT8_C(  30),  INT8_C(  58),
        -INT8_C(  60), -INT8_C(  12),  INT8_C(  93),  INT8_C(  16),  INT8_C(  98), -INT8_C(  56),  INT8_C(  91), -INT8_C(  45) },
      UINT16_C(16712) },
    { UINT16_C(54213),
      {  INT8_C(  97),  INT8_C( 124),  INT8_C(  76), -INT8_C(  20), -INT8_C( 112), -INT8_C(  13),  INT8_C(  17), -INT8_C(  28),
         INT8_C(  16),  INT8_C(  97),  INT8_C(  35), -INT8_C(  10),  INT8_C(  15), -INT8_C(   7), -INT8_C(  68),  INT8_C(  80) },
      {  INT8_C(  24),  INT8_C(  36), -INT8_C(  66), -INT8_C(   5),  INT8_C(  95), -INT8_C( 126), -INT8_C(  17), -INT8_C(  68),
        -INT8_C( 110),  INT8_C(  82), -INT8_C( 124), -INT8_C(  19),  INT8_C(  37),  INT8_C(  73), -INT8_C(  63), -INT8_C( 122) },
      UINT16_C(20480) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epi8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_mask_cmplt_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int16_t a[8];
    int16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(200),
      { -INT16_C( 13569), -INT16_C( 16692),  INT16_C( 11445), -INT16_C(  2780), -INT16_C( 27690), -INT16_C( 26956),  INT16_C( 21642), -INT16_C( 26699) },
      { -INT16_C( 14176), -INT16_C( 14332), -INT16_C( 22785), -INT16_C(  1488),  INT16_C(  3993),  INT16_C( 22342), -INT16_C( 15554),  INT16_C( 15647) },
      UINT8_C(136) },
    { UINT8_C(141),
      { -INT16_C(  1044),  INT16_C(  6210),  INT16_C( 14111), -INT16_C( 19729), -INT16_C( 31253),  INT16_C( 16445), -INT16_C( 11206),  INT16_C(   992) },
      { -INT16_C( 22312),  INT16_C( 32514), -INT16_C(   808), -INT16_C(  6376),  INT16_C( 28482),  INT16_C(  1317),  INT16_C( 25230),  INT16_C( 31378) },
      UINT8_C(136) },
    { UINT8_C( 93),
      { -INT16_C( 27691),  INT16_C(  3197),  INT16_C( 12162),  INT16_C(  2040),  INT16_C( 14444),  INT16_C( 16705),  INT16_C( 17432), -INT16_C( 16359) },
      { -INT16_C( 26553),  INT16_C( 17305), -INT16_C( 32592),  INT16_C(  8070), -INT16_C( 29786),  INT16_C(  2222),  INT16_C( 10270), -INT16_C(  3226) },
      UINT8_C(  9) },
    { UINT8_C(187),
      { -INT16_C(    29),  INT16_C(  4669),  INT16_C( 17655),  INT16_C( 12159), -INT16_C( 16250), -INT16_C( 13753),  INT16_C(  2265),  INT16_C( 29201) },
      {  INT16_C( 21921),  INT16_C(  8482),  INT16_C( 17115),  INT16_C( 26311), -INT16_C( 12048),  INT16_C(  6276),  INT16_C( 30518),  INT16_C(  6612) },
      UINT8_C( 59) },
    { UINT8_C(119),
      {  INT16_C( 11025),  INT16_C( 22126), -INT16_C( 24918),  INT16_C( 27356), -INT16_C( 22811), -INT16_C(  4796), -INT16_C( 18760),  INT16_C(  3470) },
      { -INT16_C( 20264),  INT16_C(  6888),  INT16_C( 20087),  INT16_C( 18186),  INT16_C(  9171),  INT16_C( 19069), -INT16_C( 26889),  INT16_C(  2241) },
      UINT8_C( 52) },
    { UINT8_C(194),
      {  INT16_C( 24112), -INT16_C( 12692), -INT16_C( 10438), -INT16_C(  7757), -INT16_C( 24293), -INT16_C( 11879), -INT16_C( 22993), -INT16_C(  8279) },
      { -INT16_C( 15218), -INT16_C(  9129), -INT16_C( 24882), -INT16_C(  3665), -INT16_C(  1508), -INT16_C( 19736), -INT16_C(  3653), -INT16_C(  5260) },
      UINT8_C(194) },
    { UINT8_C( 79),
      { -INT16_C( 17951), -INT16_C( 18294),  INT16_C( 27501),  INT16_C(  3795), -INT16_C( 23548), -INT16_C( 21955),  INT16_C(  7501),  INT16_C(  4408) },
      {  INT16_C(  5236),  INT16_C(  4832), -INT16_C( 11836), -INT16_C( 16850), -INT16_C(  7750), -INT16_C( 21639),  INT16_C( 25941),  INT16_C( 14074) },
      UINT8_C( 67) },
    { UINT8_C( 30),
      { -INT16_C(  4476), -INT16_C(  4213), -INT16_C( 26175),  INT16_C( 26099), -INT16_C( 25129), -INT16_C(  2893), -INT16_C( 15147), -INT16_C(  5528) },
      {  INT16_C( 31396),  INT16_C( 30382),  INT16_C( 27817), -INT16_C( 30160), -INT16_C(  9243),  INT16_C( 19167),  INT16_C(  5845),  INT16_C( 23145) },
      UINT8_C( 22) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epi16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_mask_cmplt_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int32_t a[4];
    int32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(197),
      { -INT32_C(  1144735140),  INT32_C(   953910229),  INT32_C(  1863469959), -INT32_C(   333716736) },
      { -INT32_C(  1982813975),  INT32_C(   450621222),  INT32_C(   739552407), -INT32_C(   420358007) },
      UINT8_C(  0) },
    { UINT8_C(146),
      {  INT32_C(   946315701), -INT32_C(   943677572), -INT32_C(  1782042703), -INT32_C(   226577334) },
      {  INT32_C(  1947731845), -INT32_C(  1895091229),  INT32_C(   521746503), -INT32_C(   541917399) },
      UINT8_C(  0) },
    { UINT8_C(160),
      { -INT32_C(  1189341415),  INT32_C(   107668695),  INT32_C(  1615921068), -INT32_C(  2048572802) },
      { -INT32_C(  1905764005), -INT32_C(  1646921627),  INT32_C(   281474321), -INT32_C(  1062165081) },
      UINT8_C(  0) },
    { UINT8_C(189),
      { -INT32_C(  1332446772), -INT32_C(   480470044),  INT32_C(   778157547), -INT32_C(    58071134) },
      {  INT32_C(  1197545295), -INT32_C(   514261268),  INT32_C(  1787324612), -INT32_C(   450410216) },
      UINT8_C(  5) },
    { UINT8_C(194),
      {  INT32_C(  1453757883), -INT32_C(  1354593550), -INT32_C(   766414613),  INT32_C(   270683641) },
      { -INT32_C(  1392744018),  INT32_C(   712105666), -INT32_C(  1354507418), -INT32_C(  1116592126) },
      UINT8_C(  2) },
    { UINT8_C(189),
      { -INT32_C(  1565584360), -INT32_C(   963748266), -INT32_C(    37789520), -INT32_C(   324219006) },
      { -INT32_C(  1448191797), -INT32_C(  1542399799),  INT32_C(  1135001371),  INT32_C(  1241605169) },
      UINT8_C( 13) },
    { UINT8_C(120),
      {  INT32_C(   265219248), -INT32_C(   608201606),  INT32_C(   576568403), -INT32_C(  1041413784) },
      { -INT32_C(   796223497),  INT32_C(  1726689191),  INT32_C(   966275029), -INT32_C(   525213136) },
      UINT8_C(  8) },
    { UINT8_C(206),
      {  INT32_C(   323612543),  INT32_C(  1801856174), -INT32_C(   875329407),  INT32_C(   230855798) },
      {  INT32_C(  1320456735), -INT32_C(  1390208386), -INT32_C(  1797431886), -INT32_C(  1906066162) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epi32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_mask_cmplt_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t a[2];
    int64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 83),
      {  INT64_C( 6518896244511925628), -INT64_C( 7942494022464116321) },
      { -INT64_C( 1165695184279065282),  INT64_C(  519768334569897933) },
      UINT8_C(  2) },
    { UINT8_C(123),
      { -INT64_C( 1270738553217511944),  INT64_C( 6457955897836758474) },
      {  INT64_C( 7086606851453784301),  INT64_C( 7658641035478514487) },
      UINT8_C(  3) },
    { UINT8_C(130),
      {  INT64_C( 7679609903262345051),  INT64_C( 2618965866558452775) },
      {  INT64_C( 8581020008858444591), -INT64_C( 8841320149695274282) },
      UINT8_C(  0) },
    { UINT8_C(122),
      { -INT64_C( 2923002312043638829),  INT64_C( 1521248223038790270) },
      { -INT64_C( 1020972344001103339), -INT64_C( 4648662680333475920) },
      UINT8_C(  0) },
    { UINT8_C(240),
      {  INT64_C( 5792378398845306747),  INT64_C( 7402113955415609108) },
      {  INT64_C( 3053272922022393832),  INT64_C( 6418990518974702211) },
      UINT8_C(  0) },
    { UINT8_C(221),
      { -INT64_C( 1250693791274025498), -INT64_C( 5141209711364126682) },
      {  INT64_C( 1666733963915515191),  INT64_C( 4626691437356205443) },
      UINT8_C(  1) },
    { UINT8_C(140),
      { -INT64_C( 4876525984747368311),  INT64_C(  967277449718862196) },
      {  INT64_C( 8190034706008509112), -INT64_C( 4636162014345231922) },
      UINT8_C(  0) },
    { UINT8_C(219),
      {  INT64_C( 5864263137805462740),  INT64_C( 1006297244938901449) },
      { -INT64_C( 4143404266587349544),  INT64_C( 1971657747896049014) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epi64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_mask_cmplt_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmplt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[16];
    uint8_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { { UINT8_C(  6), UINT8_C(248), UINT8_C(250), UINT8_C(115), UINT8_C( 96), UINT8_C(191), UINT8_C(231), UINT8_C( 14),
        UINT8_C(214), UINT8_C(162), UINT8_C(128), UINT8_C( 26), UINT8_C( 65), UINT8_C(242), UINT8_C( 40), UINT8_C(203) },
      { UINT8_C(249), UINT8_C(238), UINT8_C(111), UINT8_C(118), UINT8_C( 68), UINT8_C( 62), UINT8_C(184), UINT8_C(144),
        UINT8_C(149), UINT8_C( 32), UINT8_C(  8), UINT8_C(130), UINT8_C(169), UINT8_C(244), UINT8_C( 94), UINT8_C(175) },
      UINT16_C(30857) },
    { { UINT8_C(236), UINT8_C( 88), UINT8_C( 35), UINT8_C( 76), UINT8_C( 24), UINT8_C( 10), UINT8_C( 91), UINT8_C(238),
        UINT8_C(172), UINT8_C(219), UINT8_C(  8), UINT8_C(238), UINT8_C(205), UINT8_C( 49), UINT8_C(185), UINT8_C(198) },
      { UINT8_C( 31), UINT8_C( 40), UINT8_C( 61), UINT8_C( 99), UINT8_C(102), UINT8_C(245), UINT8_C(244), UINT8_C(251),
        UINT8_C( 21), UINT8_C(252), UINT8_C(126), UINT8_C(191), UINT8_C(240), UINT8_C(220), UINT8_C(110), UINT8_C(220) },
      UINT16_C(46844) },
    { { UINT8_C( 52), UINT8_C(145), UINT8_C( 41), UINT8_C( 76), UINT8_C(156), UINT8_C(132), UINT8_C( 58), UINT8_C( 72),
        UINT8_C( 95), UINT8_C( 67), UINT8_C( 54), UINT8_C( 44), UINT8_C(116), UINT8_C(239), UINT8_C(243), UINT8_C(147) },
      { UINT8_C( 24), UINT8_C( 48), UINT8_C(247), UINT8_C(126), UINT8_C( 37), UINT8_C(235), UINT8_C(122), UINT8_C( 59),
        UINT8_C(231), UINT8_C(248), UINT8_C(250), UINT8_C(216), UINT8_C(212), UINT8_C(104), UINT8_C(180), UINT8_C(  8) },
      UINT16_C( 8044) },
    { { UINT8_C(250), UINT8_C(221), UINT8_C( 85), UINT8_C(150), UINT8_C( 97), UINT8_C(143), UINT8_C(222), UINT8_C(192),
        UINT8_C(210), UINT8_C( 21), UINT8_C(237), UINT8_C( 70), UINT8_C(  4), UINT8_C(224), UINT8_C(218), UINT8_C( 28) },
      { UINT8_C( 16), UINT8_C(209), UINT8_C(155), UINT8_C( 53), UINT8_C(188), UINT8_C( 21), UINT8_C(112), UINT8_C(163),
        UINT8_C( 13), UINT8_C(106), UINT8_C(123), UINT8_C(225), UINT8_C(211), UINT8_C( 48), UINT8_C(233), UINT8_C(205) },
      UINT16_C(55828) },
    { { UINT8_C( 13), UINT8_C( 62), UINT8_C( 99), UINT8_C(111), UINT8_C(206), UINT8_C( 65), UINT8_C( 47), UINT8_C(160),
        UINT8_C( 86), UINT8_C( 28), UINT8_C(231), UINT8_C( 91), UINT8_C(252), UINT8_C(193), UINT8_C(119), UINT8_C( 12) },
      { UINT8_C(146), UINT8_C( 18), UINT8_C( 66), UINT8_C( 78), UINT8_C( 39), UINT8_C(178), UINT8_C(241), UINT8_C( 52),
        UINT8_C( 29), UINT8_C(109), UINT8_C( 21), UINT8_C(240), UINT8_C(157),    UINT8_MAX, UINT8_C(189), UINT8_C(170) },
      UINT16_C(60001) },
    { { UINT8_C( 61), UINT8_C( 32), UINT8_C( 25), UINT8_C( 11), UINT8_C( 97), UINT8_C( 73), UINT8_C(172), UINT8_C(184),
        UINT8_C(101), UINT8_C(147), UINT8_C( 19), UINT8_C( 98), UINT8_C( 84), UINT8_C(138), UINT8_C(110), UINT8_C(230) },
      { UINT8_C(157), UINT8_C(176), UINT8_C( 52), UINT8_C(196), UINT8_C( 99), UINT8_C( 37), UINT8_C(249), UINT8_C(128),
        UINT8_C(146), UINT8_C( 14), UINT8_C(112), UINT8_C( 47), UINT8_C( 13), UINT8_C( 45), UINT8_C(218), UINT8_C( 75) },
      UINT16_C(17759) },
    { { UINT8_C( 77), UINT8_C(243), UINT8_C( 86), UINT8_C(174), UINT8_C( 60), UINT8_C(  2), UINT8_C(102), UINT8_C(162),
        UINT8_C(149), UINT8_C(121), UINT8_C(  4), UINT8_C(233), UINT8_C(  4), UINT8_C(114), UINT8_C(207), UINT8_C(161) },
      { UINT8_C( 35), UINT8_C(  3), UINT8_C(101), UINT8_C(134), UINT8_C( 41), UINT8_C( 94), UINT8_C(  6), UINT8_C(187),
        UINT8_C(109), UINT8_C(118), UINT8_C(235), UINT8_C(122), UINT8_C(163), UINT8_C(197), UINT8_C(197), UINT8_C(240) },
      UINT16_C(46244) },
    { { UINT8_C(184), UINT8_C( 28), UINT8_C(158), UINT8_C(245), UINT8_C( 30), UINT8_C(  5), UINT8_C(151), UINT8_C(180),
        UINT8_C(126), UINT8_C(155), UINT8_C(157), UINT8_C(130), UINT8_C( 13), UINT8_C(109), UINT8_C( 35), UINT8_C( 48) },
      { UINT8_C(112), UINT8_C(137), UINT8_C(182), UINT8_C(153), UINT8_C(231), UINT8_C(188), UINT8_C( 85), UINT8_C( 84),
        UINT8_C( 50), UINT8_C( 64), UINT8_C(207), UINT8_C(213), UINT8_C(  5), UINT8_C(148), UINT8_C(197), UINT8_C(189) },
      UINT16_C(60470) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epu8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmplt_epu8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__mmask16 r = easysimd_mm_cmplt_epu8_mask(a, b);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmplt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[8];
    uint16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT16_C(13792), UINT16_C(64405), UINT16_C(54354), UINT16_C(20169), UINT16_C( 8540), UINT16_C(45485), UINT16_C(50470), UINT16_C(19518) },
      { UINT16_C(43228), UINT16_C(25248), UINT16_C(55328), UINT16_C(41925), UINT16_C(22462), UINT16_C(29037), UINT16_C( 3929), UINT16_C(14621) },
      UINT8_C( 29) },
    { { UINT16_C(45636), UINT16_C(38708), UINT16_C(64902), UINT16_C(58341), UINT16_C(37407), UINT16_C(17812), UINT16_C(53847), UINT16_C(13202) },
      { UINT16_C(12922), UINT16_C(39574), UINT16_C(23306), UINT16_C(51261), UINT16_C(43954), UINT16_C( 2874), UINT16_C(22458), UINT16_C(65092) },
      UINT8_C(146) },
    { { UINT16_C(30730), UINT16_C(37013), UINT16_C(31350), UINT16_C(38259), UINT16_C( 2060), UINT16_C(25818), UINT16_C(27866), UINT16_C(21911) },
      { UINT16_C(11678), UINT16_C(43503), UINT16_C(11657), UINT16_C(15217), UINT16_C(43992), UINT16_C(37447), UINT16_C(35587), UINT16_C( 3472) },
      UINT8_C(114) },
    { { UINT16_C( 9732), UINT16_C(31389), UINT16_C( 4512), UINT16_C(44303), UINT16_C(59673), UINT16_C(62225), UINT16_C(43094), UINT16_C(62536) },
      { UINT16_C(14550), UINT16_C(24477), UINT16_C( 3941), UINT16_C(15770), UINT16_C(57786), UINT16_C(48591), UINT16_C(24429), UINT16_C(29130) },
      UINT8_C(  1) },
    { { UINT16_C(26757), UINT16_C( 9963), UINT16_C(64121), UINT16_C(37587), UINT16_C(58595), UINT16_C(14725), UINT16_C(52876), UINT16_C(25134) },
      { UINT16_C(51974), UINT16_C(27585), UINT16_C(23770), UINT16_C(38312), UINT16_C(30525), UINT16_C(43602), UINT16_C( 7638), UINT16_C(23579) },
      UINT8_C( 43) },
    { { UINT16_C( 1669), UINT16_C(65154), UINT16_C(21760), UINT16_C(58512), UINT16_C( 5433), UINT16_C(50461), UINT16_C(19427), UINT16_C(59688) },
      { UINT16_C(59671), UINT16_C(61780), UINT16_C(64581), UINT16_C(33670), UINT16_C(55667), UINT16_C(18989), UINT16_C(18934), UINT16_C(31654) },
      UINT8_C( 21) },
    { { UINT16_C(10319), UINT16_C(20601), UINT16_C( 2429), UINT16_C(46644), UINT16_C(20766), UINT16_C(  635), UINT16_C(41885), UINT16_C(46315) },
      { UINT16_C(16525), UINT16_C(53925), UINT16_C(11324), UINT16_C(45141), UINT16_C(33541), UINT16_C(64506), UINT16_C(41164), UINT16_C( 7030) },
      UINT8_C( 55) },
    { { UINT16_C(61384), UINT16_C(17771), UINT16_C(40952), UINT16_C( 5883), UINT16_C(30449), UINT16_C(36376), UINT16_C( 1050), UINT16_C(42818) },
      { UINT16_C(59204), UINT16_C(32889), UINT16_C(53011), UINT16_C( 6192), UINT16_C(10834), UINT16_C( 7699), UINT16_C(35274), UINT16_C(37433) },
      UINT8_C( 78) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmplt_epu16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 r = easysimd_mm_cmplt_epu16_mask(a, b);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmplt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t a[4];
    uint32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT32_C(1476202829), UINT32_C(1165709476), UINT32_C(1872611479), UINT32_C( 758704402) },
      { UINT32_C( 138870557), UINT32_C(1151102663), UINT32_C( 539887219), UINT32_C( 674900955) },
      UINT8_C(  0) },
    { { UINT32_C(3766433596), UINT32_C( 522582663), UINT32_C(3582903234), UINT32_C(3372402348) },
      { UINT32_C(2379303110), UINT32_C( 718433718), UINT32_C(1330249844), UINT32_C(1752663084) },
      UINT8_C(  2) },
    { { UINT32_C(1112143547), UINT32_C(3009507057), UINT32_C(3716738865), UINT32_C(2091355062) },
      { UINT32_C(2315876563), UINT32_C(1521802214), UINT32_C( 145358556), UINT32_C(1030758530) },
      UINT8_C(  1) },
    { { UINT32_C( 125810966), UINT32_C(1505485096), UINT32_C(2251768784), UINT32_C(2718097102) },
      { UINT32_C(1009519702), UINT32_C(3281445095), UINT32_C(1623932894), UINT32_C(1990016095) },
      UINT8_C(  3) },
    { { UINT32_C( 494738677), UINT32_C(3463919869), UINT32_C(1247063676), UINT32_C(3807139724) },
      { UINT32_C(1243551843), UINT32_C(3591288312), UINT32_C(1412880885), UINT32_C( 197841685) },
      UINT8_C(  7) },
    { { UINT32_C(3978840304), UINT32_C(4240154496), UINT32_C(3645313101), UINT32_C(3401331559) },
      { UINT32_C(1142217548), UINT32_C(2233148048), UINT32_C( 299520508), UINT32_C( 354198565) },
      UINT8_C(  0) },
    { { UINT32_C(1828865516), UINT32_C( 845790948), UINT32_C( 889958606), UINT32_C( 822069220) },
      { UINT32_C( 863245218), UINT32_C( 850956086), UINT32_C( 105091809), UINT32_C( 588996663) },
      UINT8_C(  2) },
    { { UINT32_C(2307923365), UINT32_C(2847668955), UINT32_C(2396964778), UINT32_C( 834592142) },
      { UINT32_C( 660878321), UINT32_C(2740526274), UINT32_C(3869875375), UINT32_C(2701772028) },
      UINT8_C( 12) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmplt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__mmask8 r = easysimd_mm_cmplt_epu32_mask(a, b);

    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmplt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint64_t a[2];
    uint64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT64_C( 4781858612497707220), UINT64_C( 9116465292753226040) },
      { UINT64_C(16903419844893387135), UINT64_C( 6422839530179198771) },
      UINT8_C(  1) },
    { { UINT64_C(18322423144443622386), UINT64_C(12716890128920019227) },
      { UINT64_C(15696933279777203233), UINT64_C(17695733661032153268) },
      UINT8_C(  2) },
    { { UINT64_C(17787319511997319188), UINT64_C( 7359847456335711297) },
      { UINT64_C(12578880287226872850), UINT64_C( 4944805265320398760) },
      UINT8_C(  0) },
    { { UINT64_C( 1933161180161808643), UINT64_C( 5147246265465947844) },
      { UINT64_C(10420364067439952117), UINT64_C(   96223751803995361) },
      UINT8_C(  1) },
    { { UINT64_C( 3415895016304767859), UINT64_C(11166014081657730169) },
      { UINT64_C( 7556799962836235622), UINT64_C( 7541786605497424553) },
      UINT8_C(  1) },
    { { UINT64_C(13032040989352888385), UINT64_C( 2256399868453065488) },
      { UINT64_C(12122651157662193997), UINT64_C(14179295918935815528) },
      UINT8_C(  2) },
    { { UINT64_C(12065509091304373122), UINT64_C( 8662865383362777951) },
      { UINT64_C( 6714736153239266951), UINT64_C(18195809654896492349) },
      UINT8_C(  2) },
    { { UINT64_C(11617645753685112315), UINT64_C(14817390627892402174) },
      { UINT64_C(12892260324695338392), UINT64_C( 2339610271799223298) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmplt_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmplt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 r = easysimd_mm_cmplt_epu64_mask(a, b);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    uint8_t a[16];
    uint8_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C( 7497),
      { UINT8_C( 47), UINT8_C(245), UINT8_C(227), UINT8_C(158), UINT8_C( 92), UINT8_C(110), UINT8_C( 58), UINT8_C(211),
        UINT8_C( 24), UINT8_C( 53), UINT8_C( 89), UINT8_C( 38), UINT8_C( 92), UINT8_C( 66), UINT8_C(125), UINT8_C(220) },
      { UINT8_C(178), UINT8_C(  0), UINT8_C( 15), UINT8_C( 27), UINT8_C( 22), UINT8_C( 18), UINT8_C(105), UINT8_C(216),
        UINT8_C( 44), UINT8_C( 91), UINT8_C(147), UINT8_C(213), UINT8_C( 12), UINT8_C(220), UINT8_C(242), UINT8_C( 59) },
      UINT16_C( 3393) },
    { UINT16_C(54737),
      { UINT8_C(218), UINT8_C( 46), UINT8_C( 67), UINT8_C( 20), UINT8_C(  1), UINT8_C( 91), UINT8_C( 73), UINT8_C( 90),
        UINT8_C(129), UINT8_C(165), UINT8_C(156),    UINT8_MAX, UINT8_C(129), UINT8_C( 78),    UINT8_MAX, UINT8_C(144) },
      { UINT8_C(105), UINT8_C( 21), UINT8_C(162), UINT8_C(211), UINT8_C(237), UINT8_C(207), UINT8_C( 46), UINT8_C(129),
        UINT8_C(164), UINT8_C( 58), UINT8_C( 93), UINT8_C(150), UINT8_C(118), UINT8_C( 47), UINT8_C(107), UINT8_C( 80) },
      UINT16_C(  400) },
    { UINT16_C(44637),
      { UINT8_C(100), UINT8_C( 94), UINT8_C( 10), UINT8_C(173), UINT8_C(184), UINT8_C(139), UINT8_C( 83), UINT8_C( 84),
        UINT8_C(138), UINT8_C(212), UINT8_C(162), UINT8_C(137), UINT8_C(101), UINT8_C( 11), UINT8_C(159), UINT8_C(  7) },
      { UINT8_C(222), UINT8_C(140), UINT8_C(214), UINT8_C( 12), UINT8_C( 13), UINT8_C(122), UINT8_C( 71), UINT8_C(107),
        UINT8_C( 16), UINT8_C(189), UINT8_C(154), UINT8_C(123), UINT8_C( 13), UINT8_C(247), UINT8_C( 42), UINT8_C(113) },
      UINT16_C(40965) },
    { UINT16_C(13397),
      { UINT8_C( 30), UINT8_C( 13), UINT8_C(191), UINT8_C(113), UINT8_C( 97), UINT8_C( 74), UINT8_C( 70), UINT8_C(  3),
        UINT8_C(211), UINT8_C(171), UINT8_C( 14), UINT8_C(114), UINT8_C(178), UINT8_C(237),    UINT8_MAX, UINT8_C(137) },
      { UINT8_C(249), UINT8_C( 12), UINT8_C(  3), UINT8_C( 64), UINT8_C(119), UINT8_C( 20), UINT8_C(253), UINT8_C( 17),
        UINT8_C(143), UINT8_C( 10), UINT8_C(  8), UINT8_C(185), UINT8_C(123), UINT8_C( 93), UINT8_C(237), UINT8_C(154) },
      UINT16_C(   81) },
    { UINT16_C(44394),
      { UINT8_C( 11), UINT8_C(203), UINT8_C(247), UINT8_C( 81), UINT8_C(206), UINT8_C(202), UINT8_C(252), UINT8_C(221),
        UINT8_C( 61), UINT8_C(175), UINT8_C(202), UINT8_C( 60), UINT8_C( 56), UINT8_C(195), UINT8_C( 72), UINT8_C( 59) },
      { UINT8_C(  4), UINT8_C(192), UINT8_C( 79), UINT8_C(  1), UINT8_C(209), UINT8_C(223), UINT8_C( 12), UINT8_C(218),
        UINT8_C(152), UINT8_C(135), UINT8_C( 55), UINT8_C(134), UINT8_C( 33), UINT8_C(162), UINT8_C( 51), UINT8_C( 45) },
      UINT16_C( 2336) },
    { UINT16_C(10861),
      { UINT8_C(126), UINT8_C( 60), UINT8_C(244), UINT8_C(123), UINT8_C( 25), UINT8_C( 49), UINT8_C( 42), UINT8_C(227),
        UINT8_C(109), UINT8_C( 98), UINT8_C(166), UINT8_C(182), UINT8_C(157), UINT8_C(170), UINT8_C(118), UINT8_C(237) },
      { UINT8_C(172), UINT8_C( 71), UINT8_C(204), UINT8_C(184), UINT8_C( 33), UINT8_C(100), UINT8_C( 63), UINT8_C( 89),
        UINT8_C(234), UINT8_C( 97), UINT8_C(251), UINT8_C( 29), UINT8_C(142), UINT8_C(104), UINT8_C( 71), UINT8_C( 12) },
      UINT16_C(  105) },
    { UINT16_C(15524),
      { UINT8_C(135), UINT8_C(189), UINT8_C(109), UINT8_C(177), UINT8_C(160), UINT8_C(219), UINT8_C( 19), UINT8_C( 71),
        UINT8_C(145), UINT8_C(177), UINT8_C(241), UINT8_C(  7), UINT8_C(158), UINT8_C(157), UINT8_C( 78), UINT8_C(106) },
      { UINT8_C( 85), UINT8_C(112), UINT8_C(206), UINT8_C(149), UINT8_C(201), UINT8_C(185), UINT8_C(246), UINT8_C(196),
        UINT8_C(214), UINT8_C(132), UINT8_C( 44), UINT8_C( 30), UINT8_C(144), UINT8_C(209), UINT8_C( 90), UINT8_C( 24) },
      UINT16_C(10372) },
    { UINT16_C(51086),
      { UINT8_C(201), UINT8_C( 47), UINT8_C(162), UINT8_C(221), UINT8_C(118), UINT8_C( 51), UINT8_C(142), UINT8_C(103),
        UINT8_C( 58), UINT8_C( 44), UINT8_C(  5), UINT8_C(137), UINT8_C(150), UINT8_C( 90), UINT8_C(249), UINT8_C(100) },
      { UINT8_C(239), UINT8_C(194), UINT8_C( 29), UINT8_C(229), UINT8_C(134), UINT8_C(244), UINT8_C(105), UINT8_C(178),
        UINT8_C( 18), UINT8_C(250), UINT8_C(131), UINT8_C(108), UINT8_C( 18), UINT8_C( 18), UINT8_C( 51), UINT8_C(219) },
      UINT16_C(34442) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epu8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epu8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__mmask16 r = easysimd_mm_mask_cmplt_epu8_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint16_t a[8];
    uint16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 80),
      { UINT16_C(10010), UINT16_C(12409), UINT16_C(59724), UINT16_C(36894), UINT16_C(11257), UINT16_C(27877), UINT16_C(25860), UINT16_C(18519) },
      { UINT16_C(19236), UINT16_C(  928), UINT16_C( 2373), UINT16_C(26066), UINT16_C(17438), UINT16_C(34431), UINT16_C(28150), UINT16_C( 4310) },
      UINT8_C( 80) },
    { UINT8_C(148),
      { UINT16_C(16463), UINT16_C(14816), UINT16_C(28766), UINT16_C(35378), UINT16_C(40534), UINT16_C(48014), UINT16_C(55029), UINT16_C(16608) },
      { UINT16_C(58230), UINT16_C(32901), UINT16_C(60342), UINT16_C(64158), UINT16_C( 9322), UINT16_C(55536), UINT16_C(  250), UINT16_C(19052) },
      UINT8_C(132) },
    { UINT8_C( 64),
      { UINT16_C(33613), UINT16_C(48543), UINT16_C(10677), UINT16_C(21267), UINT16_C(53175), UINT16_C(36168), UINT16_C(35247), UINT16_C(37379) },
      { UINT16_C(33550), UINT16_C(63816), UINT16_C(17185), UINT16_C(17764), UINT16_C(15411), UINT16_C(13376), UINT16_C(35496), UINT16_C(62836) },
      UINT8_C( 64) },
    { UINT8_C( 13),
      { UINT16_C(45843), UINT16_C(15554), UINT16_C( 5574), UINT16_C(38387), UINT16_C(32861), UINT16_C(58948), UINT16_C(55172), UINT16_C( 2037) },
      { UINT16_C(60959), UINT16_C(25129), UINT16_C(28242), UINT16_C(36502), UINT16_C(51886), UINT16_C(14391), UINT16_C(11326), UINT16_C(21061) },
      UINT8_C(  5) },
    { UINT8_C(223),
      { UINT16_C(36359), UINT16_C( 7334), UINT16_C(15234), UINT16_C(  634), UINT16_C(24704), UINT16_C(22406), UINT16_C(36437), UINT16_C(17526) },
      { UINT16_C(55735), UINT16_C( 9622), UINT16_C( 9583), UINT16_C(14804), UINT16_C( 3164), UINT16_C(34935), UINT16_C(51538), UINT16_C(22888) },
      UINT8_C(203) },
    { UINT8_C( 88),
      { UINT16_C(30222), UINT16_C(18906), UINT16_C(56560), UINT16_C(20681), UINT16_C( 8291), UINT16_C(61862), UINT16_C(60055), UINT16_C(28840) },
      { UINT16_C(52608), UINT16_C(42463), UINT16_C( 6305), UINT16_C(44545), UINT16_C(35471), UINT16_C(22784), UINT16_C(23026), UINT16_C(  177) },
      UINT8_C( 24) },
    { UINT8_C(207),
      { UINT16_C(18827), UINT16_C(26559), UINT16_C( 4115), UINT16_C(13258), UINT16_C(48054), UINT16_C(41162), UINT16_C(14947), UINT16_C(12576) },
      { UINT16_C(50713), UINT16_C(12754), UINT16_C(32967), UINT16_C(20929), UINT16_C( 6784), UINT16_C(55875), UINT16_C(17355), UINT16_C(22185) },
      UINT8_C(205) },
    { UINT8_C(141),
      { UINT16_C(48489), UINT16_C(31136), UINT16_C(54152), UINT16_C(17199), UINT16_C(53150), UINT16_C(55463), UINT16_C(55535), UINT16_C(46578) },
      { UINT16_C( 9130), UINT16_C(11133), UINT16_C(52964), UINT16_C(65195), UINT16_C(34066), UINT16_C(21961), UINT16_C( 7983), UINT16_C(39138) },
      UINT8_C(  8) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epu16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epu16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 r = easysimd_mm_mask_cmplt_epu16_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint32_t a[4];
    uint32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(212),
      { UINT32_C(2631809006), UINT32_C(2836227474), UINT32_C( 475456845), UINT32_C(2617665890) },
      { UINT32_C(4256056936), UINT32_C(3051426202), UINT32_C( 960753375), UINT32_C(2467165349) },
      UINT8_C(  4) },
    { UINT8_C( 43),
      { UINT32_C(1354641643), UINT32_C( 580740925), UINT32_C( 646232509), UINT32_C( 110043328) },
      { UINT32_C(3768618191), UINT32_C(1556043118), UINT32_C(2348939416), UINT32_C(4038628357) },
      UINT8_C( 11) },
    { UINT8_C(196),
      { UINT32_C(3707847030), UINT32_C(2560238558), UINT32_C(3361259943), UINT32_C(3700907600) },
      { UINT32_C(1414166527), UINT32_C( 804103734), UINT32_C(1010071975), UINT32_C(2801804593) },
      UINT8_C(  0) },
    { UINT8_C(102),
      { UINT32_C( 625247234), UINT32_C(3771587615), UINT32_C(2469434676), UINT32_C(2744257580) },
      { UINT32_C(4258915927), UINT32_C(1302661331), UINT32_C(1652482365), UINT32_C(3821545185) },
      UINT8_C(  0) },
    { UINT8_C(170),
      { UINT32_C(3922266380), UINT32_C(1797106134), UINT32_C(3868700889), UINT32_C( 691878466) },
      { UINT32_C( 486291987), UINT32_C(3227077087), UINT32_C(4003642312), UINT32_C(2409137539) },
      UINT8_C( 10) },
    { UINT8_C(142),
      { UINT32_C( 191133794), UINT32_C(1189466006), UINT32_C(2693385062), UINT32_C(1119138312) },
      { UINT32_C(4163031214), UINT32_C(3837846057), UINT32_C( 174567044), UINT32_C(2828596806) },
      UINT8_C( 10) },
    { UINT8_C(111),
      { UINT32_C(3422926077), UINT32_C(1681083289), UINT32_C(2255279060), UINT32_C(1463135879) },
      { UINT32_C(3011522000), UINT32_C(2604098797), UINT32_C(3269542347), UINT32_C(3610348250) },
      UINT8_C( 14) },
    { UINT8_C( 62),
      { UINT32_C(2178392886), UINT32_C(2857778134), UINT32_C(1429331111), UINT32_C(1042712849) },
      { UINT32_C(1848367369), UINT32_C(1379517968), UINT32_C( 841808808), UINT32_C(1651508012) },
      UINT8_C(  8) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epu32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__mmask8 r = easysimd_mm_mask_cmplt_epu32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmplt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint64_t a[2];
    uint64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(189),
      { UINT64_C(11997223177898407471), UINT64_C( 9837134999224290106) },
      { UINT64_C( 3685842334879346355), UINT64_C( 8807864717929721788) },
      UINT8_C(  0) },
    { UINT8_C(136),
      { UINT64_C( 1647782596919299314), UINT64_C( 7324761711716561772) },
      { UINT64_C(13246683029767316059), UINT64_C( 9290219656851968914) },
      UINT8_C(  0) },
    { UINT8_C(137),
      { UINT64_C( 8567300060530091359), UINT64_C( 6956886588335773801) },
      { UINT64_C(14810966412568864076), UINT64_C( 4335238956462108450) },
      UINT8_C(  1) },
    { UINT8_C(  1),
      { UINT64_C( 8201190820230247485), UINT64_C( 3533442848532660457) },
      { UINT64_C( 9005362580019675831), UINT64_C(   49985738587609556) },
      UINT8_C(  1) },
    { UINT8_C( 73),
      { UINT64_C( 4114693809888258230), UINT64_C(16234340721904440900) },
      { UINT64_C(13109955541938872993), UINT64_C(  514105600409902916) },
      UINT8_C(  1) },
    { UINT8_C(224),
      { UINT64_C( 2926867321489799173), UINT64_C(11434333222944125207) },
      { UINT64_C( 6307132640356649304), UINT64_C( 9818040162153160935) },
      UINT8_C(  0) },
    { UINT8_C(231),
      { UINT64_C(12808908594963145305), UINT64_C( 5092388642311032682) },
      { UINT64_C(15410287089593282182), UINT64_C(15390454127714123006) },
      UINT8_C(  3) },
    { UINT8_C(194),
      { UINT64_C( 1268875185654759631), UINT64_C( 9832718747860659745) },
      { UINT64_C( 9378978522304768723), UINT64_C( 4237275281141716523) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmplt_epu64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmplt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 r = easysimd_mm_mask_cmplt_epu64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t a[32];
    int8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { {  INT8_C(  39),  INT8_C(   4), -INT8_C(  31), -INT8_C(  28), -INT8_C( 107), -INT8_C(   3), -INT8_C( 116),  INT8_C(  50),
         INT8_C(  15),  INT8_C(  40),  INT8_C(   7), -INT8_C( 120), -INT8_C( 117), -INT8_C( 111), -INT8_C(  88),  INT8_C(   3),
         INT8_C( 104), -INT8_C(  42), -INT8_C( 108), -INT8_C(  32), -INT8_C( 124), -INT8_C( 121), -INT8_C(  50), -INT8_C(  35),
         INT8_C(  22), -INT8_C(  44),  INT8_C(  47), -INT8_C(  56),  INT8_C( 111),  INT8_C(  64),  INT8_C( 105), -INT8_C( 106) },
      {  INT8_C(  68),  INT8_C(  74),  INT8_C( 122), -INT8_C(  38),  INT8_C(  71),  INT8_C(   6),  INT8_C(  12),  INT8_C(  87),
         INT8_C(  46),  INT8_C(  20), -INT8_C(  33), -INT8_C(  71), -INT8_C(  91), -INT8_C( 121), -INT8_C(  67),  INT8_C(  13),
         INT8_C(  93),  INT8_C(  81), -INT8_C(  18), -INT8_C(  30), -INT8_C(  40), -INT8_C(  68), -INT8_C(  65), -INT8_C(  18),
        -INT8_C( 112), -INT8_C(  17), -INT8_C(  73), -INT8_C(   1),  INT8_C(  47),  INT8_C(  32), -INT8_C( 107),  INT8_C( 115) },
      UINT32_C(2327763447) },
    { {  INT8_C( 107),  INT8_C(  15),  INT8_C(  77), -INT8_C(  78),  INT8_C(  21),  INT8_C(  90),  INT8_C(   9),  INT8_C(  67),
         INT8_C( 110), -INT8_C(  23), -INT8_C(   4),  INT8_C(  19),  INT8_C( 112), -INT8_C(  71),  INT8_C(  33), -INT8_C(  50),
         INT8_C(  10),  INT8_C(  15), -INT8_C(  80), -INT8_C(  29), -INT8_C(  53),  INT8_C( 111), -INT8_C(  47),  INT8_C(  91),
         INT8_C(  94), -INT8_C( 120),  INT8_C(  90), -INT8_C( 115), -INT8_C(  87), -INT8_C(  17),  INT8_C(   1),  INT8_C(  20) },
      { -INT8_C(   2),  INT8_C(  78), -INT8_C(  58),  INT8_C(  19), -INT8_C(  88), -INT8_C(  48),  INT8_C(  86),  INT8_C(  22),
        -INT8_C(  71),  INT8_C(  82),  INT8_C(  42),  INT8_C(  41),  INT8_C(  12),  INT8_C(  75), -INT8_C(   9),  INT8_C(  22),
         INT8_C(  90), -INT8_C(  89), -INT8_C(   7),  INT8_C(  37),  INT8_C(  23), -INT8_C(  53),      INT8_MIN,  INT8_C( 117),
         INT8_C(  83), -INT8_C(  38),  INT8_C(   3), -INT8_C(   4), -INT8_C(  55),  INT8_C(   4),  INT8_C(  16), -INT8_C(  57) },
      UINT32_C(2057154122) },
    { {  INT8_C(  82), -INT8_C(  41), -INT8_C(  38), -INT8_C(   5), -INT8_C(  89),  INT8_C(  48),  INT8_C(  17),  INT8_C(  96),
        -INT8_C( 126),  INT8_C(  59), -INT8_C( 119), -INT8_C( 114), -INT8_C( 122), -INT8_C( 127), -INT8_C(  91), -INT8_C(  32),
         INT8_C(  40), -INT8_C(  98),  INT8_C(   5),  INT8_C(  63),  INT8_C( 105), -INT8_C( 123), -INT8_C(  75), -INT8_C(  67),
         INT8_C(  95), -INT8_C(  72), -INT8_C(  71),  INT8_C(  40), -INT8_C(  68), -INT8_C(  54), -INT8_C(  17),  INT8_C(  14) },
      { -INT8_C(  95), -INT8_C(  55),  INT8_C(   9),  INT8_C(  72), -INT8_C(   7),  INT8_C(  27), -INT8_C(  88),  INT8_C( 124),
         INT8_C(  86),  INT8_C(  49),  INT8_C(  10), -INT8_C(  35), -INT8_C(  78), -INT8_C(  81), -INT8_C(  67), -INT8_C(  37),
         INT8_C(  78), -INT8_C(  61),  INT8_C(  26), -INT8_C(  73),  INT8_C(  72), -INT8_C(  49),  INT8_C( 116), -INT8_C(  88),
        -INT8_C( 121),  INT8_C(  46), -INT8_C(  48),  INT8_C(  67), -INT8_C(   8), -INT8_C(  64),  INT8_C(  82), -INT8_C( 103) },
      UINT32_C(1583840668) },
    { { -INT8_C( 119),  INT8_C(  91), -INT8_C(  31), -INT8_C( 125),  INT8_C( 118), -INT8_C( 119), -INT8_C(   1), -INT8_C(  51),
        -INT8_C(  70),  INT8_C(   9), -INT8_C(  86),  INT8_C( 109), -INT8_C(  71),  INT8_C( 103),  INT8_C(  72),  INT8_C(   7),
         INT8_C(  42),  INT8_C(  98), -INT8_C(  66),  INT8_C( 115),  INT8_C(  50),  INT8_C(  51),  INT8_C(  27), -INT8_C(  71),
         INT8_C(  97), -INT8_C(  21), -INT8_C(   3),  INT8_C(  89), -INT8_C(  85),  INT8_C(  79), -INT8_C(  14),  INT8_C(  53) },
      { -INT8_C(  86), -INT8_C(  45), -INT8_C(  72),  INT8_C(  33),  INT8_C(  92), -INT8_C(  73), -INT8_C(  18),  INT8_C(  22),
        -INT8_C(  64), -INT8_C( 104), -INT8_C( 125),  INT8_C( 121), -INT8_C(   1), -INT8_C(  53),      INT8_MIN,  INT8_C(  42),
         INT8_C(  46),  INT8_C(  63), -INT8_C(  99),  INT8_C(  96),  INT8_C( 114), -INT8_C(  72),  INT8_C(  25), -INT8_C(  45),
        -INT8_C(  93),  INT8_C(  22),  INT8_C(  44),  INT8_C(  79),  INT8_C( 101),  INT8_C(  30), -INT8_C( 124),  INT8_C(  16) },
      UINT32_C( 378640809) },
    { { -INT8_C(  15),  INT8_C(  60),  INT8_C(  49),  INT8_C(  77), -INT8_C(  13),  INT8_C(  31),  INT8_C(  99), -INT8_C(  77),
        -INT8_C(  73), -INT8_C(  25),  INT8_C(  45), -INT8_C(  74), -INT8_C(  78), -INT8_C(  83), -INT8_C(  32), -INT8_C(  32),
        -INT8_C(  20),  INT8_C( 125),  INT8_C(  64),  INT8_C(  94),  INT8_C(  53),  INT8_C(  90),  INT8_C(  49), -INT8_C(  39),
         INT8_C( 112),  INT8_C(  93),  INT8_C(  40), -INT8_C(  42),  INT8_C( 123), -INT8_C(  84), -INT8_C(  26),  INT8_C( 108) },
      { -INT8_C(  24),  INT8_C(  23), -INT8_C(  71), -INT8_C(  37),  INT8_C(  54),  INT8_C(  29), -INT8_C( 114), -INT8_C(  19),
         INT8_C(   4), -INT8_C(  69), -INT8_C(  93), -INT8_C(  74),  INT8_C( 105), -INT8_C( 124), -INT8_C( 105),  INT8_C(  85),
         INT8_C(   1), -INT8_C(  41), -INT8_C(  76),  INT8_C(  55),  INT8_C(  49), -INT8_C(  27),  INT8_C(  16), -INT8_C(  94),
         INT8_C(  67),  INT8_C(  56),  INT8_C( 120), -INT8_C(  66), -INT8_C(  28),  INT8_C(  94),  INT8_C(  43), -INT8_C(  52) },
      UINT32_C(1677824400) },
    { {  INT8_C( 117), -INT8_C(  28), -INT8_C(  89), -INT8_C(  85),  INT8_C(   1),  INT8_C(  53), -INT8_C( 104),  INT8_C(   5),
        -INT8_C(  15),  INT8_C(  59), -INT8_C(  68),  INT8_C(  90), -INT8_C(  65),  INT8_C(  83), -INT8_C(  81), -INT8_C(  63),
         INT8_C(  42),  INT8_C(  99), -INT8_C(   8),  INT8_C(  92),  INT8_C(  73),  INT8_C(   8), -INT8_C(   2), -INT8_C( 116),
         INT8_C(  64),  INT8_C( 118),  INT8_C(  74),  INT8_C(  36), -INT8_C(  44),  INT8_C( 117), -INT8_C(  16),  INT8_C(  73) },
      {  INT8_C(  90), -INT8_C( 105), -INT8_C(  12),  INT8_C(  91), -INT8_C(  52), -INT8_C( 116),  INT8_C(  97), -INT8_C(  67),
        -INT8_C(  57),  INT8_C(  29),  INT8_C(  23), -INT8_C( 121),  INT8_C( 112), -INT8_C(  57),  INT8_C(  72), -INT8_C( 102),
         INT8_C(  42),  INT8_C(  64), -INT8_C(  10),  INT8_C( 115),  INT8_C(  72), -INT8_C(  12), -INT8_C(   1), -INT8_C( 120),
         INT8_C( 106),  INT8_C(  74), -INT8_C(  84),  INT8_C(  62), -INT8_C(  65), -INT8_C( 100), -INT8_C( 121),  INT8_C(  25) },
      UINT32_C( 155735116) },
    { {  INT8_C(  51),  INT8_C( 123),  INT8_C( 117), -INT8_C(   1),  INT8_C(   7), -INT8_C(  42), -INT8_C(  67), -INT8_C(  49),
        -INT8_C(  13), -INT8_C(  44),  INT8_C(  86),  INT8_C(  99), -INT8_C( 101), -INT8_C(  98), -INT8_C(   3), -INT8_C(  58),
        -INT8_C(  34), -INT8_C(  12),  INT8_C(  57),  INT8_C(  38), -INT8_C(  24),  INT8_C(  57), -INT8_C(  82),  INT8_C(  83),
        -INT8_C( 125),  INT8_C(  90), -INT8_C( 111),  INT8_C(  66), -INT8_C(  10),  INT8_C(  25),  INT8_C(  92),  INT8_C(  41) },
      { -INT8_C( 108), -INT8_C(  47),  INT8_C(  40), -INT8_C( 100), -INT8_C(  89), -INT8_C(  27),  INT8_C( 107), -INT8_C( 102),
        -INT8_C(  70), -INT8_C(  63), -INT8_C(   3),  INT8_C(  85),  INT8_C(  95), -INT8_C(   6),  INT8_C(  27),  INT8_C(  61),
        -INT8_C(  18),  INT8_C(  85),  INT8_C(  99), -INT8_C(  41), -INT8_C( 114),  INT8_C(  17),  INT8_C(  42),  INT8_C(  17),
         INT8_C( 107), -INT8_C(  69),  INT8_C(  83),  INT8_C(  97), -INT8_C(  44), -INT8_C(  81), -INT8_C( 118),  INT8_C( 105) },
      UINT32_C(2370302048) },
    { {      INT8_MIN, -INT8_C(  78),  INT8_C(   5),  INT8_C(  39), -INT8_C( 104),  INT8_C( 112), -INT8_C(  63),  INT8_C(  82),
         INT8_C(  49), -INT8_C(  66), -INT8_C(  89), -INT8_C( 112), -INT8_C(  71), -INT8_C(  61), -INT8_C(  51), -INT8_C(  89),
         INT8_C(  24),  INT8_C(  48),  INT8_C( 126), -INT8_C(  90),  INT8_C(  65), -INT8_C(  88), -INT8_C(  73), -INT8_C(  84),
         INT8_C( 100),  INT8_C(  10),  INT8_C(  13),  INT8_C(  56), -INT8_C(  70), -INT8_C( 105), -INT8_C(  95),  INT8_C(  58) },
      {  INT8_C(  73), -INT8_C(  90),  INT8_C(  98), -INT8_C(  31),  INT8_C(  22),  INT8_C(  35),  INT8_C(  51),  INT8_C(  71),
        -INT8_C(  30), -INT8_C(  37), -INT8_C(  41), -INT8_C( 101), -INT8_C(  98), -INT8_C(  92),  INT8_C(  66), -INT8_C(  74),
        -INT8_C(  44), -INT8_C(  63),  INT8_C(  92),  INT8_C(  21),  INT8_C( 105),  INT8_C(  19), -INT8_C(  63), -INT8_C(  51),
         INT8_C(  29), -INT8_C(  50),  INT8_C(   6), -INT8_C(  41),  INT8_C( 101), -INT8_C(  89),  INT8_C(  18), -INT8_C(  81) },
      UINT32_C(1895353941) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmplt_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 r = easysimd_mm256_cmplt_epi8_mask(a, b);

    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[16];
    int16_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { {  INT16_C( 20218),  INT16_C(  9740), -INT16_C( 23316), -INT16_C( 30389), -INT16_C( 24014),  INT16_C( 20412),  INT16_C( 24968),  INT16_C( 18258),
         INT16_C(  1591),  INT16_C( 28274), -INT16_C( 17003), -INT16_C( 29300), -INT16_C( 24400),  INT16_C( 20878),  INT16_C(  6153),  INT16_C(   906) },
      { -INT16_C( 27034),  INT16_C( 21033),  INT16_C( 29754),  INT16_C( 27867), -INT16_C( 26602), -INT16_C( 24900),  INT16_C(  3833),  INT16_C( 12773),
         INT16_C( 22293), -INT16_C( 21857),  INT16_C( 11029), -INT16_C( 15049), -INT16_C( 14644), -INT16_C( 10986), -INT16_C( 24098),  INT16_C( 17624) },
      UINT16_C(40206) },
    { {  INT16_C(   311),  INT16_C( 29334),  INT16_C( 29301), -INT16_C( 29730), -INT16_C( 26102),  INT16_C(   809),  INT16_C(  3753), -INT16_C( 16844),
        -INT16_C( 11419),  INT16_C( 31336), -INT16_C( 24577), -INT16_C( 13505),  INT16_C( 22117),  INT16_C( 17312),  INT16_C( 30967),  INT16_C( 11911) },
      {  INT16_C(  7801), -INT16_C(  4448),  INT16_C( 32656), -INT16_C( 25991), -INT16_C( 24039), -INT16_C( 15715), -INT16_C( 11600),  INT16_C(  5504),
        -INT16_C(  5979), -INT16_C( 23408), -INT16_C( 12408), -INT16_C(  4753),  INT16_C(  3877),  INT16_C(  7217), -INT16_C( 18297),  INT16_C(    75) },
      UINT16_C( 3485) },
    { { -INT16_C(  5162),  INT16_C( 26350),  INT16_C( 26474), -INT16_C( 31744), -INT16_C( 25079), -INT16_C( 18106), -INT16_C( 14480),  INT16_C(  5583),
         INT16_C( 24495),  INT16_C( 14266),  INT16_C( 10542),  INT16_C( 21541),  INT16_C( 22073), -INT16_C( 16272), -INT16_C( 17650), -INT16_C(  6719) },
      { -INT16_C( 20569),  INT16_C(  4427),  INT16_C( 19479),  INT16_C(  8341), -INT16_C(  8982),  INT16_C( 23258), -INT16_C( 22109),  INT16_C( 21103),
         INT16_C( 10504),  INT16_C( 13962), -INT16_C( 20653), -INT16_C( 29558), -INT16_C(  1275),  INT16_C(  4940),  INT16_C(  3510),  INT16_C( 24056) },
      UINT16_C(57528) },
    { {  INT16_C( 17597), -INT16_C( 11153),  INT16_C(  1168),  INT16_C( 31476), -INT16_C( 12576), -INT16_C( 31788),  INT16_C( 17271),  INT16_C( 32726),
         INT16_C( 24685), -INT16_C( 16202),  INT16_C( 16399),  INT16_C(  5196), -INT16_C( 26565), -INT16_C(  3545),  INT16_C(  8358),  INT16_C( 25423) },
      { -INT16_C( 16796), -INT16_C(  3017),  INT16_C( 11203), -INT16_C( 23698),  INT16_C( 17146),  INT16_C( 28967), -INT16_C(   635), -INT16_C(  3343),
        -INT16_C( 22691),  INT16_C( 27826), -INT16_C(   281),  INT16_C(  9088), -INT16_C( 22633),  INT16_C( 15637),  INT16_C( 25799),  INT16_C( 11168) },
      UINT16_C(31286) },
    { { -INT16_C( 10461), -INT16_C(  6625), -INT16_C( 29438), -INT16_C(   887), -INT16_C( 20273),  INT16_C( 21870),  INT16_C( 24493),  INT16_C(  2631),
        -INT16_C(  1530), -INT16_C(  4746), -INT16_C(  2312), -INT16_C( 28912),  INT16_C(  9630),  INT16_C( 26060),  INT16_C( 27786), -INT16_C( 21103) },
      { -INT16_C( 20413),  INT16_C( 18067),  INT16_C(  7230),  INT16_C(  3394), -INT16_C( 20275),  INT16_C( 31330), -INT16_C( 22001),  INT16_C(  5509),
        -INT16_C(  1116), -INT16_C( 25597),  INT16_C(  5106), -INT16_C( 28628), -INT16_C(  1991), -INT16_C( 15371), -INT16_C( 31131), -INT16_C( 22416) },
      UINT16_C( 3502) },
    { {  INT16_C(   823),  INT16_C( 30190),  INT16_C( 12575), -INT16_C(  4990), -INT16_C(  6687), -INT16_C(  3737), -INT16_C(  4977),  INT16_C( 13062),
         INT16_C(  2535), -INT16_C(  9777), -INT16_C(  1251),  INT16_C( 22121),  INT16_C( 24564),  INT16_C( 22809), -INT16_C( 30235),  INT16_C(  7169) },
      { -INT16_C(  3956), -INT16_C( 21615),  INT16_C(  5153),  INT16_C(   664), -INT16_C(     7), -INT16_C( 30477), -INT16_C(  1301), -INT16_C( 11589),
        -INT16_C( 30205),  INT16_C(  8364),  INT16_C(  5510),  INT16_C( 31350), -INT16_C( 28812),  INT16_C( 23251), -INT16_C( 11240), -INT16_C( 23434) },
      UINT16_C(28248) },
    { {  INT16_C(  2244), -INT16_C(  6832), -INT16_C(  6116),  INT16_C(  5608), -INT16_C(  9241), -INT16_C( 11619),  INT16_C( 22741), -INT16_C(  9820),
         INT16_C( 20706),  INT16_C( 26873),  INT16_C( 28774), -INT16_C(  9502), -INT16_C( 18945),  INT16_C(  6196), -INT16_C( 21622),  INT16_C( 20156) },
      {  INT16_C(  3251), -INT16_C( 12492),  INT16_C(  7412), -INT16_C(  9244), -INT16_C( 32265), -INT16_C( 12883),  INT16_C( 21209), -INT16_C( 17498),
        -INT16_C( 24670),  INT16_C(  2084),  INT16_C(  1551),  INT16_C(  4067),  INT16_C(  6076),  INT16_C( 17959), -INT16_C(  7230),  INT16_C( 30100) },
      UINT16_C(63493) },
    { { -INT16_C( 14096), -INT16_C(  7100),  INT16_C( 10468), -INT16_C(  9024),  INT16_C( 28073), -INT16_C( 32087),  INT16_C( 20415),  INT16_C( 25150),
         INT16_C( 25326), -INT16_C(   406),  INT16_C( 19816),  INT16_C(  9229),  INT16_C( 13413),  INT16_C( 10090), -INT16_C(   233),  INT16_C(  1949) },
      { -INT16_C(  7737), -INT16_C( 21268), -INT16_C( 21494), -INT16_C( 19576),  INT16_C( 12569), -INT16_C(  9930),  INT16_C( 29824),  INT16_C( 28219),
        -INT16_C( 23082),  INT16_C( 15980),  INT16_C( 31219),  INT16_C( 22627), -INT16_C( 12883), -INT16_C( 14977),  INT16_C(  7372), -INT16_C( 27444) },
      UINT16_C(20193) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epi16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmplt_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 r = easysimd_mm256_cmplt_epi16_mask(a, b);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t a[16];
    const int32_t b[16];
    const uint8_t r;
  } test_vec[8] = {
    { { -INT32_C(  1493608736),  INT32_C(   132599565),  INT32_C(  1392869592),  INT32_C(   815428635),  INT32_C(  1471323802),  INT32_C(   801353096),  INT32_C(  2007367676), -INT32_C(  1556683581) },
      {  INT32_C(  1212821563),  INT32_C(  1464873343), -INT32_C(   861252175),  INT32_C(  1694254537),  INT32_C(  1807462371),  INT32_C(  1486519900),  INT32_C(  1104101246),  INT32_C(  1592002083) },
      UINT8_C(187) },
    { { -INT32_C(  1247334858),  INT32_C(   269285215),  INT32_C(   366851916), -INT32_C(   545662468), -INT32_C(   448121464),  INT32_C(   826139827),  INT32_C(  1181879587),  INT32_C(  1235572499) },
      { -INT32_C(   436319099), -INT32_C(  1879766461), -INT32_C(  1113271615),  INT32_C(   882646700),  INT32_C(   102360659), -INT32_C(   315074614),  INT32_C(  1999940196), -INT32_C(  2017404671) },
      UINT8_C( 89) },
    { {  INT32_C(  1751957029), -INT32_C(  1963499064), -INT32_C(   532178124),  INT32_C(   202695865), -INT32_C(  1793904950), -INT32_C(   377336955), -INT32_C(   144656651), -INT32_C(  1266802545) },
      { -INT32_C(  1508054306),  INT32_C(  2133857099),  INT32_C(  1751087279),  INT32_C(   645231708),  INT32_C(   666601634), -INT32_C(   938459437), -INT32_C(  2067828492),  INT32_C(  1849179536) },
      UINT8_C(158) },
    { {  INT32_C(  1930712359),  INT32_C(   401753448),  INT32_C(   427840189),  INT32_C(  1749022150),  INT32_C(  1351613309),  INT32_C(   756588345), -INT32_C(  1615734769),  INT32_C(  1024322069) },
      { -INT32_C(  1481629377),  INT32_C(   599761510), -INT32_C(  1170456588), -INT32_C(  1323140044), -INT32_C(  1342066313),  INT32_C(  1608325456),  INT32_C(   134123505), -INT32_C(  1203500167) },
      UINT8_C( 98) },
    { { -INT32_C(  1822428115), -INT32_C(  1950933354), -INT32_C(  1857686691), -INT32_C(   431855505),  INT32_C(  1788232473),  INT32_C(  1321825373),  INT32_C(  2085996547),  INT32_C(     3447251) },
      {  INT32_C(   613716877),  INT32_C(   263146418), -INT32_C(  1365117890),  INT32_C(  1989469020), -INT32_C(  2065683673), -INT32_C(  1563252321),  INT32_C(  1159604337),  INT32_C(  1329943233) },
      UINT8_C(143) },
    { { -INT32_C(  1754015259),  INT32_C(  1671897636),  INT32_C(  1930512406),  INT32_C(  1391043883),  INT32_C(  1876347344), -INT32_C(   468604558), -INT32_C(  1842794543),  INT32_C(  1742827137) },
      {  INT32_C(  1828607048), -INT32_C(  1915771530),  INT32_C(   419487981),  INT32_C(  1449912710),  INT32_C(   616972978), -INT32_C(  1140271125), -INT32_C(  2008141561), -INT32_C(   386977632) },
      UINT8_C(  9) },
    { { -INT32_C(    78319996), -INT32_C(  2121784173), -INT32_C(  1969584124),  INT32_C(   601949553),  INT32_C(   843556679), -INT32_C(  2047979394),  INT32_C(   554581377), -INT32_C(   251003539) },
      {  INT32_C(  2146197227), -INT32_C(  2046790526),  INT32_C(  1829870332), -INT32_C(   409931105),  INT32_C(   387569817), -INT32_C(  1449326552), -INT32_C(  1295275195), -INT32_C(  1801202520) },
      UINT8_C( 39) },
    { { -INT32_C(  1273786318),  INT32_C(    20648708),  INT32_C(  1282297005), -INT32_C(   684458178), -INT32_C(     1159721), -INT32_C(  1700230315), -INT32_C(   548637898),  INT32_C(  2054418248) },
      { -INT32_C(  2077260161),  INT32_C(  1183148697), -INT32_C(   191695946), -INT32_C(   909392142),  INT32_C(  1757985043),  INT32_C(  2063757637),  INT32_C(   744115940), -INT32_C(  1113076419) },
      UINT8_C(114) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm256_cmplt_epi32_mask");

    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 r = easysimd_mm256_cmplt_epi32_mask(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[4];
    int64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { {  INT64_C(  364719979207405168), -INT64_C( 2226929729246568493),  INT64_C( 1715881249878495671),  INT64_C( 4488338192264520098) },
      {  INT64_C(   62717435391864566),  INT64_C( 8234793136816414512), -INT64_C( 5427935883283592531),  INT64_C( 2884339175676603131) },
      UINT8_C(  2) },
    { { -INT64_C( 3830627481706968918),  INT64_C( 2696380159076899647), -INT64_C( 3942788436318308991),  INT64_C( 2318115165357357777) },
      {  INT64_C( 5459995578907110498), -INT64_C( 1295716806832032408), -INT64_C(  649139410616271259), -INT64_C( 8387070120455204859) },
      UINT8_C(  5) },
    { { -INT64_C( 4059911091599732144),  INT64_C(  260553051497491778), -INT64_C( 1115707521861476247), -INT64_C(  974457977301074491) },
      { -INT64_C( 8088474491483651687),  INT64_C( 8828986087280895717), -INT64_C( 5127473572045275396), -INT64_C( 1973104077350485237) },
      UINT8_C(  2) },
    { { -INT64_C( 3819442924725342943),  INT64_C( 1553411026140871938),  INT64_C( 3403827572188682129),  INT64_C( 2685968382735213335) },
      { -INT64_C( 4460935551254256792), -INT64_C( 3891844311685897009), -INT64_C( 1995223496303707653),  INT64_C(  978843136204090221) },
      UINT8_C(  0) },
    { { -INT64_C( 8416539699449288272),  INT64_C( 7482292539818772693),  INT64_C( 1881614783800091499), -INT64_C( 7010942510540561919) },
      {  INT64_C( 6165781382960123466), -INT64_C( 2153502960001088183),  INT64_C( 8873277063593370716),  INT64_C( 1812090611883487128) },
      UINT8_C( 13) },
    { { -INT64_C( 4703251747380972572), -INT64_C( 5688168009324815178), -INT64_C(  635888944797832906),  INT64_C(  609854176137989773) },
      { -INT64_C(  213596320161753386),  INT64_C( 4920527902235917635),  INT64_C( 8123123610726761572), -INT64_C( 5576789098916660881) },
      UINT8_C(  7) },
    { {  INT64_C( 8295074363149715673), -INT64_C( 1741110182352828029),  INT64_C( 6605486443268393817), -INT64_C( 7697343645968598881) },
      { -INT64_C( 5389168068893725806), -INT64_C( 1258723567373696995),  INT64_C( 4049252396665811945),  INT64_C( 4368457180206276430) },
      UINT8_C( 10) },
    { {  INT64_C( 5112204779245688075),  INT64_C( 4609275158961011059),  INT64_C( 5004421638600860066),  INT64_C( 1318409025730061925) },
      { -INT64_C( 1174241600171194044), -INT64_C( 6048070370312283270), -INT64_C( 3436991524856453871), -INT64_C( 3487991132646826880) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epi64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmplt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 r = easysimd_mm256_cmplt_epi64_mask(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    int8_t a[32];
    int8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C(1701620487),
      { -INT8_C(  88), -INT8_C( 124),  INT8_C( 116),  INT8_C( 106), -INT8_C(  23), -INT8_C(  73),  INT8_C( 111), -INT8_C( 106),
        -INT8_C(  22),  INT8_C( 111),  INT8_C(  57), -INT8_C(  28),  INT8_C(  61), -INT8_C(  52), -INT8_C( 110),  INT8_C( 120),
        -INT8_C(  47),  INT8_C( 124), -INT8_C( 119), -INT8_C(  97), -INT8_C(  44), -INT8_C( 126), -INT8_C(  21),  INT8_C( 108),
         INT8_C( 115), -INT8_C(  69), -INT8_C( 110),  INT8_C( 123),  INT8_C( 102), -INT8_C(   2), -INT8_C(  32),  INT8_C(  14) },
      { -INT8_C( 126),  INT8_C(  84),  INT8_C( 120),  INT8_C( 107),  INT8_C(  11), -INT8_C(  25),  INT8_C(   1), -INT8_C(  10),
         INT8_C(  86),  INT8_C(  59), -INT8_C(  38), -INT8_C( 108),  INT8_C(   7),  INT8_C( 108),  INT8_C(  12), -INT8_C(  40),
        -INT8_C(  24), -INT8_C( 106),  INT8_C( 119), -INT8_C(  68),  INT8_C(  24),  INT8_C(  98),  INT8_C(  40), -INT8_C( 117),
         INT8_C(  29), -INT8_C(  70),  INT8_C(   6), -INT8_C( 125), -INT8_C(  72), -INT8_C(  26), -INT8_C( 110),  INT8_C(  58) },
      UINT32_C(  74195206) },
    { UINT32_C(1185286715),
      { -INT8_C(  14), -INT8_C(  89),  INT8_C(  60),  INT8_C(  72), -INT8_C(  30),  INT8_C(  22), -INT8_C(  36), -INT8_C(  23),
        -INT8_C( 125), -INT8_C(  23), -INT8_C(  63),  INT8_C( 107),      INT8_MAX,  INT8_C(  56),  INT8_C(  40), -INT8_C( 105),
        -INT8_C( 102),  INT8_C(  80),  INT8_C(  34), -INT8_C(  73),  INT8_C(  11),  INT8_C(  41),  INT8_C(  59), -INT8_C(  61),
         INT8_C(  15), -INT8_C(  51), -INT8_C(   2),  INT8_C(  74), -INT8_C(  41), -INT8_C(  92), -INT8_C( 111), -INT8_C(  55) },
      {  INT8_C(  75), -INT8_C(  51),  INT8_C(  18),  INT8_C(  46), -INT8_C(  28), -INT8_C(  18),  INT8_C(  23),  INT8_C( 103),
        -INT8_C(  41), -INT8_C(  39), -INT8_C(  46),  INT8_C(  86),  INT8_C(  17), -INT8_C(   6), -INT8_C(  19), -INT8_C(  84),
         INT8_C(  75),  INT8_C(  16),  INT8_C(  99),  INT8_C(  86),  INT8_C(  57), -INT8_C(  98),  INT8_C(  25),  INT8_C(  72),
         INT8_C( 107),  INT8_C(  23), -INT8_C( 109),  INT8_C(  67), -INT8_C(  69),  INT8_C(  36),  INT8_C(  12),  INT8_C(   7) },
      UINT32_C(1115947027) },
    { UINT32_C(3577028337),
      {  INT8_C(  13),  INT8_C(  76),  INT8_C(  60), -INT8_C(  28),  INT8_C(  37),  INT8_C(  15),  INT8_C(  59),  INT8_C(  55),
         INT8_C(   9),  INT8_C(  40), -INT8_C(  29),  INT8_C(  84),  INT8_C(  56),  INT8_C(  70), -INT8_C(  86),  INT8_C( 113),
        -INT8_C(  27), -INT8_C(  60), -INT8_C(  70),  INT8_C(  80), -INT8_C(  37),  INT8_C(  77), -INT8_C( 109), -INT8_C( 105),
         INT8_C( 113), -INT8_C(  96), -INT8_C(  98),  INT8_C(  98), -INT8_C(  66), -INT8_C(  45),  INT8_C(  56), -INT8_C(  53) },
      {  INT8_C(  31),  INT8_C( 116), -INT8_C(  80),  INT8_C(  69), -INT8_C( 125), -INT8_C(  21),  INT8_C( 124), -INT8_C( 115),
         INT8_C(  19),  INT8_C(  95), -INT8_C(  31),  INT8_C(  76), -INT8_C(  91), -INT8_C( 116), -INT8_C(  67), -INT8_C( 118),
         INT8_C(  80),  INT8_C( 119), -INT8_C(  37),  INT8_C(  43), -INT8_C(  60),  INT8_C( 110), -INT8_C(  62),  INT8_C(  53),
         INT8_C(  14),  INT8_C(  96), -INT8_C( 104), -INT8_C(  51),  INT8_C(  51), -INT8_C(  48), -INT8_C( 104),  INT8_C(  83) },
      UINT32_C(2418344513) },
    { UINT32_C(3365423172),
      {  INT8_C(  51),  INT8_C(  20),  INT8_C(  85),  INT8_C(  71),  INT8_C( 115),  INT8_C(  54), -INT8_C( 109),  INT8_C(  24),
        -INT8_C(  62),  INT8_C(  80), -INT8_C(  93),  INT8_C(  18), -INT8_C(  56),  INT8_C( 126),  INT8_C(  62), -INT8_C( 116),
        -INT8_C(  20),  INT8_C(   0), -INT8_C(  62), -INT8_C(   5),  INT8_C(  97),  INT8_C(  90), -INT8_C(  56), -INT8_C( 108),
         INT8_C(  42),  INT8_C(  96), -INT8_C(  25),  INT8_C( 110), -INT8_C(  87),      INT8_MAX,  INT8_C(  54), -INT8_C(  36) },
      { -INT8_C( 109), -INT8_C( 117),  INT8_C(  35),  INT8_C(   6), -INT8_C(  62), -INT8_C(  74),  INT8_C(  31), -INT8_C( 124),
         INT8_C(   7), -INT8_C(  62), -INT8_C( 105), -INT8_C(  49),  INT8_C(  64), -INT8_C(  43),  INT8_C(  91),  INT8_C(  44),
        -INT8_C(  43),  INT8_C(  29),  INT8_C(  39),  INT8_C(  54),  INT8_C( 119), -INT8_C(  17), -INT8_C(  53), -INT8_C(  95),
         INT8_C(  80), -INT8_C(  78),  INT8_C(  16), -INT8_C(   7),  INT8_C(  50),  INT8_C(  70), -INT8_C(  43), -INT8_C(  59) },
      UINT32_C(   9977920) },
    { UINT32_C(2496461266),
      { -INT8_C(  81), -INT8_C(  21),  INT8_C(  24), -INT8_C(  74), -INT8_C(  83), -INT8_C(  81), -INT8_C( 123), -INT8_C(  19),
        -INT8_C( 124), -INT8_C(  31),  INT8_C(  25),  INT8_C(  90), -INT8_C(   2),  INT8_C(  65), -INT8_C( 112),  INT8_C( 118),
         INT8_C(  48),  INT8_C(  91),  INT8_C(  23),      INT8_MIN,  INT8_C(  14),  INT8_C(  39),  INT8_C( 121),  INT8_C(  64),
         INT8_C( 110),  INT8_C(  79),  INT8_C(   5),  INT8_C(  64),  INT8_C(  72), -INT8_C(  47), -INT8_C(  44), -INT8_C(   9) },
      { -INT8_C(  68), -INT8_C(  20), -INT8_C(  82),  INT8_C( 105), -INT8_C( 100),  INT8_C(  51),  INT8_C(  86),  INT8_C(  32),
         INT8_C(  20),  INT8_C( 112),  INT8_C( 122),  INT8_C(  19), -INT8_C(  79),  INT8_C(  11), -INT8_C( 119), -INT8_C(  31),
         INT8_C( 102), -INT8_C(  96),  INT8_C(  98),  INT8_C( 116), -INT8_C(  56), -INT8_C(  37), -INT8_C(  76),  INT8_C(  54),
         INT8_C(  42), -INT8_C(  70),  INT8_C( 118),  INT8_C( 114), -INT8_C( 117),  INT8_C(  74),  INT8_C( 106),  INT8_C(  72) },
      UINT32_C(2215379394) },
    { UINT32_C(3534821430),
      {  INT8_C(  75),  INT8_C(   8), -INT8_C(  13),  INT8_C(  96),  INT8_C( 120),  INT8_C( 109),  INT8_C( 115),  INT8_C(  41),
         INT8_C( 120), -INT8_C(   4),  INT8_C(  10), -INT8_C(  33), -INT8_C( 100),  INT8_C( 108),  INT8_C(  83),  INT8_C( 100),
         INT8_C(  72),  INT8_C(   8), -INT8_C( 102),  INT8_C( 114), -INT8_C(  62),  INT8_C(  16), -INT8_C(  27),  INT8_C(  77),
         INT8_C(  90),  INT8_C(  79), -INT8_C( 107), -INT8_C( 111),  INT8_C( 103),  INT8_C(  71),  INT8_C(  99), -INT8_C(  78) },
      {  INT8_C(  79),  INT8_C(  86),  INT8_C(  18), -INT8_C(  57), -INT8_C(  60), -INT8_C( 123), -INT8_C(  16),  INT8_C(  60),
        -INT8_C( 127), -INT8_C(   6),  INT8_C(  27),  INT8_C(  30),  INT8_C( 103),  INT8_C( 111), -INT8_C( 126), -INT8_C(  81),
         INT8_C( 119),  INT8_C(  29),  INT8_C(  33),  INT8_C(  57),  INT8_C(  45),  INT8_C(   6), -INT8_C( 122), -INT8_C( 120),
         INT8_C(  85),  INT8_C(  28),  INT8_C(  25), -INT8_C(  68),  INT8_C(  99),  INT8_C( 124),  INT8_C( 111), -INT8_C(  78) },
      UINT32_C(1074862086) },
    { UINT32_C(2541322707),
      {  INT8_C(   7),  INT8_C( 105), -INT8_C(  45), -INT8_C( 120),  INT8_C(  99), -INT8_C(  17), -INT8_C(  90), -INT8_C(  54),
         INT8_C(  94),  INT8_C(  41),  INT8_C( 121), -INT8_C(  43),  INT8_C(  70), -INT8_C( 101),  INT8_C(  14),  INT8_C( 115),
        -INT8_C(  95), -INT8_C( 108), -INT8_C(   5), -INT8_C(   9), -INT8_C(  80),  INT8_C(  20), -INT8_C(  77),  INT8_C(  19),
        -INT8_C( 111),  INT8_C(  34), -INT8_C(  59),  INT8_C( 100), -INT8_C(  92),  INT8_C(  62), -INT8_C(   5), -INT8_C(  85) },
      { -INT8_C(  89), -INT8_C(  50),  INT8_C(  51),  INT8_C(  11), -INT8_C(  67), -INT8_C(  38), -INT8_C(  43),  INT8_C(  27),
         INT8_C(   3),  INT8_C(  79), -INT8_C(  16),  INT8_C(  73), -INT8_C(  22), -INT8_C(   2), -INT8_C(  68), -INT8_C( 117),
        -INT8_C( 109), -INT8_C(  72), -INT8_C( 126),  INT8_C(  67), -INT8_C(  52),  INT8_C(  54),  INT8_C(  87),  INT8_C(  93),
         INT8_C(  88),  INT8_C(  28), -INT8_C(  63), -INT8_C(   4),  INT8_C(  91), -INT8_C(  68), -INT8_C(  89),  INT8_C(   2) },
      UINT32_C(2440560832) },
    { UINT32_C(1208867723),
      { -INT8_C(  75), -INT8_C(  29),  INT8_C( 100), -INT8_C(  72),  INT8_C(  50),  INT8_C(  84),  INT8_C(   1),  INT8_C(  28),
         INT8_C(  83), -INT8_C(  67), -INT8_C(  89), -INT8_C(  26),  INT8_C( 117),  INT8_C(  42),  INT8_C(  41),  INT8_C(  66),
         INT8_C(  96),      INT8_MIN, -INT8_C(  97), -INT8_C(  72), -INT8_C(  99),  INT8_C(  97), -INT8_C(  75), -INT8_C(   8),
         INT8_C(  29),  INT8_C(  92), -INT8_C(   6), -INT8_C(  88),  INT8_C(  55),  INT8_C(   8), -INT8_C(  15), -INT8_C(  20) },
      { -INT8_C(  21),  INT8_C(  85), -INT8_C(  92),  INT8_C(  29), -INT8_C(  87), -INT8_C(  91),  INT8_C(  57), -INT8_C(   4),
         INT8_C(  99), -INT8_C(  32), -INT8_C(  30), -INT8_C(  40),  INT8_C(  10),  INT8_C(  12),  INT8_C(  26),  INT8_C( 106),
        -INT8_C( 116), -INT8_C(  70),  INT8_C(  35),  INT8_C(  41),  INT8_C(  27), -INT8_C(  40),  INT8_C(  33),  INT8_C(  56),
         INT8_C(  52),  INT8_C(  28), -INT8_C(  31),  INT8_C( 108),  INT8_C(  36), -INT8_C(  46),  INT8_C(  88),  INT8_C(  15) },
      UINT32_C(1208779531) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epi8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 r = easysimd_mm256_mask_cmplt_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int16_t a[16];
    int16_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(57425),
      { -INT16_C( 23549),  INT16_C( 13888), -INT16_C( 11728),  INT16_C( 30799),  INT16_C( 26079),  INT16_C( 16510), -INT16_C( 17153),  INT16_C( 18063),
         INT16_C( 21420),  INT16_C( 14557),  INT16_C( 15742), -INT16_C( 18108), -INT16_C(  7222), -INT16_C( 22746),  INT16_C( 30686), -INT16_C(  7801) },
      { -INT16_C( 14564),  INT16_C( 19480),  INT16_C( 26521),  INT16_C( 30916),  INT16_C( 17356), -INT16_C( 13384),  INT16_C( 18687), -INT16_C( 21742),
        -INT16_C(  4197),  INT16_C(  6627),  INT16_C( 10028), -INT16_C(  2094), -INT16_C(  1782), -INT16_C(  5986),  INT16_C(  9584), -INT16_C( 29494) },
      UINT16_C( 8257) },
    { UINT16_C(58093),
      { -INT16_C( 31015), -INT16_C( 25271),  INT16_C(  5631), -INT16_C( 18464), -INT16_C(  7967), -INT16_C(  3073), -INT16_C( 25717),  INT16_C( 28642),
         INT16_C(  4020), -INT16_C( 30826), -INT16_C( 24314), -INT16_C( 23424), -INT16_C(  3959),  INT16_C( 21449), -INT16_C( 18819),  INT16_C( 22069) },
      {  INT16_C( 32317),  INT16_C( 15603), -INT16_C( 11116),  INT16_C( 30195), -INT16_C(  3148),  INT16_C( 16232),  INT16_C( 19086),  INT16_C( 17070),
         INT16_C( 17753),  INT16_C( 24521),  INT16_C( 18918),  INT16_C( 28419), -INT16_C( 12998), -INT16_C( 18493), -INT16_C(  1917), -INT16_C( 16371) },
      UINT16_C(17001) },
    { UINT16_C(  119),
      {  INT16_C(  3068), -INT16_C(  3884), -INT16_C( 30592), -INT16_C(  5917),  INT16_C( 29128),  INT16_C( 30258), -INT16_C( 29517),  INT16_C( 32187),
        -INT16_C( 24085), -INT16_C(  4154),  INT16_C(    17), -INT16_C( 11076),  INT16_C( 16311), -INT16_C( 15156),  INT16_C( 17152), -INT16_C(   827) },
      { -INT16_C( 26290), -INT16_C( 12564), -INT16_C( 12510), -INT16_C(  5450), -INT16_C(  5824), -INT16_C(  2976),  INT16_C(  7285),  INT16_C( 24689),
         INT16_C( 14269), -INT16_C( 12721),  INT16_C(  2872), -INT16_C(  4190),  INT16_C( 28491),  INT16_C( 19380),  INT16_C( 31154),  INT16_C(   327) },
      UINT16_C(   68) },
    { UINT16_C(13330),
      {  INT16_C( 13519), -INT16_C( 31229),  INT16_C( 17438),  INT16_C( 32623), -INT16_C(  7112), -INT16_C( 22117),  INT16_C( 22596), -INT16_C( 27424),
         INT16_C(  6183), -INT16_C( 13921), -INT16_C(  5624), -INT16_C( 17352), -INT16_C(  5323),  INT16_C( 32053),  INT16_C( 18412), -INT16_C( 17487) },
      { -INT16_C( 19332), -INT16_C( 26047), -INT16_C( 20232),  INT16_C( 12313), -INT16_C( 19308), -INT16_C(  9767), -INT16_C( 17907),  INT16_C( 13421),
         INT16_C(  3282), -INT16_C(  9475),  INT16_C( 14071),  INT16_C( 11414), -INT16_C( 13535),  INT16_C(  3497),  INT16_C( 23059), -INT16_C( 28728) },
      UINT16_C( 1026) },
    { UINT16_C( 2575),
      {  INT16_C(  1833),  INT16_C( 17338),  INT16_C( 20280),  INT16_C(  4599),  INT16_C(  1064), -INT16_C( 27189), -INT16_C( 25032),  INT16_C( 13985),
        -INT16_C( 26504),  INT16_C(  3948), -INT16_C( 29243),  INT16_C( 28378), -INT16_C(  4710),  INT16_C( 25289), -INT16_C( 10116), -INT16_C( 22932) },
      {  INT16_C( 10207),  INT16_C(  6121), -INT16_C(  8074), -INT16_C( 25047), -INT16_C(  2843),  INT16_C(  7475), -INT16_C( 11118),  INT16_C(  2899),
        -INT16_C( 16531),  INT16_C( 12826), -INT16_C(  2996), -INT16_C(  6496),  INT16_C( 27106),  INT16_C( 24137), -INT16_C( 19135),  INT16_C(  8452) },
      UINT16_C(  513) },
    { UINT16_C(60892),
      {  INT16_C( 21048),  INT16_C( 25038), -INT16_C( 19472),  INT16_C(  9046), -INT16_C(  5936),  INT16_C(  9464),  INT16_C( 26099),  INT16_C(  3555),
         INT16_C( 12439),  INT16_C( 14082), -INT16_C(  7146),  INT16_C( 24481), -INT16_C(  7614),  INT16_C( 18197), -INT16_C(  3837),  INT16_C( 15412) },
      {  INT16_C(   580),  INT16_C( 13469), -INT16_C(  3147), -INT16_C( 31144),  INT16_C( 20700), -INT16_C( 12374), -INT16_C( 29259),  INT16_C( 19677),
        -INT16_C(  8259), -INT16_C( 11133),  INT16_C(  9411),  INT16_C(  1331),  INT16_C( 18439),  INT16_C(  2636), -INT16_C( 32454),  INT16_C( 32326) },
      UINT16_C(33940) },
    { UINT16_C(58499),
      {  INT16_C( 14770),  INT16_C(  2775), -INT16_C( 19521),  INT16_C( 26970),  INT16_C(  3971),  INT16_C( 24822), -INT16_C( 19365), -INT16_C(  8385),
         INT16_C(   648), -INT16_C( 17661),  INT16_C(  2567),  INT16_C( 21508),  INT16_C( 15893),  INT16_C( 23509),  INT16_C( 22716),  INT16_C( 28223) },
      {  INT16_C(  6033),  INT16_C( 20601), -INT16_C( 11318),  INT16_C( 19897), -INT16_C( 20253),  INT16_C( 16045), -INT16_C(  5020), -INT16_C(  5091),
         INT16_C(  8686), -INT16_C(  2393), -INT16_C( 21717),  INT16_C( 16458),  INT16_C(  8169), -INT16_C( 23140), -INT16_C(  9353),  INT16_C(  2324) },
      UINT16_C(  130) },
    { UINT16_C(36338),
      { -INT16_C( 17063),  INT16_C(  4960),  INT16_C( 17162), -INT16_C( 18237),  INT16_C( 10114), -INT16_C( 24668), -INT16_C( 27885), -INT16_C( 17728),
        -INT16_C(  4983), -INT16_C( 11418),  INT16_C( 20268), -INT16_C( 14094),  INT16_C( 27125),  INT16_C(  2468), -INT16_C( 27022), -INT16_C( 13162) },
      { -INT16_C(  2477),  INT16_C( 24287), -INT16_C( 24006), -INT16_C( 17386), -INT16_C( 17719), -INT16_C(  9125),  INT16_C(  7245), -INT16_C( 10602),
        -INT16_C(  1016),  INT16_C( 13481), -INT16_C( 25780),  INT16_C( 16893), -INT16_C( 24315),  INT16_C( 30538), -INT16_C(  8137), -INT16_C( 29885) },
      UINT16_C( 2530) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epi16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 r = easysimd_mm256_mask_cmplt_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int32_t a[8];
    int32_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(150),
      { -INT32_C(   412671373), -INT32_C(  1636558239),  INT32_C(   419323604),  INT32_C(   525638266),  INT32_C(   124564693), -INT32_C(   834213525), -INT32_C(   813758708), -INT32_C(   580567446) },
      { -INT32_C(  1027289760), -INT32_C(  1151321881),  INT32_C(   299064982), -INT32_C(   835704840),  INT32_C(  1205181660), -INT32_C(  1860887675), -INT32_C(  1939762142),  INT32_C(   862570195) },
      UINT8_C(146) },
    { UINT8_C(147),
      {  INT32_C(  1702556973), -INT32_C(  1275316651),  INT32_C(   833359113), -INT32_C(   653428163),  INT32_C(  1801344079), -INT32_C(    24252566),  INT32_C(   399579728), -INT32_C(  1314257533) },
      {  INT32_C(  1343628794),  INT32_C(  1677922907),  INT32_C(  1553313567),  INT32_C(  2033558057),  INT32_C(  1642370039), -INT32_C(   748719741),  INT32_C(   267006347),  INT32_C(   817927478) },
      UINT8_C(130) },
    { UINT8_C(186),
      { -INT32_C(   384466730),  INT32_C(   856193412), -INT32_C(  1319279345),  INT32_C(   782816922),  INT32_C(   733022650), -INT32_C(  1699249047),  INT32_C(    80791151),  INT32_C(  1555955846) },
      {  INT32_C(    88462209),  INT32_C(  1530416716),  INT32_C(  1292735923),  INT32_C(   628864363),  INT32_C(   676408511),  INT32_C(   549587121),  INT32_C(  1411683022),  INT32_C(   330359442) },
      UINT8_C( 34) },
    { UINT8_C(181),
      {  INT32_C(  1140922614), -INT32_C(   419996335),  INT32_C(   525485162), -INT32_C(   320964672), -INT32_C(   795015480),  INT32_C(  1520352712), -INT32_C(  1007815967), -INT32_C(  1720188765) },
      {  INT32_C(  1792899609),  INT32_C(  1095816407), -INT32_C(   664755688), -INT32_C(   490389734),  INT32_C(   246637125),  INT32_C(    23613728), -INT32_C(   406497980),  INT32_C(  1870675286) },
      UINT8_C(145) },
    { UINT8_C(183),
      {  INT32_C(   831445341), -INT32_C(   867578071),  INT32_C(  1877353008),  INT32_C(  1236584679), -INT32_C(   848706949), -INT32_C(  2146342101),  INT32_C(  1842804784), -INT32_C(   718977672) },
      {  INT32_C(  1208398622), -INT32_C(  1290514557),  INT32_C(  1478687345),  INT32_C(  1050859202), -INT32_C(  1005909095), -INT32_C(  1505485706), -INT32_C(  1944839404),  INT32_C(  2137078112) },
      UINT8_C(161) },
    { UINT8_C(236),
      { -INT32_C(  1234188441), -INT32_C(   718789925),  INT32_C(   462913604),  INT32_C(   783602978), -INT32_C(    56330016), -INT32_C(   686797892), -INT32_C(  1741120417),  INT32_C(  1686419453) },
      {  INT32_C(  1495004286),  INT32_C(  1529758486), -INT32_C(   428423741),  INT32_C(  2064919195),  INT32_C(  1584969890),  INT32_C(  1647740931),  INT32_C(   586837541), -INT32_C(  1551401179) },
      UINT8_C(104) },
    { UINT8_C(115),
      { -INT32_C(   443941726), -INT32_C(   274144214), -INT32_C(  2071294374),  INT32_C(  1529218722),  INT32_C(   106857854),  INT32_C(   690733499),  INT32_C(   978210491),  INT32_C(  2007888341) },
      {  INT32_C(   391919597),  INT32_C(  1980105755),  INT32_C(   905613715),  INT32_C(   361767319),  INT32_C(  1629220774),  INT32_C(  1804224432),  INT32_C(  1789319317), -INT32_C(  1226746935) },
      UINT8_C(115) },
    { UINT8_C(138),
      {  INT32_C(  1101450557),  INT32_C(  1691622611),  INT32_C(   939198998), -INT32_C(  1982000998),  INT32_C(  1933131308), -INT32_C(  1610046008), -INT32_C(  1637256629), -INT32_C(  1876353197) },
      { -INT32_C(  1059991572),  INT32_C(    19179243),  INT32_C(  1245257904),  INT32_C(  1557403184),  INT32_C(   500108629), -INT32_C(    37824590), -INT32_C(  1667487927),  INT32_C(   858572103) },
      UINT8_C(136) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epi32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 r = easysimd_mm256_mask_cmplt_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t a[4];
    int64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 97),
      {  INT64_C( 8487713427195652732), -INT64_C( 2601777154081470691),  INT64_C(  696084700121062549),  INT64_C( 6134635902477137020) },
      { -INT64_C(  441766332699217660),  INT64_C( 6061686490743381710), -INT64_C( 1896647208584254389), -INT64_C( 3610539133708652875) },
      UINT8_C(  0) },
    { UINT8_C( 58),
      {  INT64_C( 3744196723746420385), -INT64_C( 4155056253658055781), -INT64_C( 2966159497016313249), -INT64_C( 1459328686919555652) },
      { -INT64_C( 9157214535087303005), -INT64_C( 8978522738976367292),  INT64_C( 7446836860906935771),  INT64_C(  510502579481671648) },
      UINT8_C(  8) },
    { UINT8_C(129),
      { -INT64_C( 8685500113178083544),  INT64_C( 1165919356948452722),  INT64_C(   84479916350616017), -INT64_C( 7837551516580489714) },
      { -INT64_C(  262270314622967253),  INT64_C(  421788280310138447), -INT64_C( 4235215748687882961),  INT64_C( 3878084207792004819) },
      UINT8_C(  1) },
    { UINT8_C( 91),
      {  INT64_C( 7863532781096471190),  INT64_C( 6475515202902288944), -INT64_C( 1276254482139703560), -INT64_C( 1195654741045508269) },
      {  INT64_C( 6849094524665285944),  INT64_C( 5469317541569342921),  INT64_C( 2251248732081354241), -INT64_C( 7072458574367279215) },
      UINT8_C(  0) },
    { UINT8_C(226),
      { -INT64_C(  773933828911645543), -INT64_C( 6023119515476701315),  INT64_C( 3454709482179311957),  INT64_C( 5636591063559101379) },
      {  INT64_C( 3939533367770761838), -INT64_C( 2361566767217707560),  INT64_C(  930045316739019457),  INT64_C( 1949669058280717323) },
      UINT8_C(  2) },
    { UINT8_C(231),
      {  INT64_C( 8337807225030562516),  INT64_C( 3476457294407680886),  INT64_C( 6581033249063696154), -INT64_C(  617234893213461270) },
      {  INT64_C( 6766528592337965302), -INT64_C( 2776159385928604468),  INT64_C( 7155225273588511505),  INT64_C( 8116992282565694860) },
      UINT8_C(  4) },
    { UINT8_C(105),
      { -INT64_C( 5706431486596356999), -INT64_C( 8884395275034093431),  INT64_C( 5003324298765378668),  INT64_C( 4399506403316670027) },
      { -INT64_C( 7525780999569973485), -INT64_C( 5459226652200730842), -INT64_C( 1889251683467839086), -INT64_C( 3880528692068645083) },
      UINT8_C(  0) },
    { UINT8_C(246),
      {  INT64_C( 4078461608605821285), -INT64_C( 4155357301999382315),  INT64_C( 2951904642086262987),  INT64_C( 7664186506543620302) },
      { -INT64_C( 3403746070606553006),  INT64_C( 4835152373671478215), -INT64_C( 8247483181258409535),  INT64_C( 7304081117763824365) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epi64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 r = easysimd_mm256_mask_cmplt_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[32];
    uint8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { { UINT8_C( 41), UINT8_C( 75), UINT8_C(156), UINT8_C(186), UINT8_C(100), UINT8_C(189), UINT8_C( 67), UINT8_C(118),
        UINT8_C(111), UINT8_C(229), UINT8_C(196), UINT8_C( 92), UINT8_C( 46), UINT8_C(110), UINT8_C( 26), UINT8_C(217),
        UINT8_C(163), UINT8_C(107), UINT8_C(226), UINT8_C(124), UINT8_C( 52), UINT8_C( 78), UINT8_C(169), UINT8_C(147),
        UINT8_C( 54), UINT8_C(143), UINT8_C( 44), UINT8_C(203), UINT8_C(106), UINT8_C(221), UINT8_C( 11), UINT8_C(147) },
      { UINT8_C( 41), UINT8_C(167), UINT8_C( 77), UINT8_C(141), UINT8_C(101), UINT8_C(145), UINT8_C(  3), UINT8_C(212),
        UINT8_C(118), UINT8_C(200), UINT8_C( 48), UINT8_C(165), UINT8_C( 54), UINT8_C( 74), UINT8_C(126), UINT8_C(217),
        UINT8_C(182), UINT8_C( 96), UINT8_C( 85), UINT8_C(234), UINT8_C(175),    UINT8_MAX, UINT8_C(125), UINT8_C(229),
        UINT8_C(142), UINT8_C(169), UINT8_C(177), UINT8_C(248), UINT8_C(134), UINT8_C(188), UINT8_C(140), UINT8_C(175) },
      UINT32_C(3753466258) },
    { { UINT8_C( 99), UINT8_C(217), UINT8_C( 61), UINT8_C(200), UINT8_C(106), UINT8_C( 64), UINT8_C(156), UINT8_C(225),
        UINT8_C(  8), UINT8_C(204), UINT8_C(134), UINT8_C( 62), UINT8_C( 23), UINT8_C(  4), UINT8_C( 23), UINT8_C(205),
        UINT8_C(101), UINT8_C(109), UINT8_C(183), UINT8_C( 20), UINT8_C(108), UINT8_C( 52), UINT8_C(249), UINT8_C(250),
        UINT8_C(221), UINT8_C(170), UINT8_C(242), UINT8_C( 99), UINT8_C(102), UINT8_C(126), UINT8_C( 19), UINT8_C(202) },
      { UINT8_C( 88), UINT8_C( 80), UINT8_C(146), UINT8_C(194), UINT8_C(144), UINT8_C( 47), UINT8_C(163), UINT8_C(153),
        UINT8_C(251), UINT8_C( 41), UINT8_C(215), UINT8_C( 18), UINT8_C( 46), UINT8_C(239), UINT8_C(223), UINT8_C(147),
        UINT8_C( 92), UINT8_C(150), UINT8_C(167), UINT8_C(200), UINT8_C(202), UINT8_C(160), UINT8_C(194), UINT8_C(167),
        UINT8_C( 75), UINT8_C(180), UINT8_C( 11), UINT8_C(177), UINT8_C( 51), UINT8_C( 30), UINT8_C(123), UINT8_C(139) },
      UINT32_C(1245345108) },
    { { UINT8_C(110), UINT8_C( 14), UINT8_C( 77), UINT8_C(254), UINT8_C( 61), UINT8_C(241), UINT8_C(151), UINT8_C( 56),
        UINT8_C( 26), UINT8_C(111), UINT8_C( 75), UINT8_C( 72), UINT8_C( 94), UINT8_C( 42), UINT8_C(219), UINT8_C(186),
        UINT8_C(193), UINT8_C(130), UINT8_C(130), UINT8_C(139), UINT8_C( 35), UINT8_C( 68), UINT8_C( 51), UINT8_C(110),
        UINT8_C(248), UINT8_C( 62), UINT8_C( 31), UINT8_C( 43), UINT8_C( 92), UINT8_C(155), UINT8_C(182), UINT8_C(202) },
      { UINT8_C(169), UINT8_C(  4), UINT8_C(200), UINT8_C(230), UINT8_C(245), UINT8_C( 96), UINT8_C( 30), UINT8_C( 15),
        UINT8_C(207), UINT8_C(105), UINT8_C( 88), UINT8_C( 45), UINT8_C(148), UINT8_C( 51), UINT8_C(231), UINT8_C( 85),
        UINT8_C(182), UINT8_C(105), UINT8_C(224), UINT8_C(217), UINT8_C(173), UINT8_C( 19), UINT8_C( 71), UINT8_C(165),
        UINT8_C( 81), UINT8_C(102), UINT8_C(209), UINT8_C(173), UINT8_C(  1), UINT8_C(135), UINT8_C(119), UINT8_C(170) },
      UINT32_C( 249328917) },
    { { UINT8_C(139), UINT8_C( 64), UINT8_C(144), UINT8_C(128), UINT8_C(160), UINT8_C(175), UINT8_C(144), UINT8_C(111),
        UINT8_C( 24), UINT8_C(232), UINT8_C(156), UINT8_C(172), UINT8_C( 27), UINT8_C(131), UINT8_C(  1), UINT8_C(209),
        UINT8_C(236), UINT8_C(226), UINT8_C(170), UINT8_C(153), UINT8_C(245), UINT8_C(241), UINT8_C( 62), UINT8_C( 71),
        UINT8_C( 88), UINT8_C( 15), UINT8_C(244), UINT8_C( 89), UINT8_C(151), UINT8_C(108), UINT8_C(  4), UINT8_C( 34) },
      { UINT8_C(172), UINT8_C(148), UINT8_C(163), UINT8_C( 76), UINT8_C( 67), UINT8_C( 51), UINT8_C(187), UINT8_C( 92),
        UINT8_C( 27), UINT8_C( 87), UINT8_C(  8), UINT8_C( 54), UINT8_C(218), UINT8_C( 10), UINT8_C(  8), UINT8_C(198),
        UINT8_C(236), UINT8_C(178), UINT8_C( 95), UINT8_C(225), UINT8_C(164), UINT8_C(157), UINT8_C( 40), UINT8_C(252),
        UINT8_C(173), UINT8_C( 29), UINT8_C( 85), UINT8_C( 68), UINT8_C(137), UINT8_C( 89), UINT8_C(102), UINT8_C( 53) },
      UINT32_C(3280490823) },
    { { UINT8_C(238), UINT8_C(  9), UINT8_C(129), UINT8_C( 49), UINT8_C( 60), UINT8_C( 60), UINT8_C(141), UINT8_C( 87),
        UINT8_C(147), UINT8_C(150), UINT8_C(142), UINT8_C(109), UINT8_C(160), UINT8_C(150), UINT8_C( 51), UINT8_C(140),
        UINT8_C( 72), UINT8_C(146), UINT8_C(109), UINT8_C(236), UINT8_C( 47), UINT8_C(150), UINT8_C(232), UINT8_C(220),
        UINT8_C(179), UINT8_C( 62), UINT8_C( 32), UINT8_C( 60), UINT8_C(151), UINT8_C(135), UINT8_C(113), UINT8_C(133) },
      { UINT8_C(144), UINT8_C(242), UINT8_C(183), UINT8_C(205), UINT8_C( 46), UINT8_C( 68), UINT8_C( 36), UINT8_C(193),
        UINT8_C(218), UINT8_C(178), UINT8_C( 46), UINT8_C(122), UINT8_C( 72), UINT8_C( 97), UINT8_C(  6), UINT8_C(145),
        UINT8_C(243), UINT8_C(116), UINT8_C(125), UINT8_C( 34), UINT8_C( 10), UINT8_C(102),    UINT8_MAX, UINT8_C(189),
        UINT8_C(164), UINT8_C( 31), UINT8_C(249), UINT8_C( 59), UINT8_C(166), UINT8_C(106), UINT8_C(193), UINT8_C( 55) },
      UINT32_C(1413843886) },
    { { UINT8_C( 92), UINT8_C(120), UINT8_C(  4), UINT8_C(138), UINT8_C(188), UINT8_C( 40), UINT8_C( 75), UINT8_C(151),
        UINT8_C(219), UINT8_C(121), UINT8_C( 17), UINT8_C( 35), UINT8_C(218), UINT8_C( 24), UINT8_C(180), UINT8_C(205),
        UINT8_C(140), UINT8_C( 50), UINT8_C(239), UINT8_C(150), UINT8_C(152), UINT8_C(238), UINT8_C( 83), UINT8_C( 60),
        UINT8_C( 14), UINT8_C( 76), UINT8_C(119), UINT8_C(180), UINT8_C(182), UINT8_C( 56), UINT8_C(235), UINT8_C( 18) },
      { UINT8_C(176), UINT8_C(239), UINT8_C(156), UINT8_C(109), UINT8_C( 24), UINT8_C(231), UINT8_C(  4), UINT8_C(243),
        UINT8_C( 96), UINT8_C( 21), UINT8_C( 22), UINT8_C( 58), UINT8_C( 45), UINT8_C(203), UINT8_C(  7), UINT8_C(185),
        UINT8_C(253), UINT8_C(246), UINT8_C( 79), UINT8_C(149), UINT8_C(229), UINT8_C(162), UINT8_C(209), UINT8_C(243),
        UINT8_C(238), UINT8_C( 72), UINT8_C(167), UINT8_C(164), UINT8_C(129), UINT8_C(147), UINT8_C(182), UINT8_C( 49) },
      UINT32_C(2782080167) },
    { { UINT8_C(130), UINT8_C( 82), UINT8_C(158), UINT8_C(154), UINT8_C( 57), UINT8_C(162), UINT8_C(141), UINT8_C(153),
        UINT8_C(184), UINT8_C(164), UINT8_C(211), UINT8_C(229), UINT8_C(111), UINT8_C(218), UINT8_C(159), UINT8_C(108),
        UINT8_C(209), UINT8_C(238), UINT8_C(  1), UINT8_C(182), UINT8_C(145), UINT8_C(210), UINT8_C(169), UINT8_C(127),
        UINT8_C( 26), UINT8_C( 80), UINT8_C( 36), UINT8_C(155), UINT8_C(227), UINT8_C(218), UINT8_C(205), UINT8_C(102) },
      { UINT8_C( 45), UINT8_C(107), UINT8_C(  0), UINT8_C(102), UINT8_C( 14), UINT8_C(142), UINT8_C(  0), UINT8_C(198),
        UINT8_C( 50), UINT8_C(211), UINT8_C(171), UINT8_C(161), UINT8_C(174), UINT8_C( 74), UINT8_C( 13), UINT8_C(127),
        UINT8_C( 57), UINT8_C( 14), UINT8_C( 53), UINT8_C(202), UINT8_C(224), UINT8_C(222), UINT8_C( 73), UINT8_C(250),
        UINT8_C( 46), UINT8_C(109), UINT8_C(150), UINT8_C( 18), UINT8_C( 72), UINT8_C( 99), UINT8_C(120), UINT8_C(117) },
      UINT32_C(2277282434) },
    { { UINT8_C(206), UINT8_C(120), UINT8_C(219), UINT8_C(220), UINT8_C(  6), UINT8_C(219), UINT8_C(162), UINT8_C( 56),
        UINT8_C(175), UINT8_C( 78), UINT8_C(217), UINT8_C( 93), UINT8_C(152), UINT8_C(230), UINT8_C(220), UINT8_C(209),
        UINT8_C(244), UINT8_C( 17), UINT8_C(155), UINT8_C(212), UINT8_C(239), UINT8_C(229), UINT8_C(207), UINT8_C( 29),
        UINT8_C( 82), UINT8_C(101), UINT8_C( 47), UINT8_C(154), UINT8_C(200), UINT8_C(167), UINT8_C( 15), UINT8_C(150) },
      { UINT8_C( 32), UINT8_C(235), UINT8_C(115), UINT8_C( 38), UINT8_C(198), UINT8_C( 21), UINT8_C( 95), UINT8_C(117),
        UINT8_C( 99), UINT8_C( 56), UINT8_C(210), UINT8_C(252), UINT8_C( 31), UINT8_C(174), UINT8_C(205), UINT8_C( 19),
        UINT8_C(191), UINT8_C(105), UINT8_C(232), UINT8_C(174), UINT8_C( 78), UINT8_C(183), UINT8_C(204), UINT8_C(160),
        UINT8_C( 28), UINT8_C(251), UINT8_C( 59), UINT8_C(228), UINT8_C(163), UINT8_C( 74), UINT8_C(122), UINT8_C(195) },
      UINT32_C(3464890514) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epu8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmplt_epu8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_test_x86_random_u8x32();
    easysimd__mmask32 r = easysimd_mm256_cmplt_epu8_mask(a, b);

    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[16];
    uint16_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { { UINT16_C(13055), UINT16_C(36968), UINT16_C(46975), UINT16_C(25742), UINT16_C(16326), UINT16_C(55188), UINT16_C(18017), UINT16_C(38477),
        UINT16_C(42128), UINT16_C(56848), UINT16_C(29378), UINT16_C( 9276), UINT16_C(30596), UINT16_C(62005), UINT16_C(31041), UINT16_C(16790) },
      { UINT16_C(65451), UINT16_C(11217), UINT16_C(24758), UINT16_C(32143), UINT16_C( 9119), UINT16_C(   84), UINT16_C(41578), UINT16_C(64150),
        UINT16_C(42566), UINT16_C( 2264), UINT16_C( 5145), UINT16_C(40237), UINT16_C(25227), UINT16_C(52367), UINT16_C( 9947), UINT16_C(34573) },
      UINT16_C(35273) },
    { { UINT16_C(57125), UINT16_C(56242), UINT16_C(16703), UINT16_C(56920), UINT16_C(44389), UINT16_C(53214), UINT16_C(29775), UINT16_C(38345),
        UINT16_C(41242), UINT16_C(13214), UINT16_C(52149), UINT16_C(16593), UINT16_C(24621), UINT16_C( 2316), UINT16_C( 6790), UINT16_C(43920) },
      { UINT16_C(17145), UINT16_C(14471), UINT16_C(57219), UINT16_C(59414), UINT16_C(62604), UINT16_C(56247), UINT16_C(32872), UINT16_C(33393),
        UINT16_C( 3873), UINT16_C(54966), UINT16_C(34778), UINT16_C( 1814), UINT16_C( 9191), UINT16_C(28176), UINT16_C(41021), UINT16_C(13849) },
      UINT16_C(25212) },
    { { UINT16_C(41186), UINT16_C(26222), UINT16_C(33920), UINT16_C( 3150), UINT16_C( 1656), UINT16_C(57576), UINT16_C(22918), UINT16_C(43106),
        UINT16_C( 6248), UINT16_C(17022), UINT16_C(38303), UINT16_C(34633), UINT16_C(23224), UINT16_C(62965), UINT16_C( 3834), UINT16_C(56619) },
      { UINT16_C(39343), UINT16_C(12099), UINT16_C(37149), UINT16_C(38203), UINT16_C( 9111), UINT16_C( 7797), UINT16_C(55164), UINT16_C(58566),
        UINT16_C(17648), UINT16_C(36646), UINT16_C(28889), UINT16_C(37142), UINT16_C( 3018), UINT16_C(50310), UINT16_C(45338), UINT16_C(51617) },
      UINT16_C(19420) },
    { { UINT16_C(58442), UINT16_C(26616), UINT16_C(13174), UINT16_C( 3580), UINT16_C(29015), UINT16_C(54059), UINT16_C(61769), UINT16_C(14776),
        UINT16_C(56886), UINT16_C( 4040), UINT16_C(57166), UINT16_C( 6305), UINT16_C(10218), UINT16_C( 1245), UINT16_C(32473), UINT16_C( 9165) },
      { UINT16_C(50531), UINT16_C(55691), UINT16_C(34809), UINT16_C(20710), UINT16_C( 4857), UINT16_C(16931), UINT16_C(56067), UINT16_C(14715),
        UINT16_C(17338), UINT16_C( 2121), UINT16_C(59938), UINT16_C( 3361), UINT16_C(65041), UINT16_C(59921), UINT16_C(57212), UINT16_C(57102) },
      UINT16_C(62478) },
    { { UINT16_C(39332), UINT16_C(40376), UINT16_C(40736), UINT16_C( 6637), UINT16_C( 4529), UINT16_C(46171), UINT16_C(55020), UINT16_C(42734),
        UINT16_C(14106), UINT16_C(15535), UINT16_C(53281), UINT16_C(12873), UINT16_C(23502), UINT16_C(18973), UINT16_C(11066), UINT16_C(56874) },
      { UINT16_C(58052), UINT16_C(58492), UINT16_C(27009), UINT16_C(13054), UINT16_C(22906), UINT16_C(26599), UINT16_C(54576), UINT16_C(18957),
        UINT16_C(48140), UINT16_C(11654), UINT16_C(53388), UINT16_C(23135), UINT16_C(31787), UINT16_C(26021), UINT16_C(53159), UINT16_C(27459) },
      UINT16_C(32027) },
    { { UINT16_C(49073), UINT16_C(13136), UINT16_C(20009), UINT16_C(41829), UINT16_C(19623), UINT16_C(55050), UINT16_C( 6177), UINT16_C(11553),
        UINT16_C(43220), UINT16_C(24922), UINT16_C(47736), UINT16_C(41915), UINT16_C(24630), UINT16_C(56840), UINT16_C(19247), UINT16_C(57673) },
      { UINT16_C(39179), UINT16_C(13332), UINT16_C(31207), UINT16_C(36823), UINT16_C(58054), UINT16_C(59238), UINT16_C(35066), UINT16_C(52757),
        UINT16_C(28464), UINT16_C(43055), UINT16_C(60201), UINT16_C(24651), UINT16_C(21323), UINT16_C(31550), UINT16_C(34718), UINT16_C(43356) },
      UINT16_C(18166) },
    { { UINT16_C(28705), UINT16_C( 2269), UINT16_C(46569), UINT16_C(44951), UINT16_C(65175), UINT16_C(37271), UINT16_C(44166), UINT16_C(46687),
        UINT16_C(36635), UINT16_C(17758), UINT16_C(43386), UINT16_C(50597), UINT16_C(58364), UINT16_C(39488), UINT16_C(40042), UINT16_C(35652) },
      { UINT16_C( 8460), UINT16_C(63124), UINT16_C(11222), UINT16_C(28069), UINT16_C(15401), UINT16_C(45054), UINT16_C(24296), UINT16_C( 1125),
        UINT16_C(50157), UINT16_C(26441), UINT16_C(61036), UINT16_C(26668), UINT16_C(28113), UINT16_C(15107), UINT16_C(18185), UINT16_C( 5831) },
      UINT16_C( 1826) },
    { { UINT16_C(23400), UINT16_C(16140), UINT16_C(45446), UINT16_C(45228), UINT16_C(44014), UINT16_C(54879), UINT16_C(50441), UINT16_C(63194),
        UINT16_C( 9096), UINT16_C(62813), UINT16_C(35089), UINT16_C(57949), UINT16_C(24822), UINT16_C(   30), UINT16_C(58791), UINT16_C( 4118) },
      { UINT16_C( 8768), UINT16_C(50767), UINT16_C(64467), UINT16_C(49526), UINT16_C(54950), UINT16_C(44952), UINT16_C(29339), UINT16_C( 9125),
        UINT16_C(  662), UINT16_C(42776), UINT16_C(30348), UINT16_C(33418), UINT16_C(43222), UINT16_C(32386), UINT16_C(39053), UINT16_C(52622) },
      UINT16_C(45086) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmplt_epu16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__mmask16 r = easysimd_mm256_cmplt_epu16_mask(a, b);

    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t a[8];
    uint32_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT32_C(3556508530), UINT32_C(2174898879), UINT32_C(2386224435), UINT32_C(1130644671), UINT32_C(1283342712), UINT32_C(1439914490), UINT32_C( 690938542), UINT32_C(3941761912) },
      { UINT32_C(1186852486), UINT32_C(1942445888), UINT32_C(  67174725), UINT32_C(3175572805), UINT32_C(2684994982), UINT32_C(3438730270), UINT32_C( 938812607), UINT32_C( 774039463) },
      UINT8_C(120) },
    { { UINT32_C( 359981013), UINT32_C(2223586111), UINT32_C(2189986364), UINT32_C(2520764400), UINT32_C(2989967764), UINT32_C(3833474341), UINT32_C(4162614097), UINT32_C( 807812699) },
      { UINT32_C(1548130845), UINT32_C( 316723157), UINT32_C(1234462809), UINT32_C(3420508983), UINT32_C(1115494172), UINT32_C(2502359876), UINT32_C(3398255215), UINT32_C(2667230336) },
      UINT8_C(137) },
    { { UINT32_C( 620380238), UINT32_C(1765202703), UINT32_C(2075314755), UINT32_C(3125187229), UINT32_C(3992765609), UINT32_C( 780280511), UINT32_C(3858239589), UINT32_C( 310637508) },
      { UINT32_C(1127644723), UINT32_C(2628545625), UINT32_C(3558301238), UINT32_C(2593021681), UINT32_C(3783821858), UINT32_C( 286263980), UINT32_C(3740731418), UINT32_C( 804354811) },
      UINT8_C(135) },
    { { UINT32_C(1366435832), UINT32_C(3404603028), UINT32_C(1839072636), UINT32_C(2231905379), UINT32_C(1667731638), UINT32_C(3044308890), UINT32_C(2073258879), UINT32_C(3735717094) },
      { UINT32_C(1076894892), UINT32_C(3054116410), UINT32_C(2267326755), UINT32_C(2349608149), UINT32_C(1458533308), UINT32_C(1779131370), UINT32_C(3051724751), UINT32_C(3482554147) },
      UINT8_C( 76) },
    { { UINT32_C(3843015595), UINT32_C(  94051041), UINT32_C(2559360963), UINT32_C(2804193515), UINT32_C(4143846156), UINT32_C(1180698999), UINT32_C(3439019432), UINT32_C(2140901076) },
      { UINT32_C( 862235474), UINT32_C(2285371589), UINT32_C(2871051455), UINT32_C(1766999389), UINT32_C(3479130200), UINT32_C(  34979930), UINT32_C(3670937605), UINT32_C(4049169055) },
      UINT8_C(198) },
    { { UINT32_C(3659841045), UINT32_C(2103598526), UINT32_C(2116583969), UINT32_C( 552041415), UINT32_C( 636438475), UINT32_C( 203949319), UINT32_C(3035035157), UINT32_C(1973764192) },
      { UINT32_C(3159345918), UINT32_C(1211740455), UINT32_C(4224148020), UINT32_C(2820386525), UINT32_C(4241361909), UINT32_C( 621409808), UINT32_C(1306193900), UINT32_C( 767721263) },
      UINT8_C( 60) },
    { { UINT32_C(1894322761), UINT32_C(4156105667), UINT32_C(1660125317), UINT32_C( 604638766), UINT32_C( 690018329), UINT32_C(3142527438), UINT32_C(1208494361), UINT32_C(4067871400) },
      { UINT32_C(2690801628), UINT32_C( 127343490), UINT32_C(3395914395), UINT32_C(3001971865), UINT32_C( 467406412), UINT32_C(1372990264), UINT32_C(4237942356), UINT32_C(2246971304) },
      UINT8_C( 77) },
    { { UINT32_C(4045754735), UINT32_C( 150584428), UINT32_C(3771884103), UINT32_C( 596820182), UINT32_C( 104754894), UINT32_C(3981907097), UINT32_C(2599088626), UINT32_C(1864357888) },
      { UINT32_C(2522956841), UINT32_C(1218337281), UINT32_C(2468901052), UINT32_C(4273388080), UINT32_C(3255170089), UINT32_C(4205861896), UINT32_C(1318361677), UINT32_C(2629678194) },
      UINT8_C(186) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmplt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__mmask16 r = easysimd_mm256_cmplt_epu32_mask(a, b);

    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmplt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint64_t a[4];
    uint64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT64_C(17170822660276055973), UINT64_C(14183939714599050625), UINT64_C( 7214131357747524637), UINT64_C( 6598360275907003086) },
      { UINT64_C( 1490608544086068869), UINT64_C( 9428910776671857169), UINT64_C(17514513406293469036), UINT64_C( 2837442183950011471) },
      UINT8_C(  4) },
    { { UINT64_C(10363793733856053548), UINT64_C(15522078791628567425), UINT64_C(13114362706432055783), UINT64_C(15485656635115440927) },
      { UINT64_C(16856806723482407139), UINT64_C(18221062805364118939), UINT64_C( 1489574325450727825), UINT64_C( 7174725141494326802) },
      UINT8_C(  3) },
    { { UINT64_C( 3089385296006227598), UINT64_C( 8671980043853999688), UINT64_C( 8825116179513237873), UINT64_C(11198137857182420867) },
      { UINT64_C( 2857095114488097774), UINT64_C( 6376779206231381791), UINT64_C(10241809025809116735), UINT64_C( 1022026984258099334) },
      UINT8_C(  4) },
    { { UINT64_C( 5704144485312282659), UINT64_C(14647910010526739720), UINT64_C(14004951554616133058), UINT64_C(  656544665500511726) },
      { UINT64_C(16195846784923969091), UINT64_C(12625867863850162657), UINT64_C( 3322205446013299484), UINT64_C( 3941873373134221253) },
      UINT8_C(  9) },
    { { UINT64_C(15327559275848056831), UINT64_C(13340640238352041606), UINT64_C( 4508995448019722276), UINT64_C( 2865025110449442970) },
      { UINT64_C( 6927579501563898972), UINT64_C(  256977715235939642), UINT64_C( 7371825018925785495), UINT64_C( 3394506666106107595) },
      UINT8_C( 12) },
    { { UINT64_C(  602535460043888930), UINT64_C(13756368378465890311), UINT64_C( 1676776453234905106), UINT64_C( 1104833183071549522) },
      { UINT64_C( 8941730437123527488), UINT64_C( 8754516535124621208), UINT64_C(13474380209022586011), UINT64_C( 8201675259722533589) },
      UINT8_C( 13) },
    { { UINT64_C(16067320945311800584), UINT64_C( 4715018838291143822), UINT64_C( 4146451195883997033), UINT64_C(12093235508864051761) },
      { UINT64_C( 8704929124153840555), UINT64_C(17465765885937347706), UINT64_C(16375456246654012324), UINT64_C(13113945597642357944) },
      UINT8_C( 14) },
    { { UINT64_C(13605220823284917100), UINT64_C(13716626472764784096), UINT64_C( 2692879264722067146), UINT64_C( 2955060358487516227) },
      { UINT64_C(18202228204863727774), UINT64_C( 9395566246418086027), UINT64_C( 8390591142311176672), UINT64_C(  570202786323623427) },
      UINT8_C(  5) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmplt_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmplt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__mmask16 r = easysimd_mm256_cmplt_epu64_mask(a, b);

    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    uint8_t a[32];
    uint8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C(2310139172),
      { UINT8_C( 75), UINT8_C( 21), UINT8_C( 47), UINT8_C(225), UINT8_C( 65),    UINT8_MAX, UINT8_C( 74), UINT8_C(161),
        UINT8_C(239), UINT8_C( 95), UINT8_C( 41), UINT8_C(236), UINT8_C(136), UINT8_C(123), UINT8_C( 74), UINT8_C(228),
        UINT8_C(146), UINT8_C( 22), UINT8_C(225), UINT8_C( 18), UINT8_C(107), UINT8_C(156), UINT8_C(235), UINT8_C(234),
        UINT8_C(193), UINT8_C( 15), UINT8_C(233), UINT8_C(229), UINT8_C(252), UINT8_C(155), UINT8_C(111), UINT8_C( 71) },
      { UINT8_C(176), UINT8_C(158), UINT8_C( 41), UINT8_C(241), UINT8_C(157), UINT8_C(115), UINT8_C(146), UINT8_C(141),
        UINT8_C(210), UINT8_C(187), UINT8_C(121), UINT8_C( 91), UINT8_C( 55), UINT8_C(195), UINT8_C( 63), UINT8_C(201),
        UINT8_C(217), UINT8_C( 32), UINT8_C(220), UINT8_C( 68), UINT8_C(189), UINT8_C(199), UINT8_C( 46), UINT8_C(126),
        UINT8_C(214), UINT8_C( 24), UINT8_C(100), UINT8_C(210), UINT8_C(179), UINT8_C(211), UINT8_C( 26), UINT8_C( 99) },
      UINT32_C(2167481344) },
    { UINT32_C( 240403313),
      { UINT8_C(182), UINT8_C(230), UINT8_C(155), UINT8_C(136), UINT8_C(161), UINT8_C( 20), UINT8_C(227), UINT8_C(216),
        UINT8_C(215), UINT8_C( 35), UINT8_C(162), UINT8_C(176), UINT8_C( 67), UINT8_C(126), UINT8_C(245), UINT8_C(  0),
        UINT8_C( 69), UINT8_C( 35), UINT8_C(127), UINT8_C( 28), UINT8_C( 59), UINT8_C(227), UINT8_C(238), UINT8_C(238),
        UINT8_C(182), UINT8_C(  8), UINT8_C( 81), UINT8_C( 39), UINT8_C( 75), UINT8_C(165), UINT8_C( 53), UINT8_C(  1) },
      { UINT8_C(139), UINT8_C(209), UINT8_C(138), UINT8_C( 45), UINT8_C(229), UINT8_C(109), UINT8_C(  5), UINT8_C(189),
        UINT8_C(144), UINT8_C(167), UINT8_C(109), UINT8_C(212), UINT8_C( 37), UINT8_C( 98), UINT8_C(212), UINT8_C(107),
        UINT8_C(134), UINT8_C( 83), UINT8_C(135), UINT8_C(193), UINT8_C( 54), UINT8_C(117), UINT8_C(176), UINT8_C(236),
        UINT8_C(126), UINT8_C(  1), UINT8_C( 19), UINT8_C(201), UINT8_C(167), UINT8_C( 73), UINT8_C(203), UINT8_C( 50) },
      UINT32_C( 134480432) },
    { UINT32_C(4284437786),
      { UINT8_C(194), UINT8_C(101), UINT8_C(188), UINT8_C( 83), UINT8_C( 12), UINT8_C( 42), UINT8_C( 39), UINT8_C( 50),
        UINT8_C(140), UINT8_C(251), UINT8_C(157), UINT8_C( 18), UINT8_C( 79), UINT8_C( 36), UINT8_C(212), UINT8_C(133),
        UINT8_C(153), UINT8_C(132), UINT8_C(114), UINT8_C( 23), UINT8_C(133), UINT8_C(133), UINT8_C(225), UINT8_C( 44),
        UINT8_C(206), UINT8_C(172), UINT8_C( 95), UINT8_C(232), UINT8_C(  1), UINT8_C(190), UINT8_C(232), UINT8_C(195) },
      { UINT8_C( 35), UINT8_C(164), UINT8_C( 22), UINT8_C( 48), UINT8_C(206), UINT8_C( 61), UINT8_C( 98), UINT8_C( 91),
        UINT8_C( 57),    UINT8_MAX, UINT8_C(109), UINT8_C(136), UINT8_C( 35), UINT8_C( 65), UINT8_C( 13), UINT8_C(188),
        UINT8_C(197), UINT8_C(127), UINT8_C(212), UINT8_C( 75), UINT8_C(  5), UINT8_C(181), UINT8_C(119), UINT8_C(211),
        UINT8_C( 97), UINT8_C(214), UINT8_C(188), UINT8_C( 98), UINT8_C(149), UINT8_C(164), UINT8_C( 37), UINT8_C(184) },
      UINT32_C( 369950738) },
    { UINT32_C( 401095752),
      { UINT8_C(121), UINT8_C( 74), UINT8_C(114), UINT8_C(178), UINT8_C( 73), UINT8_C(223), UINT8_C( 58), UINT8_C(108),
        UINT8_C( 33), UINT8_C( 72), UINT8_C( 41), UINT8_C(230), UINT8_C(199), UINT8_C(253), UINT8_C( 49), UINT8_C(204),
        UINT8_C(178), UINT8_C(169), UINT8_C(160), UINT8_C( 19), UINT8_C(127), UINT8_C( 92), UINT8_C(117), UINT8_C( 20),
        UINT8_C(  0), UINT8_C(154), UINT8_C(205), UINT8_C( 72), UINT8_C(214), UINT8_C(181), UINT8_C( 95), UINT8_C( 80) },
      { UINT8_C(  0), UINT8_C(209), UINT8_C(  2), UINT8_C( 73), UINT8_C(177), UINT8_C( 61), UINT8_C(182), UINT8_C(210),
        UINT8_C(133), UINT8_C(223), UINT8_C(184), UINT8_C( 76), UINT8_C(220), UINT8_C(234), UINT8_C( 25), UINT8_C(142),
        UINT8_C(147), UINT8_C(185), UINT8_C(161), UINT8_C( 18), UINT8_C( 21), UINT8_C( 22), UINT8_C( 39), UINT8_C( 21),
        UINT8_C(176), UINT8_C(244), UINT8_C( 93), UINT8_C(135), UINT8_C(169), UINT8_C(189), UINT8_C(215), UINT8_C(169) },
      UINT32_C(  58725440) },
    { UINT32_C(1072945550),
      { UINT8_C( 22), UINT8_C(169), UINT8_C( 17), UINT8_C(155), UINT8_C(136), UINT8_C(202), UINT8_C(232), UINT8_C(100),
        UINT8_C(180), UINT8_C(  1), UINT8_C(242), UINT8_C( 71), UINT8_C(186), UINT8_C(147), UINT8_C( 89), UINT8_C(207),
        UINT8_C(169), UINT8_C(128), UINT8_C(228), UINT8_C( 89), UINT8_C(116), UINT8_C( 65), UINT8_C(224), UINT8_C( 30),
        UINT8_C(254), UINT8_C(183), UINT8_C(199), UINT8_C(141), UINT8_C(145), UINT8_C(186), UINT8_C(204), UINT8_C(167) },
      { UINT8_C( 99), UINT8_C(222), UINT8_C( 67), UINT8_C(235), UINT8_C(168), UINT8_C( 43), UINT8_C( 79), UINT8_C( 92),
        UINT8_C( 44), UINT8_C( 65), UINT8_C(163), UINT8_C(230), UINT8_C(212), UINT8_C(252), UINT8_C(181), UINT8_C(125),
        UINT8_C(125), UINT8_C(153), UINT8_C(215), UINT8_C(241), UINT8_C(218), UINT8_C(183), UINT8_C( 15), UINT8_C(217),
        UINT8_C(111), UINT8_C(215), UINT8_C(102), UINT8_C(  0), UINT8_C(145), UINT8_C( 50), UINT8_C(167), UINT8_C(245) },
      UINT32_C(  45242382) },
    { UINT32_C(3101747728),
      { UINT8_C( 21), UINT8_C( 48), UINT8_C( 20), UINT8_C( 65), UINT8_C(113), UINT8_C(183), UINT8_C( 39), UINT8_C( 70),
        UINT8_C(180), UINT8_C(220), UINT8_C(195), UINT8_C( 49), UINT8_C(117), UINT8_C(154), UINT8_C( 34), UINT8_C( 80),
        UINT8_C( 82), UINT8_C( 50), UINT8_C( 41), UINT8_C(193), UINT8_C(  9), UINT8_C(143), UINT8_C(193), UINT8_C(154),
        UINT8_C(193), UINT8_C(104), UINT8_C(143), UINT8_C(210), UINT8_C( 83), UINT8_C(112), UINT8_C(138), UINT8_C(104) },
      { UINT8_C(160), UINT8_C(159), UINT8_C(170), UINT8_C( 17), UINT8_C( 86), UINT8_C(209), UINT8_C( 87), UINT8_C( 10),
        UINT8_C(174), UINT8_C( 27), UINT8_C( 59), UINT8_C( 35), UINT8_C(181), UINT8_C( 94), UINT8_C(115), UINT8_C(  7),
        UINT8_C(144), UINT8_C(156), UINT8_C(200), UINT8_C(153), UINT8_C( 43), UINT8_C(137), UINT8_C( 51), UINT8_C(237),
        UINT8_C(242), UINT8_C(195), UINT8_C(191), UINT8_C( 69), UINT8_C( 51), UINT8_C( 73), UINT8_C(173), UINT8_C(211) },
      UINT32_C(2155888640) },
    { UINT32_C(1071929320),
      { UINT8_C( 41), UINT8_C( 60), UINT8_C( 73), UINT8_C(215), UINT8_C( 87), UINT8_C(133), UINT8_C(250), UINT8_C( 12),
        UINT8_C(227), UINT8_C(110), UINT8_C( 20), UINT8_C(115), UINT8_C( 10), UINT8_C(220), UINT8_C( 12), UINT8_C( 54),
        UINT8_C(102), UINT8_C( 63), UINT8_C( 35), UINT8_C( 88), UINT8_C(  2), UINT8_C(226), UINT8_C(157), UINT8_C( 53),
        UINT8_C( 43), UINT8_C( 74), UINT8_C(  8), UINT8_C( 20), UINT8_C(162), UINT8_C(237), UINT8_C( 83), UINT8_C(203) },
      { UINT8_C( 41), UINT8_C(156), UINT8_C(162), UINT8_C(128), UINT8_C( 33), UINT8_C(156), UINT8_C(140), UINT8_C(  4),
        UINT8_C( 10), UINT8_C(160), UINT8_C(119), UINT8_C( 21), UINT8_C(125), UINT8_C(131), UINT8_C( 75), UINT8_C(227),
        UINT8_C(195), UINT8_C(110), UINT8_C( 59), UINT8_C(197), UINT8_C( 80), UINT8_C(216), UINT8_C(251), UINT8_C(123),
        UINT8_C( 34), UINT8_C(  3), UINT8_C(143), UINT8_C(196), UINT8_C(240), UINT8_C(226), UINT8_C(143), UINT8_C( 25) },
      UINT32_C( 482629152) },
    { UINT32_C(2694394239),
      { UINT8_C(206), UINT8_C( 38), UINT8_C(165), UINT8_C(216), UINT8_C(198), UINT8_C( 28), UINT8_C(237), UINT8_C( 67),
        UINT8_C(160), UINT8_C( 56), UINT8_C( 38), UINT8_C( 99), UINT8_C(166), UINT8_C( 97), UINT8_C( 40), UINT8_C(246),
        UINT8_C( 57), UINT8_C( 35), UINT8_C(114), UINT8_C( 92), UINT8_C( 39), UINT8_C(  1), UINT8_C( 32), UINT8_C( 23),
        UINT8_C(228), UINT8_C(176), UINT8_C( 49), UINT8_C( 99), UINT8_C(225), UINT8_C(202), UINT8_C(  3), UINT8_C(175) },
      { UINT8_C(240), UINT8_C(168), UINT8_C(136), UINT8_C(183), UINT8_C(197), UINT8_C(117), UINT8_C(250), UINT8_C(101),
        UINT8_C(174), UINT8_C( 33), UINT8_C(200), UINT8_C( 84), UINT8_C(130), UINT8_C(240), UINT8_C( 75), UINT8_C(188),
        UINT8_C( 20), UINT8_C(189), UINT8_C( 24), UINT8_C( 59), UINT8_C(190), UINT8_C( 56), UINT8_C( 82), UINT8_C(162),
        UINT8_C(232), UINT8_C(131), UINT8_C(  5), UINT8_C(202), UINT8_C( 78), UINT8_C(  9), UINT8_C(121), UINT8_C( 62) },
      UINT32_C(   9445731) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epu8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epu8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 r = easysimd_mm256_mask_cmplt_epu8_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    uint16_t a[16];
    uint16_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(63792),
      { UINT16_C(58987), UINT16_C(10999), UINT16_C(13858), UINT16_C(15996), UINT16_C( 1322), UINT16_C(24750), UINT16_C(28539), UINT16_C(   51),
        UINT16_C(19869), UINT16_C(22402), UINT16_C(60215), UINT16_C(18375), UINT16_C(38119), UINT16_C(64649), UINT16_C(47565), UINT16_C(14581) },
      { UINT16_C(60575), UINT16_C(49506), UINT16_C(57122), UINT16_C(19711), UINT16_C(44516), UINT16_C(24493), UINT16_C(57372), UINT16_C(47455),
        UINT16_C(57901), UINT16_C(25617), UINT16_C(55501), UINT16_C(46252), UINT16_C(13676), UINT16_C(14768), UINT16_C(42478), UINT16_C(36466) },
      UINT16_C(35088) },
    { UINT16_C(54417),
      { UINT16_C(46159), UINT16_C(20403), UINT16_C(38912), UINT16_C(44540), UINT16_C( 6647), UINT16_C(22413), UINT16_C(47826), UINT16_C(58169),
        UINT16_C( 1567), UINT16_C(52155), UINT16_C(10427), UINT16_C(27392), UINT16_C(61025), UINT16_C(54033), UINT16_C(41596), UINT16_C(52392) },
      { UINT16_C(23382), UINT16_C(22299), UINT16_C( 6131), UINT16_C(60164), UINT16_C(37424), UINT16_C(  834), UINT16_C(31564), UINT16_C(27622),
        UINT16_C(41601), UINT16_C(15414), UINT16_C(14026), UINT16_C(11176), UINT16_C(47397), UINT16_C(41471), UINT16_C(42843), UINT16_C(45677) },
      UINT16_C(17424) },
    { UINT16_C(34818),
      { UINT16_C(62985), UINT16_C( 3488), UINT16_C(53473), UINT16_C( 9119), UINT16_C(60627), UINT16_C(47774), UINT16_C( 8023), UINT16_C(36444),
        UINT16_C( 9820), UINT16_C( 1220), UINT16_C(59729), UINT16_C(20669), UINT16_C( 6283), UINT16_C(63735), UINT16_C(64202), UINT16_C(54145) },
      { UINT16_C( 8688), UINT16_C(53729), UINT16_C(33009), UINT16_C(50676), UINT16_C(37484), UINT16_C(50303), UINT16_C(56241), UINT16_C( 3410),
        UINT16_C( 5633), UINT16_C(21009), UINT16_C(52736), UINT16_C(35747), UINT16_C(39655), UINT16_C(45443), UINT16_C( 1172), UINT16_C(33925) },
      UINT16_C( 2050) },
    { UINT16_C(26149),
      { UINT16_C( 5973), UINT16_C(18918), UINT16_C(21468), UINT16_C(23515), UINT16_C(36119), UINT16_C(26934), UINT16_C(14234), UINT16_C(44159),
        UINT16_C(32649), UINT16_C(11386), UINT16_C(24842), UINT16_C(36551), UINT16_C(23315), UINT16_C(39058), UINT16_C(47328), UINT16_C(13822) },
      { UINT16_C(58575), UINT16_C(43903), UINT16_C(23095), UINT16_C(19974), UINT16_C(15591), UINT16_C(33463), UINT16_C(14195), UINT16_C(64558),
        UINT16_C(43190), UINT16_C(49449), UINT16_C(61450), UINT16_C( 7503), UINT16_C(57675), UINT16_C(11189), UINT16_C(45977), UINT16_C(26721) },
      UINT16_C( 1573) },
    { UINT16_C(57495),
      { UINT16_C(53011), UINT16_C( 6458), UINT16_C( 8733), UINT16_C(54613), UINT16_C(51364), UINT16_C(53772), UINT16_C(49861), UINT16_C(61050),
        UINT16_C(33923), UINT16_C(53982), UINT16_C(10657), UINT16_C(22196), UINT16_C(19797), UINT16_C(46601), UINT16_C(41398), UINT16_C(51606) },
      { UINT16_C(53360), UINT16_C(36323), UINT16_C(14578), UINT16_C(38498), UINT16_C(28161), UINT16_C(50792), UINT16_C(58161), UINT16_C(46260),
        UINT16_C(37479), UINT16_C( 2439), UINT16_C(15291), UINT16_C( 4191), UINT16_C(27016), UINT16_C(16070), UINT16_C(23562), UINT16_C(31240) },
      UINT16_C(    7) },
    { UINT16_C(60205),
      { UINT16_C( 7943), UINT16_C(27171), UINT16_C( 9398), UINT16_C( 7896), UINT16_C( 2538), UINT16_C(40449), UINT16_C(27070), UINT16_C(17712),
        UINT16_C(60530), UINT16_C(53632), UINT16_C( 2300), UINT16_C(49978), UINT16_C(17479), UINT16_C(20255), UINT16_C(19646), UINT16_C(50746) },
      { UINT16_C(23916), UINT16_C( 8752), UINT16_C( 2178), UINT16_C(27712), UINT16_C(16914), UINT16_C(53259), UINT16_C(15275), UINT16_C( 7445),
        UINT16_C(38183), UINT16_C( 9454), UINT16_C(10653), UINT16_C(58599), UINT16_C( 1645), UINT16_C(11315), UINT16_C(27987), UINT16_C(49138) },
      UINT16_C(18473) },
    { UINT16_C( 8907),
      { UINT16_C(19937), UINT16_C( 8490), UINT16_C(15545), UINT16_C(50275), UINT16_C( 3596), UINT16_C( 8448), UINT16_C(10027), UINT16_C( 6838),
        UINT16_C(21579), UINT16_C(12867), UINT16_C(45112), UINT16_C(27705), UINT16_C(36060), UINT16_C(52953), UINT16_C(42059), UINT16_C(11504) },
      { UINT16_C( 7153), UINT16_C(43853), UINT16_C(45399), UINT16_C(25711), UINT16_C(28607), UINT16_C(60293), UINT16_C(15511), UINT16_C(57861),
        UINT16_C(18576), UINT16_C(51221), UINT16_C(20216), UINT16_C(54580), UINT16_C( 3802), UINT16_C( 9635), UINT16_C(38066), UINT16_C(42065) },
      UINT16_C(  706) },
    { UINT16_C(40623),
      { UINT16_C( 1615), UINT16_C(48719), UINT16_C( 3946), UINT16_C(61486), UINT16_C(50682), UINT16_C(65324), UINT16_C(48295), UINT16_C(48199),
        UINT16_C(16260), UINT16_C(47370), UINT16_C(58388), UINT16_C(47303), UINT16_C(30985), UINT16_C(23116), UINT16_C(64285), UINT16_C(27897) },
      { UINT16_C(18433), UINT16_C(27691), UINT16_C(22871), UINT16_C(20828), UINT16_C(34846), UINT16_C(50512), UINT16_C(38724), UINT16_C(51330),
        UINT16_C(36055), UINT16_C(60289), UINT16_C(18545), UINT16_C(31395), UINT16_C(61378), UINT16_C(57301), UINT16_C(52970), UINT16_C(60492) },
      UINT16_C(37509) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epu16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epu16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 r = easysimd_mm256_mask_cmplt_epu16_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint32_t a[8];
    uint32_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 13),
      { UINT32_C( 551128270), UINT32_C(1960008781), UINT32_C( 606671176), UINT32_C(1409500845), UINT32_C(4017936960), UINT32_C(1779647402), UINT32_C(2312225318), UINT32_C(3382053115) },
      { UINT32_C(3807014804), UINT32_C( 341229003), UINT32_C(2067300302), UINT32_C( 114244550), UINT32_C(3237301270), UINT32_C(3039430799), UINT32_C(3326016459), UINT32_C(2559563012) },
      UINT8_C(  5) },
    { UINT8_C( 68),
      { UINT32_C( 907049593), UINT32_C(1342514384), UINT32_C(2551611484), UINT32_C(2628656208), UINT32_C( 439053842), UINT32_C(2498093209), UINT32_C(4103645983), UINT32_C(3023581499) },
      { UINT32_C(2095794347), UINT32_C(3385651308), UINT32_C(3227640432), UINT32_C( 274468862), UINT32_C( 371951485), UINT32_C(2276135016), UINT32_C(4152050620), UINT32_C( 548123508) },
      UINT8_C( 68) },
    { UINT8_C(252),
      { UINT32_C(2255002774), UINT32_C(1257714024), UINT32_C(2705897362), UINT32_C(2602522900), UINT32_C(2483238276), UINT32_C( 609258208), UINT32_C(3113764613), UINT32_C(2310387955) },
      { UINT32_C(3155172692), UINT32_C(3775268687), UINT32_C(3531820990), UINT32_C( 745448104), UINT32_C(3099619799), UINT32_C(  31199740), UINT32_C(1270510680), UINT32_C(2178248493) },
      UINT8_C( 20) },
    { UINT8_C(141),
      { UINT32_C(3973856997), UINT32_C(2477440324), UINT32_C(3795549504), UINT32_C(1555720171), UINT32_C( 962097704), UINT32_C(3264305742), UINT32_C(2230312212), UINT32_C(2534502834) },
      { UINT32_C(4085509551), UINT32_C(3951504810), UINT32_C(2513289898), UINT32_C(1391626026), UINT32_C(1200311033), UINT32_C(3104447652), UINT32_C(2872965625), UINT32_C( 423775850) },
      UINT8_C(  1) },
    { UINT8_C( 59),
      { UINT32_C(4091874757), UINT32_C(1453183124), UINT32_C( 629158814), UINT32_C(1864356389), UINT32_C(2048157277), UINT32_C(1785974128), UINT32_C(1490296586), UINT32_C( 664006241) },
      { UINT32_C(2400876795), UINT32_C(3890591561), UINT32_C( 252471018), UINT32_C(2524916536), UINT32_C(  34640786), UINT32_C(1785496416), UINT32_C(  79839650), UINT32_C( 707482927) },
      UINT8_C( 10) },
    { UINT8_C(205),
      { UINT32_C(4229347653), UINT32_C(  82312606), UINT32_C( 893253130), UINT32_C( 147313525), UINT32_C(1718143715), UINT32_C(1997132342), UINT32_C(3919973780), UINT32_C(2109198392) },
      { UINT32_C( 679071113), UINT32_C(3576455371), UINT32_C(3423234391), UINT32_C( 534106684), UINT32_C(3532012956), UINT32_C(2756349712), UINT32_C(3566137500), UINT32_C(1246840257) },
      UINT8_C( 12) },
    { UINT8_C( 18),
      { UINT32_C( 735933130), UINT32_C( 142783135), UINT32_C(2403684029), UINT32_C(1630233635), UINT32_C(2037513706), UINT32_C( 940905799), UINT32_C(3908692387), UINT32_C( 100352826) },
      { UINT32_C(1429264566), UINT32_C(1214100107), UINT32_C( 618111488), UINT32_C(4052025863), UINT32_C(1198257919), UINT32_C(2944368651), UINT32_C(2761390186), UINT32_C(1906938555) },
      UINT8_C(  2) },
    { UINT8_C(107),
      { UINT32_C(2348205785), UINT32_C(3331079716), UINT32_C( 399355925), UINT32_C( 722911029), UINT32_C(2855689514), UINT32_C(1410655708), UINT32_C( 252754301), UINT32_C(1014661474) },
      { UINT32_C(1825009736), UINT32_C(3274855342), UINT32_C( 953810947), UINT32_C(3915641279), UINT32_C( 731093582), UINT32_C(4236224639), UINT32_C(3272380257), UINT32_C(1509918225) },
      UINT8_C(104) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epu32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 r = easysimd_mm256_mask_cmplt_epu32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmplt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint64_t a[4];
    uint64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(246),
      { UINT64_C( 1273985844701224424), UINT64_C( 1925887459947050194), UINT64_C(13211990046268215444), UINT64_C(  578301472483921595) },
      { UINT64_C( 4529196725281378664), UINT64_C( 1400584846612783076), UINT64_C( 7339417863205109552), UINT64_C( 5355198743240333096) },
      UINT8_C(  0) },
    { UINT8_C(205),
      { UINT64_C( 5505783501686379155), UINT64_C( 7759384590204779934), UINT64_C(  170989082838880350), UINT64_C(10103877412797589743) },
      { UINT64_C( 9761967709692653888), UINT64_C( 9353328965668127798), UINT64_C( 5345799537593803140), UINT64_C( 9459508938177582272) },
      UINT8_C(  5) },
    { UINT8_C( 94),
      { UINT64_C( 7855912903495954518), UINT64_C( 1588664785393772501), UINT64_C( 9269740005103844299), UINT64_C(18067110902376849008) },
      { UINT64_C(12145310800772989941), UINT64_C(18122833649386659127), UINT64_C( 2117681032669950584), UINT64_C( 7025675254126749409) },
      UINT8_C(  2) },
    { UINT8_C( 57),
      { UINT64_C( 2905019719411859274), UINT64_C( 3502203182268333285), UINT64_C(  547182826586209586), UINT64_C(16637937068234906741) },
      { UINT64_C(10054093368599834963), UINT64_C( 7240177632278339340), UINT64_C(14460126372583528388), UINT64_C( 4994589784928362445) },
      UINT8_C(  1) },
    { UINT8_C( 74),
      { UINT64_C(  769406538664158288), UINT64_C(14426050440455845672), UINT64_C(11833176708089633209), UINT64_C( 4682016213924552790) },
      { UINT64_C(10139166699126122880), UINT64_C( 5457851002868461957), UINT64_C( 1383146855154964743), UINT64_C( 2719536069455562892) },
      UINT8_C(  0) },
    { UINT8_C(164),
      { UINT64_C( 8696895659414442398), UINT64_C( 4595171795036383119), UINT64_C( 5512985953296994825), UINT64_C( 6570776855407886289) },
      { UINT64_C( 6622043852285360359), UINT64_C( 5586395673103232992), UINT64_C( 7133637564346630703), UINT64_C(17211383414208262340) },
      UINT8_C(  4) },
    { UINT8_C( 86),
      { UINT64_C( 3565862730011401315), UINT64_C(14673505256163623829), UINT64_C( 4940376587647079686), UINT64_C( 5629563219087134636) },
      { UINT64_C( 6055202673146872469), UINT64_C( 5133375979603328553), UINT64_C(  241119975435311551), UINT64_C( 8208353327621972912) },
      UINT8_C(  0) },
    { UINT8_C( 47),
      { UINT64_C(14943862004705544403), UINT64_C(12480696208413742602), UINT64_C( 4952280223703124073), UINT64_C( 1437372033638880981) },
      { UINT64_C( 3385211739314184666), UINT64_C(14575855533214075526), UINT64_C( 2955363370348595999), UINT64_C(10392019539065578725) },
      UINT8_C( 10) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmplt_epu64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmplt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 r = easysimd_mm256_mask_cmplt_epu64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}


static int
test_easysimd_mm_cmpnlt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int8_t a[16];
    const int8_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { {  INT8_C(  34),  INT8_C( 103), -INT8_C(  33), -INT8_C(  83),  INT8_C( 102), -INT8_C(  71), -INT8_C( 119),  INT8_C(  65),
        -INT8_C( 106),  INT8_C(  85), -INT8_C(  17),  INT8_C( 116), -INT8_C(  36),  INT8_C(  92), -INT8_C(  96),  INT8_C(  22) },
      {  INT8_C(  30), -INT8_C(  47), -INT8_C(  89),  INT8_C(  80), -INT8_C(  93),  INT8_C(  91), -INT8_C(  31), -INT8_C(  34),
        -INT8_C(  97),  INT8_C(  10), -INT8_C(  61),  INT8_C(  39), -INT8_C(  42), -INT8_C(  63), -INT8_C( 108), -INT8_C(   8) },
      UINT16_C(65175) },
    { {  INT8_C(  41),  INT8_C( 115), -INT8_C(  90), -INT8_C( 113),  INT8_C(  45),  INT8_C(  47), -INT8_C(  48), -INT8_C(  61),
        -INT8_C( 124), -INT8_C(  65),  INT8_C(  55),  INT8_C(  97),  INT8_C(  27), -INT8_C(  41),  INT8_C( 119),  INT8_C(  58) },
      { -INT8_C(  88),  INT8_C(  30), -INT8_C( 118),  INT8_C(  76),  INT8_C( 122),  INT8_C( 107),  INT8_C(  42),  INT8_C(  25),
         INT8_C( 118), -INT8_C(  19),  INT8_C(  65),  INT8_C(  76), -INT8_C(  82), -INT8_C(  43),  INT8_C(  69), -INT8_C(  41) },
      UINT16_C(63495) },
    { {  INT8_C(  73), -INT8_C(  21),  INT8_C( 102),  INT8_C( 118),  INT8_C(  26),  INT8_C(  55),  INT8_C(  57), -INT8_C(  97),
        -INT8_C(  10),  INT8_C( 113),  INT8_C(   0),  INT8_C(  18),  INT8_C(  72),  INT8_C( 119),  INT8_C(  76), -INT8_C(  15) },
      { -INT8_C( 107), -INT8_C(  42),  INT8_C(  61),  INT8_C(  15),  INT8_C(  66),  INT8_C( 103),  INT8_C(  41), -INT8_C(  72),
         INT8_C(  84),  INT8_C( 106),  INT8_C(   4),  INT8_C(   2),  INT8_C(  63),  INT8_C(  73), -INT8_C(  38), -INT8_C( 120) },
      UINT16_C(64079) },
    { {  INT8_C(  52),  INT8_C(  64), -INT8_C(   2),  INT8_C(  79),  INT8_C( 119),  INT8_C(  56), -INT8_C(  18),  INT8_C( 110),
        -INT8_C(  87), -INT8_C(  18),      INT8_MIN, -INT8_C(  15),  INT8_C( 101), -INT8_C(  52), -INT8_C(  30), -INT8_C(   6) },
      { -INT8_C(  94),  INT8_C(  31),  INT8_C(  10), -INT8_C(  28), -INT8_C( 122),  INT8_C(  51), -INT8_C( 100), -INT8_C(  38),
        -INT8_C(  99), -INT8_C(  95), -INT8_C(  35), -INT8_C(  36), -INT8_C(  22), -INT8_C(  73),  INT8_C( 101),  INT8_C(  31) },
      UINT16_C(15355) },
    { { -INT8_C(   9),  INT8_C(  99),  INT8_C( 110),  INT8_C( 111), -INT8_C( 101),  INT8_C(  92), -INT8_C(  35),  INT8_C(  68),
         INT8_C(  74),  INT8_C(  93),  INT8_C(  54), -INT8_C(  81),  INT8_C(  41),  INT8_C(  24), -INT8_C(  87), -INT8_C(  53) },
      {  INT8_C(  56), -INT8_C(  77), -INT8_C(  80), -INT8_C(  66), -INT8_C(  26),  INT8_C(  76), -INT8_C( 103), -INT8_C( 125),
        -INT8_C(  19),  INT8_C( 118),  INT8_C(  96), -INT8_C(  40),  INT8_C(  45), -INT8_C(  59), -INT8_C(   9),  INT8_C(  36) },
      UINT16_C( 8686) },
    { {  INT8_C(  40),  INT8_C( 101), -INT8_C( 109), -INT8_C(  60), -INT8_C(  63),  INT8_C( 112),  INT8_C(   8),  INT8_C(  11),
        -INT8_C(  51),  INT8_C(  62), -INT8_C(  70), -INT8_C(  10),  INT8_C(  87),  INT8_C(  99), -INT8_C(  62), -INT8_C( 113) },
      {  INT8_C(  23),  INT8_C( 114),  INT8_C(  77), -INT8_C(   3), -INT8_C(  66), -INT8_C(  26), -INT8_C( 127), -INT8_C(  84),
         INT8_C(  92), -INT8_C(  31), -INT8_C( 124), -INT8_C( 119), -INT8_C(  90),  INT8_C( 123), -INT8_C(  82), -INT8_C(  50) },
      UINT16_C(24305) },
    { { -INT8_C(  32),  INT8_C(  65), -INT8_C( 110), -INT8_C(  95), -INT8_C(  78), -INT8_C( 101), -INT8_C(  84),      INT8_MAX,
        -INT8_C(  39),  INT8_C( 102),  INT8_C( 118),  INT8_C(  48), -INT8_C(  55),  INT8_C(  56), -INT8_C(  65), -INT8_C(  32) },
      { -INT8_C(  86),  INT8_C(  13), -INT8_C(  34),  INT8_C( 104), -INT8_C(  13),  INT8_C(  95),  INT8_C(  20),  INT8_C(  80),
         INT8_C(  64), -INT8_C( 104), -INT8_C(  39), -INT8_C(  26),  INT8_C(  19), -INT8_C( 121), -INT8_C(  76), -INT8_C(  13) },
      UINT16_C(28291) },
    { { -INT8_C(  55),  INT8_C(  71), -INT8_C( 108),  INT8_C( 123), -INT8_C(  30),  INT8_C(  64), -INT8_C(   6), -INT8_C(  69),
        -INT8_C(  90),  INT8_C( 112), -INT8_C(  20),  INT8_C( 112), -INT8_C(  88), -INT8_C(  85),  INT8_C(  80),  INT8_C(  82) },
      { -INT8_C(  72),  INT8_C(  46), -INT8_C(  69), -INT8_C(  84), -INT8_C( 115), -INT8_C(  49), -INT8_C(   4), -INT8_C(  51),
         INT8_C( 104), -INT8_C(  43), -INT8_C(  77),  INT8_C( 123),  INT8_C(  93),  INT8_C( 104),  INT8_C( 111),  INT8_C(  38) },
      UINT16_C(34363) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpnlt_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cmpnlt_epi8_mask")
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_cmpnlt_epi8_mask(a, b);

    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpnlt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t a[8];
    const int32_t b[8];
    const uint8_t r;
  } test_vec[8] = {
    { { -INT32_C(   331229613), -INT32_C(   164444608), -INT32_C(  1393900103), -INT32_C(  1636452981), -INT32_C(  1291436434), -INT32_C(  1825634129), -INT32_C(  1816098474), -INT32_C(  1437571241) },
      {  INT32_C(  2107085117),  INT32_C(   292800855), -INT32_C(  1393900103), -INT32_C(  1636452981),  INT32_C(   523550064),  INT32_C(   464741317),  INT32_C(  1135571948),  INT32_C(   401539034) },
      UINT8_C( 12) },
    { {  INT32_C(  1995880036),  INT32_C(  1509338857),  INT32_C(   830024299),  INT32_C(  2051812493),  INT32_C(  2042493855), -INT32_C(  1936675845),  INT32_C(  2138317872), -INT32_C(  1889767893) },
      {  INT32_C(  1995880036),  INT32_C(  1818753792),  INT32_C(   830024299), -INT32_C(  1423054580), -INT32_C(   534451227), -INT32_C(   949177194),  INT32_C(    54976728), -INT32_C(  1889767893) },
      UINT8_C(221) },
    { { -INT32_C(   747831609), -INT32_C(   864118297),  INT32_C(   464298628),  INT32_C(   786569302),  INT32_C(  1194404089), -INT32_C(  1095056438), -INT32_C(   307020966), -INT32_C(   358280158) },
      { -INT32_C(    88272622),  INT32_C(   801520811),  INT32_C(   894071774), -INT32_C(  2073875317),  INT32_C(   516724052), -INT32_C(  1095056438), -INT32_C(   459239231), -INT32_C(  1211218523) },
      UINT8_C(248) },
    { {  INT32_C(  1994095851), -INT32_C(    33866072),  INT32_C(   740018131),  INT32_C(   266336333),  INT32_C(   787709833),  INT32_C(   468042180),  INT32_C(   337483596), -INT32_C(    88453361) },
      {  INT32_C(  1936758219), -INT32_C(  1485804589),  INT32_C(   740018131),  INT32_C(   227521156), -INT32_C(   180649423),  INT32_C(   468042180), -INT32_C(   945606984),  INT32_C(  1220631933) },
      UINT8_C(127) },
    { {  INT32_C(  1028604088),  INT32_C(   592109298), -INT32_C(  1558673824),  INT32_C(  1597188519), -INT32_C(   718875048),  INT32_C(   639559476), -INT32_C(  1242768872), -INT32_C(  1081779962) },
      { -INT32_C(  1946364775),  INT32_C(   330188467),  INT32_C(  1941424076),  INT32_C(  1221782256), -INT32_C(   199296832),  INT32_C(   639559476),  INT32_C(   481101590), -INT32_C(    86298015) },
      UINT8_C( 43) },
    { { -INT32_C(   329349893),  INT32_C(   791949422),  INT32_C(   136532521), -INT32_C(  1543487858), -INT32_C(  1480479162), -INT32_C(   408838944), -INT32_C(  1834932364),  INT32_C(  1467772764) },
      { -INT32_C(   329349893),  INT32_C(   791949422),  INT32_C(  1497150666),  INT32_C(   972897779), -INT32_C(  1480479162), -INT32_C(   810319525),  INT32_C(    90264745), -INT32_C(  1218650685) },
      UINT8_C(179) },
    { {  INT32_C(  1362834270),  INT32_C(  1535785328),  INT32_C(  1378314999), -INT32_C(  1792943380), -INT32_C(   224755154),  INT32_C(   413791840), -INT32_C(  1361329514), -INT32_C(   896413076) },
      {  INT32_C(  1981598725),  INT32_C(  1535785328),  INT32_C(  1378314999),  INT32_C(    26374610),  INT32_C(  1408445683), -INT32_C(  1184130014), -INT32_C(   899201442), -INT32_C(   896413076) },
      UINT8_C(166) },
    { {  INT32_C(   811931486),  INT32_C(  2066872200),  INT32_C(  1171203107),  INT32_C(   519977664), -INT32_C(  1712822655),  INT32_C(   599227742), -INT32_C(  2064757971),  INT32_C(   267171249) },
      {  INT32_C(  1413435852),  INT32_C(  1808822600),  INT32_C(  1171203107),  INT32_C(  1517596633),  INT32_C(  1962171669),  INT32_C(   599227742), -INT32_C(  1467185673),  INT32_C(   129464379) },
      UINT8_C(166) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpnlt_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm256_cmpnlt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[8];
    int32_t b_[8];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if ((easysimd_test_codegen_rand() & 3) == 0)
        b_[j] = a_[j];

    easysimd__m256i a = easysimd_mm256_loadu_epi32(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(b_);
    easysimd__mmask8 r = easysimd_mm256_cmpnlt_epi32_mask(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmplt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask64 r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C( -77), INT8_C( -25), INT8_C( -46), INT8_C(  46),
                           INT8_C(  13), INT8_C(   4), INT8_C( -85), INT8_C( -85),
                           INT8_C( -84), INT8_C( -54), INT8_C(  24), INT8_C(  27),
                           INT8_C(-115), INT8_C(  32), INT8_C( -88), INT8_C( -66),
                           INT8_C( 113), INT8_C( -86), INT8_C(  24), INT8_C(  39),
                           INT8_C( -74), INT8_C(  61), INT8_C( 110), INT8_C(  70),
                           INT8_C(  64), INT8_C(  85), INT8_C(-108), INT8_C(  -7),
                           INT8_C(-115), INT8_C( -79), INT8_C( -97), INT8_C( -25),
                           INT8_C( 108), INT8_C( -91), INT8_C(  98), INT8_C(  90),
                           INT8_C( -86), INT8_C(-115), INT8_C(  85), INT8_C( -77),
                           INT8_C( -94), INT8_C( -97), INT8_C( 118), INT8_C( -72),
                           INT8_C( -59), INT8_C(   1), INT8_C(   5), INT8_C(  40),
                           INT8_C( -51), INT8_C(-106), INT8_C( -27), INT8_C(-101),
                           INT8_C(  27), INT8_C( -43), INT8_C( -67), INT8_C(-118),
                           INT8_C(   9), INT8_C( -56), INT8_C(-102), INT8_C( -49),
                           INT8_C( -36), INT8_C(  56), INT8_C(  85), INT8_C(  82)),
      easysimd_mm512_set_epi8(INT8_C(  15), INT8_C( -44), INT8_C(  -4), INT8_C(  65),
                           INT8_C( 115), INT8_C(  75), INT8_C(-128), INT8_C( -29),
                           INT8_C( -41), INT8_C( -89), INT8_C( -75), INT8_C( -12),
                           INT8_C(   8), INT8_C( -18), INT8_C(   0), INT8_C(  50),
                           INT8_C( -20), INT8_C(  66), INT8_C(  59), INT8_C(  42),
                           INT8_C( 112), INT8_C(-128), INT8_C(  83), INT8_C(   7),
                           INT8_C(  66), INT8_C( -29), INT8_C( -70), INT8_C(  42),
                           INT8_C(-100), INT8_C(  85), INT8_C( -81), INT8_C( -93),
                           INT8_C(   9), INT8_C( -66), INT8_C( -12), INT8_C(  91),
                           INT8_C(  12), INT8_C(  45), INT8_C( 127), INT8_C(-123),
                           INT8_C( -53), INT8_C(  78), INT8_C(  39), INT8_C(-107),
                           INT8_C(   4), INT8_C( -32), INT8_C( -36), INT8_C(  -2),
                           INT8_C( -88), INT8_C(  39), INT8_C( -66), INT8_C( -14),
                           INT8_C( -93), INT8_C( -77), INT8_C( -69), INT8_C(   8),
                           INT8_C(-103), INT8_C( 105), INT8_C(  72), INT8_C(  -8),
                           INT8_C(  49), INT8_C(-104), INT8_C(  40), INT8_C(  17)),
      UINT64_C(13658143053960466808) },
    { easysimd_mm512_set_epi8(INT8_C(-103), INT8_C(  71), INT8_C(  97), INT8_C(  13),
                           INT8_C( -23), INT8_C(-103), INT8_C(-115), INT8_C(  49),
                           INT8_C(  20), INT8_C(  -2), INT8_C(  49), INT8_C(  20),
                           INT8_C( -48), INT8_C(  19), INT8_C(  36), INT8_C(   5),
                           INT8_C( 110), INT8_C( -17), INT8_C( -78), INT8_C( -81),
                           INT8_C( -87), INT8_C( -42), INT8_C( -90), INT8_C( -74),
                           INT8_C( -81), INT8_C( -59), INT8_C( -73), INT8_C( 101),
                           INT8_C( -93), INT8_C(  16), INT8_C(  54), INT8_C(-126),
                           INT8_C(  73), INT8_C(-113), INT8_C( -63), INT8_C(   0),
                           INT8_C(  -4), INT8_C( -98), INT8_C(  69), INT8_C( -82),
                           INT8_C(-103), INT8_C(-100), INT8_C( -26), INT8_C(   6),
                           INT8_C( -99), INT8_C(  22), INT8_C(   1), INT8_C(-119),
                           INT8_C(  62), INT8_C( -91), INT8_C( 126), INT8_C( -42),
                           INT8_C( -88), INT8_C( -10), INT8_C(  66), INT8_C(-108),
                           INT8_C(   4), INT8_C( -81), INT8_C(-101), INT8_C( -87),
                           INT8_C( -36), INT8_C( 114), INT8_C( 105), INT8_C(  70)),
      easysimd_mm512_set_epi8(INT8_C(-106), INT8_C( 118), INT8_C(-104), INT8_C(  91),
                           INT8_C(  22), INT8_C( -65), INT8_C(  99), INT8_C(  90),
                           INT8_C( -59), INT8_C( 121), INT8_C(  74), INT8_C(  82),
                           INT8_C(  17), INT8_C( -59), INT8_C(  19), INT8_C(-103),
                           INT8_C(  84), INT8_C(  62), INT8_C( 105), INT8_C( -37),
                           INT8_C( -29), INT8_C(-118), INT8_C(  43), INT8_C(  43),
                           INT8_C(  33), INT8_C(-102), INT8_C(  -5), INT8_C( -51),
                           INT8_C( -22), INT8_C(   3), INT8_C(  47), INT8_C( -50),
                           INT8_C( -25), INT8_C(  25), INT8_C( -33), INT8_C( -68),
                           INT8_C(  52), INT8_C( -60), INT8_C(  19), INT8_C(-103),
                           INT8_C(  71), INT8_C(  88), INT8_C(  82), INT8_C( -50),
                           INT8_C( -90), INT8_C(  -3), INT8_C( -25), INT8_C(  35),
                           INT8_C(-100), INT8_C( -67), INT8_C(  46), INT8_C(  77),
                           INT8_C( -70), INT8_C(   1), INT8_C(  82), INT8_C( 114),
                           INT8_C( -55), INT8_C(  11), INT8_C( -95), INT8_C( 111),
                           INT8_C(  48), INT8_C(  71), INT8_C(  90), INT8_C( -66)),
      UINT64_C(6879384398415355768) },
    { easysimd_mm512_set_epi8(INT8_C( -71), INT8_C( 111), INT8_C( -21), INT8_C(  61),
                           INT8_C(   5), INT8_C(  96), INT8_C(  81), INT8_C(   9),
                           INT8_C(-121), INT8_C(  39), INT8_C( -77), INT8_C( 111),
                           INT8_C( -66), INT8_C(  56), INT8_C( -30), INT8_C( -47),
                           INT8_C(  60), INT8_C( -59), INT8_C(  45), INT8_C(  86),
                           INT8_C( -57), INT8_C( -53), INT8_C( 106), INT8_C(  23),
                           INT8_C( -11), INT8_C(  82), INT8_C(  92), INT8_C( -19),
                           INT8_C(  64), INT8_C( 103), INT8_C( -89), INT8_C( -98),
                           INT8_C( -33), INT8_C(  99), INT8_C(  24), INT8_C(  46),
                           INT8_C(  12), INT8_C(  -4), INT8_C( -89), INT8_C( 107),
                           INT8_C( -35), INT8_C(  71), INT8_C(  43), INT8_C( 111),
                           INT8_C( -31), INT8_C( -90), INT8_C(  -5), INT8_C(  22),
                           INT8_C(  27), INT8_C( -27), INT8_C( -44), INT8_C( 119),
                           INT8_C( -73), INT8_C(  55), INT8_C(-108), INT8_C( -98),
                           INT8_C( 121), INT8_C(-114), INT8_C(  51), INT8_C(  28),
                           INT8_C(  69), INT8_C(  37), INT8_C(  17), INT8_C( -56)),
      easysimd_mm512_set_epi8(INT8_C( -65), INT8_C(  52), INT8_C(  -9), INT8_C(  14),
                           INT8_C(  23), INT8_C(-117), INT8_C( 110), INT8_C(  60),
                           INT8_C(  81), INT8_C(-125), INT8_C( 120), INT8_C(  71),
                           INT8_C( 100), INT8_C( -41), INT8_C( 116), INT8_C(-126),
                           INT8_C(-101), INT8_C(  34), INT8_C( -39), INT8_C(  84),
                           INT8_C( -70), INT8_C(-112), INT8_C(  -6), INT8_C( -59),
                           INT8_C(  82), INT8_C(  94), INT8_C(  76), INT8_C( 114),
                           INT8_C( 127), INT8_C( -86), INT8_C( -89), INT8_C(  93),
                           INT8_C(   1), INT8_C( 119), INT8_C( -87), INT8_C( -11),
                           INT8_C( -62), INT8_C( -56), INT8_C( -72), INT8_C( -84),
                           INT8_C(  46), INT8_C(  34), INT8_C( -72), INT8_C(-127),
                           INT8_C(  99), INT8_C( 102), INT8_C(  60), INT8_C(  57),
                           INT8_C( -62), INT8_C( -50), INT8_C( -30), INT8_C(  99),
                           INT8_C( -14), INT8_C(  93), INT8_C( -12), INT8_C(-120),
                           INT8_C( 126), INT8_C(   7), INT8_C(  84), INT8_C(  10),
                           INT8_C( -53), INT8_C( 100), INT8_C( 113), INT8_C( -25)),
      UINT64_C(12369770630542733031) },
    { easysimd_mm512_set_epi8(INT8_C(  56), INT8_C(  -4), INT8_C(  -4), INT8_C( -38),
                           INT8_C(  25), INT8_C( 103), INT8_C( -62), INT8_C(  92),
                           INT8_C(  31), INT8_C( -43), INT8_C( -18), INT8_C(  90),
                           INT8_C( -11), INT8_C(  80), INT8_C(-102), INT8_C( -32),
                           INT8_C( -48), INT8_C(  -4), INT8_C(-124), INT8_C(-111),
                           INT8_C( 122), INT8_C(  50), INT8_C(  20), INT8_C(  73),
                           INT8_C(  10), INT8_C( -61), INT8_C(  30), INT8_C(  39),
                           INT8_C(  47), INT8_C(  -1), INT8_C( -24), INT8_C(-123),
                           INT8_C(  91), INT8_C(   9), INT8_C( -27), INT8_C(  35),
                           INT8_C(   0), INT8_C( -62), INT8_C( -75), INT8_C( 104),
                           INT8_C(  44), INT8_C(  16), INT8_C(-126), INT8_C(  97),
                           INT8_C( -20), INT8_C( -56), INT8_C( -70), INT8_C( -25),
                           INT8_C(  96), INT8_C( 112), INT8_C( 102), INT8_C(   2),
                           INT8_C( -70), INT8_C( -47), INT8_C(  77), INT8_C( -51),
                           INT8_C( -63), INT8_C(  15), INT8_C( -89), INT8_C( -32),
                           INT8_C(  72), INT8_C( 102), INT8_C( -45), INT8_C(  80)),
      easysimd_mm512_set_epi8(INT8_C(  46), INT8_C( -69), INT8_C(  -3), INT8_C(   7),
                           INT8_C( -53), INT8_C( -75), INT8_C(-113), INT8_C(  66),
                           INT8_C(-117), INT8_C(  -4), INT8_C( -75), INT8_C(-100),
                           INT8_C( -10), INT8_C(-123), INT8_C( 108), INT8_C(  99),
                           INT8_C(  34), INT8_C(  28), INT8_C(  20), INT8_C(  -3),
                           INT8_C( -41), INT8_C(  37), INT8_C( 126), INT8_C(  37),
                           INT8_C(  86), INT8_C(-126), INT8_C(  57), INT8_C( -79),
                           INT8_C(-102), INT8_C( -22), INT8_C( -65), INT8_C( -86),
                           INT8_C( -13), INT8_C(  63), INT8_C( -35), INT8_C(  12),
                           INT8_C(  52), INT8_C( 115), INT8_C(  54), INT8_C(-115),
                           INT8_C(  85), INT8_C(-123), INT8_C(  -3), INT8_C( -67),
                           INT8_C(  75), INT8_C( -64), INT8_C(  52), INT8_C( 126),
                           INT8_C(  62), INT8_C( -88), INT8_C(  10), INT8_C(  75),
                           INT8_C( -91), INT8_C(  62), INT8_C(  97), INT8_C(  54),
                           INT8_C( -80), INT8_C( -98), INT8_C( -77), INT8_C(  80),
                           INT8_C(  14), INT8_C( 105), INT8_C( -43), INT8_C(  19)),
      UINT64_C(3480141911697332022) },
    { easysimd_mm512_set_epi8(INT8_C(  30), INT8_C(  13), INT8_C(  72), INT8_C( 124),
                           INT8_C(   6), INT8_C( -85), INT8_C( -61), INT8_C( -45),
                           INT8_C(-117), INT8_C(  64), INT8_C(-110), INT8_C(  -2),
                           INT8_C(  83), INT8_C(  64), INT8_C(  94), INT8_C(  33),
                           INT8_C(  87), INT8_C( -89), INT8_C( -85), INT8_C( -82),
                           INT8_C(  61), INT8_C( -90), INT8_C(  27), INT8_C(-115),
                           INT8_C( -84), INT8_C( -79), INT8_C( -56), INT8_C(  66),
                           INT8_C(  57), INT8_C(  48), INT8_C(  34), INT8_C(  90),
                           INT8_C(  51), INT8_C(  19), INT8_C( 105), INT8_C( -57),
                           INT8_C( -67), INT8_C(-104), INT8_C(   1), INT8_C(-128),
                           INT8_C(  95), INT8_C(  94), INT8_C( -45), INT8_C( -86),
                           INT8_C( 116), INT8_C(  95), INT8_C(  64), INT8_C(-106),
                           INT8_C(  32), INT8_C( -60), INT8_C(-105), INT8_C(  23),
                           INT8_C( 115), INT8_C( -71), INT8_C( -22), INT8_C( -60),
                           INT8_C( -51), INT8_C(  42), INT8_C( -96), INT8_C(  -3),
                           INT8_C(  39), INT8_C( -17), INT8_C(  55), INT8_C(-100)),
      easysimd_mm512_set_epi8(INT8_C(  33), INT8_C( -73), INT8_C(  56), INT8_C(-105),
                           INT8_C( 103), INT8_C(-109), INT8_C(  18), INT8_C( -30),
                           INT8_C(  97), INT8_C(  18), INT8_C( 119), INT8_C( -24),
                           INT8_C( 104), INT8_C(  64), INT8_C( -85), INT8_C( -21),
                           INT8_C(  18), INT8_C(-115), INT8_C(  98), INT8_C(  20),
                           INT8_C(  51), INT8_C(  30), INT8_C( -90), INT8_C( -24),
                           INT8_C( -99), INT8_C( -91), INT8_C(  11), INT8_C( -84),
                           INT8_C(  56), INT8_C(  98), INT8_C( -91), INT8_C( -93),
                           INT8_C(   0), INT8_C( 119), INT8_C( 113), INT8_C( 107),
                           INT8_C( 103), INT8_C(  96), INT8_C( -68), INT8_C(-127),
                           INT8_C( -84), INT8_C(  51), INT8_C( 103), INT8_C(  56),
                           INT8_C( -12), INT8_C(  53), INT8_C(  32), INT8_C(-117),
                           INT8_C( -90), INT8_C( -10), INT8_C(  29), INT8_C( 115),
                           INT8_C( 127), INT8_C(  10), INT8_C( -23), INT8_C( 108),
                           INT8_C(  92), INT8_C(  -2), INT8_C( -94), INT8_C(  83),
                           INT8_C( -55), INT8_C( 115), INT8_C( -67), INT8_C( -25)),
      UINT64_C(10063351798194798005) },
    { easysimd_mm512_set_epi8(INT8_C(  30), INT8_C( -62), INT8_C(  -2), INT8_C( 110),
                           INT8_C( -99), INT8_C(   0), INT8_C( 114), INT8_C(-101),
                           INT8_C( -98), INT8_C(-101), INT8_C( 110), INT8_C( 127),
                           INT8_C( -57), INT8_C( 112), INT8_C(   1), INT8_C( -68),
                           INT8_C( -53), INT8_C(  40), INT8_C(  60), INT8_C(  -7),
                           INT8_C( 119), INT8_C( -84), INT8_C(  59), INT8_C(  41),
                           INT8_C(  94), INT8_C(  56), INT8_C( -73), INT8_C(-113),
                           INT8_C(-101), INT8_C(  70), INT8_C(  -5), INT8_C(-102),
                           INT8_C( -24), INT8_C( -88), INT8_C( -82), INT8_C( -98),
                           INT8_C( 103), INT8_C( 114), INT8_C( -24), INT8_C(  -1),
                           INT8_C(  33), INT8_C( -48), INT8_C(  56), INT8_C( -37),
                           INT8_C( -82), INT8_C( 126), INT8_C(  -6), INT8_C( 117),
                           INT8_C(-112), INT8_C( -39), INT8_C(  59), INT8_C(  25),
                           INT8_C( -19), INT8_C(  35), INT8_C( -12), INT8_C( -23),
                           INT8_C(-128), INT8_C( -16), INT8_C( -33), INT8_C( -91),
                           INT8_C( -40), INT8_C( -21), INT8_C( -38), INT8_C(  79)),
      easysimd_mm512_set_epi8(INT8_C(  42), INT8_C( 118), INT8_C(  39), INT8_C(  16),
                           INT8_C(  45), INT8_C( -16), INT8_C(  38), INT8_C(  64),
                           INT8_C( -47), INT8_C(  99), INT8_C(  10), INT8_C( -26),
                           INT8_C( -31), INT8_C( -27), INT8_C(  -3), INT8_C( 108),
                           INT8_C( -96), INT8_C( -87), INT8_C( -84), INT8_C(   9),
                           INT8_C(  24), INT8_C(  14), INT8_C( 123), INT8_C( -80),
                           INT8_C(   1), INT8_C(  70), INT8_C(  95), INT8_C(   7),
                           INT8_C( -79), INT8_C( -64), INT8_C(  81), INT8_C( -84),
                           INT8_C(-123), INT8_C( -69), INT8_C( -73), INT8_C(  98),
                           INT8_C( -88), INT8_C( 100), INT8_C( -60), INT8_C( 125),
                           INT8_C(  62), INT8_C( -77), INT8_C(  32), INT8_C(  40),
                           INT8_C(  68), INT8_C(  75), INT8_C( 112), INT8_C(-128),
                           INT8_C( -18), INT8_C(  -3), INT8_C(   2), INT8_C(  28),
                           INT8_C( -88), INT8_C(  15), INT8_C( -42), INT8_C(  -1),
                           INT8_C(  10), INT8_C( -70), INT8_C( -90), INT8_C( -93),
                           INT8_C(  65), INT8_C( 119), INT8_C( 107), INT8_C(  29)),
      UINT64_C(16846020600598811022) },
    { easysimd_mm512_set_epi8(INT8_C(  21), INT8_C( 119), INT8_C(  54), INT8_C( -98),
                           INT8_C(-127), INT8_C(  24), INT8_C( -58), INT8_C(-124),
                           INT8_C(  58), INT8_C(   6), INT8_C(  90), INT8_C( -82),
                           INT8_C(  81), INT8_C(-114), INT8_C( -76), INT8_C( -79),
                           INT8_C(-107), INT8_C( 107), INT8_C( -44), INT8_C(  36),
                           INT8_C(  -3), INT8_C( -89), INT8_C( 118), INT8_C(-104),
                           INT8_C(  90), INT8_C( 122), INT8_C(   4), INT8_C(  68),
                           INT8_C(  34), INT8_C(  55), INT8_C(  65), INT8_C(  86),
                           INT8_C(  74), INT8_C( -50), INT8_C(-117), INT8_C(   7),
                           INT8_C(  11), INT8_C(  -4), INT8_C(  -3), INT8_C( 109),
                           INT8_C( -44), INT8_C( -96), INT8_C(  98), INT8_C(  87),
                           INT8_C( -59), INT8_C(  95), INT8_C( -16), INT8_C( -64),
                           INT8_C( -50), INT8_C(  -5), INT8_C( -97), INT8_C( -47),
                           INT8_C( -88), INT8_C(  77), INT8_C( -27), INT8_C( -13),
                           INT8_C( -76), INT8_C( -43), INT8_C( 104), INT8_C(  53),
                           INT8_C(   4), INT8_C( -45), INT8_C(  81), INT8_C( 115)),
      easysimd_mm512_set_epi8(INT8_C(  58), INT8_C(  41), INT8_C(  85), INT8_C( -51),
                           INT8_C(   1), INT8_C(  51), INT8_C(  56), INT8_C(-109),
                           INT8_C( 109), INT8_C( 112), INT8_C(  72), INT8_C( -55),
                           INT8_C( -35), INT8_C( -66), INT8_C( -30), INT8_C( -94),
                           INT8_C(  71), INT8_C(  55), INT8_C( 100), INT8_C(  34),
                           INT8_C(  17), INT8_C( 115), INT8_C( 127), INT8_C(  32),
                           INT8_C( 101), INT8_C(  91), INT8_C(  97), INT8_C( -40),
                           INT8_C(  45), INT8_C( -66), INT8_C( -66), INT8_C(  34),
                           INT8_C( -55), INT8_C(  18), INT8_C( -74), INT8_C(  -1),
                           INT8_C(  33), INT8_C( -59), INT8_C(  16), INT8_C( -80),
                           INT8_C(  -4), INT8_C(  84), INT8_C(  30), INT8_C( -62),
                           INT8_C(-115), INT8_C(  37), INT8_C(  41), INT8_C(  57),
                           INT8_C( -41), INT8_C( -67), INT8_C( -77), INT8_C(  48),
                           INT8_C( 113), INT8_C(  84), INT8_C(  44), INT8_C(  98),
                           INT8_C( -38), INT8_C(   2), INT8_C(-122), INT8_C( -26),
                           INT8_C(  77), INT8_C(  90), INT8_C(  30), INT8_C( 127)),
      UINT64_C(13823429244140896205) },
    { easysimd_mm512_set_epi8(INT8_C( -22), INT8_C( 124), INT8_C(  34), INT8_C( -16),
                           INT8_C(   8), INT8_C( 122), INT8_C(  46), INT8_C(-121),
                           INT8_C(  29), INT8_C(  20), INT8_C(  -3), INT8_C( -93),
                           INT8_C(  62), INT8_C( 115), INT8_C(  28), INT8_C( -90),
                           INT8_C(  27), INT8_C(  21), INT8_C( -60), INT8_C(  15),
                           INT8_C(  72), INT8_C(-113), INT8_C(   5), INT8_C( -77),
                           INT8_C(  42), INT8_C( -54), INT8_C( -42), INT8_C(-118),
                           INT8_C( -22), INT8_C(  63), INT8_C( -17), INT8_C( -10),
                           INT8_C( -67), INT8_C(  58), INT8_C( -23), INT8_C(-108),
                           INT8_C(-119), INT8_C( 102), INT8_C(  88), INT8_C( 112),
                           INT8_C(-126), INT8_C(-115), INT8_C(  -3), INT8_C(  -3),
                           INT8_C( 115), INT8_C( -28), INT8_C(  73), INT8_C(  86),
                           INT8_C(  27), INT8_C( -49), INT8_C(  40), INT8_C(   0),
                           INT8_C(  84), INT8_C(-105), INT8_C( -76), INT8_C( -85),
                           INT8_C(  88), INT8_C( 123), INT8_C(  42), INT8_C(  35),
                           INT8_C(  67), INT8_C( -90), INT8_C(  86), INT8_C( -23)),
      easysimd_mm512_set_epi8(INT8_C(-115), INT8_C(  -9), INT8_C(  73), INT8_C(  35),
                           INT8_C(-126), INT8_C(-117), INT8_C( -35), INT8_C(  40),
                           INT8_C(-119), INT8_C( -28), INT8_C( -11), INT8_C( -21),
                           INT8_C( -21), INT8_C(  36), INT8_C( -54), INT8_C( -56),
                           INT8_C(  78), INT8_C(   7), INT8_C( -48), INT8_C( -59),
                           INT8_C( 121), INT8_C(  74), INT8_C( -58), INT8_C( 102),
                           INT8_C(  91), INT8_C( 126), INT8_C( -48), INT8_C( 121),
                           INT8_C( -54), INT8_C(  92), INT8_C( -18), INT8_C( 115),
                           INT8_C( -82), INT8_C(-102), INT8_C( -44), INT8_C(  35),
                           INT8_C(  20), INT8_C(-107), INT8_C(  72), INT8_C(  -9),
                           INT8_C( -37), INT8_C( -54), INT8_C( -20), INT8_C( -31),
                           INT8_C(  67), INT8_C(  96), INT8_C( -94), INT8_C(  -5),
                           INT8_C( -22), INT8_C(  67), INT8_C(  22), INT8_C( -62),
                           INT8_C( -72), INT8_C( -76), INT8_C( -32), INT8_C(  86),
                           INT8_C(  -8), INT8_C(  57), INT8_C( -71), INT8_C( -16),
                           INT8_C(-124), INT8_C( -88), INT8_C(  49), INT8_C(  35)),
      UINT64_C(3535798313217705733) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_cmplt_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmplt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t   arr_a[32];
    int16_t   arr_b[32];
    easysimd__mmask32  r;
  } test_vec[8] = {
    { {  INT16_C(  9899), -INT16_C(  4424), -INT16_C(  8946),  INT16_C( 26502),  INT16_C( 26824),  INT16_C(  1832),  INT16_C(   706), -INT16_C( 18820),
        -INT16_C(  4476), -INT16_C( 30725),  INT16_C( 30934),  INT16_C( 16088),  INT16_C( 16989), -INT16_C( 25768), -INT16_C(  2070), -INT16_C( 27016),
         INT16_C( 12573),  INT16_C( 11140),  INT16_C(  2830), -INT16_C( 10605), -INT16_C( 17549),  INT16_C( 13789),  INT16_C( 22973),  INT16_C( 17132),
        -INT16_C(  6328),  INT16_C(  7881), -INT16_C( 24225), -INT16_C( 17315), -INT16_C( 18973), -INT16_C( 12712), -INT16_C( 12116), -INT16_C( 13980) },
      { -INT16_C(  6143),  INT16_C(  4341), -INT16_C( 30477),  INT16_C( 26342), -INT16_C( 15293),  INT16_C(   156), -INT16_C( 30691),  INT16_C( 25922),
         INT16_C(  2927), -INT16_C( 12668), -INT16_C(  7763), -INT16_C( 28533), -INT16_C(  7274),  INT16_C( 17246), -INT16_C( 15693), -INT16_C( 19188),
         INT16_C(   427), -INT16_C( 24891), -INT16_C( 21623), -INT16_C( 13307), -INT16_C( 24209), -INT16_C( 29235),  INT16_C(  3881), -INT16_C( 26382),
         INT16_C( 30235), -INT16_C( 14234), -INT16_C(  3753), -INT16_C(  4520), -INT16_C( 18476), -INT16_C( 30671),  INT16_C( 15737),  INT16_C(  9277) },
      UINT32_C(3707806594) },
    { {  INT16_C(   575), -INT16_C( 14141), -INT16_C( 14163),  INT16_C(  7573),  INT16_C( 25193), -INT16_C( 27990), -INT16_C( 25487), -INT16_C( 29654),
        -INT16_C( 28653),  INT16_C( 27220), -INT16_C( 21118),  INT16_C( 22104), -INT16_C( 30364), -INT16_C(  8738),  INT16_C(  7111),  INT16_C(  1538),
        -INT16_C( 15075), -INT16_C( 13362),  INT16_C( 25485), -INT16_C(  2328), -INT16_C( 27963),  INT16_C( 14216), -INT16_C( 19922),  INT16_C( 16835),
         INT16_C(  6210), -INT16_C( 15188),  INT16_C(  1221),  INT16_C( 10523), -INT16_C(  1650),  INT16_C( 21766),  INT16_C(  2069),  INT16_C( 12891) },
      {  INT16_C( 10701),  INT16_C( 23293), -INT16_C(  6771),  INT16_C( 21072), -INT16_C( 10121), -INT16_C( 22903),  INT16_C( 19850), -INT16_C( 12825),
        -INT16_C( 27803),  INT16_C( 10897), -INT16_C( 21352),  INT16_C(  9811),  INT16_C( 22950), -INT16_C( 17541), -INT16_C( 10654),  INT16_C( 12269),
        -INT16_C(  5121), -INT16_C( 29558), -INT16_C(  9520),  INT16_C( 18655),  INT16_C( 26803),  INT16_C( 15854), -INT16_C( 10827),  INT16_C(  6666),
        -INT16_C( 25495),  INT16_C(   324), -INT16_C( 26808), -INT16_C(  4569), -INT16_C( 23823),  INT16_C( 21417), -INT16_C( 26760),  INT16_C( 30594) },
      UINT32_C(2189005295) },
    { {  INT16_C(  3202),  INT16_C( 20996), -INT16_C(  7193), -INT16_C( 25958), -INT16_C( 30645),  INT16_C(   471), -INT16_C(  7586), -INT16_C( 14565),
         INT16_C( 24702), -INT16_C( 14648), -INT16_C(  4105), -INT16_C(  5963),  INT16_C( 24209),  INT16_C(  2363), -INT16_C( 16651),  INT16_C( 30592),
        -INT16_C( 31542), -INT16_C( 20022),  INT16_C( 25703), -INT16_C( 19637),  INT16_C(  9197),  INT16_C( 19380), -INT16_C( 12539), -INT16_C( 31982),
        -INT16_C(  9681),  INT16_C( 10057), -INT16_C(   311),  INT16_C( 23055),  INT16_C( 19293),  INT16_C( 21091), -INT16_C(  7415), -INT16_C( 11318) },
      { -INT16_C( 27544), -INT16_C( 12411), -INT16_C( 12040), -INT16_C(  6782),  INT16_C( 14067), -INT16_C(  2000),  INT16_C( 16902),  INT16_C( 13691),
        -INT16_C( 15076), -INT16_C(  6820),  INT16_C( 27843),  INT16_C(  8255), -INT16_C( 23881), -INT16_C( 16269),  INT16_C( 15750), -INT16_C(  4461),
         INT16_C(  6353), -INT16_C( 13891),  INT16_C( 16617), -INT16_C(  9041), -INT16_C(  8330),  INT16_C( 31957),  INT16_C( 20514),  INT16_C( 16050),
         INT16_C(  3605), -INT16_C(  9948),  INT16_C( 25466),  INT16_C( 12793),  INT16_C( 27654), -INT16_C( 29455), -INT16_C( 31319),  INT16_C( 31354) },
      UINT32_C(2515226328) },
    { {  INT16_C( 14237), -INT16_C( 31164), -INT16_C(  3209), -INT16_C(  4509),  INT16_C( 14546), -INT16_C(  2966),  INT16_C(  7304), -INT16_C( 25037),
         INT16_C( 22315), -INT16_C( 23177),  INT16_C( 28858), -INT16_C( 16169), -INT16_C( 14115), -INT16_C( 31156), -INT16_C( 14771), -INT16_C(  5375),
         INT16_C( 17918),  INT16_C( 30065), -INT16_C( 11208),  INT16_C(  2659), -INT16_C( 12788), -INT16_C( 27137),  INT16_C( 13034),  INT16_C(  5427),
        -INT16_C( 21879),  INT16_C( 17339), -INT16_C( 28134), -INT16_C(  2300),  INT16_C( 20570), -INT16_C( 22402),  INT16_C( 32535),  INT16_C(  5523) },
      {  INT16_C(  1220), -INT16_C(   886), -INT16_C(  4391), -INT16_C(  6906),  INT16_C(  1468), -INT16_C( 22918), -INT16_C( 21193), -INT16_C( 16196),
         INT16_C( 30551),  INT16_C( 29188),  INT16_C(  2057),  INT16_C( 25449), -INT16_C(  6312),  INT16_C( 28427), -INT16_C( 24986),  INT16_C( 10884),
         INT16_C(  4003),  INT16_C( 31782),  INT16_C( 11773), -INT16_C( 18079), -INT16_C(  9166),  INT16_C( 27231),  INT16_C(  7049), -INT16_C(  7894),
         INT16_C( 11922), -INT16_C( 25773), -INT16_C( 17354), -INT16_C( 28673),  INT16_C(  2724),  INT16_C(  2814), -INT16_C( 31831),  INT16_C( 19509) },
      UINT32_C(2771827586) },
    { {  INT16_C( 23442), -INT16_C( 28728),  INT16_C( 10632), -INT16_C( 17592), -INT16_C( 22779), -INT16_C( 28891),  INT16_C( 20419),  INT16_C( 21872),
        -INT16_C( 15490), -INT16_C( 19215), -INT16_C(  3969),  INT16_C(  9027),  INT16_C( 17146), -INT16_C( 23762),  INT16_C( 25541),  INT16_C( 22511),
        -INT16_C( 18498),  INT16_C( 18406),  INT16_C( 12001), -INT16_C(  6654),  INT16_C( 10197), -INT16_C( 26507), -INT16_C(  6794), -INT16_C(  2834),
        -INT16_C(  8280),  INT16_C( 10409), -INT16_C(  4913), -INT16_C( 14005),  INT16_C( 31022), -INT16_C(  3219),  INT16_C( 23772), -INT16_C( 25782) },
      {  INT16_C( 12308), -INT16_C(  2590), -INT16_C(  7074),  INT16_C( 13531),  INT16_C( 20747), -INT16_C( 32308), -INT16_C( 17866), -INT16_C(  8330),
         INT16_C(  8089),  INT16_C( 26631),  INT16_C( 21003),  INT16_C( 14898), -INT16_C( 24628), -INT16_C( 22483),  INT16_C( 30971),  INT16_C(  3907),
         INT16_C(  9640),  INT16_C(  1796), -INT16_C(  8183),  INT16_C(  5179),  INT16_C(  1841),  INT16_C( 26518),  INT16_C(  3266),  INT16_C( 23366),
         INT16_C( 19755),  INT16_C( 14020), -INT16_C(  2400),  INT16_C( 27760), -INT16_C( 24939), -INT16_C( 28652),  INT16_C( 22550), -INT16_C( 16736) },
      UINT32_C(2414440218) },
    { { -INT16_C( 23427), -INT16_C( 30779),  INT16_C(   132), -INT16_C( 19045),  INT16_C( 12552), -INT16_C( 13795),  INT16_C( 25405),  INT16_C( 26661),
        -INT16_C(  5711),  INT16_C( 20895),  INT16_C(  4063),  INT16_C( 29885), -INT16_C( 11859), -INT16_C( 15611), -INT16_C( 23255), -INT16_C( 22654),
         INT16_C( 18249), -INT16_C( 12754), -INT16_C( 14008),  INT16_C( 20611), -INT16_C( 24325),  INT16_C( 14362),  INT16_C( 16132), -INT16_C( 19039),
         INT16_C( 16425),  INT16_C(  2054), -INT16_C( 15537), -INT16_C(   643), -INT16_C( 32108), -INT16_C( 16704),  INT16_C( 16935),  INT16_C( 28773) },
      { -INT16_C( 27766), -INT16_C( 11714), -INT16_C( 15780),  INT16_C( 22306),  INT16_C( 15458),  INT16_C( 26256),  INT16_C( 12667), -INT16_C( 23525),
         INT16_C(  8561), -INT16_C( 16211),  INT16_C( 10980),  INT16_C( 31165),  INT16_C( 32428), -INT16_C( 11465), -INT16_C( 25408),  INT16_C( 19011),
        -INT16_C( 32209), -INT16_C( 29924),  INT16_C( 15940), -INT16_C( 22813),  INT16_C( 29562), -INT16_C(  2547),  INT16_C( 10404),  INT16_C(  5530),
         INT16_C( 18250),  INT16_C( 11989), -INT16_C( 27791),  INT16_C(  7591), -INT16_C(  8687), -INT16_C( 11792),  INT16_C( 13434), -INT16_C( 22244) },
      UINT32_C( 999603514) },
    { {  INT16_C( 14518), -INT16_C(  1483),  INT16_C(  6263), -INT16_C(  3680), -INT16_C( 21109),  INT16_C( 12263), -INT16_C( 32042),  INT16_C(  8260),
         INT16_C(  6601),  INT16_C( 15182), -INT16_C(  2388), -INT16_C( 17064),  INT16_C( 18900),  INT16_C( 20367), -INT16_C( 21635),  INT16_C( 13304),
         INT16_C( 11747),  INT16_C( 23085), -INT16_C( 12987), -INT16_C( 12212),  INT16_C( 13179),  INT16_C( 20991),  INT16_C( 17333),  INT16_C( 32625),
        -INT16_C( 16547),  INT16_C(  2490),  INT16_C(  4789), -INT16_C( 30009),  INT16_C( 22107), -INT16_C( 10023), -INT16_C( 12031), -INT16_C(  7157) },
      {  INT16_C( 14591),  INT16_C( 17471), -INT16_C( 29946), -INT16_C( 32491),  INT16_C(  5310),  INT16_C( 29906),  INT16_C( 17240), -INT16_C( 18957),
        -INT16_C( 21246), -INT16_C( 18242), -INT16_C( 31297),  INT16_C(  6978),  INT16_C(  7131), -INT16_C(  8973), -INT16_C(    20), -INT16_C(  5183),
         INT16_C(    55),  INT16_C( 15664),  INT16_C( 17803),  INT16_C( 18878), -INT16_C( 28583), -INT16_C( 20035), -INT16_C( 20269), -INT16_C( 10650),
         INT16_C(  9565),  INT16_C(  7566), -INT16_C( 12118), -INT16_C( 31176),  INT16_C( 11243), -INT16_C( 10398),  INT16_C(  9002),  INT16_C( 25283) },
      UINT32_C(3272362099) },
    { { -INT16_C(  3293), -INT16_C( 20833),  INT16_C( 24120), -INT16_C( 28168), -INT16_C( 18962), -INT16_C( 15805), -INT16_C( 22170), -INT16_C( 15464),
         INT16_C(  9934),  INT16_C( 31200),  INT16_C(  6390), -INT16_C(  7681),  INT16_C( 24900),  INT16_C( 28344),  INT16_C( 31621), -INT16_C( 22320),
         INT16_C( 28782), -INT16_C( 22953),  INT16_C( 20430), -INT16_C( 17352),  INT16_C( 31492),  INT16_C( 27262),  INT16_C(  5668), -INT16_C(  3282),
         INT16_C(  3644),  INT16_C( 12908),  INT16_C( 27431),  INT16_C( 27411), -INT16_C( 13108),  INT16_C( 20953), -INT16_C( 21945), -INT16_C( 18694) },
      {  INT16_C( 20762), -INT16_C(  6052), -INT16_C( 27488), -INT16_C( 23388),  INT16_C(  8975),  INT16_C( 13327),  INT16_C( 15673),  INT16_C( 30247),
        -INT16_C( 27829),  INT16_C( 29352), -INT16_C( 17154), -INT16_C( 13603), -INT16_C( 18552), -INT16_C( 12516),  INT16_C(  5729),  INT16_C( 31621),
        -INT16_C(  7577),  INT16_C(  1891),  INT16_C(  1910), -INT16_C( 31061), -INT16_C( 17878),  INT16_C( 25786), -INT16_C(  7689),  INT16_C( 17370),
        -INT16_C( 32140),  INT16_C( 29365), -INT16_C( 27842), -INT16_C( 14788),  INT16_C( 22602), -INT16_C( 21610),  INT16_C(  7022), -INT16_C( 10970) },
      UINT32_C(3531768059) }
  };
  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512((void *)&test_vec[i].arr_a);
    easysimd__m512i b = easysimd_mm512_loadu_si512((void *)&test_vec[i].arr_b);
    easysimd__mmask32 r = test_vec[i].r;
    easysimd__mmask32 rk;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      rk = easysimd_mm512_cmplt_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmplt_epi16_mask");
    easysimd_assert_equal_mmask32(rk, r);
  }

  return 0;

#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 r = easysimd_mm512_cmplt_epi16_mask(a, b);

    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif

}

static int
test_easysimd_mm512_cmplt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int32_t a[16];
    const int32_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { {  INT32_C(  2112495169), -INT32_C(  1712505492),  INT32_C(  1695133573), -INT32_C(   890904325), -INT32_C(   264257920), -INT32_C(   737416202),  INT32_C(   421395668),  INT32_C(  1456679189),
        -INT32_C(  1261126329), -INT32_C(  2142387718),  INT32_C(  1810191984), -INT32_C(  1221146058),  INT32_C(  2141681032),  INT32_C(   894677856), -INT32_C(  1001492305), -INT32_C(   518315622) },
      { -INT32_C(   644485154),  INT32_C(   542761904),  INT32_C(  1888239161), -INT32_C(  1876442616), -INT32_C(  1760571849),  INT32_C(   818700929),  INT32_C(  1844779731),  INT32_C(   441323579),
        -INT32_C(  1342970625),  INT32_C(    13585607), -INT32_C(  1838130294),  INT32_C(  1411553053), -INT32_C(   403951002),  INT32_C(  1729673364),  INT32_C(   248778195),  INT32_C(   472392477) },
      UINT16_C(60006) },
    { { -INT32_C(   825550073), -INT32_C(   238118297),  INT32_C(   327368694),  INT32_C(  1013491414),  INT32_C(  1814320344), -INT32_C(   539739124),  INT32_C(  1726916681), -INT32_C(   763226421),
        -INT32_C(  1717547726), -INT32_C(   578064409), -INT32_C(  2064576850), -INT32_C(  1933485644), -INT32_C(  1174805075),  INT32_C(  1788464417),  INT32_C(  1104185205), -INT32_C(   820817251) },
      { -INT32_C(  2039892833), -INT32_C(   781913053), -INT32_C(  1219144446),  INT32_C(  1531123374),  INT32_C(   471088379),  INT32_C(  2139532553), -INT32_C(   759146956),  INT32_C(  1201787816),
        -INT32_C(  1429337465),  INT32_C(    24851198),  INT32_C(   901304711), -INT32_C(   477037593),  INT32_C(  1107272760), -INT32_C(  2034203054), -INT32_C(  2057797412), -INT32_C(   607323564) },
      UINT16_C(40872) },
    { {  INT32_C(    59152900),  INT32_C(  1392771532), -INT32_C(  1165443886), -INT32_C(   274917193),  INT32_C(   254909629), -INT32_C(     6950622), -INT32_C(   964366734), -INT32_C(   324972312),
        -INT32_C(  1209063445), -INT32_C(    83102935),  INT32_C(  1739953071),  INT32_C(  1767264940),  INT32_C(   293111791), -INT32_C(   351269511), -INT32_C(   474901253), -INT32_C(   791653659) },
      { -INT32_C(  1551384710),  INT32_C(  1654559411), -INT32_C(   758557658), -INT32_C(  1791287130),  INT32_C(   547861415), -INT32_C(  1140082752),  INT32_C(   832552012), -INT32_C(  1996394737),
        -INT32_C(   500398033),  INT32_C(  1095027227), -INT32_C(   988606946), -INT32_C(   715502034), -INT32_C(  1040842495),  INT32_C(    92078521), -INT32_C(   852091458), -INT32_C(  1151977588) },
      UINT16_C( 9046) },
    { { -INT32_C(   627211585),  INT32_C(  1796989516),  INT32_C(   506474224),  INT32_C(  2113112700),  INT32_C(  1161816460), -INT32_C(  1454719765),  INT32_C(  1719042265),  INT32_C(  1981926583),
        -INT32_C(  1705918642), -INT32_C(  1861915487),  INT32_C(   397358491),  INT32_C(  1284874944),  INT32_C(  2006045836),  INT32_C(  1780538256),  INT32_C(   315659867), -INT32_C(  1333202590) },
      {  INT32_C(  1363859888), -INT32_C(   505262010),  INT32_C(  1173918085), -INT32_C(  1064202956), -INT32_C(   231267743),  INT32_C(  1482446845),  INT32_C(  1332358381), -INT32_C(   822086883),
         INT32_C(   304040396),  INT32_C(   536019609), -INT32_C(   949687149), -INT32_C(   611846535),  INT32_C(   382582296),  INT32_C(    40773909),  INT32_C(  1917966677), -INT32_C(  1740615476) },
      UINT16_C(17189) },
    { {  INT32_C(   866869402), -INT32_C(   179134878),  INT32_C(    79542154), -INT32_C(   975223635),  INT32_C(   417049603),  INT32_C(   706431701), -INT32_C(   274961373),  INT32_C(  1468521917),
        -INT32_C(  1618333123),  INT32_C(  1536482769),  INT32_C(  1096765844), -INT32_C(  1727644010), -INT32_C(  1078861078),  INT32_C(  1340722220), -INT32_C(   163674567), -INT32_C(  1605515933) },
      { -INT32_C(   918562824),  INT32_C(  1210373044), -INT32_C(  1148615643), -INT32_C(  1386966846), -INT32_C(  1637087886),  INT32_C(   200103634),  INT32_C(  1057041372), -INT32_C(   388018448),
        -INT32_C(   625926619),  INT32_C(   371382001),  INT32_C(   483503194), -INT32_C(  1362549444), -INT32_C(    45337045),  INT32_C(  1745369484),  INT32_C(  1420233060),  INT32_C(  2084406871) },
      UINT16_C(63810) },
    { { -INT32_C(  1789464924),  INT32_C(   514619844),  INT32_C(  1631288613), -INT32_C(   837876573), -INT32_C(   959751366), -INT32_C(   131148908),  INT32_C(   860673500), -INT32_C(     5273253),
         INT32_C(  1016399480), -INT32_C(  1537523329),  INT32_C(  1627756222), -INT32_C(   718334822),  INT32_C(    60553839), -INT32_C(  1443116339), -INT32_C(    69449825),  INT32_C(  1241156817) },
      {  INT32_C(   293965714), -INT32_C(  1883905840),  INT32_C(   300989046),  INT32_C(  1038491854), -INT32_C(   398425830), -INT32_C(   342803637),  INT32_C(  1407610498), -INT32_C(  1935875846),
         INT32_C(  1084039792),  INT32_C(  2043630082), -INT32_C(   628441076), -INT32_C(    99127072),  INT32_C(  1038243825),  INT32_C(   338195602), -INT32_C(   597160222),  INT32_C(  1583875310) },
      UINT16_C(47961) },
    { {  INT32_C(   698287398),  INT32_C(  1671589463),  INT32_C(   238890030), -INT32_C(  1928768356),  INT32_C(  1036708779),  INT32_C(  1095889503), -INT32_C(   283264768), -INT32_C(   464681538),
        -INT32_C(   519181174), -INT32_C(  1991987365),  INT32_C(  2006483419), -INT32_C(  2147114539), -INT32_C(   323104884), -INT32_C(  1037234238), -INT32_C(  2035201080),  INT32_C(  1533804496) },
      {  INT32_C(  1178368235),  INT32_C(    63996200), -INT32_C(   663001086), -INT32_C(  1789362167),  INT32_C(   293672527), -INT32_C(   304828635), -INT32_C(   898333190),  INT32_C(  1881530244),
         INT32_C(  2142659159), -INT32_C(   444365085), -INT32_C(   138543378), -INT32_C(   846391682),  INT32_C(  1373572652), -INT32_C(  1220627523), -INT32_C(  1115508168), -INT32_C(   382883951) },
      UINT16_C(23433) },
    { { -INT32_C(   328670455),  INT32_C(  1490217834),  INT32_C(  1733332969), -INT32_C(   784999003), -INT32_C(  1457384213), -INT32_C(    10460729), -INT32_C(  1514348013), -INT32_C(  1819350646),
         INT32_C(   931198669), -INT32_C(   879799582), -INT32_C(  2026643487), -INT32_C(  1470601028),  INT32_C(  1129413500), -INT32_C(   280841764),  INT32_C(   513081236), -INT32_C(  1246682392) },
      { -INT32_C(    68407015),  INT32_C(  1707506819),  INT32_C(   418183515), -INT32_C(   574602143), -INT32_C(  1692397121),  INT32_C(  1451975618),  INT32_C(  1249124194),  INT32_C(  1526736450),
        -INT32_C(   615060393), -INT32_C(  1002431384),  INT32_C(  2010917910),  INT32_C(   810916976),  INT32_C(  1875604909),  INT32_C(   986076888), -INT32_C(  1199228298), -INT32_C(  1206680224) },
      UINT16_C(48363) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_cmplt_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 r = easysimd_mm512_cmplt_epi32_mask(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_cmplt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t   arr_a[8];
    int64_t   arr_b[8];
    easysimd__mmask8  r;
  } test_vec[8] = {
    { {  INT64_C(   56396756891636860),  INT64_C( 2678342955461388991), -INT64_C(   96738417780737442),  INT64_C( 7471831228619272609),
        -INT64_C( 3501720167931661350),  INT64_C( 1904522073738140302),  INT64_C( 7771997768272738802), -INT64_C( 9173734969730178977) },
      {  INT64_C( 4605299716042214411),  INT64_C( 2413918604426278569), -INT64_C( 7879652059463645841), -INT64_C( 5809207404539458502),
         INT64_C( 4740517550512228824), -INT64_C( 5262133699098029655),  INT64_C(  896099579474278314), -INT64_C( 5821941735006319903) },
      UINT8_C(145) },
    { {  INT64_C( 6516710517023243671), -INT64_C( 1922989505913940398), -INT64_C( 5332207163802501086), -INT64_C( 3242187956156633732),
        -INT64_C( 8180943500368266644), -INT64_C(  672593438613984887),  INT64_C( 1892332760138935735),  INT64_C( 6477987074236317143) },
      { -INT64_C( 1913205912748234584), -INT64_C( 4008111234578998890),  INT64_C(  971254925157026753), -INT64_C(  118775473716009195),
         INT64_C( 5994102801035350254), -INT64_C( 1168932395366360594), -INT64_C( 3914813673481233303), -INT64_C( 4610237063834711119) },
      UINT8_C( 28) },
    { {  INT64_C( 3028170732317894901),  INT64_C( 5417533355797890066), -INT64_C(  537453070088831731),  INT64_C( 8066514805462095200),
         INT64_C( 5139496908060032236),  INT64_C( 8870601903592340973), -INT64_C(  472377028214193358),  INT64_C( 6991307862407082012) },
      {  INT64_C( 4509755042672814105),  INT64_C(  235356858209868635),  INT64_C( 7330996294032064811),  INT64_C( 2104590620625266526),
        -INT64_C(  524105730237939926),  INT64_C( 1940777591959046172), -INT64_C( 5172777407146464121),  INT64_C( 6819788167085568642) },
      UINT8_C(  5) },
    { {  INT64_C( 3176890483626086272),  INT64_C( 7306818671284377693),  INT64_C( 7027423977051407732),  INT64_C( 6066503539472470686),
        -INT64_C(  155505438468469861),  INT64_C( 7828576632520175712), -INT64_C( 2002201573861958923),  INT64_C( 5114653947448401619) },
      {  INT64_C( 6165086996469476593),  INT64_C( 9087481782849558713), -INT64_C( 1568227836872665326), -INT64_C( 3848172120373595480),
         INT64_C(  883526123103423224),  INT64_C( 2362377759474177730),  INT64_C( 5562988586711128881),  INT64_C( 5136698764087510182) },
      UINT8_C(211) },
    { {  INT64_C( 5774027649390519490), -INT64_C( 1622312122281370035),  INT64_C( 5796524873022043519),  INT64_C(  774364360474562600),
        -INT64_C( 5144447319774595908),  INT64_C( 1855361665702573598),  INT64_C( 9115195851709215110),  INT64_C( 2527000315756433086) },
      { -INT64_C( 8012737780743523927), -INT64_C(  508020058230431424), -INT64_C( 7139709685503348713),  INT64_C( 4575575403835678966),
         INT64_C( 4028225737912020454), -INT64_C( 8400052831749321214),  INT64_C( 3724162693821076788), -INT64_C( 1326293566035635518) },
      UINT8_C( 26) },
    { {  INT64_C( 1092734298018344188),  INT64_C( 8320948142942206842),  INT64_C( 1582791324133060743), -INT64_C( 7891989252837374074),
         INT64_C( 7209894576805822193),  INT64_C(  909669418275082988), -INT64_C( 2403857493407030212),  INT64_C(  610208161033032015) },
      { -INT64_C( 7281453432588272220),  INT64_C( 5564025889551107495),  INT64_C( 2287969145823365257), -INT64_C( 5355207591166989424),
         INT64_C( 8124835440328160633),  INT64_C( 8420275461969184380), -INT64_C( 4266574381023119909), -INT64_C( 4393515104475558335) },
      UINT8_C( 60) },
    { { -INT64_C(  213732409195476274),  INT64_C( 6021423168832054321),  INT64_C( 4328744273055835673),  INT64_C( 7324424848561569176),
         INT64_C( 7736628057690968032),  INT64_C( 7311057760358557866), -INT64_C( 3011001327054260738),  INT64_C( 4197457497507284291) },
      { -INT64_C( 1255667279277113876),  INT64_C( 3572346479497452257),  INT64_C( 5769217827805021947),  INT64_C( 1421846820973889830),
         INT64_C( 3664183320246835112),  INT64_C( 7803034392655293157), -INT64_C(  246089577508998087), -INT64_C( 3262514699220172673) },
      UINT8_C(100) },
    { { -INT64_C( 6696122849876315505), -INT64_C( 7107238330328452735),  INT64_C( 4200302344059094136), -INT64_C( 8127420630365583560),
         INT64_C(  931543128873104267),  INT64_C( 2734631298302650610), -INT64_C( 8767174431434615200),  INT64_C( 6418506662468400260) },
      { -INT64_C(  745469967647735280),  INT64_C( 6992470648820639032), -INT64_C( 4545873395363119595),  INT64_C(   12200111535084036),
        -INT64_C( 7651261321025172007),  INT64_C( 8235258932490804711), -INT64_C( 7350226898391743486),  INT64_C( 8251041578364912273) },
      UINT8_C(235) }
  };
  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512((void *)&test_vec[i].arr_a);
    easysimd__m512i b = easysimd_mm512_loadu_si512((void *)&test_vec[i].arr_b);
    easysimd__mmask8 r = test_vec[i].r;
    easysimd__mmask8 rk;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      rk = easysimd_mm512_cmplt_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmplt_epi64_mask");
    easysimd_assert_equal_mmask8(rk, r);
  }

  return 0;

#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 r = easysimd_mm512_cmplt_epi64_mask(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif

}

static int
test_easysimd_mm512_mask_cmplt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask64 k;
    int8_t a[64];
    int8_t b[64];
    easysimd__mmask64 r;
  } test_vec[8] = {
    { UINT64_C( 8260198303694194944),
      {  INT8_C(  31),  INT8_C(  72),  INT8_C(  59),  INT8_C( 124), -INT8_C( 121), -INT8_C(  41), -INT8_C( 122),  INT8_C(  21),
        -INT8_C(  76),  INT8_C(  70),  INT8_C(  67),  INT8_C(  52), -INT8_C(  82), -INT8_C(  95), -INT8_C(  66),  INT8_C( 120),
        -INT8_C( 115),  INT8_C(  55),  INT8_C(  45),  INT8_C(  49), -INT8_C(  20),  INT8_C( 106), -INT8_C(  12), -INT8_C(  20),
         INT8_C( 115), -INT8_C(  18), -INT8_C(  83),  INT8_C(  11),  INT8_C(  12),  INT8_C(  79),  INT8_C( 125),  INT8_C(  44),
        -INT8_C( 105), -INT8_C(  72), -INT8_C(  88),  INT8_C(  30), -INT8_C( 112),  INT8_C(  46),  INT8_C(  51),  INT8_C(  68),
         INT8_C( 116),  INT8_C( 118),  INT8_C( 120),  INT8_C(  35),  INT8_C(  23),  INT8_C(  54), -INT8_C( 101), -INT8_C(  92),
         INT8_C( 109), -INT8_C(  56), -INT8_C(  42),  INT8_C(  89),  INT8_C(  50), -INT8_C(  54),  INT8_C(  69), -INT8_C(  90),
        -INT8_C(  71), -INT8_C(  14), -INT8_C(  79), -INT8_C(  59),  INT8_C(  65),  INT8_C(  46), -INT8_C(  15), -INT8_C(  40) },
      { -INT8_C(  26), -INT8_C( 103), -INT8_C(   9),  INT8_C( 118), -INT8_C(  56),  INT8_C(  42), -INT8_C(  70),  INT8_C(  60),
        -INT8_C(  95),  INT8_C(  51),  INT8_C(  95), -INT8_C(  72),  INT8_C( 105), -INT8_C(   5),  INT8_C(  93), -INT8_C(  41),
        -INT8_C(  61),  INT8_C(  51),  INT8_C(  48), -INT8_C(  10), -INT8_C(   3),  INT8_C( 118), -INT8_C( 100), -INT8_C(  74),
         INT8_C( 104),  INT8_C(  77),  INT8_C( 124), -INT8_C(  86),  INT8_C( 123),  INT8_C( 109), -INT8_C( 126),  INT8_C(  97),
         INT8_C(   7),  INT8_C( 121), -INT8_C(  40), -INT8_C(  49), -INT8_C(  92), -INT8_C( 110),  INT8_C(  11),  INT8_C(  69),
        -INT8_C(  59),  INT8_C( 107), -INT8_C(   3),  INT8_C(  47),  INT8_C( 102),  INT8_C(  90),  INT8_C(   6),  INT8_C(  41),
        -INT8_C( 115),  INT8_C(  54),  INT8_C(  31), -INT8_C( 117), -INT8_C(  84), -INT8_C(  69),  INT8_C(  65),  INT8_C(  21),
         INT8_C(   8), -INT8_C(  67), -INT8_C(  65), -INT8_C( 125),  INT8_C(  43),  INT8_C(  65), -INT8_C(  27),  INT8_C(  50) },
      UINT64_C( 2342461795155836928) },
    { UINT64_C( 1559384906012933563),
      {  INT8_C( 119), -INT8_C(  95),  INT8_C(  68), -INT8_C(  35), -INT8_C(   4),  INT8_C(  74),  INT8_C(   7), -INT8_C( 119),
             INT8_MIN,  INT8_C(  38),  INT8_C(  20),  INT8_C(  45), -INT8_C(  30),  INT8_C(  86),  INT8_C(  66), -INT8_C(  22),
         INT8_C(  19),  INT8_C(   1),  INT8_C( 110),  INT8_C(  62),  INT8_C(  66),  INT8_C(  83),  INT8_C( 112), -INT8_C(   3),
         INT8_C(  16),  INT8_C( 113),  INT8_C(  92),  INT8_C(  95),  INT8_C( 126),  INT8_C(   0),  INT8_C( 116), -INT8_C(  11),
        -INT8_C(  94), -INT8_C(  72), -INT8_C(  45), -INT8_C(  98),  INT8_C(   2), -INT8_C(  38),  INT8_C(  39), -INT8_C( 125),
         INT8_C(   0),  INT8_C(  60), -INT8_C(  80), -INT8_C(  30), -INT8_C( 110), -INT8_C(  14), -INT8_C(  51), -INT8_C(  91),
        -INT8_C(  13),  INT8_C(  59), -INT8_C(  28),  INT8_C(  53), -INT8_C( 114),  INT8_C(  84),  INT8_C(  51), -INT8_C(  98),
        -INT8_C(  58), -INT8_C( 113), -INT8_C(   3),  INT8_C(  68), -INT8_C( 112),  INT8_C( 114),  INT8_C(  57),  INT8_C(  50) },
      {  INT8_C(  42),  INT8_C(  12), -INT8_C(  48),  INT8_C(  45), -INT8_C(  26), -INT8_C(   9), -INT8_C(  80), -INT8_C(  25),
         INT8_C(  51),  INT8_C(  96), -INT8_C(  55), -INT8_C(  59),  INT8_C(  82), -INT8_C( 106),  INT8_C( 107),  INT8_C(  69),
        -INT8_C(  47),  INT8_C(  79),  INT8_C( 122),  INT8_C(  95), -INT8_C(  93), -INT8_C(  83), -INT8_C(   3),  INT8_C( 105),
         INT8_C(  61), -INT8_C(   5), -INT8_C(  83), -INT8_C(  51),  INT8_C( 109), -INT8_C(  25), -INT8_C(   1), -INT8_C( 105),
        -INT8_C(  13), -INT8_C(  49), -INT8_C(  60), -INT8_C(  38), -INT8_C(  58),  INT8_C( 116), -INT8_C(  63), -INT8_C(   6),
        -INT8_C(  44), -INT8_C( 118), -INT8_C(  65),  INT8_C(  38),  INT8_C(  33),  INT8_C(  42),  INT8_C( 107), -INT8_C(  14),
         INT8_C( 121), -INT8_C(  26),  INT8_C(  82),  INT8_C(  29), -INT8_C( 109),  INT8_C(  79), -INT8_C( 122), -INT8_C(  48),
         INT8_C(  74),  INT8_C(  52), -INT8_C(  99), -INT8_C(  73),  INT8_C(  27), -INT8_C( 100),  INT8_C(  79),  INT8_C(  14) },
      UINT64_C( 1262147036971569546) },
    { UINT64_C( 6641869952899355499),
      {  INT8_C(  52), -INT8_C(  21), -INT8_C( 125),  INT8_C(  85),  INT8_C(  22), -INT8_C(  18),  INT8_C(  71), -INT8_C( 113),
        -INT8_C(  44), -INT8_C( 103), -INT8_C(  84),  INT8_C( 104), -INT8_C(  23),  INT8_C(  51),  INT8_C(  56),  INT8_C(  51),
         INT8_C( 103), -INT8_C(  42), -INT8_C(  21), -INT8_C( 126),  INT8_C( 114),  INT8_C(  58), -INT8_C( 112), -INT8_C(  34),
         INT8_C(  77),  INT8_C( 121),  INT8_C(  16), -INT8_C(  43),  INT8_C(  34),  INT8_C(  60),  INT8_C(  50),  INT8_C(  86),
         INT8_C(  39), -INT8_C(  75), -INT8_C(  85),  INT8_C(  61), -INT8_C(  93), -INT8_C(  13), -INT8_C(  51),  INT8_C( 120),
        -INT8_C( 116),  INT8_C( 121), -INT8_C(  32),  INT8_C( 117), -INT8_C(  84),  INT8_C(  24), -INT8_C(  87),  INT8_C(  19),
        -INT8_C(  18), -INT8_C( 108), -INT8_C( 107),  INT8_C(  97), -INT8_C(  50),  INT8_C(  38),  INT8_C(  63),  INT8_C(  27),
        -INT8_C(  97),  INT8_C(  79), -INT8_C(  15), -INT8_C(  63), -INT8_C( 117),  INT8_C(  35),  INT8_C(  24), -INT8_C(  78) },
      { -INT8_C(  40), -INT8_C(  61), -INT8_C(  16),  INT8_C( 123), -INT8_C(  74), -INT8_C(  67), -INT8_C(  13),  INT8_C(  67),
         INT8_C(  54), -INT8_C(  45), -INT8_C(  72), -INT8_C(  29), -INT8_C(  20),  INT8_C(  97), -INT8_C(  10), -INT8_C(  38),
        -INT8_C(  11), -INT8_C( 116),  INT8_C(  59), -INT8_C(  61), -INT8_C(  78),  INT8_C( 122), -INT8_C(  33),  INT8_C(  81),
        -INT8_C(  55), -INT8_C(  48),  INT8_C(  18),  INT8_C(  84), -INT8_C(  13),  INT8_C(  42),  INT8_C(   7), -INT8_C(  53),
        -INT8_C(  18), -INT8_C(   9),  INT8_C(  70), -INT8_C(  92), -INT8_C(  76),  INT8_C(  58), -INT8_C(  25), -INT8_C(  22),
         INT8_C(  13), -INT8_C(  96), -INT8_C(  51), -INT8_C(   7),  INT8_C(   1), -INT8_C(  60), -INT8_C(  44), -INT8_C(   9),
         INT8_C(  80),  INT8_C(  15), -INT8_C(  70),  INT8_C(   2), -INT8_C( 118), -INT8_C( 103),  INT8_C(  83),  INT8_C(  83),
         INT8_C( 105),  INT8_C( 101), -INT8_C(  88),  INT8_C(  92), -INT8_C( 112), -INT8_C(  81),  INT8_C(  39),  INT8_C( 126) },
      UINT64_C( 6342195274771337992) },
    { UINT64_C(13061576537464270502),
      { -INT8_C(  86),  INT8_C(  18), -INT8_C(  81), -INT8_C(  85), -INT8_C(  42), -INT8_C( 125), -INT8_C(  94),  INT8_C(  38),
        -INT8_C( 110),  INT8_C(  93),  INT8_C(  40),  INT8_C(  28), -INT8_C(  10),  INT8_C( 123),  INT8_C( 112),  INT8_C(  96),
        -INT8_C(  32),  INT8_C(  24), -INT8_C(  68),  INT8_C( 112), -INT8_C(  57), -INT8_C(  28), -INT8_C(  18),  INT8_C( 109),
         INT8_C(  82),  INT8_C(  17), -INT8_C(  57), -INT8_C(   6),  INT8_C(  27),  INT8_C(  11), -INT8_C(  81), -INT8_C(  59),
         INT8_C(  29),  INT8_C(  94),  INT8_C( 112), -INT8_C(  13), -INT8_C(  31),  INT8_C(  19),  INT8_C(  25),  INT8_C( 116),
         INT8_C( 112),  INT8_C(  65), -INT8_C( 112),  INT8_C( 102), -INT8_C(  68),  INT8_C(   0), -INT8_C(  58), -INT8_C(  99),
         INT8_C(  24), -INT8_C( 125),  INT8_C(  13), -INT8_C(  33),  INT8_C( 103), -INT8_C(   4),  INT8_C(  76), -INT8_C(  71),
         INT8_C(  13),  INT8_C(  19), -INT8_C(  77),  INT8_C(  40),  INT8_C(  31),  INT8_C(  98), -INT8_C(  19),  INT8_C(  60) },
      { -INT8_C(  63),  INT8_C(  93),  INT8_C(  48), -INT8_C(  94),  INT8_C( 112),  INT8_C(  73),  INT8_C(  22), -INT8_C(  32),
        -INT8_C( 117), -INT8_C(  89),  INT8_C(  71),  INT8_C(  71), -INT8_C(  89),  INT8_C(  13), -INT8_C(  28), -INT8_C(  64),
        -INT8_C( 112), -INT8_C(  14), -INT8_C(  97), -INT8_C(   9), -INT8_C(  18), -INT8_C(  20), -INT8_C(  80), -INT8_C(   5),
        -INT8_C(   1),  INT8_C(  99),  INT8_C(  35),  INT8_C(  30), -INT8_C(  58),  INT8_C(  16),  INT8_C(  91), -INT8_C( 121),
         INT8_C( 109), -INT8_C( 117),  INT8_C(  41), -INT8_C(  34), -INT8_C(  44),  INT8_C(  64), -INT8_C(  66),  INT8_C(  95),
        -INT8_C(  25),  INT8_C(   5), -INT8_C(  89), -INT8_C( 114),  INT8_C(  19), -INT8_C( 117),  INT8_C(  78), -INT8_C(  93),
         INT8_C( 125), -INT8_C(  18), -INT8_C( 101),  INT8_C( 107), -INT8_C(  38),  INT8_C(  75),  INT8_C( 102), -INT8_C(  39),
        -INT8_C(  81), -INT8_C( 119), -INT8_C(   8),  INT8_C( 117), -INT8_C( 103),  INT8_C(  83), -INT8_C(   4),  INT8_C(   7) },
      UINT64_C(  306244913343761446) },
    { UINT64_C( 5481623353651701214),
      { -INT8_C(  87), -INT8_C(  71), -INT8_C(  37), -INT8_C(  68),  INT8_C(  68),  INT8_C(  41),  INT8_C(  95), -INT8_C(  62),
         INT8_C(  23), -INT8_C(   6),  INT8_C(  45), -INT8_C(  15),  INT8_C(  70), -INT8_C( 108), -INT8_C(  53), -INT8_C(  11),
         INT8_C(  29), -INT8_C(  61),  INT8_C( 106), -INT8_C(  73),  INT8_C(  22),  INT8_C( 102), -INT8_C(  66), -INT8_C(  12),
        -INT8_C( 117), -INT8_C(  93), -INT8_C(  90), -INT8_C(  15),  INT8_C(  70), -INT8_C(  72),  INT8_C(  61), -INT8_C(  17),
         INT8_C( 113),  INT8_C(  24), -INT8_C(  85), -INT8_C(  74),  INT8_C(  66),  INT8_C(  11),  INT8_C( 120),  INT8_C(  89),
         INT8_C(   5), -INT8_C(  91),  INT8_C(  75),  INT8_C(  75),  INT8_C(  57),  INT8_C(  22),  INT8_C(  64),  INT8_C(  87),
        -INT8_C(  39), -INT8_C(  86),  INT8_C(  14), -INT8_C(  17),  INT8_C(  16), -INT8_C(  52), -INT8_C(  29), -INT8_C( 100),
         INT8_C( 111), -INT8_C( 119), -INT8_C( 115), -INT8_C(  75),  INT8_C(  66), -INT8_C(  54), -INT8_C(  91), -INT8_C(  77) },
      { -INT8_C(  29),  INT8_C(  80),  INT8_C( 105),  INT8_C(  37),  INT8_C(  91), -INT8_C(  31),  INT8_C( 126),  INT8_C(  97),
        -INT8_C( 121), -INT8_C(  55), -INT8_C(  84), -INT8_C(  64), -INT8_C(  33), -INT8_C(  19),  INT8_C(  23), -INT8_C(  72),
        -INT8_C( 105),  INT8_C(  37), -INT8_C(  89), -INT8_C(  88), -INT8_C(  15), -INT8_C( 118),  INT8_C(  68),  INT8_C(  96),
         INT8_C(  20), -INT8_C(  47),  INT8_C(  22),  INT8_C(  86), -INT8_C( 101), -INT8_C(  69),  INT8_C(   9),  INT8_C( 126),
         INT8_C(  11),  INT8_C( 115), -INT8_C(  93),  INT8_C( 103),  INT8_C(  84),  INT8_C(  34), -INT8_C(  56), -INT8_C(  37),
        -INT8_C(  21),  INT8_C( 116), -INT8_C( 100), -INT8_C(  53),  INT8_C(  97), -INT8_C(  77), -INT8_C( 125), -INT8_C(   7),
        -INT8_C(  39),  INT8_C(  43), -INT8_C(  95), -INT8_C(  54), -INT8_C(  75), -INT8_C(  27),  INT8_C(  43), -INT8_C(  55),
        -INT8_C(  74),  INT8_C(  65),  INT8_C(  31),  INT8_C(  81), -INT8_C(   4),  INT8_C(  41), -INT8_C(  48),  INT8_C(   7) },
      UINT64_C( 5476942436028653790) },
    { UINT64_C( 9352910543753474972),
      { -INT8_C(  85),  INT8_C( 104),  INT8_C(  76),  INT8_C(  12),  INT8_C(  27), -INT8_C(  49),  INT8_C(   5), -INT8_C(  12),
        -INT8_C(   6), -INT8_C(  90), -INT8_C(  65), -INT8_C(  80), -INT8_C( 117), -INT8_C(  22),  INT8_C( 121),  INT8_C(  65),
         INT8_C(  43), -INT8_C( 103), -INT8_C( 109),  INT8_C(  39), -INT8_C(  62),  INT8_C(  99),  INT8_C(  46),  INT8_C(  94),
        -INT8_C(  42), -INT8_C(  99),  INT8_C(  78),  INT8_C( 108), -INT8_C(  45),  INT8_C(  26), -INT8_C(  19),  INT8_C( 126),
        -INT8_C( 126),  INT8_C(  57), -INT8_C( 117), -INT8_C(  98),  INT8_C(   8), -INT8_C( 112), -INT8_C( 110),  INT8_C(   3),
         INT8_C(  55),  INT8_C(  81), -INT8_C(  77), -INT8_C(  62),  INT8_C(  59),  INT8_C(  44),  INT8_C(   4),  INT8_C( 102),
        -INT8_C(  59), -INT8_C( 105), -INT8_C( 115), -INT8_C( 121), -INT8_C(   6), -INT8_C(  68), -INT8_C(  27), -INT8_C(  48),
         INT8_C(  89),  INT8_C(  52),  INT8_C(  60),  INT8_C(  44),  INT8_C(  78),  INT8_C(  41), -INT8_C(  85), -INT8_C(  47) },
      {  INT8_C(  98),  INT8_C(  54),  INT8_C( 111),  INT8_C( 107), -INT8_C(  58),  INT8_C(   1),  INT8_C( 110), -INT8_C(   3),
         INT8_C(  83),  INT8_C(  33), -INT8_C(  64), -INT8_C( 114),  INT8_C(  77), -INT8_C(  60), -INT8_C(  11),  INT8_C(  19),
         INT8_C(  91), -INT8_C( 126), -INT8_C( 102),  INT8_C(  85),  INT8_C(  62),      INT8_MIN,  INT8_C(  37), -INT8_C( 105),
        -INT8_C(  76),  INT8_C(  98), -INT8_C(  60),  INT8_C(   2), -INT8_C( 117),  INT8_C( 111), -INT8_C(  45), -INT8_C(  18),
        -INT8_C(  91),  INT8_C(  66),  INT8_C(  89),  INT8_C( 107),  INT8_C(  68), -INT8_C(  57),  INT8_C( 105), -INT8_C( 105),
        -INT8_C(  24),  INT8_C(  41),  INT8_C(  37),  INT8_C(  53), -INT8_C(  19),  INT8_C(  26),  INT8_C(  72),  INT8_C(  72),
        -INT8_C(  99), -INT8_C(  29), -INT8_C(  99), -INT8_C(  37),  INT8_C(  99), -INT8_C(  62),  INT8_C( 115),  INT8_C(  23),
         INT8_C(  36),  INT8_C(  55),  INT8_C(  25), -INT8_C(  80), -INT8_C(  90), -INT8_C(  19), -INT8_C(  98),  INT8_C(  75) },
      UINT64_C( 9280797420882236300) },
    { UINT64_C(11964410262908761903),
      {  INT8_C(  72),  INT8_C(  48), -INT8_C(  37),  INT8_C(  53),  INT8_C(  74),  INT8_C(  36),  INT8_C( 125), -INT8_C(  25),
         INT8_C(   7),  INT8_C(  26), -INT8_C(  61),  INT8_C( 106), -INT8_C(  35),  INT8_C(  54), -INT8_C( 127),  INT8_C(   1),
         INT8_C( 109), -INT8_C( 102), -INT8_C(  79),  INT8_C(  19), -INT8_C( 121),  INT8_C(  79),  INT8_C(  94), -INT8_C(  73),
         INT8_C(  70),  INT8_C(  20),  INT8_C(  42),  INT8_C(   4),  INT8_C(  52),  INT8_C(  53), -INT8_C(  86),  INT8_C( 124),
         INT8_C( 101), -INT8_C( 122), -INT8_C(  78), -INT8_C(  81), -INT8_C(  86),  INT8_C(  47), -INT8_C( 105), -INT8_C(  79),
         INT8_C(  74),  INT8_C(  90),  INT8_C(  27),  INT8_C(  39), -INT8_C( 112), -INT8_C( 100),  INT8_C(  40), -INT8_C(   3),
         INT8_C(  54), -INT8_C(  38),  INT8_C(  16), -INT8_C(  66),  INT8_C(  41),  INT8_C( 110),  INT8_C( 117),  INT8_C( 112),
        -INT8_C( 126), -INT8_C(  97),  INT8_C( 116), -INT8_C(  74), -INT8_C(  44),  INT8_C(  31),  INT8_C(  51),  INT8_C(  57) },
      { -INT8_C(  91), -INT8_C(  27), -INT8_C(  23),  INT8_C(  79),  INT8_C(  20),      INT8_MIN,  INT8_C(   0),  INT8_C(  94),
        -INT8_C(  38),  INT8_C(  27), -INT8_C( 123),  INT8_C( 106), -INT8_C(  73), -INT8_C(  82),  INT8_C( 103), -INT8_C(  19),
        -INT8_C( 120),  INT8_C( 119), -INT8_C(  85), -INT8_C(  79), -INT8_C(  27),  INT8_C(  32),  INT8_C(  33),  INT8_C( 103),
        -INT8_C(  64), -INT8_C( 106),  INT8_C(  30), -INT8_C( 108), -INT8_C(  75),  INT8_C(  81), -INT8_C(  50),  INT8_C(  90),
         INT8_C(  54), -INT8_C(  73), -INT8_C(  87),  INT8_C(  74),  INT8_C(  55), -INT8_C(  87), -INT8_C(  87),  INT8_C(  17),
        -INT8_C(  60),  INT8_C(  46),  INT8_C( 123),  INT8_C( 123), -INT8_C(  36), -INT8_C(  30),  INT8_C( 104),  INT8_C( 100),
         INT8_C(  89),  INT8_C(  20),  INT8_C(  22),  INT8_C(  62),  INT8_C(  52),  INT8_C(  55), -INT8_C(  91), -INT8_C(  12),
        -INT8_C(  51), -INT8_C(  61), -INT8_C( 119), -INT8_C( 126),  INT8_C(  20),  INT8_C(  87), -INT8_C(  36),  INT8_C(  74) },
      UINT64_C(11676176433282171404) },
    { UINT64_C(17462212973632521486),
      {  INT8_C( 108), -INT8_C(  47),  INT8_C( 109),  INT8_C(  73), -INT8_C(  77), -INT8_C(  42), -INT8_C(  83),  INT8_C(  12),
        -INT8_C(  22), -INT8_C(  61),  INT8_C(  74),  INT8_C(  30), -INT8_C(   5), -INT8_C(  17),  INT8_C(  19), -INT8_C(  56),
        -INT8_C(  77), -INT8_C( 100),  INT8_C(  75), -INT8_C(  57), -INT8_C(  13),  INT8_C(  39),  INT8_C(  18),  INT8_C(   1),
        -INT8_C(  83), -INT8_C(  89),  INT8_C(  70), -INT8_C(  37), -INT8_C(  27), -INT8_C( 100), -INT8_C(  50),  INT8_C(  81),
         INT8_C( 109),  INT8_C(  59), -INT8_C( 102),  INT8_C(  32),  INT8_C(  17),  INT8_C(  72),  INT8_C(  44), -INT8_C(   5),
         INT8_C(  11),  INT8_C( 118),  INT8_C(  26),  INT8_C(   6),  INT8_C( 101),  INT8_C(  45), -INT8_C(  49),  INT8_C(  24),
        -INT8_C(  55),  INT8_C(  26), -INT8_C(  32), -INT8_C(  68),  INT8_C(  65), -INT8_C(  14), -INT8_C(  67), -INT8_C(  18),
        -INT8_C( 103),  INT8_C(   3), -INT8_C(  54),  INT8_C( 126), -INT8_C(  97), -INT8_C( 104), -INT8_C(  49),  INT8_C(  12) },
      { -INT8_C(  45),  INT8_C( 106),  INT8_C(  44), -INT8_C(  27), -INT8_C(  78),  INT8_C(  88), -INT8_C(  32), -INT8_C(  67),
        -INT8_C(  50), -INT8_C(   6), -INT8_C(  60),  INT8_C(  51),  INT8_C(  39), -INT8_C( 109),  INT8_C(  76), -INT8_C(  16),
        -INT8_C(  83),  INT8_C(  44), -INT8_C(  84), -INT8_C(  18),  INT8_C(  30),  INT8_C( 105), -INT8_C(  35), -INT8_C(  73),
         INT8_C( 108), -INT8_C(  89),  INT8_C(  53),  INT8_C(  11),  INT8_C(  63),  INT8_C(   4),  INT8_C(  23),  INT8_C(  18),
         INT8_C( 110),  INT8_C(  67), -INT8_C(   9),  INT8_C(  32), -INT8_C( 101), -INT8_C(  40), -INT8_C(  34),  INT8_C( 105),
        -INT8_C(  46), -INT8_C(  94), -INT8_C(  99), -INT8_C(   6),  INT8_C(  53), -INT8_C(  23), -INT8_C(  22), -INT8_C(  30),
         INT8_C(  21), -INT8_C( 105), -INT8_C(  48),  INT8_C(  51),  INT8_C(   0), -INT8_C(  83), -INT8_C(  22),  INT8_C( 109),
         INT8_C(  84),  INT8_C(  31),  INT8_C( 120), -INT8_C( 109),  INT8_C(  35), -INT8_C( 112), -INT8_C(  90), -INT8_C( 110) },
      UINT64_C( 1315051118053588994) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epi8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i b = easysimd_test_x86_random_i8x64();
    easysimd__mmask64 r = easysimd_mm512_mask_cmplt_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask64(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmplt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    int16_t a[32];
    int16_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C(3733490575),
      {  INT16_C( 31855),  INT16_C( 26269), -INT16_C( 12048), -INT16_C( 27847),  INT16_C( 14177),  INT16_C(  6403),  INT16_C( 14372), -INT16_C( 17717),
         INT16_C(  6218),  INT16_C(   796),  INT16_C( 17882), -INT16_C( 17003),  INT16_C( 22357), -INT16_C(  6957),  INT16_C( 23522),  INT16_C( 20930),
         INT16_C( 24535), -INT16_C( 14409), -INT16_C(  4048), -INT16_C( 28325),  INT16_C( 24103),  INT16_C( 19626),  INT16_C( 30102), -INT16_C(  8186),
         INT16_C(  8845),  INT16_C( 26595),  INT16_C( 30823), -INT16_C( 17116), -INT16_C(  2097), -INT16_C( 20063),  INT16_C( 25683),  INT16_C( 10754) },
      { -INT16_C( 17981), -INT16_C(  3086),  INT16_C( 19882), -INT16_C( 11900),  INT16_C( 12203),  INT16_C( 16669),  INT16_C(  9124),  INT16_C( 12833),
         INT16_C(  1094), -INT16_C( 21095), -INT16_C( 16772),  INT16_C( 19562),  INT16_C(  3253),  INT16_C(  2301),  INT16_C(   112),  INT16_C( 13107),
         INT16_C(  9657),  INT16_C( 25383), -INT16_C( 21646),  INT16_C(  7477),  INT16_C( 21210),  INT16_C( 32606),  INT16_C( 32630), -INT16_C( 17231),
         INT16_C( 19075), -INT16_C(   151), -INT16_C( 11256), -INT16_C( 16821),  INT16_C( 18912),  INT16_C( 20678), -INT16_C(  1719),  INT16_C(   643) },
      UINT32_C( 403179660) },
    { UINT32_C(2422647326),
      { -INT16_C( 25770),  INT16_C( 12461),  INT16_C(  3053),  INT16_C( 25519),  INT16_C( 24714),  INT16_C(  3359), -INT16_C( 30293), -INT16_C( 19699),
         INT16_C( 22621),  INT16_C( 15729),  INT16_C( 14497), -INT16_C(  5491),  INT16_C(  4145),  INT16_C( 20717),  INT16_C( 21435),  INT16_C(  4576),
        -INT16_C( 28946), -INT16_C(  9407), -INT16_C(  3687),  INT16_C(  9279),  INT16_C( 24145), -INT16_C(   975),  INT16_C( 16103),  INT16_C( 17584),
         INT16_C(  8599),  INT16_C( 14465),  INT16_C(  3673), -INT16_C( 29917),  INT16_C(  4127), -INT16_C(  9509), -INT16_C( 17565),  INT16_C( 20971) },
      {  INT16_C( 11337), -INT16_C(  7380),  INT16_C( 27421),  INT16_C( 28423),  INT16_C( 14538), -INT16_C( 20117),  INT16_C(  7031),  INT16_C(  3830),
         INT16_C( 30525), -INT16_C( 27066),  INT16_C( 27014), -INT16_C( 23263), -INT16_C(   903), -INT16_C(  9089),  INT16_C( 27320),  INT16_C(   301),
         INT16_C( 23190), -INT16_C( 19228), -INT16_C(  5179), -INT16_C( 28893), -INT16_C( 29148), -INT16_C( 25791),  INT16_C( 14250), -INT16_C(  6231),
        -INT16_C(  4178),  INT16_C( 13437), -INT16_C( 24743), -INT16_C( 11559),  INT16_C( 22683),  INT16_C( 21423), -INT16_C(  9022),  INT16_C( 22869) },
      UINT32_C(2415919116) },
    { UINT32_C(4228725046),
      {  INT16_C( 12325),  INT16_C( 18827), -INT16_C( 13122),  INT16_C( 26852), -INT16_C( 29437), -INT16_C( 19889), -INT16_C( 12932), -INT16_C( 10778),
        -INT16_C( 16276),  INT16_C(  1960),  INT16_C( 22296), -INT16_C(  9381), -INT16_C( 20429),  INT16_C( 27188),  INT16_C( 16873),  INT16_C(  3686),
        -INT16_C(  3727),  INT16_C( 12119),  INT16_C( 15294), -INT16_C( 15976), -INT16_C(  6200),  INT16_C( 17779),  INT16_C( 23220),  INT16_C(  8218),
        -INT16_C( 15846),  INT16_C( 12840), -INT16_C( 31975),  INT16_C( 19725),  INT16_C( 16691),  INT16_C(  7351),  INT16_C(  7554), -INT16_C(  3285) },
      { -INT16_C( 32242), -INT16_C( 13277), -INT16_C( 17474), -INT16_C( 31090),  INT16_C(   418),  INT16_C( 22475), -INT16_C(  6565),  INT16_C( 30071),
        -INT16_C( 24664), -INT16_C( 15704), -INT16_C( 19166),  INT16_C( 21775), -INT16_C( 14601),  INT16_C( 31090), -INT16_C( 25117), -INT16_C(  3731),
        -INT16_C( 28641), -INT16_C(  8770),  INT16_C( 19531), -INT16_C(  4764),  INT16_C( 12109), -INT16_C( 22204), -INT16_C( 17387), -INT16_C( 16866),
        -INT16_C( 14757),  INT16_C( 32384), -INT16_C( 28804),  INT16_C( 29651),  INT16_C( 17749),  INT16_C( 14572),  INT16_C( 23010),  INT16_C(   553) },
      UINT32_C(4228659248) },
    { UINT32_C( 887089129),
      {  INT16_C( 17203), -INT16_C( 32478),  INT16_C( 26227), -INT16_C( 30678),  INT16_C( 18466),  INT16_C( 32326), -INT16_C( 14833), -INT16_C( 29700),
        -INT16_C( 12459), -INT16_C( 21762), -INT16_C(  5611), -INT16_C(  2078),  INT16_C(  3140),  INT16_C( 11769), -INT16_C(  9741),  INT16_C( 10082),
        -INT16_C( 31716), -INT16_C( 28760), -INT16_C( 11542),  INT16_C(  3352),  INT16_C( 24090),  INT16_C( 10635), -INT16_C( 30939),  INT16_C( 31412),
        -INT16_C( 19882),  INT16_C( 27429),  INT16_C(  1949), -INT16_C(  7837),  INT16_C( 23571),  INT16_C(  1806),  INT16_C( 28725),  INT16_C( 21038) },
      { -INT16_C( 10508), -INT16_C(  8223), -INT16_C(  1624), -INT16_C( 15636),  INT16_C( 30552),  INT16_C( 32236), -INT16_C( 24322),  INT16_C( 21751),
         INT16_C(  7251), -INT16_C(  3904),  INT16_C(  8996),  INT16_C( 14289), -INT16_C(  8321), -INT16_C( 19138),  INT16_C( 27728),  INT16_C( 17415),
        -INT16_C(  6078), -INT16_C(  5597),  INT16_C(  4066),  INT16_C( 15021), -INT16_C( 26234), -INT16_C( 31561), -INT16_C( 20935), -INT16_C( 29479),
        -INT16_C( 26165), -INT16_C(  4228),  INT16_C( 19900),  INT16_C( 15142),  INT16_C( 25901),  INT16_C( 32240), -INT16_C(  2095),  INT16_C(  5313) },
      UINT32_C( 877643656) },
    { UINT32_C(3271484896),
      { -INT16_C( 21516),  INT16_C( 31740), -INT16_C( 19644),  INT16_C( 32511), -INT16_C( 10143),  INT16_C( 11274), -INT16_C( 30863),  INT16_C( 11547),
         INT16_C( 17108),  INT16_C(   361),  INT16_C( 22951),  INT16_C( 30846),  INT16_C( 16465),  INT16_C( 12684), -INT16_C( 29915),  INT16_C(  6643),
        -INT16_C(  4298),  INT16_C( 31636), -INT16_C( 27486),  INT16_C(  1017),  INT16_C(   876), -INT16_C(  8656),  INT16_C( 19338),  INT16_C( 24331),
         INT16_C( 29837),  INT16_C( 13408), -INT16_C(  8242),  INT16_C(  8109),  INT16_C( 14623),  INT16_C( 17488),  INT16_C( 17348), -INT16_C(  1187) },
      { -INT16_C(  3534), -INT16_C( 11146),  INT16_C( 28550), -INT16_C(  3369),  INT16_C(  1906), -INT16_C(   560), -INT16_C(  9133), -INT16_C(  8100),
        -INT16_C( 17328),  INT16_C(  7701), -INT16_C( 15717), -INT16_C( 17859), -INT16_C( 29189), -INT16_C( 16130),  INT16_C( 23760),  INT16_C(   699),
         INT16_C( 12622), -INT16_C( 11050), -INT16_C( 20832),  INT16_C(  4806), -INT16_C( 26699),  INT16_C(  2063),  INT16_C( 27507), -INT16_C( 15383),
        -INT16_C(   472), -INT16_C( 15390),  INT16_C(  8128), -INT16_C( 17538),  INT16_C( 31917),  INT16_C( 32123),  INT16_C( 14040),  INT16_C(  9856) },
      UINT32_C(2154577984) },
    { UINT32_C( 133846631),
      { -INT16_C( 16124), -INT16_C( 17894),  INT16_C( 10584), -INT16_C( 13374), -INT16_C( 21611), -INT16_C( 17010),  INT16_C( 28841),  INT16_C( 27008),
        -INT16_C(   368),  INT16_C( 15653), -INT16_C( 24453),  INT16_C( 21434),  INT16_C( 15063),  INT16_C( 15994),  INT16_C( 29841), -INT16_C( 27322),
         INT16_C( 24629), -INT16_C( 29361),  INT16_C(  4745),  INT16_C(  7768), -INT16_C(  6211),  INT16_C( 26587),  INT16_C( 23639), -INT16_C(  6192),
        -INT16_C(  2726), -INT16_C( 10972), -INT16_C(  8298),  INT16_C( 27945), -INT16_C( 23783), -INT16_C( 21845), -INT16_C(  3817),  INT16_C( 19776) },
      { -INT16_C( 28847), -INT16_C(  9254),  INT16_C( 13217),  INT16_C( 24569), -INT16_C( 10982),  INT16_C( 29126), -INT16_C( 27087), -INT16_C( 29863),
         INT16_C( 32140),  INT16_C(  8801), -INT16_C( 30116),  INT16_C( 30351),  INT16_C( 14893),  INT16_C( 17440),  INT16_C( 24620),  INT16_C( 32145),
         INT16_C( 27888), -INT16_C( 28328),  INT16_C( 21151), -INT16_C( 17936), -INT16_C( 18905),  INT16_C( 22570), -INT16_C( 31923), -INT16_C(  9757),
         INT16_C( 17409),  INT16_C( 24059), -INT16_C( 30002), -INT16_C(  1069), -INT16_C(  2876), -INT16_C(  4032), -INT16_C( 11948),  INT16_C( 17518) },
      UINT32_C(  50462758) },
    { UINT32_C(3705062973),
      { -INT16_C( 14824),  INT16_C( 16277), -INT16_C( 16259), -INT16_C( 13673),  INT16_C( 31555),  INT16_C( 17571), -INT16_C( 24897), -INT16_C( 29022),
         INT16_C( 29992), -INT16_C(  4983), -INT16_C( 13975), -INT16_C( 16675),  INT16_C( 19355), -INT16_C( 10238), -INT16_C( 10223),  INT16_C( 10933),
         INT16_C( 19103),  INT16_C(  7273),  INT16_C(   266),  INT16_C( 20198), -INT16_C( 30340),  INT16_C( 15250),  INT16_C( 13351),  INT16_C( 20425),
         INT16_C( 21418),  INT16_C(  4923),  INT16_C(  6172), -INT16_C( 18479), -INT16_C( 11165),  INT16_C( 30096),  INT16_C( 17836),  INT16_C( 19359) },
      {  INT16_C(  2191), -INT16_C( 26009),  INT16_C( 19721), -INT16_C( 31256),  INT16_C( 31446), -INT16_C(   575), -INT16_C( 30033),  INT16_C( 22860),
        -INT16_C( 30499), -INT16_C(  1428),  INT16_C( 16032),  INT16_C(  1201),  INT16_C( 16658), -INT16_C( 16775),  INT16_C(  6278),  INT16_C(  5642),
         INT16_C( 28960),  INT16_C( 10928), -INT16_C( 26433), -INT16_C( 27217),  INT16_C( 28690), -INT16_C( 15981), -INT16_C(  8197), -INT16_C( 10214),
        -INT16_C( 30873),  INT16_C(  2258), -INT16_C( 31547), -INT16_C( 10484), -INT16_C( 31291),  INT16_C( 19605), -INT16_C( 24675), -INT16_C( 17054) },
      UINT32_C( 135415301) },
    { UINT32_C(3504804369),
      { -INT16_C( 26710), -INT16_C( 17307), -INT16_C(  2041),  INT16_C(   638), -INT16_C( 26408),  INT16_C( 16347), -INT16_C( 21217), -INT16_C(  7097),
         INT16_C( 21297), -INT16_C(  2117),  INT16_C( 20952),  INT16_C( 30019), -INT16_C( 23056),  INT16_C(   307),  INT16_C(  6839),  INT16_C( 25041),
         INT16_C( 14257), -INT16_C( 18147), -INT16_C( 25809),  INT16_C(  1979), -INT16_C( 27084),  INT16_C( 21319), -INT16_C( 29116),  INT16_C( 30008),
        -INT16_C(  3102), -INT16_C( 17812), -INT16_C( 20668),  INT16_C( 13616),  INT16_C( 25428),  INT16_C(  2870),  INT16_C(  2173),  INT16_C( 12140) },
      { -INT16_C( 30145),  INT16_C( 28392), -INT16_C( 23771),  INT16_C( 22902), -INT16_C( 17094),  INT16_C( 32429), -INT16_C(  6837),  INT16_C( 11763),
         INT16_C( 24792),  INT16_C(  7656),  INT16_C(  6159),  INT16_C( 25682), -INT16_C( 30597), -INT16_C(  1937), -INT16_C(  9072), -INT16_C( 12505),
         INT16_C(  3942), -INT16_C( 29890), -INT16_C( 19277), -INT16_C(  4635), -INT16_C( 28047), -INT16_C( 17301),  INT16_C( 24183),  INT16_C( 20458),
        -INT16_C( 11586), -INT16_C( 12692), -INT16_C( 16662),  INT16_C( 25906), -INT16_C( 24249), -INT16_C( 10403), -INT16_C( 31363), -INT16_C(  7257) },
      UINT32_C(   4456976) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epi16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epi16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 r = easysimd_mm512_mask_cmplt_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmplt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int32_t a[16];
    int32_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C( 3506),
      { -INT32_C(  2004561150),  INT32_C(  1964540175),  INT32_C(  1337577994), -INT32_C(    11026497),  INT32_C(   959822125), -INT32_C(    18130030), -INT32_C(  1259690788),  INT32_C(   667065381),
         INT32_C(  2125481583), -INT32_C(   772486969),  INT32_C(  1495313818), -INT32_C(  1722255508), -INT32_C(  1009611471), -INT32_C(   977158680), -INT32_C(  2038846623), -INT32_C(  1213318328) },
      {  INT32_C(  1211522689), -INT32_C(  1072092634),  INT32_C(  1125726935), -INT32_C(   488869455), -INT32_C(   408571906), -INT32_C(   827562388),  INT32_C(  1515463954), -INT32_C(   502201759),
        -INT32_C(  2043984032),  INT32_C(  1212564593),  INT32_C(   814440318), -INT32_C(   820877104), -INT32_C(  2085177321),  INT32_C(   810639902), -INT32_C(   393566585),  INT32_C(   164338856) },
      UINT16_C( 2048) },
    { UINT16_C(63203),
      { -INT32_C(   700820081),  INT32_C(   691386781), -INT32_C(    74381591), -INT32_C(  1951160107),  INT32_C(  2112737835), -INT32_C(  1927056382),  INT32_C(   673827677),  INT32_C(  1679690964),
        -INT32_C(    12953503), -INT32_C(    98013423),  INT32_C(  1257683317), -INT32_C(  1932154527),  INT32_C(  2114568827), -INT32_C(  1811207113), -INT32_C(   860080904), -INT32_C(  1573856704) },
      {  INT32_C(  1151429170),  INT32_C(  1329514970), -INT32_C(   476433278),  INT32_C(  1500475358),  INT32_C(  1775728946), -INT32_C(  1644305755),  INT32_C(  1466611990), -INT32_C(   973497709),
        -INT32_C(   553018875), -INT32_C(   449951645),  INT32_C(  1523107964),  INT32_C(  1790195512),  INT32_C(  1456704432), -INT32_C(  2064396178),  INT32_C(   484138377), -INT32_C(    35531528) },
      UINT16_C(50275) },
    { UINT16_C(60270),
      {  INT32_C(   171168220),  INT32_C(  2127802294), -INT32_C(  1095365878),  INT32_C(  1212769909), -INT32_C(  1340557380),  INT32_C(   386769212), -INT32_C(  1628699203), -INT32_C(   561423870),
        -INT32_C(   487998420),  INT32_C(  1986116716),  INT32_C(  1010046919),  INT32_C(   981761405),  INT32_C(  1911200821), -INT32_C(    91687107), -INT32_C(     6785795), -INT32_C(    69328177) },
      {  INT32_C(  1256048606),  INT32_C(  1254112899), -INT32_C(   746130347), -INT32_C(  1492317582), -INT32_C(   484903002), -INT32_C(   337796882), -INT32_C(   454331115),  INT32_C(  1994377623),
         INT32_C(   331398288),  INT32_C(  1348305403), -INT32_C(   417078411), -INT32_C(  1819397907),  INT32_C(   376874535),  INT32_C(  1543590727),  INT32_C(  1614867656),  INT32_C(  1171660725) },
      UINT16_C(57668) },
    { UINT16_C(38619),
      { -INT32_C(  1256729000),  INT32_C(  1251511591),  INT32_C(    58426740), -INT32_C(  1901485544), -INT32_C(  1159597896), -INT32_C(  1935234484), -INT32_C(   525640694), -INT32_C(    92895327),
        -INT32_C(  2068869539), -INT32_C(  1882240997), -INT32_C(   426620211), -INT32_C(  1519108884),  INT32_C(  2036291116),  INT32_C(   168101120),  INT32_C(    65712481), -INT32_C(  1778556872) },
      {  INT32_C(   152743150), -INT32_C(  1030100492),  INT32_C(   531114803), -INT32_C(  1799087001),  INT32_C(  1913463666), -INT32_C(  1971580375), -INT32_C(    74619197), -INT32_C(  1248752953),
         INT32_C(   733981495), -INT32_C(   940746604), -INT32_C(   337209981),  INT32_C(   612346802), -INT32_C(   141062962),  INT32_C(  1652691870),  INT32_C(  1096617850), -INT32_C(   772346214) },
      UINT16_C(34393) },
    { UINT16_C(46489),
      { -INT32_C(   368235012), -INT32_C(   612396556), -INT32_C(    75091588), -INT32_C(   309898154), -INT32_C(   838851252), -INT32_C(   438470009), -INT32_C(  1294764101),  INT32_C(  1164406088),
        -INT32_C(  1892715110), -INT32_C(  2106937850),  INT32_C(   914223328), -INT32_C(  1876753340), -INT32_C(  1319230679),  INT32_C(  1503018142), -INT32_C(    66360909),  INT32_C(  1900114902) },
      { -INT32_C(   301960984), -INT32_C(    26187234), -INT32_C(  1640698534),  INT32_C(   439310321),  INT32_C(   432770427),  INT32_C(  2104648137), -INT32_C(  1585873205), -INT32_C(   653083919),
         INT32_C(  1221005866), -INT32_C(   699975812),  INT32_C(   360020772),  INT32_C(  1311745235), -INT32_C(    77071567),  INT32_C(   662231388),  INT32_C(  1221194071), -INT32_C(   719201365) },
      UINT16_C( 4377) },
    { UINT16_C(59630),
      {  INT32_C(  1679780381), -INT32_C(  1226816703), -INT32_C(  2007321768),  INT32_C(  1753451265), -INT32_C(    29237370), -INT32_C(   789604089),  INT32_C(    61577953), -INT32_C(  1913939601),
         INT32_C(  1173424900), -INT32_C(  1476669106), -INT32_C(  2060495484),  INT32_C(  1743631072), -INT32_C(  1738199151), -INT32_C(  1452780088),  INT32_C(  1605113071), -INT32_C(  1309894483) },
      { -INT32_C(   235479645),  INT32_C(   865661359),  INT32_C(   666486598),  INT32_C(   193898105), -INT32_C(  1650199595),  INT32_C(   944180296), -INT32_C(   845679840),  INT32_C(   780108683),
         INT32_C(   253785440), -INT32_C(  1388070809), -INT32_C(   103482240),  INT32_C(  2013553314), -INT32_C(  1642747818), -INT32_C(   724149068), -INT32_C(   610177712),  INT32_C(  1359618288) },
      UINT16_C(43174) },
    { UINT16_C(10902),
      { -INT32_C(  1545405088),  INT32_C(  2141151914),  INT32_C(  1625375324), -INT32_C(   821545030), -INT32_C(  1406354218),  INT32_C(   840596624),  INT32_C(  1632766551), -INT32_C(  1131681701),
        -INT32_C(  1872793883),  INT32_C(   739246032), -INT32_C(    74583999), -INT32_C(    20277976), -INT32_C(   492112302), -INT32_C(   904608654),  INT32_C(   707487695),  INT32_C(   904312655) },
      { -INT32_C(   171620827), -INT32_C(  2044537787), -INT32_C(   327045180), -INT32_C(  1763030204), -INT32_C(  1267100350),  INT32_C(   679382361),  INT32_C(  1146333940), -INT32_C(  2038875807),
        -INT32_C(   998556033), -INT32_C(   699753198), -INT32_C(  1849439412),  INT32_C(  1495772439), -INT32_C(  1676828606),  INT32_C(   583306286), -INT32_C(  1754917066), -INT32_C(   803348655) },
      UINT16_C(10256) },
    { UINT16_C(38941),
      { -INT32_C(   550097003), -INT32_C(   911506938),  INT32_C(   997638675),  INT32_C(   685488411),  INT32_C(   431229269), -INT32_C(  1842222548), -INT32_C(  1636662655), -INT32_C(   415789230),
        -INT32_C(   976785985),  INT32_C(    59667183),  INT32_C(  1329464372),  INT32_C(   309795261),  INT32_C(  1328294947), -INT32_C(  1730061034),  INT32_C(   825643999), -INT32_C(  1575391773) },
      { -INT32_C(   899161894), -INT32_C(  2033322670), -INT32_C(  1210774535),  INT32_C(  1204374564), -INT32_C(  1885932168),  INT32_C(   824670290), -INT32_C(  1369219637), -INT32_C(  1521451829),
        -INT32_C(  1368410276), -INT32_C(  1506526036),  INT32_C(  1818036551), -INT32_C(   827120043),  INT32_C(  1851607580), -INT32_C(  1918860094), -INT32_C(  1405352991), -INT32_C(   615347073) },
      UINT16_C(36872) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epi32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 r = easysimd_mm512_mask_cmplt_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmplt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t a[8];
    int64_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 62),
      {  INT64_C( 5436270409989212626), -INT64_C( 7881761334238864076),  INT64_C( 2070210919304500132), -INT64_C( 1771219687195319385),
        -INT64_C( 9037323338077582553), -INT64_C( 2522986417782530921), -INT64_C( 4100125997298138336),  INT64_C( 3011887816368301264) },
      { -INT64_C( 1802558812428544725), -INT64_C( 7352605258619491358), -INT64_C( 1695412743423219056), -INT64_C(  468768204013081261),
        -INT64_C( 2800271439910670098),  INT64_C( 5723488811090146139),  INT64_C(  473855094572450708), -INT64_C( 1753066863956631323) },
      UINT8_C( 58) },
    { UINT8_C(202),
      { -INT64_C( 4978278280952994786),  INT64_C( 3317966734718844301), -INT64_C( 6113946237501655095), -INT64_C( 6636462501839709886),
        -INT64_C( 3480690691364771920), -INT64_C( 4670423072023106345),  INT64_C( 4106763849116487689), -INT64_C( 2366576528896777472) },
      {  INT64_C( 5400812605291321826), -INT64_C( 7751685018790184522), -INT64_C( 7657554690881065589),  INT64_C( 3213391218517610121),
        -INT64_C( 5739762988117626981), -INT64_C( 6224880553618005124), -INT64_C(  837698754551512011),  INT64_C( 6226584630312401395) },
      UINT8_C(136) },
    { UINT8_C(180),
      {  INT64_C( 6879352596594582713), -INT64_C( 6793637030174752582),  INT64_C( 6497288954214908498), -INT64_C( 8455671225894213458),
         INT64_C( 7992762010792515023), -INT64_C( 2072313814947108481), -INT64_C(  475053572042896852), -INT64_C( 5185732339851035219) },
      {  INT64_C(  867558163480874729), -INT64_C( 3383358197524268136),  INT64_C( 1833886900706747951), -INT64_C( 2565467444267753532),
         INT64_C( 8126609019494770990), -INT64_C( 8866281080809230466), -INT64_C( 2499653815544851510), -INT64_C( 5358737619963681070) },
      UINT8_C( 16) },
    { UINT8_C(153),
      { -INT64_C( 6405277312233607143),  INT64_C(  498352765374578299),  INT64_C( 6252400839942414464),  INT64_C( 4904861080768776583),
        -INT64_C( 2170177133217807614), -INT64_C( 2345272185846636998),  INT64_C( 1716318476006691340),  INT64_C( 4723888153340849695) },
      { -INT64_C( 6599641066797044190),  INT64_C( 7023790597052918331),  INT64_C(   53391050493741590),  INT64_C(  999193198797146972),
        -INT64_C( 1056939951925143226),  INT64_C(    8731126248727912),  INT64_C(  517742069060926542), -INT64_C( 7518898865401792082) },
      UINT8_C( 16) },
    { UINT8_C(218),
      {  INT64_C(  190138117293885500),  INT64_C( 9033916842118963468),  INT64_C( 3864986493322300817),  INT64_C( 2318824819866877139),
         INT64_C( 5405724999327399526),  INT64_C( 7880941851486449760), -INT64_C( 8486746838703962297),  INT64_C( 4919071940747968577) },
      { -INT64_C(  119543722935935028), -INT64_C( 5439436390483905989),  INT64_C( 3028373784558626244), -INT64_C( 4696247316421548623),
        -INT64_C( 6859226986266879437), -INT64_C( 5695862846639395117),  INT64_C( 5706959703325256735),  INT64_C( 5244356094525866518) },
      UINT8_C(192) },
    { UINT8_C(204),
      { -INT64_C( 7600156523788953199),  INT64_C( 7412980764343715463), -INT64_C( 7015110561947719331),  INT64_C(  553938866746404401),
         INT64_C( 4076954695904190485), -INT64_C( 6474247408288393722), -INT64_C( 5945355703203583549),  INT64_C( 4757559523157225865) },
      {  INT64_C(  434831563511873567), -INT64_C( 6855907047900839189),  INT64_C( 8232577823483629334),  INT64_C( 3287196781086101294),
        -INT64_C( 6971847430816014792),  INT64_C(  614088182955443720),  INT64_C( 3537816116985988540),  INT64_C( 4243282560270829992) },
      UINT8_C( 76) },
    { UINT8_C(  9),
      { -INT64_C( 5121053802828565536),  INT64_C( 6816395230265987887), -INT64_C( 2319192913665245874),  INT64_C( 5355934900675446303),
        -INT64_C( 7135920154991390323), -INT64_C( 6509935294266432527), -INT64_C( 4935210049362720435),  INT64_C(  781191456143794723) },
      {  INT64_C( 1761825713331767402),  INT64_C( 5323829783688309734),  INT64_C( 7956011488063041299),  INT64_C( 3699939951140106192),
        -INT64_C( 3739818493524401445), -INT64_C( 7444753076326925463), -INT64_C( 2858184836227032421), -INT64_C(  136376671020888033) },
      UINT8_C(  1) },
    { UINT8_C( 32),
      { -INT64_C( 7706272379809955803), -INT64_C( 5016250081576586533), -INT64_C( 1240008741126780422), -INT64_C(  982047720047725888),
         INT64_C( 1350977870567466256), -INT64_C( 8723687380296738539),  INT64_C(  870033978167505285),  INT64_C( 5002559328956851373) },
      { -INT64_C( 5019210497335033330), -INT64_C(  857128212534185465), -INT64_C(  871682618572715121), -INT64_C( 1601070506186034867),
         INT64_C( 7727199926411540188), -INT64_C( 7342246136574432071), -INT64_C( 6152097012864699547), -INT64_C( 4372164113012617843) },
      UINT8_C( 32) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epi64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 r = easysimd_mm512_mask_cmplt_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmplt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask64 r;
  } test_vec[8] = {
    { easysimd_x_mm512_set_epu8(UINT8_C(200), UINT8_C( 64), UINT8_C(228), UINT8_C(187),
                             UINT8_C( 53), UINT8_C(115), UINT8_C(212), UINT8_C(224),
                             UINT8_C(234), UINT8_C( 45), UINT8_C(183), UINT8_C(185),
                             UINT8_C(117), UINT8_C(137), UINT8_C(224), UINT8_C( 48),
                             UINT8_C(225), UINT8_C(229), UINT8_C(194), UINT8_C(201),
                             UINT8_C(105), UINT8_C(193), UINT8_C(219), UINT8_C(144),
                             UINT8_C( 87), UINT8_C(205), UINT8_C( 38), UINT8_C(200),
                             UINT8_C( 89), UINT8_C(  3), UINT8_C(155), UINT8_C(245),
                             UINT8_C( 35), UINT8_C(246), UINT8_C( 15), UINT8_C(254),
                             UINT8_C(226), UINT8_C(163), UINT8_C( 22), UINT8_C(135),
                             UINT8_C(105), UINT8_C(246), UINT8_C(210), UINT8_C(152),
                             UINT8_C(131), UINT8_C(226), UINT8_C(111), UINT8_C(207),
                             UINT8_C( 48), UINT8_C(166), UINT8_C( 61), UINT8_C( 72),
                             UINT8_C(115), UINT8_C( 10), UINT8_C(124), UINT8_C( 60),
                             UINT8_C(127), UINT8_C( 50), UINT8_C( 32), UINT8_C( 65),
                             UINT8_C(138), UINT8_C(206), UINT8_C( 67), UINT8_C( 10)),
      easysimd_x_mm512_set_epu8(UINT8_C(101), UINT8_C(156), UINT8_C(229), UINT8_C( 20),
                             UINT8_C(123), UINT8_C(204), UINT8_C(216), UINT8_C( 73),
                             UINT8_C(103), UINT8_C(232), UINT8_C(253), UINT8_C(122),
                             UINT8_C( 63), UINT8_C(129), UINT8_C(179), UINT8_C(165),
                             UINT8_C(219), UINT8_C( 23), UINT8_C( 44), UINT8_C(209),
                             UINT8_C( 74), UINT8_C(150), UINT8_C(180), UINT8_C(217),
                             UINT8_C( 86), UINT8_C(119), UINT8_C( 26), UINT8_C( 59),
                             UINT8_C(214), UINT8_C( 93), UINT8_C( 27), UINT8_C( 37),
                             UINT8_C( 61), UINT8_C( 47), UINT8_C(126), UINT8_C(138),
                             UINT8_C(246), UINT8_C( 88), UINT8_C(146), UINT8_C(144),
                             UINT8_C(216), UINT8_C( 36), UINT8_C(167), UINT8_C(247),
                             UINT8_C(118), UINT8_C( 82), UINT8_C( 16), UINT8_C(237),
                             UINT8_C(182), UINT8_C(219), UINT8_C( 49), UINT8_C( 46),
                             UINT8_C(225), UINT8_C( 22), UINT8_C(189), UINT8_C( 65),
                             UINT8_C(101), UINT8_C(226), UINT8_C( 23), UINT8_C(220),
                             UINT8_C( 28), UINT8_C( 93), UINT8_C(158), UINT8_C(  8)),
      UINT64_C(7953657163028746066) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 72), UINT8_C( 33), UINT8_C(154), UINT8_C(122),
                             UINT8_C( 74), UINT8_C(178), UINT8_C( 50), UINT8_C(213),
                             UINT8_C( 98), UINT8_C(217), UINT8_C( 79), UINT8_C(232),
                             UINT8_C(132), UINT8_C(243), UINT8_C(145), UINT8_C(149),
                             UINT8_C( 59), UINT8_C(222), UINT8_C(125), UINT8_C(147),
                             UINT8_C(164), UINT8_C(104), UINT8_C(229), UINT8_C(111),
                             UINT8_C( 98), UINT8_C(138), UINT8_C(193), UINT8_C(124),
                             UINT8_C( 63), UINT8_C(242), UINT8_C(  1), UINT8_C( 40),
                             UINT8_C(135), UINT8_C(187), UINT8_C(113), UINT8_C(230),
                             UINT8_C(208), UINT8_C(113), UINT8_C(106), UINT8_C( 33),
                             UINT8_C(173), UINT8_C( 82), UINT8_C( 93), UINT8_C(149),
                             UINT8_C(  4), UINT8_C(122), UINT8_C( 61), UINT8_C( 54),
                             UINT8_C(106), UINT8_C(212), UINT8_C( 67), UINT8_C(253),
                             UINT8_C(216), UINT8_C(134), UINT8_C(207), UINT8_C( 61),
                             UINT8_C(208), UINT8_C( 81), UINT8_C(150), UINT8_C(127),
                             UINT8_C( 37), UINT8_C(137), UINT8_C(225), UINT8_C( 78)),
      easysimd_x_mm512_set_epu8(UINT8_C( 96), UINT8_C(106), UINT8_C(154), UINT8_C(  0),
                             UINT8_C(  1), UINT8_C(122), UINT8_C(193), UINT8_C( 25),
                             UINT8_C(239), UINT8_C(152), UINT8_C( 87), UINT8_C( 80),
                             UINT8_C( 31), UINT8_C(133), UINT8_C(251), UINT8_C( 23),
                             UINT8_C(140), UINT8_C(137), UINT8_C(157), UINT8_C(114),
                             UINT8_C( 93), UINT8_C(199), UINT8_C( 63), UINT8_C( 63),
                             UINT8_C(  7), UINT8_C(151), UINT8_C( 52), UINT8_C( 78),
                             UINT8_C(106), UINT8_C( 19), UINT8_C( 61), UINT8_C( 80),
                             UINT8_C(227), UINT8_C( 61), UINT8_C(244), UINT8_C( 16),
                             UINT8_C(153), UINT8_C(241), UINT8_C(118), UINT8_C(250),
                             UINT8_C(216), UINT8_C( 38), UINT8_C(  9), UINT8_C(176),
                             UINT8_C( 88), UINT8_C(222), UINT8_C( 84), UINT8_C(243),
                             UINT8_C(  6), UINT8_C( 75), UINT8_C(  1), UINT8_C(141),
                             UINT8_C(140), UINT8_C( 75), UINT8_C(187), UINT8_C(128),
                             UINT8_C(101), UINT8_C(169), UINT8_C(202), UINT8_C(205),
                             UINT8_C(101), UINT8_C( 13), UINT8_C(249), UINT8_C(158)),
      UINT64_C(14024952834426863995) },
    { easysimd_x_mm512_set_epu8(UINT8_C(118), UINT8_C(178), UINT8_C(224), UINT8_C( 82),
                             UINT8_C( 86), UINT8_C(103), UINT8_C( 51), UINT8_C( 73),
                             UINT8_C(131), UINT8_C(150), UINT8_C( 58), UINT8_C(120),
                             UINT8_C( 23), UINT8_C(173), UINT8_C( 81), UINT8_C(210),
                             UINT8_C(130), UINT8_C( 18), UINT8_C(188), UINT8_C( 71),
                             UINT8_C( 31), UINT8_C(136), UINT8_C(151), UINT8_C(116),
                             UINT8_C(141), UINT8_C( 84), UINT8_C( 91), UINT8_C( 42),
                             UINT8_C( 78), UINT8_C(105), UINT8_C( 68), UINT8_C(247),
                             UINT8_C(246), UINT8_C( 98), UINT8_C(236), UINT8_C(254),
                             UINT8_C( 34), UINT8_C( 84), UINT8_C(115), UINT8_C(219),
                             UINT8_C( 18), UINT8_C(  7), UINT8_C( 63), UINT8_C(118),
                             UINT8_C( 52), UINT8_C( 47), UINT8_C(109), UINT8_C( 86),
                             UINT8_C( 94), UINT8_C( 32), UINT8_C( 34), UINT8_C(216),
                             UINT8_C(187), UINT8_C(179), UINT8_C( 76), UINT8_C(176),
                             UINT8_C(254), UINT8_C(105), UINT8_C( 86), UINT8_C(220),
                             UINT8_C(  7), UINT8_C( 77), UINT8_C(  8), UINT8_C(213)),
      easysimd_x_mm512_set_epu8(UINT8_C(139), UINT8_C( 33), UINT8_C( 28), UINT8_C(170),
                             UINT8_C( 51), UINT8_C(116), UINT8_C(243), UINT8_C( 67),
                             UINT8_C(171), UINT8_C( 22), UINT8_C( 12), UINT8_C( 38),
                             UINT8_C(216), UINT8_C(230), UINT8_C(112), UINT8_C( 69),
                             UINT8_C(229), UINT8_C(115), UINT8_C(110), UINT8_C(236),
                             UINT8_C( 84), UINT8_C(115), UINT8_C(237), UINT8_C(235),
                             UINT8_C( 57), UINT8_C(112), UINT8_C( 59), UINT8_C(169),
                             UINT8_C(188), UINT8_C( 43), UINT8_C( 43), UINT8_C(171),
                             UINT8_C(177), UINT8_C( 79), UINT8_C(216), UINT8_C( 10),
                             UINT8_C(218), UINT8_C(219), UINT8_C(205), UINT8_C( 15),
                             UINT8_C(248), UINT8_C( 98), UINT8_C( 68), UINT8_C( 51),
                             UINT8_C(  2), UINT8_C( 76), UINT8_C(147), UINT8_C(206),
                             UINT8_C(115), UINT8_C(128), UINT8_C(102), UINT8_C(116),
                             UINT8_C( 28), UINT8_C(  5), UINT8_C( 73), UINT8_C(210),
                             UINT8_C(179), UINT8_C(184), UINT8_C( 57), UINT8_C( 24),
                             UINT8_C( 10), UINT8_C( 26), UINT8_C(  0), UINT8_C( 23)),
      UINT64_C(10848849723635786056) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 59), UINT8_C(122), UINT8_C( 87), UINT8_C(211),
                             UINT8_C(197), UINT8_C(251), UINT8_C( 70), UINT8_C(103),
                             UINT8_C( 31), UINT8_C(245), UINT8_C(135), UINT8_C( 65),
                             UINT8_C(214), UINT8_C(158), UINT8_C(167), UINT8_C(  6),
                             UINT8_C(130), UINT8_C( 46), UINT8_C( 33), UINT8_C( 54),
                             UINT8_C( 11), UINT8_C(245), UINT8_C( 84), UINT8_C(244),
                             UINT8_C(176), UINT8_C( 82), UINT8_C(238), UINT8_C(130),
                             UINT8_C(209), UINT8_C(204), UINT8_C(248), UINT8_C(  0),
                             UINT8_C(157), UINT8_C(108), UINT8_C(156), UINT8_C(156),
                             UINT8_C(237), UINT8_C( 39), UINT8_C(113), UINT8_C(164),
                             UINT8_C( 74), UINT8_C( 17), UINT8_C(157), UINT8_C(212),
                             UINT8_C( 94), UINT8_C(225), UINT8_C(154), UINT8_C(241),
                             UINT8_C(234), UINT8_C( 71), UINT8_C( 97), UINT8_C( 49),
                             UINT8_C(235), UINT8_C( 52), UINT8_C(222), UINT8_C( 20),
                             UINT8_C(  5), UINT8_C(231), UINT8_C(123), UINT8_C( 11),
                             UINT8_C( 62), UINT8_C(215), UINT8_C(218), UINT8_C( 46)),
      easysimd_x_mm512_set_epu8(UINT8_C(124), UINT8_C(178), UINT8_C(245), UINT8_C(131),
                             UINT8_C( 34), UINT8_C(224), UINT8_C( 87), UINT8_C(197),
                             UINT8_C(182), UINT8_C(209), UINT8_C(166), UINT8_C(  5),
                             UINT8_C(234), UINT8_C(185), UINT8_C(158), UINT8_C(144),
                             UINT8_C( 14), UINT8_C(107), UINT8_C(220), UINT8_C( 70),
                             UINT8_C(119), UINT8_C(160), UINT8_C( 31), UINT8_C(191),
                             UINT8_C(230), UINT8_C(198), UINT8_C(152), UINT8_C( 33),
                             UINT8_C( 22), UINT8_C( 95), UINT8_C(212), UINT8_C(255),
                             UINT8_C(113), UINT8_C(254), UINT8_C(  2), UINT8_C(210),
                             UINT8_C(195), UINT8_C(115), UINT8_C(106), UINT8_C(226),
                             UINT8_C( 64), UINT8_C(138), UINT8_C( 67), UINT8_C( 75),
                             UINT8_C( 11), UINT8_C(130), UINT8_C(117), UINT8_C( 51),
                             UINT8_C(106), UINT8_C(104), UINT8_C( 34), UINT8_C(112),
                             UINT8_C( 78), UINT8_C( 85), UINT8_C(189), UINT8_C( 13),
                             UINT8_C(140), UINT8_C( 10), UINT8_C( 60), UINT8_C( 18),
                             UINT8_C(  8), UINT8_C(179), UINT8_C( 57), UINT8_C(196)),
      UINT64_C(16405901789334885521) },
    { easysimd_x_mm512_set_epu8(UINT8_C(133), UINT8_C( 51), UINT8_C(153), UINT8_C( 32),
                             UINT8_C( 25), UINT8_C(207), UINT8_C(  4), UINT8_C( 40),
                             UINT8_C( 26), UINT8_C( 76), UINT8_C( 93), UINT8_C(  5),
                             UINT8_C(177), UINT8_C(180), UINT8_C(109), UINT8_C(128),
                             UINT8_C(101), UINT8_C( 26), UINT8_C(223), UINT8_C( 68),
                             UINT8_C( 88), UINT8_C(  3), UINT8_C(  1), UINT8_C(148),
                             UINT8_C(  0), UINT8_C(113), UINT8_C( 62), UINT8_C(107),
                             UINT8_C(163), UINT8_C(164), UINT8_C(165), UINT8_C(185),
                             UINT8_C(198), UINT8_C(  6), UINT8_C(199), UINT8_C(198),
                             UINT8_C(207), UINT8_C(139), UINT8_C(  4), UINT8_C( 52),
                             UINT8_C( 26), UINT8_C(160), UINT8_C(162), UINT8_C(224),
                             UINT8_C( 24), UINT8_C(137), UINT8_C(101), UINT8_C( 24),
                             UINT8_C(115), UINT8_C(224), UINT8_C(208), UINT8_C( 54),
                             UINT8_C(102), UINT8_C( 97), UINT8_C(207), UINT8_C( 72),
                             UINT8_C( 87), UINT8_C( 19), UINT8_C(168), UINT8_C(205),
                             UINT8_C( 92), UINT8_C( 79), UINT8_C( 86), UINT8_C(144)),
      easysimd_x_mm512_set_epu8(UINT8_C( 76), UINT8_C(120), UINT8_C(206), UINT8_C( 35),
                             UINT8_C( 84), UINT8_C(143), UINT8_C(212), UINT8_C( 97),
                             UINT8_C(238), UINT8_C(159), UINT8_C(181), UINT8_C(100),
                             UINT8_C(208), UINT8_C(157), UINT8_C( 32), UINT8_C(247),
                             UINT8_C( 15), UINT8_C(143), UINT8_C(  2), UINT8_C(229),
                             UINT8_C( 17), UINT8_C( 50), UINT8_C(  1), UINT8_C(241),
                             UINT8_C( 90), UINT8_C(132), UINT8_C( 93), UINT8_C( 20),
                             UINT8_C( 46), UINT8_C(183), UINT8_C(154), UINT8_C(255),
                             UINT8_C(118), UINT8_C(135), UINT8_C(  8), UINT8_C(254),
                             UINT8_C(139), UINT8_C(221), UINT8_C(207), UINT8_C(230),
                             UINT8_C(231), UINT8_C( 92), UINT8_C(100), UINT8_C(108),
                             UINT8_C(158), UINT8_C(233), UINT8_C(  8), UINT8_C(234),
                             UINT8_C(189), UINT8_C(236), UINT8_C( 58), UINT8_C(205),
                             UINT8_C(125), UINT8_C(116), UINT8_C(230), UINT8_C(218),
                             UINT8_C(185), UINT8_C(225), UINT8_C( 61), UINT8_C(183),
                             UINT8_C(233), UINT8_C(244), UINT8_C(138), UINT8_C(204)),
      UINT64_C(8933265779370876879) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 74), UINT8_C(  8), UINT8_C( 71), UINT8_C( 14),
                             UINT8_C(239), UINT8_C(140), UINT8_C( 39), UINT8_C( 68),
                             UINT8_C( 18), UINT8_C(182), UINT8_C(128), UINT8_C(142),
                             UINT8_C( 75), UINT8_C(196), UINT8_C(121), UINT8_C(239),
                             UINT8_C( 67), UINT8_C(139), UINT8_C( 89), UINT8_C( 42),
                             UINT8_C(150), UINT8_C(200), UINT8_C( 22), UINT8_C( 70),
                             UINT8_C( 92), UINT8_C(114), UINT8_C(  0), UINT8_C(232),
                             UINT8_C(121), UINT8_C(124), UINT8_C(100), UINT8_C(100),
                             UINT8_C(142), UINT8_C( 19), UINT8_C(218), UINT8_C(104),
                             UINT8_C(159), UINT8_C(120), UINT8_C(122), UINT8_C( 55),
                             UINT8_C(213), UINT8_C(170), UINT8_C(221), UINT8_C(149),
                             UINT8_C(230), UINT8_C(250), UINT8_C(104), UINT8_C( 36),
                             UINT8_C( 99), UINT8_C( 18), UINT8_C(124), UINT8_C(175),
                             UINT8_C(103), UINT8_C(186), UINT8_C(205), UINT8_C( 43),
                             UINT8_C(141), UINT8_C(148), UINT8_C(140), UINT8_C( 44),
                             UINT8_C(237), UINT8_C(120), UINT8_C(114), UINT8_C(100)),
      easysimd_x_mm512_set_epu8(UINT8_C(124), UINT8_C(149), UINT8_C( 71), UINT8_C(212),
                             UINT8_C(137), UINT8_C(252), UINT8_C(249), UINT8_C( 42),
                             UINT8_C(167), UINT8_C(191), UINT8_C(236), UINT8_C(252),
                             UINT8_C( 26), UINT8_C( 50), UINT8_C( 98), UINT8_C(162),
                             UINT8_C( 91), UINT8_C(215), UINT8_C( 44), UINT8_C( 48),
                             UINT8_C( 41), UINT8_C(167), UINT8_C( 25), UINT8_C( 39),
                             UINT8_C(183), UINT8_C(181), UINT8_C(250), UINT8_C( 47),
                             UINT8_C(  5), UINT8_C(113), UINT8_C( 48), UINT8_C(195),
                             UINT8_C(111), UINT8_C( 46), UINT8_C( 74), UINT8_C( 84),
                             UINT8_C(145), UINT8_C( 27), UINT8_C(231), UINT8_C(119),
                             UINT8_C( 33), UINT8_C(230), UINT8_C( 22), UINT8_C( 69),
                             UINT8_C( 48), UINT8_C(  7), UINT8_C( 45), UINT8_C(104),
                             UINT8_C( 71), UINT8_C( 82), UINT8_C(107), UINT8_C( 14),
                             UINT8_C( 73), UINT8_C(202), UINT8_C( 78), UINT8_C(132),
                             UINT8_C( 67), UINT8_C( 79), UINT8_C(233), UINT8_C(140),
                             UINT8_C(133), UINT8_C( 99), UINT8_C(202), UINT8_C( 75)),
      UINT64_C(15488110983464961330) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 78), UINT8_C( 83), UINT8_C(217), UINT8_C( 23),
                             UINT8_C(204), UINT8_C( 27), UINT8_C( 84), UINT8_C( 42),
                             UINT8_C(170), UINT8_C( 43), UINT8_C(212), UINT8_C(144),
                             UINT8_C( 56), UINT8_C(177), UINT8_C(191), UINT8_C(215),
                             UINT8_C( 39), UINT8_C( 30), UINT8_C(  2), UINT8_C(234),
                             UINT8_C( 49), UINT8_C(151), UINT8_C(136), UINT8_C(175),
                             UINT8_C(252), UINT8_C(162), UINT8_C(152), UINT8_C(153),
                             UINT8_C(239), UINT8_C(231), UINT8_C(133), UINT8_C(178),
                             UINT8_C(148), UINT8_C( 35), UINT8_C(158), UINT8_C(129),
                             UINT8_C( 19), UINT8_C(213), UINT8_C( 89), UINT8_C(159),
                             UINT8_C(156), UINT8_C( 31), UINT8_C(228), UINT8_C(142),
                             UINT8_C( 99), UINT8_C( 45), UINT8_C(244), UINT8_C(239),
                             UINT8_C( 20), UINT8_C( 92), UINT8_C(183), UINT8_C( 74),
                             UINT8_C(105), UINT8_C(182), UINT8_C(238), UINT8_C( 27),
                             UINT8_C(161), UINT8_C(150), UINT8_C(240), UINT8_C( 67),
                             UINT8_C( 60), UINT8_C(157), UINT8_C( 26), UINT8_C( 30)),
      easysimd_x_mm512_set_epu8(UINT8_C(112), UINT8_C(252), UINT8_C(254), UINT8_C(234),
                             UINT8_C(115), UINT8_C(252), UINT8_C(144), UINT8_C(157),
                             UINT8_C(106), UINT8_C(131), UINT8_C(237), UINT8_C( 28),
                             UINT8_C( 29), UINT8_C( 85), UINT8_C(  8), UINT8_C(128),
                             UINT8_C(244), UINT8_C(127), UINT8_C(116), UINT8_C( 60),
                             UINT8_C( 88), UINT8_C(104), UINT8_C(162), UINT8_C(203),
                             UINT8_C(144), UINT8_C( 38), UINT8_C(193), UINT8_C(181),
                             UINT8_C(155), UINT8_C( 59), UINT8_C( 61), UINT8_C(  4),
                             UINT8_C( 63), UINT8_C(240), UINT8_C( 88), UINT8_C( 14),
                             UINT8_C( 73), UINT8_C(125), UINT8_C(224), UINT8_C(117),
                             UINT8_C(118), UINT8_C(109), UINT8_C( 68), UINT8_C( 42),
                             UINT8_C(150), UINT8_C( 79), UINT8_C(167), UINT8_C( 25),
                             UINT8_C( 58), UINT8_C(250), UINT8_C(130), UINT8_C(160),
                             UINT8_C( 75), UINT8_C(145), UINT8_C(152), UINT8_C(149),
                             UINT8_C(134), UINT8_C(252), UINT8_C( 13), UINT8_C(165),
                             UINT8_C(218), UINT8_C( 88), UINT8_C( 59), UINT8_C(228)),
      UINT64_C(17825505917769929051) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 80), UINT8_C(151), UINT8_C(216), UINT8_C(130),
                             UINT8_C(149), UINT8_C(124), UINT8_C( 37), UINT8_C( 84),
                             UINT8_C(103), UINT8_C( 99), UINT8_C(115), UINT8_C(151),
                             UINT8_C(233), UINT8_C(197), UINT8_C(132), UINT8_C(158),
                             UINT8_C( 23), UINT8_C( 54), UINT8_C(164), UINT8_C(107),
                             UINT8_C(233), UINT8_C(122), UINT8_C( 62), UINT8_C( 22),
                             UINT8_C(179), UINT8_C( 56), UINT8_C(117), UINT8_C(196),
                             UINT8_C(102), UINT8_C( 82), UINT8_C(  6), UINT8_C(242),
                             UINT8_C(100), UINT8_C(238), UINT8_C(103), UINT8_C( 83),
                             UINT8_C(139), UINT8_C(142), UINT8_C(174), UINT8_C(130),
                             UINT8_C(118), UINT8_C( 29), UINT8_C(246), UINT8_C(127),
                             UINT8_C(235), UINT8_C( 33), UINT8_C(253), UINT8_C(147),
                             UINT8_C( 41), UINT8_C( 14), UINT8_C(193), UINT8_C(126),
                             UINT8_C(220), UINT8_C(114), UINT8_C( 22), UINT8_C( 77),
                             UINT8_C( 40), UINT8_C(150), UINT8_C(218), UINT8_C(187),
                             UINT8_C(209), UINT8_C(123), UINT8_C( 46), UINT8_C(156)),
      easysimd_x_mm512_set_epu8(UINT8_C(176), UINT8_C( 45), UINT8_C(210), UINT8_C(149),
                             UINT8_C(149), UINT8_C(249), UINT8_C( 13), UINT8_C(137),
                             UINT8_C(118), UINT8_C(232), UINT8_C(127), UINT8_C( 30),
                             UINT8_C(175), UINT8_C(210), UINT8_C(248), UINT8_C(191),
                             UINT8_C( 96), UINT8_C( 79), UINT8_C(110), UINT8_C(154),
                             UINT8_C(119), UINT8_C(253), UINT8_C(133), UINT8_C( 16),
                             UINT8_C(243), UINT8_C(  4), UINT8_C(  4), UINT8_C(112),
                             UINT8_C(245), UINT8_C(173), UINT8_C( 10), UINT8_C(196),
                             UINT8_C(208), UINT8_C( 87), UINT8_C( 86), UINT8_C(157),
                             UINT8_C(215), UINT8_C( 65), UINT8_C(145), UINT8_C(212),
                             UINT8_C( 76), UINT8_C(163), UINT8_C( 24), UINT8_C(147),
                             UINT8_C( 61), UINT8_C(161), UINT8_C( 63), UINT8_C( 34),
                             UINT8_C(236), UINT8_C( 70), UINT8_C(243), UINT8_C(236),
                             UINT8_C( 49), UINT8_C( 99), UINT8_C(157), UINT8_C( 87),
                             UINT8_C(  4), UINT8_C( 86), UINT8_C(157), UINT8_C( 37),
                             UINT8_C( 44), UINT8_C( 59), UINT8_C(149), UINT8_C( 14)),
      UINT64_C(10801838139217605378) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_epu8_mask(test_vec[i].a, test_vec[i].b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cmplt_epu8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmplt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[32];
    uint16_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { { UINT16_C(23096), UINT16_C(34242), UINT16_C(18579), UINT16_C(20527), UINT16_C(50738), UINT16_C(44750), UINT16_C(40381), UINT16_C(20026),
        UINT16_C(22864), UINT16_C(59978), UINT16_C(16626), UINT16_C(55597), UINT16_C(15588), UINT16_C(58943), UINT16_C(28240), UINT16_C(34992),
        UINT16_C(29385), UINT16_C(23565), UINT16_C(15803), UINT16_C(60844), UINT16_C(31235), UINT16_C(49308), UINT16_C(54807), UINT16_C(26638),
        UINT16_C(22575), UINT16_C( 8530), UINT16_C(32664), UINT16_C(32251), UINT16_C(15036), UINT16_C( 3171), UINT16_C( 5033), UINT16_C(29332) },
      { UINT16_C(41350), UINT16_C(16846), UINT16_C(31454), UINT16_C(57646), UINT16_C(51957), UINT16_C( 3234), UINT16_C(45216), UINT16_C(53364),
        UINT16_C(50697), UINT16_C(41457), UINT16_C(60486), UINT16_C(  542), UINT16_C(33063), UINT16_C(53262), UINT16_C(41621), UINT16_C( 6978),
        UINT16_C( 4163), UINT16_C( 8796), UINT16_C(35466), UINT16_C(32515), UINT16_C(42325), UINT16_C(62860), UINT16_C(   86), UINT16_C(24517),
        UINT16_C(47047), UINT16_C( 3328), UINT16_C( 8099), UINT16_C(51727), UINT16_C( 7584), UINT16_C(13722), UINT16_C(56511), UINT16_C(  592) },
      UINT32_C(1765037533) },
    { { UINT16_C(44268), UINT16_C(30500), UINT16_C(10295), UINT16_C(36086), UINT16_C(33485), UINT16_C( 9089), UINT16_C(18307), UINT16_C(19074),
        UINT16_C(33790), UINT16_C(41303), UINT16_C(26274), UINT16_C(17004), UINT16_C( 1667), UINT16_C(17016), UINT16_C(51427), UINT16_C(53060),
        UINT16_C(26997), UINT16_C(44102), UINT16_C(15761), UINT16_C(24120), UINT16_C(47551), UINT16_C(17026), UINT16_C( 1024), UINT16_C(65164),
        UINT16_C(58247), UINT16_C(10656), UINT16_C( 3145), UINT16_C(52332), UINT16_C(58386), UINT16_C(62734), UINT16_C(21420), UINT16_C( 8645) },
      { UINT16_C( 3004), UINT16_C(19917), UINT16_C( 1352), UINT16_C( 2219), UINT16_C(11711), UINT16_C(48970), UINT16_C(55090), UINT16_C(47550),
        UINT16_C(24250), UINT16_C( 1251), UINT16_C(20330), UINT16_C(31952), UINT16_C(57139), UINT16_C(57202), UINT16_C(14130), UINT16_C(60929),
        UINT16_C(52802), UINT16_C(35643), UINT16_C(59092), UINT16_C(37779), UINT16_C(56596), UINT16_C(18002), UINT16_C( 4276), UINT16_C(28671),
        UINT16_C(57966), UINT16_C(55411), UINT16_C(17201), UINT16_C(25685), UINT16_C(50978), UINT16_C(21572), UINT16_C(17918), UINT16_C(16450) },
      UINT32_C(2256386272) },
    { { UINT16_C(32019), UINT16_C(59339), UINT16_C(24164), UINT16_C(30842), UINT16_C(52540), UINT16_C(61630), UINT16_C(48605), UINT16_C(19551),
        UINT16_C(53920), UINT16_C(53540), UINT16_C(30998), UINT16_C(14390), UINT16_C(31296), UINT16_C(16013), UINT16_C(53183), UINT16_C(53887),
        UINT16_C(19021), UINT16_C(45498), UINT16_C(13481), UINT16_C(58665), UINT16_C(59137), UINT16_C(57301), UINT16_C(13732), UINT16_C(17451),
        UINT16_C(20231), UINT16_C( 7446), UINT16_C(19657), UINT16_C( 2390), UINT16_C(58310), UINT16_C(34120), UINT16_C(51122), UINT16_C(65367) },
      { UINT16_C( 4369), UINT16_C(47792), UINT16_C(55622), UINT16_C(18335), UINT16_C(30144), UINT16_C(25894), UINT16_C(20906), UINT16_C(45481),
        UINT16_C(49057), UINT16_C(27343), UINT16_C( 9483), UINT16_C(53619), UINT16_C(47880), UINT16_C(47702), UINT16_C(44674), UINT16_C(38074),
        UINT16_C(27327), UINT16_C( 1358), UINT16_C(60996), UINT16_C( 1101), UINT16_C(29539), UINT16_C( 3433), UINT16_C( 5061), UINT16_C(26302),
        UINT16_C(36306), UINT16_C(57040), UINT16_C(17330), UINT16_C(47791), UINT16_C( 1791), UINT16_C(33141), UINT16_C(12212), UINT16_C(29461) },
      UINT32_C( 193280132) },
    { { UINT16_C(25753), UINT16_C(56697), UINT16_C(50770), UINT16_C(46562), UINT16_C(19257), UINT16_C(65218), UINT16_C(32862), UINT16_C(12644),
        UINT16_C(13326), UINT16_C(49167), UINT16_C(48760), UINT16_C(30587), UINT16_C(61636), UINT16_C(30968), UINT16_C( 3615), UINT16_C(47340),
        UINT16_C(25970), UINT16_C(50326), UINT16_C(30763), UINT16_C(25721), UINT16_C(15299), UINT16_C( 8803), UINT16_C(51131), UINT16_C(51539),
        UINT16_C(25340), UINT16_C(29834), UINT16_C( 1312), UINT16_C(58859), UINT16_C(58357), UINT16_C( 5213), UINT16_C(18929), UINT16_C(25548) },
      { UINT16_C(25262), UINT16_C(55591), UINT16_C(41178), UINT16_C(40510), UINT16_C(41435), UINT16_C(38848), UINT16_C( 4968), UINT16_C(25696),
        UINT16_C(60021), UINT16_C(38360), UINT16_C(50159), UINT16_C(58490), UINT16_C(55463), UINT16_C(39160), UINT16_C(50465), UINT16_C(53500),
        UINT16_C( 8999), UINT16_C(  681), UINT16_C(59332), UINT16_C(40864), UINT16_C(24712), UINT16_C(61750), UINT16_C(38771), UINT16_C(59477),
        UINT16_C(11905), UINT16_C(29053), UINT16_C(63729), UINT16_C(38997), UINT16_C(20176), UINT16_C(61745), UINT16_C(11539), UINT16_C(15041) },
      UINT32_C( 616361360) },
    { { UINT16_C(27472), UINT16_C( 5180), UINT16_C(56402), UINT16_C(56244), UINT16_C(59964), UINT16_C(45004), UINT16_C( 8577), UINT16_C(  919),
        UINT16_C( 5455), UINT16_C(16756), UINT16_C(51469), UINT16_C(56793), UINT16_C( 2583), UINT16_C(10958), UINT16_C(36919), UINT16_C(34917),
        UINT16_C(41467), UINT16_C(19868), UINT16_C(20606), UINT16_C(47656), UINT16_C(62523), UINT16_C(48234), UINT16_C(  278), UINT16_C(26047),
        UINT16_C(13078), UINT16_C( 9126), UINT16_C(33021), UINT16_C( 5120), UINT16_C(53130), UINT16_C(49727), UINT16_C(42079), UINT16_C(23114) },
      { UINT16_C(58949), UINT16_C(50087), UINT16_C(53303), UINT16_C(29310), UINT16_C(59588), UINT16_C(55854), UINT16_C(61161), UINT16_C(   64),
        UINT16_C(58913), UINT16_C( 7715), UINT16_C( 9318), UINT16_C(61747), UINT16_C(29427), UINT16_C(21171), UINT16_C(64790), UINT16_C(23468),
        UINT16_C(21475), UINT16_C( 6687), UINT16_C(40227), UINT16_C(59532), UINT16_C(48005), UINT16_C(28354), UINT16_C(  681), UINT16_C(51822),
        UINT16_C(37609), UINT16_C(20457), UINT16_C( 7350), UINT16_C(43328), UINT16_C(62350), UINT16_C(42235), UINT16_C(42992), UINT16_C(54527) },
      UINT32_C(3687610723) },
    { { UINT16_C( 7930), UINT16_C( 7918), UINT16_C(31675), UINT16_C(16390), UINT16_C(51254), UINT16_C(57263), UINT16_C( 7627), UINT16_C(46249),
        UINT16_C(37551), UINT16_C(25859), UINT16_C(17582), UINT16_C(15374), UINT16_C( 2359), UINT16_C(10464), UINT16_C(57520), UINT16_C(44028),
        UINT16_C(60158), UINT16_C(47817), UINT16_C(53093), UINT16_C(39930), UINT16_C(43415), UINT16_C(25210), UINT16_C( 9415), UINT16_C(30230),
        UINT16_C( 6838), UINT16_C(26076), UINT16_C(59998), UINT16_C(38305), UINT16_C(33524), UINT16_C(42173), UINT16_C(47458), UINT16_C(24655) },
      { UINT16_C( 6308), UINT16_C( 2330), UINT16_C( 5607), UINT16_C(32677), UINT16_C( 8126), UINT16_C(34273), UINT16_C(63555), UINT16_C(64252),
        UINT16_C(55314), UINT16_C(28767), UINT16_C(  194), UINT16_C(46597), UINT16_C(50050), UINT16_C(58459), UINT16_C(43644), UINT16_C( 8261),
        UINT16_C(24515), UINT16_C(43562), UINT16_C(53108), UINT16_C(13097), UINT16_C( 3054), UINT16_C(12984), UINT16_C(46083), UINT16_C( 5420),
        UINT16_C(35724), UINT16_C(20357), UINT16_C(35467), UINT16_C( 3589), UINT16_C(24653), UINT16_C(51954), UINT16_C(14091), UINT16_C(52970) },
      UINT32_C(2705603528) },
    { { UINT16_C( 5271), UINT16_C( 2936), UINT16_C(41699), UINT16_C(53822), UINT16_C(63405), UINT16_C(45060), UINT16_C(12459), UINT16_C(14533),
        UINT16_C(19131), UINT16_C(18055), UINT16_C(36052), UINT16_C( 8788), UINT16_C(18413), UINT16_C(63724), UINT16_C(54910), UINT16_C( 5574),
        UINT16_C(16107), UINT16_C(52769), UINT16_C(24544), UINT16_C(36256), UINT16_C(42070), UINT16_C(  573), UINT16_C(  724), UINT16_C(36666),
        UINT16_C(49484), UINT16_C( 8662), UINT16_C(10829), UINT16_C(14915), UINT16_C(12145), UINT16_C(61490), UINT16_C(63493), UINT16_C(61445) },
      { UINT16_C( 9783), UINT16_C( 6079), UINT16_C(24454), UINT16_C(56485), UINT16_C(57860), UINT16_C(55518), UINT16_C( 6373), UINT16_C(12648),
        UINT16_C(16089), UINT16_C(10066), UINT16_C(38248), UINT16_C(55905), UINT16_C(38084), UINT16_C(51914), UINT16_C(53132), UINT16_C(50106),
        UINT16_C(31222), UINT16_C(31963), UINT16_C(32985), UINT16_C(56664), UINT16_C(14178), UINT16_C(18357), UINT16_C( 7503), UINT16_C(10617),
        UINT16_C(52059), UINT16_C(50256), UINT16_C(45409), UINT16_C( 9630), UINT16_C(26693), UINT16_C(53999), UINT16_C(43575), UINT16_C(11669) },
      UINT32_C( 393059371) },
    { { UINT16_C(28707), UINT16_C(64681), UINT16_C(  752), UINT16_C(21465), UINT16_C(36665), UINT16_C(34970), UINT16_C( 5036), UINT16_C( 2225),
        UINT16_C(  479), UINT16_C(16588), UINT16_C(27315), UINT16_C(63589), UINT16_C(21970), UINT16_C( 2506), UINT16_C(24831), UINT16_C( 8759),
        UINT16_C(57552), UINT16_C(49439), UINT16_C(63714), UINT16_C( 6932), UINT16_C(44679), UINT16_C(13476), UINT16_C(21954), UINT16_C(41276),
        UINT16_C( 2135), UINT16_C( 2785), UINT16_C(18034), UINT16_C(17410), UINT16_C(52635), UINT16_C(39501), UINT16_C(33837), UINT16_C(64957) },
      { UINT16_C(56421), UINT16_C(18366), UINT16_C(53972), UINT16_C(23651), UINT16_C( 1921), UINT16_C(17296), UINT16_C(52316), UINT16_C(46052),
        UINT16_C(50644), UINT16_C(18109), UINT16_C(49163), UINT16_C(42890), UINT16_C(55181), UINT16_C(47681), UINT16_C(65116), UINT16_C(49591),
        UINT16_C(30426), UINT16_C(44808), UINT16_C(27464), UINT16_C(51467), UINT16_C(39794), UINT16_C(53004), UINT16_C(61543), UINT16_C(15234),
        UINT16_C(16565), UINT16_C(49537), UINT16_C( 2816), UINT16_C(36200), UINT16_C(43490), UINT16_C(15943), UINT16_C(65192), UINT16_C(33535) },
      UINT32_C(1265170381) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmplt_epu16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u16x32();
    easysimd__m512i b = easysimd_test_x86_random_u16x32();
    easysimd__mmask32 r = easysimd_mm512_cmplt_epu16_mask(a, b);

    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmplt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t a[16];
    uint32_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { { UINT32_C(3364886856), UINT32_C(3639632268), UINT32_C( 754606909), UINT32_C(2665678466), UINT32_C(1215757977), UINT32_C( 845653137), UINT32_C(3887092313), UINT32_C( 442719442),
        UINT32_C(  31650677), UINT32_C(2463748949), UINT32_C(3099513654), UINT32_C(1935057626), UINT32_C( 834391200), UINT32_C(3378717552), UINT32_C(1202721909), UINT32_C(3848410224) },
      { UINT32_C(1558594567), UINT32_C(1307492119), UINT32_C(1829088659), UINT32_C(4024458063), UINT32_C(2535496743), UINT32_C( 878805951), UINT32_C( 125571479), UINT32_C( 770498085),
        UINT32_C( 982110754), UINT32_C( 629635218), UINT32_C(1955761445), UINT32_C( 274952936), UINT32_C(3450307342), UINT32_C(2650933254), UINT32_C(1067810330), UINT32_C(2121044572) },
      UINT16_C( 4540) },
    { { UINT32_C(4139316836), UINT32_C(2468036718), UINT32_C(3037179341), UINT32_C( 784689696), UINT32_C(4093406701), UINT32_C(2408709749), UINT32_C(3637458812), UINT32_C( 760626121),
        UINT32_C(2669940529), UINT32_C( 473055055), UINT32_C( 231815661), UINT32_C(2419824547), UINT32_C(2038707972), UINT32_C(2970162485), UINT32_C( 361355340), UINT32_C(1162076179) },
      { UINT32_C(1055156207), UINT32_C(2472155046), UINT32_C(4104137552), UINT32_C(3330596034), UINT32_C(1228933139), UINT32_C(1794787614), UINT32_C( 880837665), UINT32_C(1400488804),
        UINT32_C(3499187754), UINT32_C(3311725429), UINT32_C(3652781078), UINT32_C(4104077024), UINT32_C(1681776454), UINT32_C(1238316840), UINT32_C( 528371643), UINT32_C(1014167314) },
      UINT16_C(20366) },
    { { UINT32_C(3389784917), UINT32_C(  93352174), UINT32_C(1440631157), UINT32_C(3460922759), UINT32_C(2234680925), UINT32_C(2043609534), UINT32_C(1654213968), UINT32_C(2594048836),
        UINT32_C(4251233039), UINT32_C(2416112667), UINT32_C(3320242238), UINT32_C(3146985310), UINT32_C(1950402230), UINT32_C( 418189255), UINT32_C(2709161564), UINT32_C(2705004946) },
      { UINT32_C(3751714756), UINT32_C(3530596756), UINT32_C(3751237249), UINT32_C( 999959429), UINT32_C(3098532849), UINT32_C(1204854250), UINT32_C(3051899683), UINT32_C( 676733796),
        UINT32_C(1443362242), UINT32_C( 388528022), UINT32_C(1408745677), UINT32_C(3700330987), UINT32_C(1469333100), UINT32_C(4271793627), UINT32_C( 347375280), UINT32_C(1799097001) },
      UINT16_C(10327) },
    { { UINT32_C(2512536575), UINT32_C(2293099195), UINT32_C(2514199722), UINT32_C(2725341749), UINT32_C(2214135464), UINT32_C( 461477739), UINT32_C(3324982557), UINT32_C(1060203328),
        UINT32_C(1775629230), UINT32_C(2297594590), UINT32_C(1528745254), UINT32_C(3757936439), UINT32_C(   6485653), UINT32_C(2853954701), UINT32_C(1500531225), UINT32_C(1687790261) },
      { UINT32_C(1942843029), UINT32_C( 385662960), UINT32_C(3295746701), UINT32_C(1067741097), UINT32_C(4080993893), UINT32_C(  77421546), UINT32_C(1532825253), UINT32_C(1170208432),
        UINT32_C(1421446244), UINT32_C(3647649100), UINT32_C(2023611599), UINT32_C(2964799819), UINT32_C( 849606472), UINT32_C(4147528018), UINT32_C(4283601999), UINT32_C(4014215562) },
      UINT16_C(63124) },
    { { UINT32_C(3930324382), UINT32_C(2177085106), UINT32_C(3589955722), UINT32_C(3934630306), UINT32_C(4196149672), UINT32_C(3119666026), UINT32_C(1907901671), UINT32_C(4099996758),
        UINT32_C(2900272378), UINT32_C(3694043474), UINT32_C(2746296321), UINT32_C(2190292697), UINT32_C(3380390495), UINT32_C(3833753341), UINT32_C( 156580531), UINT32_C( 838710839) },
      { UINT32_C(2900220762), UINT32_C(2106067836), UINT32_C( 220281139), UINT32_C(3482300015), UINT32_C(1436027736), UINT32_C( 758717306), UINT32_C(2352385877), UINT32_C(2679976773),
        UINT32_C(2320210702), UINT32_C(3657946022), UINT32_C(2078746636), UINT32_C( 793409239), UINT32_C(4219855745), UINT32_C(1395179262), UINT32_C(2464177741), UINT32_C(2670828945) },
      UINT16_C(53312) },
    { { UINT32_C(3744037944), UINT32_C(1538863439), UINT32_C( 819437657), UINT32_C(2539659542), UINT32_C(  43246852), UINT32_C(4048993187), UINT32_C(2877503002), UINT32_C( 206288339),
        UINT32_C(2179691569), UINT32_C(4292650149), UINT32_C(1513075524), UINT32_C(3656486869), UINT32_C( 417105012), UINT32_C(1510552128), UINT32_C( 990219368), UINT32_C(1934053441) },
      { UINT32_C(1794388677), UINT32_C( 443142358), UINT32_C(1500813700), UINT32_C(2637325864), UINT32_C( 716508906), UINT32_C(2827271744), UINT32_C(2363787850), UINT32_C(2684300250),
        UINT32_C( 873132894), UINT32_C(1196389315), UINT32_C( 899728140), UINT32_C( 349360937), UINT32_C( 574523361), UINT32_C(2412430149), UINT32_C( 656125517), UINT32_C( 952572634) },
      UINT16_C(12444) },
    { { UINT32_C(3513569549), UINT32_C(1360575300), UINT32_C(2827401599), UINT32_C(1841059980), UINT32_C( 613415903), UINT32_C( 196303550), UINT32_C(3794980616), UINT32_C(4145740265),
        UINT32_C( 264800202), UINT32_C(3244351554), UINT32_C( 627762841), UINT32_C( 496182846), UINT32_C(3745587745), UINT32_C(2246767740), UINT32_C(2909216195), UINT32_C(3785654806) },
      { UINT32_C(1274047497), UINT32_C(3859632204), UINT32_C(1946908470), UINT32_C(3213991581), UINT32_C(1033818817), UINT32_C(2328005062), UINT32_C(3157732006), UINT32_C(3047021484),
        UINT32_C(2466286919), UINT32_C( 326700509), UINT32_C( 579306884), UINT32_C(3839957027), UINT32_C(2971828202), UINT32_C(2923160584), UINT32_C(3127603726), UINT32_C(2490304589) },
      UINT16_C(26938) },
    { { UINT32_C(1931964566), UINT32_C(  42443133), UINT32_C(1227099686), UINT32_C( 288228647), UINT32_C(2378322052), UINT32_C(1128004916), UINT32_C(3170739823), UINT32_C(1162898863),
        UINT32_C(1522039005), UINT32_C(1063010073), UINT32_C(1971880014), UINT32_C( 176600709), UINT32_C( 999770374), UINT32_C(3061764678), UINT32_C( 678591353), UINT32_C(3312305128) },
      { UINT32_C(1411392827), UINT32_C(3012787301), UINT32_C(2183666685), UINT32_C(3633098450), UINT32_C(1041441783), UINT32_C(1878299126), UINT32_C(4120339981), UINT32_C(1689977897),
        UINT32_C(2394479401), UINT32_C(1413565271), UINT32_C( 970418535), UINT32_C( 252797720), UINT32_C(2102207879), UINT32_C(3287040438), UINT32_C(3518596008), UINT32_C(2956358791) },
      UINT16_C(31726) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmplt_epu32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u32x16();
    easysimd__m512i b = easysimd_test_x86_random_u32x16();
    easysimd__mmask16 r = easysimd_mm512_cmplt_epu32_mask(a, b);

    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmplt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint64_t a[8];
    uint64_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT64_C( 8952281287208663355), UINT64_C( 6201582920469320307), UINT64_C(17287045629148734784), UINT64_C(11682865639644334652),
        UINT64_C(16914433295593739226), UINT64_C( 2619695180373389622), UINT64_C(16538274091197119787), UINT64_C(15840179141866299851) },
      { UINT64_C( 1483048628160018994), UINT64_C(13497416245793360457), UINT64_C(13096418147543492913), UINT64_C( 1139616163889036787),
        UINT64_C(  893487077381499764), UINT64_C( 2676134286385053655), UINT64_C( 3986458921589043891), UINT64_C( 7487511400947595517) },
      UINT8_C( 34) },
    { { UINT64_C(14318949753981346279), UINT64_C( 3222696590962520223), UINT64_C( 8448220153334315459), UINT64_C( 5525133225964300628),
        UINT64_C( 2778348818837956569), UINT64_C(15006964883398507677), UINT64_C( 8942483274405973292), UINT64_C( 8732603513806481791) },
      { UINT64_C( 8180173908721346263), UINT64_C(16026847291420196748), UINT64_C(15455962868683522777), UINT64_C( 9880874156201492741),
        UINT64_C(13643211082731839761), UINT64_C( 9528423477893652395), UINT64_C( 6931606463094046212), UINT64_C(16753150748317379652) },
      UINT8_C(158) },
    { { UINT64_C( 9112228813282829958), UINT64_C(10534226904966024194), UINT64_C(18415254860374915029), UINT64_C( 4813068977107736076),
        UINT64_C(13703449522833299674), UINT64_C(17680178996998572091), UINT64_C( 8506936213274708564), UINT64_C( 5856972508533228363) },
      { UINT64_C(16949345234586649278), UINT64_C(10828642805588796467), UINT64_C( 7799024137353391195), UINT64_C( 8754922613227361718),
        UINT64_C(15895784367565537021), UINT64_C(13450130437840981712), UINT64_C(13290328113415147234), UINT64_C(11278250001104757584) },
      UINT8_C(219) },
    { { UINT64_C( 8315866628030625762), UINT64_C(17171149011277944370), UINT64_C(15507410325504969851), UINT64_C( 7026292167666768454),
        UINT64_C(16426128332991352031), UINT64_C( 2586686488132811498), UINT64_C( 9362092170827323277), UINT64_C( 1572011590430412783) },
      { UINT64_C(18178933902227783093), UINT64_C(  923756341983624454), UINT64_C( 3848903429751665071), UINT64_C( 7937722681471062492),
        UINT64_C( 5926205273530482481), UINT64_C(13215198515371717082), UINT64_C( 8051885912065410626), UINT64_C(  505716896797188828) },
      UINT8_C( 41) },
    { { UINT64_C(14918324354554648681), UINT64_C( 2051827075926984889), UINT64_C(  150698125917755924), UINT64_C(12439676934937942469),
        UINT64_C( 3186695255279987573), UINT64_C(  173784333821043058), UINT64_C( 3938610568007126657), UINT64_C( 5929397780082671706) },
      { UINT64_C( 9871596907321404254), UINT64_C(15296262087233563734), UINT64_C(  932847323392504139), UINT64_C(10778224648071529316),
        UINT64_C(11113587851208904606), UINT64_C(11059192850387177554), UINT64_C( 8994893627043214614), UINT64_C( 1069411879610055947) },
      UINT8_C(118) },
    { { UINT64_C(17689573791723765407), UINT64_C(11021176358420831245), UINT64_C(11285798088677017866), UINT64_C( 5490087700983602354),
        UINT64_C(18198111438842688243), UINT64_C( 5053964830736000777), UINT64_C( 7764215344564634515), UINT64_C( 5214814686822637412) },
      { UINT64_C( 5015720513836650160), UINT64_C(14850324259515986323), UINT64_C(16977079138532884342), UINT64_C(10803673598015645522),
        UINT64_C(14729531900619443689), UINT64_C(11802722055229225409), UINT64_C(14025172895938034992), UINT64_C(16433168827906891923) },
      UINT8_C(238) },
    { { UINT64_C(15216099714237219621), UINT64_C(12068410565510350251), UINT64_C( 5515505165615975768), UINT64_C( 8179051823811130530),
        UINT64_C(12105106230575335703), UINT64_C(13897539350616329662), UINT64_C(16162316518767017031), UINT64_C(18286583602214670857) },
      { UINT64_C( 5560295784934970759), UINT64_C(16868619034462169584), UINT64_C(13959248683274511003), UINT64_C(13559174319310255688),
        UINT64_C( 7371945591305097842), UINT64_C(14672682715203948562), UINT64_C(  644681871932098938), UINT64_C( 6378613190597937557) },
      UINT8_C( 46) },
    { { UINT64_C( 4580063158468199604), UINT64_C(11489688864309923448), UINT64_C(  929082342718380060), UINT64_C( 3461961329698740940),
        UINT64_C( 9551265148792628255), UINT64_C( 3601928711800914231), UINT64_C( 6718228875015212472), UINT64_C(10067827115644948491) },
      { UINT64_C( 3679270456425714303), UINT64_C(10604725723515075557), UINT64_C( 3767820919762991194), UINT64_C(17511250394242373901),
        UINT64_C( 2789854901914557575), UINT64_C( 3402552895153915850), UINT64_C(  351390686330419843), UINT64_C(11035086902087491004) },
      UINT8_C(140) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmplt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m512i b = easysimd_test_x86_random_u64x8();
    easysimd__mmask8 r = easysimd_mm512_cmplt_epu64_mask(a, b);

    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmplt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask64 k;
    uint8_t a[64];
    uint8_t b[64];
    easysimd__mmask64 r;
  } test_vec[8] = {
    { UINT64_C( 1593132328996063350),
      { UINT8_C( 82), UINT8_C(106), UINT8_C(220), UINT8_C(171), UINT8_C( 91), UINT8_C(133), UINT8_C(  0), UINT8_C( 13),
        UINT8_C( 94), UINT8_C(117), UINT8_C(177), UINT8_C(118), UINT8_C( 67), UINT8_C(204), UINT8_C(145), UINT8_C(189),
        UINT8_C(133), UINT8_C(100), UINT8_C(121), UINT8_C(201), UINT8_C(138), UINT8_C( 78), UINT8_C( 38), UINT8_C(  0),
        UINT8_C(150), UINT8_C( 39), UINT8_C(151),    UINT8_MAX, UINT8_C( 24), UINT8_C(178), UINT8_C( 21), UINT8_C(107),
        UINT8_C( 28), UINT8_C(241), UINT8_C( 22), UINT8_C(120), UINT8_C(119), UINT8_C( 22), UINT8_C(133), UINT8_C(213),
        UINT8_C(139), UINT8_C( 55), UINT8_C( 75), UINT8_C(206), UINT8_C(  3), UINT8_C(220), UINT8_C(140), UINT8_C(136),
        UINT8_C( 64), UINT8_C(  5), UINT8_C( 81), UINT8_C(202), UINT8_C( 83), UINT8_C(119), UINT8_C(203), UINT8_C(233),
        UINT8_C(158), UINT8_C( 98), UINT8_C(232), UINT8_C(183), UINT8_C( 20), UINT8_C(253), UINT8_C( 34), UINT8_C( 49) },
      { UINT8_C(239), UINT8_C( 56), UINT8_C(169), UINT8_C(102), UINT8_C( 78), UINT8_C( 46), UINT8_C( 59), UINT8_C(217),
        UINT8_C(101), UINT8_C(134), UINT8_C(168), UINT8_C(104), UINT8_C( 99), UINT8_C( 52), UINT8_C(240), UINT8_C(163),
        UINT8_C( 57), UINT8_C( 65), UINT8_C(110), UINT8_C(141), UINT8_C(185), UINT8_C( 57), UINT8_C(118), UINT8_C( 87),
        UINT8_C(155), UINT8_C( 95), UINT8_C( 14), UINT8_C(175), UINT8_C( 92), UINT8_C( 48), UINT8_C(224), UINT8_C( 75),
        UINT8_C(104), UINT8_C(137), UINT8_C(177), UINT8_C(182), UINT8_C(184), UINT8_C(236), UINT8_C(144), UINT8_C( 29),
        UINT8_C(115), UINT8_C( 56), UINT8_C(134), UINT8_C(214), UINT8_C(108), UINT8_C(118), UINT8_C(121), UINT8_C(165),
        UINT8_C(184), UINT8_C(231), UINT8_C( 50), UINT8_C(113), UINT8_C( 32), UINT8_C(169), UINT8_C(200), UINT8_C(187),
        UINT8_C(  8), UINT8_C(215), UINT8_C(107), UINT8_C(100), UINT8_C(  7), UINT8_C( 75), UINT8_C(176), UINT8_C(112) },
      UINT64_C(  145118393953960000) },
    { UINT64_C(13955166843852710357),
      { UINT8_C(238), UINT8_C( 48), UINT8_C(151), UINT8_C( 90), UINT8_C(167), UINT8_C( 16), UINT8_C(  0), UINT8_C( 95),
        UINT8_C(248), UINT8_C( 50), UINT8_C(208), UINT8_C( 24), UINT8_C(219), UINT8_C(152), UINT8_C(212), UINT8_C(227),
        UINT8_C(111), UINT8_C( 63), UINT8_C( 72), UINT8_C(119), UINT8_C(138), UINT8_C(248), UINT8_C(231), UINT8_C( 95),
        UINT8_C( 89), UINT8_C( 13), UINT8_C(236), UINT8_C(167), UINT8_C(196), UINT8_C(151), UINT8_C(104), UINT8_C(178),
        UINT8_C(199),    UINT8_MAX, UINT8_C( 13), UINT8_C(110), UINT8_C( 16), UINT8_C( 13), UINT8_C(205), UINT8_C(  8),
        UINT8_C( 63), UINT8_C(157), UINT8_C( 32), UINT8_C( 27), UINT8_C( 54), UINT8_C(244), UINT8_C(254), UINT8_C(165),
        UINT8_C( 51), UINT8_C( 70), UINT8_C( 28), UINT8_C(190), UINT8_C( 62), UINT8_C(  3), UINT8_C( 29), UINT8_C(152),
        UINT8_C( 17), UINT8_C( 10), UINT8_C( 63), UINT8_C(213), UINT8_C(161), UINT8_C(168), UINT8_C(135), UINT8_C(104) },
      { UINT8_C(167), UINT8_C(148), UINT8_C(215), UINT8_C(183), UINT8_C(161), UINT8_C(164), UINT8_C(191), UINT8_C(225),
        UINT8_C( 66), UINT8_C(224), UINT8_C(252), UINT8_C(120), UINT8_C(212), UINT8_C(250), UINT8_C( 29), UINT8_C(  8),
        UINT8_C( 65), UINT8_C( 58), UINT8_C(198), UINT8_C(127), UINT8_C( 61), UINT8_C(227), UINT8_C( 23), UINT8_C( 78),
        UINT8_C(237), UINT8_C( 87), UINT8_C( 35), UINT8_C(142),    UINT8_MAX, UINT8_C(171), UINT8_C(247), UINT8_C(166),
        UINT8_C( 63), UINT8_C(206), UINT8_C( 94), UINT8_C(225), UINT8_C(114), UINT8_C( 29), UINT8_C(194), UINT8_C(180),
        UINT8_C(253), UINT8_C(190), UINT8_C( 44), UINT8_C(210), UINT8_C(184), UINT8_C( 74), UINT8_C(218), UINT8_C(249),
        UINT8_C(132), UINT8_C(160), UINT8_C(121), UINT8_C(193), UINT8_C(131), UINT8_C(144), UINT8_C( 16), UINT8_C(113),
        UINT8_C(231), UINT8_C( 51),    UINT8_MAX, UINT8_C(230), UINT8_C(222), UINT8_C(246), UINT8_C(141), UINT8_C( 30) },
      UINT64_C( 4695730539787985092) },
    { UINT64_C(  498704424797989828),
      { UINT8_C(127), UINT8_C( 24), UINT8_C(216), UINT8_C( 55), UINT8_C( 98), UINT8_C(178), UINT8_C( 49), UINT8_C(230),
        UINT8_C( 82), UINT8_C(170), UINT8_C(167), UINT8_C(213), UINT8_C( 58), UINT8_C(183), UINT8_C( 70), UINT8_C( 34),
        UINT8_C(235), UINT8_C( 70), UINT8_C(  8), UINT8_C(201), UINT8_C( 60), UINT8_C(149), UINT8_C(231), UINT8_C(  1),
        UINT8_C(128), UINT8_C(230), UINT8_C( 56), UINT8_C(137), UINT8_C(167), UINT8_C( 35), UINT8_C(143), UINT8_C( 38),
        UINT8_C( 59), UINT8_C(103), UINT8_C( 94), UINT8_C(157), UINT8_C( 25), UINT8_C(143), UINT8_C(131), UINT8_C(107),
        UINT8_C( 57), UINT8_C( 43), UINT8_C( 64), UINT8_C(115), UINT8_C(226), UINT8_C(135), UINT8_C(149), UINT8_C(205),
        UINT8_C(205), UINT8_C(158), UINT8_C(151), UINT8_C(  9), UINT8_C( 51), UINT8_C(126), UINT8_C( 10), UINT8_C(180),
        UINT8_C(101), UINT8_C( 66), UINT8_C( 61), UINT8_C( 12), UINT8_C(102), UINT8_C(204), UINT8_C( 51), UINT8_C(161) },
      { UINT8_C( 51), UINT8_C(145), UINT8_C( 63), UINT8_C( 76), UINT8_C( 32), UINT8_C(194), UINT8_C(183), UINT8_C( 89),
        UINT8_C(237), UINT8_C(247), UINT8_C(204), UINT8_C(208), UINT8_C(126), UINT8_C( 98), UINT8_C(157), UINT8_C( 75),
        UINT8_C(  0), UINT8_C( 52), UINT8_C( 85), UINT8_C( 51), UINT8_C(179), UINT8_C( 95), UINT8_C(231), UINT8_C( 24),
        UINT8_C(162), UINT8_C( 36), UINT8_C( 36), UINT8_C(  8), UINT8_C(240), UINT8_C( 87), UINT8_C(169), UINT8_C( 35),
        UINT8_C(232), UINT8_C(232), UINT8_C(111), UINT8_C(  8), UINT8_C(171), UINT8_C( 38), UINT8_C( 97), UINT8_C(152),
        UINT8_C( 30), UINT8_C( 46), UINT8_C(104), UINT8_C(156), UINT8_C(144), UINT8_C(  6), UINT8_C(232), UINT8_C(144),
        UINT8_C( 58), UINT8_C( 61), UINT8_C(195), UINT8_C(237), UINT8_C(156), UINT8_C(171), UINT8_C(  5), UINT8_C( 62),
        UINT8_C(207), UINT8_C( 42), UINT8_C( 70), UINT8_C(192), UINT8_C(129), UINT8_C(240), UINT8_C(227), UINT8_C(106) },
      UINT64_C(  299559744796148544) },
    { UINT64_C(10888811617436455896),
      { UINT8_C(  2), UINT8_C(132), UINT8_C( 52), UINT8_C(146), UINT8_C(138), UINT8_C( 28), UINT8_C( 34), UINT8_C(197),
        UINT8_C( 89), UINT8_C(229), UINT8_C(178), UINT8_C(245), UINT8_C(144), UINT8_C(184), UINT8_C( 52), UINT8_C( 96),
        UINT8_C(226), UINT8_C(122), UINT8_C( 32), UINT8_C( 99), UINT8_C(106), UINT8_C(  3), UINT8_C(205), UINT8_C( 67),
        UINT8_C( 86), UINT8_C( 64), UINT8_C(198), UINT8_C(208), UINT8_C( 20), UINT8_C(226), UINT8_C(103), UINT8_C( 22),
        UINT8_C(103), UINT8_C(155), UINT8_C(168), UINT8_C(241), UINT8_C(183), UINT8_C(202), UINT8_C(182), UINT8_C( 16),
        UINT8_C(175), UINT8_C(105), UINT8_C(  6), UINT8_C( 64), UINT8_C( 33), UINT8_C( 58), UINT8_C(160), UINT8_C(  3),
        UINT8_C(180), UINT8_C(192), UINT8_C(102), UINT8_C( 31), UINT8_C(195), UINT8_C( 52), UINT8_C( 98), UINT8_C( 26),
        UINT8_C(116), UINT8_C( 40), UINT8_C(234), UINT8_C(136), UINT8_C( 11), UINT8_C( 81), UINT8_C(158), UINT8_C(114) },
      { UINT8_C(237), UINT8_C( 70), UINT8_C( 99), UINT8_C(164), UINT8_C( 16), UINT8_C( 26), UINT8_C(181), UINT8_C(191),
        UINT8_C(131), UINT8_C(187),    UINT8_MAX, UINT8_C(164), UINT8_C(245), UINT8_C(159), UINT8_C(167), UINT8_C(169),
        UINT8_C( 95), UINT8_C( 13), UINT8_C(200), UINT8_C( 35), UINT8_C( 65), UINT8_C( 42), UINT8_C( 61), UINT8_C(181),
        UINT8_C( 83), UINT8_C( 39), UINT8_C( 61), UINT8_C( 94), UINT8_C(120), UINT8_C(219), UINT8_C(208), UINT8_C(101),
        UINT8_C( 33), UINT8_C( 51), UINT8_C( 10), UINT8_C( 49), UINT8_C( 77), UINT8_C(191), UINT8_C(241), UINT8_C(208),
        UINT8_C(122), UINT8_C(240), UINT8_C(116), UINT8_C(111), UINT8_C(144), UINT8_C( 27), UINT8_C( 24), UINT8_C(239),
        UINT8_C( 41), UINT8_C(225), UINT8_C( 18), UINT8_C(106), UINT8_C( 11), UINT8_C( 79), UINT8_C( 32), UINT8_C( 94),
        UINT8_C(118), UINT8_C( 93), UINT8_C(188), UINT8_C(239), UINT8_C( 57), UINT8_C(140), UINT8_C( 84), UINT8_C( 90) },
      UINT64_C( 1371509091282735432) },
    { UINT64_C(10943321708291710656),
      { UINT8_C(109), UINT8_C( 82), UINT8_C(  6), UINT8_C(253), UINT8_C(110), UINT8_C( 31), UINT8_C(237), UINT8_C(151),
        UINT8_C(  0),    UINT8_MAX, UINT8_C(  1), UINT8_C( 11), UINT8_C( 79), UINT8_C( 33), UINT8_C(106), UINT8_C(197),
        UINT8_C(127), UINT8_C( 38), UINT8_C(180), UINT8_C(184), UINT8_C(179), UINT8_C(  9), UINT8_C( 18), UINT8_C(115),
        UINT8_C(103), UINT8_C(158), UINT8_C(128), UINT8_C(133), UINT8_C( 27), UINT8_C( 94), UINT8_C( 28), UINT8_C(137),
        UINT8_C(177), UINT8_C( 35), UINT8_C(134), UINT8_C( 31), UINT8_C( 66), UINT8_C(115), UINT8_C(182), UINT8_C( 66),
        UINT8_C(115), UINT8_C(183), UINT8_C( 77), UINT8_C(194), UINT8_C(217), UINT8_C(183), UINT8_C(135), UINT8_C( 88),
        UINT8_C(222), UINT8_C( 60), UINT8_C( 16), UINT8_C(145), UINT8_C( 69), UINT8_C( 34), UINT8_C(  4), UINT8_C(172),
        UINT8_C(193), UINT8_C(132), UINT8_C( 49), UINT8_C(220), UINT8_C(227), UINT8_C( 78), UINT8_C(101), UINT8_C(148) },
      { UINT8_C(113), UINT8_C(236), UINT8_C(179), UINT8_C(179), UINT8_C( 95), UINT8_C(105), UINT8_C(245), UINT8_C(210),
        UINT8_C( 32), UINT8_C( 66), UINT8_C(148), UINT8_C(249), UINT8_C(250), UINT8_C( 28), UINT8_C( 81), UINT8_C(216),
        UINT8_C( 88), UINT8_C( 97), UINT8_C(105), UINT8_C(157), UINT8_C(132), UINT8_C(109), UINT8_C( 73), UINT8_C( 69),
        UINT8_C(241), UINT8_C(123), UINT8_C( 33), UINT8_C(212), UINT8_C(201), UINT8_C(135), UINT8_C(104), UINT8_C( 58),
        UINT8_C(115), UINT8_C( 27), UINT8_C(237), UINT8_C(210), UINT8_C(132), UINT8_C(226), UINT8_C(165), UINT8_C(165),
        UINT8_C( 36), UINT8_C( 57), UINT8_C(158), UINT8_C( 30), UINT8_C( 85), UINT8_C(240), UINT8_C(246), UINT8_C(173),
        UINT8_C( 81), UINT8_C( 95), UINT8_C( 74), UINT8_C(213), UINT8_C(204), UINT8_C(148), UINT8_C( 26), UINT8_C(190),
        UINT8_C( 15), UINT8_C( 60), UINT8_C(146), UINT8_C(216), UINT8_C(195), UINT8_C(251), UINT8_C( 18), UINT8_C( 54) },
      UINT64_C(  350827892554341568) },
    { UINT64_C(  378493553179426582),
      { UINT8_C(231), UINT8_C(222), UINT8_C( 36), UINT8_C( 60), UINT8_C(206), UINT8_C( 26), UINT8_C(234), UINT8_C( 32),
        UINT8_C(122), UINT8_C( 52), UINT8_C(245), UINT8_C( 70), UINT8_C(200), UINT8_C( 16), UINT8_C(  4), UINT8_C(215),
        UINT8_C( 76), UINT8_C(151), UINT8_C(175), UINT8_C( 15), UINT8_C(146), UINT8_C(193), UINT8_C( 69), UINT8_C(168),
        UINT8_C(192), UINT8_C( 77), UINT8_C( 67), UINT8_C(161), UINT8_C(251), UINT8_C(131), UINT8_C(167), UINT8_C(226),
        UINT8_C( 98), UINT8_C(203), UINT8_C( 30), UINT8_C( 48), UINT8_C(229), UINT8_C(  8), UINT8_C( 80), UINT8_C( 95),
        UINT8_C( 61), UINT8_C( 70), UINT8_C(166), UINT8_C(  5), UINT8_C( 86), UINT8_C(170), UINT8_C(221), UINT8_C(162),
        UINT8_C( 65), UINT8_C(140), UINT8_C(177), UINT8_C(211), UINT8_C( 78), UINT8_C(246), UINT8_C(124), UINT8_C( 14),
        UINT8_C( 67), UINT8_C(191), UINT8_C(176), UINT8_C( 62), UINT8_C( 67), UINT8_C( 87), UINT8_C( 32), UINT8_C(165) },
      { UINT8_C( 34), UINT8_C( 63), UINT8_C(213), UINT8_C(  7), UINT8_C( 71), UINT8_C( 38), UINT8_C(103), UINT8_C(132),
        UINT8_C(108), UINT8_C( 13), UINT8_C(138), UINT8_C(194), UINT8_C(183), UINT8_C(103), UINT8_C(100), UINT8_C(249),
        UINT8_C(243), UINT8_C( 21), UINT8_C(204), UINT8_C( 65), UINT8_C( 11), UINT8_C( 72), UINT8_C( 80), UINT8_C( 78),
        UINT8_C(  8), UINT8_C(  0), UINT8_C(141), UINT8_C( 75), UINT8_C( 87), UINT8_C(173), UINT8_C(240), UINT8_C(121),
        UINT8_C(236), UINT8_C(197), UINT8_C(128), UINT8_C( 52), UINT8_C(235), UINT8_C(231), UINT8_C(184), UINT8_C( 87),
        UINT8_C(244), UINT8_C( 66), UINT8_C( 25), UINT8_C(172), UINT8_C(169), UINT8_C(125), UINT8_C(165), UINT8_C(157),
        UINT8_C(146), UINT8_C(113), UINT8_C(222), UINT8_C(157), UINT8_C(186), UINT8_C( 46), UINT8_C(236), UINT8_C(194),
        UINT8_C( 46), UINT8_C(121), UINT8_C( 13), UINT8_C(133), UINT8_C( 38), UINT8_C(253), UINT8_C(254), UINT8_C( 19) },
      UINT64_C(   18024710726543364) },
    { UINT64_C( 6558929248810663874),
      { UINT8_C( 66), UINT8_C( 31), UINT8_C(  7), UINT8_C(235), UINT8_C(156), UINT8_C(172), UINT8_C(136), UINT8_C( 47),
        UINT8_C( 29), UINT8_C(103), UINT8_C(204), UINT8_C(215), UINT8_C(149), UINT8_C(184), UINT8_C(153), UINT8_C(196),
        UINT8_C( 49), UINT8_C(166), UINT8_C( 73), UINT8_C( 88), UINT8_C(163), UINT8_C( 72), UINT8_C(107), UINT8_C(102),
        UINT8_C(199), UINT8_C(178), UINT8_C( 20), UINT8_C( 45), UINT8_C(177), UINT8_C( 25), UINT8_C(136), UINT8_C(243),
        UINT8_C( 56), UINT8_C(143), UINT8_C(223), UINT8_C(213), UINT8_C( 59), UINT8_C(103), UINT8_C(  4), UINT8_C( 89),
        UINT8_C(206), UINT8_C(208), UINT8_C( 48), UINT8_C(100), UINT8_C(137), UINT8_C(202), UINT8_C( 40), UINT8_C(186),
        UINT8_C(112), UINT8_C(113), UINT8_C( 18), UINT8_C( 20), UINT8_C(185), UINT8_C(125), UINT8_C(122), UINT8_C(128),
        UINT8_C( 47), UINT8_C(142), UINT8_C(174), UINT8_C(225), UINT8_C(167), UINT8_C( 54), UINT8_C(212), UINT8_C(224) },
      { UINT8_C(198), UINT8_C(179), UINT8_C(181), UINT8_C(  1), UINT8_C( 27), UINT8_C(185), UINT8_C( 90), UINT8_C(233),
        UINT8_C(137), UINT8_C(139), UINT8_C( 77), UINT8_C( 18), UINT8_C( 85), UINT8_C(117), UINT8_C(205), UINT8_C(197),
        UINT8_C(231), UINT8_C(223), UINT8_C(217), UINT8_C(160), UINT8_C( 93), UINT8_C( 83), UINT8_C( 33), UINT8_C(140),
        UINT8_C(225), UINT8_C(207), UINT8_C(109), UINT8_C(137), UINT8_C(  5), UINT8_C( 66), UINT8_C(105), UINT8_C(203),
        UINT8_C(245), UINT8_C( 30), UINT8_C(205), UINT8_C( 16), UINT8_C(215), UINT8_C( 39), UINT8_C(250), UINT8_C( 96),
        UINT8_C(178), UINT8_C( 71), UINT8_C(115), UINT8_C(  7), UINT8_C(189), UINT8_C( 64), UINT8_C(205), UINT8_C(164),
        UINT8_C( 31), UINT8_C(166), UINT8_C( 68), UINT8_C(124), UINT8_C(250), UINT8_C(101), UINT8_C(  9), UINT8_C(219),
        UINT8_C( 52), UINT8_C(118), UINT8_C(100), UINT8_C( 58), UINT8_C(184), UINT8_C(205), UINT8_C(  5), UINT8_C(174) },
      UINT64_C( 1226197633178485634) },
    { UINT64_C(12403961185857098475),
      { UINT8_C(  0), UINT8_C(150), UINT8_C(180), UINT8_C(189), UINT8_C(214), UINT8_C(129), UINT8_C( 97), UINT8_C(245),
        UINT8_C( 39), UINT8_C(165), UINT8_C(114), UINT8_C( 33), UINT8_C( 11), UINT8_C(123), UINT8_C(253), UINT8_C( 63),
        UINT8_C(241), UINT8_C( 97), UINT8_C(121), UINT8_C(170), UINT8_C( 47), UINT8_C(127), UINT8_C( 88), UINT8_C( 26),
        UINT8_C( 81), UINT8_C( 22), UINT8_C(221), UINT8_C( 75), UINT8_C(207), UINT8_C(  0), UINT8_C(248), UINT8_C(207),
        UINT8_C(150), UINT8_C(172), UINT8_C(140), UINT8_C(108), UINT8_C( 45), UINT8_C(237), UINT8_C( 97), UINT8_C( 84),
        UINT8_C(146), UINT8_C(211), UINT8_C(118), UINT8_C(157), UINT8_C( 78), UINT8_C(115), UINT8_C(221), UINT8_C( 64),
        UINT8_C(212), UINT8_C( 86), UINT8_C(234), UINT8_C(  3), UINT8_C(213), UINT8_C( 66), UINT8_C( 30), UINT8_C( 39),
        UINT8_C( 88), UINT8_C(251), UINT8_C(114), UINT8_C( 39), UINT8_C(251), UINT8_C(106), UINT8_C(246), UINT8_C(145) },
      { UINT8_C( 22), UINT8_C(130), UINT8_C(253), UINT8_C( 67), UINT8_C(111), UINT8_C( 94), UINT8_C(152), UINT8_C(  2),
        UINT8_C( 50), UINT8_C( 14), UINT8_C(159), UINT8_C(128), UINT8_C(129), UINT8_C(124), UINT8_C(192), UINT8_C( 85),
        UINT8_C(211), UINT8_C(170), UINT8_C( 89), UINT8_C(168), UINT8_C(236), UINT8_C(119), UINT8_C(207), UINT8_C( 69),
        UINT8_C(114), UINT8_C( 66), UINT8_C(108), UINT8_C(109), UINT8_C(172), UINT8_C( 99), UINT8_C(254), UINT8_C(195),
        UINT8_C(229), UINT8_C(251), UINT8_C(  6), UINT8_C( 85), UINT8_C( 89), UINT8_C(158), UINT8_C( 87), UINT8_C(139),
        UINT8_C(172), UINT8_C(246), UINT8_C( 12), UINT8_C( 45), UINT8_C(115), UINT8_C(204), UINT8_C(131), UINT8_C( 70),
        UINT8_C(119), UINT8_C(220), UINT8_C(238), UINT8_C( 99), UINT8_C( 83), UINT8_C(190), UINT8_C(168), UINT8_C(197),
        UINT8_C(  0), UINT8_C( 21), UINT8_C( 50), UINT8_C(172), UINT8_C(120), UINT8_C( 48), UINT8_C(111), UINT8_C( 93) },
      UINT64_C(  586225043740201025) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epu8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epu8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i b = easysimd_test_x86_random_i8x64();
    easysimd__mmask64 r = easysimd_mm512_mask_cmplt_epu8_mask(k, a, b);

    easysimd_test_x86_write_mmask64(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmplt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    uint16_t a[32];
    uint16_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C(4127245949),
      { UINT16_C(18764), UINT16_C( 2351), UINT16_C(19442), UINT16_C(34549), UINT16_C(28179), UINT16_C(26108), UINT16_C(26366), UINT16_C(46068),
        UINT16_C(61026), UINT16_C(10366), UINT16_C(41393), UINT16_C( 5765), UINT16_C(37920), UINT16_C(40242), UINT16_C(13146), UINT16_C(42643),
        UINT16_C(49788), UINT16_C(28336), UINT16_C(42254), UINT16_C( 8692), UINT16_C(61459), UINT16_C( 4487), UINT16_C(31574), UINT16_C(47300),
        UINT16_C(17257), UINT16_C( 7136), UINT16_C(26340), UINT16_C( 1073), UINT16_C(25850), UINT16_C(21921), UINT16_C(13463), UINT16_C( 5115) },
      { UINT16_C(44023), UINT16_C( 1409), UINT16_C(30032), UINT16_C(25638), UINT16_C(44389), UINT16_C(47989), UINT16_C(14889), UINT16_C(37491),
        UINT16_C(21373), UINT16_C(25005), UINT16_C(57273), UINT16_C(46181), UINT16_C( 1603), UINT16_C(55817), UINT16_C( 1082), UINT16_C(12781),
        UINT16_C(28336), UINT16_C(   54), UINT16_C(24035), UINT16_C(18532), UINT16_C(55818), UINT16_C(13059), UINT16_C(30228), UINT16_C(37318),
        UINT16_C(29641), UINT16_C(33778), UINT16_C(22354), UINT16_C(38199), UINT16_C(16477), UINT16_C(38767), UINT16_C(23620), UINT16_C(62665) },
      UINT32_C(3791652405) },
    { UINT32_C(2918580170),
      { UINT16_C(22876), UINT16_C(26613), UINT16_C(63539), UINT16_C(18330), UINT16_C(24686), UINT16_C(14552), UINT16_C(51924), UINT16_C( 9915),
        UINT16_C(61985), UINT16_C(32444), UINT16_C(11058), UINT16_C(30230), UINT16_C(57224), UINT16_C(21099), UINT16_C(24798), UINT16_C(15104),
        UINT16_C(62905), UINT16_C(60834), UINT16_C(15598), UINT16_C(23604), UINT16_C( 3485), UINT16_C(29076), UINT16_C(20439), UINT16_C(63895),
        UINT16_C(21313), UINT16_C(29559), UINT16_C(36223), UINT16_C( 2026), UINT16_C(21868), UINT16_C(19289), UINT16_C(22965), UINT16_C(28294) },
      { UINT16_C(10319), UINT16_C(15707), UINT16_C(36964), UINT16_C(  409), UINT16_C(11933), UINT16_C(29810), UINT16_C( 2685), UINT16_C(49005),
        UINT16_C(58717), UINT16_C(56370), UINT16_C( 7282), UINT16_C(57315), UINT16_C(15729), UINT16_C( 9770), UINT16_C(45206), UINT16_C(58773),
        UINT16_C(61656), UINT16_C(15394), UINT16_C(48256), UINT16_C( 7486), UINT16_C(45290), UINT16_C(26514), UINT16_C(65466), UINT16_C( 6182),
        UINT16_C(23012), UINT16_C(22516), UINT16_C(55413), UINT16_C(59190), UINT16_C(24597), UINT16_C(43789), UINT16_C(41488), UINT16_C(59537) },
      UINT32_C(2908015232) },
    { UINT32_C( 321172371),
      { UINT16_C(25199), UINT16_C(22833), UINT16_C(49939), UINT16_C(52673), UINT16_C(59330), UINT16_C(42981), UINT16_C(55872), UINT16_C(46846),
        UINT16_C(13490), UINT16_C(51101), UINT16_C(43668), UINT16_C(42098), UINT16_C(  845), UINT16_C(57484), UINT16_C(45239), UINT16_C( 9971),
        UINT16_C( 9235), UINT16_C( 9856), UINT16_C(16871), UINT16_C(43763), UINT16_C(55592), UINT16_C(26961), UINT16_C(20403), UINT16_C(25887),
        UINT16_C(48259), UINT16_C( 5932), UINT16_C(40550), UINT16_C(46011), UINT16_C(18338), UINT16_C(22931), UINT16_C(34807), UINT16_C( 2687) },
      { UINT16_C(65451), UINT16_C(37680), UINT16_C( 9280), UINT16_C(26941), UINT16_C(36605), UINT16_C(45266), UINT16_C(61917), UINT16_C(24597),
        UINT16_C(16813), UINT16_C( 4983), UINT16_C(13023), UINT16_C(33223), UINT16_C(23161), UINT16_C(28890), UINT16_C(23265), UINT16_C(36219),
        UINT16_C(43865), UINT16_C(39456), UINT16_C(24015), UINT16_C(52227), UINT16_C(54763), UINT16_C(51324), UINT16_C(37318), UINT16_C(29480),
        UINT16_C(40914), UINT16_C(45702), UINT16_C(19921), UINT16_C(18995), UINT16_C( 3752), UINT16_C(35258), UINT16_C(13672), UINT16_C(49430) },
      UINT32_C(  35950851) },
    { UINT32_C(2958767841),
      { UINT16_C(24211), UINT16_C(32381), UINT16_C(63795), UINT16_C(63814), UINT16_C(28299), UINT16_C(23916), UINT16_C(62221), UINT16_C(56847),
        UINT16_C(17216), UINT16_C(59432), UINT16_C(58193), UINT16_C(47474), UINT16_C(34840), UINT16_C(63866), UINT16_C(54975), UINT16_C(21162),
        UINT16_C(10036), UINT16_C(26833), UINT16_C( 5920), UINT16_C(43873), UINT16_C(52870), UINT16_C(37641), UINT16_C( 6337), UINT16_C(  370),
        UINT16_C(39515), UINT16_C(44266), UINT16_C(23677), UINT16_C(38501), UINT16_C(57572), UINT16_C(41871), UINT16_C(14774), UINT16_C(60150) },
      { UINT16_C(51040), UINT16_C(33106), UINT16_C(46302), UINT16_C(25644), UINT16_C(13698), UINT16_C(17400), UINT16_C(27214), UINT16_C(43332),
        UINT16_C(11780), UINT16_C(33366), UINT16_C(48010), UINT16_C(28440), UINT16_C(42907), UINT16_C(20754), UINT16_C( 2273), UINT16_C(16700),
        UINT16_C(36559), UINT16_C(44738), UINT16_C(61250), UINT16_C(50194), UINT16_C( 2596), UINT16_C(29191), UINT16_C(19572), UINT16_C(31004),
        UINT16_C(29306), UINT16_C( 1531), UINT16_C( 4909), UINT16_C(51572), UINT16_C(34490), UINT16_C(39706), UINT16_C(22159), UINT16_C(24285) },
      UINT32_C(   4919297) },
    { UINT32_C( 655138789),
      { UINT16_C( 8078), UINT16_C(46060), UINT16_C(62249), UINT16_C(40485), UINT16_C(16703), UINT16_C(47639), UINT16_C( 4787), UINT16_C(57791),
        UINT16_C(13093), UINT16_C(57258), UINT16_C(50361), UINT16_C(18555), UINT16_C(22555), UINT16_C(  167), UINT16_C(46071), UINT16_C(34343),
        UINT16_C( 5074), UINT16_C(64569), UINT16_C(24071), UINT16_C(18074), UINT16_C(45472), UINT16_C(21248), UINT16_C(49091), UINT16_C(59444),
        UINT16_C(57074), UINT16_C(44231), UINT16_C(17059), UINT16_C(48884), UINT16_C(39834), UINT16_C(37566), UINT16_C(58703), UINT16_C( 8472) },
      { UINT16_C(20985), UINT16_C(   29), UINT16_C(47023), UINT16_C(20294), UINT16_C(18280), UINT16_C(11171), UINT16_C(55046), UINT16_C(63763),
        UINT16_C(56246), UINT16_C(22949), UINT16_C(39197), UINT16_C(47127), UINT16_C(54581), UINT16_C(33866), UINT16_C(25274), UINT16_C(45989),
        UINT16_C(50099), UINT16_C(25267), UINT16_C(64122), UINT16_C(58290), UINT16_C(21825), UINT16_C(18190), UINT16_C( 8748), UINT16_C(57920),
        UINT16_C(58877), UINT16_C( 6715), UINT16_C(21119), UINT16_C(46290), UINT16_C( 7207), UINT16_C(57912), UINT16_C(56702), UINT16_C(12693) },
      UINT32_C( 621582785) },
    { UINT32_C( 462703008),
      { UINT16_C(17987), UINT16_C(34046), UINT16_C( 3227), UINT16_C(51147), UINT16_C( 3118), UINT16_C(11178), UINT16_C(58865), UINT16_C(28742),
        UINT16_C( 6200), UINT16_C(24356), UINT16_C(23605), UINT16_C(45889), UINT16_C(55098), UINT16_C(56037), UINT16_C(31008), UINT16_C(25589),
        UINT16_C(62399), UINT16_C(23271), UINT16_C(45568), UINT16_C(11809), UINT16_C(52158), UINT16_C(45146), UINT16_C(41137), UINT16_C(59680),
        UINT16_C(17848), UINT16_C(60744), UINT16_C(35489), UINT16_C(56225), UINT16_C(34401), UINT16_C(33206), UINT16_C(44031), UINT16_C(48868) },
      { UINT16_C(52127), UINT16_C(40728), UINT16_C(14717), UINT16_C(15565), UINT16_C( 9989), UINT16_C(46828), UINT16_C( 3271), UINT16_C(32927),
        UINT16_C(59217), UINT16_C(62317), UINT16_C( 3697), UINT16_C(53966), UINT16_C(33940), UINT16_C(37715), UINT16_C(14128), UINT16_C(53073),
        UINT16_C(26882), UINT16_C(32878), UINT16_C(15267), UINT16_C(43196), UINT16_C(43107), UINT16_C(10846), UINT16_C(64948), UINT16_C( 1706),
        UINT16_C( 6372), UINT16_C(22265), UINT16_C(50982), UINT16_C(47912), UINT16_C(31820), UINT16_C(31822), UINT16_C(41139), UINT16_C(46667) },
      UINT32_C(      2464) },
    { UINT32_C(2889267465),
      { UINT16_C(62196), UINT16_C(22356), UINT16_C(45722), UINT16_C(20098), UINT16_C(11439), UINT16_C(37972), UINT16_C(19780), UINT16_C(27626),
        UINT16_C( 4629), UINT16_C(24870), UINT16_C(29838), UINT16_C(17117), UINT16_C(10260), UINT16_C( 7928), UINT16_C(12001), UINT16_C(54730),
        UINT16_C( 7968), UINT16_C(47661), UINT16_C(45009), UINT16_C(33032), UINT16_C(24027), UINT16_C( 8213), UINT16_C(65450), UINT16_C(49035),
        UINT16_C(45329), UINT16_C(40992), UINT16_C(64805), UINT16_C(15074), UINT16_C(55845), UINT16_C( 1624), UINT16_C( 8712), UINT16_C(10460) },
      { UINT16_C( 2369), UINT16_C( 5090), UINT16_C(60088), UINT16_C(37780), UINT16_C(43335), UINT16_C(62131), UINT16_C(16040), UINT16_C(47537),
        UINT16_C(53999), UINT16_C( 5465), UINT16_C(15311), UINT16_C(62799), UINT16_C(42773), UINT16_C( 7675), UINT16_C(55241), UINT16_C( 2885),
        UINT16_C(10208), UINT16_C(38942), UINT16_C(45586), UINT16_C(22828), UINT16_C(57179), UINT16_C(  843), UINT16_C(64798), UINT16_C( 3516),
        UINT16_C( 5839), UINT16_C(40482), UINT16_C(29009), UINT16_C(26515), UINT16_C(36632), UINT16_C(57988), UINT16_C(51814), UINT16_C(18413) },
      UINT32_C(2819889416) },
    { UINT32_C(  64949233),
      { UINT16_C( 3005), UINT16_C( 6237), UINT16_C(43243), UINT16_C( 2331), UINT16_C(55205), UINT16_C(29718), UINT16_C(14829), UINT16_C(16147),
        UINT16_C(42666), UINT16_C(50086), UINT16_C(10805), UINT16_C(40101), UINT16_C(37620), UINT16_C(59107), UINT16_C(49821), UINT16_C(23273),
        UINT16_C(18126), UINT16_C(47474), UINT16_C(36335), UINT16_C(38082), UINT16_C(55396), UINT16_C(21001), UINT16_C( 7185), UINT16_C(48273),
        UINT16_C(14274), UINT16_C(63615), UINT16_C( 9313), UINT16_C(22164), UINT16_C(30646), UINT16_C(21308), UINT16_C( 9529), UINT16_C( 1965) },
      { UINT16_C( 8044), UINT16_C(23488), UINT16_C(33452), UINT16_C( 4335), UINT16_C(63579), UINT16_C(27746), UINT16_C(62228), UINT16_C(55080),
        UINT16_C(42794), UINT16_C(36047), UINT16_C(25547), UINT16_C(33250), UINT16_C( 7898), UINT16_C( 5076), UINT16_C(33091), UINT16_C(44827),
        UINT16_C(56224), UINT16_C(19466), UINT16_C(64094), UINT16_C(47453), UINT16_C(49138), UINT16_C( 1829), UINT16_C(20147), UINT16_C(56798),
        UINT16_C(44533), UINT16_C(49513), UINT16_C(19216), UINT16_C(59970), UINT16_C( 5993), UINT16_C(44541), UINT16_C( 6296), UINT16_C(14684) },
      UINT32_C(  30212561) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epu16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epu16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 r = easysimd_mm512_mask_cmplt_epu16_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmplt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    uint32_t a[16];
    uint32_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(17086),
      { UINT32_C(1046577759), UINT32_C(2958473057), UINT32_C(1514670123), UINT32_C( 162294849), UINT32_C( 575154332), UINT32_C( 613517629), UINT32_C(2581735170), UINT32_C( 282829233),
        UINT32_C(2219719715), UINT32_C( 523544052), UINT32_C(4000939437), UINT32_C(2247566825), UINT32_C(2326216525), UINT32_C(3400480968), UINT32_C( 291737951), UINT32_C(1428241970) },
      { UINT32_C(1876586619), UINT32_C(3264089621), UINT32_C(1957693323), UINT32_C(2046404396), UINT32_C(2936250599), UINT32_C( 947565273), UINT32_C(1984552260), UINT32_C(2546756380),
        UINT32_C(4026967771), UINT32_C(1085445300), UINT32_C(3350487707), UINT32_C(4030770697), UINT32_C( 664814414), UINT32_C( 979376886), UINT32_C( 330344951), UINT32_C(4020927508) },
      UINT16_C(17086) },
    { UINT16_C(45090),
      { UINT32_C(2437208031), UINT32_C(3421757207), UINT32_C(3866754470), UINT32_C(2368391405), UINT32_C(1336353007), UINT32_C( 184131418), UINT32_C(1569131954), UINT32_C(3691883005),
        UINT32_C(2540589696), UINT32_C(3630391601), UINT32_C(1287576670), UINT32_C(2497308836), UINT32_C(1659076872), UINT32_C(3530415136), UINT32_C(3878679786), UINT32_C( 516111773) },
      { UINT32_C(3233100175), UINT32_C(4053276819), UINT32_C(2570934261), UINT32_C(1210914623), UINT32_C(3098153368), UINT32_C(3616151533), UINT32_C(2847848971), UINT32_C(2261222135),
        UINT32_C(1179024563), UINT32_C(2318983061), UINT32_C(1965258038), UINT32_C( 616386956), UINT32_C(1339844706), UINT32_C(2334615423), UINT32_C( 406119713), UINT32_C( 463403879) },
      UINT16_C(   34) },
    { UINT16_C(58487),
      { UINT32_C(2579696737), UINT32_C(3121609110), UINT32_C( 738958191), UINT32_C(2626973120), UINT32_C(3808629948), UINT32_C(3553174687), UINT32_C(3670945852), UINT32_C(2914993483),
        UINT32_C(3896934994), UINT32_C(3953284476), UINT32_C(2971119089), UINT32_C(3612191770), UINT32_C(1606046144), UINT32_C(2989654901), UINT32_C(4287365300), UINT32_C(2544651077) },
      { UINT32_C(1249965006), UINT32_C( 959783496), UINT32_C(3941223632), UINT32_C(3099670776), UINT32_C(4262952073), UINT32_C(3014674943), UINT32_C(2410888521), UINT32_C(1445355400),
        UINT32_C(2610996818), UINT32_C(2580862409), UINT32_C( 444841762), UINT32_C(2161329655), UINT32_C(3229543105), UINT32_C(2104766260), UINT32_C(4094437228), UINT32_C(3645584262) },
      UINT16_C(32788) },
    { UINT16_C(60377),
      { UINT32_C(1220649588), UINT32_C(3204965179), UINT32_C(3489988349), UINT32_C(4256941438), UINT32_C(4180471685), UINT32_C(2032179308), UINT32_C(3618416524), UINT32_C(4106454400),
        UINT32_C(1664910376), UINT32_C(1679967335), UINT32_C(3241485890), UINT32_C(1908338923), UINT32_C(1265298143), UINT32_C( 247761794), UINT32_C(3001446450), UINT32_C( 514238966) },
      { UINT32_C(2491540013), UINT32_C(1777902630), UINT32_C(3056217546), UINT32_C(4230473757), UINT32_C(1414042066), UINT32_C(1315048476), UINT32_C(1929398397), UINT32_C( 512861937),
        UINT32_C(2947682953), UINT32_C(2165877686), UINT32_C(4130816728), UINT32_C(4243742250), UINT32_C( 206584559), UINT32_C(3294278471), UINT32_C(3979828219), UINT32_C(2316027905) },
      UINT16_C(60161) },
    { UINT16_C(48858),
      { UINT32_C(1382650170), UINT32_C(1234518290), UINT32_C( 715636535), UINT32_C( 207918780), UINT32_C(4257196962), UINT32_C(2790833007), UINT32_C(3010353576), UINT32_C( 510740964),
        UINT32_C(3966819034), UINT32_C(1395983644), UINT32_C(2172509381), UINT32_C( 361620082), UINT32_C(4245835149), UINT32_C(2963499528), UINT32_C(2808287939), UINT32_C( 902157659) },
      { UINT32_C(3407951535), UINT32_C(   2053691), UINT32_C(2759957554), UINT32_C( 213454718), UINT32_C(1678363484), UINT32_C(4178881589), UINT32_C( 429946814), UINT32_C(4232996428),
        UINT32_C(3620171676), UINT32_C(4158187205), UINT32_C(  27023747), UINT32_C(3289208168), UINT32_C(1445467681), UINT32_C(2169453763), UINT32_C(  10219700), UINT32_C(4076661078) },
      UINT16_C(35464) },
    { UINT16_C(50265),
      { UINT32_C(2712280777), UINT32_C(3002805526), UINT32_C(1007117103), UINT32_C(1347627048), UINT32_C(3465352831), UINT32_C( 851329431), UINT32_C(1042027585), UINT32_C(3489821958),
        UINT32_C(2842799251), UINT32_C( 156986586), UINT32_C(4165297104), UINT32_C( 172529803), UINT32_C(1188681134), UINT32_C(1484298006), UINT32_C(2996212908), UINT32_C(2642581513) },
      { UINT32_C( 507966532), UINT32_C( 807903840), UINT32_C(2418568453), UINT32_C(3030085893), UINT32_C(1576694854), UINT32_C(3082121739), UINT32_C( 275401479), UINT32_C( 665709795),
        UINT32_C(1095169248), UINT32_C(2607902102), UINT32_C(3760954074), UINT32_C(1368704523), UINT32_C(1185844794), UINT32_C( 134046464), UINT32_C(2434295726), UINT32_C( 884589908) },
      UINT16_C(    8) },
    { UINT16_C(65465),
      { UINT32_C(3882635125), UINT32_C( 377571306), UINT32_C(3151793191), UINT32_C(2353600477), UINT32_C(1525631325), UINT32_C(1774362193), UINT32_C(3895334447), UINT32_C(3236423754),
        UINT32_C( 581391415), UINT32_C(3258460315), UINT32_C(2440893620), UINT32_C(2283652651), UINT32_C(1625492751), UINT32_C(3670648235), UINT32_C(  96729275), UINT32_C( 398830304) },
      { UINT32_C(2570677502), UINT32_C(1213952404), UINT32_C(2983909510), UINT32_C(2923100062), UINT32_C(2936937732), UINT32_C(2106251458), UINT32_C(2961329616), UINT32_C(4140255479),
        UINT32_C(1234108852), UINT32_C(4170312562), UINT32_C(1655335875), UINT32_C(1729160291), UINT32_C(3273072129), UINT32_C(3326124534), UINT32_C(3849765614), UINT32_C(3218816522) },
      UINT16_C(54200) },
    { UINT16_C(27455),
      { UINT32_C(2572595464), UINT32_C(1409620394), UINT32_C(2335729787), UINT32_C(3869915599), UINT32_C(1015521532), UINT32_C(3724440935), UINT32_C( 907741531), UINT32_C(3500235464),
        UINT32_C(3060397836), UINT32_C(2349493777), UINT32_C(2786607830), UINT32_C(2005713531), UINT32_C(3383956578), UINT32_C(3836195209), UINT32_C(2182857402), UINT32_C( 693287965) },
      { UINT32_C(3319774388), UINT32_C(  22145322), UINT32_C(2795989291), UINT32_C(2384278315), UINT32_C(3512193095), UINT32_C(1018560129), UINT32_C(3737047233), UINT32_C(1090982285),
        UINT32_C(4144424653), UINT32_C(4210579407), UINT32_C(3969949633), UINT32_C( 444251603), UINT32_C( 250335885), UINT32_C(2437587408), UINT32_C(4268689777), UINT32_C(3879695898) },
      UINT16_C(17173) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epu32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epu32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 r = easysimd_mm512_mask_cmplt_epu32_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmplt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint64_t a[8];
    uint64_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(120),
      { UINT64_C(18297607425600780018), UINT64_C( 2339875610453729655), UINT64_C(15266767446147919348), UINT64_C( 7399714853628421590),
        UINT64_C(14773934698008751111), UINT64_C(10666978718657853552), UINT64_C( 8616639125641309349), UINT64_C( 7022064378605291672) },
      { UINT64_C( 1419040306324182159), UINT64_C( 5791351677906580010), UINT64_C(   20448063614219311), UINT64_C(16908904830973311469),
        UINT64_C(12558710341242917523), UINT64_C( 8605771163411956850), UINT64_C( 1167434109766041291), UINT64_C( 5966987755947245448) },
      UINT8_C(  8) },
    { UINT8_C( 18),
      { UINT64_C( 1697932728765680029), UINT64_C(10979728726687657404), UINT64_C(  375431293333901853), UINT64_C( 9598603362803390755),
        UINT64_C(14666309802949594015), UINT64_C(15750424619508557793), UINT64_C( 4547715448770547537), UINT64_C(10827102065153728724) },
      { UINT64_C(17887331349875450978), UINT64_C(12398278224502889592), UINT64_C( 2752507865351821015), UINT64_C(10181551528236717247),
        UINT64_C( 6707894925733549063), UINT64_C( 7700771165925739478), UINT64_C( 4661383400863408440), UINT64_C(13608544045511381539) },
      UINT8_C(  2) },
    { UINT8_C(150),
      { UINT64_C(18224618471562140377), UINT64_C(11189187751118314604), UINT64_C( 1372011848997327191), UINT64_C( 1021336576909138259),
        UINT64_C(17099055634753693790), UINT64_C( 3296145464759147560), UINT64_C(10063046215950719507), UINT64_C( 5961162765738536824) },
      { UINT64_C( 1253107807824305397), UINT64_C(13100675142058794954), UINT64_C( 8759484786928116428), UINT64_C(17556967575993626720),
        UINT64_C(  123793149379823805), UINT64_C(12286505731848530732), UINT64_C( 2147440480687672402), UINT64_C(10438525710253552621) },
      UINT8_C(134) },
    { UINT8_C( 95),
      { UINT64_C(15654679353543243335), UINT64_C( 6699774669600787768), UINT64_C(  952852282739346703), UINT64_C( 5196996747336950696),
        UINT64_C( 9543017988175476136), UINT64_C(10850621239440713385), UINT64_C(10204061528473382516), UINT64_C(10847038373546982719) },
      { UINT64_C( 3022368900326941728), UINT64_C(13321920669951401410), UINT64_C( 2096956546981741479), UINT64_C(17430309144956331359),
        UINT64_C( 8030850166210564553), UINT64_C( 8460182112534386284), UINT64_C(17390960647130427229), UINT64_C( 7296433500578625152) },
      UINT8_C( 78) },
    { UINT8_C( 52),
      { UINT64_C(17896268229758791584), UINT64_C( 2594421336315399178), UINT64_C( 5537358906515602749), UINT64_C(  881947323669162469),
        UINT64_C( 8797596536176058365), UINT64_C( 6340259145796541796), UINT64_C( 4485734049271231069), UINT64_C( 8297747580763711664) },
      { UINT64_C(16036941664909140900), UINT64_C( 3168199662431088007), UINT64_C(15996327576568047207), UINT64_C( 8030235671106504327),
        UINT64_C( 3790924424262386755), UINT64_C( 8050408671036134591), UINT64_C(13726459526422452083), UINT64_C(15829639867886499700) },
      UINT8_C( 36) },
    { UINT8_C( 49),
      { UINT64_C( 9872192977648982321), UINT64_C( 2172988227706851048), UINT64_C(13678297686222424113), UINT64_C(17378881617494610639),
        UINT64_C(  686857358993984969), UINT64_C( 5488170624746396355), UINT64_C( 6967770738852593326), UINT64_C(   18447843662235536) },
      { UINT64_C( 7908751587957241239), UINT64_C( 1371805019836809121), UINT64_C(14055940267694404639), UINT64_C(10302641621104356546),
        UINT64_C( 2441522196276246986), UINT64_C( 1945528633227369094), UINT64_C( 9624995883872028571), UINT64_C(14324958634734233158) },
      UINT8_C( 16) },
    { UINT8_C(115),
      { UINT64_C( 8547855303204616722), UINT64_C(10553494734594215155), UINT64_C( 8968596321406895189), UINT64_C(10539898122131714905),
        UINT64_C( 4686923737594607992), UINT64_C( 6770757455495003352), UINT64_C( 9089956358203984852), UINT64_C( 6037241168743407286) },
      { UINT64_C(11051538542074819791), UINT64_C(16057006168941033839), UINT64_C( 2008798283786904125), UINT64_C(10843570345966907725),
        UINT64_C(14332443405810099341), UINT64_C( 1400922771403236234), UINT64_C(17795224790139075189), UINT64_C(17214053414135023495) },
      UINT8_C( 83) },
    { UINT8_C(254),
      { UINT64_C(  446450252536691863), UINT64_C( 3594699573452129882), UINT64_C(17889653081618693222), UINT64_C( 1047165492094669551),
        UINT64_C( 6029451259595447683), UINT64_C(16075777514361883055), UINT64_C(12828749567455130462), UINT64_C( 3168217092625031538) },
      { UINT64_C( 5183219415064358561), UINT64_C(12708955088831295239), UINT64_C( 5526137585669306938), UINT64_C( 5908208899109284906),
        UINT64_C( 6105433370557992923), UINT64_C( 7677286274722752906), UINT64_C( 4491240103202999086), UINT64_C(10377717089901250819) },
      UINT8_C(154) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmplt_epu64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmplt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 r = easysimd_mm512_mask_cmplt_epu64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

#if !defined(EASYSIMD_BUG_GCC_96174)
static int
test_easysimd_mm512_cmplt_ps_mask (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -679.30), EASYSIMD_FLOAT32_C(   966.54), EASYSIMD_FLOAT32_C(    -8.95), EASYSIMD_FLOAT32_C(   958.36),
        EASYSIMD_FLOAT32_C(  -725.44), EASYSIMD_FLOAT32_C(  -760.28), EASYSIMD_FLOAT32_C(   751.05), EASYSIMD_FLOAT32_C(   763.86),
        EASYSIMD_FLOAT32_C(  -137.17), EASYSIMD_FLOAT32_C(  -526.42), EASYSIMD_FLOAT32_C(  -580.26), EASYSIMD_FLOAT32_C(    72.73),
        EASYSIMD_FLOAT32_C(   848.96), EASYSIMD_FLOAT32_C(  -167.99), EASYSIMD_FLOAT32_C(    95.30), EASYSIMD_FLOAT32_C(   277.78) },
      { EASYSIMD_FLOAT32_C(   425.87), EASYSIMD_FLOAT32_C(  -693.81), EASYSIMD_FLOAT32_C(   225.64), EASYSIMD_FLOAT32_C(  -374.81),
        EASYSIMD_FLOAT32_C(  -490.07), EASYSIMD_FLOAT32_C(    62.38), EASYSIMD_FLOAT32_C(   630.88), EASYSIMD_FLOAT32_C(   308.80),
        EASYSIMD_FLOAT32_C(  -539.84), EASYSIMD_FLOAT32_C(  -683.39), EASYSIMD_FLOAT32_C(  -735.32), EASYSIMD_FLOAT32_C(  -999.26),
        EASYSIMD_FLOAT32_C(   848.96), EASYSIMD_FLOAT32_C(   579.92), EASYSIMD_FLOAT32_C(   118.33), EASYSIMD_FLOAT32_C(  -830.34) },
      UINT16_C(24629) },
    { { EASYSIMD_FLOAT32_C(  -833.10), EASYSIMD_FLOAT32_C(   667.53), EASYSIMD_FLOAT32_C(  -768.58), EASYSIMD_FLOAT32_C(    27.81),
        EASYSIMD_FLOAT32_C(   969.40), EASYSIMD_FLOAT32_C(  -884.80), EASYSIMD_FLOAT32_C(  -758.63), EASYSIMD_FLOAT32_C(  -724.10),
        EASYSIMD_FLOAT32_C(  -716.35), EASYSIMD_FLOAT32_C(  -476.71), EASYSIMD_FLOAT32_C(   419.04), EASYSIMD_FLOAT32_C(   832.05),
        EASYSIMD_FLOAT32_C(   151.35), EASYSIMD_FLOAT32_C(  -175.30), EASYSIMD_FLOAT32_C(    66.61), EASYSIMD_FLOAT32_C(   351.20) },
      { EASYSIMD_FLOAT32_C(   118.05), EASYSIMD_FLOAT32_C(  -502.75), EASYSIMD_FLOAT32_C(  -814.79), EASYSIMD_FLOAT32_C(   929.98),
        EASYSIMD_FLOAT32_C(   432.78), EASYSIMD_FLOAT32_C(  -886.46), EASYSIMD_FLOAT32_C(   577.10), EASYSIMD_FLOAT32_C(  -862.12),
        EASYSIMD_FLOAT32_C(   136.94), EASYSIMD_FLOAT32_C(   908.37), EASYSIMD_FLOAT32_C(  -807.53), EASYSIMD_FLOAT32_C(  -626.26),
        EASYSIMD_FLOAT32_C(    93.44), EASYSIMD_FLOAT32_C(   143.01), EASYSIMD_FLOAT32_C(   933.29), EASYSIMD_FLOAT32_C(   260.34) },
      UINT16_C(25417) },
    { { EASYSIMD_FLOAT32_C(   397.82), EASYSIMD_FLOAT32_C(    82.73), EASYSIMD_FLOAT32_C(  -728.93), EASYSIMD_FLOAT32_C(  -716.45),
        EASYSIMD_FLOAT32_C(   278.34), EASYSIMD_FLOAT32_C(  -422.65), EASYSIMD_FLOAT32_C(  -540.28), EASYSIMD_FLOAT32_C(   265.15),
        EASYSIMD_FLOAT32_C(   279.24), EASYSIMD_FLOAT32_C(  -171.08), EASYSIMD_FLOAT32_C(  -468.61), EASYSIMD_FLOAT32_C(   443.34),
        EASYSIMD_FLOAT32_C(   751.73), EASYSIMD_FLOAT32_C(  -744.43), EASYSIMD_FLOAT32_C(   566.91), EASYSIMD_FLOAT32_C(  -904.35) },
      { EASYSIMD_FLOAT32_C(  -692.80), EASYSIMD_FLOAT32_C(    82.73), EASYSIMD_FLOAT32_C(   507.25), EASYSIMD_FLOAT32_C(  -716.45),
        EASYSIMD_FLOAT32_C(  -871.32), EASYSIMD_FLOAT32_C(   909.91), EASYSIMD_FLOAT32_C(  -907.02), EASYSIMD_FLOAT32_C(  -102.77),
        EASYSIMD_FLOAT32_C(   677.37), EASYSIMD_FLOAT32_C(  -171.08), EASYSIMD_FLOAT32_C(  -468.61), EASYSIMD_FLOAT32_C(  -257.08),
        EASYSIMD_FLOAT32_C(   751.73), EASYSIMD_FLOAT32_C(   841.70), EASYSIMD_FLOAT32_C(   271.27), EASYSIMD_FLOAT32_C(   149.55) },
      UINT16_C(41252) },
    { { EASYSIMD_FLOAT32_C(  -351.59), EASYSIMD_FLOAT32_C(  -757.31), EASYSIMD_FLOAT32_C(  -739.49), EASYSIMD_FLOAT32_C(   354.82),
        EASYSIMD_FLOAT32_C(   779.77), EASYSIMD_FLOAT32_C(   796.84), EASYSIMD_FLOAT32_C(   253.65), EASYSIMD_FLOAT32_C(  -980.02),
        EASYSIMD_FLOAT32_C(  -824.56), EASYSIMD_FLOAT32_C(  -806.24), EASYSIMD_FLOAT32_C(   218.91), EASYSIMD_FLOAT32_C(   807.03),
        EASYSIMD_FLOAT32_C(  -499.44), EASYSIMD_FLOAT32_C(   683.75), EASYSIMD_FLOAT32_C(   242.90), EASYSIMD_FLOAT32_C(   681.31) },
      { EASYSIMD_FLOAT32_C(   698.06), EASYSIMD_FLOAT32_C(   143.17), EASYSIMD_FLOAT32_C(   645.90), EASYSIMD_FLOAT32_C(   354.82),
        EASYSIMD_FLOAT32_C(   561.25), EASYSIMD_FLOAT32_C(  -928.28), EASYSIMD_FLOAT32_C(   482.94), EASYSIMD_FLOAT32_C(    28.55),
        EASYSIMD_FLOAT32_C(   701.67), EASYSIMD_FLOAT32_C(   834.16), EASYSIMD_FLOAT32_C(   386.75), EASYSIMD_FLOAT32_C(   807.03),
        EASYSIMD_FLOAT32_C(   558.03), EASYSIMD_FLOAT32_C(  -756.03), EASYSIMD_FLOAT32_C(   930.12), EASYSIMD_FLOAT32_C(  -793.56) },
      UINT16_C(22471) },
    { { EASYSIMD_FLOAT32_C(   434.87), EASYSIMD_FLOAT32_C(  -355.05), EASYSIMD_FLOAT32_C(  -653.48), EASYSIMD_FLOAT32_C(   594.11),
        EASYSIMD_FLOAT32_C(   799.49), EASYSIMD_FLOAT32_C(   264.31), EASYSIMD_FLOAT32_C(    -8.19), EASYSIMD_FLOAT32_C(  -922.96),
        EASYSIMD_FLOAT32_C(   308.23), EASYSIMD_FLOAT32_C(  -871.48), EASYSIMD_FLOAT32_C(   543.66), EASYSIMD_FLOAT32_C(   721.18),
        EASYSIMD_FLOAT32_C(  -314.45), EASYSIMD_FLOAT32_C(   897.43), EASYSIMD_FLOAT32_C(   646.34), EASYSIMD_FLOAT32_C(  -691.19) },
      { EASYSIMD_FLOAT32_C(  -506.84), EASYSIMD_FLOAT32_C(  -355.05), EASYSIMD_FLOAT32_C(    70.02), EASYSIMD_FLOAT32_C(  -186.22),
        EASYSIMD_FLOAT32_C(   745.56), EASYSIMD_FLOAT32_C(  -329.15), EASYSIMD_FLOAT32_C(  -306.53), EASYSIMD_FLOAT32_C(  -665.08),
        EASYSIMD_FLOAT32_C(   -81.67), EASYSIMD_FLOAT32_C(   690.25), EASYSIMD_FLOAT32_C(  -343.01), EASYSIMD_FLOAT32_C(   742.59),
        EASYSIMD_FLOAT32_C(  -989.44), EASYSIMD_FLOAT32_C(   198.45), EASYSIMD_FLOAT32_C(   334.24), EASYSIMD_FLOAT32_C(   445.42) },
      UINT16_C(35460) },
    { { EASYSIMD_FLOAT32_C(    72.70), EASYSIMD_FLOAT32_C(  -926.98), EASYSIMD_FLOAT32_C(   386.60), EASYSIMD_FLOAT32_C(  -166.44),
        EASYSIMD_FLOAT32_C(  -372.12), EASYSIMD_FLOAT32_C(   156.01), EASYSIMD_FLOAT32_C(  -432.45), EASYSIMD_FLOAT32_C(  -171.34),
        EASYSIMD_FLOAT32_C(  -100.09), EASYSIMD_FLOAT32_C(   220.75), EASYSIMD_FLOAT32_C(  -427.23), EASYSIMD_FLOAT32_C(  -735.37),
        EASYSIMD_FLOAT32_C(   440.82), EASYSIMD_FLOAT32_C(  -646.62), EASYSIMD_FLOAT32_C(   895.12), EASYSIMD_FLOAT32_C(   585.45) },
      { EASYSIMD_FLOAT32_C(   -15.73), EASYSIMD_FLOAT32_C(   536.94), EASYSIMD_FLOAT32_C(  -374.81), EASYSIMD_FLOAT32_C(   158.91),
        EASYSIMD_FLOAT32_C(   525.00), EASYSIMD_FLOAT32_C(   478.37), EASYSIMD_FLOAT32_C(  -432.45), EASYSIMD_FLOAT32_C(  -483.69),
        EASYSIMD_FLOAT32_C(   887.57), EASYSIMD_FLOAT32_C(   220.75), EASYSIMD_FLOAT32_C(   709.30), EASYSIMD_FLOAT32_C(   187.04),
        EASYSIMD_FLOAT32_C(  -436.07), EASYSIMD_FLOAT32_C(   329.70), EASYSIMD_FLOAT32_C(    57.53), EASYSIMD_FLOAT32_C(   636.63) },
      UINT16_C(44346) },
    { { EASYSIMD_FLOAT32_C(  -715.67), EASYSIMD_FLOAT32_C(  -253.10), EASYSIMD_FLOAT32_C(   805.99), EASYSIMD_FLOAT32_C(   896.48),
        EASYSIMD_FLOAT32_C(  -683.44), EASYSIMD_FLOAT32_C(  -642.77), EASYSIMD_FLOAT32_C(  -746.45), EASYSIMD_FLOAT32_C(   318.24),
        EASYSIMD_FLOAT32_C(  -949.63), EASYSIMD_FLOAT32_C(  -203.63), EASYSIMD_FLOAT32_C(  -894.66), EASYSIMD_FLOAT32_C(   648.89),
        EASYSIMD_FLOAT32_C(   110.40), EASYSIMD_FLOAT32_C(   662.12), EASYSIMD_FLOAT32_C(   821.38), EASYSIMD_FLOAT32_C(   820.81) },
      { EASYSIMD_FLOAT32_C(   147.48), EASYSIMD_FLOAT32_C(   715.61), EASYSIMD_FLOAT32_C(  -594.01), EASYSIMD_FLOAT32_C(   128.99),
        EASYSIMD_FLOAT32_C(   847.91), EASYSIMD_FLOAT32_C(  -246.50), EASYSIMD_FLOAT32_C(  -172.62), EASYSIMD_FLOAT32_C(   927.56),
        EASYSIMD_FLOAT32_C(  -949.63), EASYSIMD_FLOAT32_C(  -193.40), EASYSIMD_FLOAT32_C(   284.28), EASYSIMD_FLOAT32_C(   354.14),
        EASYSIMD_FLOAT32_C(  -296.72), EASYSIMD_FLOAT32_C(   320.79), EASYSIMD_FLOAT32_C(   108.95), EASYSIMD_FLOAT32_C(   -12.38) },
      UINT16_C( 1779) },
    { { EASYSIMD_FLOAT32_C(   372.34), EASYSIMD_FLOAT32_C(   943.17), EASYSIMD_FLOAT32_C(  -546.38), EASYSIMD_FLOAT32_C(  -534.61),
        EASYSIMD_FLOAT32_C(  -390.69), EASYSIMD_FLOAT32_C(   249.11), EASYSIMD_FLOAT32_C(   492.46), EASYSIMD_FLOAT32_C(    83.28),
        EASYSIMD_FLOAT32_C(   -13.87), EASYSIMD_FLOAT32_C(   563.95), EASYSIMD_FLOAT32_C(    27.19), EASYSIMD_FLOAT32_C(    69.48),
        EASYSIMD_FLOAT32_C(  -499.31), EASYSIMD_FLOAT32_C(   588.53), EASYSIMD_FLOAT32_C(   881.11), EASYSIMD_FLOAT32_C(  -291.35) },
      { EASYSIMD_FLOAT32_C(   896.28), EASYSIMD_FLOAT32_C(  -328.16), EASYSIMD_FLOAT32_C(   -58.67), EASYSIMD_FLOAT32_C(  -222.16),
        EASYSIMD_FLOAT32_C(   369.25), EASYSIMD_FLOAT32_C(   249.11), EASYSIMD_FLOAT32_C(    39.79), EASYSIMD_FLOAT32_C(   257.60),
        EASYSIMD_FLOAT32_C(   -13.87), EASYSIMD_FLOAT32_C(   385.43), EASYSIMD_FLOAT32_C(   657.69), EASYSIMD_FLOAT32_C(   261.33),
        EASYSIMD_FLOAT32_C(  -197.63), EASYSIMD_FLOAT32_C(  -362.80), EASYSIMD_FLOAT32_C(   -10.34), EASYSIMD_FLOAT32_C(  -825.29) },
      UINT16_C( 7325) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_ps_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cmplt_ps_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmplt_pd_mask (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   159.59), EASYSIMD_FLOAT64_C(   210.36), EASYSIMD_FLOAT64_C(  -469.27), EASYSIMD_FLOAT64_C(  -961.30),
        EASYSIMD_FLOAT64_C(  -565.87), EASYSIMD_FLOAT64_C(  -556.86), EASYSIMD_FLOAT64_C(   785.14), EASYSIMD_FLOAT64_C(   -76.60) },
      { EASYSIMD_FLOAT64_C(   549.87), EASYSIMD_FLOAT64_C(  -373.87), EASYSIMD_FLOAT64_C(   375.69), EASYSIMD_FLOAT64_C(   255.43),
        EASYSIMD_FLOAT64_C(  -924.84), EASYSIMD_FLOAT64_C(   236.64), EASYSIMD_FLOAT64_C(  -838.91), EASYSIMD_FLOAT64_C(   432.31) },
      UINT8_C(173) },
    { { EASYSIMD_FLOAT64_C(   -86.33), EASYSIMD_FLOAT64_C(   998.88), EASYSIMD_FLOAT64_C(   169.13), EASYSIMD_FLOAT64_C(   558.70),
        EASYSIMD_FLOAT64_C(   146.37), EASYSIMD_FLOAT64_C(    90.58), EASYSIMD_FLOAT64_C(   405.41), EASYSIMD_FLOAT64_C(  -319.04) },
      { EASYSIMD_FLOAT64_C(  -110.18), EASYSIMD_FLOAT64_C(   182.04), EASYSIMD_FLOAT64_C(  -496.16), EASYSIMD_FLOAT64_C(  -883.07),
        EASYSIMD_FLOAT64_C(   321.39), EASYSIMD_FLOAT64_C(  -344.51), EASYSIMD_FLOAT64_C(   -99.97), EASYSIMD_FLOAT64_C(  -263.70) },
      UINT8_C(144) },
    { { EASYSIMD_FLOAT64_C(    29.15), EASYSIMD_FLOAT64_C(   -41.18), EASYSIMD_FLOAT64_C(  -110.04), EASYSIMD_FLOAT64_C(   548.12),
        EASYSIMD_FLOAT64_C(   271.03), EASYSIMD_FLOAT64_C(  -770.85), EASYSIMD_FLOAT64_C(   346.58), EASYSIMD_FLOAT64_C(   912.24) },
      { EASYSIMD_FLOAT64_C(    79.16), EASYSIMD_FLOAT64_C(   358.77), EASYSIMD_FLOAT64_C(  -231.91), EASYSIMD_FLOAT64_C(   206.83),
        EASYSIMD_FLOAT64_C(   115.65), EASYSIMD_FLOAT64_C(  -336.81), EASYSIMD_FLOAT64_C(  -732.53), EASYSIMD_FLOAT64_C(   334.52) },
      UINT8_C( 35) },
    { { EASYSIMD_FLOAT64_C(   256.18), EASYSIMD_FLOAT64_C(  -459.33), EASYSIMD_FLOAT64_C(   101.00), EASYSIMD_FLOAT64_C(  -417.14),
        EASYSIMD_FLOAT64_C(  -900.86), EASYSIMD_FLOAT64_C(  -806.81), EASYSIMD_FLOAT64_C(    -5.42), EASYSIMD_FLOAT64_C(   857.96) },
      { EASYSIMD_FLOAT64_C(  -232.59), EASYSIMD_FLOAT64_C(   931.14), EASYSIMD_FLOAT64_C(  -321.87), EASYSIMD_FLOAT64_C(   407.35),
        EASYSIMD_FLOAT64_C(   262.90), EASYSIMD_FLOAT64_C(   592.56), EASYSIMD_FLOAT64_C(  -812.34), EASYSIMD_FLOAT64_C(   950.75) },
      UINT8_C(186) },
    { { EASYSIMD_FLOAT64_C(  -662.53), EASYSIMD_FLOAT64_C(   872.08), EASYSIMD_FLOAT64_C(  -996.83), EASYSIMD_FLOAT64_C(   245.09),
        EASYSIMD_FLOAT64_C(  -755.15), EASYSIMD_FLOAT64_C(   154.86), EASYSIMD_FLOAT64_C(   690.61), EASYSIMD_FLOAT64_C(  -850.32) },
      { EASYSIMD_FLOAT64_C(   718.59), EASYSIMD_FLOAT64_C(  -644.78), EASYSIMD_FLOAT64_C(  -744.92), EASYSIMD_FLOAT64_C(   162.05),
        EASYSIMD_FLOAT64_C(  -429.20), EASYSIMD_FLOAT64_C(   382.77), EASYSIMD_FLOAT64_C(  -712.41), EASYSIMD_FLOAT64_C(   553.41) },
      UINT8_C(181) },
    { { EASYSIMD_FLOAT64_C(  -767.88), EASYSIMD_FLOAT64_C(   220.93), EASYSIMD_FLOAT64_C(  -852.88), EASYSIMD_FLOAT64_C(  -422.20),
        EASYSIMD_FLOAT64_C(    24.06), EASYSIMD_FLOAT64_C(   396.29), EASYSIMD_FLOAT64_C(   393.46), EASYSIMD_FLOAT64_C(   825.11) },
      { EASYSIMD_FLOAT64_C(  -326.63), EASYSIMD_FLOAT64_C(   260.49), EASYSIMD_FLOAT64_C(    21.96), EASYSIMD_FLOAT64_C(  -870.80),
        EASYSIMD_FLOAT64_C(   390.98), EASYSIMD_FLOAT64_C(  -810.50), EASYSIMD_FLOAT64_C(   -47.31), EASYSIMD_FLOAT64_C(   928.47) },
      UINT8_C(151) },
    { { EASYSIMD_FLOAT64_C(   764.04), EASYSIMD_FLOAT64_C(  -755.85), EASYSIMD_FLOAT64_C(   350.20), EASYSIMD_FLOAT64_C(  -122.92),
        EASYSIMD_FLOAT64_C(    41.32), EASYSIMD_FLOAT64_C(   468.91), EASYSIMD_FLOAT64_C(   941.23), EASYSIMD_FLOAT64_C(  -826.92) },
      { EASYSIMD_FLOAT64_C(   -79.39), EASYSIMD_FLOAT64_C(  -301.22), EASYSIMD_FLOAT64_C(  -613.48), EASYSIMD_FLOAT64_C(  -831.83),
        EASYSIMD_FLOAT64_C(  -533.10), EASYSIMD_FLOAT64_C(   168.63), EASYSIMD_FLOAT64_C(   232.01), EASYSIMD_FLOAT64_C(  -589.49) },
      UINT8_C(130) },
    { { EASYSIMD_FLOAT64_C(   431.35), EASYSIMD_FLOAT64_C(  -312.15), EASYSIMD_FLOAT64_C(  -300.41), EASYSIMD_FLOAT64_C(  -919.37),
        EASYSIMD_FLOAT64_C(    97.60), EASYSIMD_FLOAT64_C(   323.36), EASYSIMD_FLOAT64_C(   650.47), EASYSIMD_FLOAT64_C(   378.00) },
      { EASYSIMD_FLOAT64_C(  -942.80), EASYSIMD_FLOAT64_C(   278.12), EASYSIMD_FLOAT64_C(   437.54), EASYSIMD_FLOAT64_C(  -207.26),
        EASYSIMD_FLOAT64_C(   628.37), EASYSIMD_FLOAT64_C(  -977.34), EASYSIMD_FLOAT64_C(   -73.78), EASYSIMD_FLOAT64_C(   -44.83) },
      UINT8_C( 30) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmplt_pd_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_cmplt_pd_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

#endif /* !defined(EASYSIMD_BUG_GCC_96174) */

static int
test_easysimd_mm512_cmpnlt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int8_t a[64];
    const int8_t b[64];
    const uint64_t r;
  } test_vec[8] = {
    { { -INT8_C(  32), -INT8_C(  45), -INT8_C(  69), -INT8_C(  68),  INT8_C(  29), -INT8_C(  65),  INT8_C(  54), -INT8_C(  48),
         INT8_C(  81), -INT8_C( 116),  INT8_C(  44), -INT8_C( 117), -INT8_C(  11), -INT8_C(  29),  INT8_C(  12), -INT8_C( 115),
         INT8_C(  20),  INT8_C(  29),  INT8_C(  50),  INT8_C( 110), -INT8_C(  73), -INT8_C(  53),  INT8_C(  92),  INT8_C( 112),
         INT8_C(  95), -INT8_C(  99), -INT8_C( 105), -INT8_C(  17),  INT8_C( 107), -INT8_C(  63), -INT8_C(  61),  INT8_C(  75),
        -INT8_C( 107),  INT8_C( 126),  INT8_C(   8), -INT8_C(  78),  INT8_C(  62),  INT8_C(  62), -INT8_C( 126), -INT8_C( 113),
        -INT8_C(  54), -INT8_C(  81),  INT8_C(  27), -INT8_C(  64), -INT8_C( 110),  INT8_C(  39),  INT8_C(  77), -INT8_C(  90),
         INT8_C(  68),      INT8_MIN,  INT8_C(  20), -INT8_C(   5),  INT8_C(  75),  INT8_C( 112),  INT8_C( 107), -INT8_C(  86),
         INT8_C(  14),  INT8_C(   3), -INT8_C( 102),  INT8_C( 121), -INT8_C(  60),  INT8_C(  93), -INT8_C(  60),  INT8_C(  89) },
      { -INT8_C(  32), -INT8_C(  52),  INT8_C(  12),  INT8_C(  25),  INT8_C(  11), -INT8_C(  65), -INT8_C(  87), -INT8_C(  43),
         INT8_C(  81), -INT8_C(  60), -INT8_C( 107), -INT8_C(  49), -INT8_C(  21), -INT8_C(  29),  INT8_C( 117),  INT8_C(  48),
         INT8_C(  99), -INT8_C( 119),  INT8_C(  43), -INT8_C(  82), -INT8_C(   6), -INT8_C( 105),  INT8_C(  89),  INT8_C( 112),
         INT8_C(  95), -INT8_C(  13), -INT8_C( 105),  INT8_C(  94),  INT8_C( 107), -INT8_C(  63), -INT8_C(  61),  INT8_C(  75),
         INT8_C(  18), -INT8_C(  60),  INT8_C(  69), -INT8_C(  78),  INT8_C(  82), -INT8_C(  18), -INT8_C(  14), -INT8_C( 112),
        -INT8_C(  78), -INT8_C(  81),  INT8_C(  95), -INT8_C(  64), -INT8_C( 110),  INT8_C(  39), -INT8_C(  51), -INT8_C(  50),
         INT8_C(  94), -INT8_C(   7),  INT8_C( 124),  INT8_C(  88),  INT8_C(  75), -INT8_C(  43),  INT8_C(  96), -INT8_C(  86),
         INT8_C(  14), -INT8_C(  31), -INT8_C( 102),  INT8_C( 121),  INT8_C(  39),  INT8_C(  93),  INT8_C(  68),  INT8_C(  89) },
      UINT64_C(12677768375492818291) },
    { { -INT8_C(  61),  INT8_C(  84),  INT8_C( 117), -INT8_C(  42), -INT8_C(  89),  INT8_C(  30), -INT8_C(  71),  INT8_C(  64),
        -INT8_C( 114), -INT8_C(  40), -INT8_C(  12), -INT8_C( 110),  INT8_C(   1),  INT8_C(  35),  INT8_C(  35), -INT8_C(  81),
         INT8_C(  36), -INT8_C(  47), -INT8_C(  71),  INT8_C(  88), -INT8_C(  58), -INT8_C(  25), -INT8_C(  96), -INT8_C( 117),
         INT8_C(  82),  INT8_C(  36),  INT8_C(  95),  INT8_C(  61),  INT8_C(  29),  INT8_C(  72),  INT8_C(  49), -INT8_C(  32),
        -INT8_C(  99), -INT8_C(  90), -INT8_C(  74),  INT8_C(  68), -INT8_C(  59),  INT8_C( 111), -INT8_C( 124),  INT8_C(  83),
         INT8_C(  72),  INT8_C( 120), -INT8_C(  26),  INT8_C(  73), -INT8_C( 101),  INT8_C(   9), -INT8_C(   8), -INT8_C(  65),
        -INT8_C(  37), -INT8_C(  78),  INT8_C(  23), -INT8_C(  95), -INT8_C( 103), -INT8_C(  72),  INT8_C(  44), -INT8_C(  21),
        -INT8_C(  36), -INT8_C( 116),  INT8_C(  40), -INT8_C(   7), -INT8_C(  44),  INT8_C(  90), -INT8_C(  39),  INT8_C( 113) },
      {  INT8_C(   0),  INT8_C(  84), -INT8_C(  75), -INT8_C(  59), -INT8_C(   1),  INT8_C(  57),  INT8_C(  25),  INT8_C(  71),
        -INT8_C( 114), -INT8_C(   1), -INT8_C( 112),  INT8_C(  77),  INT8_C(   8), -INT8_C( 120),  INT8_C(  13), -INT8_C(  29),
         INT8_C(  58),  INT8_C(  36), -INT8_C( 123), -INT8_C(  44), -INT8_C(  36), -INT8_C(  79), -INT8_C(  65), -INT8_C(  71),
         INT8_C(  61), -INT8_C(  24), -INT8_C(  78),  INT8_C(  18),  INT8_C(  66), -INT8_C( 116), -INT8_C( 125),  INT8_C(  66),
         INT8_C(  27),  INT8_C(  57), -INT8_C(  74),  INT8_C(  68),  INT8_C( 114),  INT8_C(  33),  INT8_C(  97),  INT8_C(  36),
         INT8_C(  32), -INT8_C(  15), -INT8_C(  26),  INT8_C(  73),  INT8_C( 122),      INT8_MAX,  INT8_C(  12), -INT8_C(  65),
        -INT8_C(  93), -INT8_C(  78), -INT8_C( 120),      INT8_MIN,  INT8_C(  66),  INT8_C(  72),  INT8_C(  57),      INT8_MIN,
         INT8_C(  48), -INT8_C(  21), -INT8_C( 110), -INT8_C(   7),  INT8_C( 119),  INT8_C(  90), -INT8_C(  76),  INT8_C( 113) },
      UINT64_C(17046001085382944014) },
    { { -INT8_C(  85), -INT8_C( 112),  INT8_C( 112), -INT8_C( 112), -INT8_C(  78),  INT8_C(  93),  INT8_C( 118),  INT8_C(  43),
         INT8_C( 113), -INT8_C(  41),  INT8_C( 123), -INT8_C(  74),  INT8_C( 102),  INT8_C(  69), -INT8_C(  65), -INT8_C(  83),
         INT8_C(   9), -INT8_C(  51), -INT8_C(  12),  INT8_C(   0),  INT8_C(  96),  INT8_C( 106),  INT8_C(  22),  INT8_C(  11),
        -INT8_C( 127), -INT8_C(  63), -INT8_C(  29), -INT8_C(  17), -INT8_C( 106),  INT8_C(  55),  INT8_C( 111),  INT8_C(  65),
        -INT8_C(  57), -INT8_C(  33), -INT8_C(  47),  INT8_C( 121),  INT8_C(  60),  INT8_C(  72), -INT8_C(  92), -INT8_C(  83),
         INT8_C(  31),  INT8_C(  32),  INT8_C( 100), -INT8_C( 123),  INT8_C( 101),  INT8_C(  35),  INT8_C(  51),  INT8_C( 111),
        -INT8_C(  16),  INT8_C(  39),  INT8_C( 111),  INT8_C(  80), -INT8_C( 111), -INT8_C( 123),  INT8_C(  92),  INT8_C(  18),
         INT8_C(  71),  INT8_C(  63),  INT8_C(   1), -INT8_C(  35),  INT8_C( 118),  INT8_C( 112),  INT8_C(  30),  INT8_C(  61) },
      { -INT8_C(  85), -INT8_C(  17), -INT8_C(  74), -INT8_C( 116),  INT8_C(  55),  INT8_C(  91),  INT8_C(  57),  INT8_C(  43),
         INT8_C( 123), -INT8_C(  41), -INT8_C(  36), -INT8_C(  32), -INT8_C(  64),  INT8_C(  15),  INT8_C(  79), -INT8_C(  80),
         INT8_C(  54), -INT8_C(  65),  INT8_C(   1), -INT8_C(  57),  INT8_C(  96),  INT8_C(  93),  INT8_C(  22),  INT8_C(  11),
        -INT8_C( 100), -INT8_C(  37), -INT8_C(  29),  INT8_C(  19),  INT8_C(  75), -INT8_C( 122),  INT8_C(  80), -INT8_C( 101),
         INT8_C( 118),  INT8_C(   7), -INT8_C(  47),  INT8_C( 121),  INT8_C(  98),  INT8_C(  96), -INT8_C(  92), -INT8_C(  35),
        -INT8_C(   2), -INT8_C(  32), -INT8_C(  67), -INT8_C(  66), -INT8_C(  17),  INT8_C(  13),  INT8_C( 111),  INT8_C(  37),
        -INT8_C(  16),  INT8_C(  39), -INT8_C(  20),  INT8_C(  80), -INT8_C(  51), -INT8_C(  59), -INT8_C( 100),  INT8_C( 105),
         INT8_C(  71),  INT8_C(   4),  INT8_C( 124), -INT8_C(  20), -INT8_C( 117), -INT8_C(  51),  INT8_C(  30),  INT8_C(   1) },
      UINT64_C(17532433415263631085) },
    { {  INT8_C( 106), -INT8_C(  12),  INT8_C( 118), -INT8_C(  79),  INT8_C(  59), -INT8_C(  85), -INT8_C( 117),  INT8_C(  82),
        -INT8_C(  86), -INT8_C(  25),  INT8_C(  71), -INT8_C(  28), -INT8_C(  23), -INT8_C(  26), -INT8_C(  57),  INT8_C( 125),
         INT8_C(  62),  INT8_C(  34),  INT8_C( 106), -INT8_C(  19),  INT8_C(  93),  INT8_C(  79), -INT8_C(  10), -INT8_C(  11),
         INT8_C(  24), -INT8_C(  28), -INT8_C(  73),  INT8_C(  77), -INT8_C(  71),  INT8_C( 119), -INT8_C( 100),  INT8_C(  35),
         INT8_C( 107),  INT8_C(  18), -INT8_C(  43), -INT8_C(  90), -INT8_C(  67),  INT8_C(  96), -INT8_C(   8),  INT8_C( 104),
         INT8_C(  71),  INT8_C(  64),  INT8_C(  76),  INT8_C(  48),  INT8_C(  38),  INT8_C(  19), -INT8_C(  82),  INT8_C( 100),
         INT8_C(  53),  INT8_C(  24),  INT8_C(  81), -INT8_C( 110),  INT8_C( 103),  INT8_C(  71), -INT8_C( 121),      INT8_MAX,
         INT8_C(  44),  INT8_C(  62), -INT8_C(  51), -INT8_C(  27), -INT8_C(  75),  INT8_C( 105),  INT8_C(   9),  INT8_C(  32) },
      {  INT8_C( 106), -INT8_C(  12), -INT8_C(  58),  INT8_C(  57),  INT8_C(  62), -INT8_C(  85), -INT8_C(  95), -INT8_C( 122),
        -INT8_C(   1), -INT8_C(  19), -INT8_C(  74),  INT8_C(  37),  INT8_C(   0), -INT8_C(  26), -INT8_C(  57),  INT8_C(  54),
         INT8_C( 124), -INT8_C(  37),  INT8_C( 106), -INT8_C(  29),  INT8_C(  93),  INT8_C(  79),  INT8_C(  99),  INT8_C(  78),
        -INT8_C( 114),  INT8_C(  48), -INT8_C(  73),  INT8_C(  77), -INT8_C( 103),  INT8_C(  61),  INT8_C( 100),  INT8_C(  20),
         INT8_C(  27),  INT8_C(  18), -INT8_C(  43),  INT8_C(  89), -INT8_C(  22), -INT8_C(  18), -INT8_C(  33), -INT8_C(  23),
        -INT8_C(  37), -INT8_C( 106),  INT8_C(  76), -INT8_C(  36),  INT8_C(  38),  INT8_C(  19),  INT8_C(  18),  INT8_C( 119),
         INT8_C( 114), -INT8_C(  38),  INT8_C(  81), -INT8_C( 107),  INT8_C(  42), -INT8_C(  67), -INT8_C( 121), -INT8_C(  71),
        -INT8_C(  19),  INT8_C(  62), -INT8_C(   3), -INT8_C( 122),  INT8_C(  84),  INT8_C(  97), -INT8_C( 101),  INT8_C( 111) },
      UINT64_C( 7779475670874121383) },
    { { -INT8_C(  55),  INT8_C(  25),  INT8_C(  85),  INT8_C(   6),  INT8_C(  56), -INT8_C(  98), -INT8_C( 126), -INT8_C(  59),
         INT8_C( 118),  INT8_C(  54),  INT8_C(  87), -INT8_C(  18), -INT8_C( 101),  INT8_C(  97), -INT8_C(  31),  INT8_C(  10),
         INT8_C( 119),  INT8_C( 121),  INT8_C(  65),  INT8_C( 125),  INT8_C( 112),  INT8_C(  45), -INT8_C( 118), -INT8_C(  61),
         INT8_C(  86), -INT8_C(  12), -INT8_C(  46),  INT8_C(  85),  INT8_C(  41),  INT8_C(  43),  INT8_C(  20), -INT8_C(  14),
         INT8_C(  68),  INT8_C( 106), -INT8_C(   7),  INT8_C( 125),  INT8_C(   8),  INT8_C( 123),  INT8_C(  66),  INT8_C( 126),
        -INT8_C(  79), -INT8_C( 102),  INT8_C( 108),  INT8_C(  76), -INT8_C(   5),  INT8_C(  77),  INT8_C(  86),  INT8_C( 114),
        -INT8_C(  57), -INT8_C( 105), -INT8_C(  17),  INT8_C(  55), -INT8_C(  59),  INT8_C( 121), -INT8_C(   6),  INT8_C(  27),
         INT8_C( 109), -INT8_C(  51),  INT8_C( 112), -INT8_C( 106), -INT8_C(   8), -INT8_C( 123), -INT8_C( 119),  INT8_C(  61) },
      { -INT8_C(  17), -INT8_C( 126), -INT8_C(  70), -INT8_C(   9),  INT8_C(  56), -INT8_C(  98), -INT8_C( 126), -INT8_C(  82),
        -INT8_C( 106), -INT8_C(  30), -INT8_C(   5), -INT8_C(  18),  INT8_C(  48),  INT8_C(  97), -INT8_C(  31), -INT8_C(   9),
        -INT8_C(  23), -INT8_C(  14),  INT8_C(  46), -INT8_C(  82),  INT8_C( 108),  INT8_C(  41), -INT8_C(  55), -INT8_C(  39),
        -INT8_C(  10),  INT8_C(  57),  INT8_C( 112), -INT8_C(  18), -INT8_C(  66), -INT8_C(   7),  INT8_C(  43), -INT8_C(  83),
         INT8_C( 123),  INT8_C( 106), -INT8_C(  91),  INT8_C( 120),  INT8_C(   8),  INT8_C(  27),  INT8_C(  38),  INT8_C( 120),
        -INT8_C(  79),  INT8_C(  33),  INT8_C(  10),  INT8_C(  76),  INT8_C( 115),  INT8_C(  13),  INT8_C(  36),  INT8_C( 114),
         INT8_C(   0),  INT8_C(  83),  INT8_C(  10),  INT8_C( 108), -INT8_C(  59),  INT8_C( 121), -INT8_C(   6),  INT8_C( 114),
         INT8_C(  12), -INT8_C(  75),  INT8_C(  96), -INT8_C(  53), -INT8_C(  82), -INT8_C( 116),  INT8_C( 120),  INT8_C(  41) },
      UINT64_C(10912483575404163070) },
    { { -INT8_C(  14), -INT8_C(  40),  INT8_C(  21),  INT8_C( 126),  INT8_C(  31), -INT8_C( 121), -INT8_C(  13),  INT8_C(  59),
         INT8_C(  92), -INT8_C(  59),  INT8_C(  59), -INT8_C(  71),  INT8_C(  80),  INT8_C(  31), -INT8_C(  15), -INT8_C(  99),
         INT8_C(  82),      INT8_MIN,      INT8_MAX,  INT8_C(  26), -INT8_C(  40),  INT8_C(   4),  INT8_C(  44),  INT8_C(  89),
        -INT8_C(  11),  INT8_C(  56), -INT8_C( 112), -INT8_C(  52),  INT8_C( 102), -INT8_C(  18), -INT8_C(  90),  INT8_C(  88),
        -INT8_C(  57), -INT8_C(  68), -INT8_C(  42), -INT8_C(  26),  INT8_C(  67), -INT8_C(  55),  INT8_C(  33), -INT8_C(  97),
        -INT8_C( 113),  INT8_C(  93),  INT8_C(  88), -INT8_C(  33),  INT8_C( 124),  INT8_C(  73),  INT8_C( 124), -INT8_C(  50),
        -INT8_C(  55), -INT8_C(   5), -INT8_C(  24), -INT8_C(  95), -INT8_C(   1),  INT8_C(  20), -INT8_C(   6), -INT8_C(  11),
         INT8_C(  76), -INT8_C( 117), -INT8_C(  63), -INT8_C(  78),  INT8_C( 121),  INT8_C( 104),  INT8_C(  10),  INT8_C(  64) },
      {  INT8_C(  36), -INT8_C(  40),  INT8_C(  38),  INT8_C( 103), -INT8_C(  86),  INT8_C(  72),  INT8_C(   6),  INT8_C(  57),
         INT8_C(  92),  INT8_C(  94),  INT8_C(  24), -INT8_C(  71), -INT8_C(  89), -INT8_C( 108), -INT8_C(  15),  INT8_C( 112),
        -INT8_C( 113), -INT8_C(  41),  INT8_C(  17), -INT8_C( 113), -INT8_C(  21),  INT8_C(  11), -INT8_C( 124),  INT8_C(  56),
        -INT8_C( 106),  INT8_C(  69), -INT8_C(  22),  INT8_C(  16), -INT8_C(  83), -INT8_C(  18), -INT8_C(  90),  INT8_C(  88),
        -INT8_C(  57),  INT8_C( 119),  INT8_C(  56),      INT8_MAX, -INT8_C(  65),  INT8_C(  62), -INT8_C(  72),  INT8_C( 100),
        -INT8_C( 113),  INT8_C(  93), -INT8_C( 123),  INT8_C(  67),  INT8_C( 100),  INT8_C(  73), -INT8_C(  77), -INT8_C(  12),
         INT8_C(  75), -INT8_C(  60), -INT8_C( 125),  INT8_C(  54), -INT8_C(  48),  INT8_C(  20),  INT8_C( 110), -INT8_C(  11),
         INT8_C(  76),  INT8_C(  89),  INT8_C( 118), -INT8_C(   6),  INT8_C(  78), -INT8_C(  57), -INT8_C(  53),  INT8_C(  35) },
      UINT64_C(17417239802734804378) },
    { {  INT8_C(  21),  INT8_C(   0), -INT8_C(  58),  INT8_C(  40), -INT8_C(  85),  INT8_C(  12),  INT8_C( 103),  INT8_C(  59),
         INT8_C( 116),  INT8_C(  54),  INT8_C(  38), -INT8_C(  77), -INT8_C( 118), -INT8_C(  75), -INT8_C( 116), -INT8_C(  26),
         INT8_C(  32), -INT8_C(  49),  INT8_C(  80), -INT8_C(  43), -INT8_C(  28),  INT8_C(  21),  INT8_C(  45),  INT8_C(  64),
         INT8_C(  47),  INT8_C( 118), -INT8_C(  86),  INT8_C( 105), -INT8_C(  44),  INT8_C(  97), -INT8_C(  56), -INT8_C(  23),
         INT8_C(  98), -INT8_C( 114),  INT8_C(  17),  INT8_C(  13), -INT8_C( 102),  INT8_C( 120),  INT8_C(  72),  INT8_C(  15),
        -INT8_C(  82),  INT8_C( 110), -INT8_C(  62),  INT8_C(  57),  INT8_C(  36),  INT8_C(  78),  INT8_C(  31),  INT8_C(  68),
         INT8_C(  29),  INT8_C( 111),  INT8_C(  26),  INT8_C(   1), -INT8_C( 124),  INT8_C(  71),  INT8_C(  65), -INT8_C(  77),
        -INT8_C(  66), -INT8_C(  20),  INT8_C(  28), -INT8_C( 110),  INT8_C(  77), -INT8_C(  27),  INT8_C( 123), -INT8_C(  81) },
      {  INT8_C( 115),  INT8_C(   0), -INT8_C(  68),  INT8_C(  14),  INT8_C(   4),  INT8_C(  12),  INT8_C(  29),  INT8_C(  59),
         INT8_C( 115), -INT8_C(  33), -INT8_C(  21), -INT8_C( 105), -INT8_C( 118), -INT8_C(  75), -INT8_C(  36),  INT8_C(  74),
         INT8_C( 121), -INT8_C(  10),  INT8_C(  76), -INT8_C(   3),  INT8_C(  61), -INT8_C( 115), -INT8_C(  80),  INT8_C(  64),
         INT8_C( 121), -INT8_C(  51), -INT8_C( 115), -INT8_C(  57), -INT8_C(  78),  INT8_C(  97),  INT8_C( 118), -INT8_C(  23),
        -INT8_C( 108),  INT8_C(  51),  INT8_C(  51), -INT8_C( 104),  INT8_C(  56),  INT8_C(  80),  INT8_C(  75), -INT8_C(  85),
         INT8_C(  47),  INT8_C(  54), -INT8_C(  62),  INT8_C(  92),  INT8_C(  65),  INT8_C(  31), -INT8_C(  89), -INT8_C(  70),
         INT8_C(  29), -INT8_C(  13), -INT8_C(  72),  INT8_C(  82), -INT8_C( 124),  INT8_C( 104),  INT8_C(  78), -INT8_C(   6),
         INT8_C(  53), -INT8_C(  37), -INT8_C(  63), -INT8_C( 110), -INT8_C(  28),  INT8_C(  55),  INT8_C( 123), -INT8_C(  81) },
      UINT64_C(16003513417610837998) },
    { { -INT8_C(  70),  INT8_C(  83),  INT8_C(  62), -INT8_C(  60), -INT8_C(  40), -INT8_C(  88),  INT8_C(  97), -INT8_C(  18),
         INT8_C(  48), -INT8_C(  46), -INT8_C(  76),  INT8_C( 123), -INT8_C(  48),  INT8_C(  63),  INT8_C( 126),  INT8_C(  21),
        -INT8_C(  30), -INT8_C(  72),  INT8_C(   8), -INT8_C(  22),  INT8_C(  35), -INT8_C(  33),  INT8_C(  51), -INT8_C( 127),
        -INT8_C( 127), -INT8_C(  95),  INT8_C(   9), -INT8_C(   4), -INT8_C(  84),  INT8_C(  38), -INT8_C( 112),  INT8_C( 102),
         INT8_C( 121), -INT8_C(  50),  INT8_C(  43),  INT8_C(  82),  INT8_C( 119), -INT8_C( 116),  INT8_C(  64), -INT8_C(  89),
         INT8_C(  94), -INT8_C(  11),  INT8_C(  34),  INT8_C(  47),  INT8_C(  52), -INT8_C(  96),  INT8_C(  68),  INT8_C(  22),
         INT8_C(  88),  INT8_C(  76),  INT8_C(   0),  INT8_C( 124),  INT8_C(  43),  INT8_C(  51), -INT8_C(   3), -INT8_C(  84),
        -INT8_C(  44),  INT8_C(   7), -INT8_C(  87), -INT8_C( 127),  INT8_C(  45),  INT8_C(  57), -INT8_C(  25), -INT8_C(  90) },
      {  INT8_C(   8),  INT8_C(  18), -INT8_C(   8),      INT8_MAX, -INT8_C(  97),  INT8_C(  57),  INT8_C(  38), -INT8_C(   3),
         INT8_C(  46),  INT8_C(  72),  INT8_C(  44),  INT8_C(  98), -INT8_C(  23),  INT8_C(  63),  INT8_C( 126),  INT8_C(  65),
        -INT8_C(  30),  INT8_C( 121),  INT8_C(   8), -INT8_C(  25),  INT8_C(  35), -INT8_C(  69),  INT8_C(  51), -INT8_C( 127),
        -INT8_C( 127), -INT8_C(  95),  INT8_C(   9), -INT8_C(  17),  INT8_C( 118),  INT8_C(  38), -INT8_C( 107),  INT8_C( 126),
        -INT8_C(   4), -INT8_C( 114),  INT8_C(  43), -INT8_C( 101), -INT8_C(  57),  INT8_C(  35),  INT8_C(  64), -INT8_C(  11),
         INT8_C( 108), -INT8_C(  59),  INT8_C(  87),  INT8_C(  85),  INT8_C(  54), -INT8_C(  48), -INT8_C( 106), -INT8_C(  14),
         INT8_C(  88),  INT8_C(  76), -INT8_C(  38),  INT8_C( 124),  INT8_C(  15),  INT8_C( 110),  INT8_C( 120), -INT8_C(  47),
        -INT8_C(  85),  INT8_C(   7), -INT8_C(  64),  INT8_C(  33),  INT8_C(  45),  INT8_C(  57), -INT8_C(  96),  INT8_C(  97) },
      UINT64_C( 8295562752722561366) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpnlt_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_cmpnlt_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }
  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[64];
    int8_t b_[64];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if ((easysimd_test_codegen_rand() & 3) == 0)
        b_[j] = a_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi8(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(b_);
    easysimd__mmask64 r = easysimd_mm512_cmpnlt_epi8_mask(a, b);

    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_cmpnlt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int32_t a[16];
    const int32_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { { -INT32_C(  1688485521), -INT32_C(   187543239), -INT32_C(    57551877),  INT32_C(  1904049408),  INT32_C(  1918734706), -INT32_C(   917439378), -INT32_C(   185832029), -INT32_C(   340584324),
         INT32_C(   327552986),  INT32_C(  1527273568),  INT32_C(   744003884), -INT32_C(  1986145002), -INT32_C(   704906650), -INT32_C(  1700901898),  INT32_C(   881756856),  INT32_C(  2099200675) },
      { -INT32_C(  1688485521),  INT32_C(   705534206), -INT32_C(    57551877),  INT32_C(  1904049408),  INT32_C(  1918734706), -INT32_C(   797043688),  INT32_C(  1090849950), -INT32_C(  1614928818),
        -INT32_C(   934195254),  INT32_C(   435314407),  INT32_C(   744003884),  INT32_C(   681390907),  INT32_C(   403444480), -INT32_C(   974616025), -INT32_C(   452465257), -INT32_C(   595213038) },
      UINT16_C(51101) },
    { { -INT32_C(   876392796),  INT32_C(  1871817688), -INT32_C(  1420453735),  INT32_C(  1921505885), -INT32_C(   462541904), -INT32_C(  1613003839), -INT32_C(  1497837573),  INT32_C(    89258336),
         INT32_C(   550507591),  INT32_C(  1485791679),  INT32_C(  1459873017),  INT32_C(  1858702014),  INT32_C(  1985099701), -INT32_C(  1256903238),  INT32_C(  1566297596),  INT32_C(   241347783) },
      { -INT32_C(  2144456000), -INT32_C(  1915175276),  INT32_C(  1642388642),  INT32_C(  1921505885), -INT32_C(  1617812764),  INT32_C(  1280616527),  INT32_C(  1017753717),  INT32_C(   491457372),
        -INT32_C(   778208963), -INT32_C(   631278281), -INT32_C(  1204075695), -INT32_C(   707523855),  INT32_C(  1985099701), -INT32_C(  2067216114),  INT32_C(  1566297596),  INT32_C(   241347783) },
      UINT16_C(65307) },
    { {  INT32_C(  1468295241), -INT32_C(  1428468687),  INT32_C(   964730045),  INT32_C(   754152103),  INT32_C(   132742658), -INT32_C(  1789034648),  INT32_C(   766798929), -INT32_C(  1758091186),
         INT32_C(  1123006992), -INT32_C(   991114746),  INT32_C(   251489382), -INT32_C(   516230945), -INT32_C(   689429650), -INT32_C(   194230877),  INT32_C(     2171057), -INT32_C(   661170488) },
      {  INT32_C(   387614224), -INT32_C(  1428468687),  INT32_C(  1388697715),  INT32_C(   926089417), -INT32_C(   972154077), -INT32_C(  1789034648),  INT32_C(   766798929),  INT32_C(  1111140914),
        -INT32_C(  2124851920), -INT32_C(   818334628),  INT32_C(   251489382),  INT32_C(   571299326), -INT32_C(   689429650),  INT32_C(   803513237), -INT32_C(  1332611202), -INT32_C(   772551519) },
      UINT16_C(54643) },
    { {  INT32_C(   570921869), -INT32_C(  1252857034), -INT32_C(  2090474526), -INT32_C(   799713105),  INT32_C(   642623653), -INT32_C(  1095460300), -INT32_C(   867756024),  INT32_C(   369226377),
        -INT32_C(  1137112954), -INT32_C(   697201676),  INT32_C(   509269870), -INT32_C(   722555089), -INT32_C(  1946534825), -INT32_C(   565596458),  INT32_C(  1353421254),  INT32_C(   157723779) },
      { -INT32_C(  1446600779), -INT32_C(  1719716054),  INT32_C(  1052236046), -INT32_C(   552426104), -INT32_C(  1217786399), -INT32_C(  2104052549), -INT32_C(   867756024), -INT32_C(  1563281171),
        -INT32_C(  1137112954), -INT32_C(   697201676),  INT32_C(   509269870), -INT32_C(   722555089), -INT32_C(   141461701), -INT32_C(   565596458),  INT32_C(  1429752680),  INT32_C(  1542982275) },
      UINT16_C(12275) },
    { { -INT32_C(  1899123101),  INT32_C(  1040074453),  INT32_C(   345192337), -INT32_C(   462386367), -INT32_C(   581513522),  INT32_C(   328246983), -INT32_C(  1403555822), -INT32_C(  1065192100),
        -INT32_C(   112308445),  INT32_C(   641093013), -INT32_C(   985937532),  INT32_C(   598322004),  INT32_C(  1073807225), -INT32_C(  1219194715),  INT32_C(  1550101248),  INT32_C(  1142744608) },
      { -INT32_C(   885167306),  INT32_C(  1040074453), -INT32_C(  1862193859),  INT32_C(  1370794968),  INT32_C(  1351726250),  INT32_C(  1141368132), -INT32_C(  1314821231), -INT32_C(  1065192100),
        -INT32_C(   112308445), -INT32_C(   484686426),  INT32_C(  1249123954),  INT32_C(  1939548617),  INT32_C(   583216605), -INT32_C(  1553544430), -INT32_C(  2007693514), -INT32_C(   317699387) },
      UINT16_C(62342) },
    { {  INT32_C(  1101203503),  INT32_C(   182782419), -INT32_C(   695059951),  INT32_C(    29598596), -INT32_C(  1339780857), -INT32_C(  2066994564),  INT32_C(  1213608869),  INT32_C(  1690934069),
         INT32_C(  2007329955), -INT32_C(  2021553546),  INT32_C(  1230836676), -INT32_C(  1119215434),  INT32_C(   778989490),  INT32_C(  1051867801), -INT32_C(  1568208787),  INT32_C(  1191596195) },
      {  INT32_C(   868134077),  INT32_C(   182782419),  INT32_C(   155457362), -INT32_C(   372863433), -INT32_C(  1760086787), -INT32_C(   589968785),  INT32_C(  1988058322),  INT32_C(  1790805421),
         INT32_C(  2007329955),  INT32_C(   207837114), -INT32_C(  1525307538),  INT32_C(   865066037),  INT32_C(  2143987216),  INT32_C(  1130143600), -INT32_C(  1568208787), -INT32_C(  1844218272) },
      UINT16_C(50459) },
    { { -INT32_C(  1069992368), -INT32_C(   771516970), -INT32_C(   797197200),  INT32_C(   593661490),  INT32_C(  1171151934), -INT32_C(  1111849287), -INT32_C(    50867117), -INT32_C(   826973314),
         INT32_C(  1888415386), -INT32_C(   213740925), -INT32_C(  2117812914), -INT32_C(  1985730997),  INT32_C(  1020162947),  INT32_C(  1291487481), -INT32_C(   632753828), -INT32_C(   290914988) },
      {  INT32_C(  1868445676),  INT32_C(   392405193), -INT32_C(  1466423459), -INT32_C(   802079411), -INT32_C(  1458700368), -INT32_C(   453703801), -INT32_C(    50867117),  INT32_C(   641427258),
         INT32_C(  1888415386), -INT32_C(   213740925), -INT32_C(  2117812914), -INT32_C(  1985730997), -INT32_C(   122860687),  INT32_C(  1255973458), -INT32_C(   632753828), -INT32_C(  1606364926) },
      UINT16_C(65372) },
    { {  INT32_C(   928489957),  INT32_C(   578892867), -INT32_C(   784590641),  INT32_C(  1416723689),  INT32_C(   569997394), -INT32_C(  1710213379), -INT32_C(  1601540942), -INT32_C(   318800121),
         INT32_C(  1680103200),  INT32_C(  1518773643), -INT32_C(  1490304323), -INT32_C(  1862558658),  INT32_C(   296875284), -INT32_C(   575946453),  INT32_C(   897463854), -INT32_C(   719225419) },
      {  INT32_C(   928489957), -INT32_C(  1464221461), -INT32_C(  1051728766), -INT32_C(  1806611584),  INT32_C(   569997394), -INT32_C(   230141500),  INT32_C(  1009239687),  INT32_C(   403851588),
         INT32_C(  2037926798),  INT32_C(  1518773643), -INT32_C(  1773178602), -INT32_C(    64249668),  INT32_C(  1751634339), -INT32_C(   575946453), -INT32_C(  1159298442), -INT32_C(   719225419) },
      UINT16_C(58911) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpnlt_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_cmpnlt_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[16];
    int32_t b_[16];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < (sizeof(a_) / sizeof(a_[0])) ; j++)
      if ((easysimd_test_codegen_rand() & 3) == 0)
        b_[j] = a_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi32(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(b_);
    easysimd__mmask16 r = easysimd_mm512_cmpnlt_epi32_mask(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmplt_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmplt_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmplt_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmplt_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmplt_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpnlt_epi8_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpnlt_epi32_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpnlt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpnlt_epi32_mask)

  #if !defined(EASYSIMD_BUG_GCC_96174)
    EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_ps_mask)
    EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmplt_pd_mask)
  #endif
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
