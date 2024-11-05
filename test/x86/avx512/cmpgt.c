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

#define EASYSIMD_TEST_X86_AVX512_INSN cmpgt

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/cmpgt.h>

static int
test_easysimd_mm_cmpgt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t a[16];
    int8_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { {  INT8_C( 104), -INT8_C( 122), -INT8_C( 114),  INT8_C(   8), -INT8_C( 118), -INT8_C(  85), -INT8_C(  60), -INT8_C(  58),
        -INT8_C(  15),      INT8_MIN, -INT8_C(  50),  INT8_C(  19), -INT8_C(  95), -INT8_C(  11), -INT8_C(  95),      INT8_MIN },
      { -INT8_C(  89),  INT8_C( 104), -INT8_C(  87), -INT8_C(  85), -INT8_C(  45),  INT8_C(  29),  INT8_C( 125), -INT8_C( 117),
         INT8_C(  60), -INT8_C(  16), -INT8_C( 125),  INT8_C( 114),  INT8_C(  65),  INT8_C(  37), -INT8_C(  32), -INT8_C(  87) },
      UINT16_C( 1161) },
    { { -INT8_C(  85),  INT8_C( 110), -INT8_C(  79),  INT8_C(  54),  INT8_C(  25),  INT8_C( 118), -INT8_C(   4),  INT8_C(  11),
        -INT8_C(  10), -INT8_C(  53),  INT8_C(  30), -INT8_C( 104), -INT8_C(  64), -INT8_C(  64),  INT8_C(  24),  INT8_C( 104) },
      {  INT8_C(  40), -INT8_C(  63),  INT8_C(  19), -INT8_C(   4), -INT8_C(  33), -INT8_C( 112), -INT8_C( 121),  INT8_C(  27),
             INT8_MIN,  INT8_C(  10), -INT8_C( 114), -INT8_C(  63),  INT8_C(  48),  INT8_C( 110),  INT8_C( 106), -INT8_C(  37) },
      UINT16_C(34170) },
    { { -INT8_C(  35),  INT8_C(  28),  INT8_C(  17), -INT8_C(  10), -INT8_C( 110),  INT8_C(  14),  INT8_C(   1), -INT8_C( 120),
        -INT8_C(  39),  INT8_C(  32),  INT8_C(  32), -INT8_C( 103), -INT8_C(  32),  INT8_C(  57),  INT8_C(   1),  INT8_C(   8) },
      { -INT8_C(   6),  INT8_C(  20),  INT8_C(   4), -INT8_C(  39), -INT8_C(  91), -INT8_C( 116), -INT8_C(  11),  INT8_C(  37),
        -INT8_C( 106), -INT8_C( 125), -INT8_C(  25), -INT8_C(  58), -INT8_C(  15),  INT8_C(  81), -INT8_C(  94), -INT8_C(  50) },
      UINT16_C(51054) },
    { {  INT8_C( 109), -INT8_C(  77), -INT8_C(  59), -INT8_C(   1), -INT8_C(  63), -INT8_C(  58), -INT8_C( 120), -INT8_C( 102),
        -INT8_C(  26), -INT8_C(  88),  INT8_C(  52), -INT8_C(  58), -INT8_C(  31),  INT8_C(  53), -INT8_C(  49), -INT8_C(  36) },
      {  INT8_C(  74), -INT8_C(  45), -INT8_C(  75), -INT8_C(  17),  INT8_C(  95), -INT8_C(  86),  INT8_C(  20), -INT8_C(  10),
         INT8_C(  45), -INT8_C(   5), -INT8_C(  68),  INT8_C(  31),  INT8_C(  77),  INT8_C(  94), -INT8_C(  19), -INT8_C(  70) },
      UINT16_C(33837) },
    { {  INT8_C(  18), -INT8_C(  78), -INT8_C(  70), -INT8_C(  45),  INT8_C( 121),  INT8_C(  66),  INT8_C( 110),  INT8_C(  95),
        -INT8_C(  22), -INT8_C(  94),  INT8_C(  38), -INT8_C(  52), -INT8_C(  41), -INT8_C(  11), -INT8_C(  88),  INT8_C(  33) },
      { -INT8_C(  56),  INT8_C(  93),  INT8_C(  16),  INT8_C(  40),  INT8_C(   8),  INT8_C(  37),  INT8_C(  30),  INT8_C(  53),
         INT8_C(  32), -INT8_C(  38),  INT8_C(  84),  INT8_C( 109),  INT8_C(  57),  INT8_C(  66),  INT8_C(  40),  INT8_C(  75) },
      UINT16_C(  241) },
    { { -INT8_C(  12), -INT8_C(  30),  INT8_C(  30),  INT8_C( 109),  INT8_C(  36), -INT8_C( 116), -INT8_C(  51),  INT8_C(  14),
         INT8_C(  46), -INT8_C(  13), -INT8_C(  38),  INT8_C(   6), -INT8_C(  24), -INT8_C( 126),  INT8_C(  39), -INT8_C(  80) },
      { -INT8_C(  32),  INT8_C(  56), -INT8_C(  40), -INT8_C(  24),  INT8_C(  93), -INT8_C(  10),  INT8_C(  29),  INT8_C( 125),
        -INT8_C(  47),  INT8_C( 114), -INT8_C(  21),  INT8_C(  10), -INT8_C(  76),  INT8_C(  19),  INT8_C(  85), -INT8_C(  88) },
      UINT16_C(37133) },
    { { -INT8_C(  11),  INT8_C( 115),  INT8_C(  22),  INT8_C(  25),  INT8_C(   0), -INT8_C(  29),  INT8_C(  39),  INT8_C(  46),
        -INT8_C(  42),  INT8_C(   2),  INT8_C(  52), -INT8_C(  66), -INT8_C( 124),  INT8_C(  92),  INT8_C( 110),  INT8_C( 100) },
      { -INT8_C( 108),  INT8_C(  71),  INT8_C(  76), -INT8_C(  15),  INT8_C(  61),  INT8_C( 106),  INT8_C( 110),  INT8_C(  14),
        -INT8_C(  36),  INT8_C(  89),  INT8_C(  24), -INT8_C( 112),  INT8_C( 108),  INT8_C( 109),  INT8_C(  56),  INT8_C(  97) },
      UINT16_C(52363) },
    { { -INT8_C(  31),  INT8_C(  78),  INT8_C( 122), -INT8_C(  31),  INT8_C(  49), -INT8_C(  94),  INT8_C(  15),  INT8_C(   7),
        -INT8_C(  92),  INT8_C(  68), -INT8_C(  59),  INT8_C(  40), -INT8_C(  96),  INT8_C(  52), -INT8_C( 115),  INT8_C(  52) },
      {  INT8_C( 123), -INT8_C(  39),  INT8_C(  37), -INT8_C(  72),  INT8_C(  67), -INT8_C( 109), -INT8_C(  57),  INT8_C(  31),
        -INT8_C(  19), -INT8_C(  33), -INT8_C(  81),  INT8_C(  89),  INT8_C(  77), -INT8_C(  24), -INT8_C(  69),  INT8_C(  46) },
      UINT16_C(42606) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_cmpgt_epi8_mask(a, b);

    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpgt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[8];
    int16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { -INT16_C(  3374),  INT16_C(   363), -INT16_C( 11623),  INT16_C( 23066), -INT16_C( 17352), -INT16_C( 31042), -INT16_C(  3617),  INT16_C(  2670) },
      { -INT16_C( 17249),  INT16_C( 11610), -INT16_C( 17190),  INT16_C( 10330),  INT16_C( 21880),  INT16_C( 26233),  INT16_C(  6170), -INT16_C(  4883) },
      UINT8_C(141) },
    { {  INT16_C( 22538), -INT16_C( 23314),  INT16_C(  2090),  INT16_C( 25342), -INT16_C( 16956), -INT16_C( 23320),  INT16_C( 22446),  INT16_C( 19886) },
      {  INT16_C(  2067), -INT16_C(  4742), -INT16_C( 10811),  INT16_C( 15637), -INT16_C( 28886),  INT16_C( 17571), -INT16_C( 28505), -INT16_C( 19920) },
      UINT8_C(221) },
    { {  INT16_C(  7912),  INT16_C(  4950),  INT16_C( 21543), -INT16_C(  5259),  INT16_C( 24081), -INT16_C( 16497),  INT16_C( 15797), -INT16_C( 14323) },
      { -INT16_C( 30906),  INT16_C(  2997), -INT16_C( 13732), -INT16_C( 31160), -INT16_C(  5287),  INT16_C(   458), -INT16_C(  1157),  INT16_C( 25523) },
      UINT8_C( 95) },
    { {  INT16_C(  2329),  INT16_C( 16502), -INT16_C(  5027),  INT16_C( 28460), -INT16_C( 17590), -INT16_C(   210),  INT16_C( 15353),  INT16_C( 16327) },
      {  INT16_C( 31939),  INT16_C(  8010), -INT16_C( 28090), -INT16_C( 24410),  INT16_C( 28797), -INT16_C(  1887),  INT16_C( 21611), -INT16_C( 31397) },
      UINT8_C(174) },
    { { -INT16_C( 11683), -INT16_C( 17723), -INT16_C(  3650),  INT16_C(  2089),  INT16_C( 22701), -INT16_C( 23033), -INT16_C( 12653),  INT16_C( 22245) },
      {  INT16_C( 12106), -INT16_C( 28554),  INT16_C(  7361),  INT16_C( 15920), -INT16_C( 11892), -INT16_C(  1994), -INT16_C( 28379), -INT16_C( 32131) },
      UINT8_C(210) },
    { {  INT16_C( 16995),  INT16_C(  8509),  INT16_C( 26164), -INT16_C(  7895),  INT16_C( 12478),  INT16_C( 21127),  INT16_C( 27902),  INT16_C( 18600) },
      {  INT16_C(  7835),  INT16_C( 23769),  INT16_C(  2362), -INT16_C( 14438), -INT16_C( 12069),  INT16_C(   191),  INT16_C( 15457), -INT16_C( 14973) },
      UINT8_C(253) },
    { { -INT16_C( 16258), -INT16_C( 19738),  INT16_C(  4134), -INT16_C(  6765),  INT16_C(  6720),  INT16_C( 16183), -INT16_C(  8314),  INT16_C(  8583) },
      {  INT16_C( 24830),  INT16_C( 14461),  INT16_C(  5994),  INT16_C( 17919), -INT16_C( 16665),  INT16_C( 18757), -INT16_C( 14086),  INT16_C( 30990) },
      UINT8_C( 80) },
    { { -INT16_C(  2936), -INT16_C( 20693), -INT16_C( 16636),  INT16_C( 17812), -INT16_C( 13351),  INT16_C( 24708),  INT16_C(  2986), -INT16_C( 22399) },
      { -INT16_C(   148), -INT16_C( 10527), -INT16_C(  8170), -INT16_C(   485),  INT16_C( 24735), -INT16_C( 26297),  INT16_C( 21801), -INT16_C( 20206) },
      UINT8_C( 40) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epi16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_cmpgt_epi16_mask(a, b);

    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpgt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[4];
    int32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { {  INT32_C(  2140371590), -INT32_C(   200387208),  INT32_C(  1014571390),  INT32_C(    83179171) },
      {  INT32_C(   655113209),  INT32_C(  1631803085), -INT32_C(   318056717),  INT32_C(  1664531677) },
      UINT8_C(  5) },
    { { -INT32_C(  1344091850), -INT32_C(  1650200545), -INT32_C(  1244062702),  INT32_C(  1287245395) },
      { -INT32_C(   696990455),  INT32_C(   255375132),  INT32_C(  1811694221),  INT32_C(   902705918) },
      UINT8_C(  8) },
    { {  INT32_C(   467972604), -INT32_C(  1279752287), -INT32_C(   144141916),  INT32_C(  1766007391) },
      {  INT32_C(    54507495), -INT32_C(    65898641), -INT32_C(  1201205574),  INT32_C(  1038956097) },
      UINT8_C( 13) },
    { { -INT32_C(  2007445017), -INT32_C(    46395303),  INT32_C(    16032929), -INT32_C(  1368836154) },
      {  INT32_C(  1588701679), -INT32_C(   631585760),  INT32_C(   328450770), -INT32_C(   548372232) },
      UINT8_C(  2) },
    { { -INT32_C(  1419270062),  INT32_C(  1504224184),  INT32_C(   240819272), -INT32_C(  1027816493) },
      { -INT32_C(  1927254420),  INT32_C(    73890610),  INT32_C(   890829373), -INT32_C(   854300549) },
      UINT8_C(  3) },
    { { -INT32_C(   914851055),  INT32_C(  1730355231), -INT32_C(  1888125508), -INT32_C(  1387122112) },
      { -INT32_C(   767921504),  INT32_C(   718709229),  INT32_C(   392162972),  INT32_C(  1759802199) },
      UINT8_C(  2) },
    { {  INT32_C(   238116079),  INT32_C(   947213436),  INT32_C(   315091665), -INT32_C(  1128326884) },
      {  INT32_C(  2039413132),  INT32_C(   916743578), -INT32_C(  1437793453),  INT32_C(  1712468343) },
      UINT8_C(  6) },
    { {  INT32_C(   158614669),  INT32_C(  1782704536), -INT32_C(   260306477), -INT32_C(  1364444382) },
      { -INT32_C(   836224204), -INT32_C(   217723744),  INT32_C(  1184781007),  INT32_C(   296530052) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epi32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_cmpgt_epi32_mask(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpgt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[2];
    int64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { -INT64_C(  365350150593370684),  INT64_C( 2303868253065867541) },
      { -INT64_C( 6124068856967694420), -INT64_C(  351463055147214693) },
      UINT8_C(  3) },
    { {  INT64_C( 8606995033046015615),  INT64_C( 7257746201623706785) },
      { -INT64_C( 6078269534506835928), -INT64_C( 4201733903832978306) },
      UINT8_C(  3) },
    { { -INT64_C( 4348032878836321585),  INT64_C( 4145304045858087019) },
      {  INT64_C( 5831838064585222964), -INT64_C( 9164625469700226875) },
      UINT8_C(  2) },
    { {  INT64_C( 2921808925629156775),  INT64_C( 3460403871888435386) },
      { -INT64_C( 2873188729332942339), -INT64_C( 9019385299618028126) },
      UINT8_C(  3) },
    { { -INT64_C( 7374968885364357695), -INT64_C( 4544578684553203263) },
      {  INT64_C( 7461814516528997026), -INT64_C( 1858985932938907435) },
      UINT8_C(  0) },
    { {  INT64_C( 1768800126511644436), -INT64_C(  252548635058310917) },
      { -INT64_C( 7759351693280473562), -INT64_C( 5856518333565498193) },
      UINT8_C(  3) },
    { {  INT64_C( 2478718291760193014),  INT64_C( 2770944126372297758) },
      {  INT64_C( 1423245664624649837),  INT64_C( 8266252095664358510) },
      UINT8_C(  1) },
    { { -INT64_C( 6627003391341995586), -INT64_C( 5365316270250203037) },
      {  INT64_C( 7795627592763153335),  INT64_C( 8400389465221437093) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epi64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_cmpgt_epi64_mask(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int8_t a[16];
    int8_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C( 7622),
      {  INT8_C( 106),  INT8_C(  41), -INT8_C( 105),  INT8_C(  67), -INT8_C(  63), -INT8_C(  96), -INT8_C(  91), -INT8_C(  64),
         INT8_C(  43),  INT8_C(  56),  INT8_C(  96),  INT8_C(  82), -INT8_C(  44), -INT8_C(  22),  INT8_C(  81), -INT8_C( 127) },
      {  INT8_C(   5),  INT8_C(  33), -INT8_C(  61),  INT8_C( 125), -INT8_C(  20), -INT8_C(   9),  INT8_C(  58),  INT8_C(  79),
        -INT8_C(  95), -INT8_C(  13),  INT8_C( 107), -INT8_C(   2),      INT8_MAX,  INT8_C(  49),  INT8_C(  28), -INT8_C(  22) },
      UINT16_C( 2306) },
    { UINT16_C(45914),
      {  INT8_C(  45),  INT8_C(  27),  INT8_C(  83), -INT8_C(  46), -INT8_C(  37),  INT8_C( 126),  INT8_C(  10),  INT8_C(  59),
        -INT8_C(  48), -INT8_C(  34),  INT8_C(  37),  INT8_C(  33),  INT8_C(  95),  INT8_C(  43),  INT8_C(  66),  INT8_C(  34) },
      { -INT8_C(  88),  INT8_C(  46),  INT8_C(  26), -INT8_C(  29),  INT8_C( 126), -INT8_C(  69), -INT8_C(  42), -INT8_C(  23),
        -INT8_C(  71),  INT8_C(  85),  INT8_C(  26), -INT8_C(  43),  INT8_C(  63),  INT8_C( 116), -INT8_C( 120),  INT8_C( 108) },
      UINT16_C( 4416) },
    { UINT16_C(56207),
      {  INT8_C(  62),  INT8_C( 106),  INT8_C(  89),  INT8_C(  72), -INT8_C(  90),  INT8_C(  42),  INT8_C(  39), -INT8_C(  53),
         INT8_C(  75), -INT8_C( 122), -INT8_C(  10), -INT8_C( 114), -INT8_C(  87), -INT8_C(  97), -INT8_C(  68), -INT8_C(  61) },
      { -INT8_C( 126),  INT8_C(  58),  INT8_C( 126),  INT8_C(  88),  INT8_C(  35),  INT8_C(  55), -INT8_C(  83),  INT8_C(  61),
         INT8_C(  13), -INT8_C(  19), -INT8_C(  79), -INT8_C( 107),  INT8_C(  89),  INT8_C(  64),  INT8_C( 113), -INT8_C( 104) },
      UINT16_C(33027) },
    { UINT16_C(51883),
      { -INT8_C(  32),  INT8_C(  81), -INT8_C(  12),  INT8_C(   7),  INT8_C(  28),  INT8_C(  64), -INT8_C( 114),  INT8_C(  19),
        -INT8_C(  50),  INT8_C(  55), -INT8_C(  78), -INT8_C( 118), -INT8_C(   6),  INT8_C(  52), -INT8_C(  59),  INT8_C( 120) },
      { -INT8_C( 116), -INT8_C(  24), -INT8_C(  81),  INT8_C(  57),  INT8_C(  38), -INT8_C(  68),  INT8_C(  38), -INT8_C(  41),
         INT8_C(  82),      INT8_MIN,  INT8_C(  24), -INT8_C(  61),  INT8_C(  24), -INT8_C(  61), -INT8_C( 115), -INT8_C(   8) },
      UINT16_C(49827) },
    { UINT16_C(33300),
      {  INT8_C(   0),  INT8_C(  48), -INT8_C(  62), -INT8_C( 114),  INT8_C(  67), -INT8_C( 112), -INT8_C(  59), -INT8_C(  11),
         INT8_C(  26), -INT8_C(  65),  INT8_C(  41), -INT8_C(  33),  INT8_C(  55), -INT8_C(  75), -INT8_C(  56), -INT8_C(  26) },
      { -INT8_C(  17), -INT8_C(  18), -INT8_C(  93),  INT8_C(  21), -INT8_C(  59), -INT8_C(  11), -INT8_C( 107), -INT8_C(  35),
        -INT8_C(  72), -INT8_C(  83), -INT8_C(  96),  INT8_C(  69), -INT8_C(  90), -INT8_C(  76), -INT8_C(  57), -INT8_C(  90) },
      UINT16_C(33300) },
    { UINT16_C(35301),
      {  INT8_C(  52),  INT8_C(  40),  INT8_C(  25), -INT8_C(   7),  INT8_C(  30),  INT8_C(  52), -INT8_C(  72),  INT8_C(  71),
         INT8_C(  19), -INT8_C(  17), -INT8_C(   3), -INT8_C(  37), -INT8_C(  43), -INT8_C(  20), -INT8_C(  55),  INT8_C( 120) },
      {  INT8_C(   1), -INT8_C( 113),  INT8_C( 109), -INT8_C( 105),  INT8_C( 108),  INT8_C(  37),  INT8_C(  68),  INT8_C(  13),
         INT8_C( 107), -INT8_C(  22), -INT8_C(  63),  INT8_C(  50), -INT8_C( 112), -INT8_C(  90), -INT8_C(  68), -INT8_C(  60) },
      UINT16_C(32929) },
    { UINT16_C(54735),
      { -INT8_C(  67), -INT8_C(  19),  INT8_C(   9),  INT8_C( 117),  INT8_C(  52),  INT8_C(  29),  INT8_C( 100),  INT8_C(  49),
        -INT8_C(   8),  INT8_C(  58),  INT8_C(  29), -INT8_C(  62), -INT8_C(  78),  INT8_C(  31),  INT8_C(  81),  INT8_C(  32) },
      { -INT8_C(  74), -INT8_C(  67),  INT8_C(  69), -INT8_C(   6), -INT8_C(  54), -INT8_C(  80), -INT8_C(  27), -INT8_C( 116),
        -INT8_C(  29),  INT8_C( 117),  INT8_C(  50), -INT8_C(  97),  INT8_C(  58),  INT8_C(   1),  INT8_C( 116), -INT8_C(   9) },
      UINT16_C(33227) },
    { UINT16_C(32494),
      {  INT8_C( 109),  INT8_C(  35), -INT8_C( 101), -INT8_C(  47),  INT8_C(  84), -INT8_C( 109),  INT8_C(  11),  INT8_C( 114),
         INT8_C(  85), -INT8_C(  66), -INT8_C( 111), -INT8_C(  90), -INT8_C(  34),  INT8_C(  71),  INT8_C( 100),  INT8_C(  35) },
      {  INT8_C(  65),  INT8_C(  46), -INT8_C(  44),  INT8_C(  38), -INT8_C(  70), -INT8_C(  73), -INT8_C( 100), -INT8_C(  19),
         INT8_C(  86), -INT8_C(  42), -INT8_C(  18), -INT8_C(  54), -INT8_C(  51), -INT8_C(  35),  INT8_C(  72),  INT8_C(  58) },
      UINT16_C(28864) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epi8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_mask_cmpgt_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int16_t a[8];
    int16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(112),
      {  INT16_C( 13242),  INT16_C( 18509), -INT16_C( 10439),  INT16_C( 13078),  INT16_C(  7534), -INT16_C(  1563), -INT16_C( 31465), -INT16_C(  9292) },
      {  INT16_C( 28309),  INT16_C( 22791), -INT16_C( 11038),  INT16_C( 28745),  INT16_C( 20140),  INT16_C( 18622),  INT16_C( 19261), -INT16_C(  2120) },
      UINT8_C(  0) },
    { UINT8_C(126),
      {  INT16_C( 16389), -INT16_C(  9033), -INT16_C(  5546),  INT16_C( 29514),  INT16_C( 17360),  INT16_C( 21898),  INT16_C( 26360),  INT16_C( 26347) },
      {  INT16_C( 17517),  INT16_C( 16712), -INT16_C( 18035), -INT16_C(  9234),  INT16_C( 13943), -INT16_C( 15847),  INT16_C(  4334), -INT16_C(  3264) },
      UINT8_C(124) },
    { UINT8_C( 80),
      { -INT16_C( 12297), -INT16_C(  7769),  INT16_C(  6681),  INT16_C( 23729),  INT16_C(  1957),  INT16_C(  2900), -INT16_C( 17422),  INT16_C( 13944) },
      { -INT16_C( 18173), -INT16_C( 17213), -INT16_C( 24921), -INT16_C(  8909), -INT16_C(  2633), -INT16_C( 14133), -INT16_C( 16843),  INT16_C( 11288) },
      UINT8_C( 16) },
    { UINT8_C(141),
      {  INT16_C(  3775), -INT16_C(  9562),  INT16_C(   959), -INT16_C( 14721), -INT16_C( 30121),  INT16_C(  4792), -INT16_C(  4606), -INT16_C( 17642) },
      { -INT16_C( 11599),  INT16_C( 20579),  INT16_C( 16390), -INT16_C(  1273), -INT16_C( 12532), -INT16_C( 13775),  INT16_C( 24040), -INT16_C( 22696) },
      UINT8_C(129) },
    { UINT8_C(107),
      { -INT16_C( 32258),  INT16_C(   299), -INT16_C(  3840), -INT16_C( 30119),  INT16_C( 27562), -INT16_C( 26484),  INT16_C( 18561),  INT16_C( 21578) },
      { -INT16_C( 25941), -INT16_C(  5286),  INT16_C( 21921),  INT16_C( 29175), -INT16_C( 15738), -INT16_C(  7079),  INT16_C(    26),  INT16_C(  6223) },
      UINT8_C( 66) },
    { UINT8_C(130),
      {  INT16_C(  6778),  INT16_C( 27778),  INT16_C(  3443), -INT16_C(  8682), -INT16_C( 20839), -INT16_C(  7840), -INT16_C( 19208), -INT16_C( 28020) },
      {  INT16_C( 30734),  INT16_C( 25396), -INT16_C( 23185),  INT16_C( 12778), -INT16_C( 12546), -INT16_C(   437),  INT16_C( 25629), -INT16_C( 26496) },
      UINT8_C(  2) },
    { UINT8_C(126),
      {  INT16_C(  1027),  INT16_C(  4337), -INT16_C( 12518), -INT16_C( 14167), -INT16_C( 29905), -INT16_C(  7231),  INT16_C( 21271), -INT16_C( 28687) },
      {  INT16_C( 21895),  INT16_C( 11519),  INT16_C( 12351),  INT16_C(  3370),  INT16_C( 10620), -INT16_C(  8150), -INT16_C( 15703), -INT16_C( 21410) },
      UINT8_C( 96) },
    { UINT8_C(198),
      { -INT16_C( 17329),  INT16_C(  7904), -INT16_C( 22170), -INT16_C(  3762),  INT16_C( 12650), -INT16_C( 17144), -INT16_C( 26589),  INT16_C( 30789) },
      {  INT16_C( 29079), -INT16_C( 14409), -INT16_C( 15204), -INT16_C( 15037),  INT16_C(  9198), -INT16_C( 20114),  INT16_C(  7041), -INT16_C( 12169) },
      UINT8_C(130) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epi16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_mask_cmpgt_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int32_t a[4];
    int32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 38),
      { -INT32_C(   170870544),  INT32_C(  1902919298),  INT32_C(  1158122812), -INT32_C(  2016987825) },
      {  INT32_C(  1376099023),  INT32_C(   526869750),  INT32_C(  1698101953), -INT32_C(  1232373818) },
      UINT8_C(  2) },
    { UINT8_C( 48),
      { -INT32_C(  1716343717), -INT32_C(  1512692712),  INT32_C(  1626676011),  INT32_C(  2083486946) },
      { -INT32_C(   445415039), -INT32_C(   492334360),  INT32_C(  1068043464), -INT32_C(   210804841) },
      UINT8_C(  0) },
    { UINT8_C( 11),
      {  INT32_C(  1176734754),  INT32_C(  2104609122),  INT32_C(   979358142), -INT32_C(  2118394880) },
      { -INT32_C(   513171121),  INT32_C(  1420381256),  INT32_C(  1424746741),  INT32_C(  2053168728) },
      UINT8_C(  3) },
    { UINT8_C(107),
      {  INT32_C(  1288552579),  INT32_C(    34229041), -INT32_C(  2046671702), -INT32_C(  1579842816) },
      {  INT32_C(   954840812),  INT32_C(  1210924383),  INT32_C(   127959592),  INT32_C(  1701976802) },
      UINT8_C(  1) },
    { UINT8_C(218),
      { -INT32_C(  1978944961),  INT32_C(    20254140), -INT32_C(  1845380337),  INT32_C(  1199481489) },
      { -INT32_C(   928532597),  INT32_C(  1727066084), -INT32_C(  1454835825), -INT32_C(  1450987927) },
      UINT8_C(  8) },
    { UINT8_C( 96),
      { -INT32_C(  1692650610),  INT32_C(   615128424), -INT32_C(  1061864418),  INT32_C(  1900805306) },
      { -INT32_C(  1839852637), -INT32_C(    81675260), -INT32_C(  1285174779), -INT32_C(   619508147) },
      UINT8_C(  0) },
    { UINT8_C( 65),
      {  INT32_C(  1303017007),  INT32_C(  1550568992),  INT32_C(  2132225155), -INT32_C(  1960605577) },
      { -INT32_C(  1685080610), -INT32_C(  1583314217), -INT32_C(    34647057), -INT32_C(  1790981530) },
      UINT8_C(  1) },
    { UINT8_C( 65),
      { -INT32_C(  1218321687),  INT32_C(  2050670158),  INT32_C(  1576122837), -INT32_C(  1841529636) },
      { -INT32_C(  1771448565), -INT32_C(   897250697), -INT32_C(  1003453447), -INT32_C(  1425685054) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epi32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_mask_cmpgt_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t a[2];
    int64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(182),
      {  INT64_C( 3056840829983867730), -INT64_C( 2608612690705642233) },
      { -INT64_C( 1218941766115832877),  INT64_C( 8929616791766476956) },
      UINT8_C(  0) },
    { UINT8_C(163),
      { -INT64_C(  999422100329694971),  INT64_C( 7308324161699925569) },
      {  INT64_C( 3258426372511658026),  INT64_C( 6908190159122361820) },
      UINT8_C(  2) },
    { UINT8_C( 75),
      { -INT64_C( 1683744169015543798),  INT64_C( 1744978447316433834) },
      {  INT64_C( 8707651656870646714), -INT64_C(  270442537371430111) },
      UINT8_C(  2) },
    { UINT8_C(169),
      {  INT64_C( 4196840279443201392),  INT64_C( 6147406355639651117) },
      {  INT64_C( 3856163348150242199),  INT64_C( 2775664013611595690) },
      UINT8_C(  1) },
    { UINT8_C(128),
      {  INT64_C( 4765012880351279560),  INT64_C( 5776738458794243738) },
      { -INT64_C( 8647852396401947271),  INT64_C( 2698328381895596239) },
      UINT8_C(  0) },
    { UINT8_C( 29),
      { -INT64_C( 1277519018406438671), -INT64_C( 7949341507386549989) },
      { -INT64_C( 3085201051500871497), -INT64_C( 8725909456447987098) },
      UINT8_C(  1) },
    { UINT8_C(111),
      {  INT64_C( 5651620013038579963),  INT64_C( 5946690298482697098) },
      {  INT64_C( 3968836847157634693), -INT64_C( 4531656116063642026) },
      UINT8_C(  3) },
    { UINT8_C(130),
      {  INT64_C( 2658045726452880323),  INT64_C( 2162229441046426265) },
      { -INT64_C( 6732481477391175913),  INT64_C( 2092811920672833079) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epi64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_mask_cmpgt_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpgt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[16];
    uint8_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { { UINT8_C( 17), UINT8_C(169), UINT8_C( 77), UINT8_C(116), UINT8_C(103), UINT8_C(233), UINT8_C( 13), UINT8_C(104),
        UINT8_C( 92), UINT8_C(  6), UINT8_C(247), UINT8_C(175), UINT8_C( 93), UINT8_C(162), UINT8_C(148), UINT8_C( 93) },
      { UINT8_C( 23), UINT8_C(183), UINT8_C( 32), UINT8_C(211), UINT8_C( 32), UINT8_C(193), UINT8_C(180), UINT8_C(100),
        UINT8_C( 55), UINT8_C( 66), UINT8_C( 40), UINT8_C(210), UINT8_C(  4), UINT8_C(218), UINT8_C(124), UINT8_C( 21) },
      UINT16_C(54708) },
    { { UINT8_C(131), UINT8_C(201), UINT8_C(137), UINT8_C(235), UINT8_C(179), UINT8_C(151), UINT8_C( 83), UINT8_C( 15),
        UINT8_C(157), UINT8_C( 75), UINT8_C(191), UINT8_C(250), UINT8_C(237), UINT8_C( 83), UINT8_C( 87), UINT8_C(  5) },
      { UINT8_C( 10), UINT8_C(119), UINT8_C(216), UINT8_C( 42), UINT8_C( 56), UINT8_C(140), UINT8_C(142), UINT8_C(112),
        UINT8_C(206), UINT8_C(183), UINT8_C( 66), UINT8_C(210), UINT8_C(145), UINT8_C(190), UINT8_C(231), UINT8_C( 21) },
      UINT16_C( 7227) },
    { { UINT8_C(136), UINT8_C(113), UINT8_C(  0), UINT8_C( 59), UINT8_C(  8), UINT8_C( 83), UINT8_C( 74), UINT8_C(165),
        UINT8_C(158), UINT8_C(  9), UINT8_C(159), UINT8_C(140), UINT8_C( 92), UINT8_C(247), UINT8_C(145), UINT8_C(102) },
      { UINT8_C(110), UINT8_C(105), UINT8_C(144), UINT8_C(167), UINT8_C(245), UINT8_C( 31), UINT8_C( 23), UINT8_C(196),
        UINT8_C(214), UINT8_C( 89), UINT8_C(150), UINT8_C(103), UINT8_C( 24), UINT8_C(126), UINT8_C(124), UINT8_C(160) },
      UINT16_C(31843) },
    { { UINT8_C(239), UINT8_C(124), UINT8_C(219), UINT8_C(247), UINT8_C(208), UINT8_C( 37), UINT8_C(156), UINT8_C(110),
        UINT8_C( 47), UINT8_C( 59), UINT8_C(250), UINT8_C(139), UINT8_C( 50), UINT8_C(139), UINT8_C(242), UINT8_C(161) },
      { UINT8_C(244), UINT8_C(130), UINT8_C( 72), UINT8_C(234), UINT8_C(161), UINT8_C( 95), UINT8_C(174), UINT8_C(119),
        UINT8_C(184), UINT8_C( 68), UINT8_C(223), UINT8_C(208), UINT8_C(194), UINT8_C( 91), UINT8_C(112), UINT8_C(177) },
      UINT16_C(25628) },
    { { UINT8_C(216), UINT8_C( 75), UINT8_C(168), UINT8_C(168), UINT8_C(113), UINT8_C( 68), UINT8_C( 22), UINT8_C(160),
        UINT8_C(128), UINT8_C( 17), UINT8_C( 43), UINT8_C(178), UINT8_C(156), UINT8_C( 29), UINT8_C( 83), UINT8_C(145) },
      { UINT8_C(160), UINT8_C(155), UINT8_C(123), UINT8_C( 65), UINT8_C(250), UINT8_C( 41), UINT8_C(185), UINT8_C(179),
        UINT8_C(109), UINT8_C(152), UINT8_C(131), UINT8_C( 48), UINT8_C(243), UINT8_C(244), UINT8_C(225), UINT8_C(203) },
      UINT16_C( 2349) },
    { { UINT8_C( 63), UINT8_C(138), UINT8_C(115), UINT8_C(176), UINT8_C(206), UINT8_C(138), UINT8_C( 80), UINT8_C( 78),
        UINT8_C(155), UINT8_C(124), UINT8_C(  1), UINT8_C( 55), UINT8_C(153), UINT8_C( 84), UINT8_C(200), UINT8_C( 57) },
      { UINT8_C(240), UINT8_C( 67), UINT8_C(123), UINT8_C(234), UINT8_C(108), UINT8_C( 52), UINT8_C(157), UINT8_C(218),
        UINT8_C(204), UINT8_C( 33), UINT8_C( 10), UINT8_C(191), UINT8_C( 21), UINT8_C(235), UINT8_C(139), UINT8_C( 84) },
      UINT16_C(21042) },
    { { UINT8_C(117), UINT8_C(254), UINT8_C(  5), UINT8_C( 68), UINT8_C(136), UINT8_C( 85), UINT8_C(146), UINT8_C( 35),
        UINT8_C(209), UINT8_C(147), UINT8_C( 91), UINT8_C(107), UINT8_C(232), UINT8_C( 35), UINT8_C(164), UINT8_C(216) },
      { UINT8_C(103), UINT8_C( 31), UINT8_C(194), UINT8_C(211), UINT8_C( 83), UINT8_C( 96), UINT8_C(173), UINT8_C( 31),
        UINT8_C(129), UINT8_C(183), UINT8_C(223), UINT8_C(150), UINT8_C(163), UINT8_C(106), UINT8_C(234), UINT8_C( 24) },
      UINT16_C(37267) },
    { { UINT8_C(104), UINT8_C(239), UINT8_C( 92), UINT8_C(241), UINT8_C( 69), UINT8_C(239), UINT8_C( 20), UINT8_C( 22),
        UINT8_C(130), UINT8_C(111), UINT8_C(129), UINT8_C(106), UINT8_C(147), UINT8_C( 38), UINT8_C( 66), UINT8_C(250) },
      { UINT8_C( 69), UINT8_C(  5), UINT8_C(205), UINT8_C(153), UINT8_C(101), UINT8_C(123), UINT8_C(184), UINT8_C(230),
        UINT8_C( 50), UINT8_C(151), UINT8_C(124), UINT8_C(213), UINT8_C(  1), UINT8_C(102), UINT8_C(238), UINT8_C(106) },
      UINT16_C(38187) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epu8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epu8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__mmask16 r = easysimd_mm_cmpgt_epu8_mask(a, b);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpgt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[8];
    uint16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT16_C(35761), UINT16_C( 6396), UINT16_C(56106), UINT16_C(28798), UINT16_C(46191), UINT16_C(47613), UINT16_C(52401), UINT16_C(60566) },
      { UINT16_C(12720), UINT16_C( 5456), UINT16_C( 7559), UINT16_C(33284), UINT16_C(40250), UINT16_C(30586), UINT16_C(10606), UINT16_C( 7937) },
      UINT8_C(247) },
    { { UINT16_C(64948), UINT16_C(57144), UINT16_C(46809), UINT16_C(18511), UINT16_C(19562), UINT16_C( 6913), UINT16_C(38936), UINT16_C(51208) },
      { UINT16_C(22729), UINT16_C(20958), UINT16_C(57973), UINT16_C(45011), UINT16_C(19839), UINT16_C(60966), UINT16_C(10103), UINT16_C(11021) },
      UINT8_C(195) },
    { { UINT16_C(17701), UINT16_C(65034), UINT16_C(23035), UINT16_C(25926), UINT16_C(18597), UINT16_C(48513), UINT16_C(35296), UINT16_C(43398) },
      { UINT16_C(25825), UINT16_C(22266), UINT16_C(52806), UINT16_C(50437), UINT16_C(11035), UINT16_C(37555), UINT16_C(49491), UINT16_C(30910) },
      UINT8_C(178) },
    { { UINT16_C(51206), UINT16_C(  630), UINT16_C(48162), UINT16_C(51047), UINT16_C(59396), UINT16_C(58501), UINT16_C( 2929), UINT16_C(21134) },
      { UINT16_C(34927), UINT16_C(46504), UINT16_C(44374), UINT16_C(29306), UINT16_C(11993), UINT16_C(11268), UINT16_C(49903), UINT16_C(62884) },
      UINT8_C( 61) },
    { { UINT16_C( 6795), UINT16_C(44535), UINT16_C(24534), UINT16_C(56180), UINT16_C(63815), UINT16_C(47551), UINT16_C(19716), UINT16_C(29451) },
      { UINT16_C(46294), UINT16_C(11304), UINT16_C(41825), UINT16_C(15006), UINT16_C(41937), UINT16_C(49254), UINT16_C( 2661), UINT16_C(61621) },
      UINT8_C( 90) },
    { { UINT16_C(44324), UINT16_C(64413), UINT16_C( 4620), UINT16_C(21462), UINT16_C(38155), UINT16_C( 4108), UINT16_C( 6371), UINT16_C(47491) },
      { UINT16_C(44236), UINT16_C(11749), UINT16_C(33871), UINT16_C( 8296), UINT16_C(52775), UINT16_C(36064), UINT16_C(38361), UINT16_C(64893) },
      UINT8_C( 11) },
    { { UINT16_C( 6722), UINT16_C(20216), UINT16_C(52780), UINT16_C(14498), UINT16_C(44644), UINT16_C(18248), UINT16_C(52166), UINT16_C(37376) },
      { UINT16_C(58743), UINT16_C(50880), UINT16_C(10345), UINT16_C(37094), UINT16_C(50934), UINT16_C(53021), UINT16_C(39516), UINT16_C(40653) },
      UINT8_C( 68) },
    { { UINT16_C(50612), UINT16_C(57837), UINT16_C(36756), UINT16_C(63513), UINT16_C(24893), UINT16_C( 1087), UINT16_C(16172), UINT16_C(42134) },
      { UINT16_C(22052), UINT16_C(36458), UINT16_C(20862), UINT16_C(29982), UINT16_C(15127), UINT16_C(29508), UINT16_C( 4565), UINT16_C(35346) },
      UINT8_C(223) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epu16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 r = easysimd_mm_cmpgt_epu16_mask(a, b);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpgt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t a[4];
    uint32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT32_C( 830439375), UINT32_C(2050532155), UINT32_C(3759522020), UINT32_C(3355119486) },
      { UINT32_C( 577403986), UINT32_C(   2285898), UINT32_C(2120728243), UINT32_C(1369405570) },
      UINT8_C( 15) },
    { { UINT32_C( 864231159), UINT32_C(2796403649), UINT32_C(4186358395), UINT32_C( 599818961) },
      { UINT32_C(1212558334), UINT32_C(3209193484), UINT32_C(2839392039), UINT32_C( 536599592) },
      UINT8_C( 12) },
    { { UINT32_C(3142745850), UINT32_C(3026321209), UINT32_C(2477713602), UINT32_C(1756851818) },
      { UINT32_C(2779839897), UINT32_C(2355427429), UINT32_C(3493241256), UINT32_C(2012164477) },
      UINT8_C(  3) },
    { { UINT32_C(3895673519), UINT32_C(  60593217), UINT32_C(3868674684), UINT32_C(1380863673) },
      { UINT32_C(2969108043), UINT32_C(2671598839), UINT32_C(2070901758), UINT32_C(1408458404) },
      UINT8_C(  5) },
    { { UINT32_C(3795527328), UINT32_C( 937809850), UINT32_C(3676142626), UINT32_C( 355298506) },
      { UINT32_C(1640375658), UINT32_C(2147484546), UINT32_C( 452685686), UINT32_C(1852698318) },
      UINT8_C(  5) },
    { { UINT32_C(3478169620), UINT32_C(2701538943), UINT32_C(2105287602), UINT32_C(4203915919) },
      { UINT32_C(1364941007), UINT32_C(3520158811), UINT32_C(2582367691), UINT32_C(3490207931) },
      UINT8_C(  9) },
    { { UINT32_C(2157926400), UINT32_C(1092724110), UINT32_C(1488887496), UINT32_C( 391270472) },
      { UINT32_C(  74034601), UINT32_C(3587586569), UINT32_C(3278815495), UINT32_C( 445871642) },
      UINT8_C(  1) },
    { { UINT32_C(1570386639), UINT32_C(2677980375), UINT32_C(2734120026), UINT32_C(1454983597) },
      { UINT32_C(   5907191), UINT32_C(1691693149), UINT32_C( 203900146), UINT32_C(2301016762) },
      UINT8_C(  7) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__mmask8 r = easysimd_mm_cmpgt_epu32_mask(a, b);

    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpgt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint64_t a[2];
    uint64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT64_C( 3745823813174945754), UINT64_C(14825267419323340980) },
      { UINT64_C( 8821952607340326110), UINT64_C(18300734576288655103) },
      UINT8_C(  0) },
    { { UINT64_C(   92438468347328044), UINT64_C(  430184092602284288) },
      { UINT64_C(12672931097439293758), UINT64_C(13065334534968132402) },
      UINT8_C(  0) },
    { { UINT64_C( 2749468627107036891), UINT64_C( 5654632237376912716) },
      { UINT64_C(15383332758914029596), UINT64_C( 2816292709934491606) },
      UINT8_C(  2) },
    { { UINT64_C( 4649535552466378809), UINT64_C( 4508715802156128308) },
      { UINT64_C(12681722963193185669), UINT64_C(15988099342545070190) },
      UINT8_C(  0) },
    { { UINT64_C(15796309344200241155), UINT64_C(11456226271934016411) },
      { UINT64_C(18194196761785337847), UINT64_C(13001778533385080302) },
      UINT8_C(  0) },
    { { UINT64_C(  741935531875347656), UINT64_C( 1406183137062972109) },
      { UINT64_C(18326660981509475019), UINT64_C( 7877639785019346158) },
      UINT8_C(  0) },
    { { UINT64_C(12404647162348961196), UINT64_C( 9061547782423375506) },
      { UINT64_C(17541004901022606432), UINT64_C(12198814319681601080) },
      UINT8_C(  0) },
    { { UINT64_C(16315955752819042273), UINT64_C( 8433080554740580313) },
      { UINT64_C( 4750825959268409078), UINT64_C( 5041895658074570577) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpgt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 r = easysimd_mm_cmpgt_epu64_mask(a, b);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    uint8_t a[16];
    uint8_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(60606),
      { UINT8_C(154), UINT8_C( 54), UINT8_C( 96), UINT8_C( 99), UINT8_C(104), UINT8_C(129), UINT8_C( 79), UINT8_C(213),
        UINT8_C(113), UINT8_C( 37), UINT8_C( 95), UINT8_C( 54), UINT8_C( 55), UINT8_C(237), UINT8_C( 59), UINT8_C(111) },
      { UINT8_C( 99), UINT8_C( 28), UINT8_C(189), UINT8_C(206), UINT8_C( 31), UINT8_C(205), UINT8_C( 66), UINT8_C( 37),
        UINT8_C(235), UINT8_C(204), UINT8_C( 41), UINT8_C(134), UINT8_C(119), UINT8_C(231), UINT8_C(115), UINT8_C( 17) },
      UINT16_C(42130) },
    { UINT16_C(54046),
      { UINT8_C(117), UINT8_C(134), UINT8_C( 84), UINT8_C(196), UINT8_C( 91), UINT8_C(197), UINT8_C(233), UINT8_C(186),
        UINT8_C(251), UINT8_C( 32), UINT8_C(167), UINT8_C( 55), UINT8_C(143), UINT8_C( 10), UINT8_C( 83), UINT8_C( 77) },
      { UINT8_C(216), UINT8_C(114), UINT8_C( 26), UINT8_C( 26), UINT8_C(151), UINT8_C(  5), UINT8_C(231), UINT8_C(192),
        UINT8_C(139), UINT8_C( 94), UINT8_C(168), UINT8_C(254), UINT8_C(112), UINT8_C(198), UINT8_C(210), UINT8_C(229) },
      UINT16_C( 4366) },
    { UINT16_C( 9804),
      { UINT8_C(169), UINT8_C(167), UINT8_C(236), UINT8_C(147), UINT8_C( 98), UINT8_C(231), UINT8_C(179), UINT8_C(  9),
        UINT8_C( 30), UINT8_C( 67), UINT8_C( 20), UINT8_C(113), UINT8_C(144), UINT8_C(236), UINT8_C(228), UINT8_C(170) },
      { UINT8_C(  7), UINT8_C(123), UINT8_C(175), UINT8_C(238), UINT8_C( 60), UINT8_C( 58), UINT8_C( 76), UINT8_C(228),
        UINT8_C( 57), UINT8_C(188), UINT8_C(170), UINT8_C( 11), UINT8_C(161), UINT8_C(246), UINT8_C( 49), UINT8_C( 75) },
      UINT16_C(   68) },
    { UINT16_C( 7581),
      { UINT8_C(222),    UINT8_MAX, UINT8_C(  5), UINT8_C(145), UINT8_C(  9), UINT8_C( 35), UINT8_C(212), UINT8_C( 29),
        UINT8_C(149), UINT8_C(100), UINT8_C(  9), UINT8_C(121), UINT8_C( 14), UINT8_C( 16), UINT8_C(244), UINT8_C(189) },
      { UINT8_C(254), UINT8_C( 48), UINT8_C(248), UINT8_C( 75), UINT8_C( 20), UINT8_C( 49), UINT8_C(  7), UINT8_C(190),
        UINT8_C( 60), UINT8_C(169), UINT8_C(180), UINT8_C(109), UINT8_C(244), UINT8_C( 82), UINT8_C(139), UINT8_C(210) },
      UINT16_C( 2312) },
    { UINT16_C(36945),
      { UINT8_C( 99), UINT8_C( 90), UINT8_C(179), UINT8_C( 56), UINT8_C(119), UINT8_C( 72), UINT8_C(156), UINT8_C(129),
        UINT8_C(193), UINT8_C(171), UINT8_C(145), UINT8_C(182), UINT8_C(104), UINT8_C(144), UINT8_C(230), UINT8_C( 96) },
      { UINT8_C(219), UINT8_C(251), UINT8_C(145), UINT8_C(226), UINT8_C(185), UINT8_C(205), UINT8_C(139), UINT8_C(110),
        UINT8_C( 59), UINT8_C(127), UINT8_C(192), UINT8_C(198), UINT8_C( 81), UINT8_C( 17), UINT8_C( 86), UINT8_C(181) },
      UINT16_C( 4160) },
    { UINT16_C( 2412),
      { UINT8_C(237), UINT8_C(227), UINT8_C( 82), UINT8_C(137), UINT8_C(100), UINT8_C( 19), UINT8_C( 52), UINT8_C(246),
        UINT8_C(201), UINT8_C(157), UINT8_C(134), UINT8_C(176), UINT8_C(253), UINT8_C( 97), UINT8_C(171), UINT8_C(143) },
      { UINT8_C( 67), UINT8_C(100), UINT8_C( 92), UINT8_C(207), UINT8_C(210), UINT8_C(151), UINT8_C( 78), UINT8_C(146),
        UINT8_C( 93), UINT8_C(160), UINT8_C(164), UINT8_C(179), UINT8_C( 85), UINT8_C( 16), UINT8_C(189), UINT8_C( 66) },
      UINT16_C(  256) },
    { UINT16_C( 4083),
      { UINT8_C(203), UINT8_C( 88), UINT8_C( 34), UINT8_C(  0), UINT8_C( 78), UINT8_C(236), UINT8_C(157), UINT8_C(212),
        UINT8_C(156), UINT8_C(154), UINT8_C( 53), UINT8_C( 71), UINT8_C( 41), UINT8_C(120), UINT8_C(171), UINT8_C(134) },
      { UINT8_C( 71), UINT8_C(126), UINT8_C( 29), UINT8_C(150), UINT8_C( 16), UINT8_C(123), UINT8_C( 54), UINT8_C(180),
        UINT8_C( 46), UINT8_C(139), UINT8_C(196), UINT8_C(235), UINT8_C(205), UINT8_C(184), UINT8_C(250), UINT8_C(152) },
      UINT16_C( 1009) },
    { UINT16_C( 7440),
      { UINT8_C(152), UINT8_C( 94), UINT8_C(  9), UINT8_C( 53), UINT8_C( 50), UINT8_C(165), UINT8_C(208), UINT8_C(103),
        UINT8_C(236), UINT8_C(249), UINT8_C(223), UINT8_C(151), UINT8_C(127), UINT8_C( 39), UINT8_C( 21), UINT8_C(157) },
      { UINT8_C(189), UINT8_C( 38), UINT8_C( 24), UINT8_C(243), UINT8_C(218), UINT8_C( 70), UINT8_C(126), UINT8_C(159),
        UINT8_C( 50), UINT8_C( 75), UINT8_C( 87), UINT8_C( 44), UINT8_C(227), UINT8_C(103), UINT8_C( 73), UINT8_C(124) },
      UINT16_C( 3328) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epu8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epu8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__mmask16 r = easysimd_mm_mask_cmpgt_epu8_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint16_t a[8];
    uint16_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 75),
      { UINT16_C( 6855), UINT16_C(  611), UINT16_C(46909), UINT16_C(  377), UINT16_C(27689), UINT16_C(32703), UINT16_C(64574), UINT16_C(26526) },
      { UINT16_C(56056), UINT16_C(21567), UINT16_C(48387), UINT16_C( 9279), UINT16_C( 5996), UINT16_C(62356), UINT16_C( 5914), UINT16_C(57918) },
      UINT8_C( 64) },
    { UINT8_C( 50),
      { UINT16_C(58529), UINT16_C(22895), UINT16_C(28765), UINT16_C(51586), UINT16_C(  303), UINT16_C(11015), UINT16_C(28320), UINT16_C(31267) },
      { UINT16_C(30638), UINT16_C(27517), UINT16_C(41398), UINT16_C(52695), UINT16_C(52021), UINT16_C(19944), UINT16_C(51721), UINT16_C(43903) },
      UINT8_C(  0) },
    { UINT8_C(174),
      { UINT16_C( 1262), UINT16_C(24075), UINT16_C(54406), UINT16_C(34958), UINT16_C(47579), UINT16_C(18984), UINT16_C(41693), UINT16_C(21752) },
      { UINT16_C(25375), UINT16_C(49419), UINT16_C(55354), UINT16_C( 1526), UINT16_C(17344), UINT16_C(35343), UINT16_C(47810), UINT16_C(45112) },
      UINT8_C(  8) },
    { UINT8_C(190),
      { UINT16_C( 3907), UINT16_C( 5956), UINT16_C(52381), UINT16_C(22259), UINT16_C(15860), UINT16_C(38451), UINT16_C(34869), UINT16_C(39094) },
      { UINT16_C(30611), UINT16_C(27602), UINT16_C(55405), UINT16_C(45356), UINT16_C(46823), UINT16_C(41331), UINT16_C( 9455), UINT16_C(12895) },
      UINT8_C(128) },
    { UINT8_C( 51),
      { UINT16_C(19107), UINT16_C(28880), UINT16_C( 9789), UINT16_C(31332), UINT16_C(64346), UINT16_C(58031), UINT16_C(18353), UINT16_C(10357) },
      { UINT16_C(57369), UINT16_C(61845), UINT16_C(17932), UINT16_C(50136), UINT16_C(31162), UINT16_C(57010), UINT16_C(58584), UINT16_C(31761) },
      UINT8_C( 48) },
    { UINT8_C( 46),
      { UINT16_C(60641), UINT16_C( 1899), UINT16_C(58704), UINT16_C(19297), UINT16_C(17300), UINT16_C(56316), UINT16_C( 9400), UINT16_C(39413) },
      { UINT16_C(59066), UINT16_C(  165), UINT16_C(26815), UINT16_C(14522), UINT16_C(38938), UINT16_C(65297), UINT16_C(36265), UINT16_C(35373) },
      UINT8_C( 14) },
    { UINT8_C(121),
      { UINT16_C(37529), UINT16_C(32457), UINT16_C( 5619), UINT16_C(14099), UINT16_C(60945), UINT16_C(14063), UINT16_C(35043), UINT16_C(51952) },
      { UINT16_C(61486), UINT16_C(38537), UINT16_C(49579), UINT16_C(17329), UINT16_C(45266), UINT16_C(24557), UINT16_C(30685), UINT16_C(30424) },
      UINT8_C( 80) },
    { UINT8_C(  9),
      { UINT16_C(62882), UINT16_C(47101), UINT16_C(13320), UINT16_C(63176), UINT16_C(65059), UINT16_C(44250), UINT16_C(42222), UINT16_C(57306) },
      { UINT16_C(28717), UINT16_C(61066), UINT16_C(52513), UINT16_C(53697), UINT16_C( 8378), UINT16_C(12975), UINT16_C( 9721), UINT16_C(39739) },
      UINT8_C(  9) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epu16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epu16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 r = easysimd_mm_mask_cmpgt_epu16_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint32_t a[4];
    uint32_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 16),
      { UINT32_C(2472638747), UINT32_C(1858974235), UINT32_C(3642499669), UINT32_C(2358201723) },
      { UINT32_C(2514788778), UINT32_C(1350678576), UINT32_C(2269524519), UINT32_C(3096905373) },
      UINT8_C(  0) },
    { UINT8_C(127),
      { UINT32_C(2879015929), UINT32_C( 989858072), UINT32_C(1924586021), UINT32_C( 169624169) },
      { UINT32_C(3745165607), UINT32_C(1678215730), UINT32_C(3590491856), UINT32_C( 525777446) },
      UINT8_C(  0) },
    { UINT8_C(  6),
      { UINT32_C(4179544816), UINT32_C(2770229707), UINT32_C(1376686095), UINT32_C(1584994477) },
      { UINT32_C(3700512850), UINT32_C(4004377951), UINT32_C(2987688951), UINT32_C(3401069529) },
      UINT8_C(  0) },
    { UINT8_C(253),
      { UINT32_C( 818463702), UINT32_C(1916759522), UINT32_C(2468319611), UINT32_C(1675984394) },
      { UINT32_C(  79872527), UINT32_C(4076646511), UINT32_C(4157386436), UINT32_C(1022727782) },
      UINT8_C(  9) },
    { UINT8_C( 89),
      { UINT32_C( 725314749), UINT32_C(1034333868), UINT32_C(1263024845), UINT32_C(3780815647) },
      { UINT32_C( 508583789), UINT32_C( 165823323), UINT32_C(2775570959), UINT32_C(2365500367) },
      UINT8_C(  9) },
    { UINT8_C( 24),
      { UINT32_C(3905206074), UINT32_C(2561999198), UINT32_C(4105634121), UINT32_C(3127023963) },
      { UINT32_C( 739606761), UINT32_C(1010507362), UINT32_C( 957079693), UINT32_C( 424777951) },
      UINT8_C(  8) },
    { UINT8_C( 79),
      { UINT32_C( 380436757), UINT32_C(3076539830), UINT32_C(2517849341), UINT32_C( 914345398) },
      { UINT32_C(  26848483), UINT32_C(3364869607), UINT32_C(2041039073), UINT32_C( 784908313) },
      UINT8_C( 13) },
    { UINT8_C(193),
      { UINT32_C(3145155702), UINT32_C(4189597604), UINT32_C( 263147074), UINT32_C(2062739150) },
      { UINT32_C(1415705727), UINT32_C(1261775235), UINT32_C(2422517456), UINT32_C(3981546103) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epu32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__mmask8 r = easysimd_mm_mask_cmpgt_epu32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpgt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint64_t a[2];
    uint64_t b[2];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(249),
      { UINT64_C( 6072018200856598727), UINT64_C(16529038381760656124) },
      { UINT64_C(13053096982521773105), UINT64_C( 6349908967954109654) },
      UINT8_C(  0) },
    { UINT8_C(  0),
      { UINT64_C(14556679180371776482), UINT64_C(14590402534563605338) },
      { UINT64_C(10534776865587370355), UINT64_C( 6587776112076740868) },
      UINT8_C(  0) },
    { UINT8_C(206),
      { UINT64_C(15050520217289908942), UINT64_C( 4451923286659312079) },
      { UINT64_C(12884110625268770238), UINT64_C( 8429197700841028017) },
      UINT8_C(  0) },
    { UINT8_C(232),
      { UINT64_C(   27055508952054112), UINT64_C(11737192601699595452) },
      { UINT64_C(12834605250106862558), UINT64_C(11837374359819274712) },
      UINT8_C(  0) },
    { UINT8_C(198),
      { UINT64_C( 5291340973695906964), UINT64_C( 2234743588653470275) },
      { UINT64_C(15413652789621452642), UINT64_C( 2393308667162740643) },
      UINT8_C(  0) },
    { UINT8_C(241),
      { UINT64_C(15255327192788685780), UINT64_C( 5575007596144220820) },
      { UINT64_C(11452283691902594815), UINT64_C( 7599628200790679240) },
      UINT8_C(  1) },
    { UINT8_C( 32),
      { UINT64_C( 7665185461805177853), UINT64_C( 3018421909001362557) },
      { UINT64_C( 8856361773360492144), UINT64_C(16264601502454082890) },
      UINT8_C(  0) },
    { UINT8_C( 83),
      { UINT64_C( 7695611686646164761), UINT64_C(13951113239206408342) },
      { UINT64_C( 1240839162745961679), UINT64_C(11543807244180574055) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpgt_epu64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_mask_cmpgt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 r = easysimd_mm_mask_cmpgt_epu64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t a[32];
    int8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { { -INT8_C(  85), -INT8_C( 117),  INT8_C(  95), -INT8_C(  11),  INT8_C(  40), -INT8_C(  97), -INT8_C(  53),  INT8_C(  38),
        -INT8_C(  87), -INT8_C(  35), -INT8_C(  30),  INT8_C(  68),  INT8_C(  86), -INT8_C(  62), -INT8_C(  37),      INT8_MAX,
        -INT8_C(  34),  INT8_C(  48),  INT8_C(  94), -INT8_C(  67),  INT8_C( 104), -INT8_C(  19),  INT8_C(  70), -INT8_C(  64),
        -INT8_C(  67), -INT8_C(  16),  INT8_C(  16),  INT8_C( 104),  INT8_C(  19),  INT8_C(  45), -INT8_C( 121), -INT8_C(  65) },
      { -INT8_C(  72), -INT8_C(  26), -INT8_C(  76), -INT8_C(  32), -INT8_C( 123),      INT8_MIN,  INT8_C(   6),  INT8_C(  47),
         INT8_C(  93), -INT8_C(  23),  INT8_C( 115), -INT8_C(  77), -INT8_C(  85),  INT8_C(  78),  INT8_C(  50), -INT8_C( 119),
             INT8_MAX, -INT8_C( 112),  INT8_C(  70), -INT8_C(  25),  INT8_C( 126), -INT8_C( 115), -INT8_C(  89),  INT8_C(  59),
         INT8_C( 125), -INT8_C(  73), -INT8_C(  93), -INT8_C( 111), -INT8_C(  28),  INT8_C(  42),  INT8_C(  80), -INT8_C( 100) },
      UINT32_C(3194394684) },
    { {  INT8_C(  16),  INT8_C(   4),  INT8_C( 124), -INT8_C( 107), -INT8_C( 124), -INT8_C( 126), -INT8_C(  60), -INT8_C(  31),
         INT8_C( 107),  INT8_C(  56), -INT8_C( 107),  INT8_C(  22), -INT8_C( 122), -INT8_C(  57), -INT8_C(  97),  INT8_C(   5),
         INT8_C(  88), -INT8_C(  26), -INT8_C(  20), -INT8_C(  42),  INT8_C( 115), -INT8_C( 109),  INT8_C(  17), -INT8_C(  16),
         INT8_C(  74), -INT8_C(  76), -INT8_C( 127),  INT8_C(  46), -INT8_C(  34), -INT8_C(  47), -INT8_C(  54), -INT8_C(  18) },
      { -INT8_C(  42),  INT8_C(  70), -INT8_C( 125),  INT8_C(  90), -INT8_C(  55),  INT8_C(  72),  INT8_C(  60),  INT8_C(  52),
             INT8_MIN, -INT8_C(  47),  INT8_C(  75),  INT8_C(   6), -INT8_C( 104), -INT8_C(  22),  INT8_C(  12), -INT8_C(  16),
        -INT8_C(  48), -INT8_C(   8), -INT8_C(  58),  INT8_C(  67), -INT8_C( 116), -INT8_C(  41),  INT8_C(  52), -INT8_C(  42),
        -INT8_C( 117), -INT8_C(  75),  INT8_C(   5),  INT8_C( 105), -INT8_C( 121), -INT8_C(  49),  INT8_C(  87),  INT8_C(  93) },
      UINT32_C( 831884037) },
    { {  INT8_C(  22), -INT8_C(  37), -INT8_C(  73), -INT8_C(  33),  INT8_C(  35), -INT8_C(  13),  INT8_C(  19), -INT8_C(  93),
        -INT8_C(  60),  INT8_C(  94), -INT8_C(  87),  INT8_C(  93),  INT8_C(  73), -INT8_C(  75),  INT8_C(  77),  INT8_C(  25),
        -INT8_C(  82),  INT8_C(  20),  INT8_C(  93),  INT8_C(  58), -INT8_C(  21), -INT8_C( 111),  INT8_C(  16),  INT8_C( 119),
         INT8_C(  70),  INT8_C(  21), -INT8_C(  32), -INT8_C(  51), -INT8_C(  27),  INT8_C(  56),  INT8_C(  42), -INT8_C(   5) },
      {  INT8_C(  19), -INT8_C(  30), -INT8_C(  38),  INT8_C(  54), -INT8_C(  43), -INT8_C(  19), -INT8_C(  39), -INT8_C( 102),
         INT8_C(  76), -INT8_C( 126), -INT8_C(   9), -INT8_C( 107),  INT8_C(  56),  INT8_C(  68), -INT8_C(  82), -INT8_C(  26),
         INT8_C(  88),  INT8_C(  11),  INT8_C(  32),  INT8_C(  68), -INT8_C( 100),  INT8_C(  48), -INT8_C(  69), -INT8_C(  29),
         INT8_C(  70), -INT8_C( 101), -INT8_C(  80),  INT8_C(  43), -INT8_C(  45), -INT8_C(  37),  INT8_C(  38), -INT8_C(  26) },
      UINT32_C(4141275889) },
    { { -INT8_C(  67),  INT8_C(   0),  INT8_C(  28), -INT8_C( 110), -INT8_C(  19), -INT8_C(  11),  INT8_C(  44),  INT8_C(  57),
         INT8_C( 120),  INT8_C(  35), -INT8_C(  50), -INT8_C(  80),  INT8_C( 104),  INT8_C( 125), -INT8_C( 106), -INT8_C(  64),
        -INT8_C( 120), -INT8_C(  74),  INT8_C(   4),  INT8_C(  37), -INT8_C(  26), -INT8_C(  65),  INT8_C(   8),  INT8_C(  44),
         INT8_C(  91), -INT8_C(  72),  INT8_C(  87),  INT8_C(  46), -INT8_C( 109),  INT8_C( 125),  INT8_C(  21),  INT8_C(  80) },
      {  INT8_C( 125),  INT8_C(  49), -INT8_C(  29),  INT8_C( 107),  INT8_C(  39),  INT8_C(  15), -INT8_C(  92), -INT8_C(  97),
         INT8_C(  51),  INT8_C( 115),  INT8_C(  79), -INT8_C( 101), -INT8_C(  16), -INT8_C(  27),  INT8_C(  91),  INT8_C( 120),
        -INT8_C( 101),  INT8_C(  96), -INT8_C(  99), -INT8_C( 127),  INT8_C(  31), -INT8_C(  91), -INT8_C(  82),  INT8_C( 122),
         INT8_C(  94),  INT8_C(   5), -INT8_C(  87), -INT8_C(  15), -INT8_C( 125), -INT8_C(  66),  INT8_C(  66),  INT8_C(   0) },
      UINT32_C(3161209284) },
    { { -INT8_C(  17),  INT8_C(  37),  INT8_C( 107),  INT8_C(  22),  INT8_C(  52),  INT8_C(  16), -INT8_C(  75),  INT8_C( 103),
        -INT8_C( 125),  INT8_C(   4),  INT8_C(   2),  INT8_C( 115), -INT8_C(  23),  INT8_C(  94), -INT8_C(  21), -INT8_C( 124),
        -INT8_C(  66), -INT8_C( 119),  INT8_C(   6), -INT8_C(  35),  INT8_C(  46), -INT8_C(  76),  INT8_C(  88), -INT8_C( 116),
        -INT8_C(  71),  INT8_C(   1),  INT8_C( 126),  INT8_C(  60), -INT8_C(  65), -INT8_C(  64),  INT8_C(  61), -INT8_C(  82) },
      { -INT8_C(  27), -INT8_C(  88), -INT8_C(  59),  INT8_C(  25), -INT8_C(  72),  INT8_C( 122), -INT8_C( 127),  INT8_C(  59),
             INT8_MAX, -INT8_C( 125), -INT8_C(  82),  INT8_C( 104), -INT8_C(  31), -INT8_C( 102), -INT8_C(  19), -INT8_C(  97),
         INT8_C(  35), -INT8_C(  13),  INT8_C( 125),  INT8_C(  81), -INT8_C(  89), -INT8_C(  43), -INT8_C(  34),  INT8_C(  96),
        -INT8_C(  42),  INT8_C(  92), -INT8_C(  99), -INT8_C( 107),  INT8_C(  28), -INT8_C(  38),  INT8_C(  67),  INT8_C(   1) },
      UINT32_C( 206585559) },
    { { -INT8_C( 126),  INT8_C(   8),  INT8_C(  26),  INT8_C(  59), -INT8_C( 125), -INT8_C( 101),  INT8_C( 118),  INT8_C(   2),
         INT8_C(  31),  INT8_C(  37),  INT8_C( 106),  INT8_C(   0), -INT8_C(  65),  INT8_C(  87), -INT8_C(  96), -INT8_C(  30),
         INT8_C(  74),  INT8_C(  29),  INT8_C(  51), -INT8_C(  15), -INT8_C(  14),  INT8_C(  17),  INT8_C(  82), -INT8_C(  56),
         INT8_C( 109), -INT8_C(  17),  INT8_C(  93), -INT8_C( 119), -INT8_C(  55), -INT8_C(  96), -INT8_C( 118),  INT8_C(  75) },
      { -INT8_C(  87), -INT8_C(  91), -INT8_C( 122),  INT8_C(  44),  INT8_C(  64), -INT8_C(   3),  INT8_C(  46),  INT8_C(  95),
         INT8_C(  34), -INT8_C( 104),  INT8_C(  96), -INT8_C(  31), -INT8_C(  16),  INT8_C(   0), -INT8_C(  61),  INT8_C(  58),
         INT8_C(  29), -INT8_C(  10),  INT8_C(  44),  INT8_C(  15),  INT8_C(   8),  INT8_C( 126), -INT8_C(  41),  INT8_C( 117),
         INT8_C( 109),  INT8_C(  52), -INT8_C(   1),  INT8_C(  54), -INT8_C(  44), -INT8_C( 119), -INT8_C( 127),  INT8_C( 125) },
      UINT32_C(1682386510) },
    { {  INT8_C(  46),  INT8_C(   8), -INT8_C(  87),  INT8_C( 111),  INT8_C(   5), -INT8_C(  41), -INT8_C(  50),  INT8_C(  39),
         INT8_C( 112),  INT8_C(  46),  INT8_C(   8),  INT8_C(  96),  INT8_C(  46), -INT8_C(  53), -INT8_C( 102),  INT8_C(  75),
        -INT8_C(  63), -INT8_C(  58),  INT8_C(  90), -INT8_C(  55),  INT8_C(  68),  INT8_C(  49),  INT8_C(  63), -INT8_C(  79),
         INT8_C( 101),  INT8_C(  62), -INT8_C(  25),  INT8_C(  58), -INT8_C(  57),  INT8_C( 105), -INT8_C(  73), -INT8_C(  10) },
      {  INT8_C( 113),  INT8_C(  97),  INT8_C( 101),  INT8_C( 118),  INT8_C(  56),  INT8_C(  51), -INT8_C(  99), -INT8_C(  88),
         INT8_C(  98), -INT8_C(  91),  INT8_C(   8), -INT8_C( 112),  INT8_C( 112), -INT8_C(  93), -INT8_C(  36),  INT8_C(  49),
         INT8_C( 105),  INT8_C(  54), -INT8_C(   5), -INT8_C(  82),  INT8_C( 104),  INT8_C(  58),  INT8_C(  95), -INT8_C(  51),
         INT8_C( 120),  INT8_C(  71),  INT8_C(   7),  INT8_C(  63), -INT8_C(  80), -INT8_C(  65),  INT8_C(  53),  INT8_C(  33) },
      UINT32_C( 806136768) },
    { {  INT8_C(  32), -INT8_C( 102), -INT8_C( 105),  INT8_C(  88), -INT8_C(  50),  INT8_C(  52),  INT8_C(   1),  INT8_C(  48),
        -INT8_C(  39),  INT8_C(   9), -INT8_C(  64),  INT8_C(  73), -INT8_C(  84), -INT8_C( 100),  INT8_C( 122),  INT8_C(  22),
        -INT8_C(  45),  INT8_C( 117), -INT8_C(  60),  INT8_C(  59), -INT8_C(  81),  INT8_C(  35),  INT8_C(   8),  INT8_C(  39),
         INT8_C( 106),  INT8_C(  16),  INT8_C( 103),  INT8_C(  26), -INT8_C(  49), -INT8_C( 100),  INT8_C(  59), -INT8_C(  17) },
      {  INT8_C(  55), -INT8_C(  46),  INT8_C(  71),  INT8_C(   5),  INT8_C(   6),  INT8_C(  72),  INT8_C(  53), -INT8_C(  33),
         INT8_C(  82), -INT8_C(  11),  INT8_C(  40), -INT8_C(   2), -INT8_C( 110), -INT8_C(  93),  INT8_C(  20),  INT8_C( 101),
         INT8_C(  24), -INT8_C(  40), -INT8_C(  96), -INT8_C(  56), -INT8_C(   4), -INT8_C(  88), -INT8_C(  17),  INT8_C( 102),
        -INT8_C(  72),  INT8_C(  86), -INT8_C( 127), -INT8_C( 121), -INT8_C(  13), -INT8_C(  68),  INT8_C( 118),  INT8_C(  42) },
      UINT32_C( 225335944) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpgt_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 r = easysimd_mm256_cmpgt_epi8_mask(a, b);

    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[16];
    int16_t b[16];
    easysimd__mmask16 r;
  } test_vec[16] = {
    { {  INT16_C( 32247),  INT16_C( 18373), -INT16_C( 12624), -INT16_C( 10612),  INT16_C(    57),  INT16_C( 23079),  INT16_C( 13671),  INT16_C( 21447),
         INT16_C( 16174),  INT16_C(  7570),  INT16_C(  6461),  INT16_C( 15104), -INT16_C(  8285),  INT16_C( 25417), -INT16_C(  4580),  INT16_C(  5107) },
      { -INT16_C( 18325),  INT16_C(  7002), -INT16_C(  6266), -INT16_C( 16399),  INT16_C(  6375),  INT16_C( 19993), -INT16_C(  8114),  INT16_C( 31905),
         INT16_C( 13344),  INT16_C( 23962), -INT16_C( 26035), -INT16_C(  3944), -INT16_C(  7559), -INT16_C( 27308),  INT16_C( 18384),  INT16_C( 15528) },
      UINT16_C(11627) },
    { {  INT16_C(  1023), -INT16_C( 31401),  INT16_C( 18922), -INT16_C( 11964),  INT16_C( 23905), -INT16_C( 20704), -INT16_C( 16067),  INT16_C( 23852),
        -INT16_C( 14603),  INT16_C( 17082),  INT16_C( 21344), -INT16_C(  9933), -INT16_C( 30923),  INT16_C(  1391),  INT16_C(  6094), -INT16_C( 12991) },
      { -INT16_C( 26342),  INT16_C(  1106), -INT16_C( 26910),  INT16_C( 17366), -INT16_C(  2317),  INT16_C( 12531),  INT16_C(  8119), -INT16_C( 21106),
         INT16_C( 18661),  INT16_C( 17903),  INT16_C(  8859), -INT16_C( 12258), -INT16_C( 29271),  INT16_C( 30678),  INT16_C(  6053), -INT16_C( 16572) },
      UINT16_C(52373) },
    { { -INT16_C( 26960), -INT16_C( 27964), -INT16_C( 26068),  INT16_C(  8150), -INT16_C( 13936),  INT16_C( 18256), -INT16_C(  8472), -INT16_C( 12812),
        -INT16_C(  7130), -INT16_C( 15854),  INT16_C( 12294), -INT16_C( 20334),  INT16_C( 26814),  INT16_C( 25383),  INT16_C( 27776),  INT16_C( 12322) },
      { -INT16_C(  6654),  INT16_C( 12227), -INT16_C( 26240),  INT16_C(  4174), -INT16_C( 24990),  INT16_C( 19032),  INT16_C( 19580), -INT16_C( 23785),
         INT16_C( 10544),  INT16_C( 14181), -INT16_C(  2215),  INT16_C(  6119),  INT16_C(  3680), -INT16_C(  8070), -INT16_C( 25222),  INT16_C( 32016) },
      UINT16_C(29852) },
    { { -INT16_C( 11389),  INT16_C(  1196), -INT16_C(  1428), -INT16_C( 12780),  INT16_C( 27801),  INT16_C(  5400),  INT16_C( 12217), -INT16_C(  5704),
         INT16_C(  7512), -INT16_C( 19936),  INT16_C(  1813),  INT16_C( 30153),  INT16_C( 17430), -INT16_C( 28587),  INT16_C( 26081),  INT16_C( 25613) },
      { -INT16_C( 18119), -INT16_C( 23192),  INT16_C( 32180),  INT16_C( 19828), -INT16_C( 29463), -INT16_C( 23966),  INT16_C(  7100),  INT16_C(  5260),
        -INT16_C( 21448),  INT16_C( 19910), -INT16_C( 28492), -INT16_C( 13630),  INT16_C(  6100), -INT16_C( 19110),  INT16_C( 26749), -INT16_C( 18919) },
      UINT16_C(40307) },
    { { -INT16_C( 32223), -INT16_C( 10917), -INT16_C( 12289), -INT16_C(  6110), -INT16_C( 31396),  INT16_C(  6283),  INT16_C(  6048), -INT16_C( 10196),
        -INT16_C(  3133),  INT16_C( 30502), -INT16_C(  6013),  INT16_C( 22337), -INT16_C( 25600),  INT16_C( 32012),  INT16_C(  9476),  INT16_C(  9523) },
      { -INT16_C( 29017), -INT16_C( 22789),  INT16_C(  7518), -INT16_C( 17777),  INT16_C(  6818),  INT16_C( 17106), -INT16_C(   463), -INT16_C(  3045),
         INT16_C( 16881),  INT16_C( 29804), -INT16_C( 21207),  INT16_C( 10699), -INT16_C( 10423),  INT16_C( 19878), -INT16_C(  9731), -INT16_C( 23437) },
      UINT16_C(61002) },
    { {  INT16_C( 28264), -INT16_C( 14773), -INT16_C(  9589),  INT16_C( 11904),  INT16_C( 21236),  INT16_C(  9584), -INT16_C( 29872),  INT16_C( 16921),
        -INT16_C( 31284), -INT16_C(  2378), -INT16_C( 32205),  INT16_C( 31775), -INT16_C( 14759),  INT16_C( 22218),  INT16_C( 15775),  INT16_C(  2043) },
      {  INT16_C( 18091),  INT16_C( 14029),  INT16_C( 19744),  INT16_C(  5220), -INT16_C( 10849), -INT16_C(  4039),  INT16_C( 21088),  INT16_C( 11570),
        -INT16_C(  5928),  INT16_C(  2851),  INT16_C( 17002), -INT16_C( 15225),  INT16_C( 20744), -INT16_C( 22502),  INT16_C(  5518),  INT16_C( 14767) },
      UINT16_C(26809) },
    { {  INT16_C( 32091),  INT16_C( 31600), -INT16_C( 11062),  INT16_C( 27279), -INT16_C( 14167),  INT16_C(  2650), -INT16_C( 29669), -INT16_C(  3273),
         INT16_C( 23156), -INT16_C(  8194), -INT16_C( 31332), -INT16_C( 23133), -INT16_C( 16937),  INT16_C( 25933), -INT16_C(   813),  INT16_C( 11935) },
      {  INT16_C(  3961),  INT16_C( 17578),  INT16_C( 14819), -INT16_C( 29266),  INT16_C(  2050),  INT16_C(  7575), -INT16_C( 12652),  INT16_C(  2064),
         INT16_C(  3624), -INT16_C( 15129), -INT16_C( 30061),  INT16_C( 27241), -INT16_C( 18872),  INT16_C(  7120),  INT16_C( 28595),  INT16_C( 11337) },
      UINT16_C(45835) },
    { { -INT16_C(  3202),  INT16_C( 24944),  INT16_C(  7725),  INT16_C( 12270), -INT16_C( 31450), -INT16_C( 17844),  INT16_C( 23635),  INT16_C( 31683),
        -INT16_C( 21910), -INT16_C(   704), -INT16_C( 22219),  INT16_C( 32104),  INT16_C( 14432),  INT16_C(  5016), -INT16_C(  7769),  INT16_C(  9535) },
      { -INT16_C( 20267),  INT16_C(   646),  INT16_C( 30158), -INT16_C(  2767),  INT16_C( 32250),  INT16_C( 20143),  INT16_C( 29401),  INT16_C( 17353),
         INT16_C(  2333),  INT16_C( 21056), -INT16_C( 22349),  INT16_C(  5071),  INT16_C( 26592), -INT16_C( 30938),  INT16_C( 25928),  INT16_C(  7596) },
      UINT16_C(44171) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epi16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpgt_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 r = easysimd_mm256_cmpgt_epi16_mask(a, b);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[8];
    int32_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { {  INT32_C(   704341566),  INT32_C(  1214601820),  INT32_C(  1552079130),  INT32_C(  1254420885), -INT32_C(  1362778956), -INT32_C(  1364155066), -INT32_C(   656637900),  INT32_C(  1785961260) },
      {  INT32_C(  1569943297), -INT32_C(   458884919),  INT32_C(  1732258002), -INT32_C(   844037095), -INT32_C(   293898584),  INT32_C(  1352477723), -INT32_C(   668436052),  INT32_C(   356686868) },
      UINT8_C(202) },
    { { -INT32_C(   730606325), -INT32_C(  1615324979),  INT32_C(  1493694528), -INT32_C(  1524188932),  INT32_C(  1251189295),  INT32_C(  2056925390), -INT32_C(  1101872214),  INT32_C(  1792316511) },
      {  INT32_C(   910051177), -INT32_C(  1596524705), -INT32_C(   319169041), -INT32_C(   997121899), -INT32_C(  1878055742), -INT32_C(    16078507), -INT32_C(   876782484),  INT32_C(  1496682992) },
      UINT8_C(180) },
    { {  INT32_C(   948925656),  INT32_C(  1524131179), -INT32_C(   666447550), -INT32_C(  1281566735),  INT32_C(  1363389436), -INT32_C(  1051701931), -INT32_C(  1718809175),  INT32_C(  2012398239) },
      { -INT32_C(  1582333386),  INT32_C(   721192935),  INT32_C(  1241662041),  INT32_C(   385785370), -INT32_C(  1620623030),  INT32_C(   945862543),  INT32_C(  1674767812), -INT32_C(   421804880) },
      UINT8_C(147) },
    { {  INT32_C(   780700230),  INT32_C(  1800963090), -INT32_C(   524985658),  INT32_C(  1123464184), -INT32_C(  2065605131), -INT32_C(   641973740), -INT32_C(   549679569), -INT32_C(  1715071149) },
      { -INT32_C(  1262006622), -INT32_C(  1742790702),  INT32_C(  1920586873),  INT32_C(  2108977032), -INT32_C(   519989555),  INT32_C(   129678808), -INT32_C(  1612187828), -INT32_C(  1338462962) },
      UINT8_C( 67) },
    { { -INT32_C(   849084165), -INT32_C(  1721400545), -INT32_C(   536093096),  INT32_C(   459128654),  INT32_C(   771513941),  INT32_C(  1731573531), -INT32_C(  1140450130), -INT32_C(   999473207) },
      {  INT32_C(  1586614591), -INT32_C(  1376258475),  INT32_C(   596443861),  INT32_C(   389999298),  INT32_C(  1665481288), -INT32_C(  1614054671),  INT32_C(  1599918486),  INT32_C(  1327745296) },
      UINT8_C( 40) },
    { { -INT32_C(   273763174),  INT32_C(  2140972458),  INT32_C(  1789012392),  INT32_C(  1535238163),  INT32_C(   213894683), -INT32_C(   693335488),  INT32_C(  1815414875),  INT32_C(  1807440081) },
      { -INT32_C(  1218746100), -INT32_C(  1221134577),  INT32_C(   874633505), -INT32_C(   728718663), -INT32_C(  1444851863),  INT32_C(   880840153),  INT32_C(  1721808277),  INT32_C(   449928206) },
      UINT8_C(223) },
    { { -INT32_C(   724489019),  INT32_C(  1166739492), -INT32_C(  1703301919), -INT32_C(  1200682673),  INT32_C(   828461144),  INT32_C(  1919345117), -INT32_C(  1512569193),  INT32_C(   683649378) },
      { -INT32_C(    84110890),  INT32_C(  2050984089), -INT32_C(  2095794124),  INT32_C(   440107970), -INT32_C(  1337156141),  INT32_C(   354595454),  INT32_C(   465238712),  INT32_C(  2051242660) },
      UINT8_C( 52) },
    { { -INT32_C(  1535885557), -INT32_C(    65096761),  INT32_C(   763310699), -INT32_C(  1991722058), -INT32_C(   700804008), -INT32_C(    18064314), -INT32_C(    82205097),  INT32_C(   729111584) },
      {  INT32_C(  1674570140),  INT32_C(   123727260), -INT32_C(   701112544), -INT32_C(   228622950),  INT32_C(  1472764177),  INT32_C(  1297462518),  INT32_C(  2068344667),  INT32_C(  1739046347) },
      UINT8_C(  4) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epi32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpgt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 r = easysimd_mm256_cmpgt_epi32_mask(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[4];
    int64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { {  INT64_C(  505593838372618661),  INT64_C( 2499953540411598877), -INT64_C( 4083214473827183828),  INT64_C( 2417463763889323034) },
      { -INT64_C( 5259651744296509425),  INT64_C(   89882692361009054), -INT64_C( 6323273167240230041), -INT64_C( 2498831707368570956) },
      UINT8_C( 15) },
    { { -INT64_C( 2950182965082635331),  INT64_C( 7774676427839156495), -INT64_C( 5371349589424879553),  INT64_C( 6920969000896054882) },
      { -INT64_C( 3851486171310333061), -INT64_C(  871845974959554186),  INT64_C( 5625900748328979036), -INT64_C( 6016938633557247620) },
      UINT8_C( 11) },
    { {  INT64_C( 8283508377139623534), -INT64_C( 7847571596185329041), -INT64_C( 8514051738843424867),  INT64_C( 5753955322006361852) },
      { -INT64_C( 7436731434821731491),  INT64_C( 3015392545648457743), -INT64_C( 3944718087211464278),  INT64_C( 8003688262511553250) },
      UINT8_C(  1) },
    { {  INT64_C(  490492099398502924),  INT64_C(  576219262772546123),  INT64_C( 2342436125102027501),  INT64_C( 6155283013856098795) },
      {  INT64_C( 1259980704684127103), -INT64_C( 2790006442245437213), -INT64_C(  866919800531449958),  INT64_C( 5705317255482432398) },
      UINT8_C( 14) },
    { { -INT64_C(  835908038609684338), -INT64_C( 3400363717652361718),  INT64_C(  395648777566315138),  INT64_C( 6240310989546089851) },
      { -INT64_C( 5925027764935892799), -INT64_C(  547623351508207008),  INT64_C( 2977161577276575445), -INT64_C(  464979931731579283) },
      UINT8_C(  9) },
    { { -INT64_C( 6142749049187780753), -INT64_C( 5909945544096062324), -INT64_C( 1401380902810321227), -INT64_C( 9175612474309672327) },
      { -INT64_C( 7793387864853350939),  INT64_C( 8447505782275447223), -INT64_C(  470230884250209009),  INT64_C( 3313052046645107561) },
      UINT8_C(  1) },
    { { -INT64_C( 7538404686108521016), -INT64_C( 7457009316736337843),  INT64_C( 3174931922558540950), -INT64_C( 3120817466647332550) },
      { -INT64_C( 8167835097196458962), -INT64_C( 4314414158316778242),  INT64_C( 2336219102061479547), -INT64_C( 2196820162015718749) },
      UINT8_C(  5) },
    { {  INT64_C( 3039814491449599584),  INT64_C( 7374312245262666595), -INT64_C( 4675563125735083753),  INT64_C( 8338399402098868258) },
      { -INT64_C( 6742376557906188520),  INT64_C(  675691616483352591),  INT64_C( 4077203614345856442),  INT64_C( 1038859224736400355) },
      UINT8_C( 11) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epi64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpgt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 r = easysimd_mm256_cmpgt_epi64_mask(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    int8_t a[32];
    int8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C(3013622953),
      { -INT8_C(  32), -INT8_C(  11), -INT8_C(  83), -INT8_C(  98),  INT8_C(  64), -INT8_C(  58), -INT8_C(   1),  INT8_C(  88),
         INT8_C(  64), -INT8_C(  53),  INT8_C(  68), -INT8_C(  33), -INT8_C( 125), -INT8_C(  98), -INT8_C(  19),  INT8_C(  91),
         INT8_C(  11), -INT8_C( 117), -INT8_C(  31), -INT8_C(  63),  INT8_C(  76), -INT8_C(  36),  INT8_C(  39), -INT8_C(  43),
        -INT8_C(  42), -INT8_C(  62), -INT8_C(  63),      INT8_MAX, -INT8_C(   2),  INT8_C(  97),  INT8_C(  50), -INT8_C(  34) },
      {  INT8_C(  86), -INT8_C(  33),  INT8_C( 124), -INT8_C( 105), -INT8_C(  91),  INT8_C( 124), -INT8_C(  17), -INT8_C(  26),
         INT8_C(  71),  INT8_C(  51), -INT8_C(  59), -INT8_C(  54), -INT8_C(  46), -INT8_C(  78),  INT8_C(  37), -INT8_C(  35),
         INT8_C(  61),  INT8_C(   6), -INT8_C(  98), -INT8_C( 118), -INT8_C(  29), -INT8_C(  59),  INT8_C(  95), -INT8_C(  71),
        -INT8_C( 121),  INT8_C(  32),  INT8_C(  57), -INT8_C( 123), -INT8_C( 126),  INT8_C( 107),  INT8_C( 100), -INT8_C(  40) },
      UINT32_C(2443185288) },
    { UINT32_C(4033863755),
      {  INT8_C(  92),  INT8_C(  94), -INT8_C(  42), -INT8_C(  93), -INT8_C( 110), -INT8_C( 101),  INT8_C( 109),  INT8_C( 100),
         INT8_C(  77), -INT8_C( 110),  INT8_C(  65), -INT8_C( 117), -INT8_C( 103), -INT8_C(  33),  INT8_C(  21),  INT8_C( 124),
        -INT8_C(  91),  INT8_C( 116),  INT8_C(  53),  INT8_C(  44), -INT8_C( 108),  INT8_C( 110), -INT8_C(  78),  INT8_C(  22),
        -INT8_C(  38),  INT8_C(  22), -INT8_C(  17),  INT8_C(  37), -INT8_C(  10),  INT8_C(  94),  INT8_C(  21),  INT8_C(  83) },
      { -INT8_C(  67), -INT8_C(  20), -INT8_C(  10),  INT8_C(  79), -INT8_C( 121),  INT8_C( 100), -INT8_C(  77), -INT8_C(  43),
        -INT8_C(  10), -INT8_C(  12),  INT8_C(  96), -INT8_C( 113), -INT8_C(  45),  INT8_C( 117),  INT8_C(  11),  INT8_C( 120),
        -INT8_C(  23),  INT8_C(  65), -INT8_C(  91),  INT8_C( 125), -INT8_C(  81),  INT8_C(  87), -INT8_C( 108), -INT8_C( 119),
         INT8_C( 109), -INT8_C( 125), -INT8_C(  82),  INT8_C(  99), -INT8_C(  31), -INT8_C(  60), -INT8_C(  74), -INT8_C(  98) },
      UINT32_C(4033265731) },
    { UINT32_C( 938323376),
      {  INT8_C(  17), -INT8_C(  96),  INT8_C(  12),  INT8_C(   7), -INT8_C( 108),  INT8_C( 108), -INT8_C( 105),  INT8_C( 104),
        -INT8_C(  31), -INT8_C(  94), -INT8_C(  32), -INT8_C(  54), -INT8_C(  29), -INT8_C( 123),  INT8_C(  72), -INT8_C( 109),
        -INT8_C(  36), -INT8_C(  36),  INT8_C(  28),  INT8_C(  73),  INT8_C(  95), -INT8_C(  53), -INT8_C(  83),  INT8_C(  64),
        -INT8_C( 113),  INT8_C(  99), -INT8_C(  33),  INT8_C(  63),  INT8_C(  16), -INT8_C(  52),  INT8_C( 118),  INT8_C(  33) },
      {  INT8_C( 109), -INT8_C( 125),  INT8_C(  41),  INT8_C(   1), -INT8_C(  17), -INT8_C(  64),  INT8_C( 105), -INT8_C(  47),
         INT8_C(  98),  INT8_C(  74), -INT8_C( 101),  INT8_C(  70), -INT8_C(  49), -INT8_C(  29), -INT8_C(  39), -INT8_C(  84),
        -INT8_C(  65), -INT8_C(  11), -INT8_C(  11),  INT8_C(  30), -INT8_C(  64), -INT8_C(  94),  INT8_C(  95),  INT8_C(  79),
         INT8_C(   6),  INT8_C(  62), -INT8_C( 114),  INT8_C(  22),  INT8_C(  10),  INT8_C(   5),  INT8_C(  56),  INT8_C( 119) },
      UINT32_C( 372049056) },
    { UINT32_C(2004443528),
      {  INT8_C(  33), -INT8_C(  30),  INT8_C(  72), -INT8_C( 125),  INT8_C(  44), -INT8_C(  28), -INT8_C(  55), -INT8_C(   4),
        -INT8_C(  57), -INT8_C(  94), -INT8_C(  88), -INT8_C( 121), -INT8_C( 104), -INT8_C(  99), -INT8_C(  91),  INT8_C(  88),
         INT8_C(  64),  INT8_C(   4), -INT8_C(  88),  INT8_C(  70),  INT8_C(  66),  INT8_C(  54),  INT8_C(  92),  INT8_C(  77),
         INT8_C(  59), -INT8_C( 108), -INT8_C(  60), -INT8_C(  61), -INT8_C(  11),  INT8_C(  61),  INT8_C(  59),  INT8_C(  22) },
      {  INT8_C(  32), -INT8_C( 125), -INT8_C( 102),  INT8_C(  76),  INT8_C( 103),  INT8_C(  99),  INT8_C(  72),  INT8_C(  47),
         INT8_C(   6), -INT8_C(  16), -INT8_C(  74), -INT8_C(  98), -INT8_C( 114),  INT8_C(  91), -INT8_C(  10), -INT8_C(  50),
         INT8_C(  96), -INT8_C(  98),  INT8_C(  20), -INT8_C(  94), -INT8_C(  43),  INT8_C( 112), -INT8_C(  17),  INT8_C(  16),
         INT8_C(   5), -INT8_C(  76), -INT8_C(  44), -INT8_C(   6), -INT8_C(  15),  INT8_C(  15),  INT8_C(  17),  INT8_C(  17) },
      UINT32_C(1901592576) },
    { UINT32_C(4200508306),
      {  INT8_C(  14), -INT8_C(  90),  INT8_C(  41),  INT8_C(  20), -INT8_C( 105), -INT8_C(  33), -INT8_C(  78),  INT8_C(  37),
         INT8_C(  58), -INT8_C(  87), -INT8_C(  13), -INT8_C( 102),  INT8_C(  71),  INT8_C(   7),  INT8_C(  61),  INT8_C(  28),
         INT8_C( 119),  INT8_C(  44),  INT8_C(  45),  INT8_C( 124), -INT8_C(  32),  INT8_C(   1),  INT8_C( 119), -INT8_C(  46),
         INT8_C(  16), -INT8_C( 120), -INT8_C(  29), -INT8_C(  94),  INT8_C(  51),  INT8_C(  65), -INT8_C( 100),  INT8_C(  65) },
      { -INT8_C(  24), -INT8_C(  59),  INT8_C(  86),      INT8_MAX, -INT8_C(  92),  INT8_C(   8), -INT8_C(  92), -INT8_C(  33),
        -INT8_C(  79), -INT8_C( 105),  INT8_C( 121), -INT8_C(   7), -INT8_C(  98), -INT8_C(  74),  INT8_C(  21),  INT8_C(  21),
        -INT8_C(  29),  INT8_C(  66), -INT8_C( 110), -INT8_C(  61),  INT8_C(  67),  INT8_C(   9), -INT8_C( 107),  INT8_C(  83),
        -INT8_C( 111),  INT8_C( 121), -INT8_C(  10), -INT8_C(  60), -INT8_C(  70), -INT8_C( 110),  INT8_C(   5), -INT8_C(  94) },
      UINT32_C(2957812608) },
    { UINT32_C(4230044504),
      {  INT8_C( 100), -INT8_C(  59), -INT8_C(  37),  INT8_C(  21),  INT8_C(  92),  INT8_C(  85),  INT8_C(  14), -INT8_C(   6),
         INT8_C(  11),  INT8_C(  36),  INT8_C(  16), -INT8_C(  18),  INT8_C( 102), -INT8_C(  94), -INT8_C(  78), -INT8_C(  86),
        -INT8_C(  85),  INT8_C(  71), -INT8_C(   3),  INT8_C(  60), -INT8_C(  64), -INT8_C(  13),  INT8_C(   0),  INT8_C( 123),
        -INT8_C( 122),  INT8_C(   5),  INT8_C(  29), -INT8_C(  34),  INT8_C(  97),  INT8_C(  63), -INT8_C(  38), -INT8_C(  59) },
      {  INT8_C(   4), -INT8_C(  74), -INT8_C(  38),  INT8_C(  97),  INT8_C(  11), -INT8_C(  23),  INT8_C(  91),  INT8_C(  22),
         INT8_C(  13),  INT8_C( 107),  INT8_C(   5),  INT8_C( 115),  INT8_C(  13), -INT8_C(  73),  INT8_C(  29), -INT8_C(  72),
        -INT8_C(   2),  INT8_C(  27), -INT8_C(  12), -INT8_C(  65),  INT8_C(  14), -INT8_C(  12),  INT8_C(  58), -INT8_C( 108),
        -INT8_C(   6),  INT8_C(  87),  INT8_C( 114),  INT8_C(  91), -INT8_C( 106),  INT8_C(  77),  INT8_C(  32), -INT8_C( 101) },
      UINT32_C(2415923216) },
    { UINT32_C( 251460099),
      { -INT8_C(  29),  INT8_C(  87),  INT8_C(  36), -INT8_C(  16), -INT8_C(  61),  INT8_C(  41),  INT8_C( 100), -INT8_C(  48),
        -INT8_C(  32), -INT8_C( 127), -INT8_C( 119), -INT8_C(  33), -INT8_C( 100),  INT8_C( 125), -INT8_C(  98), -INT8_C(  85),
         INT8_C( 114), -INT8_C(  40),  INT8_C(  63),  INT8_C( 108),  INT8_C(  47), -INT8_C(  78), -INT8_C(  57), -INT8_C(  58),
        -INT8_C(   1), -INT8_C(  25),  INT8_C(  97),  INT8_C(   2), -INT8_C(  31),  INT8_C(  93),  INT8_C(  16), -INT8_C(  59) },
      { -INT8_C(  76),  INT8_C(  52), -INT8_C(  75),  INT8_C( 119),  INT8_C(  94),  INT8_C(  25),  INT8_C(  72),  INT8_C(  62),
        -INT8_C( 101), -INT8_C(  47),  INT8_C(  29),  INT8_C(  55),  INT8_C(  78), -INT8_C(  69), -INT8_C(  30), -INT8_C(  64),
        -INT8_C( 109),  INT8_C(  34),  INT8_C(  44), -INT8_C(  61), -INT8_C(  44), -INT8_C(  13), -INT8_C( 119), -INT8_C(  45),
        -INT8_C(  38), -INT8_C(  22), -INT8_C(  43), -INT8_C(  68),  INT8_C(  71), -INT8_C(  27), -INT8_C( 127), -INT8_C(   5) },
      UINT32_C( 207364099) },
    { UINT32_C(2004039193),
      {  INT8_C(  80), -INT8_C(  69), -INT8_C(  74), -INT8_C(  21), -INT8_C( 116), -INT8_C(  45),  INT8_C(  34), -INT8_C(  38),
        -INT8_C( 113),  INT8_C(   5), -INT8_C( 101),  INT8_C(  34),  INT8_C(  39), -INT8_C(  57), -INT8_C(  27), -INT8_C(   5),
        -INT8_C(  69),  INT8_C( 110), -INT8_C(  50), -INT8_C( 107),  INT8_C(  88), -INT8_C(  93),  INT8_C(  81), -INT8_C(  97),
        -INT8_C( 120), -INT8_C(  46), -INT8_C( 101), -INT8_C(  95),  INT8_C(   9),  INT8_C(  14),  INT8_C(  25),  INT8_C(  89) },
      { -INT8_C(  55), -INT8_C(  49),  INT8_C(  68),  INT8_C(  85), -INT8_C(  94),  INT8_C( 102),  INT8_C(  47),  INT8_C(  49),
         INT8_C( 107), -INT8_C(  54),  INT8_C(  84), -INT8_C( 110), -INT8_C( 110),  INT8_C(  57), -INT8_C( 115),  INT8_C(  77),
        -INT8_C(  88),  INT8_C(  91), -INT8_C(  30),  INT8_C(   0), -INT8_C(   2),  INT8_C(  52), -INT8_C(  96), -INT8_C( 122),
         INT8_C(   6),  INT8_C(  59),  INT8_C(  40),  INT8_C(  15),  INT8_C(  73),  INT8_C(  65),  INT8_C( 104),  INT8_C(  18) },
      UINT32_C(   5444097) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epi8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 r = easysimd_mm256_mask_cmpgt_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int16_t a[16];
    int16_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(40467),
      {  INT16_C( 29473), -INT16_C(  8678),  INT16_C(  5163),  INT16_C(  6985), -INT16_C(  2244), -INT16_C(  8098),  INT16_C(  8648),  INT16_C( 13130),
         INT16_C( 10050), -INT16_C( 13386),  INT16_C( 28076),  INT16_C( 28075),  INT16_C( 25274),  INT16_C(  2935), -INT16_C( 29809), -INT16_C( 20055) },
      { -INT16_C( 15362),  INT16_C( 10639), -INT16_C( 10025),  INT16_C(  5188), -INT16_C( 23857), -INT16_C( 26636),  INT16_C( 16067),  INT16_C(  1482),
        -INT16_C( 32667),  INT16_C(  4561),  INT16_C( 31982), -INT16_C( 22402), -INT16_C(  2338),  INT16_C( 28339),  INT16_C( 23937),  INT16_C( 32543) },
      UINT16_C( 6161) },
    { UINT16_C(44576),
      { -INT16_C(  1880), -INT16_C(  4986),  INT16_C( 22028),  INT16_C(   142),  INT16_C( 20973), -INT16_C( 18370), -INT16_C( 23721),  INT16_C( 10296),
         INT16_C(  9908),  INT16_C( 12964), -INT16_C( 31794), -INT16_C( 32216), -INT16_C( 22031),  INT16_C(  4319), -INT16_C(   216), -INT16_C( 12098) },
      {  INT16_C( 17655),  INT16_C(   956),  INT16_C( 19098), -INT16_C( 30717),  INT16_C( 16796), -INT16_C(  3264),  INT16_C( 30948), -INT16_C( 26597),
        -INT16_C( 16481),  INT16_C( 28107), -INT16_C(  3262),  INT16_C( 13295), -INT16_C( 12643), -INT16_C( 15037),  INT16_C(   462), -INT16_C( 14954) },
      UINT16_C(40960) },
    { UINT16_C(21062),
      { -INT16_C(  7991), -INT16_C( 13155),  INT16_C( 14696), -INT16_C( 22514), -INT16_C(  3540),  INT16_C( 18209), -INT16_C( 16245),  INT16_C( 22022),
         INT16_C( 18733),  INT16_C(  7497), -INT16_C(  6532), -INT16_C( 16149), -INT16_C( 18004),  INT16_C( 17089),  INT16_C(  1919),  INT16_C( 18580) },
      {  INT16_C( 12776),  INT16_C( 20500),  INT16_C(  8810), -INT16_C( 26887),  INT16_C(  6677), -INT16_C( 24355), -INT16_C(  6950),  INT16_C(  2038),
         INT16_C( 16173), -INT16_C( 22236),  INT16_C(  4134), -INT16_C( 11671),  INT16_C( 11209),  INT16_C( 18452), -INT16_C( 22478),  INT16_C(  6800) },
      UINT16_C(16900) },
    { UINT16_C(42458),
      {  INT16_C( 17515),  INT16_C( 25799), -INT16_C(  8997), -INT16_C( 18306),  INT16_C( 22652),  INT16_C( 29340), -INT16_C( 13985), -INT16_C( 31566),
        -INT16_C( 10125), -INT16_C(  9068),  INT16_C( 23978), -INT16_C( 16889),  INT16_C( 15014),  INT16_C( 13926),  INT16_C( 16468), -INT16_C( 16421) },
      { -INT16_C( 23675),  INT16_C( 24611), -INT16_C( 24193), -INT16_C(  1000), -INT16_C( 18951),  INT16_C( 22894),  INT16_C(  8318), -INT16_C(  3619),
         INT16_C( 29176), -INT16_C( 23858), -INT16_C( 10802),  INT16_C( 29792), -INT16_C( 14577),  INT16_C( 25771), -INT16_C( 31225), -INT16_C( 29661) },
      UINT16_C(33810) },
    { UINT16_C(18217),
      { -INT16_C( 22036),  INT16_C(  1512), -INT16_C(  7515),  INT16_C(  5050),  INT16_C( 14395),  INT16_C(  6196),  INT16_C( 11306), -INT16_C(  1911),
         INT16_C( 22479),  INT16_C( 12237), -INT16_C(  8756),  INT16_C( 30710), -INT16_C(   447),  INT16_C( 25853),  INT16_C( 10122),  INT16_C( 30635) },
      { -INT16_C( 27440),  INT16_C( 30076),  INT16_C( 13942), -INT16_C( 20088), -INT16_C( 17298), -INT16_C( 26423),  INT16_C( 21225), -INT16_C( 18288),
         INT16_C( 24233),  INT16_C( 30183), -INT16_C(  8645),  INT16_C( 31980), -INT16_C(  5412),  INT16_C( 26336), -INT16_C( 29679), -INT16_C(  7715) },
      UINT16_C(16425) },
    { UINT16_C(22816),
      { -INT16_C( 27050), -INT16_C(  8561), -INT16_C(   441),  INT16_C(  4251), -INT16_C( 31594),  INT16_C( 10082),  INT16_C(  2876),  INT16_C(  9093),
        -INT16_C( 16255),  INT16_C( 27905), -INT16_C(  8900),  INT16_C(  7255),  INT16_C( 26692),  INT16_C(  8616), -INT16_C( 14263), -INT16_C( 24709) },
      {  INT16_C(  2654), -INT16_C( 23170),  INT16_C(  6408), -INT16_C( 24651),  INT16_C(  6045), -INT16_C(  9786),  INT16_C( 19235), -INT16_C( 23300),
        -INT16_C(   501),  INT16_C( 18193),  INT16_C( 27099),  INT16_C(  8035),  INT16_C(  3281),  INT16_C(  6977), -INT16_C( 17196),  INT16_C( 13242) },
      UINT16_C(20512) },
    { UINT16_C(14534),
      { -INT16_C( 12328), -INT16_C( 29103), -INT16_C(  4498),  INT16_C( 13477), -INT16_C( 14137), -INT16_C( 15233), -INT16_C( 30100),  INT16_C( 32450),
        -INT16_C( 25135),  INT16_C( 13543), -INT16_C( 18243), -INT16_C(   448),  INT16_C(  5587), -INT16_C( 28998), -INT16_C( 32696),  INT16_C(  8390) },
      {  INT16_C(  6223), -INT16_C( 16978),  INT16_C( 21510), -INT16_C( 12559),  INT16_C( 28700), -INT16_C( 30318),  INT16_C( 21754), -INT16_C( 13561),
        -INT16_C(  4367), -INT16_C( 20992),  INT16_C( 16550),  INT16_C( 31404),  INT16_C( 26197), -INT16_C( 25336), -INT16_C( 12569),  INT16_C( 14014) },
      UINT16_C(  128) },
    { UINT16_C(27878),
      { -INT16_C(  4620), -INT16_C(  6720), -INT16_C(  8773),  INT16_C( 19798),  INT16_C( 20582),  INT16_C( 28065), -INT16_C( 28132),  INT16_C(  7259),
         INT16_C(   321), -INT16_C(  4772), -INT16_C( 19845), -INT16_C( 31916),  INT16_C( 15183),  INT16_C(  3410),  INT16_C( 14449),  INT16_C( 25978) },
      {  INT16_C( 14885), -INT16_C(  8117), -INT16_C( 24297),  INT16_C( 32045), -INT16_C( 12559),  INT16_C(  3562),  INT16_C( 17761), -INT16_C( 24023),
        -INT16_C( 31161), -INT16_C( 15729), -INT16_C(  7368), -INT16_C( 30906), -INT16_C( 26594), -INT16_C( 28523),  INT16_C(  4048), -INT16_C(  2315) },
      UINT16_C(24742) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epi16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 r = easysimd_mm256_mask_cmpgt_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int32_t a[8];
    int32_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(241),
      { -INT32_C(    21007127), -INT32_C(   570233976),  INT32_C(   168606078), -INT32_C(   606717871), -INT32_C(   568823502),  INT32_C(   581527994),  INT32_C(   679422078),  INT32_C(  2048519825) },
      { -INT32_C(   361179038),  INT32_C(  1103723459), -INT32_C(  2058627788),  INT32_C(  1063264525),  INT32_C(  1293777043),  INT32_C(  1617938402), -INT32_C(  2138509585),  INT32_C(  1056613084) },
      UINT8_C(193) },
    { UINT8_C(122),
      { -INT32_C(   314693262), -INT32_C(   954105870), -INT32_C(   321607989),  INT32_C(  2139034375), -INT32_C(   144585424),  INT32_C(   736543036), -INT32_C(   335059382), -INT32_C(   748272032) },
      {  INT32_C(  1640014959), -INT32_C(   282533340), -INT32_C(  1881408376),  INT32_C(  1057970959),  INT32_C(  1681354792),  INT32_C(  2089753906), -INT32_C(   462907516),  INT32_C(  1287114461) },
      UINT8_C( 72) },
    { UINT8_C(115),
      {  INT32_C(  1519889784), -INT32_C(   773683499), -INT32_C(  1126141343), -INT32_C(   236707967),  INT32_C(  1948469590),  INT32_C(  1878564824), -INT32_C(   699606009),  INT32_C(   206149780) },
      {  INT32_C(   442949701), -INT32_C(   940881818),  INT32_C(   998558650),  INT32_C(  1093495019), -INT32_C(  1967828815), -INT32_C(   134631952),  INT32_C(   516769418),  INT32_C(   606738142) },
      UINT8_C( 51) },
    { UINT8_C(246),
      { -INT32_C(   665043312), -INT32_C(   174906326),  INT32_C(   283168424),  INT32_C(  1287791355), -INT32_C(  2059645737), -INT32_C(  1961938107),  INT32_C(   392834305), -INT32_C(   418476457) },
      { -INT32_C(   155227444),  INT32_C(   921457294),  INT32_C(   474467361), -INT32_C(   966260242), -INT32_C(  1706318763), -INT32_C(   651797800), -INT32_C(   554659705), -INT32_C(   356123106) },
      UINT8_C( 64) },
    { UINT8_C(105),
      { -INT32_C(   671620732), -INT32_C(  1711788339),  INT32_C(  2122847349),  INT32_C(   567496317), -INT32_C(   201757287),  INT32_C(   595251859), -INT32_C(  1052682045), -INT32_C(  1591071715) },
      { -INT32_C(   629661171), -INT32_C(   982290352),  INT32_C(    38009733), -INT32_C(   501017015),  INT32_C(   399842691), -INT32_C(  1304801297), -INT32_C(   998998873), -INT32_C(  1251565912) },
      UINT8_C( 40) },
    { UINT8_C(192),
      {  INT32_C(  1326485470), -INT32_C(    19606270),  INT32_C(   776459800),  INT32_C(   380775161),  INT32_C(  1325844991),  INT32_C(  2146875395), -INT32_C(   886588628), -INT32_C(     7611359) },
      {  INT32_C(  1850645611), -INT32_C(  1972624783), -INT32_C(   222776328), -INT32_C(   586650915),  INT32_C(   925634099), -INT32_C(   206167353), -INT32_C(    21045539),  INT32_C(   654199482) },
      UINT8_C(  0) },
    { UINT8_C(230),
      {  INT32_C(  1868010572), -INT32_C(  1285037824),  INT32_C(    76634522),  INT32_C(  1882746466), -INT32_C(  1137217638),  INT32_C(    60435237),  INT32_C(   868129001), -INT32_C(   484842346) },
      {  INT32_C(  2018668920), -INT32_C(   332678830),  INT32_C(  1978776595), -INT32_C(   991614678), -INT32_C(  1115611752),  INT32_C(   834739016),  INT32_C(  1248165811), -INT32_C(   617775517) },
      UINT8_C(128) },
    { UINT8_C(239),
      {  INT32_C(   943870847),  INT32_C(   994782846),  INT32_C(  1214627871), -INT32_C(  1008719194), -INT32_C(   972317013), -INT32_C(   562480033),  INT32_C(   541180833),  INT32_C(  1863261424) },
      { -INT32_C(   307736209), -INT32_C(  1624706176),  INT32_C(  1525190324),  INT32_C(  1662896312), -INT32_C(   987158426),  INT32_C(   111387236),  INT32_C(  1445389414),  INT32_C(  1875260672) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epi32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 r = easysimd_mm256_mask_cmpgt_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t a[4];
    int64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(156),
      { -INT64_C( 7307043608125727390),  INT64_C( 6165709699429661090),  INT64_C(  680447407674550196),  INT64_C( 6250401396019196100) },
      { -INT64_C( 1610192692261020289),  INT64_C(  832428533376571115), -INT64_C( 8673542815114767288),  INT64_C( 6277566912338130387) },
      UINT8_C(  4) },
    { UINT8_C(130),
      {  INT64_C( 7085766454649906424),  INT64_C( 7938936962427751810),  INT64_C( 6623872995820684470), -INT64_C( 3942467365191306399) },
      {  INT64_C( 5214474331764233199), -INT64_C( 7881764595127544043), -INT64_C( 1523175957116235748), -INT64_C( 8509837806545072309) },
      UINT8_C(  2) },
    { UINT8_C( 14),
      { -INT64_C(  524919076140317671),  INT64_C( 6249984210804991816),  INT64_C( 3095601435260173842),  INT64_C( 1262219520815403387) },
      { -INT64_C( 4779609530076800815), -INT64_C(  491425730561999022),  INT64_C( 5880743260252736856),  INT64_C( 9129442603749040297) },
      UINT8_C(  2) },
    { UINT8_C(163),
      {  INT64_C( 6566764876026884453),  INT64_C( 3793221826122021170),  INT64_C( 1449979530885928926), -INT64_C( 2948942470752124892) },
      {  INT64_C( 3650387401612143473),  INT64_C(  224655677153340098), -INT64_C( 6268278930835691206),  INT64_C( 8404609724424242009) },
      UINT8_C(  3) },
    { UINT8_C( 87),
      {  INT64_C( 6112943157909373258),  INT64_C(  389136802406645256),  INT64_C( 1939138660956591351), -INT64_C( 6642648537389746867) },
      { -INT64_C( 5110869031603787949),  INT64_C(   12468501166263370),  INT64_C( 4112203088655333280),  INT64_C( 1453681642684165679) },
      UINT8_C(  3) },
    { UINT8_C(170),
      {  INT64_C( 8050208400478032354), -INT64_C( 4474474619730272518),  INT64_C( 2140308164328676874),  INT64_C( 5388260533153447064) },
      {  INT64_C( 3537137937390510531),  INT64_C( 1376797505340542799),  INT64_C( 5064661367870965820), -INT64_C( 2577985219854194810) },
      UINT8_C(  8) },
    { UINT8_C( 73),
      {  INT64_C( 3004377703068500977), -INT64_C( 1314130638777742808),  INT64_C( 6894415335384674268), -INT64_C( 4128300207790580668) },
      {  INT64_C( 1124732115740014040), -INT64_C( 1083299247059185948), -INT64_C( 5125161846280056248), -INT64_C( 8863969533411174560) },
      UINT8_C(  9) },
    { UINT8_C( 15),
      {  INT64_C( 6909090902945756708),  INT64_C( 8448018302431914907), -INT64_C(  819757810909069958), -INT64_C(  743053460049439546) },
      {  INT64_C( 5047362740716545562),  INT64_C( 6564128267882364608),  INT64_C( 7693900389184831797), -INT64_C( 6425587133439121651) },
      UINT8_C( 11) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epi64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 r = easysimd_mm256_mask_cmpgt_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[32];
    uint8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { { UINT8_C( 70), UINT8_C( 77), UINT8_C( 94), UINT8_C(106), UINT8_C( 23), UINT8_C( 36), UINT8_C(  7), UINT8_C(198),
        UINT8_C(252), UINT8_C(252), UINT8_C(206), UINT8_C( 20), UINT8_C(200), UINT8_C( 71), UINT8_C(249), UINT8_C(140),
        UINT8_C(173), UINT8_C(180), UINT8_C(159), UINT8_C(244), UINT8_C(175), UINT8_C( 50), UINT8_C(123), UINT8_C( 84),
        UINT8_C(152), UINT8_C( 75), UINT8_C(216), UINT8_C( 74), UINT8_C( 24), UINT8_C(254), UINT8_C( 84), UINT8_C( 95) },
      { UINT8_C( 75), UINT8_C(178), UINT8_C(201), UINT8_C( 98), UINT8_C(214), UINT8_C(208), UINT8_C( 40), UINT8_C(210),
        UINT8_C(205), UINT8_C(246), UINT8_C(230), UINT8_C(149), UINT8_C( 62), UINT8_C(223), UINT8_C( 33), UINT8_C(235),
        UINT8_C(147), UINT8_C(192), UINT8_C(224), UINT8_C( 66), UINT8_C(242), UINT8_C( 91), UINT8_C(150), UINT8_C(139),
        UINT8_C(166), UINT8_C(110), UINT8_C(213), UINT8_C(191), UINT8_C(108), UINT8_C( 42), UINT8_C( 30), UINT8_C(183) },
      UINT32_C(1678332680) },
    { { UINT8_C(220), UINT8_C(231), UINT8_C( 25), UINT8_C(179), UINT8_C(183), UINT8_C( 65), UINT8_C(133), UINT8_C(132),
        UINT8_C( 56), UINT8_C(108), UINT8_C( 25), UINT8_C(118), UINT8_C( 75), UINT8_C( 58), UINT8_C( 97), UINT8_C(223),
        UINT8_C(250), UINT8_C( 65), UINT8_C( 33), UINT8_C(237), UINT8_C(157), UINT8_C(184), UINT8_C(120), UINT8_C( 67),
        UINT8_C( 38), UINT8_C( 77), UINT8_C(  2), UINT8_C(147), UINT8_C(119), UINT8_C( 32), UINT8_C( 74), UINT8_C( 84) },
      { UINT8_C(  7), UINT8_C(100), UINT8_C(  7), UINT8_C(191), UINT8_C(165), UINT8_C(140), UINT8_C( 67), UINT8_C(221),
        UINT8_C(248), UINT8_C( 93), UINT8_C( 83), UINT8_C( 68), UINT8_C(151), UINT8_C(181), UINT8_C( 35), UINT8_C(146),
        UINT8_C(246), UINT8_C( 68), UINT8_C(127), UINT8_C(147), UINT8_C(252), UINT8_C(247), UINT8_C(215), UINT8_C( 35),
        UINT8_C( 68), UINT8_C(217), UINT8_C(182), UINT8_C(188), UINT8_C(250), UINT8_C(  0), UINT8_C( 16), UINT8_C(  1) },
      UINT32_C(3767126615) },
    { { UINT8_C(100), UINT8_C( 23), UINT8_C(192), UINT8_C( 10), UINT8_C(163), UINT8_C(  4), UINT8_C(231), UINT8_C(156),
        UINT8_C( 97), UINT8_C( 59), UINT8_C(224), UINT8_C(248), UINT8_C(240), UINT8_C(  3), UINT8_C(138), UINT8_C(230),
        UINT8_C( 71), UINT8_C(  9), UINT8_C(122), UINT8_C( 68), UINT8_C(  0), UINT8_C( 81), UINT8_C(103), UINT8_C( 69),
        UINT8_C( 42), UINT8_C( 29), UINT8_C(  1), UINT8_C( 36), UINT8_C( 29), UINT8_C( 17), UINT8_C( 38), UINT8_C(130) },
      { UINT8_C( 40), UINT8_C(230), UINT8_C(140), UINT8_C(203), UINT8_C(234), UINT8_C(115), UINT8_C(103), UINT8_C( 75),
        UINT8_C(174), UINT8_C( 71), UINT8_C( 68), UINT8_C(158), UINT8_C( 74), UINT8_C(206), UINT8_C(133), UINT8_C(146),
        UINT8_C(216),    UINT8_MAX, UINT8_C(214), UINT8_C(216), UINT8_C( 80), UINT8_C( 61), UINT8_C( 29), UINT8_C(122),
        UINT8_C( 90), UINT8_C( 30), UINT8_C(159), UINT8_C(119), UINT8_C( 47), UINT8_C(197), UINT8_C(249), UINT8_C( 87) },
      UINT32_C(2153831621) },
    { { UINT8_C(171), UINT8_C(133), UINT8_C( 35), UINT8_C(150), UINT8_C(249), UINT8_C(138), UINT8_C(225), UINT8_C(167),
        UINT8_C(210), UINT8_C( 37), UINT8_C( 70), UINT8_C( 28), UINT8_C(244), UINT8_C(203), UINT8_C(174), UINT8_C(204),
        UINT8_C(202), UINT8_C(132), UINT8_C(164), UINT8_C( 26), UINT8_C(193), UINT8_C(194), UINT8_C(148), UINT8_C( 27),
        UINT8_C(224), UINT8_C( 51), UINT8_C(147), UINT8_C( 16), UINT8_C(248), UINT8_C(140), UINT8_C(103), UINT8_C(164) },
      { UINT8_C( 18), UINT8_C(138), UINT8_C( 58), UINT8_C( 11), UINT8_C( 21), UINT8_C( 27), UINT8_C(178), UINT8_C(231),
        UINT8_C( 65), UINT8_C(248), UINT8_C(  3), UINT8_C( 53), UINT8_C(195), UINT8_C(178), UINT8_C(  1), UINT8_C(141),
        UINT8_C( 54), UINT8_C(165), UINT8_C(167), UINT8_C(248), UINT8_C(103), UINT8_C( 60), UINT8_C( 19), UINT8_C( 72),
        UINT8_C(111), UINT8_C(166), UINT8_C( 88), UINT8_C(104), UINT8_C( 51), UINT8_C(191), UINT8_C( 12), UINT8_C( 69) },
      UINT32_C(3581015417) },
    { { UINT8_C( 74), UINT8_C( 70), UINT8_C( 80), UINT8_C( 95), UINT8_C( 97), UINT8_C(  2), UINT8_C( 70), UINT8_C(162),
        UINT8_C(251), UINT8_C( 73), UINT8_C(215), UINT8_C(190), UINT8_C(251), UINT8_C(216), UINT8_C( 76), UINT8_C( 50),
        UINT8_C(126), UINT8_C(243), UINT8_C( 42), UINT8_C(229), UINT8_C( 47), UINT8_C( 61), UINT8_C( 45), UINT8_C(159),
        UINT8_C(228), UINT8_C(133), UINT8_C(  7), UINT8_C( 23), UINT8_C( 69), UINT8_C( 19), UINT8_C( 92), UINT8_C(143) },
      { UINT8_C( 89), UINT8_C(172), UINT8_C(238), UINT8_C(186), UINT8_C(174), UINT8_C( 52), UINT8_C( 93), UINT8_C(169),
        UINT8_C(125), UINT8_C( 52), UINT8_C(104), UINT8_C(121), UINT8_C( 13), UINT8_C(180), UINT8_C(171), UINT8_C(139),
        UINT8_C(167), UINT8_C(213), UINT8_C(112), UINT8_C(215), UINT8_C( 18), UINT8_C(158), UINT8_C(118), UINT8_C(246),
        UINT8_C( 35), UINT8_C(125), UINT8_C( 13), UINT8_C(104), UINT8_C(144), UINT8_C(105), UINT8_C(247), UINT8_C(233) },
      UINT32_C(  52051712) },
    { { UINT8_C( 21), UINT8_C(229), UINT8_C(163), UINT8_C(196), UINT8_C( 25), UINT8_C(  0), UINT8_C(109), UINT8_C(151),
        UINT8_C( 53), UINT8_C(213), UINT8_C( 16), UINT8_C( 66), UINT8_C(137), UINT8_C(187), UINT8_C(205), UINT8_C( 49),
        UINT8_C(144), UINT8_C( 61), UINT8_C(  8), UINT8_C(162), UINT8_C(219), UINT8_C(126), UINT8_C(153),    UINT8_MAX,
        UINT8_C(251), UINT8_C(166), UINT8_C(103), UINT8_C(139), UINT8_C( 16), UINT8_C( 95), UINT8_C(116), UINT8_C( 37) },
      { UINT8_C( 68), UINT8_C( 23), UINT8_C(233), UINT8_C( 94), UINT8_C( 24), UINT8_C( 87), UINT8_C(245), UINT8_C( 77),
        UINT8_C( 44), UINT8_C(  5), UINT8_C(143), UINT8_C(182), UINT8_C(192), UINT8_C( 92), UINT8_C(231), UINT8_C( 80),
        UINT8_C(153), UINT8_C(239), UINT8_C(242), UINT8_C(117), UINT8_C(109), UINT8_C(139), UINT8_C(116), UINT8_C(104),
        UINT8_C( 50), UINT8_C(219), UINT8_C(243), UINT8_C( 66), UINT8_C( 58), UINT8_C(103), UINT8_C(103), UINT8_C(127) },
      UINT32_C(1238901658) },
    { { UINT8_C(126), UINT8_C( 81), UINT8_C(221), UINT8_C(150), UINT8_C(168), UINT8_C(210), UINT8_C(227), UINT8_C(212),
        UINT8_C(215), UINT8_C(114), UINT8_C(138), UINT8_C(151), UINT8_C(206), UINT8_C(113), UINT8_C(231), UINT8_C(104),
        UINT8_C( 96), UINT8_C(217), UINT8_C(221), UINT8_C(205), UINT8_C(101), UINT8_C( 81), UINT8_C( 53), UINT8_C(151),
        UINT8_C( 44), UINT8_C( 40), UINT8_C(217), UINT8_C(103), UINT8_C(143), UINT8_C( 64), UINT8_C(230), UINT8_C( 14) },
      { UINT8_C(145), UINT8_C(195), UINT8_C(164), UINT8_C( 57), UINT8_C(149), UINT8_C(136), UINT8_C( 14), UINT8_C(108),
        UINT8_C(250), UINT8_C(152), UINT8_C(  3), UINT8_C(201), UINT8_C( 10), UINT8_C(234), UINT8_C( 49), UINT8_C(106),
        UINT8_C(195), UINT8_C( 14), UINT8_C( 56), UINT8_C( 40), UINT8_C( 95), UINT8_C(109), UINT8_C(191), UINT8_C(139),
        UINT8_C(150), UINT8_C(152), UINT8_C(242), UINT8_C( 37), UINT8_C(217), UINT8_C(216), UINT8_C( 51), UINT8_C(106) },
      UINT32_C(1218335996) },
    { { UINT8_C(155), UINT8_C(216), UINT8_C(164), UINT8_C( 48), UINT8_C( 96), UINT8_C(178), UINT8_C(156), UINT8_C( 90),
        UINT8_C( 74), UINT8_C(159), UINT8_C( 35), UINT8_C( 84), UINT8_C(137), UINT8_C( 84), UINT8_C(191), UINT8_C( 77),
        UINT8_C( 98), UINT8_C(247), UINT8_C(117), UINT8_C(193), UINT8_C(100), UINT8_C( 53), UINT8_C( 77), UINT8_C(250),
        UINT8_C(205), UINT8_C( 63), UINT8_C( 32), UINT8_C(166), UINT8_C( 24), UINT8_C( 83), UINT8_C( 17), UINT8_C(179) },
      { UINT8_C( 43), UINT8_C(181), UINT8_C(228), UINT8_C(139), UINT8_C(103), UINT8_C(128), UINT8_C(230), UINT8_C(177),
        UINT8_C( 32), UINT8_C(  9), UINT8_C(  6), UINT8_C(169), UINT8_C( 94), UINT8_C(197), UINT8_C(246), UINT8_C(192),
        UINT8_C(188), UINT8_C(108), UINT8_C(130), UINT8_C( 32), UINT8_C(161), UINT8_C(207), UINT8_C( 27), UINT8_C(110),
        UINT8_C( 14), UINT8_C( 59), UINT8_C( 21), UINT8_C( 38), UINT8_C(142), UINT8_C( 38), UINT8_C(218), UINT8_C(186) },
      UINT32_C( 801773347) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epu8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmpgt_epu8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_test_x86_random_u8x32();
    easysimd__mmask32 r = easysimd_mm256_cmpgt_epu8_mask(a, b);

    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[16];
    uint16_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { { UINT16_C(21031), UINT16_C( 3383), UINT16_C(14058), UINT16_C(34150), UINT16_C(48119), UINT16_C(50470), UINT16_C(42790), UINT16_C(11089),
        UINT16_C(25029), UINT16_C( 5670), UINT16_C(60933), UINT16_C(28196), UINT16_C(51026), UINT16_C(19719), UINT16_C(60792), UINT16_C(40996) },
      { UINT16_C(23359), UINT16_C(10925), UINT16_C( 5009), UINT16_C(34991), UINT16_C(54991), UINT16_C(62798), UINT16_C(40829), UINT16_C(16929),
        UINT16_C(18176), UINT16_C( 1368), UINT16_C(31797), UINT16_C(34676), UINT16_C(31556), UINT16_C(48340), UINT16_C(63592), UINT16_C(43100) },
      UINT16_C( 5956) },
    { { UINT16_C( 2387), UINT16_C(58834), UINT16_C(33053), UINT16_C(60525), UINT16_C(47959), UINT16_C(54753), UINT16_C(  603), UINT16_C(23319),
        UINT16_C(28745), UINT16_C(32353), UINT16_C(54764), UINT16_C(12294), UINT16_C(55888), UINT16_C(47597), UINT16_C(18899), UINT16_C( 9825) },
      { UINT16_C(13139), UINT16_C(28683), UINT16_C(31156), UINT16_C( 3164), UINT16_C(15668), UINT16_C(36833), UINT16_C(63552), UINT16_C(35307),
        UINT16_C(19560), UINT16_C(21768), UINT16_C( 3617), UINT16_C(29061), UINT16_C(29416), UINT16_C(47914), UINT16_C(35772), UINT16_C( 4066) },
      UINT16_C(38718) },
    { { UINT16_C(60862), UINT16_C(29567), UINT16_C(56166), UINT16_C(39807), UINT16_C(24600), UINT16_C(22570), UINT16_C( 5464), UINT16_C(49634),
        UINT16_C(60001), UINT16_C(33302), UINT16_C(39928), UINT16_C(57588), UINT16_C( 7694), UINT16_C(51868), UINT16_C(32426), UINT16_C(26841) },
      { UINT16_C(22635), UINT16_C(53979), UINT16_C(23091), UINT16_C(19309), UINT16_C(38842), UINT16_C( 5028), UINT16_C(34477), UINT16_C( 3796),
        UINT16_C(60016), UINT16_C(26769), UINT16_C(34181), UINT16_C(37704), UINT16_C(58531), UINT16_C(19805), UINT16_C(13922), UINT16_C(52918) },
      UINT16_C(28333) },
    { { UINT16_C(37262), UINT16_C(49568), UINT16_C( 3564), UINT16_C(42509), UINT16_C(45476), UINT16_C(20921), UINT16_C(36151), UINT16_C(42848),
        UINT16_C(61815), UINT16_C(64783), UINT16_C(22390), UINT16_C( 6544), UINT16_C(60988), UINT16_C(40551), UINT16_C( 7460), UINT16_C(45932) },
      { UINT16_C( 3246), UINT16_C(39540), UINT16_C(33049), UINT16_C(48705), UINT16_C(64050), UINT16_C(26895), UINT16_C(28552), UINT16_C(65296),
        UINT16_C( 8032), UINT16_C(55036), UINT16_C(36215), UINT16_C(46064), UINT16_C(22395), UINT16_C(40785), UINT16_C(48756), UINT16_C( 8786) },
      UINT16_C(37699) },
    { { UINT16_C(51146), UINT16_C(58557), UINT16_C(65096), UINT16_C(31650), UINT16_C(45560), UINT16_C(32996), UINT16_C(62753), UINT16_C(33152),
        UINT16_C(31764), UINT16_C(35672), UINT16_C(18441), UINT16_C(33854), UINT16_C(37023), UINT16_C( 4900), UINT16_C(30286), UINT16_C( 6197) },
      { UINT16_C(62013), UINT16_C(34556), UINT16_C(40688), UINT16_C(59649), UINT16_C(58704), UINT16_C(29033), UINT16_C(59866), UINT16_C(61426),
        UINT16_C(19046), UINT16_C(28538), UINT16_C(47506), UINT16_C(12788), UINT16_C( 6217), UINT16_C(38724), UINT16_C(31374), UINT16_C(52399) },
      UINT16_C( 7014) },
    { { UINT16_C(44140), UINT16_C(23890), UINT16_C(21322), UINT16_C(39494), UINT16_C(44856), UINT16_C( 4875), UINT16_C(65177), UINT16_C(65282),
        UINT16_C(31816), UINT16_C(56174), UINT16_C(25141), UINT16_C(32268), UINT16_C(20858), UINT16_C( 2325), UINT16_C(50635), UINT16_C(14293) },
      { UINT16_C(10097), UINT16_C(48020), UINT16_C(55930), UINT16_C(45654), UINT16_C(24970), UINT16_C( 9157), UINT16_C(51039), UINT16_C(43042),
        UINT16_C(36932), UINT16_C(31107), UINT16_C(36851), UINT16_C(28152), UINT16_C( 3552), UINT16_C(43894), UINT16_C(19410), UINT16_C(17379) },
      UINT16_C(23249) },
    { { UINT16_C(30578), UINT16_C(60671), UINT16_C(21842), UINT16_C(56479), UINT16_C(25782), UINT16_C( 5887), UINT16_C( 8492), UINT16_C(28862),
        UINT16_C(16817), UINT16_C(42217), UINT16_C(57808), UINT16_C(45330), UINT16_C(35055), UINT16_C(49500), UINT16_C(16340), UINT16_C(17925) },
      { UINT16_C( 1207), UINT16_C( 2355), UINT16_C(53849), UINT16_C( 4069), UINT16_C(58422), UINT16_C(25125), UINT16_C(58117), UINT16_C(46802),
        UINT16_C(48164), UINT16_C(62811), UINT16_C(28061), UINT16_C(36006), UINT16_C(  757), UINT16_C(51534), UINT16_C(21314), UINT16_C(63760) },
      UINT16_C( 7179) },
    { { UINT16_C(17239), UINT16_C(45058), UINT16_C(59157), UINT16_C(19391), UINT16_C(58827), UINT16_C(53422), UINT16_C(32968), UINT16_C(60806),
        UINT16_C(57660), UINT16_C(56034), UINT16_C(34894), UINT16_C(17510), UINT16_C(46218), UINT16_C(52237), UINT16_C( 7431), UINT16_C(24261) },
      { UINT16_C(51040), UINT16_C(29966), UINT16_C(52910), UINT16_C(31169), UINT16_C(28595), UINT16_C(31561), UINT16_C(53487), UINT16_C(11368),
        UINT16_C(19121), UINT16_C(    6), UINT16_C(27858), UINT16_C(23876), UINT16_C(20769), UINT16_C(10281), UINT16_C(61295), UINT16_C(53127) },
      UINT16_C(14262) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmpgt_epu16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__mmask16 r = easysimd_mm256_cmpgt_epu16_mask(a, b);

    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t a[8];
    uint32_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT32_C(1699059126), UINT32_C( 383649379), UINT32_C(1687300213), UINT32_C(2844850936), UINT32_C( 396990021), UINT32_C( 611642627), UINT32_C(2924256831), UINT32_C(1132319629) },
      { UINT32_C(3433611881), UINT32_C(1038321608), UINT32_C(2812442031), UINT32_C(3025154671), UINT32_C(3435985609), UINT32_C( 653279463), UINT32_C(1809071326), UINT32_C(2041532944) },
      UINT8_C( 64) },
    { { UINT32_C(3712309012), UINT32_C(2367301854), UINT32_C( 221559965), UINT32_C(3099690479), UINT32_C(1719963007), UINT32_C(2894951630), UINT32_C(3222823344), UINT32_C(3359229875) },
      { UINT32_C(4255481630), UINT32_C(1166720935), UINT32_C(1800585084), UINT32_C(3273855812), UINT32_C(1865000865), UINT32_C(3407590939), UINT32_C(3414962968), UINT32_C( 429114874) },
      UINT8_C(130) },
    { { UINT32_C(3960879172), UINT32_C(1949409528), UINT32_C(2749334367), UINT32_C( 929432214), UINT32_C(3299250345), UINT32_C(1586479686), UINT32_C(4029291509), UINT32_C( 621395425) },
      { UINT32_C(3994099702), UINT32_C( 526533311), UINT32_C(1556234693), UINT32_C(3969067331), UINT32_C(4289739449), UINT32_C(4049485820), UINT32_C(1021413467), UINT32_C( 996272709) },
      UINT8_C( 70) },
    { { UINT32_C(3374936841), UINT32_C(2078837685), UINT32_C( 265792204), UINT32_C(2365287123), UINT32_C(2693573540), UINT32_C(1184033515), UINT32_C(3078779762), UINT32_C(1744036702) },
      { UINT32_C( 204479574), UINT32_C(1955010728), UINT32_C(2525257411), UINT32_C(1831043016), UINT32_C( 369995563), UINT32_C( 190619545), UINT32_C(1908596243), UINT32_C( 416855489) },
      UINT8_C(251) },
    { { UINT32_C(2049182162), UINT32_C(3840846625), UINT32_C(3514528265), UINT32_C( 490643186), UINT32_C(3862121549), UINT32_C(4277243883), UINT32_C( 779072365), UINT32_C( 994461801) },
      { UINT32_C(1924491857), UINT32_C( 509059861), UINT32_C( 150000150), UINT32_C(3173330544), UINT32_C(1721981050), UINT32_C(1415877863), UINT32_C(2961364039), UINT32_C(1844168988) },
      UINT8_C( 55) },
    { { UINT32_C(1239392307), UINT32_C(1516713540), UINT32_C(2019710728), UINT32_C(   3508102), UINT32_C(3328628959), UINT32_C(3021654892), UINT32_C(3143933087), UINT32_C(2569556069) },
      { UINT32_C( 887228400), UINT32_C(1183729982), UINT32_C( 666890401), UINT32_C(1445459063), UINT32_C( 958172877), UINT32_C(4176295513), UINT32_C( 951276243), UINT32_C(2463226786) },
      UINT8_C(215) },
    { { UINT32_C( 549958626), UINT32_C(2657572349), UINT32_C(3183814214), UINT32_C(3876908058), UINT32_C(3542167674), UINT32_C( 986386023), UINT32_C(  41057888), UINT32_C(1016415321) },
      { UINT32_C(4099693815), UINT32_C(4153590705), UINT32_C(  78993385), UINT32_C(3203123524), UINT32_C(1620184313), UINT32_C(2073713947), UINT32_C( 897453532), UINT32_C(1215370065) },
      UINT8_C( 28) },
    { { UINT32_C( 540921455), UINT32_C(2065223569), UINT32_C(1803537703), UINT32_C(2401856150), UINT32_C(2465250167), UINT32_C(4111305241), UINT32_C(3895102359), UINT32_C( 221355166) },
      { UINT32_C(4214124138), UINT32_C(1685472829), UINT32_C(2848978195), UINT32_C(3610835296), UINT32_C(3446220980), UINT32_C(1254258355), UINT32_C(2687757570), UINT32_C(4088292489) },
      UINT8_C( 98) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmpgt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__mmask16 r = easysimd_mm256_cmpgt_epu32_mask(a, b);

    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpgt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint64_t a[4];
    uint64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT64_C( 3851814778279222482), UINT64_C(17406718484852654938), UINT64_C( 8303654139368241727), UINT64_C(11955581276935909230) },
      { UINT64_C(11021479533268555933), UINT64_C( 2925902600149848684), UINT64_C( 5170790813509769438), UINT64_C( 3836830039119156934) },
      UINT8_C( 14) },
    { { UINT64_C( 9932382340031771649), UINT64_C(16991179853123430843), UINT64_C(  236154473183700517), UINT64_C(12583298896754410707) },
      { UINT64_C( 4140711853027005647), UINT64_C(11161233200210880748), UINT64_C(17352613583843237776), UINT64_C( 1783138789415939427) },
      UINT8_C( 11) },
    { { UINT64_C( 6432963726533765776), UINT64_C( 6787548948859069475), UINT64_C(  893116430838188630), UINT64_C( 5461417997822115689) },
      { UINT64_C( 7803505919296851588), UINT64_C( 3165809353953681091), UINT64_C( 9346380006137207552), UINT64_C( 3515338422144455106) },
      UINT8_C( 10) },
    { { UINT64_C( 9475975762719941068), UINT64_C(13478151890266260747), UINT64_C( 8122271263518731039), UINT64_C(13098639155623029996) },
      { UINT64_C( 7069045404692310487), UINT64_C( 5856736918922014416), UINT64_C(12833078360167460090), UINT64_C(10721682841706038828) },
      UINT8_C( 11) },
    { { UINT64_C(12876995168119330555), UINT64_C( 4681637251640063861), UINT64_C(17125220986357579824), UINT64_C(   25277797085999879) },
      { UINT64_C( 4681499282779571556), UINT64_C(16899159167890568702), UINT64_C( 8506608576287042911), UINT64_C( 4864894606383911038) },
      UINT8_C(  5) },
    { { UINT64_C(15718831385156900120), UINT64_C( 5430114007318924470), UINT64_C( 5113125750180625425), UINT64_C(15900325461046669448) },
      { UINT64_C( 7699155024714150434), UINT64_C(12332885043823966192), UINT64_C(10556191279860650610), UINT64_C(13326330062869907792) },
      UINT8_C(  9) },
    { { UINT64_C( 9651934819520065748), UINT64_C( 8416251346379984006), UINT64_C( 7545267807423690522), UINT64_C( 2721419165901303414) },
      { UINT64_C(10463725656816319260), UINT64_C( 4170754502882924395), UINT64_C( 3672425721707204470), UINT64_C(18173997982678938561) },
      UINT8_C(  6) },
    { { UINT64_C(17157275884511421566), UINT64_C(14038350387667062138), UINT64_C(  340664172411957196), UINT64_C( 8761838818083403963) },
      { UINT64_C(11791017739848664369), UINT64_C(  646283297793375977), UINT64_C( 5513896407101163839), UINT64_C( 1853372981655935812) },
      UINT8_C( 11) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpgt_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmpgt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__mmask16 r = easysimd_mm256_cmpgt_epu64_mask(a, b);

    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    uint8_t a[32];
    uint8_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C(3627281496),
      { UINT8_C(113), UINT8_C(221), UINT8_C(147), UINT8_C( 83), UINT8_C( 30), UINT8_C(239), UINT8_C(208), UINT8_C(100),
        UINT8_C( 17), UINT8_C(222), UINT8_C( 84), UINT8_C( 70), UINT8_C(215), UINT8_C(163), UINT8_C(123), UINT8_C( 94),
        UINT8_C( 49), UINT8_C( 38), UINT8_C(179), UINT8_C( 22), UINT8_C(109), UINT8_C(162), UINT8_C( 34), UINT8_C(164),
        UINT8_C( 52), UINT8_C( 85), UINT8_C(142), UINT8_C(140), UINT8_C( 65), UINT8_C(193), UINT8_C(100), UINT8_C(178) },
      { UINT8_C(159), UINT8_C(247), UINT8_C(  5), UINT8_C(189), UINT8_C(230), UINT8_C(213), UINT8_C( 34), UINT8_C(248),
        UINT8_C(179), UINT8_C(118), UINT8_C( 62), UINT8_C(139), UINT8_C( 26), UINT8_C(185), UINT8_C(233), UINT8_C( 75),
        UINT8_C(223), UINT8_C(156), UINT8_C( 97), UINT8_C( 77), UINT8_C( 62), UINT8_C(131), UINT8_C(241), UINT8_C(115),
        UINT8_C(216), UINT8_C(128),    UINT8_MAX, UINT8_C( 25), UINT8_C( 65), UINT8_C(100), UINT8_C(203), UINT8_C(224) },
      UINT32_C( 137397312) },
    { UINT32_C(1117704283),
      { UINT8_C(165), UINT8_C(192), UINT8_C( 58), UINT8_C( 89), UINT8_C( 54), UINT8_C(120), UINT8_C(228), UINT8_C( 80),
        UINT8_C( 49), UINT8_C(205), UINT8_C(155), UINT8_C( 16), UINT8_C(105), UINT8_C(253), UINT8_C( 93), UINT8_C(167),
        UINT8_C(128), UINT8_C( 79), UINT8_C( 26), UINT8_C( 89), UINT8_C(207), UINT8_C( 26), UINT8_C(114), UINT8_C( 16),
        UINT8_C(126), UINT8_C( 62), UINT8_C(241), UINT8_C(217), UINT8_C( 14), UINT8_C(143), UINT8_C( 27), UINT8_C(180) },
      { UINT8_C( 79), UINT8_C( 85), UINT8_C( 13), UINT8_C(133), UINT8_C(205), UINT8_C(241), UINT8_C(214), UINT8_C(254),
        UINT8_C(190), UINT8_C(113), UINT8_C( 15), UINT8_C( 39), UINT8_C(110), UINT8_C(108), UINT8_C(206), UINT8_C(239),
        UINT8_C(187), UINT8_C(233), UINT8_C( 72), UINT8_C(138), UINT8_C(  3), UINT8_C(186), UINT8_C(155), UINT8_C(129),
        UINT8_C(248), UINT8_C(140), UINT8_C( 90), UINT8_C(  7), UINT8_C( 27), UINT8_C(118), UINT8_C(187), UINT8_C(106) },
      UINT32_C(   1048643) },
    { UINT32_C(2582628555),
      { UINT8_C(185), UINT8_C(197), UINT8_C(151), UINT8_C(119), UINT8_C( 55), UINT8_C(166), UINT8_C(158), UINT8_C(165),
        UINT8_C( 19), UINT8_C(108), UINT8_C(148), UINT8_C(206), UINT8_C( 85), UINT8_C(220), UINT8_C( 89), UINT8_C( 88),
        UINT8_C(151), UINT8_C(244), UINT8_C(217), UINT8_C(143), UINT8_C(128), UINT8_C( 52), UINT8_C(150), UINT8_C(155),
        UINT8_C(170), UINT8_C( 81), UINT8_C(  5), UINT8_C(117), UINT8_C( 25), UINT8_C(244), UINT8_C( 14), UINT8_C(210) },
      { UINT8_C(186), UINT8_C(166), UINT8_C( 73), UINT8_C(241), UINT8_C( 76), UINT8_C(231), UINT8_C(150), UINT8_C( 95),
        UINT8_C( 84), UINT8_C( 43), UINT8_C( 46), UINT8_C(169), UINT8_C(  7), UINT8_C(135), UINT8_C(  2), UINT8_C(158),
        UINT8_C(123), UINT8_C(219), UINT8_C( 46), UINT8_C(251), UINT8_C( 15), UINT8_C(196), UINT8_C(150), UINT8_C(185),
        UINT8_C( 22), UINT8_C(155), UINT8_C( 47), UINT8_C( 47), UINT8_C(143), UINT8_C( 61), UINT8_C(  2), UINT8_C( 73) },
      UINT32_C(2298955970) },
    { UINT32_C( 809126883),
      { UINT8_C( 51), UINT8_C(209), UINT8_C(143), UINT8_C(135), UINT8_C(252), UINT8_C(189), UINT8_C( 48), UINT8_C(  3),
        UINT8_C( 68), UINT8_C( 50), UINT8_C(162), UINT8_C(191), UINT8_C( 14), UINT8_C(208), UINT8_C(186), UINT8_C( 29),
        UINT8_C(148), UINT8_C( 80), UINT8_C(215), UINT8_C(170), UINT8_C(235), UINT8_C(  6), UINT8_C(218), UINT8_C(123),
        UINT8_C( 67), UINT8_C(220), UINT8_C(196), UINT8_C( 39), UINT8_C( 39),    UINT8_MAX, UINT8_C( 87), UINT8_C( 90) },
      { UINT8_C(208), UINT8_C(230), UINT8_C(225), UINT8_C(204), UINT8_C(164), UINT8_C( 18), UINT8_C(207), UINT8_C(232),
        UINT8_C( 68), UINT8_C(113), UINT8_C(168), UINT8_C( 82), UINT8_C( 65), UINT8_C( 98), UINT8_C(112), UINT8_C(214),
        UINT8_C(179), UINT8_C( 71), UINT8_C(128), UINT8_C(158), UINT8_C( 77), UINT8_C( 90), UINT8_C( 25), UINT8_C(144),
        UINT8_C( 54), UINT8_C(222), UINT8_C(183), UINT8_C( 94), UINT8_C(221), UINT8_C( 14), UINT8_C(184), UINT8_C(173) },
      UINT32_C( 538593312) },
    { UINT32_C(2574883573),
      { UINT8_C(172), UINT8_C( 72), UINT8_C(129), UINT8_C(240), UINT8_C(186), UINT8_C( 41), UINT8_C( 67), UINT8_C(251),
        UINT8_C(140), UINT8_C(179), UINT8_C(209), UINT8_C( 63), UINT8_C(250), UINT8_C( 82), UINT8_C(221), UINT8_C( 71),
        UINT8_C(172), UINT8_C(247), UINT8_C(215), UINT8_C(227), UINT8_C(213), UINT8_C(143), UINT8_C( 65), UINT8_C(178),
        UINT8_C(157), UINT8_C(249), UINT8_C( 95), UINT8_C(146), UINT8_C(147), UINT8_C(216), UINT8_C( 43), UINT8_C( 63) },
      { UINT8_C( 32), UINT8_C(173), UINT8_C( 48), UINT8_C(218), UINT8_C(214), UINT8_C(115), UINT8_C(214), UINT8_C( 98),
        UINT8_C( 38), UINT8_C(167), UINT8_C(161), UINT8_C( 32), UINT8_C(249), UINT8_C(127), UINT8_C(103), UINT8_C(166),
        UINT8_C(118), UINT8_C( 62), UINT8_C(137), UINT8_C( 75), UINT8_C(205), UINT8_C(202), UINT8_C(253), UINT8_C(107),
        UINT8_C(195), UINT8_C( 92), UINT8_C(253), UINT8_C( 87), UINT8_C( 52), UINT8_C( 41), UINT8_C(150), UINT8_C( 84) },
      UINT32_C( 404298373) },
    { UINT32_C(2888812246),
      { UINT8_C( 57), UINT8_C(  5), UINT8_C( 15), UINT8_C( 95), UINT8_C(172), UINT8_C(176), UINT8_C(127), UINT8_C(166),
        UINT8_C( 47), UINT8_C(230), UINT8_C( 76), UINT8_C(165), UINT8_C( 37), UINT8_C(213), UINT8_C(240), UINT8_C(242),
        UINT8_C(159), UINT8_C(237), UINT8_C( 93), UINT8_C( 98), UINT8_C( 73), UINT8_C( 91), UINT8_C(185), UINT8_C(125),
        UINT8_C(132), UINT8_C( 80), UINT8_C(210), UINT8_C( 90), UINT8_C( 22), UINT8_C(  1), UINT8_C(  6), UINT8_C( 80) },
      { UINT8_C(  6), UINT8_C( 21), UINT8_C(175), UINT8_C(178), UINT8_C(198), UINT8_C( 47), UINT8_C( 88), UINT8_C(245),
        UINT8_C( 21), UINT8_C(164), UINT8_C(155), UINT8_C( 58), UINT8_C(121), UINT8_C(139), UINT8_C( 45), UINT8_C( 24),
        UINT8_C(121), UINT8_C(138), UINT8_C(123), UINT8_C(194), UINT8_C(229), UINT8_C( 52), UINT8_C( 64), UINT8_C(105),
        UINT8_C(132), UINT8_C( 18), UINT8_C(195), UINT8_C(155), UINT8_C( 19), UINT8_C(202), UINT8_C(235), UINT8_C( 25) },
      UINT32_C(2216936000) },
    { UINT32_C(2781584095),
      { UINT8_C(201), UINT8_C( 36), UINT8_C(155), UINT8_C(223), UINT8_C(200), UINT8_C( 54), UINT8_C( 25), UINT8_C( 66),
        UINT8_C(193), UINT8_C( 70), UINT8_C( 90), UINT8_C( 58), UINT8_C(209), UINT8_C(213), UINT8_C(253), UINT8_C(182),
        UINT8_C( 10), UINT8_C( 61), UINT8_C( 32), UINT8_C(142), UINT8_C( 79), UINT8_C(227), UINT8_C( 41), UINT8_C( 98),
        UINT8_C(173), UINT8_C( 20), UINT8_C(123), UINT8_C(141), UINT8_C(175), UINT8_C( 70), UINT8_C( 50), UINT8_C(120) },
      { UINT8_C(106), UINT8_C(205), UINT8_C( 87), UINT8_C( 51), UINT8_C(  3), UINT8_C(113), UINT8_C(117), UINT8_C(197),
        UINT8_C(183), UINT8_C(207),    UINT8_MAX, UINT8_C(136), UINT8_C(165), UINT8_C(252), UINT8_C( 63), UINT8_C(175),
        UINT8_C( 57), UINT8_C( 95), UINT8_C( 61), UINT8_C(136), UINT8_C( 66), UINT8_C(103), UINT8_C(234), UINT8_C(240),
        UINT8_C(123), UINT8_C(101), UINT8_C(125), UINT8_C( 42), UINT8_C(172), UINT8_C(175), UINT8_C(163), UINT8_C( 22) },
      UINT32_C(2164822045) },
    { UINT32_C(2152331901),
      { UINT8_C(107), UINT8_C(190), UINT8_C( 69), UINT8_C( 35), UINT8_C(142), UINT8_C( 69), UINT8_C(171), UINT8_C( 51),
        UINT8_C( 65), UINT8_C(234), UINT8_C(226), UINT8_C(123), UINT8_C( 73), UINT8_C( 31), UINT8_C(  3), UINT8_C(140),
        UINT8_C(134), UINT8_C(238), UINT8_C(124), UINT8_C(  2), UINT8_C( 83), UINT8_C(249), UINT8_C( 44),    UINT8_MAX,
        UINT8_C(168), UINT8_C(207), UINT8_C( 22), UINT8_C( 37), UINT8_C(202), UINT8_C( 95), UINT8_C(166), UINT8_C( 53) },
      { UINT8_C( 30), UINT8_C(235), UINT8_C( 88), UINT8_C(172), UINT8_C( 48), UINT8_C(  4), UINT8_C(223), UINT8_C(114),
        UINT8_C(238), UINT8_C(193), UINT8_C(237), UINT8_C( 56), UINT8_C(224), UINT8_C(240), UINT8_C(196), UINT8_C(103),
        UINT8_C(222), UINT8_C( 64), UINT8_C(105), UINT8_C( 50), UINT8_C( 57), UINT8_C(149), UINT8_C( 49), UINT8_C(225),
        UINT8_C(101), UINT8_C( 71), UINT8_C(  7), UINT8_C( 47), UINT8_C(167), UINT8_C(173), UINT8_C(100), UINT8_C(197) },
      UINT32_C(     35377) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epu8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epu8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 r = easysimd_mm256_mask_cmpgt_epu8_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    uint16_t a[16];
    uint16_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(32976),
      { UINT16_C(12299), UINT16_C(42116), UINT16_C(16371), UINT16_C( 2048), UINT16_C(43499), UINT16_C(13073), UINT16_C( 7481), UINT16_C(30123),
        UINT16_C(58415), UINT16_C(29762), UINT16_C(63249), UINT16_C(39172), UINT16_C(36819), UINT16_C( 2383), UINT16_C( 8064), UINT16_C(35977) },
      { UINT16_C( 3408), UINT16_C(17200), UINT16_C(12620), UINT16_C(14411), UINT16_C(24026), UINT16_C( 4971), UINT16_C( 6010), UINT16_C(43400),
        UINT16_C(52219), UINT16_C( 3101), UINT16_C( 8898), UINT16_C(38309), UINT16_C(62641), UINT16_C(12702), UINT16_C(10003), UINT16_C(25533) },
      UINT16_C(32848) },
    { UINT16_C(60981),
      { UINT16_C(33190), UINT16_C(61983), UINT16_C(63929), UINT16_C( 9551), UINT16_C(51468), UINT16_C(38204), UINT16_C(14194), UINT16_C(36704),
        UINT16_C( 8771), UINT16_C(59569), UINT16_C(25272), UINT16_C(22236), UINT16_C(61332), UINT16_C(20862), UINT16_C(45907), UINT16_C(63807) },
      { UINT16_C(24116), UINT16_C(61163), UINT16_C(14935), UINT16_C(25619), UINT16_C(20227), UINT16_C(30201), UINT16_C(22918), UINT16_C(51461),
        UINT16_C(46715), UINT16_C(13233), UINT16_C(36121), UINT16_C(44426), UINT16_C( 2172), UINT16_C(53246), UINT16_C(16059), UINT16_C(61385) },
      UINT16_C(49717) },
    { UINT16_C(46236),
      { UINT16_C(62685), UINT16_C(61679), UINT16_C(62040), UINT16_C(20799), UINT16_C(50536), UINT16_C(28074), UINT16_C( 9614), UINT16_C(16163),
        UINT16_C(15449), UINT16_C(58316), UINT16_C(18921), UINT16_C(59627), UINT16_C(42520), UINT16_C(57638), UINT16_C(49813), UINT16_C(29590) },
      { UINT16_C(34230), UINT16_C( 3683), UINT16_C(41847), UINT16_C(57183), UINT16_C( 2408), UINT16_C(63308), UINT16_C(28719), UINT16_C(34870),
        UINT16_C(  940), UINT16_C(38507), UINT16_C(22092), UINT16_C(25726), UINT16_C(42236), UINT16_C(37190), UINT16_C(56422), UINT16_C( 7428) },
      UINT16_C(45076) },
    { UINT16_C(26721),
      { UINT16_C(55339), UINT16_C(35595), UINT16_C(29624), UINT16_C( 1172), UINT16_C(50026), UINT16_C(41332), UINT16_C( 8523), UINT16_C(46756),
        UINT16_C(61623), UINT16_C(13580), UINT16_C( 2132), UINT16_C(39641), UINT16_C(16282), UINT16_C(40566), UINT16_C(55132), UINT16_C(34822) },
      { UINT16_C( 4528), UINT16_C(26643), UINT16_C(42885), UINT16_C(61292), UINT16_C(57707), UINT16_C(46736), UINT16_C(13314), UINT16_C(47469),
        UINT16_C(31012), UINT16_C(31214), UINT16_C(51074), UINT16_C( 7187), UINT16_C(35334), UINT16_C(25530), UINT16_C(49505), UINT16_C( 4587) },
      UINT16_C(26625) },
    { UINT16_C(65234),
      { UINT16_C(22393), UINT16_C(59045), UINT16_C( 4167), UINT16_C(55239), UINT16_C(51655), UINT16_C(13324), UINT16_C(12418), UINT16_C(28845),
        UINT16_C(12201), UINT16_C(48439), UINT16_C(15691), UINT16_C( 1607), UINT16_C(43168), UINT16_C(35783), UINT16_C(39354), UINT16_C(13193) },
      { UINT16_C(12273), UINT16_C(14361), UINT16_C(57407), UINT16_C( 1551), UINT16_C( 7081), UINT16_C(11066), UINT16_C(59468), UINT16_C(62875),
        UINT16_C(53783), UINT16_C(25522), UINT16_C(63760), UINT16_C(45161), UINT16_C(12450), UINT16_C(23612), UINT16_C(50633), UINT16_C(47759) },
      UINT16_C(12818) },
    { UINT16_C(43508),
      { UINT16_C(13554), UINT16_C(  649), UINT16_C(13114), UINT16_C(29981), UINT16_C(26974), UINT16_C(64093), UINT16_C(29791), UINT16_C( 4556),
        UINT16_C(56535), UINT16_C(16395), UINT16_C(44429), UINT16_C(51568), UINT16_C(14857), UINT16_C(39054), UINT16_C(33780), UINT16_C(59201) },
      { UINT16_C(52151), UINT16_C(61929), UINT16_C( 1790), UINT16_C(23654), UINT16_C(50032), UINT16_C(53078), UINT16_C( 9016), UINT16_C( 4064),
        UINT16_C(60415), UINT16_C(35920), UINT16_C(49304), UINT16_C(41301), UINT16_C(58618), UINT16_C(61242), UINT16_C(31591), UINT16_C( 7894) },
      UINT16_C(35044) },
    { UINT16_C(48966),
      { UINT16_C(17423), UINT16_C(30405), UINT16_C(13729), UINT16_C(63289), UINT16_C(28932), UINT16_C(58650), UINT16_C( 6785), UINT16_C(53712),
        UINT16_C(27046), UINT16_C(64657), UINT16_C(35850), UINT16_C(17632), UINT16_C(18299), UINT16_C(20928), UINT16_C( 1637), UINT16_C(29712) },
      { UINT16_C(54603), UINT16_C(60650), UINT16_C( 9227), UINT16_C( 4067), UINT16_C(65173), UINT16_C( 5876), UINT16_C(50456), UINT16_C(48871),
        UINT16_C(31022), UINT16_C(14522), UINT16_C(39429), UINT16_C(32893), UINT16_C(15841), UINT16_C(18129), UINT16_C(57667), UINT16_C(36539) },
      UINT16_C(12804) },
    { UINT16_C(42422),
      { UINT16_C(49530), UINT16_C(24265), UINT16_C(24529), UINT16_C(50524), UINT16_C(29813), UINT16_C(23946), UINT16_C(47154), UINT16_C(60886),
        UINT16_C(56305), UINT16_C(28295), UINT16_C(26971), UINT16_C(11435), UINT16_C(61103), UINT16_C(27149), UINT16_C(50045), UINT16_C(63248) },
      { UINT16_C(55685), UINT16_C(22101), UINT16_C(45368), UINT16_C(44571), UINT16_C(42533), UINT16_C(22539), UINT16_C(57694), UINT16_C(20293),
        UINT16_C(52412), UINT16_C( 6077), UINT16_C(26677), UINT16_C(58691), UINT16_C(20567), UINT16_C(54351), UINT16_C(24339), UINT16_C(39115) },
      UINT16_C(34210) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epu16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epu16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 r = easysimd_mm256_mask_cmpgt_epu16_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint32_t a[8];
    uint32_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(115),
      { UINT32_C( 197169677), UINT32_C( 304618279), UINT32_C(1957297679), UINT32_C( 302250331), UINT32_C( 681845924), UINT32_C(1005149899), UINT32_C(1726942487), UINT32_C(1104768052) },
      { UINT32_C( 625777150), UINT32_C(3275257012), UINT32_C(3443057010), UINT32_C(2145336282), UINT32_C( 715621215), UINT32_C(4184248545), UINT32_C(3529463198), UINT32_C(3205708225) },
      UINT8_C(  0) },
    { UINT8_C(210),
      { UINT32_C(3548898399), UINT32_C(4249176604), UINT32_C(3185054338), UINT32_C(1964857330), UINT32_C(2404861950), UINT32_C(  36524205), UINT32_C(3905159087), UINT32_C(1908113938) },
      { UINT32_C(2202288743), UINT32_C( 260147596), UINT32_C(2378979739), UINT32_C(2919492016), UINT32_C(3728562737), UINT32_C(1507945130), UINT32_C(2067965033), UINT32_C(2397895975) },
      UINT8_C( 66) },
    { UINT8_C( 63),
      { UINT32_C(3117093168), UINT32_C(3948272274), UINT32_C(2426135207), UINT32_C(1069632229), UINT32_C(4075397255), UINT32_C( 626738049), UINT32_C(2186073989), UINT32_C(4106345155) },
      { UINT32_C(2125303020), UINT32_C( 241828455), UINT32_C(3399353829), UINT32_C(3607781455), UINT32_C(2177495808), UINT32_C(3148227638), UINT32_C(3208508411), UINT32_C(3115581133) },
      UINT8_C( 19) },
    { UINT8_C(139),
      { UINT32_C(1676818528), UINT32_C(2806513826), UINT32_C(4277605022), UINT32_C( 285134364), UINT32_C(3141959575), UINT32_C( 431424038), UINT32_C(1055290943), UINT32_C(2311692329) },
      { UINT32_C(2062334936), UINT32_C(1495348411), UINT32_C(1666717767), UINT32_C(2104710886), UINT32_C(4231641814), UINT32_C(4212519100), UINT32_C(2402941798), UINT32_C(1930953371) },
      UINT8_C(130) },
    { UINT8_C(189),
      { UINT32_C( 964226309), UINT32_C( 662753807), UINT32_C(2148459562), UINT32_C( 290884439), UINT32_C(3033354948), UINT32_C(1662699879), UINT32_C( 100575490), UINT32_C(3351409346) },
      { UINT32_C(1845508959), UINT32_C( 932610317), UINT32_C(3166151781), UINT32_C(4107144751), UINT32_C(3366493024), UINT32_C(1730921316), UINT32_C( 778840428), UINT32_C(4227149467) },
      UINT8_C(  0) },
    { UINT8_C(105),
      { UINT32_C(2004249078), UINT32_C(2749148671), UINT32_C(1943246949), UINT32_C(  47433574), UINT32_C( 845585263), UINT32_C(4037004742), UINT32_C(1753992505), UINT32_C(3117516483) },
      { UINT32_C(4012919024), UINT32_C(1519586549), UINT32_C( 198010532), UINT32_C(2618138925), UINT32_C(  63927100), UINT32_C(2062773825), UINT32_C(4276256315), UINT32_C(4122457093) },
      UINT8_C( 32) },
    { UINT8_C(252),
      { UINT32_C(4092716263), UINT32_C(3717745783), UINT32_C(3121259289), UINT32_C( 603432880), UINT32_C(3831822966), UINT32_C(1797251053), UINT32_C(1970282177), UINT32_C(3178325461) },
      { UINT32_C(3249562442), UINT32_C(3365882031), UINT32_C(2609097195), UINT32_C(3334437456), UINT32_C(1638540148), UINT32_C(3301755394), UINT32_C(3174645224), UINT32_C(3967462306) },
      UINT8_C( 20) },
    { UINT8_C( 14),
      { UINT32_C(1941810731), UINT32_C(4133455181), UINT32_C(2185755144), UINT32_C(3723890105), UINT32_C(2195675064), UINT32_C(1634378532), UINT32_C(2281908189), UINT32_C(3449221282) },
      { UINT32_C(3946861470), UINT32_C(3789660120), UINT32_C(1399007386), UINT32_C(3996146230), UINT32_C(3580891313), UINT32_C(2419514035), UINT32_C(2736273921), UINT32_C(3362827818) },
      UINT8_C(  6) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epu32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 r = easysimd_mm256_mask_cmpgt_epu32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpgt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint64_t a[4];
    uint64_t b[4];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 12),
      { UINT64_C(11896637522884990985), UINT64_C(11714632060864082703), UINT64_C( 8408546572390355999), UINT64_C( 3016593850323752289) },
      { UINT64_C( 7575344129143341488), UINT64_C( 3469708023991063994), UINT64_C( 6320949369777737397), UINT64_C(12511850775999734875) },
      UINT8_C(  4) },
    { UINT8_C(  6),
      { UINT64_C(11480458622830324113), UINT64_C(14677197092044199704), UINT64_C(10417980061213979572), UINT64_C( 4921373030171348801) },
      { UINT64_C(10110070391689751652), UINT64_C(10578733277374774790), UINT64_C(17033564502459388200), UINT64_C(10332804743288778863) },
      UINT8_C(  2) },
    { UINT8_C( 43),
      { UINT64_C( 9905434882843032898), UINT64_C(15713051720729197234), UINT64_C( 3932892920692466373), UINT64_C( 6723293818065630627) },
      { UINT64_C(14646626389932829998), UINT64_C( 7630155103977253567), UINT64_C( 9016244773317404835), UINT64_C(11551712647181205975) },
      UINT8_C(  2) },
    { UINT8_C(218),
      { UINT64_C( 5125868904785336605), UINT64_C( 8634618805519854448), UINT64_C( 6907340806941621986), UINT64_C( 4266159117918391917) },
      { UINT64_C( 6174083131178297291), UINT64_C(14947547141382773226), UINT64_C(18110855980078317592), UINT64_C( 7265864866644436861) },
      UINT8_C(  0) },
    { UINT8_C(176),
      { UINT64_C( 1377514871600827251), UINT64_C( 5108155679701538416), UINT64_C( 6327785278632417700), UINT64_C(12924579142396045860) },
      { UINT64_C( 5525075464351511071), UINT64_C( 2514691388681465368), UINT64_C(12491372971236374995), UINT64_C(13664968687149075120) },
      UINT8_C(  0) },
    { UINT8_C(171),
      { UINT64_C(12872365275940870024), UINT64_C(16192783891024593643), UINT64_C( 6898977058381741000), UINT64_C(15696658033557745266) },
      { UINT64_C( 2073916635326209298), UINT64_C( 1464155087541762122), UINT64_C( 4966028733994679157), UINT64_C(13022233788865767310) },
      UINT8_C( 11) },
    { UINT8_C(161),
      { UINT64_C(10047511605154073117), UINT64_C(17014012466282553583), UINT64_C( 8753311655808378892), UINT64_C( 3411142546546252145) },
      { UINT64_C(  412293865921587355), UINT64_C( 9710265995707125667), UINT64_C(17545722340974952307), UINT64_C(  470624423085906502) },
      UINT8_C(  1) },
    { UINT8_C( 38),
      { UINT64_C( 8881452424674161627), UINT64_C( 6209473053193410286), UINT64_C( 5341493118507692525), UINT64_C(15895379020409567603) },
      { UINT64_C(  240586427878527983), UINT64_C(12898284892977908434), UINT64_C( 1663425456861141018), UINT64_C( 8431337212935195173) },
      UINT8_C(  4) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpgt_epu64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_mask_cmpgt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 r = easysimd_mm256_mask_cmpgt_epu64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpgt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i   a;
    easysimd__m512i   b;
    easysimd__mmask64 r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C(  92), INT8_C(-121), INT8_C( 120), INT8_C( -19),
                           INT8_C( -73), INT8_C(  22), INT8_C( -66), INT8_C( -29),
                           INT8_C(  55), INT8_C( -63), INT8_C( -45), INT8_C(-119),
                           INT8_C(  30), INT8_C( -29), INT8_C( -26), INT8_C(  89),
                           INT8_C( -13), INT8_C( 119), INT8_C( -88), INT8_C( 109),
                           INT8_C(  68), INT8_C( -10), INT8_C(   0), INT8_C( -18),
                           INT8_C(   1), INT8_C( 116), INT8_C( -13), INT8_C( -73),
                           INT8_C(-122), INT8_C(   0), INT8_C( 111), INT8_C(  95),
                           INT8_C( -20), INT8_C(   3), INT8_C( 122), INT8_C( -43),
                           INT8_C(  13), INT8_C(  25), INT8_C( -35), INT8_C(-127),
                           INT8_C(  82), INT8_C(  90), INT8_C(  53), INT8_C( 123),
                           INT8_C(  73), INT8_C( 108), INT8_C( -18), INT8_C(  15),
                           INT8_C( 115), INT8_C(  54), INT8_C( 102), INT8_C( 118),
                           INT8_C(  39), INT8_C( -77), INT8_C(  45), INT8_C(  81),
                           INT8_C( -17), INT8_C( -28), INT8_C(  67), INT8_C( -85),
                           INT8_C(  79), INT8_C(-113), INT8_C(-122), INT8_C( 124)),
      easysimd_mm512_set_epi8(INT8_C(   9), INT8_C(-121), INT8_C( 120), INT8_C( -36),
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
                           INT8_C( 110), INT8_C(  31), INT8_C(  51), INT8_C(-104),
                           INT8_C(-115), INT8_C( -27), INT8_C(  62), INT8_C( -85),
                           INT8_C(  49), INT8_C(-115), INT8_C(  38), INT8_C(   4)),
      UINT64_C(0x969d5243643621ad) },
    { easysimd_mm512_set_epi8(INT8_C(  16), INT8_C(  41), INT8_C( 102), INT8_C( -95),
                           INT8_C(  17), INT8_C(  42), INT8_C( -86), INT8_C(  38),
                           INT8_C(  62), INT8_C( -56), INT8_C(  60), INT8_C(  19),
                           INT8_C(-100), INT8_C(-107), INT8_C( 105), INT8_C( -76),
                           INT8_C(  10), INT8_C(  -9), INT8_C( -12), INT8_C( -56),
                           INT8_C( -71), INT8_C(  96), INT8_C(  31), INT8_C(  24),
                           INT8_C(  68), INT8_C(  -7), INT8_C( -27), INT8_C(   3),
                           INT8_C( -21), INT8_C(  50), INT8_C( -67), INT8_C( -39),
                           INT8_C(-108), INT8_C(  62), INT8_C( 101), INT8_C(  56),
                           INT8_C( -79), INT8_C( -37), INT8_C( -99), INT8_C( -56),
                           INT8_C(-119), INT8_C( -56), INT8_C(  -8), INT8_C( -16),
                           INT8_C( 115), INT8_C( -95), INT8_C( -73), INT8_C(  54),
                           INT8_C( 125), INT8_C(  85), INT8_C(  78), INT8_C( -65),
                           INT8_C(   4), INT8_C(  38), INT8_C( -84), INT8_C( -53),
                           INT8_C(  68), INT8_C(-108), INT8_C(-121), INT8_C(-115),
                           INT8_C( -59), INT8_C( -78), INT8_C(-111), INT8_C(  56)),
      easysimd_mm512_set_epi8(INT8_C(-104), INT8_C(-102), INT8_C( 102), INT8_C( -95),
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
                           INT8_C(  81), INT8_C(  38), INT8_C( -84), INT8_C(  -2),
                           INT8_C( -14), INT8_C(  85), INT8_C( -80), INT8_C(  80),
                           INT8_C(  48), INT8_C(  93), INT8_C(  79), INT8_C( 127)),
      UINT64_C(0xc46ac5246043f080) },
    { easysimd_mm512_set_epi8(INT8_C(  50), INT8_C(  43), INT8_C( -68), INT8_C(  97),
                           INT8_C( -26), INT8_C(-103), INT8_C(  71), INT8_C(-107),
                           INT8_C(  91), INT8_C(  45), INT8_C( -11), INT8_C(  47),
                           INT8_C(  29), INT8_C( -56), INT8_C(  26), INT8_C(  -9),
                           INT8_C(  10), INT8_C(  36), INT8_C(-116), INT8_C( -53),
                           INT8_C(  41), INT8_C(   1), INT8_C( -23), INT8_C(  61),
                           INT8_C(-127), INT8_C(  -4), INT8_C(  48), INT8_C( -68),
                           INT8_C(  89), INT8_C(-112), INT8_C( -31), INT8_C( 120),
                           INT8_C(  35), INT8_C(  62), INT8_C( -21), INT8_C(-114),
                           INT8_C(-104), INT8_C(  57), INT8_C(  42), INT8_C(-111),
                           INT8_C(  94), INT8_C( -63), INT8_C(  87), INT8_C(  64),
                           INT8_C( -65), INT8_C(  -2), INT8_C( 110), INT8_C(  -8),
                           INT8_C(  63), INT8_C( -51), INT8_C(  -4), INT8_C(  32),
                           INT8_C( -65), INT8_C(  55), INT8_C(  14), INT8_C(  81),
                           INT8_C(-123), INT8_C(-100), INT8_C( -39), INT8_C( -44),
                           INT8_C(  22), INT8_C( 112), INT8_C(  16), INT8_C(  15)),
      easysimd_mm512_set_epi8(INT8_C(  50), INT8_C( -11), INT8_C( -68), INT8_C( -31),
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
                           INT8_C(  -5), INT8_C(  -3), INT8_C(-127), INT8_C(  41),
                           INT8_C(  74), INT8_C(  72), INT8_C(  -4), INT8_C(  28),
                           INT8_C(-115), INT8_C(  93), INT8_C( 102), INT8_C(  44),
                           INT8_C(-103), INT8_C( -29), INT8_C( -50), INT8_C(  48),
                           INT8_C( -96), INT8_C( -50), INT8_C(  46), INT8_C( -65)),
      UINT64_C(0x5491482be6d6192d) },
    { easysimd_mm512_set_epi8(INT8_C( -97), INT8_C(  28), INT8_C( -58), INT8_C(  11),
                           INT8_C( -14), INT8_C( 126), INT8_C(  81), INT8_C(  45),
                           INT8_C( -23), INT8_C( 120), INT8_C( -83), INT8_C( -16),
                           INT8_C(   7), INT8_C(  51), INT8_C( -57), INT8_C( -50),
                           INT8_C( -21), INT8_C(  98), INT8_C(  88), INT8_C(   0),
                           INT8_C( -66), INT8_C(   3), INT8_C( 124), INT8_C(-113),
                           INT8_C(  50), INT8_C(  32), INT8_C( -85), INT8_C( -93),
                           INT8_C( -44), INT8_C( -13), INT8_C( -94), INT8_C(  17),
                           INT8_C( 122), INT8_C(  79), INT8_C(-116), INT8_C(  43),
                           INT8_C( -77), INT8_C(-125), INT8_C( -23), INT8_C(-120),
                           INT8_C(  96), INT8_C( -64), INT8_C( -23), INT8_C( -46),
                           INT8_C( -29), INT8_C( -71), INT8_C(  71), INT8_C( -80),
                           INT8_C(  44), INT8_C( -92), INT8_C( -31), INT8_C(  26),
                           INT8_C(   8), INT8_C(  52), INT8_C( 117), INT8_C( 123),
                           INT8_C( -63), INT8_C(  45), INT8_C(  95), INT8_C(  24),
                           INT8_C(-108), INT8_C(  18), INT8_C( -60), INT8_C(  28)),
      easysimd_mm512_set_epi8(INT8_C( -23), INT8_C(-101), INT8_C( 116), INT8_C( 127),
                           INT8_C(  96), INT8_C(  40), INT8_C( -97), INT8_C(  40),
                           INT8_C(  86), INT8_C( -44), INT8_C(  70), INT8_C( -71),
                           INT8_C(  62), INT8_C( -21), INT8_C(  66), INT8_C(  68),
                           INT8_C( -87), INT8_C( -61), INT8_C(  48), INT8_C( -70),
                           INT8_C(  18), INT8_C( -78), INT8_C( -98), INT8_C( 117),
                           INT8_C(  74), INT8_C(  32), INT8_C(  93), INT8_C( 125),
                           INT8_C( -47), INT8_C( -60), INT8_C( -86), INT8_C( 117),
                           INT8_C( 122), INT8_C( -54), INT8_C(  50), INT8_C( 123),
                           INT8_C( -31), INT8_C( -74), INT8_C( -64), INT8_C(  54),
                           INT8_C( -81), INT8_C(  60), INT8_C(  31), INT8_C( -23),
                           INT8_C( 108), INT8_C(-119), INT8_C( -92), INT8_C( -80),
                           INT8_C( -30), INT8_C( -37), INT8_C(  51), INT8_C( -36),
                           INT8_C(   8), INT8_C(  52), INT8_C(  97), INT8_C( 123),
                           INT8_C( -49), INT8_C(-124), INT8_C(  95), INT8_C( -83),
                           INT8_C(  70), INT8_C( -50), INT8_C( -61), INT8_C(  25)),
      UINT64_C(0x4754f60c42869257) },
    { easysimd_mm512_set_epi8(INT8_C( 121), INT8_C( -20), INT8_C( -89), INT8_C( -94),
                           INT8_C( 112), INT8_C( -27), INT8_C(  81), INT8_C( -54),
                           INT8_C( -64), INT8_C(-114), INT8_C(  48), INT8_C( -89),
                           INT8_C( -61), INT8_C(  26), INT8_C(  43), INT8_C(  29),
                           INT8_C(   0), INT8_C( 125), INT8_C( -42), INT8_C( -67),
                           INT8_C(  15), INT8_C( 120), INT8_C(  36), INT8_C(  40),
                           INT8_C( -53), INT8_C(  34), INT8_C(-108), INT8_C( -58),
                           INT8_C(  26), INT8_C(-111), INT8_C(  63), INT8_C( -98),
                           INT8_C(  65), INT8_C(   8), INT8_C(-124), INT8_C(  96),
                           INT8_C( -13), INT8_C( -98), INT8_C(  99), INT8_C(  90),
                           INT8_C(  56), INT8_C( -45), INT8_C(-108), INT8_C( -19),
                           INT8_C(-124), INT8_C( -27), INT8_C(  22), INT8_C( 126),
                           INT8_C(-106), INT8_C( -68), INT8_C( -60), INT8_C(   8),
                           INT8_C( -54), INT8_C(  93), INT8_C( -33), INT8_C( -27),
                           INT8_C(  -7), INT8_C(  27), INT8_C(-122), INT8_C( -88),
                           INT8_C(  23), INT8_C(   6), INT8_C(  45), INT8_C( -21)),
      easysimd_mm512_set_epi8(INT8_C( 105), INT8_C(  52), INT8_C(  85), INT8_C(-104),
                           INT8_C(  57), INT8_C( -31), INT8_C( -38), INT8_C(-124),
                           INT8_C(-107), INT8_C(  -2), INT8_C(  55), INT8_C(  46),
                           INT8_C( -71), INT8_C(  77), INT8_C(  18), INT8_C(  70),
                           INT8_C(  89), INT8_C( 125), INT8_C( -42), INT8_C(-125),
                           INT8_C( 121), INT8_C( -11), INT8_C( -69), INT8_C( -59),
                           INT8_C( -53), INT8_C(  34), INT8_C(   9), INT8_C(  64),
                           INT8_C( -61), INT8_C( -25), INT8_C(-115), INT8_C( 100),
                           INT8_C(  65), INT8_C(   8), INT8_C(  69), INT8_C(  -8),
                           INT8_C( -15), INT8_C( -51), INT8_C(   1), INT8_C(  90),
                           INT8_C( 115), INT8_C(  51), INT8_C( -91), INT8_C(  56),
                           INT8_C(  64), INT8_C( -39), INT8_C(-119), INT8_C( -28),
                           INT8_C( -54), INT8_C(  28), INT8_C(  54), INT8_C(  -8),
                           INT8_C( -54), INT8_C(-128), INT8_C( -28), INT8_C( -71),
                           INT8_C( 107), INT8_C( -66), INT8_C(-114), INT8_C( -88),
                           INT8_C(  34), INT8_C( -83), INT8_C( -21), INT8_C( -64)),
      UINT64_C(0x9f8a170a1a071547) },
    { easysimd_mm512_set_epi8(INT8_C(  33), INT8_C( 121), INT8_C( 125), INT8_C(  35),
                           INT8_C(-103), INT8_C( -48), INT8_C( -22), INT8_C(  38),
                           INT8_C( -81), INT8_C(   9), INT8_C( -11), INT8_C(-124),
                           INT8_C(  71), INT8_C(   9), INT8_C( -42), INT8_C( 118),
                           INT8_C(  67), INT8_C(  45), INT8_C(  51), INT8_C( -92),
                           INT8_C( 126), INT8_C( 108), INT8_C(-123), INT8_C( -71),
                           INT8_C( 113), INT8_C(  32), INT8_C(  71), INT8_C(   3),
                           INT8_C( -26), INT8_C(  82), INT8_C( -81), INT8_C( -20),
                           INT8_C( -55), INT8_C( 112), INT8_C(  66), INT8_C(  37),
                           INT8_C(  67), INT8_C( -69), INT8_C(  64), INT8_C(  39),
                           INT8_C(  72), INT8_C(  45), INT8_C( 120), INT8_C(  -5),
                           INT8_C(-109), INT8_C(  62), INT8_C(  17), INT8_C(  31),
                           INT8_C( -30), INT8_C( -58), INT8_C(  56), INT8_C(  21),
                           INT8_C(  72), INT8_C( -75), INT8_C( -34), INT8_C( 120),
                           INT8_C(  95), INT8_C( 108), INT8_C(  32), INT8_C(  64),
                           INT8_C(-128), INT8_C( 102), INT8_C( -21), INT8_C(  28)),
      easysimd_mm512_set_epi8(INT8_C( 100), INT8_C( 121), INT8_C(  18), INT8_C(  28),
                           INT8_C(-117), INT8_C( 107), INT8_C(   3), INT8_C( -62),
                           INT8_C(  42), INT8_C(  72), INT8_C(  91), INT8_C(  86),
                           INT8_C( -72), INT8_C(   9), INT8_C( -80), INT8_C( 118),
                           INT8_C( 122), INT8_C(-108), INT8_C( -70), INT8_C( -63),
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
                           INT8_C(  11), INT8_C( -94), INT8_C( -32), INT8_C( 110)),
      UINT64_C(0x390a6cac5bf669f6) },
    { easysimd_mm512_set_epi8(INT8_C(   2), INT8_C( -81), INT8_C(  14), INT8_C(  90),
                           INT8_C(-100), INT8_C(-122), INT8_C( -35), INT8_C(  81),
                           INT8_C( -14), INT8_C( -42), INT8_C( 125), INT8_C(-125),
                           INT8_C( -57), INT8_C(  90), INT8_C(  -9), INT8_C(  63),
                           INT8_C(  53), INT8_C(  77), INT8_C(  63), INT8_C( -84),
                           INT8_C(  27), INT8_C(  22), INT8_C( -28), INT8_C( -37),
                           INT8_C(  65), INT8_C( 118), INT8_C(-126), INT8_C(  97),
                           INT8_C( 109), INT8_C(-119), INT8_C(-114), INT8_C( -75),
                           INT8_C(-125), INT8_C( 121), INT8_C(-128), INT8_C( 103),
                           INT8_C(   0), INT8_C( 101), INT8_C( -41), INT8_C(  89),
                           INT8_C(  67), INT8_C( -65), INT8_C(   9), INT8_C(  -7),
                           INT8_C( -63), INT8_C(  13), INT8_C( 105), INT8_C(  92),
                           INT8_C( -18), INT8_C( -21), INT8_C(-102), INT8_C(-114),
                           INT8_C(  74), INT8_C( 121), INT8_C( -45), INT8_C(  52),
                           INT8_C( -63), INT8_C( -93), INT8_C(  98), INT8_C( 106),
                           INT8_C(-109), INT8_C( -47), INT8_C(  37), INT8_C(  70)),
      easysimd_mm512_set_epi8(INT8_C( -42), INT8_C(-124), INT8_C(  54), INT8_C(  74),
                           INT8_C( -92), INT8_C(  99), INT8_C(  79), INT8_C(  -3),
                           INT8_C(  61), INT8_C( -89), INT8_C(  84), INT8_C( -94),
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
                           INT8_C( 106), INT8_C(  22), INT8_C(-124), INT8_C(  -4)),
      UINT64_C(0xd167f84855174f33) },
    { easysimd_mm512_set_epi8(INT8_C(  18), INT8_C(  13), INT8_C(  14), INT8_C(   4),
                           INT8_C(  -3), INT8_C( -64), INT8_C(  17), INT8_C(-115),
                           INT8_C(  21), INT8_C( -34), INT8_C( 125), INT8_C( -60),
                           INT8_C( -72), INT8_C(  74), INT8_C(  -5), INT8_C( -21),
                           INT8_C( -41), INT8_C(  22), INT8_C(  45), INT8_C( 102),
                           INT8_C(  59), INT8_C( -80), INT8_C( -15), INT8_C( -63),
                           INT8_C(  84), INT8_C( -71), INT8_C(   8), INT8_C(  12),
                           INT8_C( -11), INT8_C( -76), INT8_C(  62), INT8_C(  93),
                           INT8_C( -75), INT8_C( -77), INT8_C( -84), INT8_C(-108),
                           INT8_C( -35), INT8_C(  14), INT8_C( -60), INT8_C(  18),
                           INT8_C(  23), INT8_C( -60), INT8_C( -63), INT8_C(-114),
                           INT8_C( -55), INT8_C(  75), INT8_C( -99), INT8_C( -55),
                           INT8_C(  58), INT8_C(  76), INT8_C(-102), INT8_C(-118),
                           INT8_C(  10), INT8_C(  39), INT8_C( 119), INT8_C(  85),
                           INT8_C(  -8), INT8_C( -72), INT8_C( -60), INT8_C( -94),
                           INT8_C(-112), INT8_C( 119), INT8_C( 124), INT8_C(  76)),
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
      UINT64_C(0x48a450990f8b8607) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpgt_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpgt_epi8_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
}

static int
test_easysimd_mm512_cmpgt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i   a;
    easysimd__m512i   b;
    easysimd__mmask64 r;
  } test_vec[8] = {
    { easysimd_x_mm512_set_epu8(UINT8_C( 92), UINT8_C(135), UINT8_C(120), UINT8_C(237),
                             UINT8_C(183), UINT8_C( 22), UINT8_C(190), UINT8_C(227),
                             UINT8_C( 55), UINT8_C(193), UINT8_C(211), UINT8_C(137),
                             UINT8_C( 30), UINT8_C(227), UINT8_C(230), UINT8_C( 89),
                             UINT8_C(243), UINT8_C(119), UINT8_C(168), UINT8_C(109),
                             UINT8_C( 68), UINT8_C(246), UINT8_C(  0), UINT8_C(238),
                             UINT8_C(  1), UINT8_C(116), UINT8_C(243), UINT8_C(183),
                             UINT8_C(134), UINT8_C(  0), UINT8_C(111), UINT8_C( 95),
                             UINT8_C(236), UINT8_C(  3), UINT8_C(122), UINT8_C(213),
                             UINT8_C( 13), UINT8_C( 25), UINT8_C(221), UINT8_C(129),
                             UINT8_C( 82), UINT8_C( 90), UINT8_C( 53), UINT8_C(123),
                             UINT8_C( 73), UINT8_C(108), UINT8_C(238), UINT8_C( 15),
                             UINT8_C(115), UINT8_C( 54), UINT8_C(102), UINT8_C(118),
                             UINT8_C( 39), UINT8_C(179), UINT8_C( 45), UINT8_C( 81),
                             UINT8_C(239), UINT8_C(228), UINT8_C( 67), UINT8_C(171),
                             UINT8_C( 79), UINT8_C(143), UINT8_C(134), UINT8_C(124)),
      easysimd_x_mm512_set_epu8(UINT8_C(  9), UINT8_C(135), UINT8_C(120), UINT8_C(220),
                             UINT8_C( 36), UINT8_C(153), UINT8_C(152), UINT8_C( 25),
                             UINT8_C(145), UINT8_C(193), UINT8_C(221), UINT8_C(136),
                             UINT8_C( 28), UINT8_C(212), UINT8_C(230), UINT8_C(170),
                             UINT8_C(243), UINT8_C( 53), UINT8_C(168), UINT8_C(149),
                             UINT8_C( 68), UINT8_C( 42), UINT8_C(138), UINT8_C(111),
                             UINT8_C( 54), UINT8_C(198), UINT8_C(243), UINT8_C( 27),
                             UINT8_C( 23), UINT8_C( 41), UINT8_C(137), UINT8_C( 44),
                             UINT8_C(  7), UINT8_C(136), UINT8_C( 32), UINT8_C(213),
                             UINT8_C(114), UINT8_C(184), UINT8_C( 73), UINT8_C(160),
                             UINT8_C( 96), UINT8_C(110), UINT8_C(175), UINT8_C(180),
                             UINT8_C(103), UINT8_C(156), UINT8_C(234), UINT8_C( 18),
                             UINT8_C(115), UINT8_C( 54), UINT8_C(216), UINT8_C(125),
                             UINT8_C(110), UINT8_C( 31), UINT8_C( 51), UINT8_C(152),
                             UINT8_C(141), UINT8_C(229), UINT8_C( 62), UINT8_C(171),
                             UINT8_C( 49), UINT8_C(141), UINT8_C( 38), UINT8_C(  4)),
      UINT64_C(0x9b1c4519a20204af) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 16), UINT8_C( 41), UINT8_C(102), UINT8_C(161),
                             UINT8_C( 17), UINT8_C( 42), UINT8_C(170), UINT8_C( 38),
                             UINT8_C( 62), UINT8_C(200), UINT8_C( 60), UINT8_C( 19),
                             UINT8_C(156), UINT8_C(149), UINT8_C(105), UINT8_C(180),
                             UINT8_C( 10), UINT8_C(247), UINT8_C(244), UINT8_C(200),
                             UINT8_C(185), UINT8_C( 96), UINT8_C( 31), UINT8_C( 24),
                             UINT8_C( 68), UINT8_C(249), UINT8_C(229), UINT8_C(  3),
                             UINT8_C(235), UINT8_C( 50), UINT8_C(189), UINT8_C(217),
                             UINT8_C(148), UINT8_C( 62), UINT8_C(101), UINT8_C( 56),
                             UINT8_C(177), UINT8_C(219), UINT8_C(157), UINT8_C(200),
                             UINT8_C(137), UINT8_C(200), UINT8_C(248), UINT8_C(240),
                             UINT8_C(115), UINT8_C(161), UINT8_C(183), UINT8_C( 54),
                             UINT8_C(125), UINT8_C( 85), UINT8_C( 78), UINT8_C(191),
                             UINT8_C(  4), UINT8_C( 38), UINT8_C(172), UINT8_C(203),
                             UINT8_C( 68), UINT8_C(148), UINT8_C(135), UINT8_C(141),
                             UINT8_C(197), UINT8_C(178), UINT8_C(145), UINT8_C( 56)),
      easysimd_x_mm512_set_epu8(UINT8_C(152), UINT8_C(154), UINT8_C(102), UINT8_C(161),
                             UINT8_C( 69), UINT8_C( 13), UINT8_C(235), UINT8_C( 45),
                             UINT8_C( 62), UINT8_C(136), UINT8_C(145), UINT8_C( 32),
                             UINT8_C(149), UINT8_C(226), UINT8_C( 99), UINT8_C(192),
                             UINT8_C(  8), UINT8_C(214), UINT8_C( 81), UINT8_C(222),
                             UINT8_C(210), UINT8_C( 26), UINT8_C( 31), UINT8_C(254),
                             UINT8_C( 68), UINT8_C(249), UINT8_C(185), UINT8_C( 46),
                             UINT8_C(235), UINT8_C(183), UINT8_C( 21), UINT8_C( 83),
                             UINT8_C(148), UINT8_C(159), UINT8_C(187), UINT8_C( 73),
                             UINT8_C( 57), UINT8_C(219), UINT8_C( 21), UINT8_C( 82),
                             UINT8_C(137), UINT8_C(130), UINT8_C(126), UINT8_C( 91),
                             UINT8_C(115), UINT8_C( 31), UINT8_C(177), UINT8_C( 28),
                             UINT8_C(150), UINT8_C(238), UINT8_C( 65), UINT8_C(152),
                             UINT8_C( 81), UINT8_C( 38), UINT8_C(172), UINT8_C(254),
                             UINT8_C(242), UINT8_C( 85), UINT8_C(176), UINT8_C( 80),
                             UINT8_C( 48), UINT8_C( 93), UINT8_C( 79), UINT8_C(127)),
      UINT64_C(0x44ae4230b77305e) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 50), UINT8_C( 43), UINT8_C(188), UINT8_C( 97),
                             UINT8_C(230), UINT8_C(153), UINT8_C( 71), UINT8_C(149),
                             UINT8_C( 91), UINT8_C( 45), UINT8_C(245), UINT8_C( 47),
                             UINT8_C( 29), UINT8_C(200), UINT8_C( 26), UINT8_C(247),
                             UINT8_C( 10), UINT8_C( 36), UINT8_C(140), UINT8_C(203),
                             UINT8_C( 41), UINT8_C(  1), UINT8_C(233), UINT8_C( 61),
                             UINT8_C(129), UINT8_C(252), UINT8_C( 48), UINT8_C(188),
                             UINT8_C( 89), UINT8_C(144), UINT8_C(225), UINT8_C(120),
                             UINT8_C( 35), UINT8_C( 62), UINT8_C(235), UINT8_C(142),
                             UINT8_C(152), UINT8_C( 57), UINT8_C( 42), UINT8_C(145),
                             UINT8_C( 94), UINT8_C(193), UINT8_C( 87), UINT8_C( 64),
                             UINT8_C(191), UINT8_C(254), UINT8_C(110), UINT8_C(248),
                             UINT8_C( 63), UINT8_C(205), UINT8_C(252), UINT8_C( 32),
                             UINT8_C(191), UINT8_C( 55), UINT8_C( 14), UINT8_C( 81),
                             UINT8_C(133), UINT8_C(156), UINT8_C(217), UINT8_C(212),
                             UINT8_C( 22), UINT8_C(112), UINT8_C( 16), UINT8_C( 15)),
      easysimd_x_mm512_set_epu8(UINT8_C( 50), UINT8_C(245), UINT8_C(188), UINT8_C(225),
                             UINT8_C(105), UINT8_C(150), UINT8_C( 98), UINT8_C( 51),
                             UINT8_C( 58), UINT8_C(103), UINT8_C(111), UINT8_C(129),
                             UINT8_C( 68), UINT8_C(200), UINT8_C(124), UINT8_C(137),
                             UINT8_C( 74), UINT8_C(194), UINT8_C(140), UINT8_C( 37),
                             UINT8_C(244), UINT8_C(114), UINT8_C(  0), UINT8_C( 61),
                             UINT8_C(103), UINT8_C(252), UINT8_C(151), UINT8_C(188),
                             UINT8_C( 39), UINT8_C(156), UINT8_C(163), UINT8_C( 11),
                             UINT8_C(176), UINT8_C(237), UINT8_C(234), UINT8_C(217),
                             UINT8_C(127), UINT8_C(218), UINT8_C(131), UINT8_C(145),
                             UINT8_C( 84), UINT8_C(160), UINT8_C( 87), UINT8_C(234),
                             UINT8_C(251), UINT8_C(253), UINT8_C(129), UINT8_C( 41),
                             UINT8_C( 74), UINT8_C( 72), UINT8_C(252), UINT8_C( 28),
                             UINT8_C(141), UINT8_C( 93), UINT8_C(102), UINT8_C( 44),
                             UINT8_C(153), UINT8_C(227), UINT8_C(206), UINT8_C( 48),
                             UINT8_C(160), UINT8_C(206), UINT8_C( 46), UINT8_C(191)),
      UINT64_C(0xda1128b28c55930) },
    { easysimd_x_mm512_set_epu8(UINT8_C(159), UINT8_C( 28), UINT8_C(198), UINT8_C( 11),
                             UINT8_C(242), UINT8_C(126), UINT8_C( 81), UINT8_C( 45),
                             UINT8_C(233), UINT8_C(120), UINT8_C(173), UINT8_C(240),
                             UINT8_C(  7), UINT8_C( 51), UINT8_C(199), UINT8_C(206),
                             UINT8_C(235), UINT8_C( 98), UINT8_C( 88), UINT8_C(  0),
                             UINT8_C(190), UINT8_C(  3), UINT8_C(124), UINT8_C(143),
                             UINT8_C( 50), UINT8_C( 32), UINT8_C(171), UINT8_C(163),
                             UINT8_C(212), UINT8_C(243), UINT8_C(162), UINT8_C( 17),
                             UINT8_C(122), UINT8_C( 79), UINT8_C(140), UINT8_C( 43),
                             UINT8_C(179), UINT8_C(131), UINT8_C(233), UINT8_C(136),
                             UINT8_C( 96), UINT8_C(192), UINT8_C(233), UINT8_C(210),
                             UINT8_C(227), UINT8_C(185), UINT8_C( 71), UINT8_C(176),
                             UINT8_C( 44), UINT8_C(164), UINT8_C(225), UINT8_C( 26),
                             UINT8_C(  8), UINT8_C( 52), UINT8_C(117), UINT8_C(123),
                             UINT8_C(193), UINT8_C( 45), UINT8_C( 95), UINT8_C( 24),
                             UINT8_C(148), UINT8_C( 18), UINT8_C(196), UINT8_C( 28)),
      easysimd_x_mm512_set_epu8(UINT8_C(233), UINT8_C(155), UINT8_C(116), UINT8_C(127),
                             UINT8_C( 96), UINT8_C( 40), UINT8_C(159), UINT8_C( 40),
                             UINT8_C( 86), UINT8_C(212), UINT8_C( 70), UINT8_C(185),
                             UINT8_C( 62), UINT8_C(235), UINT8_C( 66), UINT8_C( 68),
                             UINT8_C(169), UINT8_C(195), UINT8_C( 48), UINT8_C(186),
                             UINT8_C( 18), UINT8_C(178), UINT8_C(158), UINT8_C(117),
                             UINT8_C( 74), UINT8_C( 32), UINT8_C( 93), UINT8_C(125),
                             UINT8_C(209), UINT8_C(196), UINT8_C(170), UINT8_C(117),
                             UINT8_C(122), UINT8_C(202), UINT8_C( 50), UINT8_C(123),
                             UINT8_C(225), UINT8_C(182), UINT8_C(192), UINT8_C( 54),
                             UINT8_C(175), UINT8_C( 60), UINT8_C( 31), UINT8_C(233),
                             UINT8_C(108), UINT8_C(137), UINT8_C(164), UINT8_C(176),
                             UINT8_C(226), UINT8_C(219), UINT8_C( 51), UINT8_C(220),
                             UINT8_C(  8), UINT8_C( 52), UINT8_C( 97), UINT8_C(123),
                             UINT8_C(207), UINT8_C(132), UINT8_C( 95), UINT8_C(173),
                             UINT8_C( 70), UINT8_C(206), UINT8_C(195), UINT8_C( 25)),
      UINT64_C(0x2db3a93c236c220b) },
    { easysimd_x_mm512_set_epu8(UINT8_C(121), UINT8_C(236), UINT8_C(167), UINT8_C(162),
                             UINT8_C(112), UINT8_C(229), UINT8_C( 81), UINT8_C(202),
                             UINT8_C(192), UINT8_C(142), UINT8_C( 48), UINT8_C(167),
                             UINT8_C(195), UINT8_C( 26), UINT8_C( 43), UINT8_C( 29),
                             UINT8_C(  0), UINT8_C(125), UINT8_C(214), UINT8_C(189),
                             UINT8_C( 15), UINT8_C(120), UINT8_C( 36), UINT8_C( 40),
                             UINT8_C(203), UINT8_C( 34), UINT8_C(148), UINT8_C(198),
                             UINT8_C( 26), UINT8_C(145), UINT8_C( 63), UINT8_C(158),
                             UINT8_C( 65), UINT8_C(  8), UINT8_C(132), UINT8_C( 96),
                             UINT8_C(243), UINT8_C(158), UINT8_C( 99), UINT8_C( 90),
                             UINT8_C( 56), UINT8_C(211), UINT8_C(148), UINT8_C(237),
                             UINT8_C(132), UINT8_C(229), UINT8_C( 22), UINT8_C(126),
                             UINT8_C(150), UINT8_C(188), UINT8_C(196), UINT8_C(  8),
                             UINT8_C(202), UINT8_C( 93), UINT8_C(223), UINT8_C(229),
                             UINT8_C(249), UINT8_C( 27), UINT8_C(134), UINT8_C(168),
                             UINT8_C( 23), UINT8_C(  6), UINT8_C( 45), UINT8_C(235)),
      easysimd_x_mm512_set_epu8(UINT8_C(105), UINT8_C( 52), UINT8_C( 85), UINT8_C(152),
                             UINT8_C( 57), UINT8_C(225), UINT8_C(218), UINT8_C(132),
                             UINT8_C(149), UINT8_C(254), UINT8_C( 55), UINT8_C( 46),
                             UINT8_C(185), UINT8_C( 77), UINT8_C( 18), UINT8_C( 70),
                             UINT8_C( 89), UINT8_C(125), UINT8_C(214), UINT8_C(131),
                             UINT8_C(121), UINT8_C(245), UINT8_C(187), UINT8_C(197),
                             UINT8_C(203), UINT8_C( 34), UINT8_C(  9), UINT8_C( 64),
                             UINT8_C(195), UINT8_C(231), UINT8_C(141), UINT8_C(100),
                             UINT8_C( 65), UINT8_C(  8), UINT8_C( 69), UINT8_C(248),
                             UINT8_C(241), UINT8_C(205), UINT8_C(  1), UINT8_C( 90),
                             UINT8_C(115), UINT8_C( 51), UINT8_C(165), UINT8_C( 56),
                             UINT8_C( 64), UINT8_C(217), UINT8_C(137), UINT8_C(228),
                             UINT8_C(202), UINT8_C( 28), UINT8_C( 54), UINT8_C(248),
                             UINT8_C(202), UINT8_C(128), UINT8_C(228), UINT8_C(185),
                             UINT8_C(107), UINT8_C(190), UINT8_C(142), UINT8_C(168),
                             UINT8_C( 34), UINT8_C(173), UINT8_C(235), UINT8_C(192)),
      UINT64_C(0xfd9a10312a5c6181) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 33), UINT8_C(121), UINT8_C(125), UINT8_C( 35),
                             UINT8_C(153), UINT8_C(208), UINT8_C(234), UINT8_C( 38),
                             UINT8_C(175), UINT8_C(  9), UINT8_C(245), UINT8_C(132),
                             UINT8_C( 71), UINT8_C(  9), UINT8_C(214), UINT8_C(118),
                             UINT8_C( 67), UINT8_C( 45), UINT8_C( 51), UINT8_C(164),
                             UINT8_C(126), UINT8_C(108), UINT8_C(133), UINT8_C(185),
                             UINT8_C(113), UINT8_C( 32), UINT8_C( 71), UINT8_C(  3),
                             UINT8_C(230), UINT8_C( 82), UINT8_C(175), UINT8_C(236),
                             UINT8_C(201), UINT8_C(112), UINT8_C( 66), UINT8_C( 37),
                             UINT8_C( 67), UINT8_C(187), UINT8_C( 64), UINT8_C( 39),
                             UINT8_C( 72), UINT8_C( 45), UINT8_C(120), UINT8_C(251),
                             UINT8_C(147), UINT8_C( 62), UINT8_C( 17), UINT8_C( 31),
                             UINT8_C(226), UINT8_C(198), UINT8_C( 56), UINT8_C( 21),
                             UINT8_C( 72), UINT8_C(181), UINT8_C(222), UINT8_C(120),
                             UINT8_C( 95), UINT8_C(108), UINT8_C( 32), UINT8_C( 64),
                             UINT8_C(128), UINT8_C(102), UINT8_C(235), UINT8_C( 28)),
      easysimd_x_mm512_set_epu8(UINT8_C(100), UINT8_C(121), UINT8_C( 18), UINT8_C( 28),
                             UINT8_C(139), UINT8_C(107), UINT8_C(  3), UINT8_C(194),
                             UINT8_C( 42), UINT8_C( 72), UINT8_C( 91), UINT8_C( 86),
                             UINT8_C(184), UINT8_C(  9), UINT8_C(176), UINT8_C(118),
                             UINT8_C(122), UINT8_C(148), UINT8_C(186), UINT8_C(193),
                             UINT8_C( 56), UINT8_C( 71), UINT8_C(242), UINT8_C( 49),
                             UINT8_C(183), UINT8_C( 53), UINT8_C(227), UINT8_C(  3),
                             UINT8_C(183), UINT8_C( 43), UINT8_C(234), UINT8_C( 85),
                             UINT8_C(230), UINT8_C(247), UINT8_C( 66), UINT8_C(  1),
                             UINT8_C(243), UINT8_C( 60), UINT8_C(137), UINT8_C(173),
                             UINT8_C(134), UINT8_C(192), UINT8_C(173), UINT8_C(182),
                             UINT8_C(119), UINT8_C(248), UINT8_C( 12), UINT8_C(113),
                             UINT8_C(244), UINT8_C(172), UINT8_C(  6), UINT8_C( 69),
                             UINT8_C(  2), UINT8_C(181), UINT8_C(222), UINT8_C(130),
                             UINT8_C(  3), UINT8_C(128), UINT8_C(247), UINT8_C( 24),
                             UINT8_C( 11), UINT8_C(162), UINT8_C(224), UINT8_C(110)),
      UINT64_C(0x3eb20d0d141a689a) },
    { easysimd_x_mm512_set_epu8(UINT8_C(  2), UINT8_C(175), UINT8_C( 14), UINT8_C( 90),
                             UINT8_C(156), UINT8_C(134), UINT8_C(221), UINT8_C( 81),
                             UINT8_C(242), UINT8_C(214), UINT8_C(125), UINT8_C(131),
                             UINT8_C(199), UINT8_C( 90), UINT8_C(247), UINT8_C( 63),
                             UINT8_C( 53), UINT8_C( 77), UINT8_C( 63), UINT8_C(172),
                             UINT8_C( 27), UINT8_C( 22), UINT8_C(228), UINT8_C(219),
                             UINT8_C( 65), UINT8_C(118), UINT8_C(130), UINT8_C( 97),
                             UINT8_C(109), UINT8_C(137), UINT8_C(142), UINT8_C(181),
                             UINT8_C(131), UINT8_C(121), UINT8_C(128), UINT8_C(103),
                             UINT8_C(  0), UINT8_C(101), UINT8_C(215), UINT8_C( 89),
                             UINT8_C( 67), UINT8_C(191), UINT8_C(  9), UINT8_C(249),
                             UINT8_C(193), UINT8_C( 13), UINT8_C(105), UINT8_C( 92),
                             UINT8_C(238), UINT8_C(235), UINT8_C(154), UINT8_C(142),
                             UINT8_C( 74), UINT8_C(121), UINT8_C(211), UINT8_C( 52),
                             UINT8_C(193), UINT8_C(163), UINT8_C( 98), UINT8_C(106),
                             UINT8_C(147), UINT8_C(209), UINT8_C( 37), UINT8_C( 70)),
      easysimd_x_mm512_set_epu8(UINT8_C(214), UINT8_C(132), UINT8_C( 54), UINT8_C( 74),
                             UINT8_C(164), UINT8_C( 99), UINT8_C( 79), UINT8_C(253),
                             UINT8_C( 61), UINT8_C(167), UINT8_C( 84), UINT8_C(162),
                             UINT8_C( 31), UINT8_C(140), UINT8_C(189), UINT8_C(154),
                             UINT8_C(184), UINT8_C(165), UINT8_C(151), UINT8_C(148),
                             UINT8_C(212), UINT8_C( 74), UINT8_C(228), UINT8_C(124),
                             UINT8_C(120), UINT8_C(215), UINT8_C(177), UINT8_C(122),
                             UINT8_C( 87), UINT8_C(137), UINT8_C( 54), UINT8_C(254),
                             UINT8_C(209), UINT8_C( 84), UINT8_C(130), UINT8_C(192),
                             UINT8_C( 14), UINT8_C( 11), UINT8_C( 37), UINT8_C(233),
                             UINT8_C( 67), UINT8_C(124), UINT8_C( 58), UINT8_C(162),
                             UINT8_C( 30), UINT8_C(223), UINT8_C( 70), UINT8_C(232),
                             UINT8_C( 38), UINT8_C(159), UINT8_C(200), UINT8_C(196),
                             UINT8_C(197), UINT8_C( 65), UINT8_C(182), UINT8_C( 45),
                             UINT8_C(245), UINT8_C( 55), UINT8_C(174), UINT8_C( 12),
                             UINT8_C(106), UINT8_C( 22), UINT8_C(132), UINT8_C(252)),
      UINT64_C(0x56ea110a465ac75c) },
    { easysimd_x_mm512_set_epu8(UINT8_C( 18), UINT8_C( 13), UINT8_C( 14), UINT8_C(  4),
                             UINT8_C(253), UINT8_C(192), UINT8_C( 17), UINT8_C(141),
                             UINT8_C( 21), UINT8_C(222), UINT8_C(125), UINT8_C(196),
                             UINT8_C(184), UINT8_C( 74), UINT8_C(251), UINT8_C(235),
                             UINT8_C(215), UINT8_C( 22), UINT8_C( 45), UINT8_C(102),
                             UINT8_C( 59), UINT8_C(176), UINT8_C(241), UINT8_C(193),
                             UINT8_C( 84), UINT8_C(185), UINT8_C(  8), UINT8_C( 12),
                             UINT8_C(245), UINT8_C(180), UINT8_C( 62), UINT8_C( 93),
                             UINT8_C(181), UINT8_C(179), UINT8_C(172), UINT8_C(148),
                             UINT8_C(221), UINT8_C( 14), UINT8_C(196), UINT8_C( 18),
                             UINT8_C( 23), UINT8_C(196), UINT8_C(193), UINT8_C(142),
                             UINT8_C(201), UINT8_C( 75), UINT8_C(157), UINT8_C(201),
                             UINT8_C( 58), UINT8_C( 76), UINT8_C(154), UINT8_C(138),
                             UINT8_C( 10), UINT8_C( 39), UINT8_C(119), UINT8_C( 85),
                             UINT8_C(248), UINT8_C(184), UINT8_C(196), UINT8_C(162),
                             UINT8_C(144), UINT8_C(119), UINT8_C(124), UINT8_C( 76)),
      easysimd_x_mm512_set_epu8(UINT8_C( 18), UINT8_C(182), UINT8_C( 14), UINT8_C( 36),
                             UINT8_C(249), UINT8_C(113), UINT8_C( 40), UINT8_C( 48),
                             UINT8_C(149), UINT8_C(222), UINT8_C(181), UINT8_C( 85),
                             UINT8_C(221), UINT8_C(140), UINT8_C( 65), UINT8_C(235),
                             UINT8_C( 15), UINT8_C(  3), UINT8_C( 45), UINT8_C( 21),
                             UINT8_C( 72), UINT8_C( 93), UINT8_C(108), UINT8_C(125),
                             UINT8_C(  1), UINT8_C( 75), UINT8_C( 21), UINT8_C(220),
                             UINT8_C(130), UINT8_C(122), UINT8_C( 71), UINT8_C( 76),
                             UINT8_C( 28), UINT8_C(200), UINT8_C( 32), UINT8_C(101),
                             UINT8_C(149), UINT8_C(145), UINT8_C(168), UINT8_C(237),
                             UINT8_C(179), UINT8_C( 19), UINT8_C(235), UINT8_C(145),
                             UINT8_C(188), UINT8_C( 82), UINT8_C(138), UINT8_C(180),
                             UINT8_C( 47), UINT8_C(127), UINT8_C( 62), UINT8_C(240),
                             UINT8_C( 10), UINT8_C(242), UINT8_C(156), UINT8_C( 86),
                             UINT8_C( 29), UINT8_C(107), UINT8_C( 56), UINT8_C( 21),
                             UINT8_C( 24), UINT8_C( 68), UINT8_C(160), UINT8_C( 64)),
      UINT64_C(0xd12d7cdba4ba0fd) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 r;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpgt_epu8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpgt_epu8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
   }

  return 0;
}

static int
test_easysimd_mm512_cmpgt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t   arr_a[32];
    int16_t   arr_b[32];
    easysimd__mmask32  r;
  } test_vec[8] = {
    { { -INT16_C( 13460),  INT16_C(  2734),  INT16_C( 14875), -INT16_C(  4031),  INT16_C( 29735),  INT16_C(  4329),  INT16_C( 22910), -INT16_C( 29294),
         INT16_C( 32649),  INT16_C( 21977), -INT16_C(  3246),  INT16_C( 13968), -INT16_C(  6545), -INT16_C( 28649),  INT16_C(  1579), -INT16_C( 26384),
        -INT16_C( 24879), -INT16_C(  4958), -INT16_C(  6951),  INT16_C(   220), -INT16_C( 15016), -INT16_C( 10736), -INT16_C( 23778), -INT16_C( 22429),
         INT16_C( 15650),  INT16_C( 29949), -INT16_C( 29392), -INT16_C( 24661), -INT16_C( 15756), -INT16_C( 24785),  INT16_C(  8136), -INT16_C( 26313) },
      { -INT16_C(  9538), -INT16_C( 26747),  INT16_C( 25022),  INT16_C(  5783), -INT16_C( 22489),  INT16_C( 17901),  INT16_C( 20555),  INT16_C( 28141),
        -INT16_C(  5491), -INT16_C( 16671), -INT16_C( 29576), -INT16_C(  5027), -INT16_C( 29361),  INT16_C(  6027), -INT16_C( 15444),  INT16_C( 27313),
         INT16_C( 13981),  INT16_C( 23297), -INT16_C( 26216), -INT16_C( 16527),  INT16_C( 24129), -INT16_C( 29692), -INT16_C(  3409),  INT16_C( 15609),
        -INT16_C(  9508),  INT16_C( 21754),  INT16_C( 22631), -INT16_C( 18880), -INT16_C( 13083), -INT16_C( 28211),  INT16_C( 32399),  INT16_C( 11516) },
      UINT32_C( 590110546) },
    { { -INT16_C(   587),  INT16_C( 19847), -INT16_C(  1898), -INT16_C( 10484),  INT16_C(  4183),  INT16_C(  1635),  INT16_C( 23554), -INT16_C(  8382),
         INT16_C( 15671), -INT16_C( 25037),  INT16_C( 29845),  INT16_C( 31316),  INT16_C(  8512), -INT16_C( 12533),  INT16_C(  1952),  INT16_C( 22011),
        -INT16_C( 32251), -INT16_C( 25694), -INT16_C( 20870), -INT16_C( 11917), -INT16_C( 10562), -INT16_C( 15913),  INT16_C(  6707),  INT16_C( 27296),
        -INT16_C( 11433), -INT16_C(  5112),  INT16_C( 23623), -INT16_C( 30874),  INT16_C( 29053),  INT16_C(  7510),  INT16_C( 20857),  INT16_C( 32370) },
      {  INT16_C(  5331),  INT16_C( 19993), -INT16_C( 29502), -INT16_C( 32481), -INT16_C(  2205), -INT16_C( 27070), -INT16_C(  7663),  INT16_C( 26624),
         INT16_C(  2229), -INT16_C(   684), -INT16_C( 17820), -INT16_C(  7804), -INT16_C(  9429), -INT16_C( 23297),  INT16_C( 28972),  INT16_C(    34),
         INT16_C( 15494),  INT16_C( 18510),  INT16_C( 28104),  INT16_C( 11209),  INT16_C(  2916),  INT16_C( 30145), -INT16_C( 15891), -INT16_C( 23587),
         INT16_C( 12745),  INT16_C( 11680),  INT16_C(  9451),  INT16_C(  5903),  INT16_C(  3839),  INT16_C( 11451), -INT16_C(  8577),  INT16_C(  1324) },
      UINT32_C(3569401212) },
    { {  INT16_C( 31258), -INT16_C(  7602),  INT16_C(  6119),  INT16_C( 19470), -INT16_C( 12509),  INT16_C(  4289), -INT16_C( 24687),  INT16_C( 23219),
         INT16_C( 21456), -INT16_C( 17272), -INT16_C( 26760),  INT16_C( 30675), -INT16_C( 29019),  INT16_C(  9379), -INT16_C( 12436), -INT16_C( 31190),
         INT16_C( 30793),  INT16_C( 12649),  INT16_C( 30607), -INT16_C( 19843),  INT16_C( 15942), -INT16_C( 10301),  INT16_C( 30429), -INT16_C( 20942),
        -INT16_C( 17718),  INT16_C( 17002),  INT16_C( 15697), -INT16_C(  2375),  INT16_C( 24011),  INT16_C( 14362),  INT16_C( 17452),  INT16_C( 30398) },
      {  INT16_C( 10172),  INT16_C( 19623),  INT16_C(  9374), -INT16_C(  6658), -INT16_C( 16030),  INT16_C( 16572), -INT16_C(  4552),  INT16_C(   750),
         INT16_C( 22696), -INT16_C(  1724), -INT16_C(   619),  INT16_C( 24815),  INT16_C(  2650), -INT16_C( 30824),  INT16_C( 22350),  INT16_C(  3069),
        -INT16_C( 23426),  INT16_C(  7511),  INT16_C( 21960),  INT16_C( 10754), -INT16_C( 16873),  INT16_C( 20330),  INT16_C( 22701),  INT16_C( 21841),
        -INT16_C( 27216),  INT16_C( 17743),  INT16_C( 16018), -INT16_C(  4698),  INT16_C( 15944), -INT16_C( 26764),  INT16_C( 29077),  INT16_C(  5282) },
      UINT32_C(3109496985) },
    { { -INT16_C(  1771), -INT16_C(  8911),  INT16_C( 13134),  INT16_C( 25863),  INT16_C( 29425), -INT16_C( 24908),  INT16_C(  1482),  INT16_C( 31732),
         INT16_C( 17306),  INT16_C( 11712),  INT16_C( 26241), -INT16_C( 13798), -INT16_C( 29019),  INT16_C( 14945),  INT16_C(  1023),  INT16_C(  5198),
         INT16_C( 32764),  INT16_C( 19185), -INT16_C(  1870), -INT16_C( 23376),  INT16_C( 25706),  INT16_C( 13634),  INT16_C( 13930),  INT16_C(  1200),
         INT16_C( 28793), -INT16_C(  1231),  INT16_C( 19415),  INT16_C( 31941),  INT16_C(  9945), -INT16_C( 10058),  INT16_C(  1321),  INT16_C(  9708) },
      { -INT16_C(  8828),  INT16_C( 14191),  INT16_C(  8150),  INT16_C( 16603),  INT16_C(  7556), -INT16_C(  4491),  INT16_C(  9556), -INT16_C( 12814),
         INT16_C(  9366),  INT16_C( 28104), -INT16_C( 29329),  INT16_C( 18921), -INT16_C( 24653), -INT16_C(  9183),  INT16_C(  3748),  INT16_C( 10497),
         INT16_C( 29163), -INT16_C( 16032),  INT16_C( 15248),  INT16_C(  5122),  INT16_C( 30552), -INT16_C( 21502), -INT16_C(  2659),  INT16_C( 13178),
         INT16_C( 16921), -INT16_C( 30560), -INT16_C( 30256), -INT16_C( 31791), -INT16_C(  3288), -INT16_C( 12960),  INT16_C( 24833), -INT16_C(  4874) },
      UINT32_C(3210945949) },
    { {  INT16_C( 22226),  INT16_C( 25518), -INT16_C( 20335), -INT16_C(  5769),  INT16_C( 31271), -INT16_C( 15210),  INT16_C(  4207), -INT16_C( 30473),
        -INT16_C( 26798),  INT16_C(  8720), -INT16_C(  7648),  INT16_C( 18854),  INT16_C(  1749), -INT16_C( 10730),  INT16_C(  3175),  INT16_C( 15042),
         INT16_C( 28770), -INT16_C(  3171),  INT16_C(  5152),  INT16_C( 18652),  INT16_C( 29326), -INT16_C(   756),  INT16_C(  1154), -INT16_C( 10875),
        -INT16_C( 26981), -INT16_C( 17161), -INT16_C( 25224),  INT16_C( 19717),  INT16_C(  7075),  INT16_C(  2851), -INT16_C(  6873), -INT16_C( 30395) },
      { -INT16_C(  7594),  INT16_C( 30332),  INT16_C( 22774), -INT16_C( 31298), -INT16_C( 13365),  INT16_C( 19842),  INT16_C(  2255),  INT16_C( 27170),
         INT16_C(  6814),  INT16_C(  5670),  INT16_C( 11191),  INT16_C( 23395), -INT16_C( 31162),  INT16_C( 28006), -INT16_C( 21653), -INT16_C( 15882),
         INT16_C( 29325), -INT16_C( 31944), -INT16_C(  2357), -INT16_C( 27128), -INT16_C( 29759), -INT16_C( 28445),  INT16_C(  1683),  INT16_C( 12795),
         INT16_C(  8480), -INT16_C( 10425), -INT16_C( 21939), -INT16_C( 27854), -INT16_C( 26576), -INT16_C( 25855), -INT16_C(  2237), -INT16_C( 12195) },
      UINT32_C( 943641177) },
    { { -INT16_C( 27286),  INT16_C( 13652),  INT16_C( 23691),  INT16_C( 19915), -INT16_C( 20761),  INT16_C( 31453), -INT16_C( 10060), -INT16_C( 11093),
        -INT16_C(  3334),  INT16_C( 18348), -INT16_C(  8548), -INT16_C( 13094), -INT16_C(  9353), -INT16_C( 17816), -INT16_C( 14893),  INT16_C( 15755),
        -INT16_C(  8358), -INT16_C(  6798),  INT16_C( 15675),  INT16_C(  9010),  INT16_C(  4331), -INT16_C( 24419),  INT16_C( 18920), -INT16_C(  7564),
         INT16_C(  8251), -INT16_C( 10199),  INT16_C(  1279),  INT16_C( 30372),  INT16_C(  3295), -INT16_C( 19920), -INT16_C( 17455),  INT16_C( 11247) },
      {  INT16_C( 24986), -INT16_C( 10735),  INT16_C( 17310), -INT16_C( 29959), -INT16_C( 27053),  INT16_C( 15402), -INT16_C( 24865),  INT16_C(  6942),
         INT16_C( 18623), -INT16_C( 16653), -INT16_C( 26804),  INT16_C( 11060),  INT16_C( 25764),  INT16_C( 30174), -INT16_C( 13024), -INT16_C( 17759),
        -INT16_C( 19921), -INT16_C( 12912), -INT16_C( 30219),  INT16_C( 18775), -INT16_C( 32480), -INT16_C(   123), -INT16_C( 23776), -INT16_C(  8422),
         INT16_C(  3563),  INT16_C( 14237), -INT16_C( 11867),  INT16_C( 18787),  INT16_C( 16693),  INT16_C( 21950),  INT16_C( 24334),  INT16_C( 15632) },
      UINT32_C( 232228478) },
    { { -INT16_C( 24559),  INT16_C(  1803),  INT16_C( 25130),  INT16_C( 19024), -INT16_C( 10780),  INT16_C(  1097),  INT16_C( 25720),  INT16_C( 25827),
        -INT16_C( 32655),  INT16_C(  5787), -INT16_C(   431), -INT16_C( 31137),  INT16_C(  7743),  INT16_C( 20188), -INT16_C(  4995), -INT16_C( 28789),
        -INT16_C( 26996), -INT16_C( 18794), -INT16_C(  6407), -INT16_C(  8960),  INT16_C( 19131),  INT16_C( 13281), -INT16_C( 15186),  INT16_C(  8087),
         INT16_C( 13124), -INT16_C( 27338), -INT16_C( 27343),  INT16_C( 28955), -INT16_C(  2125),  INT16_C( 12735),  INT16_C( 19171),  INT16_C( 28864) },
      {  INT16_C( 22241), -INT16_C(  9690),  INT16_C( 10044), -INT16_C(  2121), -INT16_C( 26511),  INT16_C(  7978), -INT16_C( 15780), -INT16_C( 24514),
         INT16_C( 29941),  INT16_C(  9781),  INT16_C( 20490), -INT16_C( 17001),  INT16_C( 22088),  INT16_C( 11246), -INT16_C( 20831), -INT16_C( 32101),
        -INT16_C( 15868),  INT16_C( 16476),  INT16_C(  5097),  INT16_C( 23095),  INT16_C( 25259),  INT16_C(  1913), -INT16_C( 18652),  INT16_C(  6567),
        -INT16_C(  9172),  INT16_C( 13887), -INT16_C( 10452),  INT16_C( 29939), -INT16_C(  7635), -INT16_C( 12640),  INT16_C( 15248), -INT16_C( 27312) },
      UINT32_C(4058046686) },
    { { -INT16_C( 21251), -INT16_C(  6443),  INT16_C(  3519),  INT16_C( 27200), -INT16_C( 18065), -INT16_C( 27791),  INT16_C(  6257), -INT16_C( 25172),
        -INT16_C(  5132),  INT16_C(  8659), -INT16_C( 14654), -INT16_C(  3947),  INT16_C( 13736),  INT16_C( 14782),  INT16_C(  3953),  INT16_C( 28366),
        -INT16_C( 23621),  INT16_C( 31573), -INT16_C( 27216),  INT16_C(  8165),  INT16_C( 22351), -INT16_C( 16206),  INT16_C( 24175),  INT16_C( 25693),
         INT16_C( 12362),  INT16_C(  3205),  INT16_C(  6902), -INT16_C( 24580), -INT16_C( 17584), -INT16_C( 15912), -INT16_C( 22838), -INT16_C( 31441) },
      { -INT16_C( 31671), -INT16_C(  1536), -INT16_C(  6630),  INT16_C( 26905), -INT16_C( 13251), -INT16_C( 21463), -INT16_C( 31190),  INT16_C( 29712),
        -INT16_C( 27210), -INT16_C( 21375),  INT16_C( 32176),  INT16_C(    75),  INT16_C(  9016),  INT16_C(   705), -INT16_C(  3895),  INT16_C(  5000),
        -INT16_C( 30603), -INT16_C( 28915),  INT16_C(  9838), -INT16_C( 21512),  INT16_C(  8690),  INT16_C(  7512),  INT16_C( 26791),  INT16_C( 23953),
         INT16_C(  4862), -INT16_C( 20983),  INT16_C( 21904), -INT16_C( 14162),  INT16_C( 28536),  INT16_C( 17099),  INT16_C( 21343), -INT16_C( 11179) },
      UINT32_C(  60552013) }
  };
  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512((void *)&test_vec[i].arr_a);
    easysimd__m512i b = easysimd_mm512_loadu_si512((void *)&test_vec[i].arr_b);
    easysimd__mmask32 r = test_vec[i].r;
    easysimd__mmask32 rk;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      rk = easysimd_mm512_cmpgt_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpgt_epi16_mask");
    easysimd_assert_equal_mmask32(rk, r);
  }

  return 0;

#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 r = easysimd_mm512_cmpgt_epi16_mask(a, b);

    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif

}

static int
test_easysimd_mm512_cmpgt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask16 r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( -126651070), INT32_C( 1757388710), INT32_C(  617530196), INT32_C(  407807901),
                            INT32_C( 1271989524), INT32_C( 1251214807), INT32_C(-1247045111), INT32_C(-1024057759),
                            INT32_C(   50729453), INT32_C(  464444874), INT32_C( 1840702207), INT32_C( 1916050591),
                            INT32_C(  484601458), INT32_C( -782065931), INT32_C(-1485735658), INT32_C(-1326388993)),
      easysimd_mm512_set_epi32(INT32_C(  111072774), INT32_C( 1757388710), INT32_C( 1496897687), INT32_C(  407807901),
                            INT32_C( 1271989524), INT32_C( 1496985365), INT32_C(-1247045111), INT32_C(-1225014979),
                            INT32_C(   50729453), INT32_C(  464444874), INT32_C(  924537351), INT32_C( 1916050591),
                            INT32_C(  484601458), INT32_C( -782065931), INT32_C(-1485735658), INT32_C(-1326388993)),
      UINT16_C(  288) },
    { easysimd_mm512_set_epi32(INT32_C( 2106044062), INT32_C( 1752498924), INT32_C(-1086695378), INT32_C(  627787891),
                            INT32_C(-1783053554), INT32_C(-1485517848), INT32_C( 1105114322), INT32_C(-1862707588),
                            INT32_C(  823946037), INT32_C(-2030244995), INT32_C( -219180660), INT32_C(  810910725),
                            INT32_C( -642105946), INT32_C(  760774613), INT32_C(  -62603432), INT32_C(-2064446807)),
      easysimd_mm512_set_epi32(INT32_C( 2106044062), INT32_C( 1752498924), INT32_C( -582421212), INT32_C( 1649238471),
                            INT32_C( 1446053889), INT32_C(-1485517848), INT32_C( 1105114322), INT32_C(-1862707588),
                            INT32_C( -846383385), INT32_C(-2030244995), INT32_C( -905258415), INT32_C(  810910725),
                            INT32_C(-1668595380), INT32_C( -760772652), INT32_C( 2145797270), INT32_C(   57887151)),
      UINT16_C(  172) },
    { easysimd_mm512_set_epi32(INT32_C(  948728954), INT32_C(  965445469), INT32_C( -298261731), INT32_C( 1889741023),
                            INT32_C(  101476677), INT32_C( -598834633), INT32_C( 1592735604), INT32_C(  428243294),
                            INT32_C(-2001034764), INT32_C( -639043872), INT32_C(  567427880), INT32_C(-1305749494),
                            INT32_C( -204185535), INT32_C( -550643286), INT32_C( -170363385), INT32_C( 1483518213)),
      easysimd_mm512_set_epi32(INT32_C(  948728954), INT32_C(  965445469), INT32_C(  364841947), INT32_C(-1221758106),
                            INT32_C(  101476677), INT32_C( -598834633), INT32_C( 1592735604), INT32_C(-1456245493),
                            INT32_C(-2001034764), INT32_C( -639043872), INT32_C(  567427880), INT32_C(-1305749494),
                            INT32_C( -204185535), INT32_C(  830345587), INT32_C( -170363385), INT32_C( -603563929)),
      UINT16_C( 4353) },
    { easysimd_mm512_set_epi32(INT32_C( -163413000), INT32_C( -831194762), INT32_C( -664019578), INT32_C( 2031024026),
                            INT32_C( 1912388774), INT32_C(  982200166), INT32_C(  596130243), INT32_C(  446035443),
                            INT32_C( 1373006598), INT32_C(-1540837035), INT32_C( 1581631435), INT32_C(-2083299381),
                            INT32_C( 1992847454), INT32_C(  448258110), INT32_C(  875345838), INT32_C( 1612926819)),
      easysimd_mm512_set_epi32(INT32_C(   81573630), INT32_C( -831194762), INT32_C( -664019578), INT32_C( 2031024026),
                            INT32_C( 1912388774), INT32_C(-2072470454), INT32_C(  596130243), INT32_C( 1961646011),
                            INT32_C( 1373006598), INT32_C( -197223193), INT32_C( 1581631435), INT32_C(-2083299381),
                            INT32_C( 1457480410), INT32_C( 1181119535), INT32_C( 1263228451), INT32_C(-1998542716)),
      UINT16_C( 1033) },
    { easysimd_mm512_set_epi32(INT32_C( 1436278246), INT32_C(   99684976), INT32_C( 1345577484), INT32_C(  166701508),
                            INT32_C( -780731111), INT32_C( -840749601), INT32_C( 1523342039), INT32_C( 1058674665),
                            INT32_C( -523908416), INT32_C( 1659465207), INT32_C(-1927062215), INT32_C(-1156760340),
                            INT32_C(  715569317), INT32_C(-1515814414), INT32_C( 1243253180), INT32_C( 2080215882)),
      easysimd_mm512_set_epi32(INT32_C(  432908742), INT32_C(   99684976), INT32_C(  -14330157), INT32_C(-1223154556),
                            INT32_C( -780731111), INT32_C(  696697372), INT32_C( 1523342039), INT32_C( 1058674665),
                            INT32_C( -523908416), INT32_C( 1659465207), INT32_C(-1927062215), INT32_C(-1156760340),
                            INT32_C( -171262349), INT32_C(-1515814414), INT32_C(-1234169573), INT32_C(-1847568101)),
      UINT16_C(45067) },
    { easysimd_mm512_set_epi32(INT32_C( 1399825551), INT32_C(-1064541474), INT32_C( 2112452992), INT32_C(  575137303),
                            INT32_C( -979898374), INT32_C(-1476679333), INT32_C( 1320423852), INT32_C( 1767893242),
                            INT32_C( -389599783), INT32_C(-1459729991), INT32_C(  995424065), INT32_C( -522129019),
                            INT32_C( -466751981), INT32_C( 1371238810), INT32_C( 1006677155), INT32_C( 1609037982)),
      easysimd_mm512_set_epi32(INT32_C( 1399825551), INT32_C(-1064541474), INT32_C( 2112452992), INT32_C(  134645750),
                            INT32_C(  500192289), INT32_C( 1600988950), INT32_C( 1320423852), INT32_C( 1198845893),
                            INT32_C( -389599783), INT32_C( 1504468794), INT32_C(  995424065), INT32_C(-2123865443),
                            INT32_C( -466751981), INT32_C(-1711282630), INT32_C( 1006677155), INT32_C( 1609037982)),
      UINT16_C( 4372) },
    { easysimd_mm512_set_epi32(INT32_C(-1862774816), INT32_C(   28374488), INT32_C(  250156705), INT32_C( -932694837),
                            INT32_C(-2079251566), INT32_C( -246439183), INT32_C( -875109534), INT32_C( 1740046060),
                            INT32_C( 1735819269), INT32_C( 1371885292), INT32_C( -914870851), INT32_C( -473073032),
                            INT32_C( -580976455), INT32_C( 1688786028), INT32_C(  637430498), INT32_C(-1740972685)),
      easysimd_mm512_set_epi32(INT32_C(-1862774816), INT32_C(   28374488), INT32_C(  580744870), INT32_C( -666445473),
                            INT32_C( -129274908), INT32_C( -928751425), INT32_C( -388443661), INT32_C( 1740046060),
                            INT32_C(-1909361652), INT32_C( 1371885292), INT32_C(  857928163), INT32_C(   37075976),
                            INT32_C( -580976455), INT32_C(-1545948444), INT32_C(  637430498), INT32_C(-1740972685)),
      UINT16_C( 1156) },
    { easysimd_mm512_set_epi32(INT32_C(-1890406982), INT32_C( 2110791016), INT32_C( 1083476771), INT32_C( -620691621),
                            INT32_C(  543588207), INT32_C( -227503647), INT32_C( -759273149), INT32_C(  775085710),
                            INT32_C( 1404885802), INT32_C(-1395233065), INT32_C(  832528180), INT32_C( 1065959566),
                            INT32_C(-2083201484), INT32_C(  937916550), INT32_C( -710457746), INT32_C( -246147415)),
      easysimd_mm512_set_epi32(INT32_C(   84669207), INT32_C(  470641840), INT32_C( 1083476771), INT32_C( -620691621),
                            INT32_C( 1099959895), INT32_C( -961354454), INT32_C(-1751384146), INT32_C(  775085710),
                            INT32_C( 1075765582), INT32_C(  834655006), INT32_C(  832528180), INT32_C( 1065959566),
                            INT32_C(  954342416), INT32_C(  937916550), INT32_C(-1946395018), INT32_C(  757651617)),
      UINT16_C(18050) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpgt_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_cmpgt_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_cmpgt_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask64 k;
    int8_t a[64];
    int8_t b[64];
    easysimd__mmask64 r;
  } test_vec[8] = {
    { UINT64_C( 1946247329555361586),
      {  INT8_C(  92), -INT8_C(  26),  INT8_C(  11),  INT8_C(  92),  INT8_C(  75),  INT8_C(  40), -INT8_C(  78), -INT8_C(  36),
         INT8_C(   2),  INT8_C(  87),  INT8_C(  41),  INT8_C(  45), -INT8_C(  58), -INT8_C(  26), -INT8_C(  69),  INT8_C(  76),
        -INT8_C(  93),  INT8_C( 103),  INT8_C(  60),  INT8_C(  74),  INT8_C(  44),  INT8_C(  27), -INT8_C(  72),  INT8_C(  94),
         INT8_C(  74), -INT8_C(  21), -INT8_C(  47), -INT8_C(  19),  INT8_C(  96), -INT8_C(  44),  INT8_C(   8), -INT8_C(  67),
        -INT8_C(  70),  INT8_C(  19),  INT8_C(  25),  INT8_C(   5),  INT8_C(  60), -INT8_C(  53), -INT8_C(  31),  INT8_C(  62),
         INT8_C(  34),  INT8_C(  11),  INT8_C( 108), -INT8_C(  24), -INT8_C(  15),  INT8_C(  39),  INT8_C(  52), -INT8_C( 108),
        -INT8_C( 113),  INT8_C( 112), -INT8_C(  34), -INT8_C(  69), -INT8_C( 117), -INT8_C( 106),  INT8_C(  25), -INT8_C(  42),
        -INT8_C( 127), -INT8_C(  22), -INT8_C(  61), -INT8_C(  31), -INT8_C(  66), -INT8_C(  53), -INT8_C(  98),  INT8_C( 121) },
      { -INT8_C(  34), -INT8_C(  73),  INT8_C( 126),  INT8_C(  26), -INT8_C( 126),  INT8_C(  96),  INT8_C(  89), -INT8_C(  92),
         INT8_C( 107), -INT8_C(  59), -INT8_C( 116),  INT8_C(  92), -INT8_C(  20), -INT8_C(  64), -INT8_C(  16),  INT8_C( 123),
         INT8_C(  48), -INT8_C(  50),  INT8_C(  54), -INT8_C(  68),  INT8_C( 100),  INT8_C(  79), -INT8_C( 110), -INT8_C(  27),
         INT8_C(  58),  INT8_C(  85), -INT8_C(  58), -INT8_C(   8),  INT8_C(  32),  INT8_C( 101),  INT8_C( 113), -INT8_C(   2),
         INT8_C(  28), -INT8_C(  16),  INT8_C(  25), -INT8_C(  97),  INT8_C(  80),  INT8_C( 114),  INT8_C(  67), -INT8_C(  69),
         INT8_C(  55), -INT8_C(  48),  INT8_C(  23),  INT8_C(  35), -INT8_C( 112),  INT8_C(   7), -INT8_C(  97), -INT8_C(  63),
        -INT8_C(  43), -INT8_C(  43),  INT8_C( 125),  INT8_C(  57),  INT8_C(  37),  INT8_C(  15),  INT8_C(  30),  INT8_C(  95),
         INT8_C( 100), -INT8_C(  28),  INT8_C(  87), -INT8_C( 124),  INT8_C(  73), -INT8_C(  55), -INT8_C( 126),  INT8_C( 102) },
      UINT64_C(  721266992312624658) },
    { UINT64_C( 4955164710837525433),
      {  INT8_C(  24), -INT8_C(  37),  INT8_C( 104), -INT8_C(  87), -INT8_C(  30),  INT8_C(   7),  INT8_C( 106), -INT8_C(  73),
        -INT8_C(  36), -INT8_C(  25), -INT8_C(  16),  INT8_C(   1), -INT8_C(  10),  INT8_C(  14),  INT8_C(  96),  INT8_C(  90),
        -INT8_C(  14), -INT8_C(  72), -INT8_C(  34),  INT8_C(  60), -INT8_C( 127),  INT8_C(  96), -INT8_C(  94),  INT8_C(  58),
        -INT8_C(   4), -INT8_C(  89),  INT8_C(  67),  INT8_C(   9), -INT8_C(  17),  INT8_C(   7),  INT8_C(  78),  INT8_C(   8),
        -INT8_C(  30), -INT8_C(  74), -INT8_C(  79), -INT8_C(  60), -INT8_C(  67),  INT8_C(  27),  INT8_C( 123), -INT8_C( 103),
         INT8_C(   2),  INT8_C( 107), -INT8_C( 101), -INT8_C(   8),  INT8_C( 121), -INT8_C(   5),  INT8_C(  82),  INT8_C( 107),
        -INT8_C(  77),  INT8_C(  48), -INT8_C(  89),  INT8_C(  52), -INT8_C( 112),  INT8_C(  73),  INT8_C( 110), -INT8_C( 116),
        -INT8_C(  16), -INT8_C(  79), -INT8_C( 106), -INT8_C(  32), -INT8_C(  72), -INT8_C(  28), -INT8_C(  24), -INT8_C( 102) },
      { -INT8_C( 102), -INT8_C( 103),  INT8_C(  94),  INT8_C(  87), -INT8_C(  76), -INT8_C(  39), -INT8_C(  16), -INT8_C(  74),
         INT8_C(  68), -INT8_C( 117), -INT8_C(  82), -INT8_C(  67), -INT8_C( 121),  INT8_C(   0),  INT8_C(  41),  INT8_C(  58),
         INT8_C(  48), -INT8_C(  48),  INT8_C( 111), -INT8_C(  64),  INT8_C(  26), -INT8_C(  35),  INT8_C(  77),  INT8_C(  10),
        -INT8_C( 113), -INT8_C(  29), -INT8_C(  22),  INT8_C(  71), -INT8_C(  57), -INT8_C(  46), -INT8_C(  30),  INT8_C(  97),
         INT8_C( 107),  INT8_C(  64), -INT8_C(  72),  INT8_C(  31),  INT8_C(  26), -INT8_C(  88), -INT8_C(  43),  INT8_C(  94),
         INT8_C(  52), -INT8_C( 125),  INT8_C(  28), -INT8_C(  69), -INT8_C( 125),  INT8_C(  69), -INT8_C(  11), -INT8_C(  77),
         INT8_C(  21),  INT8_C( 100),  INT8_C( 116),  INT8_C(  47),  INT8_C(  66), -INT8_C(  63),  INT8_C(  58), -INT8_C(  47),
        -INT8_C(  92),  INT8_C(  36),  INT8_C(  24),  INT8_C( 107), -INT8_C(   9), -INT8_C(   6), -INT8_C(  52),  INT8_C(  98) },
      UINT64_C( 4629779581790886577) },
    { UINT64_C( 6968008896397083707),
      { -INT8_C(  37), -INT8_C(  49),  INT8_C(  27),  INT8_C(  94),  INT8_C(  20),  INT8_C(  17),  INT8_C(  18),  INT8_C(  42),
         INT8_C( 117), -INT8_C( 122),  INT8_C(  89), -INT8_C(  73),  INT8_C(  71), -INT8_C( 109), -INT8_C( 120), -INT8_C(  21),
        -INT8_C(  72), -INT8_C(  95),  INT8_C(  86), -INT8_C(  81), -INT8_C( 101),  INT8_C(  34),  INT8_C(  17), -INT8_C(  42),
        -INT8_C(  90), -INT8_C( 109),  INT8_C(  43), -INT8_C(  46), -INT8_C(  21), -INT8_C(  33),  INT8_C(  51), -INT8_C(  58),
        -INT8_C(  82),  INT8_C(  78),  INT8_C(  36), -INT8_C(  61),  INT8_C(  95),  INT8_C(  54), -INT8_C(  19), -INT8_C(  43),
        -INT8_C(  68),  INT8_C(  70), -INT8_C( 116),  INT8_C(   3), -INT8_C(  38),  INT8_C(  21), -INT8_C(  18), -INT8_C( 110),
        -INT8_C(  74),  INT8_C(  68),  INT8_C(  65),  INT8_C(  81),  INT8_C( 102),  INT8_C(  82),  INT8_C(  40),  INT8_C(  12),
        -INT8_C(  26),  INT8_C(  83), -INT8_C(  33), -INT8_C(  47),  INT8_C(  50),  INT8_C(  18), -INT8_C( 105), -INT8_C(  31) },
      {  INT8_C(  96), -INT8_C(  69), -INT8_C(  92), -INT8_C(  64), -INT8_C(  14), -INT8_C( 111), -INT8_C( 107), -INT8_C(  82),
        -INT8_C(  41),  INT8_C(  33), -INT8_C(  78), -INT8_C(  79),  INT8_C(  54), -INT8_C(  96),  INT8_C(  67), -INT8_C(  20),
        -INT8_C(  27), -INT8_C( 124),  INT8_C(  62),  INT8_C(  75), -INT8_C(  41),  INT8_C( 102),  INT8_C(  88), -INT8_C(  67),
        -INT8_C(  71),  INT8_C(  55), -INT8_C( 114), -INT8_C(  20),  INT8_C(  73),  INT8_C(  37), -INT8_C(  51), -INT8_C(  87),
        -INT8_C(  32),  INT8_C( 113),  INT8_C( 105), -INT8_C(  46),  INT8_C(   2), -INT8_C(   2), -INT8_C( 127), -INT8_C(  39),
         INT8_C(  32),  INT8_C(  51), -INT8_C( 117),  INT8_C(  86), -INT8_C(  45), -INT8_C(  50),  INT8_C(  67), -INT8_C(  72),
         INT8_C(  83), -INT8_C( 127),  INT8_C(   4),  INT8_C(  42), -INT8_C(  25),  INT8_C(  92), -INT8_C(  25), -INT8_C(  96),
        -INT8_C( 109),  INT8_C( 117), -INT8_C( 116), -INT8_C(  36), -INT8_C( 102),  INT8_C(  89), -INT8_C( 123),  INT8_C( 122) },
      UINT64_C( 4652805692871279674) },
    { UINT64_C(  983701089357262794),
      {  INT8_C(   1),  INT8_C(  49),  INT8_C( 100), -INT8_C(  44), -INT8_C(   1), -INT8_C(  89), -INT8_C( 115),  INT8_C(  82),
         INT8_C(  40), -INT8_C( 111),  INT8_C( 124),  INT8_C(  15), -INT8_C(  19),  INT8_C(  99), -INT8_C(  81),      INT8_MIN,
        -INT8_C(  40),  INT8_C(  60),  INT8_C(  92),  INT8_C( 114), -INT8_C( 107), -INT8_C(  31), -INT8_C(  19),  INT8_C(  96),
        -INT8_C(  48),  INT8_C(  58),  INT8_C(  44), -INT8_C(  66),  INT8_C(   8), -INT8_C(  46), -INT8_C(  53),  INT8_C(   9),
         INT8_C(   3),  INT8_C(  47), -INT8_C(  35),  INT8_C(   3), -INT8_C(  42),  INT8_C( 106),  INT8_C(  85), -INT8_C(   2),
        -INT8_C(   5), -INT8_C(  46),  INT8_C(  13), -INT8_C(  24),  INT8_C(  53), -INT8_C(  67),  INT8_C( 104),  INT8_C(  14),
        -INT8_C(   7), -INT8_C(  60),      INT8_MIN, -INT8_C( 114), -INT8_C(  90),  INT8_C( 109), -INT8_C(  18),  INT8_C( 118),
        -INT8_C(  89),  INT8_C(  27),  INT8_C(  52), -INT8_C(  81), -INT8_C(  19),  INT8_C(   0), -INT8_C(  72), -INT8_C(  15) },
      {  INT8_C(  47), -INT8_C( 106), -INT8_C(  12),  INT8_C(   6),  INT8_C(   0),  INT8_C(  73),  INT8_C(   4), -INT8_C(   4),
         INT8_C(  27),  INT8_C(  18), -INT8_C(  28),  INT8_C(  81), -INT8_C(  49),  INT8_C(  77),  INT8_C(  95), -INT8_C(  56),
         INT8_C(  17), -INT8_C(  33),  INT8_C(  86), -INT8_C(  73),  INT8_C(  77),  INT8_C(  69),  INT8_C(  46), -INT8_C(  12),
         INT8_C(  96),  INT8_C(  98), -INT8_C(  92),  INT8_C(  77),  INT8_C(  98),  INT8_C(  92),  INT8_C(  62), -INT8_C( 110),
        -INT8_C(  14),  INT8_C(  50), -INT8_C( 104), -INT8_C(  13),  INT8_C( 124), -INT8_C( 100), -INT8_C(  17), -INT8_C( 105),
        -INT8_C(  82), -INT8_C(  45), -INT8_C(  24),  INT8_C( 125),  INT8_C(  32),  INT8_C(  71),  INT8_C(  69),  INT8_C(  50),
         INT8_C(  39), -INT8_C( 100), -INT8_C(  23),  INT8_C( 116), -INT8_C(  31),  INT8_C(  23),  INT8_C( 104),  INT8_C(  65),
         INT8_C( 122),  INT8_C(  12), -INT8_C( 114), -INT8_C(  36),  INT8_C( 105), -INT8_C(  51),  INT8_C( 110),  INT8_C(  91) },
      UINT64_C(  333905109292164482) },
    { UINT64_C( 5842080913987340031),
      {  INT8_C(  17), -INT8_C(   5), -INT8_C(  49),  INT8_C(  49),  INT8_C(  67),  INT8_C(  20),  INT8_C(  99),  INT8_C( 106),
        -INT8_C(  80),  INT8_C(  77), -INT8_C(  34), -INT8_C( 111),  INT8_C( 100),  INT8_C(  70), -INT8_C(  46), -INT8_C(  34),
         INT8_C(  83),  INT8_C(  97), -INT8_C(  69), -INT8_C(  68),  INT8_C(  46),  INT8_C(  41),  INT8_C(  23),  INT8_C(  45),
         INT8_C(  48),  INT8_C( 102), -INT8_C(  87), -INT8_C(  45), -INT8_C(  93), -INT8_C(  68),  INT8_C(  36), -INT8_C(  76),
        -INT8_C(  73), -INT8_C(  13), -INT8_C(  26), -INT8_C(   6),  INT8_C(   8),  INT8_C(  73),  INT8_C( 100), -INT8_C(  72),
        -INT8_C( 106),  INT8_C(  66),  INT8_C(  74), -INT8_C(   5), -INT8_C( 119),  INT8_C(  28), -INT8_C(  39), -INT8_C(  36),
         INT8_C( 125), -INT8_C( 108), -INT8_C( 104), -INT8_C(  85), -INT8_C(  66), -INT8_C(  81), -INT8_C(  39), -INT8_C(  18),
         INT8_C(  21), -INT8_C( 126), -INT8_C(  63), -INT8_C(  71),  INT8_C(  62), -INT8_C(  27),  INT8_C( 109), -INT8_C(  11) },
      { -INT8_C(  39),  INT8_C(  83), -INT8_C(  16), -INT8_C(  31), -INT8_C(  99),  INT8_C(  84), -INT8_C( 103),  INT8_C(  51),
        -INT8_C( 105), -INT8_C(  29),  INT8_C(  46),  INT8_C(  32),  INT8_C(   0),  INT8_C(   8), -INT8_C(   4),  INT8_C( 125),
        -INT8_C( 100), -INT8_C( 108),  INT8_C(  41),  INT8_C(  90),  INT8_C(  67),  INT8_C(   2),  INT8_C(  72),  INT8_C(  89),
        -INT8_C( 124),  INT8_C(   9),  INT8_C(  18), -INT8_C(  62), -INT8_C(  17),      INT8_MAX, -INT8_C(  73), -INT8_C(  56),
        -INT8_C(  45), -INT8_C(  89), -INT8_C(  87),  INT8_C( 112), -INT8_C(   4),  INT8_C(  66), -INT8_C(  93), -INT8_C( 109),
         INT8_C(  38), -INT8_C(  46), -INT8_C(  77),  INT8_C(  38), -INT8_C(  38), -INT8_C(  81), -INT8_C(  93),  INT8_C( 118),
         INT8_C(  67), -INT8_C(  52), -INT8_C(  47), -INT8_C( 122), -INT8_C(  50),  INT8_C(  25), -INT8_C(  33),  INT8_C(  82),
         INT8_C(  35), -INT8_C(  15),  INT8_C(  20),  INT8_C(  18),  INT8_C( 113), -INT8_C(  52), -INT8_C(  38),  INT8_C(  68) },
      UINT64_C( 4612007772865823449) },
    { UINT64_C(16934194054360761203),
      {  INT8_C(  41), -INT8_C(  75),  INT8_C(  17),  INT8_C(   3),  INT8_C( 100), -INT8_C(  75),  INT8_C( 122), -INT8_C(  89),
        -INT8_C( 127),  INT8_C(  75),  INT8_C(  46),  INT8_C(  80),  INT8_C( 100),  INT8_C(  13), -INT8_C(  94), -INT8_C( 121),
        -INT8_C(   1), -INT8_C(  73), -INT8_C( 103),  INT8_C( 112), -INT8_C( 125),  INT8_C( 115), -INT8_C(  76), -INT8_C(  10),
        -INT8_C(  10),  INT8_C( 104),  INT8_C( 102), -INT8_C(  68), -INT8_C(  65),  INT8_C( 104), -INT8_C(  89), -INT8_C(  23),
         INT8_C(  30), -INT8_C(  71), -INT8_C(  20), -INT8_C( 126),  INT8_C( 110),  INT8_C( 102),  INT8_C(  42), -INT8_C(  17),
        -INT8_C(  79),  INT8_C(  88),  INT8_C(  63),  INT8_C(  22),  INT8_C( 101), -INT8_C(  30), -INT8_C(  99),  INT8_C( 100),
        -INT8_C( 103),  INT8_C(  55), -INT8_C(  44),  INT8_C(  28), -INT8_C(  86), -INT8_C( 120),  INT8_C(  18), -INT8_C(  95),
        -INT8_C(  16),  INT8_C( 120),  INT8_C(  93), -INT8_C(  80), -INT8_C(  31),  INT8_C(   4), -INT8_C( 103), -INT8_C(   1) },
      { -INT8_C(  67), -INT8_C( 123), -INT8_C( 127),  INT8_C(  43), -INT8_C(  20), -INT8_C(  85),  INT8_C(  27), -INT8_C(  99),
         INT8_C(   3),  INT8_C(  90), -INT8_C(  77),  INT8_C( 105),  INT8_C(  60),  INT8_C(  81), -INT8_C(  51), -INT8_C(  43),
        -INT8_C( 120), -INT8_C(  94), -INT8_C(  15),  INT8_C(  50),  INT8_C(  42),  INT8_C(   4), -INT8_C(  45),  INT8_C(  27),
         INT8_C( 124),  INT8_C(  48), -INT8_C(  53),  INT8_C(  93),  INT8_C(  53),  INT8_C( 100),  INT8_C(  92), -INT8_C(  14),
        -INT8_C(  23), -INT8_C(  34),  INT8_C(  30), -INT8_C(  43), -INT8_C( 119),  INT8_C(  57),  INT8_C( 115), -INT8_C( 115),
        -INT8_C( 109),  INT8_C(  38), -INT8_C(  10), -INT8_C(  48),  INT8_C( 119), -INT8_C(  61), -INT8_C(  91), -INT8_C(   1),
         INT8_C( 101), -INT8_C( 105),  INT8_C(  50), -INT8_C( 112), -INT8_C( 101),  INT8_C(   5), -INT8_C(  85),  INT8_C(  23),
         INT8_C(  54),  INT8_C( 118),  INT8_C( 117),  INT8_C( 107), -INT8_C(  38), -INT8_C(  47),  INT8_C(  93), -INT8_C(  61) },
      UINT64_C(11673901435369554035) },
    { UINT64_C( 5243892784319527855),
      {  INT8_C(  50), -INT8_C(  68),  INT8_C(  24), -INT8_C(  86),      INT8_MAX, -INT8_C(  67), -INT8_C(  87), -INT8_C(  27),
         INT8_C(  84), -INT8_C(  37),  INT8_C( 117), -INT8_C(  17), -INT8_C(  31),  INT8_C(  32),  INT8_C(   7),  INT8_C(  23),
        -INT8_C( 106),  INT8_C( 124), -INT8_C( 126),  INT8_C( 112),  INT8_C(  77), -INT8_C(  33),  INT8_C(  51), -INT8_C(   3),
         INT8_C(  91), -INT8_C(  52),  INT8_C(  54),  INT8_C(  15), -INT8_C(  40), -INT8_C(   4),  INT8_C(  87),  INT8_C(  11),
        -INT8_C(  72),  INT8_C( 111), -INT8_C(  75),  INT8_C(  55),  INT8_C(  45),  INT8_C(  94),  INT8_C(  28), -INT8_C( 127),
         INT8_C(  58), -INT8_C( 111),  INT8_C( 113),  INT8_C(  27), -INT8_C(  79),  INT8_C( 120),  INT8_C(  50),  INT8_C(  71),
        -INT8_C(  12), -INT8_C(  76), -INT8_C(  73),  INT8_C(  65), -INT8_C( 109), -INT8_C(  21),  INT8_C(  62), -INT8_C(  18),
        -INT8_C(  73),  INT8_C( 116), -INT8_C(   2), -INT8_C( 112),  INT8_C( 112),  INT8_C(  85), -INT8_C( 101),  INT8_C(  40) },
      { -INT8_C(  59),  INT8_C(  80),  INT8_C(  96), -INT8_C(  14), -INT8_C(  82),  INT8_C( 124),  INT8_C( 115), -INT8_C(  24),
         INT8_C(  14), -INT8_C(  28),  INT8_C(   3), -INT8_C(  65),  INT8_C(  92),  INT8_C(  53),  INT8_C(   7),  INT8_C(  80),
        -INT8_C(  23), -INT8_C(  66), -INT8_C( 110),  INT8_C( 125), -INT8_C(  87), -INT8_C(  48),  INT8_C( 107),  INT8_C(  97),
         INT8_C(  69),  INT8_C( 105), -INT8_C(  15), -INT8_C(  75), -INT8_C(  65), -INT8_C( 116), -INT8_C(  34), -INT8_C( 124),
        -INT8_C(  36),  INT8_C(  62),  INT8_C( 118), -INT8_C( 118), -INT8_C(  70), -INT8_C(  23),  INT8_C( 115), -INT8_C(  56),
        -INT8_C(  50),  INT8_C( 118), -INT8_C( 120),  INT8_C(  42), -INT8_C(  84), -INT8_C( 113),  INT8_C( 123), -INT8_C( 107),
         INT8_C(  77),  INT8_C(  13),  INT8_C(  18), -INT8_C(   9), -INT8_C(  35),  INT8_C( 126),  INT8_C(  88),  INT8_C(  34),
        -INT8_C(  25),  INT8_C(  73), -INT8_C(  40), -INT8_C(  90), -INT8_C(  43), -INT8_C(  74),  INT8_C(  42), -INT8_C(  79) },
      UINT64_C(       4605162293505) },
    { UINT64_C( 6374755708218089716),
      {  INT8_C(  37), -INT8_C(   1), -INT8_C( 126), -INT8_C(  47), -INT8_C( 114), -INT8_C(   3),  INT8_C( 102), -INT8_C(  37),
         INT8_C(  10),  INT8_C( 121), -INT8_C(  46), -INT8_C(  24), -INT8_C(   9),  INT8_C(  42),  INT8_C(  10), -INT8_C(  34),
         INT8_C( 115), -INT8_C(  30), -INT8_C( 123),  INT8_C(  72), -INT8_C( 104), -INT8_C(  81), -INT8_C(   7), -INT8_C( 116),
         INT8_C(  80),  INT8_C(  53),  INT8_C(  59), -INT8_C(  38), -INT8_C(  29), -INT8_C(  78),  INT8_C(  50),  INT8_C(   8),
        -INT8_C(  79), -INT8_C(  76), -INT8_C(  39),  INT8_C(  63), -INT8_C(  78),  INT8_C(  64),  INT8_C(  26), -INT8_C(  68),
        -INT8_C(  71), -INT8_C(  19), -INT8_C(  92), -INT8_C(  80),  INT8_C(  23), -INT8_C(  81), -INT8_C( 114), -INT8_C( 117),
        -INT8_C( 111),  INT8_C(  19), -INT8_C(  45),  INT8_C(  42), -INT8_C(  61), -INT8_C(  51), -INT8_C(  74),  INT8_C(  19),
         INT8_C(   2), -INT8_C(  15), -INT8_C(  19), -INT8_C(  27), -INT8_C(  93),  INT8_C(  31), -INT8_C(  18),  INT8_C(  84) },
      { -INT8_C(  45), -INT8_C(  57), -INT8_C( 109), -INT8_C( 123),  INT8_C(   7), -INT8_C(  82),  INT8_C(  66), -INT8_C(  64),
        -INT8_C( 101), -INT8_C(  26),  INT8_C( 112), -INT8_C(  78), -INT8_C( 107), -INT8_C(   1),  INT8_C(  61),  INT8_C(  39),
         INT8_C(  18),  INT8_C(  17),  INT8_C(  81), -INT8_C(  43), -INT8_C(  34),  INT8_C(   7), -INT8_C(  24), -INT8_C(  32),
        -INT8_C(   7), -INT8_C(  43), -INT8_C(  59), -INT8_C( 100), -INT8_C(  12), -INT8_C(  77), -INT8_C(  15), -INT8_C(  56),
         INT8_C( 123), -INT8_C( 124),  INT8_C(  77), -INT8_C( 126),  INT8_C(  50), -INT8_C( 113),  INT8_C(  67), -INT8_C(  51),
         INT8_C( 118), -INT8_C(  77),      INT8_MIN,  INT8_C(  11), -INT8_C(  78), -INT8_C(  67),  INT8_C(  50), -INT8_C(  59),
        -INT8_C(  50), -INT8_C( 125), -INT8_C( 102), -INT8_C(  84), -INT8_C( 117), -INT8_C( 125), -INT8_C( 116), -INT8_C( 124),
         INT8_C(  88),  INT8_C(  82),  INT8_C(  32),  INT8_C(  77),  INT8_C(   5),  INT8_C(  17),  INT8_C(  21),      INT8_MIN },
      UINT64_C(   33220689654259936) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epi8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i b = easysimd_test_x86_random_i8x64();
    easysimd__mmask64 r = easysimd_mm512_mask_cmpgt_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask64(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpgt_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    int16_t a[32];
    int16_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C( 445146884),
      {  INT16_C( 15910),  INT16_C( 32341), -INT16_C(  7581), -INT16_C(  2966), -INT16_C( 31899), -INT16_C( 18062),  INT16_C( 32246), -INT16_C( 26487),
        -INT16_C( 19952),  INT16_C( 31933),  INT16_C( 26579), -INT16_C( 12616), -INT16_C( 21684),  INT16_C( 20583), -INT16_C(  4333),  INT16_C( 14698),
        -INT16_C( 16595), -INT16_C( 28489),  INT16_C(  8609),  INT16_C(  1668), -INT16_C(  2396), -INT16_C( 25664),  INT16_C( 18803), -INT16_C( 31949),
        -INT16_C(  3845), -INT16_C( 12289), -INT16_C( 18601), -INT16_C( 23651),  INT16_C(  1123),  INT16_C( 30451),  INT16_C( 24051),  INT16_C(  8367) },
      {  INT16_C( 26396), -INT16_C( 16720),  INT16_C( 13448),  INT16_C( 11716), -INT16_C( 31701), -INT16_C( 24888), -INT16_C(  1074), -INT16_C( 14046),
         INT16_C(  8683),  INT16_C( 17304),  INT16_C( 13785),  INT16_C( 15590), -INT16_C(  9671),  INT16_C( 11442),  INT16_C( 24887),  INT16_C( 21580),
        -INT16_C(   824),  INT16_C( 20754), -INT16_C( 10703),  INT16_C( 23678),  INT16_C( 18011),  INT16_C( 10746),  INT16_C(  7233),  INT16_C( 11506),
        -INT16_C( 29890),  INT16_C(  5999),  INT16_C( 22208), -INT16_C(  1453),  INT16_C(  1328),  INT16_C( 26406),  INT16_C( 29542),  INT16_C( 12219) },
      UINT32_C(      9728) },
    { UINT32_C(2692795759),
      { -INT16_C(   348), -INT16_C(     4), -INT16_C(  2236), -INT16_C( 31448),  INT16_C(  6675),  INT16_C( 20913),  INT16_C(  8613),  INT16_C( 26216),
        -INT16_C( 17545), -INT16_C( 22688), -INT16_C( 31040),  INT16_C(  9998), -INT16_C( 13575),  INT16_C( 26966), -INT16_C( 10601),  INT16_C( 15113),
         INT16_C(  1748),  INT16_C(  6202),  INT16_C( 25341),  INT16_C(  4253),  INT16_C( 20093),  INT16_C(  8802), -INT16_C( 13713), -INT16_C(  6520),
        -INT16_C(  6010),  INT16_C( 18061), -INT16_C( 25489),  INT16_C( 26733), -INT16_C( 15514), -INT16_C(   559), -INT16_C(  9319),  INT16_C( 27961) },
      {  INT16_C( 29665), -INT16_C(  8571),  INT16_C(  8918),  INT16_C( 21486),  INT16_C( 20593), -INT16_C(  8075), -INT16_C(   485), -INT16_C( 24121),
         INT16_C( 21734),  INT16_C( 21991),  INT16_C( 22000),  INT16_C( 22206), -INT16_C( 28904), -INT16_C( 19884), -INT16_C( 29334),  INT16_C( 19231),
        -INT16_C( 23296), -INT16_C( 10711),  INT16_C(  6343),  INT16_C( 14377), -INT16_C( 24728), -INT16_C( 31975), -INT16_C(  8035), -INT16_C( 31964),
         INT16_C(  3124),  INT16_C(  9689), -INT16_C( 26783),  INT16_C( 31099), -INT16_C( 12506), -INT16_C( 28373),  INT16_C( 19292),  INT16_C( 24028) },
      UINT32_C(2692759650) },
    { UINT32_C(3073574640),
      {  INT16_C( 23838), -INT16_C( 30992),  INT16_C(  2556), -INT16_C( 26358),  INT16_C( 12009),  INT16_C(  7452), -INT16_C(  2758), -INT16_C( 25790),
        -INT16_C( 16756), -INT16_C( 19691),  INT16_C( 16525), -INT16_C(  5564),  INT16_C(  8331),  INT16_C( 31559),  INT16_C( 31270),  INT16_C( 17459),
         INT16_C(  9175), -INT16_C( 11317), -INT16_C( 10964),  INT16_C(  5484), -INT16_C( 30461),  INT16_C( 15922),  INT16_C( 30078),  INT16_C(  3033),
        -INT16_C(  4557), -INT16_C( 16194),  INT16_C(   559), -INT16_C( 17750), -INT16_C(  3806),  INT16_C( 18742),  INT16_C( 26988),  INT16_C( 17293) },
      {  INT16_C( 22668), -INT16_C( 18409), -INT16_C( 31955),  INT16_C( 12749), -INT16_C(   244), -INT16_C( 29841),  INT16_C( 18548), -INT16_C( 22634),
         INT16_C( 21559),  INT16_C( 26216),  INT16_C(  4694),  INT16_C( 30752),  INT16_C( 22020),  INT16_C( 28865),  INT16_C( 20415),  INT16_C( 19379),
        -INT16_C( 13657), -INT16_C( 11005), -INT16_C( 12210),  INT16_C( 23046),  INT16_C( 30160),  INT16_C( 17637),  INT16_C( 31677), -INT16_C(  2836),
         INT16_C( 21711),  INT16_C(  9562),  INT16_C( 31590),  INT16_C( 27294),  INT16_C( 24529), -INT16_C( 28198), -INT16_C( 29010),  INT16_C( 22236) },
      UINT32_C( 536937520) },
    { UINT32_C(2787893336),
      {  INT16_C( 12720), -INT16_C( 32767), -INT16_C(  6490),  INT16_C( 25541), -INT16_C( 20126),  INT16_C( 12632), -INT16_C( 19963),  INT16_C( 27479),
        -INT16_C(  2771), -INT16_C(    42), -INT16_C( 20396),  INT16_C(   912),  INT16_C( 27710), -INT16_C( 26791), -INT16_C( 31668), -INT16_C(   707),
         INT16_C( 16053),  INT16_C( 23421),  INT16_C( 16933), -INT16_C( 30786),  INT16_C(  5875), -INT16_C(  1864),  INT16_C(  4041), -INT16_C(  2460),
         INT16_C( 14852),  INT16_C( 23029), -INT16_C( 31254),  INT16_C( 10588), -INT16_C( 18958),  INT16_C( 16064), -INT16_C(   711), -INT16_C(  4549) },
      { -INT16_C( 18116),  INT16_C( 24905),  INT16_C(  2043), -INT16_C(  4120), -INT16_C( 24546), -INT16_C(  6169),  INT16_C( 19376), -INT16_C( 19235),
        -INT16_C( 11387),  INT16_C( 28685),  INT16_C( 26968),  INT16_C( 19097),  INT16_C( 22814),  INT16_C( 22409), -INT16_C( 15274), -INT16_C( 28091),
        -INT16_C( 29059),  INT16_C( 31219), -INT16_C(  9322), -INT16_C( 19352),  INT16_C( 20348),  INT16_C( 11419),  INT16_C( 30875),  INT16_C(  8416),
        -INT16_C(  4533), -INT16_C( 23408),  INT16_C( 10583),  INT16_C( 30446),  INT16_C( 30594), -INT16_C(  9779),  INT16_C(  4924), -INT16_C( 18069) },
      UINT32_C(2718007320) },
    { UINT32_C( 926048161),
      { -INT16_C( 26054), -INT16_C( 18709), -INT16_C( 30998), -INT16_C( 31262), -INT16_C( 15361),  INT16_C( 19109),  INT16_C( 14001),  INT16_C(  2286),
        -INT16_C(  8865), -INT16_C(  7554),  INT16_C( 19540), -INT16_C( 28485),  INT16_C(  9823),  INT16_C(    74),  INT16_C( 31877), -INT16_C( 16328),
         INT16_C(  8983),  INT16_C(   374),  INT16_C( 22954), -INT16_C( 22138),  INT16_C( 11036), -INT16_C( 12813), -INT16_C(  7583), -INT16_C( 15915),
         INT16_C( 21695),  INT16_C(  5027),  INT16_C( 24224), -INT16_C(    92), -INT16_C(  4476),  INT16_C(  2815),  INT16_C( 14186), -INT16_C( 32310) },
      {  INT16_C( 16475),  INT16_C(  1410),  INT16_C(  2201), -INT16_C( 19026), -INT16_C( 24268), -INT16_C( 27262),  INT16_C( 22659),  INT16_C( 16982),
        -INT16_C(  1620),  INT16_C( 19542), -INT16_C(  1449), -INT16_C(  9141),  INT16_C( 19176),  INT16_C( 21222), -INT16_C( 20350), -INT16_C(  8748),
         INT16_C( 22256), -INT16_C( 29982), -INT16_C( 28577), -INT16_C( 27841), -INT16_C( 15823), -INT16_C( 19160),  INT16_C( 32538), -INT16_C( 14601),
         INT16_C( 19832), -INT16_C( 12270),  INT16_C( 23879),  INT16_C( 12204), -INT16_C( 27993),  INT16_C( 10626),  INT16_C( 22082),  INT16_C( 12806) },
      UINT32_C( 389170208) },
    { UINT32_C( 196929708),
      { -INT16_C(   904), -INT16_C( 21858), -INT16_C( 14402), -INT16_C( 10145),  INT16_C( 22086), -INT16_C( 16738), -INT16_C( 20316), -INT16_C(  5234),
         INT16_C( 14861), -INT16_C( 19429), -INT16_C( 25140),  INT16_C(  3806), -INT16_C(  6925), -INT16_C( 24767), -INT16_C(   563),  INT16_C( 17835),
         INT16_C( 18937), -INT16_C( 18449),  INT16_C( 19984),  INT16_C( 22159),  INT16_C( 11685),  INT16_C( 18709), -INT16_C( 23587), -INT16_C(  5580),
         INT16_C( 20446), -INT16_C( 21857),  INT16_C( 32236), -INT16_C(  8263), -INT16_C(  1439),  INT16_C( 11903),  INT16_C( 10999), -INT16_C(  3724) },
      {  INT16_C( 25459), -INT16_C( 31576),  INT16_C( 14514),  INT16_C( 22490), -INT16_C(  4251),  INT16_C( 17312), -INT16_C( 11117),  INT16_C( 28973),
        -INT16_C( 13276),  INT16_C(  4123), -INT16_C( 11191), -INT16_C( 21520),  INT16_C( 28622), -INT16_C( 14631),  INT16_C( 19865),  INT16_C(  3255),
         INT16_C( 24497),  INT16_C( 25488),  INT16_C( 27543), -INT16_C(   582),  INT16_C( 23130), -INT16_C(  4800),  INT16_C( 27950),  INT16_C( 21086),
         INT16_C( 31290), -INT16_C( 31901),  INT16_C( 21326),  INT16_C(  7470),  INT16_C(  2242),  INT16_C( 23523), -INT16_C( 26027),  INT16_C(  1639) },
      UINT32_C(  36210688) },
    { UINT32_C(2439641337),
      {  INT16_C(  9059), -INT16_C( 17010), -INT16_C( 12675), -INT16_C( 21333),  INT16_C(  2363),  INT16_C( 30206),  INT16_C( 24963), -INT16_C( 11527),
         INT16_C( 10164),  INT16_C( 30447), -INT16_C( 11729), -INT16_C( 31279),  INT16_C( 14700),  INT16_C( 25995), -INT16_C(  2767), -INT16_C( 27402),
        -INT16_C( 31720), -INT16_C( 27055), -INT16_C(   942), -INT16_C( 29118),  INT16_C( 16390), -INT16_C( 30461), -INT16_C(   862),  INT16_C( 22107),
         INT16_C( 18980),  INT16_C( 21453), -INT16_C( 25060), -INT16_C( 30504),  INT16_C( 25815),  INT16_C(  2286), -INT16_C(  7079),  INT16_C( 29084) },
      { -INT16_C(  4503), -INT16_C( 17657),  INT16_C( 18922), -INT16_C(  4023),  INT16_C( 19850),  INT16_C( 11386), -INT16_C( 10935),  INT16_C( 28034),
         INT16_C( 20256),  INT16_C( 15553), -INT16_C( 26130), -INT16_C( 14907), -INT16_C( 19459),  INT16_C( 22222),  INT16_C( 27287),  INT16_C(   200),
        -INT16_C( 12456),  INT16_C( 17340),  INT16_C(  1305), -INT16_C( 23757), -INT16_C( 21166), -INT16_C( 25393),  INT16_C( 20867), -INT16_C( 23799),
        -INT16_C( 13663), -INT16_C( 28705), -INT16_C( 23452),  INT16_C( 24916),  INT16_C(  8791), -INT16_C(  4168), -INT16_C( 32627), -INT16_C(  6673) },
      UINT32_C(2432708705) },
    { UINT32_C(1747495759),
      {  INT16_C( 23729),  INT16_C(   779), -INT16_C(  9719), -INT16_C( 29537), -INT16_C( 22228), -INT16_C( 13009),  INT16_C(  3955), -INT16_C( 10404),
        -INT16_C( 20301),  INT16_C(  2873), -INT16_C(  3629),  INT16_C( 24826), -INT16_C(  5775), -INT16_C( 16315),  INT16_C( 28309),  INT16_C( 17961),
         INT16_C( 13514), -INT16_C( 11447), -INT16_C(  5873),  INT16_C( 15200), -INT16_C( 28782),  INT16_C(  1288),  INT16_C( 25758),  INT16_C( 21213),
         INT16_C(  5652), -INT16_C(  6307),  INT16_C( 22279),  INT16_C( 30791), -INT16_C( 29376), -INT16_C( 10952),  INT16_C( 25083), -INT16_C( 15077) },
      {  INT16_C( 26006), -INT16_C( 23144), -INT16_C(  1970), -INT16_C(  7968), -INT16_C(  6008),  INT16_C(  9957), -INT16_C( 15796),  INT16_C( 24696),
        -INT16_C( 10792), -INT16_C(  8376), -INT16_C( 28884),  INT16_C( 27991), -INT16_C( 28644),  INT16_C(  5954),  INT16_C( 24305), -INT16_C( 30756),
         INT16_C( 30147),  INT16_C(  4396),  INT16_C(  3181), -INT16_C(  2575), -INT16_C( 10508),  INT16_C( 16412), -INT16_C( 27495),  INT16_C( 29089),
        -INT16_C(  5782), -INT16_C( 27055), -INT16_C( 22408), -INT16_C( 27389),  INT16_C( 17976),  INT16_C( 10924), -INT16_C( 30300),  INT16_C( 26545) },
      UINT32_C(1208517186) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epi16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epi16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 r = easysimd_mm512_mask_cmpgt_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpgt_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask16 r;
  } test_vec[8] = {
       { UINT16_C(12249),
      easysimd_mm512_set_epi32(INT32_C(-1151856667), INT32_C(  -49918748), INT32_C(-1709830250), INT32_C( 1750293451),
                            INT32_C(-1728641738), INT32_C(   79295022), INT32_C(  308064941), INT32_C( 1216157597),
                            INT32_C(  320231148), INT32_C( -697875804), INT32_C(  990066955), INT32_C(-2091005609),
                            INT32_C( 1037816180), INT32_C( -748290940), INT32_C( 1166526776), INT32_C(-1428331975)),
      easysimd_mm512_set_epi32(INT32_C( 1846695950), INT32_C(  884046092), INT32_C( -392734818), INT32_C(-1618937497),
                            INT32_C(  670851975), INT32_C(-1269946840), INT32_C(-1917256160), INT32_C(  228203505),
                            INT32_C( 1263965918), INT32_C(-2053175323), INT32_C(-1206891799), INT32_C( -371464947),
                            INT32_C(  -99745007), INT32_C( -847464628), INT32_C(  -93605380), INT32_C(-1859612096)),
      UINT16_C( 1865) },
    { UINT16_C(47912),
      easysimd_mm512_set_epi32(INT32_C(  238725197), INT32_C( 1521340392), INT32_C(-2077410041), INT32_C( 2110592657),
                            INT32_C(  630925822), INT32_C(  689275449), INT32_C(-1970822997), INT32_C(-1836727953),
                            INT32_C(  237271984), INT32_C( -578417637), INT32_C( -768235708), INT32_C( 1362514984),
                            INT32_C( 2019276284), INT32_C(-1411239380), INT32_C( 2052618114), INT32_C(-1238213534)),
      easysimd_mm512_set_epi32(INT32_C( -669654436), INT32_C( -822780196), INT32_C(  205688995), INT32_C( 1680146061),
                            INT32_C(  393599682), INT32_C(-1451941808), INT32_C(  947305201), INT32_C(  -75999449),
                            INT32_C( -484166756), INT32_C(  833555969), INT32_C( -549302423), INT32_C( 1610578173),
                            INT32_C(-2070337581), INT32_C(  664412106), INT32_C(  255732930), INT32_C( 1319359183)),
      UINT16_C(38920) },
    { UINT16_C(35968),
      easysimd_mm512_set_epi32(INT32_C( -359339347), INT32_C( -666198998), INT32_C(  830421084), INT32_C(-2087460228),
                            INT32_C(-1475104408), INT32_C(  721289147), INT32_C( 1281161083), INT32_C(  852871551),
                            INT32_C(-1589170839), INT32_C( -746357372), INT32_C( -154169474), INT32_C( -148954645),
                            INT32_C(-1357418925), INT32_C(-2112116028), INT32_C(  174617048), INT32_C(   -4103177)),
      easysimd_mm512_set_epi32(INT32_C( -871886017), INT32_C(-1688556984), INT32_C(  524690471), INT32_C( -124192434),
                            INT32_C( 1423100629), INT32_C( -161239972), INT32_C( -396308843), INT32_C( 1070701282),
                            INT32_C(-1826016016), INT32_C(  425347357), INT32_C(  366897524), INT32_C( -401692546),
                            INT32_C( -812557761), INT32_C( 1614519786), INT32_C(-1648390428), INT32_C( 1830061179)),
      UINT16_C(33920) },
    { UINT16_C(16809),
      easysimd_mm512_set_epi32(INT32_C(-1280324837), INT32_C( -161347329), INT32_C(  735858798), INT32_C( -674499230),
                            INT32_C(  -21391979), INT32_C( -381504266), INT32_C( 1528317100), INT32_C(  430345703),
                            INT32_C( -679679907), INT32_C(  515475896), INT32_C( -731085275), INT32_C( 1308429395),
                            INT32_C(  582932299), INT32_C(  489160586), INT32_C( 1760701165), INT32_C(  118948129)),
      easysimd_mm512_set_epi32(INT32_C( 1240889055), INT32_C(  888122014), INT32_C(-1469013917), INT32_C( 1209972337),
                            INT32_C( -691361230), INT32_C(  361393556), INT32_C(-1167116782), INT32_C( 1467757879),
                            INT32_C( 2124803699), INT32_C( -154870634), INT32_C(-1603618479), INT32_C(-2032203238),
                            INT32_C( 2111773805), INT32_C( -496949166), INT32_C( 1844580247), INT32_C(-2053814402)),
      UINT16_C(   33) },
    { UINT16_C(44464),
      easysimd_mm512_set_epi32(INT32_C( 1072149321), INT32_C( 1813169024), INT32_C(-1284365076), INT32_C(-1623700182),
                            INT32_C(  530512850), INT32_C(  116537892), INT32_C(  258206492), INT32_C(  690441736),
                            INT32_C( 1005371642), INT32_C( 1116924342), INT32_C( 1297564984), INT32_C( -835039581),
                            INT32_C( 1286263864), INT32_C(-1749149234), INT32_C(  558298824), INT32_C( 1064688827)),
      easysimd_mm512_set_epi32(INT32_C( 1323805616), INT32_C(-1558886902), INT32_C( 1778691088), INT32_C(  108147743),
                            INT32_C( 1106435712), INT32_C( -967535450), INT32_C(  600280311), INT32_C(  109364043),
                            INT32_C(  423389578), INT32_C( 1225761441), INT32_C( -440804681), INT32_C( -707540326),
                            INT32_C(-1898655855), INT32_C(-1268681648), INT32_C(-1360056367), INT32_C( -275254487)),
      UINT16_C( 1440) },
    { UINT16_C(23993),
      easysimd_mm512_set_epi32(INT32_C(-2038065128), INT32_C( -446679229), INT32_C(   78082001), INT32_C(  379830516),
                            INT32_C(-1929569644), INT32_C( 1595859976), INT32_C(  320798226), INT32_C( -738570818),
                            INT32_C( -165441023), INT32_C( -172594873), INT32_C(  912601062), INT32_C(  -56802863),
                            INT32_C(  503255814), INT32_C( 2046199592), INT32_C( -622599746), INT32_C( 1337235103)),
      easysimd_mm512_set_epi32(INT32_C(-1519343201), INT32_C( -448055921), INT32_C(-1909251875), INT32_C( -347447915),
                            INT32_C(  397553753), INT32_C(  713040821), INT32_C(-1458903601), INT32_C(  -45886582),
                            INT32_C( 1230465483), INT32_C( -828483015), INT32_C( -699493978), INT32_C(-1811052070),
                            INT32_C( 1577065087), INT32_C( -109599940), INT32_C(-1093577090), INT32_C(-1788879767)),
      UINT16_C(21553) },
    { UINT16_C(10358),
      easysimd_mm512_set_epi32(INT32_C(  648390363), INT32_C(  -30837841), INT32_C(-1635592815), INT32_C( -694389961),
                            INT32_C( -883952626), INT32_C( -761345991), INT32_C(  346040825), INT32_C(-1780780575),
                            INT32_C( 1510717568), INT32_C(-1185143236), INT32_C( 2143540932), INT32_C(  880567806),
                            INT32_C(-1670993371), INT32_C(-1942419167), INT32_C(-1196759463), INT32_C( 1386099146)),
      easysimd_mm512_set_epi32(INT32_C(-1614031176), INT32_C(  414071648), INT32_C(-1152911954), INT32_C(  424701353),
                            INT32_C( 1739922394), INT32_C( -506382165), INT32_C(  257126844), INT32_C( 1724223193),
                            INT32_C( 1096709845), INT32_C(-1643231112), INT32_C(-1639890652), INT32_C( -403971200),
                            INT32_C( 1318667734), INT32_C(  206062573), INT32_C(  -18472190), INT32_C(   -1701112)),
      UINT16_C(  112) },
    { UINT16_C(35023),
      easysimd_mm512_set_epi32(INT32_C(  228305355), INT32_C(-1904004735), INT32_C(  118523411), INT32_C( 1661507666),
                            INT32_C(-1400326500), INT32_C(   63010183), INT32_C(   62197704), INT32_C( -635599967),
                            INT32_C( 1677709284), INT32_C(-1294080152), INT32_C( -900737233), INT32_C(-1991940005),
                            INT32_C( -240404149), INT32_C(-1448242105), INT32_C(-1972665039), INT32_C( 1511694245)),
      easysimd_mm512_set_epi32(INT32_C(-1506289043), INT32_C(   82234507), INT32_C( -557930538), INT32_C( -911612825),
                            INT32_C( 1352158017), INT32_C( -554125937), INT32_C( -727289650), INT32_C(-1102664191),
                            INT32_C( 1941639559), INT32_C(-2124299952), INT32_C( -385431179), INT32_C(  112242864),
                            INT32_C(  -66697069), INT32_C( 1379403470), INT32_C(-1996504296), INT32_C(  658235880)),
      UINT16_C(32835) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmpgt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask8 r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(-3344943500899736927), INT64_C( -508674271294480923),
                            INT64_C( 4367550852745697236), INT64_C(-1765523250257788813),
                            INT64_C(-6325172456788566604), INT64_C( 3340966423446181237),
                            INT64_C( 4899669249714194025), INT64_C(-3109134868060088778)),
      easysimd_mm512_set_epi64(INT64_C(-3344943500899736927), INT64_C( -115747659216396604),
                            INT64_C( 7173930031566073597), INT64_C(-1765523250257788813),
                            INT64_C(-4525526016698522721), INT64_C( 3340966423446181237),
                            INT64_C( 4899669249714194025), INT64_C(-3109134868060088778)),
      UINT8_C(  0) },
    { easysimd_mm512_set_epi64(INT64_C(  161300839730502102), INT64_C(-4154011080047619988),
                            INT64_C( 3510183776865199207), INT64_C( 5188039415407364902),
                            INT64_C(-8649541106015277925), INT64_C( 2036065042708484429),
                            INT64_C(-7714656548902029220), INT64_C(-9105101012109136347)),
      easysimd_mm512_set_epi64(INT64_C(  161300839730502102), INT64_C(-4154011080047619988),
                            INT64_C( 6120426103009778754), INT64_C( -248023738166197182),
                            INT64_C(-8649541106015277925), INT64_C( 2693047687180833180),
                            INT64_C(-7714656548902029220), INT64_C(-9105101012109136347)),
      UINT8_C( 16) },
    { easysimd_mm512_set_epi64(INT64_C(-2825253727352691686), INT64_C( 4405965118825132522),
                            INT64_C(-6791426899562410985), INT64_C(-4409900925389880930),
                            INT64_C( 1845845856613597884), INT64_C(-4842241423465696621),
                            INT64_C(  163081221433998591), INT64_C( 4482804709675222173)),
      easysimd_mm512_set_epi64(INT64_C( 3285810068291760082), INT64_C( 4405965118825132522),
                            INT64_C(-6791426899562410985), INT64_C(-4943963491966669187),
                            INT64_C( 1845845856613597884), INT64_C( 7384036385676540465),
                            INT64_C(-7313503223753260102), INT64_C( 5128036791088991318)),
      UINT8_C( 18) },
    { easysimd_mm512_set_epi64(INT64_C(-1633105180711142836), INT64_C(-4287337651200520652),
                            INT64_C( 8346707004388378871), INT64_C(-5848595418894109542),
                            INT64_C(-7300386321370732776), INT64_C( -648586863376006844),
                            INT64_C(-3473939784680689044), INT64_C(-1628617817613399979)),
      easysimd_mm512_set_epi64(INT64_C( 1934898870952452550), INT64_C(-4287337651200520652),
                            INT64_C( 1557479703737443505), INT64_C(-5848595418894109542),
                            INT64_C(-7179299072208562799), INT64_C( -648586863376006844),
                            INT64_C(-3473939784680689044), INT64_C( 2817575692173645704)),
      UINT8_C( 32) },
    { easysimd_mm512_set_epi64(INT64_C(-8952123954418726140), INT64_C( 5461301954902244462),
                            INT64_C(-5820184907423972656), INT64_C(  420402622060248705),
                            INT64_C(-1664441445637860283), INT64_C(-9088734991256809986),
                            INT64_C( 5606803261787264235), INT64_C( 3392608019150722653)),
      easysimd_mm512_set_epi64(INT64_C(-8952123954418726140), INT64_C(-6318099565586317695),
                            INT64_C(-5820184907423972656), INT64_C(  420402622060248705),
                            INT64_C(-1664441445637860283), INT64_C( 6565206217411025613),
                            INT64_C( 8598198622090956400), INT64_C(-7576266643160730964)),
      UINT8_C( 65) },
    { easysimd_mm512_set_epi64(INT64_C(-3313522622815895345), INT64_C(-6452175545498154090),
                            INT64_C( -937049212555566038), INT64_C(-4143019958444030865),
                            INT64_C(-3410665359562609619), INT64_C(  966786109195223540),
                            INT64_C( 7283097367839393163), INT64_C(-2640534975929709368)),
      easysimd_mm512_set_epi64(INT64_C(-3313522622815895345), INT64_C(-6452175545498154090),
                            INT64_C( 7057508826094118763), INT64_C(-2466255848420720587),
                            INT64_C(-3410665359562609619), INT64_C(-7091282311083875172),
                            INT64_C(-5778676633446214654), INT64_C(-2640534975929709368)),
      UINT8_C(  6) },
    { easysimd_mm512_set_epi64(INT64_C( 7946101066156420330), INT64_C(-1199223599247032864),
                            INT64_C(-1997073553979895023), INT64_C(-2305098272308636911),
                            INT64_C( -630363562210498119), INT64_C( 4426020973322885294),
                            INT64_C( 8782098874831326668), INT64_C(-6058337867533474769)),
      easysimd_mm512_set_epi64(INT64_C( 7946101066156420330), INT64_C(-1199223599247032864),
                            INT64_C(-1997073553979895023), INT64_C(-2305098272308636911),
                            INT64_C( -630363562210498119), INT64_C( 8629524505567702841),
                            INT64_C( 8782098874831326668), INT64_C( 2660246489815857132)),
      UINT8_C(  0) },
    { easysimd_mm512_set_epi64(INT64_C(-2815932903868980343), INT64_C(  791308056982133256),
                            INT64_C( 8277712790583824674), INT64_C(-3943050990178000322),
                            INT64_C(-2127265598488665647), INT64_C( 4379715049649431166),
                            INT64_C(-9154071905230416728), INT64_C(-2123362159730266714)),
      easysimd_mm512_set_epi64(INT64_C(-2815932903868980343), INT64_C(  791308056982133256),
                            INT64_C(-6685750631550937327), INT64_C( 1585978438239301211),
                            INT64_C( 3432556139556266760), INT64_C( 4379715049649431166),
                            INT64_C(-9154071905230416728), INT64_C(-1483875325616410698)),
      UINT8_C( 32) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpgt_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpgt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_cmpgt_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C( 16),
      easysimd_mm512_set_epi64(INT64_C( 2255026789087372129), INT64_C( 6954636019969939696),
                            INT64_C( 8135587588110756767), INT64_C(-6775895683000468083),
                            INT64_C( -900701109459786534), INT64_C(-7915280239032503360),
                            INT64_C(-1216817989725562739), INT64_C(-8432176550710264558)),
      easysimd_mm512_set_epi64(INT64_C( 2255026789087372129), INT64_C( 6954636019969939696),
                            INT64_C( 8135587588110756767), INT64_C(-6775895683000468083),
                            INT64_C( -900701109459786534), INT64_C(-3598046066747317833),
                            INT64_C(-1709941778912207388), INT64_C( 3169788859761289772)),
      UINT8_C(  0) },
    { UINT8_C(  6),
      easysimd_mm512_set_epi64(INT64_C(-2239217399172416071), INT64_C(-1788368623206610532),
                            INT64_C(-8621353419023287056), INT64_C( 2167646577764527484),
                            INT64_C( 7373916114077140533), INT64_C( 4679914315089846813),
                            INT64_C(-4785189148228967071), INT64_C(-2291884851836211058)),
      easysimd_mm512_set_epi64(INT64_C(-4674113889822188979), INT64_C( 6851357122574510655),
                            INT64_C(-8621353419023287056), INT64_C( 2167646577764527484),
                            INT64_C( 7373916114077140533), INT64_C(-2091630176064440564),
                            INT64_C(-4166997073722787613), INT64_C(-2291884851836211058)),
      UINT8_C(  4) },
    { UINT8_C(231),
      easysimd_mm512_set_epi64(INT64_C( -437845083503418422), INT64_C( 7030863114044209442),
                            INT64_C( 8238064857893579595), INT64_C( 3062172269146343786),
                            INT64_C( 7457006241836305381), INT64_C(-9078752323516671886),
                            INT64_C(-6382075143273833301), INT64_C( 3840898770164583597)),
      easysimd_mm512_set_epi64(INT64_C(-4268408126209392137), INT64_C( 7030863114044209442),
                            INT64_C(  663353489862938549), INT64_C( 3062172269146343786),
                            INT64_C( 7457006241836305381), INT64_C( 8174310593560152615),
                            INT64_C(-6382075143273833301), INT64_C(-4495103935185291795)),
      UINT8_C(161) },
    { UINT8_C( 60),
      easysimd_mm512_set_epi64(INT64_C( 3543184366849060052), INT64_C( 8101296544771348510),
                            INT64_C( 1359772700119148960), INT64_C(-8357828074665392254),
                            INT64_C(-5672294839872616078), INT64_C(-2918525673450782654),
                            INT64_C(-6303315662009814438), INT64_C( 4773615511108508590)),
      easysimd_mm512_set_epi64(INT64_C( 3543184366849060052), INT64_C(  286276641590586651),
                            INT64_C( 1359772700119148960), INT64_C(-3217204137928962858),
                            INT64_C(-5672294839872616078), INT64_C(-2918525673450782654),
                            INT64_C(-2554453706959743566), INT64_C(-6197005744039272430)),
      UINT8_C(  0) },
    { UINT8_C( 97),
      easysimd_mm512_set_epi64(INT64_C(-4278296701436995238), INT64_C( 3569507405853529045),
                            INT64_C(-3380367559374400304), INT64_C(-4948363566435325304),
                            INT64_C(-6678298576976263631), INT64_C( 8848650777417470336),
                            INT64_C( 6320411494008491541), INT64_C( 2280208700508329072)),
      easysimd_mm512_set_epi64(INT64_C(  326944370261152484), INT64_C( 3569507405853529045),
                            INT64_C(  715678757448860576), INT64_C(-4948363566435325304),
                            INT64_C(-6678298576976263631), INT64_C(-5367013526541491012),
                            INT64_C( 1008601224594483315), INT64_C( 2280208700508329072)),
      UINT8_C(  0) },
    { UINT8_C(153),
      easysimd_mm512_set_epi64(INT64_C( 8361426666750729591), INT64_C(-6668359429543518025),
                            INT64_C( 2952092805333509636), INT64_C( 8284871946243647248),
                            INT64_C(-8896262213455925533), INT64_C( 3194469353298560173),
                            INT64_C( 5466230282228711049), INT64_C(-1091365868294702661)),
      easysimd_mm512_set_epi64(INT64_C(-8667260419906723988), INT64_C(-6668359429543518025),
                            INT64_C( 2952092805333509636), INT64_C( 8284871946243647248),
                            INT64_C( 3185065043241333471), INT64_C( -134870333477219304),
                            INT64_C( 5466230282228711049), INT64_C(-4571723861926798973)),
      UINT8_C(129) },
    { UINT8_C( 60),
      easysimd_mm512_set_epi64(INT64_C(-5632979726637184794), INT64_C( 3790754159972080576),
                            INT64_C(-7842038005332057398), INT64_C(-1292705499011984897),
                            INT64_C( 7597886654367336733), INT64_C( 1457057381762531412),
                            INT64_C(-1572264173383359920), INT64_C(-8716209376375056305)),
      easysimd_mm512_set_epi64(INT64_C(-5632979726637184794), INT64_C( 3790754159972080576),
                            INT64_C( 1913605115921194336), INT64_C(-6143563121944184390),
                            INT64_C( 7597886654367336733), INT64_C( 1457057381762531412),
                            INT64_C( 7253226870637562008), INT64_C(-6283001429373579825)),
      UINT8_C( 16) },
    { UINT8_C( 88),
      easysimd_mm512_set_epi64(INT64_C(-2374777447002601129), INT64_C(-5785141086360428669),
                            INT64_C( 6450311718709789609), INT64_C( 4609381622161693926),
                            INT64_C( -638886780002324864), INT64_C(-5739159461288227194),
                            INT64_C(-4392084870376418631), INT64_C( 2798977638636065147)),
      easysimd_mm512_set_epi64(INT64_C(  753500986908300233), INT64_C(-5785141086360428669),
                            INT64_C( 6450311718709789609), INT64_C(-4648819914956469219),
                            INT64_C(-8767820380557260648), INT64_C(-5739159461288227194),
                            INT64_C( 2360822030941279123), INT64_C(-6092063218708168180)),
      UINT8_C( 24) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 r;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask8 k = test_vec[i].k;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmpgt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[32];
    uint16_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { { UINT16_C(41909), UINT16_C(29473), UINT16_C(15100), UINT16_C(54041), UINT16_C(23065), UINT16_C(14439), UINT16_C(41059), UINT16_C(10232),
        UINT16_C(32394), UINT16_C( 9031), UINT16_C(63531), UINT16_C(57322), UINT16_C(58387), UINT16_C(41888), UINT16_C(31882), UINT16_C(16329),
        UINT16_C(59935), UINT16_C( 7346), UINT16_C(52005), UINT16_C(16111), UINT16_C(22053), UINT16_C(34934), UINT16_C(28406), UINT16_C(32943),
        UINT16_C(63212), UINT16_C( 6051), UINT16_C(36334), UINT16_C(  503), UINT16_C(38769), UINT16_C(64421), UINT16_C(28179), UINT16_C(13114) },
      { UINT16_C(60505), UINT16_C(32335), UINT16_C(16055), UINT16_C(56508), UINT16_C(12948), UINT16_C(35428), UINT16_C( 5025), UINT16_C(36106),
        UINT16_C(44297), UINT16_C(63397), UINT16_C(39994), UINT16_C(44024), UINT16_C(40243), UINT16_C(18086), UINT16_C(57356), UINT16_C(25977),
        UINT16_C(51404), UINT16_C(33763), UINT16_C(40710), UINT16_C(39519), UINT16_C(50129), UINT16_C(29220), UINT16_C(11990), UINT16_C(57088),
        UINT16_C(42459), UINT16_C( 5590), UINT16_C(52801), UINT16_C(29888), UINT16_C(26220), UINT16_C(30906), UINT16_C(13382), UINT16_C( 4829) },
      UINT32_C(4083498064) },
    { { UINT16_C(49404), UINT16_C(  917), UINT16_C(62559), UINT16_C(12445), UINT16_C(49847), UINT16_C(36259), UINT16_C(41968), UINT16_C(52332),
        UINT16_C(16968), UINT16_C(35297), UINT16_C(41489), UINT16_C(32253), UINT16_C(46856), UINT16_C(20469), UINT16_C(53995), UINT16_C(59489),
        UINT16_C(63378), UINT16_C(61931), UINT16_C(35051), UINT16_C(41761), UINT16_C(50250), UINT16_C(15152), UINT16_C(40295), UINT16_C(44807),
        UINT16_C(59615), UINT16_C(61496), UINT16_C(13706), UINT16_C(37741), UINT16_C(25325), UINT16_C(55522), UINT16_C(17204), UINT16_C(50880) },
      { UINT16_C(43834), UINT16_C( 9911), UINT16_C(55604), UINT16_C(32457), UINT16_C(63901), UINT16_C( 1465), UINT16_C(49302), UINT16_C(30388),
        UINT16_C(60841), UINT16_C(13158), UINT16_C(54306), UINT16_C( 4038), UINT16_C(43062), UINT16_C(27624), UINT16_C(43244), UINT16_C( 9777),
        UINT16_C(59732), UINT16_C(34892), UINT16_C( 5570), UINT16_C(24326), UINT16_C(49167), UINT16_C(42340), UINT16_C( 6528), UINT16_C(10523),
        UINT16_C(33286), UINT16_C(10333), UINT16_C( 9046), UINT16_C(35896), UINT16_C( 8396), UINT16_C(47351), UINT16_C(10696), UINT16_C( 7390) },
      UINT32_C(4292860581) },
    { { UINT16_C(11026), UINT16_C(54436), UINT16_C(43840), UINT16_C(20275), UINT16_C(39019), UINT16_C(60405), UINT16_C( 4273), UINT16_C(46869),
        UINT16_C(29330), UINT16_C(59615), UINT16_C( 6037), UINT16_C(24949), UINT16_C(27703), UINT16_C(   25), UINT16_C(63637), UINT16_C(42780),
        UINT16_C(49443), UINT16_C(25467), UINT16_C(44908), UINT16_C(55219), UINT16_C(43079), UINT16_C(63682), UINT16_C(55224), UINT16_C(19375),
        UINT16_C(36425), UINT16_C(57139), UINT16_C(43174), UINT16_C(56640), UINT16_C(23061), UINT16_C(43741), UINT16_C(64082), UINT16_C(30034) },
      { UINT16_C(52667), UINT16_C(10200), UINT16_C(35708), UINT16_C(50174), UINT16_C(49203), UINT16_C(60603), UINT16_C(27288), UINT16_C(57655),
        UINT16_C(27385), UINT16_C(40896), UINT16_C(  275), UINT16_C(10364), UINT16_C(23131), UINT16_C(44498), UINT16_C( 9300), UINT16_C( 3874),
        UINT16_C(64242), UINT16_C(28214), UINT16_C(13446), UINT16_C(47410), UINT16_C(60916), UINT16_C(36005), UINT16_C(56408), UINT16_C(20846),
        UINT16_C(11847), UINT16_C(23280), UINT16_C(27695), UINT16_C(35458), UINT16_C(21702), UINT16_C( 6711), UINT16_C(22905), UINT16_C(27433) },
      UINT32_C(4281130758) },
    { { UINT16_C(24404), UINT16_C(56025), UINT16_C( 2963), UINT16_C(34963), UINT16_C(14841), UINT16_C(20756), UINT16_C(33301), UINT16_C(23714),
        UINT16_C(37553), UINT16_C(57526), UINT16_C(14590), UINT16_C(50539), UINT16_C(41613), UINT16_C( 1759), UINT16_C( 2556), UINT16_C(20593),
        UINT16_C(19048), UINT16_C(64554), UINT16_C(48470), UINT16_C(20356), UINT16_C(39158), UINT16_C( 3232), UINT16_C(16923), UINT16_C(52328),
        UINT16_C( 8148), UINT16_C(53932), UINT16_C( 5975), UINT16_C(58519), UINT16_C(30650), UINT16_C(46826), UINT16_C(23424), UINT16_C(59398) },
      { UINT16_C(12454), UINT16_C(64740), UINT16_C(26861), UINT16_C(58443), UINT16_C(60161), UINT16_C( 7408), UINT16_C(22573), UINT16_C(  488),
        UINT16_C(38007), UINT16_C(53203), UINT16_C(27564), UINT16_C(26291), UINT16_C(40674), UINT16_C(25116), UINT16_C( 8953), UINT16_C(40778),
        UINT16_C(12114), UINT16_C(16283), UINT16_C(59031), UINT16_C(38947), UINT16_C( 5073), UINT16_C(65204), UINT16_C(40044), UINT16_C(58367),
        UINT16_C(54065), UINT16_C(56754), UINT16_C(26174), UINT16_C( 8259), UINT16_C(24324), UINT16_C(64898), UINT16_C(52353), UINT16_C(54173) },
      UINT32_C(2551388897) },
    { { UINT16_C(14587), UINT16_C(37650), UINT16_C(13855), UINT16_C(61483), UINT16_C(57417), UINT16_C(46575), UINT16_C(61052), UINT16_C(44441),
        UINT16_C(19393), UINT16_C(65418), UINT16_C(52657), UINT16_C(46367), UINT16_C(41260), UINT16_C(44467), UINT16_C(20590), UINT16_C(27008),
        UINT16_C(37768), UINT16_C(43004), UINT16_C(10441), UINT16_C( 4760), UINT16_C(34568), UINT16_C(33992), UINT16_C(24949), UINT16_C(14130),
        UINT16_C(48300), UINT16_C(24118), UINT16_C(22154), UINT16_C(46611), UINT16_C(50935), UINT16_C(25956), UINT16_C(58390), UINT16_C(40911) },
      { UINT16_C(52087), UINT16_C(16454), UINT16_C(57075), UINT16_C(64339), UINT16_C( 7013), UINT16_C(56192), UINT16_C(45692), UINT16_C(10258),
        UINT16_C(18542), UINT16_C(63622), UINT16_C(39582), UINT16_C(38575), UINT16_C( 4960), UINT16_C(30715), UINT16_C(51959), UINT16_C(28438),
        UINT16_C(23702), UINT16_C(35247), UINT16_C(  571), UINT16_C(41093), UINT16_C( 1309), UINT16_C(39291), UINT16_C(36279), UINT16_C( 9666),
        UINT16_C(18646), UINT16_C(29726), UINT16_C(52706), UINT16_C(17162), UINT16_C( 1760), UINT16_C(55226), UINT16_C(53456), UINT16_C(26182) },
      UINT32_C(3650568146) },
    { { UINT16_C(63020), UINT16_C(26608), UINT16_C(30200), UINT16_C( 5640), UINT16_C(33658), UINT16_C(12719), UINT16_C(28945), UINT16_C(59222),
        UINT16_C(29882), UINT16_C(40027), UINT16_C(26177), UINT16_C( 8671), UINT16_C(39276), UINT16_C(15609), UINT16_C(16233), UINT16_C(38563),
        UINT16_C(37685), UINT16_C(12029), UINT16_C( 1288), UINT16_C(33348), UINT16_C(62345), UINT16_C(39603), UINT16_C( 2405), UINT16_C( 8065),
        UINT16_C(56446), UINT16_C(49083), UINT16_C(39746), UINT16_C(44769), UINT16_C(55860), UINT16_C(40683), UINT16_C(36377), UINT16_C(20276) },
      { UINT16_C(12577), UINT16_C(10621), UINT16_C(49463), UINT16_C(49323), UINT16_C(24244), UINT16_C( 6490), UINT16_C(56167), UINT16_C(58680),
        UINT16_C(62647), UINT16_C(64165), UINT16_C(34447), UINT16_C(50088), UINT16_C(37728), UINT16_C(31073), UINT16_C(38177), UINT16_C(17096),
        UINT16_C(17863), UINT16_C(65131), UINT16_C( 5638), UINT16_C(48062), UINT16_C( 6260), UINT16_C(56532), UINT16_C( 3571), UINT16_C(43713),
        UINT16_C(26113), UINT16_C(37028), UINT16_C(19948), UINT16_C(19539), UINT16_C(46560), UINT16_C(  710), UINT16_C(36426), UINT16_C( 4420) },
      UINT32_C(3205599411) },
    { { UINT16_C(45268), UINT16_C(55823), UINT16_C(52678), UINT16_C(15253), UINT16_C(27365), UINT16_C(55319), UINT16_C(55415), UINT16_C(30851),
        UINT16_C(10047), UINT16_C(11016), UINT16_C(23412), UINT16_C(21880), UINT16_C(15888), UINT16_C(23383), UINT16_C(39884), UINT16_C(41068),
        UINT16_C(31819), UINT16_C( 4731), UINT16_C( 4169), UINT16_C(12109), UINT16_C(25722), UINT16_C(61703), UINT16_C(35388), UINT16_C(31593),
        UINT16_C(29106), UINT16_C( 9895), UINT16_C( 8141), UINT16_C(56699), UINT16_C(53853), UINT16_C(10552), UINT16_C(42350), UINT16_C(47562) },
      { UINT16_C(17697), UINT16_C(27339), UINT16_C( 6229), UINT16_C(53401), UINT16_C(41340), UINT16_C(47553), UINT16_C(11051), UINT16_C(56628),
        UINT16_C(56220), UINT16_C(26884), UINT16_C(32762), UINT16_C(22343), UINT16_C(32594), UINT16_C(49281), UINT16_C(19236), UINT16_C(17785),
        UINT16_C(17808), UINT16_C(58800), UINT16_C(18781), UINT16_C(55989), UINT16_C(30698), UINT16_C( 5779), UINT16_C(51106), UINT16_C(16115),
        UINT16_C(63395), UINT16_C(40360), UINT16_C(61303), UINT16_C(51701), UINT16_C(30318), UINT16_C(37769), UINT16_C(  705), UINT16_C(20952) },
      UINT32_C(3634479207) },
    { { UINT16_C(34887), UINT16_C(42294), UINT16_C(60626), UINT16_C(48255), UINT16_C( 4707), UINT16_C( 1490), UINT16_C(50905), UINT16_C(31811),
        UINT16_C(60349), UINT16_C(13338), UINT16_C( 4058), UINT16_C(18941), UINT16_C(34437), UINT16_C(18140), UINT16_C(46217), UINT16_C(53399),
        UINT16_C(52541), UINT16_C( 3957), UINT16_C(62649), UINT16_C( 7371), UINT16_C(40454), UINT16_C(57377), UINT16_C(25956), UINT16_C( 8540),
        UINT16_C(30288), UINT16_C(11094), UINT16_C(21381), UINT16_C( 2676), UINT16_C(20698), UINT16_C(25424), UINT16_C(59140), UINT16_C(16691) },
      { UINT16_C(43445), UINT16_C(28240), UINT16_C( 7325), UINT16_C(42123), UINT16_C(44218), UINT16_C( 7812), UINT16_C(57361), UINT16_C(25151),
        UINT16_C(38231), UINT16_C(56461), UINT16_C(  489), UINT16_C(50151), UINT16_C(14161), UINT16_C(21798), UINT16_C(22815), UINT16_C(54423),
        UINT16_C(59138), UINT16_C(41026), UINT16_C(52483), UINT16_C(48452), UINT16_C(51322), UINT16_C(35803), UINT16_C( 7080), UINT16_C(65517),
        UINT16_C(31408), UINT16_C(39388), UINT16_C(50043), UINT16_C(52316), UINT16_C(33530), UINT16_C( 6434), UINT16_C(47580), UINT16_C(57069) },
      UINT32_C(1617188238) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpgt_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmpgt_epu16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u16x32();
    easysimd__m512i b = easysimd_test_x86_random_u16x32();
    easysimd__mmask32 r = easysimd_mm512_cmpgt_epu16_mask(a, b);

    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpgt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t a[16];
    uint32_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { { UINT32_C(1106179719), UINT32_C(1286909711), UINT32_C(2601701866), UINT32_C(1613598397), UINT32_C( 806098878), UINT32_C(2735713405), UINT32_C(3824532331), UINT32_C(1228010690),
        UINT32_C(4186644458), UINT32_C(3108388815), UINT32_C(3444856848), UINT32_C(2771223015), UINT32_C( 383072665), UINT32_C(1354360037), UINT32_C(1647619743), UINT32_C(2360042914) },
      { UINT32_C(1401304708), UINT32_C(2232208501), UINT32_C( 189948196), UINT32_C(2075164898), UINT32_C(2677114297), UINT32_C( 166677353), UINT32_C(2624267257), UINT32_C( 220731016),
        UINT32_C(3261115981), UINT32_C(2672323962), UINT32_C(2963970766), UINT32_C(3559676698), UINT32_C(1249099232), UINT32_C(  39019016), UINT32_C( 245284486), UINT32_C( 555468500) },
      UINT16_C(59364) },
    { { UINT32_C(4024728693), UINT32_C(3079548137), UINT32_C(3781638598), UINT32_C(1958056852), UINT32_C(1505634385), UINT32_C( 274403722), UINT32_C(2753558991), UINT32_C( 902118080),
        UINT32_C(2669980086), UINT32_C(2622927829), UINT32_C(2155724524), UINT32_C(2733912657), UINT32_C(3841766234), UINT32_C(2499106756), UINT32_C( 288887889), UINT32_C(  88538446) },
      { UINT32_C(2091149991), UINT32_C( 152632093), UINT32_C( 176788921), UINT32_C( 565018311), UINT32_C(4110854449), UINT32_C(1367997184), UINT32_C(1566753039), UINT32_C(1717741759),
        UINT32_C( 820119315), UINT32_C(3141204738), UINT32_C(1489355664), UINT32_C(1920561729), UINT32_C( 476610331), UINT32_C(2305683834), UINT32_C(1927794867), UINT32_C(2346207608) },
      UINT16_C(15695) },
    { { UINT32_C(1388100176), UINT32_C(1175320245), UINT32_C(4221490105), UINT32_C(1634539333), UINT32_C( 293459351), UINT32_C(2057038535), UINT32_C( 871137978), UINT32_C( 482264267),
        UINT32_C( 879655550), UINT32_C( 712670320), UINT32_C(2485459023), UINT32_C(3337982511), UINT32_C( 802714216), UINT32_C( 396981085), UINT32_C(3226113525), UINT32_C(3621521753) },
      { UINT32_C(4094380931), UINT32_C( 371099079), UINT32_C(3450487709), UINT32_C(1033085141), UINT32_C(1869376274), UINT32_C(3548845534), UINT32_C(  60084650), UINT32_C(1591439578),
        UINT32_C(2186471099), UINT32_C( 160985196), UINT32_C(2295743411), UINT32_C(4123421411), UINT32_C(3026531029), UINT32_C(4068994120), UINT32_C(2566265789), UINT32_C(1207357836) },
      UINT16_C(50766) },
    { { UINT32_C( 600459447), UINT32_C(1798136504), UINT32_C(2297627557), UINT32_C(1132378477), UINT32_C( 888660972), UINT32_C(2351333071), UINT32_C( 639900826), UINT32_C(2775390957),
        UINT32_C( 449328994), UINT32_C(1065743770), UINT32_C(1724414457), UINT32_C( 514410034), UINT32_C(4166164521), UINT32_C(3112466719), UINT32_C(2212473237), UINT32_C( 640175299) },
      { UINT32_C( 507572356), UINT32_C(3747464934), UINT32_C(1900356927), UINT32_C(2492526443), UINT32_C(2928468623), UINT32_C(4050063707), UINT32_C(2104772282), UINT32_C( 396598419),
        UINT32_C(1916134540), UINT32_C(3914437290), UINT32_C( 593139640), UINT32_C( 364440198), UINT32_C( 683951309), UINT32_C( 253307733), UINT32_C(  93162866), UINT32_C(3055300649) },
      UINT16_C(31877) },
    { { UINT32_C(3190313236), UINT32_C(2611444451), UINT32_C(2545877521), UINT32_C(3115087852), UINT32_C( 300052667), UINT32_C( 237042588), UINT32_C(2987634057), UINT32_C(4066914270),
        UINT32_C(1672581504), UINT32_C( 486496267), UINT32_C(1202962010), UINT32_C(4026556213), UINT32_C(1812062928), UINT32_C(1736057566), UINT32_C(2904133071), UINT32_C(1017152188) },
      { UINT32_C( 513757459), UINT32_C(  70950569), UINT32_C(2437672284), UINT32_C( 478301004), UINT32_C( 210273070), UINT32_C(1970471589), UINT32_C(1260555407), UINT32_C( 562545166),
        UINT32_C(3175032595), UINT32_C( 583104965), UINT32_C(3014855782), UINT32_C(2244949335), UINT32_C(1586583737), UINT32_C(3939697754), UINT32_C(2671113873), UINT32_C(3435249080) },
      UINT16_C(22751) },
    { { UINT32_C(2844328164), UINT32_C(3771419258), UINT32_C(2912124758), UINT32_C(1832084404), UINT32_C( 365741243), UINT32_C(1526702025), UINT32_C(1324955029), UINT32_C(3592076018),
        UINT32_C( 897557435), UINT32_C(1125469165), UINT32_C(2129701322), UINT32_C(3354141452), UINT32_C(2984032488), UINT32_C(3976977495), UINT32_C(  54199313), UINT32_C(2061063615) },
      { UINT32_C(3853474040), UINT32_C(1831388323), UINT32_C(2045515885), UINT32_C( 608229436), UINT32_C(3889503632), UINT32_C( 181723385), UINT32_C(2735542244), UINT32_C(1545463396),
        UINT32_C(3795962942), UINT32_C(4266617233), UINT32_C(3195550594), UINT32_C(2749544467), UINT32_C(3465263061), UINT32_C(2077777815), UINT32_C(3542082927), UINT32_C( 170933451) },
      UINT16_C(43182) },
    { { UINT32_C(2599186697), UINT32_C(1570257883), UINT32_C(2317029495), UINT32_C(2653879753), UINT32_C(1265482164), UINT32_C(2277917976), UINT32_C(4133217579), UINT32_C( 721455906),
        UINT32_C(3620072700), UINT32_C(2671009064), UINT32_C( 925454190), UINT32_C(  13981516), UINT32_C( 692797968), UINT32_C(3014660744), UINT32_C( 430509047), UINT32_C(2470750870) },
      { UINT32_C(3194620310), UINT32_C(3579682663), UINT32_C( 990676974), UINT32_C(4013679070), UINT32_C(2887288612), UINT32_C(2422196377), UINT32_C(1789528276), UINT32_C(1241378482),
        UINT32_C(1594321144), UINT32_C(4113851655), UINT32_C(3392159980), UINT32_C(1169779745), UINT32_C(2364658163), UINT32_C(1847349402), UINT32_C( 198755929), UINT32_C(2908018357) },
      UINT16_C(24900) },
    { { UINT32_C(1158437950), UINT32_C(2906276033), UINT32_C(2708958080), UINT32_C(3404149207), UINT32_C(2622937090), UINT32_C(2181722665), UINT32_C(4002276153), UINT32_C(4154188473),
        UINT32_C(4282165054), UINT32_C(1739355879), UINT32_C(3087541217), UINT32_C(1468198740), UINT32_C(4059289800), UINT32_C(2222194251), UINT32_C(2591162593), UINT32_C( 546377186) },
      { UINT32_C(2602552756), UINT32_C( 620940099), UINT32_C(1138559727), UINT32_C(3264897274), UINT32_C(2209582648), UINT32_C(1829185164), UINT32_C( 151484710), UINT32_C( 975804550),
        UINT32_C(2832550245), UINT32_C(  63821588), UINT32_C(3695684578), UINT32_C(1134485771), UINT32_C(4224143727), UINT32_C(2640891511), UINT32_C(3467013960), UINT32_C(1812582407) },
      UINT16_C( 3070) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpgt_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmpgt_epu32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u32x16();
    easysimd__m512i b = easysimd_test_x86_random_u32x16();
    easysimd__mmask16 r = easysimd_mm512_cmpgt_epu32_mask(a, b);

    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpgt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint64_t a[8];
    uint64_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { { UINT64_C( 9578898152910572725), UINT64_C( 7462447345553309654), UINT64_C( 7879409581741859821), UINT64_C(12512307048847968526),
        UINT64_C( 8111840110130791920), UINT64_C( 3730302402354697482), UINT64_C(11873480479073783062), UINT64_C(10358577789550721564) },
      { UINT64_C(17554804745797819796), UINT64_C(10015746588171879047), UINT64_C(16905576002357607009), UINT64_C(12381449907234277285),
        UINT64_C( 2987990616704184732), UINT64_C( 4412109113156618043), UINT64_C( 8027615528425430703), UINT64_C(13878834169870771672) },
      UINT8_C( 88) },
    { { UINT64_C( 3009456708990039543), UINT64_C( 6817625955345414531), UINT64_C( 8892993686269055537), UINT64_C(11280116996536737447),
        UINT64_C( 1122233638624612475), UINT64_C(12592078290789821107), UINT64_C( 9726671141592472798), UINT64_C(16749693427961128753) },
      { UINT64_C( 6073354379100006573), UINT64_C(11635462795108464462), UINT64_C( 7455462024674419219), UINT64_C( 6042897931621484934),
        UINT64_C( 6867378180459343447), UINT64_C(10228135974444598156), UINT64_C(13358510143562519493), UINT64_C(11941445381779201983) },
      UINT8_C(172) },
    { { UINT64_C(11901019411804144823), UINT64_C(13616323369886831598), UINT64_C( 3162680786406569465), UINT64_C(15180307349405471160),
        UINT64_C( 2399226952622491789), UINT64_C( 2218444088574296087), UINT64_C( 5875841375115313396), UINT64_C(17486635560349408192) },
      { UINT64_C( 2639078653272017227), UINT64_C(11270534767695702052), UINT64_C(11558074890204263035), UINT64_C(  862766708354170531),
        UINT64_C(11142794392455713394), UINT64_C( 5446342680997254225), UINT64_C(13582511340603347709), UINT64_C(10759307468980157060) },
      UINT8_C(139) },
    { { UINT64_C( 6007721650459826767), UINT64_C(10135037112488838027), UINT64_C( 5857750230592259700), UINT64_C( 1989155945083965154),
        UINT64_C(17372143418450423488), UINT64_C(13375112846817125251), UINT64_C(15138165086235265016), UINT64_C( 5946284856195984286) },
      { UINT64_C( 2915848373176460577), UINT64_C(12505888916227408783), UINT64_C(  189581526373159903), UINT64_C(  943488646184204595),
        UINT64_C(16727788872055074135), UINT64_C( 1527082247159308238), UINT64_C( 6810434582918505244), UINT64_C( 4105095802335260671) },
      UINT8_C(253) },
    { { UINT64_C( 7325203302217714791), UINT64_C(14556790329436628152), UINT64_C( 7946545948057594302), UINT64_C(  925017644751022781),
        UINT64_C( 2079384727295167198), UINT64_C(11407917883335623898), UINT64_C( 2798652270089575673), UINT64_C( 7055903132374806585) },
      { UINT64_C(  515013638912194725), UINT64_C( 4248266919746941302), UINT64_C( 9515112375819336572), UINT64_C( 1192694599024562601),
        UINT64_C(10176515672576189445), UINT64_C( 2804572321097173360), UINT64_C(12547337548731288914), UINT64_C(15449150820038902762) },
      UINT8_C( 35) },
    { { UINT64_C( 3346250109779804108), UINT64_C( 8997851390355133197), UINT64_C(17080809797854030423), UINT64_C( 1902112572510037694),
        UINT64_C(13841089649251807016), UINT64_C( 1686411967746078144), UINT64_C( 8732901594732200000), UINT64_C( 5102135402182505573) },
      { UINT64_C(17964931728563620828), UINT64_C( 6777437085619243063), UINT64_C( 7493941407986647755), UINT64_C( 5885334403988195144),
        UINT64_C( 6045573545816349108), UINT64_C(16245988078417267449), UINT64_C( 3765686505426065626), UINT64_C(17957336355194953070) },
      UINT8_C( 86) },
    { { UINT64_C( 4651881170951607839), UINT64_C(  760769896510904471), UINT64_C(14629143343689758180), UINT64_C(  415707646007353221),
        UINT64_C( 1117119719162292985), UINT64_C( 9862965255779178406), UINT64_C( 8315719615881266751), UINT64_C(17503523599523059197) },
      { UINT64_C(11399371485151963564), UINT64_C( 8698514905092189835), UINT64_C( 7817258765737267227), UINT64_C( 7456897461132442256),
        UINT64_C(17147663811336712034), UINT64_C(16022915512088161708), UINT64_C( 8989014343142633700), UINT64_C(12196171411496439326) },
      UINT8_C(132) },
    { { UINT64_C( 3927415447657630768), UINT64_C( 6127841570945932649), UINT64_C( 8453088652508590823), UINT64_C(  461216363337176182),
        UINT64_C(14779998572784478557), UINT64_C(10843828628131231927), UINT64_C(  271263319600998134), UINT64_C(  137683748954051085) },
      { UINT64_C( 6107169845179614095), UINT64_C(13079572345532496702), UINT64_C(12876009749526867231), UINT64_C( 2845570828174979544),
        UINT64_C(12246374166539494658), UINT64_C( 9803239524310347671), UINT64_C(12492956597228998407), UINT64_C( 8436832493795669078) },
      UINT8_C( 48) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpgt_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmpgt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m512i b = easysimd_test_x86_random_u64x8();
    easysimd__mmask8 r = easysimd_mm512_cmpgt_epu64_mask(a, b);

    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpgt_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask64 k;
    uint8_t a[64];
    uint8_t b[64];
    easysimd__mmask64 r;
  } test_vec[8] = {
    { UINT64_C(10947484077258789258),
      { UINT8_C(177), UINT8_C(119), UINT8_C(209), UINT8_C(152), UINT8_C( 82), UINT8_C(118), UINT8_C(151), UINT8_C( 63),
        UINT8_C(102), UINT8_C(207), UINT8_C( 96), UINT8_C( 15), UINT8_C( 62), UINT8_C(211), UINT8_C(  9), UINT8_C(205),
        UINT8_C( 35), UINT8_C(134), UINT8_C(236), UINT8_C(249), UINT8_C( 54), UINT8_C( 53), UINT8_C(236), UINT8_C(192),
        UINT8_C(134), UINT8_C( 45), UINT8_C(217), UINT8_C( 75), UINT8_C(115), UINT8_C(198), UINT8_C(226), UINT8_C( 36),
        UINT8_C( 61), UINT8_C(180), UINT8_C(189), UINT8_C(144), UINT8_C( 42), UINT8_C( 84), UINT8_C(207), UINT8_C(144),
        UINT8_C( 35), UINT8_C( 47), UINT8_C(160), UINT8_C( 97), UINT8_C(  2), UINT8_C(169), UINT8_C( 46), UINT8_C( 38),
        UINT8_C( 47), UINT8_C( 26), UINT8_C( 31), UINT8_C(101), UINT8_C( 79), UINT8_C( 11), UINT8_C( 37), UINT8_C(213),
        UINT8_C( 57), UINT8_C(254), UINT8_C( 32), UINT8_C(172), UINT8_C(197), UINT8_C(  3), UINT8_C(209), UINT8_C(  2) },
      { UINT8_C(183), UINT8_C(142), UINT8_C(146), UINT8_C(225), UINT8_C(226), UINT8_C( 98), UINT8_C(113), UINT8_C(  5),
        UINT8_C(145), UINT8_C( 17), UINT8_C(102), UINT8_C(148), UINT8_C(187), UINT8_C(148), UINT8_C(186), UINT8_C(234),
        UINT8_C(174), UINT8_C(217), UINT8_C( 80), UINT8_C(253), UINT8_C(228), UINT8_C(117), UINT8_C(210), UINT8_C( 29),
        UINT8_C(116), UINT8_C(243), UINT8_C(202), UINT8_C( 57), UINT8_C(246), UINT8_C(155), UINT8_C( 59), UINT8_C(173),
        UINT8_C( 41), UINT8_C(206), UINT8_C(142), UINT8_C( 11), UINT8_C( 48),    UINT8_MAX, UINT8_C( 16), UINT8_C(193),
        UINT8_C( 17), UINT8_C(118), UINT8_C( 85), UINT8_C(204), UINT8_C( 10), UINT8_C( 15), UINT8_C(182), UINT8_C(184),
        UINT8_C(232), UINT8_C(  6), UINT8_C(181), UINT8_C(205), UINT8_C(124), UINT8_C(135), UINT8_C(234), UINT8_C(240),
        UINT8_C(122), UINT8_C(180), UINT8_C( 41), UINT8_C(112), UINT8_C( 79), UINT8_C(100), UINT8_C( 29), UINT8_C(120) },
      UINT64_C( 1297041382942179456) },
    { UINT64_C(13557123142593522482),
      { UINT8_C(  9), UINT8_C(121), UINT8_C(136), UINT8_C( 19), UINT8_C(137), UINT8_C( 62), UINT8_C(203), UINT8_C(113),
        UINT8_C( 69), UINT8_C(128), UINT8_C( 62), UINT8_C(193), UINT8_C(  8), UINT8_C( 41), UINT8_C(177), UINT8_C(130),
        UINT8_C(221), UINT8_C(218), UINT8_C(243), UINT8_C( 45), UINT8_C( 62), UINT8_C( 16), UINT8_C(165), UINT8_C(113),
        UINT8_C(188), UINT8_C( 41), UINT8_C(211), UINT8_C(103), UINT8_C(188), UINT8_C(247), UINT8_C( 35), UINT8_C(198),
        UINT8_C(113), UINT8_C(171), UINT8_C(217), UINT8_C(250), UINT8_C(233), UINT8_C(165), UINT8_C(107), UINT8_C( 46),
        UINT8_C( 37), UINT8_C(170), UINT8_C(239), UINT8_C( 45), UINT8_C(211), UINT8_C(160), UINT8_C(176), UINT8_C(176),
        UINT8_C(122), UINT8_C(163), UINT8_C(221), UINT8_C(185), UINT8_C(179), UINT8_C(131), UINT8_C( 42), UINT8_C(111),
        UINT8_C(172), UINT8_C(253), UINT8_C(214), UINT8_C(104), UINT8_C(245), UINT8_C(249), UINT8_C( 46), UINT8_C(102) },
      { UINT8_C(164), UINT8_C(  8), UINT8_C( 96), UINT8_C(142), UINT8_C(173), UINT8_C(203), UINT8_C(188), UINT8_C(210),
        UINT8_C(117), UINT8_C(172), UINT8_C(  0), UINT8_C( 72), UINT8_C( 76), UINT8_C(176), UINT8_C(249), UINT8_C(199),
        UINT8_C( 83), UINT8_C(214), UINT8_C(128), UINT8_C(  6), UINT8_C( 89), UINT8_C(170), UINT8_C(118), UINT8_C(  5),
        UINT8_C(167), UINT8_C( 76), UINT8_C(110), UINT8_C(156), UINT8_C( 70), UINT8_C(156), UINT8_C(  2), UINT8_C(234),
        UINT8_C(164), UINT8_C( 98), UINT8_C(120), UINT8_C( 81), UINT8_C( 46), UINT8_C( 53), UINT8_C( 36), UINT8_C(163),
        UINT8_C(225), UINT8_C( 36), UINT8_C(236), UINT8_C( 45), UINT8_C(212), UINT8_C(229), UINT8_C(244), UINT8_C( 39),
        UINT8_C(187), UINT8_C(116), UINT8_C( 45), UINT8_C( 21), UINT8_C( 30), UINT8_C(163), UINT8_C( 26), UINT8_C(198),
        UINT8_C(240), UINT8_C(136), UINT8_C( 98), UINT8_C( 54), UINT8_C( 37), UINT8_C(101), UINT8_C( 32), UINT8_C(201) },
      UINT64_C( 4324724660701956098) },
    { UINT64_C(12653214786182158791),
      { UINT8_C( 99), UINT8_C(133), UINT8_C(220), UINT8_C( 55), UINT8_C(106), UINT8_C(209), UINT8_C( 94), UINT8_C( 37),
        UINT8_C( 69), UINT8_C(139), UINT8_C( 58), UINT8_C(100), UINT8_C( 47), UINT8_C( 85), UINT8_C( 42), UINT8_C( 31),
        UINT8_C(221), UINT8_C(140), UINT8_C( 85), UINT8_C(  2), UINT8_C(241), UINT8_C(117), UINT8_C(204), UINT8_C(185),
        UINT8_C( 14), UINT8_C(231), UINT8_C(174), UINT8_C(220), UINT8_C( 38), UINT8_C( 71), UINT8_C(139), UINT8_C(137),
        UINT8_C(204), UINT8_C(104), UINT8_C(192), UINT8_C( 54), UINT8_C( 57), UINT8_C( 30), UINT8_C( 92), UINT8_C(126),
        UINT8_C(169), UINT8_C(150), UINT8_C(226), UINT8_C(216), UINT8_C(235), UINT8_C( 12), UINT8_C(247), UINT8_C(201),
        UINT8_C(153), UINT8_C( 76), UINT8_C(203), UINT8_C(138), UINT8_C(194), UINT8_C(151), UINT8_C( 67), UINT8_C(208),
        UINT8_C(126), UINT8_C(242), UINT8_C(173), UINT8_C(164), UINT8_C( 57), UINT8_C( 56), UINT8_C( 45), UINT8_C(  6) },
      { UINT8_C(160), UINT8_C(237), UINT8_C( 60), UINT8_C(217), UINT8_C( 11), UINT8_C(152), UINT8_C( 88), UINT8_C(181),
        UINT8_C( 47), UINT8_C( 58), UINT8_C(141), UINT8_C( 26), UINT8_C( 71), UINT8_C(133), UINT8_C(227), UINT8_C(224),
        UINT8_C(209), UINT8_C(175), UINT8_C(106), UINT8_C(147), UINT8_C( 70), UINT8_C(174), UINT8_C(100), UINT8_C(197),
        UINT8_C(160), UINT8_C( 17), UINT8_C(105), UINT8_C(217), UINT8_C( 73), UINT8_C(151), UINT8_C(223), UINT8_C(234),
        UINT8_C(132), UINT8_C( 28), UINT8_C(195), UINT8_C(144), UINT8_C(180), UINT8_C( 27), UINT8_C( 69), UINT8_C(227),
        UINT8_C( 86), UINT8_C(210), UINT8_C(254), UINT8_C(157), UINT8_C( 87), UINT8_C(225), UINT8_C(125), UINT8_C( 41),
        UINT8_C(144), UINT8_C(231), UINT8_C(188), UINT8_C(215), UINT8_C(149), UINT8_C( 32), UINT8_C(156), UINT8_C( 53),
        UINT8_C( 49), UINT8_C(  5), UINT8_C( 15), UINT8_C(123), UINT8_C(156), UINT8_C(238), UINT8_C(101), UINT8_C( 33) },
      UINT64_C( 1121705553518725444) },
    { UINT64_C(11142739203202295818),
      { UINT8_C(200), UINT8_C(160), UINT8_C( 55), UINT8_C( 32), UINT8_C(130), UINT8_C(180), UINT8_C( 73), UINT8_C( 18),
        UINT8_C(155), UINT8_C(  5), UINT8_C(233), UINT8_C( 49), UINT8_C( 38), UINT8_C(133), UINT8_C(102), UINT8_C( 87),
        UINT8_C(139), UINT8_C(117), UINT8_C(210), UINT8_C( 39), UINT8_C(100), UINT8_C( 55), UINT8_C( 72), UINT8_C(110),
        UINT8_C( 96), UINT8_C(249), UINT8_C( 45), UINT8_C(164), UINT8_C(239), UINT8_C(208), UINT8_C( 62), UINT8_C(184),
        UINT8_C(112), UINT8_C(117), UINT8_C(216), UINT8_C(242), UINT8_C( 41), UINT8_C( 33), UINT8_C(  5), UINT8_C(196),
        UINT8_C( 38), UINT8_C(238), UINT8_C(245), UINT8_C( 76), UINT8_C(116), UINT8_C( 92), UINT8_C(164),    UINT8_MAX,
        UINT8_C(209), UINT8_C(118), UINT8_C( 38), UINT8_C( 53), UINT8_C(174), UINT8_C(111), UINT8_C(164), UINT8_C( 14),
        UINT8_C(104), UINT8_C(209), UINT8_C(178), UINT8_C( 88), UINT8_C(161), UINT8_C(240), UINT8_C( 16), UINT8_C( 18) },
      { UINT8_C(101), UINT8_C(232), UINT8_C(  4), UINT8_C(142), UINT8_C(  9), UINT8_C(  9), UINT8_C( 82), UINT8_C( 47),
        UINT8_C(248), UINT8_C( 72), UINT8_C(124), UINT8_C(108), UINT8_C(164), UINT8_C( 32), UINT8_C(107), UINT8_C(117),
        UINT8_C(150), UINT8_C(145), UINT8_C(171), UINT8_C( 68), UINT8_C(  0), UINT8_C( 79), UINT8_C( 82), UINT8_C(105),
        UINT8_C( 32), UINT8_C(  4), UINT8_C(193), UINT8_C(194), UINT8_C(244), UINT8_C(209), UINT8_C(212), UINT8_C( 89),
        UINT8_C(185), UINT8_C(216), UINT8_C(231), UINT8_C(194), UINT8_C(226), UINT8_C( 58), UINT8_C(241), UINT8_C(218),
        UINT8_C(130), UINT8_C(109), UINT8_C( 70), UINT8_C( 38), UINT8_C(141), UINT8_C(177), UINT8_C(155), UINT8_C( 36),
        UINT8_C( 66), UINT8_C( 70), UINT8_C(104), UINT8_C( 67), UINT8_C(149), UINT8_C(187), UINT8_C(172), UINT8_C(182),
        UINT8_C(191), UINT8_C(109), UINT8_C(120), UINT8_C(180), UINT8_C( 62), UINT8_C( 76), UINT8_C( 13), UINT8_C(247) },
      UINT64_C( 1297817348145684480) },
    { UINT64_C(12817431658449466660),
      { UINT8_C( 24), UINT8_C( 38), UINT8_C(215), UINT8_C(165), UINT8_C(215), UINT8_C(114), UINT8_C(201), UINT8_C( 26),
        UINT8_C(185), UINT8_C( 50), UINT8_C( 93), UINT8_C( 78), UINT8_C(237), UINT8_C(  9), UINT8_C(  4), UINT8_C(172),
        UINT8_C(118), UINT8_C(124), UINT8_C( 96), UINT8_C(180), UINT8_C(200), UINT8_C(110), UINT8_C(171), UINT8_C(237),
        UINT8_C( 99), UINT8_C(100), UINT8_C(243), UINT8_C(146), UINT8_C( 14), UINT8_C(212), UINT8_C( 67), UINT8_C( 38),
        UINT8_C(250), UINT8_C( 26), UINT8_C(204), UINT8_C(210), UINT8_C(140), UINT8_C(149), UINT8_C(236), UINT8_C( 69),
        UINT8_C(199), UINT8_C( 73), UINT8_C(148), UINT8_C(180), UINT8_C( 82), UINT8_C(152), UINT8_C( 97), UINT8_C(200),
        UINT8_C( 21), UINT8_C(193), UINT8_C(124), UINT8_C(221), UINT8_C( 47), UINT8_C( 39), UINT8_C(202), UINT8_C(146),
        UINT8_C(139), UINT8_C(190), UINT8_C( 36), UINT8_C(153), UINT8_C(146), UINT8_C(103), UINT8_C(192), UINT8_C(140) },
      { UINT8_C(129), UINT8_C(140), UINT8_C( 94), UINT8_C( 14), UINT8_C( 33), UINT8_C( 74), UINT8_C( 83), UINT8_C(233),
        UINT8_C(147), UINT8_C(231), UINT8_C(157), UINT8_C(229), UINT8_C(128), UINT8_C(254), UINT8_C(173), UINT8_C(149),
        UINT8_C(192), UINT8_C( 41), UINT8_C(114), UINT8_C(239), UINT8_C( 80), UINT8_C( 61), UINT8_C(130), UINT8_C(219),
        UINT8_C(251), UINT8_C(166), UINT8_C(117), UINT8_C(141), UINT8_C( 14), UINT8_C( 53), UINT8_C( 25), UINT8_C(143),
        UINT8_C(193), UINT8_C(120), UINT8_C(157), UINT8_C(226), UINT8_C(194), UINT8_C(241), UINT8_C(203), UINT8_C( 86),
        UINT8_C(216), UINT8_C(105), UINT8_C( 59), UINT8_C( 88), UINT8_C(103), UINT8_C(233), UINT8_C(237), UINT8_C( 39),
        UINT8_C( 18), UINT8_C( 96), UINT8_C( 23), UINT8_C( 99), UINT8_C(157), UINT8_C(153), UINT8_C( 62), UINT8_C(152),
        UINT8_C( 63), UINT8_C(179), UINT8_C( 37), UINT8_C( 77), UINT8_C(232), UINT8_C( 62), UINT8_C(221), UINT8_C(169) },
      UINT64_C( 2396064556895998244) },
    { UINT64_C( 4958277833870572214),
      { UINT8_C(192), UINT8_C( 10), UINT8_C(156), UINT8_C( 40), UINT8_C(243), UINT8_C(138), UINT8_C( 79), UINT8_C(  6),
        UINT8_C(234), UINT8_C(102), UINT8_C(105), UINT8_C(135),    UINT8_MAX, UINT8_C(167), UINT8_C( 31), UINT8_C( 63),
        UINT8_C( 91), UINT8_C( 68), UINT8_C(140), UINT8_C( 67), UINT8_C(130), UINT8_C(105), UINT8_C(237), UINT8_C( 57),
        UINT8_C(228), UINT8_C(121), UINT8_C(178), UINT8_C( 79), UINT8_C(208), UINT8_C(129), UINT8_C(147), UINT8_C(145),
        UINT8_C(139), UINT8_C( 48), UINT8_C(185), UINT8_C(127), UINT8_C(186), UINT8_C(  8), UINT8_C(133), UINT8_C(164),
        UINT8_C(111), UINT8_C(238), UINT8_C( 43), UINT8_C(110), UINT8_C(149), UINT8_C( 74), UINT8_C(173), UINT8_C(240),
        UINT8_C(142), UINT8_C( 58), UINT8_C( 52), UINT8_C( 16), UINT8_C(163), UINT8_C( 33), UINT8_C( 73), UINT8_C(135),
        UINT8_C(154), UINT8_C(251), UINT8_C(215), UINT8_C(106), UINT8_C(124), UINT8_C(106), UINT8_C(251), UINT8_C(  8) },
      { UINT8_C(154), UINT8_C(180), UINT8_C(135), UINT8_C( 84), UINT8_C(189), UINT8_C( 12), UINT8_C(248), UINT8_C( 44),
        UINT8_C(250), UINT8_C( 35), UINT8_C(154), UINT8_C(143), UINT8_C(109), UINT8_C( 72), UINT8_C(128), UINT8_C(251),
        UINT8_C(130), UINT8_C(180), UINT8_C( 12), UINT8_C( 37), UINT8_C(213), UINT8_C( 85), UINT8_C(173), UINT8_C(111),
        UINT8_C( 81), UINT8_C(132), UINT8_C(217), UINT8_C(205), UINT8_C(238), UINT8_C(213), UINT8_C(213), UINT8_C(137),
        UINT8_C(137), UINT8_C( 92), UINT8_C(221), UINT8_C( 70), UINT8_C(104), UINT8_C(214), UINT8_C(114), UINT8_C( 98),
        UINT8_C(249), UINT8_C( 13), UINT8_C(242), UINT8_C(103), UINT8_C( 85), UINT8_C(114), UINT8_C( 98), UINT8_C(215),
        UINT8_C( 38), UINT8_C(110), UINT8_C(252), UINT8_C(251), UINT8_C(196), UINT8_C(169), UINT8_C(106), UINT8_C( 21),
        UINT8_C( 45), UINT8_C( 67), UINT8_C(226), UINT8_C( 28), UINT8_C( 24), UINT8_C(184), UINT8_C(165), UINT8_C(162) },
      UINT64_C( 4648086763926729268) },
    { UINT64_C( 5971592066544206356),
      { UINT8_C(104), UINT8_C(209), UINT8_C(185), UINT8_C(189), UINT8_C( 67), UINT8_C( 27), UINT8_C(148), UINT8_C(105),
        UINT8_C(138), UINT8_C(144), UINT8_C(100), UINT8_C( 78), UINT8_C( 58), UINT8_C(206), UINT8_C( 99), UINT8_C(103),
        UINT8_C( 18), UINT8_C( 69), UINT8_C(131), UINT8_C( 42), UINT8_C(253), UINT8_C( 40), UINT8_C(204), UINT8_C( 18),
        UINT8_C(171), UINT8_C(181), UINT8_C(143), UINT8_C(  3), UINT8_C( 16), UINT8_C(110), UINT8_C( 85), UINT8_C(120),
        UINT8_C( 64), UINT8_C( 14), UINT8_C( 53), UINT8_C(131), UINT8_C( 42), UINT8_C(201), UINT8_C(237), UINT8_C(180),
        UINT8_C( 89), UINT8_C( 81), UINT8_C(  2), UINT8_C(147), UINT8_C( 32), UINT8_C(101), UINT8_C(251), UINT8_C( 50),
        UINT8_C(170), UINT8_C(126), UINT8_C( 92), UINT8_C(168), UINT8_C(167), UINT8_C( 41), UINT8_C(186), UINT8_C( 82),
        UINT8_C(222), UINT8_C( 73), UINT8_C( 85), UINT8_C(238), UINT8_C(183), UINT8_C(171), UINT8_C(102), UINT8_C(247) },
      { UINT8_C(185), UINT8_C(155), UINT8_C(123), UINT8_C(227), UINT8_C(100), UINT8_C(104), UINT8_C(151), UINT8_C(189),
        UINT8_C(185), UINT8_C(153), UINT8_C( 81), UINT8_C(217), UINT8_C(254), UINT8_C( 76), UINT8_C( 11), UINT8_C(169),
        UINT8_C(202), UINT8_C(104), UINT8_C( 81), UINT8_C(113), UINT8_C(145), UINT8_C( 11), UINT8_C(195), UINT8_C(111),
        UINT8_C( 84), UINT8_C( 25), UINT8_C( 93), UINT8_C( 11), UINT8_C(196), UINT8_C(195), UINT8_C(  3), UINT8_C(125),
        UINT8_C( 94), UINT8_C(126), UINT8_C( 97), UINT8_C(194), UINT8_C(230), UINT8_C(248), UINT8_C(127), UINT8_C(159),
        UINT8_C(146), UINT8_C(208), UINT8_C(121), UINT8_C(144), UINT8_C( 28), UINT8_C(132), UINT8_C( 57), UINT8_C(231),
        UINT8_C(236), UINT8_C(138), UINT8_C( 88), UINT8_C(125), UINT8_C(149), UINT8_C( 28), UINT8_C(236), UINT8_C(233),
        UINT8_C( 53), UINT8_C( 73), UINT8_C(245), UINT8_C(249), UINT8_C( 12), UINT8_C(248), UINT8_C(118), UINT8_C(106) },
      UINT64_C( 1160899837019815940) },
    { UINT64_C( 7132484445770930038),
      { UINT8_C(124), UINT8_C(116), UINT8_C(242), UINT8_C(153), UINT8_C(249), UINT8_C( 44), UINT8_C(128), UINT8_C(229),
        UINT8_C(182), UINT8_C(216), UINT8_C( 99), UINT8_C( 76), UINT8_C(244), UINT8_C( 79), UINT8_C( 53), UINT8_C( 41),
        UINT8_C(153), UINT8_C( 42), UINT8_C( 34), UINT8_C(165), UINT8_C( 34), UINT8_C(153), UINT8_C( 16), UINT8_C(152),
        UINT8_C(112), UINT8_C( 60), UINT8_C(244), UINT8_C( 64), UINT8_C(232), UINT8_C(240), UINT8_C(162), UINT8_C(101),
        UINT8_C(100), UINT8_C(149), UINT8_C(254), UINT8_C( 93), UINT8_C(193), UINT8_C(126), UINT8_C( 67), UINT8_C(119),
        UINT8_C( 86), UINT8_C(166), UINT8_C(195), UINT8_C( 75), UINT8_C(245), UINT8_C(249), UINT8_C(116), UINT8_C(142),
        UINT8_C( 35), UINT8_C(151), UINT8_C( 52), UINT8_C( 70), UINT8_C( 48), UINT8_C( 68), UINT8_C(222), UINT8_C(160),
        UINT8_C(128), UINT8_C(211), UINT8_C(225), UINT8_C(105), UINT8_C(195), UINT8_C(131), UINT8_C(206), UINT8_C( 39) },
      { UINT8_C( 24), UINT8_C(204), UINT8_C(133), UINT8_C(217), UINT8_C( 74), UINT8_C(200), UINT8_C( 81), UINT8_C(160),
        UINT8_C(110), UINT8_C( 20), UINT8_C(235), UINT8_C( 99), UINT8_C( 13), UINT8_C( 96), UINT8_C(242), UINT8_C( 49),
        UINT8_C(247), UINT8_C( 38), UINT8_C(119), UINT8_C( 39), UINT8_C(106), UINT8_C( 85), UINT8_C(199), UINT8_C(234),
        UINT8_C( 40), UINT8_C(168), UINT8_C( 83), UINT8_C(235), UINT8_C( 44), UINT8_C( 33), UINT8_C( 19), UINT8_C( 68),
        UINT8_C(237), UINT8_C(152), UINT8_C( 30), UINT8_C( 55), UINT8_C( 96), UINT8_C(111), UINT8_C(216), UINT8_C(206),
        UINT8_C(131), UINT8_C(195), UINT8_C( 49), UINT8_C(145), UINT8_C( 35), UINT8_C( 35), UINT8_C(194), UINT8_C( 26),
        UINT8_C( 73), UINT8_C( 57), UINT8_C( 65), UINT8_C(179), UINT8_C(142), UINT8_C(  9), UINT8_C(158), UINT8_C(183),
        UINT8_C(177), UINT8_C(241), UINT8_C(162), UINT8_C(221), UINT8_C( 19), UINT8_C(181), UINT8_C( 34), UINT8_C(  0) },
      UINT64_C( 4639450956183376724) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epu8_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epu8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i b = easysimd_test_x86_random_i8x64();
    easysimd__mmask64 r = easysimd_mm512_mask_cmpgt_epu8_mask(k, a, b);

    easysimd_test_x86_write_mmask64(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpgt_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    uint16_t a[32];
    uint16_t b[32];
    easysimd__mmask32 r;
  } test_vec[8] = {
    { UINT32_C(4282923555),
      { UINT16_C(51465), UINT16_C(16701), UINT16_C(59314), UINT16_C(33475), UINT16_C( 6006), UINT16_C(26006), UINT16_C(56693), UINT16_C(39330),
        UINT16_C(26446), UINT16_C(20419), UINT16_C(65034), UINT16_C(34978), UINT16_C(18620), UINT16_C(57241), UINT16_C(57986), UINT16_C(36062),
        UINT16_C( 7083), UINT16_C(24013), UINT16_C(37123), UINT16_C(31200), UINT16_C(30376), UINT16_C( 7646), UINT16_C(33107), UINT16_C(41399),
        UINT16_C(31464), UINT16_C(62193), UINT16_C(37753), UINT16_C(13691), UINT16_C( 5340), UINT16_C(24084), UINT16_C(62454), UINT16_C(41706) },
      { UINT16_C(47118), UINT16_C( 4607), UINT16_C(57161), UINT16_C(61835), UINT16_C(26966), UINT16_C(43278), UINT16_C(50666), UINT16_C(53835),
        UINT16_C(15424), UINT16_C(47557), UINT16_C(16591), UINT16_C(44014), UINT16_C(  596), UINT16_C(19210), UINT16_C(62709), UINT16_C( 1261),
        UINT16_C(60588), UINT16_C(62741), UINT16_C(41164), UINT16_C( 8934), UINT16_C(62730), UINT16_C(62667), UINT16_C( 5818), UINT16_C(64199),
        UINT16_C(35922), UINT16_C( 8883), UINT16_C(41420), UINT16_C( 8397), UINT16_C(55204), UINT16_C(39275), UINT16_C(22732), UINT16_C(30877) },
      UINT32_C(3393728515) },
    { UINT32_C( 292467525),
      { UINT16_C(21587), UINT16_C(23859), UINT16_C(65097), UINT16_C( 1106), UINT16_C( 6421), UINT16_C(26622), UINT16_C(45733), UINT16_C(29065),
        UINT16_C(22355), UINT16_C(63377), UINT16_C(64814), UINT16_C(64145), UINT16_C(11861), UINT16_C(39539), UINT16_C(57825), UINT16_C(13739),
        UINT16_C(56885), UINT16_C(32658), UINT16_C(58589), UINT16_C(62083), UINT16_C(33277), UINT16_C(41561), UINT16_C(58163), UINT16_C(34579),
        UINT16_C(42298), UINT16_C(26750), UINT16_C( 4002), UINT16_C(63331), UINT16_C(54846), UINT16_C( 8082), UINT16_C(15799), UINT16_C(60500) },
      { UINT16_C(59164), UINT16_C(63851), UINT16_C(61131), UINT16_C(51691), UINT16_C(17520), UINT16_C(41835), UINT16_C(32551), UINT16_C(24874),
        UINT16_C(43300), UINT16_C(50890), UINT16_C(11704), UINT16_C(63165), UINT16_C(20227), UINT16_C(47638), UINT16_C(27277), UINT16_C(43430),
        UINT16_C( 4689), UINT16_C( 7586), UINT16_C(36096), UINT16_C(28902), UINT16_C(20945), UINT16_C(63764), UINT16_C(16080), UINT16_C(62554),
        UINT16_C( 9447), UINT16_C(41146), UINT16_C(30801), UINT16_C(21654), UINT16_C(44231), UINT16_C(21518), UINT16_C(46359), UINT16_C(26877) },
      UINT32_C( 290325060) },
    { UINT32_C(3347423175),
      { UINT16_C(27436), UINT16_C(65080), UINT16_C(19645), UINT16_C(36343), UINT16_C(20874), UINT16_C(29314), UINT16_C(15478), UINT16_C(50962),
        UINT16_C(43188), UINT16_C(31772), UINT16_C(10837), UINT16_C(27856), UINT16_C(52959), UINT16_C(42708), UINT16_C(23149), UINT16_C(39534),
        UINT16_C(42693), UINT16_C(33432), UINT16_C(36850), UINT16_C(31760), UINT16_C(37600), UINT16_C(22254), UINT16_C(  206), UINT16_C(33566),
        UINT16_C(15017), UINT16_C(65279), UINT16_C(53092), UINT16_C(17514), UINT16_C(16029), UINT16_C( 3050), UINT16_C(22680), UINT16_C(24229) },
      { UINT16_C(15870), UINT16_C(61664), UINT16_C(61644), UINT16_C(44141), UINT16_C(23426), UINT16_C(20739), UINT16_C( 8540), UINT16_C( 1492),
        UINT16_C(54107), UINT16_C(48899), UINT16_C(28066), UINT16_C(16387), UINT16_C(61099), UINT16_C(17483), UINT16_C(61510), UINT16_C(17826),
        UINT16_C(33325), UINT16_C(63797), UINT16_C(41587), UINT16_C(62885), UINT16_C(43262), UINT16_C(23110), UINT16_C( 6857), UINT16_C( 9311),
        UINT16_C(25325), UINT16_C(37092), UINT16_C(59343), UINT16_C(31440), UINT16_C( 7125), UINT16_C( 7358), UINT16_C(24587), UINT16_C(14433) },
      UINT32_C(2189527235) },
    { UINT32_C(1446090467),
      { UINT16_C(54841), UINT16_C(14155), UINT16_C(37503), UINT16_C(18577), UINT16_C(61612), UINT16_C(39533), UINT16_C(20818), UINT16_C( 8490),
        UINT16_C(64056), UINT16_C( 3739), UINT16_C(23061), UINT16_C( 8234), UINT16_C(35770), UINT16_C(40280), UINT16_C(35105), UINT16_C(23283),
        UINT16_C(16223), UINT16_C(56977), UINT16_C( 8913), UINT16_C(32039), UINT16_C(37906), UINT16_C(25623), UINT16_C(16869), UINT16_C( 7557),
        UINT16_C( 8507), UINT16_C(20523), UINT16_C(21883), UINT16_C(13680), UINT16_C(51424), UINT16_C(  723), UINT16_C(50769), UINT16_C(45404) },
      { UINT16_C(60933), UINT16_C(54927), UINT16_C(46608), UINT16_C( 9044), UINT16_C(27466), UINT16_C(12167), UINT16_C( 3501), UINT16_C(59469),
        UINT16_C(30766), UINT16_C(43321), UINT16_C(43470), UINT16_C(44766), UINT16_C(45426), UINT16_C(50096), UINT16_C( 3448), UINT16_C(32116),
        UINT16_C( 1275), UINT16_C( 2900), UINT16_C(43194), UINT16_C( 1326), UINT16_C(46611), UINT16_C(49204), UINT16_C(33219), UINT16_C(61865),
        UINT16_C(58106), UINT16_C(51354), UINT16_C(30859), UINT16_C(64886), UINT16_C(10026), UINT16_C(41665), UINT16_C(13620), UINT16_C(12063) },
      UINT32_C(1342242912) },
    { UINT32_C(4097471289),
      { UINT16_C(26907), UINT16_C(12281), UINT16_C(11551), UINT16_C(58095), UINT16_C(39087), UINT16_C(43475), UINT16_C(28026), UINT16_C( 1649),
        UINT16_C(59365), UINT16_C( 3843), UINT16_C(50190), UINT16_C(17073), UINT16_C(53754), UINT16_C(13169), UINT16_C(44100), UINT16_C(24615),
        UINT16_C( 8213), UINT16_C(13455), UINT16_C(32334), UINT16_C(64790), UINT16_C(59671), UINT16_C(37286), UINT16_C( 5974), UINT16_C(15255),
        UINT16_C(39934), UINT16_C( 3403), UINT16_C(64607), UINT16_C(22863), UINT16_C(49613), UINT16_C( 4749), UINT16_C(46189), UINT16_C(33394) },
      { UINT16_C(  469), UINT16_C( 9142), UINT16_C(52351), UINT16_C(38432), UINT16_C(50869), UINT16_C( 2856), UINT16_C(49117), UINT16_C(56134),
        UINT16_C(37210), UINT16_C(47848), UINT16_C(14478), UINT16_C(23315), UINT16_C(41209), UINT16_C(26221), UINT16_C(57173), UINT16_C(10984),
        UINT16_C(40672), UINT16_C(24653), UINT16_C(28010), UINT16_C( 8182), UINT16_C( 7731), UINT16_C( 4138), UINT16_C(28894), UINT16_C(14571),
        UINT16_C(54274), UINT16_C(37106), UINT16_C( 1548), UINT16_C( 1515), UINT16_C(22950), UINT16_C(64363), UINT16_C(21304), UINT16_C( 6437) },
      UINT32_C(3560444201) },
    { UINT32_C(1534685937),
      { UINT16_C(28639), UINT16_C( 4730), UINT16_C(42126), UINT16_C(27682), UINT16_C( 3604), UINT16_C( 5796), UINT16_C(38882), UINT16_C(61094),
        UINT16_C(37533), UINT16_C(17395), UINT16_C(24299), UINT16_C( 9023), UINT16_C(25777), UINT16_C(41532), UINT16_C(46551), UINT16_C(46845),
        UINT16_C(30501), UINT16_C(46025), UINT16_C(60187), UINT16_C(12063), UINT16_C(50169), UINT16_C(56134), UINT16_C(60506), UINT16_C(63433),
        UINT16_C(48254), UINT16_C(26939), UINT16_C(31258), UINT16_C(52109), UINT16_C(51678), UINT16_C(46445), UINT16_C(27263), UINT16_C(42092) },
      { UINT16_C(13793), UINT16_C(64599), UINT16_C(30240), UINT16_C( 6700), UINT16_C(29241), UINT16_C(38133), UINT16_C(48990), UINT16_C(56715),
        UINT16_C(50811), UINT16_C(38470), UINT16_C(54080), UINT16_C( 8033), UINT16_C(53149), UINT16_C( 7380), UINT16_C(16441), UINT16_C( 7104),
        UINT16_C( 6005), UINT16_C(38423), UINT16_C(17293), UINT16_C(50864), UINT16_C(42421), UINT16_C( 5210), UINT16_C(58980), UINT16_C(57585),
        UINT16_C(14252), UINT16_C(60790), UINT16_C(55051), UINT16_C(43020), UINT16_C(57510), UINT16_C(57540), UINT16_C(33825), UINT16_C(38651) },
      UINT32_C( 158425217) },
    { UINT32_C( 673976987),
      { UINT16_C(56406), UINT16_C( 3054), UINT16_C(18818), UINT16_C(58911), UINT16_C( 4143), UINT16_C(56262), UINT16_C(15432), UINT16_C(21448),
        UINT16_C(54292), UINT16_C(47867), UINT16_C(49077), UINT16_C(54938), UINT16_C(38211), UINT16_C(56940), UINT16_C(39336), UINT16_C(65030),
        UINT16_C(62581), UINT16_C(63241), UINT16_C(10557), UINT16_C(27870), UINT16_C(42041), UINT16_C(33096), UINT16_C( 4321), UINT16_C(62932),
        UINT16_C(53221), UINT16_C(39599), UINT16_C(19086), UINT16_C(53616), UINT16_C(56543), UINT16_C(34735), UINT16_C(46453), UINT16_C(60293) },
      { UINT16_C(36778), UINT16_C(59362), UINT16_C(49336), UINT16_C(61780), UINT16_C(40037), UINT16_C(18035), UINT16_C(18348), UINT16_C(37179),
        UINT16_C(59927), UINT16_C(42283), UINT16_C(39732), UINT16_C( 5239), UINT16_C( 9848), UINT16_C(60827), UINT16_C( 8668), UINT16_C(34520),
        UINT16_C(48048), UINT16_C(26733), UINT16_C(49531), UINT16_C(57433), UINT16_C(52317), UINT16_C( 2598), UINT16_C(24852), UINT16_C(11163),
        UINT16_C(51020), UINT16_C(32976), UINT16_C(18274), UINT16_C(55956), UINT16_C(12398), UINT16_C(19144), UINT16_C(41041), UINT16_C(  464) },
      UINT32_C( 538972673) },
    { UINT32_C(3613998427),
      { UINT16_C(49919), UINT16_C(23735), UINT16_C(56975), UINT16_C(41830), UINT16_C(  575), UINT16_C(35790), UINT16_C(40649), UINT16_C(11020),
        UINT16_C(41190), UINT16_C(21510), UINT16_C(52944), UINT16_C( 8606), UINT16_C(28270), UINT16_C(51746), UINT16_C(35755), UINT16_C(43681),
        UINT16_C(22606), UINT16_C(56583), UINT16_C(27958), UINT16_C(30336), UINT16_C(20079), UINT16_C(14337), UINT16_C( 3564), UINT16_C(53860),
        UINT16_C(27310), UINT16_C(32294), UINT16_C(50232), UINT16_C(42656), UINT16_C(49714), UINT16_C(56944), UINT16_C( 4430), UINT16_C(40072) },
      { UINT16_C(36714), UINT16_C(41081), UINT16_C(63997), UINT16_C(27670), UINT16_C( 6215), UINT16_C(13221), UINT16_C( 2341), UINT16_C(54022),
        UINT16_C(11379), UINT16_C(43858), UINT16_C(62193), UINT16_C( 9041), UINT16_C(49844), UINT16_C(  513), UINT16_C(35539), UINT16_C(15774),
        UINT16_C( 5913), UINT16_C( 5854), UINT16_C(62480), UINT16_C(22403), UINT16_C(10252), UINT16_C(12939), UINT16_C(37169), UINT16_C(41989),
        UINT16_C(22461), UINT16_C(44623), UINT16_C(41033), UINT16_C(65234), UINT16_C(54114), UINT16_C(13824), UINT16_C(40797), UINT16_C(30579) },
      UINT32_C(2234065225) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epu16_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epu16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 r = easysimd_mm512_mask_cmpgt_epu16_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpgt_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    uint32_t a[16];
    uint32_t b[16];
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(56869),
      { UINT32_C(1564212707), UINT32_C(2222052664), UINT32_C(1365516395), UINT32_C(3884603181), UINT32_C( 934391363), UINT32_C( 551511852), UINT32_C( 501697743), UINT32_C(3740994810),
        UINT32_C(1111176714), UINT32_C(2009508875), UINT32_C(4207422412), UINT32_C(3319878274), UINT32_C( 771527168), UINT32_C(3377323002), UINT32_C( 585512232), UINT32_C(1258348865) },
      { UINT32_C( 596523800), UINT32_C(3030013159), UINT32_C(  28205951), UINT32_C(3066531765), UINT32_C( 467911457), UINT32_C(3353686175), UINT32_C(2817117029), UINT32_C(3321031341),
        UINT32_C( 233340965), UINT32_C(1421968341), UINT32_C(2606133222), UINT32_C( 525409790), UINT32_C(2134586592), UINT32_C(3393593445), UINT32_C(2557554923), UINT32_C(1063085082) },
      UINT16_C(36357) },
    { UINT16_C(18148),
      { UINT32_C( 231323980), UINT32_C(1685892878), UINT32_C(2625731146), UINT32_C(3587203482), UINT32_C( 670381537), UINT32_C(1901584384), UINT32_C(3604312441), UINT32_C(4246518449),
        UINT32_C(2165040499), UINT32_C(3756361620), UINT32_C(2625332738), UINT32_C(2842774472), UINT32_C(2177918593), UINT32_C(3220318278), UINT32_C(1251395737), UINT32_C(4115182210) },
      { UINT32_C( 746017688), UINT32_C(3691732186), UINT32_C(2339931843), UINT32_C(1395976658), UINT32_C(2530477135), UINT32_C(3327510060), UINT32_C( 269544334), UINT32_C( 906385566),
        UINT32_C(2254601643), UINT32_C(2623696601), UINT32_C(3324500980), UINT32_C( 337206212), UINT32_C(2360012128), UINT32_C(1112735668), UINT32_C(2303943659), UINT32_C(1740593595) },
      UINT16_C(  708) },
    { UINT16_C( 8662),
      { UINT32_C(1334816749), UINT32_C(1932166220), UINT32_C(1691348810), UINT32_C(2907778819), UINT32_C( 246154683), UINT32_C(2607912776), UINT32_C(3757321248), UINT32_C(2147601043),
        UINT32_C(3318780025), UINT32_C(1597635092), UINT32_C(3972204777), UINT32_C(4070118455), UINT32_C(1644185114), UINT32_C(4278022621), UINT32_C( 836628894), UINT32_C( 884137659) },
      { UINT32_C(2214232687), UINT32_C(1726100348), UINT32_C(1918018875), UINT32_C(3563383994), UINT32_C( 255223090), UINT32_C(1947022294), UINT32_C(3752258340), UINT32_C( 940792009),
        UINT32_C(1455164890), UINT32_C(2075958848), UINT32_C(4276948804), UINT32_C( 768758523), UINT32_C(2369521847), UINT32_C(1593985595), UINT32_C(4265519157), UINT32_C(3661058304) },
      UINT16_C( 8642) },
    { UINT16_C(62303),
      { UINT32_C(3985743664), UINT32_C( 150787355), UINT32_C(2774202323), UINT32_C(1621954852), UINT32_C(2712332447), UINT32_C(2236211015), UINT32_C( 366430686), UINT32_C(1393047075),
        UINT32_C(4030765525), UINT32_C(1123630191), UINT32_C(1474843443), UINT32_C(  79205733), UINT32_C(3315950206), UINT32_C( 541847106), UINT32_C(1513431607), UINT32_C( 766328152) },
      { UINT32_C(1159655126), UINT32_C(1585977130), UINT32_C(3484774506), UINT32_C(2211671301), UINT32_C( 306804944), UINT32_C(2637337702), UINT32_C( 251094966), UINT32_C(2050729380),
        UINT32_C(3183499667), UINT32_C(3675998320), UINT32_C(3182088632), UINT32_C( 239173182), UINT32_C(1562413814), UINT32_C(3573174814), UINT32_C(1575154361), UINT32_C( 718806423) },
      UINT16_C(37201) },
    { UINT16_C(39031),
      { UINT32_C(  65071079), UINT32_C(1842649282), UINT32_C(2531988053), UINT32_C(1092673825), UINT32_C( 966016574), UINT32_C(4096478482), UINT32_C(2199044779), UINT32_C(3541797356),
        UINT32_C( 869727088), UINT32_C(3902843539), UINT32_C(3732900797), UINT32_C(2870976364), UINT32_C(4041520094), UINT32_C(2900692737), UINT32_C(3174037457), UINT32_C(4035988096) },
      { UINT32_C(3626198853), UINT32_C(3468739345), UINT32_C(3148627790), UINT32_C(3177630943), UINT32_C(2158840447), UINT32_C( 707564121), UINT32_C( 166157193), UINT32_C(3942283173),
        UINT32_C(4039253470), UINT32_C( 801014753), UINT32_C(2716494786), UINT32_C(3059634231), UINT32_C(4080471194), UINT32_C( 673014686), UINT32_C(1680934079), UINT32_C(1515137916) },
      UINT16_C(32864) },
    { UINT16_C( 4425),
      { UINT32_C( 160705098), UINT32_C(1148475225), UINT32_C(1469361144), UINT32_C(2556637025), UINT32_C(1090257186), UINT32_C(1531231017), UINT32_C(1854324767), UINT32_C(1702940443),
        UINT32_C(1399723257), UINT32_C(1687675499), UINT32_C(4022021005), UINT32_C(2106007130), UINT32_C(1237156639), UINT32_C(1587806526), UINT32_C(3721145026), UINT32_C(4081208570) },
      { UINT32_C(3427184993), UINT32_C( 556850579), UINT32_C(1662053129), UINT32_C( 702584585), UINT32_C(1500683547), UINT32_C(1639388831), UINT32_C( 977175616), UINT32_C( 825131216),
        UINT32_C(3321787441), UINT32_C(1541811794), UINT32_C( 599717402), UINT32_C(2823593869), UINT32_C(3674390076), UINT32_C( 356301268), UINT32_C( 240089661), UINT32_C( 742358523) },
      UINT16_C(   72) },
    { UINT16_C(15858),
      { UINT32_C(3614196977), UINT32_C(1573750431), UINT32_C(4143733673), UINT32_C(  95697155), UINT32_C(1371506964), UINT32_C(3989568670), UINT32_C(1248577034), UINT32_C(3834076659),
        UINT32_C(1086124961), UINT32_C( 597526905), UINT32_C(3893992164), UINT32_C(3891121619), UINT32_C(4097354838), UINT32_C(3017933993), UINT32_C(3187494346), UINT32_C(1268942250) },
      { UINT32_C(4069219960), UINT32_C(3423938791), UINT32_C(2511613634), UINT32_C(1383965179), UINT32_C(4148606286), UINT32_C(2225744057), UINT32_C( 524396661), UINT32_C(2775245613),
        UINT32_C( 697824577), UINT32_C(3774196766), UINT32_C(3598100954), UINT32_C(2552820554), UINT32_C(1653567144), UINT32_C( 216414871), UINT32_C( 254552034), UINT32_C(1286903307) },
      UINT16_C(15840) },
    { UINT16_C(19596),
      { UINT32_C(1794681461), UINT32_C(   1299338), UINT32_C(3522387625), UINT32_C(2252315894), UINT32_C(3837843198), UINT32_C( 252420835), UINT32_C(1705318065), UINT32_C(3635491171),
        UINT32_C(1715710683), UINT32_C( 644241021), UINT32_C(2885114548), UINT32_C(4096866038), UINT32_C(4040749325), UINT32_C(1157620627), UINT32_C(1571398906), UINT32_C(2973064150) },
      { UINT32_C(2182576133), UINT32_C(2208857807), UINT32_C(3459162072), UINT32_C(3804389333), UINT32_C(3822230096), UINT32_C(1999098237), UINT32_C(1289015670), UINT32_C( 838666796),
        UINT32_C(1370690946), UINT32_C(1809144723), UINT32_C(3476620282), UINT32_C(3014851427), UINT32_C( 311919765), UINT32_C(3431644758), UINT32_C(3138936463), UINT32_C(3958117736) },
      UINT16_C( 2180) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epu32_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epu32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 r = easysimd_mm512_mask_cmpgt_epu32_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpgt_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    uint64_t a[8];
    uint64_t b[8];
    easysimd__mmask8 r;
  } test_vec[8] = {
    { UINT8_C(109),
      { UINT64_C( 5164517563230593903), UINT64_C( 8174327323478913420), UINT64_C( 5724411643897633593), UINT64_C( 5508013244681157225),
        UINT64_C(12261719716471836807), UINT64_C( 2965696150978683986), UINT64_C( 7169222607433091103), UINT64_C(14291565658254007083) },
      { UINT64_C(13668375850702670567), UINT64_C(16559908206868202167), UINT64_C(17060359693342642020), UINT64_C(16677473732573538413),
        UINT64_C(11996634957115824979), UINT64_C(17350150055390998557), UINT64_C(15949797626070621612), UINT64_C(16870480455571950905) },
      UINT8_C(  0) },
    { UINT8_C( 23),
      { UINT64_C( 1568286372954286347), UINT64_C( 5531143198671480569), UINT64_C(15432033312083488735), UINT64_C(10103077551148458274),
        UINT64_C(16157498085767068416), UINT64_C( 5501350818558172061), UINT64_C(14863960886468154344), UINT64_C(  924400155852846278) },
      { UINT64_C(11107403891005785949), UINT64_C(13619165462685686599), UINT64_C( 7461468020186954770), UINT64_C(17744327225876903977),
        UINT64_C( 2675807637479322250), UINT64_C(15640396531005769851), UINT64_C(15644981397416262421), UINT64_C(15720745030644518299) },
      UINT8_C( 20) },
    { UINT8_C(192),
      { UINT64_C( 5494887991109116262), UINT64_C(15676164568299835805), UINT64_C( 3260995772443345539), UINT64_C(  467907759195387248),
        UINT64_C( 8973438609496218809), UINT64_C(  727190083697089439), UINT64_C( 8309124833291051457), UINT64_C(15928861501330410855) },
      { UINT64_C( 9810989433436691147), UINT64_C( 6616031505125087806), UINT64_C(15705684067712073854), UINT64_C(11717221461627250585),
        UINT64_C(10824811103642301897), UINT64_C( 6594162508265422735), UINT64_C( 7419631789055738028), UINT64_C(10336536054896842746) },
      UINT8_C(192) },
    { UINT8_C(117),
      { UINT64_C( 8773965875570884799), UINT64_C( 8040810378459278271), UINT64_C( 9465620473308820623), UINT64_C( 6916922782392069575),
        UINT64_C(17081122244737092120), UINT64_C(11826573656620229185), UINT64_C(14362370636985556002), UINT64_C( 9984000377719003792) },
      { UINT64_C(13344492763096097046), UINT64_C(11044103328809683083), UINT64_C( 4444129989325441766), UINT64_C( 6651529981034625874),
        UINT64_C( 2327747018499320605), UINT64_C( 2803776917294069722), UINT64_C( 6501429885813936206), UINT64_C(13787327855989794483) },
      UINT8_C(116) },
    { UINT8_C(216),
      { UINT64_C( 6386822947556643831), UINT64_C( 3287557205986440013), UINT64_C( 8878724437087984587), UINT64_C( 9148179215929913774),
        UINT64_C( 4263195734205489456), UINT64_C( 5469106629356863964), UINT64_C( 6491664722017852359), UINT64_C( 4701029110975378743) },
      { UINT64_C( 5400725720573052106), UINT64_C( 8271399113388687357), UINT64_C(15351123643334563844), UINT64_C(12947234204727607130),
        UINT64_C(17873766556029993044), UINT64_C(18229515728396357599), UINT64_C( 9964977440464571095), UINT64_C(  203404901441002806) },
      UINT8_C(128) },
    { UINT8_C( 99),
      { UINT64_C(16104124396068652094), UINT64_C( 1000876651771791309), UINT64_C( 9254271209329940608), UINT64_C(11367983066311986364),
        UINT64_C( 5162735385284649717), UINT64_C(15118363294103272639), UINT64_C(10282410444883233680), UINT64_C(  503473046052331521) },
      { UINT64_C(12580366782227494868), UINT64_C(12758204206384474861), UINT64_C(10136360910573540939), UINT64_C( 2346067871318973047),
        UINT64_C(13240026319916325152), UINT64_C(13385264813764259567), UINT64_C(12912243704128251417), UINT64_C( 2776690199612824859) },
      UINT8_C( 33) },
    { UINT8_C( 35),
      { UINT64_C( 2516286184643966455), UINT64_C( 7184729859247111064), UINT64_C(12230742547191847455), UINT64_C(11708186588183733135),
        UINT64_C(11898660904417257482), UINT64_C( 1776377553951436933), UINT64_C( 2919586662608029263), UINT64_C( 1362053049748632472) },
      { UINT64_C(10598711611647624293), UINT64_C(11308955709306394592), UINT64_C(13151631444654140097), UINT64_C( 7370387086013760558),
        UINT64_C( 7639774617008560484), UINT64_C( 1742107276782852933), UINT64_C(14822743440595592415), UINT64_C(17465606481893371643) },
      UINT8_C( 32) },
    { UINT8_C(165),
      { UINT64_C( 5002540867793085753), UINT64_C(15055731651371663068), UINT64_C( 5699373090630266151), UINT64_C(12696158032122156096),
        UINT64_C(15340743540331733045), UINT64_C( 6646703050522460482), UINT64_C( 3077245980806589891), UINT64_C(10866596979722916262) },
      { UINT64_C( 3032360608444359752), UINT64_C(10605367790373763996), UINT64_C(16111535550287619714), UINT64_C( 9166453853032042045),
        UINT64_C( 2486221767266298733), UINT64_C( 8085123440328251600), UINT64_C( 1647882989018623175), UINT64_C( 1258732610846579394) },
      UINT8_C(129) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpgt_epu64_mask(k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_cmpgt_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 r = easysimd_mm512_mask_cmpgt_epu64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpgt_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpgt_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpgt_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpgt_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpgt_epu64_mask)

EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
