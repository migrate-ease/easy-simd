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
 *   2020      Hidayat Khan <huk2209@gmail.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN test

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/test.h>

static int
test_easysimd_mm_test_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[16];
    const int8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { -INT8_C(  80), -INT8_C(  22),  INT8_C(  36), -INT8_C(  73), -INT8_C(  13),  INT8_C(  93), -INT8_C(  94), -INT8_C(  17),
        -INT8_C(  53), -INT8_C(  68), -INT8_C(  31),  INT8_C(  45), -INT8_C(  88),  INT8_C(  65),  INT8_C(  38), -INT8_C(  77) },
      { -INT8_C(   4), -INT8_C(  92), -INT8_C(  75),  INT8_C(  62), -INT8_C(  79), -INT8_C(  36),  INT8_C( 105),  INT8_C(  14),
        -INT8_C( 113), -INT8_C(  56),  INT8_C(  93), -INT8_C(  55), -INT8_C(  92), -INT8_C(  73),  INT8_C(  89),  INT8_C(  84) },
      UINT16_C(49151) },
    { { -INT8_C(  95),  INT8_C( 126),  INT8_C(  11), -INT8_C( 108), -INT8_C(  37), -INT8_C(  83), -INT8_C( 125), -INT8_C(  90),
         INT8_C( 106),  INT8_C( 101), -INT8_C(  44),  INT8_C(  18), -INT8_C(  90), -INT8_C(   6), -INT8_C(  59), -INT8_C(  93) },
      { -INT8_C(  97),  INT8_C( 122), -INT8_C(  31),  INT8_C(  80),  INT8_C(  86),  INT8_C(  75),  INT8_C(  95), -INT8_C(  27),
         INT8_C(  19), -INT8_C(  68), -INT8_C(  81), -INT8_C(  73),  INT8_C( 115),  INT8_C(   8),  INT8_C(  12),  INT8_C(  21) },
           UINT16_MAX },
    { { -INT8_C( 122),  INT8_C(  23), -INT8_C(  87),  INT8_C(  97), -INT8_C(  59),  INT8_C(  45),  INT8_C(   8),  INT8_C(  47),
        -INT8_C( 110), -INT8_C(  36),  INT8_C(  65),  INT8_C(  56), -INT8_C(  42),  INT8_C(   6), -INT8_C(  37),  INT8_C( 117) },
      {      INT8_MIN, -INT8_C(  67), -INT8_C(  58), -INT8_C(  41),  INT8_C(   8),  INT8_C(  37), -INT8_C(  68),  INT8_C(  27),
        -INT8_C(  31),  INT8_C( 107), -INT8_C(  46),  INT8_C(  84),  INT8_C( 116), -INT8_C(  34),  INT8_C( 105), -INT8_C(   6) },
      UINT16_C(65519) },
    { { -INT8_C(  10),  INT8_C(  19),  INT8_C(  92), -INT8_C(  69),  INT8_C(  64),  INT8_C( 100), -INT8_C(  22), -INT8_C(  46),
         INT8_C(  64),  INT8_C(  43),  INT8_C(  10),  INT8_C(  22),  INT8_C(  49), -INT8_C(  26), -INT8_C( 116), -INT8_C(  79) },
      { -INT8_C(  93),  INT8_C(  82), -INT8_C( 120), -INT8_C(  85),  INT8_C( 119),  INT8_C(  69), -INT8_C(  58),  INT8_C(  88),
        -INT8_C(  80), -INT8_C( 104), -INT8_C(  84),  INT8_C(  36),  INT8_C( 119),  INT8_C(  22),  INT8_C(  31),  INT8_C( 109) },
      UINT16_C(65279) },
    { {  INT8_C(  41),  INT8_C( 123),  INT8_C(  40),  INT8_C( 105), -INT8_C(  33),  INT8_C(  18),  INT8_C(  59),  INT8_C(  31),
         INT8_C(  61),  INT8_C(  69),  INT8_C(  53),  INT8_C( 110),  INT8_C(  43), -INT8_C(  63),  INT8_C(  31), -INT8_C(  50) },
      {  INT8_C(  19), -INT8_C(  88),  INT8_C( 121), -INT8_C( 118), -INT8_C(  19),  INT8_C(  63), -INT8_C(  30), -INT8_C(  99),
        -INT8_C(  40), -INT8_C( 113), -INT8_C(  62),  INT8_C(  79), -INT8_C(  91), -INT8_C(  31), -INT8_C(  68), -INT8_C(  50) },
      UINT16_C(64511) },
    { {  INT8_C(  92), -INT8_C(  28),  INT8_C(  55),  INT8_C(  59), -INT8_C(  10),  INT8_C( 114),  INT8_C(  90),  INT8_C(  51),
        -INT8_C(  73), -INT8_C( 113), -INT8_C(  95), -INT8_C(  29),  INT8_C(  81), -INT8_C(  64), -INT8_C(  79),  INT8_C( 100) },
      {  INT8_C( 104),  INT8_C(  43), -INT8_C(  17),  INT8_C(  85),  INT8_C( 106), -INT8_C(  47), -INT8_C(  13),  INT8_C(  66),
         INT8_C(  96), -INT8_C(  75), -INT8_C( 111),  INT8_C(   5), -INT8_C( 106),  INT8_C(  77), -INT8_C(  45), -INT8_C(  14) },
           UINT16_MAX },
    { {  INT8_C(  49),  INT8_C(  10),  INT8_C(  45),  INT8_C(  39),  INT8_C( 124), -INT8_C( 121),  INT8_C(  90),  INT8_C(  52),
         INT8_C(  22), -INT8_C(   5),  INT8_C(  23),  INT8_C( 103), -INT8_C(  68), -INT8_C(  56), -INT8_C(  52),  INT8_C(  36) },
      { -INT8_C(  13), -INT8_C(  69),  INT8_C( 122),  INT8_C(  94), -INT8_C( 116),  INT8_C( 109), -INT8_C(  96), -INT8_C(  19),
         INT8_C(  34),  INT8_C(  50), -INT8_C(  14), -INT8_C(  72),      INT8_MAX, -INT8_C(  58), -INT8_C(  86), -INT8_C(  79) },
      UINT16_C(65471) },
    { { -INT8_C(  48), -INT8_C(  41), -INT8_C(  40),  INT8_C(  77),  INT8_C(  94),  INT8_C(  51), -INT8_C( 127),  INT8_C( 116),
         INT8_C(  46), -INT8_C( 104), -INT8_C(  36), -INT8_C(  22),  INT8_C(  96), -INT8_C(  88),  INT8_C(  15),  INT8_C(  84) },
      {  INT8_C(  99), -INT8_C( 119), -INT8_C(  78), -INT8_C(  17), -INT8_C(  10),  INT8_C(  82), -INT8_C(  36),  INT8_C(  24),
        -INT8_C( 124), -INT8_C(  49), -INT8_C(  48),  INT8_C(   4), -INT8_C( 107),  INT8_C( 122), -INT8_C(  75),  INT8_C( 101) },
      UINT16_C(59391) }

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_test_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_test_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_test_epi8_mask(a, b);

    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_test_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[8];
    const int16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT16_C(  8744), -INT16_C( 22043), -INT16_C( 26037), -INT16_C(   987),  INT16_C( 30069),  INT16_C(  7534),  INT16_C(  9480),  INT16_C( 23813) },
      { -INT16_C(  9116), -INT16_C( 15640),  INT16_C( 10116),  INT16_C( 25671), -INT16_C(  1111),  INT16_C(  7511), -INT16_C( 19180), -INT16_C(  5037) },
         UINT8_MAX },
    { {  INT16_C( 14482), -INT16_C(  8810), -INT16_C( 17454),  INT16_C( 18393),  INT16_C( 18224),  INT16_C( 14693),  INT16_C( 27245), -INT16_C( 11882) },
      {  INT16_C( 32326), -INT16_C( 13677), -INT16_C(  9562),  INT16_C( 20271), -INT16_C( 31018), -INT16_C(  5524), -INT16_C( 16581), -INT16_C( 12841) },
         UINT8_MAX },
    { {  INT16_C( 28152), -INT16_C( 13653), -INT16_C( 31704),  INT16_C( 22546),  INT16_C( 30668),  INT16_C( 14737),  INT16_C( 10209),  INT16_C( 10250) },
      { -INT16_C( 24922),  INT16_C( 19698),  INT16_C(  8568),  INT16_C( 20123),  INT16_C(  1959), -INT16_C(  7623),  INT16_C(  4295), -INT16_C( 16464) },
         UINT8_MAX },
    { {  INT16_C( 23421), -INT16_C( 23159), -INT16_C( 25633), -INT16_C( 21507), -INT16_C( 28910), -INT16_C(  2844), -INT16_C(  4170),  INT16_C( 23580) },
      {  INT16_C(  3725),  INT16_C(  1448),  INT16_C( 17200), -INT16_C( 10412), -INT16_C( 29365),  INT16_C(  4794),  INT16_C( 27293),  INT16_C(  6865) },
         UINT8_MAX },
    { {  INT16_C( 23237), -INT16_C( 23361), -INT16_C( 17162),  INT16_C(  2128),  INT16_C( 13387),  INT16_C(   764),  INT16_C(  6179), -INT16_C( 20386) },
      {  INT16_C(  1831),  INT16_C( 22454),  INT16_C(  2634), -INT16_C( 27346), -INT16_C(  5993),  INT16_C( 13479),  INT16_C( 30802),  INT16_C(  5966) },
      UINT8_C(247) },
    { {  INT16_C(  3539), -INT16_C( 13892),  INT16_C(  3273),  INT16_C(  5585), -INT16_C( 12736),  INT16_C( 25623),  INT16_C( 30182),  INT16_C(  3348) },
      { -INT16_C( 13700), -INT16_C( 14492), -INT16_C( 27692),  INT16_C( 27484),  INT16_C(  1147), -INT16_C( 12641), -INT16_C(  4740),  INT16_C( 20453) },
         UINT8_MAX },
    { { -INT16_C( 24070), -INT16_C( 15336), -INT16_C(  5459), -INT16_C(  4391), -INT16_C(  3912), -INT16_C( 25006),  INT16_C( 26213), -INT16_C(  7508) },
      {  INT16_C(  4145),  INT16_C(  1449),  INT16_C(  1443),  INT16_C(  8049),  INT16_C(  4105), -INT16_C( 30995), -INT16_C( 11522), -INT16_C(  1835) },
         UINT8_MAX },
    { { -INT16_C(  4492),  INT16_C(  8636), -INT16_C( 27176), -INT16_C( 28657),  INT16_C( 24965), -INT16_C(  5330), -INT16_C(  9528), -INT16_C(  1587) },
      {  INT16_C( 30443), -INT16_C( 28930),  INT16_C( 28539), -INT16_C( 31315), -INT16_C( 25984),  INT16_C( 32267), -INT16_C(  8083), -INT16_C(  7818) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_test_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_test_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_test_epi16_mask(a, b);

    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_test_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[4];
    const int32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(   765531338),  INT32_C(  1967696310), -INT32_C(   609593534), -INT32_C(    31775930) },
      {  INT32_C(  1330662168),  INT32_C(  1289707212),  INT32_C(   943867559), -INT32_C(  1331329158) },
      UINT8_C( 15) },
    { {  INT32_C(   696451954), -INT32_C(   274805843),  INT32_C(  1758152738), -INT32_C(  2073631125) },
      { -INT32_C(    19679694), -INT32_C(  1135955435),  INT32_C(  1945406712), -INT32_C(  2044487404) },
      UINT8_C( 15) },
    { {  INT32_C(  1253025437), -INT32_C(  1824961167),  INT32_C(    33293461),  INT32_C(   495280875) },
      {  INT32_C(   773543961),  INT32_C(    65692939),  INT32_C(    91676401),  INT32_C(   344758903) },
      UINT8_C( 15) },
    { { -INT32_C(  1302447296),  INT32_C(   507877513), -INT32_C(  2027994724), -INT32_C(  1113283420) },
      {  INT32_C(   149667837),  INT32_C(   369874212),  INT32_C(   706446003),  INT32_C(  1547675420) },
      UINT8_C( 15) },
    { {  INT32_C(  1812897251), -INT32_C(   762686411),  INT32_C(   962177685),  INT32_C(  1274478158) },
      { -INT32_C(   497819203),  INT32_C(  1777885110), -INT32_C(    40627231), -INT32_C(  1638280261) },
      UINT8_C( 15) },
    { { -INT32_C(  1509267344),  INT32_C(  1366856892), -INT32_C(  1920216770), -INT32_C(  1915190833) },
      {  INT32_C(   426716259),  INT32_C(  1820551051),  INT32_C(   896079738),  INT32_C(  1523827434) },
      UINT8_C( 15) },
    { { -INT32_C(   436150998), -INT32_C(  1338476430),  INT32_C(   423478090), -INT32_C(  1482287548) },
      { -INT32_C(   842984126), -INT32_C(   147241860),  INT32_C(  1160553051), -INT32_C(  1885405084) },
      UINT8_C( 15) },
    { {  INT32_C(  1333108957),  INT32_C(  1644211480), -INT32_C(  1250149008), -INT32_C(  1789124013) },
      { -INT32_C(  1268638408), -INT32_C(  1129604255), -INT32_C(  1576937411), -INT32_C(  1255038504) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_test_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_test_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_test_epi32_mask(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_test_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { {  INT64_C( 3541757927509132222),  INT64_C( 7687597079528229179) },
      { -INT64_C( 6520346684714355228),  INT64_C( 3871675620523790299) },
      UINT8_C(  3) },
    { {  INT64_C( 7726286006862894583), -INT64_C( 8157066027950615150) },
      { -INT64_C( 1108273545875881540),  INT64_C( 7321278967605787374) },
      UINT8_C(  3) },
    { { -INT64_C( 7612870055471235906), -INT64_C(  669693595313638788) },
      {  INT64_C( 6758650077858681030), -INT64_C( 6927841478366299897) },
      UINT8_C(  3) },
    { {  INT64_C( 7832859191106422808),  INT64_C( 4037555621201095897) },
      { -INT64_C( 8438656620290640105),  INT64_C( 2006594436586317399) },
      UINT8_C(  3) },
    { {  INT64_C( 2774194611750987299),  INT64_C( 8017042133057278360) },
      { -INT64_C( 8338738839514950088),  INT64_C( 4943999258182633087) },
      UINT8_C(  3) },
    { {  INT64_C( 6374518444059513859),  INT64_C( 2932024709197508584) },
      { -INT64_C(  163491339522824710),  INT64_C( 2437884587083268207) },
      UINT8_C(  3) },
    { { -INT64_C( 1510702572440048400),  INT64_C(  698017322709510807) },
      {  INT64_C( 3169957705799996033), -INT64_C( 8668987727557133197) },
      UINT8_C(  3) },
    { { -INT64_C( 6725212298845674563),  INT64_C( 3729500228545009577) },
      {  INT64_C( 6868606172504783210),  INT64_C( 6686615777173245612) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_test_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_test_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_test_epi64_mask(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_test_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int8_t a[16];
    const int8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(13301),
      {  INT8_C( 108),  INT8_C(  72),  INT8_C(  63), -INT8_C( 104), -INT8_C(   3), -INT8_C( 117),  INT8_C(  61), -INT8_C( 108),
        -INT8_C(  14), -INT8_C( 115), -INT8_C(  45), -INT8_C(  76), -INT8_C(  50),  INT8_C(  60),  INT8_C(   2), -INT8_C(  69) },
      {  INT8_C(  99),  INT8_C( 116),  INT8_C(   6), -INT8_C( 105),  INT8_C(  69), -INT8_C(  90), -INT8_C(  20), -INT8_C(  69),
        -INT8_C(   6), -INT8_C(  86),  INT8_C(  94), -INT8_C(  51),  INT8_C(  81),  INT8_C(  83),  INT8_C(   1), -INT8_C(  67) },
      UINT16_C(13301) },
    { UINT16_C(16540),
      {  INT8_C(  85), -INT8_C( 103), -INT8_C(  53), -INT8_C( 110),  INT8_C(  45), -INT8_C(  67),  INT8_C(  31),  INT8_C(   1),
         INT8_C( 114), -INT8_C(  18),  INT8_C(  61),  INT8_C( 116), -INT8_C(  87), -INT8_C(  96), -INT8_C(  24), -INT8_C(  80) },
      {  INT8_C(  55),  INT8_C(  46),  INT8_C(  86),  INT8_C(  35), -INT8_C(  23),  INT8_C(  80), -INT8_C(  50),  INT8_C(  72),
         INT8_C(  30),  INT8_C(  31), -INT8_C( 101),  INT8_C(  31), -INT8_C(  36),  INT8_C(  55),  INT8_C(  95),  INT8_C(  50) },
      UINT16_C(16412) },
    { UINT16_C(10961),
      { -INT8_C(  60), -INT8_C(   2), -INT8_C(  25), -INT8_C(  28), -INT8_C(   1),  INT8_C(  89), -INT8_C(  46),  INT8_C(  60),
        -INT8_C(  51),  INT8_C( 123), -INT8_C(  36), -INT8_C(  74),  INT8_C(  43),  INT8_C(  20), -INT8_C(  28), -INT8_C( 126) },
      {  INT8_C(  55), -INT8_C(  51), -INT8_C(  46),  INT8_C(   5),  INT8_C(  21), -INT8_C(  16),  INT8_C(  36), -INT8_C(  79),
         INT8_C(  15),  INT8_C(   1), -INT8_C(  24),  INT8_C( 110),  INT8_C(  51), -INT8_C(  71), -INT8_C( 104), -INT8_C(   9) },
      UINT16_C(10897) },
    { UINT16_C(32952),
      { -INT8_C(  37), -INT8_C(  73), -INT8_C(  39), -INT8_C(  83), -INT8_C(  12), -INT8_C(  89),  INT8_C(  41), -INT8_C(  48),
         INT8_C(  93),  INT8_C(  84), -INT8_C(  28),  INT8_C(  65), -INT8_C(  42),  INT8_C(  28),  INT8_C(  14), -INT8_C(  87) },
      {  INT8_C(  33),  INT8_C(  36), -INT8_C( 103),  INT8_C(  70), -INT8_C(  43), -INT8_C(  87),  INT8_C(  71), -INT8_C(  67),
         INT8_C(  23),  INT8_C( 122),  INT8_C( 119), -INT8_C(  80),  INT8_C( 113),  INT8_C(  47),  INT8_C(  48),  INT8_C(  77) },
      UINT16_C(32952) },
    { UINT16_C( 2534),
      { -INT8_C(   6), -INT8_C(  38), -INT8_C(  80),  INT8_C(  35), -INT8_C(  85),  INT8_C(  13),  INT8_C( 120), -INT8_C( 113),
         INT8_C(  78),  INT8_C(  78), -INT8_C(  85),  INT8_C(  93), -INT8_C(   9), -INT8_C(  51), -INT8_C( 127), -INT8_C( 111) },
      {  INT8_C(  19),  INT8_C(  86),  INT8_C(  58),  INT8_C(  90),  INT8_C(  19),  INT8_C(  81), -INT8_C(  44), -INT8_C( 118),
         INT8_C(   1),  INT8_C(  69), -INT8_C(  71),  INT8_C(  49), -INT8_C( 110), -INT8_C(  96),  INT8_C(  59), -INT8_C( 115) },
      UINT16_C( 2278) },
    { UINT16_C(60282),
      { -INT8_C(  80),  INT8_C(  37), -INT8_C(   7),  INT8_C(  40), -INT8_C(  75),  INT8_C(  71),  INT8_C( 119),  INT8_C(  96),
        -INT8_C(  92),  INT8_C( 110),  INT8_C(  45),  INT8_C(  37), -INT8_C(   1),  INT8_C(  64),  INT8_C( 123),  INT8_C(  57) },
      { -INT8_C( 102), -INT8_C( 113), -INT8_C( 117),  INT8_C( 110),  INT8_C(  25), -INT8_C( 116), -INT8_C(  76), -INT8_C(  45),
        -INT8_C(  66),  INT8_C(  70),  INT8_C( 115), -INT8_C(   7), -INT8_C(  45), -INT8_C(  19), -INT8_C(  28), -INT8_C( 124) },
      UINT16_C(27514) },
    { UINT16_C(56595),
      { -INT8_C(  84), -INT8_C(  56),  INT8_C(  37),  INT8_C(  35),  INT8_C(  40), -INT8_C(  55), -INT8_C( 110),  INT8_C(  86),
        -INT8_C(  17), -INT8_C( 111), -INT8_C( 106),  INT8_C( 106), -INT8_C(  53),  INT8_C(  49), -INT8_C(   7),  INT8_C(  86) },
      { -INT8_C(  97),  INT8_C(  19), -INT8_C(  30),  INT8_C(  83), -INT8_C(  26), -INT8_C(  96), -INT8_C( 102),  INT8_C(  89),
        -INT8_C( 103),  INT8_C( 109),  INT8_C(  70),  INT8_C( 126), -INT8_C(  15),  INT8_C(  89),  INT8_C(  91), -INT8_C(  98) },
      UINT16_C(56593) },
    { UINT16_C(32801),
      { -INT8_C(  63),  INT8_C(  74),  INT8_C(  74),  INT8_C(  83), -INT8_C(  96),  INT8_C(  57), -INT8_C(  27),  INT8_C(  54),
        -INT8_C(  93), -INT8_C(  80),  INT8_C( 103), -INT8_C(  99),  INT8_C(   6),  INT8_C(   7), -INT8_C(  80), -INT8_C(  24) },
      {  INT8_C(  90), -INT8_C( 106), -INT8_C( 119), -INT8_C(  12), -INT8_C(  17),  INT8_C(  34),  INT8_C(  98),  INT8_C(  53),
        -INT8_C(  96),  INT8_C(  83), -INT8_C( 113), -INT8_C(   4), -INT8_C(  15), -INT8_C(  80),  INT8_C( 124), -INT8_C(  77) },
      UINT16_C(32801) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_test_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_test_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_mask_test_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_test_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 61),
      {  INT16_C( 14345),  INT16_C( 22680),  INT16_C( 16551), -INT16_C( 25642), -INT16_C( 23951), -INT16_C( 22915),  INT16_C( 32232),  INT16_C( 18055) },
      {  INT16_C( 11264),  INT16_C(  8692),  INT16_C(  3392), -INT16_C( 25724),  INT16_C(  2814),  INT16_C( 20115),  INT16_C(  6252),  INT16_C( 30091) },
      UINT8_C( 57) },
    { UINT8_C( 80),
      { -INT16_C( 13021),  INT16_C( 25847), -INT16_C( 27996),  INT16_C( 18133),  INT16_C( 31503), -INT16_C( 29649),  INT16_C( 29954),  INT16_C( 12173) },
      { -INT16_C( 20887),  INT16_C( 30319),  INT16_C(  2610),  INT16_C( 15732), -INT16_C( 15714), -INT16_C( 18775),  INT16_C(  8013),  INT16_C( 28934) },
      UINT8_C( 80) },
    { UINT8_C(236),
      { -INT16_C( 10755), -INT16_C( 28784), -INT16_C( 10326),  INT16_C(  9630),  INT16_C( 10758),  INT16_C( 31527),  INT16_C( 22199),  INT16_C( 26340) },
      {  INT16_C( 23238), -INT16_C( 12136), -INT16_C( 10802), -INT16_C( 28562),  INT16_C(  9343), -INT16_C( 24867),  INT16_C( 20010),  INT16_C( 10122) },
      UINT8_C(236) },
    { UINT8_C( 35),
      { -INT16_C( 18917), -INT16_C(  3379), -INT16_C(  3500),  INT16_C( 32760),  INT16_C( 29466),  INT16_C( 28726), -INT16_C( 25513), -INT16_C( 20170) },
      {  INT16_C(  1845),  INT16_C(  2687),  INT16_C(  3957), -INT16_C( 25975),  INT16_C( 10220),  INT16_C( 15300), -INT16_C(  4942), -INT16_C( 12962) },
      UINT8_C( 35) },
    { UINT8_C(162),
      { -INT16_C( 16596),  INT16_C(  7927),  INT16_C( 30391),  INT16_C( 10808), -INT16_C( 22100),  INT16_C( 18817),  INT16_C( 13023), -INT16_C(  6530) },
      { -INT16_C( 30543), -INT16_C( 16292), -INT16_C(  2542),  INT16_C( 14764), -INT16_C(  6214), -INT16_C( 22805), -INT16_C( 18362),  INT16_C( 29257) },
      UINT8_C(162) },
    { UINT8_C(119),
      { -INT16_C( 28608), -INT16_C( 18898),  INT16_C( 22729),  INT16_C( 29282), -INT16_C( 21543),  INT16_C(  2897),  INT16_C( 14377), -INT16_C( 19780) },
      {  INT16_C( 31892), -INT16_C( 30012), -INT16_C(   727),  INT16_C(  4164), -INT16_C(  5143), -INT16_C( 24234), -INT16_C( 14284),  INT16_C( 29721) },
      UINT8_C(119) },
    { UINT8_C( 89),
      {  INT16_C( 10823), -INT16_C( 24542), -INT16_C( 27508),  INT16_C( 14457), -INT16_C( 31259),  INT16_C(  7521),  INT16_C(  4929), -INT16_C( 16719) },
      {  INT16_C( 15319), -INT16_C( 10777), -INT16_C(  2176),  INT16_C( 27582),  INT16_C( 24398),  INT16_C(  5791),  INT16_C(  4984), -INT16_C( 16273) },
      UINT8_C( 89) },
    { UINT8_C( 61),
      {  INT16_C( 24721),  INT16_C(  9673),  INT16_C(   473),  INT16_C( 24075),  INT16_C( 10339),  INT16_C( 30368),  INT16_C( 24282),  INT16_C(  5454) },
      {  INT16_C(  9029),  INT16_C( 15509),  INT16_C(   225),  INT16_C( 16522), -INT16_C( 24161), -INT16_C( 19783),  INT16_C( 30992), -INT16_C( 23825) },
      UINT8_C( 61) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_test_epi16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_test_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_mask_test_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_test_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 81),
      {  INT32_C(  2136238892), -INT32_C(  1476242093), -INT32_C(   194429540),  INT32_C(   505228311) },
      { -INT32_C(  1816943728),  INT32_C(  1503665518), -INT32_C(   634657429), -INT32_C(    47454768) },
      UINT8_C(  1) },
    { UINT8_C( 80),
      { -INT32_C(   727417729), -INT32_C(  1133491074), -INT32_C(   489462346),  INT32_C(   628355458) },
      { -INT32_C(  1030551900), -INT32_C(  1959924570), -INT32_C(    44300264), -INT32_C(  1286710989) },
      UINT8_C(  0) },
    { UINT8_C(213),
      {  INT32_C(  1045727218),  INT32_C(  1576276216), -INT32_C(   723527709),  INT32_C(  1350108234) },
      { -INT32_C(  2064237673),  INT32_C(  1889304936),  INT32_C(   916756957), -INT32_C(   653502489) },
      UINT8_C(  5) },
    { UINT8_C(222),
      {  INT32_C(  1893079136), -INT32_C(   481086708),  INT32_C(   372057874), -INT32_C(   609321568) },
      { -INT32_C(   180080013),  INT32_C(  1741862094), -INT32_C(  1353774760),  INT32_C(  1972250645) },
      UINT8_C( 14) },
    { UINT8_C( 64),
      { -INT32_C(  1739725468),  INT32_C(  1604988984), -INT32_C(   620707490), -INT32_C(  1588602001) },
      { -INT32_C(   730905569),  INT32_C(   523032086),  INT32_C(  1312087077), -INT32_C(   829511318) },
      UINT8_C(  0) },
    { UINT8_C(142),
      {  INT32_C(   214329051), -INT32_C(   764795375),  INT32_C(    21054758), -INT32_C(   668867948) },
      {  INT32_C(   670037329), -INT32_C(    45281759), -INT32_C(   328754365),  INT32_C(    91960873) },
      UINT8_C( 14) },
    { UINT8_C(156),
      {  INT32_C(  1739395393), -INT32_C(  1064468613), -INT32_C(  1571450944), -INT32_C(  1510789712) },
      {  INT32_C(   734403356),  INT32_C(    57590888),  INT32_C(  1630296619),  INT32_C(   402534869) },
      UINT8_C( 12) },
    { UINT8_C( 66),
      {  INT32_C(   733839019), -INT32_C(  1679065588),  INT32_C(     4951763), -INT32_C(  1675759487) },
      {  INT32_C(  2047101110),  INT32_C(   279316406),  INT32_C(  1709573940), -INT32_C(  1314325243) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_test_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_test_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_mask_test_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}


static int
test_easysimd_mm_mask_test_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t b[2];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(114),
      { -INT64_C( 6458725214833552692), -INT64_C( 8545314885556938335) },
      {  INT64_C( 5750452733465070314),  INT64_C( 5160792549948597513) },
      UINT8_C(  2) },
    { UINT8_C(163),
      {  INT64_C(  593047639399223962), -INT64_C( 6463242041493408910) },
      {  INT64_C( 8903405724415133904), -INT64_C(  945969947535295893) },
      UINT8_C(  3) },
    { UINT8_C(143),
      { -INT64_C(  466888070310949850), -INT64_C( 4794861126668654414) },
      {  INT64_C( 5032647833779698476),  INT64_C( 1824117225684800644) },
      UINT8_C(  3) },
    { UINT8_C( 28),
      { -INT64_C( 7383127457707971441), -INT64_C( 6928633387807789455) },
      {  INT64_C( 3436650153172944031),  INT64_C( 5960580913365742092) },
      UINT8_C(  0) },
    { UINT8_C(105),
      { -INT64_C( 2240342136021440101), -INT64_C( 5715872305544653694) },
      { -INT64_C(  159319734191662514), -INT64_C( 5061181634653428194) },
      UINT8_C(  1) },
    { UINT8_C( 59),
      { -INT64_C( 6046982894223806242), -INT64_C( 6069505600128539822) },
      { -INT64_C( 5715854857228353132), -INT64_C( 6250564402239567884) },
      UINT8_C(  3) },
    { UINT8_C(185),
      {  INT64_C( 8236378175522814766),  INT64_C(  645121436830385046) },
      {  INT64_C( 3912505678158332452), -INT64_C( 4533811374991519022) },
      UINT8_C(  1) },
    { UINT8_C(229),
      { -INT64_C( 3532327294662754843), -INT64_C( 3531562848648603158) },
      { -INT64_C(  826635232684976719), -INT64_C( 2610599084330261963) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_test_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_test_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_mask_test_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_test_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[16];
    const int16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { {  INT16_C(  7026),  INT16_C( 31568),  INT16_C( 24597), -INT16_C( 20474),  INT16_C( 23615),  INT16_C( 17278),  INT16_C( 15775),  INT16_C( 21285),
         INT16_C( 13700),  INT16_C(  1776), -INT16_C( 12380), -INT16_C( 12539), -INT16_C( 14813),  INT16_C( 25663),  INT16_C(  3177), -INT16_C(  3628) },
      { -INT16_C( 20616), -INT16_C( 29349),  INT16_C( 18960),  INT16_C( 20473),  INT16_C( 30630), -INT16_C( 32622), -INT16_C( 15776), -INT16_C( 21286),
        -INT16_C( 13701),  INT16_C(  8114),  INT16_C( 12954),  INT16_C( 12538),  INT16_C( 14812),  INT16_C( 19348), -INT16_C(  3178),  INT16_C(  3627) },
      UINT16_C( 9783) },
    { {  INT16_C(  7955), -INT16_C( 22460),  INT16_C( 18168), -INT16_C(   489), -INT16_C( 22615), -INT16_C( 15924), -INT16_C( 30488), -INT16_C( 31741),
        -INT16_C(  2733),  INT16_C( 20922),  INT16_C(  8416),  INT16_C(  1323),  INT16_C( 12287),  INT16_C(  2751), -INT16_C( 30560),  INT16_C( 30875) },
      { -INT16_C(  7956), -INT16_C(  7136),  INT16_C( 16422),  INT16_C(   488),  INT16_C( 22614),  INT16_C( 15923),  INT16_C( 14049),  INT16_C( 13506),
         INT16_C( 31787),  INT16_C(  3205), -INT16_C( 20323), -INT16_C( 18415), -INT16_C( 12288), -INT16_C( 24126),  INT16_C( 24152),  INT16_C( 17433) },
      UINT16_C(61382) },
    { { -INT16_C( 20823), -INT16_C( 27897),  INT16_C(  5365), -INT16_C( 28567),  INT16_C(   357),  INT16_C( 13429),  INT16_C( 11264),  INT16_C(  1568),
         INT16_C(  5622), -INT16_C( 22697), -INT16_C( 21170),  INT16_C( 21872), -INT16_C(  1994), -INT16_C( 10632), -INT16_C( 22789), -INT16_C( 10792) },
      {  INT16_C( 16044),  INT16_C( 27896), -INT16_C(  5366),  INT16_C( 28566),  INT16_C(  3053), -INT16_C( 16477), -INT16_C( 11265), -INT16_C(  1569),
        -INT16_C(  5623),  INT16_C( 22696),  INT16_C(  3479), -INT16_C( 21873),  INT16_C(  1993), -INT16_C(  2687),  INT16_C( 22788), -INT16_C( 20278) },
      UINT16_C(42033) },
    { { -INT16_C( 21813), -INT16_C( 13928), -INT16_C( 30718), -INT16_C( 16640), -INT16_C( 29380), -INT16_C( 23005), -INT16_C( 11942), -INT16_C( 17838),
         INT16_C( 30102), -INT16_C(  1162), -INT16_C( 24680), -INT16_C( 11339),  INT16_C( 22724), -INT16_C(  1330), -INT16_C(  3554), -INT16_C(  5384) },
      {  INT16_C( 21812),  INT16_C( 13927),  INT16_C( 26845),  INT16_C(  6646),  INT16_C( 29379),  INT16_C( 23004),  INT16_C( 11941),  INT16_C( 13588),
        -INT16_C( 30103),  INT16_C(  6705),  INT16_C( 24679),  INT16_C( 11338), -INT16_C(   839),  INT16_C(  1329),  INT16_C(  3553),  INT16_C(  5383) },
      UINT16_C( 4748) },
    { {  INT16_C( 28757), -INT16_C( 25167), -INT16_C(  5472),  INT16_C( 26310), -INT16_C( 17770), -INT16_C(  1344),  INT16_C( 18728),  INT16_C( 21959),
        -INT16_C( 13925),  INT16_C( 12111), -INT16_C(  8108), -INT16_C( 30547), -INT16_C(  6196),  INT16_C( 11096),  INT16_C( 16589),  INT16_C(  8951) },
      { -INT16_C(   336),  INT16_C( 25166),  INT16_C(  5471), -INT16_C(  2616), -INT16_C( 30513), -INT16_C( 27921), -INT16_C( 18729), -INT16_C(  6425),
         INT16_C( 13924),  INT16_C( 17430),  INT16_C(  8107),  INT16_C( 30546),  INT16_C( 28166), -INT16_C( 11097), -INT16_C( 24658),  INT16_C( 24310) },
      UINT16_C(53945) },
    { {  INT16_C(  4460),  INT16_C(  6318), -INT16_C( 13228),  INT16_C( 26013),  INT16_C( 26039),  INT16_C( 17229),  INT16_C(  5449), -INT16_C( 11202),
         INT16_C( 32257), -INT16_C( 30654),  INT16_C( 12862),  INT16_C(  3856), -INT16_C(  4364),  INT16_C( 13756), -INT16_C( 14141), -INT16_C(  8941) },
      { -INT16_C(  4461), -INT16_C(  6319), -INT16_C(  4165),  INT16_C( 26700), -INT16_C( 26040),  INT16_C( 17323), -INT16_C(  5450), -INT16_C( 18409),
         INT16_C( 28520),  INT16_C( 30653), -INT16_C( 12863), -INT16_C( 19066),  INT16_C( 17340),  INT16_C( 32746), -INT16_C(   757), -INT16_C( 24995) },
      UINT16_C(63916) },
    { {  INT16_C( 32173), -INT16_C(  4723), -INT16_C( 20657), -INT16_C( 17360),  INT16_C( 20996),  INT16_C( 27460),  INT16_C( 10904),  INT16_C( 18714),
        -INT16_C( 25006), -INT16_C( 18247), -INT16_C(  8279), -INT16_C(  2342),  INT16_C( 31982), -INT16_C(  1201), -INT16_C( 21673),  INT16_C( 10228) },
      { -INT16_C( 27453),  INT16_C(  4722), -INT16_C( 23997), -INT16_C( 21042), -INT16_C( 20997), -INT16_C( 27461), -INT16_C( 10905), -INT16_C( 17699),
        -INT16_C( 26764),  INT16_C(  7538),  INT16_C( 19830),  INT16_C( 25619),  INT16_C( 25289),  INT16_C( 17248),  INT16_C( 21672),  INT16_C( 27499) },
      UINT16_C(49037) },
    { {  INT16_C(   815), -INT16_C( 18924), -INT16_C(  5360),  INT16_C( 18728), -INT16_C( 15692), -INT16_C(    85), -INT16_C( 25093), -INT16_C( 22549),
         INT16_C( 19653), -INT16_C(  8585), -INT16_C( 23472),  INT16_C( 22739),  INT16_C(  5353), -INT16_C(  9270), -INT16_C(  2547),  INT16_C( 15815) },
      { -INT16_C(  4359),  INT16_C( 18923),  INT16_C(  5359), -INT16_C( 31342),  INT16_C( 15691),  INT16_C( 28292),  INT16_C( 25092),  INT16_C( 22548),
        -INT16_C( 19654),  INT16_C(  8584),  INT16_C( 23471),  INT16_C( 27001), -INT16_C(  5354),  INT16_C(  9269), -INT16_C(   799), -INT16_C(  9375) },
      UINT16_C(51241) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_test_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_test_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[16];
    int16_t b_[16];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < 16 ; j++)
      if (!(easysimd_test_codegen_random_i16() & 1))
        a_[j] = ~b_[j];

    easysimd__m256i a = easysimd_mm256_loadu_epi16(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(b_);
    easysimd__mmask16 r = easysimd_mm256_test_epi16_mask(a, b);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_test_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t b[8];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { {  INT32_C(  1576173070), -INT32_C(  1227665335), -INT32_C(    59825545),  INT32_C(  1808402821), -INT32_C(  1055795563),  INT32_C(  1242636844),  INT32_C(  1083448238),  INT32_C(  1669148756) },
      { -INT32_C(   910135378), -INT32_C(  1227665335), -INT32_C(    59825545),  INT32_C(   946288939), -INT32_C(  1055795563), -INT32_C(  1509219336), -INT32_C(  1813602498),  INT32_C(  2130076623) },
         UINT8_MAX },
    { {  INT32_C(  1502482515),  INT32_C(  1742168106),  INT32_C(   516903074), -INT32_C(   416387279), -INT32_C(   157941727), -INT32_C(   515281222),  INT32_C(  1806101534), -INT32_C(  1606980531) },
      { -INT32_C(  1493580420), -INT32_C(   938618842),  INT32_C(  1390861345),  INT32_C(  1832457548), -INT32_C(   245051627), -INT32_C(   515281222),  INT32_C(  1806101534), -INT32_C(   737419688) },
         UINT8_MAX },
    { { -INT32_C(   627219232), -INT32_C(  1103432491),  INT32_C(  1980801001),  INT32_C(  1931030305), -INT32_C(  2050119686),  INT32_C(  1420913802),  INT32_C(  1535332521), -INT32_C(  1634268738) },
      { -INT32_C(   627219232), -INT32_C(  1103432491), -INT32_C(  1254864237),  INT32_C(   724061561), -INT32_C(  2050119686),  INT32_C(   316243475),  INT32_C(  1535332521), -INT32_C(  1634268738) },
         UINT8_MAX },
    { { -INT32_C(   425524547), -INT32_C(  1281186635),  INT32_C(  1029697425), -INT32_C(   321205076), -INT32_C(   803313641), -INT32_C(  1699357683), -INT32_C(   988117364), -INT32_C(   153068999) },
      {  INT32_C(   316441692),  INT32_C(    46497320),  INT32_C(  1029697425), -INT32_C(   165070149), -INT32_C(   803313641), -INT32_C(  1699357683), -INT32_C(   988117364),  INT32_C(  1019018208) },
         UINT8_MAX },
    { { -INT32_C(  1685150537), -INT32_C(  1182621014), -INT32_C(  1525995620), -INT32_C(  2119346167),  INT32_C(  1199582608), -INT32_C(  1285189037),  INT32_C(  1917372580),  INT32_C(   575268558) },
      { -INT32_C(  1685150537), -INT32_C(  1182621014), -INT32_C(  1525995620), -INT32_C(  1473791873),  INT32_C(  1199582608), -INT32_C(  1285189037),  INT32_C(  1917372580),  INT32_C(  1704235438) },
         UINT8_MAX },
    { {  INT32_C(  1968273329), -INT32_C(  1512641798), -INT32_C(   931040065),  INT32_C(  1675206665),  INT32_C(   249445735),  INT32_C(   157184935),  INT32_C(  2082877845), -INT32_C(   954928960) },
      {  INT32_C(    17636102), -INT32_C(  1784225018), -INT32_C(   931040065),  INT32_C(   120347157),  INT32_C(   249445735), -INT32_C(   854115152),  INT32_C(  2082877845), -INT32_C(   954928960) },
         UINT8_MAX },
    { {  INT32_C(  2074731646), -INT32_C(   534163715), -INT32_C(   107072631),  INT32_C(  1726010991),  INT32_C(  1015704662),  INT32_C(   157484525), -INT32_C(  1867478357), -INT32_C(   153817992) },
      {  INT32_C(  2074731646), -INT32_C(  1487154382), -INT32_C(   107072631), -INT32_C(  1117815287),  INT32_C(  1015704662),  INT32_C(   157484525), -INT32_C(  1867478357), -INT32_C(   153817992) },
         UINT8_MAX },
    { {  INT32_C(   889530340), -INT32_C(    71396860), -INT32_C(  1090526217), -INT32_C(  1673489020), -INT32_C(  1140162102),  INT32_C(   946919153),  INT32_C(  1308033563),  INT32_C(  1297132648) },
      { -INT32_C(   914205228), -INT32_C(    71396860), -INT32_C(  1933918602), -INT32_C(  1673489020), -INT32_C(  1140162102),  INT32_C(  1928624727), -INT32_C(   457118852),  INT32_C(  1798377623) },
         UINT8_MAX }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_test_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_test_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[8];
    int32_t b_[8];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < 8 ; j++)
      if (!(easysimd_test_codegen_random_i32() & 1))
        a_[j] = b_[j];

    easysimd__m256i a = easysimd_mm256_loadu_epi32(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(b_);
    easysimd__mmask16 r = easysimd_mm256_test_epi32_mask(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_test_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { {  INT64_C( 8835812740395334310),  INT64_C( 5573947357810759464),  INT64_C( 6276595581306043722), -INT64_C(  861790280415799233) },
      { -INT64_C( 8835812740395334311), -INT64_C( 5573947357810759465),  INT64_C( 8979882026993200601),  INT64_C(  861790280415799232) },
      UINT8_C(  4) },
    { { -INT64_C( 5941988136692490827), -INT64_C( 2392387728582491484),  INT64_C( 1243652977704752283),  INT64_C( 4129196832974778823) },
      {  INT64_C( 2404128300749562474),  INT64_C( 2392387728582491483), -INT64_C( 1243652977704752284), -INT64_C( 4129196832974778824) },
      UINT8_C(  1) },
    { { -INT64_C( 8676600770535304317),  INT64_C( 2961317920001018364), -INT64_C(   57994133409048994), -INT64_C( 4855659705579216101) },
      {  INT64_C( 8802058416649140495), -INT64_C( 2010308002063855622),  INT64_C( 4858728858343125449),  INT64_C( 9044989945217102480) },
      UINT8_C( 15) },
    { { -INT64_C( 1428062264493277478), -INT64_C( 5964462973635439358),  INT64_C( 5187749680417646244),  INT64_C( 6251760974764989534) },
      {  INT64_C( 1428062264493277477), -INT64_C( 1510930984531439916),  INT64_C( 7484733603987847359), -INT64_C( 6251760974764989535) },
      UINT8_C(  6) },
    { {  INT64_C( 6032260234097132544), -INT64_C( 4702793544162396315), -INT64_C( 4442624843964900496),  INT64_C( 1697096292824225257) },
      { -INT64_C( 6032260234097132545),  INT64_C( 4486021522440354899), -INT64_C( 1141759230797125849), -INT64_C( 2419644802034068931) },
      UINT8_C( 14) },
    { {  INT64_C( 8307013365677434994), -INT64_C(  537615070302991619),  INT64_C( 2538636305041568169),  INT64_C( 8024818589548783293) },
      {  INT64_C( 7647531552279493897),  INT64_C( 9075063126622196231), -INT64_C( 6615330567197762870),  INT64_C( 2139889408194458437) },
      UINT8_C( 15) },
    { { -INT64_C( 7825902930912875420),  INT64_C( 1639439112074188263),  INT64_C( 4151411919778673172), -INT64_C( 7318874730829296078) },
      {  INT64_C( 7232898085805272590), -INT64_C(  430701902001966656), -INT64_C( 5105127561473245240), -INT64_C( 5976345289885057729) },
      UINT8_C( 15) },
    { {  INT64_C( 7802704781339070151),  INT64_C( 8931343896071189564),  INT64_C( 2419363636774825821), -INT64_C( 3845408552048773561) },
      { -INT64_C( 7802704781339070152), -INT64_C( 2820228193823759251), -INT64_C( 2419363636774825822), -INT64_C( 2018395849564533017) },
      UINT8_C( 10) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_test_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_test_epi64_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[4];
    int64_t b_[4];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < 4 ; j++)
      if (!(easysimd_test_codegen_random_i8() & 1))
        a_[j] = ~b_[j];

    easysimd__m256i a = easysimd_mm256_loadu_epi64(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(b_);
    easysimd__mmask8 r = easysimd_mm256_test_epi64_mask(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_mask_test_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k;
    const int8_t a[32];
    const int8_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(2880730040),
      {  INT8_C(   8),  INT8_C(   7),  INT8_C(  69), -INT8_C(  30), -INT8_C(  15), -INT8_C(  31),  INT8_C(  15),  INT8_C(  10),
        -INT8_C(  47),  INT8_C( 116),  INT8_C(  20),  INT8_C(  31), -INT8_C( 125), -INT8_C(  94),  INT8_C(  20), -INT8_C( 127),
        -INT8_C(  28), -INT8_C(  11), -INT8_C(  29),  INT8_C(  49), -INT8_C(  72), -INT8_C(  40),  INT8_C( 123), -INT8_C(  84),
        -INT8_C( 102), -INT8_C(  73),  INT8_C( 126), -INT8_C(  81),  INT8_C(  42),  INT8_C(  90), -INT8_C( 109),  INT8_C(  51) },
      {  INT8_C(  97), -INT8_C(  40),  INT8_C(  21),  INT8_C(  17),  INT8_C(  10),  INT8_C(   4),  INT8_C(  80), -INT8_C( 117),
         INT8_C(   8),  INT8_C( 113), -INT8_C(  85),  INT8_C(   0),  INT8_C(  20), -INT8_C(  65), -INT8_C( 120),  INT8_C( 120),
         INT8_C(  16),      INT8_MAX,  INT8_C(  41),  INT8_C(  76),  INT8_C(  64), -INT8_C(  92),  INT8_C(   0),  INT8_C(  82),
         INT8_C(  68), -INT8_C( 106), -INT8_C( 127),  INT8_C(   0), -INT8_C(  16),  INT8_C(  36),  INT8_C(  40),  INT8_C(  64) },
      UINT32_C(  35922560) },
    { UINT32_C(2978481183),
      {  INT8_C( 118),  INT8_C(   6),  INT8_C(  54), -INT8_C(  22),  INT8_C(   3), -INT8_C(  36),  INT8_C( 114), -INT8_C(  65),
         INT8_C(  31), -INT8_C(  14),  INT8_C(  48), -INT8_C(  30), -INT8_C( 101),  INT8_C(  13), -INT8_C(   4), -INT8_C(  23),
         INT8_C(   2),  INT8_C(   9), -INT8_C( 109), -INT8_C( 114), -INT8_C(  74), -INT8_C(  60),  INT8_C(  11), -INT8_C( 104),
        -INT8_C(  82), -INT8_C(  39), -INT8_C( 122),  INT8_C( 102),  INT8_C(  77),  INT8_C(  58),  INT8_C(  17), -INT8_C(  61) },
      {  INT8_C(  65),  INT8_C(  71), -INT8_C(  83),  INT8_C(  68),  INT8_C(  36),  INT8_C(  31),  INT8_C(   4),  INT8_C(  67),
         INT8_C(  17),  INT8_C(   4),  INT8_C(   5),  INT8_C(  12),  INT8_C(  64),  INT8_C(  34),  INT8_C(   2),  INT8_C(  67),
         INT8_C(  43),  INT8_C(  32),  INT8_C(  64), -INT8_C(  31), -INT8_C(  19),  INT8_C(  25),  INT8_C( 112), -INT8_C( 101),
        -INT8_C(  74),  INT8_C(  38),  INT8_C(   2),  INT8_C(   3),  INT8_C(  50),  INT8_C(  19), -INT8_C(  58),  INT8_C( 123) },
      UINT32_C(2710044687) },
    { UINT32_C(4254367656),
      { -INT8_C(  57),  INT8_C(  74),  INT8_C(  85), -INT8_C(  65),  INT8_C(  50), -INT8_C(  91), -INT8_C(   9),  INT8_C(  60),
        -INT8_C( 117),  INT8_C( 115),  INT8_C( 113), -INT8_C( 102), -INT8_C(  63), -INT8_C( 120), -INT8_C( 105), -INT8_C(  20),
         INT8_C(  24),  INT8_C(  48), -INT8_C(  50), -INT8_C(  89), -INT8_C(  54), -INT8_C(  77),  INT8_C( 113),  INT8_C( 120),
         INT8_C(  94), -INT8_C(  75), -INT8_C( 127),  INT8_C( 125), -INT8_C(  71),  INT8_C(   9),  INT8_C(  47), -INT8_C( 127) },
      {  INT8_C(  84), -INT8_C( 124),  INT8_C(   0),  INT8_C(   0),  INT8_C(   9),  INT8_C(  55),  INT8_C(   0), -INT8_C(  76),
         INT8_C(  32),  INT8_C(   4),  INT8_C(  14),  INT8_C( 100),  INT8_C(  60), -INT8_C(  26),  INT8_C(  89),  INT8_C(  16),
         INT8_C(   6),  INT8_C(  39),  INT8_C( 124),  INT8_C(  64),  INT8_C(  16),  INT8_C(  76),  INT8_C(   8),  INT8_C(   0),
        -INT8_C(  96),  INT8_C(  74),  INT8_C(  54),  INT8_C(   0), -INT8_C(  28), -INT8_C(  27), -INT8_C(  48),  INT8_C(  56) },
      UINT32_C( 805593248) },
    { UINT32_C(2744507799),
      {  INT8_C(   0), -INT8_C(  38), -INT8_C(   4), -INT8_C(  74),  INT8_C( 111),  INT8_C( 104),  INT8_C(  41), -INT8_C(  23),
         INT8_C(  45),  INT8_C( 112),  INT8_C( 122),  INT8_C(  25),  INT8_C(  51), -INT8_C(  21), -INT8_C(  33), -INT8_C(  29),
        -INT8_C(  75), -INT8_C(  34),  INT8_C(  53),  INT8_C(  89), -INT8_C( 110), -INT8_C(  28), -INT8_C(  30),  INT8_C(  44),
         INT8_C( 111), -INT8_C(  94),  INT8_C(  46),  INT8_C(  24),  INT8_C(  33), -INT8_C(  61),  INT8_C(  21),  INT8_C(  34) },
      { -INT8_C(  99),  INT8_C(   1),  INT8_C(   0),  INT8_C(  12),  INT8_C( 122),  INT8_C(   1), -INT8_C(  42), -INT8_C(  89),
         INT8_C(  80),  INT8_C( 112), -INT8_C(  63), -INT8_C(  92),  INT8_C(  72), -INT8_C(  96), -INT8_C( 121),  INT8_C(  16),
         INT8_C(  74), -INT8_C(  67),  INT8_C( 105),  INT8_C(  17),  INT8_C(  33),  INT8_C(  11),  INT8_C(  61),  INT8_C(  16),
             INT8_MIN,  INT8_C(  73),  INT8_C(   0),  INT8_C(   7),  INT8_C(  46),  INT8_C(  61),  INT8_C(  32), -INT8_C(  52) },
      UINT32_C( 537150608) },
    { UINT32_C( 217934035),
      { -INT8_C(  85),  INT8_C(   6),  INT8_C(  32), -INT8_C(  22),  INT8_C(  55),  INT8_C(  65), -INT8_C( 125),  INT8_C(   9),
        -INT8_C(  22),  INT8_C(  45),  INT8_C(  90),  INT8_C(  80),  INT8_C(  64), -INT8_C(  69),  INT8_C(  87), -INT8_C(  98),
         INT8_C(  90),  INT8_C( 111), -INT8_C(  21),  INT8_C( 100), -INT8_C(  81),  INT8_C(  71), -INT8_C( 100),  INT8_C(  44),
        -INT8_C(  44), -INT8_C(  95), -INT8_C(   7),  INT8_C( 108),  INT8_C( 126), -INT8_C( 114),  INT8_C(  15),  INT8_C(  41) },
      { -INT8_C( 108),  INT8_C(  47),  INT8_C(  19), -INT8_C(  52),  INT8_C(  64), -INT8_C( 105),  INT8_C(  84),  INT8_C(  82),
         INT8_C(   4),  INT8_C(   2), -INT8_C(  95),  INT8_C(   4), -INT8_C(  21),  INT8_C(   3), -INT8_C(  96),  INT8_C(  69),
         INT8_C(  32), -INT8_C( 115),  INT8_C(   0),  INT8_C(  33), -INT8_C(  44),  INT8_C(  69),  INT8_C(  65), -INT8_C(  87),
         INT8_C(  34),  INT8_C(  70),  INT8_C(   4),  INT8_C( 100), -INT8_C(  43),  INT8_C(  32),      INT8_MIN,  INT8_C(  64) },
      UINT32_C( 146284547) },
    { UINT32_C(3237893845),
      {  INT8_C( 115),  INT8_C(  28),  INT8_C(   8), -INT8_C(  83), -INT8_C(  26),  INT8_C(   9), -INT8_C(  45), -INT8_C(  77),
        -INT8_C(  84),  INT8_C(  61), -INT8_C(  14), -INT8_C(  36),  INT8_C(  81),  INT8_C(  82), -INT8_C(  31), -INT8_C(  86),
        -INT8_C(   1), -INT8_C( 113), -INT8_C(  23), -INT8_C(  13),  INT8_C(  82), -INT8_C( 115), -INT8_C(  68),  INT8_C(  57),
        -INT8_C(  65), -INT8_C(  17),  INT8_C( 115), -INT8_C( 109),  INT8_C(  87),  INT8_C( 113), -INT8_C(  97), -INT8_C(  54) },
      { -INT8_C( 115), -INT8_C(  89),  INT8_C( 119),  INT8_C( 115), -INT8_C(  80),  INT8_C(  66),  INT8_C(  36),  INT8_C(  92),
         INT8_C(   0),  INT8_C(  24),  INT8_C(   9),  INT8_C(   1),  INT8_C( 106),  INT8_C(  26),  INT8_C(   2),  INT8_C( 105),
         INT8_C(   0),  INT8_C( 108),  INT8_C(  20),  INT8_C(  12), -INT8_C(  88),  INT8_C(  16),  INT8_C(  54), -INT8_C(  71),
         INT8_C(   8),  INT8_C(   0),  INT8_C(  12),  INT8_C(  76),  INT8_C(  26), -INT8_C( 118),  INT8_C(  41),  INT8_C(  37) },
      UINT32_C(1086460561) },
    { UINT32_C( 503967736),
      { -INT8_C( 106), -INT8_C(  98),  INT8_C(  52), -INT8_C(  17),  INT8_C( 119),      INT8_MIN, -INT8_C(  79),  INT8_C( 106),
         INT8_C(  80), -INT8_C(  35),  INT8_C(   8), -INT8_C( 116),  INT8_C( 102), -INT8_C(  94), -INT8_C(  62),  INT8_C(   9),
         INT8_C( 114), -INT8_C(  79), -INT8_C(  76), -INT8_C(  21), -INT8_C(  19), -INT8_C(  66),      INT8_MAX,  INT8_C(  21),
        -INT8_C(  14), -INT8_C(  69), -INT8_C(  49), -INT8_C(  57),  INT8_C(  17), -INT8_C(  51), -INT8_C( 120), -INT8_C(  89) },
      {  INT8_C( 107),  INT8_C(  32), -INT8_C( 106),  INT8_C(   0),  INT8_C(   8),  INT8_C(  72),  INT8_C(  76), -INT8_C( 116),
         INT8_C(  37),  INT8_C(  85),  INT8_C(  24),  INT8_C(   0), -INT8_C(   9),  INT8_C(  88), -INT8_C( 107),  INT8_C( 105),
        -INT8_C( 116),  INT8_C(  72),  INT8_C(  84),  INT8_C( 121),  INT8_C(   2), -INT8_C(  44), -INT8_C( 114), -INT8_C(   6),
        -INT8_C( 113),  INT8_C(  68), -INT8_C(  63), -INT8_C(  96),  INT8_C(  42),  INT8_C(   0),  INT8_C(  71), -INT8_C( 105) },
      UINT32_C( 201901696) },
    { UINT32_C(3893704031),
      { -INT8_C(  74), -INT8_C(  40),  INT8_C( 106), -INT8_C(  47), -INT8_C(  65),  INT8_C(  65), -INT8_C(  28),      INT8_MIN,
        -INT8_C(  82),  INT8_C(  95), -INT8_C(  50),  INT8_C(  99),  INT8_C(  47), -INT8_C( 107),  INT8_C(  32), -INT8_C(  45),
        -INT8_C(  22), -INT8_C(  41),  INT8_C(   6), -INT8_C(  98),  INT8_C(  80), -INT8_C(  39),  INT8_C( 125),  INT8_C(  18),
        -INT8_C(  12), -INT8_C(  13), -INT8_C(  39), -INT8_C(  20), -INT8_C(  30), -INT8_C(  29),  INT8_C(  11), -INT8_C( 104) },
      { -INT8_C(  69),  INT8_C( 117),  INT8_C( 106),  INT8_C( 122), -INT8_C(  74),  INT8_C(  14),  INT8_C(  26),  INT8_C( 100),
         INT8_C(   1),      INT8_MIN, -INT8_C(  56), -INT8_C( 100),  INT8_C(  80),  INT8_C( 104), -INT8_C( 112),  INT8_C(   8),
        -INT8_C(  64),  INT8_C(  32), -INT8_C(  32),  INT8_C(  16), -INT8_C( 112),  INT8_C(  34),  INT8_C(   2), -INT8_C( 124),
         INT8_C(   2), -INT8_C(   5),  INT8_C( 113),  INT8_C(  16), -INT8_C(  34),  INT8_C(  28), -INT8_C(  48),  INT8_C(   1) },
      UINT32_C(   1115167) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_test_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_test_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[32];
    int8_t b_[32];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < 32 ; j++)
      if (!(easysimd_test_codegen_random_i8() & 1))
        b_[j] &= ~a_[j];

    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_mm256_loadu_epi8(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(b_);
    easysimd__mmask32 r = easysimd_mm256_mask_test_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32( 2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_mask_test_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int16_t a[16];
    const int16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(34335),
      { -INT16_C( 16426),  INT16_C(  4582),  INT16_C( 28568),  INT16_C(  7215),  INT16_C( 18081),  INT16_C(  7549),  INT16_C(  9671), -INT16_C( 23391),
         INT16_C( 15699),  INT16_C( 20249), -INT16_C( 10404), -INT16_C( 29130), -INT16_C( 11749), -INT16_C( 21314),  INT16_C( 29339),  INT16_C( 29151) },
      {  INT16_C( 16417), -INT16_C( 14335), -INT16_C( 28635), -INT16_C( 10267),  INT16_C(  8280), -INT16_C( 16140), -INT16_C( 28672),  INT16_C( 23108),
         INT16_C( 32466),  INT16_C( 12074),  INT16_C( 24661),  INT16_C( 28861),  INT16_C( 31538), -INT16_C( 12771), -INT16_C( 29596),  INT16_C(  3616) },
      UINT16_C( 1544) },
    { UINT16_C(63389),
      { -INT16_C( 27479),  INT16_C( 30804), -INT16_C( 31487),  INT16_C( 11835), -INT16_C( 27843), -INT16_C( 12353), -INT16_C( 12355),  INT16_C(  6028),
        -INT16_C( 22221),  INT16_C(  2093), -INT16_C( 16392),  INT16_C(  5882), -INT16_C(  5826), -INT16_C(  9101), -INT16_C( 28006),  INT16_C( 17250) },
      { -INT16_C( 18650),  INT16_C( 10171), -INT16_C(  2500),  INT16_C( 20804),  INT16_C(  1152),  INT16_C( 17992), -INT16_C( 11036),  INT16_C(  5981),
        -INT16_C( 29827),  INT16_C( 30240),  INT16_C(     2), -INT16_C( 30580),  INT16_C(  5633), -INT16_C( 25244),  INT16_C( 17409), -INT16_C( 18208) },
      UINT16_C(41861) },
    { UINT16_C(59140),
      {  INT16_C( 11901), -INT16_C( 20692),  INT16_C( 30281), -INT16_C( 21974),  INT16_C( 31396), -INT16_C(  5784), -INT16_C( 19897), -INT16_C( 30129),
        -INT16_C( 12475),  INT16_C( 24921),  INT16_C( 10066),  INT16_C( 29664),  INT16_C(  9862), -INT16_C( 24971),  INT16_C(  4842),  INT16_C( 26517) },
      { -INT16_C( 16128), -INT16_C( 30442),  INT16_C(   310),  INT16_C( 20753), -INT16_C( 25413),  INT16_C(   708),  INT16_C(   264),  INT16_C(  4480),
         INT16_C(  8354),  INT16_C(  5284), -INT16_C( 12276), -INT16_C( 32760),  INT16_C(  6521), -INT16_C(  6864), -INT16_C( 15099),  INT16_C( 28748) },
      UINT16_C(40960) },
    { UINT16_C(37353),
      { -INT16_C( 22358),  INT16_C( 17172), -INT16_C(  8953),  INT16_C( 26784),  INT16_C(  4340), -INT16_C( 25003),  INT16_C( 10083), -INT16_C(  5315),
        -INT16_C( 20219),  INT16_C( 30236),  INT16_C( 27169),  INT16_C( 30231),  INT16_C( 32231), -INT16_C( 10985),  INT16_C(  7274),  INT16_C(  5564) },
      {  INT16_C( 20548), -INT16_C( 13224), -INT16_C(  1875), -INT16_C( 32492), -INT16_C( 30200),  INT16_C( 27711),  INT16_C( 22672),  INT16_C(  5186),
         INT16_C( 16938),  INT16_C(  2337),  INT16_C( 17629), -INT16_C( 14907), -INT16_C(  8767),  INT16_C( 11418),  INT16_C( 22265), -INT16_C( 17087) },
      UINT16_C(36896) },
    { UINT16_C(64655),
      {  INT16_C( 31324), -INT16_C( 12209), -INT16_C( 26861),  INT16_C(  6531), -INT16_C( 19312),  INT16_C( 16758),  INT16_C( 17939), -INT16_C( 16098),
         INT16_C( 16628),  INT16_C( 26421), -INT16_C( 31197),  INT16_C( 12916), -INT16_C( 10288), -INT16_C( 10110),  INT16_C( 27488), -INT16_C( 17303) },
      { -INT16_C( 32607),  INT16_C( 10368),  INT16_C(  4175), -INT16_C( 14832), -INT16_C( 30780), -INT16_C( 27136),  INT16_C( 14796),  INT16_C(   640),
        -INT16_C( 12673), -INT16_C( 24022),  INT16_C(  6228),  INT16_C(  1408),  INT16_C( 22134), -INT16_C( 10499),  INT16_C( 26305),  INT16_C(   658) },
      UINT16_C(28676) },
    { UINT16_C(30223),
      { -INT16_C( 16488), -INT16_C(  6740), -INT16_C(  6990),  INT16_C( 32082), -INT16_C(   279), -INT16_C( 20000),  INT16_C(  4393), -INT16_C(  2397),
         INT16_C( 13462),  INT16_C( 16791), -INT16_C( 29126),  INT16_C(  6120), -INT16_C(  2490), -INT16_C( 23417),  INT16_C(  5921), -INT16_C( 17760) },
      {  INT16_C( 16454),  INT16_C(  2067),  INT16_C(  4353),  INT16_C(   517), -INT16_C(  6673),  INT16_C(  6348),  INT16_C( 28662), -INT16_C( 29426),
        -INT16_C( 32479), -INT16_C(  8498), -INT16_C( 18893),  INT16_C( 26645),  INT16_C( 32172),  INT16_C( 18968), -INT16_C( 22380),  INT16_C( 27272) },
      UINT16_C( 5632) },
    { UINT16_C(24330),
      {  INT16_C( 10142), -INT16_C(  2962),  INT16_C( 19246),  INT16_C( 19734), -INT16_C( 22497), -INT16_C( 32646),  INT16_C( 19677), -INT16_C(  3921),
        -INT16_C( 27311), -INT16_C( 12296),  INT16_C( 32295), -INT16_C( 30870), -INT16_C( 15721),  INT16_C(  7023),  INT16_C( 32293), -INT16_C( 15215) },
      { -INT16_C(    91),  INT16_C(   912), -INT16_C( 12470),  INT16_C( 27168),  INT16_C(  4704),  INT16_C( 21632), -INT16_C( 28126),  INT16_C( 14148),
         INT16_C( 15407),  INT16_C( 22022),  INT16_C( 28858),  INT16_C( 21213),  INT16_C(  3104),  INT16_C( 22637), -INT16_C( 32566),  INT16_C( 28444) },
      UINT16_C( 3848) },
    { UINT16_C( 9430),
      { -INT16_C( 21077),  INT16_C( 23745), -INT16_C( 16952),  INT16_C( 14072),  INT16_C( 11935), -INT16_C( 18199), -INT16_C( 10093),  INT16_C( 20183),
         INT16_C( 26241),  INT16_C( 32230), -INT16_C( 13785),  INT16_C( 16869),  INT16_C( 19576), -INT16_C( 27274), -INT16_C( 32593),  INT16_C( 23284) },
      { -INT16_C( 19154), -INT16_C( 24010),  INT16_C(   562),  INT16_C(  4396),  INT16_C(  4416),  INT16_C( 29129), -INT16_C( 24339),  INT16_C(  8448),
        -INT16_C( 32762),  INT16_C( 12012),  INT16_C(  4432), -INT16_C( 22518), -INT16_C( 24315),  INT16_C( 18441),  INT16_C( 29542), -INT16_C( 31741) },
      UINT16_C(   64) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_test_epi16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_test_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[16];
    int16_t b_[16];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < 16 ; j++)
      if (!(easysimd_test_codegen_random_i16() & 1))
        b_[j] &= ~a_[j];

    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_mm256_loadu_epi16(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(b_);
    easysimd__mmask16 r = easysimd_mm256_mask_test_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16( 2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_mask_test_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(135),
      { -INT32_C(   522097702),  INT32_C(  1591087810), -INT32_C(   472653921),  INT32_C(    93736658),  INT32_C(   291227707),  INT32_C(   835924589), -INT32_C(   870150944),  INT32_C(   695603791) },
      {  INT32_C(      529445),  INT32_C(   144629609),  INT32_C(   271327264), -INT32_C(   100630528), -INT32_C(  1223888566),  INT32_C(    36228354),  INT32_C(   852296203), -INT32_C(  1696316132) },
      UINT8_C(130) },
    { UINT8_C( 39),
      { -INT32_C(  1904651137), -INT32_C(    58170400), -INT32_C(  1297218291),  INT32_C(  1231245771), -INT32_C(   525405417), -INT32_C(  1535631634), -INT32_C(   784443989),  INT32_C(  1322436302) },
      {  INT32_C(  1887701760),  INT32_C(    58065934),  INT32_C(    88186976),  INT32_C(    68976692), -INT32_C(  1309922109), -INT32_C(   162110389), -INT32_C(  2000186183),  INT32_C(  1976998234) },
      UINT8_C( 32) },
    { UINT8_C( 46),
      { -INT32_C(   963756680),  INT32_C(  1921030589), -INT32_C(  1515910247),  INT32_C(  1469973848),  INT32_C(  1353514159), -INT32_C(  1296525805), -INT32_C(  1826113062),  INT32_C(   414928543) },
      {  INT32_C(   156254851),  INT32_C(  2092654307), -INT32_C(  1037999767),  INT32_C(   354009190), -INT32_C(  1891565040),  INT32_C(   914431324),  INT32_C(  1992927447), -INT32_C(  2121366282) },
      UINT8_C( 46) },
    { UINT8_C(136),
      {  INT32_C(  1670117685),  INT32_C(   615674413),  INT32_C(   557105599),  INT32_C(   652806796), -INT32_C(  1539190980), -INT32_C(  1093335925), -INT32_C(   142570819),  INT32_C(  1157410318) },
      { -INT32_C(   727218265), -INT32_C(  1740109360), -INT32_C(  1495651302),  INT32_C(  1926021942),  INT32_C(   504761746),  INT32_C(  1074326288),  INT32_C(     4543554), -INT32_C(  1325383680) },
      UINT8_C(  8) },
    { UINT8_C(174),
      {  INT32_C(  1346589012), -INT32_C(  1138364877),  INT32_C(   588299983),  INT32_C(   864973546),  INT32_C(   626648292),  INT32_C(     2874544), -INT32_C(  1491471511),  INT32_C(  1500291588) },
      {  INT32_C(   648654835),  INT32_C(  1826803357),  INT32_C(  2005988237),  INT32_C(  1235885669), -INT32_C(  1843003390),  INT32_C(  1236441568), -INT32_C(  1846489972),  INT32_C(   545414250) },
      UINT8_C( 46) },
    { UINT8_C(210),
      {  INT32_C(  1404440382),  INT32_C(  1677202430), -INT32_C(   529286187), -INT32_C(   145062016), -INT32_C(  1205834117),  INT32_C(  1072043534),  INT32_C(  1046114425),  INT32_C(  1133119493) },
      {  INT32_C(    67125249),  INT32_C(   134319105),  INT32_C(   151536648), -INT32_C(   272604300),  INT32_C(  1200128384),  INT32_C(  1075349745),  INT32_C(    61765886), -INT32_C(  1756940160) },
      UINT8_C(192) },
    { UINT8_C(248),
      { -INT32_C(  1500561741),  INT32_C(  1923261171), -INT32_C(  1632501464),  INT32_C(  1086116401),  INT32_C(  1919772906), -INT32_C(  1062727150), -INT32_C(   189524169), -INT32_C(  1302493441) },
      {  INT32_C(   407945540), -INT32_C(  1221919857),  INT32_C(   541383892), -INT32_C(  1720839505), -INT32_C(  1929210863),  INT32_C(   642097644), -INT32_C(   904265013),  INT32_C(  1293971712) },
      UINT8_C( 72) },
    { UINT8_C(113),
      { -INT32_C(   311048382), -INT32_C(  1913749101), -INT32_C(  1862925764), -INT32_C(  1393608048), -INT32_C(   971636115), -INT32_C(  2026274337), -INT32_C(  1230363531), -INT32_C(  1241734541) },
      {  INT32_C(   276967464),  INT32_C(   805376096),  INT32_C(   616595859),  INT32_C(    17858603),  INT32_C(   300541074),  INT32_C(  1476405280), -INT32_C(  1063992499),  INT32_C(  1241583620) },
      UINT8_C( 64) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_test_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_test_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[8];
    int32_t b_[8];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < 8 ; j++)
      if (!(easysimd_test_codegen_random_i32() & 1))
        b_[j] &= ~a_[j];

    easysimd__mmask8 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_mm256_loadu_epi32(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(b_);
    easysimd__mmask8 r = easysimd_mm256_mask_test_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8( 2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_test_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(134),
      { -INT64_C( 1474422343923538257),  INT64_C( 8471674331771489254), -INT64_C( 3794005404919717864),  INT64_C( 2902917925816995333) },
      {  INT64_C(  303449877613121536),  INT64_C(  721298405418206209),  INT64_C( 1487879271123803044),  INT64_C( 9035006291336043968) },
      UINT8_C(  0) },
    { UINT8_C(147),
      {  INT64_C( 6128190466885730063),  INT64_C( 7343379998160017718), -INT64_C( 8064997200576035429), -INT64_C( 1191087885606154046) },
      {  INT64_C(  735284198310740000), -INT64_C( 8069887563829870584),  INT64_C(  657666287388371008), -INT64_C( 2595070968594739924) },
      UINT8_C(  0) },
    { UINT8_C( 95),
      { -INT64_C( 7529988811569400573),  INT64_C( 8506722292601082077), -INT64_C( 8785203380115794413), -INT64_C( 5259776979315789652) },
      {  INT64_C( 2315554754903837432),  INT64_C(  730580360936359123),  INT64_C( 2537917438287857312),  INT64_C( 4673068741859279682) },
      UINT8_C(  6) },
    { UINT8_C(109),
      { -INT64_C( 1550962991058197635),  INT64_C( 7728245052065678083), -INT64_C( 8673653654515195018), -INT64_C( 4029589943874916137) },
      {  INT64_C(   73183520944030848),  INT64_C( 8259376300405473094),  INT64_C( 5553461117696309653),  INT64_C( 1246110269055339272) },
      UINT8_C(  4) },
    { UINT8_C( 84),
      { -INT64_C( 3453443298068026537), -INT64_C( 7172562119808350842), -INT64_C( 2675843219334905343),  INT64_C( 3527367277072070365) },
      {  INT64_C(  342581434957169696),  INT64_C( 3114951804607216094),  INT64_C( 2675279053636771914),  INT64_C( 1082066784879640610) },
      UINT8_C(  0) },
    { UINT8_C( 96),
      { -INT64_C( 2632123826456870891), -INT64_C( 5778904768325508341),  INT64_C( 4978974915204631070), -INT64_C( 5772471670163679340) },
      {  INT64_C( 2307292872611468874),  INT64_C( 5764829919930482704),  INT64_C( 1351925071205209376),  INT64_C(  931336639536541347) },
      UINT8_C(  0) },
    { UINT8_C( 76),
      { -INT64_C( 4204866888779778517),  INT64_C( 1833392542339406548),  INT64_C( 7691016375220683599), -INT64_C( 6759892634778023706) },
      {  INT64_C(  745363502599768512),  INT64_C( 3285080011303901615),  INT64_C( 1244124108640358576), -INT64_C( 3186522054022985042) },
      UINT8_C(  8) },
    { UINT8_C( 70),
      {  INT64_C( 2016899226662578118),  INT64_C( 4372198458943154701),  INT64_C( 1660719834417783023),  INT64_C( 1250257189029142742) },
      { -INT64_C( 4611113427899756496), -INT64_C( 6491121141108012199), -INT64_C( 4917178844514953784), -INT64_C( 1394391279826626551) },
      UINT8_C(  6) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_test_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_test_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[4];
    int64_t b_[4];

    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));
    for (size_t j = 0 ; j < 4 ; j++)
      if (!(easysimd_test_codegen_random_i8() & 1))
        b_[j] &= ~a_[j];

    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_mm256_loadu_epi64(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(b_);
    easysimd__mmask8 r = easysimd_mm256_mask_test_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4( 2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_mask_test_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const int16_t a[32];
    const int16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C( 729589500),
      { -INT16_C( 21039),  INT16_C(  7594), -INT16_C( 26143), -INT16_C(  3584),  INT16_C( 10710),  INT16_C( 22134),  INT16_C( 11147), -INT16_C( 19843),
         INT16_C( 21651),  INT16_C( 20557),  INT16_C( 25768),  INT16_C( 18307), -INT16_C( 21839), -INT16_C( 13908),  INT16_C(  9876), -INT16_C(  8232),
         INT16_C( 16084),  INT16_C(  8140), -INT16_C( 13132),  INT16_C(  2833), -INT16_C( 29249), -INT16_C(  5655), -INT16_C( 24960), -INT16_C( 10165),
         INT16_C( 11736), -INT16_C( 31959), -INT16_C( 21330),  INT16_C( 21460), -INT16_C( 23722), -INT16_C( 23028), -INT16_C( 20057),  INT16_C( 31534) },
      {  INT16_C( 21038), -INT16_C(  7595),  INT16_C( 26142), -INT16_C( 24339), -INT16_C( 10711), -INT16_C( 22135), -INT16_C( 11148),  INT16_C( 19842),
        -INT16_C( 21652),  INT16_C(  6864), -INT16_C( 25769), -INT16_C( 21127), -INT16_C( 31170),  INT16_C( 13907), -INT16_C(  9877), -INT16_C( 26260),
        -INT16_C( 16085),  INT16_C( 18812),  INT16_C( 26919),  INT16_C( 20969),  INT16_C( 29248), -INT16_C( 19206),  INT16_C( 31814), -INT16_C( 19711),
        -INT16_C( 11737),  INT16_C( 32461),  INT16_C( 18285), -INT16_C( 21461),  INT16_C( 32717),  INT16_C( 14562),  INT16_C( 20056), -INT16_C( 31535) },
      UINT32_C( 577536520) },
    { UINT32_C( 302971227),
      { -INT16_C( 13115), -INT16_C( 19071),  INT16_C( 12077),  INT16_C(  8771),  INT16_C( 27746),  INT16_C( 13498), -INT16_C(  8099), -INT16_C( 27443),
        -INT16_C(  2264), -INT16_C( 16926), -INT16_C( 10784), -INT16_C( 22105), -INT16_C( 11549),  INT16_C(  9016), -INT16_C( 28981), -INT16_C( 14112),
         INT16_C( 23332),  INT16_C( 15189), -INT16_C( 15727),  INT16_C( 13431), -INT16_C( 29429),  INT16_C( 26707),  INT16_C(  8302),  INT16_C(  6606),
        -INT16_C( 27258),  INT16_C( 26398), -INT16_C( 14997),  INT16_C( 19984),  INT16_C( 32041),  INT16_C( 11498), -INT16_C( 13609),  INT16_C( 17374) },
      {  INT16_C( 13114),  INT16_C( 19070), -INT16_C( 12078), -INT16_C(  8772),  INT16_C(  3933), -INT16_C( 13499),  INT16_C( 16687), -INT16_C( 18710),
         INT16_C(  2263),  INT16_C( 16925),  INT16_C( 11725),  INT16_C( 25744),  INT16_C(  7869), -INT16_C(  9017), -INT16_C( 22787),  INT16_C( 14111),
        -INT16_C( 24871), -INT16_C( 21631),  INT16_C( 15726), -INT16_C( 13432), -INT16_C( 12980),  INT16_C( 31639), -INT16_C( 32497), -INT16_C(  6607),
         INT16_C( 20105),  INT16_C( 22056), -INT16_C( 18309),  INT16_C( 14778), -INT16_C( 32042), -INT16_C( 11499),  INT16_C( 13608),  INT16_C(   266) },
      UINT32_C(  33708112) },
    { UINT32_C(3777173016),
      { -INT16_C(   627),  INT16_C(  2271), -INT16_C( 31277), -INT16_C( 28867),  INT16_C(  6684),  INT16_C( 22437), -INT16_C( 31723),  INT16_C( 20962),
        -INT16_C( 22682), -INT16_C( 23040),  INT16_C( 32539), -INT16_C( 28658), -INT16_C(  1712),  INT16_C( 10354), -INT16_C( 11131),  INT16_C(  4868),
        -INT16_C(    60),  INT16_C( 17436), -INT16_C( 30076), -INT16_C( 13425),  INT16_C(  1908),  INT16_C( 23586), -INT16_C( 28532),  INT16_C( 17326),
        -INT16_C( 20820), -INT16_C( 16377),  INT16_C( 27569),  INT16_C(   620),  INT16_C( 12144), -INT16_C(  5489), -INT16_C( 27410), -INT16_C( 16130) },
      {  INT16_C(   626), -INT16_C(  2272), -INT16_C( 20595),  INT16_C( 28866), -INT16_C(  6685), -INT16_C( 16435),  INT16_C( 31722), -INT16_C( 27134),
        -INT16_C(  5079),  INT16_C( 23491), -INT16_C( 32540),  INT16_C( 21660),  INT16_C(  1711),  INT16_C( 29522),  INT16_C( 11130), -INT16_C(  4869),
         INT16_C(  6958), -INT16_C( 17437), -INT16_C( 22838), -INT16_C( 21205), -INT16_C(  1909),  INT16_C( 30061),  INT16_C( 28531), -INT16_C( 25333),
        -INT16_C( 12709),  INT16_C( 16376), -INT16_C( 27570), -INT16_C(   621), -INT16_C(  6501),  INT16_C(  5488),  INT16_C( 27409),  INT16_C( 16129) },
      UINT32_C(  18940416) },
    { UINT32_C(3714267986),
      {  INT16_C( 25185),  INT16_C( 12792),  INT16_C( 10477), -INT16_C( 22254), -INT16_C( 21385),  INT16_C( 30944),  INT16_C( 22376), -INT16_C( 32250),
        -INT16_C(  9887), -INT16_C(  2295),  INT16_C(  3391), -INT16_C( 24650), -INT16_C( 27822), -INT16_C( 23404), -INT16_C(  2097),  INT16_C( 12417),
         INT16_C( 29785),  INT16_C( 10160), -INT16_C( 15630),  INT16_C(  1744), -INT16_C( 20221),  INT16_C( 27518), -INT16_C( 22520),  INT16_C( 27023),
         INT16_C( 10865),  INT16_C( 31162),  INT16_C(  7223),  INT16_C( 16892), -INT16_C( 28512),  INT16_C( 28645),  INT16_C( 26503), -INT16_C(  8033) },
      {  INT16_C( 20699), -INT16_C( 12793), -INT16_C( 10478),  INT16_C(  5844),  INT16_C( 21384), -INT16_C( 28543),  INT16_C(  4347),  INT16_C( 32249),
        -INT16_C( 19687),  INT16_C(  2294), -INT16_C(  3392),  INT16_C( 24649),  INT16_C( 11906),  INT16_C(  2511),  INT16_C( 28309),  INT16_C( 29161),
        -INT16_C(  3906), -INT16_C( 11969),  INT16_C(  5064),  INT16_C( 20711),  INT16_C( 26726),  INT16_C( 25313), -INT16_C(  9607), -INT16_C( 27937),
        -INT16_C( 10866),  INT16_C( 20122), -INT16_C(  7224),  INT16_C( 19118),  INT16_C( 32018), -INT16_C( 22700),  INT16_C( 15851), -INT16_C( 21992) },
      UINT32_C(3630379328) },
    { UINT32_C(3903854973),
      { -INT16_C(  2437), -INT16_C( 21731), -INT16_C(  4834),  INT16_C(  8329),  INT16_C( 11606),  INT16_C(  3105), -INT16_C( 19486),  INT16_C(  1449),
         INT16_C(  8891),  INT16_C(  9098), -INT16_C( 26655),  INT16_C( 27927),  INT16_C( 22472), -INT16_C( 24819), -INT16_C( 10289),  INT16_C(  8872),
        -INT16_C(  8755), -INT16_C(  5231),  INT16_C(  6859),  INT16_C( 12017),  INT16_C(  5085),  INT16_C( 19123), -INT16_C(  6969),  INT16_C( 25454),
        -INT16_C(  1827), -INT16_C( 29521), -INT16_C( 28888),  INT16_C( 19755), -INT16_C( 13338), -INT16_C( 28015), -INT16_C( 21719),  INT16_C(  2485) },
      { -INT16_C(   119),  INT16_C( 21730), -INT16_C(  4582), -INT16_C( 27617), -INT16_C( 11607), -INT16_C(  3106),  INT16_C( 19485), -INT16_C(  1450),
        -INT16_C(  8892), -INT16_C( 11335), -INT16_C(  7045), -INT16_C( 12512), -INT16_C( 22473),  INT16_C( 24818),  INT16_C( 24916), -INT16_C(  8873),
         INT16_C( 14944),  INT16_C( 31281),  INT16_C( 20520), -INT16_C( 12018), -INT16_C(  5086),  INT16_C( 16324),  INT16_C(  6968),  INT16_C( 31801),
        -INT16_C(  3336),  INT16_C( 29520),  INT16_C( 28887),  INT16_C(  3650),  INT16_C( 13337),  INT16_C( 28014), -INT16_C( 14699), -INT16_C(  2486) },
      UINT32_C(1218447373) },
    { UINT32_C(4154045996),
      { -INT16_C( 17182), -INT16_C( 24092),  INT16_C(  4177),  INT16_C( 10821), -INT16_C( 30163), -INT16_C( 31643), -INT16_C( 29846), -INT16_C( 18436),
         INT16_C(  9984),  INT16_C( 24019),  INT16_C( 16234), -INT16_C( 26639),  INT16_C(  5577),  INT16_C(  9464),  INT16_C( 25036),  INT16_C(   492),
        -INT16_C( 12044),  INT16_C( 17826), -INT16_C(  6176),  INT16_C( 21871), -INT16_C( 10143), -INT16_C( 13115), -INT16_C(  9373),  INT16_C( 25822),
        -INT16_C( 19965),  INT16_C( 28097), -INT16_C( 19471), -INT16_C( 11259), -INT16_C(   701), -INT16_C( 30727),  INT16_C( 19902), -INT16_C( 19808) },
      {  INT16_C( 17181), -INT16_C(   520),  INT16_C( 26410),  INT16_C( 30546),  INT16_C( 30162),  INT16_C( 31642),  INT16_C( 29845), -INT16_C(  4374),
        -INT16_C(  9985), -INT16_C( 24020), -INT16_C( 16235),  INT16_C( 26638),  INT16_C( 28445), -INT16_C(  9465), -INT16_C( 22340), -INT16_C(  9843),
        -INT16_C( 31253),  INT16_C(  5590),  INT16_C( 10733), -INT16_C( 16499),  INT16_C( 10142),  INT16_C( 13114),  INT16_C(  9372), -INT16_C( 25823),
         INT16_C( 19964), -INT16_C( 28098),  INT16_C( 19470),  INT16_C( 11258),  INT16_C(   700),  INT16_C( 30726), -INT16_C( 27734), -INT16_C( 27310) },
      UINT32_C(3221852168) },
    { UINT32_C( 562338000),
      {  INT16_C( 18896),  INT16_C(  2168), -INT16_C( 14015),  INT16_C( 27690),  INT16_C( 30605),  INT16_C(  4579),  INT16_C( 13180), -INT16_C( 18771),
        -INT16_C( 22799), -INT16_C( 25709), -INT16_C(  6070), -INT16_C( 20492), -INT16_C(  4029), -INT16_C(  1922), -INT16_C(  8877),  INT16_C( 19329),
        -INT16_C( 27654),  INT16_C(  1068), -INT16_C( 23352),  INT16_C( 17071),  INT16_C( 15460), -INT16_C(  1714), -INT16_C(  1157), -INT16_C(  5392),
         INT16_C( 15077), -INT16_C( 27727), -INT16_C( 17384),  INT16_C( 23399), -INT16_C(  6483),  INT16_C(   340),  INT16_C(  1988), -INT16_C( 30939) },
      {  INT16_C(  6703), -INT16_C(  2169),  INT16_C( 14014),  INT16_C(  9017), -INT16_C( 30606), -INT16_C(  4580), -INT16_C( 13181), -INT16_C(  2218),
         INT16_C( 22798),  INT16_C( 14773),  INT16_C(  8785),  INT16_C( 11508),  INT16_C(  4028),  INT16_C(  1921),  INT16_C(  8876), -INT16_C(  9396),
        -INT16_C( 11204), -INT16_C(  1069),  INT16_C(  3082),  INT16_C( 32030),  INT16_C( 14996),  INT16_C(  6251), -INT16_C( 16122),  INT16_C(  5391),
        -INT16_C( 15078),  INT16_C( 27726),  INT16_C( 17383), -INT16_C( 23400),  INT16_C(  6482), -INT16_C(   341), -INT16_C(  1989),  INT16_C( 30938) },
      UINT32_C(    297088) },
    { UINT32_C( 168615504),
      {  INT16_C( 24431),  INT16_C( 14936),  INT16_C(  3341), -INT16_C( 25332), -INT16_C(  3277),  INT16_C( 28215),  INT16_C( 15657), -INT16_C( 24176),
        -INT16_C(  2560),  INT16_C(  7101), -INT16_C(  3073),  INT16_C( 12678),  INT16_C( 29635),  INT16_C( 10854),  INT16_C( 29391), -INT16_C( 21077),
        -INT16_C( 29486),  INT16_C(  3002), -INT16_C( 31414), -INT16_C( 25440), -INT16_C( 27411), -INT16_C( 24085),  INT16_C( 31490),  INT16_C( 17773),
        -INT16_C( 20527), -INT16_C( 26255), -INT16_C( 17121), -INT16_C(  7591), -INT16_C( 17871), -INT16_C( 18455),  INT16_C(  7587),  INT16_C( 30197) },
      {  INT16_C( 28073), -INT16_C(  3227), -INT16_C(  3342),  INT16_C( 27504),  INT16_C( 23735), -INT16_C( 17908),  INT16_C( 20439), -INT16_C( 22120),
         INT16_C(  2559), -INT16_C(  7102),  INT16_C(  3072), -INT16_C(  5981), -INT16_C( 29636), -INT16_C(  8289), -INT16_C( 27479),  INT16_C( 21076),
        -INT16_C( 18175), -INT16_C(  3003), -INT16_C( 18772),  INT16_C( 25439),  INT16_C( 27410), -INT16_C(  5859), -INT16_C( 19013), -INT16_C( 17774),
        -INT16_C( 11074), -INT16_C( 16738),  INT16_C( 17120),  INT16_C(  7590),  INT16_C( 17870),  INT16_C( 30972),  INT16_C( 20953), -INT16_C(  9270) },
      UINT32_C(  33835088) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k1 = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_test_epi16_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_test_epi16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[32];
    int16_t b_[32];
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 32 ; j++) if (rand() & 1) a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi16(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(b_);
    easysimd__mmask32 r = easysimd_mm512_mask_test_epi16_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_test_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(13733),
      easysimd_mm512_set_epi32(INT32_C(-1058044212), INT32_C( 1745554146), INT32_C( -938028173), INT32_C( 1123843978),
                            INT32_C(          0), INT32_C(  369104615), INT32_C(  288860030), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( 1441724088), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)),
      easysimd_mm512_set_epi32(INT32_C( -173456664), INT32_C(          0), INT32_C( 1847983972), INT32_C(          0),
                            INT32_C( 1618889484), INT32_C(-1607295283), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1566474746), INT32_C(          0), INT32_C(          0), INT32_C( -427236235),
                            INT32_C( 1934991438), INT32_C( -587656653), INT32_C(  -72145550), INT32_C(          0)),
      UINT16_C( 9216) },
    { UINT16_C(63633),
      easysimd_mm512_set_epi32(INT32_C( -839841286), INT32_C(          0), INT32_C(-1528318158), INT32_C(-1752243244),
                            INT32_C(          0), INT32_C(-1210241348), INT32_C(-1908519219), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(  521483609), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(  543914332),
                            INT32_C( 2140721152), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)),
      UINT16_C( 4096) },
    { UINT16_C(46032),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(  329000619),
                            INT32_C(-1407469850), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( -102278232), INT32_C(   43450551), INT32_C(          0), INT32_C( 1603148048),
                            INT32_C(          0), INT32_C(  306941628), INT32_C(          0), INT32_C( 1716086600)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( -262488052), INT32_C(          0), INT32_C( -979944078),
                            INT32_C(          0), INT32_C(          0), INT32_C( -671140903), INT32_C( 2141932562),
                            INT32_C( -937827627), INT32_C( -402281975), INT32_C( 1596901579), INT32_C(  735909604),
                            INT32_C(          0), INT32_C(-1202496908), INT32_C(          0), INT32_C( 1187626051)),
      UINT16_C( 4304) },
    { UINT16_C(22689),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C( 1163951893), INT32_C(          0),
                            INT32_C(          0), INT32_C( -972105386), INT32_C(          0), INT32_C( -189029459),
                            INT32_C( -225058862), INT32_C(-1537122616), INT32_C(          0), INT32_C(-1756788873),
                            INT32_C(          0), INT32_C(          0), INT32_C( -184793757), INT32_C( 1418114595)),
      easysimd_mm512_set_epi32(INT32_C(-1354385088), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(-1841164366), INT32_C(  -94325362), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(  594617140), INT32_C(          0),
                            INT32_C(  239227634), INT32_C(          0), INT32_C(  895132574), INT32_C( -123965580)),
      UINT16_C(    1) },
    { UINT16_C(25521),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( 1916556758), INT32_C(  729046451), INT32_C(  516587379),
                            INT32_C( 1306886393), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( 2010256146), INT32_C( -351991039), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)),
      easysimd_mm512_set_epi32(INT32_C(-1549799175), INT32_C( -926637576), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1633420708), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1070649171), INT32_C( -629074644),
                            INT32_C( -658686596), INT32_C( 1058031607), INT32_C(          0), INT32_C(  590507721)),
      UINT16_C(16416) },
    { UINT16_C(62461),
      easysimd_mm512_set_epi32(INT32_C(-1777930200), INT32_C(-2074304971), INT32_C(-1238248254), INT32_C(          0),
                            INT32_C(          0), INT32_C(-2126101806), INT32_C( 1488735838), INT32_C(-1941730876),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1287988933), INT32_C( -234166092),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C( 1899039383)),
      easysimd_mm512_set_epi32(INT32_C(-2019508611), INT32_C(          0), INT32_C(  630346390), INT32_C( 1865079685),
                            INT32_C( -135665525), INT32_C(          0), INT32_C(  751170486), INT32_C(  580518147),
                            INT32_C(  875750845), INT32_C(  263799439), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( -889468036), INT32_C(  606319308), INT32_C(          0)),
      UINT16_C(41728) },
    { UINT16_C(64894),
      easysimd_mm512_set_epi32(INT32_C( 1983606396), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1668321234), INT32_C( 1230207891), INT32_C(-1334904081), INT32_C(-1369460101),
                            INT32_C(          0), INT32_C(          0), INT32_C(  556969985), INT32_C(  844640930),
                            INT32_C(-1447169687), INT32_C( -621740127), INT32_C( 1847803343), INT32_C(          0)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C( -578157153),
                            INT32_C(          0), INT32_C(-1740019927), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(  136767631), INT32_C(          0), INT32_C(-1994887012),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-1203448248)),
      UINT16_C( 1040) },
    { UINT16_C(   72),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( -302271050), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1073370549), INT32_C(  911478852), INT32_C( 2032788905), INT32_C(          0),
                            INT32_C(  398076492), INT32_C(  901163928), INT32_C(          0), INT32_C(-1022133203),
                            INT32_C( -345543230), INT32_C(          0), INT32_C( 2110415254), INT32_C(          0)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( 1062395075), INT32_C( -125220770), INT32_C(          0),
                            INT32_C(          0), INT32_C(-1603705335), INT32_C(-1563808089), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1591467282), INT32_C(  -57010818), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1904594022), INT32_C(          0), INT32_C( -170170916)),
      UINT16_C(   64) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_test_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_test_epi32_mask");
    easysimd_assert_equal_mmask16(r, HEDLEY_STATIC_CAST(uint16_t, test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_test_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask8 r;
  } test_vec[8] = {
      { UINT8_C( 51),
      easysimd_mm512_set_epi64(INT64_C(-1021777151925940720), INT64_C(-8606874489654438743),
                            INT64_C( 2982642907250026668), INT64_C( 1121557266302837638),
                            INT64_C( 2431134246593116065), INT64_C(  613583019816833791),
                            INT64_C( 1064733599562411669), INT64_C( 2423175833785490495)),
      easysimd_mm512_set_epi64(INT64_C( 1320056489604494378), INT64_C( 8016850443050245285),
                            INT64_C(-1985397303830803362), INT64_C( 3490054501014041612),
                            INT64_C(-5459290507928993205), INT64_C( 7649116850053105819),
                            INT64_C( 2306883418990405350), INT64_C( -533675138939657219)),
      UINT8_C( 51) },
    { UINT8_C(107),
      easysimd_mm512_set_epi64(INT64_C(-5107242476853461048), INT64_C( 8011490560794608442),
                            INT64_C( 6781928010904462143), INT64_C(-2849262341734469560),
                            INT64_C( -680409599063304510), INT64_C(-5566534159487294326),
                            INT64_C( 1845249206901740837), INT64_C( -995351966812894672)),
      easysimd_mm512_set_epi64(INT64_C( 5576021829988608395), INT64_C( 6950975376569962720),
                            INT64_C(-6894298607029431267), INT64_C(  627810606408873629),
                            INT64_C( 5554191263207946561), INT64_C(-8076655846804767103),
                            INT64_C(  298038848695084217), INT64_C(  984575648746107146)),
      UINT8_C(107) },
    { UINT8_C( 37),
      easysimd_mm512_set_epi64(INT64_C( 7893528652770105648), INT64_C( 6530680127391412513),
                            INT64_C( 8552423929938430730), INT64_C(-2756847038438887047),
                            INT64_C(-4669427049793346259), INT64_C(-1241821792536551210),
                            INT64_C(-3299061013747874632), INT64_C(-6836802816564857048)),
      easysimd_mm512_set_epi64(INT64_C(-6002777026216974201), INT64_C(-4636334703117125394),
                            INT64_C( 1607894826575669284), INT64_C( 4780984352817723804),
                            INT64_C( 6091782245426739371), INT64_C(  678622961946838177),
                            INT64_C(-5016020090857558708), INT64_C( 6626170086889501267)),
      UINT8_C( 37) },
    { UINT8_C(219),
      easysimd_mm512_set_epi64(INT64_C(-2186517171747754940), INT64_C(-7709006576096545178),
                            INT64_C(-4148769136791016488), INT64_C(-1135438727484114896),
                            INT64_C( 8260953320246547916), INT64_C(-1100980776979386438),
                            INT64_C(-5918460005278991975), INT64_C(-9003997367625569086)),
      easysimd_mm512_set_epi64(INT64_C( 4216784995205036162), INT64_C( -610629877628633270),
                            INT64_C( 8989369503655563805), INT64_C( 7802074548335401303),
                            INT64_C(-8877282827036376912), INT64_C(  503689220758847744),
                            INT64_C(-1271186344370845002), INT64_C(-7863377575460447119)),
      UINT8_C(219) },
    { UINT8_C(231),
      easysimd_mm512_set_epi64(INT64_C(-1315792135193853217), INT64_C(  645488177526442731),
                            INT64_C(-9165154544452808856), INT64_C( 6660247336280542891),
                            INT64_C( 6275701276122863077), INT64_C( 6727198969638040444),
                            INT64_C(-7884988216343642316), INT64_C(-2772157409458686119)),
      easysimd_mm512_set_epi64(INT64_C(-3947691112861846019), INT64_C(-6027683744476655452),
                            INT64_C( 7878786769508596925), INT64_C(  898779895364838727),
                            INT64_C( 1691913248175202869), INT64_C(  925600519637433490),
                            INT64_C(-7786666185996880531), INT64_C(   83985053119720048)),
      UINT8_C(231) },
    { UINT8_C(185),
      easysimd_mm512_set_epi64(INT64_C(-1840026123275896761), INT64_C( 2958979742732699034),
                            INT64_C( 8087751231081622093), INT64_C(-8600769129449664089),
                            INT64_C( 4437078864330373252), INT64_C(-7780807660685149956),
                            INT64_C(-2075154109213332653), INT64_C( 4181892367727099352)),
      easysimd_mm512_set_epi64(INT64_C(  774147715399030450), INT64_C( -164954981177608742),
                            INT64_C( 2704996814905355295), INT64_C(-8428935692581869552),
                            INT64_C( 9058580247051597222), INT64_C(-5265103403663862315),
                            INT64_C(  -26456365246944797), INT64_C(-2390663458830548776)),
      UINT8_C(185) },
    { UINT8_C(101),
      easysimd_mm512_set_epi64(INT64_C( -500144007601202828), INT64_C( 1632600048776630049),
                            INT64_C( -841897496147484648), INT64_C(-9027608573250153699),
                            INT64_C( 5145549241007852287), INT64_C( 3253065934430492129),
                            INT64_C(-7159974623801904746), INT64_C(-5093282550607071469)),
      easysimd_mm512_set_epi64(INT64_C(-5525231520825547752), INT64_C(-8701772324698157173),
                            INT64_C(-8951932473020511764), INT64_C( 3775424668993203418),
                            INT64_C( 6848385375992501495), INT64_C(  530028931531319450),
                            INT64_C(-2966939250988120843), INT64_C( 8223727294178057916)),
      UINT8_C(101) },
    { UINT8_C( 49),
      easysimd_mm512_set_epi64(INT64_C(-6089529524580685715), INT64_C(-7930651770137634148),
                            INT64_C( 5513049874271089607), INT64_C( -418315818469941774),
                            INT64_C( 5583867531021541749), INT64_C( 3646229040969122240),
                            INT64_C(  298432255739992787), INT64_C( 8383322056459699927)),
      easysimd_mm512_set_epi64(INT64_C( 6173222450906561792), INT64_C(-2639921841557235108),
                            INT64_C( 1632309106004579853), INT64_C(-5099037996367773534),
                            INT64_C(-4229050909659913228), INT64_C(-6254506698641899913),
                            INT64_C( 1797655404582044593), INT64_C( 7065183211262770455)),
      UINT8_C( 49) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_test_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_test_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_test_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask64 k1;
    const int8_t a[64];
    const int8_t b[64];
    const easysimd__mmask64 r;
  } test_vec[] = {
    { UINT64_C(15709700008231627249),
      {  INT8_C(  54),  INT8_C(  59),  INT8_C(  44), -INT8_C(  56),  INT8_C(  94), -INT8_C(  10),  INT8_C(  44), -INT8_C( 123),
         INT8_C(  39),  INT8_C(  69), -INT8_C(  11),  INT8_C(  99),  INT8_C(  87), -INT8_C(  46),  INT8_C(   7), -INT8_C(  48),
        -INT8_C( 115), -INT8_C( 104),  INT8_C(  62),  INT8_C( 125),  INT8_C(  11), -INT8_C(  24),  INT8_C( 106), -INT8_C(  84),
        -INT8_C(  99),  INT8_C(   3), -INT8_C( 125),  INT8_C(   4), -INT8_C( 100), -INT8_C( 121), -INT8_C( 105),  INT8_C(  74),
         INT8_C(  40), -INT8_C(   2), -INT8_C(  94), -INT8_C(  95),  INT8_C( 116), -INT8_C(  35),  INT8_C(  38), -INT8_C( 101),
         INT8_C(  21), -INT8_C(  29), -INT8_C(  25),  INT8_C( 113),  INT8_C(  35), -INT8_C(  18),  INT8_C(  34), -INT8_C(  80),
        -INT8_C( 122),  INT8_C( 124), -INT8_C(  47),  INT8_C( 111),  INT8_C(  30),  INT8_C(  95), -INT8_C( 113), -INT8_C(  54),
         INT8_C(  65),  INT8_C(  18), -INT8_C(  49), -INT8_C(  34), -INT8_C( 103), -INT8_C(  94), -INT8_C(  72), -INT8_C(  92) },
      { -INT8_C(  61), -INT8_C(  60), -INT8_C(  45),  INT8_C(  55), -INT8_C(  95), -INT8_C(   7), -INT8_C(  45), -INT8_C( 103),
         INT8_C( 121), -INT8_C(  70),  INT8_C(  10), -INT8_C( 100), -INT8_C(  88),  INT8_C(  45),  INT8_C(  77),  INT8_C(  47),
         INT8_C(  99),  INT8_C(   7), -INT8_C(  63), -INT8_C( 126),  INT8_C(  43),  INT8_C(  80),  INT8_C( 113),  INT8_C(  83),
         INT8_C(  98),  INT8_C(  30), -INT8_C( 114), -INT8_C(   5),  INT8_C(  99),  INT8_C(  19),  INT8_C(  45),  INT8_C(  38),
        -INT8_C(  41),  INT8_C(   1),  INT8_C(  93),  INT8_C( 120), -INT8_C(   6),  INT8_C(  48),  INT8_C(  17),  INT8_C( 116),
        -INT8_C(  22),  INT8_C(  28),  INT8_C(  16), -INT8_C( 109),  INT8_C(  73),  INT8_C(  93), -INT8_C(  62), -INT8_C(  84),
         INT8_C( 100), -INT8_C( 125),  INT8_C(  46), -INT8_C( 112), -INT8_C(  45), -INT8_C(  96), -INT8_C(  29),  INT8_C(  53),
        -INT8_C(  66),  INT8_C( 113),  INT8_C(  48),  INT8_C(  33), -INT8_C( 124),  INT8_C(  93),  INT8_C(  71),  INT8_C(  91) },
      UINT64_C( 1297055008672465313) },
    { UINT64_C(16213930982914302159),
      { -INT8_C(  18),  INT8_C( 103), -INT8_C(  84),  INT8_C(  66), -INT8_C(  63), -INT8_C(  47), -INT8_C(  93), -INT8_C( 103),
        -INT8_C(  81), -INT8_C(  71), -INT8_C( 125), -INT8_C( 126),  INT8_C(  60),  INT8_C(  73), -INT8_C( 117), -INT8_C(   3),
        -INT8_C(  77), -INT8_C(  62), -INT8_C( 106),  INT8_C( 116),  INT8_C( 126), -INT8_C(  98), -INT8_C(  41),  INT8_C(  68),
        -INT8_C(  59),  INT8_C(  63),  INT8_C(  19),  INT8_C(  67), -INT8_C(  76), -INT8_C(  59),  INT8_C( 104), -INT8_C(  94),
         INT8_C(  22), -INT8_C(  70),  INT8_C(  30),  INT8_C(  64), -INT8_C(  25), -INT8_C(  63),  INT8_C(  63), -INT8_C( 106),
        -INT8_C(  61),  INT8_C( 124), -INT8_C(  34),  INT8_C(   0), -INT8_C(  10),  INT8_C(  97), -INT8_C(  61),  INT8_C(  22),
         INT8_C(  63),  INT8_C(  97), -INT8_C( 117),  INT8_C(  90), -INT8_C(  85), -INT8_C(   5), -INT8_C(  97), -INT8_C(  58),
         INT8_C(  58), -INT8_C(  11), -INT8_C(  32), -INT8_C(  18), -INT8_C(  17),  INT8_C(   5),  INT8_C( 102),  INT8_C( 110) },
      { -INT8_C(  42), -INT8_C(  81), -INT8_C(  82), -INT8_C(  67),  INT8_C( 112),  INT8_C(  46),  INT8_C(  84),  INT8_C( 102),
         INT8_C(  49),  INT8_C(  70), -INT8_C(  62),  INT8_C( 125), -INT8_C(  61), -INT8_C( 122),  INT8_C( 124),  INT8_C(   2),
        -INT8_C(  32), -INT8_C( 122), -INT8_C( 100), -INT8_C( 117), -INT8_C( 127),  INT8_C(  97),  INT8_C(  40), -INT8_C(  69),
         INT8_C(  58),  INT8_C(   9), -INT8_C(  86),  INT8_C(  41),  INT8_C(  14),  INT8_C(  58), -INT8_C( 105), -INT8_C(  27),
        -INT8_C(  23),  INT8_C(  69), -INT8_C(  94),  INT8_C(  90),  INT8_C( 115), -INT8_C(  10), -INT8_C(  64), -INT8_C(  92),
         INT8_C(  60), -INT8_C( 125),  INT8_C(  33), -INT8_C(   1),  INT8_C(   9), -INT8_C(  98),  INT8_C(   1), -INT8_C(  23),
         INT8_C(  36), -INT8_C(  98),  INT8_C( 116), -INT8_C(  91), -INT8_C(   1), -INT8_C(  99),  INT8_C(  96),  INT8_C(  57),
        -INT8_C(  90),  INT8_C(  10),  INT8_C(  99), -INT8_C(  76),  INT8_C(  69), -INT8_C(   6), -INT8_C( 103),  INT8_C(  46) },
      UINT64_C( 9295781545615163399) },
    { UINT64_C(  481084291829640640),
      {  INT8_C(  54),  INT8_C(  23), -INT8_C( 121), -INT8_C( 112), -INT8_C(  73),  INT8_C( 109),  INT8_C(   1), -INT8_C(  16),
        -INT8_C(  31), -INT8_C(  77), -INT8_C(  30), -INT8_C(  65),  INT8_C( 124),  INT8_C(  73), -INT8_C(  24),  INT8_C(  76),
         INT8_C( 104), -INT8_C(  94), -INT8_C(  81),  INT8_C(  46),  INT8_C(  27), -INT8_C(  34), -INT8_C( 126), -INT8_C(  32),
         INT8_C(  63),  INT8_C(   0),  INT8_C(  45), -INT8_C(  36), -INT8_C(  88), -INT8_C(  83), -INT8_C(  30), -INT8_C(  25),
        -INT8_C(  71),  INT8_C( 106),  INT8_C( 119),  INT8_C( 125), -INT8_C(  52),  INT8_C(  95),  INT8_C( 120), -INT8_C(  33),
         INT8_C(  62),  INT8_C(  90),  INT8_C(  78), -INT8_C(  87), -INT8_C(  93),  INT8_C(  54),  INT8_C(  77),  INT8_C( 125),
        -INT8_C(  39), -INT8_C(  56), -INT8_C(  84), -INT8_C(  12), -INT8_C(  36), -INT8_C( 113), -INT8_C(  48), -INT8_C(  67),
         INT8_C(  55), -INT8_C(   3), -INT8_C(  57),  INT8_C(  57), -INT8_C(  86),  INT8_C(  63),  INT8_C(  32),  INT8_C( 111) },
      { -INT8_C(  63), -INT8_C(  24),  INT8_C(  25),  INT8_C( 111),  INT8_C(  72), -INT8_C( 110), -INT8_C(   2), -INT8_C( 122),
        -INT8_C(  20),  INT8_C(  76), -INT8_C( 124), -INT8_C( 112), -INT8_C( 125), -INT8_C(  49),  INT8_C(  13),  INT8_C(  92),
        -INT8_C( 105), -INT8_C(  71),  INT8_C(  80), -INT8_C(  15), -INT8_C(  24),  INT8_C(  33), -INT8_C( 118),  INT8_C(  31),
         INT8_C(  30), -INT8_C(   1), -INT8_C( 123), -INT8_C(  10),  INT8_C(  87),  INT8_C(  82), -INT8_C( 120),  INT8_C(  24),
         INT8_C(  58), -INT8_C(  95), -INT8_C( 120), -INT8_C( 126),  INT8_C(  51), -INT8_C( 122),  INT8_C(   8),  INT8_C(  32),
        -INT8_C(  45), -INT8_C( 115), -INT8_C(  80),  INT8_C(  86),  INT8_C(  92), -INT8_C(  67), -INT8_C(  78), -INT8_C(  12),
         INT8_C( 119),  INT8_C(   2), -INT8_C(  27),  INT8_C(  95),  INT8_C(  35),  INT8_C( 112),  INT8_C( 126),  INT8_C(  66),
         INT8_C( 111),  INT8_C(   3),  INT8_C(  56), -INT8_C(  58),  INT8_C(  85), -INT8_C(  64), -INT8_C(  33), -INT8_C( 112) },
      UINT64_C(  147812851065728384) },
    { UINT64_C(10541233849462194310),
      {  INT8_C(  12), -INT8_C( 125), -INT8_C(  15), -INT8_C( 107), -INT8_C(  15), -INT8_C(  22),  INT8_C(   4),  INT8_C(  58),
         INT8_C( 120),  INT8_C(  20),  INT8_C(  36),  INT8_C(  66), -INT8_C(  25),  INT8_C(  33),  INT8_C(  43),  INT8_C(  72),
        -INT8_C(  32), -INT8_C(  36), -INT8_C(  45),  INT8_C(  49), -INT8_C( 121), -INT8_C(  75), -INT8_C(  80),  INT8_C( 118),
         INT8_C(  39),  INT8_C(  53), -INT8_C( 119),  INT8_C(   8),  INT8_C(  49), -INT8_C(  46),  INT8_C(  17),  INT8_C(  73),
         INT8_C(  85),  INT8_C(  80), -INT8_C(  98),  INT8_C(  78), -INT8_C(  66), -INT8_C(  94), -INT8_C( 119),  INT8_C(  83),
        -INT8_C(  53), -INT8_C(  73), -INT8_C( 103),  INT8_C(  12),  INT8_C(  54), -INT8_C(  91), -INT8_C(   4),  INT8_C(  28),
        -INT8_C(  41), -INT8_C(  33), -INT8_C(   6), -INT8_C( 114),  INT8_C( 108), -INT8_C(  67),  INT8_C(   4), -INT8_C( 108),
         INT8_C(  96), -INT8_C( 114),  INT8_C(  92), -INT8_C(  99),  INT8_C(  96),  INT8_C( 102), -INT8_C(  51), -INT8_C(  74) },
      { -INT8_C(  13),  INT8_C( 108), -INT8_C(  63),  INT8_C( 106),  INT8_C(  14), -INT8_C( 110), -INT8_C(  67), -INT8_C(  59),
        -INT8_C( 121),  INT8_C(  86), -INT8_C(  46), -INT8_C(  67),  INT8_C(  24), -INT8_C(  34), -INT8_C(  44), -INT8_C(  73),
        -INT8_C(  67), -INT8_C(  50),  INT8_C(  69), -INT8_C(  50),  INT8_C( 120),  INT8_C(  74), -INT8_C( 111),  INT8_C(  88),
        -INT8_C(  40),  INT8_C(  92),  INT8_C( 105),  INT8_C(  56), -INT8_C(  61),  INT8_C(  54), -INT8_C(  18), -INT8_C(  74),
        -INT8_C(  94), -INT8_C(  81),  INT8_C(  32), -INT8_C(  79),  INT8_C(  65), -INT8_C(  35),  INT8_C( 118), -INT8_C(  55),
         INT8_C(  52),  INT8_C(  72), -INT8_C( 122),  INT8_C(  76),  INT8_C(  38),  INT8_C(  90),  INT8_C(   3), -INT8_C(  29),
         INT8_C(  40),  INT8_C(  73), -INT8_C(  79), -INT8_C(  95), -INT8_C( 109),  INT8_C(  66), -INT8_C(   7),  INT8_C( 107),
        -INT8_C(  97),  INT8_C(  98), -INT8_C(  93),  INT8_C(  98), -INT8_C( 104), -INT8_C( 110),  INT8_C(  24),  INT8_C(  59) },
      UINT64_C( 9369769811380535300) },
    { UINT64_C(12422559118012878291),
      { -INT8_C(  70), -INT8_C(  66), -INT8_C(  24),  INT8_C(  66), -INT8_C(  81), -INT8_C(  31), -INT8_C(  47),  INT8_C(  18),
        -INT8_C(  64),  INT8_C(  64), -INT8_C(  70),  INT8_C(  33),  INT8_C(  79), -INT8_C(  74), -INT8_C(  44),  INT8_C(   5),
         INT8_C(  55),  INT8_C(  60), -INT8_C(  17), -INT8_C(  59),  INT8_C( 123), -INT8_C(  22),  INT8_C(  40),  INT8_C(  31),
         INT8_C(  46), -INT8_C( 102),  INT8_C(  77), -INT8_C(  30),  INT8_C(  23),      INT8_MAX, -INT8_C( 114),  INT8_C(   7),
         INT8_C(   7),  INT8_C(  97),  INT8_C(  97), -INT8_C(  18),  INT8_C(  49),  INT8_C(  25),  INT8_C(   6),  INT8_C( 111),
        -INT8_C(  99),  INT8_C(  40), -INT8_C( 112), -INT8_C(  20), -INT8_C(  49), -INT8_C(  44),  INT8_C(  59), -INT8_C(  44),
        -INT8_C(  42),  INT8_C(  55),  INT8_C(  47), -INT8_C( 116),  INT8_C(  16),  INT8_C(  23),  INT8_C(  16), -INT8_C(  81),
         INT8_C(  91),  INT8_C(  93),  INT8_C(  45), -INT8_C(   9), -INT8_C( 116), -INT8_C( 107), -INT8_C(  33), -INT8_C(  53) },
      {  INT8_C(  16),  INT8_C(  65),  INT8_C(  23), -INT8_C(  65),  INT8_C(  80),  INT8_C(  30),  INT8_C(  46), -INT8_C(  19),
        -INT8_C(  34), -INT8_C(  65), -INT8_C(  39),  INT8_C(  85),  INT8_C(  36),  INT8_C(  89),  INT8_C(  41), -INT8_C(   6),
        -INT8_C(  56), -INT8_C(  61),  INT8_C(   4), -INT8_C(  40), -INT8_C( 124),  INT8_C(  21),  INT8_C(  23), -INT8_C(  32),
         INT8_C( 114),  INT8_C(  56), -INT8_C(  96), -INT8_C( 126), -INT8_C(  24),      INT8_MIN, -INT8_C( 104), -INT8_C(   8),
        -INT8_C(  63), -INT8_C(  80), -INT8_C(  72),  INT8_C(  17), -INT8_C(  50), -INT8_C(  26), -INT8_C(   2), -INT8_C(  84),
        -INT8_C(  91), -INT8_C(  41),  INT8_C(   2), -INT8_C(  55),  INT8_C(  48),  INT8_C(  43), -INT8_C(  60), -INT8_C(   8),
        -INT8_C(  18), -INT8_C(  56), -INT8_C(  48),  INT8_C( 115), -INT8_C(  35), -INT8_C(  24),  INT8_C(  83),  INT8_C(  80),
         INT8_C(  32), -INT8_C(  13), -INT8_C(  46),  INT8_C(   8),  INT8_C( 115),  INT8_C( 106),  INT8_C(   1),  INT8_C(  52) },
      UINT64_C(   18447070327213313) },
    { UINT64_C( 1677542360563664629),
      { -INT8_C(  92),  INT8_C(  95),  INT8_C(  14),  INT8_C(  39), -INT8_C( 118),  INT8_C( 104),  INT8_C( 109), -INT8_C(  32),
        -INT8_C( 127), -INT8_C(  17),  INT8_C(  49),  INT8_C(  53), -INT8_C(  67), -INT8_C(  98),  INT8_C( 112),  INT8_C(  22),
        -INT8_C( 125), -INT8_C(   5), -INT8_C(  10), -INT8_C(  28), -INT8_C(  79), -INT8_C( 126),  INT8_C( 119), -INT8_C( 114),
         INT8_C(  81), -INT8_C(  50),  INT8_C(  31),  INT8_C(  94),  INT8_C(  92),  INT8_C(   7),  INT8_C( 118),  INT8_C(  21),
         INT8_C( 103), -INT8_C( 124),  INT8_C(  72), -INT8_C(   8),  INT8_C(  95), -INT8_C(  86),  INT8_C( 125), -INT8_C(  70),
         INT8_C(  12),  INT8_C(  74), -INT8_C(  77), -INT8_C(  55), -INT8_C(  24), -INT8_C(  18), -INT8_C(  78), -INT8_C(  42),
        -INT8_C(  22),  INT8_C(  70), -INT8_C(   5), -INT8_C( 125), -INT8_C(  56),  INT8_C( 114),  INT8_C(  17),  INT8_C(  70),
        -INT8_C(  47),  INT8_C(  49), -INT8_C(  92),  INT8_C(  87), -INT8_C( 105),  INT8_C(  26), -INT8_C( 102),  INT8_C(  46) },
      { -INT8_C(  98), -INT8_C(  96),  INT8_C( 126), -INT8_C(  40),  INT8_C( 111), -INT8_C( 105), -INT8_C( 110),  INT8_C( 123),
        -INT8_C(  31),  INT8_C(  16),  INT8_C(  69), -INT8_C(  54), -INT8_C(   1), -INT8_C(  27), -INT8_C(  31), -INT8_C(  23),
         INT8_C( 124), -INT8_C(  36),  INT8_C( 108),  INT8_C(  71),  INT8_C(  78),  INT8_C( 125), -INT8_C( 115),  INT8_C(  73),
        -INT8_C(  82),  INT8_C(  49), -INT8_C(  96),  INT8_C(  70),  INT8_C(  76), -INT8_C(   8), -INT8_C( 117), -INT8_C(  22),
        -INT8_C( 104),  INT8_C(   9), -INT8_C(  62),  INT8_C(   7), -INT8_C(  96),  INT8_C(  85), -INT8_C( 126), -INT8_C( 126),
         INT8_C( 101), -INT8_C(  57),  INT8_C(  76),  INT8_C( 100), -INT8_C(  83),  INT8_C(  45),  INT8_C(  77),  INT8_C(  41),
         INT8_C(   9), -INT8_C(  71),  INT8_C( 112),  INT8_C(  87),  INT8_C(  55), -INT8_C(   3), -INT8_C(  95), -INT8_C(  27),
         INT8_C(  46),  INT8_C(  65),  INT8_C(  43),  INT8_C( 122),  INT8_C(  57), -INT8_C(  73),  INT8_C( 101), -INT8_C(  47) },
      UINT64_C( 1604710282972644501) },
    { UINT64_C( 4529592901992082943),
      { -INT8_C( 104),  INT8_C(  19),  INT8_C(  84),  INT8_C( 110),  INT8_C( 116),  INT8_C(  91), -INT8_C( 115),  INT8_C(  48),
         INT8_C(  40),  INT8_C(  51),  INT8_C(  58),  INT8_C(  16),  INT8_C(  67),  INT8_C( 100),  INT8_C(  91),  INT8_C(  55),
        -INT8_C( 106),  INT8_C(  93),  INT8_C(  73), -INT8_C(  87), -INT8_C(  49),  INT8_C(  95),  INT8_C( 117),  INT8_C(  82),
         INT8_C(  71),  INT8_C(  60), -INT8_C( 113),  INT8_C(   5),  INT8_C(  70), -INT8_C(  18),  INT8_C(  81), -INT8_C(  25),
         INT8_C(  96),  INT8_C(  93),  INT8_C(  74),  INT8_C(  62), -INT8_C(  72), -INT8_C(  40),  INT8_C( 121),  INT8_C(  16),
         INT8_C(  11), -INT8_C(  77), -INT8_C(  69),  INT8_C( 120), -INT8_C(   7),  INT8_C( 113),  INT8_C(  56), -INT8_C(  75),
         INT8_C( 116), -INT8_C(   7),  INT8_C(  57),  INT8_C(  68),  INT8_C(  89), -INT8_C(  81), -INT8_C(  23), -INT8_C(   9),
         INT8_C( 124),  INT8_C(  39),  INT8_C(  32), -INT8_C(  61),  INT8_C( 118),  INT8_C( 124), -INT8_C( 109),  INT8_C(  24) },
      { -INT8_C(  39), -INT8_C(  20), -INT8_C(  85), -INT8_C( 111), -INT8_C(  60),  INT8_C(  36), -INT8_C(  95), -INT8_C(  49),
        -INT8_C(  41),  INT8_C( 101),  INT8_C(  30), -INT8_C(  17), -INT8_C(  41),  INT8_C(  86), -INT8_C(  92), -INT8_C(  56),
         INT8_C( 105), -INT8_C(  94), -INT8_C(  74),  INT8_C(  86),  INT8_C(  48), -INT8_C(  96), -INT8_C( 118), -INT8_C(  83),
         INT8_C(  97), -INT8_C(  61),  INT8_C( 112), -INT8_C(  41),  INT8_C(  63),  INT8_C(  17), -INT8_C(  82),  INT8_C(  24),
        -INT8_C(   3),  INT8_C(  89), -INT8_C(  87), -INT8_C(  63),  INT8_C( 125),  INT8_C(  75), -INT8_C( 111),  INT8_C(  84),
        -INT8_C(  80), -INT8_C(  81),  INT8_C(  68), -INT8_C( 121),  INT8_C(   6), -INT8_C(  24),  INT8_C(  80),  INT8_C( 111),
        -INT8_C( 117),  INT8_C(   6), -INT8_C(  58), -INT8_C(  69), -INT8_C(  90),  INT8_C(  80),  INT8_C( 104),  INT8_C(   8),
         INT8_C(  20), -INT8_C(  40), -INT8_C(  33),  INT8_C(  83), -INT8_C(  22), -INT8_C( 115),  INT8_C( 108), -INT8_C(  25) },
      UINT64_C( 4053310815464334417) },
    { UINT64_C( 6137157573880594433),
      { -INT8_C(   8),  INT8_C(  37),  INT8_C(  66),  INT8_C(  73),  INT8_C( 125), -INT8_C(  98), -INT8_C( 119),  INT8_C( 124),
         INT8_C(  38),  INT8_C(  95), -INT8_C(  18),  INT8_C(  65), -INT8_C(  46),  INT8_C(  82), -INT8_C(  56), -INT8_C(  62),
         INT8_C(  32), -INT8_C(  91), -INT8_C(  68),  INT8_C(  69), -INT8_C(  99),  INT8_C(  86), -INT8_C( 125), -INT8_C(  32),
        -INT8_C(  74), -INT8_C( 116),  INT8_C( 107),  INT8_C(   8),  INT8_C(  11),  INT8_C(  71),  INT8_C(  40), -INT8_C(  65),
         INT8_C(  41), -INT8_C( 104), -INT8_C(   7), -INT8_C( 120),  INT8_C( 106), -INT8_C( 125), -INT8_C(  85), -INT8_C( 112),
        -INT8_C(  30), -INT8_C( 103), -INT8_C(  47), -INT8_C(   4),  INT8_C(  33),  INT8_C(  19),  INT8_C(  14), -INT8_C(  57),
         INT8_C(  70),  INT8_C(  35),  INT8_C(  46), -INT8_C(   3), -INT8_C(  38),  INT8_C(  11), -INT8_C(  60),  INT8_C(  48),
        -INT8_C(  67), -INT8_C(  96), -INT8_C(   9), -INT8_C(  93),  INT8_C( 119),      INT8_MIN,  INT8_C(  99), -INT8_C(  32) },
      {  INT8_C(  76),  INT8_C(  30), -INT8_C( 112), -INT8_C(  74), -INT8_C(  95),  INT8_C(  59),  INT8_C(  70), -INT8_C( 125),
        -INT8_C(  44),  INT8_C(  23),  INT8_C(  55), -INT8_C(  41), -INT8_C(  80), -INT8_C(  83), -INT8_C(  97), -INT8_C(  18),
        -INT8_C(  33), -INT8_C(  51), -INT8_C(  21), -INT8_C(  70),  INT8_C(  98), -INT8_C(  87),  INT8_C( 124),  INT8_C(  31),
         INT8_C(  73),  INT8_C( 115), -INT8_C( 108), -INT8_C(   9), -INT8_C(  12), -INT8_C(  72), -INT8_C(  41),  INT8_C(  64),
        -INT8_C(  42),  INT8_C( 103), -INT8_C(  10),  INT8_C( 119), -INT8_C(  94),  INT8_C(  60), -INT8_C(   6),  INT8_C( 118),
         INT8_C(  83),  INT8_C(  49),  INT8_C(  77),  INT8_C(   3), -INT8_C(  34), -INT8_C(  20), -INT8_C(  15), -INT8_C(  66),
        -INT8_C(  71), -INT8_C(  36),  INT8_C( 120),  INT8_C(  28), -INT8_C( 122), -INT8_C(  12),  INT8_C(  59), -INT8_C(  49),
         INT8_C( 104), -INT8_C(  49), -INT8_C(  58),  INT8_C(  92), -INT8_C( 120), -INT8_C(  99), -INT8_C( 100),  INT8_C(  94) },
      UINT64_C(  362680799549472769) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k1 = test_vec[i].k1;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_test_epi8_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_test_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int8_t a_[64];
    int8_t b_[64];
    easysimd__mmask64 k1 = easysimd_test_x86_random_mmask64();
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 64 ; j++) if (rand() & 1) a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi8(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(b_);
    easysimd__mmask64 r = easysimd_mm512_mask_test_epi8_mask(k1, a, b);

    easysimd_test_x86_write_mmask64(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_test_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { { -INT16_C( 10245),  INT16_C( 18801),  INT16_C( 31642), -INT16_C( 30834),  INT16_C(  4021), -INT16_C( 24214),  INT16_C(  3143),  INT16_C( 21759),
         INT16_C(  5919),  INT16_C( 28063), -INT16_C( 18558),  INT16_C( 14004), -INT16_C( 15245), -INT16_C(   788), -INT16_C( 30313), -INT16_C(  3515),
         INT16_C( 13971),  INT16_C( 19259), -INT16_C(  1420), -INT16_C( 29772), -INT16_C( 24822),  INT16_C( 27851), -INT16_C( 21228),  INT16_C( 13267),
         INT16_C(  6664), -INT16_C( 29948),  INT16_C( 19325),  INT16_C( 17857),  INT16_C( 17484), -INT16_C(  6928),  INT16_C( 12391), -INT16_C( 27800) },
      {  INT16_C( 10244), -INT16_C( 18802),  INT16_C( 17119),  INT16_C(  5953), -INT16_C(  4022),  INT16_C( 24213),  INT16_C( 26781),  INT16_C( 25233),
         INT16_C( 13018),  INT16_C( 22512),  INT16_C( 18557), -INT16_C( 14005),  INT16_C( 15244),  INT16_C( 23213), -INT16_C( 21495),  INT16_C(  3514),
         INT16_C( 18644), -INT16_C( 19260),  INT16_C(  1419), -INT16_C( 10805),  INT16_C( 24821), -INT16_C( 27852), -INT16_C( 14903), -INT16_C( 23563),
        -INT16_C(  6665),  INT16_C( 29947),  INT16_C( 17966), -INT16_C( 17858), -INT16_C(  5247), -INT16_C( 30188), -INT16_C( 12392),  INT16_C( 27799) },
      UINT32_C( 885613516) },
    { {  INT16_C(  4355),  INT16_C( 22602), -INT16_C( 15849),  INT16_C( 25519), -INT16_C( 26514), -INT16_C(  8328),  INT16_C( 16186),  INT16_C( 15176),
         INT16_C(  9956), -INT16_C(  8659), -INT16_C(  9919),  INT16_C( 29603), -INT16_C( 27243), -INT16_C( 26945),  INT16_C(  6837), -INT16_C(  3886),
         INT16_C(  9516),  INT16_C( 18143), -INT16_C( 18121),  INT16_C( 19610),  INT16_C( 20839), -INT16_C( 23822), -INT16_C( 31087),  INT16_C( 24898),
         INT16_C(  1812), -INT16_C(  9466),  INT16_C( 24137), -INT16_C(  7388),  INT16_C( 30497), -INT16_C( 19442),  INT16_C( 20114),  INT16_C(  7589) },
      { -INT16_C(  4356), -INT16_C(  9644), -INT16_C(  4378),  INT16_C( 30246),  INT16_C( 20736),  INT16_C( 17534), -INT16_C( 16187),  INT16_C( 24428),
         INT16_C( 29232),  INT16_C( 31035),  INT16_C( 24528), -INT16_C( 29604),  INT16_C( 27242),  INT16_C( 26944), -INT16_C(  6838),  INT16_C( 18054),
        -INT16_C(  9517), -INT16_C( 18144),  INT16_C( 18120), -INT16_C( 14032), -INT16_C( 20840),  INT16_C( 23821),  INT16_C( 31086), -INT16_C( 24899),
        -INT16_C(  1813), -INT16_C( 17641),  INT16_C( 29783), -INT16_C( 16057), -INT16_C( 30498),  INT16_C( 10538), -INT16_C( 20115),  INT16_C( 16751) },
      UINT32_C(2919794622) },
    { { -INT16_C( 15752), -INT16_C(  2825),  INT16_C( 30044),  INT16_C( 16469),  INT16_C( 30696),  INT16_C( 26898), -INT16_C(  3822),  INT16_C(  3323),
         INT16_C(  5870), -INT16_C(  2908),  INT16_C(  1191),  INT16_C(  4091),  INT16_C( 10184),  INT16_C( 15686),  INT16_C(  8528), -INT16_C( 30914),
        -INT16_C( 27952), -INT16_C( 17194), -INT16_C( 27468), -INT16_C( 24703), -INT16_C( 16373),  INT16_C( 11998), -INT16_C( 21889),  INT16_C( 14309),
        -INT16_C( 30159), -INT16_C(  7124),  INT16_C( 10247), -INT16_C(  1753),  INT16_C( 20718), -INT16_C( 19658),  INT16_C( 29820), -INT16_C( 31607) },
      {  INT16_C(  4103),  INT16_C(  2824), -INT16_C( 30045), -INT16_C( 16470), -INT16_C( 30697), -INT16_C( 26899), -INT16_C( 20941), -INT16_C(  3324),
        -INT16_C( 11814),  INT16_C(  2907), -INT16_C( 32179), -INT16_C(  4092),  INT16_C( 15088),  INT16_C( 27811), -INT16_C(  8529), -INT16_C( 18760),
        -INT16_C( 15890), -INT16_C( 27967),  INT16_C( 27467),  INT16_C( 25169),  INT16_C( 16372),  INT16_C( 10233), -INT16_C(   531), -INT16_C( 14310),
         INT16_C( 30158),  INT16_C(  7123), -INT16_C( 10248), -INT16_C(  6133), -INT16_C( 20719), -INT16_C( 16299),  INT16_C(  3469),  INT16_C( 31606) },
      UINT32_C(1751889216) },
    { {  INT16_C(  2253), -INT16_C(  9518), -INT16_C(  2678),  INT16_C( 18558),  INT16_C( 17914), -INT16_C(  9036), -INT16_C(  9487),  INT16_C(  9521),
         INT16_C( 30510),  INT16_C( 11234), -INT16_C(  8024), -INT16_C(  3270),  INT16_C(  2708),  INT16_C(  9621),  INT16_C( 30858),  INT16_C( 22534),
        -INT16_C( 10111), -INT16_C(  6861), -INT16_C( 20199),  INT16_C( 23698),  INT16_C(  3575), -INT16_C(  5696), -INT16_C(  3353), -INT16_C(  9457),
        -INT16_C(  6197),  INT16_C(  9687), -INT16_C(  7689),  INT16_C(  7395),  INT16_C( 25863), -INT16_C( 27929), -INT16_C(  4642), -INT16_C( 15351) },
      { -INT16_C(  2254),  INT16_C(  9517), -INT16_C( 16484), -INT16_C( 18559), -INT16_C( 17915),  INT16_C( 21280),  INT16_C(  9486), -INT16_C(  9522),
         INT16_C( 29964),  INT16_C(  1155),  INT16_C(  8023),  INT16_C( 17039),  INT16_C( 16464),  INT16_C( 13694), -INT16_C( 30859), -INT16_C( 22535),
         INT16_C( 10110),  INT16_C(  6860),  INT16_C( 20198), -INT16_C(  4910), -INT16_C(  3576),  INT16_C(  5695),  INT16_C(  3352),  INT16_C(  9456),
         INT16_C( 29571), -INT16_C(  9688), -INT16_C( 18286), -INT16_C(  7396), -INT16_C( 25864),  INT16_C( 27928),  INT16_C(  4641), -INT16_C( 24812) },
      UINT32_C(2231909156) },
    { {  INT16_C(  9041),  INT16_C(  9642), -INT16_C( 19085), -INT16_C( 19117),  INT16_C( 31931), -INT16_C( 15038),  INT16_C( 17383), -INT16_C( 32600),
        -INT16_C(  6106),  INT16_C( 24788),  INT16_C(  1209),  INT16_C(  2608),  INT16_C( 27263),  INT16_C(  6932), -INT16_C( 16763),  INT16_C( 24863),
        -INT16_C( 29727),  INT16_C(  9129), -INT16_C(  7553), -INT16_C( 18471), -INT16_C( 19573),  INT16_C( 29655),  INT16_C(  9555),  INT16_C(  7436),
        -INT16_C(  5646), -INT16_C( 31998),  INT16_C( 13549),  INT16_C( 12497), -INT16_C( 31606), -INT16_C(  2619), -INT16_C(  6812),  INT16_C( 17403) },
      { -INT16_C(  9042),  INT16_C( 11621), -INT16_C( 28482),  INT16_C( 19116), -INT16_C( 31932),  INT16_C( 15037), -INT16_C( 13939),  INT16_C( 32599),
         INT16_C( 14002), -INT16_C( 24789), -INT16_C(   917), -INT16_C(  2609), -INT16_C( 27264), -INT16_C(  6933),  INT16_C( 16762),  INT16_C( 10255),
         INT16_C( 29726), -INT16_C(  9130),  INT16_C(   516),  INT16_C( 18470), -INT16_C(  7290),  INT16_C(  4995), -INT16_C(  9556),  INT16_C( 24211),
        -INT16_C( 16879),  INT16_C( 31997), -INT16_C( 12870),  INT16_C( 14961),  INT16_C( 23650), -INT16_C(  9186),  INT16_C( 11678), -INT16_C( 17404) },
      UINT32_C(2108982598) },
    { { -INT16_C( 12657),  INT16_C( 29389), -INT16_C( 32627), -INT16_C(  1451),  INT16_C(  1129),  INT16_C( 23456),  INT16_C(  9124),  INT16_C(  7380),
        -INT16_C(  7801), -INT16_C( 28013),  INT16_C( 23030), -INT16_C( 17430), -INT16_C( 21837),  INT16_C( 14817), -INT16_C( 22330),  INT16_C( 22062),
         INT16_C( 31863), -INT16_C( 11285),  INT16_C( 16617),  INT16_C(  5581),  INT16_C(  6028),  INT16_C( 24972),  INT16_C( 28107), -INT16_C( 20355),
         INT16_C( 25422),  INT16_C( 29414), -INT16_C( 16651), -INT16_C( 14291),  INT16_C(  7387), -INT16_C( 23851),  INT16_C( 16996), -INT16_C(  8378) },
      {  INT16_C( 12656),  INT16_C( 22962),  INT16_C( 32626), -INT16_C(   402), -INT16_C(  1130), -INT16_C( 23457), -INT16_C(  9125),  INT16_C( 26708),
         INT16_C(  7800),  INT16_C( 28012),  INT16_C(  3292),  INT16_C( 17429), -INT16_C(  2418), -INT16_C(  3567),  INT16_C( 22329), -INT16_C( 22063),
        -INT16_C( 31864), -INT16_C(  1533),  INT16_C( 28931), -INT16_C( 26120),  INT16_C( 22380), -INT16_C( 14274), -INT16_C( 28108), -INT16_C( 21456),
        -INT16_C( 25423), -INT16_C( 29415),  INT16_C( 11945),  INT16_C( 14290), -INT16_C(  7388),  INT16_C( 23850), -INT16_C(  1222), -INT16_C( 15865) },
      UINT32_C(3300799626) },
    { { -INT16_C( 26169),  INT16_C( 17028),  INT16_C( 30871), -INT16_C(  1688), -INT16_C( 24805),  INT16_C( 13473),  INT16_C(  4179),  INT16_C( 23485),
        -INT16_C( 21361), -INT16_C( 15522), -INT16_C( 17495),  INT16_C(  1419), -INT16_C( 10043), -INT16_C( 14686),  INT16_C( 32399),  INT16_C( 22216),
         INT16_C( 19479),  INT16_C( 13283),  INT16_C( 19662), -INT16_C(  5587),  INT16_C( 23965),  INT16_C( 22729),  INT16_C( 19431),  INT16_C( 30352),
        -INT16_C(  4360), -INT16_C( 29061),  INT16_C(  1768), -INT16_C(  4823), -INT16_C(  7201), -INT16_C( 18509),  INT16_C( 18213),  INT16_C( 24589) },
      { -INT16_C( 23097), -INT16_C( 19770), -INT16_C( 30872),  INT16_C(  1687),  INT16_C( 24804), -INT16_C( 13474), -INT16_C(  4180), -INT16_C( 23486),
         INT16_C( 31965), -INT16_C( 30907),  INT16_C( 28370), -INT16_C(  1420),  INT16_C( 10042), -INT16_C( 31823), -INT16_C( 16734),  INT16_C( 27108),
        -INT16_C( 21917), -INT16_C( 13284), -INT16_C( 19663),  INT16_C(  5586),  INT16_C( 12307), -INT16_C( 16415),  INT16_C(  8991), -INT16_C(   669),
        -INT16_C( 22369),  INT16_C( 29060), -INT16_C(  1769),  INT16_C( 20843),  INT16_C(  7200), -INT16_C( 15404), -INT16_C( 18214),  INT16_C( 15660) },
      UINT32_C(2851202819) },
    { {  INT16_C( 16382), -INT16_C( 26259),  INT16_C( 28224),  INT16_C( 19465), -INT16_C( 10248), -INT16_C( 23176), -INT16_C( 10938), -INT16_C(  6927),
        -INT16_C( 20915), -INT16_C( 28118), -INT16_C( 26651),  INT16_C(  6842),  INT16_C( 17199),  INT16_C( 27442), -INT16_C( 27399), -INT16_C( 12782),
         INT16_C( 32675),  INT16_C( 14834), -INT16_C( 23193),  INT16_C( 23942), -INT16_C(   302),  INT16_C(  9883), -INT16_C(  2914),  INT16_C(  1738),
        -INT16_C(  2694),  INT16_C( 31199),  INT16_C( 20364),  INT16_C( 22633), -INT16_C( 25813), -INT16_C(  1085),  INT16_C(  2912), -INT16_C( 23607) },
      { -INT16_C( 16383),  INT16_C( 26258),  INT16_C(  6331), -INT16_C( 28989), -INT16_C( 15082), -INT16_C( 19094),  INT16_C( 10937),  INT16_C( 23713),
        -INT16_C( 32577),  INT16_C( 28117),  INT16_C( 26650), -INT16_C(  2231),  INT16_C(  9268), -INT16_C( 27443), -INT16_C( 25552),  INT16_C( 12781),
        -INT16_C( 32676),  INT16_C(  6295),  INT16_C( 23192), -INT16_C( 20570),  INT16_C(  4128), -INT16_C(  9884),  INT16_C(  1338), -INT16_C(  1739),
         INT16_C(  2693), -INT16_C( 24730), -INT16_C( 20365), -INT16_C( 22634),  INT16_C( 25812),  INT16_C(  1084),  INT16_C( 10496),  INT16_C( 23606) },
      UINT32_C(1113217468) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_test_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_test_epi16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[32];
    int16_t b_[32];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 32 ; j++) if (rand() & 1) a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi16(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(b_);
    easysimd__mmask32 r = easysimd_mm512_test_epi16_mask(a, b);

    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_test_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { -INT32_C(  1791876952),  INT32_C(  1844499212), -INT32_C(   365199811), -INT32_C(  1866534401), -INT32_C(   232364997),  INT32_C(  1453189598), -INT32_C(   461789425),  INT32_C(   971959440),
        -INT32_C(   355524340), -INT32_C(  2146953968),  INT32_C(   864052768),  INT32_C(   199938932),  INT32_C(    54748553),  INT32_C(  1624897104),  INT32_C(   776295070),  INT32_C(  1281897535) },
      {  INT32_C(  1791876951), -INT32_C(   491122398),  INT32_C(    10679440),  INT32_C(  1236071149), -INT32_C(   930854296),  INT32_C(   709694853),  INT32_C(   461789424), -INT32_C(   971959441),
         INT32_C(   355524339),  INT32_C(  2146953967), -INT32_C(   864052769), -INT32_C(   199938933), -INT32_C(    54748554), -INT32_C(  1624897105), -INT32_C(   776295071), -INT32_C(  1281897536) },
      UINT16_C(   62) },
    { {  INT32_C(   356677238),  INT32_C(     6053022), -INT32_C(   909044215), -INT32_C(   745850423), -INT32_C(   656871898),  INT32_C(  1601452805),  INT32_C(   634323680), -INT32_C(   206023555),
         INT32_C(  1343095986), -INT32_C(  1117669651),  INT32_C(  1664295297), -INT32_C(  1158762358), -INT32_C(    73592075),  INT32_C(   542773838),  INT32_C(   699834084),  INT32_C(  1123018103) },
      { -INT32_C(   356677239),  INT32_C(  1968377588), -INT32_C(  1176992209),  INT32_C(   745850422),  INT32_C(   656871897),  INT32_C(   122126627), -INT32_C(   634323681),  INT32_C(   206023554),
         INT32_C(   703988789),  INT32_C(  1117669650),  INT32_C(   117143248),  INT32_C(   355626812), -INT32_C(  1539548799), -INT32_C(   106134310),  INT32_C(    80993409), -INT32_C(  1123018104) },
      UINT16_C(32038) },
    { { -INT32_C(  1256567730), -INT32_C(  1903775732), -INT32_C(  1021880878), -INT32_C(  2029233078),  INT32_C(   646471133), -INT32_C(  1782033256), -INT32_C(  2056503897),  INT32_C(  1206237333),
         INT32_C(  1562435178), -INT32_C(   258166754),  INT32_C(   658140595), -INT32_C(  1212544241), -INT32_C(    22118461),  INT32_C(  1800265398), -INT32_C(   506967100), -INT32_C(   733180567) },
      {  INT32_C(  1256567729), -INT32_C(  1049575411),  INT32_C(  1021880877),  INT32_C(  2029233077), -INT32_C(   646471134),  INT32_C(  1782033255), -INT32_C(  1560143285), -INT32_C(  1808014877),
        -INT32_C(  1562435179),  INT32_C(   258166753), -INT32_C(  1370796807),  INT32_C(   287784943),  INT32_C(  1223335649), -INT32_C(  1800265399),  INT32_C(   506967099),  INT32_C(   733180566) },
      UINT16_C( 7362) },
    { { -INT32_C(  1774688435),  INT32_C(  1689956846),  INT32_C(  1778816600),  INT32_C(  1734383060), -INT32_C(  1672529762), -INT32_C(   798399687),  INT32_C(  1044867632),  INT32_C(  1731118106),
         INT32_C(  1744780664), -INT32_C(   277403568),  INT32_C(  1655951952), -INT32_C(  1295906541),  INT32_C(  2085491266),  INT32_C(   854392578),  INT32_C(   199072177),  INT32_C(   602644472) },
      {  INT32_C(  1440213253), -INT32_C(  1689956847), -INT32_C(  1778816601), -INT32_C(  1599422482),  INT32_C(  1672529761),  INT32_C(   798399686), -INT32_C(   927791152), -INT32_C(  1410621786),
        -INT32_C(  1744780665), -INT32_C(  1221311216), -INT32_C(  1655951953),  INT32_C(  1295906540), -INT32_C(  2085491267),  INT32_C(  1521698698), -INT32_C(   199072178), -INT32_C(  1952510716) },
      UINT16_C(41673) },
    { {  INT32_C(     4120059),  INT32_C(    88068023),  INT32_C(   972644661),  INT32_C(  1069848687), -INT32_C(  1495964580), -INT32_C(   995051713),  INT32_C(  2000019868), -INT32_C(  1425017777),
        -INT32_C(  1985099141), -INT32_C(   312115224),  INT32_C(   608280808), -INT32_C(   371249999), -INT32_C(  1090510320),  INT32_C(  1377285814), -INT32_C(   406238057), -INT32_C(  1517102783) },
      { -INT32_C(     4120060),  INT32_C(   607052274), -INT32_C(  1396402181), -INT32_C(   678058813),  INT32_C(  1495964579), -INT32_C(   391364271), -INT32_C(   808487538),  INT32_C(  1400136014),
         INT32_C(  1985099140),  INT32_C(   312115223), -INT32_C(   608280809), -INT32_C(  1498262525),  INT32_C(  1090510319), -INT32_C(  1406620898),  INT32_C(  1870395425), -INT32_C(   574427047) },
      UINT16_C(59630) },
    { {  INT32_C(  1995142522), -INT32_C(  1025328991),  INT32_C(  1237123720),  INT32_C(  1944581197), -INT32_C(  1632683255),  INT32_C(  2001403434),  INT32_C(  1825100299),  INT32_C(   599571980),
         INT32_C(  1448627380),  INT32_C(   503137172), -INT32_C(   440448360),  INT32_C(  2115725344),  INT32_C(  1859488131),  INT32_C(   293810182),  INT32_C(  1990734309), -INT32_C(   125731111) },
      { -INT32_C(  1995142523),  INT32_C(  1547578820), -INT32_C(  1237123721), -INT32_C(  2030134620), -INT32_C(  1466768222),  INT32_C(   398005525),  INT32_C(  1797666401), -INT32_C(   198617233),
        -INT32_C(  1501675807), -INT32_C(   503137173),  INT32_C(  1587102906), -INT32_C(  2115725345), -INT32_C(  1859488132), -INT32_C(   324476021), -INT32_C(  1990734310),  INT32_C(   125731110) },
      UINT16_C( 9722) },
    { { -INT32_C(  1236631766), -INT32_C(   726892159), -INT32_C(   418972991),  INT32_C(   988710011),  INT32_C(  1318750596), -INT32_C(  1543400492), -INT32_C(   213622153), -INT32_C(  1249473866),
         INT32_C(  1242584808),  INT32_C(   164082504),  INT32_C(   686869933), -INT32_C(  1827472114), -INT32_C(   955660677), -INT32_C(  1191058623),  INT32_C(  1621039920),  INT32_C(  1824172932) },
      {  INT32_C(  1253559041),  INT32_C(   726892158), -INT32_C(  1672264629), -INT32_C(  1636977373), -INT32_C(  1318750597),  INT32_C(  1543400491),  INT32_C(   213622152),  INT32_C(  1249473865),
        -INT32_C(  1097584576), -INT32_C(   102111058),  INT32_C(  1335180332),  INT32_C(  1827472113), -INT32_C(  1558293896),  INT32_C(  1191058622),  INT32_C(    39041465),  INT32_C(  1867369519) },
      UINT16_C(55053) },
    { {  INT32_C(  2003202614), -INT32_C(   475412712),  INT32_C(  1141350272),  INT32_C(  1446086632), -INT32_C(  1605510442),  INT32_C(  1184231666),  INT32_C(  1653332520), -INT32_C(  1518195858),
        -INT32_C(   103716083), -INT32_C(   845194817), -INT32_C(   603306187),  INT32_C(   310425404),  INT32_C(  1591703846), -INT32_C(   423016258),  INT32_C(   759780642), -INT32_C(  1598267758) },
      { -INT32_C(  2003202615),  INT32_C(   475412711), -INT32_C(  1141350273), -INT32_C(  1446086633),  INT32_C(    70025286), -INT32_C(   353700152), -INT32_C(   585616565),  INT32_C(  1518195857),
         INT32_C(  1239619426), -INT32_C(   815449777), -INT32_C(  1316332135),  INT32_C(   475683030), -INT32_C(  1591703847), -INT32_C(   947123588), -INT32_C(   811228098), -INT32_C(   634772872) },
      UINT16_C(61296) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_test_epi32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_test_epi32_mask");
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

    for (size_t j = 0 ; j < 16 ; j++) if (rand() & 1) a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi32(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(b_);
    easysimd__mmask16 r = easysimd_mm512_test_epi32_mask(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_test_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT64_C( 1935719795857156052), -INT64_C( 6142336982440005382), -INT64_C( 3755735260302931735),  INT64_C( 8565924432478488344),
        -INT64_C( 7625521580720828296),  INT64_C( 3735482225277759375),  INT64_C( 1389743071009894742),  INT64_C( 4211258168981445345) },
      { -INT64_C(  118278527101448595),  INT64_C(  530621773771449840),  INT64_C( 3755735260302931734), -INT64_C( 8565924432478488345),
         INT64_C( 7625521580720828295), -INT64_C( 3735482225277759376), -INT64_C( 2277422861904188195), -INT64_C( 1416974286643090513) },
      UINT8_C(195) },
    { {  INT64_C( 3457588026753764434),  INT64_C( 1717706511224668770),  INT64_C( 5859823844452325122), -INT64_C(  136148487240149447),
        -INT64_C( 4072292484336041004),  INT64_C( 7336384931628364992),  INT64_C(  899403616361728444), -INT64_C( 4966276177273367448) },
      { -INT64_C( 3457588026753764435), -INT64_C( 1717706511224668771),  INT64_C( 7682991252262311636),  INT64_C(  136148487240149446),
         INT64_C( 7005422631704761958), -INT64_C( 7336384931628364993), -INT64_C(  899403616361728445),  INT64_C( 4966276177273367447) },
      UINT8_C( 20) },
    { {  INT64_C( 8111300885188067376),  INT64_C( 8433345625914411573), -INT64_C( 1415332508365564871), -INT64_C( 6383252409235804746),
         INT64_C( 5059076272894766680),  INT64_C( 3611479204207783805), -INT64_C( 2146346760320498508),  INT64_C( 3145857604965282112) },
      { -INT64_C( 5472754878176835303), -INT64_C( 8433345625914411574), -INT64_C( 8007106683827110144),  INT64_C( 6383252409235804745),
        -INT64_C( 5059076272894766681), -INT64_C( 3611479204207783806),  INT64_C(  659405438131680690), -INT64_C( 3145857604965282113) },
      UINT8_C( 69) },
    { { -INT64_C( 3849252375540350663),  INT64_C( 3904726803647939769), -INT64_C(  216416808084101174),  INT64_C( 8434442409886415556),
         INT64_C( 4350871313423223305),  INT64_C( 8504883711772806009), -INT64_C( 6152271255572725614), -INT64_C( 8271005281003021549) },
      {  INT64_C( 3849252375540350662), -INT64_C( 3904726803647939770),  INT64_C(  216416808084101173), -INT64_C( 5671441591776678087),
        -INT64_C( 4389237782810184900), -INT64_C( 8504883711772806010),  INT64_C( 6152271255572725613),  INT64_C( 7011924461457909682) },
      UINT8_C(152) },
    { { -INT64_C( 7428643418601908980),  INT64_C( 2294962712527383143), -INT64_C( 5612589600369147080), -INT64_C( 5974249584702473137),
         INT64_C( 6472844327359209748), -INT64_C( 2618330417723800803), -INT64_C( 4939930458487302337),  INT64_C(  137830299006646676) },
      {  INT64_C( 3349624863759605754),  INT64_C(  467355425918448966),  INT64_C( 5612589600369147079),  INT64_C( 5974249584702473136),
        -INT64_C( 6472844327359209749),  INT64_C(  908254077988168780),  INT64_C( 4939930458487302336), -INT64_C(  137830299006646677) },
      UINT8_C( 35) },
    { {  INT64_C( 3748689352051745012),  INT64_C( 3765597295249760617),  INT64_C( 8461435782336917084), -INT64_C( 7396666994511550337),
        -INT64_C( 6102301345150566269),  INT64_C( 5468756357428140262),  INT64_C( 3501708573596139009), -INT64_C( 4673070998841589811) },
      { -INT64_C( 3748689352051745013),  INT64_C( 2957844122011673298),  INT64_C( 8976476619900634683),  INT64_C( 6553670180077683509),
         INT64_C( 7755375136595222255), -INT64_C( 5333888708520436925), -INT64_C( 3501708573596139010),  INT64_C( 4673070998841589810) },
      UINT8_C( 62) },
    { {  INT64_C( 3340205165359449056),  INT64_C( 8961155326498102258),  INT64_C( 5187257904649764459), -INT64_C( 7718806074294071207),
         INT64_C( 4351002021841475629),  INT64_C( 2783331170132641387), -INT64_C( 6088879537201316956), -INT64_C( 8404271755238698236) },
      { -INT64_C( 3340205165359449057),  INT64_C( 4010264786319470234), -INT64_C( 7508926013520240691),  INT64_C( 1795458474933612802),
        -INT64_C( 8222122719232016645), -INT64_C( 2783331170132641388), -INT64_C( 4653973519871652290), -INT64_C( 8879680603371353420) },
      UINT8_C(222) },
    { { -INT64_C( 7072750869937231124), -INT64_C( 1235259015713123945), -INT64_C( 4453778579372969693), -INT64_C( 6391677988780903072),
        -INT64_C( 5475290417517831931),  INT64_C( 8023331291259204593), -INT64_C( 2934575014531591542), -INT64_C( 7984994343078542901) },
      {  INT64_C( 7072750869937231123),  INT64_C( 1235259015713123944), -INT64_C( 6953033857279573993),  INT64_C( 6391677988780903071),
         INT64_C(  159581583610422626), -INT64_C(   37474756131855391), -INT64_C( 8338930464116371428),  INT64_C( 7984994343078542900) },
      UINT8_C(116) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_test_epi64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_test_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[8];
    int64_t b_[8];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 8 ; j++) if (rand() & 1) a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi64(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(b_);
    easysimd__mmask8 r = easysimd_mm512_test_epi64_mask(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_test_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[64];
    const int8_t b[64];
    const easysimd__mmask64 r;
  } test_vec[] = {
    { { -INT8_C(  41),  INT8_C(  28), -INT8_C(  10), -INT8_C(  79),  INT8_C(  87),  INT8_C(  18), -INT8_C(  97), -INT8_C(   8),
         INT8_C(  52),  INT8_C(  80),  INT8_C(  67),  INT8_C( 113),  INT8_C( 122),  INT8_C(  64),  INT8_C(  57), -INT8_C(  57),
         INT8_C(  36),  INT8_C(  82), -INT8_C(  12), -INT8_C(  67), -INT8_C(  18),  INT8_C(  48), -INT8_C(  17), -INT8_C(  14),
        -INT8_C(  22), -INT8_C( 117),  INT8_C( 115),  INT8_C( 110),  INT8_C(  49),  INT8_C(  37),  INT8_C(  92),  INT8_C(   9),
        -INT8_C(  34), -INT8_C(  43), -INT8_C(   6), -INT8_C(   3),  INT8_C(  54), -INT8_C( 103),  INT8_C(  46), -INT8_C(   9),
        -INT8_C(  18), -INT8_C(  10),  INT8_C( 105), -INT8_C( 103),  INT8_C( 107),  INT8_C( 110),  INT8_C(  97), -INT8_C( 112),
        -INT8_C(  28),  INT8_C(  10),  INT8_C(   2), -INT8_C( 117), -INT8_C(  29), -INT8_C( 121),  INT8_C(  67), -INT8_C(  50),
        -INT8_C(  90), -INT8_C(  80),  INT8_C(  60), -INT8_C(  62), -INT8_C(  35), -INT8_C(  25),  INT8_C(  77), -INT8_C(  58) },
      { -INT8_C(  22),  INT8_C(  71), -INT8_C(  73),  INT8_C(  78), -INT8_C(  31), -INT8_C(  97),  INT8_C(  70), -INT8_C(  49),
        -INT8_C(  53), -INT8_C(  81),  INT8_C(  55),  INT8_C(  54),  INT8_C(  84),  INT8_C(  84), -INT8_C(  58),  INT8_C(  56),
         INT8_C(  94), -INT8_C(  83),  INT8_C(  11),  INT8_C(  66),  INT8_C(  53), -INT8_C(  49),  INT8_C(  16),  INT8_C(  72),
             INT8_MAX,  INT8_C(  76), -INT8_C( 116), -INT8_C(  84), -INT8_C(  28), -INT8_C(  38),  INT8_C( 115), -INT8_C(  50),
         INT8_C(  33),  INT8_C(  42),  INT8_C(  28),  INT8_C(   2), -INT8_C(  55),  INT8_C(  98), -INT8_C(  47), -INT8_C( 108),
         INT8_C(  17),  INT8_C(   9), -INT8_C(  53),  INT8_C( 102),  INT8_C(  93), -INT8_C( 111), -INT8_C(  98), -INT8_C(  69),
         INT8_C(  63), -INT8_C(  87), -INT8_C(   3),  INT8_C( 116),  INT8_C( 121),  INT8_C(  13), -INT8_C(  68), -INT8_C(   8),
         INT8_C(  89),  INT8_C(  72), -INT8_C(  91),  INT8_C(  61),  INT8_C(  34),  INT8_C(  24),  INT8_C(  11),  INT8_C(  68) },
      UINT64_C(14173835750605405431) },
    { {  INT8_C(  42), -INT8_C(  95), -INT8_C(  42), -INT8_C(  82),  INT8_C( 100),  INT8_C(  11),  INT8_C(  61),  INT8_C(  49),
        -INT8_C(  23),  INT8_C(  85), -INT8_C(   2), -INT8_C(  25), -INT8_C( 121),  INT8_C(   7),  INT8_C(  61),  INT8_C(  47),
        -INT8_C(  52),      INT8_MAX,  INT8_C(  88),  INT8_C(   1),  INT8_C(  29),  INT8_C( 103), -INT8_C( 119), -INT8_C(  61),
        -INT8_C(  93),  INT8_C(  53),  INT8_C(  80), -INT8_C(  71),  INT8_C(  51), -INT8_C(  13),  INT8_C(  23),  INT8_C(  93),
         INT8_C( 107), -INT8_C(  19),  INT8_C(  32), -INT8_C(   7),  INT8_C(  40), -INT8_C( 109), -INT8_C(   4),  INT8_C(  60),
        -INT8_C( 110), -INT8_C(   6),  INT8_C(  87),  INT8_C(  54),  INT8_C(   2), -INT8_C( 108),  INT8_C(   7),  INT8_C(  98),
        -INT8_C(  74),  INT8_C( 100), -INT8_C(  90), -INT8_C(  68), -INT8_C(  53), -INT8_C(  18),  INT8_C(  13),  INT8_C( 120),
        -INT8_C(  52),  INT8_C( 121), -INT8_C(  37),  INT8_C(   0),  INT8_C(  81),  INT8_C(  64),  INT8_C(  93), -INT8_C(  25) },
      {  INT8_C(  88),  INT8_C(  94),  INT8_C(  32),  INT8_C(  81), -INT8_C( 101),  INT8_C(  28), -INT8_C( 115),  INT8_C(  46),
         INT8_C(  22), -INT8_C(  28),  INT8_C( 103),  INT8_C(  24),  INT8_C( 120), -INT8_C(  49), -INT8_C(  38),  INT8_C(  46),
         INT8_C(  51),      INT8_MIN, -INT8_C(  89), -INT8_C(   2), -INT8_C(  30), -INT8_C(  29),  INT8_C( 118), -INT8_C(  81),
         INT8_C(  92), -INT8_C(  54), -INT8_C(  81),  INT8_C(  70),  INT8_C(  53),  INT8_C(  12), -INT8_C( 101), -INT8_C( 115),
         INT8_C( 106), -INT8_C(  69), -INT8_C(  33),  INT8_C(   6), -INT8_C(  41),  INT8_C( 108),  INT8_C(  52), -INT8_C(  19),
         INT8_C(  81), -INT8_C( 101),  INT8_C(   6), -INT8_C(  55),  INT8_C( 106), -INT8_C(  32), -INT8_C(   8), -INT8_C(  99),
         INT8_C(  97), -INT8_C(  97), -INT8_C( 101),  INT8_C(  67), -INT8_C( 126),  INT8_C(  17), -INT8_C(  14), -INT8_C(  34),
        -INT8_C(  37), -INT8_C(  95),  INT8_C(  36),  INT8_C(  16), -INT8_C(  82), -INT8_C(  65), -INT8_C(  98),  INT8_C(  24) },
      UINT64_C( 4870422836182836961) },
    { {  INT8_C(  93), -INT8_C(  97),  INT8_C( 109), -INT8_C(  63),  INT8_C( 100), -INT8_C(  86),  INT8_C(  67), -INT8_C( 112),
         INT8_C(  41),  INT8_C(  29), -INT8_C(  31), -INT8_C(  16), -INT8_C(  20), -INT8_C(  72),  INT8_C(  60),  INT8_C(  74),
         INT8_C(  89), -INT8_C(  64),  INT8_C(  93), -INT8_C(  75),  INT8_C(  96), -INT8_C(  60),  INT8_C( 111),  INT8_C( 116),
         INT8_C(  20),  INT8_C(  53),  INT8_C( 105),  INT8_C(  22), -INT8_C(   7),  INT8_C(  20),  INT8_C( 110),  INT8_C(  74),
        -INT8_C(  86), -INT8_C(  73),  INT8_C(  11),  INT8_C(  14),  INT8_C(  98),  INT8_C(  79), -INT8_C(  97), -INT8_C( 117),
         INT8_C( 115),  INT8_C( 116),  INT8_C( 125), -INT8_C(  57),  INT8_C(  93), -INT8_C( 100), -INT8_C(  64),  INT8_C(  90),
         INT8_C(  92), -INT8_C(  21), -INT8_C(  88),  INT8_C(  84),  INT8_C(   3), -INT8_C(  73),  INT8_C( 120),  INT8_C( 125),
        -INT8_C(  20),  INT8_C(  50),  INT8_C( 125), -INT8_C(  59), -INT8_C(   2), -INT8_C( 100),  INT8_C(  28), -INT8_C(  88) },
      { -INT8_C(  94),  INT8_C(  96), -INT8_C(  74),  INT8_C(  36), -INT8_C(  81),  INT8_C(  85), -INT8_C(  81),  INT8_C(  28),
        -INT8_C(  42),  INT8_C(  15),  INT8_C( 116),  INT8_C(  15), -INT8_C(  85),  INT8_C(  23), -INT8_C(  95),  INT8_C(   7),
         INT8_C(   2), -INT8_C(  24), -INT8_C(  94), -INT8_C(  78), -INT8_C(  97),  INT8_C(  26),  INT8_C(  47), -INT8_C( 117),
        -INT8_C(  21), -INT8_C(  83), -INT8_C( 106), -INT8_C(  23),  INT8_C(   6), -INT8_C(  21), -INT8_C( 111), -INT8_C(  87),
         INT8_C(  75),  INT8_C(  72), -INT8_C(  51), -INT8_C(   5), -INT8_C(  99),  INT8_C( 125),  INT8_C(  23),  INT8_C( 115),
        -INT8_C( 116), -INT8_C( 117), -INT8_C( 126),  INT8_C(  56), -INT8_C(  94),  INT8_C(  35),  INT8_C(  63), -INT8_C(  91),
         INT8_C(  12), -INT8_C(  30),  INT8_C(  87), -INT8_C(  85), -INT8_C(   4), -INT8_C( 121),  INT8_C(  55), -INT8_C(  24),
         INT8_C(  52), -INT8_C(  51), -INT8_C(  47),  INT8_C(  58), -INT8_C(  72),  INT8_C(  99), -INT8_C(  29),  INT8_C(   3) },
      UINT64_C( 1577105314603005652) },
    { { -INT8_C(   8),  INT8_C(  28),  INT8_C(  84), -INT8_C(  27), -INT8_C(  64),      INT8_MAX,  INT8_C( 115), -INT8_C(  37),
         INT8_C( 124), -INT8_C(  99),  INT8_C(  88),  INT8_C(  33), -INT8_C(   1),  INT8_C(  48),  INT8_C( 126), -INT8_C(   7),
         INT8_C( 111),  INT8_C(  63),  INT8_C(  63), -INT8_C( 100), -INT8_C(  43),  INT8_C( 118),  INT8_C(  15), -INT8_C( 109),
         INT8_C( 103),  INT8_C(  87),  INT8_C( 126),  INT8_C( 114),  INT8_C(  52), -INT8_C(  94), -INT8_C( 106), -INT8_C(  53),
         INT8_C(  37),  INT8_C(  19), -INT8_C(  49),  INT8_C(  17),      INT8_MAX, -INT8_C(  55),  INT8_C(  86), -INT8_C(  74),
        -INT8_C(  69), -INT8_C( 115),  INT8_C(  29), -INT8_C(  62), -INT8_C(  67), -INT8_C( 105),  INT8_C(  80), -INT8_C(  94),
        -INT8_C(  36), -INT8_C( 112),  INT8_C(  63),  INT8_C(  78), -INT8_C(  18),  INT8_C(  78), -INT8_C(  31), -INT8_C(  19),
        -INT8_C(  91),  INT8_C(  96),  INT8_C(  16), -INT8_C(  61), -INT8_C(  95), -INT8_C(  32), -INT8_C( 115), -INT8_C(  58) },
      {  INT8_C( 105), -INT8_C(  29), -INT8_C(  85), -INT8_C(   5), -INT8_C(  83),  INT8_C( 107), -INT8_C(  79), -INT8_C(  69),
        -INT8_C( 125),  INT8_C(  98), -INT8_C(  55), -INT8_C(  34), -INT8_C(   7), -INT8_C(  49), -INT8_C( 127),  INT8_C( 114),
        -INT8_C( 112), -INT8_C(  64), -INT8_C(  64), -INT8_C(  58),  INT8_C(  14), -INT8_C(  95),  INT8_C( 100), -INT8_C(  77),
         INT8_C(   1),  INT8_C( 116), -INT8_C(  48), -INT8_C(  93), -INT8_C(  53),  INT8_C(  93),  INT8_C( 105),  INT8_C(  52),
         INT8_C(  65),  INT8_C(  21),  INT8_C(  48), -INT8_C(  18),      INT8_MIN, -INT8_C(  31), -INT8_C(  87),  INT8_C(   4),
         INT8_C(  68),  INT8_C( 114), -INT8_C(  30),  INT8_C(  61),  INT8_C(  66),  INT8_C(  99), -INT8_C(  81), -INT8_C(  46),
         INT8_C(  35),  INT8_C( 111), -INT8_C( 104),  INT8_C(  49),  INT8_C(  17), -INT8_C(   3), -INT8_C(  28),  INT8_C(  18),
         INT8_C( 113), -INT8_C(  76), -INT8_C(  75),  INT8_C(  60),  INT8_C(  18),  INT8_C(  31),  INT8_C( 113),  INT8_C(  83) },
      UINT64_C(14367785333426787577) },
    { {  INT8_C(  93), -INT8_C(  50),  INT8_C(   5),  INT8_C(  79), -INT8_C( 118),  INT8_C( 116),  INT8_C( 113), -INT8_C(  87),
         INT8_C( 126),  INT8_C( 118),  INT8_C(  54),  INT8_C(  69),  INT8_C(  75),  INT8_C(   8),  INT8_C(  47), -INT8_C(  45),
         INT8_C( 121), -INT8_C(  36), -INT8_C(  87),  INT8_C(   8),  INT8_C(   5),  INT8_C( 109),  INT8_C( 107),  INT8_C(  18),
         INT8_C(  18),  INT8_C(  89), -INT8_C( 114),  INT8_C(  94), -INT8_C( 103), -INT8_C(  40),  INT8_C(  54),  INT8_C( 107),
         INT8_C(  38), -INT8_C(  52), -INT8_C(  70), -INT8_C(  80), -INT8_C(  80), -INT8_C(  68), -INT8_C(  61),  INT8_C(  93),
        -INT8_C( 123),  INT8_C(  92), -INT8_C(   5),  INT8_C(  17),  INT8_C( 118),  INT8_C(  42),  INT8_C(  82), -INT8_C(  17),
         INT8_C( 120), -INT8_C(   4),  INT8_C( 108), -INT8_C(  67), -INT8_C(  11),  INT8_C( 110), -INT8_C( 109),  INT8_C(  13),
         INT8_C(  48), -INT8_C(  96), -INT8_C( 115), -INT8_C(  55),  INT8_C( 121), -INT8_C(  61),  INT8_C(  52),  INT8_C( 100) },
      { -INT8_C(  94), -INT8_C(  18), -INT8_C(   6), -INT8_C(  39), -INT8_C(  47), -INT8_C( 117), -INT8_C( 114),  INT8_C(  86),
         INT8_C(  83), -INT8_C( 119),  INT8_C( 103), -INT8_C(  55), -INT8_C(  76), -INT8_C(   9), -INT8_C(  72),  INT8_C(  44),
         INT8_C(   1),  INT8_C(  35),  INT8_C(  86), -INT8_C(   9), -INT8_C(   6), -INT8_C( 110), -INT8_C(   2),  INT8_C(  42),
         INT8_C(  93),  INT8_C( 100), -INT8_C(  13),      INT8_MIN,  INT8_C(  58),  INT8_C(  39), -INT8_C(  55), -INT8_C(  36),
         INT8_C(  21), -INT8_C(  61), -INT8_C(  75), -INT8_C(  26),  INT8_C(  79),  INT8_C(  67),  INT8_C(  60), -INT8_C(  94),
        -INT8_C(  51), -INT8_C(  93),  INT8_C( 108), -INT8_C( 127), -INT8_C( 102),  INT8_C(  36), -INT8_C(  83), -INT8_C( 100),
         INT8_C(  72),  INT8_C(   3), -INT8_C( 109),  INT8_C(  66), -INT8_C( 107), -INT8_C( 111),  INT8_C( 108), -INT8_C(  14),
        -INT8_C(  11),  INT8_C(  95),  INT8_C( 114),  INT8_C(  47), -INT8_C( 122),  INT8_C(  60),  INT8_C(  11), -INT8_C( 101) },
      UINT64_C(  653511295613619482) },
    { {  INT8_C(  91), -INT8_C(  55), -INT8_C(  18), -INT8_C(  31),  INT8_C( 100),  INT8_C( 100),  INT8_C(  16),  INT8_C(  95),
         INT8_C(  55),  INT8_C( 118), -INT8_C( 117),  INT8_C(   9), -INT8_C(  57), -INT8_C(  86),  INT8_C(  52),  INT8_C(  23),
        -INT8_C(  43),  INT8_C( 118), -INT8_C(  30), -INT8_C(  55), -INT8_C(  58), -INT8_C( 108),  INT8_C( 109),  INT8_C(  80),
        -INT8_C(  20),  INT8_C(  18),  INT8_C(  90), -INT8_C( 121), -INT8_C( 109), -INT8_C(  67),  INT8_C(  51), -INT8_C(  39),
         INT8_C(  46),  INT8_C( 107), -INT8_C(  64), -INT8_C( 109),  INT8_C(  30), -INT8_C(  15),  INT8_C(  59), -INT8_C(  79),
         INT8_C(   5), -INT8_C(  58), -INT8_C(  70), -INT8_C(  52),  INT8_C(  56), -INT8_C(  69), -INT8_C(  29), -INT8_C(  28),
         INT8_C(  49),  INT8_C(  41), -INT8_C(  83),  INT8_C(  29), -INT8_C(  30), -INT8_C(  81),  INT8_C(  73),  INT8_C(  91),
         INT8_C( 118), -INT8_C(  88),  INT8_C(  51),  INT8_C(   9),  INT8_C( 101), -INT8_C(  80), -INT8_C(  30),  INT8_C(  11) },
      { -INT8_C(  92),  INT8_C(  54),  INT8_C(  17),  INT8_C(  30), -INT8_C( 101),  INT8_C(  77), -INT8_C(  49), -INT8_C(  96),
         INT8_C(  19), -INT8_C( 119),  INT8_C( 108), -INT8_C( 124),  INT8_C(  32),  INT8_C(  79), -INT8_C(  53),  INT8_C(   0),
         INT8_C(  84), -INT8_C( 119),  INT8_C(  29),  INT8_C(  54),  INT8_C(  57),  INT8_C( 107), -INT8_C( 110), -INT8_C(  81),
         INT8_C(  19), -INT8_C(  19), -INT8_C(  72),  INT8_C( 120),  INT8_C( 124), -INT8_C( 101), -INT8_C( 125),  INT8_C(  32),
        -INT8_C(  47), -INT8_C( 108),  INT8_C(  63),  INT8_C( 108), -INT8_C(  31),  INT8_C(  14),  INT8_C(  12), -INT8_C(  11),
        -INT8_C( 104),  INT8_C( 120),  INT8_C( 121), -INT8_C(  72), -INT8_C(  57),  INT8_C(  68), -INT8_C(  72),  INT8_C(  27),
        -INT8_C(  50), -INT8_C(  42),  INT8_C(  82),  INT8_C(   7),  INT8_C(  65), -INT8_C(  28), -INT8_C(  74),  INT8_C(  84),
        -INT8_C(  47),  INT8_C( 110), -INT8_C(  52),  INT8_C(  77),  INT8_C(   9),  INT8_C(  79),  INT8_C( 110), -INT8_C(  37) },
      UINT64_C(15832491078507898144) },
    { { -INT8_C(  50),  INT8_C(  11),  INT8_C(  17),  INT8_C(  45), -INT8_C(  72), -INT8_C(  66),  INT8_C(  22), -INT8_C( 119),
        -INT8_C(  96), -INT8_C( 123), -INT8_C(  15),  INT8_C( 121), -INT8_C(  30), -INT8_C(  83), -INT8_C(  54), -INT8_C(  44),
         INT8_C( 123), -INT8_C( 121), -INT8_C(  67),  INT8_C( 106),  INT8_C(  44),  INT8_C(  63), -INT8_C(  77), -INT8_C(  40),
         INT8_C(   0),  INT8_C(  41),  INT8_C(   0), -INT8_C(  66), -INT8_C(  42), -INT8_C(  49),  INT8_C(  59), -INT8_C(  92),
        -INT8_C(  38),  INT8_C(  94),  INT8_C(  76), -INT8_C(  33), -INT8_C(  28), -INT8_C(  30),  INT8_C(  44),  INT8_C(  23),
        -INT8_C(  23),  INT8_C(  29),  INT8_C( 121), -INT8_C(  52), -INT8_C(  83), -INT8_C(  93), -INT8_C( 124),  INT8_C(  41),
         INT8_C( 123), -INT8_C(  26), -INT8_C( 109),  INT8_C(  85),  INT8_C( 114), -INT8_C( 114),  INT8_C(  34), -INT8_C( 116),
        -INT8_C(  72), -INT8_C(  39), -INT8_C( 119),  INT8_C(  11), -INT8_C(  88), -INT8_C(  42),  INT8_C(  42),  INT8_C(  85) },
      {  INT8_C(  52), -INT8_C(  31),  INT8_C(  21), -INT8_C(  46), -INT8_C(  61),  INT8_C(  65), -INT8_C(  23),  INT8_C( 118),
         INT8_C(  95),  INT8_C( 122), -INT8_C(   6),  INT8_C(  12),  INT8_C(  29),  INT8_C(  82),  INT8_C(  53),  INT8_C(  71),
         INT8_C( 104), -INT8_C(  56), -INT8_C(  98), -INT8_C(  38),  INT8_C(  37), -INT8_C(  64),  INT8_C(  76),  INT8_C(  39),
        -INT8_C( 103), -INT8_C(  42), -INT8_C(   1),  INT8_C(  65), -INT8_C(  84),  INT8_C( 124), -INT8_C(  60), -INT8_C(  31),
         INT8_C(  93), -INT8_C(  39), -INT8_C(  77),  INT8_C(  32),  INT8_C(  27), -INT8_C( 100), -INT8_C( 106),  INT8_C( 122),
         INT8_C(  22), -INT8_C( 112), -INT8_C( 122),  INT8_C(  51), -INT8_C(  29), -INT8_C(  68),  INT8_C( 123),  INT8_C(  75),
        -INT8_C( 124),  INT8_C(  25),  INT8_C(  37), -INT8_C(  86), -INT8_C(  39),  INT8_C( 113), -INT8_C(  47),  INT8_C( 115),
         INT8_C(  71), -INT8_C(  47), -INT8_C(  76), -INT8_C(  12),  INT8_C(  77),  INT8_C( 120), -INT8_C(  43), -INT8_C(  86) },
      UINT64_C( 3896936268564499479) },
    { { -INT8_C(  95), -INT8_C(  68),  INT8_C( 116),  INT8_C(  68), -INT8_C(  88), -INT8_C(  77),  INT8_C(   3),  INT8_C( 109),
        -INT8_C(  75), -INT8_C(  56),  INT8_C(  38),  INT8_C(  68), -INT8_C(  23),  INT8_C( 102),  INT8_C(  82),  INT8_C(  61),
         INT8_C(  76),  INT8_C(  69),  INT8_C( 121), -INT8_C( 121),  INT8_C( 101),  INT8_C(  26), -INT8_C(  15), -INT8_C( 100),
         INT8_C( 107),  INT8_C(  66), -INT8_C(  14), -INT8_C(  19),  INT8_C(  94), -INT8_C(  48), -INT8_C(  30),  INT8_C(  54),
        -INT8_C( 116),  INT8_C( 110),  INT8_C(  87),  INT8_C( 106),  INT8_C(  93),  INT8_C(   0), -INT8_C(  41), -INT8_C(  40),
        -INT8_C(  55), -INT8_C(   2),  INT8_C(   2),  INT8_C( 101), -INT8_C( 117),  INT8_C(  45),  INT8_C(  46), -INT8_C(  40),
         INT8_C(  94),  INT8_C(  28), -INT8_C( 103), -INT8_C(  61), -INT8_C( 125), -INT8_C( 118),  INT8_C( 116), -INT8_C( 106),
         INT8_C(  31), -INT8_C(  82), -INT8_C( 112), -INT8_C(  58),  INT8_C( 126),  INT8_C(  35), -INT8_C(   4),  INT8_C(  44) },
      {  INT8_C(  94),  INT8_C(  40),  INT8_C( 116), -INT8_C(  69),  INT8_C(  87),  INT8_C(  76),  INT8_C(  87),  INT8_C( 108),
         INT8_C(  74),  INT8_C(  55),  INT8_C( 107), -INT8_C(  43),  INT8_C( 100), -INT8_C( 103), -INT8_C(  83), -INT8_C(  62),
        -INT8_C(  11), -INT8_C(  70), -INT8_C( 122),  INT8_C( 120), -INT8_C(  33), -INT8_C(  27),  INT8_C(  14),  INT8_C(  71),
        -INT8_C( 108),  INT8_C( 107),  INT8_C(  13),  INT8_C(  18), -INT8_C( 114),  INT8_C(   9),  INT8_C(  29), -INT8_C(  19),
         INT8_C(  49), -INT8_C( 111), -INT8_C(  88), -INT8_C( 120), -INT8_C(  35), -INT8_C(   1), -INT8_C(  11),  INT8_C(  39),
         INT8_C(  54),  INT8_C(  96), -INT8_C(   3), -INT8_C( 102), -INT8_C(   7), -INT8_C(  86),  INT8_C(  93), -INT8_C(  18),
         INT8_C( 101), -INT8_C(  29),  INT8_C( 102),  INT8_C(  68), -INT8_C(  56),  INT8_C( 117), -INT8_C( 117),  INT8_C(  92),
        -INT8_C(  32), -INT8_C( 104),  INT8_C( 111),  INT8_C( 110), -INT8_C(  94), -INT8_C( 116),  INT8_C(  91), -INT8_C(  45) },
      UINT64_C( 6528515597080272070) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_test_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_test_epi8_mask");
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

    for (size_t j = 0 ; j < 64 ; j++) if (rand() & 1) a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi8(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(b_);
    easysimd__mmask64 r = easysimd_mm512_test_epi8_mask(a, b);

    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_test_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_test_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_test_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_test_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_test_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_test_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_test_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_test_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_test_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_test_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_test_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_test_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_test_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_test_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_test_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_test_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_test_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_test_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_test_epi8_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_test_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_test_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_test_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_test_epi8_mask)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
