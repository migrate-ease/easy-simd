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
 */

#define EASYSIMD_TEST_X86_AVX512_INSN packus

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/packus.h>

static int
test_easysimd_mm_mask_packus_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t src[16];
    const uint16_t k;
    const int16_t a[8];
    const int16_t b[8];
    const uint8_t r[16];
  } test_vec[] = {
    { { UINT8_C(119), UINT8_C(176), UINT8_C( 66), UINT8_C(154), UINT8_C(147), UINT8_C( 16), UINT8_C(133), UINT8_C(196),
        UINT8_C(145), UINT8_C(108), UINT8_C(228), UINT8_C(146), UINT8_C(243), UINT8_C(151), UINT8_C(183), UINT8_C( 24) },
      UINT16_C(53309),
      {  INT16_C( 21388),  INT16_C( 11168), -INT16_C( 10154),  INT16_C( 23137),  INT16_C( 13282),  INT16_C( 14888), -INT16_C( 24667), -INT16_C(  6166) },
      {  INT16_C( 32313), -INT16_C( 16393), -INT16_C( 30654),  INT16_C(  9771),  INT16_C(  7962), -INT16_C( 11587), -INT16_C(  1225), -INT16_C( 15454) },
      {    UINT8_MAX, UINT8_C(176), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(133), UINT8_C(196),
        UINT8_C(145), UINT8_C(108), UINT8_C(228), UINT8_C(146),    UINT8_MAX, UINT8_C(151), UINT8_C(  0), UINT8_C(  0) } },
    { { UINT8_C( 78), UINT8_C( 66), UINT8_C(238), UINT8_C(164), UINT8_C( 27), UINT8_C( 80), UINT8_C(254), UINT8_C(253),
        UINT8_C(131), UINT8_C( 38), UINT8_C( 55), UINT8_C( 41), UINT8_C(197), UINT8_C( 34), UINT8_C( 16),    UINT8_MAX },
      UINT16_C( 2208),
      { -INT16_C(  7490), -INT16_C(  5744), -INT16_C( 21752), -INT16_C( 14840),  INT16_C( 16509),  INT16_C(  8129),  INT16_C(  3843), -INT16_C(  3486) },
      {  INT16_C( 32180), -INT16_C( 19902), -INT16_C( 14982), -INT16_C( 20007), -INT16_C( 24850), -INT16_C(    45),  INT16_C( 29597),  INT16_C( 23303) },
      { UINT8_C( 78), UINT8_C( 66), UINT8_C(238), UINT8_C(164), UINT8_C( 27),    UINT8_MAX, UINT8_C(254), UINT8_C(  0),
        UINT8_C(131), UINT8_C( 38), UINT8_C( 55), UINT8_C(  0), UINT8_C(197), UINT8_C( 34), UINT8_C( 16),    UINT8_MAX } },
    { { UINT8_C( 85), UINT8_C(151), UINT8_C( 69), UINT8_C( 94), UINT8_C( 66), UINT8_C( 77), UINT8_C( 36), UINT8_C(191),
        UINT8_C(141), UINT8_C(229), UINT8_C(223), UINT8_C(145), UINT8_C(244), UINT8_C( 65), UINT8_C(131), UINT8_C(168) },
      UINT16_C(50622),
      {  INT16_C( 14427),  INT16_C( 13450),  INT16_C( 31209), -INT16_C( 16942),  INT16_C( 28792),  INT16_C( 32560), -INT16_C( 31029),  INT16_C(  4118) },
      {  INT16_C( 23012),  INT16_C(  2142), -INT16_C(  5352), -INT16_C(  2067), -INT16_C(  7812), -INT16_C(   200), -INT16_C(  2422), -INT16_C(  6716) },
      { UINT8_C( 85),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C( 36),    UINT8_MAX,
           UINT8_MAX, UINT8_C(229), UINT8_C(  0), UINT8_C(145), UINT8_C(244), UINT8_C( 65), UINT8_C(  0), UINT8_C(  0) } },
    { { UINT8_C( 46), UINT8_C( 79), UINT8_C( 25), UINT8_C( 24), UINT8_C(200), UINT8_C(235), UINT8_C(213), UINT8_C( 64),
        UINT8_C( 91), UINT8_C(  5), UINT8_C(191), UINT8_C( 39), UINT8_C(139), UINT8_C(213), UINT8_C( 55), UINT8_C(111) },
      UINT16_C(38190),
      {  INT16_C( 18295),  INT16_C( 25729), -INT16_C(   706),  INT16_C( 30534), -INT16_C( 12035), -INT16_C( 16019), -INT16_C( 25419), -INT16_C( 12784) },
      { -INT16_C( 10060), -INT16_C( 30279),  INT16_C(  5400), -INT16_C( 10354),  INT16_C(  6716),  INT16_C( 29613), -INT16_C(  9335),  INT16_C(   265) },
      { UINT8_C( 46),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(200), UINT8_C(  0), UINT8_C(213), UINT8_C( 64),
        UINT8_C(  0), UINT8_C(  5),    UINT8_MAX, UINT8_C( 39),    UINT8_MAX, UINT8_C(213), UINT8_C( 55),    UINT8_MAX } },
    { { UINT8_C( 34), UINT8_C(138), UINT8_C(101), UINT8_C( 97), UINT8_C(135), UINT8_C(171), UINT8_C(216), UINT8_C(132),
        UINT8_C(123), UINT8_C( 69), UINT8_C( 70), UINT8_C( 48), UINT8_C(225), UINT8_C( 86), UINT8_C(254), UINT8_C(149) },
      UINT16_C(47151),
      {  INT16_C( 18206), -INT16_C( 21043),  INT16_C(  2335), -INT16_C( 13113),  INT16_C( 20604), -INT16_C( 31321), -INT16_C( 13743), -INT16_C( 18673) },
      { -INT16_C( 26837),  INT16_C(   866), -INT16_C(  8677),  INT16_C( 24904),  INT16_C( 10766),  INT16_C(  3512), -INT16_C(  6209), -INT16_C(  8507) },
      {    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(135), UINT8_C(  0), UINT8_C(216), UINT8_C(132),
        UINT8_C(123), UINT8_C( 69), UINT8_C( 70),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(254), UINT8_C(  0) } },
    { { UINT8_C( 46), UINT8_C(146), UINT8_C(139), UINT8_C( 77), UINT8_C(155), UINT8_C( 82), UINT8_C( 25), UINT8_C( 23),
        UINT8_C(162), UINT8_C(193), UINT8_C(157), UINT8_C(244), UINT8_C(139), UINT8_C(172), UINT8_C(171), UINT8_C(182) },
      UINT16_C( 3395),
      {  INT16_C( 24505),  INT16_C(   491), -INT16_C(  1344),  INT16_C( 30763), -INT16_C(  5369), -INT16_C( 13217), -INT16_C( 28983),  INT16_C( 21598) },
      { -INT16_C(  1573), -INT16_C(  2650),  INT16_C( 18448), -INT16_C( 21066),  INT16_C( 16700), -INT16_C(  6310), -INT16_C( 25097), -INT16_C( 20235) },
      {    UINT8_MAX,    UINT8_MAX, UINT8_C(139), UINT8_C( 77), UINT8_C(155), UINT8_C( 82), UINT8_C(  0), UINT8_C( 23),
        UINT8_C(  0), UINT8_C(193),    UINT8_MAX, UINT8_C(  0), UINT8_C(139), UINT8_C(172), UINT8_C(171), UINT8_C(182) } },
    { { UINT8_C(252), UINT8_C(224), UINT8_C(177), UINT8_C(189), UINT8_C(218), UINT8_C(221), UINT8_C( 53), UINT8_C(225),
        UINT8_C(200), UINT8_C(149), UINT8_C(173), UINT8_C(145), UINT8_C( 35), UINT8_C( 11), UINT8_C(229), UINT8_C(254) },
      UINT16_C(35588),
      {  INT16_C(  5619), -INT16_C( 22061),  INT16_C(  4290),  INT16_C(  7402), -INT16_C(  7689), -INT16_C(  4934), -INT16_C( 18799),  INT16_C( 17357) },
      { -INT16_C( 22669), -INT16_C( 22240), -INT16_C(  6007),  INT16_C( 13886),  INT16_C( 24953),  INT16_C( 24130),  INT16_C( 18015),  INT16_C( 21481) },
      { UINT8_C(252), UINT8_C(224),    UINT8_MAX, UINT8_C(189), UINT8_C(218), UINT8_C(221), UINT8_C( 53), UINT8_C(225),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(173),    UINT8_MAX, UINT8_C( 35), UINT8_C( 11), UINT8_C(229),    UINT8_MAX } },
    { { UINT8_C( 91), UINT8_C(188), UINT8_C(252), UINT8_C( 30), UINT8_C(204), UINT8_C(231), UINT8_C( 58), UINT8_C(196),
        UINT8_C(200), UINT8_C(244), UINT8_C(176), UINT8_C( 90), UINT8_C(171), UINT8_C(125), UINT8_C(157), UINT8_C( 30) },
      UINT16_C(48421),
      { -INT16_C( 20793),  INT16_C(  1445),  INT16_C(  7908),  INT16_C(  9830), -INT16_C( 14724),  INT16_C( 25965), -INT16_C( 14311),  INT16_C(  5409) },
      { -INT16_C(  4378),  INT16_C(  8700), -INT16_C( 14926),  INT16_C( 25109), -INT16_C( 16353), -INT16_C( 17184),  INT16_C(  1503), -INT16_C( 22919) },
      { UINT8_C(  0), UINT8_C(188),    UINT8_MAX, UINT8_C( 30), UINT8_C(204),    UINT8_MAX, UINT8_C( 58), UINT8_C(196),
        UINT8_C(  0), UINT8_C(244), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(157), UINT8_C(  0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_packus_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_packus_epi16");
    easysimd_test_x86_assert_equal_u8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u8x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_mask_packus_epi16(src, k, a, b);

    easysimd_test_x86_write_u8x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_packus_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int16_t a[8];
    const int16_t b[8];
    const uint8_t r[16];
  } test_vec[] = {
    { UINT16_C(55316),
      { -INT16_C( 20982), -INT16_C( 27540), -INT16_C( 24177), -INT16_C(  6373), -INT16_C(  2347), -INT16_C( 18039), -INT16_C( 19130), -INT16_C( 14372) },
      { -INT16_C( 31971),  INT16_C( 16814), -INT16_C( 17866),  INT16_C(  9568),  INT16_C(  8972), -INT16_C( 30578), -INT16_C( 23625), -INT16_C( 15776) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT16_C(52561),
      { -INT16_C(  8106),  INT16_C( 29038),  INT16_C( 17351),  INT16_C( 20583), -INT16_C( 20739), -INT16_C(  9978),  INT16_C(  9077),  INT16_C(  9052) },
      { -INT16_C( 28060), -INT16_C( 15139), -INT16_C(  5705),  INT16_C( 18151), -INT16_C( 24718), -INT16_C( 11543),  INT16_C( 14945), -INT16_C( 18529) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0) } },
    { UINT16_C( 3610),
      { -INT16_C(  7896), -INT16_C( 28591),  INT16_C( 20018),  INT16_C( 14398), -INT16_C( 19673), -INT16_C( 31909), -INT16_C( 16426), -INT16_C( 19691) },
      { -INT16_C( 12925),  INT16_C( 27548),  INT16_C(  3603), -INT16_C(  1014),  INT16_C( 27617), -INT16_C( 32714),  INT16_C( 20514),  INT16_C( 19086) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT16_C(57393),
      {  INT16_C( 25562),  INT16_C(  6190),  INT16_C( 22171), -INT16_C(  2357), -INT16_C( 24103), -INT16_C(  4171),  INT16_C( 14676), -INT16_C(  3652) },
      { -INT16_C( 12380), -INT16_C( 20737), -INT16_C(  7989),  INT16_C(   281),  INT16_C( 15201), -INT16_C(  4271), -INT16_C( 32123),  INT16_C( 24783) },
      {    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX } },
    { UINT16_C(65254),
      { -INT16_C( 32392),  INT16_C( 17492),  INT16_C( 11640),  INT16_C( 11749),  INT16_C( 14876), -INT16_C( 10138),  INT16_C(  2603),  INT16_C( 10919) },
      {  INT16_C( 29368), -INT16_C( 12021),  INT16_C( 27763), -INT16_C( 15348), -INT16_C( 28069),  INT16_C( 11079),  INT16_C( 11762),  INT16_C( 27177) },
      { UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,
        UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX } },
    { UINT16_C(32174),
      {  INT16_C(  9902), -INT16_C( 27478), -INT16_C( 14508), -INT16_C( 17714), -INT16_C(  1633),  INT16_C( 18373),  INT16_C( 32035),  INT16_C( 11961) },
      {  INT16_C( 11599),  INT16_C( 23450), -INT16_C(  2319),  INT16_C( 14573), -INT16_C(  8415),  INT16_C( 19045),  INT16_C(  5194), -INT16_C(  1849) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,
           UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0) } },
    { UINT16_C(28986),
      { -INT16_C( 29044),  INT16_C( 23096), -INT16_C( 10167),  INT16_C(  3667),  INT16_C( 30495), -INT16_C( 10101), -INT16_C(  9563),  INT16_C( 16389) },
      { -INT16_C(  2250),  INT16_C(  9014),  INT16_C( 22319), -INT16_C( 27389),  INT16_C( 19873),  INT16_C( 26793), -INT16_C(  7355), -INT16_C( 11559) },
      { UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0) } },
    { UINT16_C( 4722),
      { -INT16_C( 17620), -INT16_C( 32534),  INT16_C(  2505),  INT16_C( 21751), -INT16_C( 25375), -INT16_C(  6353),  INT16_C( 26076),  INT16_C(  4830) },
      {  INT16_C(  3464), -INT16_C( 29847),  INT16_C(  2722),  INT16_C( 19416),  INT16_C(  7794),  INT16_C( 19503), -INT16_C( 24080),  INT16_C(  7262) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_packus_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_packus_epi16");
    easysimd_test_x86_assert_equal_u8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_packus_epi16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_packus_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t src[8];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const uint16_t r[8];
  } test_vec[] = {
    { { UINT16_C( 7859), UINT16_C(38828), UINT16_C( 4668), UINT16_C(47294), UINT16_C(11224), UINT16_C(61725), UINT16_C(16115), UINT16_C(55815) },
      UINT8_C( 44),
      { -INT32_C(   924910845), -INT32_C(   773373680),  INT32_C(   649110305),  INT32_C(   987321884) },
      {  INT32_C(   360083458),  INT32_C(  1508716078), -INT32_C(  1974608053), -INT32_C(   373938202) },
      { UINT16_C( 7859), UINT16_C(38828),      UINT16_MAX,      UINT16_MAX, UINT16_C(11224),      UINT16_MAX, UINT16_C(16115), UINT16_C(55815) } },
    { { UINT16_C(38178), UINT16_C(12978), UINT16_C(39382), UINT16_C(63235), UINT16_C(45885), UINT16_C(22813), UINT16_C(62986), UINT16_C( 3220) },
      UINT8_C(102),
      {  INT32_C(   966074634), -INT32_C(   293278193),  INT32_C(  1658064443),  INT32_C(  1518648773) },
      {  INT32_C(   154187631),  INT32_C(  1850091450),  INT32_C(   980983620),  INT32_C(  1050706995) },
      { UINT16_C(38178), UINT16_C(    0),      UINT16_MAX, UINT16_C(63235), UINT16_C(45885),      UINT16_MAX,      UINT16_MAX, UINT16_C( 3220) } },
    { { UINT16_C(13734), UINT16_C(46455), UINT16_C(64292), UINT16_C(24483), UINT16_C(30474), UINT16_C(53186), UINT16_C(17972), UINT16_C(42025) },
      UINT8_C(253),
      { -INT32_C(  2135380647), -INT32_C(  1832638733),  INT32_C(   583466654), -INT32_C(   725089122) },
      {  INT32_C(  1995996539), -INT32_C(  1753196768),  INT32_C(  1624002329), -INT32_C(   782405512) },
      { UINT16_C(    0), UINT16_C(46455),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0) } },
    { { UINT16_C( 5405), UINT16_C( 4177), UINT16_C( 5435), UINT16_C(55714), UINT16_C(26643), UINT16_C(45820), UINT16_C(50284), UINT16_C(59270) },
      UINT8_C( 66),
      { -INT32_C(   714973570),  INT32_C(   787479262),  INT32_C(   916869062), -INT32_C(  1034717012) },
      { -INT32_C(   537041975),  INT32_C(  1861408517), -INT32_C(  1747278637), -INT32_C(  1462123990) },
      { UINT16_C( 5405),      UINT16_MAX, UINT16_C( 5435), UINT16_C(55714), UINT16_C(26643), UINT16_C(45820), UINT16_C(    0), UINT16_C(59270) } },
    { { UINT16_C(15392), UINT16_C(65150), UINT16_C(27958), UINT16_C(64556), UINT16_C(54204), UINT16_C(26674), UINT16_C(34123), UINT16_C( 5162) },
      UINT8_C(232),
      { -INT32_C(     1182936), -INT32_C(  1965925402), -INT32_C(   122328778),  INT32_C(  2132303171) },
      {  INT32_C(  1219827419),  INT32_C(   369406275),  INT32_C(  1751215587), -INT32_C(  1085246057) },
      { UINT16_C(15392), UINT16_C(65150), UINT16_C(27958),      UINT16_MAX, UINT16_C(54204),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { { UINT16_C(15721), UINT16_C(20414), UINT16_C(37017), UINT16_C(53209), UINT16_C(36602), UINT16_C(15815), UINT16_C(57324), UINT16_C(51132) },
      UINT8_C(245),
      {  INT32_C(   574099569), -INT32_C(  2130358764),  INT32_C(   622423471), -INT32_C(    91301699) },
      {  INT32_C(   664001943),  INT32_C(  1159815862),  INT32_C(   137453097), -INT32_C(  1946224614) },
      {      UINT16_MAX, UINT16_C(20414),      UINT16_MAX, UINT16_C(53209),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { { UINT16_C(13832), UINT16_C( 7597), UINT16_C(45701), UINT16_C(13470), UINT16_C(46879), UINT16_C(56409), UINT16_C(59280), UINT16_C(10199) },
      UINT8_C(196),
      { -INT32_C(   847557014), -INT32_C(   822689680), -INT32_C(   370540559),  INT32_C(   871527677) },
      { -INT32_C(   726135006),  INT32_C(  1710550445),  INT32_C(   787861574),  INT32_C(   317856935) },
      { UINT16_C(13832), UINT16_C( 7597), UINT16_C(    0), UINT16_C(13470), UINT16_C(46879), UINT16_C(56409),      UINT16_MAX,      UINT16_MAX } },
    { { UINT16_C(28010), UINT16_C(56031), UINT16_C(54573), UINT16_C( 7849), UINT16_C(37588), UINT16_C(53512), UINT16_C(64006), UINT16_C(10245) },
      UINT8_C(  9),
      { -INT32_C(  1430848067), -INT32_C(  1041163279),  INT32_C(   745086736),  INT32_C(  2140633873) },
      {  INT32_C(   799830362), -INT32_C(  1408972006), -INT32_C(   843852333), -INT32_C(  1730749478) },
      { UINT16_C(    0), UINT16_C(56031), UINT16_C(54573),      UINT16_MAX, UINT16_C(37588), UINT16_C(53512), UINT16_C(64006), UINT16_C(10245) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_packus_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_packus_epi32");
    easysimd_test_x86_assert_equal_u16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_packus_epi32(src, k, a, b);

    easysimd_test_x86_write_u16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_packus_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const uint16_t r[8];
  } test_vec[] = {
    { UINT8_C( 92),
      {  INT32_C(  1361419336),  INT32_C(   808614291),  INT32_C(   218896808),  INT32_C(    93724663) },
      { -INT32_C(  1817763448),  INT32_C(   403043322), -INT32_C(  1022865118), -INT32_C(   148953937) },
      { UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C(193),
      { -INT32_C(  1118549948), -INT32_C(  1805220741), -INT32_C(  1332972655),  INT32_C(   725127433) },
      {  INT32_C(   723897144),  INT32_C(   575487441),  INT32_C(  1775374405),  INT32_C(  1932183855) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX } },
    { UINT8_C( 18),
      {  INT32_C(    59584895),  INT32_C(   177480087),  INT32_C(  1041515693),  INT32_C(  1215774589) },
      { -INT32_C(  1592155548),  INT32_C(    15088880), -INT32_C(   684699890), -INT32_C(   102128774) },
      { UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C(212),
      { -INT32_C(  1737753482), -INT32_C(   716867951), -INT32_C(   917338998), -INT32_C(  1657955333) },
      { -INT32_C(   259141964),  INT32_C(    67014324),  INT32_C(  1635636926),  INT32_C(   909539007) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX } },
    { UINT8_C(115),
      {  INT32_C(   386191009), -INT32_C(  1751000813), -INT32_C(   946705876),  INT32_C(  1702572183) },
      {  INT32_C(  1259957437),  INT32_C(  1074339178),  INT32_C(   285240218),  INT32_C(  1115960993) },
      {      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { UINT8_C(  4),
      {  INT32_C(  1628920456),  INT32_C(  1720561659),  INT32_C(  1912427585),  INT32_C(  1009738704) },
      { -INT32_C(  1717142916), -INT32_C(   281745532),  INT32_C(   495994343),  INT32_C(  1361171145) },
      { UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C( 44),
      { -INT32_C(   416763336), -INT32_C(  1809215936), -INT32_C(   295396725),  INT32_C(  1147904201) },
      {  INT32_C(   784860231),  INT32_C(  2098575160),  INT32_C(   424030791), -INT32_C(  1958308013) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C( 74),
      { -INT32_C(    58035346),  INT32_C(   914824860),  INT32_C(   587232899), -INT32_C(   446020383) },
      { -INT32_C(  1021405172), -INT32_C(   552952915),  INT32_C(  2033329122), -INT32_C(   675037591) },
      { UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_packus_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_packus_epi32");
    easysimd_test_x86_assert_equal_u16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_packus_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_packus_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const uint8_t r[64];
  } test_vec[] = {
    { { -INT16_C( 11809),  INT16_C(  3301),  INT16_C(  4381), -INT16_C( 29201), -INT16_C( 11622), -INT16_C(  1564),  INT16_C(  3475), -INT16_C(  8537),
         INT16_C(  4169), -INT16_C( 23067),  INT16_C( 13975),  INT16_C( 16305), -INT16_C( 18418),  INT16_C( 12904), -INT16_C( 19774), -INT16_C( 24123),
        -INT16_C( 21629), -INT16_C( 24403), -INT16_C( 25412),  INT16_C( 22062),  INT16_C(  4719),  INT16_C(   591), -INT16_C(  2528),  INT16_C( 27104),
        -INT16_C( 15098), -INT16_C( 25330), -INT16_C( 16389),  INT16_C(  2780),  INT16_C( 17527),  INT16_C( 14652),  INT16_C(   758),  INT16_C( 31195) },
      {  INT16_C(   136),  INT16_C(   105),  INT16_C(    72),  INT16_C(   148),  INT16_C(    14),  INT16_C(   122),  INT16_C(   119),  INT16_C(    10),
         INT16_C(   241),  INT16_C(    56),  INT16_C(   132),  INT16_C(    39),  INT16_C(   126),  INT16_C(   191),  INT16_C(    60),  INT16_C(    45),
         INT16_C(    83),  INT16_C(   233),  INT16_C(    85),  INT16_C(   245),  INT16_C(    20),  INT16_C(   103),  INT16_C(    83),  INT16_C(   199),
         INT16_C(    26),  INT16_C(   245),  INT16_C(    65),  INT16_C(   103),  INT16_C(   126),  INT16_C(    64),  INT16_C(    96),  INT16_C(   126) },
      { UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),
        UINT8_C(136), UINT8_C(105), UINT8_C( 72), UINT8_C(148), UINT8_C( 14), UINT8_C(122), UINT8_C(119), UINT8_C( 10),
           UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),
        UINT8_C(241), UINT8_C( 56), UINT8_C(132), UINT8_C( 39), UINT8_C(126), UINT8_C(191), UINT8_C( 60), UINT8_C( 45),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,
        UINT8_C( 83), UINT8_C(233), UINT8_C( 85), UINT8_C(245), UINT8_C( 20), UINT8_C(103), UINT8_C( 83), UINT8_C(199),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,
        UINT8_C( 26), UINT8_C(245), UINT8_C( 65), UINT8_C(103), UINT8_C(126), UINT8_C( 64), UINT8_C( 96), UINT8_C(126) } },
    { {  INT16_C(  1203),  INT16_C( 20072), -INT16_C(  6822), -INT16_C( 17085), -INT16_C( 19463), -INT16_C( 31707), -INT16_C( 26873),  INT16_C( 19532),
         INT16_C( 19377),  INT16_C( 20289),  INT16_C( 24205),  INT16_C( 19895), -INT16_C(  8484), -INT16_C( 26995), -INT16_C(  1218), -INT16_C(  3819),
         INT16_C( 32000),  INT16_C( 23103), -INT16_C( 32158),  INT16_C( 23575),  INT16_C( 15414),  INT16_C( 15840),  INT16_C( 11475), -INT16_C( 31607),
        -INT16_C( 13704),  INT16_C(  1492), -INT16_C( 29911),  INT16_C(  1362), -INT16_C(  8343), -INT16_C( 22628), -INT16_C( 20005), -INT16_C(  9320) },
      {  INT16_C(   215),  INT16_C(   144),  INT16_C(    76),  INT16_C(   143),  INT16_C(   205),  INT16_C(    92),  INT16_C(    85),  INT16_C(   113),
         INT16_C(   181),  INT16_C(    73),  INT16_C(   200),  INT16_C(   169),  INT16_C(   234),  INT16_C(   131),  INT16_C(   232),  INT16_C(   201),
         INT16_C(   147),  INT16_C(    24),  INT16_C(    70),  INT16_C(   104),  INT16_C(   116),  INT16_C(    13),  INT16_C(   166),  INT16_C(   234),
         INT16_C(   245),  INT16_C(   155),  INT16_C(   129),  INT16_C(   101),  INT16_C(   148),  INT16_C(     7),  INT16_C(    70),  INT16_C(    59) },
      {    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,
        UINT8_C(215), UINT8_C(144), UINT8_C( 76), UINT8_C(143), UINT8_C(205), UINT8_C( 92), UINT8_C( 85), UINT8_C(113),
           UINT8_MAX,    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(181), UINT8_C( 73), UINT8_C(200), UINT8_C(169), UINT8_C(234), UINT8_C(131), UINT8_C(232), UINT8_C(201),
           UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C(147), UINT8_C( 24), UINT8_C( 70), UINT8_C(104), UINT8_C(116), UINT8_C( 13), UINT8_C(166), UINT8_C(234),
        UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(245), UINT8_C(155), UINT8_C(129), UINT8_C(101), UINT8_C(148), UINT8_C(  7), UINT8_C( 70), UINT8_C( 59) } },
    { {  INT16_C( 11225), -INT16_C( 18093), -INT16_C(  1167), -INT16_C( 31455), -INT16_C(  6544),  INT16_C( 14994),  INT16_C(  4236), -INT16_C(  6364),
         INT16_C( 22277), -INT16_C( 15486), -INT16_C( 14632),  INT16_C( 17448),  INT16_C(  4442), -INT16_C( 10676),  INT16_C(  7511),  INT16_C( 12561),
         INT16_C( 25928), -INT16_C( 17942),  INT16_C(  2912), -INT16_C( 12226), -INT16_C( 12046),  INT16_C( 32266),  INT16_C( 12001), -INT16_C(  6554),
        -INT16_C(  6011),  INT16_C( 24233), -INT16_C( 11601),  INT16_C(  2466), -INT16_C(  4381),  INT16_C( 15072), -INT16_C(  3829),  INT16_C( 21355) },
      {  INT16_C(    85),  INT16_C(   183),  INT16_C(    75),  INT16_C(    83),  INT16_C(   146),  INT16_C(   253),  INT16_C(    55),  INT16_C(    70),
         INT16_C(   141),  INT16_C(   207),  INT16_C(    70),  INT16_C(    66),  INT16_C(   184),  INT16_C(    64),  INT16_C(   232),  INT16_C(     0),
         INT16_C(   161),  INT16_C(   158),  INT16_C(    63),  INT16_C(     8),  INT16_C(   195),  INT16_C(   145),  INT16_C(   233),  INT16_C(    26),
         INT16_C(   123),  INT16_C(   213),  INT16_C(   194),  INT16_C(   247),  INT16_C(   147),  INT16_C(    36),  INT16_C(   203),  INT16_C(   185) },
      {    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C( 85), UINT8_C(183), UINT8_C( 75), UINT8_C( 83), UINT8_C(146), UINT8_C(253), UINT8_C( 55), UINT8_C( 70),
           UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,
        UINT8_C(141), UINT8_C(207), UINT8_C( 70), UINT8_C( 66), UINT8_C(184), UINT8_C( 64), UINT8_C(232), UINT8_C(  0),
           UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C(161), UINT8_C(158), UINT8_C( 63), UINT8_C(  8), UINT8_C(195), UINT8_C(145), UINT8_C(233), UINT8_C( 26),
        UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,
        UINT8_C(123), UINT8_C(213), UINT8_C(194), UINT8_C(247), UINT8_C(147), UINT8_C( 36), UINT8_C(203), UINT8_C(185) } },
    { { -INT16_C(  9108),  INT16_C( 22871),  INT16_C( 18715), -INT16_C(  5023),  INT16_C( 26380),  INT16_C(  1662),  INT16_C( 21840), -INT16_C( 14815),
         INT16_C(  2769), -INT16_C( 27749), -INT16_C( 19764),  INT16_C( 18314), -INT16_C( 16059), -INT16_C( 16021), -INT16_C( 28531), -INT16_C(  1670),
        -INT16_C( 11923), -INT16_C( 30638), -INT16_C( 19430),  INT16_C(  9845), -INT16_C(  3301),  INT16_C( 27437),  INT16_C( 20040),  INT16_C(  6449),
        -INT16_C( 13224),  INT16_C(  9644),  INT16_C( 13950), -INT16_C( 15508), -INT16_C( 10248), -INT16_C( 31356), -INT16_C(   408), -INT16_C( 10882) },
      {  INT16_C(   209),  INT16_C(   234),  INT16_C(   210),  INT16_C(   160),  INT16_C(    62),  INT16_C(    14),  INT16_C(    60),  INT16_C(   228),
         INT16_C(   212),  INT16_C(   134),  INT16_C(   117),  INT16_C(     2),  INT16_C(   206),  INT16_C(   181),  INT16_C(     6),  INT16_C(   156),
         INT16_C(   231),  INT16_C(    92),  INT16_C(   152),  INT16_C(   127),  INT16_C(     7),  INT16_C(    98),  INT16_C(   181),  INT16_C(    75),
         INT16_C(    80),  INT16_C(   147),  INT16_C(    26),  INT16_C(    18),  INT16_C(    29),  INT16_C(   181),  INT16_C(    81),  INT16_C(   250) },
      { UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C(209), UINT8_C(234), UINT8_C(210), UINT8_C(160), UINT8_C( 62), UINT8_C( 14), UINT8_C( 60), UINT8_C(228),
           UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(212), UINT8_C(134), UINT8_C(117), UINT8_C(  2), UINT8_C(206), UINT8_C(181), UINT8_C(  6), UINT8_C(156),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,
        UINT8_C(231), UINT8_C( 92), UINT8_C(152), UINT8_C(127), UINT8_C(  7), UINT8_C( 98), UINT8_C(181), UINT8_C( 75),
        UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 80), UINT8_C(147), UINT8_C( 26), UINT8_C( 18), UINT8_C( 29), UINT8_C(181), UINT8_C( 81), UINT8_C(250) } },
    { { -INT16_C( 10183), -INT16_C(  3242),  INT16_C( 21104),  INT16_C( 18034),  INT16_C(    89), -INT16_C( 25432), -INT16_C(  4171),  INT16_C( 16103),
        -INT16_C( 18369),  INT16_C(  1233),  INT16_C( 26579), -INT16_C( 17641), -INT16_C(  8571), -INT16_C( 22416), -INT16_C( 15824),  INT16_C( 27043),
        -INT16_C(  1638),  INT16_C(  2908), -INT16_C( 12724), -INT16_C( 23215), -INT16_C(  1330), -INT16_C( 31934),  INT16_C( 10729),  INT16_C( 10433),
        -INT16_C( 27678), -INT16_C( 19156),  INT16_C( 17402),  INT16_C( 32624), -INT16_C(  7902),  INT16_C( 21032), -INT16_C( 13405),  INT16_C( 15803) },
      {  INT16_C(    23),  INT16_C(    16),  INT16_C(   154),  INT16_C(   180),  INT16_C(   248),  INT16_C(   125),  INT16_C(   249),  INT16_C(     3),
         INT16_C(   209),  INT16_C(   134),  INT16_C(    41),  INT16_C(    55),  INT16_C(    46),  INT16_C(   173),  INT16_C(    68),  INT16_C(   189),
         INT16_C(    51),  INT16_C(    64),  INT16_C(   132),  INT16_C(    97),  INT16_C(    44),  INT16_C(   157),  INT16_C(   131),  INT16_C(   177),
         INT16_C(    89),  INT16_C(   105),  INT16_C(    61),  INT16_C(   140),  INT16_C(    41),  INT16_C(   100),  INT16_C(    36),  INT16_C(   200) },
      { UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C( 89), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,
        UINT8_C( 23), UINT8_C( 16), UINT8_C(154), UINT8_C(180), UINT8_C(248), UINT8_C(125), UINT8_C(249), UINT8_C(  3),
        UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,
        UINT8_C(209), UINT8_C(134), UINT8_C( 41), UINT8_C( 55), UINT8_C( 46), UINT8_C(173), UINT8_C( 68), UINT8_C(189),
        UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,
        UINT8_C( 51), UINT8_C( 64), UINT8_C(132), UINT8_C( 97), UINT8_C( 44), UINT8_C(157), UINT8_C(131), UINT8_C(177),
        UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,
        UINT8_C( 89), UINT8_C(105), UINT8_C( 61), UINT8_C(140), UINT8_C( 41), UINT8_C(100), UINT8_C( 36), UINT8_C(200) } },
    { { -INT16_C(  4009),  INT16_C(  9225), -INT16_C(   652), -INT16_C(  3963),  INT16_C( 25385),  INT16_C( 20109),  INT16_C( 12006),  INT16_C( 15103),
         INT16_C( 14216),  INT16_C(  2724),  INT16_C( 17524), -INT16_C(  8041), -INT16_C( 12178), -INT16_C(  9404),  INT16_C( 26356),  INT16_C( 19364),
        -INT16_C( 21162), -INT16_C( 13713), -INT16_C(  2902), -INT16_C( 11078),  INT16_C( 18519),  INT16_C( 15650),  INT16_C(  8822), -INT16_C(   392),
         INT16_C(  7257), -INT16_C( 13047), -INT16_C( 24480), -INT16_C( 12627), -INT16_C(  3472),  INT16_C( 26026),  INT16_C( 20056), -INT16_C( 20560) },
      {  INT16_C(    32),  INT16_C(   165),  INT16_C(    52),  INT16_C(   108),  INT16_C(   156),  INT16_C(   242),  INT16_C(    33),  INT16_C(    23),
         INT16_C(   250),  INT16_C(   158),  INT16_C(   146),  INT16_C(    10),  INT16_C(    22),  INT16_C(   220),  INT16_C(    32),  INT16_C(    95),
         INT16_C(     5),  INT16_C(    84),  INT16_C(   126),  INT16_C(   181),  INT16_C(   106),  INT16_C(   216),  INT16_C(   152),  INT16_C(   201),
         INT16_C(   212),  INT16_C(    44),  INT16_C(   211),  INT16_C(   234),  INT16_C(   166),  INT16_C(    78),  INT16_C(    82),  INT16_C(     6) },
      { UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,
        UINT8_C( 32), UINT8_C(165), UINT8_C( 52), UINT8_C(108), UINT8_C(156), UINT8_C(242), UINT8_C( 33), UINT8_C( 23),
           UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,
        UINT8_C(250), UINT8_C(158), UINT8_C(146), UINT8_C( 10), UINT8_C( 22), UINT8_C(220), UINT8_C( 32), UINT8_C( 95),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C(  5), UINT8_C( 84), UINT8_C(126), UINT8_C(181), UINT8_C(106), UINT8_C(216), UINT8_C(152), UINT8_C(201),
           UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C(212), UINT8_C( 44), UINT8_C(211), UINT8_C(234), UINT8_C(166), UINT8_C( 78), UINT8_C( 82), UINT8_C(  6) } },
    { { -INT16_C( 19625), -INT16_C( 28581),  INT16_C(  6961),  INT16_C( 19525), -INT16_C(  4987),  INT16_C(  4388),  INT16_C(  5253),  INT16_C(  6106),
         INT16_C( 16872),  INT16_C( 20036),  INT16_C( 31508), -INT16_C(   456), -INT16_C(   479), -INT16_C(  6067), -INT16_C(  1200), -INT16_C( 22546),
         INT16_C( 18862), -INT16_C(  8393),  INT16_C( 31845), -INT16_C(  5589),  INT16_C( 20585), -INT16_C(  4357), -INT16_C( 10908),  INT16_C( 19461),
         INT16_C( 18710),  INT16_C( 11162), -INT16_C( 11580), -INT16_C(  6615),  INT16_C( 30416),  INT16_C(  8654), -INT16_C( 17295),  INT16_C(  8136) },
      {  INT16_C(     0),  INT16_C(   107),  INT16_C(    42),  INT16_C(   229),  INT16_C(    81),  INT16_C(   222),  INT16_C(   217),  INT16_C(    61),
         INT16_C(   196),  INT16_C(   231),  INT16_C(   145),  INT16_C(   103),  INT16_C(   155),  INT16_C(   121),  INT16_C(    80),  INT16_C(    93),
         INT16_C(   152),  INT16_C(   205),  INT16_C(    30),  INT16_C(    61),  INT16_C(   134),  INT16_C(   149),  INT16_C(    70),  INT16_C(   129),
         INT16_C(    58),  INT16_C(   161),  INT16_C(    53),  INT16_C(   212),  INT16_C(   144),  INT16_C(    40),  INT16_C(   230),  INT16_C(    49) },
      { UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,
        UINT8_C(  0), UINT8_C(107), UINT8_C( 42), UINT8_C(229), UINT8_C( 81), UINT8_C(222), UINT8_C(217), UINT8_C( 61),
           UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(196), UINT8_C(231), UINT8_C(145), UINT8_C(103), UINT8_C(155), UINT8_C(121), UINT8_C( 80), UINT8_C( 93),
           UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,
        UINT8_C(152), UINT8_C(205), UINT8_C( 30), UINT8_C( 61), UINT8_C(134), UINT8_C(149), UINT8_C( 70), UINT8_C(129),
           UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),    UINT8_MAX,
        UINT8_C( 58), UINT8_C(161), UINT8_C( 53), UINT8_C(212), UINT8_C(144), UINT8_C( 40), UINT8_C(230), UINT8_C( 49) } },
    { {  INT16_C( 20094),  INT16_C( 16894), -INT16_C( 20372), -INT16_C(  9346), -INT16_C( 26314), -INT16_C( 27280),  INT16_C( 17375), -INT16_C(  5609),
         INT16_C( 32637),  INT16_C( 18827), -INT16_C( 27723), -INT16_C( 31459),  INT16_C( 27427),  INT16_C(   941),  INT16_C( 13137), -INT16_C( 12236),
         INT16_C( 12929), -INT16_C(  4847), -INT16_C( 28701),  INT16_C(  6600),  INT16_C( 14376),  INT16_C(  2223), -INT16_C( 14725), -INT16_C(  1550),
         INT16_C( 32069), -INT16_C(  1470),  INT16_C( 24592),  INT16_C( 13184),  INT16_C( 11723),  INT16_C(  7222),  INT16_C( 27488), -INT16_C(  7700) },
      {  INT16_C(   253),  INT16_C(   128),  INT16_C(   150),  INT16_C(   181),  INT16_C(    73),  INT16_C(    74),  INT16_C(   175),  INT16_C(    84),
         INT16_C(   134),  INT16_C(    60),  INT16_C(   207),  INT16_C(   177),  INT16_C(   165),  INT16_C(    93),  INT16_C(   186),  INT16_C(   174),
         INT16_C(    13),  INT16_C(    68),  INT16_C(   200),  INT16_C(   114),  INT16_C(   182),  INT16_C(    32),  INT16_C(     0),  INT16_C(   145),
         INT16_C(   196),  INT16_C(   108),  INT16_C(    60),  INT16_C(   143),  INT16_C(   235),  INT16_C(   242),  INT16_C(    43),  INT16_C(    92) },
      {    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),    UINT8_MAX, UINT8_C(  0),
        UINT8_C(253), UINT8_C(128), UINT8_C(150), UINT8_C(181), UINT8_C( 73), UINT8_C( 74), UINT8_C(175), UINT8_C( 84),
           UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C(134), UINT8_C( 60), UINT8_C(207), UINT8_C(177), UINT8_C(165), UINT8_C( 93), UINT8_C(186), UINT8_C(174),
           UINT8_MAX, UINT8_C(  0), UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 13), UINT8_C( 68), UINT8_C(200), UINT8_C(114), UINT8_C(182), UINT8_C( 32), UINT8_C(  0), UINT8_C(145),
           UINT8_MAX, UINT8_C(  0),    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,    UINT8_MAX,    UINT8_MAX, UINT8_C(  0),
        UINT8_C(196), UINT8_C(108), UINT8_C( 60), UINT8_C(143), UINT8_C(235), UINT8_C(242), UINT8_C( 43), UINT8_C( 92) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_packus_epi16(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_packus_epi16");
    easysimd_test_x86_assert_equal_u8x64(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_packus_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const uint16_t r[32];
  } test_vec[] = {
    { {  INT32_C(       32838),  INT32_C(         707),  INT32_C(       18249),  INT32_C(       43411),  INT32_C(       33031),  INT32_C(       48266),  INT32_C(       46389),  INT32_C(       30506),
         INT32_C(       19447),  INT32_C(       16717),  INT32_C(        9608),  INT32_C(       32719),  INT32_C(       16128),  INT32_C(         507),  INT32_C(        9398),  INT32_C(       24219) },
      { -INT32_C(   374762927), -INT32_C(   768936372),  INT32_C(  1090040461), -INT32_C(   926955570),  INT32_C(  1560788893), -INT32_C(  1621228982), -INT32_C(  1144842958),  INT32_C(  1192845046),
         INT32_C(  1009828848),  INT32_C(  1175411385), -INT32_C(   611907827),  INT32_C(  1805862606),  INT32_C(  1355393542), -INT32_C(   554752084),  INT32_C(   848933692),  INT32_C(    41595665) },
      { UINT16_C(32838), UINT16_C(  707), UINT16_C(18249), UINT16_C(43411), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),
        UINT16_C(33031), UINT16_C(48266), UINT16_C(46389), UINT16_C(30506),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,
        UINT16_C(19447), UINT16_C(16717), UINT16_C( 9608), UINT16_C(32719),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,
        UINT16_C(16128), UINT16_C(  507), UINT16_C( 9398), UINT16_C(24219),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX } },
    { {  INT32_C(       12094),  INT32_C(        4726),  INT32_C(        8941),  INT32_C(       18830),  INT32_C(       59545),  INT32_C(       48070),  INT32_C(       19949),  INT32_C(       35151),
         INT32_C(        6072),  INT32_C(       12329),  INT32_C(       28498),  INT32_C(       58296),  INT32_C(       46795),  INT32_C(        6001),  INT32_C(        1124),  INT32_C(       55437) },
      {  INT32_C(   502220354), -INT32_C(  1605560204), -INT32_C(   703619026), -INT32_C(  1195784320), -INT32_C(   194083815),  INT32_C(   118218517),  INT32_C(    51081277),  INT32_C(  1725667620),
         INT32_C(  1401146079),  INT32_C(   301191650), -INT32_C(   236518799), -INT32_C(   475422518),  INT32_C(   970463012),  INT32_C(   876667894),  INT32_C(  2000112723), -INT32_C(   992144411) },
      { UINT16_C(12094), UINT16_C( 4726), UINT16_C( 8941), UINT16_C(18830),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(59545), UINT16_C(48070), UINT16_C(19949), UINT16_C(35151), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,
        UINT16_C( 6072), UINT16_C(12329), UINT16_C(28498), UINT16_C(58296),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),
        UINT16_C(46795), UINT16_C( 6001), UINT16_C( 1124), UINT16_C(55437),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0) } },
    { {  INT32_C(       49175),  INT32_C(       41937),  INT32_C(       55188),  INT32_C(       31931),  INT32_C(       19637),  INT32_C(       51840),  INT32_C(       10049),  INT32_C(       43243),
         INT32_C(       45672),  INT32_C(        6997),  INT32_C(       18930),  INT32_C(       32197),  INT32_C(       47049),  INT32_C(       45697),  INT32_C(       52185),  INT32_C(       24947) },
      { -INT32_C(   736896057),  INT32_C(    99575828),  INT32_C(  2035212882), -INT32_C(   789179505), -INT32_C(    24658035),  INT32_C(   162531336), -INT32_C(  1395356982),  INT32_C(   353191758),
         INT32_C(   921313570), -INT32_C(   616834679),  INT32_C(  1263897019),  INT32_C(   689654684),  INT32_C(   321364491),  INT32_C(  1948047530), -INT32_C(  1340018590),  INT32_C(  1506160183) },
      { UINT16_C(49175), UINT16_C(41937), UINT16_C(55188), UINT16_C(31931), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),
        UINT16_C(19637), UINT16_C(51840), UINT16_C(10049), UINT16_C(43243), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,
        UINT16_C(45672), UINT16_C( 6997), UINT16_C(18930), UINT16_C(32197),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,
        UINT16_C(47049), UINT16_C(45697), UINT16_C(52185), UINT16_C(24947),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),      UINT16_MAX } },
    { {  INT32_C(       55439),  INT32_C(       17844),  INT32_C(       61328),  INT32_C(       24345),  INT32_C(       63347),  INT32_C(       31339),  INT32_C(       46891),  INT32_C(        2321),
         INT32_C(       10977),  INT32_C(       48751),  INT32_C(       62382),  INT32_C(       63314),  INT32_C(        8430),  INT32_C(       54682),  INT32_C(       41100),  INT32_C(       22441) },
      { -INT32_C(  1451062722), -INT32_C(  1100484320), -INT32_C(  1682893327), -INT32_C(   460127012),  INT32_C(   503611849), -INT32_C(  1040998693),  INT32_C(   442597476),  INT32_C(  1534200349),
        -INT32_C(  1257966443), -INT32_C(   697078555),  INT32_C(  1584539009), -INT32_C(   230554327),  INT32_C(  1645299334),  INT32_C(  1210254564), -INT32_C(  1570536060),  INT32_C(   620615055) },
      { UINT16_C(55439), UINT16_C(17844), UINT16_C(61328), UINT16_C(24345), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(63347), UINT16_C(31339), UINT16_C(46891), UINT16_C( 2321),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,
        UINT16_C(10977), UINT16_C(48751), UINT16_C(62382), UINT16_C(63314), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),
        UINT16_C( 8430), UINT16_C(54682), UINT16_C(41100), UINT16_C(22441),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),      UINT16_MAX } },
    { {  INT32_C(       44761),  INT32_C(       61317),  INT32_C(       39757),  INT32_C(       33421),  INT32_C(       47844),  INT32_C(        9986),  INT32_C(        7369),  INT32_C(         833),
         INT32_C(       14258),  INT32_C(       55590),  INT32_C(       10868),  INT32_C(       55724),  INT32_C(       17299),  INT32_C(        9835),  INT32_C(       13634),  INT32_C(       50233) },
      {  INT32_C(   100395934),  INT32_C(  1356800546), -INT32_C(  1720036458), -INT32_C(   160291243),  INT32_C(  1345914295), -INT32_C(  1770609509), -INT32_C(   724846119), -INT32_C(   627506116),
         INT32_C(   299930863),  INT32_C(  1281474486),  INT32_C(  1759959826), -INT32_C(  1184999422), -INT32_C(   116746402),  INT32_C(   361726012),  INT32_C(  1995004473),  INT32_C(  1313899103) },
      { UINT16_C(44761), UINT16_C(61317), UINT16_C(39757), UINT16_C(33421),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),
        UINT16_C(47844), UINT16_C( 9986), UINT16_C( 7369), UINT16_C(  833),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(14258), UINT16_C(55590), UINT16_C(10868), UINT16_C(55724),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX, UINT16_C(    0),
        UINT16_C(17299), UINT16_C( 9835), UINT16_C(13634), UINT16_C(50233), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX } },
    { {  INT32_C(       52576),  INT32_C(         538),  INT32_C(       40810),  INT32_C(       46680),  INT32_C(       39855),  INT32_C(        7344),  INT32_C(       63634),  INT32_C(       13126),
         INT32_C(         769),  INT32_C(        1285),  INT32_C(       29604),  INT32_C(       38442),  INT32_C(       16946),  INT32_C(       45406),  INT32_C(       39337),  INT32_C(       59340) },
      { -INT32_C(    18166378),  INT32_C(    50589672), -INT32_C(  1787320482),  INT32_C(    36479395), -INT32_C(  1841013126), -INT32_C(  1119640768),  INT32_C(  1750527124),  INT32_C(  1917788892),
        -INT32_C(   663733520), -INT32_C(  1998818519), -INT32_C(  1122151654),  INT32_C(  1858095604), -INT32_C(   402586457),  INT32_C(  1000686759),  INT32_C(   228850481),  INT32_C(   226489117) },
      { UINT16_C(52576), UINT16_C(  538), UINT16_C(40810), UINT16_C(46680), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,
        UINT16_C(39855), UINT16_C( 7344), UINT16_C(63634), UINT16_C(13126), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,
        UINT16_C(  769), UINT16_C( 1285), UINT16_C(29604), UINT16_C(38442), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX,
        UINT16_C(16946), UINT16_C(45406), UINT16_C(39337), UINT16_C(59340), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX } },
    { {  INT32_C(       22246),  INT32_C(       31966),  INT32_C(        2361),  INT32_C(       60791),  INT32_C(       42453),  INT32_C(       61153),  INT32_C(       37627),  INT32_C(       42144),
         INT32_C(       52219),  INT32_C(       23879),  INT32_C(        7014),  INT32_C(       30728),  INT32_C(        4893),  INT32_C(       52225),  INT32_C(       64094),  INT32_C(       57247) },
      { -INT32_C(   861234556),  INT32_C(  1227485555), -INT32_C(   345731215), -INT32_C(  1016894355), -INT32_C(  1596554935),  INT32_C(    40687487),  INT32_C(  1241369299),  INT32_C(  1294507209),
        -INT32_C(  1457860042),  INT32_C(   888292291),  INT32_C(  1075861203),  INT32_C(   184779714), -INT32_C(  2069112572), -INT32_C(  2088364112), -INT32_C(  1412660254),  INT32_C(  1442378783) },
      { UINT16_C(22246), UINT16_C(31966), UINT16_C( 2361), UINT16_C(60791), UINT16_C(    0),      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),
        UINT16_C(42453), UINT16_C(61153), UINT16_C(37627), UINT16_C(42144), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,
        UINT16_C(52219), UINT16_C(23879), UINT16_C( 7014), UINT16_C(30728), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX,
        UINT16_C( 4893), UINT16_C(52225), UINT16_C(64094), UINT16_C(57247), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),      UINT16_MAX } },
    { {  INT32_C(       35327),  INT32_C(       10685),  INT32_C(        2665),  INT32_C(       25878),  INT32_C(       62953),  INT32_C(       47992),  INT32_C(        4966),  INT32_C(       65128),
         INT32_C(       51079),  INT32_C(       41456),  INT32_C(       33707),  INT32_C(        2792),  INT32_C(       23807),  INT32_C(       13591),  INT32_C(       62280),  INT32_C(       19697) },
      {  INT32_C(  1897101336), -INT32_C(   569244740),  INT32_C(   560053852),  INT32_C(    36391551),  INT32_C(  1583229468),  INT32_C(  1553167777), -INT32_C(   833626894), -INT32_C(  1525006195),
         INT32_C(  1964453560), -INT32_C(  1907152591),  INT32_C(  1739568615),  INT32_C(   459922431), -INT32_C(  1485191163),  INT32_C(   805506109),  INT32_C(  1979601896),  INT32_C(  1276844179) },
      { UINT16_C(35327), UINT16_C(10685), UINT16_C( 2665), UINT16_C(25878),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,
        UINT16_C(62953), UINT16_C(47992), UINT16_C( 4966), UINT16_C(65128),      UINT16_MAX,      UINT16_MAX, UINT16_C(    0), UINT16_C(    0),
        UINT16_C(51079), UINT16_C(41456), UINT16_C(33707), UINT16_C( 2792),      UINT16_MAX, UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,
        UINT16_C(23807), UINT16_C(13591), UINT16_C(62280), UINT16_C(19697), UINT16_C(    0),      UINT16_MAX,      UINT16_MAX,      UINT16_MAX } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_packus_epi32(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_packus_epi32");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_packus_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_packus_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_packus_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_packus_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_packus_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_packus_epi32)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
