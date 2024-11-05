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

#define EASYSIMD_TEST_X86_AVX512_INSN mulhi

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/mulhi.h>

static int
test_easysimd_mm_mask_mulhi_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C( 11833),  INT16_C(  8312), -INT16_C( 16666), -INT16_C( 18095), -INT16_C( 14691),  INT16_C(  3080),  INT16_C( 13341),  INT16_C( 21208) },
      UINT8_C(181),
      {  INT16_C( 22233), -INT16_C(  7090),  INT16_C( 24959),  INT16_C(  1103), -INT16_C(  2932), -INT16_C(  7392),  INT16_C( 20810),  INT16_C( 30748) },
      {  INT16_C( 15817), -INT16_C( 30881),  INT16_C(  6286),  INT16_C( 21540),  INT16_C( 12321),  INT16_C( 21874), -INT16_C( 15352), -INT16_C(  7669) },
      {  INT16_C(  5365),  INT16_C(  8312),  INT16_C(  2393), -INT16_C( 18095), -INT16_C(   552), -INT16_C(  2468),  INT16_C( 13341), -INT16_C(  3599) } },
    { {  INT16_C( 22810), -INT16_C( 26170),  INT16_C(  5562),  INT16_C( 18077), -INT16_C( 17143),  INT16_C( 21290),  INT16_C( 17934), -INT16_C( 10292) },
      UINT8_C(131),
      {  INT16_C( 24363),  INT16_C( 17170),  INT16_C( 26243), -INT16_C( 19356), -INT16_C( 17704), -INT16_C( 25412), -INT16_C( 24891),  INT16_C(  7862) },
      {  INT16_C( 20325),  INT16_C( 31449),  INT16_C(  8172), -INT16_C( 21884), -INT16_C( 10423), -INT16_C( 28488), -INT16_C( 28509), -INT16_C( 12781) },
      {  INT16_C(  7555),  INT16_C(  8239),  INT16_C(  5562),  INT16_C( 18077), -INT16_C( 17143),  INT16_C( 21290),  INT16_C( 17934), -INT16_C(  1534) } },
    { {  INT16_C(  9711),  INT16_C( 29202),  INT16_C( 30348),  INT16_C( 25638), -INT16_C(  7376), -INT16_C(  2815), -INT16_C( 18559), -INT16_C(  6636) },
      UINT8_C(  7),
      {  INT16_C( 25069),  INT16_C(  3315), -INT16_C( 25115), -INT16_C( 17322), -INT16_C(  6570), -INT16_C(  6560),  INT16_C( 12025),  INT16_C(  8149) },
      {  INT16_C( 18240), -INT16_C( 18517),  INT16_C(  3950),  INT16_C( 20967), -INT16_C(  8944), -INT16_C( 14126), -INT16_C( 17935), -INT16_C(  8497) },
      {  INT16_C(  6977), -INT16_C(   937), -INT16_C(  1514),  INT16_C( 25638), -INT16_C(  7376), -INT16_C(  2815), -INT16_C( 18559), -INT16_C(  6636) } },
    { { -INT16_C( 15846), -INT16_C(    22),  INT16_C( 16480), -INT16_C( 18757),  INT16_C(  6950),  INT16_C(  8348),  INT16_C( 29002), -INT16_C( 30145) },
      UINT8_C(184),
      {  INT16_C( 16874), -INT16_C(  1754),  INT16_C( 30505),  INT16_C(  1546), -INT16_C( 11702),  INT16_C(  1015), -INT16_C( 10847),  INT16_C( 25373) },
      {  INT16_C(  7359),  INT16_C(   195),  INT16_C( 31191), -INT16_C(  3290),  INT16_C( 17941), -INT16_C( 31171), -INT16_C( 14459),  INT16_C( 28479) },
      { -INT16_C( 15846), -INT16_C(    22),  INT16_C( 16480), -INT16_C(    78), -INT16_C(  3204), -INT16_C(   483),  INT16_C( 29002),  INT16_C( 11025) } },
    { {  INT16_C( 25865),  INT16_C( 12905),  INT16_C( 29661),  INT16_C( 10040),  INT16_C( 12101), -INT16_C(  6614),  INT16_C( 18180), -INT16_C( 15543) },
      UINT8_C( 99),
      { -INT16_C( 15603), -INT16_C( 31174),  INT16_C( 11754),  INT16_C( 12444),  INT16_C(  8810),  INT16_C( 12982),  INT16_C(  9569), -INT16_C( 14533) },
      {  INT16_C( 28046),  INT16_C(   420), -INT16_C( 13403), -INT16_C( 11194),  INT16_C( 11509),  INT16_C( 15576), -INT16_C( 25738), -INT16_C( 31841) },
      { -INT16_C(  6678), -INT16_C(   200),  INT16_C( 29661),  INT16_C( 10040),  INT16_C( 12101),  INT16_C(  3085), -INT16_C(  3759), -INT16_C( 15543) } },
    { { -INT16_C(  9889),  INT16_C( 18697), -INT16_C( 23289),  INT16_C( 29049),  INT16_C( 12232),  INT16_C( 10659), -INT16_C(  8619), -INT16_C(  7184) },
      UINT8_C( 75),
      { -INT16_C(  6764),  INT16_C( 24560), -INT16_C( 15317),  INT16_C( 22612), -INT16_C( 28516),  INT16_C( 14542),  INT16_C( 20783),  INT16_C(  2455) },
      { -INT16_C(  8102),  INT16_C(    16), -INT16_C( 32423), -INT16_C( 30264), -INT16_C(  3803),  INT16_C(   990), -INT16_C( 15902),  INT16_C( 30287) },
      {  INT16_C(   836),  INT16_C(     5), -INT16_C( 23289), -INT16_C( 10443),  INT16_C( 12232),  INT16_C( 10659), -INT16_C(  5043), -INT16_C(  7184) } },
    { {  INT16_C( 16294), -INT16_C( 11562),  INT16_C( 10756), -INT16_C( 24534), -INT16_C(  1861), -INT16_C(  5416),  INT16_C( 28489), -INT16_C( 23565) },
      UINT8_C( 79),
      { -INT16_C( 23805), -INT16_C( 31319),  INT16_C( 12907),  INT16_C( 23978), -INT16_C( 21232), -INT16_C( 11969), -INT16_C( 18948),  INT16_C( 15480) },
      {  INT16_C( 19083), -INT16_C( 18880), -INT16_C(  8076),  INT16_C( 27761),  INT16_C( 23481),  INT16_C( 10421),  INT16_C( 22607),  INT16_C( 21112) },
      { -INT16_C(  6932),  INT16_C(  9022), -INT16_C(  1591),  INT16_C( 10157), -INT16_C(  1861), -INT16_C(  5416), -INT16_C(  6537), -INT16_C( 23565) } },
    { {  INT16_C(  8700),  INT16_C( 26583), -INT16_C( 32429),  INT16_C( 25540),  INT16_C(   815),  INT16_C( 11060), -INT16_C( 21319),  INT16_C( 17511) },
      UINT8_C(246),
      { -INT16_C(  1369), -INT16_C( 30614), -INT16_C( 10645), -INT16_C( 14527),  INT16_C( 27019), -INT16_C(  7146),  INT16_C( 26849),  INT16_C(   736) },
      {  INT16_C( 18240), -INT16_C( 16043), -INT16_C( 18420),  INT16_C(  4080),  INT16_C(  7405), -INT16_C( 26168),  INT16_C(  3459),  INT16_C( 11152) },
      {  INT16_C(  8700),  INT16_C(  7494),  INT16_C(  2991),  INT16_C( 25540),  INT16_C(  3052),  INT16_C(  2853),  INT16_C(  1417),  INT16_C(   125) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mulhi_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mulhi_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_mask_mulhi_epi16(src, k, a, b);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_mulhi_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(249),
      {  INT16_C( 24819),  INT16_C( 25715),  INT16_C( 21838),  INT16_C( 14571),  INT16_C( 12793), -INT16_C(  9304),  INT16_C( 18124), -INT16_C( 10778) },
      {  INT16_C( 10820),  INT16_C(  6048),  INT16_C( 24627), -INT16_C( 17345), -INT16_C( 15698),  INT16_C(  9075), -INT16_C( 11709),  INT16_C( 14109) },
      {  INT16_C(  4097),  INT16_C(     0),  INT16_C(     0), -INT16_C(  3857), -INT16_C(  3065), -INT16_C(  1289), -INT16_C(  3239), -INT16_C(  2321) } },
    { UINT8_C( 50),
      { -INT16_C( 25712), -INT16_C(  6783), -INT16_C( 18042), -INT16_C( 18466), -INT16_C( 17823), -INT16_C( 22397),  INT16_C( 22688), -INT16_C( 13332) },
      {  INT16_C(  1016),  INT16_C( 22782), -INT16_C( 17854),  INT16_C(  1031),  INT16_C( 10798),  INT16_C(    72),  INT16_C( 32583), -INT16_C( 10445) },
      {  INT16_C(     0), -INT16_C(  2358),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2937), -INT16_C(    25),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 26),
      { -INT16_C( 16972),  INT16_C( 28065),  INT16_C( 22683),  INT16_C( 21966),  INT16_C( 30428),  INT16_C( 13558), -INT16_C( 16030),  INT16_C( 25901) },
      { -INT16_C( 31297),  INT16_C( 31400), -INT16_C( 21364), -INT16_C( 18520), -INT16_C( 22284),  INT16_C( 29694), -INT16_C( 10533), -INT16_C( 28786) },
      {  INT16_C(     0),  INT16_C( 13446),  INT16_C(     0), -INT16_C(  6208), -INT16_C( 10347),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(147),
      { -INT16_C(   977), -INT16_C( 30930), -INT16_C( 31541),  INT16_C( 16739), -INT16_C( 26502),  INT16_C( 15268),  INT16_C(  2501),  INT16_C( 19194) },
      {  INT16_C( 29873),  INT16_C( 24279), -INT16_C( 29156), -INT16_C( 15022), -INT16_C( 14708),  INT16_C( 25248),  INT16_C( 12372), -INT16_C( 31755) },
      { -INT16_C(   446), -INT16_C( 11459),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5947),  INT16_C(     0),  INT16_C(     0), -INT16_C(  9301) } },
    { UINT8_C( 44),
      {  INT16_C(  2596), -INT16_C( 22281),  INT16_C( 14702),  INT16_C(  1570),  INT16_C( 24029), -INT16_C(  6453),  INT16_C(  5463), -INT16_C( 13160) },
      { -INT16_C(  2324),  INT16_C( 31464), -INT16_C( 21176),  INT16_C(  3591),  INT16_C( 26958),  INT16_C( 32354), -INT16_C(  6817), -INT16_C( 31830) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C(  4751),  INT16_C(    86),  INT16_C(     0), -INT16_C(  3186),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(240),
      {  INT16_C( 11170), -INT16_C(  9378),  INT16_C( 25677), -INT16_C( 21832), -INT16_C( 25041),  INT16_C( 17409), -INT16_C( 13002),  INT16_C( 11313) },
      { -INT16_C( 21578),  INT16_C( 25461), -INT16_C( 31822),  INT16_C(  7345),  INT16_C( 12262), -INT16_C( 13445), -INT16_C(   294),  INT16_C( 31931) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  4686), -INT16_C(  3572),  INT16_C(    58),  INT16_C(  5512) } },
    { UINT8_C( 41),
      {  INT16_C( 22297),  INT16_C( 32118),  INT16_C(  8207), -INT16_C( 21076), -INT16_C(  3807), -INT16_C(  4124),  INT16_C(  4130), -INT16_C( 12891) },
      {  INT16_C(  2181),  INT16_C(  2432), -INT16_C( 25414), -INT16_C(  5649), -INT16_C( 17897),  INT16_C(  5571),  INT16_C( 16246), -INT16_C( 28866) },
      {  INT16_C(   742),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1816),  INT16_C(     0), -INT16_C(   351),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(150),
      {  INT16_C(  3508), -INT16_C( 11099),  INT16_C( 21433), -INT16_C( 21771), -INT16_C(  7113),  INT16_C( 18380), -INT16_C( 25975), -INT16_C( 27955) },
      { -INT16_C( 10726), -INT16_C( 18868),  INT16_C( 13765),  INT16_C( 32717), -INT16_C(  7431),  INT16_C( 14581), -INT16_C( 31456), -INT16_C( 11057) },
      {  INT16_C(     0),  INT16_C(  3195),  INT16_C(  4501),  INT16_C(     0),  INT16_C(   806),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4716) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mulhi_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mulhi_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_mulhi_epi16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_mulhi_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t src[8];
    const uint8_t k;
    const uint16_t a[8];
    const uint16_t b[8];
    const uint16_t r[8];
  } test_vec[] = {
    { { UINT16_C(17637), UINT16_C(30901), UINT16_C(63598), UINT16_C(39413), UINT16_C(61985), UINT16_C( 9215), UINT16_C( 1289), UINT16_C(51133) },
      UINT8_C(246),
      { UINT16_C(49248), UINT16_C(50463), UINT16_C(12011), UINT16_C(33185), UINT16_C(18979), UINT16_C(39643), UINT16_C(64233), UINT16_C(11903) },
      { UINT16_C(63407), UINT16_C(42908), UINT16_C(14060), UINT16_C(57032), UINT16_C(60213), UINT16_C(15336), UINT16_C(44969), UINT16_C( 2353) },
      { UINT16_C(17637), UINT16_C(33039), UINT16_C( 2576), UINT16_C(39413), UINT16_C(17437), UINT16_C( 9276), UINT16_C(44074), UINT16_C(  427) } },
    { { UINT16_C(20592), UINT16_C(23502), UINT16_C(28543), UINT16_C(41692), UINT16_C(47290), UINT16_C(41788), UINT16_C(48050), UINT16_C(25041) },
      UINT8_C(178),
      { UINT16_C( 2414), UINT16_C(42142), UINT16_C(32209), UINT16_C(48601), UINT16_C( 5221), UINT16_C( 5222), UINT16_C(28486), UINT16_C(38532) },
      { UINT16_C(57149), UINT16_C(44053), UINT16_C(47036), UINT16_C(29798), UINT16_C( 2803), UINT16_C(44582), UINT16_C(34779), UINT16_C(18784) },
      { UINT16_C(20592), UINT16_C(28327), UINT16_C(28543), UINT16_C(41692), UINT16_C(  223), UINT16_C( 3552), UINT16_C(48050), UINT16_C(11044) } },
    { { UINT16_C(65424), UINT16_C(25325), UINT16_C(51068), UINT16_C(57631), UINT16_C(34267), UINT16_C( 8693), UINT16_C(31476), UINT16_C(12728) },
      UINT8_C( 89),
      { UINT16_C(56781), UINT16_C(34069), UINT16_C(35140), UINT16_C(20088), UINT16_C(10159), UINT16_C(14121), UINT16_C(29575), UINT16_C(34503) },
      { UINT16_C(10592), UINT16_C( 9986), UINT16_C(58184), UINT16_C(52483), UINT16_C( 9433), UINT16_C(21441), UINT16_C(62172), UINT16_C(43692) },
      { UINT16_C( 9177), UINT16_C(25325), UINT16_C(51068), UINT16_C(16087), UINT16_C( 1462), UINT16_C( 8693), UINT16_C(28056), UINT16_C(12728) } },
    { { UINT16_C(49872), UINT16_C( 5167), UINT16_C(42827), UINT16_C(64354), UINT16_C(35790), UINT16_C(22066), UINT16_C(63998), UINT16_C(24540) },
      UINT8_C( 35),
      { UINT16_C(34527), UINT16_C(49771), UINT16_C(14729), UINT16_C(44699), UINT16_C(61178), UINT16_C(60810), UINT16_C(13467), UINT16_C(23997) },
      { UINT16_C(53603), UINT16_C( 2984), UINT16_C(41779), UINT16_C(48857), UINT16_C(12245), UINT16_C(53181), UINT16_C( 7180), UINT16_C(60402) },
      { UINT16_C(28240), UINT16_C( 2266), UINT16_C(42827), UINT16_C(64354), UINT16_C(35790), UINT16_C(49345), UINT16_C(63998), UINT16_C(24540) } },
    { { UINT16_C(23970), UINT16_C(11437), UINT16_C(18838), UINT16_C(37338), UINT16_C(25655), UINT16_C(53886), UINT16_C(15257), UINT16_C(64559) },
      UINT8_C( 12),
      { UINT16_C( 2008), UINT16_C(31551), UINT16_C(64993), UINT16_C( 4177), UINT16_C( 8378), UINT16_C(54812), UINT16_C( 1810), UINT16_C(28537) },
      { UINT16_C(42421), UINT16_C(65030), UINT16_C(38783), UINT16_C(58165), UINT16_C( 2069), UINT16_C(20604), UINT16_C(31031), UINT16_C( 3932) },
      { UINT16_C(23970), UINT16_C(11437), UINT16_C(38461), UINT16_C( 3707), UINT16_C(25655), UINT16_C(53886), UINT16_C(15257), UINT16_C(64559) } },
    { { UINT16_C(39808), UINT16_C(24971), UINT16_C(56472), UINT16_C(21362), UINT16_C(36604), UINT16_C( 3625), UINT16_C(41622), UINT16_C(19325) },
      UINT8_C( 71),
      { UINT16_C(18819), UINT16_C( 6854), UINT16_C(43646), UINT16_C(34351), UINT16_C(32550), UINT16_C(40894), UINT16_C(52699), UINT16_C(30240) },
      { UINT16_C(33112), UINT16_C(13327), UINT16_C(25331), UINT16_C(33328), UINT16_C(16011), UINT16_C(11800), UINT16_C(25532), UINT16_C(16245) },
      { UINT16_C( 9508), UINT16_C( 1393), UINT16_C(16870), UINT16_C(21362), UINT16_C(36604), UINT16_C( 3625), UINT16_C(20530), UINT16_C(19325) } },
    { { UINT16_C(15532), UINT16_C(10842), UINT16_C(35302), UINT16_C( 3249), UINT16_C(28425), UINT16_C(58540), UINT16_C(52284), UINT16_C(38235) },
      UINT8_C( 77),
      { UINT16_C(51562), UINT16_C(52289), UINT16_C(50170), UINT16_C(14423), UINT16_C(34267), UINT16_C(16116), UINT16_C(13563), UINT16_C(14314) },
      { UINT16_C( 5262), UINT16_C( 5917), UINT16_C(10693), UINT16_C(13344), UINT16_C( 1493), UINT16_C(41329), UINT16_C( 1632), UINT16_C(51951) },
      { UINT16_C( 4140), UINT16_C(10842), UINT16_C( 8185), UINT16_C( 2936), UINT16_C(28425), UINT16_C(58540), UINT16_C(  337), UINT16_C(38235) } },
    { { UINT16_C(12495), UINT16_C(51606), UINT16_C(60915), UINT16_C(52738), UINT16_C(63091), UINT16_C(28172), UINT16_C(63018), UINT16_C(47269) },
      UINT8_C( 10),
      { UINT16_C(53442), UINT16_C(60368), UINT16_C( 1264), UINT16_C(62913), UINT16_C(25205), UINT16_C(31573), UINT16_C( 8017), UINT16_C(33099) },
      { UINT16_C( 5301), UINT16_C(41844), UINT16_C(16918), UINT16_C( 3350), UINT16_C(33870), UINT16_C(17463), UINT16_C(61481), UINT16_C(60239) },
      { UINT16_C(12495), UINT16_C(38544), UINT16_C(60915), UINT16_C( 3215), UINT16_C(63091), UINT16_C(28172), UINT16_C(63018), UINT16_C(47269) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mulhi_epu16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mulhi_epu16");
    easysimd_test_x86_assert_equal_u16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__m128i r = easysimd_mm_mask_mulhi_epu16(src, k, a, b);

    easysimd_test_x86_write_u16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_mulhi_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint16_t a[8];
    const uint16_t b[8];
    const uint16_t r[8];
  } test_vec[] = {
    { UINT8_C(146),
      { UINT16_C(43124), UINT16_C(51019), UINT16_C(63133), UINT16_C(33534), UINT16_C(18114), UINT16_C(23563), UINT16_C(40211), UINT16_C(59766) },
      { UINT16_C(11497), UINT16_C( 8110), UINT16_C(11769), UINT16_C(56088), UINT16_C(20515), UINT16_C(43259), UINT16_C(53023), UINT16_C(37946) },
      { UINT16_C(    0), UINT16_C( 6313), UINT16_C(    0), UINT16_C(    0), UINT16_C( 5670), UINT16_C(    0), UINT16_C(    0), UINT16_C(34605) } },
    { UINT8_C(119),
      { UINT16_C(23429), UINT16_C(31509), UINT16_C(38746), UINT16_C(41022), UINT16_C(39586), UINT16_C(16563), UINT16_C(39953), UINT16_C(15657) },
      { UINT16_C(18506), UINT16_C(30519), UINT16_C( 4704), UINT16_C(45466), UINT16_C(16910), UINT16_C(56784), UINT16_C(25724), UINT16_C(  597) },
      { UINT16_C( 6615), UINT16_C(14673), UINT16_C( 2781), UINT16_C(    0), UINT16_C(10214), UINT16_C(14351), UINT16_C(15682), UINT16_C(    0) } },
    { UINT8_C(192),
      { UINT16_C(32106), UINT16_C(  282), UINT16_C(47803), UINT16_C(22179), UINT16_C(58221), UINT16_C( 2407), UINT16_C(41997), UINT16_C(21843) },
      { UINT16_C(51931), UINT16_C(61110), UINT16_C(26469), UINT16_C(43004), UINT16_C(55607), UINT16_C(39972), UINT16_C( 9774), UINT16_C(39004) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 6263), UINT16_C(12999) } },
    { UINT8_C(163),
      { UINT16_C(39286), UINT16_C(12383), UINT16_C(46397), UINT16_C( 8349), UINT16_C(42524), UINT16_C(49197), UINT16_C(33785), UINT16_C(50076) },
      { UINT16_C(35385), UINT16_C(41000), UINT16_C(53382), UINT16_C(24535), UINT16_C(29684), UINT16_C( 6798), UINT16_C( 9935), UINT16_C(17853) },
      { UINT16_C(21211), UINT16_C( 7746), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 5103), UINT16_C(    0), UINT16_C(13641) } },
    { UINT8_C(192),
      { UINT16_C(29980), UINT16_C(53757), UINT16_C( 7442), UINT16_C(47341), UINT16_C(44619), UINT16_C(52913), UINT16_C(30026), UINT16_C(54279) },
      { UINT16_C(42909), UINT16_C(27994), UINT16_C(47486), UINT16_C(62049), UINT16_C(31559), UINT16_C(28353), UINT16_C( 1849), UINT16_C(21806) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(  847), UINT16_C(18060) } },
    { UINT8_C(124),
      { UINT16_C(10027), UINT16_C(18575), UINT16_C(18196), UINT16_C(49811), UINT16_C(25081), UINT16_C(28172), UINT16_C(57448), UINT16_C( 3851) },
      { UINT16_C(31034), UINT16_C(62606), UINT16_C(32986), UINT16_C(22075), UINT16_C(43329), UINT16_C(18575), UINT16_C(58583), UINT16_C(  709) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C( 9158), UINT16_C(16778), UINT16_C(16582), UINT16_C( 7984), UINT16_C(51353), UINT16_C(    0) } },
    { UINT8_C( 11),
      { UINT16_C(19284), UINT16_C(39712), UINT16_C(58078), UINT16_C(16532), UINT16_C(  751), UINT16_C(53160), UINT16_C(47118), UINT16_C(34570) },
      { UINT16_C(65094), UINT16_C(50785), UINT16_C(46905), UINT16_C(58119), UINT16_C(20550), UINT16_C(11194), UINT16_C(48405), UINT16_C(26934) },
      { UINT16_C(19153), UINT16_C(30773), UINT16_C(    0), UINT16_C(14661), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C(  8),
      { UINT16_C( 1110), UINT16_C(14822), UINT16_C( 9881), UINT16_C(39720), UINT16_C(63439), UINT16_C(34729), UINT16_C(12289), UINT16_C(65485) },
      { UINT16_C(37778), UINT16_C(18745), UINT16_C( 7322), UINT16_C(60048), UINT16_C(48086), UINT16_C(37887), UINT16_C(26865), UINT16_C(18587) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(36393), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mulhi_epu16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mulhi_epu16");
    easysimd_test_x86_assert_equal_u16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__m128i r = easysimd_mm_maskz_mulhi_epu16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mulhi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t src[16];
    uint16_t k;
    int16_t a[16];
    int16_t b[16];
    int16_t r[16];
  } test_vec[] = {
    { {  INT16_C(  5860), -INT16_C( 14422),  INT16_C( 16938),  INT16_C( 27734),  INT16_C( 27901), -INT16_C( 27579), -INT16_C( 26091),  INT16_C( 31702),
        -INT16_C(    74),  INT16_C(  1885), -INT16_C(  7628),  INT16_C( 22248), -INT16_C(   837), -INT16_C( 16859), -INT16_C(  7960), -INT16_C( 13301) },
      UINT16_C(46582),
      {  INT16_C(  8596), -INT16_C(  5385), -INT16_C(  2675), -INT16_C( 11689),  INT16_C( 27785),  INT16_C( 24428),  INT16_C(  8935),  INT16_C( 17758),
        -INT16_C( 27863),  INT16_C(  4647), -INT16_C(  7447),  INT16_C(  3854), -INT16_C(  2399), -INT16_C( 21265), -INT16_C(  6717),  INT16_C( 22370) },
      {  INT16_C( 22790), -INT16_C( 27839), -INT16_C( 26546), -INT16_C( 10395), -INT16_C( 11772), -INT16_C(  5066), -INT16_C( 27148),  INT16_C(  7729),
         INT16_C( 22568),  INT16_C(  4400),  INT16_C( 15931), -INT16_C(  9184),  INT16_C(  3893), -INT16_C(  1912), -INT16_C(  5387), -INT16_C(  1201) },
      {  INT16_C(  5860),  INT16_C(  2287),  INT16_C(  1083),  INT16_C( 27734), -INT16_C(  4991), -INT16_C(  1889), -INT16_C(  3702),  INT16_C(  2094),
        -INT16_C(  9595),  INT16_C(  1885), -INT16_C(  1811),  INT16_C( 22248), -INT16_C(   143),  INT16_C(   620), -INT16_C(  7960), -INT16_C(   410) } },
    { { -INT16_C( 28604), -INT16_C( 28017), -INT16_C(  3031),  INT16_C( 11626), -INT16_C( 24378), -INT16_C( 17639),  INT16_C( 18997),  INT16_C( 24025),
         INT16_C(  2467), -INT16_C(  8593), -INT16_C( 28857),  INT16_C( 31930),  INT16_C( 17055), -INT16_C( 27532), -INT16_C( 15571),  INT16_C( 29071) },
      UINT16_C( 7764),
      {  INT16_C( 32003),  INT16_C( 27923), -INT16_C(  9814), -INT16_C( 15346),  INT16_C( 17300),  INT16_C( 27918), -INT16_C( 20063),  INT16_C(  4214),
        -INT16_C( 16753),  INT16_C( 18847),  INT16_C( 15930), -INT16_C( 20596), -INT16_C( 17966),  INT16_C( 25202), -INT16_C( 14806),  INT16_C( 11648) },
      { -INT16_C( 27837), -INT16_C(  4453), -INT16_C( 22163),  INT16_C(   434), -INT16_C( 16148), -INT16_C( 29329), -INT16_C(  6798),  INT16_C(   413),
         INT16_C( 15779), -INT16_C(  8629), -INT16_C( 10373),  INT16_C( 20109), -INT16_C(   112), -INT16_C( 17744),  INT16_C( 12486),  INT16_C(  2535) },
      { -INT16_C( 28604), -INT16_C( 28017),  INT16_C(  3318),  INT16_C( 11626), -INT16_C(  4263), -INT16_C( 17639),  INT16_C(  2081),  INT16_C( 24025),
         INT16_C(  2467), -INT16_C(  2482), -INT16_C(  2522), -INT16_C(  6320),  INT16_C(    30), -INT16_C( 27532), -INT16_C( 15571),  INT16_C( 29071) } },
    { { -INT16_C( 32060),  INT16_C( 12791), -INT16_C( 22229),  INT16_C(  6194), -INT16_C( 24214), -INT16_C(  9051),  INT16_C( 17287),  INT16_C( 10973),
         INT16_C( 10368), -INT16_C(  1272), -INT16_C( 27137), -INT16_C( 28855), -INT16_C(  1643),  INT16_C( 23369),  INT16_C( 12586), -INT16_C(  4508) },
      UINT16_C(23731),
      { -INT16_C(  8417),  INT16_C( 20741),  INT16_C( 28663), -INT16_C( 25357),  INT16_C( 31307),  INT16_C( 10719),  INT16_C( 24484), -INT16_C( 21167),
         INT16_C( 20827), -INT16_C( 23486), -INT16_C( 10272),  INT16_C( 10910), -INT16_C( 14286), -INT16_C( 26789),  INT16_C(  3766), -INT16_C( 10765) },
      { -INT16_C(  1811), -INT16_C(  7130),  INT16_C(  6504), -INT16_C( 19583),  INT16_C( 24723),  INT16_C( 14556),  INT16_C( 11968),  INT16_C(  7141),
         INT16_C( 10111),  INT16_C( 24511),  INT16_C( 24063),  INT16_C( 12681), -INT16_C(  7131), -INT16_C(  9272), -INT16_C( 17421), -INT16_C(  8016) },
      {  INT16_C(   232), -INT16_C(  2257), -INT16_C( 22229),  INT16_C(  6194),  INT16_C( 11810),  INT16_C(  2380),  INT16_C( 17287), -INT16_C(  2307),
         INT16_C( 10368), -INT16_C(  1272), -INT16_C(  3772),  INT16_C(  2111),  INT16_C(  1554),  INT16_C( 23369), -INT16_C(  1002), -INT16_C(  4508) } },
    { { -INT16_C( 10316),  INT16_C(  7365),  INT16_C( 18160), -INT16_C( 31537), -INT16_C( 21338),  INT16_C( 26300), -INT16_C( 24102),  INT16_C( 22913),
         INT16_C( 16840), -INT16_C( 14408),  INT16_C( 17054), -INT16_C( 15111), -INT16_C( 16090),  INT16_C(  6559),  INT16_C( 20605),  INT16_C( 12794) },
      UINT16_C(48935),
      {  INT16_C(  5965),  INT16_C(  7173), -INT16_C( 21605),  INT16_C( 22472), -INT16_C( 24046), -INT16_C( 27656), -INT16_C( 15877), -INT16_C( 19244),
         INT16_C( 29576), -INT16_C( 32266),  INT16_C(  7223), -INT16_C( 10685), -INT16_C( 16330),  INT16_C( 12326),  INT16_C( 19953),  INT16_C( 16111) },
      { -INT16_C(  2971),  INT16_C(    90),  INT16_C(  9119), -INT16_C( 20136),  INT16_C( 20677), -INT16_C( 16059),  INT16_C(  6417), -INT16_C( 25995),
         INT16_C( 27532), -INT16_C( 15589),  INT16_C( 24199), -INT16_C( 16998), -INT16_C( 16354),  INT16_C(  4077), -INT16_C(  9202),  INT16_C( 29517) },
      { -INT16_C(   271),  INT16_C(     9), -INT16_C(  3007), -INT16_C( 31537), -INT16_C( 21338),  INT16_C(  6776), -INT16_C( 24102),  INT16_C( 22913),
         INT16_C( 12425),  INT16_C(  7675),  INT16_C(  2667),  INT16_C(  2771),  INT16_C(  4075),  INT16_C(   766),  INT16_C( 20605),  INT16_C(  7256) } },
    { { -INT16_C( 22320),  INT16_C( 28787), -INT16_C( 13365), -INT16_C( 28639),  INT16_C( 26140),  INT16_C( 11601), -INT16_C( 14720),  INT16_C(  3271),
        -INT16_C(  7375), -INT16_C( 17968),  INT16_C( 27201),  INT16_C( 24694),  INT16_C( 25642),  INT16_C( 14447), -INT16_C( 17088),  INT16_C(  4523) },
      UINT16_C( 8037),
      {  INT16_C( 12417), -INT16_C( 23830),  INT16_C(  1728),  INT16_C(  4617), -INT16_C( 30412), -INT16_C(  1064),  INT16_C(  2709),  INT16_C( 26078),
         INT16_C(  8387),  INT16_C( 14799), -INT16_C(  1408), -INT16_C(  4195), -INT16_C(  8654), -INT16_C(  8532),  INT16_C(  4591),  INT16_C( 28925) },
      { -INT16_C(  6335),  INT16_C(   530),  INT16_C(  7150),  INT16_C(  8724), -INT16_C(  4956),  INT16_C( 14877), -INT16_C(   778), -INT16_C( 18017),
         INT16_C( 28444), -INT16_C( 25357), -INT16_C( 28567), -INT16_C( 25717),  INT16_C( 14446),  INT16_C( 23929),  INT16_C( 30281), -INT16_C( 29747) },
      { -INT16_C(  1201),  INT16_C( 28787),  INT16_C(   188), -INT16_C( 28639),  INT16_C( 26140), -INT16_C(   242), -INT16_C(    33),  INT16_C(  3271),
         INT16_C(  3640), -INT16_C(  5726),  INT16_C(   613),  INT16_C(  1646), -INT16_C(  1908),  INT16_C( 14447), -INT16_C( 17088),  INT16_C(  4523) } },
    { { -INT16_C(  8098),  INT16_C( 19597), -INT16_C( 24069), -INT16_C( 24466), -INT16_C( 29811), -INT16_C( 31526),  INT16_C( 31111), -INT16_C( 23747),
         INT16_C( 12520),  INT16_C( 20799), -INT16_C( 13375),  INT16_C( 12269),  INT16_C( 26115),  INT16_C( 19597),  INT16_C( 23261),  INT16_C( 15319) },
      UINT16_C(25658),
      {  INT16_C( 13959), -INT16_C(  2811), -INT16_C( 27690), -INT16_C( 20352),  INT16_C(  2071),  INT16_C( 21545),  INT16_C(  4779), -INT16_C(  5243),
         INT16_C( 18019),  INT16_C( 20662), -INT16_C( 18059),  INT16_C(   695), -INT16_C( 27643), -INT16_C(  8867), -INT16_C( 26673),  INT16_C( 22081) },
      {  INT16_C( 18381), -INT16_C( 23733), -INT16_C( 13350), -INT16_C(  3757),  INT16_C( 32211),  INT16_C( 32581), -INT16_C( 13681), -INT16_C(  3478),
         INT16_C(  8208), -INT16_C( 31165), -INT16_C(  1319), -INT16_C(  8568), -INT16_C(  6770),  INT16_C( 23995), -INT16_C(   643),  INT16_C( 19123) },
      { -INT16_C(  8098),  INT16_C(  1017), -INT16_C( 24069),  INT16_C(  1166),  INT16_C(  1017),  INT16_C( 10711),  INT16_C( 31111), -INT16_C( 23747),
         INT16_C( 12520),  INT16_C( 20799),  INT16_C(   363),  INT16_C( 12269),  INT16_C( 26115), -INT16_C(  3247),  INT16_C(   261),  INT16_C( 15319) } },
    { { -INT16_C(   444),  INT16_C(  7918),  INT16_C( 16841), -INT16_C( 25329),  INT16_C( 21694),  INT16_C( 19740), -INT16_C( 31201),  INT16_C( 12096),
        -INT16_C( 31834),  INT16_C( 32693),  INT16_C( 15997),  INT16_C(  2909),  INT16_C(  6435), -INT16_C( 24472),  INT16_C(  6934),  INT16_C( 23275) },
      UINT16_C(55577),
      { -INT16_C(  7560), -INT16_C( 30950), -INT16_C(  9857), -INT16_C( 25637), -INT16_C(  1498),  INT16_C( 26145), -INT16_C( 14550), -INT16_C(  8215),
         INT16_C( 26182), -INT16_C( 23523),  INT16_C( 16753), -INT16_C(  9795), -INT16_C( 11295), -INT16_C( 13068),  INT16_C(  3373), -INT16_C( 23131) },
      { -INT16_C( 16144),  INT16_C( 28460),  INT16_C(  1945), -INT16_C( 16629),  INT16_C( 11266),  INT16_C( 11302),  INT16_C(  4084),  INT16_C( 14859),
         INT16_C( 10614), -INT16_C(  6178), -INT16_C( 25750),  INT16_C( 19393), -INT16_C( 19090), -INT16_C( 25832), -INT16_C( 16957), -INT16_C( 19648) },
      {  INT16_C(  1862),  INT16_C(  7918),  INT16_C( 16841),  INT16_C(  6505), -INT16_C(   258),  INT16_C( 19740), -INT16_C( 31201),  INT16_C( 12096),
         INT16_C(  4240),  INT16_C( 32693),  INT16_C( 15997), -INT16_C(  2899),  INT16_C(  3290), -INT16_C( 24472), -INT16_C(   873),  INT16_C(  6934) } },
    { {  INT16_C( 27773),  INT16_C(  5666),  INT16_C( 11636),  INT16_C( 30422), -INT16_C(   934),  INT16_C( 20130), -INT16_C( 21237), -INT16_C( 32376),
         INT16_C( 26582),  INT16_C( 16489),  INT16_C( 10754),  INT16_C( 29068), -INT16_C( 23329), -INT16_C( 24052),  INT16_C( 19809), -INT16_C(  8363) },
      UINT16_C(30905),
      {  INT16_C( 11765), -INT16_C( 13403), -INT16_C(    93),  INT16_C( 17863), -INT16_C( 11443), -INT16_C( 10509), -INT16_C( 13996), -INT16_C( 17091),
         INT16_C( 16138), -INT16_C( 26905), -INT16_C( 14416), -INT16_C( 17094), -INT16_C( 25751), -INT16_C( 16630), -INT16_C( 15494),  INT16_C( 28727) },
      { -INT16_C(  8975), -INT16_C( 27589),  INT16_C(   988),  INT16_C( 10714), -INT16_C( 12842),  INT16_C( 11007),  INT16_C( 15510), -INT16_C( 24344),
        -INT16_C( 12420),  INT16_C( 11318),  INT16_C( 28822),  INT16_C(   233), -INT16_C(  3316), -INT16_C( 31041), -INT16_C(  2377), -INT16_C( 22282) },
      { -INT16_C(  1612),  INT16_C(  5666),  INT16_C( 11636),  INT16_C(  2920),  INT16_C(  2242), -INT16_C(  1766), -INT16_C( 21237),  INT16_C(  6348),
         INT16_C( 26582),  INT16_C( 16489),  INT16_C( 10754), -INT16_C(    61),  INT16_C(  1302),  INT16_C(  7876),  INT16_C(   561), -INT16_C(  8363) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mulhi_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mulhi_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_mask_mulhi_epi16(src, k, a, b);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_mulhi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t k;
    int16_t a[16];
    int16_t b[16];
    int16_t r[16];
  } test_vec[] = {
    { UINT16_C(51313),
      {  INT16_C(  9115),  INT16_C( 18133), -INT16_C( 19621),  INT16_C(  7884),  INT16_C(  6460),  INT16_C( 32681),  INT16_C( 14342), -INT16_C( 31075),
         INT16_C(  5867),  INT16_C(  5928),  INT16_C( 29749), -INT16_C( 23488),  INT16_C(  6757),  INT16_C( 25540),  INT16_C( 14033),  INT16_C( 27691) },
      {  INT16_C(    89), -INT16_C( 19277),  INT16_C( 32692), -INT16_C(  3885),  INT16_C( 31896), -INT16_C( 24721),  INT16_C(  3253), -INT16_C( 24539),
         INT16_C( 19747),  INT16_C( 22711), -INT16_C(  1854),  INT16_C( 10236), -INT16_C( 16110), -INT16_C(  7286), -INT16_C( 18953),  INT16_C( 20560) },
      {  INT16_C(    12),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3144), -INT16_C( 12328),  INT16_C(   711),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  3669),  INT16_C(     0),  INT16_C(     0), -INT16_C(  4059),  INT16_C(  8687) } },
    { UINT16_C(  950),
      {  INT16_C( 27140), -INT16_C( 10366),  INT16_C(  7002), -INT16_C( 13996),  INT16_C(  2490), -INT16_C(  8234), -INT16_C(  1623),  INT16_C( 24877),
        -INT16_C(  4271),  INT16_C( 19801),  INT16_C( 27414), -INT16_C( 24306),  INT16_C(  1359), -INT16_C( 24746),  INT16_C(  3157),  INT16_C( 23202) },
      {  INT16_C(  9334), -INT16_C( 12239), -INT16_C( 31425), -INT16_C(  1638),  INT16_C( 28814),  INT16_C( 14553),  INT16_C(  1641), -INT16_C( 17767),
        -INT16_C(  3339),  INT16_C(  2823),  INT16_C(  5725), -INT16_C( 21332),  INT16_C(   795),  INT16_C( 29003), -INT16_C(  4849), -INT16_C( 31029) },
      {  INT16_C(     0),  INT16_C(  1935), -INT16_C(  3358),  INT16_C(     0),  INT16_C(  1094), -INT16_C(  1829),  INT16_C(     0), -INT16_C(  6745),
         INT16_C(   217),  INT16_C(   852),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(64530),
      {  INT16_C( 20822), -INT16_C(  3966),  INT16_C(  4171),  INT16_C(  9312), -INT16_C( 14008), -INT16_C(  7894),  INT16_C(  8067), -INT16_C( 29741),
         INT16_C( 12586), -INT16_C( 10335), -INT16_C( 17187),  INT16_C( 10714), -INT16_C(  5843), -INT16_C(  2026),  INT16_C( 10351), -INT16_C( 14603) },
      {  INT16_C( 30586), -INT16_C( 14922),  INT16_C(  6023), -INT16_C( 12055),  INT16_C(  5088),  INT16_C( 25777), -INT16_C( 31438),  INT16_C( 23791),
        -INT16_C( 28490), -INT16_C( 27853),  INT16_C(  3404),  INT16_C( 31420), -INT16_C( 11273),  INT16_C( 26226),  INT16_C( 26619),  INT16_C( 29996) },
      {  INT16_C(     0),  INT16_C(   903),  INT16_C(     0),  INT16_C(     0), -INT16_C(  1088),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C(   893),  INT16_C(  5136),  INT16_C(  1005), -INT16_C(   811),  INT16_C(  4204), -INT16_C(  6684) } },
    { UINT16_C(58334),
      {  INT16_C( 26170),  INT16_C(  9210), -INT16_C(  9674), -INT16_C(  6346),  INT16_C( 26686),  INT16_C( 11628),  INT16_C(  8901), -INT16_C(  1859),
         INT16_C(  2742),  INT16_C( 29190), -INT16_C(   636), -INT16_C(  2491),  INT16_C( 16739), -INT16_C( 28578),  INT16_C( 15542), -INT16_C(  3725) },
      {  INT16_C( 28066), -INT16_C( 10220),  INT16_C( 19271), -INT16_C( 31040),  INT16_C( 11443),  INT16_C( 30899),  INT16_C( 29007),  INT16_C(  1393),
         INT16_C( 30587), -INT16_C(   137), -INT16_C( 17036), -INT16_C( 10251),  INT16_C( 21502), -INT16_C( 19353), -INT16_C(  9584),  INT16_C( 12965) },
      {  INT16_C(     0), -INT16_C(  1437), -INT16_C(  2845),  INT16_C(  3005),  INT16_C(  4659),  INT16_C(     0),  INT16_C(  3939), -INT16_C(    40),
         INT16_C(  1279), -INT16_C(    62),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8439), -INT16_C(  2273), -INT16_C(   737) } },
    { UINT16_C(47687),
      { -INT16_C( 28917), -INT16_C( 13563), -INT16_C( 18411), -INT16_C( 14089),  INT16_C( 17969), -INT16_C( 24007), -INT16_C( 19381), -INT16_C( 15591),
        -INT16_C( 29261), -INT16_C( 22144),  INT16_C( 32356), -INT16_C( 13060), -INT16_C( 29646), -INT16_C( 10074), -INT16_C(  4417), -INT16_C( 13678) },
      { -INT16_C( 26755), -INT16_C( 28011), -INT16_C( 29617), -INT16_C( 32678), -INT16_C( 27437),  INT16_C(  7714),  INT16_C( 15176), -INT16_C(   799),
         INT16_C( 25032),  INT16_C( 11685), -INT16_C( 24097),  INT16_C(  4857), -INT16_C( 24786), -INT16_C(  4630),  INT16_C( 31885),  INT16_C(  2743) },
      {  INT16_C( 11805),  INT16_C(  5797),  INT16_C(  8320),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  4489),  INT16_C(     0),
         INT16_C(     0), -INT16_C(  3949),  INT16_C(     0), -INT16_C(   968),  INT16_C( 11212),  INT16_C(   711),  INT16_C(     0), -INT16_C(   573) } },
    { UINT16_C(19475),
      {  INT16_C( 25244), -INT16_C(  2088), -INT16_C( 21533),  INT16_C(  1419), -INT16_C( 11318), -INT16_C( 21695),  INT16_C(  2511),  INT16_C( 29709),
        -INT16_C(  5066),  INT16_C( 12054),  INT16_C( 17662), -INT16_C(  5937),  INT16_C( 23601), -INT16_C(  6044),  INT16_C( 30567),  INT16_C(   820) },
      {  INT16_C(  3290), -INT16_C( 16902), -INT16_C( 31304), -INT16_C( 32062),  INT16_C(   857),  INT16_C( 10285),  INT16_C( 14861),  INT16_C( 17309),
        -INT16_C( 19673),  INT16_C(  9587),  INT16_C( 17143),  INT16_C( 10254),  INT16_C( 29342),  INT16_C(  1296),  INT16_C( 17642), -INT16_C( 15351) },
      {  INT16_C(  1267),  INT16_C(   538),  INT16_C(     0),  INT16_C(     0), -INT16_C(   149),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  4620), -INT16_C(   929),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8228),  INT16_C(     0) } },
    { UINT16_C(  848),
      {  INT16_C(  2177),  INT16_C( 17289), -INT16_C(  7542), -INT16_C( 18361),  INT16_C( 21514), -INT16_C( 22542),  INT16_C(  6551),  INT16_C(  2650),
         INT16_C( 20799),  INT16_C( 19788), -INT16_C(  5255), -INT16_C( 30273), -INT16_C( 22032), -INT16_C(  1587),  INT16_C(  7789), -INT16_C(  4355) },
      { -INT16_C( 31194), -INT16_C( 20174),  INT16_C( 31080),  INT16_C( 29289),  INT16_C( 23501),  INT16_C( 25626),  INT16_C( 29813), -INT16_C( 19345),
        -INT16_C( 17466),  INT16_C( 16129), -INT16_C( 16218), -INT16_C( 26679), -INT16_C( 27030), -INT16_C( 10352), -INT16_C( 29260), -INT16_C(  9274) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7714),  INT16_C(     0),  INT16_C(  2980),  INT16_C(     0),
        -INT16_C(  5544),  INT16_C(  4870),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(63507),
      {  INT16_C( 31628), -INT16_C(  2703),  INT16_C( 16110),  INT16_C(  2128), -INT16_C( 14942),  INT16_C(  4476),  INT16_C( 17017),  INT16_C( 31437),
         INT16_C( 29570),  INT16_C( 19259), -INT16_C( 23286), -INT16_C( 25631), -INT16_C( 27012),  INT16_C( 16936),  INT16_C( 15473), -INT16_C(   710) },
      { -INT16_C( 21577), -INT16_C( 23054),  INT16_C( 17129), -INT16_C( 29523),  INT16_C( 10760), -INT16_C( 32355),  INT16_C( 27244), -INT16_C(  4356),
         INT16_C( 14302), -INT16_C(  6087),  INT16_C(  7132),  INT16_C( 22659), -INT16_C( 21327),  INT16_C(  8859), -INT16_C( 10776), -INT16_C( 24801) },
      { -INT16_C( 10414),  INT16_C(   950),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2454),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  8862),  INT16_C(  8790),  INT16_C(  2289), -INT16_C(  2545),  INT16_C(   268) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mulhi_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mulhi_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_maskz_mulhi_epi16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mulhi_epu16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t src[16];
    uint16_t k;
    uint16_t a[16];
    uint16_t b[16];
    uint16_t r[16];
  } test_vec[] = {
    { { UINT16_C(13010), UINT16_C(44604), UINT16_C( 5685), UINT16_C( 3032), UINT16_C(55267), UINT16_C(31285), UINT16_C( 7444), UINT16_C(36890),
        UINT16_C(20973), UINT16_C(33724), UINT16_C(42689), UINT16_C(52611), UINT16_C(17049), UINT16_C(20564), UINT16_C(19000), UINT16_C( 3064) },
      UINT16_C(13692),
      { UINT16_C(45497), UINT16_C(37195), UINT16_C(12220), UINT16_C(62057), UINT16_C(32169), UINT16_C(49935), UINT16_C(64525), UINT16_C(51476),
        UINT16_C(54912), UINT16_C(  879), UINT16_C( 2467), UINT16_C(63302), UINT16_C(32345), UINT16_C(21058), UINT16_C(48777), UINT16_C(17287) },
      { UINT16_C(53872), UINT16_C(11476), UINT16_C(15617), UINT16_C(43550), UINT16_C(11962), UINT16_C(51054), UINT16_C(33322), UINT16_C(43665),
        UINT16_C(   88), UINT16_C(64686), UINT16_C(62473), UINT16_C(25587), UINT16_C(13682), UINT16_C(64693), UINT16_C(15604), UINT16_C(25663) },
      { UINT16_C(13010), UINT16_C(44604), UINT16_C( 2911), UINT16_C(41238), UINT16_C( 5871), UINT16_C(38900), UINT16_C(32807), UINT16_C(36890),
        UINT16_C(   73), UINT16_C(33724), UINT16_C( 2351), UINT16_C(52611), UINT16_C( 6752), UINT16_C(20787), UINT16_C(19000), UINT16_C( 3064) } },
    { { UINT16_C( 4878), UINT16_C( 4240), UINT16_C(44881), UINT16_C( 3002), UINT16_C(10461), UINT16_C( 2003), UINT16_C(25771), UINT16_C(  946),
        UINT16_C(24676), UINT16_C(28415), UINT16_C(62292), UINT16_C(50897), UINT16_C(34344), UINT16_C( 7362), UINT16_C(  450), UINT16_C(53376) },
      UINT16_C( 4373),
      { UINT16_C(26336), UINT16_C(39872), UINT16_C(40305), UINT16_C(17603), UINT16_C(28324), UINT16_C(22184), UINT16_C( 3442), UINT16_C(29110),
        UINT16_C( 2683), UINT16_C(19556), UINT16_C(36305), UINT16_C(37842), UINT16_C(38057), UINT16_C(10901), UINT16_C(43620), UINT16_C(17723) },
      { UINT16_C(64272), UINT16_C(33248), UINT16_C(41880), UINT16_C(15558), UINT16_C(28178), UINT16_C(33939), UINT16_C(18811), UINT16_C(63221),
        UINT16_C(23124), UINT16_C( 9538), UINT16_C( 5351), UINT16_C(37048), UINT16_C(19880), UINT16_C( 3514), UINT16_C(62967), UINT16_C( 1874) },
      { UINT16_C(25828), UINT16_C( 4240), UINT16_C(25756), UINT16_C( 3002), UINT16_C(12178), UINT16_C( 2003), UINT16_C(25771), UINT16_C(  946),
        UINT16_C(  946), UINT16_C(28415), UINT16_C(62292), UINT16_C(50897), UINT16_C(11544), UINT16_C( 7362), UINT16_C(  450), UINT16_C(53376) } },
    { { UINT16_C(13040), UINT16_C(34953), UINT16_C(20437), UINT16_C(59333), UINT16_C(22717), UINT16_C(14699), UINT16_C(24993), UINT16_C(62767),
        UINT16_C(29371), UINT16_C(41498), UINT16_C(54150), UINT16_C(12082), UINT16_C(60704), UINT16_C( 6204), UINT16_C(36578), UINT16_C(54047) },
      UINT16_C(43200),
      { UINT16_C(38235), UINT16_C( 8439), UINT16_C(46461), UINT16_C(59512), UINT16_C( 6894), UINT16_C( 7497), UINT16_C( 1039), UINT16_C(10895),
        UINT16_C( 5798), UINT16_C(55805), UINT16_C( 7493), UINT16_C(33222), UINT16_C(43061), UINT16_C(21775), UINT16_C(53115), UINT16_C(55293) },
      { UINT16_C(62820), UINT16_C(57847), UINT16_C(28842), UINT16_C(39114), UINT16_C( 5002), UINT16_C(39349), UINT16_C(17688), UINT16_C(48835),
        UINT16_C(49243), UINT16_C(41111), UINT16_C(24030), UINT16_C( 4897), UINT16_C(12294), UINT16_C(33128), UINT16_C(26367), UINT16_C(25432) },
      { UINT16_C(13040), UINT16_C(34953), UINT16_C(20437), UINT16_C(59333), UINT16_C(22717), UINT16_C(14699), UINT16_C(  280), UINT16_C( 8118),
        UINT16_C(29371), UINT16_C(41498), UINT16_C(54150), UINT16_C( 2482), UINT16_C(60704), UINT16_C(11007), UINT16_C(36578), UINT16_C(21457) } },
    { { UINT16_C(20571), UINT16_C( 1349), UINT16_C( 4032), UINT16_C(19101), UINT16_C(21026), UINT16_C(15075), UINT16_C(42903), UINT16_C(62201),
        UINT16_C(36967), UINT16_C(17810), UINT16_C(46062), UINT16_C(62553), UINT16_C(49635), UINT16_C(57973), UINT16_C(52775), UINT16_C(33350) },
      UINT16_C(35614),
      { UINT16_C(56967), UINT16_C( 9370), UINT16_C(48168), UINT16_C( 2935), UINT16_C( 3831), UINT16_C(61618), UINT16_C( 6657), UINT16_C(37760),
        UINT16_C(28255), UINT16_C(47175), UINT16_C(10850), UINT16_C(55418), UINT16_C(41229), UINT16_C(21414), UINT16_C(50212), UINT16_C(43998) },
      { UINT16_C(30882), UINT16_C(51920), UINT16_C(18228), UINT16_C(11221), UINT16_C(34901), UINT16_C(22043), UINT16_C(40098), UINT16_C(  490),
        UINT16_C(12554), UINT16_C(28090), UINT16_C(13403), UINT16_C(26693), UINT16_C(60373), UINT16_C(63931), UINT16_C(39343), UINT16_C(20901) },
      { UINT16_C(20571), UINT16_C( 7423), UINT16_C(13397), UINT16_C(  502), UINT16_C( 2040), UINT16_C(15075), UINT16_C(42903), UINT16_C(62201),
        UINT16_C( 5412), UINT16_C(20220), UINT16_C(46062), UINT16_C(22571), UINT16_C(49635), UINT16_C(57973), UINT16_C(52775), UINT16_C(14032) } },
    { { UINT16_C(29969), UINT16_C(17947), UINT16_C(61628), UINT16_C( 4465), UINT16_C(36216), UINT16_C( 6760), UINT16_C(21033), UINT16_C(13084),
        UINT16_C(54915), UINT16_C(56992), UINT16_C(58634), UINT16_C(57159), UINT16_C(  720), UINT16_C(32729), UINT16_C(32412), UINT16_C(44496) },
      UINT16_C(60403),
      { UINT16_C(45043), UINT16_C(26076), UINT16_C(21696), UINT16_C(10482), UINT16_C( 7023), UINT16_C(35706), UINT16_C(64846), UINT16_C(61281),
        UINT16_C(27612), UINT16_C( 9172), UINT16_C(42314), UINT16_C( 8997), UINT16_C(49444), UINT16_C(62881), UINT16_C(37999), UINT16_C(25312) },
      { UINT16_C(48195), UINT16_C( 1223), UINT16_C(47377), UINT16_C(32812), UINT16_C(42964), UINT16_C( 8971), UINT16_C(27812), UINT16_C(32786),
        UINT16_C(59095), UINT16_C( 8611), UINT16_C(51595), UINT16_C(45125), UINT16_C(59018), UINT16_C(63909), UINT16_C(34171), UINT16_C(48732) },
      { UINT16_C(33124), UINT16_C(  486), UINT16_C(61628), UINT16_C( 4465), UINT16_C( 4604), UINT16_C( 4887), UINT16_C(27519), UINT16_C(30657),
        UINT16_C(24898), UINT16_C( 1205), UINT16_C(58634), UINT16_C( 6194), UINT16_C(  720), UINT16_C(61319), UINT16_C(19812), UINT16_C(18821) } },
    { { UINT16_C( 9026), UINT16_C(21442), UINT16_C(61405), UINT16_C(45523), UINT16_C(56982), UINT16_C(15060), UINT16_C(58954), UINT16_C( 8635),
        UINT16_C(24269), UINT16_C(22594), UINT16_C(34599), UINT16_C(45576), UINT16_C(44398), UINT16_C(59819), UINT16_C( 1843), UINT16_C(30119) },
      UINT16_C(27179),
      { UINT16_C( 2248), UINT16_C(39769), UINT16_C(61369), UINT16_C(36473), UINT16_C(49961), UINT16_C(58484), UINT16_C(16868), UINT16_C( 9795),
        UINT16_C(27290), UINT16_C(41646), UINT16_C( 7196), UINT16_C(51280), UINT16_C(33541), UINT16_C(44239), UINT16_C(64248), UINT16_C(49174) },
      { UINT16_C(28418), UINT16_C(48219), UINT16_C(54366), UINT16_C(34890), UINT16_C(48791), UINT16_C(31596), UINT16_C(44800), UINT16_C(39585),
        UINT16_C(20250), UINT16_C(13884), UINT16_C(35947), UINT16_C(28926), UINT16_C(52751), UINT16_C( 1821), UINT16_C(13256), UINT16_C(52167) },
      { UINT16_C(  974), UINT16_C(29260), UINT16_C(61405), UINT16_C(19417), UINT16_C(56982), UINT16_C(28196), UINT16_C(58954), UINT16_C( 8635),
        UINT16_C(24269), UINT16_C( 8822), UINT16_C(34599), UINT16_C(22633), UINT16_C(44398), UINT16_C( 1229), UINT16_C(12995), UINT16_C(30119) } },
    { { UINT16_C( 8867), UINT16_C(  391), UINT16_C(53750), UINT16_C(36233), UINT16_C(63119), UINT16_C(36616), UINT16_C(43685), UINT16_C(48937),
        UINT16_C(26361), UINT16_C(26102), UINT16_C(62706), UINT16_C(  725), UINT16_C(62146), UINT16_C(35593), UINT16_C(53542), UINT16_C(51542) },
      UINT16_C(56819),
      { UINT16_C(60106), UINT16_C(21678), UINT16_C(15735), UINT16_C(32842), UINT16_C(61389), UINT16_C(63018), UINT16_C( 9135), UINT16_C(42332),
        UINT16_C(20360), UINT16_C(24217), UINT16_C(23633), UINT16_C(23120), UINT16_C(30439), UINT16_C(15659), UINT16_C( 7999), UINT16_C( 2586) },
      { UINT16_C(51209), UINT16_C(32862), UINT16_C(43013), UINT16_C(53760), UINT16_C(10903), UINT16_C(18121), UINT16_C( 9550), UINT16_C(55019),
        UINT16_C(34164), UINT16_C(50484), UINT16_C(34273), UINT16_C(51232), UINT16_C(19451), UINT16_C(15109), UINT16_C( 8042), UINT16_C(29509) },
      { UINT16_C(46966), UINT16_C(10870), UINT16_C(53750), UINT16_C(36233), UINT16_C(10213), UINT16_C(17424), UINT16_C( 1331), UINT16_C(35538),
        UINT16_C(10613), UINT16_C(26102), UINT16_C(12359), UINT16_C(18073), UINT16_C( 9034), UINT16_C(35593), UINT16_C(  981), UINT16_C( 1164) } },
    { { UINT16_C(41959), UINT16_C(60660), UINT16_C(62539), UINT16_C(58047), UINT16_C(34847), UINT16_C(27945), UINT16_C( 5293), UINT16_C( 8771),
        UINT16_C(30873), UINT16_C(31463), UINT16_C( 2045), UINT16_C(63554), UINT16_C(18259), UINT16_C(48435), UINT16_C(30822), UINT16_C(19761) },
      UINT16_C( 9499),
      { UINT16_C(26170), UINT16_C(63769), UINT16_C(14409), UINT16_C(29313), UINT16_C(11941), UINT16_C(59782), UINT16_C( 8272), UINT16_C(14433),
        UINT16_C(24218), UINT16_C(56639), UINT16_C(37462), UINT16_C(35364), UINT16_C(35664), UINT16_C(33026), UINT16_C( 7896), UINT16_C( 4774) },
      { UINT16_C(49028), UINT16_C(52491), UINT16_C(36088), UINT16_C(40255), UINT16_C(50875), UINT16_C( 2950), UINT16_C(59366), UINT16_C(32835),
        UINT16_C(33605), UINT16_C(40029), UINT16_C(33301), UINT16_C(25894), UINT16_C(10253), UINT16_C(58854), UINT16_C(35910), UINT16_C(52216) },
      { UINT16_C(19577), UINT16_C(51075), UINT16_C(62539), UINT16_C(18005), UINT16_C( 9269), UINT16_C(27945), UINT16_C( 5293), UINT16_C( 8771),
        UINT16_C(12418), UINT16_C(31463), UINT16_C(19035), UINT16_C(63554), UINT16_C(18259), UINT16_C(29658), UINT16_C(30822), UINT16_C(19761) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mulhi_epu16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mulhi_epu16");
    easysimd_test_x86_assert_equal_u16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_u16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__m256i r = easysimd_mm256_mask_mulhi_epu16(src, k, a, b);

    easysimd_test_x86_write_u16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_mulhi_epu16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t k;
    uint16_t a[16];
    uint16_t b[16];
    uint16_t r[16];
  } test_vec[] = {
    { UINT16_C( 4481),
      { UINT16_C(27205), UINT16_C(62035), UINT16_C(23542), UINT16_C(37916), UINT16_C(35293), UINT16_C(55806), UINT16_C(56439), UINT16_C(45328),
        UINT16_C(60613), UINT16_C(18636), UINT16_C(32068), UINT16_C(57332), UINT16_C(56479), UINT16_C(48821), UINT16_C(13948), UINT16_C(49615) },
      { UINT16_C( 8864), UINT16_C(38835), UINT16_C(53374), UINT16_C(23339), UINT16_C(10585), UINT16_C(53300), UINT16_C(17414), UINT16_C(52097),
        UINT16_C(19760), UINT16_C(29715), UINT16_C( 2250), UINT16_C(26964), UINT16_C( 2532), UINT16_C(24615), UINT16_C(63039), UINT16_C(57121) },
      { UINT16_C( 3679), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(36032),
        UINT16_C(18275), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 2182), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT16_C(54553),
      { UINT16_C(38774), UINT16_C(41381), UINT16_C(65266), UINT16_C( 9931), UINT16_C(53710), UINT16_C(20586), UINT16_C(39580), UINT16_C(44957),
        UINT16_C(26638), UINT16_C(25271), UINT16_C(40145), UINT16_C(63851), UINT16_C(43772), UINT16_C( 7919), UINT16_C( 2186), UINT16_C(  243) },
      { UINT16_C(39071), UINT16_C(37282), UINT16_C(28054), UINT16_C(25783), UINT16_C( 8510), UINT16_C(55988), UINT16_C(21179), UINT16_C(51849),
        UINT16_C(16826), UINT16_C(35628), UINT16_C(39133), UINT16_C(55684), UINT16_C(29762), UINT16_C(52471), UINT16_C(60028), UINT16_C( 7373) },
      { UINT16_C(23116), UINT16_C(    0), UINT16_C(    0), UINT16_C( 3907), UINT16_C( 6974), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C( 6839), UINT16_C(    0), UINT16_C(23971), UINT16_C(    0), UINT16_C(19878), UINT16_C(    0), UINT16_C( 2002), UINT16_C(   27) } },
    { UINT16_C(28546),
      { UINT16_C( 6317), UINT16_C(26076), UINT16_C( 6781), UINT16_C(12678), UINT16_C(17140), UINT16_C(32131), UINT16_C(15628), UINT16_C(14526),
        UINT16_C(39881), UINT16_C(19920), UINT16_C( 4981), UINT16_C(27841), UINT16_C(16095), UINT16_C(44119), UINT16_C(55642), UINT16_C( 1819) },
      { UINT16_C(63474), UINT16_C(28524), UINT16_C(62225), UINT16_C( 1440), UINT16_C( 9269), UINT16_C(16771), UINT16_C(16737), UINT16_C(10873),
        UINT16_C(19165), UINT16_C(21112), UINT16_C(14685), UINT16_C(15550), UINT16_C( 5495), UINT16_C(53737), UINT16_C( 1263), UINT16_C(57817) },
      { UINT16_C(    0), UINT16_C(11349), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 2409),
        UINT16_C(11662), UINT16_C( 6417), UINT16_C( 1116), UINT16_C( 6605), UINT16_C(    0), UINT16_C(36175), UINT16_C( 1072), UINT16_C(    0) } },
    { UINT16_C(17916),
      { UINT16_C( 3408), UINT16_C(61496), UINT16_C(27923), UINT16_C(38420), UINT16_C(30382), UINT16_C(10455), UINT16_C(46240), UINT16_C( 6258),
        UINT16_C(52998), UINT16_C(50514), UINT16_C(51467), UINT16_C(62682), UINT16_C(51611), UINT16_C(29945), UINT16_C(62890), UINT16_C(64185) },
      { UINT16_C(61954), UINT16_C( 5611), UINT16_C(65375), UINT16_C( 3755), UINT16_C(33653), UINT16_C( 5686), UINT16_C(43063), UINT16_C(15918),
        UINT16_C(32887), UINT16_C(33283), UINT16_C(56650), UINT16_C(58743), UINT16_C(28839), UINT16_C(20825), UINT16_C( 4709), UINT16_C(26444) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(27854), UINT16_C( 2201), UINT16_C(15601), UINT16_C(  907), UINT16_C(30383), UINT16_C( 1520),
        UINT16_C(26595), UINT16_C(    0), UINT16_C(44488), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 4518), UINT16_C(    0) } },
    { UINT16_C(14084),
      { UINT16_C(25725), UINT16_C(10294), UINT16_C(44146), UINT16_C(43179), UINT16_C(58306), UINT16_C(61520), UINT16_C(50977), UINT16_C( 9329),
        UINT16_C(47945), UINT16_C(49153), UINT16_C(43168), UINT16_C(63792), UINT16_C(38394), UINT16_C(17931), UINT16_C( 4349), UINT16_C(31357) },
      { UINT16_C(45940), UINT16_C(59042), UINT16_C(20063), UINT16_C( 8590), UINT16_C(56881), UINT16_C(21010), UINT16_C(33701), UINT16_C(61046),
        UINT16_C(30526), UINT16_C(57007), UINT16_C(57120), UINT16_C( 6871), UINT16_C(57973), UINT16_C(29280), UINT16_C(56818), UINT16_C(26348) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(13514), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(22332), UINT16_C(42756), UINT16_C(37624), UINT16_C(    0), UINT16_C(33963), UINT16_C( 8011), UINT16_C(    0), UINT16_C(    0) } },
    { UINT16_C(36496),
      { UINT16_C(61516), UINT16_C(56028), UINT16_C( 3345), UINT16_C( 9144), UINT16_C(23903), UINT16_C(54694), UINT16_C(58444), UINT16_C(64333),
        UINT16_C(28098), UINT16_C(39386), UINT16_C(20359), UINT16_C(59260), UINT16_C(28353), UINT16_C(44484), UINT16_C(21717), UINT16_C( 8508) },
      { UINT16_C( 6212), UINT16_C(22268), UINT16_C(46118), UINT16_C(34169), UINT16_C( 8210), UINT16_C(24155), UINT16_C(43012), UINT16_C(51033),
        UINT16_C(13077), UINT16_C(40032), UINT16_C(56451), UINT16_C(17539), UINT16_C(18251), UINT16_C( 8434), UINT16_C(11931), UINT16_C(57409) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 2994), UINT16_C(    0), UINT16_C(    0), UINT16_C(50096),
        UINT16_C(    0), UINT16_C(24058), UINT16_C(17536), UINT16_C(15859), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 7452) } },
    { UINT16_C(15686),
      { UINT16_C(27702), UINT16_C(45042), UINT16_C( 1266), UINT16_C(19919), UINT16_C(54370), UINT16_C(48117), UINT16_C( 2715), UINT16_C(64494),
        UINT16_C(29094), UINT16_C(10712), UINT16_C( 9142), UINT16_C(43120), UINT16_C( 2883), UINT16_C(34006), UINT16_C( 7403), UINT16_C( 8642) },
      { UINT16_C(46217), UINT16_C(31697), UINT16_C(41144), UINT16_C( 6856), UINT16_C(48500), UINT16_C( 4053), UINT16_C(50119), UINT16_C(27915),
        UINT16_C(58165), UINT16_C(60310), UINT16_C( 1542), UINT16_C(18835), UINT16_C(26897), UINT16_C(64973), UINT16_C(36741), UINT16_C( 3614) },
      { UINT16_C(    0), UINT16_C(21784), UINT16_C(  794), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 2076), UINT16_C(    0),
        UINT16_C(25821), UINT16_C(    0), UINT16_C(  215), UINT16_C(12392), UINT16_C( 1183), UINT16_C(33713), UINT16_C(    0), UINT16_C(    0) } },
    { UINT16_C(61251),
      { UINT16_C(64393), UINT16_C(20880), UINT16_C( 1045), UINT16_C(59918), UINT16_C(54548), UINT16_C( 8110), UINT16_C(58178), UINT16_C(55298),
        UINT16_C( 2254), UINT16_C(25054), UINT16_C(61521), UINT16_C( 7882), UINT16_C(20461), UINT16_C( 2990), UINT16_C(61790), UINT16_C(59387) },
      { UINT16_C(35821), UINT16_C(  569), UINT16_C(18319), UINT16_C(41965), UINT16_C(39709), UINT16_C(24514), UINT16_C(50302), UINT16_C(19512),
        UINT16_C( 5836), UINT16_C( 7597), UINT16_C(30470), UINT16_C(62268), UINT16_C(60102), UINT16_C( 9471), UINT16_C(64219), UINT16_C(51212) },
      { UINT16_C(35196), UINT16_C(  181), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(44654), UINT16_C(    0),
        UINT16_C(  200), UINT16_C( 2904), UINT16_C(28603), UINT16_C( 7488), UINT16_C(    0), UINT16_C(  432), UINT16_C(60548), UINT16_C(46406) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mulhi_epu16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mulhi_epu16");
    easysimd_test_x86_assert_equal_u16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__m256i r = easysimd_mm256_maskz_mulhi_epu16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mulhi_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 16752),  INT16_C(  3096),  INT16_C( 21789),  INT16_C( 27232), -INT16_C( 17038),  INT16_C( 13798), -INT16_C(  6599), -INT16_C( 28056),
        -INT16_C( 14632),  INT16_C( 22925), -INT16_C( 11459),  INT16_C( 11763),  INT16_C( 13417),  INT16_C( 15127),  INT16_C(  4707), -INT16_C(  3144),
        -INT16_C( 12079), -INT16_C(  4353),  INT16_C( 24613), -INT16_C( 26792),  INT16_C( 15901),  INT16_C( 22476),  INT16_C( 13349), -INT16_C(   535),
         INT16_C( 30459),  INT16_C( 14422),  INT16_C( 18761), -INT16_C( 19867),  INT16_C( 31870), -INT16_C(  7698), -INT16_C( 22898),  INT16_C( 24533) },
      { -INT16_C( 11145), -INT16_C( 25523), -INT16_C( 22988),  INT16_C( 21044),  INT16_C(   228),  INT16_C(  2473), -INT16_C( 28107),  INT16_C( 12294),
         INT16_C( 23560),  INT16_C( 20840), -INT16_C( 12890),  INT16_C(  9220), -INT16_C(  3511), -INT16_C( 10491), -INT16_C(  9576),  INT16_C(  3895),
        -INT16_C( 31569), -INT16_C(  7252), -INT16_C(  8150),  INT16_C(  3893), -INT16_C(  8480),  INT16_C(  5400),  INT16_C(  8048),  INT16_C( 30789),
        -INT16_C( 21125),  INT16_C(  8650), -INT16_C( 12678), -INT16_C( 15547),  INT16_C( 19392),  INT16_C( 22683), -INT16_C( 11739), -INT16_C( 11160) },
      {  INT16_C(  2848), -INT16_C(  1206), -INT16_C(  7643),  INT16_C(  8744), -INT16_C(    60),  INT16_C(   520),  INT16_C(  2830), -INT16_C(  5264),
        -INT16_C(  5261),  INT16_C(  7289),  INT16_C(  2253),  INT16_C(  1654), -INT16_C(   719), -INT16_C(  2422), -INT16_C(   688), -INT16_C(   187),
         INT16_C(  5818),  INT16_C(   481), -INT16_C(  3061), -INT16_C(  1592), -INT16_C(  2058),  INT16_C(  1851),  INT16_C(  1639), -INT16_C(   252),
        -INT16_C(  9819),  INT16_C(  1903), -INT16_C(  3630),  INT16_C(  4713),  INT16_C(  9430), -INT16_C(  2665),  INT16_C(  4101), -INT16_C(  4178) } },
    { {  INT16_C(  5206), -INT16_C( 32328), -INT16_C(  4620), -INT16_C( 11120), -INT16_C( 22324),  INT16_C( 15594),  INT16_C( 12231),  INT16_C( 17333),
         INT16_C( 32733),  INT16_C( 22372), -INT16_C( 21939),  INT16_C(  3355), -INT16_C( 18699),  INT16_C(  6757), -INT16_C( 12920), -INT16_C(  8465),
        -INT16_C( 22559), -INT16_C( 10913), -INT16_C(  4204),  INT16_C( 24746), -INT16_C( 27496),  INT16_C( 24477),  INT16_C( 21187), -INT16_C( 24414),
         INT16_C(  2001),  INT16_C(  7928),  INT16_C(  5041), -INT16_C( 22997), -INT16_C( 28471),  INT16_C( 20928), -INT16_C( 20642),  INT16_C( 16175) },
      { -INT16_C( 28842), -INT16_C(  5355), -INT16_C( 16514),  INT16_C(  5707), -INT16_C(  6061),  INT16_C(  5750),  INT16_C(  6202),  INT16_C(  2999),
        -INT16_C( 20705), -INT16_C( 12247),  INT16_C( 21698), -INT16_C( 29834),  INT16_C( 14309),  INT16_C( 17372),  INT16_C(  3046),  INT16_C( 15746),
        -INT16_C( 26726),  INT16_C(  6440),  INT16_C( 29526), -INT16_C( 22225), -INT16_C( 23204), -INT16_C( 26944),  INT16_C( 30654), -INT16_C(  8798),
        -INT16_C( 13530), -INT16_C(  5970),  INT16_C(  9248),  INT16_C(  1395),  INT16_C( 20315),  INT16_C( 16968), -INT16_C( 13734), -INT16_C(  2689) },
      { -INT16_C(  2292),  INT16_C(  2641),  INT16_C(  1164), -INT16_C(   969),  INT16_C(  2064),  INT16_C(  1368),  INT16_C(  1157),  INT16_C(   793),
        -INT16_C( 10342), -INT16_C(  4181), -INT16_C(  7264), -INT16_C(  1528), -INT16_C(  4083),  INT16_C(  1791), -INT16_C(   601), -INT16_C(  2034),
         INT16_C(  9199), -INT16_C(  1073), -INT16_C(  1895), -INT16_C(  8393),  INT16_C(  9735), -INT16_C( 10064),  INT16_C(  9910),  INT16_C(  3277),
        -INT16_C(   414), -INT16_C(   723),  INT16_C(   711), -INT16_C(   490), -INT16_C(  8826),  INT16_C(  5418),  INT16_C(  4325), -INT16_C(   664) } },
    { { -INT16_C( 22686), -INT16_C( 18418),  INT16_C( 15642),  INT16_C( 30306),  INT16_C(  8931), -INT16_C( 24307), -INT16_C( 20583), -INT16_C( 16514),
         INT16_C( 11386), -INT16_C( 25945),  INT16_C(  6737), -INT16_C( 21345), -INT16_C(  6295), -INT16_C( 15378),  INT16_C( 28082),  INT16_C(  5304),
        -INT16_C( 14828),  INT16_C( 12236),  INT16_C( 11780), -INT16_C(  6235), -INT16_C( 19888), -INT16_C(  5752),  INT16_C(  1633), -INT16_C(  9048),
         INT16_C( 20275), -INT16_C( 31626),  INT16_C(  5737), -INT16_C( 11728),  INT16_C(  8189), -INT16_C( 20586),  INT16_C( 20108), -INT16_C( 24125) },
      { -INT16_C( 28651),  INT16_C(  6608),  INT16_C( 30142),  INT16_C(  3840), -INT16_C( 30680), -INT16_C( 30216), -INT16_C( 24178), -INT16_C( 16027),
        -INT16_C(  8976),  INT16_C( 23109),  INT16_C( 30450), -INT16_C(  4308), -INT16_C( 15723),  INT16_C(  8607),  INT16_C( 25105),  INT16_C(  9922),
        -INT16_C( 27918), -INT16_C( 20161),  INT16_C( 16136),  INT16_C( 12480), -INT16_C( 18233),  INT16_C( 21945),  INT16_C(  8025),  INT16_C( 18967),
         INT16_C( 23803), -INT16_C(  4700), -INT16_C( 12078),  INT16_C( 26588),  INT16_C( 31635), -INT16_C( 23415),  INT16_C( 19422), -INT16_C( 12086) },
      {  INT16_C(  9917), -INT16_C(  1858),  INT16_C(  7194),  INT16_C(  1775), -INT16_C(  4181),  INT16_C( 11206),  INT16_C(  7593),  INT16_C(  4038),
        -INT16_C(  1560), -INT16_C(  9149),  INT16_C(  3130),  INT16_C(  1403),  INT16_C(  1510), -INT16_C(  2020),  INT16_C( 10757),  INT16_C(   803),
         INT16_C(  6316), -INT16_C(  3765),  INT16_C(  2900), -INT16_C(  1188),  INT16_C(  5533), -INT16_C(  1927),  INT16_C(   199), -INT16_C(  2619),
         INT16_C(  7363),  INT16_C(  2268), -INT16_C(  1058), -INT16_C(  4759),  INT16_C(  3952),  INT16_C(  7355),  INT16_C(  5959),  INT16_C(  4449) } },
    { {  INT16_C(  2526), -INT16_C(  6527),  INT16_C( 16712),  INT16_C(  3862), -INT16_C( 12294),  INT16_C( 21348),  INT16_C( 31726), -INT16_C(  5731),
         INT16_C( 16856), -INT16_C( 21802), -INT16_C( 19694), -INT16_C( 23278), -INT16_C( 25810),  INT16_C(  3145),  INT16_C(  5094), -INT16_C( 15139),
         INT16_C( 24092),  INT16_C( 25770), -INT16_C( 16224), -INT16_C( 25997), -INT16_C( 10352),  INT16_C( 32493), -INT16_C( 29869),  INT16_C( 11112),
         INT16_C( 16076), -INT16_C(  8491), -INT16_C(  6159),  INT16_C(  8323), -INT16_C( 13182),  INT16_C( 26924),  INT16_C(  2527), -INT16_C(  1235) },
      { -INT16_C( 10136),  INT16_C(  2143), -INT16_C( 11624),  INT16_C( 10402), -INT16_C( 28758), -INT16_C(   601),  INT16_C(  3866), -INT16_C(  6360),
        -INT16_C(   691),  INT16_C( 16325),  INT16_C( 18917),  INT16_C( 26463), -INT16_C( 29931), -INT16_C(  2608), -INT16_C(   363), -INT16_C(   528),
         INT16_C( 20694),  INT16_C( 28165), -INT16_C( 22750), -INT16_C( 13161),  INT16_C( 15926),  INT16_C( 20937), -INT16_C(  3763), -INT16_C( 26056),
        -INT16_C(   529), -INT16_C( 11047),  INT16_C( 14406),  INT16_C( 23611),  INT16_C(  3268),  INT16_C( 22865),  INT16_C( 16650), -INT16_C(  8106) },
      { -INT16_C(   391), -INT16_C(   214), -INT16_C(  2965),  INT16_C(   612),  INT16_C(  5394), -INT16_C(   196),  INT16_C(  1871),  INT16_C(   556),
        -INT16_C(   178), -INT16_C(  5431), -INT16_C(  5685), -INT16_C(  9400),  INT16_C( 11787), -INT16_C(   126), -INT16_C(    29),  INT16_C(   121),
         INT16_C(  7607),  INT16_C( 11075),  INT16_C(  5631),  INT16_C(  5220), -INT16_C(  2516),  INT16_C( 10380),  INT16_C(  1715), -INT16_C(  4418),
        -INT16_C(   130),  INT16_C(  1431), -INT16_C(  1354),  INT16_C(  2998), -INT16_C(   658),  INT16_C(  9393),  INT16_C(   642),  INT16_C(   152) } },
    { {  INT16_C( 23441), -INT16_C( 19378), -INT16_C(  6910),  INT16_C( 14464),  INT16_C( 18979),  INT16_C( 28809), -INT16_C( 16069),  INT16_C( 10763),
        -INT16_C(  6977),  INT16_C(  1534),  INT16_C( 14877), -INT16_C(  7839), -INT16_C( 19898),  INT16_C( 20538), -INT16_C( 28428), -INT16_C( 31440),
         INT16_C( 32491), -INT16_C(  4807), -INT16_C( 17820), -INT16_C( 30939), -INT16_C( 20732),  INT16_C( 16376),  INT16_C(   880),  INT16_C( 12138),
         INT16_C( 26855),  INT16_C(  1077), -INT16_C( 26974), -INT16_C(  5915),  INT16_C(  8009),  INT16_C( 15672),  INT16_C( 26799), -INT16_C( 25918) },
      { -INT16_C(   793),  INT16_C( 19335), -INT16_C( 21066), -INT16_C( 17710), -INT16_C( 13732), -INT16_C( 13063),  INT16_C( 25549), -INT16_C( 18948),
         INT16_C( 12748),  INT16_C( 28345), -INT16_C( 24633),  INT16_C(  4183), -INT16_C( 28738),  INT16_C( 28237),  INT16_C(  4344), -INT16_C(  8440),
        -INT16_C( 28660), -INT16_C( 15830), -INT16_C(   963), -INT16_C( 26244),  INT16_C( 30151), -INT16_C( 27547),  INT16_C( 25049), -INT16_C( 23223),
         INT16_C(   914),  INT16_C( 23059),  INT16_C( 27298),  INT16_C( 24682), -INT16_C( 18182), -INT16_C(  3378), -INT16_C( 10296), -INT16_C( 11055) },
      { -INT16_C(   284), -INT16_C(  5718),  INT16_C(  2221), -INT16_C(  3909), -INT16_C(  3977), -INT16_C(  5743), -INT16_C(  6265), -INT16_C(  3112),
        -INT16_C(  1358),  INT16_C(   663), -INT16_C(  5592), -INT16_C(   501),  INT16_C(  8725),  INT16_C(  8849), -INT16_C(  1885),  INT16_C(  4048),
        -INT16_C( 14209),  INT16_C(  1161),  INT16_C(   261),  INT16_C( 12389), -INT16_C(  9539), -INT16_C(  6884),  INT16_C(   336), -INT16_C(  4302),
         INT16_C(   374),  INT16_C(   378), -INT16_C( 11236), -INT16_C(  2228), -INT16_C(  2222), -INT16_C(   808), -INT16_C(  4211),  INT16_C(  4372) } },
    { { -INT16_C(  1177), -INT16_C( 23402),  INT16_C(  4855), -INT16_C( 16835), -INT16_C( 23929),  INT16_C( 24659), -INT16_C( 25596), -INT16_C( 27131),
         INT16_C(  6559),  INT16_C( 16880),  INT16_C( 23427),  INT16_C( 32162),  INT16_C( 28691), -INT16_C(  9361),  INT16_C( 16455), -INT16_C( 20817),
         INT16_C( 17723),  INT16_C( 13138), -INT16_C( 28841), -INT16_C(  8463),  INT16_C( 17458),  INT16_C( 13887),  INT16_C( 17633), -INT16_C( 32564),
        -INT16_C( 17059), -INT16_C(  7742),  INT16_C( 25624),  INT16_C( 11102), -INT16_C( 12588),  INT16_C(  7174), -INT16_C( 19186),  INT16_C( 19146) },
      {  INT16_C(  7674),  INT16_C( 20861),  INT16_C( 28332), -INT16_C(  8657),  INT16_C( 28339), -INT16_C( 27628), -INT16_C(  7757),  INT16_C(  4116),
        -INT16_C( 10594), -INT16_C( 18703),  INT16_C( 20538),  INT16_C(  4065), -INT16_C(  6370),  INT16_C( 11307), -INT16_C(  2660), -INT16_C( 27018),
        -INT16_C(  3310), -INT16_C( 16409),  INT16_C(  5730),  INT16_C(  5533), -INT16_C( 19835),  INT16_C( 14505), -INT16_C( 17005),  INT16_C( 12616),
         INT16_C( 14996), -INT16_C( 12569), -INT16_C( 14198), -INT16_C( 22307),  INT16_C(  2223),  INT16_C( 19412),  INT16_C( 19454),  INT16_C(  4321) },
      { -INT16_C(   138), -INT16_C(  7450),  INT16_C(  2098),  INT16_C(  2223), -INT16_C( 10348), -INT16_C( 10396),  INT16_C(  3029), -INT16_C(  1704),
        -INT16_C(  1061), -INT16_C(  4818),  INT16_C(  7341),  INT16_C(  1994), -INT16_C(  2789), -INT16_C(  1616), -INT16_C(   668),  INT16_C(  8582),
        -INT16_C(   896), -INT16_C(  3290), -INT16_C(  2522), -INT16_C(   715), -INT16_C(  5284),  INT16_C(  3073), -INT16_C(  4576), -INT16_C(  6269),
        -INT16_C(  3904),  INT16_C(  1484), -INT16_C(  5552), -INT16_C(  3779), -INT16_C(   427),  INT16_C(  2124), -INT16_C(  5696),  INT16_C(  1262) } },
    { { -INT16_C( 14274), -INT16_C( 24369),  INT16_C( 28126),  INT16_C( 25525),  INT16_C( 24095), -INT16_C( 19813), -INT16_C(  7140), -INT16_C( 20253),
        -INT16_C( 13794), -INT16_C( 22402),  INT16_C( 23698),  INT16_C( 16720),  INT16_C(  9316),  INT16_C( 25228),  INT16_C( 28015), -INT16_C( 20877),
         INT16_C( 16949),  INT16_C(  4942),  INT16_C(  1199), -INT16_C( 12681),  INT16_C(  4706),  INT16_C( 32384),  INT16_C( 25590),  INT16_C(  5166),
        -INT16_C( 21203), -INT16_C( 16452),  INT16_C(  3081),  INT16_C( 27904), -INT16_C( 29647), -INT16_C( 24368),  INT16_C( 17401),  INT16_C( 11854) },
      { -INT16_C( 25211),  INT16_C( 13634), -INT16_C( 18015),  INT16_C(   771), -INT16_C( 31541), -INT16_C( 15742), -INT16_C( 20249),  INT16_C(  5590),
        -INT16_C( 27811),  INT16_C( 26324), -INT16_C( 10849), -INT16_C( 12076), -INT16_C( 23455),  INT16_C( 23409), -INT16_C( 16409),  INT16_C( 27785),
        -INT16_C( 13476), -INT16_C(   607), -INT16_C( 23164),  INT16_C( 20481), -INT16_C( 31959),  INT16_C(  4114), -INT16_C(  6093), -INT16_C( 28379),
        -INT16_C(  1413),  INT16_C(  7159), -INT16_C( 13361),  INT16_C( 12523),  INT16_C( 23663),  INT16_C( 22155),  INT16_C(  5404),  INT16_C( 30915) },
      {  INT16_C(  5491), -INT16_C(  5070), -INT16_C(  7732),  INT16_C(   300), -INT16_C( 11597),  INT16_C(  4759),  INT16_C(  2206), -INT16_C(  1728),
         INT16_C(  5853), -INT16_C(  8999), -INT16_C(  3924), -INT16_C(  3081), -INT16_C(  3335),  INT16_C(  9011), -INT16_C(  7015), -INT16_C(  8852),
        -INT16_C(  3486), -INT16_C(    46), -INT16_C(   424), -INT16_C(  3964), -INT16_C(  2295),  INT16_C(  2032), -INT16_C(  2380), -INT16_C(  2238),
         INT16_C(   457), -INT16_C(  1798), -INT16_C(   629),  INT16_C(  5332), -INT16_C( 10705), -INT16_C(  8238),  INT16_C(  1434),  INT16_C(  5591) } },
    { {  INT16_C( 25824),  INT16_C( 25974),  INT16_C( 30473),  INT16_C( 12981), -INT16_C( 14342),  INT16_C( 11587),  INT16_C( 26799),  INT16_C( 11198),
        -INT16_C( 18846),  INT16_C( 12614),  INT16_C( 12673), -INT16_C(  3742), -INT16_C(  4722), -INT16_C( 21945),  INT16_C(  2562), -INT16_C(  7390),
        -INT16_C( 26513),  INT16_C( 30792), -INT16_C(   753),  INT16_C(  2475), -INT16_C(  4412),  INT16_C( 29495), -INT16_C(  2730), -INT16_C( 18018),
        -INT16_C(  6997),  INT16_C( 11754),  INT16_C( 19478), -INT16_C( 23522),  INT16_C( 25914),  INT16_C( 15438),  INT16_C( 28784), -INT16_C(  8417) },
      {  INT16_C( 26377),  INT16_C(  6231),  INT16_C(   612),  INT16_C( 10274),  INT16_C( 23024),  INT16_C( 18332),  INT16_C( 14926), -INT16_C(  1536),
        -INT16_C(  5601),  INT16_C( 13607),  INT16_C( 17719),  INT16_C( 29145),  INT16_C( 10154),  INT16_C(  6829), -INT16_C( 12905), -INT16_C( 24327),
         INT16_C( 20788), -INT16_C( 26183), -INT16_C(  9389),  INT16_C( 17601),  INT16_C( 23860), -INT16_C( 32117), -INT16_C( 29800), -INT16_C( 18564),
        -INT16_C( 23691), -INT16_C( 21268), -INT16_C( 14872), -INT16_C( 27875), -INT16_C( 13332), -INT16_C( 31827), -INT16_C( 22632), -INT16_C( 13276) },
      {  INT16_C( 10393),  INT16_C(  2469),  INT16_C(   284),  INT16_C(  2035), -INT16_C(  5039),  INT16_C(  3241),  INT16_C(  6103), -INT16_C(   263),
         INT16_C(  1610),  INT16_C(  2618),  INT16_C(  3426), -INT16_C(  1665), -INT16_C(   732), -INT16_C(  2287), -INT16_C(   505),  INT16_C(  2743),
        -INT16_C(  8410), -INT16_C( 12303),  INT16_C(   107),  INT16_C(   664), -INT16_C(  1607), -INT16_C( 14455),  INT16_C(  1241),  INT16_C(  5103),
         INT16_C(  2529), -INT16_C(  3815), -INT16_C(  4421),  INT16_C( 10004), -INT16_C(  5272), -INT16_C(  7498), -INT16_C(  9941),  INT16_C(  1705) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mulhi_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mulhi_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mulhi_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint16_t a[32];
    const uint16_t b[32];
    const uint16_t r[32];
  } test_vec[] = {
    { { UINT16_C(13528), UINT16_C(42352), UINT16_C(54241), UINT16_C(36909), UINT16_C(32493), UINT16_C(27360), UINT16_C(58793), UINT16_C(20231),
        UINT16_C( 9730), UINT16_C(14872), UINT16_C(37059), UINT16_C(64471), UINT16_C(16040), UINT16_C(36791), UINT16_C(34617), UINT16_C( 4521),
        UINT16_C( 6587), UINT16_C(40118), UINT16_C(58349), UINT16_C(55853), UINT16_C( 3426), UINT16_C( 2884), UINT16_C(19698), UINT16_C(62811),
        UINT16_C(29554), UINT16_C(13615), UINT16_C( 1796), UINT16_C(44081), UINT16_C(59461), UINT16_C(32315), UINT16_C(58480), UINT16_C(11151) },
      { UINT16_C(18174), UINT16_C(60360), UINT16_C(62761), UINT16_C(35781), UINT16_C( 2562), UINT16_C(62871), UINT16_C(62038), UINT16_C(51434),
        UINT16_C( 6501), UINT16_C(27133), UINT16_C(11808), UINT16_C(26134), UINT16_C(20759), UINT16_C(34788), UINT16_C(29750), UINT16_C(13490),
        UINT16_C(31418), UINT16_C(58143), UINT16_C(58479), UINT16_C(29295), UINT16_C( 1774), UINT16_C(17511), UINT16_C(20984), UINT16_C(23820),
        UINT16_C( 2666), UINT16_C(35783), UINT16_C(56632), UINT16_C(20465), UINT16_C(54574), UINT16_C(25814), UINT16_C(35145), UINT16_C(  920) },
      { UINT16_C( 3751), UINT16_C(39007), UINT16_C(51944), UINT16_C(20151), UINT16_C( 1270), UINT16_C(26247), UINT16_C(55654), UINT16_C(15877),
        UINT16_C(  965), UINT16_C( 6157), UINT16_C( 6677), UINT16_C(25709), UINT16_C( 5080), UINT16_C(19529), UINT16_C(15714), UINT16_C(  930),
        UINT16_C( 3157), UINT16_C(35592), UINT16_C(52065), UINT16_C(24966), UINT16_C(   92), UINT16_C(  770), UINT16_C( 6307), UINT16_C(22829),
        UINT16_C( 1202), UINT16_C( 7433), UINT16_C( 1551), UINT16_C(13765), UINT16_C(49515), UINT16_C(12728), UINT16_C(31361), UINT16_C(  156) } },
    { { UINT16_C(46851), UINT16_C(29671), UINT16_C(22172), UINT16_C(35557), UINT16_C(19548), UINT16_C(21711), UINT16_C(56221), UINT16_C( 1969),
        UINT16_C(30949), UINT16_C( 7826), UINT16_C(33621), UINT16_C(33901), UINT16_C(17497), UINT16_C(41704), UINT16_C(33229), UINT16_C(53414),
        UINT16_C(36152), UINT16_C(54339), UINT16_C(10467), UINT16_C(16223), UINT16_C(11892), UINT16_C( 4499), UINT16_C(17417), UINT16_C(61209),
        UINT16_C(43965), UINT16_C( 4621), UINT16_C(31279), UINT16_C(34966), UINT16_C(32702), UINT16_C(35626), UINT16_C(53248), UINT16_C(14428) },
      { UINT16_C(40797), UINT16_C(16397), UINT16_C(27848), UINT16_C(15487), UINT16_C( 4762), UINT16_C(41806), UINT16_C(26455), UINT16_C( 5266),
        UINT16_C(40722), UINT16_C(16678), UINT16_C(48410), UINT16_C(55497), UINT16_C(62524), UINT16_C(15460), UINT16_C(49348), UINT16_C( 8820),
        UINT16_C(33119), UINT16_C(10082), UINT16_C(58093), UINT16_C(34660), UINT16_C(45812), UINT16_C(19243), UINT16_C(48409), UINT16_C(11103),
        UINT16_C(34397), UINT16_C(30573), UINT16_C(13891), UINT16_C(32591), UINT16_C(45866), UINT16_C(61371), UINT16_C(12147), UINT16_C(54033) },
      { UINT16_C(29165), UINT16_C( 7423), UINT16_C( 9421), UINT16_C( 8402), UINT16_C( 1420), UINT16_C(13849), UINT16_C(22694), UINT16_C(  158),
        UINT16_C(19230), UINT16_C( 1991), UINT16_C(24835), UINT16_C(28707), UINT16_C(16692), UINT16_C( 9838), UINT16_C(25021), UINT16_C( 7188),
        UINT16_C(18269), UINT16_C( 8359), UINT16_C( 9278), UINT16_C( 8579), UINT16_C( 8312), UINT16_C( 1321), UINT16_C(12865), UINT16_C(10369),
        UINT16_C(23075), UINT16_C( 2155), UINT16_C( 6629), UINT16_C(17388), UINT16_C(22886), UINT16_C(33361), UINT16_C( 9869), UINT16_C(11895) } },
    { { UINT16_C(29617), UINT16_C(40698), UINT16_C(24149), UINT16_C(18982), UINT16_C(20752), UINT16_C(10645), UINT16_C(62734), UINT16_C(27477),
        UINT16_C(49787), UINT16_C(48866), UINT16_C(13048), UINT16_C( 9021), UINT16_C(63717), UINT16_C(22802), UINT16_C( 8999), UINT16_C(55340),
        UINT16_C( 9878), UINT16_C(60535), UINT16_C(40325), UINT16_C(38198), UINT16_C(52206), UINT16_C(64703), UINT16_C( 5312), UINT16_C(15208),
        UINT16_C(19158), UINT16_C(52985), UINT16_C(13948), UINT16_C(25329), UINT16_C(  814), UINT16_C(22203), UINT16_C(59174), UINT16_C(48430) },
      { UINT16_C(42253), UINT16_C(37545), UINT16_C(57154), UINT16_C(12328), UINT16_C(59306), UINT16_C(27437), UINT16_C(38395), UINT16_C(53670),
        UINT16_C(41183), UINT16_C(23711), UINT16_C(37334), UINT16_C( 1470), UINT16_C(31124), UINT16_C(47963), UINT16_C(35168), UINT16_C(28024),
        UINT16_C( 8495), UINT16_C(28928), UINT16_C(10240), UINT16_C(43682), UINT16_C(53007), UINT16_C( 2581), UINT16_C(48228), UINT16_C(17371),
        UINT16_C(31324), UINT16_C(12959), UINT16_C(23819), UINT16_C(41015), UINT16_C(37590), UINT16_C(13915), UINT16_C(54044), UINT16_C(19364) },
      { UINT16_C(19094), UINT16_C(23315), UINT16_C(21060), UINT16_C( 3570), UINT16_C(18779), UINT16_C( 4456), UINT16_C(36753), UINT16_C(22501),
        UINT16_C(31286), UINT16_C(17679), UINT16_C( 7433), UINT16_C(  202), UINT16_C(30260), UINT16_C(16687), UINT16_C( 4829), UINT16_C(23664),
        UINT16_C( 1280), UINT16_C(26720), UINT16_C( 6300), UINT16_C(25460), UINT16_C(42225), UINT16_C( 2548), UINT16_C( 3909), UINT16_C( 4031),
        UINT16_C( 9156), UINT16_C(10477), UINT16_C( 5069), UINT16_C(15851), UINT16_C(  466), UINT16_C( 4714), UINT16_C(48797), UINT16_C(14309) } },
    { { UINT16_C(42228), UINT16_C(62652), UINT16_C(24268), UINT16_C(56222), UINT16_C(46125), UINT16_C(37349), UINT16_C(49264), UINT16_C(52437),
        UINT16_C(29754), UINT16_C(18174), UINT16_C(14034), UINT16_C(43238), UINT16_C(16840), UINT16_C(58591), UINT16_C(33556), UINT16_C( 2095),
        UINT16_C(60455), UINT16_C(62460), UINT16_C(39498), UINT16_C(30926), UINT16_C(45902), UINT16_C(48649), UINT16_C(56947), UINT16_C(44426),
        UINT16_C(35155), UINT16_C( 9715), UINT16_C(55743), UINT16_C(34765), UINT16_C(44058), UINT16_C(11884), UINT16_C(39727), UINT16_C(22070) },
      { UINT16_C(12935), UINT16_C(53833), UINT16_C( 6093), UINT16_C( 6986), UINT16_C(21450), UINT16_C(15834), UINT16_C(25650), UINT16_C(34283),
        UINT16_C(57069), UINT16_C(44202), UINT16_C(30648), UINT16_C(53812), UINT16_C(40996), UINT16_C(21249), UINT16_C(14139), UINT16_C(50090),
        UINT16_C(62314), UINT16_C(14229), UINT16_C(57099), UINT16_C(54610), UINT16_C(11314), UINT16_C(25619), UINT16_C(65169), UINT16_C(32489),
        UINT16_C(37852), UINT16_C(37931), UINT16_C(24331), UINT16_C(12135), UINT16_C(26879), UINT16_C(14978), UINT16_C(11423), UINT16_C( 2557) },
      { UINT16_C( 8334), UINT16_C(51464), UINT16_C( 2256), UINT16_C( 5993), UINT16_C(15096), UINT16_C( 9023), UINT16_C(19281), UINT16_C(27430),
        UINT16_C(25909), UINT16_C(12257), UINT16_C( 6563), UINT16_C(35502), UINT16_C(10534), UINT16_C(18997), UINT16_C( 7239), UINT16_C( 1601),
        UINT16_C(57482), UINT16_C(13561), UINT16_C(34413), UINT16_C(25770), UINT16_C( 7924), UINT16_C(19017), UINT16_C(56628), UINT16_C(22023),
        UINT16_C(20304), UINT16_C( 5622), UINT16_C(20695), UINT16_C( 6437), UINT16_C(18069), UINT16_C( 2716), UINT16_C( 6924), UINT16_C(  861) } },
    { { UINT16_C(37408), UINT16_C(11072), UINT16_C(37745), UINT16_C(41984), UINT16_C( 5055), UINT16_C(20488), UINT16_C(61969), UINT16_C(61135),
        UINT16_C(64133), UINT16_C(36994), UINT16_C(59737), UINT16_C(22719), UINT16_C(16977), UINT16_C(61842), UINT16_C(36974), UINT16_C(36602),
        UINT16_C(15138), UINT16_C(38073), UINT16_C(47822), UINT16_C(36152), UINT16_C(16589), UINT16_C(57310), UINT16_C(44338), UINT16_C(47309),
        UINT16_C(20391), UINT16_C(   72), UINT16_C( 2105), UINT16_C(35416), UINT16_C(59978), UINT16_C(47227), UINT16_C(30330), UINT16_C(40263) },
      { UINT16_C(  177), UINT16_C(32561), UINT16_C(27066), UINT16_C(34828), UINT16_C(60073), UINT16_C(56423), UINT16_C(13463), UINT16_C(16020),
        UINT16_C(56451), UINT16_C(48190), UINT16_C(38628), UINT16_C(11847), UINT16_C(49793), UINT16_C(64487), UINT16_C(11832), UINT16_C(59800),
        UINT16_C(51502), UINT16_C(59752), UINT16_C(30002), UINT16_C(56433), UINT16_C(55391), UINT16_C(63416), UINT16_C(19468), UINT16_C(36661),
        UINT16_C(29736), UINT16_C( 3404), UINT16_C(37642), UINT16_C(35643), UINT16_C( 8789), UINT16_C(36487), UINT16_C( 8016), UINT16_C(32631) },
      { UINT16_C(  101), UINT16_C( 5501), UINT16_C(15588), UINT16_C(22311), UINT16_C( 4633), UINT16_C(17639), UINT16_C(12730), UINT16_C(14944),
        UINT16_C(55242), UINT16_C(27202), UINT16_C(35209), UINT16_C( 4106), UINT16_C(12898), UINT16_C(60852), UINT16_C( 6675), UINT16_C(33398),
        UINT16_C(11896), UINT16_C(34712), UINT16_C(21892), UINT16_C(31130), UINT16_C(14021), UINT16_C(55456), UINT16_C(13170), UINT16_C(26464),
        UINT16_C( 9252), UINT16_C(    3), UINT16_C( 1209), UINT16_C(19261), UINT16_C( 8043), UINT16_C(26293), UINT16_C( 3709), UINT16_C(20047) } },
    { { UINT16_C(57577), UINT16_C( 7016), UINT16_C(55637), UINT16_C(46327), UINT16_C(44977), UINT16_C(48555), UINT16_C(57851), UINT16_C( 9292),
        UINT16_C(38997), UINT16_C(24369), UINT16_C(27691), UINT16_C(33259), UINT16_C(29327), UINT16_C(57103), UINT16_C(34449), UINT16_C(31326),
        UINT16_C(50790), UINT16_C(48022), UINT16_C(36255), UINT16_C(20592), UINT16_C( 6973), UINT16_C(14349), UINT16_C(23292), UINT16_C(20828),
        UINT16_C(36338), UINT16_C( 7857), UINT16_C(40186), UINT16_C(35231), UINT16_C(44558), UINT16_C(40808), UINT16_C(50996), UINT16_C(39706) },
      { UINT16_C(45197), UINT16_C(11606), UINT16_C(50749), UINT16_C(31357), UINT16_C(35810), UINT16_C(57011), UINT16_C( 4069), UINT16_C(55088),
        UINT16_C(57757), UINT16_C(38901), UINT16_C(38013), UINT16_C(35616), UINT16_C(34882), UINT16_C(30506), UINT16_C(17487), UINT16_C(56594),
        UINT16_C(26868), UINT16_C(12810), UINT16_C(34607), UINT16_C( 4524), UINT16_C(24338), UINT16_C(63471), UINT16_C( 8047), UINT16_C( 3279),
        UINT16_C(50176), UINT16_C(32163), UINT16_C(50009), UINT16_C(39688), UINT16_C(13131), UINT16_C(39698), UINT16_C( 9335), UINT16_C(27768) },
      { UINT16_C(39708), UINT16_C( 1242), UINT16_C(43083), UINT16_C(22166), UINT16_C(24576), UINT16_C(42238), UINT16_C( 3591), UINT16_C( 7810),
        UINT16_C(34368), UINT16_C(14465), UINT16_C(16061), UINT16_C(18074), UINT16_C(15609), UINT16_C(26580), UINT16_C( 9192), UINT16_C(27051),
        UINT16_C(20822), UINT16_C( 9386), UINT16_C(19144), UINT16_C( 1421), UINT16_C( 2589), UINT16_C(13896), UINT16_C( 2859), UINT16_C( 1042),
        UINT16_C(27821), UINT16_C( 3855), UINT16_C(30665), UINT16_C(21335), UINT16_C( 8927), UINT16_C(24719), UINT16_C( 7263), UINT16_C(16823) } },
    { { UINT16_C(33421), UINT16_C(48286), UINT16_C(18953), UINT16_C( 7373), UINT16_C(48298), UINT16_C( 6419), UINT16_C(58076), UINT16_C(56357),
        UINT16_C(51367), UINT16_C(   90), UINT16_C(25227), UINT16_C(54939), UINT16_C(44693), UINT16_C( 3441), UINT16_C(59858), UINT16_C(24441),
        UINT16_C( 5995), UINT16_C(29979), UINT16_C(59489), UINT16_C( 2961), UINT16_C(42149), UINT16_C(33060), UINT16_C(18823), UINT16_C(11869),
        UINT16_C(46865), UINT16_C(39982), UINT16_C(51482), UINT16_C(44915), UINT16_C(58487), UINT16_C(19132), UINT16_C(13774), UINT16_C(14761) },
      { UINT16_C(50508), UINT16_C(44718), UINT16_C(16301), UINT16_C(21177), UINT16_C(57060), UINT16_C(27603), UINT16_C(12583), UINT16_C(14745),
        UINT16_C(51176), UINT16_C(  725), UINT16_C(18576), UINT16_C( 2226), UINT16_C(28205), UINT16_C(64338), UINT16_C(64420), UINT16_C(61492),
        UINT16_C(58304), UINT16_C(28318), UINT16_C(22562), UINT16_C( 1728), UINT16_C(37942), UINT16_C(23921), UINT16_C( 2757), UINT16_C(44438),
        UINT16_C(27857), UINT16_C(25264), UINT16_C(25268), UINT16_C(57706), UINT16_C(48336), UINT16_C(29916), UINT16_C( 4535), UINT16_C(30821) },
      { UINT16_C(25757), UINT16_C(32947), UINT16_C( 4714), UINT16_C( 2382), UINT16_C(42051), UINT16_C( 2703), UINT16_C(11150), UINT16_C(12679),
        UINT16_C(40111), UINT16_C(    0), UINT16_C( 7150), UINT16_C( 1866), UINT16_C(19234), UINT16_C( 3378), UINT16_C(58838), UINT16_C(22932),
        UINT16_C( 5333), UINT16_C(12953), UINT16_C(20480), UINT16_C(   78), UINT16_C(24402), UINT16_C(12067), UINT16_C(  791), UINT16_C( 8048),
        UINT16_C(19920), UINT16_C(15412), UINT16_C(19849), UINT16_C(39548), UINT16_C(43137), UINT16_C( 8733), UINT16_C(  953), UINT16_C( 6941) } },
    { { UINT16_C( 1012), UINT16_C( 5862), UINT16_C(42587), UINT16_C(37149), UINT16_C(36410), UINT16_C(65519), UINT16_C(34201), UINT16_C(27309),
        UINT16_C(24049), UINT16_C(42700), UINT16_C(14015), UINT16_C(36743), UINT16_C(25842), UINT16_C(43524), UINT16_C(26997), UINT16_C(26914),
        UINT16_C( 2156), UINT16_C(51327), UINT16_C(40110), UINT16_C(59737), UINT16_C(18475), UINT16_C(50408), UINT16_C(38350), UINT16_C(48942),
        UINT16_C(64498), UINT16_C(45413), UINT16_C(60721), UINT16_C( 9281), UINT16_C(17745), UINT16_C(50894), UINT16_C(61614), UINT16_C( 6703) },
      { UINT16_C(44792), UINT16_C(42722), UINT16_C(15435), UINT16_C(30351), UINT16_C(30852), UINT16_C(21050), UINT16_C(26637), UINT16_C(   18),
        UINT16_C(30563), UINT16_C(38321), UINT16_C(62052), UINT16_C(46521), UINT16_C(34615), UINT16_C(58747), UINT16_C(43639), UINT16_C(28416),
        UINT16_C(57945), UINT16_C(42005), UINT16_C(42270), UINT16_C(41754), UINT16_C(21533), UINT16_C(10997), UINT16_C( 1980), UINT16_C( 8234),
        UINT16_C(56447), UINT16_C(58293), UINT16_C(28366), UINT16_C( 1689), UINT16_C( 5365), UINT16_C(27883), UINT16_C(60351), UINT16_C( 6363) },
      { UINT16_C(  691), UINT16_C( 3821), UINT16_C(10030), UINT16_C(17204), UINT16_C(17140), UINT16_C(21044), UINT16_C(13900), UINT16_C(    7),
        UINT16_C(11215), UINT16_C(24968), UINT16_C(13269), UINT16_C(26082), UINT16_C(13649), UINT16_C(39015), UINT16_C(17976), UINT16_C(11669),
        UINT16_C( 1906), UINT16_C(32897), UINT16_C(25870), UINT16_C(38059), UINT16_C( 6070), UINT16_C( 8458), UINT16_C( 1158), UINT16_C( 6149),
        UINT16_C(55552), UINT16_C(40393), UINT16_C(26281), UINT16_C(  239), UINT16_C( 1452), UINT16_C(21653), UINT16_C(56739), UINT16_C(  650) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mulhi_epu16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mulhi_epu16");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mulhi_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(   323161906), -INT32_C(  1299196267), -INT32_C(   421690070),  INT32_C(   184944524), -INT32_C(  1292911645),  INT32_C(   515409961),  INT32_C(  1535812508),  INT32_C(  1567843727),
        -INT32_C(   347525290),  INT32_C(   798939397), -INT32_C(   367625378),  INT32_C(  1727405186),  INT32_C(    18408920),  INT32_C(   153145453),  INT32_C(    40217203),  INT32_C(  1717557264) },
      {  INT32_C(   223455496), -INT32_C(   532811902), -INT32_C(   305507478),  INT32_C(  1196671087),  INT32_C(   306801573), -INT32_C(  1373869765),  INT32_C(   598835475),  INT32_C(  1636372569),
         INT32_C(   997186489),  INT32_C(   891006154),  INT32_C(  1847781119),  INT32_C(  1287026086),  INT32_C(   459210720),  INT32_C(  2076801640),  INT32_C(  1436515067),  INT32_C(  1152788619) },
      { -INT32_C(    16813238),  INT32_C(   161171712),  INT32_C(    29995448),  INT32_C(    51529557), -INT32_C(    92356309), -INT32_C(   164868814),  INT32_C(   214134113),  INT32_C(   597344820),
        -INT32_C(    80686884),  INT32_C(   165742803), -INT32_C(   158159815),  INT32_C(   517632704),  INT32_C(     1968250),  INT32_C(    74052421),  INT32_C(    13451235),  INT32_C(   461000126) } },
    { { -INT32_C(   830462717), -INT32_C(   805069871),  INT32_C(   675227009),  INT32_C(  2054485402),  INT32_C(  1553322740),  INT32_C(  1222073933),  INT32_C(  1688041177), -INT32_C(  1566026593),
         INT32_C(  1265641593),  INT32_C(  1159427012),  INT32_C(   846027416),  INT32_C(  1135403343),  INT32_C(    27214260),  INT32_C(  2034857888), -INT32_C(  1931614227), -INT32_C(  1271954117) },
      {  INT32_C(  1929355182), -INT32_C(  1414063341), -INT32_C(   975297163), -INT32_C(  1173845498),  INT32_C(  1824237772),  INT32_C(   216335647),  INT32_C(   664388332), -INT32_C(   170145721),
         INT32_C(  2053626727),  INT32_C(  1797660662),  INT32_C(  1244660804),  INT32_C(  1510291854),  INT32_C(    13025505), -INT32_C(  1324569659), -INT32_C(  1244093075), -INT32_C(   710232979) },
      { -INT32_C(   373054656),  INT32_C(   265059012), -INT32_C(   153329919), -INT32_C(   561505659),  INT32_C(   659755900),  INT32_C(    61555335),  INT32_C(   261123027),  INT32_C(    62038359),
         INT32_C(   605163025),  INT32_C(   485278742),  INT32_C(   245174664),  INT32_C(   399255757),  INT32_C(       82533), -INT32_C(   627551000),  INT32_C(   559517155),  INT32_C(   210335422) } },
    { { -INT32_C(  2058415473),  INT32_C(  1978692913),  INT32_C(   146743673),  INT32_C(   996328538),  INT32_C(  1245391237),  INT32_C(  1123829716),  INT32_C(  1526191341),  INT32_C(   388997512),
        -INT32_C(   459505741),  INT32_C(  1851362804),  INT32_C(   141957550),  INT32_C(  1648613597), -INT32_C(   693273087), -INT32_C(  1307006523),  INT32_C(   101519229),  INT32_C(  1679637680) },
      { -INT32_C(  1337410885), -INT32_C(   182541753), -INT32_C(  1728211781),  INT32_C(  1861959788), -INT32_C(  2092652354), -INT32_C(   835298223),  INT32_C(   466895723),  INT32_C(   998240895),
        -INT32_C(   202651476),  INT32_C(   635963754),  INT32_C(   163440029), -INT32_C(   478693339), -INT32_C(  1318667424), -INT32_C(  2105500649),  INT32_C(  1604211935), -INT32_C(   224780986) },
      {  INT32_C(   640970482), -INT32_C(    84097049), -INT32_C(    59046817),  INT32_C(   431929638), -INT32_C(   606796449), -INT32_C(   218565801),  INT32_C(   165908646),  INT32_C(    90411218),
         INT32_C(    21681076),  INT32_C(   274134715),  INT32_C(     5402030), -INT32_C(   183745369),  INT32_C(   212852991),  INT32_C(   640727365),  INT32_C(    37918416), -INT32_C(    87905353) } },
    { {  INT32_C(  1340442085),  INT32_C(   729075086), -INT32_C(   684445006),  INT32_C(  1270525162),  INT32_C(  2147230055), -INT32_C(  1660846915),  INT32_C(   402431953), -INT32_C(  1576364355),
        -INT32_C(  1443696869),  INT32_C(  1876190909), -INT32_C(  2092496744),  INT32_C(   483262900), -INT32_C(   526661086),  INT32_C(   410885191), -INT32_C(   114329284),  INT32_C(   714815759) },
      { -INT32_C(   422343383), -INT32_C(  1940543500),  INT32_C(  1678744751), -INT32_C(  1065296482), -INT32_C(   274719832), -INT32_C(   217637449), -INT32_C(  1511246186), -INT32_C(  1714452368),
         INT32_C(   159359509), -INT32_C(   107555767),  INT32_C(   257795441),  INT32_C(   735042947), -INT32_C(  1357221640),  INT32_C(   614670733), -INT32_C(   943091881),  INT32_C(   761305367) },
      { -INT32_C(   131811678), -INT32_C(   329409242), -INT32_C(   267524380), -INT32_C(   315133014), -INT32_C(   137343696),  INT32_C(    84159543), -INT32_C(   141601487),  INT32_C(   629248470),
        -INT32_C(    53566607), -INT32_C(    46984096), -INT32_C(   125597260),  INT32_C(    82705865),  INT32_C(   166426371),  INT32_C(    58803498),  INT32_C(    25104503),  INT32_C(   126704823) } },
    { { -INT32_C(  2060001477),  INT32_C(   629066932), -INT32_C(   181085326), -INT32_C(  1340079176),  INT32_C(    23018099), -INT32_C(  1306197413), -INT32_C(  1451561326), -INT32_C(  1009329529),
         INT32_C(  1850215866),  INT32_C(  1267975897),  INT32_C(  1497417633),  INT32_C(  1040801994), -INT32_C(   180393830), -INT32_C(    39295893), -INT32_C(   626646446), -INT32_C(  1231192580) },
      {  INT32_C(  1663362442),  INT32_C(  1286584235),  INT32_C(  1235611518), -INT32_C(   360206768), -INT32_C(  2115975658),  INT32_C(  2088667178), -INT32_C(  1504303702),  INT32_C(   744354722),
        -INT32_C(  2087747112), -INT32_C(  1211154887),  INT32_C(  2113958958),  INT32_C(   963151650),  INT32_C(  2008696909),  INT32_C(  2062825936),  INT32_C(     2181726),  INT32_C(   372014654) },
      { -INT32_C(   797800973),  INT32_C(   188440922), -INT32_C(    52096117),  INT32_C(   112388652), -INT32_C(    11340189), -INT32_C(   635211278),  INT32_C(   508406450), -INT32_C(   174925477),
        -INT32_C(   899374213), -INT32_C(   357561560),  INT32_C(   737020610),  INT32_C(   233401115), -INT32_C(    84367704), -INT32_C(    18873389), -INT32_C(      318320), -INT32_C(   106641483) } },
    { {  INT32_C(   949664767),  INT32_C(   686844410),  INT32_C(    10940638), -INT32_C(   986116488),  INT32_C(   658371671), -INT32_C(  1952304851), -INT32_C(  1182022789),  INT32_C(  1087420225),
         INT32_C(  1819896434), -INT32_C(  1315673645), -INT32_C(   776848807), -INT32_C(  1600656567),  INT32_C(   214422751), -INT32_C(  2137560827),  INT32_C(  1832526380),  INT32_C(  1286474457) },
      {  INT32_C(  1203251060), -INT32_C(   369537648), -INT32_C(   793007225),  INT32_C(  1987072662),  INT32_C(   729954342), -INT32_C(   827647327),  INT32_C(   373024060),  INT32_C(  1667426799),
        -INT32_C(  1599399408), -INT32_C(   276126617), -INT32_C(   440449713), -INT32_C(  1118097257),  INT32_C(   166256232),  INT32_C(   886543352),  INT32_C(  1749685113),  INT32_C(   231517436) },
      {  INT32_C(   266052116), -INT32_C(    59095880), -INT32_C(     2020040), -INT32_C(   456228181),  INT32_C(   111894044),  INT32_C(   376212385), -INT32_C(   102660372),  INT32_C(   422167038),
        -INT32_C(   677709766),  INT32_C(    84585629),  INT32_C(    79665992),  INT32_C(   416694608),  INT32_C(     8300207), -INT32_C(   441223463),  INT32_C(   746535166),  INT32_C(    69346574) } },
    { {  INT32_C(   783120327),  INT32_C(  1780299547),  INT32_C(   324001148),  INT32_C(  1976609549), -INT32_C(  2122401655), -INT32_C(   977971636),  INT32_C(  1697448041),  INT32_C(  1936914860),
        -INT32_C(  1952309136), -INT32_C(   722092201), -INT32_C(  1444461156),  INT32_C(  2032056560), -INT32_C(  1124426384),  INT32_C(  1552003059),  INT32_C(  1556197295),  INT32_C(   433009832) },
      { -INT32_C(  1415286444), -INT32_C(   847275471), -INT32_C(   814258209), -INT32_C(  1891068641),  INT32_C(   625754674), -INT32_C(  1585328655),  INT32_C(   637354876), -INT32_C(   885076873),
         INT32_C(  1870062142),  INT32_C(  1530721916),  INT32_C(  2083173213),  INT32_C(  2064347721), -INT32_C(  1499375692), -INT32_C(  1589173723), -INT32_C(   590986139),  INT32_C(  1319568400) },
      { -INT32_C(   258055419), -INT32_C(   351202707), -INT32_C(    61425519), -INT32_C(   870298672), -INT32_C(   309223020),  INT32_C(   360982133),  INT32_C(   251894068), -INT32_C(   399145891),
        -INT32_C(   850050572), -INT32_C(   257352916), -INT32_C(   700602026),  INT32_C(   976694591),  INT32_C(   392537933), -INT32_C(   574254077), -INT32_C(   214132255),  INT32_C(   133036191) } },
    { {  INT32_C(  1673338599),  INT32_C(  1908406804), -INT32_C(   152180307),  INT32_C(   275970140),  INT32_C(  1974866768), -INT32_C(  1692992203),  INT32_C(  1383587137), -INT32_C(   929030175),
         INT32_C(  1361862205),  INT32_C(    96660312),  INT32_C(   821866452), -INT32_C(   146706777), -INT32_C(  1234372735),  INT32_C(   911311861),  INT32_C(  1099483488),  INT32_C(   621423080) },
      { -INT32_C(   545900921), -INT32_C(   169592799), -INT32_C(  1910054681), -INT32_C(   813275314),  INT32_C(  1401352798), -INT32_C(   712321163), -INT32_C(  2011753824), -INT32_C(  1028841157),
         INT32_C(  2023826263),  INT32_C(  1114474075), -INT32_C(  1244556442),  INT32_C(  1501845498), -INT32_C(  1079244215), -INT32_C(  2104215838), -INT32_C(  2079610039),  INT32_C(   591902924) },
      { -INT32_C(   212685458), -INT32_C(    75356116),  INT32_C(    67677513), -INT32_C(    52256441),  INT32_C(   644355330),  INT32_C(   280783086), -INT32_C(   648069362),  INT32_C(   222545228),
         INT32_C(   641721416),  INT32_C(    25081776), -INT32_C(   238152963), -INT32_C(    51299789),  INT32_C(   310174569), -INT32_C(   446475310), -INT32_C(   532366545),  INT32_C(    85640265) } }
  };
  
  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mulhi_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mulhi_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mulhi_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint32_t a[16];
    const uint32_t b[16];
    const uint32_t r[16];
  } test_vec[] = {
    { { UINT32_C(3170275046), UINT32_C(1713346016), UINT32_C(1587499631), UINT32_C( 648314611), UINT32_C( 654045001), UINT32_C( 531491943), UINT32_C(1262948465), UINT32_C( 345142062),
        UINT32_C(3788539905), UINT32_C(2370301725), UINT32_C( 904652353), UINT32_C(2925236068), UINT32_C(3872675454), UINT32_C(3003482434), UINT32_C(3221114001), UINT32_C(3235090623) },
      { UINT32_C( 916562713), UINT32_C(3569609107), UINT32_C( 873050063), UINT32_C(3185730878), UINT32_C(4272142011), UINT32_C(3383863352), UINT32_C(3028922612), UINT32_C(1500798016),
        UINT32_C(2475693568), UINT32_C(3462878207), UINT32_C(1090679042), UINT32_C(2449401046), UINT32_C(3549405595), UINT32_C(1033650505), UINT32_C( 837887473), UINT32_C(2190173826) },
      { UINT32_C( 676549015), UINT32_C(1423986522), UINT32_C( 322695507), UINT32_C( 480878137), UINT32_C( 650569127), UINT32_C( 418745006), UINT32_C( 890664095), UINT32_C( 120603600),
        UINT32_C(2183780044), UINT32_C(1911089333), UINT32_C( 229730587), UINT32_C(1668249323), UINT32_C(3200419229), UINT32_C( 722834638), UINT32_C( 628393858), UINT32_C(1649700758) } },
    { { UINT32_C(2064980860), UINT32_C(1900641390), UINT32_C(3283242221), UINT32_C(3411390512), UINT32_C(2594104401), UINT32_C( 383204133), UINT32_C(3796420960), UINT32_C(2875511599),
        UINT32_C(1546025454), UINT32_C(3821891574), UINT32_C(3970400187), UINT32_C(2159541295), UINT32_C( 102389472), UINT32_C(4045206161), UINT32_C(3939787963), UINT32_C( 630536247) },
      { UINT32_C(2827140018), UINT32_C(3867889450), UINT32_C(4275188431), UINT32_C( 260016430), UINT32_C(1880463839), UINT32_C(1180840331), UINT32_C(3442488982), UINT32_C( 552846702),
        UINT32_C(2882041216), UINT32_C(2475775172), UINT32_C(3046269830), UINT32_C(3435401708), UINT32_C( 909957546), UINT32_C(2692521482), UINT32_C(1131326932), UINT32_C(4083376498) },
      { UINT32_C(1359262975), UINT32_C(1711647673), UINT32_C(3268122477), UINT32_C( 206524874), UINT32_C(1135775707), UINT32_C( 105356540), UINT32_C(3042895655), UINT32_C( 370134856),
        UINT32_C(1037425612), UINT32_C(2203077140), UINT32_C(2816065750), UINT32_C(1727345365), UINT32_C(  21692847), UINT32_C(2535945849), UINT32_C(1037770004), UINT32_C( 599472991) } },
    { { UINT32_C(2594057430), UINT32_C( 103690112), UINT32_C(2126233746), UINT32_C(2068479953), UINT32_C(1672578904), UINT32_C(4194512421), UINT32_C(1295872475), UINT32_C(2822807762),
        UINT32_C(1279516364), UINT32_C(2673045773), UINT32_C(  35524145), UINT32_C(3866978446), UINT32_C( 357117935), UINT32_C( 940526941), UINT32_C(2441432254), UINT32_C(3107571436) },
      { UINT32_C(2986704036), UINT32_C( 508647661), UINT32_C(4112543591), UINT32_C(3353059032), UINT32_C( 719070669), UINT32_C( 811789170), UINT32_C( 616687415), UINT32_C(1390279597),
        UINT32_C(1694818935), UINT32_C(2726516027), UINT32_C(2643961029), UINT32_C( 258241090), UINT32_C( 154747287), UINT32_C(1681562412), UINT32_C( 814283650), UINT32_C(1854039542) },
      { UINT32_C(1803897739), UINT32_C(  12279891), UINT32_C(2035924458), UINT32_C(1614851734), UINT32_C( 280025981), UINT32_C( 792802254), UINT32_C( 186066200), UINT32_C( 913742007),
        UINT32_C( 504904557), UINT32_C(1696893512), UINT32_C(  21868491), UINT32_C( 232507644), UINT32_C(  12866927), UINT32_C( 368234410), UINT32_C( 462871595), UINT32_C(1341467798) } },
    { { UINT32_C(2194900551), UINT32_C(2686736091), UINT32_C(1010678778), UINT32_C(3310068270), UINT32_C( 265258211), UINT32_C(2725447967), UINT32_C(4224908036), UINT32_C(2825475168),
        UINT32_C(3039444186), UINT32_C(2371243922), UINT32_C( 952734474), UINT32_C( 419304757), UINT32_C(3106459033), UINT32_C(3663436758), UINT32_C(4157943191), UINT32_C(1537162881) },
      { UINT32_C( 219204218), UINT32_C( 597321241), UINT32_C( 794584058), UINT32_C( 306731640), UINT32_C(4257968167), UINT32_C(2732008971), UINT32_C(3566906707), UINT32_C(1714371051),
        UINT32_C( 477314819), UINT32_C(2688552358), UINT32_C(3922697328), UINT32_C( 502994934), UINT32_C(2468005511), UINT32_C(1060499948), UINT32_C(2316554142), UINT32_C( 217072137) },
      { UINT32_C( 112022147), UINT32_C( 373656986), UINT32_C( 186979129), UINT32_C( 236393573), UINT32_C( 262973135), UINT32_C(1733644934), UINT32_C(3508723529), UINT32_C(1127811342),
        UINT32_C( 337784120), UINT32_C(1484345048), UINT32_C( 870155397), UINT32_C(  49105884), UINT32_C(1785056202), UINT32_C( 904564394), UINT32_C(2242648164), UINT32_C(  77689818) } },
    { { UINT32_C( 657023873), UINT32_C(3771165040), UINT32_C(4224292613), UINT32_C( 907592878), UINT32_C(1992897162), UINT32_C(3266706979), UINT32_C(3612133582), UINT32_C(2363702282),
        UINT32_C( 263392415), UINT32_C(2062515061), UINT32_C(3228940562), UINT32_C( 150375805), UINT32_C(3816734655), UINT32_C(2359637182), UINT32_C( 123990524), UINT32_C(3432203821) },
      { UINT32_C(3369813587), UINT32_C(3544435393), UINT32_C(  26523779), UINT32_C(  84511302), UINT32_C( 149456714), UINT32_C(3096743355), UINT32_C(2881484670), UINT32_C(2423738941),
        UINT32_C(1515803288), UINT32_C(2687343645), UINT32_C(2594292052), UINT32_C(2527111756), UINT32_C(3986589746), UINT32_C(2493854230), UINT32_C(1715495977), UINT32_C(1341634486) },
      { UINT32_C( 515498214), UINT32_C(3112165918), UINT32_C(  26087324), UINT32_C(  17858542), UINT32_C(  69349040), UINT32_C(2355350444), UINT32_C(2423372944), UINT32_C(1333886120),
        UINT32_C(  92957887), UINT32_C(1290507321), UINT32_C(1950379190), UINT32_C(  88479478), UINT32_C(3542694085), UINT32_C(1370113149), UINT32_C(  49524299), UINT32_C(1072129935) } },
    { { UINT32_C( 665407498), UINT32_C(1086838508), UINT32_C(3839584664), UINT32_C(1165654803), UINT32_C( 422778883), UINT32_C(1940838474), UINT32_C(4091145789), UINT32_C(2940391589),
        UINT32_C( 215411488), UINT32_C(1515036354), UINT32_C( 440281095), UINT32_C(2791356579), UINT32_C( 448828368), UINT32_C(2827841131), UINT32_C(  27027036), UINT32_C(1471274551) },
      { UINT32_C(2355398602), UINT32_C( 753316133), UINT32_C(2085037273), UINT32_C(2887952348), UINT32_C(2781274682), UINT32_C(2890814288), UINT32_C(4037995193), UINT32_C(2454216648),
        UINT32_C( 203336934), UINT32_C( 909640797), UINT32_C(  78806824), UINT32_C(1622201382), UINT32_C( 117864119), UINT32_C(2209633481), UINT32_C( 108225086), UINT32_C(2811870145) },
      { UINT32_C( 364915442), UINT32_C( 190626127), UINT32_C(1863966960), UINT32_C( 783790723), UINT32_C( 273777219), UINT32_C(1306320445), UINT32_C(3846368526), UINT32_C(1680189275),
        UINT32_C(  10198240), UINT32_C( 320872961), UINT32_C(   8078560), UINT32_C(1054290332), UINT32_C(  12316918), UINT32_C(1454840517), UINT32_C(    681030), UINT32_C( 963228052) } },
    { { UINT32_C(3300112231), UINT32_C(3841649852), UINT32_C(2464787563), UINT32_C( 955423105), UINT32_C(3644848144), UINT32_C(2321347404), UINT32_C( 378654805), UINT32_C(4089326219),
        UINT32_C(2646045153), UINT32_C(3380785757), UINT32_C(3747310430), UINT32_C( 337071364), UINT32_C(2465093446), UINT32_C(2686274122), UINT32_C(2796989978), UINT32_C(3113841880) },
      { UINT32_C(1129795814), UINT32_C(1628231938), UINT32_C(1212180292), UINT32_C(4217198773), UINT32_C(4186786735), UINT32_C(2962860693), UINT32_C( 810963032), UINT32_C(2867523524),
        UINT32_C(1122910527), UINT32_C(1587804698), UINT32_C( 396813154), UINT32_C(3927049019), UINT32_C(3840188238), UINT32_C(2727640394), UINT32_C(2463361741), UINT32_C( 406633945) },
      { UINT32_C( 868098108), UINT32_C(1456378256), UINT32_C( 695643692), UINT32_C( 938123358), UINT32_C(3553042621), UINT32_C(1601369347), UINT32_C(  71496481), UINT32_C(2730227804),
        UINT32_C( 691803162), UINT32_C(1249841299), UINT32_C( 346214992), UINT32_C( 308196937), UINT32_C(2204073326), UINT32_C(1705994318), UINT32_C(1604202692), UINT32_C( 294808719) } },
    { { UINT32_C( 408562430), UINT32_C(2272722213), UINT32_C( 480124129), UINT32_C(1845997600), UINT32_C(2572348239), UINT32_C( 909960808), UINT32_C(2848460752), UINT32_C(3401712844),
        UINT32_C(1424104495), UINT32_C(4225456154), UINT32_C(2484566388), UINT32_C(2013470249), UINT32_C(1913804041), UINT32_C( 212356668), UINT32_C( 699822173), UINT32_C(2750642292) },
      { UINT32_C(2935477652), UINT32_C(2712261165), UINT32_C(1949745483), UINT32_C(3924638175), UINT32_C(3395026830), UINT32_C(2866217805), UINT32_C(3889401203), UINT32_C(2576074245),
        UINT32_C(3360195227), UINT32_C(2691297621), UINT32_C(2450890674), UINT32_C(1736114904), UINT32_C(1311888897), UINT32_C(1291323609), UINT32_C(2587085717), UINT32_C( 741654161) },
      { UINT32_C( 279239817), UINT32_C(1435218424), UINT32_C( 217957387), UINT32_C(1686828362), UINT32_C(2033354548), UINT32_C( 607256281), UINT32_C(2579485689), UINT32_C(2040310028),
        UINT32_C(1114157290), UINT32_C(2647740788), UINT32_C(1417799058), UINT32_C( 813886455), UINT32_C( 584567494), UINT32_C(  63847093), UINT32_C( 421539868), UINT32_C( 474980403) } }
  };
  
  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mulhi_epu32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mulhi_epu32");
    easysimd_test_x86_assert_equal_u32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mulhi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mulhi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mulhi_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mulhi_epu16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mulhi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mulhi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mulhi_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mulhi_epu16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mulhi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mulhi_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mulhi_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mulhi_epu32)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
