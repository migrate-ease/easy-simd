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

#define EASYSIMD_TEST_X86_AVX512_INSN or

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/or.h>

static int
test_easysimd_mm_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   306285541),  INT32_C(  1898534361),  INT32_C(   802101116), -INT32_C(    49634963) },
      { -INT32_C(  1991632176), -INT32_C(  1711363909), -INT32_C(  1175263633),  INT32_C(   692946701) },
      { -INT32_C(   302088485), -INT32_C(   100663813), -INT32_C(  1073744001), -INT32_C(    45111443) } },
    { { -INT32_C(   719975429), -INT32_C(   582598815), -INT32_C(  1022618026), -INT32_C(  2017454409) },
      { -INT32_C(   334493135),  INT32_C(   562368178),  INT32_C(    47937780), -INT32_C(    81057537) },
      { -INT32_C(    48886789), -INT32_C(    37273613), -INT32_C(  1008763146), -INT32_C(     1360129) } },
    { { -INT32_C(  1781513932), -INT32_C(   697166208), -INT32_C(   476479700), -INT32_C(   966108779) },
      {  INT32_C(   364018275),  INT32_C(  2100770953), -INT32_C(  1350626640),  INT32_C(  1856678458) },
      { -INT32_C(  1779270793), -INT32_C(     8962423), -INT32_C(   268468292), -INT32_C(   286590017) } },
    { {  INT32_C(  1795390187), -INT32_C(  1119783279), -INT32_C(  1969104139), -INT32_C(  1773073613) },
      {  INT32_C(   262931334), -INT32_C(   343088325),  INT32_C(   798690549), -INT32_C(  1566685770) },
      {  INT32_C(  1873771503), -INT32_C(     3278917), -INT32_C(  1346642187), -INT32_C(  1226879049) } },
    { {  INT32_C(  1359847872),  INT32_C(   202264343),  INT32_C(  1570221865),  INT32_C(  1106503867) },
      {  INT32_C(   659595243),  INT32_C(  1997724802), -INT32_C(  1616400920), -INT32_C(  1287567885) },
      {  INT32_C(  2002632683),  INT32_C(  2132729751), -INT32_C(   541610007), -INT32_C(   202117637) } },
    { { -INT32_C(    33272090), -INT32_C(   955641187),  INT32_C(  2099552706),  INT32_C(  1975392137) },
      {  INT32_C(   966528695), -INT32_C(   743330070),  INT32_C(  1332893788), -INT32_C(  2096974947) },
      { -INT32_C(     6533385), -INT32_C(   675561729),  INT32_C(  2138503646), -INT32_C(   138496099) } },
    { { -INT32_C(  1618934271), -INT32_C(   630813672), -INT32_C(  1218999763),  INT32_C(  1479284129) },
      {  INT32_C(   244500515), -INT32_C(   756989066),  INT32_C(   941708187),  INT32_C(   129704710) },
      { -INT32_C(  1617703389), -INT32_C(   622342274), -INT32_C(  1082664001),  INT32_C(  1606367143) } },
    { {  INT32_C(  1118190889), -INT32_C(   165933879),  INT32_C(   950891670), -INT32_C(  1383016055) },
      {  INT32_C(   414917281),  INT32_C(    15375461), -INT32_C(   180876049),  INT32_C(  1509749807) },
      {  INT32_C(  1522483113), -INT32_C(   151085843), -INT32_C(    37913345), -INT32_C(    33751633) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_or_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_or_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_or_epi32(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(  2126209185),  INT32_C(  1764647464), -INT32_C(     9783841), -INT32_C(  2090734537) },
      UINT8_C( 35),
      {  INT32_C(  1849464647), -INT32_C(   686069167), -INT32_C(  1789183774),  INT32_C(   909537493) },
      { -INT32_C(  1906395861), -INT32_C(  1737635869),  INT32_C(   516910130),  INT32_C(   356602830) },
      { -INT32_C(   293685393), -INT32_C(   545263629), -INT32_C(     9783841), -INT32_C(  2090734537) } },
    { {  INT32_C(  1015250666), -INT32_C(   904683544), -INT32_C(  1403031849),  INT32_C(   165844702) },
      UINT8_C( 75),
      {  INT32_C(   137271104),  INT32_C(  1899677189), -INT32_C(   398501739),  INT32_C(   399725977) },
      {  INT32_C(  2013204440), -INT32_C(  1873884638), -INT32_C(  1083245783),  INT32_C(   504002525) },
      {  INT32_C(  2147459032), -INT32_C(   243347929), -INT32_C(  1403031849),  INT32_C(   534476765) } },
    { {  INT32_C(   338049039), -INT32_C(  1803198210),  INT32_C(  1367131576), -INT32_C(   227979494) },
      UINT8_C( 94),
      {  INT32_C(   847276648), -INT32_C(  1285812040), -INT32_C(   158262402),  INT32_C(  1577430822) },
      {  INT32_C(   895293909),  INT32_C(  1676538270), -INT32_C(  1115865491),  INT32_C(   270233767) },
      {  INT32_C(   338049039), -INT32_C(   201461314), -INT32_C(      180353),  INT32_C(  1579155367) } },
    { { -INT32_C(  1841128230),  INT32_C(   726048428), -INT32_C(   534652998),  INT32_C(  1530799750) },
      UINT8_C( 63),
      { -INT32_C(  1931636581), -INT32_C(  1141227139),  INT32_C(   778287038),  INT32_C(  1846047698) },
      {  INT32_C(  1411095477), -INT32_C(  1207023903), -INT32_C(  1925255577), -INT32_C(   926115540) },
      { -INT32_C(   589325377), -INT32_C(  1140963331), -INT32_C(  1350565889), -INT32_C(   288556034) } },
    { { -INT32_C(  1504401111), -INT32_C(  1453240598), -INT32_C(   656948219), -INT32_C(   314122185) },
      UINT8_C(123),
      { -INT32_C(  1487126175),  INT32_C(  1024332879), -INT32_C(   345400494),  INT32_C(   286536039) },
      { -INT32_C(   721700218), -INT32_C(   522542053),  INT32_C(  1545056892),  INT32_C(  1524040952) },
      { -INT32_C(   134218777), -INT32_C(    35736481), -INT32_C(   656948219),  INT32_C(  1540830719) } },
    { { -INT32_C(  1811860667), -INT32_C(  1714352057),  INT32_C(   327433131), -INT32_C(   232482708) },
      UINT8_C( 82),
      { -INT32_C(   999438561),  INT32_C(  1396723105),  INT32_C(  1766562917), -INT32_C(  1498438285) },
      { -INT32_C(  1209187417),  INT32_C(  1348634133), -INT32_C(  1564707574), -INT32_C(  1175146599) },
      { -INT32_C(  1811860667),  INT32_C(  1398984629),  INT32_C(   327433131), -INT32_C(   232482708) } },
    { {  INT32_C(   394092918),  INT32_C(   342539695), -INT32_C(   864176807),  INT32_C(    41036891) },
      UINT8_C(112),
      { -INT32_C(   444221089), -INT32_C(  1846553317),  INT32_C(  1076531601), -INT32_C(   424221819) },
      {  INT32_C(   496356704), -INT32_C(   310990537),  INT32_C(  1397244455),  INT32_C(   331565748) },
      {  INT32_C(   394092918),  INT32_C(   342539695), -INT32_C(   864176807),  INT32_C(    41036891) } },
    { {  INT32_C(   519587843), -INT32_C(  1347426531), -INT32_C(    34612616),  INT32_C(   518301373) },
      UINT8_C(115),
      {  INT32_C(   598424441), -INT32_C(   196437838),  INT32_C(   715759072), -INT32_C(  1456620447) },
      { -INT32_C(  1664660555), -INT32_C(   720013573),  INT32_C(   194122341),  INT32_C(  1887416566) },
      { -INT32_C(  1074824195), -INT32_C(   178258181), -INT32_C(    34612616),  INT32_C(   518301373) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_or_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_or_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_or_epi32(src, k, a, b);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(158),
      {  INT32_C(  1820003848),  INT32_C(   655852144),  INT32_C(  1378693194),  INT32_C(  1027681672) },
      {  INT32_C(  1449310853),  INT32_C(   883081134),  INT32_C(   901187473),  INT32_C(  2027182192) },
      {  INT32_C(           0),  INT32_C(   934789118),  INT32_C(  2009020379),  INT32_C(  2111143416) } },
    { UINT8_C(102),
      { -INT32_C(   774446001),  INT32_C(   740031996),  INT32_C(  1471442218),  INT32_C(  1776087470) },
      {  INT32_C(   303510099), -INT32_C(   610054956),  INT32_C(  1531697155), -INT32_C(    71187540) },
      {  INT32_C(           0), -INT32_C(     4456964),  INT32_C(  1610612011),  INT32_C(           0) } },
    { UINT8_C(168),
      { -INT32_C(  1801139049),  INT32_C(  1438568680),  INT32_C(  1963201924),  INT32_C(   600337905) },
      { -INT32_C(   789063036),  INT32_C(  1439945341), -INT32_C(   536728035), -INT32_C(  2037842449) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1480000001) } },
    { UINT8_C(202),
      { -INT32_C(    55436756), -INT32_C(   310376488),  INT32_C(  2027877643), -INT32_C(  1745026627) },
      { -INT32_C(   904606216), -INT32_C(   806917471),  INT32_C(  1774176364),  INT32_C(  2100512080) },
      {  INT32_C(           0), -INT32_C(   270045191),  INT32_C(           0), -INT32_C(       47619) } },
    { UINT8_C( 96),
      { -INT32_C(   315065883), -INT32_C(   268949766),  INT32_C(   112029956),  INT32_C(  1006519406) },
      { -INT32_C(  1008940711),  INT32_C(  2016390320), -INT32_C(  1328965525), -INT32_C(  1324333364) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(191),
      {  INT32_C(  1874435657),  INT32_C(   141797783), -INT32_C(  1703511722),  INT32_C(  1106489720) },
      {  INT32_C(   988919438),  INT32_C(  2124770021),  INT32_C(  2034914611), -INT32_C(  1355220122) },
      {  INT32_C(  2147073743),  INT32_C(  2130177015), -INT32_C(    75596425), -INT32_C(   268698754) } },
    { UINT8_C(154),
      { -INT32_C(  1691279630),  INT32_C(   200358289),  INT32_C(  1619233711),  INT32_C(   904840319) },
      {  INT32_C(   521873589),  INT32_C(   592615885), -INT32_C(   561394717), -INT32_C(   159893500) },
      {  INT32_C(           0),  INT32_C(   737393117),  INT32_C(           0), -INT32_C(   134284161) } },
    { UINT8_C( 86),
      { -INT32_C(   488140375),  INT32_C(   227668610),  INT32_C(   965538421),  INT32_C(   166707936) },
      {  INT32_C(  1993739997),  INT32_C(   744094049), -INT32_C(  1171245182),  INT32_C(  1494230704) },
      {  INT32_C(           0),  INT32_C(   769260515), -INT32_C(  1145243657),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_or_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_or_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_or_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 5111787543517176625),  INT64_C( 1829040305974713899) },
      { -INT64_C( 6298535069951648463), -INT64_C(  814982468286427353) },
      { -INT64_C( 1227301542101795023), -INT64_C(  147879950383325393) } },
    { {  INT64_C( 3328968278238186300), -INT64_C( 1221243237492871043) },
      { -INT64_C( 9096394549512225182),  INT64_C( 3937422580223860972) },
      { -INT64_C( 5768021115797602434), -INT64_C(   23135923856477955) } },
    { { -INT64_C( 1223760346542337069), -INT64_C( 8164197999918717180) },
      {  INT64_C( 4686958951846721583),  INT64_C( 4610444037465433036) },
      { -INT64_C( 1220620119698588673), -INT64_C( 4612821346727761972) } },
    { {  INT64_C( 8469610024947490089), -INT64_C( 8161953684465881856) },
      { -INT64_C( 7225987327664364766),  INT64_C( 7615969876726884870) },
      { -INT64_C(   19637277687316693), -INT64_C( 1172063596197581050) } },
    { { -INT64_C( 2206325245768054918), -INT64_C( 6491718396377458181) },
      {  INT64_C( 2924758254730673963), -INT64_C( 3591819350457518210) },
      { -INT64_C( 1587555277167853701), -INT64_C( 1157482450660622337) } },
    { { -INT64_C( 5344617471821665518), -INT64_C( 3036561217602723405) },
      {  INT64_C( 5350389384091858934),  INT64_C( 4071307005896809054) },
      { -INT64_C(   12245269655977994), -INT64_C(  154257428168311809) } },
    { {  INT64_C(  561541840158538076), -INT64_C( 4080514353911113247) },
      { -INT64_C(  853677772023845103), -INT64_C( 4802765329012477702) },
      { -INT64_C(  580964528594927779), -INT64_C(   45247164066104837) } },
    { {  INT64_C( 7330626011753776223),  INT64_C( 2526043380936341878) },
      {  INT64_C(  578756588752399742),  INT64_C( 3780351687863610968) },
      {  INT64_C( 7907121987151129983),  INT64_C( 3998862041475448702) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_or_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_or_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_or_epi64(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 6732074665011631380), -INT64_C( 5943069286010700934) },
      UINT8_C( 68),
      {  INT64_C( 8153508047595981132), -INT64_C( 2574977400955259993) },
      { -INT64_C( 5781988103428316822),  INT64_C( 7946975522572442892) },
      { -INT64_C( 6732074665011631380), -INT64_C( 5943069286010700934) } },
    { {  INT64_C( 4930694602604710302),  INT64_C( 1664607468151166347) },
      UINT8_C(193),
      { -INT64_C( 4754705951335555401), -INT64_C( 2702635513291994996) },
      { -INT64_C( 4800066544590095489),  INT64_C( 8068622835631431512) },
      { -INT64_C( 4655599512256151553),  INT64_C( 1664607468151166347) } },
    { { -INT64_C( 7008048444104888609),  INT64_C( 7898748519106715249) },
      UINT8_C(120),
      {  INT64_C( 5055887237814764565), -INT64_C( 8509148683651030392) },
      {  INT64_C( 5021578074502523349),  INT64_C( 2751593891000185785) },
      { -INT64_C( 7008048444104888609),  INT64_C( 7898748519106715249) } },
    { {  INT64_C( 4022315554558218468),  INT64_C( 1471034682123604357) },
      UINT8_C( 19),
      {  INT64_C( 8573445230089097990),  INT64_C( 8194208128628601138) },
      { -INT64_C( 1924067194042301788), -INT64_C( 7145741914088330088) },
      { -INT64_C(  576743224656637018), -INT64_C(  146439595072438854) } },
    { { -INT64_C( 4537984037911949428), -INT64_C( 2460853015736447864) },
      UINT8_C(141),
      {  INT64_C( 2083345768867079417),  INT64_C( 1257154665329700012) },
      { -INT64_C( 7815046045617880532), -INT64_C( 4049112315938933515) },
      { -INT64_C( 6923184952705898755), -INT64_C( 2460853015736447864) } },
    { { -INT64_C( 6604245217168560051), -INT64_C( 8628619413177103187) },
      UINT8_C(254),
      {  INT64_C(  808282983290539232), -INT64_C( 5378529111501605019) },
      { -INT64_C(  115033162535561032), -INT64_C( 5271775801915042085) },
      { -INT64_C( 6604245217168560051), -INT64_C( 5197184928649054209) } },
    { { -INT64_C( 2624984550493135761), -INT64_C( 4350119847558288474) },
      UINT8_C(221),
      {  INT64_C( 8753536302782053454),  INT64_C(  250518394524094760) },
      { -INT64_C(   85193926877052419), -INT64_C( 6025591595141055171) },
      { -INT64_C(    1162236403941889), -INT64_C( 4350119847558288474) } },
    { {  INT64_C( 2784849635759042376), -INT64_C(  630032282069895914) },
      UINT8_C(173),
      { -INT64_C( 5671088612834270278),  INT64_C( 6799233869008588359) },
      {  INT64_C( 2513713575711442694),  INT64_C( 2021012741222572211) },
      { -INT64_C( 5481232599636793410), -INT64_C(  630032282069895914) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_or_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_or_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_mask_or_epi64(src, k, a, b);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(183),
      { -INT64_C( 4656643064977605384), -INT64_C( 7533772822546441773) },
      {  INT64_C( 8789658961381484671), -INT64_C( 4741302193709533611) },
      { -INT64_C(    1293240490132225), -INT64_C( 4651185084602662953) } },
    { UINT8_C(190),
      { -INT64_C( 6735332212670158484),  INT64_C( 8580697111969602795) },
      {  INT64_C( 4481981617597418938),  INT64_C( 5664617810463403867) },
      {  INT64_C(                   0),  INT64_C( 9195439974159204347) } },
    { UINT8_C(108),
      { -INT64_C( 6530473956037295576), -INT64_C( 7728269289392377021) },
      { -INT64_C( 9155154136749204123), -INT64_C( 7813551379269365920) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(178),
      {  INT64_C( 6440676770826618630), -INT64_C( 4456605238729546219) },
      {  INT64_C( 9209319932214623506), -INT64_C( 8513915993785175435) },
      {  INT64_C(                   0), -INT64_C( 3747276743310051723) } },
    { UINT8_C(133),
      {  INT64_C( 7929224899215468308), -INT64_C( 5715817851582690139) },
      { -INT64_C( 2561740137182585879), -INT64_C( 8877465196522661687) },
      { -INT64_C(  109530140052045827),  INT64_C(                   0) } },
    { UINT8_C(143),
      {  INT64_C( 5809931062999480759),  INT64_C(  421072176721050776) },
      { -INT64_C( 2451705024092318894), -INT64_C( 4349401865804431317) },
      { -INT64_C( 2451700072883160073), -INT64_C( 4037491436909855557) } },
    { UINT8_C( 76),
      { -INT64_C( 7144925211046920390),  INT64_C( 8189666855745865314) },
      {  INT64_C( 1256667922575054713),  INT64_C( 6989834178025915362) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(160),
      {  INT64_C( 4635220489025921488),  INT64_C( 2748970726665857058) },
      {  INT64_C( 6808046100486253317),  INT64_C(   47822365039865639) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_or_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_or_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_or_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_or_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t src[4];
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint32_t r[4];
  } test_vec[] = {
    { { UINT32_C(3277418579), UINT32_C( 341938906), UINT32_C(2038302947), UINT32_C( 880448879) },
      UINT8_C(214),
      { UINT32_C(3425443398), UINT32_C(   2711491), UINT32_C(3623675999), UINT32_C(3173683533) },
      { UINT32_C(2107108842), UINT32_C(1465953103), UINT32_C(3200703017), UINT32_C(2593454676) },
      { UINT32_C(3277418579), UINT32_C(1466564559), UINT32_C(4294900351), UINT32_C( 880448879) } },
    { { UINT32_C(3680944152), UINT32_C(2145095712), UINT32_C(3310803064), UINT32_C(1401061481) },
      UINT8_C(110),
      { UINT32_C(3300773913), UINT32_C( 200086577), UINT32_C(3579817179), UINT32_C(  32438592) },
      { UINT32_C(4011968863), UINT32_C(2103943333), UINT32_C(1994796278), UINT32_C(3353622958) },
      { UINT32_C(3680944152), UINT32_C(2146415797), UINT32_C(4160728319), UINT32_C(3354327534) } },
    { { UINT32_C( 999006474), UINT32_C(2437314998), UINT32_C(1701225765), UINT32_C(4251341982) },
      UINT8_C( 30),
      { UINT32_C( 667151751), UINT32_C(2166177876), UINT32_C(1596953638), UINT32_C( 443152248) },
      { UINT32_C(4224754818), UINT32_C(2401264106), UINT32_C( 472745671), UINT32_C(1949969388) },
      { UINT32_C( 999006474), UINT32_C(2403164670), UINT32_C(1596954343), UINT32_C(2122055676) } },
    { { UINT32_C(1822162200), UINT32_C(1676523837), UINT32_C(3317833037), UINT32_C(2531208212) },
      UINT8_C(208),
      { UINT32_C( 280728239), UINT32_C( 953698994), UINT32_C(2737173624), UINT32_C( 733714734) },
      { UINT32_C(3983025972), UINT32_C( 842713877), UINT32_C(3108372621), UINT32_C(2408242399) },
      { UINT32_C(1822162200), UINT32_C(1676523837), UINT32_C(3317833037), UINT32_C(2531208212) } },
    { { UINT32_C( 564086126), UINT32_C( 123303823), UINT32_C(2578087531), UINT32_C(1287939351) },
      UINT8_C(141),
      { UINT32_C(4154603820), UINT32_C(1954927732), UINT32_C(4132650522), UINT32_C( 224781000) },
      { UINT32_C(4187850370), UINT32_C(1583719647), UINT32_C(3027631695), UINT32_C(4014064067) },
      { UINT32_C(4290756526), UINT32_C( 123303823), UINT32_C(4135059039), UINT32_C(4016432075) } },
    { { UINT32_C(1877468155), UINT32_C(3521342647), UINT32_C(1942501034), UINT32_C(2608868633) },
      UINT8_C(179),
      { UINT32_C(3264386077), UINT32_C(4161925369), UINT32_C( 666617190), UINT32_C(3928140295) },
      { UINT32_C(4255289745), UINT32_C(2879943540), UINT32_C(1757682491), UINT32_C(3105578907) },
      { UINT32_C(4289893789), UINT32_C(4223267837), UINT32_C(1942501034), UINT32_C(2608868633) } },
    { { UINT32_C(3984305907), UINT32_C(  82152606), UINT32_C(1479319633), UINT32_C(3695398474) },
      UINT8_C(224),
      { UINT32_C(1481955813), UINT32_C(2627010433), UINT32_C( 574160067), UINT32_C(3323392280) },
      { UINT32_C(4167304044), UINT32_C(2286512616), UINT32_C(3822232213), UINT32_C(3401821925) },
      { UINT32_C(3984305907), UINT32_C(  82152606), UINT32_C(1479319633), UINT32_C(3695398474) } },
    { { UINT32_C( 153229448), UINT32_C(3685135895), UINT32_C(3422412467), UINT32_C( 999363535) },
      UINT8_C( 22),
      { UINT32_C(1593717749), UINT32_C( 519276156), UINT32_C( 117692249), UINT32_C(2995768730) },
      { UINT32_C(2798295536), UINT32_C( 492414271), UINT32_C(3068929186), UINT32_C(2882283445) },
      { UINT32_C( 153229448), UINT32_C( 536586111), UINT32_C(3085957115), UINT32_C( 999363535) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_mask_or_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_or_ps");
    easysimd_test_x86_assert_equal_u32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_mask_or_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

    easysimd_test_x86_write_u32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_or_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint32_t r[4];
  } test_vec[] = {
    { UINT8_C(114),
      { UINT32_C( 281127575), UINT32_C(1682805892), UINT32_C( 338832623), UINT32_C(3155712318) },
      { UINT32_C(2055584088), UINT32_C(3908524559), UINT32_C(1541520018), UINT32_C(2026829537) },
      { UINT32_C(         0), UINT32_C(3976191631), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(161),
      { UINT32_C( 589662351), UINT32_C(  17992149), UINT32_C( 356460219), UINT32_C(4167957567) },
      { UINT32_C(3674728321), UINT32_C(2574184415), UINT32_C(3329935824), UINT32_C( 661123735) },
      { UINT32_C(4213698447), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(123),
      { UINT32_C( 374360716), UINT32_C(2211533405), UINT32_C(2378360465), UINT32_C( 990820947) },
      { UINT32_C(2954554049), UINT32_C( 562082648), UINT32_C( 532236077), UINT32_C(1000005806) },
      { UINT32_C(3059411661), UINT32_C(2748445533), UINT32_C(         0), UINT32_C(1000274687) } },
    { UINT8_C( 42),
      { UINT32_C(1032278507), UINT32_C( 164498210), UINT32_C(2287754445), UINT32_C(1414109034) },
      { UINT32_C(1689057969), UINT32_C(3247558266), UINT32_C(1735438727), UINT32_C( 915516235) },
      { UINT32_C(         0), UINT32_C(3386888058), UINT32_C(         0), UINT32_C(1993981803) } },
    { UINT8_C(252),
      { UINT32_C( 605975321), UINT32_C(2666604354), UINT32_C( 453540228), UINT32_C(3184352707) },
      { UINT32_C(3610718473), UINT32_C(1952381379), UINT32_C( 348112233), UINT32_C(1880159831) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C( 532676077), UINT32_C(4259184599) } },
    { UINT8_C(105),
      { UINT32_C(1454085166), UINT32_C(4292495749), UINT32_C(2948789586), UINT32_C(4105732034) },
      { UINT32_C(2948042934), UINT32_C(3021482990), UINT32_C(3758828778), UINT32_C(1800043324) },
      { UINT32_C(4290745534), UINT32_C(         0), UINT32_C(         0), UINT32_C(4294606846) } },
    { UINT8_C( 15),
      { UINT32_C(1066713589), UINT32_C(2425459611), UINT32_C(3562225749), UINT32_C(2290829304) },
      { UINT32_C( 695679742), UINT32_C(2131962707), UINT32_C(2981950518), UINT32_C( 868230974) },
      { UINT32_C(1073216511), UINT32_C(4287871963), UINT32_C(4127192183), UINT32_C(3150669822) } },
    { UINT8_C(232),
      { UINT32_C(3900928596), UINT32_C(1128076035), UINT32_C(2906395238), UINT32_C(3618358429) },
      { UINT32_C(1714082875), UINT32_C(3701254888), UINT32_C(2367311206), UINT32_C(1651854605) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(4160736669) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_maskz_or_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_or_ps");
    easysimd_test_x86_assert_equal_u32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_maskz_or_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_or_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t src[2];
    const uint8_t k;
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { { UINT64_C(11537635906182052365), UINT64_C( 3478735107162750128) },
      UINT8_C(140),
      { UINT64_C(  172494427628378114), UINT64_C( 7577746654917642840) },
      { UINT64_C( 5399749493952573781), UINT64_C( 5935920577896253920) },
      { UINT64_C(11537635906182052365), UINT64_C( 3478735107162750128) } },
    { { UINT64_C( 1260536997545502392), UINT64_C(12138704384845267869) },
      UINT8_C(200),
      { UINT64_C(16230024003964037272), UINT64_C( 6298428603188187152) },
      { UINT64_C( 2766091308544091250), UINT64_C( 9109975990959501390) },
      { UINT64_C( 1260536997545502392), UINT64_C(12138704384845267869) } },
    { { UINT64_C( 8341535553083851530), UINT64_C(16928902870078845991) },
      UINT8_C(112),
      { UINT64_C( 9471062401933881599), UINT64_C( 5017264532945966195) },
      { UINT64_C( 4178808590542930625), UINT64_C( 9600438033915680320) },
      { UINT64_C( 8341535553083851530), UINT64_C(16928902870078845991) } },
    { { UINT64_C( 8926720537093606987), UINT64_C(17578026751291902761) },
      UINT8_C( 52),
      { UINT64_C( 6914911956805274266), UINT64_C(  885639015395869284) },
      { UINT64_C(15442719475078834449), UINT64_C( 9693993715362233249) },
      { UINT64_C( 8926720537093606987), UINT64_C(17578026751291902761) } },
    { { UINT64_C( 1868487750273728965), UINT64_C( 5764153695081714677) },
      UINT8_C( 12),
      { UINT64_C(13854037574325597458), UINT64_C( 2249040165943940725) },
      { UINT64_C( 9698200199063979118), UINT64_C(11130106100937165062) },
      { UINT64_C( 1868487750273728965), UINT64_C( 5764153695081714677) } },
    { { UINT64_C(17369960980384845449), UINT64_C( 6580207571278346773) },
      UINT8_C( 92),
      { UINT64_C( 2024812866880601125), UINT64_C(  645771742364795737) },
      { UINT64_C( 1012593395228689785), UINT64_C(13243210262363390166) },
      { UINT64_C(17369960980384845449), UINT64_C( 6580207571278346773) } },
    { { UINT64_C(14195877557299763282), UINT64_C( 6586841982276117607) },
      UINT8_C(110),
      { UINT64_C(11616981780984321260), UINT64_C( 5076096013714040010) },
      { UINT64_C(18230140659244881588), UINT64_C(13060783929061957546) },
      { UINT64_C(14195877557299763282), UINT64_C(17830309403115388906) } },
    { { UINT64_C(11281602634912760717), UINT64_C( 7036080215495066609) },
      UINT8_C(132),
      { UINT64_C( 6303588087819401560), UINT64_C(10555545341306818229) },
      { UINT64_C(13774543118062152682), UINT64_C( 9687188024410867901) },
      { UINT64_C(11281602634912760717), UINT64_C( 7036080215495066609) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_mask_or_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_or_pd");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_mask_or_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

    easysimd_test_x86_write_u64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_or_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { UINT8_C(243),
      { UINT64_C(11751481618011832135), UINT64_C(16629179652240816975) },
      { UINT64_C(16024300457197285003), UINT64_C(16105924055848486436) },
      { UINT64_C(18407872341460631503), UINT64_C(18430909842978226031) } },
    { UINT8_C(123),
      { UINT64_C(12869427323933352355), UINT64_C(16905494011861235290) },
      { UINT64_C( 1271463746606672321), UINT64_C( 6674486071761909097) },
      { UINT64_C(12951622451687057891), UINT64_C(18355803175824095099) } },
    { UINT8_C(106),
      { UINT64_C(12979114910920320353), UINT64_C(17704453658914074767) },
      { UINT64_C( 1058505171018920737), UINT64_C(15505086985592559162) },
      { UINT64_C(                   0), UINT64_C(17852263420841621183) } },
    { UINT8_C(200),
      { UINT64_C(16947814762316995727), UINT64_C(  623657106502263723) },
      { UINT64_C( 8500244816056151595), UINT64_C( 8475989894960495029) },
      { UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 95),
      { UINT64_C( 7930000340179097766), UINT64_C( 2995821990872650684) },
      { UINT64_C(18361805997787860781), UINT64_C(17314750995211069992) },
      { UINT64_C(18365465217063346095), UINT64_C(18004087817380412348) } },
    { UINT8_C(115),
      { UINT64_C(17448401119624450660), UINT64_C( 8136409280194317680) },
      { UINT64_C( 9476616886540437277), UINT64_C( 2118870756074811033) },
      { UINT64_C(17557209340366968701), UINT64_C( 9074750749445844985) } },
    { UINT8_C( 15),
      { UINT64_C( 4234052892898527013), UINT64_C( 8070674187877580685) },
      { UINT64_C( 5534867717532244609), UINT64_C( 5721297941864760217) },
      { UINT64_C( 9137782742332727205), UINT64_C( 9180283739385626525) } },
    { UINT8_C(169),
      { UINT64_C(16339781153824612821), UINT64_C(18390656498281457131) },
      { UINT64_C(  102220762053438981), UINT64_C(18373133047180280366) },
      { UINT64_C(16423424425407279061), UINT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_maskz_or_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_or_pd");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_maskz_or_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1115888701), -INT32_C(  2140190811),  INT32_C(    52283965), -INT32_C(  1071686441), -INT32_C(   130015812),  INT32_C(  2054130810), -INT32_C(   136342624), -INT32_C(   668548843) },
      {  INT32_C(  1234543268),  INT32_C(   667485417), -INT32_C(  1507137842), -INT32_C(    10073789), -INT32_C(   503863450), -INT32_C(   732207309),  INT32_C(   264977146), -INT32_C(  1612189189) },
      { -INT32_C(    33692697), -INT32_C(  1477490707), -INT32_C(  1488982273), -INT32_C(     8430121), -INT32_C(   101203970), -INT32_C(    25170053), -INT32_C(     2114566), -INT32_C(   538447361) } },
    { {  INT32_C(  2095611283),  INT32_C(  1352905089), -INT32_C(   604582248),  INT32_C(  2128239639),  INT32_C(   929026307),  INT32_C(   839629368), -INT32_C(   264120587),  INT32_C(  1519397319) },
      {  INT32_C(   668367014), -INT32_C(  1032357334),  INT32_C(  1604152648), -INT32_C(   841123894), -INT32_C(  2130428856), -INT32_C(   323809289), -INT32_C(  1394805275), -INT32_C(  1006212066) },
      {  INT32_C(  2147384759), -INT32_C(   755500117), -INT32_C(   536875048), -INT32_C(     2129953), -INT32_C(  1218446005), -INT32_C(    21250049), -INT32_C(    52561931), -INT32_C(   560566817) } },
    { {  INT32_C(   267116005), -INT32_C(  1613667497), -INT32_C(  1694601520),  INT32_C(   761847013),  INT32_C(   263089176),  INT32_C(  1643864700),  INT32_C(  1963907159),  INT32_C(   691606596) },
      {  INT32_C(  1211639025),  INT32_C(  1491536263),  INT32_C(  1559422583), -INT32_C(   628467006),  INT32_C(  1139423431), -INT32_C(   240851558),  INT32_C(    40284861), -INT32_C(  1205100601) },
      {  INT32_C(  1341914613), -INT32_C(   537433129), -INT32_C(   553718025), -INT32_C(     1384729),  INT32_C(  1341029599), -INT32_C(   234887170),  INT32_C(  2003761919), -INT32_C(  1187274809) } },
    { {  INT32_C(  1258382532), -INT32_C(   442242963), -INT32_C(  1874749746), -INT32_C(  1217672208), -INT32_C(  1627761404), -INT32_C(   124739782),  INT32_C(   419100241),  INT32_C(  1523656086) },
      { -INT32_C(   140127607), -INT32_C(  1982051910), -INT32_C(   820437537), -INT32_C(   309885719),  INT32_C(   344687065),  INT32_C(  1913396000), -INT32_C(  1467349486), -INT32_C(  1258136789) },
      { -INT32_C(     5900595), -INT32_C(   301995521), -INT32_C(   547774497), -INT32_C(     1060871), -INT32_C(  1627662883), -INT32_C(    90398918), -INT32_C(  1191512493), -INT32_C(     2916417) } },
    { { -INT32_C(   391338195), -INT32_C(   814642960), -INT32_C(  1902146907), -INT32_C(   394582513), -INT32_C(   923007065),  INT32_C(   876218402),  INT32_C(   987546638),  INT32_C(  1307565600) },
      {  INT32_C(  1983224710), -INT32_C(   934894045),  INT32_C(  1079502129), -INT32_C(  1305947637), -INT32_C(    59103783),  INT32_C(  1009824813), -INT32_C(  1720316551),  INT32_C(  1910924779) },
      { -INT32_C(    21119057), -INT32_C(   814240013), -INT32_C(   824184907), -INT32_C(    92539377), -INT32_C(    50452481),  INT32_C(  1010482223), -INT32_C(  1140929153),  INT32_C(  2112880619) } },
    { {  INT32_C(   602414080), -INT32_C(   202690878),  INT32_C(   506675731), -INT32_C(   288334827), -INT32_C(  1360376959),  INT32_C(  2028608511),  INT32_C(   319905832), -INT32_C(   981075771) },
      { -INT32_C(   689410796), -INT32_C(  1362439013),  INT32_C(   734854422), -INT32_C(   635790247), -INT32_C(   410450713),  INT32_C(  1197437727), -INT32_C(  1722060332),  INT32_C(  2103435369) },
      { -INT32_C(   135758572), -INT32_C(     1311013),  INT32_C(  1073741591), -INT32_C(    19211171), -INT32_C(   269791257),  INT32_C(  2147449855), -INT32_C(  1688505860), -INT32_C(    35653395) } },
    { { -INT32_C(   397195443),  INT32_C(   848698651),  INT32_C(  1952277019), -INT32_C(   430999554), -INT32_C(  1681008516),  INT32_C(   534981707),  INT32_C(   112803485),  INT32_C(  1803753502) },
      {  INT32_C(  2069092191),  INT32_C(   263055860),  INT32_C(  1233390155), -INT32_C(    30420094), -INT32_C(   157680469), -INT32_C(   971604951), -INT32_C(   657600582),  INT32_C(  1178816743) },
      { -INT32_C(    78391457),  INT32_C(  1069547007),  INT32_C(  2111662683), -INT32_C(    26216450), -INT32_C(     2228993), -INT32_C(   537428885), -INT32_C(   553779265),  INT32_C(  1875073279) } },
    { {  INT32_C(   482448935), -INT32_C(   903123329), -INT32_C(    82530439),  INT32_C(   771310466),  INT32_C(  1764004416), -INT32_C(   919651825), -INT32_C(   257754103),  INT32_C(  1949754701) },
      { -INT32_C(    74385284), -INT32_C(   540623770), -INT32_C(   287647125),  INT32_C(  1562104605),  INT32_C(  1975926630), -INT32_C(  2109802887),  INT32_C(  1064493298),  INT32_C(  1119136198) },
      { -INT32_C(     3015041), -INT32_C(   537919873), -INT32_C(     2162821),  INT32_C(  2113655711),  INT32_C(  2112274278), -INT32_C(   884998529), -INT32_C(      852741),  INT32_C(  1991699919) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_or_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_or_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_or_epi32(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1808559275),  INT32_C(   982492771),  INT32_C(  1767276795), -INT32_C(   904132915), -INT32_C(  1761284432), -INT32_C(  1299182893),  INT32_C(  1201951238),  INT32_C(    55545261) },
      UINT8_C( 33),
      {  INT32_C(   679778178), -INT32_C(  1490764249),  INT32_C(   443911444), -INT32_C(  1731575895),  INT32_C(  1298882883),  INT32_C(  1146298097),  INT32_C(  1358011330),  INT32_C(  1819407594) },
      { -INT32_C(  1282017908), -INT32_C(   950355533), -INT32_C(   270414010),  INT32_C(  1384622863), -INT32_C(    23006451), -INT32_C(   750587119),  INT32_C(  2015573134), -INT32_C(  1260022744) },
      { -INT32_C(  1147799666),  INT32_C(   982492771),  INT32_C(  1767276795), -INT32_C(   904132915), -INT32_C(  1761284432), -INT32_C(   682360847),  INT32_C(  1201951238),  INT32_C(    55545261) } },
    { {  INT32_C(  1013414537),  INT32_C(  2030289459), -INT32_C(  1603738479), -INT32_C(  1644957552), -INT32_C(   191130653),  INT32_C(   365420166),  INT32_C(   982379282),  INT32_C(   166687359) },
      UINT8_C(236),
      {  INT32_C(   404702550),  INT32_C(   766154825), -INT32_C(   239252991), -INT32_C(   791389635), -INT32_C(   732444426),  INT32_C(  2061921423),  INT32_C(  1828331769),  INT32_C(  1717043983) },
      { -INT32_C(  1853982648),  INT32_C(   297740304), -INT32_C(  1342014349), -INT32_C(   847128873),  INT32_C(   782358686),  INT32_C(  1034454852), -INT32_C(  1213619545), -INT32_C(   300088667) },
      {  INT32_C(  1013414537),  INT32_C(  2030289459), -INT32_C(   239108493), -INT32_C(   573186305), -INT32_C(   191130653),  INT32_C(  2146365391), -INT32_C(      285953), -INT32_C(   295894097) } },
    { { -INT32_C(  1971348614),  INT32_C(   932986564), -INT32_C(  1830314309),  INT32_C(   325019765), -INT32_C(  2076114880),  INT32_C(   784525959),  INT32_C(   853896076), -INT32_C(   417332627) },
      UINT8_C(158),
      { -INT32_C(   563973473), -INT32_C(  1399219954), -INT32_C(   383702144), -INT32_C(  1977010806),  INT32_C(  1611771510), -INT32_C(   605208720),  INT32_C(   659103525), -INT32_C(   557502401) },
      { -INT32_C(  1329846366),  INT32_C(  1096635840),  INT32_C(   170557056),  INT32_C(   714363827),  INT32_C(  1904911617),  INT32_C(   172849125), -INT32_C(   718170474),  INT32_C(  1773467590) },
      { -INT32_C(  1971348614), -INT32_C(   304226866), -INT32_C(   349470848), -INT32_C(  1430423621),  INT32_C(  1906028407),  INT32_C(   784525959),  INT32_C(   853896076), -INT32_C(      657409) } },
    { { -INT32_C(   551981026),  INT32_C(  1176532678), -INT32_C(  1471067403), -INT32_C(  1613568611),  INT32_C(  1880120715),  INT32_C(  1803181524), -INT32_C(  1170166797), -INT32_C(  1054608222) },
      UINT8_C(101),
      { -INT32_C(  1288986564),  INT32_C(   178811328), -INT32_C(  1465429822), -INT32_C(  2144123357), -INT32_C(  1269521577), -INT32_C(   928465123), -INT32_C(   194289152), -INT32_C(  1051120507) },
      { -INT32_C(  1938520884), -INT32_C(  1198121738), -INT32_C(  1872741011), -INT32_C(   619670652),  INT32_C(  1401906230),  INT32_C(   605828900),  INT32_C(   504924057),  INT32_C(  2145415859) },
      { -INT32_C(  1082153732),  INT32_C(  1176532678), -INT32_C(  1192788497), -INT32_C(  1613568611),  INT32_C(  1880120715), -INT32_C(   323174595), -INT32_C(    25434215), -INT32_C(  1054608222) } },
    { { -INT32_C(   334801674), -INT32_C(   559570575),  INT32_C(  1651377630), -INT32_C(   818053479),  INT32_C(   103009762), -INT32_C(  1641398524),  INT32_C(  2042381254), -INT32_C(  1409770315) },
      UINT8_C(241),
      { -INT32_C(  1537042429),  INT32_C(  1115832381),  INT32_C(   752608686), -INT32_C(   284251358),  INT32_C(   234100174), -INT32_C(  2100063681), -INT32_C(   348697522),  INT32_C(  1205658436) },
      { -INT32_C(  1192542597),  INT32_C(   754609534),  INT32_C(  1968756306),  INT32_C(  1331979905), -INT32_C(  1151575941),  INT32_C(   960311274), -INT32_C(  1088129669), -INT32_C(   754581416) },
      { -INT32_C(  1125400965), -INT32_C(   559570575),  INT32_C(  1651377630), -INT32_C(   818053479), -INT32_C(  1073979905), -INT32_C(  1140867073), -INT32_C(    13140609), -INT32_C(   673258660) } },
    { { -INT32_C(  1131679426), -INT32_C(  1310161570), -INT32_C(   601472933),  INT32_C(   573278886), -INT32_C(   841119773),  INT32_C(   822483638), -INT32_C(   403690865),  INT32_C(  1757083178) },
      UINT8_C(231),
      { -INT32_C(   884595643),  INT32_C(  1277622028), -INT32_C(  1477246179), -INT32_C(  1232464594),  INT32_C(   208427250), -INT32_C(  1986224546), -INT32_C(  2051832946), -INT32_C(  2090066882) },
      {  INT32_C(  1297003072), -INT32_C(   963021399), -INT32_C(  1502704520), -INT32_C(  1822623583), -INT32_C(  1348417200), -INT32_C(   180863897), -INT32_C(    42275905),  INT32_C(  1216472839) },
      { -INT32_C(   816925115), -INT32_C(   826280019), -INT32_C(  1476423811),  INT32_C(   573278886), -INT32_C(   841119773), -INT32_C(    37962113), -INT32_C(    33821761), -INT32_C(   873595073) } },
    { {  INT32_C(  1133891481), -INT32_C(  1140249020),  INT32_C(  1550022587), -INT32_C(  1057964176), -INT32_C(   277901176), -INT32_C(  1947883572), -INT32_C(  1702273133), -INT32_C(   522057146) },
      UINT8_C(217),
      { -INT32_C(  1507974281), -INT32_C(  1537091028), -INT32_C(    48972483),  INT32_C(  1032180909), -INT32_C(   351701692), -INT32_C(  1182886566),  INT32_C(   671095070),  INT32_C(  1929502971) },
      {  INT32_C(   790175491),  INT32_C(   936606458), -INT32_C(   449517769),  INT32_C(     2276028), -INT32_C(  1981076433), -INT32_C(   549295423),  INT32_C(  2114404995),  INT32_C(   636553250) },
      { -INT32_C(  1356906633), -INT32_C(  1140249020),  INT32_C(  1550022587),  INT32_C(  1034419901), -INT32_C(   336888465), -INT32_C(  1947883572),  INT32_C(  2114411423),  INT32_C(  2012342523) } },
    { {  INT32_C(   575998504), -INT32_C(  1135007612), -INT32_C(   861827568),  INT32_C(  2009908040), -INT32_C(  1342130193), -INT32_C(  1534115295), -INT32_C(  1490839931), -INT32_C(   942861153) },
      UINT8_C( 31),
      {  INT32_C(  1252256034), -INT32_C(   799383742), -INT32_C(  1005050368), -INT32_C(  1431072782), -INT32_C(   758422385), -INT32_C(  1973980941), -INT32_C(  1490420077), -INT32_C(   305729333) },
      {  INT32_C(   456682201), -INT32_C(   890465591), -INT32_C(  1416756039),  INT32_C(   592789908), -INT32_C(  1711988315),  INT32_C(   606293137),  INT32_C(   382422091),  INT32_C(   352621116) },
      {  INT32_C(  1539042299), -INT32_C(   620830773), -INT32_C(   274848071), -INT32_C(  1409822730), -INT32_C(   604019281), -INT32_C(  1534115295), -INT32_C(  1490839931), -INT32_C(   942861153) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_or_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_or_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_or_epi32(src, k, a, b);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(231),
      { -INT32_C(   175948367),  INT32_C(  1229795372), -INT32_C(  1502768503), -INT32_C(  1050847354), -INT32_C(   741945652),  INT32_C(  1632731820),  INT32_C(   922092483), -INT32_C(   652369880) },
      {  INT32_C(   181313757),  INT32_C(  1515396305),  INT32_C(   872464814), -INT32_C(   453616104), -INT32_C(   642204628),  INT32_C(    20580670),  INT32_C(   876032012), -INT32_C(  1374857776) },
      { -INT32_C(     3162627),  INT32_C(  1532968189), -INT32_C(  1234316369),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1635486654),  INT32_C(   922170319), -INT32_C(    14813704) } },
    { UINT8_C(245),
      { -INT32_C(   121194276), -INT32_C(   861527797),  INT32_C(  2145704737), -INT32_C(  1918055983), -INT32_C(  1966373503), -INT32_C(   275329857),  INT32_C(  1489029891), -INT32_C(  1269928232) },
      {  INT32_C(   833360934),  INT32_C(  1476285237), -INT32_C(     2694610),  INT32_C(   747405995), -INT32_C(   961128697),  INT32_C(   649481507), -INT32_C(   260082152),  INT32_C(   178638308) },
      { -INT32_C(   101794562),  INT32_C(           0), -INT32_C(      591057),  INT32_C(           0), -INT32_C(   822093945), -INT32_C(   273232449), -INT32_C(   117440741), -INT32_C(  1091571716) } },
    { UINT8_C(225),
      { -INT32_C(  1541981359),  INT32_C(   483552825), -INT32_C(   943206076), -INT32_C(  1261505443), -INT32_C(   136866390), -INT32_C(  1055916469),  INT32_C(  1252327549), -INT32_C(   164909147) },
      {  INT32_C(   614154986), -INT32_C(   180326992), -INT32_C(  1665398977), -INT32_C(  1504605445),  INT32_C(  1788684319), -INT32_C(  1557418714),  INT32_C(  1391317165),  INT32_C(  1783110015) },
      { -INT32_C(  1533051909),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   483393681),  INT32_C(  1525534973), -INT32_C(    26492929) } },
    { UINT8_C( 91),
      {  INT32_C(  1359777507), -INT32_C(   711982642),  INT32_C(  1204890813), -INT32_C(  1503168643), -INT32_C(  1043541740), -INT32_C(   848400131),  INT32_C(  1984807005), -INT32_C(   321734903) },
      {  INT32_C(   322821701), -INT32_C(  1662464545),  INT32_C(  2011412986),  INT32_C(  1159547440),  INT32_C(   419883548), -INT32_C(  1209633446),  INT32_C(  1043215157),  INT32_C(   791347434) },
      {  INT32_C(  1396563687), -INT32_C(   570896929),  INT32_C(           0), -INT32_C(   411074691), -INT32_C(   640746724),  INT32_C(           0),  INT32_C(  2121266045),  INT32_C(           0) } },
    { UINT8_C(222),
      {  INT32_C(   918372968), -INT32_C(   466593493), -INT32_C(  2028624068), -INT32_C(  1348248891), -INT32_C(   720782240), -INT32_C(   703872862), -INT32_C(   289388050), -INT32_C(   573771660) },
      {  INT32_C(  1561561394),  INT32_C(   524436450), -INT32_C(  1348053014),  INT32_C(   291391921), -INT32_C(  1478007035),  INT32_C(   377352744), -INT32_C(  1341833669),  INT32_C(  1619906862) },
      {  INT32_C(           0), -INT32_C(     9282581), -INT32_C(  1346897922), -INT32_C(  1073783819), -INT32_C(   135266459),  INT32_C(           0), -INT32_C(    20611073), -INT32_C(    36834946) } },
    { UINT8_C( 91),
      { -INT32_C(   482492768),  INT32_C(  1473076224),  INT32_C(  1258847490),  INT32_C(  1129322971), -INT32_C(   227805184), -INT32_C(  1288797835),  INT32_C(  1491197574),  INT32_C(   196297323) },
      {  INT32_C(    15659008),  INT32_C(  1331149645),  INT32_C(   345726776),  INT32_C(  2035805048),  INT32_C(  1500234467), -INT32_C(   905143997), -INT32_C(   501027209),  INT32_C(   820892976) },
      { -INT32_C(   469762400),  INT32_C(  1608515405),  INT32_C(           0),  INT32_C(  2069363707), -INT32_C(    76809501),  INT32_C(           0), -INT32_C(    85721353),  INT32_C(           0) } },
    { UINT8_C(197),
      { -INT32_C(  1760415525), -INT32_C(   422616441), -INT32_C(   396434435), -INT32_C(    53684422), -INT32_C(   599775933),  INT32_C(   525601329),  INT32_C(    21968428), -INT32_C(     3768540) },
      {  INT32_C(   915855791),  INT32_C(   941385275), -INT32_C(  2078246071), -INT32_C(  1786713006),  INT32_C(  1131593745), -INT32_C(   161298742),  INT32_C(   553104124), -INT32_C(   517947855) },
      { -INT32_C(  1214842369),  INT32_C(           0), -INT32_C(   327222275),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   570406652), -INT32_C(     1638603) } },
    { UINT8_C(151),
      {  INT32_C(   483530678), -INT32_C(  1352267212),  INT32_C(   386001450),  INT32_C(   724080490), -INT32_C(   805999863), -INT32_C(  2134119218), -INT32_C(  1582175005), -INT32_C(  1036479988) },
      { -INT32_C(   555808086),  INT32_C(  1066222868), -INT32_C(  1722380753),  INT32_C(   784629285), -INT32_C(  1191331094), -INT32_C(  1992767066), -INT32_C(  1054086731),  INT32_C(   646144892) },
      { -INT32_C(   555802690), -INT32_C(  1074835660), -INT32_C(  1621627345),  INT32_C(           0), -INT32_C(      132117),  INT32_C(           0),  INT32_C(           0), -INT32_C(   423890052) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_or_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_or_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_or_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 7847244174372717729), -INT64_C(  335388885211495971),  INT64_C(  554415635139927989),  INT64_C( 7897057326640079759) },
      { -INT64_C( 5355803772986207169),  INT64_C( 3507470111674992157),  INT64_C( 6576399825256783144), -INT64_C( 4148581040875599053) },
      { -INT64_C(  148794880162933569), -INT64_C(  288936262684194851),  INT64_C( 6914642660359610301), -INT64_C( 1152921717912917057) } },
    { {  INT64_C( 6677063905988502043), -INT64_C( 8474441468855035464),  INT64_C(  432359667386532825),  INT64_C( 1433399186120159383) },
      {  INT64_C( 6120866455623232086), -INT64_C( 7640660527591981303),  INT64_C( 7027932955154234798),  INT64_C( 6482477577704279787) },
      {  INT64_C( 6699596747599440479), -INT64_C( 6920062321833279559),  INT64_C( 7460282995354632191),  INT64_C( 6626615808749453055) } },
    { { -INT64_C( 5386903645513122739),  INT64_C( 6786668111226520798), -INT64_C( 7998840831930766892),  INT64_C( 3265789094775717366) },
      { -INT64_C(  160349343829311387), -INT64_C( 3375792077373184232),  INT64_C( 7306084818279663152), -INT64_C( 6414902401876174869) },
      { -INT64_C(  144150647426581395), -INT64_C( 2364434893561665570), -INT64_C(  721018017639600140), -INT64_C( 5765740153230327809) } },
    { {  INT64_C( 8866600730316181488), -INT64_C( 7335121987443645339), -INT64_C(  341197077345229968),  INT64_C( 4724768942168973739) },
      {  INT64_C( 3219332418763770319), -INT64_C( 3889207534580469759), -INT64_C(  904682254152207003), -INT64_C( 4026093852659074931) },
      {  INT64_C( 9200150244975136767), -INT64_C( 2722730013671558043), -INT64_C(  327637590168569995), -INT64_C( 3913081644677875281) } },
    { { -INT64_C( 3482222806983609927),  INT64_C( 7186690564811275160), -INT64_C( 2382041011873815115), -INT64_C(  413242367957675901) },
      {  INT64_C( 9145728812573212239),  INT64_C( 2072550175543469666),  INT64_C( 7620667705256483986), -INT64_C( 6046038119346241572) },
      { -INT64_C(    5436060146630657),  INT64_C( 9223211507930316794), -INT64_C(    3572347640742473), -INT64_C(  118221287662420001) } },
    { {  INT64_C( 2147009459128996710), -INT64_C(  261419432516174628),  INT64_C( 7074206714859322076), -INT64_C( 6905258230939643615) },
      { -INT64_C( 8790261578043952085), -INT64_C( 7252720922526121997), -INT64_C(  321010278842091358), -INT64_C( 3570047581046633573) },
      { -INT64_C( 6932245187923150993), -INT64_C(   45202025214017537), -INT64_C(  310840875353670914), -INT64_C( 1261081855017804869) } },
    { {  INT64_C(  921467814105276825),  INT64_C( 6742102750410674741),  INT64_C( 1427377718047403770), -INT64_C( 8751517799707817701) },
      {  INT64_C( 3162251477230633149),  INT64_C(  521486330004261668), -INT64_C( 1356801943873470049),  INT64_C( 3271927186775086068) },
      {  INT64_C( 3453054744971141565),  INT64_C( 6898656735174187829), -INT64_C(    4578388482330625), -INT64_C( 5770099592222974977) } },
    { { -INT64_C( 1613724884636847496),  INT64_C(  812060166454819376), -INT64_C( 4502105611870985787), -INT64_C( 2089939152943238814) },
      {  INT64_C( 1867844752667312957),  INT64_C( 8942912695629689765), -INT64_C( 1203981117198942532),  INT64_C( 3897233224811802058) },
      { -INT64_C(  433489140368854147),  INT64_C( 9178230484207844277), -INT64_C( 1166467704233181187), -INT64_C(  648575546918961686) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_or_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_or_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_or_epi64(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 8678686282522542852),  INT64_C(  349178915764509985),  INT64_C( 7304486864526626473), -INT64_C( 5161828051738416628) },
      UINT8_C(188),
      {  INT64_C( 3083578219770314126),  INT64_C( 2843898206059331653),  INT64_C( 4249329939549106475),  INT64_C( 3197466256835232035) },
      {  INT64_C( 4639731240995907883), -INT64_C( 6439156281095820253), -INT64_C( 5872541583302007754),  INT64_C( 5807022439989868235) },
      { -INT64_C( 8678686282522542852),  INT64_C(  349178915764509985), -INT64_C( 4685802328017341121),  INT64_C( 8998120307721467883) } },
    { {  INT64_C( 3735431397411155354), -INT64_C( 6919690215142464296), -INT64_C( 2274205568171865546),  INT64_C(  887738481332896477) },
      UINT8_C( 78),
      {  INT64_C( 6673650170726359770),  INT64_C( 8343310734738190400), -INT64_C( 4787773923623662774), -INT64_C( 4908293704328900685) },
      {  INT64_C( 1419230721318339038),  INT64_C( 2767865341305779982),  INT64_C(  986962709693592057), -INT64_C(  175214737722580235) },
      {  INT64_C( 3735431397411155354),  INT64_C( 8640559392840087374), -INT64_C( 4774257626036831237), -INT64_C(    3448069157687305) } },
    { {  INT64_C( 4443370783826514372), -INT64_C( 5224002126867050626), -INT64_C( 5551144221413713777), -INT64_C( 2351414354361411775) },
      UINT8_C( 43),
      {  INT64_C( 8013759853251688114),  INT64_C( 7636105031148474572),  INT64_C( 7940648614477881745), -INT64_C( 6726573611357203871) },
      {  INT64_C( 7236600262542688478),  INT64_C( 1269107448938928930), -INT64_C(  149123895201185112), -INT64_C( 4802773581409257530) },
      {  INT64_C( 8034309280072693502),  INT64_C( 8790155184755940334), -INT64_C( 5551144221413713777), -INT64_C( 4611827855427766297) } },
    { {  INT64_C( 7501633888511046686),  INT64_C( 6365054878546127299),  INT64_C( 5877977666976962553),  INT64_C( 5104938102016182844) },
      UINT8_C( 22),
      {  INT64_C(  180078124931546515),  INT64_C( 8038936435580126047), -INT64_C( 4475368877530143262),  INT64_C( 7767636065135492948) },
      {  INT64_C( 5568885576769677613), -INT64_C( 1733477914052920470), -INT64_C( 8498165819747865704),  INT64_C( 6248196004013734258) },
      {  INT64_C( 7501633888511046686), -INT64_C( 1157008262240666753), -INT64_C( 3750236559757021190),  INT64_C( 5104938102016182844) } },
    { { -INT64_C( 1265260239106810570),  INT64_C( 6975057434065786079),  INT64_C( 4687614780367657804), -INT64_C( 7982558476615594311) },
      UINT8_C( 31),
      {  INT64_C( 8761913765775258836), -INT64_C( 6507525220180394750), -INT64_C( 8085732364913462448), -INT64_C( 5070635484063290304) },
      { -INT64_C( 2752531881597451217), -INT64_C(  110358330739978616), -INT64_C( 2165711762436986079), -INT64_C( 8853790623268611196) },
      { -INT64_C(  442026798504217345), -INT64_C(    2271664196976758), -INT64_C( 1154612684713428111), -INT64_C( 4782400630810738748) } },
    { {  INT64_C( 2808178665246593864),  INT64_C( 4855565541772375477), -INT64_C( 3612847556559277733), -INT64_C( 8336911386576339087) },
      UINT8_C(234),
      { -INT64_C( 2844957046023345369), -INT64_C( 4940098828314679130), -INT64_C( 8580534551514419997),  INT64_C( 2830609124416074503) },
      {  INT64_C( 2773788009675739395), -INT64_C( 7320691669182154655),  INT64_C( 4966343653209697305),  INT64_C( 1358375215282960306) },
      {  INT64_C( 2808178665246593864), -INT64_C( 4938267365541480217), -INT64_C( 3612847556559277733),  INT64_C( 4024523788152387511) } },
    { { -INT64_C( 6169590330653356100),  INT64_C( 7277743650079415726), -INT64_C( 4203071134611664087), -INT64_C(  753982155348554136) },
      UINT8_C( 32),
      {  INT64_C( 3397423389175459905), -INT64_C( 7111220601662305432), -INT64_C( 5710999311749237084),  INT64_C(   98479679830832876) },
      {  INT64_C( 4670407466825598222), -INT64_C( 3724276605910627799),  INT64_C( 6953242766497415276),  INT64_C( 6113630197039064855) },
      { -INT64_C( 6169590330653356100),  INT64_C( 7277743650079415726), -INT64_C( 4203071134611664087), -INT64_C(  753982155348554136) } },
    { {  INT64_C( 1790397380927180992),  INT64_C( 6990462886120996622),  INT64_C( 2157697039699289372),  INT64_C( 5309686276506816022) },
      UINT8_C( 27),
      { -INT64_C(  542927050365017692), -INT64_C( 3015916757955269139),  INT64_C( 3838186903385716351), -INT64_C(  948844111786500114) },
      { -INT64_C( 1253458641808531480),  INT64_C( 6198674747198834815), -INT64_C( 3841350732384836909),  INT64_C( 6820518527045496422) },
      { -INT64_C(   72072188699869204), -INT64_C( 3015335910066770433),  INT64_C( 2157697039699289372), -INT64_C(   74485870874394642) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_or_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_or_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_mask_or_epi64(src, k, a, b);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(110),
      {  INT64_C( 2294233891516580962), -INT64_C( 1260881225967112169),  INT64_C( 6480431856177395825), -INT64_C( 7630510419264294839) },
      {  INT64_C( 8552358088756010495), -INT64_C( 2763085464976165500), -INT64_C( 6733723285700002876),  INT64_C( 6808432592989152178) },
      {  INT64_C(                   0), -INT64_C(   24770361693440617), -INT64_C(  292981944665818123), -INT64_C( 2414215823336934405) } },
    { UINT8_C(  6),
      { -INT64_C( 7831236801923256647), -INT64_C( 1425435798907647257),  INT64_C( 6601970259568520169), -INT64_C( 5069056464589781961) },
      { -INT64_C( 3586878613774463994), -INT64_C(  900587938658568990), -INT64_C( 1857879230795299101),  INT64_C( 1205326466473521408) },
      {  INT64_C(                   0), -INT64_C(   20268689605706009), -INT64_C(   18017701341499413),  INT64_C(                   0) } },
    { UINT8_C( 59),
      {  INT64_C( 2521155892051883329), -INT64_C( 3058472944781193422), -INT64_C( 7819820556902700291),  INT64_C( 8528319288054742001) },
      { -INT64_C( 6885569049375310494), -INT64_C( 5043791080956650213), -INT64_C( 2343368315005270972),  INT64_C( 2797959968446093097) },
      { -INT64_C( 6701923664936305309), -INT64_C(   31842133037048005),  INT64_C(                   0),  INT64_C( 8565562771624656889) } },
    { UINT8_C(214),
      { -INT64_C( 8932075226133138076), -INT64_C( 5093857923285568208), -INT64_C( 4177222039466724049), -INT64_C( 7011066428014303868) },
      {  INT64_C(  905673070949037861), -INT64_C(  235483723840695603),  INT64_C( 7786408094029406873),  INT64_C( 7935058081305237574) },
      {  INT64_C(                   0), -INT64_C(  144115200787220483), -INT64_C( 1292560650135963713),  INT64_C(                   0) } },
    { UINT8_C(245),
      { -INT64_C( 2517269151247926613),  INT64_C( 3454725790039151347), -INT64_C( 2258972716695078528), -INT64_C( 1098711987356358962) },
      {  INT64_C( 7060030605583683857), -INT64_C( 3795502821227780778), -INT64_C( 4824249176157295645), -INT64_C( 2509268275821005319) },
      { -INT64_C(  145560567515186245),  INT64_C(                   0), -INT64_C(  166959812381279261),  INT64_C(                   0) } },
    { UINT8_C(218),
      { -INT64_C( 7279036400697476511), -INT64_C( 2381927620189806749),  INT64_C( 6748795948967468253),  INT64_C( 8373888336857261586) },
      { -INT64_C( 8246067647748669695),  INT64_C(  325819550486132276), -INT64_C( 1211929146252111192),  INT64_C( 6974528115582850882) },
      {  INT64_C(                   0), -INT64_C( 2380790705267179657),  INT64_C(                   0),  INT64_C( 8430324396176701266) } },
    {    UINT8_MAX,
      {  INT64_C( 6714223726896436963), -INT64_C( 6015533278432013232), -INT64_C( 2676976001211608005), -INT64_C( 2575327840560468030) },
      {  INT64_C( 8228923288497515503),  INT64_C( 8853463611315501014), -INT64_C( 6880396464140163546),  INT64_C( 4895454937504529178) },
      {  INT64_C( 9169248473881346031), -INT64_C(   81668974985676842), -INT64_C(  370425610211985857), -INT64_C( 2309573663642632230) } },
    { UINT8_C(145),
      {  INT64_C(  260405365155502533),  INT64_C( 1255304364940250739),  INT64_C( 6040610882049742030), -INT64_C( 7121621644721264525) },
      {  INT64_C( 8453820559961414915),  INT64_C( 2272952564036834511),  INT64_C( 9017804090878032442), -INT64_C( 5113203274667340651) },
      {  INT64_C( 8637663630746705351),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_or_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_or_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_or_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_or_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   593.01), EASYSIMD_FLOAT32_C(   481.06), EASYSIMD_FLOAT32_C(   496.39), EASYSIMD_FLOAT32_C(  -336.58),
        EASYSIMD_FLOAT32_C(   242.01), EASYSIMD_FLOAT32_C(   223.48), EASYSIMD_FLOAT32_C(  -715.46), EASYSIMD_FLOAT32_C(  -672.58) },
      UINT8_C(104),
      { EASYSIMD_FLOAT32_C(   135.38), EASYSIMD_FLOAT32_C(  -934.83), EASYSIMD_FLOAT32_C(   168.34), EASYSIMD_FLOAT32_C(   979.56),
        EASYSIMD_FLOAT32_C(  -244.95), EASYSIMD_FLOAT32_C(   226.54), EASYSIMD_FLOAT32_C(    48.15), EASYSIMD_FLOAT32_C(  -213.73) },
      { EASYSIMD_FLOAT32_C(  -920.95), EASYSIMD_FLOAT32_C(   647.38), EASYSIMD_FLOAT32_C(   742.36), EASYSIMD_FLOAT32_C(   -67.60),
        EASYSIMD_FLOAT32_C(  -590.36), EASYSIMD_FLOAT32_C(     7.70), EASYSIMD_FLOAT32_C(   177.53), EASYSIMD_FLOAT32_C(   502.79) },
      { EASYSIMD_FLOAT32_C(   593.01), EASYSIMD_FLOAT32_C(   481.06), EASYSIMD_FLOAT32_C(   496.39), EASYSIMD_FLOAT32_C(-31737.98),
        EASYSIMD_FLOAT32_C(   242.01), EASYSIMD_FLOAT32_C(   493.86), EASYSIMD_FLOAT32_C(   241.62), EASYSIMD_FLOAT32_C(  -672.58) } },
    { { EASYSIMD_FLOAT32_C(   488.20), EASYSIMD_FLOAT32_C(   -26.52), EASYSIMD_FLOAT32_C(  -390.57), EASYSIMD_FLOAT32_C(  -734.37),
        EASYSIMD_FLOAT32_C(    19.02), EASYSIMD_FLOAT32_C(   805.87), EASYSIMD_FLOAT32_C(   858.64), EASYSIMD_FLOAT32_C(  -499.92) },
      UINT8_C(130),
      { EASYSIMD_FLOAT32_C(  -477.94), EASYSIMD_FLOAT32_C(   742.08), EASYSIMD_FLOAT32_C(  -474.26), EASYSIMD_FLOAT32_C(  -193.40),
        EASYSIMD_FLOAT32_C(  -930.50), EASYSIMD_FLOAT32_C(   882.94), EASYSIMD_FLOAT32_C(   941.98), EASYSIMD_FLOAT32_C(  -865.33) },
      { EASYSIMD_FLOAT32_C(    51.28), EASYSIMD_FLOAT32_C(   921.54), EASYSIMD_FLOAT32_C(  -110.29), EASYSIMD_FLOAT32_C(  -722.18),
        EASYSIMD_FLOAT32_C(   -30.31), EASYSIMD_FLOAT32_C(   675.98), EASYSIMD_FLOAT32_C(  -643.13), EASYSIMD_FLOAT32_C(  -382.93) },
      { EASYSIMD_FLOAT32_C(   488.20), EASYSIMD_FLOAT32_C(  1023.62), EASYSIMD_FLOAT32_C(  -390.57), EASYSIMD_FLOAT32_C(  -734.37),
        EASYSIMD_FLOAT32_C(    19.02), EASYSIMD_FLOAT32_C(   805.87), EASYSIMD_FLOAT32_C(   858.64), EASYSIMD_FLOAT32_C(-130798.24) } },
    { { EASYSIMD_FLOAT32_C(   418.34), EASYSIMD_FLOAT32_C(   289.28), EASYSIMD_FLOAT32_C(    26.72), EASYSIMD_FLOAT32_C(  -573.96),
        EASYSIMD_FLOAT32_C(  -533.19), EASYSIMD_FLOAT32_C(  -470.49), EASYSIMD_FLOAT32_C(   914.23), EASYSIMD_FLOAT32_C(   440.29) },
      UINT8_C(157),
      { EASYSIMD_FLOAT32_C(  -820.14), EASYSIMD_FLOAT32_C(  -540.69), EASYSIMD_FLOAT32_C(   -55.20), EASYSIMD_FLOAT32_C(  -961.50),
        EASYSIMD_FLOAT32_C(   -40.61), EASYSIMD_FLOAT32_C(  -752.94), EASYSIMD_FLOAT32_C(  -439.44), EASYSIMD_FLOAT32_C(  -298.53) },
      { EASYSIMD_FLOAT32_C(  -227.19), EASYSIMD_FLOAT32_C(   367.16), EASYSIMD_FLOAT32_C(  -229.03), EASYSIMD_FLOAT32_C(  -344.26),
        EASYSIMD_FLOAT32_C(   309.14), EASYSIMD_FLOAT32_C(   -94.36), EASYSIMD_FLOAT32_C(   707.02), EASYSIMD_FLOAT32_C(   230.68) },
      { EASYSIMD_FLOAT32_C(-61240.96), EASYSIMD_FLOAT32_C(   289.28), EASYSIMD_FLOAT32_C(  -253.81), EASYSIMD_FLOAT32_C(-129218.56),
        EASYSIMD_FLOAT32_C(  -373.89), EASYSIMD_FLOAT32_C(  -470.49), EASYSIMD_FLOAT32_C(   914.23), EASYSIMD_FLOAT32_C(  -495.87) } },
    { { EASYSIMD_FLOAT32_C(   795.35), EASYSIMD_FLOAT32_C(   984.84), EASYSIMD_FLOAT32_C(  -799.63), EASYSIMD_FLOAT32_C(   471.33),
        EASYSIMD_FLOAT32_C(  -658.28), EASYSIMD_FLOAT32_C(  -182.56), EASYSIMD_FLOAT32_C(  -110.33), EASYSIMD_FLOAT32_C(   631.00) },
      UINT8_C(159),
      { EASYSIMD_FLOAT32_C(   315.70), EASYSIMD_FLOAT32_C(  -902.19), EASYSIMD_FLOAT32_C(  -626.33), EASYSIMD_FLOAT32_C(   229.94),
        EASYSIMD_FLOAT32_C(   538.10), EASYSIMD_FLOAT32_C(   512.60), EASYSIMD_FLOAT32_C(   409.80), EASYSIMD_FLOAT32_C(   997.41) },
      { EASYSIMD_FLOAT32_C(  -542.60), EASYSIMD_FLOAT32_C(   448.30), EASYSIMD_FLOAT32_C(   -43.20), EASYSIMD_FLOAT32_C(  -295.53),
        EASYSIMD_FLOAT32_C(  -991.13), EASYSIMD_FLOAT32_C(   658.27), EASYSIMD_FLOAT32_C(   477.28), EASYSIMD_FLOAT32_C(   376.03) },
      { EASYSIMD_FLOAT32_C(-81919.98), EASYSIMD_FLOAT32_C(-115548.87), EASYSIMD_FLOAT32_C(-12087.47), EASYSIMD_FLOAT32_C(  -495.91),
        EASYSIMD_FLOAT32_C(  -991.23), EASYSIMD_FLOAT32_C(  -182.56), EASYSIMD_FLOAT32_C(  -110.33), EASYSIMD_FLOAT32_C(129719.99) } },
    { { EASYSIMD_FLOAT32_C(  -570.76), EASYSIMD_FLOAT32_C(  -866.98), EASYSIMD_FLOAT32_C(  -314.83), EASYSIMD_FLOAT32_C(   334.87),
        EASYSIMD_FLOAT32_C(   840.04), EASYSIMD_FLOAT32_C(   915.85), EASYSIMD_FLOAT32_C(   130.22), EASYSIMD_FLOAT32_C(   824.89) },
      UINT8_C( 30),
      { EASYSIMD_FLOAT32_C(  -398.45), EASYSIMD_FLOAT32_C(  -833.39), EASYSIMD_FLOAT32_C(   -66.35), EASYSIMD_FLOAT32_C(   491.22),
        EASYSIMD_FLOAT32_C(   797.60), EASYSIMD_FLOAT32_C(  -222.19), EASYSIMD_FLOAT32_C(  -193.07), EASYSIMD_FLOAT32_C(   895.41) },
      { EASYSIMD_FLOAT32_C(   151.47), EASYSIMD_FLOAT32_C(  -963.14), EASYSIMD_FLOAT32_C(   433.51), EASYSIMD_FLOAT32_C(  -335.92),
        EASYSIMD_FLOAT32_C(   446.66), EASYSIMD_FLOAT32_C(   430.92), EASYSIMD_FLOAT32_C(   121.48), EASYSIMD_FLOAT32_C(  -105.03) },
      { EASYSIMD_FLOAT32_C(  -570.76), EASYSIMD_FLOAT32_C(  -963.39), EASYSIMD_FLOAT32_C(  -441.90), EASYSIMD_FLOAT32_C(  -495.98),
        EASYSIMD_FLOAT32_C(114412.99), EASYSIMD_FLOAT32_C(   915.85), EASYSIMD_FLOAT32_C(   130.22), EASYSIMD_FLOAT32_C(   824.89) } },
    { { EASYSIMD_FLOAT32_C(  -612.28), EASYSIMD_FLOAT32_C(   825.95), EASYSIMD_FLOAT32_C(   -96.17), EASYSIMD_FLOAT32_C(  -954.02),
        EASYSIMD_FLOAT32_C(   303.22), EASYSIMD_FLOAT32_C(  -720.14), EASYSIMD_FLOAT32_C(  -524.78), EASYSIMD_FLOAT32_C(   436.24) },
      UINT8_C(117),
      { EASYSIMD_FLOAT32_C(   810.09), EASYSIMD_FLOAT32_C(   276.29), EASYSIMD_FLOAT32_C(  -119.12), EASYSIMD_FLOAT32_C(   -59.69),
        EASYSIMD_FLOAT32_C(   101.17), EASYSIMD_FLOAT32_C(    -2.91), EASYSIMD_FLOAT32_C(   541.86), EASYSIMD_FLOAT32_C(   267.78) },
      { EASYSIMD_FLOAT32_C(   930.74), EASYSIMD_FLOAT32_C(    33.08), EASYSIMD_FLOAT32_C(    65.38), EASYSIMD_FLOAT32_C(  -291.45),
        EASYSIMD_FLOAT32_C(   840.01), EASYSIMD_FLOAT32_C(   -39.20), EASYSIMD_FLOAT32_C(   860.02), EASYSIMD_FLOAT32_C(   876.87) },
      { EASYSIMD_FLOAT32_C(   938.75), EASYSIMD_FLOAT32_C(   825.95), EASYSIMD_FLOAT32_C(  -119.50), EASYSIMD_FLOAT32_C(  -954.02),
        EASYSIMD_FLOAT32_C( 27947.84), EASYSIMD_FLOAT32_C(   -47.75), EASYSIMD_FLOAT32_C(   861.86), EASYSIMD_FLOAT32_C(   436.24) } },
    { { EASYSIMD_FLOAT32_C(  -605.69), EASYSIMD_FLOAT32_C(  -475.90), EASYSIMD_FLOAT32_C(   323.54), EASYSIMD_FLOAT32_C(   825.23),
        EASYSIMD_FLOAT32_C(   645.58), EASYSIMD_FLOAT32_C(  -781.50), EASYSIMD_FLOAT32_C(  -787.05), EASYSIMD_FLOAT32_C(   471.53) },
      UINT8_C(183),
      { EASYSIMD_FLOAT32_C(  -741.06), EASYSIMD_FLOAT32_C(  -225.25), EASYSIMD_FLOAT32_C(   402.20), EASYSIMD_FLOAT32_C(  -265.84),
        EASYSIMD_FLOAT32_C(  -789.00), EASYSIMD_FLOAT32_C(  -632.77), EASYSIMD_FLOAT32_C(  -455.75), EASYSIMD_FLOAT32_C(   487.28) },
      { EASYSIMD_FLOAT32_C(   248.10), EASYSIMD_FLOAT32_C(   484.56), EASYSIMD_FLOAT32_C(  -411.54), EASYSIMD_FLOAT32_C(  -754.81),
        EASYSIMD_FLOAT32_C(    26.42), EASYSIMD_FLOAT32_C(   856.23), EASYSIMD_FLOAT32_C(  -824.07), EASYSIMD_FLOAT32_C(  -940.50) },
      { EASYSIMD_FLOAT32_C(-63835.87), EASYSIMD_FLOAT32_C(  -486.56), EASYSIMD_FLOAT32_C(  -411.73), EASYSIMD_FLOAT32_C(   825.23),
        EASYSIMD_FLOAT32_C( -6891.52), EASYSIMD_FLOAT32_C(  -889.00), EASYSIMD_FLOAT32_C(  -787.05), EASYSIMD_FLOAT32_C(-128839.68) } },
    { { EASYSIMD_FLOAT32_C(   -78.38), EASYSIMD_FLOAT32_C(  -115.52), EASYSIMD_FLOAT32_C(   899.51), EASYSIMD_FLOAT32_C(   882.41),
        EASYSIMD_FLOAT32_C(  -255.49), EASYSIMD_FLOAT32_C(   776.39), EASYSIMD_FLOAT32_C(  -723.28), EASYSIMD_FLOAT32_C(   268.61) },
      UINT8_C( 14),
      { EASYSIMD_FLOAT32_C(  -898.04), EASYSIMD_FLOAT32_C(   -85.81), EASYSIMD_FLOAT32_C(   318.42), EASYSIMD_FLOAT32_C(  -685.09),
        EASYSIMD_FLOAT32_C(  -614.28), EASYSIMD_FLOAT32_C(  -559.24), EASYSIMD_FLOAT32_C(  -426.16), EASYSIMD_FLOAT32_C(   160.47) },
      { EASYSIMD_FLOAT32_C(   842.96), EASYSIMD_FLOAT32_C(   308.00), EASYSIMD_FLOAT32_C(   371.47), EASYSIMD_FLOAT32_C(  -789.82),
        EASYSIMD_FLOAT32_C(   852.25), EASYSIMD_FLOAT32_C(  -141.24), EASYSIMD_FLOAT32_C(   458.29), EASYSIMD_FLOAT32_C(   336.80) },
      { EASYSIMD_FLOAT32_C(   -78.38), EASYSIMD_FLOAT32_C(  -375.24), EASYSIMD_FLOAT32_C(   383.48), EASYSIMD_FLOAT32_C(  -957.84),
        EASYSIMD_FLOAT32_C(  -255.49), EASYSIMD_FLOAT32_C(   776.39), EASYSIMD_FLOAT32_C(  -723.28), EASYSIMD_FLOAT32_C(   268.61) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_or_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_or_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_or_ps(src, k, a, b);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_or_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(244),
      { EASYSIMD_FLOAT32_C(  -462.55), EASYSIMD_FLOAT32_C(  -287.38), EASYSIMD_FLOAT32_C(  -832.87), EASYSIMD_FLOAT32_C(  -714.13),
        EASYSIMD_FLOAT32_C(   -49.95), EASYSIMD_FLOAT32_C(   731.37), EASYSIMD_FLOAT32_C(  -454.82), EASYSIMD_FLOAT32_C(   -73.90) },
      { EASYSIMD_FLOAT32_C(  -865.78), EASYSIMD_FLOAT32_C(  -223.38), EASYSIMD_FLOAT32_C(  -940.76), EASYSIMD_FLOAT32_C(   -18.33),
        EASYSIMD_FLOAT32_C(   824.74), EASYSIMD_FLOAT32_C(   536.37), EASYSIMD_FLOAT32_C(   437.18), EASYSIMD_FLOAT32_C(   458.90) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1004.87), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-13307.97), EASYSIMD_FLOAT32_C(   731.37), EASYSIMD_FLOAT32_C(  -504.00), EASYSIMD_FLOAT32_C(  -496.00) } },
    { UINT8_C(100),
      { EASYSIMD_FLOAT32_C(   271.69), EASYSIMD_FLOAT32_C(  -975.55), EASYSIMD_FLOAT32_C(  -534.64), EASYSIMD_FLOAT32_C(   -62.33),
        EASYSIMD_FLOAT32_C(   217.05), EASYSIMD_FLOAT32_C(  -495.35), EASYSIMD_FLOAT32_C(  -510.42), EASYSIMD_FLOAT32_C(  -371.35) },
      { EASYSIMD_FLOAT32_C(   131.86), EASYSIMD_FLOAT32_C(  -636.87), EASYSIMD_FLOAT32_C(  -472.65), EASYSIMD_FLOAT32_C(   -11.35),
        EASYSIMD_FLOAT32_C(   573.25), EASYSIMD_FLOAT32_C(  -578.20), EASYSIMD_FLOAT32_C(   526.10), EASYSIMD_FLOAT32_C(  -714.13) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-121847.93), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-126809.60), EASYSIMD_FLOAT32_C(-130927.80), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 60),
      { EASYSIMD_FLOAT32_C(   811.97), EASYSIMD_FLOAT32_C(   235.92), EASYSIMD_FLOAT32_C(  -679.70), EASYSIMD_FLOAT32_C(  -642.86),
        EASYSIMD_FLOAT32_C(  -837.98), EASYSIMD_FLOAT32_C(  -545.49), EASYSIMD_FLOAT32_C(   133.76), EASYSIMD_FLOAT32_C(  -778.74) },
      { EASYSIMD_FLOAT32_C(   436.18), EASYSIMD_FLOAT32_C(   -41.50), EASYSIMD_FLOAT32_C(   757.63), EASYSIMD_FLOAT32_C(  -126.65),
        EASYSIMD_FLOAT32_C(  -582.59), EASYSIMD_FLOAT32_C(   194.28), EASYSIMD_FLOAT32_C(  -854.95), EASYSIMD_FLOAT32_C(  -558.15) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -759.70), EASYSIMD_FLOAT32_C(-32511.90),
        EASYSIMD_FLOAT32_C(  -840.00), EASYSIMD_FLOAT32_C(-51807.99), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(242),
      { EASYSIMD_FLOAT32_C(    82.72), EASYSIMD_FLOAT32_C(   658.90), EASYSIMD_FLOAT32_C(  -835.71), EASYSIMD_FLOAT32_C(   572.29),
        EASYSIMD_FLOAT32_C(  -712.45), EASYSIMD_FLOAT32_C(   296.16), EASYSIMD_FLOAT32_C(   935.42), EASYSIMD_FLOAT32_C(  -185.09) },
      { EASYSIMD_FLOAT32_C(  -715.19), EASYSIMD_FLOAT32_C(   508.67), EASYSIMD_FLOAT32_C(   236.71), EASYSIMD_FLOAT32_C(   810.91),
        EASYSIMD_FLOAT32_C(   794.54), EASYSIMD_FLOAT32_C(   825.63), EASYSIMD_FLOAT32_C(   622.88), EASYSIMD_FLOAT32_C(    30.46) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(130555.71), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -986.98), EASYSIMD_FLOAT32_C(113912.96), EASYSIMD_FLOAT32_C(  1007.92), EASYSIMD_FLOAT32_C(  -503.49) } },
    { UINT8_C(218),
      { EASYSIMD_FLOAT32_C(   980.02), EASYSIMD_FLOAT32_C(   192.48), EASYSIMD_FLOAT32_C(  -399.56), EASYSIMD_FLOAT32_C(   113.78),
        EASYSIMD_FLOAT32_C(   413.74), EASYSIMD_FLOAT32_C(  -963.38), EASYSIMD_FLOAT32_C(  -927.71), EASYSIMD_FLOAT32_C(   171.38) },
      { EASYSIMD_FLOAT32_C(   -90.03), EASYSIMD_FLOAT32_C(  -510.31), EASYSIMD_FLOAT32_C(  -634.34), EASYSIMD_FLOAT32_C(    55.02),
        EASYSIMD_FLOAT32_C(   -68.45), EASYSIMD_FLOAT32_C(  -974.70), EASYSIMD_FLOAT32_C(  -862.27), EASYSIMD_FLOAT32_C(  -409.55) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -511.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   127.81),
        EASYSIMD_FLOAT32_C(  -413.99), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -991.96), EASYSIMD_FLOAT32_C(  -479.81) } },
    { UINT8_C(190),
      { EASYSIMD_FLOAT32_C(   710.03), EASYSIMD_FLOAT32_C(  -121.99), EASYSIMD_FLOAT32_C(   485.75), EASYSIMD_FLOAT32_C(   645.45),
        EASYSIMD_FLOAT32_C(   692.91), EASYSIMD_FLOAT32_C(   770.56), EASYSIMD_FLOAT32_C(   154.12), EASYSIMD_FLOAT32_C(   -70.38) },
      { EASYSIMD_FLOAT32_C(   581.47), EASYSIMD_FLOAT32_C(   -51.34), EASYSIMD_FLOAT32_C(  -244.75), EASYSIMD_FLOAT32_C(   204.35),
        EASYSIMD_FLOAT32_C(   979.12), EASYSIMD_FLOAT32_C(   -98.82), EASYSIMD_FLOAT32_C(   184.37), EASYSIMD_FLOAT32_C(   171.60) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -128.00), EASYSIMD_FLOAT32_C(  -493.75), EASYSIMD_FLOAT32_C( 60765.87),
        EASYSIMD_FLOAT32_C(  1016.00), EASYSIMD_FLOAT32_C(-25297.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -351.72) } },
    { UINT8_C(231),
      { EASYSIMD_FLOAT32_C(  -701.84), EASYSIMD_FLOAT32_C(  -414.65), EASYSIMD_FLOAT32_C(   538.23), EASYSIMD_FLOAT32_C(  -629.55),
        EASYSIMD_FLOAT32_C(   756.72), EASYSIMD_FLOAT32_C(  -551.80), EASYSIMD_FLOAT32_C(  -139.86), EASYSIMD_FLOAT32_C(  -877.62) },
      { EASYSIMD_FLOAT32_C(   503.22), EASYSIMD_FLOAT32_C(   791.69), EASYSIMD_FLOAT32_C(  -852.32), EASYSIMD_FLOAT32_C(   640.95),
        EASYSIMD_FLOAT32_C(  -617.86), EASYSIMD_FLOAT32_C(  -662.73), EASYSIMD_FLOAT32_C(   350.98), EASYSIMD_FLOAT32_C(   260.15) },
      { EASYSIMD_FLOAT32_C(-131067.84), EASYSIMD_FLOAT32_C(-106494.46), EASYSIMD_FLOAT32_C(  -862.48), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -696.00), EASYSIMD_FLOAT32_C(  -351.98), EASYSIMD_FLOAT32_C(-112367.49) } },
    { UINT8_C( 90),
      { EASYSIMD_FLOAT32_C(    -3.57), EASYSIMD_FLOAT32_C(   -46.94), EASYSIMD_FLOAT32_C(   593.58), EASYSIMD_FLOAT32_C(  -849.45),
        EASYSIMD_FLOAT32_C(   882.67), EASYSIMD_FLOAT32_C(   175.06), EASYSIMD_FLOAT32_C(    99.21), EASYSIMD_FLOAT32_C(  -362.08) },
      { EASYSIMD_FLOAT32_C(  -620.59), EASYSIMD_FLOAT32_C(    78.33), EASYSIMD_FLOAT32_C(   539.10), EASYSIMD_FLOAT32_C(   563.78),
        EASYSIMD_FLOAT32_C(  -750.07), EASYSIMD_FLOAT32_C(    40.72), EASYSIMD_FLOAT32_C(   861.94), EASYSIMD_FLOAT32_C(  -164.72) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -95.96), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -883.97),
        EASYSIMD_FLOAT32_C( -1022.73), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 27583.84), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_or_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_or_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_or_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_or_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -703.78), EASYSIMD_FLOAT64_C(  -976.88), EASYSIMD_FLOAT64_C(  -984.20), EASYSIMD_FLOAT64_C(   724.12) },
      UINT8_C( 52),
      { EASYSIMD_FLOAT64_C(  -947.59), EASYSIMD_FLOAT64_C(   862.10), EASYSIMD_FLOAT64_C(  -842.48), EASYSIMD_FLOAT64_C(  -766.44) },
      { EASYSIMD_FLOAT64_C(   361.06), EASYSIMD_FLOAT64_C(  -630.24), EASYSIMD_FLOAT64_C(   398.65), EASYSIMD_FLOAT64_C(   952.89) },
      { EASYSIMD_FLOAT64_C(  -703.78), EASYSIMD_FLOAT64_C(  -976.88), EASYSIMD_FLOAT64_C(-110527.46), EASYSIMD_FLOAT64_C(   724.12) } },
    { { EASYSIMD_FLOAT64_C(  -792.49), EASYSIMD_FLOAT64_C(  -910.35), EASYSIMD_FLOAT64_C(  -657.38), EASYSIMD_FLOAT64_C(   273.30) },
      UINT8_C(121),
      { EASYSIMD_FLOAT64_C(  -893.96), EASYSIMD_FLOAT64_C(   684.62), EASYSIMD_FLOAT64_C(   252.44), EASYSIMD_FLOAT64_C(   499.03) },
      { EASYSIMD_FLOAT64_C(  -663.10), EASYSIMD_FLOAT64_C(   670.92), EASYSIMD_FLOAT64_C(  -401.31), EASYSIMD_FLOAT64_C(   105.41) },
      { EASYSIMD_FLOAT64_C( -1023.99), EASYSIMD_FLOAT64_C(  -910.35), EASYSIMD_FLOAT64_C(  -657.38), EASYSIMD_FLOAT64_C(   503.66) } },
    { { EASYSIMD_FLOAT64_C(  -835.66), EASYSIMD_FLOAT64_C(  -504.86), EASYSIMD_FLOAT64_C(  -174.01), EASYSIMD_FLOAT64_C(  -230.86) },
      UINT8_C(253),
      { EASYSIMD_FLOAT64_C(   122.21), EASYSIMD_FLOAT64_C(  -207.74), EASYSIMD_FLOAT64_C(  -736.15), EASYSIMD_FLOAT64_C(  -153.67) },
      { EASYSIMD_FLOAT64_C(   159.17), EASYSIMD_FLOAT64_C(  -683.74), EASYSIMD_FLOAT64_C(  -291.57), EASYSIMD_FLOAT64_C(   316.69) },
      { EASYSIMD_FLOAT64_C(   510.84), EASYSIMD_FLOAT64_C(  -504.86), EASYSIMD_FLOAT64_C(-95123.98), EASYSIMD_FLOAT64_C(  -319.97) } },
    { { EASYSIMD_FLOAT64_C(  -450.19), EASYSIMD_FLOAT64_C(  -930.51), EASYSIMD_FLOAT64_C(   686.45), EASYSIMD_FLOAT64_C(   948.46) },
      UINT8_C( 63),
      { EASYSIMD_FLOAT64_C(   893.96), EASYSIMD_FLOAT64_C(  -961.89), EASYSIMD_FLOAT64_C(  -635.00), EASYSIMD_FLOAT64_C(   167.25) },
      { EASYSIMD_FLOAT64_C(   362.62), EASYSIMD_FLOAT64_C(  -528.95), EASYSIMD_FLOAT64_C(  -148.12), EASYSIMD_FLOAT64_C(  -384.94) },
      { EASYSIMD_FLOAT64_C(130814.97), EASYSIMD_FLOAT64_C(  -977.95), EASYSIMD_FLOAT64_C(-40670.72), EASYSIMD_FLOAT64_C(  -462.94) } },
    { { EASYSIMD_FLOAT64_C(   970.08), EASYSIMD_FLOAT64_C(   188.78), EASYSIMD_FLOAT64_C(  -714.02), EASYSIMD_FLOAT64_C(  -431.24) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT64_C(  -549.68), EASYSIMD_FLOAT64_C(    63.91), EASYSIMD_FLOAT64_C(   120.18), EASYSIMD_FLOAT64_C(   219.46) },
      { EASYSIMD_FLOAT64_C(   311.95), EASYSIMD_FLOAT64_C(  -757.61), EASYSIMD_FLOAT64_C(  -988.29), EASYSIMD_FLOAT64_C(   575.80) },
      { EASYSIMD_FLOAT64_C(   970.08), EASYSIMD_FLOAT64_C(-16377.97), EASYSIMD_FLOAT64_C(  -714.02), EASYSIMD_FLOAT64_C( 57335.95) } },
    { { EASYSIMD_FLOAT64_C(    88.72), EASYSIMD_FLOAT64_C(   170.88), EASYSIMD_FLOAT64_C(   892.06), EASYSIMD_FLOAT64_C(   797.14) },
      UINT8_C( 25),
      { EASYSIMD_FLOAT64_C(  -558.13), EASYSIMD_FLOAT64_C(   866.63), EASYSIMD_FLOAT64_C(  -825.97), EASYSIMD_FLOAT64_C(  -609.66) },
      { EASYSIMD_FLOAT64_C(   889.01), EASYSIMD_FLOAT64_C(  -932.01), EASYSIMD_FLOAT64_C(  -571.55), EASYSIMD_FLOAT64_C(  -745.98) },
      { EASYSIMD_FLOAT64_C(  -895.14), EASYSIMD_FLOAT64_C(   170.88), EASYSIMD_FLOAT64_C(   892.06), EASYSIMD_FLOAT64_C(  -745.98) } },
    { { EASYSIMD_FLOAT64_C(   235.24), EASYSIMD_FLOAT64_C(   791.07), EASYSIMD_FLOAT64_C(  -274.94), EASYSIMD_FLOAT64_C(  -912.89) },
      UINT8_C( 96),
      { EASYSIMD_FLOAT64_C(  -304.86), EASYSIMD_FLOAT64_C(   275.89), EASYSIMD_FLOAT64_C(  -307.90), EASYSIMD_FLOAT64_C(   263.90) },
      { EASYSIMD_FLOAT64_C(   570.08), EASYSIMD_FLOAT64_C(   142.42), EASYSIMD_FLOAT64_C(  -672.19), EASYSIMD_FLOAT64_C(  -309.75) },
      { EASYSIMD_FLOAT64_C(   235.24), EASYSIMD_FLOAT64_C(   791.07), EASYSIMD_FLOAT64_C(  -274.94), EASYSIMD_FLOAT64_C(  -912.89) } },
    { { EASYSIMD_FLOAT64_C(  -638.13), EASYSIMD_FLOAT64_C(   639.76), EASYSIMD_FLOAT64_C(   -67.36), EASYSIMD_FLOAT64_C(  -626.41) },
      UINT8_C( 44),
      { EASYSIMD_FLOAT64_C(  -978.64), EASYSIMD_FLOAT64_C(   544.47), EASYSIMD_FLOAT64_C(   107.63), EASYSIMD_FLOAT64_C(   818.50) },
      { EASYSIMD_FLOAT64_C(  -967.95), EASYSIMD_FLOAT64_C(   549.50), EASYSIMD_FLOAT64_C(   685.14), EASYSIMD_FLOAT64_C(  -793.92) },
      { EASYSIMD_FLOAT64_C(  -638.13), EASYSIMD_FLOAT64_C(   639.76), EASYSIMD_FLOAT64_C( 32677.50), EASYSIMD_FLOAT64_C(  -827.92) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_or_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_or_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_or_pd(src, k, a, b);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_or_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(180),
      { EASYSIMD_FLOAT64_C(   734.08), EASYSIMD_FLOAT64_C(   267.96), EASYSIMD_FLOAT64_C(  -415.49), EASYSIMD_FLOAT64_C(   792.48) },
      { EASYSIMD_FLOAT64_C(  -534.26), EASYSIMD_FLOAT64_C(   571.03), EASYSIMD_FLOAT64_C(   440.23), EASYSIMD_FLOAT64_C(   211.61) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -447.50), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 26),
      { EASYSIMD_FLOAT64_C(   -59.21), EASYSIMD_FLOAT64_C(  -882.00), EASYSIMD_FLOAT64_C(  -113.69), EASYSIMD_FLOAT64_C(  -178.80) },
      { EASYSIMD_FLOAT64_C(  -325.64), EASYSIMD_FLOAT64_C(   318.49), EASYSIMD_FLOAT64_C(    -0.95), EASYSIMD_FLOAT64_C(   516.89) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-114557.44), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-46076.99) } },
    { UINT8_C(214),
      { EASYSIMD_FLOAT64_C(  -116.42), EASYSIMD_FLOAT64_C(  -464.38), EASYSIMD_FLOAT64_C(  -658.77), EASYSIMD_FLOAT64_C(  -713.14) },
      { EASYSIMD_FLOAT64_C(  -341.79), EASYSIMD_FLOAT64_C(   831.21), EASYSIMD_FLOAT64_C(  -270.32), EASYSIMD_FLOAT64_C(   957.69) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-122875.91), EASYSIMD_FLOAT64_C(-85875.94), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 56),
      { EASYSIMD_FLOAT64_C(    63.37), EASYSIMD_FLOAT64_C(   222.59), EASYSIMD_FLOAT64_C(  -798.75), EASYSIMD_FLOAT64_C(   510.69) },
      { EASYSIMD_FLOAT64_C(   -43.33), EASYSIMD_FLOAT64_C(   469.21), EASYSIMD_FLOAT64_C(  -904.79), EASYSIMD_FLOAT64_C(  -250.85) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -511.70) } },
    { UINT8_C( 34),
      { EASYSIMD_FLOAT64_C(   666.24), EASYSIMD_FLOAT64_C(  -810.62), EASYSIMD_FLOAT64_C(   146.56), EASYSIMD_FLOAT64_C(   961.00) },
      { EASYSIMD_FLOAT64_C(   130.16), EASYSIMD_FLOAT64_C(   264.56), EASYSIMD_FLOAT64_C(  -152.70), EASYSIMD_FLOAT64_C(   951.36) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-105935.36), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 51),
      { EASYSIMD_FLOAT64_C(  -834.20), EASYSIMD_FLOAT64_C(   -49.59), EASYSIMD_FLOAT64_C(   455.81), EASYSIMD_FLOAT64_C(   619.14) },
      { EASYSIMD_FLOAT64_C(   833.98), EASYSIMD_FLOAT64_C(   991.43), EASYSIMD_FLOAT64_C(   960.37), EASYSIMD_FLOAT64_C(  -879.16) },
      { EASYSIMD_FLOAT64_C(  -835.98), EASYSIMD_FLOAT64_C(-15863.92), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(194),
      { EASYSIMD_FLOAT64_C(   791.58), EASYSIMD_FLOAT64_C(  -149.48), EASYSIMD_FLOAT64_C(  -392.67), EASYSIMD_FLOAT64_C(   434.35) },
      { EASYSIMD_FLOAT64_C(   913.90), EASYSIMD_FLOAT64_C(   829.91), EASYSIMD_FLOAT64_C(   635.60), EASYSIMD_FLOAT64_C(   424.59) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-57210.99), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 73),
      { EASYSIMD_FLOAT64_C(   104.81), EASYSIMD_FLOAT64_C(   519.80), EASYSIMD_FLOAT64_C(   535.73), EASYSIMD_FLOAT64_C(    39.76) },
      { EASYSIMD_FLOAT64_C(   186.04), EASYSIMD_FLOAT64_C(   725.11), EASYSIMD_FLOAT64_C(  -813.67), EASYSIMD_FLOAT64_C(   147.04) },
      { EASYSIMD_FLOAT64_C(   503.24), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   159.04) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_or_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_or_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_or_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_or_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   480.60), EASYSIMD_FLOAT32_C(  -511.13), EASYSIMD_FLOAT32_C(  -479.78), EASYSIMD_FLOAT32_C(   269.24),
        EASYSIMD_FLOAT32_C(  -874.76), EASYSIMD_FLOAT32_C(   -72.46), EASYSIMD_FLOAT32_C(   197.37), EASYSIMD_FLOAT32_C(  -811.08),
        EASYSIMD_FLOAT32_C(    97.54), EASYSIMD_FLOAT32_C(  -611.27), EASYSIMD_FLOAT32_C(   407.81), EASYSIMD_FLOAT32_C(    -9.58),
        EASYSIMD_FLOAT32_C(  -941.56), EASYSIMD_FLOAT32_C(  -785.37), EASYSIMD_FLOAT32_C(   331.51), EASYSIMD_FLOAT32_C(  -275.33) },
      { EASYSIMD_FLOAT32_C(   603.88), EASYSIMD_FLOAT32_C(  -554.12), EASYSIMD_FLOAT32_C(   900.59), EASYSIMD_FLOAT32_C(   137.08),
        EASYSIMD_FLOAT32_C(  -120.48), EASYSIMD_FLOAT32_C(  -863.13), EASYSIMD_FLOAT32_C(  -707.03), EASYSIMD_FLOAT32_C(   972.38),
        EASYSIMD_FLOAT32_C(   820.00), EASYSIMD_FLOAT32_C(  -330.32), EASYSIMD_FLOAT32_C(   241.94), EASYSIMD_FLOAT32_C(   338.15),
        EASYSIMD_FLOAT32_C(  -659.11), EASYSIMD_FLOAT32_C(   398.68), EASYSIMD_FLOAT32_C(   573.47), EASYSIMD_FLOAT32_C(   358.72) },
      { EASYSIMD_FLOAT32_C(126457.74), EASYSIMD_FLOAT32_C(-130863.36), EASYSIMD_FLOAT32_C(-122831.68), EASYSIMD_FLOAT32_C(   287.24),
        EASYSIMD_FLOAT32_C(-32122.95), EASYSIMD_FLOAT32_C(-27637.92), EASYSIMD_FLOAT32_C(-62943.98), EASYSIMD_FLOAT32_C( -1007.46),
        EASYSIMD_FLOAT32_C( 26506.24), EASYSIMD_FLOAT32_C(-97267.98), EASYSIMD_FLOAT32_C(   503.93), EASYSIMD_FLOAT32_C(  -370.69),
        EASYSIMD_FLOAT32_C(  -959.62), EASYSIMD_FLOAT32_C(-102063.36), EASYSIMD_FLOAT32_C( 90046.72), EASYSIMD_FLOAT32_C(  -375.99) } },
    { { EASYSIMD_FLOAT32_C(   993.93), EASYSIMD_FLOAT32_C(  -985.03), EASYSIMD_FLOAT32_C(   659.60), EASYSIMD_FLOAT32_C(  -173.26),
        EASYSIMD_FLOAT32_C(    63.41), EASYSIMD_FLOAT32_C(  -232.36), EASYSIMD_FLOAT32_C(   790.92), EASYSIMD_FLOAT32_C(   958.34),
        EASYSIMD_FLOAT32_C(   817.72), EASYSIMD_FLOAT32_C(  -499.64), EASYSIMD_FLOAT32_C(   607.64), EASYSIMD_FLOAT32_C(   603.33),
        EASYSIMD_FLOAT32_C(   226.40), EASYSIMD_FLOAT32_C(  -876.92), EASYSIMD_FLOAT32_C(  -443.68), EASYSIMD_FLOAT32_C(  -893.18) },
      { EASYSIMD_FLOAT32_C(   265.66), EASYSIMD_FLOAT32_C(   933.56), EASYSIMD_FLOAT32_C(   320.39), EASYSIMD_FLOAT32_C(   825.80),
        EASYSIMD_FLOAT32_C(  -854.99), EASYSIMD_FLOAT32_C(   141.14), EASYSIMD_FLOAT32_C(   123.87), EASYSIMD_FLOAT32_C(  -184.24),
        EASYSIMD_FLOAT32_C(  -440.97), EASYSIMD_FLOAT32_C(   558.13), EASYSIMD_FLOAT32_C(   445.71), EASYSIMD_FLOAT32_C(  -893.22),
        EASYSIMD_FLOAT32_C(  -301.77), EASYSIMD_FLOAT32_C(   180.84), EASYSIMD_FLOAT32_C(  -576.68), EASYSIMD_FLOAT32_C(  -226.07) },
      { EASYSIMD_FLOAT32_C(129535.99), EASYSIMD_FLOAT32_C( -1021.56), EASYSIMD_FLOAT32_C( 84463.86), EASYSIMD_FLOAT32_C(-61299.75),
        EASYSIMD_FLOAT32_C(-16239.97), EASYSIMD_FLOAT32_C(  -237.50), EASYSIMD_FLOAT32_C( 31711.97), EASYSIMD_FLOAT32_C(-65469.95),
        EASYSIMD_FLOAT32_C(-112892.48), EASYSIMD_FLOAT32_C(-128947.98), EASYSIMD_FLOAT32_C(114677.93), EASYSIMD_FLOAT32_C(  -895.49),
        EASYSIMD_FLOAT32_C(  -493.80), EASYSIMD_FLOAT32_C(-65535.92), EASYSIMD_FLOAT32_C(-113663.12), EASYSIMD_FLOAT32_C(-65371.93) } },
    { { EASYSIMD_FLOAT32_C(   454.76), EASYSIMD_FLOAT32_C(  -837.79), EASYSIMD_FLOAT32_C(  -704.90), EASYSIMD_FLOAT32_C(   766.06),
        EASYSIMD_FLOAT32_C(  -788.99), EASYSIMD_FLOAT32_C(  -546.21), EASYSIMD_FLOAT32_C(  -221.91), EASYSIMD_FLOAT32_C(   380.98),
        EASYSIMD_FLOAT32_C(  -951.29), EASYSIMD_FLOAT32_C(  -366.31), EASYSIMD_FLOAT32_C(  -652.14), EASYSIMD_FLOAT32_C(  -549.49),
        EASYSIMD_FLOAT32_C(   782.12), EASYSIMD_FLOAT32_C(  -971.68), EASYSIMD_FLOAT32_C(   893.59), EASYSIMD_FLOAT32_C(   570.16) },
      { EASYSIMD_FLOAT32_C(   642.89), EASYSIMD_FLOAT32_C(  -941.41), EASYSIMD_FLOAT32_C(  -206.47), EASYSIMD_FLOAT32_C(  -125.77),
        EASYSIMD_FLOAT32_C(   256.81), EASYSIMD_FLOAT32_C(   243.45), EASYSIMD_FLOAT32_C(  -290.65), EASYSIMD_FLOAT32_C(  -982.09),
        EASYSIMD_FLOAT32_C(    -3.25), EASYSIMD_FLOAT32_C(  -692.16), EASYSIMD_FLOAT32_C(   812.74), EASYSIMD_FLOAT32_C(  -312.12),
        EASYSIMD_FLOAT32_C(   275.94), EASYSIMD_FLOAT32_C(  -213.58), EASYSIMD_FLOAT32_C(   384.73), EASYSIMD_FLOAT32_C(   236.37) },
      { EASYSIMD_FLOAT32_C(116723.98), EASYSIMD_FLOAT32_C( -1005.92), EASYSIMD_FLOAT32_C(-65145.85), EASYSIMD_FLOAT32_C(-32710.00),
        EASYSIMD_FLOAT32_C(-101119.98), EASYSIMD_FLOAT32_C(-64511.45), EASYSIMD_FLOAT32_C(  -443.97), EASYSIMD_FLOAT32_C(-131067.90),
        EASYSIMD_FLOAT32_C( -1015.29), EASYSIMD_FLOAT32_C(-97887.49), EASYSIMD_FLOAT32_C(  -940.75), EASYSIMD_FLOAT32_C(-80574.72),
        EASYSIMD_FLOAT32_C(104447.98), EASYSIMD_FLOAT32_C(-63488.00), EASYSIMD_FLOAT32_C(114427.90), EASYSIMD_FLOAT32_C( 61150.74) } },
    { { EASYSIMD_FLOAT32_C(   695.36), EASYSIMD_FLOAT32_C(   970.40), EASYSIMD_FLOAT32_C(  -483.18), EASYSIMD_FLOAT32_C(  -766.40),
        EASYSIMD_FLOAT32_C(  -816.54), EASYSIMD_FLOAT32_C(   446.99), EASYSIMD_FLOAT32_C(   488.36), EASYSIMD_FLOAT32_C(  -116.86),
        EASYSIMD_FLOAT32_C(  -123.76), EASYSIMD_FLOAT32_C(   -85.46), EASYSIMD_FLOAT32_C(  -395.60), EASYSIMD_FLOAT32_C(  -799.86),
        EASYSIMD_FLOAT32_C(   677.35), EASYSIMD_FLOAT32_C(   270.67), EASYSIMD_FLOAT32_C(  -887.56), EASYSIMD_FLOAT32_C(   725.18) },
      { EASYSIMD_FLOAT32_C(    61.47), EASYSIMD_FLOAT32_C(  -848.75), EASYSIMD_FLOAT32_C(  -941.17), EASYSIMD_FLOAT32_C(  -221.91),
        EASYSIMD_FLOAT32_C(   392.93), EASYSIMD_FLOAT32_C(    38.45), EASYSIMD_FLOAT32_C(   198.62), EASYSIMD_FLOAT32_C(   165.26),
        EASYSIMD_FLOAT32_C(  -481.15), EASYSIMD_FLOAT32_C(  -648.25), EASYSIMD_FLOAT32_C(   912.49), EASYSIMD_FLOAT32_C(   198.88),
        EASYSIMD_FLOAT32_C(   535.07), EASYSIMD_FLOAT32_C(   853.64), EASYSIMD_FLOAT32_C(  -950.23), EASYSIMD_FLOAT32_C(  -538.25) },
      { EASYSIMD_FLOAT32_C( 16253.82), EASYSIMD_FLOAT32_C(  -986.90), EASYSIMD_FLOAT32_C(-128959.84), EASYSIMD_FLOAT32_C(-65529.99),
        EASYSIMD_FLOAT32_C(-104687.12), EASYSIMD_FLOAT32_C(   447.99), EASYSIMD_FLOAT32_C(   493.49), EASYSIMD_FLOAT32_C(  -475.96),
        EASYSIMD_FLOAT32_C(  -495.18), EASYSIMD_FLOAT32_C(-21885.76), EASYSIMD_FLOAT32_C(-117695.73), EASYSIMD_FLOAT32_C(-51191.29),
        EASYSIMD_FLOAT32_C(   695.35), EASYSIMD_FLOAT32_C(110331.93), EASYSIMD_FLOAT32_C( -1015.75), EASYSIMD_FLOAT32_C(  -735.43) } },
    { { EASYSIMD_FLOAT32_C(  -410.72), EASYSIMD_FLOAT32_C(  -917.00), EASYSIMD_FLOAT32_C(   -67.17), EASYSIMD_FLOAT32_C(  -908.76),
        EASYSIMD_FLOAT32_C(   534.17), EASYSIMD_FLOAT32_C(  -240.43), EASYSIMD_FLOAT32_C(  -833.05), EASYSIMD_FLOAT32_C(   947.68),
        EASYSIMD_FLOAT32_C(  -393.55), EASYSIMD_FLOAT32_C(  -335.35), EASYSIMD_FLOAT32_C(  -257.27), EASYSIMD_FLOAT32_C(    91.70),
        EASYSIMD_FLOAT32_C(  -820.62), EASYSIMD_FLOAT32_C(  -157.46), EASYSIMD_FLOAT32_C(  -507.25), EASYSIMD_FLOAT32_C(   705.00) },
      { EASYSIMD_FLOAT32_C(   880.62), EASYSIMD_FLOAT32_C(   602.69), EASYSIMD_FLOAT32_C(  -582.79), EASYSIMD_FLOAT32_C(  -873.03),
        EASYSIMD_FLOAT32_C(   911.31), EASYSIMD_FLOAT32_C(   402.01), EASYSIMD_FLOAT32_C(   647.66), EASYSIMD_FLOAT32_C(  -853.70),
        EASYSIMD_FLOAT32_C(  -128.13), EASYSIMD_FLOAT32_C(   472.58), EASYSIMD_FLOAT32_C(   575.79), EASYSIMD_FLOAT32_C(  -650.07),
        EASYSIMD_FLOAT32_C(   200.59), EASYSIMD_FLOAT32_C(  -601.69), EASYSIMD_FLOAT32_C(  -623.66), EASYSIMD_FLOAT32_C(   186.55) },
      { EASYSIMD_FLOAT32_C(-113407.37), EASYSIMD_FLOAT32_C(  -991.69), EASYSIMD_FLOAT32_C(-19451.78), EASYSIMD_FLOAT32_C( -1005.78),
        EASYSIMD_FLOAT32_C(   927.44), EASYSIMD_FLOAT32_C(  -498.87), EASYSIMD_FLOAT32_C(  -967.68), EASYSIMD_FLOAT32_C( -1015.75),
        EASYSIMD_FLOAT32_C(  -393.81), EASYSIMD_FLOAT32_C(  -479.87), EASYSIMD_FLOAT32_C(-73701.12), EASYSIMD_FLOAT32_C(-23539.25),
        EASYSIMD_FLOAT32_C(-52663.68), EASYSIMD_FLOAT32_C(-40829.92), EASYSIMD_FLOAT32_C(-131028.48), EASYSIMD_FLOAT32_C( 47820.80) } },
    { { EASYSIMD_FLOAT32_C(  -701.53), EASYSIMD_FLOAT32_C(  -543.04), EASYSIMD_FLOAT32_C(  -823.70), EASYSIMD_FLOAT32_C(    31.11),
        EASYSIMD_FLOAT32_C(   866.09), EASYSIMD_FLOAT32_C(   355.48), EASYSIMD_FLOAT32_C(   555.77), EASYSIMD_FLOAT32_C(   886.63),
        EASYSIMD_FLOAT32_C(  -481.79), EASYSIMD_FLOAT32_C(   592.72), EASYSIMD_FLOAT32_C(  -189.48), EASYSIMD_FLOAT32_C(  -527.23),
        EASYSIMD_FLOAT32_C(   860.31), EASYSIMD_FLOAT32_C(  -791.91), EASYSIMD_FLOAT32_C(   352.98), EASYSIMD_FLOAT32_C(   475.12) },
      { EASYSIMD_FLOAT32_C(  -598.83), EASYSIMD_FLOAT32_C(  -554.85), EASYSIMD_FLOAT32_C(   653.69), EASYSIMD_FLOAT32_C(   649.29),
        EASYSIMD_FLOAT32_C(   536.82), EASYSIMD_FLOAT32_C(   402.83), EASYSIMD_FLOAT32_C(   333.46), EASYSIMD_FLOAT32_C(   382.59),
        EASYSIMD_FLOAT32_C(   275.47), EASYSIMD_FLOAT32_C(  -132.03), EASYSIMD_FLOAT32_C(   922.98), EASYSIMD_FLOAT32_C(   461.96),
        EASYSIMD_FLOAT32_C(    85.99), EASYSIMD_FLOAT32_C(  -806.33), EASYSIMD_FLOAT32_C(    56.71), EASYSIMD_FLOAT32_C(  -909.56) },
      { EASYSIMD_FLOAT32_C(  -767.84), EASYSIMD_FLOAT32_C(  -575.86), EASYSIMD_FLOAT32_C(  -959.70), EASYSIMD_FLOAT32_C(  8030.48),
        EASYSIMD_FLOAT32_C(   890.84), EASYSIMD_FLOAT32_C(   500.00), EASYSIMD_FLOAT32_C( 89591.82), EASYSIMD_FLOAT32_C(131031.68),
        EASYSIMD_FLOAT32_C(  -499.98), EASYSIMD_FLOAT32_C(-37935.74), EASYSIMD_FLOAT32_C(-65534.97), EASYSIMD_FLOAT32_C(-118781.95),
        EASYSIMD_FLOAT32_C( 32765.98), EASYSIMD_FLOAT32_C(  -823.99), EASYSIMD_FLOAT32_C(   486.00), EASYSIMD_FLOAT32_C(-122847.74) } },
    { { EASYSIMD_FLOAT32_C(   944.34), EASYSIMD_FLOAT32_C(  -560.36), EASYSIMD_FLOAT32_C(    -4.19), EASYSIMD_FLOAT32_C(  -479.80),
        EASYSIMD_FLOAT32_C(    51.14), EASYSIMD_FLOAT32_C(  -569.84), EASYSIMD_FLOAT32_C(   718.42), EASYSIMD_FLOAT32_C(   535.49),
        EASYSIMD_FLOAT32_C(    31.54), EASYSIMD_FLOAT32_C(   142.94), EASYSIMD_FLOAT32_C(   349.37), EASYSIMD_FLOAT32_C(  -194.12),
        EASYSIMD_FLOAT32_C(  -641.81), EASYSIMD_FLOAT32_C(  -963.18), EASYSIMD_FLOAT32_C(  -221.26), EASYSIMD_FLOAT32_C(  -763.94) },
      { EASYSIMD_FLOAT32_C(   382.41), EASYSIMD_FLOAT32_C(  -851.69), EASYSIMD_FLOAT32_C(  -422.97), EASYSIMD_FLOAT32_C(  -784.40),
        EASYSIMD_FLOAT32_C(   546.04), EASYSIMD_FLOAT32_C(  -653.53), EASYSIMD_FLOAT32_C(   109.51), EASYSIMD_FLOAT32_C(   533.50),
        EASYSIMD_FLOAT32_C(   473.25), EASYSIMD_FLOAT32_C(  -161.95), EASYSIMD_FLOAT32_C(    68.59), EASYSIMD_FLOAT32_C(   837.71),
        EASYSIMD_FLOAT32_C(  -640.75), EASYSIMD_FLOAT32_C(   889.72), EASYSIMD_FLOAT32_C(  -478.90), EASYSIMD_FLOAT32_C(  -840.17) },
      { EASYSIMD_FLOAT32_C(130667.96), EASYSIMD_FLOAT32_C(  -883.99), EASYSIMD_FLOAT32_C(  -430.97), EASYSIMD_FLOAT32_C(-122879.98),
        EASYSIMD_FLOAT32_C( 13091.97), EASYSIMD_FLOAT32_C(  -701.84), EASYSIMD_FLOAT32_C( 32208.00), EASYSIMD_FLOAT32_C(   535.99),
        EASYSIMD_FLOAT32_C(   505.89), EASYSIMD_FLOAT32_C(  -175.95), EASYSIMD_FLOAT32_C(   351.37), EASYSIMD_FLOAT32_C(-54143.97),
        EASYSIMD_FLOAT32_C(  -641.81), EASYSIMD_FLOAT32_C( -1019.74), EASYSIMD_FLOAT32_C(  -510.90), EASYSIMD_FLOAT32_C( -1019.98) } },
    { { EASYSIMD_FLOAT32_C(  -711.36), EASYSIMD_FLOAT32_C(   214.45), EASYSIMD_FLOAT32_C(   194.07), EASYSIMD_FLOAT32_C(  -275.37),
        EASYSIMD_FLOAT32_C(  -213.28), EASYSIMD_FLOAT32_C(  -677.66), EASYSIMD_FLOAT32_C(   637.68), EASYSIMD_FLOAT32_C(  -440.50),
        EASYSIMD_FLOAT32_C(   586.73), EASYSIMD_FLOAT32_C(  -829.70), EASYSIMD_FLOAT32_C(  -729.50), EASYSIMD_FLOAT32_C(  -650.44),
        EASYSIMD_FLOAT32_C(   123.67), EASYSIMD_FLOAT32_C(   515.02), EASYSIMD_FLOAT32_C(    23.03), EASYSIMD_FLOAT32_C(  -964.09) },
      { EASYSIMD_FLOAT32_C(   587.18), EASYSIMD_FLOAT32_C(   655.47), EASYSIMD_FLOAT32_C(   537.23), EASYSIMD_FLOAT32_C(  -697.87),
        EASYSIMD_FLOAT32_C(   944.44), EASYSIMD_FLOAT32_C(  -768.05), EASYSIMD_FLOAT32_C(  -535.33), EASYSIMD_FLOAT32_C(   695.04),
        EASYSIMD_FLOAT32_C(  -482.15), EASYSIMD_FLOAT32_C(   455.20), EASYSIMD_FLOAT32_C(   561.68), EASYSIMD_FLOAT32_C(   223.05),
        EASYSIMD_FLOAT32_C(   840.35), EASYSIMD_FLOAT32_C(  -187.16), EASYSIMD_FLOAT32_C(   369.47), EASYSIMD_FLOAT32_C(  -383.36) },
      { EASYSIMD_FLOAT32_C(  -719.49), EASYSIMD_FLOAT32_C( 63487.21), EASYSIMD_FLOAT32_C( 50783.98), EASYSIMD_FLOAT32_C(-90111.98),
        EASYSIMD_FLOAT32_C(-64863.68), EASYSIMD_FLOAT32_C(  -933.68), EASYSIMD_FLOAT32_C(  -639.99), EASYSIMD_FLOAT32_C(-129925.12),
        EASYSIMD_FLOAT32_C(-124799.46), EASYSIMD_FLOAT32_C(-122875.74), EASYSIMD_FLOAT32_C(  -761.68), EASYSIMD_FLOAT32_C(-65436.93),
        EASYSIMD_FLOAT32_C( 31659.71), EASYSIMD_FLOAT32_C(-48105.99), EASYSIMD_FLOAT32_C(   369.48), EASYSIMD_FLOAT32_C(-130911.68) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r = easysimd_mm512_or_ps(a, b);
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_or_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_or_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_or_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(  2128798874), -INT32_C(  1460828476), -INT32_C(   811498352),  INT32_C(   956018954), -INT32_C(  1867631550), -INT32_C(   202668842), -INT32_C(   170676108), -INT32_C(  1095150044),
        -INT32_C(    12739781), -INT32_C(  1314379231), -INT32_C(  1182709329),  INT32_C(  1022524666), -INT32_C(  2033344336), -INT32_C(  1720076251), -INT32_C(  1936831128), -INT32_C(   565557341) },
      UINT16_C(34786),
      { -INT32_C(  2035153954),  INT32_C(   902783412),  INT32_C(   212977946), -INT32_C(   743612154),  INT32_C(  1636553447), -INT32_C(   105974933), -INT32_C(   918531713),  INT32_C(   240198192),
        -INT32_C(   644611291),  INT32_C(  2114937700),  INT32_C(   848019500),  INT32_C(   151336994),  INT32_C(  1969917962),  INT32_C(    40769667), -INT32_C(  1697862038), -INT32_C(   173531696) },
      { -INT32_C(  2066858976), -INT32_C(   889004641), -INT32_C(  1057059426), -INT32_C(   808909883),  INT32_C(   390345876), -INT32_C(  1239764404),  INT32_C(   810608224),  INT32_C(   589690883),
        -INT32_C(   743967692),  INT32_C(  1872734673), -INT32_C(    63988425),  INT32_C(   885848480),  INT32_C(  2035028013),  INT32_C(   590374595),  INT32_C(  1330937932), -INT32_C(  1385006727) },
      {  INT32_C(  2128798874), -INT32_C(     3146305), -INT32_C(   811498352),  INT32_C(   956018954), -INT32_C(  1867631550), -INT32_C(     4263057), -INT32_C(   112134273),  INT32_C(   796260915),
        -INT32_C(   608700619),  INT32_C(  2141187061), -INT32_C(    22028993),  INT32_C(  1022524666), -INT32_C(  2033344336), -INT32_C(  1720076251), -INT32_C(  1936831128), -INT32_C(    33915399) } },
    { {  INT32_C(  1065425261), -INT32_C(   106028862),  INT32_C(  1576459965),  INT32_C(    76726999), -INT32_C(  1786913070), -INT32_C(  1866879676), -INT32_C(  1478554322), -INT32_C(   195800698),
         INT32_C(   741594474), -INT32_C(  1306074635), -INT32_C(  1777328961), -INT32_C(  1332043042), -INT32_C(  1002039168), -INT32_C(   212533307), -INT32_C(  1835388148), -INT32_C(   293146748) },
      UINT16_C(47556),
      {  INT32_C(  1083947546),  INT32_C(  2086427244), -INT32_C(  1960953103),  INT32_C(   832806635), -INT32_C(  1238341278), -INT32_C(   135709604),  INT32_C(  1441164750),  INT32_C(  1980672603),
        -INT32_C(   776558236), -INT32_C(   179498236),  INT32_C(   947940429),  INT32_C(  1835607050), -INT32_C(   383542899), -INT32_C(  1528820267), -INT32_C(   688273798), -INT32_C(   716437647) },
      { -INT32_C(  1247411535),  INT32_C(  1672147989),  INT32_C(  1788553824), -INT32_C(   606665394),  INT32_C(  1959132062), -INT32_C(  2112314104), -INT32_C(   581430932), -INT32_C(   911039464),
        -INT32_C(  1132570201), -INT32_C(  1390466995), -INT32_C(  1592280237),  INT32_C(  1585311680), -INT32_C(   221101334),  INT32_C(  1400171239),  INT32_C(   338742779),  INT32_C(   417194865) },
      {  INT32_C(  1065425261), -INT32_C(   106028862), -INT32_C(   341886223),  INT32_C(    76726999), -INT32_C(  1786913070), -INT32_C(  1866879676), -INT32_C(   570524178), -INT32_C(     4280741),
        -INT32_C(    33621529), -INT32_C(  1306074635), -INT32_C(  1777328961),  INT32_C(  2138959818), -INT32_C(    67904529), -INT32_C(   134942729), -INT32_C(  1835388148), -INT32_C(   572659855) } },
    { { -INT32_C(  1982505924), -INT32_C(   684264316),  INT32_C(  1870220719),  INT32_C(   651097660),  INT32_C(   521707576), -INT32_C(  2039247477), -INT32_C(   879058086), -INT32_C(  1025214330),
         INT32_C(  1481357780),  INT32_C(  1563460013),  INT32_C(   197962191), -INT32_C(   684549473), -INT32_C(   956937669),  INT32_C(   827091415), -INT32_C(  1812076787),  INT32_C(   861331807) },
      UINT16_C(41370),
      { -INT32_C(  1155315829),  INT32_C(  1902441124),  INT32_C(   789251069), -INT32_C(   780515622),  INT32_C(  1496994060), -INT32_C(  2143271037),  INT32_C(   828481499),  INT32_C(  1574108114),
        -INT32_C(   417729214), -INT32_C(   447185433),  INT32_C(  1528062849), -INT32_C(  1238593878),  INT32_C(  1661954016), -INT32_C(  1981591634), -INT32_C(  1061534482), -INT32_C(  2128704193) },
      {  INT32_C(  1785214850),  INT32_C(   894419124), -INT32_C(   829332700), -INT32_C(   763052558), -INT32_C(   751463388), -INT32_C(   765716253), -INT32_C(  1684924580),  INT32_C(   639414436),
        -INT32_C(  1668250137),  INT32_C(  1775361861),  INT32_C(   892822083),  INT32_C(  1141357600),  INT32_C(   857160784), -INT32_C(  1341819820),  INT32_C(   793483403),  INT32_C(   810903368) },
      { -INT32_C(  1982505924),  INT32_C(  1970270900),  INT32_C(  1870220719), -INT32_C(   738263046), -INT32_C(   616573652), -INT32_C(  2039247477), -INT32_C(   879058086),  INT32_C(  2145319926),
        -INT32_C(     6687257),  INT32_C(  1563460013),  INT32_C(   197962191), -INT32_C(   684549473), -INT32_C(   956937669), -INT32_C(  1176010754), -INT32_C(  1812076787), -INT32_C(  1319112833) } },
    { {  INT32_C(   852289260),  INT32_C(   144416197),  INT32_C(   540922624), -INT32_C(   547076977), -INT32_C(   736985984),  INT32_C(  2072254704), -INT32_C(   123023440),  INT32_C(   589889335),
        -INT32_C(  1420430107), -INT32_C(  1833700974),  INT32_C(  1404236228), -INT32_C(  1238231499), -INT32_C(  2104867694),  INT32_C(   217911132),  INT32_C(   352692190), -INT32_C(  1942409817) },
      UINT16_C(36386),
      { -INT32_C(   343952329), -INT32_C(   119782586), -INT32_C(   938602090),  INT32_C(  1376625095),  INT32_C(   560032035),  INT32_C(  2076721014), -INT32_C(  1901564075),  INT32_C(   857524988),
        -INT32_C(  1004626818),  INT32_C(  1992096479), -INT32_C(   750859508), -INT32_C(  1893381268),  INT32_C(   732989109),  INT32_C(   463894982), -INT32_C(   458666264), -INT32_C(  1759984359) },
      {  INT32_C(  1096562273),  INT32_C(  1035409457),  INT32_C(  1326511587), -INT32_C(   153143743), -INT32_C(  2111729732), -INT32_C(   241318136),  INT32_C(   785729045),  INT32_C(  1841753356),
         INT32_C(  1437475364),  INT32_C(   512910650), -INT32_C(  1670536613), -INT32_C(  1785574184), -INT32_C(   468208676), -INT32_C(  1881819782),  INT32_C(   129936379), -INT32_C(  1133149032) },
      {  INT32_C(   852289260), -INT32_C(    33596553),  INT32_C(   540922624), -INT32_C(   547076977), -INT32_C(   736985984), -INT32_C(    69337218), -INT32_C(   123023440),  INT32_C(   589889335),
        -INT32_C(  1420430107),  INT32_C(  2126446591), -INT32_C(   545264801), -INT32_C(  1615376388), -INT32_C(  2104867694),  INT32_C(   217911132),  INT32_C(   352692190), -INT32_C(  1082276455) } },
    { { -INT32_C(   535747674), -INT32_C(   453074039),  INT32_C(   511732806), -INT32_C(  1800203592),  INT32_C(  1064946629),  INT32_C(  2077118080), -INT32_C(  1836872455), -INT32_C(  1236338672),
        -INT32_C(  1533583333),  INT32_C(  1233687811), -INT32_C(  1167587327), -INT32_C(   548529382),  INT32_C(  1713293286),  INT32_C(   266530070), -INT32_C(  1969134215),  INT32_C(  2017521757) },
      UINT16_C(55120),
      { -INT32_C(  1519561955),  INT32_C(    95317661),  INT32_C(  1998637096), -INT32_C(   969013336),  INT32_C(  1337152621),  INT32_C(    95694180), -INT32_C(   134934089), -INT32_C(  1479588470),
         INT32_C(   910965913), -INT32_C(   751043926),  INT32_C(  1800035267), -INT32_C(   802060190),  INT32_C(  1092609501), -INT32_C(   918105070),  INT32_C(  1874869477),  INT32_C(   437686145) },
      {  INT32_C(  1984979915),  INT32_C(   541690717),  INT32_C(  1233884135), -INT32_C(   132530917), -INT32_C(  1271318110), -INT32_C(   226656243),  INT32_C(  1029848508), -INT32_C(  1739032371),
         INT32_C(   957262043),  INT32_C(   458839860),  INT32_C(   107275754),  INT32_C(  1157529250), -INT32_C(   990365513),  INT32_C(  1958114744),  INT32_C(  2142312626),  INT32_C(  1813514641) },
      { -INT32_C(   535747674), -INT32_C(   453074039),  INT32_C(   511732806), -INT32_C(  1800203592), -INT32_C(     4489745),  INT32_C(  2077118080), -INT32_C(      574017), -INT32_C(  1236338672),
         INT32_C(  1062124763), -INT32_C(   612630594),  INT32_C(  1869545451), -INT32_C(   548529382), -INT32_C(   973079041),  INT32_C(   266530070),  INT32_C(  2146516215),  INT32_C(  2115932049) } },
    { { -INT32_C(   425384270),  INT32_C(  1744961406), -INT32_C(  2039585308), -INT32_C(  1698009629),  INT32_C(  1566491301), -INT32_C(   372173513), -INT32_C(  1100381651),  INT32_C(  1043038604),
         INT32_C(   623169703), -INT32_C(  1282529841),  INT32_C(  1849293962),  INT32_C(   235406185), -INT32_C(    60069947), -INT32_C(  1444529028),  INT32_C(  1265127359),  INT32_C(  2005504976) },
      UINT16_C(44387),
      {  INT32_C(   735196061),  INT32_C(   539450598), -INT32_C(   752643638),  INT32_C(   188410271), -INT32_C(   884492571), -INT32_C(   954595745), -INT32_C(   631575983), -INT32_C(     7881118),
        -INT32_C(   685090319), -INT32_C(  2131275338), -INT32_C(  2108482590), -INT32_C(   376599292), -INT32_C(  1548364733),  INT32_C(   745197531),  INT32_C(   470271418),  INT32_C(  1947963011) },
      { -INT32_C(  1639234072),  INT32_C(  2032026518),  INT32_C(  1660645982),  INT32_C(  1129023743),  INT32_C(   937820252), -INT32_C(  1973137200), -INT32_C(  1733924075), -INT32_C(   519257607),
        -INT32_C(  1669375994), -INT32_C(   116023653),  INT32_C(   257626128), -INT32_C(   179132776),  INT32_C(  1999386791), -INT32_C(  1644064631), -INT32_C(   180901893),  INT32_C(  1859535720) },
      { -INT32_C(  1076135939),  INT32_C(  2034196470), -INT32_C(  2039585308), -INT32_C(  1698009629),  INT32_C(  1566491301), -INT32_C(   813803809), -INT32_C(   620827819),  INT32_C(  1043038604),
        -INT32_C(   545302025), -INT32_C(  1282529841), -INT32_C(  1889854478), -INT32_C(    35672164), -INT32_C(    60069947), -INT32_C(  1100226597),  INT32_C(  1265127359),  INT32_C(  2128596971) } },
    { {  INT32_C(   906647195),  INT32_C(    70197492), -INT32_C(   921466320), -INT32_C(   658610639),  INT32_C(   659548830),  INT32_C(  1992708219),  INT32_C(  1600912887), -INT32_C(   590527936),
        -INT32_C(  1944922216),  INT32_C(   680542967), -INT32_C(    34495540), -INT32_C(  1462391031),  INT32_C(   349119641),  INT32_C(  1804309876), -INT32_C(   741673326), -INT32_C(   793798856) },
      UINT16_C(49519),
      { -INT32_C(   335321508),  INT32_C(  2140197006), -INT32_C(  1557227059),  INT32_C(   298305601),  INT32_C(  1738947804),  INT32_C(  1918777767), -INT32_C(  1156999668), -INT32_C(  1015252889),
         INT32_C(  1840283871),  INT32_C(   502087760),  INT32_C(   465574873), -INT32_C(  1070823197),  INT32_C(  1797771972),  INT32_C(   400393483), -INT32_C(  2083330276),  INT32_C(  1044795231) },
      {  INT32_C(   531429071),  INT32_C(   272406838), -INT32_C(  1741947468),  INT32_C(  1230526341),  INT32_C(   901021481),  INT32_C(   558666245), -INT32_C(   660332679),  INT32_C(  1024977518),
         INT32_C(   391955425),  INT32_C(   271030364),  INT32_C(   447238805), -INT32_C(   748486487), -INT32_C(  2063067008),  INT32_C(   598103210), -INT32_C(   503625101),  INT32_C(   354292276) },
      { -INT32_C(     5245217),  INT32_C(  2143082942), -INT32_C(  1154482691),  INT32_C(  1507844037),  INT32_C(   659548830),  INT32_C(  1935588263), -INT32_C(    72507523), -INT32_C(   590527936),
         INT32_C(  2147271679),  INT32_C(   680542967), -INT32_C(    34495540), -INT32_C(  1462391031),  INT32_C(   349119641),  INT32_C(  1804309876), -INT32_C(   470028417),  INT32_C(  1063149439) } },
    { {  INT32_C(   825064149), -INT32_C(  1472048109),  INT32_C(  1355016871),  INT32_C(  1797465835), -INT32_C(   386913474), -INT32_C(   217344384),  INT32_C(   349505504), -INT32_C(   282397927),
        -INT32_C(  2128586898),  INT32_C(  1378443947),  INT32_C(   950266957),  INT32_C(  1369687571),  INT32_C(  1916441586),  INT32_C(   157631785),  INT32_C(  1713191500), -INT32_C(  1672132818) },
      UINT16_C(30110),
      {  INT32_C(  1188579869),  INT32_C(  1060316572),  INT32_C(      411229), -INT32_C(   778831721), -INT32_C(   820528022), -INT32_C(   502635579),  INT32_C(   506083529), -INT32_C(   241907500),
        -INT32_C(  1355322350), -INT32_C(   286365039),  INT32_C(  1223619760),  INT32_C(  1444512236),  INT32_C(    36057149),  INT32_C(  1575235732),  INT32_C(  1014763112), -INT32_C(   382922537) },
      {  INT32_C(   228091004),  INT32_C(  2147190735),  INT32_C(  1757931899), -INT32_C(  1480662678), -INT32_C(  1515592687),  INT32_C(  2080607508),  INT32_C(  1941471132),  INT32_C(   190637455),
         INT32_C(   421066058), -INT32_C(   141028484),  INT32_C(  1717526780),  INT32_C(  1376591425),  INT32_C(   402175490), -INT32_C(   543950013),  INT32_C(   156388474),  INT32_C(  2064953137) },
      {  INT32_C(   825064149),  INT32_C(  2147198943),  INT32_C(  1757933439), -INT32_C(   138413569), -INT32_C(   272630661), -INT32_C(   217344384),  INT32_C(   349505504), -INT32_C(    69407265),
        -INT32_C(  1086325414),  INT32_C(  1378443947),  INT32_C(  1862268156),  INT32_C(  1369687571),  INT32_C(   402568767), -INT32_C(   537396265),  INT32_C(  1031687802), -INT32_C(  1672132818) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].src));
    easysimd__m512 a = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].a));
    easysimd__m512 b = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].b));
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_or_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_or_ps");
    easysimd_test_x86_assert_equal_i32x16(easysimd_mm512_castps_si512(r), easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_castps_si512(easysimd_mm512_mask_or_ps(easysimd_mm512_castsi512_ps(src), k, easysimd_mm512_castsi512_ps(a), easysimd_mm512_castsi512_ps(b)));

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_or_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(11428),
      {  INT32_C(   759111828),  INT32_C(  2005744407), -INT32_C(  1365913951),  INT32_C(   409245472),  INT32_C(  1108584366), -INT32_C(   644969081), -INT32_C(  1433878634), -INT32_C(   808047557),
         INT32_C(  1694242124), -INT32_C(   237270704),  INT32_C(  2023714903), -INT32_C(  1232076025),  INT32_C(   855155883), -INT32_C(   938768847), -INT32_C(  2139974587),  INT32_C(   223365568) },
      { -INT32_C(  1351529378),  INT32_C(   765480150), -INT32_C(  1012580164), -INT32_C(   294046141),  INT32_C(   186675674),  INT32_C(  1037249783), -INT32_C(  2135079232), -INT32_C(   309523057),
         INT32_C(   815595098),  INT32_C(   106773834), -INT32_C(  1060502659),  INT32_C(   330253113), -INT32_C(  1407266891), -INT32_C(  1142295813), -INT32_C(   952326344),  INT32_C(   246729140) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   273297731),  INT32_C(           0),  INT32_C(           0), -INT32_C(    35672585),  INT32_C(           0), -INT32_C(   270586433),
         INT32_C(           0),  INT32_C(           0), -INT32_C(   119573633), -INT32_C(  1212202177),  INT32_C(           0), -INT32_C(    68421893),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(20936),
      { -INT32_C(  1668410818), -INT32_C(   476116199),  INT32_C(  2049366219),  INT32_C(   155900907),  INT32_C(  1912292744),  INT32_C(  1024996097), -INT32_C(  1358508550),  INT32_C(   436260571),
        -INT32_C(    88699167),  INT32_C(  1692226969),  INT32_C(   417268525),  INT32_C(  1730226655),  INT32_C(  1876434286),  INT32_C(  1269625168), -INT32_C(  1711623234),  INT32_C(  1655962241) },
      {  INT32_C(   559704456), -INT32_C(   343590466),  INT32_C(   469984317), -INT32_C(    58514035), -INT32_C(  1838458046),  INT32_C(   182261836),  INT32_C(  1285871563),  INT32_C(  1504663505),
         INT32_C(  2138704833), -INT32_C(  2106916795), -INT32_C(   224498076), -INT32_C(   705814125), -INT32_C(   915908228),  INT32_C(  1020478833), -INT32_C(   292980963), -INT32_C(  1874315057) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(    36962321),  INT32_C(           0),  INT32_C(           0), -INT32_C(   274276357),  INT32_C(  1538252763),
        -INT32_C(       94239),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   268935810),  INT32_C(           0), -INT32_C(      264257),  INT32_C(           0) } },
    { UINT16_C(49731),
      {  INT32_C(  2059569167), -INT32_C(  1461115126),  INT32_C(   130645017), -INT32_C(  1168095662), -INT32_C(   486551025), -INT32_C(  1755702258),  INT32_C(  1406085387), -INT32_C(   938142791),
        -INT32_C(  1505503077),  INT32_C(   407776511), -INT32_C(    98625368),  INT32_C(  1840545630),  INT32_C(  1615836241), -INT32_C(   604526128), -INT32_C(  1959868462),  INT32_C(  1968456922) },
      {  INT32_C(   454792988),  INT32_C(  1798531779), -INT32_C(   530165118),  INT32_C(   592255697), -INT32_C(  1618764594),  INT32_C(   410679878),  INT32_C(   463776065),  INT32_C(   160561389),
         INT32_C(  1378135183), -INT32_C(  1732421866),  INT32_C(  2054693801),  INT32_C(   211666238), -INT32_C(  1465180062), -INT32_C(   591386981), -INT32_C(  1124571953), -INT32_C(   339375780) },
      {  INT32_C(  2077990687), -INT32_C(   335843381),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1542433099),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1728086017),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1073813537), -INT32_C(     2765346) } },
    { UINT16_C(59701),
      { -INT32_C(    79672259),  INT32_C(  1562307044),  INT32_C(    19029091),  INT32_C(   337740905), -INT32_C(   314917844), -INT32_C(  1856960103), -INT32_C(  1977962811), -INT32_C(   680308839),
        -INT32_C(  2133675109),  INT32_C(    14545052), -INT32_C(  1241383091), -INT32_C(  1328930172),  INT32_C(  2023556575), -INT32_C(   737546482),  INT32_C(   912139165),  INT32_C(   235786866) },
      {  INT32_C(   579788677),  INT32_C(   488795088), -INT32_C(   288152726),  INT32_C(   614440261), -INT32_C(  1315160926), -INT32_C(   930765525),  INT32_C(  1006625736),  INT32_C(   994643125),
        -INT32_C(  1134700564), -INT32_C(  1395032254), -INT32_C(   409228126), -INT32_C(   334742967), -INT32_C(  1566726026),  INT32_C(   376054350), -INT32_C(  1152292603),  INT32_C(  1660328566) },
      { -INT32_C(    70328387),  INT32_C(           0), -INT32_C(   269254805),  INT32_C(           0), -INT32_C(    37831506), -INT32_C(   640307781),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(  1126172673),  INT32_C(           0),  INT32_C(           0), -INT32_C(    53593395),  INT32_C(           0), -INT32_C(   697569458), -INT32_C(  1084265571),  INT32_C(  1862261366) } },
    { UINT16_C(21361),
      { -INT32_C(   137186530), -INT32_C(    73173920),  INT32_C(  1731587163),  INT32_C(  1964026840), -INT32_C(  1198039475), -INT32_C(   970875532),  INT32_C(  1314953048),  INT32_C(   396481273),
        -INT32_C(   452037755),  INT32_C(  1122021863),  INT32_C(  1990858142),  INT32_C(   250395329), -INT32_C(  1933147368),  INT32_C(  2018699296),  INT32_C(  2043065215),  INT32_C(   177235845) },
      { -INT32_C(  1041195046), -INT32_C(   301674416), -INT32_C(  1486508314), -INT32_C(  2135600792), -INT32_C(   183730988), -INT32_C(   479371420), -INT32_C(  1755565038),  INT32_C(  1990323611),
        -INT32_C(   600338036),  INT32_C(  1221278562),  INT32_C(  1374630121),  INT32_C(  1439802497), -INT32_C(  2075468256),  INT32_C(  1332262973), -INT32_C(  2014919444),  INT32_C(  1040025521) },
      { -INT32_C(   135086114),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(    39878947), -INT32_C(   412229772), -INT32_C(   545474726),  INT32_C(           0),
        -INT32_C(    46140531),  INT32_C(  1256963047),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1932599496),  INT32_C(           0), -INT32_C(     1640449),  INT32_C(           0) } },
    { UINT16_C(13593),
      { -INT32_C(   445613286), -INT32_C(  1307223357),  INT32_C(  2102826667),  INT32_C(   911964139),  INT32_C(  1676581115), -INT32_C(   852960537),  INT32_C(  1599396193),  INT32_C(   831810839),
        -INT32_C(  1424620312),  INT32_C(   173878110), -INT32_C(  1400392511),  INT32_C(   652403243),  INT32_C(  1636487290),  INT32_C(   204386986), -INT32_C(  1570012533), -INT32_C(   657195024) },
      {  INT32_C(  1652877571), -INT32_C(   714284780), -INT32_C(  1065159787),  INT32_C(  1340564693), -INT32_C(   542084811), -INT32_C(  1360273885),  INT32_C(  1347507808),  INT32_C(  1495868501),
         INT32_C(   582724878),  INT32_C(   620242830), -INT32_C(   270239206),  INT32_C(   322882526),  INT32_C(  1609821756),  INT32_C(   739172044), -INT32_C(  1971495115), -INT32_C(  1847351677) },
      { -INT32_C(   403375333),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2147448831), -INT32_C(       67073),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(  1413501458),  INT32_C(           0), -INT32_C(   270008613),  INT32_C(           0),  INT32_C(  2147221118),  INT32_C(   741277422),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(40531),
      { -INT32_C(  1396317772), -INT32_C(   366551291),  INT32_C(   213189838),  INT32_C(   200995352),  INT32_C(  1609156177),  INT32_C(  1891507955), -INT32_C(  1944698199), -INT32_C(  2027197997),
         INT32_C(  1345581130), -INT32_C(  1640343088),  INT32_C(  1990979422),  INT32_C(   864134625),  INT32_C(  1603431020),  INT32_C(   852447625),  INT32_C(  1723786643), -INT32_C(  1729173170) },
      { -INT32_C(  1427627558), -INT32_C(   649584005), -INT32_C(   212864238),  INT32_C(    86429849), -INT32_C(  1016809158), -INT32_C(  1644874998),  INT32_C(  1711518744),  INT32_C(  2013196701),
        -INT32_C(  1910380781),  INT32_C(   459762185), -INT32_C(   166742435), -INT32_C(  1057278586), -INT32_C(   125542418), -INT32_C(  1433044590), -INT32_C(   888104659), -INT32_C(  1656615030) },
      { -INT32_C(  1360141826), -INT32_C(    76611713),  INT32_C(           0),  INT32_C(           0), -INT32_C(   538050693),  INT32_C(           0), -INT32_C(   300419399),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1619035175), -INT32_C(   156237985), -INT32_C(   201607705), -INT32_C(     6914066),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1645285426) } },
    { UINT16_C(25846),
      { -INT32_C(  1831928021),  INT32_C(   709372699),  INT32_C(   492817954),  INT32_C(   326979215), -INT32_C(   594800826), -INT32_C(   898254150), -INT32_C(   942014587), -INT32_C(   936652899),
        -INT32_C(   363071025),  INT32_C(  1192534821),  INT32_C(    23360370),  INT32_C(   135586241), -INT32_C(  1411079951), -INT32_C(   545957542), -INT32_C(   173584552), -INT32_C(   306261474) },
      { -INT32_C(   237561396),  INT32_C(   775482300),  INT32_C(   539991135),  INT32_C(  1864909694),  INT32_C(  1041894628), -INT32_C(  1105293467), -INT32_C(    38550050),  INT32_C(  1676308887),
         INT32_C(  1196737162),  INT32_C(   209030317), -INT32_C(  1490180823), -INT32_C(   870951448), -INT32_C(   955568030), -INT32_C(  1635440193), -INT32_C(  2053424658),  INT32_C(   904496554) },
      {  INT32_C(           0),  INT32_C(   779676607),  INT32_C(  1031790207),  INT32_C(           0), -INT32_C(    23195674), -INT32_C(    25182209), -INT32_C(      262177), -INT32_C(   336855137),
         INT32_C(           0),  INT32_C(           0), -INT32_C(  1485965445),  INT32_C(           0),  INT32_C(           0), -INT32_C(   537560577), -INT32_C(   171999234),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].a));
    easysimd__m512 b = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].b));
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_or_ps(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_or_ps");
    easysimd_test_x86_assert_equal_i32x16(easysimd_mm512_castps_si512(r), easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_castps_si512(easysimd_mm512_maskz_or_ps(k, easysimd_mm512_castsi512_ps(a), easysimd_mm512_castsi512_ps(b)));

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_or_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -40.48), EASYSIMD_FLOAT64_C(  -322.78), EASYSIMD_FLOAT64_C(  -915.31), EASYSIMD_FLOAT64_C(   424.37),
        EASYSIMD_FLOAT64_C(   358.24), EASYSIMD_FLOAT64_C(   951.77), EASYSIMD_FLOAT64_C(   466.94), EASYSIMD_FLOAT64_C(  -212.54) },
      { EASYSIMD_FLOAT64_C(  -206.62), EASYSIMD_FLOAT64_C(  -632.27), EASYSIMD_FLOAT64_C(  -561.37), EASYSIMD_FLOAT64_C(  -939.45),
        EASYSIMD_FLOAT64_C(   583.50), EASYSIMD_FLOAT64_C(   851.39), EASYSIMD_FLOAT64_C(  -748.10), EASYSIMD_FLOAT64_C(   610.87) },
      { EASYSIMD_FLOAT64_C(  -240.00), EASYSIMD_FLOAT64_C(-98023.69), EASYSIMD_FLOAT64_C(  -947.37), EASYSIMD_FLOAT64_C(-130559.73),
        EASYSIMD_FLOAT64_C( 92157.44), EASYSIMD_FLOAT64_C(  1015.91), EASYSIMD_FLOAT64_C(-128764.94), EASYSIMD_FLOAT64_C(-56511.75) } },
    { { EASYSIMD_FLOAT64_C(   883.05), EASYSIMD_FLOAT64_C(  -496.23), EASYSIMD_FLOAT64_C(  -209.68), EASYSIMD_FLOAT64_C(  -122.92),
        EASYSIMD_FLOAT64_C(    19.75), EASYSIMD_FLOAT64_C(   -49.24), EASYSIMD_FLOAT64_C(   492.96), EASYSIMD_FLOAT64_C(  -866.69) },
      { EASYSIMD_FLOAT64_C(  -385.90), EASYSIMD_FLOAT64_C(   247.26), EASYSIMD_FLOAT64_C(  -268.45), EASYSIMD_FLOAT64_C(   217.01),
        EASYSIMD_FLOAT64_C(  -674.00), EASYSIMD_FLOAT64_C(   155.01), EASYSIMD_FLOAT64_C(  -699.71), EASYSIMD_FLOAT64_C(  -101.01) },
      { EASYSIMD_FLOAT64_C(-113126.40), EASYSIMD_FLOAT64_C(  -510.75), EASYSIMD_FLOAT64_C(  -431.50), EASYSIMD_FLOAT64_C(  -507.68),
        EASYSIMD_FLOAT64_C( -6096.00), EASYSIMD_FLOAT64_C(  -223.97), EASYSIMD_FLOAT64_C(-130559.89), EASYSIMD_FLOAT64_C(-27990.62) } },
    { { EASYSIMD_FLOAT64_C(   290.86), EASYSIMD_FLOAT64_C(   407.11), EASYSIMD_FLOAT64_C(   359.33), EASYSIMD_FLOAT64_C(  -773.11),
        EASYSIMD_FLOAT64_C(   409.90), EASYSIMD_FLOAT64_C(  -892.84), EASYSIMD_FLOAT64_C(   -43.07), EASYSIMD_FLOAT64_C(   160.79) },
      { EASYSIMD_FLOAT64_C(   465.85), EASYSIMD_FLOAT64_C(  -467.47), EASYSIMD_FLOAT64_C(  -782.08), EASYSIMD_FLOAT64_C(  -490.24),
        EASYSIMD_FLOAT64_C(   592.42), EASYSIMD_FLOAT64_C(   806.95), EASYSIMD_FLOAT64_C(   470.98), EASYSIMD_FLOAT64_C(  -245.49) },
      { EASYSIMD_FLOAT64_C(   499.87), EASYSIMD_FLOAT64_C(  -471.49), EASYSIMD_FLOAT64_C(-124766.50), EASYSIMD_FLOAT64_C(-125631.46),
        EASYSIMD_FLOAT64_C(113143.90), EASYSIMD_FLOAT64_C(  -894.97), EASYSIMD_FLOAT64_C(  -479.00), EASYSIMD_FLOAT64_C(  -246.00) } },
    { { EASYSIMD_FLOAT64_C(    88.45), EASYSIMD_FLOAT64_C(   537.22), EASYSIMD_FLOAT64_C(  -949.80), EASYSIMD_FLOAT64_C(   743.04),
        EASYSIMD_FLOAT64_C(   296.63), EASYSIMD_FLOAT64_C(  -483.08), EASYSIMD_FLOAT64_C(   917.67), EASYSIMD_FLOAT64_C(  -778.36) },
      { EASYSIMD_FLOAT64_C(    89.00), EASYSIMD_FLOAT64_C(   -93.55), EASYSIMD_FLOAT64_C(  -328.25), EASYSIMD_FLOAT64_C(  -923.97),
        EASYSIMD_FLOAT64_C(   880.96), EASYSIMD_FLOAT64_C(   356.45), EASYSIMD_FLOAT64_C(   842.15), EASYSIMD_FLOAT64_C(    25.32) },
      { EASYSIMD_FLOAT64_C(    89.45), EASYSIMD_FLOAT64_C(-24495.81), EASYSIMD_FLOAT64_C(-121574.40), EASYSIMD_FLOAT64_C( -1023.98),
        EASYSIMD_FLOAT64_C(112891.91), EASYSIMD_FLOAT64_C(  -487.47), EASYSIMD_FLOAT64_C(   991.69), EASYSIMD_FLOAT64_C( -6483.92) } },
    { { EASYSIMD_FLOAT64_C(  -519.30), EASYSIMD_FLOAT64_C(    88.55), EASYSIMD_FLOAT64_C(   186.42), EASYSIMD_FLOAT64_C(  -780.93),
        EASYSIMD_FLOAT64_C(   918.96), EASYSIMD_FLOAT64_C(   887.03), EASYSIMD_FLOAT64_C(   360.26), EASYSIMD_FLOAT64_C(   873.25) },
      { EASYSIMD_FLOAT64_C(   651.35), EASYSIMD_FLOAT64_C(  -822.40), EASYSIMD_FLOAT64_C(   -37.73), EASYSIMD_FLOAT64_C(  -102.45),
        EASYSIMD_FLOAT64_C(   167.81), EASYSIMD_FLOAT64_C(   376.55), EASYSIMD_FLOAT64_C(   698.04), EASYSIMD_FLOAT64_C(    15.57) },
      { EASYSIMD_FLOAT64_C(  -655.37), EASYSIMD_FLOAT64_C(-32460.80), EASYSIMD_FLOAT64_C(  -190.92), EASYSIMD_FLOAT64_C(-26623.95),
        EASYSIMD_FLOAT64_C( 59391.49), EASYSIMD_FLOAT64_C(129935.87), EASYSIMD_FLOAT64_C( 97607.62), EASYSIMD_FLOAT64_C(  4021.92) } },
    { { EASYSIMD_FLOAT64_C(  -325.10), EASYSIMD_FLOAT64_C(    24.24), EASYSIMD_FLOAT64_C(  -628.64), EASYSIMD_FLOAT64_C(   379.01),
        EASYSIMD_FLOAT64_C(    98.66), EASYSIMD_FLOAT64_C(   182.61), EASYSIMD_FLOAT64_C(  -798.49), EASYSIMD_FLOAT64_C(  -146.70) },
      { EASYSIMD_FLOAT64_C(   387.34), EASYSIMD_FLOAT64_C(   -48.74), EASYSIMD_FLOAT64_C(   849.44), EASYSIMD_FLOAT64_C(   502.00),
        EASYSIMD_FLOAT64_C(  -892.81), EASYSIMD_FLOAT64_C(   587.22), EASYSIMD_FLOAT64_C(  -574.98), EASYSIMD_FLOAT64_C(   384.87) },
      { EASYSIMD_FLOAT64_C(  -455.37), EASYSIMD_FLOAT64_C(  -392.00), EASYSIMD_FLOAT64_C(  -885.95), EASYSIMD_FLOAT64_C(   511.01),
        EASYSIMD_FLOAT64_C(-28602.00), EASYSIMD_FLOAT64_C( 46814.24), EASYSIMD_FLOAT64_C(  -831.00), EASYSIMD_FLOAT64_C(  -422.00) } },
    { { EASYSIMD_FLOAT64_C(   462.25), EASYSIMD_FLOAT64_C(  -905.39), EASYSIMD_FLOAT64_C(  -831.12), EASYSIMD_FLOAT64_C(  -716.46),
        EASYSIMD_FLOAT64_C(   498.06), EASYSIMD_FLOAT64_C(   927.73), EASYSIMD_FLOAT64_C(   312.19), EASYSIMD_FLOAT64_C(  -955.24) },
      { EASYSIMD_FLOAT64_C(  -784.67), EASYSIMD_FLOAT64_C(    13.39), EASYSIMD_FLOAT64_C(   973.39), EASYSIMD_FLOAT64_C(  -224.05),
        EASYSIMD_FLOAT64_C(   392.29), EASYSIMD_FLOAT64_C(  -737.84), EASYSIMD_FLOAT64_C(  -812.63), EASYSIMD_FLOAT64_C(    85.02) },
      { EASYSIMD_FLOAT64_C(-118357.76), EASYSIMD_FLOAT64_C( -3943.87), EASYSIMD_FLOAT64_C( -1023.50), EASYSIMD_FLOAT64_C(-62237.99),
        EASYSIMD_FLOAT64_C(   506.31), EASYSIMD_FLOAT64_C( -1024.00), EASYSIMD_FLOAT64_C(-114288.64), EASYSIMD_FLOAT64_C(-30567.75) } },
    { { EASYSIMD_FLOAT64_C(   979.20), EASYSIMD_FLOAT64_C(  -512.78), EASYSIMD_FLOAT64_C(  -370.35), EASYSIMD_FLOAT64_C(  -474.08),
        EASYSIMD_FLOAT64_C(   129.12), EASYSIMD_FLOAT64_C(    46.51), EASYSIMD_FLOAT64_C(  -292.51), EASYSIMD_FLOAT64_C(  -218.79) },
      { EASYSIMD_FLOAT64_C(   798.26), EASYSIMD_FLOAT64_C(   278.63), EASYSIMD_FLOAT64_C(   902.40), EASYSIMD_FLOAT64_C(   696.72),
        EASYSIMD_FLOAT64_C(  -256.04), EASYSIMD_FLOAT64_C(   749.65), EASYSIMD_FLOAT64_C(  -619.61), EASYSIMD_FLOAT64_C(   269.84) },
      { EASYSIMD_FLOAT64_C(   991.45), EASYSIMD_FLOAT64_C(-71395.84), EASYSIMD_FLOAT64_C(-127867.73), EASYSIMD_FLOAT64_C(-122460.48),
        EASYSIMD_FLOAT64_C(  -258.25), EASYSIMD_FLOAT64_C( 11994.94), EASYSIMD_FLOAT64_C(-79310.62), EASYSIMD_FLOAT64_C(  -445.84) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r = easysimd_mm512_or_pd(a, b);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_or_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 1322694804681624692), -INT64_C( 6220656704802057389),  INT64_C( 6910152754369147041), -INT64_C( 7538055854328849453),
         INT64_C( 3169319009436020990),  INT64_C(  367692512894360513), -INT64_C( 7697084138346449282),  INT64_C( 5407839422225767249) },
      UINT8_C( 49),
      { -INT64_C( 3761011257826003873),  INT64_C( 5738770482185846743),  INT64_C( 9047070442163107013),  INT64_C( 7815458449609327044),
        -INT64_C( 8339179195020509110),  INT64_C( 8011999029023110535), -INT64_C( 6058967782045040182),  INT64_C( 1563707618410673465) },
      { -INT64_C(  277018150962188825), -INT64_C(  404182463127103158), -INT64_C( 9159668226292510897),  INT64_C( 2443056815446414087),
        -INT64_C( 7558759479702161496),  INT64_C( 6118404002033760189), -INT64_C( 6285214590996623361), -INT64_C( 5648532039763587519) },
      { -INT64_C(    4515189066482177), -INT64_C( 6220656704802057389),  INT64_C( 6910152754369147041), -INT64_C( 7538055854328849453),
        -INT64_C( 6963148875326687254),  INT64_C( 9221391472509515711), -INT64_C( 7697084138346449282),  INT64_C( 5407839422225767249) } },
    { { -INT64_C(  378611073117431591),  INT64_C( 4123273310696327902),  INT64_C( 8244196672863716127),  INT64_C( 2273624320044295123),
         INT64_C( 2384659589498231778), -INT64_C( 1242925535621480371),  INT64_C( 6387257131878268672), -INT64_C(  188937762035625221) },
      UINT8_C( 30),
      {  INT64_C( 3816380896934698149),  INT64_C( 9134973770581442183),  INT64_C( 3293639426873489313), -INT64_C(  328403961242847600),
        -INT64_C( 2536763177469423006),  INT64_C( 7193767865162452761), -INT64_C( 4312809943194544036),  INT64_C( 1664511848765419208) },
      {  INT64_C( 8253825267704708698), -INT64_C( 1337675118988940471),  INT64_C( 2769726272071634039),  INT64_C( 7657910540467210105),
        -INT64_C( 5123226731786144741), -INT64_C( 4157330089492369200), -INT64_C( 5169320661579844418), -INT64_C(    7830840026888599) },
      { -INT64_C(  378611073117431591), -INT64_C(    4503875313893425),  INT64_C( 3455769305565102071), -INT64_C(  326687622853429255),
        -INT64_C(  220747304674658693), -INT64_C( 1242925535621480371),  INT64_C( 6387257131878268672), -INT64_C(  188937762035625221) } },
    { {  INT64_C( 8234931572901739809), -INT64_C(  279186625218278646), -INT64_C( 2500241661861719467), -INT64_C( 9023740987241450270),
        -INT64_C( 7040152522126640840), -INT64_C( 7593042301526991002), -INT64_C(  577736095091148374), -INT64_C( 4138945990197685588) },
      UINT8_C(188),
      {  INT64_C(  892538382080581794),  INT64_C( 6965068233905604326),  INT64_C( 1455626687322899983),  INT64_C( 8171896765027598377),
        -INT64_C( 8285830948907468792),  INT64_C( 4818346047104045015), -INT64_C( 1526160893970656006), -INT64_C( 2054205638248598724) },
      { -INT64_C( 7788452478502442058), -INT64_C( 6656056738546879190), -INT64_C( 6634310886840309181),  INT64_C( 4062858752841188516),
         INT64_C( 4712874171587377944), -INT64_C( 6107092855449224973),  INT64_C( 8409238446189551529), -INT64_C( 6423445272877899231) },
      {  INT64_C( 8234931572901739809), -INT64_C(  279186625218278646), -INT64_C( 5188289816638919089),  INT64_C( 8748924866724349101),
        -INT64_C( 3645665311568512232), -INT64_C( 1441363571129745417), -INT64_C(  577736095091148374), -INT64_C( 1729383495203589315) } },
    { { -INT64_C( 2240010928790016111),  INT64_C( 8173114989901285465), -INT64_C( 2765152205580199077),  INT64_C( 8920320150083507350),
        -INT64_C( 9181700375636130362),  INT64_C( 2940197229270471060),  INT64_C( 1754703470685081288), -INT64_C(  869749790542604919) },
      UINT8_C( 77),
      {  INT64_C( 5527029702464939794),  INT64_C( 3074906162057514806),  INT64_C( 6563086639133633686), -INT64_C( 8622850164651424000),
        -INT64_C(  888799306597349086), -INT64_C( 7174653097701555352), -INT64_C( 2337019806556415313), -INT64_C( 5277870680977237565) },
      {  INT64_C( 3077856035236702213), -INT64_C( 4673930400212546175),  INT64_C( 5311299249243507447), -INT64_C( 2456356426386006881),
         INT64_C( 4294276503906024620), -INT64_C( 2615776765867391262), -INT64_C( 3675669374373353682),  INT64_C( 8194273983504285698) },
      {  INT64_C( 7978124523121143575),  INT64_C( 8173114989901285465),  INT64_C( 6608406310821916407), -INT64_C( 2450661892322254945),
        -INT64_C( 9181700375636130362),  INT64_C( 2940197229270471060), -INT64_C( 2306547797413347409), -INT64_C(  869749790542604919) } },
    { { -INT64_C( 8153354067094055196), -INT64_C( 5963835184355632258),  INT64_C( 6184390424008977882), -INT64_C( 7055479551913110817),
        -INT64_C( 1551746889580415683),  INT64_C(  422643659118016689),  INT64_C( 8944113253756976758),  INT64_C( 8217438973637709489) },
      UINT8_C( 42),
      { -INT64_C( 3814687618846205113),  INT64_C( 6614939155643645901),  INT64_C(  259366773902054664),  INT64_C( 6265819966373102996),
        -INT64_C( 2692376262504903931), -INT64_C( 7099084145192560382),  INT64_C( 8955724412183999378),  INT64_C( 3497390542848164132) },
      {  INT64_C(  120940780742786246),  INT64_C( 7815604468067655235),  INT64_C( 4499654695441398541), -INT64_C( 5198566904008951938),
        -INT64_C( 3443221215239806409), -INT64_C(   45846396730321130), -INT64_C( 8470315371051347942),  INT64_C( 1640733990408479772) },
      { -INT64_C( 8153354067094055196),  INT64_C( 9223089324373245903),  INT64_C( 6184390424008977882), -INT64_C(  576747725182617602),
        -INT64_C( 1551746889580415683), -INT64_C(   36029939484926186),  INT64_C( 8944113253756976758),  INT64_C( 8217438973637709489) } },
    { { -INT64_C( 7112165093595656208),  INT64_C( 3299708147254359514),  INT64_C( 4580958117413699323), -INT64_C( 4274458292030421319),
         INT64_C( 6306883900102736481), -INT64_C(  377196261794442110), -INT64_C( 5544048811334031956), -INT64_C( 3462429812364387238) },
      UINT8_C(245),
      {  INT64_C( 5999298553777928242), -INT64_C( 4645940638102857747), -INT64_C( 4522339265942038961), -INT64_C( 8022883057768895560),
        -INT64_C( 3365998833911784777),  INT64_C( 4161663511315651688), -INT64_C( 6940510027864245624), -INT64_C( 7803617905910555194) },
      { -INT64_C(  231140893572660460),  INT64_C( 3572128188538451567), -INT64_C( 3405058824889979079),  INT64_C( 2241287428242608299),
        -INT64_C( 2001038091231910998),  INT64_C( 3473349496945850457), -INT64_C( 3793985678177512833), -INT64_C( 7041258221472690914) },
      { -INT64_C(   14677655597368522),  INT64_C( 3299708147254359514), -INT64_C( 3332684133942952065), -INT64_C( 4274458292030421319),
        -INT64_C(  757743874421572673),  INT64_C( 4175948456579153017), -INT64_C( 2306020138272457985), -INT64_C( 6918656043512570402) } },
    { { -INT64_C( 7248237650323415025), -INT64_C(  687413062199826866), -INT64_C( 1076683763796163549), -INT64_C( 2614230954937786330),
        -INT64_C(  531203023974656830),  INT64_C( 3664999973834800495),  INT64_C( 6501654372168326006),  INT64_C( 8686366944463705316) },
      UINT8_C(192),
      {  INT64_C( 6495705912994069740),  INT64_C( 4812266520139271071), -INT64_C( 6955579501930147568), -INT64_C( 6758832432267056826),
        -INT64_C( 6935405985480008375), -INT64_C( 3192566411781134200),  INT64_C( 4662597779429206218),  INT64_C( 2261341198418577673) },
      { -INT64_C( 4746154956857951869), -INT64_C( 7636733000022158482),  INT64_C( 5999180950094441223),  INT64_C( 6655885775698326632),
        -INT64_C( 7556474847964962003),  INT64_C( 3925111581293694997), -INT64_C( 1381064105003572169), -INT64_C( 8528823343692191891) },
      { -INT64_C( 7248237650323415025), -INT64_C(  687413062199826866), -INT64_C( 1076683763796163549), -INT64_C( 2614230954937786330),
        -INT64_C(  531203023974656830),  INT64_C( 3664999973834800495), -INT64_C( 1371909055454434049), -INT64_C( 6925430255641759891) } },
    { { -INT64_C( 7337472626472942740), -INT64_C(  592797568556272545), -INT64_C( 1979246864402678503), -INT64_C( 6405862239740644555),
         INT64_C( 8462659622098699857), -INT64_C(  354717972662260388),  INT64_C( 1017849633182354630),  INT64_C( 7761593429890386899) },
      UINT8_C(191),
      { -INT64_C( 6865655497726016781),  INT64_C( 8421026413544723534), -INT64_C( 8788110317754443242), -INT64_C( 4363412097973872133),
        -INT64_C( 2260289615407724723), -INT64_C( 5113883957611977557), -INT64_C( 6075080262781217553), -INT64_C( 6221774167585370092) },
      {  INT64_C( 5155296235211048217),  INT64_C( 5245252488694236014), -INT64_C( 3994373234145183179), -INT64_C( 2632543361018564494),
        -INT64_C( 2739295319043465191), -INT64_C( 1425753529935800683),  INT64_C( 4533828273520683374),  INT64_C(  356210891249457236) },
      { -INT64_C( 1748721841876383749),  INT64_C( 8998192025821763438), -INT64_C( 3559111721845016009), -INT64_C( 2632534495383979525),
        -INT64_C(  432952513073879203), -INT64_C(  200421487772240193),  INT64_C( 1017849633182354630), -INT64_C( 5911022906776758188) } },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].src));
    easysimd__m512d a = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].a));
    easysimd__m512d b = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].b));
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_or_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_or_pd");
    easysimd_test_x86_assert_equal_i64x8(easysimd_mm512_castpd_si512(r), easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_castpd_si512(easysimd_mm512_mask_or_pd(easysimd_mm512_castsi512_pd(src), k, easysimd_mm512_castsi512_pd(a), easysimd_mm512_castsi512_pd(b)));

    easysimd_test_x86_write_i64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_or_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C( 44),
      { -INT64_C( 8644503729592254503),  INT64_C( 4867591133601205217), -INT64_C( 7989567502446295554), -INT64_C( 5000044651753462680),
         INT64_C( 2861005423166470532), -INT64_C(   73075738687448992),  INT64_C( 7973624265941560168), -INT64_C( 7691732472859538267) },
      {  INT64_C( 4945363489985236531),  INT64_C( 4705128947430291021), -INT64_C( 3235554661140779867), -INT64_C( 6597039590797019188),
        -INT64_C(  769341857814662358), -INT64_C( 1274644824611205067),  INT64_C( 8339175193756411833), -INT64_C( 2755988914600667990) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 3233766295803580929), -INT64_C( 4684060331952113684),
         INT64_C(                   0), -INT64_C(   72075968984393611),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(245),
      { -INT64_C( 8443244095939418146), -INT64_C( 2874490419795467820), -INT64_C( 3406423654106398117),  INT64_C( 8429711099124766020),
         INT64_C(    1376779584510272), -INT64_C(  734230109709215546), -INT64_C( 8715664557394866305),  INT64_C( 2637853615469726354) },
      {  INT64_C( 6291427048961256739), -INT64_C( 5977932372106104453),  INT64_C( 6717105425418292845),  INT64_C( 8179979819204068734),
         INT64_C( 4807048100755547341), -INT64_C( 8026152153920371263), -INT64_C(  605320212824095120),  INT64_C(  936148657786920509) },
      { -INT64_C( 2314934063953088513),  INT64_C(                   0), -INT64_C( 2467985791030653313),  INT64_C(                   0),
         INT64_C( 4807298961743396301), -INT64_C(  729726368346546745), -INT64_C(  604611019233964161),  INT64_C( 3242556545990237887) } },
    { UINT8_C(  2),
      {  INT64_C( 2778451734942018494), -INT64_C( 6686479725356286110),  INT64_C( 2797038870634196763),  INT64_C( 1751461937719631213),
        -INT64_C( 8852821070547904548),  INT64_C( 7112728930704518645),  INT64_C( 5988881399278916333), -INT64_C( 3560116454162530012) },
      { -INT64_C( 1608647139254371068),  INT64_C( 6919694462725387869), -INT64_C( 1676338773382201981),  INT64_C( 5454982058908998387),
        -INT64_C( 3794530701388120453), -INT64_C( 8653536922151464172), -INT64_C( 6692440648396554637), -INT64_C(  101265556488288762) },
      {  INT64_C(                   0), -INT64_C( 2073912576659685505),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(186),
      {  INT64_C( 1674762660281347338), -INT64_C( 1419580910678063774),  INT64_C( 2973854198822438034),  INT64_C( 3872720930835632841),
         INT64_C( 4392307996128896536),  INT64_C( 5043013100133224545), -INT64_C( 6723090457070167579),  INT64_C( 2574762712150531931) },
      {  INT64_C( 1684897898266892876), -INT64_C( 4756887692717361659), -INT64_C( 6685319556694114116), -INT64_C( 8032199328507764904),
        -INT64_C( 9093867723175852033), -INT64_C( 3304739331239185522), -INT64_C( 4926480166036149579), -INT64_C( 5812837940910371969) },
      {  INT64_C(                   0), -INT64_C(  145060772383295641),  INT64_C(                   0), -INT64_C( 5350294570533003303),
        -INT64_C( 4756751193173202945), -INT64_C( 2882455810082999313),  INT64_C(                   0), -INT64_C( 5764634204314804353) } },
    { UINT8_C( 45),
      {  INT64_C( 2643378992013009930), -INT64_C( 6679105181315361436), -INT64_C( 5002160053507699339), -INT64_C( 5353625832551458848),
        -INT64_C( 1238060496664961411), -INT64_C( 8180552217161888612),  INT64_C( 2363887598484815397), -INT64_C( 3535507877539727218) },
      { -INT64_C( 1643728869767002440), -INT64_C( 5476317157149198247),  INT64_C( 7499503671639057332), -INT64_C( 3608067111326139698),
         INT64_C( 1120903410947680600),  INT64_C( 3907859499276389623),  INT64_C( 7522767963467416356), -INT64_C(  875539343756228105) },
      { -INT64_C( 1315209981829847366),  INT64_C(                   0), -INT64_C(  389605348370092043), -INT64_C(  144788365681950738),
         INT64_C(                   0), -INT64_C( 4720942720031830785),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(222),
      {  INT64_C( 8367304546264062440), -INT64_C( 2033043318195706844),  INT64_C( 4002826157048055639),  INT64_C( 6467667632824948845),
         INT64_C( 2232271359856105054),  INT64_C( 8360243439560028226), -INT64_C( 3046239401143598508), -INT64_C( 5767007406362424531) },
      {  INT64_C( 6334083171536435151), -INT64_C( 8308439586825459168),  INT64_C( 9024589475989076736),  INT64_C( 4491234994804225363),
        -INT64_C( 2741022317197424247), -INT64_C( 1270920071101386856), -INT64_C( 8768346627987051901),  INT64_C(  447121342482508680) },
      {  INT64_C(                   0), -INT64_C( 1154188143109362140),  INT64_C( 9204774343073070935),  INT64_C( 9211505557206990207),
        -INT64_C( 2305844136130388001),  INT64_C(                   0), -INT64_C( 2884107545450911017), -INT64_C( 5767000618703328339) } },
    { UINT8_C(  8),
      {  INT64_C(  703294478739416123),  INT64_C( 6412251682368295600), -INT64_C( 8334736702794117062), -INT64_C( 4845292104034202565),
        -INT64_C( 3650265228640943180),  INT64_C( 7890380429839453753),  INT64_C( 6322930852644391699), -INT64_C( 8279634984414623268) },
      { -INT64_C( 2359547944058945962), -INT64_C( 2512578911934183551), -INT64_C( 4105645774638851183),  INT64_C( 4611752996062369433),
        -INT64_C( 7944361769115001798), -INT64_C( 2511770214366591620), -INT64_C( 6956777166024788543),  INT64_C( 1139712889155917990) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(  233557250752553285),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(238),
      {  INT64_C( 8428080632709377925), -INT64_C( 7495224109626802733), -INT64_C( 5377512236491364312),  INT64_C( 4994379563326435151),
        -INT64_C( 5860389437443191671),  INT64_C( 3597781306574400499),  INT64_C( 1611243888556280459),  INT64_C(  666188052140910136) },
      { -INT64_C( 6368955511173218797), -INT64_C(  967453240201477033), -INT64_C(  192820507037762444), -INT64_C( 6672586903734793507),
        -INT64_C( 1545246413656316365), -INT64_C( 1581646973603509326),  INT64_C( 8798758913192666712),  INT64_C( 8752250300252229954) },
      {  INT64_C(                   0), -INT64_C(  577587307293590057), -INT64_C(  189151331508814724), -INT64_C( 1769987425368608801),
         INT64_C(                   0), -INT64_C(  293297486763987981),  INT64_C( 9106130973471190747),  INT64_C( 8754715955750762362) } },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].a));
    easysimd__m512d b = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].b));
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_or_pd(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_or_pd");
    easysimd_test_x86_assert_equal_i64x8(easysimd_mm512_castpd_si512(r), easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_castpd_si512(easysimd_mm512_maskz_or_pd(k, easysimd_mm512_castsi512_pd(a), easysimd_mm512_castsi512_pd(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(   768012835),  INT32_C(  1529621558), -INT32_C(  1724343561), -INT32_C(  1025486607),  INT32_C(  1861598911),  INT32_C(   336886298), -INT32_C(   635461259),  INT32_C(  1653142148),
         INT32_C(   892650239), -INT32_C(   443522578), -INT32_C(   310457861), -INT32_C(   659595751),  INT32_C(   927441949), -INT32_C(  1823778018), -INT32_C(  2140312580),  INT32_C(  1575155293) },
      { -INT32_C(  1500375112),  INT32_C(  1938498424),  INT32_C(    90180076), -INT32_C(  2049044632), -INT32_C(   759421517),  INT32_C(  2086996096), -INT32_C(   788737165), -INT32_C(  2127700023),
         INT32_C(  1848099062), -INT32_C(   807292189),  INT32_C(   601113275),  INT32_C(    94941777),  INT32_C(  1473734103), -INT32_C(   523027347), -INT32_C(   659501041), -INT32_C(  1520836945) },
      { -INT32_C(   155508739),  INT32_C(  2075078526), -INT32_C(  1653040129), -INT32_C(   939630599), -INT32_C(    16924737),  INT32_C(  2088073882), -INT32_C(   620760201), -INT32_C(   475136051),
         INT32_C(  2134360831), -INT32_C(   269356049), -INT32_C(   268514309), -INT32_C(   575668647),  INT32_C(  2010637791), -INT32_C(   203718785), -INT32_C(   654443521), -INT32_C(    33816833) } },
    { { -INT32_C(  2129428322), -INT32_C(   296684494), -INT32_C(  2012142538), -INT32_C(  1366443305), -INT32_C(  1945738209), -INT32_C(  1335043680),  INT32_C(  1485315241), -INT32_C(  1728191750),
        -INT32_C(  1793519518),  INT32_C(   998467845),  INT32_C(  1707316366),  INT32_C(  1829982286),  INT32_C(  1442388404), -INT32_C(  1677367821),  INT32_C(  2096467330), -INT32_C(   770313617) },
      {  INT32_C(   140979715),  INT32_C(   641985176), -INT32_C(   846526594),  INT32_C(   205168215), -INT32_C(  1419692872),  INT32_C(   457664153),  INT32_C(  1670921459),  INT32_C(   842378543),
         INT32_C(  1933221083),  INT32_C(    77168006), -INT32_C(   590273404),  INT32_C(  2078805187), -INT32_C(   651802304), -INT32_C(  1560973905), -INT32_C(   653947478),  INT32_C(   353057338) },
      { -INT32_C(  1988645217), -INT32_C(   296485190), -INT32_C(   845469826), -INT32_C(  1363165481), -INT32_C(  1352174401), -INT32_C(  1150287943),  INT32_C(  2073574651), -INT32_C(  1157763073),
        -INT32_C(   147088133),  INT32_C(  1067154823), -INT32_C(    36457330),  INT32_C(  2147179727), -INT32_C(   570467852), -INT32_C(  1091211777), -INT32_C(    34239062), -INT32_C(   685770113) } },
    { {  INT32_C(  1552500182),  INT32_C(  1197548226),  INT32_C(   170078791),  INT32_C(  2122648382),  INT32_C(    56142676), -INT32_C(  1029354216),  INT32_C(   345746394), -INT32_C(  1138120987),
        -INT32_C(  1374113045),  INT32_C(   485849557), -INT32_C(   366602068),  INT32_C(  2003413795),  INT32_C(  1853538646), -INT32_C(   399433970), -INT32_C(  1325609782),  INT32_C(  1584145779) },
      { -INT32_C(  1391688488), -INT32_C(  1429667330),  INT32_C(  1016393497), -INT32_C(   256639590), -INT32_C(   866243138),  INT32_C(   397709133), -INT32_C(   808996772), -INT32_C(  1372769322),
        -INT32_C(  1235535176),  INT32_C(  1432364091), -INT32_C(  1382943725), -INT32_C(  1331870222), -INT32_C(  1082262670), -INT32_C(   405327477), -INT32_C(  1195991326), -INT32_C(  1972968494) },
      { -INT32_C(    41040418), -INT32_C(   269933570),  INT32_C(  1052245855), -INT32_C(    21495874), -INT32_C(   815878146), -INT32_C(   675950755), -INT32_C(   538985506), -INT32_C(  1104300041),
        -INT32_C(  1101284613),  INT32_C(  1576369663), -INT32_C(   273154881), -INT32_C(   134352909), -INT32_C(     8389770), -INT32_C(   269009009), -INT32_C(  1191256342), -INT32_C(   563157005) } },
    { {  INT32_C(  1497415965), -INT32_C(   122773275), -INT32_C(  2035990636), -INT32_C(   147373436), -INT32_C(   927550403), -INT32_C(   944796187), -INT32_C(    25205460),  INT32_C(  1720247625),
        -INT32_C(  1950365530), -INT32_C(    58495640),  INT32_C(   830679213), -INT32_C(  1473725846),  INT32_C(  1399906158), -INT32_C(  1743052692), -INT32_C(   828990843),  INT32_C(   641015424) },
      {  INT32_C(  1320350950),  INT32_C(   256587106), -INT32_C(   935276962), -INT32_C(   160405112), -INT32_C(  1270226616), -INT32_C(  2041748479),  INT32_C(  2136335359), -INT32_C(   391804414),
        -INT32_C(   533244034), -INT32_C(   336625011), -INT32_C(   659345328), -INT32_C(   506583911),  INT32_C(   110434053),  INT32_C(  2072830588), -INT32_C(   923082298), -INT32_C(   357523605) },
      {  INT32_C(  1609758207), -INT32_C(     1067545), -INT32_C(   823787554), -INT32_C(   143168628), -INT32_C(    50334339), -INT32_C(   940577307), -INT32_C(     8394753), -INT32_C(   290590901),
        -INT32_C(   339746818), -INT32_C(     1052691), -INT32_C(   105694979), -INT32_C(   370229509),  INT32_C(  1475731311), -INT32_C(    73407876), -INT32_C(   822150201), -INT32_C(   290086933) } },
    { { -INT32_C(  2067077129), -INT32_C(  1183860119), -INT32_C(  2087640342),  INT32_C(  1264869190), -INT32_C(   212731529), -INT32_C(  1569792292),  INT32_C(   728393919), -INT32_C(    32171513),
         INT32_C(  1803738882), -INT32_C(  2094665319),  INT32_C(  1510389268), -INT32_C(  1918473706),  INT32_C(  1082193764), -INT32_C(  1780289835),  INT32_C(  1572883542),  INT32_C(  1750914406) },
      {  INT32_C(  1322573492), -INT32_C(   456001072), -INT32_C(   985737041), -INT32_C(  1504517054), -INT32_C(  1310272804),  INT32_C(   373737664),  INT32_C(  2071201300), -INT32_C(  1864118053),
         INT32_C(  2128525230),  INT32_C(  1617080240), -INT32_C(   903503481),  INT32_C(  1634760837),  INT32_C(   185816906),  INT32_C(   874600735),  INT32_C(  1001362784),  INT32_C(   348885605) },
      { -INT32_C(   824246281), -INT32_C(    33555463), -INT32_C(   943719697), -INT32_C(   277414074), -INT32_C(   201851905), -INT32_C(  1234247972),  INT32_C(  2071883455), -INT32_C(    17311009),
         INT32_C(  2145320878), -INT32_C(   479723591), -INT32_C(   634931305), -INT32_C(   302613865),  INT32_C(  1267988334), -INT32_C(  1243349025),  INT32_C(  2146426230),  INT32_C(  2095044455) } },
    { { -INT32_C(    91051702), -INT32_C(   530909863),  INT32_C(   464158870),  INT32_C(  1115495416), -INT32_C(  1857187726),  INT32_C(  1237676009),  INT32_C(  1787130884),  INT32_C(  1367232519),
         INT32_C(  1397428474), -INT32_C(  1691113979),  INT32_C(   515366438),  INT32_C(  1801467129), -INT32_C(  1375949116),  INT32_C(   569885213),  INT32_C(  1032551478), -INT32_C(   963769908) },
      {  INT32_C(   521787930), -INT32_C(  1497674368),  INT32_C(   616854059),  INT32_C(  1787765926), -INT32_C(   283669550), -INT32_C(  2079257011),  INT32_C(  1472306314), -INT32_C(  1071820634),
        -INT32_C(  1428211926), -INT32_C(  1353672060), -INT32_C(  1294789620),  INT32_C(   186409528),  INT32_C(  1006253293), -INT32_C(   843117758),  INT32_C(  1311015080), -INT32_C(    99728944) },
      { -INT32_C(     6554790), -INT32_C(   419693095),  INT32_C(  1072624319),  INT32_C(  1795112958), -INT32_C(    10511374), -INT32_C(   841650195),  INT32_C(  2143681678), -INT32_C(   780185433),
        -INT32_C(    69257222), -INT32_C(  1082933627), -INT32_C(  1091051986),  INT32_C(  1803319033), -INT32_C(  1073824019), -INT32_C(   302003361),  INT32_C(  2142239934), -INT32_C(    24229412) } },
    { { -INT32_C(    56300168), -INT32_C(  1800670072),  INT32_C(  1095204360), -INT32_C(   833854496), -INT32_C(   636926313), -INT32_C(    89667502),  INT32_C(   424201032), -INT32_C(  2062330356),
        -INT32_C(   880691133), -INT32_C(  1251988052), -INT32_C(  1929992277), -INT32_C(  1571143158), -INT32_C(   629382264),  INT32_C(  1943282475), -INT32_C(    91480850), -INT32_C(  1233149838) },
      {  INT32_C(    75563352), -INT32_C(   642129618), -INT32_C(  1822052472),  INT32_C(  2050342897),  INT32_C(  1297396002), -INT32_C(  1027528236), -INT32_C(  1195619003),  INT32_C(  1164852461),
         INT32_C(  1800073021),  INT32_C(  1497629649), -INT32_C(  1527993677), -INT32_C(  1944182423), -INT32_C(  1495698478), -INT32_C(   513238372), -INT32_C(   711383576), -INT32_C(  1642395807) },
      { -INT32_C(    56234632), -INT32_C(   574884434), -INT32_C(   748159096), -INT32_C(    25296911), -INT32_C(   547489865), -INT32_C(    85464618), -INT32_C(  1174614195), -INT32_C(   981500179),
        -INT32_C(   338952321), -INT32_C(    43765763), -INT32_C(  1392595013), -INT32_C(  1369545877), -INT32_C(    16944166), -INT32_C(   201540673), -INT32_C(     6472210), -INT32_C(  1098930317) } },
    { { -INT32_C(   938908169),  INT32_C(   455167336),  INT32_C(  1639976695),  INT32_C(    49143343), -INT32_C(   307706287),  INT32_C(  1238307169), -INT32_C(  1759614922),  INT32_C(  1731541360),
         INT32_C(   120536734), -INT32_C(  2094903157), -INT32_C(  1897602466),  INT32_C(   311480769), -INT32_C(   134203241),  INT32_C(  2135019337), -INT32_C(  1491705801),  INT32_C(   940460953) },
      {  INT32_C(   356466057), -INT32_C(   325557874),  INT32_C(    91913284), -INT32_C(   468252083), -INT32_C(  1931667645),  INT32_C(   504110822),  INT32_C(   415572606), -INT32_C(   145697682),
        -INT32_C(  1626566895),  INT32_C(   881632496),  INT32_C(  1832453664),  INT32_C(  1414680849),  INT32_C(  1340092264), -INT32_C(   865211315),  INT32_C(  2112107023),  INT32_C(   410268678) },
      { -INT32_C(   583041537), -INT32_C(     4624914),  INT32_C(  1710915319), -INT32_C(   419438993), -INT32_C(   302131373),  INT32_C(  1607409639), -INT32_C(  1612748162), -INT32_C(   143262338),
        -INT32_C(  1624260705), -INT32_C(  1213270789), -INT32_C(   268573058),  INT32_C(  1456656849), -INT32_C(     2080769), -INT32_C(     9572531), -INT32_C(      626113),  INT32_C(   947814303) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_or_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_or_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(    67917480),  INT32_C(  1903177832),  INT32_C(  1954583961),  INT32_C(  1772734799), -INT32_C(    84087974), -INT32_C(  1650377998),  INT32_C(  1615226614), -INT32_C(   331063660),
         INT32_C(  1877489415),  INT32_C(    14702695),  INT32_C(  1014259949), -INT32_C(  2069553622), -INT32_C(    75521783), -INT32_C(   761782053),  INT32_C(   439541638),  INT32_C(  1141274173) },
      UINT16_C(61101),
      { -INT32_C(  1807346509), -INT32_C(  1980484843),  INT32_C(   346496623),  INT32_C(   582332579),  INT32_C(  1145213356), -INT32_C(  1792817053), -INT32_C(   401841950),  INT32_C(  1490532772),
        -INT32_C(   471065138), -INT32_C(  1066606511), -INT32_C(  1563093761),  INT32_C(  1891928772),  INT32_C(  2125792795), -INT32_C(  1340811058), -INT32_C(   577232839), -INT32_C(  1489670183) },
      { -INT32_C(   561372787),  INT32_C(    27195138), -INT32_C(   811371765),  INT32_C(   423651326),  INT32_C(   999880045),  INT32_C(   116108493), -INT32_C(  1511750708), -INT32_C(  2142430733),
         INT32_C(  1029625659), -INT32_C(   650183474),  INT32_C(  1856627056), -INT32_C(  1249384120), -INT32_C(  1410326562), -INT32_C(  1749886005),  INT32_C(  1362990686), -INT32_C(   355366225) },
      { -INT32_C(   556911169),  INT32_C(  1903177832), -INT32_C(   542670993),  INT32_C(  1005975551), -INT32_C(    84087974), -INT32_C(  1746146065),  INT32_C(  1615226614), -INT32_C(   656426505),
         INT32_C(  1877489415), -INT32_C(   645989153), -INT32_C(   285346305), -INT32_C(   171447348), -INT32_C(    75521783), -INT32_C(  1212752945), -INT32_C(   574769537), -INT32_C(   269094913) } },
    { {  INT32_C(   807874401), -INT32_C(  1677105876), -INT32_C(  1911901626),  INT32_C(  2034471323),  INT32_C(  2082812849),  INT32_C(  1830082318),  INT32_C(   482234733),  INT32_C(  1007128795),
        -INT32_C(   345231681), -INT32_C(   628656492), -INT32_C(   999714520), -INT32_C(   734155741), -INT32_C(   296721696), -INT32_C(  1503960007), -INT32_C(  1866261835),  INT32_C(  1775094442) },
      UINT16_C(14840),
      { -INT32_C(   592475051), -INT32_C(   798042009), -INT32_C(   646147684), -INT32_C(  1254335387), -INT32_C(  1508215477), -INT32_C(   541012196),  INT32_C(   766077536),  INT32_C(   694591956),
        -INT32_C(  1794828754), -INT32_C(  1973062674),  INT32_C(  1768153348), -INT32_C(  1994416322),  INT32_C(   825178389),  INT32_C(  1745940488),  INT32_C(   781564250), -INT32_C(  2007499686) },
      {  INT32_C(     1924370), -INT32_C(   712342832), -INT32_C(  1572934300), -INT32_C(  1574216307), -INT32_C(  1613538666), -INT32_C(  1526209718), -INT32_C(   153903716), -INT32_C(  1417729127),
         INT32_C(  1487641736), -INT32_C(  2094189281), -INT32_C(  1356501982),  INT32_C(  1615941833), -INT32_C(   167828310), -INT32_C(  1550187001),  INT32_C(  1016753571),  INT32_C(   552016280) },
      {  INT32_C(   807874401), -INT32_C(  1677105876), -INT32_C(  1911901626), -INT32_C(  1220583955), -INT32_C(  1076133921), -INT32_C(     3674274), -INT32_C(      262148), -INT32_C(  1417696291),
        -INT32_C(   575693138), -INT32_C(   628656492), -INT32_C(   999714520), -INT32_C(   379592705), -INT32_C(   167821889), -INT32_C(   342166001), -INT32_C(  1866261835),  INT32_C(  1775094442) } },
    { { -INT32_C(   730230091), -INT32_C(   363354425), -INT32_C(   593921006),  INT32_C(  2000481228),  INT32_C(   392968975), -INT32_C(   440793535),  INT32_C(   203510899),  INT32_C(   590088557),
         INT32_C(  1677174171),  INT32_C(  1582124876), -INT32_C(  1740970293), -INT32_C(   519080239), -INT32_C(   201819215), -INT32_C(   187125119),  INT32_C(  1946220807), -INT32_C(  1634259710) },
      UINT16_C(36818),
      {  INT32_C(  1323179521), -INT32_C(  1221285507),  INT32_C(  1345193537), -INT32_C(   540287001), -INT32_C(  1433318190),  INT32_C(  1101305921), -INT32_C(  1536252403),  INT32_C(  1160986948),
        -INT32_C(   594341537), -INT32_C(    57358149), -INT32_C(  1253195058),  INT32_C(  1955862689), -INT32_C(  1507973787), -INT32_C(   874003778), -INT32_C(  1385212311), -INT32_C(   151805033) },
      {  INT32_C(  1876133556),  INT32_C(   476866382), -INT32_C(   892225239),  INT32_C(   910058961),  INT32_C(  1239178635), -INT32_C(  2011904993), -INT32_C(  1321827558), -INT32_C(   609801946),
        -INT32_C(    28673361),  INT32_C(   169522913),  INT32_C(  1104538735), -INT32_C(   596176047), -INT32_C(  1876601744),  INT32_C(   840448280), -INT32_C(   471642435),  INT32_C(   666798711) },
      { -INT32_C(   730230091), -INT32_C(  1082331265), -INT32_C(   593921006),  INT32_C(  2000481228), -INT32_C(   337814053), -INT32_C(   440793535), -INT32_C(  1249990881), -INT32_C(   541628058),
        -INT32_C(    19170305), -INT32_C(    23134469), -INT32_C(   170004753), -INT32_C(    50914319), -INT32_C(   201819215), -INT32_C(   187125119),  INT32_C(  1946220807), -INT32_C(   134239241) } },
    { { -INT32_C(   433780475),  INT32_C(   804339647),  INT32_C(  2104542763),  INT32_C(  1247406041),  INT32_C(  1406828347),  INT32_C(  1971712696), -INT32_C(  1202100159), -INT32_C(   119597069),
        -INT32_C(   522320864),  INT32_C(  1863307076),  INT32_C(  1877770133), -INT32_C(  1564916122),  INT32_C(  2113246149), -INT32_C(   957121659), -INT32_C(   696300317), -INT32_C(  2066850205) },
      UINT16_C(44387),
      {  INT32_C(  1937549156),  INT32_C(    66195990),  INT32_C(   977885313), -INT32_C(   254996742),  INT32_C(  2120962699), -INT32_C(  1731572199), -INT32_C(   201970139),  INT32_C(   379607474),
         INT32_C(   394861824), -INT32_C(  1340441809), -INT32_C(   823499821), -INT32_C(    54610063),  INT32_C(   578496777), -INT32_C(  1665514121),  INT32_C(   630239603),  INT32_C(   188428298) },
      {  INT32_C(  2082653261),  INT32_C(   321666111),  INT32_C(   283186847), -INT32_C(   687038515),  INT32_C(  1073317832),  INT32_C(  1088206029),  INT32_C(  1868983397), -INT32_C(   361061988),
        -INT32_C(  1520001946),  INT32_C(  2008585176),  INT32_C(  2005375401),  INT32_C(     5149752), -INT32_C(   398440677),  INT32_C(  1613307131),  INT32_C(   617648008), -INT32_C(  1777448400) },
      {  INT32_C(  2139023213),  INT32_C(   335429183),  INT32_C(  2104542763),  INT32_C(  1247406041),  INT32_C(  1406828347), -INT32_C(   656474403), -INT32_C(      627099), -INT32_C(   119597069),
        -INT32_C(  1209033370),  INT32_C(  1863307076), -INT32_C(     1049605), -INT32_C(    50415751),  INT32_C(  2113246149), -INT32_C(    54829569), -INT32_C(   696300317), -INT32_C(  1623229894) } },
    { { -INT32_C(  1086556697), -INT32_C(  1321733112), -INT32_C(   970408306),  INT32_C(  1858565714), -INT32_C(  1185544258), -INT32_C(  1424326877),  INT32_C(  1070590478),  INT32_C(   467000884),
         INT32_C(  1541083475), -INT32_C(  1811148282),  INT32_C(   593114576),  INT32_C(  1771119019),  INT32_C(  1260644136),  INT32_C(  1979071847),  INT32_C(  1538573863), -INT32_C(   143160668) },
      UINT16_C(21147),
      {  INT32_C(  1583653202), -INT32_C(  1869401035), -INT32_C(   391037097), -INT32_C(   875504984),  INT32_C(   470300453),  INT32_C(  1642213292),  INT32_C(    48989835), -INT32_C(   816544131),
         INT32_C(  1563277352),  INT32_C(  1156432364), -INT32_C(  1473470720), -INT32_C(  1636500359), -INT32_C(   524649421),  INT32_C(   927046828), -INT32_C(  1623643102), -INT32_C(   630288974) },
      {  INT32_C(   825793605),  INT32_C(  1567958365),  INT32_C(  1023779524), -INT32_C(   757302882), -INT32_C(  1565354250),  INT32_C(  1440346931), -INT32_C(   772468193), -INT32_C(   458464353),
         INT32_C(  1544938751), -INT32_C(   826701047), -INT32_C(   888422611),  INT32_C(   782100280), -INT32_C(  1311748226),  INT32_C(  1627826498),  INT32_C(  1513356219),  INT32_C(  1581178719) },
      {  INT32_C(  2138881367), -INT32_C(   571001475), -INT32_C(   970408306), -INT32_C(   606274626), -INT32_C(  1095059465), -INT32_C(  1424326877),  INT32_C(  1070590478), -INT32_C(   268638209),
         INT32_C(  1541083475), -INT32_C(   822227987),  INT32_C(   593114576),  INT32_C(  1771119019), -INT32_C(   235241601),  INT32_C(  1979071847), -INT32_C(   549716037), -INT32_C(   143160668) } },
    { { -INT32_C(   860138301),  INT32_C(   211449055),  INT32_C(  1826072115),  INT32_C(   194672013),  INT32_C(   129788868), -INT32_C(   798440684),  INT32_C(   489331646),  INT32_C(  1031563642),
        -INT32_C(  1677051971), -INT32_C(   542595925), -INT32_C(   666140854), -INT32_C(  1176246796),  INT32_C(  1707122768),  INT32_C(   557131875),  INT32_C(  1044340676), -INT32_C(  2055423032) },
      UINT16_C(34546),
      { -INT32_C(   919954143), -INT32_C(   951487108),  INT32_C(   816659789), -INT32_C(  1227817482), -INT32_C(  1746979998), -INT32_C(  1795710123),  INT32_C(  1565507553),  INT32_C(  1726169413),
         INT32_C(  1529876190), -INT32_C(   803047037), -INT32_C(  1392455754),  INT32_C(   795070925),  INT32_C(  1506230788), -INT32_C(   940720411), -INT32_C(  1037812611),  INT32_C(  1546193021) },
      { -INT32_C(  1716037354), -INT32_C(  2006328878),  INT32_C(  1983211945),  INT32_C(  1067817274), -INT32_C(  1046975269), -INT32_C(  1467447766),  INT32_C(  1080732866), -INT32_C(   895708236),
        -INT32_C(  1100786708), -INT32_C(   716714964),  INT32_C(  1883995190), -INT32_C(   240127723), -INT32_C(  2001581987), -INT32_C(  1875887410), -INT32_C(  1680827674),  INT32_C(   442854446) },
      { -INT32_C(   860138301), -INT32_C(   815137282),  INT32_C(  1826072115),  INT32_C(   194672013), -INT32_C(   673218565), -INT32_C(  1124096129),  INT32_C(  1567604707), -INT32_C(   285223947),
        -INT32_C(  1677051971), -INT32_C(   714605137), -INT32_C(    45350986), -INT32_C(  1176246796),  INT32_C(  1707122768),  INT32_C(   557131875),  INT32_C(  1044340676),  INT32_C(  1584229503) } },
    { { -INT32_C(   321271361), -INT32_C(   876535659), -INT32_C(  1304687204),  INT32_C(  1537469438),  INT32_C(    31675699),  INT32_C(  1972507535), -INT32_C(   602905938), -INT32_C(  1896450353),
        -INT32_C(   730149057), -INT32_C(  1935655697),  INT32_C(  1195301961), -INT32_C(    73211449), -INT32_C(   973306314),  INT32_C(  1195019929), -INT32_C(  1071428623),  INT32_C(     5118657) },
      UINT16_C(51689),
      {  INT32_C(  1946540500), -INT32_C(  1555018139),  INT32_C(   931403925),  INT32_C(   264092179), -INT32_C(  1180805249), -INT32_C(  1039888482), -INT32_C(  1646475953), -INT32_C(  1704540731),
         INT32_C(    51276702),  INT32_C(  1319526329), -INT32_C(  1970918793),  INT32_C(  1670988772),  INT32_C(   958216090),  INT32_C(   368779718), -INT32_C(  1397499929),  INT32_C(   994449820) },
      {  INT32_C(  1044272517),  INT32_C(   730719668), -INT32_C(   189459697), -INT32_C(   245936554),  INT32_C(  1277850758),  INT32_C(  2103584150), -INT32_C(  1742137860), -INT32_C(  1277988818),
         INT32_C(  2045907653),  INT32_C(   111443959), -INT32_C(   386246254),  INT32_C(   769217191),  INT32_C(  1551500230),  INT32_C(   618257448),  INT32_C(   532481009),  INT32_C(   953389171) },
      {  INT32_C(  2118114773), -INT32_C(   876535659), -INT32_C(  1304687204), -INT32_C(         425),  INT32_C(    31675699), -INT32_C(    10047586), -INT32_C(  1644308993), -INT32_C(  1141377553),
         INT32_C(  2080275423), -INT32_C(  1935655697),  INT32_C(  1195301961),  INT32_C(  1876513767), -INT32_C(   973306314),  INT32_C(  1195019929), -INT32_C(  1077946377),  INT32_C(  1003985407) } },
    { { -INT32_C(  1716402782), -INT32_C(   694135484),  INT32_C(  1438554798), -INT32_C(  1283221268), -INT32_C(  1005584997), -INT32_C(   890705447),  INT32_C(  1609147884), -INT32_C(   661144522),
        -INT32_C(   982366079),  INT32_C(  1268454045), -INT32_C(  1717544276), -INT32_C(  1924389902), -INT32_C(   112108768),  INT32_C(   818100804),  INT32_C(   361737695), -INT32_C(   336714135) },
      UINT16_C(24686),
      {  INT32_C(  1282542512), -INT32_C(   140108202),  INT32_C(    52074679),  INT32_C(  2002729765), -INT32_C(   122576076),  INT32_C(  1671794900), -INT32_C(  1802891610), -INT32_C(  1426786055),
         INT32_C(  1526097412),  INT32_C(  1011981444),  INT32_C(  1497328692), -INT32_C(   607084889),  INT32_C(   382959938), -INT32_C(  1199998958), -INT32_C(  2142502009),  INT32_C(     2769148) },
      {  INT32_C(   710549670), -INT32_C(   261706564), -INT32_C(  1085626856), -INT32_C(  2036721084), -INT32_C(  1365479780),  INT32_C(  1835407078),  INT32_C(   367964697), -INT32_C(  1743447822),
        -INT32_C(   171806663),  INT32_C(   887499036),  INT32_C(   334704847), -INT32_C(   426078902), -INT32_C(   493603077),  INT32_C(  1716517452), -INT32_C(  1636090452), -INT32_C(  1892249258) },
      { -INT32_C(  1716402782), -INT32_C(   135872770), -INT32_C(  1084571969), -INT32_C(   136365211), -INT32_C(  1005584997),  INT32_C(  1877448438), -INT32_C(  1779515713), -INT32_C(   661144522),
        -INT32_C(   982366079),  INT32_C(  1268454045), -INT32_C(  1717544276), -INT32_C(  1924389902), -INT32_C(   112108768), -INT32_C(    25167266), -INT32_C(  1635827793), -INT32_C(   336714135) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_or_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_or_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_mask_or_epi32(src, k, a, b);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_or_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(10069),
      { -INT32_C(   754578698), -INT32_C(  1968268126), -INT32_C(   940325655), -INT32_C(  1979375336), -INT32_C(  1770342166), -INT32_C(  2085670185), -INT32_C(   977031579),  INT32_C(   653006895),
        -INT32_C(   973475038),  INT32_C(  2035263631), -INT32_C(  2076163220),  INT32_C(  1309558116), -INT32_C(   723220227),  INT32_C(   542610363),  INT32_C(  1910905410),  INT32_C(  1436013106) },
      {  INT32_C(  1427804613), -INT32_C(  1513199303),  INT32_C(   271126188),  INT32_C(  1348417619),  INT32_C(  2065974208),  INT32_C(   429685975), -INT32_C(   913669482),  INT32_C(   438182484),
        -INT32_C(   328255309),  INT32_C(  1318141345), -INT32_C(  1637959093), -INT32_C(  1276199438), -INT32_C(   668069375),  INT32_C(   636603022), -INT32_C(  1578206388),  INT32_C(  1354435741) },
      { -INT32_C(   685859849),  INT32_C(           0), -INT32_C(   671363347),  INT32_C(           0), -INT32_C(     8455190),  INT32_C(           0), -INT32_C(   842289417),  INT32_C(           0),
        -INT32_C(   301990989),  INT32_C(  2145369519), -INT32_C(  1637942417),  INT32_C(           0),  INT32_C(           0),  INT32_C(   637000639),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(10820),
      { -INT32_C(   848829124), -INT32_C(  1836535245),  INT32_C(  1045396048),  INT32_C(  1548767277),  INT32_C(   421977896), -INT32_C(   225086716), -INT32_C(   805424620), -INT32_C(  1074183549),
         INT32_C(  1535991847),  INT32_C(  1659704594), -INT32_C(  1113572209), -INT32_C(  1256591219), -INT32_C(   741392433), -INT32_C(   960142158),  INT32_C(   429245334),  INT32_C(   769232389) },
      {  INT32_C(     8939246),  INT32_C(   174224763),  INT32_C(  1053229745), -INT32_C(  1024139021), -INT32_C(   761871584), -INT32_C(  1114023129),  INT32_C(   618081823), -INT32_C(  1420709699),
        -INT32_C(  1850943210),  INT32_C(    10161742),  INT32_C(    71197457),  INT32_C(  1673933379),  INT32_C(   473324789), -INT32_C(   673525832), -INT32_C(  1157910019),  INT32_C(  2003193185) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1053784817),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   268485089),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  1660886878),  INT32_C(           0), -INT32_C(   136318257),  INT32_C(           0), -INT32_C(   673189958),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C( 4646),
      { -INT32_C(  1558153976), -INT32_C(  1291439755), -INT32_C(    52016587), -INT32_C(   480715859), -INT32_C(   760082184), -INT32_C(   477908761), -INT32_C(   802102166),  INT32_C(  1675777627),
         INT32_C(  1074136011),  INT32_C(  1794379060),  INT32_C(    73849431), -INT32_C(  1377255755), -INT32_C(  1233151281), -INT32_C(  1248263606),  INT32_C(  1099287014), -INT32_C(   341546976) },
      { -INT32_C(  1624462485),  INT32_C(   252256440), -INT32_C(  1357680646), -INT32_C(    44172498), -INT32_C(   525083498), -INT32_C(   980070945),  INT32_C(   906435350), -INT32_C(   299717757),
         INT32_C(   244141654),  INT32_C(  1746769774),  INT32_C(   873935110), -INT32_C(  1036946388),  INT32_C(   816047441),  INT32_C(  1224030258), -INT32_C(   696255405), -INT32_C(    20667992) },
      {  INT32_C(           0), -INT32_C(  1089523203), -INT32_C(      561153),  INT32_C(           0),  INT32_C(           0), -INT32_C(   409469441),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  1795006334),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1224736801),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(21231),
      {  INT32_C(   719936781), -INT32_C(   597954619),  INT32_C(  1431406628), -INT32_C(   314924470), -INT32_C(   953783086),  INT32_C(   885291445),  INT32_C(   349531216),  INT32_C(  2003223658),
        -INT32_C(   425570527),  INT32_C(  1673723199), -INT32_C(   793242747), -INT32_C(  2017594443),  INT32_C(   340779871), -INT32_C(  1387785379), -INT32_C(   373219969),  INT32_C(    39921889) },
      { -INT32_C(  1226308745), -INT32_C(  2045138176),  INT32_C(  1951847102),  INT32_C(   620434373),  INT32_C(  1413040887), -INT32_C(   603881123), -INT32_C(  2134457697),  INT32_C(  1652762602),
         INT32_C(   706243370), -INT32_C(   709873130), -INT32_C(   917961212),  INT32_C(   300762138), -INT32_C(   328915313),  INT32_C(  1204381352),  INT32_C(   331845416), -INT32_C(   529184074) },
      { -INT32_C(  1092001921), -INT32_C(   564400187),  INT32_C(  1968691902), -INT32_C(   302271537),  INT32_C(           0), -INT32_C(    54134275), -INT32_C(  1797787937),  INT32_C(  2011686890),
         INT32_C(           0), -INT32_C(   135069889),  INT32_C(           0),  INT32_C(           0), -INT32_C(    59775009),  INT32_C(           0), -INT32_C(    70803585),  INT32_C(           0) } },
    { UINT16_C(36278),
      { -INT32_C(  1145058294), -INT32_C(   356400223),  INT32_C(  2049956748),  INT32_C(  1369489132),  INT32_C(  1941391530),  INT32_C(  1459806351),  INT32_C(  1755429107), -INT32_C(  1544202344),
        -INT32_C(   983648988),  INT32_C(    95428472), -INT32_C(   411049989),  INT32_C(  1194925981), -INT32_C(   138678168), -INT32_C(  1018249776),  INT32_C(   237760630),  INT32_C(  1806770503) },
      {  INT32_C(  1311838166),  INT32_C(   710140207),  INT32_C(  1561449152),  INT32_C(  1520716530),  INT32_C(   189882171), -INT32_C(  1815109603), -INT32_C(   677250416), -INT32_C(   247246053),
        -INT32_C(  1858046878),  INT32_C(   364614485),  INT32_C(  1483918694),  INT32_C(  1387402775), -INT32_C(  1822555274),  INT32_C(   858140067),  INT32_C(  1124845351),  INT32_C(  2100579866) },
      {  INT32_C(           0), -INT32_C(   355212369),  INT32_C(  2134891468),  INT32_C(           0),  INT32_C(  2079809467), -INT32_C(   674242401),  INT32_C(           0), -INT32_C(   201893989),
        -INT32_C(   715197082),  INT32_C(           0), -INT32_C(     8396801),  INT32_C(  1471881119),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2142596959) } },
    { UINT16_C(29890),
      { -INT32_C(   905439474), -INT32_C(  1634243028),  INT32_C(  2025107142),  INT32_C(  1601907201), -INT32_C(   477356099),  INT32_C(  1571533650),  INT32_C(   732677622),  INT32_C(  1369402690),
        -INT32_C(  1340364924), -INT32_C(   615599595),  INT32_C(  1632830304), -INT32_C(   339685587),  INT32_C(  1087261934), -INT32_C(   157386496), -INT32_C(  2128524993),  INT32_C(  1003667895) },
      {  INT32_C(  2129456488),  INT32_C(     5847711),  INT32_C(  1801563453),  INT32_C(  1784029820),  INT32_C(  1856644206), -INT32_C(   580564834),  INT32_C(  1230931602), -INT32_C(  1333513913),
        -INT32_C(  1104252898), -INT32_C(   390166613), -INT32_C(  1336729804), -INT32_C(  1340430015),  INT32_C(  1813956046), -INT32_C(  1622572275),  INT32_C(  1374201610), -INT32_C(   150901288) },
      {  INT32_C(           0), -INT32_C(  1629520193),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1811924982), -INT32_C(   241205945),
         INT32_C(           0),  INT32_C(           0), -INT32_C(   246210700),  INT32_C(           0),  INT32_C(  1826541038), -INT32_C(     2098419), -INT32_C(   773198017),  INT32_C(           0) } },
    { UINT16_C(12253),
      {  INT32_C(  1941407925), -INT32_C(   997004431), -INT32_C(  1234250596), -INT32_C(  1568981884), -INT32_C(   249132632),  INT32_C(   295186472), -INT32_C(  2088865407),  INT32_C(   498228072),
         INT32_C(  1435527652), -INT32_C(   250011051),  INT32_C(  2057799670),  INT32_C(  1797071811), -INT32_C(   732150868), -INT32_C(   186256013), -INT32_C(   831036570), -INT32_C(  1544869185) },
      { -INT32_C(   386368621), -INT32_C(  1797713507),  INT32_C(  1544454553),  INT32_C(  1355295652), -INT32_C(   517725074),  INT32_C(  2144733721),  INT32_C(   743263597),  INT32_C(   198129783),
         INT32_C(  1374930867),  INT32_C(  1927663065), -INT32_C(   221318322), -INT32_C(  1925016033), -INT32_C(   730896709), -INT32_C(   564902544),  INT32_C(   168468882), -INT32_C(  1927947558) },
      { -INT32_C(    67109961),  INT32_C(           0), -INT32_C(    26290787), -INT32_C(   218415196), -INT32_C(   249123346),  INT32_C(           0), -INT32_C(  1350599187),  INT32_C(   536836991),
         INT32_C(  1442050039), -INT32_C(   201461795), -INT32_C(    84936706), -INT32_C(   278939681),  INT32_C(           0), -INT32_C(    17435277),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C( 2209),
      { -INT32_C(  1009419554), -INT32_C(  1145625620),  INT32_C(  1464981013), -INT32_C(   759296925),  INT32_C(   890711777),  INT32_C(   399944204), -INT32_C(   906907468),  INT32_C(   483496510),
        -INT32_C(   102717683), -INT32_C(   525035574), -INT32_C(   801700243), -INT32_C(   207358702),  INT32_C(   807975460),  INT32_C(   407371620), -INT32_C(   304006993), -INT32_C(   670387253) },
      {  INT32_C(   617736794), -INT32_C(   301693311), -INT32_C(  1631699828),  INT32_C(  1435656497), -INT32_C(  2138654180),  INT32_C(  1771687353), -INT32_C(   782861818), -INT32_C(  2018942931),
        -INT32_C(   877888950), -INT32_C(  1933987840),  INT32_C(   506165484), -INT32_C(   176964135),  INT32_C(   829880695), -INT32_C(   845541433), -INT32_C(  1214320503), -INT32_C(  1690417327) },
      { -INT32_C(   405407010),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2145382333),  INT32_C(           0), -INT32_C(  1611009473),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   135004709),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_or_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_or_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 6429398815427941624), -INT64_C(  668819823809793933), -INT64_C( 1864586420094752608), -INT64_C( 6964511610407502025),
         INT64_C( 1602969904378442689), -INT64_C( 3335834506442223148), -INT64_C( 3507572279834119964), -INT64_C( 4859897487314131624) },
      {  INT64_C( 7302747944067050655), -INT64_C(   79420915697099292),  INT64_C( 5850102220228026311), -INT64_C( 3577600625560452446),
         INT64_C( 2553937083690429820),  INT64_C( 3737175614931113157), -INT64_C( 6501014216955967619),  INT64_C( 6107615400781023429) },
      { -INT64_C( 1738741473842712673), -INT64_C(   74344604006875657), -INT64_C(  630577615488516121), -INT64_C( 2352603316211171401),
         INT64_C( 3999175301830475773), -INT64_C(  865327213124397611), -INT64_C( 1164220517755127811), -INT64_C(  230035441899541027) } },
    { {  INT64_C(  120420968188400456),  INT64_C( 9080872039193043814),  INT64_C( 2282686581501630468), -INT64_C( 3174801842188849212),
        -INT64_C( 8510388074405211991), -INT64_C( 8761365484661914134), -INT64_C( 2489862664037179912),  INT64_C(  326908290568652074) },
      { -INT64_C( 2997762464526244679),  INT64_C( 1773621245074587796), -INT64_C( 8034742351829521528),  INT64_C( 1928331604867348322),
         INT64_C(  969781872701123507), -INT64_C( 1561692022419212773), -INT64_C(  224605373488972308), -INT64_C( 6089856293668313087) },
      { -INT64_C( 2886843662796720135),  INT64_C( 9123658572182447094), -INT64_C( 6917529583168341108), -INT64_C( 2597767831452385306),
        -INT64_C( 8217556958887019589), -INT64_C( 1262166901253998597), -INT64_C(  147990299458667012), -INT64_C( 5765315608657154773) } },
    { { -INT64_C( 1680476196060892634),  INT64_C( 9097683093164380661), -INT64_C( 2825529556040889293),  INT64_C( 2701798032050652123),
         INT64_C( 8997473558825650216), -INT64_C( 6245062807636475836), -INT64_C( 5436054925677746719), -INT64_C( 6348099103472760381) },
      {  INT64_C( 7120324182048287949), -INT64_C( 7378709589531188103),  INT64_C( 1459412915824169139), -INT64_C( 7362895231201130582),
         INT64_C( 6983602974306954601),  INT64_C( 7256400371483943777), -INT64_C( 8116277192105655047),  INT64_C( 6439173234401752055) },
      { -INT64_C( 1513772562375573777), -INT64_C(   10697444992223747), -INT64_C( 2537228811144930125), -INT64_C( 4755871579609577477),
         INT64_C( 9007199151526374761), -INT64_C( 1299290726435970203), -INT64_C( 4620850456435415559), -INT64_C(     125345336394761) } },
    { {  INT64_C( 7511664463165082103),  INT64_C( 8071354844583202542),  INT64_C( 2066614722928243770), -INT64_C( 1759433076923996341),
         INT64_C( 3455672968657860229), -INT64_C( 1694811768610719854), -INT64_C( 7564158082662058811), -INT64_C( 6344051108417968867) },
      {  INT64_C( 5858383487294487261), -INT64_C( 5605338946010932320),  INT64_C( 1219810240101104906),  INT64_C(  310768335838887816),
         INT64_C( 5538425528129914061),  INT64_C( 1669655743652575116), -INT64_C( 1004484776281913332),  INT64_C(  884372658259242265) },
      {  INT64_C( 8754975409943346175), -INT64_C(  993052514467471378),  INT64_C( 2085087425586986298), -INT64_C( 1741416447178965045),
         INT64_C( 8069734670988211917), -INT64_C(   37190235702003810), -INT64_C(  644016469005304627), -INT64_C( 5767440692609618659) } },
    { {  INT64_C( 7235442747905697904), -INT64_C( 2319343003134786644), -INT64_C( 5751397340793586223),  INT64_C( 2839298246934907419),
         INT64_C( 2367676744825372574), -INT64_C( 7898282672855074639),  INT64_C( 5489320810328769254), -INT64_C(  962909039784157498) },
      {  INT64_C( 5089705937810876174),  INT64_C(  743143574359145468), -INT64_C( 2238934313749504400), -INT64_C( 5613173280512043565),
        -INT64_C( 4268330770017975320),  INT64_C( 3063242394876153171), -INT64_C( 7859703023775577615),  INT64_C( 3577217418109012302) },
      {  INT64_C( 7416149706616205182), -INT64_C( 2319303265996538884), -INT64_C( 1085367923656146959), -INT64_C( 5224396885334302757),
        -INT64_C( 1955688174701903874), -INT64_C( 4979859703327900173), -INT64_C( 2382969628884251657), -INT64_C(  889496806583476274) } },
    { {  INT64_C(  352250201419733336),  INT64_C( 3018534595340232190), -INT64_C( 5307738001475588345),  INT64_C( 4531831678781886825),
         INT64_C(  277887291062326054),  INT64_C( 6559386569151370918), -INT64_C( 4148808817127179186), -INT64_C( 8670171034554919799) },
      { -INT64_C(  938937519609869328), -INT64_C( 6755724536193108849),  INT64_C( 6933227680470425364), -INT64_C(   89653021795897127),
        -INT64_C( 1673109969782497400), -INT64_C( 8821414289885914977), -INT64_C( 8385652262546510487),  INT64_C( 8568247751283208413) },
      { -INT64_C(  649791589396325384), -INT64_C( 6052840665212258817), -INT64_C(  686827530652680425), -INT64_C(   79518949804822023),
        -INT64_C( 1450181770996637778), -INT64_C( 2335222556925108545), -INT64_C( 3464255479336206993), -INT64_C(  581562555244251939) } },
    { {  INT64_C( 7342858363346869159),  INT64_C( 2783203391673250730), -INT64_C( 6983105264582486567),  INT64_C(  854402329842736503),
        -INT64_C( 2127121551310772658), -INT64_C( 7455536841593335131),  INT64_C( 7823943280686788146), -INT64_C( 2411769338864263657) },
      {  INT64_C( 3093530412248699603), -INT64_C( 5074125731344291009),  INT64_C(  821136122698447108), -INT64_C( 5346037864804698528),
         INT64_C( 7949045959357481633), -INT64_C(  586873182692319846),  INT64_C( 3150362272630606266), -INT64_C( 1214869851901050997) },
      {  INT64_C( 8065805954613763063), -INT64_C( 4638710090095002689), -INT64_C( 6956013085732717091), -INT64_C( 4620835162231406729),
        -INT64_C( 1262416098556647697), -INT64_C(   10240853918359617),  INT64_C( 8051450971285614522), -INT64_C(   24788633565577313) } },
    { {  INT64_C( 3263251090716877996),  INT64_C( 8083162768771822915), -INT64_C(  168649198812717646),  INT64_C( 6580477785673483285),
         INT64_C( 1166855396492994417),  INT64_C( 6655040048003973173),  INT64_C( 7277678794420877456),  INT64_C( 7516255955370600774) },
      {  INT64_C( 4278030445971342977), -INT64_C( 6402850831071796194), -INT64_C( 4202799087610102279),  INT64_C( 1112681084295475986),
        -INT64_C( 7219062955112014193),  INT64_C( 8151382797754562144), -INT64_C( 8950229773360445051),  INT64_C( 2122882682639385816) },
      {  INT64_C( 4566649574931687085), -INT64_C(  635665364676594337), -INT64_C(  167479262595645957),  INT64_C( 6877996840495281943),
        -INT64_C( 7209773731122511873),  INT64_C( 9034212056335119989), -INT64_C( 1729409199123072619),  INT64_C( 9043226952094182878) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_or_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_or_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 3661319176986321035), -INT64_C( 1528367236969856416), -INT64_C( 5104464832910522267), -INT64_C(  726225875027442177),
        -INT64_C( 9014696675506224050), -INT64_C( 9051454475470047648), -INT64_C( 7795981007606036274),  INT64_C( 6143071551276653025) },
      UINT8_C( 23),
      { -INT64_C( 2050057107331807281), -INT64_C( 5815130737346673481),  INT64_C( 5364304135242080628), -INT64_C( 3735552285056320386),
        -INT64_C( 3207261001712377427),  INT64_C( 8259492860388211849),  INT64_C( 3997073007981093630),  INT64_C( 2160162594097776945) },
      {  INT64_C(  126158479260381905), -INT64_C( 6115446643842019152), -INT64_C( 3935210835850746765), -INT64_C( 1393598984056493285),
         INT64_C( 4853923356578766820),  INT64_C( 5909307511037644458),  INT64_C( 5303518722588138295),  INT64_C( 7392232739012120718) },
      { -INT64_C( 2032038225924589601), -INT64_C( 5805823884632582985), -INT64_C( 3786446772757633673), -INT64_C(  726225875027442177),
        -INT64_C( 3207234049702002707), -INT64_C( 9051454475470047648), -INT64_C( 7795981007606036274),  INT64_C( 6143071551276653025) } },
    { { -INT64_C( 5498580644180955173), -INT64_C( 1324888738294488212), -INT64_C( 8260069012969546521), -INT64_C( 8516644795866271002),
         INT64_C( 1943488705661997116), -INT64_C(  725077280813678004), -INT64_C( 1609078079256613601), -INT64_C( 1526402865792097894) },
      UINT8_C(176),
      {  INT64_C( 3007692236954415048),  INT64_C( 6701123904443236317), -INT64_C( 6252604006634414793),  INT64_C( 2048083762722865819),
         INT64_C( 2359967858908056176),  INT64_C( 6563714040126830346),  INT64_C( 6453153310558549276), -INT64_C( 7554927845201025609) },
      {  INT64_C(  698040348146906443), -INT64_C( 3790102690545815577),  INT64_C( 5505156404407722062),  INT64_C( 5251899737511525155),
        -INT64_C( 1784405116429397709),  INT64_C( 2859601976980864821), -INT64_C( 4496111277563420743), -INT64_C( 6675762436323413738) },
      { -INT64_C( 5498580644180955173), -INT64_C( 1324888738294488212), -INT64_C( 8260069012969546521), -INT64_C( 8516644795866271002),
        -INT64_C( 1730280267062453389),  INT64_C( 9205357625425985343), -INT64_C( 1609078079256613601), -INT64_C( 5224195981274253897) } },
    { { -INT64_C( 1448449652967460354), -INT64_C( 1580342141860718318), -INT64_C(  703174105201359394),  INT64_C( 3807339619423897795),
        -INT64_C( 3443535460487156588),  INT64_C( 7138148802329511411), -INT64_C( 2456682694661159150), -INT64_C( 1742921284027617180) },
      UINT8_C(146),
      {  INT64_C( 7858536605820203522), -INT64_C( 8450066526728211236), -INT64_C( 8812274993042510540),  INT64_C( 2309150471083347858),
         INT64_C( 6541562564319206642),  INT64_C( 4850022823630191232),  INT64_C(  542199368696197739),  INT64_C( 1357143831010631574) },
      { -INT64_C( 2323799827673247000), -INT64_C( 1278757848781503250), -INT64_C( 5665963620397296669),  INT64_C( 5910195416529821395),
        -INT64_C( 7743243952069250739),  INT64_C(   48926196180856124), -INT64_C( 2057381896675353690),  INT64_C( 6940569379697776789) },
      { -INT64_C( 1448449652967460354), -INT64_C( 1226116251400342274), -INT64_C(  703174105201359394),  INT64_C( 3807339619423897795),
        -INT64_C( 2392962817406960129),  INT64_C( 7138148802329511411), -INT64_C( 2456682694661159150),  INT64_C( 8274760820845895575) } },
    { { -INT64_C( 4717636599774705331), -INT64_C( 7862956277528454717), -INT64_C( 8810793557811840571), -INT64_C( 5182845562633547874),
         INT64_C( 8655129809156513502), -INT64_C( 8445873536746546612), -INT64_C( 9157761006655562380),  INT64_C( 2363057480066559400) },
      UINT8_C( 98),
      {  INT64_C( 4032004374184492918),  INT64_C( 1411463924934257895),  INT64_C( 5811463090775456485), -INT64_C(  410194232069670488),
        -INT64_C( 2664275056143473359),  INT64_C( 7912142335821808323), -INT64_C( 2107665682917506163),  INT64_C( 3839350586672632522) },
      {  INT64_C( 4994785017172650882),  INT64_C( 7926843187899878924),  INT64_C( 5309073803146325392),  INT64_C( 2029400058945305840),
         INT64_C( 4869203973805059531), -INT64_C( 1188711299264572879), -INT64_C( 1437378370731352506),  INT64_C( 5807402188424652259) },
      { -INT64_C( 4717636599774705331),  INT64_C( 9194045827134222063), -INT64_C( 8810793557811840571), -INT64_C( 5182845562633547874),
         INT64_C( 8655129809156513502), -INT64_C( 1167034974457589005), -INT64_C( 1239202385541472305),  INT64_C( 2363057480066559400) } },
    { {  INT64_C( 4517725110942902839), -INT64_C( 6336335555842766399),  INT64_C( 1447478681830017582),  INT64_C( 4031892649141138975),
        -INT64_C( 9034405839403380397), -INT64_C( 1158353365791149855), -INT64_C(  882088891294066307), -INT64_C( 3504490474468060634) },
      UINT8_C( 29),
      { -INT64_C( 4667029045628751869), -INT64_C( 4308696435055886588), -INT64_C(  657654580496571768),  INT64_C( 7407085908525188386),
         INT64_C( 1081141035830127217),  INT64_C( 8622980344830132347), -INT64_C( 1263805274432454180), -INT64_C( 7287555194217502700) },
      {  INT64_C( 1525554353750646289), -INT64_C( 4052775759662812837),  INT64_C( 1558454381270528486), -INT64_C( 5374365949207233370),
         INT64_C( 8914872304015656200),  INT64_C( 4441260993943024367),  INT64_C( 5660620830631061880), -INT64_C( 8347834756869776691) },
      { -INT64_C( 4666859652051386861), -INT64_C( 6336335555842766399), -INT64_C(  576532612561505298), -INT64_C(  582253082654655066),
         INT64_C( 9203102728485793657), -INT64_C( 1158353365791149855), -INT64_C(  882088891294066307), -INT64_C( 3504490474468060634) } },
    { { -INT64_C( 1915465842771953458),  INT64_C( 3399199257075330967),  INT64_C( 8737043393532140619), -INT64_C( 5376996792679637424),
         INT64_C( 1000939817634336777),  INT64_C( 5774220578764071997),  INT64_C( 5866364029680994883),  INT64_C( 7961950062473794786) },
      UINT8_C( 23),
      { -INT64_C(  690006965118734815),  INT64_C( 5476461271299415957), -INT64_C( 5839587232107859825),  INT64_C( 6729364364179259076),
         INT64_C( 2249821286577271023),  INT64_C( 1163861103514235992), -INT64_C( 3038495911641386035),  INT64_C( 6444796657872791766) },
      {  INT64_C( 5556831668248771700), -INT64_C( 6585865916961523192), -INT64_C( 6005372159066849855), -INT64_C( 4517902839317542518),
        -INT64_C( 4220474570919909077), -INT64_C( 6678018137657469738),  INT64_C( 5271777547077885606),  INT64_C( 4824634032580847846) },
      { -INT64_C(   36628065222132107), -INT64_C( 1397719145344139363), -INT64_C( 5837298578245157425), -INT64_C( 5376996792679637424),
        -INT64_C( 2342438332237088273),  INT64_C( 5774220578764071997),  INT64_C( 5866364029680994883),  INT64_C( 7961950062473794786) } },
    { {  INT64_C( 7419346753861956468), -INT64_C( 3032499050133568155),  INT64_C( 7173524459889975172), -INT64_C( 3910097177235961246),
         INT64_C( 6068022085240364863),  INT64_C( 3691409832953989430),  INT64_C( 7549444775824821221),  INT64_C( 2877130456848030926) },
      UINT8_C(189),
      {  INT64_C( 8434961914935441128),  INT64_C( 4181408283957163164), -INT64_C( 8385236112120976945), -INT64_C( 7213189133895481974),
         INT64_C( 3400362389193134736),  INT64_C( 8998289558329185784),  INT64_C( 6015666477394239681),  INT64_C( 1730615395330354870) },
      { -INT64_C(  885514798386415373),  INT64_C( 2414626903070213977),  INT64_C( 8626518373486940571), -INT64_C( 8639341606507367558),
         INT64_C( 5425647820748090972), -INT64_C(  338100511992322612),  INT64_C( 7412105766290546596), -INT64_C(  114978529918468790) },
      { -INT64_C(  594750046436589829), -INT64_C( 3032499050133568155), -INT64_C(   20357526653371937), -INT64_C( 7205760816116402182),
         INT64_C( 8033234211055861468), -INT64_C(    4824735702452740),  INT64_C( 7549444775824821221), -INT64_C(  114872941904265218) } },
    { {  INT64_C(  625615774204458609),  INT64_C( 8566553283545857927), -INT64_C( 6758573349697469982), -INT64_C( 7919606190865015434),
        -INT64_C( 7520268040824001185),  INT64_C( 4318214654586181540), -INT64_C(  532481923674945118),  INT64_C( 7318111929964240347) },
      UINT8_C(149),
      {  INT64_C( 8776737915996077957), -INT64_C(  378849714569591622),  INT64_C( 8228485904037885975),  INT64_C( 4702846238338854277),
        -INT64_C( 2380090673657550627), -INT64_C( 1105711942724826500),  INT64_C( 2105563224312456048),  INT64_C( 7755777512957522084) },
      { -INT64_C( 4397516249695121930),  INT64_C( 1978513632449786540), -INT64_C( 3030895946949936582), -INT64_C( 4628900207231910967),
         INT64_C( 4245998957238004964), -INT64_C( 5942160567249708092),  INT64_C( 5473403526820358895), -INT64_C( 1823284935291426262) },
      { -INT64_C(  288794531985358857),  INT64_C( 8566553283545857927), -INT64_C(  580546031377401281), -INT64_C( 7919606190865015434),
        -INT64_C(   72908966077928195),  INT64_C( 4318214654586181540), -INT64_C(  532481923674945118), -INT64_C( 1174754992533554514) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_or_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_or_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_or_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(243),
      {  INT64_C( 6607773143298311001),  INT64_C(  747251710714923559), -INT64_C(  651814038446241381),  INT64_C( 8581770847649122825),
         INT64_C( 7934284021902262550),  INT64_C( 8289319149355930503),  INT64_C( 3001449180940391964),  INT64_C( 4515533899237856276) },
      { -INT64_C(  513943133449407448), -INT64_C( 4505286668915895762),  INT64_C( 3395742826455355607), -INT64_C(  282304193843007825),
         INT64_C( 6821627629316871233),  INT64_C( 8918217582631560028),  INT64_C(   70529624180930224), -INT64_C( 6065807279509775104) },
      { -INT64_C(  288335948595349639), -INT64_C( 3783368943172363729),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C( 9133158688141268311),  INT64_C( 8920469764648055775),  INT64_C( 3026377657701299900), -INT64_C( 4612831250118353644) } },
    { UINT8_C(  1),
      { -INT64_C( 5916729309565788611), -INT64_C( 2340802351535626765), -INT64_C( 6498641173007020482), -INT64_C( 5551324850384034809),
        -INT64_C( 6354546414697503463),  INT64_C( 3137880762619666690), -INT64_C(  920480431848309941),  INT64_C( 2132696337155049840) },
      { -INT64_C( 1054667020777172734),  INT64_C( 2399336497810298931), -INT64_C( 8952331340402571293), -INT64_C( 8621859435101771209),
        -INT64_C( 2399964864933094702),  INT64_C( 6370987589993713268),  INT64_C( 4983490336306700178),  INT64_C( 7179822547708514708) },
      { -INT64_C(  144220815423840449),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(116),
      {  INT64_C( 2590131453595428065), -INT64_C( 4189770811124925443),  INT64_C( 4149812691542966297),  INT64_C( 3956181737923129753),
         INT64_C(  165407252905396050), -INT64_C( 1204109446590307104), -INT64_C( 3492953594650288732), -INT64_C( 3259152023684951750) },
      { -INT64_C(  200333092127606735),  INT64_C( 5471818189430293034), -INT64_C( 6489652522350222869), -INT64_C( 8087346319675083920),
         INT64_C( 3354335113734700401), -INT64_C( 4002821708591526015), -INT64_C( 7273356092679582465), -INT64_C( 5910324440049368307) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 4758299390809218565),  INT64_C(                   0),
         INT64_C( 3373195054405894003), -INT64_C( 1190288546101006367), -INT64_C( 2337410815097102849),  INT64_C(                   0) } },
    { UINT8_C( 12),
      { -INT64_C( 2864220326792774464), -INT64_C( 1762457922283673492),  INT64_C( 5065896484844794982),  INT64_C( 8328615529749329978),
         INT64_C( 5768393017907127660), -INT64_C( 6368824068940315686), -INT64_C( 7520672327153504424), -INT64_C( 4678758067609787412) },
      { -INT64_C( 8102255326642551973),  INT64_C( 2786851643915069892), -INT64_C( 5510677752417447023), -INT64_C( 5162986244131412153),
         INT64_C( 1435388577372917158), -INT64_C( 6564906772021628688), -INT64_C( 3327850568640928169),  INT64_C( 3230985858688164544) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C(  590059752909506569), -INT64_C(  297958856714895489),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 17),
      { -INT64_C( 6610032513410692915),  INT64_C( 5594978057287238282), -INT64_C( 5204005490663111699), -INT64_C( 8581493933975214829),
         INT64_C( 4902204179320948720),  INT64_C(  456614092410699298), -INT64_C( 1709162420342280708), -INT64_C( 2450617988190075718) },
      {  INT64_C( 2156461137493875438), -INT64_C( 5465262492535586809), -INT64_C( 4366648816288058199), -INT64_C( 4982609783682489072),
         INT64_C( 7721228476751223568),  INT64_C( 8231933372063829456),  INT64_C( 2284545533172286183),  INT64_C( 3675770877265112443) },
      { -INT64_C( 4761012908859598097),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C( 8011746500201635824),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 17),
      { -INT64_C( 5582809435615183515), -INT64_C( 3744762563012063892), -INT64_C( 7371835497483863841),  INT64_C( 6853898065162336278),
        -INT64_C( 2358591587479154131), -INT64_C( 8161079221511339865), -INT64_C( 6250023680286156291),  INT64_C( 2073858079106334829) },
      {  INT64_C( 2068827291285117796), -INT64_C( 5192426373283728205), -INT64_C( 7664946069928561564), -INT64_C( 8380983232254093916),
         INT64_C(  373824410368499615),  INT64_C( 7943320640690863819), -INT64_C( 1816632622552081140),  INT64_C( 3102463971999532891) },
      { -INT64_C( 4704586547721142427),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C( 2345074191453724737),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(137),
      { -INT64_C( 5885112414594449552), -INT64_C( 5643532512505898935),  INT64_C( 8497405936442532747), -INT64_C( 1295018545582674093),
         INT64_C(  903476142813930091), -INT64_C( 1051488916752940894), -INT64_C( 8691544632263025857), -INT64_C(  555617802753056034) },
      {  INT64_C( 3296173732352213486),  INT64_C(  573168574036995229), -INT64_C( 4439328891215776088),  INT64_C( 7223045329763854775),
         INT64_C(  784422852770533674), -INT64_C( 4895130134412146879),  INT64_C( 1305760606038066551), -INT64_C( 3440305195935384217) },
      { -INT64_C( 5764609740397346818),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 1279185573845766153),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(  555177849903415297) } },
    { UINT8_C(165),
      {  INT64_C( 3345080952170972503), -INT64_C( 8312238056190872021),  INT64_C( 6629008511800311960),  INT64_C( 8325415308185688911),
        -INT64_C( 1879114047830451114), -INT64_C( 4137482948338176127), -INT64_C( 4454240953403868491),  INT64_C( 6468629226322276662) },
      { -INT64_C( 1194857502905769012), -INT64_C( 1718251684327097038),  INT64_C( 1530075843475047430), -INT64_C( 2416132743736335844),
        -INT64_C( 4184459149005451377),  INT64_C( 8532153540123419981),  INT64_C( 3225750579480306896), -INT64_C( 1730011902885811443) },
      { -INT64_C( 1193700811690481697),  INT64_C(                   0),  INT64_C( 6917524492155520158),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C(  649365526355251251),  INT64_C(                   0), -INT64_C(     576868332913857) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_or_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_or_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_or_si512(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 1982508443), INT32_C( -368650443), INT32_C( -190462634), INT32_C( 1539812062),
                            INT32_C( 1356046477), INT32_C(  862189546), INT32_C( 1762309251), INT32_C(-1019483096),
                            INT32_C( 1873631110), INT32_C(  -15642982), INT32_C( 1155728159), INT32_C(  -93367878),
                            INT32_C( -146552290), INT32_C(-1970782191), INT32_C(-1003501987), INT32_C(  202140132)),
      easysimd_mm512_set_epi32(INT32_C(-1200042690), INT32_C(  641010033), INT32_C( 1397158609), INT32_C(-1526446074),
                            INT32_C(  334871021), INT32_C(-1650252171), INT32_C(  346015053), INT32_C(  -55637966),
                            INT32_C(  208214931), INT32_C(-1364363811), INT32_C(-1538776181), INT32_C( 1682811579),
                            INT32_C( 1143917073), INT32_C( 1704546357), INT32_C( -526995661), INT32_C( 1822747438)),
      easysimd_mm512_set_epi32(INT32_C(  -25498177), INT32_C( -298328203), INT32_C( -135792681), INT32_C(   -3675426),
                            INT32_C( 1408743917), INT32_C(-1075578881), INT32_C( 2107635151), INT32_C(   -4199366),
                            INT32_C( 1877827479), INT32_C(   -4358177), INT32_C( -454351969), INT32_C(  -26215493),
                            INT32_C( -143668705), INT32_C( -275165131), INT32_C( -457183361), INT32_C( 1823271918)) },
    { easysimd_mm512_set_epi32(INT32_C( 1700241441), INT32_C( 1263470492), INT32_C( 1650149592), INT32_C(-1970638798),
                            INT32_C(  779737204), INT32_C(  613390512), INT32_C( 1903573583), INT32_C( 1579794471),
                            INT32_C( -751717923), INT32_C( 1508394249), INT32_C( 1822398709), INT32_C(-1695423756),
                            INT32_C( -107711426), INT32_C( -896105060), INT32_C(  529237571), INT32_C(  178579675)),
      easysimd_mm512_set_epi32(INT32_C( -875424608), INT32_C( 1367415512), INT32_C(  131368493), INT32_C(  402371418),
                            INT32_C( 1616362823), INT32_C(-2005702634), INT32_C( -384303225), INT32_C( 1749032691),
                            INT32_C(-1558748500), INT32_C( 1913654477), INT32_C(-1008865960), INT32_C(-1123849687),
                            INT32_C(   83161050), INT32_C( 1602030576), INT32_C( -421382217), INT32_C(  749511091)),
      easysimd_mm512_set_epi32(INT32_C( -271076191), INT32_C( 1540300764), INT32_C( 1742720765), INT32_C(-1610878086),
                            INT32_C( 1853881719), INT32_C(-1392510794), INT32_C( -109576241), INT32_C( 2120867575),
                            INT32_C( -214436867), INT32_C( 2079870413), INT32_C( -270533123), INT32_C(-1074533635),
                            INT32_C(  -34275330), INT32_C( -536899588), INT32_C(   -1329161), INT32_C(  783084027)) },
    { easysimd_mm512_set_epi32(INT32_C( -198115845), INT32_C( 1907998628), INT32_C(-1258846188), INT32_C(  680092843),
                            INT32_C( 1806004257), INT32_C(  809421067), INT32_C(  530040867), INT32_C(-1316120429),
                            INT32_C(  457386668), INT32_C(  815983260), INT32_C( 1763745819), INT32_C( 2000730006),
                            INT32_C(-1234863927), INT32_C( 1084046116), INT32_C(  472797794), INT32_C(-1156282262)),
      easysimd_mm512_set_epi32(INT32_C( 1507578237), INT32_C( 1923983420), INT32_C( 1994590915), INT32_C( 1646522822),
                            INT32_C(-2017657183), INT32_C(-1653054803), INT32_C(-1634459065), INT32_C( -572700558),
                            INT32_C( 1977566390), INT32_C( -646523450), INT32_C( -234450626), INT32_C(  330831665),
                            INT32_C( 1706081529), INT32_C(-1640342739), INT32_C( -694582053), INT32_C(  598851851)),
      easysimd_mm512_set_epi32(INT32_C(  -33817089), INT32_C( 1941815228), INT32_C( -151549225), INT32_C( 1789654511),
                            INT32_C( -272761183), INT32_C(-1115755601), INT32_C(-1617434009), INT32_C(  -35792653),
                            INT32_C( 2145349310), INT32_C( -101253154), INT32_C(  -81350337), INT32_C( 2012788663),
                            INT32_C( -134873863), INT32_C( -557876435), INT32_C( -557852933), INT32_C(-1145714325)) },
    { easysimd_mm512_set_epi32(INT32_C(  568896963), INT32_C( -561959153), INT32_C(  769261839), INT32_C(  619550472),
                            INT32_C( 1265145937), INT32_C(-1898129853), INT32_C(-1844756744), INT32_C(  253926616),
                            INT32_C(-1200681430), INT32_C(  757779385), INT32_C(-1090889117), INT32_C( 2001359420),
                            INT32_C( -628410960), INT32_C(-1884853401), INT32_C(  464697363), INT32_C( -267213390)),
      easysimd_mm512_set_epi32(INT32_C( 1305596604), INT32_C( 1367027235), INT32_C( 1022068839), INT32_C(-1304299428),
                            INT32_C(-1551155443), INT32_C(-1757021038), INT32_C( -634643752), INT32_C(  417623958),
                            INT32_C( 1338218088), INT32_C( 1144004768), INT32_C( -119591543), INT32_C(  343634162),
                            INT32_C(-1756432337), INT32_C( -336536481), INT32_C(  155367900), INT32_C(   27211228)),
      easysimd_mm512_set_epi32(INT32_C( 1845100543), INT32_C( -537186513), INT32_C( 1039894895), INT32_C(-1225916580),
                            INT32_C( -336871587), INT32_C(-1612841261), INT32_C( -634437896), INT32_C(  535232478),
                            INT32_C(   -1073558), INT32_C( 1832577977), INT32_C(  -16810005), INT32_C( 2004579070),
                            INT32_C( -540017217), INT32_C( -268959873), INT32_C(  468892127), INT32_C( -241191426)) },
    { easysimd_mm512_set_epi32(INT32_C(-1566019929), INT32_C( 1771648205), INT32_C(  293391222), INT32_C( -190388911),
                            INT32_C(-1413267332), INT32_C( -491216745), INT32_C(-2017086754), INT32_C( -505487315),
                            INT32_C(-1311872315), INT32_C( 1730833859), INT32_C( 1507236184), INT32_C(  127469321),
                            INT32_C(-1954223251), INT32_C(-1913468253), INT32_C(  390805157), INT32_C( 1427395916)),
      easysimd_mm512_set_epi32(INT32_C( -290198315), INT32_C( -186963818), INT32_C(  337890960), INT32_C( -133116402),
                            INT32_C( -567590842), INT32_C( 1356957734), INT32_C( -411285842), INT32_C(  212429154),
                            INT32_C(  561941682), INT32_C( 1263368380), INT32_C(   33943343), INT32_C(  477355785),
                            INT32_C(  464038301), INT32_C(  283034157), INT32_C(  882337256), INT32_C( 1854097219)),
      easysimd_mm512_set_epi32(INT32_C( -289673481), INT32_C(  -35963681), INT32_C(  360697846), INT32_C(  -55120033),
                            INT32_C(   -1359746), INT32_C( -218519369), INT32_C( -402786562), INT32_C( -301995665),
                            INT32_C(-1308692745), INT32_C( 1869577727), INT32_C( 1540882303), INT32_C(  536600329),
                            INT32_C(-1683166211), INT32_C(-1644246353), INT32_C(  937392109), INT32_C( 2140624719)) },
    { easysimd_mm512_set_epi32(INT32_C( 1586789989), INT32_C( 1873262060), INT32_C(   -1228101), INT32_C( 1094551912),
                            INT32_C( 1242820965), INT32_C( -129127728), INT32_C(  916155808), INT32_C( 1457274373),
                            INT32_C( -162664167), INT32_C( -307612047), INT32_C(-2058619353), INT32_C( 1041657370),
                            INT32_C(-1303652034), INT32_C( 1318052527), INT32_C(  343091765), INT32_C(-1843970146)),
      easysimd_mm512_set_epi32(INT32_C( -418596097), INT32_C( 1359591501), INT32_C( 1365241616), INT32_C(  975187949),
                            INT32_C( 2075206187), INT32_C(   49913508), INT32_C(  982225383), INT32_C( 2039004600),
                            INT32_C( -658027813), INT32_C( 1363761789), INT32_C( -596362918), INT32_C( -188756489),
                            INT32_C( 2075405229), INT32_C( -261325870), INT32_C( 1149275923), INT32_C( 1906788899)),
      easysimd_mm512_set_epi32(INT32_C(   -6488321), INT32_C( 2142223853), INT32_C(      -4165), INT32_C( 2067640301),
                            INT32_C( 2075390831), INT32_C(  -84033804), INT32_C( 1050410471), INT32_C( 2145173437),
                            INT32_C(  -19924005), INT32_C(  -34898307), INT32_C( -578814081), INT32_C(  -20975617),
                            INT32_C(  -67109953), INT32_C(  -17826817), INT32_C( 1425256247), INT32_C( -206078017)) },
    { easysimd_mm512_set_epi32(INT32_C(-2074326161), INT32_C(-2000089664), INT32_C(  -95906603), INT32_C(-2144457962),
                            INT32_C( -460603570), INT32_C( -616108121), INT32_C(-1801036003), INT32_C(  192023719),
                            INT32_C( 1229400941), INT32_C(   53109497), INT32_C( 1637729546), INT32_C( -377510882),
                            INT32_C(  959365464), INT32_C( -183985269), INT32_C(  446964672), INT32_C( -984185866)),
      easysimd_mm512_set_epi32(INT32_C(-1212943296), INT32_C(   40655504), INT32_C( 1783466062), INT32_C(-1105776557),
                            INT32_C( 2093068641), INT32_C(  923055475), INT32_C(-2145339184), INT32_C(  312550463),
                            INT32_C( -600919225), INT32_C(-1156369187), INT32_C( -442421904), INT32_C( -479777830),
                            INT32_C(  786467717), INT32_C(-1353894968), INT32_C(-2102502413), INT32_C(  630995848)),
      easysimd_mm512_set_epi32(INT32_C(-1207959697), INT32_C(-1964154928), INT32_C(  -95576865), INT32_C(-1103152297),
                            INT32_C(  -53756049), INT32_C(  -12124169), INT32_C(-1800994851), INT32_C(  468921535),
                            INT32_C( -579880081), INT32_C(-1153730819), INT32_C( -440537734), INT32_C( -343953442),
                            INT32_C( 1072614365), INT32_C(  -11682869), INT32_C(-1699841037), INT32_C( -438387714)) },
    { easysimd_mm512_set_epi32(INT32_C(-2063919183), INT32_C(  261182590), INT32_C( 1716894204), INT32_C(  315016729),
                            INT32_C(-1244972332), INT32_C( 1333991353), INT32_C( 1246104528), INT32_C(-1234716491),
                            INT32_C( -852837622), INT32_C(  266496100), INT32_C(-2090175093), INT32_C( 1822414148),
                            INT32_C(-1888096784), INT32_C(-1814389856), INT32_C(  716652272), INT32_C(-1702112633)),
      easysimd_mm512_set_epi32(INT32_C( -775162340), INT32_C( -717192300), INT32_C(  657226535), INT32_C( -646565165),
                            INT32_C( 1464387491), INT32_C(-1521859395), INT32_C(  -74746289), INT32_C( -342854144),
                            INT32_C( 1370164421), INT32_C( 1847323166), INT32_C(  -31713278), INT32_C( 2054986117),
                            INT32_C(-1330721270), INT32_C(  155186332), INT32_C( 1062642768), INT32_C(-1225803976)),
      easysimd_mm512_set_epi32(INT32_C( -704907331), INT32_C( -539895810), INT32_C( 1736309759), INT32_C( -604571941),
                            INT32_C( -137625609), INT32_C( -271886403), INT32_C(  -70287393), INT32_C(    -524619),
                            INT32_C( -575946801), INT32_C( 1878780542), INT32_C(   -8487541), INT32_C( 2130697669),
                            INT32_C(-1073745926), INT32_C(-1677721668), INT32_C( 1073200880), INT32_C(-1091569729)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_or_si512(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_or_si512");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_or_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_or_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_or_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_or_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_or_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_or_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_or_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_or_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_or_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_or_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_or_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_or_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_or_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_or_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_or_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_or_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_or_si512)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
