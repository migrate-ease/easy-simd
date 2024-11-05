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

#define EASYSIMD_TEST_X86_AVX512_INSN and

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/and.h>

static int
test_easysimd_mm_mask_and_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[4];
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(   522136421), -INT32_C(    79310835), -INT32_C(   736344146), -INT32_C(   371121230) },
      UINT8_C(234),
      { -INT32_C(  1479045656),  INT32_C(   298943209),  INT32_C(   454726652), -INT32_C(   343899712) },
      {  INT32_C(  1895407520),  INT32_C(   740226277),  INT32_C(   887091728), -INT32_C(  1138833196) },
      {  INT32_C(   522136421),  INT32_C(     1081569), -INT32_C(   736344146), -INT32_C(  1476362048) } },
    { {  INT32_C(  1046738517),  INT32_C(  1951348088), -INT32_C(  1987089720), -INT32_C(  1955328021) },
      UINT8_C(175),
      {  INT32_C(  1653931117),  INT32_C(   225624090),  INT32_C(  1759618719), -INT32_C(  1145201211) },
      {  INT32_C(   909376513), -INT32_C(  1241601973),  INT32_C(  1201702711),  INT32_C(  1777741052) },
      {  INT32_C(   571538433),  INT32_C(    91389962),  INT32_C(  1084261911),  INT32_C(   699665604) } },
    { {  INT32_C(  1120635432), -INT32_C(   363905462), -INT32_C(  1454231324), -INT32_C(   815525938) },
      UINT8_C( 11),
      {  INT32_C(  1079444888), -INT32_C(  1955132413), -INT32_C(   645415251),  INT32_C(  1040314548) },
      { -INT32_C(    91667524), -INT32_C(  1008831597), -INT32_C(   728659771), -INT32_C(  2065669908) },
      {  INT32_C(  1073807768), -INT32_C(  2091515901), -INT32_C(  1454231324),  INT32_C(    67133604) } },
    { {  INT32_C(  1774466917), -INT32_C(   252429245), -INT32_C(  1345750022),  INT32_C(   669895275) },
      UINT8_C( 14),
      { -INT32_C(   375316106), -INT32_C(  2001837056),  INT32_C(  1450476534), -INT32_C(  1698891421) },
      { -INT32_C(   102947395), -INT32_C(  1795895783),  INT32_C(  1660920727),  INT32_C(   124790672) },
      {  INT32_C(  1774466917), -INT32_C(  2136718336),  INT32_C(  1114932118),  INT32_C(    36708608) } },
    { {  INT32_C(  1223692616),  INT32_C(  1808834421), -INT32_C(  2050865886), -INT32_C(    81822146) },
      UINT8_C(163),
      { -INT32_C(   893585924), -INT32_C(  1939713815),  INT32_C(  1981596751),  INT32_C(  1170154292) },
      { -INT32_C(  1279654124),  INT32_C(   500508376),  INT32_C(  1717263080),  INT32_C(  1963546233) },
      { -INT32_C(  2101869292),  INT32_C(   205521096), -INT32_C(  2050865886), -INT32_C(    81822146) } },
    { {  INT32_C(   876660043),  INT32_C(  1690346005), -INT32_C(  1696866970),  INT32_C(   350198272) },
      UINT8_C(161),
      { -INT32_C(  1082538087), -INT32_C(   156789092),  INT32_C(  1215237617), -INT32_C(   611064554) },
      { -INT32_C(   940521435),  INT32_C(  1697469832), -INT32_C(   899234000),  INT32_C(  1064008358) },
      { -INT32_C(  2022653951),  INT32_C(  1690346005), -INT32_C(  1696866970),  INT32_C(   350198272) } },
    { { -INT32_C(   553654974),  INT32_C(  1842718331), -INT32_C(   894090060),  INT32_C(  1319520297) },
      UINT8_C( 16),
      { -INT32_C(   342289002),  INT32_C(   152829506), -INT32_C(   542120604),  INT32_C(   908193617) },
      { -INT32_C(  1800273938),  INT32_C(   424156884),  INT32_C(   474092499),  INT32_C(  1328320953) },
      { -INT32_C(   553654974),  INT32_C(  1842718331), -INT32_C(   894090060),  INT32_C(  1319520297) } },
    { { -INT32_C(   381958746),  INT32_C(   686970564), -INT32_C(  1928879556),  INT32_C(  2143496337) },
      UINT8_C( 40),
      { -INT32_C(  1812196492),  INT32_C(  1868961116), -INT32_C(   383221160),  INT32_C(  1955559343) },
      {  INT32_C(   154695858),  INT32_C(   222650731), -INT32_C(  1851862424),  INT32_C(   163126677) },
      { -INT32_C(   381958746),  INT32_C(   686970564), -INT32_C(  1928879556),  INT32_C(     8983941) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_and_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_and_epi32");
    easysimd_test_x86_assert_equal_i32x4(easysimd_mm_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_and_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_and_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[] = {
    { UINT8_C(210),
      { -INT32_C(   844403173),  INT32_C(  1228124655), -INT32_C(  1257796942),  INT32_C(  1676815307) },
      { -INT32_C(  2005724645),  INT32_C(  1679875999), -INT32_C(   996586481), -INT32_C(    23700510) },
      {  INT32_C(           0),  INT32_C(  1075876239),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(201),
      { -INT32_C(   205993150), -INT32_C(  1985609218),  INT32_C(   877943561),  INT32_C(  1599059789) },
      {  INT32_C(   184473386),  INT32_C(  1125737207),  INT32_C(  1462165244), -INT32_C(  1239341964) },
      {  INT32_C(    45662978),  INT32_C(           0),  INT32_C(           0),  INT32_C(   369173572) } },
    { UINT8_C(239),
      { -INT32_C(   605115687), -INT32_C(  1411090608), -INT32_C(  2080892725),  INT32_C(  1051547495) },
      { -INT32_C(  1204439210), -INT32_C(  1347126831), -INT32_C(  1021113185),  INT32_C(   112384557) },
      { -INT32_C(  1742429616), -INT32_C(  1415286448), -INT32_C(  2095052661),  INT32_C(   111170085) } },
    { UINT8_C(132),
      {  INT32_C(   399827360), -INT32_C(   572293179), -INT32_C(   834378121), -INT32_C(   886799853) },
      {  INT32_C(   832363960), -INT32_C(  1647293551),  INT32_C(  1221235566), -INT32_C(   422784699) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1212154470),  INT32_C(           0) } },
    { UINT8_C(178),
      {  INT32_C(   528022944),  INT32_C(  1184257504),  INT32_C(   475686298),  INT32_C(  1725179273) },
      {  INT32_C(   217580993),  INT32_C(  1752864213),  INT32_C(   833536864),  INT32_C(   820221071) },
      {  INT32_C(           0),  INT32_C(  1074927040),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(145),
      { -INT32_C(  1317908645),  INT32_C(  1263253734),  INT32_C(   936666898),  INT32_C(  1106787132) },
      { -INT32_C(   938015693),  INT32_C(  1109950335), -INT32_C(  1043244755), -INT32_C(  1739390659) },
      { -INT32_C(  2146302957),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 81),
      {  INT32_C(  2100775365), -INT32_C(    57638252), -INT32_C(  1858549930), -INT32_C(   993756736) },
      {  INT32_C(   272862608),  INT32_C(   255690165),  INT32_C(  1481441111),  INT32_C(   397010002) },
      {  INT32_C(   268634496),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 46),
      {  INT32_C(  1656919264), -INT32_C(   323371484),  INT32_C(  1890339574), -INT32_C(  1677692913) },
      {  INT32_C(   961614004),  INT32_C(  1301307470), -INT32_C(  1851790932),  INT32_C(  1941943954) },
      {  INT32_C(           0),  INT32_C(  1284513796),  INT32_C(   277629092),  INT32_C(   268447746) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_and_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_and_epi32");
    easysimd_test_x86_assert_equal_i32x4(easysimd_mm_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_and_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_and_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[2];
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 2592670183254177072),  INT64_C( 5596175102879998853) },
      UINT8_C( 20),
      {  INT64_C( 7902564569615128289), -INT64_C( 6163443317435384790) },
      {  INT64_C( 2535976289046955176), -INT64_C( 1129265450719484508) },
      {  INT64_C( 2592670183254177072),  INT64_C( 5596175102879998853) } },
    { { -INT64_C( 8544492463665786014),  INT64_C( 2502109653246176537) },
      UINT8_C( 75),
      {  INT64_C( 7235999266845216046), -INT64_C( 3271474806021510356) },
      {  INT64_C( 8038003954618046357), -INT64_C( 9071081619805594998) },
      {  INT64_C( 7208057392792175876), -INT64_C( 9072217418547295736) } },
    { {  INT64_C(  160921046529698008), -INT64_C( 6542935241345290912) },
      UINT8_C(106),
      {  INT64_C( 4515101219587102563),  INT64_C(   47309343142542021) },
      {  INT64_C( 7526035845499289436),  INT64_C( 7372705963292807187) },
      {  INT64_C(  160921046529698008),  INT64_C(      18694919966721) } },
    { {  INT64_C( 6851712926659952180),  INT64_C( 1675294785624057767) },
      UINT8_C( 87),
      {  INT64_C( 7639083095015106582), -INT64_C(  413255585406063887) },
      { -INT64_C( 2870214418508801194), -INT64_C( 4661036872098420446) },
      {  INT64_C( 5188834040661149718), -INT64_C( 5025874075700926432) } },
    { { -INT64_C( 3614223520433741101),  INT64_C( 7841644459675304861) },
      UINT8_C( 57),
      { -INT64_C( 9044280175496597657),  INT64_C( 2906487272829056470) },
      { -INT64_C( 2528541833961422183), -INT64_C(  359056977618091604) },
      { -INT64_C( 9194028161916812799),  INT64_C( 7841644459675304861) } },
    { { -INT64_C( 8580815580111212978), -INT64_C( 7478967156650470073) },
      UINT8_C( 32),
      { -INT64_C( 8163452946458483656), -INT64_C(  993573115921268856) },
      { -INT64_C( 7022189854693548429),  INT64_C(  483071320903227701) },
      { -INT64_C( 8580815580111212978), -INT64_C( 7478967156650470073) } },
    { { -INT64_C( 8325814290077487846),  INT64_C( 2934563230409004868) },
      UINT8_C( 71),
      { -INT64_C( 7873014555552486051),  INT64_C( 8240293215473783051) },
      {  INT64_C( 5584185044667613814),  INT64_C( 6730113972672022133) },
      {  INT64_C(   17172928656326740),  INT64_C( 5783223354420903937) } },
    { { -INT64_C( 5754294496638191976),  INT64_C( 7091847718425953780) },
      UINT8_C(143),
      {  INT64_C( 7158914834963471841),  INT64_C( 1538472810666752149) },
      { -INT64_C( 7765216143128007771), -INT64_C( 9063525441603619493) },
      {  INT64_C(    6756551566133665),  INT64_C(    4996732739923985) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_and_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_and_epi64");
    easysimd_test_x86_assert_equal_i64x2(easysimd_mm_loadu_epi64(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_mask_and_epi64(src, k, a, b);

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
test_easysimd_mm_maskz_and_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 74),
      { -INT64_C( 2794945852212259455), -INT64_C( 5247955085964040697) },
      { -INT64_C( 2386253639426153003), -INT64_C( 5408009296309816335) },
      {  INT64_C(                   0), -INT64_C( 5466661715449994751) } },
    { UINT8_C(185),
      { -INT64_C( 3331782831892794271), -INT64_C( 3791415818785900023) },
      { -INT64_C( 1262802867593373643),  INT64_C( 8335552762714595344) },
      { -INT64_C( 4593388862822117343),  INT64_C(                   0) } },
    { UINT8_C(201),
      {  INT64_C( 3251534000030953461), -INT64_C( 6367704977786033092) },
      { -INT64_C( 4409849073765308386),  INT64_C( 7264385840658563378) },
      {  INT64_C(    3659862229788692),  INT64_C(                   0) } },
    { UINT8_C(119),
      {  INT64_C( 5171424962097609667), -INT64_C(  385341641716256726) },
      { -INT64_C(  908780233931660256), -INT64_C(  148119358262842830) },
      {  INT64_C( 4845896357514993664), -INT64_C(  531209182892646366) } },
    { UINT8_C( 65),
      {  INT64_C( 8798946576784196955), -INT64_C( 5794366580644338567) },
      { -INT64_C( 7750181305314129951),  INT64_C( 4052620489209604917) },
      {  INT64_C( 1157428446570251585),  INT64_C(                   0) } },
    { UINT8_C( 81),
      { -INT64_C( 1462238008796567015), -INT64_C( 8463981400457597854) },
      {  INT64_C( 2652419217285236934),  INT64_C( 2464939081386281751) },
      {  INT64_C( 2343279185823270912),  INT64_C(                   0) } },
    { UINT8_C( 99),
      {  INT64_C( 4632117233263992528),  INT64_C( 4789732662308958412) },
      { -INT64_C( 6079401135716953791), -INT64_C( 8485108549897910879) },
      {  INT64_C(     140739672019008),  INT64_C(  160023173864128640) } },
    { UINT8_C( 52),
      { -INT64_C( 6175692133739499032), -INT64_C( 8766144021591163509) },
      {  INT64_C(  658671805852541666),  INT64_C( 2725044176554295238) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_and_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_and_epi64");
    easysimd_test_x86_assert_equal_i64x2(easysimd_mm_loadu_epi64(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_and_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_and_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[4];
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(  2065776624),  INT32_C(  1100426302),  INT32_C(   436862508), -INT32_C(   395649999) },
      UINT8_C(188),
      { -INT32_C(   269880905),  INT32_C(  1978993546), -INT32_C(  1043097672), -INT32_C(  1196350855) },
      {  INT32_C(  1878404163), -INT32_C(  1030015036),  INT32_C(   502576705), -INT32_C(   656810976) },
      {  INT32_C(  2065776624),  INT32_C(  1100426302),  INT32_C(    30447104), -INT32_C(  1735327712) } },
    { {  INT32_C(  1472709581), -INT32_C(  2033402674), -INT32_C(   867721389),  INT32_C(    75823297) },
      UINT8_C( 37),
      { -INT32_C(  1293323398), -INT32_C(   973886705),  INT32_C(  2078609567), -INT32_C(  1404518935) },
      {  INT32_C(  1098620805),  INT32_C(   194314604),  INT32_C(  1103978825),  INT32_C(  1617351142) },
      {  INT32_C(     6886144), -INT32_C(  2033402674),  INT32_C(  1103429641),  INT32_C(    75823297) } },
    { {  INT32_C(  1410486085), -INT32_C(  1709636101), -INT32_C(    15270379),  INT32_C(  1101749948) },
      UINT8_C(254),
      {  INT32_C(   678068774),  INT32_C(  2037478679),  INT32_C(   341815874),  INT32_C(  1750712089) },
      { -INT32_C(   681333294), -INT32_C(   991035706),  INT32_C(  1921051668), -INT32_C(  1099906665) },
      {  INT32_C(  1410486085),  INT32_C(  1080128518),  INT32_C(   268476416),  INT32_C(   676364561) } },
    { {  INT32_C(  1541855812), -INT32_C(  1831577776),  INT32_C(   581317385), -INT32_C(   980680717) },
      UINT8_C(172),
      { -INT32_C(   328033042),  INT32_C(  1962948489),  INT32_C(  2030924727), -INT32_C(  1094857757) },
      {  INT32_C(   135141553),  INT32_C(   538026221),  INT32_C(  1175663687), -INT32_C(  1376528193) },
      {  INT32_C(  1541855812), -INT32_C(  1831577776),  INT32_C(  1073819655), -INT32_C(  1397634909) } },
    { { -INT32_C(    23435915),  INT32_C(  1416927900), -INT32_C(   238190323), -INT32_C(    38827444) },
      UINT8_C(162),
      {  INT32_C(  1569654205),  INT32_C(  1252306966), -INT32_C(  1677071421),  INT32_C(  1125234654) },
      { -INT32_C(   337637551),  INT32_C(    83440771), -INT32_C(  1957631487),  INT32_C(  1445809561) },
      { -INT32_C(    23435915),  INT32_C(    10498050), -INT32_C(   238190323), -INT32_C(    38827444) } },
    { {  INT32_C(  1773387090),  INT32_C(   817059949),  INT32_C(   567065923), -INT32_C(   983245452) },
      UINT8_C(237),
      {  INT32_C(  2020651076), -INT32_C(  1820756567),  INT32_C(   321651909), -INT32_C(   278560206) },
      { -INT32_C(  1906520522),  INT32_C(  1070697602),  INT32_C(   917762649), -INT32_C(  1692174250) },
      {  INT32_C(   139493380),  INT32_C(   817059949),  INT32_C(   304087105), -INT32_C(  1960771566) } },
    { { -INT32_C(   770468824), -INT32_C(   832205559), -INT32_C(  1008627055),  INT32_C(  1253197588) },
      UINT8_C( 21),
      { -INT32_C(  1684547570), -INT32_C(  1678453079),  INT32_C(    32647817), -INT32_C(   500527794) },
      {  INT32_C(   770440352), -INT32_C(   205538975), -INT32_C(   502758757),  INT32_C(  1140281909) },
      {  INT32_C(   159635456), -INT32_C(   832205559),  INT32_C(         649),  INT32_C(  1253197588) } },
    { { -INT32_C(   723611861), -INT32_C(   277818779),  INT32_C(  1274045181), -INT32_C(  1892869393) },
      UINT8_C( 22),
      { -INT32_C(   780616424), -INT32_C(    26449028), -INT32_C(   969716109), -INT32_C(   705595578) },
      {  INT32_C(   674940245), -INT32_C(  1759172299),  INT32_C(   881225754), -INT32_C(  1253370211) },
      { -INT32_C(   723611861), -INT32_C(  1776015052),  INT32_C(    67256338), -INT32_C(  1892869393) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_mask_and_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_and_ps");
    easysimd_test_x86_assert_equal_i32x4(easysimd_mm_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_mask_and_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

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
test_easysimd_mm_maskz_and_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[] = {
    { UINT8_C(211),
      {  INT32_C(   776963779),  INT32_C(  1101155827), -INT32_C(   125343615),  INT32_C(   508386393) },
      { -INT32_C(  1068206697),  INT32_C(   182119322), -INT32_C(  2002317454),  INT32_C(   492526682) },
      {  INT32_C(     4457603),  INT32_C(     8538514),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(227),
      { -INT32_C(   136950870),  INT32_C(  1433933805), -INT32_C(    72453985),  INT32_C(   865258942) },
      {  INT32_C(   214848289),  INT32_C(  1014945837), -INT32_C(   594147712),  INT32_C(   197112673) },
      {  INT32_C(    80102176),  INT32_C(   343412781),  INT32_C(           0),  INT32_C(           0) } },
    {    UINT8_MAX,
      { -INT32_C(  1377041515), -INT32_C(   330546565), -INT32_C(  1112913936),  INT32_C(   769580506) },
      { -INT32_C(  2074416213),  INT32_C(  1879349097), -INT32_C(   506339283),  INT32_C(   903929248) },
      { -INT32_C(  2075655295),  INT32_C(  1610875497), -INT32_C(  1585430496),  INT32_C(   633396608) } },
    { UINT8_C(224),
      {  INT32_C(   240902860),  INT32_C(  1996441390), -INT32_C(   816792335), -INT32_C(  2038792805) },
      {  INT32_C(  1861222103), -INT32_C(   476291326),  INT32_C(   243498289),  INT32_C(   703510621) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(155),
      { -INT32_C(  1849083831), -INT32_C(   209567945),  INT32_C(   210653583), -INT32_C(   891022132) },
      {  INT32_C(  1674400260),  INT32_C(  1804906734), -INT32_C(   339172813), -INT32_C(   628690288) },
      {  INT32_C(    29954048),  INT32_C(  1669345318),  INT32_C(           0), -INT32_C(   897314688) } },
    { UINT8_C( 42),
      { -INT32_C(  1889440944),  INT32_C(  1059017965), -INT32_C(   150262813),  INT32_C(  1660671247) },
      {  INT32_C(  1380999074), -INT32_C(  1769620237),  INT32_C(  1982230916),  INT32_C(  1201668343) },
      {  INT32_C(           0),  INT32_C(   369431777),  INT32_C(           0),  INT32_C(  1117782023) } },
    { UINT8_C(107),
      {  INT32_C(  1448728321),  INT32_C(   557422838),  INT32_C(  2016489635), -INT32_C(  1961127124) },
      { -INT32_C(  1619104285),  INT32_C(  1680086259),  INT32_C(   995859003),  INT32_C(  1017619259) },
      {  INT32_C(   374883585),  INT32_C(   538972402),  INT32_C(           0),  INT32_C(   134447912) } },
    { UINT8_C(122),
      { -INT32_C(  1737453056), -INT32_C(    63204917),  INT32_C(  1428730818),  INT32_C(  1010414542) },
      {  INT32_C(  1177540657), -INT32_C(  1753115652), -INT32_C(  1848460050),  INT32_C(  1661668963) },
      {  INT32_C(           0), -INT32_C(  1811836472),  INT32_C(           0),  INT32_C(   537461314) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_maskz_and_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_and_ps");
    easysimd_test_x86_assert_equal_i32x4(easysimd_mm_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_maskz_and_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_and_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[2];
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[] = {
    { {  INT64_C(   60090816304982083),  INT64_C( 8926885697863799321) },
      UINT8_C(224),
      { -INT64_C( 7293092760966773606), -INT64_C( 8762134201275323569) },
      {  INT64_C( 9104815732093212356), -INT64_C( 5618460333724387972) },
      {  INT64_C(   60090816304982083),  INT64_C( 8926885697863799321) } },
    { { -INT64_C( 6062768291948093060), -INT64_C(  826353150829098077) },
      UINT8_C( 10),
      { -INT64_C( 4278643331594570411),  INT64_C( 8116770544040069104) },
      {  INT64_C( 5681422610232829565), -INT64_C( 5222923900572214715) },
      { -INT64_C( 6062768291948093060),  INT64_C( 3495921693914612288) } },
    { {  INT64_C( 6844198672504863399), -INT64_C( 1981817593672509631) },
      UINT8_C( 18),
      { -INT64_C( 3744703564583986869),  INT64_C( 4664573653326660257) },
      { -INT64_C( 4288820011403906624),  INT64_C( 6183739202676099731) },
      {  INT64_C( 6844198672504863399),  INT64_C( 4652504297008431745) } },
    { {  INT64_C( 7702881360273778216), -INT64_C( 5050442862177741661) },
      UINT8_C( 42),
      { -INT64_C(  488681865828961385),  INT64_C( 7870982385800836613) },
      {  INT64_C( 5707644280478520689),  INT64_C( 6809445691753618973) },
      {  INT64_C( 7702881360273778216),  INT64_C( 5476378001883125253) } },
    { {  INT64_C( 8529035328969873142),  INT64_C( 3054220899789660206) },
      UINT8_C( 26),
      { -INT64_C( 3506432804410442600), -INT64_C( 4402281636862448402) },
      {  INT64_C(  674600588288412661),  INT64_C( 5553761552512995891) },
      {  INT64_C( 8529035328969873142),  INT64_C( 4612499815685816354) } },
    { { -INT64_C(  258716122822121304), -INT64_C( 1695934070792520057) },
      UINT8_C( 32),
      { -INT64_C( 4285506606922656087), -INT64_C(  492362885293834603) },
      {  INT64_C( 3075399083708891979), -INT64_C( 9086850244571995407) },
      { -INT64_C(  258716122822121304), -INT64_C( 1695934070792520057) } },
    { {  INT64_C( 1608447529401495067), -INT64_C( 2566276007361786920) },
      UINT8_C(133),
      {  INT64_C( 1467570612966851433), -INT64_C( 5253329079940406280) },
      { -INT64_C( 4362493834358643134), -INT64_C( 6256572728081064219) },
      {  INT64_C(   24007492999578176), -INT64_C( 2566276007361786920) } },
    { { -INT64_C( 5622053946847273676), -INT64_C( 7686018027079407978) },
      UINT8_C(109),
      { -INT64_C( 8808714694754487667),  INT64_C( 8205680105901877372) },
      { -INT64_C( 6445536444375513469),  INT64_C( 1062162868406463369) },
      { -INT64_C( 8899103514309885311), -INT64_C( 7686018027079407978) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_mask_and_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_and_pd");
    easysimd_test_x86_assert_equal_i64x2(easysimd_mm_loadu_epi64(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_mask_and_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

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
test_easysimd_mm_maskz_and_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[] = {
    { UINT8_C(201),
      {  INT64_C( 1626408654053674058), -INT64_C( 2899400649694336157) },
      { -INT64_C( 6127675785709196679),  INT64_C( 5067727142908729214) },
      {  INT64_C(  185221549186877512),  INT64_C(                   0) } },
    { UINT8_C(186),
      {  INT64_C( 5287189257671674833),  INT64_C( 1299457437430047820) },
      { -INT64_C( 3800736042942153053), -INT64_C(  961221506547331903) },
      {  INT64_C(                   0),  INT64_C( 1299297432718394432) } },
    { UINT8_C( 25),
      {  INT64_C( 3574731436763573873), -INT64_C( 1081449733993977151) },
      { -INT64_C( 3591780125996957613),  INT64_C( 3557247492304082486) },
      {  INT64_C(     966476200984657),  INT64_C(                   0) } },
    { UINT8_C(207),
      {  INT64_C( 5103790160204300308), -INT64_C( 1412608369613066713) },
      {  INT64_C(  626489572828564236),  INT64_C( 7448102021003540139) },
      {  INT64_C(   40545659824930820),  INT64_C( 7225014051423135267) } },
    { UINT8_C(121),
      {  INT64_C( 9146207553939929976),  INT64_C( 3514943397870370643) },
      { -INT64_C( 1519072121259343228),  INT64_C( 5505269928422111475) },
      {  INT64_C( 7703701025353449984),  INT64_C(                   0) } },
    { UINT8_C( 32),
      {  INT64_C(   12082536892011515), -INT64_C( 1102654387712039273) },
      {  INT64_C( 6257697663972864952),  INT64_C( 2724808799787074571) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(138),
      {  INT64_C(  992351193777503296), -INT64_C( 7476632178475207697) },
      { -INT64_C( 4798358932081401458),  INT64_C( 3024462821500328184) },
      {  INT64_C(                   0),  INT64_C(  592514313653194984) } },
    { UINT8_C(  9),
      {  INT64_C( 1080086672050247299), -INT64_C( 2220190226046483721) },
      {  INT64_C(  975477875272224195),  INT64_C( 8400374327248615550) },
      {  INT64_C(  903276363024172163),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_maskz_and_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_and_pd");
    easysimd_test_x86_assert_equal_i64x2(easysimd_mm_loadu_epi64(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_maskz_and_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_and_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   260.00), EASYSIMD_FLOAT32_C(   472.07), EASYSIMD_FLOAT32_C(   343.37), EASYSIMD_FLOAT32_C(   668.63),
                         EASYSIMD_FLOAT32_C(    74.64), EASYSIMD_FLOAT32_C(  -166.33), EASYSIMD_FLOAT32_C(   962.01), EASYSIMD_FLOAT32_C(   120.25),
                         EASYSIMD_FLOAT32_C(  -633.54), EASYSIMD_FLOAT32_C(  -160.44), EASYSIMD_FLOAT32_C(  -754.35), EASYSIMD_FLOAT32_C(   920.06),
                         EASYSIMD_FLOAT32_C(  -752.65), EASYSIMD_FLOAT32_C(   -15.27), EASYSIMD_FLOAT32_C(   736.97), EASYSIMD_FLOAT32_C(   591.25)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   968.14), EASYSIMD_FLOAT32_C(   728.35), EASYSIMD_FLOAT32_C(  -574.47), EASYSIMD_FLOAT32_C(   770.03),
                         EASYSIMD_FLOAT32_C(  -456.43), EASYSIMD_FLOAT32_C(   727.04), EASYSIMD_FLOAT32_C(   -89.84), EASYSIMD_FLOAT32_C(   288.08),
                         EASYSIMD_FLOAT32_C(  -720.94), EASYSIMD_FLOAT32_C(  -964.02), EASYSIMD_FLOAT32_C(   974.54), EASYSIMD_FLOAT32_C(  -246.99),
                         EASYSIMD_FLOAT32_C(  -603.24), EASYSIMD_FLOAT32_C(  -592.85), EASYSIMD_FLOAT32_C(  -351.71), EASYSIMD_FLOAT32_C(   472.50)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     2.03), EASYSIMD_FLOAT32_C(     2.56), EASYSIMD_FLOAT32_C(     2.18), EASYSIMD_FLOAT32_C(   512.00),
                         EASYSIMD_FLOAT32_C(    66.01), EASYSIMD_FLOAT32_C(     2.57), EASYSIMD_FLOAT32_C(     2.76), EASYSIMD_FLOAT32_C(    72.00),
                         EASYSIMD_FLOAT32_C(  -592.50), EASYSIMD_FLOAT32_C(    -2.50), EASYSIMD_FLOAT32_C(   706.03), EASYSIMD_FLOAT32_C(     3.59),
                         EASYSIMD_FLOAT32_C(  -592.14), EASYSIMD_FLOAT32_C(    -2.31), EASYSIMD_FLOAT32_C(     2.63), EASYSIMD_FLOAT32_C(     2.00)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -440.60), EASYSIMD_FLOAT32_C(   215.95), EASYSIMD_FLOAT32_C(  -449.65), EASYSIMD_FLOAT32_C(   426.70),
                         EASYSIMD_FLOAT32_C(   107.08), EASYSIMD_FLOAT32_C(  -345.64), EASYSIMD_FLOAT32_C(   226.40), EASYSIMD_FLOAT32_C(   712.58),
                         EASYSIMD_FLOAT32_C(  -396.23), EASYSIMD_FLOAT32_C(  -256.01), EASYSIMD_FLOAT32_C(   622.69), EASYSIMD_FLOAT32_C(  -188.83),
                         EASYSIMD_FLOAT32_C(   358.20), EASYSIMD_FLOAT32_C(  -542.16), EASYSIMD_FLOAT32_C(   982.13), EASYSIMD_FLOAT32_C(   702.83)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   347.10), EASYSIMD_FLOAT32_C(  -175.94), EASYSIMD_FLOAT32_C(   817.30), EASYSIMD_FLOAT32_C(  -721.72),
                         EASYSIMD_FLOAT32_C(   775.39), EASYSIMD_FLOAT32_C(  -218.71), EASYSIMD_FLOAT32_C(   919.20), EASYSIMD_FLOAT32_C(  -300.97),
                         EASYSIMD_FLOAT32_C(   919.48), EASYSIMD_FLOAT32_C(   -61.84), EASYSIMD_FLOAT32_C(   121.47), EASYSIMD_FLOAT32_C(   499.98),
                         EASYSIMD_FLOAT32_C(   538.40), EASYSIMD_FLOAT32_C(  -622.49), EASYSIMD_FLOAT32_C(  -852.24), EASYSIMD_FLOAT32_C(   445.35)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   280.10), EASYSIMD_FLOAT32_C(   135.94), EASYSIMD_FLOAT32_C(     3.01), EASYSIMD_FLOAT32_C(     2.32),
                         EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(  -136.57), EASYSIMD_FLOAT32_C(     3.50), EASYSIMD_FLOAT32_C(     2.28),
                         EASYSIMD_FLOAT32_C(     3.06), EASYSIMD_FLOAT32_C(   -32.00), EASYSIMD_FLOAT32_C(     2.29), EASYSIMD_FLOAT32_C(   184.83),
                         EASYSIMD_FLOAT32_C(     2.03), EASYSIMD_FLOAT32_C(  -526.16), EASYSIMD_FLOAT32_C(   852.13), EASYSIMD_FLOAT32_C(     2.23)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -590.91), EASYSIMD_FLOAT32_C(  -663.88), EASYSIMD_FLOAT32_C(  -394.32), EASYSIMD_FLOAT32_C(  -991.87),
                         EASYSIMD_FLOAT32_C(   385.94), EASYSIMD_FLOAT32_C(  -349.46), EASYSIMD_FLOAT32_C(  -786.25), EASYSIMD_FLOAT32_C(   192.19),
                         EASYSIMD_FLOAT32_C(  -594.16), EASYSIMD_FLOAT32_C(  -602.03), EASYSIMD_FLOAT32_C(   176.16), EASYSIMD_FLOAT32_C(  -458.14),
                         EASYSIMD_FLOAT32_C(   335.26), EASYSIMD_FLOAT32_C(  -272.70), EASYSIMD_FLOAT32_C(   585.90), EASYSIMD_FLOAT32_C(  -571.61)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   530.04), EASYSIMD_FLOAT32_C(  -606.85), EASYSIMD_FLOAT32_C(   112.20), EASYSIMD_FLOAT32_C(  -437.59),
                         EASYSIMD_FLOAT32_C(  -396.36), EASYSIMD_FLOAT32_C(  -280.58), EASYSIMD_FLOAT32_C(   819.31), EASYSIMD_FLOAT32_C(  -726.73),
                         EASYSIMD_FLOAT32_C(  -263.24), EASYSIMD_FLOAT32_C(  -511.40), EASYSIMD_FLOAT32_C(  -175.25), EASYSIMD_FLOAT32_C(   728.37),
                         EASYSIMD_FLOAT32_C(   881.16), EASYSIMD_FLOAT32_C(   -49.97), EASYSIMD_FLOAT32_C(   618.76), EASYSIMD_FLOAT32_C(  -518.70)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   514.03), EASYSIMD_FLOAT32_C(  -534.75), EASYSIMD_FLOAT32_C(    96.06), EASYSIMD_FLOAT32_C(    -3.29),
                         EASYSIMD_FLOAT32_C(   384.31), EASYSIMD_FLOAT32_C(  -280.08), EASYSIMD_FLOAT32_C(   786.25), EASYSIMD_FLOAT32_C(     2.00),
                         EASYSIMD_FLOAT32_C(    -2.01), EASYSIMD_FLOAT32_C(    -2.35), EASYSIMD_FLOAT32_C(   160.00), EASYSIMD_FLOAT32_C(     2.56),
                         EASYSIMD_FLOAT32_C(     2.06), EASYSIMD_FLOAT32_C(   -32.06), EASYSIMD_FLOAT32_C(   584.76), EASYSIMD_FLOAT32_C(  -514.56)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   515.17), EASYSIMD_FLOAT32_C(   324.54), EASYSIMD_FLOAT32_C(    91.18), EASYSIMD_FLOAT32_C(  -165.19),
                         EASYSIMD_FLOAT32_C(  -882.22), EASYSIMD_FLOAT32_C(   833.89), EASYSIMD_FLOAT32_C(   476.02), EASYSIMD_FLOAT32_C(   887.60),
                         EASYSIMD_FLOAT32_C(   229.74), EASYSIMD_FLOAT32_C(   342.64), EASYSIMD_FLOAT32_C(   541.23), EASYSIMD_FLOAT32_C(  -642.89),
                         EASYSIMD_FLOAT32_C(   701.90), EASYSIMD_FLOAT32_C(   393.90), EASYSIMD_FLOAT32_C(  -103.65), EASYSIMD_FLOAT32_C(   243.25)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    21.91), EASYSIMD_FLOAT32_C(  -134.28), EASYSIMD_FLOAT32_C(   125.14), EASYSIMD_FLOAT32_C(  -667.85),
                         EASYSIMD_FLOAT32_C(  -778.80), EASYSIMD_FLOAT32_C(  -220.75), EASYSIMD_FLOAT32_C(   348.36), EASYSIMD_FLOAT32_C(    29.88),
                         EASYSIMD_FLOAT32_C(  -634.89), EASYSIMD_FLOAT32_C(  -148.88), EASYSIMD_FLOAT32_C(   827.50), EASYSIMD_FLOAT32_C(  -532.87),
                         EASYSIMD_FLOAT32_C(  -762.33), EASYSIMD_FLOAT32_C(   247.69), EASYSIMD_FLOAT32_C(  -238.64), EASYSIMD_FLOAT32_C(   244.40)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(   130.27), EASYSIMD_FLOAT32_C(    89.13), EASYSIMD_FLOAT32_C(    -2.57),
                         EASYSIMD_FLOAT32_C(  -770.03), EASYSIMD_FLOAT32_C(     3.25), EASYSIMD_FLOAT32_C(   348.02), EASYSIMD_FLOAT32_C(     3.20),
                         EASYSIMD_FLOAT32_C(     2.07), EASYSIMD_FLOAT32_C(   128.26), EASYSIMD_FLOAT32_C(   537.00), EASYSIMD_FLOAT32_C(  -512.76),
                         EASYSIMD_FLOAT32_C(   696.27), EASYSIMD_FLOAT32_C(   196.69), EASYSIMD_FLOAT32_C(   -51.50), EASYSIMD_FLOAT32_C(   240.25)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   139.73), EASYSIMD_FLOAT32_C(   808.22), EASYSIMD_FLOAT32_C(  -888.67), EASYSIMD_FLOAT32_C(   -90.81),
                         EASYSIMD_FLOAT32_C(    58.51), EASYSIMD_FLOAT32_C(  -297.55), EASYSIMD_FLOAT32_C(  -246.77), EASYSIMD_FLOAT32_C(  -391.18),
                         EASYSIMD_FLOAT32_C(   887.15), EASYSIMD_FLOAT32_C(   997.52), EASYSIMD_FLOAT32_C(   873.12), EASYSIMD_FLOAT32_C(  -969.73),
                         EASYSIMD_FLOAT32_C(   721.30), EASYSIMD_FLOAT32_C(  -128.28), EASYSIMD_FLOAT32_C(  -264.35), EASYSIMD_FLOAT32_C(  -432.42)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -80.65), EASYSIMD_FLOAT32_C(   -15.71), EASYSIMD_FLOAT32_C(    54.64), EASYSIMD_FLOAT32_C(  -420.79),
                         EASYSIMD_FLOAT32_C(  -573.45), EASYSIMD_FLOAT32_C(   578.20), EASYSIMD_FLOAT32_C(  -393.34), EASYSIMD_FLOAT32_C(   -79.47),
                         EASYSIMD_FLOAT32_C(  -837.77), EASYSIMD_FLOAT32_C(   169.23), EASYSIMD_FLOAT32_C(   110.87), EASYSIMD_FLOAT32_C(   428.31),
                         EASYSIMD_FLOAT32_C(   944.93), EASYSIMD_FLOAT32_C(   222.75), EASYSIMD_FLOAT32_C(  -792.23), EASYSIMD_FLOAT32_C(  -269.27)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    32.26), EASYSIMD_FLOAT32_C(     3.16), EASYSIMD_FLOAT32_C(     3.41), EASYSIMD_FLOAT32_C(   -72.01),
                         EASYSIMD_FLOAT32_C(     2.16), EASYSIMD_FLOAT32_C(     2.26), EASYSIMD_FLOAT32_C(  -196.50), EASYSIMD_FLOAT32_C(   -65.28),
                         EASYSIMD_FLOAT32_C(   837.02), EASYSIMD_FLOAT32_C(     2.64), EASYSIMD_FLOAT32_C(     3.38), EASYSIMD_FLOAT32_C(     3.28),
                         EASYSIMD_FLOAT32_C(   656.30), EASYSIMD_FLOAT32_C(   128.25), EASYSIMD_FLOAT32_C(    -2.06), EASYSIMD_FLOAT32_C(  -256.25)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -807.89), EASYSIMD_FLOAT32_C(  -195.99), EASYSIMD_FLOAT32_C(  -460.22), EASYSIMD_FLOAT32_C(  -214.31),
                         EASYSIMD_FLOAT32_C(  -242.49), EASYSIMD_FLOAT32_C(   293.67), EASYSIMD_FLOAT32_C(   209.36), EASYSIMD_FLOAT32_C(   -28.16),
                         EASYSIMD_FLOAT32_C(   861.78), EASYSIMD_FLOAT32_C(  -349.18), EASYSIMD_FLOAT32_C(  -840.98), EASYSIMD_FLOAT32_C(   667.88),
                         EASYSIMD_FLOAT32_C(  -431.60), EASYSIMD_FLOAT32_C(  -312.68), EASYSIMD_FLOAT32_C(   469.25), EASYSIMD_FLOAT32_C(   584.01)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   881.42), EASYSIMD_FLOAT32_C(   703.51), EASYSIMD_FLOAT32_C(   112.91), EASYSIMD_FLOAT32_C(    92.78),
                         EASYSIMD_FLOAT32_C(   506.44), EASYSIMD_FLOAT32_C(   923.94), EASYSIMD_FLOAT32_C(  -577.40), EASYSIMD_FLOAT32_C(  -437.14),
                         EASYSIMD_FLOAT32_C(  -379.29), EASYSIMD_FLOAT32_C(   791.05), EASYSIMD_FLOAT32_C(   859.09), EASYSIMD_FLOAT32_C(   612.11),
                         EASYSIMD_FLOAT32_C(   687.78), EASYSIMD_FLOAT32_C(   712.98), EASYSIMD_FLOAT32_C(  -143.15), EASYSIMD_FLOAT32_C(  -972.86)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   801.39), EASYSIMD_FLOAT32_C(     2.06), EASYSIMD_FLOAT32_C(   112.03), EASYSIMD_FLOAT32_C(    36.01),
                         EASYSIMD_FLOAT32_C(   240.22), EASYSIMD_FLOAT32_C(     2.04), EASYSIMD_FLOAT32_C(     2.26), EASYSIMD_FLOAT32_C(   -24.00),
                         EASYSIMD_FLOAT32_C(     2.33), EASYSIMD_FLOAT32_C(     2.07), EASYSIMD_FLOAT32_C(   840.07), EASYSIMD_FLOAT32_C(   512.00),
                         EASYSIMD_FLOAT32_C(     2.06), EASYSIMD_FLOAT32_C(     2.25), EASYSIMD_FLOAT32_C(   138.12), EASYSIMD_FLOAT32_C(   584.00)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   278.01), EASYSIMD_FLOAT32_C(  -815.62), EASYSIMD_FLOAT32_C(   752.91), EASYSIMD_FLOAT32_C(   710.22),
                         EASYSIMD_FLOAT32_C(  -124.40), EASYSIMD_FLOAT32_C(  -338.82), EASYSIMD_FLOAT32_C(  -853.49), EASYSIMD_FLOAT32_C(   731.62),
                         EASYSIMD_FLOAT32_C(   168.07), EASYSIMD_FLOAT32_C(  -402.61), EASYSIMD_FLOAT32_C(  -908.62), EASYSIMD_FLOAT32_C(   912.24),
                         EASYSIMD_FLOAT32_C(   241.90), EASYSIMD_FLOAT32_C(   493.82), EASYSIMD_FLOAT32_C(  -948.44), EASYSIMD_FLOAT32_C(   522.79)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -401.28), EASYSIMD_FLOAT32_C(   383.51), EASYSIMD_FLOAT32_C(  -854.57), EASYSIMD_FLOAT32_C(  -237.48),
                         EASYSIMD_FLOAT32_C(  -426.55), EASYSIMD_FLOAT32_C(  -605.26), EASYSIMD_FLOAT32_C(   140.00), EASYSIMD_FLOAT32_C(  -626.79),
                         EASYSIMD_FLOAT32_C(   473.63), EASYSIMD_FLOAT32_C(   968.53), EASYSIMD_FLOAT32_C(  -767.62), EASYSIMD_FLOAT32_C(  -339.51),
                         EASYSIMD_FLOAT32_C(   144.17), EASYSIMD_FLOAT32_C(   -47.64), EASYSIMD_FLOAT32_C(  -130.89), EASYSIMD_FLOAT32_C(   -19.38)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   272.01), EASYSIMD_FLOAT32_C(     2.18), EASYSIMD_FLOAT32_C(   592.50), EASYSIMD_FLOAT32_C(     2.52),
                         EASYSIMD_FLOAT32_C(  -104.13), EASYSIMD_FLOAT32_C(    -2.02), EASYSIMD_FLOAT32_C(     2.06), EASYSIMD_FLOAT32_C(   594.54),
                         EASYSIMD_FLOAT32_C(   168.07), EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(  -652.62), EASYSIMD_FLOAT32_C(     2.50),
                         EASYSIMD_FLOAT32_C(   144.13), EASYSIMD_FLOAT32_C(    45.63), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     2.03)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   491.53), EASYSIMD_FLOAT32_C(   985.75), EASYSIMD_FLOAT32_C(  -390.64), EASYSIMD_FLOAT32_C(   517.90),
                         EASYSIMD_FLOAT32_C(  -725.16), EASYSIMD_FLOAT32_C(     9.87), EASYSIMD_FLOAT32_C(   943.82), EASYSIMD_FLOAT32_C(   279.49),
                         EASYSIMD_FLOAT32_C(  -942.01), EASYSIMD_FLOAT32_C(    63.94), EASYSIMD_FLOAT32_C(   920.28), EASYSIMD_FLOAT32_C(   132.72),
                         EASYSIMD_FLOAT32_C(   502.41), EASYSIMD_FLOAT32_C(   855.02), EASYSIMD_FLOAT32_C(   610.59), EASYSIMD_FLOAT32_C(   860.61)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -675.66), EASYSIMD_FLOAT32_C(   128.22), EASYSIMD_FLOAT32_C(  -915.29), EASYSIMD_FLOAT32_C(  -679.65),
                         EASYSIMD_FLOAT32_C(   537.51), EASYSIMD_FLOAT32_C(  -484.11), EASYSIMD_FLOAT32_C(   502.40), EASYSIMD_FLOAT32_C(  -785.39),
                         EASYSIMD_FLOAT32_C(  -128.17), EASYSIMD_FLOAT32_C(   101.31), EASYSIMD_FLOAT32_C(  -990.73), EASYSIMD_FLOAT32_C(  -514.82),
                         EASYSIMD_FLOAT32_C(   231.21), EASYSIMD_FLOAT32_C(   964.21), EASYSIMD_FLOAT32_C(  -258.81), EASYSIMD_FLOAT32_C(   355.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     2.51), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(    -3.00), EASYSIMD_FLOAT32_C(   517.65),
                         EASYSIMD_FLOAT32_C(   529.00), EASYSIMD_FLOAT32_C(     9.00), EASYSIMD_FLOAT32_C(     3.67), EASYSIMD_FLOAT32_C(     2.00),
                         EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(    50.63), EASYSIMD_FLOAT32_C(   920.01), EASYSIMD_FLOAT32_C(     2.01),
                         EASYSIMD_FLOAT32_C(   227.20), EASYSIMD_FLOAT32_C(   836.02), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     2.27)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_and_ps(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_and_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_and_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[8];
    uint8_t k;
    int32_t a[8];
    int32_t b[8];
    int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1672621367),  INT32_C(   220170497), -INT32_C(  1809398325),  INT32_C(   768475049),  INT32_C(  1470942234),  INT32_C(   800285189), -INT32_C(   177900051),  INT32_C(  1515333265) },
      UINT8_C(248),
      {  INT32_C(   687470239), -INT32_C(   604830186),  INT32_C(   746882861),  INT32_C(   541503828), -INT32_C(  1088053667), -INT32_C(   961784751), -INT32_C(   531062086), -INT32_C(  1814514956) },
      { -INT32_C(  1078210135),  INT32_C(    77246167), -INT32_C(  1993269707),  INT32_C(   766080975),  INT32_C(  1726795541), -INT32_C(   584214237),  INT32_C(   800949563), -INT32_C(   524118729) },
      { -INT32_C(  1672621367),  INT32_C(   220170497), -INT32_C(  1809398325),  INT32_C(   536883524),  INT32_C(   639929877), -INT32_C(  1003745279),  INT32_C(   538476602), -INT32_C(  2134863820) } },
    { {  INT32_C(  1050639718),  INT32_C(  1614952747),  INT32_C(   669610839),  INT32_C(     5542635), -INT32_C(  2056830879),  INT32_C(   342004953),  INT32_C(  1346576409),  INT32_C(   472974773) },
      UINT8_C(130),
      {  INT32_C(   179133136),  INT32_C(   274795932), -INT32_C(  1980004106),  INT32_C(   501939164), -INT32_C(   151621790), -INT32_C(   233894958),  INT32_C(  1420255310),  INT32_C(  1641464721) },
      { -INT32_C(  1167358947), -INT32_C(  1999975023),  INT32_C(   840025429),  INT32_C(   592444352),  INT32_C(  1025066347), -INT32_C(  1624299184),  INT32_C(   452187785), -INT32_C(  1216624230) },
      {  INT32_C(  1050639718),  INT32_C(     4197776),  INT32_C(   669610839),  INT32_C(     5542635), -INT32_C(  2056830879),  INT32_C(   342004953),  INT32_C(  1346576409),  INT32_C(   559071632) } },
    { { -INT32_C(   546183347),  INT32_C(   157760436), -INT32_C(  1053067264), -INT32_C(   555447693),  INT32_C(   538705360), -INT32_C(  1346418138), -INT32_C(  1144409567), -INT32_C(   915257988) },
      UINT8_C( 44),
      {  INT32_C(   534816996), -INT32_C(  2027886321), -INT32_C(  1342447323), -INT32_C(  1031808571),  INT32_C(  1072275701),  INT32_C(   308320351), -INT32_C(  1483859102),  INT32_C(  1926453134) },
      {  INT32_C(   261206784), -INT32_C(  1047023204),  INT32_C(  1467060882),  INT32_C(  1612378219), -INT32_C(   257948784), -INT32_C(    50200421), -INT32_C(  1432055780), -INT32_C(   417564697) },
      { -INT32_C(   546183347),  INT32_C(   157760436),  INT32_C(   124878848),  INT32_C(  1075499073),  INT32_C(   538705360),  INT32_C(   268435483), -INT32_C(  1144409567), -INT32_C(   915257988) } },
    { { -INT32_C(   940069590), -INT32_C(   242708897), -INT32_C(  1958086368),  INT32_C(  2062312426),  INT32_C(    23759974), -INT32_C(  1459655540), -INT32_C(   464346116),  INT32_C(  1170959899) },
      UINT8_C( 28),
      {  INT32_C(  1350241474), -INT32_C(  1905234795),  INT32_C(   410582197),  INT32_C(  1954477032), -INT32_C(   905936803),  INT32_C(   583444863),  INT32_C(  1782426363),  INT32_C(   948339574) },
      {  INT32_C(   612958607),  INT32_C(   598997357), -INT32_C(   583324683),  INT32_C(  2085730846), -INT32_C(  1186573766), -INT32_C(   153416453),  INT32_C(   761272759),  INT32_C(   728098460) },
      { -INT32_C(   940069590), -INT32_C(   242708897),  INT32_C(   406333621),  INT32_C(  1951445512), -INT32_C(  2013265896), -INT32_C(  1459655540), -INT32_C(   464346116),  INT32_C(  1170959899) } },
    { {  INT32_C(  1431367399), -INT32_C(   579337240),  INT32_C(  1304146734), -INT32_C(  1479996307),  INT32_C(  1499467614), -INT32_C(   766493669), -INT32_C(   234901419),  INT32_C(  2115790231) },
      UINT8_C( 83),
      {  INT32_C(  1866191724), -INT32_C(     6416053), -INT32_C(   529732652), -INT32_C(  1019276108), -INT32_C(  1327589260), -INT32_C(  1727680024),  INT32_C(   355530416),  INT32_C(  2137632275) },
      { -INT32_C(   839932798), -INT32_C(  1848865347),  INT32_C(   745683320), -INT32_C(  1041256115),  INT32_C(   846384457),  INT32_C(   801863550), -INT32_C(  2126185618),  INT32_C(   721464745) },
      {  INT32_C(  1294696448), -INT32_C(  1853093623),  INT32_C(  1304146734), -INT32_C(  1479996307),  INT32_C(   810715200), -INT32_C(   766493669),  INT32_C(    16839200),  INT32_C(  2115790231) } },
    { {  INT32_C(   268037970), -INT32_C(   190724740),  INT32_C(  1260393470),  INT32_C(   218959812),  INT32_C(  1530888157),  INT32_C(  1686768374), -INT32_C(  1343893755), -INT32_C(   824514948) },
      UINT8_C(213),
      { -INT32_C(  1722622253), -INT32_C(  1835579777),  INT32_C(  1985405799),  INT32_C(  1867736048), -INT32_C(  1385844829),  INT32_C(   146000441), -INT32_C(  1786420561),  INT32_C(   275469116) },
      { -INT32_C(  1331053263),  INT32_C(  1782726659),  INT32_C(   333551651), -INT32_C(  1618856708),  INT32_C(   491579619),  INT32_C(  1629880242), -INT32_C(  1627936159),  INT32_C(   799957758) },
      { -INT32_C(  1879008239), -INT32_C(   190724740),  INT32_C(   306216995),  INT32_C(   218959812),  INT32_C(   222603427),  INT32_C(  1686768374), -INT32_C(  1803214303),  INT32_C(     2769468) } },
    { {  INT32_C(   585127711), -INT32_C(  1148378473), -INT32_C(  1211208005), -INT32_C(  2041163358), -INT32_C(   341597639),  INT32_C(    55363746),  INT32_C(  1906393971), -INT32_C(   996126811) },
      UINT8_C(166),
      { -INT32_C(  1556158592), -INT32_C(   497092236),  INT32_C(   411309511),  INT32_C(   223414891),  INT32_C(  1991195821), -INT32_C(   857099383), -INT32_C(  1536009644),  INT32_C(  2068461306) },
      { -INT32_C(  1860270051),  INT32_C(  1232305281), -INT32_C(    60688239), -INT32_C(  1341541630),  INT32_C(  2015803887), -INT32_C(  1069215892),  INT32_C(  1667544937),  INT32_C(   199143405) },
      {  INT32_C(   585127711),  INT32_C(  1079146496),  INT32_C(   402657409), -INT32_C(  2041163358), -INT32_C(   341597639), -INT32_C(  1069481208),  INT32_C(  1906393971),  INT32_C(   189409000) } },
    { { -INT32_C(  1180894153),  INT32_C(   151130232),  INT32_C(   168125192),  INT32_C(   112856854),  INT32_C(   880730312),  INT32_C(  1492435951),  INT32_C(  1757174138),  INT32_C(  1064540680) },
      UINT8_C(151),
      {  INT32_C(   521140239),  INT32_C(  1579620858),  INT32_C(   762589726), -INT32_C(   839550228), -INT32_C(  1128519175),  INT32_C(  2000098590),  INT32_C(  1803526097),  INT32_C(   553827858) },
      { -INT32_C(  1321135433),  INT32_C(  1225746475), -INT32_C(  2022210406), -INT32_C(   145462018), -INT32_C(  1263267690), -INT32_C(   164893915), -INT32_C(  1671255414),  INT32_C(   532505704) },
      {  INT32_C(   285282311),  INT32_C(  1208420394),  INT32_C(    91488282),  INT32_C(   112856854), -INT32_C(  1263271792),  INT32_C(  1492435951),  INT32_C(  1757174138),  INT32_C(    16786432) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_and_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_and_epi32");
    easysimd_test_x86_assert_equal_i32x8(easysimd_mm256_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_and_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_and_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[8];
    int32_t b[8];
    int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 81),
      {  INT32_C(   372482748),  INT32_C(   660824903),  INT32_C(  1839478450),  INT32_C(  1331679876), -INT32_C(  1997729534),  INT32_C(   539417629),  INT32_C(   908301385),  INT32_C(   898068345) },
      {  INT32_C(  1481423377), -INT32_C(   796938467),  INT32_C(  1765614565), -INT32_C(   122118666), -INT32_C(   863983954), -INT32_C(   873683326), -INT32_C(  1274998981), -INT32_C(  1863677825) },
      {  INT32_C(   268476944),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2004877310),  INT32_C(           0),  INT32_C(   872481801),  INT32_C(           0) } },
    { UINT8_C( 66),
      { -INT32_C(   446699210), -INT32_C(  1932906648),  INT32_C(   176305005), -INT32_C(  1850180885), -INT32_C(  1609333510), -INT32_C(  2133074319),  INT32_C(  1744802015), -INT32_C(  1331064710) },
      { -INT32_C(   510326407), -INT32_C(  1519558856),  INT32_C(  2108682130),  INT32_C(  1661954153),  INT32_C(  1560552172), -INT32_C(   522264831), -INT32_C(   381100689), -INT32_C(   426118803) },
      {  INT32_C(           0), -INT32_C(  2075652312),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1632145487),  INT32_C(           0) } },
    { UINT8_C(250),
      { -INT32_C(  1925986258),  INT32_C(   622843957), -INT32_C(   259089016), -INT32_C(   824380756), -INT32_C(   707839242), -INT32_C(   179982312), -INT32_C(   379375880), -INT32_C(   152811064) },
      {  INT32_C(  1199838993),  INT32_C(  2003608559), -INT32_C(   328730048), -INT32_C(   491043604), -INT32_C(  1766290818),  INT32_C(   848035130), -INT32_C(   216207829),  INT32_C(  1240072248) },
      {  INT32_C(           0),  INT32_C(   621576229),  INT32_C(           0), -INT32_C(  1030209364), -INT32_C(  1803023754),  INT32_C(   805416984), -INT32_C(   520081880),  INT32_C(  1088421896) } },
    { UINT8_C( 23),
      {  INT32_C(   285642862), -INT32_C(   145588484),  INT32_C(   702758629), -INT32_C(  2069379335),  INT32_C(  2076065150), -INT32_C(  1213795895),  INT32_C(   233806349), -INT32_C(   232507004) },
      { -INT32_C(   972870967), -INT32_C(  1900194392), -INT32_C(  1917345644), -INT32_C(   468623514),  INT32_C(  1700778140), -INT32_C(   836958527),  INT32_C(   618335392),  INT32_C(   253165381) },
      {  INT32_C(      131144), -INT32_C(  2045750104),  INT32_C(   161685636),  INT32_C(           0),  INT32_C(  1629360156),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 41),
      {  INT32_C(  1876088089),  INT32_C(   839082130),  INT32_C(  1989775383),  INT32_C(  1913879970), -INT32_C(   483165987), -INT32_C(  1568472683), -INT32_C(   605575204), -INT32_C(   704317763) },
      {  INT32_C(  1564858059),  INT32_C(  1318078518),  INT32_C(  2076453337), -INT32_C(  2081564762), -INT32_C(   446291888), -INT32_C(    41424607),  INT32_C(  1306029968),  INT32_C(   824499301) },
      {  INT32_C(  1296094217),  INT32_C(           0),  INT32_C(           0),  INT32_C(    33641890),  INT32_C(           0), -INT32_C(  1602027263),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(179),
      { -INT32_C(  1293316503),  INT32_C(  1200305950), -INT32_C(   739375364),  INT32_C(   321090035), -INT32_C(  1053554216), -INT32_C(    11390576), -INT32_C(   446324983),  INT32_C(   764974787) },
      {  INT32_C(  1138721316), -INT32_C(  1249219911),  INT32_C(  1669953648), -INT32_C(  1032409878),  INT32_C(  1183034037), -INT32_C(   465185317),  INT32_C(   952740469),  INT32_C(  1701143104) },
      {  INT32_C(    46760480),  INT32_C(    92938776),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1073744016), -INT32_C(   465563248),  INT32_C(           0),  INT32_C(   620757568) } },
    { UINT8_C(228),
      { -INT32_C(  1365399484), -INT32_C(  1424075982), -INT32_C(  2003467812), -INT32_C(  1589749769), -INT32_C(  1334016806),  INT32_C(  1948606665), -INT32_C(  1934336726),  INT32_C(   124787139) },
      { -INT32_C(   189395519),  INT32_C(  1017107552),  INT32_C(  1304769622),  INT32_C(  1710162827),  INT32_C(  1343581063), -INT32_C(   154912052),  INT32_C(  1535277464),  INT32_C(  1415770770) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   142934100),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1946427592),  INT32_C(   142629128),  INT32_C(    73404546) } },
    {    UINT8_MAX,
      { -INT32_C(   312457192),  INT32_C(   457415911),  INT32_C(  1688637537), -INT32_C(   336917633),  INT32_C(  1538735137),  INT32_C(  2046012672),  INT32_C(   554454575), -INT32_C(   920624975) },
      { -INT32_C(  1883864920),  INT32_C(  2125134109),  INT32_C(   165892234),  INT32_C(  2096418395),  INT32_C(   198748938), -INT32_C(  2021340328), -INT32_C(   878145510), -INT32_C(  1735079440) },
      { -INT32_C(  1927282680),  INT32_C(   436377605),  INT32_C(    10620928),  INT32_C(  1759513179),  INT32_C(   193996800),  INT32_C(    25200896),  INT32_C(    17301514), -INT32_C(  2013249360) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_and_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_and_epi32");
    easysimd_test_x86_assert_equal_i32x8(easysimd_mm256_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_and_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_and_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[4];
    uint8_t k;
    int64_t a[4];
    int64_t b[4];
    int64_t r[4];
  } test_vec[] = {
    { {  INT64_C(  138450949266079350),  INT64_C( 7231334930688008804), -INT64_C( 9004591242921111827),  INT64_C( 5077208037502970861) },
      UINT8_C(219),
      {  INT64_C( 3786195581127550790),  INT64_C( 6311345493872209612),  INT64_C( 2067667563890976896), -INT64_C( 4347044028640131084) },
      { -INT64_C(   62921528904585447),  INT64_C(  793828203958903652),  INT64_C( 2212814612851480039), -INT64_C( 1385248504763407994) },
      {  INT64_C( 3747069734680717056),  INT64_C(  217340504952545860), -INT64_C( 9004591242921111827), -INT64_C( 4574511186662653564) } },
    { { -INT64_C( 8521380979420909430), -INT64_C( 8380570378217036516), -INT64_C( 8657932785985289840), -INT64_C( 1352961825631975559) },
      UINT8_C(  0),
      { -INT64_C( 3663766910460908783),  INT64_C(  194213890478952165),  INT64_C(  824872611073911584),  INT64_C(  676486586630585875) },
      {  INT64_C( 2106622739758680327), -INT64_C( 3851440065565774327),  INT64_C( 3146385669530816090),  INT64_C( 8932030544964688115) },
      { -INT64_C( 8521380979420909430), -INT64_C( 8380570378217036516), -INT64_C( 8657932785985289840), -INT64_C( 1352961825631975559) } },
    { {  INT64_C( 5463190990416906107),  INT64_C( 6362005169308504021),  INT64_C( 3833585012795969007),  INT64_C( 3506356548985209529) },
      UINT8_C( 28),
      { -INT64_C( 4354794245964504172),  INT64_C( 7407887410379427710), -INT64_C( 3788690528268659183), -INT64_C( 2940939329024845314) },
      { -INT64_C( 5473243305614340717), -INT64_C(  580575234141991887),  INT64_C( 8505915927444429367), -INT64_C( 3618439737292256338) },
      {  INT64_C( 5463190990416906107),  INT64_C( 6362005169308504021),  INT64_C( 4758919455850402321), -INT64_C( 4248961521594824274) } },
    { { -INT64_C( 5895457668191659451),  INT64_C( 6614145340397961964),  INT64_C( 4013735900786488516),  INT64_C( 4195932039249738802) },
      UINT8_C( 76),
      { -INT64_C( 6315041849685264817),  INT64_C( 7898794750525583279), -INT64_C( 3846867317472961597), -INT64_C( 7740286684062873001) },
      { -INT64_C( 8989922819307184720), -INT64_C(  795172094024545905),  INT64_C( 8156015739986242774), -INT64_C( 7430571150557909497) },
      { -INT64_C( 5895457668191659451),  INT64_C( 6614145340397961964),  INT64_C( 4615393616202221762), -INT64_C( 8034340095270450681) } },
    { { -INT64_C( 3169853241484712533),  INT64_C( 3574259803563515765), -INT64_C(   86759201164625345),  INT64_C( 9166000428960941491) },
      UINT8_C( 17),
      {  INT64_C( 6864094964999454306),  INT64_C( 9014212506196502662), -INT64_C( 4469978501969841792),  INT64_C( 3024752948582650813) },
      {  INT64_C(  103927462873157650), -INT64_C( 7700607713052841439), -INT64_C( 4223865140508256925),  INT64_C( 2041442392253396580) },
      {  INT64_C(   90115973018177538),  INT64_C( 3574259803563515765), -INT64_C(   86759201164625345),  INT64_C( 9166000428960941491) } },
    { { -INT64_C( 6524133098237572358), -INT64_C( 6524472029489836086), -INT64_C( 9021090355658535680),  INT64_C( 2361013370182532509) },
      UINT8_C( 76),
      {  INT64_C( 6511206750032467442), -INT64_C(  276618781045081154),  INT64_C( 9063948349125374597),  INT64_C( 2853246092842332235) },
      { -INT64_C( 7296651784960251815),  INT64_C( 2304696880006912336), -INT64_C(   28124733819382241),  INT64_C( 6692599750917308292) },
      { -INT64_C( 6524133098237572358), -INT64_C( 6524472029489836086),  INT64_C( 9045501842008248837),  INT64_C(  324474677584146432) } },
    { { -INT64_C(   89089380655785652), -INT64_C( 7885811247801858346),  INT64_C( 8269257553686419983),  INT64_C( 4462235937927794061) },
      UINT8_C(164),
      { -INT64_C( 5836295504361527635), -INT64_C( 2714064187639500467), -INT64_C( 1783079266451711897),  INT64_C( 7788437617088516710) },
      {  INT64_C(  456437760850815100), -INT64_C( 7718429596574779384),  INT64_C( 5439093693603228345), -INT64_C( 1763209375918566308) },
      { -INT64_C(   89089380655785652), -INT64_C( 7885811247801858346),  INT64_C( 4846166496198002721),  INT64_C( 4462235937927794061) } },
    { { -INT64_C( 4685184290717426117), -INT64_C( 6765627199018612903),  INT64_C( 5441387316280386643),  INT64_C( 5572903351863427076) },
      UINT8_C( 12),
      { -INT64_C( 8998895481928924702),  INT64_C( 2612947331106710598),  INT64_C( 4609539771513142083), -INT64_C( 6935086314050983994) },
      {  INT64_C( 5305100736426902496),  INT64_C( 6560192253585328933),  INT64_C( 4770797982302805495),  INT64_C(  435362789154326446) },
      { -INT64_C( 4685184290717426117), -INT64_C( 6765627199018612903),  INT64_C(  157704588919773507),  INT64_C(  432512845155993478) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_and_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_and_epi64");
    easysimd_test_x86_assert_equal_i64x4(easysimd_mm256_loadu_epi64(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_mask_and_epi64(src, k, a, b);

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
test_easysimd_mm256_maskz_and_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[4];
    int64_t b[4];
    int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 73),
      {  INT64_C( 2436137135412488011),  INT64_C( 8692503412895242440), -INT64_C( 4488021537822036783),  INT64_C( 6995016107970101760) },
      {  INT64_C( 2766464019323058545), -INT64_C( 6011919732440178356), -INT64_C( 4973380567953743606),  INT64_C( 7564524514298801538) },
      {  INT64_C( 2325093603466420545),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 6922597873041375232) } },
    { UINT8_C( 16),
      { -INT64_C( 1053396941516395616),  INT64_C( 7906765186920118866), -INT64_C( 7114032711690947390),  INT64_C( 8722061641498549860) },
      {  INT64_C( 6965362244721736104), -INT64_C( 5628431979456203158),  INT64_C( 5641808475323632062), -INT64_C( 8868902834917548136) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(204),
      {  INT64_C( 6616978509511933049),  INT64_C( 3950497093701487178), -INT64_C( 5997828455568211600), -INT64_C( 7361683213430278985) },
      {  INT64_C( 3785793263047454792),  INT64_C( 4648870193010365216), -INT64_C( 3885431017134447461), -INT64_C( 1145415514502947742) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 8646862905366511600), -INT64_C( 8065375152205982686) } },
    { UINT8_C( 90),
      {  INT64_C( 8796788630602652957), -INT64_C( 8469430428920880704), -INT64_C( 8177578208730947852),  INT64_C( 4341230484239811899) },
      { -INT64_C( 4943750642189434045),  INT64_C( 3104399851334774181),  INT64_C( 6082621581212562153),  INT64_C(  662443411801481173) },
      {  INT64_C(                   0),  INT64_C(  726215496747262336),  INT64_C(                   0),  INT64_C(  590288235837523217) } },
    { UINT8_C(251),
      {  INT64_C(  309254514432996193), -INT64_C( 7189039223578831546),  INT64_C( 2330623254734378931), -INT64_C( 2398749002731008958) },
      { -INT64_C( 6683959376467481631), -INT64_C( 7914459658415933277),  INT64_C( 3235957183117385799), -INT64_C( 7127618503038284359) },
      {  INT64_C(    2392898624725857), -INT64_C( 8058592531233812478),  INT64_C(                   0), -INT64_C( 7199694239102120960) } },
    { UINT8_C( 71),
      { -INT64_C( 8427845604001655143), -INT64_C( 6650412833451632580),  INT64_C( 3546787448313642001),  INT64_C( 2126415374129655711) },
      { -INT64_C( 6197076861969488287),  INT64_C( 8686821799630367271),  INT64_C( 8204202628531755590), -INT64_C( 5386534590906316387) },
      { -INT64_C( 8572031238183582207),  INT64_C( 2343217893677860900),  INT64_C( 3537621763355842560),  INT64_C(                   0) } },
    { UINT8_C( 85),
      { -INT64_C( 7891876476553582140),  INT64_C(  905911760542328194), -INT64_C( 2219342124822872985),  INT64_C( 6449694067783980541) },
      {  INT64_C( 3732963683788932115), -INT64_C( 6950723041043101813), -INT64_C( 5239821917129184945),  INT64_C( 8839550563573267902) },
      {  INT64_C( 1317905561144000512),  INT64_C(                   0), -INT64_C( 6845400961677711289),  INT64_C(                   0) } },
    { UINT8_C(169),
      {  INT64_C( 5335232304490657112),  INT64_C( 8966422848663095309), -INT64_C( 7241808959458847145), -INT64_C( 2891737024933211440) },
      { -INT64_C( 3170708110866761679), -INT64_C(  326121037249674519),  INT64_C( 7922178766472523576), -INT64_C( 6889240313380801779) },
      {  INT64_C( 4758617610727338000),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 9204231261629787648) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_and_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_and_epi64");
    easysimd_test_x86_assert_equal_i64x4(easysimd_mm256_loadu_epi64(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_and_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_and_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float32 src[8];
    uint8_t k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   137.33), EASYSIMD_FLOAT32_C(   949.65), EASYSIMD_FLOAT32_C(   219.01), EASYSIMD_FLOAT32_C(   696.79),
        EASYSIMD_FLOAT32_C(   284.93), EASYSIMD_FLOAT32_C(   184.14), EASYSIMD_FLOAT32_C(  -209.45), EASYSIMD_FLOAT32_C(   489.98) },
      UINT8_C(164),
      { EASYSIMD_FLOAT32_C(   315.01), EASYSIMD_FLOAT32_C(  -904.16), EASYSIMD_FLOAT32_C(   194.33), EASYSIMD_FLOAT32_C(   960.59),
        EASYSIMD_FLOAT32_C(   602.99), EASYSIMD_FLOAT32_C(  -846.68), EASYSIMD_FLOAT32_C(   -33.98), EASYSIMD_FLOAT32_C(   828.87) },
      { EASYSIMD_FLOAT32_C(   -88.55), EASYSIMD_FLOAT32_C(  -766.89), EASYSIMD_FLOAT32_C(   819.58), EASYSIMD_FLOAT32_C(  -629.73),
        EASYSIMD_FLOAT32_C(   775.48), EASYSIMD_FLOAT32_C(   272.08), EASYSIMD_FLOAT32_C(    99.33), EASYSIMD_FLOAT32_C(   -99.19) },
      { EASYSIMD_FLOAT32_C(   137.33), EASYSIMD_FLOAT32_C(   949.65), EASYSIMD_FLOAT32_C(     3.00), EASYSIMD_FLOAT32_C(   696.79),
        EASYSIMD_FLOAT32_C(   284.93), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(  -209.45), EASYSIMD_FLOAT32_C(     3.10) } },
    { { EASYSIMD_FLOAT32_C(   371.45), EASYSIMD_FLOAT32_C(   809.45), EASYSIMD_FLOAT32_C(  -158.91), EASYSIMD_FLOAT32_C(  -784.04),
        EASYSIMD_FLOAT32_C(  -328.65), EASYSIMD_FLOAT32_C(  -790.62), EASYSIMD_FLOAT32_C(   353.29), EASYSIMD_FLOAT32_C(  -379.00) },
      UINT8_C( 42),
      { EASYSIMD_FLOAT32_C(    50.08), EASYSIMD_FLOAT32_C(   905.93), EASYSIMD_FLOAT32_C(  -387.47), EASYSIMD_FLOAT32_C(   840.63),
        EASYSIMD_FLOAT32_C(   395.92), EASYSIMD_FLOAT32_C(   457.18), EASYSIMD_FLOAT32_C(   155.64), EASYSIMD_FLOAT32_C(   491.76) },
      { EASYSIMD_FLOAT32_C(  -348.49), EASYSIMD_FLOAT32_C(   116.23), EASYSIMD_FLOAT32_C(    94.75), EASYSIMD_FLOAT32_C(  -195.18),
        EASYSIMD_FLOAT32_C(  -917.75), EASYSIMD_FLOAT32_C(   -76.38), EASYSIMD_FLOAT32_C(   716.27), EASYSIMD_FLOAT32_C(  -684.64) },
      { EASYSIMD_FLOAT32_C(   371.45), EASYSIMD_FLOAT32_C(     3.51), EASYSIMD_FLOAT32_C(  -158.91), EASYSIMD_FLOAT32_C(     3.03),
        EASYSIMD_FLOAT32_C(  -328.65), EASYSIMD_FLOAT32_C(    64.25), EASYSIMD_FLOAT32_C(   353.29), EASYSIMD_FLOAT32_C(  -379.00) } },
    { { EASYSIMD_FLOAT32_C(  -256.79), EASYSIMD_FLOAT32_C(  -913.46), EASYSIMD_FLOAT32_C(  -909.16), EASYSIMD_FLOAT32_C(  -984.71),
        EASYSIMD_FLOAT32_C(   185.87), EASYSIMD_FLOAT32_C(    -8.35), EASYSIMD_FLOAT32_C(   386.73), EASYSIMD_FLOAT32_C(    -4.68) },
      UINT8_C(107),
      { EASYSIMD_FLOAT32_C(   602.70), EASYSIMD_FLOAT32_C(   666.66), EASYSIMD_FLOAT32_C(  -957.88), EASYSIMD_FLOAT32_C(   -44.01),
        EASYSIMD_FLOAT32_C(  -712.33), EASYSIMD_FLOAT32_C(   470.51), EASYSIMD_FLOAT32_C(  -993.93), EASYSIMD_FLOAT32_C(  -806.40) },
      { EASYSIMD_FLOAT32_C(  -916.95), EASYSIMD_FLOAT32_C(   846.70), EASYSIMD_FLOAT32_C(   589.52), EASYSIMD_FLOAT32_C(   540.23),
        EASYSIMD_FLOAT32_C(     2.34), EASYSIMD_FLOAT32_C(    81.28), EASYSIMD_FLOAT32_C(  -808.27), EASYSIMD_FLOAT32_C(  -881.43) },
      { EASYSIMD_FLOAT32_C(   528.70), EASYSIMD_FLOAT32_C(   522.63), EASYSIMD_FLOAT32_C(  -909.16), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(   185.87), EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C(  -800.27), EASYSIMD_FLOAT32_C(    -4.68) } },
    { { EASYSIMD_FLOAT32_C(  -823.96), EASYSIMD_FLOAT32_C(    -3.44), EASYSIMD_FLOAT32_C(  -799.18), EASYSIMD_FLOAT32_C(    99.66),
        EASYSIMD_FLOAT32_C(  -287.17), EASYSIMD_FLOAT32_C(  -483.82), EASYSIMD_FLOAT32_C(   842.87), EASYSIMD_FLOAT32_C(  -200.64) },
      UINT8_C(  9),
      { EASYSIMD_FLOAT32_C(   858.16), EASYSIMD_FLOAT32_C(   985.23), EASYSIMD_FLOAT32_C(   598.67), EASYSIMD_FLOAT32_C(   244.89),
        EASYSIMD_FLOAT32_C(   -19.45), EASYSIMD_FLOAT32_C(   431.42), EASYSIMD_FLOAT32_C(  -152.41), EASYSIMD_FLOAT32_C(  -352.79) },
      { EASYSIMD_FLOAT32_C(   473.54), EASYSIMD_FLOAT32_C(   803.58), EASYSIMD_FLOAT32_C(   -65.12), EASYSIMD_FLOAT32_C(   -55.95),
        EASYSIMD_FLOAT32_C(   809.65), EASYSIMD_FLOAT32_C(   128.48), EASYSIMD_FLOAT32_C(    27.10), EASYSIMD_FLOAT32_C(   656.35) },
      { EASYSIMD_FLOAT32_C(     3.07), EASYSIMD_FLOAT32_C(    -3.44), EASYSIMD_FLOAT32_C(  -799.18), EASYSIMD_FLOAT32_C(    53.19),
        EASYSIMD_FLOAT32_C(  -287.17), EASYSIMD_FLOAT32_C(  -483.82), EASYSIMD_FLOAT32_C(   842.87), EASYSIMD_FLOAT32_C(  -200.64) } },
    { { EASYSIMD_FLOAT32_C(  -282.00), EASYSIMD_FLOAT32_C(  -432.67), EASYSIMD_FLOAT32_C(  -341.31), EASYSIMD_FLOAT32_C(   799.28),
        EASYSIMD_FLOAT32_C(  -240.94), EASYSIMD_FLOAT32_C(  -222.73), EASYSIMD_FLOAT32_C(   975.32), EASYSIMD_FLOAT32_C(   755.62) },
      UINT8_C(238),
      { EASYSIMD_FLOAT32_C(    74.98), EASYSIMD_FLOAT32_C(  -531.56), EASYSIMD_FLOAT32_C(   494.27), EASYSIMD_FLOAT32_C(   -82.15),
        EASYSIMD_FLOAT32_C(   267.81), EASYSIMD_FLOAT32_C(  -898.70), EASYSIMD_FLOAT32_C(  -223.99), EASYSIMD_FLOAT32_C(   253.04) },
      { EASYSIMD_FLOAT32_C(   699.97), EASYSIMD_FLOAT32_C(  -979.09), EASYSIMD_FLOAT32_C(  -766.41), EASYSIMD_FLOAT32_C(   131.39),
        EASYSIMD_FLOAT32_C(  -131.51), EASYSIMD_FLOAT32_C(  -119.20), EASYSIMD_FLOAT32_C(  -395.07), EASYSIMD_FLOAT32_C(  -327.93) },
      { EASYSIMD_FLOAT32_C(  -282.00), EASYSIMD_FLOAT32_C(  -531.03), EASYSIMD_FLOAT32_C(     2.86), EASYSIMD_FLOAT32_C(    32.06),
        EASYSIMD_FLOAT32_C(  -240.94), EASYSIMD_FLOAT32_C(    -3.50), EASYSIMD_FLOAT32_C(  -197.53), EASYSIMD_FLOAT32_C(   161.01) } },
    { { EASYSIMD_FLOAT32_C(   815.68), EASYSIMD_FLOAT32_C(   548.98), EASYSIMD_FLOAT32_C(  -518.28), EASYSIMD_FLOAT32_C(   -55.84),
        EASYSIMD_FLOAT32_C(  -423.92), EASYSIMD_FLOAT32_C(  -861.93), EASYSIMD_FLOAT32_C(   662.17), EASYSIMD_FLOAT32_C(   143.40) },
      UINT8_C( 77),
      { EASYSIMD_FLOAT32_C(   461.45), EASYSIMD_FLOAT32_C(   902.47), EASYSIMD_FLOAT32_C(   574.03), EASYSIMD_FLOAT32_C(   436.77),
        EASYSIMD_FLOAT32_C(   658.08), EASYSIMD_FLOAT32_C(  -447.88), EASYSIMD_FLOAT32_C(  -488.25), EASYSIMD_FLOAT32_C(  -873.47) },
      { EASYSIMD_FLOAT32_C(  -953.60), EASYSIMD_FLOAT32_C(   429.61), EASYSIMD_FLOAT32_C(   394.34), EASYSIMD_FLOAT32_C(  -852.31),
        EASYSIMD_FLOAT32_C(  -794.38), EASYSIMD_FLOAT32_C(  -352.62), EASYSIMD_FLOAT32_C(   847.66), EASYSIMD_FLOAT32_C(  -773.48) },
      { EASYSIMD_FLOAT32_C(     3.60), EASYSIMD_FLOAT32_C(   548.98), EASYSIMD_FLOAT32_C(     2.08), EASYSIMD_FLOAT32_C(     3.25),
        EASYSIMD_FLOAT32_C(  -423.92), EASYSIMD_FLOAT32_C(  -861.93), EASYSIMD_FLOAT32_C(     3.25), EASYSIMD_FLOAT32_C(   143.40) } },
    { { EASYSIMD_FLOAT32_C(  -119.03), EASYSIMD_FLOAT32_C(   -20.95), EASYSIMD_FLOAT32_C(    95.02), EASYSIMD_FLOAT32_C(   761.77),
        EASYSIMD_FLOAT32_C(   583.98), EASYSIMD_FLOAT32_C(   767.09), EASYSIMD_FLOAT32_C(   577.45), EASYSIMD_FLOAT32_C(   132.96) },
      UINT8_C(169),
      { EASYSIMD_FLOAT32_C(  -478.38), EASYSIMD_FLOAT32_C(   709.04), EASYSIMD_FLOAT32_C(  -613.12), EASYSIMD_FLOAT32_C(  -816.22),
        EASYSIMD_FLOAT32_C(  -147.56), EASYSIMD_FLOAT32_C(   183.65), EASYSIMD_FLOAT32_C(   645.23), EASYSIMD_FLOAT32_C(  -245.09) },
      { EASYSIMD_FLOAT32_C(  -242.32), EASYSIMD_FLOAT32_C(    82.00), EASYSIMD_FLOAT32_C(  -587.01), EASYSIMD_FLOAT32_C(   309.80),
        EASYSIMD_FLOAT32_C(   593.76), EASYSIMD_FLOAT32_C(  -460.48), EASYSIMD_FLOAT32_C(   356.20), EASYSIMD_FLOAT32_C(    23.36) },
      { EASYSIMD_FLOAT32_C(  -226.07), EASYSIMD_FLOAT32_C(   -20.95), EASYSIMD_FLOAT32_C(    95.02), EASYSIMD_FLOAT32_C(     2.13),
        EASYSIMD_FLOAT32_C(   583.98), EASYSIMD_FLOAT32_C(   166.14), EASYSIMD_FLOAT32_C(   577.45), EASYSIMD_FLOAT32_C(    11.00) } },
    { { EASYSIMD_FLOAT32_C(   933.86), EASYSIMD_FLOAT32_C(   503.89), EASYSIMD_FLOAT32_C(   228.98), EASYSIMD_FLOAT32_C(  -418.76),
        EASYSIMD_FLOAT32_C(   351.56), EASYSIMD_FLOAT32_C(   455.51), EASYSIMD_FLOAT32_C(   462.20), EASYSIMD_FLOAT32_C(  -669.39) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(   223.97), EASYSIMD_FLOAT32_C(   914.59), EASYSIMD_FLOAT32_C(  -682.38), EASYSIMD_FLOAT32_C(  -198.57),
        EASYSIMD_FLOAT32_C(    47.55), EASYSIMD_FLOAT32_C(  -433.57), EASYSIMD_FLOAT32_C(   323.04), EASYSIMD_FLOAT32_C(  -243.42) },
      { EASYSIMD_FLOAT32_C(   -46.69), EASYSIMD_FLOAT32_C(   506.82), EASYSIMD_FLOAT32_C(   609.03), EASYSIMD_FLOAT32_C(  -863.04),
        EASYSIMD_FLOAT32_C(   152.05), EASYSIMD_FLOAT32_C(  -636.07), EASYSIMD_FLOAT32_C(  -105.35), EASYSIMD_FLOAT32_C(  -765.95) },
      { EASYSIMD_FLOAT32_C(    38.69), EASYSIMD_FLOAT32_C(   503.89), EASYSIMD_FLOAT32_C(   228.98), EASYSIMD_FLOAT32_C(  -418.76),
        EASYSIMD_FLOAT32_C(    38.00), EASYSIMD_FLOAT32_C(   455.51), EASYSIMD_FLOAT32_C(   462.20), EASYSIMD_FLOAT32_C(  -669.39) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_and_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_and_ps");
    easysimd_test_x86_assert_equal_f32x8(easysimd_mm256_loadu_ps(test_vec[i].r), r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_and_ps(src, k, a, b);

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
test_easysimd_mm256_maskz_and_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C( 82),
      { EASYSIMD_FLOAT32_C(   162.21), EASYSIMD_FLOAT32_C(   635.64), EASYSIMD_FLOAT32_C(  -239.59), EASYSIMD_FLOAT32_C(   390.19),
        EASYSIMD_FLOAT32_C(   376.06), EASYSIMD_FLOAT32_C(   133.59), EASYSIMD_FLOAT32_C(   283.87), EASYSIMD_FLOAT32_C(   -84.73) },
      { EASYSIMD_FLOAT32_C(   246.26), EASYSIMD_FLOAT32_C(  -693.69), EASYSIMD_FLOAT32_C(  -454.27), EASYSIMD_FLOAT32_C(  -160.09),
        EASYSIMD_FLOAT32_C(   414.80), EASYSIMD_FLOAT32_C(  -390.82), EASYSIMD_FLOAT32_C(   525.89), EASYSIMD_FLOAT32_C(  -913.59) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   561.63), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   280.05), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.02), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(235),
      { EASYSIMD_FLOAT32_C(   869.69), EASYSIMD_FLOAT32_C(   252.53), EASYSIMD_FLOAT32_C(   798.05), EASYSIMD_FLOAT32_C(  -509.62),
        EASYSIMD_FLOAT32_C(   246.19), EASYSIMD_FLOAT32_C(  -137.22), EASYSIMD_FLOAT32_C(   163.95), EASYSIMD_FLOAT32_C(  -598.33) },
      { EASYSIMD_FLOAT32_C(    38.21), EASYSIMD_FLOAT32_C(   321.49), EASYSIMD_FLOAT32_C(  -929.99), EASYSIMD_FLOAT32_C(   243.03),
        EASYSIMD_FLOAT32_C(  -356.79), EASYSIMD_FLOAT32_C(  -637.54), EASYSIMD_FLOAT32_C(  -594.77), EASYSIMD_FLOAT32_C(  -721.15) },
      { EASYSIMD_FLOAT32_C(     2.38), EASYSIMD_FLOAT32_C(   160.53), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   242.03),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.14), EASYSIMD_FLOAT32_C(     2.01), EASYSIMD_FLOAT32_C(  -592.02) } },
    { UINT8_C(186),
      { EASYSIMD_FLOAT32_C(   795.42), EASYSIMD_FLOAT32_C(   654.91), EASYSIMD_FLOAT32_C(  -743.54), EASYSIMD_FLOAT32_C(    79.29),
        EASYSIMD_FLOAT32_C(  -429.82), EASYSIMD_FLOAT32_C(   502.72), EASYSIMD_FLOAT32_C(   385.59), EASYSIMD_FLOAT32_C(   115.90) },
      { EASYSIMD_FLOAT32_C(  -657.38), EASYSIMD_FLOAT32_C(  -199.60), EASYSIMD_FLOAT32_C(   725.08), EASYSIMD_FLOAT32_C(   868.51),
        EASYSIMD_FLOAT32_C(  -113.20), EASYSIMD_FLOAT32_C(  -571.10), EASYSIMD_FLOAT32_C(   738.20), EASYSIMD_FLOAT32_C(  -860.67) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.38),
        EASYSIMD_FLOAT32_C(   -97.19), EASYSIMD_FLOAT32_C(     2.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     3.11) } },
    { UINT8_C(221),
      { EASYSIMD_FLOAT32_C(  -771.43), EASYSIMD_FLOAT32_C(   385.52), EASYSIMD_FLOAT32_C(    89.73), EASYSIMD_FLOAT32_C(   392.52),
        EASYSIMD_FLOAT32_C(   787.19), EASYSIMD_FLOAT32_C(  -872.06), EASYSIMD_FLOAT32_C(  -285.99), EASYSIMD_FLOAT32_C(   857.19) },
      { EASYSIMD_FLOAT32_C(   370.96), EASYSIMD_FLOAT32_C(   357.21), EASYSIMD_FLOAT32_C(  -780.35), EASYSIMD_FLOAT32_C(   776.20),
        EASYSIMD_FLOAT32_C(   636.06), EASYSIMD_FLOAT32_C(   342.53), EASYSIMD_FLOAT32_C(   571.62), EASYSIMD_FLOAT32_C(   290.97) },
      { EASYSIMD_FLOAT32_C(     2.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.05), EASYSIMD_FLOAT32_C(     3.00),
        EASYSIMD_FLOAT32_C(   528.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.23), EASYSIMD_FLOAT32_C(     2.25) } },
    { UINT8_C(182),
      { EASYSIMD_FLOAT32_C(  -349.10), EASYSIMD_FLOAT32_C(   861.14), EASYSIMD_FLOAT32_C(   101.70), EASYSIMD_FLOAT32_C(  -963.50),
        EASYSIMD_FLOAT32_C(   -22.96), EASYSIMD_FLOAT32_C(   444.33), EASYSIMD_FLOAT32_C(  -163.11), EASYSIMD_FLOAT32_C(  -297.87) },
      { EASYSIMD_FLOAT32_C(   312.84), EASYSIMD_FLOAT32_C(   723.70), EASYSIMD_FLOAT32_C(   131.03), EASYSIMD_FLOAT32_C(    51.03),
        EASYSIMD_FLOAT32_C(   863.03), EASYSIMD_FLOAT32_C(   357.98), EASYSIMD_FLOAT32_C(   279.60), EASYSIMD_FLOAT32_C(   248.55) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   593.14), EASYSIMD_FLOAT32_C(    32.76), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     2.37), EASYSIMD_FLOAT32_C(   292.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   144.55) } },
    { UINT8_C(148),
      { EASYSIMD_FLOAT32_C(  -327.87), EASYSIMD_FLOAT32_C(    35.74), EASYSIMD_FLOAT32_C(  -424.35), EASYSIMD_FLOAT32_C(   386.13),
        EASYSIMD_FLOAT32_C(  -107.07), EASYSIMD_FLOAT32_C(   946.61), EASYSIMD_FLOAT32_C(  -256.65), EASYSIMD_FLOAT32_C(   112.58) },
      { EASYSIMD_FLOAT32_C(   722.81), EASYSIMD_FLOAT32_C(  -620.59), EASYSIMD_FLOAT32_C(  -544.89), EASYSIMD_FLOAT32_C(   294.43),
        EASYSIMD_FLOAT32_C(   670.37), EASYSIMD_FLOAT32_C(  -945.90), EASYSIMD_FLOAT32_C(   945.33), EASYSIMD_FLOAT32_C(   531.51) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     2.09), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.00) } },
    { UINT8_C(102),
      { EASYSIMD_FLOAT32_C(   981.83), EASYSIMD_FLOAT32_C(  -491.44), EASYSIMD_FLOAT32_C(  -399.87), EASYSIMD_FLOAT32_C(  -181.28),
        EASYSIMD_FLOAT32_C(   210.69), EASYSIMD_FLOAT32_C(   912.96), EASYSIMD_FLOAT32_C(  -457.58), EASYSIMD_FLOAT32_C(  -658.29) },
      { EASYSIMD_FLOAT32_C(   -36.00), EASYSIMD_FLOAT32_C(  -594.55), EASYSIMD_FLOAT32_C(   699.69), EASYSIMD_FLOAT32_C(  -756.40),
        EASYSIMD_FLOAT32_C(   654.00), EASYSIMD_FLOAT32_C(  -852.60), EASYSIMD_FLOAT32_C(   -84.27), EASYSIMD_FLOAT32_C(  -310.26) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.32), EASYSIMD_FLOAT32_C(     2.11), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   784.57), EASYSIMD_FLOAT32_C(   -80.27), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(  4),
      { EASYSIMD_FLOAT32_C(  -698.14), EASYSIMD_FLOAT32_C(   582.67), EASYSIMD_FLOAT32_C(  -330.35), EASYSIMD_FLOAT32_C(    45.20),
        EASYSIMD_FLOAT32_C(  -304.75), EASYSIMD_FLOAT32_C(  -607.54), EASYSIMD_FLOAT32_C(   424.61), EASYSIMD_FLOAT32_C(   150.36) },
      { EASYSIMD_FLOAT32_C(   686.89), EASYSIMD_FLOAT32_C(    94.98), EASYSIMD_FLOAT32_C(   204.46), EASYSIMD_FLOAT32_C(   632.22),
        EASYSIMD_FLOAT32_C(  -373.50), EASYSIMD_FLOAT32_C(  -639.74), EASYSIMD_FLOAT32_C(   614.05), EASYSIMD_FLOAT32_C(   135.06) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   132.14), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_and_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_and_ps");
    easysimd_test_x86_assert_equal_f32x8(easysimd_mm256_loadu_ps(test_vec[i].r), r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_and_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_and_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float64 src[4];
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   190.56), EASYSIMD_FLOAT64_C(  -375.84), EASYSIMD_FLOAT64_C(  -748.01), EASYSIMD_FLOAT64_C(    52.72) },
      UINT8_C(179),
      { EASYSIMD_FLOAT64_C(  -973.42), EASYSIMD_FLOAT64_C(   392.11), EASYSIMD_FLOAT64_C(  -608.91), EASYSIMD_FLOAT64_C(   368.96) },
      { EASYSIMD_FLOAT64_C(   980.02), EASYSIMD_FLOAT64_C(  -981.24), EASYSIMD_FLOAT64_C(  -175.23), EASYSIMD_FLOAT64_C(  -968.51) },
      { EASYSIMD_FLOAT64_C(   964.00), EASYSIMD_FLOAT64_C(     3.06), EASYSIMD_FLOAT64_C(  -748.01), EASYSIMD_FLOAT64_C(    52.72) } },
    { { EASYSIMD_FLOAT64_C(  -915.93), EASYSIMD_FLOAT64_C(    26.38), EASYSIMD_FLOAT64_C(  -641.05), EASYSIMD_FLOAT64_C(   355.98) },
      UINT8_C(220),
      { EASYSIMD_FLOAT64_C(  -364.15), EASYSIMD_FLOAT64_C(   353.09), EASYSIMD_FLOAT64_C(   465.92), EASYSIMD_FLOAT64_C(    -9.11) },
      { EASYSIMD_FLOAT64_C(  -586.70), EASYSIMD_FLOAT64_C(   909.58), EASYSIMD_FLOAT64_C(  -410.27), EASYSIMD_FLOAT64_C(  -613.41) },
      { EASYSIMD_FLOAT64_C(  -915.93), EASYSIMD_FLOAT64_C(    26.38), EASYSIMD_FLOAT64_C(   400.25), EASYSIMD_FLOAT64_C(    -2.27) } },
    { { EASYSIMD_FLOAT64_C(    -6.65), EASYSIMD_FLOAT64_C(    44.97), EASYSIMD_FLOAT64_C(   351.22), EASYSIMD_FLOAT64_C(   903.98) },
      UINT8_C(198),
      { EASYSIMD_FLOAT64_C(  -458.22), EASYSIMD_FLOAT64_C(  -471.86), EASYSIMD_FLOAT64_C(  -891.68), EASYSIMD_FLOAT64_C(   594.50) },
      { EASYSIMD_FLOAT64_C(   335.34), EASYSIMD_FLOAT64_C(  -865.10), EASYSIMD_FLOAT64_C(   -13.38), EASYSIMD_FLOAT64_C(   726.43) },
      { EASYSIMD_FLOAT64_C(    -6.65), EASYSIMD_FLOAT64_C(    -3.13), EASYSIMD_FLOAT64_C(    -3.34), EASYSIMD_FLOAT64_C(   903.98) } },
    { { EASYSIMD_FLOAT64_C(   503.86), EASYSIMD_FLOAT64_C(   -33.36), EASYSIMD_FLOAT64_C(   745.19), EASYSIMD_FLOAT64_C(  -671.36) },
      UINT8_C( 76),
      { EASYSIMD_FLOAT64_C(   829.27), EASYSIMD_FLOAT64_C(   355.02), EASYSIMD_FLOAT64_C(   357.08), EASYSIMD_FLOAT64_C(   185.24) },
      { EASYSIMD_FLOAT64_C(   585.29), EASYSIMD_FLOAT64_C(   992.93), EASYSIMD_FLOAT64_C(  -461.67), EASYSIMD_FLOAT64_C(    51.22) },
      { EASYSIMD_FLOAT64_C(   503.86), EASYSIMD_FLOAT64_C(   -33.36), EASYSIMD_FLOAT64_C(   325.00), EASYSIMD_FLOAT64_C(    34.03) } },
    { { EASYSIMD_FLOAT64_C(   -16.18), EASYSIMD_FLOAT64_C(   -48.37), EASYSIMD_FLOAT64_C(   -39.21), EASYSIMD_FLOAT64_C(   573.56) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT64_C(   954.14), EASYSIMD_FLOAT64_C(  -381.47), EASYSIMD_FLOAT64_C(  -310.55), EASYSIMD_FLOAT64_C(   858.12) },
      { EASYSIMD_FLOAT64_C(  -525.14), EASYSIMD_FLOAT64_C(   231.23), EASYSIMD_FLOAT64_C(  -613.74), EASYSIMD_FLOAT64_C(  -416.82) },
      { EASYSIMD_FLOAT64_C(   520.14), EASYSIMD_FLOAT64_C(   -48.37), EASYSIMD_FLOAT64_C(   -39.21), EASYSIMD_FLOAT64_C(   573.56) } },
    { { EASYSIMD_FLOAT64_C(  -174.27), EASYSIMD_FLOAT64_C(   721.60), EASYSIMD_FLOAT64_C(  -281.92), EASYSIMD_FLOAT64_C(   812.35) },
      UINT8_C(208),
      { EASYSIMD_FLOAT64_C(  -778.05), EASYSIMD_FLOAT64_C(  -221.01), EASYSIMD_FLOAT64_C(   193.22), EASYSIMD_FLOAT64_C(  -449.42) },
      { EASYSIMD_FLOAT64_C(   777.12), EASYSIMD_FLOAT64_C(    22.49), EASYSIMD_FLOAT64_C(   905.60), EASYSIMD_FLOAT64_C(   134.20) },
      { EASYSIMD_FLOAT64_C(  -174.27), EASYSIMD_FLOAT64_C(   721.60), EASYSIMD_FLOAT64_C(  -281.92), EASYSIMD_FLOAT64_C(   812.35) } },
    { { EASYSIMD_FLOAT64_C(  -792.27), EASYSIMD_FLOAT64_C(   490.89), EASYSIMD_FLOAT64_C(   127.13), EASYSIMD_FLOAT64_C(  -253.94) },
      UINT8_C(145),
      { EASYSIMD_FLOAT64_C(  -889.04), EASYSIMD_FLOAT64_C(   697.69), EASYSIMD_FLOAT64_C(   502.90), EASYSIMD_FLOAT64_C(   684.51) },
      { EASYSIMD_FLOAT64_C(    35.92), EASYSIMD_FLOAT64_C(   457.04), EASYSIMD_FLOAT64_C(  -696.96), EASYSIMD_FLOAT64_C(   725.37) },
      { EASYSIMD_FLOAT64_C(     2.22), EASYSIMD_FLOAT64_C(   490.89), EASYSIMD_FLOAT64_C(   127.13), EASYSIMD_FLOAT64_C(  -253.94) } },
    { { EASYSIMD_FLOAT64_C(   315.16), EASYSIMD_FLOAT64_C(  -222.10), EASYSIMD_FLOAT64_C(   -43.41), EASYSIMD_FLOAT64_C(   701.43) },
      UINT8_C(115),
      { EASYSIMD_FLOAT64_C(   782.32), EASYSIMD_FLOAT64_C(   423.03), EASYSIMD_FLOAT64_C(  -920.84), EASYSIMD_FLOAT64_C(   594.67) },
      { EASYSIMD_FLOAT64_C(  -128.94), EASYSIMD_FLOAT64_C(  -698.89), EASYSIMD_FLOAT64_C(  -626.35), EASYSIMD_FLOAT64_C(  -935.72) },
      { EASYSIMD_FLOAT64_C(     2.01), EASYSIMD_FLOAT64_C(     2.04), EASYSIMD_FLOAT64_C(   -43.41), EASYSIMD_FLOAT64_C(   701.43) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_and_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_and_pd");
    easysimd_test_x86_assert_equal_f64x4(easysimd_mm256_loadu_pd(test_vec[i].r), r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_and_pd(src, k, a, b);

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
test_easysimd_mm256_maskz_and_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C( 93),
      { EASYSIMD_FLOAT64_C(   865.12), EASYSIMD_FLOAT64_C(   836.88), EASYSIMD_FLOAT64_C(  -646.70), EASYSIMD_FLOAT64_C(  -292.72) },
      { EASYSIMD_FLOAT64_C(   702.02), EASYSIMD_FLOAT64_C(  -890.15), EASYSIMD_FLOAT64_C(  -923.93), EASYSIMD_FLOAT64_C(  -809.19) },
      { EASYSIMD_FLOAT64_C(   544.02), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -642.63), EASYSIMD_FLOAT64_C(    -2.04) } },
    { UINT8_C(124),
      { EASYSIMD_FLOAT64_C(  -525.15), EASYSIMD_FLOAT64_C(   449.48), EASYSIMD_FLOAT64_C(  -141.20), EASYSIMD_FLOAT64_C(     0.18) },
      { EASYSIMD_FLOAT64_C(   805.02), EASYSIMD_FLOAT64_C(   690.00), EASYSIMD_FLOAT64_C(   700.34), EASYSIMD_FLOAT64_C(   810.64) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     2.20), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 30),
      { EASYSIMD_FLOAT64_C(   213.25), EASYSIMD_FLOAT64_C(   240.46), EASYSIMD_FLOAT64_C(  -368.96), EASYSIMD_FLOAT64_C(   185.49) },
      { EASYSIMD_FLOAT64_C(   526.48), EASYSIMD_FLOAT64_C(  -672.04), EASYSIMD_FLOAT64_C(   -15.83), EASYSIMD_FLOAT64_C(  -719.55) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     2.50), EASYSIMD_FLOAT64_C(   -11.52), EASYSIMD_FLOAT64_C(     2.77) } },
    { UINT8_C(212),
      { EASYSIMD_FLOAT64_C(  -441.40), EASYSIMD_FLOAT64_C(   654.11), EASYSIMD_FLOAT64_C(   317.02), EASYSIMD_FLOAT64_C(   402.16) },
      { EASYSIMD_FLOAT64_C(   519.22), EASYSIMD_FLOAT64_C(   153.90), EASYSIMD_FLOAT64_C(   755.46), EASYSIMD_FLOAT64_C(  -773.49) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     2.45), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 81),
      { EASYSIMD_FLOAT64_C(   865.30), EASYSIMD_FLOAT64_C(  -697.42), EASYSIMD_FLOAT64_C(    46.73), EASYSIMD_FLOAT64_C(   438.17) },
      { EASYSIMD_FLOAT64_C(  -222.56), EASYSIMD_FLOAT64_C(  -503.79), EASYSIMD_FLOAT64_C(  -703.03), EASYSIMD_FLOAT64_C(   777.62) },
      { EASYSIMD_FLOAT64_C(     3.38), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(  4),
      { EASYSIMD_FLOAT64_C(   986.98), EASYSIMD_FLOAT64_C(   477.96), EASYSIMD_FLOAT64_C(  -888.13), EASYSIMD_FLOAT64_C(   599.75) },
      { EASYSIMD_FLOAT64_C(  -308.78), EASYSIMD_FLOAT64_C(   352.33), EASYSIMD_FLOAT64_C(  -769.21), EASYSIMD_FLOAT64_C(   876.70) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -768.13), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(199),
      { EASYSIMD_FLOAT64_C(  -441.26), EASYSIMD_FLOAT64_C(  -139.13), EASYSIMD_FLOAT64_C(   159.26), EASYSIMD_FLOAT64_C(   -48.31) },
      { EASYSIMD_FLOAT64_C(   419.47), EASYSIMD_FLOAT64_C(  -186.63), EASYSIMD_FLOAT64_C(  -731.29), EASYSIMD_FLOAT64_C(  -178.37) },
      { EASYSIMD_FLOAT64_C(   417.25), EASYSIMD_FLOAT64_C(  -138.13), EASYSIMD_FLOAT64_C(     2.35), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 23),
      { EASYSIMD_FLOAT64_C(   422.62), EASYSIMD_FLOAT64_C(  -422.92), EASYSIMD_FLOAT64_C(  -440.90), EASYSIMD_FLOAT64_C(  -721.46) },
      { EASYSIMD_FLOAT64_C(  -557.61), EASYSIMD_FLOAT64_C(  -138.31), EASYSIMD_FLOAT64_C(   325.27), EASYSIMD_FLOAT64_C(   880.56) },
      { EASYSIMD_FLOAT64_C(     2.05), EASYSIMD_FLOAT64_C(  -130.27), EASYSIMD_FLOAT64_C(   256.27), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_and_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_and_pd");
    easysimd_test_x86_assert_equal_f64x4(easysimd_mm256_loadu_pd(test_vec[i].r), r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_and_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_and_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  266.26), EASYSIMD_FLOAT64_C(  537.32),
                         EASYSIMD_FLOAT64_C( -326.88), EASYSIMD_FLOAT64_C( -882.50),
                         EASYSIMD_FLOAT64_C(  -89.28), EASYSIMD_FLOAT64_C( -631.60),
                         EASYSIMD_FLOAT64_C( -243.67), EASYSIMD_FLOAT64_C(   78.08)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -676.48), EASYSIMD_FLOAT64_C( -545.20),
                         EASYSIMD_FLOAT64_C(  963.41), EASYSIMD_FLOAT64_C(  343.81),
                         EASYSIMD_FLOAT64_C( -406.87), EASYSIMD_FLOAT64_C( -689.93),
                         EASYSIMD_FLOAT64_C( -169.12), EASYSIMD_FLOAT64_C( -796.89)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.02), EASYSIMD_FLOAT64_C(  513.07),
                         EASYSIMD_FLOAT64_C(    2.50), EASYSIMD_FLOAT64_C(    2.13),
                         EASYSIMD_FLOAT64_C(  -65.03), EASYSIMD_FLOAT64_C( -561.53),
                         EASYSIMD_FLOAT64_C( -161.04), EASYSIMD_FLOAT64_C(    2.06)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -354.59), EASYSIMD_FLOAT64_C( -334.01),
                         EASYSIMD_FLOAT64_C( -406.82), EASYSIMD_FLOAT64_C( -535.93),
                         EASYSIMD_FLOAT64_C(  534.72), EASYSIMD_FLOAT64_C(  276.86),
                         EASYSIMD_FLOAT64_C(  401.00), EASYSIMD_FLOAT64_C(  921.82)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -20.44), EASYSIMD_FLOAT64_C( -778.21),
                         EASYSIMD_FLOAT64_C(  -61.28), EASYSIMD_FLOAT64_C(  788.42),
                         EASYSIMD_FLOAT64_C(  286.07), EASYSIMD_FLOAT64_C(  772.65),
                         EASYSIMD_FLOAT64_C( -788.54), EASYSIMD_FLOAT64_C(  755.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -20.13), EASYSIMD_FLOAT64_C(   -2.03),
                         EASYSIMD_FLOAT64_C(  -48.26), EASYSIMD_FLOAT64_C(  532.41),
                         EASYSIMD_FLOAT64_C(    2.08), EASYSIMD_FLOAT64_C(    2.00),
                         EASYSIMD_FLOAT64_C(    3.00), EASYSIMD_FLOAT64_C(  657.31)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  197.60), EASYSIMD_FLOAT64_C( -669.73),
                         EASYSIMD_FLOAT64_C(  859.82), EASYSIMD_FLOAT64_C( -638.20),
                         EASYSIMD_FLOAT64_C( -808.24), EASYSIMD_FLOAT64_C(  961.25),
                         EASYSIMD_FLOAT64_C(  916.37), EASYSIMD_FLOAT64_C( -473.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    3.21), EASYSIMD_FLOAT64_C(   10.30),
                         EASYSIMD_FLOAT64_C(  402.59), EASYSIMD_FLOAT64_C( -919.31),
                         EASYSIMD_FLOAT64_C(  484.80), EASYSIMD_FLOAT64_C(  567.35),
                         EASYSIMD_FLOAT64_C( -979.89), EASYSIMD_FLOAT64_C(  784.39)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    3.08), EASYSIMD_FLOAT64_C(    2.57),
                         EASYSIMD_FLOAT64_C(    3.00), EASYSIMD_FLOAT64_C( -534.01),
                         EASYSIMD_FLOAT64_C(    3.03), EASYSIMD_FLOAT64_C(  513.25),
                         EASYSIMD_FLOAT64_C(  912.26), EASYSIMD_FLOAT64_C(    3.06)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  188.83), EASYSIMD_FLOAT64_C( -479.33),
                         EASYSIMD_FLOAT64_C(  811.81), EASYSIMD_FLOAT64_C( -322.50),
                         EASYSIMD_FLOAT64_C(  884.11), EASYSIMD_FLOAT64_C(  808.53),
                         EASYSIMD_FLOAT64_C( -174.95), EASYSIMD_FLOAT64_C(  -68.05)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -882.86), EASYSIMD_FLOAT64_C( -939.93),
                         EASYSIMD_FLOAT64_C( -855.90), EASYSIMD_FLOAT64_C(  170.22),
                         EASYSIMD_FLOAT64_C(  115.99), EASYSIMD_FLOAT64_C(  297.62),
                         EASYSIMD_FLOAT64_C( -527.76), EASYSIMD_FLOAT64_C(  219.88)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.45), EASYSIMD_FLOAT64_C(   -3.67),
                         EASYSIMD_FLOAT64_C(  771.77), EASYSIMD_FLOAT64_C(  160.00),
                         EASYSIMD_FLOAT64_C(    3.08), EASYSIMD_FLOAT64_C(    2.00),
                         EASYSIMD_FLOAT64_C(   -2.05), EASYSIMD_FLOAT64_C(   34.00)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -659.02), EASYSIMD_FLOAT64_C(  607.91),
                         EASYSIMD_FLOAT64_C( -268.25), EASYSIMD_FLOAT64_C(  240.07),
                         EASYSIMD_FLOAT64_C(  471.39), EASYSIMD_FLOAT64_C( -501.59),
                         EASYSIMD_FLOAT64_C(  984.94), EASYSIMD_FLOAT64_C( -801.62)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -165.03), EASYSIMD_FLOAT64_C(  382.49),
                         EASYSIMD_FLOAT64_C( -663.11), EASYSIMD_FLOAT64_C(  675.92),
                         EASYSIMD_FLOAT64_C( -427.89), EASYSIMD_FLOAT64_C( -312.23),
                         EASYSIMD_FLOAT64_C(   47.19), EASYSIMD_FLOAT64_C( -273.76)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -2.56), EASYSIMD_FLOAT64_C(    2.36),
                         EASYSIMD_FLOAT64_C(   -2.06), EASYSIMD_FLOAT64_C(    2.50),
                         EASYSIMD_FLOAT64_C(  387.39), EASYSIMD_FLOAT64_C( -304.07),
                         EASYSIMD_FLOAT64_C(    2.81), EASYSIMD_FLOAT64_C(   -2.13)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -449.51), EASYSIMD_FLOAT64_C( -396.24),
                         EASYSIMD_FLOAT64_C( -106.23), EASYSIMD_FLOAT64_C( -648.77),
                         EASYSIMD_FLOAT64_C(  178.69), EASYSIMD_FLOAT64_C( -996.05),
                         EASYSIMD_FLOAT64_C(  315.07), EASYSIMD_FLOAT64_C( -247.28)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  515.02), EASYSIMD_FLOAT64_C(  228.66),
                         EASYSIMD_FLOAT64_C(  419.85), EASYSIMD_FLOAT64_C( -810.27),
                         EASYSIMD_FLOAT64_C(  162.64), EASYSIMD_FLOAT64_C(  495.48),
                         EASYSIMD_FLOAT64_C( -567.27), EASYSIMD_FLOAT64_C(  755.82)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.01), EASYSIMD_FLOAT64_C(  196.03),
                         EASYSIMD_FLOAT64_C(  104.20), EASYSIMD_FLOAT64_C( -520.27),
                         EASYSIMD_FLOAT64_C(  162.63), EASYSIMD_FLOAT64_C(    3.77),
                         EASYSIMD_FLOAT64_C(    2.21), EASYSIMD_FLOAT64_C(    2.82)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -506.18), EASYSIMD_FLOAT64_C(  153.12),
                         EASYSIMD_FLOAT64_C( -217.93), EASYSIMD_FLOAT64_C(    6.73),
                         EASYSIMD_FLOAT64_C(  358.11), EASYSIMD_FLOAT64_C( -136.37),
                         EASYSIMD_FLOAT64_C(  141.08), EASYSIMD_FLOAT64_C( -860.28)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -303.86), EASYSIMD_FLOAT64_C( -938.78),
                         EASYSIMD_FLOAT64_C(  386.83), EASYSIMD_FLOAT64_C( -590.09),
                         EASYSIMD_FLOAT64_C( -517.39), EASYSIMD_FLOAT64_C( -324.41),
                         EASYSIMD_FLOAT64_C(  515.48), EASYSIMD_FLOAT64_C(  674.62)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -298.05), EASYSIMD_FLOAT64_C(    2.13),
                         EASYSIMD_FLOAT64_C(  193.41), EASYSIMD_FLOAT64_C(    2.30),
                         EASYSIMD_FLOAT64_C(    2.02), EASYSIMD_FLOAT64_C( -128.08),
                         EASYSIMD_FLOAT64_C(    2.00), EASYSIMD_FLOAT64_C(  512.03)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -571.57), EASYSIMD_FLOAT64_C( -865.77),
                         EASYSIMD_FLOAT64_C( -691.63), EASYSIMD_FLOAT64_C( -182.56),
                         EASYSIMD_FLOAT64_C(  -67.70), EASYSIMD_FLOAT64_C( -166.11),
                         EASYSIMD_FLOAT64_C( -833.08), EASYSIMD_FLOAT64_C( -401.07)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  341.96), EASYSIMD_FLOAT64_C(  615.56),
                         EASYSIMD_FLOAT64_C(  144.45), EASYSIMD_FLOAT64_C(  211.78),
                         EASYSIMD_FLOAT64_C(  -86.51), EASYSIMD_FLOAT64_C(  594.64),
                         EASYSIMD_FLOAT64_C(  523.21), EASYSIMD_FLOAT64_C( -747.41)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.17), EASYSIMD_FLOAT64_C(  609.52),
                         EASYSIMD_FLOAT64_C(    2.01), EASYSIMD_FLOAT64_C(  146.53),
                         EASYSIMD_FLOAT64_C(  -66.51), EASYSIMD_FLOAT64_C(    2.06),
                         EASYSIMD_FLOAT64_C(  513.08), EASYSIMD_FLOAT64_C(   -2.13)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_and_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_and_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_and_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -343.60), EASYSIMD_FLOAT32_C(  -192.26), EASYSIMD_FLOAT32_C(  -375.10), EASYSIMD_FLOAT32_C(   810.28),
                         EASYSIMD_FLOAT32_C(  -388.15), EASYSIMD_FLOAT32_C(    15.81), EASYSIMD_FLOAT32_C(   547.95), EASYSIMD_FLOAT32_C(   151.06),
                         EASYSIMD_FLOAT32_C(  -920.74), EASYSIMD_FLOAT32_C(  -676.14), EASYSIMD_FLOAT32_C(  -545.26), EASYSIMD_FLOAT32_C(   -14.56),
                         EASYSIMD_FLOAT32_C(  -393.14), EASYSIMD_FLOAT32_C(   768.60), EASYSIMD_FLOAT32_C(  -177.89), EASYSIMD_FLOAT32_C(  -467.51)),
      UINT16_C(45944),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -651.37), EASYSIMD_FLOAT32_C(    37.95), EASYSIMD_FLOAT32_C(  -182.79), EASYSIMD_FLOAT32_C(   255.51),
                         EASYSIMD_FLOAT32_C(   476.70), EASYSIMD_FLOAT32_C(   371.61), EASYSIMD_FLOAT32_C(  -494.45), EASYSIMD_FLOAT32_C(    72.18),
                         EASYSIMD_FLOAT32_C(  -723.25), EASYSIMD_FLOAT32_C(   604.60), EASYSIMD_FLOAT32_C(   545.32), EASYSIMD_FLOAT32_C(  -399.73),
                         EASYSIMD_FLOAT32_C(  -975.39), EASYSIMD_FLOAT32_C(   419.30), EASYSIMD_FLOAT32_C(  -736.37), EASYSIMD_FLOAT32_C(   655.70)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -330.93), EASYSIMD_FLOAT32_C(   963.24), EASYSIMD_FLOAT32_C(   -62.45), EASYSIMD_FLOAT32_C(   625.74),
                         EASYSIMD_FLOAT32_C(  -826.45), EASYSIMD_FLOAT32_C(  -884.51), EASYSIMD_FLOAT32_C(   544.59), EASYSIMD_FLOAT32_C(   -22.39),
                         EASYSIMD_FLOAT32_C(   750.16), EASYSIMD_FLOAT32_C(  -751.51), EASYSIMD_FLOAT32_C(  -211.00), EASYSIMD_FLOAT32_C(   886.29),
                         EASYSIMD_FLOAT32_C(   666.91), EASYSIMD_FLOAT32_C(     8.70), EASYSIMD_FLOAT32_C(  -362.66), EASYSIMD_FLOAT32_C(  -451.03)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    -2.51), EASYSIMD_FLOAT32_C(  -192.26), EASYSIMD_FLOAT32_C(   -44.20), EASYSIMD_FLOAT32_C(     2.44),
                         EASYSIMD_FLOAT32_C(  -388.15), EASYSIMD_FLOAT32_C(    15.81), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     4.50),
                         EASYSIMD_FLOAT32_C(  -920.74), EASYSIMD_FLOAT32_C(   588.50), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     3.09),
                         EASYSIMD_FLOAT32_C(   650.38), EASYSIMD_FLOAT32_C(   768.60), EASYSIMD_FLOAT32_C(  -177.89), EASYSIMD_FLOAT32_C(  -467.51)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -702.08), EASYSIMD_FLOAT32_C(   457.26), EASYSIMD_FLOAT32_C(   193.10), EASYSIMD_FLOAT32_C(   655.72),
                         EASYSIMD_FLOAT32_C(   205.91), EASYSIMD_FLOAT32_C(   807.77), EASYSIMD_FLOAT32_C(  -545.40), EASYSIMD_FLOAT32_C(  -364.12),
                         EASYSIMD_FLOAT32_C(   -42.22), EASYSIMD_FLOAT32_C(  -523.42), EASYSIMD_FLOAT32_C(  -308.90), EASYSIMD_FLOAT32_C(    22.20),
                         EASYSIMD_FLOAT32_C(  -114.47), EASYSIMD_FLOAT32_C(  -738.11), EASYSIMD_FLOAT32_C(   189.09), EASYSIMD_FLOAT32_C(  -448.58)),
      UINT16_C(10313),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -177.43), EASYSIMD_FLOAT32_C(   -28.38), EASYSIMD_FLOAT32_C(  -846.37), EASYSIMD_FLOAT32_C(   912.26),
                         EASYSIMD_FLOAT32_C(  -370.39), EASYSIMD_FLOAT32_C(   988.78), EASYSIMD_FLOAT32_C(  -359.74), EASYSIMD_FLOAT32_C(  -281.72),
                         EASYSIMD_FLOAT32_C(   166.18), EASYSIMD_FLOAT32_C(  -100.50), EASYSIMD_FLOAT32_C(  -909.51), EASYSIMD_FLOAT32_C(   -85.95),
                         EASYSIMD_FLOAT32_C(  -710.91), EASYSIMD_FLOAT32_C(  -813.11), EASYSIMD_FLOAT32_C(  -799.86), EASYSIMD_FLOAT32_C(  -823.45)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   329.92), EASYSIMD_FLOAT32_C(   113.21), EASYSIMD_FLOAT32_C(   300.37), EASYSIMD_FLOAT32_C(  -777.20),
                         EASYSIMD_FLOAT32_C(   193.77), EASYSIMD_FLOAT32_C(  -864.32), EASYSIMD_FLOAT32_C(   579.99), EASYSIMD_FLOAT32_C(   488.59),
                         EASYSIMD_FLOAT32_C(  -684.28), EASYSIMD_FLOAT32_C(   -65.28), EASYSIMD_FLOAT32_C(   876.26), EASYSIMD_FLOAT32_C(   378.65),
                         EASYSIMD_FLOAT32_C(  -964.10), EASYSIMD_FLOAT32_C(   626.06), EASYSIMD_FLOAT32_C(    97.19), EASYSIMD_FLOAT32_C(   612.33)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -702.08), EASYSIMD_FLOAT32_C(   457.26), EASYSIMD_FLOAT32_C(     2.28), EASYSIMD_FLOAT32_C(   655.72),
                         EASYSIMD_FLOAT32_C(   129.00), EASYSIMD_FLOAT32_C(   807.77), EASYSIMD_FLOAT32_C(  -545.40), EASYSIMD_FLOAT32_C(  -364.12),
                         EASYSIMD_FLOAT32_C(   -42.22), EASYSIMD_FLOAT32_C(   -64.00), EASYSIMD_FLOAT32_C(  -308.90), EASYSIMD_FLOAT32_C(    22.20),
                         EASYSIMD_FLOAT32_C(  -708.03), EASYSIMD_FLOAT32_C(  -738.11), EASYSIMD_FLOAT32_C(   189.09), EASYSIMD_FLOAT32_C(   548.31)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   153.13), EASYSIMD_FLOAT32_C(    47.64), EASYSIMD_FLOAT32_C(  -104.37), EASYSIMD_FLOAT32_C(  -453.36),
                         EASYSIMD_FLOAT32_C(   -57.32), EASYSIMD_FLOAT32_C(  -673.06), EASYSIMD_FLOAT32_C(  -857.97), EASYSIMD_FLOAT32_C(  -158.69),
                         EASYSIMD_FLOAT32_C(   504.22), EASYSIMD_FLOAT32_C(   774.61), EASYSIMD_FLOAT32_C(   -50.26), EASYSIMD_FLOAT32_C(  -594.62),
                         EASYSIMD_FLOAT32_C(   628.86), EASYSIMD_FLOAT32_C(   362.00), EASYSIMD_FLOAT32_C(   770.65), EASYSIMD_FLOAT32_C(  -621.70)),
      UINT16_C( 5674),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -181.31), EASYSIMD_FLOAT32_C(  -271.84), EASYSIMD_FLOAT32_C(   138.26), EASYSIMD_FLOAT32_C(    59.10),
                         EASYSIMD_FLOAT32_C(   703.12), EASYSIMD_FLOAT32_C(   374.71), EASYSIMD_FLOAT32_C(  -674.86), EASYSIMD_FLOAT32_C(  -198.23),
                         EASYSIMD_FLOAT32_C(   769.31), EASYSIMD_FLOAT32_C(  -859.16), EASYSIMD_FLOAT32_C(   111.69), EASYSIMD_FLOAT32_C(  -420.38),
                         EASYSIMD_FLOAT32_C(   345.23), EASYSIMD_FLOAT32_C(  -263.27), EASYSIMD_FLOAT32_C(   122.33), EASYSIMD_FLOAT32_C(   -11.31)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    13.03), EASYSIMD_FLOAT32_C(   510.73), EASYSIMD_FLOAT32_C(    -6.19), EASYSIMD_FLOAT32_C(  -107.87),
                         EASYSIMD_FLOAT32_C(  -441.23), EASYSIMD_FLOAT32_C(   120.22), EASYSIMD_FLOAT32_C(   331.67), EASYSIMD_FLOAT32_C(  -661.48),
                         EASYSIMD_FLOAT32_C(   626.32), EASYSIMD_FLOAT32_C(   505.21), EASYSIMD_FLOAT32_C(  -161.83), EASYSIMD_FLOAT32_C(  -671.34),
                         EASYSIMD_FLOAT32_C(   514.06), EASYSIMD_FLOAT32_C(  -807.61), EASYSIMD_FLOAT32_C(  -556.61), EASYSIMD_FLOAT32_C(  -451.72)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   153.13), EASYSIMD_FLOAT32_C(    47.64), EASYSIMD_FLOAT32_C(  -104.37), EASYSIMD_FLOAT32_C(    49.04),
                         EASYSIMD_FLOAT32_C(   -57.32), EASYSIMD_FLOAT32_C(    88.16), EASYSIMD_FLOAT32_C(     2.51), EASYSIMD_FLOAT32_C(  -158.69),
                         EASYSIMD_FLOAT32_C(   504.22), EASYSIMD_FLOAT32_C(   774.61), EASYSIMD_FLOAT32_C(    32.31), EASYSIMD_FLOAT32_C(  -594.62),
                         EASYSIMD_FLOAT32_C(     2.01), EASYSIMD_FLOAT32_C(   362.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(  -621.70)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -175.84), EASYSIMD_FLOAT32_C(   968.44), EASYSIMD_FLOAT32_C(    85.61), EASYSIMD_FLOAT32_C(  -394.33),
                         EASYSIMD_FLOAT32_C(   358.35), EASYSIMD_FLOAT32_C(   605.54), EASYSIMD_FLOAT32_C(  -698.35), EASYSIMD_FLOAT32_C(  -764.09),
                         EASYSIMD_FLOAT32_C(   164.55), EASYSIMD_FLOAT32_C(  -893.53), EASYSIMD_FLOAT32_C(   171.50), EASYSIMD_FLOAT32_C(   629.19),
                         EASYSIMD_FLOAT32_C(    42.86), EASYSIMD_FLOAT32_C(    22.57), EASYSIMD_FLOAT32_C(   198.87), EASYSIMD_FLOAT32_C(  -209.78)),
      UINT16_C(35386),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -72.35), EASYSIMD_FLOAT32_C(  -549.59), EASYSIMD_FLOAT32_C(   102.63), EASYSIMD_FLOAT32_C(   834.67),
                         EASYSIMD_FLOAT32_C(     4.81), EASYSIMD_FLOAT32_C(   910.94), EASYSIMD_FLOAT32_C(   192.67), EASYSIMD_FLOAT32_C(   180.42),
                         EASYSIMD_FLOAT32_C(   349.29), EASYSIMD_FLOAT32_C(   183.58), EASYSIMD_FLOAT32_C(   366.06), EASYSIMD_FLOAT32_C(  -157.87),
                         EASYSIMD_FLOAT32_C(  -312.42), EASYSIMD_FLOAT32_C(   182.79), EASYSIMD_FLOAT32_C(  -978.11), EASYSIMD_FLOAT32_C(    90.48)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   175.13), EASYSIMD_FLOAT32_C(  -712.55), EASYSIMD_FLOAT32_C(  -809.33), EASYSIMD_FLOAT32_C(   698.74),
                         EASYSIMD_FLOAT32_C(   142.25), EASYSIMD_FLOAT32_C(  -727.89), EASYSIMD_FLOAT32_C(  -520.56), EASYSIMD_FLOAT32_C(   353.74),
                         EASYSIMD_FLOAT32_C(  -705.41), EASYSIMD_FLOAT32_C(  -196.42), EASYSIMD_FLOAT32_C(   407.84), EASYSIMD_FLOAT32_C(  -285.59),
                         EASYSIMD_FLOAT32_C(   496.15), EASYSIMD_FLOAT32_C(   800.83), EASYSIMD_FLOAT32_C(  -740.01), EASYSIMD_FLOAT32_C(   769.91)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    32.03), EASYSIMD_FLOAT32_C(   968.44), EASYSIMD_FLOAT32_C(    85.61), EASYSIMD_FLOAT32_C(  -394.33),
                         EASYSIMD_FLOAT32_C(     2.13), EASYSIMD_FLOAT32_C(   605.54), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(  -764.09),
                         EASYSIMD_FLOAT32_C(   164.55), EASYSIMD_FLOAT32_C(  -893.53), EASYSIMD_FLOAT32_C(   262.03), EASYSIMD_FLOAT32_C(  -140.79),
                         EASYSIMD_FLOAT32_C(   304.13), EASYSIMD_FLOAT32_C(    22.57), EASYSIMD_FLOAT32_C(  -704.00), EASYSIMD_FLOAT32_C(  -209.78)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -273.10), EASYSIMD_FLOAT32_C(  -193.08), EASYSIMD_FLOAT32_C(   823.95), EASYSIMD_FLOAT32_C(   970.90),
                         EASYSIMD_FLOAT32_C(   -50.31), EASYSIMD_FLOAT32_C(   755.59), EASYSIMD_FLOAT32_C(  -119.92), EASYSIMD_FLOAT32_C(  -895.51),
                         EASYSIMD_FLOAT32_C(   692.21), EASYSIMD_FLOAT32_C(   544.09), EASYSIMD_FLOAT32_C(   740.64), EASYSIMD_FLOAT32_C(   817.79),
                         EASYSIMD_FLOAT32_C(   131.04), EASYSIMD_FLOAT32_C(   190.96), EASYSIMD_FLOAT32_C(   289.64), EASYSIMD_FLOAT32_C(  -908.35)),
      UINT16_C( 1662),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   563.69), EASYSIMD_FLOAT32_C(   374.34), EASYSIMD_FLOAT32_C(  -459.61), EASYSIMD_FLOAT32_C(   786.82),
                         EASYSIMD_FLOAT32_C(   257.72), EASYSIMD_FLOAT32_C(  -220.73), EASYSIMD_FLOAT32_C(  -903.10), EASYSIMD_FLOAT32_C(   520.58),
                         EASYSIMD_FLOAT32_C(  -858.27), EASYSIMD_FLOAT32_C(   784.57), EASYSIMD_FLOAT32_C(   832.81), EASYSIMD_FLOAT32_C(  -909.15),
                         EASYSIMD_FLOAT32_C(   909.58), EASYSIMD_FLOAT32_C(  -162.79), EASYSIMD_FLOAT32_C(   177.63), EASYSIMD_FLOAT32_C(    25.46)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   749.97), EASYSIMD_FLOAT32_C(   -58.76), EASYSIMD_FLOAT32_C(   952.36), EASYSIMD_FLOAT32_C(   549.26),
                         EASYSIMD_FLOAT32_C(   390.25), EASYSIMD_FLOAT32_C(  -490.70), EASYSIMD_FLOAT32_C(   974.89), EASYSIMD_FLOAT32_C(   114.95),
                         EASYSIMD_FLOAT32_C(   932.36), EASYSIMD_FLOAT32_C(  -895.93), EASYSIMD_FLOAT32_C(  -880.84), EASYSIMD_FLOAT32_C(  -351.20),
                         EASYSIMD_FLOAT32_C(  -500.77), EASYSIMD_FLOAT32_C(    42.49), EASYSIMD_FLOAT32_C(   588.62), EASYSIMD_FLOAT32_C(    67.54)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -273.10), EASYSIMD_FLOAT32_C(  -193.08), EASYSIMD_FLOAT32_C(   823.95), EASYSIMD_FLOAT32_C(   970.90),
                         EASYSIMD_FLOAT32_C(   -50.31), EASYSIMD_FLOAT32_C(  -212.10), EASYSIMD_FLOAT32_C(   902.01), EASYSIMD_FLOAT32_C(  -895.51),
                         EASYSIMD_FLOAT32_C(   692.21), EASYSIMD_FLOAT32_C(   784.50), EASYSIMD_FLOAT32_C(   832.78), EASYSIMD_FLOAT32_C(    -2.55),
                         EASYSIMD_FLOAT32_C(     3.54), EASYSIMD_FLOAT32_C(    40.19), EASYSIMD_FLOAT32_C(     2.27), EASYSIMD_FLOAT32_C(  -908.35)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   882.37), EASYSIMD_FLOAT32_C(   -29.48), EASYSIMD_FLOAT32_C(   208.93), EASYSIMD_FLOAT32_C(  -103.96),
                         EASYSIMD_FLOAT32_C(  -740.71), EASYSIMD_FLOAT32_C(   -48.33), EASYSIMD_FLOAT32_C(   -73.48), EASYSIMD_FLOAT32_C(   839.05),
                         EASYSIMD_FLOAT32_C(  -578.39), EASYSIMD_FLOAT32_C(  -527.30), EASYSIMD_FLOAT32_C(   808.78), EASYSIMD_FLOAT32_C(   273.31),
                         EASYSIMD_FLOAT32_C(  -212.18), EASYSIMD_FLOAT32_C(   358.44), EASYSIMD_FLOAT32_C(  -429.58), EASYSIMD_FLOAT32_C(   641.01)),
      UINT16_C(51954),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   159.89), EASYSIMD_FLOAT32_C(  -431.59), EASYSIMD_FLOAT32_C(   692.24), EASYSIMD_FLOAT32_C(  -189.31),
                         EASYSIMD_FLOAT32_C(    84.37), EASYSIMD_FLOAT32_C(  -971.33), EASYSIMD_FLOAT32_C(    50.60), EASYSIMD_FLOAT32_C(  -980.81),
                         EASYSIMD_FLOAT32_C(   362.99), EASYSIMD_FLOAT32_C(   722.54), EASYSIMD_FLOAT32_C(   564.98), EASYSIMD_FLOAT32_C(   242.21),
                         EASYSIMD_FLOAT32_C(  -393.24), EASYSIMD_FLOAT32_C(   738.28), EASYSIMD_FLOAT32_C(   192.78), EASYSIMD_FLOAT32_C(  -360.32)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -421.96), EASYSIMD_FLOAT32_C(  -741.98), EASYSIMD_FLOAT32_C(  -791.19), EASYSIMD_FLOAT32_C(   363.28),
                         EASYSIMD_FLOAT32_C(   168.15), EASYSIMD_FLOAT32_C(  -247.26), EASYSIMD_FLOAT32_C(   113.19), EASYSIMD_FLOAT32_C(   128.76),
                         EASYSIMD_FLOAT32_C(  -773.73), EASYSIMD_FLOAT32_C(   125.25), EASYSIMD_FLOAT32_C(   337.69), EASYSIMD_FLOAT32_C(  -644.22),
                         EASYSIMD_FLOAT32_C(   869.52), EASYSIMD_FLOAT32_C(   681.99), EASYSIMD_FLOAT32_C(   444.36), EASYSIMD_FLOAT32_C(   361.44)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   146.89), EASYSIMD_FLOAT32_C(    -2.27), EASYSIMD_FLOAT32_C(   208.93), EASYSIMD_FLOAT32_C(  -103.96),
                         EASYSIMD_FLOAT32_C(    42.04), EASYSIMD_FLOAT32_C(   -48.33), EASYSIMD_FLOAT32_C(    48.59), EASYSIMD_FLOAT32_C(   839.05),
                         EASYSIMD_FLOAT32_C(     2.02), EASYSIMD_FLOAT32_C(     2.76), EASYSIMD_FLOAT32_C(     2.13), EASYSIMD_FLOAT32_C(     2.50),
                         EASYSIMD_FLOAT32_C(  -212.18), EASYSIMD_FLOAT32_C(   358.44), EASYSIMD_FLOAT32_C(   192.02), EASYSIMD_FLOAT32_C(   641.01)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -829.72), EASYSIMD_FLOAT32_C(   349.95), EASYSIMD_FLOAT32_C(   480.03), EASYSIMD_FLOAT32_C(  -584.69),
                         EASYSIMD_FLOAT32_C(   943.11), EASYSIMD_FLOAT32_C(  -148.79), EASYSIMD_FLOAT32_C(  -861.78), EASYSIMD_FLOAT32_C(  -270.87),
                         EASYSIMD_FLOAT32_C(  -593.74), EASYSIMD_FLOAT32_C(  -232.02), EASYSIMD_FLOAT32_C(  -553.31), EASYSIMD_FLOAT32_C(   693.33),
                         EASYSIMD_FLOAT32_C(  -533.82), EASYSIMD_FLOAT32_C(  -527.51), EASYSIMD_FLOAT32_C(  -140.16), EASYSIMD_FLOAT32_C(   631.76)),
      UINT16_C(50263),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   173.33), EASYSIMD_FLOAT32_C(  -281.34), EASYSIMD_FLOAT32_C(   -45.38), EASYSIMD_FLOAT32_C(  -230.23),
                         EASYSIMD_FLOAT32_C(  -937.39), EASYSIMD_FLOAT32_C(    53.86), EASYSIMD_FLOAT32_C(  -719.43), EASYSIMD_FLOAT32_C(   465.60),
                         EASYSIMD_FLOAT32_C(   111.60), EASYSIMD_FLOAT32_C(   156.01), EASYSIMD_FLOAT32_C(  -703.23), EASYSIMD_FLOAT32_C(   763.33),
                         EASYSIMD_FLOAT32_C(   119.12), EASYSIMD_FLOAT32_C(  -295.56), EASYSIMD_FLOAT32_C(   313.51), EASYSIMD_FLOAT32_C(  -193.21)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   396.81), EASYSIMD_FLOAT32_C(  -330.26), EASYSIMD_FLOAT32_C(  -500.01), EASYSIMD_FLOAT32_C(  -117.27),
                         EASYSIMD_FLOAT32_C(   805.35), EASYSIMD_FLOAT32_C(   722.55), EASYSIMD_FLOAT32_C(   274.82), EASYSIMD_FLOAT32_C(    32.73),
                         EASYSIMD_FLOAT32_C(  -564.66), EASYSIMD_FLOAT32_C(   180.25), EASYSIMD_FLOAT32_C(  -307.87), EASYSIMD_FLOAT32_C(   888.96),
                         EASYSIMD_FLOAT32_C(   806.77), EASYSIMD_FLOAT32_C(  -526.35), EASYSIMD_FLOAT32_C(   889.50), EASYSIMD_FLOAT32_C(   196.92)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   132.27), EASYSIMD_FLOAT32_C(  -264.26), EASYSIMD_FLOAT32_C(   480.03), EASYSIMD_FLOAT32_C(  -584.69),
                         EASYSIMD_FLOAT32_C(   943.11), EASYSIMD_FLOAT32_C(     2.31), EASYSIMD_FLOAT32_C(  -861.78), EASYSIMD_FLOAT32_C(  -270.87),
                         EASYSIMD_FLOAT32_C(  -593.74), EASYSIMD_FLOAT32_C(   148.00), EASYSIMD_FLOAT32_C(  -553.31), EASYSIMD_FLOAT32_C(   632.33),
                         EASYSIMD_FLOAT32_C(  -533.82), EASYSIMD_FLOAT32_C(    -2.06), EASYSIMD_FLOAT32_C(     2.44), EASYSIMD_FLOAT32_C(   192.13)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   425.40), EASYSIMD_FLOAT32_C(  -281.85), EASYSIMD_FLOAT32_C(   596.53), EASYSIMD_FLOAT32_C(   231.55),
                         EASYSIMD_FLOAT32_C(  -189.24), EASYSIMD_FLOAT32_C(   962.54), EASYSIMD_FLOAT32_C(   598.72), EASYSIMD_FLOAT32_C(  -728.82),
                         EASYSIMD_FLOAT32_C(   -31.34), EASYSIMD_FLOAT32_C(  -498.28), EASYSIMD_FLOAT32_C(  -106.48), EASYSIMD_FLOAT32_C(  -850.40),
                         EASYSIMD_FLOAT32_C(  -763.83), EASYSIMD_FLOAT32_C(   176.55), EASYSIMD_FLOAT32_C(   356.84), EASYSIMD_FLOAT32_C(   827.17)),
      UINT16_C(54643),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   761.43), EASYSIMD_FLOAT32_C(    95.69), EASYSIMD_FLOAT32_C(   888.39), EASYSIMD_FLOAT32_C(  -555.84),
                         EASYSIMD_FLOAT32_C(    40.33), EASYSIMD_FLOAT32_C(   358.74), EASYSIMD_FLOAT32_C(  -948.08), EASYSIMD_FLOAT32_C(   313.44),
                         EASYSIMD_FLOAT32_C(  -166.07), EASYSIMD_FLOAT32_C(  -218.95), EASYSIMD_FLOAT32_C(   360.34), EASYSIMD_FLOAT32_C(   989.68),
                         EASYSIMD_FLOAT32_C(   653.42), EASYSIMD_FLOAT32_C(   345.37), EASYSIMD_FLOAT32_C(   978.06), EASYSIMD_FLOAT32_C(   493.94)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   109.54), EASYSIMD_FLOAT32_C(   122.35), EASYSIMD_FLOAT32_C(   770.11), EASYSIMD_FLOAT32_C(   306.89),
                         EASYSIMD_FLOAT32_C(  -347.63), EASYSIMD_FLOAT32_C(   772.43), EASYSIMD_FLOAT32_C(   958.72), EASYSIMD_FLOAT32_C(  -435.18),
                         EASYSIMD_FLOAT32_C(  -680.27), EASYSIMD_FLOAT32_C(  -653.21), EASYSIMD_FLOAT32_C(   453.00), EASYSIMD_FLOAT32_C(   299.53),
                         EASYSIMD_FLOAT32_C(  -837.12), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(   561.63), EASYSIMD_FLOAT32_C(  -594.20)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     2.41), EASYSIMD_FLOAT32_C(    90.06), EASYSIMD_FLOAT32_C(   596.53), EASYSIMD_FLOAT32_C(     2.13),
                         EASYSIMD_FLOAT32_C(  -189.24), EASYSIMD_FLOAT32_C(     2.02), EASYSIMD_FLOAT32_C(   598.72), EASYSIMD_FLOAT32_C(   305.13),
                         EASYSIMD_FLOAT32_C(   -31.34), EASYSIMD_FLOAT32_C(    -2.04), EASYSIMD_FLOAT32_C(   320.00), EASYSIMD_FLOAT32_C(     2.33),
                         EASYSIMD_FLOAT32_C(  -763.83), EASYSIMD_FLOAT32_C(   176.55), EASYSIMD_FLOAT32_C(   528.01), EASYSIMD_FLOAT32_C(     2.32)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = test_vec[i].src;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_and_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_and_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_and_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -128.09), EASYSIMD_FLOAT64_C( -302.68),
                         EASYSIMD_FLOAT64_C(  129.66), EASYSIMD_FLOAT64_C( -400.28),
                         EASYSIMD_FLOAT64_C( -687.60), EASYSIMD_FLOAT64_C( -568.06),
                         EASYSIMD_FLOAT64_C( -974.67), EASYSIMD_FLOAT64_C(  814.47)),
      UINT8_C( 92),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -854.57), EASYSIMD_FLOAT64_C(  353.06),
                         EASYSIMD_FLOAT64_C(  903.81), EASYSIMD_FLOAT64_C( -723.16),
                         EASYSIMD_FLOAT64_C( -194.97), EASYSIMD_FLOAT64_C(  114.89),
                         EASYSIMD_FLOAT64_C(  497.66), EASYSIMD_FLOAT64_C( -446.09)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -876.89), EASYSIMD_FLOAT64_C( -283.08),
                         EASYSIMD_FLOAT64_C(  642.58), EASYSIMD_FLOAT64_C( -973.49),
                         EASYSIMD_FLOAT64_C(  853.14), EASYSIMD_FLOAT64_C(  647.44),
                         EASYSIMD_FLOAT64_C(  237.52), EASYSIMD_FLOAT64_C( -333.12)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -128.09), EASYSIMD_FLOAT64_C(  257.02),
                         EASYSIMD_FLOAT64_C(  129.66), EASYSIMD_FLOAT64_C( -705.16),
                         EASYSIMD_FLOAT64_C(    3.00), EASYSIMD_FLOAT64_C(    2.53),
                         EASYSIMD_FLOAT64_C( -974.67), EASYSIMD_FLOAT64_C(  814.47)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   61.63), EASYSIMD_FLOAT64_C(  987.28),
                         EASYSIMD_FLOAT64_C( -845.84), EASYSIMD_FLOAT64_C( -822.08),
                         EASYSIMD_FLOAT64_C( -946.95), EASYSIMD_FLOAT64_C( -157.17),
                         EASYSIMD_FLOAT64_C(  808.43), EASYSIMD_FLOAT64_C(  716.34)),
      UINT8_C(128),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  876.49), EASYSIMD_FLOAT64_C(  503.33),
                         EASYSIMD_FLOAT64_C(  842.44), EASYSIMD_FLOAT64_C( -417.76),
                         EASYSIMD_FLOAT64_C( -171.61), EASYSIMD_FLOAT64_C(  -96.79),
                         EASYSIMD_FLOAT64_C(   45.73), EASYSIMD_FLOAT64_C(  312.19)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   54.70), EASYSIMD_FLOAT64_C( -228.57),
                         EASYSIMD_FLOAT64_C( -133.57), EASYSIMD_FLOAT64_C( -803.47),
                         EASYSIMD_FLOAT64_C(  821.61), EASYSIMD_FLOAT64_C(  198.21),
                         EASYSIMD_FLOAT64_C(  476.20), EASYSIMD_FLOAT64_C(  925.71)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    3.41), EASYSIMD_FLOAT64_C(  987.28),
                         EASYSIMD_FLOAT64_C( -845.84), EASYSIMD_FLOAT64_C( -822.08),
                         EASYSIMD_FLOAT64_C( -946.95), EASYSIMD_FLOAT64_C( -157.17),
                         EASYSIMD_FLOAT64_C(  808.43), EASYSIMD_FLOAT64_C(  716.34)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -24.63), EASYSIMD_FLOAT64_C( -758.53),
                         EASYSIMD_FLOAT64_C(  216.18), EASYSIMD_FLOAT64_C( -869.86),
                         EASYSIMD_FLOAT64_C( -556.61), EASYSIMD_FLOAT64_C( -869.93),
                         EASYSIMD_FLOAT64_C(  935.72), EASYSIMD_FLOAT64_C(  467.65)),
      UINT8_C(132),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -373.41), EASYSIMD_FLOAT64_C(  558.94),
                         EASYSIMD_FLOAT64_C( -966.64), EASYSIMD_FLOAT64_C( -741.87),
                         EASYSIMD_FLOAT64_C( -915.12), EASYSIMD_FLOAT64_C( -226.56),
                         EASYSIMD_FLOAT64_C(  374.42), EASYSIMD_FLOAT64_C(  490.85)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  143.45), EASYSIMD_FLOAT64_C(   16.49),
                         EASYSIMD_FLOAT64_C(  323.05), EASYSIMD_FLOAT64_C( -564.38),
                         EASYSIMD_FLOAT64_C( -932.37), EASYSIMD_FLOAT64_C( -126.95),
                         EASYSIMD_FLOAT64_C(   46.50), EASYSIMD_FLOAT64_C(  812.07)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  138.19), EASYSIMD_FLOAT64_C( -758.53),
                         EASYSIMD_FLOAT64_C(  216.18), EASYSIMD_FLOAT64_C( -869.86),
                         EASYSIMD_FLOAT64_C( -556.61), EASYSIMD_FLOAT64_C(  -56.13),
                         EASYSIMD_FLOAT64_C(  935.72), EASYSIMD_FLOAT64_C(  467.65)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -587.51), EASYSIMD_FLOAT64_C( -331.04),
                         EASYSIMD_FLOAT64_C(  711.75), EASYSIMD_FLOAT64_C( -149.95),
                         EASYSIMD_FLOAT64_C( -625.31), EASYSIMD_FLOAT64_C(  387.07),
                         EASYSIMD_FLOAT64_C(  510.51), EASYSIMD_FLOAT64_C( -791.87)),
      UINT8_C(197),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -995.18), EASYSIMD_FLOAT64_C(  720.96),
                         EASYSIMD_FLOAT64_C(  859.59), EASYSIMD_FLOAT64_C(   20.65),
                         EASYSIMD_FLOAT64_C( -207.40), EASYSIMD_FLOAT64_C( -632.30),
                         EASYSIMD_FLOAT64_C( -783.67), EASYSIMD_FLOAT64_C(  389.24)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -619.09), EASYSIMD_FLOAT64_C(  681.55),
                         EASYSIMD_FLOAT64_C(  914.89), EASYSIMD_FLOAT64_C(  240.13),
                         EASYSIMD_FLOAT64_C(   14.06), EASYSIMD_FLOAT64_C( -669.70),
                         EASYSIMD_FLOAT64_C(  554.04), EASYSIMD_FLOAT64_C( -602.80)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -611.02), EASYSIMD_FLOAT64_C(  640.52),
                         EASYSIMD_FLOAT64_C(  711.75), EASYSIMD_FLOAT64_C( -149.95),
                         EASYSIMD_FLOAT64_C( -625.31), EASYSIMD_FLOAT64_C( -536.00),
                         EASYSIMD_FLOAT64_C(  510.51), EASYSIMD_FLOAT64_C(    2.04)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   28.17), EASYSIMD_FLOAT64_C( -545.33),
                         EASYSIMD_FLOAT64_C( -993.85), EASYSIMD_FLOAT64_C( -636.74),
                         EASYSIMD_FLOAT64_C(  315.22), EASYSIMD_FLOAT64_C( -560.48),
                         EASYSIMD_FLOAT64_C( -264.88), EASYSIMD_FLOAT64_C(  866.66)),
      UINT8_C(152),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -378.42), EASYSIMD_FLOAT64_C( -112.43),
                         EASYSIMD_FLOAT64_C( -147.85), EASYSIMD_FLOAT64_C(  481.16),
                         EASYSIMD_FLOAT64_C(  980.68), EASYSIMD_FLOAT64_C(  999.62),
                         EASYSIMD_FLOAT64_C( -784.92), EASYSIMD_FLOAT64_C( -245.05)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  838.26), EASYSIMD_FLOAT64_C( -863.14),
                         EASYSIMD_FLOAT64_C(  336.07), EASYSIMD_FLOAT64_C(  237.32),
                         EASYSIMD_FLOAT64_C( -803.75), EASYSIMD_FLOAT64_C(  816.96),
                         EASYSIMD_FLOAT64_C(  217.54), EASYSIMD_FLOAT64_C( -660.63)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.27), EASYSIMD_FLOAT64_C( -545.33),
                         EASYSIMD_FLOAT64_C( -993.85), EASYSIMD_FLOAT64_C(  224.06),
                         EASYSIMD_FLOAT64_C(  768.50), EASYSIMD_FLOAT64_C( -560.48),
                         EASYSIMD_FLOAT64_C( -264.88), EASYSIMD_FLOAT64_C(  866.66)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  106.10), EASYSIMD_FLOAT64_C(  605.57),
                         EASYSIMD_FLOAT64_C(  481.85), EASYSIMD_FLOAT64_C(  491.86),
                         EASYSIMD_FLOAT64_C(  -77.86), EASYSIMD_FLOAT64_C( -839.61),
                         EASYSIMD_FLOAT64_C(  936.76), EASYSIMD_FLOAT64_C( -659.60)),
      UINT8_C(  7),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  505.82), EASYSIMD_FLOAT64_C( -629.98),
                         EASYSIMD_FLOAT64_C( -555.91), EASYSIMD_FLOAT64_C( -911.21),
                         EASYSIMD_FLOAT64_C(  603.24), EASYSIMD_FLOAT64_C(  -95.72),
                         EASYSIMD_FLOAT64_C(  864.74), EASYSIMD_FLOAT64_C(  280.80)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  570.36), EASYSIMD_FLOAT64_C(  765.47),
                         EASYSIMD_FLOAT64_C(  327.71), EASYSIMD_FLOAT64_C( -605.34),
                         EASYSIMD_FLOAT64_C(  509.13), EASYSIMD_FLOAT64_C( -583.43),
                         EASYSIMD_FLOAT64_C( -208.99), EASYSIMD_FLOAT64_C(  835.11)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  106.10), EASYSIMD_FLOAT64_C(  605.57),
                         EASYSIMD_FLOAT64_C(  481.85), EASYSIMD_FLOAT64_C(  491.86),
                         EASYSIMD_FLOAT64_C(  -77.86), EASYSIMD_FLOAT64_C(   -2.27),
                         EASYSIMD_FLOAT64_C(    3.25), EASYSIMD_FLOAT64_C(    2.00)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -925.64), EASYSIMD_FLOAT64_C(  122.27),
                         EASYSIMD_FLOAT64_C( -971.29), EASYSIMD_FLOAT64_C( -200.64),
                         EASYSIMD_FLOAT64_C(  268.43), EASYSIMD_FLOAT64_C(  995.23),
                         EASYSIMD_FLOAT64_C(  958.62), EASYSIMD_FLOAT64_C( -530.89)),
      UINT8_C(252),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -311.17), EASYSIMD_FLOAT64_C( -787.17),
                         EASYSIMD_FLOAT64_C( -427.34), EASYSIMD_FLOAT64_C(  839.17),
                         EASYSIMD_FLOAT64_C( -404.83), EASYSIMD_FLOAT64_C(  559.72),
                         EASYSIMD_FLOAT64_C(  982.82), EASYSIMD_FLOAT64_C( -251.36)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  643.61), EASYSIMD_FLOAT64_C(  953.53),
                         EASYSIMD_FLOAT64_C( -469.49), EASYSIMD_FLOAT64_C(   -8.31),
                         EASYSIMD_FLOAT64_C(  325.63), EASYSIMD_FLOAT64_C( -753.50),
                         EASYSIMD_FLOAT64_C( -462.28), EASYSIMD_FLOAT64_C( -779.29)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.01), EASYSIMD_FLOAT64_C(  785.01),
                         EASYSIMD_FLOAT64_C( -385.33), EASYSIMD_FLOAT64_C(    2.01),
                         EASYSIMD_FLOAT64_C(  260.50), EASYSIMD_FLOAT64_C(  545.50),
                         EASYSIMD_FLOAT64_C(  958.62), EASYSIMD_FLOAT64_C( -530.89)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -62.82), EASYSIMD_FLOAT64_C(  -95.66),
                         EASYSIMD_FLOAT64_C(  484.39), EASYSIMD_FLOAT64_C( -736.85),
                         EASYSIMD_FLOAT64_C(  893.63), EASYSIMD_FLOAT64_C( -173.06),
                         EASYSIMD_FLOAT64_C(  113.69), EASYSIMD_FLOAT64_C(  198.15)),
      UINT8_C(239),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  440.07), EASYSIMD_FLOAT64_C(  639.74),
                         EASYSIMD_FLOAT64_C(  566.84), EASYSIMD_FLOAT64_C(  207.87),
                         EASYSIMD_FLOAT64_C( -578.31), EASYSIMD_FLOAT64_C( -772.29),
                         EASYSIMD_FLOAT64_C(   70.78), EASYSIMD_FLOAT64_C(  181.63)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -750.95), EASYSIMD_FLOAT64_C(  172.27),
                         EASYSIMD_FLOAT64_C( -538.71), EASYSIMD_FLOAT64_C( -512.10),
                         EASYSIMD_FLOAT64_C( -406.87), EASYSIMD_FLOAT64_C( -470.10),
                         EASYSIMD_FLOAT64_C( -652.40), EASYSIMD_FLOAT64_C( -121.85)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.38), EASYSIMD_FLOAT64_C(    2.19),
                         EASYSIMD_FLOAT64_C(  530.58), EASYSIMD_FLOAT64_C( -736.85),
                         EASYSIMD_FLOAT64_C(   -2.00), EASYSIMD_FLOAT64_C(   -3.02),
                         EASYSIMD_FLOAT64_C(    2.02), EASYSIMD_FLOAT64_C(   44.41)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_and_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_and_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_and_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
   {  UINT16_C(57131),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   399.48), EASYSIMD_FLOAT32_C(  -238.06), EASYSIMD_FLOAT32_C(  -893.32), EASYSIMD_FLOAT32_C(  -435.26),
                         EASYSIMD_FLOAT32_C(   522.86), EASYSIMD_FLOAT32_C(  -612.44), EASYSIMD_FLOAT32_C(   652.00), EASYSIMD_FLOAT32_C(   895.17),
                         EASYSIMD_FLOAT32_C(  -820.93), EASYSIMD_FLOAT32_C(   533.04), EASYSIMD_FLOAT32_C(   403.71), EASYSIMD_FLOAT32_C(   282.24),
                         EASYSIMD_FLOAT32_C(   883.67), EASYSIMD_FLOAT32_C(    22.67), EASYSIMD_FLOAT32_C(   804.53), EASYSIMD_FLOAT32_C(   307.97)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   932.69), EASYSIMD_FLOAT32_C(   912.86), EASYSIMD_FLOAT32_C(   409.21), EASYSIMD_FLOAT32_C(   585.68),
                         EASYSIMD_FLOAT32_C(   -59.99), EASYSIMD_FLOAT32_C(  -146.01), EASYSIMD_FLOAT32_C(   160.06), EASYSIMD_FLOAT32_C(  -248.23),
                         EASYSIMD_FLOAT32_C(   780.27), EASYSIMD_FLOAT32_C(  -642.04), EASYSIMD_FLOAT32_C(   -94.76), EASYSIMD_FLOAT32_C(   563.52),
                         EASYSIMD_FLOAT32_C(  -953.85), EASYSIMD_FLOAT32_C(  -735.06), EASYSIMD_FLOAT32_C(   312.07), EASYSIMD_FLOAT32_C(  -630.77)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     3.02), EASYSIMD_FLOAT32_C(     3.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.25),
                         EASYSIMD_FLOAT32_C(     2.04), EASYSIMD_FLOAT32_C(    -2.25), EASYSIMD_FLOAT32_C(     2.50), EASYSIMD_FLOAT32_C(     3.38),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    68.75), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   817.54), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.13), EASYSIMD_FLOAT32_C(     2.40)) },
   {  UINT16_C(37107),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   145.79), EASYSIMD_FLOAT32_C(  -588.79), EASYSIMD_FLOAT32_C(   895.99), EASYSIMD_FLOAT32_C(  -454.35),
                         EASYSIMD_FLOAT32_C(   444.71), EASYSIMD_FLOAT32_C(   343.63), EASYSIMD_FLOAT32_C(   -33.93), EASYSIMD_FLOAT32_C(  -461.47),
                         EASYSIMD_FLOAT32_C(   -87.51), EASYSIMD_FLOAT32_C(  -587.34), EASYSIMD_FLOAT32_C(   -54.40), EASYSIMD_FLOAT32_C(  -339.84),
                         EASYSIMD_FLOAT32_C(  -976.14), EASYSIMD_FLOAT32_C(   850.15), EASYSIMD_FLOAT32_C(  -700.02), EASYSIMD_FLOAT32_C(  -579.46)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   720.42), EASYSIMD_FLOAT32_C(  -585.67), EASYSIMD_FLOAT32_C(  -388.81), EASYSIMD_FLOAT32_C(   165.49),
                         EASYSIMD_FLOAT32_C(   525.65), EASYSIMD_FLOAT32_C(   441.42), EASYSIMD_FLOAT32_C(   424.69), EASYSIMD_FLOAT32_C(   567.94),
                         EASYSIMD_FLOAT32_C(  -243.26), EASYSIMD_FLOAT32_C(   977.37), EASYSIMD_FLOAT32_C(  -705.87), EASYSIMD_FLOAT32_C(   365.97),
                         EASYSIMD_FLOAT32_C(  -511.37), EASYSIMD_FLOAT32_C(   335.33), EASYSIMD_FLOAT32_C(  -871.52), EASYSIMD_FLOAT32_C(  -805.60)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     2.25), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   161.17),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   -40.75), EASYSIMD_FLOAT32_C(   577.34), EASYSIMD_FLOAT32_C(    -2.25), EASYSIMD_FLOAT32_C(   321.81),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -548.02), EASYSIMD_FLOAT32_C(  -513.07)) },
   {  UINT16_C(56908),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   813.26), EASYSIMD_FLOAT32_C(  -716.74), EASYSIMD_FLOAT32_C(  -548.55), EASYSIMD_FLOAT32_C(   -83.12),
                         EASYSIMD_FLOAT32_C(   301.84), EASYSIMD_FLOAT32_C(  -843.69), EASYSIMD_FLOAT32_C(  -236.76), EASYSIMD_FLOAT32_C(   -34.42),
                         EASYSIMD_FLOAT32_C(  -591.83), EASYSIMD_FLOAT32_C(    11.80), EASYSIMD_FLOAT32_C(   521.39), EASYSIMD_FLOAT32_C(  -937.14),
                         EASYSIMD_FLOAT32_C(  -662.16), EASYSIMD_FLOAT32_C(  -974.03), EASYSIMD_FLOAT32_C(   576.46), EASYSIMD_FLOAT32_C(   704.69)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   914.44), EASYSIMD_FLOAT32_C(  -904.34), EASYSIMD_FLOAT32_C(    -4.84), EASYSIMD_FLOAT32_C(   -59.72),
                         EASYSIMD_FLOAT32_C(  -523.01), EASYSIMD_FLOAT32_C(   236.78), EASYSIMD_FLOAT32_C(    88.72), EASYSIMD_FLOAT32_C(  -251.99),
                         EASYSIMD_FLOAT32_C(  -782.65), EASYSIMD_FLOAT32_C(   -38.86), EASYSIMD_FLOAT32_C(   670.53), EASYSIMD_FLOAT32_C(   706.52),
                         EASYSIMD_FLOAT32_C(   990.40), EASYSIMD_FLOAT32_C(  -812.48), EASYSIMD_FLOAT32_C(  -152.33), EASYSIMD_FLOAT32_C(   172.86)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   768.25), EASYSIMD_FLOAT32_C(  -648.08), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -41.53),
                         EASYSIMD_FLOAT32_C(     2.04), EASYSIMD_FLOAT32_C(     3.01), EASYSIMD_FLOAT32_C(    40.06), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   662.13), EASYSIMD_FLOAT32_C(  -780.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
   {  UINT16_C(13045),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -696.33), EASYSIMD_FLOAT32_C(  -640.98), EASYSIMD_FLOAT32_C(  -234.04), EASYSIMD_FLOAT32_C(   691.30),
                         EASYSIMD_FLOAT32_C(   422.16), EASYSIMD_FLOAT32_C(    -0.53), EASYSIMD_FLOAT32_C(   150.98), EASYSIMD_FLOAT32_C(  -727.93),
                         EASYSIMD_FLOAT32_C(  -292.95), EASYSIMD_FLOAT32_C(  -168.48), EASYSIMD_FLOAT32_C(   430.75), EASYSIMD_FLOAT32_C(   298.75),
                         EASYSIMD_FLOAT32_C(  -938.39), EASYSIMD_FLOAT32_C(   166.50), EASYSIMD_FLOAT32_C(   295.10), EASYSIMD_FLOAT32_C(   -66.94)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   725.67), EASYSIMD_FLOAT32_C(   691.57), EASYSIMD_FLOAT32_C(   408.92), EASYSIMD_FLOAT32_C(  -190.91),
                         EASYSIMD_FLOAT32_C(   682.56), EASYSIMD_FLOAT32_C(   311.99), EASYSIMD_FLOAT32_C(  -213.61), EASYSIMD_FLOAT32_C(  -160.20),
                         EASYSIMD_FLOAT32_C(  -421.91), EASYSIMD_FLOAT32_C(   600.12), EASYSIMD_FLOAT32_C(   657.47), EASYSIMD_FLOAT32_C(   816.91),
                         EASYSIMD_FLOAT32_C(   267.68), EASYSIMD_FLOAT32_C(   898.52), EASYSIMD_FLOAT32_C(   -80.12), EASYSIMD_FLOAT32_C(  -724.23)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   200.00), EASYSIMD_FLOAT32_C(     2.70),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   148.59), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -292.88), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     2.07), EASYSIMD_FLOAT32_C(     2.06),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.51), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.08)) },
   {  UINT16_C(11913),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   315.17), EASYSIMD_FLOAT32_C(  -863.78), EASYSIMD_FLOAT32_C(   344.73), EASYSIMD_FLOAT32_C(  -570.00),
                         EASYSIMD_FLOAT32_C(  -265.79), EASYSIMD_FLOAT32_C(   403.67), EASYSIMD_FLOAT32_C(   -62.80), EASYSIMD_FLOAT32_C(   251.47),
                         EASYSIMD_FLOAT32_C(   143.15), EASYSIMD_FLOAT32_C(   960.55), EASYSIMD_FLOAT32_C(  -156.81), EASYSIMD_FLOAT32_C(   258.89),
                         EASYSIMD_FLOAT32_C(    14.13), EASYSIMD_FLOAT32_C(   117.08), EASYSIMD_FLOAT32_C(  -266.20), EASYSIMD_FLOAT32_C(   383.43)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -568.28), EASYSIMD_FLOAT32_C(  -745.49), EASYSIMD_FLOAT32_C(  -964.75), EASYSIMD_FLOAT32_C(   259.38),
                         EASYSIMD_FLOAT32_C(   750.99), EASYSIMD_FLOAT32_C(  -521.20), EASYSIMD_FLOAT32_C(   513.21), EASYSIMD_FLOAT32_C(   787.79),
                         EASYSIMD_FLOAT32_C(   316.72), EASYSIMD_FLOAT32_C(   -19.08), EASYSIMD_FLOAT32_C(  -845.60), EASYSIMD_FLOAT32_C(   815.31),
                         EASYSIMD_FLOAT32_C(  -301.01), EASYSIMD_FLOAT32_C(   479.36), EASYSIMD_FLOAT32_C(  -159.67), EASYSIMD_FLOAT32_C(  -155.94)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.50), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     2.01), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   142.02), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     8.13), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   155.69)) },
   {  UINT16_C(38742),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -500.21), EASYSIMD_FLOAT32_C(  -899.42), EASYSIMD_FLOAT32_C(   770.51), EASYSIMD_FLOAT32_C(   777.58),
                         EASYSIMD_FLOAT32_C(   547.07), EASYSIMD_FLOAT32_C(   747.18), EASYSIMD_FLOAT32_C(    16.17), EASYSIMD_FLOAT32_C(   859.01),
                         EASYSIMD_FLOAT32_C(    78.72), EASYSIMD_FLOAT32_C(  -378.16), EASYSIMD_FLOAT32_C(  -980.04), EASYSIMD_FLOAT32_C(   143.56),
                         EASYSIMD_FLOAT32_C(  -706.63), EASYSIMD_FLOAT32_C(  -986.84), EASYSIMD_FLOAT32_C(  -673.32), EASYSIMD_FLOAT32_C(  -774.96)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -321.38), EASYSIMD_FLOAT32_C(  -244.51), EASYSIMD_FLOAT32_C(   579.94), EASYSIMD_FLOAT32_C(   895.47),
                         EASYSIMD_FLOAT32_C(  -321.30), EASYSIMD_FLOAT32_C(    92.97), EASYSIMD_FLOAT32_C(  -270.40), EASYSIMD_FLOAT32_C(  -439.43),
                         EASYSIMD_FLOAT32_C(   971.85), EASYSIMD_FLOAT32_C(   799.33), EASYSIMD_FLOAT32_C(   -17.61), EASYSIMD_FLOAT32_C(  -762.15),
                         EASYSIMD_FLOAT32_C(  -813.48), EASYSIMD_FLOAT32_C(   494.42), EASYSIMD_FLOAT32_C(   374.64), EASYSIMD_FLOAT32_C(  -744.47)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -320.13), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   777.06),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.89), EASYSIMD_FLOAT32_C(    16.13), EASYSIMD_FLOAT32_C(     3.29),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.08), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.23),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     3.85), EASYSIMD_FLOAT32_C(     2.63), EASYSIMD_FLOAT32_C(     0.00)) },
   {  UINT16_C(53846),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   400.20), EASYSIMD_FLOAT32_C(   275.50), EASYSIMD_FLOAT32_C(   916.86), EASYSIMD_FLOAT32_C(  -531.67),
                         EASYSIMD_FLOAT32_C(  -909.37), EASYSIMD_FLOAT32_C(   993.65), EASYSIMD_FLOAT32_C(   633.64), EASYSIMD_FLOAT32_C(  -178.42),
                         EASYSIMD_FLOAT32_C(   412.35), EASYSIMD_FLOAT32_C(  -571.03), EASYSIMD_FLOAT32_C(   345.26), EASYSIMD_FLOAT32_C(   493.12),
                         EASYSIMD_FLOAT32_C(  -719.68), EASYSIMD_FLOAT32_C(   769.35), EASYSIMD_FLOAT32_C(  -373.84), EASYSIMD_FLOAT32_C(  -540.22)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -282.19), EASYSIMD_FLOAT32_C(  -584.18), EASYSIMD_FLOAT32_C(   433.06), EASYSIMD_FLOAT32_C(   752.23),
                         EASYSIMD_FLOAT32_C(  -792.10), EASYSIMD_FLOAT32_C(   940.65), EASYSIMD_FLOAT32_C(  -237.54), EASYSIMD_FLOAT32_C(  -796.45),
                         EASYSIMD_FLOAT32_C(   821.11), EASYSIMD_FLOAT32_C(  -769.48), EASYSIMD_FLOAT32_C(   951.19), EASYSIMD_FLOAT32_C(   526.89),
                         EASYSIMD_FLOAT32_C(   481.01), EASYSIMD_FLOAT32_C(  -678.70), EASYSIMD_FLOAT32_C(   690.79), EASYSIMD_FLOAT32_C(  -617.07)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   272.19), EASYSIMD_FLOAT32_C(     2.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   528.17),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.19), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -513.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.04),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   512.07), EASYSIMD_FLOAT32_C(     2.63), EASYSIMD_FLOAT32_C(     0.00)) },
   {  UINT16_C(45516),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -637.18), EASYSIMD_FLOAT32_C(   934.54), EASYSIMD_FLOAT32_C(   112.29), EASYSIMD_FLOAT32_C(   139.60),
                         EASYSIMD_FLOAT32_C(  -371.31), EASYSIMD_FLOAT32_C(  -676.65), EASYSIMD_FLOAT32_C(  -607.44), EASYSIMD_FLOAT32_C(  -108.80),
                         EASYSIMD_FLOAT32_C(  -631.32), EASYSIMD_FLOAT32_C(   553.47), EASYSIMD_FLOAT32_C(  -653.07), EASYSIMD_FLOAT32_C(  -272.71),
                         EASYSIMD_FLOAT32_C(  -438.05), EASYSIMD_FLOAT32_C(   -69.28), EASYSIMD_FLOAT32_C(   220.30), EASYSIMD_FLOAT32_C(  -879.60)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   695.29), EASYSIMD_FLOAT32_C(  -288.10), EASYSIMD_FLOAT32_C(     8.22), EASYSIMD_FLOAT32_C(   267.50),
                         EASYSIMD_FLOAT32_C(  -160.08), EASYSIMD_FLOAT32_C(   251.69), EASYSIMD_FLOAT32_C(   416.95), EASYSIMD_FLOAT32_C(   429.19),
                         EASYSIMD_FLOAT32_C(  -938.09), EASYSIMD_FLOAT32_C(  -996.83), EASYSIMD_FLOAT32_C(   772.01), EASYSIMD_FLOAT32_C(   -88.73),
                         EASYSIMD_FLOAT32_C(  -661.22), EASYSIMD_FLOAT32_C(  -945.44), EASYSIMD_FLOAT32_C(   528.59), EASYSIMD_FLOAT32_C(   677.63)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   565.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     2.01), EASYSIMD_FLOAT32_C(   129.50),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   104.30),
                         EASYSIMD_FLOAT32_C(  -546.07), EASYSIMD_FLOAT32_C(   544.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    -2.02), EASYSIMD_FLOAT32_C(    -2.13), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_and_ps(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_and_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_and_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
   {  UINT8_C( 62),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  337.23), EASYSIMD_FLOAT64_C( -706.51),
                         EASYSIMD_FLOAT64_C(  -51.03), EASYSIMD_FLOAT64_C(  -11.12),
                         EASYSIMD_FLOAT64_C(  780.39), EASYSIMD_FLOAT64_C(  482.32),
                         EASYSIMD_FLOAT64_C( -313.20), EASYSIMD_FLOAT64_C(  986.27)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -459.33), EASYSIMD_FLOAT64_C(  566.75),
                         EASYSIMD_FLOAT64_C(  454.16), EASYSIMD_FLOAT64_C( -566.29),
                         EASYSIMD_FLOAT64_C(  217.01), EASYSIMD_FLOAT64_C( -444.68),
                         EASYSIMD_FLOAT64_C(  725.53), EASYSIMD_FLOAT64_C( -673.17)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(   48.02), EASYSIMD_FLOAT64_C(   -2.02),
                         EASYSIMD_FLOAT64_C(    3.02), EASYSIMD_FLOAT64_C(  416.00),
                         EASYSIMD_FLOAT64_C(    2.31), EASYSIMD_FLOAT64_C(    0.00)) },
   {  UINT8_C(178),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  988.70), EASYSIMD_FLOAT64_C(  952.56),
                         EASYSIMD_FLOAT64_C( -917.57), EASYSIMD_FLOAT64_C( -161.93),
                         EASYSIMD_FLOAT64_C(  553.05), EASYSIMD_FLOAT64_C(  358.83),
                         EASYSIMD_FLOAT64_C( -335.21), EASYSIMD_FLOAT64_C(  243.33)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  368.67), EASYSIMD_FLOAT64_C( -326.68),
                         EASYSIMD_FLOAT64_C( -767.44), EASYSIMD_FLOAT64_C( -965.45),
                         EASYSIMD_FLOAT64_C(  160.34), EASYSIMD_FLOAT64_C( -153.49),
                         EASYSIMD_FLOAT64_C(  842.87), EASYSIMD_FLOAT64_C( -959.77)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.75), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C( -661.06), EASYSIMD_FLOAT64_C(   -2.52),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    2.04), EASYSIMD_FLOAT64_C(    0.00)) },
   {  UINT8_C(233),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  789.99), EASYSIMD_FLOAT64_C(   -0.83),
                         EASYSIMD_FLOAT64_C( -595.87), EASYSIMD_FLOAT64_C( -556.04),
                         EASYSIMD_FLOAT64_C( -673.58), EASYSIMD_FLOAT64_C(  820.52),
                         EASYSIMD_FLOAT64_C(  763.24), EASYSIMD_FLOAT64_C(  747.54)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -863.51), EASYSIMD_FLOAT64_C(  297.93),
                         EASYSIMD_FLOAT64_C(  664.70), EASYSIMD_FLOAT64_C(   43.00),
                         EASYSIMD_FLOAT64_C(  283.69), EASYSIMD_FLOAT64_C( -882.73),
                         EASYSIMD_FLOAT64_C(   56.70), EASYSIMD_FLOAT64_C( -683.31)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  789.50), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  528.57), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    2.13), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  683.04)) },
   {  UINT8_C( 29),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -942.62), EASYSIMD_FLOAT64_C(  810.42),
                         EASYSIMD_FLOAT64_C( -781.08), EASYSIMD_FLOAT64_C(  565.31),
                         EASYSIMD_FLOAT64_C( -528.23), EASYSIMD_FLOAT64_C( -642.03),
                         EASYSIMD_FLOAT64_C( -124.04), EASYSIMD_FLOAT64_C(  -13.65)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   20.85), EASYSIMD_FLOAT64_C( -517.24),
                         EASYSIMD_FLOAT64_C(  -21.32), EASYSIMD_FLOAT64_C(  729.98),
                         EASYSIMD_FLOAT64_C( -763.15), EASYSIMD_FLOAT64_C(  885.38),
                         EASYSIMD_FLOAT64_C(  783.63), EASYSIMD_FLOAT64_C(  470.19)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  529.29),
                         EASYSIMD_FLOAT64_C( -528.13), EASYSIMD_FLOAT64_C(  512.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   12.63)) },
   {  UINT8_C(102),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -36.60), EASYSIMD_FLOAT64_C(  -71.02),
                         EASYSIMD_FLOAT64_C(  654.55), EASYSIMD_FLOAT64_C( -335.18),
                         EASYSIMD_FLOAT64_C( -889.86), EASYSIMD_FLOAT64_C( -624.64),
                         EASYSIMD_FLOAT64_C(  369.01), EASYSIMD_FLOAT64_C(  798.90)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -217.14), EASYSIMD_FLOAT64_C( -334.27),
                         EASYSIMD_FLOAT64_C(  522.28), EASYSIMD_FLOAT64_C(  754.78),
                         EASYSIMD_FLOAT64_C( -987.63), EASYSIMD_FLOAT64_C(  746.58),
                         EASYSIMD_FLOAT64_C(  358.61), EASYSIMD_FLOAT64_C( -154.14)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  -67.00),
                         EASYSIMD_FLOAT64_C(  522.02), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  608.50),
                         EASYSIMD_FLOAT64_C(  352.00), EASYSIMD_FLOAT64_C(    0.00)) },
   {  UINT8_C(126),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -16.31), EASYSIMD_FLOAT64_C(  -95.18),
                         EASYSIMD_FLOAT64_C(  860.06), EASYSIMD_FLOAT64_C(  464.41),
                         EASYSIMD_FLOAT64_C(  822.39), EASYSIMD_FLOAT64_C(  185.79),
                         EASYSIMD_FLOAT64_C(  959.83), EASYSIMD_FLOAT64_C(  -98.41)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -560.42), EASYSIMD_FLOAT64_C( -521.57),
                         EASYSIMD_FLOAT64_C( -947.45), EASYSIMD_FLOAT64_C(   99.55),
                         EASYSIMD_FLOAT64_C(  108.53), EASYSIMD_FLOAT64_C(  194.26),
                         EASYSIMD_FLOAT64_C(  449.89), EASYSIMD_FLOAT64_C(  718.27)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   -2.04),
                         EASYSIMD_FLOAT64_C(  784.01), EASYSIMD_FLOAT64_C(   96.03),
                         EASYSIMD_FLOAT64_C(    3.14), EASYSIMD_FLOAT64_C(  128.26),
                         EASYSIMD_FLOAT64_C(    3.51), EASYSIMD_FLOAT64_C(    0.00)) },
   {  UINT8_C(231),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -343.46), EASYSIMD_FLOAT64_C(  643.05),
                         EASYSIMD_FLOAT64_C(  758.23), EASYSIMD_FLOAT64_C(  243.41),
                         EASYSIMD_FLOAT64_C( -569.27), EASYSIMD_FLOAT64_C(   62.99),
                         EASYSIMD_FLOAT64_C(  403.36), EASYSIMD_FLOAT64_C( -111.26)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  594.69), EASYSIMD_FLOAT64_C(  416.92),
                         EASYSIMD_FLOAT64_C(  294.94), EASYSIMD_FLOAT64_C( -386.69),
                         EASYSIMD_FLOAT64_C(  444.27), EASYSIMD_FLOAT64_C(  112.48),
                         EASYSIMD_FLOAT64_C(  775.25), EASYSIMD_FLOAT64_C(  973.66)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.01), EASYSIMD_FLOAT64_C(    2.00),
                         EASYSIMD_FLOAT64_C(    2.27), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   56.24),
                         EASYSIMD_FLOAT64_C(    3.02), EASYSIMD_FLOAT64_C(    3.28)) },
   {  UINT8_C(248),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  559.94), EASYSIMD_FLOAT64_C( -177.36),
                         EASYSIMD_FLOAT64_C(  459.52), EASYSIMD_FLOAT64_C(  151.00),
                         EASYSIMD_FLOAT64_C( -261.20), EASYSIMD_FLOAT64_C(  619.75),
                         EASYSIMD_FLOAT64_C( -541.43), EASYSIMD_FLOAT64_C( -420.37)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  279.51), EASYSIMD_FLOAT64_C(  143.05),
                         EASYSIMD_FLOAT64_C(  835.37), EASYSIMD_FLOAT64_C( -486.11),
                         EASYSIMD_FLOAT64_C(  461.53), EASYSIMD_FLOAT64_C(  410.57),
                         EASYSIMD_FLOAT64_C( -362.30), EASYSIMD_FLOAT64_C( -345.54)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.18), EASYSIMD_FLOAT64_C(  129.05),
                         EASYSIMD_FLOAT64_C(    3.01), EASYSIMD_FLOAT64_C(  147.00),
                         EASYSIMD_FLOAT64_C(  261.01), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_and_pd(test_vec[i].k, test_vec[i].a, test_vec[i].b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_and_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_and_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 2103907232), INT32_C(-1995421302), INT32_C( 1328084931), INT32_C( -379562245),
                            INT32_C( 1144599747), INT32_C(-1418414219), INT32_C( 1379143176), INT32_C(-2075387410),
                            INT32_C(-1152868472), INT32_C( 1425101887), INT32_C(-1626225579), INT32_C( 2014677400),
                            INT32_C( 1108777022), INT32_C(  877429697), INT32_C( 1558575022), INT32_C(  651318578)),
      easysimd_mm512_set_epi32(INT32_C( -803993576), INT32_C(  163813867), INT32_C(-1017884831), INT32_C(-1258992440),
                            INT32_C( 1182354404), INT32_C(-1425047283), INT32_C( 1708628167), INT32_C(-2051115159),
                            INT32_C(  143410800), INT32_C(-1041180848), INT32_C(-1023702255), INT32_C( -240255997),
                            INT32_C(  978107452), INT32_C(-1343783755), INT32_C( -880122145), INT32_C( 1183808039)),
      easysimd_mm512_set_epi32(INT32_C( 1342439424), INT32_C(  150997386), INT32_C( 1124091713), INT32_C(-1604303672),
                            INT32_C( 1144586432), INT32_C(-1425768187), INT32_C( 1075054592), INT32_C(-2079583896),
                            INT32_C(  134743552), INT32_C( 1089491984), INT32_C(-2112782319), INT32_C( 1879410688),
                            INT32_C(   33855548), INT32_C(  608436353), INT32_C( 1216374414), INT32_C(  109203490)) },
    { easysimd_mm512_set_epi32(INT32_C( 1231278072), INT32_C(  832865002), INT32_C(-1694490420), INT32_C( -466764866),
                            INT32_C( 1702721236), INT32_C( 2092439418), INT32_C(  611933067), INT32_C(-1184445720),
                            INT32_C(-1689922195), INT32_C(-1140532352), INT32_C(  158346875), INT32_C( 1089717474),
                            INT32_C( 1230253618), INT32_C( 1504968720), INT32_C(  653725328), INT32_C( 1822881045)),
      easysimd_mm512_set_epi32(INT32_C(-1605590998), INT32_C( 1810066171), INT32_C(  -10167007), INT32_C( 1359569321),
                            INT32_C(-1430119757), INT32_C(  256064867), INT32_C(  788618356), INT32_C( -392789188),
                            INT32_C( -613873031), INT32_C(  773158597), INT32_C( -194106681), INT32_C(  165465349),
                            INT32_C(-1346434037), INT32_C(-1412186885), INT32_C( -632750822), INT32_C( 1966832804)),
      easysimd_mm512_set_epi32(INT32_C(    4227112), INT32_C(  564134634), INT32_C(-1694498816), INT32_C( 1074340264),
                            INT32_C(  541069968), INT32_C(  201332066), INT32_C(  604067840), INT32_C(-1475968472),
                            INT32_C(-1690238871), INT32_C(  738480768), INT32_C(    6302275), INT32_C(   13681664),
                            INT32_C(  152308738), INT32_C(  160432144), INT32_C(   37751824), INT32_C( 1679979524)) },
    { easysimd_mm512_set_epi32(INT32_C(-1888865381), INT32_C(   15541452), INT32_C( -670396349), INT32_C(-1090081489),
                            INT32_C(  116025329), INT32_C(  130963716), INT32_C(  230354364), INT32_C( 1174065929),
                            INT32_C( 1971493681), INT32_C(-1343257591), INT32_C(-1419733408), INT32_C(  659096905),
                            INT32_C(  183711411), INT32_C( -298263182), INT32_C(-1286938208), INT32_C(  537660993)),
      easysimd_mm512_set_epi32(INT32_C( 1616282233), INT32_C(-2132390836), INT32_C( -153917613), INT32_C(  736513734),
                            INT32_C(-1311057544), INT32_C(  505476811), INT32_C( 1767695145), INT32_C(  157469724),
                            INT32_C( -371725260), INT32_C( 1996701751), INT32_C(-1377678442), INT32_C( -132601652),
                            INT32_C( -903948497), INT32_C( -794660034), INT32_C(  173952757), INT32_C( 1507611872)),
      easysimd_mm512_set_epi32(INT32_C(    4325913), INT32_C(   14943308), INT32_C( -805175229), INT32_C(  721816582),
                            INT32_C(   13255024), INT32_C(  100683776), INT32_C(  152625448), INT32_C(   23248904),
                            INT32_C( 1635944496), INT32_C(  654508033), INT32_C(-1453325824), INT32_C(  537395272),
                            INT32_C(  168957475), INT32_C(-1071623886), INT32_C(   38420640), INT32_C(     787520)) },
    { easysimd_mm512_set_epi32(INT32_C( -748350470), INT32_C( 1755197901), INT32_C( 1090059253), INT32_C( 1329426651),
                            INT32_C(-1604442789), INT32_C(  878047098), INT32_C(-1682276633), INT32_C(  -78811559),
                            INT32_C( -973139496), INT32_C(   91517188), INT32_C(   37440120), INT32_C(  829401648),
                            INT32_C(  860279707), INT32_C(  218373799), INT32_C( -362113249), INT32_C( -694086277)),
      easysimd_mm512_set_epi32(INT32_C(  892006932), INT32_C(-1158010747), INT32_C( -700127305), INT32_C(-1058033333),
                            INT32_C( 2135825335), INT32_C( -510043422), INT32_C( -139544800), INT32_C(   95748631),
                            INT32_C( 1671238992), INT32_C( 1305915968), INT32_C( -582790199), INT32_C( 1758236157),
                            INT32_C(  986405639), INT32_C( 1121234438), INT32_C( -731393712), INT32_C( -784753228)),
      easysimd_mm512_set_epi32(INT32_C(  287314448), INT32_C(  681189509), INT32_C( 1077994421), INT32_C( 1076703307),
                            INT32_C(  541982995), INT32_C(  538004066), INT32_C(-1817561568), INT32_C(   17105425),
                            INT32_C( 1100812624), INT32_C(   89403392), INT32_C(     213576), INT32_C(  541886512),
                            INT32_C(  843207427), INT32_C(     265734), INT32_C(-1067282160), INT32_C( -803138256)) },
    { easysimd_mm512_set_epi32(INT32_C( 1347900829), INT32_C(-1792354715), INT32_C( -371177698), INT32_C(  255088013),
                            INT32_C( 1961231505), INT32_C( -659343095), INT32_C( 1620234692), INT32_C(  843561067),
                            INT32_C( 1265300992), INT32_C(-1675104490), INT32_C( -873664156), INT32_C(-2045109653),
                            INT32_C( 2057630636), INT32_C(  335188274), INT32_C( 1272591061), INT32_C( -327494197)),
      easysimd_mm512_set_epi32(INT32_C(-2095740678), INT32_C(-1857753563), INT32_C(-1236342636), INT32_C( 1439297909),
                            INT32_C( -576201057), INT32_C( 1488873085), INT32_C(-1369304746), INT32_C(  567848046),
                            INT32_C( 1335236564), INT32_C( -942680632), INT32_C(-1512916560), INT32_C( -697747292),
                            INT32_C(-1779695782), INT32_C(  491327584), INT32_C( -563681080), INT32_C( 1527319596)),
      easysimd_mm512_set_epi32(INT32_C(    1376408), INT32_C(-1862220251), INT32_C(-1606402028), INT32_C(   83906821),
                            INT32_C( 1420166289), INT32_C( 1488068617), INT32_C(  537006404), INT32_C(  541108330),
                            INT32_C( 1258427904), INT32_C(-2079865600), INT32_C(-2118078176), INT32_C(-2046289888),
                            INT32_C(  278983432), INT32_C(  289935392), INT32_C( 1245847744), INT32_C( 1208484872)) },
    { easysimd_mm512_set_epi32(INT32_C(  131205926), INT32_C( 2061955170), INT32_C(   37003574), INT32_C( 1649229141),
                            INT32_C(  612060260), INT32_C(-1402263233), INT32_C( -513572270), INT32_C(  701923816),
                            INT32_C(  511549547), INT32_C(  969083331), INT32_C( 1364542630), INT32_C( -822209230),
                            INT32_C(-1549704264), INT32_C(-1157339218), INT32_C(-2025137124), INT32_C(-1631723043)),
      easysimd_mm512_set_epi32(INT32_C(  424825857), INT32_C(  434716327), INT32_C( 1663095683), INT32_C( 1730428966),
                            INT32_C( -252943126), INT32_C(  373990324), INT32_C( 2100741912), INT32_C(-1005385937),
                            INT32_C(-1470990839), INT32_C(-1200692008), INT32_C(-2098508971), INT32_C(  488824783),
                            INT32_C( 1647236603), INT32_C(-1221154150), INT32_C(-1012750616), INT32_C( 1849794587)),
      easysimd_mm512_set_epi32(INT32_C(   22151168), INT32_C(  417345570), INT32_C(   35684610), INT32_C( 1644429316),
                            INT32_C(  543704160), INT32_C(   71967028), INT32_C( 1629651472), INT32_C(    1179944),
                            INT32_C(  139468809), INT32_C(  943849664), INT32_C(    4276228), INT32_C(  203560194),
                            INT32_C(  572539320), INT32_C(-1291574646), INT32_C(-2096987128), INT32_C(  234987545)) },
    { easysimd_mm512_set_epi32(INT32_C(-2080136983), INT32_C( -623547588), INT32_C( 1015056564), INT32_C(-1333355305),
                            INT32_C( 1925062912), INT32_C(  330138155), INT32_C( -444195598), INT32_C(  874806560),
                            INT32_C(  319126943), INT32_C(  475403370), INT32_C( -873396634), INT32_C(-1835948135),
                            INT32_C(-1570208244), INT32_C( -244400530), INT32_C( 1126824505), INT32_C( 1036340167)),
      easysimd_mm512_set_epi32(INT32_C(-1989715102), INT32_C( 1785805415), INT32_C( 1739507553), INT32_C(-1932540399),
                            INT32_C( -641363523), INT32_C(   72499486), INT32_C(-2005610208), INT32_C( -497730582),
                            INT32_C(  312611747), INT32_C(  793357988), INT32_C(  607421194), INT32_C(-1375787856),
                            INT32_C( 1799684145), INT32_C( -551549754), INT32_C( 1249951615), INT32_C( 1447553529)),
      easysimd_mm512_set_epi32(INT32_C(-2147286944), INT32_C( 1246833188), INT32_C(  612401184), INT32_C(-2138668015),
                            INT32_C( 1350828288), INT32_C(         10), INT32_C(-2147217376), INT32_C(  537147680),
                            INT32_C(  301994371), INT32_C(  205521952), INT32_C(    3146242), INT32_C(-2137971568),
                            INT32_C(  574619648), INT32_C( -786430906), INT32_C( 1107343929), INT32_C(  340085185)) },
    { easysimd_mm512_set_epi32(INT32_C(  423295425), INT32_C( -460615607), INT32_C( 1208771148), INT32_C(-2128303155),
                            INT32_C( -738338972), INT32_C( 2110676823), INT32_C(-1405320678), INT32_C(-2007459833),
                            INT32_C( 1043638626), INT32_C( -542891463), INT32_C(  629803756), INT32_C(-1216921331),
                            INT32_C( -301860714), INT32_C(  317296385), INT32_C( 1833800187), INT32_C( -645353377)),
      easysimd_mm512_set_epi32(INT32_C( -257851255), INT32_C( -479522767), INT32_C(  745275629), INT32_C(-1783480446),
                            INT32_C(-1431666964), INT32_C( -422291816), INT32_C( -309252994), INT32_C(-2112946871),
                            INT32_C(-2123521230), INT32_C( -883174176), INT32_C( -332498316), INT32_C(  142889340),
                            INT32_C( 1231002435), INT32_C( 1681334055), INT32_C(  138372594), INT32_C( -878588971)),
      easysimd_mm512_set_epi32(INT32_C(  270565505), INT32_C( -536145919), INT32_C(  135004236), INT32_C(-2128598144),
                            INT32_C(-2102897052), INT32_C( 1690588688), INT32_C(-1408237542), INT32_C(-2146920447),
                            INT32_C(    2400546), INT32_C( -889192416), INT32_C(  604637796), INT32_C(     278796),
                            INT32_C( 1208064002), INT32_C(    2166529), INT32_C(  135070194), INT32_C( -914325419)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_and_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_and_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_and_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 2103907232), INT32_C(-1995421302), INT32_C( 1328084931), INT32_C( -379562245),
                            INT32_C( 1144599747), INT32_C(-1418414219), INT32_C( 1379143176), INT32_C(-2075387410),
                            INT32_C(-1152868472), INT32_C( 1425101887), INT32_C(-1626225579), INT32_C( 2014677400),
                            INT32_C( 1108777022), INT32_C(  877429697), INT32_C( 1558575022), INT32_C(  651318578)),
      UINT16_C(31271),
      easysimd_mm512_set_epi32(INT32_C( 1822881045), INT32_C( -803993576), INT32_C(  163813867), INT32_C(-1017884831),
                            INT32_C(-1258992440), INT32_C( 1182354404), INT32_C(-1425047283), INT32_C( 1708628167),
                            INT32_C(-2051115159), INT32_C(  143410800), INT32_C(-1041180848), INT32_C(-1023702255),
                            INT32_C( -240255997), INT32_C(  978107452), INT32_C(-1343783755), INT32_C( -880122145)),
      easysimd_mm512_set_epi32(INT32_C( 1966832804), INT32_C( 1231278072), INT32_C(  832865002), INT32_C(-1694490420),
                            INT32_C( -466764866), INT32_C( 1702721236), INT32_C( 2092439418), INT32_C(  611933067),
                            INT32_C(-1184445720), INT32_C(-1689922195), INT32_C(-1140532352), INT32_C(  158346875),
                            INT32_C( 1089717474), INT32_C( 1230253618), INT32_C( 1504968720), INT32_C(  653725328)),
      easysimd_mm512_set_epi32(INT32_C( 2103907232), INT32_C( 1073743896), INT32_C(   25198826), INT32_C(-2097151936),
                            INT32_C(-1541076856), INT32_C(-1418414219), INT32_C(  671613192), INT32_C(-2075387410),
                            INT32_C(-1152868472), INT32_C( 1425101887), INT32_C(-2147429632), INT32_C( 2014677400),
                            INT32_C( 1108777022), INT32_C(  138685488), INT32_C(  161742864), INT32_C(   42075792)) },
    { easysimd_mm512_set_epi32(INT32_C(  537660993), INT32_C(-1605590998), INT32_C( 1810066171), INT32_C(  -10167007),
                            INT32_C( 1359569321), INT32_C(-1430119757), INT32_C(  256064867), INT32_C(  788618356),
                            INT32_C( -392789188), INT32_C( -613873031), INT32_C(  773158597), INT32_C( -194106681),
                            INT32_C(  165465349), INT32_C(-1346434037), INT32_C(-1412186885), INT32_C( -632750822)),
      UINT16_C(57760),
      easysimd_mm512_set_epi32(INT32_C(  173952757), INT32_C( 1507611872), INT32_C(-1888865381), INT32_C(   15541452),
                            INT32_C( -670396349), INT32_C(-1090081489), INT32_C(  116025329), INT32_C(  130963716),
                            INT32_C(  230354364), INT32_C( 1174065929), INT32_C( 1971493681), INT32_C(-1343257591),
                            INT32_C(-1419733408), INT32_C(  659096905), INT32_C(  183711411), INT32_C( -298263182)),
      easysimd_mm512_set_epi32(INT32_C( -362113249), INT32_C( -694086277), INT32_C( 1616282233), INT32_C(-2132390836),
                            INT32_C( -153917613), INT32_C(  736513734), INT32_C(-1311057544), INT32_C(  505476811),
                            INT32_C( 1767695145), INT32_C(  157469724), INT32_C( -371725260), INT32_C( 1996701751),
                            INT32_C(-1377678442), INT32_C( -132601652), INT32_C( -903948497), INT32_C( -794660034)),
      easysimd_mm512_set_epi32(INT32_C(  172623381), INT32_C( 1350571104), INT32_C(    4325913), INT32_C(  -10167007),
                            INT32_C( 1359569321), INT32_C(-1430119757), INT32_C(  256064867), INT32_C(  100683776),
                            INT32_C(  152625448), INT32_C( -613873031), INT32_C( 1635944496), INT32_C( -194106681),
                            INT32_C(  165465349), INT32_C(-1346434037), INT32_C(-1412186885), INT32_C( -632750822)) },
    { easysimd_mm512_set_epi32(INT32_C( -731393712), INT32_C( -784753228), INT32_C( -748350470), INT32_C( 1755197901),
                            INT32_C( 1090059253), INT32_C( 1329426651), INT32_C(-1604442789), INT32_C(  878047098),
                            INT32_C(-1682276633), INT32_C(  -78811559), INT32_C( -973139496), INT32_C(   91517188),
                            INT32_C(   37440120), INT32_C(  829401648), INT32_C(  860279707), INT32_C(  218373799)),
      UINT16_C(44550),
      easysimd_mm512_set_epi32(INT32_C(  335188274), INT32_C( 1272591061), INT32_C( -327494197), INT32_C(  892006932),
                            INT32_C(-1158010747), INT32_C( -700127305), INT32_C(-1058033333), INT32_C( 2135825335),
                            INT32_C( -510043422), INT32_C( -139544800), INT32_C(   95748631), INT32_C( 1671238992),
                            INT32_C( 1305915968), INT32_C( -582790199), INT32_C( 1758236157), INT32_C(  986405639)),
      easysimd_mm512_set_epi32(INT32_C(  491327584), INT32_C( -563681080), INT32_C( 1527319596), INT32_C( 1347900829),
                            INT32_C(-1792354715), INT32_C( -371177698), INT32_C(  255088013), INT32_C( 1961231505),
                            INT32_C( -659343095), INT32_C( 1620234692), INT32_C(  843561067), INT32_C( 1265300992),
                            INT32_C(-1675104490), INT32_C( -873664156), INT32_C(-2045109653), INT32_C( 2057630636)),
      easysimd_mm512_set_epi32(INT32_C(  289935392), INT32_C( -784753228), INT32_C( 1208484872), INT32_C( 1755197901),
                            INT32_C(-1876295675), INT32_C(-1069529322), INT32_C(    2363657), INT32_C(  878047098),
                            INT32_C(-1682276633), INT32_C(  -78811559), INT32_C( -973139496), INT32_C(   91517188),
                            INT32_C(   37440120), INT32_C( -918531776), INT32_C(     526441), INT32_C(  218373799)) },
    { easysimd_mm512_set_epi32(INT32_C(-1157339218), INT32_C(-2025137124), INT32_C(-1631723043), INT32_C(-2095740678),
                            INT32_C(-1857753563), INT32_C(-1236342636), INT32_C( 1439297909), INT32_C( -576201057),
                            INT32_C( 1488873085), INT32_C(-1369304746), INT32_C(  567848046), INT32_C( 1335236564),
                            INT32_C( -942680632), INT32_C(-1512916560), INT32_C( -697747292), INT32_C(-1779695782)),
      UINT16_C(25528),
      easysimd_mm512_set_epi32(INT32_C( 1647236603), INT32_C(-1221154150), INT32_C(-1012750616), INT32_C( 1849794587),
                            INT32_C(  131205926), INT32_C( 2061955170), INT32_C(   37003574), INT32_C( 1649229141),
                            INT32_C(  612060260), INT32_C(-1402263233), INT32_C( -513572270), INT32_C(  701923816),
                            INT32_C(  511549547), INT32_C(  969083331), INT32_C( 1364542630), INT32_C( -822209230)),
      easysimd_mm512_set_epi32(INT32_C(-1570208244), INT32_C( -244400530), INT32_C( 1126824505), INT32_C( 1036340167),
                            INT32_C(  424825857), INT32_C(  434716327), INT32_C( 1663095683), INT32_C( 1730428966),
                            INT32_C( -252943126), INT32_C(  373990324), INT32_C( 2100741912), INT32_C(-1005385937),
                            INT32_C(-1470990839), INT32_C(-1200692008), INT32_C(-2098508971), INT32_C(  488824783)),
      easysimd_mm512_set_epi32(INT32_C(-1157339218), INT32_C(-1322866166), INT32_C( 1126212136), INT32_C(-2095740678),
                            INT32_C(-1857753563), INT32_C(-1236342636), INT32_C(   35684610), INT32_C( 1644429316),
                            INT32_C(  543704160), INT32_C(-1369304746), INT32_C( 1629651472), INT32_C(    1179944),
                            INT32_C(  139468809), INT32_C(-1512916560), INT32_C( -697747292), INT32_C(-1779695782)) },
    { easysimd_mm512_set_epi32(INT32_C( 1799684145), INT32_C( -551549754), INT32_C( 1249951615), INT32_C( 1447553529),
                            INT32_C(-2080136983), INT32_C( -623547588), INT32_C( 1015056564), INT32_C(-1333355305),
                            INT32_C( 1925062912), INT32_C(  330138155), INT32_C( -444195598), INT32_C(  874806560),
                            INT32_C(  319126943), INT32_C(  475403370), INT32_C( -873396634), INT32_C(-1835948135)),
      UINT16_C( 9392),
      easysimd_mm512_set_epi32(INT32_C(-1216921331), INT32_C( -301860714), INT32_C(  317296385), INT32_C( 1833800187),
                            INT32_C( -645353377), INT32_C(-1989715102), INT32_C( 1785805415), INT32_C( 1739507553),
                            INT32_C(-1932540399), INT32_C( -641363523), INT32_C(   72499486), INT32_C(-2005610208),
                            INT32_C( -497730582), INT32_C(  312611747), INT32_C(  793357988), INT32_C(  607421194)),
      easysimd_mm512_set_epi32(INT32_C(  142889340), INT32_C( 1231002435), INT32_C( 1681334055), INT32_C(  138372594),
                            INT32_C( -878588971), INT32_C(  423295425), INT32_C( -460615607), INT32_C( 1208771148),
                            INT32_C(-2128303155), INT32_C( -738338972), INT32_C( 2110676823), INT32_C(-1405320678),
                            INT32_C(-2007459833), INT32_C( 1043638626), INT32_C( -542891463), INT32_C(  629803756)),
      easysimd_mm512_set_epi32(INT32_C( 1799684145), INT32_C( -551549754), INT32_C(    2166529), INT32_C( 1447553529),
                            INT32_C(-2080136983), INT32_C(  153246016), INT32_C( 1015056564), INT32_C(-1333355305),
                            INT32_C(-2147188223), INT32_C(  330138155), INT32_C(   71450902), INT32_C(-2009825280),
                            INT32_C(  319126943), INT32_C(  475403370), INT32_C( -873396634), INT32_C(-1835948135)) },
    { easysimd_mm512_set_epi32(INT32_C(  861635987), INT32_C( 1823839521), INT32_C( 1391000031), INT32_C(   73229946),
                            INT32_C(  -53693878), INT32_C( -257851255), INT32_C( -479522767), INT32_C(  745275629),
                            INT32_C(-1783480446), INT32_C(-1431666964), INT32_C( -422291816), INT32_C( -309252994),
                            INT32_C(-2112946871), INT32_C(-2123521230), INT32_C( -883174176), INT32_C( -332498316)),
      UINT16_C(31381),
      easysimd_mm512_set_epi32(INT32_C( -410707923), INT32_C( -804790801), INT32_C( -675940069), INT32_C(  717543141),
                            INT32_C( 1610339352), INT32_C(  785451213), INT32_C(  -67248356), INT32_C(-1147482606),
                            INT32_C(  877778312), INT32_C( 1833609670), INT32_C( 1105011960), INT32_C(-1909564752),
                            INT32_C(-2137129603), INT32_C(-1991115340), INT32_C( -941377596), INT32_C(-1151664921)),
      easysimd_mm512_set_epi32(INT32_C( -605598510), INT32_C( 1332169075), INT32_C(  829771204), INT32_C(  806631323),
                            INT32_C(  -62111889), INT32_C( 1452741835), INT32_C(  921236435), INT32_C(-1348081811),
                            INT32_C( -883327193), INT32_C(-1324808596), INT32_C(-2119312832), INT32_C(-1371509978),
                            INT32_C(-1887676953), INT32_C(   42335263), INT32_C(  818544934), INT32_C(-1864687690)),
      easysimd_mm512_set_epi32(INT32_C(  861635987), INT32_C( 1074219363), INT32_C(  288704768), INT32_C(  537138817),
                            INT32_C( 1548227592), INT32_C( -257851255), INT32_C(  854119184), INT32_C(  745275629),
                            INT32_C(    5330176), INT32_C(-1431666964), INT32_C( -422291816), INT32_C(-1912583648),
                            INT32_C(-2112946871), INT32_C(     130068), INT32_C( -883174176), INT32_C(-1873142618)) },
    { easysimd_mm512_set_epi32(INT32_C( 1456151906), INT32_C( -346366427), INT32_C(  534496658), INT32_C( 1981510934),
                            INT32_C( -935678271), INT32_C( 1523008579), INT32_C(   -6105095), INT32_C( 2115600842),
                            INT32_C( -420343454), INT32_C(  652783640), INT32_C( -871055383), INT32_C(  142253075),
                            INT32_C(  557825344), INT32_C(  707825888), INT32_C(  944883191), INT32_C( 1704858885)),
      UINT16_C(19039),
      easysimd_mm512_set_epi32(INT32_C( 1893303454), INT32_C( 1567616976), INT32_C( 1190892677), INT32_C( 1594451864),
                            INT32_C(-1033342432), INT32_C( -738674203), INT32_C(-1847547828), INT32_C( 1893640833),
                            INT32_C(   26320713), INT32_C( 1830669951), INT32_C( 1304924639), INT32_C( -277717409),
                            INT32_C(-1566722863), INT32_C( 1534951086), INT32_C( -925669609), INT32_C(  359322092)),
      easysimd_mm512_set_epi32(INT32_C( 1160904262), INT32_C(  732990033), INT32_C(  138388028), INT32_C( 1168180194),
                            INT32_C( 1057944486), INT32_C( 1875512725), INT32_C( -910818137), INT32_C( -743685110),
                            INT32_C( -947775444), INT32_C( 1458691146), INT32_C( 1273454073), INT32_C(  927814838),
                            INT32_C(-1840329583), INT32_C( -789758267), INT32_C( -870216121), INT32_C(  449327093)),
      easysimd_mm512_set_epi32(INT32_C( 1456151906), INT32_C(  153127504), INT32_C(  534496658), INT32_C( 1981510934),
                            INT32_C(   34107936), INT32_C( 1523008579), INT32_C(-2120220668), INT32_C( 2115600842),
                            INT32_C( -420343454), INT32_C( 1142017098), INT32_C( -871055383), INT32_C(  658527254),
                            INT32_C(-2113108847), INT32_C( 1349336708), INT32_C( -939457017), INT32_C(  273154532)) },
    { easysimd_mm512_set_epi32(INT32_C(-1055194531), INT32_C( 1846727705), INT32_C(  335680535), INT32_C( -610713755),
                            INT32_C(  944256620), INT32_C(  697979892), INT32_C(   49552843), INT32_C( -460412596),
                            INT32_C(-2060335241), INT32_C(  135497979), INT32_C( -331098630), INT32_C( -140680021),
                            INT32_C(-1676162464), INT32_C(  626483741), INT32_C(  170885439), INT32_C(  230851400)),
      UINT16_C(10528),
      easysimd_mm512_set_epi32(INT32_C(-1447071985), INT32_C( -310464227), INT32_C( -679161042), INT32_C( -527096592),
                            INT32_C( 1833269922), INT32_C(-1164990327), INT32_C( 1955493691), INT32_C( 1263046717),
                            INT32_C( 1122048689), INT32_C( 2074234443), INT32_C(  723669938), INT32_C(  284884896),
                            INT32_C( -458264538), INT32_C(  514387150), INT32_C(-1369468153), INT32_C( 1859652102)),
      easysimd_mm512_set_epi32(INT32_C(-1315612420), INT32_C( -330188185), INT32_C(     890374), INT32_C( -958458643),
                            INT32_C( 2023887571), INT32_C(  226435011), INT32_C(    1929100), INT32_C(-1975437469),
                            INT32_C(  871389437), INT32_C(-2041059805), INT32_C(  636566673), INT32_C( -485819928),
                            INT32_C(  904524629), INT32_C(-1963695561), INT32_C( -297278672), INT32_C(-1666020030)),
      easysimd_mm512_set_epi32(INT32_C(-1055194531), INT32_C( 1846727705), INT32_C(     299526), INT32_C( -610713755),
                            INT32_C( 1744835202), INT32_C(  697979892), INT32_C(   49552843), INT32_C(  171970593),
                            INT32_C(-2060335241), INT32_C(  135497979), INT32_C(  555749520), INT32_C( -140680021),
                            INT32_C(-1676162464), INT32_C(  626483741), INT32_C(  170885439), INT32_C(  230851400)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_and_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_and_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_and_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT16_C(21810),
      easysimd_mm512_set_epi32(INT32_C( 1183808039), INT32_C( 2103907232), INT32_C(-1995421302), INT32_C( 1328084931),
                            INT32_C( -379562245), INT32_C( 1144599747), INT32_C(-1418414219), INT32_C( 1379143176),
                            INT32_C(-2075387410), INT32_C(-1152868472), INT32_C( 1425101887), INT32_C(-1626225579),
                            INT32_C( 2014677400), INT32_C( 1108777022), INT32_C(  877429697), INT32_C( 1558575022)),
      easysimd_mm512_set_epi32(INT32_C( 1822881045), INT32_C( -803993576), INT32_C(  163813867), INT32_C(-1017884831),
                            INT32_C(-1258992440), INT32_C( 1182354404), INT32_C(-1425047283), INT32_C( 1708628167),
                            INT32_C(-2051115159), INT32_C(  143410800), INT32_C(-1041180848), INT32_C(-1023702255),
                            INT32_C( -240255997), INT32_C(  978107452), INT32_C(-1343783755), INT32_C( -880122145)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( 1342439424), INT32_C(          0), INT32_C( 1124091713),
                            INT32_C(          0), INT32_C( 1144586432), INT32_C(          0), INT32_C( 1075054592),
                            INT32_C(          0), INT32_C(          0), INT32_C( 1089491984), INT32_C(-2112782319),
                            INT32_C(          0), INT32_C(          0), INT32_C(  608436353), INT32_C(          0)) },
    { UINT16_C( 3728),
      easysimd_mm512_set_epi32(INT32_C( -632750822), INT32_C( 1966832804), INT32_C( 1231278072), INT32_C(  832865002),
                            INT32_C(-1694490420), INT32_C( -466764866), INT32_C( 1702721236), INT32_C( 2092439418),
                            INT32_C(  611933067), INT32_C(-1184445720), INT32_C(-1689922195), INT32_C(-1140532352),
                            INT32_C(  158346875), INT32_C( 1089717474), INT32_C( 1230253618), INT32_C( 1504968720)),
      easysimd_mm512_set_epi32(INT32_C(-1286938208), INT32_C(  537660993), INT32_C(-1605590998), INT32_C( 1810066171),
                            INT32_C(  -10167007), INT32_C( 1359569321), INT32_C(-1430119757), INT32_C(  256064867),
                            INT32_C(  788618356), INT32_C( -392789188), INT32_C( -613873031), INT32_C(  773158597),
                            INT32_C( -194106681), INT32_C(  165465349), INT32_C(-1346434037), INT32_C(-1412186885)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(-1694498816), INT32_C( 1074340264), INT32_C(  541069968), INT32_C(          0),
                            INT32_C(  604067840), INT32_C(          0), INT32_C(          0), INT32_C(  738480768),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(56690),
      easysimd_mm512_set_epi32(INT32_C( -794660034), INT32_C(  173952757), INT32_C( 1507611872), INT32_C(-1888865381),
                            INT32_C(   15541452), INT32_C( -670396349), INT32_C(-1090081489), INT32_C(  116025329),
                            INT32_C(  130963716), INT32_C(  230354364), INT32_C( 1174065929), INT32_C( 1971493681),
                            INT32_C(-1343257591), INT32_C(-1419733408), INT32_C(  659096905), INT32_C(  183711411)),
      easysimd_mm512_set_epi32(INT32_C(  218373799), INT32_C( -362113249), INT32_C( -694086277), INT32_C( 1616282233),
                            INT32_C(-2132390836), INT32_C( -153917613), INT32_C(  736513734), INT32_C(-1311057544),
                            INT32_C(  505476811), INT32_C( 1767695145), INT32_C(  157469724), INT32_C( -371725260),
                            INT32_C( 1996701751), INT32_C(-1377678442), INT32_C( -132601652), INT32_C( -903948497)),
      easysimd_mm512_set_epi32(INT32_C(       4646), INT32_C(  172623381), INT32_C(          0), INT32_C(    4325913),
                            INT32_C(   14943308), INT32_C( -805175229), INT32_C(          0), INT32_C(   13255024),
                            INT32_C(          0), INT32_C(  152625448), INT32_C(   23248904), INT32_C( 1635944496),
                            INT32_C(          0), INT32_C(          0), INT32_C(  537395272), INT32_C(          0)) },
    { UINT16_C(54171),
      easysimd_mm512_set_epi32(INT32_C(  986405639), INT32_C( 1121234438), INT32_C( -731393712), INT32_C( -784753228),
                            INT32_C( -748350470), INT32_C( 1755197901), INT32_C( 1090059253), INT32_C( 1329426651),
                            INT32_C(-1604442789), INT32_C(  878047098), INT32_C(-1682276633), INT32_C(  -78811559),
                            INT32_C( -973139496), INT32_C(   91517188), INT32_C(   37440120), INT32_C(  829401648)),
      easysimd_mm512_set_epi32(INT32_C( 2057630636), INT32_C(  335188274), INT32_C( 1272591061), INT32_C( -327494197),
                            INT32_C(  892006932), INT32_C(-1158010747), INT32_C( -700127305), INT32_C(-1058033333),
                            INT32_C( 2135825335), INT32_C( -510043422), INT32_C( -139544800), INT32_C(   95748631),
                            INT32_C( 1671238992), INT32_C( 1305915968), INT32_C( -582790199), INT32_C( 1758236157)),
      easysimd_mm512_set_epi32(INT32_C(  981488388), INT32_C(   47218690), INT32_C(          0), INT32_C(-1070033536),
                            INT32_C(          0), INT32_C(          0), INT32_C( 1077994421), INT32_C( 1076703307),
                            INT32_C(  541982995), INT32_C(          0), INT32_C(          0), INT32_C(   17105425),
                            INT32_C( 1100812624), INT32_C(          0), INT32_C(     213576), INT32_C(  541886512)) },
    { UINT16_C( 6763),
      easysimd_mm512_set_epi32(INT32_C( -697747292), INT32_C(-1779695782), INT32_C(  491327584), INT32_C( -563681080),
                            INT32_C( 1527319596), INT32_C( 1347900829), INT32_C(-1792354715), INT32_C( -371177698),
                            INT32_C(  255088013), INT32_C( 1961231505), INT32_C( -659343095), INT32_C( 1620234692),
                            INT32_C(  843561067), INT32_C( 1265300992), INT32_C(-1675104490), INT32_C( -873664156)),
      easysimd_mm512_set_epi32(INT32_C( -822209230), INT32_C(-1549704264), INT32_C(-1157339218), INT32_C(-2025137124),
                            INT32_C(-1631723043), INT32_C(-2095740678), INT32_C(-1857753563), INT32_C(-1236342636),
                            INT32_C( 1439297909), INT32_C( -576201057), INT32_C( 1488873085), INT32_C(-1369304746),
                            INT32_C(  567848046), INT32_C( 1335236564), INT32_C( -942680632), INT32_C(-1512916560)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-2042443768),
                            INT32_C(  436797452), INT32_C(          0), INT32_C(-1862220251), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1420166289), INT32_C( 1488068617), INT32_C(          0),
                            INT32_C(  541108330), INT32_C(          0), INT32_C(-2079865600), INT32_C(-2118078176)) },
    { UINT16_C(17574),
      easysimd_mm512_set_epi32(INT32_C(-2098508971), INT32_C(  488824783), INT32_C( 1647236603), INT32_C(-1221154150),
                            INT32_C(-1012750616), INT32_C( 1849794587), INT32_C(  131205926), INT32_C( 2061955170),
                            INT32_C(   37003574), INT32_C( 1649229141), INT32_C(  612060260), INT32_C(-1402263233),
                            INT32_C( -513572270), INT32_C(  701923816), INT32_C(  511549547), INT32_C(  969083331)),
      easysimd_mm512_set_epi32(INT32_C( -873396634), INT32_C(-1835948135), INT32_C(-1570208244), INT32_C( -244400530),
                            INT32_C( 1126824505), INT32_C( 1036340167), INT32_C(  424825857), INT32_C(  434716327),
                            INT32_C( 1663095683), INT32_C( 1730428966), INT32_C( -252943126), INT32_C(  373990324),
                            INT32_C( 2100741912), INT32_C(-1005385937), INT32_C(-1470990839), INT32_C(-1200692008)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  268470153), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(  742457347), INT32_C(          0), INT32_C(          0),
                            INT32_C(   35684610), INT32_C(          0), INT32_C(  543704160), INT32_C(          0),
                            INT32_C(          0), INT32_C(    1179944), INT32_C(  139468809), INT32_C(          0)) },
    { UINT16_C( 5226),
      easysimd_mm512_set_epi32(INT32_C(  793357988), INT32_C(  607421194), INT32_C(-1375787856), INT32_C( 1799684145),
                            INT32_C( -551549754), INT32_C( 1249951615), INT32_C( 1447553529), INT32_C(-2080136983),
                            INT32_C( -623547588), INT32_C( 1015056564), INT32_C(-1333355305), INT32_C( 1925062912),
                            INT32_C(  330138155), INT32_C( -444195598), INT32_C(  874806560), INT32_C(  319126943)),
      easysimd_mm512_set_epi32(INT32_C( -542891463), INT32_C(  629803756), INT32_C(-1216921331), INT32_C( -301860714),
                            INT32_C(  317296385), INT32_C( 1833800187), INT32_C( -645353377), INT32_C(-1989715102),
                            INT32_C( 1785805415), INT32_C( 1739507553), INT32_C(-1932540399), INT32_C( -641363523),
                            INT32_C(   72499486), INT32_C(-2005610208), INT32_C( -497730582), INT32_C(  312611747)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C( 1778450448),
                            INT32_C(          0), INT32_C( 1207996795), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(  612401184), INT32_C(-2138668015), INT32_C(          0),
                            INT32_C(         10), INT32_C(          0), INT32_C(  537147680), INT32_C(          0)) },
    { UINT16_C(43362),
      easysimd_mm512_set_epi32(INT32_C(-2123521230), INT32_C( -883174176), INT32_C( -332498316), INT32_C(  142889340),
                            INT32_C( 1231002435), INT32_C( 1681334055), INT32_C(  138372594), INT32_C( -878588971),
                            INT32_C(  423295425), INT32_C( -460615607), INT32_C( 1208771148), INT32_C(-2128303155),
                            INT32_C( -738338972), INT32_C( 2110676823), INT32_C(-1405320678), INT32_C(-2007459833)),
      easysimd_mm512_set_epi32(INT32_C( -941377596), INT32_C(-1151664921), INT32_C( -656770411), INT32_C(  861635987),
                            INT32_C( 1823839521), INT32_C( 1391000031), INT32_C(   73229946), INT32_C(  -53693878),
                            INT32_C( -257851255), INT32_C( -479522767), INT32_C(  745275629), INT32_C(-1783480446),
                            INT32_C(-1431666964), INT32_C( -422291816), INT32_C( -309252994), INT32_C(-2112946871)),
      easysimd_mm512_set_epi32(INT32_C(-2124307712), INT32_C(          0), INT32_C( -938837484), INT32_C(          0),
                            INT32_C( 1209372929), INT32_C(          0), INT32_C(          0), INT32_C( -931102144),
                            INT32_C(          0), INT32_C( -536145919), INT32_C(  135004236), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1408237542), INT32_C(          0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_and_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_and_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_and_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 9036212757557430666), INT64_C( 5704081348870821627),
                            INT64_C( 4916018483251427189), INT64_C( 5923374839641151982),
                            INT64_C(-4951532382404389825), INT64_C(-6984585675708986984),
                            INT64_C( 4762161048923702209), INT64_C( 6694028748503799090)),
      easysimd_mm512_set_epi64(INT64_C(-3453126114950276629), INT64_C(-4371782057203512120),
                            INT64_C( 5078173500331491597), INT64_C( 7338502100533278569),
                            INT64_C(  615944699146983248), INT64_C(-4396767702011741181),
                            INT64_C( 4200939521265073333), INT64_C(-3780095828076561881)),
      easysimd_mm512_set_epi64(INT64_C( 5765733423092074890), INT64_C( 4827937147730281672),
                            INT64_C( 4915961295754526981), INT64_C( 4617324316270006632),
                            INT64_C(  578719150276367376), INT64_C(-9074330961792628736),
                            INT64_C(  145408472056594561), INT64_C( 5224288327930368034)) },
    { easysimd_mm512_set_epi64(INT64_C( 5288299052354798314), INT64_C(-7277780933457101890),
                            INT64_C( 7313132024917137274), INT64_C( 2628232513216498408),
                            INT64_C(-7258160557155099776), INT64_C(  680094650638517474),
                            INT64_C( 5283899056600645648), INT64_C( 2807728906149754133)),
      easysimd_mm512_set_epi64(INT64_C(-6895960825351935237), INT64_C(  -43666961203633751),
                            INT64_C(-6142317585422402205), INT64_C( 3387090051947463484),
                            INT64_C(-2636564591268235579), INT64_C( -833681846664639227),
                            INT64_C(-5782890152253473541), INT64_C(-2717644085040284508)),
      easysimd_mm512_set_epi64(INT64_C(   18155308360663786), INT64_C(-7277816996756381272),
                            INT64_C( 2323877817609098594), INT64_C( 2594451620184359464),
                            INT64_C(-7259520672634482048), INT64_C(   27068065029080064),
                            INT64_C(  654161048765464592), INT64_C(  162142851124327428)) },
    { easysimd_mm512_set_epi64(INT64_C(-8112615037926038324), INT64_C(-2879330391107916497),
                            INT64_C(  498324993693604100), INT64_C(  989364461044945673),
                            INT64_C( 8467500887117366281), INT64_C(-6097708555739527863),
                            INT64_C(  789034506143718770), INT64_C(-5527357514795184575)),
      easysimd_mm512_set_epi64(INT64_C( 6941879334003428428), INT64_C( -661071113376870714),
                            INT64_C(-5630949274148604213), INT64_C( 7592192837230447644),
                            INT64_C(-1596547832800395209), INT64_C(-5917083848631867188),
                            INT64_C(-3882429228383046850), INT64_C(  747121403871646944)),
      easysimd_mm512_set_epi64(INT64_C(   18579654875284556), INT64_C(-3458201275382494202),
                            INT64_C(   56929894688378880), INT64_C(  655521307720597512),
                            INT64_C( 7026328109045710849), INT64_C(-6241986883974856632),
                            INT64_C(  725666832763081010), INT64_C(  165015392292176960)) },
    { easysimd_mm512_set_epi64(INT64_C(-3214140792841031219), INT64_C( 4681768843666616539),
                            INT64_C(-6891029306179981446), INT64_C(-7225323117343838631),
                            INT64_C(-4179602309674405628), INT64_C(  160804091787717168),
                            INT64_C( 3694873207195836071), INT64_C(-1555264558302423685)),
      easysimd_mm512_set_epi64(INT64_C( 3831140603882252421), INT64_C(-3007023874774683317),
                            INT64_C( 9173299967578168034), INT64_C( -599340352231112169),
                            INT64_C( 7177916815745921600), INT64_C(-2503064843376095747),
                            INT64_C( 4236579961216216582), INT64_C(-3141312070029828684)),
      easysimd_mm512_set_epi64(INT64_C( 1234006158509482117), INT64_C( 4629950784542158923),
                            INT64_C( 2327799239051135586), INT64_C(-7806367493009374703),
                            INT64_C( 4727954219193348096), INT64_C(     917302477097008),
                            INT64_C( 3621548322709573126), INT64_C(-4583941969312410320)) },
    { easysimd_mm512_set_epi64(INT64_C( 5789189981308900965), INT64_C(-1594196073659476595),
                            INT64_C( 8423425177495484681), INT64_C( 6958855014828193899),
                            INT64_C( 5434426382856220438), INT64_C(-3752358975457584533),
                            INT64_C( 8837456289202868530), INT64_C( 5465736992144414155)),
      easysimd_mm512_set_epi64(INT64_C(-9001137670469652955), INT64_C(-5310051186831134347),
                            INT64_C(-2474764694246758787), INT64_C(-5881119101759738770),
                            INT64_C( 5734797378155697608), INT64_C(-6497927143179601756),
                            INT64_C(-7643735180027817888), INT64_C(-2420991802446640084)),
      easysimd_mm512_set_epi64(INT64_C(    5911629778699813), INT64_C(-6899444174404169467),
                            INT64_C( 6099567767624753161), INT64_C( 2306424943463671914),
                            INT64_C( 5404906694268929280), INT64_C(-9097076494042654688),
                            INT64_C( 1198224716855775264), INT64_C( 5350875317483865096)) },
    { easysimd_mm512_set_epi64(INT64_C(  563525163273351266), INT64_C(  158929141814345045),
                            INT64_C( 2628778802773961023), INT64_C(-2205776103080558104),
                            INT64_C( 2197088575617698243), INT64_C( 5860665973320586546),
                            INT64_C(-6655929129214122066), INT64_C(-8697897714832252451)),
      easysimd_mm512_set_epi64(INT64_C( 1824613162744888999), INT64_C( 7142941570334212134),
                            INT64_C(-1086382453544016972), INT64_C( 9022617812666091311),
                            INT64_C(-6317857543126326056), INT64_C(-9013027400318787633),
                            INT64_C( 7074827341732948634), INT64_C(-4349730772874059749)),
      easysimd_mm512_set_epi64(INT64_C(   95138542545547298), INT64_C(  153264234564943876),
                            INT64_C( 2335191585971118388), INT64_C( 6999299776119439656),
                            INT64_C(  599013974410920128), INT64_C(   18366259613799682),
                            INT64_C( 2459037658077471370), INT64_C(-9006491134657978343)) },
    { easysimd_mm512_set_epi64(INT64_C(-8934120309513688260), INT64_C( 4359634748931742935),
                            INT64_C( 8268082250112664107), INT64_C(-1907805565562356448),
                            INT64_C( 1370639783932859498), INT64_C(-3751209977007462503),
                            INT64_C(-6743993051839021458), INT64_C( 4839674398342728647)),
      easysimd_mm512_set_epi64(INT64_C(-8545761289661498777), INT64_C( 7471128053642413585),
                            INT64_C(-2754635356059844322), INT64_C(-8614030248086520854),
                            INT64_C( 1342657230503784100), INT64_C( 2608854166046450864),
                            INT64_C( 7729584549648139462), INT64_C( 5368501309454936569)),
      easysimd_mm512_set_epi64(INT64_C(-9222527198360950236), INT64_C( 2630243059467977745),
                            INT64_C( 5801763319471669258), INT64_C(-9222228406785787616),
                            INT64_C( 1297055947226612768), INT64_C(   13513008652297360),
                            INT64_C( 2467972599307568198), INT64_C( 4756005960819231169)) },
    { easysimd_mm512_set_epi64(INT64_C( 1818040010755772489), INT64_C( 5191632551175039949),
                            INT64_C(-3171141735991582889), INT64_C(-6035806350115039225),
                            INT64_C( 4482393771264451129), INT64_C( 2704986537996009741),
                            INT64_C(-1296481894259912959), INT64_C( 7876111834213298271)),
      easysimd_mm512_set_epi64(INT64_C(-1107462703642111951), INT64_C( 3200934455572316034),
                            INT64_C(-6148962785270933864), INT64_C(-1328231493238063799),
                            INT64_C(-9120454231799900960), INT64_C(-1428069393052184196),
                            INT64_C( 5287115201302699815), INT64_C(  594305769309064149)),
      easysimd_mm512_set_epi64(INT64_C( 1162069999159545857), INT64_C(  579838780607835008),
                            INT64_C(-9031874063504222704), INT64_C(-6048334185741379583),
                            INT64_C(   10310269968318496), INT64_C( 2596899559745798412),
                            INT64_C( 5188595380067045121), INT64_C(  580122069275017301)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_and_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_and_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_and_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 9036212757557430666), INT64_C( 5704081348870821627),
                            INT64_C( 4916018483251427189), INT64_C( 5923374839641151982),
                            INT64_C(-4951532382404389825), INT64_C(-6984585675708986984),
                            INT64_C( 4762161048923702209), INT64_C( 6694028748503799090)),
      UINT8_C( 39),
      easysimd_mm512_set_epi64(INT64_C( 7829214476264278040), INT64_C(  703575204673376097),
                            INT64_C(-5407331354528887836), INT64_C(-6120531474030028601),
                            INT64_C(-8809472528091429264), INT64_C(-4471837688110281967),
                            INT64_C(-1031891648804766660), INT64_C(-5771507277206231329)),
      easysimd_mm512_set_epi64(INT64_C( 8447482571111256056), INT64_C( 3577127948173451468),
                            INT64_C(-2004739832689101100), INT64_C( 8986958869783206795),
                            INT64_C(-5087155628682128019), INT64_C(-4898549151711613317),
                            INT64_C( 4680300913939983922), INT64_C( 6463791434556706448)),
      easysimd_mm512_set_epi64(INT64_C( 9036212757557430666), INT64_C( 5704081348870821627),
                            INT64_C(-6618874695993703740), INT64_C( 5923374839641151982),
                            INT64_C(-4951532382404389825), INT64_C(-9223140039893972463),
                            INT64_C( 4657219002685467696), INT64_C(  694680311283451536)) },
    { easysimd_mm512_set_epi64(INT64_C( 2309236383959261226), INT64_C( 7774175012325743905),
                            INT64_C( 5839305773204773555), INT64_C( 1099790230208207988),
                            INT64_C(-1687016713001301383), INT64_C( 3320690892837104327),
                            INT64_C(  710668265524759563), INT64_C(-6065296483252896486)),
      UINT8_C(160),
      easysimd_mm512_set_epi64(INT64_C(  747121403871646944), INT64_C(-8112615037926038324),
                            INT64_C(-2879330391107916497), INT64_C(  498324993693604100),
                            INT64_C(  989364461044945673), INT64_C( 8467500887117366281),
                            INT64_C(-6097708555739527863), INT64_C(  789034506143718770)),
      easysimd_mm512_set_epi64(INT64_C(-1555264558302423685), INT64_C( 6941879334003428428),
                            INT64_C( -661071113376870714), INT64_C(-5630949274148604213),
                            INT64_C( 7592192837230447644), INT64_C(-1596547832800395209),
                            INT64_C(-5917083848631867188), INT64_C(-3882429228383046850)),
      easysimd_mm512_set_epi64(INT64_C(  741411777270518880), INT64_C( 7774175012325743905),
                            INT64_C(-3458201275382494202), INT64_C( 1099790230208207988),
                            INT64_C(-1687016713001301383), INT64_C( 3320690892837104327),
                            INT64_C(  710668265524759563), INT64_C(-6065296483252896486)) },
    { easysimd_mm512_set_epi64(INT64_C(-3141312070029828684), INT64_C(-3214140792841031219),
                            INT64_C( 4681768843666616539), INT64_C(-6891029306179981446),
                            INT64_C(-7225323117343838631), INT64_C(-4179602309674405628),
                            INT64_C(  160804091787717168), INT64_C( 3694873207195836071)),
      UINT8_C(  6),
      easysimd_mm512_set_epi64(INT64_C( 1439622676105278165), INT64_C(-1406576864852774380),
                            INT64_C(-4973618283186690121), INT64_C(-4544218561177052233),
                            INT64_C(-2190619812874504416), INT64_C(  411237240453010768),
                            INT64_C( 5608866377596359625), INT64_C( 7551566793946127111)),
      easysimd_mm512_set_epi64(INT64_C( 2110235908633979080), INT64_C( 6559787716707833245),
                            INT64_C(-7698104879832611042), INT64_C( 1095594675397854353),
                            INT64_C(-2831857028248186428), INT64_C( 3623067196209165824),
                            INT64_C(-7194518998511455900), INT64_C(-8783679074311277652)),
      easysimd_mm512_set_epi64(INT64_C(-3141312070029828684), INT64_C(-3214140792841031219),
                            INT64_C( 4681768843666616539), INT64_C(-6891029306179981446),
                            INT64_C(-7225323117343838631), INT64_C(    1407388893058048),
                            INT64_C(  866558102575534400), INT64_C( 3694873207195836071)) },
    { easysimd_mm512_set_epi64(INT64_C(-4970734089418384356), INT64_C(-7008197103615375110),
                            INT64_C(-7978990794053850988), INT64_C( 6181737452074950303),
                            INT64_C( 6394661210895290710), INT64_C( 2438888788002740180),
                            INT64_C(-4048782482230560336), INT64_C(-2996801797497290918)),
      UINT8_C(184),
      easysimd_mm512_set_epi64(INT64_C( 7074827341732948634), INT64_C(-4349730772874059749),
                            INT64_C(  563525163273351266), INT64_C(  158929141814345045),
                            INT64_C( 2628778802773961023), INT64_C(-2205776103080558104),
                            INT64_C( 2197088575617698243), INT64_C( 5860665973320586546)),
      easysimd_mm512_set_epi64(INT64_C(-6743993051839021458), INT64_C( 4839674398342728647),
                            INT64_C( 1824613162744888999), INT64_C( 7142941570334212134),
                            INT64_C(-1086382453544016972), INT64_C( 9022617812666091311),
                            INT64_C(-6317857543126326056), INT64_C(-9013027400318787633)),
      easysimd_mm512_set_epi64(INT64_C( 2461362369224681994), INT64_C(-7008197103615375110),
                            INT64_C(   95138542545547298), INT64_C(  153264234564943876),
                            INT64_C( 2335191585971118388), INT64_C( 2438888788002740180),
                            INT64_C(-4048782482230560336), INT64_C(-2996801797497290918)) },
    { easysimd_mm512_set_epi64(INT64_C( 7729584549648139462), INT64_C( 5368501309454936569),
                            INT64_C(-8934120309513688260), INT64_C( 4359634748931742935),
                            INT64_C( 8268082250112664107), INT64_C(-1907805565562356448),
                            INT64_C( 1370639783932859498), INT64_C(-3751209977007462503)),
      UINT8_C(176),
      easysimd_mm512_set_epi64(INT64_C(-5226637314456684394), INT64_C( 1362777598547825147),
                            INT64_C(-2771771646272906398), INT64_C( 7669975856184215393),
                            INT64_C(-8300197808250187331), INT64_C(  311382923636166944),
                            INT64_C(-2137736571596434525), INT64_C( 3407446613087781642)),
      easysimd_mm512_set_epi64(INT64_C(  613705043478027075), INT64_C( 7221274780014437874),
                            INT64_C(-3773510896647996991), INT64_C(-1978328966883417524),
                            INT64_C(-9140992443141990556), INT64_C( 9065287930099827226),
                            INT64_C(-8621974329724982942), INT64_C(-2331701078232790292)),
      easysimd_mm512_set_epi64(INT64_C(    1197420910319618), INT64_C( 5368501309454936569),
                            INT64_C(-3926997772353251008), INT64_C( 6917828375051256384),
                            INT64_C( 8268082250112664107), INT64_C(-1907805565562356448),
                            INT64_C( 1370639783932859498), INT64_C(-3751209977007462503)) },
    { easysimd_mm512_set_epi64(INT64_C( 3700698387045520673), INT64_C( 5974299641953216122),
                            INT64_C( -230613445968297847), INT64_C(-2059534601207152403),
                            INT64_C(-7659990185762193684), INT64_C(-1813729535102735234),
                            INT64_C(-9075037706959084750), INT64_C(-3793204198629279116)),
      UINT8_C(149),
      easysimd_mm512_set_epi64(INT64_C(-1763977094002909713), INT64_C(-2903140489693440283),
                            INT64_C( 6916354853087283405), INT64_C( -288829486582280686),
                            INT64_C( 3770029145011694022), INT64_C( 4745990232274262704),
                            INT64_C(-9178901749894611532), INT64_C(-4043185984863798041)),
      easysimd_mm512_set_epi64(INT64_C(-2601025793624159885), INT64_C( 3563840185149175707),
                            INT64_C( -266768530495040309), INT64_C( 3956680363155515245),
                            INT64_C(-3793861402632321428), INT64_C(-9102379300509684954),
                            INT64_C(-8107510778505593825), INT64_C( 3515623724266758070)),
      easysimd_mm512_set_epi64(INT64_C(-4358068267243845277), INT64_C( 5974299641953216122),
                            INT64_C( -230613445968297847), INT64_C( 3668413965043549696),
                            INT64_C(-7659990185762193684), INT64_C(  111745843014421024),
                            INT64_C(-9075037706959084750), INT64_C(   54607264595188902)) },
    { easysimd_mm512_set_epi64(INT64_C( 6254124818226667045), INT64_C( 2295645667912807702),
                            INT64_C(-4018707571999816637), INT64_C(  -26221181248372278),
                            INT64_C(-1805361387364896744), INT64_C(-3741154382847501293),
                            INT64_C( 2395841610067775712), INT64_C( 4058242405589980421)),
      UINT8_C( 95),
      easysimd_mm512_set_epi64(INT64_C( 8131676417901457360), INT64_C( 5114845102355343256),
                            INT64_C(-4438171947452810779), INT64_C(-7935157497162192255),
                            INT64_C(  113046603373071999), INT64_C( 5604608652266856031),
                            INT64_C(-6729023456945537362), INT64_C(-3975720697196785172)),
      easysimd_mm512_set_epi64(INT64_C( 4986045839810005585), INT64_C(  594372055586112482),
                            INT64_C( 4543836970229042581), INT64_C(-3911934107467365366),
                            INT64_C(-4070664534473188278), INT64_C( 5469443597420811446),
                            INT64_C(-7904155369341108539), INT64_C(-3737549779697651723)),
      easysimd_mm512_set_epi64(INT64_C( 6254124818226667045), INT64_C(   16747779451847552),
                            INT64_C(-4018707571999816637), INT64_C(-9106278428011921408),
                            INT64_C(  108112814837713994), INT64_C( 5316237509498262550),
                            INT64_C(-9075733389403931004), INT64_C(-4034937163739561500)) },
    { easysimd_mm512_set_epi64(INT64_C(-4532025999716330471), INT64_C( 1441736923413036901),
                            INT64_C( 4055551302629479412), INT64_C(  212827843943377228),
                            INT64_C(-8849072478755780357), INT64_C(-1422057783446117205),
                            INT64_C(-7199062965036293603), INT64_C(  733947372098454344)),
      UINT8_C( 32),
      easysimd_mm512_set_epi64(INT64_C(-6215126846548299491), INT64_C(-2916974460339411728),
                            INT64_C( 7873834362860447881), INT64_C( 8398781451642376253),
                            INT64_C( 4819162425848909387), INT64_C( 3108138717093232544),
                            INT64_C(-1968231203112162098), INT64_C(-5881820928188872186)),
      easysimd_mm512_set_epi64(INT64_C(-5650512314146637209), INT64_C(    3824130547717357),
                            INT64_C( 8692530928452313027), INT64_C(    8285423730243427),
                            INT64_C( 3742589136248759843), INT64_C( 2734033046067673576),
                            INT64_C( 3884903702312804919), INT64_C(-1276802171409363646)),
      easysimd_mm512_set_epi64(INT64_C(-4532025999716330471), INT64_C( 1441736923413036901),
                            INT64_C( 7494010129634756737), INT64_C(  212827843943377228),
                            INT64_C(-8849072478755780357), INT64_C(-1422057783446117205),
                            INT64_C(-7199062965036293603), INT64_C(  733947372098454344)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_and_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_and_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_and_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C( 50),
      easysimd_mm512_set_epi64(INT64_C( 5084416814350799776), INT64_C(-8570269232503654461),
                            INT64_C(-1630207427926739773), INT64_C(-6092042681407238648),
                            INT64_C(-8913721049338044536), INT64_C( 6120766000801629269),
                            INT64_C( 8652973546099087422), INT64_C( 3768531854712764334)),
      easysimd_mm512_set_epi64(INT64_C( 7829214476264278040), INT64_C(  703575204673376097),
                            INT64_C(-5407331354528887836), INT64_C(-6120531474030028601),
                            INT64_C(-8809472528091429264), INT64_C(-4471837688110281967),
                            INT64_C(-1031891648804766660), INT64_C(-5771507277206231329)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-6890431802948124480), INT64_C(-6123627733767157760),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 8072007440746715196), INT64_C(                   0)) },
    { UINT8_C(144),
      easysimd_mm512_set_epi64(INT64_C(-2717644085040284508), INT64_C( 5288299052354798314),
                            INT64_C(-7277780933457101890), INT64_C( 7313132024917137274),
                            INT64_C( 2628232513216498408), INT64_C(-7258160557155099776),
                            INT64_C(  680094650638517474), INT64_C( 5283899056600645648)),
      easysimd_mm512_set_epi64(INT64_C(-5527357514795184575), INT64_C(-6895960825351935237),
                            INT64_C(  -43666961203633751), INT64_C(-6142317585422402205),
                            INT64_C( 3387090051947463484), INT64_C(-2636564591268235579),
                            INT64_C( -833681846664639227), INT64_C(-5782890152253473541)),
      easysimd_mm512_set_epi64(INT64_C(-7905821755195257856), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 2323877817609098594),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(114),
      easysimd_mm512_set_epi64(INT64_C(-3413038857294295307), INT64_C( 6475143687707440027),
                            INT64_C(   66750031696924739), INT64_C(-4681864345113958415),
                            INT64_C(  562484877412986300), INT64_C( 5042574770374351665),
                            INT64_C(-5769247420573510048), INT64_C( 2830799652053530291)),
      easysimd_mm512_set_epi64(INT64_C(  937908328941131551), INT64_C(-2981077858701114759),
                            INT64_C(-9158548898769049773), INT64_C( 3163302403568753016),
                            INT64_C( 2171006373899068201), INT64_C(  676327318613388340),
                            INT64_C( 8575768723328224150), INT64_C( -569519755344554193)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 5800658722606940697),
                            INT64_C(   64181022643847235), INT64_C( 3100178613413757296),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 2811090599545930240), INT64_C(                   0)) },
    { UINT8_C(155),
      easysimd_mm512_set_epi64(INT64_C( 4236579961216216582), INT64_C(-3141312070029828684),
                            INT64_C(-3214140792841031219), INT64_C( 4681768843666616539),
                            INT64_C(-6891029306179981446), INT64_C(-7225323117343838631),
                            INT64_C(-4179602309674405628), INT64_C(  160804091787717168)),
      easysimd_mm512_set_epi64(INT64_C( 8837456289202868530), INT64_C( 5465736992144414155),
                            INT64_C( 3831140603882252421), INT64_C(-3007023874774683317),
                            INT64_C( 9173299967578168034), INT64_C( -599340352231112169),
                            INT64_C( 7177916815745921600), INT64_C(-2503064843376095747)),
      easysimd_mm512_set_epi64(INT64_C( 4215460527910977538), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 4629950784542158923),
                            INT64_C( 2327799239051135586), INT64_C(                   0),
                            INT64_C( 4727954219193348096), INT64_C(     917302477097008)) },
    { UINT8_C(107),
      easysimd_mm512_set_epi64(INT64_C(-2996801797497290918), INT64_C( 2110235908633979080),
                            INT64_C( 6559787716707833245), INT64_C(-7698104879832611042),
                            INT64_C( 1095594675397854353), INT64_C(-2831857028248186428),
                            INT64_C( 3623067196209165824), INT64_C(-7194518998511455900)),
      easysimd_mm512_set_epi64(INT64_C(-3531361750574079048), INT64_C(-4970734089418384356),
                            INT64_C(-7008197103615375110), INT64_C(-7978990794053850988),
                            INT64_C( 6181737452074950303), INT64_C( 6394661210895290710),
                            INT64_C( 2438888788002740180), INT64_C(-4048782482230560336)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 1801439990639675400),
                            INT64_C( 1876030771317506200), INT64_C(                   0),
                            INT64_C(  360377053526492305), INT64_C(                   0),
                            INT64_C( 2324042582201603584), INT64_C(-8932954729898528480)) },
    { UINT8_C(166),
      easysimd_mm512_set_epi64(INT64_C(-9013027400318787633), INT64_C( 7074827341732948634),
                            INT64_C(-4349730772874059749), INT64_C(  563525163273351266),
                            INT64_C(  158929141814345045), INT64_C( 2628778802773961023),
                            INT64_C(-2205776103080558104), INT64_C( 2197088575617698243)),
      easysimd_mm512_set_epi64(INT64_C(-3751209977007462503), INT64_C(-6743993051839021458),
                            INT64_C( 4839674398342728647), INT64_C( 1824613162744888999),
                            INT64_C( 7142941570334212134), INT64_C(-1086382453544016972),
                            INT64_C( 9022617812666091311), INT64_C(-6317857543126326056)),
      easysimd_mm512_set_epi64(INT64_C(-9015922487669520503), INT64_C(                   0),
                            INT64_C( 4837044293220761603), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 2335191585971118388),
                            INT64_C( 6999299776119439656), INT64_C(                   0)) },
    { UINT8_C(106),
      easysimd_mm512_set_epi64(INT64_C( 3407446613087781642), INT64_C(-5908963845954273231),
                            INT64_C(-2368888154296893569), INT64_C( 6217195068479217897),
                            INT64_C(-2678116496944625484), INT64_C(-5726717426998042368),
                            INT64_C( 1417932582737550578), INT64_C( 3757265565845388703)),
      easysimd_mm512_set_epi64(INT64_C(-2331701078232790292), INT64_C(-5226637314456684394),
                            INT64_C( 1362777598547825147), INT64_C(-2771771646272906398),
                            INT64_C( 7669975856184215393), INT64_C(-8300197808250187331),
                            INT64_C(  311382923636166944), INT64_C(-2137736571596434525)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-6523745533444423664),
                            INT64_C( 1306048291191951739), INT64_C(                   0),
                            INT64_C( 5355107766639820832), INT64_C(                   0),
                            INT64_C(         45097422880), INT64_C(                   0)) },
    { UINT8_C( 98),
      easysimd_mm512_set_epi64(INT64_C(-9120454231799900960), INT64_C(-1428069393052184196),
                            INT64_C( 5287115201302699815), INT64_C(  594305769309064149),
                            INT64_C( 1818040010755772489), INT64_C( 5191632551175039949),
                            INT64_C(-3171141735991582889), INT64_C(-6035806350115039225)),
      easysimd_mm512_set_epi64(INT64_C(-4043185984863798041), INT64_C(-2820807435363842669),
                            INT64_C( 7833331097238305247), INT64_C(  314520227399119434),
                            INT64_C(-1107462703642111951), INT64_C( 3200934455572316034),
                            INT64_C(-6148962785270933864), INT64_C(-1328231493238063799)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-4032276290038922992),
                            INT64_C( 5194217179798575367), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-9031874063504222704), INT64_C(                   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_and_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_and_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_and_si512(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 1762603276), INT32_C(-1316946536), INT32_C( -409636803), INT32_C(-1096492450),
                            INT32_C( 1487241173), INT32_C(-1940071138), INT32_C( 1116126146), INT32_C( -916337722),
                            INT32_C(   52488417), INT32_C( 1044081507), INT32_C(-1035184013), INT32_C(-1384518181),
                            INT32_C(-1687535487), INT32_C(-1003450758), INT32_C(  131212491), INT32_C(-1389162000)),
      easysimd_mm512_set_epi32(INT32_C(-1226890085), INT32_C(  294090078), INT32_C( 1993383303), INT32_C( 2117895966),
                            INT32_C( -579088274), INT32_C( 1336927221), INT32_C(   43647960), INT32_C( 1458493074),
                            INT32_C( -558283956), INT32_C(  428402022), INT32_C(  974860288), INT32_C( -624769004),
                            INT32_C(-1594949458), INT32_C(  -55819000), INT32_C(  423209449), INT32_C( -449605661)),
      easysimd_mm512_set_epi32(INT32_C(  537862152), INT32_C(  293631256), INT32_C( 1720717317), INT32_C( 1042579998),
                            INT32_C( 1478578244), INT32_C(  202162452), INT32_C(   42075072), INT32_C( 1080083074),
                            INT32_C(   35668032), INT32_C(  403202402), INT32_C(   34082816), INT32_C(-2009020400),
                            INT32_C(-2140536704), INT32_C(-1004534776), INT32_C(   17834185), INT32_C(-1523383840)) },
    { easysimd_mm512_set_epi32(INT32_C(-1226448376), INT32_C( 1927095485), INT32_C( 1303264402), INT32_C(  926300607),
                            INT32_C( 2116494628), INT32_C(-1100758331), INT32_C(  853434891), INT32_C( 1856581548),
                            INT32_C( 1730450085), INT32_C(-1233336711), INT32_C(  410802607), INT32_C(-1458345357),
                            INT32_C(-1414257532), INT32_C(-1908909941), INT32_C(  216004947), INT32_C(-1153616267)),
      easysimd_mm512_set_epi32(INT32_C(-1170743204), INT32_C( 1817377482), INT32_C(  934231558), INT32_C( -128119768),
                            INT32_C(  674694491), INT32_C( 1902405145), INT32_C(-1438313883), INT32_C( -508947384),
                            INT32_C( -334819615), INT32_C(-1499616800), INT32_C( -300643115), INT32_C(-1687918613),
                            INT32_C(  204696129), INT32_C( -790552335), INT32_C( -837803722), INT32_C(-1569322126)),
      easysimd_mm512_set_epi32(INT32_C(-1306140664), INT32_C( 1615863944), INT32_C(   95304706), INT32_C(  806619176),
                            INT32_C(  673645824), INT32_C(  811614721), INT32_C(  574882305), INT32_C( 1621623816),
                            INT32_C( 1677722273), INT32_C(-1508081568), INT32_C(  135530629), INT32_C(-1996463517),
                            INT32_C(  137371648), INT32_C(-2145386367), INT32_C(  202383634), INT32_C(-1573649808)) },
    { easysimd_mm512_set_epi32(INT32_C( -967914791), INT32_C( 1028004547), INT32_C( 1106145634), INT32_C(-2126623640),
                            INT32_C(   -6485699), INT32_C( -256904631), INT32_C(-1220204919), INT32_C(  917835787),
                            INT32_C( 1623427491), INT32_C( -222464855), INT32_C( -472879958), INT32_C( -762982604),
                            INT32_C(-1085405824), INT32_C( 1812599478), INT32_C( -613988136), INT32_C(  621554720)),
      easysimd_mm512_set_epi32(INT32_C( 1377438428), INT32_C( 1201451322), INT32_C(  619734582), INT32_C(-1992526637),
                            INT32_C( -868585296), INT32_C( 1439924174), INT32_C( 1656419868), INT32_C(-1390170089),
                            INT32_C( 1058692615), INT32_C( -836785520), INT32_C(-1830636694), INT32_C(  554982917),
                            INT32_C(-1514534388), INT32_C( -747673601), INT32_C( -692811287), INT32_C(-1799571639)),
      easysimd_mm512_set_epi32(INT32_C( 1107954392), INT32_C(   84152322), INT32_C(   14704674), INT32_C(-2126756800),
                            INT32_C( -870842320), INT32_C( 1350791752), INT32_C(  570435080), INT32_C(  606146563),
                            INT32_C(  537004035), INT32_C(-1038278528), INT32_C(-2101338070), INT32_C(     282116),
                            INT32_C(-1526071296), INT32_C( 1074397878), INT32_C( -769390392), INT32_C(   67904512)) },
    { easysimd_mm512_set_epi32(INT32_C(-1436950998), INT32_C(-1742059387), INT32_C( 1377677769), INT32_C(-2097193192),
                            INT32_C( 1556973207), INT32_C(   58040738), INT32_C(-1875805492), INT32_C( -452882923),
                            INT32_C(-2070651162), INT32_C(-1417594324), INT32_C( -990171302), INT32_C(  444234765),
                            INT32_C( -651701039), INT32_C( -296257488), INT32_C( 1302666953), INT32_C( 1243668562)),
      easysimd_mm512_set_epi32(INT32_C( -228023402), INT32_C( 1737651280), INT32_C(  890037909), INT32_C(  822465192),
                            INT32_C( 1525557148), INT32_C( 1672658803), INT32_C( 1808682106), INT32_C( 1316739447),
                            INT32_C(  903813947), INT32_C(  221590740), INT32_C( 1668581990), INT32_C(-1092503304),
                            INT32_C( 1369460064), INT32_C( 1353181098), INT32_C(  652356799), INT32_C( -684439573)),
      easysimd_mm512_set_epi32(INT32_C(-1572306430), INT32_C(     147456), INT32_C(  269262977), INT32_C(     348680),
                            INT32_C( 1489766036), INT32_C(   53518626), INT32_C(      18504), INT32_C( 1140951061),
                            INT32_C(   76814882), INT32_C(  151072772), INT32_C( 1081084482), INT32_C(  442513416),
                            INT32_C( 1361069120), INT32_C( 1074225184), INT32_C(   77597833), INT32_C( 1109409858)) },
    { easysimd_mm512_set_epi32(INT32_C(-1043054173), INT32_C( -396216896), INT32_C(-1145802326), INT32_C( -804000246),
                            INT32_C( -145399860), INT32_C( -890427310), INT32_C( -401401997), INT32_C(  802016776),
                            INT32_C( 1929893502), INT32_C(   73827769), INT32_C(-1971097644), INT32_C(-1831682098),
                            INT32_C(  546355465), INT32_C( -199725455), INT32_C(  931867413), INT32_C(-1496909535)),
      easysimd_mm512_set_epi32(INT32_C(-1796636811), INT32_C(-1576316556), INT32_C( 1080356179), INT32_C(-1830141457),
                            INT32_C(-1444813077), INT32_C( 1282909316), INT32_C(  814589845), INT32_C(  563073613),
                            INT32_C( -161574330), INT32_C( 1115054069), INT32_C(-1922096352), INT32_C( 1283172543),
                            INT32_C( 1028016376), INT32_C( 1652445236), INT32_C( 1602581177), INT32_C(-1986713581)),
      easysimd_mm512_set_epi32(INT32_C(-2134884063), INT32_C(-1610477248), INT32_C(    2384130), INT32_C(-1878900726),
                            INT32_C(-1589557560), INT32_C( 1214586880), INT32_C(  536938257), INT32_C(  562938376),
                            INT32_C( 1913033286), INT32_C(    6686129), INT32_C(-2013060352), INT32_C(    5417102),
                            INT32_C(  536872968), INT32_C( 1612202032), INT32_C(  394338833), INT32_C(-2138822655)) },
    { easysimd_mm512_set_epi32(INT32_C(  213329535), INT32_C( -522060385), INT32_C( -710729699), INT32_C(  911515198),
                            INT32_C(-1475915599), INT32_C(-1846311235), INT32_C(-1624654725), INT32_C( -496488954),
                            INT32_C(-2105881976), INT32_C( -863113580), INT32_C( -870973395), INT32_C(-2135017149),
                            INT32_C( 1179500895), INT32_C(  102238134), INT32_C( 1890546920), INT32_C( 1651955955)),
      easysimd_mm512_set_epi32(INT32_C(-1460720620), INT32_C(-1283988079), INT32_C( 2139823103), INT32_C(-2058406982),
                            INT32_C( -677653135), INT32_C(  526832430), INT32_C(  918576849), INT32_C(-1987609349),
                            INT32_C( -819905099), INT32_C( 2043707434), INT32_C( 1005516756), INT32_C(  646673888),
                            INT32_C( -792085599), INT32_C(  923333390), INT32_C(  549762390), INT32_C( 1063027034)),
      easysimd_mm512_set_epi32(INT32_C(  145171476), INT32_C(-1604196975), INT32_C( 1434648605), INT32_C(   71569978),
                            INT32_C(-2147266511), INT32_C(  291668524), INT32_C(  369106001), INT32_C(-2147472382),
                            INT32_C(-2111829888), INT32_C( 1216380928), INT32_C(  134610948), INT32_C(    9056576),
                            INT32_C( 1078558977), INT32_C(  101188358), INT32_C(  545535040), INT32_C(  575963218)) },
    { easysimd_mm512_set_epi32(INT32_C(-1614227898), INT32_C(-1072924213), INT32_C(-2048516742), INT32_C(-1735505047),
                            INT32_C(  409846045), INT32_C( -501166301), INT32_C(  385735082), INT32_C(-1379445210),
                            INT32_C( 1301699864), INT32_C( -237316746), INT32_C( -173549926), INT32_C(-1638681430),
                            INT32_C( 1204990643), INT32_C( -623938106), INT32_C(  621663116), INT32_C(-2139715294)),
      easysimd_mm512_set_epi32(INT32_C( 1168648208), INT32_C(  679514223), INT32_C(-1255159953), INT32_C(-2016174737),
                            INT32_C( -817087094), INT32_C( 1605116212), INT32_C(  684814447), INT32_C( 1274003485),
                            INT32_C( 1881744290), INT32_C(  579021373), INT32_C( -658206082), INT32_C( 1152351107),
                            INT32_C( -539739024), INT32_C( 1438387923), INT32_C( -569943597), INT32_C(  -79238784)),
      easysimd_mm512_set_epi32(INT32_C(   92798976), INT32_C(       4171), INT32_C(-2061105814), INT32_C(-2138961559),
                            INT32_C(  139198728), INT32_C( 1109393696), INT32_C(   13717546), INT32_C(  164052996),
                            INT32_C( 1073742080), INT32_C(  545390644), INT32_C( -796618214), INT32_C(   67317890),
                            INT32_C( 1204826160), INT32_C( 1351353538), INT32_C(   67457408), INT32_C(-2142861056)) },
    { easysimd_mm512_set_epi32(INT32_C( -593800358), INT32_C( -124181915), INT32_C( 2110561848), INT32_C( 1255401496),
                            INT32_C( -282522813), INT32_C( -286538666), INT32_C(-2011412362), INT32_C(-1839527164),
                            INT32_C(-1330408299), INT32_C( 1769934774), INT32_C( -358481155), INT32_C( -123958768),
                            INT32_C( 1676106379), INT32_C(-1305862521), INT32_C( 1797940107), INT32_C(  653525737)),
      easysimd_mm512_set_epi32(INT32_C(-1432835313), INT32_C( 1661538833), INT32_C( 1372337273), INT32_C(-1604084834),
                            INT32_C( -921184393), INT32_C(-1395990480), INT32_C( 1258870002), INT32_C( -947895097),
                            INT32_C(-1351881935), INT32_C(   99634026), INT32_C( 2033361976), INT32_C( 1231716550),
                            INT32_C( -228173591), INT32_C(-1552770129), INT32_C( -338049103), INT32_C(-1393391283)),
      easysimd_mm512_set_epi32(INT32_C(-2003303670), INT32_C( 1611203073), INT32_C( 1372329016), INT32_C(    4425752),
                            INT32_C( -922680509), INT32_C(-1395998704), INT32_C(  134758514), INT32_C(-2113929212),
                            INT32_C(-1608286191), INT32_C(   24120098), INT32_C( 1747058744), INT32_C( 1208516608),
                            INT32_C( 1650874505), INT32_C(-1574823289), INT32_C( 1795703681), INT32_C(  619708489)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_and_si512(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_and_si512");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_and_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_and_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_and_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_and_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_and_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_and_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_and_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_and_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_and_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_and_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_and_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_and_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_and_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_and_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_and_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_and_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_and_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_and_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_and_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_and_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_and_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_and_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_and_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_and_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_and_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_and_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_and_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_and_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_and_si512)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
