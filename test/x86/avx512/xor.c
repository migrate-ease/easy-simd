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

#define EASYSIMD_TEST_X86_AVX512_INSN xor

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/xor.h>

static int
test_easysimd_mm_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(  1604416030), -INT32_C(    70199632),  INT32_C(   796477929),  INT32_C(  1086249892) },
      { -INT32_C(  1109738588), -INT32_C(  1258532507), -INT32_C(   381469744), -INT32_C(  1371623988) },
      {  INT32_C(   495203910),  INT32_C(  1328318421), -INT32_C(   969248199), -INT32_C(   293570968) } },
    { { -INT32_C(   363881414),  INT32_C(  1541742450),  INT32_C(   210394984), -INT32_C(   548583366) },
      {  INT32_C(  1771841283),  INT32_C(  1209899128), -INT32_C(  1624153901),  INT32_C(  1213099790) },
      { -INT32_C(  2083275975),  INT32_C(   335054602), -INT32_C(  1816444997), -INT32_C(  1761401036) } },
    { {  INT32_C(  2100534539),  INT32_C(   634919100), -INT32_C(  1305386121), -INT32_C(  1349419349) },
      {  INT32_C(   488123813), -INT32_C(  1738132027), -INT32_C(  1556572267),  INT32_C(   317490695) },
      {  INT32_C(  1613476014), -INT32_C(  1111609991),  INT32_C(   285865186), -INT32_C(  1115817812) } },
    { { -INT32_C(   544202973), -INT32_C(  1358665673),  INT32_C(  2002859723),  INT32_C(  1512501940) },
      { -INT32_C(   461947361),  INT32_C(   142400883),  INT32_C(  2091693173),  INT32_C(  1569626170) },
      {  INT32_C(  1005003068), -INT32_C(  1485261500),  INT32_C(   198017726),  INT32_C(   128477838) } },
    { { -INT32_C(   297984329),  INT32_C(  1386037639),  INT32_C(   751435383),  INT32_C(   260501488) },
      { -INT32_C(  1594622675),  INT32_C(  1353281755),  INT32_C(  1607226660), -INT32_C(  1531159827) },
      {  INT32_C(  1322181530),  INT32_C(    36974940),  INT32_C(  1929751379), -INT32_C(  1422215907) } },
    { {  INT32_C(     9632121), -INT32_C(  1303236549),  INT32_C(   534649902),  INT32_C(   959341579) },
      {  INT32_C(  1037640290), -INT32_C(  1232239982), -INT32_C(  1005233705),  INT32_C(   745067187) },
      {  INT32_C(  1028381467),  INT32_C(    81769129), -INT32_C(   607435271),  INT32_C(   356955832) } },
    { {  INT32_C(   103611339),  INT32_C(  1505328939), -INT32_C(  1502046309),  INT32_C(  1591715836) },
      {  INT32_C(  1536932297),  INT32_C(   319891515),  INT32_C(   886515585), -INT32_C(  1000259335) },
      {  INT32_C(  1572291074),  INT32_C(  1252546320), -INT32_C(  1833979878), -INT32_C(  1698765051) } },
    { {  INT32_C(  1724615995), -INT32_C(  1480621044),  INT32_C(   391002139), -INT32_C(  1468715553) },
      {  INT32_C(   570626278), -INT32_C(  1187703752),  INT32_C(   888015931), -INT32_C(  2013704372) },
      {  INT32_C(  1153998301),  INT32_C(   512397364),  INT32_C(   597701664),  INT32_C(   797729427) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_xor_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_xor_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_xor_epi32(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   868093254),  INT32_C(  1250251650), -INT32_C(  1335764820),  INT32_C(  1429022760) },
      UINT8_C( 56),
      {  INT32_C(   886973104),  INT32_C(   354011100), -INT32_C(   200718469), -INT32_C(  1985069929) },
      { -INT32_C(   787711106), -INT32_C(   662874624), -INT32_C(   553570633),  INT32_C(   186144347) },
      { -INT32_C(   868093254),  INT32_C(  1250251650), -INT32_C(  1335764820), -INT32_C(  2101974324) } },
    { {  INT32_C(  1413543544),  INT32_C(   963271102),  INT32_C(   942502816),  INT32_C(   767679407) },
      UINT8_C( 86),
      {  INT32_C(   592969421), -INT32_C(  1428476036), -INT32_C(  2046445008), -INT32_C(   906096174) },
      { -INT32_C(  1433971887),  INT32_C(   810270909), -INT32_C(   924875795),  INT32_C(   304024900) },
      {  INT32_C(  1413543544), -INT32_C(  1701777471),  INT32_C(  1322924509),  INT32_C(   767679407) } },
    { { -INT32_C(  2026539509), -INT32_C(   701427547), -INT32_C(  1654900789), -INT32_C(  1721345208) },
      UINT8_C(174),
      { -INT32_C(  1385479187),  INT32_C(   295345038), -INT32_C(  2024381574), -INT32_C(   124557182) },
      { -INT32_C(  1365435747), -INT32_C(  2089192628),  INT32_C(   734729936),  INT32_C(  1792631933) },
      { -INT32_C(  2026539509), -INT32_C(  1830557502), -INT32_C(  1398966870), -INT32_C(  1840640769) } },
    { {  INT32_C(   907560103),  INT32_C(  1497871071), -INT32_C(  1746821867), -INT32_C(  1550879739) },
      UINT8_C(142),
      { -INT32_C(  1596305107), -INT32_C(   529506870), -INT32_C(  1940022488), -INT32_C(  1204500364) },
      { -INT32_C(  1835570464),  INT32_C(  1336471985),  INT32_C(  1179926482), -INT32_C(    53151793) },
      {  INT32_C(   907560103), -INT32_C(  1344754565), -INT32_C(   905337606),  INT32_C(  1155547067) } },
    { {  INT32_C(   312258376),  INT32_C(   905121036),  INT32_C(   499208360), -INT32_C(   120195816) },
      UINT8_C( 95),
      {  INT32_C(  1578208109),  INT32_C(  1915772979), -INT32_C(  1404995916), -INT32_C(   101434038) },
      { -INT32_C(   419035174),  INT32_C(  1250900986), -INT32_C(   228348676), -INT32_C(   279880830) },
      { -INT32_C(  1189638985),  INT32_C(   952064969),  INT32_C(  1579342408),  INT32_C(   379938504) } },
    { {  INT32_C(   424501990),  INT32_C(  2005695938),  INT32_C(  1025756659), -INT32_C(   449374453) },
      UINT8_C( 30),
      {  INT32_C(  2014891069),  INT32_C(   141845339),  INT32_C(   562718406),  INT32_C(   436763064) },
      {  INT32_C(  1138565574),  INT32_C(  2067158189), -INT32_C(  1903791241), -INT32_C(   408065110) },
      {  INT32_C(   424501990),  INT32_C(  1933719542), -INT32_C(  1358162511), -INT32_C(    39513582) } },
    { { -INT32_C(  1839217353), -INT32_C(   291843032), -INT32_C(   233823174),  INT32_C(  1661802653) },
      UINT8_C( 57),
      {  INT32_C(  1055368938),  INT32_C(  1354064604),  INT32_C(  1408910312), -INT32_C(  1232411920) },
      {  INT32_C(   366877761), -INT32_C(   615461449), -INT32_C(   176602403),  INT32_C(   959372367) },
      {  INT32_C(   725203627), -INT32_C(   291843032), -INT32_C(   233823174), -INT32_C(  1885061441) } },
    { {  INT32_C(  1601639811),  INT32_C(  1605315703),  INT32_C(  1622321776), -INT32_C(   854180724) },
      UINT8_C( 89),
      { -INT32_C(  1039080459),  INT32_C(  1973414707),  INT32_C(  1103467620), -INT32_C(   658178366) },
      { -INT32_C(  1571871882),  INT32_C(  2098376659), -INT32_C(  1643547807),  INT32_C(  2130171529) },
      {  INT32_C(  1616887939),  INT32_C(  1605315703),  INT32_C(  1622321776), -INT32_C(  1506596789) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_xor_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_xor_epi32");
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
    easysimd__m128i r = easysimd_mm_mask_xor_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 82),
      {  INT32_C(  1135713443),  INT32_C(  1641105657),  INT32_C(  2080858544), -INT32_C(   990674568) },
      { -INT32_C(   511032597),  INT32_C(  2023881856),  INT32_C(   658325001), -INT32_C(   243722418) },
      {  INT32_C(           0),  INT32_C(   426816633),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(179),
      {  INT32_C(  1923888170),  INT32_C(  1730350342), -INT32_C(  1780441324), -INT32_C(   729701230) },
      {  INT32_C(   710173230),  INT32_C(  1110691076),  INT32_C(   563108362), -INT32_C(    36404781) },
      {  INT32_C(  1492669956),  INT32_C(   621854722),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(182),
      { -INT32_C(  1917030272),  INT32_C(   849421203), -INT32_C(  1480313085),  INT32_C(   450205880) },
      { -INT32_C(  1189150740), -INT32_C(  1916510157),  INT32_C(  1919018480),  INT32_C(   975724218) },
      {  INT32_C(           0), -INT32_C(  1083882592), -INT32_C(   710552845),  INT32_C(           0) } },
    { UINT8_C(206),
      {  INT32_C(   123848932), -INT32_C(  1609854103), -INT32_C(   279399849), -INT32_C(  2015595640) },
      { -INT32_C(   222653039),  INT32_C(  1071794265),  INT32_C(   133780649),  INT32_C(  1624650620) },
      {  INT32_C(           0), -INT32_C(  1612063952), -INT32_C(   392042754), -INT32_C(   418758412) } },
    { UINT8_C(251),
      { -INT32_C(   882546633),  INT32_C(   622986611), -INT32_C(   777186722), -INT32_C(  2090715666) },
      {  INT32_C(   937252079), -INT32_C(  1948246985),  INT32_C(  1208543253), -INT32_C(   180066114) },
      { -INT32_C(    55034664), -INT32_C(  1363011260),  INT32_C(           0),  INT32_C(  1982225744) } },
    { UINT8_C(208),
      { -INT32_C(  1371291479), -INT32_C(   183735837),  INT32_C(  1273224470), -INT32_C(  1808046528) },
      {  INT32_C(  1607168579),  INT32_C(   997480275), -INT32_C(   939934369), -INT32_C(  1432818175) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(175),
      {  INT32_C(  1150441691),  INT32_C(  1113294693), -INT32_C(   796744086),  INT32_C(  1410602977) },
      {  INT32_C(   967275490), -INT32_C(  1516707096), -INT32_C(   911843109), -INT32_C(   747089672) },
      {  INT32_C(  2100636473), -INT32_C(   406624883),  INT32_C(   421840561), -INT32_C(  2022947047) } },
    { UINT8_C(168),
      { -INT32_C(  1861412854),  INT32_C(   435900531), -INT32_C(   369373998),  INT32_C(  1405898720) },
      { -INT32_C(   667220490), -INT32_C(    21765986), -INT32_C(   688489338), -INT32_C(     8402443) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1397520875) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_xor_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_xor_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_xor_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 7174424735075124444), -INT64_C(  978143081419514139) },
      { -INT64_C( 4615264365177442559),  INT64_C(  876772771113460054) },
      { -INT64_C( 2565955798635533347), -INT64_C(  124380893962289229) } },
    { { -INT64_C( 6242966802893826128), -INT64_C( 4007119792821643420) },
      {  INT64_C( 1696719897335330632),  INT64_C( 1950871637418671830) },
      { -INT64_C( 4695157620309794568), -INT64_C( 3210720546936018510) } },
    { { -INT64_C(  979127685582944005), -INT64_C(  515725172852664761) },
      { -INT64_C( 6623299926522275523), -INT64_C( 5956928933446627378) },
      {  INT64_C( 6231925529588584902),  INT64_C( 6161902865117419913) } },
    { {  INT64_C( 2253443413152983236), -INT64_C( 6100964148652365652) },
      {  INT64_C( 3985896130812189150),  INT64_C( 8930291398183287438) },
      {  INT64_C( 2888330949232652570), -INT64_C( 3405915559225830878) } },
    { { -INT64_C( 2439523621391213779),  INT64_C( 3950940381905572794) },
      { -INT64_C( 3726391978781147479), -INT64_C( 6170499396345677524) },
      {  INT64_C( 1327471226468115844), -INT64_C( 7167069626328159594) } },
    { { -INT64_C(  547515844415968425), -INT64_C( 2758749232572620504) },
      { -INT64_C(  685614998110889870), -INT64_C( 2660160062519473683) },
      {  INT64_C( 1016373117844173605),  INT64_C(  190209691588503749) } },
    { { -INT64_C( 4425126818413755039),  INT64_C( 2194933809357698923) },
      { -INT64_C( 8984599251270082160), -INT64_C( 4597327086583141933) },
      {  INT64_C( 4739616463066408177), -INT64_C( 2429977920112124232) } },
    { { -INT64_C( 3663971448191543351), -INT64_C( 1047686828634052436) },
      {  INT64_C( 6942470958574486335),  INT64_C( 1361471686753928901) },
      { -INT64_C( 5945197992694590218), -INT64_C( 2048797907618812311) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_xor_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_xor_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_xor_epi64(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 6296841463456073657), -INT64_C( 7273775033257285773) },
      UINT8_C(127),
      { -INT64_C( 4364323046164382115),  INT64_C(  750072554643590360) },
      { -INT64_C( 2423126025098734324),  INT64_C(   56189043985323499) },
      {  INT64_C( 2103635020797078353),  INT64_C(  769927111372609843) } },
    { { -INT64_C( 4471656713588762147),  INT64_C( 1555106406350208647) },
      UINT8_C( 46),
      { -INT64_C(  734153393313611376), -INT64_C(  629772916923820090) },
      { -INT64_C( 6942401762521223003),  INT64_C( 6204199718951072018) },
      { -INT64_C( 4471656713588762147), -INT64_C( 6819815310930991404) } },
    { { -INT64_C( 8641271059959841089), -INT64_C( 1515971151670694880) },
      UINT8_C(160),
      { -INT64_C( 2050368516959171882), -INT64_C( 8158834155859642808) },
      { -INT64_C( 6601194068018928583),  INT64_C( 3356821472015901125) },
      { -INT64_C( 8641271059959841089), -INT64_C( 1515971151670694880) } },
    { {  INT64_C( 9183738175546627805),  INT64_C( 6108652524859104424) },
      UINT8_C( 47),
      {  INT64_C( 6608131927073920534), -INT64_C( 3069519532525442635) },
      { -INT64_C( 8907966781058124409),  INT64_C( 8470131115084978192) },
      { -INT64_C( 2318149218647907439), -INT64_C( 6850798151579913819) } },
    { {  INT64_C( 3881566864784274314), -INT64_C(  516186139036642879) },
      UINT8_C( 25),
      { -INT64_C( 6161762043251543582),  INT64_C( 1667315257586604453) },
      {  INT64_C( 1638408140751283938),  INT64_C( 8414272846259797129) },
      { -INT64_C( 4845365686000216320), -INT64_C(  516186139036642879) } },
    { { -INT64_C(  942273436588095695),  INT64_C( 6247528923958867256) },
      UINT8_C( 85),
      { -INT64_C( 3918430686773882551),  INT64_C( 6280498451656231543) },
      {  INT64_C( 5569147714922638258), -INT64_C( 2223344895932635870) },
      { -INT64_C( 8874512003901801733),  INT64_C( 6247528923958867256) } },
    { {  INT64_C( 7439828865636564840),  INT64_C( 2109522373116861407) },
      UINT8_C(242),
      {  INT64_C( 6675063037714050794), -INT64_C( 7433236709750049961) },
      {  INT64_C( 2828522852687760844),  INT64_C( 6901397499115876173) },
      {  INT64_C( 7439828865636564840), -INT64_C( 4102384614320481254) } },
    { { -INT64_C(  165926585325645858),  INT64_C( 5270694244941342709) },
      UINT8_C(145),
      {  INT64_C( 1866817819960519341), -INT64_C( 7897645226665263914) },
      {  INT64_C( 1460350922771708877), -INT64_C( 6595561271489080926) },
      {  INT64_C(  985287246681018720),  INT64_C( 5270694244941342709) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_xor_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_xor_epi64");
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
    easysimd__m128i r = easysimd_mm_mask_xor_epi64(src, k, a, b);

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
test_easysimd_mm_maskz_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(225),
      {  INT64_C( 6390447947533029772),  INT64_C( 7687235978183350377) },
      { -INT64_C( 5391364859213871417),  INT64_C( 1200115312946185724) },
      { -INT64_C( 1332668886174394549),  INT64_C(                   0) } },
    { UINT8_C( 59),
      {  INT64_C( 3806140491243187707), -INT64_C( 4707383967620691368) },
      { -INT64_C( 8715118929733266421),  INT64_C( 8447658307074380375) },
      { -INT64_C( 5485510659108980240), -INT64_C( 3778485719478937585) } },
    { UINT8_C(  9),
      {  INT64_C( 4286540300197158404),  INT64_C( 7025670452084498802) },
      {  INT64_C( 8155574305185567018),  INT64_C( 8114184073564008566) },
      {  INT64_C( 5355538825911619374),  INT64_C(                   0) } },
    { UINT8_C( 62),
      {  INT64_C( 8254018470017995840),  INT64_C( 1512048952378065566) },
      {  INT64_C( 3505246368668242627),  INT64_C( 1417241709466433294) },
      {  INT64_C(                   0),  INT64_C(  527182091719423376) } },
    { UINT8_C(162),
      { -INT64_C( 4872008887773745766),  INT64_C(   92132681419994276) },
      {  INT64_C( 5923977452069221893), -INT64_C( 1549456731616729905) },
      {  INT64_C(                   0), -INT64_C( 1497328814544678805) } },
    { UINT8_C(102),
      {  INT64_C( 5186981367176539954),  INT64_C( 4355676583256031003) },
      {  INT64_C( 8465829217065999226),  INT64_C(  752013005189110223) },
      {  INT64_C(                   0),  INT64_C( 3899509377927947988) } },
    { UINT8_C( 94),
      { -INT64_C( 2627930249963216497),  INT64_C(  153366924089226286) },
      { -INT64_C( 6796132735337116619), -INT64_C( 5324517932024613106) },
      {  INT64_C(                   0), -INT64_C( 5459667589035179232) } },
    { UINT8_C(228),
      { -INT64_C( 7101864723594861941),  INT64_C(  274726413163304893) },
      {  INT64_C( 1599320335563141338), -INT64_C( 7723687612006928793) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_xor_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_xor_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_xor_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_xor_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(   899812338), -INT32_C(  1067238292),  INT32_C(   965816200), -INT32_C(   657845366) },
      UINT8_C( 34),
      {  INT32_C(   210102884), -INT32_C(   607090888), -INT32_C(   494764932), -INT32_C(  1932259199) },
      { -INT32_C(   285734478), -INT32_C(  1669875348),  INT32_C(  1562816586), -INT32_C(   562037126) },
      {  INT32_C(   899812338),  INT32_C(  1202142804),  INT32_C(   965816200), -INT32_C(   657845366) } },
    { {  INT32_C(   568985065),  INT32_C(   167623308), -INT32_C(  1209303242),  INT32_C(  1111736208) },
      UINT8_C(200),
      { -INT32_C(   197906117),  INT32_C(  1480511656), -INT32_C(   170746633),  INT32_C(   551464987) },
      {  INT32_C(  1420624026),  INT32_C(  2089465597),  INT32_C(  1611415969), -INT32_C(  1087877244) },
      {  INT32_C(   568985065),  INT32_C(   167623308), -INT32_C(  1209303242), -INT32_C(  1611202657) } },
    { {  INT32_C(   682908800),  INT32_C(   612430381), -INT32_C(  1441181042), -INT32_C(  1664419838) },
      UINT8_C(248),
      {  INT32_C(   771092599), -INT32_C(  1144098438), -INT32_C(   851497346), -INT32_C(  1286734250) },
      { -INT32_C(  1512016462),  INT32_C(  1194525941),  INT32_C(   357162269),  INT32_C(   521004456) },
      {  INT32_C(   682908800),  INT32_C(   612430381), -INT32_C(  1441181042), -INT32_C(  1405084674) } },
    { {  INT32_C(  1330447061), -INT32_C(   234218636), -INT32_C(  1598076598), -INT32_C(    95220665) },
      UINT8_C(129),
      {  INT32_C(   930520883), -INT32_C(  1336558126), -INT32_C(   346527226),  INT32_C(  2059433848) },
      { -INT32_C(   521269308),  INT32_C(  1646977049), -INT32_C(  1414935905),  INT32_C(  1361879838) },
      { -INT32_C(   677867273), -INT32_C(   234218636), -INT32_C(  1598076598), -INT32_C(    95220665) } },
    { {  INT32_C(   361341506),  INT32_C(  1707466335), -INT32_C(  1068491448),  INT32_C(  1497043092) },
      UINT8_C( 31),
      {  INT32_C(   171456809),  INT32_C(   766089827), -INT32_C(   414493372),  INT32_C(   606772609) },
      {  INT32_C(    75710246),  INT32_C(   558688516),  INT32_C(  1253379385),  INT32_C(  1902710344) },
      {  INT32_C(   247137807),  INT32_C(   216363879), -INT32_C(  1375840131),  INT32_C(  1430492105) } },
    { { -INT32_C(  1434738105), -INT32_C(  2133318340), -INT32_C(    77061254), -INT32_C(   434138432) },
      UINT8_C(209),
      { -INT32_C(  1932137821),  INT32_C(  1153824311), -INT32_C(  1165226069),  INT32_C(   453115257) },
      { -INT32_C(  1638421383), -INT32_C(  1491543932), -INT32_C(   764931264), -INT32_C(   693940685) },
      {  INT32_C(   310527706), -INT32_C(  2133318340), -INT32_C(    77061254), -INT32_C(   434138432) } },
    { {  INT32_C(  1885501497),  INT32_C(   431236974), -INT32_C(  1328332745), -INT32_C(  1228155587) },
      UINT8_C(129),
      { -INT32_C(   100314078), -INT32_C(  2143638420),  INT32_C(  1672678420),  INT32_C(   664570287) },
      {  INT32_C(   311758059),  INT32_C(     4829120),  INT32_C(  1463679362), -INT32_C(   405211964) },
      { -INT32_C(   393193271),  INT32_C(   431236974), -INT32_C(  1328332745), -INT32_C(  1228155587) } },
    { { -INT32_C(  1243488952), -INT32_C(  1640686454), -INT32_C(   670963672),  INT32_C(  1560255857) },
      UINT8_C(169),
      {  INT32_C(  1147760277), -INT32_C(  1312396872), -INT32_C(  1686757721), -INT32_C(   723296778) },
      {  INT32_C(  1516148798), -INT32_C(  1249641267), -INT32_C(  1691984898), -INT32_C(   280657318) },
      {  INT32_C(   506984107), -INT32_C(  1640686454), -INT32_C(   670963672),  INT32_C(  1000791980) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_mask_xor_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_xor_ps");
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
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_mask_xor_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

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
test_easysimd_mm_maskz_xor_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(247),
      { -INT32_C(  2019705130), -INT32_C(  1754241701),  INT32_C(    19942984),  INT32_C(    26203230) },
      {  INT32_C(    56033067), -INT32_C(  1858750033),  INT32_C(  2092394977),  INT32_C(   963848803) },
      { -INT32_C(  2067054083),  INT32_C(   105241844),  INT32_C(  2106009513),  INT32_C(           0) } },
    { UINT8_C(232),
      {  INT32_C(  1984217105),  INT32_C(  2143214384), -INT32_C(   522338548),  INT32_C(  1292623438) },
      { -INT32_C(   302248396), -INT32_C(  1378972093),  INT32_C(  1863338565), -INT32_C(   816297538) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2108450832) } },
    { UINT8_C(  9),
      {  INT32_C(  2000242076), -INT32_C(  1031555069),  INT32_C(  1930453909),  INT32_C(  2108185966) },
      { -INT32_C(   390032038), -INT32_C(  1372754588), -INT32_C(   965960579), -INT32_C(  1865466892) },
      { -INT32_C(  1611018042),  INT32_C(           0),  INT32_C(           0), -INT32_C(   311990630) } },
    { UINT8_C(128),
      { -INT32_C(  1048377335), -INT32_C(   279493237), -INT32_C(  1285698987),  INT32_C(   135125618) },
      {  INT32_C(   141358491), -INT32_C(  1098507742), -INT32_C(  1028436857),  INT32_C(   625165084) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 75),
      {  INT32_C(   215475910),  INT32_C(   140625469), -INT32_C(    25553885),  INT32_C(   379159073) },
      {  INT32_C(   154706414),  INT32_C(  1955657511), -INT32_C(   309308758),  INT32_C(  1547285909) },
      {  INT32_C(    99567400),  INT32_C(  2096181530),  INT32_C(           0),  INT32_C(  1252014004) } },
    { UINT8_C(155),
      { -INT32_C(   690395120), -INT32_C(   554049079), -INT32_C(   553650341),  INT32_C(   852235920) },
      {  INT32_C(  1163515214), -INT32_C(  1192243867),  INT32_C(   307092829),  INT32_C(   648980758) },
      { -INT32_C(  1820279458),  INT32_C(  1712729260),  INT32_C(           0),  INT32_C(   342015878) } },
    { UINT8_C( 17),
      {  INT32_C(  1759247495), -INT32_C(   305939979), -INT32_C(   830627400),  INT32_C(  1126018926) },
      { -INT32_C(   693542391),  INT32_C(   791961938),  INT32_C(  1480935086),  INT32_C(  2070506484) },
      { -INT32_C(  1099784562),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(103),
      { -INT32_C(    44244156),  INT32_C(  1219905959),  INT32_C(  1991673031),  INT32_C(    75496097) },
      {  INT32_C(    72767139), -INT32_C(   793606774),  INT32_C(   902105802), -INT32_C(  1197719436) },
      { -INT32_C(   116738585), -INT32_C(  1744516051),  INT32_C(  1131646477),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_maskz_xor_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_xor_ps");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_maskz_xor_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_xor_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 6374246996544735471),  INT64_C( 4087535067914806528) },
      UINT8_C( 94),
      {  INT64_C( 6222105180717559400), -INT64_C(   98696254390363591) },
      { -INT64_C( 3686823945062933067), -INT64_C( 3819754652656097080) },
      { -INT64_C( 6374246996544735471),  INT64_C( 3773133743362880241) } },
    { {  INT64_C( 6501015476882490927), -INT64_C( 5828066961360511271) },
      UINT8_C( 40),
      {  INT64_C(  201004306266096956),  INT64_C( 3248835204993018213) },
      { -INT64_C( 3518532654560446445), -INT64_C( 6536841366716396652) },
      {  INT64_C( 6501015476882490927), -INT64_C( 5828066961360511271) } },
    { {  INT64_C( 1899236687279003902),  INT64_C(  103327624218132442) },
      UINT8_C( 16),
      { -INT64_C( 3820441719514508354), -INT64_C(  336968001461675046) },
      { -INT64_C( 4121698522994070656),  INT64_C( 4247656946547822026) },
      {  INT64_C( 1899236687279003902),  INT64_C(  103327624218132442) } },
    { {  INT64_C( 2184981682592035592), -INT64_C( 2299434646964631880) },
      UINT8_C(112),
      { -INT64_C( 6099494998393761338),  INT64_C( 7038952695098981617) },
      {  INT64_C( 1034352956518184452),  INT64_C( 5938363951051863348) },
      {  INT64_C( 2184981682592035592), -INT64_C( 2299434646964631880) } },
    { {  INT64_C( 1110202307072614431), -INT64_C(  790833695526908041) },
      UINT8_C(155),
      {  INT64_C( 7395529550595332011), -INT64_C( 2013878670385308768) },
      { -INT64_C( 4406472951843403226),  INT64_C( 3514401473644691142) },
      { -INT64_C( 6594639008236413555), -INT64_C( 3113978000727480986) } },
    { { -INT64_C( 4929401881104619200), -INT64_C(  809883983358096678) },
      UINT8_C(129),
      { -INT64_C( 5937304310442142012),  INT64_C( 8291798212314055977) },
      {  INT64_C( 7213158954905256704),  INT64_C( 9163180078789818610) },
      { -INT64_C( 3927071848155769404), -INT64_C(  809883983358096678) } },
    { {  INT64_C( 8960736451747896701), -INT64_C( 1175051447124980557) },
      UINT8_C(163),
      {  INT64_C( 4718613835772931647),  INT64_C( 3223024059297834005) },
      {  INT64_C( 5579955268550092494), -INT64_C( 7112262025958609591) },
      {  INT64_C(  870348700784937201), -INT64_C( 5623219052563363492) } },
    { { -INT64_C( 4547612340670510759),  INT64_C( 2900801232547209158) },
      UINT8_C( 29),
      {  INT64_C(   73231136873964992), -INT64_C( 4401144962480099993) },
      {  INT64_C( 5732007487619370275), -INT64_C( 6773181967658016469) },
      {  INT64_C( 5658778586093480163),  INT64_C( 2900801232547209158) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_mask_xor_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_xor_pd");
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
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_mask_xor_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

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
test_easysimd_mm_maskz_xor_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(182),
      { -INT64_C( 2851659321654933686),  INT64_C( 8908881140773182919) },
      { -INT64_C( 6440556459800992824),  INT64_C( 4456626232080193923) },
      {  INT64_C(                   0),  INT64_C( 5078839647394956356) } },
    { UINT8_C(164),
      { -INT64_C( 5849770947605109460), -INT64_C( 7718223420636262553) },
      {  INT64_C( 6062197473103502354),  INT64_C( 5392078819551495187) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(104),
      {  INT64_C( 3838307298564461932),  INT64_C( 2285149387101790024) },
      { -INT64_C( 5014214518090607595), -INT64_C( 2729142415731943116) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(120),
      {  INT64_C( 6524261185819162586),  INT64_C( 4200940245079633851) },
      { -INT64_C( 6595405061949447850),  INT64_C( 3161223822809641027) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(164),
      {  INT64_C( 2741560738301570428),  INT64_C( 3434353767939326615) },
      { -INT64_C( 7591980019828986378),  INT64_C( 7241808987864631792) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(127),
      {  INT64_C(  194109707356910624), -INT64_C( 7285206068432468823) },
      { -INT64_C(  952627964581820935),  INT64_C( 4903643896012776572) },
      { -INT64_C( 1119698481719938599), -INT64_C( 2384513575649894187) } },
    { UINT8_C(242),
      {  INT64_C( 1426150209099030148), -INT64_C( 2018357423261042364) },
      {  INT64_C( 4642295996671120730), -INT64_C( 8301918582642181479) },
      {  INT64_C(                   0),  INT64_C( 8013297466120422365) } },
    { UINT8_C(  8),
      {  INT64_C( 3491744877003648380),  INT64_C( 5449550670670494187) },
      { -INT64_C(  660295235822645378), -INT64_C(  990921152648008307) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_maskz_xor_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_xor_pd");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_maskz_xor_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  2029628447), -INT32_C(  2091543005),  INT32_C(   839745079),  INT32_C(   668697054), -INT32_C(   697132883),  INT32_C(   145493312),  INT32_C(   336157513),  INT32_C(   626250564) },
      {  INT32_C(  1252809255),  INT32_C(   684524272),  INT32_C(  1583012736),  INT32_C(   176502108),  INT32_C(   232847309),  INT32_C(  1310034948),  INT32_C(   727850983),  INT32_C(   424785394) },
      { -INT32_C(   844494394), -INT32_C(  1416063789),  INT32_C(  1817683383),  INT32_C(   761181314), -INT32_C(   611160224),  INT32_C(  1186562372),  INT32_C(  1063994542),  INT32_C(  1006790326) } },
    { {  INT32_C(     6552847),  INT32_C(  2133340671),  INT32_C(  1759347212), -INT32_C(  2072878153),  INT32_C(  1603359322), -INT32_C(   961698081), -INT32_C(  1192095803), -INT32_C(   724483132) },
      {  INT32_C(  1087649088),  INT32_C(  1925184614),  INT32_C(   903519614),  INT32_C(  1522093312),  INT32_C(  2126072479), -INT32_C(  1236965648),  INT32_C(   980301685), -INT32_C(  1173471366) },
      {  INT32_C(  1085786191),  INT32_C(   228052377),  INT32_C(  1560747890), -INT32_C(   557109577),  INT32_C(   556275909),  INT32_C(  1894301743), -INT32_C(  2103691088),  INT32_C(  1860140222) } },
    { { -INT32_C(   621092236),  INT32_C(  1548532446),  INT32_C(  1469130583),  INT32_C(   330386036), -INT32_C(  2070778988),  INT32_C(  1195038673), -INT32_C(  2004768754),  INT32_C(  1547866088) },
      {  INT32_C(  1329020273),  INT32_C(  1319863287),  INT32_C(   514145450),  INT32_C(   439506566), -INT32_C(  1818311487), -INT32_C(  1445275237),  INT32_C(  1764842369),  INT32_C(  1539732458) },
      { -INT32_C(  1781670139),  INT32_C(   317143337),  INT32_C(  1228151805),  INT32_C(   159587570),  INT32_C(   386707285), -INT32_C(   287306166), -INT32_C(   508497009),  INT32_C(   126155778) } },
    { { -INT32_C(  1481966160),  INT32_C(   737498496),  INT32_C(   390699665), -INT32_C(  1305379855), -INT32_C(   632959169),  INT32_C(   713236392),  INT32_C(  1687401594), -INT32_C(   658548440) },
      { -INT32_C(   679450282),  INT32_C(  1325561278),  INT32_C(    23481104),  INT32_C(   112433095),  INT32_C(   266467687), -INT32_C(  1824955368),  INT32_C(  1106758937),  INT32_C(  2098837286) },
      {  INT32_C(  1881838822),  INT32_C(  1693917246),  INT32_C(   372232577), -INT32_C(  1266488266), -INT32_C(   710658472), -INT32_C(  1178960976),  INT32_C(   627341667), -INT32_C(  1515786738) } },
    { { -INT32_C(   548103904),  INT32_C(   523130383),  INT32_C(  1746965921), -INT32_C(  1821387988), -INT32_C(   442281780), -INT32_C(   847717196), -INT32_C(   804360279),  INT32_C(  1196238886) },
      { -INT32_C(   802774591), -INT32_C(  1729145609),  INT32_C(   369168361), -INT32_C(  1347850013),  INT32_C(  1955876032), -INT32_C(   767423447), -INT32_C(  1566420868),  INT32_C(   988409720) },
      {  INT32_C(   259143905), -INT32_C(  2017393928),  INT32_C(  2116131400),  INT32_C(  1019651023), -INT32_C(  1858659316),  INT32_C(   523948189),  INT32_C(  1923891157),  INT32_C(  2107950942) } },
    { { -INT32_C(  2029383792),  INT32_C(  1294006884), -INT32_C(   329047799),  INT32_C(  1369181585), -INT32_C(  2101006247), -INT32_C(  1185609923), -INT32_C(   799279272),  INT32_C(  1997161959) },
      { -INT32_C(  1191242412),  INT32_C(   403054351), -INT32_C(   788174528), -INT32_C(   803036810),  INT32_C(   240314321),  INT32_C(  1204266991), -INT32_C(  2045303905), -INT32_C(  1124261272) },
      {  INT32_C(  1073027780),  INT32_C(  1428612459),  INT32_C(  1030113353), -INT32_C(  2118210329), -INT32_C(  1936205944), -INT32_C(    23945006),  INT32_C(  1447875783), -INT32_C(   872978545) } },
    { {  INT32_C(  1182137399),  INT32_C(  1532918555),  INT32_C(  1529635812), -INT32_C(   701805052),  INT32_C(   635731254), -INT32_C(   999511003),  INT32_C(   927630287), -INT32_C(   588036187) },
      {  INT32_C(  1596090436), -INT32_C(   927235613), -INT32_C(   383522844),  INT32_C(  1824476726), -INT32_C(   258890805),  INT32_C(   515177807),  INT32_C(   643169920), -INT32_C(  1979561658) },
      {  INT32_C(   425170035), -INT32_C(  1813644552), -INT32_C(  1307606016), -INT32_C(  1164705742), -INT32_C(   713695491), -INT32_C(   623357590),  INT32_C(   287276367),  INT32_C(  1458638563) } },
    { { -INT32_C(  1779882575), -INT32_C(  1973574490), -INT32_C(  1032617844), -INT32_C(  1725025586),  INT32_C(   629850070),  INT32_C(  1011105468), -INT32_C(  2090690243), -INT32_C(  1811061278) },
      {  INT32_C(   808056458),  INT32_C(   666535579), -INT32_C(   722915834),  INT32_C(   913119072), -INT32_C(  1839401002),  INT32_C(  1942921270),  INT32_C(   485896505),  INT32_C(   548406166) },
      { -INT32_C(  1514089669), -INT32_C(  1377361347),  INT32_C(   379235978), -INT32_C(  1354553938), -INT32_C(  1210693632),  INT32_C(  1334484618), -INT32_C(  1617647612), -INT32_C(  1262655884) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_xor_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_xor_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_xor_epi32(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1128528333), -INT32_C(   707341762),  INT32_C(  1487570298), -INT32_C(   770342016),  INT32_C(  1001769043), -INT32_C(  1866471213), -INT32_C(   808476113),  INT32_C(   634037234) },
      UINT8_C(169),
      {  INT32_C(  1508368774),  INT32_C(  1020509367), -INT32_C(   356766617),  INT32_C(   155029313),  INT32_C(   769423428), -INT32_C(   648254409), -INT32_C(   540333253),  INT32_C(  2089349621) },
      { -INT32_C(  1965723438), -INT32_C(  1815697364),  INT32_C(   360612820),  INT32_C(  1444920082),  INT32_C(  1803811636), -INT32_C(  1555701656),  INT32_C(    25301003), -INT32_C(   730002687) },
      { -INT32_C(   751660716), -INT32_C(   707341762),  INT32_C(  1487570298),  INT32_C(  1596077139),  INT32_C(  1001769043),  INT32_C(  2048494687), -INT32_C(   808476113), -INT32_C(  1460274444) } },
    { { -INT32_C(  1486990725), -INT32_C(   818207494), -INT32_C(  1176192601), -INT32_C(  1475345548),  INT32_C(  1729401855),  INT32_C(  2131384691),  INT32_C(  1803586921),  INT32_C(   322960792) },
      UINT8_C( 79),
      { -INT32_C(  1052132707), -INT32_C(  1368844043),  INT32_C(     2302717), -INT32_C(   956314830),  INT32_C(   943285983), -INT32_C(    22890383),  INT32_C(   899026232), -INT32_C(   377181876) },
      {  INT32_C(  1504366179), -INT32_C(   486075418),  INT32_C(  1759783478), -INT32_C(   718347274), -INT32_C(  1156683702),  INT32_C(  1505341472),  INT32_C(   160321469),  INT32_C(  1542591480) },
      { -INT32_C(  1729923842),  INT32_C(  1299057427),  INT32_C(  1757874379),  INT32_C(   332474564),  INT32_C(  1729401855),  INT32_C(  2131384691),  INT32_C(  1008222853),  INT32_C(   322960792) } },
    { { -INT32_C(   944464671), -INT32_C(   424952656), -INT32_C(   598831130), -INT32_C(  1129153166),  INT32_C(    91734245),  INT32_C(   761147504),  INT32_C(  2000088447), -INT32_C(   506255360) },
      UINT8_C(196),
      {  INT32_C(  1131718791), -INT32_C(   500540845),  INT32_C(   643106473),  INT32_C(  2013991352), -INT32_C(  1175973751),  INT32_C(  1547244911),  INT32_C(  1935454283),  INT32_C(   188235139) },
      {  INT32_C(   961457381), -INT32_C(  1340377081),  INT32_C(   936800383),  INT32_C(   196141442),  INT32_C(  1623496945), -INT32_C(   105054802),  INT32_C(   829233326),  INT32_C(   993830229) },
      { -INT32_C(   944464671), -INT32_C(   424952656),  INT32_C(   293828310), -INT32_C(  1129153166),  INT32_C(    91734245),  INT32_C(   761147504),  INT32_C(  1110550757),  INT32_C(   805607638) } },
    { {  INT32_C(  1484032849), -INT32_C(  2113368317), -INT32_C(  2101682688), -INT32_C(  1316132161), -INT32_C(  1341042430), -INT32_C(    39137713),  INT32_C(  1009653734),  INT32_C(   225930172) },
      UINT8_C(246),
      {  INT32_C(  2063165163),  INT32_C(  1299872878), -INT32_C(  1609761738), -INT32_C(   610091639), -INT32_C(  1658170417),  INT32_C(   344139773), -INT32_C(  1060061355),  INT32_C(   565632310) },
      { -INT32_C(  1298354109),  INT32_C(  1660884524), -INT32_C(  1677587693), -INT32_C(  1736989495), -INT32_C(   197811721),  INT32_C(   503888072), -INT32_C(  1361127304), -INT32_C(   103770698) },
      {  INT32_C(  1484032849),  INT32_C(   797272642),  INT32_C(  1007613733), -INT32_C(  1316132161),  INT32_C(  1763701304),  INT32_C(   176922421),  INT32_C(  1846437677), -INT32_C(   664385408) } },
    { {  INT32_C(  1907059781), -INT32_C(  1781290366),  INT32_C(  2134037942),  INT32_C(  1897376121),  INT32_C(   325405771),  INT32_C(  2083614212), -INT32_C(    64286650), -INT32_C(   352912475) },
      UINT8_C(103),
      {  INT32_C(  1290361761),  INT32_C(    50495278),  INT32_C(  1518109105), -INT32_C(   458887784), -INT32_C(  1041712813), -INT32_C(   100178454), -INT32_C(  1952512880), -INT32_C(  1678603782) },
      {  INT32_C(   317184996),  INT32_C(   185985370),  INT32_C(    40276586), -INT32_C(   756675713), -INT32_C(  1349267516), -INT32_C(   995517644), -INT32_C(  1722857057), -INT32_C(  1221312046) },
      {  INT32_C(  1578008645),  INT32_C(   135763572),  INT32_C(  1478104027),  INT32_C(  1897376121),  INT32_C(   325405771),  INT32_C(  1051655902),  INT32_C(   315641103), -INT32_C(   352912475) } },
    { {  INT32_C(  2009668636),  INT32_C(  1887624965), -INT32_C(   260904847), -INT32_C(  1195157004),  INT32_C(  1550276136), -INT32_C(  1876946447),  INT32_C(   740912986), -INT32_C(   857514320) },
      UINT8_C(122),
      { -INT32_C(  1937816659), -INT32_C(  1359089722), -INT32_C(  1146950302),  INT32_C(   132340656), -INT32_C(   738705470), -INT32_C(   835876513),  INT32_C(   276716210), -INT32_C(   360035779) },
      {  INT32_C(  1417022094),  INT32_C(  1526887417),  INT32_C(   286696801), -INT32_C(  1038550784), -INT32_C(  1751772872),  INT32_C(  1281737626),  INT32_C(  1532879645), -INT32_C(  1136269522) },
      {  INT32_C(  2009668636), -INT32_C(   167797697), -INT32_C(   260904847), -INT32_C(   973364560),  INT32_C(  1148071674), -INT32_C(  2109191483),  INT32_C(  1260632495), -INT32_C(   857514320) } },
    { { -INT32_C(   351224591), -INT32_C(  1874456017), -INT32_C(  1214095945), -INT32_C(  1887782314),  INT32_C(  1696993483), -INT32_C(   239956781), -INT32_C(  1655959697), -INT32_C(   396783114) },
      UINT8_C( 77),
      {  INT32_C(  2088620905),  INT32_C(  1983057177),  INT32_C(  1791880111),  INT32_C(  1966431333),  INT32_C(   239639426),  INT32_C(  1551776077),  INT32_C(   391256965), -INT32_C(   563856779) },
      {  INT32_C(   660267277), -INT32_C(  1633841425), -INT32_C(   569873799),  INT32_C(  1230192326),  INT32_C(   643275737),  INT32_C(  1518523861),  INT32_C(  1718736113),  INT32_C(   474273295) },
      {  INT32_C(  1529295460), -INT32_C(  1874456017), -INT32_C(  1262124586),  INT32_C(  1013342883),  INT32_C(  1696993483), -INT32_C(   239956781),  INT32_C(  1898172276), -INT32_C(   396783114) } },
    { { -INT32_C(  1505517641), -INT32_C(  1505369811),  INT32_C(   310660427),  INT32_C(  1683740555),  INT32_C(  1200271986),  INT32_C(  2040663176), -INT32_C(   253815839), -INT32_C(  1593039895) },
      UINT8_C(194),
      {  INT32_C(   837764944), -INT32_C(   629369460), -INT32_C(   261779943), -INT32_C(  1671181591),  INT32_C(  1629792852),  INT32_C(  1614978380), -INT32_C(  1622592900), -INT32_C(  1906120130) },
      { -INT32_C(  1094758094),  INT32_C(     9977063), -INT32_C(  1259209014),  INT32_C(   475026632),  INT32_C(  1266513150), -INT32_C(  1918124271),  INT32_C(   808252657),  INT32_C(   297701087) },
      { -INT32_C(  1505517641), -INT32_C(   622548629),  INT32_C(   310660427),  INT32_C(  1683740555),  INT32_C(  1200271986),  INT32_C(  2040663176), -INT32_C(  1352284531), -INT32_C(  1612946207) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_xor_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_xor_epi32");
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
    easysimd__m256i r = easysimd_mm256_mask_xor_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 79),
      {  INT32_C(  1681088779), -INT32_C(  2056105043),  INT32_C(  1835985335),  INT32_C(   776031840),  INT32_C(   688449078), -INT32_C(  1588685524),  INT32_C(   587462766), -INT32_C(  1452112994) },
      { -INT32_C(  1861311004), -INT32_C(  1357414152), -INT32_C(   920877719),  INT32_C(   167206611),  INT32_C(  1899167556),  INT32_C(    51544212),  INT32_C(   388371833),  INT32_C(  2109839513) },
      { -INT32_C(   180501265),  INT32_C(   711316309), -INT32_C(  1536004898),  INT32_C(   666243251),  INT32_C(           0),  INT32_C(           0),  INT32_C(   874900759),  INT32_C(           0) } },
    { UINT8_C( 61),
      {  INT32_C(  1328877519), -INT32_C(  1413946074),  INT32_C(  1602126081),  INT32_C(  2007271288),  INT32_C(   973804986),  INT32_C(  1035144743), -INT32_C(   858338508),  INT32_C(  1527337868) },
      { -INT32_C(  2002108574),  INT32_C(   624124452),  INT32_C(  1535488483), -INT32_C(   204330695),  INT32_C(  1697504574),  INT32_C(   547545580),  INT32_C(   955087020),  INT32_C(   781448908) },
      { -INT32_C(   945868627),  INT32_C(           0),  INT32_C(    83570914), -INT32_C(  2072596927),  INT32_C(  1596377220),  INT32_C(   487714763),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 53),
      { -INT32_C(  1621510339), -INT32_C(  1669169174),  INT32_C(   768990468), -INT32_C(  1939093329), -INT32_C(   696725259), -INT32_C(   343762573),  INT32_C(  2092415622), -INT32_C(  1967987123) },
      { -INT32_C(  2027353187), -INT32_C(  1910264950),  INT32_C(   935065736), -INT32_C(  1228659008),  INT32_C(  1787575543),  INT32_C(  1532301269),  INT32_C(   400035273), -INT32_C(  1868461325) },
      {  INT32_C(   410041504),  INT32_C(           0),  INT32_C(   443426188),  INT32_C(           0), -INT32_C(  1124799486), -INT32_C(  1328048474),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(149),
      {  INT32_C(  1981814731),  INT32_C(   872394299), -INT32_C(  1863043478),  INT32_C(   931638010),  INT32_C(  1158476086),  INT32_C(  1410295879), -INT32_C(   901306816), -INT32_C(  1839212601) },
      {  INT32_C(   688488430), -INT32_C(  1738733522),  INT32_C(   958943550),  INT32_C(   829468923), -INT32_C(   394822495),  INT32_C(   607946468),  INT32_C(  1945011116),  INT32_C(  1225150042) },
      {  INT32_C(  1596549157),  INT32_C(           0), -INT32_C(  1445173420),  INT32_C(           0), -INT32_C(  1384411753),  INT32_C(           0),  INT32_C(           0), -INT32_C(   614884963) } },
    { UINT8_C(205),
      {  INT32_C(   402354703),  INT32_C(   542479311),  INT32_C(  1813745340),  INT32_C(  2064469502),  INT32_C(  1247803076), -INT32_C(  1225358286), -INT32_C(  1072666254),  INT32_C(  2123258223) },
      { -INT32_C(  1684698676), -INT32_C(   641930467),  INT32_C(  2017843065), -INT32_C(   386706908),  INT32_C(  2066895432),  INT32_C(  1227958486),  INT32_C(    17383826),  INT32_C(  1736480667) },
      { -INT32_C(  1938883645),  INT32_C(           0),  INT32_C(   341727685), -INT32_C(  1812062246),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1055315744),  INT32_C(   420400884) } },
    { UINT8_C( 33),
      {  INT32_C(     4063765), -INT32_C(  1770383426), -INT32_C(  1363479972),  INT32_C(   938976229), -INT32_C(    32607531), -INT32_C(   460302429), -INT32_C(   125857440),  INT32_C(   656008721) },
      { -INT32_C(  1490593815), -INT32_C(   901930642),  INT32_C(  2021193875),  INT32_C(  1890611099), -INT32_C(  2073116959),  INT32_C(  1969880597), -INT32_C(  1586632560), -INT32_C(  1194817841) },
      { -INT32_C(  1491511812),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1845909066),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(221),
      { -INT32_C(  1857331216), -INT32_C(  1792731747), -INT32_C(    47145586),  INT32_C(   199204941),  INT32_C(   220226319), -INT32_C(  1247963444), -INT32_C(  1971044605), -INT32_C(   144229369) },
      {  INT32_C(   948548507),  INT32_C(  1456320200), -INT32_C(  1722483381), -INT32_C(  1381747810),  INT32_C(  1673184662),  INT32_C(  1578653787), -INT32_C(  1628922729),  INT32_C(  1939230936) },
      { -INT32_C(  1446843285),  INT32_C(           0),  INT32_C(  1684299973), -INT32_C(  1501850669),  INT32_C(  1855628953),  INT32_C(           0),  INT32_C(   342664084), -INT32_C(  2064552737) } },
    { UINT8_C(  3),
      { -INT32_C(   842290145),  INT32_C(  1998070393), -INT32_C(  1458196106),  INT32_C(   457163350), -INT32_C(   730422660),  INT32_C(  1449907386),  INT32_C(   221121213), -INT32_C(  1089428832) },
      { -INT32_C(   947070130),  INT32_C(  1933485309), -INT32_C(  1407429802), -INT32_C(  1832428523), -INT32_C(  1184481794), -INT32_C(   837823983),  INT32_C(  2094743260),  INT32_C(   775678944) },
      {  INT32_C(   172455761),  INT32_C(    69633668),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_xor_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_xor_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_xor_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 7330464224102111737), -INT64_C( 7939944293371108296),  INT64_C(  380539650787453854),  INT64_C( 4168593006022180131) },
      {  INT64_C( 7813333255835232682), -INT64_C( 1478354621574043858), -INT64_C( 5489880726801244672),  INT64_C( 7835495628425457987) },
      {  INT64_C(  708617018904866899),  INT64_C( 8841818371205628694), -INT64_C( 5289490565113793122),  INT64_C( 6153206277939776608) } },
    { {  INT64_C( 4759991533941656430), -INT64_C( 1746367616283569270), -INT64_C( 3636230596089117883), -INT64_C( 6150776643098970507) },
      { -INT64_C( 8651596313574762088),  INT64_C(  593388568452209517), -INT64_C( 1730394416690040922), -INT64_C(  917675105170010787) },
      { -INT64_C( 4187861767843900682), -INT64_C( 1153057682778552089),  INT64_C( 3059603292740742371),  INT64_C( 6478373507927358248) } },
    { {  INT64_C( 6224414394544066524), -INT64_C( 8541667253766405973),  INT64_C( 1742811396799288277), -INT64_C( 1274250339122328062) },
      {  INT64_C( 3081069622825840771),  INT64_C( 2978582857118150121), -INT64_C( 4092469397221732511),  INT64_C( 6840533363676473071) },
      {  INT64_C( 8981206347637292895), -INT64_C( 6907418259604984510), -INT64_C( 2370277142252987212), -INT64_C( 5710985666522404627) } },
    { { -INT64_C( 4103186222043974044),  INT64_C( 1474623983832049402),  INT64_C( 8612120025403287545), -INT64_C( 9186386543831963573) },
      { -INT64_C( 3043744109024752100),  INT64_C( 4170501758510061080), -INT64_C( 4575802953793400704),  INT64_C( 8773127923359479587) },
      {  INT64_C( 1354737985302493304),  INT64_C( 3284940467987738850), -INT64_C( 5189520070589089927), -INT64_C(  485527561274564760) } },
    { { -INT64_C( 5879671867690507081), -INT64_C( 6700131235836294517), -INT64_C( 7376848400057304214), -INT64_C( 6343149717423552990) },
      {  INT64_C( 3243613870928587966),  INT64_C( 4824197252051926315), -INT64_C( 1438967385529606408),  INT64_C( 4170443735352205641) },
      { -INT64_C( 8978888789782345719), -INT64_C( 2164357902251133024),  INT64_C( 8478001832487970194), -INT64_C( 7054608942135875733) } },
    { { -INT64_C( 1756805466571269424), -INT64_C( 8347095814355118321),  INT64_C( 6231575667158961094),  INT64_C( 1988181229501933538) },
      { -INT64_C( 4094286945070602870),  INT64_C( 5972573053596361922), -INT64_C( 4573865166975673310),  INT64_C( 1200970736372356420) },
      {  INT64_C( 2355586062933442394), -INT64_C( 2392538295493397555), -INT64_C( 7566789747803449372),  INT64_C(  810019900571710118) } },
    { {  INT64_C( 2546931143863056090), -INT64_C(   16224541949229336),  INT64_C( 5320357568243017783), -INT64_C( 9124149189925080374) },
      {  INT64_C( 2539303666491504312),  INT64_C( 5354320439727929201), -INT64_C(  192471494526272250),  INT64_C( 7444078200370310182) },
      {  INT64_C(   28692144810393698), -INT64_C( 5366032568390930023), -INT64_C( 5439924801152953039), -INT64_C( 1860497877712380180) } },
    { {  INT64_C( 4450329167837356387), -INT64_C( 4624392281744713099), -INT64_C( 8392423384607477416), -INT64_C( 1918067785295342261) },
      {  INT64_C( 1590411274989396488), -INT64_C( 3308051734800195936),  INT64_C( 5517442811070112417), -INT64_C( 7340401671118853446) },
      {  INT64_C( 3157296482768749419),  INT64_C( 7909917169454073045), -INT64_C( 4099970675964914695),  INT64_C( 9169328913369743345) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_xor_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_xor_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_xor_epi64(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 8879012988912173344),  INT64_C(  813612268590250086),  INT64_C( 8762317704558725372),  INT64_C( 3883623872014698485) },
      UINT8_C(229),
      {  INT64_C( 8972055307381374133), -INT64_C( 4332017556944998052),  INT64_C( 5827874014454607850),  INT64_C( 5745145732703630701) },
      { -INT64_C( 2731169851225908776),  INT64_C( 2260701690086793632), -INT64_C( 2636317248749655409), -INT64_C( 4356552130993057888) },
      { -INT64_C( 6441330844413745811),  INT64_C(  813612268590250086), -INT64_C( 8392123757379893915),  INT64_C( 3883623872014698485) } },
    { { -INT64_C( 8783294817433028925),  INT64_C(  751681649625979417),  INT64_C(  249541188798085164),  INT64_C( 6073887570532378268) },
      UINT8_C(190),
      { -INT64_C( 8486703987341093463),  INT64_C( 7997283491501082701),  INT64_C( 3772561892888975153), -INT64_C( 3127160414605614829) },
      { -INT64_C( 5195132615065846099),  INT64_C( 3177996500211264694), -INT64_C( 3072786720066180071), -INT64_C( 1935339234618433755) },
      { -INT64_C( 8783294817433028925),  INT64_C( 4820694435983680763), -INT64_C( 2233343619384181976),  INT64_C( 3584393696842743350) } },
    { { -INT64_C(  240333145893562231), -INT64_C( 3729609742530105322), -INT64_C( 8050379585521244241), -INT64_C( 3707865179600390034) },
      UINT8_C( 28),
      { -INT64_C( 7345547558457720882), -INT64_C( 8900773822910881784),  INT64_C( 3155192872512252941), -INT64_C( 3927074023687688353) },
      {  INT64_C( 5403706310497854458),  INT64_C( 8202339581863217906), -INT64_C( 6639829483147967364),  INT64_C( 8992147498902898693) },
      { -INT64_C(  240333145893562231), -INT64_C( 3729609742530105322), -INT64_C( 8641306588652296079), -INT64_C( 5383283175023171750) } },
    { {  INT64_C( 5906151360040045673),  INT64_C( 1800724741252779048),  INT64_C( 7287001628589672761),  INT64_C( 1650621685308925673) },
      UINT8_C(188),
      { -INT64_C( 4193577232928947512),  INT64_C( 2358612242320152844),  INT64_C( 9208423458503866432), -INT64_C( 1930564170903964342) },
      { -INT64_C( 3998254480795938811), -INT64_C( 7122436913557216078), -INT64_C( 3020428206424545218), -INT64_C( 8538678088906555011) },
      {  INT64_C( 5906151360040045673),  INT64_C( 1800724741252779048), -INT64_C( 6206051732963618690),  INT64_C( 7833386590624576567) } },
    { {  INT64_C( 1610540875044555733), -INT64_C( 1826130861832100946), -INT64_C( 1489967523490355473), -INT64_C(  152833046783548716) },
      UINT8_C( 18),
      { -INT64_C( 8457605172368601560),  INT64_C( 8782328596685564843),  INT64_C( 5835770862617713814),  INT64_C( 6733573264984527961) },
      {  INT64_C( 2338757396233447683),  INT64_C( 6153873946490087066),  INT64_C( 8741339683876654416),  INT64_C( 4567295693258606543) },
      {  INT64_C( 1610540875044555733),  INT64_C( 3208791559380143409), -INT64_C( 1489967523490355473), -INT64_C(  152833046783548716) } },
    { {  INT64_C( 4308429311203455284),  INT64_C( 2330770930733276483),  INT64_C( 2339178354883679965), -INT64_C( 3481134514076102859) },
      UINT8_C( 56),
      {  INT64_C( 2357938159894493657), -INT64_C( 1465339943408518083),  INT64_C( 7841712658460747568), -INT64_C( 3656969128334965631) },
      {  INT64_C( 7688485886044084302), -INT64_C( 1892952522390789191), -INT64_C( 3165653046293891448),  INT64_C( 4349944997237954716) },
      {  INT64_C( 4308429311203455284),  INT64_C( 2330770930733276483),  INT64_C( 2339178354883679965), -INT64_C( 1053334228510614499) } },
    { {  INT64_C( 6398349113327264051),  INT64_C( 7835947675171553828),  INT64_C( 3962643867170390743),  INT64_C( 2910773365256411451) },
      UINT8_C(198),
      {  INT64_C( 6257677274932611123), -INT64_C( 5649019783635881757), -INT64_C( 7199142805408856904), -INT64_C( 2868026992051077048) },
      {  INT64_C( 3957084433224079169), -INT64_C( 8992398954138557420),  INT64_C( 5311845132144705625),  INT64_C( 4698916874500851895) },
      {  INT64_C( 6398349113327264051),  INT64_C( 3651912028886439159), -INT64_C( 3053169226638146335),  INT64_C( 2910773365256411451) } },
    { {  INT64_C( 4110027403652678015), -INT64_C( 1066244148165868571), -INT64_C( 4486777054602423440), -INT64_C( 2788198373568646900) },
      UINT8_C( 21),
      { -INT64_C( 5059385477023749084),  INT64_C( 4652291726412708143),  INT64_C( 3395445555055952463), -INT64_C( 7092215614713287894) },
      {  INT64_C( 8178338194186751503),  INT64_C( 7111783128981688517),  INT64_C( 3535769944047615012),  INT64_C(  997059126677707711) },
      { -INT64_C( 3983934186288815573), -INT64_C( 1066244148165868571),  INT64_C( 2165844091777056363), -INT64_C( 2788198373568646900) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_xor_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_xor_epi64");
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
    easysimd__m256i r = easysimd_mm256_mask_xor_epi64(src, k, a, b);

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
test_easysimd_mm256_maskz_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(199),
      { -INT64_C( 8664582410286598713),  INT64_C( 8191818150558133588), -INT64_C( 5385371156125357931), -INT64_C( 4476644025661276371) },
      {  INT64_C( 3491678243758384313), -INT64_C( 3025697392656576164),  INT64_C(  226730114241365496),  INT64_C( 4034375885793311296) },
      { -INT64_C( 5209021099799916162), -INT64_C( 6364220192922502136), -INT64_C( 5303322121127137939),  INT64_C(                   0) } },
    { UINT8_C(159),
      { -INT64_C( 6164112220717636567), -INT64_C( 8424616395422432115),  INT64_C( 7259632903005570302),  INT64_C( 2313752859321236097) },
      { -INT64_C( 3350770186105350083),  INT64_C( 2901166066966096561), -INT64_C( 8316627843389348473), -INT64_C( 3640837521869801631) },
      {  INT64_C( 8866181239745526804), -INT64_C( 6676934800931787204), -INT64_C( 1717557407463956103), -INT64_C( 1340598900357362208) } },
    { UINT8_C(189),
      { -INT64_C( 8354227738701094223), -INT64_C( 5887366854825556288), -INT64_C( 8624598818086093690), -INT64_C( 3300605124353153202) },
      { -INT64_C( 4236410580926188921),  INT64_C( 5890097856274623493),  INT64_C(  714258951356511824),  INT64_C( 7865729053519315253) },
      {  INT64_C( 5276790669324685366),  INT64_C(                   0), -INT64_C( 9104365663320405290), -INT64_C( 4676623423644522885) } },
    { UINT8_C(224),
      {  INT64_C( 4079323707945064280), -INT64_C( 2353036895706698956), -INT64_C( 6676686906281182828),  INT64_C( 3292402725778205985) },
      {  INT64_C( 8399883184028030221),  INT64_C( 9175473923590755852), -INT64_C( 5272878898359439219),  INT64_C( 6893099333043182502) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(203),
      { -INT64_C( 2099943733875459731), -INT64_C( 7683619639174278916), -INT64_C( 4860544666050064659),  INT64_C( 2118501280007456613) },
      { -INT64_C( 1272891479816376025),  INT64_C( 6580354367008743959), -INT64_C( 4411051139294790485),  INT64_C( 7683355659028256467) },
      {  INT64_C(  904739455408523338), -INT64_C( 3599412068036218133),  INT64_C(                   0),  INT64_C( 8630778095010566582) } },
    { UINT8_C( 52),
      {  INT64_C( 6082977928069963148), -INT64_C(  604575836112017197),  INT64_C( 5115268323253397598), -INT64_C( 5950675538442354574) },
      {  INT64_C( 8599741840649916625), -INT64_C( 7636694053842117741), -INT64_C( 2278331655863117653), -INT64_C( 5761369601445795011) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 6441088519412387595),  INT64_C(                   0) } },
    { UINT8_C( 16),
      {  INT64_C( 2885165848127124491),  INT64_C( 8521363676410681490), -INT64_C( 6218691813144502857),  INT64_C( 2840145304656285914) },
      { -INT64_C( 2584318200466497811),  INT64_C( 3554663346201592769), -INT64_C( 2920485411704391664),  INT64_C( 2858831638630105449) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(187),
      { -INT64_C( 4056652252607929782),  INT64_C( 2240105296448033362),  INT64_C( 6907663515446469913),  INT64_C( 6304406218720502053) },
      { -INT64_C( 5409107504417635870), -INT64_C( 4858027598040911729), -INT64_C( 3930785250423099210), -INT64_C( 5778108643315872660) },
      {  INT64_C( 8312832539820401576), -INT64_C( 6664586079372598563),  INT64_C(                   0), -INT64_C(  527544851234656951) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_xor_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_xor_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_xor_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_xor_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1780197567),  INT32_C(    46075104), -INT32_C(  1960077960), -INT32_C(   241127194), -INT32_C(   250689637),  INT32_C(  1775350299), -INT32_C(  1839636139),  INT32_C(  1285469451) },
      UINT8_C( 13),
      { -INT32_C(  1897012862),  INT32_C(   906489761), -INT32_C(   904097254), -INT32_C(    94040525), -INT32_C(   837396708), -INT32_C(  1977385176), -INT32_C(  1902791208), -INT32_C(   711204525) },
      {  INT32_C(  1684310211), -INT32_C(  1852150921),  INT32_C(   811382781), -INT32_C(   517226043),  INT32_C(  1102004504), -INT32_C(  1714695488), -INT32_C(   634953593),  INT32_C(    78627393) },
      { -INT32_C(   360093375),  INT32_C(    46075104), -INT32_C(    96459289),  INT32_C(   458149878), -INT32_C(   250689637),  INT32_C(  1775350299), -INT32_C(  1839636139),  INT32_C(  1285469451) } },
    { { -INT32_C(  1050143926),  INT32_C(  2085749630),  INT32_C(  2142023354), -INT32_C(  2006919312), -INT32_C(   641134824),  INT32_C(  1752339681),  INT32_C(   910334452), -INT32_C(  1522863781) },
      UINT8_C(  5),
      { -INT32_C(  1501337949),  INT32_C(  1734410168), -INT32_C(  2083004500),  INT32_C(  1318870847), -INT32_C(  1120963287), -INT32_C(  2135779353), -INT32_C(   891557671), -INT32_C(   976256990) },
      { -INT32_C(  1637133338), -INT32_C(    33174702), -INT32_C(   377299798),  INT32_C(  1698111036),  INT32_C(  2049074835), -INT32_C(   688204547), -INT32_C(   543042116),  INT32_C(  1000632405) },
      {  INT32_C(   954742085),  INT32_C(  2085749630),  INT32_C(  1783956230), -INT32_C(  2006919312), -INT32_C(   641134824),  INT32_C(  1752339681),  INT32_C(   910334452), -INT32_C(  1522863781) } },
    { {  INT32_C(   383324355), -INT32_C(  2045452325), -INT32_C(   143681861),  INT32_C(  1197254580),  INT32_C(   197230349),  INT32_C(   283229011), -INT32_C(   437288304), -INT32_C(  1239378957) },
      UINT8_C(163),
      { -INT32_C(   645935878),  INT32_C(  2006189537),  INT32_C(   455904372),  INT32_C(  1730769896), -INT32_C(   256166859), -INT32_C(  1736324331),  INT32_C(  1300981434), -INT32_C(  2114895481) },
      { -INT32_C(   279285746), -INT32_C(   379130251),  INT32_C(  1661309562),  INT32_C(  1003105798),  INT32_C(  1999340898),  INT32_C(   168799312), -INT32_C(  1705469421), -INT32_C(   367310372) },
      {  INT32_C(   908442868), -INT32_C(  1628247148), -INT32_C(   143681861),  INT32_C(  1197254580),  INT32_C(   197230349), -INT32_C(  1836161211), -INT32_C(   437288304),  INT32_C(  1810499675) } },
    { {  INT32_C(   786003385), -INT32_C(   568901789), -INT32_C(   666821422), -INT32_C(  1408038070), -INT32_C(   517718383), -INT32_C(    18074645), -INT32_C(  1432861490),  INT32_C(  1184150413) },
      UINT8_C( 40),
      { -INT32_C(  1400146835), -INT32_C(  1468110453), -INT32_C(  1225632086), -INT32_C(  1471701143), -INT32_C(   158127933),  INT32_C(  1489277204), -INT32_C(   588943831),  INT32_C(  1862544130) },
      {  INT32_C(   723292063), -INT32_C(  1546413319),  INT32_C(  1515832817),  INT32_C(   654483556), -INT32_C(   585198135),  INT32_C(  1328996902),  INT32_C(  1395333969), -INT32_C(   423415993) },
      {  INT32_C(   786003385), -INT32_C(   568901789), -INT32_C(   666821422), -INT32_C(  1891287283), -INT32_C(   517718383),  INT32_C(   401765170), -INT32_C(  1432861490),  INT32_C(  1184150413) } },
    { { -INT32_C(  1206788161),  INT32_C(  1784407161),  INT32_C(   247838122),  INT32_C(   506906453), -INT32_C(  2080615331), -INT32_C(  2016202186), -INT32_C(  1797521843), -INT32_C(   310665682) },
      UINT8_C(125),
      {  INT32_C(  1911989644), -INT32_C(  1256496896), -INT32_C(   318035674), -INT32_C(  1286985377),  INT32_C(  1475005989), -INT32_C(  1616547423), -INT32_C(   355649204),  INT32_C(  1097317044) },
      {  INT32_C(  1622302303), -INT32_C(   451555905),  INT32_C(  1456611574),  INT32_C(  1862868298), -INT32_C(  1933052949), -INT32_C(  1322554267),  INT32_C(  1520236965),  INT32_C(   328926131) },
      {  INT32_C(   289733587),  INT32_C(  1784407161), -INT32_C(  1143404080), -INT32_C(   599575531), -INT32_C(   617792050),  INT32_C(   781065668), -INT32_C(  1336819479), -INT32_C(   310665682) } },
    { {  INT32_C(   544427361),  INT32_C(   268797978), -INT32_C(   211363671), -INT32_C(   530419467), -INT32_C(   932435613),  INT32_C(   997889941),  INT32_C(  1150621328),  INT32_C(  2069311513) },
      UINT8_C(125),
      {  INT32_C(  1385667530),  INT32_C(  2046535585),  INT32_C(  2121199118),  INT32_C(  2061586001),  INT32_C(  1360046778),  INT32_C(   987908900),  INT32_C(   273884896),  INT32_C(  1200475773) },
      {  INT32_C(   194585706), -INT32_C(   645622325), -INT32_C(   732433789), -INT32_C(    95471296),  INT32_C(   122380259), -INT32_C(  1975439958), -INT32_C(   795175853), -INT32_C(   854120605) },
      {  INT32_C(  1494138784),  INT32_C(   268797978), -INT32_C(  1439097715), -INT32_C(  2136049903),  INT32_C(  1448867161), -INT32_C(  1331468658), -INT32_C(  1060523341),  INT32_C(  2069311513) } },
    { {  INT32_C(   383299915), -INT32_C(   906994618), -INT32_C(  1902229682),  INT32_C(  1669918080), -INT32_C(   160771252),  INT32_C(  1417718529), -INT32_C(  1557849536), -INT32_C(  1938801599) },
      UINT8_C(237),
      { -INT32_C(  1540119992), -INT32_C(   621609582), -INT32_C(  2007334757), -INT32_C(   606814712), -INT32_C(   723727832),  INT32_C(  1679044938), -INT32_C(  1834633386),  INT32_C(  1870672167) },
      {  INT32_C(  1712567251),  INT32_C(  1262486960), -INT32_C(  1915512187), -INT32_C(  2123847848), -INT32_C(  1152039567), -INT32_C(   870356618),  INT32_C(  1197392928), -INT32_C(   910762251) },
      { -INT32_C(  1038085733), -INT32_C(   906994618),  INT32_C(    92871198),  INT32_C(  1522276688), -INT32_C(   160771252), -INT32_C(  1475651524), -INT32_C(   704941194), -INT32_C(  1506349102) } },
    { {  INT32_C(  1110428050),  INT32_C(  1401843662),  INT32_C(  1658872073),  INT32_C(  2061715465),  INT32_C(    53885069), -INT32_C(  1043376735),  INT32_C(   252194330), -INT32_C(  1629962740) },
      UINT8_C(135),
      {  INT32_C(  1985339655), -INT32_C(   796874641), -INT32_C(   791027064), -INT32_C(    44149563), -INT32_C(   543268470),  INT32_C(  1610178353),  INT32_C(   627771751), -INT32_C(   374600991) },
      {  INT32_C(  1516175851),  INT32_C(   824893353), -INT32_C(  2046753599), -INT32_C(   494706856), -INT32_C(   222223935), -INT32_C(   414073984), -INT32_C(  1525891900), -INT32_C(  1316046650) },
      {  INT32_C(   738910444), -INT32_C(   508921914),  INT32_C(  1457055305),  INT32_C(  2061715465),  INT32_C(    53885069), -INT32_C(  1043376735),  INT32_C(   252194330),  INT32_C(  1478668839) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_castps_si256(easysimd_mm256_mask_xor_ps(easysimd_mm256_castsi256_ps(src), k, easysimd_mm256_castsi256_ps(a), easysimd_mm256_castsi256_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_xor_ps");
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
    easysimd__m256i r = easysimd_mm256_castps_si256(easysimd_mm256_mask_xor_ps(easysimd_mm256_castsi256_ps(src), k, easysimd_mm256_castsi256_ps(a), easysimd_mm256_castsi256_ps(b)));

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
test_easysimd_mm256_maskz_xor_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(179),
      {  INT32_C(  1904848606),  INT32_C(  1689873573), -INT32_C(   563796063), -INT32_C(  1290281613), -INT32_C(   767369594), -INT32_C(    87330720),  INT32_C(   547681657),  INT32_C(  1960098710) },
      {  INT32_C(   249978217),  INT32_C(  1802674121),  INT32_C(   961140934),  INT32_C(  1022189749), -INT32_C(  1626460353),  INT32_C(   412735903),  INT32_C(  1681473230),  INT32_C(  1658392057) },
      {  INT32_C(  2138043319),  INT32_C(   265024364),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1296888249), -INT32_C(   497899009),  INT32_C(           0),  INT32_C(   369997423) } },
    { UINT8_C(106),
      {  INT32_C(  1580495039), -INT32_C(  1155162141),  INT32_C(  1232101096),  INT32_C(  2072554571), -INT32_C(  1810225221), -INT32_C(    27053120),  INT32_C(  2046281580),  INT32_C(  1608735136) },
      { -INT32_C(  1380050998), -INT32_C(  1620515914), -INT32_C(  1914120127),  INT32_C(  1074294917),  INT32_C(  1473520279), -INT32_C(  1051314347), -INT32_C(  1623568897),  INT32_C(  1912544935) },
      {  INT32_C(           0),  INT32_C(   609057877),  INT32_C(           0),  INT32_C(   998300878),  INT32_C(           0),  INT32_C(  1060439189), -INT32_C(   422737261),  INT32_C(           0) } },
    { UINT8_C( 53),
      { -INT32_C(  1595138371),  INT32_C(  1591839622), -INT32_C(   471634317), -INT32_C(  1720048778),  INT32_C(   804180472),  INT32_C(  1965994023),  INT32_C(   136105706), -INT32_C(  1975611955) },
      {  INT32_C(   824847019),  INT32_C(   680463285), -INT32_C(   267684998), -INT32_C(  1903589994),  INT32_C(  2126411862),  INT32_C(   317975592), -INT32_C(  2028269638),  INT32_C(  1192319388) },
      { -INT32_C(  1849281514),  INT32_C(           0),  INT32_C(   333978889),  INT32_C(           0),  INT32_C(  1364240814),  INT32_C(  1742560271),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(131),
      {  INT32_C(  1194883131),  INT32_C(  2059493383), -INT32_C(   267275925), -INT32_C(  1287217349),  INT32_C(  1255916637), -INT32_C(   972755529),  INT32_C(  1633848328), -INT32_C(   639325795) },
      {  INT32_C(   689970209), -INT32_C(   408690308), -INT32_C(   841501550), -INT32_C(  1317003692), -INT32_C(  1694802974),  INT32_C(  1365246025),  INT32_C(   699646860), -INT32_C(  1912432788) },
      {  INT32_C(  1847092250), -INT32_C(  1654488709),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1474707185) } },
    { UINT8_C(180),
      {  INT32_C(    53524258),  INT32_C(   261494874),  INT32_C(   224617455),  INT32_C(  1072698596),  INT32_C(   277383696), -INT32_C(  1382229270),  INT32_C(   605734285), -INT32_C(   354899768) },
      { -INT32_C(  1175582369),  INT32_C(   281576481), -INT32_C(   887215129),  INT32_C(  1342901824), -INT32_C(  2107599976), -INT32_C(    97452947), -INT32_C(  1994437951),  INT32_C(  1366620146) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   964868088),  INT32_C(           0), -INT32_C(  1830282872),  INT32_C(  1470899847),  INT32_C(           0), -INT32_C(  1146331334) } },
    { UINT8_C(  0),
      { -INT32_C(   434042270), -INT32_C(    20106541),  INT32_C(  1581160784),  INT32_C(   938905252), -INT32_C(   358254354), -INT32_C(   223567960), -INT32_C(  1226558018),  INT32_C(   196490665) },
      {  INT32_C(   334616640),  INT32_C(  1511112458), -INT32_C(    55029672), -INT32_C(   852250914), -INT32_C(   810035162),  INT32_C(   918643576),  INT32_C(  1122805401),  INT32_C(   458138587) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(123),
      { -INT32_C(    24826305), -INT32_C(  1873354944),  INT32_C(  1164858007),  INT32_C(  1584085894),  INT32_C(  1456880371), -INT32_C(  1561391620), -INT32_C(  1669516551), -INT32_C(  1088972416) },
      {  INT32_C(   129932487),  INT32_C(   311891067), -INT32_C(   313063833),  INT32_C(   877380417), -INT32_C(   108322051),  INT32_C(   681277999),  INT32_C(   751114668),  INT32_C(  2045566130) },
      { -INT32_C(   113528072), -INT32_C(  2101228741),  INT32_C(           0),  INT32_C(  1780545735), -INT32_C(  1352853490), -INT32_C(  1972078637), -INT32_C(  1330107563),  INT32_C(           0) } },
    { UINT8_C(121),
      { -INT32_C(  1091272278),  INT32_C(   505743128),  INT32_C(   559944286), -INT32_C(  2145414050), -INT32_C(  1699801057), -INT32_C(   850995276),  INT32_C(  2038395804),  INT32_C(   166918239) },
      { -INT32_C(  1832393095),  INT32_C(  1286663661),  INT32_C(  1567428863), -INT32_C(   992113499),  INT32_C(  1499368613),  INT32_C(     2532708),  INT32_C(  2004460824),  INT32_C(   394292125) },
      {  INT32_C(   741566419),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1153571067), -INT32_C(  1007577926), -INT32_C(   849317168),  INT32_C(   235329156),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_castps_si256(easysimd_mm256_maskz_xor_ps(k, easysimd_mm256_castsi256_ps(a), easysimd_mm256_castsi256_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_xor_ps");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_castps_si256(easysimd_mm256_maskz_xor_ps(k, easysimd_mm256_castsi256_ps(a), easysimd_mm256_castsi256_ps(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_xor_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 6987097089878563168), -INT64_C( 1966064696565077299), -INT64_C( 4642989144166825098), -INT64_C( 8814972574766748789) },
      UINT8_C(135),
      { -INT64_C( 7205565678448854517), -INT64_C(  714643028799702147),  INT64_C( 7447176679999417093), -INT64_C( 5811031494553175322) },
      { -INT64_C( 2947075602850338228),  INT64_C( 6991711273557051421), -INT64_C( 8225392995953805385), -INT64_C( 4335070371349699198) },
      {  INT64_C( 5411447048114114631), -INT64_C( 7560805248028544160), -INT64_C( 1549195263139128142), -INT64_C( 8814972574766748789) } },
    { { -INT64_C( 5145262953557084058),  INT64_C( 5935691102799007650), -INT64_C( 7351389618265878842),  INT64_C( 9024535355419810563) },
      UINT8_C( 79),
      { -INT64_C( 1838978449886527874),  INT64_C( 7422165111554537834), -INT64_C( 4451501024632476265),  INT64_C( 8051969878260298712) },
      { -INT64_C( 5381026079365945195),  INT64_C( 4502174274088325946), -INT64_C( 1763251545371838371),  INT64_C( 5602501813536710752) },
      {  INT64_C( 5992148833208735467),  INT64_C( 6447532329092950608),  INT64_C( 2719817284373550538),  INT64_C( 2485496913819849656) } },
    { {  INT64_C(  269385605991434173), -INT64_C(  521132659224171297),  INT64_C( 4935739465210345352), -INT64_C( 6293404539209525852) },
      UINT8_C( 92),
      {  INT64_C( 1823438250084446306), -INT64_C( 3506705863040248771),  INT64_C( 4707096606715145578), -INT64_C( 5026628135170560372) },
      { -INT64_C( 4104337775566232956), -INT64_C( 3048116248692605366), -INT64_C( 1619934269893548585),  INT64_C(  618691365543063684) },
      {  INT64_C(  269385605991434173), -INT64_C(  521132659224171297), -INT64_C( 6280796132005082947), -INT64_C( 5572117314106089976) } },
    { {  INT64_C( 9203953793739898474),  INT64_C( 1153788165344088002), -INT64_C( 2179504326402931079), -INT64_C(  138112322509108344) },
      UINT8_C( 17),
      { -INT64_C(  228339234450771441), -INT64_C( 8379714543353986520), -INT64_C( 2883771281194840934),  INT64_C( 3236513759630067590) },
      {  INT64_C( 2234933266781974874),  INT64_C( 1919958958527429141), -INT64_C( 2540162857475207780), -INT64_C( 1308400614921393522) },
      { -INT64_C( 2030888849060084907),  INT64_C( 1153788165344088002), -INT64_C( 2179504326402931079), -INT64_C(  138112322509108344) } },
    { {  INT64_C( 4532864408320480977), -INT64_C( 1166998196781039054),  INT64_C( 8052040180197448331), -INT64_C( 1999332069739891775) },
      UINT8_C(  8),
      { -INT64_C( 3071332267957572555),  INT64_C( 4201884706484523975),  INT64_C( 7153558918971208174),  INT64_C( 1126282376238242470) },
      {  INT64_C(  680216453130867207), -INT64_C( 7004471056047105872),  INT64_C( 1703040643722634832),  INT64_C( 8930164673516411140) },
      {  INT64_C( 4532864408320480977), -INT64_C( 1166998196781039054),  INT64_C( 8052040180197448331),  INT64_C( 8380929175187784610) } },
    { {  INT64_C( 8678304301725395747), -INT64_C( 6021924731160609505), -INT64_C(  125099159506415430),  INT64_C( 7464376925798921127) },
      UINT8_C(246),
      {  INT64_C( 8742390933867122093),  INT64_C( 2363406355958932030),  INT64_C( 3288218419045529453),  INT64_C(   51748874489019894) },
      { -INT64_C( 1631560168175798874),  INT64_C( 2259337731387371946), -INT64_C( 8996014412693605185),  INT64_C( 1715303499291727993) },
      {  INT64_C( 8678304301725395747),  INT64_C( 4581926899377116052), -INT64_C( 5871104333776387118),  INT64_C( 7464376925798921127) } },
    { {  INT64_C( 3748496316518957680), -INT64_C( 8693978318372994938), -INT64_C( 6318929403451510211), -INT64_C(   87464437615324608) },
      UINT8_C( 69),
      {  INT64_C( 2277980418597657282), -INT64_C( 6883912360576690574),  INT64_C( 2510326404706419439),  INT64_C( 1402219622239865069) },
      { -INT64_C( 5041841042359630647),  INT64_C( 5507177069470347166), -INT64_C( 6746188058000671570), -INT64_C( 8528048852445294271) },
      { -INT64_C( 6513670332222251509), -INT64_C( 8693978318372994938), -INT64_C( 9171931227908964799), -INT64_C(   87464437615324608) } },
    { {  INT64_C( 4075327314906435980), -INT64_C( 7128791460534254745), -INT64_C( 1223787578569001810), -INT64_C( 6082911607874988212) },
      UINT8_C(249),
      { -INT64_C( 7933431529947889658), -INT64_C( 5625526185870971953),  INT64_C( 3206650162057269752),  INT64_C( 3022750029502451153) },
      {  INT64_C( 2168358217798157431), -INT64_C( 5933407065126037702),  INT64_C( 2891419090665146526), -INT64_C( 2793899306914845713) },
      { -INT64_C( 8074597606736919439), -INT64_C( 7128791460534254745), -INT64_C( 1223787578569001810), -INT64_C( 1096375541836356034) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_castpd_si256(easysimd_mm256_mask_xor_pd(easysimd_mm256_castsi256_pd(src), k, easysimd_mm256_castsi256_pd(a), easysimd_mm256_castsi256_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_xor_pd");
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
    easysimd__m256i r = easysimd_mm256_castpd_si256(easysimd_mm256_mask_xor_pd(easysimd_mm256_castsi256_pd(src), k, easysimd_mm256_castsi256_pd(a), easysimd_mm256_castsi256_pd(b)));

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
test_easysimd_mm256_maskz_xor_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(195),
      {  INT64_C( 2458843720123825166), -INT64_C( 4716434193641740626),  INT64_C( 6696586108567881145),  INT64_C( 8177823768593980208) },
      { -INT64_C( 1500947458972079181),  INT64_C( 4308932751762695927),  INT64_C(  747066310855019884),  INT64_C( 6001963083227211895) },
      { -INT64_C( 3948529414066694211), -INT64_C( 8842893094750338983),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(237),
      {  INT64_C( 3333813299315367100), -INT64_C( 1529839462124596751),  INT64_C( 5596794720627246615), -INT64_C( 8231945898898389613) },
      { -INT64_C(  985692378178589540),  INT64_C(  394081380835394160), -INT64_C( 6969352257750418219),  INT64_C( 1281052491834995817) },
      { -INT64_C( 2587868251061964768),  INT64_C(                   0), -INT64_C( 3248172651772052798), -INT64_C( 7204221573847731718) } },
    { UINT8_C( 34),
      { -INT64_C( 8489261620510116446),  INT64_C( 8233280741505023583), -INT64_C( 5187551365429314379), -INT64_C( 6491589201930404719) },
      { -INT64_C( 9185489317279691661),  INT64_C( 3530030990098300500),  INT64_C( 3002154484354349932), -INT64_C( 8669190602565298681) },
      {  INT64_C(                   0),  INT64_C( 4809647924048117771),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(229),
      {  INT64_C(  504832009357918998),  INT64_C(  292930280002587763),  INT64_C( 9173768296818540668),  INT64_C( 2410299957728991925) },
      {  INT64_C( 8263771063210375582),  INT64_C( 6838758790093221979), -INT64_C(  121608967533722586), -INT64_C( 7171137988744912049) },
      {  INT64_C( 8480077944453659272),  INT64_C(                   0), -INT64_C( 9151260701648845734),  INT64_C(                   0) } },
    { UINT8_C(120),
      {  INT64_C( 2410528546791834440), -INT64_C( 4229897922352556032),  INT64_C( 5713956788979506184),  INT64_C( 7499031180768814489) },
      {  INT64_C( 8304343538171152684), -INT64_C( 6101843529509027508),  INT64_C(  485813085671022594),  INT64_C( 2243551224750082714) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 8589306578188021507) } },
    { UINT8_C( 58),
      { -INT64_C( 4908043458000896861),  INT64_C( 2422948426227347385), -INT64_C( 3080017067734331725), -INT64_C( 4530648580221240524) },
      { -INT64_C( 6532123897086030813),  INT64_C( 8470396522851449002), -INT64_C( 2048189296621577486),  INT64_C( 7438230536554306650) },
      {  INT64_C(                   0),  INT64_C( 6065484498925836051),  INT64_C(                   0), -INT64_C( 6474487012232485010) } },
    { UINT8_C(179),
      {  INT64_C( 1546498691703964817), -INT64_C( 8548309906051505529),  INT64_C(  802515600411204714),  INT64_C( 7313202394963090403) },
      { -INT64_C( 1788589445831648070), -INT64_C( 4791353554338904695), -INT64_C( 5221737037848725043),  INT64_C( 5151553430736721732) },
      { -INT64_C(  982942238949631957),  INT64_C( 3810041151177352974),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 26),
      { -INT64_C( 7863459164696453202),  INT64_C( 1239232447542319028),  INT64_C( 1015452004587515689),  INT64_C( 6644466058375409693) },
      { -INT64_C( 4223935469811749054),  INT64_C( 1428588769651148649),  INT64_C( 8237644835602216972),  INT64_C( 1240675519873084462) },
      {  INT64_C(                   0),  INT64_C(  207727101058388189),  INT64_C(                   0),  INT64_C( 5549034398887042099) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_castpd_si256(easysimd_mm256_maskz_xor_pd(k, easysimd_mm256_castsi256_pd(a), easysimd_mm256_castsi256_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_xor_pd");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_castpd_si256(easysimd_mm256_maskz_xor_pd(k, easysimd_mm256_castsi256_pd(a), easysimd_mm256_castsi256_pd(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_xor_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(  2140215653),  INT32_C(  1293320897), -INT32_C(  1520002534),  INT32_C(   294609697),  INT32_C(  1091317370),  INT32_C(   313266332), -INT32_C(  1143786207),  INT32_C(  1969865590),
         INT32_C(  1062829004), -INT32_C(   597668763),  INT32_C(  1683087224),  INT32_C(   739479593), -INT32_C(   468073407), -INT32_C(   573950189), -INT32_C(   974763361),  INT32_C(   527665652) },
      {  INT32_C(   816665849),  INT32_C(  1611972509), -INT32_C(  1280024725), -INT32_C(  1519923382), -INT32_C(  1677919607), -INT32_C(  1493407047),  INT32_C(  1584582304), -INT32_C(  1720041634),
         INT32_C(    70381333),  INT32_C(   746736592), -INT32_C(  1868567415),  INT32_C(  1303201364),  INT32_C(  2110232181), -INT32_C(  1733384071),  INT32_C(    69343053), -INT32_C(  1142043763) },
      {  INT32_C(  1329352092),  INT32_C(   755121500),  INT32_C(   382911857), -INT32_C(  1259819925), -INT32_C(   621751053), -INT32_C(  1269794267), -INT32_C(   442369151), -INT32_C(   334234584),
         INT32_C(   996709593), -INT32_C(   253622347), -INT32_C(   187812879),  INT32_C(  1639896701), -INT32_C(  1713483212),  INT32_C(  1164222314), -INT32_C(  1044095534), -INT32_C(  1533132167) } },
    { { -INT32_C(  1116734570),  INT32_C(  1613310165),  INT32_C(   813802444),  INT32_C(   429161548), -INT32_C(  1674708291),  INT32_C(  1572063288),  INT32_C(   653388597), -INT32_C(   887480686),
         INT32_C(  1388041778), -INT32_C(   190487029),  INT32_C(   304982209),  INT32_C(   851454399), -INT32_C(   666906362),  INT32_C(  1120951633),  INT32_C(  2076068325),  INT32_C(  1217737327) },
      { -INT32_C(   807130602),  INT32_C(   590835955),  INT32_C(   594477440), -INT32_C(   617573334),  INT32_C(   291751240),  INT32_C(  1236534164),  INT32_C(   789759175), -INT32_C(    74502962),
        -INT32_C(   134459010), -INT32_C(  1974206140), -INT32_C(  1593126896), -INT32_C(   699738723), -INT32_C(   399365675), -INT32_C(  1537291680),  INT32_C(   615597302), -INT32_C(  1491990022) },
      {  INT32_C(  1921765760),  INT32_C(  1126062118),  INT32_C(   334406220), -INT32_C(  1029379994), -INT32_C(  1924251659),  INT32_C(   336055212),  INT32_C(   165883890),  INT32_C(   815084124),
        -INT32_C(  1522038964),  INT32_C(  2129749839), -INT32_C(  1289258799), -INT32_C(   460653534),  INT32_C(   806222035), -INT32_C(   426862799),  INT32_C(  1594821907), -INT32_C(   276350059) } },
    { { -INT32_C(   289806681),  INT32_C(  1144433863),  INT32_C(  2078393105), -INT32_C(   621193477),  INT32_C(   197942373), -INT32_C(   778195138), -INT32_C(  1500149552), -INT32_C(   376058582),
         INT32_C(  1008015921), -INT32_C(   817846632), -INT32_C(  1418259563), -INT32_C(  1728720625), -INT32_C(   875417463), -INT32_C(   433179879), -INT32_C(   401326522), -INT32_C(   293254336) },
      { -INT32_C(  1611473168),  INT32_C(  1779296220),  INT32_C(  1625584338),  INT32_C(  1628081346),  INT32_C(  1516234414), -INT32_C(    67171629),  INT32_C(  1608454162), -INT32_C(   114353618),
        -INT32_C(   353092690),  INT32_C(  1984997008),  INT32_C(  1607474741), -INT32_C(  1472407712), -INT32_C(     3064222),  INT32_C(   279389725), -INT32_C(  1869192019),  INT32_C(   302689758) },
      {  INT32_C(  1900755031),  INT32_C(   775637787),  INT32_C(   453358019), -INT32_C(  1141646791),  INT32_C(  1368636107),  INT32_C(   711108077), -INT32_C(   112551742),  INT32_C(   280680196),
        -INT32_C(   689888865), -INT32_C(  1190133752), -INT32_C(   190378592),  INT32_C(   818480751),  INT32_C(   872615659), -INT32_C(   158788348),  INT32_C(  2021808875), -INT32_C(    57678178) } },
    { {  INT32_C(   419761061),  INT32_C(  1205817843), -INT32_C(  1808728463), -INT32_C(   270741600),  INT32_C(    62568967), -INT32_C(  1758665902), -INT32_C(  1620063715),  INT32_C(   975059798),
         INT32_C(   672464530), -INT32_C(  1296717020),  INT32_C(   968463109), -INT32_C(   943239776),  INT32_C(  1563835967),  INT32_C(  1537408671),  INT32_C(   771343793),  INT32_C(  1593811067) },
      { -INT32_C(  1519602963),  INT32_C(  1238941430), -INT32_C(   210338261), -INT32_C(   240423445), -INT32_C(  1881846659),  INT32_C(   210890163),  INT32_C(   374811810), -INT32_C(  1712882298),
         INT32_C(  1497831658),  INT32_C(  2098029338), -INT32_C(   368469787),  INT32_C(  2040566000), -INT32_C(  1929731099),  INT32_C(  1291609713), -INT32_C(  1096372677), -INT32_C(   147699721) },
      { -INT32_C(  1133921976),  INT32_C(   235400965),  INT32_C(  1732673626),  INT32_C(   511156811), -INT32_C(  1938819462), -INT32_C(  1682108703), -INT32_C(  1992759617), -INT32_C(  1543944496),
         INT32_C(  1901323896), -INT32_C(   809973698), -INT32_C(   743437344), -INT32_C(  1100489392), -INT32_C(   775119398),  INT32_C(   392075502), -INT32_C(  1822463094), -INT32_C(  1446123636) } },
    { { -INT32_C(  1891676286), -INT32_C(  1841226010),  INT32_C(  1540983227), -INT32_C(  1986737150),  INT32_C(   397242270),  INT32_C(   823916557), -INT32_C(  1551338568),  INT32_C(  1077412441),
         INT32_C(  1885334403), -INT32_C(  1567613993), -INT32_C(  1973232663),  INT32_C(  1561190391),  INT32_C(   194947553),  INT32_C(   332812599),  INT32_C(  1009120275),  INT32_C(  1926064119) },
      { -INT32_C(  1175618672), -INT32_C(  1361499621),  INT32_C(  2080692609),  INT32_C(    36764393), -INT32_C(   844078996),  INT32_C(    35128981), -INT32_C(  1577126054), -INT32_C(  1409287093),
         INT32_C(  1241153522),  INT32_C(  1667693632), -INT32_C(  2140278031), -INT32_C(   250677594),  INT32_C(   771501117),  INT32_C(  1049317354), -INT32_C(   683384889),  INT32_C(  1735623030) },
      {  INT32_C(   919745554),  INT32_C(  1016607997),  INT32_C(   668815418), -INT32_C(  1952172309), -INT32_C(   635629582),  INT32_C(   855899800),  INT32_C(    41321698), -INT32_C(   339214830),
         INT32_C(   967152753), -INT32_C(  1040783465),  INT32_C(   168765720), -INT32_C(  1409078959),  INT32_C(   643991004),  INT32_C(   761077469), -INT32_C(   345920556),  INT32_C(   364832385) } },
    { { -INT32_C(  1727546569), -INT32_C(  2058639326),  INT32_C(  1338749765),  INT32_C(   373465026), -INT32_C(   671124678), -INT32_C(  1919302723), -INT32_C(  1233004256), -INT32_C(  1559179697),
        -INT32_C(   107798480), -INT32_C(   385456720), -INT32_C(   898044456), -INT32_C(  1625696711), -INT32_C(   376937145), -INT32_C(  1132367764),  INT32_C(   902481945),  INT32_C(   792056806) },
      { -INT32_C(  1841227009),  INT32_C(  1539669343), -INT32_C(   709735776),  INT32_C(  1652239698), -INT32_C(  1711680384), -INT32_C(  1429694270),  INT32_C(   525789643),  INT32_C(  1259471626),
         INT32_C(  1193841178),  INT32_C(   812285822),  INT32_C(   876026405),  INT32_C(   332646634), -INT32_C(  1055303881),  INT32_C(  1713064637), -INT32_C(  1819389369),  INT32_C(   379883641) },
      {  INT32_C(   189178312), -INT32_C(   561064579), -INT32_C(  1703282203),  INT32_C(  1949938320),  INT32_C(  1309059002),  INT32_C(   659643263), -INT32_C(  1445515029), -INT32_C(   402469563),
        -INT32_C(  1094995414), -INT32_C(   647226674), -INT32_C(    28313603), -INT32_C(  1932910893),  INT32_C(   680604272), -INT32_C(   627426607), -INT32_C(  1505453474),  INT32_C(   965821343) } },
    { { -INT32_C(   349030303),  INT32_C(  1785046111), -INT32_C(  1042757457),  INT32_C(  1737927527),  INT32_C(  1161594549),  INT32_C(  1160192042), -INT32_C(   184434809), -INT32_C(  1720055340),
        -INT32_C(   339521427), -INT32_C(  1970209792), -INT32_C(  1248287430), -INT32_C(   360789525),  INT32_C(  1913349951),  INT32_C(   360975298),  INT32_C(  1185309231),  INT32_C(  1025588088) },
      { -INT32_C(   388091286),  INT32_C(  1475451470), -INT32_C(   582153313), -INT32_C(  1991214562), -INT32_C(  1091731760), -INT32_C(   868334684), -INT32_C(  1166550823), -INT32_C(   734386821),
        -INT32_C(   499790847), -INT32_C(  1911490173),  INT32_C(   998569952),  INT32_C(  1668173516),  INT32_C(   240093475),  INT32_C(  1396415836),  INT32_C(  1050460006), -INT32_C(  1184904202) },
      {  INT32_C(    65800715),  INT32_C(  1033125393),  INT32_C(   479576880), -INT32_C(   288960135), -INT32_C(    70125467), -INT32_C(  1994824818),  INT32_C(  1333159774),  INT32_C(  1296051375),
         INT32_C(   167156844),  INT32_C(    75498883), -INT32_C(  1910746406), -INT32_C(  1995405529),  INT32_C(  2084892188),  INT32_C(  1186961054),  INT32_C(  2017123657), -INT32_C(  2072055666) } },
    { {  INT32_C(  1950700306), -INT32_C(   237283605), -INT32_C(  1190591724),  INT32_C(     2981687), -INT32_C(   576818779), -INT32_C(    20979385),  INT32_C(   750065778), -INT32_C(   830997516),
        -INT32_C(   852723094),  INT32_C(  1566901338),  INT32_C(   353305803),  INT32_C(     6679193),  INT32_C(  1342823370), -INT32_C(  1377161447),  INT32_C(  1791982968),  INT32_C(   243243187) },
      { -INT32_C(   928073445), -INT32_C(    35749180), -INT32_C(   356403761), -INT32_C(   863827258),  INT32_C(   822300177),  INT32_C(  1598338091), -INT32_C(  1771112626),  INT32_C(   117434161),
         INT32_C(   676273718),  INT32_C(   693777629),  INT32_C(   320965188),  INT32_C(  1812143731), -INT32_C(  1549201693), -INT32_C(  1144932349), -INT32_C(  2083228063),  INT32_C(  1723727495) },
      { -INT32_C(  1125389815),  INT32_C(   201708591),  INT32_C(  1405662939), -INT32_C(   860981775), -INT32_C(   325238860), -INT32_C(  1577364116), -INT32_C(  1159991492), -INT32_C(   930603835),
        -INT32_C(   446471076),  INT32_C(  1950298247),  INT32_C(   103714447),  INT32_C(  1818686698), -INT32_C(   207563479),  INT32_C(   371955482), -INT32_C(   384103655),  INT32_C(  1757570612) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_castps_si512(easysimd_mm512_xor_ps(easysimd_mm512_castsi512_ps(a), easysimd_mm512_castsi512_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_xor_ps");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_xor_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(  1881296091),  INT32_C(   418875182),  INT32_C(   134011108),  INT32_C(   720570282), -INT32_C(  1853596235), -INT32_C(   496598853), -INT32_C(  1204688658),  INT32_C(  1068536932),
         INT32_C(   246469344),  INT32_C(  1026008921),  INT32_C(   692331391), -INT32_C(   548194518),  INT32_C(  1349638293),  INT32_C(  1177737048),  INT32_C(   603874239), -INT32_C(   681332745) },
      UINT16_C( 4993),
      {  INT32_C(   230349798),  INT32_C(  1563441432), -INT32_C(  1231791517), -INT32_C(  1433523911), -INT32_C(  1417550215),  INT32_C(   722419756), -INT32_C(   942012828), -INT32_C(   992322594),
         INT32_C(  1339135030),  INT32_C(   816579021), -INT32_C(  1796784037), -INT32_C(   482380438), -INT32_C(  2003845028),  INT32_C(  1706270209), -INT32_C(  2111009372),  INT32_C(   524682985) },
      {  INT32_C(  1735268250),  INT32_C(  1939347992), -INT32_C(   989298853),  INT32_C(  1353205748),  INT32_C(   148453127),  INT32_C(  2053999829),  INT32_C(    66885914),  INT32_C(   958546847),
         INT32_C(  1939968346),  INT32_C(   115751339), -INT32_C(  1395921224),  INT32_C(  1040020533), -INT32_C(  2126129493),  INT32_C(  2096870242), -INT32_C(   327157940), -INT32_C(  1792695750) },
      {  INT32_C(  1792330876),  INT32_C(   418875182),  INT32_C(   134011108),  INT32_C(   720570282), -INT32_C(  1853596235), -INT32_C(   496598853), -INT32_C(  1204688658), -INT32_C(    34072511),
         INT32_C(  1013974380),  INT32_C(   910833766),  INT32_C(   692331391), -INT32_C(   548194518),  INT32_C(   164239095),  INT32_C(  1177737048),  INT32_C(   603874239), -INT32_C(   681332745) } },
    { { -INT32_C(   569850317), -INT32_C(  1209667841),  INT32_C(   308588765), -INT32_C(   800104156), -INT32_C(  1722706633), -INT32_C(  1810478008),  INT32_C(  2122356035),  INT32_C(  1779672631),
         INT32_C(  1816730476), -INT32_C(   433902071),  INT32_C(    66684894),  INT32_C(   533940456),  INT32_C(   632890589), -INT32_C(  1279602832), -INT32_C(  1674495388),  INT32_C(  1292256480) },
      UINT16_C(20319),
      { -INT32_C(   595760711),  INT32_C(  1214536783),  INT32_C(   848383071),  INT32_C(   626421356),  INT32_C(  1324664468), -INT32_C(  1417127815), -INT32_C(    68195852),  INT32_C(  1867141046),
         INT32_C(   122407096), -INT32_C(  2091864284),  INT32_C(  1756750332), -INT32_C(   477295793),  INT32_C(  1244758481), -INT32_C(   822756647), -INT32_C(   641080029), -INT32_C(   330820812) },
      { -INT32_C(      814117),  INT32_C(  1065501507),  INT32_C(  1957115684),  INT32_C(   324482114), -INT32_C(  1906406987),  INT32_C(  1717326659),  INT32_C(  1816077624),  INT32_C(   341346105),
         INT32_C(  1561611290), -INT32_C(  1264806257),  INT32_C(   271074254),  INT32_C(   740589431),  INT32_C(  1270579720),  INT32_C(   246486997),  INT32_C(  1987768381), -INT32_C(  1836395657) },
      {  INT32_C(   596572770),  INT32_C(  2011569932),  INT32_C(  1177975675),  INT32_C(   906058286), -INT32_C(  1062580447), -INT32_C(  1810478008), -INT32_C(  1747956532),  INT32_C(  1779672631),
         INT32_C(  1516209314),  INT32_C(   936130987),  INT32_C(  2023596594), -INT32_C(   810978248),  INT32_C(   632890589), -INT32_C(  1279602832), -INT32_C(  1347218146),  INT32_C(  1292256480) } },
    { { -INT32_C(  1360028129),  INT32_C(    56790069),  INT32_C(  1192463055),  INT32_C(   309540618), -INT32_C(  1889653063), -INT32_C(  2086858938),  INT32_C(  2012813056),  INT32_C(   151618538),
         INT32_C(  1471740194),  INT32_C(  1415191173), -INT32_C(  1348768347),  INT32_C(  1589710757), -INT32_C(  2081611971),  INT32_C(   788957743), -INT32_C(  1935212638), -INT32_C(  1516851069) },
      UINT16_C(20137),
      {  INT32_C(  1449668348),  INT32_C(   516099458),  INT32_C(  2116905148), -INT32_C(  1264751929), -INT32_C(   197145362), -INT32_C(  1561009669),  INT32_C(    55736173),  INT32_C(   408025884),
        -INT32_C(  1418806743), -INT32_C(  2067189305),  INT32_C(  1627584154),  INT32_C(  1326817121),  INT32_C(  1732465772), -INT32_C(  1559677131), -INT32_C(   895067218),  INT32_C(  2145581142) },
      {  INT32_C(  2032881842),  INT32_C(   486405250),  INT32_C(  1283325931),  INT32_C(   194744991),  INT32_C(   477290215), -INT32_C(   994083818),  INT32_C(   781149911),  INT32_C(   279802206),
         INT32_C(  1149884610), -INT32_C(  1201633331),  INT32_C(   621075846),  INT32_C(  1446027119), -INT32_C(  1821138051), -INT32_C(   161992161), -INT32_C(   165353576),  INT32_C(   436654680) },
      {  INT32_C(   792952398),  INT32_C(    56790069),  INT32_C(  1192463055), -INT32_C(  1090062248), -INT32_C(  1889653063),  INT32_C(  1716231661),  INT32_C(  2012813056),  INT32_C(   150768194),
         INT32_C(  1471740194),  INT32_C(  1017755146),  INT32_C(  1141254940),  INT32_C(   421855246), -INT32_C(  2081611971),  INT32_C(   788957743),  INT32_C(  1015200822), -INT32_C(  1516851069) } },
    { {  INT32_C(  2002686122), -INT32_C(  1657815529),  INT32_C(   197342107),  INT32_C(  1348596690), -INT32_C(  1243360106), -INT32_C(  1616102649),  INT32_C(  2073481250),  INT32_C(  1301650594),
         INT32_C(  1136981036),  INT32_C(  1323431090), -INT32_C(    94788569),  INT32_C(   776649367), -INT32_C(  1763496561), -INT32_C(  1959424151),  INT32_C(    33999712), -INT32_C(  1806721944) },
      UINT16_C( 5264),
      { -INT32_C(  1190640936), -INT32_C(   379768944),  INT32_C(  1940190249),  INT32_C(   111227682), -INT32_C(    23786807), -INT32_C(  1664420459),  INT32_C(  1194865400), -INT32_C(  1638151994),
        -INT32_C(  1688771830), -INT32_C(  1132153709), -INT32_C(   886036311),  INT32_C(   617730395),  INT32_C(  1881368539),  INT32_C(  1410133340), -INT32_C(   442743777),  INT32_C(   394589964) },
      { -INT32_C(   290268325),  INT32_C(   967456400), -INT32_C(  1174087073), -INT32_C(  2032150869), -INT32_C(  1711865283),  INT32_C(   250479599),  INT32_C(  1425246792), -INT32_C(   596936831),
        -INT32_C(   473293485), -INT32_C(  1289980588), -INT32_C(    76734385),  INT32_C(   864111862),  INT32_C(  1036875854), -INT32_C(  1001669764), -INT32_C(   971489467),  INT32_C(   161711286) },
      {  INT32_C(  2002686122), -INT32_C(  1657815529),  INT32_C(   197342107),  INT32_C(  1348596690),  INT32_C(  1734478068), -INT32_C(  1616102649),  INT32_C(  2073481250),  INT32_C(  1110490951),
         INT32_C(  1136981036),  INT32_C(  1323431090),  INT32_C(   811403494),  INT32_C(   776649367),  INT32_C(  1307451285), -INT32_C(  1959424151),  INT32_C(    33999712), -INT32_C(  1806721944) } },
    { { -INT32_C(   169054815),  INT32_C(   849873122),  INT32_C(   523048489), -INT32_C(  1319915934), -INT32_C(  1544675289),  INT32_C(   543636187),  INT32_C(   803635065), -INT32_C(  1522956029),
        -INT32_C(   644209161),  INT32_C(  1460355886), -INT32_C(  1149880231),  INT32_C(   242010599), -INT32_C(   995009559),  INT32_C(   249829525), -INT32_C(  1690449001),  INT32_C(  1262515796) },
      UINT16_C(55964),
      {  INT32_C(   807258661), -INT32_C(  1754696159), -INT32_C(  1637789646),  INT32_C(   267995742), -INT32_C(   215511537), -INT32_C(   625033572), -INT32_C(  1722805415), -INT32_C(  2089489314),
        -INT32_C(   676032074),  INT32_C(   980360456), -INT32_C(   874983315),  INT32_C(   702206490), -INT32_C(    48430752),  INT32_C(   433576895),  INT32_C(  1303521262), -INT32_C(   892328428) },
      { -INT32_C(  1063091016),  INT32_C(   251269537), -INT32_C(    69610783),  INT32_C(    86356900),  INT32_C(  1946305204),  INT32_C(   210622749),  INT32_C(   341393152),  INT32_C(   517941606),
         INT32_C(  1339982254),  INT32_C(  1935595666),  INT32_C(  1332623274), -INT32_C(  1621847061), -INT32_C(   216836395),  INT32_C(   805281839),  INT32_C(  1178884320),  INT32_C(   795091841) },
      { -INT32_C(   169054815),  INT32_C(   849873122),  INT32_C(  1706590931),  INT32_C(   182254074), -INT32_C(  2027565893),  INT32_C(   543636187),  INT32_C(   803635065), -INT32_C(  1649687240),
        -INT32_C(   644209161),  INT32_C(  1227998618), -INT32_C(  1149880231), -INT32_C(  1232190991),  INT32_C(   235820981),  INT32_C(   249829525),  INT32_C(   200703758), -INT32_C(   441186923) } },
    { {  INT32_C(   914309796), -INT32_C(   995500774), -INT32_C(    15526124), -INT32_C(  2137036885), -INT32_C(   327961923),  INT32_C(   857502291),  INT32_C(  1283023050),  INT32_C(   662429059),
         INT32_C(   962460191), -INT32_C(   318961961), -INT32_C(   924118755),  INT32_C(   910789240), -INT32_C(  1876771779), -INT32_C(   104645073),  INT32_C(   558185630),  INT32_C(   944292121) },
      UINT16_C(42427),
      {  INT32_C(  1856738161),  INT32_C(  1786759551), -INT32_C(   621414255),  INT32_C(  1352086062), -INT32_C(  2054175038),  INT32_C(    96546239), -INT32_C(  1748575665), -INT32_C(  2093121262),
        -INT32_C(  1812862956),  INT32_C(  1140748721), -INT32_C(  1759644823), -INT32_C(   404245467),  INT32_C(   980186746), -INT32_C(   213963356),  INT32_C(   428475655), -INT32_C(  1684224122) },
      {  INT32_C(  1630440880),  INT32_C(  1738812670),  INT32_C(  1157546527), -INT32_C(   248781194),  INT32_C(     2856796), -INT32_C(   873239868), -INT32_C(   169574801), -INT32_C(   175079355),
         INT32_C(   207077134),  INT32_C(   192216043),  INT32_C(   877621949), -INT32_C(  1255834792), -INT32_C(   675983342),  INT32_C(   698526138),  INT32_C(  1813940007),  INT32_C(   358788871) },
      {  INT32_C(   260382401),  INT32_C(   232514945), -INT32_C(    15526124), -INT32_C(  1581460392), -INT32_C(  2052828770), -INT32_C(   835565701),  INT32_C(  1283023050),  INT32_C(  1991049559),
        -INT32_C(  1616488678), -INT32_C(   318961961), -INT32_C(  1554873900),  INT32_C(   910789240), -INT32_C(  1876771779), -INT32_C(   627211234),  INT32_C(   558185630), -INT32_C(  1895929727) } },
    { {  INT32_C(  1495447918),  INT32_C(  1919194804),  INT32_C(  1638315016),  INT32_C(  1092012847), -INT32_C(   719795429), -INT32_C(  1677804684),  INT32_C(  1225268290),  INT32_C(   962554571),
        -INT32_C(   678199005),  INT32_C(   524941079), -INT32_C(   629084245), -INT32_C(   719546694), -INT32_C(   693488542),  INT32_C(   829597935), -INT32_C(  1870955835),  INT32_C(   130734820) },
      UINT16_C(23899),
      {  INT32_C(   676623070),  INT32_C(   303497361), -INT32_C(   156708134),  INT32_C(  1378552487),  INT32_C(  1408899809), -INT32_C(   942817460),  INT32_C(   446804304), -INT32_C(  1770521416),
        -INT32_C(     4273298), -INT32_C(  1508780341),  INT32_C(  1335671464), -INT32_C(  1516124220),  INT32_C(   788045026), -INT32_C(  1409956517),  INT32_C(   784701046),  INT32_C(    12926098) },
      { -INT32_C(   738229496),  INT32_C(    24711513), -INT32_C(  1890511157), -INT32_C(  1086983459), -INT32_C(   353555058),  INT32_C(  1771430643),  INT32_C(   177757048), -INT32_C(  1626710633),
         INT32_C(   980617952), -INT32_C(   432280549), -INT32_C(   529167358),  INT32_C(   228567679), -INT32_C(   872968744), -INT32_C(   399209105),  INT32_C(  2146618600),  INT32_C(   169868585) },
      { -INT32_C(    72617514),  INT32_C(   325980616),  INT32_C(  1638315016), -INT32_C(   316737414), -INT32_C(  1189660817), -INT32_C(  1677804684),  INT32_C(   272230952),  INT32_C(   962554571),
        -INT32_C(   976371314),  INT32_C(   524941079), -INT32_C(  1343670614), -INT32_C(  1472303685), -INT32_C(   451997382),  INT32_C(   829597935),  INT32_C(  1362582174),  INT32_C(   130734820) } },
    { {  INT32_C(   574919175), -INT32_C(  2130149506), -INT32_C(  1956545268),  INT32_C(     9961512), -INT32_C(    36990835),  INT32_C(    98959133), -INT32_C(   175843380), -INT32_C(   604003372),
        -INT32_C(  1258470603), -INT32_C(   818608701), -INT32_C(  1420126589),  INT32_C(   615248534), -INT32_C(  1625196926),  INT32_C(  1101268597), -INT32_C(  1305007651),  INT32_C(    42809036) },
      UINT16_C(35706),
      { -INT32_C(   342868554),  INT32_C(  1719735052),  INT32_C(  1767380926),  INT32_C(  1558174523),  INT32_C(   526538106), -INT32_C(   850902890),  INT32_C(  2130973938), -INT32_C(   871727594),
        -INT32_C(   944268357),  INT32_C(  1831680174), -INT32_C(  1948875440), -INT32_C(   605505952), -INT32_C(  1577432565),  INT32_C(  2087666314),  INT32_C(  1845261142), -INT32_C(  1405548815) },
      {  INT32_C(  1349775777),  INT32_C(  2042470441), -INT32_C(  2046455003),  INT32_C(  1415703881), -INT32_C(  1024042185), -INT32_C(   214014563), -INT32_C(   916440360), -INT32_C(   495609279),
        -INT32_C(  1271732085), -INT32_C(  1389432952), -INT32_C(   869059709),  INT32_C(  1461818400), -INT32_C(  1944512529),  INT32_C(  1417697404), -INT32_C(   753016686),  INT32_C(    95851130) },
      {  INT32_C(   574919175),  INT32_C(   524071717), -INT32_C(  1956545268),  INT32_C(   146683506), -INT32_C(   577499571),  INT32_C(  1047930123), -INT32_C(  1234948566), -INT32_C(   604003372),
         INT32_C(  1938125616), -INT32_C(  1073490138), -INT32_C(  1420126589), -INT32_C(  1932975552), -INT32_C(  1625196926),  INT32_C(  1101268597), -INT32_C(  1305007651), -INT32_C(  1450208117) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].src));
    easysimd__m512 a = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].a));
    easysimd__m512 b = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].b));
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_xor_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_xor_ps");
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
    easysimd__m512i r = easysimd_mm512_castps_si512(easysimd_mm512_mask_xor_ps(easysimd_mm512_castsi512_ps(src), k, easysimd_mm512_castsi512_ps(a), easysimd_mm512_castsi512_ps(b)));

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
test_easysimd_mm512_maskz_xor_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(59514),
      { -INT32_C(   388496710), -INT32_C(   484746321),  INT32_C(  1215773479), -INT32_C(  1403033710),  INT32_C(  1929698291),  INT32_C(  1280611887),  INT32_C(   551472746), -INT32_C(  1912055596),
         INT32_C(   175562842),  INT32_C(  1676513595),  INT32_C(  1588290764), -INT32_C(  1089860661),  INT32_C(   372379366),  INT32_C(   258115237),  INT32_C(   690962773), -INT32_C(   189319271) },
      {  INT32_C(  1409166616), -INT32_C(  1967723586),  INT32_C(   468214351),  INT32_C(  1406858093), -INT32_C(  1503065087), -INT32_C(   390738797), -INT32_C(  1508776947),  INT32_C(   882559004),
        -INT32_C(  1282959115), -INT32_C(   750961021),  INT32_C(   233711008),  INT32_C(   442550296),  INT32_C(  1740688084), -INT32_C(  1555073386),  INT32_C(  1984520282),  INT32_C(   497738792) },
      {  INT32_C(           0),  INT32_C(  1772990481),  INT32_C(           0), -INT32_C(     8022785), -INT32_C(   714221582), -INT32_C(  1528735044), -INT32_C(  2033243545),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1519729709),  INT32_C(           0), -INT32_C(  1406275533),  INT32_C(  1600528655), -INT32_C(   383921231) } },
    { UINT16_C(12924),
      {  INT32_C(   225444048), -INT32_C(  1053617965),  INT32_C(  2106149661),  INT32_C(   642211173),  INT32_C(   345824709), -INT32_C(   898238848),  INT32_C(   397319533), -INT32_C(  1958139205),
        -INT32_C(    40257238), -INT32_C(   423703351),  INT32_C(  2103723799),  INT32_C(  1789111205),  INT32_C(   159268745), -INT32_C(  1563167947),  INT32_C(  1287291281), -INT32_C(   690551892) },
      { -INT32_C(  2032897859),  INT32_C(  1416467004),  INT32_C(  2144457178),  INT32_C(    99185788), -INT32_C(   401643597), -INT32_C(   326376869),  INT32_C(   272123236),  INT32_C(    98963272),
        -INT32_C(  1131627904),  INT32_C(   655423821),  INT32_C(  1185341898),  INT32_C(   139235157),  INT32_C(  1374772214), -INT32_C(  1589806019),  INT32_C(   162624961),  INT32_C(    84908165) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(    39361223),  INT32_C(   598616345), -INT32_C(    57427338),  INT32_C(   654185691),  INT32_C(   127326217),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1045547644),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1485792383),  INT32_C(    65965832),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(39762),
      { -INT32_C(   762011711),  INT32_C(  1823694534), -INT32_C(   251983452),  INT32_C(    38531601), -INT32_C(  2122413757), -INT32_C(   587841750),  INT32_C(  1484028745),  INT32_C(  1123272320),
         INT32_C(   739542886), -INT32_C(  1986410523), -INT32_C(   495282992), -INT32_C(   890976890),  INT32_C(  2034983758), -INT32_C(   346734174),  INT32_C(  1027852733), -INT32_C(   176212337) },
      { -INT32_C(  1574792259),  INT32_C(   724351835), -INT32_C(   703748529), -INT32_C(  1163857301), -INT32_C(   147592363), -INT32_C(   370964436), -INT32_C(   534305199),  INT32_C(   450274909),
        -INT32_C(  1782712262),  INT32_C(    46197171), -INT32_C(    86454641),  INT32_C(   364148928), -INT32_C(  1878202525), -INT32_C(  1065750673),  INT32_C(  1939972118), -INT32_C(  2121435321) },
      {  INT32_C(           0),  INT32_C(  1201661341),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1984782870),  INT32_C(           0), -INT32_C(  1202496232),  INT32_C(           0),
        -INT32_C(  1180074148), -INT32_C(  1957089706),  INT32_C(           0), -INT32_C(   548356794), -INT32_C(   381189075),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1962033608) } },
    { UINT16_C(19055),
      { -INT32_C(   701292010), -INT32_C(    39534044),  INT32_C(  1903518909), -INT32_C(  2040931975), -INT32_C(   495597463),  INT32_C(   713264009), -INT32_C(  1935553794),  INT32_C(  1641418827),
         INT32_C(  1446447666), -INT32_C(  1990992692), -INT32_C(  1157904064),  INT32_C(   188765346), -INT32_C(  1511148260),  INT32_C(  1070559297), -INT32_C(  2050265030), -INT32_C(  1293507968) },
      {  INT32_C(  2013863596),  INT32_C(   973233402), -INT32_C(   940245723),  INT32_C(  1842492497),  INT32_C(   722714858),  INT32_C(  1802166832), -INT32_C(   739232173), -INT32_C(  2071537704),
        -INT32_C(   268660747),  INT32_C(   287964907),  INT32_C(  1289232123),  INT32_C(  1018866514), -INT32_C(  1687696021),  INT32_C(    34001583), -INT32_C(   522782968), -INT32_C(  1016832818) },
      { -INT32_C(  1371915078), -INT32_C(   945381666), -INT32_C(  1233020520), -INT32_C(   343151320),  INT32_C(           0),  INT32_C(  1105815993),  INT32_C(  1599208621),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1736826329),  INT32_C(           0),  INT32_C(   939196400),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1696433970),  INT32_C(           0) } },
    { UINT16_C(24811),
      { -INT32_C(   614476110), -INT32_C(  1057400089),  INT32_C(  1634421927), -INT32_C(   265365880),  INT32_C(  2026036593), -INT32_C(  1234187552),  INT32_C(   236076458), -INT32_C(  1301349120),
        -INT32_C(  1148334637), -INT32_C(   813988056),  INT32_C(  1529931475),  INT32_C(   759914172),  INT32_C(   463801659), -INT32_C(  2117004073),  INT32_C(  1368384337), -INT32_C(  1274806560) },
      { -INT32_C(   210791989), -INT32_C(   339547624), -INT32_C(  1941507376), -INT32_C(  1933995440),  INT32_C(  2007457695), -INT32_C(   973571725),  INT32_C(  1024886876),  INT32_C(  1374755462),
        -INT32_C(  1018863445),  INT32_C(   447612746),  INT32_C(  1252455674),  INT32_C(   651583366),  INT32_C(   849182398),  INT32_C(  1408734711), -INT32_C(  1550840547), -INT32_C(   755662553) },
      {  INT32_C(   674252665),  INT32_C(   725332223),  INT32_C(           0),  INT32_C(  2090288344),  INT32_C(           0),  INT32_C(  1939321747),  INT32_C(   855946742), -INT32_C(   476059770),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   769228512), -INT32_C(   232788404),  INT32_C(           0) } },
    { UINT16_C(15073),
      {  INT32_C(  1128344469), -INT32_C(   348636347),  INT32_C(  1548467846), -INT32_C(  2116417052),  INT32_C(   840421691), -INT32_C(  1254083547), -INT32_C(   852006952), -INT32_C(   821618630),
        -INT32_C(  1995225020),  INT32_C(   175459204), -INT32_C(   295256055),  INT32_C(    91177417),  INT32_C(   926385682), -INT32_C(  1813153861),  INT32_C(   425731295), -INT32_C(  2132252868) },
      {  INT32_C(   856357807),  INT32_C(  1346207558),  INT32_C(   138323007), -INT32_C(   150098459), -INT32_C(   282114764), -INT32_C(  1685971780),  INT32_C(  2092229184), -INT32_C(   117596855),
        -INT32_C(   550762600), -INT32_C(   986748538), -INT32_C(   221418227),  INT32_C(  1324014362), -INT32_C(   599975648),  INT32_C(  1970847541), -INT32_C(   353293151),  INT32_C(  1659104969) },
      {  INT32_C(  1884016698),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   784476313), -INT32_C(  1316780648),  INT32_C(   939165043),
         INT32_C(           0), -INT32_C(   816176638),  INT32_C(           0),  INT32_C(  1267047123), -INT32_C(   351625422), -INT32_C(   426391410),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C( 4085),
      {  INT32_C(  1886944065),  INT32_C(   249398848),  INT32_C(  1659500408), -INT32_C(  2089088698),  INT32_C(  1564717285),  INT32_C(   394978341), -INT32_C(  1308273713), -INT32_C(   155059275),
        -INT32_C(  1218037386),  INT32_C(   952452031), -INT32_C(  2120569285), -INT32_C(  1677453641), -INT32_C(   319141946), -INT32_C(    83655636), -INT32_C(  1918039849),  INT32_C(  2055433731) },
      {  INT32_C(  1731324583),  INT32_C(  1755313709),  INT32_C(  1542011300),  INT32_C(   385347151),  INT32_C(  1627582773),  INT32_C(  1297876341),  INT32_C(   299501837),  INT32_C(   529227127),
         INT32_C(  1971764295),  INT32_C(  1457333682), -INT32_C(  1364080802), -INT32_C(   356210507),  INT32_C(   256689818), -INT32_C(   665016117),  INT32_C(   703149746), -INT32_C(   616008556) },
      {  INT32_C(   390697446),  INT32_C(           0),  INT32_C(   956354268),  INT32_C(           0),  INT32_C(  1010909648),  INT32_C(  1524031824), -INT32_C(  1545643326), -INT32_C(   380983614),
        -INT32_C(  1025473231),  INT32_C(  1847092749),  INT32_C(   791374181),  INT32_C(  1992343042),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(52784),
      {  INT32_C(   787735120), -INT32_C(   369798600), -INT32_C(   997021184), -INT32_C(   527815532), -INT32_C(  1735830213),  INT32_C(   399391533), -INT32_C(  1383374236), -INT32_C(  1904493506),
        -INT32_C(   675516514), -INT32_C(  1044336191), -INT32_C(   259698085), -INT32_C(  1160769666), -INT32_C(  1856874140), -INT32_C(   106422124), -INT32_C(  1046072189), -INT32_C(  1907416592) },
      {  INT32_C(  1365576848),  INT32_C(   420619965), -INT32_C(   150366344),  INT32_C(   179427750), -INT32_C(   946076877), -INT32_C(  1497349085),  INT32_C(  1751606904),  INT32_C(   418887560),
        -INT32_C(  2140513085), -INT32_C(    73827198), -INT32_C(  1158503916), -INT32_C(  1346002052), -INT32_C(   914988634),  INT32_C(   510670757),  INT32_C(   629594013),  INT32_C(  1363049870) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1595233800), -INT32_C(  1324450034),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(   978963779),  INT32_C(  1249374287),  INT32_C(   353741826),  INT32_C(           0),  INT32_C(           0), -INT32_C(   467606754), -INT32_C(   546218882) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].a));
    easysimd__m512 b = easysimd_mm512_castsi512_ps(easysimd_mm512_loadu_epi32(test_vec[i].b));
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_xor_ps(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_xor_ps");
    easysimd_test_x86_assert_equal_i32x16(easysimd_mm512_castps_si512(r), easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_castps_si512(easysimd_mm512_maskz_xor_ps(k, easysimd_mm512_castsi512_ps(a), easysimd_mm512_castsi512_ps(b)));

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_xor_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 8917272025905183984), -INT64_C( 8866677502414823733),  INT64_C( 1464795012987523672),  INT64_C( 9076492970404562796),
        -INT64_C(  471420776636169871),  INT64_C( 3961263463326435322), -INT64_C( 4926361178749313089),  INT64_C( 6187424904041771752) },
      { -INT64_C( 6187074978812186357),  INT64_C( 9129308580762170105),  INT64_C( 6737626754639454484),  INT64_C( 8553348640616316182),
        -INT64_C( 7747167415395374764),  INT64_C( 2270104474263871152),  INT64_C( 8997904634902970919), -INT64_C( 1284389140319940514) },
      {  INT64_C( 3322640854544675355), -INT64_C(  413510917362488270),  INT64_C( 5319635821359362892),  INT64_C(  812256522989615738),
         INT64_C( 7856992465972914213),  INT64_C( 2988204169892698442), -INT64_C( 4072109430317720168), -INT64_C( 4903636269142318410) } },
    { { -INT64_C( 6505531527298525732),  INT64_C( 9073156322736741803),  INT64_C( 3499178133975168607),  INT64_C( 8184285759661098913),
         INT64_C( 3319482116807104653),  INT64_C( 6307874133959206792), -INT64_C( 1590393193835026219),  INT64_C( 3334991353612573190) },
      { -INT64_C(  189158219624270114),  INT64_C( 4609744475273820848),  INT64_C( 3508270164975553796), -INT64_C( 7960726155283276816),
         INT64_C( 2082343515212321040),  INT64_C( 4039683120892358364), -INT64_C( 7716078895957568331),  INT64_C( 8874843879011272792) },
      {  INT64_C( 6406454113495114498),  INT64_C( 4761230285051520795),  INT64_C(    9133478509509467), -INT64_C( 2300849903224578991),
         INT64_C( 3671791598235879837),  INT64_C( 8036045162057386324),  INT64_C( 9009225591048387680),  INT64_C( 6152343202689438302) } },
    { {  INT64_C( 1660001668875280251), -INT64_C( 6688666841194484293),  INT64_C(  866481487179763680), -INT64_C( 6014321076695304337),
        -INT64_C( 4943803188177761355),  INT64_C( 8602557963703392155),  INT64_C( 2348766465129802213), -INT64_C( 4226418528419391895) },
      { -INT64_C(  276208556333754517), -INT64_C( 8793904538853466362), -INT64_C( 1578788849987643064), -INT64_C( 1646247322702001477),
         INT64_C( 2188768063495513738),  INT64_C( 7993857117939491420), -INT64_C( 8683858962037236194),  INT64_C( 8744573975962379253) },
      { -INT64_C( 1503296891938387952),  INT64_C( 2799218735084909245), -INT64_C( 1868607137191635288),  INT64_C( 5021386112833600980),
        -INT64_C( 6556108815733478081),  INT64_C( 1841284304668030407), -INT64_C( 6348735240143046661), -INT64_C( 4899273116385921636) } },
    { { -INT64_C( 7645977387302788113), -INT64_C( 3352234231512672247),  INT64_C( 7099980801611090558),  INT64_C( 2943395267179314445),
        -INT64_C( 4607844101183449547), -INT64_C( 8977088685291950508),  INT64_C(  143793199971752471),  INT64_C( 1279544180712774249) },
      {  INT64_C( 5042683055231339073),  INT64_C(  106425879644314133), -INT64_C( 7957364987157442167),  INT64_C( 2375707624005304798),
         INT64_C( 1229740195508336126), -INT64_C( 4558434066425528247), -INT64_C( 6013628711494954681),  INT64_C(  863595646972066585) },
      { -INT64_C( 3449971071047439954), -INT64_C( 3458657327579413988), -INT64_C(  929447554171085833),  INT64_C(  585818806074473683),
        -INT64_C( 3378460560296254517),  INT64_C( 4888597167206592029), -INT64_C( 5947680971553895600),  INT64_C( 1890879682201149808) } },
    { {  INT64_C(   19342552227360650),  INT64_C( 9015339881571254999),  INT64_C( 2111175543395945328),  INT64_C( 1063893217915658645),
         INT64_C( 5113671542448273537),  INT64_C(  294076048327577819),  INT64_C( 4136299984689877214),  INT64_C( 6998346057415234657) },
      { -INT64_C( 5822897953418392471), -INT64_C( 2652605089542910307), -INT64_C( 6710896084442025600),  INT64_C(  482490459503952710),
        -INT64_C( 4249188140389718191),  INT64_C( 3686292135995040188), -INT64_C( 7820050545947625371), -INT64_C( 4155849880699296371) },
      { -INT64_C( 5803909723175885853), -INT64_C( 6472545252534039478), -INT64_C( 4642510373644240144),  INT64_C(  608429854075970259),
        -INT64_C( 8939450874709434416),  INT64_C( 3980218559741096295), -INT64_C( 6188350928172226373), -INT64_C( 6391617066844278292) } },
    { {  INT64_C( 6004850654787511453),  INT64_C( 5940721718117239162),  INT64_C( 8299177194198841098),  INT64_C( 8739678760146743174),
        -INT64_C(   33468914264828954),  INT64_C( 4810661481483717294), -INT64_C( 6371855048832433144),  INT64_C( 2475934475524100073) },
      {  INT64_C( 6400425042904156857),  INT64_C( 7634775463157369383), -INT64_C( 8012522724089233327),  INT64_C( 3036153849740553193),
        -INT64_C( 1737102447742047799), -INT64_C( 2861167923869488012),  INT64_C(  265573619744849241),  INT64_C( 8622106189528170193) },
      {  INT64_C(  830741991547514404),  INT64_C( 4288980697190663005), -INT64_C( 2026245280167086245),  INT64_C( 6010905068804162671),
         INT64_C( 1760215543021156911), -INT64_C( 7311046646325205286), -INT64_C( 6612095300032300719),  INT64_C( 6195711379304100152) } },
    { { -INT64_C(   64316270072716246),  INT64_C( 1935443446427172380), -INT64_C( 1091493333936354380), -INT64_C( 6895415197380722231),
        -INT64_C( 1805936298755591266),  INT64_C( 8997027290326566875), -INT64_C( 7447454384865349272),  INT64_C( 6845659238006585079) },
      { -INT64_C( 3918924663269530680), -INT64_C( 5736519009328467646),  INT64_C( 3896138832546275085),  INT64_C(  114649980956301072),
        -INT64_C( 3878013855955832913), -INT64_C( 8817041431659451413), -INT64_C( 3144109803129131596), -INT64_C( 3898751961541556299) },
      {  INT64_C( 3929019560870856674), -INT64_C( 6142951854191587490), -INT64_C( 4121953400583433031), -INT64_C( 6784178172063836455),
         INT64_C( 3233162805184420913), -INT64_C(  470539099348898768),  INT64_C( 5546380282716151004), -INT64_C( 7573801740836425406) } },
    { {  INT64_C( 4005776973137074199),  INT64_C( 7761109657262433250), -INT64_C( 4087271545684412689),  INT64_C( 8176212645445862943),
         INT64_C( 1663312718869912950), -INT64_C( 5987727983636667819), -INT64_C( 2064163572997672891),  INT64_C( 4576071772732176903) },
      {  INT64_C( 3480407709618747496),  INT64_C( 2300011067054476999),  INT64_C( 2573836082898378693), -INT64_C( 8237295633617103203),
         INT64_C( 8697230954808831974), -INT64_C( 7715133602037231102),  INT64_C( 1008851380274702642), -INT64_C( 7795902896402923754) },
      {  INT64_C(  566192077266575487),  INT64_C( 8385218711665306917), -INT64_C( 1945833864310642390), -INT64_C(  227289569249472382),
         INT64_C( 8045543724174291600),  INT64_C( 4037767070791741527), -INT64_C( 1343560506274506377), -INT64_C( 6030853617239309039) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_castpd_si512(easysimd_mm512_xor_pd(easysimd_mm512_castsi512_pd(a), easysimd_mm512_castsi512_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_xor_pd");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_xor_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 6437586905583906514), -INT64_C( 5153936525484381450),  INT64_C( 6943270933310206069),  INT64_C( 7524211599809177408),
         INT64_C( 5824434538339005260),  INT64_C( 7780027683751815990), -INT64_C( 1394359224491689947), -INT64_C( 3449163012513913563) },
      UINT8_C(143),
      { -INT64_C( 8495133462940267688), -INT64_C( 8336185638516663307),  INT64_C( 7003137260866016717),  INT64_C( 7314044866347264725),
        -INT64_C( 7518365256422627072), -INT64_C( 2116391571925423679), -INT64_C( 3295476051057190490), -INT64_C( 2415543629020714745) },
      { -INT64_C( 1376229935454239572),  INT64_C( 6901383497781455056), -INT64_C( 2860763433324596356),  INT64_C(   30230331557417135),
        -INT64_C( 2431603988126275541),  INT64_C( 7131038683452621789),  INT64_C( 7918781416053946429),  INT64_C( 7173219600052179513) },
      {  INT64_C( 7421262813812002804), -INT64_C( 3203949473688325339), -INT64_C( 5081007143180949839),  INT64_C( 7344196020119140986),
         INT64_C( 5824434538339005260),  INT64_C( 7780027683751815990), -INT64_C( 1394359224491689947), -INT64_C( 4758573184432080066) } },
    { { -INT64_C( 1244407392406742067),  INT64_C( 4865711713716370749), -INT64_C( 4260908224916397389),  INT64_C(  766981091718453256),
        -INT64_C( 7658195962633560647),  INT64_C( 7738943069788650780), -INT64_C(  652534938857588831), -INT64_C( 2748924064059407726) },
      UINT8_C( 57),
      { -INT64_C( 7387267490389056577), -INT64_C( 4845665373647692064),  INT64_C( 2442816728953712025),  INT64_C( 5687231108945655512),
        -INT64_C( 8740879232912228419),  INT64_C( 6523012989000680390), -INT64_C( 6245484632013393837),  INT64_C( 7865618868345869642) },
      { -INT64_C( 4326489205870504503),  INT64_C( 6005009147614894651), -INT64_C( 2035670358823730107),  INT64_C( 3497860287354969197),
        -INT64_C( 8972132218061254469),  INT64_C( 7170527882349840943),  INT64_C( 5572422650585221081), -INT64_C( 5554076370592984492) },
      {  INT64_C( 6525187708755080822),  INT64_C( 4865711713716370749), -INT64_C( 4260908224916397389),  INT64_C( 9108530150424488629),
         INT64_C(  418480647996337926),  INT64_C( 4108603231454235113), -INT64_C(  652534938857588831), -INT64_C( 2748924064059407726) } },
    { {  INT64_C( 5751350845764241924), -INT64_C(  111744307068776463),  INT64_C( 8734385293935832072), -INT64_C( 7385587185623981865),
        -INT64_C( 4316499222092911446), -INT64_C( 6905909834716768507),  INT64_C( 3950430239385557299), -INT64_C( 3418646465755150753) },
      UINT8_C(115),
      {  INT64_C( 5273446548515559796),  INT64_C( 2043087781258943199), -INT64_C( 8419001745006677494),  INT64_C( 2284030533521448375),
        -INT64_C( 2011568597036828390), -INT64_C( 9047058233431492909),  INT64_C( 6279836063075462679),  INT64_C( 8312880031040198546) },
      { -INT64_C(  925114651358812351), -INT64_C( 5816755603916639157), -INT64_C( 7376490438547762519), -INT64_C( 4605997696600819895),
        -INT64_C(  797366201897102116), -INT64_C( 7085289264883681181), -INT64_C( 5750947845152246128),  INT64_C( 1872765433263818942) },
      { -INT64_C( 5042238749766304203), -INT64_C( 5540489550081003884),  INT64_C( 8734385293935832072), -INT64_C( 7385587185623981865),
         INT64_C( 1223385550624173510),  INT64_C( 2295035409827286704), -INT64_C( 1794980464735676281), -INT64_C( 3418646465755150753) } },
    { { -INT64_C( 7685388149650553926), -INT64_C( 7004158388354412484), -INT64_C(  483744182612675791), -INT64_C( 4844971588070237331),
         INT64_C( 1801009789046675511), -INT64_C(  345628411063398776),  INT64_C( 4394350837817960903), -INT64_C( 7189809452249894535) },
      UINT8_C(130),
      { -INT64_C( 7901434687957039784), -INT64_C( 8387014073944052033), -INT64_C( 3872086771146056212),  INT64_C( 5740063086953348988),
         INT64_C( 4496694143872862161),  INT64_C(  437603542766895973),  INT64_C( 9056197899055033163),  INT64_C( 8870204133704195539) },
      {  INT64_C( 1445857446161561683),  INT64_C( 2061432370105099130),  INT64_C( 5951135818520226584), -INT64_C( 2832705319378077300),
         INT64_C( 3544538202363265133), -INT64_C( 7941146346981504035),  INT64_C(  476890604075928524),  INT64_C( 6834191479376427898) },
      { -INT64_C( 7685388149650553926), -INT64_C( 7565771451311600187), -INT64_C(  483744182612675791), -INT64_C( 4844971588070237331),
         INT64_C( 1801009789046675511), -INT64_C(  345628411063398776),  INT64_C( 4394350837817960903),  INT64_C( 2724298826198836905) } },
    { {  INT64_C( 1861582297933447660), -INT64_C( 3926976160503129226), -INT64_C( 1810812724388222370), -INT64_C(  955399710856373650),
         INT64_C( 6454903885903836990),  INT64_C( 5606358040040725511),  INT64_C(  307375714144281330),  INT64_C( 2672641076692723998) },
      UINT8_C( 87),
      { -INT64_C( 8133057604378517587), -INT64_C( 3128369023641836069),  INT64_C( 6995617683314701298), -INT64_C(  123729894868305456),
        -INT64_C( 1094594520523767284),  INT64_C(  802529247776427357), -INT64_C( 3768360624300323556),  INT64_C( 8403439177621568592) },
      {  INT64_C( 5515365758784550533),  INT64_C( 7130815896315764176),  INT64_C( 6850654592749139788),  INT64_C( 6215987634154556774),
         INT64_C( 5373211595502647125),  INT64_C( 4125625930493472153), -INT64_C( 4138928590701922783), -INT64_C( 8968962439989850273) },
      { -INT64_C( 4347362476010100440), -INT64_C( 5305113142871574517),  INT64_C( 4469549099567023294), -INT64_C(  955399710856373650),
        -INT64_C( 5017498788745022119),  INT64_C( 5606358040040725511),  INT64_C(  953502644259945277),  INT64_C( 2672641076692723998) } },
    { { -INT64_C( 8317205185373374257),  INT64_C( 1417416665039882374),  INT64_C( 2146759881640848291),  INT64_C(  437926923877572449),
        -INT64_C( 4986361973360829620), -INT64_C( 8669190806303917910),  INT64_C( 2872827924448015107),  INT64_C( 5544434721573252020) },
      UINT8_C(165),
      { -INT64_C( 7940528556938261440),  INT64_C(  980655630240278975),  INT64_C( 7318906000190430344), -INT64_C( 5971576922601849679),
         INT64_C( 5487808273607293246),  INT64_C( 2534539895022788102), -INT64_C( 6090395042801713190), -INT64_C( 3447708299583561971) },
      { -INT64_C( 7414736347008076223),  INT64_C( 6263633501593599278), -INT64_C( 8463701320435831580),  INT64_C( 4117745779878441412),
         INT64_C( 5426028437997849889), -INT64_C( 3463548576539081187),  INT64_C(  212575448128707052),  INT64_C( 4835596213675358185) },
      {  INT64_C(  636137617506597377),  INT64_C( 1417416665039882374), -INT64_C( 1217345983919574932),  INT64_C(  437926923877572449),
        -INT64_C( 4986361973360829620), -INT64_C( 1386125238534170597),  INT64_C( 2872827924448015107), -INT64_C( 7837324708450129692) } },
    { {  INT64_C( 6949693325426827786),  INT64_C( 8216510233329632843), -INT64_C( 3061938756482517951),  INT64_C( 6929581147847842245),
        -INT64_C( 5505939268427111693), -INT64_C( 5154386854511210625), -INT64_C( 1916922516866575708), -INT64_C( 2285305309217154158) },
      UINT8_C( 25),
      {  INT64_C( 4271440688287127247),  INT64_C( 1563430506160344094), -INT64_C( 4328211410770902365), -INT64_C( 1155577783541245978),
        -INT64_C( 3850578045069396057),  INT64_C( 5703118074251067724),  INT64_C( 6573174584934947533),  INT64_C( 2270428257532307570) },
      { -INT64_C( 8855481797091050843),  INT64_C( 5494315124780299075), -INT64_C(  439671032366634220), -INT64_C( 1046790816815964703),
         INT64_C( 3532614236670979143),  INT64_C( 7645811813566749521), -INT64_C( 5274602650662085556), -INT64_C( 6038331607380633610) },
      { -INT64_C( 4729845492485051286),  INT64_C( 8216510233329632843), -INT64_C( 3061938756482517951),  INT64_C( 2202151832208001543),
        -INT64_C(  321547477604197408), -INT64_C( 5154386854511210625), -INT64_C( 1916922516866575708), -INT64_C( 2285305309217154158) } },
    { {  INT64_C( 2635789764425250421),  INT64_C( 7173969323519948495),  INT64_C( 5870444707384296172), -INT64_C( 8760690870588307366),
        -INT64_C( 7190984385298327236),  INT64_C( 2643138234015623219),  INT64_C( 4719594182013215955),  INT64_C( 1548230951492915941) },
      UINT8_C(  4),
      { -INT64_C( 9072943066938006171), -INT64_C(  913142141260504785),  INT64_C( 1298755229924884931), -INT64_C( 2338494684052390629),
        -INT64_C(  432107195710939868),  INT64_C( 6574355609988807225),  INT64_C( 5575600590773032526),  INT64_C( 8059297408314788736) },
      { -INT64_C( 6015985071739266371),  INT64_C( 1844841742087880103),  INT64_C( 1929731531138864967), -INT64_C( 5130465311139325165),
         INT64_C( 5595374708968176006),  INT64_C( 1415018467375750080), -INT64_C( 1805287231774192834),  INT64_C( 8692810336975390191) },
      {  INT64_C( 2635789764425250421),  INT64_C( 7173969323519948495),  INT64_C(  631018082675824260), -INT64_C( 8760690870588307366),
        -INT64_C( 7190984385298327236),  INT64_C( 2643138234015623219),  INT64_C( 4719594182013215955),  INT64_C( 1548230951492915941) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].src));
    easysimd__m512d a = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].a));
    easysimd__m512d b = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].b));
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_xor_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_xor_pd");
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
    easysimd__m512i r = easysimd_mm512_castpd_si512(easysimd_mm512_mask_xor_pd(easysimd_mm512_castsi512_pd(src), k, easysimd_mm512_castsi512_pd(a), easysimd_mm512_castsi512_pd(b)));

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
test_easysimd_mm512_maskz_xor_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(178),
      {  INT64_C( 4535000673791092910), -INT64_C( 3010281343767629417), -INT64_C( 8904038272174085999),  INT64_C( 2275293563450029107),
         INT64_C( 3707561576362413344),  INT64_C( 1693514566972551576), -INT64_C( 4027204785809648325), -INT64_C( 2377269758930565262) },
      {  INT64_C( 8798212751342500426),  INT64_C( 4635791733872161720),  INT64_C( 6203697368372176313),  INT64_C( 5321913355302715709),
         INT64_C( 3355262493918293813),  INT64_C( 1297812233894278332),  INT64_C( 1035961853024525095),  INT64_C(  685644267943270714) },
      {  INT64_C(                   0), -INT64_C( 7607440601921174993),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C( 2153757367459176981),  INT64_C(  396971936153175332),  INT64_C(                   0), -INT64_C( 2989870719856222648) } },
    { UINT8_C(212),
      {  INT64_C( 7330022570064486579),  INT64_C( 6470776002694773732), -INT64_C( 3322187338369804494),  INT64_C( 6195738823362944834),
        -INT64_C( 6781283329911994929), -INT64_C( 6531323974896934267),  INT64_C(   68176035082962666),  INT64_C( 1606415384555526228) },
      {  INT64_C( 3298177160410668391),  INT64_C(  591251513971575580),  INT64_C( 1067384700368749586), -INT64_C( 9218498889734817868),
         INT64_C( 4450234974760904685), -INT64_C( 8026820267700522380), -INT64_C( 7616096171536104058), -INT64_C( 7944169816828259545) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 2362942239413349600),  INT64_C(                   0),
        -INT64_C( 7194953829069757918),  INT64_C(                   0), -INT64_C( 7585181778073074836), -INT64_C( 8679701041803572365) } },
    { UINT8_C( 74),
      { -INT64_C( 2719705882611210652),  INT64_C( 3890521222854268022), -INT64_C( 2332954163621184184),  INT64_C(  229144407503527744),
        -INT64_C( 5116632051351512773), -INT64_C( 8607455037759662106), -INT64_C( 7937981019297111189), -INT64_C( 6187203240389158273) },
      { -INT64_C( 1497502927574600900),  INT64_C(   66373760537749284),  INT64_C(  816503436927822961), -INT64_C( 6250629679042344665),
         INT64_C(   13616198539808576), -INT64_C( 5594509499254410553),  INT64_C( 5247759133439251259), -INT64_C( 8455473893407261293) },
      {  INT64_C(                   0),  INT64_C( 3825287698824475474),  INT64_C(                   0), -INT64_C( 6165611562172606873),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C( 2808722857730379696),  INT64_C(                   0) } },
    { UINT8_C( 47),
      {  INT64_C( 8284103234516930512),  INT64_C(  214877212740113264), -INT64_C( 7838960910472133926), -INT64_C( 5794047091137497527),
         INT64_C( 2281872209261601591), -INT64_C( 7144158427984536279), -INT64_C( 5709308242018960228),  INT64_C( 4699981245450855084) },
      {  INT64_C( 1666652842254473299),  INT64_C( 8493447882235445120),  INT64_C( 4601952674112740806), -INT64_C(  551686946749221619),
         INT64_C( 8368281792577982748), -INT64_C( 6815752888624213733), -INT64_C( 8681344165041270862),  INT64_C( 7944058725472292768) },
      {  INT64_C( 7338105342941910915),  INT64_C( 8585457891708546288), -INT64_C( 5986690151986438372),  INT64_C( 6327394179636249412),
         INT64_C(                   0),  INT64_C( 4445982027659365426),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(200),
      { -INT64_C( 6021539633559668423), -INT64_C( 2581080677709028376),  INT64_C( 5206969282928735478), -INT64_C(   55158347487606660),
         INT64_C( 1353827748649695618),  INT64_C( 8128924183901983015), -INT64_C( 9191192612159019471), -INT64_C( 7448619434912772645) },
      { -INT64_C( 2276993227150723799), -INT64_C( 1965359776232376881), -INT64_C(  482587615769474359), -INT64_C( 7332918534963773042),
         INT64_C( 4596860684338131250),  INT64_C(  855474210167365449), -INT64_C(  206279998555230709),  INT64_C( 3347263021807418983) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 7277874962021599730),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C( 9030128929851689018), -INT64_C( 5272890609481960516) } },
    { UINT8_C(206),
      { -INT64_C( 8863509690588522143), -INT64_C( 4084252655939553020),  INT64_C( 6644273578117593769), -INT64_C( 2125833707023036716),
         INT64_C( 4397929110397500575),  INT64_C( 5201328400761081486),  INT64_C( 7730512975766119502), -INT64_C( 4270086284785180838) },
      {  INT64_C( 8632926499810858072),  INT64_C( 8360102739033674014), -INT64_C( 4506510691621921934), -INT64_C( 8849475216152424121),
        -INT64_C( 1869305245149529939),  INT64_C( 7840671269508082998),  INT64_C( 2300754294035853252), -INT64_C( 8057294997331146372) },
      {  INT64_C(                   0), -INT64_C( 5524547863824455654), -INT64_C( 7115515909856927269),  INT64_C( 7444411344080314259),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C( 8405308176221621130),  INT64_C( 6094258454041925158) } },
    { UINT8_C( 52),
      {  INT64_C( 2476948137554290500), -INT64_C( 5047539530763289977), -INT64_C( 7681258544912326657), -INT64_C( 8994804189721259973),
        -INT64_C( 7816266480404911824),  INT64_C( 3101210119854334917), -INT64_C( 8829451883390854443),  INT64_C( 2904164220738072460) },
      { -INT64_C( 3858999919556822574), -INT64_C( 4535086428222573128),  INT64_C( 3282026407837967804), -INT64_C( 8923243080841688823),
        -INT64_C( 8931248512236865729),  INT64_C(  483706269034096393),  INT64_C( 8893932455578065054), -INT64_C( 1596624292404238943) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 5122095061723301309),  INT64_C(                   0),
         INT64_C( 1696414226036203023),  INT64_C( 3296580388577313996),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 94),
      {  INT64_C( 6996533585489046842),  INT64_C( 1958416866463124758),  INT64_C( 4537104879152098493),  INT64_C( 6668091767506160517),
         INT64_C( 8690684359556707327), -INT64_C( 5373299068551870832), -INT64_C( 8523060076547933024),  INT64_C( 8344112879438766413) },
      {  INT64_C( 5389482443695536114), -INT64_C( 6396224148588342641), -INT64_C( 7523587219716487416),  INT64_C( 8890901203756993403),
        -INT64_C( 1724708946921451509), -INT64_C( 4532121262000984723), -INT64_C( 8010949269784740878), -INT64_C( 5137233140829398353) },
      {  INT64_C(                   0), -INT64_C( 4894922666030997607), -INT64_C( 6241477190632068171),  INT64_C( 2876394624261161214),
        -INT64_C( 8031297777887304716),  INT64_C(                   0),  INT64_C( 1831669897844082514),  INT64_C(                   0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].a));
    easysimd__m512d b = easysimd_mm512_castsi512_pd(easysimd_mm512_loadu_epi64(test_vec[i].b));
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_xor_pd(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_xor_pd");
    easysimd_test_x86_assert_equal_i64x8(easysimd_mm512_castpd_si512(r), easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_castpd_si512(easysimd_mm512_maskz_xor_pd(k, easysimd_mm512_castsi512_pd(a), easysimd_mm512_castsi512_pd(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   116835021),  INT32_C(   228055744), -INT32_C(   583287396), -INT32_C(  1948084724), -INT32_C(   539666230), -INT32_C(    47767560),  INT32_C(   757520745), -INT32_C(   166532568),
         INT32_C(  1828456620),  INT32_C(  2138739938),  INT32_C(  1566356817),  INT32_C(   384384587), -INT32_C(  1879720297),  INT32_C(  1200365021),  INT32_C(   527741942),  INT32_C(  1226147485) },
      {  INT32_C(  1907757455), -INT32_C(   135253851),  INT32_C(   810831333),  INT32_C(   575028619), -INT32_C(   642630405),  INT32_C(  1327513177), -INT32_C(  1905355535), -INT32_C(  1445428454),
         INT32_C(   974884501), -INT32_C(  1557066818), -INT32_C(   455899560), -INT32_C(  1090118973), -INT32_C(  1348945834), -INT32_C(   385959689),  INT32_C(  1719102796), -INT32_C(  2062528528) },
      {  INT32_C(  2000737090), -INT32_C(    93853083), -INT32_C(   311431047), -INT32_C(  1448825977),  INT32_C(   107439153), -INT32_C(  1308155487), -INT32_C(  1555542632),  INT32_C(  1607100722),
         INT32_C(  1458013753), -INT32_C(   599023780), -INT32_C(  1181797623), -INT32_C(  1443945336),  INT32_C(   544016065), -INT32_C(  1351441110),  INT32_C(  2030296762), -INT32_C(   872035987) } },
    { { -INT32_C(  1681904675), -INT32_C(  1891700426),  INT32_C(   997396855), -INT32_C(  2114291413),  INT32_C(   691049010), -INT32_C(  1777258678), -INT32_C(  1929607012), -INT32_C(  1273885481),
         INT32_C(  1850724919),  INT32_C(  1006472643), -INT32_C(   898207585),  INT32_C(   457994473),  INT32_C(  1279556610),  INT32_C(  1206015659), -INT32_C(  1227563298),  INT32_C(   577431275) },
      {  INT32_C(  2073082552), -INT32_C(   424243897), -INT32_C(   407819011), -INT32_C(  1643971172),  INT32_C(   636110713),  INT32_C(  2070727837), -INT32_C(  1775157077), -INT32_C(   558326746),
        -INT32_C(  1655027370), -INT32_C(   729542442), -INT32_C(   642042563), -INT32_C(  1418150606), -INT32_C(  1580178940), -INT32_C(   635618001), -INT32_C(  1536143747),  INT32_C(  1082271978) },
      { -INT32_C(   531590811),  INT32_C(  1770552433), -INT32_C(   591250038),  INT32_C(   536380599),  INT32_C(   215668043), -INT32_C(   310516777),  INT32_C(   449693751),  INT32_C(  1789563121),
        -INT32_C(   216687775), -INT32_C(   277242603),  INT32_C(   332219810), -INT32_C(  1338716709), -INT32_C(   309060090), -INT32_C(  1644205180),  INT32_C(   312774819),  INT32_C(  1659424257) } },
    { {  INT32_C(  1222564977),  INT32_C(   706503405), -INT32_C(   922495081), -INT32_C(  1720419436),  INT32_C(   221922782), -INT32_C(     1615998), -INT32_C(  1868343386), -INT32_C(   237951361),
        -INT32_C(   281432318), -INT32_C(  1474734831), -INT32_C(  1066328788),  INT32_C(  1985602968), -INT32_C(  1400662998), -INT32_C(  1850971413),  INT32_C(  1092767681),  INT32_C(  2016605045) },
      { -INT32_C(  1285067870), -INT32_C(   329547328),  INT32_C(   917294238), -INT32_C(   592640335), -INT32_C(  2054672230),  INT32_C(  1545024667), -INT32_C(   107136636), -INT32_C(   831402196),
        -INT32_C(    92153542), -INT32_C(   119088038),  INT32_C(  1496224680),  INT32_C(   859167641), -INT32_C(  1481064948),  INT32_C(  1979961330),  INT32_C(   879796488), -INT32_C(  1425874576) },
      { -INT32_C(    71714861), -INT32_C(   968367315), -INT32_C(     5301495),  INT32_C(  1171881253), -INT32_C(  2001570492), -INT32_C(  1544527079),  INT32_C(  1765699106),  INT32_C(  1067510099),
         INT32_C(   364410424),  INT32_C(  1358924107), -INT32_C(  1721790844),  INT32_C(  1164721665),  INT32_C(   188426790), -INT32_C(   407919335),  INT32_C(  1968369353), -INT32_C(   751824379) } },
    { {  INT32_C(   346391482),  INT32_C(   118262879), -INT32_C(  1201587425),  INT32_C(   585864726),  INT32_C(  1204396884),  INT32_C(  2076036467), -INT32_C(   558879378), -INT32_C(   913722865),
        -INT32_C(  1780601034), -INT32_C(   627184965),  INT32_C(  1016266277), -INT32_C(   379683436), -INT32_C(  1825560544),  INT32_C(  1661988341),  INT32_C(   708951835), -INT32_C(  1477194895) },
      { -INT32_C(  1254239750), -INT32_C(   510666052),  INT32_C(  1813848536), -INT32_C(  1101693794), -INT32_C(  1722645084), -INT32_C(  1896062605), -INT32_C(  1850196704),  INT32_C(    37333768),
         INT32_C(   968324733),  INT32_C(   672810576),  INT32_C(    93599847),  INT32_C(  1489300148), -INT32_C(   487516561), -INT32_C(  1754206857),  INT32_C(   858269738),  INT32_C(  1362452948) },
      { -INT32_C(  1583852992), -INT32_C(   427600157), -INT32_C(   730064185), -INT32_C(  1665209720), -INT32_C(   560257296), -INT32_C(   180245504),  INT32_C(  1325928526), -INT32_C(   877651705),
        -INT32_C(  1402382005), -INT32_C(   225989397),  INT32_C(   956745282), -INT32_C(  1315268832),  INT32_C(  1908489807), -INT32_C(   193003390),  INT32_C(   426350385), -INT32_C(   154752347) } },
    { {  INT32_C(   663416279), -INT32_C(  1689279437), -INT32_C(  1868503844),  INT32_C(  1038640334), -INT32_C(   249505414), -INT32_C(   242708282), -INT32_C(  1926975047), -INT32_C(   354526958),
         INT32_C(  2047961158), -INT32_C(   384474868),  INT32_C(   326743365), -INT32_C(  1806671334),  INT32_C(    42364987), -INT32_C(  1158476287), -INT32_C(   767092801), -INT32_C(  1212406416) },
      { -INT32_C(  1724789363),  INT32_C(  1937917486),  INT32_C(   361168123), -INT32_C(  1716856994),  INT32_C(  1218129991), -INT32_C(    33386946),  INT32_C(   382749093), -INT32_C(    70415250),
        -INT32_C(  2003435943),  INT32_C(  1073420100),  INT32_C(  1901429267), -INT32_C(  1609826471),  INT32_C(  1844028975), -INT32_C(   630461643), -INT32_C(  1544537291),  INT32_C(   564116935) },
      { -INT32_C(  1095032742), -INT32_C(   389160419), -INT32_C(  2061101017), -INT32_C(  1539132528), -INT32_C(  1178867395),  INT32_C(   260710136), -INT32_C(  1678444516),  INT32_C(   286512508),
        -INT32_C(   226191841), -INT32_C(   689015224),  INT32_C(  1647064918),  INT32_C(   878419267),  INT32_C(  1869600276),  INT32_C(  1620632884),  INT32_C(  1907829898), -INT32_C(  1776052041) } },
    { { -INT32_C(     5688133),  INT32_C(  1598006347),  INT32_C(  2144375846), -INT32_C(  1038034029), -INT32_C(  1221654142),  INT32_C(   697408500), -INT32_C(  1630698794), -INT32_C(    71340993),
        -INT32_C(   335910752),  INT32_C(   860502284),  INT32_C(  1622285261),  INT32_C(  2015548150), -INT32_C(   802204965), -INT32_C(  1007042067), -INT32_C(  2107521469), -INT32_C(   763551694) },
      { -INT32_C(  1782679416),  INT32_C(  2143815857),  INT32_C(   450853411),  INT32_C(   680657485),  INT32_C(  1106821716), -INT32_C(   972754301), -INT32_C(   347576648),  INT32_C(   264095366),
        -INT32_C(   274433218), -INT32_C(  1485935484),  INT32_C(   868306662), -INT32_C(  1537452976), -INT32_C(  1713023978), -INT32_C(    10491578), -INT32_C(   689264817), -INT32_C(  1411012755) },
      {  INT32_C(  1779911731),  INT32_C(   553102586),  INT32_C(  1695542789), -INT32_C(   357377570), -INT32_C(   153629738), -INT32_C(   275420809),  INT32_C(  1971709038), -INT32_C(   201151815),
         INT32_C(    73274270), -INT32_C(  1809558136),  INT32_C(  1400067371), -INT32_C(   595687770),  INT32_C(  1237976781),  INT32_C(  1017522347),  INT32_C(  1418420492),  INT32_C(  2040039263) } },
    { { -INT32_C(  1482979037), -INT32_C(   615642635), -INT32_C(  1492185001),  INT32_C(  2051763044),  INT32_C(   101920959),  INT32_C(  1761964570), -INT32_C(  2025853159), -INT32_C(  1187896170),
        -INT32_C(  1570714195),  INT32_C(   780054487),  INT32_C(   601263551), -INT32_C(  1214438920),  INT32_C(  1824370770),  INT32_C(  1003864610),  INT32_C(  1220679089), -INT32_C(   419302087) },
      { -INT32_C(  1719049534), -INT32_C(   792197359), -INT32_C(  1930191212),  INT32_C(   289706175),  INT32_C(  1652359488),  INT32_C(  1973244868), -INT32_C(  1581424536),  INT32_C(   394837845),
         INT32_C(   850465313), -INT32_C(  1375569639), -INT32_C(   700779241), -INT32_C(   974619003),  INT32_C(  1143432576),  INT32_C(   549045432),  INT32_C(  2042787620),  INT32_C(  1469139510) },
      {  INT32_C(  1041427425),  INT32_C(   193335012),  INT32_C(   737971907),  INT32_C(  1796209627),  INT32_C(  1684943359),  INT32_C(   479732190),  INT32_C(   646090609), -INT32_C(  1363502141),
        -INT32_C(  1865293940), -INT32_C(  2139302194), -INT32_C(   169051992),  INT32_C(  1920294781),  INT32_C(   681235922),  INT32_C(   460064410),  INT32_C(   822108821), -INT32_C(  1332691185) } },
    { {  INT32_C(  1988706908), -INT32_C(   769356869),  INT32_C(    94920320),  INT32_C(  1573556445), -INT32_C(  1365118474), -INT32_C(   623945035),  INT32_C(   122917329),  INT32_C(   945743067),
        -INT32_C(   508631258), -INT32_C(   223096206),  INT32_C(   234314800), -INT32_C(   496320020),  INT32_C(  1754336178),  INT32_C(   927096934), -INT32_C(   868248079), -INT32_C(  1610310278) },
      { -INT32_C(   142429563), -INT32_C(  1242942076),  INT32_C(  2126700945),  INT32_C(  1432366499), -INT32_C(  1631719112),  INT32_C(  1121386321),  INT32_C(   252646805),  INT32_C(   934220722),
         INT32_C(  1244606918), -INT32_C(   117499545), -INT32_C(  1653161222),  INT32_C(   670291951),  INT32_C(   449229000),  INT32_C(  1163697328),  INT32_C(  1666476977),  INT32_C(  1167786879) },
      { -INT32_C(  2129923879),  INT32_C(  1741602367),  INT32_C(  2070593297),  INT32_C(   145407358),  INT32_C(   807141582), -INT32_C(  1743149596),  INT32_C(   140280900),  INT32_C(   267515753),
        -INT32_C(  1417554208),  INT32_C(   172804885), -INT32_C(  1870553398), -INT32_C(   979823101),  INT32_C(  1918352250),  INT32_C(  1914633430), -INT32_C(  1351877056), -INT32_C(   442524155) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_xor_epi32(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_xor_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   153880592), -INT32_C(  1121219888),  INT32_C(   485857388), -INT32_C(   948181383),  INT32_C(    41449270), -INT32_C(  1756100706),  INT32_C(  1579137812),  INT32_C(   587505939),
         INT32_C(  2049716394),  INT32_C(   641161146),  INT32_C(  1765944559),  INT32_C(  1211153938), -INT32_C(   733304523), -INT32_C(   999579984),  INT32_C(  1814202969), -INT32_C(   695261652) },
      UINT16_C(47959),
      { -INT32_C(  2028859056),  INT32_C(  2058551864), -INT32_C(  1690778261),  INT32_C(  1464102413), -INT32_C(  1376390078), -INT32_C(   617066567),  INT32_C(  1258448059), -INT32_C(  1979295430),
        -INT32_C(  1575937686), -INT32_C(  2044869605), -INT32_C(  1776200311),  INT32_C(    99444419),  INT32_C(   330555994), -INT32_C(   303043791), -INT32_C(  1992756913), -INT32_C(  1273807286) },
      {  INT32_C(  1918313559),  INT32_C(  1912108264), -INT32_C(  1928914231), -INT32_C(   627903360),  INT32_C(   149833174),  INT32_C(  2146819376),  INT32_C(   403189198), -INT32_C(  1010033813),
         INT32_C(   657793855),  INT32_C(  1637363351), -INT32_C(   923885752),  INT32_C(  1789034643), -INT32_C(   160263994),  INT32_C(   997615469), -INT32_C(    11305324), -INT32_C(   641589094) },
      { -INT32_C(   180013817),  INT32_C(   189494992),  INT32_C(   373284770), -INT32_C(   948181383), -INT32_C(  1524911724), -INT32_C(  1756100706),  INT32_C(  1393183093),  INT32_C(   587505939),
        -INT32_C(  2061223339), -INT32_C(   410654068),  INT32_C(  1765944559),  INT32_C(  1867507280), -INT32_C(   440307044), -INT32_C(   694580132),  INT32_C(  1814202969),  INT32_C(  1842421456) } },
    { { -INT32_C(   620627901),  INT32_C(  1849465126), -INT32_C(   868865479),  INT32_C(  1882642602), -INT32_C(   697849751), -INT32_C(  1559044849), -INT32_C(   173841061), -INT32_C(   909154939),
        -INT32_C(  2086350755), -INT32_C(  1561206679), -INT32_C(  1267783926),  INT32_C(  1764074752),  INT32_C(  1564445774), -INT32_C(  1006546583),  INT32_C(  1018864823),  INT32_C(  1728416009) },
      UINT16_C(43353),
      { -INT32_C(   594951446), -INT32_C(   737963163),  INT32_C(  1836647240), -INT32_C(  1392916628),  INT32_C(   654139941), -INT32_C(   523586522), -INT32_C(   160836623),  INT32_C(   631292474),
        -INT32_C(   369022587),  INT32_C(    96339133),  INT32_C(  1953642248),  INT32_C(   606104575), -INT32_C(   213246259), -INT32_C(  1009511214),  INT32_C(   582630887), -INT32_C(  2058921472) },
      {  INT32_C(  1081034883),  INT32_C(  1413819724),  INT32_C(  1674164068), -INT32_C(   276305630), -INT32_C(   622604025), -INT32_C(   845301787), -INT32_C(   185640972),  INT32_C(   897136305),
        -INT32_C(   898242434),  INT32_C(  2048899861), -INT32_C(  1780619150), -INT32_C(   645634607),  INT32_C(   481519414),  INT32_C(   334057502),  INT32_C(  1493686440), -INT32_C(  1936817906) },
      { -INT32_C(  1662612887),  INT32_C(  1849465126), -INT32_C(   868865479),  INT32_C(  1132342862), -INT32_C(    65096926), -INT32_C(  1559044849),  INT32_C(    42370053), -INT32_C(   909154939),
         INT32_C(   544522747), -INT32_C(  1561206679), -INT32_C(  1267783926), -INT32_C(    39580114),  INT32_C(  1564445774), -INT32_C(   801291060),  INT32_C(  1018864823),  INT32_C(   164223758) } },
    { {  INT32_C(  2136343657),  INT32_C(   838432191),  INT32_C(   801560157),  INT32_C(  1913146171), -INT32_C(   779174990), -INT32_C(  1276872949),  INT32_C(  1561193295), -INT32_C(   689333396),
         INT32_C(  1582645151),  INT32_C(   311447221),  INT32_C(  1614894628),  INT32_C(  1423067553),  INT32_C(   270884868),  INT32_C(   650316247),  INT32_C(  1636028660),  INT32_C(   188181612) },
      UINT16_C(36011),
      { -INT32_C(    86351766), -INT32_C(  1269760397),  INT32_C(   822014558),  INT32_C(  1804599878), -INT32_C(   697014510),  INT32_C(   279341197),  INT32_C(     8131273), -INT32_C(  2004080610),
        -INT32_C(    75340152), -INT32_C(  1011821979),  INT32_C(   183807684), -INT32_C(  1015708496),  INT32_C(  2040130028),  INT32_C(   462045010),  INT32_C(  1880884818), -INT32_C(  1208440786) },
      {  INT32_C(  1957853711),  INT32_C(   288842316), -INT32_C(  1055184112), -INT32_C(  1669033552), -INT32_C(   871031430), -INT32_C(  1360486564), -INT32_C(   736230234), -INT32_C(  1148512596),
        -INT32_C(   584106352), -INT32_C(  1309775968),  INT32_C(  1114769810),  INT32_C(   366933658),  INT32_C(  1877079059),  INT32_C(   958318995),  INT32_C(  2047753421), -INT32_C(   483026605) },
      { -INT32_C(  1905780123), -INT32_C(  1519936449),  INT32_C(   801560157), -INT32_C(   149646346), -INT32_C(   779174990), -INT32_C(  1102121007),  INT32_C(  1561193295),  INT32_C(   856112818),
         INT32_C(  1582645151),  INT32_C(   311447221),  INT32_C(  1216784214), -INT32_C(   693407190),  INT32_C(   270884868),  INT32_C(   650316247),  INT32_C(  1636028660),  INT32_C(  1422733693) } },
    { {  INT32_C(  2025874903),  INT32_C(  1579790028),  INT32_C(  1386322872), -INT32_C(  1536721007),  INT32_C(   118704499), -INT32_C(   532664046), -INT32_C(  1051046290), -INT32_C(  1079734296),
        -INT32_C(  1070111244), -INT32_C(   887136237), -INT32_C(  1944141573), -INT32_C(  1288665793), -INT32_C(   507886386), -INT32_C(   473826699),  INT32_C(   832838473), -INT32_C(  1611642454) },
      UINT16_C(10414),
      {  INT32_C(  2122891615), -INT32_C(  1421966195),  INT32_C(  1093697040), -INT32_C(   343539919), -INT32_C(  1578763552), -INT32_C(  2101596450),  INT32_C(  1355507295),  INT32_C(  1702394117),
        -INT32_C(   924647365), -INT32_C(  1787616636), -INT32_C(   774462560), -INT32_C(  2101584990),  INT32_C(   857973077),  INT32_C(   800514000),  INT32_C(  1266647366),  INT32_C(   917567483) },
      {  INT32_C(  2097059064),  INT32_C(  1443983798), -INT32_C(  1222121708), -INT32_C(  1741036734),  INT32_C(  1422613892), -INT32_C(  2105245380), -INT32_C(    20053245), -INT32_C(   214663429),
        -INT32_C(   932236782), -INT32_C(  1222737501), -INT32_C(  1418836632), -INT32_C(  1388074968),  INT32_C(  1090588165), -INT32_C(  1815837296), -INT32_C(  2070834551),  INT32_C(   595051792) },
      {  INT32_C(  2025874903), -INT32_C(    47188677), -INT32_C(   166225660),  INT32_C(  1941707891),  INT32_C(   118704499),  INT32_C(     3715042), -INT32_C(  1051046290), -INT32_C(  1773402114),
        -INT32_C(  1070111244), -INT32_C(   887136237), -INT32_C(  1944141573),  INT32_C(   805303178), -INT32_C(   507886386), -INT32_C(  1133356480),  INT32_C(   832838473), -INT32_C(  1611642454) } },
    { { -INT32_C(  1695815945), -INT32_C(   799995289),  INT32_C(  2021376079),  INT32_C(  1831190120),  INT32_C(  1554917068),  INT32_C(   904884908),  INT32_C(   364478468),  INT32_C(  1010315333),
         INT32_C(  2127962902),  INT32_C(  2102274093),  INT32_C(  1358285288),  INT32_C(  1404902023), -INT32_C(   323982528), -INT32_C(   501112866),  INT32_C(  1710742048),  INT32_C(   564277002) },
      UINT16_C(30803),
      { -INT32_C(   308248417), -INT32_C(   222918403), -INT32_C(  1794359848),  INT32_C(  1073827216),  INT32_C(  1541463865), -INT32_C(  1187708735),  INT32_C(   132661349),  INT32_C(     8403809),
        -INT32_C(  1175641924), -INT32_C(  2119392343),  INT32_C(  1880602848),  INT32_C(  1068570629), -INT32_C(  1197829641), -INT32_C(   160313455),  INT32_C(  1895717391),  INT32_C(  1366326933) },
      {  INT32_C(  1191927198), -INT32_C(   523651328),  INT32_C(  1968300143), -INT32_C(   273415432),  INT32_C(   614944403),  INT32_C(   756685085),  INT32_C(   144513139),  INT32_C(   895094422),
         INT32_C(  1820091755), -INT32_C(  1957935844),  INT32_C(   486579493),  INT32_C(   839693471),  INT32_C(   542553090),  INT32_C(  1078817229),  INT32_C(   541649545),  INT32_C(  1683333880) },
      { -INT32_C(  1431577343),  INT32_C(   310329341),  INT32_C(  2021376079),  INT32_C(  1831190120),  INT32_C(  2135396778),  INT32_C(   904884908),  INT32_C(   259348502),  INT32_C(  1010315333),
         INT32_C(  2127962902),  INT32_C(  2102274093),  INT32_C(  1358285288),  INT32_C(   230534298), -INT32_C(  1731451403), -INT32_C(  1237533092),  INT32_C(  1354150022),  INT32_C(   564277002) } },
    { {  INT32_C(   600887559),  INT32_C(  1018108951),  INT32_C(  1499115450),  INT32_C(  1720477284), -INT32_C(   410590694), -INT32_C(   584592557), -INT32_C(  1224904514),  INT32_C(   437932562),
         INT32_C(   977136163), -INT32_C(  1049105401),  INT32_C(     1757596),  INT32_C(  1382458935), -INT32_C(   600183415),  INT32_C(  2126078400), -INT32_C(   483019055),  INT32_C(   754798344) },
      UINT16_C(15162),
      { -INT32_C(   584629914),  INT32_C(   481215234),  INT32_C(   717481667),  INT32_C(  1897352248), -INT32_C(   489498840), -INT32_C(  1952930986), -INT32_C(  2066046585),  INT32_C(   868160717),
         INT32_C(  1460791125),  INT32_C(  1853145002), -INT32_C(   577226843), -INT32_C(  1420841085), -INT32_C(   594730618),  INT32_C(  1281828549), -INT32_C(  1798290745), -INT32_C(  1396207530) },
      {  INT32_C(   553900151),  INT32_C(  1032812440),  INT32_C(   823863214),  INT32_C(  1574791894),  INT32_C(  1345940107),  INT32_C(  1469882768),  INT32_C(   971730403),  INT32_C(  1961210877),
         INT32_C(   597027211),  INT32_C(   258024800),  INT32_C(   591428684),  INT32_C(  1904221670),  INT32_C(   415414664),  INT32_C(  1030774362), -INT32_C(   931701813), -INT32_C(  1707320306) },
      {  INT32_C(   600887559),  INT32_C(   555857050),  INT32_C(  1499115450),  INT32_C(   751445742), -INT32_C(  1293173341), -INT32_C(   603716922), -INT32_C(  1224904514),  INT32_C(   437932562),
         INT32_C(  1954811614),  INT32_C(  1628805834),  INT32_C(     1757596), -INT32_C(   623922587), -INT32_C(  1001416690),  INT32_C(  1897363615), -INT32_C(   483019055),  INT32_C(   754798344) } },
    { { -INT32_C(  1497509307),  INT32_C(  1135943415), -INT32_C(  2140736102), -INT32_C(  1678580205), -INT32_C(    88886112), -INT32_C(   566746350), -INT32_C(  1918455937),  INT32_C(  1361568523),
        -INT32_C(  1393039947), -INT32_C(  1628460029), -INT32_C(  1273080159), -INT32_C(   598798276), -INT32_C(   673774652), -INT32_C(  1514860762), -INT32_C(   919381058), -INT32_C(   216376770) },
      UINT16_C( 4415),
      { -INT32_C(  1883421793), -INT32_C(     1744927),  INT32_C(  1662001427), -INT32_C(   714681090),  INT32_C(  1625590955),  INT32_C(  1706861106), -INT32_C(  2034238869), -INT32_C(  1919418386),
         INT32_C(   589059394), -INT32_C(   954072652),  INT32_C(   556413475), -INT32_C(  1292464121),  INT32_C(  1326701085), -INT32_C(   407580804), -INT32_C(  1217563703), -INT32_C(  1270545038) },
      {  INT32_C(   248996186), -INT32_C(  2032797085),  INT32_C(   866582572), -INT32_C(  1377395312), -INT32_C(   201524873), -INT32_C(  1847938872), -INT32_C(  1773582300), -INT32_C(  1488220595),
         INT32_C(  1387602671),  INT32_C(  1238927900),  INT32_C(   461143946), -INT32_C(  1815584228),  INT32_C(   596034907), -INT32_C(  1716166283), -INT32_C(   164561239),  INT32_C(  2073918348) },
      { -INT32_C(  2123750715),  INT32_C(  2033231234),  INT32_C(  1354178879),  INT32_C(  2021673326), -INT32_C(  1827113508), -INT32_C(   194637062), -INT32_C(  1918455937),  INT32_C(  1361568523),
         INT32_C(  1906931629), -INT32_C(  1628460029), -INT32_C(  1273080159), -INT32_C(   598798276),  INT32_C(  1821712198), -INT32_C(  1514860762), -INT32_C(   919381058), -INT32_C(   216376770) } },
    { { -INT32_C(  1160883555),  INT32_C(  1728292572),  INT32_C(  1115848486),  INT32_C(  1037454050), -INT32_C(  2057217009),  INT32_C(  1713247933), -INT32_C(  1604563436),  INT32_C(  1729886665),
         INT32_C(   673311051), -INT32_C(  1232132976), -INT32_C(  2047340125),  INT32_C(  1807994459), -INT32_C(   386915285),  INT32_C(  1313803834),  INT32_C(   653175645), -INT32_C(   259192411) },
      UINT16_C(44787),
      { -INT32_C(  1479376104),  INT32_C(   834172473), -INT32_C(  1107290885),  INT32_C(  1877093247),  INT32_C(  1652366355),  INT32_C(  1477368426), -INT32_C(  1906134271), -INT32_C(  1136831069),
         INT32_C(   291704792), -INT32_C(  2126373755), -INT32_C(  1354808784), -INT32_C(  2128731539), -INT32_C(  1444701378),  INT32_C(  1996616054),  INT32_C(  1208312740), -INT32_C(  1878768968) },
      { -INT32_C(   677288110), -INT32_C(  1286019965), -INT32_C(  1805477850), -INT32_C(   132808518), -INT32_C(  1834878948), -INT32_C(  1911970838), -INT32_C(  1076490746), -INT32_C(  1555047855),
        -INT32_C(   981798590), -INT32_C(    75901995),  INT32_C(   630184811),  INT32_C(  2015208540), -INT32_C(  2029338723),  INT32_C(  1746211682),  INT32_C(  1915218977),  INT32_C(   135624646) },
      {  INT32_C(  1886643274), -INT32_C(  2099211590),  INT32_C(  1115848486),  INT32_C(  1037454050), -INT32_C(   253959153), -INT32_C(   704153216),  INT32_C(   833928455),  INT32_C(   527601650),
         INT32_C(   673311051),  INT32_C(  2050740048), -INT32_C(  1968137893), -INT32_C(   117209551), -INT32_C(   386915285),  INT32_C(   521462292),  INT32_C(   653175645), -INT32_C(  1743702658) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_xor_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_xor_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_xor_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(31138),
      { -INT32_C(   268230814), -INT32_C(   127828744), -INT32_C(   652244455), -INT32_C(   178928494), -INT32_C(  1171138727),  INT32_C(  1381371835),  INT32_C(  1040805621), -INT32_C(   726160526),
        -INT32_C(  1027294518),  INT32_C(  1337664822),  INT32_C(  1093196463), -INT32_C(   147358050),  INT32_C(   296904789),  INT32_C(  1818429559), -INT32_C(   374707081), -INT32_C(   507682537) },
      {  INT32_C(  1386512668),  INT32_C(  1453481894), -INT32_C(   661140935), -INT32_C(  1630548408), -INT32_C(  1364229833),  INT32_C(    18551434), -INT32_C(  1762998913),  INT32_C(  1131980583),
        -INT32_C(   829088728), -INT32_C(  1272694917),  INT32_C(  1250737154), -INT32_C(  1041736566),  INT32_C(  1735432157),  INT32_C(   694717354),  INT32_C(  2009027152),  INT32_C(   565852153) },
      {  INT32_C(           0), -INT32_C(  1362943138),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1397562673),  INT32_C(           0), -INT32_C(  1748038571),
         INT32_C(   206677730),  INT32_C(           0),  INT32_C(           0),  INT32_C(   920592916),  INT32_C(  1992490888),  INT32_C(  1158382557), -INT32_C(  1642775001),  INT32_C(           0) } },
    { UINT16_C(20563),
      {  INT32_C(   344444656),  INT32_C(   265324931), -INT32_C(  1133814828), -INT32_C(  1940633316),  INT32_C(   421002928), -INT32_C(   429168601), -INT32_C(  1709283873),  INT32_C(  1995075974),
        -INT32_C(  1014271680), -INT32_C(   808297477), -INT32_C(   745849162),  INT32_C(   929030023),  INT32_C(    72382429), -INT32_C(  1091846945), -INT32_C(  1487402719), -INT32_C(  1172422022) },
      { -INT32_C(  1350719052), -INT32_C(  1149349884),  INT32_C(   361630094), -INT32_C(   968036887),  INT32_C(  1137417572),  INT32_C(  2046998105),  INT32_C(   958487231),  INT32_C(  1374961565),
        -INT32_C(   318737944),  INT32_C(  1336442561),  INT32_C(  1885615751), -INT32_C(  2009616093), -INT32_C(  1479867826),  INT32_C(  1998704056), -INT32_C(   978304472),  INT32_C(  1779868802) },
      { -INT32_C(  1141217468), -INT32_C(  1263609465),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1524392916),  INT32_C(           0), -INT32_C(  1556136096),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1550092909),  INT32_C(           0),  INT32_C(  1659390729),  INT32_C(           0) } },
    { UINT16_C( 5654),
      { -INT32_C(     7022761), -INT32_C(  1959453913), -INT32_C(  1019455092), -INT32_C(  1396339999),  INT32_C(  1383759153), -INT32_C(  1533697292),  INT32_C(  2101942119), -INT32_C(   644653183),
         INT32_C(  1574446902), -INT32_C(   823587262),  INT32_C(  1217471847), -INT32_C(   504080720), -INT32_C(   952865069),  INT32_C(  2003618064),  INT32_C(  1643427296),  INT32_C(  1245349652) },
      { -INT32_C(   240643154), -INT32_C(  2000711391),  INT32_C(  1724928438),  INT32_C(  2068366503),  INT32_C(  1128430643),  INT32_C(   633056837),  INT32_C(  2005315427), -INT32_C(   440221385),
        -INT32_C(   170497324), -INT32_C(  1317169413), -INT32_C(  1911009817),  INT32_C(  1141465105),  INT32_C(   562580700),  INT32_C(  1581663226),  INT32_C(   701877490),  INT32_C(  1628411789) },
      {  INT32_C(           0),  INT32_C(    59410950), -INT32_C(  1511258054),  INT32_C(           0),  INT32_C(   288882946),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  2140444857), -INT32_C(   964073344),  INT32_C(           0), -INT32_C(   423878129),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(58625),
      { -INT32_C(   746849194), -INT32_C(   970956114), -INT32_C(    98159887), -INT32_C(    28966282),  INT32_C(  1765884195),  INT32_C(  1949643935),  INT32_C(  1812775773),  INT32_C(  2068974884),
        -INT32_C(  1202795254),  INT32_C(   561934128),  INT32_C(   387687585), -INT32_C(   921279834),  INT32_C(  1110595491), -INT32_C(   373921909),  INT32_C(  1331020330), -INT32_C(   624252976) },
      { -INT32_C(  1517152139),  INT32_C(   684068999),  INT32_C(  1514201524), -INT32_C(   433891773),  INT32_C(   958944685), -INT32_C(   417144900),  INT32_C(  1899394977), -INT32_C(  1807024097),
        -INT32_C(  1606820584), -INT32_C(  1580662803),  INT32_C(   620431584),  INT32_C(   201989726),  INT32_C(   809841523), -INT32_C(  1290311918), -INT32_C(    47886882),  INT32_C(  1704030285) },
      {  INT32_C(  1995039779),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(   410456082),  INT32_C(           0),  INT32_C(   870362177),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1520500889), -INT32_C(  1301245964), -INT32_C(  1084500067) } },
    { UINT16_C(51790),
      { -INT32_C(   825672955), -INT32_C(   657020195), -INT32_C(   654887474), -INT32_C(  2046072255), -INT32_C(  1309860454), -INT32_C(   151073839),  INT32_C(  1516653513), -INT32_C(  1239042895),
        -INT32_C(   846926096),  INT32_C(  1722112920), -INT32_C(   784360304), -INT32_C(  1588114937),  INT32_C(   961758312), -INT32_C(   667987697),  INT32_C(  1312003485),  INT32_C(   973363274) },
      { -INT32_C(   553154234),  INT32_C(  1950723300),  INT32_C(  1346733129),  INT32_C(   938581199), -INT32_C(   277855008),  INT32_C(   868786070),  INT32_C(  2122447668), -INT32_C(  1715960493),
        -INT32_C(   226967794), -INT32_C(  1251557780),  INT32_C(   285584194),  INT32_C(   659093063), -INT32_C(   786974661), -INT32_C(  1945837737),  INT32_C(   755664346),  INT32_C(   432456458) },
      {  INT32_C(           0), -INT32_C(  1399650759), -INT32_C(  2001555065), -INT32_C(  1308952946),  INT32_C(           0),  INT32_C(           0),  INT32_C(   619163901),  INT32_C(           0),
         INT32_C(           0), -INT32_C(   742136332),  INT32_C(           0), -INT32_C(  2044740544),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1664684103),  INT32_C(   599956288) } },
    { UINT16_C(16258),
      {  INT32_C(  1929244171), -INT32_C(  1474478173), -INT32_C(  1717607087), -INT32_C(  1554851188), -INT32_C(  1350391381),  INT32_C(  1077173301),  INT32_C(  1342389897),  INT32_C(  1670350167),
         INT32_C(   399871092),  INT32_C(   482407115), -INT32_C(   457810089), -INT32_C(   460912583), -INT32_C(   409793871), -INT32_C(   265828506),  INT32_C(  1564486149),  INT32_C(   599838639) },
      {  INT32_C(   641439067), -INT32_C(   549192825), -INT32_C(  1815873190), -INT32_C(  1300805119), -INT32_C(  1147598252), -INT32_C(   693387055), -INT32_C(  1707873302),  INT32_C(   364770234),
         INT32_C(   272365704),  INT32_C(  1307541235),  INT32_C(  2027991671),  INT32_C(  1361795068),  INT32_C(   839697505),  INT32_C(  1862907781),  INT32_C(  1544109218), -INT32_C(  1200502992) },
      {  INT32_C(           0),  INT32_C(  2002699300),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1983018733),
         INT32_C(   133068028),  INT32_C(  1362070584), -INT32_C(  1672024800), -INT32_C(  1246994491), -INT32_C(   710947120), -INT32_C(  1624346397),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(44223),
      { -INT32_C(  1221872696), -INT32_C(   513170944),  INT32_C(  1178101275), -INT32_C(  1022715209), -INT32_C(   713388084),  INT32_C(   135404799), -INT32_C(   372293000), -INT32_C(  1013608454),
         INT32_C(  1115340866),  INT32_C(  2116281443),  INT32_C(    29645898), -INT32_C(  1027289354),  INT32_C(  1570258526), -INT32_C(   765089190), -INT32_C(   423938580),  INT32_C(   111759556) },
      {  INT32_C(  1950884881),  INT32_C(  1391619080), -INT32_C(  1101809976), -INT32_C(   494921852), -INT32_C(  1354819499), -INT32_C(  1367235134), -INT32_C(  1634386726), -INT32_C(  1650180468),
         INT32_C(  1779559522),  INT32_C(   549192536),  INT32_C(  1054740409),  INT32_C(  2065719078),  INT32_C(   975855735), -INT32_C(   538399995),  INT32_C(  1954381287),  INT32_C(   504439228) },
      { -INT32_C(  1016883239), -INT32_C(  1281634808), -INT32_C(   127151917),  INT32_C(   562728243),  INT32_C(  2051381145), -INT32_C(  1500268227),  INT32_C(           0),  INT32_C(  1580314998),
         INT32_C(           0),  INT32_C(           0),  INT32_C(  1058690035), -INT32_C(  1176202800),  INT32_C(           0),  INT32_C(   227344735),  INT32_C(           0),  INT32_C(   414740856) } },
    { UINT16_C( 8973),
      {  INT32_C(  1160144521),  INT32_C(  1700126854),  INT32_C(  1053063966),  INT32_C(   564018167),  INT32_C(  1590535029), -INT32_C(     2378622),  INT32_C(   975214376),  INT32_C(  1063071414),
         INT32_C(   444892052), -INT32_C(  2122327709),  INT32_C(  1287668565), -INT32_C(   194158977), -INT32_C(  2074985982),  INT32_C(   394538735),  INT32_C(  2068948165),  INT32_C(  1723510482) },
      { -INT32_C(  1786691790),  INT32_C(  1830223896), -INT32_C(  1011230908),  INT32_C(   934750005),  INT32_C(  1354500705), -INT32_C(    43499464), -INT32_C(  1216759324), -INT32_C(  1709362328),
        -INT32_C(  1959813518), -INT32_C(   470268257),  INT32_C(  1403433501),  INT32_C(   982146521), -INT32_C(  1618262425),  INT32_C(  1805513606),  INT32_C(   337778348), -INT32_C(  1129431223) },
      { -INT32_C(   794338885),  INT32_C(           0), -INT32_C(    42095526),  INT32_C(   371793090),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(  1851056666),  INT32_C(  1653055484),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2082069865),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_xor_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_xor_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 7833673612016144483), -INT64_C( 6286797061584980339), -INT64_C( 1894311286508114473),  INT64_C( 7457901623545148981),
         INT64_C( 7758400186936267162), -INT64_C( 7827311421232976810),  INT64_C( 5569310173466725242), -INT64_C( 1852634981124308256) },
      { -INT64_C( 6235004628972997840), -INT64_C( 3743688304300620355), -INT64_C( 8561205822545100854),  INT64_C( 2103383255765864446),
        -INT64_C( 2114542116453788245), -INT64_C( 7978576687197911256), -INT64_C( 1001589171339167514), -INT64_C( 1694078873958054286) },
      { -INT64_C( 4193382358810418349),  INT64_C( 7263022192053547824),  INT64_C( 7820062331317703197),  INT64_C( 8813401457401952715),
        -INT64_C( 8571197845464267727),  INT64_C(  151340619297975166), -INT64_C( 4660229538766687332),  INT64_C( 1024413272115446930) } },
    { {  INT64_C( 8415412740149389505),  INT64_C( 8136937495115801748),  INT64_C( 8472142827390787501), -INT64_C(  232688289166536801),
        -INT64_C( 3621830638293505029), -INT64_C( 8157403346233676848), -INT64_C( 8940555774217696396),  INT64_C( 1209819125977694866) },
      {  INT64_C( 3014619373212577232), -INT64_C( 8706215558938034241), -INT64_C( 5583923536013140710), -INT64_C( 3039284310411517779),
        -INT64_C(  181353180207109384),  INT64_C( 7516167267833216947), -INT64_C( 1874932065944318681), -INT64_C( 4710068369810294072) },
      {  INT64_C( 6710250289300899089), -INT64_C(  594058874992207061), -INT64_C( 4101950076832841033),  INT64_C( 2960861935095328562),
         INT64_C( 3514804452886151427), -INT64_C( 1835816431532026781),  INT64_C( 7356107569477813843), -INT64_C( 5879346165530970022) } },
    { { -INT64_C( 1666178802879790940),  INT64_C( 8017248771709566347),  INT64_C( 6370254332653643850),  INT64_C( 3219730174574923307),
        -INT64_C( 3070348517276991929),  INT64_C( 6495017567146349163), -INT64_C( 5439991518983235431), -INT64_C( 5474315090744007085) },
      {  INT64_C( 8012065384011076881),  INT64_C( 7498569038459756453), -INT64_C( 8909788908341826275),  INT64_C( 7194894378406510264),
         INT64_C( 4751279587353037180),  INT64_C( 8288320232820773582),  INT64_C( 7109622136361818100), -INT64_C( 1430348694698849934) },
      { -INT64_C( 8660395485888014923),  INT64_C(  527688101943681582), -INT64_C( 2576724598806067881),  INT64_C( 5726223408186454163),
        -INT64_C( 7742811636499433669),  INT64_C( 2964766335513985189), -INT64_C( 3014254901916840083),  INT64_C( 6350412050110132001) } },
    { { -INT64_C( 6753046675043064470),  INT64_C( 4194897748161412973),  INT64_C( 3956704487311102429),  INT64_C( 4426793051552903443),
        -INT64_C( 6594523062152668411), -INT64_C( 2500730345433285685), -INT64_C( 4041049702590534928),  INT64_C(  851383767448348095) },
      {  INT64_C( 7741231053211505812),  INT64_C( 7420181171167614491),  INT64_C( 7776599092500323953), -INT64_C( 8935477080687233300),
         INT64_C( 6912388001218618338), -INT64_C( 6814099961521935981),  INT64_C( 9145319435102817234), -INT64_C( 7467689775987562262) },
      { -INT64_C( 3952406250137193986),  INT64_C( 6687425642990135670),  INT64_C( 6702779462972608428), -INT64_C( 4714725391518644225),
        -INT64_C(  318013443086337817),  INT64_C( 8945533351659557464), -INT64_C( 5115529358649122526), -INT64_C( 7814353725753193131) } },
    { { -INT64_C(  363657528445588654), -INT64_C( 5313838659546157200), -INT64_C( 6481207919005857538),  INT64_C( 4635854309387370247),
        -INT64_C( 1807691163233783929),  INT64_C( 6833305750270831930),  INT64_C( 3401057919711447504),  INT64_C( 1561413775749067132) },
      { -INT64_C( 8859444360416820816),  INT64_C( 3069616949972669147), -INT64_C(   10721871889266815), -INT64_C( 5170953676681197927),
         INT64_C(  749826453865906582),  INT64_C(  288756504667163261),  INT64_C( 4172734175648855909), -INT64_C(  212340540276977075) },
      {  INT64_C( 9221376615234022114), -INT64_C( 7144947895540580949),  INT64_C( 6473914840305078143), -INT64_C(  546965349575837282),
        -INT64_C( 1401145767694422511),  INT64_C( 6545166095291246407),  INT64_C( 1646764641365311157), -INT64_C( 1682414516424613071) } },
    { {  INT64_C( 6059952658790214623), -INT64_C( 6712846976013345200),  INT64_C( 6538509831123994084), -INT64_C( 7337563655741956666),
        -INT64_C( 6786149137027429721),  INT64_C( 1152722674958794016), -INT64_C( 4236109339647833136),  INT64_C( 5314845287287874925) },
      {  INT64_C( 5484145387571211049), -INT64_C( 6138219535088906346),  INT64_C( 2375857380799177524),  INT64_C( 3786695004706696354),
        -INT64_C( 8939704712816601994),  INT64_C( 8564742865895594328), -INT64_C( 4446283047571463305),  INT64_C( 1643131752719750285) },
      {  INT64_C( 1730191473646272758),  INT64_C(  578577205123273158),  INT64_C( 8810670071267321552), -INT64_C( 5861795896640474780),
         INT64_C( 2467137782418923217),  INT64_C( 8728925404920833144),  INT64_C(  539853474565384359),  INT64_C( 6849847759138550240) } },
    { { -INT64_C( 7108746993929192473),  INT64_C( 2952580909138746433), -INT64_C( 3318415959729659377),  INT64_C( 8351705111930271794),
        -INT64_C( 8460680085403819854),  INT64_C( 2384100826312238396),  INT64_C( 4161117182544079274),  INT64_C( 1385686332839962939) },
      {  INT64_C( 6874356853280244370),  INT64_C( 2681720383917028356),  INT64_C(  363617062994806731), -INT64_C( 7318342761473894660),
         INT64_C( 2164784578008873609), -INT64_C( 8795130151555541168), -INT64_C( 3115818213941246733),  INT64_C( 2097447710665838201) },
      { -INT64_C( 4450058466010859147),  INT64_C(  994959937184262213), -INT64_C( 3100361591938341436), -INT64_C( 1614767496023839026),
        -INT64_C( 7737328937327470021), -INT64_C( 6564158462417554836), -INT64_C( 1333872736001888935),  INT64_C( 1018187299377882946) } },
    { { -INT64_C( 4803982436490178277), -INT64_C( 1771749786168752086), -INT64_C( 7193825152807419778),  INT64_C( 1704113265784478708),
         INT64_C( 7923416425212768202),  INT64_C( 8576185165226802729),  INT64_C(  160712378890954874),  INT64_C( 6585724530256992338) },
      {  INT64_C( 9029586771277755040), -INT64_C( 3064805812496326649), -INT64_C( 8261663046046104505), -INT64_C( 5408189745344027899),
         INT64_C( 1640407638281258483),  INT64_C( 4951300961883020935), -INT64_C( 2550739608973727279), -INT64_C( 6368259520359861034) },
      { -INT64_C( 4603986968108719173),  INT64_C( 3611576490511247405),  INT64_C( 1257290549943567417), -INT64_C( 6677578309627889423),
         INT64_C( 8878362109278874169),  INT64_C( 3725100340635705518), -INT64_C( 2404072443576090197), -INT64_C(  217769111299019644) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_xor_epi64(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_xor_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C(   45749859996612839),  INT64_C( 2814572066373608668), -INT64_C( 6582897561231417342), -INT64_C( 8687305550852076307),
         INT64_C( 8500406415190396854),  INT64_C( 5093384555923498535),  INT64_C( 8909703965306565750),  INT64_C(  546313591099609965) },
      UINT8_C(244),
      { -INT64_C( 2585626980132258187),  INT64_C(  726882011601479497), -INT64_C( 8038622157332801693), -INT64_C( 8691410639997796728),
         INT64_C( 4087190315360841190), -INT64_C( 6654652016916703164), -INT64_C( 2859196078284653307),  INT64_C(  229866223008298134) },
      { -INT64_C( 7945590762679324780), -INT64_C( 8839772767949909393), -INT64_C( 6397194970787743473), -INT64_C( 5121297404723757619),
        -INT64_C( 3311000718894445522),  INT64_C( 1769302264377325704),  INT64_C( 4288746735372129484), -INT64_C( 9076703424935504851) },
      {  INT64_C(   45749859996612839),  INT64_C( 2814572066373608668),  INT64_C( 3983867818266324588), -INT64_C( 8687305550852076307),
        -INT64_C( 1534483321180642872), -INT64_C( 4960646585427500852), -INT64_C( 2029249597307622967), -INT64_C( 9135072113565048645) } },
    { { -INT64_C( 4389873590959520546),  INT64_C( 3847016815465829697), -INT64_C( 5127463024880002259), -INT64_C( 1350777454883832725),
        -INT64_C( 8622911936856637864), -INT64_C( 6379571547940445736), -INT64_C( 8262950684676879856), -INT64_C( 3957664130431951338) },
      UINT8_C(203),
      { -INT64_C( 3288302065678132158), -INT64_C( 6867158103737196692),  INT64_C(   29078381664489015), -INT64_C( 2093545317731299942),
         INT64_C(  800449542886982807), -INT64_C( 7919193623266236573), -INT64_C( 4301136478309459027), -INT64_C( 7828699826288344616) },
      { -INT64_C( 1617647112863104542),  INT64_C( 5065560676272060766),  INT64_C( 5928115323832446807), -INT64_C( 1137963685561880410),
         INT64_C( 7301739104183587045), -INT64_C( 8568825949546695114),  INT64_C( 3848130783034911386),  INT64_C( 4286595986196214142) },
      {  INT64_C( 4310345289044696480), -INT64_C( 1801852656978844110), -INT64_C( 5127463024880002259),  INT64_C( 1353085047476567356),
        -INT64_C( 8622911936856637864), -INT64_C( 6379571547940445736), -INT64_C( 1069598699435757257), -INT64_C( 6329859119562035034) } },
    { {  INT64_C( 2488796161233055261), -INT64_C( 6022150448521041999),  INT64_C( 4933174169979353009),  INT64_C( 8640803886955777369),
         INT64_C( 4344324970652602684), -INT64_C(  898031332366467893), -INT64_C( 3887263497667196412),  INT64_C( 7478337318462021367) },
      UINT8_C(245),
      {  INT64_C( 1608838238175268707),  INT64_C(  319276987121676509),  INT64_C( 2531375131368175488), -INT64_C( 6984319707087211591),
         INT64_C( 3257643290499388229),  INT64_C( 6514463296540085930), -INT64_C( 2908017913228894916), -INT64_C( 1228629849485806824) },
      {  INT64_C( 4402061463343995290), -INT64_C( 1223561056477925679),  INT64_C( 5177755488274713062),  INT64_C( 8115643581275557640),
        -INT64_C( 4361610958211469656),  INT64_C(  631194590693431583),  INT64_C( 5869570511533257725),  INT64_C( 5115705028593717175) },
      {  INT64_C( 3117896433371354873), -INT64_C( 6022150448521041999),  INT64_C( 7276222062497414758),  INT64_C( 8640803886955777369),
        -INT64_C( 1275353631281103379),  INT64_C( 5956698770186540981), -INT64_C( 8732406741720305983), -INT64_C( 6337227441341793105) } },
    { { -INT64_C( 1655642640894306224), -INT64_C( 1928527008934104302), -INT64_C( 1780464257288025679),  INT64_C( 4645871961807080853),
         INT64_C(  396184772487084747),  INT64_C( 7694247929610583034), -INT64_C(  876321354710525762), -INT64_C( 8490798505132667909) },
      UINT8_C(153),
      { -INT64_C( 2797138209894817876), -INT64_C( 5764892471572950838),  INT64_C( 1083398023113731424), -INT64_C( 3864555640558062810),
         INT64_C( 5039295415212501623), -INT64_C( 2395207474291048132), -INT64_C(  629670723305287568), -INT64_C( 2391099797848452888) },
      {  INT64_C( 4019150386831347461),  INT64_C( 4521628108579294052), -INT64_C( 8456435593802240804), -INT64_C( 9148392511393210172),
         INT64_C( 5071836999787413040), -INT64_C( 7700616903254299628),  INT64_C(  919268992504468373), -INT64_C( 2070738063835246265) },
      { -INT64_C( 1231608716974305111), -INT64_C( 1928527008934104302), -INT64_C( 1780464257288025679),  INT64_C( 5428023262223111138),
         INT64_C(  256112104962790471),  INT64_C( 7694247929610583034), -INT64_C(  876321354710525762),  INT64_C( 4436707353278300591) } },
    { {  INT64_C( 2479355900382364993),  INT64_C( 1111304417577308446), -INT64_C( 2590687349108723784), -INT64_C( 4039761275362786215),
        -INT64_C( 8189705280462062040),  INT64_C( 4210251584504159459),  INT64_C( 1509346594468311936), -INT64_C( 7144943817645321846) },
      UINT8_C( 14),
      { -INT64_C( 8721207303514172734),  INT64_C( 8557758222839600753),  INT64_C( 6723388035183008125), -INT64_C(  395352253016219803),
         INT64_C( 9146538221902954815), -INT64_C( 5145499759254898594),  INT64_C( 8396340341971405148),  INT64_C(  170411947343828522) },
      { -INT64_C( 3244744075203875731), -INT64_C( 7474133439378900910), -INT64_C( 5127834652165225900), -INT64_C( 7466837154034172005),
         INT64_C( 8659025868352572388), -INT64_C(  173585767618699581),  INT64_C( 4130986808766912526), -INT64_C( 3838482065090187530) },
      {  INT64_C( 2479355900382364993), -INT64_C( 1259379519748575709), -INT64_C( 1902770539115504855),  INT64_C( 7125567705071739134),
        -INT64_C( 8189705280462062040),  INT64_C( 4210251584504159459),  INT64_C( 1509346594468311936), -INT64_C( 7144943817645321846) } },
    { { -INT64_C( 7322204454426325821), -INT64_C( 4462169255779407764), -INT64_C( 6962650059328556152), -INT64_C( 5561527802563267039),
        -INT64_C( 5928710453305019577),  INT64_C(  782735190010253669),  INT64_C( 1049778440287304264),  INT64_C( 1524159294001730182) },
      UINT8_C(217),
      {  INT64_C( 4406147630084273090),  INT64_C( 6172405503935594122),  INT64_C( 9083848596339727098),  INT64_C( 7864517138114643045),
        -INT64_C( 8017162673287577708),  INT64_C( 4755687165104520592), -INT64_C(  658691503089303246),  INT64_C( 1979444766424770400) },
      { -INT64_C( 8774039058048637787), -INT64_C(  694063673707057962),  INT64_C( 7347955965608798833),  INT64_C( 5114374633758407510),
        -INT64_C( 4392929463379061619), -INT64_C( 2660713234140481191), -INT64_C( 1381242144590764183), -INT64_C( 4973730811908373486) },
      { -INT64_C( 4964763279348885657), -INT64_C( 4462169255779407764), -INT64_C( 6962650059328556152),  INT64_C( 3160832092535504691),
         INT64_C( 6031583149779954457),  INT64_C(  782735190010253669),  INT64_C( 1877732794698114651), -INT64_C( 6808980108460137614) } },
    { {  INT64_C( 8216894590175538007), -INT64_C( 8015102044995065729), -INT64_C( 8319041892314537462),  INT64_C( 8069950358681507095),
         INT64_C(  121240403729136420), -INT64_C( 1487121492127105372),  INT64_C( 4995924851354721104), -INT64_C(  940257476621986452) },
      UINT8_C(196),
      {  INT64_C( 1947321805534736039), -INT64_C( 8561651653089111504), -INT64_C( 2092489041890510734),  INT64_C( 1500721690142866054),
        -INT64_C( 8085793806064609071),  INT64_C( 2163588935650821455),  INT64_C( 4667027440343170452), -INT64_C( 1686560576050858730) },
      { -INT64_C( 2582427849531715781), -INT64_C( 3968128441716664540),  INT64_C( 5195024434345973022), -INT64_C( 4187110105943349020),
        -INT64_C(   81476173827179916), -INT64_C( 5323890722526513120),  INT64_C( 1931383667822610259),  INT64_C( 8313639412841224719) },
      {  INT64_C( 8216894590175538007), -INT64_C( 8015102044995065729), -INT64_C( 6130081342750364308),  INT64_C( 8069950358681507095),
         INT64_C(  121240403729136420), -INT64_C( 1487121492127105372),  INT64_C( 6487782212416142023), -INT64_C( 7221564986028855527) } },
    { {  INT64_C( 7515754918875348596),  INT64_C( 3232135209222413451),  INT64_C( 2229984471521781921),  INT64_C( 8541946416010364953),
        -INT64_C( 2412501877195996971), -INT64_C( 4885037687150329554),  INT64_C( 9039434068048209220),  INT64_C( 1659050078198114957) },
      UINT8_C( 33),
      { -INT64_C(  429157281734249269),  INT64_C( 3203978284322067029),  INT64_C( 7455724837908040293), -INT64_C( 2817354451018840157),
         INT64_C(  335448553716800499), -INT64_C(  917138322303370029), -INT64_C( 8711837563429726603),  INT64_C( 8377595263254839645) },
      { -INT64_C( 8340727286980282227), -INT64_C( 7636439070152493692), -INT64_C( 7697670767163939903),  INT64_C( 3969997471685310498),
        -INT64_C( 8826859548180268741), -INT64_C( 3251989183356983769), -INT64_C( 6246875829788793216), -INT64_C( 3603330975078199780) },
      {  INT64_C( 8517603223431072838),  INT64_C( 3232135209222413451),  INT64_C( 2229984471521781921),  INT64_C( 8541946416010364953),
        -INT64_C( 2412501877195996971),  INT64_C( 2421589282539805428),  INT64_C( 9039434068048209220),  INT64_C( 1659050078198114957) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_xor_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_xor_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_xor_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(152),
      { -INT64_C( 8982237529053997842),  INT64_C( 3103674849943855424),  INT64_C( 1982415417451867211), -INT64_C( 9083321110428971905),
         INT64_C( 7887070877457153635),  INT64_C( 7936046385343671330),  INT64_C( 1142105130860059500), -INT64_C( 2327418192463860577) },
      {  INT64_C( 9182915906006801623), -INT64_C( 3782010632694931727), -INT64_C( 4191618534024375243),  INT64_C( 7829976987492367438),
        -INT64_C( 5664436787247809914), -INT64_C( 6275684222459520321),  INT64_C( 2362774424294093278), -INT64_C(  972614327583719042) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 1344283778605633487),
        -INT64_C( 2587491794353060635),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 3257172424768237025) } },
    { UINT8_C( 14),
      {  INT64_C( 1409767669188450032),  INT64_C(  535417729420012870), -INT64_C(  117662871603022948), -INT64_C( 2123625407593156124),
         INT64_C( 2053434539035022643), -INT64_C( 3346989035654740496),  INT64_C( 5028564073104500622), -INT64_C( 4300579678214802307) },
      {  INT64_C( 4002766890185262234), -INT64_C( 2181348245731657548), -INT64_C(  911567200057529934), -INT64_C(  648602612603414614),
        -INT64_C( 7910868655864926281), -INT64_C(  981726171945503701),  INT64_C( 3172889850273926303), -INT64_C( 2970011671702574219) },
      {  INT64_C(                   0), -INT64_C( 1813713950618380814),  INT64_C(  938028879597345326),  INT64_C( 1475191305885214286),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(149),
      { -INT64_C(  122716449265167314),  INT64_C( 3987427480104258177), -INT64_C( 2447171538408346827), -INT64_C( 3365926546413069491),
         INT64_C( 4949525000721972647),  INT64_C( 6082763436880293385),  INT64_C( 6273216357046166716), -INT64_C( 8772636185306919988) },
      {  INT64_C( 8870376520984048402),  INT64_C( 2542591394626541015), -INT64_C( 1180399248176107203),  INT64_C( 1283984387118967405),
        -INT64_C( 3908843060537718317), -INT64_C( 4206915619632812268), -INT64_C( 4662164243526913735),  INT64_C( 2190117534882835284) },
      { -INT64_C( 8838896490382629060),  INT64_C(                   0),  INT64_C( 3573473610564262408),  INT64_C(                   0),
        -INT64_C( 8254885696627231628),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 7483423931756404584) } },
    { UINT8_C(230),
      { -INT64_C( 7827641925076753334),  INT64_C( 7539554054630529375), -INT64_C(  798410052431249940), -INT64_C( 1276666438538543227),
         INT64_C( 2340348715907686594),  INT64_C( 7084207804521158000),  INT64_C(  821033393377053478), -INT64_C( 4272309647950633948) },
      { -INT64_C(  813422530811728427), -INT64_C( 8173422651522728262), -INT64_C( 7654040801051243466), -INT64_C( 4698359519177329658),
         INT64_C( 6158402678492002303), -INT64_C( 5386841171882060983), -INT64_C( 6509328055499932508),  INT64_C( 1734864526403044972) },
      {  INT64_C(                   0), -INT64_C( 1858918950121707547),  INT64_C( 7001998150177485274),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 2923345196226853319), -INT64_C( 5850495760465381502), -INT64_C( 2547120826193042872) } },
    { UINT8_C(241),
      {  INT64_C( 7402477510674193255), -INT64_C( 1863195392533566616), -INT64_C( 1608300861362718454),  INT64_C( 9026368236341560280),
         INT64_C( 2030747075921039797), -INT64_C( 5213830620423384300),  INT64_C( 3226169786223897081), -INT64_C( 7548099076051449941) },
      { -INT64_C( 1412565848515727723), -INT64_C( 5042819090609471071),  INT64_C( 5906290208258063991), -INT64_C( 3559159576159619549),
         INT64_C( 3046259127309826945), -INT64_C( 5262319553509592963),  INT64_C(  968688341824409210),  INT64_C( 6482559857543185902) },
      { -INT64_C( 8439907556581526030),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C( 3920617936044698164),  INT64_C(   98160633604465513),  INT64_C( 2428795596282832771), -INT64_C( 3546182036552269243) } },
    { UINT8_C(168),
      { -INT64_C( 1654971735735160050), -INT64_C( 3618396837604997422), -INT64_C( 2321671188407392160), -INT64_C(  852569122218184746),
        -INT64_C( 8527980720679920445),  INT64_C( 1070063051582525133),  INT64_C( 3153274373031825801),  INT64_C( 5582450922785503646) },
      {  INT64_C( 5365668961424373301),  INT64_C( 8590950464161730174),  INT64_C( 7872605299379151040), -INT64_C(   41541984007884651),
        -INT64_C( 6690553381164190369),  INT64_C( 7788140668098870447),  INT64_C( 2478321804687601704),  INT64_C( 2196821751906784692) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(  812751503155928899),
         INT64_C(                   0),  INT64_C( 7119249835665306210),  INT64_C(                   0),  INT64_C( 5982033988820085802) } },
    { UINT8_C( 34),
      { -INT64_C( 7099385906741118360), -INT64_C( 7139975430881601607), -INT64_C( 7228194637974216392),  INT64_C( 1322782488553664140),
         INT64_C( 5057636448096527218),  INT64_C( 7509173144721521260),  INT64_C( 3869178845850853074), -INT64_C(  792411965939119037) },
      { -INT64_C( 7115483786937574292),  INT64_C(  849890848911280179), -INT64_C( 8527612870863474487),  INT64_C( 5429785197312153057),
        -INT64_C( 3592881174692371884), -INT64_C( 8281559221398316595), -INT64_C( 1513871340195597267),  INT64_C( 4748387116286637243) },
      {  INT64_C(                   0), -INT64_C( 7556234139783962742),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 1935407695096568927),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(139),
      {  INT64_C( 4935371154977638186), -INT64_C( 3940202631690419418),  INT64_C( 3960841935430796437), -INT64_C( 6899968217622343414),
        -INT64_C( 9198094981434153942), -INT64_C( 8237377322867269296),  INT64_C( 5165840138584555705),  INT64_C( 7019203302184899410) },
      { -INT64_C( 2992976723773572708), -INT64_C( 4056219280331761360), -INT64_C( 8039221936049258773),  INT64_C( 4421622930838481093),
        -INT64_C( 3253945696776891781), -INT64_C( 1614628432426396734), -INT64_C( 7999378326465488595), -INT64_C( 7870629585987349017) },
      { -INT64_C( 7923174882398929226),  INT64_C( 1073260354067536406),  INT64_C(                   0), -INT64_C( 7105939320326063665),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(  888106185265150283) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_xor_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_xor_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_xor_si512(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_x_mm512_set_epu64(UINT64_C(0xedb78aa51009d043), UINT64_C(0xf8d6e1466c80412e),
                              UINT64_C(0x8d2f88ccf8d072d2), UINT64_C(0xd42ce380801d56eb),
                              UINT64_C(0x4f7a9f9f877cf207), UINT64_C(0x9ebf29784a068fec),
                              UINT64_C(0x14e312298b327bb2), UINT64_C(0xa4cae715b44968c3)),
      easysimd_x_mm512_set_epu64(UINT64_C(0xf857a2af697b20a2), UINT64_C(0xaff5e6cb29617e18),
                              UINT64_C(0x91849348f298760e), UINT64_C(0x1c2d26b7c60b0e1b),
                              UINT64_C(0x27f2529cc5e0d6da), UINT64_C(0x947c2755a9d7153c),
                              UINT64_C(0xab6596dc73591502), UINT64_C(0x6cb918d0cf4b1739)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x15e0280a7972f0e1), UINT64_C(0x5723078d45e13f36),
                              UINT64_C(0x1cab1b840a4804dc), UINT64_C(0xc801c537461658f0),
                              UINT64_C(0x6888cd03429c24dd), UINT64_C(0x0ac30e2de3d19ad0),
                              UINT64_C(0xbf8684f5f86b6eb0), UINT64_C(0xc873ffc57b027ffa)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(0x7fb6b2cc6cfe2095), UINT64_C(0x2b06563737a7554c),
                              UINT64_C(0x20e62cd04a81579d), UINT64_C(0x0c22d8b7c6b9afd0),
                              UINT64_C(0x7778227f653f51e3), UINT64_C(0x0c5d93feab9aa991),
                              UINT64_C(0x45c2fff1a0832972), UINT64_C(0xc6fefc42436c6b46)),
      easysimd_x_mm512_set_epu64(UINT64_C(0xe44de61268819c50), UINT64_C(0xec0f232714f1db42),
                              UINT64_C(0x1de6bf50b7448d81), UINT64_C(0x326b4ae523cd1df1),
                              UINT64_C(0x89856b1e9b31c8bb), UINT64_C(0xb4ab21a1f0881bd7),
                              UINT64_C(0x3d731298d95d6f21), UINT64_C(0xffc08a64375bf884)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x9bfb54de047fbcc5), UINT64_C(0xc709751023568e0e),
                              UINT64_C(0x3d009380fdc5da1c), UINT64_C(0x3e499252e574b221),
                              UINT64_C(0xfefd4961fe0e9958), UINT64_C(0xb8f6b25f5b12b246),
                              UINT64_C(0x78b1ed6979de4653), UINT64_C(0x393e7626743793c2)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(0xdcc26142d37fb5e3), UINT64_C(0x63c9856267e9999a),
                              UINT64_C(0xbf8d48fd4246771e), UINT64_C(0xe34d12aa2d211228),
                              UINT64_C(0x38f5235239303226), UINT64_C(0x264f8a7d4f0c7a44),
                              UINT64_C(0x3e3aa9db569a6f79), UINT64_C(0x47f95a5dbaf7c3fe)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x24f03a01ff0412a4), UINT64_C(0x5e69b3ae6e914583),
                              UINT64_C(0x61a1b3291cf72225), UINT64_C(0x5b7b6dafc3dcc6f8),
                              UINT64_C(0x643061d1edb662f2), UINT64_C(0x3eebdc3f8f4056db),
                              UINT64_C(0x11c4727a73fc286e), UINT64_C(0x561b4fb689bf8f8c)),
      easysimd_x_mm512_set_epu64(UINT64_C(0xf8325b432c7ba747), UINT64_C(0x3da036cc0978dc19),
                              UINT64_C(0xde2cfbd45eb1553b), UINT64_C(0xb8367f05eefdd4d0),
                              UINT64_C(0x5cc54283d48650d4), UINT64_C(0x18a45642c04c2c9f),
                              UINT64_C(0x2ffedba125664717), UINT64_C(0x11e215eb33484c72)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(0xa3db7414654163c1), UINT64_C(0x20295e6408f5e85e),
                              UINT64_C(0x6544618e6bd8d65e), UINT64_C(0x5e62e456253a5970),
                              UINT64_C(0x35200d7cedf89e63), UINT64_C(0x3a187ccb9bdbc4ff),
                              UINT64_C(0x8c83780e03d2ec0a), UINT64_C(0x25da10ac4ca3d5f3)),
      easysimd_x_mm512_set_epu64(UINT64_C(0xb4da361a4ff83c1b), UINT64_C(0x7f54a2cd02321d93),
                              UINT64_C(0x2222e01cb6f3b71d), UINT64_C(0x48d83f4fe210f439),
                              UINT64_C(0x30b5939d74f88fb5), UINT64_C(0x08451aa5c5aafb71),
                              UINT64_C(0x0908270bde506014), UINT64_C(0x14d2968168fbf701)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x1701420e2ab95fda), UINT64_C(0x5f7dfca90ac7f5cd),
                              UINT64_C(0x47668192dd2b6143), UINT64_C(0x16badb19c72aad49),
                              UINT64_C(0x05959ee1990011d6), UINT64_C(0x325d666e5e713f8e),
                              UINT64_C(0x858b5f05dd828c1e), UINT64_C(0x3108862d245822f2)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(0xb9888f8a15c6f599), UINT64_C(0xdae6980a3c15b8d5),
                              UINT64_C(0x17114f3e96d162e1), UINT64_C(0xaa441d9be0eb3305),
                              UINT64_C(0x7328bea0eddeb5b8), UINT64_C(0x38d955208ba6ab2c),
                              UINT64_C(0xd5a6f9d82f72b047), UINT64_C(0x468d076219769ecc)),
      easysimd_x_mm512_set_epu64(UINT64_C(0xf5fce010c130811a), UINT64_C(0x5b4c8bc96595cc6f),
                              UINT64_C(0x9ec90bdb77fd0d0f), UINT64_C(0xe13db6113bafebb0),
                              UINT64_C(0xbe6dfb35371e254d), UINT64_C(0xee5939c207b9c26b),
                              UINT64_C(0x7c3ef03f0a2d4864), UINT64_C(0xe807e98806d6b3fa)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x4c746f9ad4f67483), UINT64_C(0x81aa13c3598074ba),
                              UINT64_C(0x89d844e5e12c6fee), UINT64_C(0x4b79ab8adb44d8b5),
                              UINT64_C(0xcd454595dac090f5), UINT64_C(0xd6806ce28c1f6947),
                              UINT64_C(0xa99809e7255ff823), UINT64_C(0xae8aeeea1fa02d36)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(0x8234186be169c857), UINT64_C(0x6e3be8c42ba36d9a),
                              UINT64_C(0x9eebbbe6bd8adb2a), UINT64_C(0x6ce901141909d2cf),
                              UINT64_C(0x35459cc296fca858), UINT64_C(0x1a7d575fa8651237),
                              UINT64_C(0x4b008fe37abafacd), UINT64_C(0xf35eba645c1d884d)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x7ea8964c6c682a7c), UINT64_C(0x8b6605b470502155),
                              UINT64_C(0x4b16327f96bf6e87), UINT64_C(0xae618aa0114ea6c6),
                              UINT64_C(0x3c1572ee53b136fa), UINT64_C(0xacef14edc9d741a1),
                              UINT64_C(0x96f4d64c8555893a), UINT64_C(0x1fb0ce0c9ed59cf4)),
      easysimd_x_mm512_set_epu64(UINT64_C(0xfc9c8e278d01e22b), UINT64_C(0xe55ded705bf34ccf),
                              UINT64_C(0xd5fd89992b35b5ad), UINT64_C(0xc2888bb408477409),
                              UINT64_C(0x0950ee2cc54d9ea2), UINT64_C(0xb69243b261b25396),
                              UINT64_C(0xddf459afffef73f7), UINT64_C(0xecee7468c2c814b9)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(0xd4967d973e742c64), UINT64_C(0xcb3e880be1980939),
                              UINT64_C(0xc418352686ff3548), UINT64_C(0xdb9cc81b4939caef),
                              UINT64_C(0x99908ab055e14bf0), UINT64_C(0xd01deeb18277fd8f),
                              UINT64_C(0xe1f43dbe1a24fb3a), UINT64_C(0xdaa3b7846091d1be)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x475003e212ada19b), UINT64_C(0x490bdb33ee5d5470),
                              UINT64_C(0x61249881556eac3a), UINT64_C(0xbf42ccd4a27e5259),
                              UINT64_C(0x613173560a9ec8e6), UINT64_C(0x8736f836c78d1256),
                              UINT64_C(0xa4248bd9dac1f2cc), UINT64_C(0x7008605d8072d787)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x93c67e752cd98dff), UINT64_C(0x823553380fc55d49),
                              UINT64_C(0xa53cada7d3919972), UINT64_C(0x64de04cfeb4798b6),
                              UINT64_C(0xf8a1f9e65f7f8316), UINT64_C(0x572b168745faefd9),
                              UINT64_C(0x45d0b667c0e509f6), UINT64_C(0xaaabd7d9e0e30639)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(0xc6500379d74d1915), UINT64_C(0x2deb735fa56e277e),
                              UINT64_C(0xc2e0f463b67c41f4), UINT64_C(0x8f539a5e01d0c88f),
                              UINT64_C(0x68e4935ea747c9c2), UINT64_C(0xdc21f9b373f8b465),
                              UINT64_C(0xf3592239b25cb40f), UINT64_C(0xf4139e2d72ff74c8)),
      easysimd_x_mm512_set_epu64(UINT64_C(0x7e1193710ce44a9c), UINT64_C(0x253a368d6b9cc286),
                              UINT64_C(0x310c01bdff0560df), UINT64_C(0xe73fc91eec559d39),
                              UINT64_C(0xc18711aa058fbe1a), UINT64_C(0x80fe26999b91720a),
                              UINT64_C(0x12959cadf8f60c1b), UINT64_C(0xae1ad9214abbd4ef)),
      easysimd_x_mm512_set_epu64(UINT64_C(0xb8419008dba95389), UINT64_C(0x08d145d2cef2e5f8),
                              UINT64_C(0xf3ecf5de4979212b), UINT64_C(0x686c5340ed8555b6),
                              UINT64_C(0xa96382f4a2c877d8), UINT64_C(0x5cdfdf2ae869c66f),
                              UINT64_C(0xe1ccbe944aaab814), UINT64_C(0x5a09470c3844a027)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_xor_si512(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_xor_si512");
    easysimd_assert_m512i_u64(r, ==, test_vec[i].r);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_xor_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_xor_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_xor_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_xor_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_xor_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_xor_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_xor_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_xor_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_xor_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_xor_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_xor_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_xor_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_xor_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_xor_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_xor_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_xor_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_xor_si512)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
