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

#define EASYSIMD_TEST_X86_AVX512_INSN mul

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/mul.h>

static int
test_easysimd_mm_mask_mul_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 1061656836753002459), -INT64_C( 4351609322920464480) },
      UINT8_C(203),
      {  INT32_C(   737873630), -INT32_C(  1696665884),  INT32_C(   907954941),  INT32_C(  1180425302) },
      {  INT32_C(   268547796),  INT32_C(   699527930), -INT32_C(   108454913), -INT32_C(  1799074634) },
      {  INT64_C(  198154337063019480), -INT64_C(   98472174134075133) } },
    { { -INT64_C( 6063922427040120742), -INT64_C( 6375622823777274136) },
      UINT8_C(242),
      {  INT32_C(  2045556614), -INT32_C(  2072504984), -INT32_C(   365268578), -INT32_C(   163262666) },
      {  INT32_C(   748978830),  INT32_C(  1360351322),  INT32_C(  1847808817),  INT32_C(  1600178904) },
      { -INT64_C( 6063922427040120742), -INT64_C(  674946499001452226) } },
    { {  INT64_C(  103108227253161346),  INT64_C(  499108429821618625) },
      UINT8_C(178),
      { -INT32_C(   502451310), -INT32_C(  1693229496),  INT32_C(  1282703745),  INT32_C(   802083810) },
      { -INT32_C(    74336085), -INT32_C(   792948185), -INT32_C(  1370966914),  INT32_C(   861949601) },
      {  INT64_C(  103108227253161346), -INT64_C( 1758544394858892930) } },
    { {  INT64_C( 5576909668796951937),  INT64_C( 6321760492735551914) },
      UINT8_C( 31),
      { -INT32_C(   549039539), -INT32_C(  1034086898), -INT32_C(  1385952418), -INT32_C(   651258004) },
      { -INT32_C(   693765971),  INT32_C(   914420317), -INT32_C(   220066421),  INT32_C(   353450440) },
      {  INT64_C(  380904948891727369),  INT64_C(  305001588305555978) } },
    { { -INT64_C( 3648950963094071157),  INT64_C(  334295892368212062) },
      UINT8_C(160),
      {  INT32_C(   989715016),  INT32_C(  1757754203),  INT32_C(  1311815445),  INT32_C(   567887561) },
      { -INT32_C(  1919978693), -INT32_C(  1880400689),  INT32_C(  2145826262), -INT32_C(  1591743655) },
      { -INT64_C( 3648950963094071157),  INT64_C(  334295892368212062) } },
    { {  INT64_C( 7388332554222050757),  INT64_C( 4198356504089508185) },
      UINT8_C(  1),
      {  INT32_C(   802211794),  INT32_C(  1745182643),  INT32_C(   817988677), -INT32_C(  1040817244) },
      { -INT32_C(   535751105),  INT32_C(  1480161183), -INT32_C(  1168680148),  INT32_C(  1908117918) },
      { -INT64_C(  429785855079532370),  INT64_C( 4198356504089508185) } },
    { {  INT64_C( 3419540441717050200), -INT64_C( 2914015219587992278) },
      UINT8_C(108),
      {  INT32_C(   403421344),  INT32_C(  1296327665),  INT32_C(  1273757370),  INT32_C(  1151622329) },
      { -INT32_C(  1573998596),  INT32_C(  1506565411),  INT32_C(   334600893), -INT32_C(   914306775) },
      {  INT64_C( 3419540441717050200), -INT64_C( 2914015219587992278) } },
    { { -INT64_C( 6215207250393199743),  INT64_C(  297405834457361187) },
      UINT8_C( 72),
      { -INT32_C(  1402165682),  INT32_C(   208258418),  INT32_C(  2134212022), -INT32_C(  2013200643) },
      {  INT32_C(    91714528), -INT32_C(   584507086), -INT32_C(  1343945450),  INT32_C(  1962404134) },
      { -INT64_C( 6215207250393199743),  INT64_C(  297405834457361187) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mul_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mul_epi32");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_mul_epi32(src, k, a, b);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_mul_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 45),
      {  INT32_C(   458550084),  INT32_C(   655944901),  INT32_C(  1509737242),  INT32_C(  1613464787) },
      {  INT32_C(   869624023), -INT32_C(  1566814905),  INT32_C(  2054334742),  INT32_C(  1336379403) },
      {  INT64_C(  398766168795067932),  INT64_C(                   0) } },
    { UINT8_C(115),
      { -INT32_C(   398955524),  INT32_C(  1191337859), -INT32_C(   551920805), -INT32_C(   289965434) },
      { -INT32_C(  1674188209),  INT32_C(   800249734), -INT32_C(   767939510), -INT32_C(   817526317) },
      {  INT64_C(  667926634196216516),  INT64_C(  423841792550505550) } },
    { UINT8_C(244),
      { -INT32_C(   579291267),  INT32_C(   339263417),  INT32_C(  1402607833), -INT32_C(  1180530225) },
      { -INT32_C(  1774240066),  INT32_C(   484470512),  INT32_C(   837792424),  INT32_C(  1948696567) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(118),
      {  INT32_C(  1563447710), -INT32_C(  1573501814),  INT32_C(  1752271327),  INT32_C(  1764108843) },
      { -INT32_C(   681984919),  INT32_C(  1316976028),  INT32_C(   591769700),  INT32_C(  1956297430) },
      {  INT64_C(                   0),  INT64_C( 1036941077497391900) } },
    { UINT8_C( 11),
      {  INT32_C(   244699594), -INT32_C(  1863502073), -INT32_C(   759474776),  INT32_C(   943400060) },
      { -INT32_C(   220917123),  INT32_C(  1113006993),  INT32_C(   588806761), -INT32_C(   567374572) },
      { -INT64_C(   54058330305748062), -INT64_C(  447183882917760536) } },
    { UINT8_C( 94),
      { -INT32_C(    77140540),  INT32_C(   816117466), -INT32_C(   693340494), -INT32_C(  1001069390) },
      { -INT32_C(   581548359),  INT32_C(   407279773),  INT32_C(  1026320816),  INT32_C(  1536953239) },
      {  INT64_C(                   0), -INT64_C(  711589781567923104) } },
    { UINT8_C(248),
      { -INT32_C(   120432894),  INT32_C(  1906967291), -INT32_C(  1826389586),  INT32_C(   458025173) },
      { -INT32_C(   692508354), -INT32_C(   662253201),  INT32_C(   141608189),  INT32_C(  1644219232) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 34),
      { -INT32_C(   719496493), -INT32_C(  2038132987), -INT32_C(  1705306190), -INT32_C(  1915193757) },
      {  INT32_C(    16559663), -INT32_C(   100739787),  INT32_C(   274269765), -INT32_C(   634143993) },
      {  INT64_C(                   0), -INT64_C(  467713927984345350) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mul_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mul_epi32");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_mul_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_mul_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t src[2];
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint64_t r[2];
  } test_vec[] = {
    { { UINT64_C(16072654549577393039), UINT64_C( 2345354234509017607) },
      UINT8_C(210),
      { UINT32_C( 621028868), UINT32_C(1413210703), UINT32_C(2960845511), UINT32_C(1161818081) },
      { UINT32_C(2574073871), UINT32_C(2409647181), UINT32_C( 181380267), UINT32_C( 903671857) },
      { UINT64_C(16072654549577393039), UINT64_C(  537038949330931437) } },
    { { UINT64_C( 9924127916044247062), UINT64_C( 3649455166387205247) },
      UINT8_C(184),
      { UINT32_C(1594215443), UINT32_C( 285906028), UINT32_C(1413616740), UINT32_C(3513415664) },
      { UINT32_C(1720963025), UINT32_C(3169131912), UINT32_C(3487515990), UINT32_C(4270265067) },
      { UINT64_C( 9924127916044247062), UINT64_C( 3649455166387205247) } },
    { { UINT64_C( 9609388722115939806), UINT64_C(16518714238262943100) },
      UINT8_C( 18),
      { UINT32_C(4003154897), UINT32_C(1984255792), UINT32_C(1231098935), UINT32_C( 673669019) },
      { UINT32_C( 608858556), UINT32_C(1788989389), UINT32_C(3967749801), UINT32_C( 503276620) },
      { UINT64_C( 9609388722115939806), UINT64_C( 4884692554357561935) } },
    { { UINT64_C( 2906600708802124464), UINT64_C(15215860036312086373) },
      UINT8_C( 10),
      { UINT32_C(1121449843), UINT32_C(2800435608), UINT32_C( 636672192), UINT32_C(1909788887) },
      { UINT32_C(1835185436), UINT32_C(3285355276), UINT32_C(2514146044), UINT32_C(1855958523) },
      { UINT64_C( 2906600708802124464), UINT64_C( 1600686872841608448) } },
    { { UINT64_C( 8710978415092659876), UINT64_C(  197158817134138996) },
      UINT8_C( 39),
      { UINT32_C(2855497503), UINT32_C( 329709377), UINT32_C(2114927569), UINT32_C(1344437722) },
      { UINT32_C(3406323502), UINT32_C( 423592003), UINT32_C(2415889181), UINT32_C(1723203911) },
      { UINT64_C( 9726748254371215506), UINT64_C( 5109430632545730989) } },
    { { UINT64_C(12881623014152661616), UINT64_C( 9159567728429814770) },
      UINT8_C(178),
      { UINT32_C(2784315941), UINT32_C( 365039242), UINT32_C( 240931085), UINT32_C(4068393736) },
      { UINT32_C(2329096147), UINT32_C(3346892275), UINT32_C( 119032501), UINT32_C(2377750375) },
      { UINT64_C(12881623014152661616), UINT64_C(   28678629616193585) } },
    { { UINT64_C(14520156863237107426), UINT64_C( 8376789453915020614) },
      UINT8_C(132),
      { UINT32_C(2558066195), UINT32_C(3310174075), UINT32_C(3995948375), UINT32_C(3184572942) },
      { UINT32_C(3782819052), UINT32_C(2603107261), UINT32_C(1882945050), UINT32_C(3405033399) },
      { UINT64_C(14520156863237107426), UINT64_C( 8376789453915020614) } },
    { { UINT64_C(  211581956689980590), UINT64_C(14038214638709054469) },
      UINT8_C(254),
      { UINT32_C(2377884746), UINT32_C(1118328524), UINT32_C(1106843794), UINT32_C(2045756428) },
      { UINT32_C(3609467175), UINT32_C( 618473224), UINT32_C(3657101592), UINT32_C( 215465154) },
      { UINT64_C(  211581956689980590), UINT64_C( 4047840201132720048) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mul_epu32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mul_epu32");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__m128i r = easysimd_mm_mask_mul_epu32(src, k, a, b);

    easysimd_test_x86_write_u64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_mul_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint64_t r[2];
  } test_vec[] = {
    { UINT8_C( 21),
      { UINT32_C(3743067984), UINT32_C(1251123251), UINT32_C(1924017403), UINT32_C(3013688069) },
      { UINT32_C( 216572471), UINT32_C(2790384032), UINT32_C(4121780794), UINT32_C(3842738069) },
      { UINT64_C(  810645482415868464), UINT64_C(                   0) } },
    { UINT8_C( 55),
      { UINT32_C(3328886053), UINT32_C(2210510167), UINT32_C(2659726179), UINT32_C(2010463445) },
      { UINT32_C( 102228516), UINT32_C(2504047923), UINT32_C(4062852715), UINT32_C(1713967169) },
      { UINT64_C(  340307081131287348), UINT64_C(10806075727506725985) } },
    { UINT8_C(213),
      { UINT32_C(1227631764), UINT32_C( 564965358), UINT32_C(1962297912), UINT32_C(  26766623) },
      { UINT32_C(1110810244), UINT32_C( 363711199), UINT32_C(  89563125), UINT32_C(1574616265) },
      { UINT64_C( 1363665939310990416), UINT64_C(                   0) } },
    { UINT8_C(233),
      { UINT32_C(3050808838), UINT32_C(2632841298), UINT32_C(1555784175), UINT32_C(2564931066) },
      { UINT32_C(3161924594), UINT32_C(1873906896), UINT32_C(2671294178), UINT32_C(2525533584) },
      { UINT64_C( 9646427496464761772), UINT64_C(                   0) } },
    { UINT8_C( 59),
      { UINT32_C(1468877919), UINT32_C(2605066553), UINT32_C(2727715813), UINT32_C(2811506052) },
      { UINT32_C( 829903013), UINT32_C(3088377346), UINT32_C(3024728863), UINT32_C(2599477051) },
      { UINT64_C( 1219026210707269947), UINT64_C( 8250600749642610619) } },
    { UINT8_C( 43),
      { UINT32_C(2808476029), UINT32_C(3683385400), UINT32_C(3261017749), UINT32_C( 308741826) },
      { UINT32_C(1662294397), UINT32_C(1619185069), UINT32_C(4120590102), UINT32_C(2753639719) },
      { UINT64_C( 4668513967115509513), UINT64_C(13437317458975720398) } },
    { UINT8_C( 38),
      { UINT32_C(2254326662), UINT32_C(  85670359), UINT32_C(2663898520), UINT32_C(3726367301) },
      { UINT32_C(3146481646), UINT32_C( 953281281), UINT32_C(3143616390), UINT32_C(1860240616) },
      { UINT64_C(                   0), UINT64_C( 8374275048768742800) } },
    { UINT8_C( 79),
      { UINT32_C(2032661568), UINT32_C(3960613903), UINT32_C(3459363060), UINT32_C(1270616012) },
      { UINT32_C(2236381338), UINT32_C( 269190473), UINT32_C(3908617956), UINT32_C(3896010408) },
      { UINT64_C( 4545806397145017984), UINT64_C(13521328572639105360) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mul_epu32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mul_epu32");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__m128i r = easysimd_mm_maskz_mul_epu32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_mul_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -855.41), EASYSIMD_FLOAT32_C(  -582.32), EASYSIMD_FLOAT32_C(   849.83), EASYSIMD_FLOAT32_C(  -865.33) },
      UINT8_C(106),
      { EASYSIMD_FLOAT32_C(   635.14), EASYSIMD_FLOAT32_C(   714.93), EASYSIMD_FLOAT32_C(   354.28), EASYSIMD_FLOAT32_C(  -133.58) },
      { EASYSIMD_FLOAT32_C(   680.98), EASYSIMD_FLOAT32_C(    65.48), EASYSIMD_FLOAT32_C(  -233.57), EASYSIMD_FLOAT32_C(   182.25) },
      { EASYSIMD_FLOAT32_C(  -855.41), EASYSIMD_FLOAT32_C( 46813.62), EASYSIMD_FLOAT32_C(   849.83), EASYSIMD_FLOAT32_C(-24344.96) } },
    { { EASYSIMD_FLOAT32_C(  -247.41), EASYSIMD_FLOAT32_C(    67.32), EASYSIMD_FLOAT32_C(  -467.12), EASYSIMD_FLOAT32_C(   471.37) },
      UINT8_C(210),
      { EASYSIMD_FLOAT32_C(   739.84), EASYSIMD_FLOAT32_C(  -947.87), EASYSIMD_FLOAT32_C(   284.34), EASYSIMD_FLOAT32_C(  -998.00) },
      { EASYSIMD_FLOAT32_C(  -310.25), EASYSIMD_FLOAT32_C(   162.66), EASYSIMD_FLOAT32_C(  -531.68), EASYSIMD_FLOAT32_C(  -743.04) },
      { EASYSIMD_FLOAT32_C(  -247.41), EASYSIMD_FLOAT32_C(-154180.53), EASYSIMD_FLOAT32_C(  -467.12), EASYSIMD_FLOAT32_C(   471.37) } },
    { { EASYSIMD_FLOAT32_C(  -606.00), EASYSIMD_FLOAT32_C(   923.59), EASYSIMD_FLOAT32_C(   337.60), EASYSIMD_FLOAT32_C(  -193.78) },
      UINT8_C( 40),
      { EASYSIMD_FLOAT32_C(   482.20), EASYSIMD_FLOAT32_C(   223.90), EASYSIMD_FLOAT32_C(  -436.91), EASYSIMD_FLOAT32_C(   616.87) },
      { EASYSIMD_FLOAT32_C(  -400.69), EASYSIMD_FLOAT32_C(  -801.77), EASYSIMD_FLOAT32_C(   331.80), EASYSIMD_FLOAT32_C(   953.60) },
      { EASYSIMD_FLOAT32_C(  -606.00), EASYSIMD_FLOAT32_C(   923.59), EASYSIMD_FLOAT32_C(   337.60), EASYSIMD_FLOAT32_C(588247.19) } },
    { { EASYSIMD_FLOAT32_C(    64.65), EASYSIMD_FLOAT32_C(    12.78), EASYSIMD_FLOAT32_C(    19.08), EASYSIMD_FLOAT32_C(   831.08) },
      UINT8_C( 36),
      { EASYSIMD_FLOAT32_C(   771.67), EASYSIMD_FLOAT32_C(  -101.60), EASYSIMD_FLOAT32_C(  -272.09), EASYSIMD_FLOAT32_C(   243.04) },
      { EASYSIMD_FLOAT32_C(   599.67), EASYSIMD_FLOAT32_C(  -532.25), EASYSIMD_FLOAT32_C(   295.17), EASYSIMD_FLOAT32_C(  -115.98) },
      { EASYSIMD_FLOAT32_C(    64.65), EASYSIMD_FLOAT32_C(    12.78), EASYSIMD_FLOAT32_C(-80312.80), EASYSIMD_FLOAT32_C(   831.08) } },
    { { EASYSIMD_FLOAT32_C(  -530.25), EASYSIMD_FLOAT32_C(   984.92), EASYSIMD_FLOAT32_C(  -953.32), EASYSIMD_FLOAT32_C(   -61.93) },
      UINT8_C(142),
      { EASYSIMD_FLOAT32_C(  -559.32), EASYSIMD_FLOAT32_C(  -138.34), EASYSIMD_FLOAT32_C(   579.48), EASYSIMD_FLOAT32_C(   246.90) },
      { EASYSIMD_FLOAT32_C(   574.92), EASYSIMD_FLOAT32_C(    61.68), EASYSIMD_FLOAT32_C(  -529.19), EASYSIMD_FLOAT32_C(  -862.00) },
      { EASYSIMD_FLOAT32_C(  -530.25), EASYSIMD_FLOAT32_C( -8532.81), EASYSIMD_FLOAT32_C(-306655.00), EASYSIMD_FLOAT32_C(-212827.80) } },
    { { EASYSIMD_FLOAT32_C(  -321.45), EASYSIMD_FLOAT32_C(    70.12), EASYSIMD_FLOAT32_C(  -663.77), EASYSIMD_FLOAT32_C(  -989.65) },
      UINT8_C(196),
      { EASYSIMD_FLOAT32_C(   400.88), EASYSIMD_FLOAT32_C(    23.13), EASYSIMD_FLOAT32_C(  -957.21), EASYSIMD_FLOAT32_C(   231.96) },
      { EASYSIMD_FLOAT32_C(   218.16), EASYSIMD_FLOAT32_C(   814.47), EASYSIMD_FLOAT32_C(  -869.64), EASYSIMD_FLOAT32_C(   946.08) },
      { EASYSIMD_FLOAT32_C(  -321.45), EASYSIMD_FLOAT32_C(    70.12), EASYSIMD_FLOAT32_C(832428.12), EASYSIMD_FLOAT32_C(  -989.65) } },
    { { EASYSIMD_FLOAT32_C(    57.51), EASYSIMD_FLOAT32_C(   730.03), EASYSIMD_FLOAT32_C(  -586.17), EASYSIMD_FLOAT32_C(  -647.33) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT32_C(  -116.42), EASYSIMD_FLOAT32_C(  -662.41), EASYSIMD_FLOAT32_C(  -339.27), EASYSIMD_FLOAT32_C(   821.65) },
      { EASYSIMD_FLOAT32_C(  -420.53), EASYSIMD_FLOAT32_C(   101.41), EASYSIMD_FLOAT32_C(  -316.69), EASYSIMD_FLOAT32_C(  -841.05) },
      { EASYSIMD_FLOAT32_C( 48958.10), EASYSIMD_FLOAT32_C(   730.03), EASYSIMD_FLOAT32_C(  -586.17), EASYSIMD_FLOAT32_C(  -647.33) } },
    { { EASYSIMD_FLOAT32_C(  -651.68), EASYSIMD_FLOAT32_C(  -741.78), EASYSIMD_FLOAT32_C(   220.62), EASYSIMD_FLOAT32_C(  -180.87) },
      UINT8_C(221),
      { EASYSIMD_FLOAT32_C(   899.17), EASYSIMD_FLOAT32_C(   889.25), EASYSIMD_FLOAT32_C(  -267.55), EASYSIMD_FLOAT32_C(   909.52) },
      { EASYSIMD_FLOAT32_C(   -87.04), EASYSIMD_FLOAT32_C(  -866.67), EASYSIMD_FLOAT32_C(   -67.35), EASYSIMD_FLOAT32_C(   -44.24) },
      { EASYSIMD_FLOAT32_C(-78263.76), EASYSIMD_FLOAT32_C(  -741.78), EASYSIMD_FLOAT32_C( 18019.49), EASYSIMD_FLOAT32_C(-40237.17) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mul_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mul_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_mul_ps(src, k, a, b);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_mul_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C( 98),
      { EASYSIMD_FLOAT32_C(  -451.93), EASYSIMD_FLOAT32_C(  -399.92), EASYSIMD_FLOAT32_C(  -145.79), EASYSIMD_FLOAT32_C(   232.75) },
      { EASYSIMD_FLOAT32_C(  -378.94), EASYSIMD_FLOAT32_C(   780.18), EASYSIMD_FLOAT32_C(  -987.24), EASYSIMD_FLOAT32_C(   273.78) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-312009.59), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(  6),
      { EASYSIMD_FLOAT32_C(  -259.86), EASYSIMD_FLOAT32_C(   301.59), EASYSIMD_FLOAT32_C(   102.41), EASYSIMD_FLOAT32_C(  -258.01) },
      { EASYSIMD_FLOAT32_C(    25.98), EASYSIMD_FLOAT32_C(  -454.57), EASYSIMD_FLOAT32_C(  -164.41), EASYSIMD_FLOAT32_C(  -734.50) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-137093.77), EASYSIMD_FLOAT32_C(-16837.23), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 94),
      { EASYSIMD_FLOAT32_C(  -332.11), EASYSIMD_FLOAT32_C(   328.55), EASYSIMD_FLOAT32_C(   332.72), EASYSIMD_FLOAT32_C(   381.53) },
      { EASYSIMD_FLOAT32_C(   446.60), EASYSIMD_FLOAT32_C(    20.68), EASYSIMD_FLOAT32_C(   424.61), EASYSIMD_FLOAT32_C(   894.05) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  6794.41), EASYSIMD_FLOAT32_C(141276.23), EASYSIMD_FLOAT32_C(341106.91) } },
    { UINT8_C( 70),
      { EASYSIMD_FLOAT32_C(   650.30), EASYSIMD_FLOAT32_C(   202.27), EASYSIMD_FLOAT32_C(  -758.22), EASYSIMD_FLOAT32_C(  -936.59) },
      { EASYSIMD_FLOAT32_C(   750.34), EASYSIMD_FLOAT32_C(  -158.14), EASYSIMD_FLOAT32_C(   -82.38), EASYSIMD_FLOAT32_C(   -16.91) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-31986.98), EASYSIMD_FLOAT32_C( 62462.16), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(231),
      { EASYSIMD_FLOAT32_C(  -302.20), EASYSIMD_FLOAT32_C(    -4.15), EASYSIMD_FLOAT32_C(  -263.30), EASYSIMD_FLOAT32_C(   806.48) },
      { EASYSIMD_FLOAT32_C(   735.99), EASYSIMD_FLOAT32_C(  -961.71), EASYSIMD_FLOAT32_C(   -91.11), EASYSIMD_FLOAT32_C(  -522.02) },
      { EASYSIMD_FLOAT32_C(-222416.19), EASYSIMD_FLOAT32_C(  3991.10), EASYSIMD_FLOAT32_C( 23989.26), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(147),
      { EASYSIMD_FLOAT32_C(   454.32), EASYSIMD_FLOAT32_C(   313.57), EASYSIMD_FLOAT32_C(   329.77), EASYSIMD_FLOAT32_C(  -330.61) },
      { EASYSIMD_FLOAT32_C(   981.46), EASYSIMD_FLOAT32_C(  -341.68), EASYSIMD_FLOAT32_C(  -997.89), EASYSIMD_FLOAT32_C(   362.99) },
      { EASYSIMD_FLOAT32_C(445896.94), EASYSIMD_FLOAT32_C(-107140.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(137),
      { EASYSIMD_FLOAT32_C(    22.79), EASYSIMD_FLOAT32_C(  -212.40), EASYSIMD_FLOAT32_C(   998.97), EASYSIMD_FLOAT32_C(   843.58) },
      { EASYSIMD_FLOAT32_C(  -562.10), EASYSIMD_FLOAT32_C(   201.23), EASYSIMD_FLOAT32_C(  -914.64), EASYSIMD_FLOAT32_C(  -498.69) },
      { EASYSIMD_FLOAT32_C(-12810.26), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-420684.91) } },
    { UINT8_C( 24),
      { EASYSIMD_FLOAT32_C(   -72.78), EASYSIMD_FLOAT32_C(   418.93), EASYSIMD_FLOAT32_C(   934.66), EASYSIMD_FLOAT32_C(  -609.86) },
      { EASYSIMD_FLOAT32_C(  -883.27), EASYSIMD_FLOAT32_C(   -69.49), EASYSIMD_FLOAT32_C(   126.84), EASYSIMD_FLOAT32_C(   923.21) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-563028.88) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mul_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mul_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_mul_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_mul_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   285.13), EASYSIMD_FLOAT64_C(  -729.47) },
      UINT8_C(131),
      { EASYSIMD_FLOAT64_C(   491.92), EASYSIMD_FLOAT64_C(   866.17) },
      { EASYSIMD_FLOAT64_C(  -236.12), EASYSIMD_FLOAT64_C(    96.02) },
      { EASYSIMD_FLOAT64_C(-116152.15), EASYSIMD_FLOAT64_C( 83169.64) } },
    { { EASYSIMD_FLOAT64_C(  -258.88), EASYSIMD_FLOAT64_C(    -1.77) },
      UINT8_C(130),
      { EASYSIMD_FLOAT64_C(   741.79), EASYSIMD_FLOAT64_C(    78.28) },
      { EASYSIMD_FLOAT64_C(    10.33), EASYSIMD_FLOAT64_C(   -73.65) },
      { EASYSIMD_FLOAT64_C(  -258.88), EASYSIMD_FLOAT64_C( -5765.32) } },
    { { EASYSIMD_FLOAT64_C(   801.19), EASYSIMD_FLOAT64_C(   394.94) },
      UINT8_C(216),
      { EASYSIMD_FLOAT64_C(   183.62), EASYSIMD_FLOAT64_C(   -95.51) },
      { EASYSIMD_FLOAT64_C(  -624.97), EASYSIMD_FLOAT64_C(  -110.15) },
      { EASYSIMD_FLOAT64_C(   801.19), EASYSIMD_FLOAT64_C(   394.94) } },
    { { EASYSIMD_FLOAT64_C(   936.88), EASYSIMD_FLOAT64_C(  -777.01) },
      UINT8_C( 39),
      { EASYSIMD_FLOAT64_C(    96.45), EASYSIMD_FLOAT64_C(   764.98) },
      { EASYSIMD_FLOAT64_C(   341.99), EASYSIMD_FLOAT64_C(   906.24) },
      { EASYSIMD_FLOAT64_C( 32984.94), EASYSIMD_FLOAT64_C(693255.48) } },
    { { EASYSIMD_FLOAT64_C(    33.73), EASYSIMD_FLOAT64_C(   900.04) },
      UINT8_C(150),
      { EASYSIMD_FLOAT64_C(  -681.14), EASYSIMD_FLOAT64_C(  -829.43) },
      { EASYSIMD_FLOAT64_C(   -48.42), EASYSIMD_FLOAT64_C(   810.79) },
      { EASYSIMD_FLOAT64_C(    33.73), EASYSIMD_FLOAT64_C(-672493.55) } },
    { { EASYSIMD_FLOAT64_C(  -963.25), EASYSIMD_FLOAT64_C(   715.46) },
      UINT8_C(172),
      { EASYSIMD_FLOAT64_C(  -222.13), EASYSIMD_FLOAT64_C(  -286.31) },
      { EASYSIMD_FLOAT64_C(   485.26), EASYSIMD_FLOAT64_C(  -480.34) },
      { EASYSIMD_FLOAT64_C(  -963.25), EASYSIMD_FLOAT64_C(   715.46) } },
    { { EASYSIMD_FLOAT64_C(   791.97), EASYSIMD_FLOAT64_C(  -504.42) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT64_C(   593.16), EASYSIMD_FLOAT64_C(   890.52) },
      { EASYSIMD_FLOAT64_C(   635.94), EASYSIMD_FLOAT64_C(  -223.22) },
      { EASYSIMD_FLOAT64_C(377214.17), EASYSIMD_FLOAT64_C(  -504.42) } },
    { { EASYSIMD_FLOAT64_C(  -204.99), EASYSIMD_FLOAT64_C(  -989.03) },
      UINT8_C( 27),
      { EASYSIMD_FLOAT64_C(  -268.10), EASYSIMD_FLOAT64_C(  -766.04) },
      { EASYSIMD_FLOAT64_C(   430.31), EASYSIMD_FLOAT64_C(   828.35) },
      { EASYSIMD_FLOAT64_C(-115366.11), EASYSIMD_FLOAT64_C(-634549.23) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mul_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mul_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_mul_pd(src, k, a, b);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_mul_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C( 42),
      { EASYSIMD_FLOAT64_C(   383.98), EASYSIMD_FLOAT64_C(  -938.50) },
      { EASYSIMD_FLOAT64_C(   802.84), EASYSIMD_FLOAT64_C(   -36.54) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C( 34292.79) } },
    { UINT8_C( 47),
      { EASYSIMD_FLOAT64_C(  -441.83), EASYSIMD_FLOAT64_C(  -163.23) },
      { EASYSIMD_FLOAT64_C(   712.17), EASYSIMD_FLOAT64_C(  -796.39) },
      { EASYSIMD_FLOAT64_C(-314658.07), EASYSIMD_FLOAT64_C(129994.74) } },
    { UINT8_C(167),
      { EASYSIMD_FLOAT64_C(  -898.94), EASYSIMD_FLOAT64_C(   742.69) },
      { EASYSIMD_FLOAT64_C(    64.25), EASYSIMD_FLOAT64_C(   691.45) },
      { EASYSIMD_FLOAT64_C(-57756.90), EASYSIMD_FLOAT64_C(513533.00) } },
    { UINT8_C( 66),
      { EASYSIMD_FLOAT64_C(    37.03), EASYSIMD_FLOAT64_C(  -775.30) },
      { EASYSIMD_FLOAT64_C(  -740.08), EASYSIMD_FLOAT64_C(  -941.99) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(730324.85) } },
    { UINT8_C( 79),
      { EASYSIMD_FLOAT64_C(  -605.41), EASYSIMD_FLOAT64_C(  -979.61) },
      { EASYSIMD_FLOAT64_C(  -337.66), EASYSIMD_FLOAT64_C(  -266.20) },
      { EASYSIMD_FLOAT64_C(204422.74), EASYSIMD_FLOAT64_C(260772.18) } },
    { UINT8_C(242),
      { EASYSIMD_FLOAT64_C(  -540.68), EASYSIMD_FLOAT64_C(  -485.83) },
      { EASYSIMD_FLOAT64_C(  -395.20), EASYSIMD_FLOAT64_C(   167.33) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-81293.93) } },
    { UINT8_C(191),
      { EASYSIMD_FLOAT64_C(  -740.61), EASYSIMD_FLOAT64_C(  -448.69) },
      { EASYSIMD_FLOAT64_C(   278.98), EASYSIMD_FLOAT64_C(  -937.77) },
      { EASYSIMD_FLOAT64_C(-206615.38), EASYSIMD_FLOAT64_C(420768.02) } },
    { UINT8_C(209),
      { EASYSIMD_FLOAT64_C(   117.24), EASYSIMD_FLOAT64_C(  -379.60) },
      { EASYSIMD_FLOAT64_C(  -648.46), EASYSIMD_FLOAT64_C(  -170.59) },
      { EASYSIMD_FLOAT64_C(-76025.45), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mul_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mul_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_mul_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mul_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[4];
    uint8_t k;
    int32_t a[8];
    int32_t b[8];
    int64_t r[4];
  } test_vec[8] = {
    { {  INT64_C( 2451831441043393542),  INT64_C(  759383498199889318),  INT64_C( 4724431898911023843), -INT64_C( 4593565269602228771) },
      UINT8_C(237),
      {  INT32_C(  1839287494), -INT32_C(   955006126),  INT32_C(   554417730), -INT32_C(   385542585), -INT32_C(  1062145741),  INT32_C(   664662401),  INT32_C(   467679930),  INT32_C(  1074307194) },
      {  INT32_C(  1051568620), -INT32_C(  1358577300), -INT32_C(   959442561),  INT32_C(  1521472807),  INT32_C(  1209753798),  INT32_C(   208648273),  INT32_C(  1814515698), -INT32_C(   592695312) },
      {  INT64_C( 1934137011848838280),  INT64_C(  759383498199889318), -INT64_C( 1284934844204274318),  INT64_C(  848612574624541140) } },
    { { -INT64_C( 7281440871049110824),  INT64_C( 6967930199759438897),  INT64_C( 8776697116357086831), -INT64_C(  345772065897974682) },
      UINT8_C(236),
      {  INT32_C(  1845968718), -INT32_C(   140467917),  INT32_C(   277936129),  INT32_C(  2038493355), -INT32_C(  1358937704),  INT32_C(    35027214), -INT32_C(   265982882), -INT32_C(   304340833) },
      { -INT32_C(  1839406241), -INT32_C(  2054554492), -INT32_C(  1600775436), -INT32_C(  1558637046),  INT32_C(  1699879255), -INT32_C(   244881262),  INT32_C(  1960938709),  INT32_C(   191020716) },
      { -INT64_C( 7281440871049110824),  INT64_C( 6967930199759438897), -INT64_C( 2310030011866930520), -INT64_C(  521576129245179338) } },
    { { -INT64_C( 5860546809627230559), -INT64_C( 5841335467524603583),  INT64_C( 9152856688600236417),  INT64_C( 5097605994095306246) },
      UINT8_C( 19),
      { -INT32_C(  2083689380),  INT32_C(  1506114583),  INT32_C(   850399690),  INT32_C(  1102339840), -INT32_C(   303314830), -INT32_C(  1275893043), -INT32_C(  1286035874), -INT32_C(  1060659612) },
      {  INT32_C(   558076938), -INT32_C(   612759536),  INT32_C(   420293401), -INT32_C(    61095542),  INT32_C(  1458128521),  INT32_C(   252370353), -INT32_C(   406691197),  INT32_C(   178751744) },
      { -INT64_C( 1162858988933518440),  INT64_C(  357417377919445690),  INT64_C( 9152856688600236417),  INT64_C( 5097605994095306246) } },
    { {  INT64_C(  867407966627424798),  INT64_C( 7086275304932906961), -INT64_C( 6807258875617066553), -INT64_C( 1067863477742712284) },
      UINT8_C( 25),
      {  INT32_C(      794714),  INT32_C(  1087445289), -INT32_C(  1072092097), -INT32_C(   997753981), -INT32_C(   152960973), -INT32_C(   786791545),  INT32_C(   983908107), -INT32_C(   950757779) },
      { -INT32_C(   523804490), -INT32_C(  1205823367),  INT32_C(  1215838660), -INT32_C(   401866572), -INT32_C(  2015432960),  INT32_C(  2119825522), -INT32_C(  1514602953),  INT32_C(  1265372308) },
      { -INT64_C(     416274761465860),  INT64_C( 7086275304932906961), -INT64_C( 6807258875617066553), -INT64_C( 1490230124342839971) } },
    { { -INT64_C( 7953555073506003859),  INT64_C( 1666865225021331333),  INT64_C( 3442431235431858132), -INT64_C(  588492113011507722) },
      UINT8_C(117),
      {  INT32_C(  1279384832), -INT32_C(  1798188420), -INT32_C(  1851061589),  INT32_C(   728089388),  INT32_C(  1512287330), -INT32_C(   279948431),  INT32_C(  1752816167), -INT32_C(  1327664976) },
      { -INT32_C(   889446579), -INT32_C(  1654731279),  INT32_C(    70126040),  INT32_C(   774869964), -INT32_C(  1316465856), -INT32_C(   845096794),  INT32_C(  1664489907), -INT32_C(   686615671) },
      { -INT64_C( 1137944462046889728),  INT64_C( 1666865225021331333), -INT64_C( 1990874634406404480), -INT64_C(  588492113011507722) } },
    { { -INT64_C( 5349713557072048078), -INT64_C( 4466469657280647416),  INT64_C( 1706605207325281340),  INT64_C( 3138603193500099886) },
      UINT8_C(158),
      {  INT32_C(   779833135), -INT32_C(    13160176), -INT32_C(   746451734), -INT32_C(  1693498353), -INT32_C(   855576136), -INT32_C(  2030364767),  INT32_C(   545239698),  INT32_C(  1891543873) },
      {  INT32_C(   194984442),  INT32_C(  1426773610), -INT32_C(   249000734), -INT32_C(  1987233583), -INT32_C(   900297431),  INT32_C(   911298980),  INT32_C(  1163317252),  INT32_C(  1991578747) },
      { -INT64_C( 5349713557072048078),  INT64_C(  185867029661572756),  INT64_C(  770272997265706616),  INT64_C(  634286747158669896) } },
    { {  INT64_C(  868503318752744526), -INT64_C( 7533805918149003752),  INT64_C( 2230323167889639704), -INT64_C( 6019877431108482685) },
      UINT8_C(109),
      { -INT32_C(  2120719114), -INT32_C(  1466325902),  INT32_C(   756449954), -INT32_C(  1102729739), -INT32_C(  1026031089),  INT32_C(  1044838389),  INT32_C(  1939621210), -INT32_C(  1327413062) },
      {  INT32_C(   523336109), -INT32_C(  1060648163),  INT32_C(  1122884941), -INT32_C(  1711262838),  INT32_C(   710727989),  INT32_C(   711500496), -INT32_C(  1583479321), -INT32_C(    95322291) },
      { -INT64_C( 1109848889402687426), -INT64_C( 7533805918149003752), -INT64_C(  729229012536450021), -INT64_C( 3071350076607998410) } },
    { { -INT64_C( 7217616201825287176),  INT64_C( 3162335037733389245), -INT64_C( 3984913943933665609), -INT64_C( 4104261612386758462) },
      UINT8_C( 62),
      {  INT32_C(    76340260),  INT32_C(  1975658674), -INT32_C(   462747130),  INT32_C(   714905324), -INT32_C(  1475664917),  INT32_C(   627823829), -INT32_C(   101418178), -INT32_C(  1455899771) },
      {  INT32_C(  1252967576), -INT32_C(   222334740),  INT32_C(  1725377146), -INT32_C(  1416596544),  INT32_C(  1817418902), -INT32_C(  1366180240), -INT32_C(  1079540167), -INT32_C(   647438527) },
      { -INT64_C( 7217616201825287176), -INT64_C(  798413322479090980), -INT64_C( 2681901313174061134),  INT64_C(  109484996814955726) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mul_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mul_epi32");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_mul_epi32(src, k, a, b);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_mul_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[8];
    int32_t b[8];
    int64_t r[4];
  } test_vec[8] = {
    { UINT8_C(180),
      { -INT32_C(   652311273), -INT32_C(   681118161),  INT32_C(   352866108), -INT32_C(   291642326), -INT32_C(  1499848458),  INT32_C(  1337020789),  INT32_C(   812174475), -INT32_C(  1360769641) },
      {  INT32_C(    42402515),  INT32_C(   903474936),  INT32_C(  1749737789), -INT32_C(  1151932219),  INT32_C(  2103635976), -INT32_C(   859040959),  INT32_C(  2013017312),  INT32_C(  1428545666) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 3155135174796925008),  INT64_C(                   0) } },
    { UINT8_C(227),
      { -INT32_C(  1696901204),  INT32_C(   316149808),  INT32_C(  1138180187), -INT32_C(  2041867370),  INT32_C(   147376373), -INT32_C(   907504491),  INT32_C(  1900765073),  INT32_C(   827629700) },
      {  INT32_C(   684404983), -INT32_C(  1690655936),  INT32_C(  2044596707), -INT32_C(  1728042588), -INT32_C(  2019374862), -INT32_C(   313488804),  INT32_C(  1851759850),  INT32_C(   882881340) },
      { -INT64_C( 1161367639676299532),  INT64_C( 2327119462312844209),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(227),
      {  INT32_C(   237263979), -INT32_C(  1477263466), -INT32_C(   951293026),  INT32_C(   867820907),  INT32_C(   311443591),  INT32_C(   771521937), -INT32_C(  1872139556),  INT32_C(  1970511114) },
      { -INT32_C(  1887201287), -INT32_C(   180914857),  INT32_C(  1287422945), -INT32_C(   260016536),  INT32_C(  1191317686),  INT32_C(  1786052237),  INT32_C(  1929043560),  INT32_C(  1978101115) },
      { -INT64_C(  447764886527540973), -INT64_C( 1224716469090881570),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(  5),
      { -INT32_C(   530840469), -INT32_C(  1094627013), -INT32_C(  2094592499), -INT32_C(  1640360050),  INT32_C(   388727065), -INT32_C(   729836042), -INT32_C(    45092209),  INT32_C(  1141032153) },
      {  INT32_C(    69557961), -INT32_C(  1128012113), -INT32_C(  2109674764),  INT32_C(   438336001), -INT32_C(   248427269),  INT32_C(  1908781281),  INT32_C(  2087589282), -INT32_C(  1564446503) },
      { -INT64_C(   36924180639923709),  INT64_C(                   0), -INT64_C(   96570403144335485),  INT64_C(                   0) } },
    { UINT8_C(206),
      { -INT32_C(   864180251),  INT32_C(  1421883754), -INT32_C(   212515975), -INT32_C(  1343262877),  INT32_C(  1351737504), -INT32_C(  1141701978), -INT32_C(   527143056),  INT32_C(   363738927) },
      {  INT32_C(  1222716382), -INT32_C(   560160412),  INT32_C(  1221718500),  INT32_C(    16236640),  INT32_C(  1196460193), -INT32_C(   100514934), -INT32_C(   488991054), -INT32_C(  1409840947) },
      {  INT64_C(                   0), -INT64_C(  259634698203037500),  INT64_C( 1617300114921178272),  INT64_C(  257768238562221024) } },
    { UINT8_C(179),
      {  INT32_C(  2031678424), -INT32_C(  2141260145), -INT32_C(  1998543161),  INT32_C(   640278685),  INT32_C(  1957720113),  INT32_C(   136817522),  INT32_C(   248908165), -INT32_C(   658407168) },
      {  INT32_C(    72472949), -INT32_C(  1752846385), -INT32_C(   216046251),  INT32_C(  1998145606),  INT32_C(   720095672), -INT32_C(    97381772),  INT32_C(   453511195), -INT32_C(      800374) },
      {  INT64_C(  147241726806952376),  INT64_C(  431777757395739411),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(163),
      { -INT32_C(   193854651), -INT32_C(   297137784),  INT32_C(  1882471720),  INT32_C(   522758998), -INT32_C(  1449897322), -INT32_C(  1916498299),  INT32_C(  1595400342),  INT32_C(   402790099) },
      { -INT32_C(  1576176358), -INT32_C(  1483712642), -INT32_C(   367541100),  INT32_C(   101269615),  INT32_C(   397385106), -INT32_C(  1046121685),  INT32_C(   656522323), -INT32_C(   314629165) },
      {  INT64_C(  305549117794541058), -INT64_C(  691885726687692000),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(152),
      { -INT32_C(  1558737076), -INT32_C(   466108896),  INT32_C(   357835221), -INT32_C(   928490965),  INT32_C(  2096414473),  INT32_C(   567326052), -INT32_C(    84609066), -INT32_C(  2087526090) },
      { -INT32_C(  1876514448),  INT32_C(  1031102055), -INT32_C(  1437415041),  INT32_C(   745798179),  INT32_C(   514353082), -INT32_C(   213944292), -INT32_C(  1494404241), -INT32_C(  2060878060) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(  126440147057448906) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mul_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mul_epi32");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_mul_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mul_epu32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint64_t src[4];
    uint8_t k;
    uint32_t a[8];
    uint32_t b[8];
    uint64_t r[4];
  } test_vec[8] = {
    { { UINT64_C(   37122289520809636), UINT64_C( 7167469795253049868), UINT64_C(14101159149376097171), UINT64_C(15853491165216266471) },
      UINT8_C(  1),
      { UINT32_C( 143092773), UINT32_C(1226082543), UINT32_C(3843482094), UINT32_C( 595098200), UINT32_C(2863496009), UINT32_C(2240901164), UINT32_C(3703421682), UINT32_C(3789396412) },
      { UINT32_C(4125713669), UINT32_C(3678338797), UINT32_C( 935417311), UINT32_C( 408631759), UINT32_C(3787655349), UINT32_C(1785091192), UINT32_C(3393594126), UINT32_C(3266061500) },
      { UINT64_C(  590359809501214137), UINT64_C( 7167469795253049868), UINT64_C(14101159149376097171), UINT64_C(15853491165216266471) } },
    { { UINT64_C( 8237635207138088329), UINT64_C(   86137167279690314), UINT64_C( 6408421374899189005), UINT64_C(16351389754436236652) },
      UINT8_C(100),
      { UINT32_C(2549569698), UINT32_C(3168955050), UINT32_C( 420018964), UINT32_C( 572918061), UINT32_C( 896314348), UINT32_C(3483485338), UINT32_C(3039415014), UINT32_C(1461259189) },
      { UINT32_C( 250548324), UINT32_C(2395656570), UINT32_C(4205302732), UINT32_C(3357330908), UINT32_C( 318605432), UINT32_C( 870489932), UINT32_C( 535300969), UINT32_C(2054553878) },
      { UINT64_C( 8237635207138088329), UINT64_C(   86137167279690314), UINT64_C(  285570620052338336), UINT64_C(16351389754436236652) } },
    { { UINT64_C(  223582625259349265), UINT64_C(  633628274596889126), UINT64_C(15069603351464954018), UINT64_C( 2060509023176755463) },
      UINT8_C(204),
      { UINT32_C(1946331169), UINT32_C(2224686530), UINT32_C( 471112706), UINT32_C( 817765479), UINT32_C( 865578038), UINT32_C(3610929358), UINT32_C(3219282008), UINT32_C( 311164657) },
      { UINT32_C(1753648550), UINT32_C(2515345555), UINT32_C( 615580092), UINT32_C(1398042397), UINT32_C(3934710556), UINT32_C(2898379092), UINT32_C( 174826265), UINT32_C(1193080481) },
      { UINT64_C(  223582625259349265), UINT64_C(  633628274596889126), UINT64_C( 3405799043160369128), UINT64_C(  562815049440340120) } },
    { { UINT64_C( 9199901595757748868), UINT64_C(16793632523499560606), UINT64_C( 8062451955442882018), UINT64_C(14493030302617325112) },
      UINT8_C( 57),
      { UINT32_C(1862000849), UINT32_C(3926686348), UINT32_C( 347653917), UINT32_C(1811390934), UINT32_C( 180432245), UINT32_C(1598173200), UINT32_C(1101274282), UINT32_C( 259681598) },
      { UINT32_C(3648943437), UINT32_C( 230984432), UINT32_C( 673348690), UINT32_C(2459113757), UINT32_C(1469862982), UINT32_C( 800513924), UINT32_C( 963664635), UINT32_C( 340388551) },
      { UINT64_C( 6794335777646978013), UINT64_C(16793632523499560606), UINT64_C( 8062451955442882018), UINT64_C( 1061259078998417070) } },
    { { UINT64_C(11771759963807795039), UINT64_C(16059095854472724269), UINT64_C( 6081807115221301938), UINT64_C( 2243029834274887237) },
      UINT8_C(158),
      { UINT32_C(3220139790), UINT32_C(1273861068), UINT32_C(3185784926), UINT32_C(2406465813), UINT32_C(3806897911), UINT32_C(3794287884), UINT32_C(2728539339), UINT32_C(1665188437) },
      { UINT32_C(4246875953), UINT32_C( 558370754), UINT32_C(1574841159), UINT32_C(3823914476), UINT32_C(  13030900), UINT32_C(3722702354), UINT32_C(2004845858), UINT32_C(2027601735) },
      { UINT64_C(11771759963807795039), UINT64_C( 5017105225186569234), UINT64_C(   49607305988449900), UINT64_C( 5470300792184207862) } },
    { { UINT64_C( 6112155506509741550), UINT64_C(17417845272392872168), UINT64_C(10331773682228297330), UINT64_C(10239744778514653274) },
      UINT8_C(222),
      { UINT32_C(1307197328), UINT32_C(3241557521), UINT32_C(2378042351), UINT32_C(1124053187), UINT32_C(2007925922), UINT32_C(3335601637), UINT32_C( 627405636), UINT32_C( 503575438) },
      { UINT32_C(1131146546), UINT32_C( 453288236), UINT32_C(1890108332), UINT32_C( 364030067), UINT32_C( 294412332), UINT32_C(3805765022), UINT32_C(1594310097), UINT32_C(1669138992) },
      { UINT64_C( 6112155506509741550), UINT64_C( 4494757661473968532), UINT64_C(  591158153179270104), UINT64_C( 1000279140389506692) } },
    { { UINT64_C( 3909031611372857591), UINT64_C(13544110304352724846), UINT64_C(12698361903505572537), UINT64_C( 4716830434498920673) },
      UINT8_C(117),
      { UINT32_C(3338626331), UINT32_C(2335454628), UINT32_C( 874190300), UINT32_C(2381174283), UINT32_C(1215120547), UINT32_C(3173588349), UINT32_C(3104258861), UINT32_C(3425585584) },
      { UINT32_C(1385311662), UINT32_C(1071499107), UINT32_C(3899914460), UINT32_C(1920295375), UINT32_C( 599450277), UINT32_C( 769713152), UINT32_C(3504793631), UINT32_C(3751548209) },
      { UINT64_C( 4625037991394572122), UINT64_C(13544110304352724846), UINT64_C(  728404348487541519), UINT64_C( 4716830434498920673) } },
    { { UINT64_C(15196287490856136258), UINT64_C( 6793732513106384903), UINT64_C(  450187232347751186), UINT64_C( 9032657598156252490) },
      UINT8_C(161),
      { UINT32_C(2576753291), UINT32_C(1604347911), UINT32_C(1377334818), UINT32_C(3261363903), UINT32_C(1521055480), UINT32_C(3668225973), UINT32_C(4162133894), UINT32_C(  77173369) },
      { UINT32_C(3147641012), UINT32_C(3139059352), UINT32_C(1930244788), UINT32_C(2704634281), UINT32_C(2633752295), UINT32_C( 343318414), UINT32_C( 923569086), UINT32_C(3527124509) },
      { UINT64_C( 8110694336557570492), UINT64_C( 6793732513106384903), UINT64_C(  450187232347751186), UINT64_C( 9032657598156252490) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mul_epu32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mul_epu32");
    easysimd_test_x86_assert_equal_u64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_u64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__m256i r = easysimd_mm256_mask_mul_epu32(src, k, a, b);

    easysimd_test_x86_write_u64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_mul_epu32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    uint32_t a[8];
    uint32_t b[8];
    uint64_t r[4];
  } test_vec[8] = {
    { UINT8_C( 41),
      { UINT32_C(2911901007), UINT32_C(1395510666), UINT32_C( 443995936), UINT32_C(2983502410), UINT32_C(3285119819), UINT32_C(1714667826), UINT32_C( 763025838), UINT32_C(1364655874) },
      { UINT32_C(2684348181), UINT32_C(3572706484), UINT32_C(1324312835), UINT32_C(1476379404), UINT32_C(3910847926), UINT32_C(1011830158), UINT32_C( 678086950), UINT32_C(3732586697) },
      { UINT64_C( 7816556171392518267), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(  517397863260614100) } },
    { UINT8_C(167),
      { UINT32_C(2774302073), UINT32_C(3668455536), UINT32_C(3823564319), UINT32_C(3281599990), UINT32_C(2773647959), UINT32_C(2613808849), UINT32_C(3110401272), UINT32_C(3881845358) },
      { UINT32_C( 814529727), UINT32_C( 201995501), UINT32_C( 569372715), UINT32_C(2229569837), UINT32_C(3710465547), UINT32_C(3178821061), UINT32_C(1467407593), UINT32_C(3711883038) },
      { UINT64_C( 2259751510136224071), UINT64_C( 2177033197286156085), UINT64_C(10291525191376368573), UINT64_C(                   0) } },
    { UINT8_C(147),
      { UINT32_C(4269805002), UINT32_C( 120163607), UINT32_C(  87313020), UINT32_C(1695660335), UINT32_C(3609915106), UINT32_C(1119938662), UINT32_C( 895489886), UINT32_C( 533282133) },
      { UINT32_C(1646152011), UINT32_C(1399474134), UINT32_C(3243810450), UINT32_C( 975595863), UINT32_C(3172028759), UINT32_C(2550125113), UINT32_C(1070424041), UINT32_C(3898513053) },
      { UINT64_C( 7028748090620159022), UINT64_C(  283226886697059000), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C(223),
      { UINT32_C(3300280956), UINT32_C(1398147508), UINT32_C(3416921953), UINT32_C(2384651325), UINT32_C(3368542454), UINT32_C(1068589023), UINT32_C(3269259308), UINT32_C(3416376399) },
      { UINT32_C(3280951054), UINT32_C(3256280416), UINT32_C( 982368508), UINT32_C(2613620645), UINT32_C(1868796047), UINT32_C( 464393455), UINT32_C(1423804933), UINT32_C(1545567822) },
      { UINT64_C(10828060281084327624), UINT64_C( 3356676520921056124), UINT64_C( 6295118822186879338), UINT64_C( 4654787529986566364) } },
    { UINT8_C(214),
      { UINT32_C(2486575023), UINT32_C(4136761397), UINT32_C( 899402629), UINT32_C( 600061587), UINT32_C(2903651225), UINT32_C(1806839265), UINT32_C(2310604298), UINT32_C(3579778598) },
      { UINT32_C(1802081589), UINT32_C( 325188238), UINT32_C(1497955525), UINT32_C(3413904690), UINT32_C( 578326336), UINT32_C(3347917756), UINT32_C(1464878897), UINT32_C(2452402013) },
      { UINT64_C(                   0), UINT64_C( 1347265137310075225), UINT64_C( 1679257973976161600), UINT64_C(                   0) } },
    { UINT8_C( 68),
      { UINT32_C(2429746582), UINT32_C(1515644510), UINT32_C( 999075630), UINT32_C(3128710955), UINT32_C(4218920656), UINT32_C(1915502123), UINT32_C(1037010062), UINT32_C(1182884528) },
      { UINT32_C(3201782879), UINT32_C(1746414906), UINT32_C( 128230876), UINT32_C(3435274492), UINT32_C(3922147774), UINT32_C(  89977975), UINT32_C( 675425144), UINT32_C(3983524749) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C(16547230259613019744), UINT64_C(                   0) } },
    { UINT8_C( 23),
      { UINT32_C(1934732102), UINT32_C(1750055619), UINT32_C(2120570462), UINT32_C(1362899224), UINT32_C(3989317113), UINT32_C(2909130113), UINT32_C(3543829775), UINT32_C(1122641916) },
      { UINT32_C(2528459986), UINT32_C(1425933558), UINT32_C(1943167835), UINT32_C(2395278997), UINT32_C(3044773171), UINT32_C(1784864859), UINT32_C(1782422637), UINT32_C(2527865028) },
      { UINT64_C( 4891892703536670572), UINT64_C( 4120624313609489770), UINT64_C(12146565716273575323), UINT64_C(                   0) } },
    { UINT8_C(100),
      { UINT32_C(1717185634), UINT32_C(2395057707), UINT32_C(2384672128), UINT32_C(2277618170), UINT32_C( 216167980), UINT32_C(1970949592), UINT32_C(2990138506), UINT32_C(4061581200) },
      { UINT32_C( 660173052), UINT32_C(2662668830), UINT32_C(1227675983), UINT32_C(3083988618), UINT32_C(1036235620), UINT32_C(2343714048), UINT32_C(2990402337), UINT32_C(3064222906) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C(  224000960779447600), UINT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mul_epu32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mul_epu32");
    easysimd_test_x86_assert_equal_u64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__m256i r = easysimd_mm256_maskz_mul_epu32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mul_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float32 src[8];
    easysimd__mmask8 k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT32_C(   324.06), EASYSIMD_FLOAT32_C(   197.43), EASYSIMD_FLOAT32_C(   739.52), EASYSIMD_FLOAT32_C(   279.54),
        EASYSIMD_FLOAT32_C(   189.92), EASYSIMD_FLOAT32_C(  -722.21), EASYSIMD_FLOAT32_C(  -923.51), EASYSIMD_FLOAT32_C(  -873.97) },
      UINT8_C(214),
      { EASYSIMD_FLOAT32_C(  -133.91), EASYSIMD_FLOAT32_C(  -405.92), EASYSIMD_FLOAT32_C(    60.02), EASYSIMD_FLOAT32_C(  -925.14),
        EASYSIMD_FLOAT32_C(   142.59), EASYSIMD_FLOAT32_C(   519.61), EASYSIMD_FLOAT32_C(  -139.96), EASYSIMD_FLOAT32_C(   903.04) },
      { EASYSIMD_FLOAT32_C(  -777.76), EASYSIMD_FLOAT32_C(  -420.34), EASYSIMD_FLOAT32_C(   806.97), EASYSIMD_FLOAT32_C(  -759.85),
        EASYSIMD_FLOAT32_C(   441.58), EASYSIMD_FLOAT32_C(  -274.15), EASYSIMD_FLOAT32_C(  -547.27), EASYSIMD_FLOAT32_C(   137.15) },
      { EASYSIMD_FLOAT32_C(   324.06), EASYSIMD_FLOAT32_C(170624.42), EASYSIMD_FLOAT32_C( 48434.34), EASYSIMD_FLOAT32_C(   279.54),
        EASYSIMD_FLOAT32_C( 62964.89), EASYSIMD_FLOAT32_C(  -722.21), EASYSIMD_FLOAT32_C( 76595.91), EASYSIMD_FLOAT32_C(123851.93) } },
    { { EASYSIMD_FLOAT32_C(   374.14), EASYSIMD_FLOAT32_C(   815.69), EASYSIMD_FLOAT32_C(   931.35), EASYSIMD_FLOAT32_C(   503.66),
        EASYSIMD_FLOAT32_C(    72.57), EASYSIMD_FLOAT32_C(   332.07), EASYSIMD_FLOAT32_C(  -172.29), EASYSIMD_FLOAT32_C(  -729.99) },
      UINT8_C(215),
      { EASYSIMD_FLOAT32_C(  -892.74), EASYSIMD_FLOAT32_C(   459.93), EASYSIMD_FLOAT32_C(   349.37), EASYSIMD_FLOAT32_C(  -816.26),
        EASYSIMD_FLOAT32_C(   585.95), EASYSIMD_FLOAT32_C(   886.39), EASYSIMD_FLOAT32_C(    49.83), EASYSIMD_FLOAT32_C(  -819.97) },
      { EASYSIMD_FLOAT32_C(   -53.59), EASYSIMD_FLOAT32_C(   124.69), EASYSIMD_FLOAT32_C(   322.62), EASYSIMD_FLOAT32_C(  -533.97),
        EASYSIMD_FLOAT32_C(   984.72), EASYSIMD_FLOAT32_C(   225.66), EASYSIMD_FLOAT32_C(  -311.73), EASYSIMD_FLOAT32_C(  -435.61) },
      { EASYSIMD_FLOAT32_C( 47841.94), EASYSIMD_FLOAT32_C( 57348.67), EASYSIMD_FLOAT32_C(112713.75), EASYSIMD_FLOAT32_C(   503.66),
        EASYSIMD_FLOAT32_C(576996.69), EASYSIMD_FLOAT32_C(   332.07), EASYSIMD_FLOAT32_C(-15533.51), EASYSIMD_FLOAT32_C(357187.09) } },
    { { EASYSIMD_FLOAT32_C(    32.63), EASYSIMD_FLOAT32_C(   -71.58), EASYSIMD_FLOAT32_C(  -994.03), EASYSIMD_FLOAT32_C(   758.47),
        EASYSIMD_FLOAT32_C(   381.15), EASYSIMD_FLOAT32_C(   143.13), EASYSIMD_FLOAT32_C(   132.62), EASYSIMD_FLOAT32_C(   196.84) },
      UINT8_C(241),
      { EASYSIMD_FLOAT32_C(  -363.73), EASYSIMD_FLOAT32_C(  -730.59), EASYSIMD_FLOAT32_C(  -593.46), EASYSIMD_FLOAT32_C(   463.99),
        EASYSIMD_FLOAT32_C(  -460.58), EASYSIMD_FLOAT32_C(   478.12), EASYSIMD_FLOAT32_C(   571.24), EASYSIMD_FLOAT32_C(   999.34) },
      { EASYSIMD_FLOAT32_C(  -172.51), EASYSIMD_FLOAT32_C(   754.99), EASYSIMD_FLOAT32_C(   585.30), EASYSIMD_FLOAT32_C(  -286.12),
        EASYSIMD_FLOAT32_C(  -195.18), EASYSIMD_FLOAT32_C(   765.33), EASYSIMD_FLOAT32_C(   660.30), EASYSIMD_FLOAT32_C(   929.51) },
      { EASYSIMD_FLOAT32_C( 62747.06), EASYSIMD_FLOAT32_C(   -71.58), EASYSIMD_FLOAT32_C(  -994.03), EASYSIMD_FLOAT32_C(   758.47),
        EASYSIMD_FLOAT32_C( 89896.00), EASYSIMD_FLOAT32_C(365919.59), EASYSIMD_FLOAT32_C(377189.75), EASYSIMD_FLOAT32_C(928896.56) } },
    { { EASYSIMD_FLOAT32_C(    87.95), EASYSIMD_FLOAT32_C(  -873.68), EASYSIMD_FLOAT32_C(   914.23), EASYSIMD_FLOAT32_C(  -686.39),
        EASYSIMD_FLOAT32_C(  -185.41), EASYSIMD_FLOAT32_C(  -521.38), EASYSIMD_FLOAT32_C(   346.24), EASYSIMD_FLOAT32_C(   743.01) },
      UINT8_C(247),
      { EASYSIMD_FLOAT32_C(   104.71), EASYSIMD_FLOAT32_C(   124.16), EASYSIMD_FLOAT32_C(   627.72), EASYSIMD_FLOAT32_C(  -762.67),
        EASYSIMD_FLOAT32_C(  -679.01), EASYSIMD_FLOAT32_C(  -297.81), EASYSIMD_FLOAT32_C(  -126.40), EASYSIMD_FLOAT32_C(  -409.60) },
      { EASYSIMD_FLOAT32_C(   108.73), EASYSIMD_FLOAT32_C(  -662.41), EASYSIMD_FLOAT32_C(   129.82), EASYSIMD_FLOAT32_C(  -413.15),
        EASYSIMD_FLOAT32_C(   908.83), EASYSIMD_FLOAT32_C(   129.16), EASYSIMD_FLOAT32_C(   414.35), EASYSIMD_FLOAT32_C(   663.82) },
      { EASYSIMD_FLOAT32_C( 11385.12), EASYSIMD_FLOAT32_C(-82244.83), EASYSIMD_FLOAT32_C( 81490.61), EASYSIMD_FLOAT32_C(  -686.39),
        EASYSIMD_FLOAT32_C(-617104.69), EASYSIMD_FLOAT32_C(-38465.14), EASYSIMD_FLOAT32_C(-52373.84), EASYSIMD_FLOAT32_C(-271900.69) } },
    { { EASYSIMD_FLOAT32_C(  -285.54), EASYSIMD_FLOAT32_C(  -871.77), EASYSIMD_FLOAT32_C(  -531.36), EASYSIMD_FLOAT32_C(  -520.21),
        EASYSIMD_FLOAT32_C(   788.53), EASYSIMD_FLOAT32_C(  -601.86), EASYSIMD_FLOAT32_C(   567.74), EASYSIMD_FLOAT32_C(   914.85) },
      UINT8_C(  5),
      { EASYSIMD_FLOAT32_C(   881.34), EASYSIMD_FLOAT32_C(  -270.56), EASYSIMD_FLOAT32_C(  -209.01), EASYSIMD_FLOAT32_C(   227.58),
        EASYSIMD_FLOAT32_C(  -527.55), EASYSIMD_FLOAT32_C(   275.58), EASYSIMD_FLOAT32_C(  -667.71), EASYSIMD_FLOAT32_C(   596.60) },
      { EASYSIMD_FLOAT32_C(   -96.70), EASYSIMD_FLOAT32_C(  -430.38), EASYSIMD_FLOAT32_C(   917.59), EASYSIMD_FLOAT32_C(   605.49),
        EASYSIMD_FLOAT32_C(   443.22), EASYSIMD_FLOAT32_C(  -492.01), EASYSIMD_FLOAT32_C(  -285.78), EASYSIMD_FLOAT32_C(   780.80) },
      { EASYSIMD_FLOAT32_C(-85225.58), EASYSIMD_FLOAT32_C(  -871.77), EASYSIMD_FLOAT32_C(-191785.48), EASYSIMD_FLOAT32_C(  -520.21),
        EASYSIMD_FLOAT32_C(   788.53), EASYSIMD_FLOAT32_C(  -601.86), EASYSIMD_FLOAT32_C(   567.74), EASYSIMD_FLOAT32_C(   914.85) } },
    { { EASYSIMD_FLOAT32_C(   637.81), EASYSIMD_FLOAT32_C(   301.07), EASYSIMD_FLOAT32_C(   689.63), EASYSIMD_FLOAT32_C(  -233.03),
        EASYSIMD_FLOAT32_C(  -284.58), EASYSIMD_FLOAT32_C(   353.45), EASYSIMD_FLOAT32_C(   481.43), EASYSIMD_FLOAT32_C(  -156.35) },
      UINT8_C(167),
      { EASYSIMD_FLOAT32_C(   961.22), EASYSIMD_FLOAT32_C(  -367.82), EASYSIMD_FLOAT32_C(  -779.77), EASYSIMD_FLOAT32_C(   528.95),
        EASYSIMD_FLOAT32_C(  -452.98), EASYSIMD_FLOAT32_C(  -467.40), EASYSIMD_FLOAT32_C(   410.30), EASYSIMD_FLOAT32_C(   276.46) },
      { EASYSIMD_FLOAT32_C(   323.59), EASYSIMD_FLOAT32_C(  -362.12), EASYSIMD_FLOAT32_C(   748.91), EASYSIMD_FLOAT32_C(  -400.83),
        EASYSIMD_FLOAT32_C(   -29.84), EASYSIMD_FLOAT32_C(   345.51), EASYSIMD_FLOAT32_C(   502.47), EASYSIMD_FLOAT32_C(   539.78) },
      { EASYSIMD_FLOAT32_C(311041.16), EASYSIMD_FLOAT32_C(133194.98), EASYSIMD_FLOAT32_C(-583977.56), EASYSIMD_FLOAT32_C(  -233.03),
        EASYSIMD_FLOAT32_C(  -284.58), EASYSIMD_FLOAT32_C(-161491.38), EASYSIMD_FLOAT32_C(   481.43), EASYSIMD_FLOAT32_C(149227.58) } },
    { { EASYSIMD_FLOAT32_C(   263.11), EASYSIMD_FLOAT32_C(   107.95), EASYSIMD_FLOAT32_C(   -17.00), EASYSIMD_FLOAT32_C(   771.10),
        EASYSIMD_FLOAT32_C(   822.17), EASYSIMD_FLOAT32_C(  -236.20), EASYSIMD_FLOAT32_C(   408.91), EASYSIMD_FLOAT32_C(   123.25) },
      UINT8_C(220),
      { EASYSIMD_FLOAT32_C(  -824.12), EASYSIMD_FLOAT32_C(   838.66), EASYSIMD_FLOAT32_C(   806.89), EASYSIMD_FLOAT32_C(   657.32),
        EASYSIMD_FLOAT32_C(  -317.69), EASYSIMD_FLOAT32_C(   628.97), EASYSIMD_FLOAT32_C(   618.53), EASYSIMD_FLOAT32_C(   314.49) },
      { EASYSIMD_FLOAT32_C(   849.20), EASYSIMD_FLOAT32_C(   147.49), EASYSIMD_FLOAT32_C(   861.51), EASYSIMD_FLOAT32_C(  -618.20),
        EASYSIMD_FLOAT32_C(  -442.22), EASYSIMD_FLOAT32_C(   137.98), EASYSIMD_FLOAT32_C(   705.38), EASYSIMD_FLOAT32_C(   195.66) },
      { EASYSIMD_FLOAT32_C(   263.11), EASYSIMD_FLOAT32_C(   107.95), EASYSIMD_FLOAT32_C(695143.81), EASYSIMD_FLOAT32_C(-406355.25),
        EASYSIMD_FLOAT32_C(140488.88), EASYSIMD_FLOAT32_C(  -236.20), EASYSIMD_FLOAT32_C(436298.72), EASYSIMD_FLOAT32_C( 61533.11) } },
    { { EASYSIMD_FLOAT32_C(  -113.11), EASYSIMD_FLOAT32_C(  -695.45), EASYSIMD_FLOAT32_C(  -834.18), EASYSIMD_FLOAT32_C(  -767.60),
        EASYSIMD_FLOAT32_C(   807.02), EASYSIMD_FLOAT32_C(   705.60), EASYSIMD_FLOAT32_C(   495.51), EASYSIMD_FLOAT32_C(   -85.03) },
      UINT8_C(123),
      { EASYSIMD_FLOAT32_C(   266.61), EASYSIMD_FLOAT32_C(  -262.85), EASYSIMD_FLOAT32_C(   452.40), EASYSIMD_FLOAT32_C(  -324.48),
        EASYSIMD_FLOAT32_C(   860.39), EASYSIMD_FLOAT32_C(   905.83), EASYSIMD_FLOAT32_C(  -148.60), EASYSIMD_FLOAT32_C(   699.05) },
      { EASYSIMD_FLOAT32_C(   712.72), EASYSIMD_FLOAT32_C(  -491.28), EASYSIMD_FLOAT32_C(  -618.63), EASYSIMD_FLOAT32_C(   341.69),
        EASYSIMD_FLOAT32_C(  -872.75), EASYSIMD_FLOAT32_C(   695.85), EASYSIMD_FLOAT32_C(   190.89), EASYSIMD_FLOAT32_C(   274.74) },
      { EASYSIMD_FLOAT32_C(190018.27), EASYSIMD_FLOAT32_C(129132.95), EASYSIMD_FLOAT32_C(  -834.18), EASYSIMD_FLOAT32_C(-110871.58),
        EASYSIMD_FLOAT32_C(-750905.38), EASYSIMD_FLOAT32_C(630321.81), EASYSIMD_FLOAT32_C(-28366.26), EASYSIMD_FLOAT32_C(   -85.03) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mul_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mul_ps");
    easysimd_assert_m256_close(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_mul_ps(src, k, a, b);

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
test_easysimd_mm256_maskz_mul_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[8] = {
    { UINT8_C(110),
      { EASYSIMD_FLOAT32_C(  -620.05), EASYSIMD_FLOAT32_C(   415.97), EASYSIMD_FLOAT32_C(   644.97), EASYSIMD_FLOAT32_C(    34.20),
        EASYSIMD_FLOAT32_C(  -932.49), EASYSIMD_FLOAT32_C(   730.56), EASYSIMD_FLOAT32_C(   810.03), EASYSIMD_FLOAT32_C(   -68.48) },
      { EASYSIMD_FLOAT32_C(   921.22), EASYSIMD_FLOAT32_C(   282.37), EASYSIMD_FLOAT32_C(  -979.23), EASYSIMD_FLOAT32_C(  -312.45),
        EASYSIMD_FLOAT32_C(  -175.72), EASYSIMD_FLOAT32_C(  -119.01), EASYSIMD_FLOAT32_C(  -446.99), EASYSIMD_FLOAT32_C(   749.83) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(117457.45), EASYSIMD_FLOAT32_C(-631573.94), EASYSIMD_FLOAT32_C(-10685.79),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-86943.95), EASYSIMD_FLOAT32_C(-362075.31), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(228),
      { EASYSIMD_FLOAT32_C(  -837.92), EASYSIMD_FLOAT32_C(    69.61), EASYSIMD_FLOAT32_C(  -131.82), EASYSIMD_FLOAT32_C(  -735.90),
        EASYSIMD_FLOAT32_C(   990.94), EASYSIMD_FLOAT32_C(   948.99), EASYSIMD_FLOAT32_C(   584.02), EASYSIMD_FLOAT32_C(  -447.32) },
      { EASYSIMD_FLOAT32_C(  -392.84), EASYSIMD_FLOAT32_C(  -367.00), EASYSIMD_FLOAT32_C(  -126.65), EASYSIMD_FLOAT32_C(   288.33),
        EASYSIMD_FLOAT32_C(  -511.87), EASYSIMD_FLOAT32_C(  -411.83), EASYSIMD_FLOAT32_C(   668.28), EASYSIMD_FLOAT32_C(   904.10) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 16695.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-390822.53), EASYSIMD_FLOAT32_C(390288.91), EASYSIMD_FLOAT32_C(-404422.00) } },
    { UINT8_C(100),
      { EASYSIMD_FLOAT32_C(  -297.52), EASYSIMD_FLOAT32_C(   971.61), EASYSIMD_FLOAT32_C(   963.70), EASYSIMD_FLOAT32_C(  -487.50),
        EASYSIMD_FLOAT32_C(   -96.88), EASYSIMD_FLOAT32_C(   884.92), EASYSIMD_FLOAT32_C(   794.88), EASYSIMD_FLOAT32_C(   -76.10) },
      { EASYSIMD_FLOAT32_C(  -427.53), EASYSIMD_FLOAT32_C(  -380.85), EASYSIMD_FLOAT32_C(   804.89), EASYSIMD_FLOAT32_C(   125.48),
        EASYSIMD_FLOAT32_C(  -631.02), EASYSIMD_FLOAT32_C(    84.28), EASYSIMD_FLOAT32_C(   287.56), EASYSIMD_FLOAT32_C(   438.59) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(775672.50), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 74581.05), EASYSIMD_FLOAT32_C(228575.69), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(139),
      { EASYSIMD_FLOAT32_C(   551.66), EASYSIMD_FLOAT32_C(   429.53), EASYSIMD_FLOAT32_C(   901.45), EASYSIMD_FLOAT32_C(   135.68),
        EASYSIMD_FLOAT32_C(   982.20), EASYSIMD_FLOAT32_C(  -491.39), EASYSIMD_FLOAT32_C(   768.68), EASYSIMD_FLOAT32_C(  -144.45) },
      { EASYSIMD_FLOAT32_C(   796.94), EASYSIMD_FLOAT32_C(  -743.19), EASYSIMD_FLOAT32_C(   443.72), EASYSIMD_FLOAT32_C(   465.22),
        EASYSIMD_FLOAT32_C(  -839.10), EASYSIMD_FLOAT32_C(   676.87), EASYSIMD_FLOAT32_C(  -832.31), EASYSIMD_FLOAT32_C(  -867.49) },
      { EASYSIMD_FLOAT32_C(439639.91), EASYSIMD_FLOAT32_C(-319222.41), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 63121.05),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(125308.93) } },
    { UINT8_C(118),
      { EASYSIMD_FLOAT32_C(  -319.80), EASYSIMD_FLOAT32_C(    35.63), EASYSIMD_FLOAT32_C(   525.48), EASYSIMD_FLOAT32_C(  -524.92),
        EASYSIMD_FLOAT32_C(   959.53), EASYSIMD_FLOAT32_C(  -902.05), EASYSIMD_FLOAT32_C(    94.23), EASYSIMD_FLOAT32_C(   764.42) },
      { EASYSIMD_FLOAT32_C(   223.43), EASYSIMD_FLOAT32_C(   463.21), EASYSIMD_FLOAT32_C(  -151.30), EASYSIMD_FLOAT32_C(  -489.01),
        EASYSIMD_FLOAT32_C(   -98.19), EASYSIMD_FLOAT32_C(  -198.84), EASYSIMD_FLOAT32_C(  -937.35), EASYSIMD_FLOAT32_C(  -668.66) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 16504.17), EASYSIMD_FLOAT32_C(-79505.12), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-94216.26), EASYSIMD_FLOAT32_C(179363.61), EASYSIMD_FLOAT32_C(-88326.49), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(211),
      { EASYSIMD_FLOAT32_C(   198.33), EASYSIMD_FLOAT32_C(  -686.46), EASYSIMD_FLOAT32_C(   211.22), EASYSIMD_FLOAT32_C(   -32.99),
        EASYSIMD_FLOAT32_C(   169.09), EASYSIMD_FLOAT32_C(     8.16), EASYSIMD_FLOAT32_C(   223.82), EASYSIMD_FLOAT32_C(  -387.18) },
      { EASYSIMD_FLOAT32_C(  -526.62), EASYSIMD_FLOAT32_C(   384.72), EASYSIMD_FLOAT32_C(  -710.32), EASYSIMD_FLOAT32_C(  -358.92),
        EASYSIMD_FLOAT32_C(   517.23), EASYSIMD_FLOAT32_C(   930.25), EASYSIMD_FLOAT32_C(   321.27), EASYSIMD_FLOAT32_C(  -447.14) },
      { EASYSIMD_FLOAT32_C(-104444.55), EASYSIMD_FLOAT32_C(-264094.91), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( 87458.41), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 71906.65), EASYSIMD_FLOAT32_C(173123.67) } },
    { UINT8_C(141),
      { EASYSIMD_FLOAT32_C(   796.35), EASYSIMD_FLOAT32_C(  -487.61), EASYSIMD_FLOAT32_C(   553.68), EASYSIMD_FLOAT32_C(  -109.42),
        EASYSIMD_FLOAT32_C(  -723.19), EASYSIMD_FLOAT32_C(  -222.89), EASYSIMD_FLOAT32_C(  -646.21), EASYSIMD_FLOAT32_C(   125.51) },
      { EASYSIMD_FLOAT32_C(   288.11), EASYSIMD_FLOAT32_C(   255.60), EASYSIMD_FLOAT32_C(   926.67), EASYSIMD_FLOAT32_C(   350.76),
        EASYSIMD_FLOAT32_C(   586.93), EASYSIMD_FLOAT32_C(  -370.72), EASYSIMD_FLOAT32_C(  -450.91), EASYSIMD_FLOAT32_C(   900.47) },
      { EASYSIMD_FLOAT32_C(229436.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(513078.62), EASYSIMD_FLOAT32_C(-38380.16),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(113017.98) } },
    { UINT8_C(106),
      { EASYSIMD_FLOAT32_C(   516.10), EASYSIMD_FLOAT32_C(    69.57), EASYSIMD_FLOAT32_C(  -151.34), EASYSIMD_FLOAT32_C(  -260.09),
        EASYSIMD_FLOAT32_C(   682.38), EASYSIMD_FLOAT32_C(   322.04), EASYSIMD_FLOAT32_C(  -875.37), EASYSIMD_FLOAT32_C(   972.07) },
      { EASYSIMD_FLOAT32_C(   963.12), EASYSIMD_FLOAT32_C(   641.86), EASYSIMD_FLOAT32_C(   902.31), EASYSIMD_FLOAT32_C(   284.39),
        EASYSIMD_FLOAT32_C(  -805.28), EASYSIMD_FLOAT32_C(   358.04), EASYSIMD_FLOAT32_C(    80.74), EASYSIMD_FLOAT32_C(  -292.89) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 44654.20), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-73967.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(115303.21), EASYSIMD_FLOAT32_C(-70677.38), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mul_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mul_ps");
    easysimd_assert_m256_close(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_mul_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mul_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float64 src[4];
    easysimd__mmask8 k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT64_C(     1.35), EASYSIMD_FLOAT64_C(   417.14), EASYSIMD_FLOAT64_C(   -85.36), EASYSIMD_FLOAT64_C(   180.14) },
      UINT8_C( 71),
      { EASYSIMD_FLOAT64_C(  -934.95), EASYSIMD_FLOAT64_C(   624.76), EASYSIMD_FLOAT64_C(   741.82), EASYSIMD_FLOAT64_C(   744.02) },
      { EASYSIMD_FLOAT64_C(   760.00), EASYSIMD_FLOAT64_C(   223.08), EASYSIMD_FLOAT64_C(   355.03), EASYSIMD_FLOAT64_C(   741.91) },
      { EASYSIMD_FLOAT64_C(-710562.00), EASYSIMD_FLOAT64_C(139371.46), EASYSIMD_FLOAT64_C(263368.35), EASYSIMD_FLOAT64_C(   180.14) } },
    { { EASYSIMD_FLOAT64_C(   969.99), EASYSIMD_FLOAT64_C(   772.53), EASYSIMD_FLOAT64_C(  -202.08), EASYSIMD_FLOAT64_C(   221.41) },
      UINT8_C(116),
      { EASYSIMD_FLOAT64_C(  -290.70), EASYSIMD_FLOAT64_C(  -492.14), EASYSIMD_FLOAT64_C(  -870.29), EASYSIMD_FLOAT64_C(   708.97) },
      { EASYSIMD_FLOAT64_C(  -502.66), EASYSIMD_FLOAT64_C(   902.42), EASYSIMD_FLOAT64_C(   419.29), EASYSIMD_FLOAT64_C(   826.49) },
      { EASYSIMD_FLOAT64_C(   969.99), EASYSIMD_FLOAT64_C(   772.53), EASYSIMD_FLOAT64_C(-364903.89), EASYSIMD_FLOAT64_C(   221.41) } },
    { { EASYSIMD_FLOAT64_C(   837.89), EASYSIMD_FLOAT64_C(  -891.01), EASYSIMD_FLOAT64_C(   218.35), EASYSIMD_FLOAT64_C(  -100.46) },
      UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(  -780.30), EASYSIMD_FLOAT64_C(  -683.31), EASYSIMD_FLOAT64_C(  -610.01), EASYSIMD_FLOAT64_C(   399.84) },
      { EASYSIMD_FLOAT64_C(   421.34), EASYSIMD_FLOAT64_C(  -544.96), EASYSIMD_FLOAT64_C(    24.60), EASYSIMD_FLOAT64_C(   163.16) },
      { EASYSIMD_FLOAT64_C(   837.89), EASYSIMD_FLOAT64_C(  -891.01), EASYSIMD_FLOAT64_C(   218.35), EASYSIMD_FLOAT64_C( 65237.89) } },
    { { EASYSIMD_FLOAT64_C(  -800.93), EASYSIMD_FLOAT64_C(  -215.40), EASYSIMD_FLOAT64_C(  -613.76), EASYSIMD_FLOAT64_C(   554.09) },
      UINT8_C(121),
      { EASYSIMD_FLOAT64_C(  -643.77), EASYSIMD_FLOAT64_C(   326.62), EASYSIMD_FLOAT64_C(   324.43), EASYSIMD_FLOAT64_C(   577.64) },
      { EASYSIMD_FLOAT64_C(  -119.95), EASYSIMD_FLOAT64_C(  -966.27), EASYSIMD_FLOAT64_C(  -914.50), EASYSIMD_FLOAT64_C(     9.75) },
      { EASYSIMD_FLOAT64_C( 77220.21), EASYSIMD_FLOAT64_C(  -215.40), EASYSIMD_FLOAT64_C(  -613.76), EASYSIMD_FLOAT64_C(  5631.99) } },
    { { EASYSIMD_FLOAT64_C(   742.69), EASYSIMD_FLOAT64_C(  -417.16), EASYSIMD_FLOAT64_C(   -87.82), EASYSIMD_FLOAT64_C(   161.99) },
      UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(  -249.94), EASYSIMD_FLOAT64_C(   270.97), EASYSIMD_FLOAT64_C(   627.68), EASYSIMD_FLOAT64_C(   649.60) },
      { EASYSIMD_FLOAT64_C(  -253.68), EASYSIMD_FLOAT64_C(   847.38), EASYSIMD_FLOAT64_C(   966.29), EASYSIMD_FLOAT64_C(   136.32) },
      { EASYSIMD_FLOAT64_C(   742.69), EASYSIMD_FLOAT64_C(  -417.16), EASYSIMD_FLOAT64_C(   -87.82), EASYSIMD_FLOAT64_C( 88553.47) } },
    { { EASYSIMD_FLOAT64_C(   247.21), EASYSIMD_FLOAT64_C(   387.63), EASYSIMD_FLOAT64_C(   591.36), EASYSIMD_FLOAT64_C(  -728.19) },
      UINT8_C(169),
      { EASYSIMD_FLOAT64_C(   790.42), EASYSIMD_FLOAT64_C(    56.41), EASYSIMD_FLOAT64_C(   -62.96), EASYSIMD_FLOAT64_C(   344.52) },
      { EASYSIMD_FLOAT64_C(   582.91), EASYSIMD_FLOAT64_C(   293.27), EASYSIMD_FLOAT64_C(  -328.87), EASYSIMD_FLOAT64_C(   -92.66) },
      { EASYSIMD_FLOAT64_C(460743.72), EASYSIMD_FLOAT64_C(   387.63), EASYSIMD_FLOAT64_C(   591.36), EASYSIMD_FLOAT64_C(-31923.22) } },
    { { EASYSIMD_FLOAT64_C(  -129.09), EASYSIMD_FLOAT64_C(   551.18), EASYSIMD_FLOAT64_C(   -58.94), EASYSIMD_FLOAT64_C(   -43.59) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT64_C(  -316.25), EASYSIMD_FLOAT64_C(   539.24), EASYSIMD_FLOAT64_C(   473.11), EASYSIMD_FLOAT64_C(   845.74) },
      { EASYSIMD_FLOAT64_C(   948.57), EASYSIMD_FLOAT64_C(  -776.82), EASYSIMD_FLOAT64_C(   116.71), EASYSIMD_FLOAT64_C(   576.24) },
      { EASYSIMD_FLOAT64_C(-299985.26), EASYSIMD_FLOAT64_C(   551.18), EASYSIMD_FLOAT64_C(   -58.94), EASYSIMD_FLOAT64_C(487349.22) } },
    { { EASYSIMD_FLOAT64_C(   872.78), EASYSIMD_FLOAT64_C(   863.04), EASYSIMD_FLOAT64_C(   423.62), EASYSIMD_FLOAT64_C(   839.07) },
      UINT8_C( 69),
      { EASYSIMD_FLOAT64_C(  -329.17), EASYSIMD_FLOAT64_C(   226.71), EASYSIMD_FLOAT64_C(  -409.29), EASYSIMD_FLOAT64_C(   -57.36) },
      { EASYSIMD_FLOAT64_C(   777.50), EASYSIMD_FLOAT64_C(  -618.87), EASYSIMD_FLOAT64_C(   999.05), EASYSIMD_FLOAT64_C(  -285.46) },
      { EASYSIMD_FLOAT64_C(-255929.68), EASYSIMD_FLOAT64_C(   863.04), EASYSIMD_FLOAT64_C(-408901.17), EASYSIMD_FLOAT64_C(   839.07) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mul_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mul_pd");
    easysimd_assert_m256d_close(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_mul_pd(src, k, a, b);

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
test_easysimd_mm256_maskz_mul_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[8] = {
    { UINT8_C(193),
      { EASYSIMD_FLOAT64_C(   855.14), EASYSIMD_FLOAT64_C(   504.04), EASYSIMD_FLOAT64_C(   -84.95), EASYSIMD_FLOAT64_C(   925.75) },
      { EASYSIMD_FLOAT64_C(  -998.99), EASYSIMD_FLOAT64_C(  -504.03), EASYSIMD_FLOAT64_C(   -24.34), EASYSIMD_FLOAT64_C(   997.79) },
      { EASYSIMD_FLOAT64_C(-854276.31), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 72),
      { EASYSIMD_FLOAT64_C(  -796.90), EASYSIMD_FLOAT64_C(   280.89), EASYSIMD_FLOAT64_C(   -72.71), EASYSIMD_FLOAT64_C(   732.00) },
      { EASYSIMD_FLOAT64_C(   329.49), EASYSIMD_FLOAT64_C(   291.30), EASYSIMD_FLOAT64_C(   366.00), EASYSIMD_FLOAT64_C(   850.30) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(622419.60) } },
    { UINT8_C(191),
      { EASYSIMD_FLOAT64_C(  -536.99), EASYSIMD_FLOAT64_C(   228.01), EASYSIMD_FLOAT64_C(   -83.78), EASYSIMD_FLOAT64_C(   130.55) },
      { EASYSIMD_FLOAT64_C(  -676.48), EASYSIMD_FLOAT64_C(  -717.07), EASYSIMD_FLOAT64_C(  -167.58), EASYSIMD_FLOAT64_C(   798.21) },
      { EASYSIMD_FLOAT64_C(363263.00), EASYSIMD_FLOAT64_C(-163499.13), EASYSIMD_FLOAT64_C( 14039.85), EASYSIMD_FLOAT64_C(104206.32) } },
    { UINT8_C( 86),
      { EASYSIMD_FLOAT64_C(   353.85), EASYSIMD_FLOAT64_C(   263.82), EASYSIMD_FLOAT64_C(   888.15), EASYSIMD_FLOAT64_C(  -461.61) },
      { EASYSIMD_FLOAT64_C(   118.96), EASYSIMD_FLOAT64_C(   392.19), EASYSIMD_FLOAT64_C(   453.44), EASYSIMD_FLOAT64_C(    44.71) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(103467.57), EASYSIMD_FLOAT64_C(402722.74), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(222),
      { EASYSIMD_FLOAT64_C(   949.41), EASYSIMD_FLOAT64_C(  -979.63), EASYSIMD_FLOAT64_C(   391.00), EASYSIMD_FLOAT64_C(  -338.19) },
      { EASYSIMD_FLOAT64_C(  -776.53), EASYSIMD_FLOAT64_C(  -328.11), EASYSIMD_FLOAT64_C(   589.10), EASYSIMD_FLOAT64_C(   955.46) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(321426.40), EASYSIMD_FLOAT64_C(230338.10), EASYSIMD_FLOAT64_C(-323127.02) } },
    { UINT8_C(197),
      { EASYSIMD_FLOAT64_C(  -119.60), EASYSIMD_FLOAT64_C(   321.46), EASYSIMD_FLOAT64_C(   851.67), EASYSIMD_FLOAT64_C(   367.19) },
      { EASYSIMD_FLOAT64_C(   784.47), EASYSIMD_FLOAT64_C(    79.68), EASYSIMD_FLOAT64_C(  -716.59), EASYSIMD_FLOAT64_C(   -84.98) },
      { EASYSIMD_FLOAT64_C(-93822.61), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-610298.21), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(150),
      { EASYSIMD_FLOAT64_C(  -433.66), EASYSIMD_FLOAT64_C(   747.44), EASYSIMD_FLOAT64_C(   201.41), EASYSIMD_FLOAT64_C(  -333.63) },
      { EASYSIMD_FLOAT64_C(   101.29), EASYSIMD_FLOAT64_C(  -534.77), EASYSIMD_FLOAT64_C(  -445.48), EASYSIMD_FLOAT64_C(   639.68) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-399708.49), EASYSIMD_FLOAT64_C(-89724.13), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(203),
      { EASYSIMD_FLOAT64_C(   946.71), EASYSIMD_FLOAT64_C(    93.12), EASYSIMD_FLOAT64_C(  -371.11), EASYSIMD_FLOAT64_C(   339.91) },
      { EASYSIMD_FLOAT64_C(    42.53), EASYSIMD_FLOAT64_C(  -350.74), EASYSIMD_FLOAT64_C(  -269.09), EASYSIMD_FLOAT64_C(   704.34) },
      { EASYSIMD_FLOAT64_C( 40263.58), EASYSIMD_FLOAT64_C(-32660.91), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(239412.21) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mul_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mul_pd");
    easysimd_assert_m256d_close(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_mul_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mul_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT32_C(  1303646110), -INT32_C(  1991094019), -INT32_C(    60179606), -INT32_C(  1143961245),  INT32_C(    53151249),  INT32_C(  1139066569), -INT32_C(  1576434937), -INT32_C(  2053958169),
         INT32_C(   584206116),  INT32_C(    44770456),  INT32_C(  1123947743), -INT32_C(  1342320995), -INT32_C(  1632491307), -INT32_C(   169699602),  INT32_C(     9956121),  INT32_C(   612706816) },
      {  INT32_C(   172382066),  INT32_C(  1527574908), -INT32_C(  1549923834), -INT32_C(  1387095080), -INT32_C(  1303640636), -INT32_C(  1247269221),  INT32_C(   431308569), -INT32_C(   599966870),
         INT32_C(   249988242),  INT32_C(  2070606453), -INT32_C(   736163588),  INT32_C(  1736602019),  INT32_C(   303746678),  INT32_C(   365413116),  INT32_C(  1798208513),  INT32_C(  1246194615) },
      {  INT64_C(  224725209774663260),  INT64_C(   93273805660129404), -INT64_C(   69290128050554364), -INT64_C(  679929896799075153),
         INT64_C(  146044659904488072), -INT64_C(  827409403211381884), -INT64_C(  495863811365128146),  INT64_C(   17903181538658073) } },
    { {  INT32_C(  1700277743),  INT32_C(   467714591),  INT32_C(  1861287882), -INT32_C(   405441935),  INT32_C(  1023012672), -INT32_C(  1286487887), -INT32_C(   199327939), -INT32_C(   633444630),
        -INT32_C(  1287678061),  INT32_C(   617488217), -INT32_C(  1869431265),  INT32_C(  1886873392),  INT32_C(   145518935),  INT32_C(  1857813809),  INT32_C(  1734597244), -INT32_C(   750673600) },
      { -INT32_C(  1836679112), -INT32_C(  1095346785),  INT32_C(  1145980947),  INT32_C(   112510639),  INT32_C(  1745838391), -INT32_C(   606614946), -INT32_C(   465421660),  INT32_C(   347636699),
        -INT32_C(  1566163453), -INT32_C(  1486791533),  INT32_C(  1407954852), -INT32_C(  1403347083),  INT32_C(  1595238656), -INT32_C(   667227085),  INT32_C(    29129766),  INT32_C(    34960639) },
      { -INT64_C( 3122864615166604216),  INT64_C( 2133000449653984254),  INT64_C( 1786014797257090752),  INT64_C(   92771540253758740),
         INT64_C( 2016714318368104633), -INT64_C( 2632074820037247780),  INT64_C(  232137430291951360),  INT64_C(   50528411821964904) } },
    { {  INT32_C(  1185201075), -INT32_C(  1142094569),  INT32_C(   705681589),  INT32_C(  2027383160),  INT32_C(    98036946), -INT32_C(    19066408), -INT32_C(  1929405811), -INT32_C(  1047653106),
        -INT32_C(   402115632), -INT32_C(   308021960),  INT32_C(  1176023758), -INT32_C(   306254053), -INT32_C(  1275881765),  INT32_C(   900845735),  INT32_C(  2042736746), -INT32_C(  1774563131) },
      { -INT32_C(  1149353341),  INT32_C(   111747384), -INT32_C(   280182316), -INT32_C(  1948448080),  INT32_C(  1212076192),  INT32_C(   192802720),  INT32_C(  1703165599),  INT32_C(   301710990),
         INT32_C(   969767169),  INT32_C(  1849652890), -INT32_C(   413234377),  INT32_C(   913456021), -INT32_C(  1417760757),  INT32_C(  1052179359), -INT32_C(   928826823), -INT32_C(    86401287) },
      { -INT64_C( 1362214815308041575), -INT64_C(  197719501964580124),  INT64_C(  118828248182989632), -INT64_C( 3286097603805895789),
        -INT64_C(  389958538055285808), -INT64_C(  485973444974328766),  INT64_C( 1808895096988896105), -INT64_C( 1897348682012537958) } },
    { { -INT32_C(  1305237993),  INT32_C(  1394635292), -INT32_C(  1841660163), -INT32_C(   993481543), -INT32_C(    76528036),  INT32_C(  2067408449),  INT32_C(  1514397025), -INT32_C(  1823204228),
        -INT32_C(   549091389), -INT32_C(   164403463), -INT32_C(  1635226140),  INT32_C(  1986154778), -INT32_C(   646786409),  INT32_C(  1515498745),  INT32_C(    95721353),  INT32_C(  1989740723) },
      { -INT32_C(  2007572849),  INT32_C(   696158532),  INT32_C(   281478902), -INT32_C(   276354729), -INT32_C(   171378180),  INT32_C(   776936613),  INT32_C(  1714684851), -INT32_C(  1696740085),
        -INT32_C(   266194005), -INT32_C(  1306943300), -INT32_C(    20717402),  INT32_C(    99437065), -INT32_C(   386222781), -INT32_C(  2078914095),  INT32_C(  1474972236), -INT32_C(  1007565033) },
      {  INT64_C( 2620360356230052057), -INT64_C(  518388480538381026),  INT64_C(   13115235528654480),  INT64_C( 2596713637166968275),
         INT64_C(  146164835948922945),  INT64_C(   33877637303288280),  INT64_C(  249803645596983429),  INT64_C(  141186338067355308) } },
    { { -INT32_C(  1229777926),  INT32_C(  1516883123), -INT32_C(  1252512596), -INT32_C(  1178909322), -INT32_C(   878594566),  INT32_C(  1263515647),  INT32_C(   430127362), -INT32_C(    69430271),
         INT32_C(  1538428840),  INT32_C(   129309531), -INT32_C(  1111683769),  INT32_C(  1282832466),  INT32_C(   739710765), -INT32_C(   797415730), -INT32_C(  1578493024),  INT32_C(  1469892271) },
      { -INT32_C(  1313649066), -INT32_C(  1330026391), -INT32_C(   932350346),  INT32_C(   454419438), -INT32_C(   918016774),  INT32_C(   865714323), -INT32_C(  1965784101), -INT32_C(  1595772854),
         INT32_C(   676435391),  INT32_C(  1943603965),  INT32_C(  1849443968),  INT32_C(   613044522),  INT32_C(   284021373),  INT32_C(  1833142162),  INT32_C(  1425479434),  INT32_C(  1207228808) },
      {  INT64_C( 1615496623877317116),  INT64_C( 1167780552249958216),  INT64_C(  806564549133250084), -INT64_C(  845537529624671562),
         INT64_C( 1040647713911076440), -INT64_C( 2055996840900555392),  INT64_C(  210093667098180345), -INT64_C( 2250109342424468416) } },
    { {  INT32_C(  1819231854), -INT32_C(   773896112), -INT32_C(  1187046513), -INT32_C(   354563732),  INT32_C(   771410843),  INT32_C(  1553612370), -INT32_C(   575565227), -INT32_C(   635132565),
         INT32_C(  1011258603),  INT32_C(  1796023772), -INT32_C(  1390130111), -INT32_C(  1315503594),  INT32_C(   534745805),  INT32_C(   628849104),  INT32_C(  1996696587), -INT32_C(  1118754862) },
      { -INT32_C(  1728473157), -INT32_C(    50133316),  INT32_C(  1772824659), -INT32_C(   149274070), -INT32_C(  1541998124),  INT32_C(  2127204723), -INT32_C(  1862939202), -INT32_C(  1337112844),
        -INT32_C(  1706539043), -INT32_C(  1600697523), -INT32_C(  1660337549),  INT32_C(  1502880901), -INT32_C(  1862358499), -INT32_C(    99628996), -INT32_C(  1987443563),  INT32_C(   674879307) },
      { -INT64_C( 3144493425998343078), -INT64_C( 2104425329626364067), -INT64_C( 1189514072739258532),  INT64_C( 1072243024686328854),
        -INT64_C( 1725752288589136929),  INT64_C( 2308085221288837939), -INT64_C(  995888394746346695), -INT64_C( 3968321779097219481) } },
    { {  INT32_C(  1824686366),  INT32_C(  1074551501),  INT32_C(   568202908),  INT32_C(  1467707962),  INT32_C(  1508407581), -INT32_C(   699140287),  INT32_C(  1180687867), -INT32_C(   747660876),
        -INT32_C(   415289062),  INT32_C(   673729419), -INT32_C(  1689713055), -INT32_C(  1779186568),  INT32_C(  2129582909), -INT32_C(   850116142), -INT32_C(   753617890),  INT32_C(  1738965837) },
      {  INT32_C(  1078977972), -INT32_C(  1838647504), -INT32_C(   181554819), -INT32_C(  1282727818), -INT32_C(   852329989), -INT32_C(   644118853), -INT32_C(  2018726086), -INT32_C(   420523470),
         INT32_C(  1747336759),  INT32_C(   855281333), -INT32_C(  1238948032),  INT32_C(  1131000392), -INT32_C(   418276564),  INT32_C(  1556130850), -INT32_C(  1914409637), -INT32_C(   143404097) },
      {  INT64_C( 1968796394722729752), -INT64_C(  103159976117213652), -INT64_C( 1285661016921246609), -INT64_C( 2383485396536598562),
        -INT64_C(  725649843643230058),  INT64_C( 2093466664136957760), -INT64_C(  890754621929644676),  INT64_C( 1442733351231605930) } },
    { { -INT32_C(   966813167),  INT32_C(  1761106216), -INT32_C(   937549952), -INT32_C(    32732974),  INT32_C(  1172643107),  INT32_C(   614639049), -INT32_C(   760117742),  INT32_C(  1791566937),
        -INT32_C(   416274242),  INT32_C(    21964929),  INT32_C(   432696903),  INT32_C(   420992758), -INT32_C(  1134560013), -INT32_C(  1260387934), -INT32_C(   528051833),  INT32_C(  1951027125) },
      { -INT32_C(   128222601), -INT32_C(   369448286),  INT32_C(   235127832), -INT32_C(  1926751590), -INT32_C(  1186363625),  INT32_C(   258812296),  INT32_C(  1877996730), -INT32_C(  1142736573),
         INT32_C(  1437810355),  INT32_C(    20884969),  INT32_C(   185614705),  INT32_C(  1939355740),  INT32_C(  1194123711), -INT32_C(  1000957686), -INT32_C(   785103475),  INT32_C(   831264638) },
      {  INT64_C(  123967298953787367), -INT64_C(  220444087605464064), -INT64_C( 1391181127251782875), -INT64_C( 1427498633890983660),
        -INT64_C(  598523415667375910),  INT64_C(   80314908004758615), -INT64_C( 1354805013075768243),  INT64_C(  414575329068419675) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mul_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mul_epi32");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mul_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 8259215308803572895), INT64_C( 5002417564910761422),
                            INT64_C( 4825945910792190995), INT64_C(-3854692997504557014),
                            INT64_C(-5029859126276555558), INT64_C(-6821356987634887986),
                            INT64_C( 4779218217009886481), INT64_C( 4803075487209946977)),
      UINT8_C(116),
      easysimd_mm512_set_epi32(INT32_C(-1946312318), INT32_C(  344802157), INT32_C( 1813783552), INT32_C(  136328242),
                            INT32_C( -821129857), INT32_C( -162465728), INT32_C(-2000203088), INT32_C( 1070574927),
                            INT32_C( 1169396458), INT32_C(-1183467128), INT32_C(-1936534523), INT32_C( 1077760263),
                            INT32_C( 1655422492), INT32_C(  623663138), INT32_C(   29163868), INT32_C(-2106787203)),
      easysimd_mm512_set_epi32(INT32_C( 1815416844), INT32_C( 1421058674), INT32_C( -653559880), INT32_C(-1986451398),
                            INT32_C(   72292265), INT32_C( -878074326), INT32_C(    9737947), INT32_C( 1206062441),
                            INT32_C(-1794166530), INT32_C(  629576453), INT32_C(-1534533514), INT32_C(  511428134),
                            INT32_C(  206078130), INT32_C(  438975617), INT32_C(  828494683), INT32_C( -995405524)),
      easysimd_mm512_set_epi64(INT64_C( 8259215308803572895), INT64_C( -270809426907782316),
                            INT64_C(  142656984611699328), INT64_C( 1291180209731016807),
                            INT64_C(-5029859126276555558), INT64_C(  551196920205439242),
                            INT64_C( 4779218217009886481), INT64_C( 4803075487209946977)) },
    { easysimd_mm512_set_epi64(INT64_C(-2096589957716905410), INT64_C( 5843054744461330548),
                            INT64_C( 6974024204978419548), INT64_C( 3764508487718737373),
                            INT64_C( 3631623951055726390), INT64_C( 3085354128371369606),
                            INT64_C(-8195484891056484583), INT64_C( -289672705472788318)),
      UINT8_C(192),
      easysimd_mm512_set_epi32(INT32_C(  750022106), INT32_C( -164409985), INT32_C( 1508401218), INT32_C(  326736787),
                            INT32_C( 1376534770), INT32_C(  433267140), INT32_C(-1811276142), INT32_C( -423374585),
                            INT32_C(  316128811), INT32_C( 1066215241), INT32_C( -742321807), INT32_C(  771493183),
                            INT32_C( 1158801588), INT32_C( 1819952183), INT32_C( 1670004358), INT32_C(-1282166628)),
      easysimd_mm512_set_epi32(INT32_C(-1440467958), INT32_C(  673155116), INT32_C( 1160373089), INT32_C(-1560331288),
                            INT32_C( 1954920850), INT32_C(  443628207), INT32_C(-1358199964), INT32_C( 1999231031),
                            INT32_C( -922196474), INT32_C(-1304354759), INT32_C( 1973769080), INT32_C(-1007908450),
                            INT32_C(-1145320026), INT32_C( 1345168267), INT32_C(-1647348591), INT32_C(-2009668702)),
      easysimd_mm512_set_epi64(INT64_C( -110673422524233260), INT64_C( -509817631696691656),
                            INT64_C( 6974024204978419548), INT64_C( 3764508487718737373),
                            INT64_C( 3631623951055726390), INT64_C( 3085354128371369606),
                            INT64_C(-8195484891056484583), INT64_C( -289672705472788318)) },
    { easysimd_mm512_set_epi64(INT64_C(-5821818953001636176), INT64_C( 1997894375206593641),
                            INT64_C(-1188496888106000468), INT64_C( 4574447963200493304),
                            INT64_C( 3346200385521264609), INT64_C(-5642979348732921527),
                            INT64_C( -695384029725146025), INT64_C(-7115212454065332556)),
      UINT8_C(247),
      easysimd_mm512_set_epi32(INT32_C(  133614784), INT32_C( 1104524722), INT32_C( -405494742), INT32_C(   33506731),
                            INT32_C( 1866794314), INT32_C(-1942831246), INT32_C(-2066004046), INT32_C(-1057119888),
                            INT32_C(-1508387315), INT32_C( 2140894614), INT32_C(-1227481958), INT32_C( -545548506),
                            INT32_C( 2041568161), INT32_C(-1734631316), INT32_C( -578665178), INT32_C(  976865378)),
      easysimd_mm512_set_epi32(INT32_C( 1938874012), INT32_C( -315470352), INT32_C(   25227789), INT32_C(  348972975),
                            INT32_C( -119098852), INT32_C( -117531009), INT32_C(-1256068989), INT32_C( -330670492),
                            INT32_C(-1342579595), INT32_C( 1663139463), INT32_C( 1519977261), INT32_C(-2010660089),
                            INT32_C(  811843811), INT32_C( 1950445467), INT32_C( -632407557), INT32_C(  132507618)),
      easysimd_mm512_set_epi64(INT64_C( -348444802842042144), INT64_C(   11692943599594725),
                            INT64_C(  228342916659107214), INT64_C(  349558353467944896),
                            INT64_C( 3346200385521264609), INT64_C( 1096912607627777034),
                            INT64_C(-3383303787208444572), INT64_C(  129442104345449604)) },
    { easysimd_mm512_set_epi64(INT64_C(-6399381415989804252), INT64_C(-4072646889620133673),
                            INT64_C(-3499367054553152785), INT64_C(-2596410489019354993),
                            INT64_C( 3709612225265967420), INT64_C( 1617021521015256349),
                            INT64_C( 8518903223542129770), INT64_C(-7495998104551122449)),
      UINT8_C(125),
      easysimd_mm512_set_epi32(INT32_C( 1369528234), INT32_C(-2013461915), INT32_C(  979595496), INT32_C(-1220154251),
                            INT32_C(  305231144), INT32_C(  243633364), INT32_C(  911946112), INT32_C(  158189864),
                            INT32_C(  126572094), INT32_C(   -5395242), INT32_C(-1561205257), INT32_C(  515958610),
                            INT32_C( 1545815628), INT32_C( 1042892620), INT32_C( -956025439), INT32_C( -181963588)),
      easysimd_mm512_set_epi32(INT32_C( 1859688708), INT32_C( 1308950804), INT32_C(-1932687023), INT32_C( 2111441590),
                            INT32_C( 1137586884), INT32_C( 1521953186), INT32_C( 1589240826), INT32_C( 1097366673),
                            INT32_C( -573799426), INT32_C( 1360758617), INT32_C(-1894824063), INT32_C( -305567235),
                            INT32_C(-1488593619), INT32_C( 1052029738), INT32_C(-1777311621), INT32_C( -525756513)),
      easysimd_mm512_set_epi64(INT64_C(-6399381415989804252), INT64_C(-2576284431776699090),
                            INT64_C(  370798574555697704), INT64_C(  173592284760002472),
                            INT64_C(   -7341622042300314), INT64_C( -157660045832143350),
                            INT64_C( 8518903223542129770), INT64_C(   95668541519848644)) },
    { easysimd_mm512_set_epi64(INT64_C( 6860124546956220466), INT64_C(-1265261131078623514),
                            INT64_C( 5737379338676836508), INT64_C(-3711065605003334500),
                            INT64_C(-8479853253989282483), INT64_C( 7964407686671565496),
                            INT64_C( 7785652122788440203), INT64_C(-3096894189429138445)),
      UINT8_C(214),
      easysimd_mm512_set_epi32(INT32_C( -814208176), INT32_C( 1449013393), INT32_C(  623550410), INT32_C( -805020885),
                            INT32_C(-1088320756), INT32_C( 2022589200), INT32_C(  839176386), INT32_C( 1343270967),
                            INT32_C(  111940457), INT32_C( 1537061703), INT32_C(-1460061235), INT32_C( 1515709350),
                            INT32_C( 1650058892), INT32_C(   69963651), INT32_C(  758490839), INT32_C(  180779892)),
      easysimd_mm512_set_epi32(INT32_C(-1893053059), INT32_C( -525508532), INT32_C( 1089028030), INT32_C(  641037603),
                            INT32_C(  776284580), INT32_C(  143220066), INT32_C(  609964739), INT32_C(  739061585),
                            INT32_C( 1296320934), INT32_C( 1641387359), INT32_C(  450216201), INT32_C( -102009462),
                            INT32_C(  184231048), INT32_C(   68801332), INT32_C( 1909515723), INT32_C(-1558553543)),
      easysimd_mm512_set_epi64(INT64_C( -761468901003769076), INT64_C( -516048658485338655),
                            INT64_C( 5737379338676836508), INT64_C(  992759969955502695),
                            INT64_C(-8479853253989282483), INT64_C( -154616695341869700),
                            INT64_C(    4813592380383132), INT64_C(-3096894189429138445)) },
    { easysimd_mm512_set_epi64(INT64_C(-2621488480535608616), INT64_C(-6848868720227948061),
                            INT64_C( 6279616399573024356), INT64_C(  745095038278958047),
                            INT64_C(-1323215695156753279), INT64_C( -383012613214998281),
                            INT64_C( 1460565887768366290), INT64_C(-5348367197220594908)),
      UINT8_C( 92),
      easysimd_mm512_set_epi32(INT32_C(-1537831012), INT32_C(-1136146129), INT32_C(  928255499), INT32_C( 1369020603),
                            INT32_C( 1021713905), INT32_C(-1374572733), INT32_C(  981266194), INT32_C( -209600569),
                            INT32_C( -856684622), INT32_C( 1444842251), INT32_C(-1223337348), INT32_C(-1314813402),
                            INT32_C(  630708065), INT32_C( 1782361994), INT32_C(  982404882), INT32_C(  968278192)),
      easysimd_mm512_set_epi32(INT32_C( -560531037), INT32_C( 2016874130), INT32_C( 1909033660), INT32_C( -288062633),
                            INT32_C( 1926487797), INT32_C(-1384808965), INT32_C(  650303852), INT32_C( 1591608188),
                            INT32_C( 1071082983), INT32_C( 1207794171), INT32_C(-2085192565), INT32_C(  656256578),
                            INT32_C(-1465520335), INT32_C( 2093271192), INT32_C(  315880197), INT32_C( 1596114493)),
      easysimd_mm512_set_epi64(INT64_C(-2621488480535608616), INT64_C( -394363679531427699),
                            INT64_C( 6279616399573024356), INT64_C( -333601981829858972),
                            INT64_C( 1745072048772318921), INT64_C( -862854943905058356),
                            INT64_C( 1460565887768366290), INT64_C(-5348367197220594908)) },
    { easysimd_mm512_set_epi64(INT64_C( 1319224608096301911), INT64_C(-6587132379427165760),
                            INT64_C(-1318415648940904266), INT64_C( 5083686936283500523),
                            INT64_C( 2916706726526170303), INT64_C( 1232072806289907439),
                            INT64_C(-4244069429267903156), INT64_C( 1868613955189624367)),
      UINT8_C( 45),
      easysimd_mm512_set_epi32(INT32_C( 1044553244), INT32_C(  448636134), INT32_C(  422274875), INT32_C(-1037497281),
                            INT32_C(  533714637), INT32_C(-1738371545), INT32_C(  -17938559), INT32_C(-1389744139),
                            INT32_C(  827695522), INT32_C(-1482919408), INT32_C( 1233158285), INT32_C(  343037625),
                            INT32_C(-1483824200), INT32_C( -901390751), INT32_C( -727066099), INT32_C( -648215186)),
      easysimd_mm512_set_epi32(INT32_C( 1981159106), INT32_C(  410835312), INT32_C( 2072880481), INT32_C(  105988514),
                            INT32_C(  751462668), INT32_C( 1834849576), INT32_C( -217803098), INT32_C(-1411746849),
                            INT32_C(-1237635210), INT32_C( -311304150), INT32_C( -986441771), INT32_C( 1680967167),
                            INT32_C(  746636010), INT32_C(-2078030023), INT32_C(  843084787), INT32_C(  759454903)),
      easysimd_mm512_set_epi64(INT64_C( 1319224608096301911), INT64_C(-6587132379427165760),
                            INT64_C(-3189650292273714920), INT64_C( 5083686936283500523),
                            INT64_C(  461638965825943200), INT64_C(  576634984670658375),
                            INT64_C(-4244069429267903156), INT64_C( -492290201206756958)) },
    { easysimd_mm512_set_epi64(INT64_C(-1017619325410469279), INT64_C( 7670597165848860921),
                            INT64_C(-5135734722746288063), INT64_C( 8555281953176040262),
                            INT64_C( 2622398452638226743), INT64_C( 2072647407054444460),
                            INT64_C( 5884644356355100584), INT64_C(-3677310731734481669)),
      UINT8_C(226),
      easysimd_mm512_set_epi32(INT32_C(  390006051), INT32_C(  789765807), INT32_C( -514015364), INT32_C( -970761836),
                            INT32_C( -378978470), INT32_C(  -73123202), INT32_C(-1325609418), INT32_C( 1232280698),
                            INT32_C( 1916265121), INT32_C( 1820507576), INT32_C( -792248141), INT32_C( -262685644),
                            INT32_C( 1624847858), INT32_C( -403255584), INT32_C( 1568995237), INT32_C( 1227106212)),
      easysimd_mm512_set_epi32(INT32_C( -769652371), INT32_C( -261880602), INT32_C(   85687930), INT32_C(  432371064),
                            INT32_C( 1626214727), INT32_C( 1845517289), INT32_C(-2002810442), INT32_C(-2069468881),
                            INT32_C(-1294326872), INT32_C(-1409401131), INT32_C(-1446683671), INT32_C( 2011451607),
                            INT32_C( 1570003547), INT32_C(-1564123603), INT32_C( -200447069), INT32_C( -676297563)),
      easysimd_mm512_set_epi64(INT64_C( -206824344976175814), INT64_C( -419729327921913504),
                            INT64_C( -134950133518039378), INT64_C( 8555281953176040262),
                            INT64_C( 2622398452638226743), INT64_C( 2072647407054444460),
                            INT64_C(  630741576975949152), INT64_C(-3677310731734481669)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i  src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i  a = test_vec[i].a;
    easysimd__m512i  b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mul_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mul_epi32");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mul_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
   {  UINT8_C(138),
      easysimd_mm512_set_epi32(INT32_C( 1098716707), INT32_C(-1080185167), INT32_C(  796032668), INT32_C( 1756455873),
                            INT32_C(-1031023150), INT32_C(  313996055), INT32_C(-1552434635), INT32_C(   82580470),
                            INT32_C( -868810524), INT32_C(-1501290792), INT32_C( -628539172), INT32_C(  286404385),
                            INT32_C(-2116183242), INT32_C(  925268541), INT32_C( 1423169798), INT32_C(  472979926)),
      easysimd_mm512_set_epi32(INT32_C(-1589762727), INT32_C( 1342398972), INT32_C( -162164967), INT32_C( 1184007139),
                            INT32_C( 1973410894), INT32_C(  837116435), INT32_C(-1912965227), INT32_C( -221173809),
                            INT32_C(-1524627531), INT32_C(  505638542), INT32_C( 1789154769), INT32_C( 1707140994),
                            INT32_C(  111719139), INT32_C( 1287257616), INT32_C(-1103747425), INT32_C( 1951299418)),
      easysimd_mm512_set_epi64(INT64_C(-1450039457750448324), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( -759110487184905264), INT64_C(                   0),
                            INT64_C( 1191058976247458256), INT64_C(                   0)) },
   {  UINT8_C(226),
      easysimd_mm512_set_epi32(INT32_C( 1851172912), INT32_C(  432012768), INT32_C(-1336678725), INT32_C(  141506650),
                            INT32_C(  576471669), INT32_C(-2021849973), INT32_C(  610549751), INT32_C(  470887358),
                            INT32_C( 1210740282), INT32_C( -720782218), INT32_C(  967227355), INT32_C(-1907082749),
                            INT32_C( -376079371), INT32_C(  615957162), INT32_C(  189423181), INT32_C(  750118943)),
      easysimd_mm512_set_epi32(INT32_C(-1194827437), INT32_C( 1644918495), INT32_C(-1387747393), INT32_C(-1434123267),
                            INT32_C( 1354817839), INT32_C( 1324343890), INT32_C( -595811004), INT32_C(-1790143018),
                            INT32_C( -914188665), INT32_C( -647124032), INT32_C(  792952903), INT32_C( 2106780254),
                            INT32_C(  -65103351), INT32_C( -572558150), INT32_C( -801231269), INT32_C( -650481868)),
      easysimd_mm512_set_epi64(INT64_C(  710625792159344160), INT64_C( -202937979200225550),
                            INT64_C(-2677624658239214970), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( -352671293153970300), INT64_C(                   0)) },
   {  UINT8_C(205),
      easysimd_mm512_set_epi32(INT32_C( 1389849027), INT32_C( -968733779), INT32_C( -903985535), INT32_C( -505052458),
                            INT32_C(  849997016), INT32_C( -665823569), INT32_C(  843681453), INT32_C( 1241052856),
                            INT32_C( 1218361488), INT32_C( -511802096), INT32_C( 2056154947), INT32_C( -475453332),
                            INT32_C( 1793883682), INT32_C( 1281268084), INT32_C(-1443305318), INT32_C(-2002775301)),
      easysimd_mm512_set_epi32(INT32_C( -480306273), INT32_C(-1942698584), INT32_C( 1404753532), INT32_C(  237623409),
                            INT32_C( 1343658265), INT32_C(-1947335016), INT32_C( 1618135889), INT32_C(  726476998),
                            INT32_C( 1324737144), INT32_C( 1048817456), INT32_C(  319312471), INT32_C( 1894816689),
                            INT32_C( 1848939745), INT32_C(-1295730322), INT32_C( 1089929027), INT32_C( -785534579)),
      easysimd_mm512_set_epi64(INT64_C( 1881957740736268936), INT64_C( -120012286793789322),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( -536786972302187776), INT64_C( -900896908314257748),
                            INT64_C(                   0), INT64_C( 1573249252902633279)) },
   {  UINT8_C(206),
      easysimd_mm512_set_epi32(INT32_C(  163723168), INT32_C(   94537413), INT32_C( 1298848275), INT32_C(  -99870655),
                            INT32_C( 1537532032), INT32_C(-1949556986), INT32_C( -894015664), INT32_C(-1324496729),
                            INT32_C(  850348293), INT32_C(  906352618), INT32_C(-1965873722), INT32_C(-2107953605),
                            INT32_C(  559881293), INT32_C(   -5815681), INT32_C(-1173896203), INT32_C( 1760080316)),
      easysimd_mm512_set_epi32(INT32_C( 1202706763), INT32_C(-1110213669), INT32_C( 1229627598), INT32_C( -147072860),
                            INT32_C( 1883759514), INT32_C(-1191387298), INT32_C( 1673499534), INT32_C(  640453183),
                            INT32_C(-1171836364), INT32_C( -982522972), INT32_C(-1480196612), INT32_C(-2077854762),
                            INT32_C( -872251595), INT32_C( 1154127488), INT32_C(  896971913), INT32_C(-1585180342)),
      easysimd_mm512_set_epi64(INT64_C( -104956728144498297), INT64_C(   14688262860923300),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( -890512267917340696), INT64_C( 4380021436224317010),
                            INT64_C(   -6712037303539328), INT64_C(                   0)) },
   {  UINT8_C(197),
      easysimd_mm512_set_epi32(INT32_C( -650041052), INT32_C( -647184441), INT32_C(-1880618021), INT32_C(-1812158288),
                            INT32_C(  264100196), INT32_C( -263091932), INT32_C(-1182134909), INT32_C(-1890025577),
                            INT32_C( 1421768266), INT32_C(  936126513), INT32_C( -213174057), INT32_C( -588951079),
                            INT32_C(  217188364), INT32_C( 1950574682), INT32_C( -530860484), INT32_C(  713083418)),
      easysimd_mm512_set_epi32(INT32_C(-1419088193), INT32_C(  155768054), INT32_C(  575537364), INT32_C(-1651547513),
                            INT32_C(-1310582959), INT32_C( 1366625247), INT32_C(  375333442), INT32_C(  516971366),
                            INT32_C( 1204467496), INT32_C(-1684524880), INT32_C( 2029390656), INT32_C( 1244178650),
                            INT32_C(-1765716319), INT32_C(  -62663523), INT32_C(  233795696), INT32_C(  614711137)),
      easysimd_mm512_set_epi64(INT64_C( -100810660953647814), INT64_C( 2992865513708737744),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( -732760358386263350),
                            INT64_C(                   0), INT64_C(  438340318654626266)) },
   {  UINT8_C(221),
      easysimd_mm512_set_epi32(INT32_C(-1439155961), INT32_C(  680420491), INT32_C( 1277792130), INT32_C(-1440597466),
                            INT32_C( -965757835), INT32_C( 2017786190), INT32_C(-1072056911), INT32_C( -897882665),
                            INT32_C(    1375688), INT32_C( 1420052414), INT32_C( -331914389), INT32_C(-1649119241),
                            INT32_C( -706010264), INT32_C(  713383150), INT32_C(  896627462), INT32_C( 1020243588)),
      easysimd_mm512_set_epi32(INT32_C( -359263092), INT32_C(-1196237833), INT32_C( 1322331949), INT32_C(   43567177),
                            INT32_C(  414081468), INT32_C( 1462500900), INT32_C( 1902422273), INT32_C( -902219192),
                            INT32_C(-1637598569), INT32_C( -626716515), INT32_C( 1485910176), INT32_C(-1246912099),
                            INT32_C(-1032397276), INT32_C(-1436895618), INT32_C(-1408100359), INT32_C( -678052935)),
      easysimd_mm512_set_epi64(INT64_C( -813944733682636003), INT64_C(  -62762764786973482),
                            INT64_C(                   0), INT64_C(  810086972527106680),
                            INT64_C( -889970300019417210), INT64_C( 2056306734296596859),
                            INT64_C(                   0), INT64_C( -691779159258330780)) },
   {  UINT8_C(176),
      easysimd_mm512_set_epi32(INT32_C(  806025559), INT32_C(  277323133), INT32_C(-2040845209), INT32_C( 1514048177),
                            INT32_C( 1299301232), INT32_C( 1804349866), INT32_C(  505045603), INT32_C(-1270991510),
                            INT32_C(-1365476185), INT32_C( -470279784), INT32_C( 1957249393), INT32_C(  966280187),
                            INT32_C(  550173580), INT32_C( 1419279519), INT32_C(  120074737), INT32_C( -623354205)),
      easysimd_mm512_set_epi32(INT32_C(  182628708), INT32_C( -711074484), INT32_C(  700640568), INT32_C( -182451726),
                            INT32_C( 1928956599), INT32_C( 1423054326), INT32_C( 1016030809), INT32_C(-1086945734),
                            INT32_C( 1392670038), INT32_C(    3796661), INT32_C( -232044152), INT32_C( 1236628648),
                            INT32_C(   19789106), INT32_C(-2055501126), INT32_C( 1118019036), INT32_C( -365745616)),
      easysimd_mm512_set_epi64(INT64_C( -197197403699238372), INT64_C(                   0),
                            INT64_C( 2567687882428820316), INT64_C( 1381498799744718340),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
   {  UINT8_C(158),
      easysimd_mm512_set_epi32(INT32_C( 1841667236), INT32_C( 1484771017), INT32_C( -884031658), INT32_C(-1839125718),
                            INT32_C(-1569039961), INT32_C( 1041098150), INT32_C( 1388925681), INT32_C(  863701002),
                            INT32_C(  128435058), INT32_C( -263295419), INT32_C(-1184146866), INT32_C( 1159115917),
                            INT32_C(  866281726), INT32_C(-1295662984), INT32_C( -351675537), INT32_C( -710944336)),
      easysimd_mm512_set_epi32(INT32_C(-1968953227), INT32_C(  227585281), INT32_C( -737334168), INT32_C( 1230090038),
                            INT32_C(-1805794302), INT32_C( 1379277168), INT32_C(  744356262), INT32_C(-1333512317),
                            INT32_C( -486348180), INT32_C( 1206532716), INT32_C( 1803086042), INT32_C(-1291499422),
                            INT32_C( 1358104641), INT32_C(-2056773451), INT32_C(-1326911147), INT32_C( 1579123656)),
      easysimd_mm512_set_epi64(INT64_C(  337912029124600777), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-1151755924372241634),
                            INT64_C( -317674536996428004), INT64_C(-1496997536836499974),
                            INT64_C( 2664885226934637784), INT64_C(                   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i  a = test_vec[i].a;
    easysimd__m512i  b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mul_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mul_epi32");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mul_epu32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
   {  UINT8_C(166),
      easysimd_x_mm512_set_epu32(UINT32_C(4120514587), UINT32_C(1586964835), UINT32_C(1689003642), UINT32_C(2702971618),
                              UINT32_C(2798377561), UINT32_C( 356472812), UINT32_C(2899999566), UINT32_C(3229978818),
                              UINT32_C(1853356574), UINT32_C(  66069374), UINT32_C(1309687627), UINT32_C(1439779852),
                              UINT32_C(  51916795), UINT32_C(2259276195), UINT32_C( 994758469), UINT32_C(3337558808)),
      easysimd_x_mm512_set_epu32(UINT32_C(2258908256), UINT32_C(3395137062), UINT32_C(4215249296), UINT32_C(2133148800),
                              UINT32_C(2933274444), UINT32_C(1851927716), UINT32_C(4190919751), UINT32_C(1746820983),
                              UINT32_C( 828520274), UINT32_C( 635229603), UINT32_C(1544367040), UINT32_C(1918376547),
                              UINT32_C( 892222123), UINT32_C(3243361756), UINT32_C( 803359976), UINT32_C( 163739728)),
      easysimd_x_mm512_set_epu64(UINT64_C( 5387963127399214770), UINT64_C(                   0),
                              UINT64_C(  660161880543257392), UINT64_C(                   0),
                              UINT64_C(                   0), UINT64_C( 2762039900919931044),
                              UINT64_C( 7327650007104198420), UINT64_C(                   0)) },
   {  UINT8_C(219),
      easysimd_x_mm512_set_epu32(UINT32_C(1410010955), UINT32_C( 503921354), UINT32_C(1418189156), UINT32_C( 444221777),
                              UINT32_C( 148285537), UINT32_C(2288722231), UINT32_C( 344338098), UINT32_C(3454728003),
                              UINT32_C(1478480780), UINT32_C(1743148264), UINT32_C(3822764711), UINT32_C(1636469832),
                              UINT32_C(3280064546), UINT32_C( 933016241), UINT32_C(1726799481), UINT32_C(3824577952)),
      easysimd_x_mm512_set_epu32(UINT32_C(1037963842), UINT32_C(2512709916), UINT32_C( 937723538), UINT32_C(2708935661),
                              UINT32_C(3914488889), UINT32_C(2550814880), UINT32_C(2900652427), UINT32_C(2176349091),
                              UINT32_C(3028439158), UINT32_C(2773320535), UINT32_C( 444719300), UINT32_C(3233757255),
                              UINT32_C(1004532908), UINT32_C(  43390785), UINT32_C( 738451500), UINT32_C(3536836475)),
      easysimd_x_mm512_set_epu64(UINT64_C( 1266208183079946264), UINT64_C( 1203368213108089597),
                              UINT64_C(                   0), UINT64_C( 7518694148981295273),
                              UINT64_C( 4834308876100801240), UINT64_C(                   0),
                              UINT64_C(   40484307114739185), UINT64_C(13526906802114399200)) },
   {  UINT8_C(194),
      easysimd_x_mm512_set_epu32(UINT32_C(1176316177), UINT32_C(1751826934), UINT32_C(3378345958), UINT32_C( 543404964),
                              UINT32_C(2579785136), UINT32_C(2416322328), UINT32_C(  75139728), UINT32_C(2416880998),
                              UINT32_C(4234686409), UINT32_C(2660004756), UINT32_C(2106185379), UINT32_C( 797059438),
                              UINT32_C(2372191392), UINT32_C( 269555244), UINT32_C(1767269404), UINT32_C(1625455101)),
      easysimd_x_mm512_set_epu32(UINT32_C(1851434759), UINT32_C(3604871411), UINT32_C(1327258228), UINT32_C(1673018657),
                              UINT32_C( 305970996), UINT32_C(2819644549), UINT32_C(3721065063), UINT32_C(1020891301),
                              UINT32_C(4202682046), UINT32_C(4200645833), UINT32_C( 117038385), UINT32_C(3020070100),
                              UINT32_C(1353160823), UINT32_C( 440057333), UINT32_C(3056423613), UINT32_C(1896622192)),
      easysimd_x_mm512_set_epu64(UINT64_C( 6315110831396383874), UINT64_C(  909126643078413348),
                              UINT64_C(                   0), UINT64_C(                   0),
                              UINT64_C(                   0), UINT64_C(                   0),
                              UINT64_C(  118619761770804252), UINT64_C(                   0)) },
   {  UINT8_C( 67),
      easysimd_x_mm512_set_epu32(UINT32_C(2693954212), UINT32_C( 277998850), UINT32_C(4169077113), UINT32_C(3680111513),
                              UINT32_C(4156583103), UINT32_C(4105987148), UINT32_C(  11818996), UINT32_C( 514873926),
                              UINT32_C(1191268288), UINT32_C(3638344486), UINT32_C(2361786195), UINT32_C( 500533201),
                              UINT32_C(3058957194), UINT32_C( 190737734), UINT32_C(3837187385), UINT32_C(4003123598)),
      easysimd_x_mm512_set_epu32(UINT32_C(4122956852), UINT32_C(3007076678), UINT32_C(1011742851), UINT32_C( 831857768),
                              UINT32_C(2217989187), UINT32_C(3681606305), UINT32_C(3147415754), UINT32_C( 236426985),
                              UINT32_C( 216160186), UINT32_C(3279967715), UINT32_C(3874145825), UINT32_C(2203854710),
                              UINT32_C(3676418261), UINT32_C( 401038296), UINT32_C(3825112812), UINT32_C(2929607534)),
      easysimd_x_mm512_set_epu64(UINT64_C(                   0), UINT64_C( 3061329349195282984),
                              UINT64_C(                   0), UINT64_C(                   0),
                              UINT64_C(                   0), UINT64_C(                   0),
                              UINT64_C(   76493135826261264), UINT64_C(11727581052233987332)) },
   {  UINT8_C(169),
      easysimd_x_mm512_set_epu32(UINT32_C( 428651380), UINT32_C(1050238262), UINT32_C(4167120113), UINT32_C( 669121916),
                              UINT32_C(1457539263), UINT32_C(3520615042), UINT32_C(1174118849), UINT32_C(1102257957),
                              UINT32_C(1414101989), UINT32_C(3097425534), UINT32_C(1024087984), UINT32_C(1792583521),
                              UINT32_C(3354797839), UINT32_C( 580554502), UINT32_C(1472515666), UINT32_C(3870057603)),
      easysimd_x_mm512_set_epu32(UINT32_C(2083050486), UINT32_C(1796942025), UINT32_C( 755961532), UINT32_C(2214717680),
                              UINT32_C(3706324798), UINT32_C(1039769945), UINT32_C(3555811997), UINT32_C( 761202637),
                              UINT32_C(1863011574), UINT32_C(1454498620), UINT32_C( 958628441), UINT32_C(2380256526),
                              UINT32_C(4218133731), UINT32_C(3449338768), UINT32_C(3115502206), UINT32_C(2279816507)),
      easysimd_x_mm512_set_epu64(UINT64_C( 1887217269250760550), UINT64_C(                   0),
                              UINT64_C( 3660629708586512690), UINT64_C(                   0),
                              UINT64_C( 4505201164755763080), UINT64_C(                   0),
                              UINT64_C(                   0), UINT64_C( 8823021206360252721)) },
   {  UINT8_C(203),
      easysimd_x_mm512_set_epu32(UINT32_C( 968785729), UINT32_C(3446816529), UINT32_C(1989948608), UINT32_C(3935090572),
                              UINT32_C(2260595137), UINT32_C(3809743538), UINT32_C(1768049062), UINT32_C(1253090843),
                              UINT32_C(4000901225), UINT32_C(2487234584), UINT32_C( 840765913), UINT32_C(1202598978),
                              UINT32_C(2886819484), UINT32_C(2063363126), UINT32_C(2370412425), UINT32_C(1978444200)),
      easysimd_x_mm512_set_epu32(UINT32_C(1758358159), UINT32_C( 259726788), UINT32_C(1062244813), UINT32_C(1397736159),
                              UINT32_C(1484315275), UINT32_C(2101001099), UINT32_C(2659688367), UINT32_C(1816554597),
                              UINT32_C(1863116741), UINT32_C(3211066307), UINT32_C( 496281550), UINT32_C(3010953410),
                              UINT32_C(1914417911), UINT32_C(1058492483), UINT32_C(1785378717), UINT32_C(1261746977)),
      easysimd_x_mm512_set_epu64(UINT64_C(  895230585902478852), UINT64_C( 5500218381424392948),
                              UINT64_C(                   0), UINT64_C(                   0),
                              UINT64_C( 7986675170287561288), UINT64_C(                   0),
                              UINT64_C( 2184054358570381858), UINT64_C( 2496295988513183400)) },
   {  UINT8_C( 89),
      easysimd_x_mm512_set_epu32(UINT32_C( 244202415), UINT32_C(1696418382), UINT32_C(4253734840), UINT32_C(1521382913),
                              UINT32_C(2523120367), UINT32_C( 719365215), UINT32_C( 746887847), UINT32_C( 329869757),
                              UINT32_C(2935442647), UINT32_C(3965449572), UINT32_C(   2046702), UINT32_C(3055578856),
                              UINT32_C(2614828885), UINT32_C(2261447742), UINT32_C( 379053160), UINT32_C(1474182998)),
      easysimd_x_mm512_set_epu32(UINT32_C( 180314942), UINT32_C(3784268734), UINT32_C(2189933725), UINT32_C(1759707651),
                              UINT32_C(4017470040), UINT32_C( 528482752), UINT32_C(2637497058), UINT32_C(3574995683),
                              UINT32_C(2110412704), UINT32_C( 661885013), UINT32_C(3935066909), UINT32_C( 163101530),
                              UINT32_C(3963037657), UINT32_C( 399559486), UINT32_C( 875430591), UINT32_C(1854318955)),
      easysimd_x_mm512_set_epu64(UINT64_C(                   0), UINT64_C( 2677189152106767363),
                              UINT64_C(                   0), UINT64_C( 1179282957227259031),
                              UINT64_C( 2624671641514064436), UINT64_C(                   0),
                              UINT64_C(                   0), UINT64_C( 2733605476330127090)) },
   {  UINT8_C( 27),
      easysimd_x_mm512_set_epu32(UINT32_C(3300721541), UINT32_C(3440866090), UINT32_C(3838602911), UINT32_C(1016597887),
                              UINT32_C( 287068752), UINT32_C(1521867279), UINT32_C(2420112012), UINT32_C(2417142414),
                              UINT32_C( 344709524), UINT32_C(1803316517), UINT32_C( 467213234), UINT32_C( 327864893),
                              UINT32_C(2661940215), UINT32_C(4228328219), UINT32_C(1396080639), UINT32_C(4001917131)),
      easysimd_x_mm512_set_epu32(UINT32_C(4155157678), UINT32_C(1774567103), UINT32_C(1949309963), UINT32_C( 729844445),
                              UINT32_C(2587732272), UINT32_C( 138621029), UINT32_C(2703994882), UINT32_C(1904478113),
                              UINT32_C(2402800240), UINT32_C( 959065024), UINT32_C(2558227042), UINT32_C(3067418732),
                              UINT32_C(1398342314), UINT32_C(3263383247), UINT32_C(3963437622), UINT32_C(1585677583)),
      easysimd_x_mm512_set_epu64(UINT64_C(                   0), UINT64_C(                   0),
                              UINT64_C(                   0), UINT64_C( 4603394823466984782),
                              UINT64_C( 1729497798656201408), UINT64_C(                   0),
                              UINT64_C(13798655472701947093), UINT64_C( 6345750283650374373)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i  a = test_vec[i].a;
    easysimd__m512i  b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mul_epu32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mul_epu32");
    easysimd_assert_m512i_u64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mul_epu32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_x_mm512_set_epu32(UINT32_C( 768255153), UINT32_C(3116504916), UINT32_C(2849349603), UINT32_C(3380602699),
                              UINT32_C(3667150171), UINT32_C(2606748140), UINT32_C( 256440763), UINT32_C(4236376754),
                              UINT32_C( 137611130), UINT32_C(3608004165), UINT32_C(  23379469), UINT32_C( 634104346),
                              UINT32_C( 752813209), UINT32_C( 304980816), UINT32_C( 873141861), UINT32_C(  42552677)),
      easysimd_x_mm512_set_epu32(UINT32_C(3537218346), UINT32_C( 554508152), UINT32_C(2540529856), UINT32_C(1492162297),
                              UINT32_C(4027600118), UINT32_C( 897760677), UINT32_C( 832131936), UINT32_C(1226979254),
                              UINT32_C(2823494192), UINT32_C(1370605687), UINT32_C(3043623483), UINT32_C( 310819447),
                              UINT32_C(3262156609), UINT32_C(4133822383), UINT32_C( 153673388), UINT32_C( 507486240)),
      easysimd_x_mm512_set_epu64(UINT64_C( 1728127381670075232), UINT64_C( 5044407888584239603),
                              UINT64_C( 2340235974934890780), UINT64_C( 5197946389285861516),
                              UINT64_C( 4945151027268686355), UINT64_C(  197091962164016662),
                              UINT64_C( 1260736523566404528), UINT64_C(   21594898052664480)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(3038228522), UINT32_C(2177263565), UINT32_C(1580156717), UINT32_C(3293644153),
                              UINT32_C(3877520946), UINT32_C(1914222601), UINT32_C(2300352870), UINT32_C(3239916612),
                              UINT32_C(2045429998), UINT32_C(2834457902), UINT32_C(2502406118), UINT32_C(3874567768),
                              UINT32_C(1624909929), UINT32_C( 552025498), UINT32_C(3781080866), UINT32_C( 445279347)),
      easysimd_x_mm512_set_epu32(UINT32_C(3686614578), UINT32_C(2012284249), UINT32_C(4065027833), UINT32_C(2391570441),
                              UINT32_C(2365238876), UINT32_C(1345195249), UINT32_C(3876854758), UINT32_C(2935715346),
                              UINT32_C( 298066676), UINT32_C( 616522972), UINT32_C(2787002250), UINT32_C(2890879290),
                              UINT32_C(2445030057), UINT32_C(1418356119), UINT32_C( 876914337), UINT32_C(2884175418)),
      easysimd_x_mm512_set_epu64(UINT64_C( 4381273177771087685), UINT64_C( 7876981999487281473),
                              UINT64_C( 2575003148393622649), UINT64_C( 9511472917608727752),
                              UINT64_C( 1747508409749924744), UINT64_C(11200907718212724720),
                              UINT64_C(  782968742932322262), UINT64_C( 1284263746760492046)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(1327609198), UINT32_C(1911763444), UINT32_C(3596479631), UINT32_C(3933061513),
                              UINT32_C(3463498323), UINT32_C( 687111330), UINT32_C( 635542403), UINT32_C(1153148129),
                              UINT32_C(3815988413), UINT32_C(2677217701), UINT32_C( 271158343), UINT32_C(1601359912),
                              UINT32_C(4276983578), UINT32_C(2555948345), UINT32_C(1846034446), UINT32_C(1127199678)),
      easysimd_x_mm512_set_epu32(UINT32_C( 657487984), UINT32_C(2570668084), UINT32_C( 753507331), UINT32_C(1705841966),
                              UINT32_C(3937986766), UINT32_C(3019564702), UINT32_C(2409588030), UINT32_C( 467930148),
                              UINT32_C(4115653696), UINT32_C(3587871831), UINT32_C(1753201197), UINT32_C( 778129766),
                              UINT32_C(1742436576), UINT32_C(2505253535), UINT32_C(2950628411), UINT32_C(2064165361)),
      easysimd_x_mm512_set_epu64(UINT64_C( 4914509269648721296), UINT64_C( 6709181383734854558),
                              UINT64_C( 2074777118412273660), UINT64_C(  539592774668893092),
                              UINT64_C( 9605513974872480531), UINT64_C( 1246065813606340592),
                              UINT64_C( 6403298626588649575), UINT64_C( 2326726530257953758)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(2919345837), UINT32_C(4236345846), UINT32_C(3784567990), UINT32_C(4027374119),
                              UINT32_C(3288320277), UINT32_C( 390224653), UINT32_C(3910835486), UINT32_C( 865013699),
                              UINT32_C(2748043226), UINT32_C( 449666617), UINT32_C( 232557914), UINT32_C( 838031623),
                              UINT32_C(3874792609), UINT32_C( 112848728), UINT32_C( 173110782), UINT32_C(1917463852)),
      easysimd_x_mm512_set_epu32(UINT32_C(1885981625), UINT32_C( 167105269), UINT32_C(1642818873), UINT32_C( 795737603),
                              UINT32_C( 583404702), UINT32_C(1590901338), UINT32_C(2535047020), UINT32_C(1665933152),
                              UINT32_C( 271178891), UINT32_C( 492170774), UINT32_C(4067923128), UINT32_C(2457242362),
                              UINT32_C(4206213856), UINT32_C(1232486026), UINT32_C(2718472187), UINT32_C(4168006676)),
      easysimd_x_mm512_set_epu64(UINT64_C(  707915712172862574), UINT64_C( 3204733027837296757),
                              UINT64_C(  620808922578285714), UINT64_C( 1441054998098249248),
                              UINT64_C(  221312766930851558), UINT64_C( 2059246804731213526),
                              UINT64_C(  139084480311874928), UINT64_C( 7992002136124675952)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(  27104904), UINT32_C(4150065749), UINT32_C(3990632930), UINT32_C( 634032004),
                              UINT32_C(2048919564), UINT32_C(1865014244), UINT32_C( 549754386), UINT32_C(2522098959),
                              UINT32_C(2696620961), UINT32_C( 891563523), UINT32_C(2188909902), UINT32_C(2179241133),
                              UINT32_C(1743310130), UINT32_C( 377093787), UINT32_C(2755680804), UINT32_C(3712100521)),
      easysimd_x_mm512_set_epu32(UINT32_C( 672807047), UINT32_C(2773804867), UINT32_C(4088841569), UINT32_C( 619049193),
                              UINT32_C( 593052350), UINT32_C( 730103388), UINT32_C(1414198306), UINT32_C(4002179273),
                              UINT32_C(1269785901), UINT32_C( 747703241), UINT32_C( 347333415), UINT32_C( 968811996),
                              UINT32_C(4236163540), UINT32_C(2123412997), UINT32_C(  33837735), UINT32_C(1851248845)),
      easysimd_x_mm512_set_epu64(UINT64_C(11511472572946200383), UINT64_C(  392497000412372772),
                              UINT64_C( 1361653218212658672), UINT64_C(10093892178164676807),
                              UINT64_C(  666624935704478043), UINT64_C( 2111274951827031468),
                              UINT64_C(  800725848403749639), UINT64_C( 6872021802025148245)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(3744995587), UINT32_C(2704878999), UINT32_C(2216207729), UINT32_C(3174220609),
                              UINT32_C(2276590134), UINT32_C( 284809778), UINT32_C(2003404586), UINT32_C(1707085270),
                              UINT32_C(2713648433), UINT32_C(2786430472), UINT32_C( 397019195), UINT32_C( 630796576),
                              UINT32_C(1959866953), UINT32_C( 629006272), UINT32_C(2429347726), UINT32_C(3247824799)),
      easysimd_x_mm512_set_epu32(UINT32_C(3020299794), UINT32_C(2488516068), UINT32_C(3326847413), UINT32_C(1426347053),
                              UINT32_C(3015511399), UINT32_C( 258677619), UINT32_C(3923020384), UINT32_C( 835454201),
                              UINT32_C(3228303109), UINT32_C( 994730831), UINT32_C(2437482082), UINT32_C(1004732602),
                              UINT32_C(3078918689), UINT32_C(1633253517), UINT32_C(1920589043), UINT32_C(3888518352)),
      easysimd_x_mm512_set_epu64(UINT64_C( 6731134851007255932), UINT64_C( 4527540211219015277),
                              UINT64_C(   73673915240958582), UINT64_C( 1426191560286719270),
                              UINT64_C( 2771748298936282232), UINT64_C(  633781885137170752),
                              UINT64_C( 1027326705959058624), UINT64_C(12629226334992211248)) },
    { easysimd_x_mm512_set_epu32(UINT32_C( 237961802), UINT32_C(1124052031), UINT32_C(3408632402), UINT32_C(1936321731),
                              UINT32_C(3188356992), UINT32_C( 413227284), UINT32_C(1767960975), UINT32_C(2214647351),
                              UINT32_C(4011124733), UINT32_C(3189426671), UINT32_C(3040561164), UINT32_C(3376223700),
                              UINT32_C(2268266209), UINT32_C( 155837480), UINT32_C(1377610501), UINT32_C(1504228568)),
      easysimd_x_mm512_set_epu32(UINT32_C(1573768507), UINT32_C( 476780671), UINT32_C(2153500842), UINT32_C(1201914669),
                              UINT32_C(1130822801), UINT32_C(3370243267), UINT32_C(1286308912), UINT32_C(2062398363),
                              UINT32_C(1095401713), UINT32_C(4089334856), UINT32_C(2597794703), UINT32_C(2139321595),
                              UINT32_C(2505322640), UINT32_C(2764790171), UINT32_C(3415336749), UINT32_C(  59419438)),
      easysimd_x_mm512_set_epu64(UINT64_C(  535926281579092801), UINT64_C( 2327293492392372039),
                              UINT64_C( 1392676471641696828), UINT64_C( 4567485071324686413),
                              UINT64_C(13042633656376344376), UINT64_C( 7222828270960801500),
                              UINT64_C(  430857932977409080), UINT64_C(   89380416134104784)) },
    { easysimd_x_mm512_set_epu32(UINT32_C( 493235400), UINT32_C( 189383962), UINT32_C(2622533649), UINT32_C( 943550019),
                              UINT32_C( 227224723), UINT32_C(1724057992), UINT32_C(4133039778), UINT32_C(3416450213),
                              UINT32_C(1064097074), UINT32_C(1615527431), UINT32_C( 106890087), UINT32_C(3131878508),
                              UINT32_C(4228916541), UINT32_C(2298347901), UINT32_C(2681451816), UINT32_C( 956711717)),
      easysimd_x_mm512_set_epu32(UINT32_C(1959431707), UINT32_C(3425635109), UINT32_C(3493232750), UINT32_C(2950665544),
                              UINT32_C(1223627161), UINT32_C(3625235337), UINT32_C( 456501342), UINT32_C(2245318318),
                              UINT32_C(3915087897), UINT32_C(4086538960), UINT32_C(2510683850), UINT32_C(3689243003),
                              UINT32_C(2734380582), UINT32_C(3715382302), UINT32_C(3353219492), UINT32_C(1828163673)),
      easysimd_x_mm512_set_epu64(UINT64_C(  648760349308721858), UINT64_C( 2784100530103845336),
                              UINT64_C( 6250115955635663304), UINT64_C( 7671018245783901734),
                              UINT64_C( 6601915787730211760), UINT64_C(11554260871885079524),
                              UINT64_C( 8539241115214248102), UINT64_C( 1749025606552856541)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mul_epu32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mul_epu32");
    easysimd_assert_m512i_u64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mul_epu32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_x_mm512_set_epu64(UINT64_C(11617731129322750966), UINT64_C( 2428231924375211538),
                              UINT64_C(14175135673172244792), UINT64_C( 5480651963328574733),
                              UINT64_C(12032129819668007160), UINT64_C( 4424822542185790875),
                              UINT64_C(14867552050688999361), UINT64_C(18178319731812771647)),
      UINT8_C(193),
      easysimd_x_mm512_set_epu32(UINT32_C(3263550415), UINT32_C(3722438839), UINT32_C(2250736680), UINT32_C(1691034658),
                              UINT32_C(2554266733), UINT32_C(3217520562), UINT32_C(1461265118), UINT32_C( 788514619),
                              UINT32_C(2189225773), UINT32_C(1631863219), UINT32_C(1750226365), UINT32_C( 568487836),
                              UINT32_C(2400146531), UINT32_C(3640095823), UINT32_C( 206125598), UINT32_C( 929630688)),
      easysimd_x_mm512_set_epu32(UINT32_C(2599352978), UINT32_C(2499798093), UINT32_C(1296552276), UINT32_C( 187401912),
                              UINT32_C(2689124664), UINT32_C(3631388300), UINT32_C( 697343737), UINT32_C(2062287739),
                              UINT32_C(2446945975), UINT32_C( 568195668), UINT32_C( 631862630), UINT32_C( 893101483),
                              UINT32_C( 859175541), UINT32_C( 104178488), UINT32_C(4045255037), UINT32_C(4203900757)),
      easysimd_x_mm512_set_epu64(UINT64_C( 9305345511041334027), UINT64_C(  316903128167466096),
                              UINT64_C(14175135673172244792), UINT64_C( 5480651963328574733),
                              UINT64_C(12032129819668007160), UINT64_C( 4424822542185790875),
                              UINT64_C(14867552050688999361), UINT64_C( 3908075153013630816)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(13472523368263323530), UINT64_C( 9759174632444686247),
                              UINT64_C(13193200280295594145), UINT64_C(   61830035715779390),
                              UINT64_C( 3749566472430999385), UINT64_C( 4387274564618060685),
                              UINT64_C( 6292382073951294857), UINT64_C( 2998314933539498774)),
      UINT8_C( 51),
      easysimd_x_mm512_set_epu32(UINT32_C(2328158325), UINT32_C(4044751844), UINT32_C(3983880162), UINT32_C(1865776280),
                              UINT32_C( 663966634), UINT32_C(3835216878), UINT32_C(1416309334), UINT32_C(3569688362),
                              UINT32_C( 671765842), UINT32_C(4201434406), UINT32_C(2145277385), UINT32_C( 232005336),
                              UINT32_C(3364267605), UINT32_C(1204199577), UINT32_C(1905702483), UINT32_C(1405245895)),
      easysimd_x_mm512_set_epu32(UINT32_C(1978530737), UINT32_C(3532128238), UINT32_C(2586558058), UINT32_C(2555375701),
                              UINT32_C(1328514887), UINT32_C(2070014178), UINT32_C(2738790052), UINT32_C(1824660691),
                              UINT32_C(3424488035), UINT32_C(3798301173), UINT32_C( 335648721), UINT32_C( 829536855),
                              UINT32_C( 101359129), UINT32_C( 500535839), UINT32_C(3768468917), UINT32_C(3000753624)),
      easysimd_x_mm512_set_epu64(UINT64_C(13472523368263323530), UINT64_C( 9759174632444686247),
                              UINT64_C( 7938953313164896284), UINT64_C( 6513470033261578142),
                              UINT64_C( 3749566472430999385), UINT64_C( 4387274564618060685),
                              UINT64_C(  602745045597140103), UINT64_C( 4216796712032373480)) },
    { easysimd_x_mm512_set_epu64(UINT64_C( 9490244949648135949), UINT64_C( 3952247228721925392),
                              UINT64_C( 4800241040971682796), UINT64_C( 9619996883527725324),
                              UINT64_C(15935750477416943804), UINT64_C(  545362928884482916),
                              UINT64_C(13559318363578452842), UINT64_C( 7722701545450284407)),
      UINT8_C(134),
      easysimd_x_mm512_set_epu32(UINT32_C( 500898194), UINT32_C(4078085990), UINT32_C(2494049110), UINT32_C(1592224201),
                              UINT32_C( 111635698), UINT32_C(    186713), UINT32_C(1765622469), UINT32_C(4017148467),
                              UINT32_C(2543052619), UINT32_C(1161807732), UINT32_C(1925351794), UINT32_C(2298119068),
                              UINT32_C( 457010151), UINT32_C(2589010019), UINT32_C( 502276479), UINT32_C(1967748710)),
      easysimd_x_mm512_set_epu32(UINT32_C(1919012105), UINT32_C(2908857333), UINT32_C(1122604656), UINT32_C(3433647442),
                              UINT32_C(2386428500), UINT32_C( 463161035), UINT32_C( 504317420), UINT32_C(3353921428),
                              UINT32_C(1582348389), UINT32_C(1180932658), UINT32_C(1476554796), UINT32_C(2783736621),
                              UINT32_C( 330646602), UINT32_C(1769150036), UINT32_C(2020624655), UINT32_C(3683994282)),
      easysimd_x_mm512_set_epu64(UINT64_C(11862570336616064670), UINT64_C( 3952247228721925392),
                              UINT64_C( 4800241040971682796), UINT64_C( 9619996883527725324),
                              UINT64_C(15935750477416943804), UINT64_C( 6397358209009989228),
                              UINT64_C( 4580347168318210684), UINT64_C( 7722701545450284407)) },
    { easysimd_x_mm512_set_epu64(UINT64_C( 8956593975554634232), UINT64_C( 9593792923362730078),
                              UINT64_C(  751700862087837721), UINT64_C( 7205298436209283097),
                              UINT64_C( 7151721520472513082), UINT64_C( 8910303953543094872),
                              UINT64_C(  533657364826431938), UINT64_C(13265804505255182490)),
      UINT8_C(251),
      easysimd_x_mm512_set_epu32(UINT32_C(1468179080), UINT32_C(3172744829), UINT32_C(1457928522), UINT32_C(1192418034),
                              UINT32_C( 105193191), UINT32_C( 430546192), UINT32_C(1509518002), UINT32_C( 354607881),
                              UINT32_C(3139371107), UINT32_C(2393204313), UINT32_C(1496510794), UINT32_C(3916080313),
                              UINT32_C(3933358732), UINT32_C(2965437178), UINT32_C(2440098689), UINT32_C( 675981365)),
      easysimd_x_mm512_set_epu32(UINT32_C(3471092536), UINT32_C(4213288110), UINT32_C(3288478343), UINT32_C( 269318758),
                              UINT32_C(2757016548), UINT32_C( 404238758), UINT32_C(3038240298), UINT32_C(3153052129),
                              UINT32_C(1906833283), UINT32_C(1593207408), UINT32_C(  59630942), UINT32_C(3403525194),
                              UINT32_C(2850644791), UINT32_C(1343686045), UINT32_C(1484433553), UINT32_C(2500647723)),
      easysimd_x_mm512_set_epu64(UINT64_C(13367688064089683190), UINT64_C(  321140543933681772),
                              UINT64_C(  174043457915709536), UINT64_C( 1118097134147228649),
                              UINT64_C( 3812870840329150704), UINT64_C( 8910303953543094872),
                              UINT64_C( 3984616553402781010), UINT64_C( 1690391261177681895)) },
    { easysimd_x_mm512_set_epu64(UINT64_C( 9350173910558210368), UINT64_C( 8451791018593404629),
                              UINT64_C( 5111327021160397113), UINT64_C( 8067526547900849939),
                              UINT64_C( 4397190784689926414), UINT64_C(15730677711069966608),
                              UINT64_C(15155555711952095903), UINT64_C( 4601095961680188139)),
      UINT8_C( 77),
      easysimd_x_mm512_set_epu32(UINT32_C( 289816884), UINT32_C(3139170300), UINT32_C(2611086568), UINT32_C( 981312265),
                              UINT32_C(2310975133), UINT32_C(1423285786), UINT32_C(4145860146), UINT32_C( 849682935),
                              UINT32_C(3098522529), UINT32_C(1778767618), UINT32_C(2469498326), UINT32_C(3407697658),
                              UINT32_C(4257688348), UINT32_C(1327333484), UINT32_C(4206795397), UINT32_C(1498113253)),
      easysimd_x_mm512_set_epu32(UINT32_C(3119696014), UINT32_C( 934565143), UINT32_C( 659452226), UINT32_C(3987623713),
                              UINT32_C(3941044651), UINT32_C(3075534691), UINT32_C( 348385654), UINT32_C(3299605274),
                              UINT32_C( 734145932), UINT32_C(2544261168), UINT32_C(1332327027), UINT32_C(3348556299),
                              UINT32_C(3524888946), UINT32_C(2026143937), UINT32_C(3684996090), UINT32_C(3613250397)),
      easysimd_x_mm512_set_epu64(UINT64_C( 9350173910558210368), UINT64_C( 3913104057771739945),
                              UINT64_C( 5111327021160397113), UINT64_C( 8067526547900849939),
                              UINT64_C( 4525649377373257824), UINT64_C(11410867457783447742),
                              UINT64_C(15155555711952095903), UINT64_C( 5413058306153211441)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(16773967285187515106), UINT64_C( 3477992427783883408),
                              UINT64_C( 1002604261497217766), UINT64_C( 1352136840172993944),
                              UINT64_C(10899831745595212891), UINT64_C( 2236619794744991665),
                              UINT64_C( 4130838651210953091), UINT64_C(12223797258734177268)),
      UINT8_C(188),
      easysimd_x_mm512_set_epu32(UINT32_C(3198361131), UINT32_C(1105007823), UINT32_C(3912476736), UINT32_C(2446995251),
                              UINT32_C(2582844574), UINT32_C(2764726563), UINT32_C(1724064135), UINT32_C( 994964469),
                              UINT32_C( 257756540), UINT32_C( 980676724), UINT32_C(2274290616), UINT32_C(4142129112),
                              UINT32_C(1407509141), UINT32_C(1593753754), UINT32_C(3346961920), UINT32_C(3859603033)),
      easysimd_x_mm512_set_epu32(UINT32_C(1883521655), UINT32_C( 997816392), UINT32_C( 186891147), UINT32_C(2813182686),
                              UINT32_C(3818488413), UINT32_C(1262292349), UINT32_C(2615667202), UINT32_C(3433123518),
                              UINT32_C(2317895719), UINT32_C(1795398992), UINT32_C(1050555729), UINT32_C(1774700454),
                              UINT32_C( 350094657), UINT32_C(2222937199), UINT32_C( 539823167), UINT32_C( 663093254)),
      easysimd_x_mm512_set_epu64(UINT64_C( 1102594919077634616), UINT64_C( 3477992427783883408),
                              UINT64_C( 3489893187551966487), UINT64_C( 3415835918098281942),
                              UINT64_C( 1760706001747462208), UINT64_C( 7351038415593016848),
                              UINT64_C( 4130838651210953091), UINT64_C(12223797258734177268)) },
    { easysimd_x_mm512_set_epu64(UINT64_C( 9161315007163903385), UINT64_C(  901926328951971839),
                              UINT64_C(11374042021460658344), UINT64_C(14235844241233139061),
                              UINT64_C(16689996302050367513), UINT64_C(17811135944692719319),
                              UINT64_C( 7952138000462838282), UINT64_C(15106420877923679668)),
      UINT8_C(197),
      easysimd_x_mm512_set_epu32(UINT32_C(1215894565), UINT32_C( 325247992), UINT32_C(3808486726), UINT32_C(3829410744),
                              UINT32_C(1276796092), UINT32_C( 483034698), UINT32_C(3265794508), UINT32_C( 145210622),
                              UINT32_C(4212031611), UINT32_C(3325547336), UINT32_C(1445017193), UINT32_C(2689093900),
                              UINT32_C(4273435877), UINT32_C( 524026689), UINT32_C(3618756570), UINT32_C(3961201514)),
      easysimd_x_mm512_set_epu32(UINT32_C(3460615822), UINT32_C(2842020471), UINT32_C(1351189519), UINT32_C(2329879373),
                              UINT32_C(3974357402), UINT32_C(2816300347), UINT32_C( 773721318), UINT32_C(3997442937),
                              UINT32_C(2436503902), UINT32_C(3242344117), UINT32_C(1149812233), UINT32_C( 907108945),
                              UINT32_C(1385675283), UINT32_C(3399903430), UINT32_C(2550192792), UINT32_C(3214774192)),
      easysimd_x_mm512_set_epu64(UINT64_C(  924361451415644232), UINT64_C( 8922065103190183512),
                              UINT64_C(11374042021460658344), UINT64_C(14235844241233139061),
                              UINT64_C(16689996302050367513), UINT64_C( 2439301130634935500),
                              UINT64_C( 7952138000462838282), UINT64_C(12734368396518526688)) },
    { easysimd_x_mm512_set_epu64(UINT64_C(10381435592908454864), UINT64_C( 7683972863259161915),
                              UINT64_C(  312335983814548083), UINT64_C( 3934167861393427795),
                              UINT64_C(15803008790257017530), UINT64_C(12384685209313245301),
                              UINT64_C(17881738201070197485), UINT64_C(14224003016858721277)),
      UINT8_C( 76),
      easysimd_x_mm512_set_epu32(UINT32_C(3028673683), UINT32_C(2581675996), UINT32_C(3969199228), UINT32_C(1709618805),
                              UINT32_C(3286547215), UINT32_C(2496179327), UINT32_C(2647114121), UINT32_C(2818621113),
                              UINT32_C( 879830851), UINT32_C(3024057012), UINT32_C( 247658746), UINT32_C(1778653183),
                              UINT32_C( 608002580), UINT32_C(2912110970), UINT32_C(2119947745), UINT32_C( 102275654)),
      easysimd_x_mm512_set_epu32(UINT32_C(3762799031), UINT32_C(1035026982), UINT32_C( 282468805), UINT32_C( 635023104),
                              UINT32_C(1863059331), UINT32_C(4265385561), UINT32_C( 804673998), UINT32_C(2920963576),
                              UINT32_C(1218801842), UINT32_C(1010987093), UINT32_C(3172703974), UINT32_C(1792395250),
                              UINT32_C(3430253324), UINT32_C( 714780216), UINT32_C(4029344470), UINT32_C(2590869425)),
      easysimd_x_mm512_set_epu64(UINT64_C(10381435592908454864), UINT64_C( 1085647440207870720),
                              UINT64_C(  312335983814548083), UINT64_C( 3934167861393427795),
                              UINT64_C( 3057282607628146116), UINT64_C( 3188049516606580750),
                              UINT64_C(17881738201070197485), UINT64_C(14224003016858721277)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i  src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i  a = test_vec[i].a;
    easysimd__m512i  b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mul_epu32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mul_epu32");
    easysimd_assert_m512i_u64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mul_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -775.40), EASYSIMD_FLOAT32_C(  -210.92), EASYSIMD_FLOAT32_C(   987.42), EASYSIMD_FLOAT32_C(   542.45),
                         EASYSIMD_FLOAT32_C(  -745.60), EASYSIMD_FLOAT32_C(   -50.38), EASYSIMD_FLOAT32_C(   163.82), EASYSIMD_FLOAT32_C(  -164.62),
                         EASYSIMD_FLOAT32_C(  -736.65), EASYSIMD_FLOAT32_C(  -764.30), EASYSIMD_FLOAT32_C(   675.25), EASYSIMD_FLOAT32_C(  -182.15),
                         EASYSIMD_FLOAT32_C(  -748.44), EASYSIMD_FLOAT32_C(    82.10), EASYSIMD_FLOAT32_C(   684.52), EASYSIMD_FLOAT32_C(  -343.09)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   263.91), EASYSIMD_FLOAT32_C(  -350.89), EASYSIMD_FLOAT32_C(  -318.01), EASYSIMD_FLOAT32_C(  -980.00),
                         EASYSIMD_FLOAT32_C(   872.18), EASYSIMD_FLOAT32_C(    80.96), EASYSIMD_FLOAT32_C(   145.89), EASYSIMD_FLOAT32_C(   832.89),
                         EASYSIMD_FLOAT32_C(  -267.96), EASYSIMD_FLOAT32_C(  -536.57), EASYSIMD_FLOAT32_C(  -934.00), EASYSIMD_FLOAT32_C(   653.62),
                         EASYSIMD_FLOAT32_C(   984.11), EASYSIMD_FLOAT32_C(   140.30), EASYSIMD_FLOAT32_C(  -580.05), EASYSIMD_FLOAT32_C(  -915.75)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-204635.83), EASYSIMD_FLOAT32_C( 74009.72), EASYSIMD_FLOAT32_C(-314009.44), EASYSIMD_FLOAT32_C(-531601.00),
                         EASYSIMD_FLOAT32_C(-650297.38), EASYSIMD_FLOAT32_C( -4078.76), EASYSIMD_FLOAT32_C( 23899.70), EASYSIMD_FLOAT32_C(-137110.34),
                         EASYSIMD_FLOAT32_C(197392.73), EASYSIMD_FLOAT32_C(410100.44), EASYSIMD_FLOAT32_C(-630683.50), EASYSIMD_FLOAT32_C(-119056.88),
                         EASYSIMD_FLOAT32_C(-736547.25), EASYSIMD_FLOAT32_C( 11518.63), EASYSIMD_FLOAT32_C(-397055.84), EASYSIMD_FLOAT32_C(314184.66)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -910.74), EASYSIMD_FLOAT32_C(  -302.10), EASYSIMD_FLOAT32_C(   937.08), EASYSIMD_FLOAT32_C(   618.13),
                         EASYSIMD_FLOAT32_C(    85.12), EASYSIMD_FLOAT32_C(     3.50), EASYSIMD_FLOAT32_C(  -122.84), EASYSIMD_FLOAT32_C(   290.22),
                         EASYSIMD_FLOAT32_C(   606.76), EASYSIMD_FLOAT32_C(  -664.92), EASYSIMD_FLOAT32_C(   454.81), EASYSIMD_FLOAT32_C(   299.40),
                         EASYSIMD_FLOAT32_C(  -524.63), EASYSIMD_FLOAT32_C(    40.68), EASYSIMD_FLOAT32_C(   218.77), EASYSIMD_FLOAT32_C(    35.82)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   392.21), EASYSIMD_FLOAT32_C(   139.00), EASYSIMD_FLOAT32_C(  -878.97), EASYSIMD_FLOAT32_C(   778.57),
                         EASYSIMD_FLOAT32_C(  -810.83), EASYSIMD_FLOAT32_C(   413.49), EASYSIMD_FLOAT32_C(   505.44), EASYSIMD_FLOAT32_C(   291.58),
                         EASYSIMD_FLOAT32_C(  -757.25), EASYSIMD_FLOAT32_C(   594.07), EASYSIMD_FLOAT32_C(   304.96), EASYSIMD_FLOAT32_C(  -155.47),
                         EASYSIMD_FLOAT32_C(   635.03), EASYSIMD_FLOAT32_C(   654.85), EASYSIMD_FLOAT32_C(   777.61), EASYSIMD_FLOAT32_C(  -598.19)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-357201.31), EASYSIMD_FLOAT32_C(-41991.90), EASYSIMD_FLOAT32_C(-823665.19), EASYSIMD_FLOAT32_C(481257.47),
                         EASYSIMD_FLOAT32_C(-69017.85), EASYSIMD_FLOAT32_C(  1447.21), EASYSIMD_FLOAT32_C(-62088.25), EASYSIMD_FLOAT32_C( 84622.34),
                         EASYSIMD_FLOAT32_C(-459469.03), EASYSIMD_FLOAT32_C(-395009.03), EASYSIMD_FLOAT32_C(138698.86), EASYSIMD_FLOAT32_C(-46547.72),
                         EASYSIMD_FLOAT32_C(-333155.81), EASYSIMD_FLOAT32_C( 26639.30), EASYSIMD_FLOAT32_C(170117.73), EASYSIMD_FLOAT32_C(-21427.17)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   202.90), EASYSIMD_FLOAT32_C(  -396.66), EASYSIMD_FLOAT32_C(  -364.01), EASYSIMD_FLOAT32_C(    56.81),
                         EASYSIMD_FLOAT32_C(  -881.59), EASYSIMD_FLOAT32_C(   212.81), EASYSIMD_FLOAT32_C(  -968.64), EASYSIMD_FLOAT32_C(  -657.19),
                         EASYSIMD_FLOAT32_C(   232.02), EASYSIMD_FLOAT32_C(   984.70), EASYSIMD_FLOAT32_C(  -800.83), EASYSIMD_FLOAT32_C(  -826.63),
                         EASYSIMD_FLOAT32_C(   822.26), EASYSIMD_FLOAT32_C(  -892.21), EASYSIMD_FLOAT32_C(  -651.70), EASYSIMD_FLOAT32_C(  -380.50)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -221.35), EASYSIMD_FLOAT32_C(  -305.38), EASYSIMD_FLOAT32_C(   546.45), EASYSIMD_FLOAT32_C(  -697.03),
                         EASYSIMD_FLOAT32_C(    93.97), EASYSIMD_FLOAT32_C(   975.92), EASYSIMD_FLOAT32_C(   876.47), EASYSIMD_FLOAT32_C(   762.37),
                         EASYSIMD_FLOAT32_C(   880.83), EASYSIMD_FLOAT32_C(  -763.06), EASYSIMD_FLOAT32_C(  -540.57), EASYSIMD_FLOAT32_C(  -512.55),
                         EASYSIMD_FLOAT32_C(   -32.98), EASYSIMD_FLOAT32_C(   700.87), EASYSIMD_FLOAT32_C(  -425.19), EASYSIMD_FLOAT32_C(  -849.48)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-44911.91), EASYSIMD_FLOAT32_C(121132.03), EASYSIMD_FLOAT32_C(-198913.28), EASYSIMD_FLOAT32_C(-39598.28),
                         EASYSIMD_FLOAT32_C(-82843.02), EASYSIMD_FLOAT32_C(207685.53), EASYSIMD_FLOAT32_C(-848983.88), EASYSIMD_FLOAT32_C(-501021.94),
                         EASYSIMD_FLOAT32_C(204370.19), EASYSIMD_FLOAT32_C(-751385.19), EASYSIMD_FLOAT32_C(432904.69), EASYSIMD_FLOAT32_C(423689.19),
                         EASYSIMD_FLOAT32_C(-27118.13), EASYSIMD_FLOAT32_C(-625323.25), EASYSIMD_FLOAT32_C(277096.34), EASYSIMD_FLOAT32_C(323227.12)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -25.40), EASYSIMD_FLOAT32_C(  -267.80), EASYSIMD_FLOAT32_C(   353.79), EASYSIMD_FLOAT32_C(   -35.72),
                         EASYSIMD_FLOAT32_C(   125.21), EASYSIMD_FLOAT32_C(   137.22), EASYSIMD_FLOAT32_C(   310.88), EASYSIMD_FLOAT32_C(  -724.55),
                         EASYSIMD_FLOAT32_C(  -538.86), EASYSIMD_FLOAT32_C(    39.65), EASYSIMD_FLOAT32_C(  -229.28), EASYSIMD_FLOAT32_C(  -842.78),
                         EASYSIMD_FLOAT32_C(   -14.75), EASYSIMD_FLOAT32_C(  -859.98), EASYSIMD_FLOAT32_C(   215.44), EASYSIMD_FLOAT32_C(   762.83)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -551.49), EASYSIMD_FLOAT32_C(   -42.33), EASYSIMD_FLOAT32_C(  -926.18), EASYSIMD_FLOAT32_C(    36.96),
                         EASYSIMD_FLOAT32_C(   954.39), EASYSIMD_FLOAT32_C(   874.71), EASYSIMD_FLOAT32_C(  -375.00), EASYSIMD_FLOAT32_C(   949.07),
                         EASYSIMD_FLOAT32_C(   -16.18), EASYSIMD_FLOAT32_C(  -931.82), EASYSIMD_FLOAT32_C(  -687.15), EASYSIMD_FLOAT32_C(  -416.23),
                         EASYSIMD_FLOAT32_C(  -313.36), EASYSIMD_FLOAT32_C(   905.90), EASYSIMD_FLOAT32_C(     1.93), EASYSIMD_FLOAT32_C(  -464.98)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C( 14007.85), EASYSIMD_FLOAT32_C( 11335.97), EASYSIMD_FLOAT32_C(-327673.22), EASYSIMD_FLOAT32_C( -1320.21),
                         EASYSIMD_FLOAT32_C(119499.17), EASYSIMD_FLOAT32_C(120027.71), EASYSIMD_FLOAT32_C(-116580.00), EASYSIMD_FLOAT32_C(-687648.69),
                         EASYSIMD_FLOAT32_C(  8718.75), EASYSIMD_FLOAT32_C(-36946.66), EASYSIMD_FLOAT32_C(157549.75), EASYSIMD_FLOAT32_C(350790.34),
                         EASYSIMD_FLOAT32_C(  4622.06), EASYSIMD_FLOAT32_C(-779055.88), EASYSIMD_FLOAT32_C(   415.80), EASYSIMD_FLOAT32_C(-354700.72)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -861.86), EASYSIMD_FLOAT32_C(   437.62), EASYSIMD_FLOAT32_C(  -236.27), EASYSIMD_FLOAT32_C(   439.08),
                         EASYSIMD_FLOAT32_C(   476.60), EASYSIMD_FLOAT32_C(  -725.80), EASYSIMD_FLOAT32_C(   626.57), EASYSIMD_FLOAT32_C(  -848.67),
                         EASYSIMD_FLOAT32_C(  -961.54), EASYSIMD_FLOAT32_C(  -999.94), EASYSIMD_FLOAT32_C(   788.38), EASYSIMD_FLOAT32_C(  -928.14),
                         EASYSIMD_FLOAT32_C(   779.51), EASYSIMD_FLOAT32_C(   846.68), EASYSIMD_FLOAT32_C(  -858.45), EASYSIMD_FLOAT32_C(   292.21)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -182.57), EASYSIMD_FLOAT32_C(  -580.78), EASYSIMD_FLOAT32_C(   704.32), EASYSIMD_FLOAT32_C(  -124.94),
                         EASYSIMD_FLOAT32_C(  -725.23), EASYSIMD_FLOAT32_C(  -635.58), EASYSIMD_FLOAT32_C(   196.48), EASYSIMD_FLOAT32_C(  -485.66),
                         EASYSIMD_FLOAT32_C(  -906.19), EASYSIMD_FLOAT32_C(   -43.74), EASYSIMD_FLOAT32_C(   899.84), EASYSIMD_FLOAT32_C(  -720.16),
                         EASYSIMD_FLOAT32_C(   576.76), EASYSIMD_FLOAT32_C(   994.06), EASYSIMD_FLOAT32_C(  -108.56), EASYSIMD_FLOAT32_C(   212.62)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(157349.78), EASYSIMD_FLOAT32_C(-254160.95), EASYSIMD_FLOAT32_C(-166409.69), EASYSIMD_FLOAT32_C(-54858.66),
                         EASYSIMD_FLOAT32_C(-345644.62), EASYSIMD_FLOAT32_C(461303.97), EASYSIMD_FLOAT32_C(123108.47), EASYSIMD_FLOAT32_C(412165.06),
                         EASYSIMD_FLOAT32_C(871337.94), EASYSIMD_FLOAT32_C( 43737.38), EASYSIMD_FLOAT32_C(709415.88), EASYSIMD_FLOAT32_C(668409.31),
                         EASYSIMD_FLOAT32_C(449590.19), EASYSIMD_FLOAT32_C(841650.69), EASYSIMD_FLOAT32_C( 93193.33), EASYSIMD_FLOAT32_C( 62129.69)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   593.71), EASYSIMD_FLOAT32_C(  -601.77), EASYSIMD_FLOAT32_C(  -110.10), EASYSIMD_FLOAT32_C(   145.40),
                         EASYSIMD_FLOAT32_C(   740.85), EASYSIMD_FLOAT32_C(   970.19), EASYSIMD_FLOAT32_C(  -854.26), EASYSIMD_FLOAT32_C(  -208.21),
                         EASYSIMD_FLOAT32_C(   769.57), EASYSIMD_FLOAT32_C(  -297.46), EASYSIMD_FLOAT32_C(  -845.75), EASYSIMD_FLOAT32_C(  -517.72),
                         EASYSIMD_FLOAT32_C(  -240.19), EASYSIMD_FLOAT32_C(  -763.89), EASYSIMD_FLOAT32_C(  -197.03), EASYSIMD_FLOAT32_C(   -33.35)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -533.43), EASYSIMD_FLOAT32_C(   541.52), EASYSIMD_FLOAT32_C(  -293.53), EASYSIMD_FLOAT32_C(   215.59),
                         EASYSIMD_FLOAT32_C(  -844.97), EASYSIMD_FLOAT32_C(  -755.33), EASYSIMD_FLOAT32_C(   468.59), EASYSIMD_FLOAT32_C(  -772.73),
                         EASYSIMD_FLOAT32_C(   919.17), EASYSIMD_FLOAT32_C(   772.56), EASYSIMD_FLOAT32_C(  -506.06), EASYSIMD_FLOAT32_C(   848.47),
                         EASYSIMD_FLOAT32_C(   289.91), EASYSIMD_FLOAT32_C(    20.43), EASYSIMD_FLOAT32_C(   -64.43), EASYSIMD_FLOAT32_C(  -706.80)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-316702.72), EASYSIMD_FLOAT32_C(-325870.50), EASYSIMD_FLOAT32_C( 32317.65), EASYSIMD_FLOAT32_C( 31346.78),
                         EASYSIMD_FLOAT32_C(-625996.00), EASYSIMD_FLOAT32_C(-732813.62), EASYSIMD_FLOAT32_C(-400297.69), EASYSIMD_FLOAT32_C(160890.11),
                         EASYSIMD_FLOAT32_C(707365.62), EASYSIMD_FLOAT32_C(-229805.69), EASYSIMD_FLOAT32_C(428000.25), EASYSIMD_FLOAT32_C(-439269.84),
                         EASYSIMD_FLOAT32_C(-69633.48), EASYSIMD_FLOAT32_C(-15606.27), EASYSIMD_FLOAT32_C( 12694.64), EASYSIMD_FLOAT32_C( 23571.78)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -180.32), EASYSIMD_FLOAT32_C(  -914.18), EASYSIMD_FLOAT32_C(  -674.78), EASYSIMD_FLOAT32_C(   230.92),
                         EASYSIMD_FLOAT32_C(   619.73), EASYSIMD_FLOAT32_C(  -630.60), EASYSIMD_FLOAT32_C(  -418.47), EASYSIMD_FLOAT32_C(  -865.96),
                         EASYSIMD_FLOAT32_C(  -670.71), EASYSIMD_FLOAT32_C(    17.47), EASYSIMD_FLOAT32_C(    61.90), EASYSIMD_FLOAT32_C(   647.63),
                         EASYSIMD_FLOAT32_C(  -455.42), EASYSIMD_FLOAT32_C(  -850.08), EASYSIMD_FLOAT32_C(   132.45), EASYSIMD_FLOAT32_C(  -354.79)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -729.37), EASYSIMD_FLOAT32_C(  -945.53), EASYSIMD_FLOAT32_C(   198.36), EASYSIMD_FLOAT32_C(   643.43),
                         EASYSIMD_FLOAT32_C(  -994.87), EASYSIMD_FLOAT32_C(  -154.38), EASYSIMD_FLOAT32_C(  -536.68), EASYSIMD_FLOAT32_C(  -548.49),
                         EASYSIMD_FLOAT32_C(  -292.06), EASYSIMD_FLOAT32_C(  -771.61), EASYSIMD_FLOAT32_C(  -487.89), EASYSIMD_FLOAT32_C(  -482.82),
                         EASYSIMD_FLOAT32_C(   131.08), EASYSIMD_FLOAT32_C(   366.17), EASYSIMD_FLOAT32_C(   127.55), EASYSIMD_FLOAT32_C(  -936.85)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(131520.00), EASYSIMD_FLOAT32_C(864384.62), EASYSIMD_FLOAT32_C(-133849.36), EASYSIMD_FLOAT32_C(148580.86),
                         EASYSIMD_FLOAT32_C(-616550.75), EASYSIMD_FLOAT32_C( 97352.02), EASYSIMD_FLOAT32_C(224584.48), EASYSIMD_FLOAT32_C(474970.41),
                         EASYSIMD_FLOAT32_C(195887.56), EASYSIMD_FLOAT32_C(-13480.03), EASYSIMD_FLOAT32_C(-30200.39), EASYSIMD_FLOAT32_C(-312688.72),
                         EASYSIMD_FLOAT32_C(-59696.46), EASYSIMD_FLOAT32_C(-311273.81), EASYSIMD_FLOAT32_C( 16894.00), EASYSIMD_FLOAT32_C(332385.00)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   405.10), EASYSIMD_FLOAT32_C(   457.94), EASYSIMD_FLOAT32_C(   120.94), EASYSIMD_FLOAT32_C(   944.02),
                         EASYSIMD_FLOAT32_C(  -205.34), EASYSIMD_FLOAT32_C(   155.90), EASYSIMD_FLOAT32_C(  -913.86), EASYSIMD_FLOAT32_C(   170.83),
                         EASYSIMD_FLOAT32_C(  -194.64), EASYSIMD_FLOAT32_C(   505.24), EASYSIMD_FLOAT32_C(   874.71), EASYSIMD_FLOAT32_C(  -847.65),
                         EASYSIMD_FLOAT32_C(   -72.00), EASYSIMD_FLOAT32_C(   772.81), EASYSIMD_FLOAT32_C(  -151.00), EASYSIMD_FLOAT32_C(  -489.53)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   923.98), EASYSIMD_FLOAT32_C(   597.38), EASYSIMD_FLOAT32_C(  -237.17), EASYSIMD_FLOAT32_C(  -159.78),
                         EASYSIMD_FLOAT32_C(   -45.00), EASYSIMD_FLOAT32_C(  -167.53), EASYSIMD_FLOAT32_C(   681.28), EASYSIMD_FLOAT32_C(  -654.80),
                         EASYSIMD_FLOAT32_C(   504.91), EASYSIMD_FLOAT32_C(  -353.27), EASYSIMD_FLOAT32_C(  -789.06), EASYSIMD_FLOAT32_C(  -566.71),
                         EASYSIMD_FLOAT32_C(  -516.77), EASYSIMD_FLOAT32_C(   957.42), EASYSIMD_FLOAT32_C(  -465.35), EASYSIMD_FLOAT32_C(   491.11)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(374304.28), EASYSIMD_FLOAT32_C(273564.19), EASYSIMD_FLOAT32_C(-28683.34), EASYSIMD_FLOAT32_C(-150835.52),
                         EASYSIMD_FLOAT32_C(  9240.30), EASYSIMD_FLOAT32_C(-26117.93), EASYSIMD_FLOAT32_C(-622594.56), EASYSIMD_FLOAT32_C(-111859.48),
                         EASYSIMD_FLOAT32_C(-98275.68), EASYSIMD_FLOAT32_C(-178486.12), EASYSIMD_FLOAT32_C(-690198.69), EASYSIMD_FLOAT32_C(480371.75),
                         EASYSIMD_FLOAT32_C( 37207.44), EASYSIMD_FLOAT32_C(739903.75), EASYSIMD_FLOAT32_C( 70267.85), EASYSIMD_FLOAT32_C(-240413.08)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mul_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mul_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mul_round_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    easysimd_float32 a[16];
    easysimd_float32 b[16];
    easysimd_float32 nearest_inf[16];
    easysimd_float32 neg_inf[16];
    easysimd_float32 pos_inf[16];
    easysimd_float32 zero[16];
    easysimd_float32 direction[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   587.40), EASYSIMD_FLOAT32_C(  -222.58), EASYSIMD_FLOAT32_C(  -135.27), EASYSIMD_FLOAT32_C(  -844.35),
        EASYSIMD_FLOAT32_C(   107.23), EASYSIMD_FLOAT32_C(   899.39), EASYSIMD_FLOAT32_C(   -36.81), EASYSIMD_FLOAT32_C(   218.20),
        EASYSIMD_FLOAT32_C(   168.05), EASYSIMD_FLOAT32_C(  -381.54), EASYSIMD_FLOAT32_C(   241.18), EASYSIMD_FLOAT32_C(  -638.35),
        EASYSIMD_FLOAT32_C(   883.80), EASYSIMD_FLOAT32_C(   995.46), EASYSIMD_FLOAT32_C(   631.83), EASYSIMD_FLOAT32_C(  -292.46) },
      { EASYSIMD_FLOAT32_C(  -707.29), EASYSIMD_FLOAT32_C(  -724.14), EASYSIMD_FLOAT32_C(  -468.70), EASYSIMD_FLOAT32_C(  -696.19),
        EASYSIMD_FLOAT32_C(  -849.41), EASYSIMD_FLOAT32_C(  -757.87), EASYSIMD_FLOAT32_C(   972.89), EASYSIMD_FLOAT32_C(   727.44),
        EASYSIMD_FLOAT32_C(  -841.08), EASYSIMD_FLOAT32_C(  -587.75), EASYSIMD_FLOAT32_C(    39.23), EASYSIMD_FLOAT32_C(   342.06),
        EASYSIMD_FLOAT32_C(  -922.09), EASYSIMD_FLOAT32_C(   586.07), EASYSIMD_FLOAT32_C(   452.89), EASYSIMD_FLOAT32_C(   665.31) },
      { EASYSIMD_FLOAT32_C(-415462.00), EASYSIMD_FLOAT32_C(161179.00), EASYSIMD_FLOAT32_C( 63401.00), EASYSIMD_FLOAT32_C(587828.00),
        EASYSIMD_FLOAT32_C(-91082.00), EASYSIMD_FLOAT32_C(-681621.00), EASYSIMD_FLOAT32_C(-35812.00), EASYSIMD_FLOAT32_C(158727.00),
        EASYSIMD_FLOAT32_C(-141344.00), EASYSIMD_FLOAT32_C(224250.00), EASYSIMD_FLOAT32_C(  9461.00), EASYSIMD_FLOAT32_C(-218354.00),
        EASYSIMD_FLOAT32_C(-814943.00), EASYSIMD_FLOAT32_C(583409.00), EASYSIMD_FLOAT32_C(286150.00), EASYSIMD_FLOAT32_C(-194577.00) },
      { EASYSIMD_FLOAT32_C(-415463.00), EASYSIMD_FLOAT32_C(161179.00), EASYSIMD_FLOAT32_C( 63401.00), EASYSIMD_FLOAT32_C(587828.00),
        EASYSIMD_FLOAT32_C(-91083.00), EASYSIMD_FLOAT32_C(-681621.00), EASYSIMD_FLOAT32_C(-35813.00), EASYSIMD_FLOAT32_C(158727.00),
        EASYSIMD_FLOAT32_C(-141344.00), EASYSIMD_FLOAT32_C(224250.00), EASYSIMD_FLOAT32_C(  9461.00), EASYSIMD_FLOAT32_C(-218354.00),
        EASYSIMD_FLOAT32_C(-814944.00), EASYSIMD_FLOAT32_C(583409.00), EASYSIMD_FLOAT32_C(286149.00), EASYSIMD_FLOAT32_C(-194577.00) },
      { EASYSIMD_FLOAT32_C(-415462.00), EASYSIMD_FLOAT32_C(161180.00), EASYSIMD_FLOAT32_C( 63402.00), EASYSIMD_FLOAT32_C(587828.00),
        EASYSIMD_FLOAT32_C(-91082.00), EASYSIMD_FLOAT32_C(-681620.00), EASYSIMD_FLOAT32_C(-35812.00), EASYSIMD_FLOAT32_C(158728.00),
        EASYSIMD_FLOAT32_C(-141343.00), EASYSIMD_FLOAT32_C(224251.00), EASYSIMD_FLOAT32_C(  9462.00), EASYSIMD_FLOAT32_C(-218353.00),
        EASYSIMD_FLOAT32_C(-814943.00), EASYSIMD_FLOAT32_C(583410.00), EASYSIMD_FLOAT32_C(286150.00), EASYSIMD_FLOAT32_C(-194576.00) },
      { EASYSIMD_FLOAT32_C(-415462.00), EASYSIMD_FLOAT32_C(161179.00), EASYSIMD_FLOAT32_C( 63401.00), EASYSIMD_FLOAT32_C(587828.00),
        EASYSIMD_FLOAT32_C(-91082.00), EASYSIMD_FLOAT32_C(-681620.00), EASYSIMD_FLOAT32_C(-35812.00), EASYSIMD_FLOAT32_C(158727.00),
        EASYSIMD_FLOAT32_C(-141343.00), EASYSIMD_FLOAT32_C(224250.00), EASYSIMD_FLOAT32_C(  9461.00), EASYSIMD_FLOAT32_C(-218353.00),
        EASYSIMD_FLOAT32_C(-814943.00), EASYSIMD_FLOAT32_C(583409.00), EASYSIMD_FLOAT32_C(286149.00), EASYSIMD_FLOAT32_C(-194576.00) },
      { EASYSIMD_FLOAT32_C(-415462.00), EASYSIMD_FLOAT32_C(161179.00), EASYSIMD_FLOAT32_C( 63401.00), EASYSIMD_FLOAT32_C(587828.00),
        EASYSIMD_FLOAT32_C(-91082.00), EASYSIMD_FLOAT32_C(-681621.00), EASYSIMD_FLOAT32_C(-35812.00), EASYSIMD_FLOAT32_C(158727.00),
        EASYSIMD_FLOAT32_C(-141344.00), EASYSIMD_FLOAT32_C(224250.00), EASYSIMD_FLOAT32_C(  9461.00), EASYSIMD_FLOAT32_C(-218354.00),
        EASYSIMD_FLOAT32_C(-814943.00), EASYSIMD_FLOAT32_C(583409.00), EASYSIMD_FLOAT32_C(286150.00), EASYSIMD_FLOAT32_C(-194577.00) } },
    { { EASYSIMD_FLOAT32_C(  -636.50), EASYSIMD_FLOAT32_C(  -682.38), EASYSIMD_FLOAT32_C(   820.96), EASYSIMD_FLOAT32_C(   470.72),
        EASYSIMD_FLOAT32_C(  -782.99), EASYSIMD_FLOAT32_C(  -215.85), EASYSIMD_FLOAT32_C(  -311.07), EASYSIMD_FLOAT32_C(   385.06),
        EASYSIMD_FLOAT32_C(   402.62), EASYSIMD_FLOAT32_C(   930.11), EASYSIMD_FLOAT32_C(   746.71), EASYSIMD_FLOAT32_C(   286.42),
        EASYSIMD_FLOAT32_C(   925.57), EASYSIMD_FLOAT32_C(   378.55), EASYSIMD_FLOAT32_C(   993.96), EASYSIMD_FLOAT32_C(  -781.72) },
      { EASYSIMD_FLOAT32_C(   654.41), EASYSIMD_FLOAT32_C(  -474.74), EASYSIMD_FLOAT32_C(  -477.91), EASYSIMD_FLOAT32_C(   805.00),
        EASYSIMD_FLOAT32_C(  -232.61), EASYSIMD_FLOAT32_C(  -505.02), EASYSIMD_FLOAT32_C(   532.44), EASYSIMD_FLOAT32_C(   -73.70),
        EASYSIMD_FLOAT32_C(   -92.77), EASYSIMD_FLOAT32_C(  -428.34), EASYSIMD_FLOAT32_C(  -731.64), EASYSIMD_FLOAT32_C(   -14.86),
        EASYSIMD_FLOAT32_C(  -842.26), EASYSIMD_FLOAT32_C(   721.25), EASYSIMD_FLOAT32_C(  -349.55), EASYSIMD_FLOAT32_C(  -478.77) },
      { EASYSIMD_FLOAT32_C(-416532.00), EASYSIMD_FLOAT32_C(323953.00), EASYSIMD_FLOAT32_C(-392345.00), EASYSIMD_FLOAT32_C(378930.00),
        EASYSIMD_FLOAT32_C(182131.00), EASYSIMD_FLOAT32_C(109009.00), EASYSIMD_FLOAT32_C(-165626.00), EASYSIMD_FLOAT32_C(-28379.00),
        EASYSIMD_FLOAT32_C(-37351.00), EASYSIMD_FLOAT32_C(-398403.00), EASYSIMD_FLOAT32_C(-546323.00), EASYSIMD_FLOAT32_C( -4256.00),
        EASYSIMD_FLOAT32_C(-779571.00), EASYSIMD_FLOAT32_C(273029.00), EASYSIMD_FLOAT32_C(-347439.00), EASYSIMD_FLOAT32_C(374264.00) },
      { EASYSIMD_FLOAT32_C(-416532.00), EASYSIMD_FLOAT32_C(323953.00), EASYSIMD_FLOAT32_C(-392345.00), EASYSIMD_FLOAT32_C(378929.00),
        EASYSIMD_FLOAT32_C(182131.00), EASYSIMD_FLOAT32_C(109008.00), EASYSIMD_FLOAT32_C(-165627.00), EASYSIMD_FLOAT32_C(-28379.00),
        EASYSIMD_FLOAT32_C(-37352.00), EASYSIMD_FLOAT32_C(-398404.00), EASYSIMD_FLOAT32_C(-546323.00), EASYSIMD_FLOAT32_C( -4257.00),
        EASYSIMD_FLOAT32_C(-779571.00), EASYSIMD_FLOAT32_C(273029.00), EASYSIMD_FLOAT32_C(-347439.00), EASYSIMD_FLOAT32_C(374264.00) },
      { EASYSIMD_FLOAT32_C(-416531.00), EASYSIMD_FLOAT32_C(323954.00), EASYSIMD_FLOAT32_C(-392345.00), EASYSIMD_FLOAT32_C(378930.00),
        EASYSIMD_FLOAT32_C(182132.00), EASYSIMD_FLOAT32_C(109009.00), EASYSIMD_FLOAT32_C(-165626.00), EASYSIMD_FLOAT32_C(-28378.00),
        EASYSIMD_FLOAT32_C(-37351.00), EASYSIMD_FLOAT32_C(-398403.00), EASYSIMD_FLOAT32_C(-546322.00), EASYSIMD_FLOAT32_C( -4256.00),
        EASYSIMD_FLOAT32_C(-779570.00), EASYSIMD_FLOAT32_C(273030.00), EASYSIMD_FLOAT32_C(-347438.00), EASYSIMD_FLOAT32_C(374265.00) },
      { EASYSIMD_FLOAT32_C(-416531.00), EASYSIMD_FLOAT32_C(323953.00), EASYSIMD_FLOAT32_C(-392345.00), EASYSIMD_FLOAT32_C(378929.00),
        EASYSIMD_FLOAT32_C(182131.00), EASYSIMD_FLOAT32_C(109008.00), EASYSIMD_FLOAT32_C(-165626.00), EASYSIMD_FLOAT32_C(-28378.00),
        EASYSIMD_FLOAT32_C(-37351.00), EASYSIMD_FLOAT32_C(-398403.00), EASYSIMD_FLOAT32_C(-546322.00), EASYSIMD_FLOAT32_C( -4256.00),
        EASYSIMD_FLOAT32_C(-779570.00), EASYSIMD_FLOAT32_C(273029.00), EASYSIMD_FLOAT32_C(-347438.00), EASYSIMD_FLOAT32_C(374264.00) },
      { EASYSIMD_FLOAT32_C(-416532.00), EASYSIMD_FLOAT32_C(323953.00), EASYSIMD_FLOAT32_C(-392345.00), EASYSIMD_FLOAT32_C(378930.00),
        EASYSIMD_FLOAT32_C(182131.00), EASYSIMD_FLOAT32_C(109009.00), EASYSIMD_FLOAT32_C(-165626.00), EASYSIMD_FLOAT32_C(-28379.00),
        EASYSIMD_FLOAT32_C(-37351.00), EASYSIMD_FLOAT32_C(-398403.00), EASYSIMD_FLOAT32_C(-546323.00), EASYSIMD_FLOAT32_C( -4256.00),
        EASYSIMD_FLOAT32_C(-779571.00), EASYSIMD_FLOAT32_C(273029.00), EASYSIMD_FLOAT32_C(-347439.00), EASYSIMD_FLOAT32_C(374264.00) } },
    { { EASYSIMD_FLOAT32_C(  -961.13), EASYSIMD_FLOAT32_C(  -528.59), EASYSIMD_FLOAT32_C(   991.96), EASYSIMD_FLOAT32_C(  -744.12),
        EASYSIMD_FLOAT32_C(   255.56), EASYSIMD_FLOAT32_C(  -319.11), EASYSIMD_FLOAT32_C(   640.94), EASYSIMD_FLOAT32_C(  -341.82),
        EASYSIMD_FLOAT32_C(  -389.00), EASYSIMD_FLOAT32_C(   387.65), EASYSIMD_FLOAT32_C(   944.59), EASYSIMD_FLOAT32_C(  -463.44),
        EASYSIMD_FLOAT32_C(  -233.80), EASYSIMD_FLOAT32_C(   938.55), EASYSIMD_FLOAT32_C(  -245.16), EASYSIMD_FLOAT32_C(  -579.40) },
      { EASYSIMD_FLOAT32_C(  -536.19), EASYSIMD_FLOAT32_C(   276.93), EASYSIMD_FLOAT32_C(  -774.40), EASYSIMD_FLOAT32_C(   231.20),
        EASYSIMD_FLOAT32_C(   771.91), EASYSIMD_FLOAT32_C(   758.04), EASYSIMD_FLOAT32_C(  -842.50), EASYSIMD_FLOAT32_C(  -320.86),
        EASYSIMD_FLOAT32_C(  -670.29), EASYSIMD_FLOAT32_C(  -574.14), EASYSIMD_FLOAT32_C(   664.28), EASYSIMD_FLOAT32_C(  -512.56),
        EASYSIMD_FLOAT32_C(  -852.89), EASYSIMD_FLOAT32_C(  -685.27), EASYSIMD_FLOAT32_C(     8.68), EASYSIMD_FLOAT32_C(  -814.02) },
      { EASYSIMD_FLOAT32_C(515348.00), EASYSIMD_FLOAT32_C(-146382.00), EASYSIMD_FLOAT32_C(-768174.00), EASYSIMD_FLOAT32_C(-172041.00),
        EASYSIMD_FLOAT32_C(197269.00), EASYSIMD_FLOAT32_C(-241898.00), EASYSIMD_FLOAT32_C(-539992.00), EASYSIMD_FLOAT32_C(109676.00),
        EASYSIMD_FLOAT32_C(260743.00), EASYSIMD_FLOAT32_C(-222565.00), EASYSIMD_FLOAT32_C(627472.00), EASYSIMD_FLOAT32_C(237541.00),
        EASYSIMD_FLOAT32_C(199406.00), EASYSIMD_FLOAT32_C(-643160.00), EASYSIMD_FLOAT32_C( -2128.00), EASYSIMD_FLOAT32_C(471643.00) },
      { EASYSIMD_FLOAT32_C(515348.00), EASYSIMD_FLOAT32_C(-146383.00), EASYSIMD_FLOAT32_C(-768174.00), EASYSIMD_FLOAT32_C(-172041.00),
        EASYSIMD_FLOAT32_C(197269.00), EASYSIMD_FLOAT32_C(-241899.00), EASYSIMD_FLOAT32_C(-539992.00), EASYSIMD_FLOAT32_C(109676.00),
        EASYSIMD_FLOAT32_C(260742.00), EASYSIMD_FLOAT32_C(-222566.00), EASYSIMD_FLOAT32_C(627472.00), EASYSIMD_FLOAT32_C(237540.00),
        EASYSIMD_FLOAT32_C(199405.00), EASYSIMD_FLOAT32_C(-643161.00), EASYSIMD_FLOAT32_C( -2128.00), EASYSIMD_FLOAT32_C(471643.00) },
      { EASYSIMD_FLOAT32_C(515349.00), EASYSIMD_FLOAT32_C(-146382.00), EASYSIMD_FLOAT32_C(-768173.00), EASYSIMD_FLOAT32_C(-172040.00),
        EASYSIMD_FLOAT32_C(197270.00), EASYSIMD_FLOAT32_C(-241898.00), EASYSIMD_FLOAT32_C(-539991.00), EASYSIMD_FLOAT32_C(109677.00),
        EASYSIMD_FLOAT32_C(260743.00), EASYSIMD_FLOAT32_C(-222565.00), EASYSIMD_FLOAT32_C(627473.00), EASYSIMD_FLOAT32_C(237541.00),
        EASYSIMD_FLOAT32_C(199406.00), EASYSIMD_FLOAT32_C(-643160.00), EASYSIMD_FLOAT32_C( -2127.00), EASYSIMD_FLOAT32_C(471644.00) },
      { EASYSIMD_FLOAT32_C(515348.00), EASYSIMD_FLOAT32_C(-146382.00), EASYSIMD_FLOAT32_C(-768173.00), EASYSIMD_FLOAT32_C(-172040.00),
        EASYSIMD_FLOAT32_C(197269.00), EASYSIMD_FLOAT32_C(-241898.00), EASYSIMD_FLOAT32_C(-539991.00), EASYSIMD_FLOAT32_C(109676.00),
        EASYSIMD_FLOAT32_C(260742.00), EASYSIMD_FLOAT32_C(-222565.00), EASYSIMD_FLOAT32_C(627472.00), EASYSIMD_FLOAT32_C(237540.00),
        EASYSIMD_FLOAT32_C(199405.00), EASYSIMD_FLOAT32_C(-643160.00), EASYSIMD_FLOAT32_C( -2127.00), EASYSIMD_FLOAT32_C(471643.00) },
      { EASYSIMD_FLOAT32_C(515348.00), EASYSIMD_FLOAT32_C(-146382.00), EASYSIMD_FLOAT32_C(-768174.00), EASYSIMD_FLOAT32_C(-172041.00),
        EASYSIMD_FLOAT32_C(197269.00), EASYSIMD_FLOAT32_C(-241898.00), EASYSIMD_FLOAT32_C(-539992.00), EASYSIMD_FLOAT32_C(109676.00),
        EASYSIMD_FLOAT32_C(260743.00), EASYSIMD_FLOAT32_C(-222565.00), EASYSIMD_FLOAT32_C(627472.00), EASYSIMD_FLOAT32_C(237541.00),
        EASYSIMD_FLOAT32_C(199406.00), EASYSIMD_FLOAT32_C(-643160.00), EASYSIMD_FLOAT32_C( -2128.00), EASYSIMD_FLOAT32_C(471643.00) } },
    { { EASYSIMD_FLOAT32_C(  -213.87), EASYSIMD_FLOAT32_C(     0.64), EASYSIMD_FLOAT32_C(  -558.14), EASYSIMD_FLOAT32_C(  -958.31),
        EASYSIMD_FLOAT32_C(   681.53), EASYSIMD_FLOAT32_C(  -917.20), EASYSIMD_FLOAT32_C(  -300.13), EASYSIMD_FLOAT32_C(  -707.48),
        EASYSIMD_FLOAT32_C(   470.45), EASYSIMD_FLOAT32_C(  -355.54), EASYSIMD_FLOAT32_C(  -170.92), EASYSIMD_FLOAT32_C(  -763.35),
        EASYSIMD_FLOAT32_C(  -416.99), EASYSIMD_FLOAT32_C(   583.93), EASYSIMD_FLOAT32_C(  -342.75), EASYSIMD_FLOAT32_C(    46.82) },
      { EASYSIMD_FLOAT32_C(  -139.14), EASYSIMD_FLOAT32_C(  -117.15), EASYSIMD_FLOAT32_C(  -721.98), EASYSIMD_FLOAT32_C(  -367.23),
        EASYSIMD_FLOAT32_C(  -359.11), EASYSIMD_FLOAT32_C(  -564.48), EASYSIMD_FLOAT32_C(   311.92), EASYSIMD_FLOAT32_C(   -29.40),
        EASYSIMD_FLOAT32_C(  -138.62), EASYSIMD_FLOAT32_C(   -23.80), EASYSIMD_FLOAT32_C(   458.04), EASYSIMD_FLOAT32_C(     8.49),
        EASYSIMD_FLOAT32_C(   290.92), EASYSIMD_FLOAT32_C(  -533.28), EASYSIMD_FLOAT32_C(   194.48), EASYSIMD_FLOAT32_C(  -922.94) },
      { EASYSIMD_FLOAT32_C( 29758.00), EASYSIMD_FLOAT32_C(   -75.00), EASYSIMD_FLOAT32_C(402966.00), EASYSIMD_FLOAT32_C(351920.00),
        EASYSIMD_FLOAT32_C(-244744.00), EASYSIMD_FLOAT32_C(517741.00), EASYSIMD_FLOAT32_C(-93617.00), EASYSIMD_FLOAT32_C( 20800.00),
        EASYSIMD_FLOAT32_C(-65214.00), EASYSIMD_FLOAT32_C(  8462.00), EASYSIMD_FLOAT32_C(-78288.00), EASYSIMD_FLOAT32_C( -6481.00),
        EASYSIMD_FLOAT32_C(-121311.00), EASYSIMD_FLOAT32_C(-311398.00), EASYSIMD_FLOAT32_C(-66658.00), EASYSIMD_FLOAT32_C(-43212.00) },
      { EASYSIMD_FLOAT32_C( 29757.00), EASYSIMD_FLOAT32_C(   -75.00), EASYSIMD_FLOAT32_C(402965.00), EASYSIMD_FLOAT32_C(351920.00),
        EASYSIMD_FLOAT32_C(-244745.00), EASYSIMD_FLOAT32_C(517741.00), EASYSIMD_FLOAT32_C(-93617.00), EASYSIMD_FLOAT32_C( 20799.00),
        EASYSIMD_FLOAT32_C(-65214.00), EASYSIMD_FLOAT32_C(  8461.00), EASYSIMD_FLOAT32_C(-78289.00), EASYSIMD_FLOAT32_C( -6481.00),
        EASYSIMD_FLOAT32_C(-121311.00), EASYSIMD_FLOAT32_C(-311399.00), EASYSIMD_FLOAT32_C(-66659.00), EASYSIMD_FLOAT32_C(-43213.00) },
      { EASYSIMD_FLOAT32_C( 29758.00), EASYSIMD_FLOAT32_C(   -74.00), EASYSIMD_FLOAT32_C(402966.00), EASYSIMD_FLOAT32_C(351921.00),
        EASYSIMD_FLOAT32_C(-244744.00), EASYSIMD_FLOAT32_C(517742.00), EASYSIMD_FLOAT32_C(-93616.00), EASYSIMD_FLOAT32_C( 20800.00),
        EASYSIMD_FLOAT32_C(-65213.00), EASYSIMD_FLOAT32_C(  8462.00), EASYSIMD_FLOAT32_C(-78288.00), EASYSIMD_FLOAT32_C( -6480.00),
        EASYSIMD_FLOAT32_C(-121310.00), EASYSIMD_FLOAT32_C(-311398.00), EASYSIMD_FLOAT32_C(-66658.00), EASYSIMD_FLOAT32_C(-43212.00) },
      { EASYSIMD_FLOAT32_C( 29757.00), EASYSIMD_FLOAT32_C(   -74.00), EASYSIMD_FLOAT32_C(402965.00), EASYSIMD_FLOAT32_C(351920.00),
        EASYSIMD_FLOAT32_C(-244744.00), EASYSIMD_FLOAT32_C(517741.00), EASYSIMD_FLOAT32_C(-93616.00), EASYSIMD_FLOAT32_C( 20799.00),
        EASYSIMD_FLOAT32_C(-65213.00), EASYSIMD_FLOAT32_C(  8461.00), EASYSIMD_FLOAT32_C(-78288.00), EASYSIMD_FLOAT32_C( -6480.00),
        EASYSIMD_FLOAT32_C(-121310.00), EASYSIMD_FLOAT32_C(-311398.00), EASYSIMD_FLOAT32_C(-66658.00), EASYSIMD_FLOAT32_C(-43212.00) },
      { EASYSIMD_FLOAT32_C( 29758.00), EASYSIMD_FLOAT32_C(   -75.00), EASYSIMD_FLOAT32_C(402966.00), EASYSIMD_FLOAT32_C(351920.00),
        EASYSIMD_FLOAT32_C(-244744.00), EASYSIMD_FLOAT32_C(517741.00), EASYSIMD_FLOAT32_C(-93617.00), EASYSIMD_FLOAT32_C( 20800.00),
        EASYSIMD_FLOAT32_C(-65214.00), EASYSIMD_FLOAT32_C(  8462.00), EASYSIMD_FLOAT32_C(-78288.00), EASYSIMD_FLOAT32_C( -6481.00),
        EASYSIMD_FLOAT32_C(-121311.00), EASYSIMD_FLOAT32_C(-311398.00), EASYSIMD_FLOAT32_C(-66658.00), EASYSIMD_FLOAT32_C(-43212.00) } },
    { { EASYSIMD_FLOAT32_C(   467.36), EASYSIMD_FLOAT32_C(   636.34), EASYSIMD_FLOAT32_C(  -881.25), EASYSIMD_FLOAT32_C(   148.89),
        EASYSIMD_FLOAT32_C(   719.14), EASYSIMD_FLOAT32_C(  -181.38), EASYSIMD_FLOAT32_C(   441.41), EASYSIMD_FLOAT32_C(   189.59),
        EASYSIMD_FLOAT32_C(   463.09), EASYSIMD_FLOAT32_C(  -729.51), EASYSIMD_FLOAT32_C(   426.23), EASYSIMD_FLOAT32_C(  -953.90),
        EASYSIMD_FLOAT32_C(   854.42), EASYSIMD_FLOAT32_C(  -916.52), EASYSIMD_FLOAT32_C(    92.92), EASYSIMD_FLOAT32_C(  -284.72) },
      { EASYSIMD_FLOAT32_C(   -33.67), EASYSIMD_FLOAT32_C(   370.93), EASYSIMD_FLOAT32_C(   348.06), EASYSIMD_FLOAT32_C(   607.23),
        EASYSIMD_FLOAT32_C(   806.45), EASYSIMD_FLOAT32_C(  -340.03), EASYSIMD_FLOAT32_C(  -422.17), EASYSIMD_FLOAT32_C(  -332.17),
        EASYSIMD_FLOAT32_C(   636.17), EASYSIMD_FLOAT32_C(  -964.13), EASYSIMD_FLOAT32_C(   676.32), EASYSIMD_FLOAT32_C(   -72.91),
        EASYSIMD_FLOAT32_C(  -497.41), EASYSIMD_FLOAT32_C(  -129.20), EASYSIMD_FLOAT32_C(     4.15), EASYSIMD_FLOAT32_C(   969.95) },
      { EASYSIMD_FLOAT32_C(-15736.00), EASYSIMD_FLOAT32_C(236038.00), EASYSIMD_FLOAT32_C(-306728.00), EASYSIMD_FLOAT32_C( 90410.00),
        EASYSIMD_FLOAT32_C(579950.00), EASYSIMD_FLOAT32_C( 61675.00), EASYSIMD_FLOAT32_C(-186350.00), EASYSIMD_FLOAT32_C(-62976.00),
        EASYSIMD_FLOAT32_C(294604.00), EASYSIMD_FLOAT32_C(703342.00), EASYSIMD_FLOAT32_C(288268.00), EASYSIMD_FLOAT32_C( 69549.00),
        EASYSIMD_FLOAT32_C(-424997.00), EASYSIMD_FLOAT32_C(118414.00), EASYSIMD_FLOAT32_C(   386.00), EASYSIMD_FLOAT32_C(-276164.00) },
      { EASYSIMD_FLOAT32_C(-15737.00), EASYSIMD_FLOAT32_C(236037.00), EASYSIMD_FLOAT32_C(-306728.00), EASYSIMD_FLOAT32_C( 90410.00),
        EASYSIMD_FLOAT32_C(579950.00), EASYSIMD_FLOAT32_C( 61674.00), EASYSIMD_FLOAT32_C(-186351.00), EASYSIMD_FLOAT32_C(-62977.00),
        EASYSIMD_FLOAT32_C(294603.00), EASYSIMD_FLOAT32_C(703342.00), EASYSIMD_FLOAT32_C(288267.00), EASYSIMD_FLOAT32_C( 69548.00),
        EASYSIMD_FLOAT32_C(-424998.00), EASYSIMD_FLOAT32_C(118414.00), EASYSIMD_FLOAT32_C(   385.00), EASYSIMD_FLOAT32_C(-276165.00) },
      { EASYSIMD_FLOAT32_C(-15736.00), EASYSIMD_FLOAT32_C(236038.00), EASYSIMD_FLOAT32_C(-306727.00), EASYSIMD_FLOAT32_C( 90411.00),
        EASYSIMD_FLOAT32_C(579951.00), EASYSIMD_FLOAT32_C( 61675.00), EASYSIMD_FLOAT32_C(-186350.00), EASYSIMD_FLOAT32_C(-62976.00),
        EASYSIMD_FLOAT32_C(294604.00), EASYSIMD_FLOAT32_C(703343.00), EASYSIMD_FLOAT32_C(288268.00), EASYSIMD_FLOAT32_C( 69549.00),
        EASYSIMD_FLOAT32_C(-424997.00), EASYSIMD_FLOAT32_C(118415.00), EASYSIMD_FLOAT32_C(   386.00), EASYSIMD_FLOAT32_C(-276164.00) },
      { EASYSIMD_FLOAT32_C(-15736.00), EASYSIMD_FLOAT32_C(236037.00), EASYSIMD_FLOAT32_C(-306727.00), EASYSIMD_FLOAT32_C( 90410.00),
        EASYSIMD_FLOAT32_C(579950.00), EASYSIMD_FLOAT32_C( 61674.00), EASYSIMD_FLOAT32_C(-186350.00), EASYSIMD_FLOAT32_C(-62976.00),
        EASYSIMD_FLOAT32_C(294603.00), EASYSIMD_FLOAT32_C(703342.00), EASYSIMD_FLOAT32_C(288267.00), EASYSIMD_FLOAT32_C( 69548.00),
        EASYSIMD_FLOAT32_C(-424997.00), EASYSIMD_FLOAT32_C(118414.00), EASYSIMD_FLOAT32_C(   385.00), EASYSIMD_FLOAT32_C(-276164.00) },
      { EASYSIMD_FLOAT32_C(-15736.00), EASYSIMD_FLOAT32_C(236038.00), EASYSIMD_FLOAT32_C(-306728.00), EASYSIMD_FLOAT32_C( 90410.00),
        EASYSIMD_FLOAT32_C(579950.00), EASYSIMD_FLOAT32_C( 61675.00), EASYSIMD_FLOAT32_C(-186350.00), EASYSIMD_FLOAT32_C(-62976.00),
        EASYSIMD_FLOAT32_C(294604.00), EASYSIMD_FLOAT32_C(703342.00), EASYSIMD_FLOAT32_C(288268.00), EASYSIMD_FLOAT32_C( 69549.00),
        EASYSIMD_FLOAT32_C(-424997.00), EASYSIMD_FLOAT32_C(118414.00), EASYSIMD_FLOAT32_C(   386.00), EASYSIMD_FLOAT32_C(-276164.00) } },
    { { EASYSIMD_FLOAT32_C(  -492.87), EASYSIMD_FLOAT32_C(   122.90), EASYSIMD_FLOAT32_C(   118.84), EASYSIMD_FLOAT32_C(  -773.73),
        EASYSIMD_FLOAT32_C(   941.52), EASYSIMD_FLOAT32_C(  -439.75), EASYSIMD_FLOAT32_C(   415.86), EASYSIMD_FLOAT32_C(   404.61),
        EASYSIMD_FLOAT32_C(  -169.26), EASYSIMD_FLOAT32_C(  -157.91), EASYSIMD_FLOAT32_C(   450.71), EASYSIMD_FLOAT32_C(  -314.84),
        EASYSIMD_FLOAT32_C(   -74.43), EASYSIMD_FLOAT32_C(  -456.38), EASYSIMD_FLOAT32_C(   400.44), EASYSIMD_FLOAT32_C(   891.90) },
      { EASYSIMD_FLOAT32_C(   914.56), EASYSIMD_FLOAT32_C(  -251.50), EASYSIMD_FLOAT32_C(   499.13), EASYSIMD_FLOAT32_C(   721.01),
        EASYSIMD_FLOAT32_C(   408.47), EASYSIMD_FLOAT32_C(  -923.05), EASYSIMD_FLOAT32_C(  -611.17), EASYSIMD_FLOAT32_C(    44.64),
        EASYSIMD_FLOAT32_C(  -887.18), EASYSIMD_FLOAT32_C(  -934.85), EASYSIMD_FLOAT32_C(   971.74), EASYSIMD_FLOAT32_C(  -384.59),
        EASYSIMD_FLOAT32_C(   -64.05), EASYSIMD_FLOAT32_C(   -24.11), EASYSIMD_FLOAT32_C(  -414.63), EASYSIMD_FLOAT32_C(   443.08) },
      { EASYSIMD_FLOAT32_C(-450759.00), EASYSIMD_FLOAT32_C(-30909.00), EASYSIMD_FLOAT32_C( 59317.00), EASYSIMD_FLOAT32_C(-557867.00),
        EASYSIMD_FLOAT32_C(384583.00), EASYSIMD_FLOAT32_C(405911.00), EASYSIMD_FLOAT32_C(-254161.00), EASYSIMD_FLOAT32_C( 18062.00),
        EASYSIMD_FLOAT32_C(150164.00), EASYSIMD_FLOAT32_C(147622.00), EASYSIMD_FLOAT32_C(437973.00), EASYSIMD_FLOAT32_C(121084.00),
        EASYSIMD_FLOAT32_C(  4767.00), EASYSIMD_FLOAT32_C( 11003.00), EASYSIMD_FLOAT32_C(-166034.00), EASYSIMD_FLOAT32_C(395183.00) },
      { EASYSIMD_FLOAT32_C(-450760.00), EASYSIMD_FLOAT32_C(-30910.00), EASYSIMD_FLOAT32_C( 59316.00), EASYSIMD_FLOAT32_C(-557868.00),
        EASYSIMD_FLOAT32_C(384582.00), EASYSIMD_FLOAT32_C(405911.00), EASYSIMD_FLOAT32_C(-254162.00), EASYSIMD_FLOAT32_C( 18061.00),
        EASYSIMD_FLOAT32_C(150164.00), EASYSIMD_FLOAT32_C(147622.00), EASYSIMD_FLOAT32_C(437972.00), EASYSIMD_FLOAT32_C(121084.00),
        EASYSIMD_FLOAT32_C(  4767.00), EASYSIMD_FLOAT32_C( 11003.00), EASYSIMD_FLOAT32_C(-166035.00), EASYSIMD_FLOAT32_C(395183.00) },
      { EASYSIMD_FLOAT32_C(-450759.00), EASYSIMD_FLOAT32_C(-30909.00), EASYSIMD_FLOAT32_C( 59317.00), EASYSIMD_FLOAT32_C(-557867.00),
        EASYSIMD_FLOAT32_C(384583.00), EASYSIMD_FLOAT32_C(405912.00), EASYSIMD_FLOAT32_C(-254161.00), EASYSIMD_FLOAT32_C( 18062.00),
        EASYSIMD_FLOAT32_C(150165.00), EASYSIMD_FLOAT32_C(147623.00), EASYSIMD_FLOAT32_C(437973.00), EASYSIMD_FLOAT32_C(121085.00),
        EASYSIMD_FLOAT32_C(  4768.00), EASYSIMD_FLOAT32_C( 11004.00), EASYSIMD_FLOAT32_C(-166034.00), EASYSIMD_FLOAT32_C(395184.00) },
      { EASYSIMD_FLOAT32_C(-450759.00), EASYSIMD_FLOAT32_C(-30909.00), EASYSIMD_FLOAT32_C( 59316.00), EASYSIMD_FLOAT32_C(-557867.00),
        EASYSIMD_FLOAT32_C(384582.00), EASYSIMD_FLOAT32_C(405911.00), EASYSIMD_FLOAT32_C(-254161.00), EASYSIMD_FLOAT32_C( 18061.00),
        EASYSIMD_FLOAT32_C(150164.00), EASYSIMD_FLOAT32_C(147622.00), EASYSIMD_FLOAT32_C(437972.00), EASYSIMD_FLOAT32_C(121084.00),
        EASYSIMD_FLOAT32_C(  4767.00), EASYSIMD_FLOAT32_C( 11003.00), EASYSIMD_FLOAT32_C(-166034.00), EASYSIMD_FLOAT32_C(395183.00) },
      { EASYSIMD_FLOAT32_C(-450759.00), EASYSIMD_FLOAT32_C(-30909.00), EASYSIMD_FLOAT32_C( 59317.00), EASYSIMD_FLOAT32_C(-557867.00),
        EASYSIMD_FLOAT32_C(384583.00), EASYSIMD_FLOAT32_C(405911.00), EASYSIMD_FLOAT32_C(-254161.00), EASYSIMD_FLOAT32_C( 18062.00),
        EASYSIMD_FLOAT32_C(150164.00), EASYSIMD_FLOAT32_C(147622.00), EASYSIMD_FLOAT32_C(437973.00), EASYSIMD_FLOAT32_C(121084.00),
        EASYSIMD_FLOAT32_C(  4767.00), EASYSIMD_FLOAT32_C( 11003.00), EASYSIMD_FLOAT32_C(-166034.00), EASYSIMD_FLOAT32_C(395183.00) } },
    { { EASYSIMD_FLOAT32_C(  -901.21), EASYSIMD_FLOAT32_C(   704.20), EASYSIMD_FLOAT32_C(   669.35), EASYSIMD_FLOAT32_C(  -959.69),
        EASYSIMD_FLOAT32_C(  -735.55), EASYSIMD_FLOAT32_C(    85.20), EASYSIMD_FLOAT32_C(   444.92), EASYSIMD_FLOAT32_C(    95.19),
        EASYSIMD_FLOAT32_C(   927.29), EASYSIMD_FLOAT32_C(  -104.37), EASYSIMD_FLOAT32_C(   780.35), EASYSIMD_FLOAT32_C(  -147.14),
        EASYSIMD_FLOAT32_C(   439.25), EASYSIMD_FLOAT32_C(   180.80), EASYSIMD_FLOAT32_C(  -255.24), EASYSIMD_FLOAT32_C(   353.81) },
      { EASYSIMD_FLOAT32_C(   929.30), EASYSIMD_FLOAT32_C(  -756.11), EASYSIMD_FLOAT32_C(    74.82), EASYSIMD_FLOAT32_C(   337.77),
        EASYSIMD_FLOAT32_C(  -679.16), EASYSIMD_FLOAT32_C(   463.65), EASYSIMD_FLOAT32_C(  -617.58), EASYSIMD_FLOAT32_C(  -566.34),
        EASYSIMD_FLOAT32_C(   528.80), EASYSIMD_FLOAT32_C(  -645.84), EASYSIMD_FLOAT32_C(    49.07), EASYSIMD_FLOAT32_C(  -535.25),
        EASYSIMD_FLOAT32_C(   330.05), EASYSIMD_FLOAT32_C(   634.44), EASYSIMD_FLOAT32_C(   907.83), EASYSIMD_FLOAT32_C(   428.84) },
      { EASYSIMD_FLOAT32_C(-837494.00), EASYSIMD_FLOAT32_C(-532453.00), EASYSIMD_FLOAT32_C( 50081.00), EASYSIMD_FLOAT32_C(-324154.00),
        EASYSIMD_FLOAT32_C(499556.00), EASYSIMD_FLOAT32_C( 39503.00), EASYSIMD_FLOAT32_C(-274774.00), EASYSIMD_FLOAT32_C(-53910.00),
        EASYSIMD_FLOAT32_C(490351.00), EASYSIMD_FLOAT32_C( 67406.00), EASYSIMD_FLOAT32_C( 38292.00), EASYSIMD_FLOAT32_C( 78757.00),
        EASYSIMD_FLOAT32_C(144974.00), EASYSIMD_FLOAT32_C(114707.00), EASYSIMD_FLOAT32_C(-231715.00), EASYSIMD_FLOAT32_C(151728.00) },
      { EASYSIMD_FLOAT32_C(-837495.00), EASYSIMD_FLOAT32_C(-532453.00), EASYSIMD_FLOAT32_C( 50080.00), EASYSIMD_FLOAT32_C(-324155.00),
        EASYSIMD_FLOAT32_C(499556.00), EASYSIMD_FLOAT32_C( 39502.00), EASYSIMD_FLOAT32_C(-274774.00), EASYSIMD_FLOAT32_C(-53910.00),
        EASYSIMD_FLOAT32_C(490350.00), EASYSIMD_FLOAT32_C( 67406.00), EASYSIMD_FLOAT32_C( 38291.00), EASYSIMD_FLOAT32_C( 78756.00),
        EASYSIMD_FLOAT32_C(144974.00), EASYSIMD_FLOAT32_C(114706.00), EASYSIMD_FLOAT32_C(-231715.00), EASYSIMD_FLOAT32_C(151727.00) },
      { EASYSIMD_FLOAT32_C(-837494.00), EASYSIMD_FLOAT32_C(-532452.00), EASYSIMD_FLOAT32_C( 50081.00), EASYSIMD_FLOAT32_C(-324154.00),
        EASYSIMD_FLOAT32_C(499557.00), EASYSIMD_FLOAT32_C( 39503.00), EASYSIMD_FLOAT32_C(-274773.00), EASYSIMD_FLOAT32_C(-53909.00),
        EASYSIMD_FLOAT32_C(490351.00), EASYSIMD_FLOAT32_C( 67407.00), EASYSIMD_FLOAT32_C( 38292.00), EASYSIMD_FLOAT32_C( 78757.00),
        EASYSIMD_FLOAT32_C(144975.00), EASYSIMD_FLOAT32_C(114707.00), EASYSIMD_FLOAT32_C(-231714.00), EASYSIMD_FLOAT32_C(151728.00) },
      { EASYSIMD_FLOAT32_C(-837494.00), EASYSIMD_FLOAT32_C(-532452.00), EASYSIMD_FLOAT32_C( 50080.00), EASYSIMD_FLOAT32_C(-324154.00),
        EASYSIMD_FLOAT32_C(499556.00), EASYSIMD_FLOAT32_C( 39502.00), EASYSIMD_FLOAT32_C(-274773.00), EASYSIMD_FLOAT32_C(-53909.00),
        EASYSIMD_FLOAT32_C(490350.00), EASYSIMD_FLOAT32_C( 67406.00), EASYSIMD_FLOAT32_C( 38291.00), EASYSIMD_FLOAT32_C( 78756.00),
        EASYSIMD_FLOAT32_C(144974.00), EASYSIMD_FLOAT32_C(114706.00), EASYSIMD_FLOAT32_C(-231714.00), EASYSIMD_FLOAT32_C(151727.00) },
      { EASYSIMD_FLOAT32_C(-837494.00), EASYSIMD_FLOAT32_C(-532453.00), EASYSIMD_FLOAT32_C( 50081.00), EASYSIMD_FLOAT32_C(-324154.00),
        EASYSIMD_FLOAT32_C(499556.00), EASYSIMD_FLOAT32_C( 39503.00), EASYSIMD_FLOAT32_C(-274774.00), EASYSIMD_FLOAT32_C(-53910.00),
        EASYSIMD_FLOAT32_C(490351.00), EASYSIMD_FLOAT32_C( 67406.00), EASYSIMD_FLOAT32_C( 38292.00), EASYSIMD_FLOAT32_C( 78757.00),
        EASYSIMD_FLOAT32_C(144974.00), EASYSIMD_FLOAT32_C(114707.00), EASYSIMD_FLOAT32_C(-231715.00), EASYSIMD_FLOAT32_C(151728.00) } },
    { { EASYSIMD_FLOAT32_C(   338.64), EASYSIMD_FLOAT32_C(   577.18), EASYSIMD_FLOAT32_C(   469.15), EASYSIMD_FLOAT32_C(   603.09),
        EASYSIMD_FLOAT32_C(  -337.62), EASYSIMD_FLOAT32_C(   -85.93), EASYSIMD_FLOAT32_C(  -301.71), EASYSIMD_FLOAT32_C(  -410.33),
        EASYSIMD_FLOAT32_C(   809.70), EASYSIMD_FLOAT32_C(  -521.36), EASYSIMD_FLOAT32_C(   442.53), EASYSIMD_FLOAT32_C(   248.95),
        EASYSIMD_FLOAT32_C(   659.44), EASYSIMD_FLOAT32_C(  -812.71), EASYSIMD_FLOAT32_C(  -397.23), EASYSIMD_FLOAT32_C(   588.74) },
      { EASYSIMD_FLOAT32_C(  -568.82), EASYSIMD_FLOAT32_C(   677.58), EASYSIMD_FLOAT32_C(   -73.49), EASYSIMD_FLOAT32_C(  -247.98),
        EASYSIMD_FLOAT32_C(   141.23), EASYSIMD_FLOAT32_C(   308.93), EASYSIMD_FLOAT32_C(   185.67), EASYSIMD_FLOAT32_C(  -329.97),
        EASYSIMD_FLOAT32_C(   663.08), EASYSIMD_FLOAT32_C(  -765.25), EASYSIMD_FLOAT32_C(   134.78), EASYSIMD_FLOAT32_C(    -6.87),
        EASYSIMD_FLOAT32_C(   869.18), EASYSIMD_FLOAT32_C(    42.60), EASYSIMD_FLOAT32_C(  -578.04), EASYSIMD_FLOAT32_C(   207.82) },
      { EASYSIMD_FLOAT32_C(-192625.00), EASYSIMD_FLOAT32_C(391086.00), EASYSIMD_FLOAT32_C(-34478.00), EASYSIMD_FLOAT32_C(-149554.00),
        EASYSIMD_FLOAT32_C(-47682.00), EASYSIMD_FLOAT32_C(-26546.00), EASYSIMD_FLOAT32_C(-56018.00), EASYSIMD_FLOAT32_C(135397.00),
        EASYSIMD_FLOAT32_C(536896.00), EASYSIMD_FLOAT32_C(398971.00), EASYSIMD_FLOAT32_C( 59644.00), EASYSIMD_FLOAT32_C( -1710.00),
        EASYSIMD_FLOAT32_C(573172.00), EASYSIMD_FLOAT32_C(-34621.00), EASYSIMD_FLOAT32_C(229615.00), EASYSIMD_FLOAT32_C(122352.00) },
      { EASYSIMD_FLOAT32_C(-192626.00), EASYSIMD_FLOAT32_C(391085.00), EASYSIMD_FLOAT32_C(-34478.00), EASYSIMD_FLOAT32_C(-149555.00),
        EASYSIMD_FLOAT32_C(-47683.00), EASYSIMD_FLOAT32_C(-26547.00), EASYSIMD_FLOAT32_C(-56019.00), EASYSIMD_FLOAT32_C(135396.00),
        EASYSIMD_FLOAT32_C(536895.00), EASYSIMD_FLOAT32_C(398970.00), EASYSIMD_FLOAT32_C( 59644.00), EASYSIMD_FLOAT32_C( -1711.00),
        EASYSIMD_FLOAT32_C(573172.00), EASYSIMD_FLOAT32_C(-34622.00), EASYSIMD_FLOAT32_C(229614.00), EASYSIMD_FLOAT32_C(122351.00) },
      { EASYSIMD_FLOAT32_C(-192625.00), EASYSIMD_FLOAT32_C(391086.00), EASYSIMD_FLOAT32_C(-34477.00), EASYSIMD_FLOAT32_C(-149554.00),
        EASYSIMD_FLOAT32_C(-47682.00), EASYSIMD_FLOAT32_C(-26546.00), EASYSIMD_FLOAT32_C(-56018.00), EASYSIMD_FLOAT32_C(135397.00),
        EASYSIMD_FLOAT32_C(536896.00), EASYSIMD_FLOAT32_C(398971.00), EASYSIMD_FLOAT32_C( 59645.00), EASYSIMD_FLOAT32_C( -1710.00),
        EASYSIMD_FLOAT32_C(573173.00), EASYSIMD_FLOAT32_C(-34621.00), EASYSIMD_FLOAT32_C(229615.00), EASYSIMD_FLOAT32_C(122352.00) },
      { EASYSIMD_FLOAT32_C(-192625.00), EASYSIMD_FLOAT32_C(391085.00), EASYSIMD_FLOAT32_C(-34477.00), EASYSIMD_FLOAT32_C(-149554.00),
        EASYSIMD_FLOAT32_C(-47682.00), EASYSIMD_FLOAT32_C(-26546.00), EASYSIMD_FLOAT32_C(-56018.00), EASYSIMD_FLOAT32_C(135396.00),
        EASYSIMD_FLOAT32_C(536895.00), EASYSIMD_FLOAT32_C(398970.00), EASYSIMD_FLOAT32_C( 59644.00), EASYSIMD_FLOAT32_C( -1710.00),
        EASYSIMD_FLOAT32_C(573172.00), EASYSIMD_FLOAT32_C(-34621.00), EASYSIMD_FLOAT32_C(229614.00), EASYSIMD_FLOAT32_C(122351.00) },
      { EASYSIMD_FLOAT32_C(-192625.00), EASYSIMD_FLOAT32_C(391086.00), EASYSIMD_FLOAT32_C(-34478.00), EASYSIMD_FLOAT32_C(-149554.00),
        EASYSIMD_FLOAT32_C(-47682.00), EASYSIMD_FLOAT32_C(-26546.00), EASYSIMD_FLOAT32_C(-56018.00), EASYSIMD_FLOAT32_C(135397.00),
        EASYSIMD_FLOAT32_C(536896.00), EASYSIMD_FLOAT32_C(398971.00), EASYSIMD_FLOAT32_C( 59644.00), EASYSIMD_FLOAT32_C( -1710.00),
        EASYSIMD_FLOAT32_C(573172.00), EASYSIMD_FLOAT32_C(-34621.00), EASYSIMD_FLOAT32_C(229615.00), EASYSIMD_FLOAT32_C(122352.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 r;
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);

    easysimd__m512 nearest_inf = easysimd_mm512_loadu_ps(test_vec[i].nearest_inf);
    easysimd__m512 neg_inf = easysimd_mm512_loadu_ps(test_vec[i].neg_inf);
    easysimd__m512 pos_inf = easysimd_mm512_loadu_ps(test_vec[i].pos_inf);
    easysimd__m512 zero = easysimd_mm512_loadu_ps(test_vec[i].zero);
    easysimd__m512 direction = easysimd_mm512_loadu_ps(test_vec[i].direction);

    r = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512_close(r, nearest_inf, 1);

    r = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512_close(r, neg_inf, 1);

    r = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512_close(r, pos_inf, 1);

    r = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mul_round_ps");
    easysimd_assert_m512_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512 nearest_inf = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512 neg_inf = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512 pos_inf = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512 zero = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512 direction = easysimd_mm512_mul_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, nearest_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, neg_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, pos_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, zero, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, direction, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_mask_mul_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
       { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   229.27), EASYSIMD_FLOAT32_C(  -114.91), EASYSIMD_FLOAT32_C(   520.43), EASYSIMD_FLOAT32_C(  -755.19),
                         EASYSIMD_FLOAT32_C(   -68.64), EASYSIMD_FLOAT32_C(   632.30), EASYSIMD_FLOAT32_C(    98.14), EASYSIMD_FLOAT32_C(   455.87),
                         EASYSIMD_FLOAT32_C(  -873.22), EASYSIMD_FLOAT32_C(  -223.86), EASYSIMD_FLOAT32_C(   181.32), EASYSIMD_FLOAT32_C(   364.92),
                         EASYSIMD_FLOAT32_C(   946.51), EASYSIMD_FLOAT32_C(    22.05), EASYSIMD_FLOAT32_C(   444.47), EASYSIMD_FLOAT32_C(  -746.17)),
      UINT16_C( 6152),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   702.34), EASYSIMD_FLOAT32_C(   368.95), EASYSIMD_FLOAT32_C(   161.91), EASYSIMD_FLOAT32_C(   459.04),
                         EASYSIMD_FLOAT32_C(  -828.47), EASYSIMD_FLOAT32_C(   943.39), EASYSIMD_FLOAT32_C(   202.26), EASYSIMD_FLOAT32_C(   112.87),
                         EASYSIMD_FLOAT32_C(   382.91), EASYSIMD_FLOAT32_C(   124.14), EASYSIMD_FLOAT32_C(   954.24), EASYSIMD_FLOAT32_C(  -214.34),
                         EASYSIMD_FLOAT32_C(  -998.93), EASYSIMD_FLOAT32_C(  -255.92), EASYSIMD_FLOAT32_C(    57.01), EASYSIMD_FLOAT32_C(  -391.73)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -206.12), EASYSIMD_FLOAT32_C(  -322.43), EASYSIMD_FLOAT32_C(  -456.42), EASYSIMD_FLOAT32_C(   258.98),
                         EASYSIMD_FLOAT32_C(   291.55), EASYSIMD_FLOAT32_C(  -459.80), EASYSIMD_FLOAT32_C(   286.61), EASYSIMD_FLOAT32_C(    15.13),
                         EASYSIMD_FLOAT32_C(  -772.68), EASYSIMD_FLOAT32_C(  -503.52), EASYSIMD_FLOAT32_C(  -599.88), EASYSIMD_FLOAT32_C(   107.93),
                         EASYSIMD_FLOAT32_C(    -3.35), EASYSIMD_FLOAT32_C(  -993.69), EASYSIMD_FLOAT32_C(  -325.33), EASYSIMD_FLOAT32_C(   755.40)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   229.27), EASYSIMD_FLOAT32_C(  -114.91), EASYSIMD_FLOAT32_C(   520.43), EASYSIMD_FLOAT32_C(118882.19),
                         EASYSIMD_FLOAT32_C(-241540.41), EASYSIMD_FLOAT32_C(   632.30), EASYSIMD_FLOAT32_C(    98.14), EASYSIMD_FLOAT32_C(   455.87),
                         EASYSIMD_FLOAT32_C(  -873.22), EASYSIMD_FLOAT32_C(  -223.86), EASYSIMD_FLOAT32_C(   181.32), EASYSIMD_FLOAT32_C(   364.92),
                         EASYSIMD_FLOAT32_C(  3346.42), EASYSIMD_FLOAT32_C(    22.05), EASYSIMD_FLOAT32_C(   444.47), EASYSIMD_FLOAT32_C(  -746.17)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   974.52), EASYSIMD_FLOAT32_C(    42.80), EASYSIMD_FLOAT32_C(  -282.69), EASYSIMD_FLOAT32_C(  -590.27),
                         EASYSIMD_FLOAT32_C(   398.09), EASYSIMD_FLOAT32_C(   498.04), EASYSIMD_FLOAT32_C(   449.97), EASYSIMD_FLOAT32_C(  -357.92),
                         EASYSIMD_FLOAT32_C(  -441.74), EASYSIMD_FLOAT32_C(  -180.77), EASYSIMD_FLOAT32_C(  -289.47), EASYSIMD_FLOAT32_C(  -620.49),
                         EASYSIMD_FLOAT32_C(   763.75), EASYSIMD_FLOAT32_C(  -763.91), EASYSIMD_FLOAT32_C(  -576.44), EASYSIMD_FLOAT32_C(   698.61)),
      UINT16_C(15973),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -512.47), EASYSIMD_FLOAT32_C(  -526.12), EASYSIMD_FLOAT32_C(   488.92), EASYSIMD_FLOAT32_C(   -99.48),
                         EASYSIMD_FLOAT32_C(   579.58), EASYSIMD_FLOAT32_C(   601.36), EASYSIMD_FLOAT32_C(   900.16), EASYSIMD_FLOAT32_C(   871.84),
                         EASYSIMD_FLOAT32_C(   797.21), EASYSIMD_FLOAT32_C(   523.84), EASYSIMD_FLOAT32_C(  -923.94), EASYSIMD_FLOAT32_C(   -14.85),
                         EASYSIMD_FLOAT32_C(  -320.00), EASYSIMD_FLOAT32_C(  -463.51), EASYSIMD_FLOAT32_C(  -980.83), EASYSIMD_FLOAT32_C(  -194.63)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   558.10), EASYSIMD_FLOAT32_C(  -796.63), EASYSIMD_FLOAT32_C(  -676.04), EASYSIMD_FLOAT32_C(   908.46),
                         EASYSIMD_FLOAT32_C(  -870.82), EASYSIMD_FLOAT32_C(   691.21), EASYSIMD_FLOAT32_C(  -550.67), EASYSIMD_FLOAT32_C(   268.52),
                         EASYSIMD_FLOAT32_C(   837.19), EASYSIMD_FLOAT32_C(  -677.60), EASYSIMD_FLOAT32_C(  -171.06), EASYSIMD_FLOAT32_C(   -56.18),
                         EASYSIMD_FLOAT32_C(   490.37), EASYSIMD_FLOAT32_C(   -61.61), EASYSIMD_FLOAT32_C(  -109.46), EASYSIMD_FLOAT32_C(  -710.13)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   974.52), EASYSIMD_FLOAT32_C(    42.80), EASYSIMD_FLOAT32_C(-330529.47), EASYSIMD_FLOAT32_C(-90373.61),
                         EASYSIMD_FLOAT32_C(-504709.88), EASYSIMD_FLOAT32_C(415666.06), EASYSIMD_FLOAT32_C(-495691.06), EASYSIMD_FLOAT32_C(  -357.92),
                         EASYSIMD_FLOAT32_C(  -441.74), EASYSIMD_FLOAT32_C(-354954.00), EASYSIMD_FLOAT32_C(158049.17), EASYSIMD_FLOAT32_C(  -620.49),
                         EASYSIMD_FLOAT32_C(   763.75), EASYSIMD_FLOAT32_C( 28556.85), EASYSIMD_FLOAT32_C(  -576.44), EASYSIMD_FLOAT32_C(138212.61)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -360.39), EASYSIMD_FLOAT32_C(   643.16), EASYSIMD_FLOAT32_C(  -441.22), EASYSIMD_FLOAT32_C(   409.21),
                         EASYSIMD_FLOAT32_C(   666.49), EASYSIMD_FLOAT32_C(   893.19), EASYSIMD_FLOAT32_C(  -859.97), EASYSIMD_FLOAT32_C(  -253.09),
                         EASYSIMD_FLOAT32_C(  -516.49), EASYSIMD_FLOAT32_C(  -209.00), EASYSIMD_FLOAT32_C(  -119.77), EASYSIMD_FLOAT32_C(    -6.76),
                         EASYSIMD_FLOAT32_C(   978.44), EASYSIMD_FLOAT32_C(   847.98), EASYSIMD_FLOAT32_C(   812.41), EASYSIMD_FLOAT32_C(  -887.11)),
      UINT16_C(51212),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   923.25), EASYSIMD_FLOAT32_C(   251.46), EASYSIMD_FLOAT32_C(   -49.04), EASYSIMD_FLOAT32_C(  -876.73),
                         EASYSIMD_FLOAT32_C(  -239.21), EASYSIMD_FLOAT32_C(   952.17), EASYSIMD_FLOAT32_C(  -247.44), EASYSIMD_FLOAT32_C(  -278.60),
                         EASYSIMD_FLOAT32_C(   877.29), EASYSIMD_FLOAT32_C(  -266.07), EASYSIMD_FLOAT32_C(  -839.50), EASYSIMD_FLOAT32_C(  -281.99),
                         EASYSIMD_FLOAT32_C(  -652.15), EASYSIMD_FLOAT32_C(  -877.11), EASYSIMD_FLOAT32_C(   527.90), EASYSIMD_FLOAT32_C(  -842.26)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -453.51), EASYSIMD_FLOAT32_C(   863.74), EASYSIMD_FLOAT32_C(   571.67), EASYSIMD_FLOAT32_C(   552.19),
                         EASYSIMD_FLOAT32_C(  -903.81), EASYSIMD_FLOAT32_C(  -274.48), EASYSIMD_FLOAT32_C(   891.44), EASYSIMD_FLOAT32_C(    28.40),
                         EASYSIMD_FLOAT32_C(   753.09), EASYSIMD_FLOAT32_C(   415.38), EASYSIMD_FLOAT32_C(  -974.66), EASYSIMD_FLOAT32_C(  -864.92),
                         EASYSIMD_FLOAT32_C(  -696.24), EASYSIMD_FLOAT32_C(  -279.21), EASYSIMD_FLOAT32_C(  -548.00), EASYSIMD_FLOAT32_C(     3.33)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-418703.12), EASYSIMD_FLOAT32_C(217196.06), EASYSIMD_FLOAT32_C(  -441.22), EASYSIMD_FLOAT32_C(   409.21),
                         EASYSIMD_FLOAT32_C(216200.39), EASYSIMD_FLOAT32_C(   893.19), EASYSIMD_FLOAT32_C(  -859.97), EASYSIMD_FLOAT32_C(  -253.09),
                         EASYSIMD_FLOAT32_C(  -516.49), EASYSIMD_FLOAT32_C(  -209.00), EASYSIMD_FLOAT32_C(  -119.77), EASYSIMD_FLOAT32_C(    -6.76),
                         EASYSIMD_FLOAT32_C(454052.94), EASYSIMD_FLOAT32_C(244897.88), EASYSIMD_FLOAT32_C(   812.41), EASYSIMD_FLOAT32_C(  -887.11)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   325.22), EASYSIMD_FLOAT32_C(   570.14), EASYSIMD_FLOAT32_C(   680.58), EASYSIMD_FLOAT32_C(  -511.81),
                         EASYSIMD_FLOAT32_C(  -117.17), EASYSIMD_FLOAT32_C(  -613.97), EASYSIMD_FLOAT32_C(   182.50), EASYSIMD_FLOAT32_C(   480.12),
                         EASYSIMD_FLOAT32_C(  -750.83), EASYSIMD_FLOAT32_C(   220.35), EASYSIMD_FLOAT32_C(   724.25), EASYSIMD_FLOAT32_C(   984.66),
                         EASYSIMD_FLOAT32_C(   871.75), EASYSIMD_FLOAT32_C(  -772.37), EASYSIMD_FLOAT32_C(   130.52), EASYSIMD_FLOAT32_C(   736.76)),
      UINT16_C(42108),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   960.66), EASYSIMD_FLOAT32_C(  -509.07), EASYSIMD_FLOAT32_C(   477.59), EASYSIMD_FLOAT32_C(  -132.21),
                         EASYSIMD_FLOAT32_C(   254.98), EASYSIMD_FLOAT32_C(   600.06), EASYSIMD_FLOAT32_C(    43.49), EASYSIMD_FLOAT32_C(   466.19),
                         EASYSIMD_FLOAT32_C(    22.31), EASYSIMD_FLOAT32_C(  -551.17), EASYSIMD_FLOAT32_C(  -167.87), EASYSIMD_FLOAT32_C(   278.33),
                         EASYSIMD_FLOAT32_C(  -232.38), EASYSIMD_FLOAT32_C(   650.45), EASYSIMD_FLOAT32_C(  -297.78), EASYSIMD_FLOAT32_C(  -280.35)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   562.20), EASYSIMD_FLOAT32_C(  -287.04), EASYSIMD_FLOAT32_C(   876.78), EASYSIMD_FLOAT32_C(   669.33),
                         EASYSIMD_FLOAT32_C(   940.57), EASYSIMD_FLOAT32_C(  -280.66), EASYSIMD_FLOAT32_C(    24.42), EASYSIMD_FLOAT32_C(  -147.12),
                         EASYSIMD_FLOAT32_C(   -57.84), EASYSIMD_FLOAT32_C(   841.25), EASYSIMD_FLOAT32_C(  -446.10), EASYSIMD_FLOAT32_C(  -973.24),
                         EASYSIMD_FLOAT32_C(   869.66), EASYSIMD_FLOAT32_C(   982.80), EASYSIMD_FLOAT32_C(  -763.04), EASYSIMD_FLOAT32_C(  -245.47)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(540083.06), EASYSIMD_FLOAT32_C(   570.14), EASYSIMD_FLOAT32_C(418741.38), EASYSIMD_FLOAT32_C(  -511.81),
                         EASYSIMD_FLOAT32_C(  -117.17), EASYSIMD_FLOAT32_C(-168412.84), EASYSIMD_FLOAT32_C(   182.50), EASYSIMD_FLOAT32_C(   480.12),
                         EASYSIMD_FLOAT32_C(  -750.83), EASYSIMD_FLOAT32_C(-463671.75), EASYSIMD_FLOAT32_C( 74886.80), EASYSIMD_FLOAT32_C(-270881.88),
                         EASYSIMD_FLOAT32_C(-202091.59), EASYSIMD_FLOAT32_C(639262.25), EASYSIMD_FLOAT32_C(   130.52), EASYSIMD_FLOAT32_C(   736.76)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -770.00), EASYSIMD_FLOAT32_C(   327.27), EASYSIMD_FLOAT32_C(  -642.48), EASYSIMD_FLOAT32_C(   864.18),
                         EASYSIMD_FLOAT32_C(  -775.21), EASYSIMD_FLOAT32_C(   -92.35), EASYSIMD_FLOAT32_C(  -864.16), EASYSIMD_FLOAT32_C(    80.78),
                         EASYSIMD_FLOAT32_C(  -974.40), EASYSIMD_FLOAT32_C(  -299.06), EASYSIMD_FLOAT32_C(  -754.35), EASYSIMD_FLOAT32_C(  -147.65),
                         EASYSIMD_FLOAT32_C(  -797.65), EASYSIMD_FLOAT32_C(   829.71), EASYSIMD_FLOAT32_C(   269.35), EASYSIMD_FLOAT32_C(   372.83)),
      UINT16_C(61342),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -220.25), EASYSIMD_FLOAT32_C(   802.77), EASYSIMD_FLOAT32_C(  -755.69), EASYSIMD_FLOAT32_C(   -58.33),
                         EASYSIMD_FLOAT32_C(   587.03), EASYSIMD_FLOAT32_C(   375.88), EASYSIMD_FLOAT32_C(   775.50), EASYSIMD_FLOAT32_C(  -179.11),
                         EASYSIMD_FLOAT32_C(   184.41), EASYSIMD_FLOAT32_C(  -603.91), EASYSIMD_FLOAT32_C(  -170.90), EASYSIMD_FLOAT32_C(  -781.45),
                         EASYSIMD_FLOAT32_C(  -860.97), EASYSIMD_FLOAT32_C(  -616.84), EASYSIMD_FLOAT32_C(   704.72), EASYSIMD_FLOAT32_C(  -251.07)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -62.78), EASYSIMD_FLOAT32_C(  -149.86), EASYSIMD_FLOAT32_C(   102.32), EASYSIMD_FLOAT32_C(  -271.05),
                         EASYSIMD_FLOAT32_C(  -465.52), EASYSIMD_FLOAT32_C(   979.82), EASYSIMD_FLOAT32_C(   499.92), EASYSIMD_FLOAT32_C(    32.84),
                         EASYSIMD_FLOAT32_C(   792.53), EASYSIMD_FLOAT32_C(   466.38), EASYSIMD_FLOAT32_C(  -301.08), EASYSIMD_FLOAT32_C(  -381.33),
                         EASYSIMD_FLOAT32_C(  -752.23), EASYSIMD_FLOAT32_C(    18.86), EASYSIMD_FLOAT32_C(  -462.80), EASYSIMD_FLOAT32_C(  -168.70)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C( 13827.29), EASYSIMD_FLOAT32_C(-120303.12), EASYSIMD_FLOAT32_C(-77322.20), EASYSIMD_FLOAT32_C(   864.18),
                         EASYSIMD_FLOAT32_C(-273274.22), EASYSIMD_FLOAT32_C(368294.75), EASYSIMD_FLOAT32_C(387687.97), EASYSIMD_FLOAT32_C( -5881.97),
                         EASYSIMD_FLOAT32_C(146150.47), EASYSIMD_FLOAT32_C(  -299.06), EASYSIMD_FLOAT32_C(  -754.35), EASYSIMD_FLOAT32_C(297990.31),
                         EASYSIMD_FLOAT32_C(647647.44), EASYSIMD_FLOAT32_C(-11633.60), EASYSIMD_FLOAT32_C(-326144.41), EASYSIMD_FLOAT32_C(   372.83)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -62.71), EASYSIMD_FLOAT32_C(   768.74), EASYSIMD_FLOAT32_C(   172.84), EASYSIMD_FLOAT32_C(  -481.12),
                         EASYSIMD_FLOAT32_C(   290.59), EASYSIMD_FLOAT32_C(  -598.30), EASYSIMD_FLOAT32_C(  -212.50), EASYSIMD_FLOAT32_C(   657.51),
                         EASYSIMD_FLOAT32_C(  -400.85), EASYSIMD_FLOAT32_C(   353.00), EASYSIMD_FLOAT32_C(  -898.98), EASYSIMD_FLOAT32_C(  -461.75),
                         EASYSIMD_FLOAT32_C(  -690.46), EASYSIMD_FLOAT32_C(  -171.93), EASYSIMD_FLOAT32_C(   135.84), EASYSIMD_FLOAT32_C(  -604.52)),
      UINT16_C(61129),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -267.02), EASYSIMD_FLOAT32_C(   -31.24), EASYSIMD_FLOAT32_C(  -859.84), EASYSIMD_FLOAT32_C(  -104.89),
                         EASYSIMD_FLOAT32_C(   -39.96), EASYSIMD_FLOAT32_C(   179.68), EASYSIMD_FLOAT32_C(   -71.01), EASYSIMD_FLOAT32_C(   557.26),
                         EASYSIMD_FLOAT32_C(   127.39), EASYSIMD_FLOAT32_C(   271.58), EASYSIMD_FLOAT32_C(  -162.76), EASYSIMD_FLOAT32_C(   248.01),
                         EASYSIMD_FLOAT32_C(   856.68), EASYSIMD_FLOAT32_C(   762.32), EASYSIMD_FLOAT32_C(   432.07), EASYSIMD_FLOAT32_C(   743.06)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -846.50), EASYSIMD_FLOAT32_C(   988.92), EASYSIMD_FLOAT32_C(   696.05), EASYSIMD_FLOAT32_C(   647.58),
                         EASYSIMD_FLOAT32_C(   780.95), EASYSIMD_FLOAT32_C(    46.01), EASYSIMD_FLOAT32_C(   -77.15), EASYSIMD_FLOAT32_C(  -747.70),
                         EASYSIMD_FLOAT32_C(  -416.84), EASYSIMD_FLOAT32_C(   679.81), EASYSIMD_FLOAT32_C(  -124.78), EASYSIMD_FLOAT32_C(  -976.50),
                         EASYSIMD_FLOAT32_C(  -745.93), EASYSIMD_FLOAT32_C(   116.64), EASYSIMD_FLOAT32_C(  -479.84), EASYSIMD_FLOAT32_C(   919.24)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(226032.42), EASYSIMD_FLOAT32_C(-30893.86), EASYSIMD_FLOAT32_C(-598491.62), EASYSIMD_FLOAT32_C(  -481.12),
                         EASYSIMD_FLOAT32_C(-31206.76), EASYSIMD_FLOAT32_C(  8267.08), EASYSIMD_FLOAT32_C(  5478.42), EASYSIMD_FLOAT32_C(   657.51),
                         EASYSIMD_FLOAT32_C(-53101.25), EASYSIMD_FLOAT32_C(184622.80), EASYSIMD_FLOAT32_C(  -898.98), EASYSIMD_FLOAT32_C(  -461.75),
                         EASYSIMD_FLOAT32_C(-639023.31), EASYSIMD_FLOAT32_C(  -171.93), EASYSIMD_FLOAT32_C(   135.84), EASYSIMD_FLOAT32_C(683050.44)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   722.50), EASYSIMD_FLOAT32_C(     1.66), EASYSIMD_FLOAT32_C(  -227.96), EASYSIMD_FLOAT32_C(  -417.42),
                         EASYSIMD_FLOAT32_C(   237.94), EASYSIMD_FLOAT32_C(   860.12), EASYSIMD_FLOAT32_C(  -692.46), EASYSIMD_FLOAT32_C(     6.00),
                         EASYSIMD_FLOAT32_C(  -979.01), EASYSIMD_FLOAT32_C(    20.38), EASYSIMD_FLOAT32_C(    85.42), EASYSIMD_FLOAT32_C(  -156.50),
                         EASYSIMD_FLOAT32_C(    23.29), EASYSIMD_FLOAT32_C(  -569.89), EASYSIMD_FLOAT32_C(    24.40), EASYSIMD_FLOAT32_C(   257.32)),
      UINT16_C(53230),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -804.04), EASYSIMD_FLOAT32_C(  -689.70), EASYSIMD_FLOAT32_C(   435.74), EASYSIMD_FLOAT32_C(    49.95),
                         EASYSIMD_FLOAT32_C(   554.19), EASYSIMD_FLOAT32_C(   936.14), EASYSIMD_FLOAT32_C(   554.63), EASYSIMD_FLOAT32_C(  -242.02),
                         EASYSIMD_FLOAT32_C(  -909.08), EASYSIMD_FLOAT32_C(  -184.42), EASYSIMD_FLOAT32_C(  -668.15), EASYSIMD_FLOAT32_C(   202.23),
                         EASYSIMD_FLOAT32_C(   620.00), EASYSIMD_FLOAT32_C(   -11.65), EASYSIMD_FLOAT32_C(  -295.73), EASYSIMD_FLOAT32_C(  -637.18)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -700.26), EASYSIMD_FLOAT32_C(   -48.09), EASYSIMD_FLOAT32_C(  -661.22), EASYSIMD_FLOAT32_C(  -536.85),
                         EASYSIMD_FLOAT32_C(  -172.68), EASYSIMD_FLOAT32_C(   263.32), EASYSIMD_FLOAT32_C(  -189.80), EASYSIMD_FLOAT32_C(  -595.66),
                         EASYSIMD_FLOAT32_C(   244.26), EASYSIMD_FLOAT32_C(  -637.08), EASYSIMD_FLOAT32_C(  -871.35), EASYSIMD_FLOAT32_C(  -417.36),
                         EASYSIMD_FLOAT32_C(  -313.14), EASYSIMD_FLOAT32_C(  -902.95), EASYSIMD_FLOAT32_C(  -801.13), EASYSIMD_FLOAT32_C(  -357.00)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(563037.06), EASYSIMD_FLOAT32_C( 33167.67), EASYSIMD_FLOAT32_C(  -227.96), EASYSIMD_FLOAT32_C(  -417.42),
                         EASYSIMD_FLOAT32_C(-95697.52), EASYSIMD_FLOAT32_C(246504.39), EASYSIMD_FLOAT32_C(-105268.77), EASYSIMD_FLOAT32_C(144161.62),
                         EASYSIMD_FLOAT32_C(-222051.88), EASYSIMD_FLOAT32_C(117490.30), EASYSIMD_FLOAT32_C(582192.50), EASYSIMD_FLOAT32_C(  -156.50),
                         EASYSIMD_FLOAT32_C(-194146.81), EASYSIMD_FLOAT32_C( 10519.37), EASYSIMD_FLOAT32_C(236918.19), EASYSIMD_FLOAT32_C(   257.32)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   113.45), EASYSIMD_FLOAT32_C(   -47.38), EASYSIMD_FLOAT32_C(   434.74), EASYSIMD_FLOAT32_C(   133.40),
                         EASYSIMD_FLOAT32_C(    37.09), EASYSIMD_FLOAT32_C(  -195.77), EASYSIMD_FLOAT32_C(  -878.67), EASYSIMD_FLOAT32_C(   758.19),
                         EASYSIMD_FLOAT32_C(   -87.72), EASYSIMD_FLOAT32_C(  -903.51), EASYSIMD_FLOAT32_C(  -821.22), EASYSIMD_FLOAT32_C(  -102.72),
                         EASYSIMD_FLOAT32_C(   329.70), EASYSIMD_FLOAT32_C(   752.97), EASYSIMD_FLOAT32_C(  -341.79), EASYSIMD_FLOAT32_C(  -130.85)),
      UINT16_C(62361),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -994.03), EASYSIMD_FLOAT32_C(  -716.03), EASYSIMD_FLOAT32_C(  -435.51), EASYSIMD_FLOAT32_C(  -960.04),
                         EASYSIMD_FLOAT32_C(    32.26), EASYSIMD_FLOAT32_C(  -474.76), EASYSIMD_FLOAT32_C(  -182.77), EASYSIMD_FLOAT32_C(  -229.72),
                         EASYSIMD_FLOAT32_C(  -949.63), EASYSIMD_FLOAT32_C(  -938.60), EASYSIMD_FLOAT32_C(  -855.41), EASYSIMD_FLOAT32_C(  -231.99),
                         EASYSIMD_FLOAT32_C(   115.21), EASYSIMD_FLOAT32_C(   716.21), EASYSIMD_FLOAT32_C(  -407.80), EASYSIMD_FLOAT32_C(   373.68)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   186.12), EASYSIMD_FLOAT32_C(  -224.01), EASYSIMD_FLOAT32_C(   199.06), EASYSIMD_FLOAT32_C(  -162.12),
                         EASYSIMD_FLOAT32_C(  -212.56), EASYSIMD_FLOAT32_C(  -205.93), EASYSIMD_FLOAT32_C(  -577.22), EASYSIMD_FLOAT32_C(  -567.47),
                         EASYSIMD_FLOAT32_C(  -916.44), EASYSIMD_FLOAT32_C(   780.43), EASYSIMD_FLOAT32_C(  -604.79), EASYSIMD_FLOAT32_C(   540.03),
                         EASYSIMD_FLOAT32_C(  -974.56), EASYSIMD_FLOAT32_C(  -517.05), EASYSIMD_FLOAT32_C(  -241.22), EASYSIMD_FLOAT32_C(   102.85)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-185008.86), EASYSIMD_FLOAT32_C(160397.89), EASYSIMD_FLOAT32_C(-86692.62), EASYSIMD_FLOAT32_C(155641.67),
                         EASYSIMD_FLOAT32_C(    37.09), EASYSIMD_FLOAT32_C(  -195.77), EASYSIMD_FLOAT32_C(105498.50), EASYSIMD_FLOAT32_C(130359.20),
                         EASYSIMD_FLOAT32_C(870278.94), EASYSIMD_FLOAT32_C(  -903.51), EASYSIMD_FLOAT32_C(  -821.22), EASYSIMD_FLOAT32_C(-125281.57),
                         EASYSIMD_FLOAT32_C(-112279.05), EASYSIMD_FLOAT32_C(   752.97), EASYSIMD_FLOAT32_C(  -341.79), EASYSIMD_FLOAT32_C( 38432.99)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mul_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mul_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mul_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { UINT16_C(47289),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -658.59), EASYSIMD_FLOAT32_C(  -110.05), EASYSIMD_FLOAT32_C(  -529.45), EASYSIMD_FLOAT32_C(    46.72),
                         EASYSIMD_FLOAT32_C(   -62.14), EASYSIMD_FLOAT32_C(   483.09), EASYSIMD_FLOAT32_C(   301.22), EASYSIMD_FLOAT32_C(  -113.80),
                         EASYSIMD_FLOAT32_C(  -597.24), EASYSIMD_FLOAT32_C(    55.35), EASYSIMD_FLOAT32_C(   938.56), EASYSIMD_FLOAT32_C(   -50.24),
                         EASYSIMD_FLOAT32_C(    49.65), EASYSIMD_FLOAT32_C(  -991.96), EASYSIMD_FLOAT32_C(   606.92), EASYSIMD_FLOAT32_C(   149.59)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   375.27), EASYSIMD_FLOAT32_C(  -498.85), EASYSIMD_FLOAT32_C(  -342.99), EASYSIMD_FLOAT32_C(   861.28),
                         EASYSIMD_FLOAT32_C(   526.60), EASYSIMD_FLOAT32_C(  -759.33), EASYSIMD_FLOAT32_C(   328.64), EASYSIMD_FLOAT32_C(   698.74),
                         EASYSIMD_FLOAT32_C(   615.23), EASYSIMD_FLOAT32_C(   873.23), EASYSIMD_FLOAT32_C(   127.27), EASYSIMD_FLOAT32_C(   719.43),
                         EASYSIMD_FLOAT32_C(  -625.99), EASYSIMD_FLOAT32_C(  -942.07), EASYSIMD_FLOAT32_C(   458.53), EASYSIMD_FLOAT32_C(   322.40)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-247149.08), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(181596.05), EASYSIMD_FLOAT32_C( 40239.00),
                         EASYSIMD_FLOAT32_C(-32722.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(-367439.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(119450.53), EASYSIMD_FLOAT32_C(-36144.16),
                         EASYSIMD_FLOAT32_C(-31080.40), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 48227.81)) },
    { UINT16_C(37892),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -265.18), EASYSIMD_FLOAT32_C(   394.19), EASYSIMD_FLOAT32_C(   565.39), EASYSIMD_FLOAT32_C(  -167.00),
                         EASYSIMD_FLOAT32_C(   350.77), EASYSIMD_FLOAT32_C(   863.35), EASYSIMD_FLOAT32_C(  -537.11), EASYSIMD_FLOAT32_C(  -601.68),
                         EASYSIMD_FLOAT32_C(  -980.35), EASYSIMD_FLOAT32_C(  -851.86), EASYSIMD_FLOAT32_C(  -959.52), EASYSIMD_FLOAT32_C(  -856.72),
                         EASYSIMD_FLOAT32_C(   393.09), EASYSIMD_FLOAT32_C(  -263.92), EASYSIMD_FLOAT32_C(   261.53), EASYSIMD_FLOAT32_C(   409.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -261.28), EASYSIMD_FLOAT32_C(  -762.17), EASYSIMD_FLOAT32_C(   850.55), EASYSIMD_FLOAT32_C(  -684.91),
                         EASYSIMD_FLOAT32_C(    69.61), EASYSIMD_FLOAT32_C(   771.73), EASYSIMD_FLOAT32_C(  -506.14), EASYSIMD_FLOAT32_C(  -578.92),
                         EASYSIMD_FLOAT32_C(   322.24), EASYSIMD_FLOAT32_C(   192.10), EASYSIMD_FLOAT32_C(  -768.24), EASYSIMD_FLOAT32_C(  -528.40),
                         EASYSIMD_FLOAT32_C(  -871.80), EASYSIMD_FLOAT32_C(   -55.77), EASYSIMD_FLOAT32_C(   401.18), EASYSIMD_FLOAT32_C(  -914.96)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C( 69286.23), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(114379.97),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(666273.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 14718.82), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(21270),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   124.49), EASYSIMD_FLOAT32_C(   701.35), EASYSIMD_FLOAT32_C(   498.79), EASYSIMD_FLOAT32_C(   832.83),
                         EASYSIMD_FLOAT32_C(  -974.32), EASYSIMD_FLOAT32_C(  -582.20), EASYSIMD_FLOAT32_C(  -288.73), EASYSIMD_FLOAT32_C(   146.91),
                         EASYSIMD_FLOAT32_C(   866.64), EASYSIMD_FLOAT32_C(   902.02), EASYSIMD_FLOAT32_C(   -35.40), EASYSIMD_FLOAT32_C(  -390.90),
                         EASYSIMD_FLOAT32_C(  -670.61), EASYSIMD_FLOAT32_C(  -294.26), EASYSIMD_FLOAT32_C(   904.08), EASYSIMD_FLOAT32_C(  -920.18)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   282.01), EASYSIMD_FLOAT32_C(   688.91), EASYSIMD_FLOAT32_C(   333.36), EASYSIMD_FLOAT32_C(   260.07),
                         EASYSIMD_FLOAT32_C(   576.07), EASYSIMD_FLOAT32_C(   133.85), EASYSIMD_FLOAT32_C(   534.76), EASYSIMD_FLOAT32_C(  -643.54),
                         EASYSIMD_FLOAT32_C(  -999.40), EASYSIMD_FLOAT32_C(   257.62), EASYSIMD_FLOAT32_C(   420.35), EASYSIMD_FLOAT32_C(  -394.28),
                         EASYSIMD_FLOAT32_C(   211.89), EASYSIMD_FLOAT32_C(   496.82), EASYSIMD_FLOAT32_C(  -993.25), EASYSIMD_FLOAT32_C(  -590.67)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(483167.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(216594.11),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-154401.27), EASYSIMD_FLOAT32_C(-94542.46),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(154124.05),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-146194.27), EASYSIMD_FLOAT32_C(-897977.50), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(31632),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    50.34), EASYSIMD_FLOAT32_C(   -97.56), EASYSIMD_FLOAT32_C(   263.08), EASYSIMD_FLOAT32_C(  -308.40),
                         EASYSIMD_FLOAT32_C(   354.47), EASYSIMD_FLOAT32_C(   -70.93), EASYSIMD_FLOAT32_C(   486.01), EASYSIMD_FLOAT32_C(  -938.29),
                         EASYSIMD_FLOAT32_C(   -47.71), EASYSIMD_FLOAT32_C(  -345.27), EASYSIMD_FLOAT32_C(    12.62), EASYSIMD_FLOAT32_C(   733.96),
                         EASYSIMD_FLOAT32_C(   753.32), EASYSIMD_FLOAT32_C(  -397.23), EASYSIMD_FLOAT32_C(   708.66), EASYSIMD_FLOAT32_C(   404.50)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   875.93), EASYSIMD_FLOAT32_C(  -911.86), EASYSIMD_FLOAT32_C(   784.71), EASYSIMD_FLOAT32_C(    64.13),
                         EASYSIMD_FLOAT32_C(  -731.87), EASYSIMD_FLOAT32_C(  -647.62), EASYSIMD_FLOAT32_C(   107.77), EASYSIMD_FLOAT32_C(   557.50),
                         EASYSIMD_FLOAT32_C(  -491.55), EASYSIMD_FLOAT32_C(   414.15), EASYSIMD_FLOAT32_C(  -504.43), EASYSIMD_FLOAT32_C(   -27.13),
                         EASYSIMD_FLOAT32_C(  -947.21), EASYSIMD_FLOAT32_C(  -164.39), EASYSIMD_FLOAT32_C(   287.82), EASYSIMD_FLOAT32_C(   414.18)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 88961.05), EASYSIMD_FLOAT32_C(206441.50), EASYSIMD_FLOAT32_C(-19777.69),
                         EASYSIMD_FLOAT32_C(-259425.95), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 52377.30), EASYSIMD_FLOAT32_C(-523096.66),
                         EASYSIMD_FLOAT32_C( 23451.85), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-19912.33),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(47299),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -544.18), EASYSIMD_FLOAT32_C(  -903.28), EASYSIMD_FLOAT32_C(   559.95), EASYSIMD_FLOAT32_C(  -483.39),
                         EASYSIMD_FLOAT32_C(  -994.67), EASYSIMD_FLOAT32_C(  -750.48), EASYSIMD_FLOAT32_C(   312.50), EASYSIMD_FLOAT32_C(   110.85),
                         EASYSIMD_FLOAT32_C(  -430.65), EASYSIMD_FLOAT32_C(    39.80), EASYSIMD_FLOAT32_C(   -26.24), EASYSIMD_FLOAT32_C(   378.89),
                         EASYSIMD_FLOAT32_C(  -139.95), EASYSIMD_FLOAT32_C(  -775.11), EASYSIMD_FLOAT32_C(  -758.69), EASYSIMD_FLOAT32_C(   318.51)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   548.46), EASYSIMD_FLOAT32_C(   682.86), EASYSIMD_FLOAT32_C(  -635.50), EASYSIMD_FLOAT32_C(   737.24),
                         EASYSIMD_FLOAT32_C(   707.78), EASYSIMD_FLOAT32_C(  -907.72), EASYSIMD_FLOAT32_C(  -791.08), EASYSIMD_FLOAT32_C(   176.45),
                         EASYSIMD_FLOAT32_C(    64.55), EASYSIMD_FLOAT32_C(    55.56), EASYSIMD_FLOAT32_C(  -108.86), EASYSIMD_FLOAT32_C(   505.77),
                         EASYSIMD_FLOAT32_C(   224.25), EASYSIMD_FLOAT32_C(   639.22), EASYSIMD_FLOAT32_C(   369.92), EASYSIMD_FLOAT32_C(  -708.31)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-298460.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-355848.22), EASYSIMD_FLOAT32_C(-356374.44),
                         EASYSIMD_FLOAT32_C(-704007.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(-27798.46), EASYSIMD_FLOAT32_C(  2211.29), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-280654.62), EASYSIMD_FLOAT32_C(-225603.83)) },
    { UINT16_C(40773),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -86.71), EASYSIMD_FLOAT32_C(  -432.63), EASYSIMD_FLOAT32_C(  -433.37), EASYSIMD_FLOAT32_C(  -539.66),
                         EASYSIMD_FLOAT32_C(   559.54), EASYSIMD_FLOAT32_C(  -287.88), EASYSIMD_FLOAT32_C(  -991.42), EASYSIMD_FLOAT32_C(  -690.07),
                         EASYSIMD_FLOAT32_C(   345.70), EASYSIMD_FLOAT32_C(   616.00), EASYSIMD_FLOAT32_C(   341.79), EASYSIMD_FLOAT32_C(  -307.10),
                         EASYSIMD_FLOAT32_C(   709.24), EASYSIMD_FLOAT32_C(  -920.15), EASYSIMD_FLOAT32_C(   404.20), EASYSIMD_FLOAT32_C(    52.51)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   815.79), EASYSIMD_FLOAT32_C(  -788.62), EASYSIMD_FLOAT32_C(  -370.04), EASYSIMD_FLOAT32_C(  -939.88),
                         EASYSIMD_FLOAT32_C(  -591.16), EASYSIMD_FLOAT32_C(  -819.13), EASYSIMD_FLOAT32_C(   932.20), EASYSIMD_FLOAT32_C(  -243.11),
                         EASYSIMD_FLOAT32_C(   -87.62), EASYSIMD_FLOAT32_C(   725.93), EASYSIMD_FLOAT32_C(  -374.67), EASYSIMD_FLOAT32_C(   301.09),
                         EASYSIMD_FLOAT32_C(  -174.47), EASYSIMD_FLOAT32_C(  -898.14), EASYSIMD_FLOAT32_C(  -924.02), EASYSIMD_FLOAT32_C(  -333.66)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(-70737.15), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(507215.62),
                         EASYSIMD_FLOAT32_C(-330777.62), EASYSIMD_FLOAT32_C(235811.16), EASYSIMD_FLOAT32_C(-924201.75), EASYSIMD_FLOAT32_C(167762.92),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(447172.88), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(826423.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-17520.49)) },
    { UINT16_C(61172),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -690.89), EASYSIMD_FLOAT32_C(  -270.55), EASYSIMD_FLOAT32_C(   476.48), EASYSIMD_FLOAT32_C(   219.51),
                         EASYSIMD_FLOAT32_C(   642.18), EASYSIMD_FLOAT32_C(  -588.29), EASYSIMD_FLOAT32_C(  -762.74), EASYSIMD_FLOAT32_C(   -33.80),
                         EASYSIMD_FLOAT32_C(  -786.44), EASYSIMD_FLOAT32_C(  -855.21), EASYSIMD_FLOAT32_C(   145.12), EASYSIMD_FLOAT32_C(    50.96),
                         EASYSIMD_FLOAT32_C(   710.85), EASYSIMD_FLOAT32_C(   234.05), EASYSIMD_FLOAT32_C(   345.96), EASYSIMD_FLOAT32_C(   118.24)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -143.72), EASYSIMD_FLOAT32_C(  -461.06), EASYSIMD_FLOAT32_C(   866.17), EASYSIMD_FLOAT32_C(  -706.49),
                         EASYSIMD_FLOAT32_C(   594.76), EASYSIMD_FLOAT32_C(   424.92), EASYSIMD_FLOAT32_C(   166.20), EASYSIMD_FLOAT32_C(   776.85),
                         EASYSIMD_FLOAT32_C(  -191.32), EASYSIMD_FLOAT32_C(  -329.15), EASYSIMD_FLOAT32_C(  -651.62), EASYSIMD_FLOAT32_C(   -22.33),
                         EASYSIMD_FLOAT32_C(  -429.53), EASYSIMD_FLOAT32_C(   758.36), EASYSIMD_FLOAT32_C(   926.10), EASYSIMD_FLOAT32_C(    17.27)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C( 99294.71), EASYSIMD_FLOAT32_C(124739.77), EASYSIMD_FLOAT32_C(412712.69), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(381942.97), EASYSIMD_FLOAT32_C(-249976.19), EASYSIMD_FLOAT32_C(-126767.38), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(150461.70), EASYSIMD_FLOAT32_C(281492.38), EASYSIMD_FLOAT32_C(-94563.09), EASYSIMD_FLOAT32_C( -1137.94),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(177494.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(31704),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   761.96), EASYSIMD_FLOAT32_C(   858.35), EASYSIMD_FLOAT32_C(   360.42), EASYSIMD_FLOAT32_C(   321.87),
                         EASYSIMD_FLOAT32_C(  -444.03), EASYSIMD_FLOAT32_C(  -177.67), EASYSIMD_FLOAT32_C(  -802.25), EASYSIMD_FLOAT32_C(   408.95),
                         EASYSIMD_FLOAT32_C(  -337.63), EASYSIMD_FLOAT32_C(   948.17), EASYSIMD_FLOAT32_C(   248.80), EASYSIMD_FLOAT32_C(   170.02),
                         EASYSIMD_FLOAT32_C(   939.41), EASYSIMD_FLOAT32_C(  -580.14), EASYSIMD_FLOAT32_C(   237.93), EASYSIMD_FLOAT32_C(  -698.11)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   786.48), EASYSIMD_FLOAT32_C(  -475.23), EASYSIMD_FLOAT32_C(   258.84), EASYSIMD_FLOAT32_C(   824.39),
                         EASYSIMD_FLOAT32_C(  -282.56), EASYSIMD_FLOAT32_C(   765.76), EASYSIMD_FLOAT32_C(  -709.23), EASYSIMD_FLOAT32_C(   701.93),
                         EASYSIMD_FLOAT32_C(  -367.75), EASYSIMD_FLOAT32_C(   404.32), EASYSIMD_FLOAT32_C(  -447.00), EASYSIMD_FLOAT32_C(   864.94),
                         EASYSIMD_FLOAT32_C(   954.31), EASYSIMD_FLOAT32_C(   410.35), EASYSIMD_FLOAT32_C(  -565.19), EASYSIMD_FLOAT32_C(  -545.67)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-407913.66), EASYSIMD_FLOAT32_C( 93291.12), EASYSIMD_FLOAT32_C(265346.41),
                         EASYSIMD_FLOAT32_C(125465.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(568979.75), EASYSIMD_FLOAT32_C(287054.28),
                         EASYSIMD_FLOAT32_C(124163.44), EASYSIMD_FLOAT32_C(383364.09), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(147057.11),
                         EASYSIMD_FLOAT32_C(896488.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mul_ps(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mul_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mul_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -736.65), EASYSIMD_FLOAT64_C( -764.30),
                         EASYSIMD_FLOAT64_C(  675.25), EASYSIMD_FLOAT64_C( -182.15),
                         EASYSIMD_FLOAT64_C( -748.44), EASYSIMD_FLOAT64_C(   82.10),
                         EASYSIMD_FLOAT64_C(  684.52), EASYSIMD_FLOAT64_C( -343.09)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -267.96), EASYSIMD_FLOAT64_C( -536.57),
                         EASYSIMD_FLOAT64_C( -934.00), EASYSIMD_FLOAT64_C(  653.62),
                         EASYSIMD_FLOAT64_C(  984.11), EASYSIMD_FLOAT64_C(  140.30),
                         EASYSIMD_FLOAT64_C( -580.05), EASYSIMD_FLOAT64_C( -915.75)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(197392.73), EASYSIMD_FLOAT64_C(410100.45),
                         EASYSIMD_FLOAT64_C(-630683.50), EASYSIMD_FLOAT64_C(-119056.88),
                         EASYSIMD_FLOAT64_C(-736547.29), EASYSIMD_FLOAT64_C(11518.63),
                         EASYSIMD_FLOAT64_C(-397055.83), EASYSIMD_FLOAT64_C(314184.67)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -775.40), EASYSIMD_FLOAT64_C( -210.92),
                         EASYSIMD_FLOAT64_C(  987.42), EASYSIMD_FLOAT64_C(  542.45),
                         EASYSIMD_FLOAT64_C( -745.60), EASYSIMD_FLOAT64_C(  -50.38),
                         EASYSIMD_FLOAT64_C(  163.82), EASYSIMD_FLOAT64_C( -164.62)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  263.91), EASYSIMD_FLOAT64_C( -350.89),
                         EASYSIMD_FLOAT64_C( -318.01), EASYSIMD_FLOAT64_C( -980.00),
                         EASYSIMD_FLOAT64_C(  872.18), EASYSIMD_FLOAT64_C(   80.96),
                         EASYSIMD_FLOAT64_C(  145.89), EASYSIMD_FLOAT64_C(  832.89)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-204635.81), EASYSIMD_FLOAT64_C(74009.72),
                         EASYSIMD_FLOAT64_C(-314009.43), EASYSIMD_FLOAT64_C(-531601.00),
                         EASYSIMD_FLOAT64_C(-650297.41), EASYSIMD_FLOAT64_C(-4078.76),
                         EASYSIMD_FLOAT64_C(23899.70), EASYSIMD_FLOAT64_C(-137110.35)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  606.76), EASYSIMD_FLOAT64_C( -664.92),
                         EASYSIMD_FLOAT64_C(  454.81), EASYSIMD_FLOAT64_C(  299.40),
                         EASYSIMD_FLOAT64_C( -524.63), EASYSIMD_FLOAT64_C(   40.68),
                         EASYSIMD_FLOAT64_C(  218.77), EASYSIMD_FLOAT64_C(   35.82)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -757.25), EASYSIMD_FLOAT64_C(  594.07),
                         EASYSIMD_FLOAT64_C(  304.96), EASYSIMD_FLOAT64_C( -155.47),
                         EASYSIMD_FLOAT64_C(  635.03), EASYSIMD_FLOAT64_C(  654.85),
                         EASYSIMD_FLOAT64_C(  777.61), EASYSIMD_FLOAT64_C( -598.19)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-459469.01), EASYSIMD_FLOAT64_C(-395009.02),
                         EASYSIMD_FLOAT64_C(138698.86), EASYSIMD_FLOAT64_C(-46547.72),
                         EASYSIMD_FLOAT64_C(-333155.79), EASYSIMD_FLOAT64_C(26639.30),
                         EASYSIMD_FLOAT64_C(170117.74), EASYSIMD_FLOAT64_C(-21427.17)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -910.74), EASYSIMD_FLOAT64_C( -302.10),
                         EASYSIMD_FLOAT64_C(  937.08), EASYSIMD_FLOAT64_C(  618.13),
                         EASYSIMD_FLOAT64_C(   85.12), EASYSIMD_FLOAT64_C(    3.50),
                         EASYSIMD_FLOAT64_C( -122.84), EASYSIMD_FLOAT64_C(  290.22)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  392.21), EASYSIMD_FLOAT64_C(  139.00),
                         EASYSIMD_FLOAT64_C( -878.97), EASYSIMD_FLOAT64_C(  778.57),
                         EASYSIMD_FLOAT64_C( -810.83), EASYSIMD_FLOAT64_C(  413.49),
                         EASYSIMD_FLOAT64_C(  505.44), EASYSIMD_FLOAT64_C(  291.58)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-357201.34), EASYSIMD_FLOAT64_C(-41991.90),
                         EASYSIMD_FLOAT64_C(-823665.21), EASYSIMD_FLOAT64_C(481257.47),
                         EASYSIMD_FLOAT64_C(-69017.85), EASYSIMD_FLOAT64_C( 1447.22),
                         EASYSIMD_FLOAT64_C(-62088.25), EASYSIMD_FLOAT64_C(84622.35)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  232.02), EASYSIMD_FLOAT64_C(  984.70),
                         EASYSIMD_FLOAT64_C( -800.83), EASYSIMD_FLOAT64_C( -826.63),
                         EASYSIMD_FLOAT64_C(  822.26), EASYSIMD_FLOAT64_C( -892.21),
                         EASYSIMD_FLOAT64_C( -651.70), EASYSIMD_FLOAT64_C( -380.50)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  880.83), EASYSIMD_FLOAT64_C( -763.06),
                         EASYSIMD_FLOAT64_C( -540.57), EASYSIMD_FLOAT64_C( -512.55),
                         EASYSIMD_FLOAT64_C(  -32.98), EASYSIMD_FLOAT64_C(  700.87),
                         EASYSIMD_FLOAT64_C( -425.19), EASYSIMD_FLOAT64_C( -849.48)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(204370.18), EASYSIMD_FLOAT64_C(-751385.18),
                         EASYSIMD_FLOAT64_C(432904.67), EASYSIMD_FLOAT64_C(423689.21),
                         EASYSIMD_FLOAT64_C(-27118.13), EASYSIMD_FLOAT64_C(-625323.22),
                         EASYSIMD_FLOAT64_C(277096.32), EASYSIMD_FLOAT64_C(323227.14)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  202.90), EASYSIMD_FLOAT64_C( -396.66),
                         EASYSIMD_FLOAT64_C( -364.01), EASYSIMD_FLOAT64_C(   56.81),
                         EASYSIMD_FLOAT64_C( -881.59), EASYSIMD_FLOAT64_C(  212.81),
                         EASYSIMD_FLOAT64_C( -968.64), EASYSIMD_FLOAT64_C( -657.19)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -221.35), EASYSIMD_FLOAT64_C( -305.38),
                         EASYSIMD_FLOAT64_C(  546.45), EASYSIMD_FLOAT64_C( -697.03),
                         EASYSIMD_FLOAT64_C(   93.97), EASYSIMD_FLOAT64_C(  975.92),
                         EASYSIMD_FLOAT64_C(  876.47), EASYSIMD_FLOAT64_C(  762.37)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-44911.92), EASYSIMD_FLOAT64_C(121132.03),
                         EASYSIMD_FLOAT64_C(-198913.26), EASYSIMD_FLOAT64_C(-39598.27),
                         EASYSIMD_FLOAT64_C(-82843.01), EASYSIMD_FLOAT64_C(207685.54),
                         EASYSIMD_FLOAT64_C(-848983.90), EASYSIMD_FLOAT64_C(-501021.94)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -538.86), EASYSIMD_FLOAT64_C(   39.65),
                         EASYSIMD_FLOAT64_C( -229.28), EASYSIMD_FLOAT64_C( -842.78),
                         EASYSIMD_FLOAT64_C(  -14.75), EASYSIMD_FLOAT64_C( -859.98),
                         EASYSIMD_FLOAT64_C(  215.44), EASYSIMD_FLOAT64_C(  762.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -16.18), EASYSIMD_FLOAT64_C( -931.82),
                         EASYSIMD_FLOAT64_C( -687.15), EASYSIMD_FLOAT64_C( -416.23),
                         EASYSIMD_FLOAT64_C( -313.36), EASYSIMD_FLOAT64_C(  905.90),
                         EASYSIMD_FLOAT64_C(    1.93), EASYSIMD_FLOAT64_C( -464.98)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( 8718.75), EASYSIMD_FLOAT64_C(-36946.66),
                         EASYSIMD_FLOAT64_C(157549.75), EASYSIMD_FLOAT64_C(350790.32),
                         EASYSIMD_FLOAT64_C( 4622.06), EASYSIMD_FLOAT64_C(-779055.88),
                         EASYSIMD_FLOAT64_C(  415.80), EASYSIMD_FLOAT64_C(-354700.69)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -25.40), EASYSIMD_FLOAT64_C( -267.80),
                         EASYSIMD_FLOAT64_C(  353.79), EASYSIMD_FLOAT64_C(  -35.72),
                         EASYSIMD_FLOAT64_C(  125.21), EASYSIMD_FLOAT64_C(  137.22),
                         EASYSIMD_FLOAT64_C(  310.88), EASYSIMD_FLOAT64_C( -724.55)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -551.49), EASYSIMD_FLOAT64_C(  -42.33),
                         EASYSIMD_FLOAT64_C( -926.18), EASYSIMD_FLOAT64_C(   36.96),
                         EASYSIMD_FLOAT64_C(  954.39), EASYSIMD_FLOAT64_C(  874.71),
                         EASYSIMD_FLOAT64_C( -375.00), EASYSIMD_FLOAT64_C(  949.07)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(14007.85), EASYSIMD_FLOAT64_C(11335.97),
                         EASYSIMD_FLOAT64_C(-327673.22), EASYSIMD_FLOAT64_C(-1320.21),
                         EASYSIMD_FLOAT64_C(119499.17), EASYSIMD_FLOAT64_C(120027.71),
                         EASYSIMD_FLOAT64_C(-116580.00), EASYSIMD_FLOAT64_C(-687648.67)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mul_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mul_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mul_round_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  static const struct {
    easysimd_float64 a[8];
    easysimd_float64 b[8];
    easysimd_float64 nearest_inf[8];
    easysimd_float64 neg_inf[8];
    easysimd_float64 pos_inf[8];
    easysimd_float64 zero[8];
    easysimd_float64 direction[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -380.22), EASYSIMD_FLOAT64_C(   891.11), EASYSIMD_FLOAT64_C(  -189.08), EASYSIMD_FLOAT64_C(   282.16),
        EASYSIMD_FLOAT64_C(  -194.81), EASYSIMD_FLOAT64_C(   509.21), EASYSIMD_FLOAT64_C(   871.84), EASYSIMD_FLOAT64_C(  -385.11) },
      { EASYSIMD_FLOAT64_C(   987.85), EASYSIMD_FLOAT64_C(   314.37), EASYSIMD_FLOAT64_C(   863.84), EASYSIMD_FLOAT64_C(   647.29),
        EASYSIMD_FLOAT64_C(   501.67), EASYSIMD_FLOAT64_C(  -533.39), EASYSIMD_FLOAT64_C(   236.02), EASYSIMD_FLOAT64_C(   932.85) },
      { EASYSIMD_FLOAT64_C(-375600.00), EASYSIMD_FLOAT64_C(280138.00), EASYSIMD_FLOAT64_C(-163335.00), EASYSIMD_FLOAT64_C(182639.00),
        EASYSIMD_FLOAT64_C(-97730.00), EASYSIMD_FLOAT64_C(-271608.00), EASYSIMD_FLOAT64_C(205772.00), EASYSIMD_FLOAT64_C(-359250.00) },
      { EASYSIMD_FLOAT64_C(-375601.00), EASYSIMD_FLOAT64_C(280138.00), EASYSIMD_FLOAT64_C(-163335.00), EASYSIMD_FLOAT64_C(182639.00),
        EASYSIMD_FLOAT64_C(-97731.00), EASYSIMD_FLOAT64_C(-271608.00), EASYSIMD_FLOAT64_C(205771.00), EASYSIMD_FLOAT64_C(-359250.00) },
      { EASYSIMD_FLOAT64_C(-375600.00), EASYSIMD_FLOAT64_C(280139.00), EASYSIMD_FLOAT64_C(-163334.00), EASYSIMD_FLOAT64_C(182640.00),
        EASYSIMD_FLOAT64_C(-97730.00), EASYSIMD_FLOAT64_C(-271607.00), EASYSIMD_FLOAT64_C(205772.00), EASYSIMD_FLOAT64_C(-359249.00) },
      { EASYSIMD_FLOAT64_C(-375600.00), EASYSIMD_FLOAT64_C(280138.00), EASYSIMD_FLOAT64_C(-163334.00), EASYSIMD_FLOAT64_C(182639.00),
        EASYSIMD_FLOAT64_C(-97730.00), EASYSIMD_FLOAT64_C(-271607.00), EASYSIMD_FLOAT64_C(205771.00), EASYSIMD_FLOAT64_C(-359249.00) },
      { EASYSIMD_FLOAT64_C(-375600.00), EASYSIMD_FLOAT64_C(280138.00), EASYSIMD_FLOAT64_C(-163335.00), EASYSIMD_FLOAT64_C(182639.00),
        EASYSIMD_FLOAT64_C(-97730.00), EASYSIMD_FLOAT64_C(-271608.00), EASYSIMD_FLOAT64_C(205772.00), EASYSIMD_FLOAT64_C(-359250.00) } },
    { { EASYSIMD_FLOAT64_C(  -855.81), EASYSIMD_FLOAT64_C(  -837.47), EASYSIMD_FLOAT64_C(  -315.14), EASYSIMD_FLOAT64_C(   285.42),
        EASYSIMD_FLOAT64_C(   471.46), EASYSIMD_FLOAT64_C(   870.54), EASYSIMD_FLOAT64_C(   955.45), EASYSIMD_FLOAT64_C(   134.55) },
      { EASYSIMD_FLOAT64_C(  -894.72), EASYSIMD_FLOAT64_C(    90.22), EASYSIMD_FLOAT64_C(  -872.33), EASYSIMD_FLOAT64_C(   974.46),
        EASYSIMD_FLOAT64_C(  -867.17), EASYSIMD_FLOAT64_C(  -450.36), EASYSIMD_FLOAT64_C(   182.29), EASYSIMD_FLOAT64_C(  -247.39) },
      { EASYSIMD_FLOAT64_C(765710.00), EASYSIMD_FLOAT64_C(-75557.00), EASYSIMD_FLOAT64_C(274906.00), EASYSIMD_FLOAT64_C(278130.00),
        EASYSIMD_FLOAT64_C(-408836.00), EASYSIMD_FLOAT64_C(-392056.00), EASYSIMD_FLOAT64_C(174169.00), EASYSIMD_FLOAT64_C(-33286.00) },
      { EASYSIMD_FLOAT64_C(765710.00), EASYSIMD_FLOAT64_C(-75557.00), EASYSIMD_FLOAT64_C(274906.00), EASYSIMD_FLOAT64_C(278130.00),
        EASYSIMD_FLOAT64_C(-408836.00), EASYSIMD_FLOAT64_C(-392057.00), EASYSIMD_FLOAT64_C(174168.00), EASYSIMD_FLOAT64_C(-33287.00) },
      { EASYSIMD_FLOAT64_C(765711.00), EASYSIMD_FLOAT64_C(-75556.00), EASYSIMD_FLOAT64_C(274907.00), EASYSIMD_FLOAT64_C(278131.00),
        EASYSIMD_FLOAT64_C(-408835.00), EASYSIMD_FLOAT64_C(-392056.00), EASYSIMD_FLOAT64_C(174169.00), EASYSIMD_FLOAT64_C(-33286.00) },
      { EASYSIMD_FLOAT64_C(765710.00), EASYSIMD_FLOAT64_C(-75556.00), EASYSIMD_FLOAT64_C(274906.00), EASYSIMD_FLOAT64_C(278130.00),
        EASYSIMD_FLOAT64_C(-408835.00), EASYSIMD_FLOAT64_C(-392056.00), EASYSIMD_FLOAT64_C(174168.00), EASYSIMD_FLOAT64_C(-33286.00) },
      { EASYSIMD_FLOAT64_C(765710.00), EASYSIMD_FLOAT64_C(-75557.00), EASYSIMD_FLOAT64_C(274906.00), EASYSIMD_FLOAT64_C(278130.00),
        EASYSIMD_FLOAT64_C(-408836.00), EASYSIMD_FLOAT64_C(-392056.00), EASYSIMD_FLOAT64_C(174169.00), EASYSIMD_FLOAT64_C(-33286.00) } },
    { { EASYSIMD_FLOAT64_C(  -559.25), EASYSIMD_FLOAT64_C(   993.21), EASYSIMD_FLOAT64_C(  -965.23), EASYSIMD_FLOAT64_C(   245.94),
        EASYSIMD_FLOAT64_C(   502.41), EASYSIMD_FLOAT64_C(   906.61), EASYSIMD_FLOAT64_C(   860.83), EASYSIMD_FLOAT64_C(   490.26) },
      { EASYSIMD_FLOAT64_C(   220.98), EASYSIMD_FLOAT64_C(   724.67), EASYSIMD_FLOAT64_C(   137.54), EASYSIMD_FLOAT64_C(  -277.35),
        EASYSIMD_FLOAT64_C(  -808.72), EASYSIMD_FLOAT64_C(  -626.43), EASYSIMD_FLOAT64_C(  -344.50), EASYSIMD_FLOAT64_C(  -664.53) },
      { EASYSIMD_FLOAT64_C(-123583.00), EASYSIMD_FLOAT64_C(719749.00), EASYSIMD_FLOAT64_C(-132758.00), EASYSIMD_FLOAT64_C(-68211.00),
        EASYSIMD_FLOAT64_C(-406309.00), EASYSIMD_FLOAT64_C(-567928.00), EASYSIMD_FLOAT64_C(-296556.00), EASYSIMD_FLOAT64_C(-325792.00) },
      { EASYSIMD_FLOAT64_C(-123584.00), EASYSIMD_FLOAT64_C(719749.00), EASYSIMD_FLOAT64_C(-132758.00), EASYSIMD_FLOAT64_C(-68212.00),
        EASYSIMD_FLOAT64_C(-406310.00), EASYSIMD_FLOAT64_C(-567928.00), EASYSIMD_FLOAT64_C(-296556.00), EASYSIMD_FLOAT64_C(-325793.00) },
      { EASYSIMD_FLOAT64_C(-123583.00), EASYSIMD_FLOAT64_C(719750.00), EASYSIMD_FLOAT64_C(-132757.00), EASYSIMD_FLOAT64_C(-68211.00),
        EASYSIMD_FLOAT64_C(-406309.00), EASYSIMD_FLOAT64_C(-567927.00), EASYSIMD_FLOAT64_C(-296555.00), EASYSIMD_FLOAT64_C(-325792.00) },
      { EASYSIMD_FLOAT64_C(-123583.00), EASYSIMD_FLOAT64_C(719749.00), EASYSIMD_FLOAT64_C(-132757.00), EASYSIMD_FLOAT64_C(-68211.00),
        EASYSIMD_FLOAT64_C(-406309.00), EASYSIMD_FLOAT64_C(-567927.00), EASYSIMD_FLOAT64_C(-296555.00), EASYSIMD_FLOAT64_C(-325792.00) },
      { EASYSIMD_FLOAT64_C(-123583.00), EASYSIMD_FLOAT64_C(719749.00), EASYSIMD_FLOAT64_C(-132758.00), EASYSIMD_FLOAT64_C(-68211.00),
        EASYSIMD_FLOAT64_C(-406309.00), EASYSIMD_FLOAT64_C(-567928.00), EASYSIMD_FLOAT64_C(-296556.00), EASYSIMD_FLOAT64_C(-325792.00) } },
    { { EASYSIMD_FLOAT64_C(  -463.90), EASYSIMD_FLOAT64_C(   340.36), EASYSIMD_FLOAT64_C(   620.88), EASYSIMD_FLOAT64_C(  -992.43),
        EASYSIMD_FLOAT64_C(   210.89), EASYSIMD_FLOAT64_C(   576.33), EASYSIMD_FLOAT64_C(   142.11), EASYSIMD_FLOAT64_C(   316.17) },
      { EASYSIMD_FLOAT64_C(  -333.45), EASYSIMD_FLOAT64_C(   269.79), EASYSIMD_FLOAT64_C(   290.64), EASYSIMD_FLOAT64_C(  -200.62),
        EASYSIMD_FLOAT64_C(   819.42), EASYSIMD_FLOAT64_C(  -527.08), EASYSIMD_FLOAT64_C(   551.98), EASYSIMD_FLOAT64_C(  -739.82) },
      { EASYSIMD_FLOAT64_C(154687.00), EASYSIMD_FLOAT64_C( 91826.00), EASYSIMD_FLOAT64_C(180453.00), EASYSIMD_FLOAT64_C(199101.00),
        EASYSIMD_FLOAT64_C(172807.00), EASYSIMD_FLOAT64_C(-303772.00), EASYSIMD_FLOAT64_C( 78442.00), EASYSIMD_FLOAT64_C(-233909.00) },
      { EASYSIMD_FLOAT64_C(154687.00), EASYSIMD_FLOAT64_C( 91825.00), EASYSIMD_FLOAT64_C(180452.00), EASYSIMD_FLOAT64_C(199101.00),
        EASYSIMD_FLOAT64_C(172807.00), EASYSIMD_FLOAT64_C(-303773.00), EASYSIMD_FLOAT64_C( 78441.00), EASYSIMD_FLOAT64_C(-233909.00) },
      { EASYSIMD_FLOAT64_C(154688.00), EASYSIMD_FLOAT64_C( 91826.00), EASYSIMD_FLOAT64_C(180453.00), EASYSIMD_FLOAT64_C(199102.00),
        EASYSIMD_FLOAT64_C(172808.00), EASYSIMD_FLOAT64_C(-303772.00), EASYSIMD_FLOAT64_C( 78442.00), EASYSIMD_FLOAT64_C(-233908.00) },
      { EASYSIMD_FLOAT64_C(154687.00), EASYSIMD_FLOAT64_C( 91825.00), EASYSIMD_FLOAT64_C(180452.00), EASYSIMD_FLOAT64_C(199101.00),
        EASYSIMD_FLOAT64_C(172807.00), EASYSIMD_FLOAT64_C(-303772.00), EASYSIMD_FLOAT64_C( 78441.00), EASYSIMD_FLOAT64_C(-233908.00) },
      { EASYSIMD_FLOAT64_C(154687.00), EASYSIMD_FLOAT64_C( 91826.00), EASYSIMD_FLOAT64_C(180453.00), EASYSIMD_FLOAT64_C(199101.00),
        EASYSIMD_FLOAT64_C(172807.00), EASYSIMD_FLOAT64_C(-303772.00), EASYSIMD_FLOAT64_C( 78442.00), EASYSIMD_FLOAT64_C(-233909.00) } },
    { { EASYSIMD_FLOAT64_C(  -533.87), EASYSIMD_FLOAT64_C(   586.76), EASYSIMD_FLOAT64_C(   506.12), EASYSIMD_FLOAT64_C(   968.54),
        EASYSIMD_FLOAT64_C(   493.37), EASYSIMD_FLOAT64_C(   366.94), EASYSIMD_FLOAT64_C(   458.80), EASYSIMD_FLOAT64_C(  -285.65) },
      { EASYSIMD_FLOAT64_C(    91.61), EASYSIMD_FLOAT64_C(  -403.66), EASYSIMD_FLOAT64_C(   437.00), EASYSIMD_FLOAT64_C(   282.89),
        EASYSIMD_FLOAT64_C(   -30.09), EASYSIMD_FLOAT64_C(  -907.50), EASYSIMD_FLOAT64_C(   618.36), EASYSIMD_FLOAT64_C(   506.02) },
      { EASYSIMD_FLOAT64_C(-48908.00), EASYSIMD_FLOAT64_C(-236852.00), EASYSIMD_FLOAT64_C(221174.00), EASYSIMD_FLOAT64_C(273990.00),
        EASYSIMD_FLOAT64_C(-14846.00), EASYSIMD_FLOAT64_C(-332998.00), EASYSIMD_FLOAT64_C(283704.00), EASYSIMD_FLOAT64_C(-144545.00) },
      { EASYSIMD_FLOAT64_C(-48908.00), EASYSIMD_FLOAT64_C(-236852.00), EASYSIMD_FLOAT64_C(221174.00), EASYSIMD_FLOAT64_C(273990.00),
        EASYSIMD_FLOAT64_C(-14846.00), EASYSIMD_FLOAT64_C(-332999.00), EASYSIMD_FLOAT64_C(283703.00), EASYSIMD_FLOAT64_C(-144545.00) },
      { EASYSIMD_FLOAT64_C(-48907.00), EASYSIMD_FLOAT64_C(-236851.00), EASYSIMD_FLOAT64_C(221175.00), EASYSIMD_FLOAT64_C(273991.00),
        EASYSIMD_FLOAT64_C(-14845.00), EASYSIMD_FLOAT64_C(-332998.00), EASYSIMD_FLOAT64_C(283704.00), EASYSIMD_FLOAT64_C(-144544.00) },
      { EASYSIMD_FLOAT64_C(-48907.00), EASYSIMD_FLOAT64_C(-236851.00), EASYSIMD_FLOAT64_C(221174.00), EASYSIMD_FLOAT64_C(273990.00),
        EASYSIMD_FLOAT64_C(-14845.00), EASYSIMD_FLOAT64_C(-332998.00), EASYSIMD_FLOAT64_C(283703.00), EASYSIMD_FLOAT64_C(-144544.00) },
      { EASYSIMD_FLOAT64_C(-48908.00), EASYSIMD_FLOAT64_C(-236852.00), EASYSIMD_FLOAT64_C(221174.00), EASYSIMD_FLOAT64_C(273990.00),
        EASYSIMD_FLOAT64_C(-14846.00), EASYSIMD_FLOAT64_C(-332998.00), EASYSIMD_FLOAT64_C(283704.00), EASYSIMD_FLOAT64_C(-144545.00) } },
    { { EASYSIMD_FLOAT64_C(   432.85), EASYSIMD_FLOAT64_C(   239.24), EASYSIMD_FLOAT64_C(   513.58), EASYSIMD_FLOAT64_C(  -356.25),
        EASYSIMD_FLOAT64_C(  -184.43), EASYSIMD_FLOAT64_C(  -344.30), EASYSIMD_FLOAT64_C(   959.92), EASYSIMD_FLOAT64_C(   482.12) },
      { EASYSIMD_FLOAT64_C(   925.48), EASYSIMD_FLOAT64_C(   250.55), EASYSIMD_FLOAT64_C(  -718.51), EASYSIMD_FLOAT64_C(   744.91),
        EASYSIMD_FLOAT64_C(   723.48), EASYSIMD_FLOAT64_C(   833.48), EASYSIMD_FLOAT64_C(  -994.92), EASYSIMD_FLOAT64_C(  -810.39) },
      { EASYSIMD_FLOAT64_C(400594.00), EASYSIMD_FLOAT64_C( 59942.00), EASYSIMD_FLOAT64_C(-369012.00), EASYSIMD_FLOAT64_C(-265374.00),
        EASYSIMD_FLOAT64_C(-133431.00), EASYSIMD_FLOAT64_C(-286967.00), EASYSIMD_FLOAT64_C(-955044.00), EASYSIMD_FLOAT64_C(-390705.00) },
      { EASYSIMD_FLOAT64_C(400594.00), EASYSIMD_FLOAT64_C( 59941.00), EASYSIMD_FLOAT64_C(-369013.00), EASYSIMD_FLOAT64_C(-265375.00),
        EASYSIMD_FLOAT64_C(-133432.00), EASYSIMD_FLOAT64_C(-286968.00), EASYSIMD_FLOAT64_C(-955044.00), EASYSIMD_FLOAT64_C(-390706.00) },
      { EASYSIMD_FLOAT64_C(400595.00), EASYSIMD_FLOAT64_C( 59942.00), EASYSIMD_FLOAT64_C(-369012.00), EASYSIMD_FLOAT64_C(-265374.00),
        EASYSIMD_FLOAT64_C(-133431.00), EASYSIMD_FLOAT64_C(-286967.00), EASYSIMD_FLOAT64_C(-955043.00), EASYSIMD_FLOAT64_C(-390705.00) },
      { EASYSIMD_FLOAT64_C(400594.00), EASYSIMD_FLOAT64_C( 59941.00), EASYSIMD_FLOAT64_C(-369012.00), EASYSIMD_FLOAT64_C(-265374.00),
        EASYSIMD_FLOAT64_C(-133431.00), EASYSIMD_FLOAT64_C(-286967.00), EASYSIMD_FLOAT64_C(-955043.00), EASYSIMD_FLOAT64_C(-390705.00) },
      { EASYSIMD_FLOAT64_C(400594.00), EASYSIMD_FLOAT64_C( 59942.00), EASYSIMD_FLOAT64_C(-369012.00), EASYSIMD_FLOAT64_C(-265374.00),
        EASYSIMD_FLOAT64_C(-133431.00), EASYSIMD_FLOAT64_C(-286967.00), EASYSIMD_FLOAT64_C(-955044.00), EASYSIMD_FLOAT64_C(-390705.00) } },
    { { EASYSIMD_FLOAT64_C(   420.24), EASYSIMD_FLOAT64_C(   511.20), EASYSIMD_FLOAT64_C(  -841.85), EASYSIMD_FLOAT64_C(   -86.40),
        EASYSIMD_FLOAT64_C(  -121.86), EASYSIMD_FLOAT64_C(   616.95), EASYSIMD_FLOAT64_C(   627.96), EASYSIMD_FLOAT64_C(   969.75) },
      { EASYSIMD_FLOAT64_C(  -786.71), EASYSIMD_FLOAT64_C(    64.96), EASYSIMD_FLOAT64_C(   252.64), EASYSIMD_FLOAT64_C(   183.20),
        EASYSIMD_FLOAT64_C(   157.46), EASYSIMD_FLOAT64_C(  -129.00), EASYSIMD_FLOAT64_C(  -310.78), EASYSIMD_FLOAT64_C(  -409.69) },
      { EASYSIMD_FLOAT64_C(-330607.00), EASYSIMD_FLOAT64_C( 33208.00), EASYSIMD_FLOAT64_C(-212685.00), EASYSIMD_FLOAT64_C(-15828.00),
        EASYSIMD_FLOAT64_C(-19188.00), EASYSIMD_FLOAT64_C(-79587.00), EASYSIMD_FLOAT64_C(-195157.00), EASYSIMD_FLOAT64_C(-397297.00) },
      { EASYSIMD_FLOAT64_C(-330608.00), EASYSIMD_FLOAT64_C( 33207.00), EASYSIMD_FLOAT64_C(-212685.00), EASYSIMD_FLOAT64_C(-15829.00),
        EASYSIMD_FLOAT64_C(-19189.00), EASYSIMD_FLOAT64_C(-79587.00), EASYSIMD_FLOAT64_C(-195158.00), EASYSIMD_FLOAT64_C(-397297.00) },
      { EASYSIMD_FLOAT64_C(-330607.00), EASYSIMD_FLOAT64_C( 33208.00), EASYSIMD_FLOAT64_C(-212684.00), EASYSIMD_FLOAT64_C(-15828.00),
        EASYSIMD_FLOAT64_C(-19188.00), EASYSIMD_FLOAT64_C(-79586.00), EASYSIMD_FLOAT64_C(-195157.00), EASYSIMD_FLOAT64_C(-397296.00) },
      { EASYSIMD_FLOAT64_C(-330607.00), EASYSIMD_FLOAT64_C( 33207.00), EASYSIMD_FLOAT64_C(-212684.00), EASYSIMD_FLOAT64_C(-15828.00),
        EASYSIMD_FLOAT64_C(-19188.00), EASYSIMD_FLOAT64_C(-79586.00), EASYSIMD_FLOAT64_C(-195157.00), EASYSIMD_FLOAT64_C(-397296.00) },
      { EASYSIMD_FLOAT64_C(-330607.00), EASYSIMD_FLOAT64_C( 33208.00), EASYSIMD_FLOAT64_C(-212685.00), EASYSIMD_FLOAT64_C(-15828.00),
        EASYSIMD_FLOAT64_C(-19188.00), EASYSIMD_FLOAT64_C(-79587.00), EASYSIMD_FLOAT64_C(-195157.00), EASYSIMD_FLOAT64_C(-397297.00) } },
    { { EASYSIMD_FLOAT64_C(  -889.76), EASYSIMD_FLOAT64_C(  -797.19), EASYSIMD_FLOAT64_C(   234.06), EASYSIMD_FLOAT64_C(   -74.19),
        EASYSIMD_FLOAT64_C(  -141.50), EASYSIMD_FLOAT64_C(   193.98), EASYSIMD_FLOAT64_C(  -592.07), EASYSIMD_FLOAT64_C(  -216.02) },
      { EASYSIMD_FLOAT64_C(  -555.47), EASYSIMD_FLOAT64_C(  -310.58), EASYSIMD_FLOAT64_C(  -471.11), EASYSIMD_FLOAT64_C(  -831.99),
        EASYSIMD_FLOAT64_C(  -477.10), EASYSIMD_FLOAT64_C(  -466.03), EASYSIMD_FLOAT64_C(  -642.39), EASYSIMD_FLOAT64_C(   943.13) },
      { EASYSIMD_FLOAT64_C(494235.00), EASYSIMD_FLOAT64_C(247591.00), EASYSIMD_FLOAT64_C(-110268.00), EASYSIMD_FLOAT64_C( 61725.00),
        EASYSIMD_FLOAT64_C( 67510.00), EASYSIMD_FLOAT64_C(-90400.00), EASYSIMD_FLOAT64_C(380340.00), EASYSIMD_FLOAT64_C(-203735.00) },
      { EASYSIMD_FLOAT64_C(494234.00), EASYSIMD_FLOAT64_C(247591.00), EASYSIMD_FLOAT64_C(-110269.00), EASYSIMD_FLOAT64_C( 61725.00),
        EASYSIMD_FLOAT64_C( 67509.00), EASYSIMD_FLOAT64_C(-90401.00), EASYSIMD_FLOAT64_C(380339.00), EASYSIMD_FLOAT64_C(-203735.00) },
      { EASYSIMD_FLOAT64_C(494235.00), EASYSIMD_FLOAT64_C(247592.00), EASYSIMD_FLOAT64_C(-110268.00), EASYSIMD_FLOAT64_C( 61726.00),
        EASYSIMD_FLOAT64_C( 67510.00), EASYSIMD_FLOAT64_C(-90400.00), EASYSIMD_FLOAT64_C(380340.00), EASYSIMD_FLOAT64_C(-203734.00) },
      { EASYSIMD_FLOAT64_C(494234.00), EASYSIMD_FLOAT64_C(247591.00), EASYSIMD_FLOAT64_C(-110268.00), EASYSIMD_FLOAT64_C( 61725.00),
        EASYSIMD_FLOAT64_C( 67509.00), EASYSIMD_FLOAT64_C(-90400.00), EASYSIMD_FLOAT64_C(380339.00), EASYSIMD_FLOAT64_C(-203734.00) },
      { EASYSIMD_FLOAT64_C(494235.00), EASYSIMD_FLOAT64_C(247591.00), EASYSIMD_FLOAT64_C(-110268.00), EASYSIMD_FLOAT64_C( 61725.00),
        EASYSIMD_FLOAT64_C( 67510.00), EASYSIMD_FLOAT64_C(-90400.00), EASYSIMD_FLOAT64_C(380340.00), EASYSIMD_FLOAT64_C(-203735.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r;
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);

    easysimd__m512d nearest_inf = easysimd_mm512_loadu_pd(test_vec[i].nearest_inf);
    easysimd__m512d neg_inf = easysimd_mm512_loadu_pd(test_vec[i].neg_inf);
    easysimd__m512d pos_inf = easysimd_mm512_loadu_pd(test_vec[i].pos_inf);
    easysimd__m512d zero = easysimd_mm512_loadu_pd(test_vec[i].zero);
    easysimd__m512d direction = easysimd_mm512_loadu_pd(test_vec[i].direction);

    r = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512d_close(r, nearest_inf, 1);

    r = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512d_close(r, neg_inf, 1);

    r = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512d_close(r, pos_inf, 1);

    r = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512d_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mul_round_pd");
    easysimd_assert_m512d_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512d nearest_inf = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512d neg_inf = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512d pos_inf = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512d zero = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512d direction = easysimd_mm512_mul_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, nearest_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, neg_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, pos_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, zero, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, direction, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_mask_mul_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -821.30), EASYSIMD_FLOAT64_C( -768.64),
                         EASYSIMD_FLOAT64_C(  -18.18), EASYSIMD_FLOAT64_C( -679.16),
                         EASYSIMD_FLOAT64_C( -992.98), EASYSIMD_FLOAT64_C( -764.30),
                         EASYSIMD_FLOAT64_C(  419.74), EASYSIMD_FLOAT64_C(  970.61)),
      UINT8_C( 76),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -167.78), EASYSIMD_FLOAT64_C( -432.98),
                         EASYSIMD_FLOAT64_C( -407.63), EASYSIMD_FLOAT64_C(  -78.73),
                         EASYSIMD_FLOAT64_C( -377.24), EASYSIMD_FLOAT64_C( -338.63),
                         EASYSIMD_FLOAT64_C( -681.32), EASYSIMD_FLOAT64_C( -483.94)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -931.82), EASYSIMD_FLOAT64_C( -180.10),
                         EASYSIMD_FLOAT64_C( -213.80), EASYSIMD_FLOAT64_C( -618.07),
                         EASYSIMD_FLOAT64_C(  922.09), EASYSIMD_FLOAT64_C( -681.84),
                         EASYSIMD_FLOAT64_C( -317.54), EASYSIMD_FLOAT64_C(  448.08)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -821.30), EASYSIMD_FLOAT64_C(77979.70),
                         EASYSIMD_FLOAT64_C(  -18.18), EASYSIMD_FLOAT64_C( -679.16),
                         EASYSIMD_FLOAT64_C(-347849.23), EASYSIMD_FLOAT64_C(230891.48),
                         EASYSIMD_FLOAT64_C(  419.74), EASYSIMD_FLOAT64_C(  970.61)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -706.27), EASYSIMD_FLOAT64_C( -736.90),
                         EASYSIMD_FLOAT64_C(  388.85), EASYSIMD_FLOAT64_C( -452.26),
                         EASYSIMD_FLOAT64_C( -983.38), EASYSIMD_FLOAT64_C( -800.62),
                         EASYSIMD_FLOAT64_C(  310.59), EASYSIMD_FLOAT64_C(  810.60)),
      UINT8_C( 87),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -613.25), EASYSIMD_FLOAT64_C(  846.16),
                         EASYSIMD_FLOAT64_C(  824.90), EASYSIMD_FLOAT64_C( -554.53),
                         EASYSIMD_FLOAT64_C( -163.66), EASYSIMD_FLOAT64_C(  923.31),
                         EASYSIMD_FLOAT64_C( -996.35), EASYSIMD_FLOAT64_C( -303.21)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -11.46), EASYSIMD_FLOAT64_C( -598.19),
                         EASYSIMD_FLOAT64_C(  495.52), EASYSIMD_FLOAT64_C(  117.93),
                         EASYSIMD_FLOAT64_C(  291.55), EASYSIMD_FLOAT64_C(  189.90),
                         EASYSIMD_FLOAT64_C( -859.41), EASYSIMD_FLOAT64_C(    9.76)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -706.27), EASYSIMD_FLOAT64_C(-506164.45),
                         EASYSIMD_FLOAT64_C(  388.85), EASYSIMD_FLOAT64_C(-65395.72),
                         EASYSIMD_FLOAT64_C( -983.38), EASYSIMD_FLOAT64_C(175336.57),
                         EASYSIMD_FLOAT64_C(856273.15), EASYSIMD_FLOAT64_C(-2959.33)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -563.02), EASYSIMD_FLOAT64_C( -883.18),
                         EASYSIMD_FLOAT64_C(  852.82), EASYSIMD_FLOAT64_C( -331.20),
                         EASYSIMD_FLOAT64_C( -286.53), EASYSIMD_FLOAT64_C( -422.71),
                         EASYSIMD_FLOAT64_C( -717.56), EASYSIMD_FLOAT64_C( -209.20)),
      UINT8_C( 30),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -241.93), EASYSIMD_FLOAT64_C( -343.53),
                         EASYSIMD_FLOAT64_C(  736.91), EASYSIMD_FLOAT64_C( -835.83),
                         EASYSIMD_FLOAT64_C( -444.99), EASYSIMD_FLOAT64_C(  943.16),
                         EASYSIMD_FLOAT64_C(   17.49), EASYSIMD_FLOAT64_C(  -26.72)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -614.80), EASYSIMD_FLOAT64_C( -251.11),
                         EASYSIMD_FLOAT64_C(  421.22), EASYSIMD_FLOAT64_C( -961.92),
                         EASYSIMD_FLOAT64_C(  971.24), EASYSIMD_FLOAT64_C( -348.19),
                         EASYSIMD_FLOAT64_C( -171.56), EASYSIMD_FLOAT64_C( -420.89)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -563.02), EASYSIMD_FLOAT64_C( -883.18),
                         EASYSIMD_FLOAT64_C(  852.82), EASYSIMD_FLOAT64_C(804001.59),
                         EASYSIMD_FLOAT64_C(-432192.09), EASYSIMD_FLOAT64_C(-328398.88),
                         EASYSIMD_FLOAT64_C(-3000.58), EASYSIMD_FLOAT64_C( -209.20)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  442.66), EASYSIMD_FLOAT64_C(  -69.80),
                         EASYSIMD_FLOAT64_C(  151.84), EASYSIMD_FLOAT64_C(  278.19),
                         EASYSIMD_FLOAT64_C( -105.37), EASYSIMD_FLOAT64_C( -898.05),
                         EASYSIMD_FLOAT64_C(  104.61), EASYSIMD_FLOAT64_C(  131.40)),
      UINT8_C( 92),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -598.49), EASYSIMD_FLOAT64_C(  226.31),
                         EASYSIMD_FLOAT64_C(   -6.29), EASYSIMD_FLOAT64_C(  443.90),
                         EASYSIMD_FLOAT64_C( -544.30), EASYSIMD_FLOAT64_C( -925.04),
                         EASYSIMD_FLOAT64_C(  484.35), EASYSIMD_FLOAT64_C( -740.68)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   72.46), EASYSIMD_FLOAT64_C(  -87.05),
                         EASYSIMD_FLOAT64_C( -714.68), EASYSIMD_FLOAT64_C(  393.49),
                         EASYSIMD_FLOAT64_C(  651.31), EASYSIMD_FLOAT64_C(  480.47),
                         EASYSIMD_FLOAT64_C(  373.84), EASYSIMD_FLOAT64_C(  843.89)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  442.66), EASYSIMD_FLOAT64_C(-19700.29),
                         EASYSIMD_FLOAT64_C(  151.84), EASYSIMD_FLOAT64_C(174670.21),
                         EASYSIMD_FLOAT64_C(-354508.03), EASYSIMD_FLOAT64_C(-444453.97),
                         EASYSIMD_FLOAT64_C(  104.61), EASYSIMD_FLOAT64_C(  131.40)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  988.68), EASYSIMD_FLOAT64_C(  353.35),
                         EASYSIMD_FLOAT64_C( -309.14), EASYSIMD_FLOAT64_C( -266.17),
                         EASYSIMD_FLOAT64_C(  819.45), EASYSIMD_FLOAT64_C(  592.47),
                         EASYSIMD_FLOAT64_C(  382.11), EASYSIMD_FLOAT64_C(  516.02)),
      UINT8_C( 87),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   51.49), EASYSIMD_FLOAT64_C( -696.81),
                         EASYSIMD_FLOAT64_C(  178.38), EASYSIMD_FLOAT64_C(  907.89),
                         EASYSIMD_FLOAT64_C(  646.15), EASYSIMD_FLOAT64_C(  281.27),
                         EASYSIMD_FLOAT64_C(  226.71), EASYSIMD_FLOAT64_C( -906.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -823.13), EASYSIMD_FLOAT64_C( -506.07),
                         EASYSIMD_FLOAT64_C( -848.31), EASYSIMD_FLOAT64_C( -467.13),
                         EASYSIMD_FLOAT64_C(  559.51), EASYSIMD_FLOAT64_C( -498.81),
                         EASYSIMD_FLOAT64_C(  598.24), EASYSIMD_FLOAT64_C(  523.97)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  988.68), EASYSIMD_FLOAT64_C(352634.64),
                         EASYSIMD_FLOAT64_C( -309.14), EASYSIMD_FLOAT64_C(-424102.66),
                         EASYSIMD_FLOAT64_C(  819.45), EASYSIMD_FLOAT64_C(-140300.29),
                         EASYSIMD_FLOAT64_C(135626.99), EASYSIMD_FLOAT64_C(-474963.09)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -811.79), EASYSIMD_FLOAT64_C(  522.04),
                         EASYSIMD_FLOAT64_C(  594.85), EASYSIMD_FLOAT64_C(    0.75),
                         EASYSIMD_FLOAT64_C( -855.43), EASYSIMD_FLOAT64_C(  660.82),
                         EASYSIMD_FLOAT64_C( -308.44), EASYSIMD_FLOAT64_C(  882.56)),
      UINT8_C( 62),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -252.73), EASYSIMD_FLOAT64_C( -915.63),
                         EASYSIMD_FLOAT64_C( -935.95), EASYSIMD_FLOAT64_C( -722.20),
                         EASYSIMD_FLOAT64_C( -497.29), EASYSIMD_FLOAT64_C( -166.63),
                         EASYSIMD_FLOAT64_C(  516.64), EASYSIMD_FLOAT64_C( -317.86)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -925.15), EASYSIMD_FLOAT64_C(  466.19),
                         EASYSIMD_FLOAT64_C(  263.72), EASYSIMD_FLOAT64_C(  424.85),
                         EASYSIMD_FLOAT64_C(  205.96), EASYSIMD_FLOAT64_C(  401.84),
                         EASYSIMD_FLOAT64_C(  361.23), EASYSIMD_FLOAT64_C(  807.53)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -811.79), EASYSIMD_FLOAT64_C(  522.04),
                         EASYSIMD_FLOAT64_C(-246828.73), EASYSIMD_FLOAT64_C(-306826.67),
                         EASYSIMD_FLOAT64_C(-102421.85), EASYSIMD_FLOAT64_C(-66958.60),
                         EASYSIMD_FLOAT64_C(186625.87), EASYSIMD_FLOAT64_C(  882.56)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  915.95), EASYSIMD_FLOAT64_C( -721.40),
                         EASYSIMD_FLOAT64_C( -153.01), EASYSIMD_FLOAT64_C(  377.63),
                         EASYSIMD_FLOAT64_C(  983.59), EASYSIMD_FLOAT64_C( -647.06),
                         EASYSIMD_FLOAT64_C(  224.30), EASYSIMD_FLOAT64_C(  -39.06)),
      UINT8_C( 70),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  724.37), EASYSIMD_FLOAT64_C( -108.80),
                         EASYSIMD_FLOAT64_C( -716.02), EASYSIMD_FLOAT64_C( -552.47),
                         EASYSIMD_FLOAT64_C(  411.46), EASYSIMD_FLOAT64_C( -439.29),
                         EASYSIMD_FLOAT64_C(  397.99), EASYSIMD_FLOAT64_C(  -31.94)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -763.99), EASYSIMD_FLOAT64_C(  279.59),
                         EASYSIMD_FLOAT64_C(  318.18), EASYSIMD_FLOAT64_C(   57.40),
                         EASYSIMD_FLOAT64_C(   13.78), EASYSIMD_FLOAT64_C( -535.45),
                         EASYSIMD_FLOAT64_C(   52.16), EASYSIMD_FLOAT64_C( -903.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  915.95), EASYSIMD_FLOAT64_C(-30419.39),
                         EASYSIMD_FLOAT64_C( -153.01), EASYSIMD_FLOAT64_C(  377.63),
                         EASYSIMD_FLOAT64_C(  983.59), EASYSIMD_FLOAT64_C(235217.83),
                         EASYSIMD_FLOAT64_C(20759.16), EASYSIMD_FLOAT64_C(  -39.06)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -219.27), EASYSIMD_FLOAT64_C(  499.44),
                         EASYSIMD_FLOAT64_C( -493.92), EASYSIMD_FLOAT64_C(  481.91),
                         EASYSIMD_FLOAT64_C(  270.70), EASYSIMD_FLOAT64_C(  857.18),
                         EASYSIMD_FLOAT64_C( -745.19), EASYSIMD_FLOAT64_C( -960.45)),
      UINT8_C(113),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -135.86), EASYSIMD_FLOAT64_C( -159.92),
                         EASYSIMD_FLOAT64_C(  756.29), EASYSIMD_FLOAT64_C( -526.68),
                         EASYSIMD_FLOAT64_C(    5.30), EASYSIMD_FLOAT64_C(  278.11),
                         EASYSIMD_FLOAT64_C(  884.85), EASYSIMD_FLOAT64_C(  638.85)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  726.26), EASYSIMD_FLOAT64_C(  110.13),
                         EASYSIMD_FLOAT64_C( -961.42), EASYSIMD_FLOAT64_C(   96.39),
                         EASYSIMD_FLOAT64_C(  930.93), EASYSIMD_FLOAT64_C( -241.35),
                         EASYSIMD_FLOAT64_C( -108.47), EASYSIMD_FLOAT64_C(  -69.81)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -219.27), EASYSIMD_FLOAT64_C(-17611.99),
                         EASYSIMD_FLOAT64_C(-727112.33), EASYSIMD_FLOAT64_C(-50766.69),
                         EASYSIMD_FLOAT64_C(  270.70), EASYSIMD_FLOAT64_C(  857.18),
                         EASYSIMD_FLOAT64_C( -745.19), EASYSIMD_FLOAT64_C(-44598.12)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mul_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mul_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mul_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
     { UINT8_C(  4),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  232.34), EASYSIMD_FLOAT64_C(  716.29),
                         EASYSIMD_FLOAT64_C(  520.56), EASYSIMD_FLOAT64_C( -458.82),
                         EASYSIMD_FLOAT64_C(  550.79), EASYSIMD_FLOAT64_C(  687.92),
                         EASYSIMD_FLOAT64_C( -593.10), EASYSIMD_FLOAT64_C( -620.76)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -156.55), EASYSIMD_FLOAT64_C( -441.38),
                         EASYSIMD_FLOAT64_C(  554.99), EASYSIMD_FLOAT64_C(  294.84),
                         EASYSIMD_FLOAT64_C( -270.30), EASYSIMD_FLOAT64_C( -228.66),
                         EASYSIMD_FLOAT64_C(  910.49), EASYSIMD_FLOAT64_C( -483.54)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(-157299.79),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(165),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -526.05), EASYSIMD_FLOAT64_C(  453.25),
                         EASYSIMD_FLOAT64_C(  821.16), EASYSIMD_FLOAT64_C( -906.31),
                         EASYSIMD_FLOAT64_C( -873.91), EASYSIMD_FLOAT64_C( -472.79),
                         EASYSIMD_FLOAT64_C( -675.37), EASYSIMD_FLOAT64_C( -955.90)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  305.84), EASYSIMD_FLOAT64_C( -400.58),
                         EASYSIMD_FLOAT64_C( -475.09), EASYSIMD_FLOAT64_C( -582.28),
                         EASYSIMD_FLOAT64_C( -849.06), EASYSIMD_FLOAT64_C( -392.73),
                         EASYSIMD_FLOAT64_C( -370.73), EASYSIMD_FLOAT64_C( -928.94)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-160887.13), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(-390124.90), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(185678.82),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(887973.75)) },
    { UINT8_C(175),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  871.20), EASYSIMD_FLOAT64_C( -761.38),
                         EASYSIMD_FLOAT64_C( -106.42), EASYSIMD_FLOAT64_C( -228.29),
                         EASYSIMD_FLOAT64_C( -864.78), EASYSIMD_FLOAT64_C( -773.10),
                         EASYSIMD_FLOAT64_C(  984.91), EASYSIMD_FLOAT64_C( -982.29)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -488.22), EASYSIMD_FLOAT64_C(  361.25),
                         EASYSIMD_FLOAT64_C( -346.47), EASYSIMD_FLOAT64_C(  411.25),
                         EASYSIMD_FLOAT64_C(  117.68), EASYSIMD_FLOAT64_C(  448.38),
                         EASYSIMD_FLOAT64_C( -319.67), EASYSIMD_FLOAT64_C(  -97.98)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-425337.26), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(36871.34), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(-101767.31), EASYSIMD_FLOAT64_C(-346642.58),
                         EASYSIMD_FLOAT64_C(-314846.18), EASYSIMD_FLOAT64_C(96244.77)) },
    { UINT8_C(195),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -43.54), EASYSIMD_FLOAT64_C(  387.84),
                         EASYSIMD_FLOAT64_C( -190.98), EASYSIMD_FLOAT64_C(  468.25),
                         EASYSIMD_FLOAT64_C( -832.05), EASYSIMD_FLOAT64_C( -600.33),
                         EASYSIMD_FLOAT64_C( -246.00), EASYSIMD_FLOAT64_C(  160.40)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -279.42), EASYSIMD_FLOAT64_C(  980.35),
                         EASYSIMD_FLOAT64_C(  897.98), EASYSIMD_FLOAT64_C( -354.38),
                         EASYSIMD_FLOAT64_C(  689.03), EASYSIMD_FLOAT64_C(  555.84),
                         EASYSIMD_FLOAT64_C(  823.79), EASYSIMD_FLOAT64_C( -979.93)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(12165.95), EASYSIMD_FLOAT64_C(380218.94),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(-202652.34), EASYSIMD_FLOAT64_C(-157180.77)) },
    { UINT8_C(236),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  821.55), EASYSIMD_FLOAT64_C(  740.38),
                         EASYSIMD_FLOAT64_C( -934.60), EASYSIMD_FLOAT64_C(  694.91),
                         EASYSIMD_FLOAT64_C(  432.52), EASYSIMD_FLOAT64_C(  380.89),
                         EASYSIMD_FLOAT64_C(  -22.14), EASYSIMD_FLOAT64_C(  683.08)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  735.17), EASYSIMD_FLOAT64_C(  313.88),
                         EASYSIMD_FLOAT64_C( -529.80), EASYSIMD_FLOAT64_C( -869.79),
                         EASYSIMD_FLOAT64_C(  294.43), EASYSIMD_FLOAT64_C(  958.02),
                         EASYSIMD_FLOAT64_C(  383.81), EASYSIMD_FLOAT64_C(  520.19)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(603978.91), EASYSIMD_FLOAT64_C(232390.47),
                         EASYSIMD_FLOAT64_C(495151.08), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(127346.86), EASYSIMD_FLOAT64_C(364900.24),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(144),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -704.15), EASYSIMD_FLOAT64_C(  418.80),
                         EASYSIMD_FLOAT64_C( -562.82), EASYSIMD_FLOAT64_C(  910.01),
                         EASYSIMD_FLOAT64_C(  513.17), EASYSIMD_FLOAT64_C(  314.44),
                         EASYSIMD_FLOAT64_C(  866.48), EASYSIMD_FLOAT64_C(  466.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  663.36), EASYSIMD_FLOAT64_C(  883.11),
                         EASYSIMD_FLOAT64_C(  475.36), EASYSIMD_FLOAT64_C(  451.49),
                         EASYSIMD_FLOAT64_C(  246.05), EASYSIMD_FLOAT64_C( -122.55),
                         EASYSIMD_FLOAT64_C(  401.83), EASYSIMD_FLOAT64_C(  557.78)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-467104.94), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(410860.41),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(181),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -758.51), EASYSIMD_FLOAT64_C( -164.55),
                         EASYSIMD_FLOAT64_C(  334.89), EASYSIMD_FLOAT64_C( -549.60),
                         EASYSIMD_FLOAT64_C(  344.01), EASYSIMD_FLOAT64_C( -985.45),
                         EASYSIMD_FLOAT64_C( -235.88), EASYSIMD_FLOAT64_C(  450.77)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -408.01), EASYSIMD_FLOAT64_C(  723.62),
                         EASYSIMD_FLOAT64_C( -159.29), EASYSIMD_FLOAT64_C(  720.82),
                         EASYSIMD_FLOAT64_C( -893.97), EASYSIMD_FLOAT64_C(  826.45),
                         EASYSIMD_FLOAT64_C(   -3.06), EASYSIMD_FLOAT64_C(  902.05)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(309479.67), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(-53344.63), EASYSIMD_FLOAT64_C(-396162.67),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(-814425.15),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(406617.08)) },
    { UINT8_C(211),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  311.61), EASYSIMD_FLOAT64_C( -930.67),
                         EASYSIMD_FLOAT64_C(  465.44), EASYSIMD_FLOAT64_C( -366.35),
                         EASYSIMD_FLOAT64_C(  205.36), EASYSIMD_FLOAT64_C(  276.19),
                         EASYSIMD_FLOAT64_C(  975.10), EASYSIMD_FLOAT64_C( -338.46)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  817.02), EASYSIMD_FLOAT64_C( -834.50),
                         EASYSIMD_FLOAT64_C( -648.42), EASYSIMD_FLOAT64_C(  761.90),
                         EASYSIMD_FLOAT64_C(   24.27), EASYSIMD_FLOAT64_C(  838.31),
                         EASYSIMD_FLOAT64_C( -854.11), EASYSIMD_FLOAT64_C(  403.52)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(254591.60), EASYSIMD_FLOAT64_C(776644.11),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(-279122.07),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(-832842.66), EASYSIMD_FLOAT64_C(-136575.38)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mul_pd(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mul_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mul_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mul_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mul_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mul_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mul_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mul_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mul_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mul_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mul_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mul_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mul_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mul_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mul_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mul_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mul_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mul_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mul_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mul_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mul_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mul_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mul_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mul_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mul_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mul_round_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mul_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mul_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mul_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mul_round_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mul_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mul_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
