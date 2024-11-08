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
 */

#define EASYSIMD_TEST_X86_AVX512_INSN andnot

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/andnot.h>

static int
test_easysimd_mm_mask_andnot_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[4];
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(  1268820680),  INT32_C(  1907523707), -INT32_C(  2055460442), -INT32_C(   344174492) },
      UINT8_C( 61),
      { -INT32_C(  2019774380),  INT32_C(    11458960),  INT32_C(  1051068435),  INT32_C(   578234828) },
      {  INT32_C(   144518024),  INT32_C(   112070365), -INT32_C(   613731445),  INT32_C(    51992239) },
      {  INT32_C(   134286216),  INT32_C(  1907523707), -INT32_C(  1052167288),  INT32_C(    17323555) } },
    { { -INT32_C(  1752451833), -INT32_C(  1583924850),  INT32_C(   249577025), -INT32_C(   265267353) },
      UINT8_C(130),
      { -INT32_C(   614467379), -INT32_C(   630823257),  INT32_C(   663372497),  INT32_C(   271486299) },
      {  INT32_C(  1386137112), -INT32_C(  1684848547),  INT32_C(  1996726560), -INT32_C(  1627786287) },
      { -INT32_C(  1752451833),  INT32_C(    26280024),  INT32_C(   249577025), -INT32_C(   265267353) } },
    { { -INT32_C(  1837475605), -INT32_C(  1888624450),  INT32_C(  2109208354), -INT32_C(  1651645052) },
      UINT8_C(171),
      {  INT32_C(  1812590380),  INT32_C(   613196930), -INT32_C(  1695218777),  INT32_C(  1418106108) },
      { -INT32_C(   300738546),  INT32_C(  2081530501),  INT32_C(  1057066585),  INT32_C(  1223335452) },
      { -INT32_C(  2112745470),  INT32_C(  1477509637),  INT32_C(  2109208354),  INT32_C(   141036032) } },
    { {  INT32_C(   263517069),  INT32_C(  1060323480),  INT32_C(  1071262019), -INT32_C(   879533891) },
      UINT8_C(120),
      {  INT32_C(  1241430438),  INT32_C(  1503820490),  INT32_C(   427155835), -INT32_C(  1079591476) },
      { -INT32_C(  1319651727),  INT32_C(   318019561), -INT32_C(   774950031),  INT32_C(  1833540294) },
      {  INT32_C(   263517069),  INT32_C(  1060323480),  INT32_C(  1071262019),  INT32_C(  1078526466) } },
    { {  INT32_C(   515262292),  INT32_C(  1031231682),  INT32_C(   106425402),  INT32_C(   449183145) },
      UINT8_C(179),
      { -INT32_C(  1264727267), -INT32_C(   232411201),  INT32_C(   431617663), -INT32_C(  2022889921) },
      {  INT32_C(   877235420), -INT32_C(   261192189), -INT32_C(   610700067),  INT32_C(  1468970042) },
      {  INT32_C(     4195520),  INT32_C(     4849664),  INT32_C(   106425402),  INT32_C(   449183145) } },
    { {  INT32_C(  1057696639),  INT32_C(  1513173211),  INT32_C(  1702095398), -INT32_C(   320020208) },
      UINT8_C(109),
      { -INT32_C(  1133502155),  INT32_C(    77160591), -INT32_C(  1371638534),  INT32_C(   774739203) },
      { -INT32_C(   804688736), -INT32_C(  1997118562), -INT32_C(  1197909033),  INT32_C(  2099610952) },
      {  INT32_C(  1074351232),  INT32_C(  1513173211),  INT32_C(   276892421),  INT32_C(  1358954568) } },
    { {  INT32_C(   909743526), -INT32_C(   264580106),  INT32_C(  1251899463), -INT32_C(  1384527091) },
      UINT8_C( 56),
      { -INT32_C(   422150782), -INT32_C(   826450317),  INT32_C(  2098623991),  INT32_C(   790860954) },
      { -INT32_C(  1608164915),  INT32_C(   199759251),  INT32_C(  2132292275),  INT32_C(   783795627) },
      {  INT32_C(   909743526), -INT32_C(   264580106),  INT32_C(  1251899463),  INT32_C(     9716001) } },
    { { -INT32_C(  1256944318), -INT32_C(   477834773), -INT32_C(   530539962), -INT32_C(    66092242) },
      UINT8_C(221),
      {  INT32_C(  1248894005), -INT32_C(  1208058748),  INT32_C(  1516404116), -INT32_C(  1046704076) },
      {  INT32_C(  1974227620),  INT32_C(  1891340246),  INT32_C(  1939839983), -INT32_C(   531588181) },
      {  INT32_C(   898384512), -INT32_C(   477834773),  INT32_C(   563970667),  INT32_C(   541068171) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_andnot_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_andnot_epi32");
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
    easysimd__m128i r = easysimd_mm_mask_andnot_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_andnot_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 55),
      {  INT32_C(  1018964672), -INT32_C(  1513000152),  INT32_C(  1725508565),  INT32_C(   420125383) },
      { -INT32_C(   688947129), -INT32_C(   674865093), -INT32_C(  1702741505),  INT32_C(  1238459017) },
      { -INT32_C(  1035763705),  INT32_C(  1376124947), -INT32_C(  1744695254),  INT32_C(           0) } },
    { UINT8_C(140),
      {  INT32_C(    11896461), -INT32_C(  2099881385), -INT32_C(   834061260), -INT32_C(   971676858) },
      { -INT32_C(  1308431278), -INT32_C(   340665934), -INT32_C(  1116452005), -INT32_C(  1438008035) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   825508683),  INT32_C(   675847193) } },
    { UINT8_C( 67),
      {  INT32_C(  1503308798), -INT32_C(  1114825599), -INT32_C(   922526874),  INT32_C(  1562102385) },
      { -INT32_C(  1525690676), -INT32_C(   889128321),  INT32_C(   736673134),  INT32_C(    91197958) },
      { -INT32_C(  1543158784),  INT32_C(  1107354238),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 61),
      {  INT32_C(   650010121),  INT32_C(  1183611883),  INT32_C(  1236751743),  INT32_C(  1041568880) },
      {  INT32_C(   499038755),  INT32_C(  2005633466),  INT32_C(    58570609),  INT32_C(   792822310) },
      {  INT32_C(   419471394),  INT32_C(           0),  INT32_C(    38314496),  INT32_C(    21004806) } },
    { UINT8_C(224),
      {  INT32_C(  2076923391),  INT32_C(   922358497), -INT32_C(   559463479),  INT32_C(   302114136) },
      {  INT32_C(   751574947),  INT32_C(  1637696426),  INT32_C(  1132962240), -INT32_C(   517752862) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 12),
      {  INT32_C(    32398575), -INT32_C(  1714805674), -INT32_C(  1309562677), -INT32_C(   917240918) },
      {  INT32_C(   309559759), -INT32_C(  1076636386),  INT32_C(   329324124),  INT32_C(   673219385) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(    33560084),  INT32_C(   539000849) } },
    { UINT8_C(223),
      {  INT32_C(   842410254), -INT32_C(  1661022220), -INT32_C(  1002000447), -INT32_C(  2070671613) },
      {  INT32_C(  1487120003), -INT32_C(  1884003719), -INT32_C(  2016819196), -INT32_C(   161025560) },
      {  INT32_C(  1216448129),  INT32_C(    50339849),  INT32_C(    59326468),  INT32_C(  1919086824) } },
    { UINT8_C( 27),
      {  INT32_C(  1812932764), -INT32_C(   718427354),  INT32_C(    31060466),  INT32_C(   746872197) },
      {  INT32_C(  1671814144),  INT32_C(  1499936145), -INT32_C(   280891650), -INT32_C(   250988716) },
      {  INT32_C(    60871680),  INT32_C(   138548369),  INT32_C(           0), -INT32_C(   787865008) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_andnot_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_andnot_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_andnot_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_andnot_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[2];
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 5234437560920630945), -INT64_C( 7913601272916724613) },
      UINT8_C( 25),
      {  INT64_C( 5333673027490739666),  INT64_C( 1863242297812972868) },
      { -INT64_C( 1012582329507487251), -INT64_C(  890563596805475507) },
      { -INT64_C( 5624269567832096723), -INT64_C( 7913601272916724613) } },
    { { -INT64_C( 7588002068234088949), -INT64_C( 3598958972480892427) },
      UINT8_C( 78),
      {  INT64_C( 3156736604623226347),  INT64_C( 5449143328682236030) },
      {  INT64_C(  661523061603632697),  INT64_C( 1981786353195164229) },
      { -INT64_C( 7588002068234088949),  INT64_C( 1153062242676122113) } },
    { {  INT64_C(  387426692611476477),  INT64_C( 2524893121982606274) },
      UINT8_C( 62),
      { -INT64_C( 6043830103552074585),  INT64_C( 6819004898686554859) },
      { -INT64_C( 2179485797233404268),  INT64_C( 3529114058333323138) },
      {  INT64_C(  387426692611476477),  INT64_C( 2330619438996598016) } },
    { { -INT64_C( 2629772394057961398), -INT64_C( 1608312466129855107) },
      UINT8_C(246),
      {  INT64_C( 2882037303829391251), -INT64_C( 6768099132013692455) },
      { -INT64_C( 4209336887651604756),  INT64_C( 2811095626330858989) },
      { -INT64_C( 2629772394057961398),  INT64_C(  360573860527621156) } },
    { {  INT64_C( 1638081274832957419),  INT64_C( 4229424639318518876) },
      UINT8_C(214),
      { -INT64_C( 3701955703842451132), -INT64_C( 2302316114654221027) },
      { -INT64_C( 4095685916740444118),  INT64_C( 3230856541025613585) },
      {  INT64_C( 1638081274832957419),  INT64_C(  923880347640533504) } },
    { {  INT64_C( 3698157978753373344),  INT64_C( 7259346550522163756) },
      UINT8_C(  1),
      { -INT64_C( 7989346869243530789),  INT64_C(  803513165575495735) },
      { -INT64_C( 5678663101363147643), -INT64_C( 6337884312818741591) },
      {  INT64_C( 2310721151466310660),  INT64_C( 7259346550522163756) } },
    { { -INT64_C( 4437216016468580480),  INT64_C( 1295187919417959927) },
      UINT8_C( 26),
      {  INT64_C( 6132155875325449755), -INT64_C( 7528388898781518666) },
      { -INT64_C( 5742120541672611967),  INT64_C( 5959473108107410987) },
      { -INT64_C( 4437216016468580480),  INT64_C( 4625202386139093513) } },
    { {  INT64_C( 5620734883216412003),  INT64_C( 7910248304440713728) },
      UINT8_C( 55),
      {  INT64_C( 5978442832732669416),  INT64_C( 1342033399594520814) },
      {  INT64_C( 7745058078196386207),  INT64_C( 6439237557469817420) },
      {  INT64_C( 2956693734066489367),  INT64_C( 5278218973804308992) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_andnot_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_andnot_epi64");
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
    easysimd__m128i r = easysimd_mm_mask_andnot_epi64(src, k, a, b);

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
test_easysimd_mm_maskz_andnot_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[] = {
    { UINT8_C(137),
      { -INT64_C( 5981079256067857314),  INT64_C( 2519204174066891386) },
      { -INT64_C( 7569750612252003923), -INT64_C( 3279358898246052049) },
      {  INT64_C( 1297045371307970977),  INT64_C(                   0) } },
    { UINT8_C(133),
      {  INT64_C( 1983357215349728027),  INT64_C( 4966370685161364797) },
      { -INT64_C( 7166979055937351037),  INT64_C( 1563844159281855950) },
      { -INT64_C( 8932465430877958016),  INT64_C(                   0) } },
    { UINT8_C(198),
      { -INT64_C( 8944778164460642220), -INT64_C( 7182790563772097097) },
      { -INT64_C( 5593544174405047094), -INT64_C( 7676454551931810354) },
      {  INT64_C(                   0),  INT64_C(   82825387119771720) } },
    { UINT8_C(226),
      {  INT64_C( 3501843464784983326),  INT64_C( 7334065512990205794) },
      {  INT64_C( 5008651186993780982), -INT64_C( 3991003274675866631) },
      {  INT64_C(                   0), -INT64_C( 8640144881361743719) } },
    { UINT8_C( 80),
      { -INT64_C( 6155132873907918299), -INT64_C( 5649263974552464947) },
      {  INT64_C(  191875566669417827),  INT64_C( 2855128237739893811) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 30),
      {  INT64_C( 9035079487710124034), -INT64_C( 1469431739778202257) },
      {  INT64_C( 6471880423335676911), -INT64_C( 4540893142515572888) },
      {  INT64_C(                   0),  INT64_C(   27021738508818944) } },
    { UINT8_C(218),
      { -INT64_C( 4622869454361569699), -INT64_C(  679962542237478886) },
      {  INT64_C( 2101005087117356291),  INT64_C( 6843953012891387294) },
      {  INT64_C(                   0),  INT64_C(  606457628540929412) } },
    { UINT8_C(242),
      { -INT64_C( 5756046042522204224),  INT64_C( 9012919258117291976) },
      {  INT64_C(  153009126920410173), -INT64_C(  431957191557955725) },
      {  INT64_C(                   0), -INT64_C( 9078938913589952461) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_andnot_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_andnot_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_andnot_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_andnot_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[4];
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[8] = {
    { { -INT32_C(  1099624487),  INT32_C(  1877159829), -INT32_C(   742864515), -INT32_C(  1937340912) },
      UINT8_C( 77),
      {  INT32_C(   940091094),  INT32_C(    28183244), -INT32_C(  1545143419), -INT32_C(   780362046) },
      {  INT32_C(  1130773012), -INT32_C(   406792674),  INT32_C(   351769486), -INT32_C(   261979367) },
      {  INT32_C(  1130764288),  INT32_C(  1877159829),  INT32_C(   336658442),  INT32_C(   537002265) } },
    { { -INT32_C(   165123542), -INT32_C(   101198220), -INT32_C(  1818436143), -INT32_C(  1872488068) },
      UINT8_C( 83),
      { -INT32_C(  1603152950),  INT32_C(   640637075), -INT32_C(   750828720),  INT32_C(   268251045) },
      {  INT32_C(   763622487), -INT32_C(   906068757),  INT32_C(   843419929), -INT32_C(  1064970763) },
      {  INT32_C(   226632725), -INT32_C(   909106072), -INT32_C(  1818436143), -INT32_C(  1872488068) } },
    { {  INT32_C(   996210600), -INT32_C(  1620996273),  INT32_C(  2020843731),  INT32_C(   663253200) },
      UINT8_C(100),
      { -INT32_C(  2008001269), -INT32_C(   442427053),  INT32_C(   869978974),  INT32_C(  1339791960) },
      { -INT32_C(  1969285381),  INT32_C(   391986807),  INT32_C(   585619121),  INT32_C(  1770393438) },
      {  INT32_C(   996210600), -INT32_C(  1620996273),  INT32_C(     2426017),  INT32_C(   663253200) } },
    { { -INT32_C(  1208887708),  INT32_C(  1302106863), -INT32_C(  1098877083),  INT32_C(   202201873) },
      UINT8_C(113),
      { -INT32_C(   337078356), -INT32_C(   895745804), -INT32_C(   165101849),  INT32_C(   458920517) },
      {  INT32_C(   369824387),  INT32_C(   628840622),  INT32_C(   876034521), -INT32_C(   207207609) },
      {  INT32_C(   335740931),  INT32_C(  1302106863), -INT32_C(  1098877083),  INT32_C(   202201873) } },
    { { -INT32_C(   807498022),  INT32_C(  1989770126),  INT32_C(  2121056825), -INT32_C(   677722284) },
      UINT8_C(217),
      { -INT32_C(    41423451), -INT32_C(  1562989207),  INT32_C(   669584356),  INT32_C(  1057152433) },
      {  INT32_C(   919523771),  INT32_C(   745555050),  INT32_C(  2004938416),  INT32_C(   760240264) },
      {  INT32_C(    38277146),  INT32_C(  1989770126),  INT32_C(  2121056825),  INT32_C(     5242888) } },
    { { -INT32_C(  1372858299),  INT32_C(  1766916485), -INT32_C(  1114621428), -INT32_C(   738356713) },
      UINT8_C( 99),
      {  INT32_C(   265161163),  INT32_C(  1757411961), -INT32_C(   739231877),  INT32_C(  1595416199) },
      {  INT32_C(  1256507209),  INT32_C(  1381453080),  INT32_C(  1886000350), -INT32_C(   590070511) },
      {  INT32_C(  1075889664),  INT32_C(   306185472), -INT32_C(  1114621428), -INT32_C(   738356713) } },
    { { -INT32_C(  1058299322),  INT32_C(   388541340),  INT32_C(  1776949474),  INT32_C(  2143879990) },
      UINT8_C(202),
      { -INT32_C(    69023059),  INT32_C(   903427105), -INT32_C(   616085090), -INT32_C(  1088281827) },
      { -INT32_C(  1168317937),  INT32_C(   580678410),  INT32_C(  1633224030),  INT32_C(  2083248334) },
      { -INT32_C(  1058299322),  INT32_C(    33833738),  INT32_C(  1776949474),  INT32_C(  1074387138) } },
    { { -INT32_C(  1015607902), -INT32_C(   520531903), -INT32_C(  1212464999),  INT32_C(  1920392547) },
      UINT8_C(191),
      {  INT32_C(  1187589330), -INT32_C(   828052280),  INT32_C(   496764229),  INT32_C(  1035999280) },
      { -INT32_C(   545356913), -INT32_C(  1132896644), -INT32_C(   148885478),  INT32_C(  2042073766) },
      { -INT32_C(  1724480755),  INT32_C(   811143732), -INT32_C(   501207014),  INT32_C(  1077379718) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_mask_andnot_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_andnot_ps");
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
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_mask_andnot_ps(easysimd_mm_castsi128_ps(src), k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

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
test_easysimd_mm_maskz_andnot_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec[8] = {
    { UINT8_C(191),
      {  INT32_C(  1820835712),  INT32_C(  1756452195),  INT32_C(   177786866),  INT32_C(   312137103) },
      { -INT32_C(  1299220141),  INT32_C(   583814130), -INT32_C(    20331669), -INT32_C(    71482758) },
      { -INT32_C(  1844953005),  INT32_C(    38537872), -INT32_C(   197066743), -INT32_C(   383450512) } },
    { UINT8_C(  1),
      { -INT32_C(  1704695996), -INT32_C(   393425895),  INT32_C(   964138596), -INT32_C(  1433630416) },
      {  INT32_C(  1687961112), -INT32_C(   825245942), -INT32_C(   901198456), -INT32_C(   808762486) },
      {  INT32_C(  1687689240),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(171),
      { -INT32_C(    71014097), -INT32_C(  1956664075), -INT32_C(  1396926429),  INT32_C(  1673881124) },
      { -INT32_C(  1049810686), -INT32_C(   968279048), -INT32_C(   951053436),  INT32_C(   225583070) },
      {  INT32_C(     2686976),  INT32_C(  1140855560),  INT32_C(           0),  INT32_C(   204610010) } },
    { UINT8_C(136),
      { -INT32_C(   495122378), -INT32_C(    16447129),  INT32_C(   723825349), -INT32_C(  1607628937) },
      {  INT32_C(   815329268),  INT32_C(  1286889016), -INT32_C(   836076625),  INT32_C(   609695982) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(    72497288) } },
    { UINT8_C( 64),
      { -INT32_C(   576190508), -INT32_C(  1096636660),  INT32_C(  1379257803), -INT32_C(   364390917) },
      { -INT32_C(   870156434), -INT32_C(  1501860309), -INT32_C(   778810727),  INT32_C(  1964095905) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(192),
      {  INT32_C(  1640780729), -INT32_C(  1020490763), -INT32_C(  1765900352),  INT32_C(  1023715526) },
      {  INT32_C(   963170506), -INT32_C(  1764618676),  INT32_C(  1547150243),  INT32_C(  1847372980) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    {    UINT8_MAX,
      {  INT32_C(  1945489384),  INT32_C(  2050275579),  INT32_C(   507562614), -INT32_C(  1628930610) },
      { -INT32_C(   185982491), -INT32_C(  1768455949),  INT32_C(  1682699191), -INT32_C(   144459505) },
      { -INT32_C(  2079711227), -INT32_C(  2071789568),  INT32_C(  1611346305),  INT32_C(  1627619329) } },
    { UINT8_C(136),
      {  INT32_C(   277048152),  INT32_C(  1770454687), -INT32_C(  1137204162), -INT32_C(  1365125747) },
      {  INT32_C(  1084396992),  INT32_C(   536361004),  INT32_C(  1009671299),  INT32_C(   415508159) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   272900658) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castps_si128(easysimd_mm_maskz_andnot_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_andnot_ps");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_castps_si128(easysimd_mm_maskz_andnot_ps(k, easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_andnot_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[2];
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[8] = {
    { { -INT64_C( 1928466103720646242), -INT64_C( 3929552797803232727) },
      UINT8_C(251),
      {  INT64_C( 6946384699866734110), -INT64_C( 5365360231245836667) },
      {  INT64_C( 6386919651707738563),  INT64_C(  583407368484975281) },
      {  INT64_C( 1765518261698429377),  INT64_C(  581110174951960624) } },
    { { -INT64_C( 1945508444521571445), -INT64_C( 4415395709292800583) },
      UINT8_C(181),
      {  INT64_C( 5787304598209969082),  INT64_C( 7043421486383263110) },
      { -INT64_C(  798643056117981434),  INT64_C( 7100406024258186872) },
      { -INT64_C( 6581443502561737724), -INT64_C( 4415395709292800583) } },
    { { -INT64_C(  279413981641059343), -INT64_C( 6726144068294780481) },
      UINT8_C( 78),
      { -INT64_C( 6086718537983995032), -INT64_C( 5243397349038662763) },
      { -INT64_C( 8950857899367920969),  INT64_C( 2132385932943232602) },
      { -INT64_C(  279413981641059343),  INT64_C(  613685891005416522) } },
    { {  INT64_C( 8862155964249031828), -INT64_C( 4852065622990434290) },
      UINT8_C( 11),
      { -INT64_C( 7179942260866955817),  INT64_C( 9163820544029730007) },
      { -INT64_C( 3864687188819739237), -INT64_C( 3337012320024965212) },
      {  INT64_C( 4757005654084827144), -INT64_C( 9182698077456282848) } },
    { {  INT64_C( 8465411180172862801), -INT64_C( 2986666952563503918) },
      UINT8_C(183),
      { -INT64_C( 3353216337534966542), -INT64_C( 5717148429293315125) },
      { -INT64_C( 3427587846107658130), -INT64_C(  233376797435396912) },
      {  INT64_C(    2251808610093068),  INT64_C( 5495060048596001808) } },
    { { -INT64_C( 5019184889626430715),  INT64_C( 3232662858979418720) },
      UINT8_C( 65),
      { -INT64_C( 5494931169692235734),  INT64_C( 9081149701025024032) },
      {  INT64_C(  589170994202905992),  INT64_C( 6932150569251789112) },
      {  INT64_C(  576780439606789504),  INT64_C( 3232662858979418720) } },
    { { -INT64_C( 5934886080182585158), -INT64_C( 6711009844344758060) },
      UINT8_C( 66),
      { -INT64_C(  672954320355365046),  INT64_C( 5017446275842087669) },
      { -INT64_C( 7585329277892914470),  INT64_C( 4693967936240491053) },
      { -INT64_C( 5934886080182585158),  INT64_C(    1216274618122248) } },
    { {  INT64_C( 8093913540725984193), -INT64_C( 1870971811960160267) },
      UINT8_C(139),
      { -INT64_C( 6735036999582757413), -INT64_C( 2802687513161274873) },
      {  INT64_C( 5394403933233117068), -INT64_C( 8961532516375290083) },
      {  INT64_C( 5211931696795234820),  INT64_C(  189192003824031000) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_mask_andnot_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_andnot_pd");
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
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_mask_andnot_pd(easysimd_mm_castsi128_pd(src), k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

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
test_easysimd_mm_maskz_andnot_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[2];
    int64_t b[2];
    int64_t r[2];
  } test_vec[8] = {
    { UINT8_C( 82),
      {  INT64_C( 6909130553220793829), -INT64_C( 1815175444950388114) },
      {  INT64_C(  693609001403951513), -INT64_C( 5779606191060982279) },
      {  INT64_C(                   0),  INT64_C(  648663766149432721) } },
    { UINT8_C(187),
      { -INT64_C( 3465312807885637039), -INT64_C( 5890073320336148875) },
      { -INT64_C( 8926858905856475149), -INT64_C( 5689738064150227305) },
      {  INT64_C(    5982455720838562),  INT64_C( 1227719495095943298) } },
    { UINT8_C(120),
      { -INT64_C( 3776370455647867802),  INT64_C(   86412277746516222) },
      {  INT64_C(  750031259100534371),  INT64_C( 2623659864339690995) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 72),
      {  INT64_C( 2224540579161458493), -INT64_C( 6279529359143329928) },
      { -INT64_C( 3937793059405911102), -INT64_C(  939936530224952432) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(243),
      { -INT64_C( 3318888276050193583), -INT64_C( 1300190907233769584) },
      {  INT64_C(  116472044337585127), -INT64_C( 8074770242888971695) },
      {  INT64_C(    3670191372241062),  INT64_C(  144155031641524289) } },
    { UINT8_C(122),
      {  INT64_C( 6936386720198625474), -INT64_C( 4335964335418215391) },
      { -INT64_C( 3986676887027326832),  INT64_C( 6652263716379601199) },
      {  INT64_C(                   0),  INT64_C( 2017630852632316174) } },
    { UINT8_C(158),
      { -INT64_C( 4206654176352661890),  INT64_C( 4591685206807514008) },
      {  INT64_C( 7499484863724852074), -INT64_C( 4990166249758332277) },
      {  INT64_C(                   0), -INT64_C( 9221386181399126013) } },
    { UINT8_C(108),
      {  INT64_C( 7054188248267831126), -INT64_C( 2879208886991123025) },
      { -INT64_C( 1383730975835951429),  INT64_C(  274125486138387714) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_castpd_si128(easysimd_mm_maskz_andnot_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_andnot_pd");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_castpd_si128(easysimd_mm_maskz_andnot_pd(k, easysimd_mm_castsi128_pd(a), easysimd_mm_castsi128_pd(b)));

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_andnot_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[8];
    uint8_t k;
    int32_t a[8];
    int32_t b[8];
    int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(  1458302921), -INT32_C(  1247502057),  INT32_C(  1955621765), -INT32_C(  1351237652),  INT32_C(   373661843),  INT32_C(   360661251), -INT32_C(   415191734),  INT32_C(   584609625) },
      UINT8_C( 94),
      {  INT32_C(  1685486019), -INT32_C(  1997919459),  INT32_C(  2071289532),  INT32_C(  1930306771), -INT32_C(  1418255255),  INT32_C(  1375046820),  INT32_C(  1001053388),  INT32_C(  2006568372) },
      {  INT32_C(  1675300678), -INT32_C(   135477957), -INT32_C(   143498972), -INT32_C(   294944379),  INT32_C(  1234821797),  INT32_C(   983273070),  INT32_C(   511001962),  INT32_C(  1486163474) },
      {  INT32_C(  1458302921),  INT32_C(  1996801058), -INT32_C(  2080235264), -INT32_C(  1939767036),  INT32_C(  1082704516),  INT32_C(   360661251),  INT32_C(    72679714),  INT32_C(   584609625) } },
    { {  INT32_C(  1505521950),  INT32_C(  1515235382), -INT32_C(  1907178999), -INT32_C(   377701053),  INT32_C(   221386399),  INT32_C(   256363940),  INT32_C(   623754259), -INT32_C(   394345526) },
      UINT8_C( 52),
      { -INT32_C(   496352966),  INT32_C(  1424737681), -INT32_C(   728270569),  INT32_C(   192119029), -INT32_C(  2135916365), -INT32_C(  2087469113), -INT32_C(  1353860628),  INT32_C(  1910715959) },
      {  INT32_C(   156454519),  INT32_C(   710753811), -INT32_C(  1392577353),  INT32_C(   683176309), -INT32_C(  1196922639),  INT32_C(   322649127),  INT32_C(   751012085),  INT32_C(   916301502) },
      {  INT32_C(  1505521950),  INT32_C(  1515235382),  INT32_C(   677938336), -INT32_C(   377701053),  INT32_C(   940075072),  INT32_C(   271056928),  INT32_C(   623754259), -INT32_C(   394345526) } },
    { {  INT32_C(   121630964), -INT32_C(   449733586),  INT32_C(    93400976), -INT32_C(  1859303008), -INT32_C(   666249551),  INT32_C(   132940818), -INT32_C(   885805299),  INT32_C(  1241632853) },
      UINT8_C(192),
      { -INT32_C(   588361408), -INT32_C(  1301425277), -INT32_C(  1370262940),  INT32_C(  1985995936), -INT32_C(  1299695570), -INT32_C(   742420700), -INT32_C(  1842836542), -INT32_C(   850234740) },
      {  INT32_C(  1202274500),  INT32_C(  2012812819),  INT32_C(   690310281),  INT32_C(  1604289841), -INT32_C(   518969411),  INT32_C(  2025116086), -INT32_C(   401941412),  INT32_C(   330652751) },
      {  INT32_C(   121630964), -INT32_C(   449733586),  INT32_C(    93400976), -INT32_C(  1859303008), -INT32_C(   666249551),  INT32_C(   132940818),  INT32_C(  1744983068),  INT32_C(   312806467) } },
    { { -INT32_C(  1353031780), -INT32_C(    31042699), -INT32_C(   785953632), -INT32_C(  1909405999), -INT32_C(  1552989715),  INT32_C(  1847272210),  INT32_C(  1314334207),  INT32_C(   492899457) },
      UINT8_C(107),
      {  INT32_C(   266456251),  INT32_C(  1068490739), -INT32_C(   854556665), -INT32_C(   239362384),  INT32_C(   805592589), -INT32_C(  1624280454), -INT32_C(   702447927), -INT32_C(  1690222881) },
      { -INT32_C(    22404597),  INT32_C(   155015426), -INT32_C(  1982444071), -INT32_C(   126184981),  INT32_C(  1781038832), -INT32_C(  1173792783), -INT32_C(  1248843051),  INT32_C(  1951453801) },
      { -INT32_C(   267771392),  INT32_C(     1048576), -INT32_C(   785953632),  INT32_C(   138412363), -INT32_C(  1552989715),  INT32_C(   536872321),  INT32_C(   563085332),  INT32_C(   492899457) } },
    { { -INT32_C(   160236812),  INT32_C(   754954067), -INT32_C(   407513348),  INT32_C(  1474244455), -INT32_C(  1614739538),  INT32_C(   878299998),  INT32_C(  1592388341), -INT32_C(  1328399940) },
      UINT8_C( 51),
      { -INT32_C(   209279164),  INT32_C(  2096083622), -INT32_C(  1763387801), -INT32_C(  1136378955), -INT32_C(   937696259),  INT32_C(   666718013),  INT32_C(  1910709304),  INT32_C(   866423790) },
      { -INT32_C(   517592518),  INT32_C(  1130174172), -INT32_C(  1562820116),  INT32_C(  2052988541),  INT32_C(  1044543745),  INT32_C(     6684616),  INT32_C(   175196187),  INT32_C(   373102044) },
      {  INT32_C(     2099258),  INT32_C(    51381336), -INT32_C(   407513348),  INT32_C(  1474244455),  INT32_C(   910170112),  INT32_C(     4239552),  INT32_C(  1592388341), -INT32_C(  1328399940) } },
    { {  INT32_C(   469197631),  INT32_C(  1717458297),  INT32_C(   319305878),  INT32_C(  1468884566), -INT32_C(  1466511392), -INT32_C(   357958705),  INT32_C(   552868420),  INT32_C(  1865822512) },
      UINT8_C(149),
      { -INT32_C(  2096198866),  INT32_C(   555316457), -INT32_C(   478728836), -INT32_C(  1966878790),  INT32_C(  1616472933),  INT32_C(   799294228),  INT32_C(  1767883832),  INT32_C(   704564987) },
      {  INT32_C(  1135349081), -INT32_C(    26950271), -INT32_C(  1377706766),  INT32_C(   272082091),  INT32_C(   611356687),  INT32_C(   189994451), -INT32_C(   730484007), -INT32_C(   620924032) },
      {  INT32_C(  1084228689),  INT32_C(  1717458297),  INT32_C(   209768578),  INT32_C(  1468884566),  INT32_C(    69242890), -INT32_C(   357958705),  INT32_C(   552868420), -INT32_C(   771673856) } },
    { {  INT32_C(    35498368),  INT32_C(  1627423087),  INT32_C(   135192925), -INT32_C(  1810348667), -INT32_C(  1447523883),  INT32_C(  2008288158),  INT32_C(  1045178813),  INT32_C(   488130973) },
      UINT8_C(243),
      { -INT32_C(  1235083467),  INT32_C(     1360671),  INT32_C(   394599634), -INT32_C(  1091757515), -INT32_C(   564357422),  INT32_C(  1956369226),  INT32_C(  1762777375),  INT32_C(   643575537) },
      {  INT32_C(  1843248718),  INT32_C(  1399779713),  INT32_C(  1114370829), -INT32_C(   536848371),  INT32_C(   951999726),  INT32_C(  1336695088),  INT32_C(   616086835),  INT32_C(   977999084) },
      {  INT32_C(  1235066954),  INT32_C(  1399468160),  INT32_C(   135192925), -INT32_C(  1810348667),  INT32_C(   547506220),  INT32_C(   186910768),  INT32_C(    78128160),  INT32_C(   402853900) } },
    { {  INT32_C(  1420241106),  INT32_C(   648484121),  INT32_C(   375984649),  INT32_C(  1492543850), -INT32_C(   158223162),  INT32_C(  1095056654), -INT32_C(   412745989), -INT32_C(   450777070) },
      UINT8_C(216),
      { -INT32_C(   554616376), -INT32_C(   202958624), -INT32_C(   362939007), -INT32_C(  1464813836), -INT32_C(  2068404665), -INT32_C(   360712212),  INT32_C(   234645085),  INT32_C(  1357308295) },
      { -INT32_C(    80816358),  INT32_C(  1894651375),  INT32_C(   106646290),  INT32_C(  1219431169), -INT32_C(  1630706254), -INT32_C(  1165472675),  INT32_C(   986154163), -INT32_C(  2138395290) },
      {  INT32_C(  1420241106),  INT32_C(   648484121),  INT32_C(   375984649),  INT32_C(  1074727169),  INT32_C(   441008560),  INT32_C(  1095056654),  INT32_C(   839090338), -INT32_C(  2146956192) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_andnot_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_andnot_epi32");
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
    easysimd__m256i r = easysimd_mm256_mask_andnot_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_andnot_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[8];
    int32_t b[8];
    int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 66),
      {  INT32_C(  1297072714),  INT32_C(  1846162264),  INT32_C(  1476232615), -INT32_C(   116646577), -INT32_C(  2001465088), -INT32_C(  1604090738), -INT32_C(  1290344310), -INT32_C(  1896437692) },
      { -INT32_C(  1025817238),  INT32_C(   523298424), -INT32_C(  1149882773),  INT32_C(  1270121291),  INT32_C(   601122964),  INT32_C(  2009282540),  INT32_C(  1680529952), -INT32_C(   319610750) },
      {  INT32_C(           0),  INT32_C(   288408608),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1143476768),  INT32_C(           0) } },
    { UINT8_C(101),
      { -INT32_C(  1260474674),  INT32_C(   203488734), -INT32_C(   145171596), -INT32_C(   141843569), -INT32_C(  1343967625),  INT32_C(  1288657778), -INT32_C(  1496435835), -INT32_C(   183780826) },
      {  INT32_C(  1185540456),  INT32_C(  1532217831),  INT32_C(   861055908), -INT32_C(   970203826), -INT32_C(    25882740), -INT32_C(   280345494), -INT32_C(  1634396041),  INT32_C(   982753746) },
      {  INT32_C(  1109475616),  INT32_C(           0),  INT32_C(      139392),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1560280056),  INT32_C(   403769458),  INT32_C(           0) } },
    { UINT8_C(138),
      {  INT32_C(   108101692),  INT32_C(  2125122771), -INT32_C(    53617122),  INT32_C(   411603721),  INT32_C(  1283622408), -INT32_C(   389844528), -INT32_C(  1464180473),  INT32_C(   825423092) },
      {  INT32_C(  1211606132), -INT32_C(  1882791568), -INT32_C(   930376769),  INT32_C(   786436902),  INT32_C(  1769628313), -INT32_C(   615432492), -INT32_C(  1803351137),  INT32_C(  1942337023) },
      {  INT32_C(           0), -INT32_C(  2126241504),  INT32_C(           0),  INT32_C(   643825702),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1120207115) } },
    { UINT8_C( 89),
      { -INT32_C(   573916164),  INT32_C(   362633602), -INT32_C(   147036700), -INT32_C(  1466930618),  INT32_C(   595392997),  INT32_C(  1438799690), -INT32_C(  1873520934),  INT32_C(   401196827) },
      {  INT32_C(    99988355), -INT32_C(   266694132),  INT32_C(  1038571255), -INT32_C(  1494845503), -INT32_C(  1161207184), -INT32_C(  1810920518), -INT32_C(    47946782), -INT32_C(  1374351829) },
      {  INT32_C(     3473411),  INT32_C(           0),  INT32_C(           0),  INT32_C(   107353473), -INT32_C(  1736375792),  INT32_C(           0),  INT32_C(  1830822176),  INT32_C(           0) } },
    { UINT8_C(193),
      { -INT32_C(  1664175350),  INT32_C(   613662413),  INT32_C(   501600678),  INT32_C(   428772279), -INT32_C(   539801516), -INT32_C(  1144952744), -INT32_C(  1696153716), -INT32_C(   564357932) },
      {  INT32_C(   360327751), -INT32_C(  1908863512), -INT32_C(  1783882018), -INT32_C(    38847831), -INT32_C(   623082878),  INT32_C(  2006359786),  INT32_C(   823229533),  INT32_C(  1477405969) },
      {  INT32_C(    19925061),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   554778705),  INT32_C(      223489) } },
    { UINT8_C(151),
      { -INT32_C(  1769968247), -INT32_C(   998961498), -INT32_C(   194180422),  INT32_C(  1014459065), -INT32_C(   450473657),  INT32_C(  1682087399),  INT32_C(   477459375),  INT32_C(   196398466) },
      { -INT32_C(   509529029), -INT32_C(    39446974), -INT32_C(   621735135), -INT32_C(   988387202), -INT32_C(  1582678598), -INT32_C(  1996100390), -INT32_C(   509183393), -INT32_C(  2081662392) },
      {  INT32_C(  1629556786),  INT32_C(   964825152),  INT32_C(   177213697),  INT32_C(           0),  INT32_C(     8924344),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2142760376) } },
    { UINT8_C(142),
      { -INT32_C(  1529846643),  INT32_C(   516279562),  INT32_C(   664575935), -INT32_C(   203333194), -INT32_C(   137526517),  INT32_C(    55990152),  INT32_C(  1464547325), -INT32_C(  1327116765) },
      {  INT32_C(  1028961586),  INT32_C(  1113266563),  INT32_C(  1869215673),  INT32_C(  1667386200),  INT32_C(  1448751054), -INT32_C(  2091274106),  INT32_C(   182101223), -INT32_C(  1514487950) },
      {  INT32_C(           0),  INT32_C(  1075449985),  INT32_C(  1214341120),  INT32_C(      133704),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(    85602640) } },
    { UINT8_C(117),
      {  INT32_C(   687399439),  INT32_C(   887175741), -INT32_C(   292794205), -INT32_C(   490934350), -INT32_C(   110619831),  INT32_C(   299953260), -INT32_C(  2038175034), -INT32_C(  1275385692) },
      {  INT32_C(  1205662474), -INT32_C(   797197011), -INT32_C(  1061222642),  INT32_C(  1067613174), -INT32_C(    80213106), -INT32_C(  1123280649), -INT32_C(  1488744702), -INT32_C(  1034273096) },
      {  INT32_C(  1191448832),  INT32_C(           0),  INT32_C(     3344140),  INT32_C(           0),  INT32_C(    34605190), -INT32_C(  1408495469),  INT32_C(   557843712),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_andnot_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_andnot_epi32");
    easysimd_test_x86_assert_equal_i32x8(easysimd_mm256_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_andnot_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_andnot_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[4];
    uint8_t k;
    int64_t a[4];
    int64_t b[4];
    int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 2241268908403476347), -INT64_C(  144232510628937547), -INT64_C( 3813947346956595973),  INT64_C(  904819484878363048) },
      UINT8_C( 73),
      { -INT64_C( 6545711142800031734), -INT64_C( 6166009675621736483), -INT64_C( 5668467498991565430), -INT64_C(  257473796909800556) },
      {  INT64_C( 6327443651944219844),  INT64_C( 4179120205421265015), -INT64_C( 2052167741625417219), -INT64_C( 4139523832894931553) },
      {  INT64_C( 5964745139378783428), -INT64_C(  144232510628937547), -INT64_C( 3813947346956595973),  INT64_C(  180197867892284427) } },
    { {  INT64_C(  922041197360381433), -INT64_C( 3861730246090713728),  INT64_C( 3050103024666955297), -INT64_C( 3771402042833412895) },
      UINT8_C( 70),
      { -INT64_C( 6856476396750159975), -INT64_C( 4699163783477786808), -INT64_C( 1451440492191925546),  INT64_C(   53349848081874478) },
      { -INT64_C( 4074577337786984248), -INT64_C( 1196206968169045703),  INT64_C( 5759475144538566515), -INT64_C(  221939493324280957) },
      {  INT64_C(  922041197360381433),  INT64_C( 4694448526393008177),  INT64_C(  298508748297700641), -INT64_C( 3771402042833412895) } },
    { { -INT64_C( 5881344366469954533),  INT64_C( 8948015882053341221),  INT64_C( 6835738400150918764),  INT64_C( 4847899767730643990) },
      UINT8_C(119),
      {  INT64_C( 1117063404015443223),  INT64_C(  224217825749286898),  INT64_C( 3212258991163746834),  INT64_C( 4408907653264433028) },
      { -INT64_C( 6142319760213795724), -INT64_C( 9023112006494135337),  INT64_C( 7337934864794035489),  INT64_C( 3735258426253322611) },
      { -INT64_C( 6898951126736961440), -INT64_C( 9168375550714953723),  INT64_C( 4702180642215953697),  INT64_C( 4847899767730643990) } },
    { {  INT64_C( 6831021132668628585), -INT64_C( 4650554175591930236),  INT64_C( 9067221639320073614), -INT64_C( 6243417077917945241) },
      UINT8_C( 92),
      { -INT64_C( 4797105911654790592),  INT64_C( 5538330049970332004),  INT64_C( 1380303948431641474), -INT64_C( 1714468859813933590) },
      { -INT64_C( 4892809871414454330),  INT64_C( 1135723551295960224), -INT64_C( 2997718482083178447),  INT64_C( 8778609361056347972) },
      {  INT64_C( 6831021132668628585), -INT64_C( 4650554175591930236), -INT64_C( 4305400784863293391),  INT64_C( 1279866788421349892) } },
    { { -INT64_C( 8930937247107799559),  INT64_C( 3458973416482805859),  INT64_C(  763614977675415509),  INT64_C( 2507037321890973342) },
      UINT8_C(110),
      { -INT64_C( 7209464192933137248), -INT64_C( 5592403237876866352),  INT64_C(  275144797075285068), -INT64_C( 8237559887523208329) },
      { -INT64_C( 6349777686153658173), -INT64_C( 7960790440155513871), -INT64_C(  613731449974751812), -INT64_C( 5101198530878920775) },
      { -INT64_C( 8930937247107799559),  INT64_C(  109235403560851745), -INT64_C(  852846900221257296),  INT64_C( 3463444115404063880) } },
    { {  INT64_C( 5793001722519441638), -INT64_C( 1426610630619326372),  INT64_C( 5052441253818726310),  INT64_C( 2393136787098522759) },
      UINT8_C(113),
      { -INT64_C( 2706064957873529291),  INT64_C( 4509218436413002564),  INT64_C( 3153103400448276827), -INT64_C( 6011876185655131905) },
      { -INT64_C( 4142972446455995804), -INT64_C( 2464516270013052089), -INT64_C( 6340264902391195904),  INT64_C( 5925992665383986970) },
      {  INT64_C(  324563190820249664), -INT64_C( 1426610630619326372),  INT64_C( 5052441253818726310),  INT64_C( 2393136787098522759) } },
    { {  INT64_C( 8792568208029253564),  INT64_C(   28381691618310149),  INT64_C( 8196227339912707180), -INT64_C( 7169378899719977408) },
      UINT8_C( 62),
      { -INT64_C( 7423642059825620102),  INT64_C( 7076730502179750388), -INT64_C( 8104181314384999825), -INT64_C( 5572346677463894227) },
      { -INT64_C(  189257799268164377), -INT64_C(  114565198785349856),  INT64_C( 8709345176747123013),  INT64_C( 4571312332615273606) },
      {  INT64_C( 8792568208029253564), -INT64_C( 7185384310763522560),  INT64_C( 8094592796927074560),  INT64_C(  959425327640103042) } },
    { { -INT64_C( 2976203473126287700), -INT64_C( 5121924886501230499), -INT64_C( 8707178808156474779),  INT64_C( 5388756646644485680) },
      UINT8_C(203),
      {  INT64_C( 2995552021244870606), -INT64_C( 7144022468148297934), -INT64_C( 4364229153023751077), -INT64_C( 8148793213413361815) },
      { -INT64_C(  898937890447136085), -INT64_C( 3181153808333331757),  INT64_C(  453035746791684305),  INT64_C(   55747804565645944) },
      { -INT64_C( 3313523149747650527),  INT64_C( 4827858890876059841), -INT64_C( 8707178808156474779),  INT64_C(    1693523321817104) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_andnot_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_andnot_epi64");
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
    easysimd__m256i r = easysimd_mm256_mask_andnot_epi64(src, k, a, b);

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
test_easysimd_mm256_maskz_andnot_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[4];
    int64_t b[4];
    int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 49),
      { -INT64_C( 8286008112031331786),  INT64_C( 8140471166801527534), -INT64_C( 8208332592858532869),  INT64_C( 5681300931401399656) },
      {  INT64_C( 5991270082423174675),  INT64_C( 7484056158068386055), -INT64_C( 3515742960203643992), -INT64_C( 7442674999309730737) },
      {  INT64_C( 5919208066443654145),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(132),
      { -INT64_C(   52849855006363400), -INT64_C( 3506102283204316934), -INT64_C( 5751371374959067570),  INT64_C( 4994154770789144444) },
      {  INT64_C( 8093749691847686397), -INT64_C( 2390998466762326666), -INT64_C( 8477202701321631626),  INT64_C( 6419778313372602244) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(  743350167851761712),  INT64_C(                   0) } },
    { UINT8_C(219),
      { -INT64_C( 4767409852948304487), -INT64_C( 5562179004136564255),  INT64_C( 7087340712299100649), -INT64_C( 2514108914346349410) },
      {  INT64_C( 1189252868184305092), -INT64_C( 2141117503199654164), -INT64_C(  871988224124777807), -INT64_C( 9196261672800181184) },
      {  INT64_C(     301268403503172),  INT64_C( 4611703960756028940),  INT64_C(                   0),  INT64_C(   27092758963454016) } },
    { UINT8_C(189),
      { -INT64_C( 7809809352278457187), -INT64_C( 8625116221273263004),  INT64_C( 6591809390483656998), -INT64_C(  355617939185468157) },
      {  INT64_C( 6643736302714682587),  INT64_C( 6018012165405458302),  INT64_C( 3758824745216074007), -INT64_C( 5521808037959483131) },
      {  INT64_C( 5485947510839359554),  INT64_C(                   0),  INT64_C( 2594073388670590993),  INT64_C(   21964394577526788) } },
    { UINT8_C(148),
      {  INT64_C( 5031619772849804043), -INT64_C( 5386931514540633888),  INT64_C( 8673961707382901843),  INT64_C( 3228241048636605345) },
      {  INT64_C( 2316069845474564440), -INT64_C( 5751232666422307291), -INT64_C( 3440632262364031600), -INT64_C( 1333954353332317906) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 9223265347651956352),  INT64_C(                   0) } },
    { UINT8_C( 48),
      { -INT64_C( 8757370489037271811),  INT64_C( 3885642345720574424), -INT64_C( 8847544720211790690),  INT64_C(   19712053423539774) },
      {  INT64_C( 2398952169703389140),  INT64_C( 6778033716229647803), -INT64_C( 1759044159757913059), -INT64_C( 8275571327751048923) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(251),
      { -INT64_C( 3891279835485154950),  INT64_C( 6803224805107161826), -INT64_C( 4461962144530400689), -INT64_C( 2582523693540623638) },
      { -INT64_C( 5423143385384850087),  INT64_C( 9097667277163022329),  INT64_C( 2285153730647836643), -INT64_C( 7856399562287795782) },
      {  INT64_C( 3747023755107380225),  INT64_C( 2305843033380619545),  INT64_C(                   0),  INT64_C(  202791833246630160) } },
    { UINT8_C(227),
      { -INT64_C( 8417111961842121192), -INT64_C( 7227089565075762327), -INT64_C( 8079756394320844635),  INT64_C(  763178660324244029) },
      {  INT64_C( 2283450038199864801),  INT64_C( 5148922392834304644),  INT64_C( 8231535809017955406),  INT64_C( 4630471583535072325) },
      {  INT64_C( 1477199034471959009),  INT64_C( 4918077724058282116),  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_andnot_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_andnot_epi64");
    easysimd_test_x86_assert_equal_i64x4(easysimd_mm256_loadu_epi64(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_andnot_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_andnot_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float32 src[8];
    uint8_t k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -386.56), EASYSIMD_FLOAT32_C(   933.17), EASYSIMD_FLOAT32_C(   819.42), EASYSIMD_FLOAT32_C(   393.31),
        EASYSIMD_FLOAT32_C(   524.73), EASYSIMD_FLOAT32_C(   -48.96), EASYSIMD_FLOAT32_C(  -393.55), EASYSIMD_FLOAT32_C(  -268.17) },
         UINT8_MAX,
      { EASYSIMD_FLOAT32_C(   812.09), EASYSIMD_FLOAT32_C(  -950.07), EASYSIMD_FLOAT32_C(   568.73), EASYSIMD_FLOAT32_C(    -0.41),
        EASYSIMD_FLOAT32_C(  -165.86), EASYSIMD_FLOAT32_C(   353.81), EASYSIMD_FLOAT32_C(  -254.81), EASYSIMD_FLOAT32_C(  -659.77) },
      { EASYSIMD_FLOAT32_C(  -781.02), EASYSIMD_FLOAT32_C(  -315.06), EASYSIMD_FLOAT32_C(  -489.32), EASYSIMD_FLOAT32_C(   573.51),
        EASYSIMD_FLOAT32_C(  -939.51), EASYSIMD_FLOAT32_C(  -145.77), EASYSIMD_FLOAT32_C(   235.12), EASYSIMD_FLOAT32_C(   331.60) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     2.22),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(    53.66), EASYSIMD_FLOAT32_C(   678.06), EASYSIMD_FLOAT32_C(   -68.20), EASYSIMD_FLOAT32_C(  -647.74),
        EASYSIMD_FLOAT32_C(  -516.81), EASYSIMD_FLOAT32_C(    60.71), EASYSIMD_FLOAT32_C(   -34.29), EASYSIMD_FLOAT32_C(  -583.64) },
      UINT8_C( 13),
      { EASYSIMD_FLOAT32_C(  -640.98), EASYSIMD_FLOAT32_C(   941.09), EASYSIMD_FLOAT32_C(   831.17), EASYSIMD_FLOAT32_C(   -34.53),
        EASYSIMD_FLOAT32_C(  -327.08), EASYSIMD_FLOAT32_C(   430.41), EASYSIMD_FLOAT32_C(  -222.45), EASYSIMD_FLOAT32_C(  -277.14) },
      { EASYSIMD_FLOAT32_C(    -0.85), EASYSIMD_FLOAT32_C(   777.14), EASYSIMD_FLOAT32_C(   557.00), EASYSIMD_FLOAT32_C(  -647.05),
        EASYSIMD_FLOAT32_C(  -477.67), EASYSIMD_FLOAT32_C(   897.23), EASYSIMD_FLOAT32_C(  -428.07), EASYSIMD_FLOAT32_C(   207.27) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   678.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -516.81), EASYSIMD_FLOAT32_C(    60.71), EASYSIMD_FLOAT32_C(   -34.29), EASYSIMD_FLOAT32_C(  -583.64) } },
    { { EASYSIMD_FLOAT32_C(  -592.09), EASYSIMD_FLOAT32_C(  -854.56), EASYSIMD_FLOAT32_C(   267.76), EASYSIMD_FLOAT32_C(   262.13),
        EASYSIMD_FLOAT32_C(   380.56), EASYSIMD_FLOAT32_C(  -400.64), EASYSIMD_FLOAT32_C(  -684.21), EASYSIMD_FLOAT32_C(    58.62) },
      UINT8_C(  0),
      { EASYSIMD_FLOAT32_C(  -331.95), EASYSIMD_FLOAT32_C(   541.81), EASYSIMD_FLOAT32_C(  -408.13), EASYSIMD_FLOAT32_C(   633.76),
        EASYSIMD_FLOAT32_C(   958.17), EASYSIMD_FLOAT32_C(   472.00), EASYSIMD_FLOAT32_C(   992.78), EASYSIMD_FLOAT32_C(   899.26) },
      { EASYSIMD_FLOAT32_C(   303.17), EASYSIMD_FLOAT32_C(   -41.75), EASYSIMD_FLOAT32_C(  -427.82), EASYSIMD_FLOAT32_C(  -266.42),
        EASYSIMD_FLOAT32_C(   735.80), EASYSIMD_FLOAT32_C(   295.04), EASYSIMD_FLOAT32_C(   732.73), EASYSIMD_FLOAT32_C(   512.94) },
      { EASYSIMD_FLOAT32_C(  -592.09), EASYSIMD_FLOAT32_C(  -854.56), EASYSIMD_FLOAT32_C(   267.76), EASYSIMD_FLOAT32_C(   262.13),
        EASYSIMD_FLOAT32_C(   380.56), EASYSIMD_FLOAT32_C(  -400.64), EASYSIMD_FLOAT32_C(  -684.21), EASYSIMD_FLOAT32_C(    58.62) } },
    { { EASYSIMD_FLOAT32_C(  -147.96), EASYSIMD_FLOAT32_C(  -914.32), EASYSIMD_FLOAT32_C(  -964.73), EASYSIMD_FLOAT32_C(  -250.74),
        EASYSIMD_FLOAT32_C(  -342.39), EASYSIMD_FLOAT32_C(   242.54), EASYSIMD_FLOAT32_C(   157.17), EASYSIMD_FLOAT32_C(  -196.95) },
      UINT8_C( 20),
      { EASYSIMD_FLOAT32_C(  -580.69), EASYSIMD_FLOAT32_C(  -816.39), EASYSIMD_FLOAT32_C(   109.66), EASYSIMD_FLOAT32_C(  -264.90),
        EASYSIMD_FLOAT32_C(   242.23), EASYSIMD_FLOAT32_C(  -359.18), EASYSIMD_FLOAT32_C(   403.15), EASYSIMD_FLOAT32_C(  -215.97) },
      { EASYSIMD_FLOAT32_C(   232.69), EASYSIMD_FLOAT32_C(    36.91), EASYSIMD_FLOAT32_C(  -257.80), EASYSIMD_FLOAT32_C(  -295.30),
        EASYSIMD_FLOAT32_C(    29.69), EASYSIMD_FLOAT32_C(  -358.54), EASYSIMD_FLOAT32_C(  -992.13), EASYSIMD_FLOAT32_C(   987.94) },
      { EASYSIMD_FLOAT32_C(  -147.96), EASYSIMD_FLOAT32_C(  -914.32), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -250.74),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   242.54), EASYSIMD_FLOAT32_C(   157.17), EASYSIMD_FLOAT32_C(  -196.95) } },
    { { EASYSIMD_FLOAT32_C(   213.65), EASYSIMD_FLOAT32_C(  -258.55), EASYSIMD_FLOAT32_C(   723.74), EASYSIMD_FLOAT32_C(  -491.31),
        EASYSIMD_FLOAT32_C(  -525.81), EASYSIMD_FLOAT32_C(   236.68), EASYSIMD_FLOAT32_C(   360.72), EASYSIMD_FLOAT32_C(  -440.13) },
      UINT8_C(164),
      { EASYSIMD_FLOAT32_C(  -890.01), EASYSIMD_FLOAT32_C(   217.48), EASYSIMD_FLOAT32_C(  -485.51), EASYSIMD_FLOAT32_C(   267.16),
        EASYSIMD_FLOAT32_C(  -979.46), EASYSIMD_FLOAT32_C(    24.79), EASYSIMD_FLOAT32_C(   686.47), EASYSIMD_FLOAT32_C(  -795.85) },
      { EASYSIMD_FLOAT32_C(  -865.55), EASYSIMD_FLOAT32_C(  -578.43), EASYSIMD_FLOAT32_C(   446.38), EASYSIMD_FLOAT32_C(  -224.73),
        EASYSIMD_FLOAT32_C(   824.72), EASYSIMD_FLOAT32_C(  -769.59), EASYSIMD_FLOAT32_C(  -992.03), EASYSIMD_FLOAT32_C(  -138.36) },
      { EASYSIMD_FLOAT32_C(   213.65), EASYSIMD_FLOAT32_C(  -258.55), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -491.31),
        EASYSIMD_FLOAT32_C(  -525.81), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(   360.72), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   -27.39), EASYSIMD_FLOAT32_C(  -287.34), EASYSIMD_FLOAT32_C(   891.33), EASYSIMD_FLOAT32_C(   614.08),
        EASYSIMD_FLOAT32_C(  -279.47), EASYSIMD_FLOAT32_C(   879.27), EASYSIMD_FLOAT32_C(  -172.27), EASYSIMD_FLOAT32_C(   461.99) },
      UINT8_C(136),
      { EASYSIMD_FLOAT32_C(   336.41), EASYSIMD_FLOAT32_C(   936.17), EASYSIMD_FLOAT32_C(  -160.31), EASYSIMD_FLOAT32_C(  -302.86),
        EASYSIMD_FLOAT32_C(  -503.96), EASYSIMD_FLOAT32_C(  -888.36), EASYSIMD_FLOAT32_C(  -192.88), EASYSIMD_FLOAT32_C(   713.53) },
      { EASYSIMD_FLOAT32_C(  -373.87), EASYSIMD_FLOAT32_C(  -925.72), EASYSIMD_FLOAT32_C(   734.06), EASYSIMD_FLOAT32_C(   650.92),
        EASYSIMD_FLOAT32_C(   760.75), EASYSIMD_FLOAT32_C(   938.21), EASYSIMD_FLOAT32_C(   785.37), EASYSIMD_FLOAT32_C(  -817.68) },
      { EASYSIMD_FLOAT32_C(   -27.39), EASYSIMD_FLOAT32_C(  -287.34), EASYSIMD_FLOAT32_C(   891.33), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -279.47), EASYSIMD_FLOAT32_C(   879.27), EASYSIMD_FLOAT32_C(  -172.27), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(   384.59), EASYSIMD_FLOAT32_C(  -439.36), EASYSIMD_FLOAT32_C(  -992.96), EASYSIMD_FLOAT32_C(   615.00),
        EASYSIMD_FLOAT32_C(  -431.39), EASYSIMD_FLOAT32_C(  -131.32), EASYSIMD_FLOAT32_C(  -412.39), EASYSIMD_FLOAT32_C(   281.27) },
      UINT8_C(226),
      { EASYSIMD_FLOAT32_C(  -798.32), EASYSIMD_FLOAT32_C(  -998.20), EASYSIMD_FLOAT32_C(  -360.72), EASYSIMD_FLOAT32_C(    29.41),
        EASYSIMD_FLOAT32_C(   463.79), EASYSIMD_FLOAT32_C(  -757.71), EASYSIMD_FLOAT32_C(  -634.18), EASYSIMD_FLOAT32_C(   399.96) },
      { EASYSIMD_FLOAT32_C(    81.98), EASYSIMD_FLOAT32_C(    62.96), EASYSIMD_FLOAT32_C(   896.00), EASYSIMD_FLOAT32_C(   193.62),
        EASYSIMD_FLOAT32_C(   870.08), EASYSIMD_FLOAT32_C(   609.53), EASYSIMD_FLOAT32_C(   819.74), EASYSIMD_FLOAT32_C(   944.37) },
      { EASYSIMD_FLOAT32_C(   384.59), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -992.96), EASYSIMD_FLOAT32_C(   615.00),
        EASYSIMD_FLOAT32_C(  -431.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   343.59), EASYSIMD_FLOAT32_C(   470.66), EASYSIMD_FLOAT32_C(   705.12), EASYSIMD_FLOAT32_C(   281.80),
        EASYSIMD_FLOAT32_C(   256.02), EASYSIMD_FLOAT32_C(   887.44), EASYSIMD_FLOAT32_C(  -333.61), EASYSIMD_FLOAT32_C(   816.67) },
      UINT8_C(102),
      { EASYSIMD_FLOAT32_C(  -718.61), EASYSIMD_FLOAT32_C(  -614.73), EASYSIMD_FLOAT32_C(  -236.84), EASYSIMD_FLOAT32_C(  -131.01),
        EASYSIMD_FLOAT32_C(   666.54), EASYSIMD_FLOAT32_C(   523.17), EASYSIMD_FLOAT32_C(    70.68), EASYSIMD_FLOAT32_C(   668.34) },
      { EASYSIMD_FLOAT32_C(  -837.56), EASYSIMD_FLOAT32_C(  -899.91), EASYSIMD_FLOAT32_C(   132.13), EASYSIMD_FLOAT32_C(  -595.27),
        EASYSIMD_FLOAT32_C(  -534.09), EASYSIMD_FLOAT32_C(  -467.91), EASYSIMD_FLOAT32_C(   486.71), EASYSIMD_FLOAT32_C(   528.87) },
      { EASYSIMD_FLOAT32_C(   343.59), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   281.80),
        EASYSIMD_FLOAT32_C(   256.02), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   816.67) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_andnot_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_andnot_ps");
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
    easysimd__m256 r = easysimd_mm256_mask_andnot_ps(src, k, a, b);

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
test_easysimd_mm256_maskz_andnot_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C( 63),
      { EASYSIMD_FLOAT32_C(   664.94), EASYSIMD_FLOAT32_C(   108.16), EASYSIMD_FLOAT32_C(    74.80), EASYSIMD_FLOAT32_C(   294.13),
        EASYSIMD_FLOAT32_C(   670.86), EASYSIMD_FLOAT32_C(   188.25), EASYSIMD_FLOAT32_C(    64.25), EASYSIMD_FLOAT32_C(    19.90) },
      { EASYSIMD_FLOAT32_C(  -877.54), EASYSIMD_FLOAT32_C(   666.88), EASYSIMD_FLOAT32_C(   951.40), EASYSIMD_FLOAT32_C(  -457.91),
        EASYSIMD_FLOAT32_C(  -604.57), EASYSIMD_FLOAT32_C(    64.76), EASYSIMD_FLOAT32_C(  -109.82), EASYSIMD_FLOAT32_C(   753.18) },
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 87),
      { EASYSIMD_FLOAT32_C(   513.71), EASYSIMD_FLOAT32_C(  -171.34), EASYSIMD_FLOAT32_C(   224.25), EASYSIMD_FLOAT32_C(  -247.75),
        EASYSIMD_FLOAT32_C(  -487.00), EASYSIMD_FLOAT32_C(    88.51), EASYSIMD_FLOAT32_C(   771.54), EASYSIMD_FLOAT32_C(  -150.27) },
      { EASYSIMD_FLOAT32_C(   253.42), EASYSIMD_FLOAT32_C(   830.69), EASYSIMD_FLOAT32_C(   -87.12), EASYSIMD_FLOAT32_C(  -608.17),
        EASYSIMD_FLOAT32_C(    32.76), EASYSIMD_FLOAT32_C(  -322.90), EASYSIMD_FLOAT32_C(  -943.23), EASYSIMD_FLOAT32_C(  -859.07) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 57),
      { EASYSIMD_FLOAT32_C(   350.90), EASYSIMD_FLOAT32_C(   811.78), EASYSIMD_FLOAT32_C(   -59.85), EASYSIMD_FLOAT32_C(  -584.85),
        EASYSIMD_FLOAT32_C(  -168.32), EASYSIMD_FLOAT32_C(    62.61), EASYSIMD_FLOAT32_C(  -917.96), EASYSIMD_FLOAT32_C(  -216.92) },
      { EASYSIMD_FLOAT32_C(   604.70), EASYSIMD_FLOAT32_C(  -522.54), EASYSIMD_FLOAT32_C(   847.84), EASYSIMD_FLOAT32_C(  -505.12),
        EASYSIMD_FLOAT32_C(  -769.36), EASYSIMD_FLOAT32_C(   340.93), EASYSIMD_FLOAT32_C(  -991.42), EASYSIMD_FLOAT32_C(    59.31) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(160),
      { EASYSIMD_FLOAT32_C(  -239.17), EASYSIMD_FLOAT32_C(   572.31), EASYSIMD_FLOAT32_C(   653.70), EASYSIMD_FLOAT32_C(  -467.63),
        EASYSIMD_FLOAT32_C(  -577.97), EASYSIMD_FLOAT32_C(   -92.88), EASYSIMD_FLOAT32_C(  -636.94), EASYSIMD_FLOAT32_C(   334.92) },
      { EASYSIMD_FLOAT32_C(   298.95), EASYSIMD_FLOAT32_C(   395.83), EASYSIMD_FLOAT32_C(  -987.98), EASYSIMD_FLOAT32_C(   355.73),
        EASYSIMD_FLOAT32_C(   536.75), EASYSIMD_FLOAT32_C(   763.92), EASYSIMD_FLOAT32_C(  -293.37), EASYSIMD_FLOAT32_C(   348.53) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(118),
      { EASYSIMD_FLOAT32_C(   121.78), EASYSIMD_FLOAT32_C(  -819.79), EASYSIMD_FLOAT32_C(   766.67), EASYSIMD_FLOAT32_C(   203.82),
        EASYSIMD_FLOAT32_C(   -36.71), EASYSIMD_FLOAT32_C(   371.37), EASYSIMD_FLOAT32_C(   681.28), EASYSIMD_FLOAT32_C(  -188.87) },
      { EASYSIMD_FLOAT32_C(   866.25), EASYSIMD_FLOAT32_C(   911.92), EASYSIMD_FLOAT32_C(  -847.94), EASYSIMD_FLOAT32_C(   874.83),
        EASYSIMD_FLOAT32_C(   -28.77), EASYSIMD_FLOAT32_C(  -282.75), EASYSIMD_FLOAT32_C(  -364.33), EASYSIMD_FLOAT32_C(  -456.46) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(106),
      { EASYSIMD_FLOAT32_C(   168.04), EASYSIMD_FLOAT32_C(   -34.43), EASYSIMD_FLOAT32_C(   278.06), EASYSIMD_FLOAT32_C(   531.10),
        EASYSIMD_FLOAT32_C(  -699.51), EASYSIMD_FLOAT32_C(  -422.99), EASYSIMD_FLOAT32_C(   -73.07), EASYSIMD_FLOAT32_C(  -687.49) },
      { EASYSIMD_FLOAT32_C(   932.74), EASYSIMD_FLOAT32_C(  -536.32), EASYSIMD_FLOAT32_C(  -923.57), EASYSIMD_FLOAT32_C(  -360.64),
        EASYSIMD_FLOAT32_C(   812.21), EASYSIMD_FLOAT32_C(  -219.51), EASYSIMD_FLOAT32_C(   761.14), EASYSIMD_FLOAT32_C(   992.43) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(205),
      { EASYSIMD_FLOAT32_C(   -35.04), EASYSIMD_FLOAT32_C(   -44.29), EASYSIMD_FLOAT32_C(   918.53), EASYSIMD_FLOAT32_C(  -353.77),
        EASYSIMD_FLOAT32_C(   766.85), EASYSIMD_FLOAT32_C(   784.78), EASYSIMD_FLOAT32_C(  -441.84), EASYSIMD_FLOAT32_C(   918.91) },
      { EASYSIMD_FLOAT32_C(   659.61), EASYSIMD_FLOAT32_C(   529.39), EASYSIMD_FLOAT32_C(  -363.84), EASYSIMD_FLOAT32_C(  -704.72),
        EASYSIMD_FLOAT32_C(  -927.07), EASYSIMD_FLOAT32_C(     7.10), EASYSIMD_FLOAT32_C(   463.32), EASYSIMD_FLOAT32_C(    38.50) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 11),
      { EASYSIMD_FLOAT32_C(    -5.58), EASYSIMD_FLOAT32_C(   338.99), EASYSIMD_FLOAT32_C(  -137.83), EASYSIMD_FLOAT32_C(   921.35),
        EASYSIMD_FLOAT32_C(   651.49), EASYSIMD_FLOAT32_C(  -205.09), EASYSIMD_FLOAT32_C(  -614.97), EASYSIMD_FLOAT32_C(   727.92) },
      { EASYSIMD_FLOAT32_C(   434.27), EASYSIMD_FLOAT32_C(  -802.75), EASYSIMD_FLOAT32_C(  -491.59), EASYSIMD_FLOAT32_C(   195.41),
        EASYSIMD_FLOAT32_C(  -810.33), EASYSIMD_FLOAT32_C(    55.58), EASYSIMD_FLOAT32_C(  -839.64), EASYSIMD_FLOAT32_C(   145.39) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_andnot_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_andnot_ps");
    easysimd_test_x86_assert_equal_f32x8(easysimd_mm256_loadu_ps(test_vec[i].r), r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_andnot_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_andnot_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float64 src[4];
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   516.44), EASYSIMD_FLOAT64_C(   487.26), EASYSIMD_FLOAT64_C(  -492.02), EASYSIMD_FLOAT64_C(  -921.71) },
      UINT8_C(117),
      { EASYSIMD_FLOAT64_C(   199.23), EASYSIMD_FLOAT64_C(   -76.53), EASYSIMD_FLOAT64_C(   301.03), EASYSIMD_FLOAT64_C(   245.19) },
      { EASYSIMD_FLOAT64_C(   267.25), EASYSIMD_FLOAT64_C(   932.24), EASYSIMD_FLOAT64_C(  -348.92), EASYSIMD_FLOAT64_C(  -376.61) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   487.26), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(  -921.71) } },
    { { EASYSIMD_FLOAT64_C(   176.59), EASYSIMD_FLOAT64_C(   232.51), EASYSIMD_FLOAT64_C(  -530.02), EASYSIMD_FLOAT64_C(  -998.23) },
      UINT8_C(235),
      { EASYSIMD_FLOAT64_C(   744.89), EASYSIMD_FLOAT64_C(   517.12), EASYSIMD_FLOAT64_C(  -654.43), EASYSIMD_FLOAT64_C(  -335.72) },
      { EASYSIMD_FLOAT64_C(   153.18), EASYSIMD_FLOAT64_C(  -432.90), EASYSIMD_FLOAT64_C(   703.45), EASYSIMD_FLOAT64_C(    45.43) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(  -530.02), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -510.11), EASYSIMD_FLOAT64_C(   944.40), EASYSIMD_FLOAT64_C(   713.67), EASYSIMD_FLOAT64_C(    18.00) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT64_C(   230.12), EASYSIMD_FLOAT64_C(  -494.74), EASYSIMD_FLOAT64_C(  -957.73), EASYSIMD_FLOAT64_C(   308.41) },
      { EASYSIMD_FLOAT64_C(  -177.51), EASYSIMD_FLOAT64_C(   241.49), EASYSIMD_FLOAT64_C(  -768.12), EASYSIMD_FLOAT64_C(  -876.48) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(   944.40), EASYSIMD_FLOAT64_C(   713.67), EASYSIMD_FLOAT64_C(    -0.00) } },
    { { EASYSIMD_FLOAT64_C(  -513.32), EASYSIMD_FLOAT64_C(   499.13), EASYSIMD_FLOAT64_C(  -944.24), EASYSIMD_FLOAT64_C(   137.76) },
      UINT8_C(134),
      { EASYSIMD_FLOAT64_C(   232.35), EASYSIMD_FLOAT64_C(  -629.72), EASYSIMD_FLOAT64_C(  -407.50), EASYSIMD_FLOAT64_C(   234.12) },
      { EASYSIMD_FLOAT64_C(   -55.88), EASYSIMD_FLOAT64_C(  -662.61), EASYSIMD_FLOAT64_C(  -248.76), EASYSIMD_FLOAT64_C(   289.68) },
      { EASYSIMD_FLOAT64_C(  -513.32), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   137.76) } },
    { { EASYSIMD_FLOAT64_C(     1.67), EASYSIMD_FLOAT64_C(   904.42), EASYSIMD_FLOAT64_C(   856.78), EASYSIMD_FLOAT64_C(  -294.88) },
      UINT8_C(131),
      { EASYSIMD_FLOAT64_C(  -653.32), EASYSIMD_FLOAT64_C(  -350.48), EASYSIMD_FLOAT64_C(  -336.48), EASYSIMD_FLOAT64_C(   364.68) },
      { EASYSIMD_FLOAT64_C(  -816.20), EASYSIMD_FLOAT64_C(   893.64), EASYSIMD_FLOAT64_C(   869.94), EASYSIMD_FLOAT64_C(  -773.93) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   856.78), EASYSIMD_FLOAT64_C(  -294.88) } },
    { { EASYSIMD_FLOAT64_C(   202.05), EASYSIMD_FLOAT64_C(  -307.57), EASYSIMD_FLOAT64_C(   467.56), EASYSIMD_FLOAT64_C(   433.93) },
      UINT8_C(227),
      { EASYSIMD_FLOAT64_C(   954.25), EASYSIMD_FLOAT64_C(   -66.94), EASYSIMD_FLOAT64_C(  -128.28), EASYSIMD_FLOAT64_C(    92.01) },
      { EASYSIMD_FLOAT64_C(    55.58), EASYSIMD_FLOAT64_C(  -895.93), EASYSIMD_FLOAT64_C(   462.28), EASYSIMD_FLOAT64_C(   648.08) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   467.56), EASYSIMD_FLOAT64_C(   433.93) } },
    { { EASYSIMD_FLOAT64_C(   338.18), EASYSIMD_FLOAT64_C(  -593.60), EASYSIMD_FLOAT64_C(   985.47), EASYSIMD_FLOAT64_C(  -910.58) },
      UINT8_C( 53),
      { EASYSIMD_FLOAT64_C(   -12.86), EASYSIMD_FLOAT64_C(   993.84), EASYSIMD_FLOAT64_C(   552.87), EASYSIMD_FLOAT64_C(   692.27) },
      { EASYSIMD_FLOAT64_C(   -56.32), EASYSIMD_FLOAT64_C(   899.54), EASYSIMD_FLOAT64_C(  -658.21), EASYSIMD_FLOAT64_C(   607.20) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -593.60), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(  -910.58) } },
    { { EASYSIMD_FLOAT64_C(   264.22), EASYSIMD_FLOAT64_C(  -474.41), EASYSIMD_FLOAT64_C(   500.84), EASYSIMD_FLOAT64_C(   134.16) },
      UINT8_C(119),
      { EASYSIMD_FLOAT64_C(  -297.11), EASYSIMD_FLOAT64_C(   826.60), EASYSIMD_FLOAT64_C(  -780.78), EASYSIMD_FLOAT64_C(  -863.18) },
      { EASYSIMD_FLOAT64_C(  -357.45), EASYSIMD_FLOAT64_C(  -826.53), EASYSIMD_FLOAT64_C(    69.88), EASYSIMD_FLOAT64_C(   514.27) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   134.16) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_andnot_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_andnot_pd");
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
    easysimd__m256d r = easysimd_mm256_mask_andnot_pd(src, k, a, b);

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
test_easysimd_mm256_maskz_andnot_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(239),
      { EASYSIMD_FLOAT64_C(   593.87), EASYSIMD_FLOAT64_C(   897.93), EASYSIMD_FLOAT64_C(  -308.44), EASYSIMD_FLOAT64_C(  -311.45) },
      { EASYSIMD_FLOAT64_C(  -708.40), EASYSIMD_FLOAT64_C(   -91.91), EASYSIMD_FLOAT64_C(   251.18), EASYSIMD_FLOAT64_C(   450.96) },
      { EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 68),
      { EASYSIMD_FLOAT64_C(  -836.29), EASYSIMD_FLOAT64_C(   147.18), EASYSIMD_FLOAT64_C(  -448.99), EASYSIMD_FLOAT64_C(  -757.71) },
      { EASYSIMD_FLOAT64_C(   296.11), EASYSIMD_FLOAT64_C(   809.50), EASYSIMD_FLOAT64_C(  -769.71), EASYSIMD_FLOAT64_C(   267.06) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 10),
      { EASYSIMD_FLOAT64_C(  -344.31), EASYSIMD_FLOAT64_C(  -723.65), EASYSIMD_FLOAT64_C(  -927.48), EASYSIMD_FLOAT64_C(   217.33) },
      { EASYSIMD_FLOAT64_C(  -438.95), EASYSIMD_FLOAT64_C(   448.72), EASYSIMD_FLOAT64_C(   327.70), EASYSIMD_FLOAT64_C(  -645.92) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00) } },
    { UINT8_C(246),
      { EASYSIMD_FLOAT64_C(    34.88), EASYSIMD_FLOAT64_C(   895.12), EASYSIMD_FLOAT64_C(  -470.39), EASYSIMD_FLOAT64_C(  -268.91) },
      { EASYSIMD_FLOAT64_C(   488.99), EASYSIMD_FLOAT64_C(  -572.46), EASYSIMD_FLOAT64_C(   422.65), EASYSIMD_FLOAT64_C(  -822.46) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 28),
      { EASYSIMD_FLOAT64_C(  -669.27), EASYSIMD_FLOAT64_C(   428.72), EASYSIMD_FLOAT64_C(  -829.89), EASYSIMD_FLOAT64_C(   727.64) },
      { EASYSIMD_FLOAT64_C(   592.43), EASYSIMD_FLOAT64_C(   317.29), EASYSIMD_FLOAT64_C(  -721.35), EASYSIMD_FLOAT64_C(   834.72) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 34),
      { EASYSIMD_FLOAT64_C(  -911.85), EASYSIMD_FLOAT64_C(  -934.99), EASYSIMD_FLOAT64_C(   880.46), EASYSIMD_FLOAT64_C(  -287.83) },
      { EASYSIMD_FLOAT64_C(  -279.30), EASYSIMD_FLOAT64_C(  -843.18), EASYSIMD_FLOAT64_C(  -215.31), EASYSIMD_FLOAT64_C(   938.03) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(175),
      { EASYSIMD_FLOAT64_C(  -766.59), EASYSIMD_FLOAT64_C(   265.73), EASYSIMD_FLOAT64_C(    71.95), EASYSIMD_FLOAT64_C(  -929.41) },
      { EASYSIMD_FLOAT64_C(  -699.39), EASYSIMD_FLOAT64_C(   -32.93), EASYSIMD_FLOAT64_C(  -399.80), EASYSIMD_FLOAT64_C(    31.70) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 20),
      { EASYSIMD_FLOAT64_C(    27.74), EASYSIMD_FLOAT64_C(  -545.66), EASYSIMD_FLOAT64_C(  -366.40), EASYSIMD_FLOAT64_C(   746.89) },
      { EASYSIMD_FLOAT64_C(  -214.93), EASYSIMD_FLOAT64_C(  -937.68), EASYSIMD_FLOAT64_C(   917.00), EASYSIMD_FLOAT64_C(  -487.29) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_andnot_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_andnot_pd");
    easysimd_test_x86_assert_equal_f64x4(easysimd_mm256_loadu_pd(test_vec[i].r), r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_andnot_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_andnot_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C(  515723887), INT32_C( 1640697809), INT32_C(-1815268655), INT32_C( -855842079),
                            INT32_C( -876731021), INT32_C( -422224087), INT32_C( 1402147089), INT32_C(  791567468),
                            INT32_C( -405953943), INT32_C(  280958773), INT32_C(  359942894), INT32_C( -574064836),
                            INT32_C( 1041426019), INT32_C(  457085316), INT32_C( 1591682265), INT32_C( 1681068921)),
      easysimd_mm512_set_epi32(INT32_C(  219659736), INT32_C(  983393088), INT32_C(  175097612), INT32_C(-1948389752),
                            INT32_C(-1760046217), INT32_C(  996280401), INT32_C( -511102649), INT32_C(-1367446405),
                            INT32_C(-1955017682), INT32_C(-1962774388), INT32_C( -112156455), INT32_C( 1625681413),
                            INT32_C( -214953654), INT32_C(-1594240596), INT32_C( -837160883), INT32_C(   91929810)),
      easysimd_mm512_set_epi32(INT32_C(   16952720), INT32_C(  437542912), INT32_C(  136496396), INT32_C(   50400264),
                            INT32_C(  335663620), INT32_C(  421658704), INT32_C(-1610080186), INT32_C(-2142232557),
                            INT32_C(  137382918), INT32_C(-1962909560), INT32_C( -402612207), INT32_C(  539330561),
                            INT32_C(-1054863096), INT32_C(-1597947864), INT32_C(-2147430396), INT32_C(   21539970)) },
    { easysimd_mm512_set_epi32(INT32_C( -691442479), INT32_C(-1656332537), INT32_C( -736641091), INT32_C( 1498293216),
                            INT32_C( -507651370), INT32_C( 1481766884), INT32_C(-1911092113), INT32_C( -872210414),
                            INT32_C(  291047220), INT32_C(-1241987411), INT32_C( 1619041328), INT32_C( 1464413104),
                            INT32_C(-1017310468), INT32_C( 1540491270), INT32_C( 2102275128), INT32_C(-1414382909)),
      easysimd_mm512_set_epi32(INT32_C(  242206574), INT32_C(  555720064), INT32_C( -659215600), INT32_C( 1975929957),
                            INT32_C( 1131537123), INT32_C( 2072355897), INT32_C( 1377537047), INT32_C( 1623632095),
                            INT32_C(  536506999), INT32_C(-1382727392), INT32_C(   37097013), INT32_C( 2004578493),
                            INT32_C( 1803364246), INT32_C( 1342516983), INT32_C(  514234840), INT32_C(-1522858319)),
      easysimd_mm512_set_epi32(INT32_C(  136741678), INT32_C(  538542208), INT32_C(  144703488), INT32_C(  612387845),
                            INT32_C(   37756961), INT32_C(  595853337), INT32_C( 1342734352), INT32_C(  549750989),
                            INT32_C(  245526595), INT32_C(  134545664), INT32_C(   37093893), INT32_C(  540168205),
                            INT32_C(  673195266), INT32_C(     337649), INT32_C(   44077504), INT32_C(   67699760)) },
    { easysimd_mm512_set_epi32(INT32_C(  835536002), INT32_C(  -63027427), INT32_C( 2017135186), INT32_C(-1844829768),
                            INT32_C(  936597093), INT32_C( -389163916), INT32_C(-1786076372), INT32_C(   62625566),
                            INT32_C(-1459727459), INT32_C( 1125674521), INT32_C(-1286537639), INT32_C(  550088134),
                            INT32_C(  -31520277), INT32_C(-1814664190), INT32_C( 1588224923), INT32_C( 1901241906)),
      easysimd_mm512_set_epi32(INT32_C( -684209907), INT32_C( 1116413094), INT32_C(  -29612798), INT32_C(-1906935505),
                            INT32_C( 1879010472), INT32_C( -984076172), INT32_C( -987963932), INT32_C(-1705792694),
                            INT32_C( 1395521155), INT32_C( 2062716504), INT32_C( 1645135174), INT32_C(  273600119),
                            INT32_C( -650211201), INT32_C( 1039685180), INT32_C(-1680417560), INT32_C( -160711201)),
      easysimd_mm512_set_epi32(INT32_C( -969766643), INT32_C(   42014882), INT32_C(-2046548736), INT32_C(  206854151),
                            INT32_C( 1210853512), INT32_C(   84942848), INT32_C( 1075076288), INT32_C(-1740625856),
                            INT32_C( 1392616450), INT32_C(  954368064), INT32_C( 1074659590), INT32_C(  268845617),
                            INT32_C(   18911252), INT32_C(  740823612), INT32_C(-2125161376), INT32_C(-2044126771)) },
    { easysimd_mm512_set_epi32(INT32_C( -911319633), INT32_C(-1035947605), INT32_C(  -14347010), INT32_C(  135240154),
                            INT32_C( 1039097026), INT32_C(-1325726567), INT32_C( 1814577462), INT32_C( -309546152),
                            INT32_C( 2107794809), INT32_C( -690752206), INT32_C(-1567183976), INT32_C( 1570875131),
                            INT32_C( -359037430), INT32_C( 1064726494), INT32_C( -305221103), INT32_C( 2039553475)),
      easysimd_mm512_set_epi32(INT32_C( -358609490), INT32_C( 1748558231), INT32_C(  769947846), INT32_C(-2114787166),
                            INT32_C( 1221765938), INT32_C(-1846472677), INT32_C(  893676657), INT32_C(-2056337544),
                            INT32_C(-1125244927), INT32_C(-2123257127), INT32_C( 1395407144), INT32_C(-2100494303),
                            INT32_C(  123622128), INT32_C( -234312093), INT32_C( 1320504606), INT32_C( -696459867)),
      easysimd_mm512_set_epi32(INT32_C(  570425344), INT32_C(  674775060), INT32_C(   12609536), INT32_C(-2114953184),
                            INT32_C( 1074832176), INT32_C(   16781314), INT32_C(  289680449), INT32_C(    6439968),
                            INT32_C(-2142502912), INT32_C(   18877129), INT32_C( 1361580064), INT32_C(-2108948480),
                            INT32_C(   88494320), INT32_C(-1073184735), INT32_C(   36785422), INT32_C(-2039693276)) },
    { easysimd_mm512_set_epi32(INT32_C( 1741169869), INT32_C(-1806166644), INT32_C( 1030404360), INT32_C( 1645919232),
                            INT32_C( -724495967), INT32_C( 1251263729), INT32_C( -769398486), INT32_C(-1951408118),
                            INT32_C( 1006137744), INT32_C( -650052668), INT32_C( 1803988670), INT32_C( -565766270),
                            INT32_C(-2075332822), INT32_C(  -77783473), INT32_C( 1442895719), INT32_C( -423885068)),
      easysimd_mm512_set_epi32(INT32_C(-1467349800), INT32_C(-1486916034), INT32_C(  580711779), INT32_C( 1504148541),
                            INT32_C(  661197291), INT32_C( 2016703871), INT32_C(  459937445), INT32_C( 1081922115),
                            INT32_C(   93168137), INT32_C( -744509287), INT32_C(  -84767472), INT32_C( 1535078904),
                            INT32_C( 1804568444), INT32_C(-1641570308), INT32_C( 1307677448), INT32_C( -156993467)),
      easysimd_mm512_set_epi32(INT32_C(-2013134832), INT32_C(  587686450), INT32_C(   43271267), INT32_C(  430260285),
                            INT32_C(  589824074), INT32_C(  807416078), INT32_C(  155719301), INT32_C( 1078989377),
                            INT32_C(   67469833), INT32_C(   43974681), INT32_C(-1871707904), INT32_C(   20471928),
                            INT32_C( 1803747412), INT32_C(   69370288), INT32_C(  166789128), INT32_C(  268466177)) },
    { easysimd_mm512_set_epi32(INT32_C( -789590264), INT32_C( 1747530260), INT32_C(  250254813), INT32_C(  -46824160),
                            INT32_C( 1521185343), INT32_C( 1710396447), INT32_C( -401960034), INT32_C(  376331638),
                            INT32_C( -481899788), INT32_C(  951540577), INT32_C(-1886694025), INT32_C( -615462627),
                            INT32_C(-1246126101), INT32_C( 1628361415), INT32_C( 1197988194), INT32_C(-1740462923)),
      easysimd_mm512_set_epi32(INT32_C( 1235841465), INT32_C(-1524332124), INT32_C( 1158299501), INT32_C(-2030663913),
                            INT32_C( -368124005), INT32_C( -216689066), INT32_C( 2049678955), INT32_C(-1811053975),
                            INT32_C( -579248849), INT32_C(  817648154), INT32_C( 1351147076), INT32_C( -248769414),
                            INT32_C( 1542937557), INT32_C(-1429188342), INT32_C(  753897242), INT32_C( -407543559)),
      easysimd_mm512_set_epi32(INT32_C(  151007409), INT32_C(-2063300192), INT32_C( 1090529824), INT32_C(   46268951),
                            INT32_C(-1610314368), INT32_C(-1845165504), INT32_C(  304155745), INT32_C(-2147384823),
                            INT32_C(  473502987), INT32_C(     524314), INT32_C( 1342212608), INT32_C(  539759714),
                            INT32_C( 1246117908), INT32_C(-1966079736), INT32_C(  680003096), INT32_C( 1739931720)) },
    { easysimd_mm512_set_epi32(INT32_C( 1871269268), INT32_C(  408476277), INT32_C(  620349445), INT32_C(   85656022),
                            INT32_C(  530242315), INT32_C( 1600939321), INT32_C( 1166499662), INT32_C(  550456559),
                            INT32_C( 1205553840), INT32_C( -507718293), INT32_C( -629410605), INT32_C(-1400491933),
                            INT32_C(-1740280079), INT32_C(  470828561), INT32_C(  710611826), INT32_C( 1460766627)),
      easysimd_mm512_set_epi32(INT32_C(  132567711), INT32_C( -504432561), INT32_C( 1784336368), INT32_C(-1195419261),
                            INT32_C(-1432068840), INT32_C( -756951336), INT32_C(  519218456), INT32_C( 2068445443),
                            INT32_C( -769032976), INT32_C(-1464370595), INT32_C( -636201129), INT32_C(  798298919),
                            INT32_C(  605141360), INT32_C( 1690763202), INT32_C(  743563485), INT32_C( 1615889032)),
      easysimd_mm512_set_epi32(INT32_C(    6722059), INT32_C( -509206518), INT32_C( 1241647088), INT32_C(-1197189119),
                            INT32_C(-1608252400), INT32_C(-2138991936), INT32_C(  443588624), INT32_C( 1526771968),
                            INT32_C(-1876937664), INT32_C(  134425108), INT32_C(     262916), INT32_C(   51446532),
                            INT32_C(  605065472), INT32_C( 1623636418), INT32_C(   67166349), INT32_C(  541065224)) },
    { easysimd_mm512_set_epi32(INT32_C( 1287269628), INT32_C( 1003736038), INT32_C(  977850641), INT32_C(-1038923525),
                            INT32_C( -628842024), INT32_C( 1597060388), INT32_C( -643406365), INT32_C(-1390651863),
                            INT32_C( 1433162166), INT32_C(  -27649596), INT32_C( -695421854), INT32_C( 1977918902),
                            INT32_C(-1118619506), INT32_C(  218268934), INT32_C(  602753386), INT32_C( -663684258)),
      easysimd_mm512_set_epi32(INT32_C(  238738926), INT32_C( 1501256933), INT32_C( -668514921), INT32_C(  178997567),
                            INT32_C(  618897994), INT32_C(-1305584804), INT32_C(  287401445), INT32_C( -682321436),
                            INT32_C(-1248279406), INT32_C(-1232466621), INT32_C( 1932263578), INT32_C( 1672045836),
                            INT32_C(  -31634555), INT32_C( -429030840), INT32_C( 1478948841), INT32_C( -636575791)),
      easysimd_mm512_set_epi32(INT32_C(   33607938), INT32_C( 1076368385), INT32_C(-1071184762), INT32_C(  145228036),
                            INT32_C(  610468354), INT32_C(-1609676200), INT32_C(      65540), INT32_C( 1379961284),
                            INT32_C(-1601141760), INT32_C(    8389635), INT32_C(  555745432), INT32_C(   34160648),
                            INT32_C( 1108099841), INT32_C( -496172472), INT32_C( 1476572801), INT32_C(   34473089)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_castps_si512(easysimd_mm512_andnot_ps(easysimd_mm512_castsi512_ps(test_vec[i].a), easysimd_mm512_castsi512_ps(test_vec[i].b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_andnot_ps");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_andnot_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(-5692392796256408556), INT64_C( 6556277497990144923),
                            INT64_C(-8451768093244871108), INT64_C( 2502789693644361692),
                            INT64_C( 1621880469938104082), INT64_C(-7297255235572331483),
                            INT64_C(-2352677665930074905), INT64_C( 4911158619134204312)),
      easysimd_mm512_set_epi64(INT64_C(-2565389980846007780), INT64_C( 3404730604833389160),
                            INT64_C( 1179026943527716274), INT64_C(-2906973067026822223),
                            INT64_C( 7260818647608791158), INT64_C( 5579870493016706466),
                            INT64_C( 3863525595432901356), INT64_C(-6057345860275027490)),
      easysimd_mm512_set_epi64(INT64_C( 5504912800215142920), INT64_C( 2666140128227367008),
                            INT64_C( 1173388089487589762), INT64_C(-3098395176816541663),
                            INT64_C( 6935978858541748324), INT64_C( 4991411844055941506),
                            INT64_C( 2343560720521648648), INT64_C(-6068587301518688186)) },
    { easysimd_mm512_set_epi64(INT64_C(-2759528052506956708), INT64_C(-5575162278173961328),
                            INT64_C(-8694367187788105741), INT64_C(-8990346118631710119),
                            INT64_C( 4592063463963295950), INT64_C( 8368804684692221223),
                            INT64_C(-6476206435262682842), INT64_C(-8876450871330607726)),
      easysimd_mm512_set_epi64(INT64_C(-6737293364691021911), INT64_C(-4204449908398867590),
                            INT64_C(  160841008688998741), INT64_C(-6365408201349652038),
                            INT64_C( 2001896472947132682), INT64_C(  -99471897253479533),
                            INT64_C(-1326129419939093389), INT64_C(-2924053864540399741)),
      easysimd_mm512_set_epi64(INT64_C( 2450046296219059105), INT64_C( 4973873968629295210),
                            INT64_C(   11267795499616260), INT64_C( 2630102325595865506),
                            INT64_C(   18051886326876416), INT64_C(-8458876851257732464),
                            INT64_C( 5296233507537044561), INT64_C( 5992932110406914561)) },
    { easysimd_mm512_set_epi64(INT64_C(-7457051575750248602), INT64_C( 6070335147558558873),
                            INT64_C(-7490477224728001543), INT64_C( 1279317055657379478),
                            INT64_C( 8751242136386772213), INT64_C(  637231009559692595),
                            INT64_C(-7114327187130069406), INT64_C( 1391778837665435621)),
      easysimd_mm512_set_epi64(INT64_C( 6308740259462318802), INT64_C(-9185391234602091403),
                            INT64_C( 7626881538428569222), INT64_C(-3068069010762250319),
                            INT64_C(-4951576282750300305), INT64_C(  128509774881067912),
                            INT64_C( 7892251767542575626), INT64_C(-8930995426443097321)),
      easysimd_mm512_set_epi64(INT64_C( 5119472124868954256), INT64_C(-9187125192365432732),
                            INT64_C( 7048133801267954694), INT64_C(-4311066996174143199),
                            INT64_C(-9076908839077919990), INT64_C(   74319038222733960),
                            INT64_C( 6954164901648707592), INT64_C(-8931156479129911278)) },
    { easysimd_mm512_set_epi64(INT64_C( 6964163421595280406), INT64_C( 8751199602933822917),
                            INT64_C( 3889132740347846858), INT64_C(-7126879543636862431),
                            INT64_C(-7513660375211080284), INT64_C( 6453129860776144209),
                            INT64_C(-5544576080495062479), INT64_C( 5873331717169095384)),
      easysimd_mm512_set_epi64(INT64_C( 8830709936142460331), INT64_C(-9018958152160609695),
                            INT64_C( 8077330800987365186), INT64_C(-8496831583043834543),
                            INT64_C(-8535629658492460138), INT64_C(-7680526102244966263),
                            INT64_C( -372932882462668779), INT64_C( 6329513109562115746)),
      easysimd_mm512_set_epi64(INT64_C( 1875839696979362217), INT64_C(-9042040235334956512),
                            INT64_C( 4611686019604422400), INT64_C(  145522704699106640),
                            INT64_C(  576814847044362770), INT64_C(-8907768133275565944),
                            INT64_C( 5247259266967930884), INT64_C(  456185876494468130)) },
    { easysimd_mm512_set_epi64(INT64_C( -351666990455047830), INT64_C(-7399285389685964954),
                            INT64_C(-5908952440536913792), INT64_C( -611732173843171755),
                            INT64_C( 7999973001790565510), INT64_C(-8075898444541975424),
                            INT64_C( 5770350522878101247), INT64_C( 1116848091668783433)),
      easysimd_mm512_set_epi64(INT64_C( 2217552425319516429), INT64_C(-8721047939211270856),
                            INT64_C(-2880324325532209431), INT64_C( -187231364083775137),
                            INT64_C( 6742854000402878536), INT64_C(     168773737674717),
                            INT64_C( 3418235066721438872), INT64_C( 4202828047673997422)),
      easysimd_mm512_set_epi64(INT64_C(  342365941483028485), INT64_C(  479783009225555992),
                            INT64_C( 5764607798056863337), INT64_C(  604680821040318730),
                            INT64_C( 1194123438370115656), INT64_C(      26768921351005),
                            INT64_C( 3413729087268849664), INT64_C( 3458800331700312102)) },
    { easysimd_mm512_set_epi64(INT64_C(-2132909336669479608), INT64_C(-1158827795013308041),
                            INT64_C(-7670914575902882420), INT64_C(  -69696623451151043),
                            INT64_C(-4047902191338288971), INT64_C( 7092767718101885012),
                            INT64_C( 5934909912424448575), INT64_C( 5411709750270769968)),
      easysimd_mm512_set_epi64(INT64_C(-7875865474019974757), INT64_C(-3285041077981983127),
                            INT64_C( 8063284926890959108), INT64_C(-3700459330126222884),
                            INT64_C(-7671356082612531796), INT64_C( 1792383659764879933),
                            INT64_C( 2583453571264272321), INT64_C( 8675197907294370872)),
      easysimd_mm512_set_epi64(INT64_C( 1193748656575226003), INT64_C( 1152974871187164680),
                            INT64_C( 7666393384072290304), INT64_C(   46461582121208000),
                            INT64_C( 1155463611999061256), INT64_C( 1770267941057597993),
                            INT64_C( 2414567396337178048), INT64_C( 3486980733385704456)) },
    { easysimd_mm512_set_epi64(INT64_C(-3578776133799908286), INT64_C(-1505161927362377530),
                            INT64_C( 1984257760933558326), INT64_C( -235993280127523291),
                            INT64_C(-5471198518359697501), INT64_C(-3736915368061275681),
                            INT64_C(-2239211533422890096), INT64_C(-3284418263843820488)),
      easysimd_mm512_set_epi64(INT64_C(-2978941464173404520), INT64_C( 4582889970668771380),
                            INT64_C( -557407531320217043), INT64_C( 5386308122944286215),
                            INT64_C(-2433611387892931894), INT64_C( 7741810302662188301),
                            INT64_C(   18824623009495704), INT64_C( 3999273364541981338)),
      easysimd_mm512_set_epi64(INT64_C( 1200223062730085016), INT64_C( 1477497354446704688),
                            INT64_C(-2287124905805331959), INT64_C(  162132485929502722),
                            INT64_C( 5343540817909325896), INT64_C( 2544569189669112832),
                            INT64_C(     633336187848712), INT64_C( 2702160884594393730)) },
    { easysimd_mm512_set_epi64(INT64_C(  352684271852599798), INT64_C( 4911474499221167587),
                            INT64_C( 1508056965830938497), INT64_C( 3074813921141815339),
                            INT64_C( 7701628738251481990), INT64_C( -466066103765916190),
                            INT64_C( 8562974168142071295), INT64_C( -919355185316238533)),
      easysimd_mm512_set_epi64(INT64_C( 4326901039471149930), INT64_C(-7137503476184318358),
                            INT64_C(-1078077923693263341), INT64_C(-1859594942180658021),
                            INT64_C(-4053912759805256064), INT64_C(-4136267192341554803),
                            INT64_C( 7711878059533707111), INT64_C(-6181901304080395815)),
      easysimd_mm512_set_epi64(INT64_C( 4037478715378254344), INT64_C(-7434758645930761720),
                            INT64_C(-2233710626472820718), INT64_C(-4318952040280619888),
                            INT64_C(-8855183721016321024), INT64_C(  437135751054956557),
                            INT64_C(  649099009830593024), INT64_C(  576514078768369856)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_castpd_si512(easysimd_mm512_andnot_pd(easysimd_mm512_castsi512_pd(test_vec[i].a), easysimd_mm512_castsi512_pd(test_vec[i].b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_andnot_pd");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_andnot_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
     { easysimd_mm512_set_epi32(INT32_C(  483765022), INT32_C(-1234873154), INT32_C(-1289658932), INT32_C( 1557667178),
                            INT32_C( -573006378), INT32_C( -844585804), INT32_C(  908677468), INT32_C(  120945929),
                            INT32_C(-1595338087), INT32_C(-1433288415), INT32_C( 1272415402), INT32_C( 2052605464),
                            INT32_C(-1185243420), INT32_C(  696776161), INT32_C(  617510437), INT32_C( 1274220393)),
      UINT16_C(33954),
      easysimd_mm512_set_epi32(INT32_C( 1145617415), INT32_C( -605546679), INT32_C(    2927077), INT32_C(  -19849762),
                            INT32_C(-1597262180), INT32_C( -867043590), INT32_C( -107439489), INT32_C( 1287210357),
                            INT32_C( 1092392250), INT32_C( 1062404217), INT32_C( -979680700), INT32_C( 1129202470),
                            INT32_C( 1479969823), INT32_C( -738882529), INT32_C( 1249939660), INT32_C( -548556138)),
      easysimd_mm512_set_epi32(INT32_C(  283318882), INT32_C( 1846644474), INT32_C( 2020741558), INT32_C( 2084726692),
                            INT32_C(-1625067961), INT32_C( 1808817126), INT32_C(  188488265), INT32_C( -493292109),
                            INT32_C(-1012406283), INT32_C( 2120995640), INT32_C(-1978262848), INT32_C( -210107724),
                            INT32_C(  789544495), INT32_C(  -10089859), INT32_C( -531570606), INT32_C( 1286299547)),
      easysimd_mm512_set_epi32(INT32_C(  279118432), INT32_C(-1234873154), INT32_C(-1289658932), INT32_C( 1557667178),
                            INT32_C( -573006378), INT32_C(  595593476), INT32_C(  908677468), INT32_C(  120945929),
                            INT32_C(-2103221563), INT32_C(-1433288415), INT32_C(  168040576), INT32_C( 2052605464),
                            INT32_C(-1185243420), INT32_C(  696776161), INT32_C(-1605345262), INT32_C( 1274220393)) },
    { easysimd_mm512_set_epi32(INT32_C( -281178768), INT32_C( -360418194), INT32_C( 1198549209), INT32_C( -896335694),
                            INT32_C( 1796051299), INT32_C( -602464105), INT32_C( 1096879395), INT32_C( 2101844446),
                            INT32_C( 1483513958), INT32_C(   55530807), INT32_C(-1589480307), INT32_C(  -48228318),
                            INT32_C(  889897511), INT32_C( 1575441246), INT32_C(-1726327647), INT32_C(  657269965)),
      UINT16_C(34646),
      easysimd_mm512_set_epi32(INT32_C( 1170712187), INT32_C( 1365513540), INT32_C( -159824212), INT32_C(  976500494),
                            INT32_C(   -2253502), INT32_C( 1424815879), INT32_C( 1340211205), INT32_C(  478098159),
                            INT32_C(  545970493), INT32_C(-1738506699), INT32_C(  725751947), INT32_C(  876157308),
                            INT32_C(  729412496), INT32_C( 1316518940), INT32_C(  479437804), INT32_C( 1749631626)),
      easysimd_mm512_set_epi32(INT32_C( -544439732), INT32_C(  514265282), INT32_C(-1783487008), INT32_C(-1073881913),
                            INT32_C( -917759499), INT32_C(  721599990), INT32_C( 1403076580), INT32_C( -867638009),
                            INT32_C(-2070564498), INT32_C( 1494227565), INT32_C(  156325221), INT32_C( -117354474),
                            INT32_C(-2029550992), INT32_C( 1184377155), INT32_C( -325424585), INT32_C( -563054056)),
      easysimd_mm512_set_epi32(INT32_C(-1710751740), INT32_C( -360418194), INT32_C( 1198549209), INT32_C( -896335694),
                            INT32_C( 1796051299), INT32_C(  721553648), INT32_C(  268517344), INT32_C(-1073691392),
                            INT32_C( 1483513958), INT32_C( 1091567688), INT32_C(-1589480307), INT32_C( -922730494),
                            INT32_C(  889897511), INT32_C(    8388931), INT32_C( -536328173), INT32_C(  657269965)) },
    { easysimd_mm512_set_epi32(INT32_C(-1745677982), INT32_C(  235931267), INT32_C( -555741923), INT32_C(  150463911),
                            INT32_C(  354747494), INT32_C( 2006985747), INT32_C( 1517200768), INT32_C( -149674742),
                            INT32_C(-1301892689), INT32_C( 1164273534), INT32_C( -519614566), INT32_C( 1518672842),
                            INT32_C(-1430542782), INT32_C( -567985198), INT32_C( 1793594874), INT32_C( 1766364533)),
      UINT16_C(51458),
      easysimd_mm512_set_epi32(INT32_C(-1697411653), INT32_C(  213103619), INT32_C( 1166379858), INT32_C(  530625194),
                            INT32_C( 1706895557), INT32_C(-1311465088), INT32_C(  793729023), INT32_C(-1062948513),
                            INT32_C(  -58027177), INT32_C( -215831346), INT32_C(-1081872765), INT32_C(  617218322),
                            INT32_C( 1703489303), INT32_C( 1228468220), INT32_C(  705631662), INT32_C( 1003062693)),
      easysimd_mm512_set_epi32(INT32_C(-1197760733), INT32_C(-1777870117), INT32_C( 1151957666), INT32_C( -467243461),
                            INT32_C( 1044840108), INT32_C( 1467862627), INT32_C(  340861518), INT32_C( -683495543),
                            INT32_C( -171219649), INT32_C(-1277374003), INT32_C(-2049184175), INT32_C( -804992531),
                            INT32_C( 1254613706), INT32_C( -484210109), INT32_C( -976973176), INT32_C(  768220545)),
      easysimd_mm512_set_epi32(INT32_C(  537399808), INT32_C(-1845212456), INT32_C( -555741923), INT32_C(  150463911),
                            INT32_C(  440587816), INT32_C( 2006985747), INT32_C( 1517200768), INT32_C(  390205056),
                            INT32_C(-1301892689), INT32_C( 1164273534), INT32_C( -519614566), INT32_C( 1518672842),
                            INT32_C(-1430542782), INT32_C( -567985198), INT32_C( -977239552), INT32_C( 1766364533)) },
    { easysimd_mm512_set_epi32(INT32_C( 1636500168), INT32_C(  444177967), INT32_C(-1663266514), INT32_C(  191092965),
                            INT32_C(  488118829), INT32_C(-1542228246), INT32_C(-1543977108), INT32_C(-1747326233),
                            INT32_C(  472323781), INT32_C(  181690416), INT32_C(   -8111931), INT32_C(-1512462189),
                            INT32_C(-1412708648), INT32_C( -857864914), INT32_C(-1610668993), INT32_C( 2003858110)),
      UINT16_C( 5589),
      easysimd_mm512_set_epi32(INT32_C( -283174658), INT32_C(  170838247), INT32_C( -393103783), INT32_C( 2067132417),
                            INT32_C( -418400070), INT32_C(-1518152549), INT32_C( 1910825371), INT32_C(-1243038545),
                            INT32_C(  116520479), INT32_C( -366505216), INT32_C( 1914112492), INT32_C( 1911296968),
                            INT32_C( 2113218059), INT32_C( -692180631), INT32_C(-1020362892), INT32_C( -633211439)),
      easysimd_mm512_set_epi32(INT32_C( 1184440056), INT32_C(  166652038), INT32_C(-1574005475), INT32_C( 2085250974),
                            INT32_C(-1914483545), INT32_C( -801496013), INT32_C( 1887253581), INT32_C(-1389414117),
                            INT32_C(-1991582465), INT32_C(  878735212), INT32_C(-1594175370), INT32_C( 2077658842),
                            INT32_C(-1116765072), INT32_C( 1279728229), INT32_C( 1087544376), INT32_C( 2038214643)),
      easysimd_mm512_set_epi32(INT32_C( 1636500168), INT32_C(  444177967), INT32_C(-1663266514), INT32_C(   71960478),
                            INT32_C(  488118829), INT32_C( 1345855520), INT32_C(-1543977108), INT32_C(  134676752),
                            INT32_C(-1995831072), INT32_C(  339766380), INT32_C(   -8111931), INT32_C(  168987666),
                            INT32_C(-1412708648), INT32_C(  138482180), INT32_C(-1610668993), INT32_C(  557582882)) },
    { easysimd_mm512_set_epi32(INT32_C(  551147024), INT32_C( -687338198), INT32_C(   60918053), INT32_C( 1437206085),
                            INT32_C(  434041201), INT32_C( 1422808900), INT32_C(  419480808), INT32_C(-1939817409),
                            INT32_C(-1683817642), INT32_C( -409888460), INT32_C( 1718430638), INT32_C( 1457046604),
                            INT32_C(  734344028), INT32_C(  175091099), INT32_C(  770584551), INT32_C(  -95488435)),
      UINT16_C(29324),
      easysimd_mm512_set_epi32(INT32_C( 1939419432), INT32_C( -691029505), INT32_C( -442395497), INT32_C( -427009027),
                            INT32_C(  817522174), INT32_C(    8776211), INT32_C( 1606933870), INT32_C( -913009701),
                            INT32_C(-1219423042), INT32_C(  450853660), INT32_C(  761339041), INT32_C(  889962544),
                            INT32_C(-1736069360), INT32_C(-1763810886), INT32_C(-1763494181), INT32_C( 1322133292)),
      easysimd_mm512_set_epi32(INT32_C(  114683937), INT32_C( 1592723028), INT32_C(  623286176), INT32_C(-1573004789),
                            INT32_C(  386412089), INT32_C( 1236627295), INT32_C( -815669616), INT32_C( 2140872084),
                            INT32_C(-1844875837), INT32_C(  266739419), INT32_C(-1210833034), INT32_C( 1948981056),
                            INT32_C( -293676893), INT32_C( 1361522457), INT32_C(  417503278), INT32_C(  633831284)),
      easysimd_mm512_set_epi32(INT32_C(  551147024), INT32_C(  136316416), INT32_C(     394016), INT32_C(    3244034),
                            INT32_C(  434041201), INT32_C( 1422808900), INT32_C(-2145385840), INT32_C(-1939817409),
                            INT32_C(     551233), INT32_C( -409888460), INT32_C( 1718430638), INT32_C( 1457046604),
                            INT32_C( 1719292067), INT32_C( 1092685313), INT32_C(  770584551), INT32_C(  -95488435)) },
    { easysimd_mm512_set_epi32(INT32_C(-1371022440), INT32_C( 1457704499), INT32_C( -431597639), INT32_C(-1022830061),
                            INT32_C(   36727871), INT32_C(  132345530), INT32_C(-1160653220), INT32_C( 1075044178),
                            INT32_C( 1947162433), INT32_C(  484643153), INT32_C(-1413771472), INT32_C( -151443305),
                            INT32_C(  -82344071), INT32_C(-1396164880), INT32_C(  775295095), INT32_C( 1585972112)),
      UINT16_C(54244),
      easysimd_mm512_set_epi32(INT32_C( 1350970412), INT32_C(-1442308200), INT32_C( 1774467796), INT32_C( -258916798),
                            INT32_C(-1518028161), INT32_C( 1215654276), INT32_C(-1158758506), INT32_C(-1884048450),
                            INT32_C( -996858784), INT32_C( 1572275854), INT32_C(  -61363356), INT32_C(   71635930),
                            INT32_C(  890553866), INT32_C(-1657029576), INT32_C(  875900884), INT32_C(  232674574)),
      easysimd_mm512_set_epi32(INT32_C( -559322868), INT32_C(   26562494), INT32_C( 1556236736), INT32_C(  144590511),
                            INT32_C( 2137277580), INT32_C(-1485572616), INT32_C(  664308651), INT32_C(  525825403),
                            INT32_C( 1235000793), INT32_C(  818058128), INT32_C( 1639942075), INT32_C( 1363996226),
                            INT32_C(-1688385601), INT32_C(  521315224), INT32_C( -495140458), INT32_C( 2110266874)),
      easysimd_mm512_set_epi32(INT32_C(-1909899008), INT32_C(   26558502), INT32_C( -431597639), INT32_C(  135151789),
                            INT32_C(   36727871), INT32_C(  132345530), INT32_C(   84934697), INT32_C(  272909377),
                            INT32_C(  151558553), INT32_C(  541102352), INT32_C(   27788443), INT32_C( -151443305),
                            INT32_C(  -82344071), INT32_C(   33555328), INT32_C(  775295095), INT32_C( 1585972112)) },
    { easysimd_mm512_set_epi32(INT32_C(-1445633201), INT32_C(-1516803416), INT32_C( 2047415330), INT32_C(  756009385),
                            INT32_C(  795635255), INT32_C(  735619934), INT32_C(-1886661005), INT32_C( 1006199392),
                            INT32_C( -253641367), INT32_C(  505896362), INT32_C(  377279653), INT32_C(  782384760),
                            INT32_C(-2053863520), INT32_C(  173648830), INT32_C(-1212193602), INT32_C(  646275887)),
      UINT16_C( 3833),
      easysimd_mm512_set_epi32(INT32_C(-1717413045), INT32_C(   37772527), INT32_C(  997132272), INT32_C( 1212574322),
                            INT32_C(  -50264086), INT32_C( 1583086284), INT32_C(-1387426254), INT32_C(  542967980),
                            INT32_C(  321849276), INT32_C( 2124033808), INT32_C( 1752461294), INT32_C(-1726583281),
                            INT32_C( -438403938), INT32_C(-1226147069), INT32_C( 1033013441), INT32_C(-1845989576)),
      easysimd_mm512_set_epi32(INT32_C( -928885408), INT32_C( 1847851352), INT32_C(-1563646145), INT32_C(-1610113698),
                            INT32_C( -632488883), INT32_C( -579742459), INT32_C(  505595497), INT32_C( 1976491564),
                            INT32_C( 1357643236), INT32_C( -210153251), INT32_C(-1628647323), INT32_C(-1816082231),
                            INT32_C( 1251469965), INT32_C(-2146681250), INT32_C( 1797992596), INT32_C(-1790080236)),
      easysimd_mm512_set_epi32(INT32_C(-1445633201), INT32_C(-1516803416), INT32_C( 2047415330), INT32_C(  756009385),
                            INT32_C(   38598661), INT32_C(-2128607999), INT32_C(  304234569), INT32_C( 1006199392),
                            INT32_C( 1086388288), INT32_C(-2124328755), INT32_C(-1769435135), INT32_C(   46170304),
                            INT32_C(  167870977), INT32_C(  173648830), INT32_C(-1212193602), INT32_C(   67469316)) },
    { easysimd_mm512_set_epi32(INT32_C( -995130208), INT32_C(-1764606453), INT32_C( -537517512), INT32_C( 1451556674),
                            INT32_C(-2097109774), INT32_C(  404626699), INT32_C( 1345130097), INT32_C( 1798816735),
                            INT32_C(  621374452), INT32_C(  359481722), INT32_C( -121162344), INT32_C(-1051201334),
                            INT32_C( 1869160778), INT32_C( -582139350), INT32_C(  314118274), INT32_C(-1141503487)),
      UINT16_C(47272),
      easysimd_mm512_set_epi32(INT32_C( 1226690931), INT32_C(  775179034), INT32_C(-2065746086), INT32_C(  399353184),
                            INT32_C(  328691430), INT32_C(-1594470117), INT32_C(-1552077762), INT32_C(   88628502),
                            INT32_C(  772052572), INT32_C( 1376748436), INT32_C(-1273427356), INT32_C(  738624056),
                            INT32_C(  647794952), INT32_C(  804576006), INT32_C( 1968895876), INT32_C(  505069248)),
      easysimd_mm512_set_epi32(INT32_C(-1066067632), INT32_C( -638799863), INT32_C(-1513539525), INT32_C(-1037105416),
                            INT32_C(  605705140), INT32_C(-2097483540), INT32_C(   62474077), INT32_C( 2107466991),
                            INT32_C( 1856531921), INT32_C(  781853938), INT32_C( 1472528720), INT32_C( -275942665),
                            INT32_C(  990137373), INT32_C( 1633665081), INT32_C(  480667256), INT32_C( -831347442)),
      easysimd_mm512_set_epi32(INT32_C(-2141190144), INT32_C(-1764606453), INT32_C(  553650209), INT32_C(-1071512936),
                            INT32_C(  604508432), INT32_C(  404626699), INT32_C( 1345130097), INT32_C( 1798816735),
                            INT32_C( 1084778881), INT32_C(  359481722), INT32_C( 1136918800), INT32_C(-1051201334),
                            INT32_C(  419449877), INT32_C( -582139350), INT32_C(  314118274), INT32_C(-1141503487)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_castps_si512(easysimd_mm512_mask_andnot_ps(easysimd_mm512_castsi512_ps(test_vec[i].src), test_vec[i].k, easysimd_mm512_castsi512_ps(test_vec[i].a), easysimd_mm512_castsi512_ps(test_vec[i].b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_andnot_ps");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_andnot_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
  {   UINT16_C(41898),
      easysimd_mm512_set_epi32(INT32_C(-1595502197), INT32_C(-1527248547), INT32_C( 1075363080), INT32_C(-1963744626),
                            INT32_C( -841874568), INT32_C( 1348974030), INT32_C(  932258327), INT32_C(-1638556215),
                            INT32_C(  -69119366), INT32_C(-1406064931), INT32_C( -198162021), INT32_C( -674249080),
                            INT32_C( -972410055), INT32_C(-1112978451), INT32_C( -141156932), INT32_C(-1950860528)),
      easysimd_mm512_set_epi32(INT32_C( -211589013), INT32_C(  652089670), INT32_C( 1378847800), INT32_C(  904957231),
                            INT32_C(-1966320781), INT32_C(-1079187730), INT32_C( 1733727399), INT32_C(-1452272768),
                            INT32_C(-1073785858), INT32_C(  -63492051), INT32_C( 1043637479), INT32_C(-1013855000),
                            INT32_C(  942467481), INT32_C(-1080366077), INT32_C(  642537593), INT32_C(  818463971)),
      easysimd_mm512_set_epi32(INT32_C( 1392601184), INT32_C(          0), INT32_C(  304545840), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( 1078367392), INT32_C(  555745792),
                            INT32_C(   69075332), INT32_C(          0), INT32_C(  168076388), INT32_C(          0),
                            INT32_C(  941934720), INT32_C(          0), INT32_C(    4735041), INT32_C(          0)) },
  {   UINT16_C(54776),
      easysimd_mm512_set_epi32(INT32_C(-1423327830), INT32_C(  463002536), INT32_C( 1170361638), INT32_C( 1439896493),
                            INT32_C( -881601279), INT32_C(  439454207), INT32_C(  642703998), INT32_C( 1761947183),
                            INT32_C( 1210383154), INT32_C( -138151523), INT32_C(  263888472), INT32_C( 2142193967),
                            INT32_C( -741822666), INT32_C(  755920794), INT32_C(-1972313252), INT32_C(-1912811499)),
      easysimd_mm512_set_epi32(INT32_C(-1115388021), INT32_C(  769964125), INT32_C(  418227269), INT32_C(-1388492980),
                            INT32_C(  480660510), INT32_C( 1802844866), INT32_C( -429993967), INT32_C(  538553865),
                            INT32_C( 2013392956), INT32_C(  197176151), INT32_C( 2006567868), INT32_C( 1705115765),
                            INT32_C( 1202543157), INT32_C(-1263572444), INT32_C( 1425580745), INT32_C( 1097283836)),
      easysimd_mm512_set_epi32(INT32_C(  344195585), INT32_C(  610541653), INT32_C(          0), INT32_C(-1473493952),
                            INT32_C(          0), INT32_C( 1631859200), INT32_C(          0), INT32_C(    1605632),
                            INT32_C(  805371916), INT32_C(  134218306), INT32_C( 1879130532), INT32_C(         80),
                            INT32_C(   69554177), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
  {   UINT16_C(25126),
      easysimd_mm512_set_epi32(INT32_C(  633431361), INT32_C( 1663592688), INT32_C( 1399097521), INT32_C( -248917369),
                            INT32_C(-1131808104), INT32_C(  737246109), INT32_C( -548380687), INT32_C(-1607587862),
                            INT32_C(  223712677), INT32_C( -234850179), INT32_C( 1225779292), INT32_C(-1983080521),
                            INT32_C( 1083031306), INT32_C(  479812120), INT32_C( 1659393180), INT32_C( 1062780085)),
      easysimd_mm512_set_epi32(INT32_C(  112954855), INT32_C( 1790377254), INT32_C( 1893295646), INT32_C( -674583179),
                            INT32_C(   15401677), INT32_C( -641918434), INT32_C( -635981818), INT32_C( -342921360),
                            INT32_C( -977229164), INT32_C( -339160274), INT32_C(-1866080556), INT32_C(-1369988401),
                            INT32_C( 1985260264), INT32_C( 1810318993), INT32_C( -324233777), INT32_C(-1229418212)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  144086278), INT32_C(  546924046), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(     499718), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1866459520), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1667630209), INT32_C(-1945861821), INT32_C(          0)) },
  {   UINT16_C(40095),
      easysimd_mm512_set_epi32(INT32_C( -925173403), INT32_C(-1800856604), INT32_C(-1569556909), INT32_C(-2075468293),
                            INT32_C(-1622295859), INT32_C(  800838040), INT32_C(-1261125664), INT32_C(  225560714),
                            INT32_C( -249168174), INT32_C( -785821894), INT32_C(-1322298905), INT32_C( 1919393940),
                            INT32_C(  722048893), INT32_C(  667050909), INT32_C( -741637209), INT32_C(-1063733140)),
      easysimd_mm512_set_epi32(INT32_C(  283702321), INT32_C( 1760938946), INT32_C( 1862161708), INT32_C(  218779454),
                            INT32_C( 1870003832), INT32_C( -776472743), INT32_C(  235320856), INT32_C(-1927493256),
                            INT32_C( 2120699773), INT32_C( 1743164034), INT32_C(   92504126), INT32_C( -822461737),
                            INT32_C(   80989491), INT32_C( -825823244), INT32_C(   23436927), INT32_C(-1677273698)),
      easysimd_mm512_set_epi32(INT32_C(  270533648), INT32_C(          0), INT32_C(          0), INT32_C(  150995460),
                            INT32_C( 1613775920), INT32_C( -805034431), INT32_C(          0), INT32_C(          0),
                            INT32_C(  239206701), INT32_C(          0), INT32_C(          0), INT32_C(-1936188861),
                            INT32_C(   80889858), INT32_C( -939226016), INT32_C(    2366552), INT32_C(  470173074)) },
  {   UINT16_C(25708),
      easysimd_mm512_set_epi32(INT32_C( -419506034), INT32_C(-1634084803), INT32_C(-1791352038), INT32_C( 1397909248),
                            INT32_C( -128853850), INT32_C(-1917410935), INT32_C( 1700830870), INT32_C( 1339604709),
                            INT32_C(-1798365850), INT32_C(  -59209020), INT32_C(  731125713), INT32_C(  630650100),
                            INT32_C(-1338681832), INT32_C(   44002851), INT32_C( -812125291), INT32_C( 1028997312)),
      easysimd_mm512_set_epi32(INT32_C( -213890367), INT32_C( 2021869397), INT32_C( 1653460709), INT32_C(-1583015005),
                            INT32_C(-2111228672), INT32_C(  278487831), INT32_C(-1988085048), INT32_C(-1603254022),
                            INT32_C( 1778423041), INT32_C( 1070290908), INT32_C( 1862134929), INT32_C( 1387107310),
                            INT32_C(-1741926346), INT32_C(  476437588), INT32_C(  -64629687), INT32_C(  821283219)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( 1610745664), INT32_C( 1652935909), INT32_C(          0),
                            INT32_C(          0), INT32_C(  269042198), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(   58938648), INT32_C( 1147791360), INT32_C(          0),
                            INT32_C(  134742054), INT32_C(  476090452), INT32_C(          0), INT32_C(          0)) },
  {   UINT16_C(26454),
      easysimd_mm512_set_epi32(INT32_C(-1766483567), INT32_C(  -97069133), INT32_C( -984184350), INT32_C( -103594411),
                            INT32_C( 1542851117), INT32_C(  476137043), INT32_C( -197399951), INT32_C(-1770261666),
                            INT32_C(  -47794230), INT32_C( -491438206), INT32_C( -344435807), INT32_C(  255371302),
                            INT32_C( -725452804), INT32_C(  159027945), INT32_C(-1412516432), INT32_C( -472096495)),
      easysimd_mm512_set_epi32(INT32_C(  939183992), INT32_C(   45898803), INT32_C( -707307552), INT32_C( -411975944),
                            INT32_C(  630779143), INT32_C( 1898376282), INT32_C( 2124829976), INT32_C( -114883081),
                            INT32_C( 2093795280), INT32_C(-1982561427), INT32_C(  598306044), INT32_C( 1635474930),
                            INT32_C(-1398853653), INT32_C( 1652658661), INT32_C(-1858170883), INT32_C(  995216280)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(    8914944), INT32_C(  276911616), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1627826184), INT32_C(  176428296), INT32_C( 1761871009),
                            INT32_C(          0), INT32_C(  155222125), INT32_C(          0), INT32_C( 1615025616),
                            INT32_C(          0), INT32_C( 1652621572), INT32_C(  271583821), INT32_C(          0)) },
  {   UINT16_C(31670),
      easysimd_mm512_set_epi32(INT32_C(-1612308895), INT32_C( -722700317), INT32_C( 1003499766), INT32_C(  814072246),
                            INT32_C( 2008726943), INT32_C( 1223905210), INT32_C( -618135276), INT32_C(-2049729375),
                            INT32_C(  595839117), INT32_C( -226508565), INT32_C( 1598449683), INT32_C( -514630984),
                            INT32_C(  658541354), INT32_C(  567151600), INT32_C(  -71044409), INT32_C(-1688131700)),
      easysimd_mm512_set_epi32(INT32_C( 1317588071), INT32_C(-1153324271), INT32_C( 2046542506), INT32_C(  623240678),
                            INT32_C(  -39480028), INT32_C(  -33815034), INT32_C( 2056788636), INT32_C( 2095887515),
                            INT32_C( -281654456), INT32_C(-1621887341), INT32_C( 1362159003), INT32_C( 1103094461),
                            INT32_C( 1716020502), INT32_C(  102069928), INT32_C(  474901863), INT32_C(  367619581)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  721520656), INT32_C( 1076609544), INT32_C(   85992512),
                            INT32_C(-2012938208), INT32_C(          0), INT32_C(  546314888), INT32_C( 2016157722),
                            INT32_C( -869006528), INT32_C(          0), INT32_C(    3179912), INT32_C(   11313157),
                            INT32_C(          0), INT32_C(  101807112), INT32_C(   67898656), INT32_C(          0)) },
  {   UINT16_C(49857),
      easysimd_mm512_set_epi32(INT32_C(-2067220018), INT32_C( 1805947847), INT32_C( 2110487322), INT32_C( 1074104919),
                            INT32_C(-1112398120), INT32_C(  225474260), INT32_C( -545045472), INT32_C( -824857753),
                            INT32_C( -338758362), INT32_C(-1789466141), INT32_C( 1713747474), INT32_C(  808725130),
                            INT32_C( 1298412949), INT32_C(  260904797), INT32_C(  457183382), INT32_C( 2009286767)),
      easysimd_mm512_set_epi32(INT32_C(-2042099265), INT32_C( 1225391956), INT32_C( -841393362), INT32_C( -744679138),
                            INT32_C(  -30361081), INT32_C( 1490708305), INT32_C( 1603942577), INT32_C(-1226711411),
                            INT32_C( -720257963), INT32_C(  876066124), INT32_C( 1546499669), INT32_C( 1636147146),
                            INT32_C(    3608382), INT32_C( -404260643), INT32_C( 1874947312), INT32_C(-2040485747)),
      easysimd_mm512_set_epi32(INT32_C(   33554481), INT32_C(     617488), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(    1587345), INT32_C(          0),
                            INT32_C(  336660561), INT32_C(  539038732), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-2145345408)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_castps_si512(easysimd_mm512_maskz_andnot_ps(test_vec[i].k, easysimd_mm512_castsi512_ps(test_vec[i].a), easysimd_mm512_castsi512_ps(test_vec[i].b)));
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_andnot_ps");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_andnot_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
     { easysimd_mm512_set_epi64(INT64_C( -343739447634695407), INT64_C( 6094193684923690615),
                            INT64_C(-7040634603669948000), INT64_C( 8872386007247991164),
                            INT64_C(-8252638392294099885), INT64_C( 6601029892750146432),
                            INT64_C( 7279352193089347864), INT64_C( 8662714795122682384)),
      UINT8_C( 55),
      easysimd_mm512_set_epi64(INT64_C(  599279934641474098), INT64_C( 7845412443385721442),
                            INT64_C(-2777121475968104316), INT64_C( 7314283605033033979),
                            INT64_C(-8886012248836695508), INT64_C( 3313585428802692877),
                            INT64_C(-1598467827595787694), INT64_C(-5223130400950906727)),
      easysimd_mm512_set_epi64(INT64_C( 4132377007718714638), INT64_C( 6722749807664954665),
                            INT64_C(-6834862606673078980), INT64_C(-8159527519597393212),
                            INT64_C( 1169770534052573165), INT64_C(-8496887971116687127),
                            INT64_C( 4795256091623648748), INT64_C( -534912108587925882)),
      easysimd_mm512_set_epi64(INT64_C( -343739447634695407), INT64_C( 6094193684923690615),
                            INT64_C( 2305860618582136120), INT64_C(-8484217854339937788),
                            INT64_C(-8252638392294099885), INT64_C(-9079040208459267872),
                            INT64_C(  147532470349213100), INT64_C( 5192659463408496134)) },
    { easysimd_mm512_set_epi64(INT64_C( 1137601381159569274), INT64_C( 3083515373590209262),
                            INT64_C( 7172644931946125494), INT64_C( 7709434742472783251),
                            INT64_C(-5570954806909339658), INT64_C( -271406020759376737),
                            INT64_C( 4799674771715911578), INT64_C(-1218830816677094379)),
      UINT8_C( 50),
      easysimd_mm512_set_epi64(INT64_C(-2731162171219972563), INT64_C( 4361435470291369786),
                            INT64_C( 4372980053959095777), INT64_C(-4964365409406827474),
                            INT64_C(-4887932848327267276), INT64_C( 5394909549222414797),
                            INT64_C( 8601793944421926823), INT64_C( 1320541430862898557)),
      easysimd_mm512_set_epi64(INT64_C( 3987127999885683210), INT64_C(  232842063033182789),
                            INT64_C(-8565159867474411189), INT64_C( 6112914526494565862),
                            INT64_C( 3462295888398647957), INT64_C( 2362909626677485241),
                            INT64_C( 1401365959932181466), INT64_C( 2959935140000245037)),
      easysimd_mm512_set_epi64(INT64_C( 1137601381159569274), INT64_C( 3083515373590209262),
                            INT64_C(-9151314120547500022), INT64_C( 4955199880194171328),
                            INT64_C(-5570954806909339658), INT64_C( -271406020759376737),
                            INT64_C(    9008205654194264), INT64_C(-1218830816677094379)) },
    { easysimd_mm512_set_epi64(INT64_C( 4893068556614144973), INT64_C( 8066183844976877919),
                            INT64_C( 8546857359160133238), INT64_C(-8267045803572214233),
                            INT64_C( 8915887943252268838), INT64_C(-4953676046754636494),
                            INT64_C( 1510704893512358974), INT64_C( 9024635443342747538)),
      UINT8_C(216),
      easysimd_mm512_set_epi64(INT64_C(-8634103598278842542), INT64_C(-6155398791521040805),
                            INT64_C( 7775580441978642644), INT64_C(-5899929856226471257),
                            INT64_C( 7028189811487947825), INT64_C(-6189665615261290781),
                            INT64_C(  907536080618458470), INT64_C( 3906704638875451620)),
      easysimd_mm512_set_epi64(INT64_C(-4094490793791238990), INT64_C(-2091977621380611033),
                            INT64_C( 2043918654743067438), INT64_C(-4515408626818342672),
                            INT64_C( 1196379185714011362), INT64_C( 1607300510948935937),
                            INT64_C( 1394814499359692419), INT64_C(-5198396047694847294)),
      easysimd_mm512_set_epi64(INT64_C( 5116214864772401312), INT64_C( 4639917106950488100),
                            INT64_C( 8546857359160133238), INT64_C( 4701761038926749776),
                            INT64_C( 1158058982756417730), INT64_C(-4953676046754636494),
                            INT64_C( 1510704893512358974), INT64_C( 9024635443342747538)) },
    { easysimd_mm512_set_epi64(INT64_C( -233515152413640809), INT64_C(-7711023580854835359),
                            INT64_C(  685057037117132470), INT64_C(-1053400672876430250),
                            INT64_C(-6008870355673260365), INT64_C( 6732010747677860150),
                            INT64_C( 7912723632945414242), INT64_C( 6629652157771519554)),
      UINT8_C(  7),
      easysimd_mm512_set_epi64(INT64_C(  418428539766329360), INT64_C( 1870466273027415797),
                            INT64_C( 7044646027925455043), INT64_C(-7541966937157619960),
                            INT64_C(-4455685474515493219), INT64_C(-3587901153898980536),
                            INT64_C( 5978767859636931605), INT64_C( 1520054098233920669)),
      easysimd_mm512_set_epi64(INT64_C( 3839280895408034825), INT64_C(-8206971788365754506),
                            INT64_C(-4439851259277562681), INT64_C(-6789849238744039634),
                            INT64_C(-1659448540825770878), INT64_C( 2745935889893417490),
                            INT64_C( 3715019098340555278), INT64_C(-7036562755259908130)),
      easysimd_mm512_set_epi64(INT64_C( -233515152413640809), INT64_C(-7711023580854835359),
                            INT64_C(  685057037117132470), INT64_C(-1053400672876430250),
                            INT64_C(-6008870355673260365), INT64_C( 2308798513734279186),
                            INT64_C( 2379631298345005578), INT64_C(-8484488882121788606)) },
    { easysimd_mm512_set_epi64(INT64_C(-2379770324367148032), INT64_C(  269951545548960285),
                            INT64_C(-5915450755405613469), INT64_C( 4377769456724035257),
                            INT64_C( 4963028952577306253), INT64_C( 5031417887689077714),
                            INT64_C( 5062535597864084892), INT64_C(-8442033713738522560)),
      UINT8_C(129),
      easysimd_mm512_set_epi64(INT64_C(-6217210315706132893), INT64_C(-5326659911006667991),
                            INT64_C( 1028086835571864351), INT64_C(-9190513903150593462),
                            INT64_C(-5132407930629667991), INT64_C( 3081908066365846241),
                            INT64_C( 1991874275422300444), INT64_C(-8267800556778760378)),
      easysimd_mm512_set_epi64(INT64_C(-7195920316169423191), INT64_C( 8855103613986981069),
                            INT64_C(-1079557804828513091), INT64_C(-7716984285220335090),
                            INT64_C(  760039564915644558), INT64_C( 5629267284662877438),
                            INT64_C(-8887844833591355405), INT64_C( 3227154889713027186)),
      easysimd_mm512_set_epi64(INT64_C( 1441983734462308488), INT64_C(  269951545548960285),
                            INT64_C(-5915450755405613469), INT64_C( 4377769456724035257),
                            INT64_C( 4963028952577306253), INT64_C( 5031417887689077714),
                            INT64_C( 5062535597864084892), INT64_C( 2344440558065420336)) },
    { easysimd_mm512_set_epi64(INT64_C(-1632349344831082760), INT64_C(-7746252227037734078),
                            INT64_C( 8307071850644138234), INT64_C(-8586546786041619015),
                            INT64_C(  404139822791089559), INT64_C(-1877631053848650154),
                            INT64_C( 7455727023947545561), INT64_C( 9065509561364139853)),
      UINT8_C(251),
      easysimd_mm512_set_epi64(INT64_C( 6876828378130175291), INT64_C( 4443252594681514716),
                            INT64_C(-6385840203869031352), INT64_C( 6938523062457490065),
                            INT64_C( -791901096126868688), INT64_C( 5787489911096576116),
                            INT64_C(-7854643813663956328), INT64_C( 5967336075130617342)),
      easysimd_mm512_set_epi64(INT64_C( 4146719804671055125), INT64_C( 2252037785239205430),
                            INT64_C( 8454374735321895014), INT64_C(-1381892347656312574),
                            INT64_C(  706165223560180728), INT64_C( -219143018686364756),
                            INT64_C(-1570739878098539061), INT64_C(-1561542974628641964)),
      easysimd_mm512_set_epi64(INT64_C( 2341899294765252612), INT64_C(  162200290615931938),
                            INT64_C( 5769965461718695974), INT64_C(-8318139676533907198),
                            INT64_C(  633958070103335112), INT64_C(-1877631053848650154),
                            INT64_C( 7494284887317302595), INT64_C(-6339871525209882112)) },
    { easysimd_mm512_set_epi64(INT64_C( 9026638934924851598), INT64_C(  230236376028734533),
                            INT64_C( 7791847925691209473), INT64_C( 5636683834883992106),
                            INT64_C( 4666417032316259140), INT64_C(-9020764089960395704),
                            INT64_C( 8213766780006614493), INT64_C(-6694788910086219877)),
      UINT8_C( 70),
      easysimd_mm512_set_epi64(INT64_C(-6498066308492480472), INT64_C( 5728364479291594350),
                            INT64_C(-5884149762497402782), INT64_C( 6387650260207408060),
                            INT64_C(-5128486331429717841), INT64_C( -868619985199698421),
                            INT64_C(-4214853307896141180), INT64_C( -465765039913276151)),
      easysimd_mm512_set_epi64(INT64_C(-5046884246860802318), INT64_C( 1004972136752522438),
                            INT64_C(-2378507232856704687), INT64_C( 5436650347587017589),
                            INT64_C(-2667790265994842517), INT64_C(-2085203105823883971),
                            INT64_C(-5490659216814537620), INT64_C(-8036188446954416194)),
      easysimd_mm512_set_epi64(INT64_C( 9026638934924851598), INT64_C(   36100064221372544),
                            INT64_C( 7791847925691209473), INT64_C( 5636683834883992106),
                            INT64_C( 4666417032316259140), INT64_C(    3892555578216756),
                            INT64_C( 3624274549258028136), INT64_C(-6694788910086219877)) },
    { easysimd_mm512_set_epi64(INT64_C(-5230007765170990668), INT64_C(  846507549899810342),
                            INT64_C(-7111962349683310649), INT64_C( -772191960312616388),
                            INT64_C( 3123285095915363891), INT64_C( 1623466873559833442),
                            INT64_C( -366019171342533610), INT64_C(-2494634274663155684)),
      UINT8_C( 35),
      easysimd_mm512_set_epi64(INT64_C( -975714944549474590), INT64_C( 1049564164032844619),
                            INT64_C( 7303689756219555946), INT64_C(-6372981973137131801),
                            INT64_C(-5489514128660043293), INT64_C(-7367882453491102610),
                            INT64_C( 6699088752588717529), INT64_C( 1411143637466671223)),
      easysimd_mm512_set_epi64(INT64_C(-2226860933844685600), INT64_C( 1920850149099678208),
                            INT64_C( 1690361552489070319), INT64_C( 6660992074283035646),
                            INT64_C(-5836455416301421815), INT64_C(-2339252903384749197),
                            INT64_C(-3541767763730411989), INT64_C( 1379523068058767230)),
      easysimd_mm512_set_epi64(INT64_C(-5230007765170990668), INT64_C(  846507549899810342),
                            INT64_C( 1307192606745607813), INT64_C( -772191960312616388),
                            INT64_C( 3123285095915363891), INT64_C( 1623466873559833442),
                            INT64_C(-9076991572345486814), INT64_C(    9020679430670088)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_castpd_si512(easysimd_mm512_mask_andnot_pd(easysimd_mm512_castsi512_pd(test_vec[i].src), test_vec[i].k, easysimd_mm512_castsi512_pd(test_vec[i].a), easysimd_mm512_castsi512_pd(test_vec[i].b)));
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_andnot_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
  {   UINT8_C(227),
      easysimd_mm512_set_epi64(INT64_C(-4590720219282553470), INT64_C( 7052994564826635717),
                            INT64_C(  102182550423351600), INT64_C( 6550609573293042333),
                            INT64_C(-6537325874213497913), INT64_C( 8955563540957921573),
                            INT64_C( 8228815951810735558), INT64_C(-3823364876013971085)),
      easysimd_mm512_set_epi64(INT64_C( -740720849127296556), INT64_C( -933890699409471481),
                            INT64_C( 5755588500836856312), INT64_C(-7609758858126984395),
                            INT64_C( 5441557991346977587), INT64_C( -960797962792509213),
                            INT64_C(  199203171802884405), INT64_C( 1812346297232380541)),
      easysimd_mm512_set_epi64(INT64_C( 3868594310811889748), INT64_C(-7923482887668088830),
                            INT64_C( 5662415626724180168), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(   54349100887310385), INT64_C( 1226687842802237964)) },
  {   UINT8_C(150),
      easysimd_mm512_set_epi64(INT64_C( 8029427937579490996), INT64_C(-1016228199940301895),
                            INT64_C( 3581869483076202853), INT64_C( 4960784598491720813),
                            INT64_C(-7670184712449022296), INT64_C( 1368687340866524346),
                            INT64_C(   36158962521961508), INT64_C( 1367446093605161437)),
      easysimd_mm512_set_epi64(INT64_C(  210141022168102607), INT64_C(-5660044126052691316),
                            INT64_C( 8952190750537587177), INT64_C(-7520755716476597588),
                            INT64_C( 5025036600597137846), INT64_C( 1371349703128320142),
                            INT64_C( 1157825117202956749), INT64_C( 3947754344252009580)),
      easysimd_mm512_set_epi64(INT64_C(   36187148471246923), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-7845058891591449984),
                            INT64_C(                   0), INT64_C(   72059827504875524),
                            INT64_C( 1157717081520079305), INT64_C(                   0)) },
  {   UINT8_C(206),
      easysimd_mm512_set_epi64(INT64_C(-5447319738796629324), INT64_C( 7573553786407309883),
                            INT64_C( 3210166478679154113), INT64_C( -632818268169935629),
                            INT64_C( 2091039522714659767), INT64_C(-7890721085940980150),
                            INT64_C(-4051485337429119412), INT64_C(-3044005681324007212)),
      easysimd_mm512_set_epi64(INT64_C(-3107571465629414339), INT64_C(-5609659848016607327),
                            INT64_C( 3170884903864138535), INT64_C( 3780264979688453657),
                            INT64_C(-3200960942660399317), INT64_C( 5382084213528122877),
                            INT64_C(-4409193503472949179), INT64_C( 4723837911396640821)),
      easysimd_mm512_set_epi64(INT64_C( 4654611160413260809), INT64_C(-7916109799461190272),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-4426155929590676472), INT64_C( 5224458146534150581),
                            INT64_C(    2608396285542401), INT64_C(                   0)) },
  {   UINT8_C(125),
      easysimd_mm512_set_epi64(INT64_C( 4207278183660861960), INT64_C(-8995945069443043606),
                            INT64_C( 8554253801191868756), INT64_C( 3354059043086044373),
                            INT64_C( 1657475957423553689), INT64_C(-2556137084454595182),
                            INT64_C( 2422681642730518465), INT64_C(-8655840866694392843)),
      easysimd_mm512_set_epi64(INT64_C(-6233706215614972452), INT64_C( 2778576059313358974),
                            INT64_C(  521154595483651590), INT64_C(-2197561166428241391),
                            INT64_C(  751433836641726755), INT64_C( 5984411989878292578),
                            INT64_C(-2128282437357703049), INT64_C( 6129378286910417126)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 2632358518757523476),
                            INT64_C(   74594534185240066), INT64_C(-4539433801636773376),
                            INT64_C(  607176734235951394), INT64_C(  218459912932966496),
                            INT64_C(                   0), INT64_C( 5769015465569462274)) },
  {   UINT8_C( 71),
      easysimd_mm512_set_epi64(INT64_C(-8436437744293223076), INT64_C( -780741249760151942),
                            INT64_C( 4822350614887775462), INT64_C( 2188408541520193917),
                            INT64_C(-3082935350304813722), INT64_C(-5875221946234265673),
                            INT64_C(-5758090656392293952), INT64_C(-3302974504787286903)),
      easysimd_mm512_set_epi64(INT64_C(-7235195547697304884), INT64_C(-2099342694411362386),
                            INT64_C( 6587794971423114743), INT64_C(-8750716550526717441),
                            INT64_C(-6164466580259336301), INT64_C(-5605431759460432480),
                            INT64_C( 8610981953023941155), INT64_C(-5677351707943910012)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(  204210100791970180),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 1153063355977236480),
                            INT64_C( 5152215214393002531), INT64_C( 2383675803176042756)) },
  {   UINT8_C( 16),
      easysimd_mm512_set_epi64(INT64_C(-7454619922298182462), INT64_C(-4477515570225004692),
                            INT64_C( 3259262052820328758), INT64_C(-2323942451066306663),
                            INT64_C(-7533087570752357418), INT64_C(-2748624972946479401),
                            INT64_C(-7594508336042449203), INT64_C(-2829162199669149138)),
      easysimd_mm512_set_epi64(INT64_C(  231920182013128330), INT64_C( 2342360813276731434),
                            INT64_C(-3887471131024015317), INT64_C(-6063668553337722025),
                            INT64_C( 6394528685493045899), INT64_C( 7433558736916574563),
                            INT64_C(-8597186079760918784), INT64_C( 9218943275377121788)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 2323862976450265158),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
  {   UINT8_C(  0),
      easysimd_mm512_set_epi64(INT64_C(-1874339372436527846), INT64_C(-4874669033093832828),
                            INT64_C(-5258762659707925604), INT64_C( 1933045326528420333),
                            INT64_C( 8704229925049171123), INT64_C(-4249956245353677661),
                            INT64_C( 3155017878537816163), INT64_C( 8377752223970655488)),
      easysimd_mm512_set_epi64(INT64_C( 6639157720065498333), INT64_C( 7954402008564552716),
                            INT64_C(  220412799958481097), INT64_C( 3341210828844349470),
                            INT64_C(  930495958757986079), INT64_C(-5593607526362331219),
                            INT64_C( 1220298896193992740), INT64_C( 1285034736351616528)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
  {   UINT8_C( 29),
      easysimd_mm512_set_epi64(INT64_C( 6066305475956667844), INT64_C(-4992917222673652861),
                            INT64_C(-8395584014417236584), INT64_C( -142719058224734896),
                            INT64_C(  509377192188320240), INT64_C( 4417811606371822828),
                            INT64_C( 5101966917722654224), INT64_C(-9124380135803090931)),
      easysimd_mm512_set_epi64(INT64_C( 7887221249293377488), INT64_C( -182605916723991232),
                            INT64_C(  366071292133853300), INT64_C(-7235772882062384424),
                            INT64_C( 2002854046423029286), INT64_C(-3793561946903283248),
                            INT64_C(-8278200760223787155), INT64_C(-6807146722179486859)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(  112882606806164104),
                            INT64_C( 1786257950099998214), INT64_C(-4462918246344945392),
                            INT64_C(                   0), INT64_C( 2341880616049730416)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_castpd_si512(easysimd_mm512_maskz_andnot_pd(test_vec[i].k, easysimd_mm512_castsi512_pd(test_vec[i].a), easysimd_mm512_castsi512_pd(test_vec[i].b)));
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_andnot_si512(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( -335330897), INT32_C( 1860840666), INT32_C( -837102383), INT32_C( 1544121603),
                            INT32_C(  -31451516), INT32_C(  294501250), INT32_C( 1844141610), INT32_C(  711066163),
                            INT32_C( 1032767823), INT32_C(  466876164), INT32_C( 1432923079), INT32_C( -137339965),
                            INT32_C(-1216294439), INT32_C( 1110388055), INT32_C(  754234048), INT32_C( -712839683)),
      easysimd_mm512_set_epi32(INT32_C(  645969883), INT32_C(   45407696), INT32_C(-1431929981), INT32_C(-1744968675),
                            INT32_C( 1491740298), INT32_C( -669732847), INT32_C(-1678703719), INT32_C(-1110558488),
                            INT32_C(-1993251114), INT32_C( 1910816421), INT32_C( 2013403088), INT32_C(  882585036),
                            INT32_C( 1733706468), INT32_C( 1763057771), INT32_C(  303070795), INT32_C( -805966849)),
      easysimd_mm512_set_epi32(INT32_C(   41989712), INT32_C(    1364224), INT32_C(  547629826), INT32_C(-2081128420),
                            INT32_C(   13246474), INT32_C( -938475503), INT32_C(-1844444783), INT32_C(-1785972536),
                            INT32_C(-2144262512), INT32_C( 1612974241), INT32_C(  671224848), INT32_C(     730124),
                            INT32_C( 1079378468), INT32_C(  688914472), INT32_C(  302009355), INT32_C(  175440386)) },
    { easysimd_mm512_set_epi32(INT32_C( 1267266514), INT32_C(-1810114077), INT32_C(-1631053656), INT32_C(-1291259659),
                            INT32_C(-1797405973), INT32_C(-2052491824), INT32_C(  218690610), INT32_C(  434694077),
                            INT32_C(  322569513), INT32_C( -492306370), INT32_C( 1714124310), INT32_C(  757183592),
                            INT32_C( 1904845371), INT32_C( 1921390915), INT32_C( 1219016836), INT32_C( -491589854)),
      easysimd_mm512_set_epi32(INT32_C( -843887215), INT32_C(-1144045392), INT32_C( 1523671305), INT32_C( -687015924),
                            INT32_C( -651771268), INT32_C(-1812069901), INT32_C(  132880464), INT32_C( 1912329512),
                            INT32_C( -208209918), INT32_C(-1079631083), INT32_C( -134611197), INT32_C(-1062410635),
                            INT32_C( -896925558), INT32_C( -559765979), INT32_C( 1912148196), INT32_C( -437846049)),
      easysimd_mm512_set_epi32(INT32_C(-2077029375), INT32_C(  734265360), INT32_C( 1074880769), INT32_C( 1141114888),
                            INT32_C( 1226965012), INT32_C(  307626019), INT32_C(   48433216), INT32_C( 1611862016),
                            INT32_C( -528156670), INT32_C(  486941441), INT32_C(-1848604415), INT32_C(-1064549867),
                            INT32_C(-1979580288), INT32_C(-1944017372), INT32_C(  827392096), INT32_C(   88346845)) },
    { easysimd_mm512_set_epi32(INT32_C(  451034606), INT32_C(  160382101), INT32_C(-1268862602), INT32_C(  782115678),
                            INT32_C(-1160318793), INT32_C( -575355195), INT32_C( 1432838242), INT32_C(-2114154695),
                            INT32_C(-1020410376), INT32_C( -714076046), INT32_C(-1407849113), INT32_C(  996241684),
                            INT32_C(  481606881), INT32_C(-1834956523), INT32_C(  493396975), INT32_C(-1084672800)),
      easysimd_mm512_set_epi32(INT32_C( 1458493934), INT32_C( 1051105030), INT32_C( -836083742), INT32_C( 1407748874),
                            INT32_C(-1387312486), INT32_C(  776481471), INT32_C(  275093143), INT32_C( -137438390),
                            INT32_C( 1860284960), INT32_C(  540502552), INT32_C( 1411461258), INT32_C( 1517918194),
                            INT32_C( -266161178), INT32_C( 1269265702), INT32_C(  809771495), INT32_C(-1968711037)),
      easysimd_mm512_set_epi32(INT32_C( 1141686272), INT32_C(  908100354), INT32_C( 1243631232), INT32_C( 1365280768),
                            INT32_C(   84478472), INT32_C(  575154234), INT32_C(      38549), INT32_C( 1979863106),
                            INT32_C(  750793216), INT32_C(  537356808), INT32_C( 1344278664), INT32_C( 1075349218),
                            INT32_C( -536739066), INT32_C( 1225208866), INT32_C(  537133056), INT32_C(   10930691)) },
    { easysimd_mm512_set_epi32(INT32_C(-1562592645), INT32_C(  -32255724), INT32_C( -923416118), INT32_C(-2134713284),
                            INT32_C(-1313323965), INT32_C(-1729518909), INT32_C( 1286411285), INT32_C( -376910154),
                            INT32_C(-1786193108), INT32_C(-2035089818), INT32_C( 1552020826), INT32_C(  726998554),
                            INT32_C( 1864619074), INT32_C( 1828024315), INT32_C( -824341738), INT32_C(-1420030579)),
      easysimd_mm512_set_epi32(INT32_C( 1087836695), INT32_C(-2094233976), INT32_C( 1148487684), INT32_C(-1514127182),
                            INT32_C( -524459384), INT32_C(  725104708), INT32_C( 1787286694), INT32_C(-1533684832),
                            INT32_C(   46575098), INT32_C( 2086853653), INT32_C(  815292575), INT32_C(-1270435744),
                            INT32_C( 2014177347), INT32_C( 1099600134), INT32_C( -622983952), INT32_C(  822011154)),
      easysimd_mm512_set_epi32(INT32_C( 1073938436), INT32_C(   19662472), INT32_C(   67108868), INT32_C(  620757122),
                            INT32_C( 1074078344), INT32_C(  588257284), INT32_C(  570605730), INT32_C(   68489472),
                            INT32_C(   38151378), INT32_C( 2017460241), INT32_C(  538443909), INT32_C(-1811767200),
                            INT32_C(  269232129), INT32_C(   17469444), INT32_C(  268567776), INT32_C(  279109650)) },
    { easysimd_mm512_set_epi32(INT32_C(-1657115762), INT32_C( 1585840022), INT32_C(-1070898703), INT32_C( 1022031619),
                            INT32_C(-1380717315), INT32_C( 1086658406), INT32_C( -124039065), INT32_C(-1974944947),
                            INT32_C( 2044249149), INT32_C( 1638783653), INT32_C( 1466240446), INT32_C(-1803146403),
                            INT32_C( 1060682707), INT32_C(-1592428518), INT32_C(  156586666), INT32_C( -266957088)),
      easysimd_mm512_set_epi32(INT32_C( -703454581), INT32_C(  797686885), INT32_C( 1723425278), INT32_C( -158454369),
                            INT32_C(-1043830066), INT32_C(  709622512), INT32_C(-2136296570), INT32_C( -863350926),
                            INT32_C( 1844461284), INT32_C(  -21472306), INT32_C(-1932483198), INT32_C(-1320584016),
                            INT32_C( -370591173), INT32_C( -330170023), INT32_C( -975385097), INT32_C( -654562432)),
      easysimd_mm512_set_epi32(INT32_C( 1107296257), INT32_C(  554303585), INT32_C(  646971406), INT32_C(-1039923044),
                            INT32_C( 1078460930), INT32_C(  705357968), INT32_C(    2139008), INT32_C( 1149387826),
                            INT32_C(   69221056), INT32_C(-1643118262), INT32_C(-2003787776), INT32_C(  558453920),
                            INT32_C(-1061093336), INT32_C( 1279394113), INT32_C( -997683883), INT32_C(  149430528)) },
    { easysimd_mm512_set_epi32(INT32_C(  962558787), INT32_C(-1212292378), INT32_C(-1698562444), INT32_C(-1456708578),
                            INT32_C( 1605522258), INT32_C(-1389853810), INT32_C(  605095260), INT32_C(  449573803),
                            INT32_C(-1932095036), INT32_C( 1214045264), INT32_C(-1966228541), INT32_C(  484352026),
                            INT32_C(-1251622562), INT32_C(   97048183), INT32_C( 1801957969), INT32_C(   39148591)),
      easysimd_mm512_set_epi32(INT32_C( 1144673524), INT32_C(-1837539909), INT32_C(-1995926176), INT32_C( -775830454),
                            INT32_C( 1197039500), INT32_C(  605086417), INT32_C(-1681915928), INT32_C(-1694227594),
                            INT32_C(  250277648), INT32_C( 1517650405), INT32_C( -529860796), INT32_C(  319331129),
                            INT32_C( 1337610221), INT32_C( -515158609), INT32_C(-1958759875), INT32_C(  480005412)),
      easysimd_mm512_set_epi32(INT32_C( 1142949044), INT32_C(    4201753), INT32_C(   17301760), INT32_C( 1354858560),
                            INT32_C(    4787340), INT32_C(    1073233), INT32_C(-1683031392), INT32_C(-2130444204),
                            INT32_C(   36204048), INT32_C(  304152997), INT32_C( 1612858372), INT32_C(   50338593),
                            INT32_C( 1251610273), INT32_C( -536671864), INT32_C(-2145910740), INT32_C(  478675200)) },
    { easysimd_mm512_set_epi32(INT32_C(  477799556), INT32_C(  718106947), INT32_C( -702434720), INT32_C(  911156446),
                            INT32_C(  692922531), INT32_C( -634559193), INT32_C( -541024501), INT32_C(    6957260),
                            INT32_C(  891904501), INT32_C( 1674261328), INT32_C(  463285837), INT32_C(  465636281),
                            INT32_C( -567453998), INT32_C( -675807734), INT32_C( 1242869264), INT32_C(-2003535835)),
      easysimd_mm512_set_epi32(INT32_C( -440269466), INT32_C( 1069561863), INT32_C( -850138274), INT32_C( 1324108467),
                            INT32_C(  996083706), INT32_C(-1741332408), INT32_C(-1720688024), INT32_C( -195389802),
                            INT32_C( -122163269), INT32_C(-1678986062), INT32_C( -261742027), INT32_C(  147621305),
                            INT32_C( 1928957095), INT32_C(  647911914), INT32_C(-1231783784), INT32_C(-1597793099)),
      easysimd_mm512_set_epi32(INT32_C( -511704734), INT32_C(  352323588), INT32_C(  156387614), INT32_C( 1218464289),
                            INT32_C(  303171416), INT32_C(    1048648), INT32_C(    3170400), INT32_C( -200239598),
                            INT32_C( -929657334), INT32_C(-1742437214), INT32_C( -530448336), INT32_C(     820224),
                            INT32_C(  550537253), INT32_C(  537395680), INT32_C(-1266659192), INT32_C(  541295760)) },
    { easysimd_mm512_set_epi32(INT32_C(-1322452749), INT32_C(-1191485380), INT32_C(   61071601), INT32_C( -255981709),
                            INT32_C( 1745472557), INT32_C( 1521357726), INT32_C(-1111842070), INT32_C( 1783291089),
                            INT32_C(  718609371), INT32_C( -553071779), INT32_C(-1373014967), INT32_C(  751334079),
                            INT32_C( -828271800), INT32_C(-1578484948), INT32_C(-1597074675), INT32_C(  393018558)),
      easysimd_mm512_set_epi32(INT32_C(-1722624236), INT32_C( -955857282), INT32_C( 1790216473), INT32_C( -762838785),
                            INT32_C( -108799681), INT32_C( -975838651), INT32_C( 1961237228), INT32_C(   52752901),
                            INT32_C(-1440122977), INT32_C(-1167835972), INT32_C( 1345250484), INT32_C( 2101674065),
                            INT32_C( -149671798), INT32_C(  738167968), INT32_C( -764040824), INT32_C( -514982245)),
      easysimd_mm512_set_epi32(INT32_C(  139593476), INT32_C( 1191478850), INT32_C( 1746143496), INT32_C(   33554572),
                            INT32_C(-1853746926), INT32_C(-2058231743), INT32_C( 1078204420), INT32_C(   19145220),
                            INT32_C(-2144836604), INT32_C(  543437984), INT32_C( 1342603444), INT32_C( 1359282240),
                            INT32_C(  823402626), INT32_C(  169182336), INT32_C( 1378953344), INT32_C( -536804863)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_andnot_si512(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_andnot_si512");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_andnot_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 1786505147), INT32_C(  366806262), INT32_C(-1595474360), INT32_C( -741125130),
                            INT32_C(  623580589), INT32_C( 1819639708), INT32_C(-1998267151), INT32_C(   54696203),
                            INT32_C( 1230356730), INT32_C( -528215990), INT32_C(-1085976265), INT32_C(  -88891472),
                            INT32_C(  263402075), INT32_C( 2072408294), INT32_C( 1041613232), INT32_C(-1299496488)),
      easysimd_mm512_set_epi32(INT32_C(  634085978), INT32_C( 1880231468), INT32_C( 1457044755), INT32_C( -852481645),
                            INT32_C( 1274177455), INT32_C( -223885439), INT32_C(  582318482), INT32_C( 1515067636),
                            INT32_C(-1348943060), INT32_C(   88850487), INT32_C(-1067534176), INT32_C( 1770437803),
                            INT32_C(-1608407464), INT32_C(-1335624696), INT32_C( 1637100454), INT32_C(-1783030263)),
      easysimd_mm512_set_epi32(INT32_C(   92274752), INT32_C( 1610746376), INT32_C( 1444413715), INT32_C(  203433985),
                            INT32_C( 1255170562), INT32_C(-1836941311), INT32_C(  571547906), INT32_C( 1477181684),
                            INT32_C(-1501035772), INT32_C(   88850485), INT32_C( 1075488896), INT32_C(   17047563),
                            INT32_C(-1610578944), INT32_C(-2141090808), INT32_C( 1098918406), INT32_C(   87039489)) },
    { easysimd_mm512_set_epi32(INT32_C(  923576423), INT32_C(-1078925154), INT32_C( -430557576), INT32_C(-1684773193),
                            INT32_C(-1179101215), INT32_C(-1985153431), INT32_C(  584718967), INT32_C( -112765469),
                            INT32_C( 1515864234), INT32_C(-1118210252), INT32_C(  931438007), INT32_C( -352031421),
                            INT32_C( 1134370188), INT32_C( 1556623900), INT32_C(   57329867), INT32_C(  254759017)),
      easysimd_mm512_set_epi32(INT32_C( -300442460), INT32_C( -893266841), INT32_C(-1015236925), INT32_C(  262163323),
                            INT32_C( 2095940386), INT32_C( 1009617335), INT32_C(  458760718), INT32_C(-1732442867),
                            INT32_C( 1273945161), INT32_C(  946706622), INT32_C( 1469023509), INT32_C(-2064451876),
                            INT32_C( -256697390), INT32_C(-1934774398), INT32_C(  433298181), INT32_C( -530351918)),
      easysimd_mm512_set_epi32(INT32_C( -938273664), INT32_C( 1078006881), INT32_C(   19431555), INT32_C(   69208392),
                            INT32_C( 1145405442), INT32_C(  872483222), INT32_C(  419438600), INT32_C(   12058636),
                            INT32_C(   27565121), INT32_C(    2394250), INT32_C( 1074462720), INT32_C(   83006108),
                            INT32_C(-1339948462), INT32_C(-2145022590), INT32_C(  412160772), INT32_C( -532666222)) },
    { easysimd_mm512_set_epi32(INT32_C(  835311518), INT32_C(  593132209), INT32_C(-1205845883), INT32_C(-2103435972),
                            INT32_C(  331121937), INT32_C(-1122763027), INT32_C(  -11044623), INT32_C( 1217358106),
                            INT32_C(  899389553), INT32_C(   61750829), INT32_C(-1644418892), INT32_C( 1179256254),
                            INT32_C( -236468269), INT32_C( -666751062), INT32_C( -733547571), INT32_C( 2125570021)),
      easysimd_mm512_set_epi32(INT32_C(-1443754597), INT32_C( 1972174992), INT32_C(-2074962423), INT32_C( -531291976),
                            INT32_C( 1382830722), INT32_C( -282269267), INT32_C( 1453780297), INT32_C(  363272438),
                            INT32_C( 1819778130), INT32_C(-1488646809), INT32_C( 1000774887), INT32_C( 2075973242),
                            INT32_C(  251762527), INT32_C(  254090322), INT32_C( -106442053), INT32_C(-1147166459)),
      easysimd_mm512_set_epi32(INT32_C(-2009987071), INT32_C( 1418002432), INT32_C(   72521224), INT32_C( 1616191616),
                            INT32_C( 1077957250), INT32_C( 1110180096), INT32_C(   10519816), INT32_C(  354423012),
                            INT32_C( 1214514178), INT32_C(-1539243710), INT32_C(  570594371), INT32_C(  968151616),
                            INT32_C(  234885132), INT32_C(  119869520), INT32_C(  698417202), INT32_C(-2129780736)) },
    { easysimd_mm512_set_epi32(INT32_C( 1259282838), INT32_C( -167567006), INT32_C( 1470440257), INT32_C(-1702928569),
                            INT32_C(-1493129242), INT32_C( -361616020), INT32_C( 1148861436), INT32_C(-2140586026),
                            INT32_C(-1901343726), INT32_C( 1258604211), INT32_C( 1382183555), INT32_C(  464481172),
                            INT32_C(   87817013), INT32_C(  -25672201), INT32_C(-1647580547), INT32_C( -833959607)),
      easysimd_mm512_set_epi32(INT32_C( -711482206), INT32_C(-1110405208), INT32_C(  -55795162), INT32_C(-1789106875),
                            INT32_C(-1077987504), INT32_C( 2002242576), INT32_C(  879044440), INT32_C(  728498187),
                            INT32_C( -580810324), INT32_C(-1054241155), INT32_C(  416673383), INT32_C( 1924176623),
                            INT32_C( 1323235160), INT32_C(  659292758), INT32_C(-2101310960), INT32_C( 1303315999)),
      easysimd_mm512_set_epi32(INT32_C(-1802468320), INT32_C(  164662920), INT32_C(-1475837914), INT32_C(   83894272),
                            INT32_C(  415174672), INT32_C(  352698384), INT32_C(  805635072), INT32_C(  721682441),
                            INT32_C( 1363149228), INT32_C(-2144794548), INT32_C(  144018532), INT32_C( 1611698283),
                            INT32_C( 1254360136), INT32_C(   17039872), INT32_C(   33554432), INT32_C(   27592214)) },
    { easysimd_mm512_set_epi32(INT32_C( 1317706320), INT32_C( 1095937634), INT32_C(-2042379654), INT32_C( -425062813),
                            INT32_C(-1422676870), INT32_C(-1972727484), INT32_C( 1448617643), INT32_C( 1446030445),
                            INT32_C(-1203372071), INT32_C( 1257548767), INT32_C(   95515950), INT32_C(  288075556),
                            INT32_C( -562902724), INT32_C( 1866018725), INT32_C( -140491543), INT32_C( -853598261)),
      easysimd_mm512_set_epi32(INT32_C(-1862602245), INT32_C( 1299263323), INT32_C(-1100697239), INT32_C(-1165132701),
                            INT32_C(-1312528679), INT32_C(-2057483334), INT32_C(-2116201571), INT32_C(-1004874347),
                            INT32_C( -792865239), INT32_C(  167838662), INT32_C(-1158285246), INT32_C(  788705850),
                            INT32_C(-1470598876), INT32_C( -300747724), INT32_C( -732019428), INT32_C(-1060860437)),
      easysimd_mm512_set_epi32(INT32_C(-1871683157), INT32_C(  203489561), INT32_C(  941895937), INT32_C(  403009536),
                            INT32_C(  281297537), INT32_C(   85281466), INT32_C(-2121969388), INT32_C(-2146807408),
                            INT32_C( 1085800480), INT32_C(      65536), INT32_C(-1169913792), INT32_C(  771756058),
                            INT32_C(  537407488), INT32_C(-2147299312), INT32_C(    6160660), INT32_C(   12615712)) },
    { easysimd_mm512_set_epi32(INT32_C(  782435122), INT32_C( 1862046610), INT32_C( 2063073020), INT32_C(-2039040635),
                            INT32_C( 1210624813), INT32_C( 1482889596), INT32_C(-1693737823), INT32_C( -742414353),
                            INT32_C(  769657412), INT32_C(-1049696640), INT32_C(  237587070), INT32_C( 1546361918),
                            INT32_C( -364413489), INT32_C(-1858108224), INT32_C(-1524047519), INT32_C( -892082969)),
      easysimd_mm512_set_epi32(INT32_C( 1276319466), INT32_C( -348382036), INT32_C(  -54124638), INT32_C(-1613416797),
                            INT32_C( -277896350), INT32_C(-1555914365), INT32_C( 1602672291), INT32_C(  612591504),
                            INT32_C(-1670560036), INT32_C( 2118020891), INT32_C(-1204159467), INT32_C(  299945581),
                            INT32_C( 1470077526), INT32_C(-1901456818), INT32_C( 1982811443), INT32_C(  366998615)),
      easysimd_mm512_set_epi32(INT32_C( 1074795720), INT32_C(-2130703316), INT32_C(-2080374526), INT32_C(  427885090),
                            INT32_C(-1488519102), INT32_C(-1560239997), INT32_C( 1149518338), INT32_C(  603996176),
                            INT32_C(-1878178664), INT32_C( 1041238299), INT32_C(-1341082623), INT32_C(   29377089),
                            INT32_C(  362316304), INT32_C(  243274254), INT32_C( 1376193554), INT32_C(  353112080)) },
    { easysimd_mm512_set_epi32(INT32_C( -664438730), INT32_C( 1158162569), INT32_C(-1048438639), INT32_C(  819552403),
                            INT32_C(  486427093), INT32_C(-1267830843), INT32_C( 1178270581), INT32_C(-1348447676),
                            INT32_C( -981472284), INT32_C( 1962298807), INT32_C( -393093452), INT32_C(-1754911100),
                            INT32_C(-1506604227), INT32_C( -220324223), INT32_C(  856278899), INT32_C(   15706156)),
      easysimd_mm512_set_epi32(INT32_C( -689282393), INT32_C( -261985647), INT32_C(-1390325708), INT32_C(-1552766747),
                            INT32_C(-1576064212), INT32_C( -185898645), INT32_C(-1798232738), INT32_C( -401409831),
                            INT32_C( 1975803231), INT32_C( 1826250001), INT32_C(-1038398890), INT32_C( -306355124),
                            INT32_C(-1154269982), INT32_C( -209110535), INT32_C(-2033491342), INT32_C( -971905248)),
      easysimd_mm512_set_epi32(INT32_C(  109707905), INT32_C(-1335737840), INT32_C(  740376612), INT32_C(-2094888860),
                            INT32_C(-1576984024), INT32_C( 1082196010), INT32_C(-1866398710), INT32_C( 1074964633),
                            INT32_C(  813700123), INT32_C(  134746112), INT32_C(   34209858), INT32_C( 1754873928),
                            INT32_C(  419443906), INT32_C(   16851320), INT32_C(-2067652608), INT32_C( -972011776)) },
    { easysimd_mm512_set_epi32(INT32_C(-1519344071), INT32_C( 1556822852), INT32_C(-1382496853), INT32_C( -624683333),
                            INT32_C( 1477411394), INT32_C( -704833096), INT32_C(-1957423151), INT32_C( -471773069),
                            INT32_C( 1263493389), INT32_C( 2117955521), INT32_C(-1143959230), INT32_C( -832581030),
                            INT32_C(-1273834890), INT32_C( -392148704), INT32_C( 1764655366), INT32_C( -721713055)),
      easysimd_mm512_set_epi32(INT32_C(-1396008954), INT32_C( -651865449), INT32_C(  452267102), INT32_C( -741136221),
                            INT32_C( 1539744858), INT32_C(-2014766256), INT32_C(-1095604449), INT32_C(-1527666044),
                            INT32_C( -826073132), INT32_C(   -8340331), INT32_C( 1447376741), INT32_C( 1608478316),
                            INT32_C( 1253487795), INT32_C( 2056029052), INT32_C( -880457902), INT32_C( -691872315)),
      easysimd_mm512_set_epi32(INT32_C(  143267846), INT32_C(-2128330605), INT32_C(  308611156), INT32_C(   18032640),
                            INT32_C(   62923800), INT32_C(   33555008), INT32_C(  883056654), INT32_C(   68201092),
                            INT32_C(-2071978288), INT32_C(-2122280940), INT32_C( 1141188133), INT32_C(  293612580),
                            INT32_C( 1252271233), INT32_C(  302809692), INT32_C(-2105457072), INT32_C(   33576324)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_andnot_epi32(test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_andnot_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C(-1056724565), INT32_C( 1525326722), INT32_C( -860629095), INT32_C( 1674345138),
                            INT32_C( -780517906), INT32_C(-1953060088), INT32_C(-1307294727), INT32_C(-1463687440),
                            INT32_C( -675695615), INT32_C( 1308561010), INT32_C(  639253006), INT32_C( -651243687),
                            INT32_C( -612977662), INT32_C(  549809264), INT32_C(  644920842), INT32_C( 1882016760)),
      UINT16_C(49595),
      easysimd_mm512_set_epi32(INT32_C( 1189938329), INT32_C( 1797835672), INT32_C(  297527233), INT32_C( 1889709809),
                            INT32_C( -409509393), INT32_C(  994472936), INT32_C( -666015338), INT32_C( -260985600),
                            INT32_C(  750956055), INT32_C( 1716384261), INT32_C(-1960820967), INT32_C(  548004165),
                            INT32_C( 1158678479), INT32_C( 1692264694), INT32_C(  789910754), INT32_C(-1468927401)),
      easysimd_mm512_set_epi32(INT32_C( -428021677), INT32_C( 2072345032), INT32_C(-1760567882), INT32_C( -446864592),
                            INT32_C( 1299362117), INT32_C( 1402678741), INT32_C( -667918085), INT32_C( 1296019907),
                            INT32_C(-1260791809), INT32_C( 1231406525), INT32_C( -414651973), INT32_C( 1577314987),
                            INT32_C( 1453608195), INT32_C( 1199168765), INT32_C( 1764761558), INT32_C(-1053144882)),
      easysimd_mm512_set_epi32(INT32_C(-1609504702), INT32_C(  277164096), INT32_C( -860629095), INT32_C( 1674345138),
                            INT32_C( -780517906), INT32_C(-1953060088), INT32_C(-1307294727), INT32_C(  219025603),
                            INT32_C(-1877388824), INT32_C( 1308561010), INT32_C( 1682489506), INT32_C( 1577192106),
                            INT32_C(  312756224), INT32_C(  549809264), INT32_C( 1075841812), INT32_C( 1091176584)) },
    { easysimd_mm512_set_epi32(INT32_C( -922226792), INT32_C( 1177794317), INT32_C(-1987384202), INT32_C(  817250921),
                            INT32_C( 1296662639), INT32_C(   64131133), INT32_C(-1048693918), INT32_C( 1748498328),
                            INT32_C( -392119279), INT32_C(-1074948281), INT32_C( 1219088991), INT32_C(  346956559),
                            INT32_C( -778487174), INT32_C( 2030262893), INT32_C( -325938509), INT32_C( 2088865417)),
      UINT16_C(43842),
      easysimd_mm512_set_epi32(INT32_C( -911190750), INT32_C( -150954698), INT32_C(-2108244068), INT32_C( -219871492),
                            INT32_C(  954142226), INT32_C( -657696450), INT32_C(  -40171606), INT32_C(  523799369),
                            INT32_C(-1984820679), INT32_C( -352318109), INT32_C( 1527484465), INT32_C( 1078897849),
                            INT32_C( -979432773), INT32_C( -222789591), INT32_C( -127333602), INT32_C( 1547833861)),
      easysimd_mm512_set_epi32(INT32_C( 1706771302), INT32_C(-1876132949), INT32_C( -300867745), INT32_C(-1574226708),
                            INT32_C(  909541228), INT32_C(-1473521559), INT32_C(-2035272090), INT32_C( -843632177),
                            INT32_C(-1617888467), INT32_C( -960934829), INT32_C( -805571508), INT32_C( -811280081),
                            INT32_C(-1033748670), INT32_C(-1374688928), INT32_C( -924697051), INT32_C( -396703151)),
      easysimd_mm512_set_epi32(INT32_C(  604701252), INT32_C( 1177794317), INT32_C( 1812004931), INT32_C(  817250921),
                            INT32_C(  102789484), INT32_C(   64131133), INT32_C(   35663940), INT32_C(-1064882042),
                            INT32_C( -392119279), INT32_C(   79249424), INT32_C( 1219088991), INT32_C(  346956559),
                            INT32_C( -778487174), INT32_C( 2030262893), INT32_C(    8533025), INT32_C( 2088865417)) },
    { easysimd_mm512_set_epi32(INT32_C(-1406718947), INT32_C(  276558393), INT32_C(  154803470), INT32_C( 1010355861),
                            INT32_C( -906943422), INT32_C(-1458735792), INT32_C( -135902673), INT32_C( 2125322250),
                            INT32_C(  668612521), INT32_C( 2134097324), INT32_C( 1431164540), INT32_C(-1097880462),
                            INT32_C( 1895279922), INT32_C( -455917584), INT32_C(-1635623774), INT32_C( 1646110584)),
      UINT16_C(61721),
      easysimd_mm512_set_epi32(INT32_C(-1147100012), INT32_C( -529153170), INT32_C(-1710107397), INT32_C( 1085126684),
                            INT32_C( -365628842), INT32_C( 1126939173), INT32_C(-1962930746), INT32_C(-2032518388),
                            INT32_C( -893793955), INT32_C(-1793978656), INT32_C(  353794556), INT32_C(  484459160),
                            INT32_C( 1795576890), INT32_C(-1800969495), INT32_C(  570832120), INT32_C( -805110645)),
      easysimd_mm512_set_epi32(INT32_C(-1152323073), INT32_C(-1880366011), INT32_C( 1623795528), INT32_C(  779718762),
                            INT32_C( -950308445), INT32_C(  601329882), INT32_C( 1983067756), INT32_C( 1014514692),
                            INT32_C(  192697146), INT32_C( 1393627685), INT32_C( -618845734), INT32_C(-1526656596),
                            INT32_C( -668243521), INT32_C(  858775967), INT32_C( -874197170), INT32_C(-1013451033)),
      easysimd_mm512_set_epi32(INT32_C(    5263723), INT32_C(  260711425), INT32_C( 1623728896), INT32_C(  776994914),
                            INT32_C( -906943422), INT32_C(-1458735792), INT32_C( -135902673), INT32_C(  941638656),
                            INT32_C(  668612521), INT32_C( 2134097324), INT32_C( 1431164540), INT32_C(-1593765596),
                            INT32_C(-1876350587), INT32_C( -455917584), INT32_C(-1635623774), INT32_C(   60097124)) },
    { easysimd_mm512_set_epi32(INT32_C(   73765979), INT32_C( 1196192749), INT32_C( -212227718), INT32_C(-1980699203),
                            INT32_C(  -37222007), INT32_C(-1986328859), INT32_C( 1483201456), INT32_C(  129080387),
                            INT32_C( -259597220), INT32_C(-1814466623), INT32_C( 1536667113), INT32_C( 1702406736),
                            INT32_C( 1032855403), INT32_C( -907220805), INT32_C( -744099936), INT32_C( -484286001)),
      UINT16_C(60398),
      easysimd_mm512_set_epi32(INT32_C( 2131878120), INT32_C( -709717494), INT32_C(  677603870), INT32_C( 1110837767),
                            INT32_C(  137332416), INT32_C( 1049147481), INT32_C( -429123521), INT32_C(  562109282),
                            INT32_C( -475857832), INT32_C(-1750530864), INT32_C(-1098694184), INT32_C(-1278646805),
                            INT32_C(  274075622), INT32_C(  310096866), INT32_C( 1944249360), INT32_C(-1457965117)),
      easysimd_mm512_set_epi32(INT32_C(-1770120574), INT32_C(-1267999916), INT32_C(  920660290), INT32_C( 1218524275),
                            INT32_C( -813719782), INT32_C(   17574100), INT32_C( 1228269274), INT32_C( -540460196),
                            INT32_C( -544630186), INT32_C( -973323962), INT32_C( -900762472), INT32_C( 1800691074),
                            INT32_C( -934840396), INT32_C(-2024059127), INT32_C( 2050139755), INT32_C(-1648520849)),
      easysimd_mm512_set_epi32(INT32_C(-2140268030), INT32_C(  541673812), INT32_C(  377487680), INT32_C(-1980699203),
                            INT32_C( -951052006), INT32_C(-1986328859), INT32_C(  152168128), INT32_C( -565698532),
                            INT32_C(  470352390), INT32_C( 1079263494), INT32_C( 1078735872), INT32_C( 1702406736),
                            INT32_C( -939429872), INT32_C(-2063578103), INT32_C(  135266923), INT32_C( -484286001)) },
    { easysimd_mm512_set_epi32(INT32_C(  359551557), INT32_C(  851518101), INT32_C( 1700885885), INT32_C( 1144006274),
                            INT32_C(  718077661), INT32_C( 1054313754), INT32_C(   65647391), INT32_C(-1867262731),
                            INT32_C(  208941224), INT32_C(  989467762), INT32_C(-1763663368), INT32_C(  732190820),
                            INT32_C( -780985117), INT32_C(-1786203682), INT32_C( -893464048), INT32_C(-1930046056)),
      UINT16_C( 5280),
      easysimd_mm512_set_epi32(INT32_C( 2082802710), INT32_C(  398405458), INT32_C( -610997258), INT32_C(  830342728),
                            INT32_C( -327286830), INT32_C( 1285368273), INT32_C(-1636339073), INT32_C( 1467021210),
                            INT32_C( -637556884), INT32_C( 1464578281), INT32_C(  -78771124), INT32_C(-1194071193),
                            INT32_C(-1454776494), INT32_C(  224158188), INT32_C( 1578376173), INT32_C( 2022699384)),
      easysimd_mm512_set_epi32(INT32_C(-1580866758), INT32_C( 1705729088), INT32_C(-1204463345), INT32_C(  806420788),
                            INT32_C(-1410408996), INT32_C(  863225653), INT32_C(-2071560363), INT32_C( 1819484417),
                            INT32_C( -246595685), INT32_C(  243263522), INT32_C( 2052176477), INT32_C(  253176681),
                            INT32_C( 1676258794), INT32_C(-1129907739), INT32_C(  395133900), INT32_C(  -86934818)),
      easysimd_mm512_set_epi32(INT32_C(  359551557), INT32_C(  851518101), INT32_C( 1700885885), INT32_C(      65844),
                            INT32_C(  718077661), INT32_C(  862111268), INT32_C(   65647391), INT32_C(-1867262731),
                            INT32_C(  536877203), INT32_C(  989467762), INT32_C(    1159697), INT32_C(  732190820),
                            INT32_C( -780985117), INT32_C(-1786203682), INT32_C( -893464048), INT32_C(-1930046056)) },
    { easysimd_mm512_set_epi32(INT32_C( -763717484), INT32_C(-1454287993), INT32_C( -815713015), INT32_C( -381645662),
                            INT32_C( 1143121149), INT32_C(-2120634980), INT32_C( -259357121), INT32_C( -593579957),
                            INT32_C(-1529041977), INT32_C(-2065541499), INT32_C( 1009471119), INT32_C(  674532491),
                            INT32_C( -605291509), INT32_C( -802607554), INT32_C( -850350011), INT32_C(  732847081)),
      UINT16_C(41568),
      easysimd_mm512_set_epi32(INT32_C( 1295870302), INT32_C(  336570348), INT32_C(-1662536141), INT32_C(-1054381248),
                            INT32_C( 1593114303), INT32_C(-1017054773), INT32_C(-1409414000), INT32_C(  227338784),
                            INT32_C( 1117509139), INT32_C( 1937140770), INT32_C( 1843080524), INT32_C(  775622876),
                            INT32_C(  903821795), INT32_C(-1108923393), INT32_C( -348808591), INT32_C(  691553406)),
      easysimd_mm512_set_epi32(INT32_C( -957741997), INT32_C( -389978329), INT32_C(-1992364300), INT32_C(-1194120095),
                            INT32_C( 1460280679), INT32_C( -461012902), INT32_C(  191451119), INT32_C(  395863574),
                            INT32_C( 2007897293), INT32_C(  647995187), INT32_C( 1812181798), INT32_C(-1288356108),
                            INT32_C(-1946740515), INT32_C(-1688294491), INT32_C( -146679692), INT32_C( -960173252)),
      easysimd_mm512_set_epi32(INT32_C(-2101214207), INT32_C(-1454287993), INT32_C(   18368708), INT32_C( -381645662),
                            INT32_C( 1143121149), INT32_C(-2120634980), INT32_C(      82799), INT32_C( -593579957),
                            INT32_C(-1529041977), INT32_C(   76124945), INT32_C(      37410), INT32_C(  674532491),
                            INT32_C( -605291509), INT32_C( -802607554), INT32_C( -850350011), INT32_C(  732847081)) },
    { easysimd_mm512_set_epi32(INT32_C(-1543080560), INT32_C(  326946931), INT32_C(  691349892), INT32_C( 1226829378),
                            INT32_C( 1127061143), INT32_C( 1548237043), INT32_C(-1885371906), INT32_C(  673215002),
                            INT32_C(   -2545554), INT32_C(-1367277302), INT32_C( -227991301), INT32_C(  746457208),
                            INT32_C(-1737407854), INT32_C( 1988034150), INT32_C( -605858038), INT32_C( -752579769)),
      UINT16_C(24718),
      easysimd_mm512_set_epi32(INT32_C( 1517976828), INT32_C(  453076709), INT32_C( 1155311084), INT32_C(-1730593997),
                            INT32_C( 2009897302), INT32_C( -813354987), INT32_C( 1160389453), INT32_C(-1543844644),
                            INT32_C( -908777016), INT32_C(  107061968), INT32_C(-1889800585), INT32_C(-1309816398),
                            INT32_C( 1760607631), INT32_C(-1373730647), INT32_C( 1475928392), INT32_C(-1415204909)),
      easysimd_mm512_set_epi32(INT32_C(  901302066), INT32_C(  236605933), INT32_C( 1144123725), INT32_C(  765559000),
                            INT32_C( -272466037), INT32_C(  489940181), INT32_C( 1285546635), INT32_C(  894611583),
                            INT32_C(-1280504231), INT32_C( -511809158), INT32_C(  517714821), INT32_C( -458114298),
                            INT32_C(-1583011646), INT32_C( 2050708057), INT32_C(-1873361568), INT32_C( 1295393304)),
      easysimd_mm512_set_epi32(INT32_C(-1543080560), INT32_C(   68817160), INT32_C(    2183169), INT32_C( 1226829378),
                            INT32_C( 1127061143), INT32_C( 1548237043), INT32_C(-1885371906), INT32_C(  673215002),
                            INT32_C(  841483793), INT32_C(-1367277302), INT32_C( -227991301), INT32_C(  746457208),
                            INT32_C(-2130378688), INT32_C( 1344361040), INT32_C(-2147089376), INT32_C( -752579769)) },
    { easysimd_mm512_set_epi32(INT32_C( -203532895), INT32_C(-1671983312), INT32_C( -485765980), INT32_C(-1920770849),
                            INT32_C(  -87193791), INT32_C( 1659979037), INT32_C(-1337410362), INT32_C( 1209029675),
                            INT32_C(  587197109), INT32_C( -530755740), INT32_C(  281664792), INT32_C(  -47077792),
                            INT32_C( -945013045), INT32_C( -166692659), INT32_C( 1790118115), INT32_C(  689330771)),
      UINT16_C( 7519),
      easysimd_mm512_set_epi32(INT32_C( -384323470), INT32_C(  473195364), INT32_C(  206146438), INT32_C(-1217279332),
                            INT32_C(-1088463893), INT32_C(  970520784), INT32_C( -929499045), INT32_C(-1086034653),
                            INT32_C(-1051759609), INT32_C(-1753508816), INT32_C( 1464082608), INT32_C(  492133710),
                            INT32_C( 1610388137), INT32_C(-2026322187), INT32_C(-1721391979), INT32_C(  466414066)),
      easysimd_mm512_set_epi32(INT32_C( 1039275088), INT32_C( -195464931), INT32_C(-1467895249), INT32_C( 1829711637),
                            INT32_C( 2006708634), INT32_C(  837542220), INT32_C( -759309790), INT32_C( -498075629),
                            INT32_C(  922280800), INT32_C(  925077084), INT32_C( 1941328295), INT32_C(   27280850),
                            INT32_C( -499921640), INT32_C(  738410205), INT32_C(  972641353), INT32_C( 1011602801)),
      easysimd_mm512_set_epi32(INT32_C( -203532895), INT32_C(-1671983312), INT32_C( -485765980), INT32_C( 1208886529),
                            INT32_C( 1082171408), INT32_C(    2294028), INT32_C(-1337410362), INT32_C( 1074499600),
                            INT32_C(  587197109), INT32_C(  536873548), INT32_C(  281664792), INT32_C(   10485904),
                            INT32_C(-1610396400), INT32_C(  671299592), INT32_C(  546852936), INT32_C(  604181505)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_mask_andnot_epi32(test_vec[i].src, test_vec[i].k, test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_andnot_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT16_C(56303),
      easysimd_mm512_set_epi32(INT32_C(  684353163), INT32_C( -624296854), INT32_C(-1626870831), INT32_C( 1693659819),
                            INT32_C( 1814966119), INT32_C( 1428960968), INT32_C( 1709146671), INT32_C(-1269736679),
                            INT32_C( -399245267), INT32_C(  128121852), INT32_C(  623395494), INT32_C( 1817163956),
                            INT32_C( 1991308671), INT32_C( -978886098), INT32_C( 1436967950), INT32_C(  227176170)),
      easysimd_mm512_set_epi32(INT32_C( -155316348), INT32_C( 1821995326), INT32_C(-1956349521), INT32_C( 2078645861),
                            INT32_C(-2002962850), INT32_C( 1961273418), INT32_C( 1026886280), INT32_C( 1852456749),
                            INT32_C( 1549356853), INT32_C(  905982506), INT32_C( -562722910), INT32_C( 1231420121),
                            INT32_C(  786944005), INT32_C(-1682464667), INT32_C(   12357782), INT32_C(  913777965)),
      easysimd_mm512_set_epi32(INT32_C( -701232892), INT32_C(  605028628), INT32_C(          0), INT32_C(  453282884),
                            INT32_C(-2137976808), INT32_C(          0), INT32_C(  404752512), INT32_C( 1244275748),
                            INT32_C(  340348688), INT32_C(  805306370), INT32_C( -631929600), INT32_C(          0),
                            INT32_C(  138870784), INT32_C(  437289025), INT32_C(    1609872), INT32_C(  846528773)) },
    { UINT16_C(56200),
      easysimd_mm512_set_epi32(INT32_C( -452164103), INT32_C( 1890508390), INT32_C( 1258638805), INT32_C( -750109723),
                            INT32_C( -513503890), INT32_C( -379667747), INT32_C(-1651966538), INT32_C(  418163645),
                            INT32_C(-1484633406), INT32_C(  128570401), INT32_C(-1432905388), INT32_C(-1460529893),
                            INT32_C( -808466332), INT32_C(-1300168003), INT32_C(  153276923), INT32_C( -912847520)),
      easysimd_mm512_set_epi32(INT32_C( 1849401350), INT32_C(-2046167065), INT32_C(-1772087293), INT32_C(  763578781),
                            INT32_C(  -59556630), INT32_C( -574235850), INT32_C(-1931079616), INT32_C(  856557360),
                            INT32_C( 1798494574), INT32_C( -255236934), INT32_C( -498039931), INT32_C( 1916101155),
                            INT32_C( 1291737736), INT32_C(-1818740725), INT32_C( 1042711156), INT32_C(  770521823)),
      easysimd_mm512_set_epi32(INT32_C(  171122694), INT32_C(-2046746239), INT32_C(          0), INT32_C(  746668056),
                            INT32_C(  471019648), INT32_C(          0), INT32_C(    6684736), INT32_C(  587334656),
                            INT32_C( 1211142444), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(    3150472), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(29534),
      easysimd_mm512_set_epi32(INT32_C(-1569526022), INT32_C( -566961257), INT32_C( -454262297), INT32_C(-2011970966),
                            INT32_C( 1729229439), INT32_C(  515441803), INT32_C( 1629075756), INT32_C( -633945234),
                            INT32_C(-1517000454), INT32_C(-2129179491), INT32_C(-1082415130), INT32_C( -643068488),
                            INT32_C(-1177678851), INT32_C(  811665360), INT32_C(-1120986687), INT32_C( 1945770944)),
      easysimd_mm512_set_epi32(INT32_C( 1206445472), INT32_C( 1685117563), INT32_C( -105634979), INT32_C(  300875900),
                            INT32_C( 1292473590), INT32_C( -154568093), INT32_C( -725481309), INT32_C( 1537059805),
                            INT32_C(-1299234249), INT32_C( 1342055246), INT32_C( 1121196977), INT32_C( -936323200),
                            INT32_C(  284920534), INT32_C( -501374627), INT32_C(  523356394), INT32_C( 2082914622)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  541065320), INT32_C(  420487704), INT32_C(  300679188),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1799225213), INT32_C(   25766033),
                            INT32_C(          0), INT32_C( 1323836226), INT32_C(          0), INT32_C(    1069568),
                            INT32_C(    3244034), INT32_C(-1038311411), INT32_C(   34652202), INT32_C(          0)) },
    { UINT16_C(18467),
      easysimd_mm512_set_epi32(INT32_C(-1126901666), INT32_C(-1540993522), INT32_C( -310394649), INT32_C( 1569425965),
                            INT32_C( 1860055197), INT32_C( 1022884520), INT32_C(  886587779), INT32_C(   -7751100),
                            INT32_C(  725782952), INT32_C( 1524528742), INT32_C(-1901622691), INT32_C( -205155472),
                            INT32_C( 1297212229), INT32_C(-1562315637), INT32_C(-1561800150), INT32_C( 1969817622)),
      easysimd_mm512_set_epi32(INT32_C( 1691822441), INT32_C( -747576101), INT32_C(  526461787), INT32_C(-1551035253),
                            INT32_C( -494445545), INT32_C(  601243904), INT32_C( 1621282220), INT32_C(   87983768),
                            INT32_C( 1749180883), INT32_C(  653596692), INT32_C( 1933605299), INT32_C( 2110990238),
                            INT32_C( 1287872496), INT32_C( -947101027), INT32_C(-1469323630), INT32_C( -103698146)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( 1397792977), INT32_C(          0), INT32_C(          0),
                            INT32_C(-2147398654), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( 1900048802), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(  134423696), INT32_C(-2003783416)) },
    { UINT16_C(10499),
      easysimd_mm512_set_epi32(INT32_C(-1800578563), INT32_C( 1189147870), INT32_C( -918534753), INT32_C(-2046784432),
                            INT32_C( 2146267513), INT32_C( 1185116678), INT32_C(  743422455), INT32_C( -958735431),
                            INT32_C(-1272492795), INT32_C(-1993475811), INT32_C( -901911405), INT32_C( -444376352),
                            INT32_C( 1645484254), INT32_C( 1890851846), INT32_C(  632187417), INT32_C( 2142729898)),
      easysimd_mm512_set_epi32(INT32_C( -752859034), INT32_C( -661272677), INT32_C( 1736074301), INT32_C( 1246429845),
                            INT32_C(-1327059157), INT32_C(-1760626525), INT32_C(  693999571), INT32_C(  179503183),
                            INT32_C(-1261277577), INT32_C( 2014601419), INT32_C(   45385261), INT32_C( 1333239387),
                            INT32_C( 1950214560), INT32_C( 2050540474), INT32_C(  -73887902), INT32_C(-1586317941)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(  641343520), INT32_C(          0),
                            INT32_C(-2147316222), INT32_C(          0), INT32_C(          0), INT32_C(  136380486),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( -636451486), INT32_C(-2143256319)) },
    { UINT16_C(17481),
      easysimd_mm512_set_epi32(INT32_C( -441498123), INT32_C(  324738064), INT32_C(  -27713047), INT32_C(  322022433),
                            INT32_C( -623687192), INT32_C(  441486000), INT32_C(-1091397610), INT32_C(  486920838),
                            INT32_C(  727930899), INT32_C(  134578624), INT32_C( -229821250), INT32_C(-1459771681),
                            INT32_C(  786852212), INT32_C(-1562273484), INT32_C(  592450244), INT32_C( -391708168)),
      easysimd_mm512_set_epi32(INT32_C(  792156312), INT32_C(  407601311), INT32_C(-1255558455), INT32_C( 1648353396),
                            INT32_C(-1874603621), INT32_C(-1962724996), INT32_C(-1379808132), INT32_C(-1917277067),
                            INT32_C( -327375348), INT32_C( -266290190), INT32_C( -446684576), INT32_C( -218289365),
                            INT32_C( 1659849163), INT32_C(  313080914), INT32_C(  914897986), INT32_C( -690088867)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  134250639), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(-2130497204), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( -266323406), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1074331787), INT32_C(          0), INT32_C(          0), INT32_C(  374871045)) },
    { UINT16_C(25655),
      easysimd_mm512_set_epi32(INT32_C(    7734189), INT32_C(-1107618186), INT32_C( 1291997837), INT32_C( -657618671),
                            INT32_C( -523204184), INT32_C(  197247571), INT32_C(-1924672781), INT32_C( 1367953812),
                            INT32_C( 1671605226), INT32_C( -667696065), INT32_C(  734579404), INT32_C(  -25998720),
                            INT32_C( -791898275), INT32_C(-1848361166), INT32_C(  302446873), INT32_C(-1290034089)),
      easysimd_mm512_set_epi32(INT32_C(-2140777278), INT32_C( 1356458144), INT32_C(  990615850), INT32_C(  122581591),
                            INT32_C( 1842174798), INT32_C( 1633161914), INT32_C( 1487544794), INT32_C( 1680890315),
                            INT32_C(-1051319145), INT32_C( 1671869354), INT32_C( -657093416), INT32_C(   76483879),
                            INT32_C(  897241075), INT32_C(-1385812547), INT32_C(  518745683), INT32_C( 1278998383)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( 1073801344), INT32_C(  839485730), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1612189864), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( -803929072), INT32_C(    9176359),
                            INT32_C(          0), INT32_C(  740425869), INT32_C(  216598082), INT32_C( 1277186856)) },
    { UINT16_C( 9319),
      easysimd_mm512_set_epi32(INT32_C(  359510622), INT32_C( 1667719225), INT32_C(  630674948), INT32_C(  610105763),
                            INT32_C(   20744378), INT32_C(-1334671422), INT32_C( 1934181344), INT32_C( -207473635),
                            INT32_C(  -12247390), INT32_C(  935971775), INT32_C( -814870615), INT32_C(  272416728),
                            INT32_C(-2094904434), INT32_C(  118285194), INT32_C( 1770668331), INT32_C(-1463910375)),
      easysimd_mm512_set_epi32(INT32_C(  399098366), INT32_C(-1713281213), INT32_C( 2124618772), INT32_C(-1052563089),
                            INT32_C( 1851869047), INT32_C( 2020277970), INT32_C(-1035589842), INT32_C(-1789987668),
                            INT32_C(  733487930), INT32_C( -497440680), INT32_C(-1951336884), INT32_C(-1752937795),
                            INT32_C(-1263292061), INT32_C(     242422), INT32_C( 1531342059), INT32_C( -447099781)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C( 1512048656), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1208514576), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(-1072684992), INT32_C(    9498692), INT32_C(          0),
                            INT32_C(          0), INT32_C(     200820), INT32_C(  306457792), INT32_C( 1161907298)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_maskz_andnot_epi32(test_vec[i].k, test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_andnot_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(  207721957124820559), INT64_C( 7800065217939756514),
                            INT64_C(-3924116943760495845), INT64_C(-4670511705337769443),
                            INT64_C( 8681164262815197674), INT64_C(-1748050366477277388),
                            INT64_C( 6521714148432251392), INT64_C( -441034964778727222)),
      easysimd_mm512_set_epi64(INT64_C( 1906622887772594248), INT64_C(-6188571553716009650),
                            INT64_C(  264280323541139152), INT64_C( 1584607577489802492),
                            INT64_C( 1109502791419861047), INT64_C( 7178327557348084990),
                            INT64_C( 1395553581907359924), INT64_C(-6533533183118051768)),
      easysimd_mm512_set_epi64(INT64_C( 1735017709269196800), INT64_C(-9079113909020786676),
                            INT64_C(  153194412888495296), INT64_C(   58731659148920032),
                            INT64_C(  505544490090532885), INT64_C(     565705191721162),
                            INT64_C(   98516792007995572), INT64_C(  293888628881438208)) },
    { easysimd_mm512_set_epi64(INT64_C(-6724768156717290819), INT64_C(-5250906792133082841),
                            INT64_C( 7101763469273509464), INT64_C(-6606445878350250265),
                            INT64_C( -992513133092468415), INT64_C( 6991549638647222936),
                            INT64_C(-6702609966967541799), INT64_C( 6463119549714578187)),
      easysimd_mm512_set_epi64(INT64_C( -887590357697526439), INT64_C( 7877951253899372304),
                            INT64_C(-2913340636735054567), INT64_C(-9221955371178601401),
                            INT64_C(-8332586207462320569), INT64_C(-6065729331557277752),
                            INT64_C(-5495188752553836498), INT64_C(-4545091831756868823)),
      easysimd_mm512_set_epi64(INT64_C( 5837268749247317312), INT64_C( 5211811456626140688),
                            INT64_C(-7705320311288082175), INT64_C(    1134704830971904),
                            INT64_C(  883853889565267462), INT64_C(-8444247080808460992),
                            INT64_C( 1226140460016631846), INT64_C(-9201976626802327520)) },
    { easysimd_mm512_set_epi64(INT64_C(-8081018174907924542), INT64_C( 6936249846129023242),
                            INT64_C(-1059210610078769383), INT64_C( 1593162574725548027),
                            INT64_C( 2194029932784271057), INT64_C( 2297742112014824027),
                            INT64_C( 6872936620014531062), INT64_C(-4458741002964204726)),
      easysimd_mm512_set_epi64(INT64_C( 7389599045220123111), INT64_C(-4734617337151831127),
                            INT64_C(-3688698012661984630), INT64_C( 2942411497108224949),
                            INT64_C( 3088165388972230068), INT64_C(-8598989874996476457),
                            INT64_C(-2925060949778391940), INT64_C( 1600057734617632940)),
      easysimd_mm512_set_epi64(INT64_C( 6918937665425915941), INT64_C(-7059040014775614815),
                            INT64_C(  901582149085413506), INT64_C( 2936772469119858692),
                            INT64_C( 2344435893274706212), INT64_C(-9221049868269222524),
                            INT64_C(-9221110889849910264), INT64_C( 1450299817904312484)) },
    { easysimd_mm512_set_epi64(INT64_C(-2851531746227363368), INT64_C( 2067892326136395565),
                            INT64_C( 5955544350840259834), INT64_C(-9215158447496033102),
                            INT64_C(-6496129397571023850), INT64_C( 6580537045822776099),
                            INT64_C(-1881492268188536860), INT64_C( 6477581622128112348)),
      easysimd_mm512_set_epi64(INT64_C( 4736931688263401886), INT64_C( -422510099501192510),
                            INT64_C( 3904035851984069712), INT64_C(-1269778779692298262),
                            INT64_C( 7103388094266435672), INT64_C( 8538164081108009860),
                            INT64_C( 7657481289221491954), INT64_C(-6346831563088898420)),
      easysimd_mm512_set_epi64(INT64_C(  112770318310899718), INT64_C(-2161639582911543102),
                            INT64_C( 2596786860877701120), INT64_C( 7953586499903062856),
                            INT64_C( 4756998866794012744), INT64_C( 2606779805598826628),
                            INT64_C(  721778983603339282), INT64_C(-6482086895067069440)) },
    { easysimd_mm512_set_epi64(INT64_C( -821005629772787069), INT64_C(-4647973389902912809),
                            INT64_C( 6459900742609080709), INT64_C(   -1266809698382208),
                            INT64_C(  701020828809534395), INT64_C(-8547290149729742964),
                            INT64_C( -440779604644636577), INT64_C(-3509307452635316669)),
      easysimd_mm512_set_epi64(INT64_C( 8999318376500703433), INT64_C( 1719097867730734351),
                            INT64_C(  360091487853740826), INT64_C(-6254537314592943558),
                            INT64_C( -632347399973673450), INT64_C( 2614451855333869078),
                            INT64_C( 6887846494654494209), INT64_C( 6275950466702179569)),
      easysimd_mm512_set_epi64(INT64_C(  604608525006544968), INT64_C(   36136703980768520),
                            INT64_C(  313352018360009242), INT64_C(        137573240890),
                            INT64_C( -720169941136284668), INT64_C( 2596327487390613522),
                            INT64_C(  438118704866436608), INT64_C( 1157566394459521200)) },
    { easysimd_mm512_set_epi64(INT64_C(-5483950330033170066), INT64_C(-4153699507396814554),
                            INT64_C( 1686943364333831141), INT64_C(-6155572369391990976),
                            INT64_C(-2338197867102969548), INT64_C( 4970317907692585902),
                            INT64_C( -659027381808082615), INT64_C(-8301976371410819309)),
      easysimd_mm512_set_epi64(INT64_C(-5922203424268985599), INT64_C( 1802271341012641429),
                            INT64_C(-7199161640250473305), INT64_C( 4184910176757162424),
                            INT64_C(-5885970898589897236), INT64_C( 5320604596895707800),
                            INT64_C(-7049806138053003152), INT64_C( 7856069210784274088)),
      easysimd_mm512_set_epi64(INT64_C(  869198318683570689), INT64_C( 1801690747234690705),
                            INT64_C(-8640647776843037694), INT64_C( 1153141544681808056),
                            INT64_C( 2328590264702274760), INT64_C(  649785191505621008),
                            INT64_C(  585473076492838960), INT64_C( 6991285376398659752)) },
    { easysimd_mm512_set_epi64(INT64_C(  772369500911491951), INT64_C(-3487181344595680581),
                            INT64_C(-6776954808191866646), INT64_C( 1437133779275187040),
                            INT64_C(-3742444221385296201), INT64_C( 3619551202282748987),
                            INT64_C(-5676058734881350704), INT64_C( 3034639668798379519)),
      easysimd_mm512_set_epi64(INT64_C( 7799576852730631653), INT64_C(-4611614721990756478),
                            INT64_C( 4179897201710999091), INT64_C(-6554042946408561565),
                            INT64_C( 7858455943023474684), INT64_C(-4868663260305658784),
                            INT64_C(-6563387696243649675), INT64_C( -252761203575600938)),
      easysimd_mm512_set_epi64(INT64_C( 7207351508714783872), INT64_C(      71283551638784),
                            INT64_C( 1873656161226589713), INT64_C(-6626408997484215293),
                            INT64_C( 2382069952524845384), INT64_C(-8339540561327800256),
                            INT64_C(  342318686555209765), INT64_C(-3142946274104309760)) },
    { easysimd_mm512_set_epi64(INT64_C(-6272776462503295319), INT64_C(-8894851852280934479),
                            INT64_C( 6828037840473322695), INT64_C( -784763491569829334),
                            INT64_C(-6956613286547242208), INT64_C(-7641604144835014945),
                            INT64_C( 4137535773895137731), INT64_C( 3122415965305276610)),
      easysimd_mm512_set_epi64(INT64_C( 5967240469174938071), INT64_C( 2271146860082105533),
                            INT64_C( 2488999494207974941), INT64_C(-7245269557183082373),
                            INT64_C(-6094983942162054282), INT64_C( 5272800144124782830),
                            INT64_C(-1112016268759137335), INT64_C( 3873297534982922048)),
      easysimd_mm512_set_epi64(INT64_C( 5912488079989451094), INT64_C( 1945726568221376524),
                            INT64_C( 2308274862648494616), INT64_C(  747597780979417169),
                            INT64_C( 2308728562190385238), INT64_C( 5191575370143047712),
                            INT64_C(-4571152522202316280), INT64_C( 1477377559112455936)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_andnot_epi64(test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_andnot_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(-8706055201876274534), INT64_C(-2974526497282267924),
                            INT64_C(-5064099105424399850), INT64_C( 4173762680971677425),
                            INT64_C( 5058953897646810163), INT64_C( 3129329827313761969),
                            INT64_C(-7680990319456213473), INT64_C( 3095613893972693568)),
      UINT8_C(148),
      easysimd_mm512_set_epi64(INT64_C( -438459145642420823), INT64_C( 2788318060387771818),
                            INT64_C(-6405634033298828022), INT64_C( 5697280571633296693),
                            INT64_C(-4038706177987584167), INT64_C( 7050984609072161968),
                            INT64_C( 2749018709708772273), INT64_C(-2265592192997989021)),
      easysimd_mm512_set_epi64(INT64_C(-1049270424665539045), INT64_C(-5614406584732574076),
                            INT64_C(  631202638299991092), INT64_C(-8590255914187036925),
                            INT64_C(-6307315262773811693), INT64_C(-1209843912248425712),
                            INT64_C(-3872834841544228683), INT64_C(-3085083838104197908)),
      easysimd_mm512_set_epi64(INT64_C(    4565176708278802), INT64_C(-2974526497282267924),
                            INT64_C(-5064099105424399850), INT64_C(-9166793916060532222),
                            INT64_C( 5058953897646810163), INT64_C(-8203935250787499264),
                            INT64_C(-7680990319456213473), INT64_C( 3095613893972693568)) },
    { easysimd_mm512_set_epi64(INT64_C(-4842938149095873389), INT64_C( -846085209911123390),
                            INT64_C(  902030110892207375), INT64_C(-8179884512098486778),
                            INT64_C( 7136180633023633249), INT64_C(-7202514001649392691),
                            INT64_C(-4512985345247872566), INT64_C( 6280820093975482096)),
      UINT8_C( 88),
      easysimd_mm512_set_epi64(INT64_C(-5899542268894168412), INT64_C( 5687678929880926481),
                            INT64_C(  754471637334648472), INT64_C( 1530269878614188173),
                            INT64_C(-3481843836368626596), INT64_C( 7214537798473258692),
                            INT64_C( 3186147264512503626), INT64_C(-2220217993706522327)),
      easysimd_mm512_set_epi64(INT64_C( 4741426381855247639), INT64_C(-6093431436741802321),
                            INT64_C(-7277776184535270866), INT64_C(-5890238516652006119),
                            INT64_C(-8031043717190201593), INT64_C( 7604814614465185239),
                            INT64_C( 6712821644684838579), INT64_C(-1700024539209227072)),
      easysimd_mm512_set_epi64(INT64_C(-4842938149095873389), INT64_C(-6845110636993050962),
                            INT64_C(  902030110892207375), INT64_C(-6178618704551475952),
                            INT64_C( 1152921504676217603), INT64_C(-7202514001649392691),
                            INT64_C(-4512985345247872566), INT64_C( 6280820093975482096)) },
    { easysimd_mm512_set_epi64(INT64_C( -647905387169688868), INT64_C(-8461625299591442725),
                            INT64_C(-4959110866452894415), INT64_C(-6046186632754619075),
                            INT64_C(-1792277330244185216), INT64_C( 7899374623587606112),
                            INT64_C(-2530906147097710338), INT64_C(-3452464982464189359)),
      UINT8_C(234),
      easysimd_mm512_set_epi64(INT64_C( 1092825191169264761), INT64_C(  518154175979275913),
                            INT64_C(-2540128939765803497), INT64_C( 7206989642204137224),
                            INT64_C( 5053971549089664110), INT64_C(  275130895293265200),
                            INT64_C( 5870095287105445532), INT64_C( 3766077764635497461)),
      easysimd_mm512_set_epi64(INT64_C( 4726923138274336458), INT64_C( 3036293318033390010),
                            INT64_C( 3265833753663381966), INT64_C(-5548402770380826836),
                            INT64_C(-1910939043053590920), INT64_C(-2803972634053834044),
                            INT64_C( 8571307896088376800), INT64_C(-2906367800591944553)),
      easysimd_mm512_set_epi64(INT64_C( 4652501007819903618), INT64_C( 2883153157893175602),
                            INT64_C( 2395932937578488264), INT64_C(-6046186632754619075),
                            INT64_C(-6820513618777071088), INT64_C( 7899374623587606112),
                            INT64_C( 2774537390188929376), INT64_C(-3452464982464189359)) },
    { easysimd_mm512_set_epi64(INT64_C( 1235103765186305905), INT64_C( 8251648155281492223),
                            INT64_C( 6607793927948629202), INT64_C(-4956133557414585628),
                            INT64_C( -962568210701922461), INT64_C( 7520783669412628517),
                            INT64_C( 4493695514722238610), INT64_C( 6191552237626999876)),
      UINT8_C(175),
      easysimd_mm512_set_epi64(INT64_C(-1999731829913464848), INT64_C( 7072204574593617968),
                            INT64_C( -329416891633690006), INT64_C( 4219653511875682573),
                            INT64_C(-5631405021388401918), INT64_C( -157450572284011331),
                            INT64_C(-6448890677231800514), INT64_C(-7780641104162742337)),
      easysimd_mm512_set_epi64(INT64_C(  261057906798578959), INT64_C(-4964336716206621793),
                            INT64_C(-2469501117696455323), INT64_C( 2339328587648411167),
                            INT64_C( 8220620103791574591), INT64_C(  273538927111600315),
                            INT64_C(-3298288074488883789), INT64_C(-8357787233131660724)),
      easysimd_mm512_set_epi64(INT64_C(  252325274594050063), INT64_C( 8251648155281492223),
                            INT64_C(  329344140649481477), INT64_C(-4956133557414585628),
                            INT64_C( 4757067868831771709), INT64_C(  147282005282398210),
                            INT64_C( 5780933484690985089), INT64_C(  577059746971148352)) },
    { easysimd_mm512_set_epi64(INT64_C(-4285851555602414983), INT64_C(-8492982904341423564),
                            INT64_C(-2837093742585682248), INT64_C(  267283033869441308),
                            INT64_C( 4311088349833897908), INT64_C( -647706517356585524),
                            INT64_C(-3770716194274572842), INT64_C(-8566807519504738391)),
      UINT8_C( 75),
      easysimd_mm512_set_epi64(INT64_C(-6282230583383062251), INT64_C(-7841791912404359359),
                            INT64_C(-7579575622870303941), INT64_C(-2922061146712111361),
                            INT64_C( 4606944383693507801), INT64_C(-6882069134795290712),
                            INT64_C(-4540648442557822523), INT64_C( 8626282944079879495)),
      easysimd_mm512_set_epi64(INT64_C(-1823698107073259294), INT64_C( 8029233569224881686),
                            INT64_C(   46900467487790247), INT64_C( 8663098726891022114),
                            INT64_C( 2596646339415618602), INT64_C( 7059567741718714192),
                            INT64_C( 7446336952031093968), INT64_C(   16931348739669095)),
      easysimd_mm512_set_epi64(INT64_C(-4285851555602414983), INT64_C( 7800656914580246550),
                            INT64_C(-2837093742585682248), INT64_C(  267283033869441308),
                            INT64_C(         88250757154), INT64_C( -647706517356585524),
                            INT64_C( 2810971851134903312), INT64_C(    2252181026775072)) },
    { easysimd_mm512_set_epi64(INT64_C( 2037127205197222183), INT64_C( 3451898891201360501),
                            INT64_C( 1455211247092394628), INT64_C( 2206658725580708086),
                            INT64_C( 5349364315141837270), INT64_C( 7849256443344717184),
                            INT64_C( 4856719246957022704), INT64_C(-4923001172558722698)),
      UINT8_C(149),
      easysimd_mm512_set_epi64(INT64_C( 6411014556179012579), INT64_C(-8290562023531042118),
                            INT64_C( 3513406971994598159), INT64_C(  170515694744852127),
                            INT64_C( 7762613428125762288), INT64_C( 4486051683696872920),
                            INT64_C(-3347799382542858009), INT64_C( 7877354972766519961)),
      easysimd_mm512_set_epi64(INT64_C( 2384233607786009160), INT64_C( 7136321197786935066),
                            INT64_C(-2775012291419678803), INT64_C( 1447324989515017380),
                            INT64_C(-5436087904826886612), INT64_C( 7888585058472078205),
                            INT64_C(-7864278168616859201), INT64_C( 8559884086409161720)),
      easysimd_mm512_set_epi64(INT64_C( 2379589521848270856), INT64_C( 3451898891201360501),
                            INT64_C( 1455211247092394628), INT64_C( 1441191875528796192),
                            INT64_C( 5349364315141837270), INT64_C( 4699579053875929637),
                            INT64_C( 4856719246957022704), INT64_C( 1335881482333858144)) },
    { easysimd_mm512_set_epi64(INT64_C( -626073311570320561), INT64_C( 4678237318537021585),
                            INT64_C( 7326175960335696621), INT64_C( 2614088339478761539),
                            INT64_C(-3404519381245739218), INT64_C( 8481274767690754747),
                            INT64_C(-4945537623263429760), INT64_C( 5945167030889147721)),
      UINT8_C(209),
      easysimd_mm512_set_epi64(INT64_C( 1396956538408270925), INT64_C(  433531675836732237),
                            INT64_C(-2740776246441943234), INT64_C(  627773489989817177),
                            INT64_C( 2334235533617502306), INT64_C( 5200994462656867787),
                            INT64_C( 6058971438237170661), INT64_C(-1718043134590880356)),
      easysimd_mm512_set_epi64(INT64_C( 6582702301060698834), INT64_C(-6620728110496909408),
                            INT64_C(-2674893574601157335), INT64_C(-3191892667818640289),
                            INT64_C(-1755995440120031315), INT64_C(-9164966479234216120),
                            INT64_C( -811539623059483440), INT64_C(-8790398035654865383)),
      easysimd_mm512_set_epi64(INT64_C( 5194902496598033042), INT64_C(-6910140186789469024),
                            INT64_C( 7326175960335696621), INT64_C(-3242571914706752506),
                            INT64_C(-3404519381245739218), INT64_C( 8481274767690754747),
                            INT64_C(-4945537623263429760), INT64_C(  432964590381304321)) },
    { easysimd_mm512_set_epi64(INT64_C(-6743158443935274483), INT64_C( -109319504177728220),
                            INT64_C(-4028288193005214442), INT64_C(  132288430860812468),
                            INT64_C(  917336920958928215), INT64_C(-8592087087533075804),
                            INT64_C( -911564553413882344), INT64_C(-5778334739542351628)),
      UINT8_C(132),
      easysimd_mm512_set_epi64(INT64_C(-8373098054511418162), INT64_C( 7896680406183363835),
                            INT64_C( 4931162839211744539), INT64_C(-7345169465412510410),
                            INT64_C(-7349547769362151281), INT64_C(    1089692206936889),
                            INT64_C( 6524506004040415129), INT64_C( 6226593529101379713)),
      easysimd_mm512_set_epi64(INT64_C( 3458147115787789114), INT64_C( 7210094384770191006),
                            INT64_C( 7088560670460655534), INT64_C( -803268445524244375),
                            INT64_C( 4723424603414443741), INT64_C( 1370109689785890561),
                            INT64_C(-4376650697011830162), INT64_C( -620804834547376669)),
      easysimd_mm512_set_epi64(INT64_C( 2607874799996928816), INT64_C( -109319504177728220),
                            INT64_C(-4028288193005214442), INT64_C(  132288430860812468),
                            INT64_C(  917336920958928215), INT64_C( 1369094837600650240),
                            INT64_C( -911564553413882344), INT64_C(-5778334739542351628)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_mask_andnot_epi64(test_vec[i].src, test_vec[i].k, test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_andnot_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C(  7),
      easysimd_mm512_set_epi64(INT64_C(-2016264017930850215), INT64_C( 6207900603916400351),
                            INT64_C( 7392720324711365837), INT64_C( 8770333430120422633),
                            INT64_C(  490532205378570002), INT64_C(-6106476949393880649),
                            INT64_C(-1854090463849988422), INT64_C( 2161894352221900559)),
      easysimd_mm512_set_epi64(INT64_C( 2471053143203888378), INT64_C( 4307108638624930374),
                            INT64_C( 8813537095665060151), INT64_C( -722272124812023485),
                            INT64_C( -967288076808354317), INT64_C(-6013850093851417513),
                            INT64_C( 3331958923341291108), INT64_C( -281534168919433716)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(  327214808695383104),
                            INT64_C(  592518518720905284), INT64_C(-2299289876066988032)) },
    { UINT8_C( 76),
      easysimd_mm512_set_epi64(INT64_C(-7188491746248886702), INT64_C( 3795103503776882624),
                            INT64_C( 8025930014425820340), INT64_C(-7929605366413196523),
                            INT64_C( 5924420044782879602), INT64_C(-3302350069387149227),
                            INT64_C(-1821341009738891830), INT64_C(-6812922588519498817)),
      easysimd_mm512_set_epi64(INT64_C(-1266328346505933550), INT64_C( 1669938728598205410),
                            INT64_C(-7350359895777029108), INT64_C( 9139543262716722238),
                            INT64_C(-9200593584210926828), INT64_C(-3449434666635797941),
                            INT64_C( 4314658246940308870), INT64_C( -478133805478226079)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(  217316721059520546),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-9205313376526131196), INT64_C(       5910164412938),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(148),
      easysimd_mm512_set_epi64(INT64_C( 2173045647004856331), INT64_C(-9109531323294262314),
                            INT64_C(-2493109132018654878), INT64_C( 6270825741977490200),
                            INT64_C( 8719769943602297687), INT64_C(-4201021528893071940),
                            INT64_C( 9011627797455533120), INT64_C( 6620301637478416060)),
      easysimd_mm512_set_epi64(INT64_C(-4851330938418837166), INT64_C( 8567660546009495156),
                            INT64_C(-2946935282469126440), INT64_C(-3944680176869437518),
                            INT64_C(-3189291857021003507), INT64_C(  852944387991302704),
                            INT64_C( 5948575888921546761), INT64_C( 4930911444432807162)),
      easysimd_mm512_set_epi64(INT64_C(-6880224560885528240), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-8628461452576158558),
                            INT64_C(                   0), INT64_C(  739720922782507520),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C( 97),
      easysimd_mm512_set_epi64(INT64_C( -924406031683798297), INT64_C( 1082742291630099615),
                            INT64_C( 3950666752159487194), INT64_C( 8443851551588188807),
                            INT64_C( 5838662214875022266), INT64_C(-6073322957639126750),
                            INT64_C( 1174103819847041898), INT64_C(  693926700598930845)),
      easysimd_mm512_set_epi64(INT64_C(  917406711858321823), INT64_C(-2954398701286057389),
                            INT64_C(  580508427727522845), INT64_C(-4656281121400174897),
                            INT64_C(-3028496641912979897), INT64_C( 6357018899588818011),
                            INT64_C( 5102737467710367164), INT64_C( 1099306012957445482)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-3388587049163943360),
                            INT64_C(  579945122294155269), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(  450505469795237986)) },
    { UINT8_C(210),
      easysimd_mm512_set_epi64(INT64_C( 4586733821042914141), INT64_C(-6562128612845469564),
                            INT64_C(-5088971089241108691), INT64_C( 4584509013736167571),
                            INT64_C( 7541158438725419821), INT64_C(-6577447853347647248),
                            INT64_C( 8000393737083977627), INT64_C(-3838210298295657456)),
      easysimd_mm512_set_epi64(INT64_C(-3810154219907114893), INT64_C( 1390546034528663938),
                            INT64_C(-7278948997228835946), INT64_C(-6400015342302035742),
                            INT64_C(-5025729231272531675), INT64_C( -727304839347940122),
                            INT64_C( 5841837551579279726), INT64_C(-6256756974903097514)),
      easysimd_mm512_set_epi64(INT64_C(-4604784503990056926), INT64_C( 1369116277674653954),
                            INT64_C(                   0), INT64_C(-9214222637876019104),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 1157530966609723492), INT64_C(                   0)) },
    { UINT8_C(171),
      easysimd_mm512_set_epi64(INT64_C( 2614947921582018787), INT64_C(-4851561713766233132),
                            INT64_C(-9121795968209612126), INT64_C(  784084589312935430),
                            INT64_C( 3206750945776122646), INT64_C( 2956179786298753960),
                            INT64_C( 5449808455866424595), INT64_C(  314020808054955060)),
      easysimd_mm512_set_epi64(INT64_C(  420924716680581769), INT64_C(  634178498505834615),
                            INT64_C(-2861544115657502554), INT64_C(-7045300656768620560),
                            INT64_C( 3724569018417139461), INT64_C( 7684038547017787602),
                            INT64_C( 4661447160348399809), INT64_C( 8780209518656646828)),
      easysimd_mm512_set_epi64(INT64_C(  114072716522619400), INT64_C(                   0),
                            INT64_C( 6341349786890797060), INT64_C(                   0),
                            INT64_C( 1382694151414203393), INT64_C(                   0),
                            INT64_C(    4574054841401536), INT64_C( 8755017506026431112)) },
    { UINT8_C(225),
      easysimd_mm512_set_epi64(INT64_C(-3697729744057786539), INT64_C(-2459882991819182775),
                            INT64_C( 6065837030945349572), INT64_C( 8437722782224197038),
                            INT64_C( 1700648554253726454), INT64_C(-4293199790864835662),
                            INT64_C( 6581402203822969825), INT64_C(-6231169800047978744)),
      easysimd_mm512_set_epi64(INT64_C( 8326587265612039337), INT64_C( 6780517041864519531),
                            INT64_C(-7817226648374121699), INT64_C(-3500732471169369834),
                            INT64_C( 1796671772602068213), INT64_C( 1885612779837593615),
                            INT64_C(-6040660189943903948), INT64_C(  675381603587673544)),
      easysimd_mm512_set_epi64(INT64_C( 3675211075874242728), INT64_C(  144396663190979106),
                            INT64_C(-8970746854625959911), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(   25068767419582656)) },
    { UINT8_C(182),
      easysimd_mm512_set_epi64(INT64_C(-3172486234888138881), INT64_C( 3556874573334620913),
                            INT64_C(-7174772828994546158), INT64_C( -768272060832782008),
                            INT64_C(-7948383401788128664), INT64_C(-3962825949835743119),
                            INT64_C(-4177466042331622142), INT64_C(-4344904134560657490)),
      easysimd_mm512_set_epi64(INT64_C( 3131865100191000199), INT64_C( 3277342092864256055),
                            INT64_C( 2638156770812089616), INT64_C(-5499406567603861656),
                            INT64_C( 5836973950118592576), INT64_C(-4232123399129603430),
                            INT64_C( 8656431254350139121), INT64_C(-8853511068983619849)),
      easysimd_mm512_set_epi64(INT64_C( 2884148896870883456), INT64_C(                   0),
                            INT64_C( 2346526704673489152), INT64_C(  191440763308049952),
                            INT64_C(                   0), INT64_C(  307450048046744202),
                            INT64_C( 4044604204229069553), INT64_C(                   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_maskz_andnot_epi64(test_vec[i].k, test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_andnot_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_andnot_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_andnot_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_andnot_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_andnot_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_andnot_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_andnot_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_andnot_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_andnot_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_andnot_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_andnot_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_andnot_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_andnot_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_andnot_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_andnot_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_andnot_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_andnot_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_andnot_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_andnot_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_andnot_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_andnot_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_andnot_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_andnot_si512)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_andnot_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_andnot_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_andnot_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_andnot_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_andnot_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_andnot_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
