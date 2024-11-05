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

#define EASYSIMD_TEST_X86_AVX512_INSN sra

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/sra.h>

static int
test_easysimd_mm512_sra_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int64_t b[2];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C( 11061),  INT16_C(  2909),  INT16_C(  7771), -INT16_C( 15451),  INT16_C( 10536),  INT16_C( 16880),  INT16_C(  9564),  INT16_C(  7363),
         INT16_C( 22530), -INT16_C(  8028),  INT16_C(  1724), -INT16_C( 32593), -INT16_C( 10410),  INT16_C(  1671),  INT16_C( 28692),  INT16_C( 18993),
        -INT16_C( 29029), -INT16_C(  2475), -INT16_C(  1107), -INT16_C( 10822), -INT16_C( 21980), -INT16_C( 32490), -INT16_C(  9777), -INT16_C( 11619),
         INT16_C( 16689), -INT16_C(  4686),  INT16_C( 25159), -INT16_C( 25235), -INT16_C(  2759),  INT16_C( 19876), -INT16_C( 10907),  INT16_C(   407) },
      {  INT64_C(                   0),  INT64_C(                   0) },
      {  INT16_C( 11061),  INT16_C(  2909),  INT16_C(  7771), -INT16_C( 15451),  INT16_C( 10536),  INT16_C( 16880),  INT16_C(  9564),  INT16_C(  7363),
         INT16_C( 22530), -INT16_C(  8028),  INT16_C(  1724), -INT16_C( 32593), -INT16_C( 10410),  INT16_C(  1671),  INT16_C( 28692),  INT16_C( 18993),
        -INT16_C( 29029), -INT16_C(  2475), -INT16_C(  1107), -INT16_C( 10822), -INT16_C( 21980), -INT16_C( 32490), -INT16_C(  9777), -INT16_C( 11619),
         INT16_C( 16689), -INT16_C(  4686),  INT16_C( 25159), -INT16_C( 25235), -INT16_C(  2759),  INT16_C( 19876), -INT16_C( 10907),  INT16_C(   407) } },
    { { -INT16_C( 20629), -INT16_C( 19467),  INT16_C( 25361),  INT16_C( 19024), -INT16_C(  2984), -INT16_C( 17000),  INT16_C( 12234),  INT16_C( 11966),
        -INT16_C( 18916),  INT16_C(  1087),  INT16_C(  9575), -INT16_C( 15599), -INT16_C( 25054), -INT16_C(  1554), -INT16_C(  5175),  INT16_C( 13313),
        -INT16_C(  2406), -INT16_C( 21273),  INT16_C( 14425), -INT16_C( 19978), -INT16_C( 29140), -INT16_C(  2449),  INT16_C( 11710), -INT16_C(  9692),
         INT16_C( 25571),  INT16_C( 19423), -INT16_C(  3959), -INT16_C( 21746), -INT16_C(   882),  INT16_C( 22436), -INT16_C( 23065), -INT16_C( 32372) },
      {  INT64_C(                  13),  INT64_C(                   2) },
      { -INT16_C(     3), -INT16_C(     3),  INT16_C(     3),  INT16_C(     2), -INT16_C(     1), -INT16_C(     3),  INT16_C(     1),  INT16_C(     1),
        -INT16_C(     3),  INT16_C(     0),  INT16_C(     1), -INT16_C(     2), -INT16_C(     4), -INT16_C(     1), -INT16_C(     1),  INT16_C(     1),
        -INT16_C(     1), -INT16_C(     3),  INT16_C(     1), -INT16_C(     3), -INT16_C(     4), -INT16_C(     1),  INT16_C(     1), -INT16_C(     2),
         INT16_C(     3),  INT16_C(     2), -INT16_C(     1), -INT16_C(     3), -INT16_C(     1),  INT16_C(     2), -INT16_C(     3), -INT16_C(     4) } },
    { {  INT16_C( 10838), -INT16_C(  8334), -INT16_C( 32742), -INT16_C( 22389),  INT16_C( 12156),  INT16_C( 25344), -INT16_C( 29483),  INT16_C( 29156),
         INT16_C(  4863), -INT16_C( 21658),  INT16_C(  3382), -INT16_C(  6013),  INT16_C( 20771),  INT16_C( 26457), -INT16_C( 23484), -INT16_C( 25714),
         INT16_C(   206), -INT16_C(  6022),  INT16_C(  1408), -INT16_C(   880), -INT16_C( 28619),  INT16_C(  2655),  INT16_C( 17180),  INT16_C(  7291),
        -INT16_C(  7851), -INT16_C( 29753),  INT16_C( 19183),  INT16_C(  4724), -INT16_C( 12901), -INT16_C(  8071),  INT16_C(  1905),  INT16_C( 16251) },
      {  INT64_C(                   3),  INT64_C(                   7) },
      {  INT16_C(  1354), -INT16_C(  1042), -INT16_C(  4093), -INT16_C(  2799),  INT16_C(  1519),  INT16_C(  3168), -INT16_C(  3686),  INT16_C(  3644),
         INT16_C(   607), -INT16_C(  2708),  INT16_C(   422), -INT16_C(   752),  INT16_C(  2596),  INT16_C(  3307), -INT16_C(  2936), -INT16_C(  3215),
         INT16_C(    25), -INT16_C(   753),  INT16_C(   176), -INT16_C(   110), -INT16_C(  3578),  INT16_C(   331),  INT16_C(  2147),  INT16_C(   911),
        -INT16_C(   982), -INT16_C(  3720),  INT16_C(  2397),  INT16_C(   590), -INT16_C(  1613), -INT16_C(  1009),  INT16_C(   238),  INT16_C(  2031) } },
    { {  INT16_C( 18326), -INT16_C( 31481),  INT16_C( 31633),  INT16_C( 11672),  INT16_C(  4424), -INT16_C( 18163), -INT16_C( 30695),  INT16_C(  8440),
         INT16_C(  8061),  INT16_C( 30888),  INT16_C( 11222),  INT16_C(  7848), -INT16_C(  7666),  INT16_C( 13443),  INT16_C(   919),  INT16_C( 11951),
        -INT16_C( 18869), -INT16_C(  9037),  INT16_C( 19249),  INT16_C( 30985),  INT16_C(  5725),  INT16_C( 30258),  INT16_C( 10910),  INT16_C(  7318),
         INT16_C( 15945),  INT16_C(  8340),  INT16_C( 15722),  INT16_C( 30782), -INT16_C( 16097), -INT16_C( 18516),  INT16_C( 23493),  INT16_C(  4325) },
      {  INT64_C(                   4),  INT64_C(                   6) },
      {  INT16_C(  1145), -INT16_C(  1968),  INT16_C(  1977),  INT16_C(   729),  INT16_C(   276), -INT16_C(  1136), -INT16_C(  1919),  INT16_C(   527),
         INT16_C(   503),  INT16_C(  1930),  INT16_C(   701),  INT16_C(   490), -INT16_C(   480),  INT16_C(   840),  INT16_C(    57),  INT16_C(   746),
        -INT16_C(  1180), -INT16_C(   565),  INT16_C(  1203),  INT16_C(  1936),  INT16_C(   357),  INT16_C(  1891),  INT16_C(   681),  INT16_C(   457),
         INT16_C(   996),  INT16_C(   521),  INT16_C(   982),  INT16_C(  1923), -INT16_C(  1007), -INT16_C(  1158),  INT16_C(  1468),  INT16_C(   270) } },
    { {  INT16_C( 23436), -INT16_C(  2429), -INT16_C( 15720), -INT16_C( 18322),  INT16_C(  6787),  INT16_C( 18543),  INT16_C( 21621), -INT16_C( 30888),
         INT16_C( 17900), -INT16_C( 12085), -INT16_C( 30661),  INT16_C( 18193), -INT16_C( 14217), -INT16_C( 28174), -INT16_C( 18154), -INT16_C( 23819),
         INT16_C( 30741), -INT16_C( 21096),  INT16_C(  1594), -INT16_C( 16795), -INT16_C( 11232), -INT16_C( 27386),  INT16_C( 24360),  INT16_C(  5405),
        -INT16_C(  5980), -INT16_C(  8219), -INT16_C(  2192), -INT16_C(  6362),  INT16_C(  6591), -INT16_C( 10887),  INT16_C( 28370), -INT16_C(  6281) },
      {  INT64_C(                   3),  INT64_C(                  14) },
      {  INT16_C(  2929), -INT16_C(   304), -INT16_C(  1965), -INT16_C(  2291),  INT16_C(   848),  INT16_C(  2317),  INT16_C(  2702), -INT16_C(  3861),
         INT16_C(  2237), -INT16_C(  1511), -INT16_C(  3833),  INT16_C(  2274), -INT16_C(  1778), -INT16_C(  3522), -INT16_C(  2270), -INT16_C(  2978),
         INT16_C(  3842), -INT16_C(  2637),  INT16_C(   199), -INT16_C(  2100), -INT16_C(  1404), -INT16_C(  3424),  INT16_C(  3045),  INT16_C(   675),
        -INT16_C(   748), -INT16_C(  1028), -INT16_C(   274), -INT16_C(   796),  INT16_C(   823), -INT16_C(  1361),  INT16_C(  3546), -INT16_C(   786) } },
    { { -INT16_C(  3376),  INT16_C( 16583), -INT16_C(  4375), -INT16_C( 22489), -INT16_C( 24569), -INT16_C(  9858), -INT16_C(  2802), -INT16_C(  2623),
         INT16_C( 22021),  INT16_C(  6678), -INT16_C(  2736),  INT16_C(  8016),  INT16_C(  7130),  INT16_C(  7959),  INT16_C(  8963), -INT16_C( 11513),
        -INT16_C( 12523), -INT16_C(   493),  INT16_C( 15037), -INT16_C( 15193),  INT16_C(  9691), -INT16_C(  5731),  INT16_C( 24090),  INT16_C(  8158),
        -INT16_C(  2892),  INT16_C(  1338), -INT16_C( 29975), -INT16_C( 15324),  INT16_C( 15269), -INT16_C( 22301), -INT16_C(  5537),  INT16_C( 29819) },
      {  INT64_C(                  10),  INT64_C(                  14) },
      { -INT16_C(     4),  INT16_C(    16), -INT16_C(     5), -INT16_C(    22), -INT16_C(    24), -INT16_C(    10), -INT16_C(     3), -INT16_C(     3),
         INT16_C(    21),  INT16_C(     6), -INT16_C(     3),  INT16_C(     7),  INT16_C(     6),  INT16_C(     7),  INT16_C(     8), -INT16_C(    12),
        -INT16_C(    13), -INT16_C(     1),  INT16_C(    14), -INT16_C(    15),  INT16_C(     9), -INT16_C(     6),  INT16_C(    23),  INT16_C(     7),
        -INT16_C(     3),  INT16_C(     1), -INT16_C(    30), -INT16_C(    15),  INT16_C(    14), -INT16_C(    22), -INT16_C(     6),  INT16_C(    29) } },
    { { -INT16_C( 19616),  INT16_C( 18928),  INT16_C(  5181), -INT16_C(  7667), -INT16_C(  4016), -INT16_C( 20598),  INT16_C(  1499), -INT16_C( 27613),
        -INT16_C( 26989),  INT16_C( 23307),  INT16_C( 17840), -INT16_C(  4097), -INT16_C( 29667),  INT16_C( 21577), -INT16_C( 15625),  INT16_C( 22335),
         INT16_C( 12149), -INT16_C( 19807), -INT16_C( 20925), -INT16_C( 27756),  INT16_C(  7839),  INT16_C( 31298),  INT16_C( 26147), -INT16_C( 18930),
         INT16_C(  6652), -INT16_C( 21231),  INT16_C(  4191),  INT16_C( 31900), -INT16_C(  6756), -INT16_C( 27440),  INT16_C(  4007),  INT16_C(  7403) },
      {  INT64_C(                  13),  INT64_C(                  11) },
      { -INT16_C(     3),  INT16_C(     2),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     3),  INT16_C(     0), -INT16_C(     4),
        -INT16_C(     4),  INT16_C(     2),  INT16_C(     2), -INT16_C(     1), -INT16_C(     4),  INT16_C(     2), -INT16_C(     2),  INT16_C(     2),
         INT16_C(     1), -INT16_C(     3), -INT16_C(     3), -INT16_C(     4),  INT16_C(     0),  INT16_C(     3),  INT16_C(     3), -INT16_C(     3),
         INT16_C(     0), -INT16_C(     3),  INT16_C(     0),  INT16_C(     3), -INT16_C(     1), -INT16_C(     4),  INT16_C(     0),  INT16_C(     0) } },
    { {  INT16_C( 27516), -INT16_C(  9369),  INT16_C(  1147),  INT16_C(  6231),  INT16_C( 10473), -INT16_C( 28244), -INT16_C( 26825),  INT16_C( 30381),
         INT16_C( 31780),  INT16_C( 24568),  INT16_C(  3550),  INT16_C( 24377), -INT16_C( 29339),  INT16_C(  8962),  INT16_C( 23791),  INT16_C( 27614),
         INT16_C( 17863),  INT16_C( 16966), -INT16_C( 25015),  INT16_C( 13146),  INT16_C(  1734), -INT16_C(   572),  INT16_C( 29086), -INT16_C( 15757),
         INT16_C( 27629), -INT16_C( 13279),  INT16_C( 23161), -INT16_C(  8661),  INT16_C( 11751), -INT16_C( 10750), -INT16_C(  8055),  INT16_C( 20546) },
      {  INT64_C(                  14),  INT64_C(                  12) },
      {  INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     2), -INT16_C(     2),  INT16_C(     1),
         INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     1), -INT16_C(     2),  INT16_C(     0),  INT16_C(     1),  INT16_C(     1),
         INT16_C(     1),  INT16_C(     1), -INT16_C(     2),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     1), -INT16_C(     1),
         INT16_C(     1), -INT16_C(     1),  INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     1) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sra_epi16(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_sra_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_sra_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[16];
    const int64_t b[2];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1817164434),  INT32_C(  1163541606),  INT32_C(   555371306),  INT32_C(   100359090), -INT32_C(  1369349290),  INT32_C(   883287359),  INT32_C(   532606548), -INT32_C(  1234472632),
         INT32_C(   474553526), -INT32_C(  2107464616), -INT32_C(  1583121169),  INT32_C(   765894615),  INT32_C(  1104873218),  INT32_C(  1098220013), -INT32_C(  1218431889),  INT32_C(  1533987749) },
      {  INT64_C( 5458883209279813607),  INT64_C(  457839958291146069) },
      { -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),
         INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0) } },
    { { -INT32_C(  2025376102),  INT32_C(   650689719), -INT32_C(  1780668176), -INT32_C(   605008908),  INT32_C(  1578854403), -INT32_C(  1750475711),  INT32_C(  1170511425), -INT32_C(  1018487255),
         INT32_C(   189502036),  INT32_C(  1060180815),  INT32_C(   819269436),  INT32_C(  1561052506),  INT32_C(  1874601517),  INT32_C(  1141269763),  INT32_C(   613075707),  INT32_C(  1038669288) },
      { -INT64_C( 9009879560366836888), -INT64_C( 4737295799217370231) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   416416284),  INT32_C(  1579955042), -INT32_C(   427641091), -INT32_C(   215782773), -INT32_C(   458593379),  INT32_C(  1852219877), -INT32_C(    44951188), -INT32_C(  1145269289),
        -INT32_C(   224138864),  INT32_C(   441503517),  INT32_C(   268489605), -INT32_C(   620485827),  INT32_C(  1958719119), -INT32_C(  1125964465),  INT32_C(   364459326),  INT32_C(  1490122184) },
      { -INT64_C( 3992833305466997665),  INT64_C( 5298164705953093486) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),
        -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   625064565), -INT32_C(  1449746069), -INT32_C(  1631629098),  INT32_C(   620138693),  INT32_C(  1201684484),  INT32_C(  1225734107), -INT32_C(  1997150258),  INT32_C(  1339194308),
         INT32_C(   740921537),  INT32_C(   131448881), -INT32_C(   710568944),  INT32_C(   687512356), -INT32_C(  1217422629), -INT32_C(  1627357487),  INT32_C(   707327334),  INT32_C(   846854769) },
      {  INT64_C( 8269228364949004938), -INT64_C( 2516032236495018040) },
      { -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(  1365998372),  INT32_C(  1766692099), -INT32_C(    74222198), -INT32_C(   114488209),  INT32_C(   297044910), -INT32_C(  2021427521), -INT32_C(   546059299), -INT32_C(   356677619),
         INT32_C(   278417677),  INT32_C(  1903814118), -INT32_C(   898888613), -INT32_C(   960194024), -INT32_C(   455641051), -INT32_C(   865379345),  INT32_C(   883744550),  INT32_C(  1948150119) },
      { -INT64_C(  715230545926900038),  INT64_C( 4533019363640442377) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),
         INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   333266947),  INT32_C(  1102679578), -INT32_C(   730503827), -INT32_C(  2008443954), -INT32_C(   467022519), -INT32_C(   723829813), -INT32_C(   755589958),  INT32_C(   504421921),
        -INT32_C(  1207225443),  INT32_C(   788120769), -INT32_C(   150835671),  INT32_C(  1249856257), -INT32_C(   466704103), -INT32_C(  1632106012), -INT32_C(   965628251),  INT32_C(   719683980) },
      { -INT64_C( 2475893802005237580),  INT64_C(  762326532740130377) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),
        -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0) } },
    { { -INT32_C(   504446212),  INT32_C(  1904191436), -INT32_C(   499584939),  INT32_C(   638328178), -INT32_C(  1046680051),  INT32_C(   312360905), -INT32_C(   681807131), -INT32_C(  1025380155),
         INT32_C(  2124665010), -INT32_C(   856677769), -INT32_C(  2052184045),  INT32_C(  1386986053),  INT32_C(  1897088936),  INT32_C(  1820569991), -INT32_C(   398205149), -INT32_C(  2102778417) },
      { -INT64_C( 8991172290406232587),  INT64_C( 5381717585363003160) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),
         INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1) } },
    { { -INT32_C(  2101624069), -INT32_C(  1762771085), -INT32_C(   293653986),  INT32_C(  1282419031), -INT32_C(   407342985),  INT32_C(  2037051489),  INT32_C(  2044162775),  INT32_C(   549684773),
        -INT32_C(  1130201527), -INT32_C(   598502979),  INT32_C(   449499843),  INT32_C(  1919302395),  INT32_C(   190389930), -INT32_C(   427441394),  INT32_C(  1532976181),  INT32_C(   729490146) },
      {  INT64_C( 8230956893370588832), -INT64_C( 1694494486529571827) },
      { -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sra_epi32(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_sra_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m512i r = easysimd_mm512_sra_epi32(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_sra_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[2];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 1540429207916076067),  INT64_C( 3491576225965866060), -INT64_C( 1156955728976190958), -INT64_C( 8236891946831463248),
         INT64_C( 3050038509106178314), -INT64_C( 2654117947383466010),  INT64_C(  784398582962499599),  INT64_C( 8102267537843664762) },
      { -INT64_C( 3874782450354816462),  INT64_C( 3377118212050416184) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1),
         INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C( 2232936145637758561), -INT64_C( 7525172594861581125),  INT64_C( 8709927869715267377),  INT64_C( 3648573839421056084),
        -INT64_C( 8986600147476823296),  INT64_C( 8612629911882174090),  INT64_C( 7146205523439529588), -INT64_C(  228142080124803824) },
      { -INT64_C( 5149958335232802756), -INT64_C( 3136988725089195077) },
      { -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1) } },
    { { -INT64_C( 9165189796718967717),  INT64_C( 6580313339726112089),  INT64_C( 2080353382821770744),  INT64_C( 1028391584046875921),
        -INT64_C( 2699829357090058856), -INT64_C( 7248964012938843155), -INT64_C( 4849075339420121802), -INT64_C(   79531632386780892) },
      { -INT64_C( 7882548780727837121), -INT64_C( 8972854579383317701) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1) } },
    { { -INT64_C(   57755303732888630),  INT64_C( 1226592984277344161), -INT64_C( 2931441362440942567), -INT64_C( 4753778154421227720),
        -INT64_C( 7834743885506862485), -INT64_C( 1033400366171284283),  INT64_C( 2242513232322394796),  INT64_C( 8044461396934841585) },
      {  INT64_C(   95695155943036622),  INT64_C( 4098065159973039382) },
      { -INT64_C(                   1),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1),
        -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C( 9096532853784539205), -INT64_C( 2312869912992389516), -INT64_C( 5972350804284171040), -INT64_C( 5718310432345967137),
         INT64_C( 4042849024625288457),  INT64_C( 7740728485159666043), -INT64_C( 2488338854872413389), -INT64_C( 7638124156620828071) },
      { -INT64_C( 2157143511169871341), -INT64_C( 8376173758990335797) },
      { -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1) } },
    { {  INT64_C( 5026523976937326822),  INT64_C(  936628847448886238),  INT64_C( 4768735274155060964),  INT64_C( 6671899736525020340),
        -INT64_C( 7434877038873219679), -INT64_C( 2516945977470720499), -INT64_C( 7067741103672724176),  INT64_C( 9062721140545913586) },
      {  INT64_C( 1477142228811580717), -INT64_C( 5541348924341976765) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0) } },
    { { -INT64_C( 8527489379393126110), -INT64_C( 4674253102085156409),  INT64_C( 5609274059774110908),  INT64_C( 4746127837059740582),
         INT64_C( 4973368708300552419),  INT64_C( 7680336856783320645),  INT64_C(  543816669746576437), -INT64_C( 7324524564888611335) },
      { -INT64_C( 2230941409079875769),  INT64_C( 3542167954365411856) },
      { -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1) } },
    { {  INT64_C( 6265347507642979978),  INT64_C( 8819113719564805137), -INT64_C( 6634565784427333520),  INT64_C( 9036905805011592508),
        -INT64_C( 1688468509120996641),  INT64_C( 4191747114460512382), -INT64_C( 1026482387073733748), -INT64_C( 8482368631606795560) },
      { -INT64_C( 8562654560621514644),  INT64_C( 8889094041291709272) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0),
        -INT64_C(                   1),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sra_epi64(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_sra_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m512i r = easysimd_mm512_sra_epi64(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sra_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sra_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sra_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
