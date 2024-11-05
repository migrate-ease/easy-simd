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

#define EASYSIMD_TEST_X86_AVX512_INSN srli

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/srli.h>

static int
test_easysimd_mm512_srli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C(  5064), -INT16_C( 13372), -INT16_C( 27857), -INT16_C( 22226),  INT16_C( 15192),  INT16_C( 15375), -INT16_C(  9626), -INT16_C( 29567),
        -INT16_C( 11953), -INT16_C(  3869), -INT16_C( 31356),  INT16_C( 27097), -INT16_C( 12959),  INT16_C( 26598), -INT16_C( 26711),  INT16_C( 28952),
        -INT16_C(  8790), -INT16_C(  9924),  INT16_C( 27504), -INT16_C( 14205), -INT16_C( 27994),  INT16_C(  3076), -INT16_C( 31124), -INT16_C( 17511),
         INT16_C( 31831), -INT16_C(  9045), -INT16_C( 31743),  INT16_C( 25157),  INT16_C( 11345), -INT16_C(  1335), -INT16_C(  7485),  INT16_C( 28011) },
      {  INT16_C(    39),  INT16_C(   407),  INT16_C(   294),  INT16_C(   338),  INT16_C(   118),  INT16_C(   120),  INT16_C(   436),  INT16_C(   281),
         INT16_C(   418),  INT16_C(   481),  INT16_C(   267),  INT16_C(   211),  INT16_C(   410),  INT16_C(   207),  INT16_C(   303),  INT16_C(   226),
         INT16_C(   443),  INT16_C(   434),  INT16_C(   214),  INT16_C(   401),  INT16_C(   293),  INT16_C(    24),  INT16_C(   268),  INT16_C(   375),
         INT16_C(   248),  INT16_C(   441),  INT16_C(   264),  INT16_C(   196),  INT16_C(    88),  INT16_C(   501),  INT16_C(   453),  INT16_C(   218) } },
    { { -INT16_C( 22593),  INT16_C( 12102), -INT16_C( 14062), -INT16_C( 17928), -INT16_C(   933), -INT16_C( 14395),  INT16_C( 24194), -INT16_C(  9598),
         INT16_C( 11738), -INT16_C(  9290), -INT16_C(  1103),  INT16_C(   574),  INT16_C(  1831), -INT16_C(  5380),  INT16_C( 26601), -INT16_C( 22441),
        -INT16_C( 25073),  INT16_C(  8664), -INT16_C( 12185), -INT16_C( 15398), -INT16_C( 24372),  INT16_C( 20362),  INT16_C(  3582), -INT16_C(  9943),
        -INT16_C(  8390), -INT16_C(  4940), -INT16_C(  3366),  INT16_C(   750), -INT16_C(  5126), -INT16_C(  7188),  INT16_C( 17490),  INT16_C( 24972) },
      {  INT16_C(   335),  INT16_C(    94),  INT16_C(   402),  INT16_C(   371),  INT16_C(   504),  INT16_C(   399),  INT16_C(   189),  INT16_C(   437),
         INT16_C(    91),  INT16_C(   439),  INT16_C(   503),  INT16_C(     4),  INT16_C(    14),  INT16_C(   469),  INT16_C(   207),  INT16_C(   336),
         INT16_C(   316),  INT16_C(    67),  INT16_C(   416),  INT16_C(   391),  INT16_C(   321),  INT16_C(   159),  INT16_C(    27),  INT16_C(   434),
         INT16_C(   446),  INT16_C(   473),  INT16_C(   485),  INT16_C(     5),  INT16_C(   471),  INT16_C(   455),  INT16_C(   136),  INT16_C(   195) } },
    { {  INT16_C( 25826),  INT16_C( 18819),  INT16_C( 23860),  INT16_C(    12), -INT16_C( 26627), -INT16_C(   945),  INT16_C( 30884), -INT16_C(  8491),
        -INT16_C( 30377),  INT16_C( 13002), -INT16_C( 18052),  INT16_C( 30260),  INT16_C(  8356), -INT16_C(  2471), -INT16_C(  6812),  INT16_C( 18008),
        -INT16_C(  9399),  INT16_C( 32144), -INT16_C( 25544),  INT16_C( 13950), -INT16_C( 13005), -INT16_C( 10446),  INT16_C(  1862), -INT16_C( 25162),
        -INT16_C( 32624),  INT16_C(  3279),  INT16_C(   825), -INT16_C(  8830), -INT16_C(  9180), -INT16_C( 30508),  INT16_C( 11457),  INT16_C(  3023) },
      {  INT16_C(   201),  INT16_C(   147),  INT16_C(   186),  INT16_C(     0),  INT16_C(   303),  INT16_C(   504),  INT16_C(   241),  INT16_C(   445),
         INT16_C(   274),  INT16_C(   101),  INT16_C(   370),  INT16_C(   236),  INT16_C(    65),  INT16_C(   492),  INT16_C(   458),  INT16_C(   140),
         INT16_C(   438),  INT16_C(   251),  INT16_C(   312),  INT16_C(   108),  INT16_C(   410),  INT16_C(   430),  INT16_C(    14),  INT16_C(   315),
         INT16_C(   257),  INT16_C(    25),  INT16_C(     6),  INT16_C(   443),  INT16_C(   440),  INT16_C(   273),  INT16_C(    89),  INT16_C(    23) } },
    { {  INT16_C( 24327),  INT16_C( 16264),  INT16_C(  1787),  INT16_C( 12149), -INT16_C( 22572),  INT16_C(  6662), -INT16_C( 17234),  INT16_C( 16311),
        -INT16_C( 30915),  INT16_C( 30283), -INT16_C( 12662), -INT16_C( 20908),  INT16_C( 10410),  INT16_C( 27447),  INT16_C(  1620),  INT16_C( 23414),
        -INT16_C(   155),  INT16_C( 24730),  INT16_C(  4101), -INT16_C(  9841), -INT16_C( 26953),  INT16_C( 26355), -INT16_C( 21678), -INT16_C( 28763),
        -INT16_C(  4046), -INT16_C( 17402),  INT16_C( 23230),  INT16_C( 26731), -INT16_C( 23934), -INT16_C( 10540),  INT16_C( 19112),  INT16_C(  3377) },
      {  INT16_C(   190),  INT16_C(   127),  INT16_C(    13),  INT16_C(    94),  INT16_C(   335),  INT16_C(    52),  INT16_C(   377),  INT16_C(   127),
         INT16_C(   270),  INT16_C(   236),  INT16_C(   413),  INT16_C(   348),  INT16_C(    81),  INT16_C(   214),  INT16_C(    12),  INT16_C(   182),
         INT16_C(   510),  INT16_C(   193),  INT16_C(    32),  INT16_C(   435),  INT16_C(   301),  INT16_C(   205),  INT16_C(   342),  INT16_C(   287),
         INT16_C(   480),  INT16_C(   376),  INT16_C(   181),  INT16_C(   208),  INT16_C(   325),  INT16_C(   429),  INT16_C(   149),  INT16_C(    26) } },
    { { -INT16_C( 13495),  INT16_C( 20333), -INT16_C(   549), -INT16_C( 27864),  INT16_C(  7315), -INT16_C(  6663), -INT16_C( 24889), -INT16_C(  1675),
         INT16_C( 31630),  INT16_C( 19893),  INT16_C(  8405),  INT16_C( 22453), -INT16_C( 30270),  INT16_C( 27181),  INT16_C( 24276),  INT16_C(  7543),
        -INT16_C(  6871),  INT16_C(  1388), -INT16_C( 27166),  INT16_C( 30104), -INT16_C( 28239),  INT16_C( 30810), -INT16_C( 12497), -INT16_C( 17039),
         INT16_C(  9802),  INT16_C(  7946), -INT16_C( 16313),  INT16_C(  2422), -INT16_C( 23735),  INT16_C(  7540), -INT16_C(  5375),  INT16_C( 11067) },
      {  INT16_C(   406),  INT16_C(   158),  INT16_C(   507),  INT16_C(   294),  INT16_C(    57),  INT16_C(   459),  INT16_C(   317),  INT16_C(   498),
         INT16_C(   247),  INT16_C(   155),  INT16_C(    65),  INT16_C(   175),  INT16_C(   275),  INT16_C(   212),  INT16_C(   189),  INT16_C(    58),
         INT16_C(   458),  INT16_C(    10),  INT16_C(   299),  INT16_C(   235),  INT16_C(   291),  INT16_C(   240),  INT16_C(   414),  INT16_C(   378),
         INT16_C(    76),  INT16_C(    62),  INT16_C(   384),  INT16_C(    18),  INT16_C(   326),  INT16_C(    58),  INT16_C(   470),  INT16_C(    86) } },
    { { -INT16_C( 22576), -INT16_C( 19920), -INT16_C( 14276), -INT16_C(  4825), -INT16_C( 32167), -INT16_C( 30619), -INT16_C( 10671), -INT16_C( 25531),
         INT16_C( 20733),  INT16_C( 17595),  INT16_C( 12816),  INT16_C( 22861), -INT16_C( 15915), -INT16_C( 10377), -INT16_C( 19795),  INT16_C( 32002),
         INT16_C( 12889), -INT16_C( 27088),  INT16_C( 22522),  INT16_C( 21379), -INT16_C(  5671),  INT16_C( 11227),  INT16_C(  8383), -INT16_C( 17209),
        -INT16_C( 32144),       INT16_MIN,  INT16_C( 20148), -INT16_C( 29990),  INT16_C( 20751), -INT16_C( 17311),  INT16_C( 25347),  INT16_C( 23610) },
      {  INT16_C(   335),  INT16_C(   356),  INT16_C(   400),  INT16_C(   474),  INT16_C(   260),  INT16_C(   272),  INT16_C(   428),  INT16_C(   312),
         INT16_C(   161),  INT16_C(   137),  INT16_C(   100),  INT16_C(   178),  INT16_C(   387),  INT16_C(   430),  INT16_C(   357),  INT16_C(   250),
         INT16_C(   100),  INT16_C(   300),  INT16_C(   175),  INT16_C(   167),  INT16_C(   467),  INT16_C(    87),  INT16_C(    65),  INT16_C(   377),
         INT16_C(   260),  INT16_C(   256),  INT16_C(   157),  INT16_C(   277),  INT16_C(   162),  INT16_C(   376),  INT16_C(   198),  INT16_C(   184) } },
    { {  INT16_C( 27285), -INT16_C( 28686),  INT16_C( 30401), -INT16_C( 25630), -INT16_C( 17057),  INT16_C(  7878), -INT16_C( 29219),  INT16_C( 20187),
        -INT16_C(  9457), -INT16_C( 15154), -INT16_C( 22487),  INT16_C( 14670), -INT16_C( 20487), -INT16_C(   779),  INT16_C( 12050), -INT16_C( 22695),
         INT16_C( 19353),  INT16_C( 23350),  INT16_C(  6337),  INT16_C(  8438), -INT16_C( 17195), -INT16_C( 19905),  INT16_C(  6729),  INT16_C( 22528),
        -INT16_C( 12299),  INT16_C(  7964),  INT16_C( 27255),  INT16_C( 29016),  INT16_C( 19737),  INT16_C( 11117), -INT16_C( 14723),  INT16_C(  5842) },
      {  INT16_C(   213),  INT16_C(   287),  INT16_C(   237),  INT16_C(   311),  INT16_C(   378),  INT16_C(    61),  INT16_C(   283),  INT16_C(   157),
         INT16_C(   438),  INT16_C(   393),  INT16_C(   336),  INT16_C(   114),  INT16_C(   351),  INT16_C(   505),  INT16_C(    94),  INT16_C(   334),
         INT16_C(   151),  INT16_C(   182),  INT16_C(    49),  INT16_C(    65),  INT16_C(   377),  INT16_C(   356),  INT16_C(    52),  INT16_C(   176),
         INT16_C(   415),  INT16_C(    62),  INT16_C(   212),  INT16_C(   226),  INT16_C(   154),  INT16_C(    86),  INT16_C(   396),  INT16_C(    45) } },
    { {  INT16_C(  2066), -INT16_C( 11407),  INT16_C( 26400), -INT16_C(  2572),  INT16_C( 13091),  INT16_C( 27816), -INT16_C( 22451),  INT16_C( 17093),
        -INT16_C(  7817), -INT16_C(  4255), -INT16_C( 18100),  INT16_C( 25952), -INT16_C( 13049), -INT16_C( 31599),  INT16_C( 25492), -INT16_C( 22886),
         INT16_C(  3180), -INT16_C( 29575),  INT16_C( 28019), -INT16_C( 26750),  INT16_C( 10912), -INT16_C(  4861), -INT16_C( 14126),  INT16_C( 18992),
        -INT16_C( 28246), -INT16_C(  2503), -INT16_C( 26293),  INT16_C( 21083), -INT16_C(  5018), -INT16_C(  1322),  INT16_C( 28752), -INT16_C( 17248) },
      {  INT16_C(    16),  INT16_C(   422),  INT16_C(   206),  INT16_C(   491),  INT16_C(   102),  INT16_C(   217),  INT16_C(   336),  INT16_C(   133),
         INT16_C(   450),  INT16_C(   478),  INT16_C(   370),  INT16_C(   202),  INT16_C(   410),  INT16_C(   265),  INT16_C(   199),  INT16_C(   333),
         INT16_C(    24),  INT16_C(   280),  INT16_C(   218),  INT16_C(   303),  INT16_C(    85),  INT16_C(   474),  INT16_C(   401),  INT16_C(   148),
         INT16_C(   291),  INT16_C(   492),  INT16_C(   306),  INT16_C(   164),  INT16_C(   472),  INT16_C(   501),  INT16_C(   224),  INT16_C(   377) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srli_epi16(a, 7);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_srli_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_srli_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    unsigned int imm8;
    easysimd__m512i r;
  } test_vec[8] = {
     { easysimd_mm512_set_epi32(INT32_C(-2020822652), INT32_C( -257395769), INT32_C(  499270536), INT32_C( 1400500940),
                            INT32_C( 1373098033), INT32_C( 1102869287), INT32_C( 1033807112), INT32_C(-1561080563),
                            INT32_C( 1506432231), INT32_C(-1063413574), INT32_C(  341686905), INT32_C( -287206476),
                            INT32_C(  265122437), INT32_C( 1398620072), INT32_C( -665611582), INT32_C(-1504345300)),
      22,
      easysimd_mm512_set_epi32(INT32_C(        542), INT32_C(        962), INT32_C(        119), INT32_C(        333),
                            INT32_C(        327), INT32_C(        262), INT32_C(        246), INT32_C(        651),
                            INT32_C(        359), INT32_C(        770), INT32_C(         81), INT32_C(        955),
                            INT32_C(         63), INT32_C(        333), INT32_C(        865), INT32_C(        665)) },
    { easysimd_mm512_set_epi32(INT32_C( -819412546), INT32_C(-1796000475), INT32_C(-1622143581), INT32_C(-1041781306),
                            INT32_C( -155789533), INT32_C( -151174821), INT32_C( 1958936143), INT32_C( -107637458),
                            INT32_C(-1381646204), INT32_C( 1022043250), INT32_C( -104481290), INT32_C(-1025833596),
                            INT32_C( 1618482767), INT32_C( 1888220027), INT32_C(-1169248526), INT32_C( -703447035)),
      11,
      easysimd_mm512_set_epi32(INT32_C(    1697048), INT32_C(    1220198), INT32_C(    1305089), INT32_C(    1588469),
                            INT32_C(    2021082), INT32_C(    2023336), INT32_C(     956511), INT32_C(    2044594),
                            INT32_C(    1422520), INT32_C(     499044), INT32_C(    2046135), INT32_C(    1596256),
                            INT32_C(     790274), INT32_C(     921982), INT32_C(    1526229), INT32_C(    1753672)) },
    { easysimd_mm512_set_epi32(INT32_C(-1594292345), INT32_C( -684588879), INT32_C( 1676697175), INT32_C( -659819552),
                            INT32_C( 1894934939), INT32_C(  577900071), INT32_C(  818876053), INT32_C(  557599341),
                            INT32_C( -791753790), INT32_C(  286117889), INT32_C( 1667395914), INT32_C( -574374162),
                            INT32_C(  516383634), INT32_C( 1867216785), INT32_C( 1360165420), INT32_C(-1026060155)),
      3,
      easysimd_mm512_set_epi32(INT32_C(  337584368), INT32_C(  451297302), INT32_C(  209587146), INT32_C(  454393468),
                            INT32_C(  236866867), INT32_C(   72237508), INT32_C(  102359506), INT32_C(   69699917),
                            INT32_C(  437901688), INT32_C(   35764736), INT32_C(  208424489), INT32_C(  465074141),
                            INT32_C(   64547954), INT32_C(  233402098), INT32_C(  170020677), INT32_C(  408613392)) },
    { easysimd_mm512_set_epi32(INT32_C(  563818649), INT32_C( 1327166173), INT32_C( 1236848070), INT32_C( -448866475),
                            INT32_C( -173418493), INT32_C(-1571972356), INT32_C( 1881284471), INT32_C(  439987043),
                            INT32_C(  508631938), INT32_C(  763400402), INT32_C( 2004762594), INT32_C(-1789579909),
                            INT32_C( -823229171), INT32_C(-1537029967), INT32_C(-2094893814), INT32_C( 1910734558)),
      0,
      easysimd_mm512_set_epi32(INT32_C(  563818649), INT32_C( 1327166173), INT32_C( 1236848070), INT32_C( -448866475),
                            INT32_C( -173418493), INT32_C(-1571972356), INT32_C( 1881284471), INT32_C(  439987043),
                            INT32_C(  508631938), INT32_C(  763400402), INT32_C( 2004762594), INT32_C(-1789579909),
                            INT32_C( -823229171), INT32_C(-1537029967), INT32_C(-2094893814), INT32_C( 1910734558)) },
    { easysimd_mm512_set_epi32(INT32_C( 1331571680), INT32_C(-1968130549), INT32_C(-1401578233), INT32_C(-1310278942),
                            INT32_C( -553135974), INT32_C(  390049321), INT32_C( -502176380), INT32_C( -721913400),
                            INT32_C(  297997941), INT32_C(  812527594), INT32_C(-1593317379), INT32_C( -643296593),
                            INT32_C(-1978632480), INT32_C(-2010319907), INT32_C(-1081044111), INT32_C(  223565748)),
      26,
      easysimd_mm512_set_epi32(INT32_C(         19), INT32_C(         34), INT32_C(         43), INT32_C(         44),
                            INT32_C(         55), INT32_C(          5), INT32_C(         56), INT32_C(         53),
                            INT32_C(          4), INT32_C(         12), INT32_C(         40), INT32_C(         54),
                            INT32_C(         34), INT32_C(         34), INT32_C(         47), INT32_C(          3)) },
    { easysimd_mm512_set_epi32(INT32_C(   69766264), INT32_C( 1121309360), INT32_C( -164257344), INT32_C( 1544624998),
                            INT32_C(-1638151086), INT32_C(  617641637), INT32_C(-2109782153), INT32_C( -381251627),
                            INT32_C(  648330089), INT32_C( -370018417), INT32_C(-1896387892), INT32_C(-1167774485),
                            INT32_C( -297453838), INT32_C( -617551956), INT32_C(  863958459), INT32_C( 1052098740)),
      1,
      easysimd_mm512_set_epi32(INT32_C(   34883132), INT32_C(  560654680), INT32_C( 2065354976), INT32_C(  772312499),
                            INT32_C( 1328408105), INT32_C(  308820818), INT32_C( 1092592571), INT32_C( 1956857834),
                            INT32_C(  324165044), INT32_C( 1962474439), INT32_C( 1199289702), INT32_C( 1563596405),
                            INT32_C( 1998756729), INT32_C( 1838707670), INT32_C(  431979229), INT32_C(  526049370)) },
    { easysimd_mm512_set_epi32(INT32_C( -185630809), INT32_C( -795283306), INT32_C( 1353888329), INT32_C( 1750377549),
                            INT32_C( -609950002), INT32_C(-2070799804), INT32_C( -717783400), INT32_C( -489437394),
                            INT32_C(  782151967), INT32_C( -135381456), INT32_C(-1044185983), INT32_C(-1168288861),
                            INT32_C( 1570077349), INT32_C(-1514349775), INT32_C(-1300428717), INT32_C(-1070450073)),
      14,
      easysimd_mm512_set_epi32(INT32_C(     250813), INT32_C(     213603), INT32_C(      82634), INT32_C(     106834),
                            INT32_C(     224915), INT32_C(     135752), INT32_C(     218333), INT32_C(     232271),
                            INT32_C(      47738), INT32_C(     253880), INT32_C(     198411), INT32_C(     190837),
                            INT32_C(      95829), INT32_C(     169715), INT32_C(     182772), INT32_C(     196808)) },
    { easysimd_mm512_set_epi32(INT32_C(  858780966), INT32_C(  471539970), INT32_C(  308326365), INT32_C(  897623009),
                            INT32_C(  274412137), INT32_C(-1363032868), INT32_C( 2080428503), INT32_C( 1048755350),
                            INT32_C( -342337536), INT32_C( 1475004820), INT32_C( 1074270282), INT32_C( -894671787),
                            INT32_C(-2107817427), INT32_C( -444084191), INT32_C(  851286899), INT32_C( 1423269304)),
      1,
      easysimd_mm512_set_epi32(INT32_C(  429390483), INT32_C(  235769985), INT32_C(  154163182), INT32_C(  448811504),
                            INT32_C(  137206068), INT32_C( 1465967214), INT32_C( 1040214251), INT32_C(  524377675),
                            INT32_C( 1976314880), INT32_C(  737502410), INT32_C(  537135141), INT32_C( 1700147754),
                            INT32_C( 1093574934), INT32_C( 1925441552), INT32_C(  425643449), INT32_C(  711634652)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    unsigned int b = test_vec[i].imm8;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srli_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_srli_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_srli_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    unsigned int b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 7973262903512536694), INT64_C( -756652926976123625),
                            INT64_C(-7907329678808178856), INT64_C(-4613066309848201378),
                            INT64_C(  911796452309072772), INT64_C(-7947449538018331043),
                            INT64_C(-4094891379879736374), INT64_C( 2567785713935265105)),
      0xab,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { easysimd_mm512_set_epi64(INT64_C(-8733599303468285770), INT64_C(-1221042997940104437),
                            INT64_C( 1700326984023276146), INT64_C(  299160601816116482),
                            INT64_C(-8645581509002533463), INT64_C(-8083364442012234823),
                            INT64_C(-5545717914343726512), INT64_C(  419833451025710133)),
      0x8029,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { easysimd_mm512_set_epi64(INT64_C(-4893117706119522679), INT64_C( 5420919605114671392),
                            INT64_C( 5087336166907414861), INT64_C( -562883644934737039),
                            INT64_C(-3838729031805448294), INT64_C(-7418143174233432135),
                            INT64_C( 7017318210470297490), INT64_C( 3464955922400203693)),
      0xe,
      easysimd_mm512_set_epi64(INT64_C(     827247703099977), INT64_C(     330866675116862),
                            INT64_C(     310506357843470), INT64_C(    1091544215623462),
                            INT64_C(     891602480584967), INT64_C(     673132379118415),
                            INT64_C(     428303113432024), INT64_C(     211484126123059)) },
    { easysimd_mm512_set_epi64(INT64_C( 2541614580543521019), INT64_C(  499823435321299561),
                            INT64_C(-5904924501366764508), INT64_C( 7023944739814045444),
                            INT64_C( 6015406288340926104), INT64_C( 7321833489159498588),
                            INT64_C(-2737849912327243109), INT64_C(-3578554550642761007)),
      0x32,
      easysimd_mm512_set_epi64(INT64_C(                2257), INT64_C(                 443),
                            INT64_C(               11139), INT64_C(                6238),
                            INT64_C(                5342), INT64_C(                6503),
                            INT64_C(               13952), INT64_C(               13205)) },
    { easysimd_mm512_set_epi64(INT64_C(-5028928596309812666), INT64_C(-4599097054342878650),
                            INT64_C( 1737746464556527965), INT64_C( 7519897503489365685),
                            INT64_C( 2668093889339798821), INT64_C(-3758388356888738937),
                            INT64_C(-2613982157457207556), INT64_C(-2225410235035714021)),
      0x30,
      easysimd_mm512_set_epi64(INT64_C(               47669), INT64_C(               49196),
                            INT64_C(                6173), INT64_C(               26716),
                            INT64_C(                9478), INT64_C(               52183),
                            INT64_C(               56249), INT64_C(               57629)) },
    { easysimd_mm512_set_epi64(INT64_C(-5661929570079819163), INT64_C(  606174630548676143),
                            INT64_C( 4062026724724267051), INT64_C(-7721509817758052189),
                            INT64_C(-4899766988012067491), INT64_C(  849655025943263586),
                            INT64_C(-7243604229092766255), INT64_C(-3011226666080476035)),
      0x29,
      easysimd_mm512_set_epi64(INT64_C(             5813860), INT64_C(              275656),
                            INT64_C(             1847195), INT64_C(             4877271),
                            INT64_C(             6160451), INT64_C(              386378),
                            INT64_C(             5094598), INT64_C(             7019260)) },
    { easysimd_mm512_set_epi64(INT64_C( 3357536311959110775), INT64_C( 4508830932063799722),
                            INT64_C(-5800425134717732029), INT64_C( 1782066721260114087),
                            INT64_C( -181633913032181218), INT64_C(-5152953019677919849),
                            INT64_C( 3009514543526146963), INT64_C( -248934049093542484)),
      0x31,
      easysimd_mm512_set_epi64(INT64_C(                5964), INT64_C(                8009),
                            INT64_C(               22464), INT64_C(                3165),
                            INT64_C(               32445), INT64_C(               23614),
                            INT64_C(                5345), INT64_C(               32325)) },
    { easysimd_mm512_set_epi64(INT64_C( 7443398932235525007), INT64_C(-1954475805396281420),
                            INT64_C( 2896517201997827064), INT64_C(-7120983626837339415),
                            INT64_C( -201538146421797804), INT64_C(   96284688433294814),
                            INT64_C(  317424323145668713), INT64_C(-2012972091494378925)),
      0x24,
      easysimd_mm512_set_epi64(INT64_C(           108315710), INT64_C(           239994089),
                            INT64_C(            42149872), INT64_C(           164811505),
                            INT64_C(           265502689), INT64_C(             1401126),
                            INT64_C(             4619131), INT64_C(           239142856)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    unsigned int b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srli_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_srli_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_srli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const uint32_t imm8;
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C(  2679),  INT16_C(  7939),  INT16_C( 30199),  INT16_C( 21490),  INT16_C( 10443),  INT16_C( 15254),  INT16_C( 24980), -INT16_C( 16697),
        -INT16_C(  9559),  INT16_C( 14991), -INT16_C( 20636),  INT16_C( 21700), -INT16_C( 22760), -INT16_C(   919), -INT16_C( 21042),  INT16_C( 22443),
        -INT16_C( 20574), -INT16_C( 25994),  INT16_C( 26660), -INT16_C(  4115), -INT16_C( 31599),  INT16_C(  9514), -INT16_C(  3611), -INT16_C( 28957),
         INT16_C( 29643),  INT16_C( 12489), -INT16_C( 29406),  INT16_C( 14980), -INT16_C(  4812),  INT16_C(   566), -INT16_C(  7525),  INT16_C( 15706) },
      UINT32_C(3050819729),
      { -INT16_C( 15047), -INT16_C( 13659), -INT16_C( 12471),  INT16_C( 12015), -INT16_C( 11327), -INT16_C( 29507), -INT16_C( 31162),  INT16_C( 26812),
         INT16_C( 16403),  INT16_C( 18594), -INT16_C( 10194), -INT16_C( 14006), -INT16_C( 23366),  INT16_C( 19206), -INT16_C(  8587), -INT16_C( 20991),
        -INT16_C( 22877), -INT16_C(  5000),  INT16_C( 26485),  INT16_C( 13850), -INT16_C( 10438), -INT16_C( 32573),  INT16_C( 32605),  INT16_C( 29160),
        -INT16_C( 30016), -INT16_C(  4423),  INT16_C(   867),  INT16_C(  7607), -INT16_C( 16984),  INT16_C(  7529),  INT16_C( 27291),  INT16_C( 16075) },
      UINT32_C(         3),
      {  INT16_C(  6311),  INT16_C(  7939),  INT16_C( 30199),  INT16_C( 21490),  INT16_C(  6776),  INT16_C( 15254),  INT16_C( 24980),  INT16_C(  3351),
        -INT16_C(  9559),  INT16_C( 14991), -INT16_C( 20636),  INT16_C( 21700),  INT16_C(  5271), -INT16_C(   919),  INT16_C(  7118),  INT16_C(  5568),
         INT16_C(  5332),  INT16_C(  7567),  INT16_C(  3310), -INT16_C(  4115),  INT16_C(  6887),  INT16_C(  9514),  INT16_C(  4075),  INT16_C(  3645),
         INT16_C(  4440),  INT16_C( 12489),  INT16_C(   108),  INT16_C( 14980),  INT16_C(  6069),  INT16_C(   941), -INT16_C(  7525),  INT16_C(  2009) } },
    { {  INT16_C( 10819), -INT16_C( 21883), -INT16_C( 17339),  INT16_C(  7397),  INT16_C( 25983), -INT16_C(   390), -INT16_C(  5298), -INT16_C( 10050),
        -INT16_C( 21340), -INT16_C( 22725),  INT16_C( 22883),  INT16_C(  8527),  INT16_C( 27842),  INT16_C( 11452), -INT16_C(  1225),  INT16_C( 31292),
        -INT16_C( 16091),  INT16_C( 27173),  INT16_C(  2685), -INT16_C(   889),  INT16_C(   367), -INT16_C( 16901), -INT16_C( 17940), -INT16_C( 28522),
        -INT16_C( 11930), -INT16_C( 14025), -INT16_C( 30934), -INT16_C(  4886), -INT16_C( 22541),  INT16_C( 11032),  INT16_C( 21666), -INT16_C( 14427) },
      UINT32_C(2469579286),
      { -INT16_C( 17964),  INT16_C( 17552), -INT16_C( 29766), -INT16_C( 23039), -INT16_C( 26812), -INT16_C( 21962),  INT16_C( 28009), -INT16_C( 27788),
         INT16_C( 24308), -INT16_C(  6016), -INT16_C( 26619), -INT16_C( 22765), -INT16_C( 18195),  INT16_C(   879), -INT16_C( 24189),  INT16_C( 22422),
         INT16_C(  9818),  INT16_C(  5275), -INT16_C( 25167), -INT16_C(  2374), -INT16_C(  4044), -INT16_C( 25184),  INT16_C(  5213),  INT16_C( 21041),
        -INT16_C( 20109),  INT16_C( 30778),  INT16_C( 19785),  INT16_C( 13856), -INT16_C( 28923), -INT16_C( 30663), -INT16_C( 12240), -INT16_C( 29984) },
      UINT32_C(         8),
      {  INT16_C( 10819),  INT16_C(    68),  INT16_C(   139),  INT16_C(  7397),  INT16_C(   151), -INT16_C(   390), -INT16_C(  5298), -INT16_C( 10050),
        -INT16_C( 21340),  INT16_C(   232),  INT16_C( 22883),  INT16_C(   167),  INT16_C( 27842),  INT16_C( 11452),  INT16_C(   161),  INT16_C(    87),
        -INT16_C( 16091),  INT16_C(    20),  INT16_C(  2685), -INT16_C(   889),  INT16_C(   240),  INT16_C(   157), -INT16_C( 17940), -INT16_C( 28522),
         INT16_C(   177),  INT16_C(   120), -INT16_C( 30934), -INT16_C(  4886),  INT16_C(   143),  INT16_C( 11032),  INT16_C( 21666),  INT16_C(   138) } },
    { { -INT16_C( 24965),  INT16_C(  6312), -INT16_C( 25000),  INT16_C( 18509), -INT16_C(  5570),  INT16_C( 21413), -INT16_C(  2277), -INT16_C( 13114),
         INT16_C( 15921),  INT16_C( 32278),  INT16_C( 19550), -INT16_C(  4732),  INT16_C(  3206),  INT16_C( 22045), -INT16_C( 22548),  INT16_C( 26700),
        -INT16_C(  3003), -INT16_C( 25216), -INT16_C( 12910), -INT16_C( 11803), -INT16_C( 29768), -INT16_C( 11484), -INT16_C(  5502), -INT16_C( 19296),
        -INT16_C( 18904), -INT16_C( 30926), -INT16_C( 18942), -INT16_C( 30604), -INT16_C( 27965), -INT16_C( 20514),  INT16_C( 11065),  INT16_C( 32535) },
      UINT32_C(2988218399),
      {  INT16_C(   613),  INT16_C(  7555), -INT16_C( 22643),  INT16_C(  4081), -INT16_C( 28271), -INT16_C( 17981), -INT16_C(  2489),  INT16_C( 18752),
        -INT16_C( 19028),  INT16_C( 28626), -INT16_C( 20409), -INT16_C( 32737),  INT16_C( 14043), -INT16_C(  1025),  INT16_C(  7374),  INT16_C( 13485),
         INT16_C( 12318), -INT16_C( 21679),  INT16_C( 17111),  INT16_C( 26810),  INT16_C( 32467),  INT16_C(  6689),  INT16_C( 25204),  INT16_C(  8292),
         INT16_C( 13847),  INT16_C( 24208), -INT16_C( 20506), -INT16_C( 15650), -INT16_C(  8475), -INT16_C( 19267),  INT16_C( 27386),  INT16_C(  6376) },
      UINT32_C(         5),
      {  INT16_C(    19),  INT16_C(   236),  INT16_C(  1340),  INT16_C(   127),  INT16_C(  1164),  INT16_C( 21413), -INT16_C(  2277), -INT16_C( 13114),
         INT16_C( 15921),  INT16_C( 32278),  INT16_C( 19550),  INT16_C(  1024),  INT16_C(   438),  INT16_C( 22045), -INT16_C( 22548),  INT16_C(   421),
        -INT16_C(  3003), -INT16_C( 25216),  INT16_C(   534),  INT16_C(   837),  INT16_C(  1014), -INT16_C( 11484), -INT16_C(  5502), -INT16_C( 19296),
        -INT16_C( 18904),  INT16_C(   756), -INT16_C( 18942), -INT16_C( 30604),  INT16_C(  1783),  INT16_C(  1445),  INT16_C( 11065),  INT16_C(   199) } },
    { { -INT16_C( 15559),  INT16_C( 31857), -INT16_C(  9859), -INT16_C(  1201),  INT16_C( 27386),  INT16_C( 23663), -INT16_C( 28466),  INT16_C(  1139),
        -INT16_C( 12000), -INT16_C( 12310), -INT16_C( 21328), -INT16_C( 29004),  INT16_C( 26729), -INT16_C( 11384), -INT16_C( 24496), -INT16_C( 30099),
        -INT16_C(  8605), -INT16_C(  8186),  INT16_C( 21943), -INT16_C( 19748),  INT16_C( 19391), -INT16_C( 29426), -INT16_C( 32037), -INT16_C(  1135),
         INT16_C( 31827),  INT16_C(   970),  INT16_C( 32552), -INT16_C( 28015),  INT16_C(  6631),  INT16_C( 14437), -INT16_C( 11335),  INT16_C(  7362) },
      UINT32_C(1778239665),
      { -INT16_C(  9955), -INT16_C(  8933),  INT16_C( 10532),  INT16_C(   106), -INT16_C(   853), -INT16_C(     5), -INT16_C( 14728), -INT16_C( 24574),
        -INT16_C( 27579),  INT16_C( 11314), -INT16_C( 26451),  INT16_C( 26468),  INT16_C(  9835),  INT16_C(  7299), -INT16_C( 32530),  INT16_C(  3205),
        -INT16_C( 24487),  INT16_C( 32489),  INT16_C( 21450),  INT16_C( 30078),  INT16_C( 31055), -INT16_C( 14476),  INT16_C( 30527), -INT16_C( 31640),
        -INT16_C( 26101), -INT16_C( 18255),  INT16_C(  5426), -INT16_C( 25313), -INT16_C( 23748),  INT16_C( 10938),  INT16_C( 16163),  INT16_C( 32054) },
      UINT32_C(        17),
      {  INT16_C(     0),  INT16_C( 31857), -INT16_C(  9859), -INT16_C(  1201),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28466),  INT16_C(     0),
        -INT16_C( 12000), -INT16_C( 12310), -INT16_C( 21328),  INT16_C(     0),  INT16_C( 26729), -INT16_C( 11384),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C(  8186),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(   970),  INT16_C( 32552),  INT16_C(     0),  INT16_C(  6631),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7362) } },
    { { -INT16_C(  1249),  INT16_C( 29610),  INT16_C(  8057), -INT16_C(  3390), -INT16_C( 30060),  INT16_C(  2866), -INT16_C( 18702), -INT16_C( 29674),
        -INT16_C( 12697),  INT16_C( 32191),  INT16_C( 23790), -INT16_C( 28231), -INT16_C(  7402),  INT16_C( 22196),  INT16_C( 12570),  INT16_C( 14646),
        -INT16_C(  8148), -INT16_C( 23124),  INT16_C( 28671), -INT16_C( 27752), -INT16_C( 13575), -INT16_C(  5218), -INT16_C( 19328), -INT16_C(  6025),
         INT16_C( 13955),  INT16_C( 29029),  INT16_C(  7827), -INT16_C( 22270), -INT16_C( 18943),  INT16_C(  7167),  INT16_C( 13800),  INT16_C(  5205) },
      UINT32_C( 364511509),
      {  INT16_C( 21104),  INT16_C( 27048),  INT16_C( 18204), -INT16_C( 25516), -INT16_C( 13061),  INT16_C( 32388), -INT16_C(  5886), -INT16_C( 27153),
        -INT16_C(  3833),  INT16_C(  2367),  INT16_C( 16040), -INT16_C( 28636),  INT16_C( 31092), -INT16_C( 30300),  INT16_C( 24187), -INT16_C(  5218),
         INT16_C( 18352), -INT16_C( 13227), -INT16_C( 22130), -INT16_C( 30359), -INT16_C(  4747),  INT16_C( 30728), -INT16_C(  2089), -INT16_C(  8691),
         INT16_C( 19689), -INT16_C( 28185),  INT16_C(  3211), -INT16_C(   223), -INT16_C( 14971),  INT16_C(   136),  INT16_C( 10020), -INT16_C( 11028) },
      UINT32_C(         4),
      {  INT16_C(  1319),  INT16_C( 29610),  INT16_C(  1137), -INT16_C(  3390),  INT16_C(  3279),  INT16_C(  2866), -INT16_C( 18702), -INT16_C( 29674),
         INT16_C(  3856),  INT16_C( 32191),  INT16_C( 23790), -INT16_C( 28231), -INT16_C(  7402),  INT16_C( 22196),  INT16_C( 12570),  INT16_C( 14646),
        -INT16_C(  8148),  INT16_C(  3269),  INT16_C( 28671),  INT16_C(  2198),  INT16_C(  3799),  INT16_C(  1920), -INT16_C( 19328),  INT16_C(  3552),
         INT16_C(  1230),  INT16_C( 29029),  INT16_C(   200), -INT16_C( 22270),  INT16_C(  3160),  INT16_C(  7167),  INT16_C( 13800),  INT16_C(  5205) } },
    { { -INT16_C( 24255), -INT16_C(  5380), -INT16_C( 31478), -INT16_C(  2208), -INT16_C( 10099), -INT16_C( 31282), -INT16_C( 21019),  INT16_C( 12910),
        -INT16_C(   108), -INT16_C( 24387), -INT16_C( 17376), -INT16_C(  6874),  INT16_C(  9796),  INT16_C( 27401), -INT16_C(  8686),  INT16_C( 21465),
        -INT16_C( 10881), -INT16_C( 30402), -INT16_C( 24997), -INT16_C(  6016),  INT16_C( 20342),  INT16_C( 23405), -INT16_C(  9220), -INT16_C( 28531),
         INT16_C( 19162), -INT16_C(  1487),  INT16_C( 22278),  INT16_C( 19424), -INT16_C(  5763), -INT16_C( 28490), -INT16_C( 28473),  INT16_C( 18147) },
      UINT32_C(3234799973),
      {  INT16_C( 20671),  INT16_C( 13737),  INT16_C(  5791), -INT16_C( 25711),  INT16_C(  7922), -INT16_C( 13269),  INT16_C( 23657),  INT16_C( 28615),
        -INT16_C( 22605),  INT16_C( 12730),  INT16_C( 29072),  INT16_C( 22721), -INT16_C( 23551),  INT16_C( 26270),  INT16_C( 28358), -INT16_C( 31449),
        -INT16_C( 12098),  INT16_C( 23995),  INT16_C( 19686), -INT16_C(  9992),  INT16_C(  9066), -INT16_C( 11355),  INT16_C( 27776),  INT16_C( 13123),
        -INT16_C(   749), -INT16_C( 23708),  INT16_C(  9582),  INT16_C( 28667), -INT16_C( 25910), -INT16_C( 28458), -INT16_C(   760), -INT16_C( 14827) },
      UINT32_C(        13),
      {  INT16_C(     2), -INT16_C(  5380),  INT16_C(     0), -INT16_C(  2208), -INT16_C( 10099),  INT16_C(     6),  INT16_C(     2),  INT16_C( 12910),
         INT16_C(     5), -INT16_C( 24387), -INT16_C( 17376), -INT16_C(  6874),  INT16_C(  9796),  INT16_C(     3), -INT16_C(  8686),  INT16_C( 21465),
         INT16_C(     6),  INT16_C(     2),  INT16_C(     2),  INT16_C(     6),  INT16_C( 20342),  INT16_C( 23405),  INT16_C(     3),  INT16_C(     1),
         INT16_C( 19162), -INT16_C(  1487),  INT16_C( 22278),  INT16_C( 19424), -INT16_C(  5763), -INT16_C( 28490),  INT16_C(     7),  INT16_C(     6) } },
    { {  INT16_C(  9168),  INT16_C(  7347), -INT16_C( 29669),  INT16_C( 16007),  INT16_C( 23089), -INT16_C( 25154), -INT16_C(  3427), -INT16_C( 25680),
         INT16_C( 21334),  INT16_C( 31753),  INT16_C( 31055), -INT16_C(  5818), -INT16_C( 10673),  INT16_C( 19697), -INT16_C( 18453), -INT16_C( 17383),
        -INT16_C( 13094), -INT16_C(  2600),  INT16_C( 24408), -INT16_C( 30413), -INT16_C(  3398),  INT16_C( 22310), -INT16_C( 10524),  INT16_C( 15090),
        -INT16_C(   982),  INT16_C( 31158), -INT16_C(   907), -INT16_C( 15262),  INT16_C( 21458), -INT16_C( 16880),  INT16_C( 10506), -INT16_C(  7046) },
      UINT32_C(1322865397),
      {  INT16_C(  3250),  INT16_C( 27863), -INT16_C(   258), -INT16_C(  7485), -INT16_C( 18732), -INT16_C(   483), -INT16_C( 11342),  INT16_C( 10103),
        -INT16_C(  9776), -INT16_C( 23829), -INT16_C(  1236),  INT16_C( 13920), -INT16_C(  9692),  INT16_C(  6426), -INT16_C(  3283), -INT16_C(  8345),
         INT16_C( 16128), -INT16_C(   437),  INT16_C(  3645),  INT16_C(  4577), -INT16_C(   316),  INT16_C( 30224), -INT16_C( 30767), -INT16_C( 24163),
        -INT16_C( 30623), -INT16_C( 29372), -INT16_C( 23421), -INT16_C( 22588), -INT16_C(  8577), -INT16_C( 21311),  INT16_C( 10450), -INT16_C( 11637) },
      UINT32_C(        17),
      {  INT16_C(     0),  INT16_C(  7347),  INT16_C(     0),  INT16_C( 16007),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 21334),  INT16_C(     0),  INT16_C( 31055), -INT16_C(  5818),  INT16_C(     0),  INT16_C( 19697),  INT16_C(     0), -INT16_C( 17383),
         INT16_C(     0), -INT16_C(  2600),  INT16_C( 24408),  INT16_C(     0),  INT16_C(     0),  INT16_C( 22310),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(   982),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21458), -INT16_C( 16880),  INT16_C(     0), -INT16_C(  7046) } },
    { { -INT16_C( 12074), -INT16_C(  7004), -INT16_C( 18767), -INT16_C( 20567),  INT16_C(  8134),  INT16_C( 19841),  INT16_C(  8893),  INT16_C( 17838),
         INT16_C( 15462),  INT16_C(  3017),  INT16_C( 28672), -INT16_C(  8566),  INT16_C( 13873),  INT16_C( 23216), -INT16_C( 32063), -INT16_C( 26687),
         INT16_C( 26195),  INT16_C(  1147),  INT16_C(  9244), -INT16_C(  7500),  INT16_C( 13636),  INT16_C(   303), -INT16_C(  8617), -INT16_C( 16826),
         INT16_C(  3866),  INT16_C(  6857),  INT16_C( 21376), -INT16_C( 19976), -INT16_C( 22135),  INT16_C( 18955), -INT16_C( 13013),  INT16_C( 32481) },
      UINT32_C(1334008883),
      {  INT16_C( 14209), -INT16_C( 15055),  INT16_C( 24684), -INT16_C( 15418),  INT16_C(  3134),  INT16_C( 22657),  INT16_C( 18972), -INT16_C( 25486),
         INT16_C( 27549),  INT16_C(  9805),  INT16_C( 22804),  INT16_C( 16240),  INT16_C( 20774),  INT16_C( 22974),  INT16_C( 16814),  INT16_C( 12200),
        -INT16_C(  9864), -INT16_C(  6924), -INT16_C( 17863),  INT16_C( 30887),  INT16_C( 10694), -INT16_C(  7472),  INT16_C( 17267),  INT16_C(  4478),
        -INT16_C( 13138), -INT16_C( 15817), -INT16_C( 22491),  INT16_C( 19201), -INT16_C( 16391), -INT16_C( 22620),  INT16_C( 19456),  INT16_C( 30934) },
      UINT32_C(        10),
      {  INT16_C(    13),  INT16_C(    49), -INT16_C( 18767), -INT16_C( 20567),  INT16_C(     3),  INT16_C(    22),  INT16_C(  8893),  INT16_C( 17838),
         INT16_C( 15462),  INT16_C(  3017),  INT16_C(    22),  INT16_C(    15),  INT16_C(    20),  INT16_C( 23216),  INT16_C(    16), -INT16_C( 26687),
         INT16_C(    54),  INT16_C(    57),  INT16_C(  9244), -INT16_C(  7500),  INT16_C( 13636),  INT16_C(   303), -INT16_C(  8617),  INT16_C(     4),
         INT16_C(    51),  INT16_C(    48),  INT16_C(    42),  INT16_C(    18), -INT16_C( 22135),  INT16_C( 18955),  INT16_C(    19),  INT16_C( 32481) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_srli_epi16(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_srli_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m512i r = easysimd_mm512_mask_srli_epi16(src, k, a, imm8);

    easysimd_test_x86_write_i16x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_mask_srli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const uint32_t imm8;
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(  1704558126),  INT32_C(   143622008), -INT32_C(  1261434054), -INT32_C(  2071884744), -INT32_C(   651856080), -INT32_C(  1652152393),  INT32_C(   474314969),  INT32_C(  1082778642),
        -INT32_C(   609934750), -INT32_C(   589089631),  INT32_C(  2022748991),  INT32_C(  1878790463),  INT32_C(  1061757320),  INT32_C(   568184648),  INT32_C(  1514021448),  INT32_C(  1754973958) },
      UINT16_C(16361),
      {  INT32_C(   645106243), -INT32_C(   153505178),  INT32_C(   637999402), -INT32_C(   783839352), -INT32_C(  1398698033), -INT32_C(   271652431), -INT32_C(   575220669), -INT32_C(  2145607876),
        -INT32_C(  1868132567),  INT32_C(  1820753985),  INT32_C(   563252889), -INT32_C(   319563235),  INT32_C(   446207081), -INT32_C(  1073125763),  INT32_C(  2006826810), -INT32_C(  1997031074) },
      UINT32_C(        18),
      {  INT32_C(        2460),  INT32_C(   143622008), -INT32_C(  1261434054),  INT32_C(       13393), -INT32_C(   651856080),  INT32_C(       15347),  INT32_C(       14189),  INT32_C(        8199),
         INT32_C(        9257),  INT32_C(        6945),  INT32_C(        2148),  INT32_C(       15164),  INT32_C(        1702),  INT32_C(       12290),  INT32_C(  1514021448),  INT32_C(  1754973958) } },
    { {  INT32_C(   495523997),  INT32_C(   750187934),  INT32_C(  1649072264),  INT32_C(  1607153355),  INT32_C(   903669455), -INT32_C(  1351574289), -INT32_C(   233969863),  INT32_C(  2067437022),
         INT32_C(  1285145517),  INT32_C(  1098403769), -INT32_C(   224148953), -INT32_C(   934187271),  INT32_C(  1157442900),  INT32_C(    49507785),  INT32_C(   854851668),  INT32_C(  1135488662) },
      UINT16_C(18418),
      {  INT32_C(   144092047), -INT32_C(  1882538260),  INT32_C(    50185136), -INT32_C(  1993387381), -INT32_C(  1980302954), -INT32_C(   259437573),  INT32_C(   723394429), -INT32_C(   227405725),
        -INT32_C(  1476785733),  INT32_C(  2000078279),  INT32_C(   343487624),  INT32_C(   480094598), -INT32_C(  1650093150),  INT32_C(  1552756446), -INT32_C(  1316508594),  INT32_C(  2024012476) },
      UINT32_C(        39),
      {  INT32_C(   495523997),  INT32_C(           0),  INT32_C(  1649072264),  INT32_C(  1607153355),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   934187271),  INT32_C(  1157442900),  INT32_C(    49507785),  INT32_C(           0),  INT32_C(  1135488662) } },
    { {  INT32_C(  1674190750), -INT32_C(  1964293802),  INT32_C(  1678835643),  INT32_C(   805711004), -INT32_C(    15817775), -INT32_C(   582129103), -INT32_C(   325452046),  INT32_C(  1089409442),
        -INT32_C(  2019313359), -INT32_C(  1257140486),  INT32_C(   706290062),  INT32_C(   509222733), -INT32_C(   199398973), -INT32_C(   976131117),  INT32_C(   196176489), -INT32_C(  1387552644) },
      UINT16_C(61017),
      {  INT32_C(  1165841460),  INT32_C(   593890057), -INT32_C(  1874676939), -INT32_C(   285669935), -INT32_C(   900084487),  INT32_C(  1127596946),  INT32_C(   451195342), -INT32_C(  1844953762),
        -INT32_C(  1680374383), -INT32_C(   960610928), -INT32_C(  1051328272), -INT32_C(     5288186), -INT32_C(  1379268325), -INT32_C(  1695416372),  INT32_C(   246732208), -INT32_C(  1532969710) },
      UINT32_C(        38),
      {  INT32_C(           0), -INT32_C(  1964293802),  INT32_C(  1678835643),  INT32_C(           0),  INT32_C(           0), -INT32_C(   582129103),  INT32_C(           0),  INT32_C(  1089409442),
        -INT32_C(  2019313359),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   199398973),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(  1261224073), -INT32_C(    39544323),  INT32_C(  1040410095),  INT32_C(   492372756), -INT32_C(   857143347), -INT32_C(   847477512), -INT32_C(   169899464), -INT32_C(  1590131926),
        -INT32_C(  1084945470), -INT32_C(  1816266332),  INT32_C(  1926348894), -INT32_C(  1869600061),  INT32_C(   693926193),  INT32_C(   922147069), -INT32_C(  1943284126),  INT32_C(   472736601) },
      UINT16_C(33390),
      { -INT32_C(  1736764709),  INT32_C(  1985599909), -INT32_C(   593421236),  INT32_C(   156619437), -INT32_C(   220048388), -INT32_C(  1261943671), -INT32_C(    31972655),  INT32_C(   427918910),
         INT32_C(  1051917465),  INT32_C(   582290390), -INT32_C(   721529305),  INT32_C(   618550056),  INT32_C(   790085798), -INT32_C(   706420988),  INT32_C(  1070922753),  INT32_C(   458839426) },
      UINT32_C(         9),
      { -INT32_C(  1261224073),  INT32_C(     3878124),  INT32_C(     7229582),  INT32_C(      305897), -INT32_C(   857143347),  INT32_C(     5923874),  INT32_C(     8326161), -INT32_C(  1590131926),
        -INT32_C(  1084945470),  INT32_C(     1137285),  INT32_C(  1926348894), -INT32_C(  1869600061),  INT32_C(   693926193),  INT32_C(   922147069), -INT32_C(  1943284126),  INT32_C(      896170) } },
    { {  INT32_C(   371677707),  INT32_C(  1698515215), -INT32_C(  1685253561), -INT32_C(  1337871632), -INT32_C(  1464569399),  INT32_C(  1370131029), -INT32_C(  1277957794),  INT32_C(  1292169026),
         INT32_C(  1482894409), -INT32_C(  1111645835), -INT32_C(  1554494797), -INT32_C(   984376836),  INT32_C(  1601046282), -INT32_C(   273672047),  INT32_C(  1151501313), -INT32_C(  1131305101) },
      UINT16_C(62931),
      { -INT32_C(   778680300),  INT32_C(  1562069253),  INT32_C(  1073158380),  INT32_C(  1262879197),  INT32_C(   291756129),  INT32_C(  1788175815),  INT32_C(   974194857), -INT32_C(   667949884),
         INT32_C(   849986860), -INT32_C(    91241202), -INT32_C(  1170635043), -INT32_C(   385449848),  INT32_C(   536504664),  INT32_C(  2005504206),  INT32_C(  1571986072), -INT32_C(  1422532226) },
      UINT32_C(        29),
      {  INT32_C(           6),  INT32_C(           2), -INT32_C(  1685253561), -INT32_C(  1337871632),  INT32_C(           0),  INT32_C(  1370131029),  INT32_C(           1),  INT32_C(           6),
         INT32_C(           1), -INT32_C(  1111645835),  INT32_C(           5), -INT32_C(   984376836),  INT32_C(           0),  INT32_C(           3),  INT32_C(           2),  INT32_C(           5) } },
    { { -INT32_C(  1531585057), -INT32_C(   209604500),  INT32_C(  1769683945), -INT32_C(  1413389247), -INT32_C(   294002338),  INT32_C(    92795242), -INT32_C(  2071731037), -INT32_C(   131387879),
         INT32_C(  2023547147),  INT32_C(  2053840273), -INT32_C(  1696274855), -INT32_C(  1471830710), -INT32_C(   241713274),  INT32_C(  1408638640),  INT32_C(   467106050), -INT32_C(  1290533976) },
      UINT16_C(45284),
      { -INT32_C(  1764854485), -INT32_C(   746838033),  INT32_C(   125421249),  INT32_C(   113704815), -INT32_C(   433817872), -INT32_C(  1604311351),  INT32_C(  1436747841), -INT32_C(   419002437),
        -INT32_C(   327297796), -INT32_C(  1128269061),  INT32_C(   801323200),  INT32_C(   674597176), -INT32_C(   938518017), -INT32_C(  1050054785),  INT32_C(   840305783), -INT32_C(  1877402476) },
      UINT32_C(        21),
      { -INT32_C(  1531585057), -INT32_C(   209604500),  INT32_C(          59), -INT32_C(  1413389247), -INT32_C(   294002338),  INT32_C(        1283),  INT32_C(         685),  INT32_C(        1848),
         INT32_C(  2023547147),  INT32_C(  2053840273), -INT32_C(  1696274855), -INT32_C(  1471830710),  INT32_C(        1600),  INT32_C(        1547),  INT32_C(   467106050),  INT32_C(        1152) } },
    { { -INT32_C(  1846838121),  INT32_C(  1951508540), -INT32_C(   173244053),  INT32_C(   284480950),  INT32_C(  1401929188),  INT32_C(   852119590), -INT32_C(  2084111257), -INT32_C(  1384884458),
         INT32_C(   255746003),  INT32_C(  1938067207), -INT32_C(   949473264), -INT32_C(   371762171),  INT32_C(  1060922905),  INT32_C(   510789303),  INT32_C(   429996035),  INT32_C(  1673926031) },
      UINT16_C( 1141),
      { -INT32_C(   158106510),  INT32_C(  1462215919),  INT32_C(  1119104107),  INT32_C(  1386794262),  INT32_C(  2136563469),  INT32_C(   532110461),  INT32_C(   993281909),  INT32_C(   473934250),
         INT32_C(   353620518), -INT32_C(   512935306),  INT32_C(  2082677094), -INT32_C(    70267666), -INT32_C(  1451612117), -INT32_C(   104320380),  INT32_C(   590675065), -INT32_C(   868256858) },
      UINT32_C(         8),
      {  INT32_C(    16159612),  INT32_C(  1951508540),  INT32_C(     4371500),  INT32_C(   284480950),  INT32_C(     8345951),  INT32_C(     2078556),  INT32_C(     3880007), -INT32_C(  1384884458),
         INT32_C(   255746003),  INT32_C(  1938067207),  INT32_C(     8135457), -INT32_C(   371762171),  INT32_C(  1060922905),  INT32_C(   510789303),  INT32_C(   429996035),  INT32_C(  1673926031) } },
    { { -INT32_C(  1933778606),  INT32_C(  1878236494), -INT32_C(  1923190847),  INT32_C(  1723357502),  INT32_C(   116023764),  INT32_C(   645915433), -INT32_C(  1966300649),  INT32_C(   886085857),
        -INT32_C(   926905223), -INT32_C(   365448407),  INT32_C(  1635226915), -INT32_C(  1010290449), -INT32_C(  1144409455), -INT32_C(  1377744746), -INT32_C(   885477910), -INT32_C(  1090582459) },
      UINT16_C(49300),
      { -INT32_C(  1099711098),  INT32_C(   525571752), -INT32_C(  1068547336), -INT32_C(   831332091),  INT32_C(  2098595996), -INT32_C(   299237194), -INT32_C(   873041972), -INT32_C(  1265923539),
        -INT32_C(   277676217), -INT32_C(  1928346219),  INT32_C(   223174152), -INT32_C(   606289857),  INT32_C(  2119758536), -INT32_C(  1083407629),  INT32_C(   529162993),  INT32_C(   886249196) },
      UINT32_C(        16),
      { -INT32_C(  1933778606),  INT32_C(  1878236494),  INT32_C(       49231),  INT32_C(  1723357502),  INT32_C(       32022),  INT32_C(   645915433), -INT32_C(  1966300649),  INT32_C(       46219),
        -INT32_C(   926905223), -INT32_C(   365448407),  INT32_C(  1635226915), -INT32_C(  1010290449), -INT32_C(  1144409455), -INT32_C(  1377744746),  INT32_C(        8074),  INT32_C(       13523) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_srli_epi32(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_srli_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m512i r = easysimd_mm512_mask_srli_epi32(src, k, a, imm8);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_mask_srli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const uint32_t imm8;
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 4464114191548706675),  INT64_C( 2424685119115712758), -INT64_C( 4966578051013562866), -INT64_C( 3799686406603880505),
        -INT64_C(  167109247092699588),  INT64_C(  291771864113397671),  INT64_C( 1482167297323049283),  INT64_C( 1918988492345970814) },
      UINT8_C(235),
      { -INT64_C(  508080782868051204),  INT64_C(  843566114677906909),  INT64_C( 3227792426753624726),  INT64_C( 4489283171561087428),
         INT64_C(  855257520125132718),  INT64_C( 6074847640615177306), -INT64_C( 2953398102442006853), -INT64_C( 2409842308062638835) },
      UINT32_C(        66),
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 4966578051013562866),  INT64_C(                   0),
        -INT64_C(  167109247092699588),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C( 8137720455725499672),  INT64_C( 2839101203754660474),  INT64_C( 5664983255074029092), -INT64_C( 7740740543897932215),
        -INT64_C(  200947278638709470),  INT64_C(  932261511549183049), -INT64_C( 8945209637177841076), -INT64_C( 2649834320225477847) },
      UINT8_C(106),
      {  INT64_C( 4378319998942028130), -INT64_C( 6418421681607177326), -INT64_C( 5534532137792799796),  INT64_C( 8216052331371616182),
         INT64_C( 8141854869308568553),  INT64_C( 2043506478531289145),  INT64_C( 6162426909551770490),  INT64_C( 2571826117236290093) },
      UINT32_C(        31),
      {  INT64_C( 8137720455725499672),  INT64_C(          5601124089),  INT64_C( 5664983255074029092),  INT64_C(          3825897505),
        -INT64_C(  200947278638709470),  INT64_C(           951581857),  INT64_C(          2869603647), -INT64_C( 2649834320225477847) } },
    { {  INT64_C(  180767276281554075), -INT64_C( 3620772525881804985),  INT64_C(  654924203785294483), -INT64_C( 6948378009656929357),
        -INT64_C( 3013072462504503221), -INT64_C( 9079013928866119263), -INT64_C( 2049277260016685714), -INT64_C( 8829555306346523691) },
      UINT8_C(118),
      {  INT64_C( 7091723425546613343),  INT64_C(  811726136505703928),  INT64_C( 3212788755018271040),  INT64_C( 5062023882624888746),
         INT64_C( 1865187475674055838),  INT64_C( 5242525354717754391), -INT64_C( 4215193211984216837),  INT64_C( 8015507738481059343) },
      UINT32_C(        51),
      {  INT64_C(  180767276281554075),  INT64_C(                 360),  INT64_C(                1426), -INT64_C( 6948378009656929357),
         INT64_C(                 828),  INT64_C(                2328),  INT64_C(                6320), -INT64_C( 8829555306346523691) } },
    { {  INT64_C(  422577304371392331),  INT64_C( 5793131411943559737),  INT64_C( 2238342036181465440), -INT64_C( 8242325904086202690),
         INT64_C( 5991247768055959375), -INT64_C( 1628575143548865598), -INT64_C( 4596494061587017252), -INT64_C( 2967897961800194305) },
      UINT8_C(183),
      {  INT64_C( 5698219132597696290),  INT64_C( 7053131315756411618),  INT64_C( 6026980880267081643), -INT64_C( 3022282911792831122),
        -INT64_C( 1867800854771736408),  INT64_C( 6438403293687049319),  INT64_C( 5640092091561131204),  INT64_C( 8652175541067390855) },
      UINT32_C(        34),
      {  INT64_C(           331680007),  INT64_C(           410546276),  INT64_C(           350816459), -INT64_C( 8242325904086202690),
         INT64_C(           965021505),  INT64_C(           374764395), -INT64_C( 4596494061587017252),  INT64_C(           503622900) } },
    { {  INT64_C( 4458421044483996986), -INT64_C( 1298369093678934166),  INT64_C( 5866530270476037442), -INT64_C( 1006947349674416449),
        -INT64_C( 8690178041383837993), -INT64_C( 2864929732686884576), -INT64_C( 2005748744343319898), -INT64_C( 4722483914607211316) },
      UINT8_C(252),
      { -INT64_C( 7040920293449537824), -INT64_C( 2180291051653916066),  INT64_C(  917022113477859015), -INT64_C( 7738677620363805375),
        -INT64_C( 5655950125122538601), -INT64_C( 8549371673272797535), -INT64_C( 8364820259769196515),  INT64_C( 2789633133940235630) },
      UINT32_C(        66),
      {  INT64_C( 4458421044483996986), -INT64_C( 1298369093678934166),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C( 4764102023170427508),  INT64_C( 5902686331288720966), -INT64_C( 8024526460391995388), -INT64_C( 1112875662959458991),
         INT64_C( 2025926963283908943), -INT64_C( 5520616349112917728),  INT64_C( 8638057081962299036), -INT64_C( 7801611947202445488) },
      UINT8_C(117),
      { -INT64_C( 5993894679869166041),  INT64_C( 4882479532358689361), -INT64_C( 1094127381604920843),  INT64_C( 9223028265679363182),
         INT64_C(  246573946838731317), -INT64_C( 3861014434094135325),  INT64_C( 2512476106025260028),  INT64_C( 1192797329848734647) },
      UINT32_C(        25),
      {  INT64_C(        371123832280),  INT64_C( 5902686331288720966),  INT64_C(        517148276928), -INT64_C( 1112875662959458991),
         INT64_C(          7348476256),  INT64_C(        434688616979),  INT64_C(         74877622903), -INT64_C( 7801611947202445488) } },
    { {  INT64_C(  333733930058830013),  INT64_C(  556935847045810642), -INT64_C( 7367926897612969747),  INT64_C( 1740223833248681352),
         INT64_C( 5230218992909012052), -INT64_C(  181058708086297023), -INT64_C( 7167263212788435096), -INT64_C( 3168856828690826094) },
      UINT8_C(194),
      {  INT64_C( 7608410437443938012), -INT64_C( 4706693137668895866),  INT64_C( 2437414029146365309), -INT64_C( 6704322976140902338),
         INT64_C( 3765166170080488923), -INT64_C( 5773850901978826087),  INT64_C( 6849064783295066822), -INT64_C(   77255933439070844) },
      UINT32_C(        58),
      {  INT64_C(  333733930058830013),  INT64_C(                  47), -INT64_C( 7367926897612969747),  INT64_C( 1740223833248681352),
         INT64_C( 5230218992909012052), -INT64_C(  181058708086297023),  INT64_C(                  23),  INT64_C(                  63) } },
    { { -INT64_C( 3455219591924165148), -INT64_C( 4754762115468612333), -INT64_C( 6275133881303699806),  INT64_C(  436508462785705425),
         INT64_C( 2127534155019940511), -INT64_C( 2558787605905176398),  INT64_C( 7849260245963660363),  INT64_C( 2561091904213465697) },
      UINT8_C( 69),
      { -INT64_C( 1784995144734353149), -INT64_C( 3283689873757570874),  INT64_C( 6687240313342112575), -INT64_C( 2398444035508252258),
         INT64_C( 4117310901124990684), -INT64_C( 9088042762487774056), -INT64_C( 4288909922714956420),  INT64_C( 4235050916961933836) },
      UINT32_C(        32),
      {  INT64_C(          3879365727), -INT64_C( 4754762115468612333),  INT64_C(          1556994466),  INT64_C(  436508462785705425),
         INT64_C( 2127534155019940511), -INT64_C( 2558787605905176398),  INT64_C(          3296377638),  INT64_C( 2561091904213465697) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_srli_epi64(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_srli_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m512i r = easysimd_mm512_mask_srli_epi64(src, k, a, imm8);

    easysimd_test_x86_write_i64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_maskz_srli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const uint32_t imm8;
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(2698444613),
      { -INT16_C(  8218),  INT16_C(  4716),  INT16_C( 20822), -INT16_C( 13935),  INT16_C(  8887), -INT16_C( 32502),  INT16_C( 14202), -INT16_C( 22020),
         INT16_C(  8957), -INT16_C( 22643),  INT16_C( 15175),  INT16_C(  7172), -INT16_C( 12853),  INT16_C(  4174),  INT16_C(  9676), -INT16_C( 19536),
         INT16_C(  7172),  INT16_C( 23493),  INT16_C( 22125),  INT16_C(  9252),  INT16_C( 11896), -INT16_C(  3419), -INT16_C( 24219),  INT16_C( 25243),
         INT16_C( 10691),  INT16_C(  2826),  INT16_C(  3684),  INT16_C( 12071),  INT16_C( 30171), -INT16_C( 22465), -INT16_C(  4198), -INT16_C( 24741) },
      UINT32_C(         3),
      {  INT16_C(  7164),  INT16_C(     0),  INT16_C(  2602),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1775),  INT16_C(     0),
         INT16_C(  1119),  INT16_C(  5361),  INT16_C(  1896),  INT16_C(   896),  INT16_C(  6585),  INT16_C(   521),  INT16_C(  1209),  INT16_C(  5750),
         INT16_C(     0),  INT16_C(  2936),  INT16_C(  2765),  INT16_C(     0),  INT16_C(  1487),  INT16_C(     0),  INT16_C(  5164),  INT16_C(  3155),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5383),  INT16_C(     0),  INT16_C(  5099) } },
    { UINT32_C(1987639840),
      { -INT16_C( 25570),  INT16_C( 19694), -INT16_C(  8127), -INT16_C(  7246),  INT16_C(  5243), -INT16_C( 23386), -INT16_C( 20194),  INT16_C( 11273),
         INT16_C( 14552),  INT16_C( 19976), -INT16_C( 20360),  INT16_C( 26600), -INT16_C( 30965),  INT16_C( 11123), -INT16_C(  5247), -INT16_C( 24671),
        -INT16_C( 28792), -INT16_C( 13844), -INT16_C( 24977), -INT16_C(  5460),  INT16_C( 21426), -INT16_C( 11889), -INT16_C( 26620), -INT16_C(  8707),
         INT16_C(  1488),  INT16_C( 18475),  INT16_C(  5045), -INT16_C( 16208),  INT16_C(  9115),  INT16_C(  7403), -INT16_C( 29682), -INT16_C( 26948) },
      UINT32_C(         7),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   329),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(   156),  INT16_C(     0),  INT16_C(   207),  INT16_C(   270),  INT16_C(    86),  INT16_C(   471),  INT16_C(   319),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   469),  INT16_C(   167),  INT16_C(   419),  INT16_C(   304),  INT16_C(     0),
         INT16_C(     0),  INT16_C(   144),  INT16_C(    39),  INT16_C(     0),  INT16_C(    71),  INT16_C(    57),  INT16_C(   280),  INT16_C(     0) } },
    { UINT32_C(1183473832),
      {  INT16_C( 29964),  INT16_C( 24568), -INT16_C( 14076), -INT16_C( 25500),  INT16_C( 16839), -INT16_C( 13204), -INT16_C( 19092),  INT16_C( 32642),
         INT16_C( 16997), -INT16_C( 30694),  INT16_C( 14126), -INT16_C( 17770),  INT16_C( 11763), -INT16_C( 25642),  INT16_C( 24717), -INT16_C( 26143),
        -INT16_C(  9771), -INT16_C(  9735),  INT16_C( 23971),  INT16_C( 27253), -INT16_C(  7522),  INT16_C(  2614), -INT16_C( 18281), -INT16_C(   887),
        -INT16_C( 23301),  INT16_C( 10628),  INT16_C(  6875), -INT16_C( 12573), -INT16_C( 18105), -INT16_C( 11159),  INT16_C( 18970), -INT16_C(  4242) },
      UINT32_C(         1),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20018),  INT16_C(     0),  INT16_C( 26166),  INT16_C(     0),  INT16_C( 16321),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 19947),  INT16_C( 12358),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 27900),  INT16_C(     0),  INT16_C( 13626),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 32324),
         INT16_C(     0),  INT16_C(  5314),  INT16_C(  3437),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  9485),  INT16_C(     0) } },
    { UINT32_C(3301362023),
      {  INT16_C( 12350),  INT16_C(  8290),  INT16_C( 27751),  INT16_C(  8119), -INT16_C( 19467), -INT16_C( 26342),  INT16_C( 17207),  INT16_C( 21108),
         INT16_C( 16935), -INT16_C(  8039),  INT16_C( 28331), -INT16_C(  2566), -INT16_C(  5412),  INT16_C( 17177), -INT16_C(  8269), -INT16_C(  3833),
         INT16_C( 26896),  INT16_C( 30482), -INT16_C( 13867), -INT16_C( 13674), -INT16_C( 20099), -INT16_C( 19356), -INT16_C(  9996),  INT16_C(  6918),
        -INT16_C( 24549), -INT16_C( 14596), -INT16_C(  2546), -INT16_C(  5444), -INT16_C( 10784), -INT16_C( 27859),  INT16_C( 13492), -INT16_C( 15227) },
      UINT32_C(         4),
      {  INT16_C(   771),  INT16_C(   518),  INT16_C(  1734),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2449),  INT16_C(  1075),  INT16_C(     0),
         INT16_C(  1058),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3935),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3579),  INT16_C(  3856),
         INT16_C(     0),  INT16_C(  1905),  INT16_C(  3229),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3471),  INT16_C(   432),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  3936),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   843),  INT16_C(  3144) } },
    { UINT32_C(1618099095),
      {  INT16_C( 15570), -INT16_C( 31779), -INT16_C( 28000),  INT16_C( 31095), -INT16_C( 27752),  INT16_C( 14484),  INT16_C( 23183), -INT16_C( 31418),
         INT16_C( 12310), -INT16_C(  5274), -INT16_C(  1699), -INT16_C( 28256),  INT16_C( 25726),  INT16_C(  5422), -INT16_C( 24416),  INT16_C( 29302),
         INT16_C( 21469),  INT16_C( 32245),  INT16_C( 27877),  INT16_C( 32502), -INT16_C( 29953), -INT16_C( 29002), -INT16_C(   539), -INT16_C(  1260),
         INT16_C( 31277), -INT16_C( 29721), -INT16_C( 30861), -INT16_C(  3556),  INT16_C( 19435), -INT16_C( 29945),  INT16_C( 32235), -INT16_C( 14083) },
      UINT32_C(        11),
      {  INT16_C(     7),  INT16_C(    16),  INT16_C(    18),  INT16_C(     0),  INT16_C(    18),  INT16_C(     0),  INT16_C(     0),  INT16_C(    16),
         INT16_C(     6),  INT16_C(    29),  INT16_C(     0),  INT16_C(    18),  INT16_C(    12),  INT16_C(     2),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(    15),  INT16_C(     0),  INT16_C(     0),  INT16_C(    17),  INT16_C(    17),  INT16_C(    31),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    17),  INT16_C(    15),  INT16_C(     0) } },
    { UINT32_C(1605781234),
      {  INT16_C( 13372), -INT16_C( 14498), -INT16_C(  4629), -INT16_C(  5972), -INT16_C( 22783),  INT16_C( 31509), -INT16_C( 24434),  INT16_C(  5614),
        -INT16_C(  8003),  INT16_C(  2049), -INT16_C( 29464),  INT16_C( 26099), -INT16_C( 17270),  INT16_C( 31798), -INT16_C(  4862),  INT16_C( 16091),
         INT16_C( 14881),  INT16_C(  3077), -INT16_C( 20185),  INT16_C( 10484),  INT16_C(  2649), -INT16_C(  6237), -INT16_C( 28246),  INT16_C( 26621),
        -INT16_C(   398),  INT16_C( 23151),  INT16_C( 25482),  INT16_C(  5311), -INT16_C(  2529),  INT16_C(  8593),  INT16_C( 27875),  INT16_C(  1119) },
      UINT32_C(        18),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C( 382538085),
      { -INT16_C(  2811),  INT16_C(  3951),  INT16_C( 22424),  INT16_C( 10938),  INT16_C(  8532),  INT16_C( 21148), -INT16_C(  2415), -INT16_C(  2852),
        -INT16_C(  3659), -INT16_C( 21741),  INT16_C( 13442), -INT16_C(  4466), -INT16_C( 27757), -INT16_C(  1899),  INT16_C( 25252), -INT16_C( 22257),
         INT16_C( 32344), -INT16_C(  3911),  INT16_C( 29653),  INT16_C( 10522), -INT16_C( 18796),  INT16_C(  9595),  INT16_C( 22700),  INT16_C( 25113),
         INT16_C( 11337), -INT16_C( 13555), -INT16_C( 25504), -INT16_C(  2887),  INT16_C( 20015), -INT16_C( 11284), -INT16_C(  1103),  INT16_C(  2428) },
      UINT32_C(        10),
      {  INT16_C(    61),  INT16_C(     0),  INT16_C(    21),  INT16_C(     0),  INT16_C(     0),  INT16_C(    20),  INT16_C(    61),  INT16_C(     0),
         INT16_C(    60),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    36),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(    31),  INT16_C(     0),  INT16_C(    28),  INT16_C(    10),  INT16_C(     0),  INT16_C(     0),  INT16_C(    22),  INT16_C(    24),
         INT16_C(     0),  INT16_C(    50),  INT16_C(    39),  INT16_C(     0),  INT16_C(    19),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(2823813429),
      {  INT16_C( 30996), -INT16_C( 13763),  INT16_C( 25332),  INT16_C( 19575), -INT16_C(  9860), -INT16_C( 22379),  INT16_C( 24806), -INT16_C( 32247),
        -INT16_C(   742),  INT16_C( 26801), -INT16_C( 31511), -INT16_C(  6887),  INT16_C(  8705),  INT16_C( 13919), -INT16_C( 20964),  INT16_C( 12511),
         INT16_C(  7207),  INT16_C(  7418),  INT16_C( 29054), -INT16_C(  1432), -INT16_C(   438),  INT16_C( 12707), -INT16_C( 21410),  INT16_C( 30899),
         INT16_C( 26025), -INT16_C( 27935), -INT16_C(  1303), -INT16_C(  5513), -INT16_C( 10723),  INT16_C( 14625),  INT16_C(   133), -INT16_C( 21399) },
      UINT32_C(         4),
      {  INT16_C(  1937),  INT16_C(     0),  INT16_C(  1583),  INT16_C(     0),  INT16_C(  3479),  INT16_C(  2697),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  4049),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3665),  INT16_C(   544),  INT16_C(   869),  INT16_C(  2785),  INT16_C(   781),
         INT16_C(   450),  INT16_C(   463),  INT16_C(  1815),  INT16_C(  4006),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2757),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3751),  INT16_C(     0),  INT16_C(   914),  INT16_C(     0),  INT16_C(  2758) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_srli_epi16(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_srli_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    // uint32_t imm8 = easysimd_test_x86_random_mmask32();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m512i r = easysimd_mm512_maskz_srli_epi16(k, a, imm8);

    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_maskz_srli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const uint32_t imm8;
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(32211),
      { -INT32_C(  1552232725), -INT32_C(   971958742), -INT32_C(  1171430953),  INT32_C(  1318915340), -INT32_C(   162341126),  INT32_C(   491494645),  INT32_C(  1623466897), -INT32_C(   119695603),
        -INT32_C(  1868867482),  INT32_C(  1163308142),  INT32_C(   436175629), -INT32_C(  1956078448),  INT32_C(  1887550074), -INT32_C(   410137258), -INT32_C(    45657616),  INT32_C(  1341466089) },
      UINT32_C(        17),
      {  INT32_C(       20925),  INT32_C(       25352),  INT32_C(           0),  INT32_C(           0),  INT32_C(       31529),  INT32_C(           0),  INT32_C(       12386),  INT32_C(       31854),
         INT32_C(       18509),  INT32_C(           0),  INT32_C(        3327),  INT32_C(       17844),  INT32_C(       14400),  INT32_C(       29638),  INT32_C(       32419),  INT32_C(           0) } },
    { UINT16_C(57232),
      {  INT32_C(   808861163),  INT32_C(  1680914762), -INT32_C(   707998646), -INT32_C(  1219066297), -INT32_C(  1002167331),  INT32_C(   286037268), -INT32_C(   855166594), -INT32_C(  1716676690),
         INT32_C(   533324756), -INT32_C(   427558500),  INT32_C(   230379462), -INT32_C(  1295772971),  INT32_C(  1232537653),  INT32_C(   475759517),  INT32_C(  1659462324), -INT32_C(   822372615) },
      UINT32_C(         3),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   411599995),  INT32_C(           0),  INT32_C(           0),  INT32_C(   322286325),
         INT32_C(    66665594),  INT32_C(   483426099),  INT32_C(    28797432),  INT32_C(   374899290),  INT32_C(   154067206),  INT32_C(           0),  INT32_C(   207432790),  INT32_C(   434074335) } },
    { UINT16_C(60868),
      { -INT32_C(    59720170), -INT32_C(  1850163324),  INT32_C(  1196804756),  INT32_C(  1237212671),  INT32_C(   379863546),  INT32_C(  1459619573), -INT32_C(   850225409),  INT32_C(   649729552),
         INT32_C(  1478634196),  INT32_C(  2129255145), -INT32_C(  1530577244), -INT32_C(  1762819173), -INT32_C(  1179872828), -INT32_C(  1777292137),  INT32_C(  1399087683),  INT32_C(  1299783288) },
      UINT32_C(        19),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(        2282),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(        6570),  INT32_C(        1239),
         INT32_C(        2820),  INT32_C(           0),  INT32_C(        5272),  INT32_C(        4829),  INT32_C(           0),  INT32_C(        4802),  INT32_C(        2668),  INT32_C(        2479) } },
    { UINT16_C(42395),
      { -INT32_C(  1332775374), -INT32_C(  1099576038), -INT32_C(     5506967),  INT32_C(  1974156476),  INT32_C(  1787123923), -INT32_C(   288430181), -INT32_C(  1385698208),  INT32_C(  1750204981),
        -INT32_C(  1827085960),  INT32_C(   407998127),  INT32_C(  1125645702),  INT32_C(   230212153), -INT32_C(  1250476518), -INT32_C(  2052897499),  INT32_C(  1731332657), -INT32_C(  2049997555) },
      UINT32_C(        38),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C( 6375),
      {  INT32_C(   761951509),  INT32_C(  1061447676), -INT32_C(  1376319584), -INT32_C(   685492959), -INT32_C(   528848549),  INT32_C(    34833819),  INT32_C(   399612050),  INT32_C(   338737406),
         INT32_C(   708942382), -INT32_C(  1586919935), -INT32_C(  1370529651), -INT32_C(   242912362),  INT32_C(  2026963165),  INT32_C(   393929861),  INT32_C(  2066631548),  INT32_C(   831479299) },
      UINT32_C(         9),
      {  INT32_C(     1488186),  INT32_C(     2073139),  INT32_C(     5700483),  INT32_C(           0),  INT32_C(           0),  INT32_C(       68034),  INT32_C(      780492),  INT32_C(      661596),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(     7914169),  INT32_C(     3958912),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(23504),
      { -INT32_C(  1681631495), -INT32_C(  1830148637), -INT32_C(  1407754821), -INT32_C(  1300424902), -INT32_C(  1288936804),  INT32_C(  1508014046), -INT32_C(  1394065285),  INT32_C(   839432505),
        -INT32_C(   204616689), -INT32_C(  1400522767),  INT32_C(  1331207189),  INT32_C(  1342363316),  INT32_C(   369372728),  INT32_C(   561047206),  INT32_C(  1607292966),  INT32_C(   563270929) },
      UINT32_C(        13),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(      366947),  INT32_C(           0),  INT32_C(      354114),  INT32_C(      102469),
         INT32_C(      499310),  INT32_C(      353325),  INT32_C(           0),  INT32_C(      163862),  INT32_C(       45089),  INT32_C(           0),  INT32_C(      196202),  INT32_C(           0) } },
    { UINT16_C( 5215),
      {  INT32_C(  1083774867),  INT32_C(  2073572652),  INT32_C(   981298921), -INT32_C(  1119966042),  INT32_C(  1932338258), -INT32_C(  1438612150), -INT32_C(  1204021609),  INT32_C(  1271700408),
        -INT32_C(   561289806), -INT32_C(  2074467174),  INT32_C(   968808338), -INT32_C(   705233789),  INT32_C(  1783112480),  INT32_C(  1058310568),  INT32_C(  1475825823), -INT32_C(  1633500180) },
      UINT32_C(        38),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(32046),
      {  INT32_C(  1205293763),  INT32_C(   486911716),  INT32_C(   118686002), -INT32_C(  1957677535),  INT32_C(   513858015), -INT32_C(   820580232),  INT32_C(  2071058908), -INT32_C(   973562110),
        -INT32_C(   703803407), -INT32_C(  1343024771),  INT32_C(   867632914),  INT32_C(   482215741),  INT32_C(  1480285920),  INT32_C(   723997007),  INT32_C(   749115434),  INT32_C(   703700536) },
      UINT32_C(         9),
      {  INT32_C(           0),  INT32_C(      950999),  INT32_C(      231808),  INT32_C(     4565019),  INT32_C(           0),  INT32_C(     6785912),  INT32_C(           0),  INT32_C(           0),
         INT32_C(     7013991),  INT32_C(           0),  INT32_C(     1694595),  INT32_C(      941827),  INT32_C(     2891183),  INT32_C(     1414056),  INT32_C(     1463116),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_srli_epi32(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_srli_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m512i r = easysimd_mm512_maskz_srli_epi32(k, a, imm8);

    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_maskz_srli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const uint32_t imm8;
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(110),
      { -INT64_C( 4661784702378899535), -INT64_C( 4668905894164037945), -INT64_C( 3533692563858514126), -INT64_C( 5667279853620667024),
         INT64_C( 9155865013563208909), -INT64_C( 7441296342380400855), -INT64_C( 4650126801811963565),  INT64_C( 4718650309745487564) },
      UINT32_C(        47),
      {  INT64_C(                   0),  INT64_C(               97897),  INT64_C(              105963),  INT64_C(               90803),
         INT64_C(                   0),  INT64_C(               78198),  INT64_C(               98030),  INT64_C(                   0) } },
    { UINT8_C( 89),
      { -INT64_C( 1201989480307527790), -INT64_C( 3898579814786549848), -INT64_C( 7114274592101223393), -INT64_C( 8312139705191712305),
         INT64_C( 9053784281271118023),  INT64_C( 6796431288843749431),  INT64_C( 6668488054509242874),  INT64_C( 5438744729043348467) },
      UINT32_C(        13),
      {  INT64_C(    2105072582202395),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(    1237134322328837),
         INT64_C(    1105198276522353),  INT64_C(                   0),  INT64_C(     814024420716460),  INT64_C(                   0) } },
    { UINT8_C(169),
      { -INT64_C( 6237660543970620671),  INT64_C(  198836839076835048),  INT64_C( 6056550588676394839), -INT64_C( 2503412206626828002),
        -INT64_C( 5488492892412155038),  INT64_C(  338524983010280610),  INT64_C( 4539052416290975741),  INT64_C( 1746861976582594181) },
      UINT32_C(        18),
      {  INT64_C(      46573957556682),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(      60818984478312),
         INT64_C(                   0),  INT64_C(       1291370327035),  INT64_C(                   0),  INT64_C(       6663749605493) } },
    { UINT8_C(195),
      { -INT64_C( 1198985530952954045), -INT64_C(  816669894834779895), -INT64_C( 2386787924568389308), -INT64_C(  950123533886318898),
        -INT64_C( 1604012776558524469),  INT64_C( 4283573020454574959), -INT64_C( 7566507209832703464),  INT64_C( 4364959395732518387) },
      UINT32_C(         6),
      {  INT64_C(  269496227230571837),  INT64_C(  275469909044918308),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(  170003700998075752),  INT64_C(   68202490558320599) } },
    { UINT8_C(157),
      {  INT64_C( 7576026287157009162), -INT64_C( 3261222589511034128), -INT64_C( 4620092764389931537), -INT64_C( 7352920722945977404),
         INT64_C( 6644362423780374536), -INT64_C( 6069027396361720418), -INT64_C( 2532295021976554240),  INT64_C( 4825369756904394190) },
      UINT32_C(        50),
      {  INT64_C(                6728),  INT64_C(                   0),  INT64_C(               12280),  INT64_C(                9853),
         INT64_C(                5901),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                4285) } },
    { UINT8_C( 88),
      { -INT64_C( 6399198457510458353),  INT64_C( 3614459316853206638),  INT64_C( 5847778006095708479), -INT64_C( 5536724234195606823),
        -INT64_C( 1245762616920898009),  INT64_C( 4315377516138387735), -INT64_C( 1874486955331972979),  INT64_C( 3978612582850655710) },
      UINT32_C(        16),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(     196991269523833),
         INT64_C(     262466147717112),  INT64_C(                   0),  INT64_C(     252872575658837),  INT64_C(                   0) } },
    { UINT8_C( 46),
      {  INT64_C( 8635211746347869766),  INT64_C( 7103109054121366048),  INT64_C( 4210229605453176827), -INT64_C(  385621042529731264),
        -INT64_C( 8272886648619401883),  INT64_C( 4543608951434965919), -INT64_C( 1548775745592496701), -INT64_C(   85843709628736991) },
      UINT32_C(        24),
      {  INT64_C(                   0),  INT64_C(        423378291971),  INT64_C(        250949240055),  INT64_C(       1076526822518),
         INT64_C(                   0),  INT64_C(        270820197548),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(199),
      { -INT64_C( 8296698152684851272), -INT64_C( 7576053617263632160), -INT64_C( 3398428331216304454),  INT64_C( 2794440060778737456),
         INT64_C( 8129255228429982535), -INT64_C( 5908656790167388150), -INT64_C( 4106789633870119999), -INT64_C(  593473836670265553) },
      UINT32_C(        36),
      {  INT64_C(           147702607),  INT64_C(           158189365),  INT64_C(           218981815),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(           208673801),  INT64_C(           259799275) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_srli_epi64(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_srli_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m512i r = easysimd_mm512_maskz_srli_epi64(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_mask_srli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const easysimd__mmask16 k;
    const int16_t a[16];
    const uint32_t imm8;
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C(  5687), -INT16_C( 29029), -INT16_C(  6206),  INT16_C(  8903),  INT16_C( 24661),  INT16_C( 21269),  INT16_C( 28288), -INT16_C( 26302),
        -INT16_C(  1027),  INT16_C( 22625), -INT16_C(  8870), -INT16_C( 12502),  INT16_C(  5198),  INT16_C(  4481), -INT16_C( 23320),  INT16_C(  8095) },
      UINT16_C(15290),
      {  INT16_C( 32173),  INT16_C( 29730),  INT16_C( 30623), -INT16_C( 19244),  INT16_C( 21706),  INT16_C(  3106),  INT16_C(  8430),  INT16_C( 20232),
         INT16_C( 25208), -INT16_C( 23764),  INT16_C( 31537), -INT16_C( 19785), -INT16_C( 24692),  INT16_C( 11350),  INT16_C(  4542),  INT16_C( 27495) },
      UINT32_C(        16),
      {  INT16_C(  5687),  INT16_C(     0), -INT16_C(  6206),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28288),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C(  8870),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 23320),  INT16_C(  8095) } },
    { { -INT16_C(  8055),  INT16_C(    45), -INT16_C(  7756),  INT16_C(  2506), -INT16_C( 10749),  INT16_C(  9207),  INT16_C( 18142),  INT16_C( 16540),
         INT16_C( 16242), -INT16_C(  4751),  INT16_C(  9206), -INT16_C( 27270), -INT16_C( 22918), -INT16_C( 29869), -INT16_C( 16883), -INT16_C( 27111) },
      UINT16_C(18078),
      {  INT16_C( 21398),  INT16_C( 24615),  INT16_C( 10844),  INT16_C( 21302),  INT16_C(  5454), -INT16_C(  5479),  INT16_C(  2901), -INT16_C( 14551),
         INT16_C(  8185),  INT16_C( 29674),  INT16_C( 25780),  INT16_C(  1817),  INT16_C(  9967),  INT16_C(  2245),  INT16_C( 25788),  INT16_C( 21070) },
      UINT32_C(         5),
      { -INT16_C(  8055),  INT16_C(   769),  INT16_C(   338),  INT16_C(   665),  INT16_C(   170),  INT16_C(  9207),  INT16_C( 18142),  INT16_C(  1593),
         INT16_C( 16242),  INT16_C(   927),  INT16_C(   805), -INT16_C( 27270), -INT16_C( 22918), -INT16_C( 29869),  INT16_C(   805), -INT16_C( 27111) } },
    { { -INT16_C( 19851), -INT16_C( 24557),  INT16_C( 26344), -INT16_C(   530), -INT16_C(  9985),  INT16_C(  2643),  INT16_C(  6657),  INT16_C(  8195),
         INT16_C( 30212),  INT16_C( 27092), -INT16_C(  9329), -INT16_C( 19112),  INT16_C( 24992),  INT16_C(  1137), -INT16_C( 15441),  INT16_C(  9659) },
      UINT16_C(52853),
      {  INT16_C( 24261), -INT16_C( 19660),  INT16_C( 13147), -INT16_C( 20853), -INT16_C( 29634),  INT16_C( 16840), -INT16_C( 12884), -INT16_C( 32584),
         INT16_C( 18230), -INT16_C( 29093), -INT16_C(  1027),  INT16_C( 28399), -INT16_C( 24832), -INT16_C( 17614), -INT16_C( 22588), -INT16_C( 30326) },
      UINT32_C(         2),
      {  INT16_C(  6065), -INT16_C( 24557),  INT16_C(  3286), -INT16_C(   530),  INT16_C(  8975),  INT16_C(  4210),  INT16_C( 13163),  INT16_C(  8195),
         INT16_C( 30212),  INT16_C(  9110),  INT16_C( 16127),  INT16_C(  7099),  INT16_C( 24992),  INT16_C(  1137),  INT16_C( 10737),  INT16_C(  8802) } },
    { {  INT16_C( 15550), -INT16_C(  3487),  INT16_C(  4039),  INT16_C( 21296),  INT16_C( 29144), -INT16_C( 23041),  INT16_C( 32553),  INT16_C( 29147),
         INT16_C( 27098), -INT16_C( 10898), -INT16_C(  9127), -INT16_C(  1835), -INT16_C( 28402), -INT16_C( 18756),  INT16_C( 17691), -INT16_C(  9797) },
      UINT16_C( 7297),
      {  INT16_C( 18635), -INT16_C(  1236),  INT16_C(  1179), -INT16_C( 26003), -INT16_C( 26967), -INT16_C( 31719), -INT16_C(  3321),  INT16_C( 30189),
         INT16_C( 18120), -INT16_C( 25006),  INT16_C( 24638), -INT16_C(  1489),  INT16_C( 18966), -INT16_C( 11713), -INT16_C( 16349), -INT16_C(  4114) },
      UINT32_C(         3),
      {  INT16_C(  2329), -INT16_C(  3487),  INT16_C(  4039),  INT16_C( 21296),  INT16_C( 29144), -INT16_C( 23041),  INT16_C( 32553),  INT16_C(  3773),
         INT16_C( 27098), -INT16_C( 10898),  INT16_C(  3079),  INT16_C(  8005),  INT16_C(  2370), -INT16_C( 18756),  INT16_C( 17691), -INT16_C(  9797) } },
    { { -INT16_C(  5606),  INT16_C(  7843),  INT16_C( 15703), -INT16_C(  4409),  INT16_C( 19286),  INT16_C( 18933),  INT16_C( 27449),  INT16_C( 32530),
        -INT16_C( 20291),  INT16_C(  7614), -INT16_C( 18209),  INT16_C( 10548),  INT16_C(  1784), -INT16_C( 18356),  INT16_C( 15348),  INT16_C(  4033) },
      UINT16_C(25638),
      {  INT16_C( 32045), -INT16_C(  2654), -INT16_C(  1941),  INT16_C( 24896),  INT16_C( 31042),  INT16_C( 21708), -INT16_C( 30215), -INT16_C( 18684),
        -INT16_C(  7258), -INT16_C(  9617),  INT16_C( 26380),  INT16_C( 22752), -INT16_C( 10976), -INT16_C(  7788), -INT16_C( 17692),  INT16_C(  4421) },
      UINT32_C(         9),
      { -INT16_C(  5606),  INT16_C(   122),  INT16_C(   124), -INT16_C(  4409),  INT16_C( 19286),  INT16_C(    42),  INT16_C( 27449),  INT16_C( 32530),
        -INT16_C( 20291),  INT16_C(  7614),  INT16_C(    51),  INT16_C( 10548),  INT16_C(  1784),  INT16_C(   112),  INT16_C(    93),  INT16_C(  4033) } },
    { {  INT16_C(  1767), -INT16_C(  8029),  INT16_C(  1095), -INT16_C( 16350),  INT16_C( 30416),  INT16_C( 22969),  INT16_C( 28794),  INT16_C( 24063),
        -INT16_C(  9504),  INT16_C( 18281), -INT16_C( 15942), -INT16_C( 28825),  INT16_C( 18517),  INT16_C(  3955), -INT16_C( 31346),  INT16_C( 30023) },
      UINT16_C(60043),
      { -INT16_C( 11691),  INT16_C( 30702), -INT16_C( 16749),  INT16_C( 19693),  INT16_C( 26391),  INT16_C(  5821), -INT16_C( 25148),  INT16_C( 11760),
        -INT16_C( 21532),  INT16_C( 19695),  INT16_C( 17466), -INT16_C( 20844),  INT16_C(  8788), -INT16_C( 25805), -INT16_C( 16744), -INT16_C(  4731) },
      UINT32_C(        13),
      {  INT16_C(     6),  INT16_C(     3),  INT16_C(  1095),  INT16_C(     2),  INT16_C( 30416),  INT16_C( 22969),  INT16_C( 28794),  INT16_C(     1),
        -INT16_C(  9504),  INT16_C(     2), -INT16_C( 15942),  INT16_C(     5),  INT16_C( 18517),  INT16_C(     4),  INT16_C(     5),  INT16_C(     7) } },
    { {  INT16_C( 25971),  INT16_C( 12580),  INT16_C( 28754), -INT16_C( 17848),  INT16_C( 24109), -INT16_C( 13698), -INT16_C( 21425), -INT16_C(  1361),
        -INT16_C(  1125), -INT16_C(  8396), -INT16_C(  7537), -INT16_C( 19917), -INT16_C( 12779), -INT16_C( 11190),  INT16_C( 14163), -INT16_C( 14747) },
      UINT16_C(35228),
      { -INT16_C(  4105),  INT16_C( 16377),  INT16_C( 10153),  INT16_C( 10142), -INT16_C(  4623), -INT16_C( 24365),  INT16_C( 28391),  INT16_C(  7067),
         INT16_C( 11086), -INT16_C( 32258),  INT16_C(  5085),  INT16_C( 10064), -INT16_C( 23577),  INT16_C( 19550), -INT16_C(  1174),  INT16_C( 25045) },
      UINT32_C(         5),
      {  INT16_C( 25971),  INT16_C( 12580),  INT16_C(   317),  INT16_C(   316),  INT16_C(  1903), -INT16_C( 13698), -INT16_C( 21425),  INT16_C(   220),
         INT16_C(   346), -INT16_C(  8396), -INT16_C(  7537),  INT16_C(   314), -INT16_C( 12779), -INT16_C( 11190),  INT16_C( 14163),  INT16_C(   782) } },
    { { -INT16_C( 24113), -INT16_C(  2413), -INT16_C( 17857),  INT16_C( 11495), -INT16_C( 30578), -INT16_C(  1005),  INT16_C( 11811),  INT16_C( 20042),
        -INT16_C( 13268),  INT16_C( 16427),  INT16_C( 21020), -INT16_C( 16601),  INT16_C( 29873), -INT16_C( 21463), -INT16_C( 29879),  INT16_C(  6294) },
      UINT16_C(10540),
      {  INT16_C( 27406), -INT16_C(  2333),  INT16_C( 29079), -INT16_C( 21890), -INT16_C( 24210), -INT16_C( 18216),  INT16_C(  1520),  INT16_C(  7044),
        -INT16_C( 24507),  INT16_C( 27758),  INT16_C(  8032), -INT16_C( 30240),  INT16_C( 10955),  INT16_C( 24852),  INT16_C( 16450),  INT16_C( 20874) },
      UINT32_C(        17),
      { -INT16_C( 24113), -INT16_C(  2413),  INT16_C(     0),  INT16_C(     0), -INT16_C( 30578),  INT16_C(     0),  INT16_C( 11811),  INT16_C( 20042),
         INT16_C(     0),  INT16_C( 16427),  INT16_C( 21020),  INT16_C(     0),  INT16_C( 29873),  INT16_C(     0), -INT16_C( 29879),  INT16_C(  6294) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_srli_epi16(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_srli_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_mask_srli_epi16(src, k, a, imm8);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_mask_srli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const uint32_t imm8;
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   549304467),  INT32_C(  1716382917), -INT32_C(   900332091),  INT32_C(   705655434), -INT32_C(     7701536), -INT32_C(  2033576868),  INT32_C(  1774725928),  INT32_C(   571742645) },
      UINT8_C( 96),
      {  INT32_C(  1126498647),  INT32_C(  1409846350),  INT32_C(  1423889122),  INT32_C(  1563691233), -INT32_C(  1481034861),  INT32_C(   684670973),  INT32_C(   568145927), -INT32_C(  1535049651) },
      UINT32_C(         1),
      { -INT32_C(   549304467),  INT32_C(  1716382917), -INT32_C(   900332091),  INT32_C(   705655434), -INT32_C(     7701536),  INT32_C(   342335486),  INT32_C(   284072963),  INT32_C(   571742645) } },
    { {  INT32_C(   860940199), -INT32_C(  1038768912), -INT32_C(  1968936318), -INT32_C(   769850978), -INT32_C(   103824198), -INT32_C(   872351596), -INT32_C(   719773227),  INT32_C(  1255652771) },
      UINT8_C(165),
      { -INT32_C(   879395545), -INT32_C(    62040174), -INT32_C(    56895493), -INT32_C(  1179226635),  INT32_C(   877506364), -INT32_C(   787866960), -INT32_C(   243998925), -INT32_C(   577323338) },
      UINT32_C(        38),
      {  INT32_C(           0), -INT32_C(  1038768912),  INT32_C(           0), -INT32_C(   769850978), -INT32_C(   103824198),  INT32_C(           0), -INT32_C(   719773227),  INT32_C(           0) } },
    { { -INT32_C(  2100385749), -INT32_C(   847328523), -INT32_C(   759006619), -INT32_C(   535856335), -INT32_C(   477084727),  INT32_C(   756441677), -INT32_C(  1780283434), -INT32_C(   925843043) },
      UINT8_C(105),
      {  INT32_C(  1767852958),  INT32_C(  1137585353),  INT32_C(  1769251310), -INT32_C(   214805072),  INT32_C(  1195447781),  INT32_C(   857632300),  INT32_C(   315732817),  INT32_C(   578591107) },
      UINT32_C(        16),
      {  INT32_C(       26975), -INT32_C(   847328523), -INT32_C(   759006619),  INT32_C(       62258), -INT32_C(   477084727),  INT32_C(       13086),  INT32_C(        4817), -INT32_C(   925843043) } },
    { {  INT32_C(   128814043), -INT32_C(    84545191), -INT32_C(  1163239835), -INT32_C(  1499423087),  INT32_C(  1305667551), -INT32_C(  1197603323),  INT32_C(  1883025879),  INT32_C(   139812397) },
      UINT8_C(233),
      { -INT32_C(   213774590),  INT32_C(  1650015492), -INT32_C(  2047667225), -INT32_C(  1687905613),  INT32_C(  1923133804),  INT32_C(    21584208), -INT32_C(   215041387),  INT32_C(   299644431) },
      UINT32_C(        39),
      {  INT32_C(           0), -INT32_C(    84545191), -INT32_C(  1163239835),  INT32_C(           0),  INT32_C(  1305667551),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { {  INT32_C(  1531512094),  INT32_C(  1900260190),  INT32_C(   958777503), -INT32_C(   559562707), -INT32_C(  1188161440),  INT32_C(   458108769), -INT32_C(  1825947299),  INT32_C(  1020804125) },
      UINT8_C( 65),
      { -INT32_C(   845179103), -INT32_C(  1553198886), -INT32_C(   170875595),  INT32_C(  1649847882),  INT32_C(   214175709), -INT32_C(  1620385954),  INT32_C(  1186856202), -INT32_C(   142083626) },
      UINT32_C(        16),
      {  INT32_C(       52639),  INT32_C(  1900260190),  INT32_C(   958777503), -INT32_C(   559562707), -INT32_C(  1188161440),  INT32_C(   458108769),  INT32_C(       18109),  INT32_C(  1020804125) } },
    { {  INT32_C(   913032230), -INT32_C(   714338768), -INT32_C(  1944034850), -INT32_C(   949386569),  INT32_C(   639989318), -INT32_C(   584006432),  INT32_C(  2058581633),  INT32_C(   587967485) },
      UINT8_C(111),
      { -INT32_C(  2069866122),  INT32_C(   660764101),  INT32_C(   417263253), -INT32_C(   832658088), -INT32_C(  1884388150),  INT32_C(   705727412), -INT32_C(   366507201),  INT32_C(   207178390) },
      UINT32_C(        22),
      {  INT32_C(         530),  INT32_C(         157),  INT32_C(          99),  INT32_C(         825),  INT32_C(   639989318),  INT32_C(         168),  INT32_C(         936),  INT32_C(   587967485) } },
    { {  INT32_C(  1869189369), -INT32_C(   536571662),  INT32_C(   339221615), -INT32_C(     2161029),  INT32_C(  1085566644),  INT32_C(   159374974),  INT32_C(  1352624390), -INT32_C(  1124815934) },
      UINT8_C( 59),
      { -INT32_C(   299029666),  INT32_C(  1281166895),  INT32_C(  1304916550), -INT32_C(  1090402735),  INT32_C(  1497186682),  INT32_C(   694110144), -INT32_C(  1829982234),  INT32_C(    47032484) },
      UINT32_C(         7),
      {  INT32_C(    31218262),  INT32_C(    10009116),  INT32_C(   339221615),  INT32_C(    25035660),  INT32_C(    11696770),  INT32_C(     5422735),  INT32_C(  1352624390), -INT32_C(  1124815934) } },
    { {  INT32_C(   151187707), -INT32_C(  1068544434), -INT32_C(   602825707), -INT32_C(   547958626),  INT32_C(  1419816717), -INT32_C(  1103378162),  INT32_C(  1566756277), -INT32_C(  1791990630) },
      UINT8_C( 84),
      { -INT32_C(  2120049101), -INT32_C(  1969790226),  INT32_C(  1126724467),  INT32_C(  2035353801),  INT32_C(  1904715176), -INT32_C(  1389935136),  INT32_C(   189301927), -INT32_C(   396304971) },
      UINT32_C(        17),
      {  INT32_C(   151187707), -INT32_C(  1068544434),  INT32_C(        8596), -INT32_C(   547958626),  INT32_C(       14531), -INT32_C(  1103378162),  INT32_C(        1444), -INT32_C(  1791990630) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    uint32_t imm8 = test_vec[i].imm8;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_srli_epi32(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_srli_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m256i r = easysimd_mm256_mask_srli_epi32(src, k, a, imm8);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_mask_srli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const uint32_t imm8;
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 8347690191869864194),  INT64_C( 1354939642800839709), -INT64_C( 1573268761946669507), -INT64_C( 7475263349834434840) },
      UINT8_C(240),
      { -INT64_C(  955496672946684500), -INT64_C( 1254971923253697887),  INT64_C( 8713688207798245476), -INT64_C( 6821642863188724772) },
      UINT32_C(         9),
      {  INT64_C( 8347690191869864194),  INT64_C( 1354939642800839709), -INT64_C( 1573268761946669507), -INT64_C( 7475263349834434840) } },
    { { -INT64_C( 2140497352568716731),  INT64_C( 5738007275479605963),  INT64_C( 5182810377222792915), -INT64_C( 1628932366020030439) },
      UINT8_C( 32),
      { -INT64_C( 9216894958624009249), -INT64_C( 7141011409970113389),  INT64_C( 3800325480166520473), -INT64_C( 4561087723492311628) },
      UINT32_C(        17),
      { -INT64_C( 2140497352568716731),  INT64_C( 5738007275479605963),  INT64_C( 5182810377222792915), -INT64_C( 1628932366020030439) } },
    { {  INT64_C( 8604605529550163155),  INT64_C( 6344455193994493035), -INT64_C( 1556995954752835528),  INT64_C( 9106151637814933320) },
      UINT8_C(152),
      {  INT64_C( 4157159719556568210), -INT64_C( 2170467625801669473), -INT64_C( 1375785963354091473), -INT64_C( 2005767239649201659) },
      UINT32_C(        41),
      {  INT64_C( 8604605529550163155),  INT64_C( 6344455193994493035), -INT64_C( 1556995954752835528),  INT64_C(             7476490) } },
    { {  INT64_C(   51615319640648422), -INT64_C( 3522638168233779206), -INT64_C( 7520536776914256542),  INT64_C( 4883536271339238214) },
      UINT8_C(248),
      {  INT64_C( 8234503247546834156),  INT64_C( 1043568824478151686), -INT64_C( 1496800668412602556),  INT64_C( 5186893718355482164) },
      UINT32_C(        20),
      {  INT64_C(   51615319640648422), -INT64_C( 3522638168233779206), -INT64_C( 7520536776914256542),  INT64_C(       4946607321124) } },
    { {  INT64_C(  985002824565494735),  INT64_C( 6333307100325659679), -INT64_C( 6367814920384293088),  INT64_C( 7905588237522149954) },
      UINT8_C(221),
      { -INT64_C( 9133366306514005582), -INT64_C( 7228531648696819934),  INT64_C( 6729315430922197799), -INT64_C( 7314011921467100585) },
      UINT32_C(        71),
      {  INT64_C(                   0),  INT64_C( 6333307100325659679),  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C( 2802444054557181387),  INT64_C( 8410252786467187320),  INT64_C( 7894683957289015524), -INT64_C( 1915735523732344440) },
      UINT8_C(171),
      { -INT64_C( 6643232790959235425), -INT64_C( 1353207351607468517),  INT64_C( 3582215481138442401), -INT64_C( 5798166748790151444) },
      UINT32_C(         3),
      {  INT64_C( 1475438910343789523),  INT64_C( 2136692090262760387),  INT64_C( 7894683957289015524),  INT64_C( 1581072165614925021) } },
    { {  INT64_C( 3658966703441577261), -INT64_C( 4372655133938935233), -INT64_C( 2602530597049761954),  INT64_C( 8304807340925501141) },
      UINT8_C(159),
      {  INT64_C(   27205105098104799), -INT64_C( 2197169057448128274),  INT64_C( 1703190072311379929),  INT64_C( 6708060511090327005) },
      UINT32_C(        46),
      {  INT64_C(                 386),  INT64_C(              230920),  INT64_C(               24203),  INT64_C(               95327) } },
    { { -INT64_C( 1288356193590935926),  INT64_C( 4258209266136007209), -INT64_C( 4244173676023126711),  INT64_C( 3659632814980304904) },
      UINT8_C( 30),
      { -INT64_C( 4350500873265454433), -INT64_C( 2542630571653734853), -INT64_C( 7275520075084009131),  INT64_C( 5272154587489558892) },
      UINT32_C(        31),
      { -INT64_C( 1288356193590935926),  INT64_C(          7405929966),  INT64_C(          5202006548),  INT64_C(          2455038292) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_srli_epi64(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_srli_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m256i r = easysimd_mm256_mask_srli_epi64(src, k, a, imm8);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_maskz_srli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int16_t a[16];
    const uint32_t imm8;
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(35105),
      {  INT16_C(  3685),  INT16_C( 19453), -INT16_C( 13369),  INT16_C(   649), -INT16_C( 29221),  INT16_C( 16749), -INT16_C( 14116), -INT16_C(  9686),
         INT16_C( 13013),  INT16_C( 29930),  INT16_C( 18902),  INT16_C( 17489), -INT16_C( 32585),  INT16_C( 21396), -INT16_C( 19027),  INT16_C(  4828) },
      UINT32_C(         5),
      {  INT16_C(   115),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   523),  INT16_C(     0),  INT16_C(     0),
         INT16_C(   406),  INT16_C(     0),  INT16_C(     0),  INT16_C(   546),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   150) } },
    { UINT16_C(24282),
      { -INT16_C( 23158), -INT16_C( 29465),  INT16_C( 29824), -INT16_C( 15878), -INT16_C( 15791),  INT16_C( 11244),  INT16_C(  7831),  INT16_C(  2837),
         INT16_C( 24308),  INT16_C( 14428), -INT16_C(  9195),  INT16_C( 26829), -INT16_C( 32119), -INT16_C( 25531),  INT16_C(  8006), -INT16_C( 12038) },
      UINT32_C(        17),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(24033),
      {  INT16_C( 22085),  INT16_C(  1623),  INT16_C(  6567), -INT16_C( 11534),  INT16_C(  4272), -INT16_C( 17433),  INT16_C( 17669),  INT16_C( 15640),
        -INT16_C(  2982), -INT16_C( 15606), -INT16_C( 29314),  INT16_C(  6664),  INT16_C( 10195), -INT16_C( 23788), -INT16_C(  2581),  INT16_C( 12288) },
      UINT32_C(        10),
      {  INT16_C(    21),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    46),  INT16_C(    17),  INT16_C(    15),
         INT16_C(    61),  INT16_C(     0),  INT16_C(    35),  INT16_C(     6),  INT16_C(     9),  INT16_C(     0),  INT16_C(    61),  INT16_C(     0) } },
    { UINT16_C(14167),
      {  INT16_C( 28914), -INT16_C( 15319),  INT16_C( 14880), -INT16_C(  9045), -INT16_C(  4033),  INT16_C( 31988), -INT16_C(  6069),  INT16_C(  3719),
         INT16_C(  5222), -INT16_C( 32746),  INT16_C( 15847), -INT16_C( 30060), -INT16_C( 30168),  INT16_C( 22923), -INT16_C(  7467), -INT16_C( 14192) },
      UINT32_C(        14),
      {  INT16_C(     1),  INT16_C(     3),  INT16_C(     0),  INT16_C(     0),  INT16_C(     3),  INT16_C(     0),  INT16_C(     3),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     2),  INT16_C(     0),  INT16_C(     0),  INT16_C(     2),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(36025),
      { -INT16_C(  3213),  INT16_C( 20280),  INT16_C( 10290), -INT16_C( 20669),  INT16_C( 11379), -INT16_C( 32458),  INT16_C( 19090),  INT16_C(  5015),
        -INT16_C( 11215), -INT16_C( 17497),  INT16_C( 12797),  INT16_C( 22086),  INT16_C( 10503), -INT16_C( 12314), -INT16_C( 24708), -INT16_C(  4261) },
      UINT32_C(         9),
      {  INT16_C(   121),  INT16_C(     0),  INT16_C(     0),  INT16_C(    87),  INT16_C(    22),  INT16_C(    64),  INT16_C(     0),  INT16_C(     9),
         INT16_C(     0),  INT16_C(     0),  INT16_C(    24),  INT16_C(    43),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   119) } },
    { UINT16_C(16275),
      { -INT16_C( 17211),  INT16_C( 29826), -INT16_C( 20945), -INT16_C( 20054), -INT16_C(  3007),  INT16_C( 21576),  INT16_C(  7461), -INT16_C(  7685),
         INT16_C( 11546),  INT16_C( 28711),  INT16_C( 20532),  INT16_C(   854), -INT16_C(  2612), -INT16_C( 17314), -INT16_C(  3448),  INT16_C( 20219) },
      UINT32_C(        14),
      {  INT16_C(     2),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     3),  INT16_C(     0),  INT16_C(     0),  INT16_C(     3),
         INT16_C(     0),  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     3),  INT16_C(     2),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(49789),
      {  INT16_C( 11485), -INT16_C( 29075),  INT16_C( 24941), -INT16_C( 15913), -INT16_C(  2937),  INT16_C( 26812), -INT16_C(  5874),  INT16_C( 32399),
        -INT16_C(  8163),  INT16_C(  8404), -INT16_C( 13908),  INT16_C( 26751),  INT16_C( 29010), -INT16_C( 24477), -INT16_C(  7905), -INT16_C(   926) },
      UINT32_C(         3),
      {  INT16_C(  1435),  INT16_C(     0),  INT16_C(  3117),  INT16_C(  6202),  INT16_C(  7824),  INT16_C(  3351),  INT16_C(  7457),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  1050),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7203),  INT16_C(  8076) } },
    { UINT16_C(35791),
      {  INT16_C( 12666),  INT16_C( 15202),  INT16_C( 22200),  INT16_C(  8439), -INT16_C(  7836), -INT16_C(  7505), -INT16_C( 28674),  INT16_C(  8118),
         INT16_C( 32572), -INT16_C( 23394),  INT16_C(  4049),  INT16_C( 28936), -INT16_C(  5842),  INT16_C( 10964), -INT16_C( 23562),  INT16_C( 28853) },
      UINT32_C(        17),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_srli_epi16(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_srli_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    // uint32_t imm8 = easysimd_test_x86_random_mmask16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_maskz_srli_epi16(k, a, imm8);

    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_maskz_srli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const uint32_t imm8;
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 23),
      { -INT32_C(  1569878869),  INT32_C(  1552142764),  INT32_C(  1777042099),  INT32_C(  1072244641), -INT32_C(   733037876),  INT32_C(    12418092),  INT32_C(  1554232230), -INT32_C(   831293405) },
      UINT32_C(        32),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(225),
      { -INT32_C(   189615760), -INT32_C(   126458355), -INT32_C(  1205856305), -INT32_C(  1535906730),  INT32_C(  2099290048),  INT32_C(  1110492831), -INT32_C(  1766173918), -INT32_C(  1854423263) },
      UINT32_C(        25),
      {  INT32_C(         122),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(          33),  INT32_C(          75),  INT32_C(          72) } },
    { UINT8_C( 41),
      { -INT32_C(    74482299), -INT32_C(  1793958027), -INT32_C(  1971230697),  INT32_C(   740901132), -INT32_C(   286013250), -INT32_C(   985590517),  INT32_C(   579101355),  INT32_C(  2035086580) },
      UINT32_C(        39),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(219),
      { -INT32_C(  2026182795), -INT32_C(    17804676),  INT32_C(   104856796),  INT32_C(   483327529),  INT32_C(   775018988),  INT32_C(   596696479),  INT32_C(  1259898111),  INT32_C(  1931971070) },
      UINT32_C(         2),
      {  INT32_C(   567196125),  INT32_C(  1069290655),  INT32_C(           0),  INT32_C(   120831882),  INT32_C(   193754747),  INT32_C(           0),  INT32_C(   314974527),  INT32_C(   482992767) } },
    { UINT8_C( 97),
      { -INT32_C(   357306886),  INT32_C(   719818987),  INT32_C(  1680347286),  INT32_C(  1564349740), -INT32_C(   751116989),  INT32_C(   408435200), -INT32_C(  1398647163), -INT32_C(  1022487863) },
      UINT32_C(        36),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(193),
      { -INT32_C(  1789793618),  INT32_C(  1034282773), -INT32_C(  1974283956),  INT32_C(  1768396078), -INT32_C(  1046255767),  INT32_C(   571943040), -INT32_C(   483073835),  INT32_C(  1369715875) },
      UINT32_C(        13),
      {  INT32_C(      305807),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(      465319),  INT32_C(      167201) } },
    { UINT8_C(246),
      { -INT32_C(  1948437530), -INT32_C(  1017370255),  INT32_C(   442076083), -INT32_C(  1753038092), -INT32_C(  1698758013),  INT32_C(  1484428324),  INT32_C(   478944631),  INT32_C(  1427286895) },
      UINT32_C(         6),
      {  INT32_C(           0),  INT32_C(    51212453),  INT32_C(     6907438),  INT32_C(           0),  INT32_C(    40565770),  INT32_C(    23194192),  INT32_C(     7483509),  INT32_C(    22301357) } },
    { UINT8_C(239),
      {  INT32_C(  1025069025),  INT32_C(  1825098771),  INT32_C(  1764670695),  INT32_C(   325824851), -INT32_C(   978874549),  INT32_C(  1960976359),  INT32_C(  1293898043), -INT32_C(  2009268825) },
      UINT32_C(        35),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_srli_epi32(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_srli_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m256i r = easysimd_mm256_maskz_srli_epi32(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_srli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const uint32_t imm8;
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 85),
      { -INT64_C(  411505641788383035),  INT64_C( 6371738816611065458), -INT64_C( 1806028355142200389), -INT64_C( 2048802349536471297) },
      UINT32_C(        16),
      {  INT64_C(     275195898924578),  INT64_C(                   0),  INT64_C(     253917170998647),  INT64_C(                   0) } },
    { UINT8_C(179),
      {  INT64_C(  169498890343308657), -INT64_C( 2608794319335191797),  INT64_C( 6761465967463469648),  INT64_C( 5192122430202508233) },
      UINT32_C(         8),
      {  INT64_C(     662105040403549),  INT64_C(   61866991228024843),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(202),
      { -INT64_C( 3645377108911729916),  INT64_C( 3880361975129099644),  INT64_C( 4456746893850071568), -INT64_C(  162453972775554842) },
      UINT32_C(        58),
      {  INT64_C(                   0),  INT64_C(                  13),  INT64_C(                   0),  INT64_C(                  63) } },
    { UINT8_C(183),
      { -INT64_C( 3633780042653466533),  INT64_C( 3651203716165286527), -INT64_C(  856629954052531565),  INT64_C( 7858068944992408983) },
      UINT32_C(        42),
      {  INT64_C(             3368078),  INT64_C(              830187),  INT64_C(             3999528),  INT64_C(                   0) } },
    { UINT8_C(201),
      { -INT64_C( 5548856864197753294),  INT64_C( 6764111561098292953),  INT64_C( 4076206725156402784),  INT64_C( 4883495265672143854) },
      UINT32_C(        44),
      {  INT64_C(              733160),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(              277594) } },
    { UINT8_C(199),
      {  INT64_C( 1617954132294304776),  INT64_C( 8356826524108264612),  INT64_C( 5556310115210716390),  INT64_C( 8611698805055040758) },
      UINT32_C(        57),
      {  INT64_C(                  11),  INT64_C(                  57),  INT64_C(                  38),  INT64_C(                   0) } },
    { UINT8_C(202),
      { -INT64_C( 4609557677668130690), -INT64_C(  616975642475792110),  INT64_C( 5822845740408441991),  INT64_C( 3742246877265280419) },
      UINT32_C(        61),
      {  INT64_C(                   0),  INT64_C(                   7),  INT64_C(                   0),  INT64_C(                   1) } },
    { UINT8_C(218),
      {  INT64_C( 7036165474029762597),  INT64_C( 8792144168986186118), -INT64_C( 2043220617898861414), -INT64_C( 5479537636785497558) },
      UINT32_C(         2),
      {  INT64_C(                   0),  INT64_C( 2198036042246546529),  INT64_C(                   0),  INT64_C( 3241801609231013514) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_srli_epi64(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_srli_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m256i r = easysimd_mm256_maskz_srli_epi64(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int 
test_easysimd_mm_mask_srli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const easysimd__mmask8 k;
    const int16_t a[8];
    const uint32_t imm8;
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C( 19093),  INT16_C( 24669),  INT16_C( 28139),  INT16_C( 30487), -INT16_C( 17560),  INT16_C( 25221),  INT16_C( 16010),  INT16_C(  1390) },
      UINT8_C( 23),
      { -INT16_C( 20456),  INT16_C( 12636), -INT16_C( 31549),  INT16_C( 10331), -INT16_C( 10445), -INT16_C(  1326),  INT16_C( 26379),  INT16_C( 26693) },
      UINT32_C(        18),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30487),  INT16_C(     0),  INT16_C( 25221),  INT16_C( 16010),  INT16_C(  1390) } },
    { { -INT16_C( 10704), -INT16_C( 22305), -INT16_C( 25794), -INT16_C( 24531),  INT16_C( 27429),  INT16_C( 10766),  INT16_C(  3202),  INT16_C( 13122) },
      UINT8_C(104),
      { -INT16_C(  4874),  INT16_C(  7887), -INT16_C( 22753),  INT16_C(  6897),  INT16_C( 22706),  INT16_C(  6751), -INT16_C( 28896),  INT16_C(   240) },
      UINT32_C(        18),
      { -INT16_C( 10704), -INT16_C( 22305), -INT16_C( 25794),  INT16_C(     0),  INT16_C( 27429),  INT16_C(     0),  INT16_C(     0),  INT16_C( 13122) } },
    { { -INT16_C( 25810), -INT16_C( 12700), -INT16_C( 12352), -INT16_C(  5412), -INT16_C(  5806), -INT16_C( 31444), -INT16_C( 24495),  INT16_C( 15995) },
      UINT8_C(112),
      {  INT16_C(  5981),  INT16_C( 30602), -INT16_C(  7223), -INT16_C(  7210),  INT16_C( 26115),  INT16_C(   980),  INT16_C(   669),  INT16_C(   670) },
      UINT32_C(        18),
      { -INT16_C( 25810), -INT16_C( 12700), -INT16_C( 12352), -INT16_C(  5412),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 15995) } },
    { { -INT16_C( 11938),  INT16_C( 18605), -INT16_C( 27101), -INT16_C( 22411),  INT16_C(  5608),  INT16_C(  9763), -INT16_C( 17019), -INT16_C( 25469) },
      UINT8_C( 71),
      {  INT16_C( 10853),  INT16_C( 18897),  INT16_C( 14126),  INT16_C( 12573),  INT16_C(  8149), -INT16_C( 10288),  INT16_C( 12016), -INT16_C( 24920) },
      UINT32_C(         3),
      {  INT16_C(  1356),  INT16_C(  2362),  INT16_C(  1765), -INT16_C( 22411),  INT16_C(  5608),  INT16_C(  9763),  INT16_C(  1502), -INT16_C( 25469) } },
    { {  INT16_C( 13516),  INT16_C( 29932),  INT16_C(   284),  INT16_C( 17048),  INT16_C( 21895),  INT16_C(  9158), -INT16_C( 15972), -INT16_C( 14455) },
      UINT8_C(146),
      { -INT16_C( 13579),  INT16_C(  9967),  INT16_C(  3743),  INT16_C( 30454),  INT16_C(  9727), -INT16_C( 25314), -INT16_C(  5476), -INT16_C( 30511) },
      UINT32_C(         8),
      {  INT16_C( 13516),  INT16_C(    38),  INT16_C(   284),  INT16_C( 17048),  INT16_C(    37),  INT16_C(  9158), -INT16_C( 15972),  INT16_C(   136) } },
    { { -INT16_C( 30226),  INT16_C( 12535),  INT16_C( 19472),  INT16_C( 13558), -INT16_C( 18456), -INT16_C( 20547), -INT16_C( 28854),  INT16_C(  5284) },
      UINT8_C(126),
      { -INT16_C( 29517),  INT16_C( 10689), -INT16_C(  6517),  INT16_C( 10311),  INT16_C( 12930),  INT16_C(  2810), -INT16_C(  5999), -INT16_C( 30572) },
      UINT32_C(        13),
      { -INT16_C( 30226),  INT16_C(     1),  INT16_C(     7),  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     7),  INT16_C(  5284) } },
    { { -INT16_C( 11100), -INT16_C( 10225), -INT16_C( 14660),  INT16_C( 27797),  INT16_C(  9232),  INT16_C(  9232), -INT16_C(  9310),  INT16_C( 12247) },
      UINT8_C(157),
      { -INT16_C( 31814), -INT16_C(  7352),  INT16_C( 31238),  INT16_C(  4317), -INT16_C( 15093), -INT16_C( 27740),  INT16_C( 18909), -INT16_C(  5017) },
      UINT32_C(        19),
      {  INT16_C(     0), -INT16_C( 10225),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  9232), -INT16_C(  9310),  INT16_C(     0) } },
    { { -INT16_C( 19677), -INT16_C( 28745), -INT16_C(  9277), -INT16_C(  5984),  INT16_C( 31614), -INT16_C( 21057), -INT16_C( 16360), -INT16_C( 25497) },
      UINT8_C(  8),
      { -INT16_C( 32094), -INT16_C( 19929), -INT16_C(  4979),  INT16_C(  8279), -INT16_C( 24374), -INT16_C( 18809), -INT16_C( 21823),  INT16_C( 30825) },
      UINT32_C(         6),
      { -INT16_C( 19677), -INT16_C( 28745), -INT16_C(  9277),  INT16_C(   129),  INT16_C( 31614), -INT16_C( 21057), -INT16_C( 16360), -INT16_C( 25497) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_srli_epi16(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_srli_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    // uint32_t imm8 = easysimd_test_x86_random_mmask16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_mask_srli_epi16(src, k, a, imm8);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_mask_srli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const uint32_t imm8;
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(   366629933),  INT32_C(  2144622034),  INT32_C(   182883438),  INT32_C(   514601372) },
      UINT8_C( 88),
      { -INT32_C(  1253725346),  INT32_C(  1381306315),  INT32_C(   805115845), -INT32_C(   480495729) },
      UINT32_C(        38),
      {  INT32_C(   366629933),  INT32_C(  2144622034),  INT32_C(   182883438),  INT32_C(           0) } },
    { {  INT32_C(  1164359025),  INT32_C(   467326004), -INT32_C(  1974700322),  INT32_C(  1089053845) },
      UINT8_C(233),
      { -INT32_C(   185070434),  INT32_C(  1527496286),  INT32_C(  1234344685),  INT32_C(   867869566) },
      UINT32_C(        38),
      {  INT32_C(           0),  INT32_C(   467326004), -INT32_C(  1974700322),  INT32_C(           0) } },
    { { -INT32_C(   639735553), -INT32_C(   816792189),  INT32_C(   812901703),  INT32_C(   835673381) },
      UINT8_C( 85),
      { -INT32_C(   837578557),  INT32_C(  2120745195),  INT32_C(    52553289), -INT32_C(  2113721574) },
      UINT32_C(        39),
      {  INT32_C(           0), -INT32_C(   816792189),  INT32_C(           0),  INT32_C(   835673381) } },
    { {  INT32_C(   751240668),  INT32_C(  1192300244),  INT32_C(   245643071), -INT32_C(   120456600) },
      UINT8_C( 13),
      {  INT32_C(   118416543), -INT32_C(  2081532062),  INT32_C(  1588529243), -INT32_C(  1891976822) },
      UINT32_C(        17),
      {  INT32_C(         903),  INT32_C(  1192300244),  INT32_C(       12119),  INT32_C(       18333) } },
    { {  INT32_C(  2022597479),  INT32_C(  1336921514), -INT32_C(  1404495653), -INT32_C(   196389360) },
      UINT8_C(100),
      {  INT32_C(  1102861907), -INT32_C(  1991698470), -INT32_C(  1354574987), -INT32_C(   988364190) },
      UINT32_C(        25),
      {  INT32_C(  2022597479),  INT32_C(  1336921514),  INT32_C(          87), -INT32_C(   196389360) } },
    { {  INT32_C(  1042182031),  INT32_C(   123140542), -INT32_C(   245471322), -INT32_C(  1304116645) },
      UINT8_C(126),
      { -INT32_C(   829059963),  INT32_C(  1470171669), -INT32_C(   803667014),  INT32_C(   945777096) },
      UINT32_C(        15),
      {  INT32_C(  1042182031),  INT32_C(       44866),  INT32_C(      106546),  INT32_C(       28862) } },
    { { -INT32_C(   192809314),  INT32_C(  1549477886),  INT32_C(  1562294040), -INT32_C(   169697943) },
      UINT8_C( 50),
      {  INT32_C(  1379732145),  INT32_C(  2052519777),  INT32_C(   669195975),  INT32_C(  1304783702) },
      UINT32_C(        15),
      { -INT32_C(   192809314),  INT32_C(       62637),  INT32_C(  1562294040), -INT32_C(   169697943) } },
    { {  INT32_C(   353586105), -INT32_C(   976475225),  INT32_C(  1801598344),  INT32_C(   874288426) },
      UINT8_C(210),
      { -INT32_C(   993421970), -INT32_C(   219967473),  INT32_C(  2086484407), -INT32_C(   784971386) },
      UINT32_C(        33),
      {  INT32_C(   353586105),  INT32_C(           0),  INT32_C(  1801598344),  INT32_C(   874288426) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    // printf("\ni = %ld, imm8 = %6d\n", i, imm8);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_srli_epi32(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_srli_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m128i r = easysimd_mm_mask_srli_epi32(src, k, a, imm8);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_mask_srli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const int64_t a[2];
    const uint32_t imm8;
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 6896119070542235573),  INT64_C( 4541325703512422011) },
      UINT8_C(208),
      { -INT64_C( 6996596830030967094), -INT64_C( 5253936874486534763) },
      UINT32_C(        74),
      { -INT64_C( 6896119070542235573),  INT64_C( 4541325703512422011) } },
    { { -INT64_C( 3751682561263471315),  INT64_C( 2792411921202256367) },
      UINT8_C(  4),
      { -INT64_C( 1386350003615865235), -INT64_C( 5854401645908065923) },
      UINT32_C(        14),
      { -INT64_C( 3751682561263471315),  INT64_C( 2792411921202256367) } },
    { { -INT64_C( 2107367901161831367),  INT64_C( 2685142359368702229) },
      UINT8_C(167),
      {  INT64_C( 5069599722035554967),  INT64_C(  387901934830490473) },
      UINT32_C(        24),
      {  INT64_C(        302171690585),  INT64_C(         23120757033) } },
    { {  INT64_C( 9071517349041607818), -INT64_C(  923044643220249770) },
      UINT8_C(  8),
      { -INT64_C( 7928789110983030390), -INT64_C( 3287637907779897230) },
      UINT32_C(         0),
      {  INT64_C( 9071517349041607818), -INT64_C(  923044643220249770) } },
    { {  INT64_C( 5562570365468854698), -INT64_C(  609386804784858159) },
      UINT8_C(228),
      { -INT64_C( 8998017407500138896),  INT64_C( 8682055729860435315) },
      UINT32_C(        19),
      {  INT64_C( 5562570365468854698), -INT64_C(  609386804784858159) } },
    { {  INT64_C(  592722581874130443),  INT64_C( 7405027329166878919) },
      UINT8_C( 30),
      {  INT64_C( 4190651956202323242),  INT64_C( 1047599631494928129) },
      UINT32_C(         1),
      {  INT64_C(  592722581874130443),  INT64_C(  523799815747464064) } },
    { {  INT64_C( 3470397497192291015), -INT64_C( 1399140991379552089) },
      UINT8_C(252),
      { -INT64_C(  297468584397067041),  INT64_C( 6412162960085299115) },
      UINT32_C(         3),
      {  INT64_C( 3470397497192291015), -INT64_C( 1399140991379552089) } },
    { { -INT64_C( 3643660952597529603), -INT64_C( 5261319230417199736) },
      UINT8_C( 54),
      {  INT64_C( 7282900696735915012),  INT64_C( 2467440051009244996) },
      UINT32_C(        62),
      { -INT64_C( 3643660952597529603),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_srli_epi64(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_srli_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m128i r = easysimd_mm_mask_srli_epi64(src, k, a, imm8);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_maskz_srli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const uint32_t imm8;
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(164),
      { -INT16_C(  3701), -INT16_C(  4383), -INT16_C(  7326), -INT16_C( 24249),  INT16_C( 16371),  INT16_C(  8581),  INT16_C(  6643), -INT16_C( 20874) },
      UINT32_C(        11),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(    28),  INT16_C(     0),  INT16_C(     0),  INT16_C(     4),  INT16_C(     0),  INT16_C(    21) } },
    { UINT8_C( 47),
      { -INT16_C( 30123), -INT16_C( 23662),  INT16_C( 26603), -INT16_C( 15801),  INT16_C( 30511), -INT16_C(  8345),  INT16_C( 22531), -INT16_C(  3647) },
      UINT32_C(         3),
      {  INT16_C(  4426),  INT16_C(  5234),  INT16_C(  3325),  INT16_C(  6216),  INT16_C(     0),  INT16_C(  7148),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(164),
      { -INT16_C( 26533), -INT16_C(  8073),  INT16_C( 27321),  INT16_C( 12538),  INT16_C(  8984),  INT16_C( 22879), -INT16_C(  5512),  INT16_C(  7147) },
      UINT32_C(        16),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 83),
      { -INT16_C( 32105), -INT16_C(   294), -INT16_C(  8863),  INT16_C(  8791),  INT16_C(  4558),  INT16_C(  1735),  INT16_C( 24429),  INT16_C( 19837) },
      UINT32_C(         9),
      {  INT16_C(    65),  INT16_C(   127),  INT16_C(     0),  INT16_C(     0),  INT16_C(     8),  INT16_C(     0),  INT16_C(    47),  INT16_C(     0) } },
    { UINT8_C(231),
      { -INT16_C(   184), -INT16_C( 22421), -INT16_C(  7336),  INT16_C( 17554),  INT16_C( 26623),  INT16_C( 25239),  INT16_C(  6654), -INT16_C(   708) },
      UINT32_C(        19),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 26),
      { -INT16_C(  5987),  INT16_C( 25701), -INT16_C( 11537),  INT16_C( 27843), -INT16_C(  9440),  INT16_C( 26452),  INT16_C( 21284), -INT16_C( 13102) },
      UINT32_C(         9),
      {  INT16_C(     0),  INT16_C(    50),  INT16_C(     0),  INT16_C(    54),  INT16_C(   109),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(182),
      { -INT16_C( 18960), -INT16_C( 30779), -INT16_C( 15593),  INT16_C( 21408),  INT16_C(  6848),  INT16_C(  5229),  INT16_C( 22199),  INT16_C(  7034) },
      UINT32_C(        10),
      {  INT16_C(     0),  INT16_C(    33),  INT16_C(    48),  INT16_C(     0),  INT16_C(     6),  INT16_C(     5),  INT16_C(     0),  INT16_C(     6) } },
    { UINT8_C( 76),
      {  INT16_C( 27825),  INT16_C(  1466), -INT16_C(  8492), -INT16_C( 22951),  INT16_C(  1450),  INT16_C(  2140),  INT16_C(  4597),  INT16_C( 31949) },
      UINT32_C(        11),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(    27),  INT16_C(    20),  INT16_C(     0),  INT16_C(     0),  INT16_C(     2),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_srli_epi16(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_srli_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    // uint32_t imm8 = easysimd_test_x86_random_mmask16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_maskz_srli_epi16(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_maskz_srli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const uint32_t imm8;
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(144),
      {  INT32_C(   911309852),  INT32_C(  1072588265),  INT32_C(   746850783), -INT32_C(  1567082776) },
      UINT32_C(        30),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(108),
      {  INT32_C(   705926272), -INT32_C(  1909297255), -INT32_C(  1458897023), -INT32_C(   534436209) },
      UINT32_C(        20),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(        2704),  INT32_C(        3586) } },
    { UINT8_C( 15),
      {  INT32_C(   625888070),  INT32_C(  1011995476),  INT32_C(  1155459593),  INT32_C(  1775853142) },
      UINT32_C(        19),
      {  INT32_C(        1193),  INT32_C(        1930),  INT32_C(        2203),  INT32_C(        3387) } },
    { UINT8_C(114),
      {  INT32_C(  1510062809),  INT32_C(  1224936377), -INT32_C(  1893127886), -INT32_C(  2032505032) },
      UINT32_C(        35),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 46),
      {  INT32_C(  1651238489),  INT32_C(   665274832),  INT32_C(   814776488), -INT32_C(   202675726) },
      UINT32_C(         7),
      {  INT32_C(           0),  INT32_C(     5197459),  INT32_C(     6365441),  INT32_C(    31971027) } },
    { UINT8_C(164),
      {  INT32_C(   837666815),  INT32_C(   683677680),  INT32_C(   464427654),  INT32_C(   855771336) },
      UINT32_C(        31),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(210),
      {  INT32_C(   637079933), -INT32_C(  2074768751), -INT32_C(  1216921357), -INT32_C(   763463963) },
      UINT32_C(        39),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(110),
      {  INT32_C(  1888905193), -INT32_C(   896842751), -INT32_C(  1208119988),  INT32_C(  1506310752) },
      UINT32_C(        25),
      {  INT32_C(           0),  INT32_C(         101),  INT32_C(          91),  INT32_C(          44) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    // printf("\ni = %ld, imm8 = %6d\n", i, imm8);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_srli_epi32(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_srli_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m128i r = easysimd_mm_maskz_srli_epi32(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_maskz_srli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const uint32_t imm8;
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 90),
      {  INT64_C( 1913560581422380515),  INT64_C( 1214800345925749964) },
      UINT32_C(        22),
      {  INT64_C(                   0),  INT64_C(        289630972367) } },
    { UINT8_C(220),
      { -INT64_C( 4658793083873865132),  INT64_C( 6391729900274393118) },
      UINT32_C(        40),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(232),
      { -INT64_C( 1734465928478629458),  INT64_C( 5467172744438728902) },
      UINT32_C(        40),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 62),
      { -INT64_C( 2248396391154039313),  INT64_C( 7658624500278787859) },
      UINT32_C(        19),
      {  INT64_C(                   0),  INT64_C(      14607666969831) } },
    { UINT8_C(182),
      { -INT64_C( 4958862217402560934),  INT64_C( 5265823194765790689) },
      UINT32_C(         4),
      {  INT64_C(                   0),  INT64_C(  329113949672861918) } },
    { UINT8_C(120),
      { -INT64_C( 5234430371447648160), -INT64_C(  637685571906329737) },
      UINT32_C(        22),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 60),
      {  INT64_C( 2538904119976446860),  INT64_C( 2290869614960004853) },
      UINT32_C(        68),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 26),
      { -INT64_C( 7349889709485482832),  INT64_C( 6545450315855963502) },
      UINT32_C(        56),
      {  INT64_C(                   0),  INT64_C(                  90) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_srli_epi64(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_srli_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m128i r = easysimd_mm_maskz_srli_epi64(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srli_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_srli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_srli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_srli_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_srli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_srli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_srli_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_srli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_srli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_srli_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_srli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_srli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_srli_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_srli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_srli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_srli_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_srli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_srli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_srli_epi64)

EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
