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

#define EASYSIMD_TEST_X86_AVX512_INSN srl

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/srl.h>

static int
test_easysimd_mm512_srl_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int64_t b[2];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 21752), -INT16_C( 22194), -INT16_C( 18737),  INT16_C(  8433),  INT16_C( 23052),  INT16_C( 22961), -INT16_C(  7722), -INT16_C( 29326),
         INT16_C(  6474),  INT16_C( 15054),  INT16_C(  9100), -INT16_C( 16434), -INT16_C( 23329),  INT16_C( 14938), -INT16_C( 23326), -INT16_C(  5412),
         INT16_C( 10831),  INT16_C(  8083), -INT16_C( 31264), -INT16_C(  4801), -INT16_C(  3873), -INT16_C( 18874), -INT16_C( 18223),  INT16_C(  7235),
         INT16_C(  4562),  INT16_C( 24150),  INT16_C(  9268),  INT16_C(  4893),  INT16_C( 30920), -INT16_C( 21939),  INT16_C( 10524),  INT16_C( 27541) },
      {  INT64_C(                  17),  INT64_C(                  30) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C(   616), -INT16_C( 13232),  INT16_C( 28193), -INT16_C(  5664),  INT16_C( 11750),  INT16_C(   660),  INT16_C( 10583), -INT16_C( 21907),
        -INT16_C(  1967), -INT16_C(    34), -INT16_C(    63),  INT16_C( 31628), -INT16_C( 12441), -INT16_C( 30970), -INT16_C( 21163), -INT16_C(  4743),
        -INT16_C( 13910), -INT16_C( 13382), -INT16_C( 26057),  INT16_C(  7604),  INT16_C( 18631),  INT16_C(  7711), -INT16_C( 29327), -INT16_C( 15415),
        -INT16_C( 22651),  INT16_C( 18114),  INT16_C( 20135),  INT16_C(  3777), -INT16_C( 14563),  INT16_C( 29333),  INT16_C(  3700),  INT16_C(  7776) },
      {  INT64_C(                  15),  INT64_C(                  11) },
      {  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),
         INT16_C(     1),  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     1),  INT16_C(     1),  INT16_C(     1),  INT16_C(     1),
         INT16_C(     1),  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),  INT16_C(     1),
         INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C(  8950), -INT16_C( 20060),  INT16_C( 26155),  INT16_C( 18623),  INT16_C( 21549), -INT16_C( 23878),  INT16_C(  6754),  INT16_C( 15040),
        -INT16_C( 21964), -INT16_C(  6071),  INT16_C( 30024),  INT16_C( 11876), -INT16_C(   319), -INT16_C( 25978), -INT16_C( 24223),  INT16_C( 27640),
        -INT16_C( 25218), -INT16_C( 22243), -INT16_C(  9213),  INT16_C( 12529), -INT16_C( 21455), -INT16_C( 27694), -INT16_C( 27706), -INT16_C(  1075),
         INT16_C(  5693), -INT16_C( 31261),  INT16_C( 18316),  INT16_C( 19891),  INT16_C( 14917), -INT16_C( 22808), -INT16_C(  7973),  INT16_C( 23058) },
      {  INT64_C(                   7),  INT64_C(                  10) },
      {  INT16_C(   442),  INT16_C(   355),  INT16_C(   204),  INT16_C(   145),  INT16_C(   168),  INT16_C(   325),  INT16_C(    52),  INT16_C(   117),
         INT16_C(   340),  INT16_C(   464),  INT16_C(   234),  INT16_C(    92),  INT16_C(   509),  INT16_C(   309),  INT16_C(   322),  INT16_C(   215),
         INT16_C(   314),  INT16_C(   338),  INT16_C(   440),  INT16_C(    97),  INT16_C(   344),  INT16_C(   295),  INT16_C(   295),  INT16_C(   503),
         INT16_C(    44),  INT16_C(   267),  INT16_C(   143),  INT16_C(   155),  INT16_C(   116),  INT16_C(   333),  INT16_C(   449),  INT16_C(   180) } },
    { {  INT16_C( 18100),  INT16_C( 16600), -INT16_C( 29555), -INT16_C( 11379),  INT16_C( 30150), -INT16_C( 24199), -INT16_C( 29866), -INT16_C( 11269),
        -INT16_C(    70), -INT16_C( 14764),  INT16_C(  1524), -INT16_C( 27390), -INT16_C( 11640), -INT16_C( 24580),  INT16_C( 24432),  INT16_C(  9458),
        -INT16_C( 13403),  INT16_C( 12900), -INT16_C(  3753),  INT16_C(  7429),  INT16_C( 32615), -INT16_C( 16962), -INT16_C( 17910), -INT16_C( 14960),
        -INT16_C(  6983), -INT16_C( 21109), -INT16_C( 29207),  INT16_C( 29250),  INT16_C( 15968), -INT16_C( 12271),  INT16_C(   925),  INT16_C( 17140) },
      {  INT64_C(                  22),  INT64_C(                  14) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { {  INT16_C( 21730), -INT16_C( 13279),  INT16_C( 25569),  INT16_C( 16702),  INT16_C( 20385),  INT16_C( 16145),  INT16_C(  1362),  INT16_C(  8577),
        -INT16_C(  2467), -INT16_C( 22714), -INT16_C( 30351),  INT16_C( 27223), -INT16_C( 14966),  INT16_C( 17774),  INT16_C( 14275), -INT16_C( 23111),
        -INT16_C(  9589),  INT16_C( 28017), -INT16_C( 20675), -INT16_C(  8530), -INT16_C( 16130),  INT16_C( 20765), -INT16_C( 24635),  INT16_C(  9074),
        -INT16_C( 18283),  INT16_C(  1738),  INT16_C(  8513), -INT16_C( 13455), -INT16_C(  8218), -INT16_C( 22256), -INT16_C( 14057), -INT16_C( 23985) },
      {  INT64_C(                  13),  INT64_C(                   2) },
      {  INT16_C(     2),  INT16_C(     6),  INT16_C(     3),  INT16_C(     2),  INT16_C(     2),  INT16_C(     1),  INT16_C(     0),  INT16_C(     1),
         INT16_C(     7),  INT16_C(     5),  INT16_C(     4),  INT16_C(     3),  INT16_C(     6),  INT16_C(     2),  INT16_C(     1),  INT16_C(     5),
         INT16_C(     6),  INT16_C(     3),  INT16_C(     5),  INT16_C(     6),  INT16_C(     6),  INT16_C(     2),  INT16_C(     4),  INT16_C(     1),
         INT16_C(     5),  INT16_C(     0),  INT16_C(     1),  INT16_C(     6),  INT16_C(     6),  INT16_C(     5),  INT16_C(     6),  INT16_C(     5) } },
    { {  INT16_C( 12522),  INT16_C( 11031), -INT16_C( 30638),  INT16_C( 14583),  INT16_C(  1896),  INT16_C( 32738),  INT16_C( 12753),  INT16_C( 29729),
         INT16_C( 12785),  INT16_C( 24917),  INT16_C(  5359),  INT16_C( 28112), -INT16_C( 28688),  INT16_C( 27824),  INT16_C(  6081), -INT16_C( 21635),
        -INT16_C( 27577), -INT16_C( 26154), -INT16_C( 13027), -INT16_C( 31278), -INT16_C( 19243), -INT16_C( 23036),  INT16_C(  9701), -INT16_C( 10726),
         INT16_C( 28502),  INT16_C( 17720),  INT16_C(  2179),  INT16_C( 29874),  INT16_C( 25495),  INT16_C( 22752),  INT16_C( 23930), -INT16_C( 16125) },
      {  INT64_C(                  15),  INT64_C(                   2) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),
         INT16_C(     1),  INT16_C(     1),  INT16_C(     1),  INT16_C(     1),  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     1),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1) } },
    { { -INT16_C( 11092),  INT16_C( 12377),  INT16_C(  3036),  INT16_C( 29860), -INT16_C( 31634), -INT16_C(  5940), -INT16_C( 12063), -INT16_C( 11606),
         INT16_C(  1450),  INT16_C( 20961),  INT16_C( 29746),  INT16_C(  5070), -INT16_C(  4084), -INT16_C( 13863),  INT16_C( 29997), -INT16_C(  9508),
         INT16_C( 13642),  INT16_C(  9738), -INT16_C( 20927), -INT16_C( 20582),  INT16_C( 26418),  INT16_C(  5016),  INT16_C( 16951), -INT16_C(  7707),
        -INT16_C( 14777),  INT16_C( 31026),  INT16_C(    59),  INT16_C( 18316),  INT16_C( 26097),  INT16_C(  7696), -INT16_C(  4902),  INT16_C(  9464) },
      {  INT64_C(                  28),  INT64_C(                   6) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C( 26719), -INT16_C(  9044),  INT16_C( 14487), -INT16_C( 30685),  INT16_C( 13213),  INT16_C( 30887), -INT16_C( 24800),  INT16_C( 17052),
        -INT16_C(  6238),  INT16_C( 21157), -INT16_C( 18483),  INT16_C(  6453),  INT16_C( 10850),  INT16_C( 20125),  INT16_C(   261), -INT16_C( 22654),
         INT16_C( 11928),  INT16_C( 12419), -INT16_C( 22681),  INT16_C(  1208),  INT16_C( 24538), -INT16_C(  1412),  INT16_C(  6655), -INT16_C( 24260),
        -INT16_C(  7936), -INT16_C( 12813),  INT16_C( 10393), -INT16_C(  1049), -INT16_C( 31661),  INT16_C( 22601), -INT16_C( 13435),  INT16_C(  7935) },
      {  INT64_C(                   0),  INT64_C(                  31) },
      { -INT16_C( 26719), -INT16_C(  9044),  INT16_C( 14487), -INT16_C( 30685),  INT16_C( 13213),  INT16_C( 30887), -INT16_C( 24800),  INT16_C( 17052),
        -INT16_C(  6238),  INT16_C( 21157), -INT16_C( 18483),  INT16_C(  6453),  INT16_C( 10850),  INT16_C( 20125),  INT16_C(   261), -INT16_C( 22654),
         INT16_C( 11928),  INT16_C( 12419), -INT16_C( 22681),  INT16_C(  1208),  INT16_C( 24538), -INT16_C(  1412),  INT16_C(  6655), -INT16_C( 24260),
        -INT16_C(  7936), -INT16_C( 12813),  INT16_C( 10393), -INT16_C(  1049), -INT16_C( 31661),  INT16_C( 22601), -INT16_C( 13435),  INT16_C(  7935) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srl_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_srl_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_srl_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int64_t count[2];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 17272), -INT16_C( 24566),  INT16_C(  6847), -INT16_C( 17132), -INT16_C( 26375),  INT16_C( 24134), -INT16_C( 10353), -INT16_C( 26844),
        -INT16_C( 29923),  INT16_C( 14279), -INT16_C( 19096),  INT16_C( 20533), -INT16_C( 18657), -INT16_C( 13925), -INT16_C( 15872), -INT16_C( 30220),
        -INT16_C(   386),  INT16_C( 15913),  INT16_C( 15640),  INT16_C(  4603),  INT16_C( 17109),  INT16_C( 25968), -INT16_C( 27623),  INT16_C( 14076),
        -INT16_C( 15585), -INT16_C( 30867), -INT16_C( 23687), -INT16_C( 26408),  INT16_C( 29530),  INT16_C( 23393),  INT16_C( 21814), -INT16_C( 19228) },
      UINT32_C(1811025235),
      { -INT16_C(  4534),  INT16_C(  8061), -INT16_C(  4816),  INT16_C( 18820), -INT16_C( 32383), -INT16_C( 24192), -INT16_C(  4796), -INT16_C( 17112),
         INT16_C(   144), -INT16_C(  5291), -INT16_C( 18572), -INT16_C( 21946),  INT16_C( 10764),  INT16_C( 24670),  INT16_C( 20791), -INT16_C( 32309),
         INT16_C( 18495),  INT16_C( 28576),  INT16_C(  9525), -INT16_C( 18504),  INT16_C( 14502), -INT16_C(  5544), -INT16_C( 32730), -INT16_C( 18776),
        -INT16_C(   639), -INT16_C(  2655), -INT16_C(  6220), -INT16_C( 15969), -INT16_C(   751),  INT16_C( 18465), -INT16_C(  5042), -INT16_C( 29239) },
      {  INT64_C(                  17),  INT64_C(                   2) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(  6847), -INT16_C( 17132),  INT16_C(     0),  INT16_C( 24134),  INT16_C(     0), -INT16_C( 26844),
         INT16_C(     0),  INT16_C( 14279),  INT16_C(     0),  INT16_C(     0), -INT16_C( 18657), -INT16_C( 13925), -INT16_C( 15872), -INT16_C( 30220),
        -INT16_C(   386),  INT16_C(     0),  INT16_C( 15640),  INT16_C(  4603),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 23687),  INT16_C(     0),  INT16_C( 29530),  INT16_C(     0),  INT16_C(     0), -INT16_C( 19228) } },
    { {  INT16_C( 27388), -INT16_C( 19057),  INT16_C( 13601),  INT16_C( 31213),  INT16_C(  4895), -INT16_C( 14342),  INT16_C( 31690),  INT16_C( 27589),
         INT16_C( 31088),  INT16_C(  3923),  INT16_C( 25658),  INT16_C( 23308),  INT16_C( 23469),  INT16_C( 30280),  INT16_C( 32232), -INT16_C(  6688),
         INT16_C( 28647),  INT16_C(  2458), -INT16_C( 30812), -INT16_C( 15230),  INT16_C( 31899),  INT16_C( 25995),  INT16_C( 20727),  INT16_C( 26576),
         INT16_C(  9162),  INT16_C(  1142), -INT16_C( 31864),  INT16_C( 13664), -INT16_C( 22306), -INT16_C( 14677), -INT16_C( 29659),  INT16_C(  3243) },
      UINT32_C(2685748731),
      { -INT16_C( 26419),  INT16_C( 26724), -INT16_C(  4332),  INT16_C(  3277), -INT16_C( 25280),  INT16_C(  2675), -INT16_C(  5439),  INT16_C( 18702),
         INT16_C( 28269),  INT16_C( 19326),  INT16_C( 10518),  INT16_C( 15121), -INT16_C( 16971), -INT16_C( 20152),  INT16_C( 23810), -INT16_C( 12463),
        -INT16_C( 18955),  INT16_C(  2615),  INT16_C(  1188), -INT16_C(  7146), -INT16_C( 30302),  INT16_C( 25582), -INT16_C(   653), -INT16_C(  8020),
         INT16_C( 10859), -INT16_C( 32213),  INT16_C( 15699),  INT16_C(  2493),  INT16_C(  1530), -INT16_C(   838),  INT16_C(  2915),  INT16_C( 22732) },
      {  INT64_C(                   7),  INT64_C(                  13) },
      {  INT16_C(   305),  INT16_C(   208),  INT16_C( 13601),  INT16_C(    25),  INT16_C(   314),  INT16_C(    20),  INT16_C(   469),  INT16_C(   146),
         INT16_C(   220),  INT16_C(  3923),  INT16_C(    82),  INT16_C( 23308),  INT16_C( 23469),  INT16_C( 30280),  INT16_C(   186), -INT16_C(  6688),
         INT16_C(   363),  INT16_C(  2458),  INT16_C(     9), -INT16_C( 15230),  INT16_C(   275),  INT16_C( 25995),  INT16_C( 20727),  INT16_C( 26576),
         INT16_C(  9162),  INT16_C(  1142), -INT16_C( 31864),  INT16_C( 13664), -INT16_C( 22306),  INT16_C(   505), -INT16_C( 29659),  INT16_C(   177) } },
    { {  INT16_C( 25698),  INT16_C( 30728), -INT16_C( 21943),  INT16_C( 14082),  INT16_C( 29965), -INT16_C( 18124), -INT16_C( 24490), -INT16_C( 32285),
         INT16_C( 13858), -INT16_C(  8258), -INT16_C( 18369), -INT16_C(  1563),  INT16_C( 18613), -INT16_C( 32508), -INT16_C( 15200),  INT16_C(   900),
        -INT16_C( 29655),  INT16_C( 29307),  INT16_C( 32054),  INT16_C( 17321), -INT16_C(  8461),  INT16_C( 18940), -INT16_C(  8322), -INT16_C( 24374),
        -INT16_C( 30442),  INT16_C( 21887),  INT16_C( 25665), -INT16_C(  2481),  INT16_C( 21420),  INT16_C( 19831), -INT16_C(  1000),  INT16_C( 16720) },
      UINT32_C(3216231304),
      {  INT16_C( 23625),  INT16_C( 15362), -INT16_C(   198), -INT16_C( 18299),  INT16_C( 20446), -INT16_C(  2984), -INT16_C( 10024),  INT16_C(  6730),
        -INT16_C( 26308), -INT16_C(  5872), -INT16_C( 30484),  INT16_C(  1078), -INT16_C( 31100),  INT16_C(  3141), -INT16_C(  1967), -INT16_C( 25909),
        -INT16_C( 12715), -INT16_C( 28714),  INT16_C( 23501), -INT16_C( 21688), -INT16_C( 24405), -INT16_C( 31840), -INT16_C(  5512), -INT16_C( 19043),
        -INT16_C( 20861),  INT16_C( 28574), -INT16_C( 11210), -INT16_C( 17804), -INT16_C( 18086), -INT16_C( 21562), -INT16_C( 27982),  INT16_C(  1862) },
      {  INT64_C(                   4),  INT64_C(                  10) },
      {  INT16_C( 25698),  INT16_C( 30728), -INT16_C( 21943),  INT16_C(  2952),  INT16_C( 29965), -INT16_C( 18124), -INT16_C( 24490),  INT16_C(   420),
         INT16_C(  2451),  INT16_C(  3729), -INT16_C( 18369),  INT16_C(    67),  INT16_C( 18613), -INT16_C( 32508),  INT16_C(  3973),  INT16_C(  2476),
         INT16_C(  3301),  INT16_C(  2301),  INT16_C( 32054),  INT16_C( 17321),  INT16_C(  2570),  INT16_C(  2106), -INT16_C(  8322),  INT16_C(  2905),
         INT16_C(  2792),  INT16_C(  1785),  INT16_C(  3395),  INT16_C(  2983),  INT16_C(  2965),  INT16_C(  2748), -INT16_C(  1000),  INT16_C(   116) } },
    { {  INT16_C( 11670), -INT16_C(  8584),  INT16_C(  9176),  INT16_C( 30847), -INT16_C(  2138),  INT16_C( 17506), -INT16_C(  6740),  INT16_C( 19186),
         INT16_C( 10325), -INT16_C( 14050),  INT16_C( 30946), -INT16_C( 22398),  INT16_C( 13348),  INT16_C( 27194), -INT16_C( 26053), -INT16_C( 11642),
        -INT16_C(   313), -INT16_C( 24400),  INT16_C( 12065), -INT16_C( 14312),  INT16_C( 31527), -INT16_C( 11508), -INT16_C(   416), -INT16_C( 19170),
         INT16_C( 15398),  INT16_C(  2174),  INT16_C(   437), -INT16_C(  9808), -INT16_C(  5323),  INT16_C( 28995), -INT16_C( 13947),  INT16_C( 19779) },
      UINT32_C(3924685768),
      {  INT16_C(  1315),  INT16_C( 19121), -INT16_C( 17024), -INT16_C(  7907),  INT16_C( 15291), -INT16_C(  7786),  INT16_C(  5496),  INT16_C( 11753),
        -INT16_C( 26090),  INT16_C( 19206),  INT16_C( 18821),  INT16_C(  2748), -INT16_C(   238), -INT16_C(  9641),  INT16_C( 17651),  INT16_C(  5828),
         INT16_C( 30026), -INT16_C( 13728),  INT16_C( 32051), -INT16_C(  4437),  INT16_C( 17081),  INT16_C( 12752), -INT16_C( 18089),  INT16_C( 27998),
         INT16_C( 25683), -INT16_C( 10056),  INT16_C( 30125), -INT16_C( 16413),  INT16_C( 14964),  INT16_C( 26522),  INT16_C( 24191), -INT16_C( 13955) },
      {  INT64_C(                  15),  INT64_C(                  14) },
      {  INT16_C( 11670), -INT16_C(  8584),  INT16_C(  9176),  INT16_C(     1), -INT16_C(  2138),  INT16_C( 17506),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     1),  INT16_C(     0),  INT16_C( 30946), -INT16_C( 22398),  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 24400),  INT16_C(     0),  INT16_C(     1),  INT16_C( 31527),  INT16_C(     0),  INT16_C(     1),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  2174),  INT16_C(   437),  INT16_C(     1), -INT16_C(  5323),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1) } },
    { {  INT16_C(  1683),  INT16_C( 16219),  INT16_C(  5365), -INT16_C( 14975), -INT16_C( 10171), -INT16_C( 23682), -INT16_C( 11707), -INT16_C(   761),
        -INT16_C( 19286), -INT16_C( 29326), -INT16_C(  6285),  INT16_C(  3528),  INT16_C( 18254), -INT16_C( 13205),  INT16_C( 16144), -INT16_C( 23639),
         INT16_C(  1093),  INT16_C( 15074),  INT16_C( 25368),  INT16_C( 24063),  INT16_C( 32315),        INT16_MIN,  INT16_C(  1872), -INT16_C(  1410),
        -INT16_C(  3909),  INT16_C( 12168),  INT16_C( 20695),  INT16_C(  9788), -INT16_C( 22377), -INT16_C( 22542), -INT16_C( 25625),  INT16_C( 11338) },
      UINT32_C(3093769632),
      {  INT16_C( 26256), -INT16_C( 13290),  INT16_C(  5860),  INT16_C( 13388), -INT16_C( 13794), -INT16_C(  9937), -INT16_C( 18501), -INT16_C( 28152),
         INT16_C( 17671), -INT16_C( 24904), -INT16_C( 21779), -INT16_C( 11195), -INT16_C( 28858), -INT16_C(  6656),  INT16_C( 26556),  INT16_C( 19870),
        -INT16_C( 19250), -INT16_C( 19943),  INT16_C( 26059), -INT16_C(  5657),  INT16_C(  5680), -INT16_C(  5182), -INT16_C( 13363), -INT16_C( 11139),
         INT16_C( 13840), -INT16_C(   654), -INT16_C( 18464),  INT16_C(  9937), -INT16_C( 11962),  INT16_C(   780), -INT16_C( 21703),  INT16_C(  1872) },
      {  INT64_C(                  15),  INT64_C(                  18) },
      {  INT16_C(  1683),  INT16_C( 16219),  INT16_C(  5365), -INT16_C( 14975), -INT16_C( 10171),  INT16_C(     1), -INT16_C( 11707),  INT16_C(     1),
         INT16_C(     0), -INT16_C( 29326),  INT16_C(     1),  INT16_C(     1),  INT16_C( 18254),  INT16_C(     1),  INT16_C( 16144), -INT16_C( 23639),
         INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C( 24063),  INT16_C( 32315),  INT16_C(     1),  INT16_C(     1), -INT16_C(  1410),
        -INT16_C(  3909),  INT16_C( 12168),  INT16_C( 20695),  INT16_C(     0),  INT16_C(     1),  INT16_C(     0), -INT16_C( 25625),  INT16_C(     0) } },
    { {  INT16_C( 10937), -INT16_C( 24370), -INT16_C(   493), -INT16_C( 10570), -INT16_C( 31767),  INT16_C( 26529), -INT16_C( 20137), -INT16_C( 13923),
         INT16_C( 32174),  INT16_C( 32640), -INT16_C( 14428), -INT16_C( 20400), -INT16_C( 30262),  INT16_C(  6747), -INT16_C( 17520),  INT16_C( 19075),
         INT16_C( 20965), -INT16_C(  1558), -INT16_C( 24240),  INT16_C( 14799),  INT16_C( 28708),  INT16_C( 31904),  INT16_C( 15649), -INT16_C( 12475),
        -INT16_C( 14661),  INT16_C( 24398), -INT16_C( 24947),  INT16_C( 22287),  INT16_C( 27432), -INT16_C( 18319), -INT16_C(  3034),  INT16_C(  2818) },
      UINT32_C(2500128069),
      { -INT16_C( 11378), -INT16_C( 19761),  INT16_C( 28483),  INT16_C( 25646),  INT16_C( 29869),  INT16_C( 26675), -INT16_C( 32454), -INT16_C( 14393),
        -INT16_C( 10720),  INT16_C( 18462), -INT16_C( 28863),  INT16_C( 26368),  INT16_C(   899), -INT16_C( 14221),  INT16_C( 30704),  INT16_C( 32350),
         INT16_C( 11595), -INT16_C( 29136),  INT16_C( 24476),  INT16_C( 18931),  INT16_C(  9939),  INT16_C(  3505),  INT16_C( 30888), -INT16_C( 14124),
        -INT16_C(  3505), -INT16_C( 28656),  INT16_C(  4225),  INT16_C(  1272),  INT16_C( 27411),  INT16_C(   972),  INT16_C( 10978),  INT16_C( 11649) },
      {  INT64_C(                   9),  INT64_C(                  15) },
      {  INT16_C(   105), -INT16_C( 24370),  INT16_C(    55), -INT16_C( 10570), -INT16_C( 31767),  INT16_C( 26529),  INT16_C(    64), -INT16_C( 13923),
         INT16_C(   107),  INT16_C( 32640),  INT16_C(    71),  INT16_C(    51), -INT16_C( 30262),  INT16_C(   100),  INT16_C(    59),  INT16_C(    63),
         INT16_C( 20965), -INT16_C(  1558),  INT16_C(    47),  INT16_C( 14799),  INT16_C( 28708),  INT16_C( 31904),  INT16_C( 15649), -INT16_C( 12475),
         INT16_C(   121),  INT16_C( 24398),  INT16_C(     8),  INT16_C( 22287),  INT16_C(    53), -INT16_C( 18319), -INT16_C(  3034),  INT16_C(    22) } },
    { { -INT16_C(  2884), -INT16_C( 20719), -INT16_C(  7107), -INT16_C(  4139),  INT16_C( 32241), -INT16_C( 15001), -INT16_C( 18875),  INT16_C( 21943),
         INT16_C( 14407),  INT16_C( 16230),  INT16_C( 31036),  INT16_C(  2218), -INT16_C( 29571), -INT16_C(   461), -INT16_C( 30022),  INT16_C( 30384),
        -INT16_C( 16002), -INT16_C( 17371), -INT16_C(  1371), -INT16_C( 26965),  INT16_C(  4728), -INT16_C( 17061),  INT16_C(  4809),  INT16_C(  4115),
         INT16_C( 31050), -INT16_C( 31153), -INT16_C(  1550),  INT16_C( 28559), -INT16_C( 15739),  INT16_C( 16238),  INT16_C(  7756), -INT16_C( 13387) },
      UINT32_C(2240273120),
      {  INT16_C( 13013),  INT16_C( 19740),  INT16_C( 30532),  INT16_C(  3338),  INT16_C(  7562), -INT16_C( 11235),  INT16_C( 27798), -INT16_C( 30373),
        -INT16_C(  5531), -INT16_C(  5128),  INT16_C( 26284), -INT16_C(  2006), -INT16_C(  8059),  INT16_C( 26051),  INT16_C( 19130), -INT16_C( 28694),
         INT16_C(  1660), -INT16_C( 15908), -INT16_C(  6274),  INT16_C(  2254), -INT16_C(  5116), -INT16_C( 25636),  INT16_C( 14168), -INT16_C( 16860),
         INT16_C(  7201), -INT16_C( 12887), -INT16_C( 11389),  INT16_C(  2246), -INT16_C( 30285),  INT16_C( 28269),  INT16_C( 22484),  INT16_C( 20733) },
      {  INT64_C(                   8),  INT64_C(                  14) },
      { -INT16_C(  2884), -INT16_C( 20719), -INT16_C(  7107), -INT16_C(  4139),  INT16_C( 32241),  INT16_C(   212),  INT16_C(   108),  INT16_C(   137),
         INT16_C( 14407),  INT16_C(   235),  INT16_C( 31036),  INT16_C(   248),  INT16_C(   224), -INT16_C(   461),  INT16_C(    74),  INT16_C(   143),
         INT16_C(     6),  INT16_C(   193),  INT16_C(   231), -INT16_C( 26965),  INT16_C(  4728), -INT16_C( 17061),  INT16_C(  4809),  INT16_C(   190),
         INT16_C(    28), -INT16_C( 31153),  INT16_C(   211),  INT16_C( 28559), -INT16_C( 15739),  INT16_C( 16238),  INT16_C(  7756),  INT16_C(    80) } },
    { { -INT16_C(  9199), -INT16_C(  7999), -INT16_C( 14876), -INT16_C( 16180),  INT16_C(  9312), -INT16_C( 31496),  INT16_C(  6626), -INT16_C( 29791),
         INT16_C(  9447), -INT16_C( 21153),  INT16_C(  4652), -INT16_C( 26314),  INT16_C(  2688),  INT16_C( 32496),  INT16_C( 20059),  INT16_C( 27736),
         INT16_C(  6442),  INT16_C(  3660),  INT16_C(  6366),  INT16_C( 16335), -INT16_C( 14531),  INT16_C(  8131),  INT16_C( 25824), -INT16_C( 14421),
         INT16_C(  2696), -INT16_C( 19340), -INT16_C( 21732), -INT16_C( 25267),  INT16_C( 16053),  INT16_C(  4123),  INT16_C( 29580), -INT16_C( 18563) },
      UINT32_C(1791347084),
      { -INT16_C( 27422),  INT16_C(  8105),  INT16_C( 27995),  INT16_C( 15422), -INT16_C(  5679),  INT16_C( 23043),  INT16_C( 30963),  INT16_C(  4110),
         INT16_C( 23587), -INT16_C( 10067), -INT16_C( 14182),  INT16_C(  9961),  INT16_C( 26171), -INT16_C( 14371), -INT16_C( 23761),  INT16_C(  4401),
        -INT16_C(  9417), -INT16_C( 27856),  INT16_C( 28488),  INT16_C(  6607), -INT16_C( 11688),  INT16_C( 19571), -INT16_C( 32182),  INT16_C( 27996),
         INT16_C(  2526),  INT16_C( 30790),  INT16_C( 12241),  INT16_C(  3230),  INT16_C( 31893), -INT16_C( 15149),  INT16_C(  1055),  INT16_C( 22230) },
      {  INT64_C(                   8),  INT64_C(                  19) },
      { -INT16_C(  9199), -INT16_C(  7999),  INT16_C(   109),  INT16_C(    60),  INT16_C(  9312), -INT16_C( 31496),  INT16_C(  6626),  INT16_C(    16),
         INT16_C(    92), -INT16_C( 21153),  INT16_C(  4652),  INT16_C(    38),  INT16_C(  2688),  INT16_C( 32496),  INT16_C(   163),  INT16_C(    17),
         INT16_C(   219),  INT16_C(  3660),  INT16_C(   111),  INT16_C( 16335), -INT16_C( 14531),  INT16_C(  8131),  INT16_C(   130),  INT16_C(   109),
         INT16_C(  2696),  INT16_C(   120), -INT16_C( 21732),  INT16_C(    12),  INT16_C( 16053),  INT16_C(   196),  INT16_C(     4), -INT16_C( 18563) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_srl_epi16(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_srl_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m512i r = easysimd_mm512_mask_srl_epi16(src, k, a, count);

    easysimd_test_x86_write_i16x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm256_mask_srl_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const easysimd__mmask16 k;
    const int16_t a[16];
    const int64_t count[2];
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C( 27395), -INT16_C(  7785),  INT16_C( 19511), -INT16_C(  2329),  INT16_C( 19973),  INT16_C( 18161),  INT16_C( 12539),  INT16_C( 25458),
         INT16_C( 31319),  INT16_C( 16827), -INT16_C( 11745),  INT16_C( 22964),  INT16_C( 19575), -INT16_C( 26049), -INT16_C(  1871), -INT16_C( 19221) },
      UINT16_C(33379),
      { -INT16_C( 25963),  INT16_C( 32206), -INT16_C( 11120), -INT16_C( 32309), -INT16_C( 14822), -INT16_C( 29519),  INT16_C(  2346), -INT16_C(  6905),
         INT16_C(  9802), -INT16_C(   329),  INT16_C( 12160), -INT16_C( 16566), -INT16_C(  1079), -INT16_C( 19273),  INT16_C(  6832),  INT16_C( 17718) },
      {  INT64_C(                  19),  INT64_C(                  11) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 19511), -INT16_C(  2329),  INT16_C( 19973),  INT16_C(     0),  INT16_C(     0),  INT16_C( 25458),
         INT16_C( 31319),  INT16_C(     0), -INT16_C( 11745),  INT16_C( 22964),  INT16_C( 19575), -INT16_C( 26049), -INT16_C(  1871),  INT16_C(     0) } },
    { {  INT16_C( 17858), -INT16_C( 29223), -INT16_C(  3130),  INT16_C( 30804),  INT16_C( 32383), -INT16_C( 31103), -INT16_C( 13469),  INT16_C(  6829),
         INT16_C( 11721),  INT16_C(  5193),  INT16_C(  5100), -INT16_C( 23793), -INT16_C( 16441), -INT16_C(   323),  INT16_C( 28933), -INT16_C( 14589) },
      UINT16_C(56502),
      {  INT16_C( 32085), -INT16_C( 22065),  INT16_C( 20213),  INT16_C( 30247), -INT16_C( 29995), -INT16_C( 32191),  INT16_C(  2724), -INT16_C(  4433),
        -INT16_C( 25826),  INT16_C( 11777), -INT16_C( 14274), -INT16_C(  1043), -INT16_C(  3386), -INT16_C( 13972),  INT16_C(  9146),  INT16_C(  4005) },
      {  INT64_C(                   2),  INT64_C(                  11) },
      {  INT16_C( 17858),  INT16_C( 10867),  INT16_C(  5053),  INT16_C( 30804),  INT16_C(  8885),  INT16_C(  8336), -INT16_C( 13469),  INT16_C( 15275),
         INT16_C( 11721),  INT16_C(  5193),  INT16_C( 12815),  INT16_C( 16123),  INT16_C( 15537), -INT16_C(   323),  INT16_C(  2286),  INT16_C(  1001) } },
    { { -INT16_C( 27208), -INT16_C(  8253), -INT16_C( 26613),  INT16_C( 19561),  INT16_C(  3354), -INT16_C( 13994),  INT16_C( 30203), -INT16_C(   924),
        -INT16_C( 23901), -INT16_C( 28475), -INT16_C( 29795),  INT16_C(  2435),  INT16_C( 15701), -INT16_C(  1492), -INT16_C( 13236),  INT16_C(  1135) },
      UINT16_C(12897),
      {  INT16_C( 27875),  INT16_C( 19658), -INT16_C(  6984),  INT16_C(  3929),  INT16_C( 21933),  INT16_C(  4484),  INT16_C( 10065),  INT16_C(  5811),
         INT16_C( 20663),  INT16_C( 15010), -INT16_C(  2215), -INT16_C( 31113), -INT16_C( 15375),  INT16_C( 24658), -INT16_C( 19257), -INT16_C( 21870) },
      {  INT64_C(                   6),  INT64_C(                   1) },
      {  INT16_C(   435), -INT16_C(  8253), -INT16_C( 26613),  INT16_C( 19561),  INT16_C(  3354),  INT16_C(    70),  INT16_C(   157), -INT16_C(   924),
        -INT16_C( 23901),  INT16_C(   234), -INT16_C( 29795),  INT16_C(  2435),  INT16_C(   783),  INT16_C(   385), -INT16_C( 13236),  INT16_C(  1135) } },
    { { -INT16_C(  9738),  INT16_C( 20544), -INT16_C(  4632),  INT16_C( 27813), -INT16_C(  2306), -INT16_C( 20077),  INT16_C( 18957), -INT16_C( 20735),
         INT16_C( 23429), -INT16_C(   858), -INT16_C( 26655),  INT16_C( 13248), -INT16_C( 30728), -INT16_C( 29977),  INT16_C(  2098),  INT16_C( 10471) },
      UINT16_C(10209),
      { -INT16_C( 13960),  INT16_C(  7445),  INT16_C(  4917), -INT16_C( 14316),  INT16_C(  8645), -INT16_C( 14830), -INT16_C( 26672),  INT16_C( 30241),
         INT16_C(   660),  INT16_C( 21517),  INT16_C(  1334),  INT16_C(  7643),  INT16_C(  3472),  INT16_C( 30501),  INT16_C(  1590), -INT16_C( 20834) },
      {  INT64_C(                  18),  INT64_C(                   2) },
      {  INT16_C(     0),  INT16_C( 20544), -INT16_C(  4632),  INT16_C( 27813), -INT16_C(  2306),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 13248), -INT16_C( 30728),  INT16_C(     0),  INT16_C(  2098),  INT16_C( 10471) } },
    { {  INT16_C(  1228), -INT16_C(  7993), -INT16_C( 29492), -INT16_C(  8447), -INT16_C( 11950),  INT16_C( 29814),  INT16_C(  2631),  INT16_C( 21622),
        -INT16_C( 21410),  INT16_C( 14938), -INT16_C(  5430), -INT16_C(  4281),  INT16_C( 32097), -INT16_C(    10), -INT16_C( 15060), -INT16_C(  1869) },
      UINT16_C(31434),
      { -INT16_C( 26920), -INT16_C(  9978),  INT16_C( 22645), -INT16_C(  4950), -INT16_C(  3636),  INT16_C( 17398),  INT16_C( 21829), -INT16_C( 24593),
        -INT16_C( 18033), -INT16_C( 10615), -INT16_C(  5463), -INT16_C( 24748), -INT16_C( 32534), -INT16_C( 25244),  INT16_C( 11896),  INT16_C( 20503) },
      {  INT64_C(                   2),  INT64_C(                   7) },
      {  INT16_C(  1228),  INT16_C( 13889), -INT16_C( 29492),  INT16_C( 15146), -INT16_C( 11950),  INT16_C( 29814),  INT16_C(  5457),  INT16_C( 10235),
        -INT16_C( 21410),  INT16_C( 13730), -INT16_C(  5430),  INT16_C( 10197),  INT16_C(  8250),  INT16_C( 10073),  INT16_C(  2974), -INT16_C(  1869) } },
    { {  INT16_C( 14889), -INT16_C( 11403),  INT16_C( 16934),  INT16_C(  7620),  INT16_C(  2437),  INT16_C( 29810),  INT16_C(   425),  INT16_C( 12846),
        -INT16_C( 10281),  INT16_C( 11037),  INT16_C(  1910), -INT16_C(  9557),  INT16_C(  9124), -INT16_C( 17655), -INT16_C( 12685), -INT16_C( 25384) },
      UINT16_C(19720),
      {  INT16_C( 12143),  INT16_C( 13199),  INT16_C(  5196), -INT16_C( 16835), -INT16_C(  6519), -INT16_C( 18497), -INT16_C( 27112),  INT16_C( 13710),
         INT16_C(  1218),  INT16_C( 27964), -INT16_C(  7970), -INT16_C(  6255),  INT16_C(  1179),  INT16_C( 29621), -INT16_C( 16735),  INT16_C(  4289) },
      {  INT64_C(                  14),  INT64_C(                   0) },
      {  INT16_C( 14889), -INT16_C( 11403),  INT16_C( 16934),  INT16_C(     2),  INT16_C(  2437),  INT16_C( 29810),  INT16_C(   425),  INT16_C( 12846),
         INT16_C(     0),  INT16_C( 11037),  INT16_C(     3),  INT16_C(     3),  INT16_C(  9124), -INT16_C( 17655),  INT16_C(     2), -INT16_C( 25384) } },
    { {  INT16_C( 14660), -INT16_C( 32411), -INT16_C(  4361), -INT16_C( 18841),  INT16_C( 32677),  INT16_C( 13132),  INT16_C(  3765), -INT16_C(  3785),
         INT16_C(  5500),  INT16_C(  3538),  INT16_C( 28157), -INT16_C( 19951), -INT16_C( 19743), -INT16_C( 23952),  INT16_C( 24003),  INT16_C(  2034) },
      UINT16_C(22422),
      { -INT16_C( 29304), -INT16_C(  4283), -INT16_C(  5565), -INT16_C( 28562),  INT16_C(  8989),  INT16_C( 21662),  INT16_C(  6677), -INT16_C(  6294),
         INT16_C( 26407),  INT16_C( 14676),  INT16_C( 13593), -INT16_C( 29973), -INT16_C( 20777), -INT16_C( 13593),  INT16_C( 32437),  INT16_C( 15649) },
      {  INT64_C(                  16),  INT64_C(                   0) },
      {  INT16_C( 14660),  INT16_C(     0),  INT16_C(     0), -INT16_C( 18841),  INT16_C(     0),  INT16_C( 13132),  INT16_C(  3765),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 19951),  INT16_C(     0), -INT16_C( 23952),  INT16_C(     0),  INT16_C(  2034) } },
    { {  INT16_C( 20268), -INT16_C( 25775),  INT16_C( 28639),  INT16_C( 32190), -INT16_C( 11325),  INT16_C( 11672), -INT16_C( 16454),  INT16_C(  3988),
        -INT16_C( 20744), -INT16_C(  7100),  INT16_C(  7224),  INT16_C(  8082),  INT16_C( 18662),  INT16_C(  1949), -INT16_C( 22139), -INT16_C( 19858) },
      UINT16_C(49400),
      { -INT16_C( 10419),  INT16_C(  2863), -INT16_C(  3500), -INT16_C(  4897), -INT16_C( 26336), -INT16_C( 19284), -INT16_C( 23384), -INT16_C(  4766),
        -INT16_C( 25976),  INT16_C(  6921), -INT16_C(  4166),  INT16_C( 22371), -INT16_C(  5898),  INT16_C( 25856), -INT16_C(  1894), -INT16_C(  6363) },
      {  INT64_C(                  16),  INT64_C(                  10) },
      {  INT16_C( 20268), -INT16_C( 25775),  INT16_C( 28639),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 20744), -INT16_C(  7100),  INT16_C(  7224),  INT16_C(  8082),  INT16_C( 18662),  INT16_C(  1949),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_srl_epi16(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_srl_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_mask_srl_epi16(src, k, a, count);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm_mask_srl_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int64_t count[2];
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C( 14953),  INT16_C(  4688), -INT16_C( 19038),  INT16_C( 24844),  INT16_C( 10473),  INT16_C( 18941),  INT16_C(  1607),  INT16_C(  5140) },
      UINT8_C(129),
      { -INT16_C( 25305),  INT16_C(  8744), -INT16_C( 24571), -INT16_C( 19627),  INT16_C(  3447),  INT16_C( 30313), -INT16_C( 28302), -INT16_C( 21281) },
      {  INT64_C(                   4),  INT64_C(                  13) },
      {  INT16_C(  2514),  INT16_C(  4688), -INT16_C( 19038),  INT16_C( 24844),  INT16_C( 10473),  INT16_C( 18941),  INT16_C(  1607),  INT16_C(  2765) } },
    { { -INT16_C( 27058), -INT16_C( 20227),  INT16_C(  9599), -INT16_C( 14163), -INT16_C( 19348), -INT16_C( 32547),  INT16_C(  1077),  INT16_C( 23837) },
      UINT8_C( 39),
      { -INT16_C(   477), -INT16_C( 10628), -INT16_C( 30347), -INT16_C(  5312), -INT16_C( 11781), -INT16_C( 22582), -INT16_C( 17486),  INT16_C( 18677) },
      {  INT64_C(                  14),  INT64_C(                   3) },
      {  INT16_C(     3),  INT16_C(     3),  INT16_C(     2), -INT16_C( 14163), -INT16_C( 19348),  INT16_C(     2),  INT16_C(  1077),  INT16_C( 23837) } },
    { { -INT16_C(  8760), -INT16_C( 28589),  INT16_C(  1865), -INT16_C( 13971),  INT16_C( 29244), -INT16_C( 26137),  INT16_C(  2713),  INT16_C(  5527) },
      UINT8_C(224),
      { -INT16_C( 25076), -INT16_C(  2272), -INT16_C(  3687),  INT16_C( 16577),  INT16_C( 31907), -INT16_C(  5067), -INT16_C(  9420),  INT16_C(  4532) },
      {  INT64_C(                  10),  INT64_C(                  14) },
      { -INT16_C(  8760), -INT16_C( 28589),  INT16_C(  1865), -INT16_C( 13971),  INT16_C( 29244),  INT16_C(    59),  INT16_C(    54),  INT16_C(     4) } },
    { {  INT16_C( 13658),  INT16_C(  9394),  INT16_C(  9329),  INT16_C(  2571),  INT16_C(  5565), -INT16_C( 11614), -INT16_C( 20747),  INT16_C(  5744) },
      UINT8_C(166),
      {  INT16_C(  1801),  INT16_C( 18791), -INT16_C(  6997), -INT16_C( 26754),  INT16_C( 22808),  INT16_C( 10827), -INT16_C( 28793), -INT16_C( 17276) },
      {  INT64_C(                  10),  INT64_C(                   3) },
      {  INT16_C( 13658),  INT16_C(    18),  INT16_C(    57),  INT16_C(  2571),  INT16_C(  5565),  INT16_C(    10), -INT16_C( 20747),  INT16_C(    47) } },
    { {  INT16_C( 25901),  INT16_C( 14515), -INT16_C( 14302), -INT16_C(  2854), -INT16_C( 30530), -INT16_C( 11164),  INT16_C( 27950), -INT16_C( 26917) },
      UINT8_C(182),
      {  INT16_C( 31366),  INT16_C(  7477), -INT16_C( 29038), -INT16_C( 17304), -INT16_C(  2026), -INT16_C( 11711), -INT16_C(  5831), -INT16_C( 24832) },
      {  INT64_C(                  19),  INT64_C(                  17) },
      {  INT16_C( 25901),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2854),  INT16_C(     0),  INT16_C(     0),  INT16_C( 27950),  INT16_C(     0) } },
    { {  INT16_C( 26049), -INT16_C( 18926), -INT16_C( 26077), -INT16_C(  2278), -INT16_C( 30519),  INT16_C( 24531),  INT16_C( 22846),  INT16_C( 29657) },
      UINT8_C(119),
      {  INT16_C(   619),  INT16_C( 10463), -INT16_C( 10472), -INT16_C(  5527),  INT16_C( 21009), -INT16_C( 20246),  INT16_C(  8943),  INT16_C( 21873) },
      {  INT64_C(                   6),  INT64_C(                   6) },
      {  INT16_C(     9),  INT16_C(   163),  INT16_C(   860), -INT16_C(  2278),  INT16_C(   328),  INT16_C(   707),  INT16_C(   139),  INT16_C( 29657) } },
    { { -INT16_C( 12424),  INT16_C( 28738), -INT16_C( 13672), -INT16_C(  2237), -INT16_C( 25592),  INT16_C( 31952),  INT16_C( 15123), -INT16_C(  3202) },
      UINT8_C( 99),
      { -INT16_C( 13674), -INT16_C( 32564),  INT16_C(  8155), -INT16_C( 29845), -INT16_C( 29426),  INT16_C( 25597),  INT16_C(  9410), -INT16_C( 28196) },
      {  INT64_C(                  19),  INT64_C(                  19) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 13672), -INT16_C(  2237), -INT16_C( 25592),  INT16_C(     0),  INT16_C(     0), -INT16_C(  3202) } },
    { {  INT16_C( 12329),  INT16_C(  8335),  INT16_C( 11065), -INT16_C( 18960),  INT16_C( 11071),  INT16_C( 12851), -INT16_C( 13937),  INT16_C( 23548) },
      UINT8_C( 73),
      {  INT16_C( 31448),  INT16_C( 25524),  INT16_C( 17033), -INT16_C(  5024), -INT16_C( 31484), -INT16_C( 27192),  INT16_C(  5355),  INT16_C(  7358) },
      {  INT64_C(                   5),  INT64_C(                   8) },
      {  INT16_C(   982),  INT16_C(  8335),  INT16_C( 11065),  INT16_C(  1891),  INT16_C( 11071),  INT16_C( 12851),  INT16_C(   167),  INT16_C( 23548) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_srl_epi16(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_srl_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_mask_srl_epi16(src, k, a, count);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm512_maskz_srl_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int64_t count[2];
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(2511173183),
      { -INT16_C( 18352),  INT16_C( 13062),  INT16_C( 23117), -INT16_C(  6308),  INT16_C( 26199), -INT16_C( 15261), -INT16_C( 14957), -INT16_C(  1899),
        -INT16_C( 32114),  INT16_C( 23766),  INT16_C( 13337),  INT16_C( 10921),  INT16_C( 30553), -INT16_C( 26450),  INT16_C( 23533),  INT16_C( 15918),
         INT16_C( 13331),  INT16_C( 24689), -INT16_C( 12657), -INT16_C(  6585), -INT16_C( 21964), -INT16_C( 14166),  INT16_C( 16239), -INT16_C(   576),
        -INT16_C( 26943), -INT16_C(  9383),  INT16_C(   970),  INT16_C(  9221), -INT16_C( 19590),  INT16_C( 26812), -INT16_C(  5618),  INT16_C(  8614) },
      {  INT64_C(                   6),  INT64_C(                  18) },
      {  INT16_C(   737),  INT16_C(   204),  INT16_C(   361),  INT16_C(   925),  INT16_C(   409),  INT16_C(   785),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(   371),  INT16_C(   208),  INT16_C(     0),  INT16_C(   477),  INT16_C(   610),  INT16_C(   367),  INT16_C(     0),
         INT16_C(   208),  INT16_C(     0),  INT16_C(   826),  INT16_C(   921),  INT16_C(     0),  INT16_C(   802),  INT16_C(     0),  INT16_C(  1015),
         INT16_C(   603),  INT16_C(     0),  INT16_C(    15),  INT16_C(     0),  INT16_C(   717),  INT16_C(     0),  INT16_C(     0),  INT16_C(   134) } },
    { UINT32_C(3370495617),
      {  INT16_C(  6804),  INT16_C( 15986), -INT16_C(  7710), -INT16_C( 23938),  INT16_C( 16350),  INT16_C( 14137),  INT16_C(   794),  INT16_C(  7994),
        -INT16_C( 19161), -INT16_C(  6958), -INT16_C(  8163), -INT16_C( 15410), -INT16_C(  4863), -INT16_C( 32038), -INT16_C( 16229),  INT16_C( 12106),
        -INT16_C( 17190), -INT16_C( 17298), -INT16_C(  4963),  INT16_C( 31582), -INT16_C( 26837),  INT16_C( 18099), -INT16_C(  4709), -INT16_C( 15771),
         INT16_C( 14498), -INT16_C( 16474),  INT16_C( 29976),  INT16_C(  6786),  INT16_C( 23906), -INT16_C(   356), -INT16_C(  6371), -INT16_C(  2259) },
      {  INT64_C(                  13),  INT64_C(                  13) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     7),  INT16_C(     7),  INT16_C(     6),  INT16_C(     0),  INT16_C(     4),  INT16_C(     0),  INT16_C(     1),
         INT16_C(     5),  INT16_C(     0),  INT16_C(     7),  INT16_C(     0),  INT16_C(     0),  INT16_C(     2),  INT16_C(     7),  INT16_C(     6),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     7),  INT16_C(     7) } },
    { UINT32_C( 294076851),
      { -INT16_C( 19524),  INT16_C( 28585),  INT16_C( 17657),  INT16_C( 24157), -INT16_C(   250), -INT16_C( 21098), -INT16_C( 20545),  INT16_C( 16674),
        -INT16_C( 31543),  INT16_C( 26014), -INT16_C( 17534), -INT16_C( 20404), -INT16_C(  3918),  INT16_C( 25931), -INT16_C( 11471), -INT16_C(  4745),
         INT16_C(  8326),  INT16_C( 32605), -INT16_C( 17820),  INT16_C( 27357),  INT16_C( 29881),  INT16_C( 30743),  INT16_C( 14627), -INT16_C(  4934),
         INT16_C( 22718),  INT16_C( 16465), -INT16_C( 25068), -INT16_C( 14608),  INT16_C( 15502), -INT16_C( 16596), -INT16_C( 23793), -INT16_C( 27220) },
      {  INT64_C(                  11),  INT64_C(                   6) },
      {  INT16_C(    22),  INT16_C(    13),  INT16_C(     0),  INT16_C(     0),  INT16_C(    31),  INT16_C(    21),  INT16_C(     0),  INT16_C(     8),
         INT16_C(    16),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    26),  INT16_C(     0),
         INT16_C(     4),  INT16_C(    15),  INT16_C(    23),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    29),
         INT16_C(    11),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     7),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(4056098580),
      {  INT16_C( 32145), -INT16_C( 22171), -INT16_C( 30475), -INT16_C( 20510), -INT16_C( 24460), -INT16_C( 14840),  INT16_C(  7393), -INT16_C( 11932),
        -INT16_C(  3358),  INT16_C(  3597),  INT16_C(  7345),  INT16_C( 23985),  INT16_C( 29873), -INT16_C( 15001),  INT16_C( 10907),  INT16_C( 11703),
         INT16_C(  7335), -INT16_C( 25130), -INT16_C( 18267),  INT16_C(  6476),  INT16_C( 21593),  INT16_C( 15071),  INT16_C( 17264),  INT16_C( 21259),
         INT16_C(  6453), -INT16_C(  6559),  INT16_C(  4917), -INT16_C(  6332), -INT16_C( 21625),  INT16_C(  9132),  INT16_C( 25557),  INT16_C( 32080) },
      {  INT64_C(                   3),  INT64_C(                  15) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(  4382),  INT16_C(     0),  INT16_C(  5134),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  7772),  INT16_C(   449),  INT16_C(   918),  INT16_C(     0),  INT16_C(     0),  INT16_C(  6316),  INT16_C(     0),  INT16_C(     0),
         INT16_C(   916),  INT16_C(  5050),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2158),  INT16_C(  2657),
         INT16_C(   806),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5488),  INT16_C(  1141),  INT16_C(  3194),  INT16_C(  4010) } },
    { UINT32_C(1725834522),
      {  INT16_C( 14142),  INT16_C(  7867),  INT16_C( 11121),  INT16_C( 32097), -INT16_C( 26754), -INT16_C(  8042), -INT16_C( 13443), -INT16_C( 15885),
         INT16_C( 31410),  INT16_C( 24428),  INT16_C( 17053), -INT16_C(  4670),  INT16_C( 17087), -INT16_C(  9965), -INT16_C(  3481), -INT16_C( 22977),
        -INT16_C(  1495), -INT16_C( 25660),  INT16_C(  9510), -INT16_C( 23528), -INT16_C( 20804),  INT16_C( 14980),  INT16_C( 30585),  INT16_C( 11515),
         INT16_C( 26866), -INT16_C( 28789),  INT16_C( 19882),  INT16_C( 27005), -INT16_C( 28528), -INT16_C(  2238), -INT16_C( 32382), -INT16_C( 21347) },
      {  INT64_C(                  17),  INT64_C(                   0) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(1602724423),
      {  INT16_C( 17222), -INT16_C( 13555), -INT16_C( 31107),  INT16_C( 31042),  INT16_C( 13490),  INT16_C( 15841), -INT16_C( 29756),  INT16_C( 16779),
         INT16_C(  7156),  INT16_C( 14033),  INT16_C( 21522), -INT16_C( 20297),  INT16_C( 13056),  INT16_C( 18193), -INT16_C( 26411),  INT16_C(  7334),
        -INT16_C( 19492),  INT16_C( 23015),  INT16_C( 10553), -INT16_C(  4910), -INT16_C( 19618),  INT16_C(  8745), -INT16_C( 19394),  INT16_C( 12899),
         INT16_C( 13519), -INT16_C(  7576),  INT16_C(  8328), -INT16_C( 30574), -INT16_C( 23725),  INT16_C( 10703),  INT16_C( 30012),  INT16_C(  6213) },
      {  INT64_C(                  12),  INT64_C(                  12) },
      {  INT16_C(     4),  INT16_C(    12),  INT16_C(     8),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     8),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     3),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     4),  INT16_C(     0),  INT16_C(     1),
         INT16_C(    11),  INT16_C(     5),  INT16_C(     2),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     3),
         INT16_C(     3),  INT16_C(    14),  INT16_C(     2),  INT16_C(     8),  INT16_C(    10),  INT16_C(     0),  INT16_C(     7),  INT16_C(     0) } },
    { UINT32_C(1146446449),
      { -INT16_C( 19634),  INT16_C( 30711),  INT16_C( 14037),  INT16_C( 14380), -INT16_C(  1176), -INT16_C( 11923), -INT16_C(  2595),  INT16_C( 28657),
         INT16_C( 17534),  INT16_C( 19731),  INT16_C( 20333), -INT16_C( 19773), -INT16_C(  5273), -INT16_C( 10018),  INT16_C( 13389), -INT16_C( 25828),
         INT16_C(  5351), -INT16_C( 17133),  INT16_C( 16202), -INT16_C( 19723),  INT16_C( 25146),  INT16_C(  6275),  INT16_C( 29784), -INT16_C( 10617),
        -INT16_C( 25927),  INT16_C(  9763), -INT16_C(  6423),  INT16_C( 20697), -INT16_C( 18478),  INT16_C(  7977),  INT16_C( 17899), -INT16_C( 11333) },
      {  INT64_C(                   4),  INT64_C(                   1) },
      {  INT16_C(  2868),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4022),  INT16_C(  3350),  INT16_C(  3933),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  1233),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3469),  INT16_C(   836),  INT16_C(     0),
         INT16_C(   334),  INT16_C(     0),  INT16_C(  1012),  INT16_C(     0),  INT16_C(  1571),  INT16_C(     0),  INT16_C(  1861),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  3694),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1118),  INT16_C(     0) } },
    { UINT32_C(2232263568),
      {  INT16_C( 18262), -INT16_C(  9752),  INT16_C( 16479), -INT16_C(  6322),  INT16_C(  1814),  INT16_C( 14721),  INT16_C( 27437),  INT16_C(  1568),
        -INT16_C(  3397), -INT16_C(  6978), -INT16_C( 22255), -INT16_C( 13270), -INT16_C( 31876),  INT16_C(  3226), -INT16_C( 22745),  INT16_C( 32146),
         INT16_C( 31471),  INT16_C( 20054), -INT16_C( 23366), -INT16_C( 12235), -INT16_C( 18517), -INT16_C(  9975),  INT16_C( 10530), -INT16_C(  8737),
        -INT16_C( 25317),  INT16_C( 11714), -INT16_C(  5049), -INT16_C( 15367), -INT16_C( 27537), -INT16_C( 26928),  INT16_C( 25147),  INT16_C( 10771) },
      {  INT64_C(                  13),  INT64_C(                  14) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     7),  INT16_C(     7),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     3),
         INT16_C(     3),  INT16_C(     0),  INT16_C(     5),  INT16_C(     6),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     4),  INT16_C(     0),  INT16_C(     7),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_srl_epi16(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_srl_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m512i r = easysimd_mm512_maskz_srl_epi16(k, a, count);

    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm256_maskz_srl_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int16_t a[16];
    const int64_t count[2];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(35708),
      { -INT16_C( 21143),  INT16_C( 21013), -INT16_C( 20747),  INT16_C( 12494),  INT16_C(  2471), -INT16_C( 31907),  INT16_C( 28698), -INT16_C(  5717),
        -INT16_C(  9871),  INT16_C(  3481), -INT16_C( 23496), -INT16_C( 15262), -INT16_C( 17796),  INT16_C( 12460),  INT16_C( 10467),  INT16_C( 19899) },
      {  INT64_C(                  16),  INT64_C(                   2) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(51871),
      {  INT16_C( 28031),  INT16_C(  9979),  INT16_C( 22646), -INT16_C( 28247),  INT16_C( 21960),  INT16_C( 14970),  INT16_C(  4910),  INT16_C( 26183),
        -INT16_C( 22089),  INT16_C( 13099), -INT16_C( 10397),  INT16_C( 18275),  INT16_C(  8191), -INT16_C( 11116),  INT16_C( 13295),  INT16_C( 28318) },
      {  INT64_C(                  19),  INT64_C(                  18) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C( 5780),
      {  INT16_C( 16113), -INT16_C( 17753),  INT16_C(  8595), -INT16_C( 15884),  INT16_C( 15157), -INT16_C(  5081),  INT16_C( 21220),  INT16_C( 18464),
        -INT16_C( 31959),  INT16_C( 10383),  INT16_C(  9122), -INT16_C( 27908), -INT16_C( 25770), -INT16_C(  2560), -INT16_C( 27340),  INT16_C(  9740) },
      {  INT64_C(                  12),  INT64_C(                   8) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     2),  INT16_C(     0),  INT16_C(     3),  INT16_C(     0),  INT16_C(     0),  INT16_C(     4),
         INT16_C(     0),  INT16_C(     2),  INT16_C(     2),  INT16_C(     0),  INT16_C(     9),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(26336),
      { -INT16_C( 11051),  INT16_C(  2599),  INT16_C( 19983), -INT16_C(  3081),  INT16_C(  6049), -INT16_C( 13765), -INT16_C( 13670),  INT16_C( 15859),
        -INT16_C(  4115),  INT16_C( 17359), -INT16_C( 12406), -INT16_C( 16583),  INT16_C( 18020),  INT16_C( 14309), -INT16_C( 14854), -INT16_C( 12387) },
      {  INT64_C(                   3),  INT64_C(                   5) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  6471),  INT16_C(  6483),  INT16_C(  1982),
         INT16_C(     0),  INT16_C(  2169),  INT16_C(  6641),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1788),  INT16_C(  6335),  INT16_C(     0) } },
    { UINT16_C(43226),
      { -INT16_C( 12013), -INT16_C( 19301), -INT16_C( 10264), -INT16_C( 32130),  INT16_C( 29089), -INT16_C( 28737), -INT16_C( 29087), -INT16_C(  5166),
         INT16_C(  3166), -INT16_C( 15702), -INT16_C( 28846),  INT16_C( 19706), -INT16_C( 26796), -INT16_C(  4837), -INT16_C(  2724),  INT16_C( 28565) },
      {  INT64_C(                  19),  INT64_C(                   2) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(44579),
      { -INT16_C( 24312), -INT16_C( 22223), -INT16_C(  4077),  INT16_C( 29752),  INT16_C(  2943), -INT16_C(  8865),  INT16_C(  2583),  INT16_C( 27039),
        -INT16_C( 26215), -INT16_C(  4427), -INT16_C( 12239), -INT16_C( 29221),  INT16_C( 29126), -INT16_C( 29444),  INT16_C(  8098), -INT16_C( 21957) },
      {  INT64_C(                  13),  INT64_C(                   1) },
      {  INT16_C(     5),  INT16_C(     5),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     6),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     7),  INT16_C(     6),  INT16_C(     4),  INT16_C(     0),  INT16_C(     4),  INT16_C(     0),  INT16_C(     5) } },
    { UINT16_C(54099),
      { -INT16_C( 29604), -INT16_C(  9401), -INT16_C( 22633), -INT16_C( 20808),  INT16_C( 22705),  INT16_C( 18967), -INT16_C( 13071),  INT16_C(  8760),
         INT16_C(  5276),  INT16_C( 25263), -INT16_C( 21627),  INT16_C( 10223),  INT16_C( 10954), -INT16_C( 29743),  INT16_C(  9366), -INT16_C(  3490) },
      {  INT64_C(                  15),  INT64_C(                   8) },
      {  INT16_C(     1),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1) } },
    { UINT16_C(18382),
      { -INT16_C( 31155), -INT16_C(   267),  INT16_C(  3294), -INT16_C( 12216), -INT16_C( 32296),  INT16_C( 30194), -INT16_C( 23915),  INT16_C(  6871),
        -INT16_C( 14771),  INT16_C(  6209),  INT16_C(  4848), -INT16_C( 31069),  INT16_C(   310), -INT16_C(  6279),  INT16_C( 18343), -INT16_C(  3026) },
      {  INT64_C(                   6),  INT64_C(                   2) },
      {  INT16_C(     0),  INT16_C(  1019),  INT16_C(    51),  INT16_C(   833),  INT16_C(     0),  INT16_C(     0),  INT16_C(   650),  INT16_C(   107),
         INT16_C(   793),  INT16_C(    97),  INT16_C(    75),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   286),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_srl_epi16(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_srl_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_maskz_srl_epi16(k, a, count);

    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm_maskz_srl_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int64_t count[2];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(141),
      { -INT16_C( 11849), -INT16_C( 14580),  INT16_C( 22320),  INT16_C(  4655), -INT16_C(   644), -INT16_C( 24701),  INT16_C( 16204), -INT16_C( 16299) },
      {  INT64_C(                   1),  INT64_C(                   5) },
      {  INT16_C( 26843),  INT16_C(     0),  INT16_C( 11160),  INT16_C(  2327),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24618) } },
    { UINT8_C(237),
      {  INT16_C( 30757),  INT16_C( 26901),  INT16_C( 25129), -INT16_C( 10032), -INT16_C( 15815), -INT16_C( 14523),  INT16_C(  6009),  INT16_C( 16851) },
      {  INT64_C(                   9),  INT64_C(                  14) },
      {  INT16_C(    60),  INT16_C(     0),  INT16_C(    49),  INT16_C(   108),  INT16_C(     0),  INT16_C(    99),  INT16_C(    11),  INT16_C(    32) } },
    { UINT8_C(112),
      { -INT16_C( 22694), -INT16_C(  8851), -INT16_C( 18106), -INT16_C( 25571), -INT16_C( 11398),  INT16_C( 26438), -INT16_C( 16648),  INT16_C( 24956) },
      {  INT64_C(                  17),  INT64_C(                  11) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 49),
      {  INT16_C(  6336),  INT16_C(  1779),  INT16_C( 27871), -INT16_C( 19939),  INT16_C( 25773),  INT16_C(  7901), -INT16_C( 31554), -INT16_C( 25461) },
      {  INT64_C(                  11),  INT64_C(                   5) },
      {  INT16_C(     3),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    12),  INT16_C(     3),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(185),
      { -INT16_C( 16537), -INT16_C( 21108), -INT16_C( 31706), -INT16_C( 23957),  INT16_C( 21477),  INT16_C(  5761), -INT16_C( 26348),  INT16_C(  6665) },
      {  INT64_C(                   3),  INT64_C(                  13) },
      {  INT16_C(  6124),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5197),  INT16_C(  2684),  INT16_C(   720),  INT16_C(     0),  INT16_C(   833) } },
    { UINT8_C( 55),
      {  INT16_C(  9002),  INT16_C(  2203),  INT16_C( 23105), -INT16_C( 13172),  INT16_C( 22518), -INT16_C( 20719), -INT16_C( 12098),  INT16_C( 27451) },
      {  INT64_C(                   5),  INT64_C(                  10) },
      {  INT16_C(   281),  INT16_C(    68),  INT16_C(   722),  INT16_C(     0),  INT16_C(   703),  INT16_C(  1400),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(215),
      { -INT16_C( 23399),  INT16_C(  6698),  INT16_C( 16058), -INT16_C( 15437),  INT16_C( 11096), -INT16_C( 28872),  INT16_C( 23381),  INT16_C( 23851) },
      {  INT64_C(                   9),  INT64_C(                  16) },
      {  INT16_C(    82),  INT16_C(    13),  INT16_C(    31),  INT16_C(     0),  INT16_C(    21),  INT16_C(     0),  INT16_C(    45),  INT16_C(    46) } },
    { UINT8_C(234),
      {  INT16_C( 31593),  INT16_C( 31297),  INT16_C(    42),  INT16_C( 25931),  INT16_C( 16747),  INT16_C( 16932), -INT16_C( 14118), -INT16_C(  2963) },
      {  INT64_C(                   5),  INT64_C(                  15) },
      {  INT16_C(     0),  INT16_C(   978),  INT16_C(     0),  INT16_C(   810),  INT16_C(     0),  INT16_C(   529),  INT16_C(  1606),  INT16_C(  1955) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_srl_epi16(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_srl_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_maskz_srl_epi16(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_srl_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int64_t b[2];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1475230993),  INT32_C(   227741951), -INT32_C(  1586039920), -INT32_C(   909665363),  INT32_C(  1410969055),  INT32_C(   106491510),  INT32_C(   829046348),  INT32_C(  1384343139),
         INT32_C(   670733352),  INT32_C(   892636837),  INT32_C(   785820289),  INT32_C(   620272965), -INT32_C(  1049030325),  INT32_C(  1288163584),  INT32_C(  1920807183), -INT32_C(  1060831080) },
      {  INT64_C(                  25),  INT64_C(                   5) },
      {  INT32_C(          84),  INT32_C(           6),  INT32_C(          80),  INT32_C(         100),  INT32_C(          42),  INT32_C(           3),  INT32_C(          24),  INT32_C(          41),
         INT32_C(          19),  INT32_C(          26),  INT32_C(          23),  INT32_C(          18),  INT32_C(          96),  INT32_C(          38),  INT32_C(          57),  INT32_C(          96) } },
    { {  INT32_C(   116369926), -INT32_C(  1974290821), -INT32_C(  2130849816),  INT32_C(  1681965775), -INT32_C(   811784063),  INT32_C(   211618885),  INT32_C(   857250385), -INT32_C(  1772009073),
         INT32_C(  1855738099), -INT32_C(   285610234), -INT32_C(  1905330498),  INT32_C(   972206008),  INT32_C(   470323159), -INT32_C(   332880229), -INT32_C(   803192512),  INT32_C(  2120647050) },
      {  INT64_C(                  22),  INT64_C(                  23) },
      {  INT32_C(          27),  INT32_C(         553),  INT32_C(         515),  INT32_C(         401),  INT32_C(         830),  INT32_C(          50),  INT32_C(         204),  INT32_C(         601),
         INT32_C(         442),  INT32_C(         955),  INT32_C(         569),  INT32_C(         231),  INT32_C(         112),  INT32_C(         944),  INT32_C(         832),  INT32_C(         505) } },
    { {  INT32_C(  1540871615), -INT32_C(  1136197509), -INT32_C(   879990975), -INT32_C(  1186336024), -INT32_C(   426690828), -INT32_C(   141142245),  INT32_C(  1871369355), -INT32_C(  1020635388),
        -INT32_C(  1474428627),  INT32_C(  1113875969), -INT32_C(  1240534835), -INT32_C(   680568862), -INT32_C(  1447231347), -INT32_C(   476032169),  INT32_C(   726805031), -INT32_C(  1343325054) },
      {  INT64_C(                   8),  INT64_C(                  22) },
      {  INT32_C(     6019029),  INT32_C(    12338944),  INT32_C(    13339751),  INT32_C(    12143090),  INT32_C(    15110454),  INT32_C(    16225879),  INT32_C(     7310036),  INT32_C(    12790359),
         INT32_C(    11017729),  INT32_C(     4351078),  INT32_C(    11931376),  INT32_C(    14118743),  INT32_C(    11123968),  INT32_C(    14917715),  INT32_C(     2839082),  INT32_C(    11529852) } },
    { { -INT32_C(  1101192346), -INT32_C(  1650328202), -INT32_C(  1429671128), -INT32_C(   329664913),  INT32_C(   913027267),  INT32_C(   427240812),  INT32_C(   598240764),  INT32_C(   987238099),
        -INT32_C(  1476905679),  INT32_C(  1531222323), -INT32_C(    83555188), -INT32_C(  2031657278),  INT32_C(  2075939598),  INT32_C(  2090087296),  INT32_C(  1956658337),  INT32_C(  2075031626) },
      {  INT64_C(                  25),  INT64_C(                  21) },
      {  INT32_C(          95),  INT32_C(          78),  INT32_C(          85),  INT32_C(         118),  INT32_C(          27),  INT32_C(          12),  INT32_C(          17),  INT32_C(          29),
         INT32_C(          83),  INT32_C(          45),  INT32_C(         125),  INT32_C(          67),  INT32_C(          61),  INT32_C(          62),  INT32_C(          58),  INT32_C(          61) } },
    { { -INT32_C(  2128054527),  INT32_C(  1274920106),  INT32_C(  1119919608), -INT32_C(  1010995691),  INT32_C(  1420025621), -INT32_C(  1222582459), -INT32_C(  1091836385), -INT32_C(  1788107116),
        -INT32_C(   954821859),  INT32_C(  1192432719), -INT32_C(   947268687),  INT32_C(  1451902529),  INT32_C(  1789603109),  INT32_C(   740412172),  INT32_C(  1189743793), -INT32_C(   807709262) },
      {  INT64_C(                  22),  INT64_C(                  14) },
      {  INT32_C(         516),  INT32_C(         303),  INT32_C(         267),  INT32_C(         782),  INT32_C(         338),  INT32_C(         732),  INT32_C(         763),  INT32_C(         597),
         INT32_C(         796),  INT32_C(         284),  INT32_C(         798),  INT32_C(         346),  INT32_C(         426),  INT32_C(         176),  INT32_C(         283),  INT32_C(         831) } },
    { {  INT32_C(  1121763382),  INT32_C(  1013906827), -INT32_C(  1132308471), -INT32_C(  1786028371),  INT32_C(  1456218704),  INT32_C(  1225607884), -INT32_C(  1643606959), -INT32_C(   904913516),
         INT32_C(  1745743069), -INT32_C(   207324183), -INT32_C(  2119227436), -INT32_C(   719897979),  INT32_C(   690742109),  INT32_C(  2138257454),  INT32_C(  1495169988),  INT32_C(  1965239960) },
      {  INT64_C(                  16),  INT64_C(                   7) },
      {  INT32_C(       17116),  INT32_C(       15470),  INT32_C(       48258),  INT32_C(       38283),  INT32_C(       22220),  INT32_C(       18701),  INT32_C(       40456),  INT32_C(       51728),
         INT32_C(       26637),  INT32_C(       62372),  INT32_C(       33199),  INT32_C(       54551),  INT32_C(       10539),  INT32_C(       32627),  INT32_C(       22814),  INT32_C(       29987) } },
    { {  INT32_C(   711405052),  INT32_C(   715774566), -INT32_C(   310130859),  INT32_C(   291678198), -INT32_C(  2095759401),  INT32_C(  1761807809), -INT32_C(  1802041933),  INT32_C(   433232157),
         INT32_C(   759380679),  INT32_C(  1784147220), -INT32_C(  1437082700),  INT32_C(  1505475202), -INT32_C(  1159867911), -INT32_C(  1859854114),  INT32_C(    52870117),  INT32_C(   454883412) },
      {  INT64_C(                   0),  INT64_C(                  23) },
      {  INT32_C(   711405052),  INT32_C(   715774566), -INT32_C(   310130859),  INT32_C(   291678198), -INT32_C(  2095759401),  INT32_C(  1761807809), -INT32_C(  1802041933),  INT32_C(   433232157),
         INT32_C(   759380679),  INT32_C(  1784147220), -INT32_C(  1437082700),  INT32_C(  1505475202), -INT32_C(  1159867911), -INT32_C(  1859854114),  INT32_C(    52870117),  INT32_C(   454883412) } },
    { {  INT32_C(   376845112), -INT32_C(   106391020), -INT32_C(  1426272683), -INT32_C(   104523322), -INT32_C(   968880519),  INT32_C(   700969390), -INT32_C(  1138330631),  INT32_C(   326663387),
        -INT32_C(  1003819344), -INT32_C(   557985143),  INT32_C(  1720236704),  INT32_C(  1281314515),  INT32_C(   168992604), -INT32_C(  1976313456),  INT32_C(   675699021), -INT32_C(  2059682091) },
      {  INT64_C(                  26),  INT64_C(                   8) },
      {  INT32_C(           5),  INT32_C(          62),  INT32_C(          42),  INT32_C(          62),  INT32_C(          49),  INT32_C(          10),  INT32_C(          47),  INT32_C(           4),
         INT32_C(          49),  INT32_C(          55),  INT32_C(          25),  INT32_C(          19),  INT32_C(           2),  INT32_C(          34),  INT32_C(          10),  INT32_C(          33) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srl_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_srl_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_srl_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int64_t b[2];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1391069190),  INT32_C(    76515171), -INT32_C(  1669093777),  INT32_C(  2052374992), -INT32_C(    17678478), -INT32_C(  1259542507),  INT32_C(   626340292),  INT32_C(  1967227002),
        -INT32_C(   618506633),  INT32_C(  1306505694),  INT32_C(   669606487), -INT32_C(  1784595165), -INT32_C(  1835822212),  INT32_C(  1078361979),  INT32_C(   728079281), -INT32_C(  1818188005) },
      UINT16_C(49917),
      {  INT32_C(  1299438446),  INT32_C(   313510440), -INT32_C(  1840262415), -INT32_C(    98186137), -INT32_C(  1535467170),  INT32_C(  1178545121), -INT32_C(   135439530),  INT32_C(  1555688430),
        -INT32_C(   274125370), -INT32_C(   385787657), -INT32_C(  1854189526),  INT32_C(  2055971100),  INT32_C(   589169986), -INT32_C(  2023137744), -INT32_C(  1484892231),  INT32_C(   134428482) },
      {  INT64_C(                   5),  INT64_C(                  20) },
      {  INT32_C(    40607451),  INT32_C(    76515171),  INT32_C(    76709527),  INT32_C(   131149411),  INT32_C(    86234378),  INT32_C(    36829535),  INT32_C(   129985242),  INT32_C(    48615263),
        -INT32_C(   618506633),  INT32_C(   122161863),  INT32_C(   669606487), -INT32_C(  1784595165), -INT32_C(  1835822212),  INT32_C(  1078361979),  INT32_C(    87814845),  INT32_C(     4200890) } },
    { { -INT32_C(  2134442416),  INT32_C(   352792668), -INT32_C(   910391929),  INT32_C(   567394237),  INT32_C(  1870514539),  INT32_C(   178111169), -INT32_C(   462398333), -INT32_C(   142061401),
        -INT32_C(  1418244530),  INT32_C(    96501630), -INT32_C(  1043432188), -INT32_C(  1461477316),  INT32_C(   706175081), -INT32_C(  1506429661), -INT32_C(  1198873327),  INT32_C(  1135547125) },
      UINT16_C(10081),
      { -INT32_C(  1348018194), -INT32_C(  1288918299),  INT32_C(  1330866284),  INT32_C(   682605584),  INT32_C(   484299495),  INT32_C(    63041144),  INT32_C(  1444263591), -INT32_C(   394430727),
         INT32_C(   999760726),  INT32_C(   988726222),  INT32_C(  1015628331), -INT32_C(   446416642), -INT32_C(  2097069813), -INT32_C(   695811537),  INT32_C(  1915591800),  INT32_C(  1750772242) },
      {  INT64_C(                  27),  INT64_C(                  20) },
      {  INT32_C(          21),  INT32_C(   352792668), -INT32_C(   910391929),  INT32_C(   567394237),  INT32_C(  1870514539),  INT32_C(           0),  INT32_C(          10), -INT32_C(   142061401),
         INT32_C(           7),  INT32_C(           7),  INT32_C(           7), -INT32_C(  1461477316),  INT32_C(   706175081),  INT32_C(          26), -INT32_C(  1198873327),  INT32_C(  1135547125) } },
    { { -INT32_C(   265897536), -INT32_C(   188306308),  INT32_C(  1533473608),  INT32_C(  1824768158),  INT32_C(  1695049649), -INT32_C(   851058951), -INT32_C(   727752643), -INT32_C(  1602791456),
        -INT32_C(  1919902447), -INT32_C(  1837017271), -INT32_C(   387061686),  INT32_C(  1515499688),  INT32_C(   280976407),  INT32_C(  2010973242),  INT32_C(  1162640741), -INT32_C(   437926956) },
      UINT16_C(30304),
      { -INT32_C(   187848334),  INT32_C(   685512507),  INT32_C(  1423541248), -INT32_C(  1632505634), -INT32_C(   559748351), -INT32_C(  1352988829),  INT32_C(   846344268), -INT32_C(  2002202091),
        -INT32_C(  1216580229), -INT32_C(  1931519860), -INT32_C(  1142834980), -INT32_C(  1436970327), -INT32_C(   527893635),  INT32_C(  1334830083),  INT32_C(   696320276),  INT32_C(  1337010643) },
      {  INT64_C(                  12),  INT64_C(                  21) },
      { -INT32_C(   265897536), -INT32_C(   188306308),  INT32_C(  1533473608),  INT32_C(  1824768158),  INT32_C(  1695049649),  INT32_C(      718256),  INT32_C(      206627), -INT32_C(  1602791456),
        -INT32_C(  1919902447),  INT32_C(      577013),  INT32_C(      769563),  INT32_C(  1515499688),  INT32_C(      919695),  INT32_C(      325886),  INT32_C(      170000), -INT32_C(   437926956) } },
    { {  INT32_C(  1955101041),  INT32_C(  1908676701), -INT32_C(   308591335),  INT32_C(   222055535), -INT32_C(    13090182), -INT32_C(   983437273), -INT32_C(  1291026808),  INT32_C(  1717304820),
        -INT32_C(   723852425),  INT32_C(   356949755),  INT32_C(  1392697828), -INT32_C(  1486864851), -INT32_C(  1482188416),  INT32_C(   292358281),  INT32_C(  2076473735), -INT32_C(   287236233) },
      UINT16_C(48131),
      {  INT32_C(   140181186),  INT32_C(   367607315), -INT32_C(   229435503),  INT32_C(  1703662526), -INT32_C(   395438981), -INT32_C(   362679003),  INT32_C(  1376376944), -INT32_C(  1995567930),
         INT32_C(   512845835),  INT32_C(   942963623),  INT32_C(  1344964498), -INT32_C(   692669093), -INT32_C(   272751415), -INT32_C(  1982259431), -INT32_C(  1092885768), -INT32_C(    95950353) },
      {  INT64_C(                  28),  INT64_C(                  27) },
      {  INT32_C(           0),  INT32_C(           1), -INT32_C(   308591335),  INT32_C(   222055535), -INT32_C(    13090182), -INT32_C(   983437273), -INT32_C(  1291026808),  INT32_C(  1717304820),
        -INT32_C(   723852425),  INT32_C(   356949755),  INT32_C(           5),  INT32_C(          13),  INT32_C(          14),  INT32_C(           8),  INT32_C(  2076473735),  INT32_C(          15) } },
    { {  INT32_C(   701154064), -INT32_C(   625761310),  INT32_C(  1956220549), -INT32_C(   898703240),  INT32_C(   230918073), -INT32_C(  1477184301),  INT32_C(  1658202704),  INT32_C(   658365206),
        -INT32_C(  2125461602),  INT32_C(   794493866),  INT32_C(   178582674), -INT32_C(  1898704171),  INT32_C(  1838978969), -INT32_C(   602632309), -INT32_C(   801182791),  INT32_C(  1710717894) },
      UINT16_C(18306),
      {  INT32_C(  1095445734), -INT32_C(    13181605), -INT32_C(  1156445209),  INT32_C(   894610329),  INT32_C(   767941912), -INT32_C(   149389639),  INT32_C(  1165155918),  INT32_C(   697168963),
         INT32_C(  2087442464), -INT32_C(  1669619275),  INT32_C(  1163365804),  INT32_C(  1367058745),  INT32_C(  1082015878),  INT32_C(   221746878), -INT32_C(  1202541963), -INT32_C(  1075650658) },
      {  INT64_C(                  19),  INT64_C(                  17) },
      {  INT32_C(   701154064),  INT32_C(        8166),  INT32_C(  1956220549), -INT32_C(   898703240),  INT32_C(   230918073), -INT32_C(  1477184301),  INT32_C(  1658202704),  INT32_C(        1329),
         INT32_C(        3981),  INT32_C(        5007),  INT32_C(        2218), -INT32_C(  1898704171),  INT32_C(  1838978969), -INT32_C(   602632309),  INT32_C(        5898),  INT32_C(  1710717894) } },
    { {  INT32_C(  1506758042),  INT32_C(  1483081443),  INT32_C(  1326561456),  INT32_C(  1326379928),  INT32_C(   784091456),  INT32_C(  1137231103),  INT32_C(   750823204), -INT32_C(  1665429758),
        -INT32_C(  1074427172),  INT32_C(  1092115345), -INT32_C(  1399838444),  INT32_C(  1559993884), -INT32_C(   410339353), -INT32_C(  1607839108), -INT32_C(    70456327),  INT32_C(  1452836986) },
      UINT16_C(36114),
      {  INT32_C(   770286357),  INT32_C(  1968635365), -INT32_C(  1542163799),  INT32_C(  1482488782),  INT32_C(   229300450),  INT32_C(  1157145720),  INT32_C(   936145567), -INT32_C(   574234680),
         INT32_C(  1728818818),  INT32_C(  1423794603), -INT32_C(  1560743468), -INT32_C(   805612308), -INT32_C(  1529043668), -INT32_C(   370551735),  INT32_C(   237024582),  INT32_C(   401335700) },
      {  INT64_C(                   5),  INT64_C(                   1) },
      {  INT32_C(  1506758042),  INT32_C(    61519855),  INT32_C(  1326561456),  INT32_C(  1326379928),  INT32_C(     7165639),  INT32_C(  1137231103),  INT32_C(   750823204), -INT32_C(  1665429758),
         INT32_C(    54025588),  INT32_C(  1092115345),  INT32_C(    85444494),  INT32_C(   109042343), -INT32_C(   410339353), -INT32_C(  1607839108), -INT32_C(    70456327),  INT32_C(    12541740) } },
    { { -INT32_C(  1196366737), -INT32_C(    22963784), -INT32_C(   485703089),  INT32_C(  1006303143),  INT32_C(  1182366190),  INT32_C(   561122516),  INT32_C(  1985626263),  INT32_C(  2038587914),
        -INT32_C(    30330042),  INT32_C(   469554124),  INT32_C(  1023346837), -INT32_C(   310904321),  INT32_C(  1194586482), -INT32_C(  1805081091), -INT32_C(   267730202),  INT32_C(  1785302308) },
      UINT16_C(39616),
      {  INT32_C(  1684900968), -INT32_C(  1486093656), -INT32_C(  1231000769), -INT32_C(  1935207591), -INT32_C(  1036868518),  INT32_C(  1132730424),  INT32_C(  1909499912),  INT32_C(  2047578130),
        -INT32_C(   992052964), -INT32_C(  1167373701),  INT32_C(   242289845),  INT32_C(  2040207391), -INT32_C(   180630083), -INT32_C(   315047963),  INT32_C(  2036205671), -INT32_C(  1242338920) },
      {  INT64_C(                  26),  INT64_C(                  11) },
      { -INT32_C(  1196366737), -INT32_C(    22963784), -INT32_C(   485703089),  INT32_C(  1006303143),  INT32_C(  1182366190),  INT32_C(   561122516),  INT32_C(          28),  INT32_C(          30),
        -INT32_C(    30330042),  INT32_C(          46),  INT32_C(  1023346837),  INT32_C(          30),  INT32_C(          61), -INT32_C(  1805081091), -INT32_C(   267730202),  INT32_C(          45) } },
    { {  INT32_C(   743752775), -INT32_C(   367424125),  INT32_C(   778270613), -INT32_C(  1008511264),  INT32_C(  1109482535),  INT32_C(   840055105),  INT32_C(  1698886083), -INT32_C(  1295725717),
         INT32_C(   316545167),  INT32_C(  1006434213),  INT32_C(  1332305774), -INT32_C(   602780491), -INT32_C(   367119448),  INT32_C(   790376812), -INT32_C(  1902878942), -INT32_C(  1170188246) },
      UINT16_C( 8049),
      { -INT32_C(   921299252), -INT32_C(  1171749551), -INT32_C(   435757356), -INT32_C(   652628038), -INT32_C(  1257601639),  INT32_C(  1226058933), -INT32_C(    73252934), -INT32_C(   988146695),
         INT32_C(  2072916009), -INT32_C(  1992968267),  INT32_C(  1332690069), -INT32_C(  2077718293), -INT32_C(  1019661810),  INT32_C(   420236895), -INT32_C(  2045464947), -INT32_C(   347394367) },
      {  INT64_C(                   4),  INT64_C(                  20) },
      {  INT32_C(   210854252), -INT32_C(   367424125),  INT32_C(   778270613), -INT32_C(  1008511264),  INT32_C(   189835353),  INT32_C(    76628683),  INT32_C(   263857147), -INT32_C(  1295725717),
         INT32_C(   129557250),  INT32_C(   143874939),  INT32_C(    83293129),  INT32_C(   138578062),  INT32_C(   204706592),  INT32_C(   790376812), -INT32_C(  1902878942), -INT32_C(  1170188246) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_srl_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_srl_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_mask_srl_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int64_t count[2];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(   809092159),  INT32_C(  1250992194),  INT32_C(  1291273030),  INT32_C(   862311519), -INT32_C(   554346528),  INT32_C(  1096845559), -INT32_C(  1387516128),  INT32_C(  2031286330) },
      UINT8_C(188),
      { -INT32_C(   335566517),  INT32_C(  2100447546),  INT32_C(   349994560),  INT32_C(  1005850596), -INT32_C(  1926049275),  INT32_C(  1655600178), -INT32_C(  1197712448), -INT32_C(  1183574675) },
      {  INT64_C(                  18),  INT64_C(                   9) },
      {  INT32_C(   809092159),  INT32_C(  1250992194),  INT32_C(        1335),  INT32_C(        3837),  INT32_C(        9036),  INT32_C(        6315), -INT32_C(  1387516128),  INT32_C(       11869) } },
    { { -INT32_C(   692258652),  INT32_C(  1381366901),  INT32_C(    73480464), -INT32_C(  1479055756), -INT32_C(  1575286284),  INT32_C(   134077291),  INT32_C(   119303059), -INT32_C(   931407068) },
      UINT8_C(211),
      {  INT32_C(   877240119),  INT32_C(   759471092),  INT32_C(  1671514620),  INT32_C(   693585952),  INT32_C(  1050016099),  INT32_C(  1641127158), -INT32_C(  1803167303), -INT32_C(  1956164012) },
      {  INT64_C(                   9),  INT64_C(                   5) },
      {  INT32_C(     1713359),  INT32_C(     1483341),  INT32_C(    73480464), -INT32_C(  1479055756),  INT32_C(     2050812),  INT32_C(   134077291),  INT32_C(     4866796),  INT32_C(     4567975) } },
    { {  INT32_C(    72081855), -INT32_C(  1353889778),  INT32_C(    49835435),  INT32_C(   737893270),  INT32_C(  1808331162), -INT32_C(   666599085),  INT32_C(  2082904085), -INT32_C(   483585244) },
      UINT8_C(244),
      { -INT32_C(  1073551496), -INT32_C(  1586777548),  INT32_C(    87584170),  INT32_C(  1520395112), -INT32_C(  1397945557), -INT32_C(   389970609), -INT32_C(  1089716564), -INT32_C(   474681493) },
      {  INT64_C(                  12),  INT64_C(                   8) },
      {  INT32_C(    72081855), -INT32_C(  1353889778),  INT32_C(       21382),  INT32_C(   737893270),  INT32_C(      707280),  INT32_C(      953368),  INT32_C(      782531),  INT32_C(      932686) } },
    { {  INT32_C(   258476964), -INT32_C(   444788051), -INT32_C(  1219959528), -INT32_C(   322800577), -INT32_C(   495840736),  INT32_C(   102768378), -INT32_C(  1846178851),  INT32_C(   306760814) },
      UINT8_C(216),
      { -INT32_C(  1014685008), -INT32_C(  2065995105),  INT32_C(   650416818),  INT32_C(  1732751445),  INT32_C(  1080109346), -INT32_C(   736204983),  INT32_C(   692236124), -INT32_C(  1476307465) },
      {  INT64_C(                   2),  INT64_C(                  15) },
      {  INT32_C(   258476964), -INT32_C(   444788051), -INT32_C(  1219959528),  INT32_C(   433187861),  INT32_C(   270027336),  INT32_C(   102768378),  INT32_C(   173059031),  INT32_C(   704664957) } },
    { {  INT32_C(  1190139499),  INT32_C(  1591255706),  INT32_C(   269430217), -INT32_C(   180801132), -INT32_C(  1872985486), -INT32_C(  1740654250), -INT32_C(   470992926),  INT32_C(  1248421087) },
      UINT8_C(122),
      { -INT32_C(    82472871), -INT32_C(  1765510296), -INT32_C(  1272261502), -INT32_C(  1893326835),  INT32_C(   904246908),  INT32_C(   739737333), -INT32_C(   804521365), -INT32_C(  1135979165) },
      {  INT64_C(                   5),  INT64_C(                   3) },
      {  INT32_C(  1190139499),  INT32_C(    79045531),  INT32_C(   269430217),  INT32_C(    75051264),  INT32_C(    28257715),  INT32_C(    23116791),  INT32_C(   109076435),  INT32_C(  1248421087) } },
    { {  INT32_C(  2094223032),  INT32_C(   240211428),  INT32_C(   791567881), -INT32_C(   739922962), -INT32_C(   128853024),  INT32_C(   334675207), -INT32_C(   681027955), -INT32_C(   902345198) },
      UINT8_C(156),
      {  INT32_C(  1602242314), -INT32_C(   160854376),  INT32_C(  1759811773),  INT32_C(  1497938046), -INT32_C(   966770679), -INT32_C(  1991019725),  INT32_C(   714812380),  INT32_C(  1824941666) },
      {  INT64_C(                  17),  INT64_C(                   7) },
      {  INT32_C(  2094223032),  INT32_C(   240211428),  INT32_C(       13426),  INT32_C(       11428),  INT32_C(       25392),  INT32_C(   334675207), -INT32_C(   681027955),  INT32_C(       13923) } },
    { {  INT32_C(   886326731),  INT32_C(   550342971),  INT32_C(  1121471481),  INT32_C(    75686308), -INT32_C(    75975256), -INT32_C(   635022274),  INT32_C(  1145079934), -INT32_C(  1064636940) },
      UINT8_C( 50),
      { -INT32_C(   261229474),  INT32_C(   216632769),  INT32_C(  1202727781),  INT32_C(  1676653998), -INT32_C(  2120029651), -INT32_C(  1711309807), -INT32_C(  1450294340),  INT32_C(   735792845) },
      {  INT64_C(                  11),  INT64_C(                   3) },
      {  INT32_C(   886326731),  INT32_C(      105777),  INT32_C(  1121471481),  INT32_C(    75686308),  INT32_C(     1061981),  INT32_C(     1261551),  INT32_C(  1145079934), -INT32_C(  1064636940) } },
    { {  INT32_C(    97911835), -INT32_C(  1053803760),  INT32_C(  1903615618), -INT32_C(   463690942),  INT32_C(   593521956), -INT32_C(  1805247482), -INT32_C(  1595788347),  INT32_C(  2062034270) },
      UINT8_C( 41),
      { -INT32_C(    96895042), -INT32_C(  1904411985),  INT32_C(   332459632), -INT32_C(  1221086134),  INT32_C(   817715732), -INT32_C(   202026304),  INT32_C(  1498584372),  INT32_C(  1015204989) },
      {  INT64_C(                   7),  INT64_C(                  14) },
      {  INT32_C(    32797439), -INT32_C(  1053803760),  INT32_C(  1903615618),  INT32_C(    24014696),  INT32_C(   593521956),  INT32_C(    31976101), -INT32_C(  1595788347),  INT32_C(  2062034270) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_srl_epi32(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_srl_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_mask_srl_epi32(src, k, a, count);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm_mask_srl_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int64_t count[2];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(  1079421588), -INT32_C(  1389563164),  INT32_C(  1678575151), -INT32_C(   344075825) },
      UINT8_C(235),
      { -INT32_C(   261576367),  INT32_C(   751742282), -INT32_C(  1606128557),  INT32_C(  1158430192) },
      {  INT64_C(                  10),  INT64_C(                  13) },
      {  INT32_C(     3938858),  INT32_C(      734123),  INT32_C(  1678575151),  INT32_C(     1131279) } },
    { { -INT32_C(   671557591),  INT32_C(  1826946568), -INT32_C(  1175865907),  INT32_C(   140655520) },
      UINT8_C( 43),
      { -INT32_C(   654723668), -INT32_C(  1457756919), -INT32_C(  1682613930),  INT32_C(   415596863) },
      {  INT64_C(                  11),  INT64_C(                   9) },
      {  INT32_C(     1777462),  INT32_C(     1385356), -INT32_C(  1175865907),  INT32_C(      202928) } },
    { { -INT32_C(  1937737696), -INT32_C(    25872827), -INT32_C(   547311402), -INT32_C(   728494885) },
      UINT8_C(228),
      {  INT32_C(  1174485661),  INT32_C(   737958544), -INT32_C(  1292862054),  INT32_C(  1876069623) },
      {  INT64_C(                   7),  INT64_C(                   3) },
      { -INT32_C(  1937737696), -INT32_C(    25872827),  INT32_C(    23453947), -INT32_C(   728494885) } },
    { { -INT32_C(  1278000204), -INT32_C(   132938727),  INT32_C(   881664095),  INT32_C(    91105796) },
      UINT8_C(113),
      {  INT32_C(   695034110), -INT32_C(  1407522054),  INT32_C(  1350477537), -INT32_C(   167387981) },
      {  INT64_C(                  11),  INT64_C(                  14) },
      {  INT32_C(      339372), -INT32_C(   132938727),  INT32_C(   881664095),  INT32_C(    91105796) } },
    { {  INT32_C(   147534608), -INT32_C(   946476141), -INT32_C(   181026576), -INT32_C(  1638583503) },
      UINT8_C( 92),
      { -INT32_C(    76117937), -INT32_C(  1166467734), -INT32_C(   943761968),  INT32_C(   953644805) },
      {  INT64_C(                  15),  INT64_C(                  11) },
      {  INT32_C(   147534608), -INT32_C(   946476141),  INT32_C(      102270),  INT32_C(       29102) } },
    { { -INT32_C(  1837814069),  INT32_C(   348599582),  INT32_C(    74054246), -INT32_C(   846415529) },
      UINT8_C(180),
      { -INT32_C(  1322325258), -INT32_C(  2106555461), -INT32_C(  1051072376),  INT32_C(  1502361643) },
      {  INT64_C(                   4),  INT64_C(                  15) },
      { -INT32_C(  1837814069),  INT32_C(   348599582),  INT32_C(   202743432), -INT32_C(   846415529) } },
    { { -INT32_C(  1931025800),  INT32_C(  1291182408),  INT32_C(    98151992), -INT32_C(  1846489245) },
      UINT8_C(128),
      {  INT32_C(   770774187),  INT32_C(  1585895837),  INT32_C(  1575665156),  INT32_C(  1322584428) },
      {  INT64_C(                  12),  INT64_C(                   2) },
      { -INT32_C(  1931025800),  INT32_C(  1291182408),  INT32_C(    98151992), -INT32_C(  1846489245) } },
    { { -INT32_C(   480783977),  INT32_C(   247203081), -INT32_C(   100758424), -INT32_C(    49370868) },
      UINT8_C(214),
      {  INT32_C(   157079211),  INT32_C(   955456474), -INT32_C(   720503417), -INT32_C(  1100189715) },
      {  INT64_C(                   5),  INT64_C(                  10) },
      { -INT32_C(   480783977),  INT32_C(    29858014),  INT32_C(   111701996), -INT32_C(    49370868) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_srl_epi32(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_srl_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_mask_srl_epi32(src, k, a, count);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm512_maskz_srl_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int64_t b[2];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(10508),
      { -INT32_C(   652285907),  INT32_C(  1973831532),  INT32_C(  1688648047), -INT32_C(  1906907068),  INT32_C(  2083047916), -INT32_C(  1402768041), -INT32_C(  2074714565), -INT32_C(  1683135890),
        -INT32_C(  1217017014), -INT32_C(  2111038702),  INT32_C(   434557909), -INT32_C(  1482146373),  INT32_C(  1562693638), -INT32_C(  2130081978),  INT32_C(  1392861157),  INT32_C(   217035713) },
      {  INT64_C(                  10),  INT64_C(                   7) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(     1649070),  INT32_C(     2332090),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(     3005810),  INT32_C(           0),  INT32_C(           0),  INT32_C(     2746895),  INT32_C(           0),  INT32_C(     2114145),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(18918),
      { -INT32_C(  1563415399), -INT32_C(  1291733586), -INT32_C(   144260344), -INT32_C(  1839471153), -INT32_C(  1987978378), -INT32_C(  1652407249), -INT32_C(   424429119), -INT32_C(  1892705547),
         INT32_C(  1949433798), -INT32_C(  1121438796), -INT32_C(   978022666), -INT32_C(   380170125),  INT32_C(   443734251), -INT32_C(   558303972), -INT32_C(  1547408466), -INT32_C(   869075963) },
      {  INT64_C(                  17),  INT64_C(                   2) },
      {  INT32_C(           0),  INT32_C(       22912),  INT32_C(       31667),  INT32_C(           0),  INT32_C(           0),  INT32_C(       20161),  INT32_C(       29529),  INT32_C(       18327),
         INT32_C(       14872),  INT32_C(           0),  INT32_C(           0),  INT32_C(       29867),  INT32_C(           0),  INT32_C(           0),  INT32_C(       20962),  INT32_C(           0) } },
    { UINT16_C(50817),
      { -INT32_C(   407134673),  INT32_C(  1079142780),  INT32_C(  1060395021),  INT32_C(  1688414244),  INT32_C(   902642384),  INT32_C(   424592583), -INT32_C(  2101184466), -INT32_C(  1152826228),
         INT32_C(  1587676386),  INT32_C(  2074015086),  INT32_C(  1908069197), -INT32_C(   875208965),  INT32_C(  1610654360), -INT32_C(  1787212186),  INT32_C(  1394031814),  INT32_C(  1645109376) },
      {  INT64_C(                  30),  INT64_C(                  31) },
      {  INT32_C(           3),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           2),
         INT32_C(           0),  INT32_C(           1),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           1) } },
    { UINT16_C(63452),
      { -INT32_C(   666484129), -INT32_C(   283898665),  INT32_C(  1850709087), -INT32_C(  1239436042), -INT32_C(   719993465),  INT32_C(   484329144),  INT32_C(  1783972979), -INT32_C(  1570623165),
         INT32_C(  1266395252),  INT32_C(   339382196),  INT32_C(   444762660), -INT32_C(   959340226),  INT32_C(   513533542), -INT32_C(  1623557844),  INT32_C(    17469374),  INT32_C(   899968193) },
      {  INT64_C(                  26),  INT64_C(                  12) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(          27),  INT32_C(          45),  INT32_C(          53),  INT32_C(           0),  INT32_C(          26),  INT32_C(          40),
         INT32_C(          18),  INT32_C(           5),  INT32_C(           6),  INT32_C(           0),  INT32_C(           7),  INT32_C(          39),  INT32_C(           0),  INT32_C(          13) } },
    { UINT16_C(58788),
      { -INT32_C(  1117859709),  INT32_C(  2051873904), -INT32_C(  1025110498), -INT32_C(  1008600509), -INT32_C(  1619095614),  INT32_C(  1342030690), -INT32_C(  1878131385), -INT32_C(   428494494),
        -INT32_C(   240921471),  INT32_C(   241955056),  INT32_C(  1104171518),  INT32_C(   235254091), -INT32_C(  1548909759), -INT32_C(  1896699321),  INT32_C(   169803687),  INT32_C(   837850288) },
      {  INT64_C(                  16),  INT64_C(                   5) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(       49894),  INT32_C(           0),  INT32_C(           0),  INT32_C(       20477),  INT32_C(           0),  INT32_C(       58997),
         INT32_C(       61859),  INT32_C(           0),  INT32_C(       16848),  INT32_C(           0),  INT32_C(           0),  INT32_C(       36594),  INT32_C(        2590),  INT32_C(       12784) } },
    { UINT16_C(58956),
      { -INT32_C(  1080978483),  INT32_C(  1086208033),  INT32_C(   852782658), -INT32_C(  1027195745), -INT32_C(    78558572), -INT32_C(  1875693108), -INT32_C(  1772544932),  INT32_C(   326936134),
         INT32_C(  1540492601), -INT32_C(  2003070906), -INT32_C(  1648660482),  INT32_C(  1063289259), -INT32_C(  1757695541),  INT32_C(  1042837218), -INT32_C(   791379574),  INT32_C(  1642287399) },
      {  INT64_C(                   8),  INT64_C(                   6) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(     3331182),  INT32_C(    12764732),  INT32_C(           0),  INT32_C(           0),  INT32_C(     9853212),  INT32_C(           0),
         INT32_C(           0),  INT32_C(     8952720),  INT32_C(    10337135),  INT32_C(           0),  INT32_C(           0),  INT32_C(     4073582),  INT32_C(    13685889),  INT32_C(     6415185) } },
    { UINT16_C(60401),
      { -INT32_C(   212216885), -INT32_C(   428481774), -INT32_C(  1774740301), -INT32_C(  1186228483),  INT32_C(  1729138746), -INT32_C(  1269836077), -INT32_C(   118179769), -INT32_C(  1193023764),
        -INT32_C(   877969991), -INT32_C(   726523872), -INT32_C(  1167398467), -INT32_C(  1166756225),  INT32_C(   438404166), -INT32_C(  1563528869),  INT32_C(   379241001),  INT32_C(  1657700008) },
      {  INT64_C(                  10),  INT64_C(                   2) },
      {  INT32_C(     3987060),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(     1688612),  INT32_C(     2954229),  INT32_C(     4078894),  INT32_C(     3029241),
         INT32_C(     3336911),  INT32_C(     3484808),  INT32_C(           0),  INT32_C(     3054893),  INT32_C(           0),  INT32_C(     2667420),  INT32_C(      370352),  INT32_C(     1618847) } },
    { UINT16_C( 9226),
      { -INT32_C(    23763664), -INT32_C(  1564361209), -INT32_C(  1574934060), -INT32_C(   115549237),  INT32_C(  1725478582),  INT32_C(   511746317), -INT32_C(   324775702), -INT32_C(  1760514458),
         INT32_C(   446014739),  INT32_C(   951866980), -INT32_C(  1948525376),  INT32_C(  1854207927), -INT32_C(  1160487507), -INT32_C(   388475650),  INT32_C(   181763236),  INT32_C(  1034020138) },
      {  INT64_C(                   9),  INT64_C(                  20) },
      {  INT32_C(           0),  INT32_C(     5333215),  INT32_C(           0),  INT32_C(     8162925),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(     4582894),  INT32_C(           0),  INT32_C(           0),  INT32_C(     7629866),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_srl_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_srl_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_srl_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int64_t count[2];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(194),
      { -INT32_C(  1633578149),  INT32_C(  1713109013),  INT32_C(  1447468064),  INT32_C(    74267005),  INT32_C(  1720531948),  INT32_C(  1234041688), -INT32_C(  1655177221), -INT32_C(   815845773) },
      {  INT64_C(                  17),  INT64_C(                   8) },
      {  INT32_C(           0),  INT32_C(       13069),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(       20139),  INT32_C(       26543) } },
    { UINT8_C(109),
      {  INT32_C(  1015609814),  INT32_C(  1870799897), -INT32_C(  1495483231),  INT32_C(   238229416), -INT32_C(  1197724049), -INT32_C(   955213015), -INT32_C(   584696306), -INT32_C(   716560385) },
      {  INT64_C(                  17),  INT64_C(                   9) },
      {  INT32_C(        7748),  INT32_C(           0),  INT32_C(       21358),  INT32_C(        1817),  INT32_C(           0),  INT32_C(       25480),  INT32_C(       28307),  INT32_C(           0) } },
    { UINT8_C( 17),
      { -INT32_C(  1466632391),  INT32_C(  1065635993),  INT32_C(   124951544), -INT32_C(  1012686581), -INT32_C(  1848426806),  INT32_C(   716706124), -INT32_C(   445325552),  INT32_C(   972506880) },
      {  INT64_C(                  31),  INT64_C(                  24) },
      {  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(226),
      {  INT32_C(   392616920), -INT32_C(   628555053), -INT32_C(   797103603), -INT32_C(  2036024331),  INT32_C(   842970376), -INT32_C(  1415176506),  INT32_C(  1470295582),  INT32_C(    87633197) },
      {  INT64_C(                   7),  INT64_C(                  32) },
      {  INT32_C(           0),  INT32_C(    28643845),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(    22498365),  INT32_C(    11486684),  INT32_C(      684634) } },
    { UINT8_C( 28),
      { -INT32_C(  1230658852),  INT32_C(  2134167727),  INT32_C(  1260646341),  INT32_C(   696965623),  INT32_C(  1255188639),  INT32_C(   502120134), -INT32_C(   262726933), -INT32_C(    15927517) },
      {  INT64_C(                   8),  INT64_C(                  10) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(     4924399),  INT32_C(     2722521),  INT32_C(     4903080),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(181),
      { -INT32_C(   940998328), -INT32_C(  1830037178), -INT32_C(    98777904), -INT32_C(  1379220381), -INT32_C(  1483044726),  INT32_C(  1677636979), -INT32_C(   680528424), -INT32_C(   695459698) },
      {  INT64_C(                  13),  INT64_C(                   1) },
      {  INT32_C(      409420),  INT32_C(           0),  INT32_C(      512230),  INT32_C(           0),  INT32_C(      343252),  INT32_C(      204789),  INT32_C(           0),  INT32_C(      439393) } },
    { UINT8_C(158),
      {  INT32_C(  2055819240),  INT32_C(   429280543),  INT32_C(  1575180976), -INT32_C(  1929876508),  INT32_C(  1099607518),  INT32_C(  1555071109), -INT32_C(   437726962),  INT32_C(  1535336051) },
      {  INT64_C(                  18),  INT64_C(                  34) },
      {  INT32_C(           0),  INT32_C(        1637),  INT32_C(        6008),  INT32_C(        9022),  INT32_C(        4194),  INT32_C(           0),  INT32_C(           0),  INT32_C(        5856) } },
    { UINT8_C(214),
      { -INT32_C(   429106739),  INT32_C(  1724565257),  INT32_C(   928921003),  INT32_C(  1438780180), -INT32_C(   335134064),  INT32_C(   903206736), -INT32_C(  1514654902), -INT32_C(  1367620383) },
      {  INT64_C(                  21),  INT64_C(                  15) },
      {  INT32_C(           0),  INT32_C(         822),  INT32_C(         442),  INT32_C(           0),  INT32_C(        1888),  INT32_C(           0),  INT32_C(        1325),  INT32_C(        1395) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_srl_epi32(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_srl_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 35.0);
    easysimd__m256i r = easysimd_mm256_maskz_srl_epi32(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm_maskz_srl_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int64_t count[2];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(175),
      {  INT32_C(   209681283),  INT32_C(  1995346808), -INT32_C(   887354311),  INT32_C(  1312181843) },
      {  INT64_C(                  16),  INT64_C(                  15) },
      {  INT32_C(        3199),  INT32_C(       30446),  INT32_C(       51996),  INT32_C(       20022) } },
    { UINT8_C(231),
      {  INT32_C(   493425907),  INT32_C(   862760998), -INT32_C(  1083856368),  INT32_C(   977199406) },
      {  INT64_C(                   7),  INT64_C(                  25) },
      {  INT32_C(     3854889),  INT32_C(     6740320),  INT32_C(    25086804),  INT32_C(           0) } },
    { UINT8_C( 41),
      { -INT32_C(   332067888),  INT32_C(   222005463), -INT32_C(  1118815274),  INT32_C(  1277584431) },
      {  INT64_C(                  25),  INT64_C(                  12) },
      {  INT32_C(         118),  INT32_C(           0),  INT32_C(           0),  INT32_C(          38) } },
    { UINT8_C(184),
      {  INT32_C(   593749950), -INT32_C(   452685402), -INT32_C(   189309237),  INT32_C(   438944302) },
      {  INT64_C(                  23),  INT64_C(                  13) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(          52) } },
    { UINT8_C( 85),
      { -INT32_C(   141457242),  INT32_C(  1817952325), -INT32_C(   985405940), -INT32_C(   936891740) },
      {  INT64_C(                  17),  INT64_C(                   9) },
      {  INT32_C(       31688),  INT32_C(           0),  INT32_C(       25249),  INT32_C(           0) } },
    { UINT8_C(204),
      {  INT32_C(  1814725813), -INT32_C(  1573955463), -INT32_C(   933967757),  INT32_C(  1717165167) },
      {  INT64_C(                   0),  INT64_C(                  13) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   933967757),  INT32_C(  1717165167) } },
    { UINT8_C(193),
      { -INT32_C(   794286451),  INT32_C(   335170795), -INT32_C(   506672364), -INT32_C(   318025600) },
      {  INT64_C(                   3),  INT64_C(                   3) },
      {  INT32_C(   437585105),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_srl_epi32(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_srl_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 7 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 35.0);
    easysimd__m128i r = easysimd_mm_maskz_srl_epi32(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm512_srl_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[2];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 4503272731429987359), -INT64_C( 2015384591523464281),  INT64_C( 8370329603485121851),  INT64_C( 8741546637051844249),
         INT64_C( 7244483912508171930),  INT64_C( 2420618720360332110),  INT64_C( 4422947859294142848),  INT64_C( 3359849781269978573) },
      {  INT64_C(                  37),  INT64_C(                  38) },
      {  INT64_C(            32765621),  INT64_C(           119553875),  INT64_C(            60902163),  INT64_C(            63603122),
         INT64_C(            52710557),  INT64_C(            17612319),  INT64_C(            32181181),  INT64_C(            24446124) } },
    { { -INT64_C( 7637003490261708254),  INT64_C( 6415480880923232326),  INT64_C( 9218637000122400785),  INT64_C( 4196059733781788805),
        -INT64_C(  930270758595148872),  INT64_C( 4864966429638513740), -INT64_C( 4010865298925415083), -INT64_C( 2947988449335396157) },
      {  INT64_C(                  25),  INT64_C(                  63) },
      {  INT64_C(        322155373795),  INT64_C(        191196229485),  INT64_C(        274736791852),  INT64_C(        125052324944),
         INT64_C(        522031584832),  INT64_C(        144987297941),  INT64_C(        430222713195),  INT64_C(        461898911725) } },
    { { -INT64_C( 6686951679321549528), -INT64_C( 6351318751585002523), -INT64_C( 1572949627593083733), -INT64_C(  631318895338144874),
         INT64_C( 2734693502203073391), -INT64_C( 5541956386463059407),  INT64_C( 3082894904229679673), -INT64_C( 5609612383155050401) },
      {  INT64_C(                  19),  INT64_C(                  33) },
      {  INT64_C(      22430023945594),  INT64_C(      23070192951439),  INT64_C(      32184208767159),  INT64_C(      33980226856940),
         INT64_C(       5216013912588),  INT64_C(      24613929152005),  INT64_C(       5880155380687),  INT64_C(      24484885579213) } },
    { { -INT64_C( 7032322167997686871),  INT64_C( 3695752672797643348), -INT64_C( 6275681140897592453),  INT64_C( 2911922955017037974),
         INT64_C( 5599047430447746517), -INT64_C( 1275313147080310623),  INT64_C( 6872898518047554424), -INT64_C( 8520621702864799464) },
      {  INT64_C(                  37),  INT64_C(                  18) },
      {  INT64_C(            83050849),  INT64_C(            26890139),  INT64_C(            88556138),  INT64_C(            21187027),
         INT64_C(            40738431),  INT64_C(           124938603),  INT64_C(            50006918),  INT64_C(            72222045) } },
    { {  INT64_C( 4873285309233805945),  INT64_C(  439834950256813021), -INT64_C( 2716455584873858879),  INT64_C( 8498945592412536537),
         INT64_C( 3853490666578965515), -INT64_C( 5633192859071523523),  INT64_C( 6376453306728335769), -INT64_C(  565776880762709068) },
      {  INT64_C(                  29),  INT64_C(                  45) },
      {  INT64_C(          9077201242),  INT64_C(           819256436),  INT64_C(         29299945549),  INT64_C(         15830519781),
         INT64_C(          7177685697),  INT64_C(         23867099014),  INT64_C(         11877069821),  INT64_C(         33305896805) } },
    { {  INT64_C( 8157098361486837595),  INT64_C( 1321040308684797296),  INT64_C( 3012847868082094884),  INT64_C( 8923801435785072389),
         INT64_C( 8497317249283403709),  INT64_C( 8789380872681950910),  INT64_C( 2017793055357488554), -INT64_C( 8480534500030408781) },
      {  INT64_C(                  59),  INT64_C(                  24) },
      {  INT64_C(                  14),  INT64_C(                   2),  INT64_C(                   5),  INT64_C(                  15),
         INT64_C(                  14),  INT64_C(                  15),  INT64_C(                   3),  INT64_C(                  17) } },
    { { -INT64_C(  337988464786431205), -INT64_C( 1610283556321391280), -INT64_C( 6926550933078561340), -INT64_C( 7089296378098830461),
         INT64_C( 6893073757373453926),  INT64_C( 6645101486772353452), -INT64_C( 2301568806200316975),  INT64_C( 3098962276046799842) },
      {  INT64_C(                  38),  INT64_C(                  28) },
      {  INT64_C(            65879269),  INT64_C(            61250686),  INT64_C(            41910218),  INT64_C(            41318154),
         INT64_C(            25076856),  INT64_C(            24174738),  INT64_C(            58735805),  INT64_C(            11273959) } },
    { {  INT64_C( 5894349276315664526),  INT64_C( 3522860139222001906),  INT64_C( 4256409894813416583), -INT64_C( 5955799268209984695),
         INT64_C( 1855594266527086592), -INT64_C( 5539512011075848627),  INT64_C(  426499171783667312),  INT64_C( 8791384339317111350) },
      {  INT64_C(                  19),  INT64_C(                  44) },
      {  INT64_C(      11242579033500),  INT64_C(       6719322470134),  INT64_C(       8118457593561),  INT64_C(      23824586497305),
         INT64_C(       3539265187315),  INT64_C(      24618591428058),  INT64_C(        813482612197),  INT64_C(      16768234900125) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srl_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_srl_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_srl_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[2];
    const int64_t r[8];
  } test_vec[] = {
{ {  INT64_C( 5018051879103075663), -INT64_C(  660831149718276767), -INT64_C( 4444598874757254777), -INT64_C( 3636321904948482748),
         INT64_C(  694495148200162758), -INT64_C(  912842675246716212),  INT64_C( 3729394938595987999),  INT64_C( 5382103391663641445) },
      UINT8_C(  7),
      {  INT64_C( 8469222834209532005), -INT64_C( 8151281972419631149), -INT64_C(  875474453646156037), -INT64_C( 3146959030850175687),
        -INT64_C( 8202998051811360151), -INT64_C( 2609501873265243137),  INT64_C( 3431787955991037454),  INT64_C( 4695510415541147622) },
      {  INT64_C(                  52),  INT64_C(                  41) },
      {  INT64_C(                1880),  INT64_C(                2286),  INT64_C(                3901), -INT64_C( 3636321904948482748),
         INT64_C(  694495148200162758), -INT64_C(  912842675246716212),  INT64_C( 3729394938595987999),  INT64_C( 5382103391663641445) } },
    { {  INT64_C( 4774085380789252891),  INT64_C( 6351976018652674572),  INT64_C( 7285695253303204985),  INT64_C( 1704423325787105127),
        -INT64_C(  205924299156423802), -INT64_C(  863837376267240669),  INT64_C( 5459588585176950308),  INT64_C(  907179034420130690) },
      UINT8_C(237),
      { -INT64_C( 3509752084262096962), -INT64_C( 2384022715044792899),  INT64_C( 4709090717786589566),  INT64_C( 3258586805351995736),
        -INT64_C( 5862104578432690494), -INT64_C( 6022899097165107309), -INT64_C( 7856847056654750553),  INT64_C( 6004729387056741816) },
      {  INT64_C(                  54),  INT64_C(                  16) },
      {  INT64_C(                 829),  INT64_C( 6351976018652674572),  INT64_C(                 261),  INT64_C(                 180),
        -INT64_C(  205924299156423802),  INT64_C(                 689),  INT64_C(                 587),  INT64_C(                 333) } },
    { { -INT64_C( 5347199935727591939), -INT64_C( 1761696095209150550),  INT64_C( 4827803292971284187),  INT64_C( 3812233551889196554),
         INT64_C( 8473019998232436452), -INT64_C( 9115217864340507683),  INT64_C( 6537717375098455028), -INT64_C( 7821090437979903991) },
      UINT8_C(198),
      {  INT64_C( 9147944483022796001), -INT64_C( 8004967957813555785),  INT64_C(  357537112035189065), -INT64_C( 8293301012967246278),
         INT64_C( 3981911599859595706),  INT64_C( 6035127009660397092),  INT64_C( 1627267161321939317),  INT64_C( 5234344461575145580) },
      {  INT64_C(                   1),  INT64_C(                   2) },
      { -INT64_C( 5347199935727591939),  INT64_C( 5220888057947997915),  INT64_C(  178768556017594532),  INT64_C( 3812233551889196554),
         INT64_C( 8473019998232436452), -INT64_C( 9115217864340507683),  INT64_C(  813633580660969658),  INT64_C( 2617172230787572790) } },
    { { -INT64_C(  753457344752823400),  INT64_C( 9100845639399228278),  INT64_C( 6691762627973503226), -INT64_C( 6772517690425994392),
         INT64_C( 4611865007080578178), -INT64_C( 4987858906752217499), -INT64_C( 5481882081349639960), -INT64_C( 5102383525635684220) },
      UINT8_C( 77),
      { -INT64_C( 3220258701015792569), -INT64_C( 4606252918426593125), -INT64_C( 7433827182372543520), -INT64_C( 2636424837617130645),
         INT64_C( 8279327878949470286), -INT64_C( 6427567103991252564),  INT64_C(   35410411469199188),  INT64_C( 2639257638627200305) },
      {  INT64_C(                   9),  INT64_C(                  60) },
      {  INT64_C(   29739229243542498),  INT64_C( 9100845639399228278),  INT64_C(   21509603303392593),  INT64_C(   30879529757993009),
         INT64_C( 4611865007080578178), -INT64_C( 4987858906752217499),  INT64_C(      69160959900779), -INT64_C( 5102383525635684220) } },
    { { -INT64_C(  474472491459740714),  INT64_C( 1413390369529169826), -INT64_C( 8618690178674731836),  INT64_C( 7765024437675857456),
        -INT64_C( 7951878168928412957), -INT64_C( 8653482228769463353),  INT64_C( 5473607594897955195), -INT64_C( 6430896436868883521) },
      UINT8_C(119),
      {  INT64_C( 6529930248793313686),  INT64_C( 3757589078185112322),  INT64_C( 7156672293724263016), -INT64_C(  213133291419162954),
         INT64_C( 7813308203475955998), -INT64_C( 7533049731877532699),  INT64_C( 1637473394195219917), -INT64_C( 3854038203012358113) },
      {  INT64_C(                  43),  INT64_C(                   6) },
      {  INT64_C(              742367),  INT64_C(              427188),  INT64_C(              813619),  INT64_C( 7765024437675857456),
         INT64_C(              888270),  INT64_C(             1240743),  INT64_C(              186159), -INT64_C( 6430896436868883521) } },
    { {  INT64_C( 6890553886565778985),  INT64_C( 2558441506522614282), -INT64_C( 8775070714056974145), -INT64_C( 5393012673236621375),
        -INT64_C( 3633876295361783311), -INT64_C( 7040528235959716310),  INT64_C( 7969224737570684516),  INT64_C( 4740220371150316935) },
      UINT8_C( 62),
      {  INT64_C( 8796319542675600136), -INT64_C( 8882514063471855023), -INT64_C( 8572988487162016737), -INT64_C( 1585517887377303268),
         INT64_C( 2309471185300160654),  INT64_C( 9018229961860212253),  INT64_C( 1579359360142563581),  INT64_C( 7919253252302357531) },
      {  INT64_C(                  46),  INT64_C(                  26) },
      {  INT64_C( 6890553886565778985),  INT64_C(              135915),  INT64_C(              140314),  INT64_C(              239612),
         INT64_C(               32819),  INT64_C(              128156),  INT64_C( 7969224737570684516),  INT64_C( 4740220371150316935) } },
    { {  INT64_C( 7559772791110257371),  INT64_C( 7838035306306977657),  INT64_C( 4924922392629855747),  INT64_C( 1361601353282270225),
        -INT64_C( 2494766160983707733),  INT64_C( 1481962523173766543), -INT64_C( 8585818606364548669), -INT64_C(  429836139557950017) },
      UINT8_C(153),
      {  INT64_C( 1679238285005550319), -INT64_C( 6570405501343403503),  INT64_C( 3222506128052807406), -INT64_C( 6472670232636117335),
         INT64_C( 1499612929985882470),  INT64_C( 2435025945260914468),  INT64_C( 3730338983580460633),  INT64_C( 1936251741381476225) },
      {  INT64_C(                   6),  INT64_C(                  40) },
      {  INT64_C(   26238098203211723),  INT64_C( 7838035306306977657),  INT64_C( 4924922392629855747),  INT64_C(  187094903766772410),
         INT64_C(   23431452031029413),  INT64_C( 1481962523173766543), -INT64_C( 8585818606364548669),  INT64_C(   30253933459085566) } },
    { {  INT64_C( 6606217542241028080), -INT64_C( 9219868554947169474),  INT64_C( 1856557821955333813), -INT64_C(  210876376874155225),
        -INT64_C( 8426757936638374440), -INT64_C( 3856056457707543913),  INT64_C( 1369719826837412456), -INT64_C( 7978806546093840581) },
      UINT8_C(227),
      { -INT64_C(  903047239936490564),  INT64_C(  660572222557549369), -INT64_C(  236148617042914490),  INT64_C(  145317255498170763),
         INT64_C( 2940386493423826057), -INT64_C( 1241562307819219252),  INT64_C( 5636579028621965864), -INT64_C( 4867983057477439960) },
      {  INT64_C(                  30),  INT64_C(                  58) },
      {  INT64_C(         16338840903),  INT64_C(           615205823),  INT64_C( 1856557821955333813), -INT64_C(  210876376874155225),
        -INT64_C( 8426757936638374440),  INT64_C(         16023574178),  INT64_C(          5249473292),  INT64_C(         12646206669) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_srl_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_srl_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_mask_srl_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t count[2];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 8040723512987171602), -INT64_C(  593549223450704682), -INT64_C( 4218633353895709439), -INT64_C( 7250084525844116037) },
      UINT8_C(193),
      {  INT64_C( 3411370247644550173), -INT64_C( 3283890086708536798),  INT64_C( 1069860823303867573), -INT64_C( 6507364422857054871) },
      {  INT64_C(                  53),  INT64_C(                  34) },
      {  INT64_C(                 378), -INT64_C(  593549223450704682), -INT64_C( 4218633353895709439), -INT64_C( 7250084525844116037) } },
    { {  INT64_C( 8005046222614229286),  INT64_C( 7622742898608862223),  INT64_C( 3424630829787220616), -INT64_C( 2793312918079205718) },
      UINT8_C( 25),
      { -INT64_C( 2130298911003683139),  INT64_C( 4635209755284597855),  INT64_C( 5208702392401966308), -INT64_C( 7013839779876067992) },
      {  INT64_C(                   4),  INT64_C(                   2) },
      {  INT64_C( 1019777822669116779),  INT64_C( 7622742898608862223),  INT64_C( 3424630829787220616),  INT64_C(  714556518364592726) } },
    { { -INT64_C( 6603708087728891394),  INT64_C( 2935249988176187573),  INT64_C( 4226795642396079801),  INT64_C( 2788918016246516008) },
      UINT8_C(211),
      { -INT64_C(    9303040204041153), -INT64_C( 7278528780156475607),  INT64_C( 3560116019288080552),  INT64_C( 3328020415184976417) },
      {  INT64_C(                  41),  INT64_C(                  63) },
      {  INT64_C(             8384377),  INT64_C(             5078716),  INT64_C( 4226795642396079801),  INT64_C( 2788918016246516008) } },
    { {  INT64_C( 5989111318549889880),  INT64_C( 3736425679207252445),  INT64_C(  961069589461369523), -INT64_C( 3715010693929548349) },
      UINT8_C(165),
      { -INT64_C( 2208739500293094687), -INT64_C( 8147734052536044858), -INT64_C( 6494104487953626699),  INT64_C( 2748010804968805339) },
      {  INT64_C(                  45),  INT64_C(                  26) },
      {  INT64_C(              461511),  INT64_C( 3736425679207252445),  INT64_C(              339714), -INT64_C( 3715010693929548349) } },
    { {  INT64_C( 2678388240435972257),  INT64_C(  485345072831018949), -INT64_C( 2658477544053597081),  INT64_C( 2781440637059293101) },
      UINT8_C(177),
      { -INT64_C( 9101845993408028834),  INT64_C( 4252608746322551379), -INT64_C( 4143072617236181854),  INT64_C( 5184895417192304538) },
      {  INT64_C(                  58),  INT64_C(                  69) },
      {  INT64_C(                  32),  INT64_C(  485345072831018949), -INT64_C( 2658477544053597081),  INT64_C( 2781440637059293101) } },
    { { -INT64_C( 6610263498682409935), -INT64_C( 7237685243200576962),  INT64_C(  591395130552153446), -INT64_C( 7506642520504656303) },
      UINT8_C(233),
      {  INT64_C( 2233763136128120864), -INT64_C( 4191064750953840163),  INT64_C( 1165458079739919564),  INT64_C( 7419977550861332996) },
      {  INT64_C(                   9),  INT64_C(                  43) },
      {  INT64_C(    4362818625250236), -INT64_C( 7237685243200576962),  INT64_C(  591395130552153446),  INT64_C(   14492143654026041) } },
    { {  INT64_C( 2118672462558020135),  INT64_C( 9082708204243466105), -INT64_C( 8810434986230874595), -INT64_C( 8190329842263981796) },
      UINT8_C( 92),
      {  INT64_C( 5982428591386637442),  INT64_C( 6967014067027334480), -INT64_C( 5332274376053098026), -INT64_C( 1225354669340224120) },
      {  INT64_C(                  57),  INT64_C(                  61) },
      {  INT64_C( 2118672462558020135),  INT64_C( 9082708204243466105),  INT64_C(                  90),  INT64_C(                 119) } },
    { {  INT64_C( 2149316316964027001),  INT64_C(  298014471269070202), -INT64_C( 6217077944050198620), -INT64_C( 8368336741646261214) },
      UINT8_C( 44),
      {  INT64_C( 1357016002708703860), -INT64_C(  546999193220677887), -INT64_C( 1709346113602647132), -INT64_C( 5089202521634084735) },
      {  INT64_C(                  17),  INT64_C(                   6) },
      {  INT64_C( 2149316316964027001),  INT64_C(  298014471269070202),  INT64_C(     127696212464194),  INT64_C(     101909954468349) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_srl_epi64(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_srl_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 70.0);
    easysimd__m256i r = easysimd_mm256_mask_srl_epi64(src, k, a, count);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm_mask_srl_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t count[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 1484204217748110561),  INT64_C( 5261286556150015423) },
      UINT8_C(169),
      { -INT64_C( 5768892405199714598), -INT64_C( 4445760921229368725) },
      {  INT64_C(                  21),  INT64_C(                  56) },
      {  INT64_C(       6045270761732),  INT64_C( 5261286556150015423) } },
    { { -INT64_C( 7806564151838472990), -INT64_C( 3868187274754227518) },
      UINT8_C(144),
      {  INT64_C( 3079586920670531901),  INT64_C( 3713058118464649186) },
      {  INT64_C(                  19),  INT64_C(                  40) },
      { -INT64_C( 7806564151838472990), -INT64_C( 3868187274754227518) } },
    { {  INT64_C( 2062507965029437302),  INT64_C( 2308002035425742763) },
      UINT8_C(152),
      { -INT64_C( 8384421686026173693),  INT64_C( 3306882231423865403) },
      {  INT64_C(                  49),  INT64_C(                  66) },
      {  INT64_C( 2062507965029437302),  INT64_C( 2308002035425742763) } },
    { {  INT64_C(  633208275525756478),  INT64_C( 7016065118061425722) },
      UINT8_C( 64),
      { -INT64_C( 7574219868625113788), -INT64_C( 7619695093158884907) },
      {  INT64_C(                  65),  INT64_C(                  58) },
      {  INT64_C(  633208275525756478),  INT64_C( 7016065118061425722) } },
    { {  INT64_C( 2612046866533420584),  INT64_C( 8518215172018557415) },
      UINT8_C(150),
      {  INT64_C( 5003431303873850718),  INT64_C( 6426394543897696113) },
      {  INT64_C(                  49),  INT64_C(                  62) },
      {  INT64_C( 2612046866533420584),  INT64_C(               11415) } },
    { { -INT64_C( 7869776147026910430), -INT64_C( 7033453669726515327) },
      UINT8_C( 28),
      { -INT64_C( 2777767887106252633), -INT64_C( 8959796969756584147) },
      {  INT64_C(                   4),  INT64_C(                  26) },
      { -INT64_C( 7869776147026910430), -INT64_C( 7033453669726515327) } },
    { {  INT64_C( 8081787014132201421), -INT64_C(  690322418197835924) },
      UINT8_C(104),
      {  INT64_C(  873542772967823677), -INT64_C( 8727234392462423918) },
      {  INT64_C(                  41),  INT64_C(                  14) },
      {  INT64_C( 8081787014132201421), -INT64_C(  690322418197835924) } },
    { {  INT64_C( 1532271752689393406),  INT64_C(  176692781748633124) },
      UINT8_C(211),
      {  INT64_C( 2069820773848282361),  INT64_C( 5948764401585576714) },
      {  INT64_C(                  27),  INT64_C(                  14) },
      {  INT64_C(         15421366496),  INT64_C(         44321748626) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_srl_epi64(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_srl_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 70.0);
    easysimd__m128i r = easysimd_mm_mask_srl_epi64(src, k, a, count);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm512_maskz_srl_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[2];
    const int64_t r[8];
  } test_vec[] = {
{ UINT8_C(  2),
      {  INT64_C( 1273292168187332866),  INT64_C( 7422134831920816881),  INT64_C( 8778785881423789008), -INT64_C( 6870649568514933397),
         INT64_C( 5972440929581448533),  INT64_C( 2069270216126665473),  INT64_C( 2304214308246073665),  INT64_C( 6084761119011074867) },
      {  INT64_C(                  33),  INT64_C(                  10) },
      {  INT64_C(                   0),  INT64_C(           864050215),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(190),
      {  INT64_C( 1715469743755078245),  INT64_C( 3883908348811583318), -INT64_C( 6110753254294327408), -INT64_C( 9052310990052255468),
        -INT64_C( 4962707970185112987),  INT64_C( 3589194774746904202), -INT64_C( 2682461174757816112),  INT64_C( 8697536180863588116) },
      {  INT64_C(                  17),  INT64_C(                  30) },
      {  INT64_C(                   0),  INT64_C(      29631869116299),  INT64_C(      94116140895196),  INT64_C(      71673836392649),
         INT64_C(     102875031307406),  INT64_C(      27383382986045),  INT64_C(                   0),  INT64_C(      66356934973629) } },
    { UINT8_C( 54),
      { -INT64_C( 5799555120303860877),  INT64_C( 4096933604670747731),  INT64_C(  231998857741135549), -INT64_C( 3315171190985936359),
         INT64_C( 2720388382403328952),  INT64_C( 8797552889633669751),  INT64_C( 6835358425641782045),  INT64_C( 7418312242013505733) },
      {  INT64_C(                  27),  INT64_C(                  45) },
      {  INT64_C(                   0),  INT64_C(         30524534021),  INT64_C(          1728526187),  INT64_C(                   0),
         INT64_C(         20268472898),  INT64_C(         65546876859),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(110),
      { -INT64_C(  890849150588711221), -INT64_C( 1407084915905673929), -INT64_C( 1553937031705302402), -INT64_C( 6671174823202003303),
         INT64_C( 7432386549469224648),  INT64_C( 5516411252743718633),  INT64_C( 6273357867429032100), -INT64_C( 7723181002292161705) },
      {  INT64_C(                  24),  INT64_C(                   5) },
      {  INT64_C(                   0),  INT64_C(       1015642831194),  INT64_C(       1006889762997),  INT64_C(        701878622204),
         INT64_C(                   0),  INT64_C(        328803733154),  INT64_C(        373921267237),  INT64_C(                   0) } },
    { UINT8_C(  1),
      { -INT64_C( 2551546160007398705), -INT64_C( 1583229024549091734), -INT64_C( 4637370017592110076), -INT64_C( 4443569649745975869),
        -INT64_C( 7663831157342858114), -INT64_C( 8078959051578389152), -INT64_C( 4241788306351701286),  INT64_C( 1512768116375507273) },
      {  INT64_C(                  62),  INT64_C(                  40) },
      {  INT64_C(                   3),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 21),
      {  INT64_C( 8563556369835754345),  INT64_C( 1362375245873026039), -INT64_C( 2932503238363277827), -INT64_C(  467743725081221479),
         INT64_C( 7988363705945922112),  INT64_C(  546549678847879282), -INT64_C( 9007248830868667823),  INT64_C( 1945960595755801977) },
      {  INT64_C(                  22),  INT64_C(                  52) },
      {  INT64_C(       2041710941752),  INT64_C(                   0),  INT64_C(       3698883255802),  INT64_C(                   0),
         INT64_C(       1904574324118),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(180),
      { -INT64_C( 7627373761685227045), -INT64_C( 5077655539464261875), -INT64_C( 4634773583306946036),  INT64_C( 3241203397608668373),
        -INT64_C( 4208050115763857397), -INT64_C( 6766153118426920401), -INT64_C( 8678233403792115242), -INT64_C( 1354797465776172876) },
      {  INT64_C(                  15),  INT64_C(                  49) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(     421507888501056),  INT64_C(                   0),
         INT64_C(     434530455259573),  INT64_C(     356463347024006),  INT64_C(                   0),  INT64_C(     521604815915935) } },
    { UINT8_C(110),
      { -INT64_C(  701251862189271592), -INT64_C( 8672589849232389770), -INT64_C( 4551220995287630154), -INT64_C( 6372748713757113748),
        -INT64_C( 3664950061693423865),  INT64_C( 8559287630595816389), -INT64_C( 7435569239400424336), -INT64_C(  646749299245744753) },
      {  INT64_C(                  31),  INT64_C(                  18) },
      {  INT64_C(                   0),  INT64_C(          4551445238),  INT64_C(          6470607164),  INT64_C(          5622392222),
         INT64_C(                   0),  INT64_C(          3985728896),  INT64_C(          5127477848),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_srl_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_srl_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_srl_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t count[2];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(199),
      {  INT64_C( 4692793873002752211),  INT64_C( 6771537674682483862), -INT64_C( 8379700819695471080), -INT64_C( 8911417839542439422) },
      {  INT64_C(                  32),  INT64_C(                  22) },
      {  INT64_C(          1092626217),  INT64_C(          1576621475),  INT64_C(          2343916160),  INT64_C(                   0) } },
    { UINT8_C(128),
      { -INT64_C( 2829838888697007910), -INT64_C( 5880495875111645433),  INT64_C( 5067348622218409929),  INT64_C( 2777422773467388807) },
      {  INT64_C(                  64),  INT64_C(                  60) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 65),
      { -INT64_C(  395179224937558055), -INT64_C( 1565820462454289358), -INT64_C( 5931285093082719281),  INT64_C( 4578581257846788643) },
      {  INT64_C(                  40),  INT64_C(                  29) },
      {  INT64_C(            16417802),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(241),
      { -INT64_C( 6637670597105708920), -INT64_C( 2450797366221408041),  INT64_C( 4795383592161957286),  INT64_C( 7724890392600834052) },
      {  INT64_C(                   6),  INT64_C(                  23) },
      {  INT64_C(  184516773071935042),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(238),
      { -INT64_C( 8993467719088018524),  INT64_C( 3456888589890106511), -INT64_C( 2279758622424852402),  INT64_C( 3884282733493792398) },
      {  INT64_C(                  27),  INT64_C(                  27) },
      {  INT64_C(                   0),  INT64_C(         25755827053),  INT64_C(        120453428114),  INT64_C(         28940161567) } },
    { UINT8_C(125),
      {  INT64_C( 8894294867865075563), -INT64_C( 6726394243038526275), -INT64_C( 2693805817920072792),  INT64_C( 8065797418144324413) },
      {  INT64_C(                   2),  INT64_C(                   0) },
      {  INT64_C( 2223573716966268890),  INT64_C(                   0),  INT64_C( 3938234563947369706),  INT64_C( 2016449354536081103) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_srl_epi64(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_srl_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 6 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 70.0);
    easysimd__m256i r = easysimd_mm256_maskz_srl_epi64(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}


static int
test_easysimd_mm_maskz_srl_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t count[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 53),
      {  INT64_C( 4365535288225115348), -INT64_C( 6048896640280012144) },
      {  INT64_C(                  56),  INT64_C(                  10) },
      {  INT64_C(                  60),  INT64_C(                   0) } },
    { UINT8_C(106),
      {  INT64_C( 6731434387453861997),  INT64_C( 6989382472892579763) },
      {  INT64_C(                  42),  INT64_C(                  54) },
      {  INT64_C(                   0),  INT64_C(             1589201) } },
    { UINT8_C(246),
      { -INT64_C(  664035853662679367), -INT64_C( 9079710547833524598) },
      {  INT64_C(                  57),  INT64_C(                  15) },
      {  INT64_C(                   0),  INT64_C(                  64) } },
    { UINT8_C(235),
      {  INT64_C( 8361985446647598250),  INT64_C(  748724821381318686) },
      {  INT64_C(                   1),  INT64_C(                  61) },
      {  INT64_C( 4180992723323799125),  INT64_C(  374362410690659343) } },
    { UINT8_C(210),
      { -INT64_C( 4591950047175651872),  INT64_C(  212555653160014750) },
      {  INT64_C(                  27),  INT64_C(                  48) },
      {  INT64_C(                   0),  INT64_C(          1583663025) } },
    { UINT8_C( 13),
      {  INT64_C( 5117452442322456080), -INT64_C( 5659521910928970015) },
      {  INT64_C(                  21),  INT64_C(                  17) },
      {  INT64_C(       2440191479836),  INT64_C(                   0) } },
    { UINT8_C(247),
      { -INT64_C( 6009551528916209447),  INT64_C( 5218875122158155210) },
      {  INT64_C(                   3),  INT64_C(                  34) },
      {  INT64_C( 1554649068099167771),  INT64_C(  652359390269769401) } },
    { UINT8_C( 76),
      {  INT64_C( 5707196363909506644), -INT64_C( 7405024829413175879) },
      {  INT64_C(                  29),  INT64_C(                   1) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_srl_epi64(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_srl_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 70.0);
    easysimd__m128i r = easysimd_mm_maskz_srl_epi64(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_bsrli_epi128 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  static const struct {
    const int8_t a[64];
    const int8_t r[64];
  } test_vec[] = {
    { { -INT8_C( 116), -INT8_C( 109), -INT8_C(  74),  INT8_C(  56),  INT8_C( 119),  INT8_C(  61),  INT8_C(  19), -INT8_C( 107),
         INT8_C(  48), -INT8_C(  25), -INT8_C(  99), -INT8_C( 103), -INT8_C( 118),  INT8_C(  62),  INT8_C(  92),  INT8_C(  53),
        -INT8_C(  79),  INT8_C(  63), -INT8_C(   5), -INT8_C(   7),  INT8_C(  35), -INT8_C(   3),  INT8_C(  53), -INT8_C(  30),
        -INT8_C(  99), -INT8_C( 108), -INT8_C(  14),  INT8_C(   8), -INT8_C(  58),  INT8_C( 106), -INT8_C(  76),  INT8_C(  83),
        -INT8_C(   3),  INT8_C( 106), -INT8_C( 117),  INT8_C( 116), -INT8_C(  89), -INT8_C(  98),  INT8_C(   9), -INT8_C(  40),
        -INT8_C( 122), -INT8_C(  89),  INT8_C( 113),  INT8_C(  16), -INT8_C(  27), -INT8_C(  51),  INT8_C(  69), -INT8_C( 106),
         INT8_C(  12),  INT8_C(  65), -INT8_C( 113),  INT8_C(  48),  INT8_C(  62), -INT8_C(  59),  INT8_C(  18), -INT8_C(  37),
         INT8_C(  89),  INT8_C(   5), -INT8_C(  29),  INT8_C(  32),  INT8_C( 111), -INT8_C( 105),  INT8_C( 115),  INT8_C( 108) },
      { -INT8_C( 109), -INT8_C(  74),  INT8_C(  56),  INT8_C( 119),  INT8_C(  61),  INT8_C(  19), -INT8_C( 107),  INT8_C(  48),
        -INT8_C(  25), -INT8_C(  99), -INT8_C( 103), -INT8_C( 118),  INT8_C(  62),  INT8_C(  92),  INT8_C(  53),  INT8_C(   0),
         INT8_C(  63), -INT8_C(   5), -INT8_C(   7),  INT8_C(  35), -INT8_C(   3),  INT8_C(  53), -INT8_C(  30), -INT8_C(  99),
        -INT8_C( 108), -INT8_C(  14),  INT8_C(   8), -INT8_C(  58),  INT8_C( 106), -INT8_C(  76),  INT8_C(  83),  INT8_C(   0),
         INT8_C( 106), -INT8_C( 117),  INT8_C( 116), -INT8_C(  89), -INT8_C(  98),  INT8_C(   9), -INT8_C(  40), -INT8_C( 122),
        -INT8_C(  89),  INT8_C( 113),  INT8_C(  16), -INT8_C(  27), -INT8_C(  51),  INT8_C(  69), -INT8_C( 106),  INT8_C(   0),
         INT8_C(  65), -INT8_C( 113),  INT8_C(  48),  INT8_C(  62), -INT8_C(  59),  INT8_C(  18), -INT8_C(  37),  INT8_C(  89),
         INT8_C(   5), -INT8_C(  29),  INT8_C(  32),  INT8_C( 111), -INT8_C( 105),  INT8_C( 115),  INT8_C( 108),  INT8_C(   0) } },
    { {  INT8_C(   2), -INT8_C(   2), -INT8_C(  32), -INT8_C(  87), -INT8_C(  99), -INT8_C(  22), -INT8_C( 127),  INT8_C(  35),
        -INT8_C( 111), -INT8_C(  14),  INT8_C(  51),  INT8_C( 118), -INT8_C(  65),  INT8_C( 121),  INT8_C(  12), -INT8_C(  52),
        -INT8_C(  70), -INT8_C( 101), -INT8_C(   4), -INT8_C(   8),  INT8_C(  96),  INT8_C(  14), -INT8_C(  45), -INT8_C(  70),
         INT8_C(  19), -INT8_C(  73), -INT8_C(  38), -INT8_C( 126),  INT8_C(  78),  INT8_C(  77), -INT8_C(  18),  INT8_C(  80),
         INT8_C(  75), -INT8_C(  49), -INT8_C(   6), -INT8_C(  24), -INT8_C(  71),  INT8_C( 123),  INT8_C(  11),  INT8_C(  74),
         INT8_C( 110),  INT8_C(  63), -INT8_C(  64),  INT8_C(  45), -INT8_C(  72), -INT8_C(  52), -INT8_C(   7),  INT8_C( 114),
         INT8_C( 103), -INT8_C(  11),  INT8_C( 106), -INT8_C(  56),  INT8_C(   4),  INT8_C(  61), -INT8_C( 126),  INT8_C(  23),
        -INT8_C(  12),  INT8_C(  92), -INT8_C( 102),  INT8_C(  67), -INT8_C(  87), -INT8_C( 120), -INT8_C( 109), -INT8_C(  12) },
      { -INT8_C(  32), -INT8_C(  87), -INT8_C(  99), -INT8_C(  22), -INT8_C( 127),  INT8_C(  35), -INT8_C( 111), -INT8_C(  14),
         INT8_C(  51),  INT8_C( 118), -INT8_C(  65),  INT8_C( 121),  INT8_C(  12), -INT8_C(  52),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(   4), -INT8_C(   8),  INT8_C(  96),  INT8_C(  14), -INT8_C(  45), -INT8_C(  70),  INT8_C(  19), -INT8_C(  73),
        -INT8_C(  38), -INT8_C( 126),  INT8_C(  78),  INT8_C(  77), -INT8_C(  18),  INT8_C(  80),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(   6), -INT8_C(  24), -INT8_C(  71),  INT8_C( 123),  INT8_C(  11),  INT8_C(  74),  INT8_C( 110),  INT8_C(  63),
        -INT8_C(  64),  INT8_C(  45), -INT8_C(  72), -INT8_C(  52), -INT8_C(   7),  INT8_C( 114),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 106), -INT8_C(  56),  INT8_C(   4),  INT8_C(  61), -INT8_C( 126),  INT8_C(  23), -INT8_C(  12),  INT8_C(  92),
        -INT8_C( 102),  INT8_C(  67), -INT8_C(  87), -INT8_C( 120), -INT8_C( 109), -INT8_C(  12),  INT8_C(   0),  INT8_C(   0) } },
    { {  INT8_C(  87), -INT8_C( 115), -INT8_C(  35),  INT8_C(  16),  INT8_C(   9), -INT8_C(  24),  INT8_C(  90),  INT8_C( 119),
         INT8_C(  39),  INT8_C(  26), -INT8_C(  92), -INT8_C(  33), -INT8_C(  26), -INT8_C(  98),  INT8_C(  81),  INT8_C(  78),
        -INT8_C( 109), -INT8_C(  69),  INT8_C(  22), -INT8_C( 105), -INT8_C(   7), -INT8_C( 104), -INT8_C(  81), -INT8_C(  19),
        -INT8_C(  12),  INT8_C(  73),  INT8_C(  48), -INT8_C(  99), -INT8_C(  47), -INT8_C(  60), -INT8_C( 111),  INT8_C(  41),
         INT8_C(  81),  INT8_C( 110),  INT8_C(  57),  INT8_C(  90),  INT8_C(  87), -INT8_C( 108), -INT8_C(  47),  INT8_C( 126),
        -INT8_C(  82),  INT8_C( 118),  INT8_C(  94), -INT8_C( 107),  INT8_C(  20), -INT8_C(  81), -INT8_C(  29), -INT8_C(  89),
         INT8_C( 107), -INT8_C(   7),  INT8_C(  63),  INT8_C( 100), -INT8_C( 111), -INT8_C(  18),  INT8_C(  81), -INT8_C( 123),
         INT8_C(  55), -INT8_C( 126),  INT8_C(  34),  INT8_C(   8),  INT8_C(  70), -INT8_C(  77),  INT8_C(  49), -INT8_C( 105) },
      {  INT8_C(  16),  INT8_C(   9), -INT8_C(  24),  INT8_C(  90),  INT8_C( 119),  INT8_C(  39),  INT8_C(  26), -INT8_C(  92),
        -INT8_C(  33), -INT8_C(  26), -INT8_C(  98),  INT8_C(  81),  INT8_C(  78),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 105), -INT8_C(   7), -INT8_C( 104), -INT8_C(  81), -INT8_C(  19), -INT8_C(  12),  INT8_C(  73),  INT8_C(  48),
        -INT8_C(  99), -INT8_C(  47), -INT8_C(  60), -INT8_C( 111),  INT8_C(  41),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  90),  INT8_C(  87), -INT8_C( 108), -INT8_C(  47),  INT8_C( 126), -INT8_C(  82),  INT8_C( 118),  INT8_C(  94),
        -INT8_C( 107),  INT8_C(  20), -INT8_C(  81), -INT8_C(  29), -INT8_C(  89),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 100), -INT8_C( 111), -INT8_C(  18),  INT8_C(  81), -INT8_C( 123),  INT8_C(  55), -INT8_C( 126),  INT8_C(  34),
         INT8_C(   8),  INT8_C(  70), -INT8_C(  77),  INT8_C(  49), -INT8_C( 105),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { {  INT8_C(  34),  INT8_C( 107), -INT8_C(  14),  INT8_C( 121), -INT8_C(   1), -INT8_C(  61), -INT8_C(   9), -INT8_C(  83),
         INT8_C(  57),  INT8_C(  85),  INT8_C(  66),  INT8_C(  77),  INT8_C(   5),  INT8_C(  37), -INT8_C(  11),  INT8_C( 112),
         INT8_C(  30),  INT8_C(  52), -INT8_C(  44), -INT8_C(  81),  INT8_C(  34),  INT8_C(  37),  INT8_C(  52),  INT8_C(  89),
        -INT8_C(  89),  INT8_C(  86),  INT8_C(  97), -INT8_C(  19),  INT8_C(  10), -INT8_C( 109), -INT8_C( 123),  INT8_C(  44),
        -INT8_C(   2),  INT8_C( 119), -INT8_C(  91), -INT8_C(   3),  INT8_C(  58), -INT8_C( 100), -INT8_C(  86),  INT8_C( 116),
        -INT8_C(  14), -INT8_C(  19), -INT8_C(  63), -INT8_C(   9),  INT8_C(  18), -INT8_C(  74),  INT8_C( 103),  INT8_C(  49),
        -INT8_C(  22),  INT8_C(  59), -INT8_C(  32),  INT8_C(  12),  INT8_C(  96),  INT8_C(  21),  INT8_C( 101),  INT8_C(   8),
         INT8_C( 107), -INT8_C(  57), -INT8_C(  11),  INT8_C( 117),  INT8_C(  90),  INT8_C( 122), -INT8_C(  95),  INT8_C(  88) },
      { -INT8_C(   1), -INT8_C(  61), -INT8_C(   9), -INT8_C(  83),  INT8_C(  57),  INT8_C(  85),  INT8_C(  66),  INT8_C(  77),
         INT8_C(   5),  INT8_C(  37), -INT8_C(  11),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  34),  INT8_C(  37),  INT8_C(  52),  INT8_C(  89), -INT8_C(  89),  INT8_C(  86),  INT8_C(  97), -INT8_C(  19),
         INT8_C(  10), -INT8_C( 109), -INT8_C( 123),  INT8_C(  44),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  58), -INT8_C( 100), -INT8_C(  86),  INT8_C( 116), -INT8_C(  14), -INT8_C(  19), -INT8_C(  63), -INT8_C(   9),
         INT8_C(  18), -INT8_C(  74),  INT8_C( 103),  INT8_C(  49),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  96),  INT8_C(  21),  INT8_C( 101),  INT8_C(   8),  INT8_C( 107), -INT8_C(  57), -INT8_C(  11),  INT8_C( 117),
         INT8_C(  90),  INT8_C( 122), -INT8_C(  95),  INT8_C(  88),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { { -INT8_C(  15),  INT8_C(  70),  INT8_C(  85),  INT8_C(  44), -INT8_C(  29), -INT8_C(   1), -INT8_C(  96), -INT8_C(  43),
        -INT8_C(  20),  INT8_C(  97), -INT8_C(  52), -INT8_C(   1),  INT8_C(  24),  INT8_C(  51),  INT8_C(  48),  INT8_C(   2),
         INT8_C( 110),  INT8_C(  16),  INT8_C(  15), -INT8_C(  50),  INT8_C(  37),  INT8_C( 116), -INT8_C(  42), -INT8_C( 111),
         INT8_C(  59), -INT8_C(  52),  INT8_C(   6), -INT8_C( 107),  INT8_C(  70), -INT8_C(  88), -INT8_C(  19),  INT8_C(  56),
        -INT8_C(  18),  INT8_C(  66),  INT8_C( 100), -INT8_C(  47),  INT8_C(  66),  INT8_C(   4), -INT8_C(  90),  INT8_C(  46),
         INT8_C( 101),  INT8_C( 114),  INT8_C(  45),  INT8_C( 125), -INT8_C(  91),  INT8_C(  93),      INT8_MIN,  INT8_C(  19),
         INT8_C( 110), -INT8_C( 113), -INT8_C(  30), -INT8_C( 109),  INT8_C(   3), -INT8_C(  72),  INT8_C(  36),  INT8_C(  63),
        -INT8_C( 124),  INT8_C(  43), -INT8_C(  44), -INT8_C(  53), -INT8_C(  45), -INT8_C(  62),  INT8_C(   3), -INT8_C(  63) },
      { -INT8_C(   1), -INT8_C(  96), -INT8_C(  43), -INT8_C(  20),  INT8_C(  97), -INT8_C(  52), -INT8_C(   1),  INT8_C(  24),
         INT8_C(  51),  INT8_C(  48),  INT8_C(   2),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 116), -INT8_C(  42), -INT8_C( 111),  INT8_C(  59), -INT8_C(  52),  INT8_C(   6), -INT8_C( 107),  INT8_C(  70),
        -INT8_C(  88), -INT8_C(  19),  INT8_C(  56),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   4), -INT8_C(  90),  INT8_C(  46),  INT8_C( 101),  INT8_C( 114),  INT8_C(  45),  INT8_C( 125), -INT8_C(  91),
         INT8_C(  93),      INT8_MIN,  INT8_C(  19),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  72),  INT8_C(  36),  INT8_C(  63), -INT8_C( 124),  INT8_C(  43), -INT8_C(  44), -INT8_C(  53), -INT8_C(  45),
        -INT8_C(  62),  INT8_C(   3), -INT8_C(  63),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { {  INT8_C(   4),  INT8_C( 103), -INT8_C( 109),  INT8_C(  70),  INT8_C( 107),  INT8_C(  57),  INT8_C( 117), -INT8_C(  48),
        -INT8_C(  84), -INT8_C(  94),  INT8_C(  78),  INT8_C(  81),  INT8_C(   0), -INT8_C(  50),  INT8_C( 101),  INT8_C( 110),
         INT8_C(  93),  INT8_C(  71),  INT8_C(   1),  INT8_C(  96), -INT8_C(   1),  INT8_C(  38), -INT8_C(  97), -INT8_C( 124),
         INT8_C(  81),  INT8_C( 116),  INT8_C(  79),  INT8_C(  36),  INT8_C(  54),  INT8_C(  82), -INT8_C(  27),  INT8_C(  58),
        -INT8_C(  71),  INT8_C( 120), -INT8_C( 127),  INT8_C(  36), -INT8_C(  78), -INT8_C(  10), -INT8_C(  12),  INT8_C(  94),
        -INT8_C( 104),  INT8_C(  66), -INT8_C(  81), -INT8_C( 104),  INT8_C(  16),  INT8_C(  20),  INT8_C(   6),  INT8_C( 109),
         INT8_C(  91),  INT8_C(   8), -INT8_C(  50),  INT8_C(  91),  INT8_C(  46),  INT8_C( 109), -INT8_C(  33),      INT8_MAX,
        -INT8_C(  31),  INT8_C(  46), -INT8_C(  93),  INT8_C(  23),      INT8_MIN, -INT8_C( 120),  INT8_C(  82),  INT8_C(  57) },
      {  INT8_C( 117), -INT8_C(  48), -INT8_C(  84), -INT8_C(  94),  INT8_C(  78),  INT8_C(  81),  INT8_C(   0), -INT8_C(  50),
         INT8_C( 101),  INT8_C( 110),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  97), -INT8_C( 124),  INT8_C(  81),  INT8_C( 116),  INT8_C(  79),  INT8_C(  36),  INT8_C(  54),  INT8_C(  82),
        -INT8_C(  27),  INT8_C(  58),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  12),  INT8_C(  94), -INT8_C( 104),  INT8_C(  66), -INT8_C(  81), -INT8_C( 104),  INT8_C(  16),  INT8_C(  20),
         INT8_C(   6),  INT8_C( 109),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  33),      INT8_MAX, -INT8_C(  31),  INT8_C(  46), -INT8_C(  93),  INT8_C(  23),      INT8_MIN, -INT8_C( 120),
         INT8_C(  82),  INT8_C(  57),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { {  INT8_C(   1), -INT8_C(  45),  INT8_C(  93), -INT8_C(  77), -INT8_C(  55),  INT8_C(  81),  INT8_C(  17),  INT8_C(  97),
        -INT8_C( 108), -INT8_C(  64), -INT8_C(   6), -INT8_C(  92), -INT8_C(  43),  INT8_C(   0),  INT8_C(  18),  INT8_C(  48),
         INT8_C(   8), -INT8_C(  32), -INT8_C( 117),  INT8_C(  54),  INT8_C(  77),  INT8_C( 106), -INT8_C(  75),  INT8_C(  47),
        -INT8_C( 104),  INT8_C(  88),  INT8_C(  70),  INT8_C(  24), -INT8_C(  31), -INT8_C( 104),  INT8_C(  81), -INT8_C(  30),
         INT8_C( 107), -INT8_C(  82), -INT8_C( 107),  INT8_C(  52),  INT8_C(   0), -INT8_C(  90), -INT8_C( 106), -INT8_C( 108),
         INT8_C( 102), -INT8_C( 112),  INT8_C(  56),  INT8_C(  59), -INT8_C( 112),  INT8_C(  74),  INT8_C( 108), -INT8_C( 103),
         INT8_C(  42), -INT8_C(   9), -INT8_C(  49),  INT8_C( 120),  INT8_C(  98), -INT8_C( 123), -INT8_C(  89), -INT8_C(   6),
        -INT8_C(  35), -INT8_C(  19),  INT8_C(  19), -INT8_C(  66), -INT8_C( 122),  INT8_C( 100), -INT8_C(  96), -INT8_C(  15) },
      {  INT8_C(  97), -INT8_C( 108), -INT8_C(  64), -INT8_C(   6), -INT8_C(  92), -INT8_C(  43),  INT8_C(   0),  INT8_C(  18),
         INT8_C(  48),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  47), -INT8_C( 104),  INT8_C(  88),  INT8_C(  70),  INT8_C(  24), -INT8_C(  31), -INT8_C( 104),  INT8_C(  81),
        -INT8_C(  30),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 108),  INT8_C( 102), -INT8_C( 112),  INT8_C(  56),  INT8_C(  59), -INT8_C( 112),  INT8_C(  74),  INT8_C( 108),
        -INT8_C( 103),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(   6), -INT8_C(  35), -INT8_C(  19),  INT8_C(  19), -INT8_C(  66), -INT8_C( 122),  INT8_C( 100), -INT8_C(  96),
        -INT8_C(  15),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { {  INT8_C(  19),  INT8_C(  53),  INT8_C(  38),  INT8_C(  19), -INT8_C(  37), -INT8_C(  68), -INT8_C(  89),  INT8_C(  66),
         INT8_C(  76), -INT8_C(  33),  INT8_C( 125), -INT8_C(  36),  INT8_C(  42), -INT8_C(  23),  INT8_C( 117),  INT8_C(  84),
        -INT8_C(  31),  INT8_C(  69), -INT8_C(  52),  INT8_C(  67), -INT8_C(  54),  INT8_C( 115),  INT8_C(  61), -INT8_C(  89),
         INT8_C(  97),  INT8_C(  80),  INT8_C( 102), -INT8_C(  25), -INT8_C(  75),  INT8_C(   6), -INT8_C(  40), -INT8_C(  56),
         INT8_C(  60), -INT8_C(   2), -INT8_C(  37),  INT8_C(  23), -INT8_C(  70), -INT8_C( 126),  INT8_C(  89),  INT8_C(   6),
         INT8_C(  97), -INT8_C(  41), -INT8_C(  29), -INT8_C( 117), -INT8_C(  64),  INT8_C(  88), -INT8_C(  32), -INT8_C(  95),
        -INT8_C(  99), -INT8_C(  84), -INT8_C(  28),  INT8_C( 103),  INT8_C(  32),  INT8_C(  34),  INT8_C(  15), -INT8_C( 127),
         INT8_C( 114),  INT8_C( 117),  INT8_C( 104),  INT8_C(  39),  INT8_C( 123),  INT8_C(  64), -INT8_C(  17), -INT8_C(  73) },
      {  INT8_C(  76), -INT8_C(  33),  INT8_C( 125), -INT8_C(  36),  INT8_C(  42), -INT8_C(  23),  INT8_C( 117),  INT8_C(  84),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  97),  INT8_C(  80),  INT8_C( 102), -INT8_C(  25), -INT8_C(  75),  INT8_C(   6), -INT8_C(  40), -INT8_C(  56),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  97), -INT8_C(  41), -INT8_C(  29), -INT8_C( 117), -INT8_C(  64),  INT8_C(  88), -INT8_C(  32), -INT8_C(  95),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 114),  INT8_C( 117),  INT8_C( 104),  INT8_C(  39),  INT8_C( 123),  INT8_C(  64), -INT8_C(  17), -INT8_C(  73),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } }
  };

  easysimd__m512i a;
  easysimd__m512i r;

  a = easysimd_mm512_loadu_epi8(test_vec[0].a);
  r = easysimd_mm512_bsrli_epi128(a,  1);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[0].r));

  a = easysimd_mm512_loadu_epi8(test_vec[1].a);
  r = easysimd_mm512_bsrli_epi128(a,  2);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[1].r));

  a = easysimd_mm512_loadu_epi8(test_vec[2].a);
  r = easysimd_mm512_bsrli_epi128(a,  3);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[2].r));

  a = easysimd_mm512_loadu_epi8(test_vec[3].a);
  r = easysimd_mm512_bsrli_epi128(a,  4);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[3].r));

  a = easysimd_mm512_loadu_epi8(test_vec[4].a);
  r = easysimd_mm512_bsrli_epi128(a,  5);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[4].r));

  a = easysimd_mm512_loadu_epi8(test_vec[5].a);
  r = easysimd_mm512_bsrli_epi128(a,  6);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[5].r));

  a = easysimd_mm512_loadu_epi8(test_vec[6].a);
  r = easysimd_mm512_bsrli_epi128(a,  7);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[6].r));

  a = easysimd_mm512_loadu_epi8(test_vec[7].a);
  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r = easysimd_mm512_bsrli_epi128(a,  8);
  } EASYSIMD_TEST_PERF_END("easysimd_mm512_bsrli_epi128");
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[7].r));

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i r = easysimd_mm512_bsrli_epi128(a, (i + 1));

    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}


EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srl_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_srl_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_srl_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_srl_epi16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_srl_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_srl_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_srl_epi16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srl_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_srl_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_srl_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_srl_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_srl_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_srl_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_srl_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srl_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_srl_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_srl_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_srl_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_srl_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_srl_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_srl_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_bsrli_epi128)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
