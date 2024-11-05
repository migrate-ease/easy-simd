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

#define EASYSIMD_TEST_X86_AVX512_INSN sll

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/sll.h>

static int
test_easysimd_mm512_sll_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int64_t b[2];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 11176), -INT16_C( 31146),  INT16_C( 31553),  INT16_C(    68),  INT16_C(   109), -INT16_C(  6231), -INT16_C( 28515),  INT16_C(  4842),
        -INT16_C(  8039), -INT16_C( 26148), -INT16_C( 31237),  INT16_C( 19279), -INT16_C(  2919), -INT16_C( 17167),  INT16_C( 24677), -INT16_C( 16888),
         INT16_C( 24116),  INT16_C( 30020), -INT16_C( 30246),  INT16_C( 18294),  INT16_C(  8073),  INT16_C( 10031),  INT16_C(  6575),  INT16_C( 18489),
         INT16_C(  5625), -INT16_C(  2847),  INT16_C( 12442),  INT16_C( 13375),  INT16_C( 12324), -INT16_C( 29968), -INT16_C(  1903), -INT16_C( 15032) },
      {  INT64_C(                  19),  INT64_C(                  23) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C( 22508), -INT16_C( 20817), -INT16_C(  4391), -INT16_C(   542), -INT16_C( 11489), -INT16_C( 20345), -INT16_C( 12341),  INT16_C(  8821),
        -INT16_C( 20388),  INT16_C( 29011), -INT16_C( 13215),  INT16_C( 12560), -INT16_C( 10637),  INT16_C( 13488),  INT16_C( 30677), -INT16_C(  5649),
        -INT16_C( 25056), -INT16_C(  1640),  INT16_C( 31372), -INT16_C( 21514),  INT16_C( 32333),  INT16_C(  6491), -INT16_C( 11955), -INT16_C( 22213),
        -INT16_C( 28799), -INT16_C(  7397),  INT16_C( 11099), -INT16_C( 12780), -INT16_C( 15102), -INT16_C( 10493), -INT16_C(  3524),  INT16_C( 23745) },
      {  INT64_C(                   4),  INT64_C(                  14) },
      { -INT16_C( 32448), -INT16_C(  5392), -INT16_C(  4720), -INT16_C(  8672),  INT16_C( 12784),  INT16_C(  2160), -INT16_C(   848),  INT16_C( 10064),
         INT16_C(  1472),  INT16_C(  5424), -INT16_C( 14832),  INT16_C(  4352),  INT16_C( 26416),  INT16_C( 19200),  INT16_C( 32080), -INT16_C( 24848),
        -INT16_C(  7680), -INT16_C( 26240), -INT16_C( 22336), -INT16_C( 16544), -INT16_C(  6960), -INT16_C( 27216),  INT16_C(  5328), -INT16_C( 27728),
        -INT16_C(  2032),  INT16_C( 12720), -INT16_C( 19024), -INT16_C(  7872),  INT16_C( 20512),  INT16_C( 28720),  INT16_C(  9152), -INT16_C( 13296) } },
    { { -INT16_C(  9212),  INT16_C( 24409),  INT16_C( 27911),  INT16_C(  2350),  INT16_C( 12594),  INT16_C( 28641), -INT16_C( 24029), -INT16_C( 19509),
         INT16_C(  8699), -INT16_C( 12593), -INT16_C( 26771),  INT16_C( 14319),  INT16_C( 10683), -INT16_C( 20658),  INT16_C(  3999), -INT16_C( 23771),
         INT16_C( 32491), -INT16_C(  3325),  INT16_C( 12780),  INT16_C(  7932), -INT16_C(  8862), -INT16_C( 31347),  INT16_C( 22911),  INT16_C( 31288),
         INT16_C(  1914), -INT16_C(  6327),  INT16_C( 14495),  INT16_C( 23070),  INT16_C( 27746),  INT16_C(   265),  INT16_C( 12156),  INT16_C( 26532) },
      {  INT64_C(                   7),  INT64_C(                   3) },
      {  INT16_C(   512), -INT16_C( 21376), -INT16_C( 31872), -INT16_C( 26880), -INT16_C( 26368), -INT16_C(  3968),  INT16_C(  4480), -INT16_C(  6784),
        -INT16_C(   640),  INT16_C( 26496), -INT16_C( 18816), -INT16_C(  2176), -INT16_C(  8832), -INT16_C( 22784), -INT16_C( 12416), -INT16_C( 28032),
         INT16_C( 30080), -INT16_C( 32384), -INT16_C(  2560),  INT16_C( 32256), -INT16_C( 20224), -INT16_C( 14720), -INT16_C( 16512),  INT16_C(  7168),
        -INT16_C( 17152), -INT16_C( 23424),  INT16_C( 20352),  INT16_C(  3840),  INT16_C( 12544), -INT16_C( 31616), -INT16_C( 16896), -INT16_C( 11776) } },
    { {  INT16_C( 30719), -INT16_C( 24833),  INT16_C(  7600),  INT16_C(  4856),  INT16_C(   394),  INT16_C(  1555), -INT16_C( 18640), -INT16_C(  8595),
        -INT16_C( 14241),  INT16_C( 14199),  INT16_C( 12063),  INT16_C( 21362),  INT16_C( 12661),  INT16_C(  4871),  INT16_C( 13865),  INT16_C( 10284),
         INT16_C( 11181),  INT16_C( 24006), -INT16_C( 16823), -INT16_C( 11409), -INT16_C( 32065), -INT16_C(  3879),  INT16_C( 17978), -INT16_C( 26162),
         INT16_C( 17678),  INT16_C( 11728),  INT16_C( 17013), -INT16_C(  5503), -INT16_C( 30604), -INT16_C( 25091),  INT16_C( 10686),  INT16_C( 27845) },
      {  INT64_C(                   1),  INT64_C(                  19) },
      { -INT16_C(  4098),  INT16_C( 15870),  INT16_C( 15200),  INT16_C(  9712),  INT16_C(   788),  INT16_C(  3110),  INT16_C( 28256), -INT16_C( 17190),
        -INT16_C( 28482),  INT16_C( 28398),  INT16_C( 24126), -INT16_C( 22812),  INT16_C( 25322),  INT16_C(  9742),  INT16_C( 27730),  INT16_C( 20568),
         INT16_C( 22362), -INT16_C( 17524),  INT16_C( 31890), -INT16_C( 22818),  INT16_C(  1406), -INT16_C(  7758), -INT16_C( 29580),  INT16_C( 13212),
        -INT16_C( 30180),  INT16_C( 23456), -INT16_C( 31510), -INT16_C( 11006),  INT16_C(  4328),  INT16_C( 15354),  INT16_C( 21372), -INT16_C(  9846) } },
    { {  INT16_C( 24332), -INT16_C( 32308),  INT16_C( 19873),  INT16_C(  5483),  INT16_C( 26838), -INT16_C( 27470),  INT16_C( 30610), -INT16_C(  6400),
        -INT16_C( 13822),  INT16_C( 19333), -INT16_C(  2557), -INT16_C( 16812),  INT16_C( 19520), -INT16_C( 12108),  INT16_C( 16915),  INT16_C(  8047),
         INT16_C( 15521),  INT16_C( 17312),  INT16_C(  2953),  INT16_C( 24408),  INT16_C(  2931),  INT16_C(  1524), -INT16_C(  2942), -INT16_C( 31252),
         INT16_C( 29118), -INT16_C( 15920),  INT16_C(  9319), -INT16_C( 22656),  INT16_C( 13425), -INT16_C( 31624), -INT16_C(  6282),  INT16_C(  6307) },
      {  INT64_C(                  24),  INT64_C(                  22) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C( 27099), -INT16_C( 29579), -INT16_C(  2629),  INT16_C( 11316), -INT16_C( 21463), -INT16_C( 24656),  INT16_C( 21395), -INT16_C( 18505),
         INT16_C(  4758), -INT16_C(  7068),  INT16_C( 28870), -INT16_C( 31579),  INT16_C( 27761),  INT16_C( 26309), -INT16_C( 29920),  INT16_C( 17689),
        -INT16_C( 29150), -INT16_C(  8751),  INT16_C(  1411), -INT16_C( 21495), -INT16_C( 17999),  INT16_C( 17740),  INT16_C(   780), -INT16_C( 23812),
         INT16_C( 24598), -INT16_C(  9082),  INT16_C( 11216),  INT16_C( 16736),  INT16_C(  9880), -INT16_C( 18265), -INT16_C( 15951), -INT16_C( 11267) },
      {  INT64_C(                  16),  INT64_C(                  28) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C( 23002), -INT16_C(  2111),  INT16_C(  8658),  INT16_C( 27192), -INT16_C(  8121), -INT16_C(  1758),  INT16_C(  8097), -INT16_C(  3892),
         INT16_C( 32237), -INT16_C( 15933),  INT16_C( 17206), -INT16_C( 22201),  INT16_C(  4366), -INT16_C(  8921),  INT16_C( 18648), -INT16_C(   318),
        -INT16_C( 31762), -INT16_C( 16139),  INT16_C( 11941), -INT16_C(  5078),  INT16_C( 19470), -INT16_C( 20507), -INT16_C( 19861),  INT16_C( 22943),
         INT16_C( 25391),  INT16_C( 25882),  INT16_C( 24998), -INT16_C( 19442),  INT16_C( 13939),  INT16_C( 19346),  INT16_C( 21630),  INT16_C( 27721) },
      {  INT64_C(                  15),  INT64_C(                   6) },
      {  INT16_C(     0),       INT16_MIN,  INT16_C(     0),  INT16_C(     0),       INT16_MIN,  INT16_C(     0),       INT16_MIN,  INT16_C(     0),
              INT16_MIN,       INT16_MIN,  INT16_C(     0),       INT16_MIN,  INT16_C(     0),       INT16_MIN,  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),       INT16_MIN,       INT16_MIN,  INT16_C(     0),  INT16_C(     0),       INT16_MIN,       INT16_MIN,       INT16_MIN,
              INT16_MIN,  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),       INT16_MIN,  INT16_C(     0),  INT16_C(     0),       INT16_MIN } },
    { { -INT16_C( 31956), -INT16_C( 11627), -INT16_C( 23324),  INT16_C( 22407),  INT16_C(  6618),  INT16_C( 22690), -INT16_C(  5011),  INT16_C( 17860),
        -INT16_C(  3797), -INT16_C( 26430),  INT16_C( 11337), -INT16_C(  4845),  INT16_C( 15739),  INT16_C( 31996),  INT16_C( 25862),  INT16_C( 13228),
         INT16_C( 16872), -INT16_C( 13307), -INT16_C( 29467), -INT16_C( 16604), -INT16_C( 14683),  INT16_C(  4887), -INT16_C(  9038), -INT16_C(  8872),
         INT16_C(  7117),  INT16_C(  5749), -INT16_C( 30649), -INT16_C( 15869),  INT16_C(   197), -INT16_C( 13250), -INT16_C(  5531),  INT16_C( 19967) },
      {  INT64_C(                   6),  INT64_C(                  18) },
      { -INT16_C( 13568), -INT16_C( 23232),  INT16_C( 14592), -INT16_C(  7744),  INT16_C( 30336),  INT16_C( 10368),  INT16_C(  6976),  INT16_C( 28928),
         INT16_C( 19136),  INT16_C( 12416),  INT16_C(  4672),  INT16_C( 17600),  INT16_C( 24256),  INT16_C( 16128),  INT16_C( 16768), -INT16_C(  5376),
         INT16_C( 31232),  INT16_C(   320),  INT16_C( 14656), -INT16_C( 14080), -INT16_C( 22208), -INT16_C( 14912),  INT16_C( 11392),  INT16_C( 22016),
        -INT16_C(  3264), -INT16_C( 25280),  INT16_C(  4544), -INT16_C( 32576),  INT16_C( 12608),  INT16_C(  3968), -INT16_C( 26304),  INT16_C( 32704) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sll_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sll_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sll_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int64_t count[2];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 21525), -INT16_C( 32533),  INT16_C( 21413),  INT16_C(  7535), -INT16_C(  3695),  INT16_C( 13787),  INT16_C( 26396), -INT16_C( 19626),
        -INT16_C( 10603), -INT16_C( 32249), -INT16_C( 25187), -INT16_C( 28361),  INT16_C(  8365),  INT16_C( 30498), -INT16_C( 15983),  INT16_C( 32126),
         INT16_C( 26988),  INT16_C(  4605),  INT16_C( 28093),  INT16_C( 20015),  INT16_C(  2654),  INT16_C( 31364), -INT16_C(  9615),  INT16_C(  1837),
         INT16_C( 13488),  INT16_C( 19849), -INT16_C( 16175),  INT16_C( 32479),  INT16_C(   481),  INT16_C( 29430),  INT16_C( 29890),  INT16_C( 12015) },
      UINT32_C(2604723678),
      {  INT16_C( 28506), -INT16_C( 18199),  INT16_C( 28025), -INT16_C(  5326),  INT16_C( 24391), -INT16_C(  2062),  INT16_C( 31636),  INT16_C( 25925),
         INT16_C(  9275),  INT16_C(  7396), -INT16_C(  9691), -INT16_C(  6257),  INT16_C( 32334),  INT16_C( 11285),  INT16_C( 21867), -INT16_C( 14905),
        -INT16_C( 20028),  INT16_C( 15997), -INT16_C( 20706),  INT16_C( 26153),  INT16_C(  6927), -INT16_C( 23715), -INT16_C( 23914), -INT16_C( 12024),
        -INT16_C(  4922), -INT16_C(  5138),  INT16_C( 32198),  INT16_C(  5586), -INT16_C(  5893),  INT16_C( 26433),  INT16_C(  2365),  INT16_C(   556) },
      {  INT64_C( 7511557063101688506),  INT64_C(  426456846674140292) },
      { -INT16_C( 21525),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 13787),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 32249),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8365),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 26988),  INT16_C(  4605),  INT16_C( 28093),  INT16_C( 20015),  INT16_C(  2654),  INT16_C( 31364),  INT16_C(     0),  INT16_C(  1837),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 16175),  INT16_C(     0),  INT16_C(     0),  INT16_C( 29430),  INT16_C( 29890),  INT16_C(     0) } },
    { { -INT16_C(  9984), -INT16_C( 14352), -INT16_C( 15530),  INT16_C( 21212),  INT16_C(  7595), -INT16_C(  5959), -INT16_C(  6874), -INT16_C(  7958),
         INT16_C( 10895), -INT16_C(  5703), -INT16_C(  2157),  INT16_C(  5969),  INT16_C( 23955), -INT16_C( 11727),  INT16_C(  7537),  INT16_C( 29143),
        -INT16_C( 14346),  INT16_C( 19768),  INT16_C(  5258),  INT16_C( 13727),  INT16_C( 22578),  INT16_C( 22558),  INT16_C(  2109), -INT16_C( 12999),
        -INT16_C(  3533), -INT16_C( 14666),  INT16_C(  2025),  INT16_C( 32222),  INT16_C(  3940), -INT16_C( 10929),  INT16_C(  9772),  INT16_C(  9031) },
      UINT32_C(2020638701),
      {  INT16_C(  3988), -INT16_C( 14675), -INT16_C( 13465), -INT16_C( 23522),  INT16_C( 22484),  INT16_C(  1905),  INT16_C( 10057),  INT16_C( 13261),
        -INT16_C( 21713), -INT16_C( 27728), -INT16_C(    69), -INT16_C(  6295), -INT16_C( 20443),  INT16_C(  4618),  INT16_C( 31279), -INT16_C( 15478),
         INT16_C( 14473), -INT16_C(  3959), -INT16_C( 22525), -INT16_C( 10347),  INT16_C(  1791),  INT16_C( 18910), -INT16_C( 21458),  INT16_C( 23932),
         INT16_C( 11351),  INT16_C(  4848),  INT16_C( 22827),  INT16_C( 20730),  INT16_C(  1033),  INT16_C( 14690), -INT16_C(  4737),  INT16_C(  2300) },
      {  INT64_C( 3242748060613838373), -INT64_C( 2152735419077828972) },
      {  INT16_C(     0), -INT16_C( 14352),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7595),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 29143),
        -INT16_C( 14346),  INT16_C( 19768),  INT16_C(  5258),  INT16_C( 13727),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 12999),
        -INT16_C(  3533), -INT16_C( 14666),  INT16_C(  2025),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  9031) } },
    { {  INT16_C(  4126),  INT16_C( 18932), -INT16_C(  4503),  INT16_C( 29593), -INT16_C(   781),  INT16_C( 29356), -INT16_C( 22295),  INT16_C(  3706),
         INT16_C( 29486),  INT16_C( 23606),  INT16_C( 13825), -INT16_C( 26998),  INT16_C(    21), -INT16_C( 24744),  INT16_C( 30963),  INT16_C(  4481),
         INT16_C( 30344), -INT16_C(  3749), -INT16_C(  2972),  INT16_C( 22372),  INT16_C(  4336), -INT16_C(  9783),  INT16_C( 17593), -INT16_C(  6169),
         INT16_C(  7863), -INT16_C( 18108), -INT16_C( 12716),  INT16_C( 26959), -INT16_C( 22578), -INT16_C( 16119), -INT16_C( 30177), -INT16_C( 22573) },
      UINT32_C(1704537600),
      { -INT16_C(   734),  INT16_C(  5052), -INT16_C( 31218), -INT16_C( 14356), -INT16_C( 11062), -INT16_C( 32338), -INT16_C(  3342),  INT16_C( 17978),
        -INT16_C( 30272), -INT16_C( 28752), -INT16_C( 18127),  INT16_C( 20560),  INT16_C(  9027),  INT16_C( 17656), -INT16_C( 28335),  INT16_C( 29865),
         INT16_C( 25998), -INT16_C( 25465),  INT16_C( 29675), -INT16_C( 19101),  INT16_C(  4679),  INT16_C( 14647),  INT16_C( 28932), -INT16_C( 14976),
         INT16_C( 12539),  INT16_C( 11348), -INT16_C( 23319),  INT16_C( 11388),  INT16_C( 29896),  INT16_C(  6512),  INT16_C(  6405), -INT16_C( 27507) },
      { -INT64_C( 3521651594985728897), -INT64_C( 4364118018959517786) },
      {  INT16_C(  4126),  INT16_C( 18932), -INT16_C(  4503),  INT16_C( 29593), -INT16_C(   781),  INT16_C( 29356), -INT16_C( 22295),  INT16_C(  3706),
         INT16_C( 29486),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(    21),  INT16_C(     0),  INT16_C( 30963),  INT16_C(  4481),
         INT16_C(     0), -INT16_C(  3749), -INT16_C(  2972),  INT16_C(     0),  INT16_C(     0), -INT16_C(  9783),  INT16_C( 17593),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 18108),  INT16_C(     0),  INT16_C( 26959), -INT16_C( 22578),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22573) } },
    { { -INT16_C( 15431), -INT16_C( 23825),  INT16_C( 27752),  INT16_C( 12494),  INT16_C( 16352), -INT16_C(  6583), -INT16_C( 10408), -INT16_C( 10374),
        -INT16_C( 21781),  INT16_C( 29506),  INT16_C( 25150), -INT16_C(  7101),  INT16_C( 19641), -INT16_C( 32369), -INT16_C(   299), -INT16_C( 29115),
         INT16_C( 13506),  INT16_C( 10800), -INT16_C(   352), -INT16_C( 32422), -INT16_C( 23747), -INT16_C( 27033), -INT16_C(  7814),  INT16_C( 26221),
        -INT16_C( 20597), -INT16_C( 13607),  INT16_C(  7185), -INT16_C( 13650),  INT16_C( 15720),  INT16_C( 15692), -INT16_C( 28356), -INT16_C(   309) },
      UINT32_C(1713961925),
      { -INT16_C( 32006),  INT16_C( 14311),  INT16_C( 20005), -INT16_C( 24371),  INT16_C( 15151), -INT16_C( 17914), -INT16_C(  8214), -INT16_C(   892),
         INT16_C( 13308),  INT16_C( 25798),  INT16_C(  4720), -INT16_C( 21342),  INT16_C( 28067),  INT16_C( 27050), -INT16_C( 11671),  INT16_C( 25551),
        -INT16_C( 18860),  INT16_C( 31386),  INT16_C( 26628),  INT16_C( 13082),  INT16_C(  8355), -INT16_C( 29203),  INT16_C( 29439), -INT16_C(  1143),
         INT16_C( 20645),  INT16_C(  5472),  INT16_C(   610),  INT16_C(  1730),  INT16_C( 27759), -INT16_C( 10129),  INT16_C( 15935), -INT16_C( 27845) },
      { -INT64_C( 2221638839774095628),  INT64_C( 3405557541033089095) },
      {  INT16_C(     0), -INT16_C( 23825),  INT16_C(     0),  INT16_C( 12494),  INT16_C( 16352), -INT16_C(  6583),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 25150),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 13506),  INT16_C( 10800), -INT16_C(   352),  INT16_C(     0), -INT16_C( 23747),  INT16_C(     0), -INT16_C(  7814),  INT16_C( 26221),
        -INT16_C( 20597),  INT16_C(     0),  INT16_C(     0), -INT16_C( 13650),  INT16_C( 15720),  INT16_C(     0),  INT16_C(     0), -INT16_C(   309) } },
    { { -INT16_C( 23992), -INT16_C( 21947),  INT16_C(  1956),  INT16_C(  5296),  INT16_C(  8051), -INT16_C( 19732),  INT16_C( 10333),  INT16_C( 20806),
         INT16_C( 21502),  INT16_C( 15433),  INT16_C( 29819), -INT16_C( 15843), -INT16_C( 29811),  INT16_C(  5897),  INT16_C( 19587), -INT16_C( 13497),
        -INT16_C( 29458), -INT16_C( 27786),  INT16_C(  9875),  INT16_C(  1703), -INT16_C( 27834), -INT16_C( 23623), -INT16_C(    69), -INT16_C( 17931),
         INT16_C( 15954), -INT16_C( 12811),  INT16_C(  4787),  INT16_C( 16528), -INT16_C( 26210),  INT16_C(  8535), -INT16_C( 24859), -INT16_C( 11027) },
      UINT32_C(3177669418),
      {  INT16_C(  3721), -INT16_C( 12348),  INT16_C( 32161),  INT16_C( 23923),  INT16_C( 26748), -INT16_C( 12778),  INT16_C(  3238),  INT16_C( 22940),
         INT16_C( 11294), -INT16_C( 17255), -INT16_C(  3643), -INT16_C( 21538), -INT16_C( 13425), -INT16_C( 17793), -INT16_C(  6610), -INT16_C( 18569),
         INT16_C( 15348), -INT16_C( 27257), -INT16_C(  1352),  INT16_C( 13554),  INT16_C(  2402),  INT16_C(  2051), -INT16_C( 24811),  INT16_C( 13154),
        -INT16_C(  1077), -INT16_C( 28432), -INT16_C( 12564),  INT16_C( 31803), -INT16_C( 17767), -INT16_C( 14538), -INT16_C( 21088), -INT16_C( 27522) },
      {  INT64_C( 7049854150941214185),  INT64_C( 4859045864727566629) },
      { -INT16_C( 23992),  INT16_C(     0),  INT16_C(  1956),  INT16_C(     0),  INT16_C(  8051),  INT16_C(     0),  INT16_C( 10333),  INT16_C( 20806),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 29819), -INT16_C( 15843), -INT16_C( 29811),  INT16_C(     0),  INT16_C(     0), -INT16_C( 13497),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1703), -INT16_C( 27834),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17931),
         INT16_C(     0), -INT16_C( 12811),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 24859),  INT16_C(     0) } },
    { {  INT16_C( 24263), -INT16_C( 19245),  INT16_C(  3884), -INT16_C( 15056),  INT16_C( 26313),  INT16_C( 27276),  INT16_C(  2579), -INT16_C(   770),
         INT16_C( 10256),  INT16_C(  3998),  INT16_C( 29765),  INT16_C( 27249), -INT16_C(  9395), -INT16_C( 14939),  INT16_C(  5031),  INT16_C( 28168),
        -INT16_C(  9359), -INT16_C( 25310),  INT16_C( 21226), -INT16_C( 19358), -INT16_C(  4424), -INT16_C( 13282),  INT16_C(  7416),  INT16_C(  2248),
         INT16_C( 26181), -INT16_C( 30184), -INT16_C( 30246),  INT16_C( 10228), -INT16_C( 26268),  INT16_C(  3052), -INT16_C(  2900),  INT16_C(  7545) },
      UINT32_C(3132792016),
      {  INT16_C(  7406), -INT16_C( 22674), -INT16_C( 29686),  INT16_C(   883),  INT16_C( 15273), -INT16_C(  4597),  INT16_C(  9122),  INT16_C( 31864),
         INT16_C( 27820),  INT16_C(  4260), -INT16_C( 28666), -INT16_C( 19941), -INT16_C( 27259),  INT16_C( 21968), -INT16_C( 30159),  INT16_C(  7951),
         INT16_C( 32423), -INT16_C( 20026),  INT16_C( 14602), -INT16_C( 19532), -INT16_C( 16267),  INT16_C(  6049),  INT16_C(  6627), -INT16_C( 28525),
         INT16_C( 14214), -INT16_C( 29536), -INT16_C( 17208),  INT16_C( 19774),  INT16_C(  3665), -INT16_C( 32094), -INT16_C( 20071),  INT16_C( 16545) },
      {  INT64_C( 1652159849723684911),  INT64_C( 3375942324217286502) },
      {  INT16_C( 24263), -INT16_C( 19245),  INT16_C(  3884), -INT16_C( 15056),  INT16_C(     0),  INT16_C( 27276),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 10256),  INT16_C(  3998),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 14939),  INT16_C(  5031),  INT16_C(     0),
        -INT16_C(  9359),  INT16_C(     0),  INT16_C( 21226),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7416),  INT16_C(     0),
         INT16_C( 26181),  INT16_C(     0), -INT16_C( 30246),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2900),  INT16_C(     0) } },
    { {  INT16_C( 31480), -INT16_C( 16198), -INT16_C(  1738), -INT16_C( 30963), -INT16_C( 20729), -INT16_C( 24567), -INT16_C( 21919), -INT16_C( 28448),
        -INT16_C( 11758), -INT16_C( 19254), -INT16_C( 18312), -INT16_C(  8502), -INT16_C(  1977), -INT16_C(  4313),  INT16_C(   441), -INT16_C( 20194),
        -INT16_C( 10117), -INT16_C( 20110),  INT16_C( 32721), -INT16_C(  9928),  INT16_C( 16687), -INT16_C( 28551),  INT16_C( 23275), -INT16_C(   480),
        -INT16_C(  5332), -INT16_C( 23374),  INT16_C( 31907), -INT16_C(  5502), -INT16_C( 22156),  INT16_C( 11737), -INT16_C(  2134),  INT16_C(  9695) },
      UINT32_C(2715177424),
      {  INT16_C(  3792), -INT16_C(   134), -INT16_C(  2993),  INT16_C( 15247), -INT16_C( 20402),  INT16_C( 31289), -INT16_C(  5221),  INT16_C( 15902),
        -INT16_C( 24473), -INT16_C(  9176),  INT16_C(   329), -INT16_C(  3063), -INT16_C(  5895), -INT16_C( 14055), -INT16_C(  4039),  INT16_C(  2666),
        -INT16_C(  6658),  INT16_C( 19977), -INT16_C( 26151),  INT16_C( 10121), -INT16_C( 15799), -INT16_C(  7007), -INT16_C( 16467),  INT16_C(  5154),
         INT16_C( 19039), -INT16_C( 22288), -INT16_C(  1461),  INT16_C( 17564), -INT16_C( 18718),  INT16_C(  7181),  INT16_C( 30886), -INT16_C( 23514) },
      {  INT64_C( 1251292371324383069),  INT64_C( 2089414856581447229) },
      {  INT16_C( 31480), -INT16_C( 16198), -INT16_C(  1738), -INT16_C( 30963),  INT16_C(     0), -INT16_C( 24567),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 19254), -INT16_C( 18312), -INT16_C(  8502),  INT16_C(     0), -INT16_C(  4313),  INT16_C(     0), -INT16_C( 20194),
        -INT16_C( 10117),  INT16_C(     0),  INT16_C(     0), -INT16_C(  9928),  INT16_C(     0), -INT16_C( 28551),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 23374),  INT16_C( 31907), -INT16_C(  5502), -INT16_C( 22156),  INT16_C(     0), -INT16_C(  2134),  INT16_C(     0) } },
    { { -INT16_C(  4255), -INT16_C( 21052),  INT16_C( 25065), -INT16_C( 13071), -INT16_C(   233), -INT16_C( 16920),  INT16_C(  3703), -INT16_C( 11167),
         INT16_C( 21565),  INT16_C(  1546),  INT16_C( 26575),  INT16_C(  3351),  INT16_C(  3429),  INT16_C(  8951), -INT16_C(  2524), -INT16_C( 31170),
         INT16_C(   742), -INT16_C( 12493),  INT16_C(  9315),  INT16_C( 31387), -INT16_C( 31965), -INT16_C( 26057), -INT16_C( 26223), -INT16_C( 12434),
         INT16_C( 30957), -INT16_C( 17195), -INT16_C(  4897),  INT16_C( 17609), -INT16_C( 15879),  INT16_C(  7782), -INT16_C( 23369), -INT16_C( 25180) },
      UINT32_C( 174970791),
      {  INT16_C(  2299),  INT16_C(  8069), -INT16_C( 17268),  INT16_C(  7609),  INT16_C( 10325),  INT16_C( 17132), -INT16_C( 15968), -INT16_C( 32513),
        -INT16_C( 14162), -INT16_C( 22588),  INT16_C( 11145),  INT16_C( 16837),  INT16_C( 27087),  INT16_C( 30430),  INT16_C( 19264),  INT16_C( 15489),
         INT16_C(  1620), -INT16_C(  8101),  INT16_C(  5314),  INT16_C(  6397), -INT16_C(  5572), -INT16_C(  8870),  INT16_C( 22955),  INT16_C( 22877),
         INT16_C(  8482), -INT16_C( 21759), -INT16_C( 14772),  INT16_C(  7404), -INT16_C( 13520),  INT16_C( 28818),  INT16_C(  4886),  INT16_C( 27308) },
      {  INT64_C( 6409827458447181593), -INT64_C( 3875467700365603278) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 13071), -INT16_C(   233),  INT16_C(     0),  INT16_C(  3703),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3351),  INT16_C(     0),  INT16_C(  8951),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 12493),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31965),  INT16_C(     0),  INT16_C(     0), -INT16_C( 12434),
         INT16_C( 30957),  INT16_C(     0), -INT16_C(  4897),  INT16_C(     0), -INT16_C( 15879),  INT16_C(  7782), -INT16_C( 23369), -INT16_C( 25180) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sll_epi16(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sll_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sll_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int64_t count[2];
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(3130761277),
      {  INT16_C( 26780),  INT16_C(  3232),  INT16_C( 17838), -INT16_C( 22728),  INT16_C( 24616), -INT16_C( 22729),  INT16_C( 26223), -INT16_C(  9801),
         INT16_C(  7959), -INT16_C(  2390),  INT16_C( 26413), -INT16_C( 19078),  INT16_C(  2538),  INT16_C( 10062), -INT16_C(  5719),  INT16_C( 17889),
        -INT16_C( 32175), -INT16_C(   175), -INT16_C( 30265), -INT16_C(  4185), -INT16_C(  8471),  INT16_C( 22934),  INT16_C( 19781),  INT16_C( 23602),
        -INT16_C(  9108), -INT16_C( 26285), -INT16_C( 12989),  INT16_C( 11599), -INT16_C( 25130),  INT16_C( 32596),  INT16_C( 13958), -INT16_C( 10044) },
      { -INT64_C( 8615809817898445384), -INT64_C( 4684285321072802723) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C( 861360624),
      { -INT16_C( 23009), -INT16_C(  2719), -INT16_C( 19133), -INT16_C( 13707),  INT16_C( 14827), -INT16_C( 23646),  INT16_C( 31055), -INT16_C(  4830),
        -INT16_C( 28424),  INT16_C( 21877),  INT16_C( 22164), -INT16_C(  6409), -INT16_C(  2710),  INT16_C( 23204), -INT16_C(  1209),  INT16_C( 26253),
        -INT16_C(  4446), -INT16_C(  6821), -INT16_C( 12124), -INT16_C( 28753),  INT16_C( 20746),  INT16_C( 22835),  INT16_C( 21963), -INT16_C( 15546),
        -INT16_C( 17178),  INT16_C( 31256),  INT16_C(  3858),  INT16_C( 31840),  INT16_C(  1028),  INT16_C( 19414),  INT16_C( 25600), -INT16_C( 23887) },
      { -INT64_C( 1763660777605624494), -INT64_C(  858350183371458168) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(1433349699),
      { -INT16_C( 12483),  INT16_C( 17106), -INT16_C( 22316), -INT16_C( 11123),  INT16_C( 16140),  INT16_C( 24438), -INT16_C(   692),  INT16_C( 10581),
        -INT16_C(  9420), -INT16_C( 17135),  INT16_C( 20884), -INT16_C( 23792),  INT16_C( 10200),  INT16_C(  7063),  INT16_C(  1621), -INT16_C( 27791),
         INT16_C( 17366), -INT16_C( 21803),  INT16_C( 25323), -INT16_C(  1922), -INT16_C(  2911), -INT16_C(  4777), -INT16_C( 21263),  INT16_C(  9751),
         INT16_C( 10376),  INT16_C(  7395), -INT16_C(  3207),  INT16_C( 21183),  INT16_C( 22298),  INT16_C( 28781), -INT16_C(  8611),  INT16_C( 13059) },
      { -INT64_C( 2592565705582979039), -INT64_C( 8041212283578655665) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(2192394760),
      {  INT16_C( 27709),  INT16_C( 22740),  INT16_C( 16835),  INT16_C(  8648), -INT16_C( 13536),  INT16_C( 16724),  INT16_C( 12963), -INT16_C(  8882),
         INT16_C( 21389), -INT16_C(  8775), -INT16_C( 31825), -INT16_C( 18402), -INT16_C( 31389),  INT16_C( 27720), -INT16_C(  2609),  INT16_C(  3310),
        -INT16_C( 15774),  INT16_C(  9572),  INT16_C( 11267),  INT16_C(  9030), -INT16_C( 25609), -INT16_C( 26011), -INT16_C( 19507),  INT16_C( 23160),
         INT16_C( 12551), -INT16_C( 18889),  INT16_C( 21940),  INT16_C(  6254), -INT16_C( 18470), -INT16_C( 22140),  INT16_C( 29356),  INT16_C(  3766) },
      {  INT64_C( 4493319499519629876),  INT64_C( 8808286004375568405) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C( 925987970),
      { -INT16_C( 24630), -INT16_C( 23473), -INT16_C( 11434),  INT16_C(   846),  INT16_C(  1093),  INT16_C( 30993),  INT16_C( 17694),  INT16_C( 26032),
         INT16_C(  3008), -INT16_C( 10844),  INT16_C( 32203),  INT16_C( 16312), -INT16_C(  2610),  INT16_C( 20665), -INT16_C(  5527),  INT16_C( 13191),
        -INT16_C( 10614), -INT16_C(  7976),  INT16_C(  9897), -INT16_C(  4381), -INT16_C(  2774),  INT16_C( 18535),  INT16_C(  6202), -INT16_C(  1362),
         INT16_C( 21027), -INT16_C(  4144), -INT16_C( 30513), -INT16_C( 25298), -INT16_C(  6275), -INT16_C(  6419),  INT16_C( 30162),  INT16_C( 23578) },
      {  INT64_C( 4819731317782278731), -INT64_C( 8770135325163238635) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(1500846730),
      { -INT16_C( 23646),  INT16_C(  8182), -INT16_C(  7029),  INT16_C( 23813),  INT16_C(  8025), -INT16_C( 23367), -INT16_C(  2799),  INT16_C( 10649),
         INT16_C( 32021),  INT16_C( 10859), -INT16_C(  2360),  INT16_C( 11130), -INT16_C( 15314), -INT16_C( 17999),  INT16_C( 10206), -INT16_C( 32750),
         INT16_C(  2506),  INT16_C( 21919), -INT16_C( 23315),  INT16_C( 18098),  INT16_C( 27588), -INT16_C( 10774), -INT16_C( 31647),  INT16_C( 30463),
         INT16_C( 27137), -INT16_C( 13919),  INT16_C(  7008), -INT16_C( 28684), -INT16_C( 23073), -INT16_C( 17080),  INT16_C( 23244), -INT16_C( 26819) },
      {  INT64_C( 4942313014548028515),  INT64_C(  495986420026605834) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(3838804867),
      { -INT16_C( 15458),  INT16_C( 32115), -INT16_C( 17560),  INT16_C( 13626),  INT16_C( 30485),  INT16_C( 31180), -INT16_C( 18349), -INT16_C( 11319),
         INT16_C( 24663),  INT16_C( 25112),  INT16_C( 13025), -INT16_C(  6451), -INT16_C( 20661), -INT16_C( 12564), -INT16_C( 17614), -INT16_C( 12110),
         INT16_C(  9598), -INT16_C(  6579), -INT16_C( 30752), -INT16_C(  2533), -INT16_C(  6146),  INT16_C( 20847),  INT16_C( 14496), -INT16_C(  2267),
         INT16_C( 15768),  INT16_C( 31065),  INT16_C( 10095), -INT16_C( 17825),  INT16_C( 19414),  INT16_C(  2440),  INT16_C( 15110), -INT16_C( 31527) },
      { -INT64_C( 5965151098448959648),  INT64_C( 8576300502038259310) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(3488636768),
      {  INT16_C( 20614),  INT16_C( 23689),  INT16_C(  4763), -INT16_C( 23963),  INT16_C( 16205), -INT16_C( 21210), -INT16_C( 28314),  INT16_C(  5358),
         INT16_C(  9496), -INT16_C( 31039), -INT16_C( 16181), -INT16_C( 21868), -INT16_C( 26141),  INT16_C( 17441),  INT16_C(  4600),  INT16_C( 32275),
        -INT16_C( 25247), -INT16_C(   549),  INT16_C( 16559), -INT16_C(   865), -INT16_C( 14977), -INT16_C(  6743), -INT16_C( 26537),  INT16_C( 28666),
        -INT16_C( 17475), -INT16_C( 30219), -INT16_C( 30341),  INT16_C( 24371),  INT16_C( 21538),  INT16_C(  7075), -INT16_C( 18843), -INT16_C( 14439) },
      {  INT64_C( 3818599163143418963), -INT64_C(  149159470413731800) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_sll_epi16(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_sll_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_mask_sll_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const easysimd__mmask16 k;
    const int16_t a[16];
    const int64_t count[2];
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C( 26794),  INT16_C( 32580),  INT16_C( 18427), -INT16_C( 22764),  INT16_C( 25865), -INT16_C(  3979), -INT16_C(   268), -INT16_C( 21926),
         INT16_C(  6790), -INT16_C( 29160), -INT16_C( 26090),  INT16_C( 25752), -INT16_C(  5910), -INT16_C(  2741), -INT16_C(  3167),  INT16_C( 19390) },
      UINT16_C(  603),
      {  INT16_C( 22218), -INT16_C(  8630),  INT16_C( 21501),  INT16_C( 29507),  INT16_C( 14147), -INT16_C( 25231), -INT16_C(  2079), -INT16_C(  1608),
        -INT16_C( 12667),  INT16_C(  7827),  INT16_C( 32306),  INT16_C( 32262), -INT16_C( 22669),  INT16_C( 12657), -INT16_C( 13069), -INT16_C( 17100) },
      {  INT64_C(                   1),  INT64_C(                   0) },
      { -INT16_C( 21100), -INT16_C( 17260),  INT16_C( 18427), -INT16_C(  6522),  INT16_C( 28294), -INT16_C(  3979), -INT16_C(  4158), -INT16_C( 21926),
         INT16_C(  6790),  INT16_C( 15654), -INT16_C( 26090),  INT16_C( 25752), -INT16_C(  5910), -INT16_C(  2741), -INT16_C(  3167),  INT16_C( 19390) } },
    { {  INT16_C(  8092), -INT16_C(  8239),  INT16_C(  5266),  INT16_C(   791), -INT16_C(  1870),  INT16_C( 27386), -INT16_C( 32526), -INT16_C( 31432),
         INT16_C( 27294), -INT16_C( 23549),  INT16_C( 30440),  INT16_C( 22860),  INT16_C( 16296), -INT16_C(  9179),  INT16_C( 18428), -INT16_C( 26534) },
      UINT16_C(11111),
      { -INT16_C(  1672), -INT16_C( 28865), -INT16_C(  3587), -INT16_C(  2169),  INT16_C( 31067), -INT16_C( 27785),  INT16_C(  5631),  INT16_C(   766),
        -INT16_C(  6470),  INT16_C(  1657),  INT16_C(  8512),  INT16_C( 25925),  INT16_C( 16893),  INT16_C( 22445),  INT16_C(  5338),  INT16_C( 21122) },
      {  INT64_C(                  12),  INT64_C(                   8) },
      {        INT16_MIN, -INT16_C(  4096), -INT16_C( 12288),  INT16_C(   791), -INT16_C(  1870),  INT16_C( 28672), -INT16_C(  4096), -INT16_C( 31432),
        -INT16_C( 24576), -INT16_C( 28672),  INT16_C( 30440),  INT16_C( 20480),  INT16_C( 16296), -INT16_C( 12288),  INT16_C( 18428), -INT16_C( 26534) } },
    { {  INT16_C(  2785),  INT16_C( 26803),  INT16_C(  3586),  INT16_C( 31202), -INT16_C(  7774), -INT16_C( 24433),  INT16_C( 18915),  INT16_C( 23686),
        -INT16_C( 14769), -INT16_C( 27523),  INT16_C( 31276), -INT16_C(  9771), -INT16_C( 20527),  INT16_C( 21485), -INT16_C(  1535), -INT16_C(  7659) },
      UINT16_C(51205),
      {  INT16_C(  1867),  INT16_C( 11734),  INT16_C( 30848),  INT16_C(  3854), -INT16_C(  3816), -INT16_C( 24744), -INT16_C( 22706), -INT16_C( 13467),
        -INT16_C( 28357),  INT16_C(  4422),  INT16_C(  5994),  INT16_C( 22464), -INT16_C( 15765), -INT16_C( 32686),  INT16_C( 22436), -INT16_C(  4280) },
      {  INT64_C(                   6),  INT64_C(                   4) },
      { -INT16_C( 11584),  INT16_C( 26803),  INT16_C(  8192),  INT16_C( 31202), -INT16_C(  7774), -INT16_C( 24433),  INT16_C( 18915),  INT16_C( 23686),
        -INT16_C( 14769), -INT16_C( 27523),  INT16_C( 31276), -INT16_C(  4096), -INT16_C( 20527),  INT16_C( 21485), -INT16_C(  5888), -INT16_C( 11776) } },
    { { -INT16_C(  8676),  INT16_C( 10903), -INT16_C( 20498),  INT16_C( 17948),  INT16_C( 27214), -INT16_C( 19218),  INT16_C( 10549),  INT16_C( 31557),
        -INT16_C( 20422), -INT16_C(  1133), -INT16_C(   505),  INT16_C( 22973),  INT16_C( 24958), -INT16_C( 14672),  INT16_C(  3665),  INT16_C( 28132) },
      UINT16_C(31725),
      { -INT16_C(  9320), -INT16_C( 19413),  INT16_C( 31009),  INT16_C(  3870),  INT16_C( 21293),  INT16_C( 29497),  INT16_C( 29647),  INT16_C( 25123),
         INT16_C( 10862),  INT16_C( 11104), -INT16_C(  8572),  INT16_C( 13453), -INT16_C(  8540), -INT16_C( 30653),  INT16_C( 12363), -INT16_C(  7420) },
      {  INT64_C(                   3),  INT64_C(                  16) },
      { -INT16_C(  9024),  INT16_C( 10903), -INT16_C( 14072),  INT16_C( 30960),  INT16_C( 27214), -INT16_C( 26168), -INT16_C( 24968),  INT16_C(  4376),
         INT16_C( 21360),  INT16_C( 23296), -INT16_C(   505), -INT16_C( 23448), -INT16_C(  2784),  INT16_C( 16920), -INT16_C( 32168),  INT16_C( 28132) } },
    { {  INT16_C( 11415), -INT16_C( 19032), -INT16_C( 10692),  INT16_C( 29961), -INT16_C( 10167),  INT16_C( 27880),  INT16_C( 22330), -INT16_C( 25962),
         INT16_C(  6786),  INT16_C(  3960),  INT16_C(  7247), -INT16_C( 27923),  INT16_C( 14756), -INT16_C( 22334), -INT16_C( 13028), -INT16_C( 19241) },
      UINT16_C(33017),
      {  INT16_C( 13673),  INT16_C( 29270), -INT16_C( 24662), -INT16_C( 27830), -INT16_C( 31733), -INT16_C( 24086),  INT16_C( 27678), -INT16_C( 26948),
         INT16_C(  2940),  INT16_C( 27058),  INT16_C( 22429),  INT16_C( 24482), -INT16_C( 16385), -INT16_C( 10452),  INT16_C(  9587), -INT16_C(  9129) },
      {  INT64_C(                   0),  INT64_C(                  17) },
      {  INT16_C( 13673), -INT16_C( 19032), -INT16_C( 10692), -INT16_C( 27830), -INT16_C( 31733), -INT16_C( 24086),  INT16_C( 27678), -INT16_C( 26948),
         INT16_C(  6786),  INT16_C(  3960),  INT16_C(  7247), -INT16_C( 27923),  INT16_C( 14756), -INT16_C( 22334), -INT16_C( 13028), -INT16_C(  9129) } },
    { {  INT16_C(  1359), -INT16_C( 26292),  INT16_C( 22424), -INT16_C( 32226),  INT16_C( 15608), -INT16_C( 19217),  INT16_C( 27603), -INT16_C( 31297),
         INT16_C( 23764),  INT16_C( 30684), -INT16_C(  9029), -INT16_C(  6346), -INT16_C( 22093),  INT16_C(  2573),  INT16_C( 26757), -INT16_C( 11081) },
      UINT16_C(  877),
      {  INT16_C(  1646), -INT16_C( 29606),  INT16_C( 21128),  INT16_C( 30664), -INT16_C( 25849), -INT16_C( 14622), -INT16_C( 18655), -INT16_C(   733),
        -INT16_C(  8658),  INT16_C( 25817), -INT16_C( 29498), -INT16_C( 11507), -INT16_C( 28010),  INT16_C( 19771), -INT16_C( 22425), -INT16_C( 10928) },
      {  INT64_C(                   5),  INT64_C(                   7) },
      { -INT16_C( 12864), -INT16_C( 26292),  INT16_C( 20736), -INT16_C(  1792),  INT16_C( 15608), -INT16_C(  9152), -INT16_C(  7136), -INT16_C( 31297),
        -INT16_C( 14912), -INT16_C( 25824), -INT16_C(  9029), -INT16_C(  6346), -INT16_C( 22093),  INT16_C(  2573),  INT16_C( 26757), -INT16_C( 11081) } },
    { {  INT16_C( 14177),  INT16_C( 10749),  INT16_C(  1198), -INT16_C( 28219), -INT16_C(  6454), -INT16_C(  4792),  INT16_C( 30435), -INT16_C( 16948),
        -INT16_C( 27942), -INT16_C(  6327), -INT16_C(  8091), -INT16_C( 24455), -INT16_C(  8147),  INT16_C( 32328), -INT16_C(  2123),  INT16_C(  5672) },
      UINT16_C( 9518),
      { -INT16_C(  9152),  INT16_C(  1321), -INT16_C(  2963), -INT16_C( 18965), -INT16_C( 12575), -INT16_C( 21205),  INT16_C(  1419), -INT16_C( 10945),
        -INT16_C( 23316),  INT16_C( 26293), -INT16_C(  7612), -INT16_C( 29370), -INT16_C(   928), -INT16_C( 30332), -INT16_C( 19950),  INT16_C( 21166) },
      {  INT64_C(                  18),  INT64_C(                   8) },
      {  INT16_C( 14177),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  6454),  INT16_C(     0),  INT16_C( 30435), -INT16_C( 16948),
         INT16_C(     0), -INT16_C(  6327),  INT16_C(     0), -INT16_C( 24455), -INT16_C(  8147),  INT16_C(     0), -INT16_C(  2123),  INT16_C(  5672) } },
    { { -INT16_C(   937),  INT16_C( 17100), -INT16_C( 21071), -INT16_C(  8943), -INT16_C( 25509), -INT16_C( 25886), -INT16_C( 12431),  INT16_C(  9791),
        -INT16_C( 31947),  INT16_C( 31497),  INT16_C( 26896), -INT16_C( 27529), -INT16_C( 29966), -INT16_C( 24250), -INT16_C( 10788),  INT16_C( 13433) },
      UINT16_C(17873),
      { -INT16_C( 32138), -INT16_C( 30734),  INT16_C( 19807),  INT16_C( 16932), -INT16_C( 27160),  INT16_C( 10001),  INT16_C( 18108), -INT16_C( 14934),
        -INT16_C( 17471),  INT16_C( 14638),  INT16_C(  8527), -INT16_C( 26941), -INT16_C( 24638),  INT16_C( 15211),  INT16_C( 15571),  INT16_C( 19072) },
      {  INT64_C(                  19),  INT64_C(                  18) },
      {  INT16_C(     0),  INT16_C( 17100), -INT16_C( 21071), -INT16_C(  8943),  INT16_C(     0), -INT16_C( 25886),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 31497),  INT16_C(     0), -INT16_C( 27529), -INT16_C( 29966), -INT16_C( 24250),  INT16_C(     0),  INT16_C( 13433) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sll_epi16(src, test_vec[i].k, a, count);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sll_epi16");
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
    easysimd__m256i r = easysimd_mm256_mask_sll_epi16(src, k, a, count);

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
test_easysimd_mm256_maskz_sll_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int16_t a[16];
    const int64_t count[2];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(59468),
      {  INT16_C(  8823), -INT16_C( 27155),  INT16_C( 24738), -INT16_C( 20247),  INT16_C( 11113), -INT16_C(  4411),  INT16_C( 23719),  INT16_C( 23875),
        -INT16_C( 16252),  INT16_C( 10289),  INT16_C( 17994),  INT16_C( 28680), -INT16_C(  8913),  INT16_C( 21613), -INT16_C( 18019),  INT16_C(  5181) },
      {  INT64_C(                  11),  INT64_C(                  10) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(  4096),  INT16_C( 18432),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14336),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 16384),  INT16_C(     0),  INT16_C( 26624), -INT16_C(  6144), -INT16_C(  6144) } },
    { UINT16_C(32426),
      { -INT16_C( 27765), -INT16_C(  3025), -INT16_C(  2882),  INT16_C( 26082),  INT16_C(  9808), -INT16_C( 10814), -INT16_C(  3098),  INT16_C( 12541),
         INT16_C(  1594),  INT16_C( 27041),  INT16_C(  3811), -INT16_C( 32579), -INT16_C(  1336), -INT16_C( 23403),  INT16_C( 16165), -INT16_C( 20446) },
      {  INT64_C(                  16),  INT64_C(                  19) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(37284),
      { -INT16_C( 31162), -INT16_C( 26890), -INT16_C( 18004), -INT16_C( 28053),  INT16_C( 27052), -INT16_C(  6461),  INT16_C( 25711),  INT16_C( 21071),
         INT16_C(  3442),  INT16_C( 15059),  INT16_C( 26631),  INT16_C( 11486),  INT16_C(   423),  INT16_C( 31196), -INT16_C( 32686), -INT16_C( 26614) },
      {  INT64_C(                   6),  INT64_C(                  16) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 27392),  INT16_C(     0),  INT16_C(     0), -INT16_C( 20288),  INT16_C(     0), -INT16_C( 27712),
         INT16_C( 23680),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 27072),  INT16_C(     0),  INT16_C(     0),  INT16_C(   640) } },
    { UINT16_C(45871),
      { -INT16_C( 25926),  INT16_C( 26182),  INT16_C(  2307),  INT16_C( 29261), -INT16_C( 25491), -INT16_C(  8251), -INT16_C( 26455), -INT16_C( 20198),
        -INT16_C(  2048), -INT16_C( 22563), -INT16_C( 17671),  INT16_C( 19488),  INT16_C( 11066),  INT16_C( 16868),  INT16_C(  4908), -INT16_C(  6411) },
      {  INT64_C(                  10),  INT64_C(                   7) },
      { -INT16_C(  6144),  INT16_C(  6144),  INT16_C(  3072),  INT16_C( 13312),  INT16_C(     0),  INT16_C(  5120),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 29696),  INT16_C(     0),  INT16_C(     0), -INT16_C(  6144), -INT16_C( 28672),  INT16_C(     0), -INT16_C( 11264) } },
    { UINT16_C(45388),
      { -INT16_C( 26300), -INT16_C( 20188), -INT16_C(  5834), -INT16_C(  8304), -INT16_C( 21887), -INT16_C( 32368),  INT16_C( 28323), -INT16_C( 25560),
         INT16_C( 18472),  INT16_C( 25320), -INT16_C( 12941), -INT16_C( 24668), -INT16_C( 26144), -INT16_C( 29051), -INT16_C( 11564),  INT16_C(  6208) },
      {  INT64_C(                  16),  INT64_C(                  17) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(41417),
      {  INT16_C( 22861), -INT16_C( 12671),  INT16_C(  4356), -INT16_C( 22705),  INT16_C( 30591), -INT16_C( 22717),  INT16_C( 11455),  INT16_C( 13066),
        -INT16_C( 20743), -INT16_C(  9774),  INT16_C( 22599),  INT16_C(  7016), -INT16_C( 22486), -INT16_C( 27341), -INT16_C(  1012),  INT16_C( 22839) },
      {  INT64_C(                  11),  INT64_C(                   4) },
      {  INT16_C( 26624),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30720),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2048),  INT16_C( 20480),
        -INT16_C( 14336),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26624),  INT16_C(     0), -INT16_C( 18432) } },
    { UINT16_C(22823),
      {  INT16_C( 30409),  INT16_C( 18688),  INT16_C( 17645), -INT16_C( 21264), -INT16_C(  1424),  INT16_C( 27103), -INT16_C( 19800), -INT16_C(  4286),
        -INT16_C( 22006),  INT16_C( 13322),  INT16_C( 15698),  INT16_C( 24265),  INT16_C(    57), -INT16_C( 28745), -INT16_C(  8520), -INT16_C( 32024) },
      {  INT64_C(                   9),  INT64_C(                   1) },
      { -INT16_C( 28160),  INT16_C(     0), -INT16_C(  9728),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16896),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  5120),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28160),  INT16_C( 29184),  INT16_C(     0),  INT16_C( 28672),  INT16_C(     0) } },
    { UINT16_C(16843),
      { -INT16_C( 17619), -INT16_C( 25106), -INT16_C( 12874),  INT16_C( 24070),  INT16_C( 18559), -INT16_C( 30386),  INT16_C( 22771),  INT16_C( 17853),
        -INT16_C( 30826), -INT16_C( 12380),  INT16_C( 23431),  INT16_C( 16478),  INT16_C( 18234), -INT16_C( 28990), -INT16_C( 29392),  INT16_C( 24016) },
      {  INT64_C(                  13),  INT64_C(                   8) },
      { -INT16_C( 24576), -INT16_C( 16384),  INT16_C(     0), -INT16_C( 16384),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24576), -INT16_C( 24576),
        -INT16_C( 16384),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sll_epi16(test_vec[i].k, a, count);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sll_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    // easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_maskz_sll_epi16(k, a, count);

    // easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm_mask_sll_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int64_t count[2];
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C( 11914), -INT16_C( 14692),  INT16_C(  3684), -INT16_C(  4123), -INT16_C(   427),  INT16_C( 17855), -INT16_C( 20289), -INT16_C( 16908) },
      UINT8_C(136),
      {  INT16_C( 16428), -INT16_C(   332), -INT16_C( 16617),  INT16_C( 27315), -INT16_C(  5141), -INT16_C( 10946), -INT16_C( 20041), -INT16_C(  6817) },
      {  INT64_C(                  18),  INT64_C(                  12) },
      {  INT16_C( 11914), -INT16_C( 14692),  INT16_C(  3684),  INT16_C(     0), -INT16_C(   427),  INT16_C( 17855), -INT16_C( 20289),  INT16_C(     0) } },
    { {  INT16_C( 23369),  INT16_C( 14602),  INT16_C(  2225), -INT16_C(  2312), -INT16_C( 22073), -INT16_C( 31254),  INT16_C(  5681), -INT16_C(  6715) },
      UINT8_C( 20),
      { -INT16_C( 23076),  INT16_C( 18376), -INT16_C( 19568),  INT16_C( 25989),  INT16_C( 13931),  INT16_C( 20676), -INT16_C(  5757), -INT16_C(  8294) },
      {  INT64_C(                   9),  INT64_C(                   8) },
      {  INT16_C( 23369),  INT16_C( 14602),  INT16_C(  8192), -INT16_C(  2312), -INT16_C( 10752), -INT16_C( 31254),  INT16_C(  5681), -INT16_C(  6715) } },
    { { -INT16_C(  1136), -INT16_C( 31029),  INT16_C( 29891),  INT16_C( 18544), -INT16_C( 31067), -INT16_C( 29939), -INT16_C(  5734),  INT16_C( 25136) },
      UINT8_C( 48),
      {  INT16_C(  5824),  INT16_C(  9653), -INT16_C(  5247), -INT16_C( 11799), -INT16_C( 11665),  INT16_C( 20075),  INT16_C( 16069), -INT16_C( 16162) },
      {  INT64_C(                   2),  INT64_C(                   0) },
      { -INT16_C(  1136), -INT16_C( 31029),  INT16_C( 29891),  INT16_C( 18544),  INT16_C( 18876),  INT16_C( 14764), -INT16_C(  5734),  INT16_C( 25136) } },
    { {  INT16_C( 32387), -INT16_C( 13356),  INT16_C( 23076), -INT16_C( 20520), -INT16_C( 15628),  INT16_C( 22495), -INT16_C( 24590), -INT16_C( 22419) },
      UINT8_C(196),
      { -INT16_C( 27666), -INT16_C( 16467),  INT16_C( 32514),  INT16_C( 20523),  INT16_C( 26948),  INT16_C(  1070), -INT16_C( 28045), -INT16_C(  3448) },
      {  INT64_C(                   7),  INT64_C(                   5) },
      {  INT16_C( 32387), -INT16_C( 13356), -INT16_C( 32512), -INT16_C( 20520), -INT16_C( 15628),  INT16_C( 22495),  INT16_C( 14720),  INT16_C( 17408) } },
    { { -INT16_C( 16362), -INT16_C( 15060), -INT16_C(  4427),  INT16_C(  3236),  INT16_C( 17376), -INT16_C( 30599),  INT16_C( 26375), -INT16_C( 19428) },
      UINT8_C( 38),
      {  INT16_C( 13086),  INT16_C( 28497), -INT16_C( 17545),  INT16_C( 31645),  INT16_C( 12334),  INT16_C(  8195),  INT16_C( 22422),  INT16_C( 22326) },
      {  INT64_C(                  15),  INT64_C(                   9) },
      { -INT16_C( 16362),        INT16_MIN,        INT16_MIN,  INT16_C(  3236),  INT16_C( 17376),        INT16_MIN,  INT16_C( 26375), -INT16_C( 19428) } },
    { {  INT16_C( 28940),  INT16_C(  6303), -INT16_C(  7599), -INT16_C(  9583), -INT16_C(  1815), -INT16_C( 25098),  INT16_C(  5150),  INT16_C( 28880) },
      UINT8_C(131),
      {  INT16_C( 11079), -INT16_C( 15583),  INT16_C( 20825),  INT16_C( 31430),  INT16_C(  7655),  INT16_C( 16048), -INT16_C( 21344),  INT16_C(  4426) },
      {  INT64_C(                  14),  INT64_C(                  15) },
      { -INT16_C( 16384),  INT16_C( 16384), -INT16_C(  7599), -INT16_C(  9583), -INT16_C(  1815), -INT16_C( 25098),  INT16_C(  5150),        INT16_MIN } },
    { {  INT16_C( 11875),  INT16_C( 15859), -INT16_C(  5353), -INT16_C( 19149),  INT16_C( 18186),  INT16_C( 31365), -INT16_C( 12853), -INT16_C(  4955) },
      UINT8_C(144),
      {  INT16_C( 15870),  INT16_C( 30806),  INT16_C( 29732),  INT16_C( 25385), -INT16_C( 10988),  INT16_C(  9901),  INT16_C(  4128),  INT16_C( 20105) },
      {  INT64_C(                   3),  INT64_C(                  19) },
      {  INT16_C( 11875),  INT16_C( 15859), -INT16_C(  5353), -INT16_C( 19149), -INT16_C( 22368),  INT16_C( 31365), -INT16_C( 12853),  INT16_C( 29768) } },
    { { -INT16_C(  4250),  INT16_C(  7161),  INT16_C( 16633),  INT16_C( 29600),  INT16_C( 27915), -INT16_C(  2280),  INT16_C(  5885),  INT16_C( 21556) },
      UINT8_C(143),
      { -INT16_C( 14247), -INT16_C( 17224), -INT16_C( 29220),  INT16_C(   617),  INT16_C( 31149), -INT16_C(   885),  INT16_C( 20861),  INT16_C( 27746) },
      {  INT64_C(                   9),  INT64_C(                   9) },
      { -INT16_C( 19968),  INT16_C( 28672), -INT16_C( 18432), -INT16_C( 11776),  INT16_C( 27915), -INT16_C(  2280),  INT16_C(  5885), -INT16_C( 15360) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sll_epi16(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_sll_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_mask_sll_epi16(src, k, a, count);

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
test_easysimd_mm_maskz_sll_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int64_t count[2];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C( 45),
      { -INT16_C( 22695),  INT16_C( 14934), -INT16_C( 29152), -INT16_C( 14651),  INT16_C( 27797),  INT16_C(  2362), -INT16_C(  2519),  INT16_C(  5131) },
      {  INT64_C(                  17),  INT64_C(                  12) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(113),
      {  INT16_C( 22373), -INT16_C( 28451),  INT16_C( 12751), -INT16_C(  3241), -INT16_C(  3020),  INT16_C( 25147), -INT16_C(  7602), -INT16_C( 30536) },
      {  INT64_C(                   9),  INT64_C(                   0) },
      { -INT16_C( 13824),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 26624),  INT16_C( 30208), -INT16_C( 25600),  INT16_C(     0) } },
    { UINT8_C( 77),
      { -INT16_C(  9015),  INT16_C(   953), -INT16_C(  7450), -INT16_C(  3591), -INT16_C(   521),  INT16_C( 26637),  INT16_C( 25698), -INT16_C(  3514) },
      {  INT64_C(                   9),  INT64_C(                   3) },
      { -INT16_C( 28160),  INT16_C(     0), -INT16_C( 13312), -INT16_C(  3584),  INT16_C(     0),  INT16_C(     0), -INT16_C( 15360),  INT16_C(     0) } },
    { UINT8_C( 74),
      { -INT16_C( 21466),  INT16_C( 25150), -INT16_C( 29682), -INT16_C( 14780),  INT16_C( 18196),  INT16_C( 24845), -INT16_C(  5616),  INT16_C(  4891) },
      {  INT64_C(                   9),  INT64_C(                  10) },
      {  INT16_C(     0),  INT16_C( 31744),  INT16_C(     0), -INT16_C( 30720),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8192),  INT16_C(     0) } },
    { UINT8_C( 13),
      { -INT16_C(  2879), -INT16_C( 12790),  INT16_C( 27741), -INT16_C( 23758),  INT16_C( 25950), -INT16_C( 22502), -INT16_C( 14709), -INT16_C(  4633) },
      {  INT64_C(                   0),  INT64_C(                  17) },
      { -INT16_C(  2879),  INT16_C(     0),  INT16_C( 27741), -INT16_C( 23758),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 50),
      { -INT16_C( 30565), -INT16_C( 22407), -INT16_C( 30231),  INT16_C(  1170),  INT16_C( 25244), -INT16_C( 22270), -INT16_C(  2525), -INT16_C(  3661) },
      {  INT64_C(                  18),  INT64_C(                  10) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 35),
      {  INT16_C( 32502),  INT16_C(  4488),  INT16_C(  5158),  INT16_C(  3543), -INT16_C( 21503),  INT16_C( 13185),  INT16_C(  2375), -INT16_C(  4180) },
      {  INT64_C(                   0),  INT64_C(                   2) },
      {  INT16_C( 32502),  INT16_C(  4488),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 13185),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(130),
      { -INT16_C( 11529), -INT16_C(  1564),  INT16_C(  2171),  INT16_C( 12271),  INT16_C( 17401),  INT16_C(  7502), -INT16_C( 13255),  INT16_C( 19109) },
      {  INT64_C(                   6),  INT64_C(                   3) },
      {  INT16_C(     0),  INT16_C( 30976),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22208) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sll_epi16(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_sll_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_maskz_sll_epi16(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm256_mask_sll_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int64_t count[2];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   654944826), -INT32_C(  1911993306), -INT32_C(  1989719034), -INT32_C(   256653271),  INT32_C(    47343960),  INT32_C(   523635590),  INT32_C(   834474472),  INT32_C(   208069446) },
      UINT8_C( 56),
      { -INT32_C(  1470176164),  INT32_C(   984542446),  INT32_C(   476264532),  INT32_C(  1416909803),  INT32_C(   836400678), -INT32_C(  1105593940), -INT32_C(  1727771978),  INT32_C(   215093680) },
      {  INT64_C(                  13),  INT64_C(                   6) },
      { -INT32_C(   654944826), -INT32_C(  1911993306), -INT32_C(  1989719034), -INT32_C(  1971494912),  INT32_C(  1321517056),  INT32_C(  1060470784),  INT32_C(   834474472),  INT32_C(   208069446) } },
    { {  INT32_C(  1662903221), -INT32_C(  2120519395),  INT32_C(    30705293), -INT32_C(  1250362662),  INT32_C(  1152263211), -INT32_C(   410032670), -INT32_C(   789037058),  INT32_C(    16903500) },
      UINT8_C(209),
      { -INT32_C(  1880202210), -INT32_C(  2044956928),  INT32_C(  1046551876), -INT32_C(  1167452523),  INT32_C(   698134212),  INT32_C(  2099807037),  INT32_C(  1774844027),  INT32_C(   389729017) },
      {  INT64_C(                  18),  INT64_C(                   7) },
      { -INT32_C(  1871183872), -INT32_C(  2120519395),  INT32_C(    30705293), -INT32_C(  1250362662), -INT32_C(  1156579328), -INT32_C(   410032670), -INT32_C(   504627200),  INT32_C(   736362496) } },
    { { -INT32_C(  1030213978),  INT32_C(   383835317), -INT32_C(  2077461222), -INT32_C(   869011409),  INT32_C(  1112502298), -INT32_C(  1237660691),  INT32_C(  1853895732), -INT32_C(   225005749) },
      UINT8_C(221),
      {  INT32_C(   177386542),  INT32_C(   170240148), -INT32_C(   986076716),  INT32_C(  1306461660),  INT32_C(   540680533), -INT32_C(  1890258597),  INT32_C(   551273073), -INT32_C(  2013344424) },
      {  INT64_C(                   9),  INT64_C(                  12) },
      {  INT32_C(   627596288),  INT32_C(   383835317),  INT32_C(  1934862336), -INT32_C(  1106528256),  INT32_C(  1950525952), -INT32_C(  1237660691), -INT32_C(  1216028160), -INT32_C(    40194048) } },
    { { -INT32_C(  1237772655),  INT32_C(  1499467040), -INT32_C(  1319158574), -INT32_C(   992824183),  INT32_C(   682962644), -INT32_C(  1729485123),  INT32_C(  1164264007),  INT32_C(  1540745161) },
      UINT8_C(253),
      {  INT32_C(   471666958), -INT32_C(  1360103823), -INT32_C(  1992843051), -INT32_C(  1604453518), -INT32_C(   681671248), -INT32_C(  1306593681),  INT32_C(  1132159836),  INT32_C(  1195431480) },
      {  INT64_C(                  10),  INT64_C(                   2) },
      {  INT32_C(  1950627840),  INT32_C(  1499467040), -INT32_C(   561818624),  INT32_C(  2012071936),  INT32_C(  2048311296),  INT32_C(  2077867008), -INT32_C(   309497856),  INT32_C(    56156160) } },
    { {  INT32_C(  1372805475),  INT32_C(  1072802055), -INT32_C(  1875221454),  INT32_C(  1645603588), -INT32_C(   531003966), -INT32_C(  1304185545),  INT32_C(   948534264),  INT32_C(   630550978) },
      UINT8_C(202),
      {  INT32_C(   299005800), -INT32_C(   867954328),  INT32_C(   919721035),  INT32_C(  1861759977),  INT32_C(  1101387916), -INT32_C(  1757849573),  INT32_C(  1381593569),  INT32_C(  1864138502) },
      {  INT64_C(                  13),  INT64_C(                   6) },
      {  INT32_C(  1372805475), -INT32_C(  2110980096), -INT32_C(  1875221454),  INT32_C(   108863488), -INT32_C(   531003966), -INT32_C(  1304185545),  INT32_C(   775692288), -INT32_C(  1881096192) } },
    { { -INT32_C(   989897088), -INT32_C(    57128149),  INT32_C(  2049933953), -INT32_C(  1772962576),  INT32_C(   938372861),  INT32_C(  1588121349),  INT32_C(  1037938465), -INT32_C(  1641229538) },
      UINT8_C( 50),
      {  INT32_C(  1985831723),  INT32_C(  2113427963),  INT32_C(  1148088968),  INT32_C(   843187396), -INT32_C(  1053328910), -INT32_C(   790456799),  INT32_C(  1206787955),  INT32_C(  2004454731) },
      {  INT64_C(                   5),  INT64_C(                   6) },
      { -INT32_C(   989897088), -INT32_C(  1089781920),  INT32_C(  2049933953), -INT32_C(  1772962576),  INT32_C(   653213248),  INT32_C(   475186208),  INT32_C(  1037938465), -INT32_C(  1641229538) } },
    { { -INT32_C(   449844243), -INT32_C(   682117271),  INT32_C(  1054547196), -INT32_C(  2034774706),  INT32_C(  1880938638), -INT32_C(  1752199255),  INT32_C(  1327815638),  INT32_C(  1059394642) },
      UINT8_C(  0),
      {  INT32_C(   208217428), -INT32_C(  1744289668), -INT32_C(   387496421), -INT32_C(   713658883),  INT32_C(   410969737),  INT32_C(  1357846133),  INT32_C(  1336032826), -INT32_C(  1202724252) },
      {  INT64_C(                   5),  INT64_C(                   7) },
      { -INT32_C(   449844243), -INT32_C(   682117271),  INT32_C(  1054547196), -INT32_C(  2034774706),  INT32_C(  1880938638), -INT32_C(  1752199255),  INT32_C(  1327815638),  INT32_C(  1059394642) } },
    { { -INT32_C(   856128572),  INT32_C(    51516188),  INT32_C(  1903169531),  INT32_C(  1666709989),  INT32_C(     7982097), -INT32_C(  1086344163),  INT32_C(  1386324739),  INT32_C(   537569371) },
      UINT8_C( 44),
      {  INT32_C(   357100546),  INT32_C(   235949055), -INT32_C(  1259109957), -INT32_C(  1530570792), -INT32_C(  2067675440), -INT32_C(  1467514619), -INT32_C(   888940254), -INT32_C(   436788253) },
      {  INT64_C(                  16),  INT64_C(                  10) },
      { -INT32_C(   856128572),  INT32_C(    51516188), -INT32_C(  2118451200),  INT32_C(  1473773568),  INT32_C(     7982097), -INT32_C(  2130378752),  INT32_C(  1386324739),  INT32_C(   537569371) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sll_epi32(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_sll_epi32");
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
    easysimd__m256i r = easysimd_mm256_mask_sll_epi32(src, k, a, count);

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
test_easysimd_mm256_maskz_sll_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int64_t count[2];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(128),
      {  INT32_C(  1875855994),  INT32_C(   213130653),  INT32_C(  1276229329),  INT32_C(    74862193),  INT32_C(  1247398616),  INT32_C(   748006454),  INT32_C(   136760817), -INT32_C(  1853298154) },
      {  INT64_C(                  18),  INT64_C(                  19) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1470627840) } },
    { UINT8_C(  0),
      { -INT32_C(   558533167), -INT32_C(  1829803194), -INT32_C(   536330788),  INT32_C(  1597624853),  INT32_C(   888530184), -INT32_C(   564477226),  INT32_C(  1785087193),  INT32_C(  1248509561) },
      {  INT64_C(                   9),  INT64_C(                   2) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 40),
      {  INT32_C(   202935161),  INT32_C(  1209292135), -INT32_C(   897456277),  INT32_C(   616523759),  INT32_C(   478052925), -INT32_C(   981286053), -INT32_C(  2043723716), -INT32_C(   307277965) },
      {  INT64_C(                   3),  INT64_C(                   3) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   637222776),  INT32_C(           0),  INT32_C(   739646168),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(249),
      { -INT32_C(  1995629503),  INT32_C(  1376515207),  INT32_C(   135361252), -INT32_C(   880372817), -INT32_C(  1286714898),  INT32_C(   216239749), -INT32_C(   272944638),  INT32_C(  1307083276) },
      {  INT64_C(                   7),  INT64_C(                   3) },
      { -INT32_C(  2037505920),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1018570880), -INT32_C(  1490749696),  INT32_C(  1908884096), -INT32_C(   577175296), -INT32_C(   197065216) } },
    { UINT8_C(215),
      {  INT32_C(  2028136998), -INT32_C(   695642930), -INT32_C(   816046845), -INT32_C(  1088575989),  INT32_C(   933407531), -INT32_C(  1661807443),  INT32_C(   763655392),  INT32_C(   956594707) },
      {  INT64_C(                  14),  INT64_C(                   8) },
      { -INT32_C(  1165393920),  INT32_C(  1429438464),  INT32_C(   121683968),  INT32_C(           0), -INT32_C(  1429553152),  INT32_C(           0),  INT32_C(   490209280),  INT32_C(   512016384) } },
    { UINT8_C(177),
      {  INT32_C(   138033714),  INT32_C(   224679742),  INT32_C(   573180515),  INT32_C(  1841616182),  INT32_C(   526424195), -INT32_C(  1515924360),  INT32_C(   548019943), -INT32_C(  1261334398) },
      {  INT64_C(                  10),  INT64_C(                  16) },
      { -INT32_C(   387397632),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2107503616), -INT32_C(  1823350784),  INT32_C(           0),  INT32_C(  1178732544) } },
    { UINT8_C(188),
      {  INT32_C(   371221257), -INT32_C(   398447674), -INT32_C(   827583135), -INT32_C(  1204927336),  INT32_C(  1650197436),  INT32_C(   185367274), -INT32_C(  1327719172),  INT32_C(  1886185575) },
      {  INT64_C(                   2),  INT64_C(                   0) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   984634756), -INT32_C(   524742048), -INT32_C(  1989144848),  INT32_C(   741469096),  INT32_C(           0), -INT32_C(  1045192292) } },
    { UINT8_C(134),
      { -INT32_C(    70863086), -INT32_C(   408429544),  INT32_C(   739634292),  INT32_C(   176691368),  INT32_C(   236323331), -INT32_C(  1360350210), -INT32_C(  1961176294),  INT32_C(   823240479) },
      {  INT64_C(                   9),  INT64_C(                   2) },
      {  INT32_C(           0),  INT32_C(  1337470976),  INT32_C(   735635456),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   592330240) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sll_epi32(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_sll_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    // easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_maskz_sll_epi32(k, a, count);

    // easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm_mask_sll_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int64_t count[2];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(     6821392), -INT32_C(   375825863), -INT32_C(   884562605),  INT32_C(  1242469132) },
      UINT8_C( 26),
      {  INT32_C(  1382428016),  INT32_C(   272655343), -INT32_C(  1542718419), -INT32_C(  1936443486) },
      {  INT64_C(                  19),  INT64_C(                   0) },
      { -INT32_C(     6821392),  INT32_C(   527958016), -INT32_C(   884562605), -INT32_C(  1123024896) } },
    { { -INT32_C(  1355994939),  INT32_C(  1190515323),  INT32_C(   676694494),  INT32_C(   173393315) },
      UINT8_C( 23),
      {  INT32_C(  1431858501),  INT32_C(  1046500506), -INT32_C(  2083350281), -INT32_C(  1891080602) },
      {  INT64_C(                  19),  INT64_C(                   7) },
      {  INT32_C(  1781006336), -INT32_C(  1529872384), -INT32_C(   944242688),  INT32_C(   173393315) } },
    { {  INT32_C(  1374512394),  INT32_C(  1806071363),  INT32_C(   599812889), -INT32_C(   611252861) },
      UINT8_C( 90),
      {  INT32_C(  1790652203), -INT32_C(  1455664346),  INT32_C(  1223860921), -INT32_C(  1705776843) },
      {  INT64_C(                  15),  INT64_C(                   9) },
      {  INT32_C(  1374512394),  INT32_C(   697499648),  INT32_C(   599812889), -INT32_C(   191201280) } },
    { {  INT32_C(  1212828893), -INT32_C(  1995852442),  INT32_C(   320474680),  INT32_C(   608388713) },
      UINT8_C(176),
      {  INT32_C(   334264169),  INT32_C(  2030407472),  INT32_C(  1607265988),  INT32_C(   289173701) },
      {  INT64_C(                   5),  INT64_C(                  17) },
      {  INT32_C(  1212828893), -INT32_C(  1995852442),  INT32_C(   320474680),  INT32_C(   608388713) } },
    { {  INT32_C(    26111863), -INT32_C(  1155818328),  INT32_C(   704536837), -INT32_C(    23042031) },
      UINT8_C(123),
      {  INT32_C(  1249938897), -INT32_C(  1340707247),  INT32_C(  1173194291), -INT32_C(  1346604480) },
      {  INT64_C(                   9),  INT64_C(                  19) },
      {  INT32_C(    18588160),  INT32_C(   752656896),  INT32_C(   704536837),  INT32_C(  2028240896) } },
    { {  INT32_C(   333028439), -INT32_C(   888063326),  INT32_C(  1248622924),  INT32_C(  1960263156) },
      UINT8_C(135),
      { -INT32_C(   643964120),  INT32_C(  1539711766),  INT32_C(   303577187), -INT32_C(   731196104) },
      {  INT64_C(                  16),  INT64_C(                  15) },
      { -INT32_C(   483917824),  INT32_C(   588644352),  INT32_C(   946012160),  INT32_C(  1960263156) } },
    { {  INT32_C(  1116662134),  INT32_C(  2142111797),  INT32_C(  1901521916),  INT32_C(   257195634) },
      UINT8_C( 87),
      { -INT32_C(   971165078), -INT32_C(  1478601323),  INT32_C(  1645327245),  INT32_C(  1272483426) },
      {  INT64_C(                   2),  INT64_C(                   3) },
      {  INT32_C(   410306984), -INT32_C(  1619437996), -INT32_C(  2008625612),  INT32_C(   257195634) } },
    { { -INT32_C(     3660416), -INT32_C(  1839877343),  INT32_C(   887542565),  INT32_C(  1214665003) },
      UINT8_C( 23),
      { -INT32_C(  1544184069), -INT32_C(  1917540309),  INT32_C(  1432765194), -INT32_C(  2049605536) },
      {  INT64_C(                   3),  INT64_C(                  10) },
      {  INT32_C(   531429336),  INT32_C(  1839546712), -INT32_C(  1422780336),  INT32_C(  1214665003) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sll_epi32(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_sll_epi32");
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
    easysimd__m128i r = easysimd_mm_mask_sll_epi32(src, k, a, count);

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
test_easysimd_mm_maskz_sll_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int64_t count[2];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(227),
      { -INT32_C(   971254521), -INT32_C(  2055210027), -INT32_C(   129857725), -INT32_C(   707538364) },
      {  INT64_C(                   8),  INT64_C(                  12) },
      {  INT32_C(   466945792),  INT32_C(  2147210496),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(165),
      { -INT32_C(   595011023), -INT32_C(   347885261),  INT32_C(   787763531),  INT32_C(   508217688) },
      {  INT64_C(                   3),  INT64_C(                   3) },
      { -INT32_C(   465120888),  INT32_C(           0),  INT32_C(  2007140952),  INT32_C(           0) } },
    { UINT8_C(158),
      {  INT32_C(  1713735716),  INT32_C(  1412983169), -INT32_C(   469626818),  INT32_C(   493673537) },
      {  INT64_C(                   3),  INT64_C(                   3) },
      {  INT32_C(           0), -INT32_C(  1581036536),  INT32_C(   537952752), -INT32_C(   345579000) } },
    { UINT8_C( 97),
      { -INT32_C(   273520133), -INT32_C(   507966569), -INT32_C(   953722839), -INT32_C(   487805060) },
      {  INT64_C(                  11),  INT64_C(                   2) },
      { -INT32_C(  1823483904),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 27),
      { -INT32_C(  2043964284), -INT32_C(   463311752), -INT32_C(   352226678),  INT32_C(  1553820269) },
      {  INT64_C(                   2),  INT64_C(                   4) },
      {  INT32_C(   414077456), -INT32_C(  1853247008),  INT32_C(           0),  INT32_C(  1920313780) } },
    { UINT8_C( 21),
      {  INT32_C(   174969571), -INT32_C(  2034570855), -INT32_C(   405017908), -INT32_C(   183341201) },
      {  INT64_C(                  10),  INT64_C(                  13) },
      { -INT32_C(  1219785728),  INT32_C(           0),  INT32_C(  1873489920),  INT32_C(           0) } },
    { UINT8_C( 88),
      { -INT32_C(   842397493), -INT32_C(  1858652173), -INT32_C(  1455867501),  INT32_C(   655756061) },
      {  INT64_C(                  11),  INT64_C(                   9) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1336350720) } },
    { UINT8_C(225),
      {  INT32_C(   114084907), -INT32_C(   982107213), -INT32_C(  2008851152), -INT32_C(   162378711) },
      {  INT64_C(                  11),  INT64_C(                  19) },
      {  INT32_C(  1717655552),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sll_epi32(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_sll_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_maskz_sll_epi32(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm256_mask_sll_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t count[2];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 6623669887090757490), -INT64_C(  651153842592560678),  INT64_C(  442712070863664377),  INT64_C( 2955757302328805754) },
      UINT8_C(229),
      { -INT64_C( 2639709635476783228),  INT64_C( 4512282512116093697),  INT64_C( 8413722646386444148),  INT64_C( 8861458945866199583) },
      {  INT64_C(                  13),  INT64_C(                  43) },
      { -INT64_C( 4917279438213709824), -INT64_C(  651153842592560678),  INT64_C( 8180059818865623040),  INT64_C( 2955757302328805754) } },
    { {  INT64_C(  679220174243646461),  INT64_C( 4277390182436796343),  INT64_C(  816813978629474208), -INT64_C( 4455610592135984132) },
      UINT8_C(200),
      {  INT64_C(    1962118495476796),  INT64_C( 4671061536963648731), -INT64_C( 3054487786544679495), -INT64_C(  517530519486164340) },
      {  INT64_C(                  41),  INT64_C(                   9) },
      {  INT64_C(  679220174243646461),  INT64_C( 4277390182436796343),  INT64_C(  816813978629474208),  INT64_C( 3829493446427541504) } },
    { { -INT64_C( 5140135486081793465), -INT64_C( 2392147376595180859), -INT64_C( 4093294677289707774),  INT64_C( 3335932463005626043) },
      UINT8_C(158),
      { -INT64_C( 8621480514393375741),  INT64_C( 1613988234082906626),  INT64_C( 2710053410400568623),  INT64_C( 8604753211830249519) },
      {  INT64_C(                  10),  INT64_C(                  48) },
      { -INT64_C( 5140135486081793465), -INT64_C( 7483014932963260416),  INT64_C( 8083081193749527552), -INT64_C( 6276378318990164992) } },
    { { -INT64_C( 8117344271941182708),  INT64_C( 8229985469775485130), -INT64_C( 7102549818068865623),  INT64_C( 7416457004041364087) },
      UINT8_C(104),
      {  INT64_C( 4503699849204256080),  INT64_C( 6081511958696997363), -INT64_C(  252338910579479717), -INT64_C( 7111259343326715814) },
      {  INT64_C(                   0),  INT64_C(                   9) },
      { -INT64_C( 8117344271941182708),  INT64_C( 8229985469775485130), -INT64_C( 7102549818068865623), -INT64_C( 7111259343326715814) } },
    { {  INT64_C( 6450678507175849556), -INT64_C( 6470102834485957595), -INT64_C( 6883043626697732454), -INT64_C( 5127941479417002133) },
      UINT8_C(146),
      {  INT64_C( 9212913562605817163), -INT64_C( 1784016329842400241), -INT64_C( 7565815580701075952), -INT64_C( 2619770552352720339) },
      {  INT64_C(                  47),  INT64_C(                   4) },
      {  INT64_C( 6450678507175849556), -INT64_C( 5618381272633049088), -INT64_C( 6883043626697732454), -INT64_C( 5127941479417002133) } },
    { {  INT64_C( 1355032441034331280), -INT64_C( 9133290904969779283), -INT64_C( 6604668866833215075),  INT64_C( 6006001741296428946) },
      UINT8_C(238),
      { -INT64_C(  861361204361052588),  INT64_C( 5148393063823177164),  INT64_C( 5632195977360117527),  INT64_C(  898015919409815533) },
      {  INT64_C(                  29),  INT64_C(                   2) },
      {  INT64_C( 1355032441034331280), -INT64_C( 2690761442172338176), -INT64_C( 3243187792099934208),  INT64_C( 1216583217654530048) } },
    { { -INT64_C( 1842859950929629333), -INT64_C( 2869142705585610067), -INT64_C(  873318023347530664),  INT64_C( 1265083400186272481) },
      UINT8_C( 41),
      { -INT64_C( 7629182480774559590),  INT64_C(   33310169114820188),  INT64_C( 8358354186443946180),  INT64_C( 9138414854977770798) },
      {  INT64_C(                  48),  INT64_C(                  44) },
      { -INT64_C( 8603564138137911296), -INT64_C( 2869142705585610067), -INT64_C(  873318023347530664),  INT64_C( 6137843342152564736) } },
    { { -INT64_C( 4878051244099131408), -INT64_C( 7189477039947859509), -INT64_C( 6896803753342331915), -INT64_C( 7224759329293062838) },
      UINT8_C( 39),
      { -INT64_C( 5726571053293721075),  INT64_C( 1513779030634711093),  INT64_C( 7882772968820700793),  INT64_C( 5525551973477745160) },
      {  INT64_C(                  28),  INT64_C(                  49) },
      {  INT64_C( 8914394566971359232),  INT64_C( 2366268334212644864),  INT64_C(  124340916247855104), -INT64_C( 7224759329293062838) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sll_epi64(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_sll_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 50.0);
    easysimd__m256i r = easysimd_mm256_mask_sll_epi64(src, k, a, count);

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
test_easysimd_mm256_maskz_sll_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t count[2];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(146),
      {  INT64_C( 1923040702451633329), -INT64_C( 6250895126475284475),  INT64_C( 3831958393563397660),  INT64_C( 6926171942993605065) },
      {  INT64_C(                  37),  INT64_C(                  34) },
      {  INT64_C(                   0),  INT64_C( 5119467563608178688),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(231),
      { -INT64_C( 5789386780942311340),  INT64_C( 6488893355801445368),  INT64_C( 4298890010939614729), -INT64_C( 5715148892495324056) },
      {  INT64_C(                  17),  INT64_C(                  35) },
      { -INT64_C( 1239935554516680704),  INT64_C( 6647669154460467200),  INT64_C( 8313782418927648768),  INT64_C(                   0) } },
    { UINT8_C( 31),
      { -INT64_C( 2874201227652612484), -INT64_C( 7941028859927508247), -INT64_C( 5399369634321693572),  INT64_C( 6611754700652201969) },
      {  INT64_C(                  11),  INT64_C(                  25) },
      { -INT64_C( 1852754719203401728),  INT64_C( 6801167880287635456), -INT64_C( 8309310938807017472),  INT64_C(  963476832898746368) } },
    { UINT8_C(135),
      {  INT64_C( 9142005131144024096), -INT64_C( 2518692900468315584),  INT64_C( 3806868325094301882), -INT64_C( 8469445254784385132) },
      {  INT64_C(                   9),  INT64_C(                  18) },
      { -INT64_C( 4766367576485773312),  INT64_C( 1701320119891034112), -INT64_C( 6238289364929907712),  INT64_C(                   0) } },
    { UINT8_C(130),
      {  INT64_C( 1787377400097539590), -INT64_C( 4108193985696916432),  INT64_C( 3926603720155640405), -INT64_C( 3634467816034403709) },
      {  INT64_C(                  34),  INT64_C(                  49) },
      {  INT64_C(                   0),  INT64_C( 4075318682752909312),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 82),
      {  INT64_C( 4129444801938520595), -INT64_C( 1048520230517272046),  INT64_C( 2022795797001636911), -INT64_C( 5093400037016479439) },
      {  INT64_C(                  41),  INT64_C(                  31) },
      {  INT64_C(                   0),  INT64_C( 7648277649600544768),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(229),
      {  INT64_C( 1670630127860657345), -INT64_C( 6049154231925336147), -INT64_C( 6346531025958570203), -INT64_C( 2315861626156775363) },
      {  INT64_C(                   0),  INT64_C(                  35) },
      {  INT64_C( 1670630127860657345),  INT64_C(                   0), -INT64_C( 6346531025958570203),  INT64_C(                   0) } },
    { UINT8_C(217),
      { -INT64_C( 6884561258460719661), -INT64_C( 1021994202278556826), -INT64_C( 7103190746670470963),  INT64_C( 4340212647811732611) },
      {  INT64_C(                  20),  INT64_C(                  27) },
      {  INT64_C(   13141935769255936),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 8753265269273526272) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sll_epi64(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_sll_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 50.0);
    easysimd__m256i r = easysimd_mm256_maskz_sll_epi64(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm_mask_sll_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t count[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 5334227597772866860),  INT64_C( 4078513106948137401) },
      UINT8_C( 61),
      {  INT64_C( 1594799479029870184),  INT64_C( 8996513340389188472) },
      {  INT64_C(                   4),  INT64_C(                  40) },
      {  INT64_C( 7070047590768371328),  INT64_C( 4078513106948137401) } },
    { { -INT64_C( 1445937966779376840), -INT64_C( 9010315316755580914) },
      UINT8_C( 33),
      { -INT64_C( 8670232094883815262),  INT64_C( 8147636346654144093) },
      {  INT64_C(                  20),  INT64_C(                   3) },
      {  INT64_C( 6740626566203572224), -INT64_C( 9010315316755580914) } },
    { {  INT64_C( 7707411736193799557),  INT64_C( 5121155746446107440) },
      UINT8_C(184),
      { -INT64_C( 6641933373908095765),  INT64_C( 3974788559272217650) },
      {  INT64_C(                  45),  INT64_C(                  30) },
      {  INT64_C( 7707411736193799557),  INT64_C( 5121155746446107440) } },
    { { -INT64_C( 1647345260112640422),  INT64_C( 4696382347228685176) },
      UINT8_C( 88),
      {  INT64_C( 3768730656858463733),  INT64_C( 7632467100033718106) },
      {  INT64_C(                   6),  INT64_C(                   3) },
      { -INT64_C( 1647345260112640422),  INT64_C( 4696382347228685176) } },
    { {  INT64_C( 3784420571582037152), -INT64_C( 1193077030415995950) },
      UINT8_C(238),
      {  INT64_C( 1338646066156545793),  INT64_C( 1170845827451489078) },
      {  INT64_C(                  28),  INT64_C(                  29) },
      {  INT64_C( 3784420571582037152),  INT64_C( 4179314836330512384) } },
    { {  INT64_C( 6780934820035655557), -INT64_C( 7308549041803005829) },
      UINT8_C(  6),
      {  INT64_C( 4838032182848613652), -INT64_C( 3761267292713594312) },
      {  INT64_C(                  41),  INT64_C(                   9) },
      {  INT64_C( 6780934820035655557),  INT64_C(  406573011672498176) } },
    { {  INT64_C( 3909705892515303445), -INT64_C( 5514696114788822161) },
      UINT8_C(  3),
      {  INT64_C( 6404393618310223898),  INT64_C( 1382230990078159694) },
      {  INT64_C(                  46),  INT64_C(                   1) },
      {  INT64_C( 3892939665396727808),  INT64_C( 5391793916380971008) } },
    { { -INT64_C( 3035716647183491874), -INT64_C( 1829474379096081647) },
      UINT8_C(196),
      {  INT64_C( 5199915045624275372),  INT64_C( 9044097909967476114) },
      {  INT64_C(                   8),  INT64_C(                  20) },
      { -INT64_C( 3035716647183491874), -INT64_C( 1829474379096081647) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sll_epi64(src, k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_sll_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 50.0);
    easysimd__m128i r = easysimd_mm_mask_sll_epi64(src, k, a, count);

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
test_easysimd_mm_maskz_sll_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t count[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(207),
      {  INT64_C( 3660614567508124021),  INT64_C( 4358074517517044524) },
      {  INT64_C(                  34),  INT64_C(                  41) },
      { -INT64_C( 5781348876057313280),  INT64_C(  370470203777089536) } },
    { UINT8_C( 76),
      { -INT64_C( 8658800677483768776), -INT64_C( 7858196791794978255) },
      {  INT64_C(                  13),  INT64_C(                  22) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 96),
      { -INT64_C( 5253420119867311150), -INT64_C( 8264636858642180010) },
      {  INT64_C(                   9),  INT64_C(                  37) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 98),
      {  INT64_C( 3841635642129007179), -INT64_C( 2406750954364538674) },
      {  INT64_C(                  35),  INT64_C(                  22) },
      {  INT64_C(                   0),  INT64_C( 8225691060009959424) } },
    { UINT8_C(246),
      { -INT64_C(  317258937050199437), -INT64_C( 1918166578893202325) },
      {  INT64_C(                  28),  INT64_C(                  49) },
      {  INT64_C(                   0), -INT64_C( 8299934248282882048) } },
    { UINT8_C( 54),
      {  INT64_C( 3641980092192025541),  INT64_C( 3822693769059583687) },
      {  INT64_C(                  26),  INT64_C(                  45) },
      {  INT64_C(                   0), -INT64_C( 1515578805920464896) } },
    { UINT8_C(205),
      { -INT64_C( 8795299021751429481),  INT64_C( 5770392358334136759) },
      {  INT64_C(                   0),  INT64_C(                   4) },
      { -INT64_C( 8795299021751429481),  INT64_C(                   0) } },
    { UINT8_C(218),
      { -INT64_C( 8273715592286845864),  INT64_C(  290709095705004304) },
      {  INT64_C(                  21),  INT64_C(                  43) },
      {  INT64_C(                   0), -INT64_C( 3730160159494766592) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i count = easysimd_mm_loadu_epi64(test_vec[i].count);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sll_epi64(k, a, count);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_sll_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i count = gxz_easysimd_test_x86_random_i64x2(0.0, 50.0);
    easysimd__m128i r = easysimd_mm_maskz_sll_epi64(k, a, count);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, count, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_sll_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int64_t b[2];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   288158111), -INT32_C(  1833315413),  INT32_C(  1495932463), -INT32_C(   583370780), -INT32_C(  1745975803), -INT32_C(  1803847798),  INT32_C(   311362271), -INT32_C(  1679574532),
         INT32_C(  1437339818),  INT32_C(   317220323),  INT32_C(  1751847555), -INT32_C(  1891260790), -INT32_C(  1239010516), -INT32_C(  1991597398), -INT32_C(  1600398940),  INT32_C(  1027309458) },
      {  INT64_C(                  25),  INT64_C(                  24) },
      {  INT32_C(  1040187392),  INT32_C(  1442840576),  INT32_C(  1577058304), -INT32_C(   939524096),  INT32_C(   167772160),  INT32_C(   335544320), -INT32_C(  1107296256), -INT32_C(   134217728),
         INT32_C(  1409286144), -INT32_C(   973078528),  INT32_C(   100663296),  INT32_C(   335544320),  INT32_C(  1476395008),  INT32_C(  1409286144),  INT32_C(  1207959552),  INT32_C(   603979776) } },
    { {  INT32_C(  1517800879),  INT32_C(   333693295),  INT32_C(   766738330), -INT32_C(  1922371842),  INT32_C(   570424533),  INT32_C(    82871159),  INT32_C(   135997554),  INT32_C(  1422508452),
        -INT32_C(    38911602), -INT32_C(  1659858686),  INT32_C(   248169232), -INT32_C(  2036648783), -INT32_C(  1482188240), -INT32_C(  1867802595),  INT32_C(  1687733952),  INT32_C(   381182344) },
      {  INT64_C(                   1),  INT64_C(                  14) },
      { -INT32_C(  1259365538),  INT32_C(   667386590),  INT32_C(  1533476660),  INT32_C(   450223612),  INT32_C(  1140849066),  INT32_C(   165742318),  INT32_C(   271995108), -INT32_C(  1449950392),
        -INT32_C(    77823204),  INT32_C(   975249924),  INT32_C(   496338464),  INT32_C(   221669730),  INT32_C(  1330590816),  INT32_C(   559362106), -INT32_C(   919499392),  INT32_C(   762364688) } },
    { {  INT32_C(  1780008781),  INT32_C(   536527711),  INT32_C(   310678154), -INT32_C(  1775747852), -INT32_C(  1690616669),  INT32_C(  1235451233), -INT32_C(   907887991), -INT32_C(  1170603411),
         INT32_C(   690246346), -INT32_C(  1622597867), -INT32_C(  1515074127), -INT32_C(  1405298167),  INT32_C(  2001172246),  INT32_C(  2126572533),  INT32_C(   306750373),  INT32_C(  1875673765) },
      {  INT64_C(                  24),  INT64_C(                   5) },
      {  INT32_C(  1291845632),  INT32_C(  1593835520), -INT32_C(  1979711488), -INT32_C(   201326592), -INT32_C(  1560281088),  INT32_C(  1627389952), -INT32_C(  1996488704),  INT32_C(  1828716544),
        -INT32_C(   905969664),  INT32_C(   352321536), -INT32_C(  1325400064),  INT32_C(   150994944),  INT32_C(   369098752), -INT32_C(   184549376), -INT32_C(  1526726656), -INT32_C(  1526726656) } },
    { {  INT32_C(   262319130),  INT32_C(  1032741783), -INT32_C(  1420831226),  INT32_C(   739974232),  INT32_C(   487961613),  INT32_C(  1172217494), -INT32_C(   302168615),  INT32_C(    51929832),
        -INT32_C(  1508721905),  INT32_C(   618897438), -INT32_C(   825281674),  INT32_C(  1559947855),  INT32_C(   880349342), -INT32_C(  2022090834), -INT32_C(  2055899235), -INT32_C(   393703975) },
      {  INT64_C(                  22),  INT64_C(                  28) },
      {  INT32_C(   109051904), -INT32_C(   440401920), -INT32_C(  2122317824),  INT32_C(   369098752),  INT32_C(    54525952), -INT32_C(  1518338048),  INT32_C(  1983905792), -INT32_C(  1174405120),
        -INT32_C(  1010827264),  INT32_C(   125829120), -INT32_C(   578813952), -INT32_C(  1816133632),  INT32_C(   662700032), -INT32_C(   343932928), -INT32_C(   415236096),  INT32_C(  1983905792) } },
    { {  INT32_C(   957860235), -INT32_C(  1094610655), -INT32_C(   515688952),  INT32_C(   214617283), -INT32_C(  1569564313),  INT32_C(  1901395403), -INT32_C(  1687825065), -INT32_C(   612452784),
        -INT32_C(  1609263489),  INT32_C(   794744103), -INT32_C(   820993525), -INT32_C(   707011986), -INT32_C(     8958669), -INT32_C(  1586443190), -INT32_C(   247605855),  INT32_C(   197966731) },
      {  INT64_C(                  24),  INT64_C(                  24) },
      { -INT32_C(  1962934272),  INT32_C(   553648128),  INT32_C(   134217728), -INT32_C(  1023410176),  INT32_C(  1728053248), -INT32_C(   889192448),  INT32_C(  1459617792),  INT32_C(  1342177280),
         INT32_C(  2130706432),  INT32_C(   654311424),  INT32_C(   184549376),  INT32_C(  1845493760),  INT32_C(   855638016),  INT32_C(  1241513984), -INT32_C(  1593835520), -INT32_C(  1962934272) } },
    { {  INT32_C(    63268537), -INT32_C(   744147662), -INT32_C(  1765481974), -INT32_C(   274624355),  INT32_C(   661081201),  INT32_C(    48762710),  INT32_C(  1495038407), -INT32_C(  1658909724),
        -INT32_C(  1532894094),  INT32_C(   611862041), -INT32_C(   977650648),  INT32_C(  1052007373), -INT32_C(    26927961), -INT32_C(   234861269), -INT32_C(  1421140538), -INT32_C(  1706530008) },
      {  INT64_C(                  10),  INT64_C(                   9) },
      {  INT32_C(   362472448), -INT32_C(  1797994496),  INT32_C(   327690240), -INT32_C(  2042465280), -INT32_C(  1657682944), -INT32_C(  1606592512),  INT32_C(  1910971392),  INT32_C(  2083491840),
        -INT32_C(  2020489216), -INT32_C(   518495232), -INT32_C(   386883584), -INT32_C(   781241344), -INT32_C(  1804428288),  INT32_C(    20229120),  INT32_C(   746002432),  INT32_C(   564961280) } },
    { {  INT32_C(   407331821),  INT32_C(  1997162673), -INT32_C(  1927129499),  INT32_C(   271084481), -INT32_C(  2072418476),  INT32_C(   299566622), -INT32_C(   271386547), -INT32_C(  1220947766),
        -INT32_C(   439320524),  INT32_C(   761060040), -INT32_C(   239370448),  INT32_C(  1040376810),  INT32_C(  1757576010), -INT32_C(   814113150),  INT32_C(  1623149462),  INT32_C(  1343813660) },
      {  INT64_C(                  30),  INT64_C(                  10) },
      {  INT32_C(  1073741824),  INT32_C(  1073741824),  INT32_C(  1073741824),  INT32_C(  1073741824),  INT32_C(           0),            INT32_MIN,  INT32_C(  1073741824),            INT32_MIN,
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),            INT32_MIN,            INT32_MIN,            INT32_MIN,            INT32_MIN,  INT32_C(           0) } },
    { { -INT32_C(   457245854),  INT32_C(   817051801), -INT32_C(  1617923453), -INT32_C(   470833046),  INT32_C(  1394877584), -INT32_C(   968453450), -INT32_C(   926864708),  INT32_C(  1931475473),
        -INT32_C(  1890066955), -INT32_C(  1715533291), -INT32_C(   415740035), -INT32_C(  2000017160),  INT32_C(    47967820),  INT32_C(  1036591489),  INT32_C(  1560644172),  INT32_C(  2043683972) },
      {  INT64_C(                  21),  INT64_C(                  11) },
      {  INT32_C(  1816133632),  INT32_C(   320864256),  INT32_C(  1348468736),  INT32_C(   222298112), -INT32_C(  1845493760),  INT32_C(  1455423488),  INT32_C(  1468006400),  INT32_C(  1109393408),
        -INT32_C(  1096810496),  INT32_C(  1117782016), -INT32_C(   274726912),  INT32_C(   520093696), -INT32_C(   914358272),  INT32_C(   807403520),  INT32_C(  1233125376), -INT32_C(  1870659584) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sll_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sll_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sll_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int64_t b[2];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(    66481915),  INT32_C(   651953117),  INT32_C(   585778516),  INT32_C(   874194668), -INT32_C(   357929301),  INT32_C(   677909855), -INT32_C(  1595426413),  INT32_C(   575516444),
         INT32_C(   488527424), -INT32_C(  1388053927),  INT32_C(   735063359), -INT32_C(    10491053), -INT32_C(  1192687271), -INT32_C(  1293921762),  INT32_C(   659736331), -INT32_C(  1219911561) },
      UINT16_C(26614),
      {  INT32_C(   409030869), -INT32_C(   851009283),  INT32_C(   716741067), -INT32_C(  2110582376), -INT32_C(  1445702967),  INT32_C(  1450303236),  INT32_C(  1358358534),  INT32_C(  1958210975),
         INT32_C(   982325565), -INT32_C(  2063019079),  INT32_C(    95404140), -INT32_C(  1685593134),  INT32_C(   960846645),  INT32_C(  1083160122),  INT32_C(  1032881822), -INT32_C(  1330493325) },
      {  INT64_C(                  15),  INT64_C(                  12) },
      { -INT32_C(    66481915),  INT32_C(  1350467584),  INT32_C(  1290108928),  INT32_C(   874194668),  INT32_C(   694452224), -INT32_C(   276692992),  INT32_C(  1946353664), -INT32_C(   154173440),
        -INT32_C(  1935769600),  INT32_C(  1776058368), -INT32_C(   533331968), -INT32_C(    10491053), -INT32_C(  1192687271), -INT32_C(   618856448),  INT32_C(  1129250816), -INT32_C(  1219911561) } },
    { { -INT32_C(  1616878235), -INT32_C(  1176490213),  INT32_C(   687304885),  INT32_C(   433629624), -INT32_C(    97205272),  INT32_C(  1752748981),  INT32_C(  1424882722),  INT32_C(  1824198150),
        -INT32_C(  1895081101),  INT32_C(  1078520715),  INT32_C(   342441820), -INT32_C(   802274840), -INT32_C(  1161141756),  INT32_C(  1478640437),  INT32_C(  1168904255), -INT32_C(  1850644450) },
      UINT16_C(48579),
      {  INT32_C(  1772637728), -INT32_C(   123206513),  INT32_C(  1194955033),  INT32_C(   749289057),  INT32_C(   443539192), -INT32_C(   483742154), -INT32_C(  1521792781), -INT32_C(    94236966),
        -INT32_C(   329053603),  INT32_C(   686033935), -INT32_C(    26272355),  INT32_C(  1395267675),  INT32_C(   762223094),  INT32_C(   990943303), -INT32_C(  1142924319), -INT32_C(   961199511) },
      {  INT64_C(                  24),  INT64_C(                   0) },
      {  INT32_C(   536870912), -INT32_C(  1895825408),  INT32_C(   687304885),  INT32_C(   433629624), -INT32_C(    97205272),  INT32_C(  1752748981), -INT32_C(   218103808), -INT32_C(   637534208),
         INT32_C(  1560281088),  INT32_C(  1078520715), -INT32_C(  1660944384),  INT32_C(  1526726656), -INT32_C(   167772160),  INT32_C(  1191182336),  INT32_C(  1168904255),  INT32_C(  1761607680) } },
    { { -INT32_C(   869346940),  INT32_C(  1241988713),  INT32_C(    33941401),  INT32_C(  1976154921), -INT32_C(   103711788), -INT32_C(   960801774),  INT32_C(  1372945223), -INT32_C(   346933146),
         INT32_C(  1941405705),  INT32_C(  1472052926), -INT32_C(   832912475),  INT32_C(  1380131710), -INT32_C(  1337256802), -INT32_C(  1334442391), -INT32_C(   402568063), -INT32_C(  1898752892) },
      UINT16_C(35540),
      { -INT32_C(  1102540031),  INT32_C(  1149365738), -INT32_C(    26738757), -INT32_C(  1676474799),  INT32_C(   765623478),  INT32_C(   762913836), -INT32_C(   545129204), -INT32_C(  1939253621),
        -INT32_C(   750013975),  INT32_C(  1494797470),  INT32_C(   492273612),  INT32_C(  1018849925),  INT32_C(   308894950), -INT32_C(  1941904768),  INT32_C(  1802224095),  INT32_C(    16241687) },
      {  INT64_C(                  27),  INT64_C(                   4) },
      { -INT32_C(   869346940),  INT32_C(  1241988713), -INT32_C(   671088640),  INT32_C(  1976154921), -INT32_C(  1342177280), -INT32_C(   960801774),  INT32_C(  1610612736),  INT32_C(  1476395008),
         INT32_C(  1941405705), -INT32_C(   268435456), -INT32_C(   832912475),  INT32_C(   671088640), -INT32_C(  1337256802), -INT32_C(  1334442391), -INT32_C(   402568063), -INT32_C(  1207959552) } },
    { { -INT32_C(  1892182513),  INT32_C(  1461483384),  INT32_C(  1354925881), -INT32_C(   514737572),  INT32_C(   184886780),  INT32_C(  2095481105), -INT32_C(  1804738731), -INT32_C(  1598449007),
        -INT32_C(  1473187792), -INT32_C(  1593815960),  INT32_C(   804373203),  INT32_C(  2031174268),  INT32_C(  2021922407), -INT32_C(   302683241), -INT32_C(   612277686),  INT32_C(   763116285) },
      UINT16_C(44074),
      { -INT32_C(   688352554),  INT32_C(   630770483),  INT32_C(   208082427),  INT32_C(   304271246),  INT32_C(  1014872391),  INT32_C(   664782758),  INT32_C(   526490787), -INT32_C(  1614050103),
         INT32_C(  1383449374), -INT32_C(  1988686194), -INT32_C(  1315578333), -INT32_C(   171722835),  INT32_C(   389103985), -INT32_C(  1421881336),  INT32_C(   919249004),  INT32_C(  1272288556) },
      {  INT64_C(                  15),  INT64_C(                   5) },
      { -INT32_C(  1892182513),  INT32_C(  1704558592),  INT32_C(  1354925881),  INT32_C(  1741094912),  INT32_C(   184886780), -INT32_C(   472711168), -INT32_C(  1804738731), -INT32_C(  1598449007),
        -INT32_C(  1473187792), -INT32_C(  1593815960), -INT32_C(   284065792), -INT32_C(   606699520),  INT32_C(  2021922407), -INT32_C(   402391040), -INT32_C(   612277686), -INT32_C(   896139264) } },
    { {  INT32_C(   977716785),  INT32_C(    65373591),  INT32_C(  1379512357), -INT32_C(  1633874107), -INT32_C(  1283114406),  INT32_C(  1076884814), -INT32_C(  1176478469),  INT32_C(  2129098060),
        -INT32_C(   742904516), -INT32_C(   657023566), -INT32_C(  1825959859),  INT32_C(  2033305375),  INT32_C(  1328330241), -INT32_C(  1483777109),  INT32_C(    56651959),  INT32_C(   562120677) },
      UINT16_C(14709),
      { -INT32_C(   875091980),  INT32_C(   702227711), -INT32_C(   386860361), -INT32_C(  1616973453), -INT32_C(   788903360),  INT32_C(  1363194353),  INT32_C(   915940788),  INT32_C(   997133639),
         INT32_C(   872826421),  INT32_C(   576643435),  INT32_C(  1309363931), -INT32_C(  2131908288), -INT32_C(   464459789),  INT32_C(  1295356056), -INT32_C(    41693514),  INT32_C(   272167643) },
      {  INT64_C(                  31),  INT64_C(                  13) },
      {  INT32_C(           0),  INT32_C(    65373591),            INT32_MIN, -INT32_C(  1633874107),  INT32_C(           0),            INT32_MIN,  INT32_C(           0),  INT32_C(  2129098060),
                   INT32_MIN, -INT32_C(   657023566), -INT32_C(  1825959859),  INT32_C(           0),            INT32_MIN,  INT32_C(           0),  INT32_C(    56651959),  INT32_C(   562120677) } },
    { { -INT32_C(  1202519521),  INT32_C(  1225099411),  INT32_C(   843483222), -INT32_C(  1287487878), -INT32_C(   564688963),  INT32_C(   484056618),  INT32_C(  1783440623),  INT32_C(  2094661468),
        -INT32_C(  1791742974),  INT32_C(   199113140),  INT32_C(   993862849),  INT32_C(  1626308514),  INT32_C(   826164743),  INT32_C(  1414338660), -INT32_C(  1715561668), -INT32_C(  1676306534) },
      UINT16_C(18883),
      {  INT32_C(   260208689), -INT32_C(  1070382205),  INT32_C(  1832900222),  INT32_C(  1957971510), -INT32_C(   980674440),  INT32_C(   727763052),  INT32_C(  1992489825), -INT32_C(   910195049),
        -INT32_C(  2116533762),  INT32_C(    54594692),  INT32_C(   410026210),  INT32_C(  1066149063), -INT32_C(  1459349443),  INT32_C(  1121215968), -INT32_C(   138897568),  INT32_C(   465598493) },
      {  INT64_C(                  16),  INT64_C(                   2) },
      {  INT32_C(  2016477184),  INT32_C(  1132658688),  INT32_C(   843483222), -INT32_C(  1287487878), -INT32_C(   564688963),  INT32_C(   484056618), -INT32_C(    77529088), -INT32_C(  2036924416),
         INT32_C(  1107165184),  INT32_C(   199113140),  INT32_C(   993862849),  INT32_C(   617021440),  INT32_C(   826164743),  INT32_C(  1414338660), -INT32_C(  1755316224), -INT32_C(  1676306534) } },
    { {  INT32_C(   599550019),  INT32_C(   761631181),  INT32_C(  1159994920),  INT32_C(  1331750294),  INT32_C(   596507774),  INT32_C(   917163737),  INT32_C(  1448823168), -INT32_C(  1217806732),
        -INT32_C(   203807450), -INT32_C(   568311626), -INT32_C(   199015074), -INT32_C(  1471970518),  INT32_C(  1489752447),  INT32_C(   529495455),  INT32_C(   846588606), -INT32_C(   806756696) },
      UINT16_C(49924),
      { -INT32_C(   486294846), -INT32_C(  1138204263), -INT32_C(  1723837867), -INT32_C(   982859782), -INT32_C(  1489368808), -INT32_C(  1634600919),  INT32_C(   326973738), -INT32_C(   875134712),
        -INT32_C(   726738373), -INT32_C(  1852713413),  INT32_C(   573231400), -INT32_C(  1461218160), -INT32_C(   967892579),  INT32_C(  1130749977),  INT32_C(   576119322), -INT32_C(  1628623773) },
      {  INT64_C(                  19),  INT64_C(                   5) },
      {  INT32_C(   599550019),  INT32_C(   761631181), -INT32_C(  1834483712),  INT32_C(  1331750294),  INT32_C(   596507774),  INT32_C(   917163737),  INT32_C(  1448823168), -INT32_C(  1217806732),
        -INT32_C(   774373376), -INT32_C(  1311244288), -INT32_C(   199015074), -INT32_C(  1471970518),  INT32_C(  1489752447),  INT32_C(   529495455),  INT32_C(   282066944),  INT32_C(  1662517248) } },
    { { -INT32_C(   587899453),  INT32_C(  1478449726), -INT32_C(  1619364548), -INT32_C(  1472370526),  INT32_C(  1978314755), -INT32_C(  1995522636),  INT32_C(  1274006202),  INT32_C(   813366636),
        -INT32_C(  1744015526),  INT32_C(    99626185),  INT32_C(  1134848929), -INT32_C(   689118765),  INT32_C(  1179375250),  INT32_C(  1322277524), -INT32_C(  1868906716),  INT32_C(   381686972) },
      UINT16_C(52611),
      { -INT32_C(  1611051857), -INT32_C(   183854511), -INT32_C(   891888162), -INT32_C(     6264652),  INT32_C(  2136552623),  INT32_C(   490634627), -INT32_C(   835585522), -INT32_C(  1080314864),
         INT32_C(  1365218304),  INT32_C(   222718255), -INT32_C(    69788601),  INT32_C(   888829829),  INT32_C(   800281772), -INT32_C(   548605487), -INT32_C(    72450581),  INT32_C(   834357553) },
      {  INT64_C(                  25),  INT64_C(                   2) },
      {  INT32_C(  1577058304), -INT32_C(  1577058304), -INT32_C(  1619364548), -INT32_C(  1472370526),  INT32_C(  1978314755), -INT32_C(  1995522636),  INT32_C(  1274006202),  INT32_C(   536870912),
         INT32_C(           0),  INT32_C(    99626185), -INT32_C(  1912602624),  INT32_C(   167772160),  INT32_C(  1179375250),  INT32_C(  1322277524), -INT32_C(   704643072),  INT32_C(  1644167168) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sll_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sll_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sll_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int64_t b[2];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(19274),
      {  INT32_C(  1987879395), -INT32_C(   743556432),  INT32_C(   119987798),  INT32_C(  1067671101),  INT32_C(  1066290845),  INT32_C(  1149348051), -INT32_C(  1149162198), -INT32_C(  1845048402),
         INT32_C(   487097197),  INT32_C(   334542781), -INT32_C(   753265003),  INT32_C(   303218293), -INT32_C(   548298484),  INT32_C(  1948504905), -INT32_C(  1372609536), -INT32_C(   599771537) },
      {  INT64_C(                  18),  INT64_C(                  13) },
      {  INT32_C(           0), -INT32_C(   356515840),  INT32_C(           0),  INT32_C(  2029256704),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1264058368),  INT32_C(           0),
         INT32_C(   229900288), -INT32_C(   554434560),  INT32_C(           0), -INT32_C(   103546880),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1879048192),  INT32_C(           0) } },
    { UINT16_C(55836),
      {  INT32_C(  1873634636),  INT32_C(   152284633), -INT32_C(  1673559205),  INT32_C(  1508243551), -INT32_C(   146479762), -INT32_C(   476428934),  INT32_C(  1130298555), -INT32_C(  1239582103),
        -INT32_C(  1188705569), -INT32_C(   742246025),  INT32_C(   460259772),  INT32_C(  1735742713), -INT32_C(  1285637831), -INT32_C(  1181288194),  INT32_C(  1593636084),  INT32_C(  1309940334) },
      {  INT64_C(                   5),  INT64_C(                   5) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  2014287008),  INT32_C(  1019153376), -INT32_C(   392385088),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  2017930976),  INT32_C(           0), -INT32_C(   290808032),  INT32_C(  1809262368),  INT32_C(           0), -INT32_C(   543252864), -INT32_C(  1031582272) } },
    { UINT16_C(34936),
      {  INT32_C(  1971353310),  INT32_C(   728331567), -INT32_C(   414852909),  INT32_C(   757082662), -INT32_C(  1242131578),  INT32_C(   190038209), -INT32_C(  1379252861),  INT32_C(  1312113264),
        -INT32_C(   406604360), -INT32_C(    49074902),  INT32_C(   736385029),  INT32_C(   139986306),  INT32_C(  1505578648), -INT32_C(  1855647730), -INT32_C(   985780395),  INT32_C(   705983346) },
      {  INT64_C(                   1),  INT64_C(                  26) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1514165324),  INT32_C(  1810704140),  INT32_C(   380076418),  INT32_C(  1536461574),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   279972612),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1411966692) } },
    { UINT16_C(50906),
      { -INT32_C(  1864964053), -INT32_C(  1212142471),  INT32_C(    86651633),  INT32_C(  1859998556), -INT32_C(   158080602),  INT32_C(   522916331),  INT32_C(   430728465),  INT32_C(  1675593271),
        -INT32_C(   386681233), -INT32_C(   744442910),  INT32_C(  1121569509), -INT32_C(  1011829219), -INT32_C(  2101721961),  INT32_C(  1721951573), -INT32_C(  2105586101),  INT32_C(  1139105748) },
      {  INT64_C(                  14),  INT64_C(                  30) },
      {  INT32_C(           0),  INT32_C(   186531840),  INT32_C(           0),  INT32_C(  1423376384), -INT32_C(   127303680),  INT32_C(           0),  INT32_C(   423903232), -INT32_C(   510803968),
         INT32_C(           0),  INT32_C(   754483200),  INT32_C(  1924743168),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   745357312),  INT32_C(  1475674112) } },
    { UINT16_C(12712),
      {  INT32_C(   353893747), -INT32_C(   480026013), -INT32_C(   901629724), -INT32_C(  1482467461),  INT32_C(   410201934),  INT32_C(  1438386849),  INT32_C(   901387710),  INT32_C(  2053595654),
        -INT32_C(  1014006176), -INT32_C(   995691552), -INT32_C(  1500583893), -INT32_C(  1924255425),  INT32_C(    44482913), -INT32_C(  1990696245),  INT32_C(  2126451319), -INT32_C(   721934732) },
      {  INT64_C(                  20),  INT64_C(                  13) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2008023040),  INT32_C(           0), -INT32_C(  1441792000),  INT32_C(           0),  INT32_C(   543162368),
        -INT32_C(   436207616),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   370147328),  INT32_C(   749731840),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(39515),
      { -INT32_C(   990107796),  INT32_C(  1876325296), -INT32_C(   376091919), -INT32_C(  1217316577),  INT32_C(   116779965), -INT32_C(   572711791),  INT32_C(  2004346243), -INT32_C(  1156459953),
        -INT32_C(  1434513927),  INT32_C(  1914262912), -INT32_C(  1101287521), -INT32_C(  1502229272),  INT32_C(  1236036536),  INT32_C(   170297735), -INT32_C(  1115579026),  INT32_C(  1769509487) },
      {  INT64_C(                  29),  INT64_C(                  14) },
      {            INT32_MIN,  INT32_C(           0),  INT32_C(           0), -INT32_C(   536870912), -INT32_C(  1610612736),  INT32_C(           0),  INT32_C(  1610612736),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   536870912) } },
    { UINT16_C( 5771),
      { -INT32_C(   509668677), -INT32_C(  1635906275),  INT32_C(  1127345611),  INT32_C(  1765527638), -INT32_C(  2104064016),  INT32_C(   510685555), -INT32_C(  1623315915),  INT32_C(  1471531420),
        -INT32_C(   130525989),  INT32_C(   764917346), -INT32_C(    93271901), -INT32_C(  1989956712), -INT32_C(  1509164749), -INT32_C(  1597736085),  INT32_C(  1094714021), -INT32_C(  1483147829) },
      {  INT64_C(                   5),  INT64_C(                   5) },
      {  INT32_C(   870471520), -INT32_C(   809393248),  INT32_C(           0),  INT32_C(   662309568),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   155634816),
         INT32_C(           0), -INT32_C(  1292448704),  INT32_C(  1310266464),  INT32_C(           0), -INT32_C(  1048631712),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(11137),
      { -INT32_C(  1817711410), -INT32_C(   862630772),  INT32_C(   650273166), -INT32_C(  1476981752), -INT32_C(  1847689800), -INT32_C(   874849113), -INT32_C(  1823223949), -INT32_C(   960506633),
         INT32_C(  1280927424),  INT32_C(  1075441330),  INT32_C(  1466424143), -INT32_C(  1610653977), -INT32_C(  2093949477),  INT32_C(   760089273), -INT32_C(   775904806),  INT32_C(   899121013) },
      {  INT64_C(                   5),  INT64_C(                  15) },
      {  INT32_C(  1962777024),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   671441184),
        -INT32_C(  1959995392),  INT32_C(    54384192),  INT32_C(           0), -INT32_C(     1319712),  INT32_C(           0), -INT32_C(  1446947040),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_sll_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_sll_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_sll_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[2];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 3043210905362980970), -INT64_C( 8566001345715781385),  INT64_C( 8038474297071378046), -INT64_C( 4267062589809021897),
         INT64_C( 6240767933180872696), -INT64_C( 6549473941646625943), -INT64_C( 4256242992551417930),  INT64_C( 5973814186616553973) },
      {  INT64_C(                  55),  INT64_C(                  48) },
      { -INT64_C( 3819052484010180608),  INT64_C( 8899112863684100096),  INT64_C( 4539628424389459968),  INT64_C( 1981583836043018240),
        -INT64_C(  288230376151711744), -INT64_C( 5440348349863559168), -INT64_C( 2666130979403333632), -INT64_C(  396316767208603648) } },
    { {  INT64_C( 3099936928095694261),  INT64_C(  422834507516640561),  INT64_C( 1393766964600239874),  INT64_C(  931634168761272604),
        -INT64_C( 7952420843205873855),  INT64_C( 2996295799414846160),  INT64_C(  265987151192211442),  INT64_C( 7432375506683384258) },
      {  INT64_C(                  37),  INT64_C(                  17) },
      {  INT64_C( 2904178133034860544),  INT64_C( 7624495350530703360),  INT64_C( 1548921887344558080),  INT64_C( 3763077996307546112),
         INT64_C(  695216941635207168), -INT64_C( 4341934034692079616), -INT64_C( 5465120071959445504),  INT64_C( 2714043572973207552) } },
    { {  INT64_C( 2514015061285169144),  INT64_C( 8318536819918246116),  INT64_C( 5129979112103694548),  INT64_C(  524854139031869104),
        -INT64_C( 6732095577990419953),  INT64_C(  321757986159382234), -INT64_C( 7167813143326976915),  INT64_C( 2395043167232551205) },
      {  INT64_C(                  49),  INT64_C(                  41) },
      { -INT64_C( 3463268113447911424),  INT64_C( 2434195598593753088), -INT64_C( 6798183637515763712), -INT64_C( 5953758707383795712),
         INT64_C( 6061282148487266304), -INT64_C( 7083036313946947584),  INT64_C( 2943665306440040448), -INT64_C( 3293820177468096512) } },
    { { -INT64_C( 5161004695847189545), -INT64_C(  501543431971209257), -INT64_C( 5764824409340077237), -INT64_C( 2001300220254565801),
         INT64_C( 7278388255503360183),  INT64_C( 7126761795142741511),  INT64_C( 4711498488697565172), -INT64_C( 2425737990017699227) },
      {  INT64_C(                   3),  INT64_C(                  46) },
      { -INT64_C( 4394549419358413128), -INT64_C( 4012347455769674056),  INT64_C( 9221636946408036952),  INT64_C( 2436342311673025208),
         INT64_C( 2886873822898226616),  INT64_C( 1673862140013277240),  INT64_C(  798499762161418144), -INT64_C(  959159846432042200) } },
    { {  INT64_C( 3223398413973138832), -INT64_C( 4105606542351495679),  INT64_C( 8071473455876419058),  INT64_C( 7967204048965205264),
        -INT64_C( 3579618652289696162), -INT64_C( 2912707569573760719), -INT64_C( 3895383087310301655),  INT64_C( 2313497541479534798) },
      {  INT64_C(                  54),  INT64_C(                  57) },
      {  INT64_C( 7205759403792793600), -INT64_C( 9205357638345293824),  INT64_C( 8971170457722028032),  INT64_C( 4899916394579099648),
        -INT64_C( 7530018576963469312),  INT64_C( 5494391545392005120), -INT64_C( 8484781697966014464),  INT64_C( 3710966092953288704) } },
    { { -INT64_C( 5345957587203975858), -INT64_C( 3329993415690457113),  INT64_C( 3602768269637717888),  INT64_C( 3026672782606902364),
        -INT64_C( 6442679917796628485), -INT64_C(  729274773593393041), -INT64_C( 6413095861259292633), -INT64_C( 9023593494483984193) },
      {  INT64_C(                  25),  INT64_C(                   7) },
      { -INT64_C( 3743390507850530816),  INT64_C( 4851154015121047552),  INT64_C( 5642900515076440064), -INT64_C( 5173156816384688128),
         INT64_C( 9206589984553828352), -INT64_C( 1576084012714164224), -INT64_C( 7665626429949739008),  INT64_C(  952374163522191360) } },
    { {  INT64_C( 6123186393984104711), -INT64_C( 6587159339739334003), -INT64_C( 1727381194576954965), -INT64_C( 2590221981096837639),
         INT64_C( 8363509859251845671),  INT64_C( 7061958359767319009), -INT64_C( 6777061297406898921),  INT64_C( 1817377213416293511) },
      {  INT64_C(                  27),  INT64_C(                  44) },
      { -INT64_C( 6154331858101338112), -INT64_C( 2837402092895731712), -INT64_C( 6553749226088562688), -INT64_C( 6378640856341020672),
         INT64_C( 4458976846544896000), -INT64_C( 5556245606965444608),  INT64_C( 4636487052467109888), -INT64_C( 7862007161319063552) } },
    { { -INT64_C( 7592030972611328935),  INT64_C( 7710209647364420852), -INT64_C( 6517867833321531760),  INT64_C( 8838952131357791582),
         INT64_C(  931152308542859080), -INT64_C( 7453191266487837102), -INT64_C( 4960081269175780152), -INT64_C(  418449168689500522) },
      {  INT64_C(                   7),  INT64_C(                  37) },
      {  INT64_C( 5897471412356131968), -INT64_C( 9217345117669918208), -INT64_C( 4183599348226242560),  INT64_C( 6134484317514673920),
         INT64_C( 8507031051228652544),  INT64_C( 5222209722453534976), -INT64_C( 7701103948375104512),  INT64_C( 1778738628872588032) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sll_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sll_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sll_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[2];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 8844407328334556588), -INT64_C( 6468868398122554644),  INT64_C( 7919920011796737235), -INT64_C(  110988753495623940),
         INT64_C( 4968237128581166045), -INT64_C( 2053179562796574108),  INT64_C( 6433626666797115536), -INT64_C( 6081296775066477237) },
      UINT8_C(151),
      {  INT64_C( 3437429681747403670),  INT64_C( 4095217988934799264),  INT64_C( 1526702650967896079), -INT64_C(  226712261368755362),
        -INT64_C( 6992400277197501775),  INT64_C( 7482933471671956879), -INT64_C( 4866567801566171544), -INT64_C( 7446462394794026682) },
      {  INT64_C(                  33),  INT64_C(                  16) },
      { -INT64_C( 6782342784515833856), -INT64_C( 8611129602770731008), -INT64_C( 3950395616168771584), -INT64_C(  110988753495623940),
        -INT64_C( 8315450988823576576), -INT64_C( 2053179562796574108),  INT64_C( 6433626666797115536),  INT64_C( 6484816827825258496) } },
    { {  INT64_C( 6234698294050919788),  INT64_C(  140080385622181234),  INT64_C( 3322217971163811999), -INT64_C( 6330834859143905530),
         INT64_C( 6005261897915615216), -INT64_C( 6710752655294244738),  INT64_C( 7692519130933985774),  INT64_C( 2194782635921494672) },
      UINT8_C( 55),
      { -INT64_C( 3648079019115012376), -INT64_C( 7159189469750346728), -INT64_C( 3707469667518826326), -INT64_C( 5233999913943410249),
        -INT64_C(  470072137464908490),  INT64_C( 7488174172224375092),  INT64_C( 1105052447113735310), -INT64_C( 7075274418578698423) },
      {  INT64_C(                  36),  INT64_C(                  30) },
      {  INT64_C( 1261551604163674112),  INT64_C( 9008256435171098624), -INT64_C( 4606185574070484992), -INT64_C( 6330834859143905530),
        -INT64_C( 5077998383054979072), -INT64_C(  359000167195607040),  INT64_C( 7692519130933985774),  INT64_C( 2194782635921494672) } },
    { { -INT64_C( 3882008865968975175), -INT64_C( 1033099025019939164), -INT64_C( 4681000655360626152),  INT64_C( 8193093049506065233),
        -INT64_C( 5938942746147179704),  INT64_C( 4743524235269994489), -INT64_C( 4699575012095905964),  INT64_C( 5680917119143333804) },
      UINT8_C( 96),
      {  INT64_C( 2591283567823231065),  INT64_C( 6594833000054970575), -INT64_C( 6578765831402386107),  INT64_C( 5083992152416524160),
        -INT64_C( 2531241738697968113), -INT64_C(  208973486024217839), -INT64_C( 1650086124891872736), -INT64_C( 3890367105679834162) },
      {  INT64_C(                  36),  INT64_C(                  16) },
      { -INT64_C( 3882008865968975175), -INT64_C( 1033099025019939164), -INT64_C( 4681000655360626152),  INT64_C( 8193093049506065233),
        -INT64_C( 5938942746147179704), -INT64_C( 2246538581826863104),  INT64_C( 5786599954613010432),  INT64_C( 5680917119143333804) } },
    { { -INT64_C( 3414623508423488695), -INT64_C( 1217784623362428267), -INT64_C( 7574184836662452268), -INT64_C( 2158307683753578073),
         INT64_C( 3189727863122478449),  INT64_C( 3758418125259526371),  INT64_C( 8993524444907945524),  INT64_C( 5122091226845589403) },
      UINT8_C( 23),
      { -INT64_C( 3477382549134722390), -INT64_C( 2513182177820215722), -INT64_C( 2887318460848514583), -INT64_C( 2170747974958898571),
        -INT64_C( 4720764818892747346), -INT64_C(  689956682324137360), -INT64_C( 4458220251510071775),  INT64_C( 8174994033010724442) },
      {  INT64_C(                   6),  INT64_C(                  43) },
      { -INT64_C( 1191554260107613568),  INT64_C( 5177037282892158336), -INT64_C(  320940757209417152), -INT64_C( 2158307683753578073),
        -INT64_C( 6981043229783004288),  INT64_C( 3758418125259526371),  INT64_C( 8993524444907945524),  INT64_C( 5122091226845589403) } },
    { {  INT64_C( 2684738884427420210),  INT64_C( 3178743322012798185),  INT64_C( 1809505683625360218),  INT64_C(  768624430915765356),
        -INT64_C( 6326205360360479931),  INT64_C( 8141225094183971737), -INT64_C( 7378427618179971668), -INT64_C(  201186747133786797) },
      UINT8_C(225),
      {  INT64_C( 3253366127696219584),  INT64_C( 8554785335769560475), -INT64_C( 2807928582899091162),  INT64_C( 2949171008047821775),
         INT64_C( 8073789563819363740),  INT64_C( 2826067047849152209), -INT64_C( 1937422955413972370), -INT64_C( 2690655322121327081) },
      {  INT64_C(                   2),  INT64_C(                  52) },
      { -INT64_C( 5433279562924673280),  INT64_C( 3178743322012798185),  INT64_C( 1809505683625360218),  INT64_C(  768624430915765356),
        -INT64_C( 6326205360360479931), -INT64_C( 7142475882312942780), -INT64_C( 7749691821655889480),  INT64_C( 7684122785224243292) } },
    { {  INT64_C( 1214335800134800863),  INT64_C( 1726065765214309597), -INT64_C( 9162282690612319083), -INT64_C( 8401522641993554379),
        -INT64_C( 8725342017996948768),  INT64_C( 5037722244871190823),  INT64_C( 9021533900040847426),  INT64_C( 8127445436518941844) },
      UINT8_C(246),
      { -INT64_C( 5995093525679551925), -INT64_C( 3990292038261301305),  INT64_C( 5132008467138171062), -INT64_C( 8280633990523160669),
        -INT64_C( 6669548624957688431),  INT64_C( 2877693995247521439),  INT64_C( 3654956487228981488),  INT64_C( 7586989603334230139) },
      {  INT64_C(                   7),  INT64_C(                  10) },
      {  INT64_C( 1214335800134800863),  INT64_C( 5751453166420878208), -INT64_C( 7185702859857962240), -INT64_C( 8401522641993554379),
        -INT64_C( 5151996603944744832), -INT64_C(  590050082508288128),  INT64_C( 6665828522570840064), -INT64_C( 6542766679824777856) } },
    { {  INT64_C( 5905441591289369139), -INT64_C( 4095644891431942680),  INT64_C( 7380937961405110485),  INT64_C( 1611811354841625079),
         INT64_C( 4271325957586484255), -INT64_C( 9114118322039073775), -INT64_C( 7576389794988342147), -INT64_C( 8148095368328371726) },
      UINT8_C( 34),
      {  INT64_C( 2207803725923648453),  INT64_C( 2904624830350057645), -INT64_C( 4515881587224099808), -INT64_C( 2634679353523361242),
         INT64_C( 4839417511209198913),  INT64_C( 7024881225514218608), -INT64_C( 3996638305905916888), -INT64_C( 5245986071642851264) },
      {  INT64_C(                   7),  INT64_C(                  57) },
      {  INT64_C( 5905441591289369139),  INT64_C( 2857096810616346240),  INT64_C( 7380937961405110485),  INT64_C( 1611811354841625079),
         INT64_C( 4271325957586484255), -INT64_C( 4705662745948047360), -INT64_C( 7576389794988342147), -INT64_C( 8148095368328371726) } },
    { { -INT64_C( 7233002708445346763), -INT64_C( 6909888283422837522), -INT64_C( 6227385604361953852),  INT64_C( 3376376977820354361),
        -INT64_C( 6014656537924630384),  INT64_C( 7147261668170496345), -INT64_C( 1550246859726401782),  INT64_C( 2492948631282271858) },
      UINT8_C( 67),
      {  INT64_C( 1052908368995356318),  INT64_C( 3239513502534937791), -INT64_C( 8447688280260097817),  INT64_C( 7252835846219116200),
         INT64_C( 1652895107988826445), -INT64_C( 7711860561054425744),  INT64_C( 5156245123832928191),  INT64_C( 3963822889188961239) },
      {  INT64_C(                  23),  INT64_C(                   8) },
      {  INT64_C( 5377720748691423232),  INT64_C( 3383846636374458368), -INT64_C( 6227385604361953852),  INT64_C( 3376376977820354361),
        -INT64_C( 6014656537924630384),  INT64_C( 7147261668170496345), -INT64_C( 3494103453737680896),  INT64_C( 2492948631282271858) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sll_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sll_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sll_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[2];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(249),
      { -INT64_C( 2427428339764267774), -INT64_C( 4325021648514947492), -INT64_C( 6293168853304433046),  INT64_C( 7598455919355618041),
         INT64_C( 8881084306305521048),  INT64_C( 9139553048861713498),  INT64_C( 4092764080299905758),  INT64_C( 3322853429276209997) },
      {  INT64_C(                   3),  INT64_C(                   9) },
      { -INT64_C(  972682644404590576),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 5447415133716289480),
        -INT64_C( 2738301844394038080), -INT64_C(  670551903944498480), -INT64_C( 4151375505019857168),  INT64_C( 8136083360500128360) } },
    { UINT8_C( 10),
      { -INT64_C( 8466906837822125114),  INT64_C( 1194104057701151539), -INT64_C( 3376906149356639265),  INT64_C( 8427152646742977010),
        -INT64_C( 1139650684260041587),  INT64_C( 9203346217789575256),  INT64_C( 4491865967930801093),  INT64_C( 4691936849519211002) },
      {  INT64_C(                  16),  INT64_C(                  59) },
      {  INT64_C(                   0),  INT64_C( 5715164826749304832),  INT64_C(                   0),  INT64_C( 4805034157475495936),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 27),
      {  INT64_C( 4322513128922837937), -INT64_C( 8916509603300137778), -INT64_C( 6011834511385667817), -INT64_C( 3868377566883244652),
         INT64_C( 3259700536392071223),  INT64_C( 8409163185970949229),  INT64_C( 6969938125797728176), -INT64_C( 8526750800371413050) },
      {  INT64_C(                  43),  INT64_C(                  18) },
      {  INT64_C( 4872199905466122240),  INT64_C( 9018018449158307840),  INT64_C(                   0), -INT64_C( 6004248281331269632),
         INT64_C( 7332343978475388928),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(173),
      { -INT64_C( 6891676135214983324),  INT64_C( 6624932376259738478),  INT64_C( 6106958149877165062),  INT64_C( 4060898676588679201),
         INT64_C( 2034828436589892748), -INT64_C( 5574224145695100828), -INT64_C( 6434165551018262708),  INT64_C( 7160354895656493393) },
      {  INT64_C(                  46),  INT64_C(                  27) },
      {  INT64_C( 6185975563170086912),  INT64_C(                   0), -INT64_C( 4971551776151961600),  INT64_C(  290552544709574656),
         INT64_C(                   0),  INT64_C(  871728002872901632),  INT64_C(                   0),  INT64_C( 7517704046732378112) } },
    { UINT8_C(229),
      {  INT64_C( 2656580082611178829), -INT64_C( 8242890917586205594), -INT64_C( 2518468635301589409),  INT64_C( 4694838733885482908),
         INT64_C(  639139111563596087), -INT64_C( 7520806632001906255), -INT64_C( 5078862884151917943),  INT64_C( 2566245761238663784) },
      {  INT64_C(                  50),  INT64_C(                  48) },
      {  INT64_C( 3833689182799134720),  INT64_C(                   0),  INT64_C( 4142185757274013696),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C( 7405043687303938048),  INT64_C(  730709039540862976),  INT64_C(  693554342615056384) } },
    { UINT8_C(  0),
      {  INT64_C( 7945149005378185368), -INT64_C( 6294204628957749551), -INT64_C( 2834018317142721445),  INT64_C( 8279254228066662248),
        -INT64_C( 6227195325739957703), -INT64_C( 4030647529310982356), -INT64_C( 4290932790102339265), -INT64_C( 8192171598209711358) },
      {  INT64_C(                  21),  INT64_C(                  12) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(160),
      { -INT64_C( 4261881613090173619), -INT64_C( 6733542215129647253), -INT64_C( 4837672991304386724),  INT64_C( 4346223159485104864),
         INT64_C( 8841954815871569328), -INT64_C( 3364186217201152602),  INT64_C( 2216144017487699193),  INT64_C(  216487602223476949) },
      {  INT64_C(                  42),  INT64_C(                  25) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 2938994581045248000),  INT64_C(                   0),  INT64_C( 4162262839597203456) } },
    { UINT8_C(196),
      { -INT64_C( 7324786852454307535),  INT64_C( 3781035903345827567),  INT64_C( 5548967890448413156), -INT64_C( 2762645388975870916),
        -INT64_C( 4879166627240891259),  INT64_C(  803964173351991253), -INT64_C( 9050659354103553742), -INT64_C( 4754375993875554939) },
      {  INT64_C(                  59),  INT64_C(                  32) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 2305843009213693952),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C( 8070450532247928832),  INT64_C( 2882303761517117440) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_sll_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_sll_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_bslli_epi128 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  static const struct {
    const int8_t a[64];
    const int8_t r[64];
  } test_vec[] = {
    { { -INT8_C( 117),  INT8_C(  80), -INT8_C(  42), -INT8_C(  41), -INT8_C( 118), -INT8_C( 121),  INT8_C( 125),  INT8_C(   1),
        -INT8_C( 106), -INT8_C(  68),  INT8_C(   0),  INT8_C( 102), -INT8_C(  98),  INT8_C(   6),  INT8_C(  36),  INT8_C(  40),
         INT8_C(  53),  INT8_C( 109),  INT8_C(  23),  INT8_C(  34),  INT8_C(  28), -INT8_C(   3),  INT8_C(   1), -INT8_C(  70),
        -INT8_C(  80),  INT8_C(  67), -INT8_C(  77),  INT8_C(  41), -INT8_C( 124), -INT8_C( 126),  INT8_C(   7),  INT8_C(  16),
        -INT8_C(  45), -INT8_C(  35), -INT8_C(  25),  INT8_C(  93),  INT8_C( 101),  INT8_C( 100),  INT8_C(  94), -INT8_C(   5),
         INT8_C(  32),  INT8_C(  94),  INT8_C(  98), -INT8_C(  66),  INT8_C( 101), -INT8_C( 122), -INT8_C(  26), -INT8_C( 102),
        -INT8_C(  12), -INT8_C(   3), -INT8_C(  68),  INT8_C(  16), -INT8_C(   6), -INT8_C(  67), -INT8_C(  54), -INT8_C(  85),
         INT8_C(   1),  INT8_C( 125), -INT8_C(  44), -INT8_C( 123),  INT8_C(   0), -INT8_C(  37), -INT8_C( 107), -INT8_C(  45) },
      {  INT8_C(   0), -INT8_C( 117),  INT8_C(  80), -INT8_C(  42), -INT8_C(  41), -INT8_C( 118), -INT8_C( 121),  INT8_C( 125),
         INT8_C(   1), -INT8_C( 106), -INT8_C(  68),  INT8_C(   0),  INT8_C( 102), -INT8_C(  98),  INT8_C(   6),  INT8_C(  36),
         INT8_C(   0),  INT8_C(  53),  INT8_C( 109),  INT8_C(  23),  INT8_C(  34),  INT8_C(  28), -INT8_C(   3),  INT8_C(   1),
        -INT8_C(  70), -INT8_C(  80),  INT8_C(  67), -INT8_C(  77),  INT8_C(  41), -INT8_C( 124), -INT8_C( 126),  INT8_C(   7),
         INT8_C(   0), -INT8_C(  45), -INT8_C(  35), -INT8_C(  25),  INT8_C(  93),  INT8_C( 101),  INT8_C( 100),  INT8_C(  94),
        -INT8_C(   5),  INT8_C(  32),  INT8_C(  94),  INT8_C(  98), -INT8_C(  66),  INT8_C( 101), -INT8_C( 122), -INT8_C(  26),
         INT8_C(   0), -INT8_C(  12), -INT8_C(   3), -INT8_C(  68),  INT8_C(  16), -INT8_C(   6), -INT8_C(  67), -INT8_C(  54),
        -INT8_C(  85),  INT8_C(   1),  INT8_C( 125), -INT8_C(  44), -INT8_C( 123),  INT8_C(   0), -INT8_C(  37), -INT8_C( 107) } },
    { { -INT8_C(  72),  INT8_C( 124),  INT8_C(  48),  INT8_C(  29), -INT8_C(  32), -INT8_C( 114),  INT8_C(  25),  INT8_C(   1),
        -INT8_C(  20),  INT8_C( 123), -INT8_C(  65),  INT8_C(  81),  INT8_C(   1), -INT8_C(  90), -INT8_C(  20), -INT8_C(  11),
        -INT8_C(  93), -INT8_C(  88),  INT8_C(   6), -INT8_C(  98),  INT8_C( 102), -INT8_C(  48),  INT8_C(  73),  INT8_C( 103),
         INT8_C(  78),  INT8_C(  29), -INT8_C(  20),  INT8_C(  78), -INT8_C(   8), -INT8_C( 126),  INT8_C(  33), -INT8_C(  80),
        -INT8_C(   2),  INT8_C(  81), -INT8_C(  50), -INT8_C(  33), -INT8_C(  33), -INT8_C(  25), -INT8_C(  32), -INT8_C(  53),
         INT8_C(  98), -INT8_C(  97),  INT8_C(  29),  INT8_C(  99),  INT8_C(  69),  INT8_C(   9),  INT8_C(  89), -INT8_C(  23),
        -INT8_C(  79),  INT8_C(  95), -INT8_C( 121),  INT8_C(  23),  INT8_C(  47), -INT8_C(  48),  INT8_C( 126),  INT8_C( 125),
        -INT8_C(  19),  INT8_C( 107), -INT8_C(  53), -INT8_C(  27), -INT8_C(  19), -INT8_C(  20), -INT8_C( 107), -INT8_C(  21) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C(  72),  INT8_C( 124),  INT8_C(  48),  INT8_C(  29), -INT8_C(  32), -INT8_C( 114),
         INT8_C(  25),  INT8_C(   1), -INT8_C(  20),  INT8_C( 123), -INT8_C(  65),  INT8_C(  81),  INT8_C(   1), -INT8_C(  90),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  93), -INT8_C(  88),  INT8_C(   6), -INT8_C(  98),  INT8_C( 102), -INT8_C(  48),
         INT8_C(  73),  INT8_C( 103),  INT8_C(  78),  INT8_C(  29), -INT8_C(  20),  INT8_C(  78), -INT8_C(   8), -INT8_C( 126),
         INT8_C(   0),  INT8_C(   0), -INT8_C(   2),  INT8_C(  81), -INT8_C(  50), -INT8_C(  33), -INT8_C(  33), -INT8_C(  25),
        -INT8_C(  32), -INT8_C(  53),  INT8_C(  98), -INT8_C(  97),  INT8_C(  29),  INT8_C(  99),  INT8_C(  69),  INT8_C(   9),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  79),  INT8_C(  95), -INT8_C( 121),  INT8_C(  23),  INT8_C(  47), -INT8_C(  48),
         INT8_C( 126),  INT8_C( 125), -INT8_C(  19),  INT8_C( 107), -INT8_C(  53), -INT8_C(  27), -INT8_C(  19), -INT8_C(  20) } },
    { {  INT8_C(  61),  INT8_C(  99), -INT8_C(  54),  INT8_C(  28),  INT8_C(  74), -INT8_C(  86), -INT8_C(  24), -INT8_C(  84),
         INT8_C(  74),  INT8_C(   5),  INT8_C(  16), -INT8_C( 113),  INT8_C(  14),  INT8_C( 105),  INT8_C( 120), -INT8_C(  65),
        -INT8_C(  56), -INT8_C(   1), -INT8_C(  41), -INT8_C(   9), -INT8_C(  49),  INT8_C(  85),  INT8_C( 117), -INT8_C(  68),
        -INT8_C(  64),  INT8_C(  64), -INT8_C(  95), -INT8_C(  83),  INT8_C(  45),  INT8_C(  55), -INT8_C( 103),  INT8_C( 106),
        -INT8_C( 102),  INT8_C(  99), -INT8_C( 121), -INT8_C(  27),  INT8_C(  14),  INT8_C( 111), -INT8_C( 111),  INT8_C(  88),
         INT8_C( 116), -INT8_C(  95), -INT8_C(  25), -INT8_C( 126),  INT8_C(  10),  INT8_C(  96),  INT8_C(  65), -INT8_C(  46),
         INT8_C(  95),  INT8_C(  24), -INT8_C(  54),  INT8_C(  47),  INT8_C( 110),  INT8_C(  63), -INT8_C(  21),  INT8_C(  46),
             INT8_MAX, -INT8_C( 115), -INT8_C(  36), -INT8_C(  84), -INT8_C(  60),  INT8_C( 117),  INT8_C(  23),  INT8_C(  94) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  61),  INT8_C(  99), -INT8_C(  54),  INT8_C(  28),  INT8_C(  74),
        -INT8_C(  86), -INT8_C(  24), -INT8_C(  84),  INT8_C(  74),  INT8_C(   5),  INT8_C(  16), -INT8_C( 113),  INT8_C(  14),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  56), -INT8_C(   1), -INT8_C(  41), -INT8_C(   9), -INT8_C(  49),
         INT8_C(  85),  INT8_C( 117), -INT8_C(  68), -INT8_C(  64),  INT8_C(  64), -INT8_C(  95), -INT8_C(  83),  INT8_C(  45),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 102),  INT8_C(  99), -INT8_C( 121), -INT8_C(  27),  INT8_C(  14),
         INT8_C( 111), -INT8_C( 111),  INT8_C(  88),  INT8_C( 116), -INT8_C(  95), -INT8_C(  25), -INT8_C( 126),  INT8_C(  10),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  95),  INT8_C(  24), -INT8_C(  54),  INT8_C(  47),  INT8_C( 110),
         INT8_C(  63), -INT8_C(  21),  INT8_C(  46),      INT8_MAX, -INT8_C( 115), -INT8_C(  36), -INT8_C(  84), -INT8_C(  60) } },
    { { -INT8_C(  40), -INT8_C(  98),  INT8_C(  67), -INT8_C(  26),  INT8_C(  13), -INT8_C(  43),  INT8_C(  62), -INT8_C( 127),
         INT8_C( 118),  INT8_C(  38),  INT8_C(   3), -INT8_C( 127), -INT8_C( 122),  INT8_C(  68),  INT8_C(  83), -INT8_C(  27),
         INT8_C(  93),  INT8_C(  29),  INT8_C(  20), -INT8_C(  53),  INT8_C(  92),  INT8_C(   0), -INT8_C(   7), -INT8_C(  36),
        -INT8_C( 115), -INT8_C(  43), -INT8_C( 120),  INT8_C(  81),  INT8_C(  74), -INT8_C(  97), -INT8_C(  81),  INT8_C(  35),
         INT8_C(  61), -INT8_C(  13),  INT8_C(   9),  INT8_C(  74), -INT8_C(  56),  INT8_C(  72), -INT8_C(  53),  INT8_C(  62),
         INT8_C( 110), -INT8_C(  50), -INT8_C(  65), -INT8_C(  12),  INT8_C(  19),  INT8_C(  19), -INT8_C(  39),  INT8_C( 112),
         INT8_C(  48), -INT8_C(  18),  INT8_C(  59), -INT8_C( 115), -INT8_C(  18),  INT8_C(  52),  INT8_C( 105),  INT8_C( 123),
         INT8_C(  10), -INT8_C(  15), -INT8_C(  52),  INT8_C(  84), -INT8_C( 111),  INT8_C( 123),  INT8_C( 119), -INT8_C(  50) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  40), -INT8_C(  98),  INT8_C(  67), -INT8_C(  26),
         INT8_C(  13), -INT8_C(  43),  INT8_C(  62), -INT8_C( 127),  INT8_C( 118),  INT8_C(  38),  INT8_C(   3), -INT8_C( 127),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  93),  INT8_C(  29),  INT8_C(  20), -INT8_C(  53),
         INT8_C(  92),  INT8_C(   0), -INT8_C(   7), -INT8_C(  36), -INT8_C( 115), -INT8_C(  43), -INT8_C( 120),  INT8_C(  81),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  61), -INT8_C(  13),  INT8_C(   9),  INT8_C(  74),
        -INT8_C(  56),  INT8_C(  72), -INT8_C(  53),  INT8_C(  62),  INT8_C( 110), -INT8_C(  50), -INT8_C(  65), -INT8_C(  12),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  48), -INT8_C(  18),  INT8_C(  59), -INT8_C( 115),
        -INT8_C(  18),  INT8_C(  52),  INT8_C( 105),  INT8_C( 123),  INT8_C(  10), -INT8_C(  15), -INT8_C(  52),  INT8_C(  84) } },
    { {  INT8_C( 110), -INT8_C( 127),  INT8_C(  25),  INT8_C(  54), -INT8_C(  55), -INT8_C(  28),  INT8_C( 117),  INT8_C(  55),
        -INT8_C(  77),  INT8_C(  52),  INT8_C(  43), -INT8_C(  58),  INT8_C(  71),  INT8_C(   4),  INT8_C(  54),  INT8_C( 120),
        -INT8_C(  14),  INT8_C( 113),  INT8_C(   5), -INT8_C(  32), -INT8_C(  91),  INT8_C( 110),  INT8_C(  91), -INT8_C(  81),
         INT8_C(  95),  INT8_C(  39),  INT8_C(   4), -INT8_C(  16), -INT8_C(  93),  INT8_C( 123), -INT8_C(  65),  INT8_C(  17),
        -INT8_C(   4), -INT8_C(  40),  INT8_C(  72), -INT8_C(  59), -INT8_C(  68), -INT8_C(  67), -INT8_C(   4),  INT8_C( 111),
        -INT8_C(  15),  INT8_C(  39),  INT8_C(  53),  INT8_C(  57),  INT8_C(  44),  INT8_C( 107), -INT8_C(  79),  INT8_C(  30),
        -INT8_C(  36), -INT8_C(  74), -INT8_C(   1), -INT8_C( 126),  INT8_C(  36),  INT8_C(  90),  INT8_C(  49), -INT8_C( 125),
        -INT8_C( 126),  INT8_C(  53),  INT8_C( 116),  INT8_C(  37), -INT8_C(  79),  INT8_C(  51),  INT8_C(  54), -INT8_C(  83) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 110), -INT8_C( 127),  INT8_C(  25),
         INT8_C(  54), -INT8_C(  55), -INT8_C(  28),  INT8_C( 117),  INT8_C(  55), -INT8_C(  77),  INT8_C(  52),  INT8_C(  43),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  14),  INT8_C( 113),  INT8_C(   5),
        -INT8_C(  32), -INT8_C(  91),  INT8_C( 110),  INT8_C(  91), -INT8_C(  81),  INT8_C(  95),  INT8_C(  39),  INT8_C(   4),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(   4), -INT8_C(  40),  INT8_C(  72),
        -INT8_C(  59), -INT8_C(  68), -INT8_C(  67), -INT8_C(   4),  INT8_C( 111), -INT8_C(  15),  INT8_C(  39),  INT8_C(  53),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  36), -INT8_C(  74), -INT8_C(   1),
        -INT8_C( 126),  INT8_C(  36),  INT8_C(  90),  INT8_C(  49), -INT8_C( 125), -INT8_C( 126),  INT8_C(  53),  INT8_C( 116) } },
    { {  INT8_C(  11),  INT8_C( 126),  INT8_C( 115), -INT8_C(  57),  INT8_C(  59),  INT8_C( 111),  INT8_C(  55),  INT8_C(  45),
        -INT8_C( 105),  INT8_C( 108),  INT8_C( 102), -INT8_C(  61), -INT8_C(  40),  INT8_C(  23), -INT8_C(  31), -INT8_C(  76),
        -INT8_C(  51), -INT8_C(  32),  INT8_C(  54), -INT8_C(  15),  INT8_C(  59),  INT8_C( 104),  INT8_C( 116), -INT8_C(  67),
        -INT8_C(  99), -INT8_C(  24), -INT8_C(  30),  INT8_C(  78),  INT8_C(  27),  INT8_C(  24), -INT8_C(   4),  INT8_C(  38),
        -INT8_C( 105),  INT8_C( 111), -INT8_C(  18), -INT8_C(  46), -INT8_C(  34),  INT8_C(  37), -INT8_C(   1),  INT8_C( 117),
        -INT8_C( 111),  INT8_C( 101),  INT8_C(  56),  INT8_C( 105),  INT8_C( 124),  INT8_C(  26),  INT8_C(  30),  INT8_C(  73),
        -INT8_C(   6),  INT8_C(  84),  INT8_C(  58),  INT8_C(  53), -INT8_C(  68), -INT8_C(  81), -INT8_C(  14),  INT8_C(  90),
        -INT8_C( 105), -INT8_C(  44), -INT8_C(  88), -INT8_C(  77), -INT8_C(  19), -INT8_C(  92), -INT8_C(  39), -INT8_C( 124) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  11),  INT8_C( 126),
         INT8_C( 115), -INT8_C(  57),  INT8_C(  59),  INT8_C( 111),  INT8_C(  55),  INT8_C(  45), -INT8_C( 105),  INT8_C( 108),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  51), -INT8_C(  32),
         INT8_C(  54), -INT8_C(  15),  INT8_C(  59),  INT8_C( 104),  INT8_C( 116), -INT8_C(  67), -INT8_C(  99), -INT8_C(  24),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 105),  INT8_C( 111),
        -INT8_C(  18), -INT8_C(  46), -INT8_C(  34),  INT8_C(  37), -INT8_C(   1),  INT8_C( 117), -INT8_C( 111),  INT8_C( 101),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(   6),  INT8_C(  84),
         INT8_C(  58),  INT8_C(  53), -INT8_C(  68), -INT8_C(  81), -INT8_C(  14),  INT8_C(  90), -INT8_C( 105), -INT8_C(  44) } },
    { {  INT8_C(  19), -INT8_C(  57),  INT8_C(  86), -INT8_C(  14), -INT8_C(  20),  INT8_C(  86),  INT8_C( 103),  INT8_C( 126),
        -INT8_C(  69), -INT8_C(  96), -INT8_C(  25),  INT8_C(  56), -INT8_C(  70),  INT8_C(   5), -INT8_C( 127), -INT8_C(  76),
         INT8_C(  90), -INT8_C(  68), -INT8_C(  22),  INT8_C(  22),  INT8_C( 107), -INT8_C(  36),  INT8_C( 112),  INT8_C(   2),
        -INT8_C(  79),  INT8_C(  25), -INT8_C(  75), -INT8_C(  98), -INT8_C(  67), -INT8_C( 113),  INT8_C(  34), -INT8_C(  47),
         INT8_C(  86),  INT8_C( 120), -INT8_C(  61),  INT8_C(  67), -INT8_C(  50),  INT8_C(  42), -INT8_C(  63), -INT8_C( 118),
        -INT8_C(  54), -INT8_C(  88), -INT8_C(  62), -INT8_C( 124), -INT8_C(  82),  INT8_C(  67),  INT8_C(  57),  INT8_C(   8),
        -INT8_C(   1),  INT8_C(  35),  INT8_C(  30),  INT8_C( 106), -INT8_C(   1), -INT8_C( 113),  INT8_C( 109), -INT8_C(  80),
        -INT8_C(  88),  INT8_C(  34),  INT8_C(  78),  INT8_C( 101), -INT8_C(  79),  INT8_C( 112),  INT8_C(  54),  INT8_C(   8) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  19),
        -INT8_C(  57),  INT8_C(  86), -INT8_C(  14), -INT8_C(  20),  INT8_C(  86),  INT8_C( 103),  INT8_C( 126), -INT8_C(  69),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  90),
        -INT8_C(  68), -INT8_C(  22),  INT8_C(  22),  INT8_C( 107), -INT8_C(  36),  INT8_C( 112),  INT8_C(   2), -INT8_C(  79),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  86),
         INT8_C( 120), -INT8_C(  61),  INT8_C(  67), -INT8_C(  50),  INT8_C(  42), -INT8_C(  63), -INT8_C( 118), -INT8_C(  54),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(   1),
         INT8_C(  35),  INT8_C(  30),  INT8_C( 106), -INT8_C(   1), -INT8_C( 113),  INT8_C( 109), -INT8_C(  80), -INT8_C(  88) } },
    { { -INT8_C(  23), -INT8_C(   7),  INT8_C(  75), -INT8_C(  73),  INT8_C(  36),  INT8_C(  12),  INT8_C(  65), -INT8_C(  18),
        -INT8_C(  76),  INT8_C(   3),  INT8_C( 115),  INT8_C(  98),  INT8_C(  71), -INT8_C(  84),  INT8_C( 106),  INT8_C(  70),
        -INT8_C(  49), -INT8_C( 119), -INT8_C(  79), -INT8_C(  50),  INT8_C(  24),  INT8_C(  30),      INT8_MAX, -INT8_C(  64),
         INT8_C(  64), -INT8_C(  51),  INT8_C(  37), -INT8_C(  14),  INT8_C(  62),  INT8_C(  92), -INT8_C(   6),  INT8_C(  39),
         INT8_C(  85),  INT8_C(  69), -INT8_C(  34),  INT8_C( 121),  INT8_C(  81),  INT8_C(  32),  INT8_C( 104),  INT8_C(   5),
         INT8_C(  35), -INT8_C(  37),  INT8_C( 104),  INT8_C( 106), -INT8_C( 121), -INT8_C(  46), -INT8_C(  79),  INT8_C(  86),
         INT8_C(  91),  INT8_C(  98),  INT8_C(  36),  INT8_C( 115),      INT8_MIN, -INT8_C(  93),  INT8_C(  51), -INT8_C(  64),
         INT8_C( 113),  INT8_C(  89), -INT8_C(  78), -INT8_C(  81), -INT8_C(  75), -INT8_C(  84), -INT8_C(  42),  INT8_C(  10) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  23), -INT8_C(   7),  INT8_C(  75), -INT8_C(  73),  INT8_C(  36),  INT8_C(  12),  INT8_C(  65), -INT8_C(  18),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  49), -INT8_C( 119), -INT8_C(  79), -INT8_C(  50),  INT8_C(  24),  INT8_C(  30),      INT8_MAX, -INT8_C(  64),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  85),  INT8_C(  69), -INT8_C(  34),  INT8_C( 121),  INT8_C(  81),  INT8_C(  32),  INT8_C( 104),  INT8_C(   5),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  91),  INT8_C(  98),  INT8_C(  36),  INT8_C( 115),      INT8_MIN, -INT8_C(  93),  INT8_C(  51), -INT8_C(  64) } }
  };

  easysimd__m512i a;
  easysimd__m512i r;

  a = easysimd_mm512_loadu_epi8(test_vec[0].a);
  r = easysimd_mm512_bslli_epi128(a,  1);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[0].r));
  
  a = easysimd_mm512_loadu_epi8(test_vec[1].a);
  r = easysimd_mm512_bslli_epi128(a,  2);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[1].r));

  a = easysimd_mm512_loadu_epi8(test_vec[2].a);
  r = easysimd_mm512_bslli_epi128(a,  3);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[2].r));

  a = easysimd_mm512_loadu_epi8(test_vec[3].a);
  r = easysimd_mm512_bslli_epi128(a,  4);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[3].r));

  a = easysimd_mm512_loadu_epi8(test_vec[4].a);
  r = easysimd_mm512_bslli_epi128(a,  5);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[4].r));

  a = easysimd_mm512_loadu_epi8(test_vec[5].a);
  r = easysimd_mm512_bslli_epi128(a,  6);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[5].r));

  a = easysimd_mm512_loadu_epi8(test_vec[6].a);
  r = easysimd_mm512_bslli_epi128(a,  7);
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[6].r));

  a = easysimd_mm512_loadu_epi8(test_vec[7].a);
  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r = easysimd_mm512_bslli_epi128(a,  8);
  } EASYSIMD_TEST_PERF_END("easysimd_mm512_bslli_epi128");
  easysimd_assert_m512i_i8(r, ==, easysimd_mm512_loadu_epi8(test_vec[7].r));

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i r = easysimd_mm512_bslli_epi128(a, (i + 1));

    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sll_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sll_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sll_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sll_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sll_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sll_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sll_epi16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sll_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sll_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sll_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sll_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sll_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sll_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sll_epi32)

 EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sll_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sll_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sll_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sll_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sll_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sll_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sll_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_bslli_epi128)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
