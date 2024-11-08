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
 *   2020      Christopher Moore <moore@free.fr>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN mov

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/mov.h>

static int
test_easysimd_mm_mask_mov_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i src;
    easysimd__mmask16 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi8(INT8_C( -98), INT8_C( -63), INT8_C( -58), INT8_C(  25),
                        INT8_C(   9), INT8_C(  49), INT8_C( -12), INT8_C( -31),
                        INT8_C( -48), INT8_C(   0), INT8_C( -99), INT8_C( -81),
                        INT8_C( -77), INT8_C(  27), INT8_C( -33), INT8_C(-124)),
      UINT16_C(64699),
     easysimd_mm_set_epi8(INT8_C(  79), INT8_C( 100), INT8_C(-123), INT8_C(  95),
                        INT8_C( -79), INT8_C(  48), INT8_C( 112), INT8_C(   8),
                        INT8_C(  69), INT8_C( -78), INT8_C(  54), INT8_C( -48),
                        INT8_C(-119), INT8_C(  -5), INT8_C( -97), INT8_C( -44)),
      easysimd_mm_set_epi8(INT8_C(  79), INT8_C( 100), INT8_C(-123), INT8_C(  95),
                        INT8_C( -79), INT8_C(  48), INT8_C( -12), INT8_C( -31),
                        INT8_C(  69), INT8_C(   0), INT8_C(  54), INT8_C( -48),
                        INT8_C(-119), INT8_C(  27), INT8_C( -97), INT8_C( -44)) },
    { easysimd_mm_set_epi8(INT8_C(  47), INT8_C(  36), INT8_C(  45), INT8_C( -44),
                        INT8_C(  27), INT8_C( -15), INT8_C( 105), INT8_C( -69),
                        INT8_C( -10), INT8_C(   1), INT8_C(  12), INT8_C( -44),
                        INT8_C( -32), INT8_C( 113), INT8_C( 105), INT8_C( -92)),
      UINT16_C(33046),
     easysimd_mm_set_epi8(INT8_C(-118), INT8_C( -61), INT8_C( 108), INT8_C(   4),
                        INT8_C(  56), INT8_C(  96), INT8_C( -73), INT8_C( -39),
                        INT8_C(-112), INT8_C(-115), INT8_C(-113), INT8_C( -74),
                        INT8_C( -79), INT8_C(-116), INT8_C( 117), INT8_C(  -3)),
      easysimd_mm_set_epi8(INT8_C(-118), INT8_C(  36), INT8_C(  45), INT8_C( -44),
                        INT8_C(  27), INT8_C( -15), INT8_C( 105), INT8_C( -39),
                        INT8_C( -10), INT8_C(   1), INT8_C(  12), INT8_C( -74),
                        INT8_C( -32), INT8_C(-116), INT8_C( 117), INT8_C( -92)) },
    { easysimd_mm_set_epi8(INT8_C(  41), INT8_C(-106), INT8_C( -67), INT8_C(-116),
                        INT8_C( -34), INT8_C(  21), INT8_C(  64), INT8_C(  44),
                        INT8_C(  97), INT8_C( -46), INT8_C( 122), INT8_C(  42),
                        INT8_C( -54), INT8_C( -79), INT8_C(  21), INT8_C(  59)),
      UINT16_C(27487),
     easysimd_mm_set_epi8(INT8_C(   6), INT8_C(-124), INT8_C(-111), INT8_C( -39),
                        INT8_C(  55), INT8_C( -55), INT8_C( -72), INT8_C(  77),
                        INT8_C(  51), INT8_C(-103), INT8_C( -80), INT8_C(  75),
                        INT8_C( -87), INT8_C(-120), INT8_C( -14), INT8_C(  99)),
      easysimd_mm_set_epi8(INT8_C(  41), INT8_C(-124), INT8_C(-111), INT8_C(-116),
                        INT8_C(  55), INT8_C(  21), INT8_C( -72), INT8_C(  77),
                        INT8_C(  97), INT8_C(-103), INT8_C( 122), INT8_C(  75),
                        INT8_C( -87), INT8_C(-120), INT8_C( -14), INT8_C(  99)) },
    { easysimd_mm_set_epi8(INT8_C(  31), INT8_C( -90), INT8_C(-127), INT8_C( 105),
                        INT8_C( -89), INT8_C(-121), INT8_C(-110), INT8_C( -58),
                        INT8_C( -95), INT8_C(-101), INT8_C( -56), INT8_C(  22),
                        INT8_C(  18), INT8_C(   2), INT8_C(  46), INT8_C(-125)),
      UINT16_C(48165),
     easysimd_mm_set_epi8(INT8_C( 103), INT8_C(  26), INT8_C( 108), INT8_C(   4),
                        INT8_C( -49), INT8_C( -62), INT8_C(-103), INT8_C( -42),
                        INT8_C( 103), INT8_C( 115), INT8_C( 126), INT8_C(-112),
                        INT8_C( -81), INT8_C( -35), INT8_C(-106), INT8_C(  45)),
      easysimd_mm_set_epi8(INT8_C( 103), INT8_C( -90), INT8_C( 108), INT8_C(   4),
                        INT8_C( -49), INT8_C( -62), INT8_C(-110), INT8_C( -58),
                        INT8_C( -95), INT8_C(-101), INT8_C( 126), INT8_C(  22),
                        INT8_C(  18), INT8_C( -35), INT8_C(  46), INT8_C(  45)) },
    { easysimd_mm_set_epi8(INT8_C( 106), INT8_C(  23), INT8_C( -78), INT8_C( -57),
                        INT8_C(  24), INT8_C(  56), INT8_C( -46), INT8_C( -15),
                        INT8_C( -33), INT8_C(  28), INT8_C( -40), INT8_C(-116),
                        INT8_C( -34), INT8_C(  92), INT8_C( 109), INT8_C(  33)),
      UINT16_C(14870),
     easysimd_mm_set_epi8(INT8_C( -75), INT8_C(  55), INT8_C(-127), INT8_C(  70),
                        INT8_C(  78), INT8_C( 126), INT8_C( -96), INT8_C( 119),
                        INT8_C( 108), INT8_C(  50), INT8_C(  17), INT8_C( -71),
                        INT8_C( 127), INT8_C(  91), INT8_C( 110), INT8_C( -90)),
      easysimd_mm_set_epi8(INT8_C( 106), INT8_C(  23), INT8_C(-127), INT8_C(  70),
                        INT8_C(  78), INT8_C(  56), INT8_C( -96), INT8_C( -15),
                        INT8_C( -33), INT8_C(  28), INT8_C( -40), INT8_C( -71),
                        INT8_C( -34), INT8_C(  91), INT8_C( 110), INT8_C(  33)) },
    { easysimd_mm_set_epi8(INT8_C( -21), INT8_C(-122), INT8_C(-127), INT8_C(  95),
                        INT8_C( -34), INT8_C( -51), INT8_C( 107), INT8_C(  75),
                        INT8_C(  63), INT8_C(-117), INT8_C(-118), INT8_C(  52),
                        INT8_C(  15), INT8_C( 123), INT8_C( -76), INT8_C(-117)),
      UINT16_C(54314),
     easysimd_mm_set_epi8(INT8_C( 124), INT8_C( -12), INT8_C(   0), INT8_C( -14),
                        INT8_C( -54), INT8_C(  92), INT8_C(  73), INT8_C(  69),
                        INT8_C( -47), INT8_C( -62), INT8_C( 113), INT8_C( 100),
                        INT8_C(  31), INT8_C( -98), INT8_C( -86), INT8_C(  19)),
      easysimd_mm_set_epi8(INT8_C( 124), INT8_C( -12), INT8_C(-127), INT8_C( -14),
                        INT8_C( -34), INT8_C(  92), INT8_C( 107), INT8_C(  75),
                        INT8_C(  63), INT8_C(-117), INT8_C( 113), INT8_C(  52),
                        INT8_C(  31), INT8_C( 123), INT8_C( -86), INT8_C(-117)) },
    { easysimd_mm_set_epi8(INT8_C(  -9), INT8_C( -43), INT8_C(  83), INT8_C(  21),
                        INT8_C(  88), INT8_C( -52), INT8_C(-115), INT8_C(  63),
                        INT8_C(  92), INT8_C( -15), INT8_C( -24), INT8_C( -84),
                        INT8_C(-120), INT8_C( -96), INT8_C(  46), INT8_C( -78)),
      UINT16_C(44998),
     easysimd_mm_set_epi8(INT8_C( -10), INT8_C(  79), INT8_C(-113), INT8_C( -93),
                        INT8_C(  24), INT8_C(  78), INT8_C(  40), INT8_C(  22),
                        INT8_C(  31), INT8_C( -15), INT8_C(  -8), INT8_C(  60),
                        INT8_C( 114), INT8_C( -85), INT8_C(-105), INT8_C( -47)),
      easysimd_mm_set_epi8(INT8_C( -10), INT8_C( -43), INT8_C(-113), INT8_C(  21),
                        INT8_C(  24), INT8_C(  78), INT8_C(  40), INT8_C(  22),
                        INT8_C(  31), INT8_C( -15), INT8_C( -24), INT8_C( -84),
                        INT8_C(-120), INT8_C( -85), INT8_C(-105), INT8_C( -78)) },
    { easysimd_mm_set_epi8(INT8_C( -62), INT8_C( 117), INT8_C(-114), INT8_C(   7),
                        INT8_C(  17), INT8_C( 123), INT8_C(  -2), INT8_C( -15),
                        INT8_C(-120), INT8_C(  77), INT8_C(  81), INT8_C( -39),
                        INT8_C(-114), INT8_C( -52), INT8_C(-119), INT8_C(  82)),
      UINT16_C(48425),
     easysimd_mm_set_epi8(INT8_C(  68), INT8_C( -65), INT8_C(  13), INT8_C( -27),
                        INT8_C(  55), INT8_C(   2), INT8_C( -43), INT8_C(   9),
                        INT8_C( -57), INT8_C(  65), INT8_C(-111), INT8_C( -60),
                        INT8_C(  75), INT8_C(  74), INT8_C(  16), INT8_C(  19)),
      easysimd_mm_set_epi8(INT8_C(  68), INT8_C( 117), INT8_C(  13), INT8_C( -27),
                        INT8_C(  55), INT8_C(   2), INT8_C(  -2), INT8_C(   9),
                        INT8_C(-120), INT8_C(  77), INT8_C(-111), INT8_C( -39),
                        INT8_C(  75), INT8_C( -52), INT8_C(-119), INT8_C(  19)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mov_epi8(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mov_epi8");
    easysimd_assert_m128i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_mask_mov_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i src;
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi16(INT16_C(-14576), INT16_C( 14205), INT16_C( -2433), INT16_C(-27972),
                         INT16_C(  6192), INT16_C(-29093), INT16_C( 24144), INT16_C(-19045)),
      UINT8_C(231),
      easysimd_mm_set_epi16(INT16_C(-16025), INT16_C( -5226), INT16_C( -6596), INT16_C(-11796),
                         INT16_C(-24692), INT16_C( 20335), INT16_C( 26237), INT16_C( 23499)),
      easysimd_mm_set_epi16(INT16_C(-16025), INT16_C( -5226), INT16_C( -6596), INT16_C(-27972),
                         INT16_C(  6192), INT16_C( 20335), INT16_C( 26237), INT16_C( 23499)) },
    { easysimd_mm_set_epi16(INT16_C(  -839), INT16_C(-10951), INT16_C(  1310), INT16_C( -6285),
                         INT16_C(-21252), INT16_C( -7582), INT16_C(-12381), INT16_C( 24902)),
      UINT8_C(  7),
      easysimd_mm_set_epi16(INT16_C( -3233), INT16_C( 25022), INT16_C(-12043), INT16_C( 17022),
                         INT16_C(-25543), INT16_C(-17145), INT16_C(  8881), INT16_C( 28844)),
      easysimd_mm_set_epi16(INT16_C(  -839), INT16_C(-10951), INT16_C(  1310), INT16_C( -6285),
                         INT16_C(-21252), INT16_C(-17145), INT16_C(  8881), INT16_C( 28844)) },
    { easysimd_mm_set_epi16(INT16_C( 30807), INT16_C( 12936), INT16_C(-14387), INT16_C(-15179),
                         INT16_C( 23907), INT16_C(-17160), INT16_C( 23916), INT16_C( 14132)),
      UINT8_C(139),
      easysimd_mm_set_epi16(INT16_C( -1315), INT16_C(-31661), INT16_C(-10075), INT16_C(-22609),
                         INT16_C(  9167), INT16_C(  6456), INT16_C( -7329), INT16_C( -8326)),
      easysimd_mm_set_epi16(INT16_C( -1315), INT16_C( 12936), INT16_C(-14387), INT16_C(-15179),
                         INT16_C(  9167), INT16_C(-17160), INT16_C( -7329), INT16_C( -8326)) },
    { easysimd_mm_set_epi16(INT16_C( 26421), INT16_C(-12708), INT16_C( 22525), INT16_C(-31426),
                         INT16_C( 15010), INT16_C(-27490), INT16_C(-12766), INT16_C(-25791)),
      UINT8_C( 65),
      easysimd_mm_set_epi16(INT16_C( -1553), INT16_C(-19304), INT16_C( 20094), INT16_C( -2808),
                         INT16_C(-12327), INT16_C( 15252), INT16_C( 25789), INT16_C(-23968)),
      easysimd_mm_set_epi16(INT16_C( 26421), INT16_C(-19304), INT16_C( 22525), INT16_C(-31426),
                         INT16_C( 15010), INT16_C(-27490), INT16_C(-12766), INT16_C(-23968)) },
    { easysimd_mm_set_epi16(INT16_C(  7823), INT16_C( 19443), INT16_C( 13219), INT16_C( 17015),
                         INT16_C(-11739), INT16_C(-13030), INT16_C(-14482), INT16_C(-27926)),
      UINT8_C(249),
      easysimd_mm_set_epi16(INT16_C(-25131), INT16_C( 30189), INT16_C(-22900), INT16_C( 28700),
                         INT16_C(  1116), INT16_C( 30184), INT16_C(-12164), INT16_C( -7443)),
      easysimd_mm_set_epi16(INT16_C(-25131), INT16_C( 30189), INT16_C(-22900), INT16_C( 28700),
                         INT16_C(  1116), INT16_C(-13030), INT16_C(-14482), INT16_C( -7443)) },
    { easysimd_mm_set_epi16(INT16_C(-26628), INT16_C( 25963), INT16_C(-26322), INT16_C( -8077),
                         INT16_C(-22868), INT16_C( 28633), INT16_C( -4168), INT16_C( 28595)),
      UINT8_C(112),
      easysimd_mm_set_epi16(INT16_C( 14185), INT16_C( -5351), INT16_C( -8435), INT16_C(-11233),
                         INT16_C( -8273), INT16_C(-29718), INT16_C( -8221), INT16_C( 18236)),
      easysimd_mm_set_epi16(INT16_C(-26628), INT16_C( -5351), INT16_C( -8435), INT16_C(-11233),
                         INT16_C(-22868), INT16_C( 28633), INT16_C( -4168), INT16_C( 28595)) },
    { easysimd_mm_set_epi16(INT16_C(-14557), INT16_C(-28064), INT16_C( 11696), INT16_C(-19213),
                         INT16_C( 15613), INT16_C( 26380), INT16_C( 30063), INT16_C( 26293)),
      UINT8_C( 24),
      easysimd_mm_set_epi16(INT16_C( 23790), INT16_C( 10772), INT16_C( -8418), INT16_C(-27527),
                         INT16_C(  -163), INT16_C( 10898), INT16_C(-12995), INT16_C(   287)),
      easysimd_mm_set_epi16(INT16_C(-14557), INT16_C(-28064), INT16_C( 11696), INT16_C(-27527),
                         INT16_C(  -163), INT16_C( 26380), INT16_C( 30063), INT16_C( 26293)) },
    { easysimd_mm_set_epi16(INT16_C(-14768), INT16_C(-23816), INT16_C(-22775), INT16_C( -4812),
                         INT16_C(-19595), INT16_C(-14349), INT16_C( 11039), INT16_C( 15081)),
      UINT8_C( 22),
      easysimd_mm_set_epi16(INT16_C( 27063), INT16_C(  8226), INT16_C(-13582), INT16_C( 14344),
                         INT16_C(-27643), INT16_C( -1125), INT16_C(-27147), INT16_C( -4132)),
      easysimd_mm_set_epi16(INT16_C(-14768), INT16_C(-23816), INT16_C(-22775), INT16_C( 14344),
                         INT16_C(-19595), INT16_C( -1125), INT16_C(-27147), INT16_C( 15081)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mov_epi16(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mov_epi16");
    easysimd_assert_m128i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_mask_mov_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i src;
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
       { easysimd_mm_set_epi32(INT32_C(-1311777535), INT32_C(  871351059), INT32_C(-1795529748), INT32_C(-1018886524)),
      UINT8_C(193),
      easysimd_mm_set_epi32(INT32_C(-1402384713), INT32_C(  349677639), INT32_C(-2062419968), INT32_C(-2110667873)),
      easysimd_mm_set_epi32(INT32_C(-1311777535), INT32_C(  871351059), INT32_C(-1795529748), INT32_C(-2110667873)) },
    { easysimd_mm_set_epi32(INT32_C(  738895957), INT32_C(-2052149671), INT32_C( 1275190943), INT32_C(-1073987906)),
      UINT8_C(211),
      easysimd_mm_set_epi32(INT32_C(  899624021), INT32_C(-1740875066), INT32_C(  196568235), INT32_C(  146964985)),
      easysimd_mm_set_epi32(INT32_C(  738895957), INT32_C(-2052149671), INT32_C(  196568235), INT32_C(  146964985)) },
    { easysimd_mm_set_epi32(INT32_C(  692992965), INT32_C(  836600954), INT32_C(-1461227321), INT32_C( -625910795)),
      UINT8_C(122),
      easysimd_mm_set_epi32(INT32_C(-1617549669), INT32_C( 1989374100), INT32_C(-1502577107), INT32_C(-1017994073)),
      easysimd_mm_set_epi32(INT32_C(-1617549669), INT32_C(  836600954), INT32_C(-1502577107), INT32_C( -625910795)) },
    { easysimd_mm_set_epi32(INT32_C( 1143677167), INT32_C(  846204550), INT32_C( -804913221), INT32_C( 1445583278)),
      UINT8_C(231),
      easysimd_mm_set_epi32(INT32_C(-1730413187), INT32_C(-1695584840), INT32_C( -227526716), INT32_C(   -3425875)),
      easysimd_mm_set_epi32(INT32_C( 1143677167), INT32_C(-1695584840), INT32_C( -227526716), INT32_C(   -3425875)) },
    { easysimd_mm_set_epi32(INT32_C(  645689114), INT32_C(-2084714818), INT32_C( 1764055823), INT32_C(   52635923)),
      UINT8_C( 92),
      easysimd_mm_set_epi32(INT32_C(-1571852402), INT32_C(  630152776), INT32_C( -128726906), INT32_C( 1269444726)),
      easysimd_mm_set_epi32(INT32_C(-1571852402), INT32_C(  630152776), INT32_C( 1764055823), INT32_C(   52635923)) },
    { easysimd_mm_set_epi32(INT32_C(    1563221), INT32_C( -134802286), INT32_C(  714712077), INT32_C(-1827172967)),
      UINT8_C( 81),
      easysimd_mm_set_epi32(INT32_C( 1929131576), INT32_C(-1816110300), INT32_C( 1278219947), INT32_C( 1799312980)),
      easysimd_mm_set_epi32(INT32_C(    1563221), INT32_C( -134802286), INT32_C(  714712077), INT32_C( 1799312980)) },
    { easysimd_mm_set_epi32(INT32_C(  398082434), INT32_C(-1574168894), INT32_C(  -78364073), INT32_C(-1210427726)),
      UINT8_C( 81),
      easysimd_mm_set_epi32(INT32_C( -743499294), INT32_C(-2007549651), INT32_C(  404949426), INT32_C(-1228263526)),
      easysimd_mm_set_epi32(INT32_C(  398082434), INT32_C(-1574168894), INT32_C(  -78364073), INT32_C(-1228263526)) },
    { easysimd_mm_set_epi32(INT32_C( -588057094), INT32_C(-1885829296), INT32_C( 1969228625), INT32_C( 1326338893)),
      UINT8_C(219),
      easysimd_mm_set_epi32(INT32_C( 1932026039), INT32_C(-1013786585), INT32_C( 1485053584), INT32_C( 1979373999)),
      easysimd_mm_set_epi32(INT32_C( 1932026039), INT32_C(-1885829296), INT32_C( 1485053584), INT32_C( 1979373999)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mov_epi32(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mov_epi32");
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_mask_mov_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i src;
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
       { easysimd_mm_set_epi64x(INT64_C( 2277107027088284737), INT64_C( -794576880036979785)),
      UINT8_C(133),
      easysimd_mm_set_epi64x(INT64_C( 8097713530582561529), INT64_C( 1585963766693842069)),
      easysimd_mm_set_epi64x(INT64_C( 2277107027088284737), INT64_C( 1585963766693842069)) },
    { easysimd_mm_set_epi64x(INT64_C(  386114209698075166), INT64_C( 5207265957388900927)),
      UINT8_C(158),
      easysimd_mm_set_epi64x(INT64_C( 8803705323655107871), INT64_C(-8422781366242531322)),
      easysimd_mm_set_epi64x(INT64_C( 8803705323655107871), INT64_C( 5207265957388900927)) },
    { easysimd_mm_set_epi64x(INT64_C(-2685854854617637911), INT64_C( 5000183764696508529)),
      UINT8_C(188),
      easysimd_mm_set_epi64x(INT64_C( 3366037084418714211), INT64_C(-4379786006937181803)),
      easysimd_mm_set_epi64x(INT64_C(-2685854854617637911), INT64_C( 5000183764696508529)) },
    { easysimd_mm_set_epi64x(INT64_C( 5087362917606608352), INT64_C( 7748994405920281726)),
      UINT8_C( 72),
      easysimd_mm_set_epi64x(INT64_C(-3993157906773187111), INT64_C( 5848124444216740966)),
      easysimd_mm_set_epi64x(INT64_C( 5087362917606608352), INT64_C( 7748994405920281726)) },
    { easysimd_mm_set_epi64x(INT64_C(-6262495515547444433), INT64_C( 3943684472219148405)),
      UINT8_C( 56),
      easysimd_mm_set_epi64x(INT64_C( 6021985363878171356), INT64_C(-9003751561505293092)),
      easysimd_mm_set_epi64x(INT64_C(-6262495515547444433), INT64_C( 3943684472219148405)) },
    { easysimd_mm_set_epi64x(INT64_C( 7378184861631570903), INT64_C( 5065745925883054243)),
      UINT8_C(107),
      easysimd_mm_set_epi64x(INT64_C( 3940656342452910480), INT64_C( 3350136105944417294)),
      easysimd_mm_set_epi64x(INT64_C( 3940656342452910480), INT64_C( 3350136105944417294)) },
    { easysimd_mm_set_epi64x(INT64_C( 4422823463426654219), INT64_C( 1827699444722609855)),
      UINT8_C( 23),
      easysimd_mm_set_epi64x(INT64_C(-2966751886069965026), INT64_C(-8494473672325004777)),
      easysimd_mm_set_epi64x(INT64_C(-2966751886069965026), INT64_C(-8494473672325004777)) },
    { easysimd_mm_set_epi64x(INT64_C(-8917676865649705108), INT64_C( 6229148348133862992)),
      UINT8_C( 48),
      easysimd_mm_set_epi64x(INT64_C(-7968457113297908477), INT64_C(-6793891334661924961)),
      easysimd_mm_set_epi64x(INT64_C(-8917676865649705108), INT64_C( 6229148348133862992)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mov_epi64(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mov_epi64");
    easysimd_assert_m128i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_mask_mov_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d src;
    easysimd__mmask8 k;
    easysimd__m128d a;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C(210),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   55.56), EASYSIMD_FLOAT64_C(  306.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   55.56), EASYSIMD_FLOAT64_C(    0.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C(  7),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  202.21), EASYSIMD_FLOAT64_C( -678.71)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  202.21), EASYSIMD_FLOAT64_C( -678.71)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C( 50),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  680.40), EASYSIMD_FLOAT64_C(  906.67)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  680.40), EASYSIMD_FLOAT64_C(    0.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C(229),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -422.72), EASYSIMD_FLOAT64_C(  572.83)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  572.83)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C(117),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -76.19), EASYSIMD_FLOAT64_C( -654.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -654.60)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C(130),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -711.42), EASYSIMD_FLOAT64_C(  -22.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -711.42), EASYSIMD_FLOAT64_C(    0.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C( 62),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -413.23), EASYSIMD_FLOAT64_C(  547.52)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -413.23), EASYSIMD_FLOAT64_C(    0.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)),
      UINT8_C(165),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  575.41), EASYSIMD_FLOAT64_C( -702.01)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -702.01)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = test_vec[i].a;
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mov_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mov_pd");
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_mask_mov_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 src;
    easysimd__mmask8 k;
    easysimd__m128 a;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(  126),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -678.71), EASYSIMD_FLOAT32_C(   675.53), EASYSIMD_FLOAT32_C(    55.56), EASYSIMD_FLOAT32_C(   306.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -678.71), EASYSIMD_FLOAT32_C(   675.53), EASYSIMD_FLOAT32_C(    55.56), EASYSIMD_FLOAT32_C(     0.00)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(   44),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   941.87), EASYSIMD_FLOAT32_C(   680.40), EASYSIMD_FLOAT32_C(   906.67), EASYSIMD_FLOAT32_C(  -364.25)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   941.87), EASYSIMD_FLOAT32_C(   680.40), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(  117),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -76.19), EASYSIMD_FLOAT32_C(  -654.60), EASYSIMD_FLOAT32_C(  -721.91), EASYSIMD_FLOAT32_C(  -422.72)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -654.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -422.72)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(   76),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   547.52), EASYSIMD_FLOAT32_C(  -627.17), EASYSIMD_FLOAT32_C(  -711.42), EASYSIMD_FLOAT32_C(   -22.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   547.52), EASYSIMD_FLOAT32_C(  -627.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(  101),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -822.97), EASYSIMD_FLOAT32_C(   575.41), EASYSIMD_FLOAT32_C(  -702.01), EASYSIMD_FLOAT32_C(  -488.76)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   575.41), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -488.76)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(  149),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   804.55), EASYSIMD_FLOAT32_C(  -888.85), EASYSIMD_FLOAT32_C(   750.71), EASYSIMD_FLOAT32_C(   346.51)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -888.85), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   346.51)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(  115),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -17.38), EASYSIMD_FLOAT32_C(   623.33), EASYSIMD_FLOAT32_C(   459.80), EASYSIMD_FLOAT32_C(   837.15)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   459.80), EASYSIMD_FLOAT32_C(   837.15)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)),
      UINT16_C(   50),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   197.69), EASYSIMD_FLOAT32_C(   233.42), EASYSIMD_FLOAT32_C(   153.73), EASYSIMD_FLOAT32_C(   616.58)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   153.73), EASYSIMD_FLOAT32_C(     0.00)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = test_vec[i].a;
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mov_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mov_ps");
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_mov_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i src;
    easysimd__mmask32 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi8(INT8_C( -82), INT8_C( -32), INT8_C( -73), INT8_C( -78),
                           INT8_C( -21), INT8_C(  76), INT8_C(  33), INT8_C(  90),
                           INT8_C( -57), INT8_C( -12), INT8_C(-121), INT8_C( 101),
                           INT8_C(   6), INT8_C( -36), INT8_C( -50), INT8_C( -33),
                           INT8_C( -83), INT8_C( -92), INT8_C(   2), INT8_C(  69),
                           INT8_C(  62), INT8_C(  89), INT8_C( 105), INT8_C(  58),
                           INT8_C( 125), INT8_C( -76), INT8_C(  27), INT8_C(  51),
                           INT8_C(  79), INT8_C( 101), INT8_C( -42), INT8_C( -69)),
      UINT32_C( 391141390),
      easysimd_mm256_set_epi8(INT8_C(  -9), INT8_C( -52), INT8_C(  27), INT8_C( -40),
                           INT8_C(  57), INT8_C( -80), INT8_C( -28), INT8_C(  64),
                           INT8_C(  70), INT8_C( -40), INT8_C(  14), INT8_C( -38),
                           INT8_C( -38), INT8_C( -99), INT8_C( -37), INT8_C( -35),
                           INT8_C( -82), INT8_C( -60), INT8_C( -40), INT8_C( -40),
                           INT8_C(  -5), INT8_C(   8), INT8_C( 109), INT8_C(  95),
                           INT8_C( 124), INT8_C(  34), INT8_C(  19), INT8_C( -65),
                           INT8_C(  -2), INT8_C( -92), INT8_C(  18), INT8_C( -33)),
      easysimd_mm256_set_epi8(INT8_C( -82), INT8_C( -32), INT8_C( -73), INT8_C( -40),
                           INT8_C( -21), INT8_C( -80), INT8_C( -28), INT8_C(  64),
                           INT8_C( -57), INT8_C( -40), INT8_C(-121), INT8_C( -38),
                           INT8_C(   6), INT8_C( -36), INT8_C( -50), INT8_C( -33),
                           INT8_C( -83), INT8_C( -60), INT8_C(   2), INT8_C( -40),
                           INT8_C(  -5), INT8_C(  89), INT8_C( 105), INT8_C(  58),
                           INT8_C( 125), INT8_C( -76), INT8_C(  27), INT8_C(  51),
                           INT8_C(  -2), INT8_C( -92), INT8_C(  18), INT8_C( -69)) },
    { easysimd_mm256_set_epi8(INT8_C( -54), INT8_C( -68), INT8_C(  19), INT8_C(  39),
                           INT8_C(  17), INT8_C( -32), INT8_C( -47), INT8_C( -26),
                           INT8_C( -23), INT8_C(  30), INT8_C(  98), INT8_C(   3),
                           INT8_C( -92), INT8_C( -30), INT8_C(  -8), INT8_C( -30),
                           INT8_C(  26), INT8_C(-116), INT8_C(  76), INT8_C( -76),
                           INT8_C( -29), INT8_C( -31), INT8_C( -31), INT8_C(  78),
                           INT8_C(  23), INT8_C(   6), INT8_C(  61), INT8_C(  68),
                           INT8_C( -53), INT8_C(-110), INT8_C(  53), INT8_C( -67)),
      UINT32_C( 757878650),
      easysimd_mm256_set_epi8(INT8_C(  36), INT8_C(-115), INT8_C( -95), INT8_C(   4),
                           INT8_C(  50), INT8_C( -54), INT8_C(  94), INT8_C(  54),
                           INT8_C( 109), INT8_C(-103), INT8_C(-124), INT8_C(  34),
                           INT8_C( -16), INT8_C(  97), INT8_C(  -7), INT8_C(  98),
                           INT8_C(-125), INT8_C( -49), INT8_C(   3), INT8_C( -91),
                           INT8_C( -99), INT8_C(  85), INT8_C( -25), INT8_C(   6),
                           INT8_C( -42), INT8_C(  44), INT8_C(  70), INT8_C( -24),
                           INT8_C( -86), INT8_C( 112), INT8_C( 116), INT8_C( -61)),
      easysimd_mm256_set_epi8(INT8_C( -54), INT8_C( -68), INT8_C( -95), INT8_C(  39),
                           INT8_C(  50), INT8_C( -54), INT8_C( -47), INT8_C(  54),
                           INT8_C( -23), INT8_C(  30), INT8_C(-124), INT8_C(   3),
                           INT8_C( -16), INT8_C(  97), INT8_C(  -8), INT8_C( -30),
                           INT8_C(  26), INT8_C( -49), INT8_C(  76), INT8_C( -76),
                           INT8_C( -99), INT8_C(  85), INT8_C( -25), INT8_C(   6),
                           INT8_C(  23), INT8_C(  44), INT8_C(  70), INT8_C( -24),
                           INT8_C( -86), INT8_C(-110), INT8_C( 116), INT8_C( -67)) },
    { easysimd_mm256_set_epi8(INT8_C(  48), INT8_C( -19), INT8_C( -87), INT8_C( 100),
                           INT8_C( -44), INT8_C( -79), INT8_C( -72), INT8_C(  73),
                           INT8_C( -36), INT8_C(  58), INT8_C(-113), INT8_C( -42),
                           INT8_C( -85), INT8_C( 123), INT8_C(-106), INT8_C( -57),
                           INT8_C( -53), INT8_C(  96), INT8_C(  40), INT8_C( -52),
                           INT8_C( -17), INT8_C(  -6), INT8_C(-108), INT8_C(  33),
                           INT8_C( -15), INT8_C( 113), INT8_C(  31), INT8_C( -14),
                           INT8_C( 124), INT8_C(  15), INT8_C(  90), INT8_C(   1)),
      UINT32_C(2771863762),
      easysimd_mm256_set_epi8(INT8_C(  72), INT8_C( -95), INT8_C( 104), INT8_C( -28),
                           INT8_C(  25), INT8_C(  84), INT8_C(  66), INT8_C(  19),
                           INT8_C(  79), INT8_C( -84), INT8_C(  46), INT8_C(  23),
                           INT8_C( -85), INT8_C(  12), INT8_C(   6), INT8_C(  -9),
                           INT8_C(-108), INT8_C(  14), INT8_C( 103), INT8_C(  32),
                           INT8_C(  25), INT8_C(-108), INT8_C( -56), INT8_C(-111),
                           INT8_C(  23), INT8_C( -20), INT8_C(   4), INT8_C(  81),
                           INT8_C(  39), INT8_C(  39), INT8_C(  82), INT8_C( -15)),
      easysimd_mm256_set_epi8(INT8_C(  72), INT8_C( -19), INT8_C( 104), INT8_C( 100),
                           INT8_C( -44), INT8_C(  84), INT8_C( -72), INT8_C(  19),
                           INT8_C( -36), INT8_C(  58), INT8_C(  46), INT8_C(  23),
                           INT8_C( -85), INT8_C(  12), INT8_C(   6), INT8_C(  -9),
                           INT8_C( -53), INT8_C(  14), INT8_C(  40), INT8_C( -52),
                           INT8_C(  25), INT8_C(  -6), INT8_C(-108), INT8_C(  33),
                           INT8_C(  23), INT8_C( -20), INT8_C(  31), INT8_C(  81),
                           INT8_C( 124), INT8_C(  15), INT8_C(  82), INT8_C(   1)) },
    { easysimd_mm256_set_epi8(INT8_C(  57), INT8_C( -52), INT8_C( 127), INT8_C( -70),
                           INT8_C(  97), INT8_C(  95), INT8_C( -96), INT8_C( -99),
                           INT8_C(  22), INT8_C(-112), INT8_C(  66), INT8_C( -76),
                           INT8_C(  79), INT8_C(-100), INT8_C( -47), INT8_C(-114),
                           INT8_C( -72), INT8_C(  67), INT8_C(   3), INT8_C(  -9),
                           INT8_C(  88), INT8_C(  -5), INT8_C(-111), INT8_C(-100),
                           INT8_C( -94), INT8_C( -72), INT8_C( -45), INT8_C( -95),
                           INT8_C( 119), INT8_C( -81), INT8_C(  38), INT8_C(-111)),
      UINT32_C(4224621908),
      easysimd_mm256_set_epi8(INT8_C(-112), INT8_C(  63), INT8_C(  75), INT8_C(  90),
                           INT8_C(  -7), INT8_C( 116), INT8_C(-123), INT8_C( -34),
                           INT8_C(  81), INT8_C( 114), INT8_C( -76), INT8_C( -63),
                           INT8_C(  30), INT8_C(  66), INT8_C(  18), INT8_C(-119),
                           INT8_C(  26), INT8_C(  28), INT8_C(  56), INT8_C( 127),
                           INT8_C( -81), INT8_C(  -7), INT8_C( -20), INT8_C( -35),
                           INT8_C(  -7), INT8_C(  37), INT8_C( -47), INT8_C(  78),
                           INT8_C( 114), INT8_C( -18), INT8_C(  72), INT8_C(  -8)),
      easysimd_mm256_set_epi8(INT8_C(-112), INT8_C(  63), INT8_C(  75), INT8_C(  90),
                           INT8_C(  -7), INT8_C(  95), INT8_C(-123), INT8_C( -34),
                           INT8_C(  81), INT8_C( 114), INT8_C(  66), INT8_C( -76),
                           INT8_C(  30), INT8_C(  66), INT8_C(  18), INT8_C(-114),
                           INT8_C(  26), INT8_C(  67), INT8_C(   3), INT8_C( 127),
                           INT8_C( -81), INT8_C(  -7), INT8_C(-111), INT8_C( -35),
                           INT8_C( -94), INT8_C(  37), INT8_C( -45), INT8_C(  78),
                           INT8_C( 119), INT8_C( -18), INT8_C(  38), INT8_C(-111)) },
    { easysimd_mm256_set_epi8(INT8_C( -29), INT8_C(-121), INT8_C( -23), INT8_C(  64),
                           INT8_C(  12), INT8_C(   5), INT8_C(  73), INT8_C(  52),
                           INT8_C( -53), INT8_C(  62), INT8_C(   8), INT8_C(-112),
                           INT8_C(  -8), INT8_C(  99), INT8_C( -12), INT8_C(-118),
                           INT8_C( -33), INT8_C( -37), INT8_C( -98), INT8_C( -94),
                           INT8_C(-119), INT8_C(  79), INT8_C( -25), INT8_C(  47),
                           INT8_C(  80), INT8_C(  89), INT8_C(   5), INT8_C(   9),
                           INT8_C( -36), INT8_C(  79), INT8_C(   8), INT8_C(  89)),
      UINT32_C(1663316267),
      easysimd_mm256_set_epi8(INT8_C( 103), INT8_C( -43), INT8_C(   6), INT8_C( 112),
                           INT8_C( -45), INT8_C(  82), INT8_C(  16), INT8_C(   3),
                           INT8_C(  34), INT8_C( -45), INT8_C(  75), INT8_C(-106),
                           INT8_C(-107), INT8_C( -45), INT8_C( -85), INT8_C( -53),
                           INT8_C(  11), INT8_C(  28), INT8_C( 126), INT8_C(  24),
                           INT8_C( -69), INT8_C(  35), INT8_C( -37), INT8_C(  95),
                           INT8_C(  85), INT8_C(   3), INT8_C( -77), INT8_C( -35),
                           INT8_C( -83), INT8_C(  -1), INT8_C( -73), INT8_C( -18)),
      easysimd_mm256_set_epi8(INT8_C( -29), INT8_C( -43), INT8_C(   6), INT8_C(  64),
                           INT8_C(  12), INT8_C(   5), INT8_C(  16), INT8_C(   3),
                           INT8_C( -53), INT8_C(  62), INT8_C(  75), INT8_C(-112),
                           INT8_C(  -8), INT8_C( -45), INT8_C( -12), INT8_C(-118),
                           INT8_C( -33), INT8_C( -37), INT8_C( 126), INT8_C(  24),
                           INT8_C(-119), INT8_C(  79), INT8_C( -25), INT8_C(  95),
                           INT8_C(  80), INT8_C(  89), INT8_C( -77), INT8_C(   9),
                           INT8_C( -83), INT8_C(  79), INT8_C( -73), INT8_C( -18)) },
    { easysimd_mm256_set_epi8(INT8_C( -15), INT8_C(  22), INT8_C( -61), INT8_C( -49),
                           INT8_C(  -4), INT8_C(  -4), INT8_C(  91), INT8_C( -15),
                           INT8_C(  47), INT8_C( -16), INT8_C(-118), INT8_C(  86),
                           INT8_C( -37), INT8_C( -51), INT8_C(  66), INT8_C( -18),
                           INT8_C( -38), INT8_C( -22), INT8_C(   6), INT8_C(  33),
                           INT8_C( 109), INT8_C(-110), INT8_C( -53), INT8_C(-118),
                           INT8_C(  48), INT8_C( -55), INT8_C(  70), INT8_C(  -1),
                           INT8_C(-125), INT8_C( -38), INT8_C( 109), INT8_C( -62)),
      UINT32_C(1252303865),
      easysimd_mm256_set_epi8(INT8_C(-103), INT8_C(-118), INT8_C(-127), INT8_C( -69),
                           INT8_C(  28), INT8_C(  82), INT8_C( -48), INT8_C(-119),
                           INT8_C( -31), INT8_C( -65), INT8_C(-127), INT8_C( -41),
                           INT8_C(  86), INT8_C( -70), INT8_C(  -6), INT8_C(  33),
                           INT8_C( -51), INT8_C(-122), INT8_C( -14), INT8_C( 119),
                           INT8_C(  75), INT8_C(  63), INT8_C( -36), INT8_C(  31),
                           INT8_C( -76), INT8_C(  48), INT8_C(  50), INT8_C(-113),
                           INT8_C(  15), INT8_C( -75), INT8_C( -26), INT8_C(  94)),
      easysimd_mm256_set_epi8(INT8_C( -15), INT8_C(-118), INT8_C( -61), INT8_C( -49),
                           INT8_C(  28), INT8_C(  -4), INT8_C( -48), INT8_C( -15),
                           INT8_C( -31), INT8_C( -16), INT8_C(-127), INT8_C(  86),
                           INT8_C( -37), INT8_C( -70), INT8_C(  66), INT8_C( -18),
                           INT8_C( -51), INT8_C( -22), INT8_C( -14), INT8_C(  33),
                           INT8_C( 109), INT8_C(-110), INT8_C( -36), INT8_C(  31),
                           INT8_C( -76), INT8_C(  48), INT8_C(  50), INT8_C(-113),
                           INT8_C(  15), INT8_C( -38), INT8_C( 109), INT8_C(  94)) },
    { easysimd_mm256_set_epi8(INT8_C(-106), INT8_C(  63), INT8_C( -91), INT8_C( -65),
                           INT8_C(-114), INT8_C( -79), INT8_C( 118), INT8_C(  65),
                           INT8_C(-123), INT8_C(  42), INT8_C( -51), INT8_C( 112),
                           INT8_C( -55), INT8_C( 120), INT8_C(  62), INT8_C( -91),
                           INT8_C( -74), INT8_C(  98), INT8_C( -26), INT8_C( -13),
                           INT8_C( -94), INT8_C( 105), INT8_C( -49), INT8_C( -31),
                           INT8_C(  18), INT8_C(  49), INT8_C( -11), INT8_C(  72),
                           INT8_C(  -9), INT8_C( -16), INT8_C( 100), INT8_C( -64)),
      UINT32_C( 648334209),
      easysimd_mm256_set_epi8(INT8_C(   1), INT8_C( -60), INT8_C( -73), INT8_C( -13),
                           INT8_C(  63), INT8_C(-117), INT8_C(-106), INT8_C(  -9),
                           INT8_C( -71), INT8_C(-116), INT8_C( -20), INT8_C(  61),
                           INT8_C(  48), INT8_C(-114), INT8_C(-114), INT8_C( -45),
                           INT8_C( -77), INT8_C( 123), INT8_C(-120), INT8_C(-126),
                           INT8_C( 112), INT8_C( -73), INT8_C( -89), INT8_C(   6),
                           INT8_C(-118), INT8_C(   2), INT8_C( 106), INT8_C( -46),
                           INT8_C( -87), INT8_C(  71), INT8_C( -71), INT8_C(  -5)),
      easysimd_mm256_set_epi8(INT8_C(-106), INT8_C(  63), INT8_C( -73), INT8_C( -65),
                           INT8_C(-114), INT8_C(-117), INT8_C(-106), INT8_C(  65),
                           INT8_C( -71), INT8_C(  42), INT8_C( -20), INT8_C( 112),
                           INT8_C( -55), INT8_C(-114), INT8_C(  62), INT8_C( -91),
                           INT8_C( -77), INT8_C( 123), INT8_C( -26), INT8_C( -13),
                           INT8_C( 112), INT8_C( 105), INT8_C( -89), INT8_C(   6),
                           INT8_C(-118), INT8_C(  49), INT8_C( -11), INT8_C(  72),
                           INT8_C(  -9), INT8_C( -16), INT8_C( 100), INT8_C(  -5)) },
    { easysimd_mm256_set_epi8(INT8_C( -48), INT8_C(-113), INT8_C(  21), INT8_C(  68),
                           INT8_C( 115), INT8_C(  93), INT8_C(  99), INT8_C( -68),
                           INT8_C(  -9), INT8_C(  34), INT8_C(  15), INT8_C( 118),
                           INT8_C(  54), INT8_C( -58), INT8_C(  11), INT8_C(  91),
                           INT8_C( 122), INT8_C(  59), INT8_C( 108), INT8_C( -59),
                           INT8_C( -39), INT8_C(  74), INT8_C( -25), INT8_C(   1),
                           INT8_C( -26), INT8_C( -59), INT8_C(  91), INT8_C( -81),
                           INT8_C(  -8), INT8_C(  -5), INT8_C( -55), INT8_C( -59)),
      UINT32_C(2027822108),
      easysimd_mm256_set_epi8(INT8_C( -93), INT8_C( -25), INT8_C(  14), INT8_C( -22),
                           INT8_C(  85), INT8_C( -47), INT8_C( -59), INT8_C( -81),
                           INT8_C(  94), INT8_C( -67), INT8_C( -69), INT8_C( -79),
                           INT8_C(  61), INT8_C(  49), INT8_C( -27), INT8_C( 124),
                           INT8_C(  89), INT8_C(  80), INT8_C(  55), INT8_C( -47),
                           INT8_C(  45), INT8_C(-120), INT8_C(  28), INT8_C( -89),
                           INT8_C( -69), INT8_C(-127), INT8_C(  65), INT8_C(  53),
                           INT8_C( -35), INT8_C( -30), INT8_C( -74), INT8_C( -10)),
      easysimd_mm256_set_epi8(INT8_C( -48), INT8_C( -25), INT8_C(  14), INT8_C( -22),
                           INT8_C(  85), INT8_C(  93), INT8_C(  99), INT8_C( -68),
                           INT8_C(  94), INT8_C( -67), INT8_C(  15), INT8_C( -79),
                           INT8_C(  61), INT8_C(  49), INT8_C( -27), INT8_C(  91),
                           INT8_C( 122), INT8_C(  59), INT8_C( 108), INT8_C( -47),
                           INT8_C(  45), INT8_C(-120), INT8_C( -25), INT8_C(   1),
                           INT8_C( -26), INT8_C( -59), INT8_C(  91), INT8_C(  53),
                           INT8_C( -35), INT8_C( -30), INT8_C( -55), INT8_C( -59)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = test_vec[i].src;
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mov_epi8(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mov_epi8");
    easysimd_assert_m256i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_mov_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i src;
    easysimd__mmask16 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi16(INT16_C(-23030), INT16_C(  6803), INT16_C(-21055), INT16_C(  -910),
                            INT16_C( -6009), INT16_C( 10471), INT16_C(-29834), INT16_C(-14111),
                            INT16_C( -2981), INT16_C( 28733), INT16_C( 11699), INT16_C(  7781),
                            INT16_C( 29036), INT16_C( -8103), INT16_C(-21310), INT16_C(  9176)),
      UINT16_C(32768),
      easysimd_mm256_set_epi16(INT16_C( 30563), INT16_C( -5523), INT16_C(-18306), INT16_C( 14754),
                            INT16_C(-23068), INT16_C(-17313), INT16_C( 21598), INT16_C( 12635),
                            INT16_C( 17053), INT16_C(  3377), INT16_C( 28887), INT16_C( 29062),
                            INT16_C( 26146), INT16_C( -4849), INT16_C( 17375), INT16_C(-24515)),
      easysimd_mm256_set_epi16(INT16_C( 30563), INT16_C(  6803), INT16_C(-21055), INT16_C(  -910),
                            INT16_C( -6009), INT16_C( 10471), INT16_C(-29834), INT16_C(-14111),
                            INT16_C( -2981), INT16_C( 28733), INT16_C( 11699), INT16_C(  7781),
                            INT16_C( 29036), INT16_C( -8103), INT16_C(-21310), INT16_C(  9176)) },
    { easysimd_mm256_set_epi16(INT16_C(  9971), INT16_C( -9002), INT16_C(-22233), INT16_C(-13917),
                            INT16_C(-13732), INT16_C(  -199), INT16_C(  9707), INT16_C( 31342),
                            INT16_C(-13386), INT16_C(-15675), INT16_C( 10143), INT16_C( 19953),
                            INT16_C(-25473), INT16_C( 27175), INT16_C(-12968), INT16_C(-11899)),
      UINT16_C(15492),
      easysimd_mm256_set_epi16(INT16_C(-30515), INT16_C(-13927), INT16_C( 24112), INT16_C(  9227),
                            INT16_C(-20054), INT16_C(-11664), INT16_C( -7103), INT16_C(-13246),
                            INT16_C(  4285), INT16_C(-23471), INT16_C( 24470), INT16_C(-13226),
                            INT16_C(  4085), INT16_C( 10000), INT16_C(-17688), INT16_C( 28540)),
      easysimd_mm256_set_epi16(INT16_C(  9971), INT16_C( -9002), INT16_C( 24112), INT16_C(  9227),
                            INT16_C(-20054), INT16_C(-11664), INT16_C(  9707), INT16_C( 31342),
                            INT16_C(  4285), INT16_C(-15675), INT16_C( 10143), INT16_C( 19953),
                            INT16_C(-25473), INT16_C( 10000), INT16_C(-12968), INT16_C(-11899)) },
    { easysimd_mm256_set_epi16(INT16_C(-17362), INT16_C( -1830), INT16_C(-16587), INT16_C(-17056),
                            INT16_C(-14539), INT16_C(  7972), INT16_C(-26491), INT16_C( 20406),
                            INT16_C( 26939), INT16_C( 20968), INT16_C(-31196), INT16_C( 11313),
                            INT16_C(-25947), INT16_C( 19467), INT16_C( 22325), INT16_C( 14960)),
      UINT16_C(53867),
      easysimd_mm256_set_epi16(INT16_C( 15597), INT16_C(-30582), INT16_C(-21551), INT16_C(-25534),
                            INT16_C( 13374), INT16_C( 17137), INT16_C(-27681), INT16_C(-10912),
                            INT16_C(-10124), INT16_C(  1110), INT16_C(  1704), INT16_C(-17853),
                            INT16_C( -7561), INT16_C(-19432), INT16_C( 22127), INT16_C(-30033)),
      easysimd_mm256_set_epi16(INT16_C( 15597), INT16_C(-30582), INT16_C(-16587), INT16_C(-25534),
                            INT16_C(-14539), INT16_C(  7972), INT16_C(-27681), INT16_C( 20406),
                            INT16_C( 26939), INT16_C(  1110), INT16_C(  1704), INT16_C( 11313),
                            INT16_C( -7561), INT16_C( 19467), INT16_C( 22127), INT16_C(-30033)) },
    { easysimd_mm256_set_epi16(INT16_C( 14671), INT16_C( 16470), INT16_C( 30174), INT16_C( -7130),
                            INT16_C( 31852), INT16_C( 11282), INT16_C( 29705), INT16_C(-21158),
                            INT16_C( 16917), INT16_C( 10042), INT16_C(  5958), INT16_C( -4695),
                            INT16_C(-20590), INT16_C( 17528), INT16_C( -6738), INT16_C(-26754)),
      UINT16_C(25018),
      easysimd_mm256_set_epi16(INT16_C(-21192), INT16_C(  6104), INT16_C(-12947), INT16_C( 12440),
                            INT16_C( 12048), INT16_C( -8528), INT16_C(-31627), INT16_C( 26711),
                            INT16_C( -4678), INT16_C( 32013), INT16_C(   814), INT16_C( 19873),
                            INT16_C( 32199), INT16_C( -7421), INT16_C( 21197), INT16_C( 25563)),
      easysimd_mm256_set_epi16(INT16_C( 14671), INT16_C(  6104), INT16_C(-12947), INT16_C( -7130),
                            INT16_C( 31852), INT16_C( 11282), INT16_C( 29705), INT16_C( 26711),
                            INT16_C( -4678), INT16_C( 10042), INT16_C(   814), INT16_C( 19873),
                            INT16_C( 32199), INT16_C( 17528), INT16_C( 21197), INT16_C(-26754)) },
    { easysimd_mm256_set_epi16(INT16_C( 30594), INT16_C(-11819), INT16_C( 16854), INT16_C(  8281),
                            INT16_C( 32229), INT16_C( -2511), INT16_C(-10942), INT16_C(-28733),
                            INT16_C( -8714), INT16_C( -6616), INT16_C(  4922), INT16_C(  1537),
                            INT16_C( -8589), INT16_C(  6229), INT16_C(-12142), INT16_C( 12862)),
      UINT16_C(62562),
      easysimd_mm256_set_epi16(INT16_C( 28902), INT16_C( 31472), INT16_C( -9808), INT16_C(-22935),
                            INT16_C(  4498), INT16_C(-13447), INT16_C(-31030), INT16_C(-31086),
                            INT16_C(  6386), INT16_C(-11676), INT16_C(  9598), INT16_C(-30958),
                            INT16_C(-24145), INT16_C(-18452), INT16_C( -8547), INT16_C(-20619)),
      easysimd_mm256_set_epi16(INT16_C( 28902), INT16_C( 31472), INT16_C( -9808), INT16_C(-22935),
                            INT16_C( 32229), INT16_C(-13447), INT16_C(-10942), INT16_C(-28733),
                            INT16_C( -8714), INT16_C(-11676), INT16_C(  9598), INT16_C(  1537),
                            INT16_C( -8589), INT16_C(  6229), INT16_C( -8547), INT16_C( 12862)) },
    { easysimd_mm256_set_epi16(INT16_C( -1185), INT16_C( 28882), INT16_C(-25549), INT16_C(-18169),
                            INT16_C( -7221), INT16_C(  4400), INT16_C(-25724), INT16_C(-28761),
                            INT16_C(-20506), INT16_C(-24341), INT16_C(  5349), INT16_C( -9608),
                            INT16_C(-30698), INT16_C(  7741), INT16_C(  6648), INT16_C(  2085)),
      UINT16_C(40999),
      easysimd_mm256_set_epi16(INT16_C( 17256), INT16_C(-15790), INT16_C( 23704), INT16_C(-17336),
                            INT16_C( -4418), INT16_C( 28004), INT16_C(-27022), INT16_C( 29950),
                            INT16_C(-28093), INT16_C(   901), INT16_C(-13716), INT16_C(-16668),
                            INT16_C(-12954), INT16_C(  4373), INT16_C( 25556), INT16_C(-31530)),
      easysimd_mm256_set_epi16(INT16_C( 17256), INT16_C( 28882), INT16_C( 23704), INT16_C(-18169),
                            INT16_C( -7221), INT16_C(  4400), INT16_C(-25724), INT16_C(-28761),
                            INT16_C(-20506), INT16_C(-24341), INT16_C(-13716), INT16_C( -9608),
                            INT16_C(-30698), INT16_C(  4373), INT16_C( 25556), INT16_C(-31530)) },
    { easysimd_mm256_set_epi16(INT16_C( -2894), INT16_C(-32472), INT16_C( 11220), INT16_C(  6669),
                            INT16_C( 23064), INT16_C(-27024), INT16_C(-15827), INT16_C(-11722),
                            INT16_C(-26431), INT16_C(  6527), INT16_C(-14361), INT16_C(-27595),
                            INT16_C(-18051), INT16_C( -3890), INT16_C(-26121), INT16_C(-29481)),
      UINT16_C( 8894),
      easysimd_mm256_set_epi16(INT16_C( 18291), INT16_C( 26196), INT16_C(-27505), INT16_C( -8229),
                            INT16_C(-25273), INT16_C( -2374), INT16_C( 25602), INT16_C( 26391),
                            INT16_C( 16833), INT16_C(-18212), INT16_C(  6765), INT16_C( 22695),
                            INT16_C( 31217), INT16_C( 10116), INT16_C( 12733), INT16_C( 11434)),
      easysimd_mm256_set_epi16(INT16_C( -2894), INT16_C(-32472), INT16_C(-27505), INT16_C(  6669),
                            INT16_C( 23064), INT16_C(-27024), INT16_C( 25602), INT16_C(-11722),
                            INT16_C( 16833), INT16_C(  6527), INT16_C(  6765), INT16_C( 22695),
                            INT16_C( 31217), INT16_C( 10116), INT16_C( 12733), INT16_C(-29481)) },
    { easysimd_mm256_set_epi16(INT16_C( 31730), INT16_C(-24704), INT16_C( -9707), INT16_C(-27923),
                            INT16_C( 12026), INT16_C( -8313), INT16_C(-30875), INT16_C( -3866),
                            INT16_C( 13477), INT16_C( -8690), INT16_C(  7980), INT16_C( 29046),
                            INT16_C(-16244), INT16_C(-14526), INT16_C( -1470), INT16_C(  9637)),
      UINT16_C(47578),
      easysimd_mm256_set_epi16(INT16_C(-27085), INT16_C(-21439), INT16_C( -6499), INT16_C(-12213),
                            INT16_C( 32648), INT16_C(-16468), INT16_C(-15892), INT16_C( 21695),
                            INT16_C(-24474), INT16_C(  -770), INT16_C(-22665), INT16_C(-20908),
                            INT16_C(  -267), INT16_C( -8958), INT16_C( -8601), INT16_C( 15369)),
      easysimd_mm256_set_epi16(INT16_C(-27085), INT16_C(-24704), INT16_C( -6499), INT16_C(-12213),
                            INT16_C( 32648), INT16_C( -8313), INT16_C(-30875), INT16_C( 21695),
                            INT16_C(-24474), INT16_C(  -770), INT16_C(  7980), INT16_C(-20908),
                            INT16_C(  -267), INT16_C(-14526), INT16_C( -8601), INT16_C(  9637)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mov_epi16(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mov_epi16");
    easysimd_assert_m256i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_mov_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i src;
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
      { easysimd_mm256_set_epi32(INT32_C(-2051902106), INT32_C(-1489562810), INT32_C( -627115156), INT32_C(  913274595),
                            INT32_C(-1198634499), INT32_C(  139959001), INT32_C(-1600412710), INT32_C(  934654383)),
      UINT8_C(164),
      easysimd_mm256_set_epi32(INT32_C(-2030579644), INT32_C(  386335945), INT32_C( -809785463), INT32_C( 2050465670),
                            INT32_C(  458083110), INT32_C(  911315275), INT32_C(  438060664), INT32_C( 1293777364)),
      easysimd_mm256_set_epi32(INT32_C(-2030579644), INT32_C(-1489562810), INT32_C( -809785463), INT32_C(  913274595),
                            INT32_C(-1198634499), INT32_C(  911315275), INT32_C(-1600412710), INT32_C(  934654383)) },
    { easysimd_mm256_set_epi32(INT32_C( 1771679392), INT32_C(  747779042), INT32_C( 1568509396), INT32_C( -126295667),
                            INT32_C( 1160475018), INT32_C(  343988166), INT32_C( 1516295700), INT32_C(-1359069473)),
      UINT8_C(178),
      easysimd_mm256_set_epi32(INT32_C( 1326620113), INT32_C(-1696986714), INT32_C( -201743610), INT32_C( 1745319425),
                            INT32_C(-1761511775), INT32_C( 1270104738), INT32_C( 1013012890), INT32_C(  875163254)),
      easysimd_mm256_set_epi32(INT32_C( 1326620113), INT32_C(  747779042), INT32_C( -201743610), INT32_C( 1745319425),
                            INT32_C( 1160475018), INT32_C(  343988166), INT32_C( 1013012890), INT32_C(-1359069473)) },
    { easysimd_mm256_set_epi32(INT32_C(  518286759), INT32_C(-1532979566), INT32_C(-1858515332), INT32_C(  132974279),
                            INT32_C(  761595911), INT32_C(-1701198420), INT32_C( 1222823321), INT32_C( -238072978)),
      UINT8_C(112),
      easysimd_mm256_set_epi32(INT32_C( -801582728), INT32_C( 1471437069), INT32_C( 1970067030), INT32_C( 1007722212),
                            INT32_C( -224938211), INT32_C( -282706876), INT32_C( 1478523622), INT32_C(  630801793)),
      easysimd_mm256_set_epi32(INT32_C(  518286759), INT32_C( 1471437069), INT32_C( 1970067030), INT32_C( 1007722212),
                            INT32_C(  761595911), INT32_C(-1701198420), INT32_C( 1222823321), INT32_C( -238072978)) },
    { easysimd_mm256_set_epi32(INT32_C(-1331251138), INT32_C(-1232220609), INT32_C(  -83499690), INT32_C(-1933771795),
                            INT32_C( 1431588209), INT32_C(    9145992), INT32_C( 1554181542), INT32_C(-1595697445)),
      UINT8_C(209),
      easysimd_mm256_set_epi32(INT32_C(-1567962509), INT32_C(-1474212928), INT32_C(-1912431565), INT32_C( -269915367),
                            INT32_C( -487478944), INT32_C(-1785315433), INT32_C(-1130207739), INT32_C( -388075219)),
      easysimd_mm256_set_epi32(INT32_C(-1567962509), INT32_C(-1474212928), INT32_C(  -83499690), INT32_C( -269915367),
                            INT32_C( 1431588209), INT32_C(    9145992), INT32_C( 1554181542), INT32_C( -388075219)) },
    { easysimd_mm256_set_epi32(INT32_C( 1834864917), INT32_C( -675288826), INT32_C( 1896194121), INT32_C( 1512557303),
                            INT32_C( -545693873), INT32_C(  513757285), INT32_C( 1710853511), INT32_C(  367108805)),
      UINT8_C(141),
      easysimd_mm256_set_epi32(INT32_C(-1942300637), INT32_C( 1717002604), INT32_C( -236253831), INT32_C(  993211905),
                            INT32_C(  884769165), INT32_C( 1081163766), INT32_C( 1932456000), INT32_C( -153656708)),
      easysimd_mm256_set_epi32(INT32_C(-1942300637), INT32_C( -675288826), INT32_C( 1896194121), INT32_C( 1512557303),
                            INT32_C(  884769165), INT32_C( 1081163766), INT32_C( 1710853511), INT32_C( -153656708)) },
    { easysimd_mm256_set_epi32(INT32_C( 1057245798), INT32_C(-1988238659), INT32_C(  464652738), INT32_C(-1394070870),
                            INT32_C(  410687111), INT32_C(-1023380740), INT32_C(-1345956426), INT32_C( 1062641002)),
      UINT8_C( 23),
      easysimd_mm256_set_epi32(INT32_C(  804151705), INT32_C(-1030405330), INT32_C(-1199759874), INT32_C( 1385588241),
                            INT32_C(-1001762620), INT32_C( 1644327590), INT32_C( -999008446), INT32_C( 2086723218)),
      easysimd_mm256_set_epi32(INT32_C( 1057245798), INT32_C(-1988238659), INT32_C(  464652738), INT32_C( 1385588241),
                            INT32_C(  410687111), INT32_C( 1644327590), INT32_C( -999008446), INT32_C( 2086723218)) },
    { easysimd_mm256_set_epi32(INT32_C( 1481764690), INT32_C(  749562747), INT32_C( 1739109341), INT32_C( 1504825630),
                            INT32_C(-1715949382), INT32_C( -901153926), INT32_C( -433640108), INT32_C( -201965406)),
      UINT8_C( 20),
      easysimd_mm256_set_epi32(INT32_C(  657000670), INT32_C(   71096321), INT32_C(  324839890), INT32_C( 1620447032),
                            INT32_C( 1126601222), INT32_C(-1962686585), INT32_C(  174027827), INT32_C( 1092631470)),
      easysimd_mm256_set_epi32(INT32_C( 1481764690), INT32_C(  749562747), INT32_C( 1739109341), INT32_C( 1620447032),
                            INT32_C(-1715949382), INT32_C(-1962686585), INT32_C( -433640108), INT32_C( -201965406)) },
    { easysimd_mm256_set_epi32(INT32_C( 1112858374), INT32_C( 1689862137), INT32_C(-1548199384), INT32_C(  560346027),
                            INT32_C(-1831151558), INT32_C( 1961484348), INT32_C( 1845841537), INT32_C(-1490051864)),
      UINT8_C( 81),
      easysimd_mm256_set_epi32(INT32_C( 1782794803), INT32_C(-1212843470), INT32_C(  702145811), INT32_C(  712189474),
                            INT32_C( 1538408527), INT32_C( 1714734347), INT32_C(  509188796), INT32_C( 1218928521)),
      easysimd_mm256_set_epi32(INT32_C( 1112858374), INT32_C(-1212843470), INT32_C(-1548199384), INT32_C(  712189474),
                            INT32_C(-1831151558), INT32_C( 1961484348), INT32_C( 1845841537), INT32_C( 1218928521)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mov_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mov_epi32");
    easysimd_assert_m256i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_mov_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i src;
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi64x(INT64_C( -211287979567135941), INT64_C(-9075367401252635211),
                             INT64_C(  960243121462097108), INT64_C( 2005878706758239899)),
      UINT8_C( 32),
     easysimd_mm256_set_epi64x(INT64_C(-6608524325915548957), INT64_C(-1314544444369440805),
                             INT64_C( 1510534294895689397), INT64_C( 1845655456785432498)),
      easysimd_mm256_set_epi64x(INT64_C( -211287979567135941), INT64_C(-9075367401252635211),
                             INT64_C(  960243121462097108), INT64_C( 2005878706758239899)) },
    { easysimd_mm256_set_epi64x(INT64_C(-1723084715644559301), INT64_C(-2080563230649448126),
                             INT64_C( 5959215642275768669), INT64_C( 2475827768754845699)),
      UINT8_C(214),
     easysimd_mm256_set_epi64x(INT64_C( 5190090124690883989), INT64_C(-4089440710650942034),
                             INT64_C(-9158432549634317510), INT64_C( 8190130421956302558)),
      easysimd_mm256_set_epi64x(INT64_C(-1723084715644559301), INT64_C(-4089440710650942034),
                             INT64_C(-9158432549634317510), INT64_C( 2475827768754845699)) },
    { easysimd_mm256_set_epi64x(INT64_C(-3369675545100032670), INT64_C(-5453194687323465101),
                             INT64_C(-7873359915838041141), INT64_C(-2715603020778233064)),
      UINT8_C(169),
     easysimd_mm256_set_epi64x(INT64_C( 6972842025751468465), INT64_C(  -70349858703264913),
                             INT64_C( -274794754558770720), INT64_C(-4632650321932570335)),
      easysimd_mm256_set_epi64x(INT64_C( 6972842025751468465), INT64_C(-5453194687323465101),
                             INT64_C(-7873359915838041141), INT64_C(-4632650321932570335)) },
    { easysimd_mm256_set_epi64x(INT64_C( 6027108319237370493), INT64_C(-1242194223738253269),
                             INT64_C(-4627845169201021686), INT64_C( 6845596120956145572)),
      UINT8_C(143),
     easysimd_mm256_set_epi64x(INT64_C(-3877996964438243656), INT64_C(-4576357011277680458),
                             INT64_C( 6353148636895875717), INT64_C( 4412973294027016788)),
      easysimd_mm256_set_epi64x(INT64_C(-3877996964438243656), INT64_C(-4576357011277680458),
                             INT64_C( 6353148636895875717), INT64_C( 4412973294027016788)) },
    { easysimd_mm256_set_epi64x(INT64_C( 9142894596557299884), INT64_C( 8214900458994780454),
                             INT64_C( 8865669120860669544), INT64_C( 8653034493845742246)),
      UINT8_C(226),
     easysimd_mm256_set_epi64x(INT64_C( 1244643455152445841), INT64_C( 2297609102993095657),
                             INT64_C(-5233775572318758587), INT64_C(-7732116011616278804)),
      easysimd_mm256_set_epi64x(INT64_C( 9142894596557299884), INT64_C( 8214900458994780454),
                             INT64_C(-5233775572318758587), INT64_C( 8653034493845742246)) },
    { easysimd_mm256_set_epi64x(INT64_C( 4960786529412164795), INT64_C( 8678743560946050948),
                             INT64_C( 2843182024025655803), INT64_C(  -83887347445242653)),
      UINT8_C( 74),
     easysimd_mm256_set_epi64x(INT64_C( 3754067458265850846), INT64_C(-6092043402181917138),
                             INT64_C( 1306971064806148347), INT64_C(-5729735109094765451)),
      easysimd_mm256_set_epi64x(INT64_C( 3754067458265850846), INT64_C( 8678743560946050948),
                             INT64_C( 1306971064806148347), INT64_C(  -83887347445242653)) },
    { easysimd_mm256_set_epi64x(INT64_C( 2112902535792085455), INT64_C(-6619508989181003755),
                             INT64_C(-7221956771732279605), INT64_C( 6287623589682049686)),
      UINT8_C(191),
     easysimd_mm256_set_epi64x(INT64_C( 3797901248692596665), INT64_C( 7828643831964461331),
                             INT64_C( 1067056404383166060), INT64_C(-2361551563160303879)),
      easysimd_mm256_set_epi64x(INT64_C( 3797901248692596665), INT64_C( 7828643831964461331),
                             INT64_C( 1067056404383166060), INT64_C(-2361551563160303879)) },
    { easysimd_mm256_set_epi64x(INT64_C( 6637695700610981441), INT64_C( 8064523188707259542),
                             INT64_C(-3039387732265680328), INT64_C( 5125314073625570095)),
      UINT8_C(100),
     easysimd_mm256_set_epi64x(INT64_C( 4453523714429879071), INT64_C(-2274204535440821687),
                             INT64_C(-3167205970195665497), INT64_C( 3325113155733044170)),
      easysimd_mm256_set_epi64x(INT64_C( 6637695700610981441), INT64_C(-2274204535440821687),
                             INT64_C(-3039387732265680328), INT64_C( 5125314073625570095)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mov_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mov_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_mov_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d src;
    easysimd__mmask8 k;
    easysimd__m256d a;
    easysimd__m256d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  774.53), EASYSIMD_FLOAT64_C(  377.61),
                         EASYSIMD_FLOAT64_C(  717.45), EASYSIMD_FLOAT64_C(  713.04)),
      UINT8_C( 22),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  723.04), EASYSIMD_FLOAT64_C(  343.93),
                         EASYSIMD_FLOAT64_C(  199.28), EASYSIMD_FLOAT64_C( -711.48)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  774.53), EASYSIMD_FLOAT64_C(  343.93),
                         EASYSIMD_FLOAT64_C(  199.28), EASYSIMD_FLOAT64_C(  713.04)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -510.02), EASYSIMD_FLOAT64_C(  340.82),
                         EASYSIMD_FLOAT64_C(  576.36), EASYSIMD_FLOAT64_C(  -95.74)),
      UINT8_C(255),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  918.58), EASYSIMD_FLOAT64_C(  109.09),
                         EASYSIMD_FLOAT64_C( -879.13), EASYSIMD_FLOAT64_C(  336.44)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  918.58), EASYSIMD_FLOAT64_C(  109.09),
                         EASYSIMD_FLOAT64_C( -879.13), EASYSIMD_FLOAT64_C(  336.44)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  409.63), EASYSIMD_FLOAT64_C( -297.52),
                         EASYSIMD_FLOAT64_C(  108.73), EASYSIMD_FLOAT64_C(  228.30)),
      UINT8_C(234),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -549.30), EASYSIMD_FLOAT64_C( -400.24),
                         EASYSIMD_FLOAT64_C( -459.77), EASYSIMD_FLOAT64_C( -392.32)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -549.30), EASYSIMD_FLOAT64_C( -297.52),
                         EASYSIMD_FLOAT64_C( -459.77), EASYSIMD_FLOAT64_C(  228.30)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -1.67), EASYSIMD_FLOAT64_C( -827.28),
                         EASYSIMD_FLOAT64_C(  295.95), EASYSIMD_FLOAT64_C(  558.58)),
      UINT8_C(192),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  320.94), EASYSIMD_FLOAT64_C( -669.22),
                         EASYSIMD_FLOAT64_C(  941.71), EASYSIMD_FLOAT64_C( -772.39)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -1.67), EASYSIMD_FLOAT64_C( -827.28),
                         EASYSIMD_FLOAT64_C(  295.95), EASYSIMD_FLOAT64_C(  558.58)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -393.08), EASYSIMD_FLOAT64_C(   83.20),
                         EASYSIMD_FLOAT64_C(  408.44), EASYSIMD_FLOAT64_C(  326.57)),
      UINT8_C( 97),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -490.95), EASYSIMD_FLOAT64_C(  526.06),
                         EASYSIMD_FLOAT64_C( -564.61), EASYSIMD_FLOAT64_C( -582.24)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -393.08), EASYSIMD_FLOAT64_C(   83.20),
                         EASYSIMD_FLOAT64_C(  408.44), EASYSIMD_FLOAT64_C( -582.24)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -808.43), EASYSIMD_FLOAT64_C(   58.34),
                         EASYSIMD_FLOAT64_C( -379.04), EASYSIMD_FLOAT64_C(   54.10)),
      UINT8_C( 14),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  450.27), EASYSIMD_FLOAT64_C( -128.64),
                         EASYSIMD_FLOAT64_C( -995.13), EASYSIMD_FLOAT64_C(  479.76)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  450.27), EASYSIMD_FLOAT64_C( -128.64),
                         EASYSIMD_FLOAT64_C( -995.13), EASYSIMD_FLOAT64_C(   54.10)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  322.73), EASYSIMD_FLOAT64_C(  175.90),
                         EASYSIMD_FLOAT64_C( -940.90), EASYSIMD_FLOAT64_C( -692.98)),
      UINT8_C(117),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -758.62), EASYSIMD_FLOAT64_C(   71.29),
                         EASYSIMD_FLOAT64_C(  788.39), EASYSIMD_FLOAT64_C( -310.18)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  322.73), EASYSIMD_FLOAT64_C(   71.29),
                         EASYSIMD_FLOAT64_C( -940.90), EASYSIMD_FLOAT64_C( -310.18)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -378.45), EASYSIMD_FLOAT64_C(  288.81),
                         EASYSIMD_FLOAT64_C(  695.49), EASYSIMD_FLOAT64_C( -580.49)),
      UINT8_C( 27),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  130.47), EASYSIMD_FLOAT64_C(  632.45),
                         EASYSIMD_FLOAT64_C(  808.39), EASYSIMD_FLOAT64_C(  627.49)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  130.47), EASYSIMD_FLOAT64_C(  288.81),
                         EASYSIMD_FLOAT64_C(  808.39), EASYSIMD_FLOAT64_C(  627.49)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = test_vec[i].a;
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mov_pd(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mov_pd");
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_mov_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256 src;
    easysimd__mmask8 k;
    easysimd__m256 a;
    easysimd__m256 r;
  } test_vec[8] = {
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -555.53), EASYSIMD_FLOAT32_C(   800.80),
                         EASYSIMD_FLOAT32_C(   174.96), EASYSIMD_FLOAT32_C(    12.40),
                         EASYSIMD_FLOAT32_C(  -124.14), EASYSIMD_FLOAT32_C(   378.54),
                         EASYSIMD_FLOAT32_C(  -864.83), EASYSIMD_FLOAT32_C(   821.24)),
      UINT8_C(222),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   486.82), EASYSIMD_FLOAT32_C(   716.60),
                         EASYSIMD_FLOAT32_C(   497.18), EASYSIMD_FLOAT32_C(  -260.12),
                         EASYSIMD_FLOAT32_C(   283.83), EASYSIMD_FLOAT32_C(   297.46),
                         EASYSIMD_FLOAT32_C(   984.87), EASYSIMD_FLOAT32_C(    59.43)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   486.82), EASYSIMD_FLOAT32_C(   716.60),
                         EASYSIMD_FLOAT32_C(   174.96), EASYSIMD_FLOAT32_C(  -260.12),
                         EASYSIMD_FLOAT32_C(   283.83), EASYSIMD_FLOAT32_C(   297.46),
                         EASYSIMD_FLOAT32_C(   984.87), EASYSIMD_FLOAT32_C(   821.24)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -111.85), EASYSIMD_FLOAT32_C(  -140.93),
                         EASYSIMD_FLOAT32_C(    91.77), EASYSIMD_FLOAT32_C(   175.59),
                         EASYSIMD_FLOAT32_C(  -358.15), EASYSIMD_FLOAT32_C(  -375.20),
                         EASYSIMD_FLOAT32_C(   580.39), EASYSIMD_FLOAT32_C(   459.07)),
      UINT8_C(207),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   315.52), EASYSIMD_FLOAT32_C(  -581.86),
                         EASYSIMD_FLOAT32_C(   639.05), EASYSIMD_FLOAT32_C(   298.85),
                         EASYSIMD_FLOAT32_C(  -373.24), EASYSIMD_FLOAT32_C(  -178.13),
                         EASYSIMD_FLOAT32_C(    98.66), EASYSIMD_FLOAT32_C(  -334.34)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   315.52), EASYSIMD_FLOAT32_C(  -581.86),
                         EASYSIMD_FLOAT32_C(    91.77), EASYSIMD_FLOAT32_C(   175.59),
                         EASYSIMD_FLOAT32_C(  -373.24), EASYSIMD_FLOAT32_C(  -178.13),
                         EASYSIMD_FLOAT32_C(    98.66), EASYSIMD_FLOAT32_C(  -334.34)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   869.24), EASYSIMD_FLOAT32_C(   558.63),
                         EASYSIMD_FLOAT32_C(   500.11), EASYSIMD_FLOAT32_C(   448.62),
                         EASYSIMD_FLOAT32_C(   -66.45), EASYSIMD_FLOAT32_C(  -429.13),
                         EASYSIMD_FLOAT32_C(  -688.99), EASYSIMD_FLOAT32_C(  -828.86)),
      UINT8_C(106),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -209.77), EASYSIMD_FLOAT32_C(    87.73),
                         EASYSIMD_FLOAT32_C(   807.71), EASYSIMD_FLOAT32_C(  -161.53),
                         EASYSIMD_FLOAT32_C(  -720.29), EASYSIMD_FLOAT32_C(  -841.34),
                         EASYSIMD_FLOAT32_C(  -679.61), EASYSIMD_FLOAT32_C(  -751.55)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   869.24), EASYSIMD_FLOAT32_C(    87.73),
                         EASYSIMD_FLOAT32_C(   807.71), EASYSIMD_FLOAT32_C(   448.62),
                         EASYSIMD_FLOAT32_C(  -720.29), EASYSIMD_FLOAT32_C(  -429.13),
                         EASYSIMD_FLOAT32_C(  -679.61), EASYSIMD_FLOAT32_C(  -828.86)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   648.52), EASYSIMD_FLOAT32_C(  -621.11),
                         EASYSIMD_FLOAT32_C(    44.58), EASYSIMD_FLOAT32_C(   173.55),
                         EASYSIMD_FLOAT32_C(   227.71), EASYSIMD_FLOAT32_C(  -831.29),
                         EASYSIMD_FLOAT32_C(   210.07), EASYSIMD_FLOAT32_C(   469.94)),
      UINT8_C(209),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -457.88), EASYSIMD_FLOAT32_C(  -345.53),
                         EASYSIMD_FLOAT32_C(   -52.29), EASYSIMD_FLOAT32_C(   652.21),
                         EASYSIMD_FLOAT32_C(   802.89), EASYSIMD_FLOAT32_C(   706.42),
                         EASYSIMD_FLOAT32_C(    63.40), EASYSIMD_FLOAT32_C(   904.43)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -457.88), EASYSIMD_FLOAT32_C(  -345.53),
                         EASYSIMD_FLOAT32_C(    44.58), EASYSIMD_FLOAT32_C(   652.21),
                         EASYSIMD_FLOAT32_C(   227.71), EASYSIMD_FLOAT32_C(  -831.29),
                         EASYSIMD_FLOAT32_C(   210.07), EASYSIMD_FLOAT32_C(   904.43)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   592.24), EASYSIMD_FLOAT32_C(  -735.22),
                         EASYSIMD_FLOAT32_C(   596.55), EASYSIMD_FLOAT32_C(  -541.18),
                         EASYSIMD_FLOAT32_C(  -342.66), EASYSIMD_FLOAT32_C(    98.60),
                         EASYSIMD_FLOAT32_C(   188.58), EASYSIMD_FLOAT32_C(  -720.97)),
      UINT8_C( 39),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -832.08), EASYSIMD_FLOAT32_C(   690.51),
                         EASYSIMD_FLOAT32_C(   197.88), EASYSIMD_FLOAT32_C(  -345.06),
                         EASYSIMD_FLOAT32_C(  -603.10), EASYSIMD_FLOAT32_C(   528.02),
                         EASYSIMD_FLOAT32_C(  -679.70), EASYSIMD_FLOAT32_C(  -757.75)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   592.24), EASYSIMD_FLOAT32_C(  -735.22),
                         EASYSIMD_FLOAT32_C(   197.88), EASYSIMD_FLOAT32_C(  -541.18),
                         EASYSIMD_FLOAT32_C(  -342.66), EASYSIMD_FLOAT32_C(   528.02),
                         EASYSIMD_FLOAT32_C(  -679.70), EASYSIMD_FLOAT32_C(  -757.75)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   630.75), EASYSIMD_FLOAT32_C(  -765.52),
                         EASYSIMD_FLOAT32_C(   644.64), EASYSIMD_FLOAT32_C(  -522.11),
                         EASYSIMD_FLOAT32_C(  -647.87), EASYSIMD_FLOAT32_C(   408.91),
                         EASYSIMD_FLOAT32_C(  -874.53), EASYSIMD_FLOAT32_C(   777.74)),
      UINT8_C( 55),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -382.40), EASYSIMD_FLOAT32_C(   204.65),
                         EASYSIMD_FLOAT32_C(   263.52), EASYSIMD_FLOAT32_C(   553.68),
                         EASYSIMD_FLOAT32_C(   482.50), EASYSIMD_FLOAT32_C(  -416.62),
                         EASYSIMD_FLOAT32_C(   194.15), EASYSIMD_FLOAT32_C(  -653.83)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   630.75), EASYSIMD_FLOAT32_C(  -765.52),
                         EASYSIMD_FLOAT32_C(   263.52), EASYSIMD_FLOAT32_C(   553.68),
                         EASYSIMD_FLOAT32_C(  -647.87), EASYSIMD_FLOAT32_C(  -416.62),
                         EASYSIMD_FLOAT32_C(   194.15), EASYSIMD_FLOAT32_C(  -653.83)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -84.63), EASYSIMD_FLOAT32_C(   440.56),
                         EASYSIMD_FLOAT32_C(   471.24), EASYSIMD_FLOAT32_C(   544.90),
                         EASYSIMD_FLOAT32_C(  -133.99), EASYSIMD_FLOAT32_C(  -169.40),
                         EASYSIMD_FLOAT32_C(   397.71), EASYSIMD_FLOAT32_C(   495.33)),
      UINT8_C(147),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -417.11), EASYSIMD_FLOAT32_C(  -321.70),
                         EASYSIMD_FLOAT32_C(   929.20), EASYSIMD_FLOAT32_C(  -973.32),
                         EASYSIMD_FLOAT32_C(   120.89), EASYSIMD_FLOAT32_C(   122.15),
                         EASYSIMD_FLOAT32_C(   252.56), EASYSIMD_FLOAT32_C(   335.57)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -417.11), EASYSIMD_FLOAT32_C(   440.56),
                         EASYSIMD_FLOAT32_C(   471.24), EASYSIMD_FLOAT32_C(  -973.32),
                         EASYSIMD_FLOAT32_C(  -133.99), EASYSIMD_FLOAT32_C(  -169.40),
                         EASYSIMD_FLOAT32_C(   252.56), EASYSIMD_FLOAT32_C(   335.57)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   637.95), EASYSIMD_FLOAT32_C(   655.37),
                         EASYSIMD_FLOAT32_C(   156.29), EASYSIMD_FLOAT32_C(   -73.51),
                         EASYSIMD_FLOAT32_C(  -940.14), EASYSIMD_FLOAT32_C(    79.12),
                         EASYSIMD_FLOAT32_C(  -920.60), EASYSIMD_FLOAT32_C(   773.77)),
      UINT8_C(111),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -286.54), EASYSIMD_FLOAT32_C(  -686.34),
                         EASYSIMD_FLOAT32_C(   368.35), EASYSIMD_FLOAT32_C(  -817.20),
                         EASYSIMD_FLOAT32_C(  -376.39), EASYSIMD_FLOAT32_C(   454.17),
                         EASYSIMD_FLOAT32_C(   819.05), EASYSIMD_FLOAT32_C(   500.81)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   637.95), EASYSIMD_FLOAT32_C(  -686.34),
                         EASYSIMD_FLOAT32_C(   368.35), EASYSIMD_FLOAT32_C(   -73.51),
                         EASYSIMD_FLOAT32_C(  -376.39), EASYSIMD_FLOAT32_C(   454.17),
                         EASYSIMD_FLOAT32_C(   819.05), EASYSIMD_FLOAT32_C(   500.81)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = test_vec[i].a;
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mov_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mov_ps");
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mov_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C( -56), INT8_C(  10), INT8_C( 103), INT8_C(  84),
                           INT8_C(  93), INT8_C(  24), INT8_C( -78), INT8_C(  35),
                           INT8_C( 125), INT8_C( -63), INT8_C(  19), INT8_C(   4),
                           INT8_C(   3), INT8_C( -87), INT8_C(  98), INT8_C(-113),
                           INT8_C(  23), INT8_C(-124), INT8_C( -87), INT8_C(  63),
                           INT8_C( 108), INT8_C( -18), INT8_C( -27), INT8_C(-127),
                           INT8_C( -60), INT8_C(  60), INT8_C( -56), INT8_C(   3),
                           INT8_C(-128), INT8_C( -62), INT8_C(  52), INT8_C( -74),
                           INT8_C( -87), INT8_C(  32), INT8_C(  46), INT8_C(-128),
                           INT8_C(  54), INT8_C( -19), INT8_C(  12), INT8_C(  22),
                           INT8_C( -94), INT8_C( -84), INT8_C( -58), INT8_C(  92),
                           INT8_C( -70), INT8_C( -25), INT8_C(  91), INT8_C( -45),
                           INT8_C(   5), INT8_C( 109), INT8_C( -46), INT8_C(  37),
                           INT8_C(   7), INT8_C(  44), INT8_C(  41), INT8_C(-106),
                           INT8_C(  82), INT8_C(  48), INT8_C( -21), INT8_C( -90),
                           INT8_C( 105), INT8_C( 117), INT8_C( -21), INT8_C(  39)),
      UINT64_C(11516165625622400866),
      easysimd_mm512_set_epi8(INT8_C(  43), INT8_C( -65), INT8_C(  47), INT8_C(  36),
                           INT8_C(-101), INT8_C(   5), INT8_C( -76), INT8_C( -57),
                           INT8_C(  77), INT8_C(  48), INT8_C( -46), INT8_C( -15),
                           INT8_C(  78), INT8_C( 108), INT8_C( 114), INT8_C(  83),
                           INT8_C( -72), INT8_C(  21), INT8_C( 100), INT8_C( 121),
                           INT8_C(  29), INT8_C( -74), INT8_C(  81), INT8_C( -13),
                           INT8_C( -57), INT8_C( -17), INT8_C(  20), INT8_C(-109),
                           INT8_C( -87), INT8_C( 127), INT8_C(  92), INT8_C(-119),
                           INT8_C(  26), INT8_C( 123), INT8_C( -51), INT8_C( 109),
                           INT8_C(  30), INT8_C( -58), INT8_C(-117), INT8_C(  82),
                           INT8_C( 111), INT8_C( -10), INT8_C( -10), INT8_C( -68),
                           INT8_C(  -4), INT8_C(  -7), INT8_C( 117), INT8_C(  92),
                           INT8_C(  94), INT8_C( -65), INT8_C( 109), INT8_C(  81),
                           INT8_C( -71), INT8_C( -46), INT8_C( 113), INT8_C(   9),
                           INT8_C( 123), INT8_C( -39), INT8_C(  76), INT8_C(  68),
                           INT8_C(  -3), INT8_C(  36), INT8_C(   0), INT8_C(   6)),
      easysimd_mm512_set_epi8(INT8_C(  43), INT8_C(  10), INT8_C( 103), INT8_C(  36),
                           INT8_C(-101), INT8_C(   5), INT8_C( -76), INT8_C( -57),
                           INT8_C(  77), INT8_C(  48), INT8_C(  19), INT8_C( -15),
                           INT8_C(   3), INT8_C( -87), INT8_C(  98), INT8_C(  83),
                           INT8_C( -72), INT8_C(-124), INT8_C( 100), INT8_C(  63),
                           INT8_C( 108), INT8_C( -18), INT8_C(  81), INT8_C( -13),
                           INT8_C( -57), INT8_C(  60), INT8_C( -56), INT8_C(-109),
                           INT8_C( -87), INT8_C( 127), INT8_C(  92), INT8_C(-119),
                           INT8_C( -87), INT8_C(  32), INT8_C(  46), INT8_C(-128),
                           INT8_C(  30), INT8_C( -19), INT8_C(  12), INT8_C(  82),
                           INT8_C( 111), INT8_C( -10), INT8_C( -58), INT8_C(  92),
                           INT8_C(  -4), INT8_C( -25), INT8_C(  91), INT8_C(  92),
                           INT8_C(  94), INT8_C( 109), INT8_C( -46), INT8_C(  37),
                           INT8_C( -71), INT8_C( -46), INT8_C( 113), INT8_C(   9),
                           INT8_C(  82), INT8_C( -39), INT8_C(  76), INT8_C( -90),
                           INT8_C( 105), INT8_C( 117), INT8_C(   0), INT8_C(  39)) },
    { easysimd_mm512_set_epi8(INT8_C( -25), INT8_C(-127), INT8_C(  40), INT8_C( -10),
                           INT8_C(  75), INT8_C(-123), INT8_C(  78), INT8_C(  -2),
                           INT8_C( -83), INT8_C( -74), INT8_C( -51), INT8_C(  46),
                           INT8_C(  60), INT8_C( -39), INT8_C( 124), INT8_C(-117),
                           INT8_C(  70), INT8_C(  66), INT8_C( -35), INT8_C( -51),
                           INT8_C( -64), INT8_C( -61), INT8_C(-113), INT8_C(   2),
                           INT8_C(  -4), INT8_C( -72), INT8_C( 113), INT8_C( -63),
                           INT8_C( -49), INT8_C(  70), INT8_C( -50), INT8_C(  52),
                           INT8_C(   0), INT8_C(  13), INT8_C(  74), INT8_C( -60),
                           INT8_C( 103), INT8_C(  -7), INT8_C( -61), INT8_C( -37),
                           INT8_C( -79), INT8_C( -77), INT8_C( -81), INT8_C( -83),
                           INT8_C(  94), INT8_C(  52), INT8_C( -73), INT8_C(  76),
                           INT8_C(-120), INT8_C(  80), INT8_C( -52), INT8_C(-126),
                           INT8_C( -40), INT8_C( 119), INT8_C( -83), INT8_C(  62),
                           INT8_C(  20), INT8_C(  23), INT8_C( 120), INT8_C( -13),
                           INT8_C(  82), INT8_C(  32), INT8_C( -44), INT8_C( -44)),
      UINT64_C( 5249838983459854712),
      easysimd_mm512_set_epi8(INT8_C( -68), INT8_C( 121), INT8_C(-102), INT8_C( -30),
                           INT8_C(-103), INT8_C( -31), INT8_C(  24), INT8_C( -55),
                           INT8_C(  -5), INT8_C(   8), INT8_C( -38), INT8_C(  37),
                           INT8_C(  15), INT8_C(-120), INT8_C(  17), INT8_C( -63),
                           INT8_C( 107), INT8_C( -41), INT8_C( -53), INT8_C(-107),
                           INT8_C(  91), INT8_C(  -9), INT8_C(-127), INT8_C( -39),
                           INT8_C( 105), INT8_C( -27), INT8_C(  96), INT8_C( -96),
                           INT8_C(   2), INT8_C(  44), INT8_C(  11), INT8_C( -43),
                           INT8_C( -52), INT8_C( 126), INT8_C( 125), INT8_C( 121),
                           INT8_C(  87), INT8_C( -95), INT8_C( 120), INT8_C( -46),
                           INT8_C(  25), INT8_C(  71), INT8_C( 117), INT8_C(  47),
                           INT8_C(-110), INT8_C( -87), INT8_C( -36), INT8_C(  25),
                           INT8_C(  24), INT8_C( -92), INT8_C(  99), INT8_C(  15),
                           INT8_C(  39), INT8_C(  38), INT8_C( 111), INT8_C( -29),
                           INT8_C(  62), INT8_C(  34), INT8_C(-113), INT8_C( 121),
                           INT8_C( -31), INT8_C(-111), INT8_C(  76), INT8_C(-113)),
      easysimd_mm512_set_epi8(INT8_C( -25), INT8_C( 121), INT8_C(  40), INT8_C( -10),
                           INT8_C(-103), INT8_C(-123), INT8_C(  78), INT8_C(  -2),
                           INT8_C(  -5), INT8_C(   8), INT8_C( -51), INT8_C(  37),
                           INT8_C(  15), INT8_C( -39), INT8_C(  17), INT8_C( -63),
                           INT8_C(  70), INT8_C(  66), INT8_C( -53), INT8_C( -51),
                           INT8_C(  91), INT8_C(  -9), INT8_C(-113), INT8_C(   2),
                           INT8_C( 105), INT8_C( -72), INT8_C(  96), INT8_C( -96),
                           INT8_C(   2), INT8_C(  44), INT8_C( -50), INT8_C( -43),
                           INT8_C( -52), INT8_C(  13), INT8_C(  74), INT8_C( 121),
                           INT8_C(  87), INT8_C(  -7), INT8_C( -61), INT8_C( -46),
                           INT8_C( -79), INT8_C( -77), INT8_C( -81), INT8_C(  47),
                           INT8_C(-110), INT8_C( -87), INT8_C( -36), INT8_C(  25),
                           INT8_C(-120), INT8_C( -92), INT8_C( -52), INT8_C(  15),
                           INT8_C(  39), INT8_C( 119), INT8_C( -83), INT8_C( -29),
                           INT8_C(  20), INT8_C(  34), INT8_C(-113), INT8_C( 121),
                           INT8_C( -31), INT8_C(  32), INT8_C( -44), INT8_C( -44)) },
    { easysimd_mm512_set_epi8(INT8_C( 117), INT8_C(   0), INT8_C( -58), INT8_C( -82),
                           INT8_C( -40), INT8_C( -36), INT8_C( -30), INT8_C( -56),
                           INT8_C( -68), INT8_C( -93), INT8_C(  25), INT8_C( -68),
                           INT8_C(   8), INT8_C(  64), INT8_C( -70), INT8_C( -19),
                           INT8_C( -64), INT8_C( -54), INT8_C( 120), INT8_C(  61),
                           INT8_C( -73), INT8_C(  47), INT8_C(-113), INT8_C(  68),
                           INT8_C( -44), INT8_C( -96), INT8_C(-106), INT8_C( -68),
                           INT8_C(  75), INT8_C( -42), INT8_C(  94), INT8_C( -68),
                           INT8_C( -10), INT8_C(  41), INT8_C( -90), INT8_C(-110),
                           INT8_C(-116), INT8_C( -51), INT8_C( -75), INT8_C( 102),
                           INT8_C(  14), INT8_C( 110), INT8_C(  89), INT8_C(   5),
                           INT8_C( -49), INT8_C(  29), INT8_C(  63), INT8_C( -67),
                           INT8_C( -85), INT8_C(  90), INT8_C(  97), INT8_C( -38),
                           INT8_C( -35), INT8_C(   6), INT8_C(  37), INT8_C( 106),
                           INT8_C( 102), INT8_C( 109), INT8_C(  47), INT8_C(  29),
                           INT8_C( -81), INT8_C(-113), INT8_C( -49), INT8_C(  18)),
      UINT64_C(16853471664498189804),
      easysimd_mm512_set_epi8(INT8_C( -53), INT8_C( -59), INT8_C(  -6), INT8_C( -57),
                           INT8_C(  97), INT8_C(  68), INT8_C( -67), INT8_C( 117),
                           INT8_C( -92), INT8_C(  -3), INT8_C(   2), INT8_C(  59),
                           INT8_C(  53), INT8_C( -13), INT8_C( -31), INT8_C(  47),
                           INT8_C( -33), INT8_C(  67), INT8_C( -43), INT8_C( -53),
                           INT8_C( -52), INT8_C(  -3), INT8_C(  85), INT8_C(  48),
                           INT8_C( -45), INT8_C( -72), INT8_C(  96), INT8_C(  85),
                           INT8_C(  81), INT8_C(  28), INT8_C( -50), INT8_C(-107),
                           INT8_C( -56), INT8_C( -85), INT8_C( -83), INT8_C( -25),
                           INT8_C(  78), INT8_C(  13), INT8_C(  41), INT8_C(  86),
                           INT8_C( -28), INT8_C(  90), INT8_C(  29), INT8_C(-115),
                           INT8_C( -97), INT8_C(-121), INT8_C( -51), INT8_C(  53),
                           INT8_C( -73), INT8_C( -64), INT8_C( -86), INT8_C( -65),
                           INT8_C( 124), INT8_C(-109), INT8_C(  79), INT8_C(-111),
                           INT8_C(  64), INT8_C( -98), INT8_C(  -1), INT8_C( -43),
                           INT8_C(  -4), INT8_C(  72), INT8_C( 108), INT8_C( -95)),
      easysimd_mm512_set_epi8(INT8_C( -53), INT8_C( -59), INT8_C(  -6), INT8_C( -82),
                           INT8_C(  97), INT8_C( -36), INT8_C( -30), INT8_C( 117),
                           INT8_C( -92), INT8_C(  -3), INT8_C(   2), INT8_C( -68),
                           INT8_C(   8), INT8_C(  64), INT8_C( -31), INT8_C(  47),
                           INT8_C( -33), INT8_C( -54), INT8_C( 120), INT8_C(  61),
                           INT8_C( -52), INT8_C(  -3), INT8_C(  85), INT8_C(  48),
                           INT8_C( -44), INT8_C( -96), INT8_C(  96), INT8_C( -68),
                           INT8_C(  81), INT8_C(  28), INT8_C( -50), INT8_C(-107),
                           INT8_C( -10), INT8_C( -85), INT8_C( -83), INT8_C( -25),
                           INT8_C(-116), INT8_C( -51), INT8_C(  41), INT8_C( 102),
                           INT8_C( -28), INT8_C( 110), INT8_C(  89), INT8_C(   5),
                           INT8_C( -49), INT8_C(-121), INT8_C( -51), INT8_C( -67),
                           INT8_C( -73), INT8_C(  90), INT8_C( -86), INT8_C( -38),
                           INT8_C( -35), INT8_C(-109), INT8_C(  37), INT8_C(-111),
                           INT8_C(  64), INT8_C( -98), INT8_C(  -1), INT8_C(  29),
                           INT8_C(  -4), INT8_C(  72), INT8_C( -49), INT8_C(  18)) },
    { easysimd_mm512_set_epi8(INT8_C(  37), INT8_C( 104), INT8_C( -81), INT8_C( 113),
                           INT8_C(  31), INT8_C( -10), INT8_C( -32), INT8_C( -91),
                           INT8_C(  51), INT8_C( -51), INT8_C(  60), INT8_C(  38),
                           INT8_C(  -1), INT8_C( -38), INT8_C(   2), INT8_C( 110),
                           INT8_C( -61), INT8_C(  91), INT8_C( -50), INT8_C(  89),
                           INT8_C(  27), INT8_C( -13), INT8_C( 111), INT8_C( -20),
                           INT8_C(  51), INT8_C( -66), INT8_C( -26), INT8_C(  66),
                           INT8_C(  45), INT8_C( -59), INT8_C( -45), INT8_C(-102),
                           INT8_C(  84), INT8_C(-102), INT8_C( 103), INT8_C( 111),
                           INT8_C( -47), INT8_C(  74), INT8_C( 111), INT8_C(  62),
                           INT8_C(  41), INT8_C(  -4), INT8_C( -19), INT8_C(  26),
                           INT8_C(-127), INT8_C( -41), INT8_C(  14), INT8_C(  10),
                           INT8_C(  63), INT8_C(  99), INT8_C(  51), INT8_C(-115),
                           INT8_C( 118), INT8_C( -85), INT8_C(-111), INT8_C(  19),
                           INT8_C(  43), INT8_C( -97), INT8_C( 107), INT8_C( 127),
                           INT8_C(-100), INT8_C(  45), INT8_C( -77), INT8_C(  77)),
      UINT64_C( 3141946940694640625),
      easysimd_mm512_set_epi8(INT8_C( -47), INT8_C( -86), INT8_C(  35), INT8_C(-110),
                           INT8_C(  95), INT8_C(  -9), INT8_C(  86), INT8_C(   9),
                           INT8_C(  31), INT8_C(  48), INT8_C(  63), INT8_C(  -6),
                           INT8_C( -36), INT8_C( -47), INT8_C(  95), INT8_C( -20),
                           INT8_C(  21), INT8_C(  -9), INT8_C(  -2), INT8_C(  26),
                           INT8_C(  63), INT8_C(  36), INT8_C( -33), INT8_C(  58),
                           INT8_C( -40), INT8_C( 106), INT8_C(   2), INT8_C( -51),
                           INT8_C( -13), INT8_C( -76), INT8_C( -77), INT8_C( -77),
                           INT8_C(  65), INT8_C(  44), INT8_C( -48), INT8_C( 121),
                           INT8_C(-106), INT8_C(  35), INT8_C(  49), INT8_C( -67),
                           INT8_C( -35), INT8_C( -29), INT8_C(  89), INT8_C(  91),
                           INT8_C( -53), INT8_C( -62), INT8_C( 107), INT8_C( -42),
                           INT8_C(-115), INT8_C(  52), INT8_C( -17), INT8_C(  64),
                           INT8_C(-105), INT8_C(-106), INT8_C(  65), INT8_C(  97),
                           INT8_C(  85), INT8_C(  52), INT8_C( -17), INT8_C(   6),
                           INT8_C( -73), INT8_C( 109), INT8_C(  99), INT8_C(   9)),
      easysimd_mm512_set_epi8(INT8_C(  37), INT8_C( 104), INT8_C(  35), INT8_C( 113),
                           INT8_C(  95), INT8_C( -10), INT8_C(  86), INT8_C(   9),
                           INT8_C(  31), INT8_C( -51), INT8_C(  60), INT8_C(  -6),
                           INT8_C( -36), INT8_C( -38), INT8_C(  95), INT8_C( 110),
                           INT8_C( -61), INT8_C(  -9), INT8_C(  -2), INT8_C(  26),
                           INT8_C(  27), INT8_C( -13), INT8_C( 111), INT8_C( -20),
                           INT8_C(  51), INT8_C( -66), INT8_C( -26), INT8_C( -51),
                           INT8_C( -13), INT8_C( -59), INT8_C( -45), INT8_C(-102),
                           INT8_C(  65), INT8_C(-102), INT8_C( 103), INT8_C( 111),
                           INT8_C( -47), INT8_C(  35), INT8_C(  49), INT8_C( -67),
                           INT8_C(  41), INT8_C(  -4), INT8_C(  89), INT8_C(  91),
                           INT8_C( -53), INT8_C( -41), INT8_C( 107), INT8_C(  10),
                           INT8_C(-115), INT8_C(  99), INT8_C( -17), INT8_C(  64),
                           INT8_C( 118), INT8_C(-106), INT8_C(  65), INT8_C(  97),
                           INT8_C(  85), INT8_C(  52), INT8_C( -17), INT8_C(   6),
                           INT8_C(-100), INT8_C(  45), INT8_C( -77), INT8_C(   9)) },
    { easysimd_mm512_set_epi8(INT8_C(  -3), INT8_C( -87), INT8_C(   8), INT8_C( -39),
                           INT8_C(-122), INT8_C(  94), INT8_C( -13), INT8_C(  31),
                           INT8_C( 125), INT8_C( -74), INT8_C(   5), INT8_C( 127),
                           INT8_C(  68), INT8_C(  61), INT8_C(  93), INT8_C(  69),
                           INT8_C(  92), INT8_C( -67), INT8_C(   4), INT8_C(  -4),
                           INT8_C(  29), INT8_C( -70), INT8_C(  28), INT8_C( -34),
                           INT8_C( -99), INT8_C(  28), INT8_C(  -2), INT8_C(  39),
                           INT8_C( -60), INT8_C(  91), INT8_C(  66), INT8_C(-121),
                           INT8_C(  40), INT8_C( -99), INT8_C(   6), INT8_C( 105),
                           INT8_C(  36), INT8_C( -85), INT8_C(  62), INT8_C( 102),
                           INT8_C(  23), INT8_C(-110), INT8_C( -92), INT8_C(  59),
                           INT8_C(  17), INT8_C( -54), INT8_C(   5), INT8_C(  81),
                           INT8_C( -71), INT8_C(  68), INT8_C( 114), INT8_C( -60),
                           INT8_C(  39), INT8_C( -49), INT8_C( -84), INT8_C( 114),
                           INT8_C( -81), INT8_C( 122), INT8_C(  97), INT8_C( -16),
                           INT8_C(  21), INT8_C( -76), INT8_C( -80), INT8_C( -61)),
      UINT64_C( 7453836348998775155),
      easysimd_mm512_set_epi8(INT8_C(-107), INT8_C(  74), INT8_C( -78), INT8_C( -91),
                           INT8_C(   7), INT8_C(   9), INT8_C(  96), INT8_C( -14),
                           INT8_C(  10), INT8_C(  85), INT8_C(  75), INT8_C( -98),
                           INT8_C( -93), INT8_C(  66), INT8_C(-107), INT8_C( -73),
                           INT8_C(-106), INT8_C( -46), INT8_C(  35), INT8_C(  89),
                           INT8_C( -81), INT8_C( -42), INT8_C( -88), INT8_C(  17),
                           INT8_C(  34), INT8_C(  81), INT8_C(-103), INT8_C(  99),
                           INT8_C(  -3), INT8_C( 116), INT8_C( -98), INT8_C(-111),
                           INT8_C( -10), INT8_C( 120), INT8_C( 115), INT8_C(  38),
                           INT8_C( -96), INT8_C( -48), INT8_C( -20), INT8_C(  25),
                           INT8_C(  44), INT8_C( -60), INT8_C( -69), INT8_C(   1),
                           INT8_C(  63), INT8_C(   5), INT8_C( -90), INT8_C( -83),
                           INT8_C( -81), INT8_C( 119), INT8_C( -80), INT8_C(   7),
                           INT8_C(-116), INT8_C(  46), INT8_C( -50), INT8_C( -16),
                           INT8_C( -90), INT8_C(  31), INT8_C(  57), INT8_C( -10),
                           INT8_C(  87), INT8_C(-123), INT8_C(-112), INT8_C(-115)),
      easysimd_mm512_set_epi8(INT8_C(  -3), INT8_C(  74), INT8_C( -78), INT8_C( -39),
                           INT8_C(-122), INT8_C(   9), INT8_C(  96), INT8_C( -14),
                           INT8_C( 125), INT8_C(  85), INT8_C(  75), INT8_C( -98),
                           INT8_C(  68), INT8_C(  61), INT8_C(  93), INT8_C( -73),
                           INT8_C(  92), INT8_C( -46), INT8_C(   4), INT8_C(  89),
                           INT8_C( -81), INT8_C( -70), INT8_C(  28), INT8_C( -34),
                           INT8_C(  34), INT8_C(  28), INT8_C(-103), INT8_C(  39),
                           INT8_C(  -3), INT8_C(  91), INT8_C( -98), INT8_C(-121),
                           INT8_C( -10), INT8_C( 120), INT8_C(   6), INT8_C(  38),
                           INT8_C(  36), INT8_C( -85), INT8_C( -20), INT8_C(  25),
                           INT8_C(  44), INT8_C( -60), INT8_C( -69), INT8_C(   1),
                           INT8_C(  17), INT8_C(   5), INT8_C( -90), INT8_C( -83),
                           INT8_C( -71), INT8_C( 119), INT8_C( -80), INT8_C(   7),
                           INT8_C(  39), INT8_C( -49), INT8_C( -84), INT8_C( -16),
                           INT8_C( -81), INT8_C(  31), INT8_C(  57), INT8_C( -10),
                           INT8_C(  21), INT8_C( -76), INT8_C(-112), INT8_C(-115)) },
    { easysimd_mm512_set_epi8(INT8_C(  44), INT8_C(  93), INT8_C(  98), INT8_C(  56),
                           INT8_C(-118), INT8_C( -35), INT8_C( -11), INT8_C(  90),
                           INT8_C(-105), INT8_C(   2), INT8_C( 120), INT8_C(  -6),
                           INT8_C(  31), INT8_C(  70), INT8_C(  48), INT8_C(  80),
                           INT8_C( -45), INT8_C(  63), INT8_C(-108), INT8_C( -43),
                           INT8_C(  -1), INT8_C(  90), INT8_C( -88), INT8_C( -74),
                           INT8_C(  36), INT8_C(  30), INT8_C(-102), INT8_C(  22),
                           INT8_C( 127), INT8_C(-117), INT8_C(   6), INT8_C( -94),
                           INT8_C(-110), INT8_C( -41), INT8_C(  20), INT8_C(-121),
                           INT8_C(-106), INT8_C(  73), INT8_C( 119), INT8_C( -14),
                           INT8_C( 107), INT8_C(  48), INT8_C(   4), INT8_C(  95),
                           INT8_C(  84), INT8_C( -53), INT8_C( -11), INT8_C( -26),
                           INT8_C(  53), INT8_C( 115), INT8_C( -51), INT8_C( -54),
                           INT8_C( -28), INT8_C(  93), INT8_C(-128), INT8_C(-104),
                           INT8_C(  35), INT8_C(  58), INT8_C(-101), INT8_C( 110),
                           INT8_C(-115), INT8_C( -77), INT8_C( -98), INT8_C( 114)),
      UINT64_C(12105239831388369272),
      easysimd_mm512_set_epi8(INT8_C(  23), INT8_C( 124), INT8_C(  68), INT8_C(  41),
                           INT8_C( 105), INT8_C(  81), INT8_C( -85), INT8_C(   1),
                           INT8_C(  93), INT8_C(  15), INT8_C(  -8), INT8_C(  44),
                           INT8_C(-105), INT8_C(  88), INT8_C(  99), INT8_C( -39),
                           INT8_C( 119), INT8_C(  69), INT8_C( 127), INT8_C( 121),
                           INT8_C(  78), INT8_C(  25), INT8_C(-125), INT8_C(  52),
                           INT8_C(  -5), INT8_C( -83), INT8_C(-101), INT8_C(  76),
                           INT8_C( -86), INT8_C( -10), INT8_C( -96), INT8_C( -15),
                           INT8_C(  51), INT8_C( 115), INT8_C(  24), INT8_C(   5),
                           INT8_C( -93), INT8_C(  76), INT8_C( -76), INT8_C(-120),
                           INT8_C(  26), INT8_C(  95), INT8_C( -66), INT8_C(-119),
                           INT8_C( -88), INT8_C( 113), INT8_C( -39), INT8_C( -13),
                           INT8_C(  -1), INT8_C( -15), INT8_C(  -7), INT8_C(-103),
                           INT8_C(  99), INT8_C( 122), INT8_C(-107), INT8_C( -48),
                           INT8_C(-117), INT8_C(   1), INT8_C( -98), INT8_C(  41),
                           INT8_C(-124), INT8_C(  15), INT8_C(  39), INT8_C(-108)),
      easysimd_mm512_set_epi8(INT8_C(  23), INT8_C(  93), INT8_C(  68), INT8_C(  56),
                           INT8_C(-118), INT8_C(  81), INT8_C( -85), INT8_C(   1),
                           INT8_C(  93), INT8_C(  15), INT8_C(  -8), INT8_C(  44),
                           INT8_C(-105), INT8_C(  88), INT8_C(  99), INT8_C(  80),
                           INT8_C( -45), INT8_C(  69), INT8_C( 127), INT8_C( 121),
                           INT8_C(  -1), INT8_C(  90), INT8_C(-125), INT8_C(  52),
                           INT8_C(  36), INT8_C( -83), INT8_C(-101), INT8_C(  76),
                           INT8_C( -86), INT8_C( -10), INT8_C(   6), INT8_C( -15),
                           INT8_C(  51), INT8_C( -41), INT8_C(  20), INT8_C(-121),
                           INT8_C(-106), INT8_C(  76), INT8_C( -76), INT8_C( -14),
                           INT8_C(  26), INT8_C(  95), INT8_C(   4), INT8_C(-119),
                           INT8_C(  84), INT8_C( -53), INT8_C( -11), INT8_C( -26),
                           INT8_C(  53), INT8_C( 115), INT8_C(  -7), INT8_C(-103),
                           INT8_C( -28), INT8_C(  93), INT8_C(-128), INT8_C( -48),
                           INT8_C(  35), INT8_C(   1), INT8_C( -98), INT8_C(  41),
                           INT8_C(-124), INT8_C( -77), INT8_C( -98), INT8_C( 114)) },
    { easysimd_mm512_set_epi8(INT8_C( -95), INT8_C(  85), INT8_C( -91), INT8_C(  56),
                           INT8_C(  91), INT8_C( -49), INT8_C( 106), INT8_C(  16),
                           INT8_C(  15), INT8_C(  10), INT8_C(  30), INT8_C(  12),
                           INT8_C(  22), INT8_C( -73), INT8_C(  68), INT8_C(  83),
                           INT8_C( 121), INT8_C(  56), INT8_C( 108), INT8_C( -49),
                           INT8_C(-107), INT8_C(  73), INT8_C( -10), INT8_C( 107),
                           INT8_C( -99), INT8_C( 105), INT8_C( -46), INT8_C(  26),
                           INT8_C(  20), INT8_C( -18), INT8_C(  82), INT8_C(  37),
                           INT8_C( -80), INT8_C( -81), INT8_C(  99), INT8_C(  24),
                           INT8_C(  88), INT8_C(  86), INT8_C( -71), INT8_C(  54),
                           INT8_C(-121), INT8_C(  30), INT8_C(  98), INT8_C( -68),
                           INT8_C(   1), INT8_C(  93), INT8_C(  79), INT8_C( -44),
                           INT8_C( -93), INT8_C( -75), INT8_C(  53), INT8_C(  21),
                           INT8_C(  44), INT8_C(-111), INT8_C( 104), INT8_C(-101),
                           INT8_C( -63), INT8_C(-108), INT8_C(  57), INT8_C( -13),
                           INT8_C(  20), INT8_C(  -6), INT8_C( -84), INT8_C(  38)),
      UINT64_C(14977178912506627906),
      easysimd_mm512_set_epi8(INT8_C(  94), INT8_C(-107), INT8_C(  99), INT8_C(  86),
                           INT8_C(-126), INT8_C(  79), INT8_C(  11), INT8_C(-123),
                           INT8_C( 112), INT8_C(  11), INT8_C(  44), INT8_C( -11),
                           INT8_C( -10), INT8_C(  70), INT8_C( -45), INT8_C( 124),
                           INT8_C(-122), INT8_C(  27), INT8_C(  30), INT8_C(  57),
                           INT8_C( -81), INT8_C( -89), INT8_C( 107), INT8_C( -36),
                           INT8_C( 100), INT8_C( -65), INT8_C( -83), INT8_C(  -7),
                           INT8_C(  33), INT8_C( -77), INT8_C( -24), INT8_C(  93),
                           INT8_C( -88), INT8_C(   0), INT8_C( 125), INT8_C( -84),
                           INT8_C( 102), INT8_C( 110), INT8_C(  49), INT8_C( -75),
                           INT8_C(-106), INT8_C(  92), INT8_C(  31), INT8_C(  93),
                           INT8_C(-123), INT8_C( -68), INT8_C( 119), INT8_C( -49),
                           INT8_C( -54), INT8_C( 105), INT8_C(  12), INT8_C(-117),
                           INT8_C(-105), INT8_C(  27), INT8_C(  72), INT8_C( -27),
                           INT8_C(  59), INT8_C(-110), INT8_C(   8), INT8_C(-113),
                           INT8_C( -36), INT8_C(  -7), INT8_C( -64), INT8_C(  96)),
      easysimd_mm512_set_epi8(INT8_C(  94), INT8_C(-107), INT8_C( -91), INT8_C(  56),
                           INT8_C(-126), INT8_C(  79), INT8_C(  11), INT8_C(-123),
                           INT8_C( 112), INT8_C(  11), INT8_C(  30), INT8_C( -11),
                           INT8_C( -10), INT8_C( -73), INT8_C(  68), INT8_C( 124),
                           INT8_C(-122), INT8_C(  56), INT8_C(  30), INT8_C( -49),
                           INT8_C(-107), INT8_C(  73), INT8_C( -10), INT8_C( 107),
                           INT8_C( 100), INT8_C( -65), INT8_C( -46), INT8_C(  -7),
                           INT8_C(  33), INT8_C( -77), INT8_C( -24), INT8_C(  37),
                           INT8_C( -80), INT8_C(   0), INT8_C(  99), INT8_C( -84),
                           INT8_C(  88), INT8_C(  86), INT8_C( -71), INT8_C( -75),
                           INT8_C(-121), INT8_C(  92), INT8_C(  31), INT8_C( -68),
                           INT8_C(-123), INT8_C( -68), INT8_C(  79), INT8_C( -49),
                           INT8_C( -54), INT8_C( 105), INT8_C(  53), INT8_C(  21),
                           INT8_C(  44), INT8_C(  27), INT8_C(  72), INT8_C( -27),
                           INT8_C( -63), INT8_C(-110), INT8_C(  57), INT8_C( -13),
                           INT8_C(  20), INT8_C(  -6), INT8_C( -64), INT8_C(  38)) },
    { easysimd_mm512_set_epi8(INT8_C(  60), INT8_C( -31), INT8_C(  26), INT8_C(   5),
                           INT8_C(  69), INT8_C( -80), INT8_C(  85), INT8_C(   4),
                           INT8_C( -32), INT8_C(  20), INT8_C( 122), INT8_C( -81),
                           INT8_C( -84), INT8_C(-101), INT8_C(-122), INT8_C(  51),
                           INT8_C(  95), INT8_C(  44), INT8_C(-103), INT8_C( 108),
                           INT8_C( 104), INT8_C( 108), INT8_C( 116), INT8_C(-113),
                           INT8_C( -40), INT8_C( 118), INT8_C( 107), INT8_C( 127),
                           INT8_C(  64), INT8_C( -95), INT8_C( 118), INT8_C(  32),
                           INT8_C( -48), INT8_C(  49), INT8_C(  12), INT8_C(-100),
                           INT8_C( -76), INT8_C(  61), INT8_C(  79), INT8_C( 120),
                           INT8_C(  50), INT8_C( -11), INT8_C( -35), INT8_C(-127),
                           INT8_C(  54), INT8_C(  -2), INT8_C(  71), INT8_C(  96),
                           INT8_C(  27), INT8_C( -13), INT8_C( -56), INT8_C(-110),
                           INT8_C(  65), INT8_C( -57), INT8_C( 119), INT8_C(  70),
                           INT8_C( 114), INT8_C( -31), INT8_C( 120), INT8_C( 113),
                           INT8_C(  92), INT8_C(  94), INT8_C( -85), INT8_C(  19)),
      UINT64_C(12789799828226766427),
      easysimd_mm512_set_epi8(INT8_C( -16), INT8_C( -99), INT8_C(-121), INT8_C( -68),
                           INT8_C( -23), INT8_C(  79), INT8_C(  48), INT8_C( -98),
                           INT8_C( -95), INT8_C(-117), INT8_C(  13), INT8_C( -11),
                           INT8_C(  79), INT8_C(  21), INT8_C( 127), INT8_C(  88),
                           INT8_C(   9), INT8_C(-119), INT8_C(  68), INT8_C(  72),
                           INT8_C(  35), INT8_C( -56), INT8_C( -74), INT8_C(  10),
                           INT8_C( 101), INT8_C( 124), INT8_C(  29), INT8_C( -55),
                           INT8_C( -78), INT8_C( -56), INT8_C( 124), INT8_C(  35),
                           INT8_C(  11), INT8_C( 106), INT8_C(  41), INT8_C(  59),
                           INT8_C(-108), INT8_C(  82), INT8_C( -41), INT8_C( 100),
                           INT8_C(  43), INT8_C( -34), INT8_C( 124), INT8_C(  15),
                           INT8_C( 113), INT8_C( -20), INT8_C( -83), INT8_C( -87),
                           INT8_C( -60), INT8_C(  22), INT8_C( -71), INT8_C(  86),
                           INT8_C(  73), INT8_C( -38), INT8_C(-106), INT8_C( 112),
                           INT8_C(  98), INT8_C(  51), INT8_C(-116), INT8_C(-126),
                           INT8_C( -96), INT8_C(   1), INT8_C(  23), INT8_C(-109)),
      easysimd_mm512_set_epi8(INT8_C( -16), INT8_C( -31), INT8_C(-121), INT8_C( -68),
                           INT8_C(  69), INT8_C( -80), INT8_C(  85), INT8_C( -98),
                           INT8_C( -32), INT8_C(-117), INT8_C(  13), INT8_C( -11),
                           INT8_C(  79), INT8_C(  21), INT8_C( 127), INT8_C(  51),
                           INT8_C(  95), INT8_C(-119), INT8_C(  68), INT8_C(  72),
                           INT8_C(  35), INT8_C( -56), INT8_C( -74), INT8_C(  10),
                           INT8_C( -40), INT8_C( 118), INT8_C(  29), INT8_C( 127),
                           INT8_C( -78), INT8_C( -56), INT8_C( 124), INT8_C(  32),
                           INT8_C( -48), INT8_C(  49), INT8_C(  41), INT8_C(  59),
                           INT8_C( -76), INT8_C(  82), INT8_C(  79), INT8_C( 100),
                           INT8_C(  43), INT8_C( -34), INT8_C( -35), INT8_C(-127),
                           INT8_C( 113), INT8_C( -20), INT8_C( -83), INT8_C(  96),
                           INT8_C( -60), INT8_C(  22), INT8_C( -71), INT8_C(  86),
                           INT8_C(  65), INT8_C( -38), INT8_C(-106), INT8_C(  70),
                           INT8_C( 114), INT8_C(  51), INT8_C( 120), INT8_C(-126),
                           INT8_C( -96), INT8_C(  94), INT8_C(  23), INT8_C(-109)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mov_epi8(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mov_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mov_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask32 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi16(INT16_C( -1573), INT16_C( -6208), INT16_C(-22615), INT16_C( -3799),
                            INT16_C( -8282), INT16_C(-15214), INT16_C(-19149), INT16_C(-11524),
                            INT16_C(-31971), INT16_C(  -228), INT16_C(-27669), INT16_C( 30774),
                            INT16_C( 14115), INT16_C(-29587), INT16_C( 15716), INT16_C( -9534),
                            INT16_C( 31897), INT16_C(-25045), INT16_C(-20462), INT16_C( 20289),
                            INT16_C( 31765), INT16_C( 26200), INT16_C( 22392), INT16_C( 19963),
                            INT16_C( -9240), INT16_C( -2240), INT16_C( -8342), INT16_C( 31950),
                            INT16_C(-15053), INT16_C( -6789), INT16_C( -5359), INT16_C(  9700)),
      UINT32_C( 175873983),
      easysimd_mm512_set_epi16(INT16_C( 11048), INT16_C(-23497), INT16_C(-22229), INT16_C( 22523),
                            INT16_C( 32192), INT16_C( 17944), INT16_C(  1999), INT16_C(  -512),
                            INT16_C( 22838), INT16_C( 10573), INT16_C( 22536), INT16_C(-21942),
                            INT16_C( -9055), INT16_C( -9938), INT16_C(  8369), INT16_C(-32672),
                            INT16_C( 24766), INT16_C(-31364), INT16_C(-26690), INT16_C( 14381),
                            INT16_C( 18820), INT16_C(  -175), INT16_C(-17138), INT16_C(  8826),
                            INT16_C( 16551), INT16_C( 18053), INT16_C( -1223), INT16_C(-28643),
                            INT16_C( -5550), INT16_C(  5011), INT16_C( 22761), INT16_C(   728)),
      easysimd_mm512_set_epi16(INT16_C( -1573), INT16_C( -6208), INT16_C(-22615), INT16_C( -3799),
                            INT16_C( 32192), INT16_C(-15214), INT16_C(  1999), INT16_C(-11524),
                            INT16_C(-31971), INT16_C( 10573), INT16_C( 22536), INT16_C(-21942),
                            INT16_C( -9055), INT16_C(-29587), INT16_C(  8369), INT16_C(-32672),
                            INT16_C( 24766), INT16_C(-25045), INT16_C(-20462), INT16_C( 14381),
                            INT16_C( 18820), INT16_C(  -175), INT16_C(-17138), INT16_C(  8826),
                            INT16_C( 16551), INT16_C( -2240), INT16_C( -1223), INT16_C(-28643),
                            INT16_C( -5550), INT16_C(  5011), INT16_C( 22761), INT16_C(   728)) },
    { easysimd_mm512_set_epi16(INT16_C(-26134), INT16_C(-18760), INT16_C( 11789), INT16_C( 30499),
                            INT16_C(-14297), INT16_C(-24132), INT16_C(  2429), INT16_C( -7785),
                            INT16_C(-19953), INT16_C(-20176), INT16_C( -1917), INT16_C( 18470),
                            INT16_C( 24222), INT16_C( 23067), INT16_C(-11100), INT16_C( 10676),
                            INT16_C(-21685), INT16_C( 31093), INT16_C( -8360), INT16_C(  1808),
                            INT16_C(-12418), INT16_C( 11067), INT16_C(-31728), INT16_C( -5932),
                            INT16_C(-22846), INT16_C(  -963), INT16_C(-15933), INT16_C(-24302),
                            INT16_C(-30670), INT16_C( 23129), INT16_C(-13017), INT16_C(  1590)),
      UINT32_C( 305590317),
      easysimd_mm512_set_epi16(INT16_C( -1674), INT16_C( -3241), INT16_C( 14220), INT16_C(-24128),
                            INT16_C(   866), INT16_C(-16676), INT16_C(-25544), INT16_C(   108),
                            INT16_C(  5014), INT16_C(-21407), INT16_C(-24139), INT16_C(-16531),
                            INT16_C( -2292), INT16_C(-22143), INT16_C( -5932), INT16_C(-26498),
                            INT16_C( 23176), INT16_C(-18719), INT16_C(  8259), INT16_C(  -216),
                            INT16_C(-21324), INT16_C( 14052), INT16_C( 27040), INT16_C(-18518),
                            INT16_C(-27268), INT16_C( -5574), INT16_C( 30453), INT16_C( 27189),
                            INT16_C( 26223), INT16_C(-14168), INT16_C(-11169), INT16_C( 22360)),
      easysimd_mm512_set_epi16(INT16_C(-26134), INT16_C(-18760), INT16_C( 11789), INT16_C(-24128),
                            INT16_C(-14297), INT16_C(-24132), INT16_C(-25544), INT16_C( -7785),
                            INT16_C(-19953), INT16_C(-20176), INT16_C(-24139), INT16_C(-16531),
                            INT16_C( 24222), INT16_C(-22143), INT16_C( -5932), INT16_C( 10676),
                            INT16_C( 23176), INT16_C(-18719), INT16_C(  8259), INT16_C(  -216),
                            INT16_C(-12418), INT16_C( 11067), INT16_C(-31728), INT16_C( -5932),
                            INT16_C(-22846), INT16_C(  -963), INT16_C( 30453), INT16_C(-24302),
                            INT16_C( 26223), INT16_C(-14168), INT16_C(-13017), INT16_C( 22360)) },
    { easysimd_mm512_set_epi16(INT16_C( -2488), INT16_C(  1592), INT16_C( -6444), INT16_C( 30598),
                            INT16_C(-17786), INT16_C( -8406), INT16_C(  4184), INT16_C( 17081),
                            INT16_C(-10288), INT16_C(-12158), INT16_C( -9059), INT16_C(-20947),
                            INT16_C(-17395), INT16_C( 27392), INT16_C( 13857), INT16_C( 24137),
                            INT16_C( 15083), INT16_C( -2381), INT16_C(  6197), INT16_C( 26607),
                            INT16_C(  -281), INT16_C( 20513), INT16_C( 11284), INT16_C( -8182),
                            INT16_C(   154), INT16_C( 25062), INT16_C( -9545), INT16_C( -8470),
                            INT16_C( 13769), INT16_C(  3698), INT16_C( 23943), INT16_C( 22626)),
      UINT32_C(3243316268),
      easysimd_mm512_set_epi16(INT16_C( -7817), INT16_C( 19901), INT16_C(-23323), INT16_C( 16418),
                            INT16_C( 24031), INT16_C(-12678), INT16_C( 26071), INT16_C(  6078),
                            INT16_C( -6446), INT16_C( 28656), INT16_C(-20287), INT16_C(-10682),
                            INT16_C( 17023), INT16_C( 12770), INT16_C( 15020), INT16_C( 12339),
                            INT16_C( 22254), INT16_C( -6532), INT16_C( 21585), INT16_C(-29214),
                            INT16_C( -5140), INT16_C(-13775), INT16_C(-14838), INT16_C(  1876),
                            INT16_C(-10206), INT16_C( -7669), INT16_C( 13226), INT16_C(  8231),
                            INT16_C( -5215), INT16_C(-29950), INT16_C(-17119), INT16_C(  7959)),
      easysimd_mm512_set_epi16(INT16_C( -7817), INT16_C( 19901), INT16_C( -6444), INT16_C( 30598),
                            INT16_C(-17786), INT16_C( -8406), INT16_C(  4184), INT16_C(  6078),
                            INT16_C(-10288), INT16_C( 28656), INT16_C( -9059), INT16_C(-10682),
                            INT16_C(-17395), INT16_C( 27392), INT16_C( 13857), INT16_C( 12339),
                            INT16_C( 15083), INT16_C( -2381), INT16_C(  6197), INT16_C(-29214),
                            INT16_C(  -281), INT16_C(-13775), INT16_C( 11284), INT16_C( -8182),
                            INT16_C(   154), INT16_C( 25062), INT16_C( 13226), INT16_C( -8470),
                            INT16_C( -5215), INT16_C(-29950), INT16_C( 23943), INT16_C( 22626)) },
    { easysimd_mm512_set_epi16(INT16_C(-17496), INT16_C(-16278), INT16_C( 28161), INT16_C( -9022),
                            INT16_C( 14893), INT16_C( 20773), INT16_C( 13716), INT16_C(-18494),
                            INT16_C( 22637), INT16_C(-20939), INT16_C(-10174), INT16_C( 12840),
                            INT16_C(-22747), INT16_C(-14668), INT16_C(  4699), INT16_C( 31693),
                            INT16_C( -8682), INT16_C(-21674), INT16_C( -4586), INT16_C(  -243),
                            INT16_C(-24920), INT16_C( 12309), INT16_C( 15037), INT16_C( 13960),
                            INT16_C(-29756), INT16_C( -4367), INT16_C(  8434), INT16_C( 16542),
                            INT16_C(  8529), INT16_C(-28527), INT16_C( -2939), INT16_C(-28531)),
      UINT32_C(3541815971),
      easysimd_mm512_set_epi16(INT16_C(  6266), INT16_C( 18547), INT16_C(-26004), INT16_C(-14807),
                            INT16_C( 23049), INT16_C(-28984), INT16_C( 18071), INT16_C(-18277),
                            INT16_C( 31923), INT16_C(-14090), INT16_C( -6209), INT16_C( 12842),
                            INT16_C(  1554), INT16_C(-27194), INT16_C(-25297), INT16_C( 17174),
                            INT16_C(  4338), INT16_C(-25809), INT16_C(  2041), INT16_C(-19046),
                            INT16_C(-17853), INT16_C(-18639), INT16_C( 25727), INT16_C(-30630),
                            INT16_C(-22895), INT16_C(  8885), INT16_C( 29491), INT16_C(-13154),
                            INT16_C(  9738), INT16_C(-20851), INT16_C(  1418), INT16_C( 24102)),
      easysimd_mm512_set_epi16(INT16_C(  6266), INT16_C( 18547), INT16_C( 28161), INT16_C(-14807),
                            INT16_C( 14893), INT16_C( 20773), INT16_C( 18071), INT16_C(-18277),
                            INT16_C( 22637), INT16_C(-20939), INT16_C(-10174), INT16_C( 12842),
                            INT16_C(  1554), INT16_C(-14668), INT16_C(-25297), INT16_C( 17174),
                            INT16_C(  4338), INT16_C(-25809), INT16_C( -4586), INT16_C(-19046),
                            INT16_C(-24920), INT16_C( 12309), INT16_C( 25727), INT16_C( 13960),
                            INT16_C(-22895), INT16_C( -4367), INT16_C( 29491), INT16_C( 16542),
                            INT16_C(  8529), INT16_C(-28527), INT16_C(  1418), INT16_C( 24102)) },
    { easysimd_mm512_set_epi16(INT16_C(  3849), INT16_C( 25678), INT16_C( 20058), INT16_C(-14631),
                            INT16_C(  9156), INT16_C( -9469), INT16_C( 26797), INT16_C(  4095),
                            INT16_C( 10328), INT16_C( -2602), INT16_C( 29484), INT16_C( 23696),
                            INT16_C( 10492), INT16_C( 15123), INT16_C( 12075), INT16_C(   -22),
                            INT16_C( -3095), INT16_C(-21257), INT16_C(  4948), INT16_C( 32515),
                            INT16_C(-22489), INT16_C( 12880), INT16_C(-31816), INT16_C( 14894),
                            INT16_C( 17736), INT16_C(  7904), INT16_C(-21771), INT16_C(-28666),
                            INT16_C(-14552), INT16_C(-24798), INT16_C(-10273), INT16_C(-18470)),
      UINT32_C(3424030392),
      easysimd_mm512_set_epi16(INT16_C( -3100), INT16_C(-21068), INT16_C( 28535), INT16_C(-17256),
                            INT16_C(-16628), INT16_C(  1662), INT16_C(-21371), INT16_C(  7545),
                            INT16_C( -2558), INT16_C(  5671), INT16_C(-14288), INT16_C(-27939),
                            INT16_C( 10529), INT16_C(-22955), INT16_C(  1055), INT16_C( 27502),
                            INT16_C( 28704), INT16_C(-22359), INT16_C(   974), INT16_C(-13833),
                            INT16_C(-10322), INT16_C( -9220), INT16_C(-23650), INT16_C(  7138),
                            INT16_C(-26251), INT16_C(-26301), INT16_C(-11538), INT16_C(  7661),
                            INT16_C( 25835), INT16_C( -1591), INT16_C(-31336), INT16_C(-13623)),
      easysimd_mm512_set_epi16(INT16_C( -3100), INT16_C(-21068), INT16_C( 20058), INT16_C(-14631),
                            INT16_C(-16628), INT16_C(  1662), INT16_C( 26797), INT16_C(  4095),
                            INT16_C( 10328), INT16_C( -2602), INT16_C( 29484), INT16_C(-27939),
                            INT16_C( 10492), INT16_C(-22955), INT16_C(  1055), INT16_C(   -22),
                            INT16_C( 28704), INT16_C(-21257), INT16_C(  4948), INT16_C( 32515),
                            INT16_C(-10322), INT16_C( -9220), INT16_C(-23650), INT16_C( 14894),
                            INT16_C(-26251), INT16_C(  7904), INT16_C(-11538), INT16_C(  7661),
                            INT16_C( 25835), INT16_C(-24798), INT16_C(-10273), INT16_C(-18470)) },
    { easysimd_mm512_set_epi16(INT16_C( -8164), INT16_C(-26845), INT16_C( 11124), INT16_C(  8752),
                            INT16_C( 22766), INT16_C(  8670), INT16_C( 20153), INT16_C( 18240),
                            INT16_C(  9917), INT16_C( -9695), INT16_C( 13965), INT16_C( 22461),
                            INT16_C(-14283), INT16_C(-28547), INT16_C( -3283), INT16_C( 28423),
                            INT16_C( -7094), INT16_C(-23805), INT16_C(-29561), INT16_C( -8833),
                            INT16_C( 19973), INT16_C(  4641), INT16_C( 26375), INT16_C(-24343),
                            INT16_C(-25797), INT16_C( 10099), INT16_C( 15606), INT16_C( -3388),
                            INT16_C( 27200), INT16_C( 17184), INT16_C( -8305), INT16_C( -2842)),
      UINT32_C(3498958446),
      easysimd_mm512_set_epi16(INT16_C( -8480), INT16_C( 28422), INT16_C(-27516), INT16_C( 21347),
                            INT16_C(-25796), INT16_C(-16858), INT16_C( 12539), INT16_C(-24081),
                            INT16_C( 21534), INT16_C(-24785), INT16_C( 27018), INT16_C(  5065),
                            INT16_C(-18143), INT16_C(  8109), INT16_C(-17219), INT16_C( 31482),
                            INT16_C(  9138), INT16_C( 22982), INT16_C(-21234), INT16_C( 25459),
                            INT16_C(  6589), INT16_C(-13007), INT16_C( 15857), INT16_C(-20120),
                            INT16_C( -7568), INT16_C(-12198), INT16_C(-11606), INT16_C( 12227),
                            INT16_C(-14277), INT16_C( -5440), INT16_C( 23811), INT16_C( 16734)),
      easysimd_mm512_set_epi16(INT16_C( -8480), INT16_C( 28422), INT16_C( 11124), INT16_C( 21347),
                            INT16_C( 22766), INT16_C(  8670), INT16_C( 20153), INT16_C( 18240),
                            INT16_C( 21534), INT16_C( -9695), INT16_C( 13965), INT16_C( 22461),
                            INT16_C(-18143), INT16_C(  8109), INT16_C( -3283), INT16_C( 31482),
                            INT16_C(  9138), INT16_C( 22982), INT16_C(-29561), INT16_C( 25459),
                            INT16_C(  6589), INT16_C(-13007), INT16_C( 15857), INT16_C(-24343),
                            INT16_C(-25797), INT16_C(-12198), INT16_C(-11606), INT16_C( -3388),
                            INT16_C(-14277), INT16_C( -5440), INT16_C( 23811), INT16_C( -2842)) },
    { easysimd_mm512_set_epi16(INT16_C(-10740), INT16_C(-19800), INT16_C( 23089), INT16_C( 21852),
                            INT16_C( 15397), INT16_C(-10864), INT16_C(  6811), INT16_C(  1049),
                            INT16_C(-27986), INT16_C(-13885), INT16_C(-16896), INT16_C(  2159),
                            INT16_C( 21619), INT16_C(-26860), INT16_C(-26036), INT16_C(  8638),
                            INT16_C( -6244), INT16_C( 12305), INT16_C( 12521), INT16_C(-23200),
                            INT16_C( 16405), INT16_C(  1911), INT16_C(-19978), INT16_C(-24716),
                            INT16_C( 18780), INT16_C(-19576), INT16_C( 23239), INT16_C(  3968),
                            INT16_C( 10340), INT16_C(-18924), INT16_C(-27656), INT16_C( 29459)),
      UINT32_C(1078376780),
      easysimd_mm512_set_epi16(INT16_C(   273), INT16_C( -1720), INT16_C( 22076), INT16_C( -2052),
                            INT16_C(-17942), INT16_C( -7577), INT16_C(-30883), INT16_C(-19493),
                            INT16_C(-19679), INT16_C( -1198), INT16_C( -2289), INT16_C(  6912),
                            INT16_C(-20982), INT16_C(-18030), INT16_C( 27608), INT16_C(  2367),
                            INT16_C(  1167), INT16_C(-16688), INT16_C(-14772), INT16_C(-28473),
                            INT16_C(-30638), INT16_C(-20143), INT16_C( 18762), INT16_C( 11938),
                            INT16_C(  3849), INT16_C( 10905), INT16_C( 14089), INT16_C(-29438),
                            INT16_C( -8204), INT16_C(-31577), INT16_C( -4765), INT16_C(  1792)),
      easysimd_mm512_set_epi16(INT16_C(-10740), INT16_C( -1720), INT16_C( 23089), INT16_C( 21852),
                            INT16_C( 15397), INT16_C(-10864), INT16_C(  6811), INT16_C(  1049),
                            INT16_C(-27986), INT16_C( -1198), INT16_C(-16896), INT16_C(  2159),
                            INT16_C( 21619), INT16_C(-18030), INT16_C( 27608), INT16_C(  8638),
                            INT16_C(  1167), INT16_C( 12305), INT16_C(-14772), INT16_C(-28473),
                            INT16_C(-30638), INT16_C(  1911), INT16_C(-19978), INT16_C( 11938),
                            INT16_C( 18780), INT16_C( 10905), INT16_C( 23239), INT16_C(  3968),
                            INT16_C( -8204), INT16_C(-31577), INT16_C(-27656), INT16_C( 29459)) },
    { easysimd_mm512_set_epi16(INT16_C( 22307), INT16_C(-11389), INT16_C(  9226), INT16_C(  7897),
                            INT16_C( 32155), INT16_C(  2611), INT16_C( 11978), INT16_C(  5179),
                            INT16_C(-24755), INT16_C(-19543), INT16_C(-15643), INT16_C( -2365),
                            INT16_C(-27002), INT16_C(  7884), INT16_C(-20138), INT16_C( -3743),
                            INT16_C(-12844), INT16_C(-25331), INT16_C( -7592), INT16_C( 24295),
                            INT16_C(-17679), INT16_C( -9896), INT16_C( -1721), INT16_C(  1797),
                            INT16_C(-21616), INT16_C(  9515), INT16_C( 22554), INT16_C(  6233),
                            INT16_C( 21958), INT16_C( -6794), INT16_C(-15370), INT16_C( 18181)),
      UINT32_C(2433004361),
      easysimd_mm512_set_epi16(INT16_C(-21831), INT16_C( -7695), INT16_C( 23301), INT16_C(  6159),
                            INT16_C(  1766), INT16_C( 18873), INT16_C( 26864), INT16_C(-27621),
                            INT16_C( -8001), INT16_C( -8493), INT16_C(-29763), INT16_C(  9106),
                            INT16_C(  1393), INT16_C(-12504), INT16_C(  9424), INT16_C( 15142),
                            INT16_C(  3894), INT16_C( -9649), INT16_C( -4354), INT16_C(-12373),
                            INT16_C(-13406), INT16_C( 12423), INT16_C( 26727), INT16_C(-11193),
                            INT16_C( 15482), INT16_C(-21938), INT16_C(-28148), INT16_C( -1462),
                            INT16_C( -2537), INT16_C(  7119), INT16_C( -3240), INT16_C(-31830)),
      easysimd_mm512_set_epi16(INT16_C(-21831), INT16_C(-11389), INT16_C(  9226), INT16_C(  6159),
                            INT16_C( 32155), INT16_C(  2611), INT16_C( 11978), INT16_C(-27621),
                            INT16_C(-24755), INT16_C(-19543), INT16_C(-15643), INT16_C( -2365),
                            INT16_C(-27002), INT16_C(-12504), INT16_C(-20138), INT16_C( -3743),
                            INT16_C(  3894), INT16_C(-25331), INT16_C( -4354), INT16_C(-12373),
                            INT16_C(-17679), INT16_C( -9896), INT16_C( 26727), INT16_C(-11193),
                            INT16_C(-21616), INT16_C(-21938), INT16_C( 22554), INT16_C(  6233),
                            INT16_C( -2537), INT16_C( -6794), INT16_C(-15370), INT16_C(-31830)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mov_epi16(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mov_epi16");
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mov_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C(-1748841636), INT32_C(  600342911), INT32_C( 1346502861), INT32_C(-1119296012),
                            INT32_C(  542725165), INT32_C(  811581991), INT32_C(-1753809264), INT32_C(-2095888677),
                            INT32_C(   21844621), INT32_C( -668859652), INT32_C(  304402382), INT32_C( 1173008100),
                            INT32_C( -727866068), INT32_C(     599757), INT32_C( 1586862788), INT32_C(-1998308703)),
      UINT16_C(60467),
      easysimd_mm512_set_epi32(INT32_C(  646279344), INT32_C(-1381858570), INT32_C( 1528846110), INT32_C( -686931066),
                            INT32_C( 1116827472), INT32_C(-1909789352), INT32_C( 1965198777), INT32_C(  743332806),
                            INT32_C( -485827488), INT32_C(-1346955971), INT32_C(  625279893), INT32_C(  390045432),
                            INT32_C( 1242706406), INT32_C( -451702195), INT32_C( -797642518), INT32_C( 1682577743)),
      easysimd_mm512_set_epi32(INT32_C(  646279344), INT32_C(-1381858570), INT32_C( 1528846110), INT32_C(-1119296012),
                            INT32_C( 1116827472), INT32_C(-1909789352), INT32_C(-1753809264), INT32_C(-2095888677),
                            INT32_C(   21844621), INT32_C( -668859652), INT32_C(  625279893), INT32_C(  390045432),
                            INT32_C( -727866068), INT32_C(     599757), INT32_C( -797642518), INT32_C( 1682577743)) },
    { easysimd_mm512_set_epi32(INT32_C(  478337815), INT32_C( -537978403), INT32_C(-1351889488), INT32_C( 1090048308),
                            INT32_C( 1261235095), INT32_C(-1289893124), INT32_C( -387446550), INT32_C(-1938729505),
                            INT32_C(-1389958008), INT32_C( 1730413171), INT32_C( 1087827160), INT32_C( 1023459790),
                            INT32_C(-1481706049), INT32_C(  911835427), INT32_C(-1124770978), INT32_C( 1212776438)),
      UINT16_C(65510),
      easysimd_mm512_set_epi32(INT32_C( -876793269), INT32_C(  354193822), INT32_C(-1942817736), INT32_C(   48337666),
                            INT32_C(-1069034730), INT32_C( -258187388), INT32_C(-1735480646), INT32_C( 1239662333),
                            INT32_C(-1087348321), INT32_C(  777072035), INT32_C( -223191004), INT32_C( -671373205),
                            INT32_C( -333775053), INT32_C( 1946636837), INT32_C(  875386084), INT32_C(   41135181)),
      easysimd_mm512_set_epi32(INT32_C( -876793269), INT32_C(  354193822), INT32_C(-1942817736), INT32_C(   48337666),
                            INT32_C(-1069034730), INT32_C( -258187388), INT32_C(-1735480646), INT32_C( 1239662333),
                            INT32_C(-1087348321), INT32_C(  777072035), INT32_C( -223191004), INT32_C( 1023459790),
                            INT32_C(-1481706049), INT32_C( 1946636837), INT32_C(  875386084), INT32_C( 1212776438)) },
    { easysimd_mm512_set_epi32(INT32_C(  739047763), INT32_C( 1498945773), INT32_C( 1776295699), INT32_C( 1298376143),
                            INT32_C(-1413206606), INT32_C(-1101195004), INT32_C( 1096357047), INT32_C( 1201409099),
                            INT32_C(-1184934080), INT32_C(-1142871559), INT32_C(-1331799428), INT32_C( 2127606263),
                            INT32_C( 1810587941), INT32_C(-1568035201), INT32_C(-1514801640), INT32_C( 1754146272)),
      UINT16_C(17782),
      easysimd_mm512_set_epi32(INT32_C(-1637684250), INT32_C( 1624419961), INT32_C(-1721698305), INT32_C( 1216991175),
                            INT32_C( 1086797293), INT32_C( -544515074), INT32_C(-1866991972), INT32_C( 1497966040),
                            INT32_C(  183681068), INT32_C( 1846911046), INT32_C(  396433769), INT32_C( 1567943719),
                            INT32_C( 1544652060), INT32_C( 1999507462), INT32_C( -389522003), INT32_C(  660842170)),
      easysimd_mm512_set_epi32(INT32_C(  739047763), INT32_C( 1624419961), INT32_C( 1776295699), INT32_C( 1298376143),
                            INT32_C(-1413206606), INT32_C( -544515074), INT32_C( 1096357047), INT32_C( 1497966040),
                            INT32_C(-1184934080), INT32_C( 1846911046), INT32_C(  396433769), INT32_C( 1567943719),
                            INT32_C( 1810587941), INT32_C( 1999507462), INT32_C( -389522003), INT32_C( 1754146272)) },
    { easysimd_mm512_set_epi32(INT32_C(-1787060903), INT32_C( 1591528199), INT32_C( 1360730903), INT32_C( -392663993),
                            INT32_C( 1833403381), INT32_C(  667948495), INT32_C(-1351186880), INT32_C(-1869951013),
                            INT32_C(-1764668962), INT32_C( 1727501907), INT32_C(-1699520398), INT32_C(-2078068732),
                            INT32_C(-1191187391), INT32_C(  809086335), INT32_C( -915516374), INT32_C( 2044786719)),
      UINT16_C(19153),
      easysimd_mm512_set_epi32(INT32_C(-1124863619), INT32_C( -733840886), INT32_C(  225375619), INT32_C( 2033345748),
                            INT32_C(   62836182), INT32_C(-1797131359), INT32_C( -791707937), INT32_C(-1161020437),
                            INT32_C( 1933148289), INT32_C(-1354039663), INT32_C(  533923030), INT32_C(  457770626),
                            INT32_C(-2130199261), INT32_C( -201626469), INT32_C( 1603256738), INT32_C(  385840376)),
      easysimd_mm512_set_epi32(INT32_C(-1787060903), INT32_C( -733840886), INT32_C( 1360730903), INT32_C( -392663993),
                            INT32_C(   62836182), INT32_C(  667948495), INT32_C( -791707937), INT32_C(-1869951013),
                            INT32_C( 1933148289), INT32_C(-1354039663), INT32_C(-1699520398), INT32_C(  457770626),
                            INT32_C(-1191187391), INT32_C(  809086335), INT32_C( -915516374), INT32_C(  385840376)) },
    { easysimd_mm512_set_epi32(INT32_C(-1844996035), INT32_C( -483918772), INT32_C(-1530619556), INT32_C( -447486042),
                            INT32_C( -153016391), INT32_C( 1772993408), INT32_C(-1557466731), INT32_C( 1884729185),
                            INT32_C(-1170473640), INT32_C( -231873321), INT32_C( 1063107119), INT32_C( 1409583343),
                            INT32_C(  131479252), INT32_C(-1464445699), INT32_C(-1859507666), INT32_C( 1142318206)),
      UINT16_C(39686),
      easysimd_mm512_set_epi32(INT32_C(-1710909147), INT32_C( 1655743921), INT32_C(-1520991125), INT32_C(-1200934587),
                            INT32_C( -721899112), INT32_C( 1216881740), INT32_C( -481496777), INT32_C( -893026644),
                            INT32_C(-2035526652), INT32_C( -294630589), INT32_C(-1446210787), INT32_C( -547573265),
                            INT32_C( 1911285838), INT32_C(-1067024301), INT32_C(-1545394687), INT32_C( 1507767747)),
      easysimd_mm512_set_epi32(INT32_C(-1710909147), INT32_C( -483918772), INT32_C(-1530619556), INT32_C(-1200934587),
                            INT32_C( -721899112), INT32_C( 1772993408), INT32_C( -481496777), INT32_C( -893026644),
                            INT32_C(-1170473640), INT32_C( -231873321), INT32_C( 1063107119), INT32_C( 1409583343),
                            INT32_C(  131479252), INT32_C(-1067024301), INT32_C(-1545394687), INT32_C( 1142318206)) },
    { easysimd_mm512_set_epi32(INT32_C( 2003854537), INT32_C(  316518418), INT32_C(-2128378506), INT32_C( -814023178),
                            INT32_C( 2134095257), INT32_C( -273917753), INT32_C(  269941696), INT32_C(-1761573676),
                            INT32_C( -504711162), INT32_C( 1086943646), INT32_C( -304633534), INT32_C( -905159738),
                            INT32_C(-1025692186), INT32_C(-2082862175), INT32_C(-1626855678), INT32_C(-1231176910)),
      UINT16_C(13329),
      easysimd_mm512_set_epi32(INT32_C(  838273890), INT32_C( 1209103370), INT32_C(  947433971), INT32_C(   91213725),
                            INT32_C(  749577280), INT32_C(  157602752), INT32_C( 2125537515), INT32_C( -782796801),
                            INT32_C( -120430288), INT32_C( -810448185), INT32_C( -659512402), INT32_C(  419195007),
                            INT32_C( -830103963), INT32_C( -756234442), INT32_C(  376291679), INT32_C( -610488282)),
      easysimd_mm512_set_epi32(INT32_C( 2003854537), INT32_C(  316518418), INT32_C(  947433971), INT32_C(   91213725),
                            INT32_C( 2134095257), INT32_C(  157602752), INT32_C(  269941696), INT32_C(-1761573676),
                            INT32_C( -504711162), INT32_C( 1086943646), INT32_C( -304633534), INT32_C(  419195007),
                            INT32_C(-1025692186), INT32_C(-2082862175), INT32_C(-1626855678), INT32_C( -610488282)) },
    { easysimd_mm512_set_epi32(INT32_C( -974755823), INT32_C(  -98121742), INT32_C( 1561555936), INT32_C(-1281058782),
                            INT32_C(-2008886211), INT32_C( 1568326299), INT32_C( 1232828554), INT32_C(  127919997),
                            INT32_C( 1015818460), INT32_C( -681833659), INT32_C(  340145717), INT32_C( 1048452961),
                            INT32_C(  749206991), INT32_C( 1290937767), INT32_C(-1150545818), INT32_C(  -48881570)),
      UINT16_C(55435),
      easysimd_mm512_set_epi32(INT32_C( 1177945769), INT32_C(-1878447950), INT32_C( -271391312), INT32_C(-2014500164),
                            INT32_C(-2080120479), INT32_C( 1195569010), INT32_C(-1583493780), INT32_C( 1466155853),
                            INT32_C( -735473338), INT32_C( 1922464741), INT32_C( -224185100), INT32_C( -929578437),
                            INT32_C(  831459587), INT32_C(-1105963780), INT32_C(-1360707796), INT32_C( -211781248)),
      easysimd_mm512_set_epi32(INT32_C( 1177945769), INT32_C(-1878447950), INT32_C( 1561555936), INT32_C(-2014500164),
                            INT32_C(-2080120479), INT32_C( 1568326299), INT32_C( 1232828554), INT32_C(  127919997),
                            INT32_C( -735473338), INT32_C( -681833659), INT32_C(  340145717), INT32_C( 1048452961),
                            INT32_C(  831459587), INT32_C( 1290937767), INT32_C(-1360707796), INT32_C( -211781248)) },
    { easysimd_mm512_set_epi32(INT32_C( 1583932216), INT32_C(-1528139164), INT32_C(  665399981), INT32_C(  718332631),
                            INT32_C( -984331868), INT32_C(-1317077859), INT32_C(-1440392153), INT32_C(-1978382578),
                            INT32_C(  828185710), INT32_C( 1905160582), INT32_C(  120938992), INT32_C( 1613459128),
                            INT32_C( -812252493), INT32_C(-1503952372), INT32_C(  231875300), INT32_C( -885498028)),
      UINT16_C(45743),
      easysimd_mm512_set_epi32(INT32_C(-1033540577), INT32_C( -995705628), INT32_C(-2098565905), INT32_C(-1609941379),
                            INT32_C(  451122481), INT32_C(  898911803), INT32_C( -918933314), INT32_C( 1301755496),
                            INT32_C(  654535343), INT32_C( 1915381036), INT32_C( -595265918), INT32_C( -204141630),
                            INT32_C(-1824782722), INT32_C(-1457642917), INT32_C(-1358921472), INT32_C( 1013008616)),
      easysimd_mm512_set_epi32(INT32_C(-1033540577), INT32_C(-1528139164), INT32_C(-2098565905), INT32_C(-1609941379),
                            INT32_C( -984331868), INT32_C(-1317077859), INT32_C( -918933314), INT32_C(-1978382578),
                            INT32_C(  654535343), INT32_C( 1905160582), INT32_C( -595265918), INT32_C( 1613459128),
                            INT32_C(-1824782722), INT32_C(-1457642917), INT32_C(-1358921472), INT32_C( 1013008616)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mov_epi32(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mov_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mov_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 8729250599109288206), INT64_C(  925123000700261284),
                            INT64_C( -996462675499144949), INT64_C(-5486361937319788764),
                            INT64_C(-1619246833501834651), INT64_C(-1914665916415518359),
                            INT64_C( 4596079613709719053), INT64_C(-1669293344454375632)),
      UINT8_C(136),
      easysimd_mm512_set_epi64(INT64_C(-2718786087636304341), INT64_C( 6271007050593066413),
                            INT64_C( 7325428114350079264), INT64_C( 8373606416957659495),
                            INT64_C( 8585702140748752091), INT64_C(-6106352141912550191),
                            INT64_C(-7415158757307660945), INT64_C(-4168322686232168747)),
      easysimd_mm512_set_epi64(INT64_C(-2718786087636304341), INT64_C(  925123000700261284),
                            INT64_C( -996462675499144949), INT64_C(-5486361937319788764),
                            INT64_C( 8585702140748752091), INT64_C(-1914665916415518359),
                            INT64_C( 4596079613709719053), INT64_C(-1669293344454375632)) },
    { easysimd_mm512_set_epi64(INT64_C( 8240025841211248490), INT64_C( 2437990159450284908),
                            INT64_C( 2201815834941113848), INT64_C( 7879550161691977002),
                            INT64_C( 3825487759520775297), INT64_C( 6674403996216424931),
                            INT64_C(-5802137669857725171), INT64_C( 5686996017309487110)),
      UINT8_C(227),
      easysimd_mm512_set_epi64(INT64_C(  120730317606372397), INT64_C(-1410770079656234556),
                            INT64_C( 4532617684378198659), INT64_C( 9004023903916376139),
                            INT64_C( 7206885247739448460), INT64_C(-6411218032719574536),
                            INT64_C( -962636034832057562), INT64_C(-6211267245753502041)),
      easysimd_mm512_set_epi64(INT64_C(  120730317606372397), INT64_C(-1410770079656234556),
                            INT64_C( 4532617684378198659), INT64_C( 7879550161691977002),
                            INT64_C( 3825487759520775297), INT64_C( 6674403996216424931),
                            INT64_C( -962636034832057562), INT64_C(-6211267245753502041)) },
    { easysimd_mm512_set_epi64(INT64_C( 4674722797399239366), INT64_C( 2000178744548395677),
                            INT64_C(-3230169679464817239), INT64_C( 6675942378016655726),
                            INT64_C(-4074632284771109640), INT64_C(-1969073951075376054),
                            INT64_C(-7309602967246577272), INT64_C( 6746883208360816464)),
      UINT8_C(189),
      easysimd_mm512_set_epi64(INT64_C( 7111791735729821232), INT64_C(-6377956101145598745),
                            INT64_C(-4955467359912007508), INT64_C( -340840922408165844),
                            INT64_C( 3280430708356940081), INT64_C(  400669322893233577),
                            INT64_C( 6742772793155919855), INT64_C(-1365845768056837484)),
      easysimd_mm512_set_epi64(INT64_C( 7111791735729821232), INT64_C( 2000178744548395677),
                            INT64_C(-4955467359912007508), INT64_C( -340840922408165844),
                            INT64_C( 3280430708356940081), INT64_C(  400669322893233577),
                            INT64_C(-7309602967246577272), INT64_C(-1365845768056837484)) },
    { easysimd_mm512_set_epi64(INT64_C(-5185665192936807952), INT64_C( 2873887117219468065),
                            INT64_C(  944218707053685182), INT64_C(-6471325153303919649),
                            INT64_C(-1551809186210791512), INT64_C( 8676397618641344048),
                            INT64_C(-1480083839359048471), INT64_C(-2573286236881012052)),
      UINT8_C(135),
      easysimd_mm512_set_epi64(INT64_C( 4851071406626175825), INT64_C( 2006733877612279017),
                            INT64_C( 9148059701805005067), INT64_C( 3484083856858518164),
                            INT64_C( -542612751996632572), INT64_C( 6154040976669554118),
                            INT64_C( 4310055852136225460), INT64_C( 6666177398356729891)),
      easysimd_mm512_set_epi64(INT64_C( 4851071406626175825), INT64_C( 2873887117219468065),
                            INT64_C(  944218707053685182), INT64_C(-6471325153303919649),
                            INT64_C(-1551809186210791512), INT64_C( 6154040976669554118),
                            INT64_C( 4310055852136225460), INT64_C( 6666177398356729891)) },
    { easysimd_mm512_set_epi64(INT64_C(-6362423492218583699), INT64_C( 4052676248150053459),
                            INT64_C(-1785632160509127109), INT64_C( 4504790352522402260),
                            INT64_C(  214305831990150369), INT64_C( 4122674741194642780),
                            INT64_C(-9061446978520477770), INT64_C( -925260945734331795)),
      UINT8_C( 88),
      easysimd_mm512_set_epi64(INT64_C( 7816755513219693536), INT64_C(-8078701368125426812),
                            INT64_C( 5999276564615449517), INT64_C(-3747208296317683129),
                            INT64_C(-3767121149493822975), INT64_C( 3269862772677933078),
                            INT64_C(-1274534447611012205), INT64_C(  367478185734650139)),
      easysimd_mm512_set_epi64(INT64_C(-6362423492218583699), INT64_C(-8078701368125426812),
                            INT64_C(-1785632160509127109), INT64_C(-3747208296317683129),
                            INT64_C(-3767121149493822975), INT64_C( 4122674741194642780),
                            INT64_C(-9061446978520477770), INT64_C( -925260945734331795)) },
    { easysimd_mm512_set_epi64(INT64_C(-6749425177074609965), INT64_C( 8453995530571484051),
                            INT64_C(-7619559937003101591), INT64_C( 3005943923235484348),
                            INT64_C( 4327678115781969631), INT64_C( 5990841649027118513),
                            INT64_C(-1241607161778990291), INT64_C(  -91855491071654622)),
      UINT8_C( 22),
      easysimd_mm512_set_epi64(INT64_C( 4461859928182214174), INT64_C(-5186049742858346871),
                            INT64_C( -636993447067685727), INT64_C( 8339698509359201789),
                            INT64_C( 4598711567911914631), INT64_C( 7428996315725576873),
                            INT64_C( 6513452752711502515), INT64_C(-6603414145042292282)),
      easysimd_mm512_set_epi64(INT64_C(-6749425177074609965), INT64_C( 8453995530571484051),
                            INT64_C(-7619559937003101591), INT64_C( 8339698509359201789),
                            INT64_C( 4327678115781969631), INT64_C( 7428996315725576873),
                            INT64_C( 6513452752711502515), INT64_C(  -91855491071654622)) },
    { easysimd_mm512_set_epi64(INT64_C(-7023609179598013523), INT64_C(-2166290313032224989),
                            INT64_C(-1367963225958164233), INT64_C(-9082538196892642083),
                            INT64_C(-7482977792619995502), INT64_C( 4800709110944492165),
                            INT64_C( 3082355013095664677), INT64_C(-4286500001112695437)),
      UINT8_C( 42),
      easysimd_mm512_set_epi64(INT64_C(  522664068472938939), INT64_C(-5622535385140832229),
                            INT64_C( 4829749372798053845), INT64_C( -330958976268778895),
                            INT64_C(-2657198631452288613), INT64_C(-3805394135151266272),
                            INT64_C( 4429043998616724751), INT64_C( 4131511442627175760)),
      easysimd_mm512_set_epi64(INT64_C(-7023609179598013523), INT64_C(-2166290313032224989),
                            INT64_C( 4829749372798053845), INT64_C(-9082538196892642083),
                            INT64_C(-2657198631452288613), INT64_C( 4800709110944492165),
                            INT64_C( 4429043998616724751), INT64_C(-4286500001112695437)) },
    { easysimd_mm512_set_epi64(INT64_C(-5567656428388000347), INT64_C( -971128712423557311),
                            INT64_C( 3761317547504069574), INT64_C( 6096071933426825544),
                            INT64_C( 3108166743366703612), INT64_C(-3435283790563075237),
                            INT64_C( 3598996591046999900), INT64_C( 2520744130071328064)),
      UINT8_C( 13),
      easysimd_mm512_set_epi64(INT64_C( -976144998301952820), INT64_C( 5304141922221069696),
                            INT64_C( 1153833608356774417), INT64_C(-5431879705444140176),
                            INT64_C(-4200442870371425874), INT64_C( 9118970466689378415),
                            INT64_C( 7182201605874776129), INT64_C( 6344954152679193639)),
      easysimd_mm512_set_epi64(INT64_C(-5567656428388000347), INT64_C( -971128712423557311),
                            INT64_C( 3761317547504069574), INT64_C( 6096071933426825544),
                            INT64_C(-4200442870371425874), INT64_C( 9118970466689378415),
                            INT64_C( 3598996591046999900), INT64_C( 6344954152679193639)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mov_epi64(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mov_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mov_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -997.43), EASYSIMD_FLOAT64_C(  -24.75),
                         EASYSIMD_FLOAT64_C(  811.92), EASYSIMD_FLOAT64_C(  716.01),
                         EASYSIMD_FLOAT64_C( -286.81), EASYSIMD_FLOAT64_C(  360.81),
                         EASYSIMD_FLOAT64_C( -618.94), EASYSIMD_FLOAT64_C(  103.41)),
      UINT8_C( 17),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  779.73), EASYSIMD_FLOAT64_C(  -71.34),
                         EASYSIMD_FLOAT64_C(   74.67), EASYSIMD_FLOAT64_C(  569.44),
                         EASYSIMD_FLOAT64_C(  765.94), EASYSIMD_FLOAT64_C(  114.94),
                         EASYSIMD_FLOAT64_C(   85.69), EASYSIMD_FLOAT64_C(  982.40)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -997.43), EASYSIMD_FLOAT64_C(  -24.75),
                         EASYSIMD_FLOAT64_C(  811.92), EASYSIMD_FLOAT64_C(  569.44),
                         EASYSIMD_FLOAT64_C( -286.81), EASYSIMD_FLOAT64_C(  360.81),
                         EASYSIMD_FLOAT64_C( -618.94), EASYSIMD_FLOAT64_C(  982.40)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -989.28), EASYSIMD_FLOAT64_C( -906.64),
                         EASYSIMD_FLOAT64_C( -211.36), EASYSIMD_FLOAT64_C( -108.84),
                         EASYSIMD_FLOAT64_C(  211.05), EASYSIMD_FLOAT64_C( -602.13),
                         EASYSIMD_FLOAT64_C(   19.95), EASYSIMD_FLOAT64_C( -745.56)),
      UINT8_C(115),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -995.20), EASYSIMD_FLOAT64_C(   66.82),
                         EASYSIMD_FLOAT64_C(  747.55), EASYSIMD_FLOAT64_C(  590.56),
                         EASYSIMD_FLOAT64_C(  522.53), EASYSIMD_FLOAT64_C(  340.37),
                         EASYSIMD_FLOAT64_C( -323.43), EASYSIMD_FLOAT64_C( -598.33)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -989.28), EASYSIMD_FLOAT64_C(   66.82),
                         EASYSIMD_FLOAT64_C(  747.55), EASYSIMD_FLOAT64_C(  590.56),
                         EASYSIMD_FLOAT64_C(  211.05), EASYSIMD_FLOAT64_C( -602.13),
                         EASYSIMD_FLOAT64_C( -323.43), EASYSIMD_FLOAT64_C( -598.33)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  435.94), EASYSIMD_FLOAT64_C( -117.09),
                         EASYSIMD_FLOAT64_C( -343.63), EASYSIMD_FLOAT64_C( -686.94),
                         EASYSIMD_FLOAT64_C( -632.13), EASYSIMD_FLOAT64_C(  520.11),
                         EASYSIMD_FLOAT64_C(  584.62), EASYSIMD_FLOAT64_C(  269.90)),
      UINT8_C(142),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -307.53), EASYSIMD_FLOAT64_C(  533.35),
                         EASYSIMD_FLOAT64_C( -283.32), EASYSIMD_FLOAT64_C(  860.26),
                         EASYSIMD_FLOAT64_C( -955.05), EASYSIMD_FLOAT64_C( -767.10),
                         EASYSIMD_FLOAT64_C( -553.49), EASYSIMD_FLOAT64_C(  540.17)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -307.53), EASYSIMD_FLOAT64_C( -117.09),
                         EASYSIMD_FLOAT64_C( -343.63), EASYSIMD_FLOAT64_C( -686.94),
                         EASYSIMD_FLOAT64_C( -955.05), EASYSIMD_FLOAT64_C( -767.10),
                         EASYSIMD_FLOAT64_C( -553.49), EASYSIMD_FLOAT64_C(  269.90)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  591.80), EASYSIMD_FLOAT64_C( -733.65),
                         EASYSIMD_FLOAT64_C(  371.96), EASYSIMD_FLOAT64_C( -998.26),
                         EASYSIMD_FLOAT64_C(   61.01), EASYSIMD_FLOAT64_C( -918.19),
                         EASYSIMD_FLOAT64_C( -797.48), EASYSIMD_FLOAT64_C(   81.07)),
      UINT8_C(155),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  378.17), EASYSIMD_FLOAT64_C(  574.36),
                         EASYSIMD_FLOAT64_C(  687.12), EASYSIMD_FLOAT64_C( -618.22),
                         EASYSIMD_FLOAT64_C(  388.77), EASYSIMD_FLOAT64_C( -731.92),
                         EASYSIMD_FLOAT64_C(  958.30), EASYSIMD_FLOAT64_C(   51.30)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  378.17), EASYSIMD_FLOAT64_C( -733.65),
                         EASYSIMD_FLOAT64_C(  371.96), EASYSIMD_FLOAT64_C( -618.22),
                         EASYSIMD_FLOAT64_C(  388.77), EASYSIMD_FLOAT64_C( -918.19),
                         EASYSIMD_FLOAT64_C(  958.30), EASYSIMD_FLOAT64_C(   51.30)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  721.16), EASYSIMD_FLOAT64_C(   21.28),
                         EASYSIMD_FLOAT64_C( -269.14), EASYSIMD_FLOAT64_C( -241.41),
                         EASYSIMD_FLOAT64_C( -307.10), EASYSIMD_FLOAT64_C(   78.73),
                         EASYSIMD_FLOAT64_C(  336.91), EASYSIMD_FLOAT64_C( -793.36)),
      UINT8_C(174),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  944.42), EASYSIMD_FLOAT64_C(  986.58),
                         EASYSIMD_FLOAT64_C( -765.43), EASYSIMD_FLOAT64_C(  392.41),
                         EASYSIMD_FLOAT64_C(  229.44), EASYSIMD_FLOAT64_C(   52.87),
                         EASYSIMD_FLOAT64_C( -238.79), EASYSIMD_FLOAT64_C(  440.21)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  944.42), EASYSIMD_FLOAT64_C(   21.28),
                         EASYSIMD_FLOAT64_C( -765.43), EASYSIMD_FLOAT64_C( -241.41),
                         EASYSIMD_FLOAT64_C(  229.44), EASYSIMD_FLOAT64_C(   52.87),
                         EASYSIMD_FLOAT64_C( -238.79), EASYSIMD_FLOAT64_C( -793.36)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  456.66), EASYSIMD_FLOAT64_C( -366.58),
                         EASYSIMD_FLOAT64_C(  715.22), EASYSIMD_FLOAT64_C(  -16.79),
                         EASYSIMD_FLOAT64_C( -320.68), EASYSIMD_FLOAT64_C(  273.81),
                         EASYSIMD_FLOAT64_C( -581.56), EASYSIMD_FLOAT64_C(  277.97)),
      UINT8_C(205),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.81), EASYSIMD_FLOAT64_C(  801.66),
                         EASYSIMD_FLOAT64_C(  310.16), EASYSIMD_FLOAT64_C(  634.68),
                         EASYSIMD_FLOAT64_C( -889.89), EASYSIMD_FLOAT64_C( -998.37),
                         EASYSIMD_FLOAT64_C( -493.27), EASYSIMD_FLOAT64_C(  120.40)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    2.81), EASYSIMD_FLOAT64_C(  801.66),
                         EASYSIMD_FLOAT64_C(  715.22), EASYSIMD_FLOAT64_C(  -16.79),
                         EASYSIMD_FLOAT64_C( -889.89), EASYSIMD_FLOAT64_C( -998.37),
                         EASYSIMD_FLOAT64_C( -581.56), EASYSIMD_FLOAT64_C(  120.40)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   44.26), EASYSIMD_FLOAT64_C(  891.25),
                         EASYSIMD_FLOAT64_C(  290.62), EASYSIMD_FLOAT64_C(  -70.18),
                         EASYSIMD_FLOAT64_C(   -3.54), EASYSIMD_FLOAT64_C(  783.54),
                         EASYSIMD_FLOAT64_C( -718.82), EASYSIMD_FLOAT64_C(  922.75)),
      UINT8_C( 72),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -286.94), EASYSIMD_FLOAT64_C( -573.68),
                         EASYSIMD_FLOAT64_C( -931.52), EASYSIMD_FLOAT64_C(  249.22),
                         EASYSIMD_FLOAT64_C(  735.88), EASYSIMD_FLOAT64_C(  653.72),
                         EASYSIMD_FLOAT64_C(  732.59), EASYSIMD_FLOAT64_C(  161.45)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   44.26), EASYSIMD_FLOAT64_C( -573.68),
                         EASYSIMD_FLOAT64_C(  290.62), EASYSIMD_FLOAT64_C(  -70.18),
                         EASYSIMD_FLOAT64_C(  735.88), EASYSIMD_FLOAT64_C(  783.54),
                         EASYSIMD_FLOAT64_C( -718.82), EASYSIMD_FLOAT64_C(  922.75)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  729.70), EASYSIMD_FLOAT64_C( -950.99),
                         EASYSIMD_FLOAT64_C(  115.61), EASYSIMD_FLOAT64_C( -132.19),
                         EASYSIMD_FLOAT64_C(  834.99), EASYSIMD_FLOAT64_C(  471.53),
                         EASYSIMD_FLOAT64_C(   54.12), EASYSIMD_FLOAT64_C(  238.73)),
      UINT8_C(209),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -345.93), EASYSIMD_FLOAT64_C(  598.65),
                         EASYSIMD_FLOAT64_C(  954.89), EASYSIMD_FLOAT64_C( -441.90),
                         EASYSIMD_FLOAT64_C(  845.52), EASYSIMD_FLOAT64_C( -659.44),
                         EASYSIMD_FLOAT64_C( -844.59), EASYSIMD_FLOAT64_C(  331.33)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -345.93), EASYSIMD_FLOAT64_C(  598.65),
                         EASYSIMD_FLOAT64_C(  115.61), EASYSIMD_FLOAT64_C( -441.90),
                         EASYSIMD_FLOAT64_C(  834.99), EASYSIMD_FLOAT64_C(  471.53),
                         EASYSIMD_FLOAT64_C(   54.12), EASYSIMD_FLOAT64_C(  331.33)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mov_pd(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mov_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mov_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -278.44), EASYSIMD_FLOAT32_C(   958.04), EASYSIMD_FLOAT32_C(  -686.18), EASYSIMD_FLOAT32_C(  -120.52),
                         EASYSIMD_FLOAT32_C(   759.91), EASYSIMD_FLOAT32_C(   470.87), EASYSIMD_FLOAT32_C(  -723.57), EASYSIMD_FLOAT32_C(   170.04),
                         EASYSIMD_FLOAT32_C(   559.73), EASYSIMD_FLOAT32_C(   984.13), EASYSIMD_FLOAT32_C(   -84.72), EASYSIMD_FLOAT32_C(  -543.95),
                         EASYSIMD_FLOAT32_C(   998.02), EASYSIMD_FLOAT32_C(  -559.31), EASYSIMD_FLOAT32_C(   134.12), EASYSIMD_FLOAT32_C(  -230.64)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -161.72), EASYSIMD_FLOAT32_C(   540.27), EASYSIMD_FLOAT32_C(  -745.55), EASYSIMD_FLOAT32_C(   623.14),
                         EASYSIMD_FLOAT32_C(  -272.95), EASYSIMD_FLOAT32_C(   176.76), EASYSIMD_FLOAT32_C(  -957.12), EASYSIMD_FLOAT32_C(  -720.97),
                         EASYSIMD_FLOAT32_C(  -491.62), EASYSIMD_FLOAT32_C(   442.72), EASYSIMD_FLOAT32_C(    94.42), EASYSIMD_FLOAT32_C(  -425.44),
                         EASYSIMD_FLOAT32_C(   378.60), EASYSIMD_FLOAT32_C(  -248.93), EASYSIMD_FLOAT32_C(   638.30), EASYSIMD_FLOAT32_C(  -857.32)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -278.44), EASYSIMD_FLOAT32_C(   958.04), EASYSIMD_FLOAT32_C(  -686.18), EASYSIMD_FLOAT32_C(  -120.52),
                         EASYSIMD_FLOAT32_C(   759.91), EASYSIMD_FLOAT32_C(   470.87), EASYSIMD_FLOAT32_C(  -723.57), EASYSIMD_FLOAT32_C(   170.04),
                         EASYSIMD_FLOAT32_C(   559.73), EASYSIMD_FLOAT32_C(   984.13), EASYSIMD_FLOAT32_C(   -84.72), EASYSIMD_FLOAT32_C(  -543.95),
                         EASYSIMD_FLOAT32_C(   998.02), EASYSIMD_FLOAT32_C(  -559.31), EASYSIMD_FLOAT32_C(   134.12), EASYSIMD_FLOAT32_C(  -230.64)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -455.21), EASYSIMD_FLOAT32_C(  -180.02), EASYSIMD_FLOAT32_C(  -110.74), EASYSIMD_FLOAT32_C(  -586.50),
                         EASYSIMD_FLOAT32_C(    -9.89), EASYSIMD_FLOAT32_C(  -597.54), EASYSIMD_FLOAT32_C(   553.79), EASYSIMD_FLOAT32_C(   611.64),
                         EASYSIMD_FLOAT32_C(   717.03), EASYSIMD_FLOAT32_C(  -381.85), EASYSIMD_FLOAT32_C(   862.32), EASYSIMD_FLOAT32_C(   302.29),
                         EASYSIMD_FLOAT32_C(   146.86), EASYSIMD_FLOAT32_C(  -693.40), EASYSIMD_FLOAT32_C(  -247.57), EASYSIMD_FLOAT32_C(  -469.49)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   842.67), EASYSIMD_FLOAT32_C(  -856.89), EASYSIMD_FLOAT32_C(  -490.76), EASYSIMD_FLOAT32_C(   922.81),
                         EASYSIMD_FLOAT32_C(   -69.36), EASYSIMD_FLOAT32_C(   380.23), EASYSIMD_FLOAT32_C(  -846.01), EASYSIMD_FLOAT32_C(  -485.23),
                         EASYSIMD_FLOAT32_C(  -171.14), EASYSIMD_FLOAT32_C(   602.88), EASYSIMD_FLOAT32_C(  -717.33), EASYSIMD_FLOAT32_C(   336.05),
                         EASYSIMD_FLOAT32_C(  -432.71), EASYSIMD_FLOAT32_C(  -881.01), EASYSIMD_FLOAT32_C(  -255.82), EASYSIMD_FLOAT32_C(   168.04)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -455.21), EASYSIMD_FLOAT32_C(  -180.02), EASYSIMD_FLOAT32_C(  -110.74), EASYSIMD_FLOAT32_C(  -586.50),
                         EASYSIMD_FLOAT32_C(    -9.89), EASYSIMD_FLOAT32_C(  -597.54), EASYSIMD_FLOAT32_C(   553.79), EASYSIMD_FLOAT32_C(   611.64),
                         EASYSIMD_FLOAT32_C(   717.03), EASYSIMD_FLOAT32_C(  -381.85), EASYSIMD_FLOAT32_C(   862.32), EASYSIMD_FLOAT32_C(   302.29),
                         EASYSIMD_FLOAT32_C(   146.86), EASYSIMD_FLOAT32_C(  -693.40), EASYSIMD_FLOAT32_C(  -247.57), EASYSIMD_FLOAT32_C(  -469.49)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -694.40), EASYSIMD_FLOAT32_C(  -404.01), EASYSIMD_FLOAT32_C(   766.51), EASYSIMD_FLOAT32_C(  -392.19),
                         EASYSIMD_FLOAT32_C(  -908.15), EASYSIMD_FLOAT32_C(  -690.12), EASYSIMD_FLOAT32_C(  -262.73), EASYSIMD_FLOAT32_C(  -353.25),
                         EASYSIMD_FLOAT32_C(  -451.03), EASYSIMD_FLOAT32_C(   -88.58), EASYSIMD_FLOAT32_C(   658.99), EASYSIMD_FLOAT32_C(  -961.05),
                         EASYSIMD_FLOAT32_C(  -743.39), EASYSIMD_FLOAT32_C(   747.85), EASYSIMD_FLOAT32_C(  -989.89), EASYSIMD_FLOAT32_C(   -48.62)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -585.79), EASYSIMD_FLOAT32_C(  -884.44), EASYSIMD_FLOAT32_C(  -722.53), EASYSIMD_FLOAT32_C(   296.99),
                         EASYSIMD_FLOAT32_C(   791.87), EASYSIMD_FLOAT32_C(   514.23), EASYSIMD_FLOAT32_C(   110.66), EASYSIMD_FLOAT32_C(  -891.24),
                         EASYSIMD_FLOAT32_C(  -893.87), EASYSIMD_FLOAT32_C(   597.88), EASYSIMD_FLOAT32_C(  -561.25), EASYSIMD_FLOAT32_C(  -182.63),
                         EASYSIMD_FLOAT32_C(   -91.96), EASYSIMD_FLOAT32_C(   272.32), EASYSIMD_FLOAT32_C(   -87.60), EASYSIMD_FLOAT32_C(    34.84)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -694.40), EASYSIMD_FLOAT32_C(  -404.01), EASYSIMD_FLOAT32_C(   766.51), EASYSIMD_FLOAT32_C(  -392.19),
                         EASYSIMD_FLOAT32_C(  -908.15), EASYSIMD_FLOAT32_C(  -690.12), EASYSIMD_FLOAT32_C(  -262.73), EASYSIMD_FLOAT32_C(  -353.25),
                         EASYSIMD_FLOAT32_C(  -451.03), EASYSIMD_FLOAT32_C(   -88.58), EASYSIMD_FLOAT32_C(   658.99), EASYSIMD_FLOAT32_C(  -961.05),
                         EASYSIMD_FLOAT32_C(  -743.39), EASYSIMD_FLOAT32_C(   747.85), EASYSIMD_FLOAT32_C(  -989.89), EASYSIMD_FLOAT32_C(   -48.62)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   706.89), EASYSIMD_FLOAT32_C(   473.35), EASYSIMD_FLOAT32_C(   525.10), EASYSIMD_FLOAT32_C(    58.51),
                         EASYSIMD_FLOAT32_C(  -849.29), EASYSIMD_FLOAT32_C(   830.92), EASYSIMD_FLOAT32_C(   666.67), EASYSIMD_FLOAT32_C(   510.60),
                         EASYSIMD_FLOAT32_C(   494.95), EASYSIMD_FLOAT32_C(  -644.02), EASYSIMD_FLOAT32_C(   666.48), EASYSIMD_FLOAT32_C(   728.99),
                         EASYSIMD_FLOAT32_C(    57.50), EASYSIMD_FLOAT32_C(  -509.99), EASYSIMD_FLOAT32_C(   -86.32), EASYSIMD_FLOAT32_C(   945.97)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   396.65), EASYSIMD_FLOAT32_C(  -337.05), EASYSIMD_FLOAT32_C(    13.39), EASYSIMD_FLOAT32_C(   374.11),
                         EASYSIMD_FLOAT32_C(   941.83), EASYSIMD_FLOAT32_C(   -80.39), EASYSIMD_FLOAT32_C(  -533.82), EASYSIMD_FLOAT32_C(   -81.97),
                         EASYSIMD_FLOAT32_C(   -76.37), EASYSIMD_FLOAT32_C(  -466.22), EASYSIMD_FLOAT32_C(  -527.13), EASYSIMD_FLOAT32_C(   285.31),
                         EASYSIMD_FLOAT32_C(  -159.19), EASYSIMD_FLOAT32_C(  -769.18), EASYSIMD_FLOAT32_C(   908.64), EASYSIMD_FLOAT32_C(  -647.66)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   706.89), EASYSIMD_FLOAT32_C(   473.35), EASYSIMD_FLOAT32_C(   525.10), EASYSIMD_FLOAT32_C(    58.51),
                         EASYSIMD_FLOAT32_C(  -849.29), EASYSIMD_FLOAT32_C(   830.92), EASYSIMD_FLOAT32_C(   666.67), EASYSIMD_FLOAT32_C(   510.60),
                         EASYSIMD_FLOAT32_C(   494.95), EASYSIMD_FLOAT32_C(  -644.02), EASYSIMD_FLOAT32_C(   666.48), EASYSIMD_FLOAT32_C(   728.99),
                         EASYSIMD_FLOAT32_C(    57.50), EASYSIMD_FLOAT32_C(  -509.99), EASYSIMD_FLOAT32_C(   -86.32), EASYSIMD_FLOAT32_C(   945.97)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   305.38), EASYSIMD_FLOAT32_C(   354.29), EASYSIMD_FLOAT32_C(   625.78), EASYSIMD_FLOAT32_C(   840.33),
                         EASYSIMD_FLOAT32_C(   398.08), EASYSIMD_FLOAT32_C(  -775.15), EASYSIMD_FLOAT32_C(  -749.75), EASYSIMD_FLOAT32_C(  -579.50),
                         EASYSIMD_FLOAT32_C(   326.67), EASYSIMD_FLOAT32_C(  -369.97), EASYSIMD_FLOAT32_C(  -888.36), EASYSIMD_FLOAT32_C(  -369.43),
                         EASYSIMD_FLOAT32_C(   587.01), EASYSIMD_FLOAT32_C(  -977.20), EASYSIMD_FLOAT32_C(  -154.58), EASYSIMD_FLOAT32_C(  -264.71)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   472.46), EASYSIMD_FLOAT32_C(  -814.28), EASYSIMD_FLOAT32_C(   331.94), EASYSIMD_FLOAT32_C(   -36.35),
                         EASYSIMD_FLOAT32_C(   -98.00), EASYSIMD_FLOAT32_C(   862.68), EASYSIMD_FLOAT32_C(  -130.24), EASYSIMD_FLOAT32_C(    65.39),
                         EASYSIMD_FLOAT32_C(  -826.35), EASYSIMD_FLOAT32_C(    92.38), EASYSIMD_FLOAT32_C(  -698.83), EASYSIMD_FLOAT32_C(   457.07),
                         EASYSIMD_FLOAT32_C(  -472.97), EASYSIMD_FLOAT32_C(  -117.57), EASYSIMD_FLOAT32_C(  -498.77), EASYSIMD_FLOAT32_C(   798.69)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   305.38), EASYSIMD_FLOAT32_C(   354.29), EASYSIMD_FLOAT32_C(   625.78), EASYSIMD_FLOAT32_C(   840.33),
                         EASYSIMD_FLOAT32_C(   398.08), EASYSIMD_FLOAT32_C(  -775.15), EASYSIMD_FLOAT32_C(  -749.75), EASYSIMD_FLOAT32_C(  -579.50),
                         EASYSIMD_FLOAT32_C(   326.67), EASYSIMD_FLOAT32_C(  -369.97), EASYSIMD_FLOAT32_C(  -888.36), EASYSIMD_FLOAT32_C(  -369.43),
                         EASYSIMD_FLOAT32_C(   587.01), EASYSIMD_FLOAT32_C(  -977.20), EASYSIMD_FLOAT32_C(  -154.58), EASYSIMD_FLOAT32_C(  -264.71)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   845.06), EASYSIMD_FLOAT32_C(  -527.19), EASYSIMD_FLOAT32_C(  -753.05), EASYSIMD_FLOAT32_C(  -867.95),
                         EASYSIMD_FLOAT32_C(   -98.38), EASYSIMD_FLOAT32_C(   -90.28), EASYSIMD_FLOAT32_C(   321.06), EASYSIMD_FLOAT32_C(  -308.74),
                         EASYSIMD_FLOAT32_C(   969.13), EASYSIMD_FLOAT32_C(  -263.02), EASYSIMD_FLOAT32_C(  -517.54), EASYSIMD_FLOAT32_C(   566.67),
                         EASYSIMD_FLOAT32_C(  -321.03), EASYSIMD_FLOAT32_C(   -19.45), EASYSIMD_FLOAT32_C(  -773.18), EASYSIMD_FLOAT32_C(  -562.24)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -313.43), EASYSIMD_FLOAT32_C(  -900.90), EASYSIMD_FLOAT32_C(  -480.72), EASYSIMD_FLOAT32_C(   288.15),
                         EASYSIMD_FLOAT32_C(   603.38), EASYSIMD_FLOAT32_C(   964.29), EASYSIMD_FLOAT32_C(   140.98), EASYSIMD_FLOAT32_C(   269.46),
                         EASYSIMD_FLOAT32_C(   960.77), EASYSIMD_FLOAT32_C(  -220.33), EASYSIMD_FLOAT32_C(   524.23), EASYSIMD_FLOAT32_C(  -633.14),
                         EASYSIMD_FLOAT32_C(  -680.30), EASYSIMD_FLOAT32_C(   880.56), EASYSIMD_FLOAT32_C(   661.76), EASYSIMD_FLOAT32_C(  -794.03)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   845.06), EASYSIMD_FLOAT32_C(  -527.19), EASYSIMD_FLOAT32_C(  -753.05), EASYSIMD_FLOAT32_C(  -867.95),
                         EASYSIMD_FLOAT32_C(   -98.38), EASYSIMD_FLOAT32_C(   -90.28), EASYSIMD_FLOAT32_C(   321.06), EASYSIMD_FLOAT32_C(  -308.74),
                         EASYSIMD_FLOAT32_C(   969.13), EASYSIMD_FLOAT32_C(  -263.02), EASYSIMD_FLOAT32_C(  -517.54), EASYSIMD_FLOAT32_C(   566.67),
                         EASYSIMD_FLOAT32_C(  -321.03), EASYSIMD_FLOAT32_C(   -19.45), EASYSIMD_FLOAT32_C(  -773.18), EASYSIMD_FLOAT32_C(  -562.24)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -595.71), EASYSIMD_FLOAT32_C(   923.49), EASYSIMD_FLOAT32_C(  -968.66), EASYSIMD_FLOAT32_C(   136.30),
                         EASYSIMD_FLOAT32_C(   658.04), EASYSIMD_FLOAT32_C(    31.08), EASYSIMD_FLOAT32_C(   664.79), EASYSIMD_FLOAT32_C(   525.95),
                         EASYSIMD_FLOAT32_C(   643.61), EASYSIMD_FLOAT32_C(  -559.86), EASYSIMD_FLOAT32_C(  -291.18), EASYSIMD_FLOAT32_C(    35.13),
                         EASYSIMD_FLOAT32_C(  -188.19), EASYSIMD_FLOAT32_C(   767.03), EASYSIMD_FLOAT32_C(  -828.01), EASYSIMD_FLOAT32_C(   801.09)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -750.17), EASYSIMD_FLOAT32_C(   128.67), EASYSIMD_FLOAT32_C(   441.75), EASYSIMD_FLOAT32_C(   625.42),
                         EASYSIMD_FLOAT32_C(   865.73), EASYSIMD_FLOAT32_C(  -522.43), EASYSIMD_FLOAT32_C(   871.78), EASYSIMD_FLOAT32_C(   736.62),
                         EASYSIMD_FLOAT32_C(   -52.49), EASYSIMD_FLOAT32_C(  -188.89), EASYSIMD_FLOAT32_C(   163.52), EASYSIMD_FLOAT32_C(   743.65),
                         EASYSIMD_FLOAT32_C(  -912.98), EASYSIMD_FLOAT32_C(  -904.70), EASYSIMD_FLOAT32_C(   973.06), EASYSIMD_FLOAT32_C(  -214.13)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -595.71), EASYSIMD_FLOAT32_C(   923.49), EASYSIMD_FLOAT32_C(  -968.66), EASYSIMD_FLOAT32_C(   136.30),
                         EASYSIMD_FLOAT32_C(   658.04), EASYSIMD_FLOAT32_C(    31.08), EASYSIMD_FLOAT32_C(   664.79), EASYSIMD_FLOAT32_C(   525.95),
                         EASYSIMD_FLOAT32_C(   643.61), EASYSIMD_FLOAT32_C(  -559.86), EASYSIMD_FLOAT32_C(  -291.18), EASYSIMD_FLOAT32_C(    35.13),
                         EASYSIMD_FLOAT32_C(  -188.19), EASYSIMD_FLOAT32_C(   767.03), EASYSIMD_FLOAT32_C(  -828.01), EASYSIMD_FLOAT32_C(   801.09)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -351.95), EASYSIMD_FLOAT32_C(   902.78), EASYSIMD_FLOAT32_C(  -172.20), EASYSIMD_FLOAT32_C(   540.77),
                         EASYSIMD_FLOAT32_C(  -431.24), EASYSIMD_FLOAT32_C(   243.87), EASYSIMD_FLOAT32_C(   216.07), EASYSIMD_FLOAT32_C(   747.45),
                         EASYSIMD_FLOAT32_C(  -864.81), EASYSIMD_FLOAT32_C(  -982.67), EASYSIMD_FLOAT32_C(  -710.14), EASYSIMD_FLOAT32_C(  -539.39),
                         EASYSIMD_FLOAT32_C(  -100.27), EASYSIMD_FLOAT32_C(  -988.79), EASYSIMD_FLOAT32_C(  -220.83), EASYSIMD_FLOAT32_C(   489.72)),
      UINT16_C(    0),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   688.70), EASYSIMD_FLOAT32_C(  -942.30), EASYSIMD_FLOAT32_C(  -353.35), EASYSIMD_FLOAT32_C(  -645.42),
                         EASYSIMD_FLOAT32_C(   206.41), EASYSIMD_FLOAT32_C(   546.87), EASYSIMD_FLOAT32_C(  -878.90), EASYSIMD_FLOAT32_C(   614.84),
                         EASYSIMD_FLOAT32_C(   757.82), EASYSIMD_FLOAT32_C(   388.29), EASYSIMD_FLOAT32_C(  -767.39), EASYSIMD_FLOAT32_C(   567.68),
                         EASYSIMD_FLOAT32_C(   464.76), EASYSIMD_FLOAT32_C(  -828.44), EASYSIMD_FLOAT32_C(   843.54), EASYSIMD_FLOAT32_C(   504.38)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -351.95), EASYSIMD_FLOAT32_C(   902.78), EASYSIMD_FLOAT32_C(  -172.20), EASYSIMD_FLOAT32_C(   540.77),
                         EASYSIMD_FLOAT32_C(  -431.24), EASYSIMD_FLOAT32_C(   243.87), EASYSIMD_FLOAT32_C(   216.07), EASYSIMD_FLOAT32_C(   747.45),
                         EASYSIMD_FLOAT32_C(  -864.81), EASYSIMD_FLOAT32_C(  -982.67), EASYSIMD_FLOAT32_C(  -710.14), EASYSIMD_FLOAT32_C(  -539.39),
                         EASYSIMD_FLOAT32_C(  -100.27), EASYSIMD_FLOAT32_C(  -988.79), EASYSIMD_FLOAT32_C(  -220.83), EASYSIMD_FLOAT32_C(   489.72)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mov_ps(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mov_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_maskz_mov_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
    {  UINT16_C(54402),
       easysimd_mm_set_epi8(INT8_C(  36), INT8_C(  97), INT8_C(-122), INT8_C(  62),
                        INT8_C( -43), INT8_C( -34), INT8_C( -14), INT8_C(-126),
                        INT8_C(  82), INT8_C( -27), INT8_C(-110), INT8_C( -49),
                        INT8_C(  86), INT8_C(  99), INT8_C( 100), INT8_C( -41)),
       easysimd_mm_set_epi8(INT8_C(  36), INT8_C(  97), INT8_C(   0), INT8_C(  62),
                        INT8_C(   0), INT8_C( -34), INT8_C(   0), INT8_C(   0),
                        INT8_C(  82), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                        INT8_C(   0), INT8_C(   0), INT8_C( 100), INT8_C(   0)) },
    {  UINT16_C( 9320),
       easysimd_mm_set_epi8(INT8_C(  42), INT8_C( -13), INT8_C(  59), INT8_C( -76),
                        INT8_C(  44), INT8_C(-127), INT8_C( -33), INT8_C(-116),
                        INT8_C(  13), INT8_C(   9), INT8_C( -47), INT8_C(  53),
                        INT8_C( -56), INT8_C(  87), INT8_C( -89), INT8_C(  72)),
       easysimd_mm_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(  59), INT8_C(   0),
                        INT8_C(   0), INT8_C(-127), INT8_C(   0), INT8_C(   0),
                        INT8_C(   0), INT8_C(   9), INT8_C( -47), INT8_C(   0),
                        INT8_C( -56), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    {  UINT16_C( 7828),
       easysimd_mm_set_epi8(INT8_C( -41), INT8_C( -58), INT8_C(  78), INT8_C( -99),
                        INT8_C( -79), INT8_C(  93), INT8_C(  74), INT8_C(   5),
                        INT8_C(  40), INT8_C( -62), INT8_C( 109), INT8_C( -74),
                        INT8_C(   1), INT8_C( -60), INT8_C(  94), INT8_C(  12)),
       easysimd_mm_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -99),
                        INT8_C( -79), INT8_C(  93), INT8_C(  74), INT8_C(   0),
                        INT8_C(  40), INT8_C(   0), INT8_C(   0), INT8_C( -74),
                        INT8_C(   0), INT8_C( -60), INT8_C(   0), INT8_C(   0)) },
    {  UINT16_C(55181),
       easysimd_mm_set_epi8(INT8_C(  37), INT8_C(  84), INT8_C( -36), INT8_C(-122),
                        INT8_C(  25), INT8_C( 108), INT8_C(  27), INT8_C(  95),
                        INT8_C( -44), INT8_C(-128), INT8_C( 110), INT8_C( -66),
                        INT8_C(  74), INT8_C( -16), INT8_C( 122), INT8_C( -30)),
       easysimd_mm_set_epi8(INT8_C(  37), INT8_C(  84), INT8_C(   0), INT8_C(-122),
                        INT8_C(   0), INT8_C( 108), INT8_C(  27), INT8_C(  95),
                        INT8_C( -44), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                        INT8_C(  74), INT8_C( -16), INT8_C(   0), INT8_C( -30)) },
    {  UINT16_C(57564),
       easysimd_mm_set_epi8(INT8_C( -26), INT8_C(  -5), INT8_C(   7), INT8_C( -63),
                        INT8_C(  47), INT8_C(  32), INT8_C(  62), INT8_C(-108),
                        INT8_C(  26), INT8_C(  67), INT8_C( -45), INT8_C(  32),
                        INT8_C( -38), INT8_C(  61), INT8_C(-123), INT8_C(-123)),
       easysimd_mm_set_epi8(INT8_C( -26), INT8_C(  -5), INT8_C(   7), INT8_C(   0),
                        INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                        INT8_C(  26), INT8_C(  67), INT8_C(   0), INT8_C(  32),
                        INT8_C( -38), INT8_C(  61), INT8_C(   0), INT8_C(   0)) },
    {  UINT16_C(58988),
       easysimd_mm_set_epi8(INT8_C( 108), INT8_C(   5), INT8_C(-115), INT8_C( -87),
                        INT8_C( 112), INT8_C(  24), INT8_C(  18), INT8_C( -62),
                        INT8_C( 120), INT8_C(  62), INT8_C( -22), INT8_C( -32),
                        INT8_C(  32), INT8_C( -91), INT8_C(  65), INT8_C(  79)),
       easysimd_mm_set_epi8(INT8_C( 108), INT8_C(   5), INT8_C(-115), INT8_C(   0),
                        INT8_C(   0), INT8_C(  24), INT8_C(  18), INT8_C(   0),
                        INT8_C(   0), INT8_C(  62), INT8_C( -22), INT8_C(   0),
                        INT8_C(  32), INT8_C( -91), INT8_C(   0), INT8_C(   0)) },
    {  UINT16_C(50535),
       easysimd_mm_set_epi8(INT8_C(-119), INT8_C( -52), INT8_C(-117), INT8_C( 112),
                        INT8_C( -70), INT8_C(-108), INT8_C(  -6), INT8_C(  88),
                        INT8_C(   5), INT8_C( -84), INT8_C(  11), INT8_C( -55),
                        INT8_C(-116), INT8_C(   8), INT8_C(  68), INT8_C(-111)),
       easysimd_mm_set_epi8(INT8_C(-119), INT8_C( -52), INT8_C(   0), INT8_C(   0),
                        INT8_C(   0), INT8_C(-108), INT8_C(   0), INT8_C(  88),
                        INT8_C(   0), INT8_C( -84), INT8_C(  11), INT8_C(   0),
                        INT8_C(   0), INT8_C(   8), INT8_C(  68), INT8_C(-111)) },
    {  UINT16_C(21029),
       easysimd_mm_set_epi8(INT8_C(-123), INT8_C(-110), INT8_C(  43), INT8_C( -78),
                        INT8_C(-113), INT8_C(  -6), INT8_C( -22), INT8_C(-111),
                        INT8_C(-114), INT8_C(  91), INT8_C(  78), INT8_C(  20),
                        INT8_C(  94), INT8_C(   5), INT8_C( 125), INT8_C(  13)),
       easysimd_mm_set_epi8(INT8_C(   0), INT8_C(-110), INT8_C(   0), INT8_C( -78),
                        INT8_C(   0), INT8_C(   0), INT8_C( -22), INT8_C(   0),
                        INT8_C(   0), INT8_C(   0), INT8_C(  78), INT8_C(   0),
                        INT8_C(   0), INT8_C(   5), INT8_C(   0), INT8_C(  13)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mov_epi8(test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mov_epi8");
    easysimd_assert_m128i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_maskz_mov_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
    {  UINT8_C(172),
      easysimd_mm_set_epi16(INT16_C( 31369), INT16_C( 24471), INT16_C( -2198), INT16_C( 13931),
                         INT16_C(  8708), INT16_C(-30158), INT16_C( 19991), INT16_C(-25642)),
      easysimd_mm_set_epi16(INT16_C( 31369), INT16_C(     0), INT16_C( -2198), INT16_C(     0),
                         INT16_C(  8708), INT16_C(-30158), INT16_C(     0), INT16_C(     0)) },
   {  UINT8_C(174),
      easysimd_mm_set_epi16(INT16_C( 15685), INT16_C( 28576), INT16_C( 31286), INT16_C( 30917),
                         INT16_C( 32368), INT16_C( -7767), INT16_C(  5413), INT16_C( -7264)),
      easysimd_mm_set_epi16(INT16_C( 15685), INT16_C(     0), INT16_C( 31286), INT16_C(     0),
                         INT16_C( 32368), INT16_C( -7767), INT16_C(  5413), INT16_C(     0)) },
   {  UINT8_C(204),
      easysimd_mm_set_epi16(INT16_C(-32746), INT16_C( 32574), INT16_C( 12624), INT16_C( 27372),
                         INT16_C(-30923), INT16_C( 29148), INT16_C(-21083), INT16_C( 14295)),
      easysimd_mm_set_epi16(INT16_C(-32746), INT16_C( 32574), INT16_C(     0), INT16_C(     0),
                         INT16_C(-30923), INT16_C( 29148), INT16_C(     0), INT16_C(     0)) },
   {  UINT8_C( 95),
      easysimd_mm_set_epi16(INT16_C(-30267), INT16_C(-15896), INT16_C( 22574), INT16_C(  2859),
                         INT16_C(  2365), INT16_C(  -901), INT16_C( 18813), INT16_C( 18335)),
      easysimd_mm_set_epi16(INT16_C(     0), INT16_C(-15896), INT16_C(     0), INT16_C(  2859),
                         INT16_C(  2365), INT16_C(  -901), INT16_C( 18813), INT16_C( 18335)) },
   {  UINT8_C( 67),
      easysimd_mm_set_epi16(INT16_C( 16076), INT16_C( 28949), INT16_C( 18472), INT16_C( 18435),
                         INT16_C(-29130), INT16_C(-15163), INT16_C(-12433), INT16_C( -3463)),
      easysimd_mm_set_epi16(INT16_C(     0), INT16_C( 28949), INT16_C(     0), INT16_C(     0),
                         INT16_C(     0), INT16_C(     0), INT16_C(-12433), INT16_C( -3463)) },
   {  UINT8_C( 73),
      easysimd_mm_set_epi16(INT16_C(-30899), INT16_C(-31361), INT16_C(-22956), INT16_C(-14855),
                         INT16_C(  -601), INT16_C(  2058), INT16_C( 17396), INT16_C(-31263)),
      easysimd_mm_set_epi16(INT16_C(     0), INT16_C(-31361), INT16_C(     0), INT16_C(     0),
                         INT16_C(  -601), INT16_C(     0), INT16_C(     0), INT16_C(-31263)) },
   {  UINT8_C(  1),
      easysimd_mm_set_epi16(INT16_C(  5707), INT16_C(-20763), INT16_C(  8635), INT16_C( -4245),
                         INT16_C( 27666), INT16_C(-18424), INT16_C(-22687), INT16_C( 15686)),
      easysimd_mm_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                         INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C( 15686)) },
   {  UINT8_C(172),
      easysimd_mm_set_epi16(INT16_C(  8809), INT16_C( 29917), INT16_C(   520), INT16_C(-12425),
                         INT16_C( 13592), INT16_C(-10913), INT16_C(-21871), INT16_C(  6317)),
      easysimd_mm_set_epi16(INT16_C(  8809), INT16_C(     0), INT16_C(   520), INT16_C(     0),
                         INT16_C( 13592), INT16_C(-10913), INT16_C(     0), INT16_C(     0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mov_epi16(test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mov_epi16");
    easysimd_assert_m128i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_maskz_mov_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
     {  UINT8_C(192),
        easysimd_mm_set_epi32(INT32_C(  656441296), INT32_C(-1852032257), INT32_C(  299494207), INT32_C(-1616873206)),
        easysimd_mm_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
     {  UINT8_C(138),
        easysimd_mm_set_epi32(INT32_C(  707152322), INT32_C(-1311270924), INT32_C(-1503159730), INT32_C(-2099401846)),
        easysimd_mm_set_epi32(INT32_C(  707152322), INT32_C(          0), INT32_C(-1503159730), INT32_C(          0)) },
     {  UINT8_C(202),
        easysimd_mm_set_epi32(INT32_C(-1455100666), INT32_C(-2025285461), INT32_C( -179772388), INT32_C( 1367812127)),
        easysimd_mm_set_epi32(INT32_C(-1455100666), INT32_C(          0), INT32_C( -179772388), INT32_C(          0)) },
     {  UINT8_C(144),
        easysimd_mm_set_epi32(INT32_C(-1107178304), INT32_C(-1037282057), INT32_C(  779093870), INT32_C( 1250766721)),
        easysimd_mm_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
     {  UINT8_C(121),
        easysimd_mm_set_epi32(INT32_C( -756555400), INT32_C( 1672370881), INT32_C( -263709411), INT32_C(  606108964)),
        easysimd_mm_set_epi32(INT32_C( -756555400), INT32_C(          0), INT32_C(          0), INT32_C(  606108964)) },
     {  UINT8_C( 11),
        easysimd_mm_set_epi32(INT32_C(  291215521), INT32_C(  371049029), INT32_C(  324114641), INT32_C( -986925670)),
        easysimd_mm_set_epi32(INT32_C(  291215521), INT32_C(          0), INT32_C(  324114641), INT32_C( -986925670)) },
     {  UINT8_C(200),
        easysimd_mm_set_epi32(INT32_C(-1248714533), INT32_C(  110176831), INT32_C(-1962006925), INT32_C( -973547490)),
        easysimd_mm_set_epi32(INT32_C(-1248714533), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
     {  UINT8_C(138),
        easysimd_mm_set_epi32(INT32_C( -971622476), INT32_C(  -95064376), INT32_C( -736538751), INT32_C(    7991884)),
        easysimd_mm_set_epi32(INT32_C( -971622476), INT32_C(          0), INT32_C( -736538751), INT32_C(          0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mov_epi32(test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mov_epi32");
    easysimd_assert_m128i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_maskz_mov_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[8] = {
     {  UINT8_C(140),
        easysimd_mm_set_epi64x(INT64_C( 3798083087260184318), INT64_C( 5657333801282264243)),
        easysimd_mm_set_epi64x(INT64_C(                   0), INT64_C(                   0)) },
     {  UINT8_C( 59),
        easysimd_mm_set_epi64x(INT64_C( 6150838870455976373), INT64_C(-1888156961938500809)),
        easysimd_mm_set_epi64x(INT64_C( 6150838870455976373), INT64_C(-1888156961938500809)) },
     {  UINT8_C( 85),
        easysimd_mm_set_epi64x(INT64_C(-2963288110518582462), INT64_C( 4379558933354650160)),
        easysimd_mm_set_epi64x(INT64_C(                   0), INT64_C( 4379558933354650160)) },
     {  UINT8_C(190),
        easysimd_mm_set_epi64x(INT64_C( 1293362407707663546), INT64_C( 3921888525347819158)),
        easysimd_mm_set_epi64x(INT64_C( 1293362407707663546), INT64_C(                   0)) },
     {  UINT8_C(114),
        easysimd_mm_set_epi64x(INT64_C(-7166753234573077348), INT64_C( 1514796214136072870)),
        easysimd_mm_set_epi64x(INT64_C(-7166753234573077348), INT64_C(                   0)) },
     {  UINT8_C( 57),
        easysimd_mm_set_epi64x(INT64_C(-5321356301108453394), INT64_C(-2450051547146928613)),
        easysimd_mm_set_epi64x(INT64_C(                   0), INT64_C(-2450051547146928613)) },
     {  UINT8_C( 72),
        easysimd_mm_set_epi64x(INT64_C(-3635596340953309068), INT64_C(-4947516809045744754)),
        easysimd_mm_set_epi64x(INT64_C(                   0), INT64_C(                   0)) },
     {  UINT8_C( 27),
        easysimd_mm_set_epi64x(INT64_C(-4723518328184072824), INT64_C(-6365694246941149609)),
        easysimd_mm_set_epi64x(INT64_C(-4723518328184072824), INT64_C(-6365694246941149609)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mov_epi64(test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mov_epi64");
    easysimd_assert_m128i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_maskz_mov_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m128d a;
    easysimd__m128d r;
  } test_vec[8] = {
    { UINT8_C(210),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   55.56), EASYSIMD_FLOAT64_C(  306.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   55.56), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(  7),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  202.21), EASYSIMD_FLOAT64_C( -678.71)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  202.21), EASYSIMD_FLOAT64_C( -678.71)) },
    { UINT8_C( 50),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  680.40), EASYSIMD_FLOAT64_C(  906.67)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  680.40), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(229),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -422.72), EASYSIMD_FLOAT64_C(  572.83)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  572.83)) },
    { UINT8_C(117),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -76.19), EASYSIMD_FLOAT64_C( -654.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -654.60)) },
    { UINT8_C(130),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -711.42), EASYSIMD_FLOAT64_C(  -22.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -711.42), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C( 62),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -413.23), EASYSIMD_FLOAT64_C(  547.52)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -413.23), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(165),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  575.41), EASYSIMD_FLOAT64_C( -702.01)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -702.01)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = test_vec[i].a;
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mov_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mov_pd");
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_maskz_mov_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m128 a;
    easysimd__m128 r;
  } test_vec[8] = {
    { UINT16_C(  126),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -678.71), EASYSIMD_FLOAT32_C(   675.53), EASYSIMD_FLOAT32_C(    55.56), EASYSIMD_FLOAT32_C(   306.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -678.71), EASYSIMD_FLOAT32_C(   675.53), EASYSIMD_FLOAT32_C(    55.56), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(   44),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   941.87), EASYSIMD_FLOAT32_C(   680.40), EASYSIMD_FLOAT32_C(   906.67), EASYSIMD_FLOAT32_C(  -364.25)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   941.87), EASYSIMD_FLOAT32_C(   680.40), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(  117),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -76.19), EASYSIMD_FLOAT32_C(  -654.60), EASYSIMD_FLOAT32_C(  -721.91), EASYSIMD_FLOAT32_C(  -422.72)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -654.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -422.72)) },
    { UINT16_C(   76),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   547.52), EASYSIMD_FLOAT32_C(  -627.17), EASYSIMD_FLOAT32_C(  -711.42), EASYSIMD_FLOAT32_C(   -22.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   547.52), EASYSIMD_FLOAT32_C(  -627.17), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(  101),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -822.97), EASYSIMD_FLOAT32_C(   575.41), EASYSIMD_FLOAT32_C(  -702.01), EASYSIMD_FLOAT32_C(  -488.76)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   575.41), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -488.76)) },
    { UINT16_C(  149),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   804.55), EASYSIMD_FLOAT32_C(  -888.85), EASYSIMD_FLOAT32_C(   750.71), EASYSIMD_FLOAT32_C(   346.51)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -888.85), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   346.51)) },
    { UINT16_C(  115),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -17.38), EASYSIMD_FLOAT32_C(   623.33), EASYSIMD_FLOAT32_C(   459.80), EASYSIMD_FLOAT32_C(   837.15)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   459.80), EASYSIMD_FLOAT32_C(   837.15)) },
    { UINT16_C(   50),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   197.69), EASYSIMD_FLOAT32_C(   233.42), EASYSIMD_FLOAT32_C(   153.73), EASYSIMD_FLOAT32_C(   616.58)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   153.73), EASYSIMD_FLOAT32_C(     0.00)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = test_vec[i].a;
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mov_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mov_ps");
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_mov_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask32 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { UINT32_C(1332074171),
      easysimd_mm256_set_epi8(INT8_C( 121), INT8_C(  75), INT8_C(  39), INT8_C(-100),
                           INT8_C(  23), INT8_C(  80), INT8_C(  88), INT8_C(  14),
                           INT8_C( -82), INT8_C( -32), INT8_C( -73), INT8_C( -78),
                           INT8_C( -21), INT8_C(  76), INT8_C(  33), INT8_C(  90),
                           INT8_C( -57), INT8_C( -12), INT8_C(-121), INT8_C( 101),
                           INT8_C(   6), INT8_C( -36), INT8_C( -50), INT8_C( -33),
                           INT8_C( -83), INT8_C( -92), INT8_C(   2), INT8_C(  69),
                           INT8_C(  62), INT8_C(  89), INT8_C( 105), INT8_C(  58)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C(  75), INT8_C(   0), INT8_C(   0),
                           INT8_C(  23), INT8_C(  80), INT8_C(  88), INT8_C(  14),
                           INT8_C(   0), INT8_C( -32), INT8_C( -73), INT8_C(   0),
                           INT8_C(   0), INT8_C(  76), INT8_C(   0), INT8_C(  90),
                           INT8_C( -57), INT8_C( -12), INT8_C(   0), INT8_C( 101),
                           INT8_C(   0), INT8_C( -36), INT8_C( -50), INT8_C(   0),
                           INT8_C( -83), INT8_C(   0), INT8_C(   2), INT8_C(  69),
                           INT8_C(  62), INT8_C(   0), INT8_C( 105), INT8_C(  58)) },
    { UINT32_C(4272165599),
      easysimd_mm256_set_epi8(INT8_C(  23), INT8_C(   6), INT8_C(  61), INT8_C(  68),
                           INT8_C( -53), INT8_C(-110), INT8_C(  53), INT8_C( -67),
                           INT8_C(  -9), INT8_C( -52), INT8_C(  27), INT8_C( -40),
                           INT8_C(  57), INT8_C( -80), INT8_C( -28), INT8_C(  64),
                           INT8_C(  70), INT8_C( -40), INT8_C(  14), INT8_C( -38),
                           INT8_C( -38), INT8_C( -99), INT8_C( -37), INT8_C( -35),
                           INT8_C( -82), INT8_C( -60), INT8_C( -40), INT8_C( -40),
                           INT8_C(  -5), INT8_C(   8), INT8_C( 109), INT8_C(  95)),
      easysimd_mm256_set_epi8(INT8_C(  23), INT8_C(   6), INT8_C(  61), INT8_C(  68),
                           INT8_C( -53), INT8_C(-110), INT8_C(  53), INT8_C(   0),
                           INT8_C(  -9), INT8_C(   0), INT8_C(  27), INT8_C(   0),
                           INT8_C(   0), INT8_C( -80), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -38),
                           INT8_C(   0), INT8_C(   0), INT8_C( -37), INT8_C(   0),
                           INT8_C( -82), INT8_C( -60), INT8_C(   0), INT8_C( -40),
                           INT8_C(  -5), INT8_C(   8), INT8_C( 109), INT8_C(  95)) },
    { UINT32_C(3823231310),
      easysimd_mm256_set_epi8(INT8_C( -42), INT8_C(  44), INT8_C(  70), INT8_C( -24),
                           INT8_C( -86), INT8_C( 112), INT8_C( 116), INT8_C( -61),
                           INT8_C(  94), INT8_C( -56), INT8_C( -83), INT8_C(  37),
                           INT8_C(  45), INT8_C(  44), INT8_C(  79), INT8_C( 122),
                           INT8_C( -54), INT8_C( -68), INT8_C(  19), INT8_C(  39),
                           INT8_C(  17), INT8_C( -32), INT8_C( -47), INT8_C( -26),
                           INT8_C( -23), INT8_C(  30), INT8_C(  98), INT8_C(   3),
                           INT8_C( -92), INT8_C( -30), INT8_C(  -8), INT8_C( -30)),
      easysimd_mm256_set_epi8(INT8_C( -42), INT8_C(  44), INT8_C(  70), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( 116), INT8_C( -61),
                           INT8_C(  94), INT8_C( -56), INT8_C( -83), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( 122),
                           INT8_C( -54), INT8_C( -68), INT8_C(  19), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -26),
                           INT8_C(   0), INT8_C(  30), INT8_C(   0), INT8_C(   0),
                           INT8_C( -92), INT8_C( -30), INT8_C(  -8), INT8_C(   0)) },
    { UINT32_C(2639652614),
      easysimd_mm256_set_epi8(INT8_C( -53), INT8_C(  96), INT8_C(  40), INT8_C( -52),
                           INT8_C( -17), INT8_C(  -6), INT8_C(-108), INT8_C(  33),
                           INT8_C( -15), INT8_C( 113), INT8_C(  31), INT8_C( -14),
                           INT8_C( 124), INT8_C(  15), INT8_C(  90), INT8_C(   1),
                           INT8_C(  36), INT8_C(-115), INT8_C( -95), INT8_C(   4),
                           INT8_C(  50), INT8_C( -54), INT8_C(  94), INT8_C(  54),
                           INT8_C( 109), INT8_C(-103), INT8_C(-124), INT8_C(  34),
                           INT8_C( -16), INT8_C(  97), INT8_C(  -7), INT8_C(  98)),
      easysimd_mm256_set_epi8(INT8_C( -53), INT8_C(   0), INT8_C(   0), INT8_C( -52),
                           INT8_C( -17), INT8_C(  -6), INT8_C(   0), INT8_C(  33),
                           INT8_C(   0), INT8_C( 113), INT8_C(   0), INT8_C( -14),
                           INT8_C(   0), INT8_C(  15), INT8_C(   0), INT8_C(   1),
                           INT8_C(  36), INT8_C(-115), INT8_C( -95), INT8_C(   0),
                           INT8_C(   0), INT8_C( -54), INT8_C(  94), INT8_C(  54),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  97), INT8_C(  -7), INT8_C(   0)) },
    { UINT32_C(2877003463),
      easysimd_mm256_set_epi8(INT8_C(-108), INT8_C(  14), INT8_C( 103), INT8_C(  32),
                           INT8_C(  25), INT8_C(-108), INT8_C( -56), INT8_C(-111),
                           INT8_C(  23), INT8_C( -20), INT8_C(   4), INT8_C(  81),
                           INT8_C(  39), INT8_C(  39), INT8_C(  82), INT8_C( -15),
                           INT8_C( -87), INT8_C(  90), INT8_C( -91), INT8_C(   3),
                           INT8_C( -91), INT8_C(  55), INT8_C(  72), INT8_C( -46),
                           INT8_C(  48), INT8_C( -19), INT8_C( -87), INT8_C( 100),
                           INT8_C( -44), INT8_C( -79), INT8_C( -72), INT8_C(  73)),
      easysimd_mm256_set_epi8(INT8_C(-108), INT8_C(   0), INT8_C( 103), INT8_C(   0),
                           INT8_C(  25), INT8_C(   0), INT8_C( -56), INT8_C(-111),
                           INT8_C(   0), INT8_C( -20), INT8_C(   4), INT8_C(  81),
                           INT8_C(  39), INT8_C(   0), INT8_C(  82), INT8_C( -15),
                           INT8_C( -87), INT8_C(   0), INT8_C(   0), INT8_C(   3),
                           INT8_C(   0), INT8_C(  55), INT8_C(  72), INT8_C(   0),
                           INT8_C(  48), INT8_C( -19), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -79), INT8_C( -72), INT8_C(  73)) },
    { UINT32_C(2869692151),
      easysimd_mm256_set_epi8(INT8_C(  22), INT8_C(-112), INT8_C(  66), INT8_C( -76),
                           INT8_C(  79), INT8_C(-100), INT8_C( -47), INT8_C(-114),
                           INT8_C( -72), INT8_C(  67), INT8_C(   3), INT8_C(  -9),
                           INT8_C(  88), INT8_C(  -5), INT8_C(-111), INT8_C(-100),
                           INT8_C( -94), INT8_C( -72), INT8_C( -45), INT8_C( -95),
                           INT8_C( 119), INT8_C( -81), INT8_C(  38), INT8_C(-111),
                           INT8_C(  72), INT8_C( -95), INT8_C( 104), INT8_C( -28),
                           INT8_C(  25), INT8_C(  84), INT8_C(  66), INT8_C(  19)),
      easysimd_mm256_set_epi8(INT8_C(  22), INT8_C(   0), INT8_C(  66), INT8_C(   0),
                           INT8_C(  79), INT8_C(   0), INT8_C( -47), INT8_C(-114),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  88), INT8_C(  -5), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -81), INT8_C(  38), INT8_C(   0),
                           INT8_C(  72), INT8_C( -95), INT8_C( 104), INT8_C( -28),
                           INT8_C(   0), INT8_C(  84), INT8_C(  66), INT8_C(  19)) },
    { UINT32_C(1633656989),
      easysimd_mm256_set_epi8(INT8_C(  81), INT8_C( 114), INT8_C( -76), INT8_C( -63),
                           INT8_C(  30), INT8_C(  66), INT8_C(  18), INT8_C(-119),
                           INT8_C(  26), INT8_C(  28), INT8_C(  56), INT8_C( 127),
                           INT8_C( -81), INT8_C(  -7), INT8_C( -20), INT8_C( -35),
                           INT8_C(  -7), INT8_C(  37), INT8_C( -47), INT8_C(  78),
                           INT8_C( 114), INT8_C( -18), INT8_C(  72), INT8_C(  -8),
                           INT8_C(-101), INT8_C( -13), INT8_C(  76), INT8_C(  -5),
                           INT8_C(  -5), INT8_C( -50), INT8_C( -99), INT8_C(  84)),
      easysimd_mm256_set_epi8(INT8_C(   0), INT8_C( 114), INT8_C( -76), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(-119),
                           INT8_C(   0), INT8_C(  28), INT8_C(   0), INT8_C( 127),
                           INT8_C( -81), INT8_C(  -7), INT8_C( -20), INT8_C( -35),
                           INT8_C(  -7), INT8_C(   0), INT8_C( -47), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-101), INT8_C(   0), INT8_C(   0), INT8_C(  -5),
                           INT8_C(  -5), INT8_C( -50), INT8_C(   0), INT8_C(  84)) },
    { UINT32_C(4185163230),
      easysimd_mm256_set_epi8(INT8_C( -29), INT8_C(-121), INT8_C( -23), INT8_C(  64),
                           INT8_C(  12), INT8_C(   5), INT8_C(  73), INT8_C(  52),
                           INT8_C( -53), INT8_C(  62), INT8_C(   8), INT8_C(-112),
                           INT8_C(  -8), INT8_C(  99), INT8_C( -12), INT8_C(-118),
                           INT8_C( -33), INT8_C( -37), INT8_C( -98), INT8_C( -94),
                           INT8_C(-119), INT8_C(  79), INT8_C( -25), INT8_C(  47),
                           INT8_C(  80), INT8_C(  89), INT8_C(   5), INT8_C(   9),
                           INT8_C( -36), INT8_C(  79), INT8_C(   8), INT8_C(  89)),
      easysimd_mm256_set_epi8(INT8_C( -29), INT8_C(-121), INT8_C( -23), INT8_C(  64),
                           INT8_C(  12), INT8_C(   0), INT8_C(   0), INT8_C(  52),
                           INT8_C(   0), INT8_C(  62), INT8_C(   8), INT8_C(-112),
                           INT8_C(   0), INT8_C(  99), INT8_C(   0), INT8_C(   0),
                           INT8_C( -33), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  79), INT8_C(   0), INT8_C(  47),
                           INT8_C(  80), INT8_C(  89), INT8_C(   0), INT8_C(   9),
                           INT8_C( -36), INT8_C(  79), INT8_C(   8), INT8_C(   0)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mov_epi8(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mov_epi8");
    easysimd_assert_m256i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_mov_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { UINT16_C(41021),
      easysimd_mm256_set_epi16(INT16_C(-23030), INT16_C(  6803), INT16_C(-21055), INT16_C(  -910),
                            INT16_C( -6009), INT16_C( 10471), INT16_C(-29834), INT16_C(-14111),
                            INT16_C( -2981), INT16_C( 28733), INT16_C( 11699), INT16_C(  7781),
                            INT16_C( 29036), INT16_C( -8103), INT16_C(-21310), INT16_C(  9176)),
      easysimd_mm256_set_epi16(INT16_C(-23030), INT16_C(     0), INT16_C(-21055), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C( 11699), INT16_C(  7781),
                            INT16_C( 29036), INT16_C( -8103), INT16_C(     0), INT16_C(  9176)) },
    { UINT16_C(53637),
      easysimd_mm256_set_epi16(INT16_C(-17353), INT16_C(-24912), INT16_C(-16017), INT16_C(-32768),
                            INT16_C( 30563), INT16_C( -5523), INT16_C(-18306), INT16_C( 14754),
                            INT16_C(-23068), INT16_C(-17313), INT16_C( 21598), INT16_C( 12635),
                            INT16_C( 17053), INT16_C(  3377), INT16_C( 28887), INT16_C( 29062)),
      easysimd_mm256_set_epi16(INT16_C(-17353), INT16_C(-24912), INT16_C(     0), INT16_C(-32768),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C( 14754),
                            INT16_C(-23068), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(  3377), INT16_C(     0), INT16_C( 29062)) },
    { UINT16_C(52310),
      easysimd_mm256_set_epi16(INT16_C(  4085), INT16_C( 10000), INT16_C(-17688), INT16_C( 28540),
                            INT16_C(  9971), INT16_C( -9002), INT16_C(-22233), INT16_C(-13917),
                            INT16_C(-13732), INT16_C(  -199), INT16_C(  9707), INT16_C( 31342),
                            INT16_C(-13386), INT16_C(-15675), INT16_C( 10143), INT16_C( 19953)),
      easysimd_mm256_set_epi16(INT16_C(  4085), INT16_C( 10000), INT16_C(     0), INT16_C(     0),
                            INT16_C(  9971), INT16_C( -9002), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(  -199), INT16_C(     0), INT16_C( 31342),
                            INT16_C(     0), INT16_C(-15675), INT16_C( 10143), INT16_C(     0)) },
    { UINT16_C(11313),
      easysimd_mm256_set_epi16(INT16_C(-25947), INT16_C( 19467), INT16_C( 22325), INT16_C( 14960),
                            INT16_C( 16296), INT16_C(-12892), INT16_C(  9434), INT16_C( 15492),
                            INT16_C(-30515), INT16_C(-13927), INT16_C( 24112), INT16_C(  9227),
                            INT16_C(-20054), INT16_C(-11664), INT16_C( -7103), INT16_C(-13246)),
      easysimd_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C( 22325), INT16_C(     0),
                            INT16_C( 16296), INT16_C(-12892), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C( 24112), INT16_C(  9227),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(-13246)) },
    { UINT16_C(54624),
      easysimd_mm256_set_epi16(INT16_C(-10124), INT16_C(  1110), INT16_C(  1704), INT16_C(-17853),
                            INT16_C( -7561), INT16_C(-19432), INT16_C( 22127), INT16_C(-30033),
                            INT16_C(-17362), INT16_C( -1830), INT16_C(-16587), INT16_C(-17056),
                            INT16_C(-14539), INT16_C(  7972), INT16_C(-26491), INT16_C( 20406)),
      easysimd_mm256_set_epi16(INT16_C(-10124), INT16_C(  1110), INT16_C(     0), INT16_C(-17853),
                            INT16_C(     0), INT16_C(-19432), INT16_C(     0), INT16_C(-30033),
                            INT16_C(     0), INT16_C( -1830), INT16_C(-16587), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
    { UINT16_C(44378),
      easysimd_mm256_set_epi16(INT16_C( 16917), INT16_C( 10042), INT16_C(  5958), INT16_C( -4695),
                            INT16_C(-20590), INT16_C( 17528), INT16_C( -6738), INT16_C(-26754),
                            INT16_C( 30496), INT16_C(  8574), INT16_C(  3335), INT16_C(-11669),
                            INT16_C( 15597), INT16_C(-30582), INT16_C(-21551), INT16_C(-25534)),
      easysimd_mm256_set_epi16(INT16_C( 16917), INT16_C(     0), INT16_C(  5958), INT16_C(     0),
                            INT16_C(-20590), INT16_C( 17528), INT16_C(     0), INT16_C(-26754),
                            INT16_C(     0), INT16_C(  8574), INT16_C(     0), INT16_C(-11669),
                            INT16_C( 15597), INT16_C(     0), INT16_C(-21551), INT16_C(     0)) },
    { UINT16_C(12440),
      easysimd_mm256_set_epi16(INT16_C( 12048), INT16_C( -8528), INT16_C(-31627), INT16_C( 26711),
                            INT16_C( -4678), INT16_C( 32013), INT16_C(   814), INT16_C( 19873),
                            INT16_C( 32199), INT16_C( -7421), INT16_C( 21197), INT16_C( 25563),
                            INT16_C( 14671), INT16_C( 16470), INT16_C( 30174), INT16_C( -7130)),
      easysimd_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(-31627), INT16_C( 26711),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C( 32199), INT16_C(     0), INT16_C(     0), INT16_C( 25563),
                            INT16_C( 14671), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
    { UINT16_C( 8281),
      easysimd_mm256_set_epi16(INT16_C( 32229), INT16_C( -2511), INT16_C(-10942), INT16_C(-28733),
                            INT16_C( -8714), INT16_C( -6616), INT16_C(  4922), INT16_C(  1537),
                            INT16_C( -8589), INT16_C(  6229), INT16_C(-12142), INT16_C( 12862),
                            INT16_C(-16969), INT16_C( 25143), INT16_C(-29570), INT16_C( 25018)),
      easysimd_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(-10942), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(  6229), INT16_C(     0), INT16_C( 12862),
                            INT16_C(-16969), INT16_C(     0), INT16_C(     0), INT16_C( 25018)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mov_epi16(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mov_epi16");
    easysimd_assert_m256i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_mov_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
   {  UINT8_C(205),
      easysimd_mm256_set_epi32(INT32_C( -433311806), INT32_C(  408583050), INT32_C( -306453652), INT32_C( -661693879),
                            INT32_C( 1329919822), INT32_C(  -49396337), INT32_C( -975523137), INT32_C(  228489302)),
      easysimd_mm256_set_epi32(INT32_C( -433311806), INT32_C(  408583050), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1329919822), INT32_C(  -49396337), INT32_C(          0), INT32_C(  228489302)) },
   {  UINT8_C( 99),
      easysimd_mm256_set_epi32(INT32_C( 1010695071), INT32_C(  737167817), INT32_C( 1850343310), INT32_C( 1216609214),
                            INT32_C(-1976576002), INT32_C( 1498708626), INT32_C( -621595293), INT32_C(-2111598997)),
      easysimd_mm256_set_epi32(INT32_C(          0), INT32_C(  737167817), INT32_C( 1850343310), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( -621595293), INT32_C(-2111598997)) },
   {  UINT8_C(174),
      easysimd_mm256_set_epi32(INT32_C( 2023987434), INT32_C( 1558325646), INT32_C( 2137381681), INT32_C(-1489350015),
                            INT32_C(-2044242394), INT32_C(  856733879), INT32_C( 1335704151), INT32_C(-1346912573)),
      easysimd_mm256_set_epi32(INT32_C( 2023987434), INT32_C(          0), INT32_C( 2137381681), INT32_C(          0),
                            INT32_C(-2044242394), INT32_C(  856733879), INT32_C( 1335704151), INT32_C(          0)) },
   {  UINT8_C(179),
      easysimd_mm256_set_epi32(INT32_C( 1148504404), INT32_C( -491209584), INT32_C( -163352510), INT32_C(  998745259),
                            INT32_C(-1986870978), INT32_C(  -69159531), INT32_C(-1702010863), INT32_C( -273027352)),
      easysimd_mm256_set_epi32(INT32_C( 1148504404), INT32_C(          0), INT32_C( -163352510), INT32_C(  998745259),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1702010863), INT32_C( -273027352)) },
   {  UINT8_C(187),
      easysimd_mm256_set_epi32(INT32_C( -272101695), INT32_C(-1695498890), INT32_C(  700753329), INT32_C(-1444122689),
                            INT32_C(  460626918), INT32_C( 1352716216), INT32_C( -651553055), INT32_C(-1336685992)),
      easysimd_mm256_set_epi32(INT32_C( -272101695), INT32_C(          0), INT32_C(  700753329), INT32_C(-1444122689),
                            INT32_C(  460626918), INT32_C(          0), INT32_C( -651553055), INT32_C(-1336685992)) },
   {  UINT8_C(119),
      easysimd_mm256_set_epi32(INT32_C(-1143505851), INT32_C(  669916850), INT32_C( -262251672), INT32_C(  470970928),
                            INT32_C( 1041120150), INT32_C(-1070284133), INT32_C(  347280872), INT32_C( -305201154)),
      easysimd_mm256_set_epi32(INT32_C(          0), INT32_C(  669916850), INT32_C( -262251672), INT32_C(  470970928),
                            INT32_C(          0), INT32_C(-1070284133), INT32_C(  347280872), INT32_C( -305201154)) },
   {  UINT8_C( 36),
      easysimd_mm256_set_epi32(INT32_C( 1927265424), INT32_C(-1184012473), INT32_C( 1473357439), INT32_C( 1217146407),
                            INT32_C( 1884345776), INT32_C( -662443681), INT32_C( -457310112), INT32_C(-2074706314)),
      easysimd_mm256_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C( 1473357439), INT32_C(          0),
                            INT32_C(          0), INT32_C( -662443681), INT32_C(          0), INT32_C(          0)) },
   {  UINT8_C(161),
      easysimd_mm256_set_epi32(INT32_C(  454256305), INT32_C(  -89518858), INT32_C(  575434377), INT32_C( -363661293),
                            INT32_C( -271203820), INT32_C( -624953581), INT32_C( 1626853978), INT32_C(-1012779406)),
      easysimd_mm256_set_epi32(INT32_C(  454256305), INT32_C(          0), INT32_C(  575434377), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-1012779406)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mov_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mov_epi32");
    easysimd_assert_m256i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_mov_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
  {  UINT8_C(109),
     easysimd_mm256_set_epi64x(INT64_C( 7572002691338055356), INT64_C(-6931202421771137023),
                             INT64_C(-6376895216110561530), INT64_C(  101010879856088318)),
      easysimd_mm256_set_epi64x(INT64_C( 7572002691338055356), INT64_C(-6931202421771137023),
                             INT64_C(                   0), INT64_C(  101010879856088318)) },
  {  UINT8_C( 84),
     easysimd_mm256_set_epi64x(INT64_C( 4863930517396634884), INT64_C( 1339559436234782312),
                             INT64_C(-4687477333083103994), INT64_C( 2317514132307922590)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 1339559436234782312),
                             INT64_C(                   0), INT64_C(                   0)) },
  {  UINT8_C(  4),
     easysimd_mm256_set_epi64x(INT64_C(-4280812707612271736), INT64_C( 1352195411881071619),
                             INT64_C( 4401292390121558915), INT64_C( 1447000045443016421)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 1352195411881071619),
                             INT64_C(                   0), INT64_C(                   0)) },
  {  UINT8_C(243),
     easysimd_mm256_set_epi64x(INT64_C(-1554191220639548558), INT64_C(-1009828379214636119),
                             INT64_C(   87598411827204486), INT64_C( 8494576712865778531)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(                   0),
                             INT64_C(   87598411827204486), INT64_C( 8494576712865778531)) },
  {  UINT8_C(102),
     easysimd_mm256_set_epi64x(INT64_C(-3199853677394167840), INT64_C(-8026951806327199947),
                             INT64_C( 4533073424512347513), INT64_C( -348644671563309757)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-8026951806327199947),
                             INT64_C( 4533073424512347513), INT64_C(                   0)) },
  {  UINT8_C( 38),
     easysimd_mm256_set_epi64x(INT64_C(-8077475266882793195), INT64_C(-1380937485015239307),
                             INT64_C(-3426685195142795196), INT64_C( 4855530362388048180)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(-1380937485015239307),
                             INT64_C(-3426685195142795196), INT64_C(                   0)) },
  {  UINT8_C(232),
     easysimd_mm256_set_epi64x(INT64_C(-4833519388014243665), INT64_C( 2573974298093740422),
                             INT64_C( 3628954985408843732), INT64_C(-4157981558961121913)),
      easysimd_mm256_set_epi64x(INT64_C(-4833519388014243665), INT64_C(                   0),
                             INT64_C(                   0), INT64_C(                   0)) },
  {  UINT8_C(158),
     easysimd_mm256_set_epi64x(INT64_C( 8860262502878217231), INT64_C(-7256652440967705311),
                             INT64_C( 8973660985157671450), INT64_C(-1395962117275720873)),
      easysimd_mm256_set_epi64x(INT64_C( 8860262502878217231), INT64_C(-7256652440967705311),
                             INT64_C( 8973660985157671450), INT64_C(                   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mov_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mov_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_mov_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m256d a;
    easysimd__m256d r;
  } test_vec[8] = {
    {  UINT8_C(156),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -797.63), EASYSIMD_FLOAT64_C(  550.96),
                         EASYSIMD_FLOAT64_C(  215.70), EASYSIMD_FLOAT64_C(  -51.73)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -797.63), EASYSIMD_FLOAT64_C(  550.96),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
   {  UINT8_C(232),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  603.95), EASYSIMD_FLOAT64_C(   89.69),
                         EASYSIMD_FLOAT64_C(  726.92), EASYSIMD_FLOAT64_C(  286.27)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  603.95), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
   {  UINT8_C(  7),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -753.25), EASYSIMD_FLOAT64_C(  973.27),
                         EASYSIMD_FLOAT64_C(  154.94), EASYSIMD_FLOAT64_C(  621.42)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  973.27),
                         EASYSIMD_FLOAT64_C(  154.94), EASYSIMD_FLOAT64_C(  621.42)) },
   {  UINT8_C( 98),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -339.86), EASYSIMD_FLOAT64_C( -506.40),
                         EASYSIMD_FLOAT64_C(  409.52), EASYSIMD_FLOAT64_C(  202.83)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  409.52), EASYSIMD_FLOAT64_C(    0.00)) },
   {  UINT8_C( 85),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  205.42), EASYSIMD_FLOAT64_C( -996.69),
                         EASYSIMD_FLOAT64_C( -560.92), EASYSIMD_FLOAT64_C(  347.34)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -996.69),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  347.34)) },
   {  UINT8_C(149),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  226.47), EASYSIMD_FLOAT64_C(  459.36),
                         EASYSIMD_FLOAT64_C(  864.34), EASYSIMD_FLOAT64_C( -365.19)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  459.36),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -365.19)) },
   {  UINT8_C( 67),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -158.13), EASYSIMD_FLOAT64_C( -903.74),
                         EASYSIMD_FLOAT64_C(  370.86), EASYSIMD_FLOAT64_C( -800.55)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  370.86), EASYSIMD_FLOAT64_C( -800.55)) },
   {  UINT8_C(168),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -868.95), EASYSIMD_FLOAT64_C(  674.80),
                         EASYSIMD_FLOAT64_C( -866.19), EASYSIMD_FLOAT64_C( -917.43)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -868.95), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = test_vec[i].a;
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mov_pd(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mov_pd");
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_mov_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m256 a;
    easysimd__m256 r;
  } test_vec[8] = {
    { UINT8_C(230),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -916.16), EASYSIMD_FLOAT32_C(   -17.54),
                         EASYSIMD_FLOAT32_C(    72.07), EASYSIMD_FLOAT32_C(   358.38),
                         EASYSIMD_FLOAT32_C(  -323.81), EASYSIMD_FLOAT32_C(  -500.50),
                         EASYSIMD_FLOAT32_C(  -957.58), EASYSIMD_FLOAT32_C(    95.32)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -916.16), EASYSIMD_FLOAT32_C(   -17.54),
                         EASYSIMD_FLOAT32_C(    72.07), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -500.50),
                         EASYSIMD_FLOAT32_C(  -957.58), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C(248),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   820.20), EASYSIMD_FLOAT32_C(  -882.62),
                         EASYSIMD_FLOAT32_C(   245.98), EASYSIMD_FLOAT32_C(   520.70),
                         EASYSIMD_FLOAT32_C(   947.17), EASYSIMD_FLOAT32_C(  -801.95),
                         EASYSIMD_FLOAT32_C(   523.33), EASYSIMD_FLOAT32_C(    88.74)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   820.20), EASYSIMD_FLOAT32_C(  -882.62),
                         EASYSIMD_FLOAT32_C(   245.98), EASYSIMD_FLOAT32_C(   520.70),
                         EASYSIMD_FLOAT32_C(   947.17), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C( 91),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   382.59), EASYSIMD_FLOAT32_C(  -104.90),
                         EASYSIMD_FLOAT32_C(   437.21), EASYSIMD_FLOAT32_C(   669.80),
                         EASYSIMD_FLOAT32_C(   475.78), EASYSIMD_FLOAT32_C(   291.58),
                         EASYSIMD_FLOAT32_C(   932.63), EASYSIMD_FLOAT32_C(    75.72)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -104.90),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   669.80),
                         EASYSIMD_FLOAT32_C(   475.78), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   932.63), EASYSIMD_FLOAT32_C(    75.72)) },
    { UINT8_C( 28),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   325.29), EASYSIMD_FLOAT32_C(    66.25),
                         EASYSIMD_FLOAT32_C(   309.27), EASYSIMD_FLOAT32_C(    48.25),
                         EASYSIMD_FLOAT32_C(  -685.79), EASYSIMD_FLOAT32_C(   793.84),
                         EASYSIMD_FLOAT32_C(   -42.51), EASYSIMD_FLOAT32_C(  -431.02)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    48.25),
                         EASYSIMD_FLOAT32_C(  -685.79), EASYSIMD_FLOAT32_C(   793.84),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C( 95),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -193.68), EASYSIMD_FLOAT32_C(  -614.23),
                         EASYSIMD_FLOAT32_C(   420.74), EASYSIMD_FLOAT32_C(   824.23),
                         EASYSIMD_FLOAT32_C(   818.32), EASYSIMD_FLOAT32_C(  -457.30),
                         EASYSIMD_FLOAT32_C(  -144.19), EASYSIMD_FLOAT32_C(    78.38)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -614.23),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   824.23),
                         EASYSIMD_FLOAT32_C(   818.32), EASYSIMD_FLOAT32_C(  -457.30),
                         EASYSIMD_FLOAT32_C(  -144.19), EASYSIMD_FLOAT32_C(    78.38)) },
    { UINT8_C(213),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -960.32), EASYSIMD_FLOAT32_C(  -433.91),
                         EASYSIMD_FLOAT32_C(   640.12), EASYSIMD_FLOAT32_C(   816.31),
                         EASYSIMD_FLOAT32_C(  -667.16), EASYSIMD_FLOAT32_C(  -891.50),
                         EASYSIMD_FLOAT32_C(   639.25), EASYSIMD_FLOAT32_C(   310.94)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -960.32), EASYSIMD_FLOAT32_C(  -433.91),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   816.31),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -891.50),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   310.94)) },
    { UINT8_C(210),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    20.36), EASYSIMD_FLOAT32_C(   -24.88),
                         EASYSIMD_FLOAT32_C(   118.89), EASYSIMD_FLOAT32_C(   166.69),
                         EASYSIMD_FLOAT32_C(   470.98), EASYSIMD_FLOAT32_C(  -195.06),
                         EASYSIMD_FLOAT32_C(  -643.26), EASYSIMD_FLOAT32_C(  -611.78)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    20.36), EASYSIMD_FLOAT32_C(   -24.88),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   166.69),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -643.26), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C(247),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     3.65), EASYSIMD_FLOAT32_C(   -38.51),
                         EASYSIMD_FLOAT32_C(  -896.47), EASYSIMD_FLOAT32_C(   773.97),
                         EASYSIMD_FLOAT32_C(  -241.05), EASYSIMD_FLOAT32_C(  -597.57),
                         EASYSIMD_FLOAT32_C(   632.97), EASYSIMD_FLOAT32_C(  -804.93)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     3.65), EASYSIMD_FLOAT32_C(   -38.51),
                         EASYSIMD_FLOAT32_C(  -896.47), EASYSIMD_FLOAT32_C(   773.97),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -597.57),
                         EASYSIMD_FLOAT32_C(   632.97), EASYSIMD_FLOAT32_C(  -804.93)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = test_vec[i].a;
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mov_ps(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mov_ps");
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mov_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT64_C( 5922492609958636327),
      easysimd_mm512_set_epi8(INT8_C( -97), INT8_C( -47), INT8_C( -93), INT8_C( -97),
                           INT8_C(   9), INT8_C( -55), INT8_C(-113), INT8_C(  98),
                           INT8_C( -56), INT8_C(  10), INT8_C( 103), INT8_C(  84),
                           INT8_C(  93), INT8_C(  24), INT8_C( -78), INT8_C(  35),
                           INT8_C( 125), INT8_C( -63), INT8_C(  19), INT8_C(   4),
                           INT8_C(   3), INT8_C( -87), INT8_C(  98), INT8_C(-113),
                           INT8_C(  23), INT8_C(-124), INT8_C( -87), INT8_C(  63),
                           INT8_C( 108), INT8_C( -18), INT8_C( -27), INT8_C(-127),
                           INT8_C( -60), INT8_C(  60), INT8_C( -56), INT8_C(   3),
                           INT8_C(-128), INT8_C( -62), INT8_C(  52), INT8_C( -74),
                           INT8_C( -87), INT8_C(  32), INT8_C(  46), INT8_C(-128),
                           INT8_C(  54), INT8_C( -19), INT8_C(  12), INT8_C(  22),
                           INT8_C( -94), INT8_C( -84), INT8_C( -58), INT8_C(  92),
                           INT8_C( -70), INT8_C( -25), INT8_C(  91), INT8_C( -45),
                           INT8_C(   5), INT8_C( 109), INT8_C( -46), INT8_C(  37),
                           INT8_C(   7), INT8_C(  44), INT8_C(  41), INT8_C(-106)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C( -47), INT8_C(   0), INT8_C( -97),
                           INT8_C(   0), INT8_C(   0), INT8_C(-113), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( 103), INT8_C(  84),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 125), INT8_C( -63), INT8_C(  19), INT8_C(   0),
                           INT8_C(   3), INT8_C(   0), INT8_C(  98), INT8_C(-113),
                           INT8_C(  23), INT8_C(   0), INT8_C( -87), INT8_C(   0),
                           INT8_C(   0), INT8_C( -18), INT8_C( -27), INT8_C(   0),
                           INT8_C(   0), INT8_C(  60), INT8_C( -56), INT8_C(   0),
                           INT8_C(-128), INT8_C(   0), INT8_C(   0), INT8_C( -74),
                           INT8_C(   0), INT8_C(  32), INT8_C(  46), INT8_C(-128),
                           INT8_C(   0), INT8_C( -19), INT8_C(   0), INT8_C(  22),
                           INT8_C( -94), INT8_C( -84), INT8_C( -58), INT8_C(   0),
                           INT8_C( -70), INT8_C(   0), INT8_C(  91), INT8_C( -45),
                           INT8_C(   0), INT8_C(   0), INT8_C( -46), INT8_C(   0),
                           INT8_C(   0), INT8_C(  44), INT8_C(  41), INT8_C(-106)) },
    { UINT64_C( 8924247995799830534),
      easysimd_mm512_set_epi8(INT8_C(  20), INT8_C(  23), INT8_C( 120), INT8_C( -13),
                           INT8_C(  82), INT8_C(  32), INT8_C( -44), INT8_C( -44),
                           INT8_C(  43), INT8_C( -65), INT8_C(  47), INT8_C(  36),
                           INT8_C(-101), INT8_C(   5), INT8_C( -76), INT8_C( -57),
                           INT8_C(  77), INT8_C(  48), INT8_C( -46), INT8_C( -15),
                           INT8_C(  78), INT8_C( 108), INT8_C( 114), INT8_C(  83),
                           INT8_C( -72), INT8_C(  21), INT8_C( 100), INT8_C( 121),
                           INT8_C(  29), INT8_C( -74), INT8_C(  81), INT8_C( -13),
                           INT8_C( -57), INT8_C( -17), INT8_C(  20), INT8_C(-109),
                           INT8_C( -87), INT8_C( 127), INT8_C(  92), INT8_C(-119),
                           INT8_C(  26), INT8_C( 123), INT8_C( -51), INT8_C( 109),
                           INT8_C(  30), INT8_C( -58), INT8_C(-117), INT8_C(  82),
                           INT8_C( 111), INT8_C( -10), INT8_C( -10), INT8_C( -68),
                           INT8_C(  -4), INT8_C(  -7), INT8_C( 117), INT8_C(  92),
                           INT8_C(  94), INT8_C( -65), INT8_C( 109), INT8_C(  81),
                           INT8_C( -71), INT8_C( -46), INT8_C( 113), INT8_C(   9)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(  23), INT8_C( 120), INT8_C( -13),
                           INT8_C(  82), INT8_C(   0), INT8_C( -44), INT8_C( -44),
                           INT8_C(  43), INT8_C( -65), INT8_C(   0), INT8_C(  36),
                           INT8_C(-101), INT8_C(   0), INT8_C(   0), INT8_C( -57),
                           INT8_C(   0), INT8_C(  48), INT8_C(   0), INT8_C(   0),
                           INT8_C(  78), INT8_C( 108), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  21), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -74), INT8_C(   0), INT8_C(   0),
                           INT8_C( -57), INT8_C( -17), INT8_C(  20), INT8_C(-109),
                           INT8_C( -87), INT8_C( 127), INT8_C(   0), INT8_C(-119),
                           INT8_C(   0), INT8_C(   0), INT8_C( -51), INT8_C(   0),
                           INT8_C(   0), INT8_C( -58), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -46), INT8_C( 113), INT8_C(   0)) },
    { UINT64_C( 9822575649644588350),
      easysimd_mm512_set_epi8(INT8_C(  62), INT8_C(  34), INT8_C(-113), INT8_C( 121),
                           INT8_C( -31), INT8_C(-111), INT8_C(  76), INT8_C(-113),
                           INT8_C(  72), INT8_C( -37), INT8_C(  44), INT8_C( -67),
                           INT8_C(-103), INT8_C(  31), INT8_C(  89), INT8_C( 120),
                           INT8_C( -25), INT8_C(-127), INT8_C(  40), INT8_C( -10),
                           INT8_C(  75), INT8_C(-123), INT8_C(  78), INT8_C(  -2),
                           INT8_C( -83), INT8_C( -74), INT8_C( -51), INT8_C(  46),
                           INT8_C(  60), INT8_C( -39), INT8_C( 124), INT8_C(-117),
                           INT8_C(  70), INT8_C(  66), INT8_C( -35), INT8_C( -51),
                           INT8_C( -64), INT8_C( -61), INT8_C(-113), INT8_C(   2),
                           INT8_C(  -4), INT8_C( -72), INT8_C( 113), INT8_C( -63),
                           INT8_C( -49), INT8_C(  70), INT8_C( -50), INT8_C(  52),
                           INT8_C(   0), INT8_C(  13), INT8_C(  74), INT8_C( -60),
                           INT8_C( 103), INT8_C(  -7), INT8_C( -61), INT8_C( -37),
                           INT8_C( -79), INT8_C( -77), INT8_C( -81), INT8_C( -83),
                           INT8_C(  94), INT8_C(  52), INT8_C( -73), INT8_C(  76)),
      easysimd_mm512_set_epi8(INT8_C(  62), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -31), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -37), INT8_C(   0), INT8_C( -67),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -25), INT8_C(-127), INT8_C(   0), INT8_C(   0),
                           INT8_C(  75), INT8_C(-123), INT8_C(   0), INT8_C(   0),
                           INT8_C( -83), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( 124), INT8_C(   0),
                           INT8_C(  70), INT8_C(  66), INT8_C(   0), INT8_C( -51),
                           INT8_C( -64), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -72), INT8_C( 113), INT8_C( -63),
                           INT8_C(   0), INT8_C(  70), INT8_C( -50), INT8_C(  52),
                           INT8_C(   0), INT8_C(   0), INT8_C(  74), INT8_C(   0),
                           INT8_C( 103), INT8_C(  -7), INT8_C(   0), INT8_C( -37),
                           INT8_C(   0), INT8_C(   0), INT8_C( -81), INT8_C( -83),
                           INT8_C(  94), INT8_C(  52), INT8_C( -73), INT8_C(   0)) },
    { UINT64_C( 1775653069823307747),
      easysimd_mm512_set_epi8(INT8_C( -85), INT8_C(  90), INT8_C(  97), INT8_C( -38),
                           INT8_C( -35), INT8_C(   6), INT8_C(  37), INT8_C( 106),
                           INT8_C( 102), INT8_C( 109), INT8_C(  47), INT8_C(  29),
                           INT8_C( -81), INT8_C(-113), INT8_C( -49), INT8_C(  18),
                           INT8_C( -68), INT8_C( 121), INT8_C(-102), INT8_C( -30),
                           INT8_C(-103), INT8_C( -31), INT8_C(  24), INT8_C( -55),
                           INT8_C(  -5), INT8_C(   8), INT8_C( -38), INT8_C(  37),
                           INT8_C(  15), INT8_C(-120), INT8_C(  17), INT8_C( -63),
                           INT8_C( 107), INT8_C( -41), INT8_C( -53), INT8_C(-107),
                           INT8_C(  91), INT8_C(  -9), INT8_C(-127), INT8_C( -39),
                           INT8_C( 105), INT8_C( -27), INT8_C(  96), INT8_C( -96),
                           INT8_C(   2), INT8_C(  44), INT8_C(  11), INT8_C( -43),
                           INT8_C( -52), INT8_C( 126), INT8_C( 125), INT8_C( 121),
                           INT8_C(  87), INT8_C( -95), INT8_C( 120), INT8_C( -46),
                           INT8_C(  25), INT8_C(  71), INT8_C( 117), INT8_C(  47),
                           INT8_C(-110), INT8_C( -87), INT8_C( -36), INT8_C(  25)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -38),
                           INT8_C( -35), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 102), INT8_C(   0), INT8_C(  47), INT8_C(   0),
                           INT8_C(   0), INT8_C(-113), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( 121), INT8_C(-102), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  24), INT8_C( -55),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  15), INT8_C(-120), INT8_C(  17), INT8_C( -63),
                           INT8_C(   0), INT8_C(   0), INT8_C( -53), INT8_C(   0),
                           INT8_C(   0), INT8_C(  -9), INT8_C(-127), INT8_C( -39),
                           INT8_C(   0), INT8_C(   0), INT8_C(  96), INT8_C(   0),
                           INT8_C(   0), INT8_C(  44), INT8_C(  11), INT8_C(   0),
                           INT8_C(   0), INT8_C( 126), INT8_C( 125), INT8_C(   0),
                           INT8_C(  87), INT8_C( -95), INT8_C( 120), INT8_C( -46),
                           INT8_C(  25), INT8_C(  71), INT8_C( 117), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -36), INT8_C(  25)) },
    { UINT64_C( 1039866445453672381),
      easysimd_mm512_set_epi8(INT8_C( -73), INT8_C( -64), INT8_C( -86), INT8_C( -65),
                           INT8_C( 124), INT8_C(-109), INT8_C(  79), INT8_C(-111),
                           INT8_C(  64), INT8_C( -98), INT8_C(  -1), INT8_C( -43),
                           INT8_C(  -4), INT8_C(  72), INT8_C( 108), INT8_C( -95),
                           INT8_C( -23), INT8_C( -29), INT8_C(-113), INT8_C(  47),
                           INT8_C( 114), INT8_C(-122), INT8_C( -91), INT8_C( -20),
                           INT8_C( 117), INT8_C(   0), INT8_C( -58), INT8_C( -82),
                           INT8_C( -40), INT8_C( -36), INT8_C( -30), INT8_C( -56),
                           INT8_C( -68), INT8_C( -93), INT8_C(  25), INT8_C( -68),
                           INT8_C(   8), INT8_C(  64), INT8_C( -70), INT8_C( -19),
                           INT8_C( -64), INT8_C( -54), INT8_C( 120), INT8_C(  61),
                           INT8_C( -73), INT8_C(  47), INT8_C(-113), INT8_C(  68),
                           INT8_C( -44), INT8_C( -96), INT8_C(-106), INT8_C( -68),
                           INT8_C(  75), INT8_C( -42), INT8_C(  94), INT8_C( -68),
                           INT8_C( -10), INT8_C(  41), INT8_C( -90), INT8_C(-110),
                           INT8_C(-116), INT8_C( -51), INT8_C( -75), INT8_C( 102)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 124), INT8_C(-109), INT8_C(  79), INT8_C(   0),
                           INT8_C(   0), INT8_C( -98), INT8_C(  -1), INT8_C(   0),
                           INT8_C(  -4), INT8_C(  72), INT8_C( 108), INT8_C(   0),
                           INT8_C(   0), INT8_C( -29), INT8_C(   0), INT8_C(  47),
                           INT8_C( 114), INT8_C(   0), INT8_C(   0), INT8_C( -20),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -36), INT8_C(   0), INT8_C( -56),
                           INT8_C( -68), INT8_C( -93), INT8_C(   0), INT8_C(   0),
                           INT8_C(   8), INT8_C(  64), INT8_C( -70), INT8_C( -19),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  61),
                           INT8_C( -73), INT8_C(  47), INT8_C(   0), INT8_C(  68),
                           INT8_C(   0), INT8_C(   0), INT8_C(-106), INT8_C( -68),
                           INT8_C(  75), INT8_C( -42), INT8_C(  94), INT8_C( -68),
                           INT8_C( -10), INT8_C(   0), INT8_C( -90), INT8_C(-110),
                           INT8_C(-116), INT8_C( -51), INT8_C(   0), INT8_C( 102)) },
    { UINT64_C(16454496682655599925),
      easysimd_mm512_set_epi8(INT8_C(  41), INT8_C(  -4), INT8_C( -19), INT8_C(  26),
                           INT8_C(-127), INT8_C( -41), INT8_C(  14), INT8_C(  10),
                           INT8_C(  63), INT8_C(  99), INT8_C(  51), INT8_C(-115),
                           INT8_C( 118), INT8_C( -85), INT8_C(-111), INT8_C(  19),
                           INT8_C(  43), INT8_C( -97), INT8_C( 107), INT8_C( 127),
                           INT8_C(-100), INT8_C(  45), INT8_C( -77), INT8_C(  77),
                           INT8_C( -53), INT8_C( -59), INT8_C(  -6), INT8_C( -57),
                           INT8_C(  97), INT8_C(  68), INT8_C( -67), INT8_C( 117),
                           INT8_C( -92), INT8_C(  -3), INT8_C(   2), INT8_C(  59),
                           INT8_C(  53), INT8_C( -13), INT8_C( -31), INT8_C(  47),
                           INT8_C( -33), INT8_C(  67), INT8_C( -43), INT8_C( -53),
                           INT8_C( -52), INT8_C(  -3), INT8_C(  85), INT8_C(  48),
                           INT8_C( -45), INT8_C( -72), INT8_C(  96), INT8_C(  85),
                           INT8_C(  81), INT8_C(  28), INT8_C( -50), INT8_C(-107),
                           INT8_C( -56), INT8_C( -85), INT8_C( -83), INT8_C( -25),
                           INT8_C(  78), INT8_C(  13), INT8_C(  41), INT8_C(  86)),
      easysimd_mm512_set_epi8(INT8_C(  41), INT8_C(  -4), INT8_C( -19), INT8_C(   0),
                           INT8_C(   0), INT8_C( -41), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  99), INT8_C(   0), INT8_C(-115),
                           INT8_C( 118), INT8_C(   0), INT8_C(-111), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( 127),
                           INT8_C(-100), INT8_C(  45), INT8_C(   0), INT8_C(  77),
                           INT8_C( -53), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  97), INT8_C(  68), INT8_C(   0), INT8_C( 117),
                           INT8_C( -92), INT8_C(   0), INT8_C(   0), INT8_C(  59),
                           INT8_C(  53), INT8_C( -13), INT8_C( -31), INT8_C(  47),
                           INT8_C( -33), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  -3), INT8_C(  85), INT8_C(  48),
                           INT8_C( -45), INT8_C( -72), INT8_C(   0), INT8_C(   0),
                           INT8_C(  81), INT8_C(  28), INT8_C(   0), INT8_C(-107),
                           INT8_C(   0), INT8_C(   0), INT8_C( -83), INT8_C( -25),
                           INT8_C(   0), INT8_C(  13), INT8_C(   0), INT8_C(  86)) },
    { UINT64_C( 6096298775549734718),
      easysimd_mm512_set_epi8(INT8_C( -35), INT8_C( -29), INT8_C(  89), INT8_C(  91),
                           INT8_C( -53), INT8_C( -62), INT8_C( 107), INT8_C( -42),
                           INT8_C(-115), INT8_C(  52), INT8_C( -17), INT8_C(  64),
                           INT8_C(-105), INT8_C(-106), INT8_C(  65), INT8_C(  97),
                           INT8_C(  85), INT8_C(  52), INT8_C( -17), INT8_C(   6),
                           INT8_C( -73), INT8_C( 109), INT8_C(  99), INT8_C(   9),
                           INT8_C(  43), INT8_C(-102), INT8_C( 112), INT8_C(  24),
                           INT8_C(-121), INT8_C(  58), INT8_C( -73), INT8_C( -15),
                           INT8_C(  37), INT8_C( 104), INT8_C( -81), INT8_C( 113),
                           INT8_C(  31), INT8_C( -10), INT8_C( -32), INT8_C( -91),
                           INT8_C(  51), INT8_C( -51), INT8_C(  60), INT8_C(  38),
                           INT8_C(  -1), INT8_C( -38), INT8_C(   2), INT8_C( 110),
                           INT8_C( -61), INT8_C(  91), INT8_C( -50), INT8_C(  89),
                           INT8_C(  27), INT8_C( -13), INT8_C( 111), INT8_C( -20),
                           INT8_C(  51), INT8_C( -66), INT8_C( -26), INT8_C(  66),
                           INT8_C(  45), INT8_C( -59), INT8_C( -45), INT8_C(-102)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C( -29), INT8_C(   0), INT8_C(  91),
                           INT8_C(   0), INT8_C( -62), INT8_C(   0), INT8_C(   0),
                           INT8_C(-115), INT8_C(   0), INT8_C(   0), INT8_C(  64),
                           INT8_C(-105), INT8_C(   0), INT8_C(  65), INT8_C(   0),
                           INT8_C(   0), INT8_C(  52), INT8_C( -17), INT8_C(   0),
                           INT8_C(   0), INT8_C( 109), INT8_C(  99), INT8_C(   9),
                           INT8_C(   0), INT8_C(-102), INT8_C( 112), INT8_C(   0),
                           INT8_C(-121), INT8_C(  58), INT8_C( -73), INT8_C( -15),
                           INT8_C(  37), INT8_C( 104), INT8_C(   0), INT8_C( 113),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -91),
                           INT8_C(   0), INT8_C( -51), INT8_C(   0), INT8_C(   0),
                           INT8_C(  -1), INT8_C(   0), INT8_C(   2), INT8_C(   0),
                           INT8_C(   0), INT8_C(  91), INT8_C( -50), INT8_C(   0),
                           INT8_C(  27), INT8_C( -13), INT8_C( 111), INT8_C( -20),
                           INT8_C(   0), INT8_C(   0), INT8_C( -26), INT8_C(  66),
                           INT8_C(  45), INT8_C( -59), INT8_C( -45), INT8_C(   0)) },
    { UINT64_C( 4696357732069093821),
      easysimd_mm512_set_epi8(INT8_C(  40), INT8_C( -99), INT8_C(   6), INT8_C( 105),
                           INT8_C(  36), INT8_C( -85), INT8_C(  62), INT8_C( 102),
                           INT8_C(  23), INT8_C(-110), INT8_C( -92), INT8_C(  59),
                           INT8_C(  17), INT8_C( -54), INT8_C(   5), INT8_C(  81),
                           INT8_C( -71), INT8_C(  68), INT8_C( 114), INT8_C( -60),
                           INT8_C(  39), INT8_C( -49), INT8_C( -84), INT8_C( 114),
                           INT8_C( -81), INT8_C( 122), INT8_C(  97), INT8_C( -16),
                           INT8_C(  21), INT8_C( -76), INT8_C( -80), INT8_C( -61),
                           INT8_C( -47), INT8_C( -86), INT8_C(  35), INT8_C(-110),
                           INT8_C(  95), INT8_C(  -9), INT8_C(  86), INT8_C(   9),
                           INT8_C(  31), INT8_C(  48), INT8_C(  63), INT8_C(  -6),
                           INT8_C( -36), INT8_C( -47), INT8_C(  95), INT8_C( -20),
                           INT8_C(  21), INT8_C(  -9), INT8_C(  -2), INT8_C(  26),
                           INT8_C(  63), INT8_C(  36), INT8_C( -33), INT8_C(  58),
                           INT8_C( -40), INT8_C( 106), INT8_C(   2), INT8_C( -51),
                           INT8_C( -13), INT8_C( -76), INT8_C( -77), INT8_C( -77)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C( -99), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( 102),
                           INT8_C(   0), INT8_C(   0), INT8_C( -92), INT8_C(   0),
                           INT8_C(  17), INT8_C( -54), INT8_C(   0), INT8_C(   0),
                           INT8_C( -71), INT8_C(  68), INT8_C(   0), INT8_C( -60),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( 122), INT8_C(  97), INT8_C( -16),
                           INT8_C(  21), INT8_C(   0), INT8_C(   0), INT8_C( -61),
                           INT8_C( -47), INT8_C(   0), INT8_C(   0), INT8_C(-110),
                           INT8_C(   0), INT8_C(  -9), INT8_C(  86), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  63), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  95), INT8_C( -20),
                           INT8_C(   0), INT8_C(   0), INT8_C(  -2), INT8_C(  26),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  58),
                           INT8_C( -40), INT8_C(   0), INT8_C(   2), INT8_C( -51),
                           INT8_C( -13), INT8_C( -76), INT8_C(   0), INT8_C( -77)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mov_epi8(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mov_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mov_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask32 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
     {  UINT32_C(4000530422),
      easysimd_mm512_set_epi16(INT16_C( -5942), INT16_C( 25831), INT16_C(-28539), INT16_C( -1873),
                            INT16_C(-13655), INT16_C( 26989), INT16_C( 16263), INT16_C( 13938),
                            INT16_C( 10041), INT16_C( 23778), INT16_C(-21283), INT16_C( 18765),
                            INT16_C(-14856), INT16_C( 31462), INT16_C(-19403), INT16_C( 32735),
                            INT16_C(  7009), INT16_C( 10124), INT16_C(-25355), INT16_C(-21077),
                            INT16_C(  1261), INT16_C( -9315), INT16_C(-20637), INT16_C( 25513),
                            INT16_C(  5169), INT16_C( 28434), INT16_C(-28809), INT16_C( -8631),
                            INT16_C(-18627), INT16_C( 25166), INT16_C( -8628), INT16_C( 28868)),
      easysimd_mm512_set_epi16(INT16_C( -5942), INT16_C( 25831), INT16_C(-28539), INT16_C(     0),
                            INT16_C(-13655), INT16_C( 26989), INT16_C( 16263), INT16_C(     0),
                            INT16_C(     0), INT16_C( 23778), INT16_C(-21283), INT16_C( 18765),
                            INT16_C(     0), INT16_C(     0), INT16_C(-19403), INT16_C( 32735),
                            INT16_C(     0), INT16_C(     0), INT16_C(-25355), INT16_C(-21077),
                            INT16_C(  1261), INT16_C( -9315), INT16_C(-20637), INT16_C( 25513),
                            INT16_C(  5169), INT16_C( 28434), INT16_C(-28809), INT16_C( -8631),
                            INT16_C(     0), INT16_C( 25166), INT16_C( -8628), INT16_C(     0)) },
   {  UINT32_C(4070875154),
      easysimd_mm512_set_epi16(INT16_C(-12225), INT16_C( 21369), INT16_C(  -119), INT16_C(-28694),
                            INT16_C(-23457), INT16_C(-22727), INT16_C(-11767), INT16_C(-23853),
                            INT16_C(-22479), INT16_C( 23784), INT16_C( -5275), INT16_C(-13228),
                            INT16_C(-17789), INT16_C(-22944), INT16_C(-14595), INT16_C(-10966),
                            INT16_C( -2247), INT16_C(-10276), INT16_C( 27089), INT16_C(-12303),
                            INT16_C( 28587), INT16_C(-26891), INT16_C( 24467), INT16_C( 22569),
                            INT16_C( 14745), INT16_C(-19983), INT16_C( 19001), INT16_C( 25844),
                            INT16_C(-17171), INT16_C( -2706), INT16_C( -6907), INT16_C( 24391)),
      easysimd_mm512_set_epi16(INT16_C(-12225), INT16_C( 21369), INT16_C(  -119), INT16_C(-28694),
                            INT16_C(     0), INT16_C(     0), INT16_C(-11767), INT16_C(     0),
                            INT16_C(-22479), INT16_C(     0), INT16_C( -5275), INT16_C(     0),
                            INT16_C(     0), INT16_C(-22944), INT16_C(     0), INT16_C(     0),
                            INT16_C( -2247), INT16_C(     0), INT16_C( 27089), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C( 25844),
                            INT16_C(     0), INT16_C(     0), INT16_C( -6907), INT16_C(     0)) },
   {  UINT32_C(3446806878),
      easysimd_mm512_set_epi16(INT16_C( -8234), INT16_C( 29915), INT16_C(-15715), INT16_C(  6824),
                            INT16_C( 15576), INT16_C( 19574), INT16_C( 28649), INT16_C(  3361),
                            INT16_C(-14218), INT16_C(  3388), INT16_C( -7950), INT16_C( 14208),
                            INT16_C(-11822), INT16_C(-15586), INT16_C( 22828), INT16_C(-12231),
                            INT16_C(  1557), INT16_C( 15030), INT16_C(-21739), INT16_C(  9138),
                            INT16_C(-18261), INT16_C( 26404), INT16_C(-17358), INT16_C(   811),
                            INT16_C( -9806), INT16_C(-30299), INT16_C( 28809), INT16_C( 31831),
                            INT16_C(-23257), INT16_C(  4576), INT16_C( -7556), INT16_C(  7253)),
      easysimd_mm512_set_epi16(INT16_C( -8234), INT16_C( 29915), INT16_C(     0), INT16_C(     0),
                            INT16_C( 15576), INT16_C( 19574), INT16_C(     0), INT16_C(  3361),
                            INT16_C(     0), INT16_C(  3388), INT16_C( -7950), INT16_C( 14208),
                            INT16_C(     0), INT16_C(     0), INT16_C( 22828), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(  9138),
                            INT16_C(-18261), INT16_C(     0), INT16_C(     0), INT16_C(   811),
                            INT16_C(     0), INT16_C(-30299), INT16_C(     0), INT16_C( 31831),
                            INT16_C(-23257), INT16_C(  4576), INT16_C( -7556), INT16_C(     0)) },
   {  UINT32_C( 343900451),
      easysimd_mm512_set_epi16(INT16_C(-31095), INT16_C(-23377), INT16_C( -4662), INT16_C(-21413),
                            INT16_C( 30429), INT16_C(  8769), INT16_C(-28068), INT16_C( 27084),
                            INT16_C( 27030), INT16_C(-23477), INT16_C(-21313), INT16_C(-17124),
                            INT16_C(-18222), INT16_C( 32522), INT16_C( 29282), INT16_C( 28924),
                            INT16_C( 27441), INT16_C(-21554), INT16_C( -5444), INT16_C( 30253),
                            INT16_C(-29783), INT16_C(-26663), INT16_C(-11174), INT16_C(-25779),
                            INT16_C(-27773), INT16_C( 24626), INT16_C( -6955), INT16_C(-12302),
                            INT16_C( 26319), INT16_C(-26837), INT16_C(-16192), INT16_C( 18933)),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(-21413),
                            INT16_C(     0), INT16_C(  8769), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(-23477), INT16_C(-21313), INT16_C(-17124),
                            INT16_C(-18222), INT16_C( 32522), INT16_C( 29282), INT16_C( 28924),
                            INT16_C( 27441), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(-25779),
                            INT16_C(     0), INT16_C(     0), INT16_C( -6955), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(-16192), INT16_C( 18933)) },
   {  UINT32_C( 610198307),
      easysimd_mm512_set_epi16(INT16_C( 28784), INT16_C(-14670), INT16_C(-17599), INT16_C(-27901),
                            INT16_C( 10638), INT16_C(-12065), INT16_C( 21050), INT16_C( 10287),
                            INT16_C( 11470), INT16_C( -4598), INT16_C(   -40), INT16_C( 28251),
                            INT16_C(-10212), INT16_C(-29606), INT16_C( -6193), INT16_C(  2935),
                            INT16_C(-16438), INT16_C(   971), INT16_C(  3225), INT16_C( 17346),
                            INT16_C( 28916), INT16_C( 25171), INT16_C( 10807), INT16_C(  1473),
                            INT16_C( 15813), INT16_C( 32427), INT16_C(-23468), INT16_C(-21533),
                            INT16_C( 13263), INT16_C(-22199), INT16_C( 13682), INT16_C(   226)),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(-17599), INT16_C(     0),
                            INT16_C(     0), INT16_C(-12065), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C( -4598), INT16_C(     0), INT16_C( 28251),
                            INT16_C(-10212), INT16_C(-29606), INT16_C( -6193), INT16_C(     0),
                            INT16_C(-16438), INT16_C(   971), INT16_C(  3225), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C( 10807), INT16_C(  1473),
                            INT16_C(     0), INT16_C(     0), INT16_C(-23468), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C( 13682), INT16_C(   226)) },
   {  UINT32_C(1063632252),
      easysimd_mm512_set_epi16(INT16_C( -3533), INT16_C(-15311), INT16_C( -9164), INT16_C(-27075),
                            INT16_C( 30377), INT16_C( 29218), INT16_C( -8851), INT16_C(-29072),
                            INT16_C( 28941), INT16_C( -5458), INT16_C( 29621), INT16_C( 18538),
                            INT16_C(-22601), INT16_C( 13017), INT16_C( 26323), INT16_C(  2952),
                            INT16_C(-17536), INT16_C( 11831), INT16_C( 27487), INT16_C( 29413),
                            INT16_C(  5506), INT16_C( -8406), INT16_C( 23534), INT16_C( 31484),
                            INT16_C( 17532), INT16_C( 11364), INT16_C( 26550), INT16_C(-26724),
                            INT16_C( 23828), INT16_C(-27226), INT16_C(-30955), INT16_C( 28791)),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C( -9164), INT16_C(-27075),
                            INT16_C( 30377), INT16_C( 29218), INT16_C( -8851), INT16_C(-29072),
                            INT16_C(     0), INT16_C( -5458), INT16_C( 29621), INT16_C(     0),
                            INT16_C(     0), INT16_C( 13017), INT16_C(     0), INT16_C(  2952),
                            INT16_C(-17536), INT16_C(     0), INT16_C( 27487), INT16_C( 29413),
                            INT16_C(  5506), INT16_C( -8406), INT16_C(     0), INT16_C( 31484),
                            INT16_C(     0), INT16_C( 11364), INT16_C( 26550), INT16_C(-26724),
                            INT16_C( 23828), INT16_C(-27226), INT16_C(     0), INT16_C(     0)) },
   {  UINT32_C(2981066031),
      easysimd_mm512_set_epi16(INT16_C(-15776), INT16_C( 14598), INT16_C( -3252), INT16_C( 10125),
                            INT16_C( 14481), INT16_C( 12166), INT16_C(  2171), INT16_C( 29452),
                            INT16_C(-31285), INT16_C( 18516), INT16_C( 27776), INT16_C( 10973),
                            INT16_C(-32618), INT16_C(  -356), INT16_C( 12910), INT16_C(  2992),
                            INT16_C( -3498), INT16_C( -2944), INT16_C(-21668), INT16_C(  2835),
                            INT16_C(-13850), INT16_C(-21988), INT16_C(  9656), INT16_C( 32264),
                            INT16_C( 12816), INT16_C( 21193), INT16_C(-25247), INT16_C( -7370),
                            INT16_C(  5319), INT16_C(  5949), INT16_C(  9112), INT16_C( -1168)),
      easysimd_mm512_set_epi16(INT16_C(-15776), INT16_C(     0), INT16_C( -3252), INT16_C( 10125),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C( 29452),
                            INT16_C(-31285), INT16_C(     0), INT16_C( 27776), INT16_C(     0),
                            INT16_C(-32618), INT16_C(  -356), INT16_C( 12910), INT16_C(  2992),
                            INT16_C(     0), INT16_C( -2944), INT16_C(-21668), INT16_C(  2835),
                            INT16_C(     0), INT16_C(-21988), INT16_C(     0), INT16_C( 32264),
                            INT16_C(     0), INT16_C(     0), INT16_C(-25247), INT16_C(     0),
                            INT16_C(  5319), INT16_C(  5949), INT16_C(  9112), INT16_C( -1168)) },
   {  UINT32_C( 623103106),
      easysimd_mm512_set_epi16(INT16_C( 26221), INT16_C(  1202), INT16_C(-22573), INT16_C( 25677),
                            INT16_C( -9440), INT16_C( -3817), INT16_C(-15802), INT16_C( 26698),
                            INT16_C( 26873), INT16_C(  4596), INT16_C(-15991), INT16_C( 14118),
                            INT16_C( -7802), INT16_C( 10352), INT16_C( 27984), INT16_C(  1876),
                            INT16_C( 14808), INT16_C(-10243), INT16_C(  2806), INT16_C(  5765),
                            INT16_C(-26054), INT16_C( 23235), INT16_C(-10396), INT16_C(-11996),
                            INT16_C(-32195), INT16_C(-16209), INT16_C(-27816), INT16_C(-28484),
                            INT16_C(-29121), INT16_C(  7946), INT16_C( -1915), INT16_C(  9449)),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(-22573), INT16_C(     0),
                            INT16_C(     0), INT16_C( -3817), INT16_C(     0), INT16_C( 26698),
                            INT16_C(     0), INT16_C(     0), INT16_C(-15991), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C( 27984), INT16_C(  1876),
                            INT16_C( 14808), INT16_C(-10243), INT16_C(     0), INT16_C(     0),
                            INT16_C(-26054), INT16_C( 23235), INT16_C(     0), INT16_C(     0),
                            INT16_C(-32195), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C( -1915), INT16_C(     0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mov_epi16(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mov_epi16");
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mov_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT16_C(23562),
      easysimd_mm512_set_epi32(INT32_C(  413218138), INT32_C(-2056039012), INT32_C(  359898417), INT32_C(  503742711),
                            INT32_C( -964140572), INT32_C( 1845540628), INT32_C( 1555270769), INT32_C(  276306907),
                            INT32_C(  923961977), INT32_C( 2070870327), INT32_C( -106769082), INT32_C(   21505510),
                            INT32_C(-1894191102), INT32_C(  -61868066), INT32_C(-1022555483), INT32_C(  842262872)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(-2056039012), INT32_C(          0), INT32_C(  503742711),
                            INT32_C( -964140572), INT32_C( 1845540628), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(-1894191102), INT32_C(          0), INT32_C(-1022555483), INT32_C(          0)) },
    { UINT16_C(36203),
      easysimd_mm512_set_epi32(INT32_C(-1836353351), INT32_C(-1955161161), INT32_C( 1387065895), INT32_C(  829477081),
                            INT32_C( 1194773762), INT32_C( 1305535140), INT32_C(  692999175), INT32_C(-1162293370),
                            INT32_C( 1281198604), INT32_C( -270591140), INT32_C(   23870431), INT32_C(-1469107120),
                            INT32_C( 1859513610), INT32_C(-1425966851), INT32_C(  381161214), INT32_C(  706499700)),
      easysimd_mm512_set_epi32(INT32_C(-1836353351), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1194773762), INT32_C( 1305535140), INT32_C(          0), INT32_C(-1162293370),
                            INT32_C(          0), INT32_C( -270591140), INT32_C(   23870431), INT32_C(          0),
                            INT32_C( 1859513610), INT32_C(          0), INT32_C(  381161214), INT32_C(  706499700)) },
    { UINT16_C(61846),
      easysimd_mm512_set_epi32(INT32_C(-1731705333), INT32_C( 1975072423), INT32_C( -536413935), INT32_C( 1477835290),
                            INT32_C( 1453154713), INT32_C( -133697355), INT32_C( 1038848393), INT32_C(  897042603),
                            INT32_C( 1474696001), INT32_C( 1692444627), INT32_C(-1157569404), INT32_C(-1969459150),
                            INT32_C(  120064093), INT32_C(-1121934893), INT32_C( 1895180026), INT32_C( 1628067999)),
      easysimd_mm512_set_epi32(INT32_C(-1731705333), INT32_C( 1975072423), INT32_C( -536413935), INT32_C( 1477835290),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(  897042603),
                            INT32_C( 1474696001), INT32_C(          0), INT32_C(          0), INT32_C(-1969459150),
                            INT32_C(          0), INT32_C(-1121934893), INT32_C( 1895180026), INT32_C(          0)) },
    { UINT16_C( 9005),
      easysimd_mm512_set_epi32(INT32_C(  317112464), INT32_C(  741023218), INT32_C(-1717304973), INT32_C( 1768422162),
                            INT32_C(-1938535542), INT32_C( -593182598), INT32_C( -560734377), INT32_C(-1833964883),
                            INT32_C(-2069017846), INT32_C( 1509337971), INT32_C(-1663080670), INT32_C( -363349477),
                            INT32_C( -761414190), INT32_C( 1575734613), INT32_C(  758160476), INT32_C(  434110055)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(-1717304973), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( -560734377), INT32_C(-1833964883),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1663080670), INT32_C(          0),
                            INT32_C( -761414190), INT32_C( 1575734613), INT32_C(          0), INT32_C(  434110055)) },
    { UINT16_C(16381),
      easysimd_mm512_set_epi32(INT32_C(-1241873035), INT32_C(-1720080742), INT32_C( 1575508697), INT32_C(  644418481),
                            INT32_C( -191348066), INT32_C( 1363259829), INT32_C( -969945370), INT32_C(-1662256156),
                            INT32_C( -483657475), INT32_C( 1693775573), INT32_C( -588936550), INT32_C( -831491481),
                            INT32_C(-1533494499), INT32_C(  690127328), INT32_C( 1408818770), INT32_C( 1154640340)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C( 1575508697), INT32_C(  644418481),
                            INT32_C( -191348066), INT32_C( 1363259829), INT32_C( -969945370), INT32_C(-1662256156),
                            INT32_C( -483657475), INT32_C( 1693775573), INT32_C( -588936550), INT32_C( -831491481),
                            INT32_C(-1533494499), INT32_C(  690127328), INT32_C(          0), INT32_C( 1154640340)) },
    { UINT16_C(37447),
      easysimd_mm512_set_epi32(INT32_C(-2077483324), INT32_C( -857673646), INT32_C(  754202712), INT32_C(  120435698),
                            INT32_C(-1765652094), INT32_C( -229167588), INT32_C(-1388415734), INT32_C( -902383521),
                            INT32_C(-1071136130), INT32_C(  575343777), INT32_C( 2007077268), INT32_C( -686416210),
                            INT32_C( -979195146), INT32_C( -793664277), INT32_C( 1970531286), INT32_C( -266532300)),
      easysimd_mm512_set_epi32(INT32_C(-2077483324), INT32_C(          0), INT32_C(          0), INT32_C(  120435698),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1388415734), INT32_C(          0),
                            INT32_C(          0), INT32_C(  575343777), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( -793664277), INT32_C( 1970531286), INT32_C( -266532300)) },
    { UINT16_C(31625),
      easysimd_mm512_set_epi32(INT32_C(-1558092593), INT32_C(  725220263), INT32_C( 2072028486), INT32_C(-1343089166),
                            INT32_C(  151067474), INT32_C( 1411237194), INT32_C(-1069461255), INT32_C(   79796340),
                            INT32_C(  -81868792), INT32_C( -238630197), INT32_C(-1945013502), INT32_C( -908401887),
                            INT32_C( 1836974186), INT32_C(-1548825981), INT32_C( 1873806111), INT32_C(-2038561806)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  725220263), INT32_C( 2072028486), INT32_C(-1343089166),
                            INT32_C(  151067474), INT32_C(          0), INT32_C(-1069461255), INT32_C(   79796340),
                            INT32_C(  -81868792), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1836974186), INT32_C(          0), INT32_C(          0), INT32_C(-2038561806)) },
    { UINT16_C(29945),
      easysimd_mm512_set_epi32(INT32_C( 1874437031), INT32_C( 1791346696), INT32_C( 1351362877), INT32_C( 1434624201),
                            INT32_C( 1114612735), INT32_C(-2006574951), INT32_C( 1470768291), INT32_C( 1700301025),
                            INT32_C(  677818674), INT32_C( -624147248), INT32_C(  795562156), INT32_C(-1625864242),
                            INT32_C( -323693444), INT32_C(  242932397), INT32_C( 1315868789), INT32_C(-1134215759)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( 1791346696), INT32_C( 1351362877), INT32_C( 1434624201),
                            INT32_C(          0), INT32_C(-2006574951), INT32_C(          0), INT32_C(          0),
                            INT32_C(  677818674), INT32_C( -624147248), INT32_C(  795562156), INT32_C(-1625864242),
                            INT32_C( -323693444), INT32_C(          0), INT32_C(          0), INT32_C(-1134215759)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mov_epi32(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mov_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mov_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C(248),
      easysimd_mm512_set_epi64(INT64_C( 2197185227781835820), INT64_C(   15935016481556146),
                            INT64_C(-7676897351944758395), INT64_C( -396609189869225788),
                            INT64_C( 2033032872247713203), INT64_C(  196856286260699291),
                            INT64_C(-5445071775966286746), INT64_C( 4145146436042188996)),
      easysimd_mm512_set_epi64(INT64_C( 2197185227781835820), INT64_C(   15935016481556146),
                            INT64_C(-7676897351944758395), INT64_C( -396609189869225788),
                            INT64_C( 2033032872247713203), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C( 50),
      easysimd_mm512_set_epi64(INT64_C(-5159763787063667600), INT64_C( -806315631695634460),
                            INT64_C( 8295852346035342936), INT64_C(-3045923053405968902),
                            INT64_C( 8238548627246121972), INT64_C( 6711306137119169451),
                            INT64_C( 8909631005256860390), INT64_C(-3863575863957815519)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 8295852346035342936), INT64_C(-3045923053405968902),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 8909631005256860390), INT64_C(                   0)) },
    { UINT8_C(205),
      easysimd_mm512_set_epi64(INT64_C( -364876834429138531), INT64_C(-8701118401174655403),
                            INT64_C(-4225146583063624142), INT64_C( 1748175868453859972),
                            INT64_C(-2146322958238101234), INT64_C( 5260281165058225920),
                            INT64_C( 6150323032540537551), INT64_C( 7787131310828538951)),
      easysimd_mm512_set_epi64(INT64_C( -364876834429138531), INT64_C(-8701118401174655403),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-2146322958238101234), INT64_C( 5260281165058225920),
                            INT64_C(                   0), INT64_C( 7787131310828538951)) },
    { UINT8_C( 37),
      easysimd_mm512_set_epi64(INT64_C( 5789616018161199708), INT64_C( -189925922123546982),
                            INT64_C(-5486743783379366456), INT64_C(  576479268129213490),
                            INT64_C( 6799755442903924910), INT64_C( 8415809909668152758),
                            INT64_C(-5111257061290341882), INT64_C( -197393302827860380)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-5486743783379366456), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 8415809909668152758),
                            INT64_C(                   0), INT64_C( -197393302827860380)) },
    { UINT8_C( 88),
      easysimd_mm512_set_epi64(INT64_C(-1554832219963971622), INT64_C(-7375448098764531208),
                            INT64_C( 8161779997769921522), INT64_C( -561105360908971667),
                            INT64_C(-3236710360814756666), INT64_C( 5084844885557932166),
                            INT64_C(-5492461044876086653), INT64_C(-7792360489043648145)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-7375448098764531208),
                            INT64_C(                   0), INT64_C( -561105360908971667),
                            INT64_C(-3236710360814756666), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C( 37),
      easysimd_mm512_set_epi64(INT64_C( 6519579675176262812), INT64_C(-7940748567058253670),
                            INT64_C( 6289445638826848684), INT64_C( 1300334437315413424),
                            INT64_C(-2416059830887765317), INT64_C(-5031784341515283026),
                            INT64_C( 8433369475758597766), INT64_C( 5881534320792150012)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 6289445638826848684), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-5031784341515283026),
                            INT64_C(                   0), INT64_C( 5881534320792150012)) },
    { UINT8_C( 16),
      easysimd_mm512_set_epi64(INT64_C(-5091799924273173479), INT64_C(  326582266571623592),
                            INT64_C(-3763964521506166714), INT64_C( 4584033432636860229),
                            INT64_C(-1921935435734596553), INT64_C( 3382451871995350760),
                            INT64_C(  531323325001255908), INT64_C(-5053470530677804405)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 4584033432636860229),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C( 33),
      easysimd_mm512_set_epi64(INT64_C(-6532064902097701697), INT64_C(-2430912179372724686),
                            INT64_C( 3177343060104491288), INT64_C(-7094318047719451166),
                            INT64_C(-3484792886859817284), INT64_C( -117759466073012358),
                            INT64_C( 8855057132598654557), INT64_C( -457984409854209760)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 3177343060104491288), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( -457984409854209760)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mov_epi64(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mov_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mov_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { UINT8_C(198),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -717.73), EASYSIMD_FLOAT64_C( -238.83),
                         EASYSIMD_FLOAT64_C( -181.88), EASYSIMD_FLOAT64_C( -183.39),
                         EASYSIMD_FLOAT64_C(  840.23), EASYSIMD_FLOAT64_C(  345.87),
                         EASYSIMD_FLOAT64_C(  630.37), EASYSIMD_FLOAT64_C(  306.75)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -717.73), EASYSIMD_FLOAT64_C( -238.83),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  345.87),
                         EASYSIMD_FLOAT64_C(  630.37), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(246),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -518.56), EASYSIMD_FLOAT64_C( -830.99),
                         EASYSIMD_FLOAT64_C(  129.34), EASYSIMD_FLOAT64_C(  771.89),
                         EASYSIMD_FLOAT64_C( -815.64), EASYSIMD_FLOAT64_C( -128.60),
                         EASYSIMD_FLOAT64_C( -244.79), EASYSIMD_FLOAT64_C( -568.94)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -518.56), EASYSIMD_FLOAT64_C( -830.99),
                         EASYSIMD_FLOAT64_C(  129.34), EASYSIMD_FLOAT64_C(  771.89),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -128.60),
                         EASYSIMD_FLOAT64_C( -244.79), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(141),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  637.67), EASYSIMD_FLOAT64_C(  322.55),
                         EASYSIMD_FLOAT64_C(  578.22), EASYSIMD_FLOAT64_C( -961.29),
                         EASYSIMD_FLOAT64_C(  737.15), EASYSIMD_FLOAT64_C(  475.09),
                         EASYSIMD_FLOAT64_C(  178.14), EASYSIMD_FLOAT64_C(  -60.04)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  637.67), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  737.15), EASYSIMD_FLOAT64_C(  475.09),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  -60.04)) },
    { UINT8_C(231),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  304.19), EASYSIMD_FLOAT64_C(  154.72),
                         EASYSIMD_FLOAT64_C(   74.11), EASYSIMD_FLOAT64_C(  -64.46),
                         EASYSIMD_FLOAT64_C(  202.28), EASYSIMD_FLOAT64_C( -444.38),
                         EASYSIMD_FLOAT64_C(  774.34), EASYSIMD_FLOAT64_C(  215.79)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  304.19), EASYSIMD_FLOAT64_C(  154.72),
                         EASYSIMD_FLOAT64_C(   74.11), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -444.38),
                         EASYSIMD_FLOAT64_C(  774.34), EASYSIMD_FLOAT64_C(  215.79)) },
    { UINT8_C( 62),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -983.34), EASYSIMD_FLOAT64_C(  259.69),
                         EASYSIMD_FLOAT64_C(  303.29), EASYSIMD_FLOAT64_C( -160.70),
                         EASYSIMD_FLOAT64_C( -787.06), EASYSIMD_FLOAT64_C(  198.77),
                         EASYSIMD_FLOAT64_C( -144.49), EASYSIMD_FLOAT64_C(  944.24)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  303.29), EASYSIMD_FLOAT64_C( -160.70),
                         EASYSIMD_FLOAT64_C( -787.06), EASYSIMD_FLOAT64_C(  198.77),
                         EASYSIMD_FLOAT64_C( -144.49), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(229),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -172.38), EASYSIMD_FLOAT64_C(  210.60),
                         EASYSIMD_FLOAT64_C(  840.69), EASYSIMD_FLOAT64_C(  875.33),
                         EASYSIMD_FLOAT64_C(  702.20), EASYSIMD_FLOAT64_C( -408.83),
                         EASYSIMD_FLOAT64_C(  172.51), EASYSIMD_FLOAT64_C(  896.66)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -172.38), EASYSIMD_FLOAT64_C(  210.60),
                         EASYSIMD_FLOAT64_C(  840.69), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -408.83),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  896.66)) },
    { UINT8_C( 93),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -853.39), EASYSIMD_FLOAT64_C(  281.51),
                         EASYSIMD_FLOAT64_C( -719.72), EASYSIMD_FLOAT64_C(  342.79),
                         EASYSIMD_FLOAT64_C( -679.92), EASYSIMD_FLOAT64_C( -623.46),
                         EASYSIMD_FLOAT64_C(  756.10), EASYSIMD_FLOAT64_C( -762.35)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  281.51),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  342.79),
                         EASYSIMD_FLOAT64_C( -679.92), EASYSIMD_FLOAT64_C( -623.46),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -762.35)) },
    { UINT8_C(156),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -853.45), EASYSIMD_FLOAT64_C(  527.42),
                         EASYSIMD_FLOAT64_C( -111.28), EASYSIMD_FLOAT64_C(  996.35),
                         EASYSIMD_FLOAT64_C(  374.30), EASYSIMD_FLOAT64_C(  314.59),
                         EASYSIMD_FLOAT64_C( -739.54), EASYSIMD_FLOAT64_C(  477.55)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -853.45), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  996.35),
                         EASYSIMD_FLOAT64_C(  374.30), EASYSIMD_FLOAT64_C(  314.59),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mov_pd(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mov_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mov_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { UINT16_C(42363),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    27.87), EASYSIMD_FLOAT32_C(  -816.11), EASYSIMD_FLOAT32_C(   100.70), EASYSIMD_FLOAT32_C(  -687.21),
                         EASYSIMD_FLOAT32_C(   641.77), EASYSIMD_FLOAT32_C(   431.46), EASYSIMD_FLOAT32_C(  -432.41), EASYSIMD_FLOAT32_C(   128.97),
                         EASYSIMD_FLOAT32_C(   877.42), EASYSIMD_FLOAT32_C(   723.11), EASYSIMD_FLOAT32_C(   773.77), EASYSIMD_FLOAT32_C(   562.67),
                         EASYSIMD_FLOAT32_C(  -364.27), EASYSIMD_FLOAT32_C(   912.16), EASYSIMD_FLOAT32_C(  -872.01), EASYSIMD_FLOAT32_C(  -172.46)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    27.87), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   100.70), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   431.46), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   128.97),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   723.11), EASYSIMD_FLOAT32_C(   773.77), EASYSIMD_FLOAT32_C(   562.67),
                         EASYSIMD_FLOAT32_C(  -364.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -872.01), EASYSIMD_FLOAT32_C(  -172.46)) },
    { UINT16_C(38549),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   618.21), EASYSIMD_FLOAT32_C(   498.90), EASYSIMD_FLOAT32_C(  -849.91), EASYSIMD_FLOAT32_C(   -52.66),
                         EASYSIMD_FLOAT32_C(   545.34), EASYSIMD_FLOAT32_C(   794.02), EASYSIMD_FLOAT32_C(  -461.31), EASYSIMD_FLOAT32_C(   114.20),
                         EASYSIMD_FLOAT32_C(    86.28), EASYSIMD_FLOAT32_C(  -885.12), EASYSIMD_FLOAT32_C(   172.95), EASYSIMD_FLOAT32_C(   554.47),
                         EASYSIMD_FLOAT32_C(  -747.12), EASYSIMD_FLOAT32_C(  -745.25), EASYSIMD_FLOAT32_C(  -281.94), EASYSIMD_FLOAT32_C(   206.58)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   618.21), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -52.66),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   794.02), EASYSIMD_FLOAT32_C(  -461.31), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    86.28), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   554.47),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -745.25), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   206.58)) },
    { UINT16_C(52704),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   502.41), EASYSIMD_FLOAT32_C(   880.11), EASYSIMD_FLOAT32_C(  -557.95), EASYSIMD_FLOAT32_C(  -268.94),
                         EASYSIMD_FLOAT32_C(   733.29), EASYSIMD_FLOAT32_C(   706.04), EASYSIMD_FLOAT32_C(   -93.63), EASYSIMD_FLOAT32_C(  -582.14),
                         EASYSIMD_FLOAT32_C(  -836.38), EASYSIMD_FLOAT32_C(   744.38), EASYSIMD_FLOAT32_C(   -45.29), EASYSIMD_FLOAT32_C(  -703.39),
                         EASYSIMD_FLOAT32_C(  -540.13), EASYSIMD_FLOAT32_C(   467.24), EASYSIMD_FLOAT32_C(  -527.36), EASYSIMD_FLOAT32_C(   198.48)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   502.41), EASYSIMD_FLOAT32_C(   880.11), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   733.29), EASYSIMD_FLOAT32_C(   706.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -582.14),
                         EASYSIMD_FLOAT32_C(  -836.38), EASYSIMD_FLOAT32_C(   744.38), EASYSIMD_FLOAT32_C(   -45.29), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(22254),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -71.03), EASYSIMD_FLOAT32_C(   476.98), EASYSIMD_FLOAT32_C(   846.87), EASYSIMD_FLOAT32_C(   538.39),
                         EASYSIMD_FLOAT32_C(   819.31), EASYSIMD_FLOAT32_C(  -703.74), EASYSIMD_FLOAT32_C(    35.79), EASYSIMD_FLOAT32_C(  -913.43),
                         EASYSIMD_FLOAT32_C(   774.49), EASYSIMD_FLOAT32_C(  -248.35), EASYSIMD_FLOAT32_C(  -966.82), EASYSIMD_FLOAT32_C(  -517.72),
                         EASYSIMD_FLOAT32_C(  -427.16), EASYSIMD_FLOAT32_C(  -808.81), EASYSIMD_FLOAT32_C(   888.05), EASYSIMD_FLOAT32_C(  -556.04)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   476.98), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   538.39),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -703.74), EASYSIMD_FLOAT32_C(    35.79), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   774.49), EASYSIMD_FLOAT32_C(  -248.35), EASYSIMD_FLOAT32_C(  -966.82), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -427.16), EASYSIMD_FLOAT32_C(  -808.81), EASYSIMD_FLOAT32_C(   888.05), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(52364),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   995.54), EASYSIMD_FLOAT32_C(   221.44), EASYSIMD_FLOAT32_C(   899.46), EASYSIMD_FLOAT32_C(   449.06),
                         EASYSIMD_FLOAT32_C(  -950.30), EASYSIMD_FLOAT32_C(  -151.76), EASYSIMD_FLOAT32_C(  -841.60), EASYSIMD_FLOAT32_C(    17.37),
                         EASYSIMD_FLOAT32_C(  -167.30), EASYSIMD_FLOAT32_C(  -256.21), EASYSIMD_FLOAT32_C(  -735.57), EASYSIMD_FLOAT32_C(  -164.68),
                         EASYSIMD_FLOAT32_C(   752.38), EASYSIMD_FLOAT32_C(   507.77), EASYSIMD_FLOAT32_C(  -277.52), EASYSIMD_FLOAT32_C(     4.35)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   995.54), EASYSIMD_FLOAT32_C(   221.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -950.30), EASYSIMD_FLOAT32_C(  -151.76), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -167.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   752.38), EASYSIMD_FLOAT32_C(   507.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C( 1779),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -523.28), EASYSIMD_FLOAT32_C(  -985.42), EASYSIMD_FLOAT32_C(    56.90), EASYSIMD_FLOAT32_C(   872.34),
                         EASYSIMD_FLOAT32_C(  -127.19), EASYSIMD_FLOAT32_C(   894.80), EASYSIMD_FLOAT32_C(   377.19), EASYSIMD_FLOAT32_C(  -135.98),
                         EASYSIMD_FLOAT32_C(   185.79), EASYSIMD_FLOAT32_C(   425.67), EASYSIMD_FLOAT32_C(  -947.39), EASYSIMD_FLOAT32_C(  -417.93),
                         EASYSIMD_FLOAT32_C(   872.23), EASYSIMD_FLOAT32_C(   491.12), EASYSIMD_FLOAT32_C(   994.51), EASYSIMD_FLOAT32_C(    86.62)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   894.80), EASYSIMD_FLOAT32_C(   377.19), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   185.79), EASYSIMD_FLOAT32_C(   425.67), EASYSIMD_FLOAT32_C(  -947.39), EASYSIMD_FLOAT32_C(  -417.93),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   994.51), EASYSIMD_FLOAT32_C(    86.62)) },
    { UINT16_C(13470),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   900.57), EASYSIMD_FLOAT32_C(   485.77), EASYSIMD_FLOAT32_C(   272.94), EASYSIMD_FLOAT32_C(  -275.02),
                         EASYSIMD_FLOAT32_C(  -912.01), EASYSIMD_FLOAT32_C(  -611.34), EASYSIMD_FLOAT32_C(   325.35), EASYSIMD_FLOAT32_C(  -148.93),
                         EASYSIMD_FLOAT32_C(  -884.16), EASYSIMD_FLOAT32_C(   545.87), EASYSIMD_FLOAT32_C(  -690.64), EASYSIMD_FLOAT32_C(   883.50),
                         EASYSIMD_FLOAT32_C(  -329.16), EASYSIMD_FLOAT32_C(  -369.50), EASYSIMD_FLOAT32_C(   429.82), EASYSIMD_FLOAT32_C(   530.37)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   272.94), EASYSIMD_FLOAT32_C(  -275.02),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -611.34), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -884.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   883.50),
                         EASYSIMD_FLOAT32_C(  -329.16), EASYSIMD_FLOAT32_C(  -369.50), EASYSIMD_FLOAT32_C(   429.82), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(25684),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   446.23), EASYSIMD_FLOAT32_C(  -618.66), EASYSIMD_FLOAT32_C(  -992.21), EASYSIMD_FLOAT32_C(  -692.36),
                         EASYSIMD_FLOAT32_C(  -952.61), EASYSIMD_FLOAT32_C(   923.35), EASYSIMD_FLOAT32_C(  -322.87), EASYSIMD_FLOAT32_C(   288.88),
                         EASYSIMD_FLOAT32_C(   653.23), EASYSIMD_FLOAT32_C(  -162.04), EASYSIMD_FLOAT32_C(   847.98), EASYSIMD_FLOAT32_C(  -826.91),
                         EASYSIMD_FLOAT32_C(  -738.77), EASYSIMD_FLOAT32_C(   279.48), EASYSIMD_FLOAT32_C(   397.18), EASYSIMD_FLOAT32_C(   127.10)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -618.66), EASYSIMD_FLOAT32_C(  -992.21), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   923.35), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -162.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -826.91),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   279.48), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mov_ps(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mov_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mov_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mov_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mov_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mov_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mov_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mov_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mov_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mov_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mov_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mov_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mov_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mov_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mov_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mov_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mov_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mov_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mov_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mov_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mov_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mov_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mov_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mov_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mov_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mov_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mov_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mov_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mov_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mov_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mov_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mov_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mov_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mov_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mov_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mov_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mov_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mov_ps)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
