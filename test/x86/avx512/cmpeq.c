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

#define EASYSIMD_TEST_X86_AVX512_INSN cmpeq

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/cmpeq.h>
#include <easysimd/x86/avx512/blend.h>

static int
test_easysimd_mm_cmpeq_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int8_t a[16];
    const int8_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { {  INT8_C(  70), -INT8_C(  84), -INT8_C( 101),  INT8_C(  57),  INT8_C(  24), -INT8_C(  88), -INT8_C(  29),      INT8_MAX,
        -INT8_C(  60),  INT8_C(  32), -INT8_C(   1), -INT8_C(  45), -INT8_C(  70),  INT8_C(  32), -INT8_C(  71),  INT8_C(  63) },
      { -INT8_C(  71), -INT8_C(  59), -INT8_C( 102), -INT8_C(  95),  INT8_C(  93),  INT8_C(  52), -INT8_C(  83), -INT8_C(  17),
        -INT8_C( 114), -INT8_C(  52), -INT8_C(  26), -INT8_C(  70), -INT8_C(  18),  INT8_C( 114),  INT8_C(  26),  INT8_C(  52) },
      UINT16_C(    0) },
    { {  INT8_C(  30), -INT8_C(  74),  INT8_C( 110),  INT8_C(  54),  INT8_C(  94),  INT8_C(  81), -INT8_C(  75),  INT8_C(  34),
         INT8_C( 113), -INT8_C(  76), -INT8_C(  11),  INT8_C(  44), -INT8_C(  43), -INT8_C(  81),  INT8_C( 107), -INT8_C( 114) },
      {  INT8_C( 116),  INT8_C(   5),  INT8_C(  47), -INT8_C(  46),  INT8_C(  58), -INT8_C(  36), -INT8_C(  63), -INT8_C(  56),
        -INT8_C(  88), -INT8_C(  88), -INT8_C( 126), -INT8_C( 106),  INT8_C(  26), -INT8_C(  99), -INT8_C(  53),  INT8_C(  56) },
      UINT16_C(    0) },
    { {  INT8_C(  83),  INT8_C(  57),  INT8_C( 110), -INT8_C(  79), -INT8_C( 118),  INT8_C(  35), -INT8_C(  45), -INT8_C(   4),
        -INT8_C(  41), -INT8_C(  55),  INT8_C(  40), -INT8_C(  84),  INT8_C( 120), -INT8_C( 109),  INT8_C(  58), -INT8_C(  20) },
      { -INT8_C( 104),  INT8_C( 106), -INT8_C(  66), -INT8_C(  46),  INT8_C(  70),      INT8_MIN, -INT8_C( 101), -INT8_C(  17),
         INT8_C(  40),  INT8_C(  29), -INT8_C( 123),  INT8_C(  66), -INT8_C(  70),  INT8_C(  80),  INT8_C( 122),  INT8_C(  13) },
      UINT16_C(    0) },
    { { -INT8_C( 119), -INT8_C(  24), -INT8_C(  66),  INT8_C(  20),  INT8_C(  11), -INT8_C( 110),  INT8_C(  16), -INT8_C(  30),
         INT8_C(  91),  INT8_C(  56), -INT8_C( 113), -INT8_C(  45), -INT8_C(  53), -INT8_C(  55), -INT8_C(  65),  INT8_C(  99) },
      {  INT8_C(  51),  INT8_C( 126),  INT8_C(  54),  INT8_C( 122), -INT8_C(   2), -INT8_C(  47),  INT8_C( 105),  INT8_C(  38),
        -INT8_C(  18), -INT8_C(  18),  INT8_C( 104), -INT8_C(  87),  INT8_C(  63), -INT8_C(  30), -INT8_C(  74), -INT8_C(  56) },
      UINT16_C(    0) },
    { { -INT8_C(  54),  INT8_C( 117), -INT8_C(  36), -INT8_C(  43),  INT8_C(   7), -INT8_C(  20), -INT8_C(  73),  INT8_C(  98),
         INT8_C(  36),  INT8_C(  70),  INT8_C(  53), -INT8_C(  17),  INT8_C(  16), -INT8_C(  12),  INT8_C(  83),  INT8_C(  67) },
      {  INT8_C( 114), -INT8_C( 119), -INT8_C(  67),  INT8_C( 112),  INT8_C(  90),  INT8_C(  38), -INT8_C( 106),  INT8_C(  72),
         INT8_C(  21), -INT8_C(   2), -INT8_C(  15),  INT8_C(  84), -INT8_C(  32), -INT8_C(  88),  INT8_C(  28), -INT8_C(  86) },
      UINT16_C(    0) },
    { {  INT8_C(  29), -INT8_C(   7),      INT8_MAX,  INT8_C(  36), -INT8_C(  27),  INT8_C(  55), -INT8_C( 122),  INT8_C(  10),
         INT8_C( 125), -INT8_C(  69), -INT8_C(   7), -INT8_C( 115), -INT8_C(  81),  INT8_C(  76), -INT8_C(  47),  INT8_C(  34) },
      { -INT8_C(  43), -INT8_C( 114), -INT8_C( 110),  INT8_C(  47), -INT8_C(  75),  INT8_C(  41),  INT8_C( 120), -INT8_C(  54),
         INT8_C(  39),  INT8_C( 105),  INT8_C(  30),  INT8_C(   8),  INT8_C(  17),  INT8_C(  58), -INT8_C(  78),  INT8_C(  46) },
      UINT16_C(    0) },
    { {  INT8_C(  51),  INT8_C(  50),  INT8_C(  82),  INT8_C(  25),  INT8_C( 105), -INT8_C(  40),  INT8_C(  35), -INT8_C(  26),
        -INT8_C( 109),  INT8_C(  28),  INT8_C( 116),  INT8_C(  67),  INT8_C( 105),  INT8_C(  69),  INT8_C( 101),  INT8_C(  62) },
      { -INT8_C(  45), -INT8_C(   9),  INT8_C( 110), -INT8_C( 120),  INT8_C(  32), -INT8_C(  26),  INT8_C(  82),  INT8_C(  72),
         INT8_C(  79),  INT8_C( 112),  INT8_C(  80),  INT8_C(  97), -INT8_C(  85),  INT8_C(   2), -INT8_C( 113), -INT8_C(  34) },
      UINT16_C(    0) },
    { {  INT8_C(  52), -INT8_C(  30), -INT8_C(   9), -INT8_C(  99), -INT8_C(  70),  INT8_C(  26), -INT8_C( 124),  INT8_C(  78),
         INT8_C(  55), -INT8_C(   8), -INT8_C( 111), -INT8_C(  96),  INT8_C(  61), -INT8_C(  10), -INT8_C(  34),  INT8_C(  16) },
      { -INT8_C(  19),  INT8_C(  76), -INT8_C( 103),  INT8_C(  14),  INT8_C(  50), -INT8_C(  21),  INT8_C(  86), -INT8_C( 126),
         INT8_C(  92), -INT8_C(  90), -INT8_C(  29),  INT8_C(   7), -INT8_C(  88),  INT8_C( 114), -INT8_C(  27), -INT8_C(  35) },
      UINT16_C(    0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cmpeq_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_cmpeq_epi8_mask(a, b);

    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpeq_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int16_t a[8];
    const int16_t b[8];
    const uint8_t r;
  } test_vec[8] = {
    { { -INT16_C(  5658), -INT16_C( 25685),  INT16_C(  5520),  INT16_C( 29300),  INT16_C(  9991),  INT16_C( 28544), -INT16_C( 16756),  INT16_C( 15276) },
      { -INT16_C(  5658), -INT16_C(  4217), -INT16_C( 29510),  INT16_C( 31502),  INT16_C(  9991),  INT16_C( 10071), -INT16_C( 22721),  INT16_C(  9570) },
      UINT8_C( 17) },
    { {  INT16_C(  3472),  INT16_C(  8640),  INT16_C( 13347),  INT16_C( 12947),  INT16_C(  4947), -INT16_C(  8287),  INT16_C( 19921),  INT16_C(   538) },
      { -INT16_C( 24010), -INT16_C(  3854),  INT16_C(    46),  INT16_C( 13931), -INT16_C( 15833),  INT16_C( 26205), -INT16_C( 16535), -INT16_C(  1397) },
      UINT8_C(  0) },
    { {  INT16_C( 19660), -INT16_C( 25032), -INT16_C( 20864), -INT16_C( 11486), -INT16_C( 15423), -INT16_C( 27725), -INT16_C( 13039),  INT16_C( 18325) },
      { -INT16_C( 30865), -INT16_C( 25032), -INT16_C( 23673), -INT16_C( 20524),  INT16_C( 12646), -INT16_C( 27725), -INT16_C( 24080), -INT16_C( 17207) },
      UINT8_C( 34) },
    { { -INT16_C(  8708),  INT16_C( 28076), -INT16_C( 23611),  INT16_C( 21569), -INT16_C(  2927), -INT16_C( 23833),  INT16_C( 31937),  INT16_C( 12778) },
      {  INT16_C(  8708), -INT16_C( 29745), -INT16_C( 23611),  INT16_C( 11066),  INT16_C( 20692), -INT16_C( 15109), -INT16_C( 31937), -INT16_C(  8576) },
      UINT8_C(  4) },
    { {  INT16_C( 11433),  INT16_C( 15179), -INT16_C( 29446), -INT16_C( 29553),  INT16_C( 30336),  INT16_C( 16942),  INT16_C(  6387), -INT16_C(  2189) },
      {  INT16_C( 16954),  INT16_C(   130), -INT16_C( 16923), -INT16_C( 18133),  INT16_C(  9741), -INT16_C(   387), -INT16_C(   533), -INT16_C( 27428) },
      UINT8_C(  0) },
    { {  INT16_C( 10026),  INT16_C(  9423),  INT16_C( 24500),  INT16_C( 13488), -INT16_C(  8235), -INT16_C( 28426), -INT16_C(  5641),  INT16_C( 28163) },
      {  INT16_C( 16939),  INT16_C(  9423),  INT16_C( 24063),  INT16_C(  3273),  INT16_C( 18052),  INT16_C( 28426), -INT16_C(  6588),  INT16_C( 28163) },
      UINT8_C(130) },
    { { -INT16_C( 11763), -INT16_C( 15982),  INT16_C( 17201),  INT16_C(  2038),  INT16_C( 27682),  INT16_C(  6607), -INT16_C( 28842), -INT16_C( 32437) },
      {  INT16_C( 32209), -INT16_C( 12142),  INT16_C( 23515),  INT16_C( 24540), -INT16_C(  6494), -INT16_C(  6450), -INT16_C( 11828), -INT16_C(  9900) },
      UINT8_C(  0) },
    { { -INT16_C(  6493), -INT16_C( 10853), -INT16_C( 28375),  INT16_C( 19420), -INT16_C( 13981),  INT16_C( 21349), -INT16_C( 20422),  INT16_C(  3029) },
      {  INT16_C( 26414),  INT16_C(  2523), -INT16_C( 18494),  INT16_C( 19420),  INT16_C( 13981),  INT16_C( 26954), -INT16_C( 20422), -INT16_C( 21949) },
      UINT8_C( 72) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epi16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpeq_epi16_mask");

    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_cmpeq_epi16_mask(a, b);

    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpeq_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t a[4];
    const int32_t b[4];
    const uint8_t r;
  } test_vec[8] = {
    { {  INT32_C(  1331992865),  INT32_C(  2144019576), -INT32_C(   144043345), -INT32_C(   266106389) },
      {  INT32_C(  1331992865),  INT32_C(  1504526919), -INT32_C(   144043345),  INT32_C(    42442443) },
      UINT8_C(  5) },
    { {  INT32_C(     2955144), -INT32_C(   343934661), -INT32_C(   152901109), -INT32_C(  1796799117) },
      { -INT32_C(   438023266),  INT32_C(     4166029),  INT32_C(   449100623), -INT32_C(   501396391) },
      UINT8_C(  0) },
    { { -INT32_C(  1629336989),  INT32_C(  1317626435), -INT32_C(  1085969588),  INT32_C(  2016104213) },
      {  INT32_C(    83179383),  INT32_C(   386217160),  INT32_C(  1228065263),  INT32_C(  2016104213) },
      UINT8_C(  8) },
    { { -INT32_C(   585691751), -INT32_C(  1511711307),  INT32_C(  2105176076),  INT32_C(   327995036) },
      { -INT32_C(   585691751), -INT32_C(  1511711307), -INT32_C(    51505178),  INT32_C(   327995036) },
      UINT8_C( 11) },
    { { -INT32_C(  1763865818),  INT32_C(   944834604),  INT32_C(   364235897), -INT32_C(  1624686182) },
      {  INT32_C(  2020426179),  INT32_C(  1142772574), -INT32_C(   784331926),  INT32_C(  1272034596) },
      UINT8_C(  0) },
    { {  INT32_C(  1826729536),  INT32_C(  1666315512), -INT32_C(  1740285442),  INT32_C(  2022767289) },
      {  INT32_C(   232170927),  INT32_C(  1666315512),  INT32_C(  1740285442), -INT32_C(  2022767289) },
      UINT8_C(  2) },
    { {  INT32_C(  1777619123), -INT32_C(  2120640382), -INT32_C(  1927684366), -INT32_C(    68464308) },
      { -INT32_C(   284638729), -INT32_C(  1252894283),  INT32_C(   853116651),  INT32_C(  1052395659) },
      UINT8_C(  0) },
    { {  INT32_C(   447262360), -INT32_C(  1277411385),  INT32_C(  1808119071), -INT32_C(  1785683971) },
      {  INT32_C(   447262360), -INT32_C(  1277411385),  INT32_C(  1808119071), -INT32_C(  1785683971) },
      UINT8_C( 15) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epi32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpeq_epi32_mask");

    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_cmpeq_epi32_mask(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpeq_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t a[2];
    const int64_t b[2];
    const uint8_t r;
  } test_vec[8] = {
    { {  INT64_C( 9012861101846666836),  INT64_C( 2876204898361543406) },
      { -INT64_C( 5766600673480792886),  INT64_C( 2876204898361543406) },
      UINT8_C(  2) },
    { { -INT64_C( 3526045960010218457),  INT64_C( 4847701107580085249) },
      { -INT64_C( 3995049283584529246),  INT64_C( 3079176826114570572) },
      UINT8_C(  0) },
    { { -INT64_C( 7573702633131762125), -INT64_C( 7372679603699812111) },
      {  INT64_C( 8756694414224028134),  INT64_C( 6101406770248462976) },
      UINT8_C(  0) },
    { {  INT64_C( 6149156036326117041), -INT64_C( 1416874639361963066) },
      {  INT64_C( 6149156036326117041), -INT64_C( 1416874639361963066) },
      UINT8_C(  3) },
    { { -INT64_C( 5628932394306461769), -INT64_C( 3461605522857075196) },
      { -INT64_C( 5628932394306461769),  INT64_C( 3225716697335118380) },
      UINT8_C(  1) },
    { {  INT64_C( 7255753514009323469), -INT64_C( 2060287786318216069) },
      {  INT64_C( 4864822284758553602),  INT64_C( 1258732199896037388) },
      UINT8_C(  0) },
    { { -INT64_C( 4589099077245181195), -INT64_C( 1661016934327942174) },
      { -INT64_C( 4589099077245181195), -INT64_C( 1661016934327942174) },
      UINT8_C(  3) },
    { {  INT64_C( 4147667195048278090), -INT64_C( 2392754000819123972) },
      {  INT64_C( 4112416219333363101), -INT64_C( 1601383027039855182) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epi64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpeq_epi64_mask");

    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_cmpeq_epi64_mask(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const int8_t a[16];
    const int8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(43381),
      { -INT8_C(  34), -INT8_C(  79),  INT8_C(  73), -INT8_C(   8),  INT8_C(  77),  INT8_C(  85),  INT8_C(  47),  INT8_C(  30),
         INT8_C( 107), -INT8_C(   4), -INT8_C( 103),  INT8_C(  88),  INT8_C(  14),  INT8_C(  94), -INT8_C( 127),  INT8_C(  96) },
      {  INT8_C(  95), -INT8_C(  79),  INT8_C(   0), -INT8_C(  56),  INT8_C( 101), -INT8_C(  51),  INT8_C(  91),  INT8_C(  30),
         INT8_C(  23), -INT8_C(   4), -INT8_C( 103), -INT8_C( 116),      INT8_MAX,  INT8_C( 104),  INT8_C(  61), -INT8_C(  56) },
      UINT16_C(    0) },
    { UINT16_C(35425),
      {  INT8_C(  30), -INT8_C( 112), -INT8_C(  88), -INT8_C( 119), -INT8_C( 116),  INT8_C(  66), -INT8_C(  31), -INT8_C( 102),
        -INT8_C(  96),  INT8_C(  98), -INT8_C(   5),  INT8_C(  30),  INT8_C(  91),  INT8_C(  90),  INT8_C(  62),  INT8_C(  92) },
      {  INT8_C(  41),  INT8_C( 126), -INT8_C(  88), -INT8_C( 119), -INT8_C( 116),  INT8_C( 125), -INT8_C(  31), -INT8_C( 102),
        -INT8_C(  96),  INT8_C(  98), -INT8_C(  99),  INT8_C(  30),  INT8_C(  91), -INT8_C(  69),  INT8_C(  62),  INT8_C(  61) },
      UINT16_C( 2624) },
    { UINT16_C(25668),
      {      INT8_MAX,  INT8_C(  37), -INT8_C(   2),  INT8_C(  32), -INT8_C( 120), -INT8_C(   7),  INT8_C(  62), -INT8_C(  29),
         INT8_C(  84),  INT8_C( 124),  INT8_C(  63),  INT8_C( 119),  INT8_C(  32),  INT8_C( 105), -INT8_C(  11),  INT8_C(  20) },
      { -INT8_C( 111),  INT8_C(  37), -INT8_C(   2),  INT8_C( 119), -INT8_C( 120), -INT8_C(  68),  INT8_C(  62),  INT8_C(  22),
         INT8_C(  84), -INT8_C( 106),  INT8_C(  63), -INT8_C(  69),  INT8_C(  32),  INT8_C( 105), -INT8_C(  32),  INT8_C(  20) },
      UINT16_C( 9284) },
    { UINT16_C(26867),
      { -INT8_C(  14),  INT8_C(  49),  INT8_C(  76),  INT8_C(  70), -INT8_C(  83), -INT8_C( 117), -INT8_C(  67), -INT8_C(  50),
        -INT8_C(  12), -INT8_C(  78), -INT8_C(  30), -INT8_C(  98), -INT8_C(   3),  INT8_C( 115),  INT8_C(  20),  INT8_C(  28) },
      { -INT8_C(  40), -INT8_C(  87),  INT8_C(  76),  INT8_C(  79), -INT8_C(  83), -INT8_C(   1),  INT8_C(  10),  INT8_C(  57),
        -INT8_C(  46), -INT8_C(  78),  INT8_C(  50), -INT8_C(  98),  INT8_C(  83),  INT8_C( 115),  INT8_C(  20), -INT8_C(  97) },
      UINT16_C(26640) },
    { UINT16_C(41834),
      {  INT8_C(  42),  INT8_C(  39),  INT8_C( 113),  INT8_C(  31), -INT8_C(  39),  INT8_C(  83), -INT8_C(  67), -INT8_C(  42),
        -INT8_C(  57), -INT8_C(  47), -INT8_C(  14), -INT8_C(  78),  INT8_C( 103), -INT8_C(  54),  INT8_C(  91),  INT8_C(  18) },
      {  INT8_C(  17),  INT8_C(  39),  INT8_C( 113), -INT8_C(  29),  INT8_C(  14),  INT8_C(  83), -INT8_C(  67), -INT8_C(  42),
         INT8_C(  42), -INT8_C(  98), -INT8_C(  14), -INT8_C( 108),  INT8_C(  66), -INT8_C(  54),  INT8_C(  91), -INT8_C(  77) },
      UINT16_C( 8290) },
    { UINT16_C(38217),
      {  INT8_C(   7),  INT8_C(   6),  INT8_C( 107), -INT8_C(  50), -INT8_C(  40),  INT8_C(  94),      INT8_MIN,  INT8_C(  63),
         INT8_C(  40), -INT8_C(  37),  INT8_C(  81),  INT8_C(  66),  INT8_C( 118),  INT8_C(  99),  INT8_C( 101),  INT8_C(  75) },
      {  INT8_C(   7), -INT8_C(  17), -INT8_C(  44), -INT8_C(  50), -INT8_C(  40),  INT8_C(  94),  INT8_C(  16),  INT8_C(  63),
        -INT8_C(   1), -INT8_C(  52),  INT8_C(  81),  INT8_C(  66),  INT8_C(  97), -INT8_C( 118),  INT8_C(  79),  INT8_C(  75) },
      UINT16_C(33801) },
    { UINT16_C(10072),
      {  INT8_C(  42), -INT8_C(  40),  INT8_C( 102),  INT8_C(  83), -INT8_C(  77), -INT8_C(  73), -INT8_C( 107),  INT8_C(  42),
         INT8_C(  26), -INT8_C(   6),  INT8_C( 117),  INT8_C(  97),  INT8_C( 110), -INT8_C(  58),  INT8_C(  80),  INT8_C(  66) },
      {  INT8_C(  42),  INT8_C(  83),  INT8_C( 102),  INT8_C(  83), -INT8_C(  77), -INT8_C(  73),  INT8_C(  94),  INT8_C(  42),
        -INT8_C(  70), -INT8_C(   6),  INT8_C(  76),  INT8_C(  18), -INT8_C(  44), -INT8_C(  58), -INT8_C(  22),  INT8_C(  58) },
      UINT16_C( 8728) },
    { UINT16_C(40394),
      { -INT8_C(  14),  INT8_C(  95), -INT8_C(  57),  INT8_C(  12),  INT8_C(  89),  INT8_C(  60),  INT8_C( 109), -INT8_C(  57),
         INT8_C(   3), -INT8_C(  67),  INT8_C(  10),  INT8_C(  69), -INT8_C( 101),  INT8_C(  33), -INT8_C( 104),  INT8_C(  72) },
      {  INT8_C( 120), -INT8_C( 107),  INT8_C(  55),  INT8_C(  12),  INT8_C(  67), -INT8_C( 124),  INT8_C( 109), -INT8_C(  57),
        -INT8_C(   5),  INT8_C(  46),  INT8_C(  82),  INT8_C(  69), -INT8_C(  53),  INT8_C(  68), -INT8_C( 104), -INT8_C( 109) },
      UINT16_C( 2248) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epi8_mask");

   easysimd_assert_equal_mmask16(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i8x16());
    easysimd__mmask16 r = easysimd_mm_mask_cmpeq_epi8_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int16_t a[8];
    const int16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(193),
      { -INT16_C( 25429),  INT16_C( 26590), -INT16_C( 19284), -INT16_C( 21577), -INT16_C( 19965),  INT16_C( 19719),  INT16_C(  9888),  INT16_C( 20093) },
      { -INT16_C(  2389), -INT16_C(  6956), -INT16_C( 19356), -INT16_C( 21577), -INT16_C( 19965),  INT16_C( 19719),  INT16_C(  9888),  INT16_C( 20093) },
      UINT8_C(192) },
    { UINT8_C( 91),
      {  INT16_C( 23406),  INT16_C(  9488),  INT16_C(  4870),  INT16_C(  3799),  INT16_C( 30560), -INT16_C(  8908),  INT16_C( 21445), -INT16_C( 17528) },
      {  INT16_C( 23406),  INT16_C( 24848), -INT16_C( 31226),  INT16_C(  3799),  INT16_C( 30560), -INT16_C(  8908),  INT16_C( 21445), -INT16_C( 17528) },
      UINT8_C( 89) },
    { UINT8_C(212),
      { -INT16_C( 27269),  INT16_C( 21223),  INT16_C( 18595), -INT16_C( 10039), -INT16_C( 28891), -INT16_C( 20949),  INT16_C( 21322),  INT16_C( 27162) },
      { -INT16_C( 27269), -INT16_C(  4121), -INT16_C( 28253), -INT16_C( 10039), -INT16_C( 28891), -INT16_C( 20949),  INT16_C( 21322),  INT16_C( 27162) },
      UINT8_C(208) },
    { UINT8_C(109),
      {  INT16_C( 28830),  INT16_C( 26549), -INT16_C(  9400),  INT16_C( 29942),  INT16_C( 16777), -INT16_C( 23609), -INT16_C(  3669),  INT16_C( 17695) },
      {  INT16_C( 28830),  INT16_C( 26549), -INT16_C( 21688),  INT16_C( 27638),  INT16_C( 16777), -INT16_C( 23609), -INT16_C(  3669),  INT16_C( 17695) },
      UINT8_C( 97) },
    { UINT8_C( 20),
      { -INT16_C(  5940), -INT16_C( 15633),  INT16_C( 30812),  INT16_C(  9219), -INT16_C( 20709),  INT16_C( 14869), -INT16_C( 18956), -INT16_C( 16568) },
      {  INT16_C( 30923), -INT16_C( 15726),  INT16_C(  3420), -INT16_C( 11176), -INT16_C( 20709),  INT16_C( 14869), -INT16_C( 18956), -INT16_C( 16568) },
      UINT8_C( 16) },
    { UINT8_C( 54),
      {  INT16_C( 28102), -INT16_C( 13650), -INT16_C( 13935), -INT16_C( 22919),  INT16_C( 27908),  INT16_C( 19547),  INT16_C( 16941), -INT16_C( 23273) },
      {  INT16_C( 28102), -INT16_C( 13599), -INT16_C( 13987), -INT16_C( 28434),  INT16_C( 27908),  INT16_C( 19547),  INT16_C( 16941), -INT16_C( 23273) },
      UINT8_C( 48) },
    { UINT8_C(187),
      {  INT16_C(  9345), -INT16_C(  1403), -INT16_C( 30262),  INT16_C(  9832), -INT16_C( 27179), -INT16_C(  4760),  INT16_C( 15674), -INT16_C( 15366) },
      {  INT16_C(  8577),  INT16_C( 20179), -INT16_C( 30287),  INT16_C(  9832), -INT16_C( 27179), -INT16_C(  4760),  INT16_C( 15674), -INT16_C( 15366) },
      UINT8_C(184) },
    { UINT8_C(215),
      {  INT16_C(  3557),  INT16_C( 19808),  INT16_C( 13619), -INT16_C( 25374),  INT16_C(  7202),  INT16_C(  7641), -INT16_C(  2080),  INT16_C(   381) },
      { -INT16_C( 19764),  INT16_C( 12384),  INT16_C( 13619), -INT16_C( 11589),  INT16_C(  7202),  INT16_C(  7641), -INT16_C(  2080),  INT16_C(   381) },
      UINT8_C(212) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epi16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epi16_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i16x8());
    easysimd__mmask8 r = easysimd_mm_mask_cmpeq_epi16_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int32_t a[4];
    const int32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(187),
      { -INT32_C(  2080982614), -INT32_C(   989596392), -INT32_C(  1342164570), -INT32_C(  2076949226) },
      { -INT32_C(  1722121321), -INT32_C(   989596392), -INT32_C(  1342164570), -INT32_C(  2076949226) },
      UINT8_C( 10) },
    { UINT8_C( 58),
      {  INT32_C(   803289598),  INT32_C(  2118488377),  INT32_C(  1577044421), -INT32_C(  1275825872) },
      {  INT32_C(   701251719), -INT32_C(   746240882),  INT32_C(  1577044421),  INT32_C(  1802235698) },
      UINT8_C(  0) },
    { UINT8_C(128),
      {  INT32_C(  2034690480),  INT32_C(  1504355304),  INT32_C(   606363031),  INT32_C(  1252729610) },
      {  INT32_C(  1400024154), -INT32_C(  1450856774), -INT32_C(   903221222), -INT32_C(  1052479527) },
      UINT8_C(  0) },
    { UINT8_C( 18),
      {  INT32_C(  1269373678),  INT32_C(  1633013310),  INT32_C(  1313906553),  INT32_C(   615023736) },
      {  INT32_C(  1269373678),  INT32_C(   541690288), -INT32_C(  1154348339),  INT32_C(   615023736) },
      UINT8_C(  0) },
    { UINT8_C(170),
      { -INT32_C(    81542308), -INT32_C(  1737264620),  INT32_C(   378443546), -INT32_C(    57265547) },
      {  INT32_C(  1531432272), -INT32_C(  1737264620),  INT32_C(   378443546), -INT32_C(    57265547) },
      UINT8_C( 10) },
    { UINT8_C( 27),
      {  INT32_C(  1798656467),  INT32_C(   316689375), -INT32_C(   385622558),  INT32_C(  1463385482) },
      {  INT32_C(  1798656467),  INT32_C(   316689375),  INT32_C(  1719358867),  INT32_C(  1463385482) },
      UINT8_C( 11) },
    { UINT8_C(252),
      { -INT32_C(  1881237326),  INT32_C(   404342683), -INT32_C(  1780715519), -INT32_C(   123100474) },
      { -INT32_C(  1881237326),  INT32_C(   404342683), -INT32_C(  1780715519), -INT32_C(   123100474) },
      UINT8_C( 12) },
    { UINT8_C(223),
      {  INT32_C(   954206152), -INT32_C(  2046855453), -INT32_C(  1074333921), -INT32_C(  1191536217) },
      {  INT32_C(   954206152), -INT32_C(  2046855453), -INT32_C(  1074333921), -INT32_C(  1191536217) },
      UINT8_C( 15) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epi32_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_mm_mask_blend_epi32(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_i32x4());
    easysimd__mmask8 r = easysimd_mm_mask_cmpeq_epi32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int64_t a[2];
    const int64_t b[2];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 28),
      {  INT64_C( 2605051695352472842),  INT64_C( 4036595894879655384) },
      {  INT64_C( 2605051695352472842),  INT64_C( 4036595894879655384) },
      UINT8_C(  0) },
    { UINT8_C( 11),
      { -INT64_C( 2919501980829260344),  INT64_C( 7959601500015719006) },
      {  INT64_C( 9140349798493587673),  INT64_C( 7959601500015719006) },
      UINT8_C(  2) },
    { UINT8_C( 19),
      {  INT64_C( 2466255845771913676),  INT64_C( 7638571582542046237) },
      { -INT64_C( 8829268894088493372), -INT64_C(  932602640582184884) },
      UINT8_C(  0) },
    { UINT8_C(158),
      { -INT64_C( 1426397411734971427),  INT64_C( 2445374445696970635) },
      { -INT64_C( 1426397411734971427),  INT64_C( 3362545855045580962) },
      UINT8_C(  0) },
    { UINT8_C( 24),
      { -INT64_C( 7196660631704807105), -INT64_C( 8544694062749749971) },
      { -INT64_C( 8448676709702066049), -INT64_C( 8544694062749749971) },
      UINT8_C(  0) },
    { UINT8_C( 14),
      {  INT64_C( 7983929314038316780), -INT64_C( 8087867972117721927) },
      {  INT64_C( 8040568565845458917), -INT64_C( 8087867972117721927) },
      UINT8_C(  2) },
    { UINT8_C( 91),
      {  INT64_C( 8118257325544391510),  INT64_C( 4394589904640353202) },
      {  INT64_C(  385349894602818479),  INT64_C( 3297517265242137165) },
      UINT8_C(  0) },
    { UINT8_C( 70),
      { -INT64_C( 4309454181657066402), -INT64_C( 1487699805733051779) },
      {  INT64_C( 8467607421849588506), -INT64_C( 1487699805733051779) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epi64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_mm_mask_blend_epi64(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_i64x2());
    easysimd__mmask8 r = easysimd_mm_mask_cmpeq_epi64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpeq_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t a[16];
    const uint8_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { { UINT8_C( 87), UINT8_C(  9), UINT8_C(237), UINT8_C(162), UINT8_C( 50), UINT8_C(192), UINT8_C(176), UINT8_C(137),
        UINT8_C(107), UINT8_C( 32), UINT8_C( 66), UINT8_C(227), UINT8_C( 10), UINT8_C( 60), UINT8_C(123), UINT8_C(198) },
      { UINT8_C(196), UINT8_C(  9), UINT8_C(237), UINT8_C(246), UINT8_C(253), UINT8_C(192), UINT8_C(176), UINT8_C(214),
        UINT8_C(204), UINT8_C(128), UINT8_C(215), UINT8_C(132), UINT8_C(191), UINT8_C(212), UINT8_C(148), UINT8_C( 22) },
      UINT16_C(  102) },
    { { UINT8_C(222), UINT8_C(129), UINT8_C(184), UINT8_C( 16), UINT8_C(135), UINT8_C(234), UINT8_C(154), UINT8_C(243),
        UINT8_C( 11), UINT8_C(220), UINT8_C(214), UINT8_C( 21), UINT8_C( 24), UINT8_C( 82), UINT8_C(219), UINT8_C(220) },
      { UINT8_C( 15), UINT8_C( 45), UINT8_C(210), UINT8_C( 12), UINT8_C(238), UINT8_C(131), UINT8_C(226), UINT8_C(186),
        UINT8_C(  3), UINT8_C(186), UINT8_C( 62), UINT8_C(194), UINT8_C(142), UINT8_C(210), UINT8_C(216), UINT8_C(108) },
      UINT16_C(    0) },
    { { UINT8_C( 83), UINT8_C(144), UINT8_C(125), UINT8_C(219), UINT8_C(122), UINT8_C( 23), UINT8_C(206), UINT8_C(133),
        UINT8_C(243), UINT8_C(164), UINT8_C(154), UINT8_C( 11), UINT8_C(246), UINT8_C(118), UINT8_C(231), UINT8_C(  5) },
      { UINT8_C(163), UINT8_C(185), UINT8_C( 18), UINT8_C(145), UINT8_C( 60), UINT8_C(244), UINT8_C( 75), UINT8_C(133),
        UINT8_C(174), UINT8_C(137), UINT8_C(  1), UINT8_C( 61), UINT8_C( 91), UINT8_C(217), UINT8_C(169), UINT8_C(175) },
      UINT16_C(  128) },
    { { UINT8_C(105), UINT8_C( 38), UINT8_C(138), UINT8_C(228), UINT8_C( 61), UINT8_C( 88), UINT8_C(105), UINT8_C( 48),
        UINT8_C(252), UINT8_C(  4), UINT8_C( 59), UINT8_C(243), UINT8_C(122), UINT8_C( 34), UINT8_C(248), UINT8_C( 29) },
      { UINT8_C(220), UINT8_C( 10), UINT8_C(175), UINT8_C( 24),    UINT8_MAX, UINT8_C(250), UINT8_C( 88), UINT8_C(173),
        UINT8_C(132), UINT8_C( 89), UINT8_C(234), UINT8_C(223), UINT8_C( 51), UINT8_C(148), UINT8_C(142), UINT8_C(156) },
      UINT16_C(    0) },
    { { UINT8_C(186), UINT8_C( 24), UINT8_C(128), UINT8_C(248), UINT8_C(112), UINT8_C(234), UINT8_C( 40), UINT8_C(109),
        UINT8_C(238), UINT8_C(100), UINT8_C( 96), UINT8_C(104), UINT8_C(134), UINT8_C( 88), UINT8_C(133), UINT8_C( 98) },
      { UINT8_C(186), UINT8_C( 24), UINT8_C(128), UINT8_C(248), UINT8_C( 47), UINT8_C(211), UINT8_C( 15), UINT8_C(179),
        UINT8_C( 44), UINT8_C(250), UINT8_C(146), UINT8_C( 95), UINT8_C(142), UINT8_C( 33), UINT8_C(252), UINT8_C( 72) },
      UINT16_C(   15) },
    { { UINT8_C( 57), UINT8_C(124), UINT8_C( 64), UINT8_C(170), UINT8_C(102), UINT8_C(105), UINT8_C( 23), UINT8_C( 84),
        UINT8_C(205), UINT8_C(119), UINT8_C(188), UINT8_C( 83), UINT8_C(207), UINT8_C( 66), UINT8_C(182), UINT8_C( 50) },
      { UINT8_C(118), UINT8_C( 49), UINT8_C(148), UINT8_C(165), UINT8_C(  4), UINT8_C(164), UINT8_C( 88), UINT8_C( 48),
        UINT8_C(158), UINT8_C(235), UINT8_C(144), UINT8_C( 44), UINT8_C( 12), UINT8_C(140), UINT8_C(116), UINT8_C( 69) },
      UINT16_C(    0) },
    { { UINT8_C(  8), UINT8_C(181), UINT8_C(239), UINT8_C(111), UINT8_C( 30), UINT8_C(  6), UINT8_C(195), UINT8_C(235),
        UINT8_C(125), UINT8_C(128), UINT8_C( 62), UINT8_C( 77), UINT8_C(194), UINT8_C(246), UINT8_C(127), UINT8_C( 56) },
      { UINT8_C( 37), UINT8_C( 20), UINT8_C(222), UINT8_C( 41), UINT8_C(184), UINT8_C( 54), UINT8_C( 90), UINT8_C( 86),
        UINT8_C( 33), UINT8_C(234), UINT8_C(130), UINT8_C( 45), UINT8_C(118), UINT8_C(246), UINT8_C( 90), UINT8_C(126) },
      UINT16_C( 8192) },
    { { UINT8_C(171), UINT8_C( 98), UINT8_C(237), UINT8_C(201), UINT8_C(105), UINT8_C(177), UINT8_C(180), UINT8_C(230),
        UINT8_C( 49), UINT8_C(243), UINT8_C( 51), UINT8_C(243), UINT8_C(231), UINT8_C(179), UINT8_C( 43), UINT8_C( 13) },
      { UINT8_C(199), UINT8_C(  9), UINT8_C( 54), UINT8_C(127), UINT8_C( 64), UINT8_C(144), UINT8_C(213), UINT8_C( 97),
        UINT8_C(122), UINT8_C( 87), UINT8_C(143), UINT8_C(243), UINT8_C(231), UINT8_C(  2), UINT8_C(111), UINT8_C(249) },
      UINT16_C( 6144) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128((void *)test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128((void *)test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epu8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpeq_epu8_mask");

    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__mmask16 r = easysimd_mm_cmpeq_epu8_mask(a, b);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpeq_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint16_t a[8];
    const uint16_t b[8];
    const uint8_t r;
  } test_vec[8] = {
    { { UINT16_C(16260), UINT16_C(53339), UINT16_C(19595), UINT16_C(49947), UINT16_C(27886), UINT16_C(25987), UINT16_C(36586), UINT16_C(50109) },
      { UINT16_C(38413), UINT16_C(53339), UINT16_C(38074), UINT16_C(49947), UINT16_C(58653), UINT16_C(25987), UINT16_C(36586), UINT16_C(28413) },
      UINT8_C(106) },
    { { UINT16_C(22734), UINT16_C(22846), UINT16_C(23204), UINT16_C(37795), UINT16_C( 9927), UINT16_C(35832), UINT16_C(46604), UINT16_C( 6478) },
      { UINT16_C(62028), UINT16_C( 1538), UINT16_C( 7814), UINT16_C(41929), UINT16_C(22275), UINT16_C(60854), UINT16_C(46054), UINT16_C(46171) },
      UINT8_C(  0) },
    { { UINT16_C(39180), UINT16_C(45069), UINT16_C(45300), UINT16_C(47939), UINT16_C(15574), UINT16_C(57926), UINT16_C(38386), UINT16_C(16124) },
      { UINT16_C(65159), UINT16_C( 3397), UINT16_C( 3612), UINT16_C( 8368), UINT16_C(26214), UINT16_C(19469), UINT16_C(26906), UINT16_C( 9728) },
      UINT8_C(  0) },
    { { UINT16_C( 3330), UINT16_C(63190), UINT16_C(53722), UINT16_C(37809), UINT16_C(63574), UINT16_C(18549), UINT16_C(29069), UINT16_C( 5254) },
      { UINT16_C(52080), UINT16_C(35873), UINT16_C(53722), UINT16_C(16556), UINT16_C(47672), UINT16_C(21132), UINT16_C(29069), UINT16_C( 9592) },
      UINT8_C( 68) },
    { { UINT16_C(20121), UINT16_C(22044), UINT16_C(52584), UINT16_C(48873), UINT16_C(24261), UINT16_C(20998), UINT16_C(36304), UINT16_C(16486) },
      { UINT16_C(20121), UINT16_C(13004), UINT16_C(31065), UINT16_C(37234), UINT16_C(65075), UINT16_C(22243), UINT16_C(23434), UINT16_C( 9083) },
      UINT8_C(  1) },
    { { UINT16_C(38825), UINT16_C( 4729), UINT16_C(25189), UINT16_C(10960), UINT16_C(55233), UINT16_C(37245), UINT16_C(58212), UINT16_C(48337) },
      { UINT16_C(40299), UINT16_C(50415), UINT16_C(24854), UINT16_C(18773), UINT16_C(14432), UINT16_C(60063), UINT16_C( 7059), UINT16_C(15374) },
      UINT8_C(  0) },
    { { UINT16_C(34738), UINT16_C( 5966), UINT16_C( 8170), UINT16_C(43842), UINT16_C(49142), UINT16_C(23100), UINT16_C(49093), UINT16_C( 3350) },
      { UINT16_C( 1450), UINT16_C(49617), UINT16_C( 9831), UINT16_C(43842), UINT16_C(43614), UINT16_C(61873), UINT16_C(49093), UINT16_C(30510) },
      UINT8_C( 72) },
    { { UINT16_C(31815), UINT16_C(12687), UINT16_C(39736), UINT16_C(37340), UINT16_C( 6288), UINT16_C(13035), UINT16_C(46758), UINT16_C(53056) },
      { UINT16_C(31815), UINT16_C(28304), UINT16_C(39736), UINT16_C(38453), UINT16_C(59205), UINT16_C( 2696), UINT16_C(46758), UINT16_C(60801) },
      UINT8_C( 69) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128((void *)test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128((void *)test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpeq_epu16_mask");

    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 r = easysimd_mm_cmpeq_epu16_mask(a, b);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpeq_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint32_t a[4];
    const uint32_t b[4];
    const uint8_t r;
  } test_vec[8] = {
    { { UINT32_C(2205819070), UINT32_C(1251922836), UINT32_C(3280837314), UINT32_C(4056595763) },
      { UINT32_C(2205819070), UINT32_C(1251922836), UINT32_C(3280837314), UINT32_C(4056595763) },
      UINT8_C( 15) },
    { { UINT32_C(1903445213), UINT32_C(3652915991), UINT32_C(1151158445), UINT32_C(3090559194) },
      { UINT32_C(2597400342), UINT32_C(3652915991), UINT32_C(1151158445), UINT32_C(3090559194) },
      UINT8_C( 14) },
    { { UINT32_C(2334763892), UINT32_C(1801774526), UINT32_C(4289724717), UINT32_C(1297501913) },
      { UINT32_C(2763509416), UINT32_C(3544426433), UINT32_C(4289724717), UINT32_C(1297501913) },
      UINT8_C( 12) },
    { { UINT32_C(3084419321), UINT32_C(2451782757), UINT32_C(1167184702), UINT32_C(4100165196) },
      { UINT32_C( 513350237), UINT32_C(2247220417), UINT32_C(1579420037), UINT32_C(1370192471) },
      UINT8_C(  0) },
    { { UINT32_C(1594393594), UINT32_C(4260506559), UINT32_C(1245938686), UINT32_C(1866442258) },
      { UINT32_C(2190334144), UINT32_C( 956792500), UINT32_C(3734449031), UINT32_C(1866442258) },
      UINT8_C(  8) },
    { { UINT32_C(2214607045), UINT32_C(1652748899), UINT32_C(2276246901), UINT32_C( 737602411) },
      { UINT32_C(2007860163), UINT32_C(2293282049), UINT32_C(2254915552), UINT32_C(1311217289) },
      UINT8_C(  0) },
    { { UINT32_C( 852699086), UINT32_C(2392085785), UINT32_C(2232827930), UINT32_C(4021292076) },
      { UINT32_C(2439470736), UINT32_C(4078573331), UINT32_C(2232827930), UINT32_C(3862405399) },
      UINT8_C(  4) },
    { { UINT32_C(3776449224), UINT32_C(2037361759), UINT32_C( 419333612), UINT32_C( 570994322) },
      { UINT32_C( 531853068), UINT32_C(2037361759), UINT32_C(1724812622), UINT32_C(4132177198) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128((void *)test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128((void *)test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpeq_epu32_mask");

    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__mmask8 r = easysimd_mm_cmpeq_epu32_mask(a, b);

    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpeq_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint64_t a[2];
    const uint64_t b[2];
    const uint8_t r;
  } test_vec[8] = {
    { { UINT64_C(16122441627999121398), UINT64_C(17733406445843234505) },
      { UINT64_C( 3555767269133337999), UINT64_C(17733406445843234505) },
      UINT8_C(  2) },
    { { UINT64_C( 1802876413687294806), UINT64_C(14863023633818224665) },
      { UINT64_C( 1802876413687294806), UINT64_C( 9690629527878145345) },
      UINT8_C(  1) },
    { { UINT64_C(10236541713977121237), UINT64_C( 7932280507828209416) },
      { UINT64_C( 3724274524087310091), UINT64_C(11716136202105439708) },
      UINT8_C(  0) },
    { { UINT64_C( 3760435511801457452), UINT64_C( 7254838540759278655) },
      { UINT64_C(12938413844858638441), UINT64_C( 7254838540759278655) },
      UINT8_C(  2) },
    { { UINT64_C(16602858809916219145), UINT64_C(16185116238442587454) },
      { UINT64_C( 2391633376734570661), UINT64_C( 6668102972962895597) },
      UINT8_C(  0) },
    { { UINT64_C(   18291210671197755), UINT64_C(13922603491834529760) },
      { UINT64_C( 2986853278433571439), UINT64_C(13922603491834529760) },
      UINT8_C(  2) },
    { { UINT64_C( 9378555601121027685), UINT64_C(  447564555184869465) },
      { UINT64_C( 9378555601121027685), UINT64_C( 8259344490586781291) },
      UINT8_C(  1) },
    { { UINT64_C( 6620502568644227261), UINT64_C( 9127109524206037547) },
      { UINT64_C( 6620502568644227261), UINT64_C( 9127109524206037547) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128((void *)test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128((void *)test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpeq_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm_cmpeq_epu64_mask");

    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 r = easysimd_mm_cmpeq_epu64_mask(a, b);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const uint8_t a[16];
    const uint8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(19051),
      { UINT8_C( 31), UINT8_C(188), UINT8_C(167), UINT8_C(179), UINT8_C(201), UINT8_C( 74), UINT8_C(163), UINT8_C(200),
        UINT8_C(176), UINT8_C(226), UINT8_C( 65), UINT8_C(112), UINT8_C(105), UINT8_C(199), UINT8_C(253), UINT8_C( 97) },
      { UINT8_C(206), UINT8_C(204), UINT8_C(167), UINT8_C(179), UINT8_C(196), UINT8_C(112), UINT8_C(163), UINT8_C(200),
        UINT8_C( 34), UINT8_C(226), UINT8_C( 35), UINT8_C(141), UINT8_C(233), UINT8_C( 66), UINT8_C(253), UINT8_C( 97) },
      UINT16_C(16968) },
    { UINT16_C( 5109),
      { UINT8_C(219), UINT8_C(152), UINT8_C(219), UINT8_C(139), UINT8_C(122), UINT8_C( 28), UINT8_C(252), UINT8_C(227),
        UINT8_C(228), UINT8_C(249), UINT8_C( 69), UINT8_C( 23), UINT8_C( 54), UINT8_C( 19), UINT8_C(227), UINT8_C(216) },
      { UINT8_C(219), UINT8_C(152), UINT8_C(196), UINT8_C(139), UINT8_C(173), UINT8_C(231), UINT8_C(247), UINT8_C(227),
        UINT8_C(228), UINT8_C(249), UINT8_C( 69), UINT8_C( 30), UINT8_C( 54), UINT8_C(  3), UINT8_C(227), UINT8_C( 46) },
      UINT16_C( 4993) },
    { UINT16_C(12430),
      { UINT8_C( 75), UINT8_C(138), UINT8_C( 19), UINT8_C( 47), UINT8_C(131), UINT8_C( 88), UINT8_C( 70), UINT8_C(185),
        UINT8_C(107), UINT8_C( 41), UINT8_C(145), UINT8_C(223), UINT8_C(209), UINT8_C(217), UINT8_C(237), UINT8_C(149) },
      { UINT8_C(124), UINT8_C( 58), UINT8_C( 19), UINT8_C( 47), UINT8_C(131), UINT8_C( 88), UINT8_C(195), UINT8_C(185),
        UINT8_C(107), UINT8_C(121), UINT8_C(145), UINT8_C(233), UINT8_C(169), UINT8_C(217), UINT8_C(237), UINT8_C(189) },
      UINT16_C( 8332) },
    { UINT16_C(63351),
      { UINT8_C( 21), UINT8_C(189), UINT8_C(177), UINT8_C(129), UINT8_C(230), UINT8_C( 66), UINT8_C( 96), UINT8_C(184),
        UINT8_C( 28), UINT8_C( 77), UINT8_C( 77), UINT8_C( 95), UINT8_C(231), UINT8_C(202), UINT8_C(154), UINT8_C( 24) },
      { UINT8_C(112), UINT8_C( 51), UINT8_C(227), UINT8_C(203), UINT8_C(230), UINT8_C(224), UINT8_C(180), UINT8_C(184),
        UINT8_C( 40), UINT8_C( 77), UINT8_C( 19), UINT8_C( 95), UINT8_C( 32), UINT8_C(202), UINT8_C(154), UINT8_C( 24) },
      UINT16_C(57872) },
    { UINT16_C(17321),
      { UINT8_C( 19), UINT8_C( 10), UINT8_C(251), UINT8_C( 47), UINT8_C( 87), UINT8_C( 72), UINT8_C(143), UINT8_C( 63),
        UINT8_C( 18), UINT8_C( 41), UINT8_C( 87), UINT8_C(130), UINT8_C( 62), UINT8_C(199), UINT8_C(181), UINT8_C( 33) },
      { UINT8_C( 19), UINT8_C( 70), UINT8_C(251), UINT8_C( 47), UINT8_C(111), UINT8_C( 72), UINT8_C(143), UINT8_C(143),
        UINT8_C(243), UINT8_C( 41), UINT8_C( 87), UINT8_C(130), UINT8_C( 62), UINT8_C(115), UINT8_C(166), UINT8_C( 33) },
      UINT16_C(  553) },
    { UINT16_C(65187),
      { UINT8_C(173), UINT8_C( 50), UINT8_C( 61), UINT8_C(191), UINT8_C( 91), UINT8_C(148), UINT8_C( 65), UINT8_C(153),
        UINT8_C( 91), UINT8_C(246), UINT8_C(186), UINT8_C(237), UINT8_C( 88), UINT8_C(188), UINT8_C( 51), UINT8_C( 15) },
      { UINT8_C(218), UINT8_C(177), UINT8_C( 49), UINT8_C(191), UINT8_C( 91), UINT8_C(145), UINT8_C(105), UINT8_C( 65),
        UINT8_C( 91), UINT8_C( 16), UINT8_C(186), UINT8_C(237), UINT8_C( 88), UINT8_C( 82), UINT8_C( 51), UINT8_C( 75) },
      UINT16_C(23552) },
    { UINT16_C(13586),
      { UINT8_C(223), UINT8_C( 83), UINT8_C(206), UINT8_C( 58), UINT8_C( 74), UINT8_C(136), UINT8_C( 39), UINT8_C(162),
        UINT8_C( 69), UINT8_C( 90), UINT8_C(177), UINT8_C( 44), UINT8_C(253), UINT8_C(139), UINT8_C(221), UINT8_C( 46) },
      { UINT8_C(223), UINT8_C( 83), UINT8_C(206), UINT8_C(197), UINT8_C(210), UINT8_C(136), UINT8_C(109), UINT8_C(162),
        UINT8_C( 69), UINT8_C( 90), UINT8_C( 43), UINT8_C( 44), UINT8_C(124), UINT8_C( 10), UINT8_C(221), UINT8_C( 74) },
      UINT16_C(  258) },
    { UINT16_C(40260),
      { UINT8_C(210), UINT8_C(107), UINT8_C( 63), UINT8_C( 23), UINT8_C(197), UINT8_C(240), UINT8_C( 67), UINT8_C(194),
        UINT8_C(124), UINT8_C( 32), UINT8_C(241), UINT8_C(212), UINT8_C(213), UINT8_C(177), UINT8_C(150), UINT8_C(202) },
      { UINT8_C(210), UINT8_C(227), UINT8_C( 72), UINT8_C( 23), UINT8_C( 42), UINT8_C(115), UINT8_C( 82), UINT8_C(194),
        UINT8_C(124), UINT8_C( 32), UINT8_C(241), UINT8_C(193), UINT8_C(213), UINT8_C(194), UINT8_C( 44), UINT8_C(202) },
      UINT16_C(38144) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epu8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epu8_mask");

   easysimd_assert_equal_mmask16(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u8x16());
    easysimd__mmask16 r = easysimd_mm_mask_cmpeq_epu8_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint16_t a[8];
    const uint16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(207),
      { UINT16_C(31811), UINT16_C( 5998), UINT16_C(  464), UINT16_C(17753), UINT16_C(22773), UINT16_C(57620), UINT16_C(43771), UINT16_C(21298) },
      { UINT16_C(54794), UINT16_C(57054), UINT16_C(  464), UINT16_C(38613), UINT16_C(22773), UINT16_C(57620), UINT16_C(43771), UINT16_C(21298) },
      UINT8_C(196) },
    { UINT8_C(169),
      { UINT16_C(27649), UINT16_C(23211), UINT16_C(41138), UINT16_C(50866), UINT16_C(44417), UINT16_C(45936), UINT16_C(16385), UINT16_C(55229) },
      { UINT16_C(34561), UINT16_C(28930), UINT16_C(40990), UINT16_C(50866), UINT16_C(44417), UINT16_C(45936), UINT16_C(16385), UINT16_C(55229) },
      UINT8_C(168) },
    { UINT8_C( 25),
      { UINT16_C(53358), UINT16_C( 8377), UINT16_C(14998), UINT16_C( 1998), UINT16_C(53229), UINT16_C(43591), UINT16_C(26022), UINT16_C(11590) },
      { UINT16_C(19383), UINT16_C( 8427), UINT16_C( 5526), UINT16_C( 1986), UINT16_C(53229), UINT16_C(43591), UINT16_C(26022), UINT16_C(11590) },
      UINT8_C( 16) },
    { UINT8_C(217),
      { UINT16_C(63692), UINT16_C(39443), UINT16_C(  255), UINT16_C(18025), UINT16_C( 4011), UINT16_C(61867), UINT16_C( 4924), UINT16_C(34985) },
      { UINT16_C(24012), UINT16_C( 6163), UINT16_C(15350), UINT16_C(65164), UINT16_C( 4011), UINT16_C(61867), UINT16_C( 4924), UINT16_C(34985) },
      UINT8_C(208) },
    { UINT8_C( 11),
      { UINT16_C(30858), UINT16_C(62220), UINT16_C(47039), UINT16_C(27138), UINT16_C(16040), UINT16_C(20861), UINT16_C(31686), UINT16_C( 9127) },
      { UINT16_C( 6794), UINT16_C(19658), UINT16_C(47039), UINT16_C(13826), UINT16_C(16040), UINT16_C(20861), UINT16_C(31686), UINT16_C( 9127) },
      UINT8_C(  0) },
    { UINT8_C( 70),
      { UINT16_C(40147), UINT16_C(54781), UINT16_C(42246), UINT16_C(33812), UINT16_C(56055), UINT16_C(40703), UINT16_C(36606), UINT16_C( 6238) },
      { UINT16_C(40147), UINT16_C( 5885), UINT16_C(42342), UINT16_C(33793), UINT16_C(56055), UINT16_C(40703), UINT16_C(36606), UINT16_C( 6238) },
      UINT8_C( 64) },
    { UINT8_C( 18),
      { UINT16_C(58849), UINT16_C(62903), UINT16_C(44649), UINT16_C(27087), UINT16_C(52557), UINT16_C(44023), UINT16_C(20453), UINT16_C( 5462) },
      { UINT16_C(58732), UINT16_C(28343), UINT16_C(17001), UINT16_C(57556), UINT16_C(52557), UINT16_C(44023), UINT16_C(20453), UINT16_C( 5462) },
      UINT8_C( 16) },
    { UINT8_C(190),
      { UINT16_C(17755), UINT16_C(11117), UINT16_C(47790), UINT16_C(42488), UINT16_C(56933), UINT16_C(48372), UINT16_C(56819), UINT16_C(28456) },
      { UINT16_C(17755), UINT16_C(27245), UINT16_C(47790), UINT16_C(21889), UINT16_C(56933), UINT16_C(48372), UINT16_C(56819), UINT16_C(28456) },
      UINT8_C(180) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epu16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epu16_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u16x8());
    easysimd__mmask8 r = easysimd_mm_mask_cmpeq_epu16_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint32_t a[4];
    const uint32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(  0),
      { UINT32_C(2773705112), UINT32_C(3084662875), UINT32_C(3897386323), UINT32_C(2286782999) },
      { UINT32_C(2773705112), UINT32_C(3954666503), UINT32_C(2766895116), UINT32_C( 877215705) },
      UINT8_C(  0) },
    { UINT8_C(119),
      { UINT32_C(2345331493), UINT32_C(3282219577), UINT32_C( 261237248), UINT32_C(1802404048) },
      { UINT32_C(2345331493), UINT32_C(3282219577), UINT32_C( 261237248), UINT32_C(1802404048) },
      UINT8_C(  7) },
    { UINT8_C(124),
      { UINT32_C(4068240840), UINT32_C(2143456511), UINT32_C(1116679930), UINT32_C(4282518433) },
      { UINT32_C(4068240840), UINT32_C(2143456511), UINT32_C(3206130423), UINT32_C(4282518433) },
      UINT8_C(  8) },
    { UINT8_C( 33),
      { UINT32_C(2702939507), UINT32_C( 809655844), UINT32_C(1539588768), UINT32_C(1583776907) },
      { UINT32_C(2702939507), UINT32_C(2075804134), UINT32_C(1539588768), UINT32_C(2315761766) },
      UINT8_C(  1) },
    { UINT8_C( 22),
      { UINT32_C(2360851018), UINT32_C(  51843710), UINT32_C(1874687352), UINT32_C(1936148902) },
      { UINT32_C(2499794116), UINT32_C( 721053041), UINT32_C(1874687352), UINT32_C(3192125247) },
      UINT8_C(  4) },
    { UINT8_C(  9),
      { UINT32_C(3498230106), UINT32_C(3346461054), UINT32_C(1670637913), UINT32_C( 371695682) },
      { UINT32_C( 997372598), UINT32_C(3933874674), UINT32_C(3337828460), UINT32_C(2023126521) },
      UINT8_C(  0) },
    { UINT8_C(103),
      { UINT32_C(4123016972), UINT32_C(1916216274), UINT32_C(  85085770), UINT32_C(2293962710) },
      { UINT32_C(2837341297), UINT32_C(1916216274), UINT32_C(  85085770), UINT32_C( 763229274) },
      UINT8_C(  6) },
    { UINT8_C( 23),
      { UINT32_C(  39952308), UINT32_C(3369625521), UINT32_C( 424763426), UINT32_C(1066009563) },
      { UINT32_C(4051789270), UINT32_C(3369625521), UINT32_C( 424763426), UINT32_C(1066009563) },
      UINT8_C(  6) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epu32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epu32_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_mm_mask_blend_epi32(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u32x4());
    easysimd__mmask8 r = easysimd_mm_mask_cmpeq_epu32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpeq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint64_t a[2];
    const uint64_t b[2];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(205),
      { UINT64_C(15018779556501352369), UINT64_C( 2870092142046190675) },
      { UINT64_C(15018779556501352369), UINT64_C( 7456887948374681405) },
      UINT8_C(  1) },
    { UINT8_C(204),
      { UINT64_C( 3159427985319407561), UINT64_C( 9616519520574863708) },
      { UINT64_C(14629315577926026761), UINT64_C( 5774471194471459316) },
      UINT8_C(  0) },
    { UINT8_C(132),
      { UINT64_C(10964460194281619612), UINT64_C( 2914820312140597027) },
      { UINT64_C( 6104172689177482536), UINT64_C( 9254290706828019552) },
      UINT8_C(  0) },
    { UINT8_C(159),
      { UINT64_C(  653136695369198248), UINT64_C(16911534838135896211) },
      { UINT64_C(10456302706917080229), UINT64_C( 6381828019748494633) },
      UINT8_C(  0) },
    { UINT8_C(135),
      { UINT64_C( 8505548801729945931), UINT64_C( 8432552668170309993) },
      { UINT64_C( 8505548801729945931), UINT64_C( 8432552668170309993) },
      UINT8_C(  3) },
    { UINT8_C(120),
      { UINT64_C(18138220529095635477), UINT64_C(16205526785479878470) },
      { UINT64_C( 1966386415615869354), UINT64_C(16782481092709104500) },
      UINT8_C(  0) },
    { UINT8_C(128),
      { UINT64_C( 7455182534679207811), UINT64_C( 6982575445087612081) },
      { UINT64_C( 7455182534679207811), UINT64_C( 6982575445087612081) },
      UINT8_C(  0) },
    { UINT8_C( 83),
      { UINT64_C(17204615023477018275), UINT64_C(13051625560532438393) },
      { UINT64_C(12142910777652162763), UINT64_C(13051625560532438393) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpeq_epu64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpeq_epu64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_mm_mask_blend_epi64(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_u64x2());
    easysimd__mmask8 r = easysimd_mm_mask_cmpeq_epu64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int8_t a[32];
    const int8_t b[32];
    const uint32_t r;
  } test_vec[8] = {
    { {  INT8_C(  46),  INT8_C(  42), -INT8_C(   5), -INT8_C(  45),  INT8_C(  56),  INT8_C( 123),  INT8_C(  78),  INT8_C(  42),
        -INT8_C(  17), -INT8_C(  70), -INT8_C(  28), -INT8_C(  99),  INT8_C(  45), -INT8_C(  13),  INT8_C( 121),  INT8_C(  28),
         INT8_C(  51),  INT8_C(  76),  INT8_C(  56),  INT8_C( 118), -INT8_C(  19),  INT8_C(   1),  INT8_C( 111), -INT8_C( 106),
        -INT8_C(  30),  INT8_C(  15),  INT8_C(  91),  INT8_C(  91), -INT8_C(  89), -INT8_C(  23),  INT8_C( 125), -INT8_C(  43) },
      {  INT8_C(  19),  INT8_C(  42), -INT8_C(  87),  INT8_C(  75), -INT8_C(  59), -INT8_C(   9),  INT8_C( 117), -INT8_C(  76),
        -INT8_C(  79),  INT8_C(  89),  INT8_C(  81), -INT8_C(  34),  INT8_C(  76), -INT8_C(  54), -INT8_C(   5),      INT8_MAX,
         INT8_C(  22),  INT8_C(  51), -INT8_C(  11),  INT8_C(   4),  INT8_C(  52),  INT8_C( 100), -INT8_C( 102),  INT8_C(  23),
         INT8_C( 115), -INT8_C(  11),  INT8_C( 114),  INT8_C(  27), -INT8_C(  34), -INT8_C(  17), -INT8_C(  16), -INT8_C(  14) },
      UINT32_C(         2) },
    { {  INT8_C(  57), -INT8_C( 103),  INT8_C(  61), -INT8_C(  37), -INT8_C( 112), -INT8_C(  77), -INT8_C(  77),  INT8_C(  65),
         INT8_C(  12),  INT8_C(   4),  INT8_C(  32),  INT8_C(  89), -INT8_C(  49),  INT8_C(  27), -INT8_C(  40), -INT8_C(  27),
         INT8_C(  78), -INT8_C(  50), -INT8_C(  23), -INT8_C( 126),  INT8_C(  50), -INT8_C( 125), -INT8_C( 103), -INT8_C(  90),
         INT8_C( 120),  INT8_C(  11), -INT8_C(  63),  INT8_C(  87), -INT8_C(   6), -INT8_C(  79),  INT8_C(  73),  INT8_C(  51) },
      {  INT8_C(  75), -INT8_C( 122),  INT8_C(  50), -INT8_C(  37),  INT8_C(  57), -INT8_C(  27),  INT8_C(  29),  INT8_C(  70),
        -INT8_C(  23),  INT8_C(  61), -INT8_C(  97), -INT8_C(  72),  INT8_C(  49),  INT8_C( 119), -INT8_C(  98), -INT8_C(  90),
         INT8_C(  69), -INT8_C( 121),  INT8_C(  40),  INT8_C( 120),  INT8_C(  11), -INT8_C( 125),  INT8_C(  30), -INT8_C( 125),
        -INT8_C(  51), -INT8_C(  33), -INT8_C(  38), -INT8_C(  56), -INT8_C( 112),  INT8_C(  35), -INT8_C(   5), -INT8_C(  37) },
      UINT32_C(   2097160) },
    { { -INT8_C(  86),  INT8_C(  45), -INT8_C(  73), -INT8_C(  29),  INT8_C(  18), -INT8_C(  44),  INT8_C(  58), -INT8_C(   4),
         INT8_C(  17), -INT8_C( 123), -INT8_C(  76),  INT8_C( 105),  INT8_C(  64),  INT8_C( 122),  INT8_C(  15), -INT8_C( 123),
        -INT8_C(  38),  INT8_C(  55), -INT8_C(   3), -INT8_C(  27), -INT8_C(   7),  INT8_C(  27),  INT8_C( 104), -INT8_C(  57),
        -INT8_C(   6),  INT8_C(  67), -INT8_C( 113), -INT8_C( 117),  INT8_C( 102), -INT8_C( 118),  INT8_C( 102),  INT8_C(  16) },
      { -INT8_C(  72),  INT8_C(  29), -INT8_C(  12), -INT8_C(  54), -INT8_C(  15),  INT8_C(  29), -INT8_C(  58),  INT8_C(   2),
        -INT8_C(  26),  INT8_C( 123),  INT8_C( 107),  INT8_C(  38), -INT8_C(  51),  INT8_C( 122), -INT8_C(  85), -INT8_C( 123),
        -INT8_C(  78), -INT8_C(  87), -INT8_C( 116), -INT8_C(  85), -INT8_C(  60), -INT8_C(  11),  INT8_C( 114), -INT8_C(  65),
         INT8_C(  56),  INT8_C(   1),  INT8_C(  74), -INT8_C(  98), -INT8_C( 116), -INT8_C(  80), -INT8_C(  81),  INT8_C(  68) },
      UINT32_C(     40960) },
    { { -INT8_C(  50), -INT8_C(  93),  INT8_C(  14), -INT8_C(  65), -INT8_C(  64), -INT8_C(  43), -INT8_C(  62), -INT8_C(  90),
         INT8_C(  80),  INT8_C(  45), -INT8_C(  52),  INT8_C(  29), -INT8_C(  88),  INT8_C( 120), -INT8_C(  59),  INT8_C(  90),
         INT8_C(  33),  INT8_C(  81),  INT8_C(   5), -INT8_C(  27),  INT8_C(  70),  INT8_C( 120), -INT8_C(  92),  INT8_C( 126),
         INT8_C( 121), -INT8_C(  18),  INT8_C(  29),  INT8_C(   5), -INT8_C(  97), -INT8_C(  52),  INT8_C(  73),  INT8_C( 109) },
      {  INT8_C( 111),  INT8_C(  88),  INT8_C(  44),  INT8_C(  47),  INT8_C(  45), -INT8_C(  18), -INT8_C(  42),  INT8_C( 125),
         INT8_C(  28), -INT8_C(  94), -INT8_C( 102), -INT8_C(  60),  INT8_C(  26),  INT8_C(  95),  INT8_C(  30),  INT8_C(  59),
        -INT8_C(  79),  INT8_C(  35),  INT8_C(  33), -INT8_C(   9), -INT8_C( 101), -INT8_C(  59),  INT8_C( 118),  INT8_C(  21),
        -INT8_C(  76), -INT8_C( 109),  INT8_C(  26),  INT8_C(  83),  INT8_C(  95),  INT8_C( 100), -INT8_C(  64), -INT8_C(  50) },
      UINT32_C(         0) },
    { { -INT8_C(  68), -INT8_C(  20), -INT8_C(   3), -INT8_C(  23), -INT8_C(  37), -INT8_C(  45),  INT8_C( 102), -INT8_C(   9),
         INT8_C( 118),  INT8_C(   0), -INT8_C(  69), -INT8_C( 112),  INT8_C(  96), -INT8_C(  39), -INT8_C(  52),  INT8_C(  17),
        -INT8_C(   4), -INT8_C(  19),  INT8_C(   8), -INT8_C( 104), -INT8_C(  78),  INT8_C( 126), -INT8_C(  83),  INT8_C( 102),
         INT8_C(  17), -INT8_C(  57), -INT8_C(  71),  INT8_C( 112),  INT8_C(  43),  INT8_C( 121),  INT8_C(  62), -INT8_C(  25) },
      {  INT8_C( 102),  INT8_C(  60), -INT8_C(  48),  INT8_C(  65),  INT8_C(  15),  INT8_C(  54),  INT8_C(  56), -INT8_C( 123),
         INT8_C( 118),  INT8_C(   0), -INT8_C(  69), -INT8_C( 112),  INT8_C(  96), -INT8_C(  39), -INT8_C(  52),  INT8_C(  17),
        -INT8_C(  49), -INT8_C(  80),  INT8_C(  96), -INT8_C( 127),  INT8_C(  47),  INT8_C(  13), -INT8_C(  24),  INT8_C(  64),
        -INT8_C(  43), -INT8_C(  95), -INT8_C(  79),  INT8_C(   0),  INT8_C(  27), -INT8_C(  17), -INT8_C(  24), -INT8_C( 127) },
      UINT32_C(     65280) },
    { {  INT8_C(  43), -INT8_C(  72), -INT8_C(  62),  INT8_C(  59), -INT8_C(  17), -INT8_C(   6), -INT8_C(  64),  INT8_C(  38),
        -INT8_C(  19), -INT8_C(  42), -INT8_C(  67), -INT8_C(  71), -INT8_C(  72),  INT8_C( 101), -INT8_C( 127), -INT8_C( 121),
         INT8_C(  21), -INT8_C(  30),  INT8_C(   9),  INT8_C(  68), -INT8_C(  17), -INT8_C(  15), -INT8_C( 123), -INT8_C(  60),
        -INT8_C( 110),  INT8_C(  54), -INT8_C(  59), -INT8_C(  83),  INT8_C(  37), -INT8_C(  83),  INT8_C(  46),  INT8_C(  81) },
      {  INT8_C( 101), -INT8_C(  16), -INT8_C( 116),  INT8_C(  84), -INT8_C(  22),  INT8_C(  76),  INT8_C( 122), -INT8_C(  41),
         INT8_C(  35),  INT8_C(  55), -INT8_C( 112), -INT8_C(  37), -INT8_C( 100),  INT8_C(  18),  INT8_C(  99), -INT8_C(  78),
         INT8_C(  21), -INT8_C(  30),  INT8_C(   9),  INT8_C(  68), -INT8_C(  17), -INT8_C(  15), -INT8_C( 123), -INT8_C(  60),
        -INT8_C(  79),  INT8_C( 109), -INT8_C(  99), -INT8_C(  41),  INT8_C(  26), -INT8_C(  53),  INT8_C(  40),      INT8_MAX },
      UINT32_C(  16711680) },
    { { -INT8_C(  68), -INT8_C(  76), -INT8_C(  44), -INT8_C(  90),  INT8_C(   0),  INT8_C(  99),  INT8_C( 126),  INT8_C(  35),
        -INT8_C( 122),  INT8_C(  14), -INT8_C(   1),  INT8_C(  34),  INT8_C(  32),  INT8_C(  98), -INT8_C(  44),  INT8_C(  20),
        -INT8_C(  50), -INT8_C(  53), -INT8_C(   8),  INT8_C(  43),  INT8_C(  70), -INT8_C(  96),  INT8_C(  26), -INT8_C(   8),
         INT8_C(  13), -INT8_C(  73), -INT8_C(  49),  INT8_C(  39), -INT8_C( 125), -INT8_C(   9), -INT8_C(  90),  INT8_C(  63) },
      { -INT8_C(  85),  INT8_C( 122), -INT8_C(  27), -INT8_C(  85), -INT8_C(  55),  INT8_C(  99), -INT8_C(  49),  INT8_C(  79),
         INT8_C( 114), -INT8_C(  50),  INT8_C( 113), -INT8_C( 110),  INT8_C(  48),  INT8_C(  70), -INT8_C(  89), -INT8_C(   2),
         INT8_C(  17), -INT8_C(  97),  INT8_C(  41),  INT8_C(  87),  INT8_C(  63),  INT8_C(  67),  INT8_C(  79),  INT8_C(  76),
        -INT8_C(   5),  INT8_C(  30),  INT8_C( 115),  INT8_C( 126),  INT8_C(  21),  INT8_C(  25), -INT8_C(  67), -INT8_C(  64) },
      UINT32_C(        32) },
    { { -INT8_C( 108), -INT8_C(  94),  INT8_C( 108),  INT8_C(  93),  INT8_C(   6),  INT8_C(  59), -INT8_C(  84),  INT8_C( 120),
         INT8_C(   9),  INT8_C(  29),  INT8_C(  10),  INT8_C(  57),  INT8_C(  99), -INT8_C(  79),  INT8_C(  55),  INT8_C( 116),
         INT8_C(  80),  INT8_C(  96), -INT8_C(  52), -INT8_C( 113), -INT8_C(  93),  INT8_C(  27), -INT8_C(  37), -INT8_C(  98),
         INT8_C(  58),  INT8_C(  78),  INT8_C(  28),  INT8_C(  79),  INT8_C( 104), -INT8_C(  39),  INT8_C(  16), -INT8_C(   4) },
      {  INT8_C( 124),  INT8_C( 124),  INT8_C(  89), -INT8_C( 126), -INT8_C(  73),  INT8_C(   5), -INT8_C(   6), -INT8_C(  64),
         INT8_C(  34),  INT8_C(   4), -INT8_C(   7), -INT8_C( 122), -INT8_C(  74),  INT8_C(  48), -INT8_C(   6),  INT8_C(   6),
         INT8_C(  80),  INT8_C(  96), -INT8_C(  52), -INT8_C( 113), -INT8_C(  93),  INT8_C(  27), -INT8_C(  37), -INT8_C(  98),
        -INT8_C(  64), -INT8_C(  18),  INT8_C( 107),  INT8_C(  40), -INT8_C(  56),  INT8_C( 123),  INT8_C(  36),  INT8_C(  68) },
      UINT32_C(  16711680) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epi8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpeq_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 r = easysimd_mm256_cmpeq_epi8_mask(a, b);

    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int16_t a[16];
    const int16_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { { -INT16_C(  3679), -INT16_C( 19182),  INT16_C( 22641), -INT16_C( 22322),  INT16_C( 27422),  INT16_C( 30569),  INT16_C(  3437), -INT16_C( 23131),
        -INT16_C( 24626), -INT16_C( 24646),  INT16_C( 23285),  INT16_C( 10766),  INT16_C(  7157), -INT16_C( 27951),  INT16_C( 21468),  INT16_C( 32147) },
      { -INT16_C( 23227), -INT16_C( 18894),  INT16_C(   253),  INT16_C(  7006), -INT16_C( 14485), -INT16_C(  9838),  INT16_C( 14292), -INT16_C( 23682),
        -INT16_C( 24626), -INT16_C( 24646),  INT16_C( 23285),  INT16_C( 10766),  INT16_C(  7157), -INT16_C( 27951),  INT16_C( 21468),  INT16_C( 32147) },
      UINT16_C(65280) },
    { { -INT16_C(  2222),  INT16_C( 20247),  INT16_C( 30200),  INT16_C( 25450), -INT16_C(   963),  INT16_C(  4412), -INT16_C( 17868),  INT16_C(  2740),
        -INT16_C(  2061), -INT16_C( 31274), -INT16_C( 12984), -INT16_C( 19443),  INT16_C( 10133), -INT16_C( 20227), -INT16_C( 15660),  INT16_C(  9745) },
      {  INT16_C( 10426), -INT16_C( 19851), -INT16_C(  8291), -INT16_C(  9707),  INT16_C( 21212),  INT16_C(  4332), -INT16_C( 17868), -INT16_C(   230),
        -INT16_C(  3689), -INT16_C(  8315), -INT16_C( 27970),  INT16_C( 21396), -INT16_C( 28230), -INT16_C( 29181),  INT16_C(  5203),  INT16_C(  3509) },
      UINT16_C(   64) },
    { {  INT16_C( 10812), -INT16_C(  9537), -INT16_C( 10998), -INT16_C(  6476), -INT16_C( 24537),  INT16_C( 13302),  INT16_C(  4161), -INT16_C( 10189),
        -INT16_C( 18431), -INT16_C( 16456),  INT16_C( 19530),  INT16_C(  1042),  INT16_C(  5853),  INT16_C( 12435),  INT16_C( 18474),  INT16_C( 26430) },
      { -INT16_C(   654),  INT16_C( 31809), -INT16_C(  2606), -INT16_C(  6476),  INT16_C( 22678), -INT16_C( 10451),  INT16_C( 24681),  INT16_C( 27311),
         INT16_C( 26392),  INT16_C( 25130),  INT16_C( 15539), -INT16_C( 28569), -INT16_C(  1454),  INT16_C( 32193), -INT16_C(   190), -INT16_C( 19228) },
      UINT16_C(    8) },
    { {  INT16_C(  9724), -INT16_C( 12495), -INT16_C( 27878), -INT16_C( 20280), -INT16_C(  2580),  INT16_C( 21895),  INT16_C( 14165),  INT16_C( 28095),
        -INT16_C(  5730),  INT16_C( 21200),  INT16_C( 14118),  INT16_C( 30946), -INT16_C( 23759),  INT16_C( 29685), -INT16_C(  9822), -INT16_C( 24793) },
      {  INT16_C( 22782),  INT16_C(  6510),  INT16_C( 14060), -INT16_C( 20280),  INT16_C( 20780), -INT16_C( 32467), -INT16_C(  4984),  INT16_C(  9967),
        -INT16_C( 16426), -INT16_C(   904),  INT16_C( 23542),  INT16_C( 10100),  INT16_C( 27390), -INT16_C( 24166), -INT16_C( 16061),  INT16_C( 16960) },
      UINT16_C(    8) },
    { { -INT16_C( 20966),  INT16_C(  1627),  INT16_C(  9444),  INT16_C(  4318),  INT16_C(  2933), -INT16_C(   622), -INT16_C( 32265), -INT16_C( 13020),
        -INT16_C( 25536),  INT16_C( 14025),  INT16_C( 16119), -INT16_C(  2467), -INT16_C(  2136), -INT16_C(  5225), -INT16_C( 10312), -INT16_C( 11731) },
      { -INT16_C( 30587),  INT16_C( 27096), -INT16_C( 18771),  INT16_C(  8826),  INT16_C(  2933), -INT16_C( 18144),  INT16_C( 17549), -INT16_C( 12922),
         INT16_C( 20704), -INT16_C( 10237),  INT16_C( 24718),  INT16_C( 14030),  INT16_C( 25943),  INT16_C(  3873),  INT16_C( 20284), -INT16_C( 15902) },
      UINT16_C(   16) },
    { { -INT16_C( 17705), -INT16_C( 31702), -INT16_C( 23439),  INT16_C( 12967), -INT16_C( 14416),  INT16_C( 15851),  INT16_C( 29195), -INT16_C(  5366),
         INT16_C(  3522),  INT16_C( 20675), -INT16_C( 28307), -INT16_C( 15226), -INT16_C( 22538),  INT16_C( 13012), -INT16_C( 18698), -INT16_C( 12557) },
      {  INT16_C(  7792), -INT16_C(  7854), -INT16_C( 23439),  INT16_C( 29460), -INT16_C(    64), -INT16_C( 13392), -INT16_C( 17551),  INT16_C( 13239),
         INT16_C( 31432),  INT16_C( 13955),  INT16_C(  2316),  INT16_C(   762), -INT16_C( 12623), -INT16_C( 22731),  INT16_C( 10372), -INT16_C(  2699) },
      UINT16_C(    4) },
    { { -INT16_C( 14266),  INT16_C( 29498), -INT16_C(  5439), -INT16_C( 32132),  INT16_C( 11498),  INT16_C( 23373),  INT16_C(  1255), -INT16_C( 20337),
         INT16_C(  4735), -INT16_C( 29722), -INT16_C(  8164), -INT16_C( 12915), -INT16_C( 15697),  INT16_C( 13172), -INT16_C(  5397),  INT16_C( 12584) },
      { -INT16_C(    78),  INT16_C( 29498), -INT16_C( 18711), -INT16_C( 11275),  INT16_C( 17379), -INT16_C( 13777), -INT16_C( 16825), -INT16_C( 14726),
         INT16_C( 24784), -INT16_C(  5039), -INT16_C(  8383), -INT16_C(  3911),  INT16_C( 11937), -INT16_C( 29661),  INT16_C( 19480), -INT16_C( 13634) },
      UINT16_C(    2) },
    { { -INT16_C(  1973),  INT16_C( 13373),  INT16_C( 24556), -INT16_C( 28152),  INT16_C( 14198), -INT16_C( 17060), -INT16_C( 10251), -INT16_C( 14972),
        -INT16_C( 10953),  INT16_C( 30898),  INT16_C( 27572),  INT16_C( 22120), -INT16_C( 29543), -INT16_C( 19998), -INT16_C( 24360),  INT16_C(  9083) },
      { -INT16_C( 18023),  INT16_C( 18519),  INT16_C( 24556),  INT16_C( 25306),  INT16_C( 13974), -INT16_C( 29921), -INT16_C( 23795),  INT16_C( 17745),
         INT16_C(   889),  INT16_C( 11709),  INT16_C(  9838),  INT16_C(  2179),  INT16_C( 26290), -INT16_C( 30023),  INT16_C( 13574), -INT16_C( 24659) },
      UINT16_C(    4) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epi16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpeq_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r = easysimd_mm256_cmpeq_epi16_mask(a, b);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t a[8];
    const int32_t b[8];
    const uint8_t r;
  } test_vec[8] = {
    { { -INT32_C(     5245877),  INT32_C(   623865788),  INT32_C(   126711991),  INT32_C(  1624677056), -INT32_C(  1294004335),  INT32_C(   869034878),  INT32_C(   475893395),  INT32_C(  1117117943) },
      { -INT32_C(  1908325166),  INT32_C(  1790144691), -INT32_C(  1452129816),  INT32_C(  1812547803), -INT32_C(   702551720), -INT32_C(   418714796),  INT32_C(  1879271289),  INT32_C(   397580357) },
      UINT8_C(  0) },
    { { -INT32_C(  1868172067),  INT32_C(  1308317796),  INT32_C(  1979084185),  INT32_C(   232914869),  INT32_C(  1021575400),  INT32_C(  1680142059), -INT32_C(  1697372331), -INT32_C(  1649309760) },
      { -INT32_C(   550676869),  INT32_C(  1210853551),  INT32_C(  1270686357),  INT32_C(   173580066), -INT32_C(  1958265697), -INT32_C(  2131793110),  INT32_C(  1377485970), -INT32_C(   957363125) },
      UINT8_C(  0) },
    { { -INT32_C(   777708254), -INT32_C(   618999227),  INT32_C(   371644404),  INT32_C(   371293814), -INT32_C(   442406725),  INT32_C(  1701155027), -INT32_C(  1615298476),  INT32_C(  1852155724) },
      {  INT32_C(   171969476), -INT32_C(   773498147), -INT32_C(  1461187791),  INT32_C(  1153304969),  INT32_C(  1143627633),  INT32_C(  1151963119),  INT32_C(  1541628175), -INT32_C(   842381047) },
      UINT8_C(  0) },
    { {  INT32_C(   836176468), -INT32_C(  1794982812),  INT32_C(  1363012295),  INT32_C(  1687550963),  INT32_C(  1252573018),  INT32_C(  1586385487), -INT32_C(  1128631885),  INT32_C(   243958970) },
      { -INT32_C(   230727282), -INT32_C(   444120546),  INT32_C(   540460332),  INT32_C(   461688000), -INT32_C(   630903413),  INT32_C(   842658687),  INT32_C(   535819108),  INT32_C(    86866295) },
      UINT8_C(  0) },
    { { -INT32_C(   118002214), -INT32_C(   606175569),  INT32_C(    83563587),  INT32_C(  1814003936),  INT32_C(   742818989), -INT32_C(   614563977), -INT32_C(   369472142), -INT32_C(  1578227514) },
      {  INT32_C(  1150936469), -INT32_C(  1491110044),  INT32_C(  1823153036),  INT32_C(  1222167195), -INT32_C(   982245554),  INT32_C(   295817886), -INT32_C(   419783904),  INT32_C(  1485302211) },
      UINT8_C(  0) },
    { {  INT32_C(   849093070),  INT32_C(   618314904),  INT32_C(  1922139607), -INT32_C(  1631950512),  INT32_C(   660876936),  INT32_C(   540542208),  INT32_C(  1678193312), -INT32_C(   356741605) },
      {  INT32_C(  1209817519), -INT32_C(   328403435), -INT32_C(   866189956), -INT32_C(   278259609),  INT32_C(  1175899718),  INT32_C(  1952927443), -INT32_C(  1663537535), -INT32_C(  1400466180) },
      UINT8_C(  0) },
    { {  INT32_C(    49587181),  INT32_C(   367943833), -INT32_C(   975090594), -INT32_C(  1431024540), -INT32_C(   286143718), -INT32_C(  1704830951), -INT32_C(  1036567866), -INT32_C(  1133593138) },
      { -INT32_C(    88186272),  INT32_C(   554675651),  INT32_C(  1592193529),  INT32_C(  1460181565),  INT32_C(  2135292261),  INT32_C(   387557201), -INT32_C(  1327869727),  INT32_C(  1852590094) },
      UINT8_C(  0) },
    { {  INT32_C(  1852320427), -INT32_C(   779126569), -INT32_C(  1506839191),  INT32_C(  1979528975), -INT32_C(  2097921231), -INT32_C(   862319126),  INT32_C(  1820095582),  INT32_C(  1742399676) },
      { -INT32_C(   355122414),  INT32_C(   616260795), -INT32_C(   372577575),  INT32_C(  1381943073), -INT32_C(   170569206), -INT32_C(  1111396513), -INT32_C(  1624687133),  INT32_C(   923206693) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epi32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cmpeq_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 r = easysimd_mm256_cmpeq_epi32_mask(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t a[4];
    const int64_t b[4];
    const uint64_t r;
  } test_vec[8] = {
    { { -INT64_C( 1888920202157954062), -INT64_C( 2024073583914534693), -INT64_C( 9100624727096662526),  INT64_C( 1761461774747457401) },
      { -INT64_C( 1888920202157954062), -INT64_C( 2024073583914534693), -INT64_C( 9100624727096662526), -INT64_C( 1761461774747457401) },
      UINT8_C(  7) },
    { { -INT64_C( 3366538915320673666),  INT64_C( 2676982211990700041), -INT64_C( 1261855060862080012),  INT64_C(  120803908917577713) },
      {  INT64_C( 2664931528877176184),  INT64_C( 2676982211990700041), -INT64_C(   99727849200152787), -INT64_C( 4180033911082180743) },
      UINT8_C(  2) },
    { { -INT64_C( 5890733394757782529), -INT64_C( 3809177091856933982), -INT64_C( 8849830501134280411), -INT64_C( 5840956805756666937) },
      { -INT64_C( 5890733394757782529),  INT64_C( 6571260064356311240), -INT64_C(  668295531691464152), -INT64_C( 5840956805756666937) },
      UINT8_C(  9) },
    { {  INT64_C( 4842051850105838928),  INT64_C( 8742872613459297219),  INT64_C( 7227828377454764438),  INT64_C( 9082548268474419106) },
      {  INT64_C( 4842051850105838928),  INT64_C( 8050907646149914095),  INT64_C( 8175183198356465522),  INT64_C( 9082548268474419106) },
      UINT8_C(  9) },
    { { -INT64_C( 3424346633556167241), -INT64_C( 5890806733870550726),  INT64_C( 3707095086204142130), -INT64_C( 3345760993575223180) },
      {  INT64_C( 2206908866225546734), -INT64_C( 5890806733870550726), -INT64_C( 4795452579717903640),  INT64_C( 8164649556656982626) },
      UINT8_C(  2) },
    { {  INT64_C( 1466168902298916430),  INT64_C( 5495868649950647563), -INT64_C( 1160022643927061536), -INT64_C(  737541785665143089) },
      {  INT64_C( 8574159782329386000),  INT64_C( 5495868649950647563),  INT64_C( 3109555618876594091),  INT64_C( 6698613290458583642) },
      UINT8_C(  2) },
    { { -INT64_C( 5070978765106633147),  INT64_C( 1049112640123380696), -INT64_C( 4949054251498863159),  INT64_C( 6244155191432368127) },
      {  INT64_C(  594229068623310296),  INT64_C( 2933210461183355971),  INT64_C( 7493039145863805664),  INT64_C( 4887630909042505838) },
      UINT8_C(  0) },
    { { -INT64_C( 5152389052858018745), -INT64_C( 8809863694187050443),  INT64_C( 2183890282802635148),  INT64_C( 3398084566879473274) },
      {  INT64_C( 3525234364879663551),  INT64_C( 7717904625374087920),  INT64_C( 2183890282802635148),  INT64_C( 1108496081739834698) },
      UINT8_C(  4) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epi64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpeq_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r = easysimd_mm256_cmpeq_epi64_mask(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const int8_t a[32];
    const int8_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(1508208989),
      {  INT8_C( 100), -INT8_C(   1),  INT8_C( 122),  INT8_C(  74),  INT8_C( 124), -INT8_C(  81), -INT8_C(   7), -INT8_C(  88),
         INT8_C( 105),  INT8_C(  65),  INT8_C(  28), -INT8_C(  64), -INT8_C(  96), -INT8_C(  62), -INT8_C( 103), -INT8_C(  98),
        -INT8_C(  14), -INT8_C( 111),  INT8_C(  15),  INT8_C( 110), -INT8_C(  43),  INT8_C(  99), -INT8_C(  33),  INT8_C(  20),
        -INT8_C(  22), -INT8_C( 101), -INT8_C(  87),  INT8_C(  71),  INT8_C(  12), -INT8_C( 114), -INT8_C(  96),  INT8_C( 112) },
      {  INT8_C( 100), -INT8_C(  76), -INT8_C(  78),  INT8_C(  51),  INT8_C( 124), -INT8_C(  81), -INT8_C(   7), -INT8_C( 107),
         INT8_C( 105), -INT8_C( 115),  INT8_C(  28), -INT8_C( 125),  INT8_C(  30), -INT8_C(  62), -INT8_C( 103), -INT8_C(  98),
        -INT8_C(  90), -INT8_C(  48),  INT8_C(  15), -INT8_C( 112),  INT8_C( 107), -INT8_C(  80), -INT8_C(  33),  INT8_C( 119),
        -INT8_C(  22),  INT8_C( 119), -INT8_C(  87), -INT8_C(  51),  INT8_C(  12), -INT8_C( 114), -INT8_C(  96),  INT8_C( 112) },
      UINT32_C(1363435857) },
    { UINT32_C(1284409687),
      {  INT8_C(  87), -INT8_C( 126), -INT8_C(  30), -INT8_C(  25),  INT8_C(  15),  INT8_C(  22),  INT8_C( 106),  INT8_C(  45),
         INT8_C(  89),  INT8_C(  91),  INT8_C(  32), -INT8_C(   1),  INT8_C(  43),  INT8_C(  39), -INT8_C( 113), -INT8_C( 106),
        -INT8_C(  40),  INT8_C( 102),  INT8_C(  14),  INT8_C(  23), -INT8_C(  35), -INT8_C(  10), -INT8_C(  28),  INT8_C( 111),
        -INT8_C( 103), -INT8_C(  69), -INT8_C(  53), -INT8_C(  16),  INT8_C(  68),  INT8_C(  89),  INT8_C(  60), -INT8_C( 101) },
      {  INT8_C(  52), -INT8_C(  19), -INT8_C(  30), -INT8_C( 115),  INT8_C(  72),  INT8_C(  22), -INT8_C( 116),  INT8_C( 116),
         INT8_C(  89),  INT8_C(  27),  INT8_C(  10),  INT8_C(  55), -INT8_C( 127),  INT8_C(  39), -INT8_C( 113), -INT8_C( 106),
        -INT8_C(  40),  INT8_C(  50),  INT8_C(  14),  INT8_C(  23), -INT8_C(  35), -INT8_C(  10), -INT8_C(  28),  INT8_C(  49),
        -INT8_C( 103), -INT8_C(  44), -INT8_C(  53), -INT8_C(  50),  INT8_C(  68),  INT8_C(  78), -INT8_C(  71),  INT8_C(  39) },
      UINT32_C(  67928324) },
    { UINT32_C(2226442299),
      {  INT8_C(   8),  INT8_C(  65), -INT8_C(   8),  INT8_C( 103),  INT8_C(  92),  INT8_C(   2), -INT8_C(  98), -INT8_C(  34),
         INT8_C(  27), -INT8_C(  20),  INT8_C(  61),  INT8_C(  41),  INT8_C(  30),  INT8_C(  11), -INT8_C(  47),  INT8_C(  11),
        -INT8_C(  91),  INT8_C( 104),  INT8_C(  60), -INT8_C( 104),  INT8_C(  60),  INT8_C(   8),  INT8_C( 102),  INT8_C(  47),
         INT8_C(  86),  INT8_C(  31),  INT8_C(  86), -INT8_C( 110), -INT8_C(  16),  INT8_C(  10),  INT8_C(  22), -INT8_C(   8) },
      {  INT8_C(  16), -INT8_C(   3), -INT8_C(   8),  INT8_C(  43),  INT8_C(  92),  INT8_C(   2),  INT8_C(  85), -INT8_C(  34),
         INT8_C(  27),  INT8_C(  38),  INT8_C(  18),  INT8_C( 115),  INT8_C(  30),  INT8_C(  11), -INT8_C(  47),  INT8_C(  11),
         INT8_C(  86),  INT8_C( 114), -INT8_C(   6), -INT8_C(  84), -INT8_C( 111),  INT8_C(   8),  INT8_C(  62),  INT8_C(  47),
         INT8_C(  86),  INT8_C(  31),  INT8_C(  86), -INT8_C(  90), -INT8_C(  16), -INT8_C(  40),  INT8_C(  22),  INT8_C( 115) },
      UINT32_C(  77647920) },
    { UINT32_C(3198080213),
      { -INT8_C( 105), -INT8_C(  13), -INT8_C(  59),  INT8_C( 101),  INT8_C(  25), -INT8_C(  41), -INT8_C(  39), -INT8_C(  88),
         INT8_C(  37), -INT8_C(  28),  INT8_C( 115),  INT8_C( 123),  INT8_C(  86),  INT8_C( 109),  INT8_C(  40), -INT8_C(  24),
        -INT8_C(  67),  INT8_C( 102),  INT8_C( 105),  INT8_C(  23), -INT8_C(  69), -INT8_C(  29), -INT8_C(  67),  INT8_C(  29),
        -INT8_C(  69),  INT8_C(  11), -INT8_C( 112), -INT8_C( 111), -INT8_C(  33),  INT8_C(  47),  INT8_C(  79),  INT8_C( 118) },
      { -INT8_C( 105), -INT8_C(  75), -INT8_C(  59),  INT8_C( 101),  INT8_C(  25),  INT8_C(  87), -INT8_C(  39), -INT8_C(  88),
        -INT8_C(  60), -INT8_C(  28), -INT8_C(  40),  INT8_C( 123),  INT8_C(  28),  INT8_C( 109),  INT8_C(  40), -INT8_C(  24),
        -INT8_C(  67),  INT8_C( 102), -INT8_C(  12), -INT8_C(  32),  INT8_C(  97), -INT8_C(  29),  INT8_C( 113),  INT8_C(  65),
        -INT8_C(  69),  INT8_C(  11), -INT8_C(  73), -INT8_C(  42), -INT8_C(  43), -INT8_C( 109),  INT8_C(  79),  INT8_C( 118) },
      UINT32_C(2181218517) },
    { UINT32_C(3805607496),
      {  INT8_C(  77),  INT8_C(  97), -INT8_C(  46),  INT8_C(  17),  INT8_C(  23), -INT8_C(  86), -INT8_C( 110),  INT8_C(  51),
        -INT8_C(  21),  INT8_C(  43),  INT8_C(  10),  INT8_C(  16), -INT8_C( 127), -INT8_C(   2), -INT8_C(  16), -INT8_C(  30),
        -INT8_C( 125),  INT8_C(  97),  INT8_C(  35),  INT8_C(  55),  INT8_C(  33), -INT8_C(  37),  INT8_C(  14), -INT8_C(   9),
         INT8_C( 110),  INT8_C(  32), -INT8_C(  71), -INT8_C(  73),  INT8_C(  23), -INT8_C( 115), -INT8_C( 103),  INT8_C( 100) },
      {  INT8_C(  77),  INT8_C(   8),  INT8_C(  56),  INT8_C(   0),  INT8_C(  23),  INT8_C(  66),  INT8_C(  16), -INT8_C(  76),
         INT8_C(  65),  INT8_C(   0),  INT8_C(  10), -INT8_C(  60), -INT8_C( 127), -INT8_C(  70), -INT8_C(   4), -INT8_C(  30),
        -INT8_C( 125),  INT8_C(  10),  INT8_C( 122),  INT8_C(  55),  INT8_C(  42),  INT8_C(  51), -INT8_C(  69), -INT8_C(   9),
        -INT8_C(  64),  INT8_C(  32), -INT8_C(  90), -INT8_C(  73),  INT8_C(  23), -INT8_C( 115), -INT8_C( 103),  INT8_C( 100) },
      UINT32_C(3800077312) },
    { UINT32_C(1490349092),
      {  INT8_C(  47), -INT8_C(  27),  INT8_C(  12),  INT8_C( 112), -INT8_C(  27), -INT8_C(  93),  INT8_C(  52),  INT8_C(  71),
         INT8_C(  94),  INT8_C(  48), -INT8_C(  54), -INT8_C(  13),  INT8_C(  58),  INT8_C(  68), -INT8_C(   9),  INT8_C( 101),
         INT8_C( 119), -INT8_C(  78), -INT8_C(  90),  INT8_C(  55),  INT8_C(   6),  INT8_C(  76), -INT8_C(  27), -INT8_C(  59),
         INT8_C( 104), -INT8_C( 103), -INT8_C( 103), -INT8_C( 115), -INT8_C( 122),  INT8_C( 110), -INT8_C(  27), -INT8_C(  75) },
      { -INT8_C( 107),  INT8_C(  89),  INT8_C(  12),  INT8_C( 112), -INT8_C( 118), -INT8_C(  93), -INT8_C(  26),  INT8_C(  71),
        -INT8_C( 115),  INT8_C(  48), -INT8_C(  54), -INT8_C(  13), -INT8_C( 112), -INT8_C(  48),  INT8_C(  59), -INT8_C( 105),
         INT8_C(  28), -INT8_C(  78),  INT8_C(  92),  INT8_C(  55),  INT8_C(   6), -INT8_C(  10), -INT8_C(  27), -INT8_C(  59),
         INT8_C( 104), -INT8_C( 103), -INT8_C( 103), -INT8_C(  73), -INT8_C(  24),  INT8_C(  26), -INT8_C(  27), -INT8_C(  75) },
      UINT32_C(1087376420) },
    { UINT32_C(4268781428),
      { -INT8_C(  72),  INT8_C(  87), -INT8_C(  62),  INT8_C(  70),  INT8_C(  53), -INT8_C(  20),  INT8_C(  74), -INT8_C(  59),
        -INT8_C(  68), -INT8_C( 122),  INT8_C(  92), -INT8_C(  40), -INT8_C(  89), -INT8_C(  71),  INT8_C(  93),  INT8_C(  97),
        -INT8_C(  81),  INT8_C( 111), -INT8_C(  94),  INT8_C(  19),  INT8_C( 102), -INT8_C( 105), -INT8_C(  54),  INT8_C(  79),
        -INT8_C(  78), -INT8_C(  71), -INT8_C(  52),  INT8_C(  38),  INT8_C(  40),  INT8_C(  61),  INT8_C(  36), -INT8_C(  31) },
      { -INT8_C(  72),  INT8_C(  87), -INT8_C( 114),  INT8_C(  70), -INT8_C(   9), -INT8_C(  20),  INT8_C(  74), -INT8_C(  98),
        -INT8_C(  68), -INT8_C(  60),  INT8_C(   0), -INT8_C(  40), -INT8_C(  89), -INT8_C(  94),  INT8_C( 102), -INT8_C( 102),
         INT8_C(  57),  INT8_C(  48), -INT8_C(  23),  INT8_C(  19),  INT8_C( 102), -INT8_C(  74), -INT8_C(  54),  INT8_C(  79),
        -INT8_C(  13), -INT8_C(  71), -INT8_C(  52), -INT8_C( 121),  INT8_C(  40),  INT8_C(  61),  INT8_C(  80), -INT8_C(  18) },
      UINT32_C( 911214944) },
    { UINT32_C(2206064267),
      { -INT8_C(  55), -INT8_C(  28),  INT8_C(  33),  INT8_C( 109), -INT8_C(  88),  INT8_C(  33), -INT8_C(  64), -INT8_C(  36),
        -INT8_C(  61),  INT8_C(  38),  INT8_C( 119), -INT8_C(   3),  INT8_C(  86),  INT8_C(  96), -INT8_C(  24),  INT8_C(  64),
         INT8_C(  22), -INT8_C(   6),  INT8_C(  82),  INT8_C(   9),  INT8_C(  47),  INT8_C(  69), -INT8_C( 112),  INT8_C(  75),
         INT8_C(  95), -INT8_C(  32),  INT8_C(  58), -INT8_C(  22), -INT8_C(  65), -INT8_C(  73),  INT8_C( 109), -INT8_C( 120) },
      { -INT8_C(  80), -INT8_C(  74),  INT8_C(  33),  INT8_C( 116), -INT8_C(  35),  INT8_C(  33), -INT8_C(  64),  INT8_C(  51),
        -INT8_C(   9),  INT8_C(  89),  INT8_C( 115),  INT8_C(  14),  INT8_C(  86),  INT8_C(  96), -INT8_C(  24), -INT8_C( 125),
         INT8_C(  22), -INT8_C(  88), -INT8_C(  50),  INT8_C(   9), -INT8_C( 120),  INT8_C(   8),  INT8_C(  84),  INT8_C(  71),
        -INT8_C(  65), -INT8_C(  63),  INT8_C(  58), -INT8_C(  22), -INT8_C(  65), -INT8_C(  73), -INT8_C(  98), -INT8_C( 120) },
      UINT32_C(2148093952) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 k = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epi8_mask");

   easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    // easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi8(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_i8x32());
    easysimd__mmask32 r = easysimd_mm256_mask_cmpeq_epi8_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const int16_t a[16];
    const int16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(32570),
      {  INT16_C( 11239), -INT16_C(  9662),  INT16_C(  3487), -INT16_C( 25682),  INT16_C( 28300), -INT16_C( 20212),  INT16_C( 23482), -INT16_C(  1836),
        -INT16_C( 13626), -INT16_C(  1351), -INT16_C(  9990), -INT16_C( 16271),  INT16_C( 20191),  INT16_C(  6912),  INT16_C( 15106), -INT16_C(  5734) },
      {  INT16_C( 11239),  INT16_C( 29162),  INT16_C( 30369), -INT16_C( 25682),  INT16_C( 28300), -INT16_C(  1016), -INT16_C( 12654), -INT16_C(  1836),
        -INT16_C( 16184), -INT16_C(  1351),  INT16_C(  1153), -INT16_C( 32376), -INT16_C( 30177),  INT16_C(  6912),  INT16_C(  9076),  INT16_C( 14230) },
      UINT16_C( 8728) },
    { UINT16_C(32809),
      { -INT16_C( 13655), -INT16_C( 30217),  INT16_C(  8055),  INT16_C( 32547), -INT16_C( 19173), -INT16_C(  7859),  INT16_C(  5377),  INT16_C(  9890),
         INT16_C(  9039), -INT16_C( 10454),  INT16_C( 18852),  INT16_C( 24929), -INT16_C( 11006), -INT16_C( 26236), -INT16_C( 21235), -INT16_C( 18919) },
      { -INT16_C(  4545),  INT16_C( 25135),  INT16_C( 19053),  INT16_C( 32547),  INT16_C(  6444), -INT16_C( 12593),  INT16_C(  7743),  INT16_C(  9890),
         INT16_C(  9039), -INT16_C( 10454),  INT16_C( 18852),  INT16_C( 24929),  INT16_C( 14669), -INT16_C( 26236), -INT16_C( 21235), -INT16_C( 18919) },
      UINT16_C(32776) },
    { UINT16_C(42892),
      { -INT16_C(  1648), -INT16_C( 22287),  INT16_C(  7603), -INT16_C( 31807),  INT16_C(   235), -INT16_C(  9055), -INT16_C( 26775),  INT16_C(  7026),
         INT16_C( 26862),  INT16_C(  6863),  INT16_C(  7651),  INT16_C(  2644),  INT16_C( 17284), -INT16_C(  1111),  INT16_C( 13682),  INT16_C(   674) },
      { -INT16_C(  7510),  INT16_C( 27569), -INT16_C( 25243),  INT16_C(  1899),  INT16_C(   235), -INT16_C(  5218), -INT16_C( 26775),  INT16_C(  7026),
         INT16_C( 26862),  INT16_C(  6863),  INT16_C( 24641),  INT16_C(  2644), -INT16_C( 20389), -INT16_C(  1111),  INT16_C( 13682),  INT16_C( 23954) },
      UINT16_C( 9088) },
    { UINT16_C(17201),
      { -INT16_C( 26680),  INT16_C( 13536),  INT16_C( 23198),  INT16_C( 15368), -INT16_C(  1979), -INT16_C( 26168),  INT16_C( 28343), -INT16_C( 27696),
         INT16_C(  4713), -INT16_C( 22541),  INT16_C( 20220),  INT16_C(  7255),  INT16_C(  2380), -INT16_C(  8597), -INT16_C( 25242),  INT16_C( 12066) },
      { -INT16_C( 26680),  INT16_C( 13536), -INT16_C( 24050),  INT16_C( 15368),  INT16_C(  6715),  INT16_C(  3140),  INT16_C( 28343), -INT16_C( 27696),
         INT16_C(  4713), -INT16_C( 21521),  INT16_C( 20220),  INT16_C(  7255),  INT16_C(  2380), -INT16_C(  8597), -INT16_C( 25242),  INT16_C( 12066) },
      UINT16_C(16641) },
    { UINT16_C(39749),
      {  INT16_C( 21273),  INT16_C( 31805),  INT16_C( 30761),  INT16_C( 28311),  INT16_C( 17540), -INT16_C( 24037),  INT16_C( 28901), -INT16_C( 11075),
        -INT16_C(  3045), -INT16_C( 12273),  INT16_C( 10646), -INT16_C( 10516),  INT16_C( 13925), -INT16_C( 23479), -INT16_C( 28700), -INT16_C(   705) },
      {  INT16_C( 21273),  INT16_C(  4340),  INT16_C( 30761),  INT16_C( 28311),  INT16_C( 17540), -INT16_C( 10235),  INT16_C(  8462),  INT16_C(  7628),
        -INT16_C(  3045), -INT16_C( 12273), -INT16_C( 21447), -INT16_C( 32236), -INT16_C(  1968), -INT16_C( 28911), -INT16_C(  2827), -INT16_C(   705) },
      UINT16_C(33541) },
    { UINT16_C(    0),
      {  INT16_C( 31359), -INT16_C( 11143), -INT16_C( 27633),  INT16_C(  5390),  INT16_C(  7277),  INT16_C( 14646),  INT16_C( 10041), -INT16_C( 32612),
        -INT16_C( 11003),  INT16_C(  6445),  INT16_C( 32088),  INT16_C( 26897),  INT16_C(  1549),  INT16_C(  6237),  INT16_C( 23924), -INT16_C(  3304) },
      { -INT16_C(  6201), -INT16_C( 10970), -INT16_C( 27652),  INT16_C(  5390),  INT16_C( 11212),  INT16_C( 14646),  INT16_C( 24235), -INT16_C( 10178),
        -INT16_C( 27017),  INT16_C(  6445),  INT16_C( 32088),  INT16_C( 26897),  INT16_C(   891),  INT16_C(  6237),  INT16_C( 23924), -INT16_C( 16859) },
      UINT16_C(    0) },
    { UINT16_C(19321),
      {  INT16_C( 30099), -INT16_C( 31266), -INT16_C( 21593),  INT16_C(   432),  INT16_C( 23316),  INT16_C( 21087), -INT16_C( 10444), -INT16_C( 29975),
        -INT16_C(  5793), -INT16_C(  4371),  INT16_C( 26694),  INT16_C(   497), -INT16_C(  6148),  INT16_C(  8852),  INT16_C(  3493),  INT16_C( 14701) },
      {  INT16_C( 10942),  INT16_C( 28407), -INT16_C( 21593),  INT16_C(   432),  INT16_C( 23316),  INT16_C( 21087), -INT16_C( 10444),  INT16_C( 29743),
        -INT16_C(  5793), -INT16_C(  4371), -INT16_C(  9865),  INT16_C(  2952), -INT16_C(  6148),  INT16_C(  8852), -INT16_C( 25498),  INT16_C( 14701) },
      UINT16_C(  888) },
    { UINT16_C(44230),
      { -INT16_C(  3438),  INT16_C( 23735),  INT16_C(  5245), -INT16_C(  8359), -INT16_C(  7845), -INT16_C( 30048),  INT16_C( 20565),  INT16_C( 12800),
         INT16_C( 30960),  INT16_C( 30732),  INT16_C(  1923), -INT16_C( 25434),  INT16_C(  3184),  INT16_C(  9528), -INT16_C(   207), -INT16_C( 15407) },
      {  INT16_C( 28191),  INT16_C( 23735),  INT16_C(  5245), -INT16_C(  8359), -INT16_C( 20606), -INT16_C( 31939),  INT16_C( 12002), -INT16_C(  4357),
         INT16_C( 30960),  INT16_C( 30732),  INT16_C(  1923),  INT16_C( 21337),  INT16_C(  3184),  INT16_C(  9528), -INT16_C(   207),  INT16_C( 28133) },
      UINT16_C( 9222) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epi16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epi16_mask");

   easysimd_assert_equal_mmask16(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi16(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i16x16());
    easysimd__mmask16 r = easysimd_mm256_mask_cmpeq_epi16_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int32_t a[8];
    const int32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(133),
      {  INT32_C(  1325808442), -INT32_C(  1934238417),  INT32_C(   165830210), -INT32_C(   442710773), -INT32_C(  1001509002), -INT32_C(  1750109234),  INT32_C(   171272587), -INT32_C(   259035979) },
      { -INT32_C(  1268583310),  INT32_C(  1589462099), -INT32_C(  2143004151), -INT32_C(   442710773), -INT32_C(  1001509002),  INT32_C(   312157276),  INT32_C(   171272587),  INT32_C(   823017919) },
      UINT8_C(  0) },
    { UINT8_C( 53),
      { -INT32_C(  1249319567), -INT32_C(    54532190), -INT32_C(  1115275477),  INT32_C(  1992547715), -INT32_C(   254651188),  INT32_C(   427877615),  INT32_C(   668514534),  INT32_C(   945621447) },
      { -INT32_C(  1249319567),  INT32_C(  1857360875), -INT32_C(  1109100303), -INT32_C(  1129465907), -INT32_C(   254651188), -INT32_C(  1381388570), -INT32_C(  1494939976), -INT32_C(  1204301076) },
      UINT8_C( 17) },
    { UINT8_C(127),
      { -INT32_C(   664096571), -INT32_C(   573974171),  INT32_C(  1957332669),  INT32_C(  1628464947), -INT32_C(   347631043),  INT32_C(  1118041147), -INT32_C(  1423029543),  INT32_C(  1177216641) },
      { -INT32_C(   664096571), -INT32_C(  1566538897),  INT32_C(   654553322),  INT32_C(  1628464947),  INT32_C(   398439998),  INT32_C(  1118041147), -INT32_C(  1423029543), -INT32_C(   261691518) },
      UINT8_C(105) },
    { UINT8_C(208),
      { -INT32_C(  1036051393),  INT32_C(  1823334844),  INT32_C(   799986917), -INT32_C(  1636946969), -INT32_C(   660699669),  INT32_C(   917250120), -INT32_C(   793241623), -INT32_C(  1130321539) },
      { -INT32_C(  1494406207),  INT32_C(  1823334844), -INT32_C(  1316666426), -INT32_C(  1636946969), -INT32_C(   660699669),  INT32_C(   917250120),  INT32_C(  1854316201),  INT32_C(  1072629118) },
      UINT8_C( 16) },
    { UINT8_C( 48),
      {  INT32_C(  1999693276), -INT32_C(    29550406), -INT32_C(  1043861603), -INT32_C(  1358505863),  INT32_C(   553464608), -INT32_C(   792082639),  INT32_C(  1397634894),  INT32_C(    42175782) },
      {  INT32_C(  1999693276),  INT32_C(   506196389), -INT32_C(  1043861603), -INT32_C(  1358505863), -INT32_C(   530132078),  INT32_C(   808694794), -INT32_C(  1288521919),  INT32_C(    42175782) },
      UINT8_C(  0) },
    { UINT8_C( 97),
      { -INT32_C(    83386358), -INT32_C(    54188597), -INT32_C(  1100894221), -INT32_C(  1940785223),  INT32_C(   295055709),  INT32_C(   475186789), -INT32_C(  1551432200), -INT32_C(   301613340) },
      { -INT32_C(    83386358),  INT32_C(   417564511), -INT32_C(   609930114), -INT32_C(   873710746), -INT32_C(    85508606),  INT32_C(   475186789), -INT32_C(  1978227924), -INT32_C(   301613340) },
      UINT8_C( 33) },
    { UINT8_C(176),
      {  INT32_C(  1645151707),  INT32_C(   501295081),  INT32_C(   142851276), -INT32_C(   418754903), -INT32_C(  1557461963),  INT32_C(  1171215266),  INT32_C(  1794398825), -INT32_C(   384117490) },
      {  INT32_C(  1645151707), -INT32_C(  1843043095),  INT32_C(   142851276), -INT32_C(   695425741),  INT32_C(  1629182968),  INT32_C(  1171215266), -INT32_C(  1079122459),  INT32_C(  1602484494) },
      UINT8_C( 32) },
    { UINT8_C( 38),
      { -INT32_C(   938443676), -INT32_C(   797203807), -INT32_C(  1056742372),  INT32_C(  1270471152),  INT32_C(    83958773), -INT32_C(   857029146),  INT32_C(  1826269554), -INT32_C(  1869465300) },
      {  INT32_C(  1629934661), -INT32_C(   797203807), -INT32_C(   126559229),  INT32_C(  1270471152), -INT32_C(  1398085830), -INT32_C(  1122466671),  INT32_C(  1826269554), -INT32_C(  1832212659) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epi32_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi32(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_i32x8());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpeq_epi32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 62),
      {  INT64_C(  905154881773503887),  INT64_C( 5520636860476881664),  INT64_C( 6280918037372222432), -INT64_C( 1169420361499206972) },
      { -INT64_C( 4001007467064218828),  INT64_C( 8489657694765474764),  INT64_C( 6250084713797356652), -INT64_C( 1169420361499206972) },
      UINT8_C(  8) },
    { UINT8_C(254),
      { -INT64_C( 2351530485153857024),  INT64_C( 3717446232949446360), -INT64_C( 8710111072169597950), -INT64_C( 2298810669842574394) },
      { -INT64_C( 1116812523556706139),  INT64_C( 3717446232949446360),  INT64_C( 8387975705785957287),  INT64_C( 5568777530354030866) },
      UINT8_C(  2) },
    { UINT8_C( 45),
      { -INT64_C( 4904690365583681030), -INT64_C( 8413931296478872385),  INT64_C(  348594851983508400), -INT64_C( 2678241135097505950) },
      { -INT64_C( 6601563522901131833),  INT64_C( 4358145068614771787),  INT64_C(  348594851983508400), -INT64_C( 2678241135097505950) },
      UINT8_C( 12) },
    { UINT8_C(  6),
      {  INT64_C( 2052125994625095334), -INT64_C( 7847748385108463355),  INT64_C(  537168999066422293),  INT64_C( 1555778641869018569) },
      {  INT64_C( 2052125994625095334), -INT64_C( 7656990400033168550),  INT64_C(  537168999066422293),  INT64_C( 1555778641869018569) },
      UINT8_C(  4) },
    { UINT8_C( 89),
      {  INT64_C( 6849569299295194407), -INT64_C(  296181490609568768), -INT64_C( 7929436183225807129),  INT64_C( 5324737049605140880) },
      { -INT64_C( 4590249006360946480), -INT64_C(  296181490609568768), -INT64_C( 7929436183225807129),  INT64_C( 5324737049605140880) },
      UINT8_C(  8) },
    { UINT8_C( 80),
      { -INT64_C( 6024151988992122169),  INT64_C( 7309957361905871182), -INT64_C( 3888490400317350075),  INT64_C( 8800628176172298137) },
      { -INT64_C(  839185001493069704), -INT64_C( 3216909087560897540),  INT64_C(  187681700491650669),  INT64_C( 5389412177820975890) },
      UINT8_C(  0) },
    { UINT8_C(137),
      { -INT64_C( 8750725409039367946), -INT64_C( 2157906482958426014), -INT64_C( 3813871825748415176),  INT64_C( 1885574638019107223) },
      { -INT64_C( 8750725409039367946), -INT64_C( 2157906482958426014), -INT64_C( 3813871825748415176), -INT64_C( 4192824414404858803) },
      UINT8_C(  1) },
    { UINT8_C( 33),
      {  INT64_C( 8921790677600018018), -INT64_C( 4428406478314707611), -INT64_C( 7414414353305636893),  INT64_C( 3080400330308122501) },
      {  INT64_C( 8921790677600018018), -INT64_C( 3729214384005607112),  INT64_C( 8004067560308490531),  INT64_C( 3080400330308122501) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epi64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi64(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_i64x4());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpeq_epi64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t a[32];
    const uint8_t b[32];
    const uint32_t r;
  } test_vec[8] = {
    { { UINT8_C(185), UINT8_C(146), UINT8_C(225), UINT8_C(192), UINT8_C( 94), UINT8_C( 31), UINT8_C(247), UINT8_C( 14),
        UINT8_C( 46), UINT8_C(155), UINT8_C( 25), UINT8_C(171), UINT8_C( 58), UINT8_C(181), UINT8_C( 93), UINT8_C(195),
        UINT8_C(  9), UINT8_C(230), UINT8_C( 63), UINT8_C(153), UINT8_C(181), UINT8_C(188), UINT8_C(211), UINT8_C( 87),
        UINT8_C(196), UINT8_C(  3), UINT8_C(128), UINT8_C(245), UINT8_C(136), UINT8_C( 61),    UINT8_MAX, UINT8_C( 65) },
      { UINT8_C(207), UINT8_C(224), UINT8_C(  2), UINT8_C( 45),    UINT8_MAX, UINT8_C(249), UINT8_C( 59), UINT8_C( 45),
        UINT8_C( 46), UINT8_C(155), UINT8_C( 25), UINT8_C(171), UINT8_C( 58), UINT8_C(181), UINT8_C( 93), UINT8_C(195),
        UINT8_C( 28), UINT8_C(209), UINT8_C(172), UINT8_C(210), UINT8_C(141), UINT8_C(128), UINT8_C( 41), UINT8_C( 81),
        UINT8_C(131), UINT8_C(169), UINT8_C( 71), UINT8_C( 12), UINT8_C(230), UINT8_C( 70), UINT8_C( 77), UINT8_C(181) },
      UINT32_C(     65280) },
    { { UINT8_C( 38), UINT8_C( 79), UINT8_C(227), UINT8_C( 37), UINT8_C( 72), UINT8_C( 30), UINT8_C( 83), UINT8_C(220),
        UINT8_C(115), UINT8_C( 43), UINT8_C(171), UINT8_C(125), UINT8_C( 97), UINT8_C( 60), UINT8_C(144), UINT8_C(126),
        UINT8_C( 13), UINT8_C( 60), UINT8_C( 80), UINT8_C(155), UINT8_C(188), UINT8_C(121), UINT8_C(236), UINT8_C( 64),
        UINT8_C( 35), UINT8_C( 51), UINT8_C( 76), UINT8_C(  9), UINT8_C(121), UINT8_C(153), UINT8_C(191), UINT8_C(159) },
      { UINT8_C(233), UINT8_C(162), UINT8_C(197), UINT8_C( 49), UINT8_C(192), UINT8_C( 24), UINT8_C( 14), UINT8_C( 51),
        UINT8_C( 67), UINT8_C(185), UINT8_C(176), UINT8_C(165), UINT8_C(245), UINT8_C( 64), UINT8_C( 35), UINT8_C(  3),
        UINT8_C( 13), UINT8_C( 60), UINT8_C( 80), UINT8_C(155), UINT8_C(188), UINT8_C(121), UINT8_C(236), UINT8_C( 64),
        UINT8_C(190), UINT8_C(197), UINT8_C( 25), UINT8_C( 55), UINT8_C( 95), UINT8_C(216), UINT8_C(215), UINT8_C( 72) },
      UINT32_C(  16711680) },
    { { UINT8_C(122), UINT8_C(156), UINT8_C(121), UINT8_C( 58), UINT8_C(180), UINT8_C(135), UINT8_C(110), UINT8_C(247),
        UINT8_C( 64), UINT8_C( 30), UINT8_C(156), UINT8_C( 54), UINT8_C( 95), UINT8_C(191), UINT8_C( 57), UINT8_C(220),
        UINT8_C( 50), UINT8_C(215), UINT8_C( 21), UINT8_C( 31), UINT8_C( 97), UINT8_C(143), UINT8_C( 46), UINT8_C( 31),
        UINT8_C( 84), UINT8_C( 71), UINT8_C( 87), UINT8_C(179), UINT8_C( 31), UINT8_C( 46), UINT8_C(251), UINT8_C(153) },
      { UINT8_C(202), UINT8_C(117), UINT8_C(121), UINT8_C(126), UINT8_C(252), UINT8_C( 66), UINT8_C(117), UINT8_C( 61),
        UINT8_C( 96), UINT8_C( 18), UINT8_C(115), UINT8_C(191), UINT8_C(209), UINT8_C(172), UINT8_C(155), UINT8_C(  4),
        UINT8_C(131), UINT8_C(177), UINT8_C( 35), UINT8_C(228), UINT8_C( 64), UINT8_C( 81), UINT8_C(  4), UINT8_C(148),
        UINT8_C(153), UINT8_C( 91), UINT8_C( 72), UINT8_C(184), UINT8_C(137), UINT8_C( 67), UINT8_C( 82), UINT8_C( 83) },
      UINT32_C(         4) },
    { { UINT8_C(184), UINT8_C( 38), UINT8_C(209), UINT8_C(181), UINT8_C(104), UINT8_C( 70), UINT8_C(242), UINT8_C(200),
        UINT8_C( 88), UINT8_C(101), UINT8_C(136), UINT8_C( 42), UINT8_C( 17), UINT8_C( 35), UINT8_C( 46), UINT8_C(148),
        UINT8_C(212), UINT8_C( 81), UINT8_C(120), UINT8_C( 20), UINT8_C(162), UINT8_C(124), UINT8_C(169), UINT8_C( 59),
        UINT8_C(215), UINT8_C(241), UINT8_C(244), UINT8_C( 96), UINT8_C( 52), UINT8_C( 70), UINT8_C(179), UINT8_C(237) },
      { UINT8_C(108), UINT8_C(132), UINT8_C(162), UINT8_C(181), UINT8_C(203), UINT8_C(148), UINT8_C(156), UINT8_C( 35),
        UINT8_C(249), UINT8_C( 36), UINT8_C( 77), UINT8_C( 10), UINT8_C( 72), UINT8_C(123), UINT8_C(158), UINT8_C( 28),
        UINT8_C(204), UINT8_C( 22), UINT8_C( 49), UINT8_C(111), UINT8_C(147), UINT8_C(218), UINT8_C(170), UINT8_C(106),
        UINT8_C(203), UINT8_C(158), UINT8_C(203),    UINT8_MAX, UINT8_C(228), UINT8_C(126), UINT8_C(236), UINT8_C( 80) },
      UINT32_C(         8) },
    { { UINT8_C(  3), UINT8_C(142), UINT8_C( 36), UINT8_C(206), UINT8_C( 34), UINT8_C(227), UINT8_C(241), UINT8_C( 27),
        UINT8_C(229), UINT8_C( 63), UINT8_C( 37), UINT8_C( 45), UINT8_C(186), UINT8_C(195), UINT8_C( 74), UINT8_C(135),
        UINT8_C(218), UINT8_C(123), UINT8_C(246), UINT8_C(109), UINT8_C( 85), UINT8_C(160), UINT8_C(215), UINT8_C( 32),
        UINT8_C( 63), UINT8_C(162), UINT8_C( 31), UINT8_C( 35), UINT8_C( 33), UINT8_C( 12), UINT8_C(116), UINT8_C( 36) },
      { UINT8_C(154), UINT8_C(152), UINT8_C(242), UINT8_C(189), UINT8_C( 89), UINT8_C(227), UINT8_C(216), UINT8_C( 63),
        UINT8_C( 34), UINT8_C(254), UINT8_C(108), UINT8_C(221), UINT8_C(193), UINT8_C(182), UINT8_C(100), UINT8_C(155),
        UINT8_C( 49), UINT8_C( 90), UINT8_C(  8), UINT8_C(134), UINT8_C(250), UINT8_C(224), UINT8_C(166), UINT8_C( 57),
        UINT8_C(130), UINT8_C(198), UINT8_C( 93), UINT8_C(163), UINT8_C(210), UINT8_C(209), UINT8_C(199), UINT8_C(108) },
      UINT32_C(        32) },
    { { UINT8_C(105), UINT8_C(185), UINT8_C( 41), UINT8_C(195), UINT8_C(157), UINT8_C(  2), UINT8_C(  2), UINT8_C(191),
        UINT8_C(  0), UINT8_C(110), UINT8_C(156), UINT8_C(193), UINT8_C( 37), UINT8_C(  0), UINT8_C( 93), UINT8_C( 86),
        UINT8_C( 90), UINT8_C(101), UINT8_C(221), UINT8_C( 85), UINT8_C( 69), UINT8_C(131), UINT8_C(142), UINT8_C(200),
        UINT8_C( 73), UINT8_C(235), UINT8_C(107), UINT8_C( 27), UINT8_C(188), UINT8_C( 51), UINT8_C(242), UINT8_C( 38) },
      { UINT8_C(236), UINT8_C(177), UINT8_C(233), UINT8_C(137), UINT8_C(179), UINT8_C(235), UINT8_C( 73), UINT8_C(179),
        UINT8_C( 89), UINT8_C(229), UINT8_C(117), UINT8_C(126), UINT8_C(230), UINT8_C(210), UINT8_C(213), UINT8_C( 64),
        UINT8_C( 55), UINT8_C(178), UINT8_C(149), UINT8_C(125), UINT8_C( 53), UINT8_C( 36), UINT8_C( 69), UINT8_C(127),
        UINT8_C( 15), UINT8_C(176), UINT8_C(154), UINT8_C(204), UINT8_C(227), UINT8_C( 34), UINT8_C(242), UINT8_C(208) },
      UINT32_C(1073741824) },
    { { UINT8_C(212), UINT8_C(219), UINT8_C( 89), UINT8_C(135), UINT8_C(198), UINT8_C(162), UINT8_C( 59), UINT8_C( 31),
        UINT8_C(136), UINT8_C(176), UINT8_C(158), UINT8_C(110), UINT8_C(130), UINT8_C(115), UINT8_C(174), UINT8_C(185),
        UINT8_C( 37), UINT8_C( 68), UINT8_C( 54), UINT8_C( 90), UINT8_C(104), UINT8_C(123), UINT8_C(217), UINT8_C(119),
        UINT8_C( 44), UINT8_C(116), UINT8_C( 67), UINT8_C(106), UINT8_C(150), UINT8_C( 53), UINT8_C(223), UINT8_C(106) },
      { UINT8_C( 16), UINT8_C( 57), UINT8_C(242), UINT8_C(214), UINT8_C(219), UINT8_C( 45), UINT8_C(246), UINT8_C( 99),
        UINT8_C(221), UINT8_C(148), UINT8_C(209), UINT8_C( 95), UINT8_C(  7), UINT8_C(128), UINT8_C( 24), UINT8_C( 44),
        UINT8_C(196), UINT8_C( 79), UINT8_C(134), UINT8_C( 44), UINT8_C(202), UINT8_C( 96), UINT8_C(163), UINT8_C(246),
        UINT8_C(212), UINT8_C(231), UINT8_C(  6), UINT8_C(106), UINT8_C( 28), UINT8_C(229), UINT8_C(213), UINT8_C( 45) },
      UINT32_C( 134217728) },
    { { UINT8_C( 30), UINT8_C(199), UINT8_C(  3), UINT8_C(250), UINT8_C(244), UINT8_C(249), UINT8_C( 93), UINT8_C(209),
        UINT8_C(141), UINT8_C( 47), UINT8_C( 48), UINT8_C(148), UINT8_C(175), UINT8_C( 72), UINT8_C(192), UINT8_C(115),
        UINT8_C(151), UINT8_C( 71), UINT8_C(184), UINT8_C( 98), UINT8_C(167), UINT8_C( 66), UINT8_C( 88), UINT8_C(123),
        UINT8_C( 41), UINT8_C( 94), UINT8_C(229), UINT8_C( 70), UINT8_C( 68), UINT8_C(186), UINT8_C(115), UINT8_C( 98) },
      { UINT8_C(129), UINT8_C(118), UINT8_C( 92), UINT8_C(117), UINT8_C(112), UINT8_C(186), UINT8_C( 70), UINT8_C(253),
        UINT8_C(233), UINT8_C(118), UINT8_C(146), UINT8_C(152), UINT8_C(191), UINT8_C( 82), UINT8_C( 11), UINT8_C( 86),
        UINT8_C(153), UINT8_C(170), UINT8_C(184), UINT8_C( 64), UINT8_C(236), UINT8_C( 17), UINT8_C(187), UINT8_C( 22),
        UINT8_C(111), UINT8_C(161), UINT8_C( 92), UINT8_C(179), UINT8_C( 91), UINT8_C(207), UINT8_C( 22), UINT8_C(221) },
      UINT32_C(    262144) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epu8_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpeq_epu8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_test_x86_random_u8x32();
    easysimd__mmask32 r = easysimd_mm256_cmpeq_epu8_mask(a, b);

    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint16_t a[16];
    const uint16_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { { UINT16_C(62428), UINT16_C(16364), UINT16_C(60792), UINT16_C(  923), UINT16_C(39911), UINT16_C(42450), UINT16_C(44790), UINT16_C(22197),
        UINT16_C( 9500), UINT16_C(59470), UINT16_C(17128), UINT16_C( 8090), UINT16_C(34282), UINT16_C( 5154), UINT16_C(37320), UINT16_C(42177) },
      { UINT16_C(44677), UINT16_C(64995), UINT16_C(32411), UINT16_C(33281), UINT16_C(54042), UINT16_C( 4135), UINT16_C(56705), UINT16_C(40294),
        UINT16_C( 9500), UINT16_C(59470), UINT16_C(17128), UINT16_C( 8090), UINT16_C(34282), UINT16_C( 5154), UINT16_C(37320), UINT16_C(42177) },
      UINT16_C(65280) },
    { { UINT16_C(62822), UINT16_C(  319), UINT16_C(16499), UINT16_C(36227), UINT16_C(43539), UINT16_C(38046), UINT16_C( 1159), UINT16_C(35121),
        UINT16_C(47033), UINT16_C(45427), UINT16_C(31959), UINT16_C(31891), UINT16_C(35239), UINT16_C(25834), UINT16_C(64321), UINT16_C(42917) },
      { UINT16_C(62822), UINT16_C(25768), UINT16_C(11044), UINT16_C(14321), UINT16_C(36822), UINT16_C(24011), UINT16_C(64916), UINT16_C(19943),
        UINT16_C(23220), UINT16_C(35838), UINT16_C(37335), UINT16_C(32263), UINT16_C(61723), UINT16_C(23778), UINT16_C(35053), UINT16_C(56580) },
      UINT16_C(    1) },
    { { UINT16_C(44140), UINT16_C(37185), UINT16_C(13272), UINT16_C(44744), UINT16_C(38082), UINT16_C(22027), UINT16_C(62097), UINT16_C(17828),
        UINT16_C(41549), UINT16_C( 9424), UINT16_C(55092), UINT16_C(20386), UINT16_C(34249), UINT16_C(46763), UINT16_C(44813), UINT16_C(31123) },
      { UINT16_C(54620), UINT16_C(13322), UINT16_C(13272), UINT16_C(51938), UINT16_C(60775), UINT16_C(63521), UINT16_C(50656), UINT16_C(11581),
        UINT16_C( 3431), UINT16_C(39761), UINT16_C(62436), UINT16_C(44522), UINT16_C(38520), UINT16_C(34147), UINT16_C(63301), UINT16_C(41471) },
      UINT16_C(    4) },
    { { UINT16_C( 2508), UINT16_C(54485), UINT16_C(47068), UINT16_C(17310), UINT16_C(49061), UINT16_C(34107), UINT16_C(30852), UINT16_C(60594),
        UINT16_C(  901), UINT16_C(27271), UINT16_C(29430), UINT16_C(28439), UINT16_C(31496), UINT16_C(19956), UINT16_C(62322), UINT16_C(16111) },
      { UINT16_C(50429), UINT16_C(55570), UINT16_C(45180), UINT16_C( 8477), UINT16_C(49061), UINT16_C(62630), UINT16_C(22737), UINT16_C(22240),
        UINT16_C(26715), UINT16_C(20928), UINT16_C(55514), UINT16_C(58048), UINT16_C(46419), UINT16_C(50479), UINT16_C( 7848), UINT16_C(42243) },
      UINT16_C(   16) },
    { { UINT16_C( 5603), UINT16_C(24447), UINT16_C(40133), UINT16_C(13696), UINT16_C( 9972), UINT16_C(50474), UINT16_C( 2686), UINT16_C(55580),
        UINT16_C(56434), UINT16_C(19498), UINT16_C(60340), UINT16_C( 1838), UINT16_C(24224), UINT16_C(56505), UINT16_C(53116), UINT16_C(24558) },
      { UINT16_C(28132), UINT16_C(43710), UINT16_C(15881), UINT16_C(64991), UINT16_C( 2404), UINT16_C(58051), UINT16_C(57108), UINT16_C(34491),
        UINT16_C(59067), UINT16_C(28883), UINT16_C(  465), UINT16_C(29047), UINT16_C(17503), UINT16_C(56505), UINT16_C(42771), UINT16_C(63547) },
      UINT16_C( 8192) },
    { { UINT16_C(64020), UINT16_C( 7586), UINT16_C(33080), UINT16_C(40219), UINT16_C(56971), UINT16_C(40831), UINT16_C(15293), UINT16_C(30757),
        UINT16_C(63521), UINT16_C(62184), UINT16_C(24826), UINT16_C(22883), UINT16_C( 7332), UINT16_C(46901), UINT16_C(30741), UINT16_C(55471) },
      { UINT16_C(20843), UINT16_C(41974), UINT16_C( 4563), UINT16_C(24128), UINT16_C(49391), UINT16_C(44285), UINT16_C( 8955), UINT16_C( 7204),
        UINT16_C( 3355), UINT16_C( 5390), UINT16_C(29037), UINT16_C( 4462), UINT16_C(42125), UINT16_C(20936), UINT16_C(30741), UINT16_C(32810) },
      UINT16_C(16384) },
    { { UINT16_C( 8393), UINT16_C(39971), UINT16_C(25649), UINT16_C( 8442), UINT16_C(63268), UINT16_C( 8140), UINT16_C(61466), UINT16_C(13627),
        UINT16_C(18941), UINT16_C(27210), UINT16_C(47290), UINT16_C(18299), UINT16_C(17500), UINT16_C(29081), UINT16_C(50108), UINT16_C(34289) },
      { UINT16_C( 5603), UINT16_C( 5154), UINT16_C( 7289), UINT16_C(40244), UINT16_C(   20), UINT16_C(11964), UINT16_C(63472), UINT16_C(61027),
        UINT16_C(44352), UINT16_C(64088), UINT16_C(47290), UINT16_C(49729), UINT16_C(55832), UINT16_C(54323), UINT16_C( 9629), UINT16_C(32857) },
      UINT16_C( 1024) },
    { { UINT16_C(31546), UINT16_C(45972), UINT16_C(51352), UINT16_C(44112), UINT16_C( 3272), UINT16_C(47578), UINT16_C(64751), UINT16_C(17319),
        UINT16_C(65514), UINT16_C(20285), UINT16_C(32467), UINT16_C(60177), UINT16_C(17753), UINT16_C(63167), UINT16_C( 6506), UINT16_C(42103) },
      { UINT16_C( 2964), UINT16_C(11351), UINT16_C(42964), UINT16_C(44112), UINT16_C(45747), UINT16_C(46677), UINT16_C(64751), UINT16_C(55801),
        UINT16_C(14076), UINT16_C(53033), UINT16_C(15028), UINT16_C( 3515), UINT16_C(31359), UINT16_C(59652), UINT16_C(31635), UINT16_C(10381) },
      UINT16_C(   72) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epu16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpeq_epu16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__mmask16 r = easysimd_mm256_cmpeq_epu16_mask(a, b);

    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint32_t a[8];
    const uint32_t b[8];
    const uint8_t r;
  } test_vec[8] = {
    { { UINT32_C(1299773304), UINT32_C( 369448387), UINT32_C(1900648867), UINT32_C(3352978152), UINT32_C( 761227013), UINT32_C(1491951332), UINT32_C(3884072564), UINT32_C(3236063303) },
      { UINT32_C(1299773304), UINT32_C(1396183984), UINT32_C(1900648867), UINT32_C(3915096036), UINT32_C(1159182855), UINT32_C(1491951332), UINT32_C(3812585372), UINT32_C( 933438168) },
      UINT8_C( 37) },
    { { UINT32_C( 760918397), UINT32_C(1887474372), UINT32_C(4161094932), UINT32_C(3974193381), UINT32_C(4281464859), UINT32_C(4065861462), UINT32_C(1725339534), UINT32_C( 916355513) },
      { UINT32_C(3999529002), UINT32_C(2657084554), UINT32_C( 244737065), UINT32_C(3757734084), UINT32_C(3353291888), UINT32_C(3652794187), UINT32_C(2285932494), UINT32_C( 868146697) },
      UINT8_C(  0) },
    { { UINT32_C(1612849878), UINT32_C( 805273862), UINT32_C(2147240306), UINT32_C(2122921997), UINT32_C(2940561508), UINT32_C(1837760159), UINT32_C(2549467534), UINT32_C(2110436519) },
      { UINT32_C(3705531606), UINT32_C(1376509037), UINT32_C(2147240306), UINT32_C(3875374466), UINT32_C(2358657773), UINT32_C(3489210177), UINT32_C(2405887976), UINT32_C(2030842019) },
      UINT8_C(  4) },
    { { UINT32_C(2320886045), UINT32_C( 954032326), UINT32_C( 750246314), UINT32_C(2527361438), UINT32_C( 953657591), UINT32_C(2936524999), UINT32_C(1648258495), UINT32_C(3151710878) },
      { UINT32_C(4198838324), UINT32_C( 993141393), UINT32_C(1499982331), UINT32_C(2527361438), UINT32_C(3922623266), UINT32_C( 194631244), UINT32_C(3782137667), UINT32_C(1453082914) },
      UINT8_C(  8) },
    { { UINT32_C( 189850234), UINT32_C(   4620804), UINT32_C( 173649259), UINT32_C(1218510374), UINT32_C(3308416633), UINT32_C(2295450436), UINT32_C(3295231906), UINT32_C(  35260040) },
      { UINT32_C(3960302312), UINT32_C(1491882988), UINT32_C( 660751872), UINT32_C(3178168900), UINT32_C(3045302640), UINT32_C( 255677548), UINT32_C( 466855571), UINT32_C(2485055148) },
      UINT8_C(  0) },
    { { UINT32_C(1166093144), UINT32_C(2141023615), UINT32_C(4154916787), UINT32_C(1907693057), UINT32_C(1562036177), UINT32_C( 880561450), UINT32_C(3057255946), UINT32_C(1296783604) },
      { UINT32_C(  59952260), UINT32_C(3984731961), UINT32_C( 803481902), UINT32_C(4137720127), UINT32_C(1562036177), UINT32_C( 880561450), UINT32_C(1223407444), UINT32_C(2476029455) },
      UINT8_C( 48) },
    { { UINT32_C( 999696130), UINT32_C(2217220438), UINT32_C(2175995202), UINT32_C(2021086374), UINT32_C(1171624475), UINT32_C( 863654623), UINT32_C( 410805513), UINT32_C(2645299611) },
      { UINT32_C(2413314617), UINT32_C(2635333979), UINT32_C(3021915918), UINT32_C( 908891675), UINT32_C( 125567528), UINT32_C(1564210772), UINT32_C(4134909787), UINT32_C(  26419656) },
      UINT8_C(  0) },
    { { UINT32_C(3972632192), UINT32_C(2052957548), UINT32_C(2284813164), UINT32_C(1340925681), UINT32_C(2973842013), UINT32_C(2332982320), UINT32_C(4219569203), UINT32_C( 167580837) },
      { UINT32_C(3972632192), UINT32_C(2052957548), UINT32_C(3005716129), UINT32_C(1340925681), UINT32_C(1358966048), UINT32_C(2332982320), UINT32_C( 953704083), UINT32_C(4064466290) },
      UINT8_C( 43) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpeq_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__mmask8 r = easysimd_mm256_cmpeq_epu32_mask(a, b);

    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpeq_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint64_t a[4];
    const uint64_t b[4];
    const uint64_t r;
  } test_vec[8] = {
    { { UINT64_C(15162464688708473882), UINT64_C(10065132331437868157), UINT64_C(14163491473514403537), UINT64_C(11854765638759525712) },
      { UINT64_C(15162464688708473882), UINT64_C(10065132331437868157), UINT64_C(14163491473514403537), UINT64_C(11854765638759525712) },
      UINT8_C( 15) },
    { { UINT64_C( 5027447751836446390), UINT64_C(15344319204239198815), UINT64_C( 9730565778270007760), UINT64_C(16139569732655568701) },
      { UINT64_C(16343742390728525683), UINT64_C(15344319204239198815), UINT64_C( 9730565778270007760), UINT64_C(16139569732655568701) },
      UINT8_C( 14) },
    { { UINT64_C(16400807487358361756), UINT64_C(11601717350375849257), UINT64_C( 8016233180222002121), UINT64_C( 6442502330672424916) },
      { UINT64_C( 9197385452121945917), UINT64_C(13565597285926848042), UINT64_C( 8016233180222002121), UINT64_C( 6442502330672424916) },
      UINT8_C( 12) },
    { { UINT64_C( 9802815066455978036), UINT64_C( 3275923859306034910), UINT64_C( 3857741738813967230), UINT64_C(14787596421955214605) },
      { UINT64_C(11794191598027812296), UINT64_C(13811083485514683928), UINT64_C( 6989413254432415518), UINT64_C(14787596421955214605) },
      UINT8_C(  8) },
    { { UINT64_C(17884406413233454794), UINT64_C( 7471936223207114958), UINT64_C(18352531592318803455), UINT64_C( 9056594485338052292) },
      { UINT64_C( 6760816025539456762), UINT64_C( 7471936223207114958), UINT64_C(  774735988054948995), UINT64_C(10209677480145458816) },
      UINT8_C(  2) },
    { { UINT64_C(14552603722680679485), UINT64_C(13858607661294655363), UINT64_C(16166387230285448939), UINT64_C(11779687535442876248) },
      { UINT64_C(11483106049175314840), UINT64_C(12016974896900560327), UINT64_C(16166387230285448939), UINT64_C(17936787695767160290) },
      UINT8_C(  4) },
    { { UINT64_C( 2225314052862322470), UINT64_C( 4027315706916924733), UINT64_C(16645364069299223065), UINT64_C(14758888232757059961) },
      { UINT64_C( 9419260227868319041), UINT64_C(17140879130120673884), UINT64_C(16645364069299223065), UINT64_C( 9874584492662861207) },
      UINT8_C(  4) },
    { { UINT64_C(18057721772491631919), UINT64_C( 3630442364407674547), UINT64_C(15040700291304194662), UINT64_C(18323764861039704955) },
      { UINT64_C(18057721772491631919), UINT64_C(17846304697106736480), UINT64_C(13896848706290579666), UINT64_C( 3462931667461859586) },
      UINT8_C(  1) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpeq_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpeq_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__mmask8 r = easysimd_mm256_cmpeq_epu64_mask(a, b);

    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const uint8_t a[32];
    const uint8_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(3405426005),
      { UINT8_C(232), UINT8_C(116), UINT8_C(171), UINT8_C(223), UINT8_C(107), UINT8_C(  5), UINT8_C(210), UINT8_C(139),
        UINT8_C(187), UINT8_C(194), UINT8_C(136), UINT8_C(179), UINT8_C( 46), UINT8_C(248), UINT8_C( 34), UINT8_C(235),
        UINT8_C(158), UINT8_C( 59), UINT8_C( 41), UINT8_C( 19), UINT8_C( 64), UINT8_C(110), UINT8_C( 99), UINT8_C(231),
        UINT8_C( 84), UINT8_C(207), UINT8_C(222), UINT8_C(169), UINT8_C(124), UINT8_C(216), UINT8_C(115), UINT8_C(100) },
      { UINT8_C(232), UINT8_C(116), UINT8_C( 66), UINT8_C(223), UINT8_C(107), UINT8_C(  5), UINT8_C(146), UINT8_C(139),
        UINT8_C(187), UINT8_C(181), UINT8_C(242), UINT8_C( 98), UINT8_C(240), UINT8_C(248), UINT8_C( 34), UINT8_C(235),
        UINT8_C(137), UINT8_C(216), UINT8_C( 41), UINT8_C( 19), UINT8_C( 64), UINT8_C(110), UINT8_C(135), UINT8_C(231),
        UINT8_C(206), UINT8_C(250), UINT8_C(135), UINT8_C(169), UINT8_C( 25), UINT8_C(203), UINT8_C(115), UINT8_C( 60) },
      UINT32_C(1220059409) },
    { UINT32_C(3105559777),
      { UINT8_C(223), UINT8_C(174), UINT8_C(192), UINT8_C(163), UINT8_C( 99), UINT8_C(178), UINT8_C(  5), UINT8_C( 83),
        UINT8_C(205), UINT8_C(122), UINT8_C(132), UINT8_C( 86), UINT8_C( 82), UINT8_C(156), UINT8_C( 52), UINT8_C(250),
        UINT8_C(146), UINT8_C(187), UINT8_C( 29), UINT8_C( 96), UINT8_C(181), UINT8_C(165), UINT8_C(122), UINT8_C(206),
        UINT8_C(112), UINT8_C( 76), UINT8_C( 11), UINT8_C( 81), UINT8_C( 97), UINT8_C( 38), UINT8_C( 10), UINT8_C( 64) },
      { UINT8_C(223), UINT8_C(174), UINT8_C(139), UINT8_C(163), UINT8_C( 98), UINT8_C(178), UINT8_C(160), UINT8_C(181),
        UINT8_C(205), UINT8_C(212), UINT8_C(132), UINT8_C( 61), UINT8_C( 82), UINT8_C(156), UINT8_C(157), UINT8_C( 68),
        UINT8_C(113), UINT8_C( 23), UINT8_C( 29), UINT8_C( 96), UINT8_C(181), UINT8_C( 30), UINT8_C( 50), UINT8_C(197),
        UINT8_C( 68), UINT8_C( 61), UINT8_C(  5), UINT8_C( 81), UINT8_C(  7), UINT8_C(233), UINT8_C( 10), UINT8_C( 64) },
      UINT32_C(2283279393) },
    { UINT32_C( 885906385),
      { UINT8_C(234), UINT8_C(109), UINT8_C(233), UINT8_C(149), UINT8_C( 65), UINT8_C(152), UINT8_C(210), UINT8_C(208),
        UINT8_C(100), UINT8_C(111), UINT8_C( 21), UINT8_C(214), UINT8_C(135), UINT8_C( 40), UINT8_C(183), UINT8_C(235),
        UINT8_C( 70), UINT8_C(234), UINT8_C(176), UINT8_C(138), UINT8_C( 39), UINT8_C(181), UINT8_C(163), UINT8_C( 46),
        UINT8_C(158), UINT8_C(244), UINT8_C(178), UINT8_C(112), UINT8_C(207), UINT8_C(128), UINT8_C(164), UINT8_C(186) },
      { UINT8_C( 37), UINT8_C(109),    UINT8_MAX, UINT8_C(137), UINT8_C( 65), UINT8_C( 20), UINT8_C( 95), UINT8_C( 24),
        UINT8_C( 60), UINT8_C(111), UINT8_C(  3), UINT8_C(130), UINT8_C(135), UINT8_C( 40), UINT8_C(183), UINT8_C( 40),
        UINT8_C(105), UINT8_C(176), UINT8_C( 86), UINT8_C(  7), UINT8_C( 39), UINT8_C(181), UINT8_C(119), UINT8_C( 46),
        UINT8_C(137), UINT8_C( 27), UINT8_C( 46), UINT8_C(118), UINT8_C(207), UINT8_C(125), UINT8_C(164), UINT8_C(186) },
      UINT32_C( 276845072) },
    { UINT32_C( 827827615),
      { UINT8_C(185), UINT8_C(182), UINT8_C( 73), UINT8_C(246), UINT8_C(205), UINT8_C( 77), UINT8_C(120), UINT8_C(206),
        UINT8_C(  0), UINT8_C(133), UINT8_C(246), UINT8_C(105), UINT8_C( 54), UINT8_C( 77), UINT8_C(113), UINT8_C(218),
        UINT8_C( 86), UINT8_C(232), UINT8_C( 78), UINT8_C(223), UINT8_C(  4), UINT8_C(124), UINT8_C( 85), UINT8_C(172),
        UINT8_C(250), UINT8_C(251), UINT8_C(122), UINT8_C(153), UINT8_C(160), UINT8_C(209), UINT8_C(202), UINT8_C( 89) },
      { UINT8_C( 97), UINT8_C(200), UINT8_C( 35), UINT8_C(246), UINT8_C(205), UINT8_C( 77), UINT8_C(120), UINT8_C(131),
        UINT8_C(  0), UINT8_C(133), UINT8_C( 94), UINT8_C(105), UINT8_C( 36), UINT8_C( 77), UINT8_C(113), UINT8_C(218),
        UINT8_C( 41), UINT8_C(241), UINT8_C(213), UINT8_C( 35), UINT8_C(  4), UINT8_C(124), UINT8_C(188), UINT8_C(172),
        UINT8_C( 32), UINT8_C(251), UINT8_C(230), UINT8_C(153), UINT8_C(155), UINT8_C(209), UINT8_C(252), UINT8_C( 89) },
      UINT32_C( 537960728) },
    { UINT32_C(1264394493),
      { UINT8_C( 58), UINT8_C( 40), UINT8_C(206), UINT8_C(161), UINT8_C(100), UINT8_C( 44), UINT8_C( 94), UINT8_C(137),
        UINT8_C(217), UINT8_C(250), UINT8_C(177), UINT8_C(  2), UINT8_C(235), UINT8_C(134), UINT8_C( 37), UINT8_C(216),
        UINT8_C(213), UINT8_C(225), UINT8_C(100), UINT8_C(245), UINT8_C(104), UINT8_C( 74), UINT8_C(157), UINT8_C(  3),
        UINT8_C(128), UINT8_C(153),    UINT8_MAX, UINT8_C(125), UINT8_C(185), UINT8_C( 93), UINT8_C(200), UINT8_C(243) },
      { UINT8_C(195), UINT8_C( 40), UINT8_C(115), UINT8_C(161), UINT8_C(100), UINT8_C( 44), UINT8_C( 94), UINT8_C(216),
        UINT8_C(171), UINT8_C(195), UINT8_C(176), UINT8_C(  2), UINT8_C(165), UINT8_C(134), UINT8_C( 37), UINT8_C( 13),
        UINT8_C(213), UINT8_C(225), UINT8_C( 17), UINT8_C(245), UINT8_C(172), UINT8_C( 74), UINT8_C(157), UINT8_C(102),
        UINT8_C(128), UINT8_C( 37),    UINT8_MAX, UINT8_C(243), UINT8_C(185), UINT8_C(238), UINT8_C(221), UINT8_C(127) },
      UINT32_C(  21569656) },
    { UINT32_C(3441184992),
      { UINT8_C(116), UINT8_C(186), UINT8_C(165), UINT8_C( 31), UINT8_C(126), UINT8_C( 85), UINT8_C(160), UINT8_C( 35),
        UINT8_C(105), UINT8_C( 22), UINT8_C( 48), UINT8_C(200), UINT8_C( 41), UINT8_C( 65), UINT8_C(167), UINT8_C(213),
        UINT8_C( 82), UINT8_C(  4), UINT8_C( 59), UINT8_C(191), UINT8_C( 41), UINT8_C(149), UINT8_C(178), UINT8_C(229),
        UINT8_C(131), UINT8_C(143), UINT8_C(100), UINT8_C( 99), UINT8_C(223), UINT8_C(128), UINT8_C( 48), UINT8_C( 84) },
      { UINT8_C( 42), UINT8_C( 19), UINT8_C(165), UINT8_C(148), UINT8_C( 41), UINT8_C( 12), UINT8_C(160), UINT8_C( 35),
        UINT8_C( 78), UINT8_C( 22), UINT8_C( 40), UINT8_C(200), UINT8_C(  8), UINT8_C( 65), UINT8_C( 95), UINT8_C( 49),
        UINT8_C(248), UINT8_C( 18), UINT8_C( 59), UINT8_C(191), UINT8_C(161), UINT8_C(122), UINT8_C(223), UINT8_C(229),
        UINT8_C(251), UINT8_C(143), UINT8_C(100), UINT8_C( 54), UINT8_C(229), UINT8_C( 72), UINT8_C( 48), UINT8_C( 15) },
      UINT32_C(1141637312) },
    { UINT32_C(2242104156),
      { UINT8_C(215), UINT8_C(  0), UINT8_C(216), UINT8_C( 37), UINT8_C(  4), UINT8_C(  0), UINT8_C(197), UINT8_C( 12),
        UINT8_C( 99), UINT8_C( 37), UINT8_C( 61), UINT8_C( 92), UINT8_C( 55), UINT8_C( 83), UINT8_C(215), UINT8_C(216),
        UINT8_C(205), UINT8_C(182), UINT8_C( 89), UINT8_C(200), UINT8_C(198), UINT8_C( 46), UINT8_C(254), UINT8_C(171),
        UINT8_C(119), UINT8_C(237), UINT8_C(186), UINT8_C(211), UINT8_C(184), UINT8_C( 94), UINT8_C( 88), UINT8_C(144) },
      { UINT8_C(215), UINT8_C(123), UINT8_C(110), UINT8_C(148), UINT8_C(160), UINT8_C(  0), UINT8_C(240), UINT8_C( 12),
        UINT8_C( 99), UINT8_C( 37), UINT8_C( 61), UINT8_C( 92), UINT8_C(126), UINT8_C(  9), UINT8_C(215), UINT8_C(216),
        UINT8_C( 55), UINT8_C(182), UINT8_C(239), UINT8_C(200), UINT8_C(128), UINT8_C(169), UINT8_C(254), UINT8_C( 56),
        UINT8_C(119), UINT8_C(218), UINT8_C(186), UINT8_C(211), UINT8_C(184), UINT8_C(126), UINT8_C(199), UINT8_C(144) },
      UINT32_C(2231552768) },
    { UINT32_C(2580493817),
      { UINT8_C(224), UINT8_C(191), UINT8_C(112), UINT8_C(222), UINT8_C(134), UINT8_C( 31), UINT8_C(170), UINT8_C(  4),
        UINT8_C( 40), UINT8_C( 62), UINT8_C( 72), UINT8_C( 96), UINT8_C(208), UINT8_C( 55), UINT8_C( 14), UINT8_C( 80),
        UINT8_C(225), UINT8_C(144), UINT8_C(137), UINT8_C(232), UINT8_C(106), UINT8_C( 81), UINT8_C( 78), UINT8_C(116),
        UINT8_C(207), UINT8_C( 21), UINT8_C(175), UINT8_C(200), UINT8_C( 75), UINT8_C(126), UINT8_C( 97), UINT8_C( 43) },
      { UINT8_C(241), UINT8_C(191), UINT8_C(200), UINT8_C( 25), UINT8_C(242), UINT8_C( 17), UINT8_C(170), UINT8_C(  4),
        UINT8_C( 72), UINT8_C( 62), UINT8_C( 72), UINT8_C( 96), UINT8_C( 24), UINT8_C( 55), UINT8_C( 18), UINT8_C(130),
        UINT8_C(225), UINT8_C( 96), UINT8_C(137), UINT8_C(189), UINT8_C(106), UINT8_C( 81), UINT8_C( 78), UINT8_C(116),
        UINT8_C(207), UINT8_C( 21), UINT8_C(236), UINT8_C(200), UINT8_C( 75), UINT8_C(126), UINT8_C( 38), UINT8_C(169) },
      UINT32_C( 432350400) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask32 k = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epu8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epu8_mask");

   easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi8(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u8x32());
    easysimd__mmask32 r = easysimd_mm256_mask_cmpeq_epu8_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const uint16_t a[16];
    const uint16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C( 2490),
      { UINT16_C(20966), UINT16_C(65406), UINT16_C( 6241), UINT16_C(35105), UINT16_C(44112), UINT16_C( 3649), UINT16_C(21812), UINT16_C( 8536),
        UINT16_C(36859), UINT16_C(30620), UINT16_C(38548), UINT16_C(  969), UINT16_C(23160), UINT16_C(43734), UINT16_C(37190), UINT16_C(11443) },
      { UINT16_C(20966), UINT16_C(19786), UINT16_C( 6241), UINT16_C(35105), UINT16_C(44112), UINT16_C(  354), UINT16_C(24143), UINT16_C(60560),
        UINT16_C(36859), UINT16_C(40578), UINT16_C(38548), UINT16_C(  969), UINT16_C(16292), UINT16_C(22671), UINT16_C(37190), UINT16_C(11443) },
      UINT16_C( 2328) },
    { UINT16_C(54452),
      { UINT16_C(32995), UINT16_C(56687), UINT16_C( 6029), UINT16_C(61451), UINT16_C(23064), UINT16_C(43342), UINT16_C( 9030), UINT16_C(51405),
        UINT16_C(62914), UINT16_C(48067), UINT16_C(26612), UINT16_C(33786), UINT16_C(26047), UINT16_C(18933), UINT16_C(43515), UINT16_C(57118) },
      { UINT16_C(32995), UINT16_C(51108), UINT16_C( 6029), UINT16_C(62753), UINT16_C(23064), UINT16_C(13081), UINT16_C( 9030), UINT16_C(51405),
        UINT16_C( 7574), UINT16_C(48067), UINT16_C( 6816), UINT16_C(38389), UINT16_C(26047), UINT16_C(18933), UINT16_C(43515), UINT16_C(35599) },
      UINT16_C(20628) },
    { UINT16_C(45856),
      { UINT16_C(51282), UINT16_C(29808), UINT16_C(54973), UINT16_C(55004), UINT16_C( 3082), UINT16_C(13233), UINT16_C(18176), UINT16_C(23376),
        UINT16_C(61655), UINT16_C(52341), UINT16_C(55686), UINT16_C(50621), UINT16_C(35931), UINT16_C(27182), UINT16_C(19992), UINT16_C(27166) },
      { UINT16_C(51282), UINT16_C(47717), UINT16_C(28586), UINT16_C(55004), UINT16_C(51106), UINT16_C(13233), UINT16_C(18176), UINT16_C(23376),
        UINT16_C(61655), UINT16_C( 1137), UINT16_C(52269), UINT16_C(23441), UINT16_C(35931), UINT16_C(27182), UINT16_C(19992), UINT16_C(62179) },
      UINT16_C(12576) },
    { UINT16_C(18580),
      { UINT16_C(16300), UINT16_C(29623), UINT16_C(22939), UINT16_C(15930), UINT16_C(23627), UINT16_C(11961), UINT16_C(  500), UINT16_C(26006),
        UINT16_C(50181), UINT16_C(38449), UINT16_C(26655), UINT16_C(51519), UINT16_C(21437), UINT16_C(41354), UINT16_C( 7749), UINT16_C(61929) },
      { UINT16_C(63589), UINT16_C(29623), UINT16_C(17975), UINT16_C(61692), UINT16_C(61556), UINT16_C(11961), UINT16_C(63317), UINT16_C(26006),
        UINT16_C(61069), UINT16_C(38449), UINT16_C(26655), UINT16_C(51519), UINT16_C(21437), UINT16_C(14176), UINT16_C( 7749), UINT16_C(48088) },
      UINT16_C(18560) },
    { UINT16_C(53942),
      { UINT16_C(60763), UINT16_C(22296), UINT16_C(36061), UINT16_C(53063), UINT16_C(40087), UINT16_C(26054), UINT16_C(21282), UINT16_C( 4435),
        UINT16_C( 2848), UINT16_C(16574), UINT16_C( 2892), UINT16_C(44197), UINT16_C(64578), UINT16_C( 6762), UINT16_C( 8375), UINT16_C( 4845) },
      { UINT16_C(60009), UINT16_C(22296), UINT16_C(10681), UINT16_C(32588), UINT16_C(40087), UINT16_C(26054), UINT16_C(21282), UINT16_C( 4435),
        UINT16_C(14644), UINT16_C(16574), UINT16_C(36070), UINT16_C(44197), UINT16_C(64578), UINT16_C( 6762), UINT16_C( 8375), UINT16_C( 4845) },
      UINT16_C(53938) },
    { UINT16_C(11111),
      { UINT16_C( 8633), UINT16_C( 1620), UINT16_C(58272), UINT16_C(29557), UINT16_C(62917), UINT16_C(45671), UINT16_C(39732), UINT16_C(32491),
        UINT16_C(53620), UINT16_C(18954), UINT16_C(45345), UINT16_C(37335), UINT16_C(30277), UINT16_C(57102), UINT16_C(30335), UINT16_C(14602) },
      { UINT16_C(14143), UINT16_C(46146), UINT16_C( 1963), UINT16_C(29557), UINT16_C(56761), UINT16_C(45671), UINT16_C(39732), UINT16_C(26230),
        UINT16_C(38763), UINT16_C(16919), UINT16_C(23849), UINT16_C(14265), UINT16_C(14396), UINT16_C(57102), UINT16_C(17521), UINT16_C(14602) },
      UINT16_C( 8288) },
    { UINT16_C(59260),
      { UINT16_C(10084), UINT16_C( 3822), UINT16_C(42809), UINT16_C(59115), UINT16_C(18252), UINT16_C(49671), UINT16_C(29613), UINT16_C(50265),
        UINT16_C(33461), UINT16_C(28193), UINT16_C(23994), UINT16_C(26535), UINT16_C( 6308), UINT16_C(18860), UINT16_C(10441), UINT16_C(11569) },
      { UINT16_C(34875), UINT16_C(10183), UINT16_C( 4974), UINT16_C(30062), UINT16_C(18252), UINT16_C(49671), UINT16_C(40671), UINT16_C(50265),
        UINT16_C(27404), UINT16_C(45918), UINT16_C(  722), UINT16_C(32460), UINT16_C(38220), UINT16_C(18860), UINT16_C(10441), UINT16_C(11569) },
      UINT16_C(57392) },
    { UINT16_C(25469),
      { UINT16_C(60197), UINT16_C(37750), UINT16_C(19297), UINT16_C(18862), UINT16_C(36218), UINT16_C(11239), UINT16_C(62606), UINT16_C(60822),
        UINT16_C(26791), UINT16_C(29679), UINT16_C(15335), UINT16_C(36104), UINT16_C(52152), UINT16_C(21891), UINT16_C(  201), UINT16_C(61112) },
      { UINT16_C(60197), UINT16_C(37750), UINT16_C(62614), UINT16_C(32444), UINT16_C(36218), UINT16_C(46450), UINT16_C( 6456), UINT16_C(10014),
        UINT16_C( 1421), UINT16_C(38243), UINT16_C( 7058), UINT16_C( 5472), UINT16_C(52152), UINT16_C(10518), UINT16_C(  201), UINT16_C(61112) },
      UINT16_C(16401) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epu16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epu16_mask");

   easysimd_assert_equal_mmask16(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi16(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u16x16());
    easysimd__mmask16 r = easysimd_mm256_mask_cmpeq_epu16_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint32_t a[8];
    const uint32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(161),
      { UINT32_C( 907940316), UINT32_C( 135064876), UINT32_C( 416575092), UINT32_C(3151648986), UINT32_C(1069647802), UINT32_C(3527760995), UINT32_C(4116948553), UINT32_C(2375484081) },
      { UINT32_C( 390647715), UINT32_C( 405740606), UINT32_C( 416575092), UINT32_C(3151648986), UINT32_C( 969323248), UINT32_C(3527760995), UINT32_C(4116948553), UINT32_C(2375484081) },
      UINT8_C(160) },
    { UINT8_C( 42),
      { UINT32_C(3815257543), UINT32_C(1868005477), UINT32_C(3892348244), UINT32_C(2765747442), UINT32_C(3824947898), UINT32_C( 118359361), UINT32_C(4152577436), UINT32_C(2468454860) },
      { UINT32_C(1571540233), UINT32_C(1145482066), UINT32_C(1508384671), UINT32_C(2765747442), UINT32_C(2256161514), UINT32_C( 118359361), UINT32_C(1996005023), UINT32_C( 833777191) },
      UINT8_C( 40) },
    { UINT8_C( 60),
      { UINT32_C( 143560285), UINT32_C(4087862228), UINT32_C(2686779579), UINT32_C(2257229884), UINT32_C( 270405650), UINT32_C( 749699982), UINT32_C(3914605947), UINT32_C( 874874327) },
      { UINT32_C(1121706887), UINT32_C( 534905059), UINT32_C(2686779579), UINT32_C(2257229884), UINT32_C(3258411591), UINT32_C( 749699982), UINT32_C(3914605947), UINT32_C( 874874327) },
      UINT8_C( 44) },
    { UINT8_C(185),
      { UINT32_C(3902557927), UINT32_C(2642459441), UINT32_C( 605694816), UINT32_C(1349199334), UINT32_C(3906940253), UINT32_C(2868463065), UINT32_C(3039763935), UINT32_C( 275659561) },
      { UINT32_C( 643332549), UINT32_C(2642459441), UINT32_C( 605694816), UINT32_C(3168648163), UINT32_C(3906940253), UINT32_C(2868463065), UINT32_C(3039763935), UINT32_C(3558089998) },
      UINT8_C( 48) },
    { UINT8_C(158),
      { UINT32_C(3745774188), UINT32_C(4202155588), UINT32_C(1692298409), UINT32_C( 543529545), UINT32_C(2517845249), UINT32_C(3731304666), UINT32_C(2196619613), UINT32_C(1109508565) },
      { UINT32_C(3690633266), UINT32_C(1296095236), UINT32_C(1692298409), UINT32_C(3289022954), UINT32_C( 245592497), UINT32_C( 126914610), UINT32_C(2196619613), UINT32_C(1191930389) },
      UINT8_C(  4) },
    { UINT8_C(  2),
      { UINT32_C(3708166917), UINT32_C( 139482211), UINT32_C(1123206081), UINT32_C(1056159437), UINT32_C(3916431705), UINT32_C(1127905426), UINT32_C( 744048321), UINT32_C(1462673490) },
      { UINT32_C(1244562825), UINT32_C( 378347592), UINT32_C(1123206081), UINT32_C(1056159437), UINT32_C(3916431705), UINT32_C(1127905426), UINT32_C( 340023633), UINT32_C( 389773710) },
      UINT8_C(  0) },
    { UINT8_C(254),
      { UINT32_C(2319933801), UINT32_C(1868652014), UINT32_C(1995542961), UINT32_C(  11732395), UINT32_C( 228307548), UINT32_C( 895387868), UINT32_C(1187214028), UINT32_C( 390388397) },
      { UINT32_C(2319933801), UINT32_C(1868652014), UINT32_C(3963602063), UINT32_C(1392106102), UINT32_C( 780687202), UINT32_C(2004110281), UINT32_C(1187214028), UINT32_C( 390388397) },
      UINT8_C(194) },
    { UINT8_C( 49),
      { UINT32_C(2965686054), UINT32_C(2604663767), UINT32_C( 470887490), UINT32_C(2088657957), UINT32_C( 927313388), UINT32_C(3663576097), UINT32_C(2032123722), UINT32_C(1923763276) },
      { UINT32_C(3045483122), UINT32_C(3016857486), UINT32_C( 470887490), UINT32_C(2088657957), UINT32_C(2096725041), UINT32_C(3663576097), UINT32_C(2032123722), UINT32_C(1923763276) },
      UINT8_C( 32) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epu32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epu32_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi32(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u32x8());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpeq_epu32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpeq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint64_t a[4];
    const uint64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 73),
      { UINT64_C( 4787380267728237851), UINT64_C(12712417408624511657), UINT64_C(16685013712582971567), UINT64_C(16822772882079407712) },
      { UINT64_C( 4787380267728237851), UINT64_C(12712417408624511657), UINT64_C( 7979219632205298692), UINT64_C(16822772882079407712) },
      UINT8_C(  9) },
    { UINT8_C(212),
      { UINT64_C(12719063545257010658), UINT64_C( 4391796616136077000), UINT64_C( 5266860254501979190), UINT64_C(10226732103321675140) },
      { UINT64_C(17412164910023831720), UINT64_C(  358915232909286195), UINT64_C( 5266860254501979190), UINT64_C(15489026108429448094) },
      UINT8_C(  4) },
    { UINT8_C(  3),
      { UINT64_C(13696484285026811761), UINT64_C( 5574123459325117879), UINT64_C( 8792573787871742559), UINT64_C(14935764157547964976) },
      { UINT64_C( 4653799380845960839), UINT64_C( 5574123459325117879), UINT64_C(14719256781130338327), UINT64_C( 4499321114185299026) },
      UINT8_C(  2) },
    { UINT8_C(106),
      { UINT64_C( 3564189104504276657), UINT64_C( 6698696307315762226), UINT64_C(11904481323381320596), UINT64_C( 6690830257527471011) },
      { UINT64_C( 3564189104504276657), UINT64_C( 6698696307315762226), UINT64_C(11904481323381320596), UINT64_C( 6690830257527471011) },
      UINT8_C( 10) },
    { UINT8_C(108),
      { UINT64_C( 1520103953653256250), UINT64_C( 4451901033582685621), UINT64_C( 2509699925138020918), UINT64_C( 1851011192103659275) },
      { UINT64_C( 1520103953653256250), UINT64_C( 4451901033582685621), UINT64_C( 2419173746363439258), UINT64_C(15484513259310784380) },
      UINT8_C(  0) },
    { UINT8_C(236),
      { UINT64_C( 7574085925154737939), UINT64_C(10068563266999416557), UINT64_C( 7333655443689911209), UINT64_C(10619769943742847851) },
      { UINT64_C( 3039607048844134833), UINT64_C( 4195167712880913933), UINT64_C(12819515449057173973), UINT64_C(17413383741203973781) },
      UINT8_C(  0) },
    { UINT8_C(141),
      { UINT64_C( 3580705591641675209), UINT64_C( 2142370630731859116), UINT64_C( 9119561616361197801), UINT64_C(12215926994591758134) },
      { UINT64_C( 6799057148994075265), UINT64_C( 2142370630731859116), UINT64_C( 6316510760427741471), UINT64_C(12215926994591758134) },
      UINT8_C(  8) },
    { UINT8_C(213),
      { UINT64_C(16591444697571504635), UINT64_C( 7423955316979557376), UINT64_C(11853984045471369066), UINT64_C( 4296164942439490607) },
      { UINT64_C(16591444697571504635), UINT64_C( 7423955316979557376), UINT64_C( 4109608863280689548), UINT64_C(16162207313762081067) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpeq_epu64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpeq_epu64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi64(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_u64x4());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpeq_epu64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpeq_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i   a;
    easysimd__m512i   b;
    easysimd__mmask64 r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C(  73), INT8_C(  68), INT8_C( -71), INT8_C( -32),
                           INT8_C( 100), INT8_C( 125), INT8_C(  89), INT8_C(  95),
                           INT8_C( -23), INT8_C(  76), INT8_C(  84), INT8_C( -43),
                           INT8_C(  86), INT8_C(  29), INT8_C(  64), INT8_C(  30),
                           INT8_C( -17), INT8_C( -61), INT8_C( 115), INT8_C( -53),
                           INT8_C(-100), INT8_C( 104), INT8_C( 111), INT8_C( -59),
                           INT8_C( -31), INT8_C(  20), INT8_C(  31), INT8_C(-121),
                           INT8_C(  90), INT8_C(  18), INT8_C(   6), INT8_C(  39),
                           INT8_C(  15), INT8_C(  62), INT8_C(  39), INT8_C( -25),
                           INT8_C(  60), INT8_C( 110), INT8_C(  45), INT8_C( 113),
                           INT8_C(  81), INT8_C(  78), INT8_C( -53), INT8_C(  11),
                           INT8_C( -27), INT8_C(-113), INT8_C(  -3), INT8_C(  14),
                           INT8_C( 109), INT8_C(  43), INT8_C( -54), INT8_C( 111),
                           INT8_C( -91), INT8_C( -21), INT8_C( 102), INT8_C(   8),
                           INT8_C( -41), INT8_C( -47), INT8_C( -90), INT8_C(   3),
                           INT8_C(  18), INT8_C(  32), INT8_C(  89), INT8_C( -62)),
      easysimd_mm512_set_epi8(INT8_C(  73), INT8_C(  68), INT8_C( -71), INT8_C( -46),
                           INT8_C(  72), INT8_C(  42), INT8_C(  65), INT8_C( -27),
                           INT8_C( -37), INT8_C(  76), INT8_C(  84), INT8_C( -95),
                           INT8_C( -29), INT8_C(  29), INT8_C(  64), INT8_C(  75),
                           INT8_C( -17), INT8_C( -61), INT8_C(  76), INT8_C(  88),
                           INT8_C(-100), INT8_C( -91), INT8_C( 111), INT8_C( -59),
                           INT8_C( -31), INT8_C( -17), INT8_C(  31), INT8_C(  68),
                           INT8_C(  90), INT8_C(  18), INT8_C(-110), INT8_C(  39),
                           INT8_C(  15), INT8_C(  62), INT8_C(  24), INT8_C( -25),
                           INT8_C(  80), INT8_C( 122), INT8_C( 103), INT8_C(-115),
                           INT8_C(  81), INT8_C(  24), INT8_C(  32), INT8_C(  82),
                           INT8_C( -27), INT8_C( 105), INT8_C(  -3), INT8_C(  14),
                           INT8_C( 109), INT8_C(  43), INT8_C( -54), INT8_C(-121),
                           INT8_C( -91), INT8_C(-124), INT8_C( 102), INT8_C(  46),
                           INT8_C( -41), INT8_C( 101), INT8_C(  51), INT8_C(   3),
                           INT8_C(  18), INT8_C(  32), INT8_C( -12), INT8_C( -62)),
      UINT64_C(0xe066cbadd08bea9d) },
    { easysimd_mm512_set_epi8(INT8_C( -84), INT8_C(-108), INT8_C(  13), INT8_C( -97),
                           INT8_C( -34), INT8_C(  27), INT8_C( 124), INT8_C(-120),
                           INT8_C(   3), INT8_C(  26), INT8_C(  43), INT8_C( -96),
                           INT8_C( -63), INT8_C(  49), INT8_C( 127), INT8_C(   0),
                           INT8_C( -20), INT8_C( -31), INT8_C( 125), INT8_C(  14),
                           INT8_C( -53), INT8_C( -87), INT8_C( 115), INT8_C( -20),
                           INT8_C( -93), INT8_C(  70), INT8_C(  29), INT8_C( -90),
                           INT8_C( 105), INT8_C( -54), INT8_C(  96), INT8_C(-106),
                           INT8_C(  94), INT8_C( -52), INT8_C( -50), INT8_C( -79),
                           INT8_C( -54), INT8_C(  78), INT8_C(  17), INT8_C(  81),
                           INT8_C(  35), INT8_C( 120), INT8_C(  47), INT8_C( -25),
                           INT8_C( 110), INT8_C(  55), INT8_C(  40), INT8_C( -31),
                           INT8_C(  93), INT8_C( -23), INT8_C(   4), INT8_C(  45),
                           INT8_C(  59), INT8_C( -33), INT8_C( 124), INT8_C( -52),
                           INT8_C(  42), INT8_C( -14), INT8_C(-119), INT8_C(  88),
                           INT8_C( -38), INT8_C( -31), INT8_C(-113), INT8_C(  33)),
      easysimd_mm512_set_epi8(INT8_C(  25), INT8_C(-108), INT8_C(  13), INT8_C( -97),
                           INT8_C( -34), INT8_C( 119), INT8_C( 122), INT8_C(  82),
                           INT8_C(   3), INT8_C(-111), INT8_C(  60), INT8_C( -96),
                           INT8_C(  26), INT8_C(   3), INT8_C(  -7), INT8_C(  -8),
                           INT8_C(  94), INT8_C( -31), INT8_C( -71), INT8_C(  14),
                           INT8_C(-105), INT8_C( -87), INT8_C( 115), INT8_C( -68),
                           INT8_C( -93), INT8_C(  70), INT8_C( -47), INT8_C(-106),
                           INT8_C( 105), INT8_C( -54), INT8_C(  96), INT8_C( 105),
                           INT8_C(  94), INT8_C(  84), INT8_C( -50), INT8_C( -79),
                           INT8_C( -54), INT8_C(  78), INT8_C(  17), INT8_C(  39),
                           INT8_C(  35), INT8_C( -87), INT8_C( -83), INT8_C( -25),
                           INT8_C( 110), INT8_C(   2), INT8_C( -90), INT8_C( -31),
                           INT8_C(  19), INT8_C( -23), INT8_C(   4), INT8_C(  16),
                           INT8_C(  59), INT8_C( -33), INT8_C( 124), INT8_C( 127),
                           INT8_C( -60), INT8_C( -14), INT8_C(-119), INT8_C(  88),
                           INT8_C( -38), INT8_C( 109), INT8_C(-113), INT8_C(  25)),
      UINT64_C(0x789056cebe996e7a) },
    { easysimd_mm512_set_epi8(INT8_C(  93), INT8_C( 110), INT8_C( 120), INT8_C(  25),
                           INT8_C( -37), INT8_C( -25), INT8_C( -34), INT8_C(-108),
                           INT8_C( -77), INT8_C(-114), INT8_C(  79), INT8_C( -50),
                           INT8_C( -94), INT8_C(  22), INT8_C( -28), INT8_C(-105),
                           INT8_C( 110), INT8_C(  29), INT8_C(  -9), INT8_C( -13),
                           INT8_C( -71), INT8_C( 107), INT8_C(-115), INT8_C(  86),
                           INT8_C(-127), INT8_C(-100), INT8_C(   1), INT8_C(  21),
                           INT8_C( -55), INT8_C( -85), INT8_C( -55), INT8_C( -81),
                           INT8_C( -41), INT8_C(  39), INT8_C(  18), INT8_C( -92),
                           INT8_C(  11), INT8_C( -32), INT8_C( -53), INT8_C(  38),
                           INT8_C( -49), INT8_C(-118), INT8_C(  20), INT8_C(  66),
                           INT8_C(-106), INT8_C(-109), INT8_C(  45), INT8_C( -24),
                           INT8_C( -47), INT8_C(  95), INT8_C(  50), INT8_C( 105),
                           INT8_C(  58), INT8_C(  25), INT8_C( -53), INT8_C( -61),
                           INT8_C( -90), INT8_C(  92), INT8_C(  83), INT8_C( 120),
                           INT8_C( 107), INT8_C( -72), INT8_C(   3), INT8_C(  -1)),
      easysimd_mm512_set_epi8(INT8_C(  93), INT8_C( 110), INT8_C( 120), INT8_C( -75),
                           INT8_C( -37), INT8_C( -25), INT8_C(  14), INT8_C(-108),
                           INT8_C(   5), INT8_C(-114), INT8_C(  79), INT8_C(  57),
                           INT8_C( -94), INT8_C(  22), INT8_C(   4), INT8_C(   3),
                           INT8_C(  66), INT8_C(  29), INT8_C(  -9), INT8_C(  20),
                           INT8_C(  93), INT8_C( 107), INT8_C(   6), INT8_C(  86),
                           INT8_C(  84), INT8_C(-100), INT8_C(  81), INT8_C(  21),
                           INT8_C( 117), INT8_C(  22), INT8_C( -55), INT8_C( -81),
                           INT8_C( -41), INT8_C( -17), INT8_C(  19), INT8_C( 106),
                           INT8_C( 114), INT8_C( -32), INT8_C(  98), INT8_C( -16),
                           INT8_C( -49), INT8_C(-118), INT8_C(  20), INT8_C(  66),
                           INT8_C(-106), INT8_C(-109), INT8_C(  97), INT8_C(  29),
                           INT8_C( -47), INT8_C( -47), INT8_C(  50), INT8_C(-111),
                           INT8_C(  58), INT8_C( 115), INT8_C( -53), INT8_C(  93),
                           INT8_C( -90), INT8_C( -27), INT8_C(  41), INT8_C( 120),
                           INT8_C( 119), INT8_C(  86), INT8_C( -36), INT8_C(  -1)),
      UINT64_C(0xed6c655384fcaa91) },
    { easysimd_mm512_set_epi8(INT8_C(  10), INT8_C(  75), INT8_C(  91), INT8_C( -99),
                           INT8_C( -88), INT8_C(  99), INT8_C( -86), INT8_C(  96),
                           INT8_C(  14), INT8_C(  -1), INT8_C(  14), INT8_C( 100),
                           INT8_C(-114), INT8_C(  63), INT8_C(  68), INT8_C(-113),
                           INT8_C( -59), INT8_C( -42), INT8_C( -14), INT8_C(-111),
                           INT8_C(   6), INT8_C(  68), INT8_C(  11), INT8_C(-108),
                           INT8_C( -62), INT8_C(  87), INT8_C( -72), INT8_C( -23),
                           INT8_C(  78), INT8_C( -18), INT8_C( -36), INT8_C(  -6),
                           INT8_C( -68), INT8_C(-115), INT8_C( -24), INT8_C( 127),
                           INT8_C( -36), INT8_C(  21), INT8_C(  38), INT8_C(-106),
                           INT8_C(  33), INT8_C( -66), INT8_C(-121), INT8_C(  36),
                           INT8_C(  24), INT8_C(  61), INT8_C(  66), INT8_C(  20),
                           INT8_C(  63), INT8_C( -18), INT8_C(  11), INT8_C(-103),
                           INT8_C( -19), INT8_C( -42), INT8_C( -69), INT8_C(  53),
                           INT8_C( -40), INT8_C( 112), INT8_C(   8), INT8_C( -69),
                           INT8_C(-102), INT8_C(  62), INT8_C(  85), INT8_C(  62)),
      easysimd_mm512_set_epi8(INT8_C(  10), INT8_C(  75), INT8_C( -74), INT8_C( -47),
                           INT8_C( -88), INT8_C(  99), INT8_C( -86), INT8_C(-128),
                           INT8_C(  94), INT8_C(  -1), INT8_C(  99), INT8_C( 100),
                           INT8_C( -25), INT8_C(   7), INT8_C(  59), INT8_C(-113),
                           INT8_C( 119), INT8_C( -42), INT8_C( -14), INT8_C(  79),
                           INT8_C(   4), INT8_C(-111), INT8_C(  11), INT8_C(  80),
                           INT8_C( -78), INT8_C(  87), INT8_C( -72), INT8_C(-111),
                           INT8_C( -95), INT8_C( -18), INT8_C( -36), INT8_C( -40),
                           INT8_C( -68), INT8_C(-115), INT8_C( -24), INT8_C( -50),
                           INT8_C( -36), INT8_C(  10), INT8_C(  47), INT8_C(  62),
                           INT8_C( -15), INT8_C( -66), INT8_C(-122), INT8_C(  36),
                           INT8_C( -22), INT8_C(  61), INT8_C( -11), INT8_C(  20),
                           INT8_C(  63), INT8_C(  82), INT8_C(-113), INT8_C(-103),
                           INT8_C( -19), INT8_C( -42), INT8_C( -69), INT8_C(  53),
                           INT8_C( -40), INT8_C( 112), INT8_C(   8), INT8_C( -69),
                           INT8_C(-102), INT8_C(  23), INT8_C(  85), INT8_C(  62)),
      UINT64_C(0xce516266e8559ffb) },
    { easysimd_mm512_set_epi8(INT8_C(  -7), INT8_C(   2), INT8_C(-111), INT8_C(  64),
                           INT8_C(-100), INT8_C(  87), INT8_C( 100), INT8_C( -30),
                           INT8_C( -39), INT8_C( -38), INT8_C( 121), INT8_C(  55),
                           INT8_C( -64), INT8_C(  81), INT8_C(  -3), INT8_C(  79),
                           INT8_C( -41), INT8_C( 118), INT8_C( -37), INT8_C( -34),
                           INT8_C( -13), INT8_C(  63), INT8_C(  26), INT8_C( -81),
                           INT8_C(  90), INT8_C(  43), INT8_C( -31), INT8_C( -17),
                           INT8_C(-100), INT8_C( -71), INT8_C(-104), INT8_C( -66),
                           INT8_C( -94), INT8_C( -89), INT8_C( 100), INT8_C(  36),
                           INT8_C(  17), INT8_C( 116), INT8_C( -30), INT8_C(  16),
                           INT8_C( 110), INT8_C(  98), INT8_C(  11), INT8_C( -42),
                           INT8_C( -78), INT8_C( -68), INT8_C( -26), INT8_C( -35),
                           INT8_C(  12), INT8_C( -40), INT8_C( -27), INT8_C( -40),
                           INT8_C(-102), INT8_C(-109), INT8_C(  39), INT8_C(  29),
                           INT8_C(  21), INT8_C(   9), INT8_C(  49), INT8_C( -13),
                           INT8_C( -49), INT8_C(   7), INT8_C(  91), INT8_C(  15)),
      easysimd_mm512_set_epi8(INT8_C(  78), INT8_C(   2), INT8_C( -91), INT8_C(  64),
                           INT8_C(-100), INT8_C(  41), INT8_C( -34), INT8_C( -46),
                           INT8_C( -39), INT8_C(  31), INT8_C(  13), INT8_C(  55),
                           INT8_C( -42), INT8_C(  33), INT8_C(  -3), INT8_C(  79),
                           INT8_C( -41), INT8_C( 118), INT8_C( -37), INT8_C(  90),
                           INT8_C( -13), INT8_C(  63), INT8_C(  51), INT8_C( -81),
                           INT8_C(  90), INT8_C(  43), INT8_C( -31), INT8_C(-112),
                           INT8_C(-100), INT8_C(  41), INT8_C(-104), INT8_C( -66),
                           INT8_C( -94), INT8_C( -89), INT8_C( -85), INT8_C(-109),
                           INT8_C( 113), INT8_C( 116), INT8_C( 100), INT8_C(  16),
                           INT8_C(   5), INT8_C( -50), INT8_C( -51), INT8_C( -42),
                           INT8_C( -95), INT8_C( -68), INT8_C( -26), INT8_C( -35),
                           INT8_C( -73), INT8_C(  71), INT8_C(  65), INT8_C( -40),
                           INT8_C(-102), INT8_C(   7), INT8_C(  94), INT8_C(  29),
                           INT8_C(  65), INT8_C(   9), INT8_C(  49), INT8_C( -13),
                           INT8_C( -33), INT8_C(   7), INT8_C(-101), INT8_C(  15)),
      UINT64_C(0x5893edebc5171975) },
    { easysimd_mm512_set_epi8(INT8_C( -34), INT8_C( -12), INT8_C( 105), INT8_C(-124),
                           INT8_C( -33), INT8_C( -79), INT8_C(  -6), INT8_C(  54),
                           INT8_C(  81), INT8_C( -11), INT8_C(  67), INT8_C(  63),
                           INT8_C( 103), INT8_C( 119), INT8_C( -89), INT8_C(  40),
                           INT8_C(   8), INT8_C( -38), INT8_C(  71), INT8_C(  66),
                           INT8_C(-106), INT8_C( -45), INT8_C(  18), INT8_C( 100),
                           INT8_C( 122), INT8_C(  93), INT8_C( -42), INT8_C(   5),
                           INT8_C( -39), INT8_C(  37), INT8_C( -70), INT8_C(  13),
                           INT8_C(  99), INT8_C( -57), INT8_C( -88), INT8_C( -36),
                           INT8_C(-103), INT8_C(  25), INT8_C(  94), INT8_C(-107),
                           INT8_C( -32), INT8_C( -12), INT8_C( -14), INT8_C(  32),
                           INT8_C( -38), INT8_C(  10), INT8_C(  89), INT8_C( -69),
                           INT8_C(  -8), INT8_C(  69), INT8_C( -20), INT8_C(-122),
                           INT8_C( -75), INT8_C( -71), INT8_C(   3), INT8_C( 102),
                           INT8_C( 119), INT8_C( -58), INT8_C( -49), INT8_C(  80),
                           INT8_C( -15), INT8_C( -97), INT8_C(  45), INT8_C(  96)),
      easysimd_mm512_set_epi8(INT8_C( -34), INT8_C( -12), INT8_C(  81), INT8_C(-115),
                           INT8_C( -33), INT8_C( -79), INT8_C(-117), INT8_C( -34),
                           INT8_C(  81), INT8_C( -11), INT8_C( -63), INT8_C( -61),
                           INT8_C(  53), INT8_C( 119), INT8_C(  26), INT8_C(  40),
                           INT8_C(   8), INT8_C( -38), INT8_C(  25), INT8_C( -23),
                           INT8_C( -16), INT8_C( -45), INT8_C( -64), INT8_C( 100),
                           INT8_C(  91), INT8_C(  93), INT8_C( -42), INT8_C(   5),
                           INT8_C(  81), INT8_C( -76), INT8_C( -70), INT8_C(  13),
                           INT8_C(  26), INT8_C( -57), INT8_C( -88), INT8_C( -64),
                           INT8_C( -68), INT8_C( -91), INT8_C(-123), INT8_C(  38),
                           INT8_C( -32), INT8_C(  29), INT8_C(  82), INT8_C(  54),
                           INT8_C(-107), INT8_C(  10), INT8_C(  89), INT8_C(  28),
                           INT8_C( -27), INT8_C(  41), INT8_C( -20), INT8_C(-122),
                           INT8_C( -75), INT8_C( -71), INT8_C(   3), INT8_C( -30),
                           INT8_C(  97), INT8_C(  18), INT8_C( -90), INT8_C( 107),
                           INT8_C(  99), INT8_C(  10), INT8_C(  45), INT8_C(  96)),
      UINT64_C(0xccc5c57360863e03) },
    { easysimd_mm512_set_epi8(INT8_C(  48), INT8_C(  94), INT8_C( 112), INT8_C(-107),
                           INT8_C( -34), INT8_C( -86), INT8_C(  65), INT8_C(  92),
                           INT8_C(  97), INT8_C( -99), INT8_C(  28), INT8_C(  47),
                           INT8_C(-117), INT8_C( -22), INT8_C(-111), INT8_C( -67),
                           INT8_C( 113), INT8_C(-107), INT8_C( -23), INT8_C(  77),
                           INT8_C(  60), INT8_C( 104), INT8_C(-116), INT8_C( -86),
                           INT8_C(-113), INT8_C( -79), INT8_C( -64), INT8_C( -15),
                           INT8_C(-123), INT8_C(  99), INT8_C(  25), INT8_C(  27),
                           INT8_C( -40), INT8_C( 126), INT8_C( -66), INT8_C( -45),
                           INT8_C(  57), INT8_C( -30), INT8_C( -12), INT8_C(  16),
                           INT8_C( 122), INT8_C( 124), INT8_C( -75), INT8_C(  50),
                           INT8_C(  -6), INT8_C(  41), INT8_C( -47), INT8_C(  -3),
                           INT8_C(  29), INT8_C( -20), INT8_C( -45), INT8_C( -46),
                           INT8_C( -45), INT8_C( -14), INT8_C(  99), INT8_C(  84),
                           INT8_C( -62), INT8_C( -99), INT8_C(-104), INT8_C( -78),
                           INT8_C( 106), INT8_C(-117), INT8_C( -94), INT8_C(   3)),
      easysimd_mm512_set_epi8(INT8_C( -88), INT8_C( -74), INT8_C(   2), INT8_C( -83),
                           INT8_C(-112), INT8_C( -91), INT8_C(  65), INT8_C(  92),
                           INT8_C(  97), INT8_C(  52), INT8_C(  28), INT8_C( -65),
                           INT8_C(-117), INT8_C( -90), INT8_C(-111), INT8_C( -67),
                           INT8_C( 113), INT8_C(  16), INT8_C(-124), INT8_C(  77),
                           INT8_C(  60), INT8_C( -64), INT8_C(-116), INT8_C( -89),
                           INT8_C(  -7), INT8_C( -79), INT8_C(  46), INT8_C( 114),
                           INT8_C(-107), INT8_C(  99), INT8_C( -79), INT8_C(  80),
                           INT8_C( -40), INT8_C( -81), INT8_C( -66), INT8_C( -45),
                           INT8_C(  57), INT8_C( -30), INT8_C(  66), INT8_C(  71),
                           INT8_C( 122), INT8_C(  95), INT8_C( -43), INT8_C(  50),
                           INT8_C(  -6), INT8_C(  41), INT8_C( -47), INT8_C(  95),
                           INT8_C(   5), INT8_C( -20), INT8_C( -45), INT8_C(-118),
                           INT8_C(  70), INT8_C(  81), INT8_C(   0), INT8_C(  84),
                           INT8_C( -62), INT8_C(-101), INT8_C(  13), INT8_C(  33),
                           INT8_C(-104), INT8_C(-117), INT8_C( -11), INT8_C(  20)),
      UINT64_C(0x3ab9a44bc9e6184) },
    { easysimd_mm512_set_epi8(INT8_C(  67), INT8_C(  34), INT8_C( -33), INT8_C(  31),
                           INT8_C(-128), INT8_C(  55), INT8_C(  93), INT8_C(  58),
                           INT8_C(  57), INT8_C( 104), INT8_C(-110), INT8_C(  59),
                           INT8_C(  55), INT8_C(  33), INT8_C(-122), INT8_C(  69),
                           INT8_C(  57), INT8_C(  30), INT8_C( -13), INT8_C( -65),
                           INT8_C( -22), INT8_C(-100), INT8_C(  18), INT8_C( -65),
                           INT8_C( -60), INT8_C(-105), INT8_C(  27), INT8_C( -71),
                           INT8_C(  52), INT8_C(  12), INT8_C(  -4), INT8_C(  64),
                           INT8_C(  20), INT8_C(  51), INT8_C(  87), INT8_C(  43),
                           INT8_C(  26), INT8_C(   6), INT8_C( -66), INT8_C( -40),
                           INT8_C(  87), INT8_C(   1), INT8_C( -26), INT8_C(  92),
                           INT8_C( -33), INT8_C(   8), INT8_C(  42), INT8_C( -93),
                           INT8_C(  44), INT8_C( -55), INT8_C(-113), INT8_C( -43),
                           INT8_C(  32), INT8_C( 105), INT8_C( -27), INT8_C(  96),
                           INT8_C(  72), INT8_C(  48), INT8_C( -46), INT8_C(  24),
                           INT8_C( -10), INT8_C( -98), INT8_C( -56), INT8_C( -41)),
      easysimd_mm512_set_epi8(INT8_C(  67), INT8_C( -63), INT8_C( -33), INT8_C(  31),
                           INT8_C(-128), INT8_C(  55), INT8_C(  93), INT8_C(  -8),
                           INT8_C(  82), INT8_C( 104), INT8_C(-110), INT8_C(  59),
                           INT8_C(  55), INT8_C(   0), INT8_C( -25), INT8_C(  69),
                           INT8_C(  27), INT8_C(  30), INT8_C( -13), INT8_C( -65),
                           INT8_C(  -7), INT8_C( -28), INT8_C(  18), INT8_C( -65),
                           INT8_C(  67), INT8_C(  -3), INT8_C(  57), INT8_C( -68),
                           INT8_C(  52), INT8_C(  12), INT8_C(  -4), INT8_C(-128),
                           INT8_C(  20), INT8_C(  37), INT8_C(   9), INT8_C(  80),
                           INT8_C(  26), INT8_C(   6), INT8_C( -66), INT8_C(   9),
                           INT8_C( -98), INT8_C(   1), INT8_C( -26), INT8_C(  92),
                           INT8_C( -33), INT8_C(   8), INT8_C( -81), INT8_C( -93),
                           INT8_C( 116), INT8_C( -55), INT8_C(-113), INT8_C( -43),
                           INT8_C(  32), INT8_C( 105), INT8_C( -27), INT8_C(  37),
                           INT8_C(  72), INT8_C( -73), INT8_C( -19), INT8_C(  96),
                           INT8_C(  52), INT8_C( -98), INT8_C( -45), INT8_C( -41)),
      UINT64_C(0xbe79730e8e7d7e85) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_cmpeq_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmpeq_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int16_t a[32];
    const int16_t b[32];
    const uint32_t r;
  } test_vec[8] = {
    { { -INT16_C( 17530), -INT16_C(  2175), -INT16_C( 13708),  INT16_C(  8082), -INT16_C( 15341), -INT16_C(  9321),  INT16_C( 29543), -INT16_C( 19946),
        -INT16_C(  6245), -INT16_C(  2367),  INT16_C( 27494),  INT16_C( 20111), -INT16_C(  1569),  INT16_C( 27220),  INT16_C(   592), -INT16_C( 10525),
         INT16_C( 25790),  INT16_C( 13006),  INT16_C( 24622),  INT16_C( 16722), -INT16_C(  5851), -INT16_C( 29668),  INT16_C( 13149), -INT16_C(  1985),
         INT16_C(    26), -INT16_C( 32274),  INT16_C( 32363),  INT16_C( 19151),  INT16_C(  9335), -INT16_C( 14412), -INT16_C( 26842), -INT16_C(  7011) },
      {  INT16_C( 27643),  INT16_C( 10519),  INT16_C( 27084), -INT16_C(  3734), -INT16_C( 30894), -INT16_C( 20611), -INT16_C( 17222), -INT16_C( 11096),
        -INT16_C(  6245), -INT16_C(  2367),  INT16_C( 27494),  INT16_C( 20111), -INT16_C(  1569),  INT16_C( 27220),  INT16_C(   592), -INT16_C( 10525),
         INT16_C( 27483),  INT16_C( 10208),  INT16_C( 19412),  INT16_C(  9752), -INT16_C( 26926), -INT16_C( 29482),  INT16_C( 32338),  INT16_C(  3936),
        -INT16_C( 18924),  INT16_C( 10550), -INT16_C( 22309),  INT16_C(  9396),  INT16_C(  1997), -INT16_C( 30061), -INT16_C(  6153),  INT16_C( 21057) },
      UINT32_C(     65280) },
    { {  INT16_C(  8786),  INT16_C(  9850), -INT16_C( 28051),  INT16_C( 16205),  INT16_C(  9000),  INT16_C( 31691),  INT16_C( 11169), -INT16_C( 19062),
        -INT16_C( 16159), -INT16_C( 17186), -INT16_C( 27800),  INT16_C( 14048),  INT16_C( 29850), -INT16_C( 28224),  INT16_C(   347), -INT16_C( 20765),
         INT16_C( 23843), -INT16_C( 28460),  INT16_C(  8688),  INT16_C(  6351), -INT16_C( 26044), -INT16_C(  6765),  INT16_C(  7622), -INT16_C( 22629),
         INT16_C( 31198),  INT16_C( 18020),  INT16_C( 17420), -INT16_C( 22916),  INT16_C( 15544),  INT16_C(  5175),  INT16_C(  6974),  INT16_C( 25026) },
      { -INT16_C( 27016),  INT16_C( 26866), -INT16_C( 15944), -INT16_C(   895),  INT16_C(  5212),  INT16_C(  8930),  INT16_C( 32050),  INT16_C(  4297),
         INT16_C( 11766),  INT16_C(   854), -INT16_C( 11406),  INT16_C( 10921), -INT16_C(  7921),  INT16_C( 19774),  INT16_C(   252),  INT16_C( 29871),
         INT16_C( 23843), -INT16_C( 28460),  INT16_C(  8688),  INT16_C(  6351), -INT16_C( 26044), -INT16_C(  6765),  INT16_C(  7622), -INT16_C( 22629),
         INT16_C(  3031),  INT16_C( 18852),  INT16_C( 19934), -INT16_C(  4748), -INT16_C( 19922),  INT16_C( 10811), -INT16_C(  5453),  INT16_C( 19103) },
      UINT32_C(         0) },
    { {  INT16_C( 31883), -INT16_C( 25319), -INT16_C(  6950),  INT16_C( 19628), -INT16_C( 29678), -INT16_C( 17167), -INT16_C( 23242),  INT16_C(  3677),
         INT16_C(   432), -INT16_C( 25319), -INT16_C( 13489),  INT16_C( 32124), -INT16_C( 18562),  INT16_C( 12712),  INT16_C( 18337),  INT16_C( 11387),
         INT16_C(  5315), -INT16_C( 25319), -INT16_C( 14856),  INT16_C(  2793), -INT16_C(  9646), -INT16_C( 30521),  INT16_C(  9344),  INT16_C( 12438),
        -INT16_C(  4570),  INT16_C( 25319),  INT16_C( 15289),  INT16_C( 14322), -INT16_C( 25870), -INT16_C( 27800), -INT16_C(  7199), -INT16_C( 23361) },
      { -INT16_C(  9993), -INT16_C( 25319),  INT16_C( 11166), -INT16_C(  3846), -INT16_C( 16123), -INT16_C( 31368),  INT16_C(  4070),  INT16_C(  3254),
         INT16_C( 30205), -INT16_C( 25319),  INT16_C( 29616), -INT16_C( 23826),  INT16_C( 22030), -INT16_C(  4299), -INT16_C(  3014),  INT16_C( 12692),
        -INT16_C( 10804),  INT16_C( 25319),  INT16_C(  7168),  INT16_C(  1626), -INT16_C( 11299), -INT16_C( 15477),  INT16_C( 16866), -INT16_C(  8241),
         INT16_C( 20662),  INT16_C( 25319), -INT16_C( 31804), -INT16_C( 11768),  INT16_C( 15834),  INT16_C(  5313),  INT16_C( 21809), -INT16_C(   443) },
      UINT32_C(       514) },
    { {  INT16_C( 26411),  INT16_C( 11112), -INT16_C(  2345),  INT16_C( 24625), -INT16_C( 17002),  INT16_C( 30756), -INT16_C(  3074), -INT16_C( 19113),
        -INT16_C(  5052),  INT16_C(  2075),  INT16_C(  2345),  INT16_C( 19162), -INT16_C( 25759), -INT16_C( 27810), -INT16_C( 23567),  INT16_C(  7313),
        -INT16_C(  1782), -INT16_C( 29369),  INT16_C(  2345),  INT16_C( 21230),  INT16_C(  4662),  INT16_C( 13514),  INT16_C(  8453),  INT16_C( 18921),
         INT16_C(  1294),  INT16_C( 32337),  INT16_C(  2345), -INT16_C( 30008),  INT16_C(  9927), -INT16_C( 18403), -INT16_C( 20791), -INT16_C( 11052) },
      {  INT16_C(  7080),  INT16_C( 25697),  INT16_C(  2345), -INT16_C( 13641), -INT16_C( 32415),  INT16_C( 26623), -INT16_C(  5981), -INT16_C( 20048),
         INT16_C(   749),  INT16_C(  5679), -INT16_C(  2345), -INT16_C(  2911), -INT16_C( 16867), -INT16_C(  6484), -INT16_C( 32659),  INT16_C(  5562),
         INT16_C(  7324),  INT16_C( 12409),  INT16_C(  2345), -INT16_C( 12805), -INT16_C(  1358),  INT16_C( 21812), -INT16_C(  6942), -INT16_C( 12282),
         INT16_C( 13798),  INT16_C(  5350), -INT16_C(  2345),  INT16_C( 18696), -INT16_C( 19130), -INT16_C( 19665), -INT16_C(  5579), -INT16_C( 11832) },
      UINT32_C(         0) },
    { {  INT16_C( 16646),  INT16_C( 28930), -INT16_C(   654),  INT16_C(  4422),  INT16_C( 29431), -INT16_C(  9863),  INT16_C( 32599),  INT16_C( 15785),
        -INT16_C( 28492), -INT16_C(  8111),  INT16_C( 23063),  INT16_C(  4422),  INT16_C( 22543),  INT16_C( 17424), -INT16_C( 10174),  INT16_C( 18454),
         INT16_C(  6170), -INT16_C( 29510), -INT16_C(  2027),  INT16_C(  4422),  INT16_C( 10603), -INT16_C( 15643), -INT16_C( 28760),  INT16_C( 23807),
         INT16_C( 20767),  INT16_C( 13884),  INT16_C( 26027), -INT16_C(  4422), -INT16_C( 23363),  INT16_C(   254),  INT16_C(  5245), -INT16_C( 26808) },
      {  INT16_C(   556),  INT16_C( 16675), -INT16_C( 11269),  INT16_C(  4422),  INT16_C( 13308), -INT16_C( 23512),  INT16_C( 10178), -INT16_C(  7936),
         INT16_C( 15480),  INT16_C(  8983), -INT16_C( 21599),  INT16_C(  4422), -INT16_C(  9136), -INT16_C( 12962), -INT16_C( 22544),  INT16_C(  7524),
        -INT16_C( 30807), -INT16_C( 23458), -INT16_C( 21414),  INT16_C(  4422),  INT16_C( 13023), -INT16_C( 24070), -INT16_C(  1446), -INT16_C( 11646),
        -INT16_C( 26314), -INT16_C( 10250), -INT16_C( 11451), -INT16_C(  4422), -INT16_C( 27473), -INT16_C( 24478), -INT16_C( 14789), -INT16_C(  6979) },
      UINT32_C(      2056) },
    { {  INT16_C(  6989), -INT16_C( 22647), -INT16_C( 27705), -INT16_C( 22787), -INT16_C(  8765),  INT16_C(  8263), -INT16_C( 13839),  INT16_C( 10226),
        -INT16_C(  6045), -INT16_C( 22274),  INT16_C( 13244),  INT16_C( 27453), -INT16_C(  8765),  INT16_C(   523), -INT16_C( 14235), -INT16_C( 19737),
         INT16_C( 28900), -INT16_C( 21671),  INT16_C( 22019), -INT16_C( 13998), -INT16_C(  8765),  INT16_C( 16105), -INT16_C(  9117), -INT16_C( 14747),
         INT16_C( 25540), -INT16_C( 32658), -INT16_C( 21610),  INT16_C( 24300), -INT16_C(  8765), -INT16_C( 20640),  INT16_C( 18368), -INT16_C( 23455) },
      { -INT16_C( 17737), -INT16_C( 17585), -INT16_C( 24304),  INT16_C( 23940),  INT16_C(  8765), -INT16_C( 24933),  INT16_C(    74),  INT16_C(  3684),
        -INT16_C( 11677), -INT16_C(  1649),  INT16_C( 31613), -INT16_C( 14505), -INT16_C(  8765),  INT16_C( 12918), -INT16_C( 10241), -INT16_C( 18474),
         INT16_C(  9873), -INT16_C( 24206), -INT16_C(  2361),  INT16_C(   766), -INT16_C(  8765), -INT16_C( 20832),  INT16_C(  1177), -INT16_C(   835),
         INT16_C( 19670),  INT16_C( 21493),  INT16_C( 19911),  INT16_C( 14618), -INT16_C(  8765),  INT16_C(  1132),  INT16_C( 16999), -INT16_C(  1861) },
      UINT32_C(      4096) },
    { {  INT16_C( 11624),  INT16_C( 12441), -INT16_C( 26844), -INT16_C( 30670), -INT16_C( 11472), -INT16_C( 12345), -INT16_C(  2857), -INT16_C( 20795),
        -INT16_C( 17600),  INT16_C(  1793),  INT16_C(  7176),  INT16_C(  3392), -INT16_C( 21332),  INT16_C( 12345), -INT16_C( 12817),  INT16_C( 22284),
        -INT16_C( 22790),  INT16_C(  7815), -INT16_C( 17859),  INT16_C( 28327), -INT16_C(  8563),  INT16_C( 12345), -INT16_C(   558),  INT16_C(  4626),
         INT16_C(  5304), -INT16_C( 16359),  INT16_C( 22832), -INT16_C(  9011), -INT16_C(  8698), -INT16_C( 12345), -INT16_C(   597), -INT16_C( 22964) },
      { -INT16_C( 11101), -INT16_C(  7996),  INT16_C( 27534),  INT16_C(  6990), -INT16_C( 31159),  INT16_C( 12345), -INT16_C( 28029),  INT16_C( 15149),
         INT16_C( 18086), -INT16_C( 10501), -INT16_C( 14176), -INT16_C( 22862), -INT16_C( 23642),  INT16_C( 12345), -INT16_C(  6240),  INT16_C( 17400),
        -INT16_C( 17221),  INT16_C( 18723),  INT16_C( 29224),  INT16_C( 29028), -INT16_C(  6920),  INT16_C( 12345), -INT16_C( 17802),  INT16_C(  7350),
        -INT16_C( 20223), -INT16_C( 24078), -INT16_C( 23431),  INT16_C(  8007), -INT16_C(  7609), -INT16_C( 12345),  INT16_C( 27081), -INT16_C( 31446) },
      UINT32_C(      8192) },
    { {  INT16_C( 20006),  INT16_C( 20174),  INT16_C( 13248), -INT16_C( 18241),  INT16_C( 19479), -INT16_C( 29389), -INT16_C(  5881),  INT16_C(  2217),
        -INT16_C( 25702),  INT16_C(  5033), -INT16_C(  4033), -INT16_C( 30926), -INT16_C( 23342), -INT16_C( 25746), -INT16_C( 26355),  INT16_C( 13088),
        -INT16_C(  4121), -INT16_C( 22655),  INT16_C( 16674),  INT16_C( 14687), -INT16_C( 28019), -INT16_C( 27450),  INT16_C( 28539),  INT16_C(  5532),
         INT16_C( 17674),  INT16_C( 18728),  INT16_C( 23093),  INT16_C(  2000),  INT16_C( 16382),  INT16_C(  3235), -INT16_C( 15400), -INT16_C( 16577) },
      { -INT16_C( 15950), -INT16_C( 11162), -INT16_C( 15102), -INT16_C( 28915), -INT16_C( 11433), -INT16_C( 11740), -INT16_C( 16318),  INT16_C( 19687),
         INT16_C(  3846),  INT16_C( 15254),  INT16_C( 26217),  INT16_C( 26691), -INT16_C(  6491),  INT16_C( 32116), -INT16_C( 19543),  INT16_C( 23612),
        -INT16_C( 23948),  INT16_C( 30256),  INT16_C( 15975), -INT16_C( 16890),  INT16_C( 10769),  INT16_C( 21648),  INT16_C( 30698), -INT16_C(  3936),
         INT16_C( 13958), -INT16_C(  4052),  INT16_C( 28573),  INT16_C( 16984), -INT16_C( 13227), -INT16_C(   320), -INT16_C(   897), -INT16_C(  2982) },
      UINT32_C(         0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epi16_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmpeq_epi16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u16x32();
    easysimd__m512i b = easysimd_test_x86_random_u16x32();
    easysimd__mmask32 r = easysimd_mm512_cmpeq_epi16_mask(a, b);

    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpeq_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint32_t a[16];
    const uint32_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { { UINT32_C(2846675896), UINT32_C(  47043300), UINT32_C( 640876255), UINT32_C( 699289613), UINT32_C(1521546431), UINT32_C(  33301209), UINT32_C( 565810230), UINT32_C(1620427432),
        UINT32_C(2047492501), UINT32_C(4085045011), UINT32_C(3810111190), UINT32_C(3188508671), UINT32_C(2484714683), UINT32_C( 362091999), UINT32_C(1379290794), UINT32_C(2796735504) },
      { UINT32_C( 555793421), UINT32_C(1779735700), UINT32_C(1246571851), UINT32_C(2953337588), UINT32_C(4114882838), UINT32_C(3775584823), UINT32_C( 959660328), UINT32_C( 450880781),
        UINT32_C(2047492501), UINT32_C(4085045011), UINT32_C(3810111190), UINT32_C(3188508671), UINT32_C(2484714683), UINT32_C( 362091999), UINT32_C(1379290794), UINT32_C(2796735504) },
      UINT16_C(65280) },
    { { UINT32_C( 442696319), UINT32_C(1122334455), UINT32_C( 943665648), UINT32_C(3314966634), UINT32_C(  89582021), UINT32_C(3865954631), UINT32_C(4187098734), UINT32_C( 220617102),
        UINT32_C(2888272219), UINT32_C(1122334455), UINT32_C(3300243802), UINT32_C(3750316825), UINT32_C(2095374388), UINT32_C( 392319401), UINT32_C(3977377119), UINT32_C(2851747662) },
      { UINT32_C(1297424833), UINT32_C(1122334455), UINT32_C(2741534601), UINT32_C(4286771914), UINT32_C(2071684818), UINT32_C( 378723767), UINT32_C( 537109458), UINT32_C(2630483419),
        UINT32_C(1760108318), UINT32_C(1122334455), UINT32_C(4263212339), UINT32_C( 972922215), UINT32_C(3132389379), UINT32_C( 684738133), UINT32_C(3309884394), UINT32_C(4016116432) },
      UINT16_C(  514) },
    { { UINT32_C( 542591537), UINT32_C( 848848126), UINT32_C( 223344453), UINT32_C(1434004817), UINT32_C(4195298725), UINT32_C(1562566771), UINT32_C(2216848307), UINT32_C(2926805885),
        UINT32_C(3436104398), UINT32_C( 134112814), UINT32_C( 223344453), UINT32_C(  12697691), UINT32_C(1643892973), UINT32_C(1690181040), UINT32_C(  99148168), UINT32_C( 850615140) },
      { UINT32_C(1426030885), UINT32_C(  39583208), UINT32_C( 223344453), UINT32_C(1367879523), UINT32_C(2964488960), UINT32_C( 689205408), UINT32_C(3056532561), UINT32_C(2112414295),
        UINT32_C(1272047459), UINT32_C( 290335973), UINT32_C( 223344453), UINT32_C(3968082156), UINT32_C(1151088291), UINT32_C(4167938470), UINT32_C(  95329197), UINT32_C(3783431805) },
      UINT16_C( 1028) },
    { { UINT32_C(1663849342), UINT32_C(1333033599), UINT32_C( 595725367), UINT32_C(1111122222), UINT32_C(3793005372), UINT32_C( 182092124), UINT32_C(2584709149), UINT32_C(2642120991),
        UINT32_C(1660987620), UINT32_C(1504867362), UINT32_C(2910598272), UINT32_C(1111122222), UINT32_C(2472120887), UINT32_C( 832385812), UINT32_C(3687558332), UINT32_C( 561530685) },
      { UINT32_C( 293959919), UINT32_C(1818965996), UINT32_C(2786781036), UINT32_C(1111122222), UINT32_C(1044215338), UINT32_C(1701829545), UINT32_C(3309321095), UINT32_C(1910945922),
        UINT32_C( 478309168), UINT32_C( 260697507), UINT32_C(1219863509), UINT32_C(1111122222), UINT32_C(4070846280), UINT32_C(2455180042), UINT32_C(3495401550), UINT32_C(2168536400) },
      UINT16_C( 2056) },
    { { UINT32_C(1285407913), UINT32_C(2254120625), UINT32_C(  80613577), UINT32_C( 728416483), UINT32_C( 334423233), UINT32_C(1888253217), UINT32_C(1564533517), UINT32_C(3403579937),
        UINT32_C(4145445958), UINT32_C(1820225954), UINT32_C(1685081217), UINT32_C(4254128653), UINT32_C( 334423233), UINT32_C( 796558370), UINT32_C(2290989671), UINT32_C(2186439484) },
      { UINT32_C(2323278055), UINT32_C(1542912217), UINT32_C(1371498052), UINT32_C( 692997953), UINT32_C( 334423233), UINT32_C( 827305418), UINT32_C(2763709544), UINT32_C( 791088199),
        UINT32_C(1320788341), UINT32_C(3718885273), UINT32_C(1445947669), UINT32_C(3045096888), UINT32_C( 334423233), UINT32_C(3384747105), UINT32_C(1198356736), UINT32_C(4202075269) },
      UINT16_C( 4112) },
    { { UINT32_C(3460902709), UINT32_C(4104975070), UINT32_C( 340450139), UINT32_C( 482986584), UINT32_C(3752435326), UINT32_C( 543210123), UINT32_C(1728714209), UINT32_C(3747709098),
        UINT32_C(2393811632), UINT32_C(4169292445), UINT32_C(2366426421), UINT32_C( 363452055), UINT32_C( 938759028), UINT32_C( 543210123), UINT32_C(1543766962), UINT32_C( 876307844) },
      { UINT32_C(2915232016), UINT32_C(2024097091), UINT32_C(2835788306), UINT32_C(4240420744), UINT32_C(3191059202), UINT32_C( 543210123), UINT32_C(3026083632), UINT32_C(3572013508),
        UINT32_C(3330386819), UINT32_C(  37693424), UINT32_C(1638614489), UINT32_C(4150094580), UINT32_C(1857392926), UINT32_C( 543210123), UINT32_C( 866373231), UINT32_C(3926428519) },
      UINT16_C( 8224) },
    { { UINT32_C( 649169206), UINT32_C(2301161648), UINT32_C( 703255605), UINT32_C(1545619518), UINT32_C(2546652633), UINT32_C(1418017509), UINT32_C(1234567890), UINT32_C(3931803828),
        UINT32_C(3390114585), UINT32_C( 810760699), UINT32_C(1280982541), UINT32_C(1604876934), UINT32_C( 905343824), UINT32_C(3062528941), UINT32_C(1234567890), UINT32_C(3141697698) },
      { UINT32_C(2273661836), UINT32_C(2595805581), UINT32_C(2649100823), UINT32_C(3707539340), UINT32_C(2937254402), UINT32_C( 291871853), UINT32_C(1234567890), UINT32_C(2550901772),
        UINT32_C(2350879231), UINT32_C(2166871914), UINT32_C(1981681129), UINT32_C(2656180892), UINT32_C(2035180556), UINT32_C(2928390912), UINT32_C(1234567890), UINT32_C(2699233952) },
      UINT16_C(16448) },
    { { UINT32_C(  86770331), UINT32_C(3280360410), UINT32_C(4248413281), UINT32_C(3399257278), UINT32_C(4047760112), UINT32_C(3701460637), UINT32_C(1663540931), UINT32_C( 987654321),
        UINT32_C(3880267789), UINT32_C(3836399235), UINT32_C( 820175986), UINT32_C(1627029104), UINT32_C(  89210216), UINT32_C(3487756556), UINT32_C( 926026127), UINT32_C( 987654321) },
      { UINT32_C(3909665382), UINT32_C(  47100560), UINT32_C( 103985302), UINT32_C(2523343918), UINT32_C(1973139561), UINT32_C( 944012713), UINT32_C(2607773575), UINT32_C( 987654321),
        UINT32_C(1006421163), UINT32_C( 222153334), UINT32_C(2819846010), UINT32_C(  87980699), UINT32_C(3699038771), UINT32_C(3725967191), UINT32_C(3816391734), UINT32_C( 987654321) },
      UINT16_C(32896) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epu32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmpeq_epu32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u32x16();
    easysimd__m512i b = easysimd_test_x86_random_u32x16();
    easysimd__mmask16 r = easysimd_mm512_cmpeq_epu32_mask(a, b);

    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpeq_epu64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint64_t a[8];
    const uint64_t b[8];
    const uint8_t r;
  } test_vec[8] = {
    { { UINT64_C( 2615591445491065718), UINT64_C( 1634401326662604411), UINT64_C( 6718272270732061212), UINT64_C( 4630128997490972073),
        UINT64_C( 4241361430014217512), UINT64_C(12509814016792199725), UINT64_C(13086038896884052531), UINT64_C(17821488984243849800) },
      { UINT64_C(16279541460740562418), UINT64_C(  273878240964511374), UINT64_C(13886245235001951476), UINT64_C(12820382266304960804),
        UINT64_C(13048984371357457026), UINT64_C( 8369837103748841045), UINT64_C( 7376660080775948071), UINT64_C(15758734290404913730) },
      UINT8_C(  0) },
    { { UINT64_C(13311277529392288991), UINT64_C(11112222333344445555), UINT64_C( 8868810570954444651), UINT64_C(10573658850269621898),
        UINT64_C(11339593230587579971), UINT64_C(11112222333344445555), UINT64_C( 5341950583200245870), UINT64_C( 1161905397425872857) },
      { UINT64_C(17170840044688553912), UINT64_C(11112222333344445555), UINT64_C(15341131676470382556), UINT64_C(13052135394968081220),
        UINT64_C( 4337159874713802807), UINT64_C(11112222333344445555), UINT64_C( 8313089862934825616), UINT64_C( 8488905133269499675) },
      UINT8_C( 34) },
    { { UINT64_C(11576758897045090943), UINT64_C(  165458301244174621), UINT64_C(10304213353958829149), UINT64_C(12204320867389768874),
        UINT64_C( 1957022938994906791), UINT64_C( 2712342115804902447), UINT64_C(14383003093974528408), UINT64_C( 6703905696786694453) },
      { UINT64_C(  809819846637592356), UINT64_C(16352659628652043778), UINT64_C(10304213353958829149), UINT64_C( 5250337504185882057),
        UINT64_C(15202367628177555729), UINT64_C( 8787803765686970055), UINT64_C( 3678171417080816174), UINT64_C(17942216562042926227) },
      UINT8_C(  4) },
    { { UINT64_C(15776636242888125007), UINT64_C( 5257916394853076358), UINT64_C( 2260478860993368001), UINT64_C(17197231154174234825),
        UINT64_C(17522941470666755648), UINT64_C(13469818952648563083), UINT64_C( 6278937682226034807), UINT64_C( 1216446682037221716) },
      { UINT64_C( 8619487182507470397), UINT64_C(17175615687333056426), UINT64_C(15756148437977023171), UINT64_C(17197231154174234825),
        UINT64_C(10088318125444977052), UINT64_C(16959583302533979638), UINT64_C( 8227766883525152461), UINT64_C( 5085758783205997462) },
      UINT8_C(  8) },
    { { UINT64_C(13474435573490900111), UINT64_C( 9727639650585554412), UINT64_C(13377951636146240723), UINT64_C(14336157415948417027),
        UINT64_C( 9123508424627766048), UINT64_C( 2135710395813512548), UINT64_C( 9919737306453081550), UINT64_C(13614709998745967092) },
      { UINT64_C(14122511486876133869), UINT64_C(17251746601171455747), UINT64_C(12542676392168373505), UINT64_C( 6790243451999438562),
        UINT64_C( 1951988386027723763), UINT64_C( 8969994506622006891), UINT64_C(14264892211622795920), UINT64_C(13614709998745967092) },
      UINT8_C(128) },
    { { UINT64_C( 5588675640063690586), UINT64_C( 6477113913096402668), UINT64_C(11109302458535015725), UINT64_C( 8471910930901212266),
        UINT64_C( 6160208955653452365), UINT64_C( 6525328467796186732), UINT64_C( 3877674915669487749), UINT64_C( 9255121757876710880) },
      { UINT64_C( 4178264915597875263), UINT64_C(15891228736548157571), UINT64_C(11109302458535015725), UINT64_C( 8271993432970716696),
        UINT64_C( 3271045620182907998), UINT64_C(18231385824197071822), UINT64_C(11577665610379976025), UINT64_C(11974149231296135815) },
      UINT8_C(  4) },
    { { UINT64_C( 3029856421833426485), UINT64_C(11878054127463327498), UINT64_C( 8419589369584143280), UINT64_C(12939786755068839333),
        UINT64_C(11900654933161174433), UINT64_C(12169634364235840091), UINT64_C(13151226762957663522), UINT64_C(12502616043679887696) },
      { UINT64_C(13393091434532967580), UINT64_C(11878054127463327498), UINT64_C( 2194491956588135085), UINT64_C(16920146181431826199),
        UINT64_C(11900654933161174433), UINT64_C(12169634364235840091), UINT64_C( 4929429680196863867), UINT64_C( 5127972441748362737) },
      UINT8_C( 50) },
    { { UINT64_C(12074552249083120817), UINT64_C(17224748449983183719), UINT64_C( 4310403659939278944), UINT64_C( 2306307482990622966),
        UINT64_C( 6117488002825224178), UINT64_C(10614069748653529833), UINT64_C( 2069106226383324464), UINT64_C(14607663811869946153) },
      { UINT64_C(12074552249083120817), UINT64_C( 4633664228225011591), UINT64_C(10006560082444906395), UINT64_C(16448513135454893218),
        UINT64_C(13880025027994733744), UINT64_C( 1756907968432662545), UINT64_C( 3468819056316521529), UINT64_C(15122504953470010988) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epu64_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_cmpeq_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else

  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m512i b = easysimd_test_x86_random_u64x8();
    easysimd__mmask8 r = easysimd_mm512_cmpeq_epu64_mask(a, b);

    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpeq_epu8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t   arr_a[64];
    uint8_t   arr_b[64];
    easysimd__mmask64 r;
  } test_vec[] = {
    { { UINT8_C( 36), UINT8_C(105), UINT8_C( 21), UINT8_C(  6), UINT8_C(  1), UINT8_C(128), UINT8_C(192), UINT8_C(253),
        UINT8_C(237), UINT8_C( 60), UINT8_C(135), UINT8_C(130), UINT8_C( 22), UINT8_C(236), UINT8_C( 45), UINT8_C( 17),
        UINT8_C(176), UINT8_C(180), UINT8_C( 97), UINT8_C(141), UINT8_C(143), UINT8_C(136), UINT8_C(175), UINT8_C( 12),
        UINT8_C(121), UINT8_C( 28), UINT8_C( 33), UINT8_C(115), UINT8_C( 35), UINT8_C(150), UINT8_C(177), UINT8_C( 71),
           UINT8_MAX, UINT8_C(198), UINT8_C( 78), UINT8_C(  0), UINT8_C( 70), UINT8_C( 14), UINT8_C(253), UINT8_C( 51),
        UINT8_C( 75), UINT8_C(132), UINT8_C(181), UINT8_C( 97), UINT8_C(113), UINT8_C(227), UINT8_C(115), UINT8_C( 33),
        UINT8_C(151), UINT8_C(212), UINT8_C(175), UINT8_C( 38), UINT8_C( 93), UINT8_C( 94), UINT8_C( 50), UINT8_C(214),
        UINT8_C(123), UINT8_C( 84), UINT8_C( 73), UINT8_C(158), UINT8_C(234), UINT8_C(250), UINT8_C(230), UINT8_C(233) },
      { UINT8_C(193), UINT8_C( 52), UINT8_C(233), UINT8_C(  7), UINT8_C( 66), UINT8_C(230), UINT8_C( 59), UINT8_C(141),
        UINT8_C(107), UINT8_C(240), UINT8_C(239), UINT8_C(220), UINT8_C(211), UINT8_C( 98), UINT8_C(253), UINT8_C(106),
        UINT8_C( 54), UINT8_C(172), UINT8_C(144), UINT8_C(147), UINT8_C( 11), UINT8_C(195), UINT8_C(105), UINT8_C(134),
        UINT8_C( 23), UINT8_C(179), UINT8_C( 36), UINT8_C(  1), UINT8_C(173), UINT8_C( 10), UINT8_C(234), UINT8_C(110),
        UINT8_C( 62), UINT8_C(211), UINT8_C(118), UINT8_C(129), UINT8_C(185), UINT8_C(177), UINT8_C( 14), UINT8_C( 36),
        UINT8_C(161), UINT8_C(253), UINT8_C(  0), UINT8_C(117), UINT8_C( 95), UINT8_C(254), UINT8_C(223), UINT8_C(150),
        UINT8_C(170), UINT8_C(112), UINT8_C( 41), UINT8_C(181), UINT8_C( 51), UINT8_C(147), UINT8_C( 59), UINT8_C( 74),
        UINT8_C( 70), UINT8_C( 96), UINT8_C( 75), UINT8_C(243), UINT8_C(106), UINT8_C( 53), UINT8_C( 98), UINT8_C(169) },
      UINT64_C(                   0) },
    { { UINT8_C(  8), UINT8_C(216), UINT8_C( 42), UINT8_C(193), UINT8_C(137), UINT8_C( 56), UINT8_C(230), UINT8_C( 42),
        UINT8_C( 54), UINT8_C(230), UINT8_C(159), UINT8_C(149), UINT8_C(228), UINT8_C(127), UINT8_C( 43), UINT8_C(143),
        UINT8_C(239), UINT8_C( 85), UINT8_C( 68), UINT8_C( 34), UINT8_C(232), UINT8_C(128), UINT8_C(108), UINT8_C( 46),
        UINT8_C(224), UINT8_C(183), UINT8_C( 33), UINT8_C( 74), UINT8_C(236), UINT8_C(131), UINT8_C(243), UINT8_C(244),
        UINT8_C( 91), UINT8_C( 29), UINT8_C(181), UINT8_C(228), UINT8_C( 86), UINT8_C(155), UINT8_C( 15), UINT8_C(140),
        UINT8_C(130), UINT8_C(174), UINT8_C( 33), UINT8_C(102), UINT8_C( 45), UINT8_C( 77), UINT8_C(245), UINT8_C( 28),
        UINT8_C(162), UINT8_C( 58), UINT8_C( 62), UINT8_C(138), UINT8_C(186), UINT8_C(170), UINT8_C(184), UINT8_C(154),
        UINT8_C( 97), UINT8_C(217), UINT8_C(228), UINT8_C( 77), UINT8_C( 93), UINT8_C(216), UINT8_C( 65), UINT8_C(184) },
      { UINT8_C(245), UINT8_C(247), UINT8_C(157), UINT8_C( 75), UINT8_C(146), UINT8_C(172), UINT8_C(215), UINT8_C( 20),
        UINT8_C( 90), UINT8_C(249), UINT8_C(123), UINT8_C(136), UINT8_C( 70), UINT8_C(112), UINT8_C(164), UINT8_C(232),
        UINT8_C(170), UINT8_C(227), UINT8_C(114), UINT8_C(100), UINT8_C(141), UINT8_C( 42), UINT8_C(254), UINT8_C(239),
        UINT8_C(  3), UINT8_C(227), UINT8_C( 60), UINT8_C( 96), UINT8_C(187), UINT8_C(126), UINT8_C( 25), UINT8_C(176),
        UINT8_C(117), UINT8_C(182), UINT8_C(252), UINT8_C(  7), UINT8_C( 98), UINT8_C(211), UINT8_C( 28), UINT8_C(188),
        UINT8_C(204), UINT8_C(151), UINT8_C( 68), UINT8_C( 18), UINT8_C(  7), UINT8_C(233), UINT8_C(250), UINT8_C(178),
        UINT8_C(204), UINT8_C(108), UINT8_C( 22), UINT8_C( 89), UINT8_C(150), UINT8_C( 21), UINT8_C( 72), UINT8_C(154),
        UINT8_C(248), UINT8_C(133), UINT8_C(250), UINT8_C(179), UINT8_C(  3), UINT8_C( 19), UINT8_C( 99), UINT8_C(120) },
      UINT64_C(   36028797018963968) },
    { { UINT8_C(201), UINT8_C( 95), UINT8_C(127), UINT8_C( 43), UINT8_C( 51), UINT8_C(155), UINT8_C(232),    UINT8_MAX,
        UINT8_C( 50), UINT8_C( 44), UINT8_C( 18), UINT8_C( 58), UINT8_C( 21), UINT8_C( 12), UINT8_C(236), UINT8_C(225),
        UINT8_C(121), UINT8_C(  2), UINT8_C( 59), UINT8_C( 15), UINT8_C( 23), UINT8_C(131), UINT8_C(169), UINT8_C( 15),
        UINT8_C(  8), UINT8_C(164), UINT8_C(194), UINT8_C( 11), UINT8_C(183), UINT8_C( 38), UINT8_C(131), UINT8_C(129),
        UINT8_C(133), UINT8_C(  3), UINT8_C(172), UINT8_C(184), UINT8_C(158), UINT8_C(148), UINT8_C(184), UINT8_C(209),
        UINT8_C(193), UINT8_C(202), UINT8_C( 11), UINT8_C(214), UINT8_C(214), UINT8_C(247), UINT8_C(184), UINT8_C( 79),
        UINT8_C(249), UINT8_C(243), UINT8_C( 95), UINT8_C( 17), UINT8_C(118), UINT8_C(  8), UINT8_C( 32), UINT8_C(127),
        UINT8_C(172), UINT8_C(227), UINT8_C(138), UINT8_C(100), UINT8_C(  9), UINT8_C( 14), UINT8_C(229), UINT8_C(142) },
      { UINT8_C( 17), UINT8_C(145), UINT8_C( 71), UINT8_C(175), UINT8_C( 38),    UINT8_MAX, UINT8_C(128), UINT8_C(231),
        UINT8_C(201), UINT8_C(139), UINT8_C(189), UINT8_C(159), UINT8_C(130), UINT8_C(117), UINT8_C(239), UINT8_C(124),
        UINT8_C(104), UINT8_C( 78), UINT8_C(141), UINT8_C(223), UINT8_C( 86), UINT8_C(173), UINT8_C( 94), UINT8_C(  3),
        UINT8_C(144), UINT8_C(232), UINT8_C(103), UINT8_C(153), UINT8_C(246), UINT8_C( 76), UINT8_C( 40), UINT8_C(  7),
        UINT8_C(221), UINT8_C(111), UINT8_C(183), UINT8_C(  3), UINT8_C(110), UINT8_C( 55), UINT8_C(234), UINT8_C( 55),
        UINT8_C(195), UINT8_C(168), UINT8_C(214), UINT8_C( 69), UINT8_C( 29), UINT8_C(197), UINT8_C(193), UINT8_C(134),
        UINT8_C( 19), UINT8_C( 78), UINT8_C(101), UINT8_C(106), UINT8_C(252), UINT8_C(195), UINT8_C(109), UINT8_C(140),
        UINT8_C(171), UINT8_C(212), UINT8_C( 38), UINT8_C(162), UINT8_C( 32), UINT8_C( 78), UINT8_C(169), UINT8_C(253) },
      UINT64_C(                   0) },
    { { UINT8_C(189), UINT8_C( 96), UINT8_C(  1), UINT8_C( 43), UINT8_C(152), UINT8_C(235), UINT8_C( 98), UINT8_C( 91),
        UINT8_C(147), UINT8_C( 56), UINT8_C(160), UINT8_C(177), UINT8_C(254), UINT8_C( 98), UINT8_C( 55), UINT8_C( 17),
        UINT8_C(176), UINT8_C(156), UINT8_C(123), UINT8_C(172), UINT8_C( 95), UINT8_C(232), UINT8_C( 57), UINT8_C( 10),
        UINT8_C(188), UINT8_C( 95), UINT8_C(172), UINT8_C(220), UINT8_C(173), UINT8_C( 86), UINT8_C(218), UINT8_C(106),
        UINT8_C(182), UINT8_C(219), UINT8_C(149), UINT8_C( 78), UINT8_C(198), UINT8_C(247), UINT8_C(169), UINT8_C( 90),
        UINT8_C( 47), UINT8_C( 74), UINT8_C( 11), UINT8_C( 45), UINT8_C(172), UINT8_C( 66), UINT8_C( 63), UINT8_C( 92),
        UINT8_C(222), UINT8_C(186), UINT8_C(  9), UINT8_C( 61), UINT8_C(163), UINT8_C( 66), UINT8_C( 71), UINT8_C( 95),
        UINT8_C(161), UINT8_C(244), UINT8_C( 60), UINT8_C( 78), UINT8_C( 74), UINT8_C( 22), UINT8_C(184), UINT8_C(  0) },
      { UINT8_C(241), UINT8_C( 77), UINT8_C( 79), UINT8_C(183), UINT8_C( 68), UINT8_C(248), UINT8_C( 17), UINT8_C(115),
        UINT8_C( 66), UINT8_C( 28), UINT8_C(161), UINT8_C(238), UINT8_C( 94), UINT8_C(224), UINT8_C( 75), UINT8_C( 60),
        UINT8_C(154), UINT8_C( 84), UINT8_C(121), UINT8_C( 61), UINT8_C(150), UINT8_C(193), UINT8_C(157), UINT8_C( 55),
        UINT8_C(181), UINT8_C(217), UINT8_C(133),    UINT8_MAX, UINT8_C(239), UINT8_C( 61),    UINT8_MAX, UINT8_C(224),
        UINT8_C(138), UINT8_C( 78), UINT8_C(151), UINT8_C(206), UINT8_C( 71), UINT8_C(169), UINT8_C( 65), UINT8_C(137),
        UINT8_C(197), UINT8_C(226), UINT8_C(120), UINT8_C( 36), UINT8_C(194), UINT8_C(195), UINT8_C( 96), UINT8_C( 93),
        UINT8_C( 23), UINT8_C(218), UINT8_C(154), UINT8_C(173), UINT8_C(155), UINT8_C( 55), UINT8_C(228), UINT8_C( 80),
        UINT8_C( 16), UINT8_C(105), UINT8_C( 79),    UINT8_MAX, UINT8_C(166), UINT8_C( 78), UINT8_C(223), UINT8_C( 48) },
      UINT64_C(                   0) },
    { { UINT8_C(157), UINT8_C(119), UINT8_C(254), UINT8_C(228), UINT8_C( 32), UINT8_C( 63), UINT8_C(109), UINT8_C(229),
        UINT8_C( 34), UINT8_C(229), UINT8_C(  9), UINT8_C(228), UINT8_C(168), UINT8_C(106), UINT8_C( 65), UINT8_C(191),
        UINT8_C( 68), UINT8_C(220), UINT8_C(108), UINT8_C(223), UINT8_C( 19), UINT8_C( 80), UINT8_C( 47), UINT8_C( 36),
        UINT8_C(185), UINT8_C(126), UINT8_C( 35), UINT8_C( 95), UINT8_C(204), UINT8_C(  3), UINT8_C(143), UINT8_C(105),
        UINT8_C(122), UINT8_C(141), UINT8_C( 77), UINT8_C(154), UINT8_C(205), UINT8_C(187), UINT8_C(127), UINT8_C(239),
        UINT8_C(160), UINT8_C(137), UINT8_C(211), UINT8_C( 73), UINT8_C(243), UINT8_C( 21), UINT8_C(  8), UINT8_C( 55),
        UINT8_C(241), UINT8_C(117), UINT8_C( 22), UINT8_C(  4), UINT8_C(197), UINT8_C( 69), UINT8_C( 40), UINT8_C(127),
        UINT8_C(195), UINT8_C( 76), UINT8_C(222), UINT8_C(143), UINT8_C( 79), UINT8_C(110), UINT8_C(249), UINT8_C(201) },
      { UINT8_C(251), UINT8_C( 70), UINT8_C( 99), UINT8_C(200), UINT8_C(  1), UINT8_C(226), UINT8_C(183), UINT8_C(162),
        UINT8_C(107), UINT8_C(139), UINT8_C(235), UINT8_C( 94), UINT8_C(160), UINT8_C(243), UINT8_C(149), UINT8_C(145),
        UINT8_C(104), UINT8_C(171), UINT8_C(149), UINT8_C( 46), UINT8_C(240), UINT8_C(190), UINT8_C(173), UINT8_C(179),
        UINT8_C( 10), UINT8_C(139), UINT8_C( 67), UINT8_C( 89), UINT8_C(249), UINT8_C( 60), UINT8_C( 34), UINT8_C(245),
        UINT8_C(130), UINT8_C(133), UINT8_C(189), UINT8_C(132), UINT8_C(103), UINT8_C(117), UINT8_C( 38), UINT8_C(211),
        UINT8_C(  0), UINT8_C( 17), UINT8_C( 49), UINT8_C(160), UINT8_C(  4), UINT8_C(199), UINT8_C( 49), UINT8_C(109),
        UINT8_C(114), UINT8_C(198), UINT8_C(155), UINT8_C( 99), UINT8_C(132), UINT8_C( 72), UINT8_C( 22), UINT8_C(142),
        UINT8_C(211), UINT8_C( 89), UINT8_C(231), UINT8_C(205), UINT8_C(149), UINT8_C(  9), UINT8_C(194), UINT8_C( 24) },
      UINT64_C(                   0) },
    { { UINT8_C(142), UINT8_C(127), UINT8_C(156), UINT8_C(246), UINT8_C(244), UINT8_C(194), UINT8_C(201), UINT8_C(244),
        UINT8_C(211), UINT8_C(250), UINT8_C(148), UINT8_C(215), UINT8_C(193), UINT8_C(197), UINT8_C( 68), UINT8_C( 52),
        UINT8_C(140), UINT8_C(223), UINT8_C(151), UINT8_C( 16), UINT8_C( 39), UINT8_C(173), UINT8_C(159), UINT8_C(251),
        UINT8_C(  7), UINT8_C(134), UINT8_C(200), UINT8_C(156), UINT8_C(144), UINT8_C(138), UINT8_C(180), UINT8_C( 30),
        UINT8_C(  9), UINT8_C( 80), UINT8_C( 20), UINT8_C(254), UINT8_C( 18), UINT8_C(221), UINT8_C(242), UINT8_C(229),
        UINT8_C(216), UINT8_C(135), UINT8_C(189), UINT8_C(153), UINT8_C( 76), UINT8_C(  1), UINT8_C(205), UINT8_C(216),
        UINT8_C(225), UINT8_C(100), UINT8_C(233), UINT8_C(  8), UINT8_C( 18), UINT8_C(136), UINT8_C(  3), UINT8_C( 25),
        UINT8_C( 14), UINT8_C(203), UINT8_C(181), UINT8_C(158), UINT8_C( 85), UINT8_C(106), UINT8_C(189), UINT8_C( 95) },
      { UINT8_C(186), UINT8_C(209), UINT8_C( 93), UINT8_C(205), UINT8_C(175), UINT8_C( 79), UINT8_C(178), UINT8_C(135),
        UINT8_C(214), UINT8_C(111), UINT8_C( 32), UINT8_C( 35), UINT8_C(113), UINT8_C(238), UINT8_C(251), UINT8_C( 82),
        UINT8_C( 82), UINT8_C(228), UINT8_C( 90), UINT8_C(100), UINT8_C(108), UINT8_C( 94), UINT8_C(125), UINT8_C(123),
        UINT8_C( 41), UINT8_C( 51), UINT8_C( 25), UINT8_C(127), UINT8_C(157), UINT8_C(214), UINT8_C(222), UINT8_C( 87),
        UINT8_C(168), UINT8_C( 59), UINT8_C( 36), UINT8_C( 87), UINT8_C(138), UINT8_C(215), UINT8_C(222), UINT8_C( 97),
        UINT8_C( 70), UINT8_C(254), UINT8_C(132), UINT8_C(183), UINT8_C(236), UINT8_C(127), UINT8_C(  9), UINT8_C( 63),
        UINT8_C(100), UINT8_C(100), UINT8_C(163), UINT8_C(208), UINT8_C(194), UINT8_C( 33), UINT8_C( 75), UINT8_C(235),
        UINT8_C( 84), UINT8_C(101), UINT8_C(106), UINT8_C(241), UINT8_C( 59), UINT8_C( 72), UINT8_C( 72), UINT8_C(227) },
      UINT64_C(     562949953421312) },
    { { UINT8_C(131), UINT8_C(109), UINT8_C( 58), UINT8_C( 14), UINT8_C( 68), UINT8_C( 24), UINT8_C(111), UINT8_C(138),
        UINT8_C( 23), UINT8_C(243), UINT8_C( 66), UINT8_C(  3), UINT8_C(114), UINT8_C( 75), UINT8_C( 66), UINT8_C(214),
        UINT8_C(175), UINT8_C(230), UINT8_C(167), UINT8_C(113), UINT8_C(  7), UINT8_C(242), UINT8_C( 93), UINT8_C( 91),
        UINT8_C( 87), UINT8_C(199), UINT8_C( 76), UINT8_C(147), UINT8_C( 16), UINT8_C(148), UINT8_C(118), UINT8_C(147),
        UINT8_C(  1), UINT8_C(177), UINT8_C(161), UINT8_C( 69), UINT8_C(201), UINT8_C( 16), UINT8_C(208), UINT8_C(224),
        UINT8_C(  3), UINT8_C( 18), UINT8_C(228), UINT8_C(118), UINT8_C( 93), UINT8_C( 38), UINT8_C( 76), UINT8_C( 13),
        UINT8_C( 12), UINT8_C(243), UINT8_C(126), UINT8_C( 19), UINT8_C(230), UINT8_C(219), UINT8_C(110), UINT8_C( 61),
        UINT8_C(163), UINT8_C(186), UINT8_C(208), UINT8_C(179), UINT8_C( 79), UINT8_C( 71), UINT8_C( 70), UINT8_C( 80) },
      { UINT8_C(248), UINT8_C(232), UINT8_C(150), UINT8_C(193), UINT8_C(248), UINT8_C(102), UINT8_C(162), UINT8_C(252),
        UINT8_C(120), UINT8_C(134), UINT8_C(114), UINT8_C(213), UINT8_C(172), UINT8_C(190), UINT8_C(226), UINT8_C(185),
        UINT8_C(178), UINT8_C( 97), UINT8_C(204), UINT8_C(152), UINT8_C( 60), UINT8_C( 59), UINT8_C(213), UINT8_C(223),
        UINT8_C(245), UINT8_C(166), UINT8_C(146), UINT8_C( 68), UINT8_C(237), UINT8_C(217), UINT8_C(149), UINT8_C(229),
        UINT8_C(193), UINT8_C( 43), UINT8_C(166), UINT8_C(185), UINT8_C(145), UINT8_C( 72), UINT8_C(181), UINT8_C(  9),
        UINT8_C(206), UINT8_C( 39), UINT8_C(222), UINT8_C(123), UINT8_C(230), UINT8_C(193), UINT8_C( 52), UINT8_C(152),
        UINT8_C( 34), UINT8_C(  0), UINT8_C( 48), UINT8_C( 94), UINT8_C( 59), UINT8_C(  5), UINT8_C( 62), UINT8_C( 49),
        UINT8_C(171), UINT8_C(208), UINT8_C(117), UINT8_C(152), UINT8_C(169), UINT8_C( 10), UINT8_C(125), UINT8_C(106) },
      UINT64_C(                   0) },
    { { UINT8_C( 53), UINT8_C( 36), UINT8_C( 36), UINT8_C(198), UINT8_C(108), UINT8_C(217), UINT8_C(207), UINT8_C( 59),
        UINT8_C(  1), UINT8_C(174), UINT8_C(182), UINT8_C(231), UINT8_C(111), UINT8_C(234), UINT8_C(127), UINT8_C(145),
        UINT8_C(234), UINT8_C(175), UINT8_C(239), UINT8_C( 38), UINT8_C(180), UINT8_C( 45), UINT8_C( 87), UINT8_C( 96),
        UINT8_C(254), UINT8_C(204), UINT8_C(248), UINT8_C(167), UINT8_C(215), UINT8_C(118), UINT8_C( 18), UINT8_C( 12),
        UINT8_C(154), UINT8_C( 54), UINT8_C(211), UINT8_C(  6), UINT8_C( 15), UINT8_C(162), UINT8_C( 65), UINT8_C( 16),
        UINT8_C( 80), UINT8_C(247), UINT8_C(247), UINT8_C(191), UINT8_C(225), UINT8_C(118), UINT8_C( 80), UINT8_C(204),
        UINT8_C( 37), UINT8_C( 64), UINT8_C(242), UINT8_C(218), UINT8_C(109), UINT8_C( 73), UINT8_C( 58), UINT8_C(107),
        UINT8_C( 21), UINT8_C( 50), UINT8_C( 19), UINT8_C(236), UINT8_C(168), UINT8_C( 37), UINT8_C(249), UINT8_C( 66) },
      { UINT8_C( 91), UINT8_C(204), UINT8_C( 73), UINT8_C(106), UINT8_C(110), UINT8_C(138), UINT8_C(123), UINT8_C(191),
        UINT8_C(130), UINT8_C(114), UINT8_C(126), UINT8_C( 99), UINT8_C(233), UINT8_C(207), UINT8_C( 47), UINT8_C( 14),
        UINT8_C( 15), UINT8_C( 33), UINT8_C(232), UINT8_C(124), UINT8_C(106), UINT8_C( 34), UINT8_C(232), UINT8_C(128),
        UINT8_C( 85), UINT8_C(251), UINT8_C(108), UINT8_C(253), UINT8_C( 32), UINT8_C(101), UINT8_C( 64), UINT8_C(123),
        UINT8_C( 49), UINT8_C(137), UINT8_C(229), UINT8_C(160), UINT8_C( 19), UINT8_C( 96), UINT8_C( 95), UINT8_C(149),
        UINT8_C(211), UINT8_C(221), UINT8_C(249), UINT8_C(188), UINT8_C(172), UINT8_C( 40), UINT8_C(202), UINT8_C(187),
        UINT8_C( 74), UINT8_C(179), UINT8_C( 56), UINT8_C(180), UINT8_C(213), UINT8_C( 32), UINT8_C( 52), UINT8_C( 42),
        UINT8_C( 27), UINT8_C(161), UINT8_C( 40), UINT8_C( 59), UINT8_C(  6), UINT8_C(104), UINT8_C(182), UINT8_C( 56) },
      UINT64_C(                   0) }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512((void *)&(test_vec[i].arr_a));
    easysimd__m512i b = easysimd_mm512_loadu_si512((void *)&(test_vec[i].arr_b));
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epu8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpeq_epu8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;


#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u8x64();
    easysimd__m512i b = easysimd_test_x86_random_u8x64();
    // easysimd__mmask64 k = easysimd_test_x86_random_mmask8();
    int64_t r = easysimd_mm512_cmpeq_epu8_mask(a, b);

    easysimd_test_x86_write_u8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_cmpeq_epu16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t   arr_a[32];
    uint16_t   arr_b[32];
    easysimd__mmask32 r;
  } test_vec[] = {
    { { UINT16_C( 4906), UINT16_C(56918), UINT16_C(59263), UINT16_C(56311), UINT16_C(49077), UINT16_C(10656), UINT16_C(11280), UINT16_C(43447),
        UINT16_C(28901), UINT16_C(24020), UINT16_C(33875), UINT16_C( 3148), UINT16_C(47700), UINT16_C(55537), UINT16_C(58653), UINT16_C(18384),
        UINT16_C(10232), UINT16_C(30501), UINT16_C( 7182), UINT16_C(50002), UINT16_C(62428), UINT16_C(60652), UINT16_C(41759), UINT16_C( 1173),
        UINT16_C(26899), UINT16_C(26465), UINT16_C(44781), UINT16_C(17011), UINT16_C(25704), UINT16_C(34330), UINT16_C(60233), UINT16_C(16845) },
      { UINT16_C(62226), UINT16_C( 8377), UINT16_C( 2831), UINT16_C(60387), UINT16_C(53246), UINT16_C( 7639), UINT16_C(27763), UINT16_C(34337),
        UINT16_C(33749), UINT16_C(50157), UINT16_C(24881), UINT16_C(39173), UINT16_C( 8133), UINT16_C( 3871), UINT16_C(60682), UINT16_C( 7248),
        UINT16_C( 2528), UINT16_C(61244), UINT16_C( 7957), UINT16_C( 5083), UINT16_C(45807), UINT16_C(25137), UINT16_C(21023), UINT16_C(62696),
        UINT16_C(54997), UINT16_C( 1719), UINT16_C(48183), UINT16_C(64672), UINT16_C(49116), UINT16_C(58891), UINT16_C(23724), UINT16_C(35843) },
      UINT32_C(         0) },
    { { UINT16_C(16229), UINT16_C(31356), UINT16_C(22367), UINT16_C(20110), UINT16_C(48905), UINT16_C(10416), UINT16_C(38929), UINT16_C(59165),
        UINT16_C(54382), UINT16_C(42477), UINT16_C(36241), UINT16_C(28066), UINT16_C(44365), UINT16_C(63827), UINT16_C(22025), UINT16_C(28550),
        UINT16_C(  662), UINT16_C(62953), UINT16_C(30553), UINT16_C(25155), UINT16_C(62262), UINT16_C(18571), UINT16_C(43147), UINT16_C(64047),
        UINT16_C( 7292), UINT16_C( 3487), UINT16_C(16810), UINT16_C(63354), UINT16_C(52975), UINT16_C(63728), UINT16_C(30244), UINT16_C(47719) },
      { UINT16_C(20856), UINT16_C(53679), UINT16_C(62152), UINT16_C(65332), UINT16_C(49125), UINT16_C(28999), UINT16_C(30311), UINT16_C(58219),
        UINT16_C( 2706), UINT16_C(15601), UINT16_C(27468), UINT16_C(15155), UINT16_C( 9273), UINT16_C(24115), UINT16_C(39834), UINT16_C( 4888),
        UINT16_C(51436), UINT16_C(46308), UINT16_C( 6330), UINT16_C(41139), UINT16_C(64215), UINT16_C(15889), UINT16_C(31856), UINT16_C(  802),
        UINT16_C( 4998), UINT16_C(53823), UINT16_C(29566), UINT16_C(47117), UINT16_C(16791), UINT16_C(12566), UINT16_C(11996), UINT16_C(51268) },
      UINT32_C(         0) },
    { { UINT16_C(10742), UINT16_C(45436), UINT16_C(12353), UINT16_C( 6481), UINT16_C(25130), UINT16_C(39767), UINT16_C(31198), UINT16_C(25758),
        UINT16_C(56716), UINT16_C( 2871), UINT16_C(17488), UINT16_C(59331), UINT16_C(55685), UINT16_C(24857), UINT16_C(23815), UINT16_C(65065),
        UINT16_C(42630), UINT16_C(51375), UINT16_C(  214), UINT16_C(  225), UINT16_C(14434), UINT16_C(16539), UINT16_C(14770), UINT16_C(16036),
        UINT16_C(56087), UINT16_C(26441), UINT16_C( 3104), UINT16_C(42319), UINT16_C(26853), UINT16_C(60679), UINT16_C(12485), UINT16_C(19691) },
      { UINT16_C(39638), UINT16_C(44052), UINT16_C(62874), UINT16_C(64685), UINT16_C(18477), UINT16_C(57148), UINT16_C(57474), UINT16_C(39198),
        UINT16_C(26556), UINT16_C(56320), UINT16_C(20340), UINT16_C(22913), UINT16_C(34999), UINT16_C(32070), UINT16_C(12729), UINT16_C(36809),
        UINT16_C(56779), UINT16_C(25916), UINT16_C(59858), UINT16_C(65377), UINT16_C(40241), UINT16_C(46047), UINT16_C(64894), UINT16_C(14924),
        UINT16_C(19812), UINT16_C(55318), UINT16_C(38812), UINT16_C(21554), UINT16_C(30752), UINT16_C(55761), UINT16_C(39594), UINT16_C(30056) },
      UINT32_C(         0) },
    { { UINT16_C(42103), UINT16_C(18907), UINT16_C(15501), UINT16_C(48968), UINT16_C(10202), UINT16_C(22642), UINT16_C(48932), UINT16_C(35218),
        UINT16_C(43020), UINT16_C(43105), UINT16_C(37695), UINT16_C(24572), UINT16_C(52492), UINT16_C(46648), UINT16_C(41319), UINT16_C(56875),
        UINT16_C( 1605), UINT16_C(54055), UINT16_C(28739), UINT16_C( 7570), UINT16_C( 1175), UINT16_C(48245), UINT16_C( 1987), UINT16_C(53061),
        UINT16_C(42671), UINT16_C(61048), UINT16_C(29754), UINT16_C(17998), UINT16_C(34370), UINT16_C(43516), UINT16_C(10023), UINT16_C(28040) },
      { UINT16_C(44846), UINT16_C(28992), UINT16_C(53791), UINT16_C(46990), UINT16_C(  982), UINT16_C(39539), UINT16_C(47114), UINT16_C(47465),
        UINT16_C(57694), UINT16_C(39079), UINT16_C(62806), UINT16_C(39134), UINT16_C(55932), UINT16_C(41793), UINT16_C(51458), UINT16_C(12304),
        UINT16_C(20601), UINT16_C(39073), UINT16_C(12066), UINT16_C(63823), UINT16_C(49714), UINT16_C(15507), UINT16_C(64634), UINT16_C(55797),
        UINT16_C(40158), UINT16_C(13425), UINT16_C(20626), UINT16_C( 3788), UINT16_C( 3370), UINT16_C(11441), UINT16_C(49879), UINT16_C(20572) },
      UINT32_C(         0) },
    { { UINT16_C(64786), UINT16_C(13800), UINT16_C(14380), UINT16_C(24110), UINT16_C(49658), UINT16_C(30106), UINT16_C(36797), UINT16_C(39758),
        UINT16_C(48940), UINT16_C(48847), UINT16_C(39695), UINT16_C(15052), UINT16_C(32169), UINT16_C(32870), UINT16_C(49983), UINT16_C(21200),
        UINT16_C(47296), UINT16_C(60807), UINT16_C(46576), UINT16_C(60235), UINT16_C(58998), UINT16_C(13152), UINT16_C(44661), UINT16_C(41423),
        UINT16_C(40557), UINT16_C(32095), UINT16_C(11066), UINT16_C(58295), UINT16_C( 7593), UINT16_C(59491), UINT16_C(13280), UINT16_C(41274) },
      { UINT16_C(49643), UINT16_C(56462), UINT16_C(55670), UINT16_C(60615), UINT16_C(10175), UINT16_C(13600), UINT16_C(61397), UINT16_C(17110),
        UINT16_C(13965), UINT16_C(51135), UINT16_C(30305), UINT16_C( 2730), UINT16_C( 3476), UINT16_C(29939), UINT16_C(11584), UINT16_C(11285),
        UINT16_C(41967), UINT16_C(25864), UINT16_C(53117), UINT16_C(15442), UINT16_C(29430), UINT16_C(52081), UINT16_C(18529), UINT16_C(60941),
        UINT16_C(52606), UINT16_C(57270), UINT16_C(24643), UINT16_C(55274), UINT16_C(56686), UINT16_C(44620), UINT16_C(24842), UINT16_C(63962) },
      UINT32_C(         0) },
    { { UINT16_C(57861), UINT16_C(33375), UINT16_C(45489), UINT16_C(42942), UINT16_C(12323), UINT16_C(33906), UINT16_C(32888), UINT16_C(63090),
        UINT16_C(10317), UINT16_C(37077), UINT16_C(49033), UINT16_C(63336), UINT16_C(46236), UINT16_C(42917), UINT16_C(32789), UINT16_C( 6816),
        UINT16_C(65378), UINT16_C( 5276), UINT16_C(23472), UINT16_C(54203), UINT16_C(11915), UINT16_C(  855), UINT16_C(51886), UINT16_C(64505),
        UINT16_C(52978), UINT16_C(31627), UINT16_C(62350), UINT16_C(10866), UINT16_C( 6311), UINT16_C(48593), UINT16_C(29336), UINT16_C(64215) },
      { UINT16_C(29809), UINT16_C( 8718), UINT16_C(51919), UINT16_C(23285), UINT16_C(19960), UINT16_C(42589), UINT16_C(22039), UINT16_C( 2465),
        UINT16_C(11300), UINT16_C(45701), UINT16_C(63264), UINT16_C(51165), UINT16_C(44559), UINT16_C(42884), UINT16_C(23584), UINT16_C(37538),
        UINT16_C(45264), UINT16_C(40884), UINT16_C(43386), UINT16_C(29433), UINT16_C(22262), UINT16_C( 3352), UINT16_C(47532), UINT16_C(53271),
        UINT16_C(40166), UINT16_C( 1667), UINT16_C(24723), UINT16_C(41933), UINT16_C(21006), UINT16_C(12106), UINT16_C(60590), UINT16_C(32449) },
      UINT32_C(         0) },
    { { UINT16_C(30109), UINT16_C( 5917), UINT16_C( 5662), UINT16_C( 5514), UINT16_C(41580), UINT16_C( 6178), UINT16_C(14684), UINT16_C(17128),
        UINT16_C(27605), UINT16_C(26952), UINT16_C( 5579), UINT16_C(55820), UINT16_C(22119), UINT16_C( 5385), UINT16_C(51779), UINT16_C(57491),
        UINT16_C(45119), UINT16_C(24055), UINT16_C(33222), UINT16_C(12914), UINT16_C(38180), UINT16_C(32842), UINT16_C(13262), UINT16_C(42178),
        UINT16_C( 2718), UINT16_C(27149), UINT16_C( 6431), UINT16_C(34628), UINT16_C(19823), UINT16_C(45724), UINT16_C(12311), UINT16_C(22162) },
      { UINT16_C(35552), UINT16_C(42931), UINT16_C( 9739), UINT16_C(12249), UINT16_C( 9403), UINT16_C(35247), UINT16_C(29015), UINT16_C(62765),
        UINT16_C(14971), UINT16_C(39775), UINT16_C(41811), UINT16_C(49954), UINT16_C(48880), UINT16_C( 1909), UINT16_C( 2286), UINT16_C(53085),
        UINT16_C( 4498), UINT16_C(40310), UINT16_C(20279), UINT16_C(62157), UINT16_C(31859), UINT16_C(51835), UINT16_C(43502), UINT16_C(27072),
        UINT16_C( 8163), UINT16_C(14084), UINT16_C( 9923), UINT16_C(46074), UINT16_C(28645), UINT16_C(54203), UINT16_C( 6263), UINT16_C( 2466) },
      UINT32_C(         0) },
    { { UINT16_C( 6185), UINT16_C(24743), UINT16_C(29800), UINT16_C(56146), UINT16_C(52976), UINT16_C(56998), UINT16_C(26231), UINT16_C(23112),
        UINT16_C(19589), UINT16_C(18577), UINT16_C(35699), UINT16_C(22780), UINT16_C(47099), UINT16_C(29227), UINT16_C(52943), UINT16_C(63868),
        UINT16_C( 9190), UINT16_C(20057), UINT16_C(44183), UINT16_C(34602), UINT16_C(53370), UINT16_C(61798), UINT16_C(44598), UINT16_C(47947),
        UINT16_C(56826), UINT16_C(27908), UINT16_C(  104), UINT16_C(25541), UINT16_C(61879), UINT16_C(34518), UINT16_C(21183), UINT16_C(42367) },
      { UINT16_C(55669), UINT16_C( 3316), UINT16_C( 7813), UINT16_C(65427), UINT16_C(63982), UINT16_C( 9456), UINT16_C(15271), UINT16_C(41695),
        UINT16_C(58136), UINT16_C(33039), UINT16_C(54755), UINT16_C(39652), UINT16_C(47814), UINT16_C(34081), UINT16_C(40972), UINT16_C(33066),
        UINT16_C( 7801), UINT16_C(65165), UINT16_C( 8508), UINT16_C(11005), UINT16_C(60698), UINT16_C(49742), UINT16_C(11817), UINT16_C(16740),
        UINT16_C(29457), UINT16_C(62914), UINT16_C(42824), UINT16_C( 3727), UINT16_C(45153), UINT16_C(28307), UINT16_C(48721), UINT16_C(51951) },
      UINT32_C(         0) }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512((void *)&(test_vec[i].arr_a));
    easysimd__m512i b = easysimd_mm512_loadu_si512((void *)&(test_vec[i].arr_b));
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epu16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpeq_epu16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;


#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u16x32();
    easysimd__m512i b = easysimd_test_x86_random_u16x32();
    // easysimd__mmask32 k = easysimd_test_x86_random_mmask8();
    uint32_t r = easysimd_mm512_cmpeq_epu16_mask(a, b);

    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpeq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask64 k1;
    const uint8_t a[64];
    const uint8_t b[64];
    const easysimd__mmask64 r;
  } test_vec[] = {
    { UINT64_C( 3418655210855122011),
      { UINT8_C(230), UINT8_C(170), UINT8_C(105), UINT8_C( 81), UINT8_C( 82), UINT8_C( 92), UINT8_C( 38), UINT8_C( 75),
        UINT8_C( 78), UINT8_C(146), UINT8_C( 61), UINT8_C( 98), UINT8_C(145), UINT8_C( 81), UINT8_C( 38), UINT8_C(190),
        UINT8_C(173), UINT8_C(159), UINT8_C(218), UINT8_C(217), UINT8_C( 43), UINT8_C( 54), UINT8_C(206), UINT8_C(135),
        UINT8_C(250), UINT8_C( 92), UINT8_C(179), UINT8_C(199), UINT8_C(221), UINT8_C( 37), UINT8_C(246), UINT8_C(195),
        UINT8_C(207), UINT8_C( 96), UINT8_C( 20), UINT8_C( 33), UINT8_C(188), UINT8_C( 58), UINT8_C(109), UINT8_C( 10),
        UINT8_C(204), UINT8_C(170), UINT8_C(108), UINT8_C( 94), UINT8_C(251), UINT8_C(146), UINT8_C( 28), UINT8_C(168),
        UINT8_C( 49), UINT8_C(246), UINT8_C(129), UINT8_C( 92), UINT8_C( 45), UINT8_C( 79), UINT8_C(227), UINT8_C( 39),
        UINT8_C(172), UINT8_C(151), UINT8_C(239), UINT8_C(137), UINT8_C(188), UINT8_C(229), UINT8_C( 76), UINT8_C(139) },
      { UINT8_C( 69), UINT8_C(170), UINT8_C(172), UINT8_C(  1), UINT8_C(154), UINT8_C( 92), UINT8_C( 11), UINT8_C( 75),
        UINT8_C(195), UINT8_C(146), UINT8_C(197), UINT8_C( 98), UINT8_C(  9), UINT8_C(225), UINT8_C(102), UINT8_C(190),
        UINT8_C(173), UINT8_C(232), UINT8_C(218), UINT8_C(  5), UINT8_C( 55), UINT8_C(122), UINT8_C( 44), UINT8_C(227),
        UINT8_C( 17), UINT8_C( 27), UINT8_C(179), UINT8_C(205), UINT8_C(  1), UINT8_C(184), UINT8_C( 88), UINT8_C(195),
        UINT8_C(207), UINT8_C( 96), UINT8_C( 20), UINT8_C(179), UINT8_C(188), UINT8_C( 58), UINT8_C( 26), UINT8_C(226),
        UINT8_C(204), UINT8_C(170), UINT8_C(160), UINT8_C( 94), UINT8_C(192), UINT8_C(146), UINT8_C( 28), UINT8_C(168),
        UINT8_C(239), UINT8_C(246), UINT8_C(157), UINT8_C( 38), UINT8_C( 32), UINT8_C( 79), UINT8_C( 10), UINT8_C( 39),
        UINT8_C(229), UINT8_C(118), UINT8_C(239), UINT8_C(137), UINT8_C( 47), UINT8_C(229), UINT8_C( 76), UINT8_C( 71) },
      UINT64_C( 3179682095954165762) },
    { UINT64_C(17603575909869747955),
      { UINT8_C( 29), UINT8_C(234), UINT8_C( 26), UINT8_C( 61), UINT8_C(180), UINT8_C( 36), UINT8_C(111), UINT8_C(153),
        UINT8_C(155), UINT8_C(111), UINT8_C(128), UINT8_C(202), UINT8_C(199), UINT8_C(173), UINT8_C( 17), UINT8_C( 36),
        UINT8_C( 34), UINT8_C( 12), UINT8_C(159), UINT8_C(234), UINT8_C( 32), UINT8_C(253), UINT8_C(126), UINT8_C( 20),
        UINT8_C(251), UINT8_C(230), UINT8_C(200), UINT8_C(  0), UINT8_C( 93), UINT8_C( 20), UINT8_C(244), UINT8_C(122),
        UINT8_C(254), UINT8_C( 14), UINT8_C(183), UINT8_C(178), UINT8_C( 51), UINT8_C( 39), UINT8_C( 76), UINT8_C(206),
        UINT8_C(150), UINT8_C(204), UINT8_C(152), UINT8_C( 93), UINT8_C(121), UINT8_C(169), UINT8_C(129), UINT8_C(155),
        UINT8_C(181), UINT8_C( 32), UINT8_C(133), UINT8_C(214), UINT8_C( 29), UINT8_C(  3), UINT8_C(234), UINT8_C( 24),
        UINT8_C(233), UINT8_C(178), UINT8_C( 24), UINT8_C( 70), UINT8_C(198), UINT8_C( 12), UINT8_C(192), UINT8_C(197) },
      { UINT8_C( 27), UINT8_C(234), UINT8_C(119), UINT8_C( 78), UINT8_C(159), UINT8_C( 36), UINT8_C( 28), UINT8_C(153),
        UINT8_C(143), UINT8_C(111), UINT8_C(146), UINT8_C(  8), UINT8_C(199), UINT8_C( 19), UINT8_C( 17), UINT8_C( 19),
        UINT8_C( 51), UINT8_C( 12), UINT8_C(233), UINT8_C(234), UINT8_C( 44), UINT8_C(211), UINT8_C(126), UINT8_C( 20),
        UINT8_C(251), UINT8_C(230), UINT8_C(200), UINT8_C( 75), UINT8_C( 93), UINT8_C( 29), UINT8_C(244), UINT8_C(169),
        UINT8_C(149), UINT8_C( 14), UINT8_C(183), UINT8_C(178), UINT8_C( 51), UINT8_C( 39), UINT8_C(105), UINT8_C(219),
        UINT8_C(150), UINT8_C(251), UINT8_C(227), UINT8_C( 37), UINT8_C( 14), UINT8_C(169), UINT8_C(129), UINT8_C( 65),
        UINT8_C(176), UINT8_C( 33), UINT8_C(133), UINT8_C(214), UINT8_C( 29), UINT8_C(  3), UINT8_C(234), UINT8_C(121),
        UINT8_C(233), UINT8_C(178), UINT8_C( 24), UINT8_C( 12), UINT8_C(198), UINT8_C( 12), UINT8_C(192), UINT8_C(  1) },
      UINT64_C( 8380179676777697954) },
    { UINT64_C(11450654865767622553),
      { UINT8_C(  6), UINT8_C(123), UINT8_C(122), UINT8_C(250), UINT8_C(119), UINT8_C(109), UINT8_C(115), UINT8_C(245),
        UINT8_C(188), UINT8_C( 56), UINT8_C(  1), UINT8_C( 40), UINT8_C( 13), UINT8_C(183), UINT8_C( 41), UINT8_C(106),
        UINT8_C(101), UINT8_C( 94), UINT8_C( 18), UINT8_C( 38), UINT8_C(252), UINT8_C(150), UINT8_C(174), UINT8_C(149),
        UINT8_C(253), UINT8_C( 92), UINT8_C( 60), UINT8_C(235), UINT8_C( 65), UINT8_C( 36), UINT8_C(137), UINT8_C( 72),
        UINT8_C(159), UINT8_C(  3), UINT8_C( 66), UINT8_C( 22), UINT8_C(112), UINT8_C(182), UINT8_C( 11), UINT8_C( 44),
        UINT8_C(238), UINT8_C( 13), UINT8_C( 84), UINT8_C(251), UINT8_C(196), UINT8_C(125), UINT8_C(101), UINT8_C( 41),
        UINT8_C(219), UINT8_C(119), UINT8_C( 79), UINT8_C(215), UINT8_C( 13), UINT8_C(254), UINT8_C(108), UINT8_C( 10),
        UINT8_C( 90), UINT8_C(168), UINT8_C(245), UINT8_C(155), UINT8_C(205), UINT8_C(126), UINT8_C(227), UINT8_C(108) },
      { UINT8_C(130), UINT8_C(123), UINT8_C(131), UINT8_C(242), UINT8_C(119), UINT8_C(109), UINT8_C( 31), UINT8_C(245),
        UINT8_C(155), UINT8_C(115), UINT8_C(  1), UINT8_C( 40), UINT8_C( 13), UINT8_C( 42), UINT8_C( 41), UINT8_C(204),
        UINT8_C(101), UINT8_C(217), UINT8_C(164), UINT8_C(175), UINT8_C(252), UINT8_C( 16), UINT8_C(185), UINT8_C(149),
        UINT8_C(185), UINT8_C( 92), UINT8_C( 60), UINT8_C(235), UINT8_C( 45), UINT8_C( 36), UINT8_C(242), UINT8_C( 72),
        UINT8_C(159), UINT8_C(  3), UINT8_C(162), UINT8_C( 22), UINT8_C(112), UINT8_C(193), UINT8_C(124), UINT8_C( 44),
        UINT8_C(238), UINT8_C( 65), UINT8_C( 84), UINT8_C( 37), UINT8_C(196), UINT8_C(137), UINT8_C(242), UINT8_C( 12),
        UINT8_C( 98), UINT8_C(119), UINT8_C( 79), UINT8_C(215), UINT8_C(166), UINT8_C(117), UINT8_C(106), UINT8_C( 95),
        UINT8_C( 90), UINT8_C(168), UINT8_C(245), UINT8_C( 81), UINT8_C(230), UINT8_C(126), UINT8_C(227), UINT8_C(188) },
      UINT64_C(  434603457098368144) },
    { UINT64_C(15826687551776682027),
      { UINT8_C( 70), UINT8_C( 94), UINT8_C( 20), UINT8_C(236), UINT8_C(211), UINT8_C(126), UINT8_C( 76), UINT8_C(247),
        UINT8_C(181), UINT8_C( 49), UINT8_C( 73), UINT8_C(155), UINT8_C(  9), UINT8_C( 74), UINT8_C( 88), UINT8_C( 87),
        UINT8_C(237), UINT8_C(198), UINT8_C(168), UINT8_C( 81), UINT8_C(177), UINT8_C(153), UINT8_C(233), UINT8_C(220),
        UINT8_C(138), UINT8_C(167), UINT8_C(115), UINT8_C(  3), UINT8_C( 87), UINT8_C( 22), UINT8_C(223), UINT8_C(157),
        UINT8_C(116), UINT8_C(243), UINT8_C(138), UINT8_C( 72), UINT8_C(114), UINT8_C(214), UINT8_C( 63), UINT8_C( 39),
        UINT8_C(  7), UINT8_C(136), UINT8_C(194), UINT8_C( 17), UINT8_C(210), UINT8_C( 26), UINT8_C(104), UINT8_C(191),
        UINT8_C(225), UINT8_C( 16), UINT8_C( 16), UINT8_C(146), UINT8_C(170), UINT8_C(250), UINT8_C(110), UINT8_C( 52),
        UINT8_C(161), UINT8_C(225), UINT8_C( 55), UINT8_C(249), UINT8_C(247), UINT8_C( 22), UINT8_C(150), UINT8_C(108) },
      { UINT8_C( 70), UINT8_C( 32), UINT8_C(180), UINT8_C(236), UINT8_C(246), UINT8_C(243), UINT8_C(163), UINT8_C(254),
        UINT8_C(181), UINT8_C(101), UINT8_C( 73), UINT8_C( 78), UINT8_C(  9), UINT8_C( 74), UINT8_C( 14), UINT8_C( 87),
        UINT8_C(237), UINT8_C( 30), UINT8_C(243), UINT8_C( 49), UINT8_C( 24), UINT8_C(153), UINT8_C(101), UINT8_C(186),
        UINT8_C(138), UINT8_C(167), UINT8_C(179), UINT8_C(  3), UINT8_C( 87), UINT8_C( 22), UINT8_C(166), UINT8_C(157),
        UINT8_C(106), UINT8_C( 90), UINT8_C( 57), UINT8_C( 72), UINT8_C(114), UINT8_C(220), UINT8_C( 63), UINT8_C( 39),
        UINT8_C( 66), UINT8_C(136), UINT8_C( 24), UINT8_C(194), UINT8_C(228), UINT8_C( 38), UINT8_C(104), UINT8_C(191),
        UINT8_C(225), UINT8_C( 22), UINT8_C(157), UINT8_C( 93), UINT8_C(170), UINT8_C(250), UINT8_C(110), UINT8_C( 52),
        UINT8_C(160), UINT8_C(225), UINT8_C( 55), UINT8_C( 83), UINT8_C(247), UINT8_C(155), UINT8_C( 17), UINT8_C(108) },
      UINT64_C(10565867318684987401) },
    { UINT64_C(10591145043713271466),
      { UINT8_C(100), UINT8_C(152), UINT8_C(239), UINT8_C(219), UINT8_C(155), UINT8_C(  7), UINT8_C(150), UINT8_C( 59),
        UINT8_C(209), UINT8_C(139), UINT8_C(143), UINT8_C(229), UINT8_C( 38), UINT8_C(160), UINT8_C( 99), UINT8_C( 28),
        UINT8_C(234), UINT8_C( 66), UINT8_C( 96), UINT8_C( 17), UINT8_C(127), UINT8_C(110), UINT8_C(122), UINT8_C( 41),
        UINT8_C(149), UINT8_C(165), UINT8_C(184), UINT8_C(226), UINT8_C(243), UINT8_C(179), UINT8_C(116), UINT8_C( 87),
        UINT8_C( 76), UINT8_C( 99), UINT8_C( 51), UINT8_C(231), UINT8_C(106), UINT8_C(201), UINT8_C( 35), UINT8_C( 60),
        UINT8_C( 84), UINT8_C(178), UINT8_C( 33), UINT8_C(122), UINT8_C( 82), UINT8_C(133), UINT8_C(151), UINT8_C( 60),
        UINT8_C(199), UINT8_C(247), UINT8_C( 78), UINT8_C( 70), UINT8_C(102), UINT8_C(200), UINT8_C(111), UINT8_C(251),
        UINT8_C(110), UINT8_C( 40), UINT8_C(221), UINT8_C( 97), UINT8_C(219), UINT8_C( 81), UINT8_C(185), UINT8_C( 39) },
      { UINT8_C(180), UINT8_C(236), UINT8_C(239), UINT8_C( 31), UINT8_C(155), UINT8_C( 50), UINT8_C(150), UINT8_C( 59),
        UINT8_C(228), UINT8_C(139), UINT8_C(143), UINT8_C(229), UINT8_C(  1), UINT8_C(160), UINT8_C( 99), UINT8_C( 28),
        UINT8_C( 18), UINT8_C( 66), UINT8_C( 96), UINT8_C( 17), UINT8_C(137), UINT8_C(110), UINT8_C(115), UINT8_C(247),
        UINT8_C(166), UINT8_C(165), UINT8_C( 88), UINT8_C(226), UINT8_C(161), UINT8_C(179), UINT8_C(116), UINT8_C( 87),
        UINT8_C(253), UINT8_C( 99), UINT8_C( 51), UINT8_C(231), UINT8_C(106), UINT8_C(207), UINT8_C(187), UINT8_C(206),
        UINT8_C( 84), UINT8_C(178), UINT8_C(  4), UINT8_C( 77), UINT8_C( 82), UINT8_C(133), UINT8_C(151), UINT8_C(107),
        UINT8_C( 55), UINT8_C( 36), UINT8_C( 78), UINT8_C( 70), UINT8_C(102), UINT8_C( 86), UINT8_C(183), UINT8_C( 72),
        UINT8_C(166), UINT8_C( 40), UINT8_C(202), UINT8_C( 71), UINT8_C(219), UINT8_C(115), UINT8_C(185), UINT8_C( 39) },
      UINT64_C(10527236750603593344) },
    { UINT64_C(10705215043994838987),
      { UINT8_C(181), UINT8_C(116), UINT8_C( 84), UINT8_C( 87), UINT8_C(202), UINT8_C( 11), UINT8_C(160), UINT8_C(113),
        UINT8_C( 27), UINT8_C(106), UINT8_C(184), UINT8_C( 60), UINT8_C(221), UINT8_C( 85), UINT8_C( 90), UINT8_C(  8),
        UINT8_C(103), UINT8_C( 43), UINT8_C( 29), UINT8_C( 72), UINT8_C(184), UINT8_C(  0), UINT8_C(117), UINT8_C(131),
        UINT8_C(231), UINT8_C(239), UINT8_C(168), UINT8_C( 68), UINT8_C(128), UINT8_C( 57), UINT8_C(217), UINT8_C( 53),
        UINT8_C(173), UINT8_C( 45), UINT8_C(140), UINT8_C(119), UINT8_C( 57), UINT8_C( 44), UINT8_C(232), UINT8_C( 84),
        UINT8_C(150), UINT8_C(161), UINT8_C(144), UINT8_C(115), UINT8_C(246), UINT8_C(234), UINT8_C(123), UINT8_C( 93),
        UINT8_C( 22), UINT8_C(152), UINT8_C(165), UINT8_C(206), UINT8_C(152), UINT8_C( 26), UINT8_C( 81), UINT8_C(127),
        UINT8_C( 10), UINT8_C(250), UINT8_C(196), UINT8_C(138), UINT8_C( 51), UINT8_C(157), UINT8_C(191), UINT8_C(224) },
      { UINT8_C(181), UINT8_C( 75), UINT8_C( 84), UINT8_C(  3), UINT8_C(202), UINT8_C( 11), UINT8_C( 87), UINT8_C(113),
        UINT8_C(225), UINT8_C(231), UINT8_C(184), UINT8_C( 60), UINT8_C(221), UINT8_C( 85), UINT8_C( 53), UINT8_C(  8),
        UINT8_C(103), UINT8_C( 43), UINT8_C( 29), UINT8_C( 46), UINT8_C(245), UINT8_C(  0), UINT8_C(174),    UINT8_MAX,
        UINT8_C(231), UINT8_C(114), UINT8_C(137), UINT8_C( 68), UINT8_C( 15), UINT8_C( 72), UINT8_C( 20), UINT8_C(217),
        UINT8_C(173), UINT8_C( 45), UINT8_C(140), UINT8_C( 11), UINT8_C(172), UINT8_C( 52), UINT8_C( 26), UINT8_C( 84),
        UINT8_C(150), UINT8_C(156), UINT8_C(144), UINT8_C(115), UINT8_C(153), UINT8_C(153), UINT8_C(214), UINT8_C( 47),
        UINT8_C(116), UINT8_C(140), UINT8_C(165), UINT8_C(206), UINT8_C(152), UINT8_C( 26), UINT8_C( 81), UINT8_C(149),
        UINT8_C( 10), UINT8_C(250), UINT8_C(201), UINT8_C(138), UINT8_C( 57), UINT8_C(157), UINT8_C(191), UINT8_C(204) },
      UINT64_C(    4503621121254529) },
    { UINT64_C(17678696174003087246),
      { UINT8_C(228), UINT8_C(181), UINT8_C( 94), UINT8_C(119), UINT8_C(193), UINT8_C(198), UINT8_C( 12), UINT8_C( 63),
        UINT8_C(183), UINT8_C(214), UINT8_C(204), UINT8_C(240), UINT8_C(180), UINT8_C( 50), UINT8_C(188), UINT8_C(254),
        UINT8_C(118), UINT8_C(148), UINT8_C(244), UINT8_C(238), UINT8_C(134), UINT8_C(119), UINT8_C(130), UINT8_C( 20),
        UINT8_C( 94), UINT8_C(  4), UINT8_C( 60), UINT8_C(223), UINT8_C( 92), UINT8_C(147), UINT8_C(212), UINT8_C( 64),
        UINT8_C( 72), UINT8_C( 50), UINT8_C(183), UINT8_C(  9), UINT8_C(248), UINT8_C(196), UINT8_C( 72), UINT8_C(175),
        UINT8_C(154), UINT8_C( 20), UINT8_C(159), UINT8_C( 78), UINT8_C( 70), UINT8_C( 92), UINT8_C( 76), UINT8_C(188),
        UINT8_C(240), UINT8_C( 64), UINT8_C(170), UINT8_C(119), UINT8_C(183), UINT8_C( 44), UINT8_C(139), UINT8_C( 21),
        UINT8_C( 48), UINT8_C(199), UINT8_C(245), UINT8_C(140), UINT8_C( 90), UINT8_C(201), UINT8_C(204), UINT8_C(162) },
      { UINT8_C(228), UINT8_C(181), UINT8_C( 94), UINT8_C(119), UINT8_C( 72), UINT8_C(198), UINT8_C(164), UINT8_C( 63),
        UINT8_C(  7), UINT8_C( 67), UINT8_C(204), UINT8_C( 78), UINT8_C(159), UINT8_C(124), UINT8_C(188), UINT8_C(144),
        UINT8_C(118), UINT8_C(181), UINT8_C(  7), UINT8_C(115), UINT8_C(134), UINT8_C(146), UINT8_C(130), UINT8_C( 20),
        UINT8_C( 94), UINT8_C(125), UINT8_C(158), UINT8_C(223), UINT8_C( 92), UINT8_C(147), UINT8_C( 87), UINT8_C( 64),
        UINT8_C(239), UINT8_C(  2), UINT8_C( 55), UINT8_C(  9), UINT8_C(246), UINT8_C(196), UINT8_C( 72), UINT8_C(253),
        UINT8_C( 31), UINT8_C( 73), UINT8_C( 75), UINT8_C( 78), UINT8_C( 70), UINT8_C( 92), UINT8_C( 78), UINT8_C(188),
        UINT8_C( 11), UINT8_C( 85), UINT8_C(170), UINT8_C(119), UINT8_C(183), UINT8_C( 44), UINT8_C(254), UINT8_C( 21),
        UINT8_C( 48), UINT8_C(157), UINT8_C(246), UINT8_C(140), UINT8_C(  8), UINT8_C( 77), UINT8_C(204), UINT8_C(247) },
      UINT64_C( 4689399500840649870) },
    { UINT64_C(17354273603867414416),
      { UINT8_C( 24), UINT8_C(202), UINT8_C(220), UINT8_C(  0), UINT8_C( 70), UINT8_C(219), UINT8_C( 66), UINT8_C( 64),
        UINT8_C(120), UINT8_C( 57), UINT8_C(129), UINT8_C(128), UINT8_C(134), UINT8_C(  5), UINT8_C(119), UINT8_C(214),
        UINT8_C(193), UINT8_C(165), UINT8_C( 28), UINT8_C( 88), UINT8_C(236), UINT8_C( 96), UINT8_C( 14), UINT8_C(124),
        UINT8_C(239), UINT8_C(130), UINT8_C(209), UINT8_C(212), UINT8_C( 69), UINT8_C(167), UINT8_C(196), UINT8_C( 94),
        UINT8_C(113), UINT8_C(160), UINT8_C( 94), UINT8_C(183), UINT8_C(123), UINT8_C(161), UINT8_C(248), UINT8_C(243),
        UINT8_C(218), UINT8_C(121), UINT8_C(115), UINT8_C( 96), UINT8_C(127), UINT8_C(234), UINT8_C( 55), UINT8_C( 64),
        UINT8_C(143), UINT8_C( 83), UINT8_C(152), UINT8_C(123), UINT8_C(179), UINT8_C(166), UINT8_C(247), UINT8_C(162),
        UINT8_C( 40), UINT8_C(200), UINT8_C(118), UINT8_C(110), UINT8_C(111), UINT8_C( 58), UINT8_C(204), UINT8_C(224) },
      { UINT8_C(219), UINT8_C(202), UINT8_C(220), UINT8_C(  0), UINT8_C( 70), UINT8_C(219), UINT8_C( 66), UINT8_C(165),
        UINT8_C(120), UINT8_C(189), UINT8_C(129), UINT8_C(128), UINT8_C(168), UINT8_C( 61), UINT8_C(119), UINT8_C(214),
        UINT8_C(193), UINT8_C(165), UINT8_C(179), UINT8_C( 88), UINT8_C(  6), UINT8_C( 96), UINT8_C(230), UINT8_C( 47),
        UINT8_C(239), UINT8_C(130), UINT8_C(157), UINT8_C(226), UINT8_C( 69), UINT8_C(105), UINT8_C(195), UINT8_C( 94),
        UINT8_C(147), UINT8_C(160), UINT8_C(201), UINT8_C(183), UINT8_C(123), UINT8_C(161), UINT8_C(  4), UINT8_C(243),
        UINT8_C(218), UINT8_C(121), UINT8_C(115), UINT8_C(120), UINT8_C( 71), UINT8_C(234), UINT8_C(176), UINT8_C(216),
        UINT8_C(143), UINT8_C( 83), UINT8_C(152), UINT8_C(123), UINT8_C(179), UINT8_C(  2), UINT8_C(219), UINT8_C(162),
        UINT8_C( 40), UINT8_C(120), UINT8_C( 99), UINT8_C(110), UINT8_C(225), UINT8_C( 58), UINT8_C(204), UINT8_C(224) },
      UINT64_C(16183126297019452688) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 k = test_vec[i].k1;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epu8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpeq_epu8_mask");

   easysimd_assert_equal_mmask64(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k1 = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_u8x64();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi8(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_u8x64());
    easysimd__mmask64 r = easysimd_mm512_mask_cmpeq_epu8_mask(k1, a, b);

    easysimd_test_x86_write_mmask64(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpeq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const uint16_t a[32];
    const uint16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(2404478353),
      { UINT16_C(33111), UINT16_C(40175), UINT16_C(14339), UINT16_C(44774), UINT16_C(25109), UINT16_C(49057), UINT16_C(52742), UINT16_C( 7891),
        UINT16_C(30749), UINT16_C(31364), UINT16_C(64431), UINT16_C(16419), UINT16_C(29707), UINT16_C(40032), UINT16_C(45537), UINT16_C(14380),
        UINT16_C( 6962), UINT16_C(14037), UINT16_C(47955), UINT16_C(26852), UINT16_C(34077), UINT16_C( 9255), UINT16_C(64339), UINT16_C(28738),
        UINT16_C(51059), UINT16_C( 8938), UINT16_C( 3778), UINT16_C(52835), UINT16_C(50050), UINT16_C(25706), UINT16_C(38516), UINT16_C(42652) },
      { UINT16_C(33111), UINT16_C(40175), UINT16_C(49452), UINT16_C(19054), UINT16_C(38214), UINT16_C(39534), UINT16_C(52742), UINT16_C(  778),
        UINT16_C(62839), UINT16_C(31364), UINT16_C(35075), UINT16_C(34056), UINT16_C(29260), UINT16_C(49385), UINT16_C(45537), UINT16_C(14380),
        UINT16_C(17399), UINT16_C( 9408), UINT16_C(11780), UINT16_C(26852), UINT16_C(56516), UINT16_C(21732), UINT16_C(61324), UINT16_C( 1112),
        UINT16_C(51059), UINT16_C( 8938), UINT16_C( 3778), UINT16_C(52835), UINT16_C(50050), UINT16_C(25706), UINT16_C(31196), UINT16_C(54140) },
      UINT32_C( 251674625) },
    { UINT32_C( 789276011),
      { UINT16_C(61249), UINT16_C(52868), UINT16_C(56542), UINT16_C(49874), UINT16_C( 4186), UINT16_C(25001), UINT16_C( 5718), UINT16_C( 3764),
        UINT16_C(51052), UINT16_C(18640), UINT16_C(19520), UINT16_C(64795), UINT16_C( 5001), UINT16_C(62909), UINT16_C(51320), UINT16_C(47652),
        UINT16_C(43192), UINT16_C(38536), UINT16_C(23172), UINT16_C(56921), UINT16_C(  618), UINT16_C(49215), UINT16_C(62232), UINT16_C(33998),
        UINT16_C(40634), UINT16_C(64460), UINT16_C(59627), UINT16_C(29944), UINT16_C(46587), UINT16_C(29545), UINT16_C(36478), UINT16_C(13869) },
      { UINT16_C(46390), UINT16_C(52868), UINT16_C(56542), UINT16_C(49874), UINT16_C( 4186), UINT16_C(16441), UINT16_C( 5718), UINT16_C( 3764),
        UINT16_C(37286), UINT16_C(37250), UINT16_C(19520), UINT16_C(64795), UINT16_C( 5001), UINT16_C(44520), UINT16_C( 5629), UINT16_C(47652),
        UINT16_C(45259), UINT16_C(38536), UINT16_C(35029), UINT16_C(56921), UINT16_C(  618), UINT16_C(11838), UINT16_C(62232), UINT16_C(33998),
        UINT16_C(14228), UINT16_C( 3789), UINT16_C(54193), UINT16_C(29944), UINT16_C(27203), UINT16_C(16526), UINT16_C(29056), UINT16_C(19316) },
      UINT32_C( 134874186) },
    { UINT32_C(1307867628),
      { UINT16_C(12807), UINT16_C(40059), UINT16_C(12341), UINT16_C(51928), UINT16_C(42599), UINT16_C( 6360), UINT16_C(23161), UINT16_C(48377),
        UINT16_C(34757), UINT16_C(17917), UINT16_C(29176), UINT16_C( 6800), UINT16_C(46549), UINT16_C(49425), UINT16_C( 1327), UINT16_C(13838),
        UINT16_C(35384), UINT16_C(28114), UINT16_C(43962), UINT16_C( 8759), UINT16_C( 3921), UINT16_C(51770), UINT16_C(13162), UINT16_C(12167),
        UINT16_C(33978), UINT16_C(45940), UINT16_C( 1269), UINT16_C(51917), UINT16_C(57017), UINT16_C(59531), UINT16_C(39651), UINT16_C( 6942) },
      { UINT16_C(61732), UINT16_C(40059), UINT16_C(12341), UINT16_C(60672), UINT16_C(42599), UINT16_C( 6360), UINT16_C(23161), UINT16_C(10601),
        UINT16_C(56770), UINT16_C(47324), UINT16_C(29176), UINT16_C(39554), UINT16_C( 3719), UINT16_C(27267), UINT16_C( 1327), UINT16_C(13838),
        UINT16_C(35384), UINT16_C(28114), UINT16_C(43962), UINT16_C( 8759), UINT16_C( 3921), UINT16_C(51770), UINT16_C(13162), UINT16_C(54397),
        UINT16_C(33978), UINT16_C(45940), UINT16_C( 1269), UINT16_C(35227), UINT16_C( 7708), UINT16_C(59531), UINT16_C(31423), UINT16_C( 6942) },
      UINT32_C(  91504740) },
    { UINT32_C(3438845158),
      { UINT16_C(53615), UINT16_C(32800), UINT16_C(40468), UINT16_C(13140), UINT16_C(57591), UINT16_C(64052), UINT16_C(53231), UINT16_C( 2947),
        UINT16_C(30701), UINT16_C(44240), UINT16_C(24817), UINT16_C(31486), UINT16_C(32667), UINT16_C(33235), UINT16_C(51995), UINT16_C(35405),
        UINT16_C(28316), UINT16_C(45066), UINT16_C(24332), UINT16_C(  996), UINT16_C( 6207), UINT16_C(12029), UINT16_C(33255), UINT16_C(54330),
        UINT16_C( 2808), UINT16_C(60032), UINT16_C(32618), UINT16_C( 1636), UINT16_C(14334), UINT16_C( 6535), UINT16_C(54530), UINT16_C(40867) },
      { UINT16_C(44355), UINT16_C(32800), UINT16_C(40468), UINT16_C(19538), UINT16_C(20555), UINT16_C(12922), UINT16_C(46289), UINT16_C(51462),
        UINT16_C(34750), UINT16_C(44240), UINT16_C(24817), UINT16_C( 1071), UINT16_C(46671), UINT16_C(21021), UINT16_C(51995), UINT16_C(35405),
        UINT16_C(28316), UINT16_C(31261), UINT16_C(28788), UINT16_C(49094), UINT16_C( 6207), UINT16_C(12029), UINT16_C(63733), UINT16_C(54330),
        UINT16_C( 3711), UINT16_C(60032), UINT16_C( 2854), UINT16_C(30089), UINT16_C(14334), UINT16_C(19911), UINT16_C(47206), UINT16_C(40867) },
      UINT32_C(2159051782) },
    { UINT32_C(1764496553),
      { UINT16_C( 7764), UINT16_C(18938), UINT16_C(21783), UINT16_C(38653), UINT16_C(55651), UINT16_C(35100), UINT16_C(42469), UINT16_C(43006),
        UINT16_C(50764), UINT16_C(45812), UINT16_C( 4222), UINT16_C(30598), UINT16_C(54346), UINT16_C(62436), UINT16_C( 4584), UINT16_C(15709),
        UINT16_C(22319), UINT16_C(18054), UINT16_C(33708), UINT16_C( 4061), UINT16_C(63837), UINT16_C(17048), UINT16_C(38814), UINT16_C(60137),
        UINT16_C(56669), UINT16_C(56221), UINT16_C( 9198), UINT16_C(14419), UINT16_C(14328), UINT16_C(57387), UINT16_C(34888), UINT16_C(30749) },
      { UINT16_C(42208), UINT16_C(18938), UINT16_C(39719), UINT16_C(38653), UINT16_C(13460), UINT16_C(13254), UINT16_C(42469), UINT16_C(10269),
        UINT16_C(47757), UINT16_C(45812), UINT16_C(22494), UINT16_C(30598), UINT16_C(56974), UINT16_C(55222), UINT16_C( 4584), UINT16_C(18255),
        UINT16_C(22319), UINT16_C(40915), UINT16_C(28585), UINT16_C(15652), UINT16_C(60068), UINT16_C(17048), UINT16_C(38814), UINT16_C(10136),
        UINT16_C(56669), UINT16_C( 9890), UINT16_C(22003), UINT16_C(33276), UINT16_C(45875), UINT16_C(57387), UINT16_C(42887), UINT16_C(30749) },
      UINT32_C( 555745288) },
    { UINT32_C(3365650980),
      { UINT16_C( 3245), UINT16_C(18232), UINT16_C(53402), UINT16_C(57966), UINT16_C( 4204), UINT16_C(24329), UINT16_C( 1381), UINT16_C(39136),
        UINT16_C(14776), UINT16_C(16179), UINT16_C( 5344), UINT16_C(38206), UINT16_C(56777), UINT16_C(61171), UINT16_C(36767), UINT16_C(19638),
        UINT16_C(61083), UINT16_C(13715), UINT16_C(  446), UINT16_C(10775), UINT16_C( 8209), UINT16_C(30345), UINT16_C(27174), UINT16_C(56847),
        UINT16_C(17059), UINT16_C(33566), UINT16_C(23638), UINT16_C( 8217), UINT16_C( 3129), UINT16_C(55566), UINT16_C(50331), UINT16_C(13861) },
      { UINT16_C( 3245), UINT16_C(29035), UINT16_C(53402), UINT16_C(57966), UINT16_C( 4204), UINT16_C(51522), UINT16_C( 1381), UINT16_C(39136),
        UINT16_C(14776), UINT16_C(60086), UINT16_C( 5344), UINT16_C(23562), UINT16_C( 6363), UINT16_C(30517), UINT16_C(36767), UINT16_C(19638),
        UINT16_C(61083), UINT16_C(52737), UINT16_C(40348), UINT16_C(10775), UINT16_C( 8209), UINT16_C(21001), UINT16_C(45358), UINT16_C(56847),
        UINT16_C(17059), UINT16_C(39339), UINT16_C(46345), UINT16_C(58869), UINT16_C(10957), UINT16_C(43612), UINT16_C(50331), UINT16_C(38969) },
      UINT32_C(1083817988) },
    { UINT32_C(2600337623),
      { UINT16_C( 2013), UINT16_C( 3052), UINT16_C(28856), UINT16_C(12236), UINT16_C(30891), UINT16_C(46280), UINT16_C(48685), UINT16_C(64409),
        UINT16_C(62952), UINT16_C(28069), UINT16_C(57087), UINT16_C( 8454), UINT16_C(27673), UINT16_C(61664), UINT16_C(56941), UINT16_C(19082),
        UINT16_C(30437), UINT16_C(40277), UINT16_C( 8679), UINT16_C(37580), UINT16_C(38041), UINT16_C(51014), UINT16_C(57426), UINT16_C(15298),
        UINT16_C(26581), UINT16_C(54440), UINT16_C(44613), UINT16_C(24310), UINT16_C(54811), UINT16_C(34895), UINT16_C(55732), UINT16_C(39378) },
      { UINT16_C(10064), UINT16_C( 3052), UINT16_C(  584), UINT16_C(12236), UINT16_C( 3990), UINT16_C(59817), UINT16_C(27631), UINT16_C(50468),
        UINT16_C(62952), UINT16_C(28069), UINT16_C(36731), UINT16_C(38518), UINT16_C(27673), UINT16_C( 6430), UINT16_C(56941), UINT16_C(19082),
        UINT16_C(30437), UINT16_C(40277), UINT16_C( 8679), UINT16_C(37580), UINT16_C(60158), UINT16_C(60778), UINT16_C(36437), UINT16_C(10162),
        UINT16_C(26581), UINT16_C(54440), UINT16_C(44613), UINT16_C(24310), UINT16_C(54811), UINT16_C( 6234), UINT16_C( 3449), UINT16_C(36871) },
      UINT32_C( 437125122) },
    { UINT32_C( 425799963),
      { UINT16_C(51996), UINT16_C(28934), UINT16_C(47449), UINT16_C(45977), UINT16_C(55301), UINT16_C(57481), UINT16_C(62605), UINT16_C( 1825),
        UINT16_C(31870), UINT16_C(63263), UINT16_C( 9865), UINT16_C(32392), UINT16_C(30803), UINT16_C(28254), UINT16_C(49065), UINT16_C(50567),
        UINT16_C(36234), UINT16_C(58167), UINT16_C(53318), UINT16_C(19351), UINT16_C( 8360), UINT16_C(13612), UINT16_C(19732), UINT16_C(37436),
        UINT16_C(23497), UINT16_C(21130), UINT16_C( 4738), UINT16_C(54737), UINT16_C(12170), UINT16_C(13123), UINT16_C(51951), UINT16_C(31225) },
      { UINT16_C(51996), UINT16_C(40541), UINT16_C(62464), UINT16_C(45977), UINT16_C(55301), UINT16_C(57481), UINT16_C( 6499), UINT16_C( 1825),
        UINT16_C(17780), UINT16_C(63103), UINT16_C(20567), UINT16_C(32392), UINT16_C( 3711), UINT16_C(28180), UINT16_C( 3544), UINT16_C(12520),
        UINT16_C(36234), UINT16_C(15822), UINT16_C(53318), UINT16_C(19941), UINT16_C( 8360), UINT16_C(13612), UINT16_C(12507), UINT16_C(37436),
        UINT16_C(56181), UINT16_C(52294), UINT16_C( 4651), UINT16_C(43949), UINT16_C(49696), UINT16_C(63769), UINT16_C(  463), UINT16_C(31225) },
      UINT32_C(   2162713) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 k = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epu16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpeq_epu16_mask");

   easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_u16x32();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi16(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u16x32());
    easysimd__mmask32 r = easysimd_mm512_mask_cmpeq_epu16_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpeq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const uint32_t a[16];
    const uint32_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C( 8115),
      { UINT32_C(3565287600), UINT32_C(2071057050), UINT32_C(3582340825), UINT32_C( 563971427), UINT32_C(3830102047), UINT32_C(2936452307), UINT32_C(3812316015), UINT32_C(2768433141),
        UINT32_C(2189067495), UINT32_C( 805170006), UINT32_C(2080736281), UINT32_C( 681419273), UINT32_C( 755885914), UINT32_C( 199037851), UINT32_C(1542330470), UINT32_C(4009816071) },
      { UINT32_C(3413211764), UINT32_C(2071057050), UINT32_C(3582340825), UINT32_C(4213479329), UINT32_C(3830102047), UINT32_C(2854553156), UINT32_C( 637866782), UINT32_C(2014643460),
        UINT32_C(2189067495), UINT32_C( 805170006), UINT32_C(2080736281), UINT32_C(1977222902), UINT32_C( 755885914), UINT32_C( 199037851), UINT32_C(1542330470), UINT32_C(2957337392) },
      UINT16_C( 5906) },
    { UINT16_C(56725),
      { UINT32_C(  80017862), UINT32_C(1323542360), UINT32_C(3854810541), UINT32_C(3395310273), UINT32_C(3505554951), UINT32_C(3768336560), UINT32_C(2358305187), UINT32_C( 812263018),
        UINT32_C(1999908383), UINT32_C(2210731989), UINT32_C(2523433173), UINT32_C(3797993691), UINT32_C(4289876814), UINT32_C( 937381779), UINT32_C(2596499504), UINT32_C(3049991574) },
      { UINT32_C(  80017862), UINT32_C(3971281431), UINT32_C(1451367547), UINT32_C( 859366117), UINT32_C(3505554951), UINT32_C(3768336560), UINT32_C(2358305187), UINT32_C( 812263018),
        UINT32_C(3867081423), UINT32_C(2210731989), UINT32_C( 884888655), UINT32_C(3797993691), UINT32_C( 456497889), UINT32_C( 780547500), UINT32_C(2596499504), UINT32_C( 708252763) },
      UINT16_C(18577) },
    { UINT16_C(64273),
      { UINT32_C( 946070504), UINT32_C(1869357368), UINT32_C(4191933464), UINT32_C( 454365551), UINT32_C(1061854022), UINT32_C(2166984742), UINT32_C(3014353060), UINT32_C(4272864278),
        UINT32_C(3627422368), UINT32_C(1262986035), UINT32_C(3863290487), UINT32_C(2046974515), UINT32_C( 465128693), UINT32_C(2174542812), UINT32_C(1496598339), UINT32_C(2757288452) },
      { UINT32_C( 679251700), UINT32_C(1869357368), UINT32_C( 428849381), UINT32_C( 127111442), UINT32_C(3122810077), UINT32_C(1933295151), UINT32_C( 181169926), UINT32_C(4272864278),
        UINT32_C(3832425139), UINT32_C(1262986035), UINT32_C(2901089690), UINT32_C(2046974515), UINT32_C(4198946507), UINT32_C(2607645077), UINT32_C(1101347568), UINT32_C( 294081374) },
      UINT16_C( 2560) },
    { UINT16_C(27382),
      { UINT32_C(1899791319), UINT32_C(1327311008), UINT32_C(1910231462), UINT32_C(1030496680), UINT32_C(2463685026), UINT32_C(1926462739), UINT32_C(1300454352), UINT32_C( 666335568),
        UINT32_C(2644046845), UINT32_C(3270293019), UINT32_C( 791922567), UINT32_C(2322374632), UINT32_C(2350662777), UINT32_C(2449404097), UINT32_C(2615050827), UINT32_C(4173567483) },
      { UINT32_C(1899791319), UINT32_C(2573632017), UINT32_C( 852007498), UINT32_C(2998744377), UINT32_C(2463685026), UINT32_C(1926462739), UINT32_C(3148917695), UINT32_C( 666335568),
        UINT32_C(2644046845), UINT32_C( 377083084), UINT32_C( 791922567), UINT32_C(4032431478), UINT32_C(2821429727), UINT32_C(2449404097), UINT32_C(2615050827), UINT32_C(4173567483) },
      UINT16_C(24752) },
    { UINT16_C( 6798),
      { UINT32_C(1563428847), UINT32_C(3204806985), UINT32_C(1571774590), UINT32_C(3490110201), UINT32_C(1952957394), UINT32_C( 274184240), UINT32_C( 213779645), UINT32_C( 220613918),
        UINT32_C(2641057620), UINT32_C(1314746576), UINT32_C(3400273104), UINT32_C(3113923303), UINT32_C(2771190133), UINT32_C(2008384954), UINT32_C(2810409865), UINT32_C( 364227264) },
      { UINT32_C(3518111745), UINT32_C(3204806985), UINT32_C(  53201692), UINT32_C(3490110201), UINT32_C(2174216647), UINT32_C( 274184240), UINT32_C(2157935808), UINT32_C( 220613918),
        UINT32_C(  83380340), UINT32_C(1936004951), UINT32_C(1618383075), UINT32_C(3113923303), UINT32_C(2325670428), UINT32_C(2008384954), UINT32_C( 947331346), UINT32_C( 364227264) },
      UINT16_C( 2186) },
    { UINT16_C(44269),
      { UINT32_C(1377850223), UINT32_C( 968005347), UINT32_C(3847620041), UINT32_C( 661648624), UINT32_C(2636050827), UINT32_C(2278987026), UINT32_C(4151326114), UINT32_C(4238630541),
        UINT32_C(2488190129), UINT32_C( 600703066), UINT32_C(4127728646), UINT32_C(2753394713), UINT32_C(2084649833), UINT32_C(1929582544), UINT32_C(3664409677), UINT32_C(2211843794) },
      { UINT32_C( 756491474), UINT32_C( 968005347), UINT32_C( 589322506), UINT32_C( 661648624), UINT32_C(2636050827), UINT32_C(1841084960), UINT32_C(4151326114), UINT32_C( 126033205),
        UINT32_C(1697946177), UINT32_C(2324727168), UINT32_C(4127728646), UINT32_C(2753394713), UINT32_C(2084649833), UINT32_C(1929582544), UINT32_C(3664409677), UINT32_C(2211843794) },
      UINT16_C(44104) },
    { UINT16_C( 9492),
      { UINT32_C(1722786952), UINT32_C(1108696148), UINT32_C(1302986448), UINT32_C(4176042142), UINT32_C(1451291588), UINT32_C(2971469891), UINT32_C(3110071059), UINT32_C(1994290158),
        UINT32_C(1826393623), UINT32_C(3148804586), UINT32_C(2382911472), UINT32_C(1955066288), UINT32_C( 684328932), UINT32_C( 601483024), UINT32_C(3252435154), UINT32_C(3275209644) },
      { UINT32_C(1722786952), UINT32_C(1108696148), UINT32_C(1302986448), UINT32_C(3428387816), UINT32_C(1451291588), UINT32_C(2971469891), UINT32_C(3110071059), UINT32_C( 645256669),
        UINT32_C(4015695337), UINT32_C(3101968515), UINT32_C( 664692799), UINT32_C(2264135539), UINT32_C( 684328932), UINT32_C(3137531573), UINT32_C(4000161809), UINT32_C(3275209644) },
      UINT16_C(   20) },
    { UINT16_C( 2894),
      { UINT32_C(4139987895), UINT32_C( 253583771), UINT32_C(1955926361), UINT32_C(2929278457), UINT32_C(1030396972), UINT32_C(2989217596), UINT32_C(1108492474), UINT32_C(1733124271),
        UINT32_C( 794628499), UINT32_C(3409869682), UINT32_C(2235552652), UINT32_C(1010030864), UINT32_C(2977537397), UINT32_C( 795125109), UINT32_C(2507306725), UINT32_C(1794949078) },
      { UINT32_C(1134123472), UINT32_C( 253583771), UINT32_C(3135590058), UINT32_C(2633439527), UINT32_C(1030396972), UINT32_C(2989217596), UINT32_C(4270902567), UINT32_C(1733124271),
        UINT32_C( 794628499), UINT32_C(3409869682), UINT32_C( 591134972), UINT32_C(3485414169), UINT32_C(2977537397), UINT32_C(3870578366), UINT32_C(2507306725), UINT32_C(3054259408) },
      UINT16_C(  770) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epu32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpeq_epu32_mask");

   easysimd_assert_equal_mmask16(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_u32x16();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi32(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u32x16());
    easysimd__mmask16 r = easysimd_mm512_mask_cmpeq_epu32_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpeq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint64_t a[8];
    const uint64_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(152),
      { UINT64_C( 4405257634242682024), UINT64_C(18040030605316602469), UINT64_C(12866607207502795423), UINT64_C( 1851032709146144798),
        UINT64_C(15115396926709419276), UINT64_C( 9593666551390721767), UINT64_C( 5739229909688570590), UINT64_C( 3049525311951072711) },
      { UINT64_C( 4405257634242682024), UINT64_C(13598264882609410379), UINT64_C(12866607207502795423), UINT64_C( 8253451660061603394),
        UINT64_C(18158482570013195011), UINT64_C(16631479337520765237), UINT64_C( 5739229909688570590), UINT64_C( 3049525311951072711) },
      UINT8_C(128) },
    { UINT8_C(240),
      { UINT64_C( 1257921454265330397), UINT64_C( 6514408155692387355), UINT64_C(14267189561981418120), UINT64_C( 8581327177541569528),
        UINT64_C( 1000302636950612458), UINT64_C( 3504467536919319701), UINT64_C(16838406987074679704), UINT64_C( 6658050274664346452) },
      { UINT64_C( 1733056283607349518), UINT64_C( 6514408155692387355), UINT64_C(14267189561981418120), UINT64_C( 8581327177541569528),
        UINT64_C( 1000302636950612458), UINT64_C( 3504467536919319701), UINT64_C(16838406987074679704), UINT64_C( 6658050274664346452) },
      UINT8_C(240) },
    { UINT8_C( 31),
      { UINT64_C( 8159647288088530248), UINT64_C(15388121856257190185), UINT64_C( 2967774273154728537), UINT64_C( 8542532646120167201),
        UINT64_C(11143528839194905913), UINT64_C( 4129679772316092461), UINT64_C(  357935993727565267), UINT64_C( 9934941119050818363) },
      { UINT64_C( 8159647288088530248), UINT64_C(15388121856257190185), UINT64_C( 1749867108877875296), UINT64_C( 5242579997482466281),
        UINT64_C(10039330481948163786), UINT64_C( 4129679772316092461), UINT64_C(  357935993727565267), UINT64_C(13111137308933654769) },
      UINT8_C(  3) },
    { UINT8_C( 57),
      { UINT64_C( 2172018189084818860), UINT64_C(  310474698351078708), UINT64_C(  685721069528189429), UINT64_C(14989600398055469015),
        UINT64_C(16097821830773939007), UINT64_C( 6448037994102873185), UINT64_C(13423219354637809554), UINT64_C( 8087772450722770505) },
      { UINT64_C( 2172018189084818860), UINT64_C( 9455401914995435899), UINT64_C(  685721069528189429), UINT64_C( 6852890635978813049),
        UINT64_C( 7839042854351524712), UINT64_C( 6448037994102873185), UINT64_C(13423219354637809554), UINT64_C( 8087772450722770505) },
      UINT8_C( 33) },
    { UINT8_C(146),
      { UINT64_C(14186120539830029428), UINT64_C(14857193593156196338), UINT64_C(16809743595504024020), UINT64_C( 7866746184515251384),
        UINT64_C(14525105878411746647), UINT64_C(  651376205718149985), UINT64_C( 6991082683812574304), UINT64_C(15610365268834596413) },
      { UINT64_C(14186120539830029428), UINT64_C(14857193593156196338), UINT64_C(16933543045486501622), UINT64_C(16831857414648232765),
        UINT64_C(14525105878411746647), UINT64_C(  651376205718149985), UINT64_C( 6991082683812574304), UINT64_C( 2575018916832561840) },
      UINT8_C( 18) },
    { UINT8_C( 65),
      { UINT64_C( 8130250307711730959), UINT64_C(13737439057630327871), UINT64_C( 1288348740871151981), UINT64_C( 2590954155474415862),
        UINT64_C(15201107324096290800), UINT64_C( 3740637265366586914), UINT64_C( 8373787640436776138), UINT64_C( 8201648110903465371) },
      { UINT64_C( 8908097358902209338), UINT64_C(13737439057630327871), UINT64_C(10079038302661985516), UINT64_C( 6662244559708882564),
        UINT64_C(15495980692003187511), UINT64_C( 3740637265366586914), UINT64_C( 8373787640436776138), UINT64_C(15420099306073133697) },
      UINT8_C( 64) },
    { UINT8_C( 40),
      { UINT64_C(16872202515115996262), UINT64_C( 3954548169270061584), UINT64_C( 9878421265464759597), UINT64_C(17437191789369603542),
        UINT64_C( 1730098069453991245), UINT64_C( 1425272624588817909), UINT64_C( 9295623881385146538), UINT64_C( 7125348914161957990) },
      { UINT64_C(14366621153424459584), UINT64_C( 6051746813887124482), UINT64_C( 9878421265464759597), UINT64_C(17437191789369603542),
        UINT64_C(17825062562500671686), UINT64_C( 1425272624588817909), UINT64_C( 9295623881385146538), UINT64_C( 4876865674453843600) },
      UINT8_C( 40) },
    { UINT8_C(165),
      { UINT64_C( 6913702174132046043), UINT64_C(16722899197375531387), UINT64_C(12593171288611950179), UINT64_C( 6935645150769660397),
        UINT64_C( 2216771174486719716), UINT64_C( 1041760516332125782), UINT64_C(12565208719036764961), UINT64_C( 7619115875475990999) },
      { UINT64_C( 4289273906290513861), UINT64_C(13422766042221414030), UINT64_C(12593171288611950179), UINT64_C( 1721395758695238133),
        UINT64_C(14778004999086518519), UINT64_C( 1041760516332125782), UINT64_C(12565208719036764961), UINT64_C( 7619115875475990999) },
      UINT8_C(164) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epu64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpeq_epu64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u64x8());
    easysimd__mmask8 r = easysimd_mm512_mask_cmpeq_epu64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpeq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask64 k1;
    const int8_t a[64];
    const int8_t b[64];
    const easysimd__mmask64 r;
  } test_vec[] = {
    { UINT64_C( 1772034261841821414),
      { -INT8_C( 110), -INT8_C(  78),  INT8_C(   8), -INT8_C( 120),  INT8_C( 115),  INT8_C( 120), -INT8_C(  38), -INT8_C(   7),
        -INT8_C(  54), -INT8_C( 105), -INT8_C(  64), -INT8_C(  98),  INT8_C(  96), -INT8_C(   6), -INT8_C(  55), -INT8_C(  25),
        -INT8_C( 106),  INT8_C( 111), -INT8_C(  25), -INT8_C(  35), -INT8_C(  99),  INT8_C(  73),  INT8_C(  96), -INT8_C( 125),
        -INT8_C( 121),  INT8_C(  72),  INT8_C(  70),  INT8_C(  77), -INT8_C(  49), -INT8_C(  35),  INT8_C( 101),  INT8_C(  97),
        -INT8_C( 113),  INT8_C( 109), -INT8_C(  23),  INT8_C(   2), -INT8_C(  26), -INT8_C(  61), -INT8_C(   4), -INT8_C(  80),
         INT8_C(  90), -INT8_C(  68),  INT8_C(  78), -INT8_C(  69), -INT8_C(  73),  INT8_C(  23), -INT8_C(  94),  INT8_C(  77),
        -INT8_C( 121), -INT8_C( 119),  INT8_C(  42),  INT8_C(  36), -INT8_C(  46), -INT8_C( 118), -INT8_C(  88),  INT8_C(  90),
        -INT8_C(  46), -INT8_C(  18), -INT8_C(  89), -INT8_C(  95), -INT8_C(  53),  INT8_C(  12),  INT8_C(   2),  INT8_C(  90) },
      { -INT8_C( 110), -INT8_C(  78),  INT8_C(  93),  INT8_C(  96),  INT8_C( 115),  INT8_C( 120),  INT8_C(  16),  INT8_C(  10),
        -INT8_C(  54), -INT8_C( 105), -INT8_C(  59), -INT8_C(  52),  INT8_C(  96),  INT8_C( 103),  INT8_C(  25), -INT8_C(  25),
        -INT8_C( 106),  INT8_C( 111), -INT8_C(  25), -INT8_C(  62), -INT8_C(  50),  INT8_C(  73),  INT8_C(  28), -INT8_C(  96),
        -INT8_C( 121),  INT8_C(  72),  INT8_C(  70),  INT8_C(  77), -INT8_C(  49),  INT8_C(  68), -INT8_C(  35),  INT8_C(  97),
         INT8_C(  48),  INT8_C(  58), -INT8_C(  86),  INT8_C(   2), -INT8_C(  26), -INT8_C(  70), -INT8_C(   4), -INT8_C(  80),
         INT8_C(  90), -INT8_C(  81),  INT8_C(  78), -INT8_C(  69), -INT8_C(  73),  INT8_C(  23), -INT8_C( 115),  INT8_C(   6),
        -INT8_C(  46), -INT8_C( 119),  INT8_C(  42), -INT8_C(  96), -INT8_C(  46), -INT8_C( 118), -INT8_C(  88),  INT8_C(  90),
        -INT8_C(  46), -INT8_C(  18), -INT8_C(  89), -INT8_C(  95), -INT8_C(  53),  INT8_C(  12), -INT8_C(  62),  INT8_C(  90) },
      UINT64_C( 1771609827792327202) },
    { UINT64_C(16180692448418233969),
      {  INT8_C(  13),  INT8_C(  85),      INT8_MIN, -INT8_C( 121),  INT8_C(  58), -INT8_C(  63), -INT8_C(  72), -INT8_C(  29),
         INT8_C(  68),  INT8_C( 109),  INT8_C(  91),  INT8_C(  12), -INT8_C(   1),  INT8_C(  30),  INT8_C(   4), -INT8_C(  53),
        -INT8_C( 118), -INT8_C(  36),  INT8_C(  43), -INT8_C(  79), -INT8_C(  98),  INT8_C(  52), -INT8_C(  14),  INT8_C(  15),
        -INT8_C(  77), -INT8_C(  62), -INT8_C( 106), -INT8_C(  64),  INT8_C(  32),  INT8_C(  35), -INT8_C(  96),  INT8_C(  46),
         INT8_C( 120),  INT8_C(  33), -INT8_C(  75), -INT8_C(  77), -INT8_C(  30),  INT8_C( 109), -INT8_C( 106),  INT8_C(  39),
        -INT8_C(  38), -INT8_C(  15),  INT8_C(  51), -INT8_C(  39),  INT8_C(  15),  INT8_C(  55), -INT8_C(  91), -INT8_C( 102),
         INT8_C(  19), -INT8_C(  48),  INT8_C(  75), -INT8_C(  79),  INT8_C(   5),  INT8_C(  61), -INT8_C(  64), -INT8_C(  72),
         INT8_C(   0),  INT8_C(  86),  INT8_C( 120),  INT8_C(  32),  INT8_C( 121),  INT8_C(  25),  INT8_C(  78), -INT8_C(  15) },
      {  INT8_C(  58),  INT8_C(   3),      INT8_MIN,  INT8_C(  28),  INT8_C( 113), -INT8_C(  63), -INT8_C(  72), -INT8_C(  29),
         INT8_C(  44),  INT8_C( 109),  INT8_C(  91),  INT8_C(  59), -INT8_C(   1), -INT8_C(  54),  INT8_C(   4), -INT8_C(  53),
        -INT8_C( 102),  INT8_C(  33),  INT8_C(  43), -INT8_C(  97), -INT8_C(  98),  INT8_C(  52),  INT8_C(  87),  INT8_C(  15),
        -INT8_C(  77), -INT8_C(  62),      INT8_MAX,  INT8_C(   0),  INT8_C(  32), -INT8_C(  51), -INT8_C(  14),  INT8_C(  35),
         INT8_C( 120),  INT8_C(  33),  INT8_C(  63),  INT8_C(  66), -INT8_C(  30), -INT8_C( 125), -INT8_C( 106), -INT8_C(   3),
        -INT8_C(   7), -INT8_C(  15),  INT8_C(  51), -INT8_C(  89),  INT8_C( 124),  INT8_C(  55),  INT8_C( 103),  INT8_C(  23),
         INT8_C(  47), -INT8_C(  48),  INT8_C(  75), -INT8_C( 115),  INT8_C(   5),  INT8_C(  14), -INT8_C(  20), -INT8_C( 110),
         INT8_C(   0),  INT8_C( 107), -INT8_C( 110),  INT8_C(  32),  INT8_C( 121), -INT8_C( 124),  INT8_C(  78),  INT8_C(   9) },
      UINT64_C( 4612818519758755424) },
    { UINT64_C( 6782337677938991499),
      { -INT8_C( 115), -INT8_C(  43), -INT8_C(  20), -INT8_C( 104), -INT8_C(  29), -INT8_C(  40),  INT8_C(  42), -INT8_C(  63),
         INT8_C(  67), -INT8_C(  68), -INT8_C( 120),  INT8_C( 123),  INT8_C(  65),  INT8_C( 114), -INT8_C( 123),  INT8_C(  92),
        -INT8_C( 100), -INT8_C(  48),  INT8_C(  72),  INT8_C(  72), -INT8_C(  87),  INT8_C(  49), -INT8_C(  18),  INT8_C(  53),
         INT8_C(  82),  INT8_C(  59),  INT8_C(  61), -INT8_C( 126), -INT8_C(  16),  INT8_C(  92), -INT8_C(  32),  INT8_C( 125),
         INT8_C(  49), -INT8_C(  52),  INT8_C(  21),  INT8_C(  21), -INT8_C(  92),  INT8_C(  63), -INT8_C(  42), -INT8_C(  25),
        -INT8_C(   4),  INT8_C(  95),  INT8_C(  99),  INT8_C(  61), -INT8_C(  47), -INT8_C(  24), -INT8_C( 103),  INT8_C( 109),
        -INT8_C(  72), -INT8_C(  31), -INT8_C(  74),  INT8_C(  98),  INT8_C(  18), -INT8_C(  92), -INT8_C( 105),  INT8_C( 100),
        -INT8_C(  32), -INT8_C(  44), -INT8_C(  26), -INT8_C(  48),  INT8_C(  48), -INT8_C(  57),  INT8_C(  77),  INT8_C(  97) },
      { -INT8_C( 109),  INT8_C(  99),  INT8_C( 118),  INT8_C(  56), -INT8_C(  94), -INT8_C(  40),  INT8_C(  42), -INT8_C(  98),
         INT8_C(  67), -INT8_C( 126), -INT8_C(  37),  INT8_C( 125),  INT8_C( 106),  INT8_C( 116), -INT8_C( 123),  INT8_C(  92),
        -INT8_C( 100), -INT8_C(  48), -INT8_C( 123),  INT8_C( 103), -INT8_C(  87),  INT8_C(  49), -INT8_C(  18),  INT8_C(  37),
         INT8_C(  82),  INT8_C(  59), -INT8_C(  11), -INT8_C( 126), -INT8_C(  16),  INT8_C(  67), -INT8_C( 127),  INT8_C(  13),
         INT8_C(  49), -INT8_C(   8),  INT8_C(  21),  INT8_C(  21), -INT8_C(  92),  INT8_C( 100), -INT8_C(  42), -INT8_C(  15),
        -INT8_C(  25), -INT8_C(  62),  INT8_C(  99),  INT8_C(  61),  INT8_C(  55),  INT8_C(  89),  INT8_C( 116),  INT8_C( 109),
        -INT8_C(   6), -INT8_C(  31), -INT8_C(  12),  INT8_C(  98),  INT8_C(  21), -INT8_C(  92),  INT8_C( 101),  INT8_C(   5),
         INT8_C( 114), -INT8_C(  44), -INT8_C(  26), -INT8_C(  20),  INT8_C(  48), -INT8_C(  57),  INT8_C(  77),  INT8_C(  68) },
      UINT64_C( 6199913028536828160) },
    { UINT64_C(10950871506499617590),
      {  INT8_C(  73), -INT8_C(  19), -INT8_C(  41),  INT8_C(  94), -INT8_C(  83),  INT8_C(  61),  INT8_C( 100),  INT8_C(  32),
        -INT8_C( 104), -INT8_C( 119),  INT8_C(  12),  INT8_C(  54),  INT8_C(  48),  INT8_C(   5),  INT8_C( 122), -INT8_C(  49),
         INT8_C(  67),  INT8_C(   6), -INT8_C(  77), -INT8_C(  27),  INT8_C( 122), -INT8_C( 120),  INT8_C( 111), -INT8_C(  80),
        -INT8_C(  52),  INT8_C(  74),  INT8_C(  29),  INT8_C( 105), -INT8_C( 103),  INT8_C(  22),  INT8_C(   0), -INT8_C(  30),
         INT8_C(   4), -INT8_C(  40),  INT8_C(  65), -INT8_C(  79),  INT8_C(  21), -INT8_C(  91), -INT8_C(  47), -INT8_C(  83),
         INT8_C(  46), -INT8_C(  35), -INT8_C(  29),  INT8_C(  95), -INT8_C(  30),  INT8_C(  93),  INT8_C(  46),  INT8_C(  37),
         INT8_C(  99), -INT8_C(  30),  INT8_C(  11), -INT8_C(  35),  INT8_C( 106),  INT8_C( 122), -INT8_C( 115),  INT8_C(  54),
        -INT8_C(  60), -INT8_C(  86), -INT8_C(  97),  INT8_C(  93), -INT8_C(  63), -INT8_C(  96),  INT8_C(  64), -INT8_C(  59) },
      {  INT8_C( 120), -INT8_C( 127), -INT8_C(  41), -INT8_C( 115), -INT8_C(  83),  INT8_C(  72),  INT8_C(  58),  INT8_C(  32),
         INT8_C(  37),  INT8_C(  29),  INT8_C(  12),  INT8_C(   8),  INT8_C(  48), -INT8_C(  30),  INT8_C(  45), -INT8_C(  49),
         INT8_C(  67),  INT8_C(   6), -INT8_C(  69),  INT8_C(  46),  INT8_C( 122),  INT8_C(  72),  INT8_C( 111), -INT8_C(  80),
        -INT8_C(  52),  INT8_C(   4),  INT8_C(  29),  INT8_C( 105), -INT8_C( 103),  INT8_C(  20),  INT8_C(   0),  INT8_C(  28),
        -INT8_C( 107), -INT8_C(  17), -INT8_C(  87), -INT8_C(  69),  INT8_C(  21), -INT8_C(  91),  INT8_C(  15), -INT8_C(  83),
         INT8_C(   0), -INT8_C(  61), -INT8_C(  29),  INT8_C( 122), -INT8_C(  91), -INT8_C( 110),  INT8_C(  46),  INT8_C(  37),
        -INT8_C(  53),  INT8_C(  19), -INT8_C( 105),  INT8_C( 125),  INT8_C(  91), -INT8_C(   4), -INT8_C(  12),  INT8_C(  78),
         INT8_C(   1), -INT8_C(  56),  INT8_C(   2), -INT8_C(  91), -INT8_C(  63), -INT8_C(  96), -INT8_C(  62), -INT8_C(  59) },
      UINT64_C(10376368928033275924) },
    { UINT64_C(13982589390078764286),
      {  INT8_C(  53), -INT8_C(  92),  INT8_C(  63), -INT8_C( 112), -INT8_C(  96),  INT8_C(  51), -INT8_C(  33), -INT8_C(  95),
        -INT8_C(   5), -INT8_C(  31),  INT8_C(  71), -INT8_C(  41),  INT8_C(  93),  INT8_C(   9),  INT8_C(  72), -INT8_C(  56),
         INT8_C( 116),  INT8_C( 116),  INT8_C( 106), -INT8_C(  61), -INT8_C(  80),  INT8_C( 106),  INT8_C(  19), -INT8_C(  82),
        -INT8_C(  50), -INT8_C(  35),  INT8_C(  82), -INT8_C(  59), -INT8_C(   1),  INT8_C(  94), -INT8_C( 121),  INT8_C(  52),
         INT8_C(   2), -INT8_C(  57), -INT8_C(  60), -INT8_C(  93), -INT8_C(   6), -INT8_C(  93),  INT8_C(  68), -INT8_C(  10),
        -INT8_C( 123), -INT8_C( 117), -INT8_C(  51), -INT8_C(  30), -INT8_C( 108),  INT8_C(  22), -INT8_C(  86),  INT8_C(   9),
        -INT8_C( 118),  INT8_C(  20), -INT8_C(  52),  INT8_C(  58),  INT8_C( 126), -INT8_C(  33), -INT8_C(  23),  INT8_C(  77),
        -INT8_C(  68),  INT8_C(  59),  INT8_C(  18), -INT8_C(  69), -INT8_C( 103), -INT8_C( 102), -INT8_C(  17), -INT8_C( 100) },
      {  INT8_C(  97), -INT8_C(  76),  INT8_C(  63),  INT8_C(  91), -INT8_C(  96), -INT8_C( 125), -INT8_C(  33), -INT8_C(  36),
         INT8_C(  15),  INT8_C(  31), -INT8_C(  66), -INT8_C(  41),  INT8_C(  93),  INT8_C( 104), -INT8_C(  84), -INT8_C(  65),
         INT8_C( 125),  INT8_C( 121),  INT8_C( 106), -INT8_C(   5), -INT8_C(  80), -INT8_C(  29),  INT8_C(  19), -INT8_C(  82),
         INT8_C(  30), -INT8_C(  35), -INT8_C(  48), -INT8_C(  59), -INT8_C(   1),  INT8_C(  94),  INT8_C(  83),  INT8_C(  86),
         INT8_C(   2), -INT8_C( 110), -INT8_C(  60), -INT8_C(  53), -INT8_C(   6),  INT8_C(   3), -INT8_C(  88),  INT8_C(  37),
         INT8_C(  34),  INT8_C( 102), -INT8_C(  51), -INT8_C(  30), -INT8_C(  49),  INT8_C(  22),  INT8_C(  22),  INT8_C(  76),
        -INT8_C( 118),  INT8_C(  16), -INT8_C(  52),  INT8_C(  70),  INT8_C( 126), -INT8_C( 112),  INT8_C(  91),  INT8_C(  17),
        -INT8_C(  68),  INT8_C(  59), -INT8_C(  55), -INT8_C(  32), -INT8_C( 103), -INT8_C( 102), -INT8_C(  17), -INT8_C( 100) },
      UINT64_C(13980334418414272596) },
    { UINT64_C( 1537269155896734266),
      { -INT8_C( 118), -INT8_C( 100),  INT8_C(  92),  INT8_C( 126),  INT8_C(  44), -INT8_C(  73), -INT8_C( 113),  INT8_C(  23),
        -INT8_C(  29),  INT8_C(  88), -INT8_C(   9), -INT8_C(  49),  INT8_C( 117),  INT8_C(  45),  INT8_C(  47),  INT8_C(  36),
         INT8_C(  21),  INT8_C(  91), -INT8_C(  23), -INT8_C(   1),  INT8_C(  46), -INT8_C(  45),  INT8_C(  12),  INT8_C( 104),
        -INT8_C( 123),  INT8_C( 111),  INT8_C( 113), -INT8_C(  83), -INT8_C(  23), -INT8_C(  58), -INT8_C(  62),  INT8_C( 116),
         INT8_C(  99),  INT8_C(  30), -INT8_C(  14), -INT8_C( 113), -INT8_C(  42), -INT8_C( 127), -INT8_C(  89), -INT8_C(  71),
        -INT8_C(  38), -INT8_C(  98), -INT8_C( 119),  INT8_C(  79), -INT8_C(  52), -INT8_C(  72),  INT8_C( 115), -INT8_C(  31),
         INT8_C(  19),  INT8_C(  92), -INT8_C(  32),  INT8_C(  66),  INT8_C(  47), -INT8_C(  20), -INT8_C(  86), -INT8_C(  76),
         INT8_C(  92),  INT8_C(  28),  INT8_C(  97),  INT8_C(  69), -INT8_C(  30),  INT8_C(  36), -INT8_C(  71),  INT8_C(  69) },
      { -INT8_C( 118), -INT8_C(  85),  INT8_C(  92),  INT8_C( 126),  INT8_C(  45), -INT8_C(  73), -INT8_C( 113),  INT8_C(   7),
         INT8_C(  26),  INT8_C(  88), -INT8_C(   9), -INT8_C(  49),  INT8_C( 117), -INT8_C(  55),  INT8_C(  47),  INT8_C(  36),
         INT8_C(  21), -INT8_C(  88),  INT8_C( 105), -INT8_C(   1),  INT8_C(  46),  INT8_C(  19),  INT8_C(  12),  INT8_C( 104),
         INT8_C(  47),  INT8_C( 106),  INT8_C( 113),  INT8_C(  18), -INT8_C( 114), -INT8_C(  17), -INT8_C(  62),  INT8_C( 116),
         INT8_C(  99),  INT8_C(  30), -INT8_C(  23), -INT8_C(  56), -INT8_C(  88), -INT8_C( 127), -INT8_C(  49), -INT8_C(  61),
         INT8_C(  22), -INT8_C(  98), -INT8_C(  87),  INT8_C(  79), -INT8_C(  18),  INT8_C( 113),  INT8_C(  80),  INT8_C(  19),
         INT8_C(  19), -INT8_C(  71),  INT8_C( 103), -INT8_C(  83), -INT8_C(  51),  INT8_C( 111), -INT8_C(  98), -INT8_C(   4),
         INT8_C(  92), -INT8_C(  44),  INT8_C(  97),  INT8_C(  69), -INT8_C(  61),  INT8_C( 102),  INT8_C(  56),  INT8_C(  94) },
      UINT64_C(  360580590610780712) },
    { UINT64_C( 3538540782700701466),
      {  INT8_C(  37), -INT8_C( 126), -INT8_C(  33), -INT8_C(  14), -INT8_C(  14),  INT8_C( 125), -INT8_C(  17), -INT8_C(  53),
         INT8_C(  81), -INT8_C(   3),  INT8_C(  51),  INT8_C(  20),  INT8_C(  99),  INT8_C( 107),  INT8_C( 115), -INT8_C(  10),
        -INT8_C( 116), -INT8_C( 103),  INT8_C(  49),  INT8_C( 104), -INT8_C( 113),  INT8_C(  47),  INT8_C(  90), -INT8_C(  87),
        -INT8_C(  42),  INT8_C( 117), -INT8_C(  78), -INT8_C(  17), -INT8_C(  31), -INT8_C(  51),  INT8_C(  32),  INT8_C(   7),
         INT8_C(  80), -INT8_C(   1), -INT8_C(   7),  INT8_C(  66),  INT8_C( 124), -INT8_C(  24),  INT8_C(  13), -INT8_C(  51),
        -INT8_C(  26),  INT8_C(  64), -INT8_C(  30),  INT8_C(  73), -INT8_C(  85),  INT8_C(  85),  INT8_C(  63),  INT8_C(  55),
        -INT8_C(  18),  INT8_C( 112), -INT8_C(  97),  INT8_C( 125), -INT8_C(  97), -INT8_C(   7),  INT8_C(  39),  INT8_C( 118),
         INT8_C( 111), -INT8_C(  39),  INT8_C( 101),  INT8_C(  80), -INT8_C(  90), -INT8_C( 123),  INT8_C(  87), -INT8_C(  10) },
      {  INT8_C(  37),  INT8_C(  81),  INT8_C(  56), -INT8_C(  14), -INT8_C(  14),  INT8_C( 125), -INT8_C(  17), -INT8_C(  53),
        -INT8_C( 122), -INT8_C(  79),  INT8_C( 105),  INT8_C(  50),  INT8_C(   6), -INT8_C(  88),  INT8_C( 115), -INT8_C(  12),
        -INT8_C( 116),  INT8_C(   9),  INT8_C( 114),  INT8_C( 104),  INT8_C(   2),  INT8_C(  47),  INT8_C(  90), -INT8_C(  87),
         INT8_C( 114), -INT8_C( 109), -INT8_C(  78), -INT8_C(  17),  INT8_C(  25), -INT8_C(  51),  INT8_C(  32), -INT8_C(  98),
         INT8_C( 106), -INT8_C(   1), -INT8_C(  97), -INT8_C(  92),  INT8_C( 124),  INT8_C( 110),  INT8_C(  13), -INT8_C(  51),
        -INT8_C(  26),  INT8_C(  44), -INT8_C(  30),  INT8_C(  37), -INT8_C(  43),  INT8_C(  85),  INT8_C(  26), -INT8_C(  18),
        -INT8_C(  72), -INT8_C( 116), -INT8_C(  90),  INT8_C( 125), -INT8_C(  97), -INT8_C(  43),  INT8_C(  39), -INT8_C( 105),
         INT8_C( 104), -INT8_C(  39), -INT8_C(  81), -INT8_C( 127), -INT8_C(  90), -INT8_C( 123),  INT8_C(  31), -INT8_C(  10) },
      UINT64_C( 3465559564534480920) },
    { UINT64_C( 6181626573162212614),
      {  INT8_C(  24),  INT8_C( 112),  INT8_C(  16),  INT8_C(  61),  INT8_C(  69),  INT8_C(  61), -INT8_C(  44), -INT8_C(  83),
         INT8_C(  43), -INT8_C( 124),  INT8_C(  47),  INT8_C(  51),  INT8_C(  66),  INT8_C(  78), -INT8_C(  90),  INT8_C(  72),
         INT8_C(  13), -INT8_C(  68), -INT8_C(  36),  INT8_C(  59), -INT8_C( 106), -INT8_C( 125), -INT8_C( 120), -INT8_C(  99),
         INT8_C( 113), -INT8_C(   6),  INT8_C( 120),  INT8_C(  14), -INT8_C( 121),  INT8_C(  66),  INT8_C(  99), -INT8_C(  97),
        -INT8_C(  78),  INT8_C( 116), -INT8_C(  35), -INT8_C(   9), -INT8_C(  79), -INT8_C(  79), -INT8_C(  92), -INT8_C(  36),
         INT8_C(  53), -INT8_C(  45),  INT8_C(  16),  INT8_C( 120),  INT8_C(  34), -INT8_C(  74), -INT8_C(  64),  INT8_C(  47),
         INT8_C( 114), -INT8_C( 100),  INT8_C( 106),  INT8_C(   9),  INT8_C(  32), -INT8_C(  14), -INT8_C(  90), -INT8_C( 111),
        -INT8_C(  19),  INT8_C(  30), -INT8_C(  97),  INT8_C( 116),  INT8_C(  96),  INT8_C(   2),  INT8_C(  19),  INT8_C(  18) },
      {  INT8_C(  24),  INT8_C( 112),  INT8_C(   9),  INT8_C(  61),  INT8_C(  69), -INT8_C(  82),  INT8_C(   4), -INT8_C(  83),
         INT8_C(  43),  INT8_C(  20),  INT8_C(  47),  INT8_C(  51), -INT8_C(  54),  INT8_C(  78), -INT8_C(  45),  INT8_C(  72),
         INT8_C(  13), -INT8_C(  68), -INT8_C(  36), -INT8_C(  52),  INT8_C(  48), -INT8_C(  21), -INT8_C( 120),  INT8_C(  29),
         INT8_C(  10), -INT8_C(   6),  INT8_C( 120),  INT8_C( 106), -INT8_C(   1),  INT8_C(  66),  INT8_C(  99),  INT8_C( 117),
        -INT8_C( 107), -INT8_C( 122), -INT8_C(  35), -INT8_C(   9),  INT8_C(  52), -INT8_C(  95),  INT8_C(  14), -INT8_C(  74),
        -INT8_C(  75),  INT8_C(  94),  INT8_C(  89),  INT8_C( 120),  INT8_C(  34), -INT8_C(  74), -INT8_C(  69),  INT8_C(  26),
         INT8_C( 106),  INT8_C(   1), -INT8_C(  25), -INT8_C( 102),  INT8_C(  32), -INT8_C(  14), -INT8_C(  73), -INT8_C( 111),
        -INT8_C(  19),  INT8_C(  30), -INT8_C(  97),  INT8_C(  64),  INT8_C(  96), -INT8_C(  34),  INT8_C(  19), -INT8_C( 127) },
      UINT64_C( 6160933138987134210) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__mmask64 k = test_vec[i].k1;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_cmpeq_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k1 = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi8(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_i8x64());
    easysimd__mmask64 r = easysimd_mm512_mask_cmpeq_epi8_mask(k1, a, b);

    easysimd_test_x86_write_mmask64(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpeq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const int16_t a[32];
    const int16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(3485694314),
      { -INT16_C( 11913),  INT16_C( 26173),  INT16_C( 27931),  INT16_C( 10126),  INT16_C( 21525),  INT16_C( 21314),  INT16_C( 17436),  INT16_C( 13906),
        -INT16_C( 24483), -INT16_C( 21415), -INT16_C( 10991), -INT16_C( 10785),  INT16_C(  7397),  INT16_C( 20242), -INT16_C( 10859),  INT16_C(  3358),
         INT16_C( 23463), -INT16_C( 15757),  INT16_C(   712), -INT16_C(  8727),  INT16_C( 11094),  INT16_C( 29233), -INT16_C( 31889), -INT16_C( 12887),
         INT16_C(   547),  INT16_C( 13689),  INT16_C( 22743), -INT16_C( 17398),  INT16_C(  7541),  INT16_C(  2572),  INT16_C( 10994), -INT16_C( 26345) },
      { -INT16_C( 11913),  INT16_C( 20059),  INT16_C( 17549), -INT16_C(  7380),  INT16_C( 21525), -INT16_C(  8362), -INT16_C(    32),  INT16_C( 13906),
        -INT16_C( 24483), -INT16_C( 10184), -INT16_C( 10991), -INT16_C( 10785), -INT16_C( 24224),  INT16_C( 21245), -INT16_C( 10859),  INT16_C(  3358),
         INT16_C( 23463), -INT16_C( 15757), -INT16_C( 13172), -INT16_C(  1264),  INT16_C( 26153),  INT16_C(  2522), -INT16_C( 31131),  INT16_C( 26124),
         INT16_C(   547),  INT16_C( 10815),  INT16_C( 22743), -INT16_C(  6115),  INT16_C(  6773),  INT16_C( 16442),  INT16_C(  9775), -INT16_C( 12398) },
      UINT32_C(  84101376) },
    { UINT32_C( 670371326),
      { -INT16_C( 12173), -INT16_C(  9936),  INT16_C( 15446),  INT16_C(   575),  INT16_C( 32385),  INT16_C(  2348),  INT16_C( 18770), -INT16_C( 14351),
         INT16_C( 11364), -INT16_C( 27896), -INT16_C( 26030), -INT16_C( 16285),  INT16_C( 24524), -INT16_C( 13638), -INT16_C( 20372), -INT16_C(  7951),
         INT16_C(  8576), -INT16_C( 10567), -INT16_C(  1955), -INT16_C(  8231),  INT16_C(  1399), -INT16_C( 13848), -INT16_C(  9649), -INT16_C( 19567),
        -INT16_C( 26362),  INT16_C( 22598), -INT16_C( 22221), -INT16_C(   231), -INT16_C( 11511),  INT16_C( 30153), -INT16_C( 17789),  INT16_C(   853) },
      {  INT16_C(  3803), -INT16_C(  9936), -INT16_C( 19705),  INT16_C(   575),  INT16_C(   184),  INT16_C(  2348),  INT16_C( 18770), -INT16_C(  8006),
         INT16_C( 11364), -INT16_C( 23496), -INT16_C( 26030), -INT16_C( 19549),  INT16_C( 27685), -INT16_C( 22487), -INT16_C( 20372),  INT16_C(   428),
        -INT16_C( 31091), -INT16_C( 27590), -INT16_C(  1955), -INT16_C(  3822),  INT16_C( 22865), -INT16_C( 13848), -INT16_C( 19662), -INT16_C( 23797),
        -INT16_C( 26362),  INT16_C( 24392), -INT16_C(  5227), -INT16_C( 17902), -INT16_C( 11511),  INT16_C( 30153),  INT16_C(  4026),  INT16_C( 18304) },
      UINT32_C( 556008810) },
    { UINT32_C(1572859147),
      { -INT16_C( 18362),  INT16_C( 30856), -INT16_C( 27540),  INT16_C(  8220),  INT16_C( 25816),  INT16_C( 28031), -INT16_C( 28081), -INT16_C( 22744),
        -INT16_C( 29747), -INT16_C( 30938), -INT16_C( 22886),  INT16_C( 12238), -INT16_C( 22176),  INT16_C( 27645), -INT16_C( 17258), -INT16_C(  8760),
         INT16_C( 20853), -INT16_C(  7851),  INT16_C( 29157), -INT16_C( 17151), -INT16_C( 32299),  INT16_C(  9514),  INT16_C( 21011), -INT16_C(  7988),
        -INT16_C(  3363),  INT16_C( 30568),  INT16_C( 13976), -INT16_C(  1882), -INT16_C( 23584),  INT16_C( 30308),  INT16_C( 11360), -INT16_C( 10925) },
      { -INT16_C( 22147),  INT16_C( 30856), -INT16_C( 18662), -INT16_C(  4065),  INT16_C( 25816),  INT16_C( 28031), -INT16_C(  7780), -INT16_C( 22744),
        -INT16_C( 27436),  INT16_C( 27889), -INT16_C( 22886), -INT16_C( 21915), -INT16_C( 22176),  INT16_C( 27645),  INT16_C( 29941),  INT16_C( 29552),
         INT16_C( 20853),  INT16_C( 14549),  INT16_C( 29157), -INT16_C( 17151), -INT16_C( 32299),  INT16_C(  9514),  INT16_C( 21011), -INT16_C(  3499),
         INT16_C( 18210), -INT16_C(  5025),  INT16_C( 13976),  INT16_C(  6807), -INT16_C( 23584), -INT16_C( 32074),  INT16_C( 11360), -INT16_C( 10925) },
      UINT32_C(1413293058) },
    { UINT32_C(4282493632),
      { -INT16_C( 23577),  INT16_C(  1498),  INT16_C( 12337),  INT16_C( 21496),  INT16_C( 22391),  INT16_C( 22080), -INT16_C( 10469), -INT16_C( 22416),
         INT16_C(  9871), -INT16_C( 17622),  INT16_C(  8269), -INT16_C( 26107), -INT16_C( 30741), -INT16_C( 21563),  INT16_C(  1585),  INT16_C(  6314),
        -INT16_C( 31574), -INT16_C(  9442),  INT16_C(  5812),  INT16_C( 11055),  INT16_C( 28525), -INT16_C( 30591), -INT16_C(  3514), -INT16_C( 10960),
         INT16_C( 23064),  INT16_C( 26000), -INT16_C( 27014),  INT16_C( 26111), -INT16_C( 15331),  INT16_C( 20240), -INT16_C( 17717),  INT16_C( 30055) },
      { -INT16_C( 31425), -INT16_C(  3248),  INT16_C( 12337),  INT16_C( 21496),  INT16_C( 22391),  INT16_C( 22080), -INT16_C( 10469), -INT16_C( 22416),
        -INT16_C( 26085), -INT16_C( 17622),  INT16_C(  8269),  INT16_C( 19963),  INT16_C(  3028), -INT16_C( 24676),  INT16_C(  1222),  INT16_C(  1300),
         INT16_C( 25993),  INT16_C(  9720),  INT16_C(  6116),  INT16_C( 11055), -INT16_C( 16712),  INT16_C( 18951),  INT16_C(  4478), -INT16_C( 10960),
         INT16_C( 23064),  INT16_C( 26000), -INT16_C( 27014), -INT16_C(  5592), -INT16_C( 15331), -INT16_C(  1142), -INT16_C( 24887),  INT16_C( 20992) },
      UINT32_C( 385876672) },
    { UINT32_C(3367740688),
      { -INT16_C( 15773), -INT16_C(  7917),  INT16_C(  2259),  INT16_C( 32379), -INT16_C( 22002),  INT16_C(  9305), -INT16_C( 32044),  INT16_C(  2319),
        -INT16_C( 26297),  INT16_C(  4101),  INT16_C(  1335),  INT16_C( 15202), -INT16_C(  9474),  INT16_C(  3875), -INT16_C(  8577), -INT16_C(  7465),
        -INT16_C(  5472),  INT16_C( 29891),  INT16_C( 16115),  INT16_C(   498),  INT16_C( 19688), -INT16_C( 17370),  INT16_C( 13774),  INT16_C(  5574),
        -INT16_C( 13362),  INT16_C(  1317), -INT16_C( 30768), -INT16_C( 12480),  INT16_C( 25441), -INT16_C(  7970), -INT16_C( 19135), -INT16_C(  7486) },
      { -INT16_C( 31072), -INT16_C(  7917),  INT16_C( 18628), -INT16_C( 21100), -INT16_C( 17772),  INT16_C(  9305), -INT16_C( 32044), -INT16_C( 17033),
        -INT16_C( 26297),  INT16_C(  4101),  INT16_C(   804), -INT16_C( 31334), -INT16_C(  9474), -INT16_C( 22426),  INT16_C( 10285), -INT16_C( 12918),
        -INT16_C(  8018),  INT16_C( 29891), -INT16_C(  2775), -INT16_C( 17120),  INT16_C( 19688), -INT16_C( 17370),  INT16_C( 13774), -INT16_C( 19620),
        -INT16_C( 13362),  INT16_C( 22654),  INT16_C(  6179), -INT16_C( 12480),  INT16_C( 25441), -INT16_C(  7970), -INT16_C( 17044),  INT16_C(  6795) },
      UINT32_C( 137494784) },
    { UINT32_C(2424614369),
      { -INT16_C( 23497), -INT16_C(  4049), -INT16_C( 29637),  INT16_C( 28579),  INT16_C(  8875), -INT16_C( 12601), -INT16_C( 23238), -INT16_C( 13480),
        -INT16_C( 29720),  INT16_C( 21641),  INT16_C(  5192), -INT16_C(  6801), -INT16_C(  1024), -INT16_C(  7764),  INT16_C( 12458), -INT16_C(  7822),
        -INT16_C( 24108),  INT16_C(  4049),  INT16_C( 29741), -INT16_C(  9857),  INT16_C( 18070), -INT16_C( 11865),  INT16_C(   235), -INT16_C( 11108),
         INT16_C(  9611), -INT16_C( 11480), -INT16_C( 26823),  INT16_C( 15032),  INT16_C( 25748),  INT16_C( 15899), -INT16_C( 29292),  INT16_C( 26655) },
      { -INT16_C(  4049), -INT16_C(  4049), -INT16_C( 29637),  INT16_C( 28579),  INT16_C(  8875),  INT16_C( 10700), -INT16_C( 23238), -INT16_C( 13480),
        -INT16_C( 29720),  INT16_C( 21641),  INT16_C(  5192),  INT16_C( 20736),  INT16_C(  7256), -INT16_C(  7764), -INT16_C( 20823), -INT16_C(  7822),
        -INT16_C( 24108),  INT16_C(  4049),  INT16_C( 29741),  INT16_C(   509), -INT16_C( 14009),  INT16_C(  9258),  INT16_C(   235), -INT16_C( 11108),
        -INT16_C( 14515), -INT16_C( 11480), -INT16_C( 31301),  INT16_C( 15032),  INT16_C( 25748),  INT16_C( 19455),  INT16_C( 21656),  INT16_C( 13859) },
      UINT32_C( 277128640) },
    { UINT32_C( 182924995),
      {  INT16_C(  4607),  INT16_C( 12591), -INT16_C( 17607), -INT16_C( 30993),  INT16_C( 29827),  INT16_C( 16016), -INT16_C(  5126), -INT16_C( 25775),
         INT16_C( 20693),  INT16_C( 28134),  INT16_C(  2725), -INT16_C( 14685), -INT16_C(  9374),  INT16_C(  9644), -INT16_C( 27887),  INT16_C(  4400),
         INT16_C( 24484), -INT16_C(  8894),  INT16_C( 12570), -INT16_C( 25245), -INT16_C(  3162), -INT16_C( 24357),  INT16_C( 11486), -INT16_C( 19653),
         INT16_C(  8829),  INT16_C(  8736), -INT16_C( 15572), -INT16_C( 28952), -INT16_C( 27489), -INT16_C( 20300), -INT16_C(  7129), -INT16_C( 13119) },
      {  INT16_C(  4607),  INT16_C( 12591), -INT16_C( 17607), -INT16_C( 30993),  INT16_C( 29827),  INT16_C( 16016), -INT16_C(  5126), -INT16_C( 32622),
         INT16_C( 20693),  INT16_C( 28134),  INT16_C(  2725),  INT16_C(  5523), -INT16_C(  9374),  INT16_C( 18118), -INT16_C( 30933),  INT16_C( 28178),
         INT16_C( 24484), -INT16_C( 16181),  INT16_C( 12570), -INT16_C( 25245),  INT16_C(  5789), -INT16_C( 24357),  INT16_C( 11486), -INT16_C( 19653),
        -INT16_C( 15634),  INT16_C(  8736), -INT16_C( 15572),  INT16_C( 27514), -INT16_C( 27489), -INT16_C( 20300), -INT16_C( 15161), -INT16_C( 13119) },
      UINT32_C(  48567875) },
    { UINT32_C(1259580846),
      { -INT16_C( 17469), -INT16_C( 28437),  INT16_C(  3062), -INT16_C(  7115), -INT16_C(  8499),  INT16_C(  6729), -INT16_C( 15589), -INT16_C( 24955),
         INT16_C( 14083), -INT16_C( 13747),  INT16_C( 27132),  INT16_C( 31773),  INT16_C( 12113),  INT16_C(   198), -INT16_C(  9763), -INT16_C( 24501),
         INT16_C( 14228), -INT16_C( 29904),  INT16_C( 25922),  INT16_C(  4207), -INT16_C( 18364),  INT16_C( 24362), -INT16_C( 20613),  INT16_C( 32509),
         INT16_C( 19175), -INT16_C(  7351),  INT16_C( 26292),  INT16_C(  1375),  INT16_C(  9621),  INT16_C( 29189),  INT16_C( 20990), -INT16_C( 27885) },
      {  INT16_C( 17288), -INT16_C( 13794),  INT16_C(  3062), -INT16_C(  4646), -INT16_C(  8499),  INT16_C(  6729),  INT16_C( 18868), -INT16_C( 25792),
         INT16_C( 14083), -INT16_C( 13747), -INT16_C(  8721), -INT16_C( 31667),  INT16_C( 21251),  INT16_C(   503), -INT16_C(  9763),  INT16_C( 11412),
        -INT16_C( 19891), -INT16_C( 29904),  INT16_C( 25922), -INT16_C( 31005), -INT16_C( 18364),  INT16_C( 24362), -INT16_C( 30855),  INT16_C(  3364),
        -INT16_C( 24048), -INT16_C(   171),  INT16_C( 26292), -INT16_C( 31868),  INT16_C(  9621),  INT16_C( 29189),  INT16_C( 20990), -INT16_C( 27885) },
      UINT32_C(1074921764) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 k = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epi16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpeq_epi16_mask");

   easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi16(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_i16x32());
    easysimd__mmask32 r = easysimd_mm512_mask_cmpeq_epi16_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpeq_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask16 r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 1955445938), INT32_C( 1791143901), INT32_C(-1554982337), INT32_C(-1864115653),
                            INT32_C(-1774796435), INT32_C( 1168347531), INT32_C(  660969508), INT32_C( 1153796239),
                            INT32_C(  609464964), INT32_C( 1687040663), INT32_C( -477087011), INT32_C(  309017072),
                            INT32_C(-2144421691), INT32_C(  682838279), INT32_C( 1954361304), INT32_C( 1694661655)),
      easysimd_mm512_set_epi32(INT32_C( 1955445938), INT32_C( 1791143901), INT32_C(-1554982337), INT32_C(-1021004514),
                            INT32_C(-1774796435), INT32_C( 1168347531), INT32_C( 1662960443), INT32_C( 1153796239),
                            INT32_C(  609464964), INT32_C(  428287860), INT32_C(  954212709), INT32_C(  309017072),
                            INT32_C(-2144421691), INT32_C( 1263291650), INT32_C( 1954361304), INT32_C( 1694661655)),
      UINT16_C(60827) },
    { easysimd_mm512_set_epi32(INT32_C(-1966343749), INT32_C(  200215975), INT32_C(-1192030414), INT32_C(  122433675),
                            INT32_C( 2024442800), INT32_C(-1612503082), INT32_C( -352764842), INT32_C( -964919402),
                            INT32_C(  198779956), INT32_C(-1052322954), INT32_C(-2053476283), INT32_C( 1410443780),
                            INT32_C( -220052733), INT32_C( 1401656692), INT32_C(  344284159), INT32_C(  358488145)),
      easysimd_mm512_set_epi32(INT32_C(-1966343749), INT32_C(  200215975), INT32_C( 1606502639), INT32_C(  122433675),
                            INT32_C( 2024442800), INT32_C( 1676122341), INT32_C( 1013297923), INT32_C( 1764819720),
                            INT32_C(-1006160158), INT32_C(  500282446), INT32_C(-2053476283), INT32_C( 1410443780),
                            INT32_C( -891361692), INT32_C( 1401656692), INT32_C(-1666424273), INT32_C(  358488145)),
      UINT16_C(55349) },
    { easysimd_mm512_set_epi32(INT32_C( 2029514541), INT32_C(-1405121342), INT32_C( -922299669), INT32_C(-1157312309),
                            INT32_C(  348700274), INT32_C( 1888848128), INT32_C( -909115111), INT32_C(-1689020830),
                            INT32_C( -310128293), INT32_C(-2105034529), INT32_C( 1894199289), INT32_C( -520350707),
                            INT32_C( 2077151614), INT32_C(  947639177), INT32_C(  972348424), INT32_C(  688864219)),
      easysimd_mm512_set_epi32(INT32_C( 2029514541), INT32_C(-1405121342), INT32_C( -922299669), INT32_C(-1157312309),
                            INT32_C( -582496350), INT32_C(  321618555), INT32_C( -707023911), INT32_C(-1689020830),
                            INT32_C(-1223923200), INT32_C( -293312096), INT32_C( 1894199289), INT32_C(  -89856908),
                            INT32_C( 2077151614), INT32_C(  947639177), INT32_C(  352633301), INT32_C( -580477735)),
      UINT16_C(61740) },
    { easysimd_mm512_set_epi32(INT32_C(   63224893), INT32_C( -945243939), INT32_C( 1472177104), INT32_C(-1518639341),
                            INT32_C( 1244274955), INT32_C(-1053463974), INT32_C(  207788622), INT32_C( -375726536),
                            INT32_C( -219606791), INT32_C( -871332353), INT32_C( 2036105492), INT32_C( 1465626684),
                            INT32_C(  930923741), INT32_C(-1996202276), INT32_C(  336972805), INT32_C(-1729161210)),
      easysimd_mm512_set_epi32(INT32_C( 1739683658), INT32_C( -945243939), INT32_C( 1472177104), INT32_C( -959740920),
                            INT32_C( 1244274955), INT32_C(  236525659), INT32_C(  207788622), INT32_C( 1159372556),
                            INT32_C( -632841040), INT32_C( -871332353), INT32_C( 2036105492), INT32_C(-1821161401),
                            INT32_C(  626098659), INT32_C(-1099705903), INT32_C( 1892226777), INT32_C(  183269504)),
      UINT16_C(27232) },
    { easysimd_mm512_set_epi32(INT32_C( 1660264790), INT32_C( -188014963), INT32_C( 1960568786), INT32_C(  630575470),
                            INT32_C(-1560285386), INT32_C(-1080983958), INT32_C( -186614663), INT32_C(-1365084922),
                            INT32_C( 1687374482), INT32_C( 2091712477), INT32_C( 1770300152), INT32_C( 1222615684),
                            INT32_C(  987382002), INT32_C( -869689297), INT32_C( 1381156346), INT32_C(  352829646)),
      easysimd_mm512_set_epi32(INT32_C( 1495188549), INT32_C( -188014963), INT32_C( -879412194), INT32_C( 1172150075),
                            INT32_C( 1163780404), INT32_C(-1080983958), INT32_C( -186614663), INT32_C(-1365084922),
                            INT32_C( 1196042729), INT32_C( 2091712477), INT32_C( 1770300152), INT32_C( 1222615684),
                            INT32_C(  987382002), INT32_C( -453542339), INT32_C(-1460537486), INT32_C( 1311735715)),
      UINT16_C(18296) },
    { easysimd_mm512_set_epi32(INT32_C( -830898164), INT32_C( 2065530031), INT32_C( 1849339474), INT32_C( -161498764),
                            INT32_C(  726295410), INT32_C(-1366062470), INT32_C(  110025501), INT32_C(-2061598845),
                            INT32_C(-1911113344), INT32_C(-2008355607), INT32_C( 1140427951), INT32_C( 1963231912),
                            INT32_C( 1593065931), INT32_C( 1712671682), INT32_C(-2139143015), INT32_C(  330252777)),
      easysimd_mm512_set_epi32(INT32_C( -830898164), INT32_C(-2132024757), INT32_C( 1102342058), INT32_C( -161498764),
                            INT32_C(-2106128090), INT32_C(  648329890), INT32_C(-1284054768), INT32_C(-2061598845),
                            INT32_C(-1911113344), INT32_C(-2008355607), INT32_C( 1715485148), INT32_C(  155412419),
                            INT32_C( 1273550758), INT32_C( 1712671682), INT32_C(-1857983881), INT32_C( 1633779150)),
      UINT16_C(37316) },
    { easysimd_mm512_set_epi32(INT32_C( 1956746364), INT32_C( 1930323834), INT32_C(  923874794), INT32_C(  121318212),
                            INT32_C(-1375858452), INT32_C( -462992597), INT32_C( 1495829546), INT32_C(  697040437),
                            INT32_C(  727111035), INT32_C(-2061427382), INT32_C( -815432287), INT32_C(  913775211),
                            INT32_C(-1333809472), INT32_C(  114048073), INT32_C( 1312920985), INT32_C(-1819914035)),
      easysimd_mm512_set_epi32(INT32_C( 1659376087), INT32_C( 1930323834), INT32_C(  923874794), INT32_C(  121318212),
                            INT32_C(-1375858452), INT32_C(-1480844812), INT32_C(-1803673478), INT32_C(  697040437),
                            INT32_C( 1793922150), INT32_C(  391658500), INT32_C( -815432287), INT32_C(  913775211),
                            INT32_C(-1333809472), INT32_C( -169359358), INT32_C(  140424991), INT32_C(-1819914035)),
      UINT16_C(31033) },
    { easysimd_mm512_set_epi32(INT32_C( 1334496661), INT32_C(-1765072906), INT32_C(-1980138391), INT32_C(-1150536116),
                            INT32_C( -711226926), INT32_C( 1955166809), INT32_C( 1418224832), INT32_C( 1791996583),
                            INT32_C(-1305868646), INT32_C( -507537618), INT32_C(  272749509), INT32_C(-1826072492),
                            INT32_C( -629068596), INT32_C(-2142583585), INT32_C( 2048200365), INT32_C(-1377550438)),
      easysimd_mm512_set_epi32(INT32_C( 1334496661), INT32_C(-1765072906), INT32_C( -890751438), INT32_C(-1150536116),
                            INT32_C(  307879329), INT32_C( 1955166809), INT32_C(-1884386825), INT32_C( 1791996583),
                            INT32_C( 1128431085), INT32_C( -507537618), INT32_C(  272749509), INT32_C( 1579228324),
                            INT32_C( 1577134581), INT32_C(-2142583585), INT32_C( 1998674783), INT32_C(-1377550438)),
      UINT16_C(54629) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("easysimd_mm512_cmpeq_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_cmpeq_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask16 r;
  } test_vec[8] = {
    { UINT16_C(15798),
      easysimd_mm512_set_epi32(INT32_C(-1396783922), INT32_C( 2147469122), INT32_C(  245941047), INT32_C(-1608794680),
                            INT32_C( 1508622706), INT32_C( -820009589), INT32_C(-2056933337), INT32_C( 1399160559),
                            INT32_C( -781782717), INT32_C( -745938688), INT32_C( 1376412239), INT32_C(  358147007),
                            INT32_C(-2141927311), INT32_C(  -13921241), INT32_C(  993767039), INT32_C( 1399506469)),
      easysimd_mm512_set_epi32(INT32_C(-1396783922), INT32_C( 1246457300), INT32_C( 1479047358), INT32_C(  -46329110),
                            INT32_C( 1508622706), INT32_C( -820009589), INT32_C( 2080173402), INT32_C( 1937046285),
                            INT32_C( 1642337112), INT32_C(  717149471), INT32_C(  913865239), INT32_C(  358147007),
                            INT32_C( 1535690363), INT32_C(  -13921241), INT32_C(  993767039), INT32_C(  520030741)),
      UINT16_C( 3094) },
    { UINT16_C(11237),
      easysimd_mm512_set_epi32(INT32_C( -503773363), INT32_C(-1842967576), INT32_C(  214407733), INT32_C( 2136243689),
                            INT32_C(  619563347), INT32_C(-2046647578), INT32_C( -882198373), INT32_C( -832110086),
                            INT32_C(  828288790), INT32_C( 1819046419), INT32_C(  292064966), INT32_C( -747926472),
                            INT32_C( -881463995), INT32_C(-1469434386), INT32_C( -207196114), INT32_C(-1865668915)),
      easysimd_mm512_set_epi32(INT32_C(  261150369), INT32_C(-1842967576), INT32_C(-1148601855), INT32_C( 2136243689),
                            INT32_C(-2013121518), INT32_C(-2046647578), INT32_C(-1816537753), INT32_C( -832110086),
                            INT32_C(  828288790), INT32_C( 1801925150), INT32_C(  292064966), INT32_C(-1755078036),
                            INT32_C(-1190065823), INT32_C(  764421376), INT32_C( -207196114), INT32_C(-1865668915)),
      UINT16_C(  417) },
    { UINT16_C(   61),
      easysimd_mm512_set_epi32(INT32_C(  219790698), INT32_C(  346516102), INT32_C(-2082495343), INT32_C(  775700745),
                            INT32_C( -630025741), INT32_C( 1914162819), INT32_C( -226095226), INT32_C( -350619769),
                            INT32_C( 1825330894), INT32_C(  -79420124), INT32_C(  939662489), INT32_C(  667606641),
                            INT32_C(-1935137328), INT32_C(  247120548), INT32_C( -894861328), INT32_C(-1690188311)),
      easysimd_mm512_set_epi32(INT32_C(  400967600), INT32_C(  346516102), INT32_C(-2082495343), INT32_C(  775700745),
                            INT32_C( -630025741), INT32_C( 1914162819), INT32_C(-1771983812), INT32_C( -350619769),
                            INT32_C( 1825330894), INT32_C(-1887033010), INT32_C(  939662489), INT32_C( -440942769),
                            INT32_C(-1935137328), INT32_C(  247120548), INT32_C(-1352163534), INT32_C(-1690188311)),
      UINT16_C(   45) },
    { UINT16_C(40799),
      easysimd_mm512_set_epi32(INT32_C(-1103261115), INT32_C(-1530092257), INT32_C( -178076517), INT32_C( 1725180044),
                            INT32_C( -619562554), INT32_C(-2028225116), INT32_C( -516465044), INT32_C( -790492311),
                            INT32_C(  770588317), INT32_C(  966229539), INT32_C( 1696981823), INT32_C( 1846986452),
                            INT32_C(  201003704), INT32_C(  -88144749), INT32_C( -459260562), INT32_C(-1462493092)),
      easysimd_mm512_set_epi32(INT32_C(-1103261115), INT32_C(-1530092257), INT32_C( 1272329003), INT32_C( 1725180044),
                            INT32_C(-1714282865), INT32_C( 1964019033), INT32_C(   -6888948), INT32_C( -794446809),
                            INT32_C(  770588317), INT32_C(  966229539), INT32_C(-1587543669), INT32_C( 1846986452),
                            INT32_C(  201003704), INT32_C(  -88144749), INT32_C(-1549459108), INT32_C(-1462493092)),
      UINT16_C(36957) },
    { UINT16_C(18708),
      easysimd_mm512_set_epi32(INT32_C( 1654168369), INT32_C(-1358646009), INT32_C(  945188582), INT32_C( 1242452940),
                            INT32_C(-2068238117), INT32_C(  613827224), INT32_C( 1766050173), INT32_C(  788865946),
                            INT32_C( -226150288), INT32_C(   20626714), INT32_C(-1790747056), INT32_C(-1510999017),
                            INT32_C(-2059568770), INT32_C(  525242273), INT32_C(-1970979230), INT32_C( -983788353)),
      easysimd_mm512_set_epi32(INT32_C(-1802152524), INT32_C(-1358646009), INT32_C( 1400410557), INT32_C( 1242452940),
                            INT32_C(-2068238117), INT32_C(-1745049433), INT32_C(-1272787498), INT32_C(   36641197),
                            INT32_C( -226150288), INT32_C(-2087200149), INT32_C( -530182364), INT32_C(-1510999017),
                            INT32_C(-2082577633), INT32_C(  525242273), INT32_C(-1970979230), INT32_C( -983788353)),
      UINT16_C(18452) },
    { UINT16_C(48938),
      easysimd_mm512_set_epi32(INT32_C(-1802886705), INT32_C(  505130099), INT32_C( 1294359394), INT32_C(  564426410),
                            INT32_C( -813242663), INT32_C(-1097324530), INT32_C( 1599346411), INT32_C(-1815738445),
                            INT32_C( 2114996332), INT32_C( 2143192037), INT32_C(  342894910), INT32_C( 1933006347),
                            INT32_C(  215936041), INT32_C( 2138148935), INT32_C(-1975112588), INT32_C(-1313889253)),
      easysimd_mm512_set_epi32(INT32_C( 1272515820), INT32_C(-1571014987), INT32_C( 1294359394), INT32_C(  564426410),
                            INT32_C( -305474417), INT32_C(-2099686495), INT32_C(  217917259), INT32_C( 1770631752),
                            INT32_C( 2114996332), INT32_C( 2143192037), INT32_C(  -26985081), INT32_C(  603877714),
                            INT32_C( 1592556524), INT32_C(  420570241), INT32_C(-1975112588), INT32_C(-1313889253)),
      UINT16_C(12290) },
    { UINT16_C(14127),
      easysimd_mm512_set_epi32(INT32_C(  452796731), INT32_C( -256668338), INT32_C(-1710549095), INT32_C( 1982965424),
                            INT32_C( 1184306045), INT32_C( -221254467), INT32_C( 1420239721), INT32_C( 2028887361),
                            INT32_C(-1950932361), INT32_C( 1650853943), INT32_C(  239751123), INT32_C( 1018010808),
                            INT32_C( -248946240), INT32_C(  701510715), INT32_C(  824235240), INT32_C( 1829156606)),
      easysimd_mm512_set_epi32(INT32_C(  452796731), INT32_C( 1031814185), INT32_C(-1710549095), INT32_C(  406415467),
                            INT32_C( 1184306045), INT32_C( -221254467), INT32_C(  419739010), INT32_C( 1708161231),
                            INT32_C(-1950932361), INT32_C( 1650853943), INT32_C(  239751123), INT32_C( 1018010808),
                            INT32_C( -248946240), INT32_C(  701510715), INT32_C(-1571248435), INT32_C( 1829156606)),
      UINT16_C( 9261) },
    { UINT16_C(22801),
      easysimd_mm512_set_epi32(INT32_C( 1869800572), INT32_C(  184060195), INT32_C(   81710208), INT32_C( -451284065),
                            INT32_C(  397153235), INT32_C(  120564446), INT32_C(-2128920097), INT32_C( 1498011427),
                            INT32_C( -602736654), INT32_C( -931955343), INT32_C(  270436915), INT32_C( -984637478),
                            INT32_C( 2080482721), INT32_C( 1599947836), INT32_C(  374268618), INT32_C(  202341051)),
      easysimd_mm512_set_epi32(INT32_C( 1869800572), INT32_C(  350721255), INT32_C( 1725621650), INT32_C( 2020045509),
                            INT32_C(  397153235), INT32_C( 2059505832), INT32_C(-2128920097), INT32_C( 1498011427),
                            INT32_C(  884679844), INT32_C( -931955343), INT32_C(-1565261303), INT32_C( -984637478),
                            INT32_C( 1047792745), INT32_C(  969830078), INT32_C(  374268618), INT32_C(  202341051)),
      UINT16_C( 2321) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 r;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpeq_epi32_mask");
    easysimd_assert_equal_mmask16(HEDLEY_STATIC_CAST(uint16_t, r), HEDLEY_STATIC_CAST(uint16_t, test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_cmpeq_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask8 r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 1145569124203592220), INT64_C( 8866992319046943109),
                            INT64_C( 1920152028348566704), INT64_C( 5434169962120345100),
                            INT64_C( 2279810443797316081), INT64_C( 8202334326145056493),
                            INT64_C(-3178237508368099649), INT64_C(  691539695110831332)),
      easysimd_mm512_set_epi64(INT64_C( 1145569124203592220), INT64_C( 7456258988741005047),
                            INT64_C( 1920152028348566704), INT64_C(-5531344059509930758),
                            INT64_C( 2279810443797316081), INT64_C( 4212838215119313762),
                            INT64_C(-3178237508368099649), INT64_C(  805234941373423218)),
      UINT8_C(170) },
    { easysimd_mm512_set_epi64(INT64_C(-9153030146845322943), INT64_C(-3269802522838954453),
                            INT64_C( 4057416377680772191), INT64_C(-4770808841142629569),
                            INT64_C(-3341852712217874901), INT64_C( 2807150139607841402),
                            INT64_C(-4019938614639236980), INT64_C(-5612844059017155926)),
      easysimd_mm512_set_epi64(INT64_C(-9153030146845322943), INT64_C(-3269802522838954453),
                            INT64_C( 4057416377680772191), INT64_C( -940603917401247915),
                            INT64_C(-3341852712217874901), INT64_C( 2807150139607841402),
                            INT64_C(-4019938614639236980), INT64_C(-5612844059017155926)),
      UINT8_C(239) },
    { easysimd_mm512_set_epi64(INT64_C(-6535160640888369935), INT64_C( 4320507985166557084),
                            INT64_C( 5472888856009482484), INT64_C(-3128657528300165720),
                            INT64_C( 7430914891859325397), INT64_C( -863913734109164798),
                            INT64_C(-7794735185814972844), INT64_C( 6163895309000776850)),
      easysimd_mm512_set_epi64(INT64_C(-4220461743444256802), INT64_C( 2616373937292152846),
                            INT64_C( 5472888856009482484), INT64_C(-3128657528300165720),
                            INT64_C(-4693544589206901195), INT64_C( -863913734109164798),
                            INT64_C(-7794735185814972844), INT64_C(-7914074467628930001)),
      UINT8_C( 54) },
    { easysimd_mm512_set_epi64(INT64_C(-2366835550617683621), INT64_C(-7526131353484035546),
                            INT64_C(  792273323175818479), INT64_C(-3293855504021481788),
                            INT64_C( 5766970185621377329), INT64_C( 8555682923177627503),
                            INT64_C( 6363802824474944219), INT64_C(-2294667848278645794)),
      easysimd_mm512_set_epi64(INT64_C(-2869910567061155737), INT64_C(-7526131353484035546),
                            INT64_C(  792273323175818479), INT64_C( 7002007300200137801),
                            INT64_C( 5766970185621377329), INT64_C( 8555682923177627503),
                            INT64_C( 6363802824474944219), INT64_C( 1235305386817146646)),
      UINT8_C(110) },
    { easysimd_mm512_set_epi64(INT64_C( 1781453983767744725), INT64_C( 8717105342239974860),
                            INT64_C(-6555437070120516632), INT64_C( -945434448901737124),
                            INT64_C( 2467182069214891728), INT64_C( 6380355612148977321),
                            INT64_C(-8948278762783625779), INT64_C( 4377995125356963906)),
      easysimd_mm512_set_epi64(INT64_C( 1781453983767744725), INT64_C( 8717105342239974860),
                            INT64_C(-1574265126641933862), INT64_C(-3609136820784425910),
                            INT64_C( 8413977304256335681), INT64_C(-4046135395677773903),
                            INT64_C(-8948278762783625779), INT64_C( 4377995125356963906)),
      UINT8_C(195) },
    { easysimd_mm512_set_epi64(INT64_C( 3163831140256245302), INT64_C(-7476767975543057611),
                            INT64_C( 4783231484571490676), INT64_C( 4356333140933542181),
                            INT64_C(-5734470250608567849), INT64_C( 5428089484060124354),
                            INT64_C( 6915844972668556704), INT64_C(-4128418267655054219)),
      easysimd_mm512_set_epi64(INT64_C( 3163831140256245302), INT64_C(-7945608864951271413),
                            INT64_C(-3728561425301803734), INT64_C( 6344562138072151085),
                            INT64_C(-5734470250608567849), INT64_C( 5868624435741359252),
                            INT64_C( 6915844972668556704), INT64_C(-4128418267655054219)),
      UINT8_C(139) },
    { easysimd_mm512_set_epi64(INT64_C(-8545152605640787948), INT64_C(-5234822620280611494),
                            INT64_C(-2932089901585751375), INT64_C( 4017618173912988951),
                            INT64_C(-8696436452927061736), INT64_C( 3602642025812661939),
                            INT64_C( 5777136991119584953), INT64_C(-5473038481952171581)),
      easysimd_mm512_set_epi64(INT64_C(-8545152605640787948), INT64_C(-7808252369899371159),
                            INT64_C(-2932089901585751375), INT64_C(  -26139149052321087),
                            INT64_C(-6986660196527912755), INT64_C( 3602642025812661939),
                            INT64_C( 5777136991119584953), INT64_C(-5473038481952171581)),
      UINT8_C(167) },
    { easysimd_mm512_set_epi64(INT64_C( 7780170108497689334), INT64_C(-8001282944915881932),
                            INT64_C(  382835809361431399), INT64_C(-5014881555296189468),
                            INT64_C(-1844642447215154571), INT64_C(-5452282829002750089),
                            INT64_C(-7793611854809744260), INT64_C(  938166230586687295)),
      easysimd_mm512_set_epi64(INT64_C( 7780170108497689334), INT64_C( 1739290942316187796),
                            INT64_C( 7815402837606564081), INT64_C(-5014881555296189468),
                            INT64_C(-1844642447215154571), INT64_C( -253128228754997390),
                            INT64_C(-7793611854809744260), INT64_C(  938166230586687295)),
      UINT8_C(155) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 r;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpeq_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_cmpeq_epi64_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__mmask8 r;
  } test_vec[8] = {
       { UINT8_C( 90),
      easysimd_mm512_set_epi64(INT64_C( 7722926897436765530), INT64_C( 7338279138551748064),
                            INT64_C( 8433308126101200079), INT64_C(-4390305748733976547),
                            INT64_C(-1482589068035252753), INT64_C(-5002011091694476743),
                            INT64_C( 5809674310022718254), INT64_C( 7620249298233351482)),
      easysimd_mm512_set_epi64(INT64_C( 7722926897436765530), INT64_C( 7338279138551748064),
                            INT64_C(-2656726859984743367), INT64_C(-4390305748733976547),
                            INT64_C(-1482589068035252753), INT64_C(-5269390469191050553),
                            INT64_C( 5809674310022718254), INT64_C( 7620249298233351482)),
      UINT8_C( 90) },
    { UINT8_C(178),
      easysimd_mm512_set_epi64(INT64_C(-8806453660480970182), INT64_C(-2722914847628644365),
                            INT64_C( 2171146127585219679), INT64_C(-7200523266853707115),
                            INT64_C( 8505301695237968355), INT64_C( 6373940775215479358),
                            INT64_C( 8209357864908427195), INT64_C( -995665125730760835)),
      easysimd_mm512_set_epi64(INT64_C(-8806453660480970182), INT64_C(-2722914847628644365),
                            INT64_C(-1180134256156200317), INT64_C(-7200523266853707115),
                            INT64_C(-1610604796376715795), INT64_C( 5419019224867820225),
                            INT64_C( 8209357864908427195), INT64_C( -995665125730760835)),
      UINT8_C(146) },
    { UINT8_C(171),
      easysimd_mm512_set_epi64(INT64_C(-6245801519083893310), INT64_C(-7866373458730819532),
                            INT64_C(-5627757407772356197), INT64_C(-2425546480980122794),
                            INT64_C(-8451301604567613199), INT64_C( 1369383717682198649),
                            INT64_C( -532343328754521574), INT64_C(-1062878680437210584)),
      easysimd_mm512_set_epi64(INT64_C(-9163399881020056955), INT64_C(-2992244142829238392),
                            INT64_C( -213476403626539965), INT64_C(-8591297333400286921),
                            INT64_C(-8451301604567613199), INT64_C(-8139768780594538635),
                            INT64_C(-4714070518405120331), INT64_C(-1062878680437210584)),
      UINT8_C(  9) },
    { UINT8_C( 28),
      easysimd_mm512_set_epi64(INT64_C( 7845956693704412298), INT64_C(-5781930833336581955),
                            INT64_C( 2851517750261041799), INT64_C(-5814293521236182366),
                            INT64_C( 2292150971239308783), INT64_C( 2594053186857735013),
                            INT64_C( 7307535341641173075), INT64_C(-4427478291595527940)),
      easysimd_mm512_set_epi64(INT64_C(  536264388241191871), INT64_C(-5781930833336581955),
                            INT64_C( 2851517750261041799), INT64_C( 1349842462394812975),
                            INT64_C( 2292150971239308783), INT64_C( 2594053186857735013),
                            INT64_C( 7307535341641173075), INT64_C(-4427478291595527940)),
      UINT8_C( 12) },
    { UINT8_C(248),
      easysimd_mm512_set_epi64(INT64_C( 4900816215694077255), INT64_C(-2732029741423656661),
                            INT64_C( 1082977534221618055), INT64_C(-3092044493389993636),
                            INT64_C(-4299277917890019767), INT64_C(-2055775203132417874),
                            INT64_C( -778633101599852237), INT64_C( -563223173848121636)),
      easysimd_mm512_set_epi64(INT64_C( 7049304296219110648), INT64_C(-2732029741423656661),
                            INT64_C( 7088083428992159722), INT64_C(-3092044493389993636),
                            INT64_C(-4299277917890019767), INT64_C( 4225506809727089751),
                            INT64_C( -778633101599852237), INT64_C( -563223173848121636)),
      UINT8_C( 88) },
    { UINT8_C(171),
      easysimd_mm512_set_epi64(INT64_C(-1412821155990992029), INT64_C( 4454576651901490962),
                            INT64_C(-7284760734604447652), INT64_C(-7443130466673006479),
                            INT64_C(  320054597637804434), INT64_C(-8860872372305530355),
                            INT64_C(-8428145646879978292), INT64_C(-6547252853189215611)),
      easysimd_mm512_set_epi64(INT64_C(-1412821155990992029), INT64_C(-2354123670646573707),
                            INT64_C( 4506838144989822528), INT64_C(-7443130466673006479),
                            INT64_C(-5147543239321546686), INT64_C(-8860872372305530355),
                            INT64_C(-8428145646879978292), INT64_C(-6547252853189215611)),
      UINT8_C(131) },
    { UINT8_C( 29),
      easysimd_mm512_set_epi64(INT64_C( 5675137803130124480), INT64_C( 1211541157654985046),
                            INT64_C( 8724633375562564314), INT64_C(-2760658800846254598),
                            INT64_C(-6714474269646576270), INT64_C( 3484180661422871715),
                            INT64_C( 1469796163712815354), INT64_C(-2336393240308600160)),
      easysimd_mm512_set_epi64(INT64_C( 5675137803130124480), INT64_C( 1211541157654985046),
                            INT64_C(-8867413355151838495), INT64_C(-8867147959443474315),
                            INT64_C(-6714474269646576270), INT64_C( 3484180661422871715),
                            INT64_C(-7735267815657951749), INT64_C(  413036036281601883)),
      UINT8_C( 12) },
    { UINT8_C(211),
      easysimd_mm512_set_epi64(INT64_C(-6713502673628263139), INT64_C( 1559753162601267291),
                            INT64_C( 5045660940436454371), INT64_C( 7013290440433503154),
                            INT64_C(-8475145246816690249), INT64_C(-6834826688677600633),
                            INT64_C(-2109099044497919348), INT64_C( 1351143524438105934)),
      easysimd_mm512_set_epi64(INT64_C( 5625319538109918668), INT64_C( 1559753162601267291),
                            INT64_C( 5045660940436454371), INT64_C(-4654386914804892920),
                            INT64_C( 2407237530895996207), INT64_C(-6834826688677600633),
                            INT64_C( 4684210505965066200), INT64_C( 1351143524438105934)),
      UINT8_C( 65) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 r;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpeq_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpeq_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

#if !defined(EASYSIMD_BUG_GCC_96174)

static int
test_easysimd_mm512_cmpeq_ps_mask (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -330.05), EASYSIMD_FLOAT32_C(   847.28), EASYSIMD_FLOAT32_C(    61.79), EASYSIMD_FLOAT32_C(   748.75),
        EASYSIMD_FLOAT32_C(  -125.94), EASYSIMD_FLOAT32_C(  -287.83), EASYSIMD_FLOAT32_C(  -156.45), EASYSIMD_FLOAT32_C(  -904.26),
        EASYSIMD_FLOAT32_C(   393.62), EASYSIMD_FLOAT32_C(   694.71), EASYSIMD_FLOAT32_C(   345.37), EASYSIMD_FLOAT32_C(   245.98),
        EASYSIMD_FLOAT32_C(  -522.67), EASYSIMD_FLOAT32_C(   140.34), EASYSIMD_FLOAT32_C(  -555.38), EASYSIMD_FLOAT32_C(   596.45) },
      { EASYSIMD_FLOAT32_C(  -330.05), EASYSIMD_FLOAT32_C(  -812.74), EASYSIMD_FLOAT32_C(    61.79), EASYSIMD_FLOAT32_C(  -304.55),
        EASYSIMD_FLOAT32_C(    95.53), EASYSIMD_FLOAT32_C(  -287.83), EASYSIMD_FLOAT32_C(  -156.45), EASYSIMD_FLOAT32_C(   699.14),
        EASYSIMD_FLOAT32_C(   676.85), EASYSIMD_FLOAT32_C(   694.71), EASYSIMD_FLOAT32_C(   345.37), EASYSIMD_FLOAT32_C(   245.98),
        EASYSIMD_FLOAT32_C(  -161.51), EASYSIMD_FLOAT32_C(   140.34), EASYSIMD_FLOAT32_C(  -399.39), EASYSIMD_FLOAT32_C(   596.45) },
      UINT16_C(44645) },
    { { EASYSIMD_FLOAT32_C(  -717.84), EASYSIMD_FLOAT32_C(   512.02), EASYSIMD_FLOAT32_C(    98.80), EASYSIMD_FLOAT32_C(  -966.72),
        EASYSIMD_FLOAT32_C(   -60.71), EASYSIMD_FLOAT32_C(  -584.27), EASYSIMD_FLOAT32_C(   204.10), EASYSIMD_FLOAT32_C(   295.96),
        EASYSIMD_FLOAT32_C(   -70.24), EASYSIMD_FLOAT32_C(   661.77), EASYSIMD_FLOAT32_C(   894.04), EASYSIMD_FLOAT32_C(   352.28),
        EASYSIMD_FLOAT32_C(   620.44), EASYSIMD_FLOAT32_C(   936.22), EASYSIMD_FLOAT32_C(   428.81), EASYSIMD_FLOAT32_C(   543.55) },
      { EASYSIMD_FLOAT32_C(  -717.84), EASYSIMD_FLOAT32_C(   313.36), EASYSIMD_FLOAT32_C(  -806.61), EASYSIMD_FLOAT32_C(   690.40),
        EASYSIMD_FLOAT32_C(   646.60), EASYSIMD_FLOAT32_C(  -584.27), EASYSIMD_FLOAT32_C(   204.10), EASYSIMD_FLOAT32_C(   460.04),
        EASYSIMD_FLOAT32_C(   733.57), EASYSIMD_FLOAT32_C(   661.77), EASYSIMD_FLOAT32_C(   894.04), EASYSIMD_FLOAT32_C(  -252.47),
        EASYSIMD_FLOAT32_C(  -340.15), EASYSIMD_FLOAT32_C(   936.22), EASYSIMD_FLOAT32_C(   428.81), EASYSIMD_FLOAT32_C(   543.55) },
      UINT16_C(58977) },
    { { EASYSIMD_FLOAT32_C(   375.91), EASYSIMD_FLOAT32_C(   -19.72), EASYSIMD_FLOAT32_C(   336.05), EASYSIMD_FLOAT32_C(  -540.20),
        EASYSIMD_FLOAT32_C(  -665.47), EASYSIMD_FLOAT32_C(  -492.58), EASYSIMD_FLOAT32_C(    15.30), EASYSIMD_FLOAT32_C(   126.92),
        EASYSIMD_FLOAT32_C(   767.58), EASYSIMD_FLOAT32_C(   861.15), EASYSIMD_FLOAT32_C(   -58.47), EASYSIMD_FLOAT32_C(  -387.52),
        EASYSIMD_FLOAT32_C(   800.70), EASYSIMD_FLOAT32_C(  -537.31), EASYSIMD_FLOAT32_C(  -644.51), EASYSIMD_FLOAT32_C(  -955.44) },
      { EASYSIMD_FLOAT32_C(   375.91), EASYSIMD_FLOAT32_C(  -356.80), EASYSIMD_FLOAT32_C(   336.05), EASYSIMD_FLOAT32_C(   -10.02),
        EASYSIMD_FLOAT32_C(   -64.34), EASYSIMD_FLOAT32_C(   408.76), EASYSIMD_FLOAT32_C(  -734.89), EASYSIMD_FLOAT32_C(   126.92),
        EASYSIMD_FLOAT32_C(    10.21), EASYSIMD_FLOAT32_C(   861.15), EASYSIMD_FLOAT32_C(   153.18), EASYSIMD_FLOAT32_C(   569.21),
        EASYSIMD_FLOAT32_C(   321.66), EASYSIMD_FLOAT32_C(  -537.31), EASYSIMD_FLOAT32_C(   613.36), EASYSIMD_FLOAT32_C(  -776.54) },
      UINT16_C( 8837) },
    { { EASYSIMD_FLOAT32_C(  -842.85), EASYSIMD_FLOAT32_C(  -336.15), EASYSIMD_FLOAT32_C(  -966.61), EASYSIMD_FLOAT32_C(   670.20),
        EASYSIMD_FLOAT32_C(   783.55), EASYSIMD_FLOAT32_C(   477.19), EASYSIMD_FLOAT32_C(  -864.95), EASYSIMD_FLOAT32_C(  -372.20),
        EASYSIMD_FLOAT32_C(   -94.30), EASYSIMD_FLOAT32_C(  -879.40), EASYSIMD_FLOAT32_C(  -161.82), EASYSIMD_FLOAT32_C(   100.12),
        EASYSIMD_FLOAT32_C(   850.32), EASYSIMD_FLOAT32_C(   476.49), EASYSIMD_FLOAT32_C(  -174.44), EASYSIMD_FLOAT32_C(   934.13) },
      { EASYSIMD_FLOAT32_C(   404.40), EASYSIMD_FLOAT32_C(  -570.57), EASYSIMD_FLOAT32_C(   -86.01), EASYSIMD_FLOAT32_C(   670.20),
        EASYSIMD_FLOAT32_C(   312.45), EASYSIMD_FLOAT32_C(   381.45), EASYSIMD_FLOAT32_C(  -864.95), EASYSIMD_FLOAT32_C(  -372.20),
        EASYSIMD_FLOAT32_C(   -94.30), EASYSIMD_FLOAT32_C(  -879.40), EASYSIMD_FLOAT32_C(  -161.82), EASYSIMD_FLOAT32_C(   100.12),
        EASYSIMD_FLOAT32_C(   850.32), EASYSIMD_FLOAT32_C(   386.54), EASYSIMD_FLOAT32_C(   295.11), EASYSIMD_FLOAT32_C(  -992.52) },
      UINT16_C( 8136) },
    { { EASYSIMD_FLOAT32_C(   877.31), EASYSIMD_FLOAT32_C(   884.47), EASYSIMD_FLOAT32_C(  -380.38), EASYSIMD_FLOAT32_C(  -700.84),
        EASYSIMD_FLOAT32_C(   945.89), EASYSIMD_FLOAT32_C(   280.68), EASYSIMD_FLOAT32_C(   832.06), EASYSIMD_FLOAT32_C(   359.22),
        EASYSIMD_FLOAT32_C(   586.70), EASYSIMD_FLOAT32_C(   448.55), EASYSIMD_FLOAT32_C(   510.98), EASYSIMD_FLOAT32_C(  -325.00),
        EASYSIMD_FLOAT32_C(   847.13), EASYSIMD_FLOAT32_C(  -548.42), EASYSIMD_FLOAT32_C(  -663.23), EASYSIMD_FLOAT32_C(   110.33) },
      { EASYSIMD_FLOAT32_C(   877.31), EASYSIMD_FLOAT32_C(   884.47), EASYSIMD_FLOAT32_C(  -380.38), EASYSIMD_FLOAT32_C(  -700.84),
        EASYSIMD_FLOAT32_C(   945.89), EASYSIMD_FLOAT32_C(   280.68), EASYSIMD_FLOAT32_C(   832.06), EASYSIMD_FLOAT32_C(   359.22),
        EASYSIMD_FLOAT32_C(  -806.36), EASYSIMD_FLOAT32_C(  -673.67), EASYSIMD_FLOAT32_C(   510.98), EASYSIMD_FLOAT32_C(  -346.39),
        EASYSIMD_FLOAT32_C(   789.45), EASYSIMD_FLOAT32_C(  -548.42), EASYSIMD_FLOAT32_C(   989.10), EASYSIMD_FLOAT32_C(  -487.94) },
      UINT16_C( 9471) },
    { { EASYSIMD_FLOAT32_C(  -787.05), EASYSIMD_FLOAT32_C(   806.72), EASYSIMD_FLOAT32_C(   520.29), EASYSIMD_FLOAT32_C(  -321.05),
        EASYSIMD_FLOAT32_C(  -366.95), EASYSIMD_FLOAT32_C(  -748.89), EASYSIMD_FLOAT32_C(   687.71), EASYSIMD_FLOAT32_C(  -416.88),
        EASYSIMD_FLOAT32_C(  -561.92), EASYSIMD_FLOAT32_C(  -926.01), EASYSIMD_FLOAT32_C(   843.79), EASYSIMD_FLOAT32_C(   849.56),
        EASYSIMD_FLOAT32_C(   -51.86), EASYSIMD_FLOAT32_C(  -481.78), EASYSIMD_FLOAT32_C(   491.33), EASYSIMD_FLOAT32_C(  -936.26) },
      { EASYSIMD_FLOAT32_C(  -787.05), EASYSIMD_FLOAT32_C(   806.72), EASYSIMD_FLOAT32_C(   299.54), EASYSIMD_FLOAT32_C(   884.74),
        EASYSIMD_FLOAT32_C(  -278.71), EASYSIMD_FLOAT32_C(  -748.89), EASYSIMD_FLOAT32_C(   570.30), EASYSIMD_FLOAT32_C(  -416.88),
        EASYSIMD_FLOAT32_C(  -561.92), EASYSIMD_FLOAT32_C(    59.09), EASYSIMD_FLOAT32_C(   843.79), EASYSIMD_FLOAT32_C(   849.56),
        EASYSIMD_FLOAT32_C(  -136.84), EASYSIMD_FLOAT32_C(  -481.78), EASYSIMD_FLOAT32_C(   491.33), EASYSIMD_FLOAT32_C(  -936.26) },
      UINT16_C(60835) },
    { { EASYSIMD_FLOAT32_C(  -837.49), EASYSIMD_FLOAT32_C(   -79.02), EASYSIMD_FLOAT32_C(  -844.39), EASYSIMD_FLOAT32_C(  -973.47),
        EASYSIMD_FLOAT32_C(  -499.80), EASYSIMD_FLOAT32_C(   961.14), EASYSIMD_FLOAT32_C(   336.59), EASYSIMD_FLOAT32_C(  -368.95),
        EASYSIMD_FLOAT32_C(   727.99), EASYSIMD_FLOAT32_C(  -900.81), EASYSIMD_FLOAT32_C(   655.07), EASYSIMD_FLOAT32_C(  -624.42),
        EASYSIMD_FLOAT32_C(   244.09), EASYSIMD_FLOAT32_C(   360.96), EASYSIMD_FLOAT32_C(  -837.70), EASYSIMD_FLOAT32_C(  -929.19) },
      { EASYSIMD_FLOAT32_C(  -837.49), EASYSIMD_FLOAT32_C(   -79.02), EASYSIMD_FLOAT32_C(  -169.54), EASYSIMD_FLOAT32_C(   100.98),
        EASYSIMD_FLOAT32_C(  -499.80), EASYSIMD_FLOAT32_C(   961.14), EASYSIMD_FLOAT32_C(  -254.87), EASYSIMD_FLOAT32_C(   592.42),
        EASYSIMD_FLOAT32_C(   312.40), EASYSIMD_FLOAT32_C(   958.12), EASYSIMD_FLOAT32_C(  -284.13), EASYSIMD_FLOAT32_C(  -624.42),
        EASYSIMD_FLOAT32_C(  -196.30), EASYSIMD_FLOAT32_C(   360.96), EASYSIMD_FLOAT32_C(  -837.70), EASYSIMD_FLOAT32_C(  -975.45) },
      UINT16_C(26675) },
    { { EASYSIMD_FLOAT32_C(   928.69), EASYSIMD_FLOAT32_C(    -3.95), EASYSIMD_FLOAT32_C(  -214.33), EASYSIMD_FLOAT32_C(  -971.80),
        EASYSIMD_FLOAT32_C(  -780.70), EASYSIMD_FLOAT32_C(   950.39), EASYSIMD_FLOAT32_C(  -857.68), EASYSIMD_FLOAT32_C(  -246.08),
        EASYSIMD_FLOAT32_C(   789.62), EASYSIMD_FLOAT32_C(  -840.89), EASYSIMD_FLOAT32_C(   194.42), EASYSIMD_FLOAT32_C(  -873.48),
        EASYSIMD_FLOAT32_C(  -365.78), EASYSIMD_FLOAT32_C(  -117.81), EASYSIMD_FLOAT32_C(   601.86), EASYSIMD_FLOAT32_C(   913.26) },
      { EASYSIMD_FLOAT32_C(   928.69), EASYSIMD_FLOAT32_C(    -3.95), EASYSIMD_FLOAT32_C(  -214.33), EASYSIMD_FLOAT32_C(   377.34),
        EASYSIMD_FLOAT32_C(  -525.21), EASYSIMD_FLOAT32_C(  -436.16), EASYSIMD_FLOAT32_C(   186.25), EASYSIMD_FLOAT32_C(  -246.08),
        EASYSIMD_FLOAT32_C(   623.36), EASYSIMD_FLOAT32_C(  -840.89), EASYSIMD_FLOAT32_C(   194.42), EASYSIMD_FLOAT32_C(  -873.48),
        EASYSIMD_FLOAT32_C(  -679.52), EASYSIMD_FLOAT32_C(   447.41), EASYSIMD_FLOAT32_C(  -608.79), EASYSIMD_FLOAT32_C(   721.43) },
      UINT16_C( 3719) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_ps_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpeq_ps_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmpeq_pd_mask (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   271.69), EASYSIMD_FLOAT64_C(   217.70), EASYSIMD_FLOAT64_C(   925.72), EASYSIMD_FLOAT64_C(   520.03),
        EASYSIMD_FLOAT64_C(   127.68), EASYSIMD_FLOAT64_C(   -63.68), EASYSIMD_FLOAT64_C(  -338.02), EASYSIMD_FLOAT64_C(   823.86) },
      { EASYSIMD_FLOAT64_C(   690.45), EASYSIMD_FLOAT64_C(   347.44), EASYSIMD_FLOAT64_C(  -649.53), EASYSIMD_FLOAT64_C(  -641.60),
        EASYSIMD_FLOAT64_C(   658.05), EASYSIMD_FLOAT64_C(  -212.84), EASYSIMD_FLOAT64_C(   796.21), EASYSIMD_FLOAT64_C(   -36.15) },
      UINT8_C(  0) },
    { { EASYSIMD_FLOAT64_C(   417.99), EASYSIMD_FLOAT64_C(  -883.29), EASYSIMD_FLOAT64_C(   571.34), EASYSIMD_FLOAT64_C(   535.08),
        EASYSIMD_FLOAT64_C(  -923.74), EASYSIMD_FLOAT64_C(   773.69), EASYSIMD_FLOAT64_C(  -589.26), EASYSIMD_FLOAT64_C(   350.94) },
      { EASYSIMD_FLOAT64_C(   179.83), EASYSIMD_FLOAT64_C(   445.85), EASYSIMD_FLOAT64_C(  -677.60), EASYSIMD_FLOAT64_C(  -480.10),
        EASYSIMD_FLOAT64_C(  -974.87), EASYSIMD_FLOAT64_C(  -558.93), EASYSIMD_FLOAT64_C(    47.77), EASYSIMD_FLOAT64_C(   924.57) },
      UINT8_C(  0) },
    { { EASYSIMD_FLOAT64_C(  -695.12), EASYSIMD_FLOAT64_C(  -819.01), EASYSIMD_FLOAT64_C(   861.37), EASYSIMD_FLOAT64_C(  -968.92),
        EASYSIMD_FLOAT64_C(  -642.09), EASYSIMD_FLOAT64_C(   475.36), EASYSIMD_FLOAT64_C(  -653.40), EASYSIMD_FLOAT64_C(   274.91) },
      { EASYSIMD_FLOAT64_C(   408.72), EASYSIMD_FLOAT64_C(  -646.21), EASYSIMD_FLOAT64_C(  -843.45), EASYSIMD_FLOAT64_C(   107.84),
        EASYSIMD_FLOAT64_C(   465.38), EASYSIMD_FLOAT64_C(  -336.34), EASYSIMD_FLOAT64_C(  -820.42), EASYSIMD_FLOAT64_C(  -749.33) },
      UINT8_C(  0) },
    { { EASYSIMD_FLOAT64_C(    -9.72), EASYSIMD_FLOAT64_C(   643.44), EASYSIMD_FLOAT64_C(   336.27), EASYSIMD_FLOAT64_C(  -313.97),
        EASYSIMD_FLOAT64_C(  -863.83), EASYSIMD_FLOAT64_C(  -448.10), EASYSIMD_FLOAT64_C(   771.84), EASYSIMD_FLOAT64_C(   249.27) },
      { EASYSIMD_FLOAT64_C(  -506.33), EASYSIMD_FLOAT64_C(    28.98), EASYSIMD_FLOAT64_C(  -919.42), EASYSIMD_FLOAT64_C(  -710.08),
        EASYSIMD_FLOAT64_C(  -376.38), EASYSIMD_FLOAT64_C(   181.22), EASYSIMD_FLOAT64_C(  -315.61), EASYSIMD_FLOAT64_C(  -521.71) },
      UINT8_C(  0) },
    { { EASYSIMD_FLOAT64_C(  -309.90), EASYSIMD_FLOAT64_C(  -566.85), EASYSIMD_FLOAT64_C(   953.96), EASYSIMD_FLOAT64_C(  -760.71),
        EASYSIMD_FLOAT64_C(   715.80), EASYSIMD_FLOAT64_C(   511.82), EASYSIMD_FLOAT64_C(   185.57), EASYSIMD_FLOAT64_C(   958.96) },
      { EASYSIMD_FLOAT64_C(  -823.31), EASYSIMD_FLOAT64_C(   653.67), EASYSIMD_FLOAT64_C(   300.89), EASYSIMD_FLOAT64_C(  -999.35),
        EASYSIMD_FLOAT64_C(  -123.69), EASYSIMD_FLOAT64_C(  -935.82), EASYSIMD_FLOAT64_C(  -283.75), EASYSIMD_FLOAT64_C(  -911.67) },
      UINT8_C(  0) },
    { { EASYSIMD_FLOAT64_C(  -370.17), EASYSIMD_FLOAT64_C(   581.24), EASYSIMD_FLOAT64_C(   903.15), EASYSIMD_FLOAT64_C(  -702.97),
        EASYSIMD_FLOAT64_C(  -784.81), EASYSIMD_FLOAT64_C(  -282.51), EASYSIMD_FLOAT64_C(  -162.91), EASYSIMD_FLOAT64_C(   -67.74) },
      { EASYSIMD_FLOAT64_C(  -458.51), EASYSIMD_FLOAT64_C(  -138.00), EASYSIMD_FLOAT64_C(   634.22), EASYSIMD_FLOAT64_C(  -641.32),
        EASYSIMD_FLOAT64_C(  -700.95), EASYSIMD_FLOAT64_C(  -830.62), EASYSIMD_FLOAT64_C(  -270.15), EASYSIMD_FLOAT64_C(  -342.52) },
      UINT8_C(  0) },
    { { EASYSIMD_FLOAT64_C(  -741.30), EASYSIMD_FLOAT64_C(  -961.63), EASYSIMD_FLOAT64_C(  -159.42), EASYSIMD_FLOAT64_C(   596.72),
        EASYSIMD_FLOAT64_C(  -872.26), EASYSIMD_FLOAT64_C(   -77.79), EASYSIMD_FLOAT64_C(   608.69), EASYSIMD_FLOAT64_C(   181.91) },
      { EASYSIMD_FLOAT64_C(  -693.78), EASYSIMD_FLOAT64_C(  -430.90), EASYSIMD_FLOAT64_C(  -141.87), EASYSIMD_FLOAT64_C(  -384.25),
        EASYSIMD_FLOAT64_C(   -74.70), EASYSIMD_FLOAT64_C(   434.70), EASYSIMD_FLOAT64_C(    -4.99), EASYSIMD_FLOAT64_C(   104.05) },
      UINT8_C(  0) },
    { { EASYSIMD_FLOAT64_C(   817.79), EASYSIMD_FLOAT64_C(   652.33), EASYSIMD_FLOAT64_C(  -345.32), EASYSIMD_FLOAT64_C(   150.71),
        EASYSIMD_FLOAT64_C(   939.32), EASYSIMD_FLOAT64_C(  -867.25), EASYSIMD_FLOAT64_C(   158.96), EASYSIMD_FLOAT64_C(  -396.12) },
      { EASYSIMD_FLOAT64_C(   363.34), EASYSIMD_FLOAT64_C(   571.53), EASYSIMD_FLOAT64_C(  -232.25), EASYSIMD_FLOAT64_C(   496.58),
        EASYSIMD_FLOAT64_C(    40.81), EASYSIMD_FLOAT64_C(   -69.57), EASYSIMD_FLOAT64_C(   792.81), EASYSIMD_FLOAT64_C(   833.83) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpeq_pd_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpeq_pd_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
}

#endif /* !defined(EASYSIMD_BUG_GCC_96174) */

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpeq_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpeq_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpeq_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpeq_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpeq_epu64_mask)

  #if !defined(EASYSIMD_BUG_GCC_96174)
    EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_ps_mask)
    EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpeq_pd_mask)
  #endif
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
