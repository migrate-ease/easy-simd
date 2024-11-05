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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN testn

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/testn.h>

static int
test_easysimd_mm_testn_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[16];
    const int8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { {  INT8_C(  79), -INT8_C( 121),  INT8_C(  64),  INT8_C(  13), -INT8_C(  62),  INT8_C(  76), -INT8_C(  38),  INT8_C(  52),
             INT8_MIN, -INT8_C(  77), -INT8_C(   6),  INT8_C(  41), -INT8_C(  91),  INT8_C(  94),  INT8_C( 102),  INT8_C(  70) },
      {  INT8_C(  93),  INT8_C(  49), -INT8_C(  96),  INT8_C(  70),  INT8_C(  76), -INT8_C( 117), -INT8_C(  66), -INT8_C( 108),
        -INT8_C(  40), -INT8_C(  48), -INT8_C(  18),  INT8_C(  61), -INT8_C(  48),  INT8_C(  13), -INT8_C(  88),  INT8_C(  31) },
      UINT16_C(    4) },
    { { -INT8_C( 108), -INT8_C(  24),  INT8_C(  44),  INT8_C(  86),  INT8_C(  52),  INT8_C(   7), -INT8_C( 118), -INT8_C(  75),
        -INT8_C(  70), -INT8_C( 124), -INT8_C(  34),  INT8_C(  95), -INT8_C(  30),  INT8_C(  69), -INT8_C(  91),  INT8_C(  64) },
      {  INT8_C( 118),  INT8_C(  69), -INT8_C( 122), -INT8_C(  62), -INT8_C(  48),  INT8_C(  68),  INT8_C(  86), -INT8_C(  87),
         INT8_C(  20),  INT8_C(  68), -INT8_C(  26), -INT8_C(  27),  INT8_C(  82), -INT8_C( 114),  INT8_C(   4), -INT8_C(  26) },
      UINT16_C(    0) },
    { {  INT8_C( 119),  INT8_C(  49),  INT8_C(  61), -INT8_C(  85),  INT8_C(  56), -INT8_C(  57),  INT8_C(  96), -INT8_C(  14),
         INT8_C(  76),  INT8_C(  63),  INT8_C(  81),  INT8_C(  46), -INT8_C( 124), -INT8_C(  10),  INT8_C( 110), -INT8_C(   6) },
      {  INT8_C(  59), -INT8_C(  12), -INT8_C(  68),  INT8_C(  11),  INT8_C(  57),  INT8_C(  19), -INT8_C(  76),  INT8_C(  77),
         INT8_C(  87), -INT8_C( 101),  INT8_C(  50), -INT8_C(  87),  INT8_C(  41),  INT8_C(  55), -INT8_C( 112), -INT8_C(  96) },
      UINT16_C(20480) },
    { {  INT8_C( 104), -INT8_C(  51),  INT8_C(  76), -INT8_C(  96), -INT8_C( 108), -INT8_C(  84), -INT8_C( 110), -INT8_C(  32),
        -INT8_C(  21), -INT8_C(  29),  INT8_C(  15),  INT8_C( 111), -INT8_C(  39),  INT8_C( 125),  INT8_C( 105),  INT8_C(  20) },
      {  INT8_C( 114),  INT8_C(  38),  INT8_C(  31), -INT8_C(  85),  INT8_C(  57), -INT8_C(  44), -INT8_C(   8), -INT8_C( 112),
         INT8_C( 111),  INT8_C(  43),  INT8_C(  58), -INT8_C( 104),  INT8_C(  98), -INT8_C(  54),  INT8_C(  57), -INT8_C(  54) },
      UINT16_C(32768) },
    { { -INT8_C( 105), -INT8_C( 123),  INT8_C( 106),  INT8_C(  43),  INT8_C(  49), -INT8_C(   4),  INT8_C(  12),  INT8_C(  29),
        -INT8_C(  33),  INT8_C(  27), -INT8_C( 116), -INT8_C(  72), -INT8_C( 104), -INT8_C(  10), -INT8_C(  52),  INT8_C(  10) },
      {  INT8_C(  28), -INT8_C(  21), -INT8_C(  75),  INT8_C(  85), -INT8_C(  65), -INT8_C(  82), -INT8_C(  27),  INT8_C(  46),
        -INT8_C(  39),  INT8_C(  31), -INT8_C(  57),  INT8_C(  59), -INT8_C(  23),  INT8_C(   0),  INT8_C(   5),      INT8_MIN },
      UINT16_C(40960) },
    { { -INT8_C( 123),  INT8_C( 111), -INT8_C(  84), -INT8_C(  74),  INT8_C( 107), -INT8_C(  72), -INT8_C(  45),  INT8_C(  74),
        -INT8_C(  45),  INT8_C(  96),  INT8_C(   2),  INT8_C( 107),  INT8_C(  86), -INT8_C(  50),  INT8_C( 118),  INT8_C( 114) },
      { -INT8_C(  71),  INT8_C(  43), -INT8_C(  57),  INT8_C( 121), -INT8_C(  39), -INT8_C(  84), -INT8_C(  89), -INT8_C(  78),
        -INT8_C(  52),  INT8_C( 110), -INT8_C(  19), -INT8_C(  75),  INT8_C( 110), -INT8_C(  14),  INT8_C(  54), -INT8_C(  13) },
      UINT16_C( 1024) },
    { {  INT8_C(  97), -INT8_C(  30), -INT8_C(  86), -INT8_C(  52), -INT8_C( 102),  INT8_C( 125),  INT8_C(  22),  INT8_C( 109),
        -INT8_C(  35),  INT8_C(  24), -INT8_C(  40),  INT8_C(  51), -INT8_C(  26),  INT8_C(  78), -INT8_C(  91), -INT8_C(  96) },
      {  INT8_C( 122),  INT8_C( 108),  INT8_C(  25),  INT8_C(  83),  INT8_C(  25), -INT8_C(  64),  INT8_C(   6), -INT8_C(  27),
         INT8_C(  47), -INT8_C(  13), -INT8_C( 102), -INT8_C(  99), -INT8_C(  26), -INT8_C(  48), -INT8_C( 111),  INT8_C(  71) },
      UINT16_C(32768) },
    { { -INT8_C(  78),  INT8_C(  59),  INT8_C(  20),  INT8_C(  76), -INT8_C(  72),  INT8_C(  42), -INT8_C(  71), -INT8_C( 106),
         INT8_C(  67), -INT8_C( 110), -INT8_C(  55),  INT8_C(  41), -INT8_C(  32),  INT8_C( 111), -INT8_C(  55),  INT8_C(  90) },
      { -INT8_C(  37), -INT8_C(  30), -INT8_C(  82), -INT8_C(  12), -INT8_C(  93), -INT8_C(  76), -INT8_C(  39), -INT8_C(  46),
        -INT8_C(  89),  INT8_C( 116),  INT8_C( 111), -INT8_C( 115),  INT8_C(  68),  INT8_C(   0), -INT8_C(  43), -INT8_C(   9) },
      UINT16_C( 8192) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_testn_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_testn_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_testn_epi8_mask(a, b);

    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_testn_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[8];
    const int16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT16_C( 27705), -INT16_C(  4326),  INT16_C( 23277), -INT16_C( 32364), -INT16_C( 21551), -INT16_C( 16783), -INT16_C( 23773), -INT16_C(  3013) },
      {  INT16_C( 15971),  INT16_C( 32102), -INT16_C( 31726),  INT16_C( 21154),  INT16_C(  9456),  INT16_C( 23792), -INT16_C(  6909), -INT16_C( 13413) },
      UINT8_C(  4) },
    { { -INT16_C( 19080),  INT16_C( 26042),  INT16_C( 19983), -INT16_C(  7962),  INT16_C( 22522),  INT16_C(  7583), -INT16_C(  9478),  INT16_C( 24081) },
      {  INT16_C( 30488),  INT16_C( 11227),  INT16_C( 32507), -INT16_C(  5251),  INT16_C( 28322), -INT16_C( 22968), -INT16_C(  7341), -INT16_C( 13455) },
      UINT8_C(  0) },
    { {  INT16_C( 11160), -INT16_C( 22736),  INT16_C(  5754),  INT16_C( 29831),  INT16_C(  9837),  INT16_C( 26769), -INT16_C( 24063),  INT16_C(  6598) },
      { -INT16_C( 24295),  INT16_C(  5188), -INT16_C( 15841), -INT16_C( 15617),  INT16_C( 18224), -INT16_C( 31896), -INT16_C(  9942), -INT16_C( 15794) },
      UINT8_C( 32) },
    { {  INT16_C( 32260),  INT16_C( 32361), -INT16_C(  3692),  INT16_C(   498), -INT16_C( 31977),  INT16_C(  6249),  INT16_C( 12069),  INT16_C( 15922) },
      {  INT16_C( 30417), -INT16_C(  4014),  INT16_C( 21048),  INT16_C( 26802),  INT16_C(  6809), -INT16_C( 15125),  INT16_C( 14835), -INT16_C(  1914) },
      UINT8_C(  0) },
    { { -INT16_C(  3913),  INT16_C( 19318),  INT16_C( 27105), -INT16_C(  1971), -INT16_C( 18708),  INT16_C(  4625),  INT16_C( 17382), -INT16_C( 18608) },
      { -INT16_C( 23623), -INT16_C(  3417),  INT16_C( 23285), -INT16_C( 29094),  INT16_C( 18036),  INT16_C( 26706), -INT16_C(  9857),  INT16_C( 14176) },
      UINT8_C(  0) },
    { { -INT16_C( 10551), -INT16_C( 21886), -INT16_C( 12481),  INT16_C( 11426), -INT16_C( 19578),  INT16_C( 27710), -INT16_C( 28938), -INT16_C( 20445) },
      { -INT16_C( 13775),  INT16_C(  9890), -INT16_C(   988), -INT16_C( 26187),  INT16_C(  1858), -INT16_C( 15871),  INT16_C( 25056), -INT16_C( 22023) },
      UINT8_C(  0) },
    { {  INT16_C( 31543),  INT16_C( 30547), -INT16_C(  2485), -INT16_C( 11869), -INT16_C(  7767), -INT16_C( 24515),  INT16_C( 24687), -INT16_C( 24240) },
      { -INT16_C(  3542),  INT16_C( 20423),  INT16_C( 31982),  INT16_C( 12776), -INT16_C(  5756),  INT16_C( 25843), -INT16_C(  5046), -INT16_C( 32498) },
      UINT8_C(  0) },
    { {  INT16_C( 24935), -INT16_C( 19720), -INT16_C( 25769),  INT16_C(   387), -INT16_C( 16260), -INT16_C(  4959), -INT16_C(  3808),  INT16_C( 19341) },
      {  INT16_C( 21731), -INT16_C( 11878), -INT16_C( 32047),  INT16_C( 21762), -INT16_C(  2709), -INT16_C( 19015), -INT16_C( 14367),  INT16_C( 18742) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_testn_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_testn_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_testn_epi16_mask(a, b);

    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_testn_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[4];
    const int32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(   877802841),  INT32_C(   628699196),  INT32_C(   423400341),  INT32_C(   248205649) },
      {  INT32_C(   308663343), -INT32_C(   307203953), -INT32_C(    43350499), -INT32_C(   458700740) },
      UINT8_C(  0) },
    { { -INT32_C(   810592622),  INT32_C(   468985990),  INT32_C(   204746939), -INT32_C(  1340407935) },
      {  INT32_C(  1656913875),  INT32_C(   223310575),  INT32_C(   889895416),  INT32_C(   337228673) },
      UINT8_C(  0) },
    { { -INT32_C(  1880897527), -INT32_C(  1414866960), -INT32_C(  2001215993), -INT32_C(  1321676323) },
      {  INT32_C(  1075051088),  INT32_C(  1682793324), -INT32_C(  1634117860),  INT32_C(   330478090) },
      UINT8_C(  0) },
    { {  INT32_C(  1789039994),  INT32_C(  1930775660),  INT32_C(   150719530), -INT32_C(   289852515) },
      { -INT32_C(  1724986323),  INT32_C(  1291680559), -INT32_C(   571828270), -INT32_C(   990864311) },
      UINT8_C(  0) },
    { { -INT32_C(  1657892047),  INT32_C(   168838367), -INT32_C(  1374549232),  INT32_C(  1805437758) },
      { -INT32_C(   955987305),  INT32_C(   387121477), -INT32_C(   487260776), -INT32_C(   895031911) },
      UINT8_C(  0) },
    { {  INT32_C(  1466422392),  INT32_C(   694253336), -INT32_C(  1059622014), -INT32_C(   701795522) },
      { -INT32_C(  2103627971), -INT32_C(   912674768),  INT32_C(  1185648301), -INT32_C(   351252109) },
      UINT8_C(  0) },
    { {  INT32_C(  1044543269),  INT32_C(  1885840622),  INT32_C(  1446002199), -INT32_C(   299082831) },
      { -INT32_C(  1167013494),  INT32_C(   646121849),  INT32_C(   174861975), -INT32_C(  1510638465) },
      UINT8_C(  0) },
    { { -INT32_C(   505202445), -INT32_C(   212776228),  INT32_C(   961118600),  INT32_C(  1713862108) },
      { -INT32_C(  1222600898),  INT32_C(   954049696),  INT32_C(  1380075986), -INT32_C(  1191757627) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_testn_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_testn_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_testn_epi32_mask(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_testn_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT64_C( 7042599164444798984),  INT64_C( 5494575579314622345) },
      {  INT64_C( 7821332943026021055), -INT64_C( 2408513317554895629) },
      UINT8_C(  0) },
    { { -INT64_C( 7761201841682423220),  INT64_C( 5109492685074390843) },
      { -INT64_C( 1135421686890297101),  INT64_C( 2212545955083294191) },
      UINT8_C(  0) },
    { {  INT64_C(  399436953437216236),  INT64_C( 6899948302643617012) },
      {  INT64_C( 9043350709838373522), -INT64_C( 1228067618916436220) },
      UINT8_C(  0) },
    { { -INT64_C( 3894362618413596892), -INT64_C( 5613449826730207923) },
      { -INT64_C( 4723218575458743705), -INT64_C( 1069452297444669896) },
      UINT8_C(  0) },
    { { -INT64_C( 3753798126241051319), -INT64_C( 7549266419053679767) },
      {  INT64_C( 8530099438231583976), -INT64_C( 3925774928414968122) },
      UINT8_C(  0) },
    { { -INT64_C( 6759676299076737818),  INT64_C( 4305835226101954041) },
      {  INT64_C( 1483724207033306850),  INT64_C(  467858830779696949) },
      UINT8_C(  0) },
    { {  INT64_C( 3344655224767407760), -INT64_C( 2387443283420277425) },
      {  INT64_C( 3515224417468595302),  INT64_C( 3099430490191785949) },
      UINT8_C(  0) },
    { {  INT64_C( 6281498973005114825),  INT64_C(  202055726854210547) },
      {  INT64_C( 4521472607903009684), -INT64_C( 3335691746893126417) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_testn_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_testn_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_testn_epi64_mask(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_testn_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int8_t a[16];
    const int8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(22462),
      {  INT8_C( 112), -INT8_C(  45),  INT8_C(  34),  INT8_C(  63), -INT8_C( 107), -INT8_C( 103),  INT8_C( 110),  INT8_C(  99),
        -INT8_C(  30),  INT8_C(  99),  INT8_C( 117),  INT8_C( 112), -INT8_C(  40), -INT8_C(  32),  INT8_C(   5),  INT8_C(  86) },
      {  INT8_C(  58), -INT8_C(  24),  INT8_C(  12), -INT8_C(  79), -INT8_C(  81), -INT8_C( 116), -INT8_C(  67),  INT8_C(  25),
        -INT8_C(  91),  INT8_C(  93),  INT8_C( 111), -INT8_C(  32), -INT8_C(  58),  INT8_C(  46),  INT8_C(  55),  INT8_C(  54) },
      UINT16_C(    4) },
    { UINT16_C(22785),
      {  INT8_C( 117), -INT8_C( 106), -INT8_C(  13), -INT8_C(  29), -INT8_C(   6), -INT8_C(  43),  INT8_C(  70),  INT8_C( 111),
         INT8_C(  69),  INT8_C(  30),  INT8_C(  80),  INT8_C(  74),  INT8_C( 116), -INT8_C( 118),  INT8_C(  50), -INT8_C( 127) },
      {  INT8_C(  59), -INT8_C(  31),  INT8_C(  13), -INT8_C(   7), -INT8_C(   5), -INT8_C(  77),  INT8_C(  86),  INT8_C( 106),
        -INT8_C( 109),  INT8_C(  28), -INT8_C( 104), -INT8_C(  53),  INT8_C(  82), -INT8_C( 102),  INT8_C(  36), -INT8_C(  57) },
      UINT16_C(    0) },
    { UINT16_C( 5936),
      { -INT8_C(  85),  INT8_C(  42), -INT8_C(  20), -INT8_C(  15), -INT8_C( 102),  INT8_C(  49),  INT8_C(  16), -INT8_C(  22),
         INT8_C( 123), -INT8_C( 124),  INT8_C( 116), -INT8_C(  83),  INT8_C(   5), -INT8_C(  81), -INT8_C( 113),  INT8_C(  19) },
      { -INT8_C(  88), -INT8_C( 118), -INT8_C(  58), -INT8_C(   2), -INT8_C(  12),  INT8_C(  89),  INT8_C(  26), -INT8_C( 115),
         INT8_C(  36),  INT8_C( 108),  INT8_C(  39),  INT8_C(  73),  INT8_C(  52),  INT8_C(  87),  INT8_C(  96), -INT8_C(  33) },
      UINT16_C(    0) },
    { UINT16_C(19842),
      { -INT8_C(  48),  INT8_C(  28),  INT8_C( 126), -INT8_C(  32),  INT8_C(   6), -INT8_C(   6),  INT8_C( 101),  INT8_C( 122),
        -INT8_C(  89),  INT8_C( 106),  INT8_C(  41),  INT8_C(  54),  INT8_C( 125), -INT8_C(  46), -INT8_C(  64),  INT8_C(  67) },
      { -INT8_C(  48), -INT8_C(  75), -INT8_C(  99), -INT8_C(  21),  INT8_C(  66), -INT8_C(  63),  INT8_C(  87),  INT8_C( 105),
         INT8_C(  10), -INT8_C( 117), -INT8_C(  64),  INT8_C( 107),  INT8_C( 106),  INT8_C(  66), -INT8_C(  72),  INT8_C(  59) },
      UINT16_C( 1024) },
    { UINT16_C(13918),
      {  INT8_C(  27),  INT8_C( 100),  INT8_C(  48),      INT8_MIN, -INT8_C(  34), -INT8_C(  40), -INT8_C(  21),  INT8_C(   8),
         INT8_C(  14),  INT8_C( 104), -INT8_C(  38), -INT8_C(  49), -INT8_C(  84), -INT8_C(  86), -INT8_C( 124),  INT8_C(  73) },
      { -INT8_C( 107), -INT8_C(  58),  INT8_C(  10), -INT8_C(  19),  INT8_C(  47),  INT8_C(  21),  INT8_C( 120), -INT8_C(  17),
             INT8_MIN, -INT8_C(  29),  INT8_C(  50),  INT8_C(  56),  INT8_C(  30), -INT8_C( 112),  INT8_C( 110),  INT8_C(  57) },
      UINT16_C(    4) },
    { UINT16_C(40949),
      { -INT8_C(  70), -INT8_C(  45),  INT8_C( 119), -INT8_C(  91), -INT8_C(  37), -INT8_C( 123),  INT8_C(  13), -INT8_C(  75),
         INT8_C(  84), -INT8_C(  71),  INT8_C(  96), -INT8_C(  40),  INT8_C(   2), -INT8_C(  11), -INT8_C(  98),  INT8_C(  13) },
      { -INT8_C(  30), -INT8_C(  51),  INT8_C(  34),  INT8_C(  91), -INT8_C(  67), -INT8_C(  94),  INT8_C(  62), -INT8_C(  17),
        -INT8_C(  38),  INT8_C(  92),      INT8_MAX,  INT8_C(  72), -INT8_C( 107),  INT8_C( 116), -INT8_C(  25),  INT8_C(  79) },
      UINT16_C( 4096) },
    { UINT16_C(24136),
      { -INT8_C(  12),  INT8_C(  35), -INT8_C(  28),  INT8_C(   2), -INT8_C(  39),  INT8_C(  56), -INT8_C(  69),  INT8_C(  57),
         INT8_C(  17), -INT8_C(  66),  INT8_C(  46), -INT8_C(  81), -INT8_C(  53),  INT8_C(  17),  INT8_C( 125), -INT8_C(  19) },
      {  INT8_C( 108),  INT8_C(  58), -INT8_C( 113), -INT8_C(  86),  INT8_C(  41),  INT8_C( 105),  INT8_C(   6), -INT8_C(  88),
        -INT8_C(  79), -INT8_C( 101),  INT8_C(  29), -INT8_C( 103), -INT8_C(  21),  INT8_C( 101), -INT8_C(   9), -INT8_C(  33) },
      UINT16_C(    0) },
    { UINT16_C(56200),
      { -INT8_C(  31),  INT8_C(  97),  INT8_C(  20), -INT8_C(  99), -INT8_C( 102),  INT8_C(  37),  INT8_C(  91), -INT8_C(  55),
        -INT8_C(  44),  INT8_C(  38), -INT8_C(  38),  INT8_C(  81),  INT8_C(  19),  INT8_C(  70), -INT8_C( 117), -INT8_C(  94) },
      { -INT8_C(  16), -INT8_C(  76),  INT8_C(  11), -INT8_C(  10),  INT8_C(  93), -INT8_C(  68), -INT8_C( 111),  INT8_C( 122),
         INT8_C(  85),  INT8_C( 124), -INT8_C(  33),  INT8_C(  77),  INT8_C(  92),  INT8_C( 103),  INT8_C(  40),  INT8_C(  61) },
      UINT16_C(    0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_testn_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_testn_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 r = easysimd_mm_mask_testn_epi8_mask(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_testn_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
        { UINT8_C(118),
      {  INT16_C( 15567),  INT16_C( 18478), -INT16_C( 14664), -INT16_C(   299), -INT16_C( 15678),  INT16_C( 27049),  INT16_C( 31492), -INT16_C( 30286) },
      { -INT16_C( 13101),  INT16_C(  9896), -INT16_C( 20993),  INT16_C( 10993),  INT16_C( 12913), -INT16_C( 14705), -INT16_C( 30072),  INT16_C( 22332) },
      UINT8_C(  0) },
    { UINT8_C(198),
      { -INT16_C( 24725),  INT16_C( 12670),  INT16_C( 31860),  INT16_C( 14067),  INT16_C( 23589), -INT16_C( 24262), -INT16_C( 15602), -INT16_C(  9356) },
      { -INT16_C( 26005),  INT16_C(  6618),  INT16_C(  1420), -INT16_C( 16758),  INT16_C( 20628),  INT16_C(  7750), -INT16_C( 25203), -INT16_C(  1819) },
      UINT8_C(  0) },
    { UINT8_C( 60),
      {  INT16_C( 10595), -INT16_C(  8016), -INT16_C(  6628),  INT16_C( 30981), -INT16_C( 23007), -INT16_C(  7033),  INT16_C( 25114), -INT16_C( 19120) },
      {  INT16_C( 26941),  INT16_C( 16961), -INT16_C(    13),  INT16_C( 17622), -INT16_C(  2746), -INT16_C(  7215), -INT16_C( 13862),  INT16_C( 15648) },
      UINT8_C(  0) },
    { UINT8_C(242),
      {  INT16_C(  7632), -INT16_C( 18674), -INT16_C( 30941), -INT16_C( 13864), -INT16_C( 17393),  INT16_C( 29156), -INT16_C( 26356),  INT16_C( 30126) },
      { -INT16_C(  3878), -INT16_C(  9879), -INT16_C( 21049), -INT16_C( 17377),  INT16_C(   894),  INT16_C( 18326), -INT16_C( 11485), -INT16_C(  3271) },
      UINT8_C(  0) },
    { UINT8_C(241),
      { -INT16_C( 21945), -INT16_C( 12524), -INT16_C(  8830),  INT16_C( 16350),  INT16_C( 20417),  INT16_C( 23115), -INT16_C( 15874), -INT16_C(  4556) },
      {  INT16_C(  3626), -INT16_C( 10315),  INT16_C( 28973),  INT16_C( 12373), -INT16_C( 25593), -INT16_C(  9389),  INT16_C( 18389),  INT16_C(  7372) },
      UINT8_C(  0) },
    { UINT8_C(241),
      { -INT16_C(  5152), -INT16_C( 17036), -INT16_C( 19511),  INT16_C(  6527), -INT16_C(  9730), -INT16_C( 16617),  INT16_C(  1294),  INT16_C(  7401) },
      { -INT16_C( 16197),  INT16_C( 11337),  INT16_C( 31253), -INT16_C( 20172),  INT16_C(  4045),  INT16_C(  5254), -INT16_C( 23589), -INT16_C( 17658) },
      UINT8_C(  0) },
    { UINT8_C(142),
      {  INT16_C( 30842),  INT16_C( 11608),  INT16_C( 29175), -INT16_C( 11989), -INT16_C(  5240), -INT16_C( 29217), -INT16_C(  1068), -INT16_C( 27320) },
      {  INT16_C( 30020), -INT16_C( 16726),  INT16_C( 23721), -INT16_C( 18292), -INT16_C( 24350), -INT16_C( 31341),  INT16_C( 20134),  INT16_C(  8212) },
      UINT8_C(128) },
    { UINT8_C(198),
      {  INT16_C( 19820), -INT16_C(  8770), -INT16_C( 28807),  INT16_C( 25701), -INT16_C(  3474),  INT16_C( 26936), -INT16_C( 12997), -INT16_C( 20307) },
      {  INT16_C( 27768), -INT16_C( 11175),  INT16_C(  4600), -INT16_C( 26442),  INT16_C( 15524), -INT16_C(  3521),  INT16_C( 24400), -INT16_C( 17224) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_testn_epi16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_testn_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 r = easysimd_mm_mask_testn_epi16_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_testn_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(143),
      {  INT32_C(  1945315797),  INT32_C(  1181632960),  INT32_C(   776858020), -INT32_C(  1413804614) },
      {  INT32_C(   749223106), -INT32_C(  2069353001), -INT32_C(  1857442727),  INT32_C(  1210106739) },
      UINT8_C(  0) },
    { UINT8_C(237),
      {  INT32_C(  1554889491),  INT32_C(   302117673),  INT32_C(  1322069825),  INT32_C(   655390955) },
      {  INT32_C(  1308507168), -INT32_C(  1851358492), -INT32_C(  1895548981),  INT32_C(  1786530903) },
      UINT8_C(  0) },
    { UINT8_C(  7),
      {  INT32_C(   489735977), -INT32_C(   144817208), -INT32_C(  1998410736), -INT32_C(   123205188) },
      { -INT32_C(  1965230841), -INT32_C(   749376100), -INT32_C(  1121196943), -INT32_C(  1966828192) },
      UINT8_C(  0) },
    { UINT8_C( 92),
      {  INT32_C(   941926389), -INT32_C(  1320674299),  INT32_C(   141414654),  INT32_C(  1829725560) },
      { -INT32_C(  1358325439), -INT32_C(   752820753), -INT32_C(  1657479672), -INT32_C(  1745174878) },
      UINT8_C(  0) },
    { UINT8_C(101),
      {  INT32_C(   980078366), -INT32_C(   415687913), -INT32_C(   312524408), -INT32_C(   366031536) },
      { -INT32_C(  1277567530), -INT32_C(   591680258), -INT32_C(  1635886624),  INT32_C(  1896027731) },
      UINT8_C(  0) },
    { UINT8_C(229),
      { -INT32_C(  1996641171),  INT32_C(   638641381),  INT32_C(   293010756),  INT32_C(   166158380) },
      { -INT32_C(   435709127), -INT32_C(  1345920170),  INT32_C(  2013422946), -INT32_C(   698518424) },
      UINT8_C(  0) },
    { UINT8_C( 32),
      {  INT32_C(  1057316442),  INT32_C(  1820535663),  INT32_C(    43553953),  INT32_C(   373006971) },
      { -INT32_C(  1922227799),  INT32_C(  1307516136), -INT32_C(  1816828129),  INT32_C(   531860420) },
      UINT8_C(  0) },
    { UINT8_C(234),
      { -INT32_C(   480682312),  INT32_C(  1971635681),  INT32_C(    15763038), -INT32_C(   475461694) },
      { -INT32_C(  1865730444), -INT32_C(  1934682075),  INT32_C(  1498497742), -INT32_C(  1388089099) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_testn_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_testn_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 r = easysimd_mm_mask_testn_epi32_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}


static int
test_easysimd_mm_mask_testn_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t b[2];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(133),
      { -INT64_C( 5077839427542936190), -INT64_C( 3500499475789550180) },
      {  INT64_C( 7678198118418894311), -INT64_C(  555896076234334087) },
      UINT8_C(  0) },
    { UINT8_C(164),
      { -INT64_C( 6657077580810309859),  INT64_C( 5048737463043395182) },
      { -INT64_C( 9146755173588659418),  INT64_C( 4245469607285543990) },
      UINT8_C(  0) },
    { UINT8_C(159),
      {  INT64_C( 7439235036376021888), -INT64_C( 8629275714633243295) },
      { -INT64_C( 5054820103990160016), -INT64_C( 8560480833380147293) },
      UINT8_C(  0) },
    { UINT8_C( 76),
      { -INT64_C(  296520051708307448),  INT64_C( 1010487717130205933) },
      { -INT64_C( 6957871755076327646),  INT64_C( 4602171083484365954) },
      UINT8_C(  0) },
    { UINT8_C(139),
      { -INT64_C( 7098487052649316349), -INT64_C( 4636486501673856372) },
      { -INT64_C( 1655671111146310185),  INT64_C( 8738496715027421064) },
      UINT8_C(  0) },
    { UINT8_C( 32),
      { -INT64_C( 7418166876095444634), -INT64_C( 5768706252826635722) },
      {  INT64_C( 1159253014642717871),  INT64_C( 8943867483740623262) },
      UINT8_C(  0) },
    { UINT8_C(  9),
      { -INT64_C( 7027820990661853470), -INT64_C(  578000604180737317) },
      {  INT64_C( 8086354553906170139),  INT64_C( 3967674935446812167) },
      UINT8_C(  0) },
    { UINT8_C(  1),
      { -INT64_C( 8960285353624897119),  INT64_C( 8369374668465303939) },
      {  INT64_C( 3311699301386993517), -INT64_C( 5976072144557966325) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_testn_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_testn_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {

    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 r = easysimd_mm_mask_testn_epi64_mask(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_testn_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[16];
    const int16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { -INT16_C( 27187),  INT16_C( 22747), -INT16_C( 22135), -INT16_C(  7118), -INT16_C( 28747), -INT16_C( 18972), -INT16_C( 28131), -INT16_C( 30573),
        -INT16_C( 14670),  INT16_C(  6951), -INT16_C( 21022), -INT16_C( 17294), -INT16_C( 25728), -INT16_C( 13117), -INT16_C( 16708), -INT16_C( 21842) },
      {  INT16_C( 27186), -INT16_C( 22748),  INT16_C( 22134), -INT16_C(  2933),  INT16_C( 28746),  INT16_C( 26793),  INT16_C( 15362), -INT16_C( 19216),
        -INT16_C( 25854), -INT16_C(  6952),  INT16_C( 19017), -INT16_C(  4447),  INT16_C( 25727),  INT16_C( 15291),  INT16_C( 25379),  INT16_C( 21841) },
      UINT16_C(37399) },
    { { -INT16_C( 30966), -INT16_C(   382),  INT16_C( 16528), -INT16_C( 31059), -INT16_C(  1015), -INT16_C(  7078), -INT16_C(  8865),  INT16_C( 11650),
         INT16_C( 32594),  INT16_C(  7793), -INT16_C( 24419),  INT16_C(  4945),  INT16_C( 22510), -INT16_C( 12692), -INT16_C( 22240),  INT16_C( 10950) },
      { -INT16_C( 13776),  INT16_C(   381), -INT16_C( 16529),  INT16_C( 31058), -INT16_C(  8773),  INT16_C(  7077),  INT16_C( 10170),  INT16_C(  3400),
        -INT16_C( 18010), -INT16_C( 20949),  INT16_C( 24418), -INT16_C(  4946), -INT16_C( 22511),  INT16_C( 12691),  INT16_C( 22865), -INT16_C( 32420) },
      UINT16_C(15406) },
    { {  INT16_C( 22289), -INT16_C( 11730), -INT16_C(  9212), -INT16_C(  2056),  INT16_C( 20868),  INT16_C( 29985), -INT16_C( 23894), -INT16_C( 12713),
        -INT16_C(  9604),  INT16_C(  5218),  INT16_C( 28335),  INT16_C( 25192), -INT16_C( 10464), -INT16_C(  1329),  INT16_C( 18830), -INT16_C( 10318) },
      { -INT16_C( 22290), -INT16_C(  3376), -INT16_C( 29052),  INT16_C(  2055),  INT16_C( 19935), -INT16_C( 29986),  INT16_C( 14064),  INT16_C( 27736),
        -INT16_C( 17904), -INT16_C( 16256), -INT16_C(  5847),  INT16_C( 18722), -INT16_C(  3648),  INT16_C( 20291), -INT16_C( 17094),  INT16_C( 10317) },
      UINT16_C(33321) },
    { {  INT16_C( 23958),  INT16_C( 21207),  INT16_C( 32100),  INT16_C( 31341), -INT16_C(  8214),  INT16_C(   342), -INT16_C( 28260),  INT16_C(   588),
         INT16_C(  5284),  INT16_C( 23531), -INT16_C(  8825), -INT16_C( 30681), -INT16_C( 11351), -INT16_C( 19372),  INT16_C(  2229),  INT16_C( 19368) },
      { -INT16_C( 23959), -INT16_C( 25113), -INT16_C( 32101), -INT16_C( 31342), -INT16_C( 10911), -INT16_C(   343), -INT16_C(  2714),  INT16_C(  5376),
        -INT16_C(  5285), -INT16_C(  7567),  INT16_C( 22729),  INT16_C( 30680),  INT16_C( 11350), -INT16_C( 15829), -INT16_C(  2230), -INT16_C( 19369) },
      UINT16_C(55725) },
    { { -INT16_C(   387), -INT16_C( 11113),  INT16_C( 28670),  INT16_C( 21579),  INT16_C( 30619),  INT16_C(  8461),  INT16_C( 28014),  INT16_C(  1946),
        -INT16_C(  8824),  INT16_C( 27708),  INT16_C(   230), -INT16_C( 29903), -INT16_C( 20901), -INT16_C( 31530), -INT16_C(  9010),  INT16_C( 10923) },
      {  INT16_C(   386), -INT16_C( 32338), -INT16_C(  1680),  INT16_C(  3285), -INT16_C(  5008), -INT16_C(  8462), -INT16_C( 29607),  INT16_C(  1510),
         INT16_C(  8823),  INT16_C( 18033), -INT16_C(   231),  INT16_C( 29902),  INT16_C( 30381),  INT16_C( 31529), -INT16_C( 28078), -INT16_C( 10924) },
      UINT16_C(44321) },
    { { -INT16_C(  9760), -INT16_C(  1291),  INT16_C(  6886), -INT16_C( 27679),  INT16_C(  2704), -INT16_C(  7410), -INT16_C( 27657),  INT16_C( 12472),
         INT16_C(  3685),  INT16_C( 19505),  INT16_C( 25539),  INT16_C( 31995),  INT16_C( 14667),  INT16_C(   424),  INT16_C( 14409), -INT16_C(  7580) },
      {  INT16_C(  9759),  INT16_C(  1290), -INT16_C(  5312), -INT16_C( 11880), -INT16_C( 22794), -INT16_C( 27724),  INT16_C( 27656),  INT16_C( 28099),
        -INT16_C(  1926), -INT16_C( 19506), -INT16_C( 25540), -INT16_C( 31996), -INT16_C( 14668), -INT16_C(   425),  INT16_C( 12798),  INT16_C(  7579) },
      UINT16_C(48707) },
    { {  INT16_C( 10816),  INT16_C(  4534),  INT16_C( 17263), -INT16_C(  4855), -INT16_C(  4991),  INT16_C( 14957), -INT16_C( 12153),  INT16_C( 30109),
        -INT16_C( 18416), -INT16_C( 27248),  INT16_C( 30330), -INT16_C( 20550),  INT16_C( 30433),  INT16_C(  7236),  INT16_C(  6721),  INT16_C(  5663) },
      { -INT16_C( 10817),  INT16_C(    39), -INT16_C( 17264),  INT16_C(  4854), -INT16_C(  5720), -INT16_C( 14958),  INT16_C( 12152), -INT16_C( 21702),
         INT16_C( 18415),  INT16_C( 27247), -INT16_C( 17731),  INT16_C( 20549), -INT16_C( 30434),  INT16_C( 24428), -INT16_C( 29789),  INT16_C( 25461) },
      UINT16_C( 7021) },
    { { -INT16_C( 10680), -INT16_C( 31372),  INT16_C(  7882), -INT16_C(  8489), -INT16_C( 13733), -INT16_C( 28659), -INT16_C( 32178),  INT16_C( 10590),
        -INT16_C( 16892),  INT16_C( 30624),  INT16_C( 15130), -INT16_C(  8095),  INT16_C( 15673), -INT16_C(   338),  INT16_C( 12191), -INT16_C( 19399) },
      { -INT16_C( 11798),  INT16_C( 31371), -INT16_C(  7883),  INT16_C(  8488),  INT16_C( 13732), -INT16_C(  3407), -INT16_C( 23369), -INT16_C( 10591),
         INT16_C( 16891), -INT16_C( 21683), -INT16_C( 15131),  INT16_C(  8094),  INT16_C( 19458),  INT16_C( 16413), -INT16_C( 12192),  INT16_C( 19398) },
      UINT16_C(52638) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_testn_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_testn_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[16];
    int16_t b_[16];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 16 ; j++)
      if (easysimd_test_codegen_random_i8() & 1)
        a_[j] = ~b_[j];

    easysimd__m256i a = easysimd_mm256_loadu_epi16(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(b_);
    easysimd__mmask16 r = easysimd_mm256_testn_epi16_mask(a, b);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_testn_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(  1317048103), -INT32_C(  1052505433),  INT32_C(   320403176), -INT32_C(  1830205483),  INT32_C(  2098139176),  INT32_C(  2031216042), -INT32_C(   291529423), -INT32_C(   917736619) },
      {  INT32_C(   924555148),  INT32_C(  1052505432), -INT32_C(   320403177),  INT32_C(  2038354000),  INT32_C(  1744211681), -INT32_C(  2031216043),  INT32_C(  1232375479),  INT32_C(   917736618) },
      UINT8_C(166) },
    { { -INT32_C(   509867693), -INT32_C(   275865405),  INT32_C(   700117716), -INT32_C(   995120883),  INT32_C(  1125306099), -INT32_C(   736640592),  INT32_C(  1414563216),  INT32_C(  1743256084) },
      {  INT32_C(   509867692),  INT32_C(   275865404),  INT32_C(  1597582418),  INT32_C(   203679957), -INT32_C(  1125306100), -INT32_C(  2104490766),  INT32_C(   936829065), -INT32_C(  1743256085) },
      UINT8_C(147) },
    { {  INT32_C(  1736306898),  INT32_C(  1983229802),  INT32_C(     9354790), -INT32_C(  1336622554), -INT32_C(   540383217), -INT32_C(   290151319), -INT32_C(  1030204985),  INT32_C(  1501801221) },
      { -INT32_C(  1736306899), -INT32_C(  1626425410), -INT32_C(     9354791), -INT32_C(  1498429949),  INT32_C(   540383216),  INT32_C(   290151318),  INT32_C(  1030204984), -INT32_C(  2020152743) },
      UINT8_C(117) },
    { {  INT32_C(  1923663982), -INT32_C(   885505606), -INT32_C(  2053157394),  INT32_C(   392802479), -INT32_C(   920237390), -INT32_C(   941118059),  INT32_C(  1361661206), -INT32_C(  1558322340) },
      {  INT32_C(  1218834105),  INT32_C(   885505605), -INT32_C(  1262918133), -INT32_C(   392802480),  INT32_C(   920237389),  INT32_C(   941118058), -INT32_C(  1361661207),  INT32_C(  1558322339) },
      UINT8_C(250) },
    { {  INT32_C(   553991377),  INT32_C(  1100683921),  INT32_C(    42371895),  INT32_C(   519251765), -INT32_C(   485615927),  INT32_C(   838088039),  INT32_C(  2084535606), -INT32_C(   216733454) },
      { -INT32_C(  1948462337), -INT32_C(  1100683922), -INT32_C(    42371896),  INT32_C(  1964797339),  INT32_C(   485615926), -INT32_C(   838088040),  INT32_C(   826952264),  INT32_C(   216733453) },
      UINT8_C(182) },
    { { -INT32_C(   650902826), -INT32_C(    72960909),  INT32_C(   116086338), -INT32_C(  2065802436), -INT32_C(  1296750171), -INT32_C(   981932755),  INT32_C(   360863080),  INT32_C(   880682333) },
      {  INT32_C(   650902825),  INT32_C(    72960908), -INT32_C(   116086339), -INT32_C(  1451363580),  INT32_C(  1549546012),  INT32_C(   981932754), -INT32_C(  1051784848), -INT32_C(   880682334) },
      UINT8_C(167) },
    { {  INT32_C(   988851114), -INT32_C(   491893024), -INT32_C(   790036995), -INT32_C(  2012582120),  INT32_C(   457792121), -INT32_C(  2039555171), -INT32_C(   539558142), -INT32_C(   758656985) },
      { -INT32_C(   988851115),  INT32_C(   491893023), -INT32_C(   957531730),  INT32_C(   575600553), -INT32_C(  2059561135),  INT32_C(  2039555170),  INT32_C(   539558141),  INT32_C(   758656984) },
      UINT8_C(227) },
    { {  INT32_C(   644721789), -INT32_C(   179782492), -INT32_C(  1250196141), -INT32_C(  1490088790), -INT32_C(   673960353),  INT32_C(   169017223),  INT32_C(  1957175485), -INT32_C(   974754744) },
      { -INT32_C(  1108650983),  INT32_C(  1672622864),  INT32_C(  1662594489),  INT32_C(  1175078969),  INT32_C(   673960352),  INT32_C(   405945809), -INT32_C(  1957175486),  INT32_C(  1699757644) },
      UINT8_C( 80) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_testn_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_testn_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[8];
    int32_t b_[8];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 8 ; j++)
      if (easysimd_test_codegen_random_i8() & 1)
        a_[j] = ~b_[j];

    easysimd__m256i a = easysimd_mm256_loadu_epi32(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(b_);
    easysimd__mmask8 r = easysimd_mm256_testn_epi32_mask(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm256_testn_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { {  INT64_C( 3578554183208484545), -INT64_C( 3564703118603852296),  INT64_C( 5563595533409103557),  INT64_C( 3791736709128723150) },
      { -INT64_C( 3578554183208484546),  INT64_C( 3564703118603852295), -INT64_C( 5563595533409103558), -INT64_C( 3791736709128723151) },
      UINT8_C( 15) },
    { {  INT64_C(  124875281609084202), -INT64_C(  213726888905151384),  INT64_C( 7434353208686671241),  INT64_C( 8553226627376713003) },
      { -INT64_C( 1810055749143413640),  INT64_C(  213726888905151383), -INT64_C( 3420791075336584124),  INT64_C( 6644917097154001269) },
      UINT8_C(  2) },
    { { -INT64_C( 6127811188253254334), -INT64_C( 8151555045596182428),  INT64_C( 8901555530903182673), -INT64_C( 3242939165045615234) },
      { -INT64_C( 7707154704643719888),  INT64_C( 8151555045596182427), -INT64_C( 2588512383710788351),  INT64_C( 3242939165045615233) },
      UINT8_C( 10) },
    { {  INT64_C( 8002961748416960618), -INT64_C( 1587209672310312609), -INT64_C( 1310182806851735107), -INT64_C( 5335034933172326705) },
      { -INT64_C( 2735735032157683895), -INT64_C( 2519705708300342808),  INT64_C( 1310182806851735106),  INT64_C( 5335034933172326704) },
      UINT8_C( 12) },
    { { -INT64_C( 3863830463506852598),  INT64_C( 2349556018954415220), -INT64_C( 5401197756279622718), -INT64_C( 1906477455660555051) },
      {  INT64_C( 3863830463506852597), -INT64_C( 2349556018954415221),  INT64_C( 5401197756279622717),  INT64_C( 1906477455660555050) },
      UINT8_C( 15) },
    { {  INT64_C( 6601750141561765513),  INT64_C( 8896846796839692847), -INT64_C(  733534987545627509), -INT64_C( 7628830158126823904) },
      { -INT64_C( 6601750141561765514), -INT64_C( 8896846796839692848), -INT64_C( 6735683537292587584),  INT64_C(   21313524849940050) },
      UINT8_C(  3) },
    { { -INT64_C( 3845970484521166176), -INT64_C( 5566524798793571474),  INT64_C( 7264003674745357766),  INT64_C( 3985599602792602020) },
      {  INT64_C( 3845970484521166175),  INT64_C( 7000602196855480388),  INT64_C(  911777062194936935),  INT64_C( 1968416892694104929) },
      UINT8_C(  1) },
    { {  INT64_C( 6355138441039699319),  INT64_C( 4380089100480895007),  INT64_C( 3417668727623567905), -INT64_C( 7608038693603686295) },
      {  INT64_C( 2288817073458119598),  INT64_C( 3237083937329479131), -INT64_C( 3417668727623567906),  INT64_C( 7348820947921986888) },
      UINT8_C(  4) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_testn_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_testn_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[4];
    int64_t b_[4];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 4 ; j++)
      if (easysimd_test_codegen_random_i8() & 1)
        a_[j] = ~b_[j];

    easysimd__m256i a = easysimd_mm256_loadu_epi64(a_);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(b_);
    easysimd__mmask8 r = easysimd_mm256_testn_epi64_mask(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_testn_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { {  INT16_C( 24035), -INT16_C(  4069), -INT16_C( 14348),  INT16_C( 22307),  INT16_C( 21629), -INT16_C(  2263), -INT16_C( 22037), -INT16_C( 23340),
         INT16_C(  3287),  INT16_C( 19731),  INT16_C( 28795),  INT16_C( 29560),  INT16_C( 11694),  INT16_C(  1818),  INT16_C(  5311),  INT16_C( 28699),
         INT16_C( 25695),  INT16_C( 11765), -INT16_C(  8600), -INT16_C( 25174),  INT16_C(  6194),  INT16_C(  6804),  INT16_C( 26767), -INT16_C( 13634),
        -INT16_C( 11646), -INT16_C( 32123), -INT16_C( 28435), -INT16_C( 25546),  INT16_C( 20925),  INT16_C( 11427), -INT16_C(  5294),  INT16_C( 15279) },
      { -INT16_C( 16305),  INT16_C( 31236), -INT16_C( 20954),  INT16_C( 32023), -INT16_C( 21630),  INT16_C(  4503),  INT16_C( 22036), -INT16_C( 26917),
        -INT16_C(  3288),  INT16_C(  5465), -INT16_C( 28796),  INT16_C( 16817),  INT16_C( 21728),  INT16_C( 13165), -INT16_C(  5312), -INT16_C( 28700),
        -INT16_C(  5716), -INT16_C( 11766),  INT16_C(  8599),  INT16_C(  6479), -INT16_C(  6195), -INT16_C(  7894),  INT16_C(  1597),  INT16_C( 25975),
        -INT16_C( 12039),  INT16_C( 32122),  INT16_C( 11359),  INT16_C( 16575),  INT16_C( 11392), -INT16_C( 16269),  INT16_C( 22296), -INT16_C( 15280) },
      UINT32_C(2184627536) },
    { {  INT16_C( 27039),  INT16_C( 28809),  INT16_C( 23411), -INT16_C( 28863), -INT16_C(  4608),  INT16_C(  8884), -INT16_C( 21665), -INT16_C(   194),
        -INT16_C( 10484),  INT16_C( 19298), -INT16_C(  8401), -INT16_C( 19528),  INT16_C(  2943),  INT16_C( 13719), -INT16_C( 20299), -INT16_C( 19113),
         INT16_C( 19111), -INT16_C( 31695), -INT16_C(  2513),  INT16_C( 12078), -INT16_C(  7452),  INT16_C( 29266),  INT16_C( 30567),  INT16_C(  4721),
        -INT16_C(  3420), -INT16_C( 27084), -INT16_C(  6389), -INT16_C( 32439), -INT16_C( 15117), -INT16_C( 20546),  INT16_C(  5450), -INT16_C(  3739) },
      { -INT16_C( 27040), -INT16_C( 28810), -INT16_C( 23412),  INT16_C( 28862),  INT16_C(  4231), -INT16_C( 29469),  INT16_C( 21664),  INT16_C(  1950),
        -INT16_C( 11627), -INT16_C( 19299), -INT16_C(  6210), -INT16_C( 18122), -INT16_C(  2944), -INT16_C( 13720), -INT16_C( 13047),  INT16_C( 27068),
         INT16_C( 12900), -INT16_C(  3848), -INT16_C( 18474),  INT16_C( 23905),  INT16_C( 17607),  INT16_C( 26858), -INT16_C( 30568),  INT16_C( 11631),
         INT16_C(  3419),  INT16_C(  6626),  INT16_C(  6388),  INT16_C( 29906),  INT16_C( 15116),  INT16_C(  5439), -INT16_C(  1272),  INT16_C( 27775) },
      UINT32_C( 356528735) },
    { {  INT16_C( 21256), -INT16_C( 29229),  INT16_C(  3813), -INT16_C(  6146),  INT16_C( 23641),  INT16_C( 11650),  INT16_C( 18986),  INT16_C(  1115),
         INT16_C( 23856),  INT16_C(  4507),  INT16_C(  5822),  INT16_C(  4996), -INT16_C( 29051),  INT16_C(  1604),  INT16_C( 15562),  INT16_C(  2851),
        -INT16_C(  2152),  INT16_C( 16537), -INT16_C( 21637), -INT16_C( 11216), -INT16_C( 19961),  INT16_C( 12545),  INT16_C( 24060), -INT16_C(  4042),
         INT16_C( 23009), -INT16_C( 16730),  INT16_C( 10783), -INT16_C( 22091), -INT16_C(  1355), -INT16_C( 18001), -INT16_C( 29825), -INT16_C(  7424) },
      { -INT16_C( 21257),  INT16_C( 29228),  INT16_C( 23639),  INT16_C( 24391),  INT16_C( 18447),  INT16_C(  2960), -INT16_C( 14683), -INT16_C( 22788),
        -INT16_C( 23857), -INT16_C(  4508),  INT16_C( 14029),  INT16_C( 24485),  INT16_C( 29050), -INT16_C(  1605), -INT16_C( 17412), -INT16_C(  2852),
         INT16_C(  2151), -INT16_C( 16538), -INT16_C( 21147),  INT16_C( 29726), -INT16_C( 20746), -INT16_C( 25729),  INT16_C( 31605),  INT16_C( 17473),
        -INT16_C( 23010), -INT16_C(  5325), -INT16_C( 10020),  INT16_C( 22090),  INT16_C(  1354),  INT16_C( 18000),  INT16_C( 11456),  INT16_C( 10042) },
      UINT32_C( 956543747) },
    { {  INT16_C( 12260),  INT16_C( 12870), -INT16_C( 19332), -INT16_C(  1877), -INT16_C(  1773), -INT16_C(  5618), -INT16_C( 16842),  INT16_C( 20415),
        -INT16_C(  2403), -INT16_C( 27932), -INT16_C( 24104),  INT16_C( 23925), -INT16_C( 21415), -INT16_C( 24575),  INT16_C( 19104), -INT16_C( 31668),
         INT16_C(   845), -INT16_C( 13723),  INT16_C( 11750), -INT16_C( 13373), -INT16_C( 16885), -INT16_C(   590), -INT16_C( 32131),  INT16_C(  6801),
         INT16_C(  8679),  INT16_C( 26313),  INT16_C( 11804),  INT16_C( 30127), -INT16_C(  3621),  INT16_C( 29500), -INT16_C(  1899), -INT16_C(  7323) },
      {  INT16_C( 21926), -INT16_C( 29512),  INT16_C( 19331),  INT16_C(  1876),  INT16_C(  1772),  INT16_C(  2308),  INT16_C( 16841), -INT16_C( 20416),
         INT16_C(  2402),  INT16_C( 25111),  INT16_C( 24103), -INT16_C( 23926),  INT16_C( 21414),  INT16_C( 14781),  INT16_C( 23655),  INT16_C(  3652),
        -INT16_C(   846),  INT16_C( 13722), -INT16_C(  4537),  INT16_C( 13372),  INT16_C( 16884), -INT16_C( 16835),  INT16_C( 32130), -INT16_C(  6802),
        -INT16_C( 31354), -INT16_C( 20921), -INT16_C( 11805), -INT16_C( 30128),  INT16_C(  3620), -INT16_C( 29501),  INT16_C(  1898),  INT16_C(  7322) },
      UINT32_C(4242218462) },
    { { -INT16_C( 26885), -INT16_C(  3759), -INT16_C( 10388),  INT16_C( 25482), -INT16_C(  8563), -INT16_C(  8647), -INT16_C( 30174), -INT16_C(  5242),
        -INT16_C(  2700),  INT16_C( 22168), -INT16_C( 12365),  INT16_C( 32714), -INT16_C( 13879), -INT16_C( 13386),  INT16_C(  2659), -INT16_C( 19054),
         INT16_C( 19859), -INT16_C(    66),  INT16_C( 18725),  INT16_C( 32740), -INT16_C( 25202),  INT16_C(  1955), -INT16_C( 17984),  INT16_C( 13759),
        -INT16_C( 20800),  INT16_C( 29471), -INT16_C(  5762), -INT16_C(   235),  INT16_C( 22649),  INT16_C( 13300), -INT16_C(  7245),  INT16_C( 25159) },
      {  INT16_C( 26884), -INT16_C(  2492),  INT16_C( 10387), -INT16_C( 25483), -INT16_C( 17568),  INT16_C(  8646), -INT16_C(  9081),  INT16_C( 18410),
         INT16_C(  2699),  INT16_C(  2491), -INT16_C( 20749), -INT16_C( 22960),  INT16_C( 25431),  INT16_C(  3004),  INT16_C(   838),  INT16_C( 19053),
        -INT16_C( 19860),  INT16_C(    65), -INT16_C( 18726),  INT16_C( 15004),  INT16_C( 25201), -INT16_C(  1956),  INT16_C( 17983), -INT16_C( 13760),
        -INT16_C(  1200),  INT16_C( 17619),  INT16_C(  9129),  INT16_C(   234), -INT16_C( 22650), -INT16_C( 13301),  INT16_C( 31146),  INT16_C(  5911) },
      UINT32_C( 955744557) },
    { { -INT16_C( 13488), -INT16_C(  6894), -INT16_C( 19965), -INT16_C( 22171), -INT16_C(  8466), -INT16_C( 26078),  INT16_C(  7318), -INT16_C(  4111),
         INT16_C(  9839),  INT16_C( 16152),  INT16_C( 23964), -INT16_C(  5213), -INT16_C( 17909),  INT16_C( 22421),  INT16_C( 24997),  INT16_C( 31834),
         INT16_C( 27714), -INT16_C( 10654), -INT16_C( 14402),  INT16_C( 31729),  INT16_C(  5034), -INT16_C( 27626), -INT16_C( 32614), -INT16_C( 22394),
        -INT16_C( 25554), -INT16_C( 10549), -INT16_C( 30160), -INT16_C( 25413),  INT16_C( 19902),  INT16_C( 13661),  INT16_C( 17332), -INT16_C(  2350) },
      {  INT16_C( 13487),  INT16_C( 28109),  INT16_C( 19964), -INT16_C( 24295), -INT16_C( 26933),  INT16_C( 26077),  INT16_C( 25366),  INT16_C(  4110),
        -INT16_C(  9840), -INT16_C( 16153), -INT16_C( 23965),  INT16_C(  5212),  INT16_C( 17908), -INT16_C( 22422),  INT16_C( 15497),  INT16_C( 14495),
         INT16_C( 27761),  INT16_C( 28070), -INT16_C( 16455), -INT16_C( 31730), -INT16_C(  5035),  INT16_C( 27625), -INT16_C(  2225), -INT16_C(  8324),
         INT16_C( 25553),  INT16_C( 13471), -INT16_C(  1275), -INT16_C(  1720), -INT16_C( 19903), -INT16_C( 13662),  INT16_C( 16879),  INT16_C( 24578) },
      UINT32_C( 825769893) },
    { { -INT16_C( 28367), -INT16_C( 12472), -INT16_C(  6461), -INT16_C( 26445),  INT16_C(  4838), -INT16_C( 12907),  INT16_C( 20560),  INT16_C( 17185),
         INT16_C( 32442), -INT16_C( 16560),  INT16_C( 22026),  INT16_C(  7907),  INT16_C(  4134),  INT16_C( 22409), -INT16_C( 21555), -INT16_C(  4107),
         INT16_C( 31634),  INT16_C( 21956),  INT16_C( 29606), -INT16_C( 29335), -INT16_C( 32313), -INT16_C( 10406), -INT16_C( 15847),  INT16_C( 18245),
         INT16_C( 21599),  INT16_C( 17881), -INT16_C( 16992), -INT16_C( 14492),  INT16_C(  9415), -INT16_C( 29666),  INT16_C(  5325),  INT16_C( 25258) },
      {  INT16_C( 28366),  INT16_C( 12471), -INT16_C( 26720),  INT16_C( 26444), -INT16_C(  4839),  INT16_C( 12906), -INT16_C( 20561),  INT16_C( 25210),
        -INT16_C( 32443),  INT16_C( 12576), -INT16_C( 19430), -INT16_C(  7908), -INT16_C(  4135), -INT16_C( 22410),  INT16_C(  8258),  INT16_C(  4106),
        -INT16_C( 15729),  INT16_C( 12096), -INT16_C( 29607),  INT16_C( 29334),  INT16_C(   121),  INT16_C( 10405),  INT16_C(  8112), -INT16_C(  2678),
        -INT16_C( 21600), -INT16_C( 17882),  INT16_C( 16991),  INT16_C( 14491),  INT16_C(  4657),  INT16_C( 29665), -INT16_C(  5326), -INT16_C( 15997) },
      UINT32_C(1865202043) },
    { {  INT16_C( 10624), -INT16_C( 12225), -INT16_C( 17093), -INT16_C( 30434), -INT16_C( 30249),  INT16_C(  5231), -INT16_C( 25596), -INT16_C( 25604),
        -INT16_C( 17563),  INT16_C( 27057), -INT16_C( 15055),  INT16_C(   329), -INT16_C( 32244),  INT16_C( 19161), -INT16_C( 19905), -INT16_C( 16177),
         INT16_C(  3803), -INT16_C( 29552), -INT16_C( 20584),  INT16_C(  9206), -INT16_C( 13540), -INT16_C( 19022), -INT16_C(  9280), -INT16_C( 16673),
         INT16_C( 12951), -INT16_C( 13937), -INT16_C( 29956),  INT16_C( 23278), -INT16_C( 14160),  INT16_C(   677),  INT16_C( 28438), -INT16_C( 14378) },
      { -INT16_C( 25475),  INT16_C( 17278),  INT16_C( 17092),  INT16_C( 30433), -INT16_C( 27787),  INT16_C( 17707),  INT16_C(  2871),  INT16_C( 25603),
         INT16_C( 17562), -INT16_C( 27058),  INT16_C( 15054), -INT16_C(  9569),  INT16_C( 16286), -INT16_C( 19162),  INT16_C( 12974),  INT16_C( 11175),
         INT16_C(  9678), -INT16_C( 27793),  INT16_C( 20583), -INT16_C(  9207),  INT16_C( 13539),  INT16_C(  6689),  INT16_C(  9279), -INT16_C(  9858),
        -INT16_C( 12952),  INT16_C( 13936),  INT16_C(  3847), -INT16_C( 23279),  INT16_C( 14159), -INT16_C(   678),  INT16_C(   362),  INT16_C( 14377) },
      UINT32_C(3143378828) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_testn_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_testn_epi16_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int16_t a_[32];
    int16_t b_[32];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 32 ; j++)
      if (easysimd_test_codegen_random_i8() & 1)
        a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi16(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(b_);
    easysimd__mmask32 r = easysimd_mm512_testn_epi16_mask(a, b);

    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_testn_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { -INT32_C(   398957831), -INT32_C(   196229814),  INT32_C(   228853395), -INT32_C(  1459879865),  INT32_C(   311601822),  INT32_C(   736949667), -INT32_C(   314789545), -INT32_C(  1389276492),
         INT32_C(   525913394),  INT32_C(  1358160687),  INT32_C(  1888039285), -INT32_C(   949023866), -INT32_C(  1642587269), -INT32_C(   242661123), -INT32_C(   305074887), -INT32_C(   191233729) },
      { -INT32_C(  1695338389), -INT32_C(  1813322221), -INT32_C(   228853396), -INT32_C(   407254524), -INT32_C(   311601823),  INT32_C(   365960924),  INT32_C(  1258463499),  INT32_C(   222272930),
        -INT32_C(   525913395), -INT32_C(  2072801000), -INT32_C(  1888039286), -INT32_C(    92917863),  INT32_C(  1642587268),  INT32_C(   242661122), -INT32_C(    78022312), -INT32_C(   485976042) },
      UINT16_C(13588) },
    { {  INT32_C(  1822045701), -INT32_C(  1882822858), -INT32_C(   158720032),  INT32_C(  1397369401), -INT32_C(  2073846718), -INT32_C(   665794093), -INT32_C(   307090174),  INT32_C(   843058812),
        -INT32_C(  1848187638), -INT32_C(  1723217619), -INT32_C(  2039423793), -INT32_C(  1324922513), -INT32_C(  1187681562), -INT32_C(   661551821),  INT32_C(  1030592728), -INT32_C(  1562392299) },
      { -INT32_C(  1822045702),  INT32_C(  1208858745),  INT32_C(    63864212), -INT32_C(  1397369402),  INT32_C(  1919281727), -INT32_C(  1605699984),  INT32_C(   307090173), -INT32_C(  1766576997),
         INT32_C(  1848187637),  INT32_C(  1723217618),  INT32_C(  2053801139), -INT32_C(  1742332071),  INT32_C(  2030734345), -INT32_C(  2145758078), -INT32_C(  1030592729), -INT32_C(  1990703212) },
      UINT16_C(17225) },
    { {  INT32_C(   463810456), -INT32_C(  1415856252),  INT32_C(   324346229), -INT32_C(   716913036),  INT32_C(   859750403),  INT32_C(  1086224809), -INT32_C(   881295094), -INT32_C(   134951073),
        -INT32_C(  2095932929), -INT32_C(  2144424615), -INT32_C(  2107218976), -INT32_C(  1754738284),  INT32_C(  1003202449),  INT32_C(  1975850703),  INT32_C(  1576531212), -INT32_C(  1763408465) },
      {  INT32_C(   599904970),  INT32_C(  1520693119), -INT32_C(   324346230),  INT32_C(   446025566), -INT32_C(   859750404), -INT32_C(  1086224810),  INT32_C(   881295093),  INT32_C(  2037532335),
        -INT32_C(  1734528232), -INT32_C(  1376632798),  INT32_C(  2107218975),  INT32_C(  1754738283), -INT32_C(  1003202450), -INT32_C(   746359330), -INT32_C(  1576531213),  INT32_C(  1763408464) },
      UINT16_C(56436) },
    { {  INT32_C(  1302828672), -INT32_C(    13239796), -INT32_C(   633697923),  INT32_C(  1723710904), -INT32_C(   578930042),  INT32_C(  1116124338),  INT32_C(  1855186760),  INT32_C(   576411553),
         INT32_C(   560989166),  INT32_C(   927228723),  INT32_C(  1485629088), -INT32_C(   225869717), -INT32_C(   372242501),  INT32_C(  1945898409), -INT32_C(  1002340062), -INT32_C(   924344871) },
      { -INT32_C(  1302828673),  INT32_C(  1844092108),  INT32_C(   633697922), -INT32_C(   952204097),  INT32_C(   578930041), -INT32_C(    74089006), -INT32_C(  1855186761), -INT32_C(   576411554),
        -INT32_C(   560989167),  INT32_C(   457931161), -INT32_C(  1354690321),  INT32_C(   225869716),  INT32_C(   372242500),  INT32_C(   739427701),  INT32_C(  1002340061),  INT32_C(   924344870) },
      UINT16_C(55765) },
    { {  INT32_C(  1066299976), -INT32_C(   439336953),  INT32_C(   912117210),  INT32_C(  1343961585), -INT32_C(    58316004),  INT32_C(   573373072), -INT32_C(  1196762777),  INT32_C(   949689098),
         INT32_C(   863124333), -INT32_C(   951407351),  INT32_C(   749431355),  INT32_C(   449791229),  INT32_C(    18306416), -INT32_C(   796157408),  INT32_C(   141676317),  INT32_C(   926686921) },
      { -INT32_C(  1066299977),  INT32_C(  1971829562), -INT32_C(   912117211), -INT32_C(   263381255),  INT32_C(    58316003), -INT32_C(   573373073),  INT32_C(  1196762776), -INT32_C(   866974443),
        -INT32_C(   863124334), -INT32_C(  1539173249),  INT32_C(  1080943687), -INT32_C(   449791230), -INT32_C(    18306417),  INT32_C(  1205582767), -INT32_C(   141676318), -INT32_C(   926686922) },
      UINT16_C(55669) },
    { {  INT32_C(  1126089095), -INT32_C(   395046534), -INT32_C(  1054848117), -INT32_C(   762649811), -INT32_C(   949964529), -INT32_C(  1320828682), -INT32_C(   395431434),  INT32_C(   342430029),
         INT32_C(   580710823), -INT32_C(   821213629), -INT32_C(  1081070190), -INT32_C(   581940275),  INT32_C(   803047481),  INT32_C(  1674063154),  INT32_C(  1614046095), -INT32_C(   239531959) },
      { -INT32_C(  1126089096),  INT32_C(   395046533),  INT32_C(   752229147),  INT32_C(  1869637430),  INT32_C(   949964528), -INT32_C(   998801552),  INT32_C(   795188942), -INT32_C(  1490281938),
        -INT32_C(   580710824),  INT32_C(   418705148),  INT32_C(  1061472777),  INT32_C(   581940274),  INT32_C(  1935298050), -INT32_C(  1674063155), -INT32_C(  1614046096),  INT32_C(   239531958) },
      UINT16_C(59667) },
    { { -INT32_C(    96252943), -INT32_C(    39276627), -INT32_C(  1874573856),  INT32_C(   890666179), -INT32_C(  1564746844), -INT32_C(  1755239700), -INT32_C(  1686879864), -INT32_C(  1380944944),
        -INT32_C(   128436020),  INT32_C(  1011724635), -INT32_C(  1160998409), -INT32_C(  1276058866),  INT32_C(  2050742414), -INT32_C(  1961780222),  INT32_C(  2057900383),  INT32_C(   176689767) },
      {  INT32_C(    96252942),  INT32_C(    39276626),  INT32_C(  1874573855), -INT32_C(   890666180),  INT32_C(  1564746843), -INT32_C(  1612163647), -INT32_C(   766701862), -INT32_C(   224941596),
        -INT32_C(  1325968547), -INT32_C(  1011724636),  INT32_C(  1160998408),  INT32_C(  1276058865),  INT32_C(    44651585), -INT32_C(  2069785943), -INT32_C(  2057900384), -INT32_C(   176689768) },
      UINT16_C(52767) },
    { { -INT32_C(  1303423100),  INT32_C(   957788225), -INT32_C(   850389076),  INT32_C(  1805928384),  INT32_C(  2042720823),  INT32_C(   466529877), -INT32_C(  2100110966),  INT32_C(  1656477171),
         INT32_C(   974804984), -INT32_C(  1267453689),  INT32_C(   159565128),  INT32_C(   688478175), -INT32_C(  1439625388), -INT32_C(  1733758925),  INT32_C(   999257944),  INT32_C(   295933464) },
      {  INT32_C(  1303423099), -INT32_C(   957788226),  INT32_C(   850389075), -INT32_C(  1805928385), -INT32_C(  2042720824),  INT32_C(  1708000140),  INT32_C(   154245275), -INT32_C(  1728553187),
        -INT32_C(   974804985),  INT32_C(  1267453688), -INT32_C(   159565129), -INT32_C(  1148528397), -INT32_C(   230569883),  INT32_C(  1733758924), -INT32_C(   999257945), -INT32_C(   295933465) },
      UINT16_C(59167) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_testn_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_testn_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int32_t a_[16];
    int32_t b_[16];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 16 ; j++)
      if (easysimd_test_codegen_random_i8() & 1)
        a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi32(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(b_);
    easysimd__mmask16 r = easysimd_mm512_testn_epi32_mask(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_testn_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { {  INT64_C( 4508187367566017372), -INT64_C( 6569918542864078877), -INT64_C( 8962991843626193926),  INT64_C( 5940106029300099442),
        -INT64_C( 3474806220522527646), -INT64_C( 8461692430822565366),  INT64_C(  206077686324757234), -INT64_C( 8406745884122882184) },
      {  INT64_C( 5939611601085195705), -INT64_C( 7245777598779645353), -INT64_C( 2024139227130431915),  INT64_C( 7972417749947520834),
         INT64_C( 3474806220522527645), -INT64_C( 6708805109161690720), -INT64_C( 1406406580617291393),  INT64_C( 8406745884122882183) },
      UINT8_C(144) },
    { {  INT64_C(  502440011780718091), -INT64_C( 6158853987843597074),  INT64_C( 8480346530472994281),  INT64_C( 9054718441418449049),
        -INT64_C( 5262948987930223368),  INT64_C( 5817434754285894735),  INT64_C( 5038994455620387643), -INT64_C( 6410086071321220720) },
      { -INT64_C( 6256883783212495956),  INT64_C( 6158853987843597073), -INT64_C( 8480346530472994282),  INT64_C( 1831974856728808061),
         INT64_C( 5262948987930223367), -INT64_C( 5797282151545048028), -INT64_C( 5038994455620387644),  INT64_C( 6410086071321220719) },
      UINT8_C(214) },
    { { -INT64_C(  576688130814031434), -INT64_C( 1611003909097220691), -INT64_C(  859982337976997018),  INT64_C(   63488577835525727),
        -INT64_C( 8877278934273448285), -INT64_C( 5966482186928774029),  INT64_C( 5625287344307916414), -INT64_C( 8025167225870642222) },
      { -INT64_C( 1521877185384271226), -INT64_C( 2333405148534386816),  INT64_C( 8338992646982506338), -INT64_C(   63488577835525728),
         INT64_C( 4130150199787816355),  INT64_C( 5966482186928774028),  INT64_C( 8637754373134207434), -INT64_C( 4212515870200188911) },
      UINT8_C( 40) },
    { { -INT64_C( 5107500865354007352), -INT64_C( 7172652405436439845),  INT64_C( 7466936592040059958),  INT64_C( 5156385719971833639),
        -INT64_C( 1148429196945830273),  INT64_C( 5187442117424490317), -INT64_C( 3727061135569862247),  INT64_C( 7493839429109749816) },
      {  INT64_C( 5107500865354007351),  INT64_C( 7172652405436439844), -INT64_C( 7466936592040059959), -INT64_C( 5156385719971833640),
         INT64_C( 9086446945411128794), -INT64_C( 6401783450134545048),  INT64_C( 3727061135569862246), -INT64_C( 7493839429109749817) },
      UINT8_C(207) },
    { { -INT64_C( 5798513945331171839),  INT64_C( 5428960175976437193),  INT64_C( 6757813930750603122), -INT64_C( 5112048783095745732),
        -INT64_C( 4372678601869856680),  INT64_C( 3629998252239901186), -INT64_C( 6484313291659159634), -INT64_C( 2344923696908176739) },
      { -INT64_C( 1777860332040727659),  INT64_C( 2933009366244146152),  INT64_C( 7209367014820676603), -INT64_C( 2433034198997195791),
         INT64_C( 6831330025702076531), -INT64_C( 3629998252239901187),  INT64_C( 3470093055996161706),  INT64_C( 2344923696908176738) },
      UINT8_C(160) },
    { { -INT64_C( 6908294271050297472),  INT64_C( 8921913121712221078), -INT64_C( 6420116432106797287),  INT64_C( 1432456675824479355),
        -INT64_C( 3610282089748642993), -INT64_C( 1185021928708486906),  INT64_C( 7223262427560527820), -INT64_C(  752967942511614581) },
      { -INT64_C( 2734774782049771346), -INT64_C( 8921913121712221079),  INT64_C(  936150044919129132), -INT64_C( 1432456675824479356),
        -INT64_C( 2111564278826051324),  INT64_C( 5789745815574619276), -INT64_C( 7223262427560527821),  INT64_C(  166602392970991908) },
      UINT8_C( 74) },
    { {  INT64_C( 5016976028114287712), -INT64_C( 1200294191847835227), -INT64_C( 3025277623662505428), -INT64_C( 2013804801763072228),
         INT64_C( 3712337795058573632), -INT64_C(  823518641492363816), -INT64_C( 6729914387693378463), -INT64_C( 7263218271959429465) },
      { -INT64_C( 5016976028114287713), -INT64_C(  598996799044366445),  INT64_C( 3025277623662505427), -INT64_C( 8440351624730350469),
         INT64_C( 2915826315157796827), -INT64_C( 7425147411344599705), -INT64_C( 5178316874887116031),  INT64_C( 7263218271959429464) },
      UINT8_C(133) },
    { { -INT64_C( 3473694996123600275), -INT64_C(   57162890175005889),  INT64_C( 3713233947962996891), -INT64_C( 5822670248827836014),
         INT64_C( 7390313696471364338), -INT64_C( 2518613618650359962), -INT64_C( 1270988866395681138),  INT64_C( 7504489525265874484) },
      {  INT64_C( 1805020628100122495), -INT64_C( 7913049926615503344), -INT64_C( 1095754884881632895), -INT64_C( 5425558594338232362),
        -INT64_C( 6458612020008002158),  INT64_C( 4101134079099438164), -INT64_C( 7780565674210515367), -INT64_C( 8525150306086917177) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_testn_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_testn_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    int64_t a_[8];
    int64_t b_[8];
    easysimd_test_codegen_random_memory(sizeof(a_), HEDLEY_REINTERPRET_CAST(uint8_t*, a_));
    easysimd_test_codegen_random_memory(sizeof(b_), HEDLEY_REINTERPRET_CAST(uint8_t*, b_));

    for (size_t j = 0 ; j < 8 ; j++)
      if (easysimd_test_codegen_random_i8() & 1)
        a_[j] = ~b_[j];

    easysimd__m512i a = easysimd_mm512_loadu_epi64(a_);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(b_);
    easysimd__mmask8 r = easysimd_mm512_testn_epi64_mask(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_testn_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_testn_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_testn_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_testn_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_testn_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_testn_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_testn_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_testn_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_testn_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_testn_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_testn_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_testn_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_testn_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_testn_epi64_mask)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
