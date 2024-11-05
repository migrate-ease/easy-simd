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

#define EASYSIMD_TEST_X86_AVX512_INSN cmpneq

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/cmpneq.h>
#include <easysimd/x86/avx512/blend.h>

static int
test_easysimd_mm_cmpneq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[16];
    const int8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { {  INT8_C(  15),  INT8_C( 112), -INT8_C(  32),  INT8_C(  43), -INT8_C(  96),  INT8_C(  81),  INT8_C( 113), -INT8_C(  57),
         INT8_C(  36),  INT8_C(   4), -INT8_C(  56),  INT8_C(  19),  INT8_C(  21),      INT8_MAX,  INT8_C(  99),  INT8_C( 120) },
      { -INT8_C(  86),  INT8_C( 112), -INT8_C(  32),  INT8_C(   5), -INT8_C(  96),  INT8_C(  81), -INT8_C(  59),  INT8_C( 102),
         INT8_C(  36), -INT8_C( 125), -INT8_C(  56),  INT8_C(  49),  INT8_C(  21),  INT8_C(  89), -INT8_C( 118),  INT8_C( 120) },
      UINT16_C(27337) },
    { {  INT8_C(  67),  INT8_C( 106), -INT8_C(  69), -INT8_C(  76),  INT8_C(  49), -INT8_C(  33), -INT8_C(  71), -INT8_C(   7),
        -INT8_C(  13), -INT8_C(  50),  INT8_C( 120),  INT8_C(  86),  INT8_C(  70),  INT8_C(  35), -INT8_C(  57), -INT8_C(   2) },
      {  INT8_C(  67), -INT8_C(  85), -INT8_C(  69), -INT8_C(  19),  INT8_C(  17), -INT8_C(  33),  INT8_C( 112), -INT8_C(   7),
        -INT8_C(  13), -INT8_C(  50),  INT8_C(  38),  INT8_C(  86), -INT8_C( 111),  INT8_C(  35), -INT8_C(  57), -INT8_C(   2) },
      UINT16_C( 5210) },
    { { -INT8_C( 119), -INT8_C( 117), -INT8_C(  13),  INT8_C(  66), -INT8_C( 124), -INT8_C(  26),  INT8_C(  17), -INT8_C(   3),
         INT8_C(  61),  INT8_C(  87),  INT8_C(  32),  INT8_C(   4),  INT8_C(  85),  INT8_C(  72), -INT8_C(  81), -INT8_C(  36) },
      {  INT8_C(  54), -INT8_C( 117), -INT8_C(  13), -INT8_C(  90), -INT8_C( 124), -INT8_C(  26),  INT8_C(  17), -INT8_C(  76),
         INT8_C(  95), -INT8_C(  79), -INT8_C(  92), -INT8_C(  72), -INT8_C( 122), -INT8_C(   2), -INT8_C(  81),  INT8_C(  16) },
      UINT16_C(49033) },
    { {  INT8_C(  82),  INT8_C(  13), -INT8_C(  90),  INT8_C(  99),  INT8_C(  10), -INT8_C(  29), -INT8_C(  69),  INT8_C(  42),
        -INT8_C(  25),  INT8_C(  16),  INT8_C( 115), -INT8_C( 105), -INT8_C(  20), -INT8_C(  87),  INT8_C(  87), -INT8_C(   6) },
      {  INT8_C(  79),  INT8_C(  13), -INT8_C(  90),  INT8_C(  99), -INT8_C( 103), -INT8_C(  29),  INT8_C(  33),  INT8_C(  61),
        -INT8_C( 125),  INT8_C(  16),  INT8_C(  59), -INT8_C( 105), -INT8_C(  73), -INT8_C(  60),  INT8_C(  87),  INT8_C(  10) },
      UINT16_C(46545) },
    { {  INT8_C( 109), -INT8_C(  36), -INT8_C( 104),  INT8_C(  40),  INT8_C(   6),      INT8_MAX,  INT8_C(  57),  INT8_C( 121),
         INT8_C(  22),  INT8_C(  37),  INT8_C(  34),  INT8_C( 110),  INT8_C(  32),  INT8_C( 114),  INT8_C(  83), -INT8_C( 116) },
      { -INT8_C(  31), -INT8_C(  36),  INT8_C(  87),  INT8_C(  40),  INT8_C(  41),      INT8_MAX, -INT8_C(  86),  INT8_C( 100),
         INT8_C(  22),  INT8_C(  97),  INT8_C(  40),  INT8_C( 110),  INT8_C(  32),  INT8_C( 114),  INT8_C(  83), -INT8_C(  39) },
      UINT16_C(34517) },
    { {  INT8_C(   1), -INT8_C(  36),  INT8_C(   6),  INT8_C(  58),  INT8_C(  85),  INT8_C(  28),  INT8_C(  96),  INT8_C( 120),
        -INT8_C( 118),      INT8_MIN, -INT8_C(  22), -INT8_C(  35),  INT8_C(  12), -INT8_C(  53), -INT8_C(  55),  INT8_C(  99) },
      { -INT8_C(  50), -INT8_C(  36),  INT8_C(   6),  INT8_C( 120),  INT8_C(  85),  INT8_C( 105),  INT8_C(  96),  INT8_C( 120),
        -INT8_C(  94),      INT8_MIN,  INT8_C( 120), -INT8_C( 111),  INT8_C(  30), -INT8_C(  53), -INT8_C(  55),  INT8_C(  99) },
      UINT16_C( 7465) },
    { {  INT8_C(  90),      INT8_MAX,  INT8_C(  58), -INT8_C(  70), -INT8_C(   9), -INT8_C(  60),  INT8_C(  58), -INT8_C(  31),
        -INT8_C(  94),  INT8_C(  70), -INT8_C(  84),  INT8_C( 107), -INT8_C(  87),  INT8_C( 122),  INT8_C(  94), -INT8_C(  24) },
      { -INT8_C(  14),      INT8_MAX,  INT8_C(  58), -INT8_C(  70), -INT8_C(   9), -INT8_C(  13),  INT8_C(  58), -INT8_C(  85),
        -INT8_C( 124),  INT8_C(  47), -INT8_C(  84),  INT8_C( 107),  INT8_C(  78),  INT8_C(  34), -INT8_C(  71), -INT8_C(  88) },
      UINT16_C(62369) },
    { {  INT8_C(  98), -INT8_C( 104), -INT8_C(  72), -INT8_C( 100),  INT8_C( 121),  INT8_C(  90), -INT8_C(  30),  INT8_C(  37),
        -INT8_C(  59), -INT8_C( 116), -INT8_C(  96),  INT8_C(  35),  INT8_C( 116), -INT8_C( 110), -INT8_C(  40), -INT8_C(  59) },
      {  INT8_C(  94), -INT8_C( 104), -INT8_C(  72),  INT8_C( 111),  INT8_C( 121),  INT8_C(  90), -INT8_C(  30),  INT8_C(  37),
        -INT8_C(  59), -INT8_C(  19), -INT8_C(  48), -INT8_C( 110), -INT8_C( 107),  INT8_C( 113), -INT8_C(  40), -INT8_C(  59) },
      UINT16_C(15881) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i8x16());
    easysimd__mmask16 r = easysimd_mm_cmpneq_epi8_mask(a, b);

    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const int8_t a[16];
    const int8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(43285),
      {  INT8_C( 119),  INT8_C(  46), -INT8_C(  81), -INT8_C(  25), -INT8_C( 126),  INT8_C(  57), -INT8_C(  39),  INT8_C(  89),
         INT8_C(  78), -INT8_C(  40), -INT8_C(  34),  INT8_C(  75), -INT8_C(  57), -INT8_C(  90),  INT8_C(  94),  INT8_C(  70) },
      {  INT8_C( 119),  INT8_C(  77), -INT8_C(  95), -INT8_C(  25), -INT8_C( 126),  INT8_C(  57), -INT8_C(  39),  INT8_C(  66),
         INT8_C(  78),  INT8_C( 112), -INT8_C(  34),  INT8_C(  75), -INT8_C( 106), -INT8_C(  90),  INT8_C(  94),  INT8_C(  70) },
      UINT16_C(    4) },
    { UINT16_C( 2293),
      {  INT8_C(  76), -INT8_C(  49),  INT8_C(  97), -INT8_C( 102), -INT8_C(  89),  INT8_C(  64), -INT8_C(  27),  INT8_C( 110),
        -INT8_C(  26),  INT8_C(  67), -INT8_C(  76), -INT8_C(  96), -INT8_C( 111),  INT8_C(  85),  INT8_C( 119),  INT8_C(  88) },
      { -INT8_C(  59), -INT8_C(  49), -INT8_C( 102), -INT8_C( 102), -INT8_C(  89), -INT8_C(  35),  INT8_C(  57),  INT8_C(  18),
        -INT8_C(  26), -INT8_C( 100),  INT8_C(  32), -INT8_C(  96), -INT8_C( 111),  INT8_C(  22),  INT8_C( 119),  INT8_C(  88) },
      UINT16_C(  229) },
    { UINT16_C(35989),
      {  INT8_C( 102),  INT8_C( 122), -INT8_C(   6),  INT8_C(  76), -INT8_C(  67), -INT8_C(  82), -INT8_C(  20),  INT8_C(  78),
         INT8_C(   4),  INT8_C(  99), -INT8_C(  89), -INT8_C(  55),  INT8_C( 111),  INT8_C(  65),  INT8_C(  71), -INT8_C(  21) },
      {  INT8_C( 102),      INT8_MIN, -INT8_C(   3),  INT8_C(  76),  INT8_C(  29), -INT8_C(  82),  INT8_C(  16), -INT8_C(  52),
         INT8_C(  52), -INT8_C(  43), -INT8_C(  57),  INT8_C(  25), -INT8_C(   5),  INT8_C(  65),  INT8_C(  71),  INT8_C(  97) },
      UINT16_C(35988) },
    { UINT16_C(37805),
      {  INT8_C(  77), -INT8_C( 103), -INT8_C(  30),  INT8_C(  81), -INT8_C(   4), -INT8_C( 119),  INT8_C(  26),  INT8_C( 107),
        -INT8_C(  54),  INT8_C(  98),  INT8_C(  86), -INT8_C(  23), -INT8_C(  30),  INT8_C(  83),  INT8_C(  62), -INT8_C(   1) },
      {  INT8_C( 113), -INT8_C( 103), -INT8_C(  53),  INT8_C(  81), -INT8_C(   4), -INT8_C( 119), -INT8_C(  66),  INT8_C(  30),
        -INT8_C(  54),  INT8_C(  99),  INT8_C(  86), -INT8_C(  60),  INT8_C(   2),  INT8_C(  44),  INT8_C(  62), -INT8_C(   1) },
      UINT16_C( 4741) },
    { UINT16_C(49569),
      { -INT8_C(  61), -INT8_C(  68),  INT8_C(  44), -INT8_C( 115),  INT8_C(  30), -INT8_C( 126),  INT8_C( 119),  INT8_C(   0),
        -INT8_C(  42), -INT8_C(  75),  INT8_C(   0),  INT8_C(  71),  INT8_C(   3), -INT8_C(  53), -INT8_C(  19),  INT8_C(  39) },
      {  INT8_C(  94), -INT8_C(  85),  INT8_C(  69),  INT8_C(  76),  INT8_C(  15), -INT8_C(  59),  INT8_C( 119),  INT8_C(  17),
        -INT8_C(  15), -INT8_C(  75),  INT8_C(  97),  INT8_C(  71),  INT8_C(   3),  INT8_C(   3), -INT8_C(  19),  INT8_C( 102) },
      UINT16_C(33185) },
    { UINT16_C(56819),
      {  INT8_C(  39),  INT8_C( 106), -INT8_C(  35), -INT8_C(   3),  INT8_C(  31), -INT8_C(  35),  INT8_C(  69),  INT8_C(  35),
        -INT8_C(  87),  INT8_C(  50),  INT8_C(  74),  INT8_C(   7), -INT8_C(  35), -INT8_C( 113),  INT8_C(  83), -INT8_C(  20) },
      {  INT8_C(  39),  INT8_C( 106), -INT8_C(   2), -INT8_C(   3), -INT8_C(  51),  INT8_C(  95),  INT8_C(  69),  INT8_C(  35),
        -INT8_C(  87),  INT8_C(  50), -INT8_C(  42),  INT8_C(  33),  INT8_C(  26), -INT8_C( 113), -INT8_C(   2),  INT8_C(  66) },
      UINT16_C(56368) },
    { UINT16_C(21567),
      { -INT8_C(  71), -INT8_C( 124),  INT8_C( 119),  INT8_C(  98), -INT8_C(  74), -INT8_C(  63),  INT8_C( 105), -INT8_C( 108),
         INT8_C(  80), -INT8_C(  67),      INT8_MIN, -INT8_C(  91),  INT8_C(  33),  INT8_C( 126), -INT8_C(  21), -INT8_C(  17) },
      { -INT8_C(  71), -INT8_C(  24),  INT8_C(  95),  INT8_C(  64),  INT8_C(  93),  INT8_C(  54),  INT8_C(  98), -INT8_C( 108),
         INT8_C(   0),  INT8_C(  96), -INT8_C(  70),  INT8_C(  52),  INT8_C(  60), -INT8_C(   7), -INT8_C( 120), -INT8_C(  10) },
      UINT16_C(21566) },
    { UINT16_C(13400),
      { -INT8_C(  64), -INT8_C(  62), -INT8_C(  56),  INT8_C(  17),      INT8_MAX,  INT8_C(  73), -INT8_C(  74), -INT8_C(  96),
        -INT8_C(  57), -INT8_C(  95), -INT8_C( 113), -INT8_C(  91), -INT8_C( 119), -INT8_C(  17), -INT8_C(  26), -INT8_C(  26) },
      {  INT8_C(  37), -INT8_C(  62),  INT8_C(  94),  INT8_C(  17), -INT8_C(  88),  INT8_C(  24),  INT8_C(  89), -INT8_C(  27),
         INT8_C(  18), -INT8_C(  95), -INT8_C(  37), -INT8_C( 112), -INT8_C( 119), -INT8_C(  17), -INT8_C(  26), -INT8_C(  94) },
      UINT16_C( 1104) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epi8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i8x16());
    easysimd__mmask16 r = easysimd_mm_mask_cmpneq_epi8_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpneq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t a[16];
    const uint8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { UINT8_C( 40), UINT8_C(174), UINT8_C(150), UINT8_C(231), UINT8_C( 66), UINT8_C( 19), UINT8_C( 35), UINT8_C( 15),
        UINT8_C(109), UINT8_C( 29), UINT8_C(173), UINT8_C( 55), UINT8_C(124), UINT8_C(133), UINT8_C( 86), UINT8_C(191) },
      { UINT8_C( 40), UINT8_C(174), UINT8_C(100), UINT8_C(231), UINT8_C( 66), UINT8_C( 29), UINT8_C( 35), UINT8_C( 15),
        UINT8_C(147), UINT8_C( 29), UINT8_C( 13), UINT8_C(243), UINT8_C(196), UINT8_C(133), UINT8_C( 86), UINT8_C(236) },
      UINT16_C(40228) },
    { { UINT8_C(211), UINT8_C(102), UINT8_C(176), UINT8_C(246), UINT8_C(118), UINT8_C( 29), UINT8_C( 20), UINT8_C( 35),
        UINT8_C( 85), UINT8_C(144), UINT8_C(168), UINT8_C(171), UINT8_C( 80), UINT8_C(223), UINT8_C(182), UINT8_C(180) },
      { UINT8_C(211), UINT8_C(205), UINT8_C(176), UINT8_C(150), UINT8_C(118), UINT8_C( 29), UINT8_C( 11), UINT8_C(201),
        UINT8_C( 87), UINT8_C(207), UINT8_C(168), UINT8_C( 94), UINT8_C( 80), UINT8_C( 99), UINT8_C(182), UINT8_C(143) },
      UINT16_C(43978) },
    { { UINT8_C(133), UINT8_C( 64), UINT8_C(201), UINT8_C(153), UINT8_C( 99), UINT8_C( 30), UINT8_C( 42), UINT8_C( 11),
        UINT8_C(201), UINT8_C(122), UINT8_C(235), UINT8_C(127), UINT8_C( 46), UINT8_C( 41), UINT8_C( 76),    UINT8_MAX },
      { UINT8_C(133), UINT8_C(  8), UINT8_C( 99), UINT8_C(202), UINT8_C(209), UINT8_C(186), UINT8_C(153), UINT8_C( 11),
        UINT8_C( 24), UINT8_C( 84), UINT8_C(116), UINT8_C(127), UINT8_C( 46), UINT8_C( 41), UINT8_C( 76), UINT8_C(105) },
      UINT16_C(34686) },
    { { UINT8_C(  2), UINT8_C(225), UINT8_C(165), UINT8_C( 44), UINT8_C(236), UINT8_C(110), UINT8_C(166), UINT8_C(215),
        UINT8_C(238), UINT8_C(212), UINT8_C(  0), UINT8_C( 58), UINT8_C(211), UINT8_C(191), UINT8_C( 67), UINT8_C( 54) },
      { UINT8_C(  2), UINT8_C( 20), UINT8_C(240), UINT8_C( 44), UINT8_C( 37), UINT8_C(  8), UINT8_C(166), UINT8_C(215),
        UINT8_C( 27), UINT8_C( 90), UINT8_C(215), UINT8_C( 58), UINT8_C(211), UINT8_C(191), UINT8_C( 67), UINT8_C( 54) },
      UINT16_C( 1846) },
    { { UINT8_C(242), UINT8_C( 34), UINT8_C(117), UINT8_C(153), UINT8_C(250), UINT8_C( 99), UINT8_C(109), UINT8_C(250),
        UINT8_C(158), UINT8_C( 65), UINT8_C(186), UINT8_C(225), UINT8_C(119), UINT8_C( 67), UINT8_C(245), UINT8_C(104) },
      { UINT8_C(242), UINT8_C( 34), UINT8_C(117), UINT8_C(221), UINT8_C(179), UINT8_C(140), UINT8_C(109), UINT8_C(250),
        UINT8_C(158), UINT8_C( 65), UINT8_C(223), UINT8_C(225), UINT8_C(119), UINT8_C( 67), UINT8_C(206), UINT8_C(104) },
      UINT16_C(17464) },
    { { UINT8_C( 76), UINT8_C( 50), UINT8_C(167), UINT8_C(186), UINT8_C( 44), UINT8_C( 69), UINT8_C(251), UINT8_C(230),
        UINT8_C( 38), UINT8_C(114), UINT8_C( 42), UINT8_C( 28), UINT8_C(218), UINT8_C(144), UINT8_C( 54), UINT8_C( 75) },
      { UINT8_C(109), UINT8_C(234), UINT8_C(167), UINT8_C(186), UINT8_C(116), UINT8_C( 69), UINT8_C(159), UINT8_C( 84),
        UINT8_C( 38), UINT8_C( 96), UINT8_C(105), UINT8_C(211), UINT8_C( 20), UINT8_C(161), UINT8_C( 54), UINT8_C( 96) },
      UINT16_C(48851) },
    { { UINT8_C( 26), UINT8_C(  0), UINT8_C(  4), UINT8_C( 21), UINT8_C(230), UINT8_C( 42), UINT8_C(136), UINT8_C( 16),
        UINT8_C( 70), UINT8_C( 98), UINT8_C(160), UINT8_C(125), UINT8_C(173), UINT8_C( 13), UINT8_C(103), UINT8_C(132) },
      { UINT8_C( 26), UINT8_C(219), UINT8_C(193), UINT8_C( 81), UINT8_C(230), UINT8_C( 42), UINT8_C(136), UINT8_C( 16),
        UINT8_C( 70), UINT8_C(198), UINT8_C(160), UINT8_C(125), UINT8_C( 38), UINT8_C( 14), UINT8_C(110), UINT8_C(132) },
      UINT16_C(29198) },
    { { UINT8_C( 86), UINT8_C(244), UINT8_C(157), UINT8_C(222), UINT8_C(  5), UINT8_C(227), UINT8_C( 65), UINT8_C(165),
        UINT8_C( 96), UINT8_C(238), UINT8_C(179), UINT8_C(199), UINT8_C(115), UINT8_C(101), UINT8_C(163), UINT8_C( 52) },
      { UINT8_C( 86), UINT8_C(210), UINT8_C(250), UINT8_C(222), UINT8_C(107), UINT8_C(147), UINT8_C( 65), UINT8_C(166),
        UINT8_C( 67), UINT8_C(238), UINT8_C(179), UINT8_C(199), UINT8_C(115), UINT8_C(101), UINT8_C( 36), UINT8_C(236) },
      UINT16_C(49590) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epu8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epu8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u8x16());
    easysimd__mmask16 r = easysimd_mm_cmpneq_epu8_mask(a, b);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const uint8_t a[16];
    const uint8_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(48075),
      { UINT8_C(165), UINT8_C( 12), UINT8_C( 97), UINT8_C(  5), UINT8_C(250), UINT8_C( 20), UINT8_C(205), UINT8_C(109),
        UINT8_C(121), UINT8_C(112), UINT8_C(162), UINT8_C( 47), UINT8_C( 66), UINT8_C(156), UINT8_C(152), UINT8_C(174) },
      { UINT8_C( 48), UINT8_C( 12), UINT8_C( 84), UINT8_C(  5), UINT8_C( 27), UINT8_C(  8), UINT8_C( 37), UINT8_C(109),
        UINT8_C(202), UINT8_C(112), UINT8_C(158), UINT8_C(128), UINT8_C( 11), UINT8_C(156), UINT8_C(152), UINT8_C(176) },
      UINT16_C(39233) },
    { UINT16_C(28598),
      { UINT8_C(177), UINT8_C(131), UINT8_C(221), UINT8_C( 42), UINT8_C(243), UINT8_C(127), UINT8_C( 89), UINT8_C( 53),
        UINT8_C( 27), UINT8_C(241), UINT8_C(227), UINT8_C( 75), UINT8_C(184), UINT8_C( 55), UINT8_C(191), UINT8_C(211) },
      { UINT8_C( 63), UINT8_C(228), UINT8_C(221), UINT8_C( 42), UINT8_C(243), UINT8_C( 35), UINT8_C( 89), UINT8_C( 53),
        UINT8_C(140), UINT8_C(241), UINT8_C(234), UINT8_C(  1), UINT8_C(184), UINT8_C( 55), UINT8_C(112), UINT8_C(211) },
      UINT16_C(19746) },
    { UINT16_C( 5694),
      { UINT8_C(204), UINT8_C(151), UINT8_C( 76), UINT8_C(232), UINT8_C(137), UINT8_C( 47), UINT8_C( 51), UINT8_C( 65),
        UINT8_C(103), UINT8_C(242), UINT8_C( 20), UINT8_C(166), UINT8_C(215), UINT8_C(153), UINT8_C(176), UINT8_C(  5) },
      { UINT8_C(204), UINT8_C(151), UINT8_C( 63), UINT8_C( 72), UINT8_C(137), UINT8_C( 47), UINT8_C( 73), UINT8_C( 65),
        UINT8_C(103), UINT8_C(242), UINT8_C( 20), UINT8_C(166), UINT8_C(  7), UINT8_C(153), UINT8_C(  4), UINT8_C(  5) },
      UINT16_C( 4108) },
    { UINT16_C(54716),
      { UINT8_C(128), UINT8_C(239), UINT8_C( 22), UINT8_C(231), UINT8_C(226), UINT8_C( 43), UINT8_C(141), UINT8_C(185),
        UINT8_C(196), UINT8_C( 61), UINT8_C(190), UINT8_C(129), UINT8_C(119), UINT8_C(254), UINT8_C(201), UINT8_C(119) },
      { UINT8_C(128), UINT8_C(239), UINT8_C(218), UINT8_C(231), UINT8_C(205), UINT8_C( 43), UINT8_C(224), UINT8_C(185),
        UINT8_C(  6), UINT8_C(229), UINT8_C(168), UINT8_C( 83), UINT8_C( 53), UINT8_C(100), UINT8_C(201), UINT8_C(119) },
      UINT16_C( 5396) },
    { UINT16_C(13980),
      { UINT8_C(106), UINT8_C( 42), UINT8_C(239), UINT8_C( 46), UINT8_C(103), UINT8_C(173), UINT8_C(175), UINT8_C(223),
        UINT8_C(171), UINT8_C(121), UINT8_C( 86), UINT8_C(211), UINT8_C(140), UINT8_C( 49), UINT8_C(198), UINT8_C( 89) },
      { UINT8_C(130), UINT8_C( 42), UINT8_C( 45), UINT8_C(137), UINT8_C(139), UINT8_C(214), UINT8_C(175), UINT8_C(223),
        UINT8_C( 58), UINT8_C(  4), UINT8_C( 86), UINT8_C(211), UINT8_C( 67), UINT8_C( 19), UINT8_C(198), UINT8_C(173) },
      UINT16_C(12828) },
    { UINT16_C(42204),
      { UINT8_C( 97), UINT8_C(139), UINT8_C(131), UINT8_C( 12), UINT8_C(  4), UINT8_C(218), UINT8_C(224), UINT8_C(144),
        UINT8_C( 11), UINT8_C(166), UINT8_C(233), UINT8_C(141), UINT8_C( 76), UINT8_C( 23), UINT8_C( 22), UINT8_C(216) },
      { UINT8_C(237), UINT8_C(242), UINT8_C(153), UINT8_C( 39), UINT8_C(  4), UINT8_C(218), UINT8_C(182), UINT8_C(144),
        UINT8_C( 34), UINT8_C(122), UINT8_C(232), UINT8_C(141), UINT8_C( 76), UINT8_C( 23), UINT8_C( 22), UINT8_C(143) },
      UINT16_C(33868) },
    { UINT16_C(21659),
      { UINT8_C( 97), UINT8_C(123), UINT8_C(228), UINT8_C(108), UINT8_C( 33), UINT8_C(206), UINT8_C(250), UINT8_C(110),
        UINT8_C(229), UINT8_C( 16), UINT8_C( 70), UINT8_C(210), UINT8_C(  3), UINT8_C(223), UINT8_C(249), UINT8_C(250) },
      { UINT8_C( 97), UINT8_C(175), UINT8_C( 52), UINT8_C( 17), UINT8_C( 42), UINT8_C( 28), UINT8_C(112), UINT8_C( 88),
        UINT8_C(229), UINT8_C( 16), UINT8_C( 70), UINT8_C( 48), UINT8_C(  3), UINT8_C(130), UINT8_C(132), UINT8_C(250) },
      UINT16_C(16538) },
    { UINT16_C( 8138),
      { UINT8_C( 54), UINT8_C(196), UINT8_C(141), UINT8_C( 27), UINT8_C(212), UINT8_C(211), UINT8_C(237), UINT8_C(215),
        UINT8_C(178), UINT8_C(231), UINT8_C(209), UINT8_C(161), UINT8_C(150), UINT8_C(  6), UINT8_C(178), UINT8_C(192) },
      { UINT8_C( 54), UINT8_C( 34), UINT8_C( 24), UINT8_C(  3), UINT8_C(151),    UINT8_MAX, UINT8_C( 51), UINT8_C(215),
        UINT8_C(130), UINT8_C(231), UINT8_C(240), UINT8_C(128), UINT8_C(150), UINT8_C(186), UINT8_C(178), UINT8_C(192) },
      UINT16_C( 3402) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__mmask16 k1 = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epu8_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epu8_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_mm_mask_blend_epi8(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u8x16());
    easysimd__mmask16 r = easysimd_mm_mask_cmpneq_epu8_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpneq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[8];
    const int16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT16_C( 18013),  INT16_C( 18652), -INT16_C( 26325),  INT16_C(  6164), -INT16_C( 21146), -INT16_C( 29070),  INT16_C( 25810), -INT16_C(  8957) },
      { -INT16_C( 32319),  INT16_C( 18652), -INT16_C( 20263),  INT16_C(  6164),  INT16_C( 15127), -INT16_C( 29070),  INT16_C( 25810), -INT16_C(  8957) },
      UINT8_C( 21) },
    { { -INT16_C( 22599),  INT16_C( 21057),  INT16_C( 22971),  INT16_C( 26808),  INT16_C( 18123),  INT16_C( 12347),  INT16_C(  6218), -INT16_C( 13327) },
      { -INT16_C( 27994),  INT16_C( 21057),  INT16_C( 16880),  INT16_C( 11118),  INT16_C( 18123),  INT16_C( 13798),  INT16_C( 17914), -INT16_C( 19637) },
      UINT8_C(237) },
    { {  INT16_C(  1420), -INT16_C(  6744),  INT16_C(  4541),  INT16_C(  1201), -INT16_C(  7860),  INT16_C( 25678),  INT16_C(  6610),  INT16_C( 25610) },
      {  INT16_C(  1420), -INT16_C(   428),  INT16_C( 32719), -INT16_C(  4905),  INT16_C(  3430),  INT16_C( 25678),  INT16_C(  6610), -INT16_C(  7016) },
      UINT8_C(158) },
    { { -INT16_C( 14015),  INT16_C( 21084),  INT16_C( 24698),  INT16_C( 23454),  INT16_C(   686), -INT16_C( 14547), -INT16_C( 28147),  INT16_C( 28292) },
      { -INT16_C( 31770),  INT16_C( 21084),  INT16_C( 10586),  INT16_C( 26572),  INT16_C(   686), -INT16_C( 22081), -INT16_C( 23792),  INT16_C( 28292) },
      UINT8_C(109) },
    { { -INT16_C( 23644),  INT16_C(  1255),  INT16_C( 17217),  INT16_C( 17330),  INT16_C( 31088),  INT16_C(   592), -INT16_C( 16643), -INT16_C( 32535) },
      { -INT16_C( 23644),  INT16_C(  9691),  INT16_C( 16923),  INT16_C( 17330), -INT16_C(  8702),  INT16_C(   592), -INT16_C( 16643), -INT16_C( 13806) },
      UINT8_C(150) },
    { { -INT16_C( 12550),  INT16_C( 15831),  INT16_C(  7040), -INT16_C(  1619), -INT16_C( 20373),  INT16_C( 10999),  INT16_C( 30617), -INT16_C(  6107) },
      {  INT16_C( 19026),  INT16_C( 15831), -INT16_C( 27265),  INT16_C( 24215),  INT16_C( 15416),  INT16_C( 10999),  INT16_C( 30617), -INT16_C(  6107) },
      UINT8_C( 29) },
    { { -INT16_C( 31079), -INT16_C( 19298), -INT16_C( 26829), -INT16_C(  7392),  INT16_C( 19086),  INT16_C(  1660),  INT16_C( 25711), -INT16_C( 17832) },
      { -INT16_C(  4761), -INT16_C(   711), -INT16_C( 26748), -INT16_C( 16075),  INT16_C( 24604),  INT16_C(  1660),  INT16_C( 22818), -INT16_C( 17272) },
      UINT8_C(223) },
    { {  INT16_C( 28710), -INT16_C( 16878), -INT16_C(  2416), -INT16_C(  9652),  INT16_C( 21106), -INT16_C( 10422),  INT16_C(  1195), -INT16_C( 26562) },
      {  INT16_C( 28710), -INT16_C( 16878), -INT16_C(  2416), -INT16_C(  9652),  INT16_C( 21106),  INT16_C( 18419),  INT16_C(  1195), -INT16_C( 26562) },
      UINT8_C( 32) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_mm_mask_blend_epi16(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i16x8());
    easysimd__mmask8 r = easysimd_mm_cmpneq_epi16_mask(a, b);

    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int16_t a[8];
    const int16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 62),
      {  INT16_C( 17578), -INT16_C(  3927),  INT16_C( 20836), -INT16_C( 31549),  INT16_C( 32108),  INT16_C( 11114), -INT16_C( 24083),  INT16_C( 24275) },
      {  INT16_C(  8028), -INT16_C(  3927),  INT16_C( 20836), -INT16_C( 31549),  INT16_C( 32108),  INT16_C( 24573), -INT16_C( 25145),  INT16_C( 29342) },
      UINT8_C( 32) },
    { UINT8_C( 71),
      {  INT16_C( 18018),  INT16_C(  9624),  INT16_C(  1226),  INT16_C( 13475), -INT16_C( 28625),  INT16_C(   725),  INT16_C( 13038), -INT16_C( 24543) },
      {  INT16_C( 18018), -INT16_C( 14494),  INT16_C(  1226), -INT16_C(  2565), -INT16_C( 15791), -INT16_C(  4205),  INT16_C( 13038), -INT16_C( 27082) },
      UINT8_C(  2) },
    { UINT8_C(206),
      { -INT16_C( 31300),  INT16_C( 24530),  INT16_C(   697), -INT16_C( 28689), -INT16_C(  8956),  INT16_C(  9921),  INT16_C(  1149), -INT16_C(  8219) },
      { -INT16_C( 31300), -INT16_C( 14384),  INT16_C(   697),  INT16_C( 26249), -INT16_C( 16880),  INT16_C(  9921),  INT16_C(  1149), -INT16_C(  8219) },
      UINT8_C( 10) },
    { UINT8_C(231),
      { -INT16_C( 11409),  INT16_C( 24297), -INT16_C(  4766),  INT16_C(  9019), -INT16_C( 18413), -INT16_C(  2008), -INT16_C(  2921),  INT16_C( 26838) },
      { -INT16_C( 11409),  INT16_C( 24297), -INT16_C( 26097), -INT16_C(  5886), -INT16_C( 18413), -INT16_C(  2690), -INT16_C( 26521),  INT16_C( 26838) },
      UINT8_C(100) },
    { UINT8_C(197),
      { -INT16_C( 12747),  INT16_C( 29106), -INT16_C( 14606),  INT16_C(  6697), -INT16_C( 15938), -INT16_C( 27634), -INT16_C( 14039), -INT16_C( 19907) },
      {  INT16_C( 19469),  INT16_C(  4172),  INT16_C( 11574),  INT16_C(  6697), -INT16_C( 15938), -INT16_C( 27634), -INT16_C( 14039), -INT16_C(  9277) },
      UINT8_C(133) },
    { UINT8_C(117),
      {  INT16_C( 31052),  INT16_C( 30011), -INT16_C(  1389), -INT16_C( 24266),  INT16_C( 24462), -INT16_C( 13206),  INT16_C( 30738),  INT16_C( 24088) },
      {  INT16_C( 31052), -INT16_C(  4213), -INT16_C(  1389),  INT16_C( 20669),  INT16_C( 24462), -INT16_C( 13206),  INT16_C( 30738),  INT16_C( 24088) },
      UINT8_C(  0) },
    { UINT8_C( 31),
      { -INT16_C( 25089),  INT16_C( 13849), -INT16_C( 22465), -INT16_C( 22123), -INT16_C( 22668), -INT16_C( 29663), -INT16_C( 22266), -INT16_C( 28197) },
      { -INT16_C(  8552),  INT16_C( 22079), -INT16_C(  5586),  INT16_C( 14265), -INT16_C( 22668),  INT16_C( 15815), -INT16_C( 11903), -INT16_C( 28197) },
      UINT8_C( 15) },
    { UINT8_C(118),
      { -INT16_C( 20810),  INT16_C( 19486), -INT16_C( 28073),  INT16_C( 31219), -INT16_C(  1762), -INT16_C(  1758), -INT16_C( 17525), -INT16_C( 13609) },
      { -INT16_C( 20810), -INT16_C( 13644), -INT16_C( 28073),  INT16_C( 31219),  INT16_C( 16970), -INT16_C( 22571),  INT16_C( 17602),  INT16_C( 31005) },
      UINT8_C(114) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epi16_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epi16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_mm_mask_blend_epi16(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i16x8());
    easysimd__mmask8 r = easysimd_mm_mask_cmpneq_epi16_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpneq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t a[8];
    const uint16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { UINT16_C(50491), UINT16_C(52554), UINT16_C(50104), UINT16_C(45803), UINT16_C(58853), UINT16_C(41021), UINT16_C( 1980), UINT16_C(49841) },
      { UINT16_C(31675), UINT16_C(51710), UINT16_C(  572), UINT16_C(32275), UINT16_C(47832), UINT16_C( 7233), UINT16_C(47831), UINT16_C(49841) },
      UINT8_C(127) },
    { { UINT16_C(57177), UINT16_C( 7223), UINT16_C(59851), UINT16_C(45057), UINT16_C(41510), UINT16_C(11628), UINT16_C(11859), UINT16_C(53225) },
      { UINT16_C(45613), UINT16_C( 7223), UINT16_C(59851), UINT16_C(32775), UINT16_C(41510), UINT16_C(11628), UINT16_C(27187), UINT16_C(53225) },
      UINT8_C( 73) },
    { { UINT16_C(43067), UINT16_C( 9492), UINT16_C(50345), UINT16_C(19275), UINT16_C(31025), UINT16_C(24479), UINT16_C(28258), UINT16_C( 5260) },
      { UINT16_C(43067), UINT16_C(  985), UINT16_C(22979), UINT16_C(59342), UINT16_C(31025), UINT16_C(24479), UINT16_C(42583), UINT16_C( 5260) },
      UINT8_C( 78) },
    { { UINT16_C(47225), UINT16_C(15864), UINT16_C(17155), UINT16_C(31854), UINT16_C(52962), UINT16_C(20702), UINT16_C(62042), UINT16_C( 5834) },
      { UINT16_C(47225), UINT16_C( 9690), UINT16_C(17155), UINT16_C(31854), UINT16_C(61916), UINT16_C(20702), UINT16_C(62042), UINT16_C(53201) },
      UINT8_C(146) },
    { { UINT16_C( 3273), UINT16_C( 3221), UINT16_C( 4731), UINT16_C(18927), UINT16_C(16368), UINT16_C(58275), UINT16_C(47625), UINT16_C(55215) },
      { UINT16_C( 3273), UINT16_C(21875), UINT16_C(25259), UINT16_C(39985), UINT16_C(16368), UINT16_C(58275), UINT16_C(49541), UINT16_C(19989) },
      UINT8_C(206) },
    { { UINT16_C(23211), UINT16_C(48457), UINT16_C(37449), UINT16_C(35245), UINT16_C(36917), UINT16_C(61330), UINT16_C(26943), UINT16_C( 5251) },
      { UINT16_C(23211), UINT16_C(48457), UINT16_C(23306), UINT16_C(48872), UINT16_C(36917), UINT16_C( 3907), UINT16_C(37249), UINT16_C(11485) },
      UINT8_C(236) },
    { { UINT16_C(59686), UINT16_C(47157), UINT16_C(48791), UINT16_C(10222), UINT16_C(56657), UINT16_C(47719), UINT16_C(31585), UINT16_C(14999) },
      { UINT16_C(59686), UINT16_C(38212), UINT16_C(48791), UINT16_C(10222), UINT16_C(56657), UINT16_C(47719), UINT16_C(55504), UINT16_C(63428) },
      UINT8_C(194) },
    { { UINT16_C(45049), UINT16_C(47193), UINT16_C(32925), UINT16_C(31497), UINT16_C(50151), UINT16_C(25308), UINT16_C( 5722), UINT16_C(12444) },
      { UINT16_C(12890), UINT16_C(24047), UINT16_C( 6421), UINT16_C( 2212), UINT16_C(31941), UINT16_C(40665), UINT16_C(53312), UINT16_C(12444) },
      UINT8_C(127) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epu16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epu16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_mm_mask_blend_epi16(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u16x8());
    easysimd__mmask8 r = easysimd_mm_cmpneq_epu16_mask(a, b);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint16_t a[8];
    const uint16_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(185),
      { UINT16_C( 7665), UINT16_C(64057), UINT16_C( 8600), UINT16_C(29886), UINT16_C( 6275), UINT16_C( 8330), UINT16_C(58441), UINT16_C(14418) },
      { UINT16_C(26434), UINT16_C(64057), UINT16_C( 6000), UINT16_C(29886), UINT16_C(41653), UINT16_C( 5401), UINT16_C(58441), UINT16_C(52686) },
      UINT8_C(177) },
    { UINT8_C(  7),
      { UINT16_C(19911), UINT16_C(34088), UINT16_C(44225), UINT16_C(19358), UINT16_C(59340), UINT16_C( 7728), UINT16_C(29215), UINT16_C(28805) },
      { UINT16_C(62808), UINT16_C(47751), UINT16_C(44225), UINT16_C(19358), UINT16_C(14161), UINT16_C( 8176), UINT16_C(42244), UINT16_C(52263) },
      UINT8_C(  3) },
    { UINT8_C( 79),
      { UINT16_C(46161), UINT16_C(61435), UINT16_C(50944), UINT16_C(12502), UINT16_C(62949), UINT16_C(27554), UINT16_C(64102), UINT16_C(60768) },
      { UINT16_C(40884), UINT16_C( 4138), UINT16_C(50944), UINT16_C(12502), UINT16_C(19611), UINT16_C(49804), UINT16_C(64102), UINT16_C(60768) },
      UINT8_C(  3) },
    { UINT8_C( 13),
      { UINT16_C(13145), UINT16_C(12244), UINT16_C(47715), UINT16_C( 1317), UINT16_C(35621), UINT16_C(34303), UINT16_C(45944), UINT16_C(41508) },
      { UINT16_C( 7107), UINT16_C( 2846), UINT16_C(47715), UINT16_C( 1317), UINT16_C(28539), UINT16_C(35852), UINT16_C(16600), UINT16_C(41508) },
      UINT8_C(  1) },
    { UINT8_C(110),
      { UINT16_C(55137), UINT16_C(34344), UINT16_C(19932), UINT16_C(56337), UINT16_C(35282), UINT16_C(63375), UINT16_C(21292), UINT16_C(18962) },
      { UINT16_C(55137), UINT16_C(34344), UINT16_C(19932), UINT16_C(44324), UINT16_C(64522), UINT16_C(42221), UINT16_C(21292), UINT16_C(18962) },
      UINT8_C( 40) },
    { UINT8_C( 58),
      { UINT16_C( 5141), UINT16_C( 9863), UINT16_C(23024), UINT16_C(32943), UINT16_C(56144), UINT16_C(25299), UINT16_C(12581), UINT16_C(10358) },
      { UINT16_C( 6118), UINT16_C( 2726), UINT16_C(45508), UINT16_C(45574), UINT16_C(13397), UINT16_C(25299), UINT16_C(19395), UINT16_C(10358) },
      UINT8_C( 26) },
    { UINT8_C( 40),
      { UINT16_C(20734), UINT16_C(44673), UINT16_C(53968), UINT16_C(41865), UINT16_C(44852), UINT16_C(43732), UINT16_C(47831), UINT16_C(32449) },
      { UINT16_C(34500), UINT16_C(44673), UINT16_C(33848), UINT16_C(41865), UINT16_C(44852), UINT16_C(43732), UINT16_C(62875), UINT16_C(32449) },
      UINT8_C(  0) },
    { UINT8_C( 53),
      { UINT16_C( 5447), UINT16_C(53511), UINT16_C(15544), UINT16_C(35968), UINT16_C(22502), UINT16_C(43078), UINT16_C( 2773), UINT16_C( 1070) },
      { UINT16_C(26325), UINT16_C(54408), UINT16_C(29617), UINT16_C(35968), UINT16_C(12799), UINT16_C(43078), UINT16_C( 2773), UINT16_C( 4841) },
      UINT8_C( 21) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epu16_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epu16_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_mm_mask_blend_epi16(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u16x8());
    easysimd__mmask8 r = easysimd_mm_mask_cmpneq_epu16_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpneq_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[4];
    const int32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { {  INT32_C(   743498736), -INT32_C(  1156326301),  INT32_C(   764459811),  INT32_C(  1325569513) },
      { -INT32_C(  1862216163), -INT32_C(  1634711699),  INT32_C(    90784899),  INT32_C(   496860205) },
      UINT8_C( 15) },
    { { -INT32_C(   909948179),  INT32_C(   418152029), -INT32_C(  1560208882), -INT32_C(   205434852) },
      { -INT32_C(  1755295152),  INT32_C(   639303394), -INT32_C(  1560208882), -INT32_C(  1438420547) },
      UINT8_C( 11) },
    { { -INT32_C(  2078772474),  INT32_C(  2056466528), -INT32_C(  2120862159),  INT32_C(  1238469111) },
      { -INT32_C(   399742743),  INT32_C(  2056466528),  INT32_C(   392115366),  INT32_C(  2127694199) },
      UINT8_C( 13) },
    { {  INT32_C(   450691818), -INT32_C(   867477611),  INT32_C(  2009320685),  INT32_C(    90181021) },
      {  INT32_C(   450691818), -INT32_C(   867477611), -INT32_C(  1824241527),  INT32_C(    90181021) },
      UINT8_C(  4) },
    { {  INT32_C(  1146756845),  INT32_C(   892413545), -INT32_C(   153966359),  INT32_C(  1362089737) },
      {  INT32_C(  1146756845),  INT32_C(   890418924), -INT32_C(   153966359),  INT32_C(  1362089737) },
      UINT8_C(  2) },
    { {  INT32_C(    31254235), -INT32_C(  1058402024), -INT32_C(  1144397340), -INT32_C(   887481584) },
      {  INT32_C(    31254235),  INT32_C(  1538715062), -INT32_C(  1144397340), -INT32_C(   838618125) },
      UINT8_C( 10) },
    { { -INT32_C(   256716833), -INT32_C(  1680575814),  INT32_C(  1470861372),  INT32_C(  1314682794) },
      {  INT32_C(   470030127), -INT32_C(  1680575814),  INT32_C(  1470861372),  INT32_C(  1314682794) },
      UINT8_C(  1) },
    { { -INT32_C(   494158992),  INT32_C(   538846864),  INT32_C(  1238005202), -INT32_C(   378005295) },
      { -INT32_C(   494158992),  INT32_C(   538846864),  INT32_C(  1238005202),  INT32_C(  1056804046) },
      UINT8_C(  8) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_mm_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i32x4());
    easysimd__mmask8 r = easysimd_mm_cmpneq_epi32_mask(a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int32_t a[4];
    const int32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(137),
      {  INT32_C(  1068488737), -INT32_C(  1028292104),  INT32_C(  1121683146),  INT32_C(   651622255) },
      {  INT32_C(  1068488737),  INT32_C(  1517307544),  INT32_C(  1894137714),  INT32_C(   651622255) },
      UINT8_C(  0) },
    { UINT8_C(168),
      { -INT32_C(  1825898786), -INT32_C(  1676020543),  INT32_C(   227772727), -INT32_C(   875558993) },
      { -INT32_C(   686659225), -INT32_C(  1676020543), -INT32_C(   304573196),  INT32_C(   278322738) },
      UINT8_C(  8) },
    { UINT8_C(192),
      { -INT32_C(  1112236381), -INT32_C(   850990278), -INT32_C(   908790279), -INT32_C(   768459840) },
      { -INT32_C(  1112236381), -INT32_C(   850990278), -INT32_C(   908790279), -INT32_C(   768459840) },
      UINT8_C(  0) },
    { UINT8_C( 67),
      { -INT32_C(  1490089375), -INT32_C(  1399052072),  INT32_C(   619207921), -INT32_C(  2045117649) },
      { -INT32_C(  1490089375), -INT32_C(  2049832401),  INT32_C(   619207921), -INT32_C(  2045117649) },
      UINT8_C(  2) },
    { UINT8_C(219),
      {  INT32_C(   436431486), -INT32_C(     8915945),  INT32_C(   855287320), -INT32_C(    34988500) },
      { -INT32_C(   771220992),  INT32_C(  1453711775),  INT32_C(   855287320),  INT32_C(  1909551603) },
      UINT8_C( 11) },
    { UINT8_C(212),
      {  INT32_C(    63500940), -INT32_C(   341188111), -INT32_C(    49621741), -INT32_C(   636024110) },
      {  INT32_C(    63500940), -INT32_C(   342030835), -INT32_C(    49621741), -INT32_C(  1902036990) },
      UINT8_C(  0) },
    { UINT8_C(105),
      {  INT32_C(   977935505),  INT32_C(   286219527),  INT32_C(  1914495323),  INT32_C(  2016016828) },
      {  INT32_C(   977935505),  INT32_C(   286219527), -INT32_C(   390681785), -INT32_C(   900603847) },
      UINT8_C(  8) },
    { UINT8_C(156),
      {  INT32_C(   335086596),  INT32_C(   838095893), -INT32_C(   265637689), -INT32_C(  1507416792) },
      {  INT32_C(   335086596),  INT32_C(   838095893), -INT32_C(  1587786929),  INT32_C(   423450645) },
      UINT8_C( 12) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epi32_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_mm_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i32x4());
    easysimd__mmask8 r = easysimd_mm_mask_cmpneq_epi32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpneq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t a[4];
    const uint32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { UINT32_C(2339515446), UINT32_C(3511853856), UINT32_C(1945715406), UINT32_C(2939265128) },
      { UINT32_C( 678795456), UINT32_C(4118278314), UINT32_C(2768902220), UINT32_C(2348884821) },
      UINT8_C( 15) },
    { { UINT32_C( 393222003), UINT32_C(2883928425), UINT32_C(3658700858), UINT32_C( 580633226) },
      { UINT32_C( 684573495), UINT32_C(1400160826), UINT32_C(3658700858), UINT32_C(2386572315) },
      UINT8_C( 11) },
    { { UINT32_C(4055147952), UINT32_C(3811270538), UINT32_C( 896337522), UINT32_C(1684836257) },
      { UINT32_C( 530486364), UINT32_C(3811270538), UINT32_C( 896337522), UINT32_C(1684836257) },
      UINT8_C(  1) },
    { { UINT32_C( 788315598), UINT32_C(4020297705), UINT32_C(3683702092), UINT32_C(3594056770) },
      { UINT32_C( 788315598), UINT32_C( 522060355), UINT32_C( 188542015), UINT32_C(3594056770) },
      UINT8_C(  6) },
    { { UINT32_C(1487864697), UINT32_C(2040831651), UINT32_C( 582713134), UINT32_C( 246714807) },
      { UINT32_C(1934702705), UINT32_C(2040831651), UINT32_C( 582713134), UINT32_C( 246714807) },
      UINT8_C(  1) },
    { { UINT32_C(3368356906), UINT32_C( 804667056), UINT32_C(2330401017), UINT32_C( 100398541) },
      { UINT32_C(3368356906), UINT32_C(3747979041), UINT32_C(2330401017), UINT32_C(3223899798) },
      UINT8_C( 10) },
    { { UINT32_C( 730499565), UINT32_C(2552543615), UINT32_C(2523246496), UINT32_C( 433941162) },
      { UINT32_C( 730499565), UINT32_C(2552543615), UINT32_C(  27940147), UINT32_C( 433941162) },
      UINT8_C(  4) },
    { { UINT32_C( 510891621), UINT32_C(1790905275), UINT32_C(3675542896), UINT32_C(3349228850) },
      { UINT32_C(2160133991), UINT32_C(1790905275), UINT32_C(3675542896), UINT32_C(4108292751) },
      UINT8_C(  9) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epu32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_mm_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u32x4());
    easysimd__mmask8 r = easysimd_mm_cmpneq_epu32_mask(a, b);

    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint32_t a[4];
    const uint32_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 82),
      { UINT32_C(3529434131), UINT32_C(1646775886), UINT32_C(1267751337), UINT32_C(3780310816) },
      { UINT32_C(3529434131), UINT32_C(1646775886), UINT32_C(1267751337), UINT32_C(2001941604) },
      UINT8_C(  0) },
    { UINT8_C(177),
      { UINT32_C(1887397449), UINT32_C(1489578440), UINT32_C(3327191412), UINT32_C(2154678474) },
      { UINT32_C(3552034547), UINT32_C(1094965302), UINT32_C(3327191412), UINT32_C(2154678474) },
      UINT8_C(  1) },
    { UINT8_C( 92),
      { UINT32_C(2693119191), UINT32_C(1686763796), UINT32_C( 741102527), UINT32_C(2330599635) },
      { UINT32_C( 912466419), UINT32_C(1686763796), UINT32_C(1164311400), UINT32_C(2330599635) },
      UINT8_C(  4) },
    { UINT8_C( 39),
      { UINT32_C(1646275033), UINT32_C(3132481678), UINT32_C(4158294284), UINT32_C(1981337107) },
      { UINT32_C(3803347460), UINT32_C(3101497938), UINT32_C(  17907807), UINT32_C(1395140217) },
      UINT8_C(  7) },
    { UINT8_C( 72),
      { UINT32_C(1797840309), UINT32_C(1928607128), UINT32_C(1145046828), UINT32_C( 817775998) },
      { UINT32_C(  72028455), UINT32_C(1928607128), UINT32_C(3540041387), UINT32_C(3323744017) },
      UINT8_C(  8) },
    { UINT8_C( 68),
      { UINT32_C( 628737329), UINT32_C(2016257335), UINT32_C(2831002601), UINT32_C( 649716955) },
      { UINT32_C( 768705256), UINT32_C(2016257335), UINT32_C(1823210576), UINT32_C(4105230530) },
      UINT8_C(  4) },
    { UINT8_C( 42),
      { UINT32_C(1188025625), UINT32_C( 183613773), UINT32_C( 451857761), UINT32_C(3298612979) },
      { UINT32_C(1188025625), UINT32_C(4293562708), UINT32_C( 451857761), UINT32_C(3134682529) },
      UINT8_C( 10) },
    { UINT8_C(167),
      { UINT32_C(4066416385), UINT32_C(2411708833), UINT32_C(2003219419), UINT32_C(3094309239) },
      { UINT32_C(4066416385), UINT32_C(2599372482), UINT32_C(2003219419), UINT32_C(3094309239) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epu32_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_mm_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u32x4());
    easysimd__mmask8 r = easysimd_mm_mask_cmpneq_epu32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpneq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT64_C( 2395685559586529103), -INT64_C( 3134027570735926721) },
      { -INT64_C( 5893510356011409085),  INT64_C( 2190428841884919221) },
      UINT8_C(  3) },
    { {  INT64_C( 4450509753596267188),  INT64_C( 6943029722049953124) },
      {  INT64_C( 2536543627709958002),  INT64_C( 6943029722049953124) },
      UINT8_C(  1) },
    { { -INT64_C(  934144298817686975),  INT64_C( 4687986054940205060) },
      {  INT64_C( 1624499570496933120),  INT64_C( 4687986054940205060) },
      UINT8_C(  1) },
    { {  INT64_C( 4459366402878805149),  INT64_C( 1990301376776208268) },
      {  INT64_C( 4074334652475325238),  INT64_C( 1990301376776208268) },
      UINT8_C(  1) },
    { { -INT64_C( 6737592695842783207), -INT64_C(  636894597407040006) },
      { -INT64_C( 6737592695842783207),  INT64_C( 2818018671853296476) },
      UINT8_C(  2) },
    { { -INT64_C( 5233002883489654238),  INT64_C( 4342714117228024531) },
      { -INT64_C( 5233002883489654238),  INT64_C( 4342714117228024531) },
      UINT8_C(  0) },
    { {  INT64_C(  300530274922025397), -INT64_C( 1532705406965561051) },
      {  INT64_C(  300530274922025397), -INT64_C( 1532705406965561051) },
      UINT8_C(  0) },
    { { -INT64_C( 8878351628961443598), -INT64_C( 1278463652121126335) },
      {  INT64_C( 5449155417840321052),  INT64_C( 2368901301691889176) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_mm_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i64x2());
    easysimd__mmask8 r = easysimd_mm_cmpneq_epi64_mask(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(158),
      { -INT64_C( 1592125996090240060),  INT64_C( 3342237076986133407) },
      { -INT64_C( 1592125996090240060),  INT64_C( 3342237076986133407) },
      UINT8_C(  0) },
    { UINT8_C( 65),
      {  INT64_C( 1128576297155641542),  INT64_C( 4044621420906517090) },
      { -INT64_C( 4626274607261234607),  INT64_C( 4044621420906517090) },
      UINT8_C(  1) },
    { UINT8_C(117),
      { -INT64_C( 4167982818674628883), -INT64_C( 1654939679374907107) },
      {  INT64_C( 6773276987916339697), -INT64_C( 1654939679374907107) },
      UINT8_C(  1) },
    { UINT8_C(210),
      {  INT64_C( 7011058131223147323),  INT64_C( 7735157032161602950) },
      { -INT64_C( 2881527237705801334), -INT64_C( 7289730897741366744) },
      UINT8_C(  2) },
    { UINT8_C(  0),
      { -INT64_C( 7658424242303116803),  INT64_C( 7464694914884963934) },
      { -INT64_C( 7191594749958367882),  INT64_C( 7464694914884963934) },
      UINT8_C(  0) },
    { UINT8_C(116),
      {  INT64_C( 8302563008153091186),  INT64_C( 4811487179498893656) },
      {  INT64_C( 8302563008153091186), -INT64_C( 2902708067688760298) },
      UINT8_C(  0) },
    { UINT8_C(250),
      { -INT64_C( 2023536238105194751), -INT64_C( 2730580753305238813) },
      { -INT64_C( 2023536238105194751),  INT64_C( 1211400198052657182) },
      UINT8_C(  2) },
    { UINT8_C(107),
      { -INT64_C( 1065590353167428279),  INT64_C( 1887148247034133483) },
      { -INT64_C( 5106215605561347561), -INT64_C( 8276685637411916824) },
      UINT8_C(  3) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epi64_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_mm_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i64x2());
    easysimd__mmask8 r = easysimd_mm_mask_cmpneq_epi64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpneq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[4];
    const uint64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { UINT64_C( 8192554557433954435), UINT64_C(11194507111956920516) },
      { UINT64_C( 8192554557433954435), UINT64_C(11194507111956920516) },
      UINT8_C(  0) },
    { { UINT64_C( 6926981152710588049), UINT64_C(  221208262895756379) },
      { UINT64_C( 6305611008499841907), UINT64_C(16032591800372074214) },
      UINT8_C(  3) },
    { { UINT64_C(13947985996387924898), UINT64_C(15696034117424565628) },
      { UINT64_C(13947985996387924898), UINT64_C( 5864482414301360130) },
      UINT8_C(  2) },
    { { UINT64_C(16637753985556252470), UINT64_C(12114856824361207213) },
      { UINT64_C(16637753985556252470), UINT64_C(12114856824361207213) },
      UINT8_C(  0) },
    { { UINT64_C(12069553967307521445), UINT64_C(16596441715800256367) },
      { UINT64_C( 6043267580424852514), UINT64_C(16596441715800256367) },
      UINT8_C(  1) },
    { { UINT64_C(16682400882115197032), UINT64_C( 8508599194069930122) },
      { UINT64_C(16682400882115197032), UINT64_C( 5681192957810524882) },
      UINT8_C(  2) },
    { { UINT64_C( 8453394922095403514), UINT64_C(17242258045645832755) },
      { UINT64_C( 5517159178812375068), UINT64_C(17242258045645832755) },
      UINT8_C(  1) },
    { { UINT64_C(14982697015241489725), UINT64_C(14302610009214008978) },
      { UINT64_C(14982697015241489725), UINT64_C( 3148409690873808719) },
      UINT8_C(  2) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpneq_epu64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpneq_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_mm_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u64x2());
    easysimd__mmask8 r = easysimd_mm_cmpneq_epu64_mask(a, b);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_cmpneq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint64_t a[4];
    const uint64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(119),
      { UINT64_C( 2562624049221087205), UINT64_C( 3341020412631291813) },
      { UINT64_C( 2562624049221087205), UINT64_C( 5806142034820218987) },
      UINT8_C(  2) },
    { UINT8_C( 69),
      { UINT64_C( 1750262947806559522), UINT64_C( 2510147485344245584) },
      { UINT64_C( 1750262947806559522), UINT64_C( 2510147485344245584) },
      UINT8_C(  0) },
    { UINT8_C(166),
      { UINT64_C(12534685329645686018), UINT64_C(12184034610476403410) },
      { UINT64_C(12534685329645686018), UINT64_C(12184034610476403410) },
      UINT8_C(  0) },
    { UINT8_C(133),
      { UINT64_C( 6732174296479313327), UINT64_C(15296287027956242724) },
      { UINT64_C( 6732174296479313327), UINT64_C(13842346793370496739) },
      UINT8_C(  0) },
    { UINT8_C(195),
      { UINT64_C(  660169092870696291), UINT64_C(16941336737288629151) },
      { UINT64_C(  660169092870696291), UINT64_C(16941336737288629151) },
      UINT8_C(  0) },
    { UINT8_C( 87),
      { UINT64_C(  381498267298143991), UINT64_C(18017410305297828672) },
      { UINT64_C(  381498267298143991), UINT64_C(18017410305297828672) },
      UINT8_C(  0) },
    { UINT8_C( 91),
      { UINT64_C( 7027855829276925936), UINT64_C( 6500397098080714320) },
      { UINT64_C( 2708750386108858614), UINT64_C( 3652268945294979507) },
      UINT8_C(  3) },
    { UINT8_C(101),
      { UINT64_C( 3581394273848441710), UINT64_C(18355604207257349125) },
      { UINT64_C( 3581394273848441710), UINT64_C(  253344612200506224) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_cmpneq_epu64_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_cmpneq_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_mm_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u64x2());
    easysimd__mmask8 r = easysimd_mm_mask_cmpneq_epu64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t a[32];
    const int8_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { { -INT8_C(  50),  INT8_C(  42), -INT8_C( 104),  INT8_C(  30),  INT8_C(  19), -INT8_C(   9),  INT8_C(  52),  INT8_C( 116),
        -INT8_C(  17),  INT8_C(  27),  INT8_C( 120), -INT8_C(  58), -INT8_C(  74),  INT8_C(  37), -INT8_C(  41),  INT8_C(  83),
        -INT8_C(  63),  INT8_C( 104),  INT8_C(  49), -INT8_C(  93), -INT8_C( 124), -INT8_C( 121), -INT8_C(  37),  INT8_C( 106),
         INT8_C( 100),  INT8_C(  52), -INT8_C(  52),  INT8_C(  73),  INT8_C(  71), -INT8_C( 106),  INT8_C(  10),  INT8_C(  21) },
      { -INT8_C(  50),  INT8_C(  42), -INT8_C( 104),  INT8_C(  30),  INT8_C(  19),  INT8_C( 104),  INT8_C(  52), -INT8_C( 119),
        -INT8_C(  17), -INT8_C(  64),  INT8_C( 120),  INT8_C(  57), -INT8_C(  74),  INT8_C(  39), -INT8_C( 116), -INT8_C(  89),
        -INT8_C(  63),  INT8_C( 104),  INT8_C(  74), -INT8_C(  93),  INT8_C(  69),  INT8_C(  37),  INT8_C( 126),  INT8_C( 106),
         INT8_C( 100),  INT8_C(  74), -INT8_C(  52), -INT8_C(  95), -INT8_C(  32), -INT8_C(   2),  INT8_C(  10),  INT8_C(  21) },
      UINT32_C( 980740768) },
    { {  INT8_C(  82), -INT8_C(  67), -INT8_C(  60), -INT8_C(  43),  INT8_C( 125),  INT8_C(  19),  INT8_C(  15),  INT8_C(  99),
         INT8_C(  58), -INT8_C( 101),  INT8_C(  10), -INT8_C(  54),  INT8_C(  89),  INT8_C(  85), -INT8_C(  35), -INT8_C(  97),
         INT8_C( 122),  INT8_C(  91),  INT8_C(  73), -INT8_C(  44), -INT8_C(  91),  INT8_C(  60),  INT8_C( 117), -INT8_C( 123),
         INT8_C(  58),  INT8_C(  43),  INT8_C(  38), -INT8_C(  37),  INT8_C(  21), -INT8_C( 102),  INT8_C(  21),  INT8_C( 103) },
      {  INT8_C(  82), -INT8_C(  39), -INT8_C(  60), -INT8_C(  43),  INT8_C( 125),  INT8_C(  76),  INT8_C(  56),  INT8_C(  99),
        -INT8_C(  25), -INT8_C( 101),  INT8_C(  10), -INT8_C(  54),  INT8_C(  89),  INT8_C(  85), -INT8_C(  32), -INT8_C(  97),
         INT8_C( 122),  INT8_C(  41),  INT8_C(  73), -INT8_C(  44), -INT8_C(  91),  INT8_C(  60),  INT8_C(  85), -INT8_C( 123),
        -INT8_C( 121),  INT8_C( 123),  INT8_C( 123), -INT8_C(  37),  INT8_C(  22), -INT8_C( 102),  INT8_C(   4),  INT8_C( 103) },
      UINT32_C(1463959906) },
    { { -INT8_C( 115),  INT8_C( 123),  INT8_C( 126),  INT8_C( 116), -INT8_C(  66),  INT8_C( 112), -INT8_C(  75),  INT8_C(  86),
         INT8_C(  63), -INT8_C( 107),  INT8_C( 104),  INT8_C( 105), -INT8_C(  66),  INT8_C(  79),  INT8_C(  57),  INT8_C(  36),
        -INT8_C(  86), -INT8_C( 113), -INT8_C(  60),  INT8_C(  49),  INT8_C(  10),  INT8_C(  63), -INT8_C(  50),  INT8_C(  32),
        -INT8_C(  49), -INT8_C(  46), -INT8_C( 114),  INT8_C(  57),  INT8_C(  19), -INT8_C(  48), -INT8_C( 112), -INT8_C(  96) },
      {  INT8_C(  75),  INT8_C( 123),  INT8_C(  20),  INT8_C(   9), -INT8_C(  66), -INT8_C(  54),  INT8_C(  95), -INT8_C(  66),
         INT8_C(  63), -INT8_C(  56),  INT8_C( 104),  INT8_C(  30),  INT8_C(  23),  INT8_C(  79),  INT8_C(  66),  INT8_C(  36),
        -INT8_C(  86),  INT8_C(   6), -INT8_C(  13), -INT8_C(   6),  INT8_C(  10),  INT8_C(  63),  INT8_C(  27),  INT8_C(  20),
        -INT8_C(  49), -INT8_C(  46),  INT8_C(  78), -INT8_C(  90),  INT8_C(  19), -INT8_C(  34),  INT8_C(  70), -INT8_C(  96) },
      UINT32_C(1825463021) },
    { {  INT8_C(  36),  INT8_C(  46),  INT8_C(  42), -INT8_C( 124), -INT8_C(  10),  INT8_C(  82), -INT8_C(  94),  INT8_C(  13),
        -INT8_C(  77), -INT8_C(  28), -INT8_C(  50), -INT8_C(  93), -INT8_C(  22), -INT8_C(  63), -INT8_C(  99),  INT8_C(  47),
        -INT8_C( 126), -INT8_C(  72),  INT8_C(  67),  INT8_C(  21),  INT8_C(  97), -INT8_C( 111), -INT8_C(  69), -INT8_C(  37),
         INT8_C( 112),  INT8_C(   1), -INT8_C(  96),  INT8_C(  93),  INT8_C(  92),  INT8_C( 110), -INT8_C(  54),      INT8_MIN },
      {  INT8_C(  36),  INT8_C(  46),  INT8_C(   4), -INT8_C( 124), -INT8_C(  10),  INT8_C(  82), -INT8_C(  94),  INT8_C(  13),
        -INT8_C( 118), -INT8_C(  28), -INT8_C( 100), -INT8_C(  93), -INT8_C(  22),  INT8_C(  58), -INT8_C(  93),  INT8_C(  47),
        -INT8_C( 126), -INT8_C(  72), -INT8_C(  57),  INT8_C(  84),  INT8_C(  97), -INT8_C( 125),  INT8_C(  47), -INT8_C(  37),
        -INT8_C( 124), -INT8_C(  49), -INT8_C(  96), -INT8_C(  32),  INT8_C(  92),  INT8_C( 110),  INT8_C(  97),      INT8_MIN },
      UINT32_C(1265394948) },
    { {  INT8_C(  12),  INT8_C(  12),  INT8_C(  68), -INT8_C( 106),  INT8_C( 122), -INT8_C(  31),  INT8_C(  11), -INT8_C(  87),
         INT8_C(  27), -INT8_C(  82),  INT8_C(  91),  INT8_C(  13), -INT8_C( 107),  INT8_C(  35),  INT8_C(  97),  INT8_C(  14),
        -INT8_C(  90), -INT8_C( 112), -INT8_C(  10),  INT8_C(  42),  INT8_C(  95),  INT8_C(  60),  INT8_C(  11), -INT8_C(  99),
         INT8_C(  76),  INT8_C( 108),  INT8_C( 119),  INT8_C(  81), -INT8_C(  47), -INT8_C(  29), -INT8_C( 100), -INT8_C(  35) },
      {  INT8_C(  12), -INT8_C(  32),  INT8_C( 116),  INT8_C( 105), -INT8_C(  63), -INT8_C(  31),  INT8_C(  11), -INT8_C(  87),
         INT8_C(  27), -INT8_C(  82), -INT8_C(  22), -INT8_C(  61), -INT8_C( 111),  INT8_C(  75), -INT8_C(  47),  INT8_C(  55),
        -INT8_C(  90), -INT8_C( 112),  INT8_C(  98),  INT8_C(  59),  INT8_C(  95),  INT8_C(  60),  INT8_C(  11),  INT8_C(  80),
        -INT8_C(  39),  INT8_C(  79), -INT8_C(  95), -INT8_C(  86),  INT8_C(  51), -INT8_C(  29), -INT8_C( 120),  INT8_C(  34) },
      UINT32_C(3750558750) },
    { {  INT8_C( 123), -INT8_C(  97), -INT8_C(  68), -INT8_C(  88),  INT8_C(  13), -INT8_C(  90),  INT8_C( 107), -INT8_C(  97),
        -INT8_C(  15),  INT8_C(  60), -INT8_C(  42), -INT8_C(  51),  INT8_C(   4),  INT8_C(  56),  INT8_C(   9),  INT8_C(   8),
        -INT8_C(  91), -INT8_C(  31),  INT8_C(  88),  INT8_C( 126),  INT8_C(  49), -INT8_C(   6),  INT8_C(  41),  INT8_C( 100),
         INT8_C(  55), -INT8_C(  79), -INT8_C( 122),  INT8_C(  85), -INT8_C(  83),  INT8_C(  18),  INT8_C(  53),  INT8_C(  40) },
      {  INT8_C( 123), -INT8_C(  15), -INT8_C(  68), -INT8_C(  65),  INT8_C(  13), -INT8_C(  90),  INT8_C( 107), -INT8_C( 120),
        -INT8_C(  15),  INT8_C(  60), -INT8_C(  42), -INT8_C(  51),  INT8_C( 109),  INT8_C(  56),  INT8_C(   9),  INT8_C(   8),
         INT8_C(  64), -INT8_C(  35), -INT8_C( 111),  INT8_C( 113), -INT8_C(  41), -INT8_C(   6), -INT8_C(  43),  INT8_C(  14),
         INT8_C( 107), -INT8_C(  79), -INT8_C( 122),  INT8_C(  85), -INT8_C(  83), -INT8_C( 103),  INT8_C(  53),  INT8_C(  40) },
      UINT32_C( 568266890) },
    { {  INT8_C(  76),  INT8_C(  61), -INT8_C(  87), -INT8_C(  59),  INT8_C( 113), -INT8_C(   1),  INT8_C(  65), -INT8_C(  34),
         INT8_C(  94), -INT8_C(  58), -INT8_C(  15), -INT8_C(  97), -INT8_C(  93), -INT8_C( 126),  INT8_C(  16),  INT8_C( 122),
         INT8_C(  60), -INT8_C(  26), -INT8_C( 120), -INT8_C(  89),  INT8_C(  66), -INT8_C(  20), -INT8_C(  65), -INT8_C(  80),
        -INT8_C( 123), -INT8_C(   1), -INT8_C(  48),  INT8_C(  15),  INT8_C(  15), -INT8_C(  81),  INT8_C(  48),  INT8_C(  92) },
      {  INT8_C(  76), -INT8_C(  38), -INT8_C(  87),  INT8_C(  94), -INT8_C(  39), -INT8_C(   1),  INT8_C(  60), -INT8_C(  34),
         INT8_C(  40), -INT8_C(  58), -INT8_C(  41), -INT8_C(  53), -INT8_C(  81), -INT8_C( 126),  INT8_C(  69), -INT8_C(  21),
         INT8_C(  60), -INT8_C(  50), -INT8_C( 120),  INT8_C(  15),  INT8_C(  66), -INT8_C(  20), -INT8_C(  65),  INT8_C(  64),
         INT8_C(  80), -INT8_C( 112), -INT8_C(  48),  INT8_C(  15),  INT8_C(  64),      INT8_MIN,  INT8_C(  48),  INT8_C(  92) },
      UINT32_C( 864738650) },
    { {  INT8_C(  63), -INT8_C(  57),  INT8_C( 107),  INT8_C( 104), -INT8_C(  12),  INT8_C(  66),  INT8_C(  51), -INT8_C(  92),
         INT8_C(  42),  INT8_C( 121), -INT8_C( 113), -INT8_C(   9),  INT8_C(  71),  INT8_C(  34),  INT8_C(   7),  INT8_C(   1),
         INT8_C( 115), -INT8_C(  57),  INT8_C(  65), -INT8_C(  60),  INT8_C(  87), -INT8_C( 111),  INT8_C(  36), -INT8_C( 105),
         INT8_C(  17), -INT8_C(  32), -INT8_C(  60),  INT8_C( 107), -INT8_C(  67),  INT8_C(  78), -INT8_C(  98), -INT8_C(   4) },
      {  INT8_C(  63), -INT8_C(  57),  INT8_C( 107),  INT8_C( 104),  INT8_C(  76),  INT8_C(  66), -INT8_C(  82), -INT8_C(  92),
         INT8_C(  17),  INT8_C( 121), -INT8_C( 113),  INT8_C(  88),  INT8_C(  95),  INT8_C( 117),  INT8_C(   7), -INT8_C(  45),
         INT8_C(  60), -INT8_C(  57), -INT8_C( 105), -INT8_C( 109),  INT8_C(  44), -INT8_C( 111),  INT8_C(  43), -INT8_C( 105),
        -INT8_C( 101), -INT8_C(  32), -INT8_C(  88),  INT8_C(  88),  INT8_C(  61),  INT8_C(  78), -INT8_C(  98),  INT8_C(  83) },
      UINT32_C(2640165200) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi8(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_i8x32());
    easysimd__mmask32 r = easysimd_mm256_cmpneq_epi8_mask(a, b);

    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const int8_t a[32];
    const int8_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(2633305875),
      {  INT8_C(  94),  INT8_C(  57),  INT8_C( 114), -INT8_C( 115), -INT8_C(  89),  INT8_C(  68), -INT8_C(  54), -INT8_C(  15),
        -INT8_C( 118),  INT8_C(   4), -INT8_C(  40), -INT8_C(  27),  INT8_C(  58),  INT8_C(  16),  INT8_C(   7), -INT8_C( 125),
        -INT8_C( 127),  INT8_C(  11), -INT8_C(  48), -INT8_C(  10), -INT8_C(  12), -INT8_C(  67),  INT8_C(  14),  INT8_C(  64),
        -INT8_C(  29),  INT8_C(   9),  INT8_C(  22), -INT8_C(  10),  INT8_C(  24),  INT8_C(  11), -INT8_C( 109),  INT8_C( 119) },
      {  INT8_C(  94),  INT8_C(   5),  INT8_C(   4), -INT8_C(  20), -INT8_C(  89),  INT8_C(  68), -INT8_C(  54), -INT8_C(  45),
        -INT8_C(  46), -INT8_C(  75), -INT8_C(  72), -INT8_C(  27),  INT8_C(  58), -INT8_C(  64), -INT8_C( 113),  INT8_C(  70),
        -INT8_C( 127),  INT8_C(  95), -INT8_C(  48), -INT8_C(  10),  INT8_C(  28), -INT8_C(  67),  INT8_C(   0),  INT8_C(  64),
        -INT8_C(  29),  INT8_C(   9),  INT8_C(  22),  INT8_C( 108),  INT8_C(  34),  INT8_C(  11), -INT8_C(  29),  INT8_C( 102) },
      UINT32_C(2555381506) },
    { UINT32_C(2292920245),
      { -INT8_C(  28),  INT8_C( 100), -INT8_C( 108), -INT8_C(  87),  INT8_C(  36),  INT8_C(  36), -INT8_C(  17), -INT8_C(  17),
        -INT8_C( 125),  INT8_C(  43), -INT8_C(  81), -INT8_C(  96),  INT8_C( 118), -INT8_C(  81), -INT8_C(  96), -INT8_C(  54),
        -INT8_C(  58), -INT8_C( 106),  INT8_C(  54), -INT8_C(  24),  INT8_C(  32),  INT8_C(  26),  INT8_C(  78), -INT8_C(  82),
         INT8_C(   1), -INT8_C(  95), -INT8_C( 122), -INT8_C(  73), -INT8_C(  48),  INT8_C(  50),  INT8_C(  63), -INT8_C(  75) },
      { -INT8_C( 106),  INT8_C( 100),  INT8_C(  94), -INT8_C(  87), -INT8_C(   9),  INT8_C(  36), -INT8_C(  87),  INT8_C( 123),
        -INT8_C( 125),  INT8_C(  89), -INT8_C(  81), -INT8_C(  17),  INT8_C( 118), -INT8_C(  81), -INT8_C(  96), -INT8_C(  50),
        -INT8_C(  58), -INT8_C( 106),  INT8_C(  54),  INT8_C( 113),  INT8_C(  32),  INT8_C(   5),  INT8_C(  78), -INT8_C(  82),
         INT8_C(   1), -INT8_C(  95), -INT8_C(  62),  INT8_C( 118), -INT8_C(  48),  INT8_C(  50),  INT8_C(  43),  INT8_C( 110) },
      UINT32_C(2284325525) },
    { UINT32_C(1363661528),
      {  INT8_C(  43),  INT8_C(  98),  INT8_C(  65),  INT8_C(  51),  INT8_C(  29), -INT8_C(   6),  INT8_C(   2),  INT8_C( 111),
        -INT8_C(  22), -INT8_C(  72), -INT8_C(  32), -INT8_C(  12), -INT8_C(  67),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),
        -INT8_C(  89), -INT8_C(  62), -INT8_C(  38),      INT8_MAX, -INT8_C(  60),  INT8_C(   5), -INT8_C(  18), -INT8_C( 103),
        -INT8_C( 113),  INT8_C(  22),  INT8_C( 101),  INT8_C( 103), -INT8_C(  24), -INT8_C(  83), -INT8_C(  71),  INT8_C(  19) },
      {  INT8_C(  43),  INT8_C(  98),  INT8_C(  71),  INT8_C(  45),  INT8_C(  29),  INT8_C(  73),  INT8_C(   2), -INT8_C(  33),
         INT8_C(   1),  INT8_C( 124), -INT8_C(  32), -INT8_C(  65), -INT8_C(  67), -INT8_C(  45),  INT8_C(  34),  INT8_C(  36),
        -INT8_C(  89), -INT8_C(   4), -INT8_C(  38),      INT8_MAX, -INT8_C(  60),  INT8_C(   5), -INT8_C(  13), -INT8_C( 103),
        -INT8_C( 113),  INT8_C(  22),  INT8_C( 101),  INT8_C( 103), -INT8_C(  24), -INT8_C(  78), -INT8_C(  71),  INT8_C(  21) },
      UINT32_C(   4375176) },
    { UINT32_C( 897572404),
      {  INT8_C(  90),  INT8_C(  83), -INT8_C(  12), -INT8_C(  41),  INT8_C(  38),  INT8_C(  23), -INT8_C(   5), -INT8_C(  68),
         INT8_C(  19), -INT8_C(  97),  INT8_C(  22),  INT8_C(  21),  INT8_C(  48),  INT8_C(   9), -INT8_C(  89), -INT8_C(  40),
         INT8_C(  98), -INT8_C(  96),  INT8_C( 105),  INT8_C( 103),  INT8_C(  82),  INT8_C(  13),  INT8_C( 124), -INT8_C(   2),
        -INT8_C(   8), -INT8_C(  66), -INT8_C(  98),  INT8_C(  44), -INT8_C( 100),  INT8_C(  30),  INT8_C(  97), -INT8_C(   9) },
      {  INT8_C( 113),  INT8_C(  86), -INT8_C(  12), -INT8_C(  41),  INT8_C(  38),  INT8_C(  23),  INT8_C(  84),      INT8_MIN,
         INT8_C(  19),  INT8_C( 106),  INT8_C(  22), -INT8_C( 103),  INT8_C( 116),  INT8_C(   9),  INT8_C( 114), -INT8_C(  42),
         INT8_C(  98), -INT8_C(  37),  INT8_C( 105),  INT8_C( 103), -INT8_C(  24), -INT8_C(  70),  INT8_C(  45), -INT8_C(  32),
        -INT8_C(   8), -INT8_C(  66), -INT8_C(  98),  INT8_C(  44), -INT8_C(  23),  INT8_C( 109),  INT8_C(  97), -INT8_C(   9) },
      UINT32_C( 812833280) },
    { UINT32_C( 229721764),
      { -INT8_C(  80),  INT8_C(  71), -INT8_C(  89),  INT8_C(  36), -INT8_C( 124),  INT8_C(  25), -INT8_C(   6),  INT8_C(  97),
        -INT8_C(  12),  INT8_C(  56), -INT8_C( 112), -INT8_C(  36), -INT8_C(  14), -INT8_C(  67), -INT8_C(  68),  INT8_C( 106),
        -INT8_C( 120), -INT8_C(  56),      INT8_MAX,  INT8_C( 114),  INT8_C(  53), -INT8_C( 117), -INT8_C(  52), -INT8_C(   7),
         INT8_C( 102), -INT8_C(  66),  INT8_C(  41),  INT8_C(  10),  INT8_C(   4), -INT8_C(  38),  INT8_C(  24), -INT8_C(  75) },
      {  INT8_C(  33), -INT8_C(  65), -INT8_C(  89), -INT8_C(  91), -INT8_C(  40), -INT8_C(  44), -INT8_C(   6), -INT8_C(  52),
        -INT8_C(  12),  INT8_C(  56), -INT8_C(  88), -INT8_C(   2), -INT8_C(  14),  INT8_C( 100),  INT8_C( 104),  INT8_C( 106),
         INT8_C(  44), -INT8_C(  56),  INT8_C(  78),  INT8_C( 114),  INT8_C(  53),  INT8_C(  26), -INT8_C(  52), -INT8_C(   7),
        -INT8_C(  39), -INT8_C( 124),  INT8_C(  41),  INT8_C(  10),  INT8_C(  94), -INT8_C(  38),  INT8_C(  24),      INT8_MIN },
      UINT32_C(  18957472) },
    { UINT32_C(1281305664),
      { -INT8_C(  62),  INT8_C(   7),  INT8_C(  74),  INT8_C(  22),  INT8_C( 107), -INT8_C(  78), -INT8_C(  14), -INT8_C( 105),
        -INT8_C( 102),  INT8_C(  64), -INT8_C(   8),  INT8_C(  14),  INT8_C(  90),  INT8_C(  83), -INT8_C(  25),  INT8_C(  51),
        -INT8_C(  41), -INT8_C(  53),  INT8_C(  17),  INT8_C(  53), -INT8_C(  57), -INT8_C(  93), -INT8_C(  75), -INT8_C( 126),
         INT8_C(  15), -INT8_C(  37),  INT8_C(  21),  INT8_C(  79),  INT8_C(   7),  INT8_C( 116), -INT8_C( 101), -INT8_C(  55) },
      { -INT8_C(  62),  INT8_C(   7), -INT8_C(  33),  INT8_C(  22), -INT8_C( 104), -INT8_C(  47), -INT8_C(  14), -INT8_C( 105),
        -INT8_C( 102),  INT8_C( 118),  INT8_C(  64),  INT8_C(  14), -INT8_C(  55),  INT8_C(  83), -INT8_C(  25),  INT8_C(  51),
        -INT8_C(  13), -INT8_C(  53),  INT8_C(  17), -INT8_C(  69), -INT8_C(  57), -INT8_C( 117), -INT8_C(  75), -INT8_C( 126),
         INT8_C(  15), -INT8_C(  37), -INT8_C(  77),  INT8_C( 109),  INT8_C(   7),  INT8_C( 116),  INT8_C(  54),  INT8_C(  67) },
      UINT32_C(1275659264) },
    { UINT32_C(4194215911),
      {  INT8_C(  29),  INT8_C(  63),  INT8_C( 101), -INT8_C(  26),  INT8_C( 103),  INT8_C(   4), -INT8_C( 122),  INT8_C(  90),
        -INT8_C(  75),  INT8_C(  91),  INT8_C(  21),  INT8_C(   9), -INT8_C(  26),  INT8_C(  83),  INT8_C( 108),  INT8_C(  76),
        -INT8_C(  90),  INT8_C(  31), -INT8_C(  71),  INT8_C( 109),  INT8_C( 110), -INT8_C(  16), -INT8_C(  80), -INT8_C(  94),
         INT8_C(   6), -INT8_C(  38),  INT8_C( 110), -INT8_C(  19), -INT8_C( 127),  INT8_C( 108), -INT8_C(  26), -INT8_C(  98) },
      { -INT8_C(  85),  INT8_C(  75), -INT8_C( 124), -INT8_C(  26),  INT8_C( 103),  INT8_C(  10),  INT8_C( 109),  INT8_C(  90),
        -INT8_C(  75), -INT8_C( 126),  INT8_C(  14),  INT8_C(   9), -INT8_C(  43),  INT8_C(  83),  INT8_C( 108),  INT8_C(  76),
        -INT8_C(  90),  INT8_C(  82), -INT8_C(  71),  INT8_C( 109),  INT8_C( 110), -INT8_C(  16), -INT8_C(  86), -INT8_C(  94),
         INT8_C( 115),  INT8_C(  24),  INT8_C(  53), -INT8_C(  19), -INT8_C( 124),  INT8_C(  28), -INT8_C(  26),  INT8_C(  48) },
      UINT32_C(2973894247) },
    { UINT32_C(2260512544),
      {  INT8_C(  50), -INT8_C(  54), -INT8_C(  46),  INT8_C(   7),  INT8_C(  69),  INT8_C( 106), -INT8_C( 125), -INT8_C(  33),
        -INT8_C(  68),  INT8_C( 108), -INT8_C(  25), -INT8_C(   2),  INT8_C(   5), -INT8_C( 111),  INT8_C(  70),  INT8_C( 121),
        -INT8_C(  87),  INT8_C( 124),  INT8_C( 109),  INT8_C(  45), -INT8_C( 104),  INT8_C(   0),  INT8_C(  93), -INT8_C(   1),
         INT8_C(  22), -INT8_C(  96), -INT8_C(  73),  INT8_C(  55),  INT8_C(  79),  INT8_C( 115), -INT8_C(  67), -INT8_C( 127) },
      {  INT8_C(  62), -INT8_C(  54), -INT8_C( 119), -INT8_C( 125), -INT8_C(   7),  INT8_C(  12),  INT8_C(  98), -INT8_C(  74),
        -INT8_C(  68),  INT8_C( 108), -INT8_C(  25),  INT8_C( 125),  INT8_C(   5), -INT8_C( 111),  INT8_C(  70),  INT8_C( 121),
        -INT8_C(  87),  INT8_C( 100),  INT8_C( 109),  INT8_C(  15),  INT8_C( 100),  INT8_C(  14),  INT8_C(  14),  INT8_C( 122),
        -INT8_C(  82), -INT8_C(  59), -INT8_C(  79),  INT8_C(  55),  INT8_C(  57),  INT8_C( 110),      INT8_MAX,  INT8_C( 119) },
      UINT32_C(2260207648) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__mmask32 k1 = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epi8_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epi8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi8(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_i8x32());
    easysimd__mmask32 r = easysimd_mm256_mask_cmpneq_epi8_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t a[32];
    const uint8_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { { UINT8_C( 20), UINT8_C( 92), UINT8_C(173), UINT8_C(140), UINT8_C(165), UINT8_C( 97), UINT8_C(  9), UINT8_C(127),
        UINT8_C( 92), UINT8_C(  0), UINT8_C(  2), UINT8_C(211), UINT8_C(100), UINT8_C(178), UINT8_C(226), UINT8_C(200),
        UINT8_C(192), UINT8_C(241), UINT8_C( 66), UINT8_C(110), UINT8_C(182), UINT8_C(244), UINT8_C(108), UINT8_C(239),
        UINT8_C( 98), UINT8_C(235), UINT8_C(102), UINT8_C( 96), UINT8_C(243), UINT8_C( 96), UINT8_C( 87), UINT8_C(  7) },
      { UINT8_C( 20), UINT8_C( 92), UINT8_C(147), UINT8_C( 97), UINT8_C(101), UINT8_C( 97), UINT8_C(  9), UINT8_C(127),
        UINT8_C( 92), UINT8_C(  0), UINT8_C(  2), UINT8_C(  0), UINT8_C(100), UINT8_C(120), UINT8_C(226), UINT8_C( 85),
        UINT8_C(192), UINT8_C(241), UINT8_C(196), UINT8_C(110),    UINT8_MAX, UINT8_C( 48), UINT8_C( 15), UINT8_C( 97),
        UINT8_C( 98), UINT8_C(117), UINT8_C(102), UINT8_C( 96), UINT8_C(243), UINT8_C( 96), UINT8_C( 87), UINT8_C(146) },
      UINT32_C(2197071900) },
    { { UINT8_C( 68), UINT8_C(212), UINT8_C( 68), UINT8_C(225), UINT8_C(183), UINT8_C(217), UINT8_C(225), UINT8_C( 76),
        UINT8_C( 81), UINT8_C(170), UINT8_C(161), UINT8_C(186), UINT8_C(181), UINT8_C(101), UINT8_C(218), UINT8_C(180),
        UINT8_C(149), UINT8_C(233), UINT8_C( 21), UINT8_C(176), UINT8_C( 94), UINT8_C(215), UINT8_C(190), UINT8_C( 52),
        UINT8_C(239), UINT8_C(211), UINT8_C(199), UINT8_C( 12), UINT8_C(123), UINT8_C(187), UINT8_C(142), UINT8_C(192) },
      { UINT8_C(143), UINT8_C(212), UINT8_C(161), UINT8_C( 70), UINT8_C(171), UINT8_C(217), UINT8_C(146), UINT8_C(253),
        UINT8_C( 81), UINT8_C( 52), UINT8_C(183), UINT8_C(225), UINT8_C(181), UINT8_C(101), UINT8_C(149), UINT8_C(180),
        UINT8_C(122), UINT8_C(233), UINT8_C(223), UINT8_C(176), UINT8_C( 94), UINT8_C(158), UINT8_C( 13), UINT8_C( 52),
        UINT8_C(113), UINT8_C(211), UINT8_C(199), UINT8_C(237), UINT8_C(123), UINT8_C(187), UINT8_C(142), UINT8_C( 31) },
      UINT32_C(2305117917) },
    { { UINT8_C(208), UINT8_C(248), UINT8_C(134), UINT8_C(253), UINT8_C( 44), UINT8_C( 61), UINT8_C(222), UINT8_C(197),
        UINT8_C(207), UINT8_C(116), UINT8_C(244), UINT8_C( 73), UINT8_C( 31), UINT8_C(212), UINT8_C( 34), UINT8_C(161),
        UINT8_C(114), UINT8_C( 48), UINT8_C( 18), UINT8_C(227), UINT8_C(  4), UINT8_C(144), UINT8_C(208), UINT8_C(148),
        UINT8_C(155), UINT8_C(125), UINT8_C(179), UINT8_C(121), UINT8_C(203), UINT8_C( 24), UINT8_C(  2), UINT8_C(156) },
      { UINT8_C(208), UINT8_C(248), UINT8_C(153), UINT8_C( 60), UINT8_C( 44), UINT8_C(119), UINT8_C(  2), UINT8_C(148),
        UINT8_C(207), UINT8_C(246), UINT8_C(222), UINT8_C( 10), UINT8_C( 31), UINT8_C(  0), UINT8_C(171), UINT8_C(161),
        UINT8_C( 48), UINT8_C( 48), UINT8_C( 18), UINT8_C(227), UINT8_C(  4), UINT8_C(240), UINT8_C(201), UINT8_C(233),
        UINT8_C(155), UINT8_C(124), UINT8_C(179), UINT8_C(121), UINT8_C(148), UINT8_C(100), UINT8_C(  2), UINT8_C(165) },
      UINT32_C(3001118444) },
    { { UINT8_C(230), UINT8_C(227), UINT8_C( 70), UINT8_C(209), UINT8_C(218), UINT8_C( 36), UINT8_C(220), UINT8_C(164),
        UINT8_C( 37), UINT8_C(135), UINT8_C(225), UINT8_C( 85), UINT8_C( 69), UINT8_C(  1), UINT8_C(138), UINT8_C(147),
        UINT8_C(241), UINT8_C( 83), UINT8_C(125), UINT8_C( 95), UINT8_C(207), UINT8_C(223), UINT8_C(153), UINT8_C(100),
        UINT8_C( 68), UINT8_C(110), UINT8_C(  9), UINT8_C( 48), UINT8_C(221), UINT8_C(234), UINT8_C(226), UINT8_C(195) },
      { UINT8_C(206), UINT8_C(227), UINT8_C(148), UINT8_C(168), UINT8_C(218), UINT8_C(112), UINT8_C(220), UINT8_C(114),
        UINT8_C(248), UINT8_C( 45), UINT8_C(225), UINT8_C( 61), UINT8_C( 46), UINT8_C( 82), UINT8_C(138), UINT8_C( 32),
        UINT8_C(166), UINT8_C( 78), UINT8_C(125), UINT8_C( 95), UINT8_C(207), UINT8_C(223), UINT8_C(217), UINT8_C(100),
        UINT8_C(135), UINT8_C(226), UINT8_C(  9), UINT8_C(100), UINT8_C(205), UINT8_C(132), UINT8_C( 39), UINT8_C(155) },
      UINT32_C(4215520173) },
    { { UINT8_C( 44), UINT8_C(143), UINT8_C(109), UINT8_C( 36), UINT8_C(189), UINT8_C( 53), UINT8_C( 97), UINT8_C(235),
        UINT8_C(136), UINT8_C( 50), UINT8_C( 11), UINT8_C( 46), UINT8_C(128), UINT8_C(139), UINT8_C(163), UINT8_C(174),
        UINT8_C(163), UINT8_C(125), UINT8_C( 31), UINT8_C( 42), UINT8_C( 95), UINT8_C(193), UINT8_C(142), UINT8_C( 44),
        UINT8_C( 70), UINT8_C(181), UINT8_C(199), UINT8_C(243), UINT8_C(113), UINT8_C( 10), UINT8_C(238), UINT8_C(157) },
      { UINT8_C( 44), UINT8_C(143), UINT8_C(109), UINT8_C( 87), UINT8_C(189), UINT8_C( 34), UINT8_C( 66), UINT8_C( 25),
        UINT8_C(136), UINT8_C( 78), UINT8_C( 11), UINT8_C(213), UINT8_C(217), UINT8_C(235), UINT8_C(163), UINT8_C(124),
        UINT8_C(104), UINT8_C(163), UINT8_C(167), UINT8_C( 42), UINT8_C(100), UINT8_C( 53), UINT8_C(142), UINT8_C(170),
        UINT8_C( 70), UINT8_C(187), UINT8_C(199), UINT8_C( 92), UINT8_C(198), UINT8_C(140), UINT8_C(249), UINT8_C(157) },
      UINT32_C(2058861288) },
    { { UINT8_C(220), UINT8_C(249), UINT8_C(147), UINT8_C( 49), UINT8_C( 71), UINT8_C(219), UINT8_C(  7), UINT8_C( 32),
        UINT8_C(198), UINT8_C(138), UINT8_C(157), UINT8_C( 46), UINT8_C( 45), UINT8_C( 68), UINT8_C(245), UINT8_C(146),
        UINT8_C(121), UINT8_C(233), UINT8_C( 60), UINT8_C(100), UINT8_C(165), UINT8_C(218), UINT8_C(192), UINT8_C(107),
        UINT8_C(103), UINT8_C(185), UINT8_C(203), UINT8_C( 79), UINT8_C(115), UINT8_C(130), UINT8_C(201), UINT8_C( 80) },
      { UINT8_C(220), UINT8_C( 93), UINT8_C(129), UINT8_C( 49), UINT8_C( 71), UINT8_C(219), UINT8_C(227), UINT8_C(254),
        UINT8_C( 19), UINT8_C(138), UINT8_C(157), UINT8_C( 46), UINT8_C( 45), UINT8_C( 33), UINT8_C(210), UINT8_C(146),
        UINT8_C(121), UINT8_C( 15), UINT8_C(162), UINT8_C(100), UINT8_C(165), UINT8_C( 99), UINT8_C(192), UINT8_C( 80),
        UINT8_C(103), UINT8_C(230), UINT8_C(160), UINT8_C(144), UINT8_C(104), UINT8_C(105), UINT8_C(224), UINT8_C(227) },
      UINT32_C(4272316870) },
    { { UINT8_C(234), UINT8_C(138), UINT8_C(252), UINT8_C(253), UINT8_C( 10), UINT8_C( 40), UINT8_C( 61), UINT8_C(207),
        UINT8_C( 74), UINT8_C( 16), UINT8_C( 13), UINT8_C( 85), UINT8_C( 31), UINT8_C(175), UINT8_C(  5), UINT8_C(  8),
        UINT8_C( 18), UINT8_C( 32), UINT8_C( 89), UINT8_C( 47), UINT8_C(  6), UINT8_C(249), UINT8_C(191), UINT8_C(110),
        UINT8_C( 98), UINT8_C(159), UINT8_C( 81), UINT8_C( 41), UINT8_C(  0), UINT8_C(248), UINT8_C( 39), UINT8_C(234) },
      { UINT8_C(130), UINT8_C(138), UINT8_C(231), UINT8_C(253), UINT8_C( 10), UINT8_C( 40), UINT8_C( 61), UINT8_C(150),
        UINT8_C( 74), UINT8_C(104), UINT8_C( 13), UINT8_C( 84), UINT8_C( 31), UINT8_C(175), UINT8_C(  5), UINT8_C( 42),
        UINT8_C( 18), UINT8_C( 32), UINT8_C( 89), UINT8_C( 22), UINT8_C(174), UINT8_C(249), UINT8_C(132), UINT8_C( 17),
        UINT8_C( 98), UINT8_C(214), UINT8_C( 81), UINT8_C( 41), UINT8_C(206), UINT8_C(248), UINT8_C(162), UINT8_C( 80) },
      UINT32_C(3537406597) },
    { { UINT8_C(175), UINT8_C( 56), UINT8_C(104), UINT8_C(228), UINT8_C(160), UINT8_C( 84), UINT8_C( 56), UINT8_C(184),
        UINT8_C( 68), UINT8_C(148), UINT8_C(227), UINT8_C( 85), UINT8_C( 74), UINT8_C( 60), UINT8_C(107), UINT8_C(248),
        UINT8_C( 85), UINT8_C(240), UINT8_C(  9), UINT8_C( 12), UINT8_C(198), UINT8_C( 67), UINT8_C(196), UINT8_C(148),
        UINT8_C(165), UINT8_C(103), UINT8_C(228), UINT8_C( 42), UINT8_C(241), UINT8_C(192), UINT8_C(252), UINT8_C(160) },
      { UINT8_C(248), UINT8_C( 56), UINT8_C(104), UINT8_C(228), UINT8_C(185), UINT8_C( 84), UINT8_C( 56), UINT8_C(184),
        UINT8_C( 80), UINT8_C( 52), UINT8_C(227), UINT8_C( 85), UINT8_C(113), UINT8_C( 60), UINT8_C(107), UINT8_C(248),
        UINT8_C( 85), UINT8_C(156), UINT8_C(210), UINT8_C(116), UINT8_C(224), UINT8_C(151), UINT8_C(196), UINT8_C(148),
        UINT8_C(165), UINT8_C(236), UINT8_C(228), UINT8_C(239), UINT8_C(241), UINT8_C(192), UINT8_C(143), UINT8_C(165) },
      UINT32_C(3393065745) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epu8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epu8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi8(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u8x32());
    easysimd__mmask32 r = easysimd_mm256_cmpneq_epu8_mask(a, b);

    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const uint8_t a[32];
    const uint8_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C( 515966288),
      { UINT8_C(176), UINT8_C(116), UINT8_C(169), UINT8_C( 62), UINT8_C(127), UINT8_C(163), UINT8_C( 42), UINT8_C(254),
        UINT8_C(155), UINT8_C(101), UINT8_C( 19), UINT8_C( 53), UINT8_C(153), UINT8_C( 41), UINT8_C( 51), UINT8_C(145),
        UINT8_C(173), UINT8_C(250), UINT8_C(104), UINT8_C( 69), UINT8_C(162), UINT8_C( 11), UINT8_C( 45), UINT8_C(248),
        UINT8_C(232), UINT8_C( 28), UINT8_C(234), UINT8_C( 56), UINT8_C( 33), UINT8_C(171), UINT8_C( 86), UINT8_C(210) },
      { UINT8_C(176), UINT8_C(116), UINT8_C( 16), UINT8_C( 62), UINT8_C(163), UINT8_C( 58), UINT8_C( 42), UINT8_C( 62),
        UINT8_C(159), UINT8_C(101), UINT8_C( 19), UINT8_C( 56), UINT8_C(218), UINT8_C( 41), UINT8_C( 51), UINT8_C(145),
        UINT8_C(161), UINT8_C(250), UINT8_C(104), UINT8_C( 69), UINT8_C( 62), UINT8_C( 11), UINT8_C( 45), UINT8_C( 38),
        UINT8_C(232), UINT8_C( 28), UINT8_C(234), UINT8_C( 55), UINT8_C(209), UINT8_C(171), UINT8_C(  9), UINT8_C(210) },
      UINT32_C( 411107600) },
    { UINT32_C(4069928787),
      { UINT8_C(224), UINT8_C( 10), UINT8_C( 43), UINT8_C(187), UINT8_C(177), UINT8_C(245), UINT8_C( 66), UINT8_C( 83),
        UINT8_C( 39), UINT8_C( 15), UINT8_C(150), UINT8_C(101), UINT8_C(  9), UINT8_C(210), UINT8_C(139), UINT8_C( 31),
        UINT8_C(248), UINT8_C(233), UINT8_C( 86), UINT8_C(201), UINT8_C(158), UINT8_C( 96), UINT8_C(187), UINT8_C( 82),
        UINT8_C(121), UINT8_C( 76), UINT8_C(170), UINT8_C(205), UINT8_C(123), UINT8_C( 65), UINT8_C(191), UINT8_C( 91) },
      { UINT8_C( 75), UINT8_C(234), UINT8_C( 22), UINT8_C(187), UINT8_C(223), UINT8_C( 89), UINT8_C( 66), UINT8_C( 83),
        UINT8_C( 39), UINT8_C(230), UINT8_C(108), UINT8_C(113), UINT8_C(  9), UINT8_C(248), UINT8_C(144), UINT8_C( 31),
        UINT8_C(248), UINT8_C(230), UINT8_C(122), UINT8_C(201), UINT8_C(158), UINT8_C( 96), UINT8_C(187), UINT8_C( 82),
        UINT8_C(121), UINT8_C(124), UINT8_C(141), UINT8_C(205), UINT8_C(189), UINT8_C( 65), UINT8_C(191), UINT8_C( 91) },
      UINT32_C( 302394899) },
    { UINT32_C( 790451911),
      { UINT8_C( 60), UINT8_C(138), UINT8_C(160), UINT8_C(245), UINT8_C(130), UINT8_C( 48), UINT8_C(165), UINT8_C( 99),
        UINT8_C( 22), UINT8_C( 31), UINT8_C(227), UINT8_C( 93), UINT8_C( 84), UINT8_C(181), UINT8_C( 29), UINT8_C(213),
        UINT8_C( 49), UINT8_C(170), UINT8_C(209), UINT8_C(239), UINT8_C(246), UINT8_C( 41), UINT8_C(248), UINT8_C( 45),
        UINT8_C(151), UINT8_C(254), UINT8_C( 68), UINT8_C( 94), UINT8_C( 84), UINT8_C( 97), UINT8_C(141), UINT8_C(144) },
      { UINT8_C( 60), UINT8_C( 45), UINT8_C(133), UINT8_C(109), UINT8_C( 93), UINT8_C( 48), UINT8_C(165), UINT8_C( 99),
        UINT8_C( 74), UINT8_C( 31), UINT8_C(227), UINT8_C(159), UINT8_C( 84), UINT8_C(237), UINT8_C(116), UINT8_C(154),
        UINT8_C( 49), UINT8_C( 70), UINT8_C(209), UINT8_C(142), UINT8_C(111), UINT8_C(129), UINT8_C(248), UINT8_C(  6),
        UINT8_C(127),    UINT8_MAX, UINT8_C( 68), UINT8_C(211), UINT8_C( 97), UINT8_C(241), UINT8_C(100), UINT8_C(144) },
      UINT32_C( 723009542) },
    { UINT32_C(1609468692),
      { UINT8_C( 63), UINT8_C(191), UINT8_C(254), UINT8_C(168), UINT8_C(172), UINT8_C(114), UINT8_C( 66), UINT8_C( 68),
        UINT8_C(184), UINT8_C(204), UINT8_C(210), UINT8_C( 39), UINT8_C( 77), UINT8_C(141), UINT8_C( 45), UINT8_C(205),
        UINT8_C(141), UINT8_C(145), UINT8_C(160), UINT8_C(238), UINT8_C(130), UINT8_C(  4), UINT8_C( 58), UINT8_C(160),
        UINT8_C(238), UINT8_C(244), UINT8_C( 27), UINT8_C(  2), UINT8_C(127), UINT8_C( 10), UINT8_C( 97), UINT8_C(190) },
      { UINT8_C( 63), UINT8_C( 95), UINT8_C(102), UINT8_C(168), UINT8_C(172), UINT8_C(169), UINT8_C(185), UINT8_C(138),
        UINT8_C(117), UINT8_C(139), UINT8_C(178), UINT8_C( 39), UINT8_C( 77), UINT8_C(223), UINT8_C(143), UINT8_C(205),
        UINT8_C(141), UINT8_C(145), UINT8_C(160), UINT8_C(243), UINT8_C( 52), UINT8_C(206), UINT8_C(148), UINT8_C( 34),
        UINT8_C(238), UINT8_C(244), UINT8_C( 27), UINT8_C( 66), UINT8_C(185), UINT8_C(134), UINT8_C( 97), UINT8_C(130) },
      UINT32_C( 417858308) },
    { UINT32_C(2235740432),
      { UINT8_C( 61), UINT8_C(244), UINT8_C( 72), UINT8_C( 86), UINT8_C(212), UINT8_C(215), UINT8_C(252), UINT8_C( 69),
        UINT8_C(  7), UINT8_C(144), UINT8_C( 56), UINT8_C( 60), UINT8_C( 94), UINT8_C(204), UINT8_C( 94), UINT8_C( 33),
        UINT8_C(124), UINT8_C(131), UINT8_C(100), UINT8_C( 53), UINT8_C( 10), UINT8_C(101), UINT8_C(184), UINT8_C(240),
        UINT8_C(204), UINT8_C(176), UINT8_C(168), UINT8_C(221), UINT8_C( 97), UINT8_C(234), UINT8_C( 98), UINT8_C(158) },
      { UINT8_C( 61), UINT8_C(244), UINT8_C( 72), UINT8_C( 86), UINT8_C(212), UINT8_C(215), UINT8_C(252), UINT8_C(137),
        UINT8_C(128), UINT8_C( 48), UINT8_C( 56), UINT8_C(223), UINT8_C( 94), UINT8_C(204), UINT8_C( 94), UINT8_C( 33),
        UINT8_C(124), UINT8_C(100), UINT8_C(174), UINT8_C( 53), UINT8_C( 10), UINT8_C(101), UINT8_C(161), UINT8_C(150),
        UINT8_C(204), UINT8_C( 73), UINT8_C(168), UINT8_C(221), UINT8_C( 97), UINT8_C(234), UINT8_C( 98), UINT8_C(158) },
      UINT32_C(   4325632) },
    { UINT32_C(2089533179),
      { UINT8_C(238), UINT8_C( 81), UINT8_C( 91), UINT8_C(235), UINT8_C(117), UINT8_C( 91), UINT8_C(100), UINT8_C( 28),
        UINT8_C(192), UINT8_C( 19), UINT8_C(206), UINT8_C(137), UINT8_C(121), UINT8_C(111), UINT8_C( 31), UINT8_C(144),
        UINT8_C(185), UINT8_C(146), UINT8_C(  8), UINT8_C(237), UINT8_C(104), UINT8_C( 30), UINT8_C(  0), UINT8_C(232),
        UINT8_C( 41), UINT8_C(198), UINT8_C(234), UINT8_C( 37), UINT8_C(132), UINT8_C(117), UINT8_C(161), UINT8_C(114) },
      { UINT8_C(198), UINT8_C( 81), UINT8_C( 91), UINT8_C(235), UINT8_C( 87), UINT8_C(194), UINT8_C( 88), UINT8_C( 23),
        UINT8_C(192), UINT8_C( 38), UINT8_C(161), UINT8_C(137), UINT8_C(149), UINT8_C(111), UINT8_C(223), UINT8_C(144),
        UINT8_C(185), UINT8_C(146), UINT8_C( 59), UINT8_C(237), UINT8_C(104), UINT8_C( 30), UINT8_C(  0), UINT8_C(232),
        UINT8_C( 41), UINT8_C(198), UINT8_C(234), UINT8_C(133), UINT8_C(132), UINT8_C(117), UINT8_C(248), UINT8_C(114) },
      UINT32_C(1207965425) },
    { UINT32_C(3999292440),
      { UINT8_C(130), UINT8_C(  1), UINT8_C( 61), UINT8_C( 24), UINT8_C(193), UINT8_C( 28), UINT8_C(102), UINT8_C( 20),
        UINT8_C(  3), UINT8_C(162), UINT8_C(207), UINT8_C(  8), UINT8_C(221), UINT8_C(114), UINT8_C( 55), UINT8_C(223),
           UINT8_MAX, UINT8_C(139), UINT8_C(100), UINT8_C(  2), UINT8_C(128), UINT8_C( 92), UINT8_C(203), UINT8_C(113),
        UINT8_C(178), UINT8_C(207), UINT8_C(186), UINT8_C(203), UINT8_C( 44), UINT8_C( 26), UINT8_C(185), UINT8_C(174) },
      { UINT8_C( 27), UINT8_C(  1), UINT8_C( 61), UINT8_C(220), UINT8_C( 18), UINT8_C( 45), UINT8_C(241), UINT8_C( 20),
        UINT8_C(  3), UINT8_C(162), UINT8_C(207), UINT8_C(  8), UINT8_C( 51), UINT8_C( 85), UINT8_C(139), UINT8_C(223),
           UINT8_MAX, UINT8_C(240), UINT8_C( 52), UINT8_C( 97), UINT8_C( 76), UINT8_C( 92), UINT8_C(203),    UINT8_MAX,
        UINT8_C(207), UINT8_C(140), UINT8_C(186), UINT8_C(251), UINT8_C( 44), UINT8_C( 26), UINT8_C(185), UINT8_C(193) },
      UINT32_C(2315276312) },
    { UINT32_C(1822461853),
      { UINT8_C( 79), UINT8_C(189), UINT8_C( 24), UINT8_C(130), UINT8_C( 18), UINT8_C(164), UINT8_C(181), UINT8_C(243),
        UINT8_C(148), UINT8_C(233), UINT8_C( 84), UINT8_C(224), UINT8_C(233), UINT8_C( 38), UINT8_C(223), UINT8_C(184),
        UINT8_C(179), UINT8_C(169), UINT8_C(179), UINT8_C( 89), UINT8_C( 44), UINT8_C( 92), UINT8_C( 27), UINT8_C(165),
        UINT8_C(204), UINT8_C(185), UINT8_C( 48), UINT8_C(105), UINT8_C( 72), UINT8_C(208), UINT8_C(213), UINT8_C(151) },
      { UINT8_C( 79), UINT8_C(238), UINT8_C( 24), UINT8_C(130), UINT8_C( 18), UINT8_C(207), UINT8_C(147), UINT8_C(243),
        UINT8_C(148), UINT8_C(233), UINT8_C( 84), UINT8_C(161), UINT8_C(233), UINT8_C(230), UINT8_C( 89), UINT8_C(193),
        UINT8_C(143), UINT8_C(169), UINT8_C(179), UINT8_C( 89), UINT8_C(105), UINT8_C( 92), UINT8_C( 27), UINT8_C( 53),
        UINT8_C(204), UINT8_C(185), UINT8_C(159), UINT8_C(105), UINT8_C( 98), UINT8_C(116), UINT8_C(206), UINT8_C(240) },
      UINT32_C(1686145024) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__mmask32 k1 = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epu8_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epu8_mask");
    easysimd_assert_equal_mmask32(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi8(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u8x32());
    easysimd__mmask32 r = easysimd_mm256_mask_cmpneq_epu8_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t a[16];
    const int16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { {  INT16_C( 20215),  INT16_C(  3225), -INT16_C( 10842),  INT16_C( 18746),  INT16_C(  9081), -INT16_C( 20312),  INT16_C(  5157), -INT16_C(  5095),
         INT16_C(  6709), -INT16_C( 30698), -INT16_C( 20858),  INT16_C( 12807),  INT16_C(  4033), -INT16_C( 14799), -INT16_C( 21214),  INT16_C(  6576) },
      {  INT16_C( 18939),  INT16_C(  3225), -INT16_C( 10842), -INT16_C( 26390), -INT16_C( 27773), -INT16_C( 20312),  INT16_C( 24999), -INT16_C(  9068),
         INT16_C(  6709), -INT16_C( 30698),  INT16_C( 27481),  INT16_C(  6709),  INT16_C( 26234), -INT16_C( 14799), -INT16_C( 28653),  INT16_C(  3766) },
      UINT16_C(56537) },
    { { -INT16_C(  1872), -INT16_C( 26052), -INT16_C( 16496), -INT16_C( 10195), -INT16_C( 11161), -INT16_C(  1223), -INT16_C( 19024),  INT16_C(  5286),
        -INT16_C(    72), -INT16_C(  4736), -INT16_C(  1510), -INT16_C(  1453),  INT16_C( 26519),  INT16_C( 19851),  INT16_C( 25717),  INT16_C(  9513) },
      {  INT16_C( 25948), -INT16_C(  4928), -INT16_C( 16496), -INT16_C( 10195), -INT16_C(   318),  INT16_C( 29318),  INT16_C( 11699),  INT16_C(  5286),
        -INT16_C(    72),  INT16_C( 18008), -INT16_C( 21503), -INT16_C( 26559),  INT16_C( 26519), -INT16_C( 30491),  INT16_C(  3632),  INT16_C(  9513) },
      UINT16_C(28275) },
    { { -INT16_C( 26759),  INT16_C( 15963),  INT16_C(  7458), -INT16_C( 22212), -INT16_C(  4208),  INT16_C(  6102),  INT16_C(   603), -INT16_C( 19682),
         INT16_C(  8009), -INT16_C( 30113),  INT16_C( 29368), -INT16_C( 25258), -INT16_C( 30981), -INT16_C( 22100),  INT16_C(  7955), -INT16_C( 29417) },
      { -INT16_C( 26759),  INT16_C( 15963),  INT16_C(  1936), -INT16_C( 22212), -INT16_C(  4208),  INT16_C(  6102),  INT16_C(   603), -INT16_C( 19682),
         INT16_C(  8009),  INT16_C( 11310), -INT16_C( 31529), -INT16_C( 25258),  INT16_C( 30218),  INT16_C(  7803), -INT16_C( 28011), -INT16_C( 29417) },
      UINT16_C(30212) },
    { { -INT16_C( 27610), -INT16_C( 22403),  INT16_C( 29620), -INT16_C(  5375),  INT16_C( 23749), -INT16_C( 13760), -INT16_C( 19200),  INT16_C( 11822),
         INT16_C(  1505), -INT16_C( 21582), -INT16_C( 17193),  INT16_C( 21025), -INT16_C( 18470), -INT16_C( 31260), -INT16_C(  5885),  INT16_C( 10747) },
      {  INT16_C( 30845),  INT16_C( 13010), -INT16_C( 11284), -INT16_C( 20195),  INT16_C( 24111), -INT16_C( 13760), -INT16_C( 19200), -INT16_C(  2979),
         INT16_C(  4015), -INT16_C( 31072), -INT16_C( 15925),  INT16_C( 21025), -INT16_C( 17032),  INT16_C( 31787),  INT16_C( 10150),  INT16_C( 10747) },
      UINT16_C(30623) },
    { { -INT16_C( 29866),  INT16_C( 29514),  INT16_C( 31036), -INT16_C( 18479), -INT16_C(  7000),  INT16_C(  1377),  INT16_C(  4313),  INT16_C( 30996),
        -INT16_C(  8042),  INT16_C( 28474), -INT16_C( 19578), -INT16_C( 20179), -INT16_C( 11473), -INT16_C( 11048),  INT16_C( 30967),  INT16_C( 19788) },
      { -INT16_C( 29866),  INT16_C( 29514), -INT16_C( 28144), -INT16_C( 18185),  INT16_C( 22647),  INT16_C(  1377),  INT16_C(  4313), -INT16_C(    55),
         INT16_C(   946),  INT16_C( 28474), -INT16_C( 25674), -INT16_C( 20179), -INT16_C( 15761),  INT16_C( 26298),  INT16_C(  1594),  INT16_C( 19788) },
      UINT16_C(30108) },
    { { -INT16_C( 21378),  INT16_C( 29959),  INT16_C( 32357),  INT16_C(  9166),  INT16_C( 14030), -INT16_C( 26635), -INT16_C( 22475), -INT16_C( 23397),
         INT16_C( 20960), -INT16_C( 13761), -INT16_C( 20937), -INT16_C(  3699), -INT16_C( 14571), -INT16_C( 13833), -INT16_C( 27899), -INT16_C( 31938) },
      { -INT16_C( 21378), -INT16_C( 23047), -INT16_C( 14396),  INT16_C(  9166),  INT16_C( 14030),  INT16_C( 13098), -INT16_C( 22475), -INT16_C( 23397),
         INT16_C(  5654),  INT16_C( 19728), -INT16_C( 25147), -INT16_C(  3699), -INT16_C( 14571),  INT16_C( 27299), -INT16_C(  7735),  INT16_C(  2542) },
      UINT16_C(59174) },
    { { -INT16_C(  5458),  INT16_C( 30382), -INT16_C( 21635), -INT16_C( 22733), -INT16_C( 26146), -INT16_C( 19092), -INT16_C( 32033), -INT16_C(  4148),
        -INT16_C( 28208),  INT16_C(  3725), -INT16_C(  3477),  INT16_C(  3652),  INT16_C(  3420),  INT16_C( 19183),  INT16_C(  5398), -INT16_C( 15311) },
      { -INT16_C(  5458),  INT16_C( 30382),  INT16_C( 28043),  INT16_C( 26916), -INT16_C( 26146), -INT16_C( 19092), -INT16_C( 32033), -INT16_C(  4148),
        -INT16_C( 28208), -INT16_C(  6159),  INT16_C( 13652),  INT16_C(  3652), -INT16_C(  7102),  INT16_C( 19183),  INT16_C( 11513), -INT16_C( 15311) },
      UINT16_C(22028) },
    { { -INT16_C( 26762), -INT16_C( 25917), -INT16_C( 13824),  INT16_C(  7978),  INT16_C( 15791), -INT16_C( 31734), -INT16_C( 31201),  INT16_C(  4326),
         INT16_C( 14957),  INT16_C( 25157), -INT16_C( 30741), -INT16_C(  6586),  INT16_C( 16607), -INT16_C(  1262),  INT16_C(  7737), -INT16_C( 20399) },
      { -INT16_C( 26762), -INT16_C( 25917), -INT16_C( 13824),  INT16_C(  7978), -INT16_C(  8014), -INT16_C( 12013), -INT16_C(  1690), -INT16_C( 11038),
         INT16_C( 10036),  INT16_C( 25157), -INT16_C( 30741), -INT16_C(  6586),  INT16_C(  6077), -INT16_C(  2422), -INT16_C(  9418), -INT16_C(  5210) },
      UINT16_C(61936) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi16(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i16x16());
    easysimd__mmask16 r = easysimd_mm256_cmpneq_epi16_mask(a, b);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const int16_t a[16];
    const int16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(53153),
      {  INT16_C( 30566),  INT16_C(  6238),  INT16_C( 29015), -INT16_C( 16919), -INT16_C( 13462), -INT16_C( 24943), -INT16_C( 14093), -INT16_C( 23875),
        -INT16_C( 15803),  INT16_C(   560), -INT16_C( 17702),  INT16_C(  4344), -INT16_C( 24682), -INT16_C( 30981), -INT16_C( 25200), -INT16_C(  2474) },
      { -INT16_C( 19436),  INT16_C(  6238),  INT16_C( 29015), -INT16_C( 28888), -INT16_C( 13462), -INT16_C( 18898), -INT16_C(  5246), -INT16_C( 23875),
        -INT16_C( 15803),  INT16_C(   560), -INT16_C( 16061),  INT16_C(  4344), -INT16_C( 27808), -INT16_C(  4001), -INT16_C( 19152),  INT16_C( 17638) },
      UINT16_C(50209) },
    { UINT16_C(36527),
      { -INT16_C( 10004), -INT16_C( 20706),  INT16_C( 19602),  INT16_C(  5221), -INT16_C( 17097), -INT16_C(  6693), -INT16_C( 23483), -INT16_C( 30611),
         INT16_C(  1381), -INT16_C( 14751), -INT16_C( 15975), -INT16_C( 13898), -INT16_C( 25226), -INT16_C(  8178), -INT16_C( 17007),  INT16_C( 32110) },
      { -INT16_C( 10004), -INT16_C( 20706),  INT16_C( 19602),  INT16_C(  5221),  INT16_C(  5710), -INT16_C(  6693), -INT16_C( 23483),  INT16_C(  8220),
         INT16_C( 32360), -INT16_C( 14751), -INT16_C( 25537), -INT16_C( 13898), -INT16_C(  9927), -INT16_C( 13419),  INT16_C(  1174),  INT16_C( 32110) },
      UINT16_C( 1152) },
    { UINT16_C(26963),
      { -INT16_C( 28922),  INT16_C( 21881),  INT16_C( 28325),  INT16_C( 24809),  INT16_C(  1489),  INT16_C( 14976),  INT16_C( 26243), -INT16_C( 15813),
         INT16_C(  1538),  INT16_C( 15480),  INT16_C(  3551),  INT16_C( 30215),  INT16_C( 20241), -INT16_C( 23902), -INT16_C(  2620), -INT16_C( 13557) },
      { -INT16_C( 28922),  INT16_C( 21881),  INT16_C(  2546), -INT16_C( 15222),  INT16_C(  2574),  INT16_C( 14976),  INT16_C( 26243), -INT16_C( 15813),
         INT16_C(  1538),  INT16_C( 15480), -INT16_C( 18982),  INT16_C( 30215),  INT16_C( 14085), -INT16_C( 13939), -INT16_C(  2620), -INT16_C( 20076) },
      UINT16_C( 8208) },
    { UINT16_C( 4059),
      {  INT16_C( 26045), -INT16_C( 13101), -INT16_C( 11921), -INT16_C(  8354), -INT16_C( 19958),  INT16_C( 19026),  INT16_C(   127),  INT16_C( 22890),
        -INT16_C(    74), -INT16_C( 17596), -INT16_C( 11721),  INT16_C( 25732),  INT16_C(  6506), -INT16_C( 30955), -INT16_C(  3635), -INT16_C( 29802) },
      {  INT16_C( 26045), -INT16_C( 14761), -INT16_C( 19142), -INT16_C(  8354), -INT16_C( 19958),  INT16_C( 19026), -INT16_C(  1544),  INT16_C( 22890),
        -INT16_C(    74),  INT16_C( 12137), -INT16_C(  4778),  INT16_C( 25732), -INT16_C( 22266), -INT16_C( 11193), -INT16_C(  3635), -INT16_C(  4001) },
      UINT16_C( 1602) },
    { UINT16_C(32950),
      {  INT16_C( 23659), -INT16_C( 11579),  INT16_C( 21587),  INT16_C( 19385), -INT16_C(  1971),  INT16_C( 17913),  INT16_C( 25212), -INT16_C( 11659),
         INT16_C(  2128),  INT16_C( 22163), -INT16_C(  9551),  INT16_C( 19242), -INT16_C( 30280), -INT16_C(   452), -INT16_C(  3521), -INT16_C( 21889) },
      {  INT16_C( 23659), -INT16_C( 11579),  INT16_C( 13976),  INT16_C( 19385), -INT16_C(  1971), -INT16_C( 21718), -INT16_C( 24759), -INT16_C( 11659),
         INT16_C(  4264),  INT16_C( 23024), -INT16_C(  9551), -INT16_C( 23643), -INT16_C( 30280), -INT16_C(   452), -INT16_C(  3521), -INT16_C( 21889) },
      UINT16_C(   36) },
    { UINT16_C(64708),
      { -INT16_C( 20159),  INT16_C( 28641),  INT16_C(  3224), -INT16_C(  7654), -INT16_C( 26453),  INT16_C( 21371),  INT16_C( 27560), -INT16_C( 27731),
         INT16_C( 21126),  INT16_C( 10806), -INT16_C( 10189),  INT16_C(  1549), -INT16_C( 25608),  INT16_C( 23848), -INT16_C(  4954), -INT16_C(  6311) },
      { -INT16_C( 20159),  INT16_C( 13911),  INT16_C( 28999), -INT16_C(  3560), -INT16_C( 27639), -INT16_C( 19898),  INT16_C( 27560), -INT16_C( 27731),
         INT16_C( 31813),  INT16_C( 10806), -INT16_C( 17068),  INT16_C( 19582), -INT16_C( 22696), -INT16_C(    87), -INT16_C(  4954), -INT16_C(  6311) },
      UINT16_C(15364) },
    { UINT16_C(34152),
      { -INT16_C( 32593), -INT16_C( 18313), -INT16_C( 17132),  INT16_C(  5226), -INT16_C( 20304), -INT16_C(  2663),  INT16_C( 18732), -INT16_C( 32659),
        -INT16_C(  5114),  INT16_C( 24268),  INT16_C( 30355),  INT16_C(  9821),  INT16_C( 17529), -INT16_C( 18600), -INT16_C( 16255),  INT16_C( 12348) },
      { -INT16_C( 19648), -INT16_C( 18313), -INT16_C( 17132),  INT16_C(  5226),  INT16_C(   515), -INT16_C(  2663),  INT16_C( 18732), -INT16_C( 32659),
        -INT16_C(  5114),  INT16_C( 24268),  INT16_C(  3570),  INT16_C( 27434),  INT16_C( 17529), -INT16_C( 11486),  INT16_C( 24130), -INT16_C( 32253) },
      UINT16_C(33792) },
    { UINT16_C(33495),
      {  INT16_C( 16448),  INT16_C( 17316), -INT16_C( 17597), -INT16_C( 29069),  INT16_C(  8767), -INT16_C( 20256), -INT16_C( 28514), -INT16_C( 28493),
        -INT16_C(  8803), -INT16_C(  4101),  INT16_C(  7519), -INT16_C( 24126), -INT16_C( 14981), -INT16_C( 29404), -INT16_C(  1102), -INT16_C(  3569) },
      {  INT16_C( 16448),  INT16_C( 32565), -INT16_C( 22418), -INT16_C( 20979), -INT16_C(  4661), -INT16_C( 20256),  INT16_C(  4477),  INT16_C(  7162),
        -INT16_C(  2577),  INT16_C( 19978),  INT16_C(  7519), -INT16_C( 28944), -INT16_C( 14981),  INT16_C( 17179),  INT16_C( 11023), -INT16_C(  3569) },
      UINT16_C(  726) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 k1 = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epi16_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epi16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi16(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i16x16());
    easysimd__mmask16 r = easysimd_mm256_mask_cmpneq_epi16_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t a[16];
    const uint16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { UINT16_C(19914), UINT16_C(55315), UINT16_C(57083), UINT16_C(22981), UINT16_C(17224), UINT16_C(17002), UINT16_C(22878), UINT16_C(26679),
        UINT16_C(19112), UINT16_C(38964), UINT16_C(50649), UINT16_C(62636), UINT16_C(47881), UINT16_C(15903), UINT16_C(65031), UINT16_C(53673) },
      { UINT16_C(48459), UINT16_C(18089), UINT16_C(28571), UINT16_C(58271), UINT16_C( 2482), UINT16_C(17002), UINT16_C(22878), UINT16_C(26679),
        UINT16_C(19112), UINT16_C(38964), UINT16_C(20337), UINT16_C(31349), UINT16_C(47881), UINT16_C(15903), UINT16_C(65031), UINT16_C(56803) },
      UINT16_C(35871) },
    { { UINT16_C(47907), UINT16_C(49915), UINT16_C(44446), UINT16_C(50380), UINT16_C(12221), UINT16_C(13601), UINT16_C(51258), UINT16_C(56801),
        UINT16_C(21321), UINT16_C(48684), UINT16_C(14029), UINT16_C(34386), UINT16_C(58696), UINT16_C(11241), UINT16_C( 2242), UINT16_C(59063) },
      { UINT16_C(47907), UINT16_C(25256), UINT16_C(44446), UINT16_C( 7718), UINT16_C(18339), UINT16_C(56659), UINT16_C(13583), UINT16_C(22714),
        UINT16_C(59016), UINT16_C(21782), UINT16_C(26909), UINT16_C(26076), UINT16_C(50510), UINT16_C(11241), UINT16_C( 2242), UINT16_C(37366) },
      UINT16_C(40954) },
    { { UINT16_C(23539), UINT16_C( 6419), UINT16_C(46969), UINT16_C(52320), UINT16_C(28564), UINT16_C(20225), UINT16_C(35272), UINT16_C(56885),
        UINT16_C(21215), UINT16_C(47943), UINT16_C(38327), UINT16_C(18304), UINT16_C(19878), UINT16_C(40079), UINT16_C(35294), UINT16_C(53563) },
      { UINT16_C(23539), UINT16_C(24042), UINT16_C(46969), UINT16_C(39466), UINT16_C(28564), UINT16_C(20225), UINT16_C(35272), UINT16_C(56885),
        UINT16_C(43121), UINT16_C(10575), UINT16_C(53053), UINT16_C(18304), UINT16_C(65308), UINT16_C(64384), UINT16_C(35294), UINT16_C(28108) },
      UINT16_C(46858) },
    { { UINT16_C( 4299), UINT16_C(62721), UINT16_C(48043), UINT16_C(37920), UINT16_C(54589), UINT16_C(40627), UINT16_C( 9577), UINT16_C(47174),
        UINT16_C(33614), UINT16_C(48775), UINT16_C(42087), UINT16_C(59326), UINT16_C(18335), UINT16_C(27554), UINT16_C(44468), UINT16_C(32546) },
      { UINT16_C( 9405), UINT16_C(62721), UINT16_C(48043), UINT16_C( 7677), UINT16_C(54589), UINT16_C(40627), UINT16_C( 9577), UINT16_C(47174),
        UINT16_C( 5252), UINT16_C(60386), UINT16_C(41144), UINT16_C(22482), UINT16_C(18335), UINT16_C(27554), UINT16_C(44468), UINT16_C(57115) },
      UINT16_C(36617) },
    { { UINT16_C(59464), UINT16_C(17700), UINT16_C(36613), UINT16_C(49397), UINT16_C(52067), UINT16_C(61377), UINT16_C(18158), UINT16_C(53251),
        UINT16_C(47921), UINT16_C( 1136), UINT16_C(22290), UINT16_C(54649), UINT16_C(39923), UINT16_C( 3770), UINT16_C(50042), UINT16_C(49821) },
      { UINT16_C(49835), UINT16_C(45319), UINT16_C(64849), UINT16_C(46193), UINT16_C(52067), UINT16_C(46755), UINT16_C(18158), UINT16_C(43655),
        UINT16_C(63330), UINT16_C(30126), UINT16_C(10063), UINT16_C(54649), UINT16_C(39923), UINT16_C( 3770), UINT16_C(50042), UINT16_C(49821) },
      UINT16_C( 1967) },
    { { UINT16_C(   35), UINT16_C(38148), UINT16_C(52404), UINT16_C(22728), UINT16_C(16770), UINT16_C( 2559), UINT16_C(25067), UINT16_C(39425),
        UINT16_C(20694), UINT16_C( 8385), UINT16_C(33938), UINT16_C(57892), UINT16_C(60353), UINT16_C(49359), UINT16_C(32606), UINT16_C(33223) },
      { UINT16_C(52095), UINT16_C(13334), UINT16_C(56983), UINT16_C( 6796), UINT16_C(35615), UINT16_C( 2851), UINT16_C( 9452), UINT16_C(50085),
        UINT16_C(26228), UINT16_C( 8385), UINT16_C(33938), UINT16_C(44008), UINT16_C(60353), UINT16_C(20844), UINT16_C(13111), UINT16_C(46803) },
      UINT16_C(59903) },
    { { UINT16_C(38634), UINT16_C(30408), UINT16_C(59312), UINT16_C(54273), UINT16_C(61170), UINT16_C(38904), UINT16_C(28081), UINT16_C(38142),
        UINT16_C(59507), UINT16_C(23708), UINT16_C(37012), UINT16_C(   20), UINT16_C(19425), UINT16_C(46131), UINT16_C(12801), UINT16_C(60574) },
      { UINT16_C(38634), UINT16_C(31074), UINT16_C(25677), UINT16_C(16461), UINT16_C(18002), UINT16_C(38904), UINT16_C(54707), UINT16_C(38142),
        UINT16_C(13502), UINT16_C(23708), UINT16_C(37012), UINT16_C(   20), UINT16_C(34273), UINT16_C(58202), UINT16_C(12801), UINT16_C(60574) },
      UINT16_C(12638) },
    { { UINT16_C(44026), UINT16_C(18581), UINT16_C(59371), UINT16_C(50062), UINT16_C(16874), UINT16_C(33432), UINT16_C(22119), UINT16_C(60086),
        UINT16_C(31400), UINT16_C(64128), UINT16_C(25119), UINT16_C(31104), UINT16_C(14405), UINT16_C( 5233), UINT16_C(53177), UINT16_C(45893) },
      { UINT16_C(44026), UINT16_C(18581), UINT16_C(59371), UINT16_C(44329), UINT16_C(49866), UINT16_C(12847), UINT16_C(58648), UINT16_C(60086),
        UINT16_C(40031), UINT16_C(32443), UINT16_C(25119), UINT16_C(17400), UINT16_C(26995), UINT16_C( 5233), UINT16_C(40249), UINT16_C(46304) },
      UINT16_C(56184) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epu16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epu16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi16(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u16x16());
    easysimd__mmask16 r = easysimd_mm256_cmpneq_epu16_mask(a, b);

    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const uint16_t a[16];
    const uint16_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(14874),
      { UINT16_C(17509), UINT16_C(12263), UINT16_C( 5638), UINT16_C( 7777), UINT16_C(32251), UINT16_C(23263), UINT16_C(39706), UINT16_C( 6361),
        UINT16_C(53718), UINT16_C(19036), UINT16_C(45882), UINT16_C(29558), UINT16_C(22096), UINT16_C(51239), UINT16_C(16946), UINT16_C(38659) },
      { UINT16_C(17509), UINT16_C(36038), UINT16_C(10241), UINT16_C(64682), UINT16_C(32251), UINT16_C(48983), UINT16_C(12325), UINT16_C(64472),
        UINT16_C(53718), UINT16_C(15173), UINT16_C(45882), UINT16_C(29558), UINT16_C(22096), UINT16_C(17408), UINT16_C(16946), UINT16_C(40667) },
      UINT16_C( 8714) },
    { UINT16_C(61226),
      { UINT16_C(54730), UINT16_C(28651), UINT16_C(16991), UINT16_C(33839), UINT16_C( 1906), UINT16_C(29567), UINT16_C(50491), UINT16_C( 8879),
        UINT16_C(24193), UINT16_C(37722), UINT16_C(23348), UINT16_C(19928), UINT16_C(45918), UINT16_C(19691), UINT16_C( 5717), UINT16_C( 7995) },
      { UINT16_C(10219), UINT16_C(28651), UINT16_C(48745), UINT16_C(33839), UINT16_C( 1906), UINT16_C(   79), UINT16_C(65042), UINT16_C(37666),
        UINT16_C(24193), UINT16_C(37159), UINT16_C(65496), UINT16_C(19928), UINT16_C(51634), UINT16_C( 2179), UINT16_C( 5717), UINT16_C(51751) },
      UINT16_C(42528) },
    { UINT16_C(20244),
      { UINT16_C(57972), UINT16_C(14635), UINT16_C(31280), UINT16_C(16953), UINT16_C(23673), UINT16_C(54742), UINT16_C(64985), UINT16_C(45414),
        UINT16_C(17660), UINT16_C(44775), UINT16_C(27150), UINT16_C(60854), UINT16_C(56873), UINT16_C( 3768), UINT16_C(52372), UINT16_C( 2397) },
      { UINT16_C(34991), UINT16_C(14635), UINT16_C(31280), UINT16_C(16953), UINT16_C(63448), UINT16_C(54742), UINT16_C(47348), UINT16_C(61538),
        UINT16_C(18940), UINT16_C( 2719), UINT16_C(27150), UINT16_C(60854), UINT16_C(45107), UINT16_C( 3768), UINT16_C(52372), UINT16_C( 2397) },
      UINT16_C(  784) },
    { UINT16_C(54282),
      { UINT16_C(11407), UINT16_C(26448), UINT16_C(41507), UINT16_C( 6168), UINT16_C(31322), UINT16_C(22024), UINT16_C(42948), UINT16_C(30817),
        UINT16_C(23037), UINT16_C(12373), UINT16_C(16393), UINT16_C(34296), UINT16_C(51593), UINT16_C(23473), UINT16_C(48093), UINT16_C(27695) },
      { UINT16_C(11407), UINT16_C(26448), UINT16_C(60450), UINT16_C(31779), UINT16_C(31322), UINT16_C(22024), UINT16_C(13267), UINT16_C(53411),
        UINT16_C(23037), UINT16_C(12373), UINT16_C(16393), UINT16_C(34296), UINT16_C(51593), UINT16_C(40733), UINT16_C(19591), UINT16_C(28428) },
      UINT16_C(49160) },
    { UINT16_C(61050),
      { UINT16_C(40396), UINT16_C(13162), UINT16_C(15816), UINT16_C(39774), UINT16_C(  368), UINT16_C(64875), UINT16_C(27897), UINT16_C(12946),
        UINT16_C(44389), UINT16_C(10228), UINT16_C( 4473), UINT16_C(  455), UINT16_C(54109), UINT16_C(10864), UINT16_C(60083), UINT16_C(32536) },
      { UINT16_C(40396), UINT16_C(20403), UINT16_C( 4544), UINT16_C(39774), UINT16_C(  368), UINT16_C( 3117), UINT16_C(49346), UINT16_C(12946),
        UINT16_C(44389), UINT16_C(59215), UINT16_C( 5699), UINT16_C(41448), UINT16_C(54109), UINT16_C(10864), UINT16_C(58178), UINT16_C(51483) },
      UINT16_C(52834) },
    { UINT16_C( 9752),
      { UINT16_C(  992), UINT16_C(62295), UINT16_C(33882), UINT16_C( 7423), UINT16_C(15940), UINT16_C(45636), UINT16_C(37744), UINT16_C(46233),
        UINT16_C(33193), UINT16_C(37461), UINT16_C( 8409), UINT16_C( 6958), UINT16_C(18691), UINT16_C(27364), UINT16_C(64536), UINT16_C(63632) },
      { UINT16_C(59136), UINT16_C(23275), UINT16_C(60012), UINT16_C( 7423), UINT16_C(15940), UINT16_C(39266), UINT16_C(37744), UINT16_C(63053),
        UINT16_C(41596), UINT16_C(21896), UINT16_C( 8409), UINT16_C(50544), UINT16_C(21504), UINT16_C( 6191), UINT16_C(49233), UINT16_C(20752) },
      UINT16_C( 8704) },
    { UINT16_C( 5035),
      { UINT16_C( 8677), UINT16_C( 3780), UINT16_C( 9948), UINT16_C(10663), UINT16_C(62498), UINT16_C(40480), UINT16_C(43158), UINT16_C(22772),
        UINT16_C(25695), UINT16_C(24349), UINT16_C(19897), UINT16_C( 2679), UINT16_C(34573), UINT16_C(46171), UINT16_C( 1666), UINT16_C(26568) },
      { UINT16_C( 8677), UINT16_C(  885), UINT16_C( 7346), UINT16_C(54317), UINT16_C(19728), UINT16_C(42611), UINT16_C(26613), UINT16_C(22772),
        UINT16_C(25695), UINT16_C(24349), UINT16_C(19897), UINT16_C( 2679), UINT16_C(59825), UINT16_C(46171), UINT16_C( 1666), UINT16_C(26568) },
      UINT16_C( 4138) },
    { UINT16_C(12570),
      { UINT16_C(18221), UINT16_C(15621), UINT16_C(30868), UINT16_C(35556), UINT16_C(58079), UINT16_C(43998), UINT16_C(37630), UINT16_C(26415),
        UINT16_C(48828), UINT16_C(28381), UINT16_C( 2215), UINT16_C(38817), UINT16_C(15610), UINT16_C(31150), UINT16_C(51277), UINT16_C(31402) },
      { UINT16_C(18221), UINT16_C(42167), UINT16_C(39720), UINT16_C(35556), UINT16_C( 3454), UINT16_C(43998), UINT16_C(37630), UINT16_C(23524),
        UINT16_C(48828), UINT16_C(18377), UINT16_C(27593), UINT16_C(38817), UINT16_C(15610), UINT16_C(62525), UINT16_C(51277), UINT16_C(31402) },
      UINT16_C( 8210) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 k1 = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epu16_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epu16_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi16(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u16x16());
    easysimd__mmask16 r = easysimd_mm256_mask_cmpneq_epu16_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[8];
    const int32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT32_C(   200139574),  INT32_C(   790375587),  INT32_C(   559121868), -INT32_C(  1645190390), -INT32_C(   222328761), -INT32_C(   774091716), -INT32_C(  1259523395), -INT32_C(   978259205) },
      { -INT32_C(   200139574), -INT32_C(  1142892817),  INT32_C(   559121868), -INT32_C(  1645190390), -INT32_C(  1848524459), -INT32_C(   899502835), -INT32_C(   629190689),  INT32_C(  1335832385) },
      UINT8_C(242) },
    { {  INT32_C(   803275097),  INT32_C(   361405665),  INT32_C(  1185804409), -INT32_C(  1264874510),  INT32_C(   163654748), -INT32_C(   555185010),  INT32_C(   975159819), -INT32_C(  1154650270) },
      {  INT32_C(   803275097),  INT32_C(   610760107),  INT32_C(  1185804409), -INT32_C(    66320993), -INT32_C(  1090138831), -INT32_C(   555185010),  INT32_C(   312458672), -INT32_C(  1664168916) },
      UINT8_C(218) },
    { {  INT32_C(   780529081),  INT32_C(  1754573140),  INT32_C(   403237907), -INT32_C(   985070344), -INT32_C(   148961015),  INT32_C(  1671988134), -INT32_C(   292570590),  INT32_C(  1103702920) },
      {  INT32_C(   780529081),  INT32_C(  1754573140),  INT32_C(   403237907), -INT32_C(   340955678),  INT32_C(   501468278),  INT32_C(  1921026896), -INT32_C(   832499643),  INT32_C(  1103702920) },
      UINT8_C(120) },
    { { -INT32_C(  2089762177),  INT32_C(  1987280024),  INT32_C(   223894432), -INT32_C(   712752375), -INT32_C(  1322933978),  INT32_C(   838309921),  INT32_C(   594396665),  INT32_C(  1419493844) },
      { -INT32_C(  2089762177),  INT32_C(   680610696), -INT32_C(  1355421274), -INT32_C(   712752375), -INT32_C(  1322933978),  INT32_C(   838309921),  INT32_C(   594396665),  INT32_C(  1419493844) },
      UINT8_C(  6) },
    { {  INT32_C(   210664385), -INT32_C(   156060148), -INT32_C(  1524407573), -INT32_C(  1845528857),  INT32_C(  2010413947),  INT32_C(  1127747369),  INT32_C(  1746914926),  INT32_C(  2020507575) },
      {  INT32_C(   109378810), -INT32_C(   156060148), -INT32_C(  2143150183), -INT32_C(  1845528857), -INT32_C(   427170371),  INT32_C(  1663746549),  INT32_C(  1746914926),  INT32_C(  2020507575) },
      UINT8_C( 53) },
    { {  INT32_C(  2078787908), -INT32_C(   451640123),  INT32_C(   159093959), -INT32_C(  1949922139), -INT32_C(   679432939),  INT32_C(   543613911), -INT32_C(   404116050), -INT32_C(  1323521171) },
      {  INT32_C(   975962996),  INT32_C(  1310670983),  INT32_C(   159093959),  INT32_C(  1006902822), -INT32_C(  1575779125),  INT32_C(   543613911), -INT32_C(   404116050), -INT32_C(  1323521171) },
      UINT8_C( 27) },
    { { -INT32_C(  1314716303), -INT32_C(  1098518236),  INT32_C(  1743126089), -INT32_C(  2026757885), -INT32_C(  1376791500),  INT32_C(  1121191062),  INT32_C(   429484033),  INT32_C(   506815661) },
      { -INT32_C(  1314716303), -INT32_C(  1098518236),  INT32_C(  1743126089), -INT32_C(  2015515821),  INT32_C(   372558975), -INT32_C(   765982255),  INT32_C(   535556465), -INT32_C(  1086513067) },
      UINT8_C(248) },
    { {  INT32_C(  1640123917), -INT32_C(   843787109), -INT32_C(   115340722), -INT32_C(  1267095576), -INT32_C(   461008933), -INT32_C(   665430041),  INT32_C(  1663923523), -INT32_C(  1084428878) },
      {  INT32_C(  1640123917),  INT32_C(  1084478962), -INT32_C(   115340722),  INT32_C(  1165800298),  INT32_C(   690618946), -INT32_C(  1677623207), -INT32_C(  1493160203), -INT32_C(  1084428878) },
      UINT8_C(122) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epi32_mask(a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i32x8());
    easysimd__mmask8 r = easysimd_mm256_cmpneq_epi32_mask(a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int32_t a[8];
    const int32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 46),
      {  INT32_C(   514505752), -INT32_C(   955756114), -INT32_C(   809161708), -INT32_C(   245318317),  INT32_C(   654847975),  INT32_C(   624017433),  INT32_C(  1213431609), -INT32_C(   596180540) },
      {  INT32_C(  1828397501), -INT32_C(  2026699917), -INT32_C(  2108229329), -INT32_C(   245318317), -INT32_C(    20677403),  INT32_C(   624017433),  INT32_C(   952006516), -INT32_C(   596180540) },
      UINT8_C(  6) },
    { UINT8_C( 16),
      { -INT32_C(   720123230), -INT32_C(  1479654831),  INT32_C(   962430149), -INT32_C(   441105631),  INT32_C(  1708847681),  INT32_C(   651972456), -INT32_C(  1671670649),  INT32_C(   749515658) },
      { -INT32_C(   838680707), -INT32_C(  1479654831),  INT32_C(  1979765333), -INT32_C(   441105631),  INT32_C(   263992999),  INT32_C(   651972456), -INT32_C(  1671670649),  INT32_C(   749515658) },
      UINT8_C( 16) },
    { UINT8_C(165),
      {  INT32_C(    91625360), -INT32_C(   556282914), -INT32_C(  1651184319),  INT32_C(    13908804), -INT32_C(  2120722101),  INT32_C(   269714534), -INT32_C(  1620682501), -INT32_C(  1438352614) },
      {  INT32_C(  1907341971), -INT32_C(   556282914), -INT32_C(  1218124685),  INT32_C(  1790392351), -INT32_C(  2120722101), -INT32_C(   821494060),  INT32_C(  1886285398), -INT32_C(  1438352614) },
      UINT8_C( 37) },
    { UINT8_C(201),
      {  INT32_C(  1045492718),  INT32_C(   505594810),  INT32_C(   844579451), -INT32_C(  1736250964), -INT32_C(   359246111), -INT32_C(  1806111451),  INT32_C(  2017935965), -INT32_C(  1371425601) },
      {  INT32_C(  1642893735),  INT32_C(   505594810),  INT32_C(   844579451), -INT32_C(  1736250964), -INT32_C(   359246111), -INT32_C(  1806111451), -INT32_C(   420553946),  INT32_C(   932458639) },
      UINT8_C(193) },
    { UINT8_C(128),
      {  INT32_C(   412030616), -INT32_C(   370155290),  INT32_C(  1936568286), -INT32_C(   593541039), -INT32_C(   591349688),  INT32_C(   498591535),  INT32_C(  1632454349),  INT32_C(   383848317) },
      {  INT32_C(   204370213),  INT32_C(   804593233),  INT32_C(  1936568286), -INT32_C(   593541039), -INT32_C(   591349688),  INT32_C(  1866958242),  INT32_C(  1632454349),  INT32_C(   383848317) },
      UINT8_C(  0) },
    { UINT8_C(142),
      { -INT32_C(   894667563),  INT32_C(  1194162596), -INT32_C(  1819682920), -INT32_C(  1739684662), -INT32_C(   165744210),  INT32_C(   831253088), -INT32_C(  1813827789), -INT32_C(   836696328) },
      { -INT32_C(   894667563),  INT32_C(  1194162596),  INT32_C(  1735520157),  INT32_C(   587186291),  INT32_C(   286793137),  INT32_C(   994222855), -INT32_C(  1813827789), -INT32_C(  1499926741) },
      UINT8_C(140) },
    { UINT8_C( 49),
      {  INT32_C(   720765636),  INT32_C(  1377407969), -INT32_C(    99447558), -INT32_C(   971521106),  INT32_C(   392765397),  INT32_C(   675101530), -INT32_C(  1642633210), -INT32_C(   741354738) },
      {  INT32_C(   720765636),  INT32_C(  1409881689),  INT32_C(  1364073378), -INT32_C(   971521106), -INT32_C(   557088380), -INT32_C(  1039791940),  INT32_C(  2137070961), -INT32_C(   967692048) },
      UINT8_C( 48) },
    { UINT8_C( 79),
      { -INT32_C(  2056957828), -INT32_C(   224327516),  INT32_C(  1884782681), -INT32_C(      926668), -INT32_C(  1056395590), -INT32_C(   773949072),  INT32_C(  1258475512),  INT32_C(   295368853) },
      { -INT32_C(  2056957828),  INT32_C(  1641953288),  INT32_C(  1884782681), -INT32_C(   856964334), -INT32_C(  1056395590), -INT32_C(   773949072), -INT32_C(   793725637),  INT32_C(   295368853) },
      UINT8_C( 74) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epi32_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epi32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i32x8());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpneq_epi32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t a[8];
    const uint32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { UINT32_C(2958221944), UINT32_C(4150834113), UINT32_C(1208571013), UINT32_C( 867816768), UINT32_C(3483408995), UINT32_C(  34211149), UINT32_C(4289780405), UINT32_C( 877307579) },
      { UINT32_C(1239719048), UINT32_C(4150834113), UINT32_C(3776792993), UINT32_C( 867816768), UINT32_C(3193025904), UINT32_C(  34211149), UINT32_C(4033769525), UINT32_C( 877307579) },
      UINT8_C( 85) },
    { { UINT32_C(1453713417), UINT32_C(1928821289), UINT32_C(1838274967), UINT32_C(2749174766), UINT32_C(3210582884), UINT32_C(3438595164), UINT32_C(4091798842), UINT32_C( 306741769) },
      { UINT32_C(2473127274), UINT32_C(1928821289), UINT32_C( 661362489), UINT32_C(2749174766), UINT32_C(1087317988), UINT32_C(3438595164), UINT32_C(2988502952), UINT32_C( 306741769) },
      UINT8_C( 85) },
    { { UINT32_C(2377936685), UINT32_C( 231127669), UINT32_C(1857613093), UINT32_C(3729967031), UINT32_C(1523683990), UINT32_C(2382560926), UINT32_C(2281681951), UINT32_C(2799557497) },
      { UINT32_C(3224607051), UINT32_C(2043541844), UINT32_C(2649196006), UINT32_C(3729967031), UINT32_C(1523683990), UINT32_C(1777877066), UINT32_C(1844574196), UINT32_C(2799557497) },
      UINT8_C(103) },
    { { UINT32_C(1069276742), UINT32_C( 824522156), UINT32_C(1451147804), UINT32_C(2300666429), UINT32_C(3755248543), UINT32_C(2077506947), UINT32_C(4232986926), UINT32_C(2607049813) },
      { UINT32_C(1069276742), UINT32_C(1898446932), UINT32_C(1451147804), UINT32_C(2827675656), UINT32_C(3755248543), UINT32_C(2077506947), UINT32_C(4073111709), UINT32_C(2607049813) },
      UINT8_C( 74) },
    { { UINT32_C(1771990376), UINT32_C(1009520533), UINT32_C(3208916182), UINT32_C( 338947254), UINT32_C(3484428916), UINT32_C(1567390365), UINT32_C(1804230208), UINT32_C(1421148652) },
      { UINT32_C(  79516526), UINT32_C( 960555363), UINT32_C(3438838806), UINT32_C(3856739185), UINT32_C(4088697174), UINT32_C(1263542539), UINT32_C(1804230208), UINT32_C(1254124507) },
      UINT8_C(191) },
    { { UINT32_C(1730301565), UINT32_C( 310205326), UINT32_C(2139310420), UINT32_C(3151325226), UINT32_C(1053214749), UINT32_C(4089254425), UINT32_C( 885991880), UINT32_C(1727207913) },
      { UINT32_C(4123858279), UINT32_C(3288812144), UINT32_C(3192163220), UINT32_C(3151325226), UINT32_C(4216209634), UINT32_C(4089254425), UINT32_C( 508542261), UINT32_C(1727207913) },
      UINT8_C( 87) },
    { { UINT32_C(2630300242), UINT32_C(3308293178), UINT32_C(3921211344), UINT32_C(2848704873), UINT32_C( 603768343), UINT32_C(1918375861), UINT32_C(2823386726), UINT32_C(1308591867) },
      { UINT32_C(2630300242), UINT32_C( 582490706), UINT32_C(3921211344), UINT32_C(2848704873), UINT32_C( 603768343), UINT32_C(1918375861), UINT32_C(2823386726), UINT32_C(1308591867) },
      UINT8_C(  2) },
    { { UINT32_C(2404663669), UINT32_C(1771599865), UINT32_C(1520634499), UINT32_C(1039725605), UINT32_C( 896224104), UINT32_C( 528023569), UINT32_C(2025109308), UINT32_C(3095003715) },
      { UINT32_C(2404663669), UINT32_C(1771599865), UINT32_C( 237193705), UINT32_C(1039725605), UINT32_C(2027665255), UINT32_C(4086781111), UINT32_C(2025109308), UINT32_C(2449925454) },
      UINT8_C(180) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epu32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u32x8());
    easysimd__mmask8 r = easysimd_mm256_cmpneq_epu32_mask(a, b);

    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint32_t a[8];
    const uint32_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 77),
      { UINT32_C(1949104590), UINT32_C(3854374338), UINT32_C(1862335012), UINT32_C(2066114464), UINT32_C(2010111455), UINT32_C(1019468497), UINT32_C( 790761769), UINT32_C(1920849571) },
      { UINT32_C(1949104590), UINT32_C(3816203455), UINT32_C(1862335012), UINT32_C(2066114464), UINT32_C(2010111455), UINT32_C(2355427427), UINT32_C(2914747913), UINT32_C( 723466332) },
      UINT8_C( 64) },
    { UINT8_C(  5),
      { UINT32_C( 866755005), UINT32_C(3601483908), UINT32_C(1078954374), UINT32_C(2093524551), UINT32_C( 829696973), UINT32_C(1941405111), UINT32_C(1235948330), UINT32_C(4233071935) },
      { UINT32_C(2989488174), UINT32_C(3347700288), UINT32_C(1078954374), UINT32_C(2093524551), UINT32_C(2994576123), UINT32_C(1941405111), UINT32_C(2274087240), UINT32_C(2374228574) },
      UINT8_C(  1) },
    { UINT8_C(179),
      { UINT32_C(3364688703), UINT32_C(3785486554), UINT32_C(3232833434), UINT32_C(3915033707), UINT32_C(2216587614), UINT32_C(1062561459), UINT32_C(1897509870), UINT32_C(2149903424) },
      { UINT32_C(3662196992), UINT32_C(1002236321), UINT32_C(1962634505), UINT32_C(3915033707), UINT32_C(2063301575), UINT32_C(1062561459), UINT32_C(1897509870), UINT32_C(2149903424) },
      UINT8_C( 19) },
    { UINT8_C( 13),
      { UINT32_C(1459009946), UINT32_C(4039376884), UINT32_C(3511016564), UINT32_C(1330449491), UINT32_C(1084174215), UINT32_C(1075028991), UINT32_C(2896614376), UINT32_C( 750370450) },
      { UINT32_C(1803726967), UINT32_C(4039376884), UINT32_C(1979031841), UINT32_C(1330449491), UINT32_C(1084174215), UINT32_C(1075028991), UINT32_C(2896614376), UINT32_C( 750370450) },
      UINT8_C(  5) },
    { UINT8_C(157),
      { UINT32_C(1407366391), UINT32_C(3455321304), UINT32_C(1024434553), UINT32_C(1268809942), UINT32_C(2698225648), UINT32_C( 855060374), UINT32_C( 931597341), UINT32_C(2429848728) },
      { UINT32_C(1407366391), UINT32_C(3455321304), UINT32_C(1316091000), UINT32_C(1687753076), UINT32_C(1325755833), UINT32_C(3112303772), UINT32_C(2716862473), UINT32_C(2429848728) },
      UINT8_C( 28) },
    { UINT8_C( 20),
      { UINT32_C(3555407853), UINT32_C(3704054891), UINT32_C(1290615986), UINT32_C(2780407456), UINT32_C(2007127542), UINT32_C(4269845262), UINT32_C(2126713932), UINT32_C(3616686057) },
      { UINT32_C(3786047094), UINT32_C(2495443426), UINT32_C(1290615986), UINT32_C(1207146833), UINT32_C(4240414190), UINT32_C(4269845262), UINT32_C(2126713932), UINT32_C(3616686057) },
      UINT8_C( 16) },
    { UINT8_C(190),
      { UINT32_C( 321912150), UINT32_C(1925111186), UINT32_C( 504107051), UINT32_C( 380959319), UINT32_C(4065719543), UINT32_C(2360387969), UINT32_C(4197101286), UINT32_C(2042119459) },
      { UINT32_C(3582781251), UINT32_C(4014427076), UINT32_C(2987283291), UINT32_C( 380959319), UINT32_C(4065719543), UINT32_C(2360387969), UINT32_C(3114790550), UINT32_C(2042119459) },
      UINT8_C(  6) },
    { UINT8_C(190),
      { UINT32_C(2466843468), UINT32_C( 166159611), UINT32_C(3771413783), UINT32_C(2986246522), UINT32_C(2123941561), UINT32_C( 460074611), UINT32_C( 427466983), UINT32_C(1675149591) },
      { UINT32_C(3371688397), UINT32_C(1574035014), UINT32_C(2369625107), UINT32_C(3611179805), UINT32_C(1968559617), UINT32_C(3968909573), UINT32_C( 427466983), UINT32_C(2581454540) },
      UINT8_C(190) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epu32_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epu32_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi32(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u32x8());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpneq_epu32_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { {  INT64_C( 6919654948592371252), -INT64_C( 7965550595212700457), -INT64_C( 6928874164762241056),  INT64_C( 9128223919614953140) },
      {  INT64_C( 6919654948592371252), -INT64_C( 6798605020699028443), -INT64_C( 6928874164762241056), -INT64_C(  922588431330984342) },
      UINT8_C( 10) },
    { { -INT64_C( 1625054179981216708), -INT64_C( 3690616133344655439), -INT64_C( 6654596329976953000), -INT64_C( 4023600627642180928) },
      { -INT64_C( 1625054179981216708),  INT64_C( 1011599854480419087), -INT64_C( 6654596329976953000),  INT64_C( 2422722340083455306) },
      UINT8_C( 10) },
    { {  INT64_C( 2255195888284465076), -INT64_C( 5181985575152547531),  INT64_C( 4640678396798234157), -INT64_C( 7986144531930071788) },
      { -INT64_C( 8192301326302099334),  INT64_C( 3814201469690651811),  INT64_C( 4640678396798234157), -INT64_C( 7986144531930071788) },
      UINT8_C(  3) },
    { {  INT64_C( 6912368174706780977), -INT64_C(  301289202230287526),  INT64_C(  692397981288558317),  INT64_C(  176210833079758388) },
      {  INT64_C( 6912368174706780977),  INT64_C( 8965404135288664926),  INT64_C(  692397981288558317),  INT64_C(  720447400532512811) },
      UINT8_C( 10) },
    { { -INT64_C( 5267830841453365684),  INT64_C( 1192822502880269465),  INT64_C( 2290835837807839714),  INT64_C( 1327506111182849991) },
      {  INT64_C( 4678533745053581429), -INT64_C( 2981125721010982191),  INT64_C( 2290835837807839714),  INT64_C( 5323003885373003015) },
      UINT8_C( 11) },
    { { -INT64_C( 6789807081386770057), -INT64_C( 2649969305104132834),  INT64_C( 7937537001676306912),  INT64_C( 5363941996568507514) },
      { -INT64_C( 6789807081386770057), -INT64_C(  276388514321813158), -INT64_C( 2318435280415319413),  INT64_C( 5363941996568507514) },
      UINT8_C(  6) },
    { { -INT64_C(  364127148702028653),  INT64_C(  491757040571924789),  INT64_C( 2353213313525692206), -INT64_C( 7315408168043695425) },
      { -INT64_C( 7502394026319805025),  INT64_C(  491757040571924789),  INT64_C( 2353213313525692206), -INT64_C( 7315408168043695425) },
      UINT8_C(  1) },
    { { -INT64_C( 6024291605214408230),  INT64_C( 5972883681268217672), -INT64_C( 4469769944038254947), -INT64_C( 8574866139257305823) },
      { -INT64_C( 6024291605214408230),  INT64_C( 5972883681268217672), -INT64_C( 5340048063922805448), -INT64_C( 8574866139257305823) },
      UINT8_C(  4) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i64x4());
    easysimd__mmask8 r = easysimd_mm256_cmpneq_epi64_mask(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int64_t a[4];
    const int64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C( 24),
      {  INT64_C( 5991238671981105186), -INT64_C( 4592567285600422567), -INT64_C( 3395024115577923569), -INT64_C( 7915597032342763908) },
      { -INT64_C( 6008364499405755530), -INT64_C( 4592567285600422567),  INT64_C( 2162611421324586661), -INT64_C( 5481916227252381974) },
      UINT8_C(  8) },
    { UINT8_C(199),
      {  INT64_C(  499543852077011198), -INT64_C( 4874731551745430253),  INT64_C( 4955589172369881772), -INT64_C( 5487894853027691996) },
      {  INT64_C( 7883055656130813668),  INT64_C( 6891617609777166955),  INT64_C( 8163007751854654443), -INT64_C( 5487894853027691996) },
      UINT8_C(  7) },
    { UINT8_C(141),
      {  INT64_C( 6330149848381805113), -INT64_C(  413710028633502692), -INT64_C( 9124112522984648904),  INT64_C( 7348416830271804463) },
      {  INT64_C( 1602119261592339492), -INT64_C(  413710028633502692), -INT64_C( 1069656852042604890), -INT64_C(  525134637637434844) },
      UINT8_C( 13) },
    { UINT8_C(187),
      {  INT64_C(  187466926188468187), -INT64_C( 5635481694936411813), -INT64_C( 2553921975730729883),  INT64_C( 3919911075045532405) },
      {  INT64_C( 6888508537438994537), -INT64_C( 3086419915565190856),  INT64_C( 5025155514374400865),  INT64_C( 3919911075045532405) },
      UINT8_C(  3) },
    { UINT8_C( 47),
      { -INT64_C( 4140128863712289726),  INT64_C( 3575011278073887687),  INT64_C( 2488648025453489372), -INT64_C( 5029446045441274847) },
      {  INT64_C( 6822706697808922925),  INT64_C( 2602213383114541850),  INT64_C( 2488648025453489372), -INT64_C( 5029446045441274847) },
      UINT8_C(  3) },
    { UINT8_C(218),
      { -INT64_C( 7383629284268905587), -INT64_C( 5667677927542304513), -INT64_C( 5533222050225190299), -INT64_C( 2326372738731758186) },
      { -INT64_C( 7383629284268905587), -INT64_C( 5667677927542304513), -INT64_C( 5533222050225190299), -INT64_C( 7293782319376068969) },
      UINT8_C(  8) },
    { UINT8_C(227),
      {  INT64_C( 6998870359877860831), -INT64_C( 5291488758399902108), -INT64_C( 1127527867830194694),  INT64_C( 8343533571943168287) },
      { -INT64_C( 5633914840332348691), -INT64_C( 5291488758399902108), -INT64_C( 1127527867830194694),  INT64_C( 8343533571943168287) },
      UINT8_C(  1) },
    { UINT8_C(149),
      {  INT64_C( 1133522684893015680),  INT64_C( 5454768720491511877),  INT64_C( 1421745843164180744),  INT64_C( 8212740468285562879) },
      {  INT64_C( 1133522684893015680), -INT64_C( 5473816741884366722),  INT64_C( 1421745843164180744),  INT64_C( 8212740468285562879) },
      UINT8_C(  0) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epi64_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epi64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i64x4());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpneq_epi64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cmpneq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[4];
    const uint64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { UINT64_C( 7508958499971618362), UINT64_C(13057564781669839762), UINT64_C( 9911934572948098627), UINT64_C( 5013456412997358701) },
      { UINT64_C( 5194676451118254263), UINT64_C(13057564781669839762), UINT64_C( 9911934572948098627), UINT64_C( 5013456412997358701) },
      UINT8_C(  1) },
    { { UINT64_C( 6952060514577292321), UINT64_C(14555508053418286241), UINT64_C( 4634879024334647359), UINT64_C( 8634918177407906299) },
      { UINT64_C( 6952060514577292321), UINT64_C( 3899930323119860193), UINT64_C(16190166767911600005), UINT64_C( 8634918177407906299) },
      UINT8_C(  6) },
    { { UINT64_C(13615944436381990607), UINT64_C(16914650942907042786), UINT64_C( 9285489885333891998), UINT64_C( 4369920374713444788) },
      { UINT64_C( 1612646229694522962), UINT64_C(11305454529837602193), UINT64_C( 9285489885333891998), UINT64_C( 4369920374713444788) },
      UINT8_C(  3) },
    { { UINT64_C(11120862499592152191), UINT64_C(16990451052746199434), UINT64_C( 9446695400822004342), UINT64_C(13425000946219355151) },
      { UINT64_C(11120862499592152191), UINT64_C(13278753835164240933), UINT64_C(  892950767190447734), UINT64_C(10910172702411592818) },
      UINT8_C( 14) },
    { { UINT64_C(17357190605639413991), UINT64_C( 1875163774440324594), UINT64_C( 6777394392104166967), UINT64_C(  825067084183682188) },
      { UINT64_C( 7457863631297205674), UINT64_C(15241021837681275364), UINT64_C( 5371244860847200994), UINT64_C(  825067084183682188) },
      UINT8_C(  7) },
    { { UINT64_C(15040953067368128560), UINT64_C( 4686066498927404074), UINT64_C(16963170552665552048), UINT64_C( 7003489060816865496) },
      { UINT64_C(15040953067368128560), UINT64_C( 4686066498927404074), UINT64_C( 2479207725511213245), UINT64_C( 2397373082558642610) },
      UINT8_C( 12) },
    { { UINT64_C(10992363343097709694), UINT64_C( 5728450421420805433), UINT64_C( 1992767389541464587), UINT64_C( 5625940212794821560) },
      { UINT64_C( 7020915181890924702), UINT64_C( 4555569366923716762), UINT64_C( 4979253361153200540), UINT64_C( 5369768135306793319) },
      UINT8_C( 15) },
    { { UINT64_C( 4048235304364482630), UINT64_C( 2349095941181216603), UINT64_C(  153951078649206751), UINT64_C( 2597442144518887709) },
      { UINT64_C( 8493759324889801240), UINT64_C( 2349095941181216603), UINT64_C(  153951078649206751), UINT64_C( 2597442144518887709) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cmpneq_epu64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_cmpneq_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u64x4());
    easysimd__mmask8 r = easysimd_mm256_cmpneq_epu64_mask(a, b);

    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_cmpneq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint64_t a[4];
    const uint64_t b[4];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(196),
      { UINT64_C( 8423321203886361786), UINT64_C( 4500542638721653489), UINT64_C( 3013465215719210615), UINT64_C(11075241950410493155) },
      { UINT64_C( 8423321203886361786), UINT64_C( 5107300568226966515), UINT64_C(12543620951613624389), UINT64_C(12261421575611206576) },
      UINT8_C(  4) },
    { UINT8_C( 92),
      { UINT64_C(17813821911322702177), UINT64_C( 6898109783627073034), UINT64_C(15117726553355880173), UINT64_C(14774569213450436996) },
      { UINT64_C(17813821911322702177), UINT64_C( 6898109783627073034), UINT64_C(18384141709795362332), UINT64_C(14774569213450436996) },
      UINT8_C(  4) },
    { UINT8_C(234),
      { UINT64_C(16624494916620982973), UINT64_C( 3152124323942929653), UINT64_C(16450834941350132200), UINT64_C(13067984835205292767) },
      { UINT64_C(16624494916620982973), UINT64_C(15376930618073306587), UINT64_C(16450834941350132200), UINT64_C(13067984835205292767) },
      UINT8_C(  2) },
    { UINT8_C(203),
      { UINT64_C(11501891672226311023), UINT64_C( 4662775245196987175), UINT64_C(12445973663410092232), UINT64_C( 3857433989985356673) },
      { UINT64_C(12623058174119680518), UINT64_C( 4662775245196987175), UINT64_C(12445973663410092232), UINT64_C( 3857433989985356673) },
      UINT8_C(  1) },
    { UINT8_C( 91),
      { UINT64_C( 9513606034132277559), UINT64_C(17520231896535345703), UINT64_C(17606657842508059745), UINT64_C(15729241312306395665) },
      { UINT64_C(  496036834137063446), UINT64_C(17520231896535345703), UINT64_C( 9036892027759231342), UINT64_C(15729241312306395665) },
      UINT8_C(  1) },
    { UINT8_C( 51),
      { UINT64_C(15059731879051285701), UINT64_C(17287868481314403184), UINT64_C( 4506665970792486221), UINT64_C(  659329036504521982) },
      { UINT64_C(15059731879051285701), UINT64_C(17287868481314403184), UINT64_C( 7724719468763252770), UINT64_C(  659329036504521982) },
      UINT8_C(  0) },
    { UINT8_C(144),
      { UINT64_C(10132152109655633507), UINT64_C( 7670328621203572430), UINT64_C(12162451600859848560), UINT64_C(  503365921183389836) },
      { UINT64_C(12408804413874978047), UINT64_C( 7670328621203572430), UINT64_C(12162451600859848560), UINT64_C(  503365921183389836) },
      UINT8_C(  0) },
    { UINT8_C( 57),
      { UINT64_C( 9277091886594932733), UINT64_C( 6673892919562782665), UINT64_C(13006255585392512927), UINT64_C( 2919240941080008034) },
      { UINT64_C( 5468522795843103097), UINT64_C( 8825943505004031199), UINT64_C(11608998081130118875), UINT64_C( 2919240941080008034) },
      UINT8_C(  1) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 k1 = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_cmpneq_epu64_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_cmpneq_epu64_mask");
    easysimd_assert_equal_mmask8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_mm256_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u64x4());
    easysimd__mmask8 r = easysimd_mm256_mask_cmpneq_epu64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpneq_epi8_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int8_t a[64];
    const int8_t b[64];
    const uint64_t r;
  } test_vec[8] = {
    { {  INT8_C( 116), -INT8_C(  70), -INT8_C(  72),  INT8_C(  13), -INT8_C(  60), -INT8_C(  94),  INT8_C( 107), -INT8_C(  14),
         INT8_C(  56), -INT8_C(  28), -INT8_C(  50), -INT8_C(  16), -INT8_C(  11),  INT8_C(  66), -INT8_C(  16), -INT8_C(  56),
        -INT8_C(  65), -INT8_C(  38), -INT8_C( 124),  INT8_C( 102),  INT8_C( 109),  INT8_C(  89),  INT8_C(  83),      INT8_MAX,
         INT8_C(   2),  INT8_C(   7), -INT8_C(  28),  INT8_C(  65), -INT8_C( 107),  INT8_C(  64), -INT8_C(  88),  INT8_C(   9),
        -INT8_C(   6),  INT8_C(  96),  INT8_C(  22), -INT8_C(  65),  INT8_C(   2), -INT8_C( 127), -INT8_C(  79),  INT8_C(  58),
         INT8_C( 101),      INT8_MAX,  INT8_C(  42),  INT8_C(  91), -INT8_C(  63),  INT8_C(  26),  INT8_C(  35), -INT8_C( 127),
        -INT8_C(  12), -INT8_C(  89), -INT8_C(  25),  INT8_C(  98),  INT8_C(   1),  INT8_C(  58), -INT8_C(  31),  INT8_C(   3),
         INT8_C(  65), -INT8_C(  59),  INT8_C(  68), -INT8_C(  42),  INT8_C(   5), -INT8_C(  20), -INT8_C(  33),  INT8_C(   0) },
      {  INT8_C( 116), -INT8_C(  70),  INT8_C(  55), -INT8_C(  80), -INT8_C(  60), -INT8_C(  94),  INT8_C(  49), -INT8_C(  14),
         INT8_C(   2), -INT8_C(  28),  INT8_C(  37), -INT8_C(  16),  INT8_C(  83),  INT8_C(   6),  INT8_C(   6), -INT8_C( 107),
        -INT8_C(  53),  INT8_C(  74),  INT8_C( 107), -INT8_C(  47),  INT8_C(  54),  INT8_C(  75),  INT8_C(  83), -INT8_C( 126),
         INT8_C(  64), -INT8_C( 112), -INT8_C(  47), -INT8_C(  73), -INT8_C( 107),  INT8_C(  64), -INT8_C( 108),  INT8_C(   9),
         INT8_C(  15), -INT8_C(  53), -INT8_C(  97), -INT8_C(  65),  INT8_C(  38), -INT8_C(  47), -INT8_C(  96),  INT8_C(  58),
         INT8_C( 101),      INT8_MAX,  INT8_C(  42),  INT8_C(  91), -INT8_C(  53),  INT8_C(  49), -INT8_C(  46), -INT8_C( 127),
         INT8_C( 123), -INT8_C(  89), -INT8_C(  25), -INT8_C(  79),  INT8_C(   1),  INT8_C(  58), -INT8_C(  31), -INT8_C(  55),
         INT8_C(  65), -INT8_C(  59), -INT8_C( 127), -INT8_C(  55),  INT8_C(  96), -INT8_C(  20), -INT8_C(  72),  INT8_C( 111) },
      UINT64_C(15891356417894905164) },
    { { -INT8_C(  32),  INT8_C(  87),  INT8_C(  76),  INT8_C(   6),  INT8_C(  40), -INT8_C(  19),  INT8_C(  46),  INT8_C(  18),
        -INT8_C(  78),  INT8_C(  89),  INT8_C(  80),  INT8_C( 126), -INT8_C( 118),  INT8_C(  34),  INT8_C(  21),  INT8_C(   5),
         INT8_C(  96),  INT8_C( 125), -INT8_C(  74), -INT8_C(  23), -INT8_C(  74), -INT8_C(  22), -INT8_C(  77),      INT8_MAX,
        -INT8_C(  17),  INT8_C(  52),  INT8_C(  72),  INT8_C(  79),  INT8_C(  73),  INT8_C(   0), -INT8_C(  66),  INT8_C(  41),
         INT8_C(  87),  INT8_C(  10),  INT8_C(  48),      INT8_MIN, -INT8_C(   9),  INT8_C(  94), -INT8_C( 110), -INT8_C(  86),
        -INT8_C(  72), -INT8_C(  30),  INT8_C(  40),  INT8_C(  66),  INT8_C(   5),  INT8_C(  61),  INT8_C(  72),  INT8_C( 101),
        -INT8_C(  70), -INT8_C(   2),  INT8_C(  79),  INT8_C( 112), -INT8_C(  24),  INT8_C(   2), -INT8_C(  17), -INT8_C(  41),
         INT8_C(  54),  INT8_C(  55),  INT8_C(  38),      INT8_MAX,  INT8_C(  55), -INT8_C(  28), -INT8_C(  88), -INT8_C( 114) },
      { -INT8_C( 125), -INT8_C(  72),  INT8_C(  49), -INT8_C( 120),  INT8_C(  40),  INT8_C( 121), -INT8_C(  18), -INT8_C(  81),
        -INT8_C(  78),  INT8_C(  89),  INT8_C(  80),  INT8_C(  96),  INT8_C(  63),  INT8_C(  34),  INT8_C(  56),  INT8_C( 117),
         INT8_C(  96),  INT8_C(  94), -INT8_C(  12),  INT8_C( 124), -INT8_C(  74), -INT8_C(  22), -INT8_C(  77),      INT8_MAX,
        -INT8_C(  17),  INT8_C(  25),  INT8_C(  24),  INT8_C(  79),  INT8_C(  73), -INT8_C(  87), -INT8_C( 101),  INT8_C(  62),
         INT8_C(  97), -INT8_C(  52), -INT8_C(  58),      INT8_MIN,  INT8_C(  70), -INT8_C(  76), -INT8_C( 110), -INT8_C(  86),
        -INT8_C(  15), -INT8_C(  30),  INT8_C(  40),  INT8_C(  66),  INT8_C(   5),  INT8_C(  86),  INT8_C(  72),  INT8_C( 122),
        -INT8_C(  70), -INT8_C(   2),  INT8_C(  79),  INT8_C( 112),  INT8_C(  54),  INT8_C(   2), -INT8_C(  17), -INT8_C(  85),
         INT8_C(  27),  INT8_C(  66),  INT8_C(  87), -INT8_C(  43),  INT8_C(  55), -INT8_C(  14),  INT8_C(  19),  INT8_C(  77) },
      UINT64_C(17262474633166117103) },
    { { -INT8_C(  66), -INT8_C(  38), -INT8_C(  92),  INT8_C(   4), -INT8_C( 114), -INT8_C(  86), -INT8_C(  62),      INT8_MIN,
        -INT8_C(  48), -INT8_C(  31), -INT8_C(  80),  INT8_C(   5),  INT8_C(  55),  INT8_C(  86),      INT8_MAX, -INT8_C(  20),
        -INT8_C(  17),  INT8_C( 117), -INT8_C(  28),  INT8_C(  37),  INT8_C( 119),  INT8_C(  14), -INT8_C(  48), -INT8_C( 110),
         INT8_C(  81),  INT8_C(  39),  INT8_C( 103),  INT8_C(  60),  INT8_C(  25),  INT8_C( 123), -INT8_C( 119), -INT8_C(  40),
         INT8_C(  85),  INT8_C(  45), -INT8_C(  36), -INT8_C(  29), -INT8_C(  40), -INT8_C(  97),  INT8_C(  99), -INT8_C(  88),
             INT8_MIN,  INT8_C(  20), -INT8_C(  83), -INT8_C(  73),  INT8_C( 106),  INT8_C(  44), -INT8_C(  92),  INT8_C(  89),
        -INT8_C(  94), -INT8_C( 120),      INT8_MAX,  INT8_C(  25), -INT8_C( 105),  INT8_C(  79), -INT8_C(  85), -INT8_C(  24),
         INT8_C( 119),  INT8_C(  18),  INT8_C(  36), -INT8_C( 112), -INT8_C( 115), -INT8_C(  82),  INT8_C( 104), -INT8_C(  30) },
      {  INT8_C(  61),  INT8_C(   9), -INT8_C(  92), -INT8_C(  89),  INT8_C(  54), -INT8_C(  86),  INT8_C(   1), -INT8_C(  40),
         INT8_C(  72), -INT8_C(  31), -INT8_C(  15),  INT8_C(   5),  INT8_C(  55),  INT8_C(  86), -INT8_C(  57), -INT8_C(  20),
        -INT8_C(  17), -INT8_C(  21), -INT8_C(  41),  INT8_C(  37),  INT8_C( 119),  INT8_C(  14),  INT8_C(  30),  INT8_C( 117),
        -INT8_C( 124), -INT8_C(  28),  INT8_C( 103),  INT8_C(  60),  INT8_C(  14), -INT8_C( 124), -INT8_C( 119),  INT8_C(  75),
         INT8_C(  85),  INT8_C(  45), -INT8_C(  13), -INT8_C(  29), -INT8_C(  40), -INT8_C(  12), -INT8_C( 100), -INT8_C(  17),
         INT8_C( 116),  INT8_C(  20), -INT8_C(  83),  INT8_C(  67),  INT8_C( 106), -INT8_C( 107), -INT8_C(  92),  INT8_C(  89),
        -INT8_C(  94), -INT8_C( 120),  INT8_C(  19),  INT8_C(  26), -INT8_C(  96),  INT8_C(  79), -INT8_C( 113), -INT8_C(  24),
         INT8_C( 119),  INT8_C(  18), -INT8_C( 115), -INT8_C( 112), -INT8_C( 115),  INT8_C(  90),  INT8_C( 112), -INT8_C(  30) },
      UINT64_C( 7231701163895571931) },
    { {  INT8_C(  66),  INT8_C(  99), -INT8_C( 114), -INT8_C(  23),  INT8_C(  87),  INT8_C(  42), -INT8_C(  39), -INT8_C(  53),
        -INT8_C(  73), -INT8_C(  89),  INT8_C(  14), -INT8_C(  32),  INT8_C(  61), -INT8_C( 104), -INT8_C(  72), -INT8_C(  66),
        -INT8_C(   7), -INT8_C(  53), -INT8_C(  40), -INT8_C( 102), -INT8_C(   3),  INT8_C( 104), -INT8_C(  65),  INT8_C(  20),
         INT8_C(  32),  INT8_C(  76),  INT8_C(  56),  INT8_C(  92), -INT8_C(  90), -INT8_C(  88),  INT8_C(  39), -INT8_C(  24),
         INT8_C(  11), -INT8_C(  75), -INT8_C(  46),  INT8_C(  98), -INT8_C(  32), -INT8_C(  85),  INT8_C(  45), -INT8_C( 105),
         INT8_C(  82),  INT8_C(  60),  INT8_C( 120), -INT8_C( 113), -INT8_C(  44),  INT8_C(  48),  INT8_C(  77), -INT8_C(  50),
        -INT8_C(   5),  INT8_C(  38),  INT8_C( 104), -INT8_C(   7), -INT8_C( 114),  INT8_C(  39),  INT8_C(  13), -INT8_C(  82),
         INT8_C( 115),  INT8_C(  69),  INT8_C(  10),  INT8_C(  26), -INT8_C(  18),  INT8_C(  49),  INT8_C(   2), -INT8_C(   7) },
      { -INT8_C(  59), -INT8_C(  42),  INT8_C(  97), -INT8_C(  23),  INT8_C(  87), -INT8_C(  81),  INT8_C( 104),  INT8_C(   2),
        -INT8_C(  73), -INT8_C(  89), -INT8_C(   5), -INT8_C(  32), -INT8_C(   9), -INT8_C( 104),  INT8_C(  17),  INT8_C( 106),
        -INT8_C(   7), -INT8_C(  53), -INT8_C( 124),  INT8_C(  59),  INT8_C(  77),  INT8_C( 104),  INT8_C(  53),  INT8_C(  20),
         INT8_C(  91), -INT8_C( 111), -INT8_C(   5),  INT8_C(  92), -INT8_C(  90), -INT8_C(  88), -INT8_C(  83), -INT8_C(  32),
         INT8_C(  48),  INT8_C(  14),  INT8_C( 122),  INT8_C(  54), -INT8_C(  67), -INT8_C(  30),  INT8_C(  56), -INT8_C( 105),
        -INT8_C(  78),  INT8_C(  60),  INT8_C( 120), -INT8_C(  87), -INT8_C(  44),  INT8_C(  48),  INT8_C(  77), -INT8_C( 119),
        -INT8_C(   5), -INT8_C( 104), -INT8_C(  60),  INT8_C( 111),  INT8_C(  31),  INT8_C(  39), -INT8_C(  93), -INT8_C(  82),
         INT8_C( 115), -INT8_C(  98),  INT8_C(  10),  INT8_C(  26), -INT8_C(   9),  INT8_C(  49), -INT8_C( 123),  INT8_C(  39) },
      UINT64_C(15158704577674269927) },
    { {  INT8_C(  17), -INT8_C(   1),  INT8_C(  94), -INT8_C(  50), -INT8_C(  31), -INT8_C( 106),  INT8_C(  97), -INT8_C( 109),
        -INT8_C(  54),  INT8_C(  86),  INT8_C(  60),  INT8_C(   5),  INT8_C(  93),  INT8_C(  79), -INT8_C( 114),      INT8_MAX,
        -INT8_C(  25),  INT8_C(  83), -INT8_C(  18),  INT8_C(   6),  INT8_C(  76), -INT8_C( 111), -INT8_C( 127), -INT8_C(  41),
         INT8_C(  47), -INT8_C(  42),  INT8_C( 124),  INT8_C(  38), -INT8_C(  39),  INT8_C(   1),  INT8_C(  78), -INT8_C(  22),
         INT8_C(   0), -INT8_C(  84), -INT8_C(  72), -INT8_C(  31),  INT8_C(  66),  INT8_C(  25),  INT8_C( 116),  INT8_C(  12),
         INT8_C( 112), -INT8_C(  80),  INT8_C(  18), -INT8_C(  51), -INT8_C(   1), -INT8_C(  96),  INT8_C(  76), -INT8_C(  25),
        -INT8_C(  13),  INT8_C(  58), -INT8_C(  19),  INT8_C(  64), -INT8_C(  53),  INT8_C( 110),  INT8_C(  23), -INT8_C(   6),
         INT8_C(  69), -INT8_C( 109),  INT8_C(  32),  INT8_C(  30), -INT8_C( 108),  INT8_C( 110),  INT8_C(   8), -INT8_C( 108) },
      {  INT8_C(  17),  INT8_C( 123),  INT8_C(  94), -INT8_C( 104),  INT8_C(  28), -INT8_C( 106),  INT8_C(  97), -INT8_C( 109),
        -INT8_C(  54),  INT8_C(  86),  INT8_C(  60),  INT8_C(   5),  INT8_C(  93),  INT8_C(  79),  INT8_C(  98),  INT8_C(  32),
        -INT8_C(   7),  INT8_C(  83),  INT8_C(  62),  INT8_C(   6), -INT8_C(  15),  INT8_C(  70),  INT8_C(  33), -INT8_C(  41),
         INT8_C(   7), -INT8_C(  42),  INT8_C( 104), -INT8_C(  31),      INT8_MAX,  INT8_C(   1),  INT8_C(  43), -INT8_C(  22),
         INT8_C(   0),  INT8_C(  66), -INT8_C(  72),  INT8_C( 105), -INT8_C(  91),  INT8_C(  25),  INT8_C( 121),  INT8_C(  66),
        -INT8_C(  99), -INT8_C(  80),  INT8_C(  18),  INT8_C( 121), -INT8_C(   1),  INT8_C(  12), -INT8_C( 103),  INT8_C(  40),
        -INT8_C( 114),  INT8_C(  58), -INT8_C(  19),      INT8_MAX, -INT8_C(  53), -INT8_C(  41), -INT8_C( 117), -INT8_C(   6),
         INT8_C(  69), -INT8_C(  13),  INT8_C(  32), -INT8_C(  19), -INT8_C( 108),  INT8_C( 110),  INT8_C(   6), -INT8_C( 108) },
      UINT64_C( 5362073955441426458) },
    { {  INT8_C( 115), -INT8_C(  73),  INT8_C( 124),  INT8_C(  24), -INT8_C(  25), -INT8_C(  11),  INT8_C(  90), -INT8_C( 123),
        -INT8_C(  66),  INT8_C(   4), -INT8_C(   2), -INT8_C(  19),  INT8_C(  16), -INT8_C( 105),  INT8_C(  21), -INT8_C(  97),
         INT8_C( 111), -INT8_C(  53),  INT8_C(  30), -INT8_C( 114), -INT8_C(  93), -INT8_C(  87), -INT8_C(  77),  INT8_C(  17),
        -INT8_C(  99), -INT8_C(  70), -INT8_C(   2),  INT8_C(  98), -INT8_C(  21),  INT8_C(   4),  INT8_C( 117),  INT8_C(  95),
        -INT8_C(  69), -INT8_C(  14),  INT8_C( 119), -INT8_C(  93), -INT8_C(  25), -INT8_C(  46),  INT8_C(  40), -INT8_C(  91),
        -INT8_C(  42),  INT8_C(  38), -INT8_C( 110), -INT8_C(  25), -INT8_C(  67), -INT8_C(  88), -INT8_C( 122),  INT8_C(  45),
         INT8_C( 115), -INT8_C(  92), -INT8_C(  69),  INT8_C(  22),  INT8_C(  78),  INT8_C( 110),  INT8_C(  39), -INT8_C(  21),
         INT8_C(  40),  INT8_C(  38),  INT8_C(  77),  INT8_C(  20),  INT8_C(  42), -INT8_C(  61),  INT8_C( 115), -INT8_C(  26) },
      { -INT8_C(  41), -INT8_C(  73),  INT8_C( 122),  INT8_C(  24),  INT8_C( 124),  INT8_C(   0),  INT8_C(  90), -INT8_C(  16),
        -INT8_C(  66),  INT8_C( 124), -INT8_C(   2), -INT8_C(  14),  INT8_C(  16),  INT8_C(  46), -INT8_C(  35),  INT8_C(  19),
         INT8_C(  84), -INT8_C(  53),  INT8_C(  30),  INT8_C( 126), -INT8_C(  93), -INT8_C(  87), -INT8_C(  77), -INT8_C(  93),
        -INT8_C(  99), -INT8_C(  70),  INT8_C(  63),  INT8_C(  65), -INT8_C(  98),  INT8_C(   4),  INT8_C( 117),  INT8_C( 117),
        -INT8_C(  69), -INT8_C(  14),  INT8_C(  10), -INT8_C(  46),  INT8_C(  78), -INT8_C(  53),  INT8_C(  40), -INT8_C(  13),
         INT8_C(  72),  INT8_C(  38), -INT8_C( 110), -INT8_C(  25), -INT8_C(   9), -INT8_C(  61), -INT8_C( 122),  INT8_C(  75),
         INT8_C( 115),  INT8_C( 110), -INT8_C(  69),  INT8_C(  22),  INT8_C(  78),  INT8_C( 110),      INT8_MAX, -INT8_C(  21),
         INT8_C(  27), -INT8_C(  66),  INT8_C(  77),  INT8_C(  20),  INT8_C(  64), -INT8_C(  61),  INT8_C( 115), -INT8_C( 106) },
      UINT64_C(10611239095676562101) },
    { { -INT8_C(  14),  INT8_C(  57),  INT8_C( 104),  INT8_C(  64),  INT8_C(   5),  INT8_C(  43),  INT8_C(  51),  INT8_C(  77),
        -INT8_C(  12),  INT8_C(  25),      INT8_MIN, -INT8_C(  21), -INT8_C(  36), -INT8_C(  58),  INT8_C(  54), -INT8_C(  54),
         INT8_C(  52), -INT8_C(   1), -INT8_C(  90),  INT8_C(  61),  INT8_C(  45),  INT8_C(  37), -INT8_C(  54),  INT8_C(  73),
        -INT8_C(  29), -INT8_C( 103),  INT8_C(   3),  INT8_C(  35),  INT8_C(  61),  INT8_C(  50), -INT8_C(  71),  INT8_C(  47),
         INT8_C( 108),  INT8_C(  34),  INT8_C( 111),  INT8_C( 113),  INT8_C(  77), -INT8_C(  93), -INT8_C(  66),  INT8_C(  65),
        -INT8_C(  68),  INT8_C(  62),  INT8_C(  44), -INT8_C( 104),  INT8_C(   4),  INT8_C(  98),  INT8_C(  98),  INT8_C(  57),
         INT8_C(  97),  INT8_C(   8),  INT8_C( 118), -INT8_C( 113),  INT8_C(  45),  INT8_C(  64), -INT8_C(  40),  INT8_C(  16),
        -INT8_C(  38), -INT8_C(  37),  INT8_C(  52),  INT8_C(  23),  INT8_C(  13), -INT8_C(  19),  INT8_C(  70),  INT8_C( 121) },
      { -INT8_C(  26), -INT8_C(  55), -INT8_C(  84), -INT8_C(  21),  INT8_C(   5),  INT8_C(  43),  INT8_C(  51),  INT8_C(  77),
         INT8_C(  22),  INT8_C(  25),  INT8_C(  28), -INT8_C(  21), -INT8_C(  38), -INT8_C(  12),  INT8_C(  54), -INT8_C(  76),
         INT8_C(  52), -INT8_C( 120), -INT8_C(  90), -INT8_C(  36),  INT8_C(  45),  INT8_C(  17),  INT8_C(  86), -INT8_C( 123),
        -INT8_C(  29), -INT8_C( 103), -INT8_C(  31),  INT8_C(  31), -INT8_C(  23),  INT8_C(  50),  INT8_C(  52),  INT8_C(  47),
         INT8_C( 108),  INT8_C(  34),  INT8_C( 111),  INT8_C( 116), -INT8_C(  17), -INT8_C(  93),  INT8_C(   1),  INT8_C(  65),
        -INT8_C(  68),  INT8_C(  62),  INT8_C(  44),  INT8_C(  83),  INT8_C(   4), -INT8_C(  99),  INT8_C(  98), -INT8_C(  32),
         INT8_C(  37),  INT8_C(   8), -INT8_C(  68), -INT8_C( 102), -INT8_C(  28),  INT8_C(  64), -INT8_C(  40), -INT8_C(  85),
        -INT8_C(  38), -INT8_C(  37), -INT8_C(  53),  INT8_C(  23),      INT8_MIN, -INT8_C(  19),  INT8_C(  70),  INT8_C( 121) },
      UINT64_C( 1485528549571605775) },
    { { -INT8_C(  33), -INT8_C(  58),  INT8_C(  60), -INT8_C(  50), -INT8_C(  92),  INT8_C(  61), -INT8_C(  44),  INT8_C(  29),
         INT8_C(  90),  INT8_C(  29),  INT8_C( 112),  INT8_C( 107), -INT8_C(  70),  INT8_C( 119),  INT8_C(  75), -INT8_C(  33),
         INT8_C(  74),  INT8_C(   8),  INT8_C( 121),  INT8_C(  47),  INT8_C(  26), -INT8_C( 103), -INT8_C(  38),  INT8_C( 109),
        -INT8_C( 102), -INT8_C(  91), -INT8_C(  87),  INT8_C(  26), -INT8_C(  92), -INT8_C(  75), -INT8_C(  30), -INT8_C( 124),
         INT8_C( 123),  INT8_C(  31),  INT8_C(  82),  INT8_C(  31),  INT8_C(  92),  INT8_C(  38),  INT8_C(  60), -INT8_C(  73),
         INT8_C(  67), -INT8_C(  84),  INT8_C(  34), -INT8_C(   3),  INT8_C(  36),  INT8_C( 110), -INT8_C(  36),  INT8_C( 110),
         INT8_C( 118),  INT8_C(  86), -INT8_C(  99), -INT8_C( 112), -INT8_C(  17),  INT8_C( 120), -INT8_C(   2), -INT8_C( 119),
         INT8_C(  29), -INT8_C(  89), -INT8_C(  93), -INT8_C(  62),  INT8_C(  92), -INT8_C( 123),  INT8_C(  70), -INT8_C(  41) },
      { -INT8_C(  33), -INT8_C(  58),  INT8_C(   0), -INT8_C(  50), -INT8_C(  92), -INT8_C(  36), -INT8_C(  44), -INT8_C(  66),
         INT8_C(  90),  INT8_C(  29),  INT8_C( 112),  INT8_C(  33), -INT8_C( 120),  INT8_C( 119),  INT8_C(  75), -INT8_C(  91),
        -INT8_C(  12),  INT8_C(  77),  INT8_C( 103),  INT8_C(  47), -INT8_C(  45), -INT8_C(  83),  INT8_C(  40),  INT8_C( 119),
         INT8_C(  70), -INT8_C(  91), -INT8_C(  87),  INT8_C(  26), -INT8_C(  92), -INT8_C(  75), -INT8_C(  30), -INT8_C( 124),
         INT8_C(  11),  INT8_C(   7),  INT8_C(  55),  INT8_C(  83), -INT8_C(  28), -INT8_C(  87),  INT8_C(  60),  INT8_C(  22),
        -INT8_C(  71),  INT8_C(  97),  INT8_C(  34), -INT8_C(   3), -INT8_C(  82), -INT8_C(  30), -INT8_C(  36),  INT8_C( 110),
         INT8_C( 118),  INT8_C(  86), -INT8_C(  99),  INT8_C(   3), -INT8_C(   4),  INT8_C(  28), -INT8_C(   2),  INT8_C(  66),
         INT8_C(  29), -INT8_C(  13), -INT8_C(  93), -INT8_C(  62),  INT8_C(  92), -INT8_C( 123),  INT8_C(  70), -INT8_C(  41) },
      UINT64_C(  195963479255390372) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epi8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epi8_mask");
    easysimd_assert_equal_mmask64(r, test_vec[i].r);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi8(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_i8x64());
    easysimd__mmask64 r = easysimd_mm512_cmpneq_epi8_mask(a, b);

    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_cmpneq_epi16_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    const int16_t a[32];
    const int16_t b[32];
    const uint32_t r;
  } test_vec[8] = {
    { { -INT16_C( 11195), -INT16_C( 31167),  INT16_C(  4028), -INT16_C(  2032), -INT16_C(  5749),  INT16_C( 31655),  INT16_C( 31413), -INT16_C( 14402),
         INT16_C( 29443),  INT16_C( 26386), -INT16_C(    83),  INT16_C(  2956),  INT16_C( 22539),  INT16_C( 15849), -INT16_C( 19672),  INT16_C( 28146),
         INT16_C( 13447),  INT16_C( 17395),  INT16_C(  1091), -INT16_C( 12741), -INT16_C(  7443), -INT16_C( 23990),  INT16_C(  2140),  INT16_C( 24426),
         INT16_C( 31867),  INT16_C( 10694),  INT16_C( 21371), -INT16_C( 31180),  INT16_C(  7595), -INT16_C( 11325), -INT16_C( 18736),  INT16_C( 22337) },
      {  INT16_C( 13546), -INT16_C( 31167), -INT16_C( 10952), -INT16_C(  2032), -INT16_C(  5749),  INT16_C( 31655),  INT16_C( 31413), -INT16_C( 13710),
         INT16_C( 29443),  INT16_C( 10995),  INT16_C( 10124),  INT16_C(  2956),  INT16_C( 29765),  INT16_C( 15849),  INT16_C( 19242),  INT16_C( 28146),
         INT16_C( 13447),  INT16_C( 17395),  INT16_C(  1091), -INT16_C( 12741), -INT16_C(  7443), -INT16_C( 23990),  INT16_C(  2140),  INT16_C( 24426),
         INT16_C( 31867),  INT16_C( 10694),  INT16_C( 21371), -INT16_C( 31180),  INT16_C(  7595), -INT16_C( 11325), -INT16_C( 18736),  INT16_C( 22337) },
      UINT32_C(     22149) },
    { {  INT16_C( 25253), -INT16_C( 31853),  INT16_C(  6135), -INT16_C( 24790),  INT16_C(  1257), -INT16_C( 31303),  INT16_C(  3468),  INT16_C( 15892),
        -INT16_C( 13588),  INT16_C(   928),  INT16_C( 30662), -INT16_C( 10460), -INT16_C( 28553), -INT16_C( 29611), -INT16_C(  9603),  INT16_C(  8930),
         INT16_C( 30013),  INT16_C( 13477), -INT16_C( 12148),  INT16_C( 30163), -INT16_C( 29484),  INT16_C( 24826),  INT16_C(  3737), -INT16_C( 31074),
         INT16_C( 16089), -INT16_C( 24695), -INT16_C( 21067),  INT16_C( 11638), -INT16_C( 13507), -INT16_C( 17735), -INT16_C( 25691), -INT16_C(  7460) },
      { -INT16_C( 32240), -INT16_C( 31853),  INT16_C(  6135), -INT16_C( 24790),  INT16_C(  3446),  INT16_C(  3974),  INT16_C(  3468),  INT16_C( 15892),
        -INT16_C( 13588),  INT16_C(  6291),  INT16_C( 30662),  INT16_C(  2373), -INT16_C( 28553), -INT16_C( 29611), -INT16_C(  9603), -INT16_C( 22180),
         INT16_C( 30013),  INT16_C( 13477), -INT16_C( 12148),  INT16_C( 30163), -INT16_C( 29484),  INT16_C( 24826),  INT16_C(  3737), -INT16_C( 31074),
         INT16_C( 16089), -INT16_C( 24695), -INT16_C( 21067),  INT16_C( 11638), -INT16_C( 13507), -INT16_C( 17735), -INT16_C( 25691), -INT16_C(  7460) },
      UINT32_C(     35377) },
    { { -INT16_C( 29340), -INT16_C(   285),  INT16_C( 18784),  INT16_C( 16926),  INT16_C( 25291),  INT16_C( 16825),  INT16_C( 20232), -INT16_C( 14518),
         INT16_C( 23985),  INT16_C(  7114), -INT16_C( 13243), -INT16_C( 22967),  INT16_C(  5990), -INT16_C( 21916), -INT16_C( 27385),  INT16_C( 27445),
         INT16_C(  6179), -INT16_C( 31895), -INT16_C( 30878),  INT16_C( 11717),  INT16_C( 32489), -INT16_C(  3474), -INT16_C( 18226),  INT16_C( 32697),
        -INT16_C( 31978),  INT16_C( 23450), -INT16_C(  7089), -INT16_C( 18943),  INT16_C( 26363),  INT16_C(   864), -INT16_C( 27141),  INT16_C(  7790) },
      { -INT16_C( 10066), -INT16_C(   285),  INT16_C( 18784),  INT16_C( 16926),  INT16_C( 25291), -INT16_C( 19653), -INT16_C(  2973), -INT16_C( 14518),
        -INT16_C( 13193), -INT16_C( 14380), -INT16_C( 13243), -INT16_C( 21379),  INT16_C(  5990),  INT16_C( 14255), -INT16_C( 27385),  INT16_C( 27445),
         INT16_C(  6179), -INT16_C( 31895), -INT16_C( 30878),  INT16_C( 11717),  INT16_C( 32489), -INT16_C(  3474), -INT16_C( 18226),  INT16_C( 32697),
        -INT16_C( 31978),  INT16_C( 23450), -INT16_C(  7089), -INT16_C( 18943),  INT16_C( 26363),  INT16_C(   864), -INT16_C( 27141),  INT16_C(  7790) },
      UINT32_C(     11105) },
    { { -INT16_C( 16536),  INT16_C(  1689), -INT16_C( 19966), -INT16_C(  1825), -INT16_C( 21458),  INT16_C(  9248),  INT16_C(  5616), -INT16_C(   785),
        -INT16_C( 28742),  INT16_C(  2948), -INT16_C(  5524), -INT16_C( 32757),  INT16_C( 10435), -INT16_C( 16918),  INT16_C( 19259), -INT16_C( 23576),
        -INT16_C( 32501),  INT16_C(  3497), -INT16_C( 30669),  INT16_C( 25094),  INT16_C(  9780),  INT16_C(  9606),  INT16_C( 30011), -INT16_C(  2783),
        -INT16_C( 23291),  INT16_C( 28928),  INT16_C(  2960),  INT16_C( 21489), -INT16_C(  9420),  INT16_C( 28433), -INT16_C(  1754),  INT16_C( 12563) },
      { -INT16_C( 17285), -INT16_C( 20929),  INT16_C( 17733), -INT16_C(  1825), -INT16_C( 21458),  INT16_C(  9248),  INT16_C(  5616),  INT16_C(  4508),
        -INT16_C( 25243),  INT16_C(  2948), -INT16_C(  5524), -INT16_C( 32757),  INT16_C( 22862), -INT16_C( 16918),  INT16_C( 24403), -INT16_C( 12634),
        -INT16_C( 32501),  INT16_C(  3497), -INT16_C( 30669),  INT16_C( 25094),  INT16_C(  9780),  INT16_C(  9606),  INT16_C( 30011), -INT16_C(  2783),
        -INT16_C( 23291),  INT16_C( 28928),  INT16_C(  2960),  INT16_C( 21489), -INT16_C(  9420),  INT16_C( 28433), -INT16_C(  1754),  INT16_C( 12563) },
      UINT32_C(     53639) },
    { { -INT16_C( 19934), -INT16_C(   930), -INT16_C( 32184), -INT16_C( 31371), -INT16_C( 21069), -INT16_C(  2722), -INT16_C( 10934), -INT16_C(  9031),
        -INT16_C(  3596), -INT16_C(  4170), -INT16_C(  5512),  INT16_C( 29495), -INT16_C(  8847), -INT16_C( 14827), -INT16_C( 25185), -INT16_C( 15720),
        -INT16_C(  2481), -INT16_C( 26690),  INT16_C( 13177),  INT16_C( 11292),  INT16_C( 31456),  INT16_C( 10785), -INT16_C(  9649),  INT16_C( 17158),
        -INT16_C( 16948),  INT16_C( 17459),  INT16_C( 27303),  INT16_C(  6583), -INT16_C( 12985), -INT16_C(  6177),  INT16_C( 30570), -INT16_C( 18007) },
      { -INT16_C( 19934), -INT16_C(  6320),  INT16_C( 27803), -INT16_C( 31371), -INT16_C( 21069), -INT16_C(  2722), -INT16_C( 10934), -INT16_C(  9607),
        -INT16_C(  3596),  INT16_C(  4382), -INT16_C(  5512),  INT16_C( 29495),  INT16_C(  2467),  INT16_C(  3397), -INT16_C( 25185), -INT16_C(  4154),
        -INT16_C(  2481), -INT16_C( 26690),  INT16_C( 13177),  INT16_C( 11292),  INT16_C( 31456),  INT16_C( 10785), -INT16_C(  9649),  INT16_C( 17158),
        -INT16_C( 16948),  INT16_C( 17459),  INT16_C( 27303),  INT16_C(  6583), -INT16_C( 12985), -INT16_C(  6177),  INT16_C( 30570), -INT16_C( 18007) },
      UINT32_C(     45702) },
    { {  INT16_C(  2226),  INT16_C(  7835), -INT16_C( 18319),  INT16_C(  3888), -INT16_C(  4380), -INT16_C(  5593), -INT16_C(  5098),  INT16_C( 20238),
         INT16_C(  2503),  INT16_C(   178),  INT16_C(  8102),  INT16_C( 20862), -INT16_C(  5620), -INT16_C(  5695),  INT16_C( 18603),  INT16_C( 23963),
         INT16_C( 14160), -INT16_C( 16005), -INT16_C( 21521), -INT16_C( 11311), -INT16_C(  1894), -INT16_C( 20291), -INT16_C( 13084), -INT16_C( 21760),
        -INT16_C( 19755),  INT16_C( 31659),  INT16_C( 10705), -INT16_C(  8500), -INT16_C( 29165), -INT16_C( 16441),  INT16_C( 25302),  INT16_C(  9756) },
      {  INT16_C(  2226),  INT16_C(  7835), -INT16_C( 18319), -INT16_C(  8868), -INT16_C(  4380), -INT16_C(  5593), -INT16_C(  5098), -INT16_C( 17599),
         INT16_C(  2503),  INT16_C(   178),  INT16_C(   790),  INT16_C( 10992), -INT16_C( 18543), -INT16_C(  5695),  INT16_C(  1305), -INT16_C( 19571),
         INT16_C( 14160), -INT16_C( 16005), -INT16_C( 21521), -INT16_C( 11311), -INT16_C(  1894), -INT16_C( 20291), -INT16_C( 13084), -INT16_C( 21760),
        -INT16_C( 19755),  INT16_C( 31659),  INT16_C( 10705), -INT16_C(  8500), -INT16_C( 29165), -INT16_C( 16441),  INT16_C( 25302),  INT16_C(  9756) },
      UINT32_C(     56456) },
    { { -INT16_C( 18862),  INT16_C(  4213),  INT16_C( 10134),  INT16_C(  3165),  INT16_C( 14272),  INT16_C(  5060),  INT16_C( 26706),  INT16_C( 32669),
         INT16_C( 10787),  INT16_C(  2204),  INT16_C( 28744), -INT16_C( 12842), -INT16_C( 21922), -INT16_C(   288),  INT16_C( 26651),  INT16_C( 28123),
         INT16_C( 20510), -INT16_C( 19075), -INT16_C(  9609),  INT16_C( 14273), -INT16_C( 31470),  INT16_C( 25675), -INT16_C(  5906),  INT16_C(  4580),
        -INT16_C( 32749),  INT16_C( 23322), -INT16_C(  3855),  INT16_C( 20265),  INT16_C(  2458), -INT16_C( 19123),  INT16_C( 10353), -INT16_C( 28894) },
      { -INT16_C( 24456), -INT16_C(  4028),  INT16_C(  1658),  INT16_C(  3165),  INT16_C( 29323),  INT16_C( 31217),  INT16_C( 26706),  INT16_C( 32669),
        -INT16_C( 23211),  INT16_C(  2204), -INT16_C(  3435),  INT16_C( 12437), -INT16_C( 21922), -INT16_C(   288),  INT16_C( 26651), -INT16_C( 31492),
         INT16_C( 20510), -INT16_C( 19075), -INT16_C(  9609),  INT16_C( 14273), -INT16_C( 31470),  INT16_C( 25675), -INT16_C(  5906),  INT16_C(  4580),
        -INT16_C( 32749),  INT16_C( 23322), -INT16_C(  3855),  INT16_C( 20265),  INT16_C(  2458), -INT16_C( 19123),  INT16_C( 10353), -INT16_C( 28894) },
      UINT32_C(     36151) },
    { {  INT16_C( 32505), -INT16_C( 22488),  INT16_C( 13904), -INT16_C( 25784), -INT16_C( 16993),  INT16_C( 30322), -INT16_C(  4729), -INT16_C( 26601),
        -INT16_C( 21762),  INT16_C( 16191), -INT16_C( 14280),  INT16_C( 13158), -INT16_C( 27555),  INT16_C( 30250),  INT16_C( 25195),  INT16_C( 25603),
         INT16_C( 11488),  INT16_C( 12300),  INT16_C( 21602),  INT16_C(   715),  INT16_C( 15633), -INT16_C( 26504), -INT16_C( 28885),  INT16_C( 10545),
         INT16_C( 28729),  INT16_C( 29289), -INT16_C( 12488), -INT16_C( 26971), -INT16_C( 12189), -INT16_C( 12788),  INT16_C(  4146),  INT16_C(  4658) },
      {  INT16_C( 32505), -INT16_C( 25022),  INT16_C(  3474), -INT16_C( 25784),  INT16_C(  6475),  INT16_C( 30322),  INT16_C( 28072), -INT16_C( 26601),
        -INT16_C( 21762),  INT16_C(  5460), -INT16_C(  1577),  INT16_C( 15275), -INT16_C( 18231),  INT16_C( 30250),  INT16_C( 25195),  INT16_C( 25603),
         INT16_C( 11488),  INT16_C( 12300),  INT16_C( 21602),  INT16_C(   715),  INT16_C( 15633), -INT16_C( 26504), -INT16_C( 28885),  INT16_C( 10545),
         INT16_C( 28729),  INT16_C( 29289), -INT16_C( 12488), -INT16_C( 26971), -INT16_C( 12189), -INT16_C( 12788),  INT16_C(  4146),  INT16_C(  4658) },
      UINT32_C(      7766) }
 };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epi16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epi16_mask");
    easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
}

static int
test_easysimd_mm512_cmpneq_epi32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int32_t a[16];
    const int32_t b[16];
    const uint16_t r;
  } test_vec[8] = {
    { { -INT32_C(    43802947),  INT32_C(   140919951), -INT32_C(   583333510),  INT32_C(   409032143), -INT32_C(  1375801937), -INT32_C(   152144409),  INT32_C(  1059685339), -INT32_C(  1082704894),
         INT32_C(  1790761946), -INT32_C(  1720573153), -INT32_C(    76042964), -INT32_C(  1307256830), -INT32_C(  1386278202),  INT32_C(  1671712136), -INT32_C(   761082416), -INT32_C(   460187126) },
      { -INT32_C(  1066331058),  INT32_C(   140919951), -INT32_C(  1375899495),  INT32_C(   409032143),  INT32_C(   365840497),  INT32_C(  1306697387),  INT32_C(    40365168),  INT32_C(   525425873),
         INT32_C(  1790761946), -INT32_C(    96187040), -INT32_C(   542686787),  INT32_C(  1986029317), -INT32_C(  1386278202),  INT32_C(  1671712136),  INT32_C(   725552986), -INT32_C(   460187126) },
      UINT16_C(20213) },
    { { -INT32_C(  1275319725),  INT32_C(   900546680), -INT32_C(  2045422207),  INT32_C(  1828484540),  INT32_C(  1892059044),  INT32_C(  1336766197), -INT32_C(   361042562), -INT32_C(   792214403),
         INT32_C(  1736754159), -INT32_C(  2070138621),  INT32_C(  1124839814), -INT32_C(   877721817), -INT32_C(  2076478065),  INT32_C(  1658055140),  INT32_C(  1397509590),  INT32_C(     2364177) },
      { -INT32_C(  1275319725),  INT32_C(  1756717149),  INT32_C(  1383128227),  INT32_C(  1828484540),  INT32_C(  1456253827),  INT32_C(  1336766197),  INT32_C(   426554613),  INT32_C(   482427573),
         INT32_C(  1736754159), -INT32_C(  2070138621),  INT32_C(  1124839814), -INT32_C(   750930096), -INT32_C(  2076478065), -INT32_C(  1597322069),  INT32_C(  1397509590), -INT32_C(  1806533767) },
      UINT16_C(43222) },
    { {  INT32_C(   286094102), -INT32_C(    72356023), -INT32_C(  1856659648), -INT32_C(  1167814057),  INT32_C(  1177194139), -INT32_C(   605555622), -INT32_C(  1542348501),  INT32_C(   859333660),
         INT32_C(   541345495),  INT32_C(  1964766261), -INT32_C(   368676461), -INT32_C(  1616614653),  INT32_C(  1407569913), -INT32_C(   298857277), -INT32_C(  1970126739),  INT32_C(  2076036004) },
      { -INT32_C(  1208662373), -INT32_C(    72356023), -INT32_C(  1856659648), -INT32_C(  1167814057), -INT32_C(  1468224903), -INT32_C(   605555622), -INT32_C(  1542348501),  INT32_C(   859333660),
         INT32_C(   156969037),  INT32_C(  1964766261), -INT32_C(   368676461), -INT32_C(  1616614653),  INT32_C(  1407569913), -INT32_C(   298857277), -INT32_C(  1970126739),  INT32_C(  2076036004) },
      UINT16_C(  273) },
    { {  INT32_C(  1088929679), -INT32_C(   228216253), -INT32_C(   932470358), -INT32_C(  1822452283), -INT32_C(   872767685),  INT32_C(  2144308424), -INT32_C(  1505885163), -INT32_C(    75193749),
        -INT32_C(  1522832542), -INT32_C(   946364387),  INT32_C(   126812738), -INT32_C(   761598313),  INT32_C(  1453167758),  INT32_C(   919956512), -INT32_C(   589556624),  INT32_C(   416768182) },
      {  INT32_C(  1421076669),  INT32_C(  1045951919), -INT32_C(   932470358),  INT32_C(  1579858625), -INT32_C(   872767685),  INT32_C(  2144308424), -INT32_C(   289623785), -INT32_C(  2029942839),
        -INT32_C(  1522832542), -INT32_C(  1520684375),  INT32_C(   126812738), -INT32_C(   761598313),  INT32_C(   809175983),  INT32_C(   919956512), -INT32_C(   589556624),  INT32_C(   416768182) },
      UINT16_C( 4811) },
    { { -INT32_C(  1419898366),  INT32_C(  1716566853), -INT32_C(  2025288892), -INT32_C(  1612936976), -INT32_C(   607185004),  INT32_C(   859397196),  INT32_C(  1977845413), -INT32_C(   692774188),
         INT32_C(  1904284716),  INT32_C(   332911055),  INT32_C(  1285168988),  INT32_C(  1022129832), -INT32_C(   635978867), -INT32_C(  1139978217), -INT32_C(  1422724905),  INT32_C(   914548490) },
      {  INT32_C(  2144717480),  INT32_C(  2023698909), -INT32_C(  2025288892), -INT32_C(  1708819325),  INT32_C(  1676295510),  INT32_C(   726909433), -INT32_C(   283943571),  INT32_C(  1005719187),
        -INT32_C(  1229207591),  INT32_C(  1999526649),  INT32_C(  1285168988),  INT32_C(  1022129832), -INT32_C(   635978867), -INT32_C(  1139978217), -INT32_C(  1422724905),  INT32_C(   914548490) },
      UINT16_C( 1019) },
    { {  INT32_C(  2031717504), -INT32_C(  2014295834), -INT32_C(    22294171),  INT32_C(   468543959),  INT32_C(  2069380881), -INT32_C(   430784284), -INT32_C(   965011396),  INT32_C(  1361661137),
        -INT32_C(  1094040872), -INT32_C(   280642934),  INT32_C(  1659760779),  INT32_C(  1803410009), -INT32_C(    35203815), -INT32_C(   756860522),  INT32_C(   580411217), -INT32_C(  2072788565) },
      { -INT32_C(  1997042110),  INT32_C(  1803059837), -INT32_C(    22294171),  INT32_C(   468543959),  INT32_C(  2069380881), -INT32_C(   430784284), -INT32_C(   965011396),  INT32_C(  1361661137),
         INT32_C(   144275850), -INT32_C(   280642934),  INT32_C(   541309155),  INT32_C(  1950645015), -INT32_C(   374038455),  INT32_C(   522277710),  INT32_C(   580411217), -INT32_C(  2072788565) },
      UINT16_C(15619) },
    { {  INT32_C(  1504253533),  INT32_C(   188292135), -INT32_C(   332694316),  INT32_C(  1080061943),  INT32_C(  1579750416),  INT32_C(    75319911),  INT32_C(  2070995044),  INT32_C(   572321221),
         INT32_C(   729531651), -INT32_C(  1254705695),  INT32_C(   698442033), -INT32_C(   513211951),  INT32_C(  2084541205),  INT32_C(  1115799005),  INT32_C(  1656615325), -INT32_C(   494544417) },
      {  INT32_C(  1504253533), -INT32_C(   693770444),  INT32_C(  2044163600), -INT32_C(  1777537705), -INT32_C(   363599255),  INT32_C(    75319911),  INT32_C(  2070995044), -INT32_C(  1411817315),
         INT32_C(   729531651), -INT32_C(  1254705695),  INT32_C(   698442033), -INT32_C(   513211951),  INT32_C(  2084541205),  INT32_C(  1115799005),  INT32_C(  1656615325), -INT32_C(   494544417) },
      UINT16_C(  158) },
    { {  INT32_C(  1142935759),  INT32_C(  1841956963),  INT32_C(  2009975241), -INT32_C(   896547628),  INT32_C(  1256952079),  INT32_C(   148502127), -INT32_C(   281132606),  INT32_C(  1358569601),
         INT32_C(  1402280432), -INT32_C(   339714270), -INT32_C(    60584408),  INT32_C(  1640428114), -INT32_C(   223563133),  INT32_C(  1794803112),  INT32_C(  1213806790),  INT32_C(  1217942616) },
      {  INT32_C(  1552715676),  INT32_C(  1841956963),  INT32_C(  2010135985),  INT32_C(  1244222366),  INT32_C(  1256952079), -INT32_C(  1787324868),  INT32_C(  2011812062),  INT32_C(  1358569601),
         INT32_C(  1628516838), -INT32_C(   339714270), -INT32_C(   686944456), -INT32_C(  1490995358), -INT32_C(   223563133),  INT32_C(  1725587080),  INT32_C(  1213806790),  INT32_C(  1217942616) },
      UINT16_C(11629) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epi32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epi32_mask");
    easysimd_assert_equal_mmask16(r, test_vec[i].r);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi32(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i32x16());
    easysimd__mmask16 r = easysimd_mm512_cmpneq_epi32_mask(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_cmpneq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { -INT64_C( 8497209778734285609),  INT64_C( 5650949043324566025), -INT64_C( 3754161847602335797), -INT64_C( 9084479896327253619),
         INT64_C( 6913176432192945837),  INT64_C( 2434705043914695235),  INT64_C( 3961387189883183230),  INT64_C( 6082567215724442941) },
      { -INT64_C( 8497209778734285609),  INT64_C( 5650949043324566025), -INT64_C(  869007026306414064),  INT64_C( 6618959116987245023),
         INT64_C( 6913176432192945837),  INT64_C( 2434705043914695235),  INT64_C( 3961387189883183230),  INT64_C( 7801631251816449450) },
      UINT8_C(140) },
    { { -INT64_C( 8176887451016628207), -INT64_C( 4393476148242326001),  INT64_C( 7928601885898456661),  INT64_C( 5163979733300932904),
         INT64_C( 7194919680706377500),  INT64_C( 5212900022670702299),  INT64_C(  361607236346695726), -INT64_C(  279841236278068063) },
      { -INT64_C( 8176887451016628207), -INT64_C( 4393476148242326001),  INT64_C(  724126714524449322),  INT64_C( 4626884551449325401),
         INT64_C( 7194919680706377500),  INT64_C( 5212900022670702299),  INT64_C(  361607236346695726), -INT64_C(  279841236278068063) },
      UINT8_C( 12) },
    { {  INT64_C( 5296548935444863987), -INT64_C( 4053424800495104417),  INT64_C( 1839631076651054842),  INT64_C( 8044254099010052359),
        -INT64_C( 3781976487088403447), -INT64_C( 8130055515448394042),  INT64_C( 8089519809793407603), -INT64_C( 5473094367616516832) },
      { -INT64_C( 4598893224764947723), -INT64_C( 4053424800495104417), -INT64_C( 1232355934932169204), -INT64_C( 8437892388096382073),
        -INT64_C( 3781976487088403447), -INT64_C( 8130055515448394042), -INT64_C( 2314726681139769820), -INT64_C( 2847691131447055525) },
      UINT8_C(205) },
    { { -INT64_C(  918437200325662363), -INT64_C( 8897121693693317810), -INT64_C( 4221636080717497955),  INT64_C( 4076775620582136269),
         INT64_C(  393494009520498759),  INT64_C( 2476962773484466917), -INT64_C( 4212949261506102423),  INT64_C( 3674125769456111345) },
      { -INT64_C( 7961754275200781370), -INT64_C( 8897121693693317810), -INT64_C(  233184374512193018),  INT64_C( 4076775620582136269),
         INT64_C(  393494009520498759), -INT64_C( 1496922520897906150), -INT64_C( 4212949261506102423),  INT64_C( 3674125769456111345) },
      UINT8_C( 37) },
    { {  INT64_C( 3138578006679774020), -INT64_C(  199006395149810927),  INT64_C(  183659036037453285), -INT64_C( 3779330945365577228),
         INT64_C( 1711030336571280769), -INT64_C( 5761720432914477231),  INT64_C( 5243542729597835478),  INT64_C( 4256250993101020351) },
      {  INT64_C( 8063850296298590410),  INT64_C(  348896128445246812),  INT64_C(  183659036037453285), -INT64_C( 5612610035387552985),
         INT64_C( 1711030336571280769),  INT64_C( 4650761431287552600),  INT64_C(  251619786452324844), -INT64_C( 4885560709026124903) },
      UINT8_C(235) },
    { {  INT64_C( 7095763195958340278), -INT64_C( 2598150881198997710), -INT64_C( 5765247149349026186),  INT64_C( 7533069385080667806),
         INT64_C( 6299131961416477953), -INT64_C( 6384047972059140606),  INT64_C( 8504776146595552788), -INT64_C(  459472748543663738) },
      {  INT64_C( 3317302639250778508), -INT64_C( 2598150881198997710), -INT64_C( 5765247149349026186),  INT64_C( 7533069385080667806),
        -INT64_C( 4330620434903843956),  INT64_C(  390690311263724016),  INT64_C( 8504776146595552788),  INT64_C( 5001376826161649235) },
      UINT8_C(177) },
    { { -INT64_C( 8834622231253746198),  INT64_C(   85373183154784556),  INT64_C( 2192884442080753260),  INT64_C( 6346166085331912428),
        -INT64_C( 5367672237622203367), -INT64_C( 8812677945242629316),  INT64_C(  229981647743772490),  INT64_C( 4529344853253097493) },
      { -INT64_C( 8834622231253746198),  INT64_C( 7216872282198115976),  INT64_C( 2192884442080753260), -INT64_C( 4822148096774479184),
        -INT64_C( 7646465595178070031), -INT64_C( 8812677945242629316), -INT64_C( 6780169303497750491), -INT64_C( 7928153252490226078) },
      UINT8_C(218) },
    { { -INT64_C( 1530524311450329821),  INT64_C( 5275710777541472179),  INT64_C( 8105311760010523417), -INT64_C( 3119409385922578607),
        -INT64_C( 3736011759920374530),  INT64_C( 7529479052665435956), -INT64_C( 2030001301365868728), -INT64_C( 9063647613101874253) },
      { -INT64_C( 1530524311450329821),  INT64_C( 5275710777541472179),  INT64_C(  330001173762495775), -INT64_C( 3119409385922578607),
         INT64_C( 1463942517429964570),  INT64_C( 7529479052665435956),  INT64_C( 2882312420111644842), -INT64_C( 9063647613101874253) },
      UINT8_C( 84) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epi64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epi64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i64x8());
    easysimd__mmask8 r = easysimd_mm512_cmpneq_epi64_mask(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpneq_epi8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask64 k1;
    const int8_t a[64];
    const int8_t b[64];
    const easysimd__mmask64 r;
  } test_vec[] = {
    { UINT64_C(12147147460169056802),
      { -INT8_C(  30), -INT8_C(  44),  INT8_C(  19), -INT8_C(  82),  INT8_C(  61), -INT8_C( 107),  INT8_C(  21),  INT8_C( 110),
        -INT8_C(  22), -INT8_C(  87),  INT8_C(  85),  INT8_C(  79), -INT8_C(  59),  INT8_C(  29),  INT8_C(  54), -INT8_C(   6),
        -INT8_C(  61), -INT8_C( 104),  INT8_C(  24),  INT8_C(  84),      INT8_MAX,  INT8_C(  47), -INT8_C(  75), -INT8_C(  95),
         INT8_C( 113),  INT8_C(  95),  INT8_C(  87), -INT8_C(  77), -INT8_C(  75), -INT8_C(  22),  INT8_C(  91), -INT8_C( 105),
        -INT8_C(  65),  INT8_C( 110),  INT8_C(  69), -INT8_C(   4),  INT8_C(   4),  INT8_C(  91),  INT8_C( 106), -INT8_C(  18),
         INT8_C(   4), -INT8_C(  65),  INT8_C(  61), -INT8_C(  55), -INT8_C(  35),  INT8_C( 115), -INT8_C(  61), -INT8_C(  96),
         INT8_C(  11), -INT8_C(  37), -INT8_C(  12), -INT8_C( 118),  INT8_C(  10), -INT8_C(  87),  INT8_C(  43),  INT8_C( 124),
         INT8_C(   9), -INT8_C( 126),  INT8_C(  47), -INT8_C(  66),  INT8_C( 109), -INT8_C( 118),  INT8_C(  86),  INT8_C(  44) },
      { -INT8_C(  30), -INT8_C( 101),  INT8_C(  40), -INT8_C(  82), -INT8_C(  10), -INT8_C( 110),  INT8_C(  21), -INT8_C(   6),
         INT8_C(  81), -INT8_C(  87),  INT8_C(  85),  INT8_C(  79), -INT8_C(  59),  INT8_C(  29),  INT8_C(  54), -INT8_C(   6),
        -INT8_C(  61), -INT8_C( 104),  INT8_C(  24),  INT8_C( 108),      INT8_MAX,  INT8_C(  91), -INT8_C(  75), -INT8_C(  95),
        -INT8_C(  34),  INT8_C(  23),  INT8_C(  52), -INT8_C(  77), -INT8_C(  75), -INT8_C( 118),  INT8_C( 119), -INT8_C( 102),
         INT8_C(  38), -INT8_C(  97),  INT8_C(  69), -INT8_C(   4),  INT8_C(   4),  INT8_C(  91),  INT8_C( 106), -INT8_C( 126),
        -INT8_C(  86), -INT8_C(  38), -INT8_C(  79),  INT8_C(  69),  INT8_C(  97),      INT8_MIN, -INT8_C(  61), -INT8_C(  96),
         INT8_C(  67), -INT8_C(  37), -INT8_C(  12), -INT8_C(  80),  INT8_C(  10),  INT8_C(  22),  INT8_C(  38),  INT8_C( 124),
         INT8_C(   9),  INT8_C(  91), -INT8_C(  96), -INT8_C(  49),  INT8_C( 109),  INT8_C(  23),  INT8_C(  86),  INT8_C(  44) },
      UINT64_C( 2882609432815468578) },
    { UINT64_C(16014060967083317785),
      {  INT8_C( 122),  INT8_C( 107), -INT8_C( 114), -INT8_C(  15), -INT8_C( 126), -INT8_C(  75),  INT8_C(  70), -INT8_C(  81),
         INT8_C(  16), -INT8_C(  26),  INT8_C( 126), -INT8_C(  11), -INT8_C(   3), -INT8_C(  24),  INT8_C(   1), -INT8_C(  77),
        -INT8_C(  23),  INT8_C(  41), -INT8_C( 102),  INT8_C( 108),  INT8_C( 104),  INT8_C(   4), -INT8_C( 102), -INT8_C( 127),
         INT8_C(  30),  INT8_C(  14), -INT8_C(   4), -INT8_C(  71),  INT8_C( 109),  INT8_C(  57), -INT8_C( 105), -INT8_C(  25),
        -INT8_C(  92),  INT8_C(  37), -INT8_C(  39),  INT8_C(  38), -INT8_C(  38),  INT8_C(  31), -INT8_C(  42), -INT8_C(  22),
         INT8_C(   6),  INT8_C(  84), -INT8_C(  32),  INT8_C(   3),  INT8_C(  60), -INT8_C(  31), -INT8_C(  73),  INT8_C(  37),
         INT8_C(  10),  INT8_C(  81), -INT8_C( 110),  INT8_C( 114),  INT8_C(  85),  INT8_C(  44), -INT8_C(  13),  INT8_C( 116),
         INT8_C(  58), -INT8_C(  17),  INT8_C(  45), -INT8_C(  89),  INT8_C(  40), -INT8_C(  60), -INT8_C( 113), -INT8_C(  51) },
      { -INT8_C(  23),  INT8_C( 104), -INT8_C( 114), -INT8_C(  15), -INT8_C( 121), -INT8_C(  55),  INT8_C(  70), -INT8_C(  81),
         INT8_C(  16), -INT8_C(  26), -INT8_C( 111), -INT8_C(  11),  INT8_C( 111), -INT8_C(  24),  INT8_C(   1), -INT8_C(  77),
        -INT8_C( 103),  INT8_C(  41), -INT8_C( 102), -INT8_C(  17),  INT8_C( 104),  INT8_C(   4),  INT8_C(  99),  INT8_C( 121),
         INT8_C(  30),  INT8_C(  14),  INT8_C(  32), -INT8_C(  71),  INT8_C(  84), -INT8_C(  81), -INT8_C( 105),  INT8_C(  61),
        -INT8_C(  92),  INT8_C(  37),  INT8_C(   1),  INT8_C(  38), -INT8_C(  38),  INT8_C(  31),  INT8_C(  44), -INT8_C(  97),
         INT8_C(   6), -INT8_C(  67), -INT8_C(   7),  INT8_C(   3),  INT8_C(   5),  INT8_C( 121),  INT8_C(  39), -INT8_C(  97),
        -INT8_C( 117),  INT8_C(  19), -INT8_C( 110),  INT8_C( 114), -INT8_C(  14),  INT8_C(  44),  INT8_C(  67),  INT8_C( 116),
         INT8_C(  58),  INT8_C(  99),  INT8_C(  45), -INT8_C(  89),  INT8_C(  40), -INT8_C(  60), -INT8_C( 113), -INT8_C(  51) },
      UINT64_C(  148995371245309969) },
    { UINT64_C( 5861673263053163699),
      { -INT8_C(  21), -INT8_C(  26),  INT8_C(  27), -INT8_C(  35), -INT8_C(  41),  INT8_C(  94), -INT8_C(  99),  INT8_C(  88),
        -INT8_C(  62),  INT8_C(  85),  INT8_C(  45), -INT8_C(  43), -INT8_C(  48),  INT8_C(  63), -INT8_C(   1),  INT8_C(   3),
         INT8_C(  83), -INT8_C(  55), -INT8_C(  73),  INT8_C(  23), -INT8_C(  65),  INT8_C(  10),  INT8_C(  26),  INT8_C( 114),
         INT8_C(  87), -INT8_C(  54),  INT8_C(  43),  INT8_C(  29), -INT8_C(  94), -INT8_C( 125),  INT8_C( 110), -INT8_C( 115),
         INT8_C( 105), -INT8_C( 118),  INT8_C( 106),  INT8_C(  64), -INT8_C(  24),  INT8_C(   8), -INT8_C( 104), -INT8_C(  86),
         INT8_C(  93), -INT8_C(  59),      INT8_MAX,  INT8_C(  45),  INT8_C(   5),      INT8_MAX,  INT8_C(  49),  INT8_C(  88),
         INT8_C(  72), -INT8_C(  24),  INT8_C( 112),  INT8_C(   7), -INT8_C(  13), -INT8_C( 118),  INT8_C( 121),  INT8_C(  74),
         INT8_C(  84), -INT8_C(  91),  INT8_C( 103), -INT8_C(   9),  INT8_C(  40), -INT8_C(  43), -INT8_C( 124), -INT8_C( 110) },
      { -INT8_C(  21), -INT8_C(  17), -INT8_C(  46),  INT8_C(  72), -INT8_C(  41),  INT8_C( 107), -INT8_C(  99),  INT8_C(  88),
        -INT8_C(  62),  INT8_C( 114),  INT8_C(  45),  INT8_C(  53), -INT8_C(  48),  INT8_C(  63), -INT8_C( 114),  INT8_C(   3),
        -INT8_C( 101), -INT8_C(  55),  INT8_C(  64), -INT8_C( 114), -INT8_C( 120),  INT8_C(  10), -INT8_C(  40),  INT8_C( 114),
         INT8_C(  87), -INT8_C(  54), -INT8_C(  45),  INT8_C(  29), -INT8_C(  94), -INT8_C( 125),  INT8_C( 110),  INT8_C( 116),
         INT8_C(  71), -INT8_C( 118),  INT8_C( 106),  INT8_C(  62),  INT8_C(  86), -INT8_C(  82), -INT8_C( 110), -INT8_C( 121),
         INT8_C(  32),  INT8_C(  19), -INT8_C(  68),  INT8_C(  17),  INT8_C(   5),  INT8_C(  74),  INT8_C(  74),  INT8_C(  97),
         INT8_C(  72), -INT8_C( 118),  INT8_C( 112), -INT8_C(  48), -INT8_C(  13), -INT8_C( 118),  INT8_C( 121),  INT8_C(  74),
         INT8_C(   6), -INT8_C(  91),  INT8_C( 103),  INT8_C(  26), -INT8_C(  40), -INT8_C(  43), -INT8_C( 124), -INT8_C( 110) },
      UINT64_C( 1227451627566286882) },
    { UINT64_C( 6569191730018240515),
      {  INT8_C(   0),  INT8_C(  25),  INT8_C(  43),  INT8_C(  68), -INT8_C(  32), -INT8_C(  40), -INT8_C(  26), -INT8_C(  26),
         INT8_C(  89),  INT8_C(  16),  INT8_C(   0),  INT8_C(  49),  INT8_C(  82), -INT8_C( 113),  INT8_C(  81),      INT8_MIN,
        -INT8_C(  39), -INT8_C(  82),  INT8_C(   5), -INT8_C(  46), -INT8_C(  98),  INT8_C(  16), -INT8_C(  20), -INT8_C(  95),
        -INT8_C(  40),  INT8_C(  23),  INT8_C( 106), -INT8_C(  21), -INT8_C( 116), -INT8_C( 108),  INT8_C(  70), -INT8_C( 116),
        -INT8_C(  83),  INT8_C( 113), -INT8_C(  48), -INT8_C( 115),  INT8_C(  74), -INT8_C(  73),  INT8_C( 115), -INT8_C(  93),
        -INT8_C(  57),  INT8_C( 115), -INT8_C(  44),  INT8_C(  25),  INT8_C(   2),  INT8_C(  37), -INT8_C( 102), -INT8_C(  36),
        -INT8_C(  44), -INT8_C(  97), -INT8_C(  82),  INT8_C( 114), -INT8_C(  81), -INT8_C( 102),  INT8_C(  19), -INT8_C( 120),
        -INT8_C(  79),  INT8_C( 125),  INT8_C( 115),  INT8_C(  62),  INT8_C(  17), -INT8_C(  71), -INT8_C(  54), -INT8_C(  66) },
      {  INT8_C(   0), -INT8_C( 101),  INT8_C(  43),  INT8_C(  68), -INT8_C(  32), -INT8_C(  40),  INT8_C(  23),  INT8_C(  25),
         INT8_C(  89), -INT8_C(  20),  INT8_C(  50),  INT8_C(  52),  INT8_C(  82), -INT8_C( 113),  INT8_C(  81), -INT8_C(  27),
        -INT8_C(  39), -INT8_C(  82),  INT8_C(   5), -INT8_C(  46), -INT8_C(  98),  INT8_C( 106), -INT8_C(  93), -INT8_C(  95),
        -INT8_C(  40),  INT8_C(  22),  INT8_C( 106), -INT8_C(  21), -INT8_C( 116), -INT8_C( 108),  INT8_C(  70), -INT8_C(   7),
        -INT8_C(  82),  INT8_C(   1), -INT8_C(  48), -INT8_C( 115), -INT8_C(  65), -INT8_C(  73),  INT8_C( 115), -INT8_C(  93),
         INT8_C( 113),  INT8_C( 115), -INT8_C(  44), -INT8_C( 125),  INT8_C(  24),  INT8_C(  53),  INT8_C( 104), -INT8_C(  36),
        -INT8_C(  13), -INT8_C(  64), -INT8_C(  82),  INT8_C( 114),  INT8_C(  42),  INT8_C(  65),  INT8_C(  87), -INT8_C( 120),
         INT8_C(  87),  INT8_C( 125),  INT8_C(  10),  INT8_C(  62),  INT8_C(  17), -INT8_C(  71), -INT8_C(  54),  INT8_C(  96) },
      UINT64_C(   81752067519055874) },
    { UINT64_C(13862203682525321413),
      {  INT8_C(  48), -INT8_C(   1),  INT8_C(  13),  INT8_C(  91),  INT8_C(  64),  INT8_C( 100),  INT8_C( 109), -INT8_C( 104),
         INT8_C(   3),  INT8_C( 119), -INT8_C(  66), -INT8_C(  74),  INT8_C(  56), -INT8_C(  34),  INT8_C(  22), -INT8_C(   5),
         INT8_C( 108),  INT8_C( 119),  INT8_C( 125),      INT8_MIN, -INT8_C(  16), -INT8_C(  16),  INT8_C(   5), -INT8_C(  75),
        -INT8_C( 120),  INT8_C(  13), -INT8_C( 110),  INT8_C(  85),  INT8_C( 125), -INT8_C(  13),  INT8_C(  21), -INT8_C(  82),
        -INT8_C(  14),  INT8_C(  34),  INT8_C(   9),  INT8_C(  50), -INT8_C( 122),  INT8_C( 118), -INT8_C(  54), -INT8_C( 118),
        -INT8_C(  19), -INT8_C( 119),  INT8_C(  64),  INT8_C(  38),  INT8_C( 103),  INT8_C(  86),  INT8_C(  33), -INT8_C(  44),
        -INT8_C(  51), -INT8_C(  98),  INT8_C(  84), -INT8_C(  66), -INT8_C( 114),  INT8_C(  89),  INT8_C( 115),  INT8_C(  22),
         INT8_C( 102),  INT8_C(   6),  INT8_C( 107), -INT8_C(  29), -INT8_C(   7),      INT8_MIN, -INT8_C( 111), -INT8_C(  21) },
      {  INT8_C(  48), -INT8_C(   1),  INT8_C(  13),  INT8_C(  91),  INT8_C(  64),  INT8_C( 100),  INT8_C( 109), -INT8_C(   2),
         INT8_C(   3), -INT8_C(  13), -INT8_C(  66), -INT8_C(  40),  INT8_C(  74), -INT8_C(  34), -INT8_C(  84),  INT8_C(  23),
         INT8_C( 108),  INT8_C(   0), -INT8_C(  43),  INT8_C( 113), -INT8_C(  16), -INT8_C(  16), -INT8_C( 121), -INT8_C(  65),
         INT8_C(  79),  INT8_C(  13), -INT8_C( 110),  INT8_C(  72),  INT8_C( 114),  INT8_C(  52),  INT8_C(  21),  INT8_C(  21),
        -INT8_C(  49),  INT8_C(  34),  INT8_C(  63), -INT8_C(  33), -INT8_C( 122),  INT8_C( 118), -INT8_C(  35), -INT8_C(  87),
        -INT8_C(  26),  INT8_C(   1),  INT8_C(  64),  INT8_C(  48),  INT8_C( 103),  INT8_C(  46),  INT8_C(  33),  INT8_C(  41),
        -INT8_C(  51),  INT8_C(  29),  INT8_C(  84), -INT8_C(  66), -INT8_C( 114),  INT8_C(  33),  INT8_C(  72),  INT8_C(  22),
         INT8_C(  19), -INT8_C(  21),  INT8_C( 107), -INT8_C(  29),  INT8_C(  31),  INT8_C(  48), -INT8_C( 111), -INT8_C(  18) },
      UINT64_C( 9250429702026860672) },
    { UINT64_C( 4764010246012396717),
      {  INT8_C(  71), -INT8_C(  73), -INT8_C(  54), -INT8_C(  83), -INT8_C(  39),  INT8_C(  18),  INT8_C(  98), -INT8_C(  20),
        -INT8_C(   3),  INT8_C(  95),  INT8_C( 114),  INT8_C(  29), -INT8_C( 113),  INT8_C(  14),  INT8_C(  11),  INT8_C(  16),
        -INT8_C(  24), -INT8_C(  39), -INT8_C(  55), -INT8_C(  75), -INT8_C( 123),  INT8_C(  43),  INT8_C( 104),  INT8_C(  50),
         INT8_C(  16),  INT8_C(  75),  INT8_C(  37),  INT8_C(  35),  INT8_C( 118),  INT8_C(  66),  INT8_C( 101), -INT8_C(  67),
        -INT8_C(   6),  INT8_C(  47),  INT8_C( 107), -INT8_C(  45),  INT8_C(  66), -INT8_C(  51), -INT8_C(  65),  INT8_C(  63),
         INT8_C(  45),  INT8_C(  50),  INT8_C(  92), -INT8_C(  68),  INT8_C(  64),  INT8_C( 104), -INT8_C(  52),  INT8_C(  40),
         INT8_C(  65), -INT8_C( 107), -INT8_C(  34), -INT8_C(  58), -INT8_C(  63),  INT8_C(  70), -INT8_C(   8), -INT8_C(  47),
        -INT8_C( 110),  INT8_C(  30), -INT8_C(  12),  INT8_C(   8),  INT8_C(  96),  INT8_C(  89), -INT8_C(  59),  INT8_C(  90) },
      {  INT8_C(  71), -INT8_C(  73), -INT8_C(  54), -INT8_C(  54), -INT8_C(  39), -INT8_C(  19),  INT8_C(  98), -INT8_C(  20),
        -INT8_C(   3),  INT8_C(  95), -INT8_C(  25),  INT8_C(  29), -INT8_C(  50),  INT8_C(  14), -INT8_C( 121),  INT8_C(  16),
         INT8_C(  73),  INT8_C( 101), -INT8_C(  42),  INT8_C(  10), -INT8_C(  84),  INT8_C(  43), -INT8_C(  37),  INT8_C(  62),
         INT8_C(  16), -INT8_C(  49),  INT8_C(  37),  INT8_C(  77),  INT8_C( 118),  INT8_C(  11), -INT8_C(  88), -INT8_C(  79),
        -INT8_C(   6), -INT8_C(  43),  INT8_C( 107),  INT8_C(  58), -INT8_C(  62), -INT8_C(  51),  INT8_C( 101), -INT8_C(  31),
         INT8_C(  45),  INT8_C(  50),  INT8_C(  64), -INT8_C(  68),  INT8_C(  64),  INT8_C( 104), -INT8_C(  54),  INT8_C(  40),
         INT8_C(  65), -INT8_C( 107),  INT8_C(  84), -INT8_C(  39), -INT8_C(  63),  INT8_C(  70),  INT8_C(  23),  INT8_C(  93),
        -INT8_C( 110),  INT8_C(  93), -INT8_C(  86),  INT8_C(   8),  INT8_C(  96),  INT8_C(  89), -INT8_C(  39), -INT8_C(  91) },
      UINT64_C( 4759178987337630760) },
    { UINT64_C(  205841584321727632),
      { -INT8_C(  20),  INT8_C(  47), -INT8_C(  37),  INT8_C(  92),  INT8_C(  95), -INT8_C(  13), -INT8_C(  71),  INT8_C(  95),
         INT8_C(  80),  INT8_C(  99), -INT8_C( 121), -INT8_C(  71), -INT8_C(  74),  INT8_C(  96),  INT8_C(  94), -INT8_C(  34),
        -INT8_C(  76),  INT8_C(  61), -INT8_C(  56), -INT8_C( 114), -INT8_C( 127), -INT8_C( 108),  INT8_C(  84),  INT8_C(  18),
        -INT8_C(  95), -INT8_C(  43), -INT8_C(  93),  INT8_C( 117),  INT8_C(  32),  INT8_C( 126),  INT8_C( 119),  INT8_C(  12),
        -INT8_C(  83),  INT8_C(  83),  INT8_C( 104),  INT8_C(  13),  INT8_C(  70),  INT8_C(  33),  INT8_C( 108), -INT8_C( 106),
        -INT8_C( 124), -INT8_C(  13),  INT8_C(  80),  INT8_C(  58),  INT8_C(  83), -INT8_C(  82),  INT8_C(  24),  INT8_C(   7),
        -INT8_C(  20), -INT8_C(  31), -INT8_C( 106),  INT8_C( 109),  INT8_C( 117), -INT8_C(  22),      INT8_MAX,  INT8_C(  22),
        -INT8_C(  65),  INT8_C(  34), -INT8_C( 116), -INT8_C(  33), -INT8_C(  96),  INT8_C(   3), -INT8_C(  21),  INT8_C(  78) },
      { -INT8_C(  20),  INT8_C(  83), -INT8_C(  37),  INT8_C(  92),  INT8_C(  95), -INT8_C(  57),  INT8_C(  51),  INT8_C(  95),
         INT8_C(  80),  INT8_C(  99), -INT8_C( 121), -INT8_C(  71),  INT8_C(  49),  INT8_C(  96),  INT8_C(  20), -INT8_C(  34),
         INT8_C(  45), -INT8_C(  86), -INT8_C(  56), -INT8_C(  94), -INT8_C( 107), -INT8_C( 108), -INT8_C(  71),  INT8_C(  18),
        -INT8_C(  95), -INT8_C(  43), -INT8_C(  93),  INT8_C( 117),  INT8_C(  32),  INT8_C(  31),  INT8_C( 119), -INT8_C(  97),
         INT8_C( 115),  INT8_C( 118),  INT8_C(  59), -INT8_C(  25),  INT8_C(  61),  INT8_C( 110),  INT8_C( 108), -INT8_C(   9),
        -INT8_C(  15),  INT8_C(  20),  INT8_C(  80),  INT8_C(  35),  INT8_C(  96),  INT8_C(  25),  INT8_C(  24),  INT8_C(   7),
        -INT8_C(  61), -INT8_C(  53),  INT8_C(  47),  INT8_C( 109), -INT8_C(  42), -INT8_C(  22),      INT8_MAX,  INT8_C(   3),
        -INT8_C(  65),  INT8_C(  34), -INT8_C( 116), -INT8_C(  33),  INT8_C(   0), -INT8_C(  20), -INT8_C(  21),  INT8_C( 115) },
      UINT64_C(   41389554007015424) },
    { UINT64_C( 8663822553725508687),
      { -INT8_C(  33),  INT8_C( 108), -INT8_C(  47), -INT8_C(  75),  INT8_C(  84),  INT8_C( 126), -INT8_C(  72), -INT8_C( 126),
         INT8_C(  95), -INT8_C( 119), -INT8_C(   8),  INT8_C(  95),  INT8_C( 117),  INT8_C(  13), -INT8_C(  45), -INT8_C(  41),
         INT8_C(  93),  INT8_C(  46),  INT8_C( 119),  INT8_C(  28),  INT8_C( 105),  INT8_C(  15), -INT8_C(  51), -INT8_C(  71),
        -INT8_C(  85), -INT8_C(  96),  INT8_C( 104),  INT8_C(  96), -INT8_C(  76), -INT8_C(  91), -INT8_C(  40), -INT8_C( 108),
         INT8_C(  17), -INT8_C(  87),  INT8_C(  73),  INT8_C( 101),  INT8_C(  39),  INT8_C(   2), -INT8_C(  25), -INT8_C( 122),
        -INT8_C( 117), -INT8_C(  33), -INT8_C(  26),  INT8_C(   0), -INT8_C(  20), -INT8_C(  71), -INT8_C(  41),  INT8_C(  74),
        -INT8_C(  25),  INT8_C(  79),  INT8_C( 102),  INT8_C(  80),  INT8_C(  94),  INT8_C(  51),  INT8_C(   9),  INT8_C(   9),
        -INT8_C(  44),  INT8_C( 114),  INT8_C( 105), -INT8_C( 120),  INT8_C(  23),  INT8_C(  65),  INT8_C(  28),  INT8_C(  40) },
      { -INT8_C(  33),  INT8_C( 102), -INT8_C( 115),  INT8_C(  18),  INT8_C(  84),  INT8_C( 126), -INT8_C( 103), -INT8_C( 126),
         INT8_C(  84), -INT8_C( 119), -INT8_C(   8),  INT8_C(  95),  INT8_C( 117),  INT8_C(  13), -INT8_C(  45),  INT8_C(  31),
         INT8_C(  93),  INT8_C(  46),  INT8_C( 111),  INT8_C( 119),  INT8_C( 105),  INT8_C( 121),      INT8_MIN, -INT8_C(   7),
        -INT8_C(  21), -INT8_C(  23),  INT8_C( 104),  INT8_C(   2),  INT8_C(  43), -INT8_C(  98),  INT8_C(  42), -INT8_C( 108),
         INT8_C(  17), -INT8_C(  73),  INT8_C(  73),  INT8_C( 101),  INT8_C(  39),  INT8_C(   2),  INT8_C(  95), -INT8_C( 122),
         INT8_C(  64),  INT8_C(  82), -INT8_C(  26),  INT8_C( 120), -INT8_C(  20), -INT8_C(  71), -INT8_C( 105),  INT8_C(  74),
        -INT8_C(  25),  INT8_C(  79), -INT8_C(  83),  INT8_C(  99),      INT8_MIN,  INT8_C(  46),  INT8_C(  92),  INT8_C( 107),
         INT8_C(  23), -INT8_C(  34),  INT8_C( 105), -INT8_C( 120),  INT8_C(  23),  INT8_C(  65),  INT8_C(  28),      INT8_MIN },
      UINT64_C(   16888499336675406) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 k = test_vec[i].k1;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epi8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epi8_mask");

   easysimd_assert_equal_mmask64(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k1 = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi8(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_i8x64());
    easysimd__mmask64 r = easysimd_mm512_mask_cmpneq_epi8_mask(k1, a, b);

    easysimd_test_x86_write_mmask64(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpneq_epi16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const int16_t a[32];
    const int16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(1647927134),
      {  INT16_C(  2473), -INT16_C( 16029), -INT16_C( 26135), -INT16_C( 21359), -INT16_C( 22902), -INT16_C( 14423),  INT16_C( 12931), -INT16_C( 22421),
         INT16_C( 18562), -INT16_C(  9674),  INT16_C( 21573), -INT16_C(  3502), -INT16_C( 27246), -INT16_C(  4007), -INT16_C( 27916), -INT16_C( 25006),
        -INT16_C( 19045), -INT16_C( 31393), -INT16_C(  3761), -INT16_C(  9935), -INT16_C(  9577),  INT16_C(  6816),  INT16_C(  3084), -INT16_C( 28989),
        -INT16_C(  1708), -INT16_C( 26264), -INT16_C( 17843), -INT16_C(  8309), -INT16_C(  7089),  INT16_C( 17615),  INT16_C(  8566),  INT16_C(  4834) },
      {  INT16_C(  2473),  INT16_C(  9879), -INT16_C( 26135), -INT16_C( 21359), -INT16_C( 22902), -INT16_C( 14423),  INT16_C( 12931), -INT16_C( 22421),
        -INT16_C( 23392), -INT16_C(  9674),  INT16_C(  9311), -INT16_C(  3502), -INT16_C( 27246),  INT16_C( 32754), -INT16_C( 27916), -INT16_C( 27247),
         INT16_C( 10262),  INT16_C( 18619), -INT16_C(  3761), -INT16_C(  9935), -INT16_C(  2470),  INT16_C(  6816),  INT16_C( 31901), -INT16_C( 28989),
        -INT16_C(  1708),  INT16_C( 32555), -INT16_C( 17843), -INT16_C(  8309),  INT16_C(  8342),  INT16_C( 21577), -INT16_C(  9483),  INT16_C(  3050) },
      UINT32_C(1645282562) },
    { UINT32_C(3129238880),
      { -INT16_C( 15269), -INT16_C(  1857), -INT16_C( 15296),  INT16_C( 24630),  INT16_C( 24930),  INT16_C(  9184),  INT16_C(  3674), -INT16_C(  3859),
         INT16_C( 13870),  INT16_C(  9029),  INT16_C( 12048),  INT16_C(  4654), -INT16_C( 32044),  INT16_C( 13316), -INT16_C( 30489),  INT16_C( 17390),
        -INT16_C( 20916), -INT16_C( 29637),  INT16_C( 29042), -INT16_C( 11027), -INT16_C( 12845),  INT16_C( 11768), -INT16_C(  6693),  INT16_C(  2334),
         INT16_C( 25372),  INT16_C( 11309),  INT16_C( 23442),  INT16_C( 26175),  INT16_C( 17373), -INT16_C( 14949), -INT16_C( 30260),  INT16_C(  6152) },
      { -INT16_C( 15269), -INT16_C(  1857), -INT16_C( 27979), -INT16_C( 30594),  INT16_C( 30303),  INT16_C(  9184), -INT16_C( 11428), -INT16_C(  3859),
         INT16_C( 28726), -INT16_C( 14172),  INT16_C( 12048), -INT16_C( 22225), -INT16_C( 32044),  INT16_C( 13316), -INT16_C( 30489), -INT16_C( 29941),
        -INT16_C( 20294),  INT16_C( 28469),  INT16_C( 29042), -INT16_C( 11027), -INT16_C( 21462),  INT16_C( 11768), -INT16_C(  6693),  INT16_C(  2334),
         INT16_C( 25372),  INT16_C( 23423), -INT16_C( 20858), -INT16_C( 21243),  INT16_C( 29560), -INT16_C( 14949), -INT16_C( 30260), -INT16_C( 23466) },
      UINT32_C(2583691584) },
    { UINT32_C(1765804607),
      {  INT16_C(  7094),  INT16_C( 14063), -INT16_C(  4806), -INT16_C( 13587),  INT16_C( 27791),  INT16_C(  5413),  INT16_C( 10778), -INT16_C( 27966),
         INT16_C( 25246), -INT16_C( 30627), -INT16_C( 19442),  INT16_C( 27180),  INT16_C( 16191),  INT16_C( 32264),  INT16_C( 18505), -INT16_C(    25),
        -INT16_C( 10652), -INT16_C( 25034),  INT16_C(  9155),  INT16_C( 21352), -INT16_C( 29041), -INT16_C( 22168),  INT16_C( 11192),  INT16_C( 22075),
        -INT16_C( 26483), -INT16_C( 25634),  INT16_C(  2636), -INT16_C( 29691),  INT16_C(  3657), -INT16_C( 28150), -INT16_C(  3498), -INT16_C( 17774) },
      { -INT16_C( 14136), -INT16_C( 29607), -INT16_C(  4806), -INT16_C( 13587),  INT16_C( 27791),  INT16_C(  5413),  INT16_C( 10778),  INT16_C(    94),
         INT16_C( 15862), -INT16_C( 30627), -INT16_C( 24249),  INT16_C( 27180),  INT16_C( 16191),  INT16_C( 32264), -INT16_C( 18997), -INT16_C( 27456),
         INT16_C(  6525), -INT16_C( 25034), -INT16_C(    38),  INT16_C( 21352), -INT16_C( 29041), -INT16_C( 18126), -INT16_C( 28573),  INT16_C( 22075),
         INT16_C( 21709), -INT16_C( 25634),  INT16_C( 27893), -INT16_C( 23386), -INT16_C( 14011), -INT16_C( 28150),  INT16_C( 27263), -INT16_C( 17774) },
      UINT32_C(1228931075) },
    { UINT32_C( 176637892),
      { -INT16_C( 18099), -INT16_C( 20285),  INT16_C( 31818),  INT16_C(  5898), -INT16_C( 22575), -INT16_C( 14804), -INT16_C( 11757),  INT16_C( 22891),
         INT16_C(  5532),  INT16_C(  7018),  INT16_C(  3967),  INT16_C(   535),  INT16_C( 31956), -INT16_C( 26529), -INT16_C(  6204),  INT16_C(  4514),
         INT16_C( 26272), -INT16_C(  5439), -INT16_C( 13086), -INT16_C( 19710),  INT16_C( 11891), -INT16_C( 30854), -INT16_C(  6911), -INT16_C( 25120),
         INT16_C( 19194),  INT16_C( 31160), -INT16_C( 12455),  INT16_C( 11643), -INT16_C(  9652),  INT16_C(  4293),  INT16_C( 26561),  INT16_C( 25121) },
      { -INT16_C( 18099), -INT16_C( 20404),  INT16_C( 31818),  INT16_C(  8803), -INT16_C( 22575),  INT16_C( 32425), -INT16_C( 11757),  INT16_C( 22891),
         INT16_C(  5532),  INT16_C( 11317), -INT16_C( 20318),  INT16_C(   535),  INT16_C(  7819), -INT16_C( 26529),  INT16_C(  8069),  INT16_C(  4514),
         INT16_C( 26272), -INT16_C(  5439),  INT16_C( 26185), -INT16_C( 19710),  INT16_C( 31556), -INT16_C( 30854), -INT16_C(  6911), -INT16_C( 10301),
        -INT16_C(  1998), -INT16_C( 11005), -INT16_C( 12455),  INT16_C( 11643), -INT16_C( 15750),  INT16_C(   128),  INT16_C( 12257),  INT16_C( 25121) },
      UINT32_C(  42223104) },
    { UINT32_C(   3827388),
      {  INT16_C( 32482), -INT16_C(  6649), -INT16_C( 13602),  INT16_C(  4286), -INT16_C( 15934),  INT16_C( 27621), -INT16_C( 22242), -INT16_C( 26465),
         INT16_C(  8299),  INT16_C( 19608), -INT16_C(  5297),  INT16_C( 31024), -INT16_C( 15295), -INT16_C(   276),  INT16_C(  9770),  INT16_C(  3326),
         INT16_C(  1445), -INT16_C( 31757), -INT16_C( 20017), -INT16_C( 28013),  INT16_C( 31090), -INT16_C( 28419), -INT16_C( 25310), -INT16_C( 29399),
        -INT16_C( 15939),  INT16_C(  3289),  INT16_C(  2477), -INT16_C(  4475),  INT16_C( 29133), -INT16_C(  1812), -INT16_C(  5224),  INT16_C( 15620) },
      { -INT16_C(  2064), -INT16_C( 16192), -INT16_C( 13602),  INT16_C(  4286),  INT16_C( 20428),  INT16_C( 27621), -INT16_C( 22242), -INT16_C( 22149),
         INT16_C(  8299),  INT16_C( 19608), -INT16_C(  5297),  INT16_C( 11313), -INT16_C( 15295), -INT16_C(   276),  INT16_C( 10249),  INT16_C(  3326),
         INT16_C( 16672), -INT16_C( 14151),  INT16_C(  2964),  INT16_C( 25059), -INT16_C( 28837), -INT16_C( 28419), -INT16_C( 13469), -INT16_C(  1551),
        -INT16_C( 23008),  INT16_C( 32316),  INT16_C( 28385), -INT16_C(  4475),  INT16_C( 29133), -INT16_C( 27183), -INT16_C(  5224),  INT16_C( 15620) },
      UINT32_C(   1720464) },
    { UINT32_C(2928198483),
      { -INT16_C( 10158), -INT16_C( 18954), -INT16_C(  6237), -INT16_C( 15441), -INT16_C(  5235),  INT16_C( 28225), -INT16_C(  5031), -INT16_C(  6661),
        -INT16_C( 13126), -INT16_C( 20102),  INT16_C(  2334), -INT16_C( 20024), -INT16_C( 22447), -INT16_C( 23335),  INT16_C( 24939), -INT16_C( 17069),
         INT16_C( 18745), -INT16_C(  9102),  INT16_C(  8496), -INT16_C( 16993), -INT16_C(  7923),  INT16_C( 26156),  INT16_C( 10189), -INT16_C( 30900),
        -INT16_C( 14604),  INT16_C(  4665),  INT16_C(   463),  INT16_C(  8388), -INT16_C( 25175),  INT16_C(  5317),  INT16_C(  6398),  INT16_C( 14545) },
      { -INT16_C( 10158), -INT16_C( 28396), -INT16_C( 19355), -INT16_C( 15441), -INT16_C(  5235),  INT16_C( 25305),  INT16_C(  9634), -INT16_C(  6661),
        -INT16_C( 13126), -INT16_C( 20102),  INT16_C( 27684), -INT16_C( 12837), -INT16_C( 22447),  INT16_C(  2274), -INT16_C( 19528), -INT16_C( 17069),
         INT16_C( 21751),  INT16_C( 23978), -INT16_C(  1784), -INT16_C( 16993), -INT16_C(  7923),  INT16_C(  5631), -INT16_C(  5683), -INT16_C( 18005),
         INT16_C( 21515),  INT16_C( 12148),  INT16_C( 20416), -INT16_C( 13571), -INT16_C( 25175), -INT16_C( 22318),  INT16_C(  4754),  INT16_C( 14545) },
      UINT32_C( 780156994) },
    { UINT32_C(3641488997),
      {  INT16_C(  3167),  INT16_C( 11502), -INT16_C( 25867),  INT16_C(   229),  INT16_C( 23022), -INT16_C( 20944),  INT16_C( 11689), -INT16_C( 26248),
         INT16_C( 18956), -INT16_C( 25023),  INT16_C(   860), -INT16_C( 15576),  INT16_C(  3952), -INT16_C( 10958),  INT16_C( 16070),  INT16_C(  9646),
        -INT16_C( 25270),  INT16_C( 16209),  INT16_C( 14135),  INT16_C(  9536),  INT16_C( 28816),  INT16_C( 14803),  INT16_C( 19613), -INT16_C( 22062),
         INT16_C(  5270), -INT16_C(  3257),  INT16_C( 28695), -INT16_C( 30794), -INT16_C(  6017),  INT16_C( 17757),  INT16_C(  2854),  INT16_C( 29034) },
      { -INT16_C( 17240), -INT16_C(  8272), -INT16_C( 25867),  INT16_C(   229), -INT16_C( 10144), -INT16_C(   579),  INT16_C( 11689), -INT16_C( 26248),
         INT16_C( 18956), -INT16_C( 17491),  INT16_C( 25438), -INT16_C(  8894), -INT16_C( 24757),  INT16_C( 29219), -INT16_C( 29269),  INT16_C(  9646),
        -INT16_C( 25270),  INT16_C( 16209),  INT16_C( 14135), -INT16_C(  6976),  INT16_C( 28816),  INT16_C( 13282), -INT16_C( 30708), -INT16_C( 20242),
        -INT16_C( 25738), -INT16_C( 11157), -INT16_C( 20993), -INT16_C( 30794), -INT16_C( 10931), -INT16_C(  1860),  INT16_C(  2854), -INT16_C( 21429) },
      UINT32_C(2433234465) },
    { UINT32_C(3315312822),
      {  INT16_C( 32037),  INT16_C( 13049), -INT16_C(  6394),  INT16_C( 31970),  INT16_C( 19842), -INT16_C( 32431),  INT16_C(  1018),  INT16_C( 18380),
        -INT16_C( 30504),  INT16_C( 14911), -INT16_C( 29912),  INT16_C( 23526), -INT16_C( 12535), -INT16_C( 16622), -INT16_C( 21129), -INT16_C( 25211),
         INT16_C( 32299),  INT16_C( 12751), -INT16_C( 20123), -INT16_C(  6227), -INT16_C(   258), -INT16_C(  1943),  INT16_C( 13569), -INT16_C(  9920),
         INT16_C( 32701), -INT16_C(  6892), -INT16_C(  1526),  INT16_C(  5184),  INT16_C( 21193),  INT16_C( 16851),  INT16_C( 22528),  INT16_C( 11230) },
      { -INT16_C( 21034),  INT16_C( 13049),  INT16_C(  2398),  INT16_C( 31970), -INT16_C( 29688),  INT16_C(  2388), -INT16_C( 27455),  INT16_C( 32483),
        -INT16_C(  2284),  INT16_C( 14911), -INT16_C( 29912), -INT16_C( 17614),  INT16_C(  1783), -INT16_C( 16622), -INT16_C( 21129),  INT16_C( 13602),
         INT16_C( 32391),  INT16_C( 12751), -INT16_C( 20123), -INT16_C(  6227), -INT16_C(   258), -INT16_C(  8039),  INT16_C( 31786), -INT16_C(  9920),
        -INT16_C( 15501), -INT16_C(  6892), -INT16_C( 28825),  INT16_C( 24095),  INT16_C(  7061), -INT16_C(  3243),  INT16_C( 30709),  INT16_C( 11230) },
      UINT32_C(1157728436) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 k = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epi16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epi16_mask");

   easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi16(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_i16x32());
    easysimd__mmask32 r = easysimd_mm512_mask_cmpneq_epi16_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpneq_epi32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k1;
    const int32_t a[16];
    const int32_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { UINT16_C(20505),
      {  INT32_C(   799125502), -INT32_C(  1445766003),  INT32_C(   641164575),  INT32_C(   312890129),  INT32_C(  1297726825), -INT32_C(  1506342336), -INT32_C(  1786702861), -INT32_C(  1713005669),
        -INT32_C(   657881270),  INT32_C(    58825955),  INT32_C(   321501442),  INT32_C(  1982189324), -INT32_C(   842826099),  INT32_C(  2088041352), -INT32_C(  1458440946), -INT32_C(   616302704) },
      {  INT32_C(   799125502), -INT32_C(  1445766003), -INT32_C(    88240403), -INT32_C(   361700771),  INT32_C(  1297726825), -INT32_C(  1506342336), -INT32_C(  1786702861), -INT32_C(  1713005669),
        -INT32_C(   582139851),  INT32_C(  1653125492),  INT32_C(   321501442),  INT32_C(  1982189324),  INT32_C(   779173887), -INT32_C(  1922378133), -INT32_C(  1170255795), -INT32_C(  1364800391) },
      UINT16_C(20488) },
    { UINT16_C(32908),
      { -INT32_C(   454945628), -INT32_C(  2122236327),  INT32_C(   168429578), -INT32_C(  1288144056), -INT32_C(  1623154094),  INT32_C(  1817804787),  INT32_C(  1461452619), -INT32_C(  1747408910),
         INT32_C(   343653051),  INT32_C(    60226809),  INT32_C(  1275961348),  INT32_C(  1828734235), -INT32_C(   619953944),  INT32_C(   189228736),  INT32_C(  1482908261), -INT32_C(   990954743) },
      { -INT32_C(   454945628),  INT32_C(  1794273126),  INT32_C(   716701455), -INT32_C(  1288144056), -INT32_C(  1623154094),  INT32_C(  1817804787), -INT32_C(  1144510798), -INT32_C(  1747408910),
         INT32_C(   343653051),  INT32_C(    60226809),  INT32_C(  2063706166),  INT32_C(  1828734235),  INT32_C(  1208922174), -INT32_C(  1296510720),  INT32_C(  1483571447),  INT32_C(  1487924535) },
      UINT16_C(32772) },
    { UINT16_C( 3807),
      {  INT32_C(  1642451243), -INT32_C(   639768711), -INT32_C(  1104116352),  INT32_C(   906444342),  INT32_C(   132693776),  INT32_C(  2002801984), -INT32_C(  1982853309),  INT32_C(   781758211),
        -INT32_C(    91259520), -INT32_C(   439128988),  INT32_C(   681772018),  INT32_C(  1365158465), -INT32_C(  1436989846), -INT32_C(   568215141), -INT32_C(   882314808),  INT32_C(   569966753) },
      { -INT32_C(   501512067), -INT32_C(   406327563),  INT32_C(   856648433),  INT32_C(  2139385109),  INT32_C(  1311366579),  INT32_C(  2002801984), -INT32_C(   584477380),  INT32_C(   781758211),
        -INT32_C(    91259520), -INT32_C(    91768056),  INT32_C(   681772018),  INT32_C(  1365158465), -INT32_C(  1436989846),  INT32_C(  1770271021),  INT32_C(  1531358662),  INT32_C(   569966753) },
      UINT16_C(  607) },
    { UINT16_C(26395),
      {  INT32_C(  1147249182),  INT32_C(   998215224), -INT32_C(   806274752),  INT32_C(  1240790300), -INT32_C(   307005401), -INT32_C(   162924250), -INT32_C(  1653491906),  INT32_C(   923110425),
         INT32_C(  1719363118),  INT32_C(   899807989),  INT32_C(  1325699635), -INT32_C(  1365641081), -INT32_C(  1751364495), -INT32_C(  2071075515), -INT32_C(  1256063332), -INT32_C(  1192417654) },
      { -INT32_C(  2111936371), -INT32_C(  1766342813), -INT32_C(   655967407),  INT32_C(   612859827),  INT32_C(   280699851), -INT32_C(  1533786104), -INT32_C(   799427002),  INT32_C(   923110425),
         INT32_C(   317433519),  INT32_C(   899807989),  INT32_C(   294620765), -INT32_C(  1365641081), -INT32_C(  1751364495), -INT32_C(  2132443847),  INT32_C(   273694259), -INT32_C(  1192417654) },
      UINT16_C(25883) },
    { UINT16_C(58694),
      {  INT32_C(  1704717831), -INT32_C(  1971967363),  INT32_C(  2036575040), -INT32_C(   692433764), -INT32_C(    44654647),  INT32_C(  1661839070), -INT32_C(     6846849), -INT32_C(   203104789),
         INT32_C(  1213759435), -INT32_C(   355217750),  INT32_C(   375666297),  INT32_C(  1273765506), -INT32_C(  1672985922),  INT32_C(  1761564136), -INT32_C(   949512484),  INT32_C(  1052396403) },
      {  INT32_C(  1955009226),  INT32_C(  1516198624),  INT32_C(   309379984), -INT32_C(   692433764), -INT32_C(  2042911074),  INT32_C(  1661839070), -INT32_C(     6846849),  INT32_C(  1803704736),
         INT32_C(  1213759435), -INT32_C(   207208861), -INT32_C(   486205951),  INT32_C(  1273765506), -INT32_C(  1672985922),  INT32_C(  1761564136),  INT32_C(  2064415451),  INT32_C(  1088851668) },
      UINT16_C(50182) },
    { UINT16_C(64140),
      {  INT32_C(   116208388), -INT32_C(  1125518775),  INT32_C(  1573677909),  INT32_C(   576807721), -INT32_C(   991216151), -INT32_C(  1958742089),  INT32_C(   499852934), -INT32_C(   266839828),
        -INT32_C(   722074229),  INT32_C(  1301340152),  INT32_C(  1957322059), -INT32_C(  1449784128),  INT32_C(     7176521), -INT32_C(     7557767), -INT32_C(  1088661549),  INT32_C(   984560815) },
      {  INT32_C(   839886394), -INT32_C(  1125518775), -INT32_C(  1119540740),  INT32_C(   576807721), -INT32_C(   696265635), -INT32_C(  1958742089),  INT32_C(   499852934),  INT32_C(  1615643174),
        -INT32_C(   722074229),  INT32_C(  1301340152),  INT32_C(  1957322059), -INT32_C(  1449784128),  INT32_C(  1880255983), -INT32_C(   607852680), -INT32_C(    17967144),  INT32_C(    39729561) },
      UINT16_C(61572) },
    { UINT16_C(37104),
      {  INT32_C(  1015524865), -INT32_C(  1850983246),  INT32_C(  1187879511), -INT32_C(  2135499000), -INT32_C(  1638171962), -INT32_C(   358791087),  INT32_C(   401406850), -INT32_C(   307766036),
         INT32_C(  1026109067), -INT32_C(  1043409558),  INT32_C(  2013764464),  INT32_C(  1090043258), -INT32_C(  1998695369),  INT32_C(   510884764),  INT32_C(  1664442231), -INT32_C(   951002052) },
      {  INT32_C(  1963227659), -INT32_C(  1850983246),  INT32_C(  1187879511), -INT32_C(  2135499000), -INT32_C(  1638171962), -INT32_C(   130076799), -INT32_C(   899976050),  INT32_C(  1553050449),
         INT32_C(  1026109067), -INT32_C(   734787481),  INT32_C(  2013764464),  INT32_C(  1090043258),  INT32_C(  1765815016), -INT32_C(   262045855),  INT32_C(  1169865972), -INT32_C(   951002052) },
      UINT16_C( 4320) },
    { UINT16_C(18690),
      { -INT32_C(  1038272901),  INT32_C(  1040570786), -INT32_C(  1554089029),  INT32_C(   252485550),  INT32_C(   318729502), -INT32_C(  1856521687), -INT32_C(   417400826), -INT32_C(   433053589),
        -INT32_C(   106410410), -INT32_C(   499667673), -INT32_C(   695822808),  INT32_C(  1407619637),  INT32_C(   694609152), -INT32_C(  1514488417),  INT32_C(   562944182),  INT32_C(  1325907448) },
      {  INT32_C(   843624203), -INT32_C(  2078965924),  INT32_C(  1247517461),  INT32_C(   765346093), -INT32_C(   967375834), -INT32_C(  1856521687), -INT32_C(   493291286), -INT32_C(  1053712458),
        -INT32_C(  1426818738),  INT32_C(   237963769), -INT32_C(   782660956),  INT32_C(  1407619637),  INT32_C(   694609152), -INT32_C(  1514488417),  INT32_C(   562944182), -INT32_C(  1164745364) },
      UINT16_C(  258) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k1;
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epi32_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epi32_mask");

   easysimd_assert_equal_mmask16(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi32(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_i32x16());
    easysimd__mmask16 r = easysimd_mm512_mask_cmpneq_epi32_mask(k1, a, b);

    easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpneq_epi64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const int64_t a[8];
    const int64_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(175),
      { -INT64_C( 1397739797383760435),  INT64_C( 7148414129711131377), -INT64_C( 8253410687176690974), -INT64_C( 3959805381162813169),
        -INT64_C( 4926457734651801690), -INT64_C( 7455467607766372002),  INT64_C(   58665479127298958), -INT64_C( 8977529081098867548) },
      { -INT64_C( 7602963233125137417), -INT64_C( 8243482880400219061), -INT64_C( 7122265818711159242), -INT64_C( 2022494548806237043),
        -INT64_C( 4926457734651801690), -INT64_C( 7455467607766372002), -INT64_C( 3455813135014518157), -INT64_C( 8977529081098867548) },
      UINT8_C( 15) },
    { UINT8_C(117),
      { -INT64_C( 1861715514496757892), -INT64_C( 6803744632543174624), -INT64_C( 2090198371877857365), -INT64_C( 9055796539698778868),
        -INT64_C( 1934260008351438424), -INT64_C( 2508668602549800128),  INT64_C( 8704492076778343747), -INT64_C( 2808089816781629570) },
      { -INT64_C( 1861715514496757892), -INT64_C( 6803744632543174624), -INT64_C( 1563178512302679447), -INT64_C( 2352718155573386160),
        -INT64_C( 1934260008351438424), -INT64_C( 2508668602549800128),  INT64_C( 8704492076778343747),  INT64_C( 3340852579729817904) },
      UINT8_C(  4) },
    { UINT8_C(148),
      { -INT64_C( 8026384070197426748),  INT64_C( 4886876209695267988), -INT64_C( 4997024177487561068),  INT64_C( 5940070401191272574),
        -INT64_C( 7392192570282183005), -INT64_C( 2626223898689627580), -INT64_C( 2649268467479257723),  INT64_C( 1020387935686888776) },
      { -INT64_C( 8026384070197426748),  INT64_C( 9011414744529907948), -INT64_C( 4997024177487561068),  INT64_C( 5940070401191272574),
        -INT64_C( 7392192570282183005),  INT64_C( 8180550245228814867), -INT64_C( 2649268467479257723),  INT64_C( 1020387935686888776) },
      UINT8_C(  0) },
    { UINT8_C( 19),
      {  INT64_C( 1956823942387080271), -INT64_C( 2728696809076407533), -INT64_C( 6838848076161031835),  INT64_C( 2768373291526266092),
         INT64_C( 9040224642772263256), -INT64_C( 3087717344335054953),  INT64_C( 2928324078436135962),  INT64_C( 6024385812961935004) },
      { -INT64_C( 5678156107559877045), -INT64_C( 6980820742493549606), -INT64_C( 6838848076161031835),  INT64_C( 2768373291526266092),
         INT64_C( 9040224642772263256), -INT64_C( 3087717344335054953),  INT64_C( 2928324078436135962), -INT64_C( 3322102520163816629) },
      UINT8_C(  3) },
    { UINT8_C(141),
      { -INT64_C( 1737669681345944223), -INT64_C( 3066326258757646790),  INT64_C( 1038836018111095850), -INT64_C(  642862911519996510),
         INT64_C( 4919193076957347004), -INT64_C( 1001422235786178366), -INT64_C( 3474214654808260896),  INT64_C( 3418349161346915554) },
      {  INT64_C( 5203294222493158118),  INT64_C( 1881281137587650400), -INT64_C( 3809307609847706331), -INT64_C(  642862911519996510),
         INT64_C( 4919193076957347004), -INT64_C( 1926240192440362513), -INT64_C( 5683444085798004373), -INT64_C( 2990745188711832366) },
      UINT8_C(133) },
    { UINT8_C( 57),
      { -INT64_C( 4557897649110785766), -INT64_C( 7161063824389406911),  INT64_C( 2784569303573550367), -INT64_C( 5965920869424055172),
        -INT64_C( 5284952765589332564), -INT64_C( 1111304087699454061), -INT64_C( 3648527781623527120), -INT64_C( 5201380444837756528) },
      { -INT64_C( 3781818378262404007), -INT64_C( 3814453660113256369), -INT64_C( 6630320015398518561), -INT64_C( 5965920869424055172),
        -INT64_C( 5284952765589332564), -INT64_C( 5290941385065856724), -INT64_C( 3648527781623527120), -INT64_C( 2557864124973282650) },
      UINT8_C( 33) },
    { UINT8_C(104),
      {  INT64_C( 5408542294071899088), -INT64_C( 9011944075400059192), -INT64_C( 5147321392766327996),  INT64_C( 6870808098288480881),
        -INT64_C(  661469293972738313),  INT64_C( 5765240895424293091), -INT64_C( 2138390203212470749), -INT64_C( 6230115791739244900) },
      {  INT64_C( 6377379047867905884),  INT64_C( 3838714032372110056),  INT64_C( 2563149104109559104), -INT64_C( 1140579948255353136),
         INT64_C( 4836017566662744655),  INT64_C( 5765240895424293091), -INT64_C( 2898821696200973285),  INT64_C( 4196376198447174583) },
      UINT8_C( 72) },
    { UINT8_C(183),
      { -INT64_C( 5400236528851797250), -INT64_C( 6243310789018330015), -INT64_C( 7870126966694059537), -INT64_C( 1445102506666904835),
        -INT64_C( 8619304580489501392), -INT64_C( 3277395797930502671),  INT64_C( 3280113875760348859), -INT64_C( 4334334674489336832) },
      { -INT64_C( 1968965895825793026), -INT64_C( 4675506052899230872), -INT64_C(  999736505449339965), -INT64_C( 2365593900986431275),
        -INT64_C( 8619304580489501392), -INT64_C( 3277395797930502671),  INT64_C( 3280113875760348859), -INT64_C( 4334334674489336832) },
      UINT8_C(  7) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epi64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epi64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_i64x8());
    easysimd__mmask8 r = easysimd_mm512_mask_cmpneq_epi64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpneq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t a[64];
    const uint8_t b[64];
    const easysimd__mmask64 r;
  } test_vec[] = {
    { { UINT8_C(164), UINT8_C(171), UINT8_C(102), UINT8_C( 69), UINT8_C( 78), UINT8_C(122), UINT8_C(189), UINT8_C(202),
        UINT8_C( 45), UINT8_C( 49), UINT8_C(  0), UINT8_C(156), UINT8_C(198), UINT8_C( 72), UINT8_C( 60), UINT8_C(155),
        UINT8_C(242), UINT8_C( 28), UINT8_C(253), UINT8_C(  7), UINT8_C(196), UINT8_C(150), UINT8_C(248), UINT8_C(238),
        UINT8_C( 20), UINT8_C( 89), UINT8_C(241), UINT8_C(120), UINT8_C( 72), UINT8_C(137), UINT8_C(215), UINT8_C(236),
        UINT8_C( 53), UINT8_C( 61), UINT8_C( 50), UINT8_C(131), UINT8_C(183), UINT8_C(239), UINT8_C( 77), UINT8_C(229),
        UINT8_C( 32), UINT8_C( 78), UINT8_C(129), UINT8_C(230), UINT8_C(150), UINT8_C(189), UINT8_C(129), UINT8_C(136),
        UINT8_C(218), UINT8_C(127), UINT8_C(144), UINT8_C(158), UINT8_C( 21), UINT8_C(136), UINT8_C(140), UINT8_C( 41),
        UINT8_C(225), UINT8_C(125), UINT8_C(161), UINT8_C( 41), UINT8_C(  7), UINT8_C(121), UINT8_C( 22), UINT8_C( 60) },
      { UINT8_C(182), UINT8_C(171), UINT8_C(191), UINT8_C(110), UINT8_C( 55), UINT8_C(122), UINT8_C(189), UINT8_C(202),
        UINT8_C( 90), UINT8_C(212), UINT8_C( 61), UINT8_C(240), UINT8_C(198), UINT8_C( 72), UINT8_C(121), UINT8_C(155),
        UINT8_C( 61), UINT8_C(  9), UINT8_C( 10), UINT8_C( 83), UINT8_C(196), UINT8_C(150), UINT8_C(124), UINT8_C(238),
        UINT8_C( 20), UINT8_C( 89), UINT8_C(241), UINT8_C(120), UINT8_C( 72), UINT8_C(137), UINT8_C( 87), UINT8_C(236),
        UINT8_C(250), UINT8_C( 61), UINT8_C(187), UINT8_C( 49), UINT8_C( 34), UINT8_C(239), UINT8_C(136), UINT8_C(229),
        UINT8_C(227), UINT8_C(197), UINT8_C(109), UINT8_C(230), UINT8_C(132), UINT8_C(230), UINT8_C(225), UINT8_C(193),
        UINT8_C(239), UINT8_C(127), UINT8_C( 20), UINT8_C(129), UINT8_C(129), UINT8_C(145), UINT8_C(140), UINT8_C(149),
        UINT8_C(225), UINT8_C(125), UINT8_C(161), UINT8_C( 41), UINT8_C(  7), UINT8_C(121), UINT8_C(147), UINT8_C( 60) },
      UINT64_C( 4665156768908660509) },
    { { UINT8_C(189), UINT8_C( 42), UINT8_C(181), UINT8_C( 65), UINT8_C( 17), UINT8_C(150), UINT8_C(  2), UINT8_C(  0),
        UINT8_C(129), UINT8_C( 23), UINT8_C(129), UINT8_C(  3), UINT8_C(168), UINT8_C(117), UINT8_C(152), UINT8_C( 87),
        UINT8_C(  6), UINT8_C( 73), UINT8_C(157), UINT8_C( 73), UINT8_C( 80), UINT8_C( 48), UINT8_C(134), UINT8_C(110),
        UINT8_C(127), UINT8_C(245), UINT8_C(174), UINT8_C(221), UINT8_C(237), UINT8_C(107), UINT8_C( 29), UINT8_C(170),
        UINT8_C(149), UINT8_C(211), UINT8_C(235), UINT8_C(166), UINT8_C(105), UINT8_C(237), UINT8_C(167), UINT8_C(235),
        UINT8_C(  4), UINT8_C( 40), UINT8_C(238), UINT8_C(172), UINT8_C(158), UINT8_C(134), UINT8_C(  3), UINT8_C(164),
        UINT8_C(207), UINT8_C(160), UINT8_C(237), UINT8_C( 32), UINT8_C(209), UINT8_C(115), UINT8_C(142), UINT8_C( 80),
        UINT8_C(105), UINT8_C( 60), UINT8_C( 45), UINT8_C( 86), UINT8_C(167), UINT8_C( 75), UINT8_C(  0), UINT8_C( 60) },
      { UINT8_C( 30), UINT8_C(235), UINT8_C(181), UINT8_C(135), UINT8_C( 17), UINT8_C(138), UINT8_C(114), UINT8_C(221),
        UINT8_C(129), UINT8_C( 23), UINT8_C(137), UINT8_C( 80), UINT8_C(231), UINT8_C(117), UINT8_C(244), UINT8_C(182),
        UINT8_C( 45), UINT8_C(225), UINT8_C(157), UINT8_C(254), UINT8_C( 85), UINT8_C( 48), UINT8_C( 79), UINT8_C(190),
        UINT8_C(127), UINT8_C(245), UINT8_C( 20), UINT8_C( 71), UINT8_C(199), UINT8_C(107), UINT8_C(132), UINT8_C(229),
        UINT8_C(149), UINT8_C(211), UINT8_C(109), UINT8_C(215), UINT8_C(241), UINT8_C(223), UINT8_C(167), UINT8_C(163),
        UINT8_C(  4), UINT8_C( 40), UINT8_C(238), UINT8_C(172), UINT8_C(203), UINT8_C(134), UINT8_C(  3), UINT8_C(248),
        UINT8_C(207), UINT8_C(160), UINT8_C(237), UINT8_C( 32), UINT8_C(209), UINT8_C(115), UINT8_C(142), UINT8_C(185),
        UINT8_C(105), UINT8_C( 60), UINT8_C(  0), UINT8_C(138), UINT8_C(  5), UINT8_C(132), UINT8_C(111), UINT8_C(  4) },
      UINT64_C(18194701635410451691) },
    { { UINT8_C(206), UINT8_C(116), UINT8_C( 35), UINT8_C(153), UINT8_C( 92), UINT8_C(  0), UINT8_C(145), UINT8_C( 38),
        UINT8_C(180), UINT8_C(136), UINT8_C( 69), UINT8_C(205), UINT8_C(206), UINT8_C( 34), UINT8_C(134), UINT8_C(145),
        UINT8_C( 19), UINT8_C(134), UINT8_C( 27), UINT8_C( 24), UINT8_C( 11), UINT8_C(138), UINT8_C( 28), UINT8_C(246),
        UINT8_C(103), UINT8_C(248), UINT8_C(211), UINT8_C( 35), UINT8_C(136), UINT8_C( 83), UINT8_C( 31), UINT8_C( 86),
        UINT8_C(199), UINT8_C( 66), UINT8_C(239), UINT8_C( 35), UINT8_C( 66), UINT8_C(128), UINT8_C( 74), UINT8_C(247),
        UINT8_C(  9), UINT8_C(143), UINT8_C(196), UINT8_C(215), UINT8_C(178), UINT8_C( 74), UINT8_C(104), UINT8_C(197),
        UINT8_C(208), UINT8_C(131), UINT8_C(222), UINT8_C(219), UINT8_C( 14), UINT8_C(250), UINT8_C(210), UINT8_C(117),
        UINT8_C(242), UINT8_C(165), UINT8_C(152), UINT8_C(122), UINT8_C(248), UINT8_C(183), UINT8_C(208), UINT8_C(191) },
      { UINT8_C(249), UINT8_C(191), UINT8_C(226), UINT8_C( 59), UINT8_C( 92), UINT8_C(  0), UINT8_C(145), UINT8_C( 38),
        UINT8_C(180), UINT8_C(246), UINT8_C( 69), UINT8_C(205), UINT8_C(206), UINT8_C(137), UINT8_C( 51), UINT8_C( 17),
        UINT8_C( 12), UINT8_C( 17), UINT8_C(236), UINT8_C( 24), UINT8_C( 12), UINT8_C(190), UINT8_C( 28), UINT8_C(254),
        UINT8_C(103), UINT8_C(248), UINT8_C(211), UINT8_C( 91), UINT8_C(222), UINT8_C( 73), UINT8_C( 31), UINT8_C( 86),
        UINT8_C(199), UINT8_C( 66), UINT8_C(239), UINT8_C( 73), UINT8_C( 66), UINT8_C( 69), UINT8_C( 74), UINT8_C(247),
        UINT8_C( 60), UINT8_C(143), UINT8_C(196), UINT8_C(124), UINT8_C(178), UINT8_C( 74), UINT8_C(141), UINT8_C(197),
        UINT8_C(208), UINT8_C(122), UINT8_C( 98), UINT8_C(164), UINT8_C( 56), UINT8_C(250), UINT8_C(210), UINT8_C(117),
        UINT8_C(242), UINT8_C(165), UINT8_C(247), UINT8_C(122), UINT8_C(248), UINT8_C( 18), UINT8_C(207), UINT8_C(191) },
      UINT64_C( 7214284090193207823) },
    { { UINT8_C(252), UINT8_C(113), UINT8_C(224), UINT8_C( 55), UINT8_C(248), UINT8_C(110), UINT8_C(127), UINT8_C(145),
        UINT8_C(232), UINT8_C(226), UINT8_C( 53), UINT8_C( 32), UINT8_C(212), UINT8_C(216), UINT8_C(188), UINT8_C(237),
        UINT8_C(244), UINT8_C(180), UINT8_C(229), UINT8_C( 90), UINT8_C(198), UINT8_C(181), UINT8_C(200), UINT8_C(213),
        UINT8_C(151), UINT8_C(128), UINT8_C( 13), UINT8_C(191), UINT8_C(201), UINT8_C( 43), UINT8_C( 35), UINT8_C(197),
        UINT8_C(157), UINT8_C(  4), UINT8_C(253), UINT8_C(149), UINT8_C(114), UINT8_C(124), UINT8_C( 38), UINT8_C( 90),
        UINT8_C( 94), UINT8_C( 92), UINT8_C(122), UINT8_C( 50), UINT8_C( 52), UINT8_C( 55), UINT8_C( 32), UINT8_C( 41),
        UINT8_C(235), UINT8_C(  5), UINT8_C(131), UINT8_C(177), UINT8_C(186), UINT8_C( 75), UINT8_C(134), UINT8_C( 82),
        UINT8_C(203), UINT8_C(147), UINT8_C( 17), UINT8_C(149), UINT8_C(191), UINT8_C( 53), UINT8_C( 90), UINT8_C( 92) },
      { UINT8_C( 57), UINT8_C( 87), UINT8_C(241), UINT8_C(171), UINT8_C(212), UINT8_C(110), UINT8_C(  5), UINT8_C(145),
        UINT8_C(116), UINT8_C(127), UINT8_C(101), UINT8_C( 32), UINT8_C(212), UINT8_C(133), UINT8_C(209), UINT8_C(237),
        UINT8_C(138), UINT8_C( 84), UINT8_C(229), UINT8_C( 90), UINT8_C(160), UINT8_C(216), UINT8_C(151), UINT8_C(213),
        UINT8_C(151), UINT8_C(128), UINT8_C( 13), UINT8_C(191), UINT8_C(221), UINT8_C( 91), UINT8_C(135), UINT8_C( 22),
        UINT8_C(157), UINT8_C(120), UINT8_C(193), UINT8_C(134), UINT8_C(114), UINT8_C(198), UINT8_C( 38), UINT8_C( 90),
        UINT8_C( 94), UINT8_C( 92), UINT8_C(173), UINT8_C(252), UINT8_C( 52), UINT8_C(126), UINT8_C( 32), UINT8_C( 41),
        UINT8_C(235), UINT8_C(  5), UINT8_C(114), UINT8_C(177), UINT8_C(201), UINT8_C(  9), UINT8_C(222), UINT8_C( 53),
        UINT8_C(203), UINT8_C(147), UINT8_C( 96), UINT8_C(149), UINT8_C( 58), UINT8_C(231), UINT8_C(166), UINT8_C( 92) },
      UINT64_C( 8427409382831253343) },
    { { UINT8_C( 74), UINT8_C(161), UINT8_C(112), UINT8_C(237), UINT8_C( 32), UINT8_C( 14), UINT8_C( 26), UINT8_C(243),
           UINT8_MAX, UINT8_C(141), UINT8_C(102), UINT8_C(200), UINT8_C(150), UINT8_C( 68), UINT8_C(253), UINT8_C( 72),
        UINT8_C( 35), UINT8_C( 93), UINT8_C(216), UINT8_C( 93), UINT8_C( 68), UINT8_C(126), UINT8_C( 74), UINT8_C(163),
        UINT8_C(229), UINT8_C(189), UINT8_C(147), UINT8_C( 19), UINT8_C(233), UINT8_C(136), UINT8_C(135), UINT8_C( 51),
        UINT8_C( 41), UINT8_C(248), UINT8_C( 32), UINT8_C( 73), UINT8_C(  6), UINT8_C( 58), UINT8_C( 60), UINT8_C(  5),
        UINT8_C(199), UINT8_C(162), UINT8_C(205), UINT8_C( 94), UINT8_C(231), UINT8_C(202), UINT8_C(166), UINT8_C( 10),
        UINT8_C( 39), UINT8_C(126), UINT8_C(104), UINT8_C(107), UINT8_C(252), UINT8_C(178), UINT8_C( 15), UINT8_C(226),
        UINT8_C(111), UINT8_C(162), UINT8_C(245), UINT8_C( 88), UINT8_C( 42), UINT8_C(125), UINT8_C(139), UINT8_C( 84) },
      { UINT8_C( 74), UINT8_C(161), UINT8_C(157), UINT8_C(123), UINT8_C(229), UINT8_C(218), UINT8_C( 26), UINT8_C(243),
        UINT8_C(124), UINT8_C( 78), UINT8_C(102), UINT8_C( 99), UINT8_C(150), UINT8_C(177), UINT8_C(253), UINT8_C( 72),
        UINT8_C( 48), UINT8_C(214), UINT8_C(172), UINT8_C( 44), UINT8_C( 68), UINT8_C(126), UINT8_C( 74), UINT8_C(163),
        UINT8_C(229), UINT8_C(189), UINT8_C(147), UINT8_C( 19), UINT8_C(129), UINT8_C(136), UINT8_C(135), UINT8_C(246),
        UINT8_C( 41), UINT8_C(121), UINT8_C(113), UINT8_C(106), UINT8_C( 83), UINT8_C( 58), UINT8_C( 60), UINT8_C(  5),
        UINT8_C(199), UINT8_C( 34), UINT8_C( 51), UINT8_C( 94), UINT8_C(231), UINT8_C(161), UINT8_C(166), UINT8_C( 10),
        UINT8_C( 39), UINT8_C(126), UINT8_C(104), UINT8_C(107), UINT8_C(252), UINT8_C( 63), UINT8_C(246), UINT8_C(226),
        UINT8_C( 67), UINT8_C( 69), UINT8_C(231), UINT8_C(196), UINT8_C( 31), UINT8_C(125), UINT8_C(186), UINT8_C( 84) },
      UINT64_C( 6872534944075164476) },
    { { UINT8_C( 73), UINT8_C(147), UINT8_C(185), UINT8_C( 29), UINT8_C( 53), UINT8_C( 83), UINT8_C( 33), UINT8_C(172),
        UINT8_C(154), UINT8_C( 81), UINT8_C(172), UINT8_C(155), UINT8_C(144), UINT8_C(162), UINT8_C(250), UINT8_C(211),
        UINT8_C(232), UINT8_C(225), UINT8_C(151), UINT8_C(  7), UINT8_C(164), UINT8_C( 81), UINT8_C(172), UINT8_C(225),
        UINT8_C(125), UINT8_C(187), UINT8_C(113), UINT8_C(155), UINT8_C(225), UINT8_C(209), UINT8_C(250), UINT8_C( 42),
        UINT8_C(100), UINT8_C(179), UINT8_C( 71), UINT8_C(153), UINT8_C(  6), UINT8_C(104), UINT8_C( 70), UINT8_C(160),
        UINT8_C(186), UINT8_C(242), UINT8_C( 60), UINT8_C( 74), UINT8_C(148), UINT8_C( 54), UINT8_C( 30), UINT8_C(124),
        UINT8_C( 24), UINT8_C(181), UINT8_C(132), UINT8_C(188), UINT8_C(  7), UINT8_C( 48), UINT8_C(157), UINT8_C(132),
        UINT8_C(235), UINT8_C( 14), UINT8_C( 31), UINT8_C(204), UINT8_C(223), UINT8_C( 25), UINT8_C(247), UINT8_C( 68) },
      { UINT8_C(204), UINT8_C( 62), UINT8_C(185), UINT8_C( 29), UINT8_C(167), UINT8_C( 83), UINT8_C( 33), UINT8_C( 97),
        UINT8_C( 21), UINT8_C( 81), UINT8_C(171), UINT8_C(170), UINT8_C(144), UINT8_C(201), UINT8_C(250), UINT8_C(253),
        UINT8_C(232), UINT8_C(225), UINT8_C(186), UINT8_C(134), UINT8_C(164), UINT8_C( 81), UINT8_C(172), UINT8_C(225),
        UINT8_C(102), UINT8_C(187), UINT8_C(146), UINT8_C( 69), UINT8_C( 66), UINT8_C(209), UINT8_C(250), UINT8_C( 42),
        UINT8_C(100), UINT8_C(179), UINT8_C( 71), UINT8_C(153), UINT8_C(  6), UINT8_C(104), UINT8_C( 70), UINT8_C(160),
        UINT8_C(  2), UINT8_C(123), UINT8_C( 60), UINT8_C(232), UINT8_C( 68), UINT8_C( 54), UINT8_C(229), UINT8_C(195),
        UINT8_C( 27), UINT8_C(181), UINT8_C( 73), UINT8_C(245), UINT8_C(247), UINT8_C( 83), UINT8_C(157), UINT8_C( 93),
        UINT8_C(124), UINT8_C( 77), UINT8_C( 31), UINT8_C(204), UINT8_C(223), UINT8_C( 25), UINT8_C(247), UINT8_C( 68) },
      UINT64_C(  269612346245950867) },
    { { UINT8_C( 86), UINT8_C(  7), UINT8_C(235), UINT8_C(155), UINT8_C(120), UINT8_C(208), UINT8_C( 94), UINT8_C(147),
        UINT8_C(112), UINT8_C(168), UINT8_C(136), UINT8_C(103), UINT8_C(251), UINT8_C( 67), UINT8_C(196), UINT8_C(120),
        UINT8_C(144), UINT8_C(102), UINT8_C( 54), UINT8_C(102), UINT8_C(146), UINT8_C(  3), UINT8_C(  4), UINT8_C( 37),
        UINT8_C(176), UINT8_C( 16), UINT8_C( 67), UINT8_C(176), UINT8_C(235), UINT8_C(  0), UINT8_C(179), UINT8_C( 66),
        UINT8_C(  8), UINT8_C(158), UINT8_C(221), UINT8_C(128), UINT8_C(111), UINT8_C( 59), UINT8_C( 19), UINT8_C(223),
        UINT8_C(227), UINT8_C(155), UINT8_C( 70), UINT8_C(223), UINT8_C(223), UINT8_C( 10), UINT8_C( 87), UINT8_C(111),
        UINT8_C(112), UINT8_C(141), UINT8_C(214), UINT8_C(  3), UINT8_C(144), UINT8_C(218), UINT8_C( 40), UINT8_C( 64),
        UINT8_C(234), UINT8_C(107), UINT8_C(241), UINT8_C(213), UINT8_C(108), UINT8_C(164), UINT8_C( 23), UINT8_C(116) },
      { UINT8_C( 86), UINT8_C(  7), UINT8_C(235), UINT8_C(155), UINT8_C(120), UINT8_C(  7), UINT8_C(145), UINT8_C(147),
        UINT8_C(112), UINT8_C(168), UINT8_C(242), UINT8_C(129), UINT8_C(225), UINT8_C( 67), UINT8_C(241), UINT8_C(120),
        UINT8_C(215), UINT8_C(102), UINT8_C( 84), UINT8_C(103), UINT8_C(146), UINT8_C(125), UINT8_C(  4), UINT8_C(139),
        UINT8_C(176), UINT8_C( 16), UINT8_C( 96), UINT8_C(176), UINT8_C(235), UINT8_C(120), UINT8_C(179), UINT8_C( 66),
        UINT8_C(108), UINT8_C(188), UINT8_C( 50), UINT8_C(156), UINT8_C(195), UINT8_C( 59), UINT8_C( 19), UINT8_C(223),
        UINT8_C(154), UINT8_C(155), UINT8_C(231), UINT8_C(123), UINT8_C(236), UINT8_C( 10), UINT8_C(205), UINT8_C(111),
        UINT8_C(112), UINT8_C( 33), UINT8_C(214), UINT8_C( 64), UINT8_C(144), UINT8_C(218), UINT8_C( 40), UINT8_C(135),
        UINT8_C(234), UINT8_C( 44), UINT8_C(241), UINT8_C(169), UINT8_C(164), UINT8_C(164), UINT8_C( 23), UINT8_C( 16) },
      UINT64_C(11135815416967683168) },
    { {    UINT8_MAX, UINT8_C(113), UINT8_C( 53), UINT8_C(235), UINT8_C( 74), UINT8_C(  2), UINT8_C(174), UINT8_C(233),
        UINT8_C( 36), UINT8_C(217), UINT8_C( 42), UINT8_C(194), UINT8_C(171), UINT8_C(245), UINT8_C( 73), UINT8_C( 23),
        UINT8_C( 33), UINT8_C( 37), UINT8_C(192), UINT8_C(197), UINT8_C(201), UINT8_C(233), UINT8_C(214), UINT8_C( 41),
        UINT8_C( 69), UINT8_C(131), UINT8_C( 77), UINT8_C(101), UINT8_C(224), UINT8_C(215), UINT8_C( 31), UINT8_C(223),
        UINT8_C( 73), UINT8_C( 84), UINT8_C(203), UINT8_C(147), UINT8_C( 87), UINT8_C(121), UINT8_C(124), UINT8_C(123),
        UINT8_C( 82), UINT8_C(166), UINT8_C( 61), UINT8_C(254), UINT8_C(156), UINT8_C(135), UINT8_C( 21), UINT8_C(189),
        UINT8_C(172), UINT8_C(213), UINT8_C(131), UINT8_C(117), UINT8_C(190), UINT8_C( 89), UINT8_C(158), UINT8_C(  4),
        UINT8_C(220), UINT8_C(236), UINT8_C(105), UINT8_C(188), UINT8_C(195), UINT8_C(136), UINT8_C(155), UINT8_C( 12) },
      {    UINT8_MAX, UINT8_C(102), UINT8_C(159), UINT8_C( 51), UINT8_C( 74), UINT8_C(  2), UINT8_C(174), UINT8_C(233),
        UINT8_C(194), UINT8_C(236), UINT8_C( 48), UINT8_C(194), UINT8_C(171), UINT8_C(245), UINT8_C( 73), UINT8_C( 31),
        UINT8_C( 26), UINT8_C( 37), UINT8_C(192), UINT8_C(197), UINT8_C(248), UINT8_C(233), UINT8_C(214), UINT8_C( 41),
        UINT8_C( 69), UINT8_C(131), UINT8_C( 77), UINT8_C(226), UINT8_C(206), UINT8_C( 43), UINT8_C(238), UINT8_C(170),
        UINT8_C( 73), UINT8_C( 84), UINT8_C(222), UINT8_C(147), UINT8_C(170), UINT8_C(121), UINT8_C(124), UINT8_C(123),
        UINT8_C( 82), UINT8_C(213), UINT8_C(203), UINT8_C(254), UINT8_C( 26), UINT8_C(231), UINT8_C( 21), UINT8_C( 53),
        UINT8_C(172), UINT8_C(213), UINT8_C( 14), UINT8_C(117), UINT8_C(190), UINT8_C(235), UINT8_C( 82), UINT8_C(  4),
        UINT8_C(220), UINT8_C(236), UINT8_C(209),    UINT8_MAX, UINT8_C(195), UINT8_C(136), UINT8_C(155), UINT8_C(159) },
      UINT64_C(10116410864158476046) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epu8_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epu8_mask");

   easysimd_assert_equal_mmask64(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u8x64();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi8(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_u8x64());
    easysimd__mmask64 r = easysimd_mm512_cmpneq_epu8_mask(a, b);

    easysimd_test_x86_write_u8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpneq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t a[32];
    const uint16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { { UINT16_C(53276), UINT16_C( 4636), UINT16_C(42142), UINT16_C(36897), UINT16_C(55829), UINT16_C( 7504), UINT16_C(27569), UINT16_C(30727),
        UINT16_C(64475), UINT16_C(38420), UINT16_C( 8730), UINT16_C( 4314), UINT16_C(  976), UINT16_C(35021), UINT16_C(29417), UINT16_C( 1574),
        UINT16_C(16963), UINT16_C(57624), UINT16_C(15078), UINT16_C(64369), UINT16_C(49428), UINT16_C(50457), UINT16_C( 8237), UINT16_C( 2110),
        UINT16_C(21019), UINT16_C(13726), UINT16_C(30836), UINT16_C(17733), UINT16_C( 4987), UINT16_C(26061), UINT16_C(62341), UINT16_C(51307) },
      { UINT16_C(53276), UINT16_C( 7337), UINT16_C(42142), UINT16_C(53784), UINT16_C(12764), UINT16_C( 2455), UINT16_C(27569), UINT16_C(30727),
        UINT16_C(45096), UINT16_C(38420), UINT16_C( 8730), UINT16_C(42209), UINT16_C(45050), UINT16_C(32777), UINT16_C(29417), UINT16_C(55368),
        UINT16_C(62199), UINT16_C(57624), UINT16_C(15078), UINT16_C(59527), UINT16_C(49428), UINT16_C(36849), UINT16_C( 8237), UINT16_C( 7419),
        UINT16_C(21019), UINT16_C(56248), UINT16_C(39557), UINT16_C(17733), UINT16_C( 4987), UINT16_C(26061), UINT16_C(18684), UINT16_C(51307) },
      UINT32_C(1185528122) },
    { { UINT16_C(12486), UINT16_C( 1071), UINT16_C( 8270), UINT16_C(17043), UINT16_C(36387), UINT16_C(54878), UINT16_C( 5932), UINT16_C(45490),
        UINT16_C(12721), UINT16_C(64048), UINT16_C(12474), UINT16_C(46821), UINT16_C(43384), UINT16_C(45738), UINT16_C(21346), UINT16_C(10488),
        UINT16_C(10115), UINT16_C(53804), UINT16_C(48968), UINT16_C(27412), UINT16_C(29518), UINT16_C(31298), UINT16_C(62602), UINT16_C(15147),
        UINT16_C(23333), UINT16_C(57141), UINT16_C( 6795), UINT16_C(  918), UINT16_C(16580), UINT16_C( 9909), UINT16_C(44692), UINT16_C( 5967) },
      { UINT16_C(31701), UINT16_C( 7657), UINT16_C( 8270), UINT16_C(17043), UINT16_C(36387), UINT16_C(54878), UINT16_C(11967), UINT16_C(58422),
        UINT16_C(27529), UINT16_C(64048), UINT16_C(12474), UINT16_C(46821), UINT16_C(52890), UINT16_C(11888), UINT16_C(49020), UINT16_C(20806),
        UINT16_C(10115), UINT16_C(53804), UINT16_C(48968), UINT16_C(40702), UINT16_C(29518), UINT16_C(33433), UINT16_C(53039), UINT16_C(47462),
        UINT16_C(23333), UINT16_C(57141), UINT16_C( 6795), UINT16_C(  918), UINT16_C(31156), UINT16_C(12365), UINT16_C(37688), UINT16_C(29570) },
      UINT32_C(4041798083) },
    { { UINT16_C(59369), UINT16_C(44175), UINT16_C(10472), UINT16_C( 6190), UINT16_C(38136), UINT16_C(13009), UINT16_C(40895), UINT16_C(17394),
        UINT16_C(64645), UINT16_C(14946), UINT16_C(45173), UINT16_C(44650), UINT16_C(60483), UINT16_C( 1569), UINT16_C( 2525), UINT16_C(50935),
        UINT16_C(34544), UINT16_C(55666), UINT16_C(41134), UINT16_C(42737), UINT16_C(49717), UINT16_C(62681), UINT16_C(52065), UINT16_C(58935),
        UINT16_C(39623), UINT16_C(15648), UINT16_C(35658), UINT16_C(36331), UINT16_C( 3191), UINT16_C(21908), UINT16_C(35605), UINT16_C( 1563) },
      { UINT16_C(59369), UINT16_C(49119), UINT16_C(53294), UINT16_C(25446), UINT16_C(16274), UINT16_C(62295), UINT16_C(40895), UINT16_C(53977),
        UINT16_C(64645), UINT16_C(14946), UINT16_C(64133), UINT16_C(44650), UINT16_C(60483), UINT16_C( 6993), UINT16_C(27935), UINT16_C(12321),
        UINT16_C(  251), UINT16_C(55666), UINT16_C(22224), UINT16_C(25229), UINT16_C(49717), UINT16_C(62681), UINT16_C(52065), UINT16_C(40049),
        UINT16_C(39623), UINT16_C(44559), UINT16_C( 4218), UINT16_C(32938), UINT16_C( 3191), UINT16_C(21908), UINT16_C(35605), UINT16_C(25844) },
      UINT32_C(2391663806) },
    { { UINT16_C( 6714), UINT16_C(53233), UINT16_C(18175), UINT16_C(29295), UINT16_C(57461), UINT16_C(40463), UINT16_C( 7777), UINT16_C(56140),
        UINT16_C(63278), UINT16_C(54108), UINT16_C(63731), UINT16_C(23703), UINT16_C(35765), UINT16_C(29632), UINT16_C(19824), UINT16_C(43522),
        UINT16_C(62312), UINT16_C(26490), UINT16_C(59705), UINT16_C(45017), UINT16_C(59593), UINT16_C(10829), UINT16_C(39431), UINT16_C(13574),
        UINT16_C(25233), UINT16_C(33800), UINT16_C(40794), UINT16_C( 4064), UINT16_C(41003), UINT16_C(39811), UINT16_C(34285), UINT16_C(21829) },
      { UINT16_C(49016), UINT16_C(53233), UINT16_C(38568), UINT16_C(29280), UINT16_C(57461), UINT16_C(40463), UINT16_C( 7777), UINT16_C(56140),
        UINT16_C(49924), UINT16_C(54108), UINT16_C(15715), UINT16_C(36462), UINT16_C(35765), UINT16_C(29632), UINT16_C(28278), UINT16_C(43522),
        UINT16_C(62312), UINT16_C(54943), UINT16_C(59705), UINT16_C(45017), UINT16_C(59593), UINT16_C(10829), UINT16_C(39431), UINT16_C(36047),
        UINT16_C(11509), UINT16_C(22762), UINT16_C(22633), UINT16_C(18150), UINT16_C( 3913), UINT16_C(48912), UINT16_C(12413), UINT16_C(21829) },
      UINT32_C(2139245837) },
    { { UINT16_C(51789), UINT16_C(64368), UINT16_C(59311), UINT16_C(14321), UINT16_C(49176), UINT16_C( 3523), UINT16_C(44524), UINT16_C(21861),
        UINT16_C(19206), UINT16_C(20379), UINT16_C(43866), UINT16_C(55311), UINT16_C(48348), UINT16_C(59779), UINT16_C( 1289), UINT16_C(22120),
        UINT16_C(55760), UINT16_C(32593), UINT16_C(17088), UINT16_C(55478), UINT16_C(30978), UINT16_C(61158), UINT16_C(19239), UINT16_C(11587),
        UINT16_C(56983), UINT16_C(61820), UINT16_C(35722), UINT16_C(26313), UINT16_C(19784), UINT16_C(20815), UINT16_C(46930), UINT16_C( 8872) },
      { UINT16_C(51789), UINT16_C(20642), UINT16_C(22588), UINT16_C(15913), UINT16_C( 4050), UINT16_C(63789), UINT16_C(28762), UINT16_C(61734),
        UINT16_C(19206), UINT16_C(20379), UINT16_C(43866), UINT16_C(30271), UINT16_C(36601), UINT16_C(19655), UINT16_C( 1289), UINT16_C(54894),
        UINT16_C( 4201), UINT16_C(32593), UINT16_C(20329), UINT16_C(55478), UINT16_C(30978), UINT16_C(47412), UINT16_C(23169), UINT16_C(53418),
        UINT16_C(36348), UINT16_C(10921), UINT16_C(59450), UINT16_C(26313), UINT16_C(19784), UINT16_C(47999), UINT16_C(61143), UINT16_C( 8872) },
      UINT32_C(1743108350) },
    { { UINT16_C(51463), UINT16_C(26274), UINT16_C(55001), UINT16_C(23071), UINT16_C(51504), UINT16_C(11562), UINT16_C(54103), UINT16_C(37207),
        UINT16_C(63675), UINT16_C(12740), UINT16_C(17504), UINT16_C(14317), UINT16_C(32306), UINT16_C(12408), UINT16_C(23862), UINT16_C(16024),
        UINT16_C(14886), UINT16_C(  164), UINT16_C(49937), UINT16_C(16730), UINT16_C(34188), UINT16_C(58222), UINT16_C(50776), UINT16_C( 5236),
        UINT16_C(14782), UINT16_C( 7749), UINT16_C(12925), UINT16_C(44885), UINT16_C(52657), UINT16_C(59359), UINT16_C(30507), UINT16_C(20773) },
      { UINT16_C(51463), UINT16_C(50001), UINT16_C(44172), UINT16_C( 6404), UINT16_C(29489), UINT16_C(11562), UINT16_C(54103), UINT16_C(37207),
        UINT16_C(63675), UINT16_C(10005), UINT16_C(17504), UINT16_C(50902), UINT16_C(46392), UINT16_C(25518), UINT16_C(23862), UINT16_C(57268),
        UINT16_C( 1693), UINT16_C(  164), UINT16_C(49937), UINT16_C(16730), UINT16_C(16153), UINT16_C(58222), UINT16_C(50776), UINT16_C( 5236),
        UINT16_C(14782), UINT16_C( 7749), UINT16_C(12925), UINT16_C(44885), UINT16_C(30476), UINT16_C(59359), UINT16_C( 6218), UINT16_C(59160) },
      UINT32_C(3490822686) },
    { { UINT16_C(21345), UINT16_C(31411), UINT16_C( 8338), UINT16_C(17101), UINT16_C( 5674), UINT16_C( 6044), UINT16_C( 7541), UINT16_C(15897),
        UINT16_C(57972), UINT16_C(33087), UINT16_C(41817), UINT16_C(42170), UINT16_C(54203), UINT16_C(55947), UINT16_C(40077), UINT16_C(61098),
        UINT16_C(24304), UINT16_C(33385), UINT16_C(13950), UINT16_C(43205), UINT16_C(24908), UINT16_C(49599), UINT16_C(55423), UINT16_C(62463),
        UINT16_C(16059), UINT16_C( 5236), UINT16_C(12257), UINT16_C(40376), UINT16_C(17410), UINT16_C(36727), UINT16_C( 8672), UINT16_C(53374) },
      { UINT16_C(59263), UINT16_C(64851), UINT16_C( 6173), UINT16_C(17101), UINT16_C( 5674), UINT16_C( 6044), UINT16_C(10813), UINT16_C(63724),
        UINT16_C(57972), UINT16_C(33087), UINT16_C(41817), UINT16_C(37351), UINT16_C(24073), UINT16_C(55947), UINT16_C(40832), UINT16_C(65466),
        UINT16_C( 3462), UINT16_C(33385), UINT16_C(41509), UINT16_C(40460), UINT16_C(24908), UINT16_C(17559), UINT16_C(33634), UINT16_C(52028),
        UINT16_C(16059), UINT16_C( 5236), UINT16_C(64781), UINT16_C( 5636), UINT16_C(17410), UINT16_C(56064), UINT16_C(47812), UINT16_C(19163) },
      UINT32_C(3975010503) },
    { { UINT16_C(64122), UINT16_C(33162), UINT16_C( 8497), UINT16_C(37829), UINT16_C(  420), UINT16_C(34910), UINT16_C(29770), UINT16_C(22523),
        UINT16_C(65393), UINT16_C(52334), UINT16_C(28197), UINT16_C(59816), UINT16_C(33576), UINT16_C(61236), UINT16_C( 8539), UINT16_C(54747),
        UINT16_C(25883), UINT16_C(19799), UINT16_C( 7303), UINT16_C(11232), UINT16_C(16158), UINT16_C(26803), UINT16_C(44723), UINT16_C( 9407),
        UINT16_C(11694), UINT16_C(54256), UINT16_C(39067), UINT16_C(50108), UINT16_C(61467), UINT16_C(30386), UINT16_C(36114), UINT16_C(11596) },
      { UINT16_C(41971), UINT16_C(33162), UINT16_C( 8497), UINT16_C(56741), UINT16_C(  420), UINT16_C(19781), UINT16_C(29770), UINT16_C(46449),
        UINT16_C(24882), UINT16_C(52334), UINT16_C(17914), UINT16_C( 5521), UINT16_C(33576), UINT16_C(61236), UINT16_C( 8539), UINT16_C(54747),
        UINT16_C(61307), UINT16_C(19799), UINT16_C(58442), UINT16_C(58392), UINT16_C(16158), UINT16_C(17457), UINT16_C(44723), UINT16_C( 9407),
        UINT16_C(33284), UINT16_C(65123), UINT16_C(39067), UINT16_C(50108), UINT16_C(40760), UINT16_C( 2628), UINT16_C(47479), UINT16_C(62158) },
      UINT32_C(4079816105) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epu16_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epu16_mask");

   easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u16x32();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi16(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u16x32());
    easysimd__mmask32 r = easysimd_mm512_cmpneq_epu16_mask(a, b);

    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpneq_epu32_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t a[16];
    const uint32_t b[16];
    const easysimd__mmask16 r;
  } test_vec[] = {
    { { UINT32_C(3983526882), UINT32_C(3583642630), UINT32_C(1351460594), UINT32_C( 295643415), UINT32_C(3222466685), UINT32_C(4066359248), UINT32_C( 169986774), UINT32_C(4137321236),
        UINT32_C(1441007951), UINT32_C( 137002518), UINT32_C(1012447268), UINT32_C(1582168033), UINT32_C(3072221415), UINT32_C(3987308055), UINT32_C(1559808840), UINT32_C(2639434318) },
      { UINT32_C(3002218140), UINT32_C(3652853172), UINT32_C(1351460594), UINT32_C( 295643415), UINT32_C(3222466685), UINT32_C(4190589105), UINT32_C(1784004379), UINT32_C(3976702033),
        UINT32_C(1441007951), UINT32_C( 137002518), UINT32_C(1956807019), UINT32_C(1582168033), UINT32_C(3072221415), UINT32_C(2006337116), UINT32_C(1474423557), UINT32_C(1917118867) },
      UINT16_C(58595) },
    { { UINT32_C(1899887109), UINT32_C(2297604582), UINT32_C(2185286941), UINT32_C(4220332923), UINT32_C(1564600775), UINT32_C(1112098657), UINT32_C(3794525342), UINT32_C(1405488718),
        UINT32_C(4005823496), UINT32_C(3396843181), UINT32_C( 122468236), UINT32_C(2835536610), UINT32_C( 604456131), UINT32_C( 694636683), UINT32_C(2064355884), UINT32_C(2848903841) },
      { UINT32_C(1899887109), UINT32_C(2297604582), UINT32_C(2833095366), UINT32_C(4220332923), UINT32_C(1564600775), UINT32_C(3587687336), UINT32_C(4132496213), UINT32_C(2342526645),
        UINT32_C(4212013233), UINT32_C( 231758663), UINT32_C(1773514229), UINT32_C(2835536610), UINT32_C( 604456131), UINT32_C( 366753472), UINT32_C(3456904217), UINT32_C(2848903841) },
      UINT16_C(26596) },
    { { UINT32_C(3368233975), UINT32_C(3983915064), UINT32_C(3287482914), UINT32_C( 314593971), UINT32_C( 910721114), UINT32_C(2707645077), UINT32_C(2320346672), UINT32_C(2733715882),
        UINT32_C(2490020956), UINT32_C(2407653229), UINT32_C(2505209314), UINT32_C( 564597447), UINT32_C( 660139922), UINT32_C(2194258769), UINT32_C( 319559273), UINT32_C(2763390280) },
      { UINT32_C(3368233975), UINT32_C(3983915064), UINT32_C(4151705647), UINT32_C( 314593971), UINT32_C( 910721114), UINT32_C(2707645077), UINT32_C(2320346672), UINT32_C(2610388457),
        UINT32_C(2490020956), UINT32_C(3126683787), UINT32_C(2058474344), UINT32_C( 564597447), UINT32_C( 660139922), UINT32_C( 704264830), UINT32_C( 270311974), UINT32_C(2763390280) },
      UINT16_C(26244) },
    { { UINT32_C(1456345081), UINT32_C(2066298826), UINT32_C(3494189489), UINT32_C(1891270677), UINT32_C(3769252070), UINT32_C(2122568034), UINT32_C(2352120801), UINT32_C(4193433344),
        UINT32_C(2421146054), UINT32_C(2836166904), UINT32_C(2792968593), UINT32_C(3071685585), UINT32_C(3466117484), UINT32_C(1934367634), UINT32_C(2667544478), UINT32_C(4237816374) },
      { UINT32_C(1456345081), UINT32_C(4015298910), UINT32_C(3494189489), UINT32_C(1836297217), UINT32_C(4282059629), UINT32_C(2122568034), UINT32_C(1029861639), UINT32_C(4193433344),
        UINT32_C(1036109535), UINT32_C(2836166904), UINT32_C(2792968593), UINT32_C(3679680878), UINT32_C(2883227269), UINT32_C( 493898774), UINT32_C( 559600317), UINT32_C(2872546508) },
      UINT16_C(63834) },
    { { UINT32_C( 353352425), UINT32_C( 181924612), UINT32_C( 730023384), UINT32_C(4206495776), UINT32_C( 608620724), UINT32_C(1224082670), UINT32_C(1557971493), UINT32_C(1465202542),
        UINT32_C(4117521649), UINT32_C( 822035543), UINT32_C(2858123913), UINT32_C(1067718027), UINT32_C(3563318246), UINT32_C( 337402351), UINT32_C(2322659612), UINT32_C( 568444464) },
      { UINT32_C( 353352425), UINT32_C( 464655761), UINT32_C( 583339415), UINT32_C( 157378851), UINT32_C( 608620724), UINT32_C( 978843934), UINT32_C(1557971493), UINT32_C(3091441038),
        UINT32_C(4117521649), UINT32_C( 111144046), UINT32_C( 489186554), UINT32_C(1067718027), UINT32_C(3563318246), UINT32_C( 337402351), UINT32_C(2322659612), UINT32_C( 568444464) },
      UINT16_C( 1710) },
    { { UINT32_C( 636624262), UINT32_C(1267330083), UINT32_C( 802510345), UINT32_C(3727828088), UINT32_C( 932917136), UINT32_C( 799088670), UINT32_C(4270108979), UINT32_C( 956576691),
        UINT32_C(1935603536), UINT32_C(3988711395), UINT32_C(3055326269), UINT32_C(1217679288), UINT32_C(2625580926), UINT32_C(2194350415), UINT32_C(2239779026), UINT32_C(3552478595) },
      { UINT32_C(1598430332), UINT32_C(1267330083), UINT32_C( 802510345), UINT32_C( 916032184), UINT32_C( 932917136), UINT32_C( 193633593), UINT32_C(4270108979), UINT32_C( 205737616),
        UINT32_C(1935603536), UINT32_C( 648984718), UINT32_C(3648496673), UINT32_C(3893301295), UINT32_C(2625580926), UINT32_C(2194350415), UINT32_C(2239779026), UINT32_C(3072467020) },
      UINT16_C(36521) },
    { { UINT32_C(3527816996), UINT32_C(3581372254), UINT32_C(1340450368), UINT32_C(2133855630), UINT32_C(3724258927), UINT32_C(4158357786), UINT32_C(3122079640), UINT32_C(  55099614),
        UINT32_C(1456836344), UINT32_C( 892030197), UINT32_C(2172915954), UINT32_C(2365633565), UINT32_C(2104163171), UINT32_C( 359941501), UINT32_C(1271892844), UINT32_C(1145968716) },
      { UINT32_C(2627347366), UINT32_C(1657914736), UINT32_C(4074985173), UINT32_C(2133855630), UINT32_C(1558964703), UINT32_C(2607898414), UINT32_C(3122079640), UINT32_C(  55099614),
        UINT32_C(1456836344), UINT32_C( 892030197), UINT32_C(3449884099), UINT32_C(2365633565), UINT32_C( 959260170), UINT32_C( 359941501), UINT32_C( 967293664), UINT32_C(1161306862) },
      UINT16_C(54327) },
    { { UINT32_C( 927011085), UINT32_C(1648624832), UINT32_C( 209925841), UINT32_C( 875728135), UINT32_C(2597566662), UINT32_C(3495211816), UINT32_C( 636633836), UINT32_C(2516134536),
        UINT32_C(4241242683), UINT32_C( 257822782), UINT32_C(1293738310), UINT32_C( 897732206), UINT32_C( 802116870), UINT32_C(4127138825), UINT32_C(3994743142), UINT32_C(1468207899) },
      { UINT32_C(2337492813), UINT32_C(2795221344), UINT32_C( 209925841), UINT32_C( 204830213), UINT32_C(2597566662), UINT32_C(2412395049), UINT32_C( 636633836), UINT32_C(1167983096),
        UINT32_C(4241242683), UINT32_C( 878144674), UINT32_C( 674515747), UINT32_C(2369022657), UINT32_C(2556522351), UINT32_C(4127138825), UINT32_C(3994743142), UINT32_C(1468207899) },
      UINT16_C( 7851) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask16 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epu32_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epu32_mask");

   easysimd_assert_equal_mmask16(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u32x16();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi32(easysimd_test_x86_random_mmask16(), a, easysimd_test_x86_random_u32x16());
    easysimd__mmask16 r = easysimd_mm512_cmpneq_epu32_mask(a, b);

    easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_cmpneq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[8];
    const uint64_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { { UINT64_C(15386239089472589375), UINT64_C( 6313632889474977020), UINT64_C(17445595932806539732), UINT64_C(17081505422296092729),
        UINT64_C( 3369836232133200897), UINT64_C( 9303382045329859868), UINT64_C( 8659737359253113385), UINT64_C(13707992707589475233) },
      { UINT64_C(15386239089472589375), UINT64_C( 5387944669058949271), UINT64_C(17445595932806539732), UINT64_C(17081505422296092729),
        UINT64_C(17332903990917719745), UINT64_C(17520403422123410104), UINT64_C( 8659737359253113385), UINT64_C(17146271597939798462) },
      UINT8_C(178) },
    { { UINT64_C( 5917160433739531660), UINT64_C(14078070028593788463), UINT64_C(  489826748169347690), UINT64_C(13062162032675141321),
        UINT64_C( 8793057595764658597), UINT64_C( 3678036584208494125), UINT64_C( 2972391560521525861), UINT64_C( 3900685249914237508) },
      { UINT64_C( 5917160433739531660), UINT64_C(14078070028593788463), UINT64_C(17341287027921952286), UINT64_C(13896560800012527122),
        UINT64_C( 1860406351119894204), UINT64_C(14782116550370638396), UINT64_C( 4025360610661008447), UINT64_C( 3900685249914237508) },
      UINT8_C(124) },
    { { UINT64_C( 3576225853814908648), UINT64_C( 5148085799138336401), UINT64_C( 8235011383543749880), UINT64_C( 8985140389500096127),
        UINT64_C(14262761703663068487), UINT64_C( 8370617738736422689), UINT64_C(13362084690255779524), UINT64_C( 9496014895631473335) },
      { UINT64_C( 3576225853814908648), UINT64_C(11596870043013510144), UINT64_C(12818384708016361969), UINT64_C(17755094574390104743),
        UINT64_C(15773624437786454126), UINT64_C( 8370617738736422689), UINT64_C( 7800925896458615191), UINT64_C( 9496014895631473335) },
      UINT8_C( 94) },
    { { UINT64_C(  731293921865894748), UINT64_C( 8450970326803456361), UINT64_C( 6497189150670444912), UINT64_C(17008916637985727223),
        UINT64_C(13895743777663042779), UINT64_C(11848047257334076474), UINT64_C( 5046199062270355345), UINT64_C( 1086209960094950805) },
      { UINT64_C(  731293921865894748), UINT64_C( 1432634523552696840), UINT64_C( 6497189150670444912), UINT64_C(17008916637985727223),
        UINT64_C(  786477375537237361), UINT64_C(11848047257334076474), UINT64_C( 3846650267459273362), UINT64_C( 1086209960094950805) },
      UINT8_C( 82) },
    { { UINT64_C( 5613558891864664026), UINT64_C( 1900700293814179493), UINT64_C(12446639240780124086), UINT64_C(10107600056507015149),
        UINT64_C(11307817348738029967), UINT64_C(10211853707493651397), UINT64_C( 2303320945551465177), UINT64_C(13828399620567517354) },
      { UINT64_C(11782909619878260721), UINT64_C(10657477943640518529), UINT64_C(12446639240780124086), UINT64_C(10107600056507015149),
        UINT64_C(11307817348738029967), UINT64_C(10211853707493651397), UINT64_C( 8816447028724500272), UINT64_C( 7751211514728063109) },
      UINT8_C(195) },
    { { UINT64_C( 7477305824461645817), UINT64_C(10086347397979331278), UINT64_C( 7199253333166024980), UINT64_C( 9710495875955030826),
        UINT64_C( 5328375711527714948), UINT64_C(12437547543629574283), UINT64_C( 1124484720337290100), UINT64_C(16935172216240131059) },
      { UINT64_C( 3410747342301065873), UINT64_C(10086347397979331278), UINT64_C( 7199253333166024980), UINT64_C( 9710495875955030826),
        UINT64_C( 5328375711527714948), UINT64_C(12437547543629574283), UINT64_C(13187569082357481031), UINT64_C(  466253090717293191) },
      UINT8_C(193) },
    { { UINT64_C( 8823187301161367750), UINT64_C(14835922941240405440), UINT64_C( 7913029682742923169), UINT64_C( 5013397538651506523),
        UINT64_C(  843897487691221026), UINT64_C(15288183838265010701), UINT64_C( 4834875779246271769), UINT64_C( 9924147312363584756) },
      { UINT64_C(10738407284997973936), UINT64_C(14835922941240405440), UINT64_C(17110042469811882824), UINT64_C( 5013397538651506523),
        UINT64_C( 7538991150373930626), UINT64_C(17588368182580711566), UINT64_C( 8602322954667430523), UINT64_C( 9924147312363584756) },
      UINT8_C(117) },
    { { UINT64_C(11129906550267144711), UINT64_C( 7190905350940670945), UINT64_C( 7211672173441269599), UINT64_C(10886786416268960560),
        UINT64_C( 5598243081892596749), UINT64_C(10440304022507091503), UINT64_C(11433967026457439656), UINT64_C(11468225765560938495) },
      { UINT64_C(11129906550267144711), UINT64_C(14347674696626713948), UINT64_C( 7211672173441269599), UINT64_C(10886786416268960560),
        UINT64_C( 5598243081892596749), UINT64_C(10440304022507091503), UINT64_C(10386930096238831013), UINT64_C(11468225765560938495) },
      UINT8_C( 66) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_cmpneq_epu64_mask(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_cmpneq_epu64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u64x8());
    easysimd__mmask8 r = easysimd_mm512_cmpneq_epu64_mask(a, b);

    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpneq_epu8_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask64 k1;
    const uint8_t a[64];
    const uint8_t b[64];
    const easysimd__mmask64 r;
  } test_vec[] = {
    { UINT64_C(  277539215786243569),
      { UINT8_C( 16), UINT8_C(103), UINT8_C(254), UINT8_C(178), UINT8_C(186), UINT8_C(234), UINT8_C(110), UINT8_C( 57),
        UINT8_C(245), UINT8_C(205), UINT8_C(226), UINT8_C(225), UINT8_C(175), UINT8_C( 18), UINT8_C( 59), UINT8_C(138),
        UINT8_C(203), UINT8_C( 45), UINT8_C(242), UINT8_C(247), UINT8_C(107), UINT8_C(125), UINT8_C( 99), UINT8_C( 93),
        UINT8_C(115), UINT8_C(145), UINT8_C(157), UINT8_C(229), UINT8_C(149), UINT8_C(119), UINT8_C(232), UINT8_C(165),
        UINT8_C(223), UINT8_C(230), UINT8_C( 88), UINT8_C(153), UINT8_C(209), UINT8_C(198), UINT8_C(211), UINT8_C(198),
        UINT8_C(147), UINT8_C(181), UINT8_C(167), UINT8_C( 66), UINT8_C(199), UINT8_C(226), UINT8_C(205), UINT8_C(146),
        UINT8_C( 15), UINT8_C(191), UINT8_C(138), UINT8_C(123), UINT8_C( 60), UINT8_C(237), UINT8_C(216), UINT8_C(175),
        UINT8_C(126), UINT8_C(117), UINT8_C(148), UINT8_C( 19), UINT8_C(236), UINT8_C(124), UINT8_C(185), UINT8_C(203) },
      { UINT8_C( 16), UINT8_C(103), UINT8_C(101), UINT8_C(178), UINT8_C(186), UINT8_C( 56), UINT8_C(110), UINT8_C(106),
        UINT8_C(237), UINT8_C(161), UINT8_C(226), UINT8_C(180), UINT8_C(175), UINT8_C(122), UINT8_C( 59), UINT8_C(147),
        UINT8_C( 57), UINT8_C(208), UINT8_C( 14), UINT8_C(247), UINT8_C(107), UINT8_C(125), UINT8_C( 37), UINT8_C( 60),
        UINT8_C( 91), UINT8_C(145), UINT8_C( 80), UINT8_C(229), UINT8_C( 54), UINT8_C(119), UINT8_C( 19), UINT8_C(165),
        UINT8_C(223), UINT8_C(120), UINT8_C( 88), UINT8_C(153), UINT8_C(176), UINT8_C(199), UINT8_C( 91), UINT8_C(198),
        UINT8_C(147), UINT8_C(  8), UINT8_C(167), UINT8_C( 66), UINT8_C(199), UINT8_C(152), UINT8_C(205), UINT8_C(146),
        UINT8_C( 15), UINT8_C(142), UINT8_C(138), UINT8_C(123), UINT8_C(117), UINT8_C( 86), UINT8_C( 99), UINT8_C(208),
        UINT8_C( 15), UINT8_C(179), UINT8_C(148), UINT8_C( 69), UINT8_C(188), UINT8_C(124), UINT8_C(222), UINT8_C(214) },
      UINT64_C(  275283017923404192) },
    { UINT64_C( 4929550280267613227),
      { UINT8_C(214), UINT8_C(154), UINT8_C(107), UINT8_C( 75), UINT8_C(240), UINT8_C(206), UINT8_C( 27),    UINT8_MAX,
        UINT8_C(129), UINT8_C( 52), UINT8_C( 69), UINT8_C( 61), UINT8_C( 96), UINT8_C( 35), UINT8_C( 19), UINT8_C(  4),
        UINT8_C(207), UINT8_C(218), UINT8_C( 89), UINT8_C( 65), UINT8_C(252), UINT8_C( 76), UINT8_C( 28), UINT8_C( 39),
        UINT8_C(144), UINT8_C(228), UINT8_C(213), UINT8_C(108), UINT8_C( 43), UINT8_C( 62), UINT8_C(176), UINT8_C(  1),
        UINT8_C(216), UINT8_C( 27), UINT8_C( 76), UINT8_C(200), UINT8_C(233), UINT8_C(104), UINT8_C(199), UINT8_C(106),
        UINT8_C(156), UINT8_C( 12), UINT8_C(167), UINT8_C(252), UINT8_C( 48), UINT8_C(186), UINT8_C(  0),    UINT8_MAX,
        UINT8_C(148), UINT8_C( 90), UINT8_C( 64), UINT8_C(145), UINT8_C(166), UINT8_C( 93), UINT8_C(184), UINT8_C( 54),
        UINT8_C( 65), UINT8_C(141), UINT8_C(162), UINT8_C(108), UINT8_C(203), UINT8_C( 82), UINT8_C(110), UINT8_C(163) },
      { UINT8_C(110), UINT8_C(154), UINT8_C(107), UINT8_C( 75), UINT8_C( 34), UINT8_C(206), UINT8_C(194),    UINT8_MAX,
        UINT8_C(129), UINT8_C( 52), UINT8_C( 69), UINT8_C( 61), UINT8_C( 36), UINT8_C(187), UINT8_C(110), UINT8_C(  4),
        UINT8_C(207), UINT8_C(175), UINT8_C( 89), UINT8_C(187), UINT8_C(252), UINT8_C( 76), UINT8_C( 28), UINT8_C( 77),
        UINT8_C(144), UINT8_C(147), UINT8_C(185), UINT8_C( 91), UINT8_C( 43), UINT8_C( 39), UINT8_C(254), UINT8_C( 83),
        UINT8_C(226), UINT8_C( 27), UINT8_C(171), UINT8_C(  4), UINT8_C(157), UINT8_C(104), UINT8_C(195), UINT8_C(220),
        UINT8_C(214), UINT8_C( 12), UINT8_C( 76), UINT8_C(250), UINT8_C( 48), UINT8_C(186), UINT8_C(179),    UINT8_MAX,
        UINT8_C(105), UINT8_C(252), UINT8_C( 64), UINT8_C(117), UINT8_C(166), UINT8_C( 93), UINT8_C(194), UINT8_C(142),
        UINT8_C( 65), UINT8_C(141), UINT8_C(233), UINT8_C(108), UINT8_C(163), UINT8_C(231), UINT8_C(110), UINT8_C(133) },
      UINT64_C(  308854863540928513) },
    { UINT64_C( 4274023596712400842),
      { UINT8_C( 94), UINT8_C( 89), UINT8_C(176), UINT8_C( 92), UINT8_C( 82), UINT8_C(115), UINT8_C(234), UINT8_C(223),
        UINT8_C(239), UINT8_C(211), UINT8_C( 81), UINT8_C(146), UINT8_C(187), UINT8_C( 22), UINT8_C( 24), UINT8_C( 12),
        UINT8_C(135), UINT8_C(162), UINT8_C(251), UINT8_C(100), UINT8_C(239), UINT8_C(198), UINT8_C( 24), UINT8_C(185),
        UINT8_C(221), UINT8_C(199), UINT8_C(188), UINT8_C(174), UINT8_C( 40), UINT8_C( 13), UINT8_C(233), UINT8_C(134),
        UINT8_C(102), UINT8_C(154), UINT8_C(227), UINT8_C(184), UINT8_C( 13), UINT8_C(205), UINT8_C(151), UINT8_C(252),
        UINT8_C(161), UINT8_C(232), UINT8_C(142), UINT8_C( 92),    UINT8_MAX, UINT8_C(166), UINT8_C(104), UINT8_C(134),
        UINT8_C( 72), UINT8_C( 99), UINT8_C(234), UINT8_C( 55), UINT8_C( 41), UINT8_C(  3), UINT8_C(241), UINT8_C(  6),
        UINT8_C(202), UINT8_C(173), UINT8_C(181), UINT8_C(242), UINT8_C(186), UINT8_C(158), UINT8_C(121), UINT8_C( 32) },
      { UINT8_C( 94), UINT8_C( 92), UINT8_C(217), UINT8_C( 92), UINT8_C( 41), UINT8_C(112), UINT8_C(234), UINT8_C(202),
        UINT8_C( 89), UINT8_C(208), UINT8_C( 38), UINT8_C(146), UINT8_C(187), UINT8_C(143), UINT8_C( 24), UINT8_C( 12),
        UINT8_C(135), UINT8_C(200), UINT8_C(246), UINT8_C( 28), UINT8_C(203), UINT8_C(198), UINT8_C( 34), UINT8_C(149),
        UINT8_C(221), UINT8_C(199), UINT8_C(188), UINT8_C(174), UINT8_C(118), UINT8_C(  1), UINT8_C(112), UINT8_C(134),
        UINT8_C( 93), UINT8_C(154), UINT8_C(244), UINT8_C(134), UINT8_C(185), UINT8_C(205), UINT8_C( 81), UINT8_C(252),
        UINT8_C(  5), UINT8_C(119), UINT8_C(106), UINT8_C(124),    UINT8_MAX, UINT8_C( 72), UINT8_C(104), UINT8_C(134),
        UINT8_C( 72), UINT8_C( 49), UINT8_C(234), UINT8_C( 55), UINT8_C( 41), UINT8_C(  3), UINT8_C(241), UINT8_C(174),
        UINT8_C(202), UINT8_C(250), UINT8_C(181), UINT8_C(242), UINT8_C(186), UINT8_C(109), UINT8_C( 51), UINT8_C( 32) },
      UINT64_C( 2449994829074925442) },
    { UINT64_C( 5090784147129953703),
      { UINT8_C( 75), UINT8_C(187), UINT8_C( 35), UINT8_C(100), UINT8_C(243), UINT8_C(149), UINT8_C( 18), UINT8_C(  2),
        UINT8_C(143), UINT8_C( 15), UINT8_C(135), UINT8_C(138), UINT8_C(125), UINT8_C(186), UINT8_C(226), UINT8_C( 51),
        UINT8_C(226), UINT8_C(192), UINT8_C(163), UINT8_C( 63), UINT8_C(240), UINT8_C( 38), UINT8_C(161), UINT8_C(151),
        UINT8_C( 19), UINT8_C(128), UINT8_C( 68), UINT8_C( 72), UINT8_C(153), UINT8_C(235), UINT8_C(143), UINT8_C(228),
        UINT8_C(166), UINT8_C(178), UINT8_C( 72), UINT8_C(153), UINT8_C( 71), UINT8_C( 90), UINT8_C(155), UINT8_C(214),
        UINT8_C(106), UINT8_C( 34), UINT8_C( 96), UINT8_C(231), UINT8_C(221), UINT8_C( 66), UINT8_C( 26), UINT8_C(191),
        UINT8_C(  2), UINT8_C(190), UINT8_C(254), UINT8_C(242), UINT8_C(228), UINT8_C(159), UINT8_C(137), UINT8_C(247),
        UINT8_C( 31), UINT8_C(206), UINT8_C( 63), UINT8_C(185), UINT8_C(185), UINT8_C(206), UINT8_C(157), UINT8_C( 95) },
      { UINT8_C(128), UINT8_C(187), UINT8_C(249), UINT8_C(199), UINT8_C( 64), UINT8_C(148), UINT8_C( 18), UINT8_C(170),
        UINT8_C(143), UINT8_C( 15), UINT8_C(135), UINT8_C(148), UINT8_C( 63), UINT8_C(172), UINT8_C(226), UINT8_C( 66),
        UINT8_C(226), UINT8_C( 81), UINT8_C(163), UINT8_C( 78), UINT8_C(240), UINT8_C(190), UINT8_C(161), UINT8_C(151),
        UINT8_C(140), UINT8_C(132), UINT8_C(201), UINT8_C( 69), UINT8_C(153), UINT8_C(102), UINT8_C(164), UINT8_C(211),
        UINT8_C(166), UINT8_C(178), UINT8_C( 72), UINT8_C(153), UINT8_C( 50), UINT8_C( 56), UINT8_C( 55), UINT8_C(233),
        UINT8_C(106), UINT8_C(201), UINT8_C( 96), UINT8_C(231), UINT8_C(221), UINT8_C(208), UINT8_C(183), UINT8_C(191),
        UINT8_C(  2), UINT8_C(190), UINT8_C(254), UINT8_C( 17), UINT8_C(170), UINT8_C(159), UINT8_C( 33), UINT8_C( 54),
        UINT8_C( 31), UINT8_C(234), UINT8_C(123), UINT8_C(185), UINT8_C(185), UINT8_C( 31), UINT8_C(157), UINT8_C( 95) },
      UINT64_C(  468374570308118693) },
    { UINT64_C( 5079870325770704171),
      { UINT8_C( 63), UINT8_C(172), UINT8_C( 87), UINT8_C(233), UINT8_C( 30), UINT8_C(121), UINT8_C( 31), UINT8_C( 20),
        UINT8_C( 99), UINT8_C(154), UINT8_C( 94), UINT8_C(180), UINT8_C(186), UINT8_C(123), UINT8_C( 82), UINT8_C(119),
        UINT8_C( 51), UINT8_C(124), UINT8_C(102), UINT8_C( 35), UINT8_C(222), UINT8_C( 62), UINT8_C( 74), UINT8_C(  9),
        UINT8_C(147), UINT8_C(230), UINT8_C(169), UINT8_C(184), UINT8_C( 57), UINT8_C( 40), UINT8_C(254), UINT8_C(121),
        UINT8_C(212), UINT8_C( 85), UINT8_C( 98), UINT8_C(242), UINT8_C(206), UINT8_C(130), UINT8_C(  7), UINT8_C( 50),
        UINT8_C( 28), UINT8_C(101), UINT8_C(230), UINT8_C(214), UINT8_C(224), UINT8_C( 56), UINT8_C( 77), UINT8_C( 19),
        UINT8_C(181), UINT8_C(179), UINT8_C( 54), UINT8_C(147), UINT8_C(241), UINT8_C(128), UINT8_C(157), UINT8_C(132),
        UINT8_C(102), UINT8_C( 70), UINT8_C( 60), UINT8_C(160), UINT8_C(111), UINT8_C( 58), UINT8_C( 25), UINT8_C( 67) },
      { UINT8_C(144), UINT8_C(172), UINT8_C( 54), UINT8_C( 94), UINT8_C( 30), UINT8_C( 61), UINT8_C(144), UINT8_C( 26),
        UINT8_C(162), UINT8_C(119), UINT8_C(240), UINT8_C(180), UINT8_C(186), UINT8_C( 62), UINT8_C( 82), UINT8_C(119),
        UINT8_C(241), UINT8_C(124), UINT8_C(248), UINT8_C(227), UINT8_C(222), UINT8_C( 62), UINT8_C( 74), UINT8_C(  9),
        UINT8_C(147), UINT8_C(164), UINT8_C( 82), UINT8_C( 74), UINT8_C( 57), UINT8_C(107), UINT8_C(142), UINT8_C(110),
        UINT8_C(212), UINT8_C( 85), UINT8_C(205), UINT8_C(242), UINT8_C(206), UINT8_C(130), UINT8_C(  7), UINT8_C(163),
        UINT8_C( 28), UINT8_C(101), UINT8_C( 37), UINT8_C(132), UINT8_C(224), UINT8_C( 56), UINT8_C( 77), UINT8_C( 19),
        UINT8_C(133), UINT8_C(179), UINT8_C( 54), UINT8_C(147), UINT8_C(117), UINT8_C(128), UINT8_C(157), UINT8_C( 81),
        UINT8_C( 13), UINT8_C( 70), UINT8_C( 60), UINT8_C(235), UINT8_C( 66), UINT8_C( 58), UINT8_C( 90), UINT8_C( 67) },
      UINT64_C( 4616471112896480553) },
    { UINT64_C(17746088467212646139),
      { UINT8_C(166), UINT8_C( 72), UINT8_C(199), UINT8_C( 27), UINT8_C(177), UINT8_C( 75), UINT8_C(108), UINT8_C(190),
        UINT8_C( 34), UINT8_C(  8), UINT8_C(169), UINT8_C(100), UINT8_C( 49), UINT8_C(  3), UINT8_C(141), UINT8_C( 31),
        UINT8_C( 42), UINT8_C(154), UINT8_C( 13), UINT8_C(175), UINT8_C(166), UINT8_C(159), UINT8_C(  8), UINT8_C(161),
        UINT8_C( 85), UINT8_C(229), UINT8_C(201), UINT8_C(198), UINT8_C(170), UINT8_C( 16), UINT8_C(188), UINT8_C( 80),
        UINT8_C( 88), UINT8_C(131), UINT8_C(108), UINT8_C(  9), UINT8_C(207), UINT8_C(216), UINT8_C(199), UINT8_C(241),
        UINT8_C(224), UINT8_C(112), UINT8_C( 85), UINT8_C( 18), UINT8_C(116), UINT8_C(226), UINT8_C( 49), UINT8_C(158),
        UINT8_C(124), UINT8_C( 62), UINT8_C( 77), UINT8_C( 35), UINT8_C(221), UINT8_C( 85), UINT8_C(196), UINT8_C( 51),
        UINT8_C( 58), UINT8_C(142), UINT8_C(249), UINT8_C(229), UINT8_C(158), UINT8_C(181), UINT8_C( 53), UINT8_C(246) },
      { UINT8_C( 56), UINT8_C( 72), UINT8_C(199), UINT8_C(  7), UINT8_C(177), UINT8_C(198), UINT8_C(108), UINT8_C( 90),
        UINT8_C( 34), UINT8_C(  8), UINT8_C(169), UINT8_C(100), UINT8_C( 47), UINT8_C(157), UINT8_C( 73), UINT8_C( 31),
        UINT8_C(220), UINT8_C(154), UINT8_C( 13), UINT8_C(185), UINT8_C(236), UINT8_C(159), UINT8_C(236), UINT8_C( 38),
        UINT8_C( 85), UINT8_C(229), UINT8_C( 11), UINT8_C(198), UINT8_C(170), UINT8_C( 65), UINT8_C(188), UINT8_C( 80),
        UINT8_C(226), UINT8_C(180), UINT8_C(108), UINT8_C(  9), UINT8_C(207), UINT8_C(216), UINT8_C(183), UINT8_C(241),
        UINT8_C(224), UINT8_C(112), UINT8_C( 85), UINT8_C( 18), UINT8_C(193), UINT8_C(226), UINT8_C( 49), UINT8_C(157),
        UINT8_C( 59), UINT8_C( 62), UINT8_C( 86), UINT8_C( 35), UINT8_C( 94), UINT8_C( 85), UINT8_C( 77), UINT8_C(128),
        UINT8_C( 58), UINT8_C(142), UINT8_C( 63), UINT8_C(229), UINT8_C(158), UINT8_C(245), UINT8_C(150), UINT8_C(246) },
      UINT64_C( 7225040715126485161) },
    { UINT64_C( 1157901010043416755),
      { UINT8_C(123), UINT8_C(104), UINT8_C( 55), UINT8_C(217), UINT8_C(171), UINT8_C(132), UINT8_C( 89), UINT8_C(211),
        UINT8_C(221), UINT8_C(153), UINT8_C(150), UINT8_C(119), UINT8_C(142), UINT8_C( 44), UINT8_C(244), UINT8_C( 55),
        UINT8_C(157), UINT8_C(205), UINT8_C( 91), UINT8_C(224), UINT8_C( 93), UINT8_C( 48), UINT8_C( 68), UINT8_C( 16),
        UINT8_C( 97), UINT8_C(248), UINT8_C(133), UINT8_C( 54), UINT8_C(168), UINT8_C(150), UINT8_C( 70), UINT8_C( 35),
        UINT8_C(254), UINT8_C(125), UINT8_C(253), UINT8_C(169), UINT8_C(  1), UINT8_C( 86), UINT8_C(125), UINT8_C(223),
        UINT8_C(239), UINT8_C( 19), UINT8_C( 86), UINT8_C(125), UINT8_C( 64), UINT8_C( 74), UINT8_C(181), UINT8_C(221),
        UINT8_C( 23), UINT8_C( 16), UINT8_C(189), UINT8_C(116), UINT8_C( 65), UINT8_C(  2), UINT8_C(133), UINT8_C(162),
        UINT8_C(250), UINT8_C( 10), UINT8_C(216), UINT8_C(163), UINT8_C(160), UINT8_C( 30), UINT8_C(198), UINT8_C(159) },
      { UINT8_C(155), UINT8_C(195), UINT8_C( 72), UINT8_C(217), UINT8_C( 26), UINT8_C(132), UINT8_C( 89), UINT8_C(211),
        UINT8_C(217), UINT8_C(210), UINT8_C(135), UINT8_C(119), UINT8_C( 28), UINT8_C( 44), UINT8_C(244), UINT8_C( 55),
        UINT8_C(157), UINT8_C(179), UINT8_C(168), UINT8_C(224), UINT8_C( 93), UINT8_C( 48), UINT8_C( 47), UINT8_C(176),
        UINT8_C( 97), UINT8_C(248), UINT8_C( 83), UINT8_C(216), UINT8_C( 37), UINT8_C(150), UINT8_C( 70), UINT8_C(192),
        UINT8_C(254), UINT8_C(125), UINT8_C(253), UINT8_C(169), UINT8_C(133), UINT8_C(216), UINT8_C(  0), UINT8_C( 94),
        UINT8_C(170), UINT8_C(135), UINT8_C(119), UINT8_C(125), UINT8_C( 64), UINT8_C( 74), UINT8_C(251), UINT8_C( 16),
        UINT8_C( 23), UINT8_C(163), UINT8_C(189), UINT8_C(214), UINT8_C(209), UINT8_C(205), UINT8_C(134), UINT8_C(  8),
        UINT8_C(250), UINT8_C(217), UINT8_C(216), UINT8_C(250), UINT8_C(242), UINT8_C( 30), UINT8_C(198), UINT8_C(207) },
      UINT64_C( 1157566735419969555) },
    { UINT64_C(18312301143702729038),
      { UINT8_C(  0), UINT8_C(191), UINT8_C(212), UINT8_C(209), UINT8_C(140), UINT8_C( 90), UINT8_C(217), UINT8_C( 97),
        UINT8_C( 51), UINT8_C(186), UINT8_C( 91), UINT8_C( 38), UINT8_C( 17), UINT8_C( 21), UINT8_C(245), UINT8_C( 40),
        UINT8_C( 45), UINT8_C(188), UINT8_C(196), UINT8_C( 29), UINT8_C(131), UINT8_C(190), UINT8_C(183), UINT8_C(209),
        UINT8_C( 47), UINT8_C( 25), UINT8_C(227), UINT8_C( 13), UINT8_C(117), UINT8_C(  5), UINT8_C( 12), UINT8_C(117),
        UINT8_C(197), UINT8_C(224), UINT8_C( 70), UINT8_C( 81), UINT8_C( 59), UINT8_C( 32), UINT8_C(178), UINT8_C(110),
        UINT8_C(218), UINT8_C( 13), UINT8_C(148), UINT8_C(235), UINT8_C( 35), UINT8_C(138), UINT8_C( 20), UINT8_C( 80),
        UINT8_C( 70), UINT8_C(216), UINT8_C(109), UINT8_C(201), UINT8_C(151), UINT8_C( 36), UINT8_C(154), UINT8_C(198),
        UINT8_C( 61), UINT8_C(126), UINT8_C(212), UINT8_C(179), UINT8_C(131), UINT8_C(224), UINT8_C( 40), UINT8_C( 72) },
      { UINT8_C(192), UINT8_C(191), UINT8_C(154), UINT8_C(251), UINT8_C(140), UINT8_C( 90), UINT8_C(217), UINT8_C( 97),
        UINT8_C( 90), UINT8_C(186), UINT8_C( 91), UINT8_C(125), UINT8_C(136), UINT8_C(104), UINT8_C(245), UINT8_C( 40),
        UINT8_C( 65), UINT8_C(188), UINT8_C(151), UINT8_C(216), UINT8_C( 94), UINT8_C(190), UINT8_C(158), UINT8_C(209),
        UINT8_C(176), UINT8_C(114), UINT8_C( 79), UINT8_C( 51), UINT8_C(117), UINT8_C(119), UINT8_C(124), UINT8_C(117),
        UINT8_C(197), UINT8_C( 22), UINT8_C( 70), UINT8_C( 81), UINT8_C( 98), UINT8_C(120), UINT8_C(178), UINT8_C(188),
        UINT8_C(218), UINT8_C( 13), UINT8_C( 57),    UINT8_MAX, UINT8_C(155), UINT8_C(  6), UINT8_C( 20), UINT8_C( 80),
        UINT8_C( 70), UINT8_C(216), UINT8_C(180), UINT8_C(159), UINT8_C(151), UINT8_C( 83), UINT8_C(154), UINT8_C(198),
        UINT8_C(197), UINT8_C(126), UINT8_C(212), UINT8_C( 24), UINT8_C(131), UINT8_C(247), UINT8_C( 40), UINT8_C( 72) },
      UINT64_C( 2891342374200488204) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask64 k = test_vec[i].k1;
    easysimd__mmask64 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epu8_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epu8_mask");

   easysimd_assert_equal_mmask64(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k1 = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_u8x64();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi8(easysimd_test_x86_random_mmask64(), a, easysimd_test_x86_random_u8x64());
    easysimd__mmask64 r = easysimd_mm512_mask_cmpneq_epu8_mask(k1, a, b);

    easysimd_test_x86_write_mmask64(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpneq_epu16_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k1;
    const uint16_t a[32];
    const uint16_t b[32];
    const easysimd__mmask32 r;
  } test_vec[] = {
    { UINT32_C(1110894618),
      { UINT16_C(45793), UINT16_C(24833), UINT16_C(33420), UINT16_C(41155), UINT16_C(39526), UINT16_C( 7237), UINT16_C(22183), UINT16_C(38435),
        UINT16_C(34181), UINT16_C(44316), UINT16_C(13127), UINT16_C(14744), UINT16_C(16259), UINT16_C(40456), UINT16_C(16167), UINT16_C( 2272),
        UINT16_C(57841), UINT16_C(32106), UINT16_C(11619), UINT16_C(51485), UINT16_C(25543), UINT16_C(28645), UINT16_C( 2489), UINT16_C(15877),
        UINT16_C( 8846), UINT16_C(55020), UINT16_C(33877), UINT16_C(55311), UINT16_C( 6339), UINT16_C(60278), UINT16_C(22359), UINT16_C(18675) },
      { UINT16_C(45793), UINT16_C(24833), UINT16_C(58251), UINT16_C(21093), UINT16_C(19270), UINT16_C(  193), UINT16_C(51028), UINT16_C(57918),
        UINT16_C(10985), UINT16_C(16056), UINT16_C(51374), UINT16_C(14744), UINT16_C(36320), UINT16_C(40456), UINT16_C(16167), UINT16_C( 7295),
        UINT16_C(17838), UINT16_C(32106), UINT16_C( 7721), UINT16_C(28555), UINT16_C(25543), UINT16_C(28645), UINT16_C( 2489), UINT16_C(64927),
        UINT16_C(22744), UINT16_C(55020), UINT16_C(20768), UINT16_C(55311), UINT16_C( 6339), UINT16_C(49719), UINT16_C(22359), UINT16_C(18675) },
      UINT32_C(    294936) },
    { UINT32_C( 513022389),
      { UINT16_C( 1126), UINT16_C(31451), UINT16_C(31666), UINT16_C(35447), UINT16_C(45779), UINT16_C(62225), UINT16_C( 2563), UINT16_C(58099),
        UINT16_C(10848), UINT16_C( 1956), UINT16_C(33760), UINT16_C(56411), UINT16_C(59675), UINT16_C(53249), UINT16_C(38402), UINT16_C(26863),
        UINT16_C(51866), UINT16_C(19682), UINT16_C(22853), UINT16_C( 6358), UINT16_C(59403), UINT16_C( 3595), UINT16_C(65266), UINT16_C(21488),
        UINT16_C(38184), UINT16_C( 2394), UINT16_C(46360), UINT16_C(13285), UINT16_C(59294), UINT16_C(40964), UINT16_C(62333), UINT16_C( 5896) },
      { UINT16_C(60093), UINT16_C(31451), UINT16_C(14659), UINT16_C(35447), UINT16_C(45779), UINT16_C(62225), UINT16_C(19749), UINT16_C(20071),
        UINT16_C(10848), UINT16_C(64343), UINT16_C(33760), UINT16_C(56411), UINT16_C(12835), UINT16_C(53249), UINT16_C(38402), UINT16_C(26863),
        UINT16_C( 6824), UINT16_C(19682), UINT16_C(  340), UINT16_C( 6358), UINT16_C(59403), UINT16_C( 3595), UINT16_C(61669), UINT16_C(21488),
        UINT16_C(62385), UINT16_C( 2394), UINT16_C(46360), UINT16_C(21309), UINT16_C(61987), UINT16_C(40964), UINT16_C(62333), UINT16_C( 5896) },
      UINT32_C( 402919557) },
    { UINT32_C(1016037139),
      { UINT16_C( 6167), UINT16_C(64650), UINT16_C( 9737), UINT16_C(47811), UINT16_C(34329), UINT16_C(18658), UINT16_C( 8055), UINT16_C(39579),
        UINT16_C(36626), UINT16_C(49891), UINT16_C( 3898), UINT16_C(65307), UINT16_C(24609), UINT16_C(13593), UINT16_C(43232), UINT16_C(63345),
        UINT16_C(64448), UINT16_C(51699), UINT16_C(46625), UINT16_C(14980), UINT16_C(26172), UINT16_C(45954), UINT16_C( 7814), UINT16_C(38990),
        UINT16_C(12717), UINT16_C(59226), UINT16_C(30273), UINT16_C(25318), UINT16_C(65494), UINT16_C(46743), UINT16_C( 2215), UINT16_C(26797) },
      { UINT16_C(40963), UINT16_C(64650), UINT16_C(46423), UINT16_C(47811), UINT16_C(34329), UINT16_C(18658), UINT16_C( 8055), UINT16_C(39579),
        UINT16_C(38086), UINT16_C( 1939), UINT16_C(30986), UINT16_C(57706), UINT16_C(24609), UINT16_C( 8343), UINT16_C(43232), UINT16_C(63345),
        UINT16_C(64448), UINT16_C(15410), UINT16_C(36975), UINT16_C(14980), UINT16_C(26172), UINT16_C(28717), UINT16_C(26540), UINT16_C(29212),
        UINT16_C(12717), UINT16_C(59226), UINT16_C(58409), UINT16_C(25318), UINT16_C(32741), UINT16_C(61378), UINT16_C(19396), UINT16_C(26797) },
      UINT32_C( 881209089) },
    { UINT32_C( 822130367),
      { UINT16_C(11725), UINT16_C(31137), UINT16_C(48789), UINT16_C(37355), UINT16_C(25965), UINT16_C(38551), UINT16_C(32585), UINT16_C(12088),
        UINT16_C(64510), UINT16_C(49694), UINT16_C( 6982), UINT16_C(19307), UINT16_C(20810), UINT16_C( 2751), UINT16_C(48903), UINT16_C(54331),
        UINT16_C(56557), UINT16_C(33357), UINT16_C(14746), UINT16_C( 2067), UINT16_C(43678), UINT16_C(59550), UINT16_C(55081), UINT16_C(10007),
        UINT16_C(13778), UINT16_C( 6377), UINT16_C(21841), UINT16_C(39779), UINT16_C( 8870), UINT16_C(44709), UINT16_C(57570), UINT16_C(53122) },
      { UINT16_C(11725), UINT16_C(31137), UINT16_C(25609), UINT16_C(37355), UINT16_C(25965), UINT16_C(38551), UINT16_C(42709), UINT16_C(42847),
        UINT16_C(64510), UINT16_C(49694), UINT16_C( 6982), UINT16_C(17608), UINT16_C(20810), UINT16_C( 9970), UINT16_C(48903), UINT16_C(54331),
        UINT16_C(56557), UINT16_C(20067), UINT16_C(49834), UINT16_C( 2067), UINT16_C(34240), UINT16_C(38385), UINT16_C(20523), UINT16_C(10007),
        UINT16_C(64409), UINT16_C(14132), UINT16_C(64797), UINT16_C(25212), UINT16_C( 8870), UINT16_C(47496), UINT16_C(32483), UINT16_C(53122) },
      UINT32_C( 553656452) },
    { UINT32_C(2871553258),
      { UINT16_C( 6641), UINT16_C( 7232), UINT16_C(32105), UINT16_C(  804), UINT16_C(22648), UINT16_C(38458), UINT16_C(46677), UINT16_C(49400),
        UINT16_C(32805), UINT16_C( 2170), UINT16_C(16382), UINT16_C(49969), UINT16_C(42855), UINT16_C(20786), UINT16_C(23059), UINT16_C( 1276),
        UINT16_C(15731), UINT16_C(56353), UINT16_C(17850), UINT16_C(13023), UINT16_C( 6813), UINT16_C(62408), UINT16_C(49360), UINT16_C(62899),
        UINT16_C(11585), UINT16_C(16382), UINT16_C(12140), UINT16_C(54018), UINT16_C(13526), UINT16_C(59941), UINT16_C( 8590), UINT16_C(  494) },
      { UINT16_C( 3934), UINT16_C( 7232), UINT16_C(32105), UINT16_C(  804), UINT16_C( 5079), UINT16_C(43237), UINT16_C(39124), UINT16_C(49400),
        UINT16_C(39878), UINT16_C( 2170), UINT16_C(16382), UINT16_C(49969), UINT16_C(11147), UINT16_C( 6795), UINT16_C(23059), UINT16_C( 1276),
        UINT16_C(15731), UINT16_C(56771), UINT16_C( 3767), UINT16_C(13023), UINT16_C( 6813), UINT16_C(62408), UINT16_C(54349), UINT16_C(62899),
        UINT16_C(11585), UINT16_C(16382), UINT16_C(12140), UINT16_C(17115), UINT16_C(13526), UINT16_C(50012), UINT16_C( 8590), UINT16_C(  494) },
      UINT32_C( 671096928) },
    { UINT32_C(1656100160),
      { UINT16_C(60874), UINT16_C( 5976), UINT16_C(25537), UINT16_C(12330), UINT16_C(28610), UINT16_C(31082), UINT16_C(17851), UINT16_C(12731),
        UINT16_C( 6059), UINT16_C(35828), UINT16_C(25230), UINT16_C(65523), UINT16_C(14740), UINT16_C(54311), UINT16_C(56911), UINT16_C( 6454),
        UINT16_C(36555), UINT16_C(35888), UINT16_C(23281), UINT16_C(46012), UINT16_C(10185), UINT16_C(33836), UINT16_C(59244), UINT16_C( 6326),
        UINT16_C(43774), UINT16_C(36259), UINT16_C(38413), UINT16_C(41356), UINT16_C(46288), UINT16_C( 8053), UINT16_C(43922), UINT16_C(23864) },
      { UINT16_C(60874), UINT16_C( 5976), UINT16_C(25537), UINT16_C(12330), UINT16_C(28610), UINT16_C(31082), UINT16_C(50929), UINT16_C(12731),
        UINT16_C( 6059), UINT16_C(35828), UINT16_C( 2442), UINT16_C(65523), UINT16_C(37821), UINT16_C(20345), UINT16_C(56911), UINT16_C( 6454),
        UINT16_C(36555), UINT16_C(56225), UINT16_C(23281), UINT16_C( 1895), UINT16_C(10185), UINT16_C(33836), UINT16_C(59244), UINT16_C(44394),
        UINT16_C(59269), UINT16_C( 4139), UINT16_C(38413), UINT16_C(44650), UINT16_C(58589), UINT16_C( 7166), UINT16_C(43669), UINT16_C(23864) },
      UINT32_C(1652692032) },
    { UINT32_C(1015214515),
      { UINT16_C(50024), UINT16_C(42423), UINT16_C( 8532), UINT16_C(55891), UINT16_C(32265), UINT16_C(64234), UINT16_C(21703), UINT16_C(42152),
        UINT16_C(42552), UINT16_C(52928), UINT16_C(21329), UINT16_C(37245), UINT16_C( 1927), UINT16_C(15116), UINT16_C(36601), UINT16_C(24951),
        UINT16_C(11857), UINT16_C(42503), UINT16_C(23120), UINT16_C(22912), UINT16_C(27352), UINT16_C(40787), UINT16_C(64446), UINT16_C(63300),
        UINT16_C( 1186), UINT16_C(62405), UINT16_C(16983), UINT16_C(56964), UINT16_C(36937), UINT16_C(16921), UINT16_C(37150), UINT16_C(28836) },
      { UINT16_C(43967), UINT16_C(42423), UINT16_C(38405), UINT16_C(55891), UINT16_C(47872), UINT16_C(64234), UINT16_C(21703), UINT16_C(22965),
        UINT16_C(42552), UINT16_C(52928), UINT16_C(21329), UINT16_C(37245), UINT16_C( 1927), UINT16_C(32328), UINT16_C(36601), UINT16_C(24951),
        UINT16_C(11857), UINT16_C(42503), UINT16_C(56474), UINT16_C(22912), UINT16_C(63127), UINT16_C(20057), UINT16_C(64446), UINT16_C(31655),
        UINT16_C(62345), UINT16_C(17814), UINT16_C(37059), UINT16_C( 9035), UINT16_C(36937), UINT16_C(18594), UINT16_C(37150), UINT16_C(28836) },
      UINT32_C( 746594449) },
    { UINT32_C(2479501052),
      { UINT16_C( 8996), UINT16_C(56034), UINT16_C(35121), UINT16_C(47701), UINT16_C(60541), UINT16_C(16384), UINT16_C(19324), UINT16_C( 8292),
        UINT16_C( 1759), UINT16_C(24681), UINT16_C( 5526), UINT16_C(11128), UINT16_C(11317), UINT16_C(12635), UINT16_C( 9562), UINT16_C(32453),
        UINT16_C(42824), UINT16_C(31065), UINT16_C(44592), UINT16_C(44340), UINT16_C(13466), UINT16_C( 6126), UINT16_C(21119), UINT16_C(24375),
        UINT16_C(41048), UINT16_C(61119), UINT16_C(14262), UINT16_C(60186), UINT16_C(30051), UINT16_C(48669), UINT16_C(58010), UINT16_C(57916) },
      { UINT16_C(38281), UINT16_C(47451), UINT16_C(36676), UINT16_C(47701), UINT16_C(60541), UINT16_C(17397), UINT16_C(11687), UINT16_C(65442),
        UINT16_C( 1759), UINT16_C(33773), UINT16_C( 5526), UINT16_C(11128), UINT16_C(35964), UINT16_C( 5817), UINT16_C( 9562), UINT16_C(32453),
        UINT16_C(21643), UINT16_C(53168), UINT16_C(44592), UINT16_C(42926), UINT16_C(13466), UINT16_C( 6126), UINT16_C(21119), UINT16_C(40466),
        UINT16_C(41048), UINT16_C(34081), UINT16_C(14262), UINT16_C(33920), UINT16_C(30051), UINT16_C(48669), UINT16_C(37680), UINT16_C(48001) },
      UINT32_C(2190090980) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask32 k = test_vec[i].k1;
    easysimd__mmask32 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epu16_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epu16_mask");

   easysimd_assert_equal_mmask32(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k1 = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_u16x32();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi16(easysimd_test_x86_random_mmask32(), a, easysimd_test_x86_random_u16x32());
    easysimd__mmask32 r = easysimd_mm512_mask_cmpneq_epu16_mask(k1, a, b);

    easysimd_test_x86_write_mmask32(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_cmpneq_epu32_mask(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k1;
    uint32_t a[16];
    uint32_t b[16];
    easysimd__mmask16 k;
  } test_vec[] = {
    { UINT16_C(14313),
      { UINT32_C(2864264490), UINT32_C(3071660493), UINT32_C(1901002265), UINT32_C( 142281530), UINT32_C(1421024467), UINT32_C(2444742007), UINT32_C(3005701573), UINT32_C(2196377688),
        UINT32_C( 321758022), UINT32_C(2395619957), UINT32_C(2063538240), UINT32_C(4135746083), UINT32_C( 239744406), UINT32_C(3365863939), UINT32_C(3095119455), UINT32_C( 473589206) },
      { UINT32_C(2117035785), UINT32_C(3909941673), UINT32_C( 895749138), UINT32_C( 489481862), UINT32_C( 522941980), UINT32_C(3639069304), UINT32_C(1720738448), UINT32_C(3498298055),
        UINT32_C(3679367730), UINT32_C(3200605100), UINT32_C(3992136039), UINT32_C( 722083599), UINT32_C( 239744406), UINT32_C(2414227967), UINT32_C(1542878867), UINT32_C(1932228929) },
      UINT16_C(10217) },
    { UINT16_C(40225),
      { UINT32_C(2170365150), UINT32_C(1992664920), UINT32_C(2166921116), UINT32_C(1909526389), UINT32_C( 347644304), UINT32_C(3452450941), UINT32_C(1720679449), UINT32_C( 906211416),
        UINT32_C( 766992853), UINT32_C(2762243080), UINT32_C(2301021204), UINT32_C(1828386780), UINT32_C( 377598616), UINT32_C(  65227242), UINT32_C( 510292678), UINT32_C(4166282531) },
      { UINT32_C(3592752078), UINT32_C(2608580999), UINT32_C(1915003286), UINT32_C( 819863192), UINT32_C(3125174224), UINT32_C(1874733481), UINT32_C(3213699228), UINT32_C(1689772437),
        UINT32_C(1933237484), UINT32_C(1007662502), UINT32_C(4004393814), UINT32_C( 572492882), UINT32_C(2514314732), UINT32_C( 721722255), UINT32_C(1491767747), UINT32_C(1589420402) },
      UINT16_C(40225) },
    { UINT16_C(63357),
      { UINT32_C(3769377745), UINT32_C( 219415391), UINT32_C( 278554353), UINT32_C(1702266504), UINT32_C( 520095003), UINT32_C( 447791920), UINT32_C(3636142620), UINT32_C(1372534912),
        UINT32_C(3140648028), UINT32_C(1892238975), UINT32_C( 880894892), UINT32_C(  77264873), UINT32_C( 740530940), UINT32_C(2051461982), UINT32_C(1968308725), UINT32_C(2512790073) },
      { UINT32_C( 491911326), UINT32_C(3935115838), UINT32_C(1713311357), UINT32_C(  23771397), UINT32_C(2972552531), UINT32_C(1428910944), UINT32_C(2932506228), UINT32_C(1044615328),
        UINT32_C(3344667785), UINT32_C( 733145262), UINT32_C(4237414903), UINT32_C(3724475274), UINT32_C(3901696904), UINT32_C( 322812575), UINT32_C(3636528952), UINT32_C( 555091352) },
      UINT16_C(63357) },
    { UINT16_C(29081),
      { UINT32_C(2589608168), UINT32_C(  90919283), UINT32_C(1258353997), UINT32_C(1652001235), UINT32_C(2937853553), UINT32_C(3954595113), UINT32_C(1156599341), UINT32_C(1488292207),
        UINT32_C(1173491665), UINT32_C(2924109408), UINT32_C( 653871699), UINT32_C(1166569683), UINT32_C(2968823174), UINT32_C( 664513274), UINT32_C(1751878649), UINT32_C(3871351060) },
      { UINT32_C(2435560240), UINT32_C(1681880337), UINT32_C(2475374783), UINT32_C( 802689961), UINT32_C(3001011384), UINT32_C(1876589174), UINT32_C( 450381061), UINT32_C(2533398630),
        UINT32_C(1546136395), UINT32_C(1606510496), UINT32_C(1223838879), UINT32_C( 410569311), UINT32_C( 231364502), UINT32_C(3615270098), UINT32_C(1357993194), UINT32_C( 954724845) },
      UINT16_C(29081) },
    { UINT16_C( 3868),
      { UINT32_C(2450622101), UINT32_C( 245503516), UINT32_C(3604545886), UINT32_C(3828248345), UINT32_C(4186439804), UINT32_C(3377296087), UINT32_C(2864331459), UINT32_C( 146462579),
        UINT32_C(2942185619), UINT32_C(2780692550), UINT32_C( 461084417), UINT32_C(2214570246), UINT32_C(2172422057), UINT32_C(3175795194), UINT32_C(2003239940), UINT32_C(1870602715) },
      { UINT32_C(2450622101), UINT32_C(3728596189), UINT32_C(2063186291), UINT32_C( 217970786), UINT32_C(2056092032), UINT32_C(1178064706), UINT32_C(3082657499), UINT32_C( 304495808),
        UINT32_C(4138353689), UINT32_C(2497046561), UINT32_C(  51302049), UINT32_C(1192168391), UINT32_C(3351354500), UINT32_C(1326315635), UINT32_C(1443285910), UINT32_C( 560475143) },
      UINT16_C( 3868) },
    { UINT16_C( 4976),
      { UINT32_C(3975909655), UINT32_C( 884710438), UINT32_C(2940240543), UINT32_C(2320221385), UINT32_C(2575482763), UINT32_C( 342104078), UINT32_C(3611323247), UINT32_C(2766844044),
        UINT32_C(1754326338), UINT32_C( 563891073), UINT32_C(2530270413), UINT32_C( 740301729), UINT32_C(3905266394), UINT32_C( 721168827), UINT32_C( 553729173), UINT32_C( 801500397) },
      { UINT32_C(1402427089), UINT32_C(1869886369), UINT32_C(2969912336), UINT32_C( 987571807), UINT32_C(2200085448), UINT32_C(1638735820), UINT32_C(1216589659), UINT32_C(1819756699),
        UINT32_C(1086262942), UINT32_C(1387213634), UINT32_C(3607344247), UINT32_C(2735857882), UINT32_C(1344680835), UINT32_C(2930889810), UINT32_C( 519451779), UINT32_C( 462057085) },
      UINT16_C( 4976) },
    { UINT16_C(19068),
      { UINT32_C( 176012891), UINT32_C( 331347216), UINT32_C(3723794892), UINT32_C(1662023484), UINT32_C(2033673159), UINT32_C( 128825873), UINT32_C(1651845848), UINT32_C(2712465990),
        UINT32_C(3249285808), UINT32_C(3956566815), UINT32_C(1103677444), UINT32_C( 128243776), UINT32_C(1283513147), UINT32_C(1834233493), UINT32_C(2664417624), UINT32_C(1816165563) },
      { UINT32_C(3308121254), UINT32_C(1538261335), UINT32_C( 178026698), UINT32_C(2349940816), UINT32_C(2966983195), UINT32_C( 404565184), UINT32_C(2981621238), UINT32_C( 253622121),
        UINT32_C( 986991331), UINT32_C( 378897484), UINT32_C(1277178620), UINT32_C(2396533106), UINT32_C(2201924035), UINT32_C(3550239965), UINT32_C(2995082057), UINT32_C( 767664714) },
      UINT16_C(19068) },
    { UINT16_C(38381),
      { UINT32_C(4229511527), UINT32_C(1865291087), UINT32_C( 983605601), UINT32_C(1844143151), UINT32_C(2211039463), UINT32_C( 567677596), UINT32_C(2244157636), UINT32_C(3021647949),
        UINT32_C( 951137257), UINT32_C(2863128392), UINT32_C(2950973568), UINT32_C(2468204460), UINT32_C( 857204375), UINT32_C(3176459769), UINT32_C(1531123726), UINT32_C(2987417032) },
      { UINT32_C(3656040849), UINT32_C( 562270880), UINT32_C(2261805018), UINT32_C(3457805622), UINT32_C(3439407316), UINT32_C( 747263518), UINT32_C( 914869614), UINT32_C(3169359659),
        UINT32_C(4187345752), UINT32_C(1058675045), UINT32_C(3083201152), UINT32_C(2894454488), UINT32_C( 762938895), UINT32_C(1247347676), UINT32_C(4236370129), UINT32_C(3501746552) },
      UINT16_C(38381) }
 };

  for(size_t i = 0; i < sizeof(test_vec) / sizeof(test_vec[0]); i++) {
    easysimd__mmask16 k1 = test_vec[i].k1;
    easysimd__m512i a = easysimd_mm512_loadu_si512((void *)test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512((void *)test_vec[i].b);
    easysimd__mmask16 r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epu32_mask(k1, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epu32_mask");

    easysimd_assert_equal_mmask16(r, test_vec[i].k);
  }
  return 0;

#else
  fputc('\n', stdout);
  int count = 0;
  for ( ;count < 7 ; ) {
    easysimd__mmask16 k1 = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_u32x16();
    easysimd__m512i b = easysimd_test_x86_random_u32x16();
    easysimd__mmask16 r = easysimd_mm512_mask_cmpneq_epu32_mask(k1, a, b);

      easysimd_test_x86_write_mmask16(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_x86_write_u32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_u32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_mmask16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
      count++;
  }
  return 1;

#endif
} 

static int
test_easysimd_mm512_mask_cmpneq_epu64_mask (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k1;
    const uint64_t a[8];
    const uint64_t b[8];
    const easysimd__mmask8 r;
  } test_vec[] = {
    { UINT8_C(192),
      { UINT64_C( 8500010159046362605), UINT64_C( 4551979138695102711), UINT64_C(17584220308874100908), UINT64_C( 9157438251551223488),
        UINT64_C( 5904460293154881606), UINT64_C( 7350789088184949211), UINT64_C( 2155279588967623360), UINT64_C( 4113906264930660613) },
      { UINT64_C(12930578724024548551), UINT64_C( 9700325639172097662), UINT64_C(14672555810928373284), UINT64_C( 9157438251551223488),
        UINT64_C(10953301963541870199), UINT64_C( 1712897224311076985), UINT64_C( 4083982713627426588), UINT64_C(   90214426040340553) },
      UINT8_C(192) },
    { UINT8_C(192),
      { UINT64_C( 8919990022475747920), UINT64_C( 7885528980955708778), UINT64_C(  767281315642151656), UINT64_C( 9353839197809225679),
        UINT64_C( 1781908870492488341), UINT64_C(10564238192964772953), UINT64_C(18382792281559613896), UINT64_C(13883997503257225467) },
      { UINT64_C( 8513415300270489696), UINT64_C( 7885528980955708778), UINT64_C(  767281315642151656), UINT64_C( 9353839197809225679),
        UINT64_C(14228329621942283746), UINT64_C(10564238192964772953), UINT64_C( 9745289061186775336), UINT64_C( 4074670613283075061) },
      UINT8_C(192) },
    { UINT8_C(149),
      { UINT64_C(14683247707536242743), UINT64_C( 9183785180634225950), UINT64_C( 3765763275622922001), UINT64_C(  459413134421563198),
        UINT64_C(11685565064272815201), UINT64_C(14558975108894645909), UINT64_C(12468597579226661398), UINT64_C(17062406753749590515) },
      { UINT64_C(14683247707536242743), UINT64_C( 9183785180634225950), UINT64_C(13906659134299075297), UINT64_C(  459413134421563198),
        UINT64_C(11685565064272815201), UINT64_C( 6153523979506964088), UINT64_C(15975855693142342439), UINT64_C( 5454529320010201114) },
      UINT8_C(132) },
    { UINT8_C(123),
      { UINT64_C(  693236228487044614), UINT64_C( 1072671896773727170), UINT64_C(  344469139897908954), UINT64_C(11268650130093722420),
        UINT64_C( 7628679575683581992), UINT64_C( 6633083733145591190), UINT64_C(13929548488119140464), UINT64_C( 2751710361343735037) },
      { UINT64_C(  693236228487044614), UINT64_C( 6148887820375061110), UINT64_C(17386692765200433660), UINT64_C(11268650130093722420),
        UINT64_C( 1410514665242739659), UINT64_C(18073898574743913113), UINT64_C(13929548488119140464), UINT64_C( 2751710361343735037) },
      UINT8_C( 50) },
    { UINT8_C(201),
      { UINT64_C(17529851326558753777), UINT64_C(13929625121331692552), UINT64_C( 7287750785118352015), UINT64_C( 7155954217949652225),
        UINT64_C(15257121086130045154), UINT64_C(11057178257520464498), UINT64_C( 5390759591261831088), UINT64_C(10992477132744075192) },
      { UINT64_C( 1584173751243732063), UINT64_C( 6400521790615366447), UINT64_C( 7287750785118352015), UINT64_C( 1767880256132782650),
        UINT64_C( 3959013746621475130), UINT64_C(11057178257520464498), UINT64_C( 8803535891791785415), UINT64_C( 7268231639160480779) },
      UINT8_C(201) },
    { UINT8_C( 31),
      { UINT64_C( 9793437029648752960), UINT64_C(11993528909880351193), UINT64_C(10754633481915445679), UINT64_C( 1073091251437791975),
        UINT64_C( 7197643403192975940), UINT64_C(17269221271345278467), UINT64_C(14486685693650911050), UINT64_C( 4130209830513054581) },
      { UINT64_C( 9793437029648752960), UINT64_C(11993528909880351193), UINT64_C(10754633481915445679), UINT64_C( 1073091251437791975),
        UINT64_C( 2612772773240211143), UINT64_C(17269221271345278467), UINT64_C(14486685693650911050), UINT64_C(12411088129737928923) },
      UINT8_C( 16) },
    { UINT8_C(  9),
      { UINT64_C( 4500416212399263535), UINT64_C( 1947277396459577933), UINT64_C( 2145804711319807050), UINT64_C( 9756687232892644383),
        UINT64_C(15796495054353784134), UINT64_C(13827027723915253326), UINT64_C(  233797833815651868), UINT64_C( 7145178635092125001) },
      { UINT64_C( 4500416212399263535), UINT64_C( 1947277396459577933), UINT64_C(15772577390333916411), UINT64_C(12562970428013495256),
        UINT64_C( 3831416811449005507), UINT64_C( 5654747977891499372), UINT64_C(18019765372725407399), UINT64_C(17729624114717495124) },
      UINT8_C(  8) },
    { UINT8_C(236),
      { UINT64_C( 1170450449167894086), UINT64_C( 1068033572545466794), UINT64_C(16016616593901483171), UINT64_C(13027544206668036306),
        UINT64_C(10805372365981459560), UINT64_C(14259392723109328885), UINT64_C( 5209471207882883710), UINT64_C(18112898888559390778) },
      { UINT64_C(15858814654108107670), UINT64_C( 1068033572545466794), UINT64_C(16813784236560805016), UINT64_C(13027544206668036306),
        UINT64_C(14498945647241358450), UINT64_C( 1555964524542355560), UINT64_C( 7667793338898308049), UINT64_C( 6730781282914860022) },
      UINT8_C(228) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k1;
    easysimd__mmask8 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_cmpneq_epu64_mask(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_cmpneq_epu64_mask");

   easysimd_assert_equal_mmask8(test_vec[i].r, r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k1 = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_u64x8();
    easysimd__m512i b = easysimd_mm512_mask_blend_epi64(easysimd_test_x86_random_mmask8(), a, easysimd_test_x86_random_u64x8());
    easysimd__mmask8 r = easysimd_mm512_mask_cmpneq_epu64_mask(k1, a, b);

    easysimd_test_x86_write_mmask8(2, k1, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpneq_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_cmpneq_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cmpneq_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_cmpneq_epu64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epi64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epi8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epi16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epi32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epi64_mask)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_cmpneq_epu64_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epu8_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epu16_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epu32_mask)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_cmpneq_epu64_mask)

EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
